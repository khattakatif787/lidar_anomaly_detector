from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# our modules
from lidar_anom.data.scan import build_manifest
from lidar_anom.data.split import group_split_manifest
from lidar_anom.features_basic import extract_features_from_pcd

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve  # ✅ added roc_curve

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier

import joblib


def list_all_corruptions(data_root: Path) -> list[str]:
    corr_root = data_root / "corrupted_dataset"
    if not corr_root.exists():
        raise FileNotFoundError(f"Not found: {corr_root}")
    return sorted([d.name for d in corr_root.iterdir() if d.is_dir()])


def cache_features(manifest_split_csv: Path, outdir: Path, feature_set: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_split_csv)

    if "split" not in df.columns:
        raise ValueError("manifest_split.csv must contain a 'split' column")

    def build(tag: str):
        part = df[df["split"] == tag].reset_index(drop=True)
        X_list = []
        for p in tqdm(part["path"].tolist(), desc=f"Extracting features ({tag})"):
            X_list.append(extract_features_from_pcd(p, feature_set))
        X = np.vstack(X_list).astype(np.float32)
        y = part["label"].to_numpy(dtype=int)
        np.save(outdir / f"X_{tag}.npy", X)
        np.save(outdir / f"y_{tag}.npy", y)
        part.to_csv(outdir / f"{tag}_meta.csv", index=False)

    build("train")
    build("test")
    return outdir


def train_10_models(cache_dir: Path, out_models_dir: Path, seed: int = 42) -> Path:
    out_models_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(cache_dir / "X_train.npy")
    y_train = np.load(cache_dir / "y_train.npy")
    X_test = np.load(cache_dir / "X_test.npy")
    y_test = np.load(cache_dir / "y_test.npy")
    test_meta = pd.read_csv(cache_dir / "test_meta.csv")

    models = {
        "lda": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis())
        ]),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed
        ),
    }

    overall_rows = []

    for name, model in models.items():
        print(f"\n=== Training model: {name} ===")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        overall_rows.append({"model": name, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})

        mdir = out_models_dir / name
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, mdir / "model.joblib")

        out = test_meta.copy()
        out["pred"] = pred
        out.to_csv(mdir / "test_predictions.csv", index=False)

    pd.DataFrame(overall_rows).to_csv(out_models_dir / "summary_overall.csv", index=False)
    print(f"\nWrote: {out_models_dir / 'summary_overall.csv'}")
    return out_models_dir


# NEW: compute ALR table automatically
def compute_fpr_at_alr(models_dir: Path, cache_dir: Path) -> Path:
    """
    Writes models/summary_fpr_at_alr.csv
    Columns = FPR@10-5 ... FPR@10-1 (target FNR = 1e-5 ... 1e-1)
    """
    X_test = np.load(cache_dir / "X_test.npy")
    y_test = np.load(cache_dir / "y_test.npy").astype(int)

    fnr_targets = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    def fpr_at_fnr(probas: np.ndarray, y_true: np.ndarray, fnr_t: float) -> float:
        y_score = probas[:, 1]  # anomaly probability (label=1)
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
        fnr = 1.0 - tpr
        for i in range(len(fpr)):
            if fnr[i] <= fnr_t:
                return float(fpr[i])
        return 1.0

    rows = []
    for mdir in sorted([d for d in models_dir.iterdir() if d.is_dir()]):
        model_path = mdir / "model.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        probas = model.predict_proba(X_test)

        row = {"ML Algorithm": mdir.name.upper()}
        for t in fnr_targets:
            row[f"FPR@10-{int(-np.log10(t))}"] = fpr_at_fnr(probas, y_test, t)
        rows.append(row)

    df = pd.DataFrame(rows)

    order = ["LDA", "GNB", "DT", "RF", "EXTRA_TREES", "XGB"]
    df["__o"] = df["ML Algorithm"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__o").drop(columns="__o")

    out_csv = models_dir / "summary_fpr_at_alr.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Run full LiDAR corruption anomaly detection pipeline.")
    ap.add_argument("--data-root", required=True,
                    help="Path to LIHRA_DATASET (contains clean_dataset/ and corrupted_dataset/).")
    ap.add_argument("--corruptions", nargs="*", default=None,
                    help="List of corruption folder names to include. Example: --corruptions c1 c2 c3")
    ap.add_argument("--all-corruptions", action="store_true",
                    help="Use ALL corruption folders under corrupted_dataset/")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--feature-set", default="corruption",
                    choices=["corruption", "rich", "corruption_plus", "corruption_plus_v2"],
                    help="Feature set: corruption (more object-agnostic) or rich (more features).")
    ap.add_argument("--corruption-sample-frac", type=float, default=0.1,
                    help="Sample fraction of each selected corruption (equal per scenario). Clean is always full.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="", help="Optional run name. If empty, uses timestamp.")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    if args.all_corruptions:
        selected = list_all_corruptions(data_root)
        print(f"Selected ALL corruptions: {len(selected)} folders")
    else:
        if not args.corruptions:
            raise SystemExit("You must provide --corruptions ... or use --all-corruptions")
        selected = args.corruptions
        print(f"Selected corruptions: {selected}")

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) manifest
    manifest_csv = run_dir / "manifest.csv"
    build_manifest(
        data_root=data_root,
        selected_corruptions=selected,
        out_csv=manifest_csv,
        corruption_sample_frac=args.corruption_sample_frac,
        seed=args.seed
    )
    print(f"Wrote: {manifest_csv}")

    # 2) split
    split_csv = run_dir / "manifest_split.csv"
    group_split_manifest(
        manifest_csv=manifest_csv,
        out_csv=split_csv,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    print(f"Wrote: {split_csv}")

    # 3) cache features
    cache_dir = run_dir / "feat_cache"
    cache_features(split_csv, cache_dir, args.feature_set)
    print(f"Cached features in: {cache_dir}")

    # 4) train models
    models_dir = run_dir / "models"
    train_10_models(cache_dir, models_dir, seed=args.seed)

    # 5) automatically compute ALR table
    compute_fpr_at_alr(models_dir, cache_dir)

    # 6) final combined report (still disabled)
    # final_csv = run_dir / "FINAL_report_all_models.csv"
    # final_report_all_models(models_dir, final_csv)

    print("\nDONE")
    print("Run folder:", run_dir)
    print("Overall summary:", models_dir / "summary_overall.csv")
    print("ALR table:", models_dir / "summary_fpr_at_alr.csv")


if __name__ == "__main__":
    main()