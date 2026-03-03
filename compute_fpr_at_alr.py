from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve


def fpr_at_fnr(probas: np.ndarray, y_true: np.ndarray, fnr_t: float) -> float:
    y_score = probas[:, 1]  # anomaly probability

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    fnr = 1.0 - tpr

    for i in range(len(fpr)):
        if fnr[i] <= fnr_t:
            return float(fpr[i])

    return 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    X_test = np.load(run_dir / "feat_cache" / "X_test.npy")
    y_test = np.load(run_dir / "feat_cache" / "y_test.npy").astype(int)

    fnr_targets = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # match your Excel columns

    rows = []
    for mdir in sorted([d for d in (run_dir / "models").iterdir() if d.is_dir()]):
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

    # order rows as you want
    order = ["LDA", "GNB", "DT", "RF", "EXTRA_TREES", "XGB"]
    df["__o"] = df["ML Algorithm"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__o").drop(columns="__o")

    out_csv = run_dir / "models" / "summary_fpr_at_alr.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
