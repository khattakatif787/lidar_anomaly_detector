from __future__ import annotations
from pathlib import Path
import pandas as pd


def counts_tp_tn_fp_fn(df: pd.DataFrame) -> dict:
    # assumes label in {0,1}, pred in {0,1}
    tp = int(((df["label"] == 1) & (df["pred"] == 1)).sum())
    tn = int(((df["label"] == 0) & (df["pred"] == 0)).sum())
    fp = int(((df["label"] == 0) & (df["pred"] == 1)).sum())
    fn = int(((df["label"] == 1) & (df["pred"] == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_report(pred_csv: Path, out_csv: Path, model_name: str | None = None) -> Path:
    df = pd.read_csv(pred_csv)

    required = {"label", "pred", "corruption", "scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {pred_csv}: {missing}")

    rows = []

    # Overall
    row = {"scope": "overall", "group": "all"}
    row.update(counts_tp_tn_fp_fn(df))
    if model_name:
        row["model"] = model_name
    rows.append(row)

    # By corruption
    for corr, g in df.groupby("corruption"):
        row = {"scope": "corruption", "group": str(corr)}
        row.update(counts_tp_tn_fp_fn(g))
        if model_name:
            row["model"] = model_name
        rows.append(row)

    # By scenario
    for sc, g in df.groupby("scenario"):
        row = {"scope": "scenario", "group": str(sc)}
        row.update(counts_tp_tn_fp_fn(g))
        if model_name:
            row["model"] = model_name
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Create TP/TN/FP/FN report overall + by corruption + by scenario")
    ap.add_argument("--pred", required=True, help="test_predictions.csv")
    ap.add_argument("--out", required=True, help="output report csv")
    ap.add_argument("--model", default="", help="optional model name")
    args = ap.parse_args()

    build_report(Path(args.pred), Path(args.out), model_name=(args.model or None))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
