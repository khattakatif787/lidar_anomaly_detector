from __future__ import annotations

from pathlib import Path
import pandas as pd


def group_split_manifest(
    manifest_csv: Path,
    out_csv: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
    group_col: str = "group_id",
) -> Path:
    """
    Split rows into train/test by grouping on group_id (prevents leakage).

    Adds a new column: split ∈ {"train","test"}.
    """
    df = pd.read_csv(manifest_csv)

    if group_col not in df.columns:
        raise ValueError(f"Missing '{group_col}' column in manifest: {manifest_csv}")

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) == 0:
        raise ValueError("No groups found to split.")

    # Shuffle groups deterministically
    rng = pd.Series(groups).sample(frac=1.0, random_state=seed).tolist()

    n_train = int(round(train_ratio * len(rng)))
    train_groups = set(rng[:n_train])

    df["split"] = df[group_col].apply(lambda g: "train" if g in train_groups else "test")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return out_csv


def main():
    import argparse

    p = argparse.ArgumentParser(description="Group-based 70/30 split for manifest.csv (no leakage).")
    p.add_argument("--manifest", required=True, help="Input manifest.csv")
    p.add_argument("--out", required=True, help="Output manifest_split.csv")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default 0.7)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = p.parse_args()

    out = group_split_manifest(
        manifest_csv=Path(args.manifest),
        out_csv=Path(args.out),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Wrote split manifest: {out}")


if __name__ == "__main__":
    main()
