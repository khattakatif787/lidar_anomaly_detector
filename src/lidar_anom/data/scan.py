from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional
import csv
import random
import math
import zlib


@dataclass(frozen=True)
class FrameRecord:
    path: str
    label: int
    scenario: str
    corruption: str
    frame_id: str
    group_id: str


def _iter_pcd_files(folder: Path):
    return sorted(folder.rglob("*.pcd"))


def _frame_id_from_path(p: Path) -> str:
    return p.stem


def scan_clean(clean_root: Path) -> List[FrameRecord]:
    records: List[FrameRecord] = []

    if not clean_root.exists():
        raise FileNotFoundError(f"Clean root not found: {clean_root}")

    for scenario_dir in sorted([d for d in clean_root.iterdir() if d.is_dir()]):
        scenario = scenario_dir.name
        for pcd_path in _iter_pcd_files(scenario_dir):
            frame_id = _frame_id_from_path(pcd_path)
            group_id = f"{scenario}_{frame_id}"

            records.append(
                FrameRecord(
                    path=str(pcd_path),
                    label=0,
                    scenario=scenario,
                    corruption="clean",
                    frame_id=frame_id,
                    group_id=group_id,
                )
            )
    return records


def scan_corrupted(
    corrupted_root: Path,
    selected_corruptions: Iterable[str],
    sample_frac: Optional[float] = None,
    seed: int = 42,
) -> List[FrameRecord]:
    """
    Reads corrupted frames.

    Folder structure supported:
      A) corrupted_root/<corr>/<scenario>/*.pcd
      B) corrupted_root/<corr>/<corr>/<scenario>/*.pcd

    If sample_frac is set (e.g. 0.1):
      For each corruption:
        - take equal samples from each scenario
        - k = floor(sample_frac * min_count_across_scenarios), minimum 1
    """
    records: List[FrameRecord] = []

    if not corrupted_root.exists():
        raise FileNotFoundError(f"Corrupted root not found: {corrupted_root}")

    if sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError(f"sample_frac must be in (0,1], got {sample_frac}")

    for corr in sorted(selected_corruptions):
        corr_dir = corrupted_root / corr
        if not corr_dir.exists():
            raise FileNotFoundError(f"Selected corruption folder not found: {corr_dir}")

        # Handle A vs B
        nested = corr_dir / corr
        base_dir = nested if (nested.exists() and nested.is_dir()) else corr_dir

        scenario_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not scenario_dirs:
            continue

        # Gather all pcd paths per scenario
        per_scenario_paths = {}
        for scenario_dir in scenario_dirs:
            scenario = scenario_dir.name
            paths = _iter_pcd_files(scenario_dir)
            if paths:
                per_scenario_paths[scenario] = paths

        if not per_scenario_paths:
            continue

        corr_key = zlib.crc32(corr.encode("utf-8")) & 0xFFFFFFFF
        rng = random.Random(seed ^ corr_key)

        if sample_frac is None:
            # take all
            for scenario, paths in per_scenario_paths.items():
                for pcd_path in paths:
                    frame_id = _frame_id_from_path(pcd_path)
                    group_id = f"{scenario}_{frame_id}"
                    records.append(
                        FrameRecord(
                            path=str(pcd_path),
                            label=1,
                            scenario=scenario,
                            corruption=corr,
                            frame_id=frame_id,
                            group_id=group_id,
                        )
                    )
        else:
            # equal-per-scenario sampling
            min_count = min(len(paths) for paths in per_scenario_paths.values())
            k = int(math.floor(sample_frac * min_count))
            k = max(1, k)

            for scenario, paths in per_scenario_paths.items():
                # sample k from each scenario
                chosen = rng.sample(paths, k) if len(paths) >= k else paths
                for pcd_path in chosen:
                    frame_id = _frame_id_from_path(pcd_path)
                    group_id = f"{scenario}_{frame_id}"
                    records.append(
                        FrameRecord(
                            path=str(pcd_path),
                            label=1,
                            scenario=scenario,
                            corruption=corr,
                            frame_id=frame_id,
                            group_id=group_id,
                        )
                    )

    return records


def build_manifest(
    data_root: Path,
    selected_corruptions: Iterable[str],
    out_csv: Path,
    corruption_sample_frac: Optional[float] = None,
    seed: int = 42,
) -> Path:
    clean_root = data_root / "clean_dataset"
    corrupted_root = data_root / "corrupted_dataset"

    records: List[FrameRecord] = []
    records.extend(scan_clean(clean_root))
    records.extend(
        scan_corrupted(
            corrupted_root,
            selected_corruptions,
            sample_frac=corruption_sample_frac,
            seed=seed,
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "scenario", "corruption", "frame_id", "group_id"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    return out_csv


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--corruptions", nargs="+", required=True)
    parser.add_argument("--corruption-sample-frac", type=float, default=None,
                        help="e.g. 0.1 means sample 10%% from each corruption, equally per scenario")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = build_manifest(
        data_root=Path(args.data_root),
        selected_corruptions=args.corruptions,
        out_csv=Path(args.out),
        corruption_sample_frac=args.corruption_sample_frac,
        seed=args.seed,
    )

    print("Wrote manifest:", out)


if __name__ == "__main__":
    main()
