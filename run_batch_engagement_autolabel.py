"""Batch engagement auto-labeling driver.

Reads outputs/<dataset>/segments.parquet, runs the heuristic in
datasetPrep/engagement_heuristic.py, writes
outputs/<dataset>/engagement_labels_auto.parquet plus a small
engagement_thresholds.json (parquet doesn't reliably round-trip
DataFrame.attrs, so thresholds are saved separately for reproducibility).

Usage:
    python run_batch_engagement_autolabel.py --dataset 03_20
    python run_batch_engagement_autolabel.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasetPrep.engagement_heuristic import ENGAGEMENT_LEVELS, REVIEW_FRACTION, label_dataframe  # noqa: E402

OUTPUT_ROOT = Path(__file__).parent / "outputs"


def process_dataset(dataset: str, review_fraction: float) -> None:
    segments_path = OUTPUT_ROOT / dataset / "segments.parquet"
    if not segments_path.exists():
        print(f"[{dataset}] No segments.parquet found — skipping.")
        return

    df = pd.read_parquet(segments_path)
    if "gt_looking_at_robot_rate" not in df.columns and "fk_gaze_target_camera_frac" not in df.columns:
        print(f"[{dataset}] Neither gaze signal present — skipping.")
        return

    labeled = label_dataframe(df, review_fraction=review_fraction)
    thresholds = labeled.attrs.get("engagement_thresholds", [])

    output_path = OUTPUT_ROOT / dataset / "engagement_labels_auto.parquet"
    labeled.to_parquet(output_path, index=False)

    thresholds_path = OUTPUT_ROOT / dataset / "engagement_thresholds.json"
    thresholds_path.write_text(json.dumps({"levels": list(ENGAGEMENT_LEVELS), "thresholds": thresholds}, indent=2))

    n_total = len(labeled)
    n_review = int(labeled["needs_review"].sum())
    level_counts = labeled["engagement_level"].value_counts().reindex(ENGAGEMENT_LEVELS, fill_value=0)

    print(f"[{dataset}] {n_total} rows labeled")
    print(f"[{dataset}]   thresholds (quartiles of proxy score): {[round(t, 4) for t in thresholds]}")
    for level in ENGAGEMENT_LEVELS:
        pct = level_counts[level] / n_total if n_total else 0.0
        print(f"[{dataset}]   {level:>10}: {level_counts[level]:4d} ({pct:.1%})")
    print(f"[{dataset}]   flagged needs_review: {n_review} ({n_review / n_total:.1%})")
    print(f"[{dataset}] Output -> {output_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset-date to process, e.g. 03_20")
    parser.add_argument("--all", action="store_true", help="Process every dataset with a segments.parquet")
    parser.add_argument("--review-fraction", type=float, default=REVIEW_FRACTION,
                         help=f"Fraction of lowest-confidence segments to flag for review (default: {REVIEW_FRACTION})")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        sys.exit("Must pass either --dataset <name> or --all")

    if args.all:
        datasets = sorted(
            p.parent.name for p in OUTPUT_ROOT.glob("*/segments.parquet")
        )
    else:
        datasets = [args.dataset]

    print(f"Review fraction: {args.review_fraction:.0%}")
    print(f"Datasets: {datasets}")
    print()

    for dataset in datasets:
        process_dataset(dataset, args.review_fraction)

    print("Batch complete.")


if __name__ == "__main__":
    main()
