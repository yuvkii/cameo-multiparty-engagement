"""Batch segment-aggregation driver.

Turns raw per-frame frame_features.csv outputs (facial_keypoints and/or
gaze_target) into one segments.parquet per dataset-date, per the design in
datasetPrep/segment_aggregation.py.

Usage:
    python run_batch_segment_aggregation.py --dataset 03_20
    python run_batch_segment_aggregation.py --dataset 03_20 --segment-length-sec 15
    python run_batch_segment_aggregation.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasetPrep.segment_aggregation import build_session_table  # noqa: E402

OUTPUT_ROOT = Path(__file__).parent / "outputs"

# Datasets known to only use the old flat outputs/facial_keypoints_<date>/
# layout (no gaze_target -- no mocap footage for these dates).
_OLD_FLAT_LAYOUT_DATASETS = {"03_24_01", "03_24_02"}

_SESSION_DIR_RE = re.compile(r"^session_(\d+)$")


def _discover_sessions(dataset: str, pipeline: str) -> dict[int, Path]:
    """Return {session_num: path_to_frame_features.csv} for a given pipeline."""
    if pipeline == "facial_keypoints" and dataset in _OLD_FLAT_LAYOUT_DATASETS:
        root = OUTPUT_ROOT / f"facial_keypoints_{dataset}"
    else:
        root = OUTPUT_ROOT / dataset / pipeline

    sessions: dict[int, Path] = {}
    if not root.is_dir():
        return sessions
    for d in root.iterdir():
        m = _SESSION_DIR_RE.match(d.name)  # excludes oddities like session_1_fixed
        if not m:
            continue
        csv_path = d / "frame_features.csv"
        if csv_path.exists():
            sessions[int(m.group(1))] = csv_path
    return sessions


def process_dataset(dataset: str, segment_length_sec: float) -> None:
    fk_sessions = _discover_sessions(dataset, "facial_keypoints")
    gt_sessions = _discover_sessions(dataset, "gaze_target")
    all_session_nums = sorted(set(fk_sessions) | set(gt_sessions))

    if not all_session_nums:
        print(f"[{dataset}] No facial_keypoints or gaze_target output found — skipping.")
        return

    print(f"[{dataset}] facial_keypoints sessions: {sorted(fk_sessions)}")
    print(f"[{dataset}] gaze_target sessions:      {sorted(gt_sessions)}")

    tables = []
    for session in all_session_nums:
        table = build_session_table(
            dataset=dataset,
            session=session,
            fk_csv=fk_sessions.get(session),
            gt_csv=gt_sessions.get(session),
            segment_length_sec=segment_length_sec,
        )
        if not table.empty:
            tables.append(table)

    if not tables:
        print(f"[{dataset}] All sessions produced empty tables — skipping write.")
        return

    combined = pd.concat(tables, ignore_index=True)
    output_path = OUTPUT_ROOT / dataset / "segments.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    n_segments = combined.groupby(["session", "segment_idx"]).ngroups
    print(f"[{dataset}] {len(combined)} rows, {n_segments} unique segments, {len(all_session_nums)} sessions")
    if "fk_detection_rate" in combined.columns:
        print(f"[{dataset}]   fk_detection_rate: mean={combined['fk_detection_rate'].mean():.1%} "
              f"nan_rate={combined['fk_detection_rate'].isna().mean():.1%}")
    if "gt_robot_detection_rate" in combined.columns:
        print(f"[{dataset}]   gt_robot_detection_rate: mean={combined['gt_robot_detection_rate'].mean():.1%} "
              f"nan_rate={combined['gt_robot_detection_rate'].isna().mean():.1%}")
    if combined["cross_camera_duration_diff_sec"].notna().any():
        diffs = combined.groupby("session")["cross_camera_duration_diff_sec"].first().dropna()
        print(f"[{dataset}]   cross_camera_duration_diff_sec by session: {diffs.to_dict()}")
    print(f"[{dataset}] Output -> {output_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset-date to process, e.g. 03_20")
    parser.add_argument("--all", action="store_true", help="Process every dataset-date found under outputs/")
    parser.add_argument("--segment-length-sec", type=float, default=10.0, help="Segment window length (default: 10s)")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        sys.exit("Must pass either --dataset <name> or --all")

    if args.all:
        datasets = set()
        for p in OUTPUT_ROOT.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("facial_keypoints_"):
                datasets.add(p.name.removeprefix("facial_keypoints_"))
            elif (p / "facial_keypoints").is_dir() or (p / "gaze_target").is_dir():
                datasets.add(p.name)
        datasets = sorted(datasets)
    else:
        datasets = [args.dataset]

    print(f"Segment length: {args.segment_length_sec}s")
    print(f"Datasets: {datasets}")
    print()

    for dataset in datasets:
        process_dataset(dataset, args.segment_length_sec)

    print("Batch complete.")


if __name__ == "__main__":
    main()
