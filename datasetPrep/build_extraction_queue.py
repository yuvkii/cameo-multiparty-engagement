"""Write a manual_queue.csv that tiles each session's FULL Cam_2 timeline.

run_batch_openface_cam2.py and run_batch_mouth_motion.py both use
manual_queue.csv purely to decide which frames to extract features for -- a
leftover from the old 3s-segment manual-labeling workflow. Dates labeled with
the continuous slider (continuous_labeler.py) never build that queue, so the
extractors have nothing to read and the date can't be featurised.

Why not reuse render_manual_clips.build_manual_queue: it derives segments
from segments.parquet, which comes from the RGB facial_keypoints pipeline and
does NOT necessarily span the whole Cam_2 recording. Measured on 05_14 it
covered only 18% of session 5 and 29% of session 6, which would have silently
dropped most of those sessions from feature extraction. It also renders a
video clip per segment, which is pure waste when no human will watch them.

This instead tiles the real Cam_2 stream end to end (multi-part aware, so
split recordings are covered in full), emitting the columns the extractors
actually read: session, source, segment_start_sec, segment_end_sec.

Usage:
    dataExtraction/openface3/.venv/bin/python3 \
        datasetPrep/build_extraction_queue.py --dataset 05_14 --sessions 1 2 3 4 5 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from render_review_clips import open_cam2_capture

PROJECT_ROOT = Path(__file__).parent.parent
SEGMENT_SEC = 3.0


def build(dataset: str, sessions: list[int]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        cap = open_cam2_capture(dataset, session)
        if cap is None:
            print(f"  session {session}: no Cam_2 footage, skipping")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total / fps
        n_seg = int(duration // SEGMENT_SEC)
        if duration % SEGMENT_SEC:
            n_seg += 1  # keep the ragged tail rather than truncating it
        for i in range(n_seg):
            start = i * SEGMENT_SEC
            rows.append({
                "dataset": dataset,
                "session": session,
                "segment_idx": i,
                "segment_start_sec": start,
                "segment_end_sec": min(start + SEGMENT_SEC, duration),
                "source": "cam2",
                "clip_path": "",          # no clips rendered; nothing reads this here
                "labeled": 1,             # extractors filter on source only, but keep the column shape
                "notes": "auto-tiled for feature extraction (continuous-labeled date)",
            })
        print(f"  session {session}: {total} frames @ {fps:.0f}fps = {duration:.1f}s -> {n_seg} segments "
              f"({len(cap.paths) if hasattr(cap, 'paths') else 1} video part(s))")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sessions", type=int, nargs="+", required=True)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / "outputs" / args.dataset / "manual_queue.csv"
    if out_path.exists():
        # Never silently clobber a real human-labeled queue -- those carry
        # `labeled` flags and notes that only exist in this file.
        raise SystemExit(f"{out_path} already exists; refusing to overwrite. Move it aside first if intended.")

    print(f"building extraction queue for {args.dataset}:")
    df = build(args.dataset, args.sessions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} segments -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
