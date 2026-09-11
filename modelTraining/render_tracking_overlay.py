"""Render a full-session debug video with session_tracker.py's tracked
local_id drawn as a colored box per person, for visual confirmation BEFORE
trusting the tracker's output for any session where the cheap position-rank
identity shortcut is known to be unsafe (people change seats mid-session).

Same "render a video, watch it, decide if it's trustworthy" pattern this
project has used for every other new perception pipeline (gaze_target,
facial_keypoints, mouth_motion, the earlier segment-level reidentify_segment
via render_auto_review_clips.py) -- this is genuinely new, unvalidated
machinery (a full-session tracker, not a 3s-segment one), so it gets the
same treatment before being wired into build_continuous_dataset.py.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/render_tracking_overlay.py --dataset 03_20 --session 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasetPrep"))
from render_review_clips import _find_cam2_raw  # noqa: E402

from session_tracker import track_session, map_tracks_to_letters  # noqa: E402

# BGR, distinct + high-contrast; index 3+ (shouldn't normally occur -- see
# MAX_TRACKS in session_tracker.py) gets a fallback grey so it's still
# visible but visually flagged as unexpected.
TRACK_COLORS = {0: (0, 0, 220), 1: (0, 200, 0), 2: (255, 140, 0)}
FALLBACK_COLOR = (160, 160, 160)


def render(dataset: str, session: int) -> Path:
    gt_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
    df = pd.read_csv(gt_path)
    tracked = track_session(df)

    try:
        letters = map_tracks_to_letters(tracked)
        print(f"local_id -> letter mapping: {letters}")
    except ValueError as e:
        letters = {}
        print(f"WARNING: could not derive a letter mapping ({e}) -- "
              f"overlay will show raw local_id numbers only, check that against the footage instead")

    video_path = _find_cam2_raw(dataset, session)
    if video_path is None:
        raise SystemExit(f"no Cam_2 footage found for {dataset} session {session}")

    by_frame: dict[int, list[tuple[int, tuple]]] = {}
    for row in tracked.dropna(subset=["local_id"]).itertuples():
        by_frame.setdefault(row.frame_index, []).append(
            (int(row.local_id), (row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1))
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = PROJECT_ROOT / "outputs" / dataset / "tracking_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_{session}_tracking_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        for lid, box in by_frame.get(frame_idx, []):
            color = TRACK_COLORS.get(lid, FALLBACK_COLOR)
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            label = letters.get(lid, f"id{lid}")
            cv2.putText(frame, label, (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}/{total_frames} ({frame_idx / fps:.1f}s)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}/{total_frames} ({frame_idx / fps:.1f}s)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"wrote {out_path} ({frame_idx} frames)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--session", required=True, type=int)
    args = parser.parse_args()
    render(args.dataset, args.session)
