"""Render one raw clip per (session, segment) for fully-manual engagement labeling.

Unlike render_review_clips.py, this does NOT split by track_id -- track_id
has been shown to drift mid-segment (facial_keypoints tracker occasionally
reassigns which physical person a track number refers to), and relying on
it to route review or disambiguate identity was the root problem with the
confidence-based review queue. Manual labeling sidesteps that entirely:
one clip per segment, showing everyone in frame together; the reviewer
identifies people by on-screen left-to-right position at labeling time
(see manual_labeler.py), not by track_id. Reconciling a labeled position
back to a track_id/bbox is deferred to a later, code-driven join step.

Prefers the wide mocap Cam_2 view over participants_rgb.avi when both
exist -- user confirmed (2026-07-20, after labeling sessions 1-2 on RGB)
Cam_2 is clearer for this manual-labeling task. Falls back to RGB only
for sessions/dates with no Cam_2 coverage.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from render_review_clips import _find_cam2_raw, _find_participants_rgb, render_clip

PROJECT_ROOT = Path(__file__).parent.parent


def build_manual_queue(dataset: str) -> pd.DataFrame:
    segments_path = PROJECT_ROOT / "outputs" / dataset / "segments.parquet"
    df = pd.read_parquet(segments_path)
    segments = (
        df[["session", "segment_idx", "segment_start_sec", "segment_end_sec"]]
        .drop_duplicates(subset=["session", "segment_idx"])
        .sort_values(["session", "segment_idx"])
        .reset_index(drop=True)
    )

    clips_dir = PROJECT_ROOT / "outputs" / dataset / "manual_clips"
    queue_path = PROJECT_ROOT / "outputs" / dataset / "manual_queue.csv"

    prior_labeled: set[tuple[int, int]] = set()
    if queue_path.exists():
        prior_df = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
        for _, prow in prior_df.iterrows():
            if prow["labeled"] == "1":
                prior_labeled.add((int(prow["session"]), int(prow["segment_idx"])))

    rows = []
    video_cache: dict[int, tuple[str, object]] = {}
    for _, row in segments.iterrows():
        session = int(row["session"])
        segment_idx = int(row["segment_idx"])

        if session not in video_cache:
            cam2 = _find_cam2_raw(dataset, session)
            video_cache[session] = ("cam2", cam2) if cam2 is not None else ("rgb", _find_participants_rgb(dataset, session))
        source_kind, video_path = video_cache[session]

        clip_name = f"session{session}_seg{segment_idx}.mp4"
        clip_path = clips_dir / clip_name
        rendered = False
        if video_path is not None:
            rendered, _ = render_clip(video_path, row["segment_start_sec"], row["segment_end_sec"], clip_path)

        rows.append({
            "dataset": dataset,
            "session": session,
            "segment_idx": segment_idx,
            "segment_start_sec": row["segment_start_sec"],
            "segment_end_sec": row["segment_end_sec"],
            "source": source_kind if rendered else "",
            "clip_path": str(clip_path.relative_to(PROJECT_ROOT)) if rendered else "",
            "labeled": "1" if (session, segment_idx) in prior_labeled else "0",
            "notes": "" if rendered else "MISSING SOURCE VIDEO — clip not rendered",
        })

    queue_df = pd.DataFrame(rows)
    queue_df.to_csv(queue_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return queue_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    queue_df = build_manual_queue(args.dataset)
    n_missing = (queue_df["clip_path"] == "").sum()
    print(f"{len(queue_df)} segments queued, {n_missing} missing source video, "
          f"{(queue_df['labeled'] == '1').sum()} already labeled.")
