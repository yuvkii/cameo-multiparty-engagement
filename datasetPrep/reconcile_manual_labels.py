"""Reconcile position-based manual engagement labels back to detected entities.

manual_engagement_labels.csv scores people by on-screen left-to-right
PIXEL position at labeling time (position 1 = leftmost), deliberately
avoiding track_id/person_idx -- both have been shown to drift/be unstable
(see datasetPrep/manual_labeler.py's docstring). This script is the
code-driven join deferred at that time: for each labeled (session,
segment), it re-derives left-to-right order from the same raw per-frame
detections the human was implicitly seeing, and matches position N to
whichever track_id/person_idx ranks Nth by on-screen x-position in that
window.

Two different sources depending on which camera a session's manual clips
were rendered from (see manual_queue.csv's `source` column):
- rgb sessions -> facial_keypoints frame_features.csv, keyed by track_id.
  Native pixel space, no resize (verified against pipeline.py / head_detector.py).
- cam2 sessions -> gaze_target frame_features.csv, keyed by person_idx.
  This pipeline has NO existing per-person rows in segments.parquet
  (person_idx is intentionally not carried into aggregation), so this
  script is the first thing to ever produce a person-level row for these
  sessions/segments -- not just a join, but new per-person data.

Robot-leak guard: gaze_target's pick_robot() splits robot vs participant
head detections by a relative-size heuristic (>=1.5x the next-largest),
which can fail (a participant leaning in, or the robot turning away can
flip it) and let the robot's own detection into the person_idx pool.
Confirmed empirically (2026-07-20 on 03_20 session 3/4): this shows up as
a spurious extra person_idx (always the 4th/5th slot beyond the real
participants) present in only 1-5 of a segment's ~90 frames, sitting at
the same x-position as the robot -- a duplicate/leftover head detection,
not a real person flickering in and out. A real head detector doesn't
need a frontal face like facial_keypoints does, so a genuine participant
is detected in nearly every frame they're in view regardless of gaze
direction; a person_idx that appears in only a handful of frames per
segment is noise, not a briefly-glimpsed person. So for the cam2/
person_idx path only, entities below _MIN_FRAMES_PERSON_IDX are dropped
from the ranking pool entirely before matching (NOT applied to the rgb/
track_id path, where sparse detection is real signal -- a face detector
genuinely fails when someone looks away, which is meaningful, not noise).

A second-line-of-defense check (flagging any surviving match whose bbox
sits close to the robot's) was tried and DROPPED (2026-07-20): visually
verified false positive on session 3 segment 25 -- some segments
genuinely have 4 real participants (not the usual 3), and a real person
standing near the robot in the wide Cam_2 shot triggered the same
distance heuristic as the actual noise case. Proximity-to-robot alone
isn't a reliable leak signal once real headcount varies; the frame-count
filter above is what actually catches the leak (verified: the one
confirmed leak was n_frames=1 out of ~90, sessions with a real 4th
person show that person detected in most/all frames like everyone else).

Where a session has fewer detected entities in-window than labeled
positions (e.g. someone was never detected -- see the same detection-
dropout-correlates-with-disengagement issue that motivated going manual
in the first place), the extra position(s) are left unmatched rather than
guessed at.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# cam2/person_idx entities detected in fewer than this many frames within a
# segment are dropped before ranking -- see robot-leak guard note above.
_MIN_FRAMES_PERSON_IDX = 5


def _fk_track_positions(dataset: str, session: int, start_sec: float, end_sec: float) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "outputs" / dataset / "facial_keypoints" / f"session_{session}" / "frame_features.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["entity_id", "median_x", "n_frames"])
    df = pd.read_csv(csv_path, usecols=["frame_index", "timestamp_sec", "track_id", "detected", "bbox_x0", "bbox_x1"])
    sub = df[(df["detected"] == 1) & (df["timestamp_sec"] >= start_sec) & (df["timestamp_sec"] < end_sec)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["entity_id", "median_x", "n_frames"])
    sub["center_x"] = (sub["bbox_x0"] + sub["bbox_x1"]) / 2
    out = sub.groupby("track_id").agg(median_x=("center_x", "median"), n_frames=("center_x", "size")).reset_index()
    return out.rename(columns={"track_id": "entity_id"})


def _gt_person_positions(dataset: str, session: int, start_sec: float, end_sec: float) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["entity_id", "median_x", "n_frames"])
    df = pd.read_csv(
        csv_path, usecols=["frame_index", "timestamp_sec", "person_idx", "bbox_x0", "bbox_x1"]
    )
    window = df[(df["timestamp_sec"] >= start_sec) & (df["timestamp_sec"] < end_sec)]

    person_rows = window.dropna(subset=["person_idx", "bbox_x0", "bbox_x1"]).copy()
    if person_rows.empty:
        return pd.DataFrame(columns=["entity_id", "median_x", "n_frames"])
    person_rows["center_x"] = (person_rows["bbox_x0"] + person_rows["bbox_x1"]) / 2
    out = (
        person_rows.groupby("person_idx")
        .agg(median_x=("center_x", "median"), n_frames=("center_x", "size"))
        .reset_index()
        .rename(columns={"person_idx": "entity_id"})
    )
    return out[out["n_frames"] >= _MIN_FRAMES_PERSON_IDX].reset_index(drop=True)


def reconcile(dataset: str) -> pd.DataFrame:
    labels_df = pd.read_csv(
        PROJECT_ROOT / "outputs" / dataset / "manual_engagement_labels.csv", dtype=str, keep_default_na=False
    )
    queue_df = pd.read_csv(
        PROJECT_ROOT / "outputs" / dataset / "manual_queue.csv", dtype=str, keep_default_na=False
    ).set_index(["session", "segment_idx"])

    out_rows = []

    for (session_s, segment_s), group in labels_df.groupby(["session", "segment_idx"]):
        session, segment_idx = int(session_s), int(segment_s)
        q_row = queue_df.loc[(session_s, segment_s)]
        start_sec, end_sec = float(q_row["segment_start_sec"]), float(q_row["segment_end_sec"])
        source = q_row["source"]

        rated = group[group["position"] != "0"].copy()
        rated["position"] = rated["position"].astype(int)
        rated = rated.sort_values("position")

        unrated = group[group["position"] == "0"]
        for _, r in unrated.iterrows():
            out_rows.append({
                "dataset": dataset, "session": session, "segment_idx": segment_idx, "position": 0,
                "human_engagement_level": r["human_engagement_level"],
                "id_type": "", "matched_id": "", "n_frames_used": "",
                "match_status": "not_rated",
            })

        if rated.empty:
            continue

        if source == "rgb":
            positions_df = _fk_track_positions(dataset, session, start_sec, end_sec)
            id_type = "track_id"
        elif source == "cam2":
            positions_df = _gt_person_positions(dataset, session, start_sec, end_sec)
            id_type = "person_idx"
        else:
            positions_df, id_type = pd.DataFrame(columns=["entity_id", "median_x", "n_frames"]), ""

        ranked = positions_df.sort_values("median_x").reset_index(drop=True)

        for _, r in rated.iterrows():
            pos = r["position"]
            rank_idx = pos - 1
            if rank_idx < len(ranked):
                entity = ranked.iloc[rank_idx]
                out_rows.append({
                    "dataset": dataset, "session": session, "segment_idx": segment_idx, "position": pos,
                    "human_engagement_level": r["human_engagement_level"],
                    "id_type": id_type, "matched_id": entity["entity_id"], "n_frames_used": int(entity["n_frames"]),
                    "match_status": "matched",
                })
            else:
                out_rows.append({
                    "dataset": dataset, "session": session, "segment_idx": segment_idx, "position": pos,
                    "human_engagement_level": r["human_engagement_level"],
                    "id_type": id_type, "matched_id": "", "n_frames_used": "",
                    "match_status": "unmatched_no_detection",
                })

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    result = reconcile(args.dataset)
    out_path = PROJECT_ROOT / "outputs" / args.dataset / "manual_labels_reconciled.csv"
    result.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"{len(result)} rows -> {out_path}")
    print(result["match_status"].value_counts())
