"""Aggregate raw per-frame extraction CSVs into fixed-length segment features.

Two independent raw sources, aggregated on each camera's own clock since
there is no shared sync signal between them (confirmed multi-second drift
even within the same physical session — see cross_camera_duration_diff_sec
below) and no shared person-identity (facial_keypoints has a stable
track_id; gaze_target's person_idx is explicitly not stable across frames):

- facial_keypoints (RGB face camera) -> per (segment, track_id): this
  camera has real per-person tracking, so aggregates stay per-person.
  Its own gaze_target column already distinguishes "looking at track_N"
  from "looking at camera/robot" from "elsewhere" -- a relational signal
  keyed by the same stable track_id, aggregated as per-segment category
  proportions.
- gaze_target (mocap camera, Gaze-LLE) -> per segment only, no per-person
  split: proportions of every gaze observation in the window landing on
  robot / another participant / elsewhere, since person_idx can't be
  trusted to mean the same individual across frames.

The two are joined by segment_idx into one row per (session, segment,
track_id); sessions/segments missing a modality get NaN for that
modality's columns rather than being dropped.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd

# Columns present in facial_keypoints frame_features.csv that are NOT
# per-person continuous signal -- everything else numeric is aggregated
# via mean+std per (segment, track_id) automatically, so new blendshape/
# pose/gaze columns added upstream don't need this file touched.
_FK_ID_COLS = {"frame_index", "timestamp_sec", "track_id", "detected", "candidate_source", "gaze_target"}
_FK_BBOX_COLS = {
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "bbox_width", "bbox_height", "center_x_norm", "center_y_norm",
}
_FK_AUDIO_COLS = {
    "audio_rms_left", "audio_rms_right", "audio_rms_mean", "audio_peak_mean",
    "audio_zcr", "audio_spectral_centroid_hz", "audio_spectral_rolloff_hz",
    "audio_stereo_corr", "audio_lr_balance",
}


def _segment_idx(timestamp_sec: pd.Series, segment_length_sec: float) -> pd.Series:
    return (timestamp_sec // segment_length_sec).astype("Int64")


def aggregate_facial_keypoints(csv_path: Path, segment_length_sec: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["segment_idx"] = _segment_idx(df["timestamp_sec"], segment_length_sec)

    continuous_cols = sorted(
        set(df.columns) - _FK_ID_COLS - _FK_BBOX_COLS - _FK_AUDIO_COLS - {"segment_idx"}
    )

    # Per-(segment, track_id): continuous mean/std, detection rate, gaze_target proportions.
    rows = []
    for (seg, track_id), g in df.groupby(["segment_idx", "track_id"]):
        row = {"segment_idx": seg, "track_id": track_id}
        row["fk_detection_rate"] = g["detected"].mean()
        for col in continuous_cols:
            row[f"fk_{col}_mean"] = g[col].mean()
            row[f"fk_{col}_std"] = g[col].std()
        gt_counts = g["gaze_target"].value_counts(normalize=True)
        for cat, frac in gt_counts.items():
            row[f"fk_gaze_target_{cat}_frac"] = frac
        rows.append(row)
    person_df = pd.DataFrame(rows)

    # Per-segment only: audio (identical across track_id within a frame, verified on 03_20).
    audio_cols = sorted(_FK_AUDIO_COLS & set(df.columns))
    audio_df = (
        df.groupby("segment_idx")[audio_cols].mean().add_prefix("fk_").reset_index()
        if audio_cols else pd.DataFrame({"segment_idx": df["segment_idx"].unique()})
    )

    return person_df.merge(audio_df, on="segment_idx", how="left")


def aggregate_gaze_target(csv_path: Path, segment_length_sec: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["segment_idx"] = _segment_idx(df["timestamp_sec"], segment_length_sec)

    rows = []
    for seg, g in df.groupby("segment_idx"):
        total_frames = g["frame_index"].nunique()
        robot_frames = g.loc[g["robot_detected"] == 1, "frame_index"].nunique()
        person_rows = g[g["robot_detected"] == 1]
        n_person_rows = len(person_rows)

        row = {
            "segment_idx": seg,
            "gt_robot_detection_rate": (robot_frames / total_frames) if total_frames else float("nan"),
            "gt_avg_participants_detected": (n_person_rows / robot_frames) if robot_frames else float("nan"),
            "gt_inout_score_mean": pd.to_numeric(person_rows["inout_score"], errors="coerce").mean(),
        }
        if n_person_rows:
            gaze_counts = person_rows["gaze_target"].value_counts(normalize=True)
            for cat in ("robot", "participant", "elsewhere"):
                row[f"gt_looking_at_{cat}_rate"] = gaze_counts.get(cat, 0.0)
        else:
            for cat in ("robot", "participant", "elsewhere"):
                row[f"gt_looking_at_{cat}_rate"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_gaze_target_person(
    csv_path: Path, segment_length_sec: float, min_frames: int = 5
) -> pd.DataFrame:
    """Per-(segment, person_idx) features from gaze_target -- built for reconciling
    manual_engagement_labels.csv against sessions with no facial_keypoints/track_id
    coverage (see reconcile_manual_labels.py). gaze_target's person_idx is not
    stable across frames, so this has no cross-segment identity meaning; each
    segment's rows stand alone, matched to a manual label by on-screen position.

    min_frames drops person_idx entities detected in very few frames within a
    segment before they're treated as a real person -- confirmed empirically
    (2026-07-20, reconcile_manual_labels.py) that a spurious extra person_idx
    can appear for only 1-5 of a segment's ~90 frames when gaze_target's
    pick_robot() size-heuristic occasionally lets the robot's own head
    detection leak into the participant pool. A real participant, by
    contrast, is detected in nearly every frame they're in view (a head
    detector doesn't need a frontal face like facial_keypoints does), so
    sparse detection here is noise, not a real briefly-glimpsed person.
    """
    df = pd.read_csv(csv_path)
    df["segment_idx"] = _segment_idx(df["timestamp_sec"], segment_length_sec)

    rows = []
    for (seg, person_idx), g in df.dropna(subset=["person_idx"]).groupby(["segment_idx", "person_idx"]):
        if len(g) < min_frames:
            continue
        row = {
            "segment_idx": seg,
            "person_idx": person_idx,
            "gtp_n_frames": len(g),
            "gtp_center_x_median": ((g["bbox_x0"] + g["bbox_x1"]) / 2).median(),
            "gtp_center_y_median": ((g["bbox_y0"] + g["bbox_y1"]) / 2).median(),
            "gtp_inout_score_mean": pd.to_numeric(g["inout_score"], errors="coerce").mean(),
        }
        gaze_counts = g["gaze_target"].value_counts(normalize=True)
        for cat in ("robot", "participant", "elsewhere"):
            row[f"gtp_looking_at_{cat}_rate"] = gaze_counts.get(cat, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def _session_duration_sec(metadata_path: Path, key_path: tuple[str, ...]) -> Optional[float]:
    if not metadata_path.exists():
        return None
    meta = json.loads(metadata_path.read_text())
    node = meta
    for k in key_path:
        node = node.get(k, {})
    if isinstance(node, (int, float)):
        return float(node)
    return None


def build_session_table(
    dataset: str,
    session: int,
    fk_csv: Optional[Path],
    gt_csv: Optional[Path],
    segment_length_sec: float,
) -> pd.DataFrame:
    fk_df = aggregate_facial_keypoints(fk_csv, segment_length_sec) if fk_csv and fk_csv.exists() else None
    gt_df = aggregate_gaze_target(gt_csv, segment_length_sec) if gt_csv and gt_csv.exists() else None

    if fk_df is not None and gt_df is not None:
        merged = fk_df.merge(gt_df, on="segment_idx", how="outer")
    elif fk_df is not None:
        merged = fk_df
    elif gt_df is not None:
        merged = gt_df
    else:
        return pd.DataFrame()

    duration_diff = None
    if fk_csv and gt_csv:
        fk_dur = _session_duration_sec(fk_csv.parent / "session_metadata.json", ("stream", "duration_sec"))
        gt_meta_path = gt_csv.parent / "session_metadata.json"
        gt_dur = None
        if gt_meta_path.exists():
            meta = json.loads(gt_meta_path.read_text())
            frames = meta.get("stream", {}).get("total_frames_in_video")
            fps = meta.get("stream", {}).get("fps")
            if frames and fps:
                gt_dur = frames / fps
        if fk_dur is not None and gt_dur is not None:
            duration_diff = fk_dur - gt_dur

    merged.insert(0, "dataset", dataset)
    merged.insert(1, "session", session)
    merged["segment_start_sec"] = merged["segment_idx"] * segment_length_sec
    merged["segment_end_sec"] = merged["segment_start_sec"] + segment_length_sec
    merged["cross_camera_duration_diff_sec"] = duration_diff
    return merged
