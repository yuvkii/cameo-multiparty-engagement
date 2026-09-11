"""Join manual_labels_reconciled.csv to real per-person features, producing
one training-ready table per dataset.

Two feature sources depending on id_type (see reconcile_manual_labels.py):
- track_id rows -> join straight into segments.parquet's existing
  (session, segment_idx, track_id) rows (facial_keypoints features).
- person_idx rows -> gaze_target has no existing per-person rows in
  segments.parquet (person_idx was intentionally left out of that
  aggregation), so this is computed fresh here via
  segment_aggregation.aggregate_gaze_target_person and joined on
  (session, segment_idx, person_idx).

Rows with no matched_id (match_status != "matched") are kept with NaN
features rather than dropped -- so nothing is silently discarded; whether
to exclude them from a given training run is a downstream decision, not
baked in here.

Drops facial_keypoints' own eye-gaze columns (2026-07-20, user request):
`fk_gaze_yaw/pitch_deg`, `fk_eye_gaze_yaw/pitch_deg`, `fk_gaze_vector_x/y/z`,
`fk_gaze_target_*_frac` all come from that pipeline's calibrated-angle gaze
estimate, which has a documented unfixed bug (participant-vs-robot
precedence in attach_gaze_targets, uncertain pitch calibration -- see
facial_keypoints_pipeline history) and was never the trusted gaze source --
that's `gt_looking_at_*`/`gtp_looking_at_*` (Gaze-LLE, validated
separately). `fk_head_yaw/pitch/roll_deg` (head pose, not eye gaze) is kept,
since it's a different, not-flagged signal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from segment_aggregation import aggregate_gaze_target_person

PROJECT_ROOT = Path(__file__).parent.parent

_UNRELIABLE_FK_GAZE_COLS = [
    "fk_gaze_yaw_deg_mean", "fk_gaze_yaw_deg_std",
    "fk_gaze_pitch_deg_mean", "fk_gaze_pitch_deg_std",
    "fk_eye_gaze_yaw_deg_mean", "fk_eye_gaze_yaw_deg_std",
    "fk_eye_gaze_pitch_deg_mean", "fk_eye_gaze_pitch_deg_std",
    "fk_gaze_vector_x_mean", "fk_gaze_vector_x_std",
    "fk_gaze_vector_y_mean", "fk_gaze_vector_y_std",
    "fk_gaze_vector_z_mean", "fk_gaze_vector_z_std",
    "fk_gaze_target_missing_frac", "fk_gaze_target_elsewhere_frac",
    "fk_gaze_target_camera_frac",
    "fk_gaze_target_track_0_frac", "fk_gaze_target_track_1_frac", "fk_gaze_target_track_2_frac",
]


def build(dataset: str, segment_length_sec: float = 3.0) -> pd.DataFrame:
    reconciled = pd.read_csv(PROJECT_ROOT / "outputs" / dataset / "manual_labels_reconciled.csv").drop(
        columns=["dataset"]
    )
    segments = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "segments.parquet").drop(columns=["dataset"])

    track_rows = reconciled[reconciled["id_type"] == "track_id"].copy()
    track_rows["matched_id"] = track_rows["matched_id"].astype("Int64")
    segments_for_join = segments.rename(columns={"track_id": "matched_id"})
    segments_for_join["matched_id"] = segments_for_join["matched_id"].astype("Int64")
    track_joined = track_rows.merge(segments_for_join, on=["session", "segment_idx", "matched_id"], how="left")

    person_rows = reconciled[reconciled["id_type"] == "person_idx"].copy()
    person_joined = pd.DataFrame()
    if not person_rows.empty:
        gt_person_frames = []
        for session in sorted(person_rows["session"].unique()):
            csv_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
            feats = aggregate_gaze_target_person(csv_path, segment_length_sec)
            feats.insert(0, "session", session)
            gt_person_frames.append(feats)
        gt_person_features = pd.concat(gt_person_frames, ignore_index=True) if gt_person_frames else pd.DataFrame()
        gt_person_features = gt_person_features.rename(columns={"person_idx": "matched_id"})
        person_joined = person_rows.merge(gt_person_features, on=["session", "segment_idx", "matched_id"], how="left")

        # Group-level gt_* columns (whole-room gaze, keyed only by session+segment_idx,
        # already computed in segments.parquet) never depended on track_id/person_idx --
        # attach them here too, not just to the track_id branch, so cam2 sessions aren't
        # missing the group-level signal alongside their new per-person gtp_ features.
        gt_group_cols = [c for c in segments.columns if c.startswith("gt_")]
        gt_group = (
            segments[["session", "segment_idx"] + gt_group_cols]
            .drop_duplicates(subset=["session", "segment_idx"])
        )
        person_joined = person_joined.merge(gt_group, on=["session", "segment_idx"], how="left")

    unmatched = reconciled[~reconciled["id_type"].isin(["track_id", "person_idx"])].copy()

    combined = pd.concat([track_joined, person_joined, unmatched], ignore_index=True, sort=False)
    combined = combined.drop(columns=_UNRELIABLE_FK_GAZE_COLS, errors="ignore")
    combined = combined.sort_values(["session", "segment_idx", "position"]).reset_index(drop=True)
    combined.insert(0, "dataset", dataset)
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    result = build(args.dataset)
    out_path = PROJECT_ROOT / "outputs" / args.dataset / "manual_labeled_dataset.parquet"
    result.to_parquet(out_path, index=False)

    print(f"{len(result)} rows, {result.shape[1]} columns -> {out_path}")
    print(result["match_status"].value_counts())
    print(f"\nhuman_engagement_level distribution among matched rows:")
    print(result.loc[result["match_status"] == "matched", "human_engagement_level"].value_counts())
