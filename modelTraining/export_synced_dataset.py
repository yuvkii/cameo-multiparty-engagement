"""Export the fully-synced, human-labelled dataset as flat CSVs (for sharing).

One row per (dataset, session, sampled frame, participant): that participant's
gaze-target features, OpenFace facial features, mouth-motion, and their
human-annotated continuous engagement value at that exact frame -- i.e. the
three separately-exported per-modality feature sets already joined and with
identity reconciled to the annotator's participant A/B/C.

This deliberately reuses build_continuous_dataset.py's own sources, sampling
grid and identity reconciliation (rank-based vs tracked per session, plus the
validated usable-window cutoffs) rather than re-deriving any of it, so the
exported table contains exactly the (frame, person, label) triples the model is
actually trained on -- no session or stretch of footage whose identity
reconciliation hasn't been visually validated is included.

Usage:
    dataExtraction/openface3/.venv/bin/python3 \
        modelTraining/export_synced_dataset.py --out-dir supervisor_export/synced
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_continuous_dataset import (  # noqa: E402
    DATASET_STABLE_RANGES,
    DATASET_TRACKED_SESSIONS,
    POSITIONS,
    TRACKER_MAX_MISSED_SEC,
    _load_session_sources,
    _rank_based_letter_lookup,
    _tracked_letter_lookup,
    frame_step_for,
)

PROJECT_ROOT = Path(__file__).parent.parent

EMOTION_LABELS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]

GAZE_COLS = ["gaze_target", "gaze_peak_x", "gaze_peak_y",
             "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
             "robot_detected", "robot_x0", "robot_y0", "robot_x1", "robot_y1"]
OF_COLS = ["match_iou", "face_conf", "emotion_argmax", "gaze_yaw", "gaze_pitch"] + [f"au_{k}" for k in range(8)]

OUT_COLS = (["dataset", "session", "fps", "frame_index", "timestamp_sec",
             "participant", "person_idx", "identity_method"]
            + GAZE_COLS
            + OF_COLS + ["emotion_label", "face_detected"]
            + ["mouth_motion", "engagement_pct"])


def session_rows(dataset: str, session: int, stable_range) -> pd.DataFrame:
    min_ts = stable_range[0] if isinstance(stable_range, tuple) else stable_range
    max_ts = stable_range[1] if isinstance(stable_range, tuple) else float("inf")

    sources = _load_session_sources(dataset, session)
    gt_df, of_df, mm_df, traces = sources["gt"], sources["of"], sources["mm"], sources["traces"]
    fps = sources["fps"]

    tracked = session in DATASET_TRACKED_SESSIONS.get(dataset, set())
    if tracked:
        lookup = _tracked_letter_lookup(
            gt_df, min_ts, max_missed_sec=TRACKER_MAX_MISSED_SEC.get(dataset, {}).get(session))
    else:
        lookup = _rank_based_letter_lookup(gt_df, min_ts)

    # Same sampling grid as the model's dataset builder: stride from the global
    # 0-based grid FIRST, then apply the usable-window cutoffs, so the sampled
    # frames always line up with the grid the facial features were extracted on.
    frames = sorted(gt_df["frame_index"].unique())[::frame_step_for(fps)]
    ts_by_frame = gt_df.drop_duplicates("frame_index").set_index("frame_index")["timestamp_sec"]
    frames = [f for f in frames if min_ts <= ts_by_frame.loc[f] <= max_ts]
    frame_set = set(int(f) for f in frames)

    sub = gt_df[gt_df["frame_index"].isin(frame_set)].dropna(subset=["person_idx", "bbox_x0", "bbox_x1"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=OUT_COLS)

    sub["person_idx"] = sub["person_idx"].astype(int)
    sub["frame_index"] = sub["frame_index"].astype(int)
    sub["participant"] = [lookup.get((f, p)) for f, p in zip(sub["frame_index"], sub["person_idx"])]
    sub = sub[sub["participant"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=OUT_COLS)

    # Human engagement label at this exact frame for this exact participant.
    label = []
    for f, p in zip(sub["frame_index"], sub["participant"]):
        trace = traces[p]
        label.append(float(trace.loc[f]) if trace is not None and f in trace.index else None)
    sub["engagement_pct"] = label
    sub = sub[sub["engagement_pct"].notna()]

    for col in GAZE_COLS:
        if col not in sub.columns:
            sub[col] = pd.NA

    if not of_df.empty:
        of = of_df.copy()
        of["frame_index"] = of["frame_index"].astype(int)
        of["person_idx"] = of["person_idx"].astype(int)
        of = of.drop_duplicates(["frame_index", "person_idx"])
        sub = sub.merge(of[["frame_index", "person_idx"] + OF_COLS], on=["frame_index", "person_idx"], how="left")
    else:
        for col in OF_COLS:
            sub[col] = pd.NA

    if not mm_df.empty:
        mm = mm_df.copy()
        mm["frame_index"] = mm["frame_index"].astype(int)
        mm["person_idx"] = mm["person_idx"].astype(int)
        mm = mm.drop_duplicates(["frame_index", "person_idx"])
        sub = sub.merge(mm[["frame_index", "person_idx", "mouth_motion"]], on=["frame_index", "person_idx"], how="left")
    else:
        sub["mouth_motion"] = pd.NA

    sub["face_detected"] = sub["face_conf"].notna().astype(int)
    sub["emotion_label"] = [EMOTION_LABELS[int(e)] if pd.notna(e) else None for e in sub["emotion_argmax"]]
    sub["dataset"] = dataset
    sub["session"] = session
    sub["fps"] = fps
    sub["identity_method"] = "tracked" if tracked else "position_rank"

    sub = sub[OUT_COLS].sort_values(["frame_index", "participant"]).reset_index(drop=True)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="supervisor_export/synced")
    args = ap.parse_args()

    out_root = PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    all_parts, manifest = [], []
    for dataset in sorted(DATASET_STABLE_RANGES):
        for session in sorted(DATASET_STABLE_RANGES[dataset]):
            rng = DATASET_STABLE_RANGES[dataset][session]
            df = session_rows(dataset, session, rng)
            out_dir = out_root / dataset / f"session_{session}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / "synced_features_labels.csv", index=False)
            all_parts.append(df)

            min_ts = rng[0] if isinstance(rng, tuple) else rng
            max_ts = rng[1] if isinstance(rng, tuple) else ""
            manifest.append({
                "dataset": dataset, "session": session,
                "fps": df["fps"].iloc[0] if not df.empty else "",
                "rows": len(df),
                "participants": "".join(sorted(set(df["participant"]))) if not df.empty else "",
                "duration_sec": round(float(df["timestamp_sec"].max() - df["timestamp_sec"].min()), 1) if not df.empty else 0,
                "window_start_sec": min_ts, "window_end_sec": max_ts,
                "identity_method": "tracked" if session in DATASET_TRACKED_SESSIONS.get(dataset, set()) else "position_rank",
                "face_coverage": round(float(df["face_detected"].mean()), 3) if not df.empty else "",
                "mean_engagement": round(float(df["engagement_pct"].mean()), 1) if not df.empty else "",
            })
            print(f"{dataset} s{session}: {len(df)} rows")

    combined = pd.concat(all_parts, ignore_index=True)
    combined.to_csv(out_root / "synced_features_labels_all.csv", index=False)
    pd.DataFrame(manifest).to_csv(out_root / "session_manifest.csv", index=False)
    print(f"\nTOTAL {len(combined)} rows -> {out_root}")


if __name__ == "__main__":
    main()
