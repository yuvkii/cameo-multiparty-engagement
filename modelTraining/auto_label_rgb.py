"""Automatic per-person engagement labeling for rgb (facial_keypoints)
sessions -- the rgb counterpart to auto_label_cam2.py, extending
auto-labeling to every dataset-date/session with facial_keypoints
coverage, not just cam2.

IMPORTANT CALIBRATION CAVEAT, unlike the cam2 classifier: 03_20 has real
hand labels for exactly ONE rgb session (session 5 -- sessions 1/2 moved
to cam2 during the queue-vs-Q&A confound investigation). There is no
second rgb session to hold out, so this classifier CANNOT be validated
with session-held-out CV the way the cam2 one was. Best available check is
plain 5-fold CV within that single session's 85 person-labels: acc=0.447,
macro_F1=0.348, linear-weighted kappa=0.215 ("fair" agreement, weaker than
cam2's "moderate" 0.445) -- and even that is optimistic, since it's still
implicitly testing on the same session/people the model trained on, just a
different 3s window. Treat this as substantially less trustworthy than the
cam2 auto-labels; it exists to test the "is it a data shortage" hypothesis
end to end per user request 2026-07-21, not as reliable signal on its own.

Segment-level rgb features are read directly from each dataset's already-
built segments.parquet (facial_keypoints aggregation already produces one
row per (session, segment_idx, track_id) there -- no need to recompute).
Some dataset-dates (03_25, 05_14, 05_15) are missing the 9 fk_audio_*
columns entirely (audio was never extracted for those dates) -- filled
with the TRAINING set's column mean (keeps them near-neutral) rather than
0, since 0 is wildly out-of-distribution for e.g. fk_audio_spectral_
centroid_hz (~3000 Hz typical) and would bias predictions for every row on
those dates in a way a genuine missing-signal convention shouldn't.
Per-row NaN within a column that IS present keeps the project's existing
nan_to_num(0) convention (real per-row missingness, not a whole-dataset
extraction gap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from assemble_dataset import _edge_tensor
from build_graph_dataset import build_rgb_edges

PROJECT_ROOT = Path(__file__).parent.parent
LEVELS = ["disengaged", "low", "medium", "high"]

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


def _fk_csv_path(dataset: str, session: int) -> Path | None:
    for candidate in (
        PROJECT_ROOT / "outputs" / dataset / "facial_keypoints" / f"session_{session}" / "frame_features.csv",
        PROJECT_ROOT / "outputs" / f"facial_keypoints_{dataset}" / f"session_{session}" / "frame_features.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def fit_rgb_classifier(train_dataset: str = "03_20") -> tuple[LogisticRegression, list[str], pd.Series]:
    data = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{train_dataset}.pt", weights_only=False)
    rgb = [d for d in data if d["source"] == "rgb" and d.get("label_source", "manual") == "manual"]
    manual = pd.read_parquet(PROJECT_ROOT / "outputs" / train_dataset / "manual_labeled_dataset.parquet")
    fk_cols = sorted(c for c in manual.columns if c.startswith("fk_") and c not in _UNRELIABLE_FK_GAZE_COLS)

    X, y = [], []
    for item in rgb:
        for i in range(item["node_features"].shape[0]):
            X.append(item["node_features"][i].numpy())
            y.append(item["labels"][i].item())
    X = np.array(X)
    train_means = pd.Series(X.mean(axis=0), index=fk_cols)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    clf.fit(X, np.array(y))
    return clf, fk_cols, train_means


def auto_label_rgb_dataset(dataset: str, clf: LogisticRegression, fk_cols: list[str], train_means: pd.Series) -> list[dict]:
    segments = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "segments.parquet")
    if "track_id" not in segments.columns:
        return []  # no facial_keypoints coverage at all for this dataset (e.g. 03_26)
    rgb_rows = segments.dropna(subset=["track_id"]).copy()
    if rgb_rows.empty:
        return []

    # whole-column gaps (dataset never extracted that signal) -> train-mean fill;
    # per-row gaps within a present column -> 0, matching the existing convention.
    for col in fk_cols:
        if col not in rgb_rows.columns:
            rgb_rows[col] = train_means[col]
        else:
            rgb_rows[col] = rgb_rows[col].fillna(0.0)

    results = []
    fk_csv_cache: dict[int, pd.DataFrame] = {}
    confidences = []

    for (session_f, segment_idx), group in rgb_rows.groupby(["session", "segment_idx"]):
        session = int(session_f)
        X = group[fk_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(X)
        preds = proba.argmax(1)
        conf = proba.max(1)
        confidences.extend(conf.tolist())
        track_ids = group["track_id"].astype(int).tolist()

        if session not in fk_csv_cache:
            csv_path = _fk_csv_path(dataset, session)
            fk_csv_cache[session] = pd.read_csv(csv_path) if csv_path else pd.DataFrame()
        fk_df = fk_csv_cache[session]
        start_sec, end_sec = float(group["segment_start_sec"].iloc[0]), float(group["segment_end_sec"].iloc[0])
        seg_window = fk_df[(fk_df["timestamp_sec"] >= start_sec) & (fk_df["timestamp_sec"] < end_sec)] if not fk_df.empty else fk_df
        edges = build_rgb_edges(seg_window, track_ids) if not seg_window.empty else {}
        edge_tensor = _edge_tensor(edges, track_ids)

        results.append({
            "dataset": dataset, "session": session, "segment_idx": int(segment_idx), "source": "rgb",
            "node_features": torch.tensor(X, dtype=torch.float32), "edge_features": edge_tensor,
            "labels": torch.tensor(preds, dtype=torch.long), "label_source": "auto",
            "confidence": torch.tensor(conf, dtype=torch.float32),
        })

    if confidences:
        c = np.array(confidences)
        print(f"  rgb confidence: mean={c.mean():.3f} median={np.median(c):.3f}")
    return results
