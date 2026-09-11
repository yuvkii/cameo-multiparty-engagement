"""Automatic per-person engagement labeling for cam2 sessions, to replace
fully-manual labeling for dataset-dates beyond 03_20 (397 segments for
03_26 alone made full manual labeling unsustainable -- see the design notes'
annotation strategy, which always intended auto-label-then-review, not
label-from-scratch).

Different from the abandoned 2026-07-19 heuristic (engagement_heuristic.py)
in the two ways that got that one abandoned:
  1. A person who never accumulates >= MIN_FRAMES_CAM2 detected frames in a
     segment gets no label at all -- not a "low confidence" placeholder,
     not a borrowed group-level signal. Same convention the real training
     pipeline already uses (build_graph_dataset.py's node-feature floor).
  2. Confidence is a genuine predict_proba output from a classifier
     calibrated against real human labels, not an ad hoc heuristic score.

Classifier is a plain LogisticRegression on the 11-dim cam2 node features
(build_cam2_node_features/CAM2_NODE_FEATURE_COLS) -- deliberately NOT
CAMEOModel itself, to keep "the tool that helps generate more labels"
separate from "the model being evaluated" (no circularity).

Validated 2026-07-21 via session-held-out CV on 03_20's 599 hand-labeled
cam2 person-labels: acc=0.524, macro_F1=0.455, linear-weighted Cohen's
kappa=0.445 ("moderate agreement" on the standard scale) against real
human labels. Confidence only weakly predicts correctness (routing the
bottom 50% by confidence to review only moved kept-label accuracy from
0.547 to 0.553) -- so this is NOT trustworthy as ground truth. Per user
decision (2026-07-21): used as additional TRAINING data only. train.py's
leave_one_session_out excludes any session with label_source="auto" from
ever being the held-out test fold -- auto-labeled sessions only ever
appear in a training pool, real evaluation stays on hand-labeled sessions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from assemble_dataset import MIN_FRAMES_CAM2, _edge_tensor
from build_graph_dataset import (
    CAM2_NODE_FEATURE_COLS,
    build_cam2_edges,
    build_cam2_node_features,
    reidentify_segment,
)

PROJECT_ROOT = Path(__file__).parent.parent
LEVELS = ["disengaged", "low", "medium", "high"]


def fit_classifier(train_dataset: str = "03_20") -> LogisticRegression:
    data = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{train_dataset}.pt", weights_only=False)
    cam2 = [d for d in data if d["source"] == "cam2" and d.get("label_source", "manual") == "manual"]
    X, y = [], []
    for item in cam2:
        for i in range(item["node_features"].shape[0]):
            X.append(item["node_features"][i].numpy())
            y.append(item["labels"][i].item())
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(np.array(X), np.array(y))
    return clf


def auto_label_dataset(dataset: str, clf: LogisticRegression, gaze_target_subdir: str = "gaze_target",
                        camera_tag: "int | None" = None) -> list[dict]:
    """gaze_target_subdir/camera_tag let this run against an auxiliary
    mocap camera's own extraction (e.g. outputs/<dataset>/gaze_target_cam5/)
    instead of the default Cam_2 output -- same classifier (fit on Cam_2
    features, but the feature *shape* is camera-agnostic: gaze-containment/
    proximity/mutual-gaze computed the same way regardless of which camera
    the raw detections came from), same segment boundaries (segments.parquet
    is time-based, not camera-specific). camera_tag is stamped onto each
    output graph purely for traceability/debugging, not used by training."""
    segments = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "segments.parquet")
    seg_bounds = (
        segments[["session", "segment_idx", "segment_start_sec", "segment_end_sec"]]
        .drop_duplicates(subset=["session", "segment_idx"])
    )

    results = []
    gt_cache: dict[int, pd.DataFrame] = {}
    confidences = []

    for row in seg_bounds.itertuples():
        session, segment_idx = int(row.session), int(row.segment_idx)
        csv_path = PROJECT_ROOT / "outputs" / dataset / gaze_target_subdir / f"session_{session}" / "frame_features.csv"
        if not csv_path.exists():
            continue  # rgb-only session, or this camera has no data for this session
        if session not in gt_cache:
            gt_cache[session] = pd.read_csv(csv_path)
        df = gt_cache[session]

        seg = df[(df["timestamp_sec"] >= row.segment_start_sec) & (df["timestamp_sec"] < row.segment_end_sec)]
        seg = seg.dropna(subset=["person_idx"])
        if seg.empty:
            continue
        seg = reidentify_segment(seg)
        node_feats = build_cam2_node_features(seg)
        node_feats = node_feats[node_feats["n_frames"] >= MIN_FRAMES_CAM2].reset_index(drop=True)
        if node_feats.empty:
            continue

        ranked = node_feats.sort_values("center_x_median").reset_index(drop=True)
        X = ranked[CAM2_NODE_FEATURE_COLS].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(X)
        preds = proba.argmax(1)
        conf = proba.max(1)
        confidences.extend(conf.tolist())

        local_ids = ranked["local_id"].astype(int).tolist()
        edges = build_cam2_edges(seg, local_ids)
        edge_tensor = _edge_tensor(edges, local_ids)

        graph = {
            "dataset": dataset, "session": session, "segment_idx": segment_idx, "source": "cam2",
            "node_features": torch.tensor(X, dtype=torch.float32), "edge_features": edge_tensor,
            "labels": torch.tensor(preds, dtype=torch.long), "label_source": "auto",
            "confidence": torch.tensor(conf, dtype=torch.float32),
        }
        if camera_tag is not None:
            graph["camera"] = camera_tag
        results.append(graph)

    if confidences:
        c = np.array(confidences)
        print(f"confidence: mean={c.mean():.3f} median={np.median(c):.3f} min={c.min():.3f} max={c.max():.3f}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="target dataset-date to auto-label, e.g. 03_26")
    parser.add_argument("--train-on", default="03_20", help="dataset-date with real manual labels to fit the classifier on")
    args = parser.parse_args()

    print(f"fitting classifier on {args.train_on} (manual cam2 labels only)...")
    clf = fit_classifier(args.train_on)

    print(f"auto-labeling {args.dataset}...")
    data = auto_label_dataset(args.dataset, clf)

    out_path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{args.dataset}.pt"
    torch.save(data, out_path)

    sizes = [d["node_features"].shape[0] for d in data]
    all_labels = torch.cat([d["labels"] for d in data]) if data else torch.tensor([])
    print(f"{len(data)} graphs -> {out_path}")
    print(f"node counts: {pd.Series(sizes).value_counts().sort_index().to_dict()}")
    print(f"predicted label distribution: {dict(zip(LEVELS, [int((all_labels == i).sum()) for i in range(4)]))}")
    print("\nNOTE: label_source='auto' -- train.py will never select these sessions as a held-out test fold.")
