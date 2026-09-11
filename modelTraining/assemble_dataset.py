"""Assemble the final per-segment graph dataset for training.

Ties together: manual position labels (manual_engagement_labels.csv),
corrected node identities + edges (build_graph_dataset.py), and node
features (fk_* from manual_labeled_dataset.parquet for rgb sessions;
freshly recomputed gaze_target aggregates for cam2 sessions, since cam2
node identity was just corrected via re-identification and no longer
matches raw person_idx).

cam2 sessions need position-matching redone here (rank corrected local_ids
by median on-screen x, same convention manual_labeler.py's reviewer used --
left to right, position 1 = leftmost) because reconcile_manual_labels.py's
original matched_id was keyed to raw person_idx, which is no longer valid
after re-identification. rgb sessions don't need this: track_id was
already stable, so the existing matched_id from manual_labeled_dataset.parquet
is reused directly.

Output: one row per segment in a list of dicts (variable node count, 1-4),
each with node feature matrix, edge feature tensor, per-node labels, and
metadata -- saved via torch.save for the training script to consume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_graph_dataset import (
    CAM2_NODE_FEATURE_COLS,
    OPENFACE_NODE_FEATURE_COLS,
    build_cam2_edges,
    build_cam2_node_features,
    build_rgb_edges,
    merge_openface_features,
    reidentify_segment,
)

ALL_CAM2_NODE_FEATURE_COLS = CAM2_NODE_FEATURE_COLS + OPENFACE_NODE_FEATURE_COLS

PROJECT_ROOT = Path(__file__).parent.parent

# Same floor used in reconcile_manual_labels.py -- entities detected in fewer
# than this many frames are dropped before ranking (robot-leak / noise guard).
MIN_FRAMES_CAM2 = 5

LEVEL_TO_IDX = {"disengaged": 0, "low": 1, "medium": 2, "high": 3}
# Anchors matching train.py's ANCHORS -- newer datasets (e.g. 05_15) are
# labeled directly as a 0-100 slider percentage rather than one of the 4
# category names, so bucket to the nearest anchor for the current 4-way
# classifier. The raw percentage itself is not lost -- it's still sitting
# in manual_engagement_labels.csv for any future continuous/ordinal scheme.
_LEVEL_ANCHORS = [0.0, 100.0 / 3, 200.0 / 3, 100.0]


def level_to_idx(value: str) -> int:
    if value in LEVEL_TO_IDX:
        return LEVEL_TO_IDX[value]
    pct = float(value)
    return min(range(4), key=lambda i: abs(_LEVEL_ANCHORS[i] - pct))


def _edge_tensor(edges: dict, ids: list[int]) -> torch.Tensor:
    """(N, N, 4) tensor: [gaze_rate, mean_dist (nan-safe), mutual_gaze,
    attending_to_speaker (added 2026-07-23, 0.0 for rgb sessions -- no
    mouth-motion extraction built for that camera)]."""
    n = len(ids)
    idx = {node_id: i for i, node_id in enumerate(ids)}
    t = torch.zeros(n, n, 4)
    for (i, j), feats in edges.items():
        if i not in idx or j not in idx:
            continue
        a, b = idx[i], idx[j]
        dist = feats["mean_dist"]
        t[a, b, 0] = feats["gaze_rate"]
        t[a, b, 1] = 0.0 if np.isnan(dist) else 1.0 / (1.0 + dist / 500.0)  # proximity, bounded (0,1]
        t[a, b, 2] = feats["mutual_gaze"]
        t[a, b, 3] = feats.get("attending_to_speaker", 0.0)
    return t


def process_cam2_segment(dataset: str, session: int, segment_idx: int, start_sec: float, end_sec: float,
                          labels: pd.DataFrame, gt_csv_cache: dict, openface_csv_cache: dict,
                          mouth_motion_cache: dict) -> dict | None:
    if session not in gt_csv_cache:
        csv_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
        gt_csv_cache[session] = pd.read_csv(csv_path)
    df = gt_csv_cache[session]

    if session not in openface_csv_cache:
        of_path = PROJECT_ROOT / "outputs" / dataset / "openface_cam2" / f"session_{session}" / "frame_features.csv"
        openface_csv_cache[session] = pd.read_csv(of_path) if of_path.exists() else pd.DataFrame()
    of_df = openface_csv_cache[session]

    if session not in mouth_motion_cache:
        mm_path = PROJECT_ROOT / "outputs" / dataset / "mouth_motion" / f"session_{session}" / "frame_features.csv"
        mouth_motion_cache[session] = pd.read_csv(mm_path) if mm_path.exists() else pd.DataFrame()
    mm_df = mouth_motion_cache[session]

    seg = df[(df["timestamp_sec"] >= start_sec) & (df["timestamp_sec"] < end_sec)].dropna(subset=["person_idx"])
    if seg.empty:
        return None
    seg = reidentify_segment(seg)
    node_feats = build_cam2_node_features(seg)
    node_feats = node_feats[node_feats["n_frames"] >= MIN_FRAMES_CAM2].reset_index(drop=True)
    if node_feats.empty:
        return None

    of_seg = of_df[(of_df["timestamp_sec"] >= start_sec) & (of_df["timestamp_sec"] < end_sec)] if not of_df.empty else of_df
    node_feats = merge_openface_features(node_feats, seg, of_seg)

    ranked = node_feats.sort_values("center_x_median").reset_index(drop=True)
    local_ids = ranked["local_id"].astype(int).tolist()

    rated = labels[labels["position"] != "0"].copy()
    rated["position"] = rated["position"].astype(int)
    rated = rated.sort_values("position")

    node_rows, level_labels = [], []
    for _, r in rated.iterrows():
        rank_idx = r["position"] - 1
        if rank_idx >= len(local_ids):
            continue
        node_rows.append(ranked.iloc[rank_idx])
        level_labels.append(level_to_idx(r["human_engagement_level"]))
    if not node_rows:
        return None

    used_ids = [int(nr["local_id"]) for nr in node_rows]
    mm_seg = mm_df[(mm_df["frame_index"] >= seg["frame_index"].min()) & (mm_df["frame_index"] <= seg["frame_index"].max())] if not mm_df.empty else mm_df
    edges = build_cam2_edges(seg, local_ids, mouth_motion_df=mm_seg)
    edge_tensor = _edge_tensor(edges, used_ids)

    node_feature_matrix = torch.tensor(
        [[nr[c] for c in ALL_CAM2_NODE_FEATURE_COLS] for nr in node_rows], dtype=torch.float32
    )
    return {
        "dataset": dataset, "session": session, "segment_idx": segment_idx, "source": "cam2",
        "node_features": node_feature_matrix, "edge_features": edge_tensor,
        "labels": torch.tensor(level_labels, dtype=torch.long), "label_source": "manual",
    }


def process_rgb_segment(rows: pd.DataFrame, fk_cols: list[str]) -> dict | None:
    rows = rows[rows["match_status"] == "matched"].dropna(subset=fk_cols)
    if rows.empty:
        return None
    rows = rows.sort_values("position")
    node_feature_matrix = torch.nan_to_num(torch.tensor(rows[fk_cols].to_numpy(dtype=np.float32)), nan=0.0)
    labels = torch.tensor([level_to_idx(lvl) for lvl in rows["human_engagement_level"]], dtype=torch.long)
    return {
        "dataset": rows["dataset"].iloc[0], "session": int(rows["session"].iloc[0]),
        "segment_idx": int(rows["segment_idx"].iloc[0]), "source": "rgb",
        "node_features": node_feature_matrix, "labels": labels, "label_source": "manual",
        "track_ids": rows["matched_id"].astype(int).tolist(),
    }


def assemble(dataset: str) -> list[dict]:
    manual_df = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "manual_labeled_dataset.parquet")
    manual_labels = pd.read_csv(PROJECT_ROOT / "outputs" / dataset / "manual_engagement_labels.csv", dtype=str)
    queue = pd.read_csv(
        PROJECT_ROOT / "outputs" / dataset / "manual_queue.csv", dtype=str, keep_default_na=False
    ).set_index(["session", "segment_idx"])
    fk_cols = [c for c in manual_df.columns if c.startswith("fk_")]

    results = []
    gt_csv_cache: dict = {}
    openface_csv_cache: dict = {}
    mouth_motion_cache: dict = {}
    edge_cache: dict[tuple[int, int], dict] = {}

    for (session_s, segment_s), group in manual_labels.groupby(["session", "segment_idx"]):
        session, segment_idx = int(session_s), int(segment_s)
        q_row = queue.loc[(session_s, segment_s)]
        source = q_row["source"]
        start_sec, end_sec = float(q_row["segment_start_sec"]), float(q_row["segment_end_sec"])

        if source == "cam2":
            item = process_cam2_segment(dataset, session, segment_idx, start_sec, end_sec, group, gt_csv_cache,
                                         openface_csv_cache, mouth_motion_cache)
        elif source == "rgb":
            rows = manual_df[(manual_df["session"] == session) & (manual_df["segment_idx"] == segment_idx)]
            item = process_rgb_segment(rows, fk_cols)
            if item is not None:
                seg_key = (session, segment_idx)
                if seg_key not in edge_cache:
                    csv_path = PROJECT_ROOT / "outputs" / dataset / "facial_keypoints" / f"session_{session}" / "frame_features.csv"
                    fk_df = pd.read_csv(csv_path)
                    seg_window = fk_df[(fk_df["timestamp_sec"] >= start_sec) & (fk_df["timestamp_sec"] < end_sec)]
                    edge_cache[seg_key] = build_rgb_edges(seg_window, item["track_ids"])
                item["edge_features"] = _edge_tensor(edge_cache[seg_key], item["track_ids"])
        else:
            item = None

        if item is not None and item["node_features"].shape[0] >= 1:
            results.append(item)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    args = parser.parse_args()

    data = assemble(args.dataset)
    out_path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{args.dataset}.pt"
    torch.save(data, out_path)

    n_cam2 = sum(1 for d in data if d["source"] == "cam2")
    n_rgb = sum(1 for d in data if d["source"] == "rgb")
    sizes = [d["node_features"].shape[0] for d in data]
    print(f"{len(data)} graphs -> {out_path} (cam2={n_cam2}, rgb={n_rgb})")
    print(f"node counts: {pd.Series(sizes).value_counts().sort_index().to_dict()}")
