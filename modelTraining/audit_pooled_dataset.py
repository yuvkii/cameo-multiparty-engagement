"""Comprehensive, read-only audit of the pooled continuous-label dataset
(all labeled dates) treated as ONE dataset -- checks completeness,
cross-date consistency, and join integrity BEFORE any further training.
Does not train or modify anything.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/audit_pooled_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS = ["03_20", "03_26", "05_15", "05_14"]

# NODE_FEATURE_COLS order (29-dim): 4 gaze_target one-hot + center_x/y (2)
# + bbox w/h (2) + of_availability (1) + face_conf (1) + gaze_yaw/pitch (2)
# + au_0..7 (8) + emotion one-hot (8) + mouth_motion (1)
DIM_NAMES = (
    ["gaze_robot", "gaze_elsewhere", "gaze_participant", "gaze_unknown",
     "center_x", "center_y", "bbox_w", "bbox_h", "of_availability", "face_conf",
     "gaze_yaw", "gaze_pitch"]
    + [f"au_{i}" for i in range(8)]
    + [f"emotion_{i}" for i in range(8)]
    + ["mouth_motion"]
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    findings = []

    section("1. Session inventory vs raw source files")
    from build_continuous_dataset import DATASET_STABLE_RANGES, DATASET_TRACKED_SESSIONS, frame_step_for
    for ds, sessions in DATASET_STABLE_RANGES.items():
        for sess in sessions:
            gt = PROJECT_ROOT / "outputs" / ds / "gaze_target" / f"session_{sess}" / "frame_features.csv"
            of = PROJECT_ROOT / "outputs" / ds / "openface_cam2" / f"session_{sess}" / "frame_features.csv"
            mm = PROJECT_ROOT / "outputs" / ds / "mouth_motion" / f"session_{sess}" / "frame_features.csv"
            missing = [name for name, p in [("gaze_target", gt), ("openface_cam2", of), ("mouth_motion", mm)] if not p.exists()]
            tracked = sess in DATASET_TRACKED_SESSIONS.get(ds, set())
            status = "OK" if not missing else f"MISSING: {missing}"
            print(f"  {ds} s{sess} (tracked={tracked}): {status}")
            if missing:
                findings.append(f"{ds} s{sess}: missing raw source file(s) {missing}")

    section("2. Loading built .pt datasets")
    data = {}
    for ds in DATASETS:
        path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{ds}_continuous.pt"
        graphs = torch.load(path, weights_only=False)
        data[ds] = graphs
        sessions = sorted({g["session"] for g in graphs})
        print(f"  {ds}: {len(graphs)} graphs, sessions {sessions}")
    all_graphs = [g for ds in DATASETS for g in data[ds]]
    print(f"  POOLED TOTAL: {len(all_graphs)} graphs")

    section("3. NaN / Inf check across all node and edge features")
    n_nan_node, n_inf_node, n_nan_edge, n_inf_edge = 0, 0, 0, 0
    for g in all_graphs:
        nf = g["node_features"]
        ef = g["edge_features"]
        if torch.isnan(nf).any():
            n_nan_node += 1
        if torch.isinf(nf).any():
            n_inf_node += 1
        if torch.isnan(ef).any():
            n_nan_edge += 1
        if torch.isinf(ef).any():
            n_inf_edge += 1
    print(f"  graphs with NaN in node_features: {n_nan_node}")
    print(f"  graphs with Inf in node_features: {n_inf_node}")
    print(f"  graphs with NaN in edge_features: {n_nan_edge}")
    print(f"  graphs with Inf in edge_features: {n_inf_edge}")
    if n_nan_node or n_inf_node or n_nan_edge or n_inf_edge:
        findings.append("NaN/Inf values present in node or edge features -- see above counts")

    section("4. of_availability (OpenFace match rate) per session")
    for ds in DATASETS:
        sessions = sorted({g["session"] for g in data[ds]})
        for sess in sessions:
            items = [g for g in data[ds] if g["session"] == sess]
            # node_features is (n_nodes, WINDOW_SIZE, 29) since the 2026-07-30
            # windowing change. feats[:, 8] indexed WINDOW STEP 8 and averaged all
            # 29 features there -- a meaningless number that read as 0.27-0.62 and
            # was reported for months as "facial features largely missing" across
            # 10 sessions. Real coverage is 86-99% nearly everywhere. Take the
            # availability flag at the CURRENT frame (last window step).
            feats = torch.cat([it["node_features"][:, -1, :] for it in items], dim=0)
            rate = feats[:, 8].mean().item()
            flag = "  <-- LOW" if rate < 0.75 else ""
            print(f"  {ds} s{sess}: of_availability={rate:.3f}{flag}")
            if rate < 0.75:
                findings.append(f"{ds} s{sess}: of_availability={rate:.3f} (facial features sparse for this session)")

    section("5. mouth_motion join coverage per session (no availability bit -- checked via raw join)")
    for ds in DATASETS:
        sessions = sorted({g["session"] for g in data[ds]})
        for sess in sessions:
            mm_path = PROJECT_ROOT / "outputs" / ds / "mouth_motion" / f"session_{sess}" / "frame_features.csv"
            gt_path = PROJECT_ROOT / "outputs" / ds / "gaze_target" / f"session_{sess}" / "frame_features.csv"
            if not mm_path.exists():
                print(f"  {ds} s{sess}: mouth_motion file MISSING")
                continue
            mm = pd.read_csv(mm_path)
            gt = pd.read_csv(gt_path)
            mm_keys = set(zip(mm["frame_index"], mm["person_idx"]))
            # sample gt rows the same way build_continuous_dataset does -- fps-aware,
            # since 05_14 is 60fps and uses step 12 where 30fps dates use 6. A fixed
            # 6 here would compare against frames the extractor never sampled and
            # report a spuriously low join rate.
            import json as _json
            _meta = PROJECT_ROOT / "outputs" / ds / "gaze_target" / f"session_{sess}" / "session_metadata.json"
            _fps = float(_json.loads(_meta.read_text())["stream"]["fps"]) if _meta.exists() else 30.0
            frames = sorted(gt["frame_index"].unique())[::frame_step_for(_fps)]
            gt_sub = gt[gt["frame_index"].isin(frames)].dropna(subset=["person_idx"])
            gt_keys = set(zip(gt_sub["frame_index"], gt_sub["person_idx"]))
            hit = len(gt_keys & mm_keys)
            total = len(gt_keys)
            rate = hit / total if total else 0.0
            flag = "  <-- LOW" if rate < 0.5 else ""
            print(f"  {ds} s{sess}: mouth_motion join rate={rate:.3f} ({hit}/{total}){flag}")
            if rate < 0.5:
                findings.append(f"{ds} s{sess}: mouth_motion join rate={rate:.3f} (likely broken)")

    section("6. Per-dataset feature scale sanity (raw, pre-normalization) -- all 29 dims, current-frame step only")
    for ds in DATASETS:
        # node_features is (N, WINDOW_SIZE, 29) since the 2026-07-30 GRU windowing
        # change -- use the last window step (the current frame) for a scale
        # snapshot, not blended with up to 3s of history.
        feats = torch.cat([g["node_features"][:, -1, :] for g in data[ds]], dim=0)
        means = feats.mean(0)
        stds = feats.std(0)
        print(f"  {ds}:")
        for i, name in enumerate(DIM_NAMES):
            print(f"    {name:16s} mean={means[i].item():7.3f} std={stds[i].item():7.3f}")

    section("7. Cross-date AU scale comparison (the bug fixed 2026-07-28 -- re-verify still sane)")
    au_idx = list(range(12, 20))
    for ds in DATASETS:
        feats = torch.cat([g["node_features"][:, -1, :] for g in data[ds]], dim=0)
        au_mean = feats[:, au_idx].mean().item()
        print(f"  {ds}: mean AU activation = {au_mean:.4f}")

    section("8. Label distribution per session")
    for ds in DATASETS:
        sessions = sorted({g["session"] for g in data[ds]})
        for sess in sessions:
            items = [g for g in data[ds] if g["session"] == sess]
            labels = torch.cat([it["labels"] for it in items])
            print(f"  {ds} s{sess}: n={len(labels)} mean={labels.mean():.1f} std={labels.std():.1f} "
                  f"min={labels.min():.0f} max={labels.max():.0f}")

    section("9. Node count / graph count sanity")
    for ds in DATASETS:
        node_counts = [g["node_features"].shape[0] for g in data[ds]]
        print(f"  {ds}: node count distribution {np.bincount(node_counts).tolist()}")

    section("10. Edge feature sanity (5 channels: gaze_containment, proximity, mutual_gaze, attending_to_speaker, neighbor_engagement_lagged)")
    edge_names = ["gaze_containment", "proximity", "mutual_gaze", "attending_to_speaker", "neighbor_engagement_lagged"]
    for ds in DATASETS:
        multi_node = [g for g in data[ds] if g["edge_features"].shape[0] > 1]
        if not multi_node:
            print(f"  {ds}: no multi-node graphs to check edges")
            continue
        all_edges = torch.cat([g["edge_features"].reshape(-1, 5) for g in multi_node], dim=0)
        print(f"  {ds}:")
        for i, name in enumerate(edge_names):
            zero_rate = (all_edges[:, i] == 0).float().mean().item()
            print(f"    {name:24s} mean={all_edges[:, i].mean().item():.3f} zero_rate={zero_rate:.3f}")

    section("11. Duplicate-graph check (same dataset/session/frame_index appearing more than once)")
    for ds in DATASETS:
        keys = [(g["dataset"], g["session"], g["frame_index"]) for g in data[ds]]
        n_dup = len(keys) - len(set(keys))
        print(f"  {ds}: {n_dup} duplicate (dataset,session,frame_index) keys")
        if n_dup:
            findings.append(f"{ds}: {n_dup} duplicate graph keys found")

    section("FINDINGS SUMMARY")
    if findings:
        for f in findings:
            print(f"  [ISSUE] {f}")
    else:
        print("  No issues flagged by automated checks above.")


if __name__ == "__main__":
    main()
