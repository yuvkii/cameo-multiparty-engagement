"""Render one clip per auto-labeled cam2 segment, with each detected
person's bbox drawn and colored/labeled by the auto-labeler's predicted
engagement class -- so the predictions can actually be eyeballed against
the footage before being trusted for anything beyond training-only signal
(see auto_label_cam2.py's docstring for why: kappa=0.445 against real
human labels on 03_20, "moderate" agreement, not ground-truth quality).

Re-derives reidentify_segment + build_cam2_node_features + the classifier
prediction fresh per segment (same code path as auto_label_cam2.py) rather
than reading graph_dataset_<dataset>.pt back, since the saved graphs only
keep aggregated per-node feature vectors -- not the per-frame bbox
trajectory needed to draw a moving box. Same reidentification code, same
inputs -> same local_id assignment and same predictions, deterministically.

Output: outputs/<dataset>/auto_review_clips/session{S}_seg{M}.mp4 +
outputs/<dataset>/auto_review_manifest.csv (one row per segment, listing
its clip path and the predicted label/confidence per person) for
review_auto_labels.py to drive the interactive viewer from.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasetPrep"))
from render_review_clips import _find_cam2_raw  # noqa: E402

from assemble_dataset import MIN_FRAMES_CAM2  # noqa: E402
from auto_label_cam2 import fit_classifier  # noqa: E402
from build_graph_dataset import (  # noqa: E402
    CAM2_NODE_FEATURE_COLS,
    build_cam2_node_features,
    reidentify_segment,
)

LEVELS = ["disengaged", "low", "medium", "high"]
# red -> orange -> yellow -> green, BGR (cv2 convention)
LEVEL_COLORS = {
    "disengaged": (0, 0, 220),
    "low": (0, 128, 255),
    "medium": (0, 220, 220),
    "high": (0, 200, 0),
}


def render_labeled_clip(video_path: Path, start_sec: float, end_sec: float, output_path: Path,
                         seg_df: pd.DataFrame, local_id_labels: dict[int, tuple[str, float]]) -> bool:
    """seg_df: reidentified per-frame rows (frame_index, local_id, bbox_*).
    local_id_labels: {local_id: (predicted_level, confidence)}, only for
    local_ids that survived the detection floor and got predicted."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames, int(end_sec * fps))
    if start_frame >= end_frame:
        cap.release()
        return False

    by_frame: dict[int, list[tuple[int, tuple]]] = {}
    for row in seg_df.itertuples():
        lid = int(row.local_id)
        if lid not in local_id_labels:
            continue
        by_frame.setdefault(row.frame_index, []).append(
            (lid, (row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1))
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = start_frame
    wrote_any = False
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        for lid, box in by_frame.get(frame_idx, []):
            level, conf = local_id_labels[lid]
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            color = LEVEL_COLORS[level]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = f"P{lid}: {level} ({conf:.2f})"
            cv2.putText(frame, label, (x0, max(15, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        writer.write(frame)
        wrote_any = True
        frame_idx += 1

    cap.release()
    writer.release()
    return wrote_any


def build(dataset: str, train_on: str = "03_20") -> pd.DataFrame:
    clf = fit_classifier(train_on)
    segments = pd.read_parquet(PROJECT_ROOT / "outputs" / dataset / "segments.parquet")
    seg_bounds = (
        segments[["session", "segment_idx", "segment_start_sec", "segment_end_sec"]]
        .drop_duplicates(subset=["session", "segment_idx"])
        .sort_values(["session", "segment_idx"])
    )

    clips_dir = PROJECT_ROOT / "outputs" / dataset / "auto_review_clips"
    manifest_path = PROJECT_ROOT / "outputs" / dataset / "auto_review_manifest.csv"

    gt_cache: dict[int, pd.DataFrame] = {}
    video_cache: dict[int, Path] = {}
    rows = []

    for row in seg_bounds.itertuples():
        session, segment_idx = int(row.session), int(row.segment_idx)
        gt_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
        if not gt_path.exists():
            continue
        if session not in gt_cache:
            gt_cache[session] = pd.read_csv(gt_path)
        if session not in video_cache:
            video_cache[session] = _find_cam2_raw(dataset, session)
        video_path = video_cache[session]
        if video_path is None:
            continue

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

        local_id_labels = {
            int(ranked.loc[i, "local_id"]): (LEVELS[preds[i]], float(conf[i]))
            for i in range(len(ranked))
        }

        clip_name = f"session{session}_seg{segment_idx}.mp4"
        clip_path = clips_dir / clip_name
        rendered = render_labeled_clip(
            video_path, row.segment_start_sec, row.segment_end_sec, clip_path, seg, local_id_labels
        )
        if not rendered:
            continue

        ordered = sorted(local_id_labels.items(), key=lambda kv: kv[0])
        rows.append({
            "dataset": dataset, "session": session, "segment_idx": segment_idx,
            "clip_path": str(clip_path.relative_to(PROJECT_ROOT)),
            "n_people": len(ordered),
            "predicted_labels": ";".join(f"P{lid}={lvl}({c:.2f})" for lid, (lvl, c) in ordered),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-on", default="03_20")
    args = parser.parse_args()

    manifest = build(args.dataset, args.train_on)
    print(f"{len(manifest)} review clips -> outputs/{args.dataset}/auto_review_clips/")
    print(f"manifest -> outputs/{args.dataset}/auto_review_manifest.csv")
