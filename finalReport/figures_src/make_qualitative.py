"""Qualitative figure: one sampled frame with the graph the model actually
consumes drawn on top of it (participant nodes, robot, gaze rays, edges).

Faces are blurred by default -- set BLUR = False to produce an unblurred
version if the recording consent covers publication of identifiable frames.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cameo_stats import ROOT, MT
from style import FULL, SERIES, BLUE, ORANGE, AQUA, INK, INK2, MUTED, save

BLUR = True
DATE, SESSION = "05_15", 2


def main():
    meta = json.loads((ROOT / "outputs" / DATE / "gaze_target" / f"session_{SESSION}" /
                       "session_metadata.json").read_text())
    video = ROOT / meta["input"]["video_path"]
    data = torch.load(MT / f"graph_dataset_{DATE}_continuous.pt", weights_only=False)
    items = [it for it in data if it["session"] == SESSION]
    # a frame where at least one participant looks at a peer, so an edge is live
    gt = pd.read_csv(ROOT / "outputs" / DATE / "gaze_target" / f"session_{SESSION}" / "frame_features.csv")
    pick = None
    for it in items:
        if (it["edge_features"][:, :, 0].sum() > 0 and len(it["labels"]) == 3
                and it["timestamp_sec"] > 120 and float(it["labels"].max()) > 60
                and float(it["labels"].min()) < 40):
            pick = it
            break
    assert pick is not None
    fi = pick["frame_index"]

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame {fi} of {video}"
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rows = gt[gt.frame_index == fi]
    robot = rows.iloc[0][["robot_x0", "robot_y0", "robot_x1", "robot_y1"]].to_numpy(dtype=float)

    if BLUR:
        for _, r in rows.iterrows():
            x0, y0, x1, y1 = [int(v) for v in (r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1)]
            h = max(1, (y1 - y0)); w = max(1, (x1 - x0))
            roi = img[y0:y0 + h, x0:x0 + w]
            if roi.size:
                k = max(9, (min(h, w) // 2) * 2 + 1)
                img[y0:y0 + h, x0:x0 + w] = cv2.GaussianBlur(roi, (k, k), 0)
        x0, y0, x1, y1 = [int(v) for v in robot]
        roi = img[y0:y1, x0:x1]
        if roi.size:
            k = max(9, (min(y1 - y0, x1 - x0) // 2) * 2 + 1)
            img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)

    fig, ax = plt.subplots(figsize=(FULL, FULL * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(Rectangle((robot[0], robot[1]), robot[2] - robot[0], robot[3] - robot[1],
                           fill=False, edgecolor=INK, lw=1.8, linestyle=(0, (4, 2))))
    ax.text((robot[0] + robot[2]) / 2, robot[1] - 14, "robot role (confederate)", color=INK,
            fontsize=7.5, weight="bold", ha="center",
            bbox=dict(facecolor="white", alpha=0.75, pad=1.2, edgecolor="none"))

    centres = {}
    for n, (pidx, part) in enumerate(zip(pick["person_idxs"], pick["participants"])):
        r = rows[rows.person_idx == pidx].iloc[0]
        x0, y0, x1, y1 = float(r.bbox_x0), float(r.bbox_y0), float(r.bbox_x1), float(r.bbox_y1)
        col = SERIES[n]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=col, lw=2.0))
        ax.text(x0 + (x1 - x0) / 2, y0 - 14 - 26 * (n % 2), f"{part}: {float(pick['labels'][n]):.0f}",
                color=col, fontsize=8, weight="bold", ha="center",
                bbox=dict(facecolor="white", alpha=0.75, pad=1.2, edgecolor="none"))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        centres[n] = (cx, cy)
        if not np.isnan(r.gaze_peak_x):
            ax.add_patch(FancyArrowPatch((cx, cy), (float(r.gaze_peak_x), float(r.gaze_peak_y)),
                                         arrowstyle="-|>", mutation_scale=9, color=col, lw=1.4,
                                         linestyle=(0, (2, 1.6)), alpha=0.95))
            ax.plot([float(r.gaze_peak_x)], [float(r.gaze_peak_y)], marker="x", ms=6, color=col, mew=1.6)

    e = pick["edge_features"]
    for i in range(len(centres)):
        for j in range(len(centres)):
            if i == j or e[i, j, 0] <= 0:
                continue
            ax.add_patch(FancyArrowPatch(centres[i], centres[j], arrowstyle="-|>", mutation_scale=11,
                                         color=ORANGE, lw=2.2, connectionstyle="arc3,rad=0.18"))
    ax.text(0.005, 0.96, f"{DATE} session {SESSION}, {pick['timestamp_sec']:.0f} s"
            + ("  |  faces blurred" if BLUR else "") + "  |  labels shown are the annotated engagement values",
            transform=ax.transAxes, fontsize=7, color="white", va="top",
            bbox=dict(facecolor=INK, alpha=0.6, pad=2, edgecolor="none"))
    ax.text(0.995, 0.02, "dashed arrow: estimated gaze ray and its target point\n"
                          "solid orange arrow: active gaze-containment edge",
            transform=ax.transAxes, fontsize=7, color="white", va="bottom", ha="right",
            bbox=dict(facecolor=INK, alpha=0.55, pad=3, edgecolor="none"))
    save(fig, "fig_scene")


if __name__ == "__main__":
    main()
