"""Run both cam2 and rgb auto-labelers across every target dataset-date,
producing one graph_dataset_<dataset>.pt per date (all label_source="auto").

Unlike the manual pipeline, a session with BOTH facial_keypoints and
gaze_target coverage (05_14 sessions 1-4, 05_15 all sessions) gets
auto-labeled on BOTH branches independently -- two separate graphs per
segment, one rgb-sourced and one cam2-sourced. No cost tradeoff to avoid
this the way there was for manual labeling (that's what made the earlier
dual-source idea too expensive to pursue by hand); here it's free extra
training signal from data already sitting on disk.

classifiers are fit once on 03_20's real hand labels and reused across all
target datasets.
"""
from __future__ import annotations

from pathlib import Path

import torch

from auto_label_cam2 import auto_label_dataset as auto_label_cam2_dataset
from auto_label_cam2 import fit_classifier as fit_cam2_classifier
from auto_label_rgb import auto_label_rgb_dataset, fit_rgb_classifier

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--train-on", default="03_20")
    args = parser.parse_args()

    print(f"fitting cam2 classifier on {args.train_on}...")
    cam2_clf = fit_cam2_classifier(args.train_on)
    print(f"fitting rgb classifier on {args.train_on}...")
    rgb_clf, fk_cols, train_means = fit_rgb_classifier(args.train_on)

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        graphs = []

        cam2_graphs = auto_label_cam2_dataset(dataset, cam2_clf)
        print(f"  cam2: {len(cam2_graphs)} graphs")
        graphs.extend(cam2_graphs)

        rgb_graphs = auto_label_rgb_dataset(dataset, rgb_clf, fk_cols, train_means)
        print(f"  rgb: {len(rgb_graphs)} graphs")
        graphs.extend(rgb_graphs)

        out_path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{dataset}.pt"
        torch.save(graphs, out_path)
        sizes = [g["node_features"].shape[0] for g in graphs]
        print(f"  {len(graphs)} total graphs -> {out_path}")
        print(f"  node counts: {dict(zip(*__import__('numpy').unique(sizes, return_counts=True)))}" if sizes else "  (empty)")
