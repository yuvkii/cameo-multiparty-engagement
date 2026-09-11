"""Auto-label 03_20's auxiliary mocap camera views (Cam_1/3/4/5/6 -- every
angle besides the main Cam_2) and combine them into one extra training-only
graph set, saved separately from graph_dataset_03_20.pt so the real
manually-labeled data is never at risk of being overwritten.

Why this exists: Cam_2 alone is structurally blind to anyone whose back is
turned to it (confirmed by inspecting raw frames, 2026-07-22) -- in this
room's circular group conversations, that's routinely half the participants
at any moment. The other 5 cameras see different subsets of faces. Person
identity is NOT matched across cameras (that needs real 3D correspondence,
which this rig's calibration data can't support -- see
[[camera3_robot_detection_pipeline]] memory) -- each camera's segments are
auto-labeled independently via auto_label_cam2's classifier, tagged with
"camera" for traceability, same treatment as any other auto-labeled data.

All output graphs carry dataset="03_20" (same date, just a different
camera), so train.py's leave_one_session_out correctly excludes them from
ever being the held-out test set for a given session while still using
them as training signal for every OTHER session's fold -- see the
2026-07-22 fix in train.py for why session-level exclusion (not just
manual/auto exclusion) matters once a session has mixed manual+auto data.
"""
from __future__ import annotations

from pathlib import Path

import torch

from auto_label_cam2 import auto_label_dataset, fit_classifier

PROJECT_ROOT = Path(__file__).parent.parent
AUX_CAMERAS = [1, 3, 4, 5, 6]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    parser.add_argument("--train-on", default="03_20")
    parser.add_argument("--cameras", type=int, nargs="+", default=AUX_CAMERAS)
    args = parser.parse_args()

    print(f"fitting classifier on {args.train_on} (manual cam2 labels only)...")
    clf = fit_classifier(args.train_on)

    all_graphs = []
    for cam in args.cameras:
        subdir = f"gaze_target_cam{cam}"
        path_check = PROJECT_ROOT / "outputs" / args.dataset / subdir
        if not path_check.is_dir():
            print(f"cam{cam}: no extraction found at {path_check}, skipping")
            continue
        print(f"\n=== cam{cam} ===")
        graphs = auto_label_dataset(args.dataset, clf, gaze_target_subdir=subdir, camera_tag=cam)
        print(f"cam{cam}: {len(graphs)} graphs")
        all_graphs.extend(graphs)

    out_path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{args.dataset}_auxcams.pt"
    torch.save(all_graphs, out_path)
    sizes = [g["node_features"].shape[0] for g in all_graphs]
    import pandas as pd
    print(f"\n{len(all_graphs)} total graphs -> {out_path}")
    print(f"by camera: {pd.Series([g['camera'] for g in all_graphs]).value_counts().sort_index().to_dict()}")
    print(f"node counts: {pd.Series(sizes).value_counts().sort_index().to_dict()}")
