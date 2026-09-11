"""Extracts a per-frame, per-person mouth-motion signal from Cam_2 footage,
used as a visual (not audio-dependent) "is this person currently talking"
proxy for the new attending-to-speaker graph edge.

Why visual, not audio: the per-speaker separated audio channels (A/B/C/R)
were found to have real, unpredictable crosstalk (cross-channel correlation
spikes at different, session-specific moments -- not a fixable fixed
window), and even a stronger audio-visual identity-linking attempt (mouth-
AU-change correlation) never got a confident channel-to-position mapping.
Mouth-region landmark motion sidesteps this entirely: it's computed
straight from Cam_2, matched to the same local_id/position already used
for gaze/labels, no cross-camera or cross-channel identity problem at all.

Uses only the landmark detector (not the slower multitask AU/emotion/gaze
model) since only motion magnitude is needed here, not full facial state --
keeps this fast enough to run at native frame density.

Output: outputs/<dataset>/mouth_motion/session_N/frame_features.csv, one
row per (frame, person_idx): mouth_motion (mean frame-to-frame displacement
of WFLW landmarks 76-95, the mouth region).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Same thread-oversubscription fix as run_batch_openface_cam2.py (2026-07-28/29):
# single-image CPU inference one frame at a time doesn't benefit from torch's
# default all-core intra-op parallelism -- it just adds thread sync overhead.
torch.set_num_threads(4)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENFACE_DIR = PROJECT_ROOT / "dataExtraction/openface3"
sys.path.insert(0, str(PROJECT_ROOT / "datasetPrep"))

from render_review_clips import open_cam2_capture  # noqa: E402
DATASET = "03_20"
# Sampling cadence, in Hz, not in frames -- 05_14 is 60fps while every other
# date is 30, so a fixed frame step would sample it twice as densely (and
# take twice as long) for no benefit. 5Hz reproduces the old FRAME_STEP=6 on
# 30fps footage exactly, so nothing changes for existing dates.
TARGET_SAMPLE_HZ = 5.0


def frame_step_for(fps: float) -> int:
    return max(1, int(round(fps / TARGET_SAMPLE_HZ)))

sys.path.insert(0, str(PROJECT_ROOT / "modelTraining"))


def process_session(session: int, face_detector, landmark_detector) -> pd.DataFrame:
    # Camera-folder override (05_15/05_14 have swapped names) and multi-part
    # .avi handling both come from render_review_clips now. This file used to
    # keep its own copy of the override dict with a "keep both in sync" note;
    # they promptly drifted (05_14 was added to one and not the other), so
    # there is deliberately only one copy left.
    cap = open_cam2_capture(DATASET, session)
    if cap is None:
        print(f"  session {session}: no Cam_2 video, skipping")
        return pd.DataFrame()
    if len(cap.paths) > 1:
        print(f"  session {session}: {len(cap.paths)} video parts, treated as one continuous stream")

    gt_path = PROJECT_ROOT / "outputs" / DATASET / "gaze_target" / f"session_{session}" / "frame_features.csv"
    gt_df = pd.read_csv(gt_path)

    queue = pd.read_csv(PROJECT_ROOT / "outputs" / DATASET / "manual_queue.csv", dtype=str, keep_default_na=False)
    seg_rows = queue[(queue.session == str(session)) & (queue.source == "cam2")]
    frame_step = frame_step_for(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tmp_path = f"/tmp/_mouth_motion_frame_{session}.png"
    rows = []

    for _, q in seg_rows.iterrows():
        start_sec, end_sec = float(q["segment_start_sec"]), float(q["segment_end_sec"])
        seg_frames_all = gt_df[(gt_df.timestamp_sec >= start_sec) & (gt_df.timestamp_sec < end_sec)]["frame_index"].unique()
        frame_indices = sorted(seg_frames_all)[::frame_step]

        prev_by_person: dict[int, tuple[np.ndarray, int]] = {}  # person_idx -> (mouth_landmarks, frame_idx)
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(tmp_path, frame)
            _, dets = face_detector.get_face(tmp_path)
            if dets is None:
                continue
            dets = np.array(dets)
            confident = dets[dets[:, 4] >= 0.5]
            if len(confident) == 0:
                continue
            try:
                landmarks = landmark_detector.detect_landmarks(frame, confident)
            except Exception:
                continue

            frame_rows = gt_df[(gt_df.frame_index == fi)].dropna(subset=["person_idx"])
            for det, lm in zip(confident, landmarks or []):
                if lm is None:
                    continue
                x0, y0, x1, y1 = det[:4]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                best_pid, best_dist = None, 1e9
                for _, fr in frame_rows.iterrows():
                    fcx, fcy = (fr.bbox_x0 + fr.bbox_x1) / 2, (fr.bbox_y0 + fr.bbox_y1) / 2
                    d = np.hypot(cx - fcx, cy - fcy)
                    if d < best_dist:
                        best_dist, best_pid = d, fr.person_idx
                if best_pid is None or best_dist > 100:
                    continue

                mouth_lm = np.array(lm).reshape(-1, 2)[76:96]
                if best_pid in prev_by_person:
                    prev_lm, prev_fi = prev_by_person[best_pid]
                    frame_gap = fi - prev_fi
                    if frame_gap > 0:
                        motion = np.linalg.norm(mouth_lm - prev_lm, axis=1).mean() / frame_gap
                        rows.append({"frame_index": int(fi), "person_idx": best_pid, "mouth_motion": float(motion)})
                prev_by_person[best_pid] = (mouth_lm, fi)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    os.chdir(OPENFACE_DIR)
    from openface.face_detection import FaceDetector
    from openface.landmark_detection import LandmarkDetector

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    DATASET = args.dataset  # noqa: F811 -- process_session reads this module global

    face_detector = FaceDetector(model_path="./weights/Alignment_RetinaFace.pth", device="cpu")
    landmark_detector = LandmarkDetector(model_path="./weights/Landmark_98.pkl", device="cpu")

    for session in args.sessions:
        print(f"\n=== session {session} ===")
        df = process_session(session, face_detector, landmark_detector)
        out_dir = PROJECT_ROOT / "outputs" / DATASET / "mouth_motion" / f"session_{session}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "frame_features.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> {len(df)} rows saved to {out_path.relative_to(PROJECT_ROOT)}")
