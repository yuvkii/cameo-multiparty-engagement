"""Extract facial landmarks/AU/emotion/gaze from Cam_2 mocap footage using
OpenFace 3.0 (RetinaFace detector + STAR 98-pt landmarks + multitask
AU/emotion/gaze head), matched to the EXISTING gaze_target person_idx via
same-frame bbox proximity.

Why same-frame matching and not cross-camera identity linking: this runs on
the exact same video/frame/pixel space that gaze_target/manual labels
already use (Cam_2), so a detected face just needs to be matched to the
already-tracked person at that instant -- no cross-camera correspondence
problem exists here at all, unlike the separate RGB camera (participants_
rgb.avi), where cross-camera behavioral correlation only found confident
matches for ~1/3 of people per session (see link_cross_camera_identity.py).

Restricted to already-labeled 3s segments (manual_queue.csv, source=='cam2')
rather than whole sessions -- segments already tile virtually the entire
labeled timeline so this isn't a real scope reduction, it just keeps the
job well-defined against what needs new features. FRAME_STEP=6 (vs the RGB
facial_keypoints pipeline's frame_step=3) since OpenFace 3.0's three
sequential model passes per frame are slower (~1.1s/frame measured on this
hardware) and per-segment aggregates (mean AU, dominant emotion) don't need
dense temporal sampling -- ~15 frames/3s-segment is plenty for a stable
aggregate, consistent with the design notes' "engagement doesn't meaningfully
change frame-to-frame" annotation rationale.

Matching a detected face to gaze_target's person_idx: both come from the
SAME frame of the SAME video, so IoU between the OpenFace bbox and each
person_idx's already-recorded bbox (bbox_x0/y0/x1/y1 in gaze_target's own
frame_features.csv) is a direct, reliable match -- no proxy/behavioral
signal needed, unlike the cross-camera case.

Output: outputs/<dataset>/openface_cam2/session_N/frame_features.csv, one
row per (frame, matched person_idx): emotion/AU/gaze plus match_iou for
auditability. Frames/faces that don't match any existing person_idx above
MIN_MATCH_IOU are dropped -- better to have a missing row (handled downstream
via an availability flag) than a wrong pairing.
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Real perf bug found 2026-07-28/29: torch defaults to intra-op parallelism
# across every available core (24 here), but this script runs single-image
# inference through three small CPU models one frame at a time -- thread
# spawn/sync overhead for that many threads on tiny per-frame tensor ops
# dominated over the actual compute, measured via /proc CPU-time sampling as
# ~12 cores busy simultaneously while wall-clock progress was 10x+ slower
# than the ~1.1s/frame this script's own docstring documents. Capping
# threads fixed it -- this is a standard PyTorch CPU-inference pitfall for
# small-batch workloads, not specific to this model.
torch.set_num_threads(4)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENFACE_DIR = PROJECT_ROOT / "dataExtraction/openface3"

sys.path.insert(0, str(OPENFACE_DIR / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(PROJECT_ROOT / "datasetPrep"))

from render_review_clips import open_cam2_capture  # noqa: E402

# Sampling cadence, in Hz, not in frames -- 05_14 is 60fps while every other
# date is 30, so a fixed frame step would sample it twice as densely (and
# take twice as long) for no benefit. 5Hz reproduces the old FRAME_STEP=6 on
# 30fps footage exactly, so nothing changes for existing dates.
TARGET_SAMPLE_HZ = 5.0


def frame_step_for(fps: float) -> int:
    return max(1, int(round(fps / TARGET_SAMPLE_HZ)))
MIN_MATCH_IOU = 0.3
TMP_FRAME_PATH = str(Path(tempfile.gettempdir()) / "_openface_batch_frame.png")


def iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


# The per-date camera-folder override (05_15/05_14 store the true Cam_2 view
# under "Cam_1") used to be duplicated here with a "keep both in sync" note.
# The copies promptly drifted -- 05_14 was added to one and not the other --
# so both the override and multi-part .avi handling now come from
# render_review_clips.open_cam2_capture, which is the single source.


def process_session(dataset: str, session: int, queue: pd.DataFrame, face_detector, landmark_detector, multitask_model) -> pd.DataFrame:
    cap = open_cam2_capture(dataset, session)
    if cap is None:
        print(f"  session {session}: no Cam_2 video found, skipping")
        return pd.DataFrame()
    if len(cap.paths) > 1:
        print(f"  session {session}: {len(cap.paths)} video parts, treated as one continuous stream")

    gt_path = PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}" / "frame_features.csv"
    gt_df = pd.read_csv(gt_path)

    frame_step = frame_step_for(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    seg_rows = queue[(queue.session == str(session)) & (queue.source == "cam2")]
    frame_indices: set[int] = set()
    for _, q in seg_rows.iterrows():
        start_sec, end_sec = float(q["segment_start_sec"]), float(q["segment_end_sec"])
        seg_frames = gt_df[(gt_df.timestamp_sec >= start_sec) & (gt_df.timestamp_sec < end_sec)]["frame_index"].unique()
        seg_frames = sorted(seg_frames)[::frame_step]
        frame_indices.update(int(f) for f in seg_frames)
    frame_indices = sorted(frame_indices)
    print(f"  session {session}: {len(seg_rows)} segments, step={frame_step} -> {len(frame_indices)} sampled frames")

    rows = []
    t0 = time.time()
    for n_done, fi in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(TMP_FRAME_PATH, frame)
        _, dets = face_detector.get_face(TMP_FRAME_PATH)
        if dets is None:
            continue
        dets = np.array(dets)
        confident = dets[dets[:, 4] >= 0.5]
        if len(confident) == 0:
            continue

        gt_frame_rows = gt_df[gt_df.frame_index == fi].dropna(subset=["person_idx"])
        if gt_frame_rows.empty:
            continue

        try:
            landmarks = landmark_detector.detect_landmarks(frame, confident)
        except Exception:
            landmarks = [None] * len(confident)

        for det, _lm in zip(confident, landmarks or [None] * len(confident)):
            x0, y0, x1, y1, conf = det[:5]
            own_crop = frame[int(y0):int(y1), int(x0):int(x1)]
            if own_crop.size == 0:
                continue

            best_person_idx, best_iou = None, 0.0
            for _, gr in gt_frame_rows.iterrows():
                score = iou((x0, y0, x1, y1), (gr.bbox_x0, gr.bbox_y0, gr.bbox_x1, gr.bbox_y1))
                if score > best_iou:
                    best_iou, best_person_idx = score, gr.person_idx
            if best_person_idx is None or best_iou < MIN_MATCH_IOU:
                continue

            try:
                emotion_logits, gaze_output, au_output = multitask_model.predict(own_crop)
                au_vals = np.asarray(au_output.detach().cpu().numpy() if hasattr(au_output, "detach") else au_output).flatten()
                row = {
                    "frame_index": fi, "timestamp_sec": gt_frame_rows.iloc[0]["timestamp_sec"],
                    "person_idx": best_person_idx, "match_iou": round(best_iou, 3),
                    "face_conf": round(float(conf), 3),
                    "emotion_argmax": int(emotion_logits.argmax(-1).item()),
                    "gaze_yaw": float(np.asarray(gaze_output).flatten()[0]),
                    "gaze_pitch": float(np.asarray(gaze_output).flatten()[1]) if np.asarray(gaze_output).size > 1 else np.nan,
                }
                for i, v in enumerate(au_vals):
                    row[f"au_{i}"] = float(v)
                rows.append(row)
            except Exception as e:
                print(f"    frame {fi} face skipped (multitask error: {e})")

        if n_done % 20 == 0 and n_done > 0:
            elapsed = time.time() - t0
            rate = elapsed / n_done
            eta = rate * (len(frame_indices) - n_done)
            print(f"    {n_done}/{len(frame_indices)} frames, {elapsed:.0f}s elapsed, eta {eta:.0f}s")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    from openface.face_detection import FaceDetector
    from openface.landmark_detection import LandmarkDetector
    from openface.multitask_model import MultitaskPredictor

    import os
    os.chdir(OPENFACE_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    face_detector = FaceDetector(model_path="./weights/Alignment_RetinaFace.pth", device="cpu")
    landmark_detector = LandmarkDetector(model_path="./weights/Landmark_98.pkl", device="cpu")
    multitask_model = MultitaskPredictor(model_path="./weights/MTL_backbone.pth", device="cpu")

    dataset = args.dataset
    queue = pd.read_csv(PROJECT_ROOT / "outputs" / dataset / "manual_queue.csv", dtype=str, keep_default_na=False)

    for session in args.sessions:
        print(f"\n=== {dataset} session {session} ===")
        df = process_session(dataset, session, queue, face_detector, landmark_detector, multitask_model)
        out_dir = PROJECT_ROOT / "outputs" / dataset / "openface_cam2" / f"session_{session}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "frame_features.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> {len(df)} rows saved to {out_path.relative_to(PROJECT_ROOT)}")
