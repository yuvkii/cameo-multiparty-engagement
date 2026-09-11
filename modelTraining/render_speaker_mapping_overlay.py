"""Renders a short debug clip per session: Cam_2 footage with each tracked
person's position number drawn on their box, plus live subtitles from the
Whisper transcripts (run_batch_whisper_transcribe.py) showing which speaker
(A/B/C/R) is talking and what they said.

Purpose: confirm the position<->speaker mapping by eye. Automated audio-
visual correlation was tried twice (facial-activity, then mouth-landmark
motion) and topped out too weak to trust for the real participants (best
correlation ~0.19-0.3) -- a human watching ~60-90s with this overlay can
resolve the mapping far more reliably and just needs to do it once per
session (not per frame), similar to how the borrowed Furhat speaker
pipeline (run_speaker_pipeline.py) expects a human-filled
speaker_to_track_ids mapping rather than fully automatic assignment.

Reuses reidentify_segment across the whole rendered window (not per 3s
label segment) so a position number stays the same person for the entire
clip, making the visual check easier than if it reset every 3s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "modelTraining"))
from build_graph_dataset import build_cam2_node_features, reidentify_segment

DATASET = "03_20"
CLIP_DURATION_SEC = 90.0
POSITION_COLORS = {1: (60, 220, 60), 2: (255, 150, 40), 3: (220, 80, 220), 4: (80, 80, 255), 5: (0, 220, 220)}


def find_cam2_video(session: int) -> Path:
    session_dir = PROJECT_ROOT / "Data" / DATASET / "6cams" / str(session) / "Cam_2"
    return list(session_dir.glob("*.avi"))[0]


def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 2), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def render_session(session: int):
    video_path = find_cam2_video(session)
    gt_path = PROJECT_ROOT / "outputs" / DATASET / "gaze_target" / f"session_{session}" / "frame_features.csv"
    gt_df = pd.read_csv(gt_path)

    transcript_path = PROJECT_ROOT / "outputs" / DATASET / "transcripts" / f"session_{session}" / "speaker_transcripts.json"
    transcripts = json.loads(transcript_path.read_text()) if transcript_path.exists() else {}

    seg = gt_df[gt_df.timestamp_sec < CLIP_DURATION_SEC].dropna(subset=["person_idx"])
    seg = reidentify_segment(seg)
    feats = build_cam2_node_features(seg)
    ranked = feats.sort_values("center_x_median").reset_index(drop=True)
    position_of_local_id = {int(r.local_id): pos for pos, r in enumerate(ranked.itertuples(), start=1)}

    box_by_frame: dict[int, list] = {}
    for row in seg.itertuples():
        box_by_frame.setdefault(row.frame_index, []).append(
            (position_of_local_id.get(int(row.local_id), 0), row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1)
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = PROJECT_ROOT / "outputs" / DATASET / "speaker_mapping_overlay" / f"session_{session}_overlay.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n_frames = int(CLIP_DURATION_SEC * fps)
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        t = fi / fps
        for pos, x0, y0, x1, y1 in box_by_frame.get(fi, []):
            color = POSITION_COLORS.get(pos, (200, 200, 200))
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 3)
            draw_label(frame, f"position {pos}", int(x0), int(y0) - 8, color)

        active_lines = []
        for spk, segments in transcripts.items():
            for s in segments:
                if s["start"] <= t <= s["end"]:
                    active_lines.append(f"{spk}: {s['text']}")
        if active_lines:
            overlay = frame.copy()
            th = 30 * len(active_lines) + 20
            cv2.rectangle(overlay, (0, h - th), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            for i, line in enumerate(active_lines):
                cv2.putText(frame, line, (15, h - th + 25 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"t={t:.1f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"session {session}: -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    for session in [1, 2, 3, 4]:
        render_session(session)
