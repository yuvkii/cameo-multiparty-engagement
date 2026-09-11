"""Gaze-target extraction pipeline for the wide-angle 6cams mocap rig
(Cam_2, Cam_3, ...): detects every head in frame (participants + the
robot-person, who always faces away from these cameras), identifies which
one is the robot-person, and uses Gaze-LLE to determine whether each
participant is looking at the robot — entirely from pixels, no camera
calibration needed.

Output format mirrors the main pipeline.py family: frame_features.csv +
session_metadata.json (+ optional debug_overlay.mp4) per session, so
downstream analysis can treat all pipelines' outputs uniformly.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .gaze_target_model import GazeTargetModel
from .head_detector import Detection, HeadDetector

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "crowdhuman_yolov5m.pt"

TRACK_COLORS = (
    (83, 214, 110),
    (74, 163, 255),
    (255, 173, 82),
    (200, 200, 80),
)
ROBOT_BOX_COLOR = (255, 255, 255)
PARTICIPANT_GAZE_COLOR = (255, 0, 255)


@dataclass(slots=True)
class PipelineConfig:
    video_path: Path
    output_dir: Path
    head_detector_weights_path: Path = _DEFAULT_WEIGHTS_PATH
    head_detector_conf_thres: float = 0.3
    # The robot-person stands much closer to the camera than participants do
    # (validated on camera 2/3 footage), so their detected head is reliably
    # the largest in frame. Require it to be clearly (not marginally) larger
    # than the next-largest head before trusting the identification.
    robot_size_ratio_threshold: float = 1.5
    frame_step: int = 1
    start_frame: int = 0
    max_frames: Optional[int] = None
    save_debug_video: bool = True
    # When True, a gaze peak that lands inside another participant's bbox
    # (instead of the robot's) is classified "participant" rather than
    # collapsing into "elsewhere". Set False to restore the prior binary
    # robot/elsewhere-only behavior if this misclassifies in practice.
    classify_other_participants: bool = True
    # The robot's detected head bbox is a tight back-of-head crop, and
    # Gaze-LLE's predicted peak has enough natural jitter that genuine
    # robot-directed gaze often lands just outside it rather than inside.
    # Validated on Data/05_15 session 4: of "elsewhere" rows whose peak
    # already vertically aligned with the robot, 93% were within 20px of
    # the robot box horizontally (median 13px) -- a systematic near-miss,
    # not scattered error. This pads the robot box by N px on every side
    # before the containment check to absorb that jitter. Set to 0 to
    # restore the exact prior (unpadded) behavior.
    robot_gaze_margin_px: int = 20
    # "cuda" moves the head detector and Gaze-LLE onto the GPU; see
    # GazeTargetModel.__init__ for the measured speedup. Default "cpu"
    # keeps prior behaviour for every existing caller.
    device: str = "cpu"


def bbox_area(box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def point_in_bbox(px: int, py: int, box: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def expand_bbox(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def pick_robot(
    detections: list[Detection], size_ratio_threshold: float
) -> tuple[Optional[Detection], list[Detection]]:
    """Split detections into (robot, participants). Returns (None, detections)
    if there's fewer than 2 heads or the largest isn't clearly bigger than
    the next, since we can't confidently tell robot from participant then."""
    if len(detections) < 2:
        return None, detections

    by_area = sorted(detections, key=lambda d: bbox_area(d.bbox_xyxy), reverse=True)
    largest, second = by_area[0], by_area[1]
    if bbox_area(largest.bbox_xyxy) / max(1.0, bbox_area(second.bbox_xyxy)) < size_ratio_threshold:
        return None, detections

    participants = [d for d in detections if d is not largest]
    return largest, participants


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    frame_index: int,
    robot_det: Optional[Detection],
    participant_rows: list[dict],
) -> np.ndarray:
    cv2.putText(
        frame_bgr, f"frame {frame_index}", (10, frame_bgr.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    if robot_det is not None:
        rx0, ry0, rx1, ry1 = robot_det.bbox_xyxy
        cv2.rectangle(frame_bgr, (rx0, ry0), (rx1, ry1), ROBOT_BOX_COLOR, 2)
        cv2.putText(frame_bgr, "ROBOT", (rx0, max(20, ry0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROBOT_BOX_COLOR, 2)

    for i, row in enumerate(participant_rows):
        x0, y0, x1, y1 = row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]
        if row["gaze_target"] == "robot":
            color = (0, 255, 255)
        elif row["gaze_target"] == "participant":
            color = PARTICIPANT_GAZE_COLOR
        else:
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.arrowedLine(frame_bgr, (cx, cy), (row["gaze_peak_x"], row["gaze_peak_y"]), color, 2, tipLength=0.15)
        label = f"gaze:{row['gaze_target']}"
        if row["inout_score"] is not None:
            label += f" io={row['inout_score']:.2f}"
        cv2.putText(frame_bgr, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame_bgr


CSV_FIELDNAMES = [
    "frame_index",
    "timestamp_sec",
    "robot_detected",
    "robot_x0",
    "robot_y0",
    "robot_x1",
    "robot_y1",
    "person_idx",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "gaze_target",
    "gaze_peak_x",
    "gaze_peak_y",
    "inout_score",
]


def run_feature_pipeline(config: PipelineConfig) -> dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    head_detector = HeadDetector(
        weights_path=config.head_detector_weights_path, conf_thres=config.head_detector_conf_thres,
        device=config.device,
    )
    gaze_model = GazeTargetModel(device=config.device)

    csv_path = output_dir / "frame_features.csv"
    metadata_path = output_dir / "session_metadata.json"
    debug_video_path = output_dir / "debug_overlay.mp4"

    video_writer = None
    if config.save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(debug_video_path), fourcc, fps, (w, h))

    processed_frames = 0
    frames_with_confident_robot = 0
    total_participant_rows = 0
    looking_at_robot_rows = 0
    looking_at_participant_rows = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        frame_idx = config.start_frame
        while config.max_frames is None or processed_frames < config.max_frames:
            for _ in range(config.frame_step):
                ret, frame_bgr = cap.read()
                if not ret:
                    break
            if not ret:
                break
            processed_frames += 1
            timestamp_sec = frame_idx / fps

            detections = head_detector.detect_heads(frame_bgr)
            robot_det, participants = pick_robot(detections, config.robot_size_ratio_threshold)

            participant_rows: list[dict] = []
            if robot_det is not None and participants:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gaze_results = gaze_model.detect_targets(frame_rgb, [p.bbox_xyxy for p in participants])
                rx0, ry0, rx1, ry1 = robot_det.bbox_xyxy
                robot_check_box = expand_bbox(robot_det.bbox_xyxy, config.robot_gaze_margin_px)

                for i, (det, result) in enumerate(zip(participants, gaze_results)):
                    px, py = result.peak_xy
                    looking_at_robot = point_in_bbox(px, py, robot_check_box)

                    gaze_target = "robot" if looking_at_robot else "elsewhere"
                    if not looking_at_robot and config.classify_other_participants:
                        for j, other in enumerate(participants):
                            if j != i and point_in_bbox(px, py, other.bbox_xyxy):
                                gaze_target = "participant"
                                break

                    row = {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 6),
                        "robot_detected": 1,
                        "robot_x0": rx0, "robot_y0": ry0, "robot_x1": rx1, "robot_y1": ry1,
                        "person_idx": len(participant_rows),
                        "bbox_x0": det.bbox_xyxy[0], "bbox_y0": det.bbox_xyxy[1],
                        "bbox_x1": det.bbox_xyxy[2], "bbox_y1": det.bbox_xyxy[3],
                        "gaze_target": gaze_target,
                        "gaze_peak_x": px,
                        "gaze_peak_y": py,
                        "inout_score": result.inout_score,
                    }
                    participant_rows.append(row)
                    writer.writerow(row)

                frames_with_confident_robot += 1
                total_participant_rows += len(participant_rows)
                looking_at_robot_rows += sum(1 for r in participant_rows if r["gaze_target"] == "robot")
                looking_at_participant_rows += sum(1 for r in participant_rows if r["gaze_target"] == "participant")
            else:
                writer.writerow({
                    "frame_index": frame_idx, "timestamp_sec": round(timestamp_sec, 6),
                    "robot_detected": 0, "robot_x0": "", "robot_y0": "", "robot_x1": "", "robot_y1": "",
                    "person_idx": "", "bbox_x0": "", "bbox_y0": "", "bbox_x1": "", "bbox_y1": "",
                    "gaze_target": "unknown", "gaze_peak_x": "", "gaze_peak_y": "", "inout_score": "",
                })

            if video_writer is not None:
                overlay = draw_debug_overlay(frame_bgr.copy(), frame_idx, robot_det, participant_rows)
                video_writer.write(overlay)

            frame_idx += config.frame_step

    cap.release()
    if video_writer is not None:
        video_writer.release()

    summary = {
        "input": {"video_path": str(config.video_path)},
        "stream": {"total_frames_in_video": total_frames_in_video, "fps": fps, "frame_width": w, "frame_height": h},
        "outputs": {
            "frame_features_csv": str(csv_path),
            "debug_video": str(debug_video_path) if config.save_debug_video else None,
        },
        "summary": {
            "processed_frames": processed_frames,
            "frames_with_confident_robot": frames_with_confident_robot,
            "robot_detection_rate": (frames_with_confident_robot / processed_frames) if processed_frames else 0.0,
            "participant_frames_total": total_participant_rows,
            "looking_at_robot": looking_at_robot_rows,
            "looking_at_robot_rate": (looking_at_robot_rows / total_participant_rows) if total_participant_rows else 0.0,
            "looking_at_participant": looking_at_participant_rows,
            "looking_at_participant_rate": (looking_at_participant_rows / total_participant_rows) if total_participant_rows else 0.0,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
