from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker

from .rosbag_reader import RosbagImageStream


@dataclass(slots=True)
class VisualizationConfig:
    bag_path: Path
    npz_path: Path
    output_path: Path
    color_topic: str | None = None
    camera_info_topic: str | None = None
    render_stride: int = 1
    max_frames: int | None = None
    start_row: int = 0
    draw_full_mesh: bool = True
    draw_points: bool = True
    point_radius: int = 1
    line_thickness: int = 1
    draw_bbox: bool = True
    draw_labels: bool = True
    only_tracks: Sequence[int] | None = None


TRACK_COLORS = (
    (83, 214, 110),
    (74, 163, 255),
    (255, 173, 82),
)

CONNECTION_GROUPS: tuple[Sequence[mp_face_landmarker.FaceLandmarksConnections.Connection], ...] = (
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
    mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
)


def render_landmark_video(config: VisualizationConfig) -> dict[str, object]:
    data = np.load(config.npz_path)
    frame_indices = data["frame_indices"].astype(np.int32)
    landmarks = data["landmarks"].astype(np.float32)
    bboxes = data["bboxes"].astype(np.float32)

    if config.start_row < 0 or config.start_row >= len(frame_indices):
        raise ValueError("start_row is outside the available NPZ frame range.")

    selected_rows = list(range(config.start_row, len(frame_indices), max(1, config.render_stride)))
    if config.max_frames is not None:
        selected_rows = selected_rows[: config.max_frames]
    if not selected_rows:
        raise ValueError("No rows selected for visualization.")

    target_indices = frame_indices[selected_rows]
    track_filter = set(config.only_tracks) if config.only_tracks is not None else None

    stream = RosbagImageStream(
        bag_path=config.bag_path,
        color_topic=config.color_topic,
        camera_info_topic=config.camera_info_topic,
    )
    stream_meta = stream.inspect_stream()

    if len(target_indices) > 1:
        effective_step = int(np.median(np.diff(target_indices)))
    else:
        effective_step = 1
    output_fps = max(1.0, stream_meta.fps / max(1, effective_step))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(config.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (stream_meta.frame_width, stream_meta.frame_height),
    )

    frame_lookup = {int(frame_indices[row]): row for row in selected_rows}
    rendered = 0

    try:
        for frame in stream.iter_color_frames(
            frame_step=1,
            start_frame=int(target_indices[0]),
        ):
            row = frame_lookup.get(frame.frame_index)
            if row is None:
                if frame.frame_index > int(target_indices[-1]):
                    break
                continue

            overlay = frame.image_bgr.copy()
            draw_header(overlay, frame.frame_index, rendered_row=row)
            for track_id in range(landmarks.shape[1]):
                if track_filter is not None and track_id not in track_filter:
                    continue
                track_landmarks = landmarks[row, track_id]
                if np.isnan(track_landmarks).all():
                    continue
                track_bbox = bboxes[row, track_id]
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
                draw_track_overlay(
                    image=overlay,
                    normalized_landmarks=track_landmarks,
                    bbox=track_bbox,
                    track_id=track_id,
                    color=color,
                    draw_full_mesh=config.draw_full_mesh,
                    draw_points=config.draw_points,
                    point_radius=config.point_radius,
                    line_thickness=config.line_thickness,
                    draw_bbox=config.draw_bbox,
                    draw_labels=config.draw_labels,
                )

            writer.write(overlay)
            rendered += 1
            if rendered >= len(selected_rows):
                break
    finally:
        writer.release()

    return {
        "output_path": str(config.output_path),
        "rendered_frames": rendered,
        "source_rows": len(selected_rows),
        "source_frame_min": int(target_indices[0]),
        "source_frame_max": int(target_indices[-1]),
        "fps": output_fps,
    }


def draw_header(image: np.ndarray, frame_index: int, rendered_row: int) -> None:
    cv2.putText(
        image,
        f"frame={frame_index} row={rendered_row}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_track_overlay(
    image: np.ndarray,
    normalized_landmarks: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    color: tuple[int, int, int],
    draw_full_mesh: bool,
    draw_points: bool,
    point_radius: int,
    line_thickness: int,
    draw_bbox: bool,
    draw_labels: bool,
) -> None:
    height, width = image.shape[:2]
    points = np.zeros((normalized_landmarks.shape[0], 2), dtype=np.int32)
    points[:, 0] = np.clip(np.round(normalized_landmarks[:, 0] * width), 0, width - 1).astype(np.int32)
    points[:, 1] = np.clip(np.round(normalized_landmarks[:, 1] * height), 0, height - 1).astype(np.int32)

    if draw_full_mesh:
        draw_connections(image, points, color=color, thickness=line_thickness)
    if draw_points:
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), point_radius, color, -1, lineType=cv2.LINE_AA)

    if draw_bbox and not np.isnan(bbox).all():
        x0, y0, x1, y1 = bbox
        cv2.rectangle(
            image,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            color,
            2,
        )
    if draw_labels and not np.isnan(bbox).all():
        x0, y0 = int(round(bbox[0])), int(round(bbox[1]))
        cv2.putText(
            image,
            f"track_{track_id}",
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_connections(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for group in CONNECTION_GROUPS:
        for connection in group:
            start = tuple(points[connection.start])
            end = tuple(points[connection.end])
            cv2.line(image, start, end, color, thickness, lineType=cv2.LINE_AA)

