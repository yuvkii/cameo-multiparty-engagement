from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker


MAR_VERTICAL_PAIRS = ((13, 14), (37, 84))
MAR_HORIZONTAL_PAIR = (78, 308)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)
RIGHT_EYE_VERTICAL = (159, 145)
LEFT_EYE_VERTICAL = (386, 374)
RIGHT_IRIS = (469, 470, 471, 472)
LEFT_IRIS = (474, 475, 476, 477)
LEFT_EYE_ROLL_REF = (33, 263)
BLENDSHAPE_NAMES = [member.name.lower() for member in mp_face_landmarker.Blendshapes]


@dataclass(slots=True)
class CandidateBox:
    xywh: tuple[int, int, int, int]
    source: str


@dataclass(slots=True)
class FaceObservation:
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    landmarks_frame: np.ndarray
    landmarks_normalized: np.ndarray
    mar: float
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    eye_gaze_yaw_deg: float
    eye_gaze_pitch_deg: float
    gaze_yaw_deg: float
    gaze_pitch_deg: float
    gaze_vector: np.ndarray
    blendshapes: dict[str, float]
    source: str
    transform_matrix: Optional[np.ndarray]


def normalize_blendshape_name(name: str) -> str:
    cleaned = name.lstrip("_")
    characters: list[str] = []
    for index, char in enumerate(cleaned):
        if char.isupper() and index > 0 and cleaned[index - 1].isalnum() and cleaned[index - 1] != "_":
            characters.append("_")
        characters.append(char.lower())
    normalized = "".join(characters)
    return normalized


def clip_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        int(max(0, min(width - 1, round(x0)))),
        int(max(0, min(height - 1, round(y0)))),
        int(max(1, min(width, round(x1)))),
        int(max(1, min(height, round(y1)))),
    )


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


class FaceCandidateDetector:
    """Multi-cascade candidate generator for small, distant faces."""

    def __init__(
        self,
        upscale_factor: float = 2.5,
        top_region_ratio: float = 0.70,
        min_size_px: int = 36,
        min_neighbors: int = 4,
    ) -> None:
        self.upscale_factor = upscale_factor
        self.top_region_ratio = top_region_ratio
        self.min_size_px = min_size_px
        self.min_neighbors = min_neighbors
        self.cascade_specs = (
            ("haarcascade_frontalface_default.xml", False),
            ("haarcascade_frontalface_alt2.xml", False),
            ("haarcascade_profileface.xml", True),
        )
        self.cascades = {
            name: cv2.CascadeClassifier(cv2.data.haarcascades + name)
            for name, _ in self.cascade_specs
        }

    def detect(self, frame_bgr: np.ndarray) -> list[CandidateBox]:
        scaled = cv2.resize(
            frame_bgr,
            None,
            fx=self.upscale_factor,
            fy=self.upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        roi_height = int(scaled.shape[0] * self.top_region_ratio)
        gray = cv2.cvtColor(scaled[:roi_height, :], cv2.COLOR_BGR2GRAY)

        raw_candidates: list[tuple[int, int, int, int, str]] = []
        for cascade_name, flip in self.cascade_specs:
            src = cv2.flip(gray, 1) if flip else gray
            boxes = self.cascades[cascade_name].detectMultiScale(
                src,
                scaleFactor=1.05,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_size_px, self.min_size_px),
            )
            for x, y, w, h in boxes:
                if flip:
                    x = gray.shape[1] - x - w
                raw_candidates.append((int(x), int(y), int(w), int(h), cascade_name))

        accepted: list[tuple[int, int, int, int, str]] = []
        for x, y, w, h, source in sorted(raw_candidates, key=lambda item: item[2] * item[3], reverse=True):
            center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
            keep = True
            for ax, ay, aw, ah, _ in accepted:
                other_center = np.array([ax + aw / 2.0, ay + ah / 2.0], dtype=np.float32)
                if np.linalg.norm(center - other_center) < 0.6 * max(max(w, h), max(aw, ah)):
                    keep = False
                    break
            if keep:
                accepted.append((x, y, w, h, source))

        candidates: list[CandidateBox] = []
        for x, y, w, h, source in accepted:
            box = (
                int(x / self.upscale_factor),
                int(y / self.upscale_factor),
                max(1, int(w / self.upscale_factor)),
                max(1, int(h / self.upscale_factor)),
            )
            candidates.append(CandidateBox(xywh=box, source=source))
        return candidates


class FaceFeatureExtractor:
    """Crop-level MediaPipe extractor with heuristic gaze estimation."""

    def __init__(
        self,
        model_path: str | Path,
        crop_upscale: float = 4.0,
        face_detection_confidence: float = 0.25,
        face_presence_confidence: float = 0.25,
        tracking_confidence: float = 0.25,
    ) -> None:
        self.model_path = str(model_path)
        self.crop_upscale = crop_upscale
        self.face_detection_confidence = face_detection_confidence
        self.face_presence_confidence = face_presence_confidence
        self.tracking_confidence = tracking_confidence
        self._landmarker = None

    def __enter__(self) -> "FaceFeatureExtractor":
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.face_detection_confidence,
            min_face_presence_confidence=self.face_presence_confidence,
            min_tracking_confidence=self.tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def extract_from_box(
        self,
        frame_bgr: np.ndarray,
        candidate_box_xywh: tuple[int, int, int, int],
        source: str,
    ) -> Optional[FaceObservation]:
        if self._landmarker is None:
            raise RuntimeError("FaceFeatureExtractor must be used as a context manager.")

        frame_h, frame_w = frame_bgr.shape[:2]
        x, y, w, h = candidate_box_xywh
        pad_x = int(w * 1.0)
        pad_y = int(h * 1.2)
        x0, y0, x1, y1 = clip_xyxy((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y), frame_w, frame_h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        upscaled_crop = cv2.resize(
            crop,
            None,
            fx=self.crop_upscale,
            fy=self.crop_upscale,
            interpolation=cv2.INTER_CUBIC,
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        landmarks_frame = np.zeros((len(face_landmarks), 3), dtype=np.float32)
        landmarks_norm = np.zeros((len(face_landmarks), 3), dtype=np.float32)
        crop_width = float(x1 - x0)
        crop_height = float(y1 - y0)

        for idx, landmark in enumerate(face_landmarks):
            frame_x = x0 + landmark.x * crop_width
            frame_y = y0 + landmark.y * crop_height
            landmarks_frame[idx, 0] = frame_x
            landmarks_frame[idx, 1] = frame_y
            landmarks_frame[idx, 2] = landmark.z
            landmarks_norm[idx, 0] = frame_x / frame_w
            landmarks_norm[idx, 1] = frame_y / frame_h
            landmarks_norm[idx, 2] = landmark.z

        xs = landmarks_frame[:, 0]
        ys = landmarks_frame[:, 1]
        bbox_xyxy = (
            float(np.min(xs)),
            float(np.min(ys)),
            float(np.max(xs)),
            float(np.max(ys)),
        )
        center_xy = (
            float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0),
            float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0),
        )

        blendshapes = {name: 0.0 for name in BLENDSHAPE_NAMES}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                if category.category_name:
                    normalized_name = normalize_blendshape_name(category.category_name)
                    if normalized_name in blendshapes:
                        blendshapes[normalized_name] = float(category.score)

        transform_matrix = None
        if result.facial_transformation_matrixes:
            transform_matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)

        mar = compute_mar(landmarks_frame)
        head_yaw, head_pitch, head_roll = compute_head_pose(landmarks_frame, transform_matrix)
        eye_yaw, eye_pitch = compute_eye_gaze_offsets(landmarks_frame)
        gaze_yaw = head_yaw + eye_yaw
        gaze_pitch = head_pitch + eye_pitch
        gaze_vector = gaze_vector_from_angles(gaze_yaw, gaze_pitch)

        return FaceObservation(
            bbox_xyxy=bbox_xyxy,
            center_xy=center_xy,
            landmarks_frame=landmarks_frame,
            landmarks_normalized=landmarks_norm,
            mar=float(mar),
            head_yaw_deg=float(head_yaw),
            head_pitch_deg=float(head_pitch),
            head_roll_deg=float(head_roll),
            eye_gaze_yaw_deg=float(eye_yaw),
            eye_gaze_pitch_deg=float(eye_pitch),
            gaze_yaw_deg=float(gaze_yaw),
            gaze_pitch_deg=float(gaze_pitch),
            gaze_vector=gaze_vector.astype(np.float32),
            blendshapes=blendshapes,
            source=source,
            transform_matrix=transform_matrix,
        )


def compute_mar(landmarks_frame: np.ndarray) -> float:
    upper_a, lower_a = MAR_VERTICAL_PAIRS[0]
    upper_b, lower_b = MAR_VERTICAL_PAIRS[1]
    left_corner, right_corner = MAR_HORIZONTAL_PAIR
    vertical_a = np.linalg.norm(landmarks_frame[upper_a, :2] - landmarks_frame[lower_a, :2])
    vertical_b = np.linalg.norm(landmarks_frame[upper_b, :2] - landmarks_frame[lower_b, :2])
    horizontal = np.linalg.norm(landmarks_frame[left_corner, :2] - landmarks_frame[right_corner, :2]) + 1e-6
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def compute_head_pose(
    landmarks_frame: np.ndarray,
    transform_matrix: Optional[np.ndarray],
) -> tuple[float, float, float]:
    if transform_matrix is not None and transform_matrix.shape[0] >= 3 and transform_matrix.shape[1] >= 3:
        rotation = transform_matrix[:3, :3]
        forward = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        norm = np.linalg.norm(forward) + 1e-6
        forward = forward / norm
        yaw = np.degrees(np.arctan2(forward[0], forward[2]))
        pitch = np.degrees(np.arctan2(-forward[1], np.linalg.norm(forward[[0, 2]]) + 1e-6))
    else:
        yaw = float("nan")
        pitch = float("nan")

    left_anchor = landmarks_frame[LEFT_EYE_ROLL_REF[0], :2]
    right_anchor = landmarks_frame[LEFT_EYE_ROLL_REF[1], :2]
    roll = np.degrees(np.arctan2(right_anchor[1] - left_anchor[1], right_anchor[0] - left_anchor[0] + 1e-6))
    return float(yaw), float(pitch), float(roll)


def _eye_offsets(
    landmarks_frame: np.ndarray,
    corners: tuple[int, int],
    vertical: tuple[int, int],
    iris_indices: Iterable[int],
) -> tuple[float, float]:
    iris_points = landmarks_frame[list(iris_indices), :2]
    iris_center = iris_points.mean(axis=0)
    eye_corners = landmarks_frame[list(corners), :2]
    eye_vertical = landmarks_frame[list(vertical), :2]

    min_x = float(np.min(eye_corners[:, 0]))
    max_x = float(np.max(eye_corners[:, 0]))
    min_y = float(np.min(eye_vertical[:, 1]))
    max_y = float(np.max(eye_vertical[:, 1]))

    horizontal_ratio = (iris_center[0] - min_x) / (max_x - min_x + 1e-6)
    vertical_ratio = (iris_center[1] - min_y) / (max_y - min_y + 1e-6)
    horizontal_offset = np.clip((horizontal_ratio - 0.5) * 2.0, -1.0, 1.0)
    vertical_offset = np.clip((vertical_ratio - 0.5) * 2.0, -1.0, 1.0)
    return float(horizontal_offset), float(vertical_offset)


def compute_eye_gaze_offsets(landmarks_frame: np.ndarray) -> tuple[float, float]:
    right_h, right_v = _eye_offsets(landmarks_frame, RIGHT_EYE_CORNERS, RIGHT_EYE_VERTICAL, RIGHT_IRIS)
    left_h, left_v = _eye_offsets(landmarks_frame, LEFT_EYE_CORNERS, LEFT_EYE_VERTICAL, LEFT_IRIS)
    horizontal = (right_h + left_h) / 2.0
    vertical = (right_v + left_v) / 2.0
    eye_yaw_deg = horizontal * 35.0
    eye_pitch_deg = -vertical * 25.0
    return float(eye_yaw_deg), float(eye_pitch_deg)


def gaze_vector_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    vector = np.array(
        [
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
            np.cos(yaw) * np.cos(pitch),
        ],
        dtype=np.float32,
    )
    norm = np.linalg.norm(vector) + 1e-6
    return vector / norm
