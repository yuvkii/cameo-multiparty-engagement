from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .audio_features import StereoAudioFeatureExtractor
from .face_features import (
    BLENDSHAPE_NAMES,
    CandidateBox,
    FaceCandidateDetector,
    FaceFeatureExtractor,
    FaceObservation,
    bbox_iou,
    clip_xyxy,
)
from .rosbag_reader import CameraInfo, RosbagImageStream, VideoFileStream


TRACK_COLORS = (
    (83, 214, 110),
    (74, 163, 255),
    (255, 173, 82),
)


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    last_mar: float
    missed_count: int = 0


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path
    model_path: Path
    wav_path: Optional[Path]
    frame_step: int
    start_frame: int
    max_frames: Optional[int]
    max_tracks: int
    redetect_interval: int
    save_debug_video: bool
    audio_offset_sec: float
    bag_path: Optional[Path] = None
    video_path: Optional[Path] = None
    color_topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    # Expected yaw/pitch angle (degrees) a participant must have to be looking at
    # the robot (played by a confederate) rather than the RGB camera used for
    # this extraction (participants_rgb.avi).
    # Recalibrated 2026-07-04 from the mocap rig's camera calibration export
    # ("camera positions/Cal 2026-06-23 16.21.57 Exported.json"). That file
    # calibrates the 6 black-and-white mocap TRACKING cameras, not the RGB
    # camera itself — "Camera 4" in it is a separate physical device (IR,
    # monochrome) that just happens to be mounted on the same tripod as the
    # RGB camera, so its calibrated position/orientation is used as a stand-in
    # for the RGB camera's own (uncalibrated) position/orientation:
    #   - Participant distance from the RGB camera ≈ 1.75 m, derived from
    #     bbox_width in frame_features.csv via similar triangles (RealSense
    #     D435 RGB spec FOV 69.4°x42.5° at 640x480 → fx≈462px), averaged
    #     across all 6 sessions.
    #   - Robot ≈ 1.2 m to the RGB camera's right, roughly co-planar with it
    #     (per rig description: the RGB camera sits to the robot's left, ~1 m
    #     apart, facing the same participant), giving a real 3D offset instead
    #     of the old flat guess.
    #   - yaw = atan2(lateral, participant_distance) in mocap camera 4's local
    #     frame (used as the RGB camera's proxy frame); pitch comes from that
    #     mocap camera's own roll (its calibrated orientation isn't perfectly
    #     level) — treat pitch with more caution than yaw, since the axis
    #     convention for camera roll couldn't be independently cross-checked.
    # Positive yaw = participant turns to their LEFT (camera operator's right).
    # Negate if the robot is to the participant's right instead.
    # Verified 2026-07-04: nose-tip vs. eye-center landmark position (a lens-
    # distortion-independent 2D cue) confirms positive yaw does mean "turned
    # toward image-right" here, consistent with the robot being on that side.
    robot_camera_yaw_deg: float = 27.8
    robot_camera_pitch_deg: float = 21.2
    gaze_tolerance_deg: float = 15.0


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    observation: FaceObservation
    mar_delta: float
    gaze_target: str = "unknown"


def run_feature_pipeline(config: PipelineConfig) -> dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.video_path is not None:
        stream = VideoFileStream(config.video_path)
    else:
        stream = RosbagImageStream(
            bag_path=config.bag_path,
            color_topic=config.color_topic,
            camera_info_topic=config.camera_info_topic,
        )
    stream_meta = stream.inspect_stream()
    camera_info = stream.load_camera_info()
    audio_extractor = StereoAudioFeatureExtractor(config.wav_path) if config.wav_path else None

    detector = FaceCandidateDetector()
    landmarks = []
    bboxes = []
    blendshape_vectors = []
    frame_indices = []
    timestamps_sec = []

    csv_path = output_dir / "frame_features.csv"
    metadata_path = output_dir / "session_metadata.json"
    camera_info_path = output_dir / "camera_info.json"
    landmarks_path = output_dir / "landmarks_and_blendshapes.npz"
    debug_video_path = output_dir / "debug_overlay.mp4"

    csv_fieldnames = build_csv_fieldnames(audio_enabled=audio_extractor is not None)
    active_tracks: dict[int, TrackState] = {}
    processed_frames = 0
    frames_with_any_face = 0
    detected_slots = 0

    video_writer = None
    if config.save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_fps = max(1.0, stream_meta.fps / max(1, config.frame_step))
        video_writer = cv2.VideoWriter(
            str(debug_video_path),
            fourcc,
            debug_fps,
            (stream_meta.frame_width, stream_meta.frame_height),
        )

    try:
        with FaceFeatureExtractor(model_path=config.model_path) as face_extractor:
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
                writer.writeheader()

                for local_frame_idx, frame in enumerate(
                    stream.iter_color_frames(
                        frame_step=config.frame_step,
                        start_frame=config.start_frame,
                        max_frames=config.max_frames,
                    )
                ):
                    processed_frames += 1
                    frame_indices.append(frame.frame_index)
                    timestamps_sec.append(frame.timestamp_sec)
                    frame_landmarks = np.full((config.max_tracks, 478, 3), np.nan, dtype=np.float32)
                    frame_bboxes = np.full((config.max_tracks, 4), np.nan, dtype=np.float32)
                    frame_blendshapes = np.full((config.max_tracks, len(BLENDSHAPE_NAMES)), np.nan, dtype=np.float32)

                    redetect = (local_frame_idx % max(1, config.redetect_interval) == 0) or (not active_tracks)
                    candidate_boxes = build_candidate_boxes(
                        frame.image_bgr,
                        detector,
                        active_tracks,
                        max_tracks=config.max_tracks,
                        run_cascades=redetect,
                    )
                    observations = extract_observations(frame.image_bgr, face_extractor, candidate_boxes)
                    observations = deduplicate_observations(observations)
                    tracked_faces, active_tracks = assign_tracks(
                        observations=observations,
                        active_tracks=active_tracks,
                        max_tracks=config.max_tracks,
                    )
                    tracked_faces = attach_gaze_targets(
                        tracked_faces,
                        robot_camera_yaw_deg=config.robot_camera_yaw_deg,
                        robot_camera_pitch_deg=config.robot_camera_pitch_deg,
                        gaze_tolerance_deg=config.gaze_tolerance_deg,
                    )

                    if tracked_faces:
                        frames_with_any_face += 1
                    audio_features = (
                        audio_extractor.extract_for_timestamp(
                            frame.timestamp_sec,
                            audio_offset_sec=config.audio_offset_sec,
                        )
                        if audio_extractor is not None
                        else {}
                    )

                    debug_items = []
                    for track_id in range(config.max_tracks):
                        tracked = tracked_faces.get(track_id)
                        row = build_empty_row(
                            frame_index=frame.frame_index,
                            timestamp_sec=frame.timestamp_sec,
                            track_id=track_id,
                            audio_features=audio_features,
                        )
                        if tracked is not None:
                            detected_slots += 1
                            row.update(build_face_row(tracked, frame.image_bgr.shape))
                            frame_landmarks[track_id] = tracked.observation.landmarks_normalized
                            frame_bboxes[track_id] = np.asarray(tracked.observation.bbox_xyxy, dtype=np.float32)
                            frame_blendshapes[track_id] = np.asarray(
                                [tracked.observation.blendshapes.get(name, np.nan) for name in BLENDSHAPE_NAMES],
                                dtype=np.float32,
                            )
                            debug_items.append((track_id, tracked))
                        writer.writerow(row)

                    landmarks.append(frame_landmarks)
                    bboxes.append(frame_bboxes)
                    blendshape_vectors.append(frame_blendshapes)

                    if video_writer is not None:
                        overlay = draw_debug_overlay(frame.image_bgr.copy(), frame.frame_index, frame.timestamp_sec, debug_items)
                        video_writer.write(overlay)
    finally:
        if video_writer is not None:
            video_writer.release()

    np.savez_compressed(
        landmarks_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(timestamps_sec, dtype=np.float32),
        landmarks=np.asarray(landmarks, dtype=np.float32),
        bboxes=np.asarray(bboxes, dtype=np.float32),
        blendshapes=np.asarray(blendshape_vectors, dtype=np.float32),
        blendshape_names=np.asarray(BLENDSHAPE_NAMES),
    )

    with camera_info_path.open("w", encoding="utf-8") as camera_file:
        json.dump(camera_info_to_json(camera_info), camera_file, indent=2)

    summary = {
        "input": {
            "bag_path": str(config.video_path or config.bag_path),
            "wav_path": str(config.wav_path) if config.wav_path else None,
            "color_topic": stream_meta.color_topic,
            "camera_info_topic": stream_meta.camera_info_topic,
            "frame_step": config.frame_step,
            "start_frame": config.start_frame,
            "max_frames": config.max_frames,
            "audio_offset_sec": config.audio_offset_sec,
        },
        "stream": {
            "total_frames_in_bag": stream_meta.total_frames,
            "duration_sec": stream_meta.duration_sec,
            "fps": stream_meta.fps,
            "frame_width": stream_meta.frame_width,
            "frame_height": stream_meta.frame_height,
            "frame_encoding": stream_meta.frame_encoding,
        },
        "audio": (
            {
                "sample_rate_hz": audio_extractor.metadata.sample_rate_hz,
                "channels": audio_extractor.metadata.channels,
                "duration_sec": audio_extractor.metadata.duration_sec,
            }
            if audio_extractor is not None
            else None
        ),
        "outputs": {
            "frame_features_csv": str(csv_path),
            "landmarks_npz": str(landmarks_path),
            "camera_info_json": str(camera_info_path),
            "debug_video": str(debug_video_path) if config.save_debug_video else None,
        },
        "summary": {
            "processed_frames": processed_frames,
            "frames_with_any_face": frames_with_any_face,
            "face_detection_rate": (frames_with_any_face / processed_frames) if processed_frames else 0.0,
            "detected_track_slots": detected_slots,
            "max_tracks": config.max_tracks,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(summary, metadata_file, indent=2)

    return summary


def build_csv_fieldnames(audio_enabled: bool) -> list[str]:
    fields = [
        "frame_index",
        "timestamp_sec",
        "track_id",
        "detected",
        "candidate_source",
        "gaze_target",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "bbox_width",
        "bbox_height",
        "center_x_norm",
        "center_y_norm",
        "mar",
        "mar_delta",
        "head_yaw_deg",
        "head_pitch_deg",
        "head_roll_deg",
        "eye_gaze_yaw_deg",
        "eye_gaze_pitch_deg",
        "gaze_yaw_deg",
        "gaze_pitch_deg",
        "gaze_vector_x",
        "gaze_vector_y",
        "gaze_vector_z",
    ]
    if audio_enabled:
        fields.extend(
            [
                "audio_rms_left",
                "audio_rms_right",
                "audio_rms_mean",
                "audio_peak_mean",
                "audio_zcr",
                "audio_spectral_centroid_hz",
                "audio_spectral_rolloff_hz",
                "audio_stereo_corr",
                "audio_lr_balance",
            ]
        )
    fields.extend(BLENDSHAPE_NAMES)
    return fields


def build_empty_row(
    frame_index: int,
    timestamp_sec: float,
    track_id: int,
    audio_features: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "track_id": track_id,
        "detected": 0,
        "candidate_source": "",
        "gaze_target": "missing",
        "bbox_x0": float("nan"),
        "bbox_y0": float("nan"),
        "bbox_x1": float("nan"),
        "bbox_y1": float("nan"),
        "bbox_width": float("nan"),
        "bbox_height": float("nan"),
        "center_x_norm": float("nan"),
        "center_y_norm": float("nan"),
        "mar": float("nan"),
        "mar_delta": float("nan"),
        "head_yaw_deg": float("nan"),
        "head_pitch_deg": float("nan"),
        "head_roll_deg": float("nan"),
        "eye_gaze_yaw_deg": float("nan"),
        "eye_gaze_pitch_deg": float("nan"),
        "gaze_yaw_deg": float("nan"),
        "gaze_pitch_deg": float("nan"),
        "gaze_vector_x": float("nan"),
        "gaze_vector_y": float("nan"),
        "gaze_vector_z": float("nan"),
    }
    row.update(audio_features)
    for name in BLENDSHAPE_NAMES:
        row[name] = float("nan")
    return row


def build_face_row(tracked: TrackedFace, frame_shape: tuple[int, int, int]) -> dict[str, object]:
    frame_h, frame_w = frame_shape[:2]
    x0, y0, x1, y1 = tracked.observation.bbox_xyxy
    width = x1 - x0
    height = y1 - y0
    center_x, center_y = tracked.observation.center_xy
    row = {
        "detected": 1,
        "candidate_source": tracked.observation.source,
        "gaze_target": tracked.gaze_target,
        "bbox_x0": round(x0, 3),
        "bbox_y0": round(y0, 3),
        "bbox_x1": round(x1, 3),
        "bbox_y1": round(y1, 3),
        "bbox_width": round(width, 3),
        "bbox_height": round(height, 3),
        "center_x_norm": round(center_x / frame_w, 6),
        "center_y_norm": round(center_y / frame_h, 6),
        "mar": round(tracked.observation.mar, 6),
        "mar_delta": round(tracked.mar_delta, 6),
        "head_yaw_deg": round(tracked.observation.head_yaw_deg, 4),
        "head_pitch_deg": round(tracked.observation.head_pitch_deg, 4),
        "head_roll_deg": round(tracked.observation.head_roll_deg, 4),
        "eye_gaze_yaw_deg": round(tracked.observation.eye_gaze_yaw_deg, 4),
        "eye_gaze_pitch_deg": round(tracked.observation.eye_gaze_pitch_deg, 4),
        "gaze_yaw_deg": round(tracked.observation.gaze_yaw_deg, 4),
        "gaze_pitch_deg": round(tracked.observation.gaze_pitch_deg, 4),
        "gaze_vector_x": round(float(tracked.observation.gaze_vector[0]), 6),
        "gaze_vector_y": round(float(tracked.observation.gaze_vector[1]), 6),
        "gaze_vector_z": round(float(tracked.observation.gaze_vector[2]), 6),
    }
    for name in BLENDSHAPE_NAMES:
        row[name] = round(float(tracked.observation.blendshapes.get(name, float("nan"))), 6)
    return row


def camera_info_to_json(camera_info: CameraInfo) -> dict[str, object]:
    return {
        "topic": camera_info.topic,
        "width": camera_info.width,
        "height": camera_info.height,
        "distortion_model": camera_info.distortion_model,
        "distortion_coeffs": camera_info.distortion_coeffs,
        "intrinsic_matrix": camera_info.intrinsic_matrix,
        "projection_matrix": camera_info.projection_matrix,
    }


def build_candidate_boxes(
    frame_bgr: np.ndarray,
    detector: FaceCandidateDetector,
    active_tracks: dict[int, TrackState],
    max_tracks: int,
    run_cascades: bool,
) -> list[CandidateBox]:
    frame_h, frame_w = frame_bgr.shape[:2]
    candidates: list[CandidateBox] = []

    for track in active_tracks.values():
        if track.missed_count > 5:
            continue
        x0, y0, x1, y1 = track.bbox_xyxy
        width = x1 - x0
        height = y1 - y0
        expanded = clip_xyxy(
            (x0 - 0.35 * width, y0 - 0.45 * height, x1 + 0.35 * width, y1 + 0.45 * height),
            frame_w,
            frame_h,
        )
        ex0, ey0, ex1, ey1 = expanded
        candidates.append(CandidateBox(xywh=(ex0, ey0, ex1 - ex0, ey1 - ey0), source=f"track_{track.track_id}"))

    if run_cascades or len(candidates) < max_tracks:
        candidates.extend(detector.detect(frame_bgr))

    unique: list[CandidateBox] = []
    for candidate in candidates:
        x, y, w, h = candidate.xywh
        candidate_xyxy = (x, y, x + w, y + h)
        duplicate = False
        for kept in unique:
            kx, ky, kw, kh = kept.xywh
            kept_xyxy = (kx, ky, kx + kw, ky + kh)
            if bbox_iou(candidate_xyxy, kept_xyxy) > 0.45:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique[: max(max_tracks * 2, 4)]


def extract_observations(
    frame_bgr: np.ndarray,
    face_extractor: FaceFeatureExtractor,
    candidate_boxes: list[CandidateBox],
) -> list[FaceObservation]:
    observations: list[FaceObservation] = []
    for candidate in candidate_boxes:
        observation = face_extractor.extract_from_box(
            frame_bgr=frame_bgr,
            candidate_box_xywh=candidate.xywh,
            source=candidate.source,
        )
        if observation is not None:
            observations.append(observation)
    return observations


def deduplicate_observations(observations: list[FaceObservation]) -> list[FaceObservation]:
    deduped: list[FaceObservation] = []
    for observation in sorted(
        observations,
        key=lambda item: (item.bbox_xyxy[2] - item.bbox_xyxy[0]) * (item.bbox_xyxy[3] - item.bbox_xyxy[1]),
        reverse=True,
    ):
        keep = True
        for existing in deduped:
            if bbox_iou(observation.bbox_xyxy, existing.bbox_xyxy) > 0.35:
                keep = False
                break
            center_a = np.array(observation.center_xy)
            center_b = np.array(existing.center_xy)
            if np.linalg.norm(center_a - center_b) < 35.0:
                keep = False
                break
        if keep:
            deduped.append(observation)
    return sorted(deduped, key=lambda item: item.center_xy[0])


def assign_tracks(
    observations: list[FaceObservation],
    active_tracks: dict[int, TrackState],
    max_tracks: int,
) -> tuple[dict[int, TrackedFace], dict[int, TrackState]]:
    tracked: dict[int, TrackedFace] = {}
    remaining_observations = list(range(len(observations)))
    matched_tracks: set[int] = set()

    pair_candidates: list[tuple[float, int, int]] = []
    for track_id, track in active_tracks.items():
        for obs_index, observation in enumerate(observations):
            distance = np.linalg.norm(np.array(track.center_xy) - np.array(observation.center_xy))
            pair_candidates.append((distance, track_id, obs_index))
    for distance, track_id, obs_index in sorted(pair_candidates, key=lambda item: item[0]):
        if track_id in matched_tracks or obs_index not in remaining_observations:
            continue
        x0, y0, x1, y1 = observations[obs_index].bbox_xyxy
        face_scale = max(x1 - x0, y1 - y0)
        if distance > max(120.0, face_scale * 1.8):
            continue
        track = active_tracks[track_id]
        mar_delta = observations[obs_index].mar - track.last_mar
        tracked[track_id] = TrackedFace(track_id=track_id, observation=observations[obs_index], mar_delta=mar_delta)
        matched_tracks.add(track_id)
        remaining_observations.remove(obs_index)

    unused_track_ids = [track_id for track_id in range(max_tracks) if track_id not in tracked]
    for obs_index in sorted(remaining_observations, key=lambda idx: observations[idx].center_xy[0]):
        if not unused_track_ids:
            break
        track_id = unused_track_ids.pop(0)
        tracked[track_id] = TrackedFace(track_id=track_id, observation=observations[obs_index], mar_delta=0.0)

    new_tracks: dict[int, TrackState] = {}
    for track_id in range(max_tracks):
        if track_id in tracked:
            observation = tracked[track_id].observation
            new_tracks[track_id] = TrackState(
                track_id=track_id,
                bbox_xyxy=observation.bbox_xyxy,
                center_xy=observation.center_xy,
                last_mar=observation.mar,
                missed_count=0,
            )
        elif track_id in active_tracks and active_tracks[track_id].missed_count < 8:
            previous = active_tracks[track_id]
            new_tracks[track_id] = TrackState(
                track_id=track_id,
                bbox_xyxy=previous.bbox_xyxy,
                center_xy=previous.center_xy,
                last_mar=previous.last_mar,
                missed_count=previous.missed_count + 1,
            )
    return tracked, new_tracks


def attach_gaze_targets(
    tracked_faces: dict[int, TrackedFace],
    robot_camera_yaw_deg: float = 27.8,
    robot_camera_pitch_deg: float = 21.2,
    gaze_tolerance_deg: float = 15.0,
) -> dict[int, TrackedFace]:
    for track_id, tracked in tracked_faces.items():
        direction_2d = np.array(
            [tracked.observation.gaze_vector[0], tracked.observation.gaze_vector[1]],
            dtype=np.float32,
        )
        direction_norm = np.linalg.norm(direction_2d)
        if direction_norm < 1e-5:
            tracked.gaze_target = "unknown"
            continue
        direction_2d /= direction_norm

        best_label = "camera"
        best_score = float("inf")
        origin = np.array(tracked.observation.center_xy, dtype=np.float32)

        for other_id, other in tracked_faces.items():
            if other_id == track_id:
                continue
            target_vector = np.array(other.observation.center_xy, dtype=np.float32) - origin
            distance = np.linalg.norm(target_vector)
            if distance < 1e-5:
                continue
            target_unit = target_vector / distance
            dot = float(np.clip(np.dot(direction_2d, target_unit), -1.0, 1.0))
            if dot <= 0.25:
                continue
            angle_deg = float(np.degrees(np.arccos(dot)))
            score = angle_deg + 0.03 * distance
            if score < best_score and angle_deg < 32.0:
                best_score = score
                best_label = f"track_{other_id}"

        if best_label == "camera":
            yaw_err = abs(tracked.observation.gaze_yaw_deg - robot_camera_yaw_deg)
            pitch_err = abs(tracked.observation.gaze_pitch_deg - robot_camera_pitch_deg)
            if yaw_err > gaze_tolerance_deg or pitch_err > gaze_tolerance_deg:
                best_label = "elsewhere"
        tracked.gaze_target = best_label
    return tracked_faces


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    debug_items: list[tuple[int, TrackedFace]],
) -> np.ndarray:
    cv2.putText(
        frame_bgr,
        f"frame={frame_index}  time={timestamp_sec:.2f}s",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    for track_id, tracked in debug_items:
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        x0, y0, x1, y1 = [int(round(v)) for v in tracked.observation.bbox_xyxy]
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
        label = f"id={track_id} MAR={tracked.observation.mar:.3f} gaze={tracked.gaze_target}"
        cv2.putText(
            frame_bgr,
            label,
            (x0, max(24, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        center = np.array(tracked.observation.center_xy, dtype=np.float32)
        arrow_length = 80.0
        arrow_tip = center + tracked.observation.gaze_vector[:2] * arrow_length
        cv2.arrowedLine(
            frame_bgr,
            tuple(center.astype(int)),
            tuple(arrow_tip.astype(int)),
            color,
            2,
            tipLength=0.2,
        )
    return frame_bgr
