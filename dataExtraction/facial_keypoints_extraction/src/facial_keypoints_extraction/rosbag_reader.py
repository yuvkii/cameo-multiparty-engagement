from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterator, Optional

import cv2
import numpy as np
from rosbags.rosbag1 import Reader


@dataclass(slots=True)
class CameraInfo:
    topic: str
    width: int
    height: int
    distortion_model: str
    distortion_coeffs: list[float]
    intrinsic_matrix: list[float]
    projection_matrix: list[float]


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    timestamp_ns: int
    timestamp_sec: float
    image_bgr: np.ndarray


@dataclass(slots=True)
class StreamMetadata:
    bag_path: Path
    color_topic: str
    camera_info_topic: str
    total_frames: int
    duration_sec: float
    fps: float
    frame_width: int
    frame_height: int
    frame_encoding: str


def _parse_ros1_string(buffer: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", buffer, offset)
    offset += 4
    value = buffer[offset : offset + length].decode("utf-8")
    offset += length
    return value, offset


def _parse_ros1_uint8_array(buffer: bytes, offset: int) -> tuple[memoryview, int]:
    (length,) = struct.unpack_from("<I", buffer, offset)
    offset += 4
    data = memoryview(buffer)[offset : offset + length]
    offset += length
    return data, offset


def _parse_ros1_float64_array(buffer: bytes, offset: int) -> tuple[list[float], int]:
    (length,) = struct.unpack_from("<I", buffer, offset)
    offset += 4
    values = list(struct.unpack_from(f"<{length}d", buffer, offset))
    offset += 8 * length
    return values, offset


def parse_image_message(buffer: bytes) -> tuple[np.ndarray, dict[str, object]]:
    """Parse a ROS1 sensor_msgs/Image payload."""
    offset = 0
    seq, sec, nsec = struct.unpack_from("<III", buffer, offset)
    offset += 12
    frame_id, offset = _parse_ros1_string(buffer, offset)
    height, width = struct.unpack_from("<II", buffer, offset)
    offset += 8
    encoding, offset = _parse_ros1_string(buffer, offset)
    is_bigendian = buffer[offset]
    offset += 1
    (step,) = struct.unpack_from("<I", buffer, offset)
    offset += 4
    image_data, offset = _parse_ros1_uint8_array(buffer, offset)

    if encoding != "bgr8":
        raise ValueError(f"Unsupported image encoding: {encoding}")
    if is_bigendian != 0:
        raise ValueError("Big-endian image payloads are not supported.")

    array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, step))
    image_bgr = array[:, : width * 3].reshape((height, width, 3))

    meta = {
        "seq": seq,
        "sec": sec,
        "nsec": nsec,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "encoding": encoding,
    }
    return image_bgr, meta


def parse_camera_info_message(buffer: bytes, topic: str) -> CameraInfo:
    """Parse a ROS1 sensor_msgs/CameraInfo payload."""
    offset = 0
    offset += 12  # seq, sec, nsec
    _, offset = _parse_ros1_string(buffer, offset)  # frame_id
    height, width = struct.unpack_from("<II", buffer, offset)
    offset += 8
    distortion_model, offset = _parse_ros1_string(buffer, offset)
    distortion_coeffs, offset = _parse_ros1_float64_array(buffer, offset)
    intrinsic_matrix = list(struct.unpack_from("<9d", buffer, offset))
    offset += 9 * 8
    offset += 9 * 8  # R matrix
    projection_matrix = list(struct.unpack_from("<12d", buffer, offset))
    offset += 12 * 8
    offset += 4 + 4  # binning_x, binning_y
    offset += 4 + 4 + 4 + 4 + 1  # ROI

    return CameraInfo(
        topic=topic,
        width=width,
        height=height,
        distortion_model=distortion_model,
        distortion_coeffs=distortion_coeffs,
        intrinsic_matrix=intrinsic_matrix,
        projection_matrix=projection_matrix,
    )


class RosbagImageStream:
    """Read RGB frames and camera metadata from a ROS1 bag."""

    def __init__(
        self,
        bag_path: str | Path,
        color_topic: Optional[str] = None,
        camera_info_topic: Optional[str] = None,
    ) -> None:
        self.bag_path = Path(bag_path)
        self.color_topic = color_topic
        self.camera_info_topic = camera_info_topic

    def _discover_topics(self) -> tuple[str, str]:
        with Reader(self.bag_path) as reader:
            color_topic = self.color_topic
            camera_info_topic = self.camera_info_topic
            if color_topic is None:
                candidates = [
                    conn.topic
                    for conn in reader.connections
                    if "Color" in conn.topic and conn.topic.endswith("/image/data")
                ]
                if not candidates:
                    raise ValueError("Could not find a color image topic in the bag.")
                color_topic = sorted(candidates)[0]
            if camera_info_topic is None:
                expected = color_topic.rsplit("/", 2)[0] + "/info/camera_info"
                if any(conn.topic == expected for conn in reader.connections):
                    camera_info_topic = expected
                else:
                    candidates = [
                        conn.topic
                        for conn in reader.connections
                        if "Color" in conn.topic and conn.topic.endswith("/camera_info")
                    ]
                    if not candidates:
                        raise ValueError("Could not find a color camera_info topic in the bag.")
                    camera_info_topic = sorted(candidates)[0]
        self.color_topic = color_topic
        self.camera_info_topic = camera_info_topic
        return color_topic, camera_info_topic

    def load_camera_info(self) -> CameraInfo:
        _, camera_info_topic = self._discover_topics()
        with Reader(self.bag_path) as reader:
            conn = next(c for c in reader.connections if c.topic == camera_info_topic)
            for _, _, raw in reader.messages(connections=[conn]):
                return parse_camera_info_message(raw, topic=camera_info_topic)
        raise RuntimeError(f"No camera info messages found for topic {camera_info_topic}.")

    def inspect_stream(self) -> StreamMetadata:
        color_topic, camera_info_topic = self._discover_topics()
        with Reader(self.bag_path) as reader:
            conn = next(c for c in reader.connections if c.topic == color_topic)
            count = 0
            first_ts = None
            last_ts = None
            frame_width = 0
            frame_height = 0
            frame_encoding = ""
            for _, ts, raw in reader.messages(connections=[conn]):
                if first_ts is None:
                    image, meta = parse_image_message(raw)
                    frame_height, frame_width = image.shape[:2]
                    frame_encoding = str(meta["encoding"])
                    first_ts = ts
                last_ts = ts
                count += 1
            if first_ts is None or last_ts is None:
                raise RuntimeError(f"No image messages found for topic {color_topic}.")
            duration_sec = (last_ts - first_ts) / 1e9 if count > 1 else 0.0
            fps = (count - 1) / duration_sec if duration_sec > 0 else 0.0
        return StreamMetadata(
            bag_path=self.bag_path,
            color_topic=color_topic,
            camera_info_topic=camera_info_topic,
            total_frames=count,
            duration_sec=duration_sec,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_encoding=frame_encoding,
        )

    def iter_color_frames(
        self,
        frame_step: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> Iterator[FrameRecord]:
        if frame_step <= 0:
            raise ValueError("frame_step must be a positive integer.")
        color_topic, _ = self._discover_topics()
        yielded = 0
        with Reader(self.bag_path) as reader:
            conn = next(c for c in reader.connections if c.topic == color_topic)
            for frame_index, (_, timestamp_ns, raw) in enumerate(reader.messages(connections=[conn])):
                if frame_index < start_frame:
                    continue
                if (frame_index - start_frame) % frame_step != 0:
                    continue
                image_bgr, _ = parse_image_message(raw)
                yield FrameRecord(
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    timestamp_sec=timestamp_ns / 1e9,
                    image_bgr=image_bgr,
                )
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break


class VideoFileStream:
    """Read RGB frames from a plain video file (.avi/.mp4) using OpenCV."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)

    def inspect_stream(self) -> StreamMetadata:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps if fps > 0 else 0.0
        finally:
            cap.release()
        return StreamMetadata(
            bag_path=self.video_path,
            color_topic="video_file",
            camera_info_topic="",
            total_frames=total_frames,
            duration_sec=duration_sec,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_encoding="bgr8",
        )

    def load_camera_info(self) -> CameraInfo:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        fx = float(width)
        fy = float(width)
        cx = width / 2.0
        cy = height / 2.0
        return CameraInfo(
            topic="video_file",
            width=width,
            height=height,
            distortion_model="plumb_bob",
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            intrinsic_matrix=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            projection_matrix=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

    def iter_color_frames(
        self,
        frame_step: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> Iterator[FrameRecord]:
        if frame_step <= 0:
            raise ValueError("frame_step must be a positive integer.")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            frame_index = 0
            yielded = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_index < start_frame:
                    frame_index += 1
                    continue
                if (frame_index - start_frame) % frame_step == 0:
                    timestamp_ns = int(frame_index / fps * 1e9)
                    yield FrameRecord(
                        frame_index=frame_index,
                        timestamp_ns=timestamp_ns,
                        timestamp_sec=timestamp_ns / 1e9,
                        image_bgr=frame_bgr,
                    )
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        break
                frame_index += 1
        finally:
            cap.release()
