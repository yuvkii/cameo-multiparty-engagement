# -*- coding: utf-8 -*-
"""
Final Furhat speaker pipeline.

Run from:
    <furhat-data-root>/

Main inputs:
    participants_cam3/*.mp4
    <session>/<trial>/participants_rgb_audio_edited.wav
    <session>/transcript/<trial>/timeline_video.json
    participants_cam3/sharingan_output/*-sharingan-bytetrack.json
    speaker_track_mappings/*_mapping.json

Recommended mapping format:
{
  "speaker_to_track_ids": {
    "A": [3, 8],
    "B": [1],
    "C": [5, 10],
    "R": []
  }
}

The same speaker may have multiple ByteTrack IDs at different times.
The code chooses whichever candidate ID is visible in the current frame.

Outputs under <furhat-data-root>/:
    03_20_1_cam3_offset.json
    03_20_1_cam3_timeline_shifted.json
    03_20_1_cam3_speaker.mp4
    03_20_1_cam3_speaker_track.json
    03_20_1_cam3_speaker_frames.csv
    speaker_pipeline_summary.csv
    speaker_pipeline.log
"""

import csv
import os
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(os.environ.get("FURHAT_DATA_ROOT", "."))

VIDEO_DIR = ROOT / "participants_cam3"
TRACKING_DIR = VIDEO_DIR / "sharingan_output"
MAPPING_DIR = ROOT / "speaker_track_mappings"

REFERENCE_AUDIO_NAME = "participants_rgb_audio_edited.wav"
TRANSCRIPT_NAME = "timeline_video.json"

# Transcript timecode format: HH:MM:SS:FF, where FF is 0-29.
TRANSCRIPT_FPS = 30.0

OUTPUT_ROOT = ROOT / "speaker_results"
OFFSET_DIR = OUTPUT_ROOT / "offset"
VIDEO_OUT_DIR = OUTPUT_ROOT / "videos"
JSON_OUT_DIR = OUTPUT_ROOT / "json"
CSV_OUT_DIR = OUTPUT_ROOT / "csv"

for d in (
    OUTPUT_ROOT,
    OFFSET_DIR,
    VIDEO_OUT_DIR,
    JSON_OUT_DIR,
    CSV_OUT_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
MATCH_DURATION_SECONDS = 120.0
MAX_OFFSET_SECONDS = 300.0

REFERENCE_START_SECONDS = 0.0
TARGET_START_SECONDS = 0.0

FORCE_OFFSET = False
FORCE_SHIFTED_TIMELINE = False
FORCE_VISUALIZATION = False

VIDEO_CODEC = "mp4v"

SHOW_ALL_TRACKS = True
SHOW_SUBTITLES = True
SHOW_TIMECODE = True

SPEAKER_COLORS = {
    "A": (60, 220, 60),
    "B": (255, 150, 40),
    "C": (220, 80, 220),
    "R": (40, 200, 255),
}

INACTIVE_COLOR = (150, 150, 150)

START_KEYS = (
    "start",
    "start_time",
    "start_sec",
    "begin",
    "start_timecode",
)
END_KEYS = (
    "end",
    "end_time",
    "end_sec",
    "stop",
    "end_timecode",
)
SPEAKER_KEYS = ("speaker", "speaker_id", "label", "participant")
TEXT_KEYS = ("text", "transcript", "utterance", "content")


# ============================================================
# BASIC HELPERS
# ============================================================

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)

    with open(
        OUTPUT_ROOT / "speaker_pipeline.log",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(line + "\n")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_valid_file(path):
    return path.is_file() and path.stat().st_size > 0


def first_value(item, keys, default=None):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


# ============================================================
# PATH MATCHING
# ============================================================

def parse_video_name(video_path):
    """
    Examples:
        03_20_1_cam3.mp4
            session = 03_20
            trial   = 1

        03_24_1_3_cam3.mp4
            session = 03_24_1
            trial   = 3
    """
    stem = video_path.stem

    if not stem.endswith("_cam3"):
        return None

    base = stem[:-len("_cam3")]

    if "_" not in base:
        return None

    session, trial = base.rsplit("_", 1)

    if not trial.isdigit():
        return None

    return {
        "session": session,
        "trial": trial,
        "stem": stem,
    }


def find_videos():
    return sorted(
        [
            path
            for path in VIDEO_DIR.glob("*.mp4")
            if path.stem.endswith("_cam3")
        ],
        key=lambda path: path.name.lower(),
    )


def get_paths(video_path, parsed):
    session = parsed["session"]
    trial = parsed["trial"]
    stem = parsed["stem"]
    offset_dir = VIDEO_DIR / "offset"
    offset_dir.mkdir(parents=True, exist_ok=True)

    return {
        "video": video_path,
        "target_audio": video_path.with_suffix(".wav"),

        "reference_audio": (
            ROOT
            / session
            / trial
            / REFERENCE_AUDIO_NAME
        ),

        "timeline": (
            ROOT
            / session
            / "transcript"
            / trial
            / TRANSCRIPT_NAME
        ),

        "tracking": (
            TRACKING_DIR
            / f"{stem}-sharingan-bytetrack.json"
        ),

        "mapping": (
            MAPPING_DIR
            / f"{stem}_mapping.json"
        ),

        # OUTPUTS
        "offset": OFFSET_DIR / f"{stem}_offset.json",

        "shifted_timeline": (
            OFFSET_DIR
            / f"{stem}_timeline_shifted.json"
        ),

        "speaker_video": (
            VIDEO_OUT_DIR
            / f"{stem}_speaker.mp4"
        ),

        "speaker_track_json": (
            JSON_OUT_DIR
            / f"{stem}_speaker_track.json"
        ),

        "speaker_frames_csv": (
            CSV_OUT_DIR
            / f"{stem}_speaker_frames.csv"
        ),
    }


# ============================================================
# AUDIO OFFSET
# ============================================================

def extract_audio(
    input_path,
    output_wav,
    start_seconds,
    duration_seconds,
):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        str(input_path),
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]

    subprocess.run(command, check=True)


def read_audio(path):
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise RuntimeError(f"Empty audio: {path}")

    audio = audio - np.mean(audio)

    std = np.std(audio)

    if std > 1e-8:
        audio = audio / std

    return audio, sample_rate


def estimate_offset(reference_audio, target_audio):
    correlation = correlate(
        target_audio,
        reference_audio,
        mode="full",
        method="fft",
    )

    lags = correlation_lags(
        len(target_audio),
        len(reference_audio),
        mode="full",
    )

    max_lag_samples = int(
        MAX_OFFSET_SECONDS * SAMPLE_RATE
    )

    valid = np.abs(lags) <= max_lag_samples
    valid_correlation = correlation[valid]
    valid_lags = lags[valid]

    best_index = int(np.argmax(valid_correlation))
    best_lag_samples = int(valid_lags[best_index])

    offset_seconds = best_lag_samples / SAMPLE_RATE

    denominator = max(
        np.linalg.norm(reference_audio)
        * np.linalg.norm(target_audio),
        1e-12,
    )

    correlation_score = float(
        valid_correlation[best_index]
        / denominator
    )

    return {
        "lag_samples": best_lag_samples,
        "offset_seconds": offset_seconds,
        "correlation_score": correlation_score,
    }


def create_offset_json(paths):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        reference_wav = temp_dir / "reference.wav"
        target_wav = temp_dir / "target.wav"

        extract_audio(
            paths["reference_audio"],
            reference_wav,
            REFERENCE_START_SECONDS,
            MATCH_DURATION_SECONDS,
        )

        extract_audio(
            paths["target_audio"],
            target_wav,
            TARGET_START_SECONDS,
            MATCH_DURATION_SECONDS,
        )

        reference_audio, reference_sr = read_audio(
            reference_wav
        )

        target_audio, target_sr = read_audio(
            target_wav
        )

        if reference_sr != target_sr:
            raise RuntimeError(
                f"Sample-rate mismatch: "
                f"{reference_sr} vs {target_sr}"
            )

        result = estimate_offset(
            reference_audio,
            target_audio,
        )

    global_offset = (
        TARGET_START_SECONDS
        - REFERENCE_START_SECONDS
        + result["offset_seconds"]
    )

    output = {
        "reference_audio": str(paths["reference_audio"]),
        "target_video": str(paths["video"]),
        "sample_rate": SAMPLE_RATE,
        "reference_start_seconds": (
            REFERENCE_START_SECONDS
        ),
        "target_start_seconds": TARGET_START_SECONDS,
        "matching_duration_seconds": (
            MATCH_DURATION_SECONDS
        ),
        "lag_samples": result["lag_samples"],
        "offset_seconds": global_offset,
        "correlation_score": (
            result["correlation_score"]
        ),
        "offset_convention": (
            "cam3_time = transcript_time + offset_seconds"
        ),
    }

    save_json(paths["offset"], output)
    return output


def timecode_to_seconds(value, fps=TRANSCRIPT_FPS):
    """
    Convert HH:MM:SS:FF timecode to seconds.

    Example at 30 FPS:
        00:00:01:05 -> 1 + 5/30 = 1.1666667 seconds

    Numeric seconds are also accepted.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    # Support numeric strings such as "1.25".
    try:
        return float(value)
    except ValueError:
        pass

    parts = value.split(":")

    if len(parts) != 4:
        raise ValueError(
            f"Unsupported time value: {value!r}. "
            "Expected HH:MM:SS:FF."
        )

    try:
        hours, minutes, seconds, frames = [
            int(part)
            for part in parts
        ]
    except ValueError as exc:
        raise ValueError(
            f"Invalid timecode: {value!r}"
        ) from exc

    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(
            f"Negative timecode component: {value!r}"
        )

    if minutes >= 60 or seconds >= 60:
        raise ValueError(
            f"Invalid minute/second value: {value!r}"
        )

    if frames < 0 or frames >= int(fps):
        raise ValueError(
            f"Frame number {frames} is outside "
            f"0-{int(fps) - 1} in timecode {value!r}."
        )

    return (
        hours * 3600.0
        + minutes * 60.0
        + seconds
        + frames / fps
    )


# ============================================================
# TIMELINE SHIFT
# ============================================================

def get_timeline_list(data):
    if isinstance(data, list):
        return data, None

    if isinstance(data, dict):
        for key in (
            "segments",
            "timeline",
            "utterances",
            "transcript",
        ):
            if isinstance(data.get(key), list):
                return data[key], key

    raise ValueError(
        "Timeline JSON does not contain a supported list."
    )


def shift_item(item, offset_seconds):
    """
    Convert transcript timecode to seconds and apply audio offset.

    The original timecode fields are preserved. Numeric "start" and
    "end" fields are added for visualization.
    """
    shifted = dict(item)

    raw_start = first_value(item, START_KEYS)
    raw_end = first_value(item, END_KEYS)

    if raw_start is not None:
        shifted["start"] = max(
            0.0,
            timecode_to_seconds(raw_start)
            + offset_seconds,
        )

    if raw_end is not None:
        shifted["end"] = max(
            0.0,
            timecode_to_seconds(raw_end)
            + offset_seconds,
        )

    if "start_timecode" in item:
        shifted["original_start_timecode"] = (
            item["start_timecode"]
        )

    if "end_timecode" in item:
        shifted["original_end_timecode"] = (
            item["end_timecode"]
        )

    return shifted


def create_shifted_timeline(paths, offset_seconds):
    timeline_data = load_json(paths["timeline"])
    timeline_list, list_key = get_timeline_list(
        timeline_data
    )

    shifted_list = [
        shift_item(item, offset_seconds)
        if isinstance(item, dict)
        else item
        for item in timeline_list
    ]

    if list_key is None:
        output = shifted_list
    else:
        output = dict(timeline_data)
        output[list_key] = shifted_list
        output["applied_offset_seconds"] = (
            offset_seconds
        )
        output["transcript_fps"] = TRANSCRIPT_FPS
        output["timecode_format"] = "HH:MM:SS:FF"
        output["offset_convention"] = (
            "cam3_time = transcript_time "
            "+ offset_seconds"
        )

    save_json(paths["shifted_timeline"], output)


# ============================================================
# LOAD TRACKING, TIMELINE, MAPPING
# ============================================================

def load_tracking(path):
    data = load_json(path)

    if isinstance(data, dict):
        frames = data.get("frames")
    elif isinstance(data, list):
        frames = data
    else:
        frames = None

    if not isinstance(frames, list):
        raise ValueError(
            "Tracking JSON has no supported frame list."
        )

    frame_map = {}

    for frame_item in frames:
        if not isinstance(frame_item, dict):
            continue

        frame_number = frame_item.get("frame")

        if frame_number is None:
            continue

        frame_map[int(frame_number)] = frame_item

    return frame_map


def load_segments(path):
    data = load_json(path)
    raw_segments, _ = get_timeline_list(data)

    segments = []

    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            continue

        start = first_value(item, START_KEYS)
        end = first_value(item, END_KEYS)
        speaker = first_value(item, SPEAKER_KEYS)
        text = first_value(item, TEXT_KEYS, "")

        if (
            start is None
            or end is None
            or speaker is None
        ):
            continue

        try:
            start_seconds = timecode_to_seconds(start)
            end_seconds = timecode_to_seconds(end)
        except (TypeError, ValueError) as exc:
            log(
                f"Skipping transcript segment "
                f"{index}: {exc}"
            )
            continue

        segments.append(
            {
                "segment_id": index,
                "start": start_seconds,
                "end": end_seconds,
                "speaker": str(speaker).strip().upper(),
                "text": str(text),
            }
        )

    segments.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    return segments


def normalize_track_ids(value):
    """
    Convert different track-ID representations into a flat list[int].

    Supported examples:
        3
        "3"
        [3]
        [3, 8]
        [[3], [8]]
        {"track_id": 3}
        None
    """
    output = []

    def collect(item):
        if item is None:
            return

        if isinstance(item, dict):
            for key in (
                "track_id",
                "track_ids",
                "id",
                "ids",
            ):
                if key in item:
                    collect(item[key])
                    return
            return

        if isinstance(item, (list, tuple, set)):
            for child in item:
                collect(child)
            return

        if isinstance(item, bool):
            return

        try:
            track_id = int(item)
        except (TypeError, ValueError):
            return

        if track_id not in output:
            output.append(track_id)

    collect(value)
    return output


def load_mapping(path):
    """
    Recommended:
    {
      "speaker_to_track_ids": {
        "A": [3, 8],
        "B": [1],
        "C": [5, 10],
        "R": []
      }
    }

    Old single-ID format is also supported.
    """
    data = load_json(path)

    if "speaker_to_track_ids" in data:
        raw_mapping = data["speaker_to_track_ids"]

    elif "speaker_to_track_id" in data:
        old_mapping = data["speaker_to_track_id"]

        raw_mapping = {
            speaker: (
                []
                if track_id is None
                else [track_id]
            )
            for speaker, track_id
            in old_mapping.items()
        }

    else:
        raw_mapping = data

    mapping = {}

    for speaker, track_ids in raw_mapping.items():
        speaker = str(speaker).strip().upper()

        mapping[speaker] = normalize_track_ids(
            track_ids
        )

    return mapping


def create_mapping_template(path):
    template = {
        "speaker_to_track_ids": {
            "A": [],
            "B": [],
            "C": [],
            "R": [],
        }
    }

    save_json(path, template)


def visible_track_for_speaker(
    speaker,
    mapping,
    people_by_track,
):
    for track_id in mapping.get(speaker, []):
        if track_id in people_by_track:
            return track_id

    return None


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def active_segments_at_time(
    segments,
    time_seconds,
    cursor,
):
    while (
        cursor < len(segments)
        and segments[cursor]["end"] < time_seconds
    ):
        cursor += 1

    active = []
    index = cursor

    while (
        index < len(segments)
        and segments[index]["start"] <= time_seconds
    ):
        segment = segments[index]

        if (
            segment["start"]
            <= time_seconds
            <= segment["end"]
        ):
            active.append(segment)

        index += 1

    return active, cursor


def clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = [
        int(round(float(value)))
        for value in bbox
    ]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def draw_label(
    frame,
    text,
    x,
    y,
    color,
    font_scale,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(
        1,
        int(round(font_scale * 2)),
    )

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )

    y = max(
        text_height + baseline + 8,
        y,
    )

    cv2.rectangle(
        frame,
        (
            x,
            y - text_height - baseline - 8,
        ),
        (
            min(
                frame.shape[1] - 1,
                x + text_width + 10,
            ),
            y + 3,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def wrap_text(text, max_chars=70):
    words = str(text).split()
    lines = []
    current = []

    for word in words:
        candidate = " ".join(current + [word])

        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))

            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines[:3]


def draw_subtitles(frame, active_segments, scale):
    if not active_segments:
        return

    height, width = frame.shape[:2]
    lines = []

    for segment in active_segments:
        speaker = segment["speaker"]

        for line in wrap_text(
            f"{speaker}: {segment['text']}"
        ):
            lines.append((speaker, line))

    line_height = int(
        34 * max(scale, 0.75)
    )

    padding = 12
    total_height = (
        line_height * len(lines)
        + padding * 2
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, max(0, height - total_height)),
        (width - 1, height - 1),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    y = (
        height
        - total_height
        + padding
        + line_height
        - 8
    )

    for speaker, line in lines:
        color = SPEAKER_COLORS.get(
            speaker,
            (255, 255, 255),
        )

        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, 0.75 * scale),
            color,
            max(1, int(round(2 * scale))),
            cv2.LINE_AA,
        )

        y += line_height


# ============================================================
# CREATE SPEAKER VISUALIZATION
# ============================================================

def create_speaker_visualization(paths):
    tracking = load_tracking(paths["tracking"])
    segments = load_segments(
        paths["shifted_timeline"]
    )
    mapping = load_mapping(paths["mapping"])

    log(
        f"{paths['video'].name}: loaded "
        f"{len(segments)} transcript segments"
    )

    if not segments:
        raise RuntimeError(
            "No valid transcript segments were loaded. "
            "Check start_timecode/end_timecode fields."
        )

    speaker_track_output = {
        "video": str(paths["video"]),
        "tracking_json": str(paths["tracking"]),
        "shifted_timeline": str(
            paths["shifted_timeline"]
        ),
        "mapping_json": str(paths["mapping"]),
        "speaker_to_track_ids": mapping,
        "segments": [
            {
                **segment,
                "candidate_track_ids": mapping.get(
                    segment["speaker"],
                    [],
                ),
            }
            for segment in segments
        ],
    }

    save_json(
        paths["speaker_track_json"],
        speaker_track_output,
    )

    capture = cv2.VideoCapture(
        str(paths["video"])
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {paths['video']}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(paths["speaker_video"]),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Cannot create output video: "
            f"{paths['speaker_video']}"
        )

    scale = max(width, height) / 1920.0
    timeline_cursor = 0
    frame_number = 0

    with open(
        paths["speaker_frames_csv"],
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "time_seconds",
                "speaker",
                "candidate_track_ids",
                "selected_track_id",
                "is_track_visible",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "text",
            ],
        )

        csv_writer.writeheader()

        try:
            progress = tqdm(
                total=(
                    total_frames
                    if total_frames > 0
                    else None
                ),
                desc=f"Visualize {paths['video'].stem}",
            )

            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                frame_number += 1
                time_seconds = (
                    frame_number - 1
                ) / fps

                active_segments, timeline_cursor = (
                    active_segments_at_time(
                        segments,
                        time_seconds,
                        timeline_cursor,
                    )
                )

                frame_tracking = tracking.get(
                    frame_number,
                    {"people": []},
                )

                people_by_track = {}

                for person in frame_tracking.get(
                    "people",
                    [],
                ):
                    raw_track_id = person.get(
                        "track_id"
                    )
                    bbox = person.get(
                        "head_bbox_xyxy"
                    )

                    track_ids = normalize_track_ids(
                        raw_track_id
                    )

                    if not track_ids or bbox is None:
                        continue

                    # Most files contain one ID, e.g. 3 or [3].
                    # If several IDs are present, index the same
                    # detection under every valid integer ID.
                    for track_id in track_ids:
                        people_by_track[
                            track_id
                        ] = person

                if SHOW_ALL_TRACKS:
                    for (
                        track_id,
                        person,
                    ) in people_by_track.items():

                        x1, y1, x2, y2 = clip_bbox(
                            person["head_bbox_xyxy"],
                            width,
                            height,
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            INACTIVE_COLOR,
                            max(
                                1,
                                int(round(2 * scale)),
                            ),
                        )

                        draw_label(
                            frame,
                            f"track {track_id}",
                            x1,
                            y1 - 5,
                            INACTIVE_COLOR,
                            max(
                                0.45,
                                0.58 * scale,
                            ),
                        )

                warning_index = 0

                for segment in active_segments:
                    speaker = segment["speaker"]

                    candidate_track_ids = (
                        mapping.get(speaker, [])
                    )

                    selected_track_id = (
                        visible_track_for_speaker(
                            speaker,
                            mapping,
                            people_by_track,
                        )
                    )

                    person = (
                        people_by_track.get(
                            selected_track_id
                        )
                        if selected_track_id
                        is not None
                        else None
                    )

                    color = SPEAKER_COLORS.get(
                        speaker,
                        (255, 255, 255),
                    )

                    bbox_values = [
                        "",
                        "",
                        "",
                        "",
                    ]

                    if person is not None:
                        x1, y1, x2, y2 = clip_bbox(
                            person["head_bbox_xyxy"],
                            width,
                            height,
                        )

                        bbox_values = [
                            x1,
                            y1,
                            x2,
                            y2,
                        ]

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            max(
                                3,
                                int(round(5 * scale)),
                            ),
                        )

                        draw_label(
                            frame,
                            (
                                f"SPEAKING: {speaker} "
                                f"| track "
                                f"{selected_track_id}"
                            ),
                            x1,
                            y1 - 5,
                            color,
                            max(
                                0.5,
                                0.68 * scale,
                            ),
                        )

                    else:
                        draw_label(
                            frame,
                            (
                                f"SPEAKING: {speaker} "
                                f"| candidates "
                                f"{candidate_track_ids} "
                                f"NOT VISIBLE"
                            ),
                            15,
                            75 + 38 * warning_index,
                            color,
                            max(
                                0.5,
                                0.65 * scale,
                            ),
                        )

                        warning_index += 1

                    csv_writer.writerow(
                        {
                            "frame": frame_number,
                            "time_seconds": (
                                f"{time_seconds:.6f}"
                            ),
                            "speaker": speaker,
                            "candidate_track_ids": (
                                json.dumps(
                                    candidate_track_ids
                                )
                            ),
                            "selected_track_id": (
                                ""
                                if selected_track_id
                                is None
                                else selected_track_id
                            ),
                            "is_track_visible": (
                                int(person is not None)
                            ),
                            "bbox_x1": bbox_values[0],
                            "bbox_y1": bbox_values[1],
                            "bbox_x2": bbox_values[2],
                            "bbox_y2": bbox_values[3],
                            "text": segment["text"],
                        }
                    )

                if SHOW_TIMECODE:
                    cv2.putText(
                        frame,
                        (
                            f"Frame {frame_number} "
                            f"| {time_seconds:.3f}s"
                        ),
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, 0.7 * scale),
                        (255, 255, 255),
                        max(
                            1,
                            int(round(2 * scale)),
                        ),
                        cv2.LINE_AA,
                    )

                if SHOW_SUBTITLES:
                    draw_subtitles(
                        frame,
                        active_segments,
                        scale,
                    )

                writer.write(frame)
                progress.update(1)

            progress.close()

        finally:
            capture.release()
            writer.release()


# ============================================================
# SUMMARY
# ============================================================

def write_summary(rows):
    output_path = (OUTPUT_ROOT / "speaker_pipeline_summary.csv")

    fields = [
        "video",
        "session",
        "trial",
        "offset_status",
        "offset_seconds",
        "correlation_score",
        "timeline_status",
        "visualization_status",
        "message",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):
    parsed = parse_video_name(video_path)

    row = {
        "video": str(video_path),
        "session": "",
        "trial": "",
        "offset_status": "",
        "offset_seconds": "",
        "correlation_score": "",
        "timeline_status": "",
        "visualization_status": "",
        "message": "",
    }

    if parsed is None:
        row["message"] = (
            "Could not parse video filename."
        )
        return row

    row["session"] = parsed["session"]
    row["trial"] = parsed["trial"]

    paths = get_paths(video_path, parsed)

    missing_required = []

    if not is_valid_file(
        paths["reference_audio"]
    ):
        missing_required.append(
            f"audio: {paths['reference_audio']}"
        )

    if not is_valid_file(paths["target_audio"]):
        missing_required.append(
            f"cam3 audio: {paths['target_audio']}"
        )

    if not is_valid_file(paths["timeline"]):
        missing_required.append(
            f"timeline: {paths['timeline']}"
        )

    if missing_required:
        row["message"] = (
            "Missing "
            + "; ".join(missing_required)
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

        return row

    # --------------------------------------------------------
    # OFFSET
    # --------------------------------------------------------
    try:
        if (
            is_valid_file(paths["offset"])
            and not FORCE_OFFSET
        ):
            offset_data = load_json(
                paths["offset"]
            )

            row["offset_status"] = (
                "skipped_existing"
            )

        else:
            log(
                f"{video_path.name}: "
                f"computing offset"
            )

            offset_data = create_offset_json(
                paths
            )

            row["offset_status"] = (
                "completed"
            )

        offset_seconds = float(
            offset_data["offset_seconds"]
        )

        row["offset_seconds"] = offset_seconds
        row["correlation_score"] = (
            offset_data.get(
                "correlation_score",
                "",
            )
        )

    except Exception as exc:
        row["offset_status"] = "failed"
        row["message"] = (
            f"Offset failed: "
            f"{type(exc).__name__}: {exc}"
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

        return row

    # --------------------------------------------------------
    # SHIFT TIMELINE
    # --------------------------------------------------------
    try:
        if (
            is_valid_file(
                paths["shifted_timeline"]
            )
            and not FORCE_SHIFTED_TIMELINE
        ):
            row["timeline_status"] = (
                "skipped_existing"
            )

        else:
            create_shifted_timeline(
                paths,
                offset_seconds,
            )

            row["timeline_status"] = (
                "completed"
            )

    except Exception as exc:
        row["timeline_status"] = "failed"
        row["message"] = (
            f"Timeline failed: "
            f"{type(exc).__name__}: {exc}"
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

        return row

    # --------------------------------------------------------
    # TRACKING / MAPPING
    # --------------------------------------------------------
    if not is_valid_file(paths["tracking"]):
        row["visualization_status"] = (
            "missing_tracking"
        )

        row["message"] = (
            f"Tracking JSON missing: "
            f"{paths['tracking']}"
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

        return row

    if not is_valid_file(paths["mapping"]):
        create_mapping_template(
            paths["mapping"]
        )

        row["visualization_status"] = (
            "mapping_template_created"
        )

        row["message"] = (
            f"Fill mapping template: "
            f"{paths['mapping']}"
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

        return row

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------
    try:
        outputs_exist = all(
            is_valid_file(path)
            for path in (
                paths["speaker_video"],
                paths["speaker_track_json"],
                paths["speaker_frames_csv"],
            )
        )

        if (
            outputs_exist
            and not FORCE_VISUALIZATION
        ):
            row["visualization_status"] = (
                "skipped_existing"
            )

        else:
            log(
                f"{video_path.name}: "
                f"creating visualization"
            )

            create_speaker_visualization(
                paths
            )

            row["visualization_status"] = (
                "completed"
            )

    except Exception as exc:
        row["visualization_status"] = (
            "failed"
        )

        row["message"] = (
            f"Visualization failed: "
            f"{type(exc).__name__}: {exc}"
        )

        log(
            f"{video_path.name}: "
            f"{row['message']}"
        )

    return row


# ============================================================
# MAIN
# ============================================================

def main():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available in PATH."
        )

    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(VIDEO_DIR)

    MAPPING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = find_videos()

    print("=" * 90)
    print("FURHAT SPEAKER PIPELINE")
    print("=" * 90)
    print(f"Root:     {ROOT}")
    print(f"Videos:   {VIDEO_DIR}")
    print(f"Tracking: {TRACKING_DIR}")
    print(f"Mappings: {MAPPING_DIR}")
    print(f"Count:    {len(videos)}")
    print("=" * 90)

    rows = []

    for index, video_path in enumerate(
        videos,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        row = process_video(video_path)
        rows.append(row)

        print(
            "  offset:",
            row["offset_status"],
        )
        print(
            "  timeline:",
            row["timeline_status"],
        )
        print(
            "  visualization:",
            row["visualization_status"],
        )

        if row["message"]:
            print(
                "  message:",
                row["message"],
            )

    summary_path = write_summary(rows)

    print()
    print("=" * 90)
    print("FINISHED")
    print("=" * 90)
    print(f"Summary: {summary_path}")
    print(
        f"Log:     "
        f"{OUTPUT_ROOT / 'speaker_pipeline.log'}"
    )
    print(f"Mappings:{MAPPING_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
