"""Render short raw-footage clips for human-review of needs_review segments.

Deliberately extracts from RAW source video, not debug_overlay.mp4 -- a
reviewer judging engagement shouldn't be biased by the model's own drawn
gaze arrows/boxes. Per-row source choice: a row with a track_id (an
individual, facial_keypoints-covered segment) is best judged from the RGB
face camera (participants_rgb.avi); a group-level row (no track_id, only
gaze_target/mocap coverage) can only usefully be judged from the wide
mocap Cam_2 view.

Track-id rows draw a plain identity box (from that track's own bbox_x0..y1
in frame_features.csv) around the person to score -- participants_rgb.avi
is a single wide shot covering everyone in session, so without this a
segment with 3 tracked people would render 3 byte-identical clips with no
way to tell who track_0/1/2 even are. This is NOT a gaze/engagement
overlay (nothing about the model's prediction is drawn) so it doesn't
reintroduce the bias raw-footage extraction was meant to avoid -- it only
answers "which person," never "what the model thinks about them."

Uses cv2 VideoWriter with mp4v codec -- the same approach already proven
in this project for debug_overlay.mp4/highlight_clip.mp4, since ffmpeg
isn't installed in this environment.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"


def _find_participants_rgb(dataset: str, session: int) -> Optional[Path]:
    for processed_dir_name in ("Processed", "processed"):
        for candidate in (
            DATA_ROOT / dataset / processed_dir_name / str(session) / "video" / "participants_rgb.avi",
            DATA_ROOT / dataset / processed_dir_name / str(session) / "participants_rgb.avi",
        ):
            if candidate.exists():
                return candidate
    return None


# 05_15's and 05_14's 6cams folder names are swapped vs. every other date
# (both user-confirmed -- 05_15 on 2026-07-24, 05_14 on 2026-08-02): the
# physical view every other date calls "Cam_2" sits under "Cam_1" for these
# dates, and their "Cam_2" folder holds what other dates would call "Cam_3".
# Do not remove these overrides without re-confirming the swap is still true
# -- it's an upstream data quirk, not something to "clean up". Verified for
# 05_14 by eye (Cam_1 = the familiar wide mocap framing at 2048x1088, the
# same resolution every other date's true Cam_2 uses; its "Cam_2" folder is
# a different, narrower 1280x1024 view).
_CAM2_FOLDER_OVERRIDE = {"05_15": "Cam_1", "05_14": "Cam_1"}


def _video_part_key(path: Path) -> tuple[str, int]:
    """Order a split recording's parts the way they were actually recorded.

    Long takes hit AVI's ~4GB limit and get written as
    "<base>.avi", "<base> (1).avi", "<base> (2).avi" (05_14 only, so far).
    A plain sorted() puts "<base> (1).avi" FIRST, because " " (0x20) sorts
    before "." (0x2E) -- so the naive candidates[0] silently returns the
    MIDDLE of the session as if it were the whole thing. Confirmed these
    really are continuations, not duplicate takes: 05_14 session 2's parts
    sum to exactly the frame count of the same session's other camera, and
    part 0's last frame vs part 1's first frame differ by a mean of only
    2.3/255 (i.e. consecutive moments), not a scene cut.
    """
    m = re.match(r"^(?P<base>.*?)(?: \((?P<idx>\d+)\))?\.avi$", path.name)
    if not m:
        return (path.name, 0)
    return (m.group("base"), int(m.group("idx") or 0))


def find_cam2_parts(dataset: str, session: int) -> list[Path]:
    """Every part of this session's Cam_2 recording, in recorded order.
    Single-file sessions (every date except some of 05_14's) return a
    one-element list."""
    cam_folder = _CAM2_FOLDER_OVERRIDE.get(dataset, "Cam_2")
    for cam_dir in (
        DATA_ROOT / dataset / "6cams" / str(session) / cam_folder,
        DATA_ROOT / dataset / str(session) / cam_folder,  # 03_26-style: no 6cams/ nesting
    ):
        if cam_dir.is_dir():
            candidates = sorted(
                (p for p in cam_dir.glob("*.avi") if not p.name.endswith(":Zone.Identifier")),
                key=_video_part_key,
            )
            if candidates:
                return candidates
    return []


def _find_cam2_raw(dataset: str, session: int) -> Optional[Path]:
    """FIRST part only. Callers that must cover a whole session should use
    find_cam2_parts()/MultiPartCapture instead -- this returns a partial
    recording for any split session."""
    parts = find_cam2_parts(dataset, session)
    return parts[0] if parts else None


class MultiPartCapture:
    """Presents a session's split .avi parts as ONE continuous stream, so
    callers index frames across the whole session instead of restarting at 0
    on each part. Drop-in for the cv2.VideoCapture subset this project uses
    (isOpened/get/set POS_FRAMES/read/release); single-part sessions behave
    exactly as before.

    Frame numbering is the concatenation: part 0 keeps its own indices, part
    1 starts at len(part 0), and so on. Any pipeline reading this session
    (labels, gaze_target, openface) must use the same convention or the
    frame_index joins between them break."""

    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError("MultiPartCapture needs at least one path")
        self.paths = list(paths)
        self.counts: list[int] = []
        for p in self.paths:
            c = cv2.VideoCapture(str(p))
            if not c.isOpened():
                raise SystemExit(f"could not open {p}")
            self.counts.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
            if not self.counts[:-1]:
                self._fps = c.get(cv2.CAP_PROP_FPS) or 30.0
                self._w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c.release()
        self.offsets = []
        run = 0
        for n in self.counts:
            self.offsets.append(run)
            run += n
        self.total = run
        self._part = 0
        self._cap = cv2.VideoCapture(str(self.paths[0]))
        self._pos = 0

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total)
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return self._cap.get(prop)

    def _open_part(self, part: int) -> None:
        if part != self._part or self._cap is None:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(str(self.paths[part]))
            self._part = part

    def set(self, prop, value) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return self._cap.set(prop, value)
        target = int(value)
        if target >= self.total:
            self._pos = self.total
            return False
        part = max(i for i, off in enumerate(self.offsets) if off <= target)
        self._open_part(part)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target - self.offsets[part])
        self._pos = target
        return True

    def grab(self) -> bool:
        """Advance one frame WITHOUT decoding it (~3.2ms vs ~4.9ms for
        read() on this footage). Used to step over frames that won't be
        displayed, so high-fps sessions can keep real-time pace."""
        if self._pos >= self.total:
            return False
        ok = self._cap.grab()
        if not ok:
            if self._part + 1 < len(self.paths):
                self._open_part(self._part + 1)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok = self._cap.grab()
            if not ok:
                return False
        self._pos += 1
        return True

    def read(self):
        if self._pos >= self.total:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            # End of this part -- roll over to the next one rather than
            # reporting end-of-session.
            if self._part + 1 < len(self.paths):
                self._open_part(self._part + 1)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                return False, None
        self._pos += 1
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def open_cam2_capture(dataset: str, session: int) -> Optional[MultiPartCapture]:
    parts = find_cam2_parts(dataset, session)
    return MultiPartCapture(parts) if parts else None


def _load_track_bboxes(dataset: str, session: int, track_id: int) -> dict[int, tuple[float, float, float, float]]:
    fk_csv = PROJECT_ROOT / "outputs" / dataset / "facial_keypoints" / f"session_{session}" / "frame_features.csv"
    if not fk_csv.exists():
        return {}
    df = pd.read_csv(
        fk_csv,
        usecols=["frame_index", "track_id", "detected", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"],
    )
    sub = df[(df["track_id"] == track_id) & (df["detected"] == 1)].dropna(
        subset=["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
    )
    return {
        int(r.frame_index): (r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1)
        for r in sub.itertuples()
    }


def render_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    track_bboxes: Optional[dict[int, tuple[float, float, float, float]]] = None,
) -> tuple[bool, bool]:
    """Returns (wrote_any, any_box_in_range). any_box_in_range is only meaningful
    when track_bboxes is not None -- False means this track had zero detections
    anywhere in [start_sec, end_sec), so no identity box could be drawn at all."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames, int(end_sec * fps))
    if start_frame >= end_frame:
        cap.release()
        return False, False

    any_box_in_range = (
        any(start_frame <= f < end_frame for f in track_bboxes) if track_bboxes is not None else False
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = start_frame
    wrote_any = False
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if track_bboxes is not None:
            box = track_bboxes.get(frame_idx)
            if box is not None:
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    frame, "SCORE THIS PERSON", (x0, max(15, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
                )
            elif not any_box_in_range:
                cv2.putText(
                    frame, "NO DETECTION IN THIS WINDOW", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )
        writer.write(frame)
        wrote_any = True
        frame_idx += 1

    cap.release()
    writer.release()
    return wrote_any, any_box_in_range


def render_review_queue(dataset: str, labeled_df: pd.DataFrame) -> pd.DataFrame:
    review_rows = labeled_df[labeled_df["needs_review"]].copy()
    clips_dir = PROJECT_ROOT / "outputs" / dataset / "review_clips"
    queue_path = PROJECT_ROOT / "outputs" / dataset / "review_queue.csv"

    # Re-rendering (e.g. after a rendering-logic fix) must never silently wipe
    # out human labels already saved in an existing queue file.
    prior_labels: dict[tuple[int, int, str], str] = {}
    if queue_path.exists():
        prior_df = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
        for _, prow in prior_df.iterrows():
            if prow["human_engagement_level"]:
                key = (int(prow["session"]), int(prow["segment_idx"]), prow["track_id"])
                prior_labels[key] = prow["human_engagement_level"]

    queue_rows = []
    for _, row in review_rows.iterrows():
        session = int(row["session"])
        segment_idx = int(row["segment_idx"])
        track_id = row.get("track_id")
        has_track = pd.notna(track_id)

        track_bboxes = None
        if has_track:
            video_path = _find_participants_rgb(dataset, session)
            clip_name = f"session{session}_seg{segment_idx}_track{int(track_id)}.mp4"
            track_bboxes = _load_track_bboxes(dataset, session, int(track_id))
        else:
            video_path = _find_cam2_raw(dataset, session)
            clip_name = f"session{session}_seg{segment_idx}.mp4"

        clip_path = clips_dir / clip_name
        rendered = False
        any_box_in_range = False
        if video_path is not None:
            rendered, any_box_in_range = render_clip(
                video_path, row["segment_start_sec"], row["segment_end_sec"], clip_path, track_bboxes
            )

        if not rendered:
            note = "MISSING SOURCE VIDEO — clip not rendered"
        elif has_track and not any_box_in_range:
            note = "NO DETECTION IN WINDOW — cannot confirm identity, treat with lower trust"
        else:
            note = ""

        track_key = str(int(track_id)) if has_track else ""
        prior_label = prior_labels.get((session, segment_idx, track_key), "")

        queue_rows.append({
            "dataset": dataset,
            "session": session,
            "segment_idx": segment_idx,
            "track_id": int(track_id) if has_track else "",
            "clip_path": str(clip_path.relative_to(PROJECT_ROOT)) if rendered else "",
            "auto_engagement_level": row["engagement_level"],
            "confidence": row["engagement_confidence"],
            "human_engagement_level": prior_label,
            "notes": note,
        })

    queue_df = pd.DataFrame(queue_rows)
    queue_df.to_csv(queue_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return queue_df
