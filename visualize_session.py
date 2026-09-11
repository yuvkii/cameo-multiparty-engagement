"""Interactive session visualizer.

Reads frame_features.csv and the original video to overlay detections,
gaze targets, head pose, and MAR in real time.

Usage:
    # Point at a session output directory (reads session_metadata.json automatically):
    python visualize_session.py --session-dir outputs/facial_keypoints_03_24_01/session_1

    # Or supply paths explicitly:
    python visualize_session.py --video Data/03_24_01/1/video/participants_rgb.avi \
                                --csv outputs/facial_keypoints_03_24_01/session_1/frame_features.csv

Controls:
    SPACE       pause / resume
    q / ESC     quit
    RIGHT       step forward one frame  (when paused)
    LEFT        step back one frame     (when paused)
    +           speed up playback
    -           slow down playback
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Match the pipeline's track colours exactly
TRACK_COLORS = [
    (83, 214, 110),   # track 0 — green
    (74, 163, 255),   # track 1 — blue
    (255, 173, 82),   # track 2 — orange
]

GAZE_TARGET_COLORS = {
    "camera":    (0, 255, 255),   # yellow  → looking at robot
    "elsewhere": (0, 0, 255),     # red     → looking away
    "missing":   (80, 80, 80),    # grey    → not detected
}


def load_csv(csv_path: Path) -> dict[int, list[dict]]:
    """Return {frame_index: [row, ...]} — one row per track per frame."""
    frames: dict[int, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fi = int(row["frame_index"])
            frames.setdefault(fi, []).append(row)
    return frames


def safe_float(val: str) -> float | None:
    try:
        v = float(val)
        return None if (v != v) else v   # NaN check
    except (ValueError, TypeError):
        return None


def draw_frame_overlay(frame: np.ndarray, rows: list[dict], frame_index: int,
                       timestamp: float, paused: bool, speed: float) -> np.ndarray:
    h, w = frame.shape[:2]
    detected_count = sum(1 for r in rows if r.get("detected") == "1")

    # ── header bar ──────────────────────────────────────────────────────────
    status = "PAUSED" if paused else f"PLAY x{speed:.1f}"
    cv2.putText(frame, f"frame {frame_index}  t={timestamp:.2f}s  faces={detected_count}  [{status}]",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    for row in rows:
        if row.get("detected") != "1":
            continue

        tid = int(row["track_id"])
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        gaze_target = row.get("gaze_target", "missing")

        # ── bounding box ────────────────────────────────────────────────────
        x0 = safe_float(row.get("bbox_x0", "nan"))
        y0 = safe_float(row.get("bbox_y0", "nan"))
        x1 = safe_float(row.get("bbox_x1", "nan"))
        y1 = safe_float(row.get("bbox_y1", "nan"))
        if None in (x0, y0, x1, y1):
            continue
        ix0, iy0, ix1, iy1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), color, 2)

        # ── gaze target indicator (coloured dot on box corner) ───────────────
        dot_color = GAZE_TARGET_COLORS.get(gaze_target, (200, 200, 200))
        cv2.circle(frame, (ix0 + 6, iy0 + 6), 6, dot_color, -1, cv2.LINE_AA)

        # ── label ───────────────────────────────────────────────────────────
        mar = safe_float(row.get("mar", "nan"))
        yaw = safe_float(row.get("head_yaw_deg", "nan"))
        pitch = safe_float(row.get("head_pitch_deg", "nan"))

        mar_str  = f"{mar:.2f}" if mar is not None else "?"
        yaw_str  = f"{yaw:+.0f}°" if yaw is not None else "?"
        pit_str  = f"{pitch:+.0f}°" if pitch is not None else "?"

        label_lines = [
            f"T{tid}  gaze:{gaze_target}",
            f"MAR:{mar_str}  yaw:{yaw_str}  pitch:{pit_str}",
        ]
        label_y = max(40, iy0 - 24)
        for i, line in enumerate(label_lines):
            cv2.putText(frame, line, (ix0, label_y + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # ── gaze arrow ──────────────────────────────────────────────────────
        gvx = safe_float(row.get("gaze_vector_x", "nan"))
        gvy = safe_float(row.get("gaze_vector_y", "nan"))
        if gvx is not None and gvy is not None:
            cx = (ix0 + ix1) // 2
            cy = (iy0 + iy1) // 2
            arrow_len = max(ix1 - ix0, iy1 - iy0) * 0.8
            tip = (int(cx + gvx * arrow_len), int(cy + gvy * arrow_len))
            cv2.arrowedLine(frame, (cx, cy), tip, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)

    # ── legend ──────────────────────────────────────────────────────────────
    legend_items = [
        ("camera",    GAZE_TARGET_COLORS["camera"],    "looking at robot"),
        ("elsewhere", GAZE_TARGET_COLORS["elsewhere"], "looking away"),
    ]
    for i, (label, lcolor, desc) in enumerate(legend_items):
        lx, ly = w - 200, h - 40 + i * 18
        cv2.circle(frame, (lx, ly), 5, lcolor, -1)
        cv2.putText(frame, desc, (lx + 12, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lcolor, 1, cv2.LINE_AA)

    return frame


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.session_dir:
        session_dir = Path(args.session_dir)
        meta_path = session_dir / "session_metadata.json"
        if not meta_path.exists():
            sys.exit(f"No session_metadata.json found in {session_dir}")
        with open(meta_path) as fh:
            meta = json.load(fh)
        video_path = Path(meta["input"]["bag_path"])
        csv_path = session_dir / "frame_features.csv"
    else:
        video_path = Path(args.video)
        csv_path = Path(args.csv)

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    return video_path, csv_path


def run(video_path: Path, csv_path: Path, window_scale: float) -> None:
    print(f"Video : {video_path}")
    print(f"CSV   : {csv_path}")
    print("Loading CSV…")
    frames_data = load_csv(csv_path)
    sorted_frame_indices = sorted(frames_data.keys())
    print(f"Loaded {len(sorted_frame_indices)} processed frames.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {total_video_frames} frames @ {native_fps:.1f} fps")
    print()
    print("Controls: SPACE=pause/play  q/ESC=quit  LEFT/RIGHT=step  +/-=speed")

    paused = False
    speed = 1.0
    cursor = 0   # index into sorted_frame_indices

    cv2.namedWindow("Session Visualizer", cv2.WINDOW_NORMAL)

    while cursor < len(sorted_frame_indices):
        frame_index = sorted_frame_indices[cursor]
        rows = frames_data.get(frame_index, [])

        # Seek video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_index}, skipping.")
            cursor += 1
            continue

        timestamp = frame_index / native_fps

        if window_scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * window_scale), int(h * window_scale)),
                               interpolation=cv2.INTER_LINEAR)
            # Scale bbox coordinates in rows to match
            scaled_rows = []
            for r in rows:
                sr = dict(r)
                for key in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
                    v = safe_float(sr.get(key, "nan"))
                    sr[key] = str(v * window_scale) if v is not None else "nan"
                scaled_rows.append(sr)
            rows = scaled_rows

        overlay = draw_frame_overlay(frame.copy(), rows, frame_index, timestamp, paused, speed)
        cv2.imshow("Session Visualizer", overlay)

        delay_ms = max(1, int(1000 / (native_fps * speed / 3)))  # frame_step=3
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key in (ord("q"), 27):   # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83 or key == ord("d"):   # RIGHT arrow or d
            cursor = min(cursor + 1, len(sorted_frame_indices) - 1)
        elif key == 81 or key == ord("a"):   # LEFT arrow or a
            cursor = max(cursor - 1, 0)
        elif key == ord("+") or key == ord("="):
            speed = min(speed * 1.5, 8.0)
        elif key == ord("-"):
            speed = max(speed / 1.5, 0.25)
        elif not paused:
            cursor += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive session visualizer.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir", type=str,
                       help="Path to a session output directory (reads metadata automatically).")
    group.add_argument("--video", type=str, help="Path to video file.")
    parser.add_argument("--csv", type=str, help="Path to frame_features.csv (required with --video).")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Window scale factor, e.g. 0.5 to halve the size. Default: 1.0")
    args = parser.parse_args()

    if args.video and not args.csv:
        parser.error("--csv is required when using --video.")

    video_path, csv_path = resolve_paths(args)
    run(video_path, csv_path, window_scale=args.scale)


if __name__ == "__main__":
    main()
