"""Simple playback tool for pre-rendered debug overlay videos (already have
boxes/arrows/labels baked in, e.g. gazelle_validation_full.mp4).

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
import sys

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a pre-rendered debug overlay video.")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video_path}")
    print(f"{total_frames} frames @ {native_fps:.1f} fps")
    print("Controls: SPACE=pause/play  q/ESC=quit  LEFT/RIGHT=step  +/-=speed")

    paused = False
    speed = 1.0
    frame_idx = 0
    cv2.namedWindow("Debug Video", cv2.WINDOW_NORMAL)

    while 0 <= frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)

        status = "PAUSED" if paused else f"PLAY x{speed:.1f}"
        cv2.putText(frame, f"[{status}]  frame {frame_idx}/{total_frames}", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Debug Video", frame)

        delay_ms = max(1, int(1000 / (native_fps * speed)))
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83 or key == ord("d"):
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == 81 or key == ord("a"):
            frame_idx = max(frame_idx - 1, 0)
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 1.5, 8.0)
        elif key == ord("-"):
            speed = max(speed / 1.5, 0.25)
        elif not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
