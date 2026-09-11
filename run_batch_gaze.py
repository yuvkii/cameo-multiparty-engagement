"""Batch MCGaze gaze estimation for all participant sessions.

Processes sessions 1-6 under Data/03_20/Processed/, skipping any that lack
a participants_rgb.avi file. Output goes to outputs/gaze/session_{n}/.

Usage:
    python run_batch_gaze.py [extra args passed through to run_gaze_on_video.py]

Extra args examples:
    python run_batch_gaze.py --frame-step 3 --device cpu
    python run_batch_gaze.py --save-debug-video --max-frames 500
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Data" / "03_20" / "Processed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gaze"
GAZE_SCRIPT = PROJECT_ROOT / "dataExtraction" / "MCGaze-master" / "run_gaze_on_video.py"

# Default checkpoint paths (relative to MCGaze-master)
MCGAZE_ROOT = PROJECT_ROOT / "dataExtraction" / "MCGaze-master"
DEFAULT_CONFIG = str(MCGAZE_ROOT / "configs" / "multiclue_gaze" / "multiclue_gaze_r50_l2cs.py")
DEFAULT_CHECKPOINT = str(MCGAZE_ROOT / "ckpts" / "multiclue_gaze_r50_l2cs.pth")
DEFAULT_HEAD_CHECKPOINT = str(MCGAZE_ROOT / "MCGaze_demo" / "yolo_head" / "crowdhuman_yolov5m.pt")


def process_session(session_num: int, extra_args: list[str]) -> None:
    session_dir = DATA_ROOT / str(session_num)
    video_path = session_dir / "video" / "participants_rgb.avi"

    if not video_path.exists():
        print(f"[session {session_num}] Skipping — participants_rgb.avi not found.")
        return

    output_dir = OUTPUT_ROOT / f"session_{session_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[session {session_num}] Starting  video={video_path}")
    print(f"[session {session_num}]            output={output_dir}")

    cmd = [
        sys.executable,
        str(GAZE_SCRIPT),
        "--video", str(video_path),
        "--output-dir", str(output_dir),
        "--config", DEFAULT_CONFIG,
        "--checkpoint", DEFAULT_CHECKPOINT,
        "--head-checkpoint", DEFAULT_HEAD_CHECKPOINT,
    ] + extra_args

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"[session {session_num}] Done. Output -> {output_dir}")
    else:
        print(
            f"[session {session_num}] ERROR — run_gaze_on_video.py exited with code {result.returncode}."
        )


def main() -> None:
    # Any args after the script name are forwarded to run_gaze_on_video.py
    # (e.g. --frame-step, --device, --save-debug-video, --max-frames)
    extra_args = sys.argv[1:]

    print(f"Data root    : {DATA_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Gaze script  : {GAZE_SCRIPT}")
    if extra_args:
        print(f"Extra args   : {' '.join(extra_args)}")
    print()

    for n in range(1, 7):
        try:
            process_session(n, extra_args)
        except Exception:
            print(f"[session {n}] ERROR — see traceback below:")
            traceback.print_exc()
        print()

    print("Batch complete.")


if __name__ == "__main__":
    main()
