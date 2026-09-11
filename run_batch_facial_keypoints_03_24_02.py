"""Batch facial keypoints extraction for 03_24_02 participant sessions.

Processes sessions 1-6 under Data/03_24_02/, skipping any that lack
a participants_rgb.avi file.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "Data/03_24_02"
OUTPUT_ROOT = Path(__file__).parent / "outputs/facial_keypoints_03_24_02"

_TOOL_SRC = Path(__file__).parent / "dataExtraction/facial_keypoints_extraction-master/src"
if str(_TOOL_SRC) not in sys.path:
    sys.path.insert(0, str(_TOOL_SRC))

_MODEL_PATH = (
    Path(__file__).parent
    / "dataExtraction/facial_keypoints_extraction-master/models/face_landmarker.task"
)

from facial_keypoints_extraction.pipeline import PipelineConfig, run_feature_pipeline  # noqa: E402


def process_session(session_num: int) -> None:
    session_dir = DATA_ROOT / str(session_num)
    video_path = session_dir / "video" / "participants_rgb.avi"

    if not video_path.exists():
        print(f"[session {session_num}] Skipping — participants_rgb.avi not found.")
        return

    wav_path = session_dir / "audio" / "aligned_audio" / "master_aligned.wav"
    wav = wav_path if wav_path.exists() else None

    output_dir = OUTPUT_ROOT / f"session_{session_num}"

    print(f"[session {session_num}] Starting  video={video_path}")
    if wav:
        print(f"[session {session_num}]            audio={wav}")
    else:
        print(f"[session {session_num}]            audio=None (file not found)")

    config = PipelineConfig(
        output_dir=output_dir,
        model_path=_MODEL_PATH,
        wav_path=wav,
        frame_step=3,
        start_frame=0,
        max_frames=None,
        max_tracks=3,
        redetect_interval=5,
        save_debug_video=False,
        audio_offset_sec=0.0,
        video_path=video_path,
    )

    summary = run_feature_pipeline(config)
    proc = summary["summary"]["processed_frames"]
    faces = summary["summary"]["frames_with_any_face"]
    rate = summary["summary"]["face_detection_rate"]
    print(
        f"[session {session_num}] Done. "
        f"frames={proc}, faces={faces}, detection_rate={rate:.1%}"
    )
    print(f"[session {session_num}] Output -> {output_dir}")


def main() -> None:
    print(f"Data root : {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Model      : {_MODEL_PATH}")
    print()

    for n in range(1, 7):
        try:
            process_session(n)
        except Exception:
            print(f"[session {n}] ERROR — see traceback below:")
            traceback.print_exc()
        print()

    print("Batch complete.")


if __name__ == "__main__":
    main()
