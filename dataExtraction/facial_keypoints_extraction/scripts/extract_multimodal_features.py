from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facial_keypoints_extraction.pipeline import PipelineConfig, run_feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract multi-party face, gaze, and aligned audio features from a ROS bag session.",
    )
    parser.add_argument("--bag", required=False, default=None, type=Path, help="Path to the ROS1 .bag file.")
    parser.add_argument("--video", type=Path, default=None, help="Path to a plain video file (.avi/.mp4) as alternative to --bag.")
    parser.add_argument("--wav", type=Path, default=None, help="Optional path to a synced WAV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV / NPZ / JSON / debug video outputs will be written.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "face_landmarker.task",
        help="Path to the MediaPipe face landmarker model bundle.",
    )
    parser.add_argument("--frame-step", type=int, default=3, help="Process every Nth frame. Default: 3.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames.")
    parser.add_argument("--max-tracks", type=int, default=3, help="Maximum number of participants to track.")
    parser.add_argument(
        "--redetect-interval",
        type=int,
        default=5,
        help="Run the cascade detector every N processed frames. Default: 5.",
    )
    parser.add_argument(
        "--audio-offset-sec",
        type=float,
        default=0.0,
        help="Shift applied to WAV timestamps relative to video timestamps.",
    )
    parser.add_argument(
        "--save-debug-video",
        action="store_true",
        help="Write an overlay video for quick visual quality checking.",
    )
    parser.add_argument("--color-topic", type=str, default=None, help="Override the RGB image topic.")
    parser.add_argument(
        "--camera-info-topic",
        type=str,
        default=None,
        help="Override the RGB camera_info topic.",
    )
    parser.add_argument(
        "--robot-camera-yaw",
        type=float,
        default=33.7,
        help=(
            "Expected yaw angle (degrees) when participant looks at the robot camera. "
            "Computed from rig geometry: arctan2(right_offset, participant_dist + back_offset). "
            "Positive = participant turns left (camera operator's right). Default: 33.7"
        ),
    )
    parser.add_argument(
        "--robot-camera-pitch",
        type=float,
        default=0.0,
        help="Expected pitch angle (degrees) when looking at robot camera. Default: 0.0",
    )
    parser.add_argument(
        "--gaze-tolerance",
        type=float,
        default=15.0,
        help="Angular tolerance (degrees) around robot camera angles for 'camera' label. Default: 15.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bag is None and args.video is None:
        print("Error: one of --bag or --video must be provided.", file=sys.stderr)
        sys.exit(1)
    if args.bag is not None and args.video is not None:
        print("Error: --bag and --video are mutually exclusive; provide only one.", file=sys.stderr)
        sys.exit(1)
    config = PipelineConfig(
        bag_path=args.bag,
        video_path=args.video,
        output_dir=args.output_dir,
        model_path=args.model,
        wav_path=args.wav,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        max_tracks=args.max_tracks,
        redetect_interval=args.redetect_interval,
        save_debug_video=args.save_debug_video,
        audio_offset_sec=args.audio_offset_sec,
        color_topic=args.color_topic,
        camera_info_topic=args.camera_info_topic,
        robot_camera_yaw_deg=args.robot_camera_yaw,
        robot_camera_pitch_deg=args.robot_camera_pitch,
        gaze_tolerance_deg=args.gaze_tolerance,
    )
    summary = run_feature_pipeline(config)
    print("Extraction finished.")
    print(f"Processed frames: {summary['summary']['processed_frames']}")
    print(f"Frames with any detected face: {summary['summary']['frames_with_any_face']}")
    print(f"CSV: {summary['outputs']['frame_features_csv']}")
    print(f"NPZ: {summary['outputs']['landmarks_npz']}")
    if summary["outputs"]["debug_video"]:
        print(f"Debug video: {summary['outputs']['debug_video']}")


if __name__ == "__main__":
    main()
