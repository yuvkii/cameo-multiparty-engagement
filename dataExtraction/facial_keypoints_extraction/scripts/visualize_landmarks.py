from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facial_keypoints_extraction.visualization import VisualizationConfig, render_landmark_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render extracted face landmarks back onto the original ROS bag frames.",
    )
    parser.add_argument("--bag", required=True, type=Path, help="Path to the ROS1 bag file.")
    parser.add_argument("--npz", required=True, type=Path, help="Path to landmarks_and_blendshapes.npz.")
    parser.add_argument("--output", required=True, type=Path, help="Output video path.")
    parser.add_argument("--color-topic", type=str, default=None, help="Optional RGB topic override.")
    parser.add_argument(
        "--camera-info-topic",
        type=str,
        default=None,
        help="Optional camera info topic override.",
    )
    parser.add_argument(
        "--render-stride",
        type=int,
        default=1,
        help="Render every Nth row from the NPZ file to keep the video small.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to render.")
    parser.add_argument("--start-row", type=int, default=0, help="Row offset inside the NPZ file.")
    parser.add_argument(
        "--tracks",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of track ids to render, e.g. --tracks 0 1 2",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Draw face mesh connections only, without point dots.",
    )
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="Draw only bounding boxes and labels, without landmarks.",
    )
    parser.add_argument("--point-radius", type=int, default=1, help="Point radius for landmark dots.")
    parser.add_argument("--line-thickness", type=int, default=1, help="Line thickness for mesh edges.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_landmark_video(
        VisualizationConfig(
            bag_path=args.bag,
            npz_path=args.npz,
            output_path=args.output,
            color_topic=args.color_topic,
            camera_info_topic=args.camera_info_topic,
            render_stride=args.render_stride,
            max_frames=args.max_frames,
            start_row=args.start_row,
            draw_full_mesh=not args.bbox_only,
            draw_points=not args.mesh_only and not args.bbox_only,
            point_radius=args.point_radius,
            line_thickness=args.line_thickness,
            draw_bbox=True,
            draw_labels=True,
            only_tracks=args.tracks,
        )
    )
    print("Landmark visualization finished.")
    print(f"Rendered frames: {summary['rendered_frames']}")
    print(f"Output video: {summary['output_path']}")


if __name__ == "__main__":
    main()
