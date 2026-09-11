"""Batch gaze-target extraction for a 6cams mocap rig dataset.

Plug-and-play for new sessions: point --data-root at any dataset that follows
the same layout (<data_root>/<session_num>/Cam_<N>/*.avi) and it will process
every session it finds, skipping any missing a Cam_<N> video.

Usage:
    python run_batch_gaze_target.py
    python run_batch_gaze_target.py --data-root Data/03_24_03/6cams --camera 2
    python run_batch_gaze_target.py --sessions 1 2 3 --camera 3 --no-debug-video

Validated on camera 2 of Data/03_20/6cams (see memory: gaze_target_pipeline).
Camera 3 should work equally well since Gaze-LLE needs no camera calibration,
but hasn't been separately validated — spot-check its debug_overlay.mp4
before trusting results if you switch cameras.
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

_TOOL_SRC = Path(__file__).parent / "dataExtraction/facial_keypoints_extraction-master/src"
if str(_TOOL_SRC) not in sys.path:
    sys.path.insert(0, str(_TOOL_SRC))

from facial_keypoints_extraction.gaze_target_pipeline import PipelineConfig, run_feature_pipeline  # noqa: E402


def find_camera_videos(session_dir: Path, camera_num: int) -> list[Path]:
    """All parts of this camera's recording, in recorded order.

    Long takes hit AVI's ~4GB limit and are written as "<base>.avi",
    "<base> (1).avi", "<base> (2).avi" (05_14 only, so far). Plain sorted()
    puts " (1).avi" FIRST -- space (0x20) sorts before "." (0x2E) -- so the
    old candidates[0] silently extracted the MIDDLE of such a session and
    reported it as the whole thing. Mirrors datasetPrep/render_review_clips'
    _video_part_key; keep the two consistent.
    """
    cam_dir = session_dir / f"Cam_{camera_num}"
    if not cam_dir.is_dir():
        return []
    def part_key(p: Path) -> tuple[str, int]:
        m = re.match(r"^(?P<base>.*?)(?: \((?P<idx>\d+)\))?\.avi$", p.name)
        return (m.group("base"), int(m.group("idx") or 0)) if m else (p.name, 0)
    return sorted(
        (p for p in cam_dir.glob("*.avi") if not p.name.endswith(":Zone.Identifier")),
        key=part_key,
    )


def _merge_parts(output_dir: Path, part_dirs: list[Path], part_frames: list[int], fps: float) -> None:
    """Stitch per-part outputs into one session, renumbering frame_index (and
    timestamp_sec) so they run continuously across the whole session. Any
    downstream join -- continuous_labels, openface_cam2, mouth_motion -- keys
    on frame_index, so all of them must agree on this same numbering."""
    import pandas as pd

    frames_so_far = 0
    merged = []
    for part_dir, n_frames in zip(part_dirs, part_frames):
        df = pd.read_csv(part_dir / "frame_features.csv")
        df["frame_index"] = df["frame_index"] + frames_so_far
        df["timestamp_sec"] = df["frame_index"] / fps
        merged.append(df)
        frames_so_far += n_frames
    out = pd.concat(merged, ignore_index=True)
    out.to_csv(output_dir / "frame_features.csv", index=False)
    return frames_so_far


def process_session(session_dir: Path, camera_num: int, output_root: Path, save_debug_video: bool,
                     device: str = "cpu") -> None:
    import json

    session_num = session_dir.name
    parts = find_camera_videos(session_dir, camera_num)

    if not parts:
        print(f"[session {session_num}] Skipping — no Cam_{camera_num} video found.")
        return

    output_dir = output_root / f"session_{session_num}"

    if len(parts) == 1:
        print(f"[session {session_num}] Starting  video={parts[0]}")
        config = PipelineConfig(
            video_path=parts[0],
            output_dir=output_dir,
            save_debug_video=save_debug_video,
            device=device,
        )
        summary = run_feature_pipeline(config)
    else:
        # Run the pipeline once per part, then renumber and concatenate.
        # Cheaper and far less invasive than teaching the vendored pipeline
        # to read a multi-file stream. person_idx is assigned per-part, but
        # nothing downstream relies on it being stable across a boundary --
        # build_continuous_dataset re-derives identity per frame by
        # x-position rank (or session_tracker.py for swapped sessions), which
        # is exactly how within-video person_idx drift is already handled.
        print(f"[session {session_num}] Split across {len(parts)} parts, processing each then merging:")
        for p in parts:
            print(f"    {p.name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        part_dirs, part_frames, summaries = [], [], []
        for i, part in enumerate(parts):
            part_dir = output_dir / f"_part{i}"
            cfg = PipelineConfig(
                video_path=part,
                output_dir=part_dir,
                save_debug_video=save_debug_video,
                device=device,
            )
            summaries.append(run_feature_pipeline(cfg))
            part_dirs.append(part_dir)
            part_frames.append(summaries[-1]["stream"]["total_frames_in_video"])
            print(f"[session {session_num}]   part {i} done ({part_frames[-1]} frames)")

        fps = summaries[0]["stream"]["fps"]
        total = _merge_parts(output_dir, part_dirs, part_frames, fps)

        # One session-level metadata file describing the merged result, so
        # downstream readers (build_continuous_dataset reads frame_width/
        # frame_height from here) see the session, not part 0 alone.
        meta = {
            "input": {"video_parts": [str(p) for p in parts]},
            "stream": {
                "total_frames_in_video": total,
                "fps": fps,
                "frame_width": summaries[0]["stream"]["frame_width"],
                "frame_height": summaries[0]["stream"]["frame_height"],
            },
            "outputs": {"frame_features_csv": str(output_dir / "frame_features.csv")},
            "summary": {
                "processed_frames": sum(s["summary"]["processed_frames"] for s in summaries),
                "per_part": [s["summary"] for s in summaries],
            },
        }
        (output_dir / "session_metadata.json").write_text(json.dumps(meta, indent=2))
        summary = {"summary": {
            "processed_frames": meta["summary"]["processed_frames"],
            "robot_detection_rate": sum(s["summary"]["robot_detection_rate"] * s["summary"]["processed_frames"]
                                        for s in summaries) / max(1, meta["summary"]["processed_frames"]),
            "looking_at_robot_rate": sum(s["summary"]["looking_at_robot_rate"] * s["summary"]["processed_frames"]
                                         for s in summaries) / max(1, meta["summary"]["processed_frames"]),
            "looking_at_participant_rate": sum(s["summary"]["looking_at_participant_rate"] * s["summary"]["processed_frames"]
                                               for s in summaries) / max(1, meta["summary"]["processed_frames"]),
        }}

    s = summary["summary"]
    print(
        f"[session {session_num}] Done. frames={s['processed_frames']} "
        f"robot_detection_rate={s['robot_detection_rate']:.1%} "
        f"looking_at_robot_rate={s['looking_at_robot_rate']:.1%} "
        f"looking_at_participant_rate={s['looking_at_participant_rate']:.1%}"
    )
    print(f"[session {session_num}] Output -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default="Data/03_20/6cams", help="Path to the 6cams dataset root")
    parser.add_argument("--camera", type=int, default=2,
                         help="Which Cam_<N> to extract from (default: 2). NOTE: 05_15's 6cams folder "
                              "names are swapped vs. every other date (user-confirmed 2026-07-24) -- "
                              "the physical view every other date calls Cam_2 sits under Cam_1 for "
                              "05_15 specifically. Pass --camera 1 for that dataset, not the default.")
    parser.add_argument("--output-root", type=str, default=None, help="Output directory (default: outputs/gaze_target_<data-root-name>)")
    parser.add_argument("--sessions", type=int, nargs="*", default=None, help="Specific session numbers to process (default: all found)")
    parser.add_argument("--no-debug-video", action="store_true", help="Skip writing debug_overlay.mp4 (faster)")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda. Gaze-LLE is ~100x faster on GPU "
                                                        "for 2048x1088 frames; needs a CUDA-enabled torch in the venv.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"Data root not found: {data_root}")

    output_root = Path(args.output_root) if args.output_root else Path("outputs") / f"gaze_target_{data_root.parent.name}"

    if args.sessions is not None:
        session_dirs = [data_root / str(n) for n in args.sessions]
    else:
        session_dirs = sorted((p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name)

    print(f"Data root  : {data_root}")
    print(f"Camera     : Cam_{args.camera}")
    print(f"Output root: {output_root}")
    print()

    for session_dir in session_dirs:
        try:
            process_session(session_dir, args.camera, output_root, save_debug_video=not args.no_debug_video,
                             device=args.device)
        except Exception:
            print(f"[session {session_dir.name}] ERROR — see traceback below:")
            traceback.print_exc()
        print()

    print("Batch complete.")


if __name__ == "__main__":
    main()
