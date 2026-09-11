"""Interactive viewer for auto-labeled segments -- plays each
render_auto_review_clips.py clip (predicted engagement class drawn per
person, color-coded) on loop so the auto-labeler's output can actually be
eyeballed against the footage.

Read-only with respect to the labels themselves (this tool doesn't relabel
anything) -- its only write is a flag, for segments whose predicted labels
look wrong on inspection, to revisit later. Mirrors manual_labeler.py's
crash-safe/resumable pattern.

Usage:
    dataExtraction/facial_keypoints_extraction-master/.venv/bin/python3 \
        modelTraining/review_auto_labels.py --dataset 03_26

Keys:
    n / space = next segment
    b         = back (previous segment)
    f         = flag this segment (predicted label looks wrong) / press
                again to unflag
    q         = quit and save
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
WINDOW = "review_auto_labels"


def _load_flags(flags_path: Path) -> set[tuple[int, int]]:
    if not flags_path.exists():
        return set()
    df = pd.read_csv(flags_path, dtype=str, keep_default_na=False)
    return {(int(r["session"]), int(r["segment_idx"])) for _, r in df.iterrows()}


def _save_flags(dataset: str, flags: set[tuple[int, int]], flags_path: Path) -> None:
    rows = [{"dataset": dataset, "session": s, "segment_idx": g} for s, g in sorted(flags)]
    pd.DataFrame(rows, columns=["dataset", "session", "segment_idx"]).to_csv(
        flags_path, index=False, quoting=csv.QUOTE_MINIMAL
    )


def _overlay(frame, row, idx, total, flagged):
    h, w = frame.shape[:2]
    labels = row["predicted_labels"].split(";") if row["predicted_labels"] else []
    lines = [
        f"[{idx + 1}/{total}] session {row['session']}  segment {row['segment_idx']}"
        + ("  *** FLAGGED ***" if flagged else ""),
        "  " + "  ".join(labels),
        "n/space=next  b=back  f=flag wrong  q=quit",
    ]
    y = h - 15 - 20 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return frame


def view_segment(clip_path: Path, row, idx, total, flagged: bool) -> str:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  could not open {clip_path}, skipping")
        return "skip"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        _overlay(frame, row, idx, total, flagged)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("n"), ord(" ")):
            cap.release()
            return "next"
        if key == ord("b"):
            cap.release()
            return "back"
        if key == ord("f"):
            cap.release()
            return "flag"
        if key == ord("q"):
            cap.release()
            return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    dataset = args.dataset

    manifest_path = PROJECT_ROOT / "outputs" / dataset / "auto_review_manifest.csv"
    flags_path = PROJECT_ROOT / "outputs" / dataset / "auto_label_flags.csv"

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    flags = _load_flags(flags_path)

    print(f"{len(manifest)} segments to review. {len(flags)} already flagged from a prior session.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    pos = 0
    while 0 <= pos < len(manifest):
        row = manifest.iloc[pos]
        session, segment_idx = int(row["session"]), int(row["segment_idx"])
        clip_path = PROJECT_ROOT / row["clip_path"]
        flagged = (session, segment_idx) in flags

        action = view_segment(clip_path, row, pos, len(manifest), flagged)

        if action == "quit":
            break
        elif action == "flag":
            if flagged:
                flags.discard((session, segment_idx))
            else:
                flags.add((session, segment_idx))
            _save_flags(dataset, flags, flags_path)
            continue  # stay on this segment so the FLAGGED overlay is visible
        elif action == "back":
            pos = max(0, pos - 1)
        elif action in ("next", "skip"):
            pos += 1

    cv2.destroyAllWindows()
    print(f"Saved. {len(flags)} segment(s) flagged -> {flags_path}")


if __name__ == "__main__":
    main()
