"""Interactive keyboard-driven labeler for human review of review_queue.csv.

Plays each not-yet-labeled clip on loop in a window; a keypress records the
human judgment and advances to the next clip. Writes to review_queue.csv
after every label so progress survives a crash or early quit — re-running
automatically resumes at the first unlabeled row.

Usage:
    dataExtraction/facial_keypoints_extraction-master/.venv/bin/python3 \
        datasetPrep/review_labeler.py --dataset 03_20

Keys:
    1 = disengaged   2 = low   3 = medium   4 = high
    b = back (revisit previous row)   n = skip (leave blank, move on)
    q = quit and save
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

LEVEL_KEYS = {
    ord("1"): "disengaged",
    ord("2"): "low",
    ord("3"): "medium",
    ord("4"): "high",
}
WINDOW = "review_labeler"


def _overlay(frame, row, idx, total):
    h, w = frame.shape[:2]
    lines = [
        f"[{idx + 1}/{total}] session {row['session']}  seg {row['segment_idx']}"
        + (f"  track {row['track_id']}" if row["track_id"] != "" else "  (group view)"),
        f"auto: {row['auto_engagement_level']}  (confidence {float(row['confidence']):.3f})",
        "1=disengaged 2=low 3=medium 4=high | b=back n=skip q=quit",
    ]
    y = h - 15 - 20 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
    return frame


def label_clip(clip_path: Path, row, idx, total) -> str:
    """Play clip on loop until a recognized key is pressed. Returns an action string."""
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
        _overlay(frame, row, idx, total)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in LEVEL_KEYS:
            cap.release()
            return LEVEL_KEYS[key]
        if key == ord("n"):
            cap.release()
            return "skip"
        if key == ord("b"):
            cap.release()
            return "back"
        if key == ord("q"):
            cap.release()
            return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. 03_20")
    args = parser.parse_args()

    queue_path = PROJECT_ROOT / "outputs" / args.dataset / "review_queue.csv"
    df = pd.read_csv(queue_path, dtype=str, keep_default_na=False)

    unlabeled_idx = [i for i in range(len(df)) if df.at[i, "human_engagement_level"] == "" and df.at[i, "clip_path"] != ""]
    if not unlabeled_idx:
        print("Nothing left to review — every row already has a human_engagement_level.")
        return

    print(f"{len(unlabeled_idx)} clips left to review out of {len(df)} total rows.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    pos = 0
    while pos < len(unlabeled_idx):
        i = unlabeled_idx[pos]
        row = df.loc[i]
        clip_path = PROJECT_ROOT / row["clip_path"]

        action = label_clip(clip_path, row, pos, len(unlabeled_idx))

        if action == "quit":
            break
        elif action == "back":
            pos = max(0, pos - 1)
            continue
        elif action == "skip":
            pos += 1
            continue
        else:
            df.at[i, "human_engagement_level"] = action
            df.to_csv(queue_path, index=False, quoting=csv.QUOTE_MINIMAL)
            pos += 1

    cv2.destroyAllWindows()
    remaining = sum(1 for i in range(len(df)) if df.at[i, "human_engagement_level"] == "" and df.at[i, "clip_path"] != "")
    print(f"Saved. {remaining} clip(s) still unlabeled — re-run the same command to resume.")


if __name__ == "__main__":
    main()
