"""Fully-manual, position-based engagement labeler.

Plays one raw clip per segment (everyone in frame together, looped) and
lets the reviewer score each visible person directly, identified by their
left-to-right on-screen position at labeling time -- not by track_id,
which has been shown to drift mid-segment and can't be trusted to name a
consistent person. Reconciling a labeled position back to a track_id/bbox
is a separate, later, code-driven step.

Usage:
    dataExtraction/facial_keypoints_extraction-master/.venv/bin/python3 \
        datasetPrep/manual_labeler.py --dataset 03_20

Keys (while a clip plays on loop):
    slider = drag to the engagement percentage (0-100) that feels right
        for the next person, left to right -- no need to overthink a
        bucket, just land on what feels right
    enter = confirm the slider value for the next person, advances to
        the next on-screen position within this same segment
    0 = unclear / not rating this segment (nobody visible, recording
        interrupted, or otherwise not worth scoring -- finalizes with
        zero people)
    u = undo the last person scored in this segment
    space = done scoring this segment, move to the next one
    b = back (revisit the previous segment; clears its saved scores so
        you can redo it)
    q = quit and save
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

WINDOW = "manual_labeler"
SLIDER = "engagement %"
LABELS_COLUMNS = ["dataset", "session", "segment_idx", "position", "human_engagement_level"]


def _load_labels(labels_path: Path) -> pd.DataFrame:
    if labels_path.exists():
        return pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=LABELS_COLUMNS)


def _save_labels(labels_df: pd.DataFrame, labels_path: Path) -> None:
    labels_df.to_csv(labels_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _remove_segment_labels(labels_df: pd.DataFrame, dataset: str, session: int, segment_idx: int) -> pd.DataFrame:
    mask = ~(
        (labels_df["dataset"] == dataset)
        & (labels_df["session"] == str(session))
        & (labels_df["segment_idx"] == str(segment_idx))
    )
    return labels_df[mask].reset_index(drop=True)


def _segment_labels(labels_df: pd.DataFrame, dataset: str, session: int, segment_idx: int) -> list[str]:
    sub = labels_df[
        (labels_df["dataset"] == dataset)
        & (labels_df["session"] == str(session))
        & (labels_df["segment_idx"] == str(segment_idx))
    ].sort_values("position")
    return sub["human_engagement_level"].tolist()


def _overlay(frame, row, idx, total, current_scores, pct):
    h, w = frame.shape[:2]
    scored = ", ".join(f"P{i + 1}={lvl}%" for i, lvl in enumerate(current_scores)) or "none yet"
    lines = [
        f"[{idx + 1}/{total}] session {row['session']}  segment {row['segment_idx']}",
        f"scored so far: {scored}",
        f"next person (P{len(current_scores) + 1}): {pct}%  -- drag slider, enter=confirm",
        "0=unclear/skip  u=undo  space=done  b=back  q=quit",
    ]
    y = h - 15 - 20 * (len(lines) - 1)
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
    return frame


def label_segment(clip_path: Path, row, idx, total, labels_df, dataset: str) -> tuple[str, pd.DataFrame]:
    """Interactively score a single segment. Returns (action, updated labels_df)."""
    session = int(row["session"])
    segment_idx = int(row["segment_idx"])

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  could not open {clip_path}, skipping")
        return "skip", labels_df
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))
    cv2.setTrackbarPos(SLIDER, WINDOW, 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        current_scores = _segment_labels(labels_df, dataset, session, segment_idx)
        pct = cv2.getTrackbarPos(SLIDER, WINDOW)
        _overlay(frame, row, idx, total, current_scores, pct)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (13, 10):  # enter
            position = len(current_scores) + 1
            new_row = pd.DataFrame([{
                "dataset": dataset, "session": str(session), "segment_idx": str(segment_idx),
                "position": str(position), "human_engagement_level": str(pct),
            }])
            labels_df = pd.concat([labels_df, new_row], ignore_index=True)
            cv2.setTrackbarPos(SLIDER, WINDOW, 50)
            continue
        if key == ord("0"):
            if not current_scores:
                new_row = pd.DataFrame([{
                    "dataset": dataset, "session": str(session), "segment_idx": str(segment_idx),
                    "position": "0", "human_engagement_level": "unclear",
                }])
                labels_df = pd.concat([labels_df, new_row], ignore_index=True)
            cap.release()
            return "done", labels_df
        if key == ord("u"):
            labels_df = _remove_segment_labels(labels_df, dataset, session, segment_idx)
            continue
        if key == ord(" "):
            cap.release()
            return "done" if current_scores else "skip", labels_df
        if key == ord("b"):
            cap.release()
            return "back", labels_df
        if key == ord("q"):
            cap.release()
            return "quit", labels_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. 03_20")
    args = parser.parse_args()
    dataset = args.dataset

    queue_path = PROJECT_ROOT / "outputs" / dataset / "manual_queue.csv"
    labels_path = PROJECT_ROOT / "outputs" / dataset / "manual_engagement_labels.csv"

    queue_df = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
    labels_df = _load_labels(labels_path)

    candidates = [i for i in range(len(queue_df)) if queue_df.at[i, "clip_path"] != ""]
    unlabeled_idx = [i for i in candidates if queue_df.at[i, "labeled"] != "1"]
    if not unlabeled_idx:
        print("Nothing left to label — every segment already marked done.")
        return

    print(f"{len(unlabeled_idx)} segments left to label out of {len(candidates)} with usable footage.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(SLIDER, WINDOW, 50, 100, lambda _pos: None)

    pos = 0
    while pos < len(unlabeled_idx):
        i = unlabeled_idx[pos]
        row = queue_df.loc[i]
        clip_path = PROJECT_ROOT / row["clip_path"]

        action, labels_df = label_segment(clip_path, row, pos, len(unlabeled_idx), labels_df, dataset)
        _save_labels(labels_df, labels_path)

        if action == "quit":
            break
        elif action == "back":
            if pos > 0:
                pos -= 1
                prev_i = unlabeled_idx[pos]
                queue_df.at[prev_i, "labeled"] = "0"
                labels_df = _remove_segment_labels(
                    labels_df, dataset, int(queue_df.at[prev_i, "session"]), int(queue_df.at[prev_i, "segment_idx"])
                )
                _save_labels(labels_df, labels_path)
                queue_df.to_csv(queue_path, index=False, quoting=csv.QUOTE_MINIMAL)
            continue
        elif action == "skip":
            pos += 1
            continue
        else:  # done
            queue_df.at[i, "labeled"] = "1"
            queue_df.to_csv(queue_path, index=False, quoting=csv.QUOTE_MINIMAL)
            pos += 1

    cv2.destroyAllWindows()
    remaining = sum(
        1 for i in range(len(queue_df)) if queue_df.at[i, "clip_path"] != "" and queue_df.at[i, "labeled"] != "1"
    )
    print(f"Saved. {remaining} segment(s) still unlabeled — re-run the same command to resume.")


if __name__ == "__main__":
    main()
