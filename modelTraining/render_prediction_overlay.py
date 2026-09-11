"""Renders a full-session debug video showing the model's per-person
engagement PREDICTIONS frame-by-frame, next to the true (human-labeled)
continuous score -- so predictions can be watched against the actual
footage, not just read off aggregate metrics.

Loads a checkpoint saved by train_and_save_checkpoint.py (a model trained
with the target session held out, i.e. genuine out-of-sample predictions),
runs inference on that session's frame-level graphs, and draws each
detected person's box colored by PREDICTED engagement level with both the
predicted and true score as on-screen text.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/render_prediction_overlay.py \
        --checkpoint modelTraining/checkpoint_03_20_s1_ordinal_full.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasetPrep"))
from render_review_clips import _find_cam2_raw  # noqa: E402

from model import CAMEOModel  # noqa: E402
from normalize import Normalizer  # noqa: E402
from train_continuous import ANCHORS, LEVELS, bucket_to_class, corn_probas_from_logits, corn_label_from_logits, embed_dataset_arg  # noqa: E402

# BGR, red->orange->yellow->green matching this project's established
# LEVEL_COLORS convention (render_auto_review_clips.py) so "what does this
# color mean" stays consistent across every debug tool in the project.
LEVEL_COLORS = {
    "disengaged": (0, 0, 220),
    "low": (0, 128, 255),
    "medium": (0, 220, 220),
    "high": (0, 200, 0),
}

# How long each robot-action caption stays on screen. Long enough to read at
# playback speed, short enough that consecutive actions do not stack.
BANNER_HOLD_SEC = 4.0

# Face regions are grown by this factor before blurring. The stored boxes are
# tight crops used for feature extraction; a tight blur leaves hairline, jaw and
# ear visible at the edges, which is not anonymisation.
BLUR_DILATE = 1.35


def load_face_boxes(dataset: str, session: int) -> dict[int, list[tuple[int, int, int, int]]]:
    """Every face region to blur, per frame: the three participants and the
    confederate in the robot role.

    Read from the gaze_target frame_features table rather than from the graph
    dataset, because the graph carries boxes only on labelled frames (roughly
    every sixth) while this table has a row for every frame of the session.
    Blurring from the graph would leave faces visible in the gaps between
    labels, including dropouts of several seconds."""
    import pandas as pd  # noqa: PLC0415 -- only needed when blurring

    csv = (PROJECT_ROOT / "outputs" / dataset / "gaze_target" / f"session_{session}"
           / "frame_features.csv")
    if not csv.exists():
        raise SystemExit(f"--blur-faces needs {csv}, which was not found")
    df = pd.read_csv(csv)

    # the confederate box is missing on a few hundred frames; carry the nearest
    # known one across those rather than leaving the face uncovered
    for col in ("robot_x0", "robot_y0", "robot_x1", "robot_y1"):
        df[col] = df[col].ffill().bfill()

    boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for row in df.itertuples(index=False):
        entry = boxes.setdefault(int(row.frame_index), [])
        entry.append((row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1))
        entry.append((row.robot_x0, row.robot_y0, row.robot_x1, row.robot_y1))
    return boxes


def blur_faces(frame, boxes, dilate: float = BLUR_DILATE) -> None:
    """Blur in place. Pixelate first, then smooth, so the result cannot be
    sharpened back the way a single Gaussian pass sometimes can."""
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in boxes:
        if any(v != v for v in (x0, y0, x1, y1)):  # NaN
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bw, bh = (x1 - x0) * dilate, (y1 - y0) * dilate
        a, b = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        c, d = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
        if c - a < 4 or d - b < 4:
            continue
        roi = frame[b:d, a:c]
        small = cv2.resize(roi, (max(2, (c - a) // 14), max(2, (d - b) // 14)),
                           interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)
        k = max(9, (min(c - a, d - b) // 4) * 2 + 1)
        frame[b:d, a:c] = cv2.GaussianBlur(pix, (k, k), 0)


def predict_graph(model, normalizer, item, head_type: str, device: str, anchors, embed_eligible: set[str]):
    x = normalizer.transform(item).to(device)
    edges = item["edge_features"].to(device)
    with torch.no_grad():
        preds, _, _ = model(x, edges, item["source"], dataset=embed_dataset_arg(item["dataset"], embed_eligible))
        if head_type == "classification":
            probs = F.softmax(preds, dim=-1)
            hard = preds.argmax(-1)
        elif head_type == "ordinal":
            probs = corn_probas_from_logits(preds)
            hard = corn_label_from_logits(preds)
        else:
            raise ValueError(f"prediction overlay not supported for head_type={head_type!r} (regression has no discrete level)")
        expected_score = (probs * anchors).sum(-1)
    return hard.cpu().tolist(), expected_score.cpu().tolist()


def render(checkpoint_path: Path, out_path: Path | None = None, **video_kwargs) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    dataset, session = ckpt["held_dataset"], ckpt["held_session"]
    head_type = ckpt["model_kwargs"]["head_type"]
    print(f"loaded checkpoint: {dataset} session {session}, head_type={head_type}, variant={ckpt['variant']}, "
          f"trained on {ckpt['trained_on']}, {ckpt['epochs']} epochs")

    model = CAMEOModel(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = Normalizer()
    normalizer.stats = ckpt["normalizer_stats"]
    anchors = ANCHORS.to(device)
    # Reproduce the fold's own embed-eligibility exactly (older checkpoints
    # predate this key and were trained without any dataset embedding).
    embed_eligible = set(ckpt.get("embed_eligible", []))

    graphs = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{dataset}_continuous.pt", weights_only=False)
    session_graphs = [g for g in graphs if g["session"] == session]
    print(f"{len(session_graphs)} frame-level graphs for this session")

    by_frame: dict[int, list[tuple]] = {}
    for g in session_graphs:
        hard, expected_score = predict_graph(model, normalizer, g, head_type, device, anchors, embed_eligible)
        true_scores = g["labels"].tolist()
        for i in range(len(hard)):
            level = LEVELS[hard[i]]
            by_frame.setdefault(g["frame_index"], []).append(
                (g["bboxes"][i], g["participants"][i], level, expected_score[i], true_scores[i])
            )

    out_path = out_path or (PROJECT_ROOT / "outputs" / dataset / "prediction_debug" /
                             f"session_{session}_{head_type}_predictions.mp4")
    return write_overlay_video(dataset, session, by_frame, out_path, **video_kwargs)


def render_from_results(dataset: str, session: int, results_path: Path, variant_key: str,
                         out_path: Path | None = None, **video_kwargs) -> Path:
    """Same overlay video as render(), but sourced from predictions ALREADY
    saved by a full leave_one_session_out sweep (train_continuous.py) rather
    than a freshly retrained checkpoint. leave_one_session_out never persists
    model weights, but it does concatenate every fold's raw per-node
    predictions (all_preds/all_hard_preds/all_labels) in a deterministic
    order: sorted fold_keys, and each fold's test_items filtered from the
    same underlying graph list in the same order every time. That means the
    exact slice belonging to one (dataset, session) fold can be recovered
    from the saved fold_results' n_test counts, without re-running training
    -- avoiding both the wait and the small risk of a fresh retrain (dropout/
    cuDNN nondeterminism) drifting from the predictions that were actually
    diagnosed."""
    results = torch.load(results_path, weights_only=False)[variant_key]
    fold_results = results["fold_results"]
    offset = 0
    n_test = None
    for fr in fold_results:
        if fr["dataset"] == dataset and fr["session"] == session:
            n_test = fr["n_test"]
            break
        offset += fr["n_test"]
    if n_test is None:
        raise SystemExit(f"no fold for {dataset} session {session} in {results_path} [{variant_key}]")

    preds = results["all_preds"][offset:offset + n_test]
    hard_preds = results["all_hard_preds"][offset:offset + n_test]
    print(f"sliced {n_test} predictions for {dataset} session {session} from {results_path} "
          f"(fold offset {offset})")

    graphs = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{dataset}_continuous.pt", weights_only=False)
    session_graphs = [g for g in graphs if g["session"] == session]
    print(f"{len(session_graphs)} frame-level graphs for this session")

    by_frame: dict[int, list[tuple]] = {}
    cursor = 0
    for g in session_graphs:
        n = len(g["labels"])
        true_scores = g["labels"].tolist()
        for i in range(n):
            level = LEVELS[hard_preds[cursor + i]]
            by_frame.setdefault(g["frame_index"], []).append(
                (g["bboxes"][i], g["participants"][i], level, preds[cursor + i], true_scores[i])
            )
        cursor += n
    assert cursor == n_test, f"node count mismatch: graphs contributed {cursor}, results had {n_test}"

    out_path = out_path or (PROJECT_ROOT / "outputs" / dataset / "prediction_debug" /
                             f"session_{session}_{variant_key}_predictions.mp4")
    return write_overlay_video(dataset, session, by_frame, out_path, **video_kwargs)


def write_overlay_video(dataset: str, session: int, by_frame: dict[int, list[tuple]], out_path: Path,
                        hold_frames: int = 0, start_sec: float | None = None,
                        end_sec: float | None = None, scale: float = 1.0,
                        demo_labels: bool = False, actions_path: Path | None = None,
                        threshold: float = 40.0, blur: bool = False) -> Path:
    """hold_frames > 0 carries the last drawn prediction forward onto the
    unlabelled frames between updates. Labels exist only every ~6th frame
    (5 Hz against 30 fps footage), so the default hold_frames=0 draws boxes
    on ~16% of frames -- fine for stepping through a debug render, unwatchable
    at speed. The hold is CAPPED rather than unbounded because the same
    session also contains real detection dropouts of up to 168 frames, and a
    box frozen over five seconds of missing detection would misrepresent the
    pipeline as tracking someone it had actually lost."""
    video_path = _find_cam2_raw(dataset, session)
    if video_path is None:
        raise SystemExit(f"no Cam_2 footage found for {dataset} session {session}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(start_sec * fps))) if start_sec is not None else 0
    end_frame = min(total_frames, int(round(end_sec * fps))) if end_sec is not None else total_frames
    out_size = (int(round(w * scale)), int(round(h * scale)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # The robot's actions are only persuasive if the viewer can see WHY they
    # fired. Burning each trigger into the same frame as the engagement value
    # that caused it is what turns "a robot head moving" into evidence that the
    # behaviour is driven by the model's output.
    banners: list[tuple[float, str]] = []
    if actions_path is not None:
        for a in json.loads(Path(actions_path).read_text()):
            if a["kind"] == "reengage":
                text = (f"{a['target']} at {a['score']:.0f}/100 - below {threshold:.0f} for "
                        f"{a['dwell_sec']}s  ->  ROBOT re-engages {a['target']}")
            elif a["kind"] == "maintain":
                text = "all engaged  ->  ROBOT holds, acknowledges group"
            else:
                text = "ROBOT returns to watching the group"
            banners.append((a["t"], text))

    face_boxes = load_face_boxes(dataset, session) if blur else {}
    blurred_on = 0

    last_dets: list[tuple] = []
    last_seen: int | None = None
    drawn_on = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # blur before any overlay is drawn, so boxes and captions stay crisp
        if blur:
            fb = face_boxes.get(frame_idx)
            if fb:
                blur_faces(frame, fb)
                blurred_on += 1

        dets = by_frame.get(frame_idx)
        if dets is not None:
            last_dets, last_seen = dets, frame_idx
        elif hold_frames and last_seen is not None and frame_idx - last_seen <= hold_frames:
            dets = last_dets
        else:
            dets = []
        if dets:
            drawn_on += 1
        # Participants stand close together and their labels collide when drawn
        # at a uniform height; stagger by horizontal order so each stays readable.
        ordered = sorted(dets, key=lambda d: d[0][0]) if demo_labels else dets
        for rank, (bbox, letter, level, pred_score, true_score) in enumerate(ordered):
            x0, y0, x1, y1 = (int(round(v)) for v in bbox)
            color = LEVEL_COLORS[level]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            if demo_labels:
                label = f"{letter}  {pred_score:.0f}   human {true_score:.0f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty = max(th + 8, y0 - 12 - rank * (th + 14))
                cv2.rectangle(frame, (x0 - 4, ty - th - 6), (x0 + tw + 8, ty + 6), color, -1)
                cv2.putText(frame, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                label = f"{letter} pred:{level} ({pred_score:.0f}) true:({true_score:.0f})"
                cv2.putText(frame, label, (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}/{total_frames} ({frame_idx / fps:.1f}s)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}/{total_frames} ({frame_idx / fps:.1f}s)", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        if actions_path is not None:
            now = frame_idx / fps
            rule = f"RULE: engagement below {threshold:.0f} for 3s  ->  robot re-engages that person"
            (rw, rh), _ = cv2.getTextSize(rule, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (0, 0), (min(w, rw + 30), rh + 22), (0, 0, 0), -1)
            cv2.putText(frame, rule, (15, rh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            active = [(t, txt) for t, txt in banners if 0 <= now - t <= BANNER_HOLD_SEC]
            if active:
                _, text = active[-1]
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                y = h - 60
                cv2.rectangle(frame, (0, y - th - 14), (min(w, tw + 30), y + 14), (0, 0, 0), -1)
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 220, 255), 2, cv2.LINE_AA)

        if scale != 1.0:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    total_out = frame_idx - start_frame
    print(f"wrote {out_path} ({total_out} frames from {start_frame}, "
          f"boxes on {drawn_on} of them, hold_frames={hold_frames}, scale={scale})")
    if blur:
        # a frame without a blur source is a frame with visible faces, so this
        # is a check to read rather than a statistic to skim past
        gap = total_out - blurred_on
        print(f"  blurred {blurred_on}/{total_out} frames" +
              (f"  WARNING: {gap} frames had no face boxes and are UNBLURRED" if gap else "  (all frames covered)"))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="checkpoint saved by train_and_save_checkpoint.py")
    parser.add_argument("--results", help="loso_results_*.pt saved by train_continuous.py's main sweep, "
                                           "used with --variant-key/--dataset/--session instead of --checkpoint")
    parser.add_argument("--variant-key", help="e.g. full_classification, the key inside --results")
    parser.add_argument("--dataset")
    parser.add_argument("--session", type=int)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--hold-frames", type=int, default=0,
                        help="carry each prediction forward this many frames (labels are only every ~6th "
                             "frame; 12 gives a watchable video, 0 keeps the frame-exact debug behaviour)")
    parser.add_argument("--start-sec", type=float, default=None, help="render only from this time")
    parser.add_argument("--end-sec", type=float, default=None, help="render only up to this time")
    parser.add_argument("--scale", type=float, default=1.0, help="output scale factor, e.g. 0.5")
    parser.add_argument("--demo-labels", action="store_true",
                        help="compact, staggered, filled-background labels for presentation use")
    parser.add_argument("--actions", default=None,
                        help="action JSON from furhat_demo_driver.py --save-actions; burns the "
                             "triggering rule and each robot action into the frame")
    parser.add_argument("--threshold", type=float, default=40.0,
                        help="must match DISENGAGED_BELOW in furhat_demo_driver.py")
    parser.add_argument("--blur-faces", action="store_true",
                        help="blur every participant and the confederate, on every frame")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    video_kwargs = dict(hold_frames=args.hold_frames, start_sec=args.start_sec,
                        end_sec=args.end_sec, scale=args.scale, demo_labels=args.demo_labels,
                        actions_path=Path(args.actions) if args.actions else None,
                        threshold=args.threshold, blur=args.blur_faces)
    if args.checkpoint:
        render(Path(args.checkpoint), out, **video_kwargs)
    elif args.results:
        if not (args.variant_key and args.dataset and args.session is not None):
            raise SystemExit("--results requires --variant-key, --dataset, and --session")
        render_from_results(args.dataset, args.session, Path(args.results), args.variant_key, out, **video_kwargs)
    else:
        raise SystemExit("pass either --checkpoint or --results")
