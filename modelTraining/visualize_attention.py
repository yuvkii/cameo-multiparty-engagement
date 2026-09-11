"""Visualize the group-attention-pooling weights on example segments --
the qualitative evidence that the model is forming subgroup/attentional
structure, not just classifying each person independently. Renders the
learned per-node pooling weight next to each person's position in the raw
segment clip.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CAMEOModel
from normalize import Normalizer
from train import ANCHORS, LEVELS, class_weights

PROJECT_ROOT = Path(__file__).parent.parent


def train_full_model(data: list[dict], held_out_session: int) -> tuple[CAMEOModel, Normalizer]:
    train_items = [d for d in data if d["session"] != held_out_session]
    normalizer = Normalizer().fit(train_items)
    model = CAMEOModel()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    weights = class_weights(train_items, "cpu")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for _ in range(200):
        opt.zero_grad()
        for item in train_items:
            x = normalizer.transform(item)
            engagement_logits, _, _ = model(x, item["edge_features"], item["source"])
            loss = loss_fn(engagement_logits, item["labels"]) / len(train_items)
            loss.backward()
        opt.step()
    model.eval()
    return model, normalizer


def visualize_segment(model: CAMEOModel, normalizer: Normalizer, item: dict, dataset: str, out_path: Path):
    with torch.no_grad():
        x = normalizer.transform(item)
        engagement_logits, _, attn_weights = model(x, item["edge_features"], item["source"])
        scores = (F.softmax(engagement_logits, dim=-1) * ANCHORS).sum(-1)

    clip_path = PROJECT_ROOT / "outputs" / dataset / "manual_clips" / f"session{item['session']}_seg{item['segment_idx']}.mp4"
    cap = cv2.VideoCapture(str(clip_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"could not read clip {clip_path}")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame_rgb)
    ax.axis("off")

    n = attn_weights.shape[0]
    title_lines = [f"session {item['session']} segment {item['segment_idx']} ({item['source']})"]
    for i in range(n):
        w = attn_weights[i].item()
        true_lvl = LEVELS[item["labels"][i].item()]
        title_lines.append(f"  node {i}: group-attn weight={w:.2f}  true={true_lvl}  pred_score={scores[i].item():.1f}")
    ax.set_title("\n".join(title_lines), fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="03_20")
    parser.add_argument("--n_examples", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(0)
    data = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{args.dataset}.pt", weights_only=False)

    out_dir = PROJECT_ROOT / "modelTraining" / "attention_figures"
    out_dir.mkdir(exist_ok=True)

    # pick a few 3-node cam2 segments (richest relational structure) across different sessions
    candidates = [d for d in data if d["source"] == "cam2" and d["node_features"].shape[0] == 3]
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(candidates), size=min(args.n_examples, len(candidates)), replace=False)

    sessions_needed = {candidates[i]["session"] for i in chosen}
    models_by_holdout = {}
    for s in sessions_needed:
        print(f"training model with session {s} held out (for visualizing its own segments)...")
        models_by_holdout[s] = train_full_model(data, s)

    for i in chosen:
        item = candidates[i]
        model, normalizer = models_by_holdout[item["session"]]
        out_path = out_dir / f"attn_session{item['session']}_seg{item['segment_idx']}.png"
        visualize_segment(model, normalizer, item, args.dataset, out_path)
