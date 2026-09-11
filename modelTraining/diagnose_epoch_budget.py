"""Quick diagnostic (2026-07-30): does the pooled 03_20+03_26+05_15 run
underperform because 120 full-batch gradient steps (this codebase does ONE
opt.step() per epoch, accumulated over the ENTIRE train set -- confirmed by
reading train_one_fold directly) just isn't enough steps for a pool that's
2-2.5x larger and more heterogeneous than any single-date run this project
has done before? Total gradient updates = epoch count, completely
independent of how much data is in the pool -- more data means each step's
average gradient is diluted across more/more-varied examples, with no extra
steps to compensate, unless epochs are scaled up too.

Retrains ONE fold (03_20 session 4 -- the worst performer in the 120-epoch
pooled run, mae=50.32/43.28 full/no_graph) up to a much higher epoch count,
reporting eval metrics at several checkpoints along the way so the trend is
visible in one run rather than needing several separate full trainings.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/diagnose_epoch_budget.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from model import CAMEOModel
from normalize import Normalizer
from build_continuous_dataset import NODE_DIM, EDGE_DIM
from train_continuous import bucket_to_class, class_weights, ANCHORS, LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS = [120, 240, 360, 480, 600]
HELD_DATASET, HELD_SESSION = "03_20", 4


def evaluate(model, normalizer, items, device, anchors):
    model.eval()
    all_preds, all_labels, all_hard, all_true = [], [], [], []
    with torch.no_grad():
        for item in items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            preds, _, _ = model(x, edges, item["source"])
            probs = F.softmax(preds, dim=-1)
            expected_score = (probs * anchors).sum(-1)
            all_preds.extend(expected_score.cpu().tolist())
            all_hard.extend(preds.argmax(-1).cpu().tolist())
            all_true.extend(bucket_to_class(labels).cpu().tolist())
            all_labels.extend(item["labels"].tolist())
    preds_arr, labels_arr = np.array(all_preds), np.array(all_labels)
    mae = float(np.abs(preds_arr - labels_arr).mean())
    tol10 = float((np.abs(preds_arr - labels_arr) <= 10).mean())
    rank_corr = float(spearmanr(preds_arr, labels_arr).correlation)
    acc = accuracy_score(all_true, all_hard)
    macro_f1 = f1_score(all_true, all_hard, labels=list(range(len(LEVELS))), average="macro", zero_division=0)
    return {"mae": mae, "tol10": tol10, "spearman": rank_corr, "acc": acc, "macro_f1": macro_f1}


def main():
    data = []
    for ds in ["03_20", "03_26", "05_15"]:
        data.extend(torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{ds}_continuous.pt", weights_only=False))
    print(f"pool: {len(data)} graphs total")

    train_items = [d for d in data if (d["dataset"], d["session"]) != (HELD_DATASET, HELD_SESSION)]
    test_items = [d for d in data if (d["dataset"], d["session"]) == (HELD_DATASET, HELD_SESSION)]
    print(f"held out {HELD_DATASET} session {HELD_SESSION}: train n={len(train_items)}, test n={len(test_items)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)
    normalizer = Normalizer().fit(train_items)
    model = CAMEOModel(dropout=0.4, head_type="classification", cam2_in_dim=NODE_DIM, edge_dim=EDGE_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-3)
    weights = class_weights(train_items, device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    anchors = ANCHORS.to(device)

    t0 = time.time()
    max_epochs = max(CHECKPOINTS)
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for item in train_items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            preds, _, _ = model(x, edges, item["source"])
            targets = bucket_to_class(labels)
            loss = ce_loss(preds, targets) / len(train_items)
            loss.backward()
            total_loss += loss.item()
        opt.step()

        if (epoch + 1) in CHECKPOINTS:
            elapsed = time.time() - t0
            metrics = evaluate(model, normalizer, test_items, device, anchors)
            print(f"epoch {epoch + 1}: loss={total_loss:.4f} elapsed={elapsed:.0f}s  "
                  f"mae={metrics['mae']:.2f} tol10={metrics['tol10']:.3f} spearman={metrics['spearman']:.3f} "
                  f"acc={metrics['acc']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
