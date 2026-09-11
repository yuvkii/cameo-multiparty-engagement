"""Quick decisive test (2026-07-30): does per-dataset-balanced loss
weighting (train_continuous.py's new dataset_weights, see its docstring)
fix 03_20 session 4's stuck-predicting-one-class problem, at the SAME 120
epochs as the original pooled run -- isolating the balancing fix from the
separate epoch-count question tested in diagnose_epoch_budget.py.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/diagnose_balanced_weighting.py
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
from train_continuous import bucket_to_class, class_weights, dataset_weights, ANCHORS, LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
EPOCHS = 120
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
    ds_weights = dataset_weights(train_items)
    print(f"dataset weights: {ds_weights}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for item in train_items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            preds, _, _ = model(x, edges, item["source"])
            targets = bucket_to_class(labels)
            loss = ce_loss(preds, targets) * ds_weights[item["dataset"]]
            loss.backward()
            total_loss += loss.item()
        opt.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            elapsed = time.time() - t0
            metrics = evaluate(model, normalizer, test_items, device, anchors)
            print(f"epoch {epoch + 1}/{EPOCHS}: loss={total_loss:.4f} elapsed={elapsed:.0f}s  "
                  f"mae={metrics['mae']:.2f} tol10={metrics['tol10']:.3f} spearman={metrics['spearman']:.3f} "
                  f"acc={metrics['acc']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
