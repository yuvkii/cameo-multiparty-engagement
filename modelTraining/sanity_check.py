"""Overfit a tiny subset to confirm the forward/backward pass is wired
correctly before real training -- per the plan's verification step.

Trains as classification (cross-entropy) -- see model.py/train.py for why
plain regression was tried and rejected. Checks both hard accuracy -> ~1.0
and the continuous expected-value score's MAE -> ~0, since both are used
downstream (train.py reports accuracy/F1 alongside MAE/tolerance-band)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CAMEOModel
from normalize import Normalizer
from train import ANCHORS

PROJECT_ROOT = Path(__file__).parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.load(PROJECT_ROOT / "modelTraining" / "graph_dataset_03_20.pt", weights_only=False)
# mix of rgb + cam2 graphs, deliberately, to sanity-check both branches at once
subset = [d for d in data if d["source"] == "rgb"][:10] + [d for d in data if d["source"] == "cam2"][:10]

normalizer = Normalizer().fit(subset)

model = CAMEOModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
loss_fn = nn.CrossEntropyLoss()
anchors = ANCHORS.to(device)


def evaluate():
    model.eval()
    correct, total_abs_err, total_n = 0, 0.0, 0
    with torch.no_grad():
        for item in subset:
            x = normalizer.transform(item).to(device)
            edge_features = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            logits, _, _ = model(x, edge_features, item["source"])
            probs = F.softmax(logits, dim=-1)
            expected_score = (probs * anchors).sum(-1)
            correct += (logits.argmax(-1) == labels).sum().item()
            total_abs_err += (expected_score - anchors[labels]).abs().sum().item()
            total_n += labels.shape[0]
    model.train()
    return correct / total_n, total_abs_err / total_n


for epoch in range(800):
    total_loss = 0.0
    opt.zero_grad()
    for item in subset:
        x = normalizer.transform(item).to(device)
        edge_features = item["edge_features"].to(device)
        labels = item["labels"].to(device)
        engagement_logits, _, _ = model(x, edge_features, item["source"])
        loss = loss_fn(engagement_logits, labels) / len(subset)
        loss.backward()
        total_loss += loss.item()
    opt.step()
    if epoch % 100 == 0 or epoch == 799:
        acc, mae = evaluate()
        print(f"epoch {epoch}: loss={total_loss:.4f}, train_acc={acc:.3f}, train_mae={mae:.3f}")

print("\nIf accuracy -> ~1.0 and MAE -> ~0, forward/backward pass is wired correctly.")
