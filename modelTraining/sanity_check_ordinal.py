"""Overfit a tiny subset to confirm the CORN ordinal head's forward/backward
pass is wired correctly -- same pattern as sanity_check.py (which checks the
classification head), adapted for train_continuous.py's continuous-label
pipeline and the new corn_loss/corn_probas_from_logits/corn_label_from_logits.

Checks both hard accuracy -> ~1.0 (via corn_label_from_logits) and the
continuous expected-value score's MAE -> ~0 (via corn_probas_from_logits +
the same anchor-weighted-expectation used elsewhere), since both are used
downstream."""
from __future__ import annotations

from pathlib import Path

import torch

from model import CAMEOModel
from normalize import Normalizer
from build_continuous_dataset import NODE_DIM, EDGE_DIM
from train_continuous import ANCHORS, LEVELS, bucket_to_class, corn_loss, corn_probas_from_logits, corn_label_from_logits

PROJECT_ROOT = Path(__file__).parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.load(PROJECT_ROOT / "modelTraining" / "graph_dataset_03_20_continuous.pt", weights_only=False)
subset = data[:20]

normalizer = Normalizer().fit(subset)

model = CAMEOModel(head_type="ordinal", cam2_in_dim=NODE_DIM, edge_dim=EDGE_DIM).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
anchors = ANCHORS.to(device)


def evaluate():
    model.eval()
    correct, total_abs_err, total_n = 0, 0.0, 0
    with torch.no_grad():
        for item in subset:
            x = normalizer.transform(item).to(device)
            edge_features = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            targets = bucket_to_class(labels)
            logits, _, _ = model(x, edge_features, item["source"])
            probs = corn_probas_from_logits(logits)
            expected_score = (probs * anchors).sum(-1)
            hard = corn_label_from_logits(logits)
            correct += (hard == targets).sum().item()
            total_abs_err += (expected_score - labels).abs().sum().item()
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
        targets = bucket_to_class(labels)
        engagement_logits, _, _ = model(x, edge_features, item["source"])
        loss = corn_loss(engagement_logits, targets, len(LEVELS)) / len(subset)
        loss.backward()
        total_loss += loss.item()
    opt.step()
    if epoch % 100 == 0 or epoch == 799:
        acc, mae = evaluate()
        print(f"epoch {epoch}: loss={total_loss:.4f}, train_acc={acc:.3f}, train_mae={mae:.3f}")

print("\nIf accuracy -> ~1.0 and MAE -> ~0, the CORN ordinal head is wired correctly.")
