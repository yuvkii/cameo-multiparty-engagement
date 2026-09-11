"""Shared metric/loading helpers for the final-report figures and tables.

Every number quoted in the report is recomputed here from the saved LOSO
prediction arrays, rather than copied from a training log -- so a figure and
the sentence next to it can never disagree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
MT = ROOT / "modelTraining"
ANCHORS = np.array([0.0, 100.0 / 3, 200.0 / 3, 100.0])
LEVELS = ["disengaged", "low", "medium", "high"]


def load(name: str, key: str = "full_classification") -> dict:
    return torch.load(MT / name, weights_only=False)[key]


def to_class(x: np.ndarray) -> np.ndarray:
    """Nearest-anchor bucketing, identical to train_continuous.bucket_to_class."""
    return np.abs(x[:, None] - ANCHORS[None, :]).argmin(1)


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient on the 0-1 scale."""
    t, p = y_true / 100.0, y_pred / 100.0
    vt, vp = t.var(), p.var()
    cov = ((t - t.mean()) * (p - p.mean())).mean()
    return float(2 * cov / (vt + vp + (t.mean() - p.mean()) ** 2))


def granularity_metrics(true_cls: np.ndarray, hard_cls: np.ndarray) -> dict:
    """Accuracy / macro-F1 at 4-level, 3-level (low+medium merged) and binary."""
    out = {}
    out["4-level"] = (accuracy_score(true_cls, hard_cls), f1_score(true_cls, hard_cls, average="macro"))
    m3 = {0: 0, 1: 1, 2: 1, 3: 2}
    t3 = np.vectorize(m3.get)(true_cls)
    p3 = np.vectorize(m3.get)(hard_cls)
    out["3-level"] = (accuracy_score(t3, p3), f1_score(t3, p3, average="macro"))
    t2, p2 = (true_cls >= 2).astype(int), (hard_cls >= 2).astype(int)
    out["binary"] = (accuracy_score(t2, p2), f1_score(t2, p2, average="macro"))
    return out


def summarise(res: dict) -> dict:
    preds = np.array(res["all_preds"])
    labels = np.array(res["all_labels"])
    hard = np.array(res["all_hard_preds"])
    true_cls = np.array(res["all_true_classes"])
    g = granularity_metrics(true_cls, hard)
    return {
        "n": len(preds),
        "ccc": ccc(labels, preds),
        "spearman": float(spearmanr(labels, preds).statistic),
        "mae": float(np.abs(labels - preds).mean()),
        "bias": float((preds - labels).mean()),
        "acc4": g["4-level"][0], "f1_4": g["4-level"][1],
        "acc3": g["3-level"][0], "f1_3": g["3-level"][1],
        "acc2": g["binary"][0], "f1_2": g["binary"][1],
        "mean_fold_acc": res["mean_accuracy"], "mean_fold_f1": res["mean_macro_f1"],
        "mean_fold_mae": res["mean_mae"], "mean_fold_spearman": res["mean_spearman"],
    }


def fold_slices(res: dict):
    """Yield (fold_dict, slice) -- prediction arrays are concatenated in fold order."""
    start = 0
    for f in res["fold_results"]:
        n = f["n_test"]
        yield f, slice(start, start + n)
        start += n
    assert start == len(res["all_preds"]), (start, len(res["all_preds"]))


def per_fold_table(res: dict) -> list[dict]:
    preds = np.array(res["all_preds"]); labels = np.array(res["all_labels"])
    hard = np.array(res["all_hard_preds"]); true_cls = np.array(res["all_true_classes"])
    rows = []
    for f, sl in fold_slices(res):
        g = granularity_metrics(true_cls[sl], hard[sl])
        rows.append({
            "dataset": f["dataset"], "session": f["session"], "n": f["n_test"],
            "mae": f["mae"], "spearman": f["spearman"],
            "acc4": g["4-level"][0], "f1_4": g["4-level"][1],
            "acc2": g["binary"][0], "f1_2": g["binary"][1],
            "ccc": ccc(labels[sl], preds[sl]),
            "bias": float((preds[sl] - labels[sl]).mean()),
            "true_mean": float(labels[sl].mean()), "pred_mean": float(preds[sl].mean()),
            "elapsed_sec": f.get("elapsed_sec", float("nan")),
        })
    return rows
