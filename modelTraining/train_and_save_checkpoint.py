"""Trains ONE specific held-out-session fold and saves the trained model +
normalizer to disk, for render_prediction_overlay.py to load and visualize
frame-by-frame predictions against. train_continuous.py's main LOSO sweep
never persists model weights (only aggregate metrics), so this is a small,
separate entry point that reuses train_one_fold directly rather than
duplicating the training loop.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/train_and_save_checkpoint.py \
        --dataset 03_20 03_26 05_15 --held-dataset 03_20 --held-session 1 \
        --head-type ordinal --variant full --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_continuous import train_one_fold, eligible_embed_datasets
from build_continuous_dataset import NODE_DIM, EDGE_DIM

PROJECT_ROOT = Path(__file__).parent.parent


def build_model_kwargs(dropout: float, head_type: str, variant: str, dataset_embed: bool = False) -> dict:
    """Mirrors train_one_fold's own model_kwargs construction exactly, so a
    saved checkpoint can be reloaded into an identically-shaped model
    without needing to re-derive this from `variant` at load time."""
    kwargs = {"dropout": dropout, "head_type": head_type, "cam2_in_dim": NODE_DIM, "edge_dim": EDGE_DIM,
              "use_dataset_embed": dataset_embed}
    if variant == "mean_pool":
        kwargs["pooling"] = "mean"
    if variant == "no_graph":
        kwargs["use_graph"] = False
    if variant == "simple_graph":
        kwargs["graph_type"] = "simple"
    return kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--held-dataset", required=True)
    parser.add_argument("--held-session", type=int, required=True)
    parser.add_argument("--head-type", choices=["regression", "classification", "ordinal"], default="ordinal")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--dataset-embed", action="store_true")
    parser.add_argument("--weighting", choices=["equal", "proportional", "sqrt"], default="equal",
                         help="how much total gradient each dataset gets (see dataset_weights)")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = []
    for ds in args.dataset:
        data.extend(torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{ds}_continuous.pt", weights_only=False))
    print(f"pool: {len(data)} graphs total")

    train_items = [d for d in data if (d["dataset"], d["session"]) != (args.held_dataset, args.held_session)]
    test_items = [d for d in data if (d["dataset"], d["session"]) == (args.held_dataset, args.held_session)]
    print(f"held out {args.held_dataset} session {args.held_session}: train n={len(train_items)}, test n={len(test_items)}")
    if not test_items:
        raise SystemExit(f"no graphs found for {args.held_dataset} session {args.held_session} in this pool")

    torch.manual_seed(0)
    np.random.seed(0)
    all_preds, all_labels, all_hard_preds, all_true_classes, model, normalizer = train_one_fold(
        train_items, test_items, device, args.variant, args.epochs, args.lr, args.weight_decay, args.dropout,
        args.epsilon, args.head_type, return_model=True, dataset_embed=args.dataset_embed,
        weighting=args.weighting,
    )

    preds_arr, labels_arr = np.array(all_preds), np.array(all_labels)
    mae = float(np.abs(preds_arr - labels_arr).mean())
    bias = float((preds_arr - labels_arr).mean())
    from sklearn.metrics import accuracy_score, f1_score
    from train_continuous import LEVELS
    acc = accuracy_score(all_true_classes, all_hard_preds)
    macro_f1 = f1_score(all_true_classes, all_hard_preds, labels=list(range(len(LEVELS))),
                        average="macro", zero_division=0)
    print(f"RESULT weighting={args.weighting} {args.held_dataset} s{args.held_session}: "
          f"n={len(all_preds)} mae={mae:.2f} bias={bias:+.1f} acc={acc:.3f} macro_f1={macro_f1:.3f} "
          f"pred_range=({preds_arr.min():.0f},{preds_arr.max():.0f})")

    suffix = "_dsembed" if args.dataset_embed else ""
    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "modelTraining" / f"checkpoint_{args.held_dataset}_s{args.held_session}_{args.head_type}_{args.variant}{suffix}_{args.weighting}.pt"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kwargs": build_model_kwargs(args.dropout, args.head_type, args.variant, args.dataset_embed),
        # Which datasets actually got a real embedding in THIS fold (the rest
        # were backed off to dataset=None) -- inference must reproduce the
        # same choice or it would feed the model an embedding it never
        # trained with. See train_continuous.eligible_embed_datasets.
        "embed_eligible": sorted(eligible_embed_datasets(train_items)) if args.dataset_embed else [],
        "variant": args.variant,
        "normalizer_stats": normalizer.stats,
        "held_dataset": args.held_dataset,
        "held_session": args.held_session,
        "trained_on": args.dataset,
        "epochs": args.epochs,
    }, out_path)
    print(f"saved checkpoint -> {out_path}")
