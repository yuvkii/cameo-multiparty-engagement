"""Leave-one-session-out training + evaluation for CAMEO's engagement head,
plus the ablation ladder that substantiates the graph module is doing real
relational work (not just "a graph was added").

Engagement scoring (2026-07-23): still trained as 4-way classification
(cross-entropy) against the human-rated categories (disengaged/low/medium/
high -- validated, inter-rater-checked, see [[manual_engagement_labeling]]/
[[cameo_model_build]] memory). A plain continuous-regression head (single
scalar output, weighted MSE against numeric anchors) was tried first and
rejected: both a regularization sweep and removing the output-bounding
sigmoid left predictions compressed toward the center of the scale --
an inherent property of squared-error loss under real predictive
uncertainty (the MSE-optimal prediction is the conditional mean), not a
fixable hyperparameter/activation issue, and it cost the classifier's real
strength at the confident extremes.

Instead, the continuous 0-100 "percentage" score this project wants is
derived at evaluation time as the EXPECTED VALUE under the classifier's own
softmax probabilities (score = sum_k P(class=k) * anchor_k) -- keeps
cross-entropy's confident-extreme behavior, while still producing a
continuous score that spreads naturally across adjacent classes exactly
when the model is torn between them (the fuzzy low/medium case). Evaluated
three ways: hard accuracy/macro-F1 (direct comparison to the pre-2026-07-23
classification numbers), MAE + Spearman correlation on the continuous
score (rank-preserving quality), and tolerance-band "accuracy" (prediction
within +-N points of the true anchor) -- reporting the tolerance band
ALONE would risk looking artificially good if the model just learned to
predict near the population mean without discriminating anything, so it's
always read alongside the other two, not instead of them.

Behavior head is present in every forward pass (see model.py) but excluded
from the loss entirely -- Option B in docs/design-notes.md, no behavior labels exist
yet. train_behavior_head=False is the explicit, visible flag for this.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model import CAMEOModel
from normalize import Normalizer

PROJECT_ROOT = Path(__file__).parent.parent
LEVELS = ["disengaged", "low", "medium", "high"]
# Numeric anchors for the 4 human-rated categories, evenly spaced 0-100 --
# no finer-grained ground truth exists to justify uneven spacing without
# re-labeling with a continuous instrument, so this is the defensible
# default, not an empirically-derived choice.
ANCHORS = torch.tensor([0.0, 100.0 / 3, 200.0 / 3, 100.0])
TOLERANCES = (5.0, 10.0)  # percentage points, both reported (not just one)
train_behavior_head = False  # Option B (docs/design-notes.md) -- architected, not trained, no behavior labels exist yet


class ProgressTracker:
    """Writes live run state to a JSON file after every epoch, for
    dashboard.html (served locally, see run_dashboard.py) to poll. Kept
    dead simple -- one file, overwritten each update, no history beyond
    a rolling per-second time estimate -- since this only needs to answer
    "how far along, how long left," not produce an audit trail."""

    def __init__(self, path: Path, variants: list[str], fold_keys: list[tuple], epochs_per_fold: int):
        self.path = path
        self.variants = variants
        self.fold_keys = fold_keys
        self.epochs_per_fold = epochs_per_fold
        self.total_steps = len(variants) * len(fold_keys) * epochs_per_fold
        self.steps_done = 0
        self.start_time = time.time()
        self.completed_folds: list[dict] = []
        self.variant_idx = 0
        self.fold_idx = 0

    def start_fold(self, variant: str, variant_idx: int, fold_key: tuple, fold_idx: int):
        self.variant_idx = variant_idx
        self.fold_idx = fold_idx
        self._current = {"variant": variant, "fold_key": fold_key}

    def on_epoch(self, epoch: int, loss: float):
        self.steps_done += 1
        elapsed = time.time() - self.start_time
        rate = self.steps_done / elapsed if elapsed > 0 else 0
        remaining_steps = self.total_steps - self.steps_done
        eta = remaining_steps / rate if rate > 0 else None
        state = {
            "status": "running",
            "variant": self._current["variant"],
            "variant_idx": self.variant_idx, "variant_total": len(self.variants),
            "fold": f"{self._current['fold_key'][0]} session {self._current['fold_key'][1]}",
            "fold_idx": self.fold_idx, "fold_total": len(self.fold_keys),
            "epoch": epoch + 1, "epoch_total": self.epochs_per_fold,
            "loss": round(loss, 4),
            "steps_done": self.steps_done, "total_steps": self.total_steps,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1) if eta is not None else None,
            "completed_folds": self.completed_folds,
            "updated_at": time.time(),
        }
        self.path.write_text(json.dumps(state))

    def finish_fold(self, variant: str, fold_key: tuple, mae: float, tol10_accuracy: float):
        self.completed_folds.append({
            "variant": variant, "dataset": fold_key[0], "session": fold_key[1],
            "mae": round(mae, 3), "tol10_accuracy": round(tol10_accuracy, 3),
        })

    def finish(self):
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        state["status"] = "done"
        state["updated_at"] = time.time()
        self.path.write_text(json.dumps(state))


def zero_relational_edge_channels(item: dict) -> dict:
    """For the proximity-only ablation: keep channel 1 (proximity), zero
    channels 0 (gaze containment / head-pose alignment) and 2 (mutual gaze)
    -- the two channels that encode actual relational/attentional structure."""
    item = dict(item)
    edges = item["edge_features"].clone()
    edges[..., 0] = 0.0
    edges[..., 2] = 0.0
    item["edge_features"] = edges
    return item


def class_weights(items: list[dict], device: str) -> torch.Tensor:
    counts = np.zeros(len(LEVELS))
    for item in items:
        for label in item["labels"].tolist():
            counts[label] += 1
    weights = counts.sum() / (len(LEVELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(train_items: list[dict], test_items: list[dict], device: str, variant: str, epochs: int = 200,
                    tracker: "ProgressTracker | None" = None, lr: float = 3e-3, weight_decay: float = 1e-4,
                    dropout: float = 0.2):
    normalizer = Normalizer().fit(train_items)

    if variant == "proximity_only":
        train_items = [zero_relational_edge_channels(it) for it in train_items]
        test_items = [zero_relational_edge_channels(it) for it in test_items]

    model_kwargs = {"dropout": dropout}
    if variant == "mean_pool":
        model_kwargs["pooling"] = "mean"
    if variant == "no_graph":
        model_kwargs["use_graph"] = False
    if variant == "simple_graph":
        model_kwargs["graph_type"] = "simple"

    model = CAMEOModel(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = class_weights(train_items, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    anchors = ANCHORS.to(device)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for item in train_items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            engagement_logits, behavior_logits, _ = model(x, edges, item["source"])
            loss = loss_fn(engagement_logits, labels) / len(train_items)
            if train_behavior_head:
                raise NotImplementedError("no behavior labels exist yet -- see docs/design-notes.md, Option B")
            loss.backward()
            total_loss += loss.item()
        opt.step()
        if tracker is not None:
            tracker.on_epoch(epoch, total_loss)

    model.eval()
    all_preds, all_hard_preds, all_labels = [], [], []
    with torch.no_grad():
        for item in test_items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            engagement_logits, _, _ = model(x, edges, item["source"])
            probs = F.softmax(engagement_logits, dim=-1)
            expected_score = (probs * anchors).sum(-1)  # continuous score for tolerance-band eval
            all_preds.extend(expected_score.cpu().tolist())
            all_hard_preds.extend(engagement_logits.argmax(-1).cpu().tolist())  # for direct accuracy/F1 comparison
            all_labels.extend(item["labels"].tolist())

    return all_preds, all_hard_preds, all_labels, model, normalizer


def leave_one_session_out(data: list[dict], variant: str, device: str, epochs: int = 200,
                           tracker: "ProgressTracker | None" = None, variant_idx: int = 0,
                           lr: float = 3e-3, weight_decay: float = 1e-4, dropout: float = 0.2) -> dict:
    # Key on (dataset, session), not session alone -- session numbers 1-6 are
    # a fixed protocol slot reused across every recording date (per docs/design-notes.md
    # dataset layout), so "session 3" in 03_20 and "session 3" in 03_26 are
    # different recordings/different people, not the same fold once datasets
    # are combined.
    #
    # A session is eligible as a held-out test fold if it has ANY manual-
    # labeled data (not "only" -- see below), since sessions can now carry a
    # MIX of manual + auto graphs from the same (dataset, session) once
    # multi-camera auxiliary views are added (2026-07-22): 03_20 sessions
    # 1-4 keep their real Cam_2 manual labels, while Cam_1/3/4/5/6's views
    # of those same sessions are auto-labeled. Test on manual data only
    # (auto-labeled predictions aren't trustworthy as ground truth, same
    # reasoning as before) -- but exclude ALL of the held-out session's data
    # (manual AND auto, every camera) from that fold's training set, not
    # just the manual portion. Otherwise another camera's auto-labeled view
    # of the literal session being tested would leak into training -- same
    # people, same moment, just a different angle, which defeats the point
    # of held-out evaluation even though the labels themselves are noisy.
    manual_keys = {(d["dataset"], d["session"]) for d in data if d.get("label_source", "manual") == "manual"}
    fold_keys = sorted(manual_keys)
    fold_results = []
    all_preds, all_hard_preds, all_labels = [], [], []

    for fold_idx, (held_dataset, held_session) in enumerate(fold_keys):
        train_items = [d for d in data if (d["dataset"], d["session"]) != (held_dataset, held_session)]
        test_items = [
            d for d in data
            if (d["dataset"], d["session"]) == (held_dataset, held_session) and d.get("label_source", "manual") == "manual"
        ]
        if not test_items or not train_items:
            continue

        # A source (rgb/cam2) with zero coverage in train can't be evaluated --
        # its branch never sees a gradient update and Normalizer has no stats
        # for it. Skip the fold rather than test an untrained branch; the
        # held-out session's own source keeps contributing to every OTHER
        # fold's training set as normal. Currently only bites 03_20 session 5
        # (the sole remaining rgb session there after 1/2 moved to cam2) --
        # general so it self-resolves once more rgb sessions exist.
        train_sources = {d["source"] for d in train_items}
        test_sources = {d["source"] for d in test_items}
        if not test_sources.issubset(train_sources):
            print(f"  skipping {held_dataset} session {held_session}: source(s) {test_sources - train_sources} unseen in training data")
            continue

        if tracker is not None:
            tracker.start_fold(variant, variant_idx, (held_dataset, held_session), fold_idx)
        preds, hard_preds, labels, _, _ = train_one_fold(train_items, test_items, device, variant, epochs=epochs,
                                                          tracker=tracker, lr=lr, weight_decay=weight_decay, dropout=dropout)
        preds_arr, targets_arr = np.array(preds), ANCHORS.numpy()[np.array(labels)]
        mae = float(np.abs(preds_arr - targets_arr).mean())
        tol_accs = {t: float((np.abs(preds_arr - targets_arr) <= t).mean()) for t in TOLERANCES}
        # Spearman, not Pearson: only the ordering (does the model rank
        # disengaged < low < medium < high correctly) is guaranteed
        # meaningful here -- the anchors' exact spacing (0/33/67/100) is an
        # assumption, not measured, so a rank correlation doesn't depend on
        # that assumption the way Pearson's linear-fit would.
        rank_corr = float(spearmanr(preds_arr, labels).correlation) if len(set(labels)) > 1 else float("nan")
        acc = accuracy_score(labels, hard_preds)
        macro_f1 = f1_score(labels, hard_preds, labels=list(range(len(LEVELS))), average="macro", zero_division=0)
        if tracker is not None:
            tracker.finish_fold(variant, (held_dataset, held_session), mae, tol_accs[10.0])
        fold_results.append({
            "dataset": held_dataset, "session": held_session, "n_test": len(labels),
            "mae": mae, "tol_accuracy": tol_accs, "spearman": rank_corr,
            "accuracy": acc, "macro_f1": macro_f1,
        })
        all_preds.extend(preds)
        all_hard_preds.extend(hard_preds)
        all_labels.extend(labels)

    return {
        "variant": variant,
        "fold_results": fold_results,
        "mean_mae": np.mean([f["mae"] for f in fold_results]),
        "mean_tol_accuracy": {t: np.mean([f["tol_accuracy"][t] for f in fold_results]) for t in TOLERANCES},
        "mean_spearman": np.nanmean([f["spearman"] for f in fold_results]),
        "mean_accuracy": np.mean([f["accuracy"] for f in fold_results]),
        "mean_macro_f1": np.mean([f["macro_f1"] for f in fold_results]),
        "all_preds": all_preds,
        "all_hard_preds": all_hard_preds,
        "all_labels": all_labels,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["03_20"], help="One or more dataset-dates to combine, e.g. --dataset 03_20 03_26")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--variants", nargs="+", default=None,
                         help="Subset of the ablation ladder to run, e.g. --variants simple_graph no_graph. Default: all 5.")
    parser.add_argument("--progress-file", default=str(PROJECT_ROOT / "modelTraining" / "training_progress.json"))
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    combined_name = "_".join(args.dataset)

    # Measured empirically (2026-07-20): CPU was ~1.8x faster than GPU here
    # (30s vs 53s per 100 epochs, one fold) -- graphs are tiny (1-3 nodes)
    # and training loops per-item rather than batched, so GPU kernel-launch
    # overhead dominates actual compute at this scale. Switched to GPU
    # anyway (2026-07-24) per explicit user instruction: sustained CPU load
    # from long training runs is a hardware-safety concern for them, which
    # outweighs the wall-clock speed difference found above.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    data = []
    for ds in args.dataset:
        data.extend(torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{ds}.pt", weights_only=False))
    print(f"datasets: {args.dataset} -> {len(data)} graphs total")

    variants = args.variants if args.variants else ["full", "proximity_only", "mean_pool", "no_graph", "simple_graph"]

    # Precompute fold_keys the same way leave_one_session_out does (a
    # session is eligible if it has ANY manual-labeled data -- sessions can
    # carry a mix of manual + auto graphs now that auxiliary-camera views
    # share a (dataset, session) with their session's real manual labels,
    # see 2026-07-22 fix), just to size the progress tracker's total step
    # count. Includes folds that'll later get skipped for source-coverage
    # reasons, so the ETA is a slight overestimate near the very end, not
    # worth the duplication to avoid.
    fold_keys = sorted({(d["dataset"], d["session"]) for d in data if d.get("label_source", "manual") == "manual"})
    tracker = ProgressTracker(Path(args.progress_file), variants, fold_keys, args.epochs)
    print(f"progress -> {args.progress_file} ({len(fold_keys)} folds x {len(variants)} variants x {args.epochs} epochs)")
    print(f"lr={args.lr} weight_decay={args.weight_decay} dropout={args.dropout}")

    results = {}
    for variant_idx, variant in enumerate(variants):
        print(f"\n=== variant: {variant} ===")
        # Reset per variant, not once globally -- otherwise each variant's weight
        # init/dropout stream depends on how many random draws the *previous*
        # variants in this invocation already consumed, so results silently depend
        # on --variants order instead of being an independent, reproducible run per
        # variant (confirmed directly: no_graph scored 0.465 run alone/first vs
        # 0.426 run 4th in a 5-variant list, same data/hyperparameters otherwise).
        torch.manual_seed(0)
        np.random.seed(0)
        result = leave_one_session_out(data, variant, device, epochs=args.epochs, tracker=tracker, variant_idx=variant_idx,
                                        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout)
        results[variant] = result
        for f in result["fold_results"]:
            tol_str = ", ".join(f"tol{int(t)}={f['tol_accuracy'][t]:.3f}" for t in TOLERANCES)
            print(f"  {f['dataset']} session {f['session']}: n={f['n_test']}, acc={f['accuracy']:.3f}, macro_f1={f['macro_f1']:.3f}, "
                  f"mae={f['mae']:.2f}, {tol_str}, spearman={f['spearman']:.3f}")
        tol_str = ", ".join(f"tol{int(t)}={result['mean_tol_accuracy'][t]:.3f}" for t in TOLERANCES)
        print(f"  MEAN: acc={result['mean_accuracy']:.3f}, macro_f1={result['mean_macro_f1']:.3f}, "
              f"mae={result['mean_mae']:.2f}, {tol_str}, spearman={result['mean_spearman']:.3f}")

    print("\n=== ablation ladder summary ===")
    for variant in variants:
        r = results[variant]
        tol_str = ", ".join(f"tol{int(t)}={r['mean_tol_accuracy'][t]:.3f}" for t in TOLERANCES)
        print(f"{variant:20s} acc={r['mean_accuracy']:.3f}  macro_f1={r['mean_macro_f1']:.3f}  "
              f"mae={r['mean_mae']:.2f}  {tol_str}  spearman={r['mean_spearman']:.3f}")

    cm_variant = "full" if "full" in results else variants[0]
    print(f"\n=== confusion matrix ({cm_variant} variant, hard argmax predictions, all folds pooled) ===")
    cm = confusion_matrix(results[cm_variant]["all_labels"], results[cm_variant]["all_hard_preds"], labels=list(range(len(LEVELS))))
    print("labels:", LEVELS)
    print(cm)

    # Residual breakdown by true category, on the continuous expected-value
    # score -- shows whether the tolerance-band framing is adding value on
    # top of the hard classification above (smaller residuals near
    # boundaries specifically) or not.
    preds_arr = np.array(results[cm_variant]["all_preds"])
    labels_arr = np.array(results[cm_variant]["all_labels"])
    targets_arr = ANCHORS.numpy()[labels_arr]
    print(f"\n=== continuous-score residuals by true category ({cm_variant} variant, all folds pooled) ===")
    for idx, name in enumerate(LEVELS):
        mask = labels_arr == idx
        if mask.sum() == 0:
            continue
        resid = preds_arr[mask] - targets_arr[mask]
        print(f"  {name:12s} (n={mask.sum():3d}): mean_pred={preds_arr[mask].mean():5.1f}  target={targets_arr[mask][0]:5.1f}  "
              f"mean_abs_resid={np.abs(resid).mean():5.2f}")

    results_path = PROJECT_ROOT / "modelTraining" / f"loso_results_{combined_name}.pt"
    if results_path.exists():
        # This run may only cover a subset of the ablation ladder (--variants) --
        # merge into whatever was already computed for this dataset combo rather
        # than clobbering previously-run variants' results.
        existing = torch.load(results_path, weights_only=False)
        existing.update(results)
        results = existing
    torch.save(results, results_path)
    print(f"\nSaved results -> {results_path.relative_to(PROJECT_ROOT)} ({sorted(results.keys())})")
    tracker.finish()
