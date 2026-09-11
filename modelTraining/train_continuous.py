"""Leave-one-session-out training + evaluation for the continuous-scale
(frame-level, not 3s-segment) 05_15 experiment.

Three head types (--head-type), sharing the same frame-level dataset/edges/
LOSO harness:

- "regression" (default): single continuous 0-100 output per node, trained
  with a tolerance-band ("epsilon-insensitive") loss instead of plain MSE --
  zero loss/gradient once a prediction is already within epsilon points of
  the true value, quadratic penalty beyond that. Measured (2026-07-25) to
  still shrink predictions toward the middle of the scale almost as much as
  plain MSE did on the segment-level pipeline -- diagnosed as the tolerance
  band rarely actually being reached during training when MAE is this large,
  so the loss behaves like squared error in practice for most of training.

- "classification": bucket the same continuous labels to the nearest of 4
  anchors (0/33.3/66.7/100) and train ordinary cross-entropy, same as
  train.py's segment-level pipeline -- doesn't have the regression-to-mean
  problem, since cross-entropy is rewarded for confident correct predictions,
  not for hedging. A continuous 0-100 score is derived at EVAL time as the
  expected value under the model's own softmax (score = sum_k P(k)*anchor_k),
  giving back a continuous-looking score while keeping cross-entropy's
  confident-extreme behavior. Reports both hard accuracy/macro-F1 and the
  same MAE/tolerance/Spearman metrics as the regression head, for direct
  comparison.

- "ordinal" (CORN, added 2026-07-30): K-1 rank-consistent binary threshold
  classifiers (corn_loss/corn_probas_from_logits/corn_label_from_logits)
  instead of plain unordered softmax. Targets a confirmed failure mode of
  the classification head: pooling multiple recording dates made its
  predictions collapse toward the two EXTREME classes (disengaged/high),
  abandoning "low"/"medium" -- plain softmax has no notion these 4 classes
  are ORDERED, so nothing penalizes confusing "medium" with "disengaged"
  any more than with "low", and as the pool gets more heterogeneous the
  least-ambiguous (extreme) classes win. CORN's conditional-threshold
  structure directly encodes the ordering. Reports the same hard accuracy/
  macro-F1 plus MAE/tolerance/Spearman as classification, via the same
  anchor-weighted-expectation scoring convention (just fed CORN's own
  per-class probabilities instead of softmax's).

Trained on GPU per explicit user instruction (protecting their CPU from
sustained high load), even though CPU was measured faster for the
segment-level dataset's much-smaller per-fold item count -- this dataset
is far larger (frame-level, not per-3s-segment), so the tradeoff may
differ anyway; not re-litigated here, GPU is used regardless per instruction.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from model import CAMEOModel
from normalize import Normalizer
from build_continuous_dataset import NODE_DIM, EDGE_DIM

PROJECT_ROOT = Path(__file__).parent.parent
TOLERANCES = (5.0, 10.0)
LEVELS = ["disengaged", "low", "medium", "high"]
ANCHORS = torch.tensor([0.0, 100.0 / 3, 200.0 / 3, 100.0])
# Minimum number of a dataset's OWN sibling sessions that must remain in a
# given fold's training set before that fold gets a real per-dataset
# embedding for it (see model.py's dataset=None backoff comment). Computed
# per-fold, not from a fixed dataset list -- verified empirically 2026-08-01
# that a real embedding for 03_20 (2 sessions total at the time) made both
# its LOSO folds worse, because holding out one session left only ONE
# sibling for the embedding to fit, so it just learned that one session's
# own average rather than a genuine per-recording-date signal. 03_20 s2
# (3 sessions total, added same day) leaves 2 siblings per fold -- still
# thin, but this threshold decides case-by-case per fold rather than
# hardcoding which whole datasets qualify, so it adapts automatically as
# more sessions get added to any date in the future.
MIN_SIBLING_SESSIONS_FOR_EMBED = 2


def eligible_embed_datasets(train_items) -> set[str]:
    """Which datasets have enough of their OWN sessions left in THIS fold's
    training set to support a real per-date embedding (see
    MIN_SIBLING_SESSIONS_FOR_EMBED). Computed from train_items, so a fold
    that holds out one of a thin dataset's few sessions automatically backs
    that dataset off to no-embedding for that fold only."""
    sessions_by_ds: dict[str, set] = {}
    for item in train_items:
        sessions_by_ds.setdefault(item["dataset"], set()).add(item["session"])
    return {ds for ds, sessions in sessions_by_ds.items()
            if len(sessions) >= MIN_SIBLING_SESSIONS_FOR_EMBED}


def embed_dataset_arg(dataset: str, eligible: set[str]) -> str | None:
    """None means "skip the embedding for this item" -- see model.py's
    dataset=None backoff comment. Not an error/missing value."""
    return dataset if dataset in eligible else None


def tolerance_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    diff = (pred - target).abs()
    excess = torch.clamp(diff - epsilon, min=0.0)
    return (excess ** 2).mean()


def bucket_to_class(labels: torch.Tensor) -> torch.Tensor:
    """Nearest-anchor snap, same convention as assemble_dataset.py's
    level_to_idx for the segment-level pipeline -- keeps the two pipelines'
    class semantics identical even though this one starts from a raw
    continuous percentage rather than a human-picked category name."""
    dists = (labels.unsqueeze(-1) - ANCHORS.to(labels.device)).abs()
    return dists.argmin(dim=-1)


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """CORN loss (Shi, Cao & Raschka 2021) -- added 2026-07-30 to target a
    confirmed adjacent-class-confusion collapse: pooling multiple recording
    dates made the plain classification head's predictions collapse toward
    the two EXTREME classes (disengaged/high), abandoning "low"/"medium"
    (measured: low class share 6.5%->3.7%->1.4%, medium 16.6%->12.6%->8.5%,
    monotonically as more dates got pooled in). Plain softmax has no notion
    that these 4 classes are ORDERED, so nothing in the loss penalizes
    confusing "medium" with "disengaged" any more than with "low" -- as the
    pool gets more heterogeneous, the two least-ambiguous (extreme) classes
    win the softmax competition and mass drains from the middle.

    CORN reframes K-way classification as K-1 conditional binary tasks: does
    y clear threshold k, GIVEN it already cleared threshold k-1 (task 0's
    condition is vacuous -- everyone is eligible). Each example only
    contributes to the tasks its true class is eligible for. This
    conditioning (not an architectural constraint like CORAL's weight-
    sharing) is what makes the thresholds rank-consistent by construction
    at inference (see corn_label_from_logits/corn_probas_from_logits) --
    logits: (n, num_classes-1) raw per-threshold logits from the model
    (unbounded, no sigmoid applied yet -- that happens here and in eval).
    targets: (n,) integer class indices, same convention as bucket_to_class.
    """
    total_loss, total_examples = 0.0, 0
    for k in range(num_classes - 1):
        mask = targets > (k - 1)
        if mask.sum() == 0:
            continue
        task_targets = (targets[mask] > k).float()
        task_logits = logits[mask, k]
        total_loss = total_loss + F.binary_cross_entropy_with_logits(task_logits, task_targets, reduction="sum")
        total_examples += int(mask.sum().item())
    return total_loss / max(total_examples, 1)


def corn_probas_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """(n, K-1) conditional threshold logits -> (n, K) proper per-class
    probabilities (rows sum to 1), so the SAME anchor-weighted-expectation
    scoring the classification head already uses (score = sum_k P(k)*anchor_k)
    works unchanged for ordinal too -- no new scoring convention to
    separately validate. sigmoid(logits) gives P(y>k | y>k-1); cumprod gives
    the (rank-consistent, non-increasing) P(y>k); per-class probs are the
    adjacent differences of that cumulative sequence."""
    cum_gt = torch.cumprod(torch.sigmoid(logits), dim=-1)  # (n, K-1): P(y>0), P(y>1), P(y>2)
    p0 = 1.0 - cum_gt[:, :1]
    p_mid = cum_gt[:, :-1] - cum_gt[:, 1:]
    p_last = cum_gt[:, -1:]
    return torch.cat([p0, p_mid, p_last], dim=-1)


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Hard predicted class = count of thresholds k where P(y>k) > 0.5 --
    standard CORN decision rule, rank-consistent by construction (can never
    predict "cleared threshold 2 but not threshold 1")."""
    cum_gt = torch.cumprod(torch.sigmoid(logits), dim=-1)
    return (cum_gt > 0.5).sum(dim=-1)


def class_weights(train_items: list[dict], device: str) -> torch.Tensor:
    counts = np.zeros(len(LEVELS))
    for item in train_items:
        for c in bucket_to_class(item["labels"]).tolist():
            counts[c] += 1
    weights = counts.sum() / (len(LEVELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Lin et al. 2017 focal loss. class_weights() alone (static inverse-
    frequency reweighting) was already active in the 60-epoch pooled run and
    the medium/high confusion persisted regardless -- that rules out frequency
    imbalance as the sole cause. Focal loss adds a SECOND, different signal:
    (1-p_t)^gamma down-weights examples the model already gets right
    (confident, easy) regardless of class, concentrating gradient on
    whatever's currently being misclassified -- e.g. medium/high frames
    specifically, if that's where the confusion actually lives, independent
    of how common those classes are."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    alpha_t = alpha[targets]
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()


def dataset_weights(train_items: list[dict], scheme: str = "equal") -> dict[str, float]:
    """Per-item loss weight so each DATASET contributes equally to the total
    gradient regardless of how many items it has in this fold's training
    set -- fixes a real bug found 2026-07-30 pooling 03_20+03_26+05_15: this
    codebase trains with full-batch gradient descent (one opt.step() per
    epoch, accumulated over every train item via `loss / len(train_items)`),
    so a plain per-item average lets whichever dataset has the most items
    dominate the gradient. Diagnosed on 03_20 session 4 (its only same-date
    training signal is ~1400 items vs ~12500 from 03_26+05_15 combined):
    hard predictions (acc/macro_f1) were BIT-FOR-BIT IDENTICAL across 120,
    240, and 360 epochs -- the model was stuck always predicting one class
    for this fold, not slowly converging, consistent with the minority
    dataset having too little effective gradient weight to shift the
    decision boundary at all, not merely training too briefly.

    weight = 1 / (n_datasets * items_in_that_dataset), so summed over all
    items this always totals exactly 1 (same overall loss scale as before),
    and every dataset's items sum to 1/n_datasets regardless of its size.
    Reduces to the exact previous behavior (uniform 1/len(train_items)) when
    all training items come from a single dataset -- single-dataset runs
    are unaffected by this change.

    `scheme` (added 2026-08-03) controls how much total gradient each dataset
    gets. "equal" is the original behaviour above. It was tuned when the pool
    was three roughly comparable dates; adding 05_14 -- on its own ~49% of all
    items -- means equal weighting now throttles the largest date to 25% while
    inflating 03_20 (8.7% of items) to the same 25%. "proportional" gives each
    dataset weight in line with its actual size (i.e. plain per-item
    averaging, reintroducing the domination problem equal weighting fixed).
    "sqrt" is the middle ground: shares proportional to sqrt(n_items), so a
    much larger date carries more weight without small dates collapsing to
    irrelevance. All three keep the total loss scale at exactly 1.0."""
    counts: dict[str, int] = {}
    for item in train_items:
        counts[item["dataset"]] = counts.get(item["dataset"], 0) + 1
    if scheme == "equal":
        n_datasets = len(counts)
        return {ds: 1.0 / (n_datasets * n) for ds, n in counts.items()}
    if scheme == "proportional":
        total = sum(counts.values())
        return {ds: 1.0 / total for ds in counts}
    if scheme == "sqrt":
        shares = {ds: n ** 0.5 for ds, n in counts.items()}
        denom = sum(shares.values())
        return {ds: (shares[ds] / denom) / n for ds, n in counts.items()}
    raise ValueError(f"unknown dataset weighting scheme {scheme!r}")


def train_one_fold(train_items, test_items, device, variant, epochs, lr, weight_decay, dropout, epsilon, head_type,
                    batch_size=16, focal_gamma=2.0, return_model=False, dataset_embed=False,
                    weighting="equal"):
    normalizer = Normalizer().fit(train_items)
    embed_eligible = eligible_embed_datasets(train_items) if dataset_embed else set()
    if dataset_embed:
        backed_off = {item["dataset"] for item in train_items} - embed_eligible
        print(f"    dataset-embed eligible: {sorted(embed_eligible)}"
              + (f"  backed off (too few sibling sessions): {sorted(backed_off)}" if backed_off else ""), flush=True)

    model_kwargs = {"dropout": dropout, "head_type": head_type, "cam2_in_dim": NODE_DIM, "edge_dim": EDGE_DIM,
                     "use_dataset_embed": dataset_embed}
    if variant == "mean_pool":
        model_kwargs["pooling"] = "mean"
    if variant == "no_graph":
        model_kwargs["use_graph"] = False
    if variant == "simple_graph":
        model_kwargs["graph_type"] = "simple"

    model = CAMEOModel(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if head_type == "classification":
        # Plain class_weights()-only CrossEntropyLoss was already tried (the
        # 60-epoch pooled run) and the medium/high confusion persisted, so
        # frequency reweighting alone isn't the fix -- focal_loss adds the
        # (1-p_t)^gamma hard-example term on top of the same alpha weights.
        weights = class_weights(train_items, device)
        anchors = ANCHORS.to(device)
    elif head_type == "ordinal":
        # Deliberately NOT reusing class_weights here for phase 1 -- starting
        # unweighted so the confusion-matrix verification isolates whether
        # CORN's structural fix alone reverses the collapse, before adding a
        # second variable. Revisit with a per-task pos_weight in
        # F.binary_cross_entropy_with_logits if verification still shows
        # minority-class neglect.
        anchors = ANCHORS.to(device)

    ds_weights = dataset_weights(train_items, scheme=weighting)

    # Real training-regime bug found 2026-07-30 via literature comparison
    # (MultiMediate/DA-Mamba and small-batch-training research): this loop
    # used to do ONE opt.step() per epoch, accumulated over the ENTIRE
    # training set -- total gradient updates = epoch count, completely
    # independent of dataset size. "Revisiting Small Batch Training for Deep
    # Neural Networks" (well-cited ML methodology paper) found batches of
    # 2-32 consistently generalize best, both from more frequent updates and
    # from the regularizing noise small batches inject. Fixed: real
    # mini-batch steps (BATCH_SIZE items), reshuffled every epoch (the
    # stochasticity is part of why this helps, not just the batch size).
    #
    # ds_weights sums to 1 across the WHOLE training set (each dataset's
    # items summing to 1/n_datasets, see dataset_weights' docstring) -- using
    # those same per-item weights unchanged within a ~16-item mini-batch
    # would under-scale the gradient by orders of magnitude (they're
    # calibrated to average out over thousands of items, not one batch).
    # Renormalizing by the batch's own weight-sum keeps both properties: a
    # properly-scaled gradient every step, AND each dataset still gets
    # equal expected representation across batches.
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_items))
        total_loss = 0.0
        for batch_start in range(0, len(order), batch_size):
            batch_idx = order[batch_start:batch_start + batch_size]
            batch = [train_items[i] for i in batch_idx]
            batch_weight_sum = sum(ds_weights[item["dataset"]] for item in batch)
            opt.zero_grad()
            batch_loss = 0.0
            for item in batch:
                x = normalizer.transform(item).to(device)
                edges = item["edge_features"].to(device)
                labels = item["labels"].to(device)
                preds, _, _ = model(x, edges, item["source"], dataset=embed_dataset_arg(item["dataset"], embed_eligible))
                item_weight = ds_weights[item["dataset"]] / batch_weight_sum
                if head_type == "classification":
                    targets = bucket_to_class(labels)
                    loss = focal_loss(preds, targets, weights, gamma=focal_gamma) * item_weight
                elif head_type == "ordinal":
                    targets = bucket_to_class(labels)
                    loss = corn_loss(preds, targets, len(LEVELS)) * item_weight
                else:
                    loss = tolerance_loss(preds, labels, epsilon) * item_weight
                loss.backward()
                batch_loss += loss.item()
            opt.step()
            total_loss += batch_loss
        if epoch == 0 or (epoch + 1) % 20 == 0:
            n_batches = (len(order) + batch_size - 1) // batch_size
            print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / n_batches:.4f}", flush=True)

    model.eval()
    all_preds, all_labels, all_hard_preds, all_true_classes = [], [], [], []
    with torch.no_grad():
        for item in test_items:
            x = normalizer.transform(item).to(device)
            edges = item["edge_features"].to(device)
            labels = item["labels"].to(device)
            preds, _, _ = model(x, edges, item["source"], dataset=embed_dataset_arg(item["dataset"], embed_eligible))
            if head_type == "classification":
                probs = F.softmax(preds, dim=-1)
                expected_score = (probs * anchors).sum(-1)
                all_preds.extend(expected_score.cpu().tolist())
                all_hard_preds.extend(preds.argmax(-1).cpu().tolist())
                all_true_classes.extend(bucket_to_class(labels).cpu().tolist())
            elif head_type == "ordinal":
                probs = corn_probas_from_logits(preds)
                expected_score = (probs * anchors).sum(-1)
                all_preds.extend(expected_score.cpu().tolist())
                all_hard_preds.extend(corn_label_from_logits(preds).cpu().tolist())
                all_true_classes.extend(bucket_to_class(labels).cpu().tolist())
            else:
                all_preds.extend(preds.cpu().tolist())
            all_labels.extend(item["labels"].tolist())

    if return_model:
        return all_preds, all_labels, all_hard_preds, all_true_classes, model, normalizer
    return all_preds, all_labels, all_hard_preds, all_true_classes


def leave_one_session_out(data, variant, device, epochs, lr, weight_decay, dropout, epsilon, head_type,
                           batch_size=16, focal_gamma=2.0, max_folds=None, dataset_embed=False):
    fold_keys = sorted({(d["dataset"], d["session"]) for d in data})
    if max_folds is not None:
        # Debug-only knob for cheap head-type comparisons (e.g. ordinal vs
        # classification on the SAME fold subset) -- not part of the CORN
        # feature itself, just fast iteration. Picking every Nth fold
        # (rather than the first max_folds) spreads the subset across
        # whatever datasets are in the pool instead of always landing on
        # one date's sessions only.
        step = max(1, len(fold_keys) // max_folds)
        fold_keys = fold_keys[::step][:max_folds]
    fold_results = []
    all_preds, all_labels, all_hard_preds, all_true_classes = [], [], [], []

    for held_dataset, held_session in fold_keys:
        train_items = [d for d in data if (d["dataset"], d["session"]) != (held_dataset, held_session)]
        test_items = [d for d in data if (d["dataset"], d["session"]) == (held_dataset, held_session)]
        if not test_items or not train_items:
            continue

        t0 = time.time()
        print(f"  fold: held-out {held_dataset} session {held_session} "
              f"(train n={len(train_items)}, test n={len(test_items)})", flush=True)
        preds, labels, hard_preds, true_classes = train_one_fold(
            train_items, test_items, device, variant, epochs, lr, weight_decay, dropout, epsilon, head_type,
            batch_size=batch_size, focal_gamma=focal_gamma, dataset_embed=dataset_embed)
        elapsed = time.time() - t0

        preds_arr, labels_arr = np.array(preds), np.array(labels)
        mae = float(np.abs(preds_arr - labels_arr).mean())
        tol_accs = {t: float((np.abs(preds_arr - labels_arr) <= t).mean()) for t in TOLERANCES}
        rank_corr = float(spearmanr(preds_arr, labels_arr).correlation) if len(set(labels)) > 1 else float("nan")

        fold_result = {
            "dataset": held_dataset, "session": held_session, "n_test": len(labels),
            "mae": mae, "tol_accuracy": tol_accs, "spearman": rank_corr, "elapsed_sec": elapsed,
        }
        log_line = f"    -> mae={mae:.2f} tol5={tol_accs[5.0]:.3f} tol10={tol_accs[10.0]:.3f} spearman={rank_corr:.3f}"
        if head_type in ("classification", "ordinal"):
            acc = accuracy_score(true_classes, hard_preds)
            macro_f1 = f1_score(true_classes, hard_preds, labels=list(range(len(LEVELS))), average="macro", zero_division=0)
            fold_result["accuracy"] = acc
            fold_result["macro_f1"] = macro_f1
            log_line += f" acc={acc:.3f} macro_f1={macro_f1:.3f}"
        print(log_line + f" ({elapsed:.0f}s)", flush=True)
        fold_results.append(fold_result)
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_hard_preds.extend(hard_preds)
        all_true_classes.extend(true_classes)

    result = {
        "variant": variant,
        "fold_results": fold_results,
        "mean_mae": np.mean([f["mae"] for f in fold_results]),
        "mean_tol_accuracy": {t: np.mean([f["tol_accuracy"][t] for f in fold_results]) for t in TOLERANCES},
        "mean_spearman": np.nanmean([f["spearman"] for f in fold_results]),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }
    if head_type in ("classification", "ordinal"):
        result["mean_accuracy"] = np.mean([f["accuracy"] for f in fold_results])
        result["mean_macro_f1"] = np.mean([f["macro_f1"] for f in fold_results])
        result["all_hard_preds"] = all_hard_preds
        result["all_true_classes"] = all_true_classes
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["05_15"],
                         help="one or more dataset names -- graphs from all of them are pooled into one "
                              "LOSO run, so folds span every (dataset, session) combination across all of them")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=["full", "no_graph"])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=10.0, help="tolerance band half-width in percentage points (regression head only)")
    parser.add_argument("--head-type", choices=["regression", "classification", "ordinal"], default="regression")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                         help="focal loss focusing parameter (classification head only); 0 reduces to plain class-weighted CE")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="mini-batch size for gradient steps (real batching, replacing full-batch GD)")
    parser.add_argument("--max-folds", type=int, default=None,
                         help="debug-only: cap LOSO to this many folds (evenly spread across the pool) for fast head-type comparisons")
    parser.add_argument("--dataset-embed", action="store_true",
                         help="add a learned per-recording-date embedding to each node's representation "
                              "(model.py's DATASET_VOCAB) -- see model.py's 2026-08-01 comment for the "
                              "diagnosis motivating this")
    parser.add_argument("-o", "--output", default=None,
                         help="results .pt path (default loso_results_<datasets>_continuous.pt); set this when "
                              "running two configs concurrently so they don't race on the merge-and-save")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: CUDA not available, falling back to CPU despite --device request")

    dataset_tag = "_".join(args.dataset)
    data = []
    for ds in args.dataset:
        ds_data = torch.load(PROJECT_ROOT / "modelTraining" / f"graph_dataset_{ds}_continuous.pt", weights_only=False)
        print(f"  loaded {ds}: {len(ds_data)} frame-level graphs")
        data.extend(ds_data)
    print(f"dataset(s): {args.dataset} -> {len(data)} frame-level graphs total")
    print(f"head_type={args.head_type} epsilon={args.epsilon} lr={args.lr} weight_decay={args.weight_decay} "
          f"dropout={args.dropout} epochs={args.epochs}")

    results = {}
    for variant in args.variants:
        print(f"\n=== variant: {variant} ===")
        torch.manual_seed(0)
        np.random.seed(0)
        result = leave_one_session_out(data, variant, device, args.epochs, args.lr, args.weight_decay, args.dropout,
                                        args.epsilon, args.head_type, batch_size=args.batch_size,
                                        focal_gamma=args.focal_gamma, max_folds=args.max_folds,
                                        dataset_embed=args.dataset_embed)
        # Keep "full"/"no_graph"/etc. as the key for regression (backward
        # compatible with the results file already committed); suffix for
        # classification/ordinal so all three head types coexist in the same
        # file rather than one overwriting another.
        suffix = {"regression": "", "classification": "_classification", "ordinal": "_ordinal"}[args.head_type]
        # Distinct key when the per-date embedding is on, so an embed run and
        # a no-embed run of the same variant/head can coexist for comparison
        # instead of one silently overwriting the other (they are different
        # models, not a re-run of the same one).
        key = f"{variant}{suffix}{'_dsembed' if args.dataset_embed else ''}"
        results[key] = result
        tol_str = ", ".join(f"tol{int(t)}={result['mean_tol_accuracy'][t]:.3f}" for t in TOLERANCES)
        line = f"  MEAN: mae={result['mean_mae']:.2f}  {tol_str}  spearman={result['mean_spearman']:.3f}"
        if args.head_type in ("classification", "ordinal"):
            line += f"  acc={result['mean_accuracy']:.3f}  macro_f1={result['mean_macro_f1']:.3f}"
        print(line)

    print("\n=== summary ===")
    for key, r in results.items():
        tol_str = ", ".join(f"tol{int(t)}={r['mean_tol_accuracy'][t]:.3f}" for t in TOLERANCES)
        line = f"{key:25s} mae={r['mean_mae']:.2f}  {tol_str}  spearman={r['mean_spearman']:.3f}"
        if args.head_type in ("classification", "ordinal"):
            line += f"  acc={r['mean_accuracy']:.3f}  macro_f1={r['mean_macro_f1']:.3f}"
        print(line)

    # -o exists so two configs can run CONCURRENTLY without racing on the
    # merge below (both would read-modify-write the same file, and whichever
    # saved last would drop the other's results).
    results_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "modelTraining" / f"loso_results_{dataset_tag}_continuous.pt")
    if results_path.exists():
        # Merge into whatever was already computed, same reasoning as
        # train.py's save logic -- a run may only cover a subset of
        # --variants (e.g. re-running just "full" after an edge-feature
        # change, since "no_graph" is mathematically unaffected by edges
        # and not worth re-running). A plain overwrite would silently
        # discard previously-computed variants' results.
        existing = torch.load(results_path, weights_only=False)
        existing.update(results)
        results = existing
    torch.save(results, results_path)
    # Not relative_to(PROJECT_ROOT) -- an -o path outside/relative to the
    # project raises ValueError there, which would crash AFTER the save and
    # look like a failed run.
    try:
        shown = results_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = results_path
    print(f"\nSaved -> {shown} ({sorted(results.keys())})")
