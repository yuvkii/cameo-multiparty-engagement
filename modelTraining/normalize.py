"""Per-(source, dataset) feature standardization.

Raw feature scales vary wildly (e.g. fk_audio_spectral_centroid_hz in the
thousands vs blendshapes in 0-1) -- confirmed this was the actual cause of
a model that looked structurally fine (correct gradients, correct einsum
shapes) but couldn't overfit even 5 tiny examples: unnormalized inputs
saturated the first layer and the optimizer converged to predicting the
majority class. Standardizing fixed it immediately (loss -> 0 on the same
diagnostic).

Stats are fit per LOSO fold from the TRAIN split only, never including the
held-out session -- fitting on the full dataset would leak held-out
session statistics into training.

**Keyed by (source, dataset), not source alone (fixed 2026-07-28).** Found a
real, well-evidenced bug pooling `03_20`+`05_15` for the continuous
frame-level pipeline: 03_20's raw OpenFace AU values run ~2-3x lower than
05_15's across every session checked (e.g. mean au~0.06-0.08 vs ~0.14-0.21)
-- plausibly a lighting/camera-setup difference between recording dates, not
a labeling or extraction bug. With one shared per-source mean/std dominated
by whichever dataset has more items in the training pool (05_15, ~7000 vs
03_20's ~2600), 03_20's features get normalized against a scale that isn't
theirs, landing as systematically low z-scores that read as "low facial
activation" regardless of the session's actual engagement level. This bit
hardest on 03_20 session 3 specifically (true label mean=67.6, high) because
a model biased to associate "low AU z-score" with low engagement
systematically underpredicted it (pred mean~35 vs true 67.6, correlation
still ~0.5-0.56 -- the direction/ranking was right, the calibration wasn't).
Session 1 (true mean=32.6, already low) coincidentally looked less broken
since the bias happened to point the right way there. Fitting stats
separately per dataset removes this cross-date scale confound regardless of
which dataset dominates the pool -- each session gets normalized against
its own recording's typical range, using only OTHER sessions from the same
dataset (never leaking the held-out session's own stats).
"""
from __future__ import annotations

import torch


class Normalizer:
    def __init__(self):
        self.stats: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, items: list[dict]) -> "Normalizer":
        keys = {(item["source"], item["dataset"]) for item in items}
        for key in keys:
            source, dataset = key
            feats = [item["node_features"] for item in items
                     if item["source"] == source and item["dataset"] == dataset]
            all_feats = torch.cat(feats, dim=0)
            mean = all_feats.mean(0)
            std = all_feats.std(0)
            std[std == 0] = 1.0
            self.stats[key] = (mean, std)
        return self

    def transform(self, item: dict) -> torch.Tensor:
        mean, std = self.stats[(item["source"], item["dataset"])]
        return (item["node_features"] - mean) / std
