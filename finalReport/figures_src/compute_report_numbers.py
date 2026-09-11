"""Recompute every quantitative claim the report makes, in one place.

Writes a human-readable dump (report_numbers.md) plus a cache of derived
arrays (report_cache.npz) that the figure scripts read, so figures and prose
are generated from the same computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameo_stats import ROOT, MT, load, summarise, per_fold_table, ccc, granularity_metrics, to_class

OUT = Path(__file__).resolve().parent
DATES = ["03_20", "03_26", "05_15", "05_14"]
lines: list[str] = []


def w(s: str = ""):
    lines.append(s)
    print(s)


def fold_means(res):
    rows = per_fold_table(res)
    keys = ["acc4", "f1_4", "acc2", "f1_2", "ccc", "spearman", "mae", "bias"]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    m["sd_f1_4"] = float(np.std([r["f1_4"] for r in rows], ddof=1))
    # 3-level needs recomputing from the arrays
    hard = np.array(res["all_hard_preds"]); true_cls = np.array(res["all_true_classes"])
    acc3, f13 = [], []
    start = 0
    for f in res["fold_results"]:
        sl = slice(start, start + f["n_test"]); start += f["n_test"]
        g = granularity_metrics(true_cls[sl], hard[sl])
        acc3.append(g["3-level"][0]); f13.append(g["3-level"][1])
    m["acc3"] = float(np.mean(acc3)); m["f1_3"] = float(np.mean(f13))
    return m, rows


# ---------------------------------------------------------------- headline
w("# Report numbers (auto-generated)\n")
best = load("loso_results_3sess_noembed.pt")
m, rows = fold_means(best)
pooled = summarise(best)
w("## 1. Standing-best run: 14-fold LOSO, 3 dates, full model\n")
w("| metric | mean over folds | pooled over all person-frames |")
w("|---|---|---|")
w(f"| 4-level accuracy | {m['acc4']:.3f} | {pooled['acc4']:.3f} |")
w(f"| 4-level macro-F1 | {m['f1_4']:.3f} (sd {m['sd_f1_4']:.3f}) | {pooled['f1_4']:.3f} |")
w(f"| 3-level accuracy | {m['acc3']:.3f} | {pooled['acc3']:.3f} |")
w(f"| 3-level macro-F1 | {m['f1_3']:.3f} | {pooled['f1_3']:.3f} |")
w(f"| binary accuracy | {m['acc2']:.3f} | {pooled['acc2']:.3f} |")
w(f"| binary macro-F1 | {m['f1_2']:.3f} | {pooled['f1_2']:.3f} |")
w(f"| CCC | {m['ccc']:.3f} | {pooled['ccc']:.3f} |")
w(f"| Spearman | {m['spearman']:.3f} | {pooled['spearman']:.3f} |")
w(f"| MAE | {m['mae']:.2f} | {pooled['mae']:.2f} |")
w(f"| n person-frames | {pooled['n']} | |")
w(f"\nbinary acc range over folds: {min(r['acc2'] for r in rows):.3f} - {max(r['acc2'] for r in rows):.3f}")
w(f"total fold training time: {sum(r['elapsed_sec'] for r in rows)/3600:.1f} h\n")

w("## 2. Per-fold detail (14 folds)\n")
w("| fold | n | acc4 | F1_4 | acc2 | CCC | rho | MAE | bias | true mean |")
w("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    w(f"| {r['dataset']} s{r['session']} | {r['n']} | {r['acc4']:.3f} | {r['f1_4']:.3f} | "
      f"{r['acc2']:.3f} | {r['ccc']:.3f} | {r['spearman']:.3f} | {r['mae']:.2f} | {r['bias']:+.1f} | {r['true_mean']:.1f} |")
w("")

# ---------------------------------------------------------------- ablation
w("## 3. Ablation (13-fold pooled run, loso_results_03_20_03_26_05_15_continuous.pt)\n")
abl = torch.load(MT / "loso_results_03_20_03_26_05_15_continuous.pt", weights_only=False)
w("| variant | acc4 | F1_4 | acc2 | CCC | rho | MAE |")
w("|---|---|---|---|---|---|---|")
abl_rows = {}
for key in abl:
    mm, rr = fold_means(abl[key])
    abl_rows[key] = (mm, rr)
    w(f"| {key} | {mm['acc4']:.3f} | {mm['f1_4']:.3f} | {mm['acc2']:.3f} | {mm['ccc']:.3f} | {mm['spearman']:.3f} | {mm['mae']:.2f} |")
w("")

# ------------------------------------------------------------ dataset embed
w("## 4. Per-date embedding (14-fold, same protocol as 1)\n")
emb = load("loso_results_3sess_dsembed.pt", key="full_classification_dsembed")
me, _ = fold_means(emb)
w("| metric | no embedding | with embedding |")
w("|---|---|---|")
for k, lab in [("acc4", "4-level acc"), ("f1_4", "4-level macro-F1"), ("acc2", "binary acc"),
               ("ccc", "CCC"), ("spearman", "Spearman"), ("mae", "MAE")]:
    w(f"| {lab} | {m[k]:.3f} | {me[k]:.3f} |")
w("")

np.savez(OUT / "report_cache.npz",
         best_preds=np.array(best["all_preds"]), best_labels=np.array(best["all_labels"]),
         best_hard=np.array(best["all_hard_preds"]), best_true=np.array(best["all_true_classes"]),
         best_n=np.array([f["n_test"] for f in best["fold_results"]]))
json.dump({"headline_fold_means": m, "headline_pooled": pooled, "per_fold": rows,
           "ablation": {k: v[0] for k, v in abl_rows.items()}, "dsembed": me},
          open(OUT / "report_numbers.json", "w"), indent=1)
Path(OUT / "report_numbers.md").write_text("\n".join(lines))
print("\nwrote", OUT / "report_numbers.md")
