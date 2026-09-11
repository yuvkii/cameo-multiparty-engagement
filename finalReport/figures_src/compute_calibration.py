"""Between-session shrinkage diagnosis and the post-hoc variance
recalibration that follows from it.

The scale factor and reference mean for a fold are estimated ONLY from the
other folds' predictions -- no held-out label is touched -- so this is a
legitimate test-time calibration, not fitting on the evaluation set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameo_stats import load, fold_slices, ccc, granularity_metrics, to_class

OUT = Path(__file__).resolve().parent
lines = []


def w(s=""):
    lines.append(s); print(s)


def metrics(t, p):
    g = granularity_metrics(to_class(t), to_class(np.clip(p, 0, 100)))
    return dict(ccc=ccc(t, p), mae=float(np.abs(t - p).mean()),
                spearman=float(spearmanr(t, p).statistic),
                acc4=g["4-level"][0], f1_4=g["4-level"][1],
                acc3=g["3-level"][0], f1_3=g["3-level"][1],
                acc2=g["binary"][0], f1_2=g["binary"][1],
                bias=float((p - t).mean()))


res = load("loso_results_3sess_noembed.pt")
preds = np.array(res["all_preds"]); labels = np.array(res["all_labels"])
hard = np.array(res["all_hard_preds"]); true_cls = np.array(res["all_true_classes"])
folds = list(fold_slices(res))

w("# Shrinkage and calibration (auto-generated)\n")
w("## 1. Variance compression\n")
sd_t = np.array([labels[sl].std() for _, sl in folds])
sd_p = np.array([preds[sl].std() for _, sl in folds])
mu_t = np.array([labels[sl].mean() for _, sl in folds])
mu_p = np.array([preds[sl].mean() for _, sl in folds])
w(f"within-session sd: true {sd_t.mean():.1f}, predicted {sd_p.mean():.1f}, "
  f"mean ratio {np.mean(sd_p/sd_t):.3f} (range {np.min(sd_p/sd_t):.2f}-{np.max(sd_p/sd_t):.2f})")
w(f"between-session sd of session means: true {mu_t.std():.1f}, predicted {mu_p.std():.1f}, "
  f"ratio {mu_p.std()/mu_t.std():.3f}")
lr = linregress(mu_t, mu_p - mu_t)
rr = pearsonr(mu_t, mu_p - mu_t)
w(f"per-fold bias vs that fold's true mean: r = {rr.statistic:.3f} (p = {rr.pvalue:.2g}), "
  f"slope {lr.slope:.3f}, R^2 {lr.rvalue**2:.3f}")
lr2 = linregress(mu_t, mu_p)
w(f"predicted mean = {lr2.slope:.3f} x true mean + {lr2.intercept:.1f}; "
  f"fixed point {lr2.intercept/(1-lr2.slope):.1f} (corpus mean {labels.mean():.1f})\n")

w("## 2. Post-hoc recalibration, leave-one-fold-out estimated\n")
rows = []
for i, (fi, sli) in enumerate(folds):
    others = [j for j in range(len(folds)) if j != i]
    scale = float(np.mean([preds[folds[j][1]].std() / labels[folds[j][1]].std() for j in others]))
    ref_t = float(np.mean([labels[folds[j][1]].mean() for j in others]))
    ref_p = float(np.mean([preds[folds[j][1]].mean() for j in others]))
    cal = np.clip(ref_t + (preds[sli] - ref_p) / scale, 0, 100)
    b = metrics(labels[sli], preds[sli])
    a = metrics(labels[sli], cal)
    # argmax classifier accuracy, the number the uncalibrated model is normally scored by
    g_arg = granularity_metrics(true_cls[sli], hard[sli])
    rows.append(dict(fold=f"{fi['dataset']} s{fi['session']}", n=fi["n_test"], scale=scale,
                     base=b, cal=a, arg_acc4=g_arg["4-level"][0], arg_f1_4=g_arg["4-level"][1],
                     arg_acc2=g_arg["binary"][0]))

w("| fold | CCC before | CCC after | MAE before | MAE after | bias before | bias after |")
w("|---|---|---|---|---|---|---|")
for r in rows:
    w(f"| {r['fold']} | {r['base']['ccc']:.3f} | {r['cal']['ccc']:.3f} | {r['base']['mae']:.2f} | "
      f"{r['cal']['mae']:.2f} | {r['base']['bias']:+.1f} | {r['cal']['bias']:+.1f} |")


def mean(key, which):
    return float(np.mean([r[which][key] for r in rows]))


w("")
w("| metric | argmax class (as trained) | expected-value score | recalibrated score |")
w("|---|---|---|---|")
w(f"| 4-level accuracy | {np.mean([r['arg_acc4'] for r in rows]):.3f} | {mean('acc4','base'):.3f} | {mean('acc4','cal'):.3f} |")
w(f"| 4-level macro-F1 | {np.mean([r['arg_f1_4'] for r in rows]):.3f} | {mean('f1_4','base'):.3f} | {mean('f1_4','cal'):.3f} |")
w(f"| binary accuracy | {np.mean([r['arg_acc2'] for r in rows]):.3f} | {mean('acc2','base'):.3f} | {mean('acc2','cal'):.3f} |")
w(f"| CCC | - | {mean('ccc','base'):.3f} | {mean('ccc','cal'):.3f} |")
w(f"| Spearman | - | {mean('spearman','base'):.3f} | {mean('spearman','cal'):.3f} |")
w(f"| MAE | - | {mean('mae','base'):.2f} | {mean('mae','cal'):.2f} |")
w(f"| mean abs. per-fold bias | - | {np.mean([abs(r['base']['bias']) for r in rows]):.1f} | "
  f"{np.mean([abs(r['cal']['bias']) for r in rows]):.1f} |")
w(f"\nfolds improved by recalibration: CCC {sum(r['cal']['ccc']>r['base']['ccc'] for r in rows)}/{len(rows)}, "
  f"MAE {sum(r['cal']['mae']<r['base']['mae'] for r in rows)}/{len(rows)}")
w(f"mean estimated scale factor {np.mean([r['scale'] for r in rows]):.3f}\n")

json.dump(rows, open(OUT / "calibration.json", "w"), indent=1)
Path(OUT / "calibration.md").write_text("\n".join(lines))
