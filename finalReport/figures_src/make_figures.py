"""Generate every data figure in the report from the saved predictions,
datasets and annotation traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cameo_stats import ROOT, MT, load, fold_slices, ccc, to_class, granularity_metrics, ANCHORS
from style import (FIGDIR, FULL, HALF, SERIES, BLUE, ORANGE, AQUA, VIOLET, RED, INK, INK2, MUTED,
                   GRID, SEQ, DATE_COLOR, LEVEL_COLOR, LEVELS, grid, save)

DATES = ["03_20", "03_26", "05_15", "05_14"]
POS = ["A", "B", "C"]
_dscache: dict[str, list] = {}


def dataset(date: str):
    if date not in _dscache:
        _dscache[date] = torch.load(MT / f"graph_dataset_{date}_continuous.pt", weights_only=False)
    return _dscache[date]


def fold_frame(res: dict, date: str, session: int) -> pd.DataFrame:
    """Rebuild (time, participant, true, predicted) for one held-out fold."""
    preds = np.array(res["all_preds"]); labels = np.array(res["all_labels"])
    for f, sl in fold_slices(res):
        if f["dataset"] == date and f["session"] == session:
            items = [it for it in dataset(date) if it["session"] == session]
            rows, k = [], sl.start
            for it in items:
                for n, p in enumerate(it["participants"]):
                    rows.append((it["timestamp_sec"], p, float(it["labels"][n]), preds[k]))
                    k += 1
            assert k == sl.stop, (k, sl.stop)
            return pd.DataFrame(rows, columns=["t", "participant", "true", "pred"])
    raise KeyError((date, session))


def traces(date, session):
    base = ROOT / "outputs" / date / "continuous_labels" / f"session_{session}"
    return {p: pd.read_csv(base / f"participant_{p}.csv") for p in POS if (base / f"participant_{p}.csv").exists()}


# =====================================================================  1
def fig_traces():
    date, session, zoom = "03_26", 6, (120, 200)
    tr = traces(date, session)
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.3))
    for k, ax in enumerate(axes):
        for a in ANCHORS[1:-1]:
            ax.axhline(float(a), color=GRID, lw=0.7, zorder=0)
        for i, (p_, df) in enumerate(sorted(tr.items())):
            ax.plot(df["timestamp_sec"], df["engagement_pct"], color=SERIES[i], lw=1.0 if k == 0 else 1.5,
                    label=f"participant {p_}")
        ax.set_ylim(-3, 103); ax.set_ylabel("engagement (0-100)")
        for y, lab in zip([0, 33.3, 66.7, 100], LEVELS):
            ax.text(1.004, y / 100, lab, transform=ax.transAxes, va="center", ha="left",
                    fontsize=6.5, color=MUTED)
    axes[0].set_xlim(0, df["timestamp_sec"].max())
    axes[0].axvspan(*zoom, color=ORANGE, alpha=0.10, zorder=0)
    axes[0].set_title("(a) whole session, three simultaneous annotation passes", loc="left")
    axes[1].set_xlim(*zoom)
    axes[1].set_title(f"(b) {zoom[1]-zoom[0]} s detail from the shaded region", loc="left")
    axes[1].set_xlabel("time into session (s)")
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.16))
    fig.subplots_adjust(hspace=0.42)
    save(fig, "fig_traces")


# =====================================================================  2
def fig_label_dist():
    c = np.load(HERE / "corpus_cache.npz", allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.hist(c["labels_all"], bins=50, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    grid(ax)
    for a in ANCHORS:
        ax.axvline(float(a), color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=3)
    ax.set_xlabel("annotated engagement (0-100)"); ax.set_ylabel("person-frames")
    ax.set_title("(a) pooled annotation distribution", loc="left")
    ax.text(0.5, 0.94, "dashed: four-level class anchors", transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

    ax = axes[1]
    left = np.zeros(len(DATES))
    for k in range(4):
        vals = np.array([(to_class(c[f"labels_{d}"]) == k).mean() * 100 for d in DATES])
        ax.barh(np.arange(len(DATES)), vals, left=left, color=LEVEL_COLOR[k], height=0.62,
                edgecolor="white", linewidth=1.0, label=LEVELS[k])
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 7:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                        color="white" if k >= 2 else INK)
        left += vals
    ax.set_yticks(np.arange(len(DATES))); ax.set_yticklabels(DATES)
    ax.set_xlim(0, 100); ax.set_xlabel("share of person-frames (%)")
    ax.set_title("(b) class balance by recording date", loc="left")
    ax.invert_yaxis(); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.40), columnspacing=1.0)
    save(fig, "fig_label_dist")


# =====================================================================  3
def fig_lag_corr():
    c = np.load(HERE / "corpus_cache.npz", allow_pickle=True)
    lags, r, dts, sess = c["lag_lags"], c["lag_r"], c["lag_date"], c["lag_sess"]
    fig, ax = plt.subplots(figsize=(HALF * 1.55, 2.3))
    ulags = np.unique(lags)
    for d in DATES:
        for s in np.unique(sess[dts == d]):
            m = (dts == d) & (sess == s)
            ax.plot(lags[m], r[m], color=DATE_COLOR[d], lw=0.8, alpha=0.35)
    mean = [r[lags == l].mean() for l in ulags]
    ax.plot(ulags, mean, color=INK, lw=2.2, zorder=5, label="mean over 20 sessions")
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("lag applied to the peer's trace (s)"); ax.set_ylabel("correlation $r$")
    grid(ax)
    handles = [Line2D([], [], color=INK, lw=2.2, label="mean over sessions")] + \
              [Line2D([], [], color=DATE_COLOR[d], lw=1.0, label=d) for d in DATES]
    ax.legend(handles=handles, ncol=1, loc="upper right", fontsize=7)
    ax.annotate(f"{mean[0]:.2f}", (ulags[0], mean[0]), textcoords="offset points", xytext=(4, 6),
                fontsize=7, color=INK)
    i1 = list(ulags).index(1.0)
    ax.annotate(f"{mean[i1]:.2f} at 1 s", (1.0, mean[i1]), textcoords="offset points", xytext=(10, -18),
                fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=2))
    save(fig, "fig_lag_corr")


# =====================================================================  4
def fig_coverage():
    j = json.load(open(HERE / "corpus_numbers.json"))
    rows = [r for r in j["session_coverage"]]
    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    x = np.arange(len(rows))
    ax.bar(x, [r["coverage"] * 100 for r in rows], color=[DATE_COLOR[r["date"]] for r in rows],
           width=0.68, zorder=3)
    grid(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"s{r['session']}" for r in rows], fontsize=7)
    ax.set_ylabel("frames with a matched face (%)"); ax.set_ylim(0, 100)
    ax.axhline(np.mean([r["coverage"] for r in rows]) * 100, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=4)
    prev, start = None, 0
    for i, r in enumerate(rows + [{"date": None}]):
        if r["date"] != prev:
            if prev is not None:
                ax.text((start + i - 1) / 2, -19, prev, ha="center", fontsize=7.5, color=DATE_COLOR[prev])
            prev, start = r["date"], i
    mu = np.mean([r["coverage"] for r in rows]) * 100
    ax.text(len(rows) - 0.55, mu + 1.5, f"mean {mu:.0f}%", ha="right", va="bottom", fontsize=7,
            color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(-1.4, len(rows) - 0.4)
    save(fig, "fig_coverage")


# =====================================================================  5
def fig_confusion():
    res = load("loso_results_3sess_noembed.pt")
    hard = np.array(res["all_hard_preds"]); true = np.array(res["all_true_classes"])
    m3 = np.vectorize({0: 0, 1: 1, 2: 1, 3: 2}.get)
    sets = [(true, hard, LEVELS, "(a) four-level"),
            (m3(true), m3(hard), ["disengaged", "low/medium", "high"], "(b) three-level"),
            ((true >= 2).astype(int), (hard >= 2).astype(int), ["disengaged", "engaged"], "(c) binary")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5))
    fig.subplots_adjust(wspace=0.75)
    for ax, (t, p, labs, title) in zip(axes, sets):
        cm = confusion_matrix(t, p, normalize="true") * 100
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=100)
        for i in range(len(labs)):
            for jj in range(len(labs)):
                ax.text(jj, i, f"{cm[i,jj]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, jj] > 55 else INK)
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=7, rotation=30, ha="right")
        ax.set_yticklabels(labs, fontsize=7)
        ax.tick_params(axis="y", pad=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("predicted"); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("annotated")
    save(fig, "fig_confusion")


# =====================================================================  6
def fig_pred_true():
    res = load("loso_results_3sess_noembed.pt")
    p = np.array(res["all_preds"]); t = np.array(res["all_labels"])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    hb = ax.hexbin(t, p, gridsize=42, cmap=SEQ, bins="log", mincnt=1, linewidths=0)
    ax.plot([0, 100], [0, 100], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=4)
    bins = np.arange(0, 101, 10)
    idx = np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2)
    mids = bins[:-1] + 5
    means = np.array([p[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(mids))])
    ax.plot(mids, means, color=ORANGE, lw=2.0, marker="o", ms=3.5, zorder=5)
    ax.set_xlabel("annotated engagement"); ax.set_ylabel("predicted score")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_title("(a) predictions versus ground truth", loc="left")
    ax.text(3, 92, "orange: mean prediction\nper 10-point band", fontsize=7, color=ORANGE, va="top")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.040); cb.set_label("person-frames", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)

    ax = axes[1]
    ax.hist(t, bins=40, color=GRID, label="annotated", zorder=2)
    ax.hist(p, bins=40, histtype="step", color=BLUE, lw=1.8, label="predicted", zorder=3)
    ax.set_yscale("log"); grid(ax)
    ax.set_xlabel("engagement (0-100)"); ax.set_ylabel("person-frames (log)")
    ax.set_title("(b) marginal distributions", loc="left"); ax.legend(loc="lower center")
    ax.text(0.5, 0.90, f"sd {t.std():.0f} annotated vs {p.std():.0f} predicted", transform=ax.transAxes,
            fontsize=7, ha="center", color=INK2)
    save(fig, "fig_pred_true")


# =====================================================================  7
def fig_per_fold():
    j = json.load(open(HERE / "report_numbers.json"))
    rows = j["per_fold"]
    labels = [f"{r['dataset']}\ns{r['session']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.5), sharex=True)
    for ax, key, lab in [(axes[0], "ccc", "CCC"), (axes[1], "f1_4", "macro-F1 (4-level)")]:
        v = [r[key] for r in rows]
        ax.bar(x, v, color=[DATE_COLOR[r["dataset"]] for r in rows], width=0.68, zorder=3)
        ax.axhline(np.mean(v), color=INK, lw=1.0, ls=(0, (4, 2)), zorder=4)
        ax.text(0.62, np.mean(v) + 0.010, f"mean {np.mean(v):.3f}", ha="left",
                va="bottom", fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        ax.set_xlim(-0.7, len(rows) - 0.3)
        grid(ax); ax.set_ylabel(lab)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylim(0, 0.95); axes[1].set_ylim(0, 0.68)
    save(fig, "fig_per_fold")


# =====================================================================  8
def fig_granularity():
    j = json.load(open(HERE / "report_numbers.json"))
    m = j["headline_fold_means"]
    names = ["four-level", "three-level", "binary"]
    acc = [m["acc4"], m["acc3"], m["acc2"]]
    f1 = [m["f1_4"], m["f1_3"], m["f1_2"]]
    res = load("loso_results_3sess_noembed.pt")
    true_cls = np.array(res["all_true_classes"])
    m3 = np.vectorize({0: 0, 1: 1, 2: 1, 3: 2}.get)(true_cls)
    chance = [max(np.bincount(true_cls)) / len(true_cls),
              max(np.bincount(m3)) / len(m3),
              max(np.bincount((true_cls >= 2).astype(int))) / len(true_cls)]
    x = np.arange(3); wd = 0.34
    fig, ax = plt.subplots(figsize=(HALF * 1.5, 2.3))
    ax.bar(x - wd / 2, acc, wd, color=BLUE, label="accuracy", zorder=3)
    ax.bar(x + wd / 2, f1, wd, color=ORANGE, label="macro-F1", zorder=3)
    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax.text(xi - wd / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=7, color=INK)
        ax.text(xi + wd / 2, f + 0.015, f"{f:.3f}", ha="center", fontsize=7, color=INK)
    for xi, c in zip(x, chance):
        ax.plot([xi - 0.42, xi + 0.42], [c, c], color=INK, lw=1.2, ls=(0, (3, 2)), zorder=5)
    ax.text(0.02, chance[0] - 0.055, "majority-class baseline", fontsize=6.8, color=INK, ha="left")
    grid(ax); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0); ax.set_ylabel("mean over 14 folds"); ax.legend(ncol=2, loc="upper left")
    save(fig, "fig_granularity")


# =====================================================================  9
def fig_ablation():
    abl = torch.load(MT / "loso_results_03_20_03_26_05_15_continuous.pt", weights_only=False)

    def per_fold(res):
        p = np.array(res["all_preds"]); t = np.array(res["all_labels"])
        return {(f["dataset"], f["session"]): (ccc(t[sl], p[sl]), f["macro_f1"]) for f, sl in fold_slices(res)}

    full, nog, orl = (per_fold(abl[k]) for k in
                      ("full_classification", "no_graph_classification", "full_ordinal"))
    keys = sorted(full)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax.plot([nog[k][0], full[k][0]], [i, i], color=GRID, lw=1.4, zorder=2)
    ax.scatter([nog[k][0] for k in keys], y, s=26, color=ORANGE, zorder=3, label="no relational module")
    ax.scatter([full[k][0] for k in keys], y, s=26, color=BLUE, zorder=3, label="full model")
    ax.set_yticks(y); ax.set_yticklabels([f"{d} s{s_}" for d, s_ in keys], fontsize=7)
    ax.invert_yaxis(); grid(ax, axis="x")
    ax.set_xlabel("CCC")
    ax.set_title("(a) relational ablation, all 13 folds", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    ax.margins(y=0.07)
    ax.text(0.98, 0.97, f"the full model is higher in "
            f"{sum(full[k][0] > nog[k][0] for k in keys)} of {len(keys)} folds",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

    ax = axes[1]
    shared = sorted(orl)
    names = ["full", "no relational\nmodule", "ordinal\nobjective"]
    vals = [np.mean([d[k][0] for k in shared]) for d in (full, nog, orl)]
    f1s = [np.mean([d[k][1] for k in shared]) for d in (full, nog, orl)]
    x = np.arange(3); wd = 0.36
    ax.bar(x - wd / 2, vals, wd, color=BLUE, label="CCC", zorder=3)
    ax.bar(x + wd / 2, f1s, wd, color=ORANGE, label="macro-F1", zorder=3)
    for xi, (a_, b_) in enumerate(zip(vals, f1s)):
        ax.text(xi - wd / 2, a_ + 0.012, f"{a_:.3f}", ha="center", fontsize=7, color=INK)
        ax.text(xi + wd / 2, b_ + 0.012, f"{b_:.3f}", ha="center", fontsize=7, color=INK)
    grid(ax); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 0.78); ax.set_ylabel("mean over the 3 shared folds")
    ax.set_title("(b) objective probe, matched folds", loc="left")
    ax.legend(ncol=2, loc="upper right")
    save(fig, "fig_ablation")


# =====================================================================  10
def fig_shrinkage():
    res = load("loso_results_3sess_noembed.pt")
    p = np.array(res["all_preds"]); t = np.array(res["all_labels"])
    folds = list(fold_slices(res))
    mt = np.array([t[sl].mean() for _, sl in folds]); mp = np.array([p[sl].mean() for _, sl in folds])
    st = np.array([t[sl].std() for _, sl in folds]); sp = np.array([p[sl].std() for _, sl in folds])
    cols = [DATE_COLOR[f["dataset"]] for f, _ in folds]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    ax = axes[0]
    ax.plot([20, 75], [20, 75], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2, label="perfect calibration")
    lr = linregress(mt, mp)
    xs = np.linspace(20, 75, 10)
    ax.plot(xs, lr.slope * xs + lr.intercept, color=ORANGE, lw=1.8, zorder=3,
            label=f"fit: slope {lr.slope:.2f}")
    ax.scatter(mt, mp, s=34, color=cols, zorder=4, edgecolor="white", linewidth=0.6)
    # place labels by trying candidate offsets and keeping the first that does
    # not collide with a label already placed (measured in display space)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken: list[tuple[float, float, float, float]] = []
    cand = [(6, 3), (6, -10), (-6, 3), (-6, -10), (6, 12), (-6, 12), (6, -19), (-6, -19),
            (6, 21), (-6, 21), (6, -28), (-6, -28), (0, 12), (0, -19)]
    for f, x0, y0 in zip([f for f, _ in folds], mt, mp):
        px, py = ax.transData.transform((x0, y0))
        txt = f"{f['dataset'][-2:]}s{f['session']}"
        for dx, dy in cand:
            ha = "left" if dx > 0 else "right"
            wpt, hpt = 4.2 * len(txt), 7.0
            x1 = px + dx if ha == "left" else px + dx - wpt
            y1 = py + dy
            bb = (x1, y1, x1 + wpt, y1 + hpt)
            if all(bb[2] < q[0] or bb[0] > q[2] or bb[3] < q[1] or bb[1] > q[3] for q in taken):
                break
        taken.append(bb)
        ax.annotate(txt, (x0, y0), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=5.8, color=INK2)
    ax.set_xlabel("session's annotated mean"); ax.set_ylabel("session's predicted mean")
    ax.set_title("(a) between-session level", loc="left"); grid(ax, axis="both"); ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot([30, 48], [30, 48], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.scatter(st, sp, s=34, color=cols, zorder=4, edgecolor="white", linewidth=0.6)
    ax.set_xlim(30, 48); ax.set_ylim(10, 48)
    ratio = (sp / st).mean()
    xs = np.linspace(30, 48, 10)
    ax.plot(xs, ratio * xs, color=ORANGE, lw=1.8, zorder=3, label=f"mean ratio {ratio:.2f}")
    ax.set_xlabel("annotated within-session sd"); ax.set_ylabel("predicted within-session sd")
    ax.set_title("(b) within-session variation", loc="left"); grid(ax, axis="both"); ax.legend(loc="upper left")
    handles = [Line2D([], [], marker="o", ls="", color=DATE_COLOR[d], label=d) for d in DATES[:3]]
    axes[1].legend(handles=handles + [Line2D([], [], color=ORANGE, lw=1.8, label=f"ratio {ratio:.2f}")],
                   loc="upper left", fontsize=6.8)
    save(fig, "fig_shrinkage")


# =====================================================================  11
def fig_calibration():
    rows = json.load(open(HERE / "calibration.json"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.annotate("", xy=(r["cal"]["ccc"], i), xytext=(r["base"]["ccc"], i),
                    arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.3, shrinkA=0, shrinkB=0))
    ax.scatter([r["base"]["ccc"] for r in rows], y, s=24, color=MUTED, zorder=3, label="expected-value score")
    ax.scatter([r["cal"]["ccc"] for r in rows], y, s=24, color=AQUA, zorder=3, label="after recalibration")
    ax.set_yticks(y); ax.set_yticklabels([r["fold"] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("CCC"); grid(ax, axis="x")
    ax.set_title("(a) concordance per fold", loc="left", y=1.10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    ax = axes[1]
    res = load("loso_results_3sess_noembed.pt")
    p = np.array(res["all_preds"]); t = np.array(res["all_labels"])
    folds = list(fold_slices(res))
    cal = np.concatenate([
        np.clip(np.mean([t[folds[j][1]].mean() for j in range(len(folds)) if j != i])
                + (p[sl] - np.mean([p[folds[j][1]].mean() for j in range(len(folds)) if j != i]))
                / np.mean([p[folds[j][1]].std() / t[folds[j][1]].std() for j in range(len(folds)) if j != i]),
                0, 100)
        for i, (_, sl) in enumerate(folds)])
    ax.hist(t, bins=40, color=GRID, label="annotated", zorder=2)
    ax.hist(p, bins=40, histtype="step", color=MUTED, lw=1.6, label="expected-value", zorder=3)
    ax.hist(cal, bins=40, histtype="step", color=AQUA, lw=1.8, label="recalibrated", zorder=4)
    grid(ax); ax.set_xlabel("engagement (0-100)"); ax.set_ylabel("person-frames")
    ax.set_title("(b) distribution of predicted values", loc="left", y=1.10)
    ax.legend(loc="upper center")
    save(fig, "fig_calibration")


# =====================================================================  12
def fig_trace_pred():
    res = load("loso_results_3sess_noembed.pt")
    picks = [("05_15", 2, "(a) strongest fold: 05\\_15 session 2"), ("03_26", 2, "(b) weakest fold: 03\\_26 session 2")]
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.6))
    for ax, (d, s, title) in zip(axes, picks):
        df = fold_frame(res, d, s)
        part = sorted(df["participant"].unique())[0]
        sub = df[df["participant"] == part].sort_values("t")
        ax.plot(sub["t"], sub["true"], color=INK, lw=1.4, label="annotated")
        ax.plot(sub["t"], sub["pred"], color=BLUE, lw=1.4, label="predicted")
        ax.fill_between(sub["t"], sub["true"], sub["pred"], color=BLUE, alpha=0.12, linewidth=0)
        ax.set_ylim(-3, 103); ax.set_xlim(sub["t"].min(), sub["t"].max())
        ax.set_ylabel("engagement")
        ax.set_title(f"{title.replace(chr(92)+'_','_')}, participant {part}", loc="left")
        grid(ax)
    axes[1].set_xlabel("time into session (s)")
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
    fig.subplots_adjust(hspace=0.55)
    save(fig, "fig_trace_pred")


# =====================================================================  13
def fig_au_domain():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw={"width_ratios": [1, 1]})
    data, covs = [], []
    for d in DATES:
        arr = torch.cat([it["node_features"][:, -1, :] for it in dataset(d)]).numpy()
        have = arr[:, 8] > 0
        data.append(arr[have][:, 12:20].mean(1))
        covs.append(have.mean() * 100)
    ax = axes[0]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, d in zip(parts["bodies"], DATES):
        pc.set_facecolor(DATE_COLOR[d]); pc.set_alpha(0.75); pc.set_edgecolor("white")
    for i, v in enumerate(data):
        ax.plot([i + 1], [np.mean(v)], marker="o", color=INK, ms=3.5, zorder=5)
        ax.text(i + 1.12, np.mean(v), f"{np.mean(v):.3f}", fontsize=6.8, va="center", color=INK)
    ax.set_xticks(range(1, len(DATES) + 1)); ax.set_xticklabels(DATES)
    ax.set_ylabel("mean action-unit activation"); grid(ax)
    ax.set_title("(a) facial activation by recording date", loc="left")
    ax.set_ylim(0, 0.6)

    ax = axes[1]
    gz = []
    for d in DATES:
        arr = torch.cat([it["node_features"][:, -1, :] for it in dataset(d)]).numpy()
        gz.append([arr[:, 0].mean() * 100, arr[:, 2].mean() * 100, arr[:, 1].mean() * 100])
    gz = np.array(gz)
    left = np.zeros(len(DATES))
    for k, (lab, col) in enumerate(zip(["at the robot", "at a peer", "elsewhere"], [BLUE, ORANGE, GRID])):
        ax.barh(np.arange(len(DATES)), gz[:, k], left=left, color=col, height=0.6,
                edgecolor="white", linewidth=1.0, label=lab)
        for i, (v, l) in enumerate(zip(gz[:, k], left)):
            if v > 8:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                        color="white" if k < 2 else INK)
        left += gz[:, k]
    ax.set_yticks(np.arange(len(DATES))); ax.set_yticklabels(DATES); ax.invert_yaxis()
    ax.set_xlabel("share of person-frames (%)"); ax.set_xlim(0, 100)
    ax.set_title("(b) gaze target by recording date", loc="left")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.38), columnspacing=1.0)
    save(fig, "fig_au_domain")




# =====================================================================  14
def fig_transitions():
    """How much engagement variation a fixed-length segment label would discard."""
    windows = [1, 3, 5, 10, 15, 30]
    frac, runs = {w: [] for w in windows}, []
    for date in DATES:
        for it_sess in sorted({it["session"] for it in dataset(date)}):
            fps = 60.0 if date == "05_14" else 30.0
            for p, df in traces(date, it_sess).items():
                v = to_class(df["engagement_pct"].to_numpy())
                ch = np.diff(v) != 0
                # run lengths of constant level
                idx = np.flatnonzero(np.concatenate(([True], ch, [True])))
                runs.extend(np.diff(idx) / fps)
                for w in windows:
                    k = int(w * fps)
                    n = len(v) // k
                    if n == 0:
                        continue
                    blocks = v[:n * k].reshape(n, k)
                    frac[w].append(float(np.mean([len(np.unique(b)) > 1 for b in blocks])))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    ax = axes[0]
    means = [np.mean(frac[w]) * 100 for w in windows]
    ax.bar(range(len(windows)), means, color=BLUE, width=0.62, zorder=3)
    for i, m in enumerate(means):
        ax.text(i, m + 1.5, f"{m:.0f}", ha="center", fontsize=7, color=INK)
    grid(ax); ax.set_xticks(range(len(windows))); ax.set_xticklabels([f"{w}s" for w in windows])
    ax.set_ylim(0, 100)
    ax.set_xlabel("hypothetical segment length"); ax.set_ylabel("segments spanning\nmore than one level (%)")
    ax.set_title("(a) cost of segment-level labelling", loc="left")
    i10 = windows.index(10)
    ax.annotate("segment length of\ncomparable corpora", (i10 - 0.32, means[i10] + 4),
                xycoords="data", xytext=(0.02, 0.93), textcoords="axes fraction",
                fontsize=6.8, color=ORANGE, ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0,
                                connectionstyle="arc3,rad=-0.15", shrinkA=2, shrinkB=2))
    ax = axes[1]
    runs = np.array(runs)
    ax.hist(np.clip(runs, 0, 30), bins=60, color=BLUE, zorder=3)
    ax.set_yscale("log"); grid(ax)
    ax.axvline(float(np.median(runs)), color=ORANGE, lw=1.6, zorder=4)
    ax.text(float(np.median(runs)) + 0.8, ax.get_ylim()[1] * 0.35,
            f"median {np.median(runs):.1f} s", color=ORANGE, fontsize=7)
    ax.set_xlabel("time spent at one engagement level (s), clipped at 30"); ax.set_ylabel("count (log)")
    ax.set_title("(b) dwell time at a level", loc="left")
    save(fig, "fig_transitions")
    print(f"    [stat] segments spanning >1 level: " +
          ", ".join(f"{w}s={np.mean(frac[w])*100:.0f}%" for w in windows) +
          f"; median dwell {np.median(runs):.2f}s, mean {runs.mean():.2f}s")


# =====================================================================  15
def fig_fourdate():
    """Three-date versus four-date pooling, on the folds both evaluations share."""
    import torch as _t
    four_res = _t.load(MT / "loso_results_4dates_full.pt", weights_only=False)["full_classification"]
    p4 = np.array(four_res["all_preds"]); t4 = np.array(four_res["all_labels"])
    four = {}
    for f, sl in fold_slices(four_res):
        four[(f["dataset"], f["session"])] = (f["macro_f1"], ccc(t4[sl], p4[sl]))
    j = json.load(open(HERE / "report_numbers.json"))
    three = {(r["dataset"], r["session"]): r for r in j["per_fold"]}
    shared = sorted(k for k in three if k in four)

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    x = np.arange(len(shared)); wd = 0.36
    a = [three[k]["f1_4"] for k in shared]
    b = [four[k][0] for k in shared]
    ax.bar(x - wd / 2, a, wd, color=BLUE, label="three dates (14 folds)", zorder=3)
    ax.bar(x + wd / 2, b, wd, color=VIOLET, label="four dates (20 folds)", zorder=3)
    grid(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"s{s_}" for _d, s_ in shared], fontsize=7)
    ax.set_ylabel("macro-F1 (4-level)"); ax.set_ylim(0, 0.68)
    prev, start = None, 0
    for i, (d, _s) in enumerate(shared + [(None, 0)]):
        if d != prev:
            if prev is not None:
                ax.plot([start - 0.38, i - 1 + 0.38], [-0.115, -0.115], color=MUTED, lw=0.8,
                        clip_on=False, transform=ax.get_xaxis_transform())
                ax.text((start + i - 1) / 2, -0.175, prev, ha="center", va="top", fontsize=7,
                        color=INK2, transform=ax.get_xaxis_transform())
            prev, start = d, i
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.set_title("(a) folds evaluated in both runs", loc="left", y=1.12)

    ax = axes[1]
    names = ["three dates\n14 folds", "four dates\n20 folds"]
    ccc_v = [0.649, np.mean([v[1] for v in four.values()])]
    cal_v = [0.728, 0.680]
    xx = np.arange(2); wd = 0.36
    ax.bar(xx - wd / 2, ccc_v, wd, color=BLUE, label="CCC", zorder=3)
    ax.bar(xx + wd / 2, cal_v, wd, color=AQUA, label="CCC, recalibrated", zorder=3)
    for i, (u, v) in enumerate(zip(ccc_v, cal_v)):
        ax.text(i - wd / 2, u + 0.012, f"{u:.3f}", ha="center", fontsize=7, color=INK)
        ax.text(i + wd / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    grid(ax); ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 0.88); ax.set_ylabel("mean over that run's folds")
    ax.set_title("(b) whole-run comparison", loc="left", y=1.12)
    ax.legend(ncol=1, loc="upper left", fontsize=6.8, handlelength=1.2)
    save(fig, "fig_fourdate")
    print(f"    [stat] shared {len(shared)}: 3d {np.mean(a):.3f} vs 4d {np.mean(b):.3f}; "
          f"4d better in {sum(y > x for x, y in zip(a, b))}/{len(shared)}")



# =====================================================================  16
def fig_interpersonal():
    """The interpersonal regularities the relational edges are built on."""
    j = json.load(open(HERE / "interpersonal.json"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6),
                             gridspec_kw={"width_ratios": [1.20, 0.95, 0.95]})

    # (a) engagement by attention direction, including two dyadic states
    ax = axes[0]
    g, st = j["gaze"], j["states"]
    names = ["looking at the receptionist", "looking at another participant",
             "looking elsewhere", "in mutual gaze with a peer", "watching a speaking peer"]
    vals = [g["the receptionist"]["mean"], g["another participant"]["mean"],
            g["elsewhere"]["mean"], st["in mutual gaze with a peer"]["on"],
            st["looking at a peer whose mouth is moving"]["on"]]
    cols = [BLUE, ORANGE, GRID, ORANGE, ORANGE]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, height=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=6.8, color=INK)
    grid(ax, axis="x")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=6.6); ax.invert_yaxis()
    ax.set_xlim(0, 92); ax.set_xlabel("mean annotated engagement")
    ax.set_title("(a) attention direction", loc="left")

    # (b) own engagement against the peers' engagement
    ax = axes[1]
    co = j["coupling"]
    xs = np.arange(len(co))
    ax.bar(xs, [c["mean"] for c in co], color=AQUA, width=0.66, zorder=3)
    for i, c in enumerate(co):
        ax.text(i, c["mean"] + 1.8, f"{c['mean']:.0f}", ha="center", fontsize=6.8, color=INK)
    grid(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c['lo']}-{c['hi']}" for c in co], fontsize=6.4, rotation=45, ha="right")
    ax.set_xlabel("peers' mean engagement")
    ax.set_ylabel("own mean engagement")
    ax.set_ylim(0, 100)
    ax.set_title("(b) peer coupling", loc="left")

    # (c) shared level and synchronous change, against a matched chance rate
    ax = axes[2]
    ag, sy = j["agreement"]["overall"], j["sync"]
    labels = ["pair at the\nsame level", "change within\n2 s of a peer"]
    obs = [ag["observed"] * 100, sy["observed"] * 100]
    exp = [ag["expected"] * 100, sy["shuffled"] * 100]
    xs = np.arange(2); wd = 0.32
    ax.bar(xs - wd / 2, obs, wd, color=VIOLET, label="observed", zorder=3)
    ax.bar(xs + wd / 2, exp, wd, color=GRID, label="chance", zorder=3)
    for i, (o, e) in enumerate(zip(obs, exp)):
        ax.text(i - wd / 2, o + 1.8, f"{o:.0f}", ha="center", fontsize=6.8, color=INK)
        ax.text(i + wd / 2, e + 1.8, f"{e:.0f}", ha="center", fontsize=6.8, color=INK)
    grid(ax); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylim(0, 105); ax.set_ylabel("share of cases (%)")
    ax.set_title("(c) group structure", loc="left")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=6.5,
              columnspacing=1.0, handlelength=1.2)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig_interpersonal")


ALL = {name[4:]: fn for name, fn in list(globals().items()) if name.startswith("fig_")}

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    for name in want:
        print(name)
        ALL[name]()
