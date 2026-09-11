"""Corpus-level statistics: session inventory, annotation properties, coverage,
label distributions, peer-correlation-vs-lag, and cross-date feature scale.

Reads the built graph datasets and the raw annotation traces; writes
corpus_numbers.md plus corpus_cache.npz for the figure scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameo_stats import ROOT, to_class

sys.path.insert(0, str(ROOT / "modelTraining"))
from build_continuous_dataset import DATASET_STABLE_RANGES, DATASET_TRACKED_SESSIONS  # noqa: E402

OUT = Path(__file__).resolve().parent
DATES = ["03_20", "03_26", "05_15", "05_14"]
POS = ["A", "B", "C"]
lines: list[str] = []


def w(s: str = ""):
    lines.append(s); print(s)


# ------------------------------------------------------------------ traces
def trace_paths(date, sess):
    base = ROOT / "outputs" / date / "continuous_labels" / f"session_{sess}"
    return {p: base / f"participant_{p}.csv" for p in POS}


def load_traces(date, sess):
    out = {}
    for p, path in trace_paths(date, sess).items():
        if path.exists():
            out[p] = pd.read_csv(path)
    return out


def fps_of(date, sess):
    meta = ROOT / "outputs" / date / "gaze_target" / f"session_{sess}" / "session_metadata.json"
    return json.loads(meta.read_text())["stream"]["fps"] if meta.exists() else np.nan


# ------------------------------------------------- session inventory table
w("# Corpus numbers (auto-generated)\n")
w("## 1. Session inventory\n")
w("| date | session | fps | duration (s) | annotated frames | used from (s) | reconciliation | graphs | person-frames | mean label |")
w("|---|---|---|---|---|---|---|---|---|---|")

inventory = []
ds_cache = {}
for date in DATES:
    data = torch.load(ROOT / "modelTraining" / f"graph_dataset_{date}_continuous.pt", weights_only=False)
    ds_cache[date] = data
    by_sess = {}
    for it in data:
        by_sess.setdefault(it["session"], []).append(it)
    for sess in sorted(set(list(by_sess) + list(DATASET_STABLE_RANGES.get(date, {})))):
        tr = load_traces(date, sess)
        fps = fps_of(date, sess)
        nframes = max((len(t) for t in tr.values()), default=0)
        dur = nframes / fps if fps and nframes else np.nan
        rng = DATASET_STABLE_RANGES.get(date, {}).get(sess, None)
        rng_s = "-" if rng is None else (f"{rng[0]:.0f}-{rng[1]:.0f}" if isinstance(rng, tuple) else f"{rng:.0f}")
        method = "tracker" if sess in DATASET_TRACKED_SESSIONS.get(date, set()) else ("rank" if rng is not None else "excluded")
        items = by_sess.get(sess, [])
        pf = sum(len(it["labels"]) for it in items)
        meanlab = float(np.mean([l for it in items for l in it["labels"].tolist()])) if items else np.nan
        row = dict(date=date, session=sess, fps=fps, duration=dur, nframes=nframes, rng=rng_s,
                   method=method, graphs=len(items), person_frames=pf, mean_label=meanlab)
        inventory.append(row)
        w(f"| {date} | {sess} | {fps:.0f} | {dur:.0f} | {nframes} | {rng_s} | {method} | {len(items)} | {pf} | "
          + (f"{meanlab:.1f} |" if items else "- |"))

tot_g = sum(r["graphs"] for r in inventory); tot_pf = sum(r["person_frames"] for r in inventory)
tot_raw = sum(r["nframes"] * min(3, 3) for r in inventory)
w(f"\n**Totals**: {tot_g} graphs, {tot_pf} person-frames, "
  f"{sum(r['nframes'] for r in inventory)*3:,} raw frame-level ratings before subsampling, "
  f"{sum(r['duration'] for r in inventory if not np.isnan(r['duration']))/60:.0f} min of annotated footage, "
  f"{len([r for r in inventory if r['graphs']>0])} sessions used of {len(inventory)} annotated.\n")

# ------------------------------------------------------- label distribution
w("## 2. Label distribution\n")
w("| date | n person-frames | mean | sd | %disengaged | %low | %medium | %high |")
w("|---|---|---|---|---|---|---|---|")
label_by_date = {}
for date in DATES:
    labs = np.array([l for it in ds_cache[date] for l in it["labels"].tolist()])
    label_by_date[date] = labs
    cls = to_class(labs)
    frac = [float((cls == k).mean()) for k in range(4)]
    w(f"| {date} | {len(labs)} | {labs.mean():.1f} | {labs.std():.1f} | " + " | ".join(f"{f*100:.1f}" for f in frac) + " |")
alll = np.concatenate([label_by_date[d] for d in DATES])
cls = to_class(alll)
w(f"| **all** | {len(alll)} | {alll.mean():.1f} | {alll.std():.1f} | "
  + " | ".join(f"{(cls==k).mean()*100:.1f}" for k in range(4)) + " |\n")

# --------------------------------------------- peer correlation versus lag
w("## 3. Cross-participant engagement correlation versus lag\n")
LAGS = [0, 0.5, 1, 2, 3, 5, 10]
rows_lag = []
for date in DATES:
    for sess in sorted(DATASET_STABLE_RANGES.get(date, {})):
        tr = load_traces(date, sess)
        if len(tr) < 2:
            continue
        fps = fps_of(date, sess)
        for lag in LAGS:
            k = int(round(lag * fps))
            rs = []
            for a in POS:
                for b in POS:
                    if a == b or a not in tr or b not in tr:
                        continue
                    x = tr[a]["engagement_pct"].to_numpy()
                    y = tr[b]["engagement_pct"].to_numpy()
                    n = min(len(x), len(y))
                    if k >= n - 10:
                        continue
                    xa = x[k:n]; yb = y[:n - k] if k else y[:n]
                    if xa.std() < 1e-6 or yb.std() < 1e-6:
                        continue
                    rs.append(np.corrcoef(xa, yb)[0, 1])
            if rs:
                rows_lag.append(dict(date=date, session=sess, lag=lag, r=float(np.mean(rs))))
lag_df = pd.DataFrame(rows_lag)
w("| lag (s) | mean r | sd across sessions | min | max |")
w("|---|---|---|---|---|")
for lag in LAGS:
    s = lag_df[lag_df.lag == lag]["r"]
    w(f"| {lag} | {s.mean():.3f} | {s.std():.3f} | {s.min():.3f} | {s.max():.3f} |")
w("")

# ------------------------------------------------------- anchoring check
w("## 4. Anchoring-bias check (max |r| between participants within a session, lag 0)\n")
z = lag_df[lag_df.lag == 0]
w(f"mean {z['r'].mean():.3f}, max {z['r'].max():.3f} ({z.loc[z['r'].idxmax(),'date']} s{int(z.loc[z['r'].idxmax(),'session'])}), "
  f"threshold for re-annotation 0.90, sessions above threshold: {(z['r']>0.9).sum()}\n")

# ------------------------------------------------------ feature coverage
w("## 5. Feature coverage and cross-date feature scale\n")
w("(node feature layout: 0-3 gaze one-hot, 4-5 centre, 6-7 size, 8 of_availability, 9 face_conf, "
  "10-11 gaze angles, 12-19 AU, 20-27 emotion one-hot, 28 mouth motion)\n")
w("| date | facial coverage | mean AU activation | mean mouth motion | gaze at robot | gaze at peer | gaze elsewhere |")
w("|---|---|---|---|---|---|---|")
cov_rows = []
for date in DATES:
    last = torch.cat([it["node_features"][:, -1, :] for it in ds_cache[date]]).numpy()
    cov = float(last[:, 8].mean())
    au = float(last[:, 12:20][last[:, 8] > 0].mean())
    mm = float(last[:, 28].mean())
    g = [float(last[:, k].mean()) for k in range(4)]  # robot, elsewhere, participant, unknown
    cov_rows.append(dict(date=date, coverage=cov, au=au, mouth=mm, robot=g[0], elsewhere=g[1], participant=g[2]))
    w(f"| {date} | {cov*100:.1f}% | {au:.3f} | {mm:.2f} | {g[0]*100:.1f}% | {g[2]*100:.1f}% | {g[1]*100:.1f}% |")
w("")

# per-session coverage for the figure
sess_cov = []
for date in DATES:
    by = {}
    for it in ds_cache[date]:
        by.setdefault(it["session"], []).append(it["node_features"][:, -1, :])
    for s, v in sorted(by.items()):
        arr = torch.cat(v).numpy()
        sess_cov.append(dict(date=date, session=s, coverage=float(arr[:, 8].mean()),
                             au=float(arr[:, 12:20][arr[:, 8] > 0].mean()) if (arr[:, 8] > 0).any() else np.nan))

# ------------------------------------------------------------ node counts
w("## 6. Graph sizes\n")
sizes = np.array([len(it["labels"]) for d in DATES for it in ds_cache[d]])
w("| nodes per graph | count | share |")
w("|---|---|---|")
for k in sorted(set(sizes.tolist())):
    w(f"| {k} | {(sizes==k).sum()} | {(sizes==k).mean()*100:.1f}% |")
w("")

np.savez(OUT / "corpus_cache.npz",
         labels_all=alll,
         **{f"labels_{d}": label_by_date[d] for d in DATES},
         lag_lags=np.array(lag_df["lag"]), lag_r=np.array(lag_df["r"]),
         lag_date=np.array(lag_df["date"]), lag_sess=np.array(lag_df["session"]),
         sizes=sizes)
json.dump({"inventory": inventory, "coverage": cov_rows, "session_coverage": sess_cov},
          open(OUT / "corpus_numbers.json", "w"), indent=1, default=float)
Path(OUT / "corpus_numbers.md").write_text("\n".join(lines))
print("\nwrote", OUT / "corpus_numbers.md")
