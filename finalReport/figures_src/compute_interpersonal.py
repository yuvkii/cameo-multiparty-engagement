"""Interpersonal regularities in the corpus: how a participant's annotated
engagement co-varies with what the people around them are doing.

Reads the built graph datasets directly (node features carry the per-person
signals, edge features the measured pairwise ones) and writes
interpersonal.json / interpersonal.md plus a cache for the figure script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameo_stats import ROOT, to_class

OUT = Path(__file__).resolve().parent
DATES = ["03_20", "03_26", "05_15", "05_14"]
MAIN = ["03_20", "03_26", "05_15"]

GAZE_ROBOT, GAZE_PEER, GAZE_ELSE, GAZE_NONE = 0, 1, 2, 3
E_GAZE, E_PROX, E_MUTUAL, E_SPEAKER, E_LAGY = 0, 1, 2, 3, 4

lines: list[str] = []
def w(s: str = ""):
    lines.append(s); print(s)


rows = []           # one per person-frame
for date in DATES:
    data = torch.load(ROOT / "modelTraining" / f"graph_dataset_{date}_continuous.pt", weights_only=False)
    for it in data:
        nf = it["node_features"].numpy()      # (n, 15, 29) -- last step is "now"
        ef = it["edge_features"].numpy()      # (n, n, 5)
        y = it["labels"].numpy()
        n = len(y)
        now = nf[:, -1, :]
        gaze = now[:, :4].argmax(1)
        mouth = now[:, 28]
        for i in range(n):
            peers = [j for j in range(n) if j != i]
            if not peers:
                continue
            rows.append(dict(
                date=date, session=it["session"], y=float(y[i]),
                gaze=int(gaze[i]),
                mutual=float(max(ef[i, j, E_MUTUAL] for j in peers)),
                at_speaker=float(max(ef[i, j, E_SPEAKER] for j in peers)),
                looked_at=float(max(ef[j, i, E_GAZE] for j in peers)),
                peer_mean=float(np.mean([y[j] for j in peers])),
                peer_max=float(np.max([y[j] for j in peers])),
                prox=float(np.mean([ef[i, j, E_PROX] for j in peers])),
                mouth=float(mouth[i]),
            ))

import pandas as pd
df = pd.DataFrame(rows)
main = df[df.date.isin(MAIN)]

w("# Interpersonal regularities (auto-generated)\n")
w(f"person-frames: {len(df)} over four dates, {len(main)} over the three main dates\n")

# 1 --------------------------------------------------------------- gaze target
w("## 1. Engagement by where the participant is looking\n")
w("| gaze target | share of frames | mean engagement | share at highest level |")
w("|---|---|---|---|")
names = {GAZE_ROBOT: "the receptionist", GAZE_PEER: "another participant",
         GAZE_ELSE: "elsewhere", GAZE_NONE: "not determined"}
gaze_stats = {}
for g, nm in names.items():
    s = df[df.gaze == g]
    if not len(s):
        continue
    hi = float((to_class(s.y.values) == 3).mean())
    gaze_stats[nm] = dict(share=len(s) / len(df), mean=float(s.y.mean()), high=hi, n=len(s))
    w(f"| {nm} | {100*len(s)/len(df):.1f}% | {s.y.mean():.1f} | {100*hi:.1f}% |")
w()

# 2 ------------------------------------------------------------- mutual gaze
w("## 2. Engagement under measured interpersonal states\n")
w("| state | share of frames | mean engagement | difference |")
w("|---|---|---|---|")
state_stats = {}
for key, nm in [("mutual", "in mutual gaze with a peer"),
                ("at_speaker", "looking at a peer whose mouth is moving"),
                ("looked_at", "being looked at by a peer")]:
    on = df[df[key] > 0.5]; off = df[df[key] <= 0.5]
    state_stats[nm] = dict(share=len(on) / len(df), on=float(on.y.mean()),
                           off=float(off.y.mean()), diff=float(on.y.mean() - off.y.mean()))
    w(f"| {nm} | {100*len(on)/len(df):.1f}% | {on.y.mean():.1f} | {on.y.mean()-off.y.mean():+.1f} |")
w()

# 3 ------------------------------------------------ peer engagement coupling
w("## 3. Own engagement as a function of the peers' engagement\n")
bins = np.arange(0, 101, 20)
w("| peers' mean engagement | n | own mean engagement | own share at highest level |")
w("|---|---|---|---|")
coupling = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = df[(df.peer_mean >= lo) & (df.peer_mean < hi if hi < 100 else df.peer_mean <= hi)]
    if len(s) < 50:
        continue
    coupling.append(dict(lo=int(lo), hi=int(hi), n=len(s), mean=float(s.y.mean()),
                         high=float((to_class(s.y.values) == 3).mean())))
    w(f"| {lo}-{hi} | {len(s)} | {s.y.mean():.1f} | {100*(to_class(s.y.values)==3).mean():.1f}% |")
w()

# 4 ---------------------------------------------------- level agreement rate
w("## 4. Do co-present participants share an engagement level more often than chance?\n")
agree_obs, agree_exp, tot = 0, 0.0, 0
per_date_agree = {}
for date in DATES:
    data = torch.load(ROOT / "modelTraining" / f"graph_dataset_{date}_continuous.pt", weights_only=False)
    cls_all = np.concatenate([to_class(it["labels"].numpy()) for it in data])
    p = np.bincount(cls_all, minlength=4) / len(cls_all)
    obs = ex = k = 0
    for it in data:
        c = to_class(it["labels"].numpy())
        n = len(c)
        for i in range(n):
            for j in range(i + 1, n):
                obs += int(c[i] == c[j]); k += 1
        ex += (n * (n - 1) / 2) * float((p ** 2).sum())
    per_date_agree[date] = dict(observed=obs / k, expected=ex / k, n_pairs=k)
    agree_obs += obs; agree_exp += ex; tot += k
    w(f"- {date}: pairs sharing a level {100*obs/k:.1f}% observed against {100*ex/k:.1f}% expected "
      f"if independent ({k} pairs)")
overall = dict(observed=agree_obs / tot, expected=agree_exp / tot, n_pairs=tot)
w(f"- **all dates**: {100*agree_obs/tot:.1f}% observed against {100*agree_exp/tot:.1f}% expected, "
  f"{tot} co-present pairs")
w()

# 5 ------------------------------------------------------- synchronous change
w("## 5. Are engagement changes synchronous across co-present participants?\n")
sync = []
for date in DATES:
    for sess in sorted({0}):
        pass
from collections import defaultdict
traces = defaultdict(dict)
for date in DATES:
    data = torch.load(ROOT / "modelTraining" / f"graph_dataset_{date}_continuous.pt", weights_only=False)
    for it in data:
        key = (date, it["session"])
        traces[key].setdefault("t", []).append(it["timestamp_sec"])
        for p, y in zip(it["participants"], it["labels"].numpy()):
            traces[key].setdefault(p, []).append(float(y))

n_sync, n_change = 0, 0
window = 10   # samples at 5 Hz = 2 s
for key, tr in traces.items():
    parts = [p for p in tr if p != "t"]
    series = {p: np.array(tr[p]) for p in parts if len(tr[p]) == len(tr["t"])}
    if len(series) < 2:
        continue
    lev = {p: to_class(v) for p, v in series.items()}
    chg = {p: np.abs(np.diff(v)) > 0 for p, v in lev.items()}
    for p in parts:
        if p not in chg:
            continue
        others = [q for q in chg if q != p]
        for idx in np.flatnonzero(chg[p]):
            n_change += 1
            lo, hi = max(0, idx - window), min(len(chg[p]), idx + window + 1)
            if any(chg[q][lo:hi].any() for q in others):
                n_sync += 1
w(f"- {100*n_sync/n_change:.1f}% of level changes have another participant also changing level "
  f"within 2 s ({n_change} changes)")

# chance rate: shuffle one participant's change times within the session
rng = np.random.default_rng(0)
n_sync_r, n_change_r = 0, 0
for key, tr in traces.items():
    parts = [p for p in tr if p != "t"]
    series = {p: np.array(tr[p]) for p in parts if len(tr[p]) == len(tr["t"])}
    if len(series) < 2:
        continue
    lev = {p: to_class(v) for p, v in series.items()}
    chg = {p: np.abs(np.diff(v)) > 0 for p, v in lev.items()}
    for p in chg:
        shuffled = chg[p].copy(); rng.shuffle(shuffled)
        others = [q for q in chg if q != p]
        for idx in np.flatnonzero(shuffled):
            n_change_r += 1
            lo, hi = max(0, idx - window), min(len(shuffled), idx + window + 1)
            if any(chg[q][lo:hi].any() for q in others):
                n_sync_r += 1
w(f"- against {100*n_sync_r/n_change_r:.1f}% when one participant's change times are shuffled "
  f"within the session (same marginal rate, no timing relationship)")

payload = dict(gaze=gaze_stats, states=state_stats, coupling=coupling,
               agreement=dict(per_date=per_date_agree, overall=overall),
               sync=dict(observed=n_sync / n_change, shuffled=n_sync_r / n_change_r,
                         n_changes=n_change))
(OUT / "interpersonal.json").write_text(json.dumps(payload, indent=2))
(OUT / "interpersonal.md").write_text("\n".join(lines) + "\n")
df.to_pickle(OUT / "interpersonal_frames.pkl")
print("\nwrote interpersonal.json / .md / _frames.pkl")
