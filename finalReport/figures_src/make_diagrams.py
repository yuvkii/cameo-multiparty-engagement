"""Schematic diagrams: processing pipeline, model architecture, evaluation
protocol. Drawn rather than data-derived, but kept in the same style so the
report reads as one document.

Layout rules used throughout, so that nothing overlaps at print size:
  * every box is placed on an explicit column/row grid, never by eye;
  * box text is wrapped to a character budget derived from the box width in
    points, so a label can never run past its own border;
  * connectors are orthogonal elbows drawn between box *edges*, so an arrow
    never crosses a box or ends in empty space.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import FULL, BLUE, ORANGE, AQUA, VIOLET, RED, INK, INK2, MUTED, GRID, save

FILL = {"data": "#eef3fb", "proc": "#fdf0ea", "model": "#e9f7f1", "out": "#efeefa"}
EDGE = {"data": BLUE, "proc": ORANGE, "model": AQUA, "out": VIOLET}

TITLE_FS = 7.2
BODY_FS = 6.3


class Canvas:
    """A 0-1 x 0-1 drawing surface that knows its own physical size, so text
    can be wrapped to the true width of each box."""

    def __init__(self, height_in: float, width_in: float = FULL):
        self.fig = plt.figure(figsize=(width_in, height_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.w_in, self.h_in = width_in, height_in

    # -- geometry -----------------------------------------------------------
    def _renderer(self):
        if getattr(self, "_rend", None) is None:
            self.fig.canvas.draw()
            self._rend = self.fig.canvas.get_renderer()
        return self._rend

    def text_width(self, s: str, fs: float, bold: bool) -> float:
        """Width of `s` in axes fractions, measured rather than estimated."""
        t = self.ax.text(0, 0, s, fontsize=fs, weight="bold" if bold else "normal")
        bb = t.get_window_extent(renderer=self._renderer())
        t.remove()
        return bb.width / self.ax.get_window_extent(renderer=self._renderer()).width

    def wrap(self, s: str, avail: float, fs: float, bold: bool) -> list[str]:
        """Greedy wrap of `s` into lines no wider than `avail` axes fractions."""
        out, cur = [], ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if cur and self.text_width(trial, fs, bold) > avail:
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out or [""]

    # -- primitives ---------------------------------------------------------
    def box(self, x, y, w, h, title=None, lines=(), kind="proc", ls="solid",
            title_fs=TITLE_FS, body_fs=BODY_FS):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=1.1, linestyle=ls,
            zorder=3))
        cx = x + w / 2
        avail = w - 0.018
        for _ in range(30):
            head = self.wrap(title, avail, title_fs, True) if title else []
            body = []
            for ln in lines:
                body.extend(self.wrap(ln, avail, body_fs, False) if ln else [""])
            lh_t = title_fs * 1.32 / (72.0 * self.h_in)
            lh_b = body_fs * 1.32 / (72.0 * self.h_in)
            total = len(head) * lh_t + (0.45 * lh_b if head and body else 0) + len(body) * lh_b
            if total <= h - 0.022 or body_fs < 4.4:
                break
            title_fs, body_fs = title_fs * 0.94, body_fs * 0.94
        top = y + h / 2 + total / 2
        cur = top - lh_t / 2
        for ln in head:
            self.ax.text(cx, cur, ln, ha="center", va="center", fontsize=title_fs,
                         color=INK, weight="bold", zorder=4)
            cur -= lh_t
        if head and body:
            cur -= 0.45 * lh_b - (lh_t - lh_b) / 2
        for ln in body:
            self.ax.text(cx, cur, ln, ha="center", va="center", fontsize=body_fs,
                         color=INK2, zorder=4)
            cur -= lh_b
        return dict(x=x, y=y, w=w, h=h, l=x, r=x + w, b=y, t=y + h,
                    cx=cx, cy=y + h / 2)

    def elbow(self, p0, p1, via=None, color=MUTED, lw=1.15, ls="solid", label=None,
              label_at=None, label_dy=0.018, label_color=None):
        """Orthogonal connector: p0 -> (optional bend x or y) -> p1."""
        pts = [p0]
        if via is not None:
            kind, v = via
            if kind == "x":
                pts += [(v, p0[1]), (v, p1[1])]
            else:
                pts += [(p0[0], v), (p1[0], v)]
        pts.append(p1)
        for a, b in zip(pts[:-1], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                         solid_capstyle="round", zorder=2)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                          mutation_scale=8, color=color, lw=lw,
                                          linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))
        if label:
            lx, ly = label_at if label_at else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            self.ax.text(lx, ly + label_dy, label, ha="center", va="bottom",
                         fontsize=6.2, color=label_color or INK2, zorder=4)

    def note(self, x, y, text, fs=6.4, color=MUTED, ha="center", va="center", wrap=None):
        if wrap:
            text = "\n".join(textwrap.wrap(text, wrap))
        self.ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=4)

    def save(self, name):
        save(self.fig, name)


# ==================================================================== pipeline
def fig_pipeline():
    c = Canvas(3.05)
    COL = {"src": (0.000, 0.172), "ext": (0.214, 0.190), "assoc": (0.444, 0.156),
           "ident": (0.634, 0.166), "out": (0.834, 0.166)}

    cam = c.box(*COL["src"][0:1], 0.560, COL["src"][1], 0.410,
                title="wide-angle camera",
                lines=["one fixed view", "20 sessions", "three participants",
                       "and a confederate"], kind="data")
    ann = c.box(COL["src"][0], 0.085, COL["src"][1], 0.290,
                title="continuous annotation",
                lines=["one pass per person", "slider sampled at", "every frame"],
                kind="data")

    ex_y = [(0.810, 0.160), (0.620, 0.160), (0.430, 0.160)]
    ex = [
        c.box(COL["ext"][0], ex_y[0][0], COL["ext"][1], ex_y[0][1], title="Gaze-LLE",
              lines=["gaze target per person"], kind="proc"),
        c.box(COL["ext"][0], ex_y[1][0], COL["ext"][1], ex_y[1][1], title="OpenFace 3.0",
              lines=["faces, action units,", "emotion, head gaze"], kind="proc"),
        c.box(COL["ext"][0], ex_y[2][0], COL["ext"][1], ex_y[2][1], title="mouth motion",
              lines=["visual speaking proxy"], kind="proc"),
    ]
    bus = COL["src"][0] + COL["src"][1] + 0.020
    for b in ex:
        c.elbow((cam["r"], cam["cy"]), (b["l"], b["cy"]), via=("x", bus))

    assoc = c.box(COL["assoc"][0], 0.430, COL["assoc"][1], 0.540,
                  title="same-frame association",
                  lines=["every signal matched", "by bounding-box", "overlap to one",
                         "tracked person"], kind="proc")
    bus2 = COL["ext"][0] + COL["ext"][1] + 0.020
    for b in ex:
        c.elbow((b["r"], b["cy"]), (assoc["l"], assoc["cy"]), via=("x", bus2))

    ident = c.box(COL["ident"][0], 0.430, COL["ident"][1], 0.540,
                  title="identity reconciliation",
                  lines=["position rank, or a", "Hungarian tracker", "with a velocity term",
                         "where people move"], kind="proc")
    c.elbow((assoc["r"], assoc["cy"]), (ident["l"], ident["cy"]))
    c.elbow((ann["r"], ann["cy"]), (ident["cx"], ident["b"]), via=("y", ann["cy"]),
            color=BLUE, lw=1.3, label="engagement trace per participant",
            label_at=((ann["r"] + ident["cx"]) / 2, ann["cy"]), label_color=BLUE)

    out = c.box(COL["out"][0], 0.430, COL["out"][1], 0.540,
                title="one graph per frame",
                lines=["5 Hz, 3 nodes", "15-step window", "29 node features",
                       "5 edge channels", "", "29,747 graphs", "88,562 person-frames"],
                kind="out")
    c.elbow((ident["r"], ident["cy"]), (out["l"], out["cy"]))

    c.note(0.5, 0.035,
           "excluded before training: setup and teardown periods; sessions with no recoverable identity "
           "assignment; frames without a full three-second history window", fs=6.4, wrap=96)
    c.save("fig_pipeline")


# ================================================================ architecture
def fig_architecture():
    c = Canvas(3.45)
    A = (0.004, 0.098)
    B = (0.138, 0.148)
    C = (0.320, 0.178)
    D = (0.536, 0.134)
    E = (0.706, 0.116)
    F = (0.852, 0.144)

    # ---- per-participant windowed input, drawn as a stack of frames
    node_y = [0.700, 0.430, 0.160]
    node_c = [BLUE, ORANGE, AQUA]
    for (yy, col, part) in zip(node_y, node_c, "ABC"):
        for s in range(3):
            c.ax.add_patch(Rectangle((A[0] + 0.006 * (2 - s), yy + 0.010 * s), 0.080, 0.135,
                                     facecolor="white" if s else FILL["data"],
                                     edgecolor=col, lw=0.9, zorder=3 + s))
        c.ax.text(A[0] + 0.045, yy + 0.075, "15 x 29", ha="center", va="center",
                  fontsize=6.3, color=INK2, zorder=7)
        c.ax.text(A[0] + 0.049, yy + 0.180, f"participant {part}", ha="center",
                  fontsize=6.8, color=col, zorder=7)
    c.note(A[0] + 0.049, 0.090, "three seconds of each person's own history, sampled at 5 Hz",
           fs=6.3, va="top", wrap=24)

    enc = c.box(B[0], 0.250, B[1], 0.660, title="temporal encoder",
                lines=["shared per-step MLP", "29-64-32", "LayerNorm, ReLU,", "dropout",
                       "", "GRU over 15 steps", "final state h_i"], kind="model")
    bus = A[0] + A[1] + 0.016
    for yy in node_y:
        c.elbow((A[0] + 0.090, yy + 0.078), (enc["l"], enc["cy"]), via=("x", bus))

    gat = c.box(C[0], 0.250, C[1], 0.660, title="relational message passing",
                lines=["2 layers, 4 heads", "", "measured edge features", "are added to the",
                       "attention logits, so", "the graph decides", "who attends to whom"],
                kind="model")
    c.elbow((enc["r"], enc["cy"]), (gat["l"], gat["cy"]))

    edg = c.box(C[0], 0.025, C[1], 0.185, title="edge features e_ij",
                lines=["gaze, proximity,", "mutual gaze,", "attending to a speaker,",
                       "lagged peer engagement"], kind="data", title_fs=6.8, body_fs=6.0)
    c.elbow((edg["cx"], edg["t"]), (gat["cx"], gat["b"]), color=BLUE, lw=1.3)

    grp = c.box(D[0], 0.600, D[1], 0.310, title="attention pooling",
                lines=["group vector g"], kind="model")
    per = c.box(D[0], 0.250, D[1], 0.290, title="per-node vector",
                lines=["h'_i"], kind="model")
    bus2 = C[0] + C[1] + 0.020
    c.elbow((gat["r"], gat["cy"]), (grp["l"], grp["cy"]), via=("x", bus2))
    c.elbow((gat["r"], gat["cy"]), (per["l"], per["cy"]), via=("x", bus2))

    trunk = c.box(E[0], 0.400, E[1], 0.360, title="shared trunk",
                  lines=["[ h'_i ; g ]", "64-dim"], kind="model")
    bus3 = D[0] + D[1] + 0.020
    c.elbow((grp["r"], grp["cy"]), (trunk["l"], trunk["cy"]), via=("x", bus3))
    c.elbow((per["r"], per["cy"]), (trunk["l"], trunk["cy"]), via=("x", bus3))

    head_e = c.box(F[0], 0.560, F[1], 0.290, title="engagement head",
                   lines=["4 levels, read out", "as an expected value"], kind="out")
    head_b = c.box(F[0], 0.130, F[1], 0.270, title="behaviour head",
                   lines=["6 actions", "(not trained)"], kind="out", ls=(0, (3, 2)))
    bus4 = E[0] + E[1] + 0.014
    c.elbow((trunk["r"], trunk["cy"]), (head_e["l"], head_e["cy"]), via=("x", bus4))
    c.elbow((trunk["r"], trunk["cy"]), (head_b["l"], head_b["cy"]), via=("x", bus4),
            ls=(0, (3, 2)))
    c.note(F[0] + F[1] / 2, 0.085, "no behaviour labels exist for this corpus",
           fs=6.2, va="top", wrap=22)
    c.save("fig_architecture")


# ==================================================================== protocol
def fig_loso():
    c = Canvas(2.25)
    sessions = [("03_20", [1, 3, 4]), ("03_26", [1, 2, 3, 4, 5, 6]),
                ("05_15", [1, 2, 4, 5, 6])]
    flat = [(d, s) for d, ss in sessions for s in ss]
    n = len(flat)
    x0, wdt, gap, hgt = 0.145, 0.0525, 0.0060, 0.115

    def xs(i):
        return x0 + i * (wdt + gap)

    rows = [(0, 0.700), (1, 0.480), (n - 1, 0.155)]
    for held, y in rows:
        for i in range(n):
            hot = i == held
            c.ax.add_patch(Rectangle((xs(i), y), wdt, hgt,
                                     facecolor=FILL["proc"] if hot else FILL["data"],
                                     edgecolor=ORANGE if hot else BLUE, lw=1.0, zorder=3))
        c.ax.text(x0 - 0.014, y + hgt / 2, f"fold {held + 1}", ha="right", va="center",
                  fontsize=7.4, color=INK)

    # continuation marker, placed in the empty band between rows 2 and 3
    for k, dy in enumerate((0.0, 0.022, 0.044)):
        c.ax.plot([x0 - 0.052], [0.345 + dy], marker="o", ms=1.6, color=MUTED)

    for i, (d, s) in enumerate(flat):
        c.ax.text(xs(i) + wdt / 2, 0.845, f"s{s}", ha="center", fontsize=6.4, color=INK2)
    start = 0
    for d, ss in sessions:
        lo, hi = xs(start), xs(start + len(ss) - 1) + wdt
        c.ax.plot([lo, hi], [0.925, 0.925], color=MUTED, lw=0.8)
        c.ax.text((lo + hi) / 2, 0.945, d, ha="center", fontsize=7.2, color=INK)
        start += len(ss)

    c.ax.add_patch(Rectangle((0.145, 0.020), 0.036, 0.055, facecolor=FILL["data"],
                             edgecolor=BLUE, lw=1.0))
    c.ax.text(0.192, 0.048, "training sessions", va="center", fontsize=7, color=INK2)
    c.ax.add_patch(Rectangle((0.430, 0.020), 0.036, 0.055, facecolor=FILL["proc"],
                             edgecolor=ORANGE, lw=1.0))
    c.ax.text(0.477, 0.048, "held-out session, never seen in training or standardisation",
              va="center", fontsize=7, color=INK2)
    c.save("fig_loso")


ALL = {k[4:]: v for k, v in list(globals().items()) if k.startswith("fig_")}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(ALL)):
        print(name)
        ALL[name]()
