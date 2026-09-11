"""Parses a train_continuous.py log file (redirected via `> logfile 2>&1`,
NOT piped through `tail`) into a self-contained HTML dashboard -- since
Artifacts can't live-poll a local file, this is meant to be re-run and
republished (same file path) whenever a progress check is wanted, not a
truly live page.

Usage:
    modelTraining/.venv/bin/python3 modelTraining/render_training_dashboard.py \
        <path-to-log> [-o output.html] [--pid <training-process-pid>]

Pass --pid so the header shows TRUE wall-clock elapsed (via `ps -o etimes=`),
not just the sum of already-completed folds' own elapsed times. Real bug hit
2026-07-29: reporting only the completed-folds sum made it look like almost
no progress happened between two checks (6h36m -> 8h22m, +1h46m) when in
reality ~65 real minutes of that gap was legitimate work already sunk into
the still-running fold -- it just doesn't count until that fold finishes.
Showing both numbers, clearly labeled, avoids re-litigating this.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

import torch


def _process_wall_clock_sec(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None  # process not found/not running -- fine, just omit the stat

PROJECT_ROOT = Path(__file__).parent.parent

VARIANT_RE = re.compile(r"^=== variant: (\S+) ===")
FOLD_RE = re.compile(r"^\s*fold: held-out (\S+) session (\d+) \(train n=(\d+), test n=(\d+)\)")
EPOCH_RE = re.compile(r"^\s*epoch (\d+)/(\d+)\s+loss=([\d.]+)")
RESULT_RE = re.compile(
    r"^\s*->\s*mae=([\d.]+)\s+tol5=([\d.]+)\s+tol10=([\d.]+)\s+spearman=([\d.]+)"
    r"(?:\s+acc=([\d.]+)\s+macro_f1=([\d.]+))?\s+\((\d+)s\)"
)
LOADED_RE = re.compile(r"^\s*loaded (\S+): (\d+) frame-level graphs")
TOTAL_RE = re.compile(r"^dataset\(s\): .* -> (\d+) frame-level graphs total")
CONFIG_RE = re.compile(
    r"^head_type=(\S+) epsilon=([\d.]+) lr=([\d.]+) weight_decay=([\d.]+) dropout=([\d.]+) epochs=(\d+)"
)


def parse_log(text: str) -> dict:
    datasets: dict[str, int] = {}
    total_graphs = 0
    config = {}
    variants: dict[str, list[dict]] = {}
    cur_variant = None
    cur_fold = None

    for line in text.splitlines():
        if m := LOADED_RE.match(line):
            datasets[m.group(1)] = int(m.group(2))
            continue
        if m := TOTAL_RE.match(line):
            total_graphs = int(m.group(1))
            continue
        if m := CONFIG_RE.match(line):
            config = {
                "head_type": m.group(1), "epsilon": float(m.group(2)), "lr": float(m.group(3)),
                "weight_decay": float(m.group(4)), "dropout": float(m.group(5)), "epochs": int(m.group(6)),
            }
            continue
        if m := VARIANT_RE.match(line):
            cur_variant = m.group(1)
            variants.setdefault(cur_variant, [])
            cur_fold = None
            continue
        if m := FOLD_RE.match(line):
            cur_fold = {
                "dataset": m.group(1), "session": int(m.group(2)),
                "train_n": int(m.group(3)), "test_n": int(m.group(4)),
                "epoch_log": [], "result": None,
            }
            variants[cur_variant].append(cur_fold)
            continue
        if m := EPOCH_RE.match(line):
            if cur_fold is not None:
                cur_fold["epoch_log"].append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
            continue
        if m := RESULT_RE.match(line):
            if cur_fold is not None:
                cur_fold["result"] = {
                    "mae": float(m.group(1)), "tol5": float(m.group(2)), "tol10": float(m.group(3)),
                    "spearman": float(m.group(4)),
                    "acc": float(m.group(5)) if m.group(5) else None,
                    "macro_f1": float(m.group(6)) if m.group(6) else None,
                    "elapsed": int(m.group(7)),
                }
            continue

    return {"datasets": datasets, "total_graphs": total_graphs, "config": config, "variants": variants}


def _sparkline(epoch_log: list[tuple[int, int, float]], width: int = 120, height: int = 32) -> str:
    if len(epoch_log) < 2:
        return ""
    losses = [e[2] for e in epoch_log]
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    n = len(losses)
    pts = [
        f"{(i / (n - 1)) * (width - 4) + 2:.1f},{height - 2 - ((v - lo) / span) * (height - 4):.1f}"
        for i, v in enumerate(losses)
    ]
    path = " ".join(pts)
    last_x, last_y = pts[-1].split(",")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="spark">'
        f'<polyline points="{path}" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.5" fill="currentColor"/>'
        f"</svg>"
    )


def _true_expected_folds(dataset_names: list[str]) -> list[tuple[str, int]]:
    """The log only shows a fold once training reaches it, so counting folds
    seen so far badly understates the true total early in a run (e.g. "0/1"
    instead of "0/13"). Load the actual graph_dataset_<name>_continuous.pt
    files -- the same source train_continuous.py itself reads -- to get the
    real fold set up front, matching leave_one_session_out's own
    `sorted({(d["dataset"], d["session"]) for d in data})` exactly."""
    keys: set[tuple[str, int]] = set()
    for name in dataset_names:
        path = PROJECT_ROOT / "modelTraining" / f"graph_dataset_{name}_continuous.pt"
        if not path.exists():
            continue
        graphs = torch.load(path, weights_only=False)
        keys.update((g["dataset"], g["session"]) for g in graphs)
    return sorted(keys)


def render(data: dict, pid: int | None = None) -> str:
    datasets = data["datasets"]
    total_graphs = data["total_graphs"]
    config = data["config"]
    variants = data["variants"]
    wall_clock_sec = _process_wall_clock_sec(pid) if pid else None

    expected_folds = _true_expected_folds(list(datasets.keys()))
    seen_folds = sorted({(f["dataset"], f["session"]) for v in variants.values() for f in v})
    if not expected_folds:  # couldn't load the .pt files (e.g. run somewhere else) -- fall back
        expected_folds = seen_folds
    n_expected = len(expected_folds) or 1

    all_done = 0
    all_total = 0
    total_elapsed = 0
    variant_sections = []
    for vname, folds in variants.items():
        n_done = sum(1 for f in folds if f["result"])
        all_done += n_done
        all_total += n_expected
        total_elapsed += sum(f["result"]["elapsed"] for f in folds if f["result"])

        chips = []
        for ds, sess in expected_folds:
            match = next((f for f in folds if f["dataset"] == ds and f["session"] == sess), None)
            if match and match["result"]:
                status = "done"
            elif match:
                status = "running"
            else:
                status = "pending"
            chips.append(
                f'<div class="chip chip-{status}" title="{html.escape(ds)} s{sess}">'
                f'<span class="chip-label">{html.escape(ds)}<br>s{sess}</span></div>'
            )

        rows = []
        for f in folds:
            spark = _sparkline(f["epoch_log"])
            if f["result"]:
                r = f["result"]
                acc_str = f'{r["acc"]:.3f}' if r["acc"] is not None else "&mdash;"
                f1_str = f'{r["macro_f1"]:.3f}' if r["macro_f1"] is not None else "&mdash;"
                rows.append(
                    f"<tr><td class='mono'>{html.escape(f['dataset'])} s{f['session']}</td>"
                    f"<td class='status-cell'><span class='dot dot-done'></span>done</td>"
                    f"<td class='mono num'>{r['mae']:.2f}</td>"
                    f"<td class='mono num'>{r['tol10']:.3f}</td>"
                    f"<td class='mono num'>{r['spearman']:.3f}</td>"
                    f"<td class='mono num'>{acc_str}</td>"
                    f"<td class='mono num'>{f1_str}</td>"
                    f"<td class='spark-cell'>{spark}</td>"
                    f"<td class='mono num muted'>{r['elapsed']}s</td></tr>"
                )
            else:
                last_epoch = f["epoch_log"][-1] if f["epoch_log"] else None
                epoch_str = f"{last_epoch[0]}/{last_epoch[1]}" if last_epoch else "starting&hellip;"
                loss_str = f"{last_epoch[2]:.4f}" if last_epoch else "&mdash;"
                rows.append(
                    f"<tr class='row-running'><td class='mono'>{html.escape(f['dataset'])} s{f['session']}</td>"
                    f"<td class='status-cell'><span class='dot dot-running'></span>epoch {epoch_str}</td>"
                    f"<td class='mono num muted' colspan='4'>loss={loss_str}</td>"
                    f"<td class='mono num muted'>&mdash;</td>"
                    f"<td class='spark-cell'>{spark}</td>"
                    f"<td class='mono num muted'>&mdash;</td></tr>"
                )

        mean_row = ""
        completed_results = [f["result"] for f in folds if f["result"]]
        if completed_results:
            n = len(completed_results)
            mean_mae = sum(r["mae"] for r in completed_results) / n
            mean_tol10 = sum(r["tol10"] for r in completed_results) / n
            mean_sp = sum(r["spearman"] for r in completed_results) / n
            accs = [r["acc"] for r in completed_results if r["acc"] is not None]
            f1s = [r["macro_f1"] for r in completed_results if r["macro_f1"] is not None]
            mean_acc = f"{sum(accs) / len(accs):.3f}" if accs else "&mdash;"
            mean_f1 = f"{sum(f1s) / len(f1s):.3f}" if f1s else "&mdash;"
            mean_row = (
                f"<tr class='row-mean'><td class='mono'>mean ({n}/{n_expected})</td><td></td>"
                f"<td class='mono num'>{mean_mae:.2f}</td><td class='mono num'>{mean_tol10:.3f}</td>"
                f"<td class='mono num'>{mean_sp:.3f}</td><td class='mono num'>{mean_acc}</td>"
                f"<td class='mono num'>{mean_f1}</td><td></td><td></td></tr>"
            )

        pct = round(100 * n_done / n_expected)
        variant_sections.append(f"""
        <section class="variant-card">
          <div class="variant-head">
            <h2>{html.escape(vname)}</h2>
            <div class="variant-progress">
              <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
              <span class="mono progress-label">{n_done}/{n_expected} folds</span>
            </div>
          </div>
          <div class="chip-row">{"".join(chips)}</div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>fold</th><th>status</th><th>mae</th><th>tol10</th><th>spearman</th>
              <th>acc</th><th>macro_f1</th><th>loss curve</th><th>time</th></tr></thead>
              <tbody>{"".join(rows)}{mean_row}</tbody>
            </table>
          </div>
        </section>""")

    overall_pct = round(100 * all_done / all_total) if all_total else 0
    hrs = total_elapsed // 3600
    mins = (total_elapsed % 3600) // 60
    wall_clock_tile = ""
    if wall_clock_sec is not None:
        wc_hrs, wc_mins = wall_clock_sec // 3600, (wall_clock_sec % 3600) // 60
        wall_clock_tile = (
            f'<div class="stat-tile"><span class="stat-value mono">{wc_hrs}h {wc_mins}m</span>'
            f'<span class="stat-label">wall-clock elapsed</span></div>'
        )
    dataset_chips = "".join(
        f'<div class="stat-tile"><span class="stat-value mono">{n:,}</span>'
        f'<span class="stat-label">{html.escape(ds)}</span></div>'
        for ds, n in datasets.items()
    )

    return f"""<!doctype html>
<title>CAMEO Training Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #f5f3ef; --surface: #ffffff; --border: #dcd7cd; --text: #23201c; --muted: #7a7268;
  --accent: #b6522f; --amber: #c9852b; --teal: #2f7d78;
  --good: #3f8a5c; --running: #2f6fa8; --pending: #b3ab9e; --bad: #b8443f;
  --row-hover: #f0ece3;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #171512; --surface: #211e1a; --border: #34302a; --text: #ece7de; --muted: #9b9184;
    --accent: #e37a52; --amber: #e6a94f; --teal: #52b0a8;
    --good: #5cbf83; --running: #6ba7dd; --pending: #57503f; --bad: #d97a70;
    --row-hover: #2a2622;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #171512; --surface: #211e1a; --border: #34302a; --text: #ece7de; --muted: #9b9184;
  --accent: #e37a52; --amber: #e6a94f; --teal: #52b0a8;
  --good: #5cbf83; --running: #6ba7dd; --pending: #57503f; --bad: #d97a70;
  --row-hover: #2a2622;
}}
:root[data-theme="light"] {{
  --bg: #f5f3ef; --surface: #ffffff; --border: #dcd7cd; --text: #23201c; --muted: #7a7268;
  --accent: #b6522f; --amber: #c9852b; --teal: #2f7d78;
  --good: #3f8a5c; --running: #2f6fa8; --pending: #b3ab9e; --bad: #b8443f;
  --row-hover: #f0ece3;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
  padding: 32px 24px 64px;
}}
.mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace; font-variant-numeric: tabular-nums; }}
.num {{ text-align: right; }}
.muted {{ color: var(--muted); }}
header {{ max-width: 1080px; margin: 0 auto 28px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }}
.subtitle {{ color: var(--muted); font-size: 0.9rem; margin: 0 0 20px; }}
.overall-row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: stretch; }}
.stat-tile {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; display: flex; flex-direction: column; gap: 2px; min-width: 92px;
}}
.stat-value {{ font-size: 1.15rem; font-weight: 600; }}
.stat-label {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
.overall-progress {{ flex: 1; min-width: 220px; display: flex; flex-direction: column; justify-content: center; gap: 6px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; }}
main {{ max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
.variant-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 20px 8px; }}
.variant-head {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }}
.variant-head h2 {{ margin: 0; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.variant-progress {{ display: flex; align-items: center; gap: 10px; }}
.progress-track {{ width: 160px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }}
.progress-fill {{ height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s; }}
.progress-label {{ font-size: 0.8rem; color: var(--muted); }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }}
.chip {{
  width: 46px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); font-size: 0.6rem; line-height: 1.15; text-align: center;
}}
.chip-label {{ color: inherit; }}
.chip-done {{ background: color-mix(in srgb, var(--good) 18%, var(--surface)); border-color: var(--good); color: var(--good); }}
.chip-running {{ background: color-mix(in srgb, var(--running) 18%, var(--surface)); border-color: var(--running); color: var(--running);
  animation: pulse 1.6s ease-in-out infinite; }}
.chip-pending {{ background: var(--surface); color: var(--pending); border-style: dashed; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
thead th {{ text-align: left; color: var(--muted); font-weight: 500; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.03em; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
thead th.num, thead th:nth-child(3), thead th:nth-child(4), thead th:nth-child(5), thead th:nth-child(6), thead th:nth-child(7), thead th:nth-child(9) {{ text-align: right; }}
tbody td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
tbody tr:hover {{ background: var(--row-hover); }}
.row-running {{ color: var(--muted); }}
.row-mean td {{ border-top: 2px solid var(--border); border-bottom: none; font-weight: 600; color: var(--text); }}
.status-cell {{ display: flex; align-items: center; gap: 6px; }}
.dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }}
.dot-done {{ background: var(--good); }}
.dot-running {{ background: var(--running); animation: pulse 1.6s ease-in-out infinite; }}
.spark-cell {{ color: var(--teal); }}
.spark {{ display: block; }}
footer {{ max-width: 1080px; margin: 24px auto 0; color: var(--muted); font-size: 0.75rem; }}
</style>
<header>
  <h1>CAMEO Training Monitor</h1>
  <p class="subtitle">Pooled continuous-label LOSO training &mdash; classification head, {config.get('epochs', '?')} epochs,
    wd={config.get('weight_decay', '?')}, dropout={config.get('dropout', '?')}</p>
  <div class="overall-row">
    {dataset_chips}
    <div class="stat-tile"><span class="stat-value mono">{total_graphs:,}</span><span class="stat-label">total graphs</span></div>
    {wall_clock_tile}
    <div class="stat-tile"><span class="stat-value mono">{hrs}h {mins}m</span><span class="stat-label">completed-fold time</span></div>
    <div class="overall-progress">
      <div style="display:flex;justify-content:space-between;">
        <span class="stat-label">overall</span>
        <span class="mono" style="font-size:0.8rem;">{all_done}/{all_total} folds</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{overall_pct}%"></div></div>
    </div>
  </div>
</header>
<main>
  {"".join(variant_sections)}
</main>
<footer>Regenerated from the training log on request &mdash; not live. Re-run render_training_dashboard.py and redeploy for an update.</footer>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("-o", "--output", default="training_dashboard.html")
    parser.add_argument("--pid", type=int, help="training process pid, for a true wall-clock-elapsed stat")
    args = parser.parse_args()

    text = Path(args.logfile).read_text()
    data = parse_log(text)
    html_out = render(data, pid=args.pid)
    Path(args.output).write_text(html_out)
    print(f"wrote {args.output}")
