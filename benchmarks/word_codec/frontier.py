"""The rate audit's frontier (the overnight brief's core plot), rendered on demand from the evidence store.

Every lane's points are artifacts of kind `codec-frontier-points/v0`: a JSON list of points in the schema below, put with

    research data put --kind codec-frontier-points/v0 --file points.json --preserve \
      --meta '{"schema": "verity/codec-frontier-point/v0", "lane": LANE, "run": RUN_ID}'

and this script selects them all (`research data select --kind codec-frontier-points/v0`), or reads `--art` ids and
`--file` paths, and draws:
- x, the encode time of one pass's data in native passes, on a log scale, with μ_rebuild = (d − τ)/x on a secondary
  axis (at d = 1 and `--tau`, default the median τ of the points); μ_rebuild's numerator is d − τ, d the step's time in
  passes of E (the red team's correction: the adversary's spare compute is the whole step), so x* = (d − τ)/μ_budget is
  a curve: a vertical line at x* for d = 1 at each tag's τ, and the boundary y = τ + μ_budget·x for d = y (μ_budget =
  0.5% leaves γ = 1% after μ_chip and μ_sample, the brief's budget);
- y, fused decode-step time over native at Llama-3.1-8B's linears, layer-weighted, as a stacked bar per point
  (base, decode, tag, commit) at batch 1 and batch 32 side by side, with a horizontal line at 2×;
- measured points filled, modelled points hollow, reference points (P2) as stars;
and prints the frontier, the lowest y_total at x ≥ x* at each batch (d = the point's y_total unless `--d` fixes it).
Never commit the image or the table.

Schema `verity/codec-frontier-point/v0`, one object per (configuration, batch):

~~~text
schema      "verity/codec-frontier-point/v0"
lane        the lane that produced it (ra-codec-wide, ra-word-codec, ra-codec-tensor, ...)
b, k        word width in bits (61, 127, 255, 521, 16448 for P2) and rounds
arithmetic  e.g. "cuda-int32/mersenne/sqr0", "tensor-core/...", free text naming the decode's arithmetic
batch       the decode step's batch (1 … 32)
y_total     (fused decode + tag + commit time) / native, layer-weighted
y_decode    decode's increment over the fused kernel without it, / native
y_tag       the tag's increment, / native
y_commit    the commit's time, / native
y_base      optional: the fused kernel's time on payloads (no decode, no tag) / native; default y_total − the three
x           encode time of one pass's data in native passes (the adversary's best encode)
delta       decode's share of compute per native pass (Δ)
tau         the tag's share (τ)
measured    true for a measured point, false for a modelled one
reference   optional: true for a reference point (P2)
label       optional free text
run, art    the run id and evidence artifact the numbers come from (art may be null when run names it)
~~~

    uv run python benchmarks/word_codec/frontier.py --out /tmp/frontier.png [--art art:… | --file points.json]...
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

SCHEMA = "verity/codec-frontier-point/v0"
KIND = "codec-frontier-points/v0"
REQUIRED = ("b", "k", "arithmetic", "batch", "y_total", "y_decode", "y_tag", "y_commit", "x", "tau")


def research(*args: str) -> str:
    return subprocess.run(["uv", "run", "research", *args], check=True, capture_output=True, text=True).stdout


def store_arts() -> list[str]:
    rows = json.loads(research("data", "select", "--kind", KIND, "--json") or "[]")
    return [r if isinstance(r, str) else r["id"] for r in rows]


def load_art(art: str) -> list[dict]:
    path = Path(research("data", "fetch", art).strip().splitlines()[-1])
    pts = json.loads(path.read_text())
    for p in pts:
        p.setdefault("art", art)
    return pts


def check(points: list[dict]) -> list[dict]:
    out = []
    for p in points:
        if p.get("schema") != SCHEMA:
            raise ValueError(f"a point not in {SCHEMA}: {p}")
        missing = [f for f in REQUIRED if f not in p]
        if missing:
            raise ValueError(f"a point lacks {missing}: {p}")
        p.setdefault("y_base", p["y_total"] - p["y_decode"] - p["y_tag"] - p["y_commit"])
        p.setdefault("measured", True)
        out.append(p)
    return out


def x_star(tau: float, budget: float, d: float = 1.0) -> float:
    return (d - tau) / budget


def step(p: dict, d: float | None) -> float:
    """d, the step's time in passes of E: `--d`, else the point's own y_total (at least one pass)."""
    return max(1.0, p["y_total"]) if d is None else d


def frontier(points: list[dict], budget: float, d: float | None = None) -> dict[int, dict | None]:
    best = {}
    for p in points:
        if p.get("reference") or p["x"] < x_star(p["tau"], budget, step(p, d)):
            continue
        b = best.get(p["batch"])
        if b is None or p["y_total"] < b["y_total"]:
            best[p["batch"]] = p
    return best


def table(points: list[dict], budget: float, batches=(1, 32), d: float | None = None) -> str:
    lines = ["| batch | b | k | arithmetic | x | x* | μ_rebuild | y_total | decode | tag | commit | measured | source |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    best = frontier(points, budget, d)
    for mb in batches:
        p = best.get(mb)
        if p is None:
            lines.append(f"| {mb} | none at x ≥ x* | | | | | | | | | | | |")
            continue
        dd = step(p, d)
        lines.append(f"| {mb} | {p['b']} | {p['k']} | {p['arithmetic']} | {p['x']:.0f} | {x_star(p['tau'], budget, dd):.0f} "
                     f"| {(dd - p['tau']) / p['x'] * 100:.2f}% | {p['y_total']:.2f} | {p['y_decode']:.2f} "
                     f"| {p['y_tag']:.2f} | {p['y_commit']:.2f} | {'yes' if p['measured'] else 'model'} "
                     f"| {p.get('art') or p.get('run') or p.get('lane')} |")
    return "\n".join(lines)


def draw(points: list[dict], out: Path, budget: float, tau_ref: float | None, batches=(1, 32)) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = [p for p in points if p["batch"] in batches]
    if not pts:
        raise SystemExit("no points at the plotted batches")
    tau_ref = statistics.median(p["tau"] for p in pts) if tau_ref is None else tau_ref
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"base": "#9e9e9e", "decode": "#1f77b4", "tag": "#ff7f0e", "commit": "#2ca02c"}
    offset = {mb: 1.0 + 0.09 * (i - (len(batches) - 1) / 2) for i, mb in enumerate(batches)}
    for p in pts:
        xc = p["x"] * offset[p["batch"]]
        width = xc * 0.07
        bottom = 0.0
        for part, key in (("base", "y_base"), ("decode", "y_decode"), ("tag", "y_tag"), ("commit", "y_commit")):
            h = max(p[key], 0.0)
            kw = {"color": colors[part]} if p["measured"] else {"facecolor": "none", "edgecolor": colors[part],
                                                                  "linewidth": 1.2}
            ax.bar(xc, h, width=width, bottom=bottom, align="center", **kw)
            bottom += h
        if p.get("reference"):
            ax.scatter([xc], [p["y_total"]], marker="*", s=160, color="k" if p["measured"] else "none", edgecolors="k",
                       zorder=5)
        ax.annotate(f"b={p['b']} k={p['k']}\nM={p['batch']}", (xc, bottom), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6)
    ax.set_xscale("log")
    ax.axhline(2.0, color="r", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[0], 2.02, " 2× native", color="r", fontsize=8, va="bottom")
    for tau in sorted({round(p["tau"], 2) for p in pts}):
        xs = x_star(tau, budget)
        if xs > 0:
            ax.axvline(xs, color="k", linestyle=":", linewidth=1)
            ax.text(xs, ax.get_ylim()[1] * 0.97, f" x* (d=1, τ={tau:.2f})", fontsize=7, va="top", rotation=90)
    lo, hi = ax.get_xlim()
    xx = np.geomspace(lo, hi, 200)
    ax.plot(xx, tau_ref + budget * xx, color="purple", linewidth=1)
    ax.text(hi, tau_ref + budget * hi, f"y = τ + {budget:g}·x (x* at d = y, τ={tau_ref:.2f}) ", color="purple",
            fontsize=7, ha="right", va="bottom")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(3.0, max(p["y_total"] for p in pts) * 1.1))
    ax.set_xlabel("x: encode time of one pass's data, in native passes (log)")
    ax.set_ylabel("y: fused decode step / native, Llama-3.1-8B linears, layer-weighted")

    def mu(x):
        return (1 - tau_ref) * 100 / np.maximum(np.asarray(x, dtype=float), 1e-9)

    sec = ax.secondary_xaxis("top", functions=(mu, mu))
    sec.set_xlabel(f"μ_rebuild = (d − τ)/x, % (d = 1, τ = {tau_ref:.2f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k")]
    ax.legend(handles, [*colors, "modelled (hollow)"], loc="upper left", fontsize=8)
    ax.set_title(f"Rate-audit frontier: batches {', '.join(map(str, batches))} (left to right at each x); "
                 f"μ budget {budget * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--art", action="append", default=[], help="a codec-frontier-points artifact (default: all)")
    ap.add_argument("--file", action="append", default=[], help="a points.json")
    ap.add_argument("--out", help="the image to write (not into the repository)")
    ap.add_argument("--budget", type=float, default=0.005, help="μ_rebuild's share of γ (default 0.5%%)")
    ap.add_argument("--tau", type=float, help="τ for the secondary axis (default: the points' median)")
    ap.add_argument("--d", type=float, help="the step's time in passes of E for x* (default: each point's y_total)")
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--p2", help="X,Y,TAU: P2 as a modelled reference point")
    a = ap.parse_args()
    points = []
    arts = a.art or ([] if a.file else store_arts())
    for art in arts:
        points += load_art(art)
    for f in a.file:
        points += json.loads(Path(f).read_text())
    batches = tuple(int(b) for b in a.batches.split(","))
    if a.p2:
        x, y, tau = (float(v) for v in a.p2.split(","))
        for mb in batches:
            points.append({"schema": SCHEMA, "lane": "reference", "b": 16448, "k": 1, "arithmetic": "P2", "batch": mb,
                           "y_total": y, "y_decode": 0.0, "y_tag": 0.0, "y_commit": 0.0, "y_base": y, "x": x,
                           "tau": tau, "measured": False, "reference": True, "label": "P2"})
    points = check(points)
    print(table(points, a.budget, batches, a.d))
    if a.out:
        draw(points, Path(a.out), a.budget, a.tau, batches)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
