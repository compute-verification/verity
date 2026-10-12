"""The rate audit's frontier (the overnight brief's core plot), rendered on demand from the evidence store.

Every lane's points are artifacts of kind `codec-frontier-points/v0`: a JSON list of points in the schema below, put with

    research data put --kind codec-frontier-points/v0 --file points.json --preserve \
      --meta '{"schema": "verity/codec-frontier-point/v0", "lane": LANE, "run": RUN_ID}'

and this script selects them all (`research data select --kind codec-frontier-points/v0`), or reads `--art` ids and
`--file` paths, adds ra-codec-tensor's runs as points (`--tensor`, its `codec-tensor.json`: `tensor_points`), and
draws:
- x, the encode time of one pass's data in native passes, on a log scale, with μ_rebuild = (d − τ)/x on a secondary
  axis (at d = 1 and `--tau`, default the median τ of the points); μ_rebuild's numerator is d − τ, d the step's time in
  passes of E (the red team's F1: the adversary's spare compute is the whole step), so x* = (d − τ)/μ_budget is a
  curve: a vertical line at x* for d = 1 at each tag's τ, and the boundary y = τ − idle + μ_budget·x for
  d = y + idle (μ_budget = 0.5% leaves γ = 1% after μ_chip and μ_sample, the brief's budget). d is the point's y_total
  plus `--idle`, the step's time outside the linears in passes of E (0.37 at 8B, 0.10 at 70B, the red team's), and x is
  the point's over `--boost`, the adversary's clock over the locked clock the points were timed at (1.16), unless the
  point's `x_boosted` says its x grants that already. x is the adversary's best encode at the point's width: the lowest
  x per round any point of that width reports, whichever lane's library it came from, or any `--rates` record gives
  (`verity/codec-frontier-rates/v0`, ra-word-codec's measured roots at the max clock; `--floor` scales them to the pipes'
  floor) (`--own-x`: the point's own); a point measured end to end (its `d`) has x* at that d;
  `--tau-cap` bounds τ by the cheapest tag the adversary knows;
- y, fused decode-step time over native at `--model`'s linears (default Llama-3.1-70B; 8B's points are the reference),
  layer-weighted, as a stacked bar per point (base, decode, tag, commit) at batch 1 and batch 32 side by side, with a
  horizontal line at 2×; `--idle` defaults to the model's;
- measured points filled, modelled points hollow, reference points (P2) as stars;
- one bar per configuration, the lowest y_total of each (lane, b, k, batch, measured) over runs and kernel variants
  (`--all` draws every point);
and prints the frontier, the lowest y_total at x ≥ x* at each batch (d = the point's y_total + idle unless `--d`
fixes it). Never commit the image or the table.

Schema `verity/codec-frontier-point/v0`, one object per (configuration, batch):

~~~text
schema      "verity/codec-frontier-point/v0"
lane        the lane that produced it (ra-codec-wide, ra-word-codec, ra-codec-tensor, ...)
model       optional: whose linears y is over, "llama-3.1-70b" or "llama-3.1-8b" (the default)
b, k        word width in bits (61, 127, 256, 512, 521, 16448 for P2) and rounds
ks          optional: each linear's k, when they differ; k and x are then the mean's, and `check` prices the point at
            the smallest (the adversary rebuilds those words: red team F21), keeping the mean as k_mean
arithmetic  e.g. "cuda-int32/mersenne/sqr0", "tensor-core/...", free text naming the decode's arithmetic
batch       the decode step's batch (1 … 32)
y_total     (fused decode + tag + commit time) / native, layer-weighted
y_decode    decode's increment over the fused kernel without it, / native
y_tag       the tag's increment, / native
y_commit    the commit's time, / native
y_base      optional: the fused kernel's time on payloads (no decode, no tag) / native; default y_total − the three
x           encode time of one pass's data in native passes (the adversary's best encode)
x_boosted   optional: true when x already grants the adversary the clock boost (default false: x at the locked clock)
delta       decode's share of compute per native pass (Δ)
tau         the tag's share (τ)
measured    true for a measured point, false for a modelled one
reference   optional: true for a reference point (P2)
ingress     optional: the share of E's bytes the step reads back from HBM besides E (tags, sketches or leaves it wrote
            earlier: red team F19), charged to μ's budget, so x* = (d − τ)/(μ_budget − ingress); a ra-codec-wide
            point without it has it from its arithmetic (`ingress_of`), any other lane's is 0
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
IDLE_8B = 0.37  # the step outside Llama-3.1-8B's linears, in passes of E (red team, F1)
IDLE = {"llama-3.1-8b": IDLE_8B, "llama-3.1-70b": 0.10}  # a point without `model` is 8B's
BOOST = 1.16    # the RTX PRO 6000's boost clock over the locked 2,092 MHz (red team)
RATES_SCHEMA = "verity/codec-frontier-rates/v0"
# SM clocks a squaring at ra-codec-theory's pipe floor (`benchmarks/word_codec/model/frontier.py`'s `pipe_clocks`, as
# ra-word-codec's rates table quotes them), for `--floor`
PIPE_CLOCKS = {512: 3.99, 768: 7.77, 1024: 12.07, 1280: 17.23, 1536: 23.86}


def model_of(p: dict) -> str:
    return p.get("model", "llama-3.1-8b")


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


TENSOR_MODELS = {"70b": "llama-3.1-70b", "8b": "llama-3.1-8b"}


def load_tensor(src: str) -> tuple[dict, str | None]:
    """A `codec-tensor.json` (ra-codec-tensor's `benchmarks/codec_tensor/bench.py`): a path, or the art of its run's
    files."""
    if not src.startswith("art:"):
        return json.loads(Path(src).read_text()), None
    root = Path(research("data", "fetch", src, "--path", "out/codec-tensor.json").strip().splitlines()[-1])
    return json.loads((root / "out" / "codec-tensor.json" if root.is_dir() else root).read_text()), src


def tensor_points(docs: list[tuple[dict, str | None]]) -> list[dict]:
    """ra-codec-tensor's runs as points: y from each run's `layer` (per batch, each linear at its best arm over native:
    `plain` the base, `fused_k<k>` − plain the decode; untagged, so tag and commit 0 and τ 0, which the sketch twins
    replace), x and Δ from whichever document's `analysis` names the field: x the adversary's best root at the squaring
    floor over CUDA cores, tensor cores and both (at the locked clock), Δ the CUDA decode's. A field no `analysis`
    names is left out."""
    rates = {}
    for doc, _ in docs:
        for name, e in doc.get("libs", {}).items():
            if e.get("analysis"):
                rates[name] = e["analysis"]["k"]
    out = []
    for doc, art in docs:
        model = TENSOR_MODELS.get((doc.get("model") or {}).get("name"), (doc.get("model") or {}).get("name"))
        for name, e in doc.get("libs", {}).items():
            if not e.get("layer"):
                continue
            if name not in rates:
                print(f"tensor: {name} has no analysis (x, Δ) in the documents given; left out", file=sys.stderr)
                continue
            for mb, row in e["layer"].items():
                for key, y in row.items():
                    if not (key.startswith("fused_k") and key[7:].isdigit()) or key[7:] not in rates[name]:
                        continue
                    k, r = int(key[7:]), rates[name][key[7:]]
                    x = min(v for u, v in r.items() if u in ("x_cuda", "x_tc", "x_both"))
                    out.append({"schema": SCHEMA, "lane": "ra-codec-tensor", "model": model,
                                "b": e.get("constants", {}).get("n", int(name.lstrip("pm"))), "k": k,
                                "arithmetic": f"tensor-lane/wt.cu/{name}/untagged", "batch": int(mb),
                                "y_total": y, "y_base": row["plain"], "y_decode": y - row["plain"], "y_tag": 0.0,
                                "y_commit": 0.0, "x": x, "delta": r.get("delta_cuda"), "tau": 0.0, "measured": True,
                                "label": f"{name} k={k} wt.cu untagged", "run": None, "art": art})
    return out


def ingress_of(p: dict) -> float:
    """The point's HBM read-back over E (F19): its own `ingress`, or for ra-codec-wide's points from before the field,
    by their arithmetic: a commit after the pass reads back 36 B a tag (the tag and a sixteenth of a 64-byte leaf), one
    committing as it goes the leaves' 4 B (its tags come back from L2), one that folds them a 64th of that; a tag
    covers 256 B of E a record (512 with tb2) or 2,048 a sketch vector, times the expansion."""
    if "ingress" in p:
        return p["ingress"]
    if p.get("lane") != "ra-codec-wide" or p.get("reference"):
        return 0.0
    a = p["arithmetic"]
    e = (2048 if "/sketch" in a else 512 if "/tb2" in a else 256) * p.get("expansion", 1.0)
    return 4 / e / 64 if "/folded-per-linear" in a else 4 / e if "-per-linear" in a else 36 / e


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
        p["ingress"] = ingress_of(p)
        if p.get("ks"):
            k = min(p["ks"].values())
            p.update(k_mean=p["k"], x=p["x"] * k / p["k"], k=k)
        out.append(p)
    return out


def x_star(tau: float, budget: float, d: float = 1.0) -> float:
    return (d - tau) / budget


def x_star_of(p: dict, budget: float, d: float | None = None, idle: float = IDLE_8B) -> float:
    """The point's x*, its read-back charged to μ's budget (inf when that leaves none)."""
    left = budget - p.get("ingress", 0.0)
    return x_star(p["tau"], left, step(p, d, idle)) if left > 0 else float("inf")


def step(p: dict, d: float | None, idle: float = IDLE_8B) -> float:
    """d, the step's time in passes of E: `--d`, else the point's own `d` when it was measured end to end, else its
    y_total plus the step's idle share (at least one pass)."""
    if d is not None:
        return d
    return p["d"] if p.get("d") is not None else max(1.0, p["y_total"] + idle)


def x_eff(p: dict, boost: float = BOOST) -> float:
    """The point's x at the adversary's clock."""
    return p["x"] if p.get("x_boosted") else p["x"] / boost


def load_rates(src: str) -> list[dict]:
    """`verity/codec-frontier-rates/v0` records (ra-word-codec's measured root rates): a path or an art."""
    path = Path(src) if not src.startswith("art:") else Path(research("data", "fetch", src).strip().splitlines()[-1])
    recs = json.loads(path.read_text())
    if any(r.get("schema") != RATES_SCHEMA for r in recs):
        raise ValueError(f"{src}: a record not in {RATES_SCHEMA}")
    return recs


def rate_x(records: list[dict], boost: float = BOOST, floor: bool = False) -> dict[int, float]:
    """The adversary's x a round per width from rates records: x_max_clock (else x ÷ boost) over k, the lowest of
    the width's; floor: each scaled to the pipes' floor (`PIPE_CLOCKS`[b] over the record's clocks a squaring)."""
    out = {}
    for r in records:
        x = (r.get("x_max_clock") or (r["x"] if r.get("x_boosted") else r["x"] / boost)) / r["k"]
        if floor:
            x *= PIPE_CLOCKS[r["b"]] / r["sm_clocks_per_squaring"]
        out[r["b"]] = min(out.get(r["b"], float("inf")), x)
    return out


def adversary(points: list[dict], boost: float = BOOST, tau_cap: float | None = None,
              rates: dict[int, float] | None = None) -> list[dict]:
    """The points with x the adversary's best encode at b: the lowest x per round at its clock that any point of that
    width reports (whichever lane's library encodes fastest) or `rates` gives (`rate_x`), times k (x_boosted; the
    point's own x kept as x_own), and τ at most tau_cap (the cheapest tag the adversary knows)."""
    per_k = dict(rates or {})
    for p in points:
        if not p.get("reference"):
            per_k[p["b"]] = min(per_k.get(p["b"], float("inf")), x_eff(p, boost) / p["k"])
    out = []
    for p in points:
        q = dict(p)
        if not p.get("reference"):
            q.update(x_own=p["x"], x=per_k[p["b"]] * p["k"], x_boosted=True)
        if tau_cap is not None:
            q["tau"] = min(p["tau"], tau_cap)
        out.append(q)
    return out


def sketch_twins(points: list[dict], cost: dict[int, float], tau: float = 0.0) -> list[dict]:
    """For each measured point (its commit serial: the overlapped twin has the same base and decode), a modelled twin
    with the sketch-then-hash tag in place of the full-byte one: its own base and decode, plus `cost[batch]`, the
    sketch's tag and commit together (ra-sketch-tag's measured y on a plain store), τ the sketch's (0 grants the
    adversary the most) and the read-back of a commit after the pass; a point whose tag is the sketch already has
    none."""
    out = []
    for p in points:
        if (p.get("reference") or not p["measured"] or p["batch"] not in cost or "/sketch" in p["arithmetic"]
                or p["arithmetic"].endswith("/commit-overlapped")):
            continue
        y = p["y_base"] + p["y_decode"] + cost[p["batch"]]
        out.append({**p, "arithmetic": p["arithmetic"] + "/sketch-model", "y_total": y, "y_tag": cost[p["batch"]],
                    "y_commit": 0.0, "tau": tau, "measured": False, "label": f"{p.get('label', '')} + sketch (model)",
                    "ingress": 36 / (2048 * p.get("expansion", 1.0))})
    return out


def tagged(p: dict) -> bool:
    """The point tags and commits what it reads (a tag or commit measured, or named in its arithmetic, as an end-to-end
    point's "/sketch16/commit"); an untagged one certifies nothing and is never on the frontier."""
    a = p["arithmetic"]
    return "untagged" not in a and (p["y_tag"] > 0 or p["y_commit"] > 0 or "/sketch" in a or "commit" in a)


def frontier(points: list[dict], budget: float, d: float | None = None, idle: float = IDLE_8B,
             boost: float = BOOST) -> dict[int, dict | None]:
    best = {}
    for p in points:
        if not admissible(p, budget, d, idle, boost):
            continue
        b = best.get(p["batch"])
        if b is None or step(p, d, idle) < step(b, d, idle):
            best[p["batch"]] = p
    return best


def table(points: list[dict], budget: float, batches=(1, 32), d: float | None = None, idle: float = IDLE_8B,
          boost: float = BOOST) -> str:
    lines = ["| batch | b | k | arithmetic | x (boosted) | x* | μ_rebuild | ingress | y_total | d | decode | tag | commit "
             "| measured | source |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    best = frontier(points, budget, d, idle, boost)
    for mb in batches:
        p = best.get(mb)
        if p is None:
            lines.append(f"| {mb} | none at x ≥ x* | | | | | | | | | | | | | |")
            continue
        dd, xe = step(p, d, idle), x_eff(p, boost)
        lines.append(f"| {mb} | {p['b']} | {p['k']:.3g} | {p['arithmetic']} | {xe:.0f} | {x_star_of(p, budget, d, idle):.0f} "
                     f"| {(dd - p['tau']) / xe * 100:.2f}% | {p.get('ingress', 0.0) * 100:.3f}% | {p['y_total']:.2f} "
                     f"| {dd:.2f}{' (end to end)' if p.get('d') is not None and d is None else ''} | {p['y_decode']:.2f} "
                     f"| {p['y_tag']:.2f} | {p['y_commit']:.2f} | {'yes' if p['measured'] else 'model'} "
                     f"| {p.get('art') or p.get('run') or p.get('lane')} |")
    return "\n".join(lines)


def admissible(p: dict, budget: float, d: float | None = None, idle: float = IDLE_8B, boost: float = BOOST) -> bool:
    return not p.get("reference") and tagged(p) and x_eff(p, boost) >= x_star_of(p, budget, d, idle)


def best_per_config(points: list[dict], budget: float | None = None, d: float | None = None, idle: float = IDLE_8B,
                    boost: float = BOOST) -> list[dict]:
    """The lowest y_total of each (lane, b, k, batch, measured, reference): one bar per configuration, whichever run,
    arithmetic or kernel variant gave it; with `budget`, the lowest of its points at x ≥ x* if it has any."""
    best = {}
    for p in points:
        key = (p.get("lane"), p["b"], p["k"], p["batch"], bool(p["measured"]), bool(p.get("reference")))
        rank = (budget is not None and not admissible(p, budget, d, idle, boost), p["y_total"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, p)
    return [p for _, p in best.values()]


def draw(points: list[dict], out: Path, budget: float, tau_ref: float | None, batches=(1, 32),
         show_all: bool = False, idle: float = IDLE_8B, boost: float = BOOST, model: str = "llama-3.1-8b") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = [p for p in points if p["batch"] in batches]
    if not show_all:
        pts = best_per_config(pts, budget, None, idle, boost)
    lanes = {p.get("lane") for p in pts if not p.get("reference")}
    if not pts:
        raise SystemExit("no points at the plotted batches")
    front = frontier(pts, budget, None, idle, boost).get(batches[0])
    if tau_ref is None:
        tau_ref = front["tau"] if front else statistics.median(p["tau"] for p in pts)
    ingress_ref = front.get("ingress", 0.0) if front else 0.0
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"base": "#9e9e9e", "decode": "#1f77b4", "tag": "#ff7f0e", "commit": "#2ca02c"}
    offset = {mb: 1.0 + 0.09 * (i - (len(batches) - 1) / 2) for i, mb in enumerate(batches)}
    for p in pts:
        xc = x_eff(p, boost) * offset[p["batch"]]
        width = xc * 0.07
        bottom = 0.0
        alpha = 1.0 if p.get("reference") or admissible(p, budget, None, idle, boost) else 0.35
        for part, key in (("base", "y_base"), ("decode", "y_decode"), ("tag", "y_tag"), ("commit", "y_commit")):
            h = max(p[key], 0.0)
            kw = {"color": colors[part]} if p["measured"] else {"facecolor": "none", "edgecolor": colors[part],
                                                                  "linewidth": 1.2}
            ax.bar(xc, h, width=width, bottom=bottom, align="center", alpha=alpha, **kw)
            bottom += h
        if p.get("reference"):
            ax.scatter([xc], [p["y_total"]], marker="*", s=160, color="k" if p["measured"] else "none", edgecolors="k",
                       zorder=5)
        if p["batch"] == batches[0] or show_all:
            name = p.get("label") if p.get("reference") else f"b={p['b']} k={p['k']}"
            if len(lanes) > 1 and not p.get("reference"):
                name += f"\n{p.get('lane')}"
            ax.annotate(name + (f" M={p['batch']}" if show_all else ""), (xc, bottom), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=6)
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
    left = budget - ingress_ref
    boundary, = ax.plot(xx, tau_ref - idle + left * xx, color="purple", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(3.0, max(p["y_total"] for p in pts) * 1.1))
    ax.set_xlabel(f"x: encode time of one pass's data, in native passes, at the adversary's clock (÷{boost:g} where "
                  "timed locked; log)")
    ax.set_ylabel(f"y: fused decode step / native, {model.replace('llama', 'Llama')} linears, layer-weighted")

    def mu(x):
        return (1 - tau_ref) * 100 / np.maximum(np.asarray(x, dtype=float), 1e-9)

    sec = ax.secondary_xaxis("top", functions=(mu, mu))
    sec.set_xlabel(f"μ_rebuild = (d − τ)/x, % (d = 1, τ = {tau_ref:.2f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k"),
                plt.Rectangle((0, 0), 1, 1, color="k", alpha=0.35), boundary]
    ax.legend(handles, [*colors, "modelled (hollow)", "x < x*, or untagged (pale)",
                        f"y = τ − {idle:g} + ({budget:g} − ingress)·x: x* at d = y + {idle:g}\n"
                        f"(τ {tau_ref:.3f}, ingress {ingress_ref * 100:.3f}%: batch {batches[0]}'s frontier)"],
              loc="upper left", fontsize=8)
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
    ap.add_argument("--d", type=float, help="the step's time in passes of E for x* (default: each point's y_total + "
                    "--idle)")
    ap.add_argument("--model", choices=sorted(IDLE), default="llama-3.1-70b",
                    help="draw this model's points (a point without `model` is 8B's)")
    ap.add_argument("--idle", type=float,
                    help="the step outside the linears, in passes of E (default by --model: "
                    + ", ".join(f"{m} {v}" for m, v in IDLE.items()) + ")")
    ap.add_argument("--boost", type=float, default=BOOST,
                    help=f"the adversary's clock over the points' locked clock (default {BOOST})")
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--p2", help="X,Y,TAU: P2 as a modelled reference point")
    ap.add_argument("--all", action="store_true", help="draw every point, not the best of each configuration")
    ap.add_argument("--own-x", action="store_true",
                    help="each point's own x, not the best encode any point of its width reports (`adversary`)")
    ap.add_argument("--rates", action="append", default=[],
                    help=f"{RATES_SCHEMA} records (a path or art): their x pooled into the adversary's best per width")
    ap.add_argument("--floor", action="store_true", help="with --rates: their x at the pipes' floor (`PIPE_CLOCKS`)")
    ap.add_argument("--tau-cap", type=float, help="τ at most this: the cheapest tag the adversary knows")
    ap.add_argument("--sketch", help="BATCH:Y,...: add each measured point's modelled twin with the sketch-then-hash "
                    "tag, Y its tag and commit at that batch (e.g. 1:0.073,32:0.150, ra-sketch-tag's plain store)")
    ap.add_argument("--sketch-tau", type=float, default=0.0, help="the sketch twins' τ (default 0)")
    ap.add_argument("--only-sketch", action="store_true", help="with --sketch: draw and rank only the twins")
    ap.add_argument("--tensor", action="append", default=[],
                    help="ra-codec-tensor's codec-tensor.json, a path or its run's files art (`tensor_points`)")
    a = ap.parse_args()
    points = []
    arts = a.art or ([] if a.file else store_arts())
    for art in arts:
        points += load_art(art)
    for f in a.file:
        points += json.loads(Path(f).read_text())
    points += tensor_points([load_tensor(s) for s in a.tensor])
    batches = tuple(int(b) for b in a.batches.split(","))
    if a.p2:
        x, y, tau = (float(v) for v in a.p2.split(","))
        for mb in batches:
            points.append({"schema": SCHEMA, "lane": "reference", "b": 16448, "k": 1, "arithmetic": "P2", "batch": mb,
                           "y_total": y, "y_decode": 0.0, "y_tag": 0.0, "y_commit": 0.0, "y_base": y, "x": x,
                           "tau": tau, "measured": False, "reference": True, "label": "P2"})
    points = check(points)
    if a.sketch:
        cost = {int(b): float(y) for b, y in (kv.split(":") for kv in a.sketch.split(","))}
        twins = sketch_twins(points, cost, a.sketch_tau)
        points = twins + [p for p in points if p.get("reference")] if a.only_sketch else points + twins
    if not a.own_x:
        rates = rate_x([r for s in a.rates for r in load_rates(s)], a.boost, a.floor) if a.rates else None
        points = adversary(points, a.boost, a.tau_cap, rates)
    elif a.tau_cap is not None:
        points = [{**p, "tau": min(p["tau"], a.tau_cap)} for p in points]
    points = [p for p in points if p.get("reference") or model_of(p) == a.model]
    idle = IDLE[a.model] if a.idle is None else a.idle
    print(f"{a.model}, idle {idle:g}")
    print(table(points, a.budget, batches, a.d, idle, a.boost))
    if a.out:
        draw(points, Path(a.out), a.budget, a.tau, batches, a.all, idle, a.boost, a.model)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
