"""The console's training panel, `training/frontier-branches`: held-out bits per byte against total training compute in
Boolean gates, every line a frontier (at each loss, the least compute anything reached under that line's rules).

    python -m te_core.panels frontier-branches --out DIR [--publish] [--pretraining DIR]
                                               [te_core.rescore's ledger and quantities options]

Its rows (`COLUMNS`) are the points of three kinds of curve, each on one data (`data`: `v3`, harness v3's OLMo 2 mix,
or `v2.3-pilot`, the v2.3 harness's FineWeb-Edu; a loss on one is no loss on the other, so every curve is per data):

- `frontier`, the best unconstrained learner: the lower envelope by total compute of every honest point, from scratch
  or continuing a start. From scratch: every start at (C0, l0), and on the pilot learning_frontier's dense AdamW
  pretraining runs (`PRETRAINING`: the runs the starts were taken from, and its largest size) at every other
  evaluation; continuing: every valid checkpoint from a start, at C0 + |C|.
- `honest`, per branch point, the best unconstrained learner starting from that checkpoint: at each loss the least |C|
  after it over every checkpoint of every valid record from it (te_core.frontier's H), at total compute C0 + |C|.
- `throttled`, per branch point and dial (c, k), the best constrained learner starting from it: the least |C|·ρ (W*),
  ρ by the formalism v2 at that (c, k) with A1's and A1''s upper bounds, its S = W*/H at its loss and W*'s basis
  (`fit`: ρ = c/(c − d) by Fit; `target`: Slowdown's bound, stated, not proved in Lean). The dial is c in tokens per
  block of the start's own forward (`TOKENS_PER_BLOCK`, c = n·w; `c_gates` = n·w) by k (`K_BYTES`; bits in the rows),
  and two cells are marked (`mark`): the default (`DEFAULT`, `reference`) and the served demo's measured block
  (`SERVED`, `served demo`, at its gates in the start's tokens). A dial cell whose curve is the honest one (ρ = 1 at
  every point of it, so S = 1 at every loss) has no rows, and a checkpoint whose ρ is infinite in a cell (no block
  there does any of its unreachable work) is on no throttled curve of that cell.

It reads the valid records of the pin and of the pilot (pin.py's `pilot`, `frontier.pinned(pilot=True)`) unless `--pin`
names the commits, and only checkpoints the scoring cap admits are scored (te_core.policy's `eligible`, the ladder's fit
from `--ladder-fit`).
The branch points are the starts (te_core.policy's `starts`) per data, each at (C0, l0) and labelled by its size and
tokens per parameter, `on_frontier` "yes" when its data's frontier passes through it (nothing reaches its loss for
less), else "no". C0 is
the start's counted training gates when its shape has them, else its training tokens at dense AdamW's counted rate
(`dense_rate`: the gates per token of ref-optim's AdamW-everything checkpoints, in forward passes of the same model, the
rate a continuation's |C| is counted at), and a from-scratch point's C0 is its tokens at that rate. A from-scratch point's loss
is learning_frontier's held-out bits per byte (fineweb-edu shard 013's first 2^21 tokens as one stream) times the mean
ratio of the starts' l0 (the search pool's whole documents) to it at the same checkpoints (`ratio`), except at a start,
whose point is its l0.

Each curve keeps its envelope points only. The console's copy (`cut`) is the headline data's rows (`headline`: v3 once
any start has v3 points, else the pilot), fit to the panels route's limits (te_core.console): the frontier, each honest
curve and each marked throttled curve keep their points `FRONTIER_STEP` apart in loss, and every other throttled curve
the most points (`POINTS`) that fit, the first to reach each of losses evenly spaced over its own span. DIR gets
`frontier-branches.json` (uncut, every data), `console/frontier-branches.json` (cut, which `--publish` PUTs) and
`frontier-branches.png`, plotted from the cut rows at the marked cells.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import VERSION, console, formalism, frontier, plots, policy, rescore

NAME = "frontier-branches"
PANEL = f"training/{NAME}"
TOKENS_PER_BLOCK = (1, 10, 100, 1000, 4096)
K_BYTES = (10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)
#: the dial's default, (tokens per block, k bits): one 4,096-token session per block, k = 10 kB (te_core.policy's
#: reference)
DEFAULT = (policy.SESSION_TOKENS, policy.REFERENCE_K_BITS)
#: the served demo's measured block, (c gates, k bits): one slot's 35-step session of Llama-3.2-1B int8 under vLLM, c
#: in one-bit gates (a lower bound: its 133 BitAtx32_v1 activations a block are unpriced), k its fresh bits
SERVED = (30_087_822_237_972, 145_568)
#: the data a record's held-out loss is on, the headline first: harness v3's, else the v2.3 harness's (`data_of`)
DATA = ("v3", "v2.3-pilot")
COLUMNS = ["curve", "data", "start", "start_compute_gates", "start_loss", "on_frontier", "c_tokens_per_block", "c_gates",
           "k_bits", "mark", "compute_total_gates", "compute_after_gates", "loss", "S", "run", "method", "delta_bits",
           "basis"]
LOSSES = ("start_loss", "loss")
#: learning_frontier's dense AdamW pretraining runs s, m and l (seed 0, 8 tokens per non-embedding parameter), each
#: run's config.json, curve.json and theta.json under its run id
PRETRAINING = "art:8c8df28509c05c7cc8f1db3ff584c44339f19019b453370afbd75b9e5fa88008"
#: forward MACs per token at context 1024 of a pretraining size no start has, by parameter count: l's, counted as
#: starts_store.shape counts a start's (on CPU, attention in its math form)
FORWARD_MACS = {205_677_568: 230_976_512}
#: the records `dense_rate` reads: a direction and a mechanism prefix (AdamW on every tensor)
DENSE = ("ref-optim", "adamw")
#: the console cut's points per unmarked throttled curve, n + 1 at most (None: every point), and the least loss step
#: of every other curve (the frontier, honest, the marked throttled ones), tried finest first (`STEPS`)
POINTS = (None, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1)
FRONTIER_STEP = 0.0025
STEPS = (FRONTIER_STEP, 2 * FRONTIER_STEP, 4 * FRONTIER_STEP, 8 * FRONTIER_STEP)
TARGETS = frontier.TARGETS


def fetch(art: str) -> Path:
    """`art`'s tree, materialised from the evidence store (`research data fetch`)."""
    out = subprocess.run(["research", "data", "fetch", art], check=True, capture_output=True, text=True).stdout
    return Path(out.strip().splitlines()[-1])


def pretraining(root: Path) -> list[dict]:
    """Each run under `root` (a directory per run id): its size, parameters, tokens per step and, per evaluation, its
    step, tokens and held-out bits per byte (learning_frontier's)."""
    out = []
    for d in sorted(p for p in root.iterdir() if (p / "curve.json").exists()):
        cfg = json.loads((d / "config.json").read_text())
        pts = [{"step": p["step"], "tokens": p["tokens"], "bpb": p["val_nats_per_byte"] / math.log(2)}
               for p in json.loads((d / "curve.json").read_text())]
        out.append({"run": d.name, "size": cfg["args"]["size"], "parameters": cfg["counts"]["total"], "points": pts})
    if not out:
        raise SystemExit(f"panels: no pretraining run under {root}: `research data fetch {PRETRAINING}` gives the tree")
    return out


def data_of(r: dict) -> str:
    """The data a ledger row's held-out loss is on (`DATA`): the evaluation's `data` v3, else the pilot."""
    return DATA[0] if r.get("data") == "v3" else DATA[1]


def dense_rate(scored: list[dict], shapes: dict[str, dict]) -> tuple[float, str]:
    """Dense AdamW's gates per training token in forward passes of its model: the median over the valid checkpoints of
    `DENSE`'s records, else 3 forward passes (te_core.policy's stub), and where it comes from."""
    v = sorted(r["gates"] / (r["measured"].tokens * policy.w_fwd(shapes[r["start"]])) for r in scored
               if r.get("direction") == DENSE[0] and str(r.get("mechanism", "")).startswith(DENSE[1])
               and r["measured"].tokens)
    if not v:
        return 3.0, "stub: 3 forward passes per training token"
    return v[len(v) // 2], f"counted: the median of {len(v)} AdamW-everything checkpoints"


def c0(shape: dict, rate: float) -> float:
    gates, how = policy.c0(shape)
    return rate * policy.w_fwd(shape) * shape["tokens"] if how.startswith("stub") else gates


def cells(w: float) -> dict[tuple[float, int], str | None]:
    """The dial at a start whose forward is `w` gates per token, (tokens per block, k bits) -> its mark: every
    `TOKENS_PER_BLOCK` by every `K_BYTES`, `DEFAULT` marked `reference`, and the served demo's block (`SERVED`) at its
    gates in the start's tokens."""
    return ({(n, 8 * kb): None for n in TOKENS_PER_BLOCK for kb in K_BYTES}
            | {DEFAULT: "reference", (SERVED[0] / w, SERVED[1]): "served demo"})


def dial(m: formalism.Measured, w: float) -> dict[tuple[float, int], dict]:
    """The formalism at each (c, k) of the dial (`cells`)."""
    return {(n, k): formalism.at(m, n * w, k) for n, k in cells(w)}


def least(stair: list[tuple], loss: float) -> float | None:
    """The least compute of a staircase (te_core.plots') reaching `loss`."""
    return next((x for x, y, _ in stair if y <= loss), None)


class Branch:
    """One start on one data as a branch point: its rows, l0, C0, label, forward w, dial and its two kinds of curve."""

    def __init__(self, name: str, shape: dict, on: list[dict], rate: float, data: str):
        self.name, self.shape, self.on, self.data = name, shape, on, data
        self.l0, self.c0, self.label = frontier.start_bpb(on), c0(shape, rate), plots.start_label(shape)
        self.w = policy.w_fwd(shape)
        self.cells = cells(self.w)
        for r in on:
            r["dial"] = dial(r["measured"], self.w)
        self.honest = plots.staircase([(r["gates"], r["val_bpb"], r) for r in on])
        self.throttled = {key: plots.staircase([(r["gates"] * r["dial"][key]["rho"], r["val_bpb"], r) for r in on
                                                if math.isfinite(r["dial"][key]["rho"])])
                          for key in self.cells}
        self.on_frontier = False

    def s_at(self, key: tuple[float, int], loss: float) -> tuple[float | None, dict | None]:
        """S = W*/H at `loss` in dial cell `key`, and the checkpoint that sets W*."""
        h = least(self.honest, loss)
        hit = next(((x, r) for x, y, r in self.throttled[key] if y <= loss), None)
        return (hit[0] / h, hit[1]) if hit and h else (None, None)


def scratch(runs: list[dict], branches: list[Branch], shapes: dict[str, dict], rate: float) -> tuple[list, float]:
    """The from-scratch points (total compute, loss, row with its `data`) and the loss ratio (module docstring): every
    start at its (C0, l0), and every other pretraining evaluation (the pilot's data) at its tokens' compute and its
    converted loss."""
    pilot = [b for b in branches if b.data == DATA[1]]

    def start_of(run, p):
        return next((b for b in pilot if b.shape.get("parameters") == run["parameters"]
                     and b.shape.get("tokens") == p["tokens"]), None)

    def what(run):
        return f"dense AdamW from scratch (learning_frontier {run['size']})"

    at = {b.name: (run, p) for run in runs for p in run["points"] if (b := start_of(run, p))}
    pairs = [(b.l0, at[b.name][1]["bpb"]) for b in pilot if b.name in at]
    ratio = sum(a / b for a, b in pairs) / len(pairs) if pairs else 1.0
    out = []
    for b in branches:
        run, p = at.get(b.name, (None, None)) if b.data == DATA[1] else (None, None)
        out.append((b.c0, b.l0, {"data": b.data, "run": f"{run['run']}@{p['step']}" if run else b.name,
                                 "start": b.name,
                                 "method": (what(run) + ", " if run else "from scratch, ") + f"the start {b.name}"}))
    for run in runs:
        same = next((s for s in shapes.values() if s.get("parameters") == run["parameters"]), None)
        macs = same["forward_macs_per_token"] if same else FORWARD_MACS.get(run["parameters"])
        if macs is None:
            raise SystemExit(f"panels: no forward MACs per token for {run['run']} ({run['parameters']} parameters): "
                             "add its count to te_core.panels' FORWARD_MACS")
        out += [(rate * policy.w_fwd({"forward_macs_per_token": macs}) * p["tokens"], p["bpb"] * ratio,
                 {"data": DATA[1], "run": f"{run['run']}@{p['step']}", "method": what(run) + ", loss converted"})
                for p in run["points"] if not start_of(run, p)]
    return out, ratio


def _method(r: dict) -> str:
    return f"{r.get('direction')}/{r.get('mechanism')}"


def _delta(r: dict) -> float | None:
    return r["measured"].delta_bits


def row(**values) -> list:
    return [values.get(c) for c in COLUMNS]


def build(scored: list[dict], shapes: dict[str, dict], runs: list[dict], source: str,
          stamp: str | None = None) -> tuple[dict, dict]:
    """The panel (uncut, every data) and what it rests on: the branch points, the rate, the ratio, each data's frontier
    and the headline data."""
    rate, rate_how = dense_rate(scored, shapes)
    branches = [b for b in (Branch(s, shapes[s], [r for r in scored if (r["start"], data_of(r)) == (s, d)], rate, d)
                            for s, d in sorted({(r["start"], data_of(r)) for r in scored})) if b.l0 is not None]
    points, ratio = scratch(runs, branches, shapes, rate)
    for b in branches:
        points += [(b.c0 + x, y, {"data": b.data, "run": f"{r['id']}@{r.get('checkpoint_step')}",
                                  "method": f"{_method(r)} from {b.name}", "delta_bits": _delta(r)})
                   for x, y, r in b.honest]
    envs = {d: plots.staircase([p for p in points if p[2]["data"] == d]) for d in DATA}
    rows = [row(curve="frontier", data=d, start="", compute_total_gates=x, loss=y, run=r["run"], method=r["method"],
                delta_bits=r.get("delta_bits")) for d, env in envs.items() for x, y, r in env]
    for b in branches:
        b.on_frontier = any(r.get("start") == b.name for _, _, r in envs[b.data])
        head = {"data": b.data, "start": b.label, "start_compute_gates": b.c0, "start_loss": b.l0,
                "on_frontier": "yes" if b.on_frontier else "no"}
        for x, y, r in b.honest:
            rows.append(row(curve="honest", **head, compute_total_gates=b.c0 + x, compute_after_gates=x, loss=y,
                            run=f"{r['id']}@{r.get('checkpoint_step')}", method=_method(r), delta_bits=_delta(r)))
        for (n, k), stair in b.throttled.items():
            if [(x, r["id"]) for x, _, r in stair] == [(x, r["id"]) for x, _, r in b.honest]:
                continue
            for x, y, r in stair:
                h = least(b.honest, y)
                rows.append(row(curve="throttled", **head, c_tokens_per_block=n, c_gates=n * b.w, k_bits=k,
                                mark=b.cells[(n, k)], compute_total_gates=b.c0 + x, compute_after_gates=x, loss=y,
                                S=x / h if h else None, run=f"{r['id']}@{r.get('checkpoint_step')}",
                                method=_method(r), delta_bits=_delta(r),
                                basis="fit" if r["dial"][(n, k)]["fit"] else "target"))
    headline = next((d for d in DATA if any(b.data == d for b in branches)), DATA[-1])
    p = {"id": PANEL, "title": "The frontier and its branches: loss against total training compute", "kind": "table",
         "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source,
         "note": note(ratio, None, headline), "columns": COLUMNS, "rows": rows}
    return p, {"branches": branches, "rate": rate, "rate_how": rate_how, "ratio": ratio, "frontier": envs,
               "headline": headline}


def note(ratio: float, n: int | None, data: str, step: float = FRONTIER_STEP) -> str:
    scratch = f"From-scratch loss: learning_frontier's x{ratio:.4f}. " if data == DATA[1] else "From scratch: the starts. "
    thin = "" if n is None else (f" Other cells: {n + 1} points"
                                 + (f", rest {100 * step:g}% apart" if step != FRONTIER_STEP else "") + ".")
    return (f"Held-out bits/byte ({data}) vs total training gates. frontier: the best unconstrained learner (from scratch "
            "or a start). honest / throttled: the best unconstrained / constrained learner starting from checkpoint "
            "`start`; throttled: |C| rho (formalism v2, A1/A1' bounds); S = W*/H, 1 at a cell with no rows. c: "
            f"start-forward tokens; marked: reference, served demo. C0: dense AdamW's gates/token. {scratch}One seed."
            + thin)


def rounded(row: list) -> list:
    return [round(v, 5) if c in LOSSES and isinstance(v, float) else
            float(f"{v:.4g}") if isinstance(v, float) else v for c, v in zip(COLUMNS, row)]


def thinned(rows: list[list], n: int | None, step: float = FRONTIER_STEP) -> list[list]:
    """The frontier's, each honest curve's and each marked throttled curve's points at least `step` apart in loss,
    and each other throttled curve's at most n + 1, the first to reach each of n + 1 losses evenly spaced over its own
    span; a curve's first and last points and the frontier's branch points always kept, and every point with n None."""
    if n is None:
        return rows
    ix = {c: i for i, c in enumerate(COLUMNS)}
    key = [ix[c] for c in ("curve", "data", "start", "c_tokens_per_block", "k_bits", "mark")]
    starts = {(r[ix["data"]], r[ix["start_compute_gates"]], r[ix["start_loss"]]) for r in rows
              if r[ix["curve"]] != "frontier"}
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(tuple(r[i] for i in key), []).append(r)
    out = []
    for (curve, data, _, _, _, mark), g in groups.items():
        loss = [r[ix["loss"]] for r in g]
        if curve != "throttled" or mark:
            kept = [0]
            for i in range(1, len(g)):
                if i == len(g) - 1 or loss[i] <= loss[kept[-1]] * (1 - step) or \
                        curve == "frontier" and (data, g[i][ix["compute_total_gates"]], loss[i]) in starts:
                    kept.append(i)
        else:
            levels = [loss[0] - j * (loss[0] - loss[-1]) / n for j in range(n + 1)]
            kept = sorted({next(i for i, y in enumerate(loss) if y <= lv + 1e-12) for lv in levels})
        out += [g[i] for i in kept]
    return out


def cut(p: dict, ratio: float, data: str) -> dict:
    """`p`'s rows on `data` at the finest loss step (`STEPS`) and then the most points per unmarked throttled curve
    (`POINTS`) whose body fits the panels route, numbers to 4 significant digits (losses to 5 decimals)."""
    on = [r for r in p["rows"] if r[COLUMNS.index("data")] == data]
    for step in STEPS:
        for n in POINTS if step == STEPS[0] else POINTS[1:]:
            q = p | {"rows": [rounded(r) for r in thinned(on, n, step)], "note": note(ratio, n, data, step)}
            if len(q["note"]) > console.NOTE_MAX:
                raise ValueError(f"{PANEL}: its note is {len(q['note'])} characters, over {console.NOTE_MAX}")
            if bad := console.refused(q):
                raise ValueError(f"{PANEL}: the panels route refuses {len(bad)} cells ({'; '.join(bad[:3])}): make each "
                                 f"a number, a string of at most {console.CELL_MAX} characters or null")
            if len(console.body(q)) <= console.BODY_MAX:
                return q
    raise ValueError(f"{PANEL}: no cut fits {console.BODY_MAX} bytes (the last is {len(console.body(q))})")


def figure(p: dict, out: Path, data: str) -> Path:
    """The figure on `data` at the marked cells: the frontier dashed, each branch point's honest curve thin, its
    throttled one at the default dial thick and at the served demo's block dotted."""
    rows = plots.frame(p)
    fig, ax = plt.subplots(figsize=(9, 6))

    def step(pts, **kw):
        if pts:
            ax.step(*zip(*pts), where="post", **kw)

    step([(r["compute_total_gates"], r["loss"]) for r in rows if r["curve"] == "frontier"], color="C0", ls="--",
         lw=1.4, label="best unconstrained learner")
    for i, s in enumerate(dict.fromkeys(r["start"] for r in rows if r["start"])):
        mine = [r for r in rows if r["start"] == s]
        b = mine[0]
        step([(b["start_compute_gates"], b["start_loss"])] +
             [(r["compute_total_gates"], r["loss"]) for r in mine if r["curve"] == "honest"], color="C0", lw=0.8)
        for mark, kw in (("reference", {"lw": 2}), ("served demo", {"lw": 1.2, "ls": ":"})):
            step([(b["start_compute_gates"], b["start_loss"])] +
                 [(r["compute_total_gates"], r["loss"]) for r in mine
                  if r["curve"] == "throttled" and r["mark"] == mark], color="C3", **kw)
        on = b["on_frontier"] == "yes"
        ax.plot([b["start_compute_gates"]], [b["start_loss"]], "o", color="k" if on else "0.6")
        ax.annotate(s + ("" if on else " (off)"), (b["start_compute_gates"], b["start_loss"]),
                    fontsize=7, xytext=(6, 4) if i % 2 else (-6, -10), ha="left" if i % 2 else "right",
                    textcoords="offset points")
    ax.plot([], [], color="C0", lw=0.8, label="best unconstrained learner from a checkpoint")
    ax.plot([], [], color="C3", lw=2, label=f"best constrained learner from it, c = {DEFAULT[0]} tokens per block, "
                                            f"k = {DEFAULT[1] / 8000:g} kB")
    ax.plot([], [], color="C3", lw=1.2, ls=":", label=f"the same at the served demo's block, c = {SERVED[0]:.3g} gates, "
                                                      f"k = {SERVED[1]:,} bits")
    ax.set_xscale("log")
    ax.set_xlabel("total training compute (Boolean gates)")
    ax.set_ylabel(f"held-out bits per byte ({data} data)")
    ax.set_title(p["title"], fontsize=10)
    ax.legend(fontsize=7)
    fig.text(0.01, 0.005, p["note"], fontsize=5.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out / f"{NAME}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary(info: dict) -> str:
    """The branch points on every data, each with whether it lies on its data's frontier, and S at the default dial
    per target."""
    lines = [f"C0 rate: {info['rate']:.4g} forward passes per training token ({info['rate_how']}); from-scratch loss "
             f"ratio {info['ratio']:.5f}; the console shows {info['headline']}", "",
             "| data | branch point | start | C0 (gates) | l0 | on the frontier | "
             + " | ".join(f"S at -{100 * t:g}%" for t in TARGETS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(TARGETS)]
    for b in info["branches"]:
        ss = []
        for t in TARGETS:
            s, r = b.s_at(DEFAULT, b.l0 * (1 - t))
            ss.append("unreached" if s is None else f"{s:.3g} ({'fit' if r['dial'][DEFAULT]['fit'] else 'target'})")
        lines.append(f"| {b.data} | {b.label} | {b.name} | {b.c0:.3g} | {b.l0:.4f} | "
                     f"{'yes' if b.on_frontier else 'no'} | " + " | ".join(ss) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="panel", required=True)
    fb = rescore.arguments(sub.add_parser(NAME, help="the frontier and its branches"))
    fb.add_argument("--publish", action="store_true", help=f"PUT the cut panel to {console.PANELS_URL}/{PANEL}")
    fb.add_argument("--pretraining", type=Path, default=None,
                    help=f"the pretraining runs, a directory per run id (default {PRETRAINING}, fetched)")
    a = ap.parse_args(argv)
    d = rescore.load(a)
    pin, shapes = d["pin"] if a.pin else frontier.pinned(pilot=True), d["shapes"]
    scored, _ = plots.load(d["rows"], pin, shapes, d["keep"], d["ladder"])
    runs = pretraining(a.pretraining or fetch(PRETRAINING))
    source = a.source or (f"python -m te_core.panels {NAME} (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                          f"{len(scored)} valid checkpoints, quantities {d['quantities']})")
    p, info = build(scored, shapes, runs, source)
    q = cut(p, info["ratio"], info["headline"])
    (a.out / f"{NAME}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
    (a.out / "console").mkdir(exist_ok=True)
    (a.out / "console" / f"{NAME}.json").write_bytes(console.body(q))
    png = figure(q, a.out, info["headline"])
    print(summary(info))
    print(f"{PANEL}: {len(p['rows'])} rows uncut; console {len(q['rows'])} rows, {len(console.body(q))} bytes, note "
          f"{len(q['note'])} characters -> {a.out}: {NAME}.json, console/{NAME}.json, {png.name}")
    if a.publish:
        status = console.put(q)
        print(f"PUT {console.PANELS_URL}/{PANEL}: {status}")
        return 0 if status.startswith("2") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
