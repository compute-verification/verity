"""Figure 1 of the SPOT post in its total-cost form: what an adversary serving a model on all of a scope's compute over
a monitoring period can exfiltrate or infiltrate while evading detection with probability above δ, against what the
deployment costs as a share of serving compute, one line per server configuration; and, for the one configuration
that bounds it, skipped work.

A configuration (`CONFIGURATIONS`) is what serving commits at serve time, the replay block that follows from it and the
design the audit samples with.  Its serve tax, what that change to the production server costs on every token, is one
number in the model's `serve_tax` card, priced (`priced_card`) or measured.  A line's x is its tax plus β, the share
of the scope's compute the audit spends on replays, interior commitments and proofs of the sampled proof blocks, and
its y the bound at β, every cost at the `proof_overhead` card's marginal overhead on Figure 1's model
(`figure1.model`).  So each line starts at its own tax.

* `tokens` (A): serving commits the token ids, the schedule record and the KV cache at each window edge; a replay block
  is a request, and the design the width-stratified two-stage one.  Exfiltration is its capped sets' form
  (`card_influenceTwo_le_served_min`), never above the scope's served bits (`card_influenceTwo_le_min`).  Infiltration
  is its write form (`write_form`) at a footprint of a request's served bits: an incorrect request is charged at most
  the cheapest explanation of what it serves.
* `layer_of_token` (B): serving also commits each layer's residual stream, K/V and indexer keys; a replay block is one
  layer of one token, holding its share of the token's proof blocks and replayed at its share of its request's
  replay.  Exfiltration is the planner's design on that grain in the capped sets' form, each replay block charged at
  most its request's served bits, which is at least the lemma's charge because min(·, S) is subadditive, and never
  above the scope's served bits.  Infiltration is its write form at a footprint of the layer's boundary bits.
* `every_block` (C): serving commits every proof block's outputs, and the audit samples them one-stage; its tax is a
  lower bound, the hashing alone.  Exfiltration is `consumers.exfiltration_bound`'s form, at most the served bits by
  the same count, which Lean states for two-stage only; infiltration is its write form (the K incorrect blocks the plan
  detects, each the widest block's bits) and skipped work the work target's plan.

Each configuration's two panels read one design, the one with the least exfiltration bound at β; skipped work is
planned on its own.  The serve tax is outside β; the token ids' hashing is in the tax, not in β, for every line.

    uv run python benchmarks/spot_figures/total_cost.py priced-card --model GLM-5.3 --out CARD.json
    uv run python benchmarks/spot_figures/total_cost.py render DIR/figure1.json [--scope "world year"] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import ceil, expm1, floor, inf, log, log1p, log10
from pathlib import Path

import cards as spot_cards
import figure1 as f1
from verity.core.protocols.sampled_proofs.optimize import Cell, leak_count, two_stage_design
from verity.ml.circuits.partition_v2 import unit_class

SCHEMA = "spot-figure1/v2"
#: β, the share of compute for replays and proofs, from 0.01% to 1,000% at four points a decade, and the design card's:
#: past 100% so that (C), whose tax alone is several times serving, has a line
BETAS = tuple(10.0 ** (e / 4) for e in range(-16, 5))
PANELS = ("exfiltration", "infiltration", "skipped_work")
#: the throughput the certified-tick rule keeps against vLLM, with a queue-dependent tick (0.9× the step while the
#: queue stands, 1.1× otherwise): the mean run's and the worst run's, simulated on Llama-3.2-1B traces
TICKS = (0.984, 0.960)
#: a request's lifetime over the 600 s window, the share of its K/V alive at an edge: the central, least and most
LIVE = (0.15, 0.05, 0.5)
#: SHA-512 at 3.1 and 4.79 G compressions a second, 44 a 4 KiB row (`benchmarks/prove_path/window_cost.py`'s `RATES`)
HASH_RATES = (4096 / 44 * 4.79e9, 4096 / 44 * 3.1e9)
#: GLM-5.3's K/V and indexer keys a token, the state a later token reads (576 BF16 a layer and the indexer keys)
KV_BYTES = 92_628


@dataclass(frozen=True)
class Configuration:
    """A line: its label, what serving commits, its replay block and design, and the panels it bounds."""

    label: str
    commits: str
    replay_block: str
    design: str
    panels: tuple[str, ...]


CONFIGURATIONS = {
    "tokens": Configuration("(A) Tokens only", "the token ids, the schedule record and the KV cache at window edges",
                            "a request", "two-stage, width-stratified", ("exfiltration", "infiltration")),
    "layer_of_token": Configuration("(B) Each layer's state",
                                    "the tokens and each layer's residual stream, K/V and indexer keys",
                                    "one layer of one token", "two-stage, width-stratified",
                                    ("exfiltration", "infiltration")),
    "every_block": Configuration("(C) Every proof-block boundary", "every proof block's outputs", "a proof block",
                                 "one-stage", PANELS),
}


def priced_card(model: str = "GLM-5.3") -> dict:
    """The model's `serve_tax` card priced from its parts: each part's GPU-seconds a token over a token's at serving's
    MFU, hashing at the `commitment` card's rate (and at `HASH_RATES` for the range), the tick rule's throughput loss
    (`TICKS`) and the K/V alive at a window edge (`LIVE`).  (C)'s is a lower bound: hashing alone."""
    c = f1.cards(model)
    sc = f1.slot_cards(model)
    token_s = 2 * c["partition"].value["served_macs_per_token"] / (c["mfu_serve"].value * c["peak_flops"].value)
    rate = c["hash_rate"].value

    def hashed(part: str, nbytes: float, source: str, scale: tuple[float, float, float] = (1, 1, 1)) -> dict:
        share = [nbytes * s / r / token_s for s, r in zip(scale, (rate, *HASH_RATES))]
        return {"part": part, "share": share[0], "low": share[1], "high": share[2],
                "source": f"{nbytes:,.0f} B a token hashed at the commitment card's rate, the range at "
                          f"{HASH_RATES[0] / 1e9:.0f} and {HASH_RATES[1] / 1e9:.0f} GB/s; {source}"}

    ticks = {"part": "certified-tick rule", "share": 1 / TICKS[0] - 1, "low": 1 / TICKS[0] - 1,
             "high": 1 / TICKS[1] - 1,
             "source": f"throughput {TICKS[0]} of vLLM's on the mean run and {TICKS[1]} on the worst, a queue-dependent "
                       "tick on Llama-3.2-1B traces, simulated and validated against them; it makes the advice 0 bits"}
    tokens = hashed("token ids", c["token_bytes"].value, "the design card's token_bytes")
    kv = hashed("K/V at window edges", KV_BYTES, f"the K/V alive at an edge, a request's lifetime over the 600 s "
                f"window: {LIVE[0]}, from {LIVE[1]} to {LIVE[2]}", (LIVE[0], LIVE[1], LIVE[2]))
    layers = hashed("each layer's state", sc["infiltration"]["value"]["boundary_bytes_per_token"],
                    "the infiltration card's boundary_bytes_per_token")
    blocks = hashed("every proof block's outputs", sc["workload"]["value"]["boundaries"]["every_block"],
                    "the workload card's every_block boundary")

    def entry(config: str, parts: list[dict], lower_bound: bool = False) -> dict:
        return {"config": config, "status": "priced", "tax": sum(p["share"] for p in parts),
                "low": sum(p["low"] for p in parts), "high": sum(p["high"] for p in parts), "lower_bound": lower_bound,
                "parts": parts}

    return spot_cards.check({
        "schema": spot_cards.SCHEMA, "slot": "serve_tax", "model": model, "status": "modeled", "provisional": True,
        "source": "priced from each configuration's parts against a token's GPU-seconds at serving's MFU on the H200's "
                  "dense FP8 peak (total_cost.priced_card); measured values replace it configuration by configuration",
        "runs": [], "owner": "@proofs",
        "value": {"baseline": "stock vLLM, compiled, on the model's served circuit",
                  "configurations": [entry("tokens", [ticks, tokens, kv]),
                                     entry("layer_of_token", [ticks, tokens, layers]),
                                     entry("every_block", [ticks, blocks], lower_bound=True)]}})


def taxes(card: dict) -> dict[str, dict]:
    """Each configuration's tax from the `serve_tax` card, every configuration given."""
    out = {t["config"]: t for t in card["value"]["configurations"]}
    if missing := [n for n in CONFIGURATIONS if n not in out]:
        raise ValueError(f"the serve_tax card gives no tax for {missing}: each configuration's line starts at its own")
    return out


def _size_class(bits: float) -> int:
    if (w := unit_class(ceil(bits))) is None:
        raise ValueError(f"a proof block of {bits} bits fits no public size class")
    return w


def _hazard(q: float, y: float) -> float:
    """h(y) = −ln(1 − q(1 − e^{−y})): a replay block's share of Λ once its incorrect blocks have rate y."""
    return -log1p(-q * -expm1(-y)) if y < inf else -log1p(-q) if q < 1 else inf


def write_form(cells: list[Cell], row: dict, replay_blocks: float, delta: float, footprint: float) -> float:
    """At least the most bits the incorrect proof blocks of an admissible set write under `row`'s two-stage design
    (`optimize.two_stage_design`'s: its `merged` size class and `k` by stratum, its `q`), each replay block charged at
    most `footprint` and its own blocks' bits.  An admissible set has Σ_ρ h(y_ρ) < Λ = ln(1/δ), y_ρ = Σ k_t/n_t over its
    blocks in ρ (`two_stage_exfil`'s acceptance), so the charge is at most Λ times the most a replay block charges a
    unit of h, and at most R·F.  A replay block charged c has y at least the least k_t/n_t of a block of at least c bits,
    or c times the least k_t/(n_t b) of a narrower one; c/h(y) only rises between the cells' bits, so its most is at one
    of them or at F."""
    split = [replace(c, stratum=0 if _size_class(c.bits) <= row["merged"] else _size_class(c.bits)) for c in cells]
    n: dict[int, float] = {}
    for c in split:
        n[c.stratum] = n.get(c.stratum, 0.0) + c.n
    rates = [(c.bits, inf if row["k"][c.stratum] >= n[c.stratum] else row["k"][c.stratum] / n[c.stratum])
             for c in split]
    top = min(footprint, sum(c.n * c.bits for c in split))
    best = 0.0
    for charge in {b for b, _ in rates if b < top} | {top}:
        wide = min((r for b, r in rates if b >= charge), default=inf)
        narrow = charge * min((r / b for b, r in rates if b < charge), default=inf)
        best = max(best, charge / _hazard(row["q"], min(wide, narrow)))
    return min(replay_blocks * top, log(1 / delta) * best)


@dataclass(frozen=True)
class Grain:
    """A two-stage configuration's replay blocks on one scope: their count, cells, replay cost and footprint."""

    replay_blocks: float
    cells: tuple[Cell, ...]
    replay: float
    footprint: float


def grain(m: f1.Model, name: str, layers: float, boundary_bits: float, tokens: float) -> Grain:
    if name == "tokens":
        return Grain(m.replay.replay_blocks, m.cells, m.replay.cost, m.served)
    per = tokens * layers
    return Grain(round(m.scope.requests * per), tuple(replace(c, n=c.n / per) for c in m.cells), m.replay.cost / per,
                 boundary_bits)


def _plan(row: dict | None) -> dict | None:
    return None if row is None else f1._plan(row)


def _point(job: tuple) -> dict:
    """One configuration's bounds at one budget, in GPU-seconds: each panel's, with the plan and, for the capped
    exfiltration, which of the count and the served bits binds."""
    name, m, g, budget = job
    served = f1.served_total(m)
    if name == "every_block":
        leak, work = f1.point(m, "one-stage", "leak", budget), f1.point(m, "one-stage", "work", budget)
        total_bits = m.scope.requests * sum(c.n * c.bits for c in m.cells)
        written = total_bits if leak["row"] is None else min(m.dot.out_bits * leak_count(m.dot, leak["bound"]),
                                                             total_bits)
        return {"exfiltration": {"y": min(leak["bound"], served), "count": leak["bound"],
                                 "binds": "served" if served <= leak["bound"] else "count", "plan": _plan(leak["row"])},
                "infiltration": {"y": written, "plan": _plan(leak["row"])},
                "skipped_work": {"y": work["bound"], "binds": "trivial" if work["capped"] else "count",
                                 "plan": _plan(work["row"])}}
    row = two_stage_design(g.cells, g.replay_blocks, m.delta, budget=budget, replay=g.replay, h=m.byte_s,
                           served=m.served)
    return {"exfiltration": {"y": min(row["bound"], served), "count": row["bound"],
                             "binds": "served" if served <= row["bound"] else "count", "plan": _plan(row)},
            "infiltration": {"y": write_form(list(g.cells), row, g.replay_blocks, m.delta, g.footprint),
                             "plan": _plan(row)}}


def _share(v: float) -> str:
    """A share of serving compute as the captions write it: a multiple from 1, a percentage from 0.01%."""
    return f"{v:.2g}×" if v >= 1 else f"{v * 100:.2g}%" if v >= 1e-4 else f"{v:.1e}"


def _binds(betas: list[float], binds: list[str]) -> str:
    """Where the served bits bind on a line, as the caption says it."""
    at = [b for b, w in zip(betas, binds) if w == "served"]
    if not at:
        return "the count binds at every β plotted"
    if len(at) == len(betas):
        return "the served bits bind at every β plotted"
    return f"the served bits bind up to β = {_share(max(at))}, the count above"


def captions(out: dict, scope: str) -> dict[str, str]:
    """Each panel's caption for `scope`: each configuration's line, its tax and what its bound rests on."""
    tax, sc = out["taxes"], out["scopes"][scope]
    lines = {p: {s["config"]: s for s in out["panels"][p]["series"] if s["scope"] == scope} for p in PANELS}

    def priced(name: str) -> str:
        t = tax[name]
        kind = "a lower bound, the hashing alone" if t["lower_bound"] else t["status"]
        parts = ", ".join(f"{p['part']} {_share(p['share'])}" for p in t["parts"])
        return f"serve tax {_share(t['tax'])} ({_share(t['low'])} to {_share(t['high'])}; {kind}: {parts})"

    head = ("x is the serve tax, what each configuration's change to the production server costs on every token, plus "
            "β, the share of compute spent on replays and proofs of the sampled proof blocks; each line starts at its "
            f"tax, hollow where the tax is priced. At the measured marginal proof overhead, {out['overhead']:.3g}×. "
            "Every β has its own plan, the one with the least bound.")
    a, b, c = CONFIGURATIONS["tokens"], CONFIGURATIONS["layer_of_token"], CONFIGURATIONS["every_block"]
    exfil = (f" {a.label}: serving commits {a.commits}, a replay block is {a.replay_block}, {priced('tokens')}; the "
             f"capped sets' form (card_influenceTwo_le_served_min) at most the {sc['served_bits']:.2g} bits the scope "
             f"serves (card_influenceTwo_le_min): {_binds(out['betas'], lines['exfiltration']['tokens']['binds'])}."
             f" {b.label}: serving commits {b.commits}, a replay block is {b.replay_block}, "
             f"{priced('layer_of_token')}; each replay block charged at most its request's served bits, at least the "
             f"lemma's charge: {_binds(out['betas'], lines['exfiltration']['layer_of_token']['binds'])}."
             f" {c.label} (dashed): serving commits {c.commits}, sampled one-stage, {priced('every_block')}; "
             "consumers.exfiltration_bound's form, which bits are incorrect and where, at most the served bits by the "
             "same count (an influence set is a set of served outputs; Lean states it for two-stage only): "
             f"{_binds(out['betas'], lines['exfiltration']['every_block']['binds'])}.")
    infil = (f" The write form: the bits the incorrect proof blocks write, each replay block charged at most its "
             f"footprint, at the design its exfiltration line plans. {a.label}: a footprint of a request's "
             f"{out['served_bits_request']:,.0f} served bits, the cheapest explanation of what it serves (the served "
             f"form). {b.label}: a footprint of its {out['footprint_bits'][1]:,.0f} boundary bits, written wherever "
             f"they sit. {c.label} (dashed): the K incorrect blocks the plan detects, each the widest block's bits. "
             "The two-stage write forms need a harm bound at the footprint, which no lemma states yet; one-stage's "
             "rests on audit_profile.")
    work = (f" Only {c.label} bounds skipped work: serving commits {c.commits}, so a sampled block's inputs and "
            f"outputs were fixed at serve time; {priced('every_block')}; the work target's plan.")
    return {"exfiltration": head + exfil, "infiltration": head + infil, "skipped_work": head + work}


def run(model_name: str = "GLM-5.3", scope_names: tuple[str, ...] | None = None, request: str = "census",
        workers: int = 1) -> dict:
    """Figure 1's JSON in its total-cost form over the scopes named (default all), planned on `workers` processes."""
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        return _run(model_name, scope_names, request, pool.map if pool else map)


def _run(model_name: str, scope_names: tuple[str, ...] | None, request: str, each) -> dict:
    sc = f1.slot_cards(model_name)
    c = f1.cards(model_name, request)
    tax = taxes(spot_cards.load("serve_tax", model_name))
    part = f1.TokenBlocks.of(c["partition"].value["strata"], c["partition"].value["served_macs_per_token"])
    fill = f1.Fill.of(c["partition"].value["strata"], c["proof_blocks"].value)
    h = c["proof_overhead"].value
    betas = sorted(set(BETAS) | set(c["budgets"].value))
    inf_card = sc["infiltration"]["value"]
    layers, boundary = inf_card["replay_blocks_per_token"], inf_card["cap_bits_per_block"]
    tokens = c["tokens_per_request"].value
    refs = f1.references(c, model_name, sc.get("references", []))
    notes = f1.reference_notes(sc.get("references", []), refs)
    chosen = [s for s in f1.scopes(c, part) if scope_names is None or s.name in scope_names]
    series: dict[str, list[dict]] = {p: [] for p in PANELS}
    out_scopes = {}
    for s in chosen:
        m = f1.model(c, part, s, h, fill)
        for name, conf in CONFIGURATIONS.items():
            g = None if name == "every_block" else grain(m, name, layers, boundary, tokens)
            pts = list(each(_point, [(name, m, g, beta * s.gpu_seconds) for beta in betas]))
            for panel in conf.panels:
                rows = [p[panel] for p in pts]
                series[panel].append({"scope": s.name, "config": name, "beta": betas,
                                      "x": [tax[name]["tax"] + b for b in betas], "y": [r["y"] for r in rows],
                                      **({"binds": [r["binds"] for r in rows]} if "binds" in rows[0] else {}),
                                      **({"count": [r["count"] for r in rows]} if "count" in rows[0] else {}),
                                      "plan": [r["plan"] for r in rows]})
        out_scopes[s.name] = {"who": s.who, "period": s.period, "gpus": c[f"{s.who}_gpus"].value,
                              "gpu_seconds": s.gpu_seconds, "requests": s.requests,
                              "tokens": s.requests * tokens, "served_bits": f1.served_total(m),
                              "output_bits": f1.cap(m, "leak"), "flop": f1.cap(m, "work"),
                              "ands": f1.cap(m, "work") * c["ands_per_mac"].value / 2}
    out = {"schema": SCHEMA, "model": model_name, "request": request, "overhead": h,
           "overhead_label": sc["proof_overhead"]["value"]["label"], "betas": betas, "budgets": c["budgets"].value,
           "served_bits_request": c["served_bits"].value, "footprint_bits": [c["served_bits"].value, boundary],
           "configurations": {n: conf.__dict__ | {"panels": list(conf.panels)} for n, conf in CONFIGURATIONS.items()},
           "taxes": tax, "inputs": {k: v.__dict__ for k, v in c.items()}, "cards": sc, "scopes": out_scopes,
           "panels": {"exfiltration": {"unit": "bit", "series": series["exfiltration"],
                                       "references": [r for r in refs["exfiltration"] if r["main"]]},
                      "infiltration": {"unit": "bit", "series": series["infiltration"],
                                       "references": refs["infiltration"]},
                      "skipped_work": {"unit": "FLOP", "series": series["skipped_work"],
                                       "references": refs["skipped_work"]}}}
    for name in out_scopes:
        caps = captions(out, name)
        for panel, sentences in notes.items():
            if panel in caps:
                caps[panel] += "".join(f" {x}" for x in sentences)
        out_scopes[name]["captions"] = caps
    return out


def at_budgets(out: dict) -> list[dict]:
    """Each line's point at the design card's budgets: the bound, its x, and its share of the scope's served bits
    (exfiltration, infiltration) or FLOP (skipped work)."""
    rows = []
    for panel, p in out["panels"].items():
        for s in p["series"]:
            sc = out["scopes"][s["scope"]]
            for beta in out["budgets"]:
                i = s["beta"].index(beta)
                whole = sc["flop"] if panel == "skipped_work" else sc["served_bits"]
                rows.append({"panel": panel, "scope": s["scope"], "config": s["config"], "beta": beta, "x": s["x"][i],
                             "bound": s["y"][i], "share": s["y"][i] / whole,
                             **({"binds": s["binds"][i]} if "binds" in s else {})})
    return rows


PRIMARY = "world year"
#: each configuration's colour, line style and width; (C) dashed, its tax a lower bound
STYLE = {"tokens": ("#08306B", "-", 3.2), "layer_of_token": ("#D95F02", "-", 2.4),
         "every_block": ("#555555", (0, (6, 3)), 2.4)}
YLABELS = {"exfiltration": "Exfiltration capacity", "infiltration": "Infiltration, bits written",
           "skipped_work": "Skipped work (H100-hours)"}


def stem(scope: str, panel: str = "exfiltration") -> str:
    base = "figure1_skipped_work" if panel == "skipped_work" else "figure1"
    return base if scope == PRIMARY else f"{base}_{scope.replace(' ', '_')}"


def _percent(v: float, _pos=None) -> str:
    return f"{v * 100:g}%"


def _panel(fig, ax, out: dict, scope: str, panel: str, wrap: int) -> None:
    """One panel of `scope`'s figure: each configuration's line from its tax, hollow at its start while the tax is
    priced, with the tax's range as a bar, the references grey, and the panel's caption under it."""
    import textwrap

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from render import CAPTION_DROP, OFF_SCALE, REFERENCE_STYLE, _bytes, _decades, _lines, _power

    p, tax = out["panels"][panel], out["taxes"]
    series = [s for s in p["series"] if s["scope"] == scope]
    scale = 1 / 8 if p["unit"] == "bit" else 1 / out["inputs"]["h100_hour_flop"]["value"]
    fmt = _bytes if p["unit"] == "bit" else _power
    xs = [x for s in series for x in s["x"]] + [tax[s["config"]][k] for s in series for k in ("low", "high")]
    lo_x, hi_x = 10.0 ** floor(log10(min(xs))), max(xs) * 1.3
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.xaxis.set_major_locator(FixedLocator(_decades(lo_x, hi_x, 1)))
    ax.xaxis.set_major_formatter(FuncFormatter(_percent))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Total cost, a share of serving compute: serve tax + β", labelpad=8)
    ax.set_ylabel(YLABELS[panel], labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    refs = _lines(p["references"], scope, scale)
    if panel == "exfiltration":
        refs = sorted(refs + [(out["scopes"][scope]["served_bits"] * scale, "Everything the scope serves", "dotted")])
    ys = [y * scale for s in series for y in s["y"]]
    far = [r for r in refs if r[0] > max(ys) * OFF_SCALE]
    refs = [r for r in refs if r not in far]
    lo, hi = 10.0 ** floor(log10(min(ys + [r[0] for r in refs]))), 10.0 ** (floor(log10(max(ys + [r[0] for r in refs]))) + 1)
    caption = out["scopes"][scope]["captions"][panel]
    if far:
        caption += " Above the axis: " + "; ".join(f"{label} ({fmt(a)})" for a, label, _ in far) + "."
    ax.text(0, -CAPTION_DROP, textwrap.fill(caption, wrap), transform=ax.transAxes, ha="left", va="top", fontsize=14,
            color="#444444")
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(_decades(lo, hi, 3 if hi / lo > 1e12 else 2)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", color="#E2E2E2", lw=.45)
    styles = REFERENCE_STYLE | {"dotted": (0, (1.2, 2.4))}
    for amount, label, style in refs:
        ax.axhline(amount, color="#909090", ls=styles[style], lw=1.3, zorder=1)
        ax.text(hi_x / 1.3, amount * 1.08, label, ha="right", va="bottom", fontsize=14, color="#555555", zorder=5,
                bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": .5})
    for s in series:
        color, ls, lw = STYLE[s["config"]]
        t = tax[s["config"]]
        y0 = s["y"][0] * scale
        ax.plot(s["x"], [y * scale for y in s["y"]], color=color, ls=ls, lw=lw, zorder=3)
        ax.plot([t["low"], t["high"]], [y0, y0], color=color, lw=1.4, zorder=3, marker="|", markersize=12)
        ax.plot([t["tax"]], [y0], marker="o", markersize=13, color=color, zorder=4, mew=2.2,
                markerfacecolor="white" if t["status"] == "priced" else color)
    configs = list(dict.fromkeys(s["config"] for s in series))
    handles = [plt_line(STYLE[n]) for n in configs]
    labels = [out["configurations"][n]["label"] + (" (priced)" if n == "every_block" else "") for n in configs]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(configs), frameon=False,
              fontsize=15, handlelength=2.6, columnspacing=1.4, borderaxespad=0)


def plt_line(style: tuple):
    import matplotlib.pyplot as plt

    color, ls, lw = style
    return plt.Line2D([], [], color=color, ls=ls, lw=lw)


def render(out: dict, scope: str, outdir: Path) -> list[Path]:
    """Draw `scope`'s Figure 1 (exfiltration and infiltration) and its skipped-work figure from `out` into `outdir`,
    as `stem(scope, panel)`.png and .svg; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render import _save, _sci

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 20, "axes.labelsize": 21, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "savefig.facecolor": "white", "figure.facecolor": "white"})
    info = out["scopes"][scope]
    where = "the world's compute" if info["who"] == "world" else "a datacenter's compute"
    paths = []
    for panels, width in ((("exfiltration", "infiltration"), 24), (("skipped_work",), 13)):
        fig, axes = plt.subplots(1, len(panels), figsize=(width, 12), squeeze=False)
        left = .06 if len(panels) > 1 else .11
        fig.subplots_adjust(left=left, right=.99, bottom=.42, top=.88, wspace=.28)
        fig.text(left, .975, f"{out['model']} served on all of {where} for a {info['period']}: {_sci(info['gpus'])} "
                 f"H200s, {_sci(info['requests'])} requests. Hollow: a priced serve tax; the bar, its range.",
                 ha="left", va="center", fontsize=21)
        for ax, panel in zip(axes[0], panels):
            _panel(fig, ax, out, scope, panel, 108 if len(panels) > 1 else 110)
        paths += _save(fig, outdir, stem(scope, panels[0]))
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("priced-card", help="write the model's priced serve_tax card")
    pc.add_argument("--model", choices=spot_cards.MODELS, default=spot_cards.MODELS[0])
    pc.add_argument("--out", type=Path, required=True)
    rd = sub.add_parser("render", help="plot Figure 1 from its JSON")
    rd.add_argument("json", type=Path)
    rd.add_argument("--scope", action="append", help="a scope's name, as the JSON's `scopes` has it; default all")
    rd.add_argument("--out", type=Path, help="default: the JSON's directory")
    args = ap.parse_args(argv)
    if args.cmd == "priced-card":
        args.out.write_text(json.dumps(priced_card(args.model), indent=1) + "\n")
        print(args.out)
        return
    out = json.loads(args.json.read_text())
    for scope in args.scope or out["scopes"]:
        for path in render(out, scope, args.out or args.json.parent):
            print(path)


if __name__ == "__main__":
    main()
