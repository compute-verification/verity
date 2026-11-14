"""Dither's skippable share of honest work as served, per model, context, phase and design (Derived).

    uv run python benchmarks/pouw/eps_budget.py [--json OUT] [--md OUT]

ε_skip and γ are `verity_machine_model.skip`'s (benchmarks/machine_model/), whose docstring has the definition, the two
weightings, the rates and the two pieces of served-only work it does not price. This file prints them as tables, and
with `--json` writes every cell:

- each design's ε_skip (`design_table`): `design_skip`, the most over the models, contexts and phases under the floor
  credit, which each registry's `[skip]` records (test_eps_budget holds them equal), on sm_120 and on every device of
  the machine model (`device_table`);
- ε_skip by int8 products and by time, with γ, per model, context and phase (`skip_table`) and per component
  (`component_table`);
- P·V's block size, for the designs whose P·V is `pv` (`pv_block_point`, `pv_block_table`);
- what decode's zero instance saves on DRAM (`decode_dram`). Decode is memory-bound and the cheater's transcript is not
  (B̃ = S is generated from U, and the zero instance streams no weights), so decode's time ratio is a bandwidth gap no
  noise design closes; it is reported, not optimized.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from verity_machine_model.eps import PRODUCTS, Design, design, designs
from verity_machine_model.model import DATA, device, devices
from verity_machine_model.skip import (
    CONTEXTS, FLOOR_WORDS, MODELS, PHASES, RATES, TS_CHEAT, Shape, Summary, decode_step, design_skip, floor_eps,
    floor_skip, parts, prices, products, pv_ts, summary,
)

#: what `grind_price.py`, `proved_ticks.py` and the tests read
__all__ = [
    "CONTEXTS", "FLOOR_WORDS", "MODELS", "PHASES", "RATES", "Shape", "TS_CHEAT", "decode_dram", "design_skip",
    "floor_eps", "floor_skip", "main", "parts", "prices", "products", "pv_block_point", "pv_ts", "summary",
]


def pv_block_point(name: str, ctx: int, d: Design, block: int, rates: dict = RATES) -> dict:
    """Prefill's whole-model ε_skip and γ for design `d` (its P V `pv`) with P in blocks of `block` keys, beside P V's
    own ε_skip and γ (`pv_ts`) and its share of honest time."""
    ps = parts(name, ctx, "prefill", d.noise, rates, block)
    pv = next(p for p in ps if p.component == "P·V")
    secs, skip = sum(p.seconds for p in ps), sum(p.seconds_skip for p in ps)
    return {"model": name, "ctx": ctx, "design": d.name, "block": block,
            "eps_pv": pv.seconds_skip / (pv.seconds - pv.seconds_skip), "gamma_pv": pv.seconds / pv.plain - 1,
            "pv_share": pv.seconds / secs, "eps": skip / (secs - skip), "gamma": secs / sum(p.plain for p in ps) - 1}


PV_BLOCKS = (0, 256, 512, 1024)


def pv_designs() -> list[Design]:
    return [d for d in map(design, designs()) if d.noise["pv"] == "pv"]


def pv_block_points(rates: dict = RATES) -> list[dict]:
    ds = pv_designs()
    return [pv_block_point(m, c, d, b, rates) for m in MODELS for c in CONTEXTS for d in ds for b in PV_BLOCKS]


def pv_block_table(points: list[dict]) -> str:
    lines = ["P V's block size (pv, prefill, TS model at the duo fold, every served output free): P V's ε_skip and γ, "
             "its share of honest time, and the whole model's ε_skip and γ", "",
             "| model | context | design | P block | P V ε_skip | P V γ | P V share | whole ε_skip | whole γ |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for q in points:
        lines.append(f"| {q['model']} | {q['ctx'] // 1024}K | {q['design']} | {q['block'] or 'one'} | "
                     f"{100 * q['eps_pv']:.3g}% | {q['gamma_pv']:.3f} | {100 * q['pv_share']:.3g}% | "
                     f"{100 * q['eps']:.3g}% | {q['gamma']:.3f} |")
    return "\n".join(lines)


def decode_dram(name: str, ctx: int, rates: dict = RATES) -> float:
    """What the zero-weight, zero-KV instance (B = 0 with B̃ generated from the coins, K = V = 0) saves on decode's binding
    facet, DRAM, as a multiple of the step's noised products' compute: the plain step beyond them. Binding B to a
    registered weight matrix with every operand derived by C1′ leaves no zero instance, and this term goes to 0."""
    s, r = MODELS[name], rates
    gemm = ((sum(k * n for _, k, n in s.linears) + s.vocab * s.h / s.layers) / r["eta_dec"]
            + 2 * s.heads * s.d * ctx / r["eta_att"]) * r["r_receipt"] / r["peak"]
    return max(decode_step(name, ctx, rates)["step"] - gemm, 0) / gemm


def skip_table(ds: list[Design], rates: dict = RATES) -> str:
    head = "| model | context | phase | " + " | ".join(f"{d.name}: ε_skip MAC / time, γ" for d in ds) + " |"
    lines = [head, "|---|---:|---|" + "---:|" * len(ds)]
    for m in MODELS:
        for c in CONTEXTS:
            for ph in PHASES:
                cells = []
                for d in ds:
                    q = summary(m, c, ph, d, rates)
                    cells.append(f"{100 * q.eps_mac:.3g}% / {100 * q.eps_time:.3g}%, {q.gamma:.2f}")
                lines.append(f"| {m} | {c // 1024}K | {ph} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def component_table(name: str, ctx: int, phase: str, ds: list[Design], rates: dict = RATES) -> str:
    sums = {d.name: summary(name, ctx, phase, d, rates) for d in ds}
    lines = [(f"{name}, {ctx // 1024}K, {phase}: per component, ε_skip by int8 products / by time (each over the whole), "
              "component γ"), "", f"| component | MAC share ({ds[0].name}) | " + " | ".join(sums) + " |",
             "|---|---:|" + "---:|" * len(ds)]
    for i, first in enumerate(sums[ds[0].name].rows):
        cells = []
        for q in sums.values():
            row = q.rows[i]
            g = f", γ {row['gamma']:.2f}" if row["gamma"] is not None else ""
            cells.append(f"{100 * row['eps_mac']:.3g}% / {100 * row['eps_time']:.3g}%{g}")
        lines.append(f"| {first['component']} | {100 * first['mac_share']:.3g}% | " + " | ".join(cells) + " |")
    lines.append("| **whole** | 100% | " + " | ".join(
        f"**{100 * q.eps_mac:.3g}% / {100 * q.eps_time:.3g}%, γ {q.gamma:.2f}**" for q in sums.values()) + " |")
    return "\n".join(lines)


def design_table(skips: dict[str, Summary], ds: list[Design]) -> str:
    lines = ["Each design's ε_skip (`design_skip`, its registry `[skip]`): the most over the models, contexts and "
             "phases under the floor credit (Corollary D3)", "",
             "| design | noise (linears, lm_head, QKᵀ, P·V) | ε_skip | credit | at | γ there |",
             "|---|---|---:|---|---|---:|"]
    for d in ds:
        q = skips[d.name]
        noise = ", ".join(d.noise[p] for p in PRODUCTS)
        lines.append(f"| {d.name} ({d.title}) | {noise} | {100 * q.eps_time:#.3g}% | {q.credit} | {q.model} "
                     f"{q.ctx // 1024}K {q.phase} | {q.gamma:.2f} |")
    return "\n".join(lines)


def device_table(data: Path = DATA, vector: str = "", rates: dict = RATES) -> str:
    """Each of a device's designs under the floor credit: ♭'s price and prefill's ε_skip at every model and context,
    served free / served checked, beside `design_skip`'s number (decode included) and its cell."""
    ds = [design(n, data) for n in designs(data)]
    vec = prices(ds[0].name, data, vector).vector
    cells = [(m, c) for m in MODELS for c in CONTEXTS]
    lines = [f"{data.name if data != DATA else 'sm_120'} on {vec}: ε_skip under the floor credit, prefill, served free "
             "/ served checked", "",
             "| design | L | ♭ a MMA | " + " | ".join(f"{m.removeprefix('llama-3.1-')} {c // 1024}K" for m, c in cells)
             + " | ε_skip | at |", "|---|---:|---:|" + "---:|" * len(cells) + "---:|---|"]
    for d in ds:
        row = [f"{100 * floor_eps(m, c, 'prefill', d, rates, k, data, vector)[0]:.3f}" for m, c in cells
               for k in (False, True)]
        q = design_skip(d, rates, data, vector)
        at = f"{q.model.removeprefix('llama-3.1-')} {q.ctx // 1024}K {q.phase}"
        lines.append(f"| {d.name} | {d.L} | {prices(d.name, data, vector).mma:.3f} | "
                     + " | ".join(f"{a} / {b}" for a, b in zip(row[::2], row[1::2]))
                     + f" | {100 * q.eps_time:#.3g}% | {at} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json")
    p.add_argument("--md")
    a = p.parse_args(argv)
    ds = [design(n) for n in designs()]
    skips = {d.name: design_skip(d) for d in ds}
    pvs = pv_block_points()
    text = "\n".join([design_table(skips, ds), "", *(device_table(device(n)) + "\n" for n in devices()),
                      skip_table(ds), "",
                      *(component_table(m, c, ph, ds) + "\n" for m in MODELS for c in (2048, 32768) for ph in PHASES),
                      pv_block_table(pvs), "",
                      "Decode's zero-weight instance saves, on DRAM, this multiple of its transcripts' compute: "
                      + ", ".join(f"{m} {c // 1024}K {decode_dram(m, c):.2f}" for m in MODELS for c in CONTEXTS)])
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        sums = [summary(m, c, ph, d) for m in MODELS for c in CONTEXTS for ph in PHASES for d in ds]
        # v2: each "gamma" is the honest overhead, honest / plain − 1 (the unversioned output before it held the ratio)
        Path(a.json).write_text(json.dumps({"schema": "verity-pouw/eps-budget/v2", "rates": RATES, "designs": {
            d.name: {"noise": d.noise, **asdict(skips[d.name])} for d in ds}, "skip": [asdict(q) for q in sums],
            "pv_blocks": pvs, "decode_zero_weight_gap": {f"{m}/{c}": decode_dram(m, c) for m in MODELS
                                                         for c in CONTEXTS}}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
