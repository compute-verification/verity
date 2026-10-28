"""The served window's partition in the planner's two modes (PRF-298): Llama-3.2-1B int8 at T 8,192 on the H200, one step
a window (b921), cut into proof blocks at each rung of a ladder, for the free-compute target and the exfiltration target,
priced proving only (`PROVING_ONLY`, Lean's `GranularityOptimum`) and proving plus `w` times the serve-time commitment
(`n b(s) h`, paid on every proof block, drawn or not) at two readings of h, a hashed byte's cost.

A model, not a measurement: each input carries its status (M measured, D derived, A assumed) and source (`IN`).

**The ladder.**  Every rung covers the window's work; per token and layer:

- `chunk L`, `coord`, `tile RxC`: the int8 linears (qkv, o_proj, gate_up, down) as K-chunks of L MACs that commit their
  32-bit running sums, as output coordinates (the served draw's `ScaledMmS8Coordinate`), or as tiles of R tokens by C
  output channels, opening each operand row once; attention (a query head's row) and the row ops (one unit an output
  coordinate) stay at today's units.  The window commits what it commits today (every Call's outputs, `Q_word_v1`), plus
  the chunks' sums.
- `attn|mlp token`, `layer token`, `4 layers token`: fused proof blocks that commit only their boundary, the residual
  stream and the K and V a later token reads (COMMIT-BYTES' lever 13): every interior value stays inside a proof block.

A proof block costs its ANDs (int8 MACs at `BlackwellS8ImmaDot32_v1`'s 4,190 per 32, bf16 MACs at `F32Fma_v2`'s 4,498, the row
ops at their pieces, and every row it opens at hm96's `row_ands`) at the cost model fitted from C-Flock's zero-knowledge
sessions (`inputs.observations`, GPU-seconds of an RTX PRO 6000), with its statement's share when statements are full.
`fits` is false for a rung whose proof blocks exceed C-Flock's statement (m = k_log + nbl <= 35, `circuit.rs` `M_MAX`: at most
2^34 ANDs a proof block) or the widest size class (`UNIT_CLASS_MAX` bits): it prices what lifting the limit would buy.

**Two views.**  `lean`: each rung as n equal proof blocks of its mean size (`optimize.Grain`), its proving cost weighted by
work (free compute) or by width (exfiltration), through `grain_costs`, the laws of `Achievability` and
`ExfiltrationOptimum`.  `planner`: the rung's parts as strata through `for_target` under the density bound, its
statements shared (`CostModel.shared`), at `commit_weight` 0 or w.

**Targets.**  Free compute: C = `EPS` of the scope's MACs, at the window and pooled over a week and a year of windows at the
stock step; the draws are about Λ/ε whatever the scope, the commitment grows with it.  Exfiltration: ℓ = `LEAK_BITS` a
window.

    uv run python benchmarks/sampling_plan/served_window.py --out DIR [--weight W]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from math import ceil, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import EPILOGUE, I, KEY, S8_STEP, observations  # noqa: E402

from verity.core.protocols.sampled_proofs.costfit import Fitted, fit  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.partition import UNIT_CLASS_MAX  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (PROVING_ONLY, Candidate, CostModel, Grain,  # noqa: E402
                                                           Objective, for_target, grain_costs, leak_count, loc_bits,
                                                           row_ands, sqrt_grain)

DELTA = 2.0 ** -40
LAMBDA = log(1 / DELTA)
EPS = 0.01
LEAK_BITS = 2.0 ** 30
#: C-Flock's largest statement, m = k_log + nbl <= 35 (`verity/ml/flock/live/src/circuit.rs` `M_MAX`)
M_MAX = 35
#: the pieces' ANDs (their circuit-check pins: `verity/ml/circuits/ops/*.circuit-check.json`)
F32_FMA, F32_MUL, F32_ADD, F32_MAX, EX2, RCP, F2BF16, F32_DIV, S8_CLAMP = 4498, 2406, 823, 283, 688, 197, 77, 2493, 71
F32_RINT = 275


@dataclass(frozen=True)
class Input:
    value: float
    status: str
    source: str


IN = {
    "tokens": Input(8192, "M", "T 8,192 a step: 8 sequences of 1,024-id prompts, one step a window (b921, "
                    "note:compute-research/20261028T1600Z-finding-commit-bytes)"),
    "sequences": Input(8, "M", "b921's 8 sequences a step"),
    "committed": Input(13_628_428_480, "M", "bytes b921 committed in every full window (Q_word_v1, every Call's outputs; "
                       "art:291c27d31367bb75b2035d35f34857cb84e19095c873cd1c57eda5ecb61bf526)"),
    "hash_fast": Input(521e9, "M", "SHA-512 rows on the whole H200, 16 KiB rows (r20261027-042422-85b2)"),
    "hash_slow": Input(398e9, "D", "the same at b921's 4 KiB rows: 85b2's bar x 91.5 B a compression (COMMIT-BYTES)"),
    "peak_macs": Input(989.5e12, "A", "H200 int8 dense, 4,096 MACs an SM-clock x 132 SMs at 1,830 MHz (the data sheet's "
                       "1,979 TOPS), to state h in MAC-times"),
    "step_s": Input(0.02614, "M", "the stock T 8,192 step, b57f (r20261028-112958-eeb0), for a week of windows"),
    "price": Input(1.0, "A", "a GPU-second proving (RTX PRO 6000, where the cost model was measured) priced as a "
                   "GPU-second hashing on the H200; `--weight` scales the commitment against it"),
}
LLAMA_1B = {"layers": 16, "hidden": 2048, "heads": 32, "kv_heads": 8, "head_dim": 64, "intermediate": 8192}
#: (K, N) a token: qkv, o_proj, gate_up, down
LINEARS = {"qkv": (2048, 3072), "o_proj": (2048, 2048), "gate_up": (2048, 16384), "down": (8192, 2048)}
CHUNKS = (32, 64, 128, 256, 512, 1024)
TILES = ((2, 2), (4, 4), (16, 16), (64, 64), (128, 128))
WEEK_S = 7 * 86400


@dataclass(frozen=True)
class Part:
    """One stratum of a rung: `units` proof blocks a window of `work` MACs each, `out_bits` wide, committing `commit_bytes`
    at serve time, of `ands` ANDs, the rows they open included."""

    name: str
    units: int
    work: float
    out_bits: float
    commit_bytes: float
    ands: float

    @property
    def fits(self) -> bool:
        return k_log(self.ands) <= M_MAX and self.out_bits <= UNIT_CLASS_MAX


def k_log(ands: float) -> int:
    return ceil(log2(max(ands, 2))) + 1


def cap(ands: float) -> int:
    """A statement's proof blocks under m <= 35 (`int8_llama.cap`), at least 1."""
    return 1 << max(0, M_MAX - k_log(ands))


def token_layers() -> int:
    return int(IN["tokens"].value) * LLAMA_1B["layers"]


def mean_keys() -> float:
    """A prompt position's keys, uniform on 1..1,024."""
    per = IN["tokens"].value / IN["sequences"].value
    return (per + 1) / 2


def attention_row() -> tuple[float, float]:
    """One query head's attention at the mean position: its bf16 MACs (QK^T and PV) and ANDs, softmax and the q row, its
    K and V, and its output row opened."""
    hd, keys = LLAMA_1B["head_dim"], mean_keys()
    macs = 2 * hd * keys
    ands = macs * F32_FMA + keys * (EX2 + F32_ADD + F32_MUL + F2BF16 + F32_MAX) + RCP \
        + 2 * row_ands(round(2 * hd * keys)) + 2 * row_ands(2 * hd)
    return macs, ands


def row_op_bytes() -> float:
    """A token-layer's committed bytes outside the linears' and attention's outputs: today's stream less them."""
    lin = sum(2 * n for _, n in LINEARS.values())
    attn = 2 * LLAMA_1B["heads"] * LLAMA_1B["head_dim"]
    return IN["committed"].value / token_layers() - lin - attn


#: one row-op proof block (a norm, quantizer, SiLU or RoPE coordinate): about 6,000 ANDs of f32 pieces and three one-block rows
ROW_OP_ANDS = 2 * F32_MUL + F32_DIV + F32_RINT + S8_CLAMP + F2BF16 + 3 * row_ands(4)


def rest_parts() -> list[Part]:
    """Attention (a query head's row) and the row ops (one 16-bit coordinate each), at today's units."""
    tl = token_layers()
    macs, ands = attention_row()
    heads = LLAMA_1B["heads"]
    rows = row_op_bytes() / 2
    return [Part("attention", tl * heads, macs, 16 * LLAMA_1B["head_dim"], 2 * LLAMA_1B["head_dim"], ands),
            Part("row ops", round(tl * rows), 0.0, 16, 2, ROW_OP_ANDS)]


def coord_ands(k: int) -> float:
    return k // 32 * S8_STEP + EPILOGUE + 2 * row_ands(k) + 2 * row_ands(4) + row_ands(2)


def linear_parts(rung: str) -> list[Part]:
    tl, out = token_layers(), []
    for op, (k, n) in LINEARS.items():
        if rung == "coord":
            out.append(Part(op, tl * n, k, 16, 2, coord_ands(k)))
        elif rung.startswith("chunk"):
            c = min(int(rung.split()[1]), k)
            last = c / k
            bits = 32 * (1 - last) + 16 * last
            ands = c // 32 * S8_STEP + 2 * row_ands(c) + 2 * row_ands(4) \
                + last * (EPILOGUE + 2 * row_ands(4) + row_ands(2))
            out.append(Part(op, tl * n * (k // c), c, bits, bits / 8, ands))
        else:
            r, c = map(int, rung.split()[1].split("x"))
            cells = r * c
            ands = cells * (k // 32 * S8_STEP + EPILOGUE) + (r + c) * row_ands(k) + row_ands(4 * r) + row_ands(4 * c) \
                + row_ands(2 * cells)
            out.append(Part(op, tl * n // cells, cells * k, 16 * cells, 2 * cells, ands))
    return out


def _fused(name: str, units: int, int8: float, bf16: float, opened: float, out_bytes: float, row_ops: float) -> Part:
    """A fused proof block: its int8 and bf16 MACs, its row ops' pieces, and the bytes it opens (weights and their scales,
    the K and V it reads, its input and output) at hm96's per-byte rate."""
    per_byte = (row_ands(1 << 20) - row_ands(0)) / (1 << 20)
    ands = int8 * S8_STEP / 32 + bf16 * F32_FMA + row_ops + opened * per_byte
    return Part(name, units, int8 + bf16, 8 * out_bytes, out_bytes, ands)


def fused_parts(rung: str) -> list[Part]:
    """Proof blocks that commit only their boundary: the residual stream (bf16), and the K and V a later token reads."""
    tl, d, f = token_layers(), LLAMA_1B["hidden"], LLAMA_1B["intermediate"]
    kv = 2 * 2 * LLAMA_1B["kv_heads"] * LLAMA_1B["head_dim"]
    macs, attn_ands = attention_row()
    heads = LLAMA_1B["heads"]
    bf16 = heads * macs
    softmax = heads * (attn_ands - macs * F32_FMA - 2 * row_ands(round(macs)) - 2 * row_ands(2 * LLAMA_1B["head_dim"]))
    row_bytes = row_op_bytes()
    row_ops = row_bytes / 2 * (ROW_OP_ANDS - 3 * row_ands(4))
    attn_w = sum(k * n for op, (k, n) in LINEARS.items() if op in ("qkv", "o_proj"))
    mlp_w = sum(k * n for op, (k, n) in LINEARS.items() if op in ("gate_up", "down"))
    scales = 4 * sum(n for _, n in LINEARS.values())
    keys = kv * mean_keys()
    if rung == "attn|mlp token":
        share = (2 * d + 2 * 2560) / (row_bytes or 1)
        return [_fused("attention block", tl, attn_w, bf16, attn_w + keys + 2 * d + 2 * d + kv, 2 * d + kv,
                       softmax + share * row_ops),
                _fused("MLP block", tl, mlp_w, 0.0, mlp_w + 4 * d, 2 * d, (1 - share) * row_ops)]
    span = 1 if rung == "layer token" else 4
    w = span * (attn_w + mlp_w + scales)
    out = 2 * d + span * kv
    return [_fused(f"{span} layer{'s' if span > 1 else ''}", tl // span, span * (attn_w + mlp_w), span * bf16,
                   w + span * keys + 2 * d + out, out, span * (softmax + row_ops))]


def rungs() -> dict[str, list[Part]]:
    out = {f"chunk {c}": linear_parts(f"chunk {c}") + rest_parts() for c in CHUNKS}
    out["coord"] = linear_parts("coord") + rest_parts()
    out |= {f"tile {r}x{c}": linear_parts(f"tile {r}x{c}") + rest_parts() for r, c in TILES}
    out |= {name: fused_parts(name) for name in ("attn|mlp token", "layer token", "4 layers token")}
    return out


def fitted() -> Fitted:
    obs = observations()
    return Fitted(fit(obs, "prove", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "verify", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "stage", ["one", "ands", "ands*staged"]), I["cpu"].value, KEY)


def unit_cost(m: CostModel, ands: float, marginal: bool = False) -> float:
    """A drawn proof block's GPU-seconds in a full statement: proved and staged, and 1/cap of the statement.  The fit's
    per-statement term grows with the subcircuit's ANDs (3.6 us an AND, fitted on 5M- and 15M-AND subcircuits), which a
    statement of one 2^34-AND proof block pays whole; `marginal` leaves it out, the reading where that term is fixed."""
    statement = m.statement_fixed + (0.0 if marginal else m.statement_per_and * ands)
    return ands * (m.per_and + m.staged_per_and) + statement / cap(ands)


def scale(parts: list[Part], windows: float) -> list[Part]:
    return [replace(p, units=round(p.units * windows)) for p in parts]


def grain(name: str, parts: list[Part], m: CostModel, by: str, marginal: bool = False) -> Grain:
    """The rung as n equal proof blocks of its mean work and width, its proving cost weighted by work (`by` "work") or by
    width ("bits"): the cost the laws pay, which draw a proof block in proportion to it."""
    n = sum(p.units for p in parts)
    weight = {p.name: p.units * (p.work if by == "work" else p.out_bits) for p in parts}
    total = sum(weight.values())
    prove = sum(weight[p.name] * unit_cost(m, p.ands, marginal) for p in parts) / total
    return Grain(name, n, sum(p.units * p.work for p in parts) / n, sum(p.units * p.out_bits for p in parts) / n,
                 sum(p.units * p.commit_bytes for p in parts) / n, prove)


def candidates(rung: str, parts: list[Part]) -> list[Candidate]:
    return [Candidate(p.name, f"{rung}:{p.name}", p.units, round(p.work), round(p.out_bits), ands=round(p.ands),
                      commit_bits=round(8 * p.commit_bytes), cap=cap(p.ands)) for p in parts]


def planner(rung: str, parts: list[Part], m: CostModel, *, h: float, weight: float, work: float | None = None,
            leak: float | None = None) -> dict | None:
    costs = replace(m, shared=True, commit_per_bit=h / 8, commit_weight=weight)
    cands = candidates(rung, parts)
    if work is not None:
        p = for_target(cands, costs, Objective(DELTA), work, exact=False, grid=41)
    else:
        n, narrow = sum(c.units for c in cands), min(c.out_bits for c in cands if c.out_bits)
        allowance = loc_bits(n, leak_count(Grain("narrowest", n, 1.0, narrow, 0.0, 0.0), leak))
        p = for_target(cands, costs, Objective(DELTA, work=0.0, bits=1.0), leak - allowance, exact=False, grid=41)
    if p is None:
        return None
    return {"total": p.cost["total"], "prove": p.cost["statements"] + p.cost["units"],
            "commit": p.cost["commit"], "draws": sum(p.sizes.values())}


def unit_work(parts: list[Part]) -> float:
    """A rung's proof-block size as the table quotes it: an int8 linear's proof block (gate_up's, most of the work), else the
    fused block's largest."""
    return next((p.work for p in parts if p.name == "gate_up"), max(p.work for p in parts))


def readings() -> dict[str, float]:
    return {"h 521 GB/s": 1 / IN["hash_fast"].value, "h 398 GB/s": 1 / IN["hash_slow"].value}


def scopes() -> dict[str, float]:
    return {"window": 1.0, "week": WEEK_S / IN["step_s"].value, "year": 365 * 86400 / IN["step_s"].value}


def run(weight: float = 1.0, *, plan: bool = True) -> dict:
    m = fitted().model()
    ladder = rungs()
    work_window = sum(p.units * p.work for p in ladder["coord"])
    out = {"inputs": {k: v.__dict__ for k, v in IN.items()}, "model": LLAMA_1B, "delta": DELTA, "eps": EPS,
           "leak_bits": LEAK_BITS, "weight": weight, "work_window": work_window,
           "rates": {"per_and": m.per_and, "staged_per_and": m.staged_per_and, "statement_fixed": m.statement_fixed,
                     "statement_per_and": m.statement_per_and},
           "h_mac_times": {k: h * IN["peak_macs"].value for k, h in readings().items()},
           "rungs": {name: {"fits": all(p.fits for p in parts),
                            "parts": [{**p.__dict__, "k_log": k_log(p.ands), "unit_s": unit_cost(m, p.ands)}
                                      for p in parts],
                            "units": sum(p.units for p in parts),
                            "committed": sum(p.units * p.commit_bytes for p in parts),
                            "unit_work": unit_work(parts)}
                     for name, parts in ladder.items()},
           "results": []}
    modes = {"proving only": PROVING_ONLY, f"proving + {weight:g} x commit": weight}
    cases = [(scope, "work", windows) for scope, windows in scopes().items()] + [("window", "leak", 1.0)]
    for scope, target, windows in cases:
        for hname, h in readings().items():
            for mode, w in modes.items():
                rows = []
                for name, parts in ladder.items():
                    parts_s = scale(parts, windows)
                    kw = {"work": EPS * work_window * windows} if target == "work" else {"leak": LEAK_BITS}
                    by = "work" if target == "work" else "bits"
                    g = grain(name, parts_s, m, by)
                    lean = grain_costs([g], DELTA, h=h, weight=w, **kw)[0]
                    marginal = grain_costs([grain(name, parts_s, m, by, marginal=True)], DELTA, h=h, weight=w, **kw)[0]
                    row = {"rung": name, "fits": out["rungs"][name]["fits"], "work": unit_work(parts),
                           "mean_work": g.work, "lean": lean, "lean_marginal": marginal}
                    if plan:
                        row["planner"] = planner(name, parts_s, m, h=h, weight=w, **kw)
                    rows.append(row)
                fit_rows = [r for r in rows if r["fits"]]
                out["results"].append({
                    "scope": scope, "target": target, "h": hname, "mode": mode, "weight": w, "rows": rows,
                    "best_lean": min(fit_rows, key=lambda r: r["lean"]["total"])["rung"],
                    "best_lean_any": min(rows, key=lambda r: r["lean"]["total"])["rung"],
                    "best_marginal_any": min(rows, key=lambda r: r["lean_marginal"]["total"])["rung"],
                    "best_planner": None if not plan else min(
                        (r for r in fit_rows if r["planner"]), key=lambda r: r["planner"]["total"])["rung"],
                    "serve_s": windows * IN["step_s"].value})
    out["sqrt_law"] = sqrt_law(m, weight)
    return out


def sqrt_law(m: CostModel, weight: float) -> list[dict]:
    """The work target's optimum on the chunk family alone, the int8 linears as K-chunks of L MACs, each committing a
    β = 4-byte running sum: the cheapest L over a fine ladder (any multiple of 32 up to the dot), against
    √(w β h C / (o Λ)), o the chunks' proving cost a MAC, as C grows by pooling windows."""
    tl = token_layers()
    work = tl * sum(k * n for k, n in LINEARS.values())
    o = (unit_cost(m, 2048 // 32 * S8_STEP + 2 * row_ands(2048)) - unit_cost(m, 1024 // 32 * S8_STEP + 2 * row_ands(1024))) \
        / 1024
    rows = []
    for hname, h in readings().items():
        for windows in (1, 16, 256, 4096, 65536):
            target = EPS * work * windows
            ladder = []
            for c in range(32, 8192 + 1, 32):
                ands = c // 32 * S8_STEP + 2 * row_ands(c) + 2 * row_ands(4)
                ladder.append(Grain(f"L={c}", round(work * windows / c), c, 32, 4.0, unit_cost(m, ands)))
            costs = grain_costs(ladder, DELTA, h=h, weight=weight, work=target)
            best = min(costs, key=lambda r: r["total"])
            rows.append({"h": hname, "windows": windows, "C": target, "o": o, "best_L": best["work"],
                         "sqrt_L": sqrt_grain(4.0, h, target, o, DELTA, weight),
                         "proving_only_L": min(grain_costs(ladder, DELTA, h=h, weight=PROVING_ONLY, work=target),
                                               key=lambda r: r["total"])["work"]})
    return rows


def _g(x: float) -> str:
    return f"{x:.3g}"


def render(out: dict) -> str:
    lines = [f"# Served window, Llama-3.2-1B int8 T 8,192: proof-block size in two modes (δ = 2^-40, ε = {EPS:.0%}, "
             f"ℓ = 2^{log2(LEAK_BITS):.0f} bits a window)", ""]
    lines.append("GPU-seconds (proving: RTX PRO 6000 ZK fit, modelled for these units; commitment: H200 SHA-512 at h). "
                 "'x serve' divides by the scope's stock serving time. s* is the cheapest rung that fits C-Flock "
                 "(m <= 35, size classes); 'any' includes the fused rungs that don't.")
    lines += ["", "| scope | target | h | mode | s* (lean) | s* (MACs a gate_up block) | total | prove | commit | x serve | "
              "s* (planner) | planner total | s* any, as fitted | s* any, marginal |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        best = next(x for x in r["rows"] if x["rung"] == r["best_lean"])
        pl = next((x for x in r["rows"] if x["rung"] == r["best_planner"]), None)
        lean = best["lean"]
        lines.append(f"| {r['scope']} | {r['target']} | {r['h']} | {r['mode']} | {r['best_lean']} | {_g(best['work'])} | "
                     f"{_g(lean['total'])} | {_g(lean['prove'])} | {_g(lean['commit'])} | "
                     f"{_g(lean['total'] / r['serve_s'])} | {r['best_planner'] or '-'} | "
                     f"{'-' if pl is None else _g(pl['planner']['total'])} | {r['best_lean_any']} | "
                     f"{r['best_marginal_any']} |")
    keys = [(s, "work") for s in scopes()] + [("window", "leak")]
    lines += ["", "Each rung's total, GPU-seconds (lean view, h 398 GB/s, proving + w x commit), as fitted / marginal:", "",
              "| rung | fits | proof blocks a window | committed GB a window | "
              + " | ".join(f"{s} {t}" for s, t in keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    pick = {(r["scope"], r["target"]): r for r in out["results"] if r["h"] == "h 398 GB/s" and r["weight"] > 0}
    for name, info in out["rungs"].items():
        cells = []
        for key in keys:
            row = next(x for x in pick[key]["rows"] if x["rung"] == name)
            cells.append(f"{_g(row['lean']['total'])} / {_g(row['lean_marginal']['total'])}")
        lines.append(f"| {name} | {'yes' if info['fits'] else 'no'} | {info['units']:,} | "
                     f"{info['committed'] / 1e9:.3g} | " + " | ".join(cells) + " |")
    lines += ["", "The work target on the chunk family alone: the cheapest L against √(w β h C/(o Λ)), β = 4 B:", "",
              "| h | windows pooled | C (MACs) | best L | √ law L | proving only L |", "|---|---|---|---|---|---|"]
    for r in out["sqrt_law"]:
        lines.append(f"| {r['h']} | {r['windows']:,} | {_g(r['C'])} | {r['best_L']:.0f} | {r['sqrt_L']:.0f} | "
                     f"{r['proving_only_L']:.0f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight", type=float, default=1.0, help="w, the commitment's weight in the second mode")
    ap.add_argument("--no-planner", action="store_true", help="the lean view only")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    out = run(args.weight, plan=not args.no_planner)
    (args.out / "served_window.json").write_text(json.dumps(out, indent=1, default=str))
    table = render(out)
    (args.out / "served_window.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
