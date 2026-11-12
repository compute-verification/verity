"""The served window's partition in the planner's two modes: Llama-3.2-1B int8 at T 8,192 on the H200, one step
a window, cut into proof blocks at each rung of a ladder, for the free-compute target and the exfiltration target,
priced proving only (`PROVING_ONLY`, Lean's `GranularityOptimum`) and proving plus `w` times the serve-time commitment
(`n b(s) h`, paid on every proof block, sampled or not) at two readings of h, a hashed byte's cost; and the second mode
again under a code-based commitment at each rate ρ of `RATES`.

A model, not a measurement: each input carries its status (M measured, D derived, A assumed) and source (`IN`, read
from `served_window_inputs.json`).

**The ladder.**  Every rung covers the window's work; per token and layer:

- `chunk L`, `coord`, `tile RxC`: the int8 linears (qkv, o_proj, gate_up, down) as K-chunks of L MACs that commit their
  32-bit running sums, as output coordinates (the served sample's `ScaledMmS8Coordinate`), or as tiles of R tokens by C
  output channels, opening each operand row once; the row ops stay at today's units (one an output coordinate), and
  attention is a query head's row in key blocks, the most keys a power of two whose block Flock holds, each block
  opening the running softmax state (its f32 output row, max and sum) and committing it, the last its output row.  The
  window commits what it commits today (every Call's outputs, `Q_word_v1`), plus the chunks' sums and attention's states.
- `attn|mlp token`, `layer token`, `4 layers token`: fused proof blocks that commit only their boundary, the residual
  stream and the K and V a later token reads: every interior value stays inside a proof block.

A proof block costs its ANDs (int8 MACs at `BlackwellS8ImmaDot32_v1`'s 4,190 per 32, bf16 MACs at `F32Fma_v2`'s 4,498, the row
ops at their pieces, and every row it opens at hm96's `row_ands`) at the cost model fitted from Flock's zero-knowledge
sessions (`inputs.observations`, GPU-seconds of an RTX PRO 6000), with its statement's share when statements are full.
`fits` is false for a rung with a proof block Flock's block can't hold (`inputs.cap`: its ANDs and rows at the composer's
occupancy past 2^`K_MAX`) or wider than the widest size class (`UNIT_CLASS_MAX` bits); such a rung is priced one proof
block a statement, what lifting the limit would buy.

**Two views.**  `lean`: each rung as n equal proof blocks of its mean size (`optimize.Grain`), its proving cost weighted by
work (free compute) or by width (exfiltration), through `grain_costs`, the laws of `Achievability` and
`ExfiltrationOptimum`.  `planner`: the rung's parts as strata through `for_target` under the density bound, its
statements shared (`CostModel.shared`), at `commit_weight` 0 or w.

**The code-based commitment.**  The served values committed in Flock's field as rows encoded at rate ρ, their codeword
columns Merkle-hashed at serve time (about 1/ρ the bytes, at h), and opened by Flock's own commitment through linear
claims: no proof block hashes a row in circuit (`Part.hashed` leaves its ANDs), and the hm96 commitment is gone.  The
encoding and the openings' own work in Flock are not priced, so its proving is a floor.  Each proof block's fit and its
statement's share stay today's (`inputs.cap`, which the hm96 and SHA-512 slots only enlarge), and no rung's fit turns on
those slots: a rung that misses Flock's block misses it on its own ANDs and rows' bits.

**Replay.**  Serving keeps only the requests' tokens and the weights (a commitment is a hash), so every mode's total adds the
rebuilding of the sampled proof blocks' witnesses (`replay`): each request a sampled proof block lies in, re-run whole from its
prompt at the stock step's share, under either commitment (`optimize.met_replay` on the `request` replay grain).  A
request's values are its own tokens' alone (the served engine is batch-invariant), so no co-scheduled request is re-run
with it.  The samples barely differ between rungs, so the replay moves the totals, not s*.

**Targets.**  Free compute: C = `EPS` of the scope's MACs, at the window and pooled over each monitoring period
(`reporting.PERIODS`: a week, a month and a year) of windows at the stock step; the samples are about Λ/ε whatever
the scope, the commitment grows with it.  Exfiltration: ℓ = `LEAK_BITS` a window.

    uv run python benchmarks/sampling_plan/served_window.py --out DIR [--weight W] [--rates R ...] [--no-planner]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from math import log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import EPILOGUE, I, K_MAX, KEY, S8_STEP, block_k_log, cap, observations

from verity.protocols import reporting
from verity.protocols.spot.verifier.profile import WINDOW_DELTA
from verity.protocols.spot.costfit import Fitted, fit
from verity.protocols.spot.verifier.one_stage.partition import UNIT_CLASS_MAX
from verity.protocols.spot.optimize import (PROVING_ONLY, Candidate, CostModel, Grain,
                                                           Objective, ReplayGrain, for_target, grain_costs, leak_count,
                                                           loc_bits, met_replay, row_ands, sqrt_grain)

ROOT = Path(__file__).resolve().parents[2]

DELTA = float(WINDOW_DELTA)
LAMBDA = log(1 / DELTA)
EPS = 0.01
LEAK_BITS = 2.0 ** 30
#: the code-based commitment's rates ρ: a codeword is 1/ρ its row
RATES = (1 / 2, 1 / 4)
#: the pieces' ANDs (their circuit-check pins: `verity/definitions/ops/*.circuit-check.json`)
F32_FMA, F32_MUL, F32_ADD, F32_MAX, EX2, RCP, F2BF16, F32_DIV, S8_CLAMP = 4498, 2406, 823, 283, 688, 197, 77, 2493, 71
F32_RINT = 275


@dataclass(frozen=True)
class Input:
    value: float
    status: str
    source: str


IN = {k: Input(**v) for k, v in json.loads(Path(__file__).with_name("served_window_inputs.json").read_text()).items()}
LLAMA_1B = {"layers": 16, "hidden": 2048, "heads": 32, "kv_heads": 8, "head_dim": 64, "intermediate": 8192}
#: (K, N) a token: qkv, o_proj, gate_up, down
LINEARS = {"qkv": (2048, 3072), "o_proj": (2048, 2048), "gate_up": (2048, 16384), "down": (8192, 2048)}
CHUNKS = (32, 64, 128, 256, 512, 1024)
TILES = ((2, 2), (4, 4), (16, 16), (64, 64), (128, 128))


@dataclass(frozen=True)
class Part:
    """One stratum of a rung: `units` proof blocks a window of `work` MACs each, `out_bits` wide, committing `commit_bytes`
    at serve time, of `ands` ANDs, the rows they open included (`hashed` of them, their openings at hm96's `row_ands`);
    `gates` and `rows` are its widest proof block's own ANDs and the rows it opens (bytes each), what Flock's block
    holds."""

    name: str
    units: int
    work: float
    out_bits: float
    commit_bytes: float
    ands: float
    gates: float
    rows: tuple[int, ...]
    hashed: float

    @property
    def k_log(self) -> int:
        return block_k_log(round(self.gates), self.rows)

    @property
    def fits(self) -> bool:
        return cap(round(self.gates), self.rows) > 0 and self.out_bits <= UNIT_CLASS_MAX

    @property
    def per_statement(self) -> int:
        """A statement's proof blocks (`inputs.cap`), one for a block Flock can't hold."""
        return cap(round(self.gates), self.rows) or 1


def openings(rows: tuple[float, ...]) -> float:
    """The ANDs of opening `rows` at hm96's `row_ands`."""
    return sum(row_ands(round(r)) for r in rows)


def opened(gates: float, rows: tuple[float, ...]) -> float:
    """A proof block's ANDs with its rows' openings."""
    return gates + openings(rows)


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
ROW_OP_GATES = 2 * F32_MUL + F32_DIV + F32_RINT + S8_CLAMP + F2BF16
ROW_OP_ROWS = (4, 4, 4)
ROW_OP_ANDS = opened(ROW_OP_GATES, ROW_OP_ROWS)


def attention_block(keys: float, last: bool) -> tuple[float, tuple[float, ...]]:
    """A query head's key block of `keys` keys: its ANDs (QK^T and PV, softmax a key, and rescaling the running output row
    and sum by 2^(m - m') to add its own; the last the reciprocal) and its rows: q, its K and V, the state in, and the
    state out (the f32 output row, max and sum) or, the last, the output row."""
    hd = LLAMA_1B["head_dim"]
    state = 4 * (hd + 2)
    gates = keys * (2 * hd * F32_FMA + EX2 + F32_ADD + F32_MUL + F2BF16 + F32_MAX) + (hd + 1) * F32_FMA + EX2 + F32_ADD \
        + (RCP if last else 0)
    return gates, (2 * hd, 2 * hd * keys, 2 * hd * keys, state, 2 * hd if last else state)


def attention_keys() -> int:
    """A key block's keys: the most, a power of two, whose block Flock holds."""
    positions = round(IN["tokens"].value / IN["sequences"].value)
    return next(kb for kb in (positions >> i for i in range(positions.bit_length()))
                if all(cap(round(g), rows) for g, rows in (attention_block(kb, last) for last in (False, True))))


def attention_parts() -> list[Part]:
    """Attention as key blocks of `attention_keys` keys, a query head's row at position p in ceil(p/kb) of them: the
    blocks before its last, and its last at the mean keys it covers (its widest at kb)."""
    tl, hd, heads, kb = token_layers(), LLAMA_1B["head_dim"], LLAMA_1B["heads"], attention_keys()
    positions = round(IN["tokens"].value / IN["sequences"].value)
    before = sum(-(-p // kb) - 1 for p in range(1, positions + 1)) / positions
    tail = sum(p - (-(-p // kb) - 1) * kb for p in range(1, positions + 1)) / positions
    state = 4 * (hd + 2)
    g, rows = attention_block(kb, False)
    g_tail, rows_tail = attention_block(tail, True)
    g_last, rows_last = attention_block(kb, True)
    return [Part("attention", round(tl * heads * before), 2 * hd * kb, 8 * state, state, opened(g, rows), g,
                 tuple(map(round, rows)), hashed=openings(rows)),
            Part("attention last", tl * heads, 2 * hd * tail, 16 * hd, 2 * hd, opened(g_tail, rows_tail), g_last,
                 tuple(map(round, rows_last)), hashed=openings(rows_tail))]


def rest_parts() -> list[Part]:
    """Attention (`attention_parts`) and the row ops (one 16-bit coordinate each, at today's units)."""
    tl = token_layers()
    rows = row_op_bytes() / 2
    return attention_parts() + [Part("row ops", round(tl * rows), 0.0, 16, 2, ROW_OP_ANDS, ROW_OP_GATES, ROW_OP_ROWS,
                                     hashed=openings(ROW_OP_ROWS))]


def coord_ands(k: int) -> float:
    return k // 32 * S8_STEP + EPILOGUE + 2 * row_ands(k) + 2 * row_ands(4) + row_ands(2)


def linear_parts(rung: str) -> list[Part]:
    tl, out = token_layers(), []
    for op, (k, n) in LINEARS.items():
        if rung == "coord":
            out.append(Part(op, tl * n, k, 16, 2, coord_ands(k), k // 32 * S8_STEP + EPILOGUE, (k, k, 4, 4, 2),
                            hashed=openings((k, k, 4, 4, 2))))
        elif rung.startswith("chunk"):
            c = min(int(rung.split()[1]), k)
            last = c / k
            bits = 32 * (1 - last) + 16 * last
            ands = c // 32 * S8_STEP + 2 * row_ands(c) + 2 * row_ands(4) \
                + last * (EPILOGUE + 2 * row_ands(4) + row_ands(2))
            out.append(Part(op, tl * n * (k // c), c, bits, bits / 8, ands, c // 32 * S8_STEP + EPILOGUE,
                            (c, c, 4, 4, 4, 4, 2), hashed=openings((c, c, 4, 4)) + last * openings((4, 4, 2))))
        else:
            r, c = map(int, rung.split()[1].split("x"))
            cells = r * c
            gates, rows = cells * (k // 32 * S8_STEP + EPILOGUE), (k,) * (r + c) + (4 * r, 4 * c, 2 * cells)
            out.append(Part(op, tl * n // cells, cells * k, 16 * cells, 2 * cells, opened(gates, rows), gates, rows,
                            hashed=openings(rows)))
    return out


def _fused(name: str, units: int, int8: float, bf16: float, nbytes: float, out_bytes: float, row_ops: float) -> Part:
    """A fused proof block: its int8 and bf16 MACs, its row ops' pieces, and the `nbytes` it opens (weights and their
    scales, the K and V it reads, its input and output) at hm96's per-byte rate."""
    per_byte = (row_ands(1 << 20) - row_ands(0)) / (1 << 20)
    gates = int8 * S8_STEP / 32 + bf16 * F32_FMA + row_ops
    return Part(name, units, int8 + bf16, 8 * out_bytes, out_bytes, gates + nbytes * per_byte, gates, (round(nbytes),),
                hashed=nbytes * per_byte)


def fused_parts(rung: str) -> list[Part]:
    """Proof blocks that commit only their boundary: the residual stream (bf16), and the K and V a later token reads."""
    tl, d = token_layers(), LLAMA_1B["hidden"]
    kv = 2 * 2 * LLAMA_1B["kv_heads"] * LLAMA_1B["head_dim"]
    macs, attn_ands = attention_row()
    heads = LLAMA_1B["heads"]
    bf16 = heads * macs
    softmax = heads * (attn_ands - macs * F32_FMA - 2 * row_ands(round(macs)) - 2 * row_ands(2 * LLAMA_1B["head_dim"]))
    row_bytes = row_op_bytes()
    row_ops = row_bytes / 2 * ROW_OP_GATES
    attn_w = sum(k * n for op, (k, n) in LINEARS.items() if op in ("qkv", "o_proj"))
    mlp_w = sum(k * n for op, (k, n) in LINEARS.items() if op in ("gate_up", "down"))
    scales = 4 * sum(n for _, n in LINEARS.values())
    keys = kv * mean_keys()
    if rung == "attn|mlp token":
        roped = (LLAMA_1B["heads"] + LLAMA_1B["kv_heads"]) * LLAMA_1B["head_dim"]
        share = (2 * d + 2 * roped) / (row_bytes or 1)
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


def coded(parts: list[Part]) -> list[Part]:
    """The rung under the code-based commitment: each proof block's ANDs less its rows' in-circuit hashing."""
    return [replace(p, ands=p.ands - p.hashed, hashed=0.0) for p in parts]


def fitted() -> Fitted:
    obs = observations()
    return Fitted(fit(obs, "prove", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "verify", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "stage", ["one", "ands", "ands*staged"]), I["cpu"].value, KEY)


def unit_cost(m: CostModel, ands: float, per: int, marginal: bool = False) -> float:
    """A sampled proof block's GPU-seconds in a full statement of `per` of them: proved and staged, and 1/per of the
    statement.  The fit's per-statement term grows with the subcircuit's ANDs, which a statement of one proof block pays
    whole; `marginal` leaves it out, the reading where that term is fixed."""
    statement = m.statement_fixed + (0.0 if marginal else m.statement_per_and * ands)
    return ands * (m.per_and + m.staged_per_and) + statement / per


def scale(parts: list[Part], windows: float) -> list[Part]:
    return [replace(p, units=round(p.units * windows)) for p in parts]


def grain(name: str, parts: list[Part], m: CostModel, by: str, marginal: bool = False) -> Grain:
    """The rung as n equal proof blocks of its mean work and width, its proving cost weighted by work (`by` "work") or by
    width ("bits"): the cost the laws pay, which sample a proof block in proportion to it."""
    n = sum(p.units for p in parts)
    weight = {p.name: p.units * (p.work if by == "work" else p.out_bits) for p in parts}
    total = sum(weight.values())
    prove = sum(weight[p.name] * unit_cost(m, p.ands, p.per_statement, marginal) for p in parts) / total
    return Grain(name, n, sum(p.units * p.work for p in parts) / n, sum(p.units * p.out_bits for p in parts) / n,
                 sum(p.units * p.commit_bytes for p in parts) / n, prove)


def candidates(rung: str, parts: list[Part]) -> list[Candidate]:
    return [Candidate(p.name, f"{rung}:{p.name}", p.units, round(p.work), round(p.out_bits), ands=round(p.ands),
                      commit_bits=round(8 * p.commit_bytes), cap=p.per_statement) for p in parts]


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


def request_grain(windows: float) -> ReplayGrain:
    """The scope's requests as replay blocks, each re-run from its prompt at `step_s` / `sequences` (the requests are
    alike)."""
    return ReplayGrain(round(IN["sequences"].value * windows), IN["step_s"].value / IN["sequences"].value)


def replayed(cost: dict | None, windows: float) -> dict | None:
    """``cost`` with its samples' `replay` (the `request` replay grain's blocks they meet) in its total."""
    if cost is None:
        return None
    r = met_replay(request_grain(windows), cost["draws"])
    return {**cost, "replay": r, "total": cost["total"] + r}


def unit_work(parts: list[Part]) -> float:
    """A rung's proof-block size as the table quotes it: an int8 linear's proof block (gate_up's, most of the work), else the
    fused block's largest."""
    return next((p.work for p in parts if p.name == "gate_up"), max(p.work for p in parts))


def readings() -> dict[str, float]:
    return {"h 521 GB/s": 1 / IN["hash_fast"].value, "h 398 GB/s": 1 / IN["hash_slow"].value}


def scopes() -> dict[str, float]:
    return {"window": 1.0} | {name: seconds / IN["step_s"].value for name, seconds in reporting.PERIODS}


def run(weight: float = 1.0, *, rates: tuple[float, ...] = RATES, plan: bool = True) -> dict:
    m = fitted().model()
    ladder = rungs()
    work_window = sum(p.units * p.work for p in ladder["coord"])
    out = {"inputs": {k: v.__dict__ for k, v in IN.items()}, "model": LLAMA_1B, "delta": DELTA, "eps": EPS,
           "leak_bits": LEAK_BITS, "weight": weight, "code_rates": list(rates), "work_window": work_window,
           "rates": {"per_and": m.per_and, "staged_per_and": m.staged_per_and, "statement_fixed": m.statement_fixed,
                     "statement_per_and": m.statement_per_and},
           "h_mac_times": {k: h * IN["peak_macs"].value for k, h in readings().items()},
           "rungs": {name: {"fits": all(p.fits for p in parts),
                            "parts": [{**p.__dict__, "k_log": p.k_log, "fits": p.fits, "per_statement": p.per_statement,
                                       "unit_s": unit_cost(m, p.ands, p.per_statement)} for p in parts],
                            "units": sum(p.units for p in parts),
                            "committed": sum(p.units * p.commit_bytes for p in parts),
                            "unit_work": unit_work(parts)}
                     for name, parts in ladder.items()},
           "results": []}
    commit = f"proving + {weight:g} x commit"
    schemes = [("hm96", None, ladder, {"proving only": PROVING_ONLY, commit: weight})]
    schemes += [(f"code ρ {Fraction(rho).limit_denominator()}", rho, {name: coded(parts) for name, parts in ladder.items()},
                 {commit: weight}) for rho in rates]
    cases = [(scope, "work", windows) for scope, windows in scopes().items()] + [("window", "leak", 1.0)]
    for scope, target, windows in cases:
        for hname, h_byte in readings().items():
            for scheme, rho, ladder_s, modes in schemes:
                h = h_byte / (rho or 1.0)
                for mode, w in modes.items():
                    rows = []
                    request = request_grain(windows)
                    for name, parts in ladder_s.items():
                        parts_s = scale(parts, windows)
                        kw = {"work": EPS * work_window * windows} if target == "work" else {"leak": LEAK_BITS}
                        by = "work" if target == "work" else "bits"
                        g = grain(name, parts_s, m, by)
                        lean = grain_costs([g], DELTA, h=h, weight=w, replay=request, **kw)[0]
                        marginal = grain_costs([grain(name, parts_s, m, by, marginal=True)], DELTA, h=h, weight=w,
                                               replay=request, **kw)[0]
                        row = {"rung": name, "fits": out["rungs"][name]["fits"], "work": unit_work(parts),
                               "mean_work": g.work, "lean": lean, "lean_marginal": marginal}
                        if plan:
                            row["planner"] = replayed(planner(name, parts_s, m, h=h, weight=w, **kw), windows)
                        rows.append(row)
                    fit_rows = [r for r in rows if r["fits"]]
                    out["results"].append({
                        "scope": scope, "target": target, "h": hname, "scheme": scheme, "rate": rho, "mode": mode,
                        "weight": w, "rows": rows,
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

    def chunk(c: int) -> float:
        gates, rows = c // 32 * S8_STEP, (c, c, 4, 4)
        return unit_cost(m, opened(gates, rows), cap(gates, rows))

    o = (chunk(2048) - chunk(1024)) / 1024
    rows = []
    for hname, h in readings().items():
        for windows in (1, 16, 256, 4096, 65536):
            target = EPS * work * windows
            ladder = [Grain(f"L={c}", round(work * windows / c), c, 32, 4.0, chunk(c)) for c in range(32, 8192 + 1, 32)]
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
    lines = [f"# Served window, Llama-3.2-1B int8 T 8,192: proof-block size in two modes and under a code-based commitment "
             f"(δ = 2^-40, ε = {EPS:.0%}, ℓ = 2^{log2(LEAK_BITS):.0f} bits a window)", ""]
    lines.append("GPU-seconds (proving: RTX PRO 6000 ZK fit, modelled for these units; commitment: H200 SHA-512 at h, hm96's "
                 "rows, or at h/ρ, a code at rate ρ whose proof blocks hash no row in circuit, its encoding and openings "
                 "unpriced; replay: H200, each request a drawn proof block lies in re-run from its prompt). 'x serve' divides "
                 "by the scope's stock serving time. s* is the cheapest rung every proof block "
                 f"of which C-Flock's block holds (k_log <= {K_MAX} at the composer's occupancy, `inputs.cap`; attention in "
                 f"key blocks of {attention_keys()}) and the size classes; 'any' includes the rungs that don't.")
    lines += ["", "| scope | target | h | commitment | mode | s* (lean) | s* (MACs a gate_up block) | total | prove | commit | "
              "replay | x serve | s* (planner) | planner total | s* any, as fitted | s* any, marginal |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        best = next(x for x in r["rows"] if x["rung"] == r["best_lean"])
        pl = next((x for x in r["rows"] if x["rung"] == r["best_planner"]), None)
        lean = best["lean"]
        lines.append(f"| {r['scope']} | {r['target']} | {r['h']} | {r['scheme']} | {r['mode']} | {r['best_lean']} | "
                     f"{_g(best['work'])} | "
                     f"{_g(lean['total'])} | {_g(lean['prove'])} | {_g(lean['commit'])} | {_g(lean['replay'])} | "
                     f"{_g(lean['total'] / r['serve_s'])} | {r['best_planner'] or '-'} | "
                     f"{'-' if pl is None else _g(pl['planner']['total'])} | {r['best_lean_any']} | "
                     f"{r['best_marginal_any']} |")
    keys = [(s, "work") for s in scopes()] + [("window", "leak")]
    lines += ["", "Each rung's total, GPU-seconds (lean view, hm96, h 398 GB/s, proving + w x commit + replay), as fitted / marginal:",
              "", "| rung | fits | widest block's k_log | proof blocks a window | committed GB a window | "
              + " | ".join(f"{s} {t}" for s, t in keys) + " |", "|---|---|---|---|---|" + "---|" * len(keys)]
    pick = {(r["scope"], r["target"]): r for r in out["results"]
            if r["h"] == "h 398 GB/s" and r["scheme"] == "hm96" and r["weight"] > 0}
    for name, info in out["rungs"].items():
        cells = []
        for key in keys:
            row = next(x for x in pick[key]["rows"] if x["rung"] == name)
            cells.append(f"{_g(row['lean']['total'])} / {_g(row['lean_marginal']['total'])}")
        widest = max(info["parts"], key=lambda p: p["k_log"])
        lines.append(f"| {name} | {'yes' if info['fits'] else 'no'} | {widest['k_log']} ({widest['name']}) | "
                     f"{info['units']:,} | "
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
    ap.add_argument("--rates", type=float, nargs="+", default=list(RATES),
                    help="the code-based commitment's rates ρ, each in (0, 1]")
    ap.add_argument("--no-planner", action="store_true", help="the lean view only")
    args = ap.parse_args(argv)
    if not all(0 < r <= 1 for r in args.rates):
        ap.error("a code's rate ρ is in (0, 1]")
    args.out.mkdir(parents=True, exist_ok=True)
    out = run(args.weight, rates=tuple(args.rates), plan=not args.no_planner)
    (args.out / "served_window.json").write_text(json.dumps(out, indent=1, default=str))
    table = render(out)
    (args.out / "served_window.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
