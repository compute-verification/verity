"""Dither's skippable share of honest work as served, per model, context, phase and design (SKIP, cmp-13; Derived).

    uv run python benchmarks/pouw/eps_budget.py [--json OUT] [--md OUT]

ε_skip = (honest work a cheater can skip) / (the cheater's unavoidable work), the brief's definition
(note:20261017T0013Z-report-epsilon-program). Each component of honest work is classified per design as unavoidable (a
transcript Verify recomputes, a commitment, a draw, noise the cheater must expand, a value a check recomputes from the
transcript) or skippable (unchecked: the strip of a low-rank noise, a product served plain, the vector work only a
served output needs). The cheater is the zero instance (A* = B* = 0, chosen before the draw), which makes every strip,
every served output and everything derived from it free unless a check recomputes it from the transcript.

- MAC-weighted: tensor MACs only, strips at their logical rank (L per output word).
- Time-weighted: every component's GPU time on one RTX PRO 6000 (sm_120) at its read clock, from the measured rates in
  RATES (each with its note and run), executed strips (two int8 limbs), epilogue and softmax included. Decode is
  memory-bound, the cheater's transcript is not (B̃ = S is generated from U, and the zero instance streams no weights),
  so decode's time ratio is a bandwidth gap no noise design closes; it is reported (`decode_dram`), not optimized.

The designs are the registry's (benchmarks/pouw/machine_model/data/designs/, `machine-model designs`): each product is
noised as its design's `[noise]` names, a leaf of the registry's NOISE, and priced per component in executed int8
products and honest seconds, with γ (`products`, `parts`, `summary`, `skip_table`, `component_table`). A checked leaf's
vector work that only the served output needs is priced in CHECKED-STRIP's tensor–scalar model (`TS_EPS`, `pv_ts`), and
`pv_block_table` prices P·V's block size for the designs whose P·V is `pv`. A design's ε_skip (`design_skip`) is the
most over the models, contexts and phases; the registry's `[skip]` is that number (test_eps_budget holds them equal).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from verity_machine_model.eps import PRODUCTS, Design, design, designs

L = 32                      # transcript stride and the low-rank noise's rank
CONTEXTS = (2048, 8192, 32768)
PHASES = ("prefill", "decode")


@dataclass(frozen=True)
class Shape:
    h: int
    layers: int
    heads: int
    kv: int
    d: int
    inter: int
    vocab: int
    tp: int                 # GPUs a decode replica uses (the decode step below is per replica)

    @property
    def linears(self) -> list[tuple[str, int, int]]:
        """(name, k, n) per layer; one GEMM per stage (q/k/v fused, gate/up fused)."""
        return [("qkv", self.h, (self.heads + 2 * self.kv) * self.d), ("o", self.heads * self.d, self.h),
                ("gate_up", self.h, 2 * self.inter), ("down", self.inter, self.h)]


MODELS = {"llama-3.1-8b": Shape(4096, 32, 32, 8, 128, 14336, 128256, 1),
          "llama-3.1-70b": Shape(8192, 80, 64, 8, 128, 28672, 128256, 2)}

#: Rates, each measured (with its note) unless marked estimate.
RATES = {
    "peak": 188 * 1014 * 2.092e9,   # IMMA s8 MAC/s at the read clock (FP-EXACT, note:20261016T1931Z)
    "eta_lin": 0.80,     # prefill int8 GEMMs alone, 32 layers x 6 stages at 2K: 46.4 ms (STAGE-COST r20261016-095504-5463)
    "eta_att": 0.70,     # fused int8 attention, 60-70% of int8 peak (STAGE-COST r20261016-100833-1b7c)
    "eta_dec": 0.71,     # decode's noised products batched, 0.97 µs a token-layer (STAGE-COST, same run)
    "r_receipt": 1.638,  # edited GEMM kernels / plain at S = 128: 1.668 (COMMIT-GPU r5 r20261016-183051-a379), less the
                         # 2-atom tail's 0.030 (EGEMM r20261016-090505-596b: 4 atoms 0.024 ms of 0.405 at o_proj)
    "tail": 1.9,         # a strip MAC costs 1.9 mainloop MACs (that 4-atom tail: 0.006 ms an atom against 32/4096 of 0.405)
    "t_quant": 0.56e-12,     # s an element, plain activation quantizer (COMMIT-GPU r5: +23 ms at 8 x 4K, 40.8 G elements)
    "t_dither": 0.48e-12,    # s an element, the dither's bits (EGEMM r20261016-104404-700b: qdither - qplain 0.016 ms, 8192 x 4096)
    "t_wedit": 0.86e-12,     # s a weight element a draw, W' = W + U written (EGEMM: wedit 0.0144 ms, 4096 x 4096)
    "t_xcommit": 1.10e-12,   # s an operand element, SHA-512 tile leaves R = 128 (COMMIT-GPU r5: +45 ms at 8 x 4K)
    "receipt_commit": 0.016,  # of plain prefill time, S = 128 (COMMIT-GPU r5)
    "draws": 0.019,           # of plain prefill time, 0.3 ms simulated draws at 8 x 4K (COMMIT-GPU r5)
    "tokens_per_draw": 32768,  # prefill stage set a GPU (STAGE-COST's recommendation)
    "softmax": 0.25,     # estimate: exp, max, sum and P's quantize per (query head, key) against 256 MACs a pair; MUFU at
                         # 16 a SM a clock against 1014 MACs gives 0.25 of the attention's tensor time (range 0.12-0.5)
    "nongemm_token": 4.68e-6,  # s a token of non-GEMM kernels at prefill, Llama-3.1-8B, vLLM (note:20261016T0215Z §2)
    "nongemm_step": 1.09e-3,   # s a decode step floor of the same kernels (same note)
    "bw": 1.41e12,       # B/s, decode attention's measured KV read (same note, dcap r20261016-023229-185e)
    "lin_bw_b32": 6.98e9 / 5.9e-3,  # B/s, 32-layer int8 linears at M = 32: 5.64 ms (M = 1) to 6.18 ms (M = 64) (same run)
    "step_fixed": 0.293e-3,    # s a decode step, the step model's constant (same note: within 1.7% of 14 steps)
    "comm_tp2_b32": 5.2e-3,    # s a TP-2 decode step of all-reduces, calibrated to HC-COMM r6's base step at B 32, 2K
    "decode_commit": 0.012,   # of the decode step, committing its operands (STAGE-COST: 0.9-1.4%)
}
DECODE_BATCH = 32


def epilogue_bytes(s: Shape) -> int:
    """Bytes a token-layer the non-GEMM kernels move (bf16, fused): RoPE on Q and K, two residual adds with RMSNorm,
    SiLU x up, the KV cache write. Only its ratio between models is used: the time is vLLM's measured 8B figure."""
    return 4 * (s.heads + s.kv) * s.d + 12 * s.h + 6 * s.inter + 2 * s.kv * s.d


def decode_step(name: str, ctx: int, rates: dict = RATES) -> dict:
    """A plain decode step at batch DECODE_BATCH, in GPU-seconds a token-layer: `step` the whole, `gemm_dram` the part
    that streams the linears' weights and the KV cache (the time the GEMMs' tensor work hides under)."""
    s, r = MODELS[name], rates
    tokens = DECODE_BATCH * s.layers
    lin = sum(k * n for _, k, n in s.linears)
    att_bytes = DECODE_BATCH * ctx * s.layers * 2 * s.kv * s.d
    dram = (s.layers * lin / r["lin_bw_b32"] + att_bytes / r["bw"]) / s.tp
    step = dram + r["step_fixed"] + r["nongemm_step"] + (r["comm_tp2_b32"] if s.tp == 2 else 0)
    return {"step": step * s.tp / tokens, "gemm_dram": dram * s.tp / tokens}


# ------------------------------------------------- ε_skip per component and γ, each product noised as its design says

#: The served model's components. The first four are the registry's PRODUCTS (`COMP`), each noised as its design's
#: `[noise]` names. Every design derives each stage's operand by C1′ from the previous stage's served output (the
#: epilogue and quantizer are checked) and serves decode edited, under one σ a request drawn once the circuit and its
#: inputs are committed (decision 72), so softmax and forming are never skippable and no pass is deferred.
COMPONENTS = ("linears", "lm_head", "QKᵀ", "P·V", "softmax", "forming", "noise, commitments, draws")
COMP = {"linears": "linears", "lm_head": "lm_head", "QKᵀ": "qk", "P·V": "pv"}
#: co32c's digit limbs: the residue sums of an 8-bit served side over k / 32 terms take up to 3 base-128 digits
CO32C_DIGITS = 3


def pv_digits(keys: int) -> int:
    """m, the base-128 digits of P V's Ycum (CHECKED-STRIP's P3): 2 up to 2K keys, 3 up to 128K."""
    return 2 if keys <= 2048 else 3


def products(leaf: str, comp: str, k: int) -> tuple[float, float, float]:
    """(transcript, strip, skippable) int8 products an output word of depth k executes, `comp` noised as `leaf`. A strip
    runs on two int8 limbs when its operand is a residue sum that leaves int8 (L = 32 classes of k / L terms: the
    one-sided peel), on one when it fits (cohalf: two terms with a 7-bit served side), on checked digit limbs when the
    sums' base-128 digits are committed (co32c; pv, (m + 1)·L for m digits). P's peel frees P V's whole strip: at a
    one-hot P row its output is a copy (CHECKED-STRIP's Result 3). A product served plain (`clean`) is `parts`'."""
    if comp == "P·V":
        if leaf == "peel":
            return k, 2 * (L + 1), 2 * (L + 1)
        if leaf == "pv":
            return k, (pv_digits(k) + 1) * L, 0
    elif leaf == "one-sided":
        return k, 2 * L, 2 * L
    elif leaf == "two-sided":
        return k, 0, 0
    elif leaf == "cohalf":
        return k, k // 2, 0
    elif leaf == "co32c":
        return k, CO32C_DIGITS * L, 0
    raise ValueError(f"no price for {leaf} on {comp}: add it to products (and to the registry's NOISE)")


@dataclass
class Part:
    component: str
    macs: float          # int8 products a token-layer the honest prover executes
    macs_skip: float     # of them, the ones Verify doesn't require
    seconds: float       # honest GPU-seconds a token-layer
    seconds_skip: float
    plain: float         # GPU-seconds of the same component served as plain int8


def parts(name: str, ctx: int, phase: str, noise: dict[str, str], rates: dict = RATES, block: int = 0) -> list[Part]:
    """Per component, a token-layer: honest int8 products, the skippable ones, honest seconds, skippable seconds and plain
    int8 seconds, each product noised as `noise` (a design's `[noise]`) says. Prefill adds a checked leaf's vector work
    that only the served output needs, in the tensor–scalar model (`TS_EPS`; pv's `pv_ts`, P in blocks of `block` keys,
    P V's honest seconds then plain int8 × pv_ts's γ), every served output free (no downstream check assumed); decode
    prices products only. Decode's seconds are the GEMMs' compute, which the DRAM-bound step hides (`decode_gamma`);
    decode's binding facet is DRAM (`decode_dram`)."""
    s, r = MODELS[name], rates
    keys = (ctx + 1) / 2 if phase == "prefill" else ctx
    t_lin = 1 / (r["eta_lin"] * r["peak"])
    t_att = 1 / (r["eta_att"] * r["peak"])
    acc: dict[str, list[float]] = {}

    def gemm(comp, words, k, t):
        leaf = noise[COMP[comp]]
        if leaf == "clean":
            c = words * k
            row = [c, c, c * t, c * t, c * t]       # served plain, never proved
        else:
            tr, st, sk = products(leaf, comp, k)
            short = st and (leaf != "cohalf" or st <= 2 * (L + 1))     # limbs of L or L + 1 terms
            strip_t = t * (r["tail"] if short else 1)
            row = [words * (tr + st), words * sk, words * (tr * t * r["r_receipt"] + st * strip_t),
                   words * sk * (strip_t if st else t * r["r_receipt"]), words * k * t]
        acc[comp] = [a + b for a, b in zip(acc.get(comp, [0] * 5), row)]

    for _, k, n in s.linears:
        gemm("linears", n, k, t_lin)
    gemm("lm_head", s.vocab / s.layers / (ctx if phase == "prefill" else 1), s.h, t_lin)
    gemm("QKᵀ", s.heads * keys, s.d, t_att)
    gemm("P·V", s.heads * s.d, int(keys), t_att)
    if phase == "prefill":
        for comp, row in acc.items():
            leaf = noise[COMP[comp]]
            if leaf == "pv":
                e, g = pv_ts(name, ctx, block)
                row[2] = row[4] * g
            elif leaf in TS_EPS:
                if comp not in TS_EPS[leaf][name]:
                    raise ValueError(f"no tensor–scalar price for {leaf} on {comp}: add it to TS_EPS "
                                     "(theory/checked_strip.py)")
                e = TS_EPS[leaf][name][comp]
            else:
                continue
            row[3] += row[2] * e / (1 + e)
    out = [Part(c, *v) for c, v in acc.items()]
    att_plain = 2 * s.heads * s.d * keys * t_att
    soft = r["softmax"] * att_plain
    out.append(Part("softmax", 0, 0, soft, 0, soft))
    act = 3 * s.h + s.inter
    qkv_el = (s.heads + 2 * s.kv) * s.d
    epi = r["nongemm_token"] / MODELS["llama-3.1-8b"].layers * epilogue_bytes(s) / epilogue_bytes(MODELS["llama-3.1-8b"])
    form = max(epi, r["t_quant"] * (act + qkv_el))
    out.append(Part("forming", 0, 0, form, 0, form))
    lin_plain = sum(k * n for _, k, n in s.linears) * t_lin
    weights = sum(k * n for _, k, n in s.linears)
    extra = (r["t_dither"] * (act + qkv_el + s.heads * s.d + weights / r["tokens_per_draw"])
             + r["t_xcommit"] * (act + qkv_el) + (r["receipt_commit"] + r["draws"]) * (lin_plain + att_plain))
    if noise["linears"] == "one-sided":                # U written into the weights once a draw
        extra += r["t_wedit"] * weights / r["tokens_per_draw"]
    out.append(Part("noise, commitments, draws", 0, 0, extra, 0, 0))
    return out


# ------------------------------------------------- the tensor–scalar model: checked leaves' vector work, pv's units

#: The tensor–scalar (TS) model's facets in SMSP-cycles, (MMA coefficient, coefficient of every other vector-pipe warp
#: instruction): `M` (the tensor pipe) and JI3e_0..2, benchmarks/pouw/machine_model/data/rows.csv (CHECKED-STRIP's
#: TS_FACETS, note:20261017T0030Z-draft-checked-strip).
TS_FACETS = ((16.0, 0.0), (8.115, 1.0), (15.030, 0.317), (15.539, 0.172))
#: pv's vector work a tile (16 queries by 8 channels: 128 cells, 4 a lane) beyond the folds, in warp instructions.
#: Estimates from instruction counts; no kernel has measured them: one digit of one Ycum entry (a shift or mask, then
#: the carry; CHECKED-STRIP's PV_DIGIT_OPS), the epilogue's offset and centring (one op a cell; PV_SERVE_OPS), and a
#: unit's I2F and FFMA into the float output (two ops a cell), which every block of P adds once.
PV_DIGIT_OPS, PV_SERVE_OPS, PV_SCALE_OPS = 2.0, 4.0, 8.0
#: A checked leaf's vector work that only the served output needs, in the TS model at the duo fold, as a fraction of its
#: component's own time, by leaf, model and component: co32c's recombination with O committed (linears, lm_head) and
#: cohalf with every zero-instance route at X_BITSLICED (Q Kᵀ, k = 128), from checked_strip.py @007ee9c54
#: (`ts_co32c_recombination`, `ts_qk_k128_all_routes`). Two-sided's served outputs are its last checked words and its
#: bit × bit words cost more than an IMMA (QSTAR's timed r20261017-055042-7c59), so it has none; pv's is `pv_ts`.
TS_EPS = {"co32c": {"llama-3.1-8b": {"linears": 7.63e-4, "lm_head": 4.9e-5},
                    "llama-3.1-70b": {"linears": 3.91e-4, "lm_head": 4.9e-5}},
          "cohalf": {"llama-3.1-8b": {"QKᵀ": 1.984e-3}, "llama-3.1-70b": {"QKᵀ": 1.984e-3}}}
TS_FOLD = 3.0
#: Warp instructions a bit-sliced 32 × 128 binary product takes (QSTAR's bx, note:20261017T0344Z-draft-eps1-qstar).
X_BITSLICED = 10.66


def ts_time(mmas: float, ops: float) -> float:
    return max(a * mmas + b * ops for a, b in TS_FACETS)


def base128_digits(g: int) -> int:
    """m, the base-128 digits a sum of g int8 terms takes (CHECKED-STRIP's n_digits over g·[−128, 127])."""
    m = 1
    while not (-128 * g >= -(128 ** m) and 127 * g // 128 ** (m - 1) <= 127):
        m += 1
    return m


def pv_unit(k: int, rep: int, fold: float = TS_FOLD, x: float = X_BITSLICED):
    """One pv unit's tile at depth k keys (its diagonal 32 included), as CHECKED-STRIP's ts_pv prices it plus the unit's
    float scaling: (MMAs, folds, free, routes). The cheater does every MMA and fold (each feeds a committed word); free
    is what only the served output needs: the epilogue, the float scaling, and Ycum's and S's adds and Ycum's digits
    (shared by the `rep` query heads of a KV head). Routes are the zero-instance ones, {direction: (dM, dO)}: T at 8x an
    IMMA, W and the D_d at 7x, O′ through c·b at x·k/32."""
    g = k // 32 - 1
    m = base128_digits(g) if g else 1
    mmas = k // 32 + (m + 1 if g else 1)
    folds = 4 * fold * (k // 32 + m)
    free = 4 * (m - 1) + PV_SERVE_OPS + PV_SCALE_OPS + (4 + 4 + 4 * m * PV_DIGIT_OPS) / rep
    routes = {"T": (-(k // 32), 8 * x * (k // 32)), "WD": (-m, 7 * x * m), ("O′", k): (-(2 if g else 1), x * k / 32)}
    return mmas, folds, free, routes


def ts_best(hon: tuple, cheat: tuple, routes: list) -> tuple[float, float]:
    """(honest's time, the best cheater's) when the cheater moves a fraction φ_j ∈ [0, 1] of each route's work, changing
    its (MMAs, other instructions) by Σ φ_j (dM_j, dO_j). The minimum over the box of the facets' max is a vertex of the
    (φ, t) program, where len(routes) + 1 facet and box constraints are tight; every vertex is solved for."""
    R, d = len(routes), np.array(routes, float).reshape(-1, 2)
    rows = [np.r_[a * d[:, 0] + b * d[:, 1], -1.0] for a, b in TS_FACETS] + [np.r_[np.eye(R)[j], 0.0] for j in range(R)
                                                                             for _ in (0, 1)]
    rhs = [-(a * cheat[0] + b * cheat[1]) for a, b in TS_FACETS] + [v for _ in range(R) for v in (0.0, 1.0)]
    combos = np.array(list(itertools.combinations(range(len(rows)), R + 1)))
    A, y = np.array(rows)[combos], np.array(rhs)[combos]
    ok = np.abs(np.linalg.det(A)) > 1e-9
    phi = np.linalg.solve(A[ok], y[ok][..., None])[..., :R, 0]
    phi = np.vstack([phi[np.all((phi >= -1e-12) & (phi <= 1 + 1e-12), axis=1)], np.zeros((1, R))])
    mm, ops = cheat[0] + phi @ d[:, 0], cheat[1] + phi @ d[:, 1]
    return ts_time(*hon), float(np.max([a * mm + b * ops for a, b in TS_FACETS], axis=0).min())


def pv_ts(name: str, ctx: int, block: int, sink: bool = True, fold: float = TS_FOLD,
          x: float = X_BITSLICED) -> tuple[float, float]:
    """pv's P V in a prefill of ctx tokens with P in blocks of `block` keys (0: one block): (ε_skip with the served
    output free, γ over plain int8), each query block's tile weighed (past 256 every 8th, weighted 8, as
    CHECKED-STRIP). Each block of P is its own unit (scale, T words, Ycum digits, served output); with `sink` key 0 is
    one more, priced as a unit of depth 32 (the arms' p_sink)."""
    s = MODELS[name]
    rep = s.heads // s.kv
    hon = cheat = plain = 0.0
    for beta in range(ctx // 32):
        if beta >= 256 and beta % 8:
            continue
        wt, k = (1 if beta < 256 else 8), 32 * (beta + 1)
        full = (k - 1) // block if block else 0
        units = ([(32, 1)] if sink else []) + ([(block, full)] if full else []) + [(k - full * block, 1)]
        mmas = folds = free = 0.0
        routes: dict = {}
        for depth, n in units:
            mm, fo, fr, rs = pv_unit(depth, rep, fold, x)
            mmas, folds, free = mmas + n * mm, folds + n * fo, free + n * fr
            for key, (dm, do) in rs.items():
                pm, po = routes.get(key, (0.0, 0.0))
                routes[key] = (pm + n * dm, po + n * do)
        th, tc = ts_best((mmas, folds + free), (mmas, folds), list(routes.values()))
        hon, cheat, plain = hon + wt * th, cheat + wt * tc, plain + wt * 16 * (k // 32)
    return hon / cheat - 1, hon / plain


def pv_block_point(name: str, ctx: int, d: Design, block: int, rates: dict = RATES) -> dict:
    """Prefill's whole-model ε_skip and γ for design `d` (its P V `pv`) with P in blocks of `block` keys, beside P V's
    own ε_skip and γ (`pv_ts`) and its share of honest time."""
    ps = parts(name, ctx, "prefill", d.noise, rates, block)
    pv = next(p for p in ps if p.component == "P·V")
    secs, skip = sum(p.seconds for p in ps), sum(p.seconds_skip for p in ps)
    return {"model": name, "ctx": ctx, "design": d.name, "block": block,
            "eps_pv": pv.seconds_skip / (pv.seconds - pv.seconds_skip), "gamma_pv": pv.seconds / pv.plain,
            "pv_share": pv.seconds / secs, "eps": skip / (secs - skip), "gamma": secs / sum(p.plain for p in ps)}


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


@dataclass
class Summary:
    model: str
    ctx: int
    phase: str
    design: str
    eps_mac: float
    eps_time: float
    gamma: float
    rows: list


def summary(name: str, ctx: int, phase: str, d: Design, rates: dict = RATES, block: int = 0) -> Summary:
    """ε_skip = skippable / (honest - skippable), by int8 products and by honest seconds, whole and per component (each
    component's skippable share over the same whole denominator); γ = honest / plain int8 time."""
    ps = parts(name, ctx, phase, d.noise, rates, block)
    macs, secs = sum(p.macs for p in ps), sum(p.seconds for p in ps)
    skip_m, skip_t = sum(p.macs_skip for p in ps), sum(p.seconds_skip for p in ps)
    rows = [{"component": p.component, "mac_share": p.macs / macs, "eps_mac": p.macs_skip / (macs - skip_m),
             "eps_time": p.seconds_skip / (secs - skip_t), "gamma": p.seconds / p.plain if p.plain else None}
            for p in ps]
    if phase == "decode":
        gemm = sum(p.seconds for p in ps if p.component in COMP)
        gamma = decode_gamma(name, ctx, gemm, rates)
    else:
        gamma = secs / sum(p.plain for p in ps)
    return Summary(name, ctx, phase, d.name, skip_m / (macs - skip_m), skip_t / (secs - skip_t), gamma, rows)


def decode_gamma(name: str, ctx: int, gemm: float, rates: dict = RATES) -> float:
    """Honest decode step / plain decode step (DRAM-bound, `decode_step`): the edited step adds the GEMMs' compute where
    it exceeds the DRAM time it hides under, and the operand commitments."""
    st = decode_step(name, ctx, rates)
    return (st["step"] + max(gemm - st["gemm_dram"], 0) + rates["decode_commit"] * st["step"]) / st["step"]


def design_skip(d: Design, rates: dict = RATES) -> Summary:
    """A design's ε_skip, the number its `[skip]` in the registry records: the most, over MODELS, CONTEXTS and PHASES,
    of the whole-model ε_skip by honest time (P in one block, every served output free), with the cell it is at."""
    return max((summary(m, c, ph, d, rates) for m in MODELS for c in CONTEXTS for ph in PHASES),
               key=lambda q: q.eps_time)


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
    for i, comp in enumerate(COMPONENTS):
        cells = []
        for q in sums.values():
            row = q.rows[i]
            g = f", γ {row['gamma']:.2f}" if row["gamma"] else ""
            cells.append(f"{100 * row['eps_mac']:.3g}% / {100 * row['eps_time']:.3g}%{g}")
        lines.append(f"| {comp} | {100 * sums[ds[0].name].rows[i]['mac_share']:.3g}% | " + " | ".join(cells) + " |")
    lines.append("| **whole** | 100% | " + " | ".join(
        f"**{100 * q.eps_mac:.3g}% / {100 * q.eps_time:.3g}%, γ {q.gamma:.2f}**" for q in sums.values()) + " |")
    return "\n".join(lines)


def design_table(skips: dict[str, Summary], ds: list[Design]) -> str:
    lines = ["Each design's ε_skip (`design_skip`, its registry `[skip]`): the most over the models, contexts and "
             "phases, by honest time", "", "| design | noise (linears, lm_head, QKᵀ, P·V) | ε_skip | at | γ there |",
             "|---|---|---:|---|---:|"]
    for d in ds:
        q = skips[d.name]
        noise = ", ".join(d.noise[p] for p in PRODUCTS)
        lines.append(f"| {d.name} ({d.title}) | {noise} | {100 * q.eps_time:.3g}% | {q.model} {q.ctx // 1024}K "
                     f"{q.phase} | {q.gamma:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json")
    p.add_argument("--md")
    a = p.parse_args(argv)
    ds = [design(n) for n in designs()]
    skips = {d.name: design_skip(d) for d in ds}
    pvs = pv_block_points()
    text = "\n".join([design_table(skips, ds), "", skip_table(ds), "",
                      *(component_table(m, c, ph, ds) + "\n" for m in MODELS for c in (2048, 32768) for ph in PHASES),
                      pv_block_table(pvs), "",
                      "Decode's zero-weight instance saves, on DRAM, this multiple of its transcripts' compute: "
                      + ", ".join(f"{m} {c // 1024}K {decode_dram(m, c):.2f}" for m in MODELS for c in CONTEXTS)])
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        sums = [summary(m, c, ph, d) for m in MODELS for c in CONTEXTS for ph in PHASES for d in ds]
        Path(a.json).write_text(json.dumps({"rates": RATES, "designs": {
            d.name: {"noise": d.noise, **asdict(skips[d.name])} for d in ds}, "skip": [asdict(q) for q in sums],
            "pv_blocks": pvs, "decode_zero_weight_gap": {f"{m}/{c}": decode_dram(m, c) for m in MODELS
                                                         for c in CONTEXTS}}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
