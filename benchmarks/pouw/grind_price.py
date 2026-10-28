"""The price of hashed seeds' grinding (Derived): ε_grind per family, model, regime and kind.

    uv run benchmarks/pouw/grind_price.py [--md OUT] [--json OUT]

A unit's seed is SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B)) with a live salt, so a cheater with T hash queries
draws T fresh seeds for a seed group and keeps the best (`Accounting.Guarantees.TS.GrindPremiumAt`). At exchange rate κ
a scope's gain is L − κ·U (lucky and unlucky savings); outside δ every seed group realizes at most

    π* = ln(T·M/δ) / θ

products, where M bounds the gain's moment at rate θ given everything but the group's own seed. ε_grind is π* over the
group's products: NOTATION's share of noised products (ε[Π]), which bounds the credit share since every family's σ_w is
below 1. Savings beyond it are at most (1 + κ)·U, and U is caught at the zero-luck sampled rate, so at the live luck
p = ¾ (`budget.P_PASS`) κ = p/(1 − p) = 3 is what `budget.py` already charges (ε_luck = ε_sample·σ_w/(1 − p)); a κ above
3 adds (κ − 3)/4·ε_luck. Each row takes the κ that minimizes that excess plus the whole model's ε_grind.

**The template's θ (`TileLuck`).** h1tile's unit is one r × c tile with its own seed, X = A + b (a fresh bit a row
entry) against W' = B + c + U (a fresh bit a column entry, U a column's keyed word); its receipt binds each 32-wide
segment's words (the fold premise, `HMu`). Given U and the data (`PrivateMoment`'s shared coins), segments and tiles are
independent, and inside a segment the c columns share the rows' bits b and each column's bits c are read by its r rows.
A word skipping w ≤ 32 products is dead (A + b = 0 on its skip set, so its value doesn't read c) with probability at
most q^w over b (q = ½ for a one-bit dither), and otherwise lucky with probability at most ½ over c
(`HMu.CoinPointMassHalf`). So M = 1 at every θ = y/r with

    q^w·e^{c·y·w} + (1 − q^w)·(½e^{yw} + ½e^{−κyw})^c ≤ 1   for w = 1 … 32,

and θ a product is y*(c, κ)/r: the premium is r/y* products a nat, the same for every contraction K. At κ = 3 and
q = ½ a one-product skip is lucky with probability exactly ¾ (dead ½, then ½ alive), the live rate, so y* = 0: κ
must exceed 3.

**h sensitivities.** At h bits a unit (compile's unit, a tile), π* = log₂(T/δ)/h tiles a seed group; h = 0.415 is
p = ¾'s log₂(4/3).

**Deployed attention's floor.** H1's QKᵀ and P·V
(one seed a matmul, a head) lose a unit's floor when one operand row's noise is matched. That is visible after the seed,
so the cheater picks the matched rows: each of N rows of b bits read by u units has moment 1 − 2^{-b} + 2^{-b}e^{θu},
and π* = min_θ (ln(T/δ) + Σ N·ln(…))/θ units (`LuckMoment` at κ = 0 with `PrivateMoment`; `visible_premium`,
`verity.core.protocols.pouw.grind`'s, which compile's seed-group check reads too). attn_noise's own form,
log₂(T/δ)/(h·N) with h the least row's noise bits over the units reading it, leaves out which rows the cheater picks:
it sits 0.5–8% below at decode and 10–21% below at prefill's QKᵀ, where the bound is smallest. Both are tabled, with
the (query head, request) pairs a decode seed must cover. A seed over several query heads draws a KV head's K rows' and
V columns' noise once, read by its heads/kv query heads (GQA). At decode the query's rows bind, so that moves nothing and
a decode seed's cover counts every (query head, request) pair; at prefill K's rows bind, so a call's floor divides by the
KV heads, not the query heads, and a prefill seed's cover counts (KV head, request) pairs.

The table shapes are `eps_budget.MODELS`; ε_sample, ε_luck and σ_w are `budget.today` per family. A seed group is a
site (one matmul of a layer at one step, every head and the step's whole batch), `compile`'s default. Assumed for every
family: h1tile's tile form (32 × 32, a seed a tile) for all kinds, attention's included (h1tile takes the s8 linears
today; attention's kinds are plugged separately).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from verity.core.protocols.pouw.grind import DELTA_BITS, FLAG, LN2, ln_t_over_delta, visible_premium

sys.path.insert(0, str(Path(__file__).parent))
import budget as BU
import eps_budget as EB

T_BITS = (80, 128)                   # the cheater's hash queries a salt's epoch
TILE = (32, 32)                      # h1tile's r × c (compile.Scheme.tile)
SEG = 32                             # a receipt word's products (h1tile's 32-wide segments)
P_ALIVE = 0.5                        # an alive word's luck over its column's coins (CoinPointMassHalf)
Q_BIT = 0.5                          # a one-bit dither's dead mass a product
Q_KEYED = 1 / 129                    # a keyed word's (PouwU129's ±64 range): the lever "raise h"
KAPPA_LIVE = BU.P_PASS / (1 - BU.P_PASS)
#: (κ, r·θ) that `Accounting.SecurityProofs.TS.TemplateCondition` proves in Lean for 32 columns and a one-bit dither
PROVED = (4.0, 1 / 256)
KAPPAS = tuple(round(3 + 0.05 * i, 2) for i in range(181))   # 3 … 12
H_BITS = (0.415, 1.44, 6.0, 16.0)    # sensitivities, bits a tile
PREFILL = (2048, 32768)
DECODE_BATCH = (1, 8, 64, 256)
DECODE_CTX = (2048, 32768)
#: attn_noise's operand rows: (rows a seed group, noise bits a row, units reading a row) for one head; a unit is a
#: QKᵀ score or a P·V output coordinate
H1_NOISE = {"q": 448, "k": 128, "p": 224}


def _log_f(w: int, y: float, c: int, kappa: float, q: float, p: float = P_ALIVE) -> float:
    """ln of TileLuck's left side at one word size w and y = rθ."""
    dead = w * math.log(q) + c * y * w
    alive_one = math.log(p * math.exp(y * w) + (1 - p) * math.exp(-kappa * y * w))
    rest = math.log1p(-(q ** w)) + c * alive_one
    m = max(dead, rest)
    return m + math.log(math.exp(dead - m) + math.exp(rest - m))


@functools.cache
def y_star(c: int, kappa: float, q: float = Q_BIT, wmax: int = SEG, p: float = P_ALIVE) -> float:
    """The largest y = rθ at which every word size meets TileLuck's condition with M = 1 (0 when none does)."""
    best = math.inf
    for w in range(1, wmax + 1):
        if q ** w + (1 - q ** w) * (p - (1 - p) * kappa) >= 0:   # no first-order slack at θ = 0
            return 0.0
        hi = 1.0 / (c * w)
        while _log_f(w, hi, c, kappa, q, p) <= 0:
            hi *= 2
        lo = 0.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _log_f(w, mid, c, kappa, q, p) <= 0 else (lo, mid)
        best = min(best, lo)
    return best


@dataclass(frozen=True)
class SeedGroup:
    """One seed group: a matmul of one layer at one step (every head, the step's batch)."""
    kind: str        # "linears", "qk", "pv"
    name: str
    macs: float
    r: int           # rows a column's coins are read by (a tile's rows in use)
    c: int           # columns sharing the rows' bits
    k: int           # contraction

    @property
    def tile_macs(self) -> float:
        return self.r * self.c * self.k

    @property
    def tiles(self) -> float:
        return self.macs / self.tile_macs


def sites(shape: EB.Shape, regime: str, n: int, ctx: int) -> list[SeedGroup]:
    """A layer's seed groups: prefill of one n-token sequence (ctx = n), or a decode step of batch n at context ctx."""
    r_lin = TILE[0] if regime == "prefill" else min(n, TILE[0])
    tokens = n
    out = [SeedGroup("linears", name, tokens * k * nn, r_lin, min(nn, TILE[1]), k) for name, k, nn in shape.linears]
    att = shape.heads * shape.d * (ctx * (ctx + 1) / 2 if regime == "prefill" else n * ctx)
    r_att = TILE[0] if regime == "prefill" else 1
    out.append(SeedGroup("qk", "qk", att, r_att, TILE[1], shape.d))
    out.append(SeedGroup("pv", "pv", att, r_att, min(shape.d, TILE[1]), ctx))
    return out


def premium(site: SeedGroup, t_bits: int, kappa: float, q: float = Q_BIT, y: float | None = None) -> float:
    """π* in products for one seed group at the template's θ = y/r (y = y* unless given; inf when θ = 0)."""
    if y is None:
        y = y_star(site.c, kappa, q)
    else:
        assert site.c == TILE[1] and kappa == PROVED[0] and q == Q_BIT, "the proved point is for 32 columns, κ = 4"
    return math.inf if y == 0 else ln_t_over_delta(t_bits) * site.r / y


def eps_by_kind(ss: list[SeedGroup], t_bits: int, kappa: float, q: float = Q_BIT,
                y: float | None = None) -> dict[str, float]:
    """ε_grind per kind and for the whole layer (the model's, since layers repeat): Σ π* / Σ products."""
    out = {}
    for kind in ("linears", "qk", "pv"):
        ks = [s for s in ss if s.kind == kind]
        out[kind] = sum(premium(s, t_bits, kappa, q, y) for s in ks) / sum(s.macs for s in ks)
    out["all"] = sum(premium(s, t_bits, kappa, q, y) for s in ss) / sum(s.macs for s in ss)
    out["worst_site"] = max(premium(s, t_bits, kappa, q, y) / s.macs for s in ss)
    return out


def eps_at_h(ss: list[SeedGroup], t_bits: int, h: float) -> dict[str, float]:
    """ε at h bits a tile: log₂(T/δ)/h tiles a seed group."""
    pi_tiles = (t_bits + DELTA_BITS) / h
    out = {}
    for kind in ("linears", "qk", "pv"):
        ks = [s for s in ss if s.kind == kind]
        out[kind] = sum(min(pi_tiles, s.tiles) * s.tile_macs for s in ks) / sum(s.macs for s in ks)
    return out


def best_kappa(ss: list[SeedGroup], t_bits: int, eps_luck: float, q: float = Q_BIT) -> tuple[float, float, float]:
    """(κ*, its excess on ε_luck, the model's ε_grind there) minimizing their sum."""
    best = None
    for kappa in KAPPAS:
        g = eps_by_kind(ss, t_bits, kappa, q)["all"]
        extra = (kappa - KAPPA_LIVE) / 4 * eps_luck
        if best is None or extra + g < best[1] + best[2]:
            best = (kappa, extra, g)
    return best


def _h1_rows(regime: str, L: int, d: int, pairs: float, gq: int) -> dict:
    """H1's rows for `pairs` (query head, request) pairs under one seed, a KV head's K rows and V columns read by gq."""
    if regime == "prefill":
        qk = ([(L / gq, H1_NOISE["k"], gq * L), (L, H1_NOISE["q"], L)], L * (L + 1) / 2)
        pv = ([(d / gq, L, gq * L), (L, H1_NOISE["p"], d)], L * d)
    else:
        qk = ([(L / gq, H1_NOISE["k"], gq), (1, H1_NOISE["q"], L)], L)
        pv = ([(d / gq, L, gq), (1, H1_NOISE["p"], d)], d)
    return {kind: ([(nr * pairs, b, u) for nr, b, u in rows], units * pairs) for kind, (rows, units) in
            (("qk", qk), ("pv", pv))}


def h1_rows(shape: EB.Shape, regime: str, n: int, ctx: int, group: str | int = "head") -> dict:
    """Per kind, H1's operand rows under one seed, (count, noise bits, units reading one), and the units it covers: one
    seed a head and request (`head`, as deployed), one a call (`call`: the layer's heads and the step's whole batch),
    one over every layer's call of the kind in a step (`step`, which no seed binding the call's roots allows: see
    `decode_lever`), or an int g: g (query head, request) pairs of one layer under one seed, each reading its own K
    rows. A call's or a step's seed draws a KV head's K rows and V columns once, read by its shape.heads/shape.kv query
    heads."""
    if isinstance(group, int):
        return _h1_rows(regime, ctx, shape.d, group, 1)
    seqs = 1 if regime == "prefill" else n
    pairs = {"head": 1, "call": shape.heads * seqs, "step": shape.heads * seqs * shape.layers}[group]
    return _h1_rows(regime, ctx, shape.d, pairs, 1 if group == "head" else shape.heads // shape.kv)


def h1_attention(shape: EB.Shape, regime: str, n: int, ctx: int, t_bits: int,
                 group: str | int = "head") -> dict[str, float]:
    """attn_noise's deployed QKᵀ and P·V floor under hashed seeds: ε per kind, the cheater matching the rows it picks
    after the seed (`visible_premium`)."""
    return {kind: min(visible_premium(rows, t_bits) / units, 1.0)
            for kind, (rows, units) in h1_rows(shape, regime, n, ctx, group).items()}


def h1_h_form(shape: EB.Shape, regime: str, n: int, ctx: int, t_bits: int,
              group: str | int = "head") -> dict[str, float]:
    """attn_noise's form of the same floor: log₂(T/δ)/(h·N), with h the least, over operand rows, of a row's noise
    bits over the units reading it, and N the units under the seed. It leaves out the cheater's choice of rows after
    the seed, which `h1_attention` charges, so it sits slightly below it."""
    return {kind: min((t_bits + DELTA_BITS) / (min(b / u for _, b, u in rows) * units), 1.0)
            for kind, (rows, units) in h1_rows(shape, regime, n, ctx, group).items()}


def decode_cover(t_bits: int, ctx: int = 2048, gmax: int = 1 << 16) -> dict[str, dict[str, int]]:
    """The least (query head, request) pairs of one layer under one decode seed that bring H1's QKᵀ and P·V under 0.1%,
    in attn_noise's h form and with the rows picked after the seed. It is the same for every model: a pair's rows don't
    depend on the head count, and the query's rows bind, so query heads sharing a KV head's rows don't change it."""
    shape = EB.MODELS["llama-3.1-8b"]
    out = {}
    for name, f in (("h_form", h1_h_form), ("visible", h1_attention)):
        out[name] = {}
        for kind in ("qk", "pv"):
            lo, hi = 1, gmax
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if f(shape, "decode", 1, ctx, t_bits, mid)[kind] <= FLAG * (1 + 1e-12) else (mid + 1, hi)
            out[name][kind] = lo
    return out


PREFILL_COVER_CTX = (16, 128, 2048, 32768)


def prefill_cover(t_bits: int, ctx: int, model: str = "llama-3.1-8b", gmax: int = 1 << 16) -> dict[str, int]:
    """The least (KV head, request) pairs of one layer under one prefill seed of ctx tokens that bring H1's QKᵀ and P·V
    under 0.1%, the rows picked after the seed. K's rows and V's columns, which a KV head's query heads share, bind
    there, so it is the same for every model."""
    shape = EB.MODELS[model]
    gq = shape.heads // shape.kv
    out = {}
    for kind in ("qk", "pv"):
        lo, hi = 1, gmax
        while lo < hi:
            mid = (lo + hi) // 2
            rows, units = _h1_rows("prefill", ctx, shape.d, mid * gq, gq)[kind]
            lo, hi = (lo, mid) if visible_premium(rows, t_bits) / units <= FLAG * (1 + 1e-12) else (mid + 1, hi)
        out[kind] = lo
    return out


DECODE_LEVER_BATCH = (1, 2, 4, 8, 16, 32, 64)
LEVER_KAPPA = 4.0


def decode_lever(shape: EB.Shape, ctx: int, b: int, t_bits: int, kappa: float = LEVER_KAPPA) -> dict:
    """A decode step's attention under each seed group. A seed binds its call's operand roots, and a step's calls run in
    order (layer ℓ + 1's operands and P·V's P exist only after the call before them), so the largest group a decode
    step has is one call: a layer's QKᵀ (or P·V) over every head and every request of the step's batch. `step` (every
    layer's call under one seed drawn from the step's inputs) needs a premise nothing states: that a cheater can't steer
    a later call's operands, after seeing its noise, without a deviation the draw catches."""
    return {"template_call": template_attention(shape, ctx, b, t_bits, kappa),
            **{f"h1_{g}": h1_attention(shape, "decode", b, ctx, t_bits, g) for g in ("head", "call", "step")}}


def template_attention(shape: EB.Shape, ctx: int, b: int, t_bits: int, kappa: float = LEVER_KAPPA) -> dict[str, float]:
    """A decode step's QKᵀ and P·V ε at the template's θ, one seed a call."""
    return {s.kind: premium(s, t_bits, kappa) / s.macs for s in sites(shape, "decode", b, ctx) if s.kind in ("qk", "pv")}


def min_batch(f, kind: str, bmax: int = 4096) -> int | None:
    """The least decode batch whose ε for `kind` is at most 0.1% (None past bmax)."""
    return next((b for b in range(1, bmax + 1) if f(b)[kind] <= FLAG), None)


def decode_levers() -> dict:
    """`decode_lever` over batches, the least batch that brings each kind under 0.1% with one seed a call, and the
    (head, request) pairs a seed must cover under H1's floor (`decode_cover`)."""
    out = {"rows": [], "min_batch": [], "cover": {t: decode_cover(t) for t in T_BITS}}
    for model, shape in EB.MODELS.items():
        for ctx in DECODE_CTX:
            for t in T_BITS:
                for b in DECODE_LEVER_BATCH:
                    out["rows"].append({"model": model, "ctx": ctx, "T_bits": t, "batch": b,
                                        **decode_lever(shape, ctx, b, t)})
                tpl = functools.partial(template_attention, shape, ctx, t_bits=t)
                h1c = functools.partial(h1_attention, shape, "decode", ctx=ctx, t_bits=t, group="call")
                h1f = functools.partial(h1_h_form, shape, "decode", ctx=ctx, t_bits=t, group="call")
                out["min_batch"].append({"model": model, "ctx": ctx, "T_bits": t,
                                         **{f"template_{k}": min_batch(tpl, k) for k in ("qk", "pv")},
                                         **{f"h1_call_{k}": min_batch(h1c, k) for k in ("qk", "pv")},
                                         **{f"h1_call_h_form_{k}": min_batch(h1f, k) for k in ("qk", "pv")},
                                         "h1_step_b1": h1_attention(shape, "decode", 1, ctx, t, "step")})
    return out


def regimes():
    for L in PREFILL:
        yield "prefill", L, L
    for ctx in DECODE_CTX:
        for b in DECODE_BATCH:
            yield "decode", b, ctx


def label(regime: str, n: int, ctx: int) -> str:
    return f"prefill {n // 1024}K" if regime == "prefill" else f"decode b{n} @{ctx // 1024}K"


def table() -> dict:
    fams = {f.name: BU.today(f) for f in BU.FAMILIES}
    rows, proved, sens, h1, templ = [], [], [], [], {}
    for kappa in (3, 3.25, 3.5, 4, 5, 6, 8, 12):
        templ[f"c={TILE[1]} κ={kappa}"] = {"y_star": y_star(TILE[1], kappa),
                                           "y_star_keyed": y_star(TILE[1], kappa, Q_KEYED)}
    for model, shape in EB.MODELS.items():
        for regime, n, ctx in regimes():
            ss = sites(shape, regime, n, ctx)
            for t in T_BITS:
                for fam, pt in fams.items():
                    kappa, extra, _ = best_kappa(ss, t, pt.eps_luck)
                    e = eps_by_kind(ss, t, kappa)
                    k_kappa, k_extra, k_g = best_kappa(ss, t, pt.eps_luck, Q_KEYED)
                    rows.append({"family": fam, "model": model, "regime": label(regime, n, ctx), "T_bits": t,
                                 "kappa": kappa, "kappa_excess": extra, **{k: e[k] for k in e},
                                 "eps_luck": pt.eps_luck, "eps_sample": pt.eps_sample,
                                 "keyed": {"kappa": k_kappa, "kappa_excess": k_extra, "all": k_g},
                                 "flag": [k for k in ("linears", "qk", "pv") if e[k] > FLAG]})
                p = eps_by_kind(ss, t, PROVED[0], y=PROVED[1])
                proved.append({"model": model, "regime": label(regime, n, ctx), "T_bits": t, **p,
                               "flag": [k for k in ("linears", "qk", "pv") if p[k] > FLAG]})
                sens.append({"model": model, "regime": label(regime, n, ctx), "T_bits": t,
                             **{f"h={h}": eps_at_h(ss, t, h) for h in H_BITS}})
                h1.append({"model": model, "regime": label(regime, n, ctx), "T_bits": t,
                           "head": h1_attention(shape, regime, n, ctx, t, "head"),
                           "call": h1_attention(shape, regime, n, ctx, t, "call"),
                           "head_h_form": h1_h_form(shape, regime, n, ctx, t, "head"),
                           "call_h_form": h1_h_form(shape, regime, n, ctx, t, "call")})
    excess = {fam: (PROVED[0] - KAPPA_LIVE) / 4 * pt.eps_luck for fam, pt in fams.items()}
    return {"rows": rows, "proved": {"rows": proved, "kappa_excess": excess}, "sensitivity": sens,
            "h1_attention": h1, "template": templ, "levers": levers(), "decode_levers": decode_levers(),
            "prefill_cover": {t: {ctx: prefill_cover(t, ctx) for ctx in PREFILL_COVER_CTX} for t in T_BITS}}


def levers() -> dict:
    """Minimum cover (tiles a seed group for ε_grind ≤ 0.1%) at the template's θ and at each h; salt refresh and slow
    hashes as the log factors they are."""
    out = {"cover_at_template": {}, "cover_at_h": {}, "log_factors": {}}
    c = TILE[1]
    for kappa, how in ((4, "y*"), (6, "y*"), (PROVED[0], "proved")):
        for name, k in {"linear K=4096": 4096, "linear K=14336": 14336, "QKᵀ K=d=128": 128, "P·V K=2048": 2048,
                        "P·V K=32768": 32768}.items():
            y = y_star(c, kappa) if how == "y*" else PROVED[1]
            pi_tiles = ln_t_over_delta(128) / (y * c * k)
            out["cover_at_template"][f"{name}, κ={kappa:g}, {how}"] = {
                "pi_star_tiles": pi_tiles, "cover": pi_tiles / FLAG, "h_tile_bits": y * c * k / LN2}
    for h in H_BITS:
        out["cover_at_h"][f"h={h}"] = {t: (t + DELTA_BITS) / h / FLAG for t in T_BITS}
    base = 128 + DELTA_BITS
    for name, bits in {"salt each 12.6 s round, 2^60 queries in it": 60, "T = 2^80": 80}.items():
        out["log_factors"][name] = (bits + DELTA_BITS) / base
    for k in (10, 20, 40):
        out["log_factors"][f"seed hash 2^{k} times slower"] = (base - k) / base
    return out


def _pct(x: float) -> str:
    if math.isinf(x):
        return "∞"
    return f"{100 * x:.4f}%" if x < 1e-2 else f"{100 * x:.2f}%"


def _cell(x: float) -> str:
    return f"**{_pct(x)}**" if x > FLAG else _pct(x)


def render(t: dict) -> str:
    lines = ["# ε_grind under hashed seeds (GRIND, decision 112)", "",
             f"δ = 2^-{DELTA_BITS}; tile {TILE[0]} × {TILE[1]}, segments of {SEG}; bold is over 0.1%.", ""]
    lines += ["## Per family, model and regime (the template's θ at the row's κ*)", "",
              ("| family | model | regime | T | κ* | (κ*−3)/4·ε_luck | linears | QKᵀ | P·V | all kinds | worst site | "
               "keyed row dither: κ*, excess, all kinds |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        k = r["keyed"]
        lines.append(f"| {r['family']} | {r['model']} | {r['regime']} | 2^{r['T_bits']} | {r['kappa']} | "
                     f"{_pct(r['kappa_excess'])} | {_cell(r['linears'])} | {_cell(r['qk'])} | {_cell(r['pv'])} | "
                     f"{_cell(r['all'])} | {_cell(r['worst_site'])} | "
                     f"{k['kappa']}, {_pct(k['kappa_excess'])}, {_cell(k['all'])} |")
    pr = t["proved"]
    lines += ["", f"## At the point Lean proves (`TemplateMoment`: κ = {PROVED[0]:g}, r·θ = 1/{1 / PROVED[1]:.0f})", "",
              "κ = 4's excess (κ − 3)/4·ε_luck: "
              + ", ".join(f"{fam} {_pct(x)}" for fam, x in pr["kappa_excess"].items()) + ".", "",
              "| model | regime | T | linears | QKᵀ | P·V | all kinds | worst site |", "|---|---|---|---|---|---|---|---|"]
    for r in pr["rows"]:
        lines.append(f"| {r['model']} | {r['regime']} | 2^{r['T_bits']} | {_cell(r['linears'])} | {_cell(r['qk'])} | "
                     f"{_cell(r['pv'])} | {_cell(r['all'])} | {_cell(r['worst_site'])} |")
    lines += ["", "## Sensitivity: h bits a tile (log₂(T/δ)/h tiles a seed group)", "",
              "| model | regime | T | " + " | ".join(f"h={h}: lin / QKᵀ / P·V" for h in H_BITS) + " |",
              "|---|---|---|" + "---|" * len(H_BITS)]
    for s in t["sensitivity"]:
        cells = " | ".join(" / ".join(_cell(s[f"h={h}"][k]) for k in ("linears", "qk", "pv")) for h in H_BITS)
        lines.append(f"| {s['model']} | {s['regime']} | 2^{s['T_bits']} | {cells} |")
    lines += ["", "## Deployed H1 attention's floor (ATTN-NOISE §5's h), visible matched rows", "",
              ("Each cell is the rows the cheater picks after the seed (`visible_premium`), then ATTN-NOISE's h form "
               "log₂(T/δ)/(h·N) (QKᵀ: h = d/L at prefill, 448/L at decode; P·V: h = 1 at prefill, 1.75 at decode), which "
               "leaves the pick out. A call's seed draws a KV head's K rows and V columns once, read by its heads/kv "
               "query heads (GQA), so its prefill h is those over heads/kv; at decode the query's rows bind."), "",
              "| model | regime | T | QKᵀ, a seed a head | P·V, a seed a head | QKᵀ, a seed a call | P·V, a seed a call |",
              "|---|---|---|---|---|---|---|"]
    for a in t["h1_attention"]:
        cells = " | ".join(f"{_cell(a[g][k])} / {_cell(a[g + '_h_form'][k])}"
                           for g, k in (("head", "qk"), ("head", "pv"), ("call", "qk"), ("call", "pv")))
        lines.append(f"| {a['model']} | {a['regime']} | 2^{a['T_bits']} | {cells} |")
    d = t["decode_levers"]
    lines += ["", "## Decode's seed groups (attention)", "",
              ("A seed binds its call's operand roots and a step's calls run in order, so the largest group a decode step "
               "has is one call: a layer's QKᵀ (or P·V) over every head and every request of the step's batch. "
               "\"Every layer\" (one seed over the step's calls of a kind) needs a premise nothing states. The template "
               f"at κ = {LEVER_KAPPA}."), "",
              ("| model | ctx | T | batch | template, a call: QKᵀ / P·V | H1, a head: QKᵀ / P·V | H1, a call: QKᵀ / P·V "
               "| H1, every layer: QKᵀ / P·V |"), "|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        cells = " | ".join(f"{_cell(r[g]['qk'])} / {_cell(r[g]['pv'])}"
                           for g in ("template_call", "h1_head", "h1_call", "h1_step"))
        lines.append(f"| {r['model']} | {r['ctx'] // 1024}K | 2^{r['T_bits']} | {r['batch']} | {cells} |")
    lines += ["", "Least decode batch for 0.1% with one seed a call (H1: rows picked after the seed / h form):", "",
              "| model | ctx | T | template QKᵀ | template P·V | H1 QKᵀ | H1 P·V | H1, every layer, batch 1: QKᵀ / P·V |",
              "|---|---|---|---|---|---|---|---|"]
    for m in d["min_batch"]:
        s = m["h1_step_b1"]
        lines.append(f"| {m['model']} | {m['ctx'] // 1024}K | 2^{m['T_bits']} | {m['template_qk']} | "
                     f"{m['template_pv']} | {m['h1_call_qk']} / {m['h1_call_h_form_qk']} | "
                     f"{m['h1_call_pv']} / {m['h1_call_h_form_pv']} | {_cell(s['qk'])} / {_cell(s['pv'])} |")
    heads = {m: s.heads for m, s in EB.MODELS.items()}
    lines += ["", ("Minimum seed coverage for decode attention under H1's floor: the (query head, request) pairs of one "
                   "layer a seed must cover for 0.1%, the same for every model and whether or not query heads share a "
                   "KV head's rows (the query's rows bind), and the batch that is for one seed a call (the layer's "
                   "heads × the batch) or one seed a (layer, head) over the batch (rows picked after the seed / h "
                   "form):"), "",
              "| T | kind | pairs a seed | " + " | ".join(f"batch, a call, {m} ({h} heads)" for m, h in heads.items())
              + " | batch, a (layer, head) |", "|---|---|---|" + "---|" * len(heads) + "---|"]
    for tb, cov in d["cover"].items():
        for kind, name in (("qk", "QKᵀ"), ("pv", "P·V")):
            v, f = cov["visible"][kind], cov["h_form"][kind]
            per = " | ".join(f"{math.ceil(v / h)} / {math.ceil(f / h)}" for h in heads.values())
            lines.append(f"| 2^{tb} | {name} | {v} / {f} | {per} | {v} / {f} |")
    lines += ["", ("Minimum seed coverage for prefill attention under H1's floor: the (KV head, request) pairs of one "
                   "layer a prefill seed must cover for 0.1% (K's rows and V's columns, shared by a KV head's query "
                   "heads, bind), the same for every model:"), "",
              "| T | " + " | ".join(f"{c} tokens: QKᵀ / P·V" for c in PREFILL_COVER_CTX) + " |",
              "|---|" + "---|" * len(PREFILL_COVER_CTX)]
    for tb, by in t["prefill_cover"].items():
        lines.append(f"| 2^{tb} | " + " | ".join(f"{c['qk']} / {c['pv']}" for c in by.values()) + " |")
    lines += ["", "## The template: y* = rθ (nats a product × rows)", "", "| c, κ | y* (one-bit dither) | y* (keyed word) |",
              "|---|---|---|"]
    for k, v in t["template"].items():
        lines.append(f"| {k} | {v['y_star']:.5f} | {v['y_star_keyed']:.5f} |")
    lines += ["", "## Levers", "", "Minimum cover at the template's θ, T = 2^128 (in tiles, the same for every r):", "",
              "| tile | π* (tiles a group) | cover for 0.1% | h (bits a tile) |", "|---|---|---|---|"]
    for k, v in t["levers"]["cover_at_template"].items():
        lines.append(f"| {k} | {v['pi_star_tiles']:.4g} | {v['cover']:.4g} | {v['h_tile_bits']:.4g} |")
    lines += ["", "| h (bits a tile) | cover for 0.1% at T = 2^80 | at 2^128 |", "|---|---|---|"]
    for k, v in t["levers"]["cover_at_h"].items():
        lines.append(f"| {k} | {v[80]:.4g} | {v[128]:.4g} |")
    lines += ["", "| non-lever | factor on π* |", "|---|---|"]
    for k, v in t["levers"]["log_factors"].items():
        lines.append(f"| {k} | {v:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    t = table()
    md = render(t)
    if a.md:
        a.md.write_text(md)
    if a.json:
        a.json.write_text(json.dumps(t, indent=1, default=str))
    if not a.md:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
