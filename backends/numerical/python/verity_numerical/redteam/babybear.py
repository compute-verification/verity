"""The BabyBear embedding of the WP2 checker: the rules every lane must follow, derived mechanically.

BabyBear is ``p = 2^31 - 2^27 + 1`` (30.9 bits).  A gadget predicate stated over the integers stays sound when read
modulo ``p`` iff the integer value of what it constrains cannot leave a window of width ``p``:

    eq(a, b)            a - b in (-p, p)                     given every hint pinned to its declared range
    rng(x, lo, hi)      x a hint: hi - lo <= p (one lookup pins one representative)
                        x an expression: its integer interval lies in [hi - p, lo + p)
    lt_pow(x, ipow, B)  the product realisation x * ipow < 2^B needs 2^(2B) <= p (B <= 15); above that the check is
                        realised *directly* against the table's bound column bnd = 2^B / ipow as
                        rng(bnd - 1 - x, 0, 2^B), sound iff the interval of bnd - 1 - x lies in [2^B - p, p)

:class:`BoundCtx` runs every gadget's ``check`` on interval values (inputs and hints at their declared ranges, table
columns at their row extremes) and records, per predicate, the integer interval of what it constrains and whether the
window condition holds -- so the list of predicates that need a rule is *computed*, not read off ``fields.py``.  Two
passes: the first collects the bounds the tight-quotient checks impose on products (``q * pow < 2^W``), the second
uses them where the same product appears in an identity (the alignment divisions are exactly this case).

The rules (:data:`RULES`), each droppable so that z3 can show it is necessary:

    LIMB16      32-bit words (the FP32 accumulator in, the FP32 word out, the epilogue's input) are two public 16-bit
                limbs; the word identities are stated per limb (the fraction gets limb hints f_lo (16) / f_hi (7))
    LT_DIRECT   every lt_pow of width > 15 is the direct check against the bound column (POW: pow <-> ipow, ALIGN:
                sr <-> isr, LEAD: new column span, NORM: unchanged, its width is 7)
    QUOT_TIGHT  the tight quotient bounds of the alignments (already in the integer checker; over BabyBear they are
                what keeps q * pow + r = 2M inside (-p, p))
    NORM_MBND   NORM gains mbnd = 2^s_bits / div and the check rng(mbnd - 1 - M', 0, 2^s_bits): the quotient of the
                normalisation division is tight relative to its dividend (the +p shadow M' = (v + p) / div is excluded)
    NORM_VBND   NORM gains vbnd = 2^acc_sig / mul (2^s_bits when k < 0) and rng(vbnd - 1 - v, 0, 2^s_bits): v * mul
                cannot wrap.  Redundant once leading_bit pins l = bitlength(v) (z3 shows it); kept as a named option
    V_PREMISE   the signed-sum wire v is a linear form of ranged hints with |v| <= s_max < 2^29.1; leading_bit and
                normalise are stated for v in [0, s_max] (the composition guarantees it; a per-gadget proof must assume it)

:class:`BabyBearCtx` evaluates the embedded checker on Python ints with mod-p semantics (the "BabyBear-rules
checker"); :func:`check_step_bb` / :func:`check_vu_bb` are the embedded unit and VU; ``python -m
verity_numerical.redteam.babybear`` writes ``backends/numerical/reports/babybear_rules.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import IntCtx, Table
from verity_numerical.checker.gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Contract,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Pack,
    Product,
    SignedSum,
    StateOut,
)
from verity_numerical.checker.tables import (
    align_table,
    lead_table,
    norm_table,
    pow_table,
)

__all__ = [
    "LIMB",
    "RULES",
    "BabyBearCtx",
    "BoundCtx",
    "Embedding",
    "P",
    "PredicateBound",
    "analyse",
    "check_step_bb",
    "check_vu_bb",
    "gen_step_bb",
    "gen_vu_bb",
    "lead_table_bb",
    "limbs",
    "norm_table_bb",
    "rules_report",
    "unlimb",
]

P = 2**31 - 2**27 + 1
LIMB = 16
RULES: tuple[str, ...] = ("LIMB16", "LT_DIRECT", "QUOT_TIGHT", "NORM_MBND", "NORM_VBND", "V_PREMISE")
ALL_RULES = frozenset(RULES)


def limbs(word: int, bits: int = 32) -> tuple[int, int]:
    """``(lo, hi)`` 16-bit limbs of a word."""
    return word & (2**LIMB - 1), word >> LIMB


def unlimb(lo, hi):
    return hi * 2**LIMB + lo


# ---------------------------------------------------------------------------------------------------------------------
# Tables with the bound columns the direct realisations need
# ---------------------------------------------------------------------------------------------------------------------


@cache
def lead_table_bb(p: Params) -> Table:
    """LEAD with a third column ``span`` = 2^s_bits / ispan: the bound of ``v - lo`` (1 at l = 0, 2^(l-1) otherwise)."""
    base = lead_table(p)

    def fn(l: int) -> tuple[int, int, int]:
        lo, ispan = base.fn(l)
        return lo, ispan, 2**p.s_bits // ispan

    return Table("LEAD", base.lo, base.hi, ("lo", "ispan", "span"), fn)


@cache
def norm_table_bb(p: Params) -> Table:
    """NORM with ``mbnd`` = 2^s_bits / div (the tight bound of M' = floor(v * mul / div) when v < 2^s_bits and mul = 1)
    and ``vbnd`` = 2^acc_sig / mul when k >= 0 (M' = v * mul < 2^acc_sig forces v < 2^(acc_sig - k)), 2^s_bits when k < 0."""
    base = norm_table(p)

    def fn(k: int) -> tuple[int, int, int, int, int]:
        mul, div, idiv = base.fn(k)
        mbnd = 2**p.s_bits // div
        vbnd = 2**p.acc_sig // mul if k >= 0 else 2**p.s_bits
        return mul, div, idiv, mbnd, vbnd

    return Table("NORM", base.lo, base.hi, ("mul", "div", "idiv", "mbnd", "vbnd"), fn)


def partner_columns(table: Table, bits: int) -> dict[str, str]:
    """Columns ``a -> b`` of a table with ``a * b = 2^bits`` on every row (the bound column of a direct check)."""
    rows = [table.fn(k) for k in range(table.lo, table.hi)]
    out = {}
    for i, a in enumerate(table.outs):
        for j, b in enumerate(table.outs):
            if i != j and all(r[i] * r[j] == 2**bits for r in rows):
                out[a] = b
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The embedded gadgets
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Embedding:
    """The BabyBear checker under a rule set.  ``rules`` is the set of active rules; dropping one restates the
    affected predicate as the integer checker has it (read modulo p, that is the naive embedding)."""

    rules: frozenset = ALL_RULES

    def has(self, rule: str) -> bool:
        if rule not in RULES:
            raise ValueError(rule)
        return rule in self.rules

    # -- the variable-bound check -----------------------------------------------------------------------------------
    @staticmethod
    def bound(ctx, name: str, x, bnd, bits: int) -> None:
        """The direct variable-bound check ``x < bnd`` (``bnd`` a table column) as ``rng(bnd - 1 - x, 0, 2^bits)``."""
        ctx.rng(name, bnd - 1 - x, 0, 2**bits)
        note = getattr(ctx, "note_bound", None)
        if note is not None:  # the interval analysis learns the product bounds the check imposes
            note(x, bnd)

    def lt(self, ctx, name: str, x, ipow, bnd, bits: int, *, nonneg: bool = False) -> None:
        """``0 <= x < 2^bits / ipow`` (= ``x < bnd``): the direct realisation when the rule is on and the width needs
        it.  The direct form only bounds ``x`` from above; ``nonneg`` adds ``rng(x, 0, 2^bits)`` for an ``x`` that is
        not itself a range-checked hint (the leading-bit window ``v - lo``; found by z3, run 1)."""
        if self.has("LT_DIRECT") and 2 ** (2 * bits) > P:
            if nonneg:
                ctx.rng(f"{name}.nonneg", x, 0, 2**bits)
            self.bound(ctx, name, x, bnd, bits)
        else:
            ctx.lt_pow(name, x, ipow, bits)

    # -- words: two limbs -----------------------------------------------------------------------------------------------
    def fp32_decode(self, ctx, p: Params, c, h: dict, tag: str = "acc"):
        """``Fp32Decode`` on the limbed word ``c = (c_lo, c_hi)``; hints gain ``f_lo``, ``f_hi`` (the fraction's limbs)."""
        s, t, nz, z = h["s"], h["t"], h["nz"], h["z"]
        ctx.boolean(f"{tag}.s", s)
        ctx.rng(f"{tag}.t", t, 0, p.t_finite)
        ctx.boolean(f"{tag}.nz", nz)
        ctx.boolean(f"{tag}.z", z)
        if self.has("LIMB16"):
            f_lo, f_hi = h["f_lo"], h["f_hi"]
            ctx.rng(f"{tag}.f_lo", f_lo, 0, 2**LIMB)
            ctx.rng(f"{tag}.f_hi", f_hi, 0, 2 ** (p.acc_frac - LIMB))
            f = f_hi * 2**LIMB + f_lo
            c_lo, c_hi = c
            ctx.eq(f"{tag}.word_lo", f_lo, c_lo)
            ctx.eq(f"{tag}.word_hi", s * 2 ** (p.acc_word_bits - 1 - LIMB) + t * 2 ** (p.acc_frac - LIMB) + f_hi, c_hi)
        else:
            f = h["f"]
            ctx.rng(f"{tag}.f", f, 0, 2**p.acc_frac)
            ctx.eq(f"{tag}.word", s * 2 ** (p.acc_word_bits - 1) + t * 2**p.acc_frac + f, unlimb(*c))
        ctx.eq(f"{tag}.nz0", (1 - nz) * t, 0)
        ctx.rng(f"{tag}.nz1", t - nz, 0, p.t_finite - 1)
        M = f + nz * 2**p.acc_frac
        ctx.eq(f"{tag}.z0", z * M, 0)
        ctx.rng(f"{tag}.z1", M + z - 1, 0, 2**p.acc_sig)
        e = t - nz - (p.bias - 1)
        e_hat = e + z * (p.floor - e)
        return s, e_hat, M

    def pack(self, ctx, p: Params, state, ovfs: list, h: dict, tag: str = "pack"):
        """``Pack`` returning the word as limbs ``(y_lo, y_hi)``; hints ``low_lo`` (16) / ``low_hi`` (7) replace ``low``."""
        s, e1, M, z = state
        hidden = h["hidden"]
        ctx.boolean(f"{tag}.hidden", hidden)
        if self.has("LIMB16"):
            low_lo, low_hi = h["low_lo"], h["low_hi"]
            ctx.rng(f"{tag}.low_lo", low_lo, 0, 2**LIMB)
            ctx.rng(f"{tag}.low_hi", low_hi, 0, 2 ** (p.acc_frac - LIMB))
            low = low_hi * 2**LIMB + low_lo
        else:
            low = h["low"]
            ctx.rng(f"{tag}.low", low, 0, 2**p.acc_frac)
        ctx.eq(f"{tag}.M", hidden * 2**p.acc_frac + low, M)
        fld = e1 + (p.bias - 1) + hidden
        any_ovf, inf_sign, none_yet = 0, 0, 1
        for ovf_g, sgn_g in ovfs:
            first = none_yet * ovf_g
            any_ovf = any_ovf + first
            inf_sign = inf_sign + first * sgn_g
            none_yet = none_yet - none_yet * ovf_g
        if self.has("LIMB16"):
            fin_lo = low_lo
            fin_hi = s * 2 ** (p.acc_word_bits - 1 - LIMB) + fld * 2 ** (p.acc_frac - LIMB) + low_hi
            inf_lo = 0
            inf_hi = inf_sign * 2 ** (p.acc_word_bits - 1 - LIMB) + p.t_finite * 2 ** (p.acc_frac - LIMB)
            return (
                any_ovf * inf_lo + (1 - any_ovf) * (1 - z) * fin_lo,
                any_ovf * inf_hi + (1 - any_ovf) * (1 - z) * fin_hi,
            )
        fin = s * 2 ** (p.acc_word_bits - 1) + fld * 2**p.acc_frac + low
        inf = inf_sign * 2 ** (p.acc_word_bits - 1) + p.t_finite * 2**p.acc_frac
        return (any_ovf * inf + (1 - any_ovf) * (1 - z) * fin, 0)

    def epilogue(self, ctx, p: Params, y32, h: dict, tag: str = "cast"):
        """``Epilogue`` on the limbed word: ``hi`` and ``lo`` *are* the public limbs (L = acc_frac - frac = 16)."""
        L = p.acc_frac - p.frac
        if L != LIMB:
            raise Reject("the epilogue limb split assumes acc_frac - frac = 16")
        hi2, lsb, carry, rem = (h[k] for k in ("hi2", "lsb", "carry", "rem"))
        if self.has("LIMB16"):
            lo, hi = y32
        else:
            hi, lo = h["hi"], h["lo"]
            ctx.rng(f"{tag}.hi", hi, 0, 2**p.word_bits)
            ctx.rng(f"{tag}.lo", lo, 0, 2**L)
            ctx.eq(f"{tag}.split", hi * 2**L + lo, unlimb(*y32))
        ctx.rng(f"{tag}.hi2", hi2, 0, 2 ** (p.word_bits - 1))
        ctx.boolean(f"{tag}.lsb", lsb)
        ctx.boolean(f"{tag}.carry", carry)
        ctx.rng(f"{tag}.rem", rem, 0, 2**L)
        ctx.eq(f"{tag}.lsb_split", 2 * hi2 + lsb, hi)
        ctx.eq(f"{tag}.round", carry * 2**L + rem, lo + lsb + (2 ** (L - 1) - 1))
        y16 = hi + carry
        ctx.rng(f"{tag}.y16", y16, 0, 2**p.word_bits)
        return y16

    # -- the variable-bound gadgets -------------------------------------------------------------------------------------
    def align_state(self, ctx, p: Params, M, d, h: dict, tag: str = "align_state"):
        q, r = h["q"], h["r"]
        W = p.acc_bits
        ctx.rng(f"{tag}.q", q, 0, 2**W)
        ctx.rng(f"{tag}.r", r, 0, 2**W)
        pow_, ipow = ctx.lookup(f"{tag}.pow", pow_table(p), d)
        ctx.eq(f"{tag}.div", q * pow_ + r, M * 2**p.acc_shift)
        self.lt(ctx, f"{tag}.rem", r, ipow, pow_, W)
        if self.has("QUOT_TIGHT"):
            self.lt(ctx, f"{tag}.quot", q, pow_, ipow, W)
        return q

    def align_product(self, ctx, p: Params, u, d, h: dict, tag: str = "align_prod"):
        q, r = h["q"], h["r"]
        U = 2 * p.sig
        ctx.rng(f"{tag}.q", q, 0, 2**U)
        ctx.rng(f"{tag}.r", r, 0, 2**U)
        sl, sr, isr = ctx.lookup(f"{tag}.align", align_table(p), d)
        ctx.eq(f"{tag}.div", q * sr + r, u)
        self.lt(ctx, f"{tag}.rem", r, isr, sr, U)
        if self.has("QUOT_TIGHT"):
            self.lt(ctx, f"{tag}.quot", q, sr, isr, U)
        return q * sl

    def leading_bit(self, ctx, p: Params, v, h: dict, tag: str = "lead"):
        l = h["l"]
        lo, ispan, span = ctx.lookup(f"{tag}.lead", lead_table_bb(p), l)
        self.lt(ctx, f"{tag}.window", v - lo, ispan, span, p.s_bits, nonneg=True)
        return l

    def normalise(self, ctx, p: Params, E, l, v, h: dict, tag: str = "norm"):
        clamp, M, r = h["clamp"], h["M"], h["r"]
        ctx.boolean(f"{tag}.clamp", clamp)
        ctx.rng(f"{tag}.M", M, 0, 2**p.acc_sig)
        ctx.rng(f"{tag}.r", r, 0, 2**p.norm_rem_bits)
        e_raw = E + l - p.width
        ctx.rng(f"{tag}.clamp_proof", clamp * (p.e_min - 1 - e_raw) + (1 - clamp) * (e_raw - p.e_min), 0, p.cmp_rows)
        e1 = e_raw + clamp * (p.e_min - e_raw)
        k = E - e1 + p.acc_sig - p.width
        mul, div, idiv, mbnd, vbnd = ctx.lookup(f"{tag}.norm", norm_table_bb(p), k)
        ctx.eq(f"{tag}.div", div * M + r, v * mul)
        self.lt(ctx, f"{tag}.rem", r, idiv, div, p.norm_rem_bits)
        if self.has("NORM_MBND"):
            self.bound(ctx, f"{tag}.mbnd", M, mbnd, p.s_bits)
        if self.has("NORM_VBND"):
            self.bound(ctx, f"{tag}.vbnd", v, vbnd, p.s_bits)
        return e1, M

    # -- contracts of the changed gadgets (the others are the integer checker's) --------------------------------------
    def contract(self, gadget: str, p: Params, m: int | None = None) -> Contract:
        base = {
            "bf16_decode": Bf16Decode.contract,
            "product": Product.contract,
            "state_out": StateOut.contract,
            "signed_sum": lambda p: SignedSum.contract(p, m),
            "group_max": lambda p: GroupMax.contract(p, m),
        }
        if gadget in base:
            return base[gadget](p)
        v_hi = p.s_max + 1 if self.has("V_PREMISE") else 2**p.s_bits
        if gadget == "fp32_decode":
            c = Fp32Decode.contract(p)
            if not self.has("LIMB16"):
                return Contract(c.gadget, {"c_lo": (0, 2**LIMB), "c_hi": (0, 2**LIMB)}, c.hints, c.checked, c.outputs)
            hints = {k: v for k, v in c.hints.items() if k != "f"}
            hints.update({"f_lo": (0, 2**LIMB), "f_hi": (0, 2 ** (p.acc_frac - LIMB))})
            return Contract(c.gadget, {"c_lo": (0, 2**LIMB), "c_hi": (0, 2**LIMB)}, hints, c.checked, c.outputs)
        if gadget == "pack":
            c = Pack.contract(p)
            outs = {"y_lo": (0, 2**LIMB), "y_hi": (0, 2**LIMB)}
            if not self.has("LIMB16"):
                return Contract(c.gadget, c.inputs, c.hints, c.checked, outs)
            hints = {k: v for k, v in c.hints.items() if k != "low"}
            hints.update({"low_lo": (0, 2**LIMB), "low_hi": (0, 2 ** (p.acc_frac - LIMB))})
            return Contract(c.gadget, c.inputs, hints, c.checked, outs)
        if gadget == "epilogue":
            c = Epilogue.contract(p)
            ins = {"y32_lo": (0, 2**LIMB), "y32_hi": (0, 2**LIMB)}
            if not self.has("LIMB16"):
                return Contract(c.gadget, ins, c.hints, c.checked, c.outputs)
            hints = {k: v for k, v in c.hints.items() if k not in ("hi", "lo")}
            return Contract(c.gadget, ins, hints, c.checked, c.outputs)
        if gadget == "leading_bit":
            c = LeadingBit.contract(p)
            return Contract(c.gadget, {"v": (0, v_hi)}, c.hints, c.checked, c.outputs, ("LEAD",))
        if gadget == "normalise":
            c = Normalise.contract(p)
            ins = dict(c.inputs)
            ins["v"] = (0, v_hi)
            checked = dict(c.checked)
            if self.has("NORM_MBND"):
                checked["mbnd-1-M"] = (0, 2**p.s_bits)
            if self.has("NORM_VBND"):
                checked["vbnd-1-v"] = (0, 2**p.s_bits)
            return Contract(c.gadget, ins, c.hints, checked, c.outputs, ("NORM",))
        if gadget == "align_state":
            return AlignState.contract(p)
        if gadget == "align_product":
            return AlignProduct.contract(p)
        raise KeyError(gadget)


DEFAULT = Embedding()


# ---------------------------------------------------------------------------------------------------------------------
# The embedded unit and VU (hints in the integer checker's tree shape, with the limb hints added)
# ---------------------------------------------------------------------------------------------------------------------


def _k(p: Params) -> int:
    return sum(p.groups)


def gen_step_bb(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> dict:
    """The integer checker's honest hints plus the limb hints (``acc.f_lo/f_hi``, ``pack.low_lo/low_hi``)."""
    from verity_numerical.checker.step import gen_step

    h = gen_step(p, c, list(a), list(b))
    h["acc"]["f_lo"], h["acc"]["f_hi"] = limbs(h["acc"]["f"])
    h["pack"]["low_lo"], h["pack"]["low_hi"] = limbs(h["pack"]["low"])
    return h


def check_step_bb(ctx, p: Params, c, a: Sequence, b: Sequence, h: dict, emb: Embedding = DEFAULT):
    """Every predicate of one unit over BabyBear; ``c`` is the limb pair; returns the output word's limbs."""
    k = _k(p)
    if len(a) != k or len(b) != k or len(h["a"]) != k or len(h["b"]) != k or len(h["prod"]) != k:
        raise Reject(f"{k} operand pairs expected")
    if len(h["groups"]) != len(p.groups):
        raise Reject(f"{len(p.groups)} groups expected")
    ctx.rng("x.c_lo", c[0], 0, 2**LIMB)
    ctx.rng("x.c_hi", c[1], 0, 2**LIMB)
    for i in range(k):
        ctx.rng(f"x.a[{i}]", a[i], 0, 2**p.word_bits)
        ctx.rng(f"x.b[{i}]", b[i], 0, 2**p.word_bits)
    state = emb.fp32_decode(ctx, p, c, h["acc"])
    dec_a = [Bf16Decode.check(ctx, p, a[i], h["a"][i], f"a[{i}]") for i in range(k)]
    dec_b = [Bf16Decode.check(ctx, p, b[i], h["b"][i], f"b[{i}]") for i in range(k)]
    prods = [Product.check(ctx, p, dec_a[i], dec_b[i], h["prod"][i], f"prod[{i}]") for i in range(k)]
    ovfs = []
    start = 0
    z = 0
    for g, size in enumerate(p.groups):
        group = prods[start : start + size]
        start += size
        hg = h["groups"][g]
        if len(hg["prod_align"]) != size:
            raise Reject(f"group {g}: {size} product alignments expected")
        tag = f"g{g}"
        cands = [state[1]] + [e for _, _, e in group]
        E, ds = GroupMax.check(ctx, p, cands, hg["max"], f"{tag}.max")
        q_acc = emb.align_state(ctx, p, state[2], ds[0], hg["acc_align"], f"{tag}.align_state")
        q_prods = [
            emb.align_product(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]")
            for i, ((u, _, _), d, hq) in enumerate(zip(group, ds[1:], hg["prod_align"]))
        ]
        terms = [(state[0], q_acc)] + [(s, q) for (_, s, _), q in zip(group, q_prods)]
        sgn, v = SignedSum.check(ctx, p, terms, hg["sum"], f"{tag}.sum")
        l = emb.leading_bit(ctx, p, v, hg["lead"], f"{tag}.lead")
        e1, M = emb.normalise(ctx, p, E, l, v, hg["norm"], f"{tag}.norm")
        s_out, e_out, M_out, ovf_eff, z = StateOut.check(ctx, p, sgn, e1, M, hg["out"], f"{tag}.out")
        ovfs.append((ovf_eff, sgn))
        state = (s_out, e_out, M_out)
    return emb.pack(ctx, p, (*state, z), ovfs, h["pack"])


def gen_vu_bb(p: Params, a: Sequence[int], b: Sequence[int]) -> dict:
    from verity_numerical.checker.step import check_step_expr

    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k:
        raise Reject(f"operand count must be a positive multiple of {k}")
    steps, c, ctx = [], 0, IntCtx()
    for start in range(0, len(a), k):
        hs = gen_step_bb(p, c, a[start : start + k], b[start : start + k])
        steps.append(hs)
        c = check_step_expr(ctx, p, c, list(a[start : start + k]), list(b[start : start + k]), hs)
    return {"steps": steps, "epilogue": Epilogue.gen(p, c)}


def check_vu_bb(ctx, p: Params, a: Sequence, b: Sequence, h: dict, emb: Embedding = DEFAULT):
    """The VU over BabyBear: units chained through their limb pairs from ``(0, 0)``, then the epilogue."""
    k = _k(p)
    if len(a) != len(b) or not a or len(a) % k or len(h["steps"]) != len(a) // k:
        raise Reject(f"operand count must be a positive multiple of {k} with one unit witness per chunk")
    c = (0, 0)
    for s, start in enumerate(range(0, len(a), k)):
        c = check_step_bb(ctx, p, c, a[start : start + k], b[start : start + k], h["steps"][s], emb)
    return emb.epilogue(ctx, p, c, h["epilogue"])


# ---------------------------------------------------------------------------------------------------------------------
# Mod-p evaluation on Python ints: the BabyBear-rules checker
# ---------------------------------------------------------------------------------------------------------------------


class BabyBearCtx(IntCtx):
    """Every predicate read over F_p on canonical representatives: ``eq`` a congruence, ``rng`` on ``(x - lo) mod p``,
    ``boolean`` mod p, lookup keys canonical (offset by the table's ``lo``), ``lt_pow`` the product read mod p.
    The embedding's rules decide which predicates reach this evaluator in which form."""

    def __init__(self, p: int = P):
        self.p = p

    def rng(self, name, x, lo, hi):
        if type(x) is not int or hi - lo > self.p or not (x - lo) % self.p < hi - lo:
            raise Reject(f"{name}: {x!r} not in [{lo}, {hi}) mod p")

    def eq(self, name, a, b):
        if (a - b) % self.p:
            raise Reject(f"{name}: {a!r} != {b!r} mod p")

    def boolean(self, name, b):
        if type(b) is not int or b % self.p not in (0, 1):
            raise Reject(f"{name}: {b!r} is not a bit mod p")

    def lookup(self, name, table: Table, key):
        k = (key - table.lo) % self.p + table.lo
        try:
            return table.row(k)
        except KeyError:
            raise Reject(f"{name}: key {key!r} is off table {table.name} mod p") from None

    def lt_pow(self, name, x, ipow, bits):
        self.rng(name, (x * ipow) % self.p, 0, 2**bits)


def accepts_step_bb(
    p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict, emb: Embedding = DEFAULT
) -> tuple[bool, str]:
    """The embedded unit on ints mod p: ``(accepted, first violated predicate)``; words as integers, limbed here."""
    ctx = BabyBearCtx()
    try:
        got = check_step_bb(ctx, p, limbs(c), list(a), list(b), h, emb)
        y_lo, y_hi = limbs(y)
        if emb.has("LIMB16"):
            ctx.eq("y_lo", got[0], y_lo)
            ctx.eq("y_hi", got[1], y_hi)
        else:
            ctx.eq("y", got[0], y)
    except Reject as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001 -- a forged hint may break the arithmetic itself
        return False, f"raised {e!r}"
    return True, ""


def accepts_vu_bb(
    p: Params, a: Sequence[int], b: Sequence[int], y16: int, h: dict, emb: Embedding = DEFAULT
) -> tuple[bool, str]:
    ctx = BabyBearCtx()
    try:
        ctx.eq("y16", check_vu_bb(ctx, p, list(a), list(b), h, emb), y16)
    except Reject as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"raised {e!r}"
    return True, ""


# ---------------------------------------------------------------------------------------------------------------------
# Interval analysis: which predicates need a rule
# ---------------------------------------------------------------------------------------------------------------------


class IVal:
    """An integer interval ``[lo, hi]``.  Affine combinations of atoms (hints, inputs, table columns) are kept as
    ``const + sum coeff * atom`` so that ``S - 2 sgn S`` and ``q - 2 s q`` are evaluated exactly (an atom's interval
    is used once per form); a product of two atoms carries its monomial key so that the bounds the tight checks
    impose on that product (``q * pow < 2^W``) apply where the identity uses it."""

    __slots__ = ("affine", "const", "ctx", "hi", "key", "lo")

    def __init__(self, ctx, lo: int, hi: int, key: frozenset | None = None, affine: dict | None = None, const: int = 0):
        self.ctx, self.key = ctx, key
        self.affine, self.const = affine, const
        if affine is not None:
            lo = hi = const
            for atom, coeff in affine.items():
                alo, ahi = ctx.atoms[atom]
                lo += min(coeff * alo, coeff * ahi)
                hi += max(coeff * alo, coeff * ahi)
        self.lo, self.hi = lo, hi

    def _lift(self, o):
        if isinstance(o, IVal):
            return o
        if isinstance(o, int):
            return IVal(self.ctx, o, o, frozenset(), affine={}, const=o)
        return NotImplemented

    def _add(self, o, sign: int):
        o = self._lift(o)
        if o is NotImplemented:
            return o
        if self.affine is not None and o.affine is not None:
            aff = dict(self.affine)
            for k, v in o.affine.items():
                aff[k] = aff.get(k, 0) + sign * v
            aff = {k: v for k, v in aff.items() if v}
            return IVal(
                self.ctx,
                0,
                0,
                frozenset()
                if not aff
                else (
                    next(iter(aff))
                    if len(aff) == 1 and aff[next(iter(aff))] == 1 and self.const + sign * o.const == 0
                    else None
                ),
                affine=aff,
                const=self.const + sign * o.const,
            )
        if sign > 0:
            return IVal(self.ctx, self.lo + o.lo, self.hi + o.hi)
        return IVal(self.ctx, self.lo - o.hi, self.hi - o.lo)

    def __add__(self, o):
        return self._add(o, 1)

    __radd__ = __add__

    def __sub__(self, o):
        return self._add(o, -1)

    def __rsub__(self, o):
        return self._lift(o).__sub__(self)

    def __neg__(self):
        return self._lift(0).__sub__(self)

    def __mul__(self, o):
        o = self._lift(o)
        if o is NotImplemented:
            return o
        if o.affine is not None and not o.affine:  # a constant
            c = o.const
            if self.affine is not None:
                return IVal(
                    self.ctx,
                    0,
                    0,
                    None,
                    affine={k: c * v for k, v in self.affine.items()} if c else {},
                    const=c * self.const,
                )
            corners = (self.lo * c, self.hi * c)
            return IVal(self.ctx, min(corners), max(corners))
        if self.affine is not None and not self.affine:
            return o.__mul__(self)
        corners = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        lo, hi = min(corners), max(corners)
        key = None
        if self.key and o.key:
            key = self.key | o.key
            if key in self.ctx.facts and lo >= 0:  # a product two tight checks bound
                hi = min(hi, self.ctx.facts[key] - 1)
        return IVal(self.ctx, lo, hi, key)

    __rmul__ = __mul__

    def __repr__(self):
        return f"[{self.lo}, {self.hi}]"


@dataclass
class PredicateBound:
    gadget: str
    predicate: str
    kind: str  # eq | rng | boolean | lt_pow | lt_direct | lookup
    lo: int  # integer interval of what the predicate constrains (a - b, x, the product ...)
    hi: int
    window_lo: int  # the interval must lie in [window_lo, window_hi) for the mod-p reading to be sound
    window_hi: int
    sound: bool
    note: str = ""

    @property
    def bound_bits(self) -> float:
        m = max(abs(self.lo), abs(self.hi))
        return m.bit_length() if m else 0


class BoundCtx:
    """Runs a gadget on intervals and records every predicate's window condition (module docstring)."""

    def __init__(self, gadget: str, modulus: int = P, facts: dict | None = None):
        self.gadget, self.p = gadget, modulus
        self.facts: dict[frozenset, int] = dict(facts or {})
        self.preds: list[PredicateBound] = []
        self.new_facts: dict[frozenset, int] = {}
        self._tables: dict[str, Table] = {}
        self.atoms: dict[str, tuple[int, int]] = {}

    # -- values --------------------------------------------------------------------------------------------------------
    def atom(self, name: str, lo: int, hi: int) -> IVal:
        """An atom with the closed interval ``[lo, hi - 1]``."""
        self.atoms[name] = (lo, hi - 1)
        return IVal(self, lo, hi - 1, frozenset({name}), affine={name: 1}, const=0)

    def _record(self, name, kind, lo, hi, wlo, whi, note=""):
        self.preds.append(PredicateBound(self.gadget, name, kind, lo, hi, wlo, whi, wlo <= lo and hi < whi, note))

    # -- predicates ------------------------------------------------------------------------------------------------------
    def rng(self, name, x, lo, hi):
        if isinstance(x, int):
            return
        if x.key is not None and len(x.key) == 1 and x.lo == lo and x.hi == hi - 1:
            # a hint at its declared range: one lookup pins one representative iff the range fits the field
            self._record(name, "rng", lo, hi - 1, lo, self.p + lo, "hint")
            return
        self._record(name, "rng", x.lo, x.hi, hi - self.p, lo + self.p)

    def boolean(self, name, b):
        if isinstance(b, int):
            return
        if b.key is not None and len(b.key) == 1:
            self._record(name, "boolean", 0, 1, 0, self.p, "hint")
            return
        self._record(name, "boolean", b.lo, b.hi, 2 - self.p, self.p)

    def eq(self, name, a, b):
        d = a - b if isinstance(a, IVal) else -(b - a) if isinstance(b, IVal) else None
        if d is None:
            return
        self._record(name, "eq", d.lo, d.hi, 1 - self.p, self.p)

    def lookup(self, name, table: Table, key):
        self._tables[table.name] = table
        if not isinstance(key, int):
            # the key is read canonically offset by the table's lo; it must not wrap into the table from outside
            self._record(name, "lookup", key.lo, key.hi, table.hi - self.p, table.lo + self.p)
        cols = list(zip(*(table.fn(k) for k in range(table.lo, table.hi))))
        return tuple(self.atom(f"{table.name}.{o}", min(c), max(c) + 1) for o, c in zip(table.outs, cols))

    def note_bound(self, x, bnd):
        """``x < bnd`` (an atom against a table column) bounds ``x * c`` by ``max_rows(bnd * c)`` for every column
        ``c`` of that table -- the products the identities use."""
        if not (isinstance(x, IVal) and isinstance(bnd, IVal) and x.key is not None and len(x.key) == 1 and bnd.key):
            return
        (bname,) = bnd.key
        tname, bcol = bname.split(".")
        table = self._tables[tname]
        rows = [table.fn(k) for k in range(table.lo, table.hi)]
        bi = table.outs.index(bcol)
        for ci, c in enumerate(table.outs):
            key = x.key | frozenset({f"{tname}.{c}"})
            fact = max(r[bi] * r[ci] for r in rows)
            self.new_facts[key] = min(fact, self.new_facts.get(key, fact))

    def lt_pow(self, name, x, ipow, bits):
        saved, self.facts = self.facts, {}  # the check's own product, before what it imposes
        prod = x * ipow
        self.facts = saved
        self._record(
            name,
            "lt_pow",
            prod.lo,
            prod.hi,
            2**bits - self.p,
            self.p,
            "product realisation" + ("" if 2 ** (2 * bits) <= self.p else f"; needs 2^{2 * bits} <= p"),
        )
        if isinstance(x, IVal) and isinstance(ipow, IVal) and x.key is not None and len(x.key) == 1 and ipow.key:
            key = x.key | ipow.key
            self.new_facts[key] = min(2**bits, self.new_facts.get(key, 2**bits))
            (cname,) = ipow.key
            tname, c = cname.split(".")
            partner = partner_columns(self._tables[tname], bits).get(c)
            if partner is not None:
                cols = list(
                    zip(*(self._tables[tname].fn(k) for k in range(self._tables[tname].lo, self._tables[tname].hi)))
                )
                col = cols[self._tables[tname].outs.index(partner)]
                self.note_bound(x, self.atom(f"{tname}.{partner}", min(col), max(col) + 1))


def _run_bound(emb: Embedding, gadget: str, p: Params, facts: dict) -> BoundCtx:
    """The gadget on intervals, every assignment of its bit-valued inputs and hints enumerated (interval arithmetic
    alone loses the correlation in ``q - 2 s q``); predicates merged by name (union of intervals, all-sound)."""
    import itertools

    c = emb.contract(gadget, p)
    bits = [
        n
        for n, (lo, hi) in {
            **{f"in.{k}": v for k, v in c.inputs.items()},
            **{f"hint.{k}": v for k, v in c.hints.items()},
        }.items()
        if (lo, hi) == (0, 2)
    ]
    merged: dict[str, PredicateBound] = {}
    order: list[str] = []
    new_facts: dict = {}
    tables: dict = {}
    for assignment in itertools.product((0, 1), repeat=len(bits)):
        ctx = _run_bound_once(emb, gadget, p, facts, dict(zip(bits, assignment)))
        new_facts.update(ctx.new_facts)
        tables.update(ctx._tables)
        for b in ctx.preds:
            if b.predicate not in merged:
                merged[b.predicate] = b
                order.append(b.predicate)
            else:
                m = merged[b.predicate]
                m.lo, m.hi, m.sound = min(m.lo, b.lo), max(m.hi, b.hi), m.sound and b.sound
    out = BoundCtx(gadget, facts=facts)
    out.new_facts, out._tables = new_facts, tables
    for name in bits:
        out.preds.append(
            PredicateBound(
                gadget,
                name.split(".", 1)[1],
                "boolean",
                0,
                1,
                0,
                P,
                True,
                "hint" if name.startswith("hint") else "input",
            )
        )
    out.preds += [merged[n] for n in order]
    return out


def _run_bound_once(emb: Embedding, gadget: str, p: Params, facts: dict, bits: dict[str, int]) -> BoundCtx:
    ctx = BoundCtx(gadget, facts=facts)
    c = emb.contract(gadget, p)
    x = {n: bits.get(f"in.{n}", ctx.atom(f"in.{n}", lo, hi)) for n, (lo, hi) in c.inputs.items()}
    h = {n: bits.get(f"hint.{n}", ctx.atom(f"hint.{n}", lo, hi)) for n, (lo, hi) in c.hints.items()}
    n = p.n
    if gadget == "bf16_decode":
        Bf16Decode.check(ctx, p, x["w"], h)
    elif gadget == "fp32_decode":
        emb.fp32_decode(ctx, p, (x["c_lo"], x["c_hi"]), h)
    elif gadget == "product":
        Product.check(ctx, p, (x["s_a"], x["m_a"], x["e_a"]), (x["s_b"], x["m_b"], x["e_b"]), h)
    elif gadget == "group_max":
        GroupMax.check(
            ctx,
            p,
            [x[f"cand[{i}]"] for i in range(n + 1)],
            {"E": h["E"], "sel": [h[f"sel[{i}]"] for i in range(n + 2)]},
        )
    elif gadget == "align_state":
        emb.align_state(ctx, p, x["M"], x["d"], h)
    elif gadget == "align_product":
        emb.align_product(ctx, p, x["u"], x["d"], h)
    elif gadget == "signed_sum":
        SignedSum.check(ctx, p, [(x[f"s[{i}]"], x[f"q[{i}]"]) for i in range(n + 1)], h)
    elif gadget == "leading_bit":
        emb.leading_bit(ctx, p, x["v"], h)
    elif gadget == "normalise":
        emb.normalise(ctx, p, x["E"], x["l"], x["v"], h)
    elif gadget == "state_out":
        StateOut.check(ctx, p, x["sgn"], x["e1"], x["M"], h)
    elif gadget == "pack":
        G = len(p.groups)
        y = emb.pack(ctx, p, (x["s"], x["e1"], x["M"], x["z"]), [(x[f"ovf[{g}]"], x[f"sgn[{g}]"]) for g in range(G)], h)
        for i, part in enumerate(y):
            if isinstance(part, IVal):
                ctx.eq(
                    f"pack.y[{i}]", part, ctx.atom(f"y[{i}]", 0, 2**LIMB if emb.has("LIMB16") else 2**p.acc_word_bits)
                )
    elif gadget == "epilogue":
        emb.epilogue(ctx, p, (x["y32_lo"], x["y32_hi"]), h)
    else:
        raise KeyError(gadget)
    return ctx


GADGETS: tuple[str, ...] = (
    "bf16_decode",
    "fp32_decode",
    "product",
    "group_max",
    "align_state",
    "align_product",
    "signed_sum",
    "leading_bit",
    "normalise",
    "state_out",
    "pack",
    "epilogue",
)


def analyse(emb: Embedding = DEFAULT, p: Params = REAL) -> list[PredicateBound]:
    """Every predicate of every gadget under ``emb`` with its integer interval and window verdict (two passes)."""
    out = []
    for g in GADGETS:
        first = _run_bound(emb, g, p, {})
        second = _run_bound(emb, g, p, first.new_facts)
        out += second.preds
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------------------------------

RULE_TEXT = {
    "LIMB16": "the word is two public 16-bit limbs; the identity is stated per limb (fraction limbs f_lo/f_hi or low_lo/low_hi as hints)",
    "LT_DIRECT": "direct variable-bound check rng(bnd - 1 - x, 0, 2^B) against the table's bound column (POW/ALIGN: the partner column; LEAD: new column span), plus rng(x, 0, 2^B) when x is not a range-checked hint (the leading-bit window v - lo: the direct form alone loses v >= lo -- z3 found it)",
    "QUOT_TIGHT": "the tight quotient bound q < 2^B / pow (direct) keeps q * pow + r below 2^B, so the division identity stays inside (-p, p)",
    "NORM_MBND": "NORM column mbnd = 2^30 / div and rng(mbnd - 1 - M', 0, 2^30): the quotient is tight relative to the dividend v < 2^30",
    "NORM_VBND": "NORM column vbnd = 2^24 / mul (k >= 0) and rng(vbnd - 1 - v, 0, 2^30): v * mul < 2^24 (redundant given leading_bit; see z3)",
    "V_PREMISE": "v is the signed-sum wire, |v| <= s_max = 570,417,150 < 2^29.1 by the ranges of its terms; leading_bit/normalise are stated for v in [0, s_max]",
}


@dataclass
class RuleRow:
    gadget: str
    predicate: str
    as_stated_bits: float
    as_stated_interval: str
    rule: str
    embedded_bits: float
    embedded_sound: bool
    added: str


def _by_key(preds: list[PredicateBound]) -> dict[tuple[str, str], PredicateBound]:
    return {(b.gadget, b.predicate): b for b in preds}


#: The integer checker read modulo p as it stands (its tight quotient bound included, no other rule).
NAIVE = Embedding(frozenset({"QUOT_TIGHT"}))


def rules_report(p: Params = REAL) -> dict:
    """The naive reading (the checker as stated, mod p), the full embedding, per predicate which rule closes it, and
    per rule which predicates leave their window when that rule alone is dropped (interval verdict; z3 decides)."""
    naive = analyse(NAIVE, p)
    full = analyse(DEFAULT, p)
    drop = {}
    for r in RULES:
        bad = [f"{b.gadget}.{b.predicate}" for b in analyse(Embedding(ALL_RULES - {r}), p) if not b.sound]
        drop[r] = bad
    fb = _by_key(full)
    rows, safe = [], []
    rule_of = {
        ("fp32_decode", "acc.word"): (
            "LIMB16",
            "+2 hint columns (f_lo 16 b, f_hi 7 b), +2 16/7-bit ranges, -1 23-bit range, +1 identity; c as 2 public limbs",
        ),
        ("pack", "pack.y[0]"): ("LIMB16", "+2 hint columns (low_lo, low_hi), +1 identity; y as 2 public limbs"),
        ("epilogue", "cast.split"): ("LIMB16", "-2 hint columns (hi, lo are the public limbs), -1 identity, -2 ranges"),
        ("align_state", "align_state.rem"): (
            "LT_DIRECT",
            "no multiplication; one 25-bit range (2 chunk lookups) on a linear wire, as before",
        ),
        ("align_state", "align_state.quot"): (
            "LT_DIRECT+QUOT_TIGHT",
            "no multiplication; one 25-bit range on a linear wire",
        ),
        ("align_state", "align_state.div"): ("QUOT_TIGHT", "none: the two direct bounds pin q * pow + r < 2^25"),
        ("align_product", "align_prod.rem"): ("LT_DIRECT", "no multiplication; one 16-bit range on a linear wire"),
        ("align_product", "align_prod.quot"): (
            "LT_DIRECT+QUOT_TIGHT",
            "no multiplication; one 16-bit range on a linear wire",
        ),
        ("align_product", "align_prod.div"): ("QUOT_TIGHT", "none: the two direct bounds pin q * sr + r < 2^16"),
        ("leading_bit", "lead.window"): (
            "LT_DIRECT+V_PREMISE",
            "LEAD gains column span; no multiplication; two 30-bit ranges (4 chunk lookups) on the linear wire v - lo (>= 0 and < span)",
        ),
        ("normalise", "norm.div"): (
            "NORM_MBND+NORM_VBND(redundant)+V_PREMISE",
            "NORM gains columns mbnd (and vbnd); +1 30-bit range (2 chunk lookups) per group [+1 if vbnd kept]; 0 hints, 0 multiplications",
        ),
    }
    for b in naive:
        key = (b.gadget, b.predicate)
        if b.sound:
            safe.append(
                {
                    "gadget": b.gadget,
                    "predicate": b.predicate,
                    "kind": b.kind,
                    "interval": f"[{b.lo}, {b.hi}]",
                    "bits": b.bound_bits,
                }
            )
            continue
        rule, added = rule_of.get(key, ("?", "?"))
        emb_pred = fb.get(key)
        # the limbed identities have new names
        if emb_pred is None and key == ("fp32_decode", "acc.word"):
            emb_pred = max(
                (fb[k] for k in fb if k[0] == "fp32_decode" and k[1].startswith("acc.word")), key=lambda x: x.hi
            )
        if emb_pred is None and key == ("pack", "pack.y[0]"):
            emb_pred = max((fb[k] for k in fb if k[0] == "pack" and k[1].startswith("pack.y")), key=lambda x: x.hi)
        if emb_pred is None and key == ("epilogue", "cast.split"):
            emb_pred = PredicateBound(
                "epilogue", "cast.split", "eq", 0, 0, 0, 1, True, "removed: limbs are the public inputs"
            )
        if emb_pred is None:
            emb_pred = PredicateBound(b.gadget, b.predicate, b.kind, 0, 0, 0, 1, False, "no embedded counterpart")
        rows.append(
            asdict(
                RuleRow(
                    b.gadget,
                    b.predicate,
                    b.bound_bits,
                    f"[{b.lo}, {b.hi}]",
                    rule,
                    emb_pred.bound_bits,
                    emb_pred.sound,
                    added,
                )
            )
        )
    unsound_full = [asdict(b) for b in full if not b.sound]
    return {
        "p": P,
        "p_bits": P.bit_length(),
        "log2_p": round(__import__("math").log2(P), 3),
        "params": {
            "acc_bits": p.acc_bits,
            "prod_bits": p.prod_bits,
            "s_bits": p.s_bits,
            "s_max": p.s_max,
            "norm_rem_bits": p.norm_rem_bits,
            "k_lo": p.k_lo,
            "k_hi": p.k_hi,
        },
        "rules": {r: RULE_TEXT[r] for r in RULES},
        "needs_rule": rows,
        "safe_as_stated": safe,
        "embedded_unsound": unsound_full,
        "embedded_predicates": len(full),
        "naive_predicates": len(naive),
        "drop_one_rule": drop,
    }


def render_md(rep: dict) -> str:
    L = []
    L.append("# BabyBear embedding rules for the WP2 checker (red-team-babybear)\n")
    L.append(
        f"p = 2^31 - 2^27 + 1 = {rep['p']} (log2 p = {rep['log2_p']}). Every predicate of every gadget contract was run on"
    )
    L.append(
        "integer intervals (`verity_numerical.redteam.babybear.analyse`): inputs and hints at their declared ranges, table"
    )
    L.append(
        "columns at their row extremes; the mod-p reading of a predicate is sound iff the interval of what it constrains lies"
    )
    L.append(
        "in a window of width p (identity: `a - b` in (-p, p); range check on an expression: in `[hi - p, lo + p)`; the"
    )
    L.append(
        "product realisation of a variable bound: `x * ipow` in `[2^B - p, p)`). Widths: acc_bits = "
        f"{rep['params']['acc_bits']}, prod_bits = {rep['params']['prod_bits']}, s_bits = {rep['params']['s_bits']}, "
        f"|S| <= s_max = {rep['params']['s_max']} (2^{rep['params']['s_max'].bit_length() - 1}.x), norm remainder "
        f"{rep['params']['norm_rem_bits']} bits, k in [{rep['params']['k_lo']}, {rep['params']['k_hi']}].\n"
    )
    L.append("## 1. Predicates whose integer bound exceeds p as stated, and the rule that closes each\n")
    L.append(
        "| gadget | predicate | as-stated interval | bits | rule | embedded bits | embedded sound | added columns / lookups |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rep["needs_rule"]:
        L.append(
            f"| {r['gadget']} | `{r['predicate']}` | {r['as_stated_interval']} | {r['as_stated_bits']} | {r['rule']} | "
            f"{r['embedded_bits']} | {'yes' if r['embedded_sound'] else 'NO'} | {r['added']} |"
        )
    L.append("\n## 2. The rules\n")
    for name, text in rep["rules"].items():
        L.append(f"- **{name}**: {text}")
    L.append("\n## 3. Safe as stated over BabyBear (bound < p): no change for the lanes\n")
    L.append("| gadget | predicate | kind | interval | bits |")
    L.append("|---|---|---|---|---|")
    for s in rep["safe_as_stated"]:
        L.append(f"| {s['gadget']} | `{s['predicate']}` | {s['kind']} | {s['interval']} | {s['bits']} |")
    L.append("")
    if rep["embedded_unsound"]:
        L.append("## 4. Embedded predicates still outside the window (MUST be empty)\n")
        for u in rep["embedded_unsound"]:
            L.append(
                f"- {u['gadget']} `{u['predicate']}` [{u['lo']}, {u['hi']}] window [{u['window_lo']}, {u['window_hi']}) {u['note']}"
            )
    else:
        L.append(
            f"## 4. Under the full rule set every one of the {rep['embedded_predicates']} predicates lies inside its window.\n"
        )
    L.append(
        "## 5. Dropping one rule at a time: predicates that leave their window (interval verdict; z3 in `z3_babybear` decides)\n"
    )
    L.append("| rule dropped | predicates outside the window by interval arithmetic |")
    L.append("|---|---|")
    for r, bad in rep["drop_one_rule"].items():
        L.append(
            f"| {r} | {', '.join(f'`{b}`' for b in bad) if bad else '(none: the interval argument does not need it)'} |"
        )
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="markdown path (default backends/numerical/reports/babybear_rules.md)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rep = rules_report()
    root = Path(__file__).resolve().parents[5]
    out = Path(args.out) if args.out else root / "backends/numerical/reports/babybear_rules.md"
    marker = "<!-- hand-written appendix: z3 verdicts, battery, accounting -->"
    appendix = ""
    if out.exists() and marker in out.read_text():
        appendix = out.read_text().split(marker, 1)[1]
    out.write_text(render_md(rep) + (f"\n{marker}{appendix}" if appendix else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(render_md(rep))
    print(f"wrote {out}")
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        Path(run_dir, "babybear_rules.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
