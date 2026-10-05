"""The v2 gadgets: the same relation as ``checker.gadgets`` (v1), arithmetised with fewer hints, fewer
multiplications and fewer, wider lookups.  Same interface: ``contract(p)``, ``gen``, ``check(ctx, p, ..., h)``.

What changed and why it is sound (z3 per gadget in ``search.py`` of this package; the composition is the same
induction as v1's):

    OpDecode     one T_OP lookup per operand word (v1: 4 hints, 3 mul, 2 lin, 3 range checks).  The table IS the
                 decode; the word's range and finiteness are the table's domain.  A zero operand gets e_op = e_zero
                 (< floor - e_max), so a zero product's e_p = e_a + e_b is below the floor without a zero flag.
    StateDecode  the accumulator word's header (s, t) by one T_HDR lookup; f, z as before (v1: 5 hints, 5 mul).
    Product      u = m_a m_b, e_p = e_a + e_b, s4 = s_a + 2 s_b: expressions only (v1: a zero flag, 5 mul, 1 range).
    GroupMax     E hint; per candidate one ALIGN4 lookup on 4 (E - cand) + s4 gives d~, s_p, [d = 0] and proves
                 E - cand in [0, shift_rows) (unique key decomposition, s4 in [0, 4)); attained: sum [d_i = 0] +
                 [E = floor] >= 1, one 4-bit range check (v1: 10 selector hints, 19 mul, 2 lin, 11 range checks).
    ProductShift one SHIFT lookup on (u, d~, s_p): the signed aligned term (v1: 2 hints, 4 mul, 1 lin, 5 lookups,
                 plus the sign product in the sum).
    StateShift   M = M_hi 2^mb + M_lo (2 hints, 1 lin), two SSHIFT lookups; exact because the low half's scaled
                 value is below the high half's unit (v1: 2 hints, 3 mul, 1 lin, 9 lookups).
    SignedSum    S is a sum of signed table outputs; sgn, v = (1 - 2 sgn) S split as v_hi 2^vb + v_lo, both halves
                 through LEAD (range + bit length); l = bit_length(v) from the halves (v1: 2 hints, 14 mul, 4 range
                 checks, plus LeadingBit's lookup and 30-bit window check).
    Normalise    one LEADNORM lookup on (l, E) gives k, A = 2^(vb + k), e_hat, z, ovf_eff, [E = floor]; one TNORM
                 lookup gives floor(v_lo 2^k); M' = v_hi A + TNORM(v_lo, k) is an expression (v1: 3 hints, 7 mul,
                 7 lookups; StateOut: 2 hints, 8 mul, 3 lookups).
    Pack         the FP32 word is linear in (s, e_hat, M, z): (e_hat + bias - 1 + z (e_min - floor)) 2^acc_frac + M
                 carries the hidden bit by itself; the infinity of the first saturating group as before
                 (v1: 2 hints, 8 mul, 1 lin, 2 lookups).

Field embedding.  No ``lt_pow`` remains: every variable-bound range check became a lookup on a packed key, and
every multiplication is either a Boolean times a value or ``m_a m_b < 2^16``, ``v_hi A < 2^(vh + vb + k_hi)`` (< 2^54
at REAL: below Goldilocks, NOT below a 31-bit prime: a 31-bit field splits A's multiplication as v1's F3 did),
``(1 - 2 sgn) S`` with |S| < 2^30 (U3: p > 2^31 - 2).  Words: the unit's c and y are 32-bit (v1 F4).
"""
from __future__ import annotations

from collections.abc import Sequence

from ..gadgets import Contract, _bit, _bits
from ..params import Params, Reject
from .params import V2
from .tables import (
    align4_table,
    hdr_table,
    lead_table,
    leadnorm_table,
    op_table,
    shift_table,
    sshift_hi_table,
    sshift_lo_table,
    tnorm_table,
)

__all__ = ["GroupMax", "Normalise", "OpDecode", "Pack", "Product", "ProductShift", "SignedSum", "StateDecode",
           "StateShift"]


class OpDecode:
    """w -> (s, m, e_op) by one T_OP lookup."""

    name = "op_decode"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract("op_decode", inputs={"w": _bits(p.word_bits)}, hints={},
                        outputs={"s": _bit(), "m": _bits(p.sig), "e": (V2.e_zero(p), p.e_max + 1)}, tables=("T_OP",))

    @staticmethod
    def gen(p: Params, w: int) -> dict:
        return {}

    @staticmethod
    def check(ctx, p: Params, w, h: dict | None = None, tag: str = "op"):
        return ctx.lookup(f"{tag}.op", op_table(p), w)


class StateDecode:
    """c -> (s, e_hat, M, z): header lookup, fraction hint, zero flag."""

    name = "state_decode"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "state_decode",
            inputs={"c": _bits(p.acc_word_bits)},
            hints={"hdr": _bits(p.acc_exp + 1), "f": _bits(p.acc_frac), "z": _bit()},
            checked={"M+z-1": _bits(p.acc_sig)},
            outputs={"s": _bit(), "e_hat": (p.floor, p.acc_e_max + 1), "M": _bits(p.acc_sig), "z": _bit()},
            tables=("T_HDR",),
        )

    @staticmethod
    def gen(p: Params, c: int) -> dict:
        f = c & (2**p.acc_frac - 1)
        t = (c >> p.acc_frac) & (2**p.acc_exp - 1)
        return {"hdr": c >> p.acc_frac, "f": f, "z": int(t == 0 and f == 0)}

    @staticmethod
    def check(ctx, p: Params, c, h: dict, tag: str = "acc"):
        hdr, f, z = h["hdr"], h["f"], h["z"]
        s, nz, e = ctx.lookup(f"{tag}.hdr", hdr_table(p), hdr)
        ctx.rng(f"{tag}.f", f, 0, 2**p.acc_frac)
        ctx.eq(f"{tag}.word", hdr * 2**p.acc_frac + f, c)
        ctx.boolean(f"{tag}.z", z)
        M = f + nz * 2**p.acc_frac
        ctx.eq(f"{tag}.z0", z * M, 0)
        ctx.rng(f"{tag}.z1", M + z - 1, 0, 2**p.acc_sig)
        e_hat = e + z * (p.floor - e)
        return s, e_hat, M, z


class Product:
    """u = m_a m_b, e_p = e_a + e_b, s4 = s_a + 2 s_b: no predicates."""

    name = "product"

    @staticmethod
    def contract(p: Params) -> Contract:
        ez = V2.e_zero(p)
        return Contract(
            "product",
            inputs={"s_a": _bit(), "m_a": _bits(p.sig), "e_a": (ez, p.e_max + 1),
                    "s_b": _bit(), "m_b": _bits(p.sig), "e_b": (ez, p.e_max + 1)},
            hints={},
            outputs={"u": _bits(2 * p.sig), "e_p": (2 * ez, p.ep_hi + 1), "s4": (0, 4)},
        )

    @staticmethod
    def gen(p: Params, *_args) -> dict:
        return {}

    @staticmethod
    def check(ctx, p: Params, a, b, h: dict | None = None, tag: str = "prod"):
        (s_a, m_a, e_a), (s_b, m_b, e_b) = a, b
        return m_a * m_b, e_a + e_b, s_a + 2 * s_b


class GroupMax:
    """E = max(floor, cands) with one hint E and one ALIGN4 lookup per candidate (key 4 (E - cand) + s4), which
    proves E - cand in [0, shift_rows) and yields (d~, s_p, [d = 0]); attained: sum [d = 0] + isfloor >= 1, where
    ``isfloor = [E = floor]`` is LEADNORM's column (an input here: the same E keys that table)."""

    name = "group_max"

    @staticmethod
    def contract(p: Params, m: int | None = None) -> Contract:
        m = p.n + 1 if m is None else m
        last, lo = V2.last(p), V2.cand_lo(p)
        return Contract(
            "group_max",
            inputs={**{f"cand[{i}]": (lo, last + 1) for i in range(m)}, **{f"s4[{i}]": (0, 4) for i in range(m)},
                    "isfloor": _bit()},
            hints={"E": (p.floor, last + 1)},
            checked={"attained-1": (0, m + 1)},
            outputs={"E": (p.floor, last + 1), **{f"dt[{i}]": (0, V2.dt(p) + 1) for i in range(m)},
                     **{f"sp[{i}]": _bit() for i in range(m)}},
            tables=("ALIGN4",),
        )

    @staticmethod
    def gen(p: Params, cands: Sequence[int]) -> dict:
        return {"E": max(p.floor, *cands)}

    @staticmethod
    def align(ctx, p: Params, E, cand, s4, tag: str):
        return ctx.lookup(f"{tag}.align", align4_table(p), 4 * (E - cand) + s4)

    @staticmethod
    def attained(ctx, p: Params, iszs: Sequence, isfloor, tag: str):
        ctx.rng(f"{tag}.attained", sum(iszs) + isfloor - 1, 0, len(iszs) + 2)

    @staticmethod
    def check(ctx, p: Params, cands: Sequence, s4s: Sequence, isfloor, h: dict, tag: str = "max"):
        E = h["E"]
        rows = [GroupMax.align(ctx, p, E, c, s, f"{tag}[{i}]") for i, (c, s) in enumerate(zip(cands, s4s))]
        GroupMax.attained(ctx, p, [r[2] for r in rows], isfloor, tag)
        return E, [r[0] for r in rows], [r[1] for r in rows]


class ProductShift:
    """The signed aligned product term by one SHIFT lookup on (u, d~, s_p)."""

    name = "product_shift"

    @staticmethod
    def contract(p: Params) -> Contract:
        B = p.prod_bits
        return Contract("product_shift", inputs={"u": _bits(2 * p.sig), "dt": (0, V2.dt(p) + 1), "sp": _bit()},
                        hints={}, outputs={"term": (-(2**B) + 1, 2**B)}, tables=("SHIFT",))

    @staticmethod
    def gen(p: Params, *_args) -> dict:
        return {}

    @staticmethod
    def check(ctx, p: Params, u, dt, sp, h: dict | None = None, tag: str = "pshift"):
        R = 2 * V2.dt_rows(p)
        (term,) = ctx.lookup(f"{tag}.shift", shift_table(p), u * R + 2 * dt + sp)
        return term


class StateShift:
    """The signed aligned state term: M = M_hi 2^mb + M_lo, two SSHIFT lookups (each key proves its half's range)."""

    name = "state_shift"

    @staticmethod
    def contract(p: Params) -> Contract:
        W = p.acc_bits
        return Contract("state_shift", inputs={"M": _bits(p.acc_sig), "dt": (0, V2.dt(p) + 1), "s": _bit()},
                        hints={"M_hi": _bits(V2.mh(p)), "M_lo": _bits(V2.mb(p))},
                        outputs={"term": (-(2**W) + 1, 2**W)}, tables=("SSHIFT_HI", "SSHIFT_LO"))

    @staticmethod
    def gen(p: Params, M: int, *_args) -> dict:
        mb = V2.mb(p)
        return {"M_hi": M >> mb, "M_lo": M & (2**mb - 1)}

    @staticmethod
    def check(ctx, p: Params, M, dt, s, h: dict, tag: str = "sshift"):
        R = 2 * V2.dt_rows(p)
        M_hi, M_lo = h["M_hi"], h["M_lo"]
        # The halves are range-checked on their own, not only through the packed lookup keys: over F_p the key
        # M_hi R + 2 dt + s is a *linear* function of the hint, so any table row kk has a field solution
        # M_hi = (kk - 2 dt - s) / R, and the two rows are then only tied by the split congruence (SAT over M31,
        # z3 UNKNOWN over Goldilocks).  With the halves bounded every key is below p and the field reading is the
        # integer one (`checker.v2.search.wrapfree_queries_v2`).
        ctx.rng(f"{tag}.hi_range", M_hi, 0, 2 ** V2.mh(p))
        ctx.rng(f"{tag}.lo_range", M_lo, 0, 2 ** V2.mb(p))
        ctx.eq(f"{tag}.split", M_hi * 2 ** V2.mb(p) + M_lo, M)
        (hi,) = ctx.lookup(f"{tag}.hi", sshift_hi_table(p), M_hi * R + 2 * dt + s)
        (lo,) = ctx.lookup(f"{tag}.lo", sshift_lo_table(p), M_lo * R + 2 * dt + s)
        return hi + lo


class SignedSum:
    """S = sum of the signed terms; v = (1 - 2 sgn) S = v_hi 2^vb + v_lo with both halves through LEAD, which
    proves their ranges and yields l = bit_length(v).  ``sgn`` is pinned whenever S != 0 (v >= 0 and |S| < 2^s_bits);
    at S = 0 it is free and every consumer masks it with z (StateOut's s = sgn (1 - z); Pack's ovf_eff = 0)."""

    name = "signed_sum"

    @staticmethod
    def premise(p: Params, m: int) -> None:
        bound = (2**p.acc_bits - 1) + m * (2**p.prod_bits - 1)
        if bound >= 2**p.s_bits:
            raise Reject(f"signed_sum premise: |S| <= {bound} is not below 2^{p.s_bits}")

    @staticmethod
    def contract(p: Params, m: int | None = None) -> Contract:
        m = p.n if m is None else m
        SignedSum.premise(p, m)
        W, B = p.acc_bits, p.prod_bits
        return Contract(
            "signed_sum",
            inputs={"t[0]": (-(2**W) + 1, 2**W), **{f"t[{i}]": (-(2**B) + 1, 2**B) for i in range(1, m + 1)}},
            hints={"sgn": _bit(), "v_hi": _bits(V2.vh(p)), "v_lo": _bits(V2.vb(p))},
            outputs={"sgn": _bit(), "v_hi": _bits(V2.vb(p)), "v_lo": _bits(V2.vb(p)), "l": (0, p.l_hi + 1)},
            tables=("LEAD",),
        )

    @staticmethod
    def gen(p: Params, terms: Sequence[int]) -> dict:
        S = sum(terms)
        v, vb = abs(S), V2.vb(p)
        return {"sgn": int(S < 0), "v_hi": v >> vb, "v_lo": v & (2**vb - 1)}

    @staticmethod
    def check(ctx, p: Params, terms: Sequence, h: dict, tag: str = "sum"):
        SignedSum.premise(p, len(terms) - 1)
        sgn, v_hi, v_lo = h["sgn"], h["v_hi"], h["v_lo"]
        vb = V2.vb(p)
        ctx.boolean(f"{tag}.sgn", sgn)
        S = sum(terms)
        v = S - 2 * sgn * S
        ctx.eq(f"{tag}.split", v_hi * 2**vb + v_lo, v)
        nz_h, l_h = ctx.lookup(f"{tag}.lead_hi", lead_table(p), v_hi)
        _nz_l, l_l = ctx.lookup(f"{tag}.lead_lo", lead_table(p), v_lo)
        l = l_h + nz_h * vb + l_l - nz_h * l_l          # v_hi != 0: vb + bit_length(v_hi); else bit_length(v_lo)
        return sgn, v_hi, v_lo, l


class Normalise:
    """(E, l, v_hi, v_lo) -> (e_hat, M', z, ovf_eff, isfloor) by one LEADNORM and one TNORM lookup;
    M' = v_hi A(k) + floor(v_lo 2^k) = floor(v 2^k) exactly (the low half is below A's unit)."""

    name = "normalise"

    @staticmethod
    def contract(p: Params) -> Contract:
        last = V2.last(p)
        return Contract(
            "normalise",
            inputs={"E": (p.floor, last + 1), "l": (0, p.l_hi + 1), "v_hi": _bits(V2.vh(p)), "v_lo": _bits(V2.vb(p))},
            hints={},
            outputs={"e_hat": (p.floor, p.exp_hi + 1), "M": _bits(p.acc_sig), "z": _bit(), "ovf": _bit(), "isfloor": _bit()},
            tables=("LEADNORM", "TNORM"),
        )

    @staticmethod
    def gen(p: Params, *_args) -> dict:
        return {}

    @staticmethod
    def check(ctx, p: Params, E, l, v_hi, v_lo, h: dict | None = None, tag: str = "norm"):
        koff, A, e_hat, z, ovf, isfloor = ctx.lookup(f"{tag}.leadnorm", leadnorm_table(p), V2.leadnorm_key(p, l, E))
        (t,) = ctx.lookup(f"{tag}.tnorm", tnorm_table(p), v_lo + 2 ** V2.vb(p) * koff)
        M = v_hi * A + t
        return e_hat, M, z, ovf, isfloor


class Pack:
    """The FP32 word of the final state: linear in (s, e_hat, M, z) unless a group saturated, then the infinity
    of the first saturating group's sum sign.  ``fin`` is 0 for the zero state (s = 0, M = 0, e_hat = floor, z = 1)."""

    name = "pack"

    @staticmethod
    def contract(p: Params) -> Contract:
        G = len(p.groups)
        return Contract(
            "pack",
            inputs={"s": _bit(), "e_hat": (p.floor, p.exp_hi + 1), "M": _bits(p.acc_sig), "z": _bit(),
                    **{f"ovf[{g}]": _bit() for g in range(G)}, **{f"sgn[{g}]": _bit() for g in range(G)}},
            hints={},
            outputs={"y": _bits(p.acc_word_bits)},
        )

    @staticmethod
    def gen(p: Params, *_args) -> dict:
        return {}

    @staticmethod
    def word(p: Params, state) -> object:
        s, e_hat, M, z = state
        fld = e_hat + (p.acc_bias - 1) + z * (p.acc_e_min - p.floor)
        return s * 2 ** (p.acc_word_bits - 1) + fld * 2**p.acc_frac + M

    @staticmethod
    def check(ctx, p: Params, state, ovfs: Sequence, h: dict | None = None, tag: str = "pack"):
        fin = Pack.word(p, state)
        any_ovf, inf_sign, none_yet = 0, 0, 1
        for ovf_g, sgn_g in ovfs:
            first = none_yet * ovf_g
            any_ovf = any_ovf + first
            inf_sign = inf_sign + first * sgn_g
            none_yet = none_yet - first
        top = 2 ** (p.acc_word_bits - 1)
        return fin - any_ovf * fin + inf_sign * top + any_ovf * (p.acc_t_finite * 2**p.acc_frac)
