"""The group sum, leading bit and normalisation of a *wide* adder in BabyBear -- the Hopper BF16 relation
(``HOPPER_BF16_M16N8K16``: one group of 16 products, a 26-bit adder), where ``|S| <= s_max = 2214559740 > p`` and
``v < 2^s_bits = 2^32``.  ``SignedSum`` / ``LeadingBit`` / ``Normalise`` as written form ``S``, ``v``, ``v mul`` and
``div M + r`` as single field elements; over BabyBear (p = 2^31 - 2^27 + 1) none of those identities would be an
integer identity, and ``LigeroCtx._linear`` refuses them.  This module proves the *same relation* (the checker's
``check_step_expr`` semantics on the group) with every linear identity split into two 16-bit limbs and a carry:

* **partials**: the 17 signed terms are grouped greedily into partial sums ``A_k`` whose magnitude fits a bit
  decomposition inside the field (``2^b_k + max|A_k| < p``; Hopper: 7 + 7 + 3 terms at 30, 30, 29 bits).  Each
  partial gets the sign-magnitude form ``A_k = (1 - 2 sgn_k) v_k`` with ``v_k`` range-checked to ``b_k`` bits -- the
  bit rows give its limbs ``v_k = vh_k 2^16 + vl_k`` for free;
* **recombination**: the prover commits ``v``'s limbs ``vh, vl`` (each range-checked), ``sgn`` and a small carry
  ``c``: ``sum_k (1 - 2 sgn_k) vl_k - (1 - 2 sgn) vl = c 2^16`` and ``sum_k (1 - 2 sgn_k) vh_k - (1 - 2 sgn) vh + c = 0``,
  which together are ``S = (1 - 2 sgn) v`` as integers; ``z = [v = 0]`` through ``vh + vl`` (both non-negative);
* **leading bit**: ``2^(l-1) <= v < 2^l`` as one window test per limb through the table ``LEADW[l]``
  (``l <= 16``: ``vh = 0`` and the window on ``vl``; ``l > 16``: the window on ``vh``, ``vl`` free);
* **normalisation**: ``div M + r = mul v`` split at bit 16 with a carry, ``M = Mh 2^16 + Ml`` from ``M``'s own bit
  rows; the theorem check ``v < 2^min(s_bits, acc_sig - k)`` (the premise that bounds ``mul v`` below ``2^acc_sig``)
  is stated per limb.

``needs_wide(p)`` decides: the narrow path (the gadgets as written, the Ampere system byte-for-byte) whenever the
plain decomposition of ``v`` fits, this path otherwise.  The partition is recorded in ``System.wide`` for the hint
generators (``hints_torch.py``, ``unit.flatten_hints``), which compute the extra hints from the same exact integer
sums they already form.
"""
from __future__ import annotations

import math

from verity_numerical.checker.ctx import Table
from verity_numerical.checker.params import Params
from verity_numerical.checker.tables import norm_table

from .compile import P, Expr, LigeroCtx

__all__ = ["H", "needs_wide", "partition", "wide_group", "wide_hints"]

H = 16      # limb width


def needs_wide(p: Params) -> bool:
    """The narrow ``SignedSum`` writes ``v = S - 2 sgn S`` and decomposes it into ``s_bits`` bits: that linear
    identity has the integer interval ``(-s_max, 2^s_bits - 1 + s_max)``, which must sit inside ``(-p, p)``."""
    return 2**p.s_bits - 1 + p.s_max >= P


def _fits(m: int) -> bool:
    """A magnitude ``|A| <= m`` admits a bit decomposition inside the field: ``v in [0, 2^b)``, identity interval
    ``(-m, 2^b - 1 + m)``."""
    b = max(1, m.bit_length())
    return 2**b - 1 + m < P


def partition(ctx: LigeroCtx, terms: list) -> list[list[int]]:
    """Greedy: consecutive runs of term indices whose signed partial sum's integer interval fits ``_fits``."""
    parts: list[list[int]] = []
    cur: list[int] = []
    acc = None
    for i, (s, q) in enumerate(terms):
        t = q - 2 * s * q
        cand = t if acc is None else acc + t
        iv = ctx.interval(cand)
        if iv is None:
            raise ValueError(f"wide sum: interval of term {i} unknown")
        m = max(-iv[0], iv[1])
        if cur and not _fits(m):
            parts.append(cur)
            cur, acc = [i], t
            iv = ctx.interval(t)
            m = max(-iv[0], iv[1])
            if not _fits(m):
                raise ValueError(f"wide sum: a single term does not fit the field: {iv}")
        else:
            cur.append(i)
            acc = cand
    if cur:
        parts.append(cur)
    return parts


def _limbs(ctx: LigeroCtx, x: Expr, nbits: int, name: str) -> tuple[Expr, Expr]:
    """``x = hi 2^H + lo`` from ``x``'s (memoised) ``nbits``-bit decomposition; both limbs with learned ranges."""
    rows = ctx.bit_rows(x, nbits, name)
    h = min(H, nbits)
    lo = Expr(ctx, {rows[i]: 2**i for i in range(h)}, 0)
    hi = Expr(ctx, {rows[i]: 2 ** (i - h) for i in range(h, nbits)}, 0)
    ctx.learn(lo, 0, 2**h)
    ctx.learn(hi, 0, 2 ** max(nbits - h, 0))
    return lo, hi


def _signed(x: Expr, sgn: Expr) -> Expr:
    return x - 2 * sgn * x


def leadw_table(p: Params) -> Table:
    """LEADW: leading-bit position ``l`` -> ``(lo_l, ispan_l, lo_h, ispan_h)``: ``(vl - lo_l) ispan_l < 2^H`` and
    ``(vh - lo_h) ispan_h < 2^HB`` (``HB = s_bits - H``) together say ``2^(l-1) <= v < 2^l`` (``l = 0``: ``v = 0``)."""
    HB = p.s_bits - H

    def fn(l: int) -> tuple[int, int, int, int]:
        if l == 0:
            return 0, 2**H, 0, 2**HB
        if l <= H:
            return 2 ** (l - 1), 2 ** (H - (l - 1)), 0, 2**HB
        j = l - H                               # window on the high limb: 2^(j-1) <= vh < 2^j
        return 0, 1, 2 ** (j - 1), 2 ** (HB - (j - 1))

    return Table("LEADW", 0, p.l_hi + 1, ("lo_l", "ispan_l", "lo_h", "ispan_h"), fn)


def wide_group(ctx: LigeroCtx, p: Params, tag: str, terms: list, E: Expr):
    """``SignedSum`` + ``LeadingBit`` + ``Normalise`` of one group on limbs; returns ``(sgn, e1, M)`` for ``StateOut``
    and records the partition in ``ctx.sys.wide[tag]``."""
    HB = p.s_bits - H
    parts = partition(ctx, terms)
    ctx.sys.wide[tag] = parts

    # --- partial sums in sign-magnitude form; their limbs from the bit rows ------------------------------------------
    ctx.scope = f"{tag}.sum"
    hs = ctx.hints(f"{tag}.sum", ["sgn", "z", "vh", "vl", "carry"] + [f"part[{k}].sgn" for k in range(len(parts))])
    sgn, z, vh, vl, carry = hs["sgn"], hs["z"], hs["vh"], hs["vl"], hs["carry"]
    SL = Expr(ctx)
    SH = Expr(ctx)
    for k, idx in enumerate(parts):
        A = sum((terms[i][1] - 2 * terms[i][0] * terms[i][1] for i in idx), Expr(ctx))
        iv = ctx.interval(A)
        b = max(1, max(-iv[0], iv[1]).bit_length())
        sk = hs[f"part[{k}].sgn"]
        ctx.boolean(f"part[{k}].sgn", sk)
        vk = _signed(A, sk)
        ctx.rng(f"part[{k}].v", vk, 0, 2**b)
        lo, hi = _limbs(ctx, vk, b, f"part[{k}].v")
        SL = SL + _signed(lo, sk)
        SH = SH + _signed(hi, sk)

    # --- recombination: S = (1 - 2 sgn) (vh 2^H + vl) as two limb identities with a carry ---------------------------
    ctx.boolean("sgn", sgn)
    ctx.boolean("z", z)
    ctx.rng("vl", vl, 0, 2**H)
    ctx.rng("vh", vh, 0, 2**HB)
    cb = (len(parts) + 1).bit_length() + 1
    ctx.rng("carry", carry + 2 ** (cb - 1), 0, 2**cb)
    ctx.eq("lo", SL - _signed(vl, sgn), carry * 2**H)
    ctx.eq("hi", SH - _signed(vh, sgn) + carry, 0)
    v = vh * 2**H + vl
    ctx.eq("z0", z * (vh + vl), 0)
    ctx.rng("z1", vh + vl + z - 1, 0, 2 ** (max(H, HB) + 1))
    ctx.eq("zero_sign", sgn * z, 0)

    # --- leading bit, one window per limb ---------------------------------------------------------------------------
    ctx.scope = f"{tag}.lead"
    l = ctx.hint(f"{tag}.lead.l")
    lo_l, ispan_l, lo_h, ispan_h = ctx.lookup("lead", leadw_table(p), l)
    ctx.lt_pow("window_lo", vl - lo_l, ispan_l, H)
    ctx.lt_pow("window_hi", vh - lo_h, ispan_h, HB)

    # --- normalisation: div M + r = mul v, split at bit H ------------------------------------------------------------
    ctx.scope = f"{tag}.norm"
    hn = ctx.hints(f"{tag}.norm", ["clamp", "M", "r", "carry"])
    clamp, M, r, cn = hn["clamp"], hn["M"], hn["r"], hn["carry"]
    ctx.boolean("clamp", clamp)
    ctx.rng("M", M, 0, 2**p.norm_bits)
    ctx.rng("r", r, 0, 2**p.norm_rem_bits)
    e_raw = E + l - p.width
    ctx.rng("clamp_proof", clamp * (p.acc_e_min - 1 - e_raw) + (1 - clamp) * (e_raw - p.acc_e_min), 0, p.cmp_rows)
    e1 = e_raw + clamp * (p.acc_e_min - e_raw)
    kk = p.norm_k(E, e1)
    mul, div, idiv = ctx.lookup("norm", norm_table(p), kk)
    # theorem check per limb: v < 2^B(k), B = min(s_bits, acc_sig - k)  <=>  vh < 2^max(B - H, 0), vl < 2^min(B, H);
    # it bounds mul * v < 2^acc_sig (k > 0) -- mul = 1 otherwise -- which is what the whitelist below declares.
    B = lambda kv: min(p.s_bits, p.acc_sig - kv)                      # noqa: E731
    vbound_h = ctx.table_by_key(mul, lambda kv: 2 ** max(B(kv) - H, 0))
    vbound_l = ctx.table_by_key(mul, lambda kv: 2 ** min(B(kv), H))
    ctx.lt_table("vbound_hi", vh, vbound_h, HB)
    ctx.lt_table("vbound_lo", vl, vbound_l, H)
    Ml, Mh = _limbs(ctx, M, p.norm_bits, "M")
    nb = max(p.norm_rem_bits + H, p.acc_sig) - H + 1                  # |carry| < 2^nb
    ctx.rng("carry", cn + 2**nb, 0, 2 ** (nb + 1))
    ctx.eq("div_lo", div * Ml + r - mul * vl, cn * 2**H)
    ctx.eq("div_hi", div * Mh - mul * vh + cn, 0)
    ctx.lt_pow("rem", r, idiv, p.norm_rem_bits)                        # r < div
    M_out = M * 2**p.norm_up if p.norm_up else M
    return sgn, e1, M_out


# ---------------------------------------------------------------------------------------------------------------------
# Hints (exact integers; ``S`` and ``terms`` as the hint generators already form them)
# ---------------------------------------------------------------------------------------------------------------------

def wide_hints(p: Params, parts: list[list[int]], terms: list[tuple[int, int]], v: int, M: int, r: int, k: int) -> dict[str, int]:
    """The extra hint rows of one wide group (names relative to the group tag), from the signed ``terms``
    ``(s_i, q_i)``, the magnitude ``v = |S|``, and ``Normalise``'s ``(M, r, k)``: partial signs, ``v``'s limbs and
    the two carries."""
    out: dict[str, int] = {}
    sgn = 0
    S = 0
    SL = SH = 0
    for kidx, idx in enumerate(parts):
        A = sum(q if not s else -q for s, q in (terms[i] for i in idx))
        S += A
        sk = int(A < 0)
        out[f"sum.part[{kidx}].sgn"] = sk
        vk = -A if sk else A
        SL += (1 - 2 * sk) * (vk & (2**H - 1))
        SH += (1 - 2 * sk) * (vk >> H)
    sgn = int(S < 0)
    assert v == (-S if sgn else S)
    vl, vh = v & (2**H - 1), v >> H
    out["sum.vl"], out["sum.vh"] = vl, vh
    c = SL - (1 - 2 * sgn) * vl
    assert c % 2**H == 0 and SH - (1 - 2 * sgn) * vh + c // 2**H == 0
    out["sum.carry"] = c // 2**H
    mul, div = 2 ** max(k, 0), 2 ** max(-k, 0)
    Ml, Mh = M & (2**H - 1), M >> H
    cn = div * Ml + r - mul * (v & (2**H - 1))
    assert cn % 2**H == 0 and div * Mh - mul * (v >> H) + cn // 2**H == 0
    out["norm.carry"] = cn // 2**H
    return out
