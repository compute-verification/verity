"""The v3 transition unit: v1's statement (operands, products, alignments, group maximum, exact sum, normalisation,
epilogue all proven in the relation) on **private** operand words, in the fewest rows we know how.

Where v2 (``pubsel``) hands every function of the public operands to the verifier, v3 keeps the operand words as
committed witness wires ``a_words[i]`` / ``b_words[i]`` (the sum of their hint-bit rows) and proves everything from
them; the public words enter only through one pin row per word (``a[i]``, ``b[i]``) and the linear ``a_words[i] = a[i]``
(the "pin variant"; lane hash-relation replaces those pin rows by a Poseidon2 sponge over the same wires).  Row-free
techniques throughout:

* operand decode: the word's 16 (8) bits are hint bits ``s``, ``x[j]`` (8 / 4 bits of ``x = t - nz``), ``f[j]`` plus the
  hidden-bit flag ``nz``; ``nz = 0 -> x = 0`` by a zero-product, and ``x`` off the top by ONE inverse (BF16: ``floor(x/2) !=
  127`` excludes ``t = 255`` (inf / NaN) and ``t = 256`` (the ``(x, nz)`` encoding's collision with the sign bit); FP8:
  ``x != 15`` excludes the collision -- the E4M3 NaN word (``t = 15, f = 7``) is rejected by the verifier's word check in
  the pin variant, see ``public_vectors_v3``);
* product: ``m_a m_b = sum 2^j u_j`` as one row-free quadratic on the product's hint bits (the bits are what the shifter
  windows read);
* alignment (``AlignProduct``: ``X = floor(u 2^(PS - d))``, ``d = E - e_i``, zero above ``d >= PB``): two stages of
  one-hot selection.  Fine: ``R in [0, RB)`` picks the radix-``2^RB`` digits ``D_k`` of ``Y = u 2^R`` as windows of ``u``'s
  bits; coarse: ``c in {-1, 0, 1, 2}`` picks ``X = floor(Y / 2^(RB c))`` -- a digit selection, affine in the ``D_k`` -- so
  ``L = PS - d = R - RB c``; two more coarse options ``dead`` (``d >= PB``: ``d = PB + (RB - 1 - R) + RB K``, ``K`` a
  short range) and ``zero`` (``u = 0``: no exponent constraint, the product's exponent does not enter the maximum, as
  in v1).  ``(c = -1, R)`` pairs with ``R + RB > PS`` are forbidden (row-free products of selectors).  The sign is the
  XOR of the operands' signs (one hint, one row-free quadratic) applied by one product row.
* group maximum: ``E`` is a hint; ``E >= e_i`` for every nonzero product is a theorem of its alignment one-hot, ``E >=
  e_acc`` of the accumulator's shift lookup; ``E`` equals a candidate by a one-hot ``which`` over {accumulator, products}
  (``_pick`` with target ``E``) whose chosen product must be nonzero: ``C = pick(which, [1, X_1, ..., X_k])`` is inverted.
  (``e_acc >= floor`` always, so ``floor`` never wins strictly.)
* the sum, normalisation, output flags, packing and epilogue are v2's (imported helpers; the sum's ``Q`` is the sum of
  the signed aligned products, in two limbs when the relation is wide).

Integer semantics: ``pubsel.relation.operand_decode_int`` / ``aligned_product`` (v1's, shared with v2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from verity_numerical.checker.params import Params

from ..compile import Expr, LigeroCtx, Quadratic, System
from ..pubsel.relation import (ACC_FRAC, CMP_BITS, LIMB, Y_PACK_BITS, Plan, _bit_exprs, _clamp_table, _hint, _norm_window,
                               _pick, _window, _zero, plan)

__all__ = ["ShiftPlan", "compile_v3_unit", "public_vectors_v3", "shift_plan"]


# ---------------------------------------------------------------------------------------------------------------------
# Derived widths of the shifter
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftPlan:
    UB: int             # the product u < 2^UB (twice the operand significand width)
    RB: int             # radix bits of the digits = the fine range R in [0, RB)
    ND: int             # digits of Y = u 2^R
    kb: int             # bits of the dead range's quotient K
    d_max: int          # the largest E - e_i
    forbidden: tuple[int, ...]   # fine positions R that cannot pair with c = -1 (L = R + RB > PS)

    @property
    def ncoarse(self) -> int:
        return 6        # c in {-1, 0, 1, 2}, dead, zero


def shift_plan(p: Params, pl: Plan) -> ShiftPlan:
    UB = 2 * (pl.op_frac + 1)
    if pl.PB != pl.PS + UB:
        raise ValueError(f"prod_bits {pl.PB} != prod_shift {pl.PS} + {UB}")
    RB = UB // 2
    if pl.wide and LIMB % RB:
        raise ValueError("the wide limb must be a whole number of digits")
    ND = math.ceil((UB + RB - 1) / RB)
    d_max = max(p.ep_hi, p.acc_e_max) - p.ep_lo
    # dead: d = PB + (RB - 1 - R) + RB K covers [PB, PB + RB - 1 + RB (2^kb - 1)]
    kb = max(1, math.ceil(math.log2((d_max - pl.PB - (RB - 1)) / RB + 1)))
    forbidden = tuple(r for r in range(RB) if r + RB > pl.PS)
    return ShiftPlan(UB=UB, RB=RB, ND=ND, kb=kb, d_max=d_max, forbidden=forbidden)


# ---------------------------------------------------------------------------------------------------------------------
# Row-free helpers (in addition to pubsel's)
# ---------------------------------------------------------------------------------------------------------------------

def _quad(ctx: LigeroCtx, name: str, a: Expr, b: Expr, c: Expr) -> None:
    """``a * b = c`` as one quadratic constraint on affine forms (no product row)."""
    ctx.sys.quadratic.append(Quadratic(f"{ctx.scope}.{name}", a, b, c))


def _hint_bit(ctx: LigeroCtx, name: str) -> Expr:
    """A hint row constrained boolean; the row's integer range is then {0, 1} (a bit is a bit as a field element)."""
    b = _hint(ctx, name)
    ctx.boolean(name.rsplit(".", 1)[-1], b)
    (idx,) = b.lin
    ctx.sys.rows[idx].lo, ctx.sys.rows[idx].hi = 0, 2
    return b


def _hint_bits(ctx: LigeroCtx, name: str, n: int) -> list[Expr]:
    return [_hint_bit(ctx, f"{name}[{j}]") for j in range(n)]


def _onehot(ctx: LigeroCtx, name: str, n: int) -> list[Expr]:
    """``n`` hint bits summing to one."""
    sels = _hint_bits(ctx, name, n)
    ctx.eq(f"{name.rsplit('.', 1)[-1]}.onehot", sum(sels[1:], sels[0]), 1)
    return sels


def _weighted(ctx: LigeroCtx, bits: list[Expr], base: int = 0) -> Expr:
    e = Expr(ctx)
    for j, b in enumerate(bits):
        e = e + b * 2 ** (j + base)
    if bits:
        ctx._learn(e, 0, 2 ** (len(bits) + base))
    return e


# ---------------------------------------------------------------------------------------------------------------------
# The gadgets
# ---------------------------------------------------------------------------------------------------------------------

def _decode_word(ctx: LigeroCtx, pl: Plan, name: str):
    """The committed operand word ``name`` (``a_words[i]`` / ``b_words[i]``) from its hint bits.  Returns (word wire, s, m, e)
    with ``m = f + 2^frac nz``, ``e = x - (bias - 1)``, ``x = t - nz`` -- ``operand_decode_int`` on the same word."""
    ctx.scope = name
    s = _hint_bit(ctx, f"{name}.s")
    nz = _hint_bit(ctx, f"{name}.nz")
    xb = _hint_bits(ctx, f"{name}.x", pl.op_exp)
    fb = _hint_bits(ctx, f"{name}.f", pl.op_frac)
    x, f = _weighted(ctx, xb), _weighted(ctx, fb)
    _zero(ctx, "nz0", 1 - nz, x)                                  # nz = 0 -> x = 0 (t = 0); nz = 1 -> t = x + 1 >= 1
    top = 2**pl.op_exp - 1
    if pl.e4m3:
        ctx.inverse_nonzero("top", x - top)                       # x != 2^exp - 1: t = x + nz <= 2^exp - 1 (no overflow into s)
    else:
        # x not in {2^exp - 2, 2^exp - 1}: t != 2^exp - 1 (inf / NaN) and t != 2^exp (overflow into s), one inverse
        ctx.inverse_nonzero("top", _weighted(ctx, xb[1:]) - (2 ** (pl.op_exp - 1) - 1))
    word = s * 2 ** (pl.op_word_bits - 1) + (x + nz) * 2**pl.op_frac + f
    ctx._learn(word, 0, 2**pl.op_word_bits + 2**pl.op_frac)
    m = f + nz * 2**pl.op_frac
    ctx._learn(m, 0, 2 ** (pl.op_frac + 1))
    e = x - (pl.op_bias - 1)
    return word, s, m, e


def _align(ctx: LigeroCtx, pl: Plan, sp: ShiftPlan, scope: str, ubits: list[Expr], u: Expr, E: Expr, e_i: Expr):
    """One product's alignment: returns (X or (XL, XH), sels) with the exponent constraint stated.  ``E - e_i = PS - R + RB c``
    (live), ``= PB + (RB - 1 - R) + RB K`` (dead), unconstrained when ``u = 0`` (zero)."""
    RB, PS, PB = sp.RB, pl.PS, pl.PB
    ctx.scope = scope
    rsel = _onehot(ctx, f"{scope}.r", RB)
    R = sum((r * rsel[r] for r in range(1, RB)), Expr(ctx))
    D = [_pick(ctx, f"D{k}", rsel, [_window(ctx, ubits, r, k * RB, (k + 1) * RB, k * RB) for r in range(RB)],
               hint=f"{scope}.D{k}") for k in range(sp.ND)]
    csel = _onehot(ctx, f"{scope}.c", sp.ncoarse)
    c_live, dead, zero = csel[:4], csel[4], csel[5]
    cval = -c_live[0] + c_live[2] + 2 * c_live[3]
    zero_e = Expr(ctx)

    def digits(c: int, lo: int, hi: int) -> Expr:
        """``floor(Y / 2^(RB c))`` restricted to bit weights [lo, hi), rebased at ``lo``."""
        e = Expr(ctx)
        top = 0
        for k in range(sp.ND):
            w = (k - c) * RB
            if lo <= w < hi:
                e = e + D[k] * 2 ** (w - lo)
                top += 2 ** (w - lo) * ctx.interval(D[k])[1]
        if e.lin:
            ctx._learn(e, 0, top + 1)
        return e

    # X = floor(u 2^L) < 2^PB since L <= PS (the forbidden pairs); the candidates' hull is looser (it does not see the
    # coupling of R and c), so the theorem is asserted (logged) for the sum's audit.
    if pl.wide:
        XL = _pick(ctx, "XL", csel, [digits(c, 0, LIMB) for c in (-1, 0, 1, 2)] + [zero_e, zero_e], hint=f"{scope}.XL")
        XH = _pick(ctx, "XH", csel, [digits(c, LIMB, PB + RB) for c in (-1, 0, 1, 2)] + [zero_e, zero_e], hint=f"{scope}.XH")
        ctx.learn(XL, 0, 2**LIMB)
        ctx.learn(XH, 0, 2 ** (PB - LIMB))
        X: Expr | tuple = (XL, XH)
        Xfull = XL + XH * 2**LIMB
        ctx._learn(Xfull, 0, 2**PB)
    else:
        X = _pick(ctx, "X", csel, [digits(c, 0, PB + RB) for c in (-1, 0, 1, 2)] + [zero_e, zero_e], hint=f"{scope}.X")
        ctx.learn(X, 0, 2**PB)
        Xfull = X
    for r in sp.forbidden:
        _zero(ctx, f"forbid{r}", c_live[0], rsel[r])
    _zero(ctx, "zero", zero, u)                                   # the zero option needs u = 0
    K = _hint(ctx, f"{scope}.K")
    ctx.rng("K", K, 0, 2**sp.kb)
    _zero(ctx, "Kdead", 1 - dead, K)                              # K = 0 unless dead
    d_expr = PS - R + RB * cval + dead * (PB + RB - 1 - PS) + RB * K
    _quad(ctx, "exp", 1 - zero, E - e_i - d_expr, zero_e)         # (1 - zero) (E - e_i - d) = 0
    return X, Xfull


# ---------------------------------------------------------------------------------------------------------------------
# The compiled unit
# ---------------------------------------------------------------------------------------------------------------------

def compile_v3_unit(p: Params, chain: bool = False) -> System:
    pl = plan(p)
    sp = shift_plan(p, pl)
    ctx = LigeroCtx(p)
    nb, drop, W, ss, PB, SB = pl.nb, pl.drop, pl.W, pl.ss, pl.PB, pl.SB
    k = sum(pl.groups)

    # --- the accumulator: word components as linked hint rows (chain) or pins (unit) -- v2's -------------------------
    ctx.scope = "acc"
    if chain:
        s_c, t_c, f_c, nz, zc = (_hint(ctx, f"acc.{n}") for n in ("s", "t", "f", "nz", "z"))
        ctx.boolean("s", s_c)
        ctx.boolean("nz", nz)
        ctx.boolean("z", zc)
        ctx.rng("t", t_c, 0, 2**p.acc_exp)
        ctx.inverse_nonzero("t.finite", p.acc_t_finite - t_c)
        _zero(ctx, "nz0", 1 - nz, t_c)
        ctx.inverse_nonzero("nz1", t_c + 1 - nz)
        f_rows = ctx._bits(f_c, ACC_FRAC, "f")
        ctx._learn(f_c, 0, 2**ACC_FRAC)
        M24 = f_c + nz * 2**ACC_FRAC
        _zero(ctx, "z0", zc, M24)
        ctx.inverse_nonzero("z1", M24 + zc)
        mbits = _bit_exprs(ctx, f_rows[drop:]) + [nz]
        e = t_c - nz - (p.acc_bias - 1)
        e_c = e + zc * (p.floor - e)
    else:
        s_c = ctx.pin("c.s", 0, 2)
        e_c = ctx.pin("c.e_hat", p.exp_lo, p.acc_e_max + 1)
        Mt = ctx.pin("c.Mt", 0, 2**nb)
        mbits = _bit_exprs(ctx, ctx._bits(Mt, nb, "Mt"))
    state = (s_c, e_c, mbits)

    # --- the operands: committed words (hint bits) pinned to the public words -----------------------------------------
    ops = []
    for i in range(k):
        wa, sa, ma, ea = _decode_word(ctx, pl, f"a_words[{i}]")
        pin = ctx.pin(f"a[{i}]", 0, 2**pl.op_word_bits)
        ctx.eq("pin", wa, pin)
        wb, sb, mb, eb = _decode_word(ctx, pl, f"b_words[{i}]")
        pin = ctx.pin(f"b[{i}]", 0, 2**pl.op_word_bits)
        ctx.eq("pin", wb, pin)
        ops.append((sa, ma, ea, sb, mb, eb))

    # --- the groups --------------------------------------------------------------------------------------------------
    ovfs = []
    final = None
    start = 0
    e_hi = max(p.ep_hi, p.acc_e_max)
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        s_st, e_st, mb = state
        idx = list(range(start, start + size))
        start += size

        # the group maximum E (a hint), certified below
        ctx.scope = f"{tag}.max"
        E = _hint(ctx, f"{tag}.E")
        ctx.learn(E, p.floor, e_hi + 1)

        # the products: u = m_a m_b in bits, sign, alignment to E
        sX, Xs, es = [], [], []
        for i in idx:
            sa, ma, ea, sb, mb_, eb = ops[i]
            scope = f"{tag}.p[{i}]"
            ctx.scope = scope
            ubits = _hint_bits(ctx, f"{scope}.u", sp.UB)
            u = _weighted(ctx, ubits)
            _quad(ctx, "u", ma, mb_, u)                           # m_a m_b = u (< 2^UB < p)
            e_i = ea + eb
            X, Xfull = _align(ctx, pl, sp, scope, ubits, u, E, e_i)
            ctx.scope = scope
            s_i = _hint_bit(ctx, f"{scope}.s")
            _quad(ctx, "xor", sa, 1 - 2 * sb, s_i - sb)           # s_i = s_a xor s_b
            if pl.wide:
                sX.append((X[0] - 2 * ctx.product(s_i, X[0]), X[1] - 2 * ctx.product(s_i, X[1])))
            else:
                sX.append(X - 2 * ctx.product(s_i, X))
            Xs.append(Xfull)
            es.append(e_i)

        # E equals one candidate: the accumulator's exponent or a nonzero product's
        ctx.scope = f"{tag}.max"
        which = _onehot(ctx, f"{tag}.which", size + 1)
        _pick(ctx, "E", which, [e_st] + es, target=E)
        C = _pick(ctx, "C", which, [Expr(ctx, None, 1)] + Xs, hint=f"{tag}.C")
        ctx.inverse_nonzero("C", C)                               # the chosen product is nonzero (d = 0 -> X = u 2^PS)

        # the accumulator's term: floor(Mt 2^ss / 2^min(dacc, W)), dacc = E - e_acc >= 0 by the lookup (E >= e_acc)
        ctx.scope = f"{tag}.acc"
        dacc = E - e_st
        (tsel,) = ctx.lookup("tsel", _clamp_table("TSEL", W), dacc, range_key=False)
        tsels = _bit_exprs(ctx, tsel.sel_rows)
        zero = Expr(ctx)
        if pl.wide:
            qL = _pick(ctx, "qL", tsels, [_window(ctx, mb, ss - t, 0, LIMB) for t in range(W)] + [zero], hint=f"{tag}.qL")
            qH = _pick(ctx, "qH", tsels, [_window(ctx, mb, ss - t, LIMB, W, LIMB) for t in range(W)] + [zero], hint=f"{tag}.qH")
        else:
            q_acc = _pick(ctx, "q", tsels, [_window(ctx, mb, ss - t, 0, W) for t in range(W)] + [zero], hint=f"{tag}.q")

        # the exact signed sum S = (1 - 2 s_acc) q_acc + sum_i sX_i = (1 - 2 sgn) v
        ctx.scope = f"{tag}.sum"
        sgn = _hint(ctx, f"{tag}.sgn")
        ctx.boolean("sgn", sgn)
        sign_st = 1 - 2 * s_st
        if pl.wide:
            vL, vH = _hint(ctx, f"{tag}.vL"), _hint(ctx, f"{tag}.vH")
            rL = ctx._bits(vL, LIMB, "vL")
            rH = ctx._bits(vH, SB - LIMB, "vH")
            carry = _hint(ctx, f"{tag}.carry")
            cb = max(3, (size + 2).bit_length() + 1)
            ctx.rng("carry", carry + 2 ** (cb - 1), 0, 2**cb)
            QL = sum((t[0] for t in sX[1:]), sX[0][0])
            QH = sum((t[1] for t in sX[1:]), sX[0][1])
            ctx.eq("sumL", sign_st * qL + QL - (vL - 2 * sgn * vL) - 2**LIMB * carry, 0)
            ctx.eq("sumH", sign_st * qH + QH - (vH - 2 * sgn * vH) + carry, 0)
            vbits = _bit_exprs(ctx, rL + rH)
        else:
            v = _hint(ctx, f"{tag}.v")
            rv = ctx._bits(v, SB, "v")
            Q = sum(sX[1:], sX[0])
            ctx.eq("sum", sign_st * q_acc + Q, v - 2 * sgn * v)
            vbits = _bit_exprs(ctx, rv)

        # RZ normalisation, flags, state -- v2's
        ctx.scope = f"{tag}.norm"
        nsels = [_hint(ctx, f"{tag}.nsel[{i}]") for i in range(pl.n_t)]
        for i, s in enumerate(nsels):
            ctx.boolean(f"nsel[{i}]", s)
        ctx.eq("nsel.onehot", sum(nsels[1:], nsels[0]), 1)
        hid, fo = _hint(ctx, f"{tag}.hid"), _hint(ctx, f"{tag}.fo")
        ctx.boolean("hid", hid)
        fo_rows = ctx._bits(fo, nb - 1, "fo")
        Mn = hid * 2 ** (nb - 1) + fo
        ctx._learn(Mn, 0, 2**nb)
        _pick(ctx, "Mn", nsels, [_norm_window(ctx, vbits, t, nb) for t in range(pl.t_lo, pl.t_hi + 1)], target=Mn)
        t_sel = sum((t * s for t, s in zip(range(pl.t_lo + 1, pl.t_hi + 1), nsels[1:])), pl.t_lo * nsels[0])
        ctx._learn(t_sel, pl.t_lo, pl.t_hi + 1)
        tc = p.acc_e_min - E + p.width - nb
        dt = t_sel - tc
        ctx.rng("dt", dt, 0, 2**CMP_BITS)
        _zero(ctx, "clamp", 1 - hid, dt)
        e1 = E + t_sel + nb - p.width
        ctx.scope = f"{tag}.out"
        z, ovf = _hint(ctx, f"{tag}.z"), _hint(ctx, f"{tag}.ovf")
        ctx.boolean("z", z)
        ctx.boolean("ovf", ovf)
        _zero(ctx, "z0", z, Mn)
        ctx.inverse_nonzero("z1", Mn + z)
        cmp = 2 * (ovf * e1) - (2 * p.acc_e_max + 1) * ovf + p.acc_e_max - e1
        ctx.rng("ovf.proof", cmp, 0, 2**CMP_BITS)
        ovf_eff = ovf - ovf * z
        s_out = sgn - sgn * z
        e_hat = e1 + z * (p.floor - e1)
        ovfs.append((ovf_eff, sgn))
        final = (s_out, e1, hid, fo, z)
        state = (s_out, e_hat, _bit_exprs(ctx, fo_rows) + [hid])

    # --- pack, component-wise -- v2's ---------------------------------------------------------------------------------
    ctx.scope = "pack"
    s_out, e1, hid, fo, z = final
    fld = e1 + (p.acc_bias - 1) + hid
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - none_yet * ovf_g
    live = (1 - any_ovf) * (1 - z)
    ys, yt = any_ovf * inf_sign + live * s_out, any_ovf * p.acc_t_finite + live * fld
    yf_hi = live * fo
    yf = yf_hi * 2**drop
    ctx.sys.outputs = {"y.s": ys, "y.t": yt, "y.f": yf}
    if not chain:
        ctx.eq_pub("y.s", ys, "y.s")
        ctx.eq_pub("y.t", yt, "y.t")
        ctx.eq_pub("y.f", yf, "y.f")
        return ctx.sys

    if pl.fp8:
        y16 = ys * 2 ** (Y_PACK_BITS - 1) + yt * 2 ** (Y_PACK_BITS - 1 - p.acc_exp) + yf_hi
    else:
        ctx.scope = "epi"
        L = ACC_FRAC - p.frac
        fbits = _bit_exprs(ctx, fo_rows)
        fhi = live * _window(ctx, fbits, 0, L, ACC_FRAC, L)
        hi = ys * 2 ** (p.word_bits - 1) + yt * 2**p.frac + fhi
        sticky = _window(ctx, fbits, 0, 0, L - 1) + fbits[L]
        zl = _hint(ctx, "epi.zl")
        ctx.boolean("zl", zl)
        _zero(ctx, "zl0", zl, sticky)
        ctx.inverse_nonzero("zl1", sticky + zl)
        carry = live * ctx.product(fbits[L - 1], 1 - zl)
        y16 = hi + carry
    ctx.sys.outputs["y16"] = y16
    names = {n: i for i, n in ctx.sys.hints}
    ctx.sys.chain = {"c": [names["acc.s"], names["acc.t"], names["acc.f"]], "y": [ys, yt, yf], "y16": y16}
    return ctx.sys


# ---------------------------------------------------------------------------------------------------------------------
# The verifier's public work: the words themselves (finite, in range) and unit mode's accumulator / output pins
# ---------------------------------------------------------------------------------------------------------------------

def public_vectors_v3(p: Params, c, a, b, y, device="cpu"):
    """``a``/``b`` (B, k) operand words -> the word pins ``a[i]``, ``b[i]``; unit mode (``c`` given) also ``c.s, c.e_hat,
    c.Mt`` and ``y.s, y.t, y.f``.  Raises ``ValueError`` off the domain (a non-finite operand word: the pin variant's word
    check, which the relation itself covers except for the E4M3 NaN word)."""
    import torch

    from ..pubsel.relation import _decode_operands

    pl = plan(p)
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    k = sum(pl.groups)
    if a.ndim != 2 or a.shape[1] != k or b.shape != a.shape:
        raise ValueError("shape mismatch")
    bad: list[str] = []
    _decode_operands(pl, a, bad, "a")
    _decode_operands(pl, b, bad, "b")
    out: dict = {}
    for i in range(k):
        out[f"a[{i}]"] = a[:, i]
        out[f"b[{i}]"] = b[:, i]
    if c is not None:
        c = torch.as_tensor(c, dtype=torch.int64, device=device)
        y = torch.as_tensor(y, dtype=torch.int64, device=device)
        if c.shape != (a.shape[0],) or y.shape != c.shape:
            raise ValueError("shape mismatch")
        if ((c < 0) | (c >= 2**32)).any():
            bad.append("x.c range")
        if ((y < 0) | (y >= 2**32)).any():
            bad.append("y range")
        t = (c >> ACC_FRAC) & (2**p.acc_exp - 1)
        f = c & (2**ACC_FRAC - 1)
        if (t == p.acc_t_finite).any():
            bad.append("acc.t off the finite range")
        nz = (t != 0).to(torch.int64)
        zz = ((t == 0) & (f == 0)).to(torch.int64)
        M = f + nz * 2**ACC_FRAC
        e = t - nz - (p.acc_bias - 1)
        out["c.s"] = c >> 31
        out["c.e_hat"] = e + zz * (p.floor - e)
        out["c.Mt"] = M >> pl.drop
        if pl.drop and (y & (2**pl.drop - 1)).any():
            bad.append("claimed word has nonzero low bits")
        out["y.s"] = y >> 31
        out["y.t"] = (y >> ACC_FRAC) & (2**p.acc_exp - 1)
        out["y.f"] = y & (2**ACC_FRAC - 1)
    if bad:
        raise ValueError("public inputs rejected: " + "; ".join(bad))
    return out
