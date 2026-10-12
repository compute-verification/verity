"""The BF16 / FP8 transition unit compiled with *public selection* (lane relmin-lookup): the ``-v2`` relations.

The relation is the one ``unit.py`` (BF16) and ``fp8/relation.py`` (FP8) prove -- the pinned instruction semantics of
``verity.ml.circuits.silicon.models.GroupSum`` seen through the checker's gadgets (``verity_numerical.checker.gadgets``: group maximum,
truncating alignment, exact signed sum, RZ normalisation with the subnormal clamp, overflow / zero flags, component-wise pack).
What changes is *who computes what*.  In v1 the prover commits the operand decodes, every product, every alignment
quotient / remainder and the normalisation's quotient / remainder, and proves each with bit decompositions (65-73 % of the
rows are bits).  Here the verifier computes everything that depends on the public operands alone, and the prover commits
only what depends on the private accumulator:

* the verifier pins, per group, ``G = max(floor, e_i)`` (the group maximum of the *products*) and the signed sum of the
  truncated aligned products for every possible extra shift ``j`` (``Q[j] = sum_i (1 - 2 s_i) align(u_i, j + G - e_i)``,
  ``j`` in ``[0, prod_bits)``; every term is shifted out for ``j >= prod_bits``).  With ``E = max(G, e_acc)`` the group's
  product sum is ``Q[E - G]``;
* the group maximum and both alignment shifts come from ONE one-hot over ``c = clamp(k, -W, PB)`` with ``k = e_acc - G`` an
  affine expression of the state and the pin (no shift hints): ``k >= 0`` gives ``E = G + k`` and ``Q[min(k, PB)]``
  (``Q[PB] = 0``), ``k < 0`` gives ``E = e_acc``, ``Q[0]`` and the accumulator shifted by ``min(-k, W)``; the key constraint
  ``k - sum_c sel_c c = diff`` with ``diff = 0`` off the ends and ``(1 - 2 sel_lo) diff`` in ``[0, 2^SHIFT_BITS)`` (one product
  row, SHIFT_BITS bits) makes the end selectors mean ``k >= PB`` / ``k <= -W``; ``delta = E - G`` is picked from the one-hot;
* the accumulator's aligned term ``floor(M 2^state_shift / 2^min(dacc, W))`` is a *window* of the accumulator's own bit rows
  (the range check of the linked fraction supplies them) selected by that one-hot: no quotient, no remainder, no power table;
* the exact sum ``S = (1 - 2 s_acc) q_acc + Q[delta] = (1 - 2 sgn) v`` with ``v`` committed in bits (two 16-bit limbs and a
  carry when ``2^s_bits + s_max`` would reach ``p``: bf16-hopper);
* the RZ normalisation is a window of ``v``'s bits: ``M = floor(v / 2^t)`` (``v 2^-t`` for ``t < 0``) with a one-hot over the
  shift ``t``; ``M = hid 2^(nb-1) + f`` with ``f`` in bits.  ``hid = 1`` (``M`` in ``[2^(nb-1), 2^nb)``) *is* the leading-bit
  proof, and the subnormal clamp is ``t = tc(E)`` when ``hid = 0``, ``t >= tc(E)`` when ``hid = 1``; the output exponent is
  ``E + t + nb - width`` in both cases.  The window carries the bits above the ``nb`` selected ones at weight ``2^nb``, so a
  selected window with a stray high bit cannot equal a value below ``2^nb``: truncation is exact and the sum stays in the
  field;
* the last group's normalised significand is picked from the sum's bit rows without committing its fraction (the bits
  above the selected window are forced to zero, the hidden bit is picked from the window's top bit), and so are the
  BF16 epilogue's fields; only an intermediate group (two-group units) commits its fraction, for the next alignment;
* every one-hot selection is *picked*, not summed (``_pick``): the selected value is a hint row ``X`` (or the value the
  relation already commits, ``M``) and each candidate costs one row-free quadratic ``sel_i (X - value_i) = 0`` instead of a
  product row.

One code path for the four relations, parametrised by ``Params`` (BF16: ``Params.from_model``) or ``FP8Params`` (the
adder-side widths with the lossy rescale: the adder's state is the top 14 bits of the FP32 significand).  ``chain=True`` is
the mode the prover runs (accumulator components as hint rows linked by the chain test; BF16: the RN-even epilogue on the
unit's own output; FP8: the 22-bit packed word at the chain end), ``chain=False`` pins the accumulator and the output
components (tests).  The witness rows are only the kinds ``witness.py`` / ``witness_device.py`` already derive (pin, hint,
prod, bit, sel, inv); the Ligero protocol is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity_numerical.checker.params import Params

from ..compile import Expr, LigeroCtx, Quadratic, System
from ..wide import needs_wide

__all__ = ["ACC_FRAC", "CMP_BITS", "LIMB", "Plan", "SHIFT_BITS", "Y_PACK_BITS", "aligned_product", "compile_v2_unit",
           "operand_decode_int", "plan", "public_pins_int", "public_vectors_v2"]

ACC_FRAC = 23                # FP32 fraction bits
SHIFT_BITS = 9               # delta, dacc in [0, 512) (>= shift_rows of every relation; conservative, sound)
CMP_BITS = 9                 # |difference| < 512 in the flag comparisons (overflow, clamp)
LIMB = 16                    # limb width of the wide sum
Y_PACK_BITS = 22             # FP8 chain end: s 2^21 + t 2^13 + (f >> 10)


@dataclass(frozen=True)
class Plan:
    """The derived widths of the v2 unit."""
    fp8: bool
    nb: int              # normalised magnitude bits = adder-side state bits (BF16 24, FP8 14)
    drop: int            # FP32 fraction bits below the adder state (BF16 0, FP8 10)
    W: int               # the state term < 2^W (acc_bits)
    ss: int              # state_shift: the state enters the adder as Mt 2^ss
    PB: int              # prod_bits: a product enters as u 2^PS < 2^PB
    PS: int              # prod_shift
    SB: int              # s_bits: |S| < 2^SB
    wide: bool           # the sum in two limbs with a carry
    t_lo: int            # normalisation shift t in [t_lo, t_hi]
    t_hi: int
    op_exp: int
    op_frac: int
    op_bias: int
    op_word_bits: int
    e4m3: bool           # operand NaN rule: e4m3 (0x7f / 0xff) or ieee (all-ones exponent)
    groups: tuple[int, ...]
    shift_bits: int = SHIFT_BITS   # the key-difference range: |e_acc - G| < 2^shift_bits (9 for every registered unit)

    @property
    def n_t(self) -> int:
        return self.t_hi - self.t_lo + 1

    def qmax(self, size: int) -> int:
        """|Q[j]| <= size (2^PB - 2^PS)."""
        return size * (2**self.PB - 2**self.PS)


def plan(p: Params) -> Plan:
    fp8 = hasattr(p, "op_exp")
    nb = p.norm_bits
    # the key difference |e_acc - G| is bounded by the relation's shift range; 9 bits for every registered unit, wider only
    # for folded multi-step units (several instruction steps per column) whose conservative exponent range passes 512
    shift_bits = max(SHIFT_BITS, (p.shift_rows - 1).bit_length())
    if fp8:
        op = (p.op_exp, p.op_frac, p.op_bias, 1 + p.op_exp + p.op_frac, True)
    else:
        op = (p.exp, p.frac, p.bias, p.word_bits, p.op_nan_rule == "e4m3")
    return Plan(fp8=fp8, nb=nb, drop=ACC_FRAC + 1 - nb, W=p.acc_bits, ss=p.state_shift, PB=p.prod_bits, PS=p.prod_shift,
                SB=p.s_bits, wide=needs_wide(p), t_lo=-nb, t_hi=-p.k_lo, op_exp=op[0], op_frac=op[1], op_bias=op[2],
                op_word_bits=op[3], e4m3=op[4], groups=tuple(p.groups), shift_bits=shift_bits)


# ---------------------------------------------------------------------------------------------------------------------
# Integer semantics shared by the verifier (pins), the hint generators and the tests
# ---------------------------------------------------------------------------------------------------------------------

def aligned_product(pl: Plan, u: int, d: int) -> int:
    """``AlignProduct``: the product term ``u 2^PS`` shifted right by ``d``, truncating."""
    if d <= pl.PS:
        return u << (pl.PS - d)
    if d < pl.PB:
        return u >> (d - pl.PS)
    return 0


def operand_decode_int(p: Params, pl: Plan, w: int) -> tuple[int, int, int]:
    """``Bf16Decode`` on an operand word: (s, m, e) with ``m = f + 2^frac [t != 0]``, ``e = max(t, 1) - bias``; raises off the
    finite domain."""
    if not 0 <= w < 2**pl.op_word_bits:
        raise ValueError("operand word out of range")
    t = (w >> pl.op_frac) & (2**pl.op_exp - 1)
    f = w & (2**pl.op_frac - 1)
    if pl.e4m3:
        if t == 2**pl.op_exp - 1 and f == 2**pl.op_frac - 1:
            raise ValueError("E4M3 NaN operand")
    elif t == 2**pl.op_exp - 1:
        raise ValueError("operand off the finite domain")
    nz = int(t != 0)
    return w >> (pl.op_word_bits - 1), f + nz * 2**pl.op_frac, t - nz - (pl.op_bias - 1)


def public_pins_int(p: Params, pl: Plan, a: list[int], b: list[int]) -> dict[str, int]:
    """The verifier's group pins of one unit on Python integers: ``g{g}.G`` and ``g{g}.Q[j]`` (or ``.lo`` / ``.hi`` limbs)."""
    out: dict[str, int] = {}
    start = 0
    for g, size in enumerate(pl.groups):
        terms = []
        for i in range(start, start + size):
            sa, ma, ea = operand_decode_int(p, pl, a[i])
            sb, mb, eb = operand_decode_int(p, pl, b[i])
            u = ma * mb
            e = ea + eb if u else p.floor
            terms.append((sa ^ sb, u, e))
        start += size
        G = max([p.floor] + [e for _, _, e in terms])
        out[f"g{g}.G"] = G
        for j in range(pl.PB):
            Q = sum((1 - 2 * s) * aligned_product(pl, u, j + G - e) for s, u, e in terms)
            if pl.wide:
                lo = Q & (2**LIMB - 1)
                out[f"g{g}.Q[{j}].lo"], out[f"g{g}.Q[{j}].hi"] = lo, (Q - lo) >> LIMB
            else:
                out[f"g{g}.Q[{j}]"] = Q
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Gadget helpers
# ---------------------------------------------------------------------------------------------------------------------

def _hint(ctx: LigeroCtx, name: str) -> Expr:
    """A hint row under its full path (row name == ``sys.hints`` name)."""
    sc, ctx.scope = ctx.scope, ""
    try:
        return ctx.hint(name)
    finally:
        ctx.scope = sc


def _zero(ctx: LigeroCtx, name: str, a: Expr, b: Expr) -> None:
    """``a * b = 0`` as one quadratic constraint (no product row)."""
    ctx.sys.quadratic.append(Quadratic(f"{ctx.scope}.{name}", a, b, Expr(ctx)))


def _pick(ctx: LigeroCtx, name: str, sels: list[Expr], values: list[Expr], target: Expr | None = None,
          hint: str | None = None) -> Expr:
    """The value a one-hot selects, without product rows: ``X`` is a hint row (or the given ``target`` expression) and every
    term is the row-free quadratic ``sel_i (X - value_i) = 0``.  Exactly one selector is 1 (the one-hot), so ``X`` equals
    that value as a field element, hence as the integer of that value's interval (every value is an integer inside
    ``(-p, p)``); ``X``'s interval is the hull, recorded with ``learn``.  Replaces ``sum_i sel_i value_i`` (one product row
    per term) at the cost of denser quadratic constraints."""
    X = target if target is not None else _hint(ctx, hint or f"{ctx.scope}.{name}")
    lo = hi = None
    for i, (s, v) in enumerate(zip(sels, values)):
        _zero(ctx, f"{name}[{i}]", s, X - v)
        iv = (v.c, v.c) if v.is_const() else ctx.interval(v)
        if iv is None:
            raise ValueError(f"{ctx.scope}.{name}[{i}]: value interval unknown")
        lo = iv[0] if lo is None else min(lo, iv[0])
        hi = iv[1] if hi is None else max(hi, iv[1])
    if target is None:
        ctx.learn(X, lo, hi + 1)
    return X


def _window(ctx: LigeroCtx, bits: list[Expr], shift: int, lo: int, hi: int, base: int = 0) -> Expr:
    """``sum 2^(pos + shift - base) bit_pos`` over the bits whose weight exponent ``pos + shift`` lies in ``[lo, hi)``."""
    e = Expr(ctx)
    top = 0
    for pos, bt in enumerate(bits):
        w = pos + shift
        if lo <= w < hi:
            e = e + bt * 2 ** (w - base)
            top += 2 ** (w - base)
    if e.lin:
        ctx._learn(e, 0, top + 1)
    return e


def _norm_window(ctx: LigeroCtx, vbits: list[Expr], t: int, nb: int) -> Expr:
    """``floor(v / 2^t)`` (``v 2^-t`` for ``t < 0``) as a window of ``nb`` bits, plus every bit above the window at weight
    ``2^nb``: equals an integer below ``2^nb`` iff the bits above are all zero and then it is the truncated shift."""
    e = Expr(ctx)
    cnt = 0
    for i, bt in enumerate(vbits):
        w = i - t
        if 0 <= w < nb:
            e = e + bt * 2**w
        elif w >= nb:
            e = e + bt * 2**nb
            cnt += 1
    if e.lin:
        ctx._learn(e, 0, 2**nb * (cnt + 1))
    return e


def _bit_exprs(ctx: LigeroCtx, rows: list[int]) -> list[Expr]:
    return [Expr(ctx, {r: 1}, 0) for r in rows]


# ---------------------------------------------------------------------------------------------------------------------
# The compiled unit
# ---------------------------------------------------------------------------------------------------------------------

def compile_v2_unit(p: Params, chain: bool = False) -> System:
    pl = plan(p)
    ctx = LigeroCtx(p)
    nb, drop, W, ss, PB, SB = pl.nb, pl.drop, pl.W, pl.ss, pl.PB, pl.SB

    # --- the accumulator: word components as linked hint rows (chain) or pins (unit) ----------------------------------
    ctx.scope = "acc"
    if chain:
        s_c, t_c, f_c, nz, zc = (_hint(ctx, f"acc.{n}") for n in ("s", "t", "f", "nz", "z"))
        ctx.boolean("s", s_c)
        ctx.boolean("nz", nz)
        ctx.boolean("z", zc)
        ctx.rng("t", t_c, 0, 2**p.acc_exp)                    # defines t as an 8-bit integer
        ctx.inverse_nonzero("t.finite", p.acc_t_finite - t_c)   # t != all ones: the accumulator is finite
        _zero(ctx, "nz0", 1 - nz, t_c)                        # nz = 0 -> t = 0
        ctx.inverse_nonzero("nz1", t_c + 1 - nz)              # nz = 1 -> t != 0
        f_rows = ctx._bits(f_c, ACC_FRAC, "f")                # defines f as a 23-bit integer
        ctx._learn(f_c, 0, 2**ACC_FRAC)
        M24 = f_c + nz * 2**ACC_FRAC                          # the 24-bit significand (the zero flag is of all 24 bits)
        _zero(ctx, "z0", zc, M24)
        ctx.inverse_nonzero("z1", M24 + zc)                   # z = [M = 0]
        mbits = _bit_exprs(ctx, f_rows[drop:]) + [nz]         # the adder-side state Mt = M >> drop, in bits
        e = t_c - nz - (p.acc_bias - 1)
        e_c = e + zc * (p.floor - e)
    else:
        s_c = ctx.pin("c.s", 0, 2)
        e_c = ctx.pin("c.e_hat", p.exp_lo, p.acc_e_max + 1)
        Mt = ctx.pin("c.Mt", 0, 2**nb)
        mbits = _bit_exprs(ctx, ctx._bits(Mt, nb, "Mt"))
    state = (s_c, e_c, mbits)

    # --- the groups --------------------------------------------------------------------------------------------------
    ovfs = []
    final = None
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        s_st, e_st, mb = state
        ctx.scope = tag
        G = ctx.pin(f"{tag}.G", p.floor, max(p.floor, p.ep_hi) + 1)
        qm = pl.qmax(size)
        if pl.wide:
            Qp = [(ctx.pin(f"{tag}.Q[{j}].lo", 0, 2**LIMB), ctx.pin(f"{tag}.Q[{j}].hi", -(2 ** (LIMB - 1)), 2 ** (LIMB - 1)))
                  for j in range(PB)]
        else:
            Qp = [ctx.pin(f"{tag}.Q[{j}]", -qm, qm + 1) for j in range(PB)]

        # the group maximum E = max(G, e_acc) and both alignment shifts from ONE one-hot over c = clamp(k, -W, PB), where
        # k = e_acc - G is an affine expression of the state and the pin (no shift hints).  k >= 0: E = G + k, the products'
        # sum is Q[min(k, PB)] (0 for k >= PB, every term shifted out), the accumulator's term is its unshifted window;
        # k < 0: E = e_acc, the sum is Q[0], the window is shifted by min(-k, W) (0 for -k >= W).  The one-hot's key
        # constraint: diff = k - sum_c sel_c c is 0 unless an end selector is on, and (1 - 2 sel_lo) diff in [0, 2^SHIFT_BITS)
        # (diff >= 0 at the top end, <= 0 at the bottom end: the end selectors stand for k >= PB and k <= -W, with the same
        # bound |k| < 2^SHIFT_BITS + end as the two lookups had): one product row and SHIFT_BITS bit rows for both shifts.
        # delta = E - G is a hint picked as k at the top end, c on the non-negative singletons, 0 below zero.
        ctx.scope = f"{tag}.max"
        k = e_st - G
        cs = list(range(-W, PB + 1))
        sels = [_hint(ctx, f"{tag}.sel[{i}]") for i in range(len(cs))]
        for i, s in enumerate(sels):
            ctx.boolean(f"sel[{i}]", s)
        ctx.eq("onehot", sum(sels[1:], sels[0]), 1)
        sel_lo, sel_hi = sels[0], sels[-1]
        ksel = sum((c * s for c, s in zip(cs[1:], sels[1:])), cs[0] * sels[0])
        ctx.learn(ksel, cs[0], cs[-1] + 1)                    # the one-hot picks one of the cs
        diff = k - ksel
        _zero(ctx, "keysel", diff, 1 - sel_lo - sel_hi)
        ctx.rng("keydiff", ctx.product(1 - 2 * sel_lo, diff), 0, 2**pl.shift_bits)
        zero = Expr(ctx)
        delta = _pick(ctx, "delta", sels, [zero if c < 0 else (k if c == PB else zero + c) for c in cs], hint=f"{tag}.delta")
        E = G + delta
        sel_nonpos = sum(sels[1:W + 1], sels[0])              # c <= 0 (a bit: the one-hot's partial sum)
        sel_nonneg = sum(sels[W + 1:], sels[W])               # c >= 0
        ctx.learn(sel_nonpos, 0, 2)
        ctx.learn(sel_nonneg, 0, 2)

        # the products' sum: Q[max(c, 0)] with Q[PB] = 0 (the top end selector); one quadratic per distinct value
        ctx.scope = f"{tag}.qsel"
        jsels = [sel_nonpos] + sels[W + 1:]                    # c <= 0 -> Q[0]; c = 1..PB-1 -> Q[c]; c = PB -> 0
        if pl.wide:
            QL = _pick(ctx, "QL", jsels, [q[0] for q in Qp] + [zero], hint=f"{tag}.QL")
            QH = _pick(ctx, "QH", jsels, [q[1] for q in Qp] + [zero], hint=f"{tag}.QH")
        else:
            Q = _pick(ctx, "Q", jsels, Qp + [zero], hint=f"{tag}.Q")

        # the accumulator's term: floor(Mt 2^ss / 2^min(-c, W)) as the window of Mt's bits selected by the one-hot
        ctx.scope = f"{tag}.acc"
        tsels = [sel_nonneg] + sels[W - 1:0:-1] + [sel_lo]     # shift 0 (c >= 0), 1..W-1 (c = -1..-(W-1)), W (c = -W: term 0)
        if pl.wide:
            qL = _pick(ctx, "qL", tsels, [_window(ctx, mb, ss - t, 0, LIMB) for t in range(W)] + [zero], hint=f"{tag}.qL")
            qH = _pick(ctx, "qH", tsels, [_window(ctx, mb, ss - t, LIMB, W, LIMB) for t in range(W)] + [zero], hint=f"{tag}.qH")
        else:
            q_acc = _pick(ctx, "q", tsels, [_window(ctx, mb, ss - t, 0, W) for t in range(W)] + [zero], hint=f"{tag}.q")

        # the exact signed sum S = (1 - 2 s_acc) q_acc + Q = (1 - 2 sgn) v
        ctx.scope = f"{tag}.sum"
        sgn = _hint(ctx, f"{tag}.sgn")
        ctx.boolean("sgn", sgn)
        sign_st = 1 - 2 * s_st
        if pl.wide:
            vL, vH = _hint(ctx, f"{tag}.vL"), _hint(ctx, f"{tag}.vH")
            rL = ctx._bits(vL, LIMB, "vL")
            rH = ctx._bits(vH, SB - LIMB, "vH")
            carry = _hint(ctx, f"{tag}.carry")
            ctx.rng("carry", carry + 4, 0, 8)
            ctx.eq("sumL", sign_st * qL + QL - (vL - 2 * sgn * vL) - 2**LIMB * carry, 0)
            ctx.eq("sumH", sign_st * qH + QH - (vH - 2 * sgn * vH) + carry, 0)
            vbits = _bit_exprs(ctx, rL + rH)
        else:
            v = _hint(ctx, f"{tag}.v")
            rv = ctx._bits(v, SB, "v")
            ctx.eq("sum", sign_st * q_acc + Q, v - 2 * sgn * v)
            vbits = _bit_exprs(ctx, rv)

        # RZ normalisation: M = window_t(v) = hid 2^(nb-1) + f, t = max(l - nb, tc(E)) proved by hid
        ctx.scope = f"{tag}.norm"
        nsels = [_hint(ctx, f"{tag}.nsel[{i}]") for i in range(pl.n_t)]
        for i, s in enumerate(nsels):
            ctx.boolean(f"nsel[{i}]", s)
        ctx.eq("nsel.onehot", sum(nsels[1:], nsels[0]), 1)
        ts = range(pl.t_lo, pl.t_hi + 1)
        hid = _hint(ctx, f"{tag}.hid")
        ctx.boolean("hid", hid)
        if g + 1 < len(pl.groups):
            # an intermediate group: the next group's alignment window reads the state's bits, so the fraction is committed
            fo = _hint(ctx, f"{tag}.fo")
            fo_rows = ctx._bits(fo, nb - 1, "fo")
            Mn = hid * 2 ** (nb - 1) + fo
            ctx._learn(Mn, 0, 2**nb)
            _pick(ctx, "Mn", nsels, [_norm_window(ctx, vbits, t, nb) for t in ts], target=Mn)
        else:
            # the last group: no fraction bits.  M is picked from the nb-bit windows of v alone, the bits of v above the
            # selected window are forced to zero (sel_t H_t = 0 with H_t their positive sum), and hid is picked from the
            # window's top bit; so M = floor(v / 2^t) < 2^nb and f = M - hid 2^(nb-1) in [0, 2^(nb-1)) as integers, from
            # v's bit rows and the one-hot alone
            fo_rows = None
            Mn = _hint(ctx, f"{tag}.M")
            _pick(ctx, "Mn", nsels, [_window(ctx, vbits, -t, 0, nb) for t in ts], target=Mn)
            for i, (t, sl) in enumerate(zip(ts, nsels)):
                high = _window(ctx, vbits, -t, nb, 10**9, nb)
                if high.lin:
                    _zero(ctx, f"Mhi[{i}]", sl, high)
            ctx.learn(Mn, 0, 2**nb)
            _pick(ctx, "hid", nsels, [vbits[t + nb - 1] if 0 <= t + nb - 1 < len(vbits) else Expr(ctx) for t in ts], target=hid)
            fo = Mn - hid * 2 ** (nb - 1)
            ctx.learn(fo, 0, 2 ** (nb - 1))
            last_norm = (nsels, vbits)
        t_sel = sum((t * s for t, s in zip(range(pl.t_lo + 1, pl.t_hi + 1), nsels[1:])), pl.t_lo * nsels[0])
        ctx._learn(t_sel, pl.t_lo, pl.t_hi + 1)
        e1 = E + t_sel + nb - p.width
        ctx.scope = f"{tag}.out"
        z, ovf = _hint(ctx, f"{tag}.z"), _hint(ctx, f"{tag}.ovf")
        ctx.boolean("z", z)
        ctx.boolean("ovf", ovf)
        _zero(ctx, "z0", z, Mn)
        ctx.inverse_nonzero("z1", Mn + z)                     # z = [M = 0]
        # The exponent the shift proves, e1 = E + t + nb - width, matters only when the sum is not zero: a zero sum (z = 1)
        # is the zero word whatever t the one-hot picked (every window of v = 0 is 0), and no t need exist with e1 in range
        # (E can sit anywhere in [floor, acc_e_max] when the terms cancel).  So the clamp / overflow proofs read the masked
        # exponent e1m = e1 when z = 0, acc_e_min when z = 1 (one product z e1, shared with e_hat below).
        ze1 = z * e1
        e1m = e1 - ze1 + p.acc_e_min * z
        ctx.scope = f"{tag}.norm"
        dt = e1m - p.acc_e_min                                # = t - tc(E), tc the clamped shift (e1 = acc_e_min), when z = 0
        ctx.rng("dt", dt, 0, 2**CMP_BITS)                     # t >= tc: e1 >= acc_e_min
        _zero(ctx, "clamp", 1 - hid, dt)                      # hid = 0 -> t = tc (the subnormal grid)
        ctx.scope = f"{tag}.out"
        cmp = 2 * (ovf * e1m) - (2 * p.acc_e_max + 1) * ovf + p.acc_e_max - e1m
        ctx.rng("ovf.proof", cmp, 0, 2**CMP_BITS)             # ovf = [e1 > acc_e_max] (ovf = 0 when z = 1)
        ovf_eff = ovf - ovf * z
        s_out = sgn - sgn * z
        e_hat = e1 + z * p.floor - ze1
        ovfs.append((ovf_eff, sgn))
        final = (s_out, e1, hid, fo, z)
        if fo_rows is not None:
            state = (s_out, e_hat, _bit_exprs(ctx, fo_rows) + [hid])

    # --- pack, component-wise -----------------------------------------------------------------------------------------
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
        # --- epilogue: f32_to_bf16 (RN-even) on the components.  The kept high part of the fraction, the half bit and the
        # sticky sum are picked from the sum's bit rows with the normalisation one-hot like M itself (fhi = bits [L, nb-1) of
        # the window, half = bit L-1, sticky = rest + lsb = bits below L-1 plus bit L: each an exact function of v's bits and
        # the selected shift, so they need no consistency constraint among themselves); the round-up bit is
        # carry = [lo + lsb >= 2^(L-1) + 1] = half AND NOT [sticky = 0], the zero flag by the usual zero-product + inverse
        # pair; fhi and carry masked by ``live`` (a zero or infinite output does not round).  No remainder, no bit rows.
        ctx.scope = "epi"
        L = ACC_FRAC - p.frac
        nsels, vbits = last_norm
        ts = range(pl.t_lo, pl.t_hi + 1)
        fhi = live * _pick(ctx, "fhi", nsels, [_window(ctx, vbits, -t, L, nb - 1, L) for t in ts], hint="epi.fhi")
        half = _pick(ctx, "half", nsels, [vbits[t + L - 1] if 0 <= t + L - 1 < len(vbits) else Expr(ctx) for t in ts],
                     hint="epi.half")
        sticky = _pick(ctx, "sticky", nsels,
                       [_window(ctx, vbits, -t, 0, L - 1) + (vbits[t + L] if 0 <= t + L < len(vbits) else Expr(ctx)) for t in ts],
                       hint="epi.sticky")                     # rest + lsb in [0, 2^(L-1)]
        hi = ys * 2 ** (p.word_bits - 1) + yt * 2**p.frac + fhi
        zl = _hint(ctx, "epi.zl")
        ctx.boolean("zl", zl)
        _zero(ctx, "zl0", zl, sticky)
        ctx.inverse_nonzero("zl1", sticky + zl)               # zl = [sticky = 0]
        carry = live * ctx.product(half, 1 - zl)
        y16 = hi + carry
    ctx.sys.outputs["y16"] = y16
    names = {n: i for i, n in ctx.sys.hints}
    ctx.sys.chain = {"c": [names["acc.s"], names["acc.t"], names["acc.f"]], "y": [ys, yt, yf], "y16": y16}
    return ctx.sys


# ---------------------------------------------------------------------------------------------------------------------
# The verifier's public work
# ---------------------------------------------------------------------------------------------------------------------

def _decode_operands(pl: Plan, w, bad: list, name: str):
    import torch

    if ((w < 0) | (w >= 2**pl.op_word_bits)).any():
        bad.append(f"x.{name} range")
    t = (w >> pl.op_frac) & (2**pl.op_exp - 1)
    f = w & (2**pl.op_frac - 1)
    if pl.e4m3:
        if ((t == 2**pl.op_exp - 1) & (f == 2**pl.op_frac - 1)).any():
            bad.append(f"{name}: E4M3 NaN operand")
    elif (t == 2**pl.op_exp - 1).any():
        bad.append(f"{name}.t off the finite range")
    nz = (t != 0).to(torch.int64)
    return w >> (pl.op_word_bits - 1), f + nz * 2**pl.op_frac, t - nz - (pl.op_bias - 1)


def group_pins_torch(p: Params, pl: Plan, a, b, bad: list | None = None) -> dict:
    """``public_pins_int`` vectorised: ``a``/``b`` (B, k) int64 -> the group pins (B,) each."""
    import torch

    bad = [] if bad is None else bad
    s_a, m_a, e_a = _decode_operands(pl, a, bad, "a")
    s_b, m_b, e_b = _decode_operands(pl, b, bad, "b")
    u = m_a * m_b
    s = (s_a + s_b) % 2
    e = torch.where(u == 0, torch.full_like(u, p.floor), e_a + e_b)
    out: dict = {}
    start = 0
    for g, size in enumerate(pl.groups):
        sl = slice(start, start + size)
        start += size
        ug, sg, eg = u[:, sl], s[:, sl], e[:, sl]
        G = torch.clamp(eg.max(dim=1).values, min=p.floor)
        out[f"g{g}.G"] = G
        gi = G[:, None] - eg                                  # >= 0
        sign = 1 - 2 * sg
        # every extra shift at once: (B, size, PB) -- d = G - e_i + j, the term shifted left by PS - d (d <= PS), right by
        # d - PS (PS < d < PB), or gone (d >= PB); then the signed sums over the group
        d = gi[:, :, None] + torch.arange(pl.PB, device=ug.device)[None, None, :]
        ug3 = ug[:, :, None]
        al = torch.where(d <= pl.PS, ug3 << torch.clamp(pl.PS - d, min=0),
                         torch.where(d < pl.PB, ug3 >> torch.clamp(d - pl.PS, min=0), torch.zeros_like(d)))
        Qs = (sign[:, :, None] * al).sum(dim=1)                # (B, PB)
        if pl.wide:
            lo = Qs & (2**LIMB - 1)
            hi = (Qs - lo) >> LIMB
        for j in range(pl.PB):
            if pl.wide:
                out[f"g{g}.Q[{j}].lo"], out[f"g{g}.Q[{j}].hi"] = lo[:, j], hi[:, j]
            else:
                out[f"g{g}.Q[{j}]"] = Qs[:, j]
    return out


def public_vectors_v2(p: Params, c, a, b, y, device="cpu"):
    """The verifier's per-unit public work: ``a``/``b`` (B, k) operand words -> the group pins; unit mode (``c`` given) also
    the accumulator pins ``c.s, c.e_hat, c.Mt`` and the output components ``y.s, y.t, y.f``.  Raises ``ValueError`` off the
    domain (a non-finite operand or accumulator, a claimed word off the format)."""
    import torch

    pl = plan(p)
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    k = sum(pl.groups)
    if a.ndim != 2 or a.shape[1] != k or b.shape != a.shape:
        raise ValueError("shape mismatch")
    bad: list[str] = []
    out = group_pins_torch(p, pl, a, b, bad)
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
