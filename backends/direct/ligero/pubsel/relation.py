"""The BF16 / FP8 transition unit compiled with *public selection* (lane relmin-lookup): the ``-v2`` relations.

The relation is the one ``unit.py`` (BF16) and ``fp8/relation.py`` (FP8) prove -- the pinned instruction semantics of
``verity.ml.tc.models.GroupSum`` seen through the checker's gadgets (``verity_numerical.checker.gadgets``: group maximum,
truncating alignment, exact signed sum, RZ normalisation with the subnormal clamp, overflow / zero flags, component-wise pack).
What changes is *who computes what*.  In v1 the prover commits the operand decodes, every product, every alignment
quotient / remainder and the normalisation's quotient / remainder, and proves each with bit decompositions (65-73 % of the
rows are bits).  Here the verifier computes everything that depends on the public operands alone, and the prover commits
only what depends on the private accumulator:

* the verifier pins, per group, ``G = max(floor, e_i)`` (the group maximum of the *products*) and the signed sum of the
  truncated aligned products for every possible extra shift ``j`` (``Q[j] = sum_i (1 - 2 s_i) align(u_i, j + G - e_i)``,
  ``j`` in ``[0, prod_bits)``; every term is shifted out for ``j >= prod_bits``).  With ``E = max(G, e_acc)`` the group's
  product sum is ``Q[E - G]`` -- the prover *selects* it with a one-hot over ``delta = E - G`` (a lookup on ``delta``);
* the group maximum needs two hints, ``delta = E - G >= 0`` and ``dacc = E - e_acc >= 0`` with ``delta dacc = 0``;
* the accumulator's aligned term ``floor(M 2^state_shift / 2^min(dacc, W))`` is a *window* of the accumulator's own bit rows
  (the range check of the linked fraction supplies them) selected by a one-hot over ``min(dacc, W)``: no quotient, no
  remainder, no power table;
* the exact sum ``S = (1 - 2 s_acc) q_acc + Q[delta] = (1 - 2 sgn) v`` with ``v`` committed in bits (two 16-bit limbs and a
  carry when ``2^s_bits + s_max`` would reach ``p``: bf16-hopper);
* the RZ normalisation is a window of ``v``'s bits: ``M = floor(v / 2^t)`` (``v 2^-t`` for ``t < 0``) with a one-hot over the
  shift ``t``; ``M = hid 2^(nb-1) + f`` with ``f`` in bits.  ``hid = 1`` (``M`` in ``[2^(nb-1), 2^nb)``) *is* the leading-bit
  proof, and the subnormal clamp is ``t = tc(E)`` when ``hid = 0``, ``t >= tc(E)`` when ``hid = 1``; the output exponent is
  ``E + t + nb - width`` in both cases.  The window carries the bits above the ``nb`` selected ones at weight ``2^nb``, so a
  selected window with a stray high bit cannot equal a value below ``2^nb``: truncation is exact and the sum stays in the
  field;
* every one-hot selection is *picked*, not summed (``_pick``): the selected value is a hint row ``X`` (or the value the
  relation already commits, ``M``) and each candidate costs one row-free quadratic ``sel_i (X - value_i) = 0`` instead of a
  product row; the two shift hints ``delta`` / ``dacc`` are ranged by their lookups' one-hot + key-difference constraints
  alone (``LigeroCtx.lookup(range_key=False)``) instead of a separate bit decomposition.

One code path for the four relations, parametrised by ``Params`` (BF16: ``Params.from_model``) or ``FP8Params`` (the
adder-side widths with the lossy rescale: the adder's state is the top 14 bits of the FP32 significand).  ``chain=True`` is
the mode the prover runs (accumulator components as hint rows linked by the chain test; BF16: the RN-even epilogue on the
unit's own output; FP8: the 22-bit packed word at the chain end), ``chain=False`` pins the accumulator and the output
components (tests).  The witness rows are only the kinds ``witness.py`` / ``witness_device.py`` already derive (pin, hint,
prod, bit, sel, inv); the Ligero protocol is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity_numerical.checker.ctx import Table
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

    @property
    def n_t(self) -> int:
        return self.t_hi - self.t_lo + 1

    def qmax(self, size: int) -> int:
        """|Q[j]| <= size (2^PB - 2^PS)."""
        return size * (2**self.PB - 2**self.PS)


def plan(p: Params) -> Plan:
    fp8 = hasattr(p, "op_exp")
    nb = p.norm_bits
    if p.shift_rows > 2**SHIFT_BITS:
        raise ValueError(f"shift range {p.shift_rows} exceeds 2^{SHIFT_BITS}")
    if fp8:
        op = (p.op_exp, p.op_frac, p.op_bias, 1 + p.op_exp + p.op_frac, True)
    else:
        op = (p.exp, p.frac, p.bias, p.word_bits, p.op_nan_rule == "e4m3")
    return Plan(fp8=fp8, nb=nb, drop=ACC_FRAC + 1 - nb, W=p.acc_bits, ss=p.state_shift, PB=p.prod_bits, PS=p.prod_shift,
                SB=p.s_bits, wide=needs_wide(p), t_lo=-nb, t_hi=-p.k_lo, op_exp=op[0], op_frac=op[1], op_bias=op[2],
                op_word_bits=op[3], e4m3=op[4], groups=tuple(p.groups))


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


def _clamp_table(name: str, top: int) -> Table:
    """key in [0, 2^SHIFT_BITS) -> min(key, top): ``top`` singleton rows and one interval row (the lookup's one-hot has
    ``top + 1`` selectors; the interval selector stands for "shifted out")."""
    return Table(f"{name}{top}", 0, 2**SHIFT_BITS, ("v",), lambda k: (min(k, top),))


def _select(ctx: LigeroCtx, name: str, sels: list[Expr], values: list[Expr]) -> Expr:
    """``sum_i sel_i value_i`` for a one-hot ``sels`` (the lookup's, or hints constrained boolean with sum 1): one product
    row per term; the sum's integer interval is the hull of the values' (a theorem of the one-hot: exactly one selector is
    1 as an integer, every value is an integer of its interval, and every product is inside the field), recorded with
    ``learn`` so that the audit of the constraints using it sees the hull instead of the sum of the intervals."""
    acc = Expr(ctx)
    lo = hi = None
    for i, (s, v) in enumerate(zip(sels, values)):
        if v.is_const():
            if v.c == 0:
                continue
            term = s * v.c
            iv = (min(0, v.c), max(0, v.c))
        else:
            term = ctx.product(s, v)
            iv = ctx.interval(v)
            if iv is None:
                raise ValueError(f"{ctx.scope}.{name}[{i}]: value interval unknown")
            iv = (min(0, iv[0]), max(0, iv[1]))
        acc = acc + term
        lo = iv[0] if lo is None else min(lo, iv[0])
        hi = iv[1] if hi is None else max(hi, iv[1])
    if lo is None:
        return acc
    ctx.learn(acc, lo, hi + 1)
    return acc


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

        # the group maximum E = max(G, e_acc): delta = E - G >= 0, dacc = E - e_acc >= 0, one of them zero.  The two
        # shifts are the keys of the lookups below, whose one-hot + key-difference constraints define them as integers in
        # [0, 2^SHIFT_BITS + top) (``range_key=False``: no separate range check).
        ctx.scope = f"{tag}.max"
        delta, dacc = _hint(ctx, f"{tag}.delta"), _hint(ctx, f"{tag}.dacc")
        ctx.scope = f"{tag}.qsel"
        (jsel,) = ctx.lookup("qsel", _clamp_table("QSEL", PB), delta, range_key=False)
        ctx.scope = f"{tag}.acc"
        (tsel,) = ctx.lookup("tsel", _clamp_table("TSEL", W), dacc, range_key=False)
        ctx.scope = f"{tag}.max"
        _zero(ctx, "max", delta, dacc)
        ctx.eq("E", G + delta, e_st + dacc)
        E = G + delta

        # the products' sum: Q[min(delta, PB)], Q[PB] = 0 (the interval selector), picked by the lookup's one-hot
        ctx.scope = f"{tag}.qsel"
        jsels = _bit_exprs(ctx, jsel.sel_rows)                # PB + 1 selectors, the last for delta >= PB
        zero = Expr(ctx)
        if pl.wide:
            QL = _pick(ctx, "QL", jsels, [q[0] for q in Qp] + [zero], hint=f"{tag}.QL")
            QH = _pick(ctx, "QH", jsels, [q[1] for q in Qp] + [zero], hint=f"{tag}.QH")
        else:
            Q = _pick(ctx, "Q", jsels, Qp + [zero], hint=f"{tag}.Q")

        # the accumulator's term: floor(Mt 2^ss / 2^min(dacc, W)) as the window of Mt's bits selected by min(dacc, W)
        ctx.scope = f"{tag}.acc"
        tsels = _bit_exprs(ctx, tsel.sel_rows)                # W + 1 selectors, the last for dacc >= W (term 0)
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
        hid, fo = _hint(ctx, f"{tag}.hid"), _hint(ctx, f"{tag}.fo")
        ctx.boolean("hid", hid)
        fo_rows = ctx._bits(fo, nb - 1, "fo")
        Mn = hid * 2 ** (nb - 1) + fo
        ctx._learn(Mn, 0, 2**nb)
        _pick(ctx, "Mn", nsels, [_norm_window(ctx, vbits, t, nb) for t in range(pl.t_lo, pl.t_hi + 1)], target=Mn)
        t_sel = sum((t * s for t, s in zip(range(pl.t_lo + 1, pl.t_hi + 1), nsels[1:])), pl.t_lo * nsels[0])
        ctx._learn(t_sel, pl.t_lo, pl.t_hi + 1)
        tc = p.acc_e_min - E + p.width - nb                    # the clamped shift: e1 = acc_e_min
        dt = t_sel - tc
        ctx.rng("dt", dt, 0, 2**CMP_BITS)                     # t >= tc: e1 >= acc_e_min
        _zero(ctx, "clamp", 1 - hid, dt)                      # hid = 0 -> t = tc (the subnormal grid)
        e1 = E + t_sel + nb - p.width
        ctx.scope = f"{tag}.out"
        z, ovf = _hint(ctx, f"{tag}.z"), _hint(ctx, f"{tag}.ovf")
        ctx.boolean("z", z)
        ctx.boolean("ovf", ovf)
        _zero(ctx, "z0", z, Mn)
        ctx.inverse_nonzero("z1", Mn + z)                     # z = [M = 0]
        cmp = 2 * (ovf * e1) - (2 * p.acc_e_max + 1) * ovf + p.acc_e_max - e1
        ctx.rng("ovf.proof", cmp, 0, 2**CMP_BITS)             # ovf = [e1 > acc_e_max]
        ovf_eff = ovf - ovf * z
        s_out = sgn - sgn * z
        e_hat = e1 + z * (p.floor - e1)
        ovfs.append((ovf_eff, sgn))
        final = (s_out, e1, hid, fo, z)
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
        # --- epilogue: f32_to_bf16 (RN-even) on the components; the low / high split of the fraction and the lsb of the
        # kept word are windows of the output fraction's own bit rows (fo's), masked by ``live`` like the fraction itself
        ctx.scope = "epi"
        L = ACC_FRAC - p.frac
        fbits = _bit_exprs(ctx, fo_rows)
        lo = live * _window(ctx, fbits, 0, 0, L)
        fhi = live * _window(ctx, fbits, 0, L, ACC_FRAC, L)
        lsb = live * fbits[L]
        hi = ys * 2 ** (p.word_bits - 1) + yt * 2**p.frac + fhi
        carry, rem = _hint(ctx, "epi.carry"), _hint(ctx, "epi.rem")
        ctx.boolean("carry", carry)
        ctx.rng("rem", rem, 0, 2**L)
        ctx.eq("round", carry * 2**L + rem, lo + lsb + (2 ** (L - 1) - 1))
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
        for j in range(pl.PB):
            d = gi + j
            sh_l = torch.clamp(pl.PS - d, min=0)
            sh_r = torch.clamp(d - pl.PS, min=0)
            al = torch.where(d <= pl.PS, ug << sh_l, torch.where(d < pl.PB, ug >> sh_r, torch.zeros_like(ug)))
            Q = (sign * al).sum(dim=1)
            if pl.wide:
                lo = Q & (2**LIMB - 1)
                out[f"g{g}.Q[{j}].lo"], out[f"g{g}.Q[{j}].hi"] = lo, (Q - lo) >> LIMB
            else:
                out[f"g{g}.Q[{j}]"] = Q
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
