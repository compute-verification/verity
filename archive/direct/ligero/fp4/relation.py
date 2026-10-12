"""The exact NVFP4 transition unit (``BLACKWELL_SM120_NVF4.step``: one ``mma.sync kind::mxf4nvf4`` firing, 64 E2M1 codes and
4 UE4M3 scale bytes per operand, an FP32 accumulator in and out) as a unit-uniform B-Ligero constraint system.

The model (``verity.ml.circuits.silicon.models.BlockScaledAlignAdd``, lane fp4-re, RTX 5090, 0 mismatches / 24M words):

1. the four group dots ``gd_g = sum_{i in g} na_i nb_i`` are exact integers (numerators in halves: |n| <= 12, |gd| <= 2304);
2. each is scaled losslessly by its two UE4M3 block scales: ``mant_g = gd_g * ma_g * mb_g`` (|mant| < 2^19) at the unit weight
   ``2^X_g``, ``X_g = -2 + ea + eb`` in [-20, 8]; a group *participates* when one of its products and both scale mantissas
   are nonzero;
3. one five-way align-add on the grid ``2^lsb``, ``lsb = max(G, acc_cand)`` with ``G = max_{participating} X_g - 27`` (-174
   when no group participates) and ``acc_cand = E_acc - 35`` (``E_acc`` the accumulator's exponent field, subnormals -126;
   -174 for a zero accumulator) -- the -174 floor is absorbed into the two candidates' defaults --, every addend truncated
   toward zero onto the grid, the exact signed sum ``T`` normalised RZ to binary32, every zero result +0.

What the verifier computes (pins, ``witness.public_vectors_fp4``): the 128 numerators, per group ``sm = ma mb``, ``X`` and
the participation flag, and ``G``.  What the prover supplies (hints): the accumulator's word components (chain-linked to the
previous unit's output), ``lsb``, and for every one of the six shifted quantities the witness of a *truncating shifter*.

The shifter (``_shifter``) is the one gadget of this relation.  A magnitude ``m < 2^bits`` shifted by ``d`` (truncating
toward zero, so ``floor(m 2^d)``) is written ``d = 12 j + r - u`` with ``(12 j + r) u = 0``: first the right shift as a
division ``m = q 2^u + rem`` (``q < 2^(bits-u)``, ``rem < 2^u``: two ``lt_pow`` range checks on a lookup of ``u``), then the
fine left shift ``x = q 2^r`` in two lookup stages (``r = r_lo + 4 r_hi``; ``x < 2^30 < p``), decomposed into 12-bit
*digits*, then the coarse shift by ``12 j`` as a one-hot placement of the digits into positions ``c + j``.  Everything the
adder sums is therefore a signed *non-canonical digit vector* over six 12-bit positions (72 bits: the exact sum has at most
49), which is what keeps every constraint inside BabyBear.  A term shifted out entirely (``d <= -bits - 1``) takes the
``dead`` path (``u = bits`` forces ``q = 0``; ``d = -bits - k``, ``k >= 1``).  The output is normalised by the same shifter
run backwards: the claimed significand ``M`` (24 bits) placed at ``sigma = t' - 150 - lsb`` must satisfy
``|T| = M 2^sigma + R, 0 <= R < 2^sigma`` (``R`` in digits below the placement), checked digit-wise with carries; ``M`` in
[2^23, 2^24) when the exponent field ``t`` is nonzero and ``M = f < 2^23`` on the subnormal grid (``t = 0``) is exactly
the RZ normalisation, and a zero ``M`` forces ``T = 0`` and the sign bit 0 (+0).

The chain: ``acc.s / acc.t / acc.f`` of unit ``j + 1`` are linked to ``y.s / y.t / y.f`` of unit ``j``; a chain starts at
``c_0 = 0`` and ends on the three *components* of the public FP32 word (``chain.py`` ``y_end``: sign bit 31, exponent
field bits 23..30, fraction bits 0..22), each < 2^31 < p, since a 32-bit word is not a field element.

Domain: every E2M1 nibble; UE4M3 bytes with bit 7 clear and other than the NaN code 0x7F (the verifier's decode
rejects the rest); finite FP32 accumulators (the exponent field 255 is rejected in the decode and by the range check of
the linked exponent field).  An NVFP4 chain from ``c_0 = 0`` cannot leave binary32 (D20), so there is no overflow path.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity_numerical.checker.ctx import Table
from verity_numerical.checker.params import Params

from ..compile import Expr, LigeroCtx, System

__all__ = ["ACC_BITS", "ACC_WINDOW", "E_FIELD_MAX", "FRAC_BITS", "GRID_FLOOR", "GROUP", "GROUPS", "K", "MAG_BITS",
           "NUMERATOR", "NVFP4", "NVFP4Params", "POSITIONS", "RADIX", "SCALES", "STEPS", "WORDS", "compile_fp4_unit",
           "decode_scale", "numerator", "shifter_plan"]

K = 64                          # E2M1 codes per operand per step (mma.m16n8k64)
GROUP = 16                      # codes per exact group dot = codes per UE4M3 scale block (nvf4: group == block)
GROUPS = K // GROUP             # 4
SCALES = K // GROUP             # 4 scale bytes per operand per step
WORDS = K + SCALES              # 68 words per operand per step (the layout ``BlockScaledAlignAdd.step`` takes)
STEPS = 1536 // K               # 24 chained steps for GemmAccumulator<1536>
ACC_BITS = 24                   # FP32 significand bits
FRAC_BITS = 27                  # adder bits below the leading participating group's unit weight
ACC_WINDOW = 36                 # accumulator window: lsb >= E_acc - 35
GRID_FLOOR = -174
RADIX = 12                      # digit width of the align-add
POSITIONS = 6                   # digits 0..5: 72 bits >= the 49-bit exact sum, and M 2^sigma (24 + 47 bits)
MAG_BITS = 19                   # |mant_g| <= 2304 * 225 = 518,400 < 2^19
DEAD_BITS = 7                   # k in [1, 128): the dead path covers shifts down to -bits - 127
NUMERATOR = (0, 1, 2, 3, 4, 6, 8, 12)   # E2M1 magnitudes in halves (0, .5, 1, 1.5, 2, 3, 4, 6)
E_FIELD_MAX = 254               # finite FP32 exponent fields: 0..254
ACC_CAND_OFF = 127 + ACC_WINDOW - 1     # acc_cand = tp - 162 (tp = max(t, 1)); -174 for a zero accumulator
ACC_E_OFF = 127 + ACC_BITS - 1          # weight of the significand's lsb: tp - 150


@dataclass(frozen=True)
class NVFP4Params(Params):
    """Widths of the NVFP4 unit as ``Params`` duck-typing (``protocol`` reads ``word_bits``; nothing else of the base is used
    by this relation): E2M1 operands (2 significand bits, 2 exponent bits), FP32 accumulator, adder width = the window."""

    sig: int = 2
    exp: int = 2
    acc_sig: int = 24
    width: int = ACC_WINDOW
    floor: int = GRID_FLOOR
    groups: tuple[int, ...] = (GROUP,) * GROUPS
    acc_exp: int = 8


NVFP4 = NVFP4Params()


def decode_scale(code: int) -> tuple[int, int]:
    """UE4M3 byte -> (mantissa, exponent) with the value ``mantissa 2^exponent`` (``models._scale_dyadic``); raises off the
    domain (bit 7 set, NaN code 0x7F)."""
    if not 0 <= code < 128:
        raise ValueError("UE4M3 scale byte with the padding bit set (or out of range)")
    e, m = (code >> 3) & 0xF, code & 7
    if e == 15 and m == 7:
        raise ValueError("UE4M3 NaN scale")
    return (m, -9) if e == 0 else (8 + m, e - 10)


def numerator(code: int) -> int:
    """The E2M1 nibble as a signed number of halves."""
    if not 0 <= code < 16:
        raise ValueError("E2M1 code outside 4 bits")
    n = NUMERATOR[code & 7]
    return -n if code & 8 else n


#: The three shifters of a unit: ``tag -> (bits, n_j, exact, dead)``; ``witness.hints_fp4`` and the census read it.
def shifter_plan() -> dict[str, tuple[int, int, bool, bool]]:
    plan = {f"g{g}.shift": (MAG_BITS, 3, False, True) for g in range(GROUPS)}
    plan["accterm.shift"] = (ACC_BITS, 2, False, True)
    plan["out.shift"] = (ACC_BITS, 4, True, False)
    return plan


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------

def _pow_table(bits: int) -> Table:
    """u in [0, bits] -> (2^u, 2^(bits - u)): the divisor of the right shift and the complementary power ``lt_pow``
    uses for ``rem < 2^u`` (``rem < 2^bits / 2^(bits-u)``)."""
    return Table(f"POW{bits}", 0, bits + 1, ("pw", "cpw"), lambda u: (2**u, 2 ** (bits - u)))


_R_LO = Table("RLO", 0, 4, ("pw", "cpw"), lambda r: (2**r, 2 ** (4 - r)))                # fine shift, low 2 bits
_R_HI = Table("RHI", 0, 3, ("pw", "cpw"), lambda r: (2 ** (4 * r), 2 ** (8 - 4 * r)))    # fine shift, high part (0, 4, 8)


# ---------------------------------------------------------------------------------------------------------------------
# The compiled unit
# ---------------------------------------------------------------------------------------------------------------------

def _hint(ctx: LigeroCtx, name: str) -> Expr:
    """A hint row under its full path (the row name and the ``sys.hints`` name coincide: hints are flat)."""
    sc, ctx.scope = ctx.scope, ""
    try:
        return ctx.hint(name)
    finally:
        ctx.scope = sc


def _digits(ctx: LigeroCtx, x: Expr, nbits: int, name: str) -> list[Expr]:
    """x < 2^nbits as RADIX-bit digit expressions (bit rows underneath; the decomposition constraint establishes x's range)."""
    rows = ctx._bits(x, nbits, name)
    out = []
    for c in range(0, nbits, RADIX):
        lin = {rows[i]: 2 ** (i - c) for i in range(c, min(c + RADIX, nbits))}
        e = Expr(ctx, lin, 0)
        ctx._learn(e, 0, 2 ** min(RADIX, nbits - c))
        out.append(e)
    return out


def _shifter(ctx: LigeroCtx, path: str, d: Expr, *, bits: int, n_j: int, sign: Expr | None, exact: bool, dead: bool):
    """The truncating shifter of the module comment on a magnitude of ``bits`` bits, its hints under ``path``.

    Hints: ``u`` (right shift, in [0, bits]), ``q`` (``bits <= 19``) or ``qh``/``ql`` (the magnitude's quotient in two
    12-bit halves), ``rem`` (unless ``exact``), ``rlo``/``rhi`` (fine left shift ``r = rlo + 4 rhi``), the ``sel*``
    one-hot of the coarse shift ``j`` (``n_j`` values), and ``dead``/``k`` when ``dead``.

    Constrains ``d = 12 j + r - u [- dead k]`` and ``(12 j + r) u = 0`` and returns ``(mag, L, placed, sels, fine)``: the
    magnitude expression ``q 2^u + rem`` the caller equates with its source, the left shift ``L = 12 j + r``, the placed
    digit contributions ``{position: [Expr]}`` (signed by ``sign`` when given: ``(1 - 2 sign)`` is folded into the coarse
    selectors), the selector rows and ``(pw_lo, cpw_lo, pw_hi, cpw_hi)`` of the fine shift."""
    sc, ctx.scope = ctx.scope, path
    u = _hint(ctx, f"{path}.u")
    pw, cpw = ctx.lookup("u", _pow_table(bits), u)
    if bits <= MAG_BITS:
        q = _hint(ctx, f"{path}.q")
        ctx.lt_pow("q", q, pw, bits)                       # q < 2^bits / 2^u  (q 2^u < 2^bits)
        q_parts = [(q, bits)]
    else:
        qh, ql = _hint(ctx, f"{path}.qh"), _hint(ctx, f"{path}.ql")
        ctx.rng("qh", qh, 0, 2 ** (bits - RADIX))
        ctx.rng("ql", ql, 0, 2**RADIX)
        q = qh * 2**RADIX + ql
        ctx.lt_pow("q", q, pw, bits)
        q_parts = [(ql, RADIX), (qh, bits - RADIX)]
    qpw = q * pw                                          # whitelisted product: < 2^bits by lt_pow
    if exact:
        mag = qpw
    else:
        rem = _hint(ctx, f"{path}.rem")
        ctx.lt_pow("rem", rem, cpw, bits)                  # rem < 2^bits / 2^(bits - u) = 2^u
        mag = qpw + rem
    # fine left shift r = rlo + 4 rhi in [0, 12)
    rlo, rhi = _hint(ctx, f"{path}.rlo"), _hint(ctx, f"{path}.rhi")
    pw_lo, cpw_lo = ctx.lookup("rlo", _R_LO, rlo)
    pw_hi, cpw_hi = ctx.lookup("rhi", _R_HI, rhi)
    r = rlo + 4 * rhi
    sels = [_hint(ctx, f"{path}.sel{j}") for j in range(n_j)]
    for j, s in enumerate(sels):
        ctx.boolean(f"sel{j}", s)
    ctx.eq("onehot", sum(sels[1:], sels[0]), 1)
    jval = sum((j * s for j, s in enumerate(sels[1:], 1)), Expr(ctx))
    L = RADIX * jval + r
    ctx.eq("lu", L * u, 0)                                # one of the two shifts is zero
    if dead:
        dd, k = _hint(ctx, f"{path}.dead"), _hint(ctx, f"{path}.k")
        ctx.boolean("dead", dd)
        ctx.rng("k", k - 1, 0, 2**DEAD_BITS)              # k >= 1: a dead term is shifted out by at least one more bit
        ctx.eq("dead.u", dd * (L + bits - u), 0)          # dead -> u = bits (so q = 0) and L = 0
        ctx.eq("d", d, L - u - dd * k)
    else:
        ctx.eq("d", d, L - u)
    # fine shift of each part, then digits
    pre: dict[int, list[Expr]] = {}
    for part, (qp, pbits) in enumerate(q_parts):
        x1 = qp * pw_lo
        x2 = x1 * pw_hi                                   # < 2^(pbits + 11) <= 2^30
        digs = _digits(ctx, x2, pbits + 11, f"x{part}")
        for c, e in enumerate(digs):
            pre.setdefault(c + part, []).append(e)        # the high half sits one digit up
    # coarse placement
    ssels = [ctx.product(1 - 2 * sign, s) for s in sels] if sign is not None else sels
    placed: dict[int, list[Expr]] = {}
    for j, s in enumerate(ssels):
        for c, es in pre.items():
            for e in es:
                placed.setdefault(c + j, []).append(ctx.product(s, e))
    ctx.scope = sc
    return mag, L, placed, sels, (pw_lo, cpw_lo, pw_hi, cpw_hi)


def compile_fp4_unit(p: NVFP4Params = NVFP4, chain: bool = True) -> System:
    """The unit as a ``System``.  ``chain=True`` (the only mode with a public claim): the accumulator's components are hint
    rows linked by the chain test; the output components are expressions and ``sys.chain`` names them, the end in three
    components of the FP32 word.  ``chain=False``: the accumulator components are pinned (``c.s, c.t, c.f``) and the
    output components are public (``y.s, y.t, y.f``) -- the unit-mode system the tests exercise directly."""
    ctx = LigeroCtx(p)
    ctx.whitelist_product("shift", "lt_pow(q, pw, bits) gives q < 2^bits / pw, so q * pw < 2^bits", 0, 2**ACC_BITS)

    # --- the accumulator -------------------------------------------------------------------------------------------
    ctx.scope = "acc"
    if chain:
        s_c, t_c, f_c = _hint(ctx, "acc.s"), _hint(ctx, "acc.t"), _hint(ctx, "acc.f")
    else:
        s_c = ctx.pin("c.s", 0, 2)
        t_c = ctx.pin("c.t", 0, E_FIELD_MAX + 1)
        f_c = ctx.pin("c.f", 0, 2**23)
    nz, z, fh, fl = (_hint(ctx, f"acc.{n}") for n in ("nz", "z", "fh", "fl"))
    ctx.boolean("s", s_c)
    ctx.boolean("nz", nz)
    ctx.boolean("z", z)
    ctx.rng("t.nz1", t_c - nz, 0, E_FIELD_MAX)             # nz = 0 -> t = 0; nz = 1 -> t in [1, 254]: finite (defines t)
    ctx.eq("t.nz0", (1 - nz) * t_c, 0)
    ctx.rng("fh", fh, 0, 2 ** (23 - RADIX))
    ctx.rng("fl", fl, 0, 2**RADIX)
    if chain:
        ctx._learn(f_c, 0, 2**23)                          # the linked row IS the integer fh 2^12 + fl (next constraint)
    ctx.eq("f.digits", f_c, fh * 2**RADIX + fl)
    Mh, Ml = nz * 2 ** (ACC_BITS - 1 - RADIX) + fh, fl        # the 24-bit significand in two 12-bit halves
    M_c = Mh * 2**RADIX + Ml
    ctx.eq("z0", z * M_c, 0)
    ctx.inverse_nonzero("z1", M_c + z)                      # z = [M = 0]
    tp = t_c + 1 - nz                                       # max(t, 1): the exponent field, subnormals as 1
    acc_cand = tp - ACC_CAND_OFF + (GRID_FLOOR - 1 + ACC_CAND_OFF) * z   # tp - 162, or -174 for a zero accumulator (tp = 1)
    acc_e = tp - ACC_E_OFF                                  # weight of the significand's lsb

    # --- the public operands --------------------------------------------------------------------------------------
    ctx.scope = "x"
    na = [ctx.pin(f"a[{i}].n", -12, 13) for i in range(K)]
    nb = [ctx.pin(f"b[{i}].n", -12, 13) for i in range(K)]
    sm = [ctx.pin(f"g[{g}].sm", 0, 226) for g in range(GROUPS)]
    X = [ctx.pin(f"g[{g}].X", -20, 9) for g in range(GROUPS)]
    part = [ctx.pin(f"g[{g}].part", 0, 2) for g in range(GROUPS)]
    G = ctx.pin("anc.G", GRID_FLOOR, 8 - FRAC_BITS + 1)

    # --- the anchor -------------------------------------------------------------------------------------------------
    ctx.scope = "anc"
    lsb = _hint(ctx, "anc.lsb")
    ctx.rng("geG", lsb - G, 0, 512)
    ctx.rng("geA", lsb - acc_cand, 0, 512)
    ctx.eq("max", (lsb - G) * (lsb - acc_cand), 0)          # lsb is one of the two candidates and >= both

    # --- the groups -------------------------------------------------------------------------------------------------
    P_sum: dict[int, list[Expr]] = {}
    for g in range(GROUPS):
        ctx.scope = f"g{g}"
        prods = [na[i] * nb[i] for i in range(GROUP * g, GROUP * (g + 1))]
        gd = sum(prods[1:], prods[0])
        mant = gd * sm[g]                                     # |mant| <= 518,400
        sgn = _hint(ctx, f"g{g}.sgn")
        ctx.boolean("sgn", sgn)
        d = part[g] * (X[g] - lsb)                            # 0 for a non-participating group (its mant is 0)
        mag, _L, placed, _sels, _fine = _shifter(ctx, f"g{g}.shift", d, bits=MAG_BITS, n_j=3, sign=sgn, exact=False, dead=True)
        ctx.eq("mag", mant, (1 - 2 * sgn) * mag)
        for pos, es in placed.items():
            P_sum.setdefault(pos, []).extend(es)

    # --- the accumulator's term ---------------------------------------------------------------------------------------
    ctx.scope = "accterm"
    d_acc = (1 - z) * (acc_e - lsb)                           # 0 for a zero accumulator
    mag_a, _L, placed, _sels, _fine = _shifter(ctx, "accterm.shift", d_acc, bits=ACC_BITS, n_j=2, sign=s_c, exact=False, dead=True)
    ctx.eq("mag", M_c, mag_a)
    for pos, es in placed.items():
        P_sum.setdefault(pos, []).extend(es)

    # --- the output: RZ normalisation as the shifter run backwards ---------------------------------------------------
    ctx.scope = "out"
    sgn_t, t_o, hid, fh_o, fl_o, nzM = (_hint(ctx, f"out.{n}") for n in ("sgn", "t", "hid", "fh", "fl", "nzM"))
    ctx.boolean("sgn", sgn_t)
    ctx.boolean("hid", hid)
    ctx.boolean("nzM", nzM)
    ctx.rng("t.hid1", t_o - hid, 0, E_FIELD_MAX)             # hid = 0 -> t = 0 (subnormal grid); hid = 1 -> t in [1, 254]
    ctx.eq("t.hid0", (1 - hid) * t_o, 0)
    ctx.rng("fh", fh_o, 0, 2 ** (23 - RADIX))
    ctx.rng("fl", fl_o, 0, 2**RADIX)
    Mh_o, Ml_o = hid * 2 ** (ACC_BITS - 1 - RADIX) + fh_o, fl_o
    M_o = Mh_o * 2**RADIX + Ml_o
    ctx.eq("nzM0", (1 - nzM) * M_o, 0)
    ctx.inverse_nonzero("nzM1", M_o + (1 - nzM))            # nzM = [M != 0]
    tp_o = t_o + 1 - hid
    sigma = nzM * (tp_o - ACC_E_OFF - lsb)                  # the placement of M on the grid; 0 when M = 0
    mag_o, L_o, placed, sels_o, (pw_lo, cpw_lo, pw_hi, cpw_hi) = _shifter(
        ctx, "out.shift", sigma, bits=ACC_BITS, n_j=4, sign=None, exact=True, dead=False)
    ctx.eq("mag", M_o, mag_o)                                # M = q 2^u exactly (a negative sigma is an exact left shift)
    # R < 2^L in digits: r_p = 0 above position j, r_j < 2^r, the rest 12-bit
    n_r = len(sels_o)
    rdig = [_hint(ctx, f"out.r{pn}") for pn in range(n_r)]
    for pn, rd in enumerate(rdig):
        ctx.rng(f"r{pn}", rd, 0, 2**RADIX)
    above = Expr(ctx)
    for pn in range(1, n_r):
        mask = sum(sels_o[:pn], Expr(ctx))                  # [pn > j]
        above = above + rdig[pn] * mask
    ctx.eq("r.above", above, 0)
    r_sel = sum((sels_o[pn] * rdig[pn] for pn in range(1, n_r)), sels_o[0] * rdig[0])
    t1 = r_sel * cpw_lo
    t2 = t1 * cpw_hi                                        # r_sel 2^(12 - r) < 2^12  <=>  r_sel < 2^r
    ctx.rng("r.top", t2, 0, 2**RADIX)
    # the carry chain: (1 - 2 sgn) T = M 2^sigma + R over the six positions
    carries = [Expr(ctx)]
    for pn in range(1, POSITIONS):
        c = _hint(ctx, f"out.c{pn}")
        ctx.rng(f"c{pn}", c + 16, 0, 32)
        carries.append(c)
    carries.append(Expr(ctx))
    for pn in range(POSITIONS):
        P_p = sum(P_sum.get(pn, []), Expr(ctx))
        sP = (1 - 2 * sgn_t) * P_p if P_p.lin else Expr(ctx)
        DM = sum(placed.get(pn, []), Expr(ctx))
        rp = rdig[pn] if pn < n_r else Expr(ctx)
        ctx.eq(f"carry{pn}", sP - DM - rp + carries[pn] - 2**RADIX * carries[pn + 1], 0)
    s_out = sgn_t * nzM                                     # +0 for a zero result
    ys, yt, yf = s_out, t_o, fh_o * 2**RADIX + fl_o
    ctx.sys.outputs = {"y.s": ys, "y.t": yt, "y.f": yf}
    if not chain:
        ctx.eq_pub("y.s", ys, "y.s")
        ctx.eq_pub("y.t", yt, "y.t")
        ctx.eq_pub("y.f", yf, "y.f")
        return ctx.sys
    names = {n: i for i, n in ctx.sys.hints}
    ctx.sys.chain = {"c": [names["acc.s"], names["acc.t"], names["acc.f"]], "y": [ys, yt, yf], "y16": None,
                     "y_end": [(ys, 31, 1), (yt, 23, 8), (yf, 0, 23)]}
    return ctx.sys
