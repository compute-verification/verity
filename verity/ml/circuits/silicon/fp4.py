"""Exact GF(2) circuit of one NVFP4 transition unit (verity.ml.circuits.silicon BLACKWELL_SM120_NVF4, ``BlockScaledAlignAdd.step_scaled``:
one ``mma.sync kind::mxf4nvf4`` firing, 64 E2M1 codes and 4 UE4M3 scale bytes per operand, FP32 accumulator in and out)
in Flock's model.

The model, literally (``models.BlockScaledAlignAdd._transducer`` with ``frac_bits`` 27, ``acc_window`` 36, ``grid_floor``
-174):

1. each group of 16 has the exact dot ``gd = sum na nb`` of signed E2M1 numerators (halves), scaled losslessly by the two
   UE4M3 mantissas: ``mant = gd ma mb`` at unit weight ``2^X``, ``X = -2 + ea + eb``; the group participates when one
   product and both scale mantissas are nonzero (a cancelling dot still participates);
2. ``lsb`` is the max of the participating groups' ``X - 27`` and, for a nonzero accumulator, ``E_acc - 35`` (``E_acc``
   the exponent field, subnormals -126). Every NVFP4 candidate is at least -161, so the -174 floor never binds, and with
   no candidate every addend is zero, so the result is +0 either way;
3. every addend is truncated toward zero onto ``2^lsb``, summed exactly and normalised RZ to binary32; zero is +0.

Exponents are carried in "U-units" (value + 256). Each addend is written pre-shifted onto its own candidate's grid
(``mant << 27`` on ``X - 27``, the 24-bit significand ``<< 12`` on ``tp - 162``), so its alignment is a right shift by
``lsb - candidate >= 0``. The normalisation takes bits [40, 64) of ``|T| << u``, ``u = min(lz, lsb_u - 67)``: ``lz`` for a
normal result, the FP32 subnormal grid otherwise, and the field is ``lsb_u - u - 67 + q[23]``.

:func:`unit` is the finite-domain unit (``lowering``'s ``fp4-nvf4`` pipe). :func:`step_total` is the IR primitives' piece, total
on every encoding: ``BlackwellNvf4OmmaDot64`` and, with UE8M0 scales (``BLACKWELL_SM120_MXF4``: one scale per 32 codes, the
floor binding, sums leaving binary32 or falling below its subnormal grid), the MXFP4 step.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import (ONE, ZERO, Circuit, NOT, OR, add, and_reduce, compress_columns, const_bits, lzc, multiply, mux,
                    or_reduce, shl_window, shr, sub)

K, GROUP, GROUPS = 64, 16, 4
EW = 10                      # U-unit exponent width (values up to 348)


def _max(C, a, b, ew: int = EW):
    _, a_lt_b = sub(C, a, b, ew)
    return mux(C, a_lt_b, b, a)


def decode_code(C, code):
    """E2M1 ``m e0 e1 s``: the magnitude in halves (4 bits, NUMERATOR[code & 7]), the sign, and ``mag != 0``."""
    m, e0, e1, s = code
    ne0, ne1 = NOT(e0), NOT(e1)
    e0_ne1 = C.AND(e0, ne1)
    e1_ne0 = C.AND(e1, ne0)
    e01 = C.AND(e0, e1)
    mag = [C.AND(m, ne1), e0_ne1 ^ C.AND(m, e1_ne0), e1_ne0 ^ C.AND(m, e01), e01]
    nz = OR(C, m, OR(C, e0, e1))
    return mag, s, nz


def decode_scale_total(C, byte):
    """`decode_scale` of a stand-in byte (bit 7 clear, never the NaN): no assertions."""
    m, e = byte[:3], byte[3:7]
    h = or_reduce(C, e)
    e_eff = [e[0] ^ C.AND(NOT(e[0]), NOT(h))] + e[1:]
    return m + [h], e_eff, OR(C, h, or_reduce(C, m))


def decode_scale(C, byte):
    """UE4M3 scale: the 4-bit mantissa (``m + 8 [e != 0]``), ``e or 1`` (4 bits), ``mant != 0``; bit 7 and 0x7F rejected."""
    m, e = byte[:3], byte[3:7]
    C.assert_zero("ue4m3 scale padding bit", byte[7])
    C.assert_zero("ue4m3 scale NaN", and_reduce(C, byte[:7]))
    h = or_reduce(C, e)
    e_eff = [e[0] ^ C.AND(NOT(e[0]), NOT(h))] + e[1:]
    return m + [h], e_eff, OR(C, h, or_reduce(C, m))


def unit():
    C = Circuit()
    x = C.input("x", 4 * K)
    w = C.input("w", 4 * K)
    c = C.input("c", 32)
    sx = C.input("sx", 8 * GROUPS)
    sw = C.input("sw", 8 * GROUPS)
    C.output("c_out", step(C, x, w, c, sx, sw))
    return C


def _scale_stand_in(C, byte, ue8m0: bool):
    """(the byte with its NaN replaced by zero and, for UE4M3, bit 7 cleared (the step ignores it), whether it is a NaN): UE4M3's
    NaN is low seven bits 0x7F, UE8M0's is 0xFF."""
    body = byte if ue8m0 else byte[:7]
    nan = and_reduce(C, body)
    ok = NOT(nan)
    return [C.AND(ok, b) for b in body] + ([] if ue8m0 else [ZERO]), nan


def step_total(C, acc, a, b, sa, sb, ue8m0: bool = False):
    """``BlackwellNvf4OmmaDot64`` (UE4M3 scales over blocks of 16) or its UE8M0 form over blocks of 32 (MXFP4), total on every
    encoding: a NaN scale byte gives 0x7FFFFFFF whatever the accumulator; else a NaN accumulator gives 0x7FFFFFFF and an infinite
    one itself; else the finite step on stand-ins, where a UE8M0 sum leaving binary32 is the infinity of its sign. Words are
    lists of bits, LSB first: 64 E2M1 codes per operand, then its scale bytes."""
    from .fp import F32_INF, TC_NAN, _classes, _finite_or_zero, select

    s_nan, sx, sw = ZERO, [], []
    for byte, out in [(y, sx) for y in sa] + [(y, sw) for y in sb]:
        s, nan = _scale_stand_in(C, list(byte), ue8m0)
        out += s
        s_nan = OR(C, s_nan, nan)
    c_nan, c_inf, _ = _classes(C, acc, 23, 8)
    x = [bit for code in a for bit in code]
    w = [bit for code in b for bit in code]
    r = step(C, x, w, _finite_or_zero(C, acc, 23, 8), sx, sw, ue8m0=ue8m0, total=True)
    r = select(C, c_inf, list(acc), r)
    return select(C, OR(C, s_nan, c_nan), const_bits(TC_NAN, 32), r)


def step(C, x, w, c, sx, sw, ue8m0: bool = False, total: bool = False):
    """One firing's FP32 word on flat bits: ``x``, ``w`` 4K bits of E2M1 codes, ``c`` the accumulator, ``sx``, ``sw`` the scale bytes
    (4 UE4M3, or 2 UE8M0 with ``ue8m0``). ``total=False`` asserts what the finite unit assumes (UE4M3 scales, a finite accumulator,
    a sum inside binary32); ``total=True`` takes finite stand-ins (`step_total`) and asserts nothing: a UE4M3 sum cannot leave
    binary32 from a finite accumulator, a UE8M0 one that does is the infinity of its sign, and a nonzero UE8M0 sum below the
    subnormal grid is +0.

    UE8M0 (``BLACKWELL_SM120_MXF4``): a scale code s is ``8 * 2^(s - 130)``, one per 32 codes, so a group's unit weight is
    ``X = sa + sb - 262`` with the lossless factor 64, its term ``gd << 33`` on the ``X - 27`` grid, and the -174 floor can bind:
    exponents are carried as value + 512 in 11 bits (value + 256 in 10 for UE4M3)."""
    ew, off = (11, 512) if ue8m0 else (EW, 256)
    terms, cands = [], []
    for g in range(GROUPS):
        C.phase = "scale decode"
        if ue8m0:
            ea, eb = sx[8 * (g // 2):8 * (g // 2) + 8], sw[8 * (g // 2):8 * (g // 2) + 8]
        else:
            ma, ea, nza = (decode_scale_total if total else decode_scale)(C, sx[8 * g:8 * g + 8])
            mb, eb, nzb = (decode_scale_total if total else decode_scale)(C, sw[8 * g:8 * g + 8])
        C.phase = "E2M1 products"
        prods, any_p = [], ZERO
        for i in range(GROUP * g, GROUP * (g + 1)):
            a, sa, na = decode_code(C, x[4 * i:4 * i + 4])
            b, sb, nb = decode_code(C, w[4 * i:4 * i + 4])
            prods.append((sa ^ sb, multiply(C, a, b)))
            any_p = OR(C, any_p, C.AND(na, nb))
        C.phase = "group dot (CSA) + abs"
        n, TW = 8, 13                                      # |product| <= 144; |gd| <= 2304 < 2^12
        cols = [[] for _ in range(TW + 2)]
        for s, p in prods:
            for j in range(n):
                cols[j].append(p[j] ^ s)
            cols[0].append(s)
            cols[n].append(NOT(s))
        neg = (-(GROUP << n)) % (1 << TW)
        for j in range(TW):
            if neg >> j & 1:
                cols[j].append(ONE)
        gd = compress_columns(C, cols)[:TW]
        gsign = gd[TW - 1]
        gabs = add(C, [bb ^ gsign for bb in gd[:12]], [gsign], width=12)
        if ue8m0:
            part = any_p
            C.phase = "anchors"
            # cand_u = X - 27 + 512 = sa + sb + 223
            cu = add(C, add(C, ea, eb, width=9), const_bits(223, ew), width=ew)
            cands.append([C.AND(part, bb) for bb in cu])
            terms.append((gsign, [ZERO] * 33 + gabs + [ZERO], cu))   # 64 gd << 27 on the X - 27 grid, 46 bits
            continue
        C.phase = "scale multiply"
        sm = multiply(C, ma, mb)
        mant = multiply(C, gabs, sm)[:19]
        part = C.AND(any_p, C.AND(nza, nzb))
        C.phase = "anchors"
        # cand_u = X - 27 + 256 = ea_eff + eb_eff + 207
        cu = add(C, add(C, ea, eb, width=5, carry_out=False), const_bits(207, ew), width=ew)
        cands.append([C.AND(part, bb) for bb in cu])
        terms.append((gsign, [ZERO] * 27 + mant, cu))      # 46 bits on the X - 27 grid

    C.phase = "accumulator decode"
    cm, cf, cs = c[:23], c[23:31], c[31]
    if not total:
        C.assert_zero("accumulator not finite", and_reduce(C, cf))
    hc = or_reduce(C, cf)
    nzc = OR(C, hc, or_reduce(C, cm))
    tp = [cf[0] ^ C.AND(NOT(cf[0]), NOT(hc))] + cf[1:]      # max(e_field, 1)
    ca = add(C, tp, const_bits(off - 162, ew), width=ew)   # tp - 162 + off
    cands.append([C.AND(nzc, bb) for bb in ca])
    terms.append((cs, [ZERO] * 12 + cm + [hc], ca))         # 36 bits on the tp - 162 grid

    C.phase = "lsb (max of candidates)"
    lsb = cands[0]
    for cnd in cands[1:]:
        lsb = _max(C, lsb, cnd, ew)
    if ue8m0:
        lsb = _max(C, lsb, const_bits(off - 174, ew), ew)  # the adder's floor

    C.phase = "align (5 shifters)"
    aligned = []
    for s, mag, cu in terms:
        d, _ = sub(C, lsb, cu, ew)
        big = or_reduce(C, d[6:])
        aligned.append((s, shr(C, mag, d[:6], big)))

    C.phase = "signed sum (CSA) + abs"
    n = 46
    TB = 49                                                # |T| < 4 * 2^46 + 2^36
    TW = TB + 1
    cols = [[] for _ in range(TW + 2)]
    for s, a in aligned:
        a = a + [ZERO] * (n - len(a))
        for j in range(n):
            cols[j].append(a[j] ^ s)
        cols[0].append(s)
        cols[n].append(NOT(s))
    neg = (-(len(aligned) << n)) % (1 << TW)
    for j in range(TW):
        if neg >> j & 1:
            cols[j].append(ONE)
    t = compress_columns(C, cols)[:TW]
    tsign = t[TW - 1]
    absv = add(C, [bb ^ tsign for bb in t[:TB]], [tsign], width=TB)

    C.phase = "normalise RZ"
    padded = absv + [ZERO] * (64 - TB)
    allz, lz = lzc(C, padded)
    nz = NOT(allz)
    A = lz + [ZERO] * (ew - len(lz))
    B, _ = sub(C, lsb, const_bits(off - 189, ew), ew)     # u on the FP32 subnormal grid
    _, a_lt_b = sub(C, A, B, ew)
    u = mux(C, a_lt_b, A, B)[:6]
    q = shl_window(C, padded, u, 40, 64)
    f1, _ = sub(C, lsb, u, ew)
    f2, _ = sub(C, f1, const_bits(off - 189, ew), ew)
    field = add(C, f2, [q[23]], width=ew)
    field_n = [C.AND(nz, bb) for bb in field]
    if not ue8m0:
        if not total:
            C.assert_zero("sum leaves binary32", OR(C, field_n[8], OR(C, field_n[9], and_reduce(C, field_n[:8]))))
        return [C.AND(nz, bb) for bb in q[:23]] + field_n[:8] + [C.AND(nz, tsign)]
    ovf = OR(C, or_reduce(C, field_n[8:]), and_reduce(C, field_n[:8]))
    sign = C.AND(tsign, or_reduce(C, q))                   # a sum truncated to nothing below the subnormal grid is +0
    word = [C.AND(nz, bb) for bb in q[:23]] + field_n[:8] + [sign]
    return mux(C, ovf, const_bits(0x7F800000, 31) + [tsign], word)
