"""Exact GF(2) circuit of one NVFP4 transition unit (verity.ml.tc BLACKWELL_SM120_NVF4, ``BlockScaledAlignAdd.step_scaled``:
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
"""
from __future__ import annotations

from .gf2 import (ONE, ZERO, Circuit, NOT, OR, add, and_reduce, compress_columns, const_bits, lzc, multiply, mux,
                  or_reduce, shl_window, shr, sub)

K, GROUP, GROUPS = 64, 16, 4
EW = 10                      # U-unit exponent width (values up to 348)


def _max(C, a, b):
    _, a_lt_b = sub(C, a, b, EW)
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

    terms, cands = [], []
    for g in range(GROUPS):
        C.phase = "scale decode"
        ma, ea, nza = decode_scale(C, sx[8 * g:8 * g + 8])
        mb, eb, nzb = decode_scale(C, sw[8 * g:8 * g + 8])
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
        C.phase = "scale multiply"
        sm = multiply(C, ma, mb)
        mant = multiply(C, gabs, sm)[:19]
        part = C.AND(any_p, C.AND(nza, nzb))
        C.phase = "anchors"
        # cand_u = X - 27 + 256 = ea_eff + eb_eff + 207
        cu = add(C, add(C, ea, eb, width=5, carry_out=False), const_bits(207, EW), width=EW)
        cands.append([C.AND(part, bb) for bb in cu])
        terms.append((gsign, [ZERO] * 27 + mant, cu))      # 46 bits on the X - 27 grid

    C.phase = "accumulator decode"
    cm, cf, cs = c[:23], c[23:31], c[31]
    C.assert_zero("accumulator not finite", and_reduce(C, cf))
    hc = or_reduce(C, cf)
    nzc = OR(C, hc, or_reduce(C, cm))
    tp = [cf[0] ^ C.AND(NOT(cf[0]), NOT(hc))] + cf[1:]      # max(e_field, 1)
    ca = add(C, tp, const_bits(94, EW), width=EW)          # tp - 162 + 256
    cands.append([C.AND(nzc, bb) for bb in ca])
    terms.append((cs, [ZERO] * 12 + cm + [hc], ca))         # 36 bits on the tp - 162 grid

    C.phase = "lsb (max of candidates)"
    lsb = cands[0]
    for cnd in cands[1:]:
        lsb = _max(C, lsb, cnd)

    C.phase = "align (5 shifters)"
    aligned = []
    for s, mag, cu in terms:
        d, _ = sub(C, lsb, cu, EW)
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
    A = lz + [ZERO] * (EW - len(lz))
    B, _ = sub(C, lsb, const_bits(67, EW), EW)
    _, a_lt_b = sub(C, A, B, EW)
    u = mux(C, a_lt_b, A, B)[:6]
    q = shl_window(C, padded, u, 40, 64)
    f1, _ = sub(C, lsb, u, EW)
    f2, _ = sub(C, f1, const_bits(67, EW), EW)
    field = add(C, f2, [q[23]], width=EW)
    field_n = [C.AND(nz, bb) for bb in field]
    C.assert_zero("sum leaves binary32", OR(C, field_n[8], OR(C, field_n[9], and_reduce(C, field_n[:8]))))
    word = [C.AND(nz, bb) for bb in q[:23]] + field_n[:8] + [C.AND(nz, tsign)]
    C.output("c_out", word)
    return C
