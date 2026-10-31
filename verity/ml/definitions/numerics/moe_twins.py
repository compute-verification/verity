"""Scalar twins of the MoE router's Definitions (`registry.moe`): the same instruction sequences evaluated directly on one f32
encoding, the probes' and tests' reference for `NvExpf_v1` and `NvSigmoid_v1`."""
from __future__ import annotations

from verity.ml.circuits.ops.scalar import F32BitsShl23, F32Neg, F32Sat
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.moe import F32FmaRm
from verity.ml.definitions.registry.prims import bits_of, f32_of


def nv_expf_bits(x: int) -> int:
    """The same sequence evaluated directly on one f32 encoding (for probes / difftests; `NvExpf` is the Program-side
    statement of it and the two are asserted equal by the tests)."""
    t = P.F32Fma.evaluate(x, 0x3BBB989D, 0x3F000000)
    s = F32Sat.evaluate(t)
    jb = F32FmaRm.evaluate(s, bits_of(252.0), bits_of(12582913.0))
    j = P.F32Add.evaluate(jb, bits_of(-12583039.0))
    f = P.F32Fma.evaluate(x, 0x3FB8AA3B, F32Neg.evaluate(j))
    f = P.F32Fma.evaluate(x, 0x32A57060, f)
    return P.F32Mul.evaluate(P.MufuEx2Ftz.evaluate(f), F32BitsShl23.evaluate(jb))


def nv_sigmoid_bits(x: int) -> int:
    """The SASS of `sigmoid_accurate` instruction by instruction on one f32 encoding (FSETP.GE, FSEL and the LOP3 sign OR as they
    are; the probes' and tests' reference for `NvSigmoid_v1`)."""
    h = P.F32Mul.evaluate(x, 0x3F000000)
    ah = h & 0x7FFFFFFF
    a = P.F32Mul.evaluate(ah, 0x4038AA3B)
    s = P.F32Mul.evaluate(h, h)
    poly = not f32_of(ah) >= f32_of(0x3F19999A)
    big = f32_of(ah) >= f32_of(0x41102CB4)
    p = P.F32Fma.evaluate(s, 0x3C80F082, 0xBD563CAE)
    e = P.MufuEx2Ftz.evaluate(a)
    p = P.F32Fma.evaluate(s, p, 0x3E085941)
    d = P.F32Add.evaluate(e, 0x3F800000)
    p = P.F32Fma.evaluate(s, p, 0xBEAAA9ED)
    q = P.F32Fma.evaluate(s, p, 0)
    r = P.MufuRcpFtz.evaluate(d)
    t = 0x3F800000 if big else P.F32Fma.evaluate(r, 0xC0000000, 0x3F800000)
    t |= h & 0x80000000
    if poly:
        t = P.F32Fma.evaluate(h, q, h)
    return P.F32Fma.evaluate(t, 0x3F000000, 0x3F000000)
