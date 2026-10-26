"""`dsv4_compressor`'s Definitions on bits: version 2 of each, its word body line for line over gates (the bodies are
`dsv4_compressor`'s, written once over its steps), so each one's word view is its `_v1` at the same statics.

    Ue8m0CeilByte_v2, Ue8m0InvScale_v2                  traced: the integer add's bits 23..30; 254 - e in bits 23..31
    Dsv4MlaC4Column_v2{ROWS}, Dsv4MlaC128Column_v2{W,R}, Dsv4MlaRmsNorm_v2{RATIO,D,QB,EPS}, Dsv4MlaQuantGroup_v2{QB},
    Dsv4MlaRope_v2{ROPE}                                traced at their statics (`EW.specialized`)
    Dsv4MlaNormRopeQuant_v2{RATIO,D,QB,EPS}             Calls of the norm, the group quantisation and the RoPE, and the pad
                                                        byte's constant bits (`EW.on_bits`)
    Dsv4MlaCompressC4_v2{ROWS,D,QB,EPS}, Dsv4MlaCompressC128_v2{W,R,D,QB,EPS}
                                                        Calls of the column per column and of the row (`EW.on_bits`)

A step is `boolean_dsv4_indexer.GateOps`' gates for the word primitive `dsv4_compressor.WORD` names, with `fp.f32_div` at the
GPU's NaN word for `div.rn.f32` (`F32Div_v2`, as `F32Div_v4` lowers it) and the two UE8M0 steps above; a constant word is
constant bits, so it folds into the gates.  An FMA whose product is a constant zero (the ratio-4 column's first row, against
its zero l and acc) is `fma_zero_product`.  No table or lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import OR, ZERO, add, const_bits, mux, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp as FP
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import boolean_dsv4_indexer as BI
from verity_vllm.program.registry import dsv4_compressor as C
from verity_vllm.program.registry.boolean_attention import _Bits

BITS = SimpleNamespace(**vars(BI.BITS), E8M0=Array(8, BL.BIT))
F32, E8 = BITS.F32, BITS.E8M0


def ue8m0_byte(C_, s):
    """`dsv4_compressor.Ue8m0CeilByte`: bits 23..30 of the 32-bit sum bits(s) + 0x7FFFFF."""
    return add(C_, list(s), const_bits(0x7FFFFF, 32), width=32)[23:31]


def ue8m0_inv(C_, e):
    """`dsv4_compressor.Ue8m0InvScale`: (254 - e) mod 2^9 in bits 23..31, the low 23 bits clear."""
    return [ZERO] * 23 + sub(C_, const_bits(254, 9), list(e) + [ZERO], 9)[0]


Ue8m0CeilByteV2 = definition("Ue8m0CeilByte", 2, ue8m0_byte, (("s", F32),), E8, word=C.Ue8m0CeilByte,
                             doc="The UE8M0 byte of a scale word on bits: Ue8m0CeilByte_v1")
Ue8m0InvScaleV2 = definition("Ue8m0InvScale", 2, ue8m0_inv, (("e", E8),), F32, word=C.Ue8m0InvScale,
                             doc="The inverse scale of a UE8M0 byte on bits: Ue8m0InvScale_v1")


def zero_word(w) -> bool:
    """Whether ``w``'s exponent and mantissa bits are the constant 0 (a zero of any sign)."""
    return all(f.is_const() and not f.c for f in w[:31])


def fma_zero_product(C_, a, b, c):
    """`fp.f32_fma` at the GPU's NaN word when ``a`` or ``b`` is a constant zero: its product's zero flag is the constant 1, so
    its last select takes the addend (or the zero of the sum's sign) and never the rounded sum. The same forms, written without
    the rounding: through it, each result form's making would pass bits that cancel out of it, and a trace writes such a form
    as an XOR tree of its own, unshared."""
    A, B, Cc = FP.unpack(C_, a), FP.unpack(C_, b), FP.unpack(C_, c)
    sp = A.s ^ B.s
    r = mux(C_, Cc.zero, FP.f32_const(0)[:31] + [C_.AND(sp, Cc.s)], list(c))
    prod_inf = OR(C_, A.inf, B.inf)
    nan = OR(C_, OR(C_, OR(C_, A.nan, B.nan), Cc.nan),
             OR(C_, OR(C_, C_.AND(A.inf, B.zero), C_.AND(A.zero, B.inf)), C_.AND(C_.AND(prod_inf, Cc.inf), sp ^ Cc.s)))
    return FP.specials(C_, r, nan, OR(C_, prod_inf, Cc.inf), mux(C_, prod_inf, [sp], [Cc.s])[0], BI.PTX)


class GateOps(BI.GateOps):
    """`boolean_dsv4_indexer.GateOps` with `dsv4_compressor.WORD`'s ``div``, ``ue8m0`` and ``inv``, and an FMA of a constant
    zero product as `fma_zero_product`."""

    def fma(self, a, b, c):
        if zero_word(a) or zero_word(b):
            return fma_zero_product(self.C, a, b, c)
        return super().fma(a, b, c)

    def div(self, a, b):
        return FP.f32_div(self.C, a, b, nan_word=BI.PTX)

    def ue8m0(self, s):
        return ue8m0_byte(self.C, s)

    def inv(self, e):
        return ue8m0_inv(self.C, e)


def traced(name: str, statics: list[str], signature, body, word):
    """The Boolean Definition ``name_v2{statics}``: `dsv4_compressor`'s ``body`` over `GateOps`, traced per binding, its word view
    ``word`` at the same statics (`boolean_dsv4_indexer.traced` over these steps)."""
    sig = signature(BITS)

    def build(S):
        counts = [len(BL.elements(t)) for _, t in sig(S)[0]]

        def run(C_, *elements):
            parts, i = [], 0
            for n in counts:
                parts.append(list(elements[i:i + n]))
                i += n
            return body(GateOps(C_), S, *parts)
        return run

    return EW.specialized(name, 2, statics, build, sig, doc=f"`{word.name}_v1`'s body on bits: {(body.__doc__ or '').strip()}",
                          word=lambda b: bind(word, **b))


Dsv4MlaC4ColumnV2 = traced("Dsv4MlaC4Column", ["ROWS"], C.c4_column_sig, C.c4_column, C.Dsv4MlaC4Column)
Dsv4MlaC128ColumnV2 = traced("Dsv4MlaC128Column", ["W", "R"], C.c128_column_sig, C.c128_column, C.Dsv4MlaC128Column)
Dsv4MlaRmsNormV2 = traced("Dsv4MlaRmsNorm", ["RATIO", "D", "QB", "EPS"], C.norm_sig, C.rms_norm, C.Dsv4MlaRmsNorm)
Dsv4MlaQuantGroupV2 = traced("Dsv4MlaQuantGroup", ["QB"], C.quant_sig, C.quant_group, C.Dsv4MlaQuantGroup)
Dsv4MlaRopeV2 = traced("Dsv4MlaRope", ["ROPE"], C.rope_sig, C.rope, C.Dsv4MlaRope)


def _nrq(B, S, x, w, cs):
    """`Dsv4MlaNormRopeQuant_v1`'s body on bits: Calls of `Dsv4MlaRmsNorm_v2`, `Dsv4MlaQuantGroup_v2{QB}` per NoPE group and
    `Dsv4MlaRope_v2{QB}`, and the pad byte's constant bits."""
    return C.norm_rope_quant(B, S, x, w, cs, BIT_DEFS, BITS)


Dsv4MlaNormRopeQuantV2 = EW.on_bits("Dsv4MlaNormRopeQuant", 2, ["RATIO", "D", "QB", "EPS"], C.nrq_sig(BITS), _nrq,
                                    C.Dsv4MlaNormRopeQuant)


def _c4(B, S, kv, score, w, cs):
    """`Dsv4MlaCompressC4_v1`'s body on bits: Calls of `Dsv4MlaC4Column_v2{ROWS}` per column and
    `Dsv4MlaNormRopeQuant_v2{4, D, QB, EPS}`."""
    return C.compress(B, S, kv, score, w, cs, bind(BIT_DEFS.c4, ROWS=S.ROWS), S.ROWS, 4, BIT_DEFS, BITS)


def _c128(B, S, kv, score, w, cs):
    """`Dsv4MlaCompressC128_v1`'s body on bits: Calls of `Dsv4MlaC128Column_v2{W, R}` per column and
    `Dsv4MlaNormRopeQuant_v2{128, D, QB, EPS}`."""
    return C.compress(B, S, kv, score, w, cs, bind(BIT_DEFS.c128, W=S.W, R=S.R), S.W * S.R, 128, BIT_DEFS, BITS)


Dsv4MlaCompressC4V2 = EW.on_bits("Dsv4MlaCompressC4", 2, ["ROWS", "D", "QB", "EPS"], C.compress_sig(BITS, C.c4_rows), _c4,
                                 C.Dsv4MlaCompressC4)
Dsv4MlaCompressC128V2 = EW.on_bits("Dsv4MlaCompressC128", 2, ["W", "R", "D", "QB", "EPS"], C.compress_sig(BITS, C.c128_rows),
                                   _c128, C.Dsv4MlaCompressC128)

BIT_DEFS = SimpleNamespace(c4=Dsv4MlaC4ColumnV2, c128=Dsv4MlaC128ColumnV2, norm=Dsv4MlaRmsNormV2, quant=Dsv4MlaQuantGroupV2,
                           rope=Dsv4MlaRopeV2, nrq=Dsv4MlaNormRopeQuantV2, byte=lambda B, v: _Bits(B).word(v, 8))

BOOLEAN_DSV4_COMPRESSOR = [Ue8m0CeilByteV2, Ue8m0InvScaleV2, Dsv4MlaC4ColumnV2, Dsv4MlaC128ColumnV2, Dsv4MlaRmsNormV2,
                           Dsv4MlaQuantGroupV2, Dsv4MlaRopeV2, Dsv4MlaNormRopeQuantV2, Dsv4MlaCompressC4V2, Dsv4MlaCompressC128V2]
