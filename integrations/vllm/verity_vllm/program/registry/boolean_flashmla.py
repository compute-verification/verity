"""`flashmla`'s DeepSeek-V4-Pro sparse MLA decode on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `flashmla`'s, written once over `flashmla.Ops`), so each one's word view is its `_v1` at the same statics
with the Boolean ``DOT`` read as its word view.

    F32Geu_v2, U32Ge_v2, E8m0ToBf16_v2                traced: setp.geu.f32, setp.ge.u32, __nv_cvt_e8m0_to_bf16raw
    NvExp2f_v2, NvLog2f_v2, NvFdividefRcp_v2
    FlashMlaRowValid_v2                               traced: one AND of `idx != -1` and `pos < len`
    FlashMlaKvRow_v2{DN,DR,QT}
    FlashMlaBlock_v2{D,BN,FIRST,DOT,SCALE}            DOT: `gemm.HopperBF16WgmmaDot16_v2` (the decode's wgmma k16 step)
    FlashMlaSplit_v2{NB,D,BN,PARTIAL,DOT,SCALE}
    FlashMlaCombine_v2{NS,D}
    FlashMlaDecodeHead_v2{SPLITS,D,BN,DOT,SCALE}
    FlashMlaDecode_v2{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}

The callees are core's Boolean pieces: `DotBf16_v3{K = D, DOT}` and `DOT`, `F2fpBf16_v2`, the GPU-NaN f32 operations
`F32Add_v4` / `F32Mul_v4` / `F32Fma_v4`, `F32Fmaxf_v2`, `F32Eq_v2`, the selects, `GuardNegInfZero_v2`, `MufuEx2Ftz_v2`,
`MufuRcpFtz_v2`, `E4m3ToF32_v2`, `Bf16MulBf16_v2`, `I32Add_v2` and `boolean_dsa`'s integer steps; a constant word is constant bits
and `-x` one NOT gate on x's sign bit.  No table or lookup gate.  Each composite is one traced circuit (`boolean_dsa.flat`;
`NvLog2f_v2` traced twice, `flat_twice`), so a row's scales, a block's rescale and the query every score reads are decoded once.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, and_reduce, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition, trace
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import flashmla as F
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.boolean_dsa import I32SubV2, I32ToF32V2, U32AndV2, flat, replay

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
E8 = Array(8, BIT)
BIT_T = F.Types(BF16, F32, BIT, E8, E8, I32)
_AB = (("a", F32), ("b", F32))


def all_any(C, x):
    """(AND, OR) of the bits x, the OR's first level reading the AND's first-level pairs (OR(a, b) = a ^ b ^ ab): a lowering
    that does not share repeated ANDs (a C-Flock piece) computes each pair once."""
    ab = [C.AND(x[i], x[i + 1]) for i in range(0, len(x) - 1, 2)]
    odd = [x[-1]] if len(x) % 2 else []
    return and_reduce(C, ab + odd), or_reduce(C, [x[2 * i] ^ x[2 * i + 1] ^ p for i, p in enumerate(ab)] + odd)


def f32_key(C, x):
    """`scalar.f32_key` over `all_any` of the exponent."""
    m, e, s = x[:23], x[23:31], x[31]
    (e_all, e_any), m_any = all_any(C, e), or_reduce(C, m)
    nan = C.AND(e_all, m_any)
    neg = C.AND(s, OR(C, e_any, m_any))
    return nan, [b ^ neg for b in x[:31]] + [NOT(neg)]


def f32_geu(C, a, b):
    """`setp.geu.f32`: NOT of the ordered `b > a` (`f32_key`: +0 and -0 one key, subnormals as they are), so a NaN on either side
    gives 1."""
    na, ka = f32_key(C, a)
    nb, kb = f32_key(C, b)
    return [NOT(C.AND(NOT(OR(C, na, nb)), SC.gt_u(C, kb, ka)))]


F32GeuV2 = definition("F32Geu", 2, f32_geu, _AB, BIT, word=F.F32Geu, doc="setp.geu.f32 on bits: F32Geu_v1")
U32GeV2 = definition("U32Ge", 2, lambda C, a, b: [NOT(sub(C, a, b, 32)[1])], _AB, BIT, word=F.U32Ge,
                     doc="setp.ge.u32 on bits: U32Ge_v1, NOT the borrow of a - b")


def e8m0_to_bf16(C, x):
    """`flashmla.e8m0_to_bf16`: bits 7..14 are x and bit 15 clear in every case; 0xFF (all ones) sets bits 0..6 (0x7FFF), 0
    (all clear) bit 6 (0x0040)."""
    nan, nonzero = all_any(C, x)
    return [nan] * 6 + [nan ^ NOT(nonzero)] + list(x) + [ZERO]


E8m0ToBf16V2 = definition("E8m0ToBf16", 2, e8m0_to_bf16, (("x", E8),), BF16, word=F.E8m0ToBf16,
                          doc="__nv_cvt_e8m0_to_bf16raw on bits: E8m0ToBf16_v1")

BITS = dict(add=EW.F32AddV4, mul=EW.F32MulV4, fma=EW.F32FmaV4, fmax=SC.F32Fmaxf, eq=SC.F32Eq, sel=SC.SelectF32, bsel=SC.SelectBf16,
            guard=BA.GuardNegInfZero, ex2=BM.MufuEx2Ftz, rcp=BM.MufuRcpFtz, fabs=SC.F32Fabs, geu=F32GeuV2, uge=U32GeV2,
            iadd=EW.I32Add, iand=U32AndV2, isub=I32SubV2, i2f=I32ToF32V2, cvt=BG.F2fpBf16, e4m3=SC.E4m3ToF32, e8m0=E8m0ToBf16V2,
            bmul=EW.Bf16MulBf16)

#: DeepSeek-V4-Pro at TP8 on bits: `flashmla.FLASHMLA_DSV4` with the Boolean DOT
FLASHMLA_DSV4 = {**F.FLASHMLA_DSV4, "DOT": BG.HopperBF16WgmmaDot16}
_LIFTED = ("DOT",)


def bit_ops(B) -> F.Ops:
    c = _Bits(B)
    return F.Ops(B, BITS, lambda v: c.word(v, 32), lambda x: Coll(F32, concat([x.refs.slice(0, 31), B.call(BL.NOT, x[31]).refs])),
                 lambda v: c.word(v, 16), BIT_T, 16)


_X = lambda S: ((("x", F32),), F32)  # noqa: E731


def _exp2(B, S, x):
    """`NvExp2f_v1`'s body on bits."""
    return F.exp2f(bit_ops(B), x)


def _log2(B, S, x):
    """`NvLog2f_v1`'s body on bits."""
    return F.log2f(bit_ops(B), x)


def _rcp1(B, S, x):
    """`NvFdividefRcp_v1`'s body on bits."""
    return F.fdividef_rcp(bit_ops(B), x)


def flat_twice(name: str, version: int, signature, body, word):
    """`boolean_dsa.flat` (no statics) traced a second time, over the gates of its first trace.  The first writes a form it cannot
    track (one made through ANDs no output reads: here an `F32Fma_v4` on constant operands) as a fresh XOR tree, which can
    repeat an XOR it already wrote; the second makes every form from the first's gates, so a repeat is one wire."""
    inner = EW.on_bits(name, version, [], signature, body, word, register=False)
    params, ret = signature(None)
    types = [t for _, t in params]

    def build(S):
        c = BS.compile(bind(inner))
        first = trace(lambda C, *x: [replay(c, C, [f for e in x for f in e])], types, ret)

        def run(C, *x):
            w = [ZERO, ONE, *(f for e in x for f in e)]
            for kind, (a, *b) in first.gates:
                w.append(C.AND(w[a], w[b[0]]) if kind == "and" else w[a] ^ w[b[0]] if kind == "xor" else NOT(w[a]))
            return [w[i] for i in first.outputs]
        return run

    return EW.specialized(name, version, [], build, signature, doc=body.__doc__ or "", word=inner.word)


NvExp2fV2 = flat("NvExp2f", 2, [], _X, _exp2, F.NvExp2f)
NvLog2fV2 = flat_twice("NvLog2f", 2, _X, _log2, F.NvLog2f)
NvFdividefRcpV2 = flat("NvFdividefRcp", 2, [], _X, _rcp1, F.NvFdividefRcp)
BITS.update(exp2=NvExp2fV2, log2=NvLog2fV2, rcp1=NvFdividefRcpV2)


def row_valid(C, idx, n, pos):
    """`idx != -1` is NOT the AND of its bits, `pos < len` (signed) the borrow of `(pos ^ 2^31) - (len ^ 2^31)`."""
    flip = lambda x: list(x[:31]) + [NOT(x[31])]  # noqa: E731
    return [C.AND(NOT(and_reduce(C, idx)), sub(C, flip(pos), flip(n), 32)[1])]


FlashMlaRowValidV2 = definition("FlashMlaRowValid", 2, row_valid, (("idx", I32), ("len", I32), ("pos", I32)), BIT,
                                word=bind(F.FlashMlaRowValid), doc="FlashMlaRowValid_v1 on bits: " + row_valid.__doc__)


def _row(B, S, nope, scale, rope, valid):
    """`FlashMlaKvRow_v1`'s body on bits."""
    return F.kv_row(bit_ops(B), S, nope, scale, rope, valid)


FlashMlaKvRowV2 = flat("FlashMlaKvRow", 2, ["DN", "DR", "QT"], lambda S: F._row_type(S, BIT_T), _row, F.FlashMlaKvRow)


def _block(B, S, q, k, nm_prev=None, l_prev=None, o_prev=None):
    """`FlashMlaBlock_v1`'s body on bits: the score chain `DotBf16_v3{K = D, DOT}`, the P V step `DOT`."""
    return F.block(bit_ops(B), S, bind(BG.DotBf16, K=S.D, DOT=S.DOT), q, k, nm_prev, l_prev, o_prev)


FlashMlaBlockV2 = flat("FlashMlaBlock", 2, ["D", "BN", "FIRST", "DOT", "SCALE"], lambda S: F._block_type(S, BIT_T), _block,
                       F.FlashMlaBlock, lifted=_LIFTED)


def _split(B, S, q, k, sink=None):
    """`FlashMlaSplit_v1`'s body on bits over `FlashMlaBlock_v2`."""
    def block_of(first):
        return bind(FlashMlaBlockV2, D=S.D, BN=S.BN, FIRST=first, DOT=S.DOT, SCALE=S.SCALE)
    return F.split(bit_ops(B), S, block_of, q, k, sink)


FlashMlaSplitV2 = flat("FlashMlaSplit", 2, ["NB", "D", "BN", "PARTIAL", "DOT", "SCALE"], lambda S: F._split_type(S, BIT_T), _split,
                       F.FlashMlaSplit, lifted=_LIFTED)


def _combine(B, S, lse, oacc, sink):
    """`FlashMlaCombine_v1`'s body on bits."""
    return F.combine(bit_ops(B), S, lse, oacc, sink)


FlashMlaCombineV2 = flat("FlashMlaCombine", 2, ["NS", "D"], lambda S: F._combine_type(S, BIT_T), _combine, F.FlashMlaCombine)


def _head(B, S, q, k, sink):
    """`FlashMlaDecodeHead_v1`'s body on bits over `FlashMlaSplit_v2` and `FlashMlaCombine_v2`."""
    def split_of(nb, partial):
        return bind(FlashMlaSplitV2, NB=nb, D=S.D, BN=S.BN, PARTIAL=partial, DOT=S.DOT, SCALE=S.SCALE)
    return F.decode_head(bit_ops(B), S, split_of, lambda ns: bind(FlashMlaCombineV2, NS=ns, D=S.D), q, k, sink)


FlashMlaDecodeHeadV2 = flat("FlashMlaDecodeHead", 2, ["SPLITS", "D", "BN", "DOT", "SCALE"], lambda S: F._head_type(S, BIT_T), _head,
                            F.FlashMlaDecodeHead, lifted=_LIFTED)


def _decode(B, S, q, nope, scale, rope, idx, *rest):
    """`FlashMlaDecode_v1`'s body on bits over `FlashMlaRowValid_v2`, `FlashMlaKvRow_v2` and `FlashMlaDecodeHead_v2`."""
    row = bind(FlashMlaKvRowV2, DN=S.DN, DR=S.DR, QT=S.QT)
    head = bind(FlashMlaDecodeHeadV2, SPLITS=tuple(S.SPLITS), D=S.DN + S.DR, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return F.decode(bit_ops(B), S, row, FlashMlaRowValidV2, head, q, nope, scale, rope, idx, rest[:-1], rest[-1])


FlashMlaDecodeV2 = flat("FlashMlaDecode", 2, ["NH", "SPLITS", "NW", "DN", "DR", "QT", "BN", "DOT", "SCALE"],
                        lambda S: F._decode_type(S, BIT_T), _decode, F.FlashMlaDecode, lifted=_LIFTED)

BOOLEAN_FLASHMLA = [F32GeuV2, U32GeV2, E8m0ToBf16V2, NvExp2fV2, NvLog2fV2, NvFdividefRcpV2, FlashMlaRowValidV2, FlashMlaKvRowV2,
                    FlashMlaBlockV2, FlashMlaSplitV2, FlashMlaCombineV2, FlashMlaDecodeHeadV2, FlashMlaDecodeV2]
