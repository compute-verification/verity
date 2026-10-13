"""DeepSeek-V4-Pro's two attention-input kernels on the H200 (sm_90, vLLM d9105ea80): the query norm, RoPE and SWA cache insert
(`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`), and the indexer's query RoPE and FP8 quantisation
(`fused_indexer_q_rope_quant`, the CuTe DSL `IndexerQFp8Kernel`).  Line numbers are at d9105ea80; "the SASS" is nvcc 12.9's sm_90a
build of csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (no fast-math; both kernels, ReducedGrid and the
one CTA per token, have the same arithmetic) and the CuTe DSL 4.6.2 sm_90a build of
vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py (coarsen 1, below 512 tokens, and 4: the same arithmetic).

    F32ToU32RzByte_v1                          (x: f32) -> byte: the low byte of `cvt.rzi.u32.f32`
    Dsv4QNormRope_v1{D,DR,EPS}                 (q: D bf16, cs: DR f32) -> D bf16: one live query head
    Dsv4KvRopeQuant_v1{DN,DR,QB}               (kv: DN+DR bf16, cs: DR f32) -> (nope: DN e4m3, scale: DN/QB + 1 e8m0,
                                                                                 rope: DR bf16): one token's cache row
    Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}
                                               (q: NH x D bf16, kv: D bf16, cs: DR f32) -> (q: NHP x D bf16, nope, scale, rope)
    Dsv4IndexerQHead_v1{D,DR,SCALE}            (q: D bf16, cs: DR f32, w: bf16) -> (q: D e4m3, w: f32): one (token, head)
    Dsv4IndexerQ_v1{H,D,DR,SCALE}              (q: H x D bf16, cs: DR f32, w: H bf16) -> (q: H x D e4m3, w: H f32): one token

D = DN + DR; cs is the token's row of the f32 `cos_sin_cache` (DR/2 cosines, then DR/2 sines).  Both kernels rotate pair p of the
last DR elements (x_e, x_o at D - DR + 2p, + 1) by c = cs[p], s = cs[DR/2 + p] as the SASS contracts it: x_e' = FFMA(x_e, c,
-FMUL(x_o, s)) (the `sub.f32` of two products fused by ptxas, the addend negated) and x_o' = FFMA(x_e, s, FMUL(x_o, c)) (an explicit
`fma.rn` in the PTX); `rope` is that body, written once.

The insert, per token (`processDeepseekV4Slot`, :178-379; 32 lanes, lane l elements 16l .. 16l + 15):

* query head h < NH: x = f32(q); each lane's sum of squares in element order (the first a product: the SASS's FFMA(e0, e0, +0) is
  the same word), the 32 lane sums by the xor butterfly at offsets 16 .. 1 (`warpSum`, :154-160; `dsa.halving_tree`), r =
  `RsqrtApprox_v1`(FFMA(sum, 1/D, eps)) (`rsqrtf(sum / 512 + eps)`, :235-236: MUFU.RSQ with its GEU 2^-126 prescale), x * r, the
  RoPE, `F2fpBf16` (:278-297).  Heads NH .. NHP - 1 are bf16 +0 (:193-203);
* the kv row, written only when `slot_mapping[t] >= 0` (`Dsv4KvRopeQuant_v1`, the Program's scatter): x = f32(kv), the RoPE (no
  norm), each element's bf16 round trip (:315-318: exact on the NoPE elements but for a NaN's payload, which no output reads); per
  block of QB NoPE elements (4 lanes: `warp4MaxAbs`, :144-151) amax = fmaxf over fabsf from +0 (max.f32 drops a NaN, so the order
  is free), fmaxf(amax, 1e-4), s = `div.rn.f32`(amax, 448) (`F32Div_v2`), e = ceilf(`NvLog2f_v1`(s)) (`F32CeilFtz_v1`: its operand
  is never subnormal, s >= 1e-4/448 > 2^-23), inv = `NvExp2f_v1`(-e) (`ex2.approx.f32`); byte i `F32ToE4m3Sat`(fminf(fmaxf(x_i * inv, -448),
  448)) (:332-343; a NaN element is -448, 0xFE); the block's scale byte `F32ToU32RzByte_v1`(fmaxf(fminf(e + 127, 255), +0))
  (:346-350; +inf amax gives 255) and a 0 byte after the DN/QB of them (:351-353); the RoPE elements' bf16 (:354-371).  The cache row
  is the DN e4m3 bytes and DR bf16 at `pos_in_block * (DN + 2 DR)` and the scale bytes at `block_size * (DN + 2 DR) + pos_in_block *
  (DN/QB + 1)` of its block: what `flashmla.FlashMlaKvRow_v1{DN, DR, QT = QB}` dequantises.

The indexer, per (token, head) (`IndexerQFp8Kernel`, fused_indexer_q_cutedsl.py; 8 lanes of 16 elements): x = f32(q); the RoPE on
the last DR elements, each rotated element rounded to bf16 (`cvt.rn.bf16x2.f32`) and widened again; amax the bf16x2 |x| and max
over the head (`max.bf16x2`, the 8-lane butterfly, then max.f32 of the halves: max.f32 over the widened |x|, a NaN dropped, from +0),
fmax(amax, 1e-4) * f32(1/448) (`mul.f32`); the scale's word rounded up to a power of two `(bits + 0x7FFFFF) & 0x7F800000` and its
inverse's `0x7F000000 - bits` (the PTX's shifts by 23; +inf amax gives the scale +inf, the inverse -inf); byte i
`cvt.rn.satfinite.e4m3x2.f32`(x_i * inv) (no clamp); the weight f32(w) * f32(SCALE) * scale (`weights_out`, SCALE the host's
`softmax_scale * head_scale` in double, narrowed to f32 at the launch).  Not GLM-5.3's `dsa.DsaIndexerQHead_v1` (bf16 cos_sin, `div.rn`
by 448 and `__nv_log2f`), nor DSV4-IDX's MXFP4 path; `IndexerQFp8Kernel` is the kernel of record on sm_90 with CuTe DSL present.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.circuits.ops.mufu import RsqrtApprox
from verity.ml.circuits.ops.prims import Bf16ToF32, F2fpBf16, F32ToE4m3Sat, const
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, F32Fminf, F32Neg, I32Add
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.kernels import dsv4_qnorm_insert_twin
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import flashmla as F
from verity_vllm.program.registry.flashmla import E8M0, WORD_T
from verity_vllm.program.registry.prims import BF16, F32

M32 = 0xFFFFFFFF
LANES = 32

_CONF = ("model-untested: read from vLLM d9105ea80's fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu and the CuTe DSL "
         "IndexerQFp8Kernel (fused_indexer_q_cutedsl.py) and their sm_90a SASS; "
         "tests/properties/dsv4_qnorm_insert_capture_gpu.py is its hardware check")
_PTX = "model-untested: as the PTX ISA defines it; tests/properties/dsv4_qnorm_insert_capture_gpu.py is its hardware check"

F_448, F_NEG448, F_127, F_255 = 0x43E00000, 0xC3E00000, 0x42FE0000, 0x437F0000
#: the indexer's `Float32(1.0 / 448.0)`, and its scale's round-up addend, exponent mask and inverse's exponent base
F_INV448, ROUND_UP, EXP_MASK, INV_BASE = 0x3B124925, 0x007FFFFF, 0x7F800000, 0x7F000000

#: DeepSeek-V4-Pro at TP8 on the H200 (config.json at b5968e91): 16 query heads per rank padded to FlashMLA's 64, NoPE 448 +
#: RoPE 64, one scale per 64 NoPE elements, rms_norm_eps 1e-6
DSV4_INSERT = {"NH": 16, "NHP": 64, "DN": 448, "DR": 64, "QB": 64, "EPS": 1e-6}
#: the indexer's 64 heads of 128 (replicated at every rank), RoPE 64, `softmax_scale * head_scale` = 128^-0.5 * 64^-0.5
DSV4_INDEXER = {"H": 64, "D": 128, "DR": 64, "SCALE": 128 ** -0.5 * 64 ** -0.5}


def _nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def f32_to_u32_rz_byte(x: int) -> int:
    """The low byte of `cvt.rzi.u32.f32` (SASS `F2I.U32.TRUNC.NTZ`): x truncated toward zero and saturated to [0, 2^32 - 1] (a
    NaN, every negative and -0 to 0; +inf and every x >= 2^32 to 0xFFFFFFFF)."""
    x &= M32
    e = (x >> 23) & 0xFF
    if _nan(x) or x >> 31 or e < 127:
        return 0
    if e >= 127 + 32:
        return 0xFF
    m, k = (x & 0x7FFFFF) | 0x800000, e - 127
    return ((m << (k - 23)) if k >= 23 else (m >> (23 - k))) & 0xFF


@primitive("F32ToU32RzByte", 1, [("x", F32)], E8M0, conformance=_PTX)
def F32ToU32RzByte(x: int) -> int:
    """`static_cast<uint8_t>(float)` as nvcc compiles it: `cvt.rzi.u32.f32` and `st.global.u8` of its low byte
    (`f32_to_u32_rz_byte`)."""
    return f32_to_u32_rz_byte(x)


DSV4_QNORM_INSERT_PRIMITIVES = [F32ToU32RzByte]


class Ops(F.Ops):
    """`flashmla.Ops` with ``zero8``, the constant byte 0 (the scale bytes' pad)."""

    def __init__(self, B, defs: dict, word, neg, half, T: F.Types, leaf: int, byte):
        super().__init__(B, defs, word, neg, half, T, leaf)
        self._byte, self._zero8 = byte, None

    @property
    def zero8(self):
        if self._zero8 is None:
            self._zero8 = self._byte(0)
        return self._zero8


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x), lambda bits: B.call(const(16, bits)),
               WORD_T, 1, lambda bits: B.call(const(8, bits)))


def rope(o: Ops, x: list, cs, DR: int) -> list:
    """x with its last DR elements rotated (the module docstring's contraction)."""
    out, base, half = list(x), len(x) - DR, DR // 2
    for p in range(half):
        xe, xo, c, s = x[base + 2 * p], x[base + 2 * p + 1], cs[p], cs[half + p]
        out[base + 2 * p] = o.fma(xe, c, o.neg(o.mul(xo, s)))
        out[base + 2 * p + 1] = o.fma(xe, s, o.mul(xo, c))
    return out


def check_rope(D_: int, DR: int, name: str) -> None:
    if DR < 2 or DR % 2 or DR > D_:
        raise ValueError(f"{name}: DR={DR} must be even, at least 2 and at most D={D_}")


# ---- the insert ---------------------------------------------------------------------------------------------------------------

def check_q_statics(D_: int, DR: int) -> None:
    if D_ < LANES or D_ & (D_ - 1):
        raise ValueError(f"Dsv4QNormRope: D={D_} must be a power of two of at least {LANES} (one warp's lanes, sum / D a product)")
    check_rope(D_, DR, "Dsv4QNormRope")


def q_norm_rope(o: Ops, S, q, cs):
    """`Dsv4QNormRope_v1`'s body."""
    D_, DR = S.D, S.DR
    check_q_statics(D_, DR)
    L = D_ // LANES
    x = [o.wide(q[i]) for i in range(D_)]
    lanes = []
    for t in range(LANES):
        acc = o.mul(x[t * L], x[t * L])
        for i in range(t * L + 1, (t + 1) * L):
            acc = o.fma(x[i], x[i], acc)
        lanes.append(acc)
    r = o.rsqrt(o.fma(D.halving_tree(lanes, o.add), o.word(D.f32_bits(1.0 / D_)), o.word(D.f32_bits(S.EPS))))
    return array_of([o.cvt(v) for v in rope(o, [o.mul(v, r) for v in x], cs, DR)])


def _q_type(S, T: F.Types = WORD_T):
    return ((("q", Array(S.D, T.bf16)), ("cs", Array(S.DR, T.f32))), Array(S.D, T.bf16))


@composite("Dsv4QNormRope", 1, ["D", "DR", "EPS"], _q_type, conformance=_CONF)
def Dsv4QNormRope(B, S, q, cs):
    """One live query head of the insert (the module docstring's first item): the weightless RMSNorm of f32(q) as one warp's
    32 lanes reduce it, `RsqrtApprox_v1`, the RoPE of its last DR elements, `F2fpBf16`."""
    return q_norm_rope(word_ops(B), S, q, cs)


def check_kv_statics(DN: int, DR: int, QB: int) -> None:
    if QB < 1 or DN < QB or DN % QB:
        raise ValueError(f"Dsv4KvRopeQuant: QB={QB} must divide DN={DN} (one scale per QB NoPE elements)")
    check_rope(DN + DR, DR, "Dsv4KvRopeQuant")


def kv_rope_quant(o: Ops, S, kv, cs):
    """`Dsv4KvRopeQuant_v1`'s body."""
    DN, DR, QB = S.DN, S.DR, S.QB
    check_kv_statics(DN, DR, QB)
    x = rope(o, [o.wide(kv[i]) for i in range(DN + DR)], cs, DR)
    nope, scale = [], []
    for j in range(DN // QB):
        blk = x[j * QB:(j + 1) * QB]
        amax = o.word(0)
        for v in blk:
            amax = o.fmax(amax, o.fabs(v))
        e = o.ceil(o.log2(o.div(o.fmax(amax, o.word(D.F_1EM4)), o.word(F_448))))
        inv = o.exp2(o.neg(e))
        nope += [o.e4m3(o.fmin(o.fmax(o.mul(v, inv), o.word(F_NEG448)), o.word(F_448))) for v in blk]
        scale.append(o.byte(o.fmax(o.fmin(o.add(e, o.word(F_127)), o.word(F_255)), o.word(0))))
    return tuple_of(array_of(nope), array_of(scale + [o.zero8]), array_of([o.cvt(v) for v in x[DN:]]))


def _kv_out(S, T: F.Types):
    return Tuple(Array(S.DN, T.e4m3), Array(S.DN // S.QB + 1, T.e8m0), Array(S.DR, T.bf16))


def _kv_type(S, T: F.Types = WORD_T):
    return ((("kv", Array(S.DN + S.DR, T.bf16)), ("cs", Array(S.DR, T.f32))), _kv_out(S, T))


@composite("Dsv4KvRopeQuant", 1, ["DN", "DR", "QB"], _kv_type, conformance=_CONF)
def Dsv4KvRopeQuant(B, S, kv, cs):
    """One token's SWA cache row (the module docstring's second item): the RoPE of f32(kv)'s last DR elements; per QB NoPE
    elements the UE8M0 scale from their amax and their saturated E4M3 bytes; the scale bytes and a 0 byte; the RoPE elements'
    bf16."""
    return kv_rope_quant(word_ops(B), S, kv, cs)


def check_insert_statics(NH: int, NHP: int) -> None:
    if not 1 <= NH <= NHP:
        raise ValueError(f"Dsv4QNormRopeKvInsert: the live heads NH={NH} must be in [1, NHP={NHP}]")


def insert(o: Ops, S, head, row, q, kv, cs):
    """`Dsv4QNormRopeKvInsert_v1`'s body over the Calls ``head`` (`Dsv4QNormRope`) and ``row`` (`Dsv4KvRopeQuant`)."""
    check_insert_statics(S.NH, S.NHP)
    D_ = S.DN + S.DR
    heads = o.B.batch(head, q, cs, axes=(0, None))
    zero = array_of([o.half(0)] * D_)
    r = o.B.call(row, kv, cs)
    return tuple_of(array_of([heads[h] for h in range(S.NH)] + [zero] * (S.NHP - S.NH)), r[0], r[1], r[2])


def _insert_type(S, T: F.Types = WORD_T):
    D_ = S.DN + S.DR
    out = _kv_out(S, T)
    return ((("q", Array(S.NH, Array(D_, T.bf16))), ("kv", Array(D_, T.bf16)), ("cs", Array(S.DR, T.f32))),
            Tuple(Array(S.NHP, Array(D_, T.bf16)), *out.fields))


@composite("Dsv4QNormRopeKvInsert", 1, ["NH", "NHP", "DN", "DR", "QB", "EPS"], _insert_type, conformance=_CONF)
def Dsv4QNormRopeKvInsert(B, S, q, kv, cs):
    """One token of `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`: head h < NH `Dsv4QNormRope_v1{D, DR, EPS}` of q[h], the
    padded heads bf16 +0, and the kv row `Dsv4KvRopeQuant_v1{DN, DR, QB}` (stored at the token's slot when it is >= 0)."""
    return insert(word_ops(B), S, bind(Dsv4QNormRope, D=S.DN + S.DR, DR=S.DR, EPS=S.EPS),
                  bind(Dsv4KvRopeQuant, DN=S.DN, DR=S.DR, QB=S.QB), q, kv, cs)


# ---- the indexer's query ------------------------------------------------------------------------------------------------------

def indexer_q_head(o: Ops, S, q, cs, w):
    """`Dsv4IndexerQHead_v1`'s body."""
    D_, DR = S.D, S.DR
    check_rope(D_, DR, "Dsv4IndexerQHead")
    x = rope(o, [o.wide(q[i]) for i in range(D_)], cs, DR)
    x = x[:D_ - DR] + [o.wide(o.cvt(v)) for v in x[D_ - DR:]]
    amax = o.word(0)
    for v in x:
        amax = o.fmax(amax, o.fabs(v))
    sb = o.iand(o.iadd(o.mul(o.fmax(amax, o.word(D.F_1EM4)), o.word(F_INV448)), o.word(ROUND_UP)), o.word(EXP_MASK))
    inv = o.isub(o.word(INV_BASE), sb)
    wout = o.mul(o.mul(o.wide(w), o.word(D.f32_bits(S.SCALE))), sb)
    return tuple_of(array_of([o.e4m3(o.mul(v, inv)) for v in x]), wout)


def _head_type(S, T: F.Types = WORD_T):
    return ((("q", Array(S.D, T.bf16)), ("cs", Array(S.DR, T.f32)), ("w", T.bf16)), Tuple(Array(S.D, T.e4m3), T.f32))


@composite("Dsv4IndexerQHead", 1, ["D", "DR", "SCALE"], _head_type, conformance=_CONF)
def Dsv4IndexerQHead(B, S, q, cs, w):
    """One (token, head) of `IndexerQFp8Kernel` (the module docstring's last paragraph): the RoPE of f32(q)'s last DR elements
    through bf16, the power-of-two scale from the head's amax, its E4M3 bytes, and the head's weight times f32(SCALE) and the
    scale (`index_weights_out`)."""
    return indexer_q_head(word_ops(B), S, q, cs, w)


def _token_type(S, T: F.Types = WORD_T):
    return ((("q", Array(S.H, Array(S.D, T.bf16))), ("cs", Array(S.DR, T.f32)), ("w", Array(S.H, T.bf16))),
            Tuple(Array(S.H, Array(S.D, T.e4m3)), Array(S.H, T.f32)))


@composite("Dsv4IndexerQ", 1, ["H", "D", "DR", "SCALE"], _token_type, conformance=_CONF)
def Dsv4IndexerQ(B, S, q, cs, w):
    """`fused_indexer_q_rope_quant`'s FP8 outputs for one token (`q_fp8` [H, D], `weights_out` [H], `DsaMqaLogit_v1{H, D}`'s q and
    w): head h `Dsv4IndexerQHead_v1{D, DR, SCALE}` over q[h] and w[h]."""
    return D.heads(B, bind(Dsv4IndexerQHead, D=S.D, DR=S.DR, SCALE=S.SCALE), S.H, S.D, q, cs, w)


WORD = dict(wide=Bf16ToF32, cvt=F2fpBf16, add=F32AddV2, mul=F32MulV2, fma=F32FmaV2, div=F32DivV2, rsqrt=RsqrtApprox, fabs=F32Fabs,
            fmax=F32Fmaxf, fmin=F32Fminf, e4m3=F32ToE4m3Sat, iadd=I32Add, iand=D.U32And, isub=D.I32Sub, ceil=D.F32CeilFtz,
            byte=F32ToU32RzByte, log2=F.NvLog2f, exp2=F.NvExp2f)

DSV4_QNORM_INSERT_COMPOSITES = [Dsv4QNormRope, Dsv4KvRopeQuant, Dsv4QNormRopeKvInsert, Dsv4IndexerQHead, Dsv4IndexerQ]

dsv4_qnorm_insert_twin.register_twin({d.id: d for d in DSV4_QNORM_INSERT_COMPOSITES})
