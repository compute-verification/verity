"""[DeepSeek-V4-Pro] The sparse MLA decode attention as the H200 runs it: DeepSeek-V4-Pro at TP8 on sm_90, vLLM d9105ea80's
`DeepseekV4FlashMLA` backend (vllm/models/deepseek_v4/nvidia/flashmla.py) over the FlashMLA it vendors (6bc49418): the `sparse_fp8`
decode kernel `flash_fwd_splitkv_mla_fp8_sparse_kernel<MODEL1, 64>` (csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh), its tile
scheduler (csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu) and its split combine
(csrc/smxx/decode/combine/combine.cu).  Line numbers are at those commits; "the SASS" is the decode kernel's sm_90a SASS at
DSV4's instance (nvcc without fast-math: FADD / FMUL / FFMA round to nearest with subnormals, every NaN result 0x7FFFFFFF).

The call (flashmla.py:248-266): per decode token one query row of h_q = 64 heads (the rank's 16 real heads padded to 64,
`get_padded_num_q_heads`, flashmla.py:78-85; the pad heads' sink -inf, attention.py:231-237), over its sliding-window rows
(`indices` = decode_swa_indices, `topk_length` = swa_lens) and, on a compressed layer (compress_ratio 4 or 128), its compressed
rows (`extra_indices_in_kvcache`, `extra_topk_length`).  Both caches are `fp8_ds_mla` (MODEL1): per token 448 E4M3 NoPE bytes and
64 bf16 RoPE elements, and 8 UE8M0 scale bytes in the block's scale region, of which 7 are read (one per 64 NoPE elements,
:553-560).  `softmax_scale` is head_dim^-0.5 = 512^-0.5 (attention.py:228); the kernel's `sm_scale_div_log2` is
`float(sm_scale * 1.44269504f)` in double (api/sparse_decode.h:414, api/common.h:31).  K = V = the 512-element row
[NoPE 448 | RoPE 64]; the output is 512 bf16 per head.

1. Rows (the producer warpgroup, :466-641).  A request's rows in kernel order are its window rows padded to
   NW = max(ceil64(swa_len), 64), then its extra rows padded to ceil64(extra_len), in blocks of 64 (`get_cur_req_info`,
   :164-183).  Row r of a segment is valid iff its index is not -1 and r < the segment's length (:533-535, :630-636):
   `FlashMlaRowValid_v1`.  An invalid row's NoPE bytes, scales and RoPE are zeroed (:576-581, :594, :615); a NoPE element is
   `__float22bfloat162_rn(float(e4m3)) * scale` (dequant.h:21-34: one bf16 multiply), the scale `__nv_cvt_e8m0x2_to_bf162raw`
   (:560; 0xFF -> 0x7FFF, 0 -> 0x0040, else e << 7): `FlashMlaKvRow_v1`.  A staged row is its D bf16 elements and its
   validity bit (sK and `is_kv_valid`, :638-641).
2. Blocks (the two consumer warpgroups, :208-270, `scale_softmax` :32-84), per head, the running state m = -1e30, l = 0, o = 0
   at a split's start (:211-213):
       s[c]  = valid[c] ? DotBf16{K = 512, HopperBF16WgmmaDot16}(q, k[c]) : -inf          (one zero-initialised wgmma chain)
       cur   = FMUL(fmaxf over {-inf} and s, scale_log2);  m' = fmaxf(cur, m);  r = exp2f(FADD(m, -m'))
       P[c]  = exp2f(FFMA(s[c], scale_log2, -m'));  o = FMUL(o, r) then 4 ascending k16 wgmma steps of bf16(P) . V
       l[t]  = FFMA(l[t], r, sum of P over thread t's 16 columns in ascending c)        (column c is thread (c % 8) / 2's)
   `FlashMlaBlock_v1`; the fmaxf fold is order-free, so the per-thread folds and their butterfly are one fold.  On a split's
   first block the rescale is an identity (FMUL(0, r) = +0, FFMA(0, r, sum) = sum: r is finite there), so it is not written.
3. A split's end (:286-362): L = (l0 + l1) + (l2 + l3) (the xor butterfly, every thread the same word).  A request in one split
   writes bf16(FMUL(o, L == 0 ? 0 : __fdividef(1, FADD(L, exp2f(FADD(FMUL(sink, L2E), -m)))))); a split of several writes
   oacc = FMUL(o, L == 0 ? 0 : __fdividef(1, L)) and lse = L == 0 ? -inf : FADD(log2f(L), m): `FlashMlaSplit_v1`.
4. The combine (combine.cu:19-160), per head over its NS splits (split s on lane s % 32): max = GuardNegInfZero(fmaxf over
   {-inf} and lse); per lane the FADD chain of exp2f(FADD(lse[s], -max)) over its splits, then the lane-0 halving tree (padding
   lanes hold +0, an exact identity on these sums); g = sum == 0 ? +inf : FADD(log2f(sum), max) (the dead `sum == -inf` arm
   dropped: no exp2f is -inf); the sink: g == +inf ? (sink == -inf ? +inf : FMUL(sink, L2E)) :
   FADD(g, log2f(FADD(1, exp2f(FFMA(sink, L2E, -g))))); out = bf16(FFMA chain over s ascending of exp2f(FADD(lse[s], -g)) .
   oacc[s] from +0): `FlashMlaCombine_v1`.  The kernel's MAX_SPLITS instance (32 to 160, by num_sm_parts) changes no word.
5. The schedule (`get_mla_metadata_kernel`, get_decoding_sched_meta.cu:12-104; num_sm_parts = max(132 / s_q / (h_q / 64), 1)
   = 132, 5 fixed-overhead blocks, block 64, api/sparse_decode.h:62-68): each request's blocks are cut into contiguous splits,
   SPLITS (their boundaries); `kernels.flashmla_twin.sched_meta` is the planner, `request_splits` a request's SPLITS.

The CUDA libm pieces, as the SASS has them (non-fast-math; a NaN operand gives 0x7FFFFFFF throughout):
    exp2f(x)           c = setp.geu(x, -126); e = MUFU.EX2(c ? x : x * 0.5); c ? e : e * e            `NvExp2f_v1`
    __fdividef(1, x)   p = setp.geu(|x|, 2^-126); MUFU.RCP(p ? x : x * 2^24) * (p ? 1 : 2^24)         `NvFdividefRcp_v1`
    log2f(x)           `NvLog2f_v1`: the prescale below 2^-126 (x 2^23, exponent -23), `dsa`'s polynomial on
                       m = x - e, the setp.ge.u32 x >= 0x7F800000 arm FFMA(x, +inf, +inf) and x == 0 -> -inf

Registered functions (all `Name_v1`):
  F32Geu, U32Ge, E8m0ToBf16                          primitives: setp.geu.f32, setp.ge.u32, __nv_cvt_e8m0_to_bf16raw
  NvExp2f, NvLog2f, NvFdividefRcp                    (x: f32) -> f32
  FlashMlaRowValid                                   (idx: i32, len: i32, pos: i32) -> bit
  FlashMlaKvRow{DN,DR,QT}                            (nope: DN e4m3, scale: DN/QT e8m0, rope: DR bf16, valid: bit) -> DN+DR bf16
  FlashMlaBlock{D,BN,FIRST,DOT,SCALE}                (q: D bf16, k: BN rows[, m, l[4], o[D]]) -> (m, l[4], o[D])
  FlashMlaSplit{NB,D,BN,PARTIAL,DOT,SCALE}           (q, k: NB*BN rows[, sink]) -> D bf16 | PARTIAL: (oacc: D f32, lse: f32)
  FlashMlaCombine{NS,D}                              (lse: NS f32, oacc: NS x D f32, sink: f32) -> D bf16
  FlashMlaDecodeHead{SPLITS,D,BN,DOT,SCALE}          (q: D bf16, k: N rows, sink: f32) -> D bf16
  FlashMlaDecode{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE} (q: NH x D bf16, nope, scale, rope, idx: the N rows' cache bytes and
                                                     indices, swa_len[, extra_len]: i32, sink: NH f32) -> NH x D bf16

D = DN + DR; a row is a staged row (D bf16, valid bit); N = SPLITS[-1] * BN (the request's rows in kernel order), SPLITS =
(0, b1, ..., NB) the block boundaries of its splits, NW the window rows (the extra rows, and `extra_len`, only when N > NW).
`FLASHMLA_DSV4` holds DeepSeek-V4-Pro's statics.
"""
from __future__ import annotations

import struct

import numpy as np

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.fp32 import Bf16MulBf16, F32AddV2, F32FmaV2, F32MulV2, GuardNegInfZero
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.mufu import MufuEx2Ftz, MufuRcpFtz
from verity.ml.circuits.ops.prims import E4m3ToF32, F2fpBf16, HopperBF16WgmmaDot16, const
from verity.ml.circuits.ops.scalar import (BitAnd, BitNot, F32Eq, F32Fabs, F32Fmaxf, F32Neg, I32Add, I32Eq, I32Le, SelectBf16,
                                           SelectF32)
from verity_vllm.program.kernels import flashmla_twin
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry.b1 import _lane
from verity_vllm.program.registry.fp8 import E4M3
from verity_vllm.program.registry.prims import BF16, BIT, F32, I32
from verity_vllm.program.registry.topp_word_gates import I32ToF32

M32 = 0xFFFFFFFF
E8M0 = Value(8)

_CONF = ("model-untested: read from FlashMLA 6bc49418's Hopper sparse_fp8 decode, its scheduler and combine (the module docstring's "
         "files) and the decode kernel's SASS; tests/properties/dsv4_flashmla_capture_gpu.py is its hardware check")
_PTX = "model-untested: as the PTX ISA defines it; tests/properties/dsv4_flashmla_capture_gpu.py is its hardware check"

F_NEG126, F_HALF, F_2M126, F_2P24, F_2P23, F_ONE = 0xC2FC0000, 0x3F000000, 0x00800000, 0x4B800000, 0x4B000000, 0x3F800000
F_2M23, F_NEG23, F_NEG1, F_PINF, F_NINF = 0x34000000, 0xC1B80000, 0xBF800000, 0x7F800000, 0xFF800000
#: `MAX_INIT_VAL` (splitkv_mla.cuh:22), the running max at a split's start
F_MAX_INIT = int(np.float32(-1e30).view(np.uint32))
L2E = D.LOG2F_LOG2E                                          # CUDART_L2E_F
#: FlashMLA's `LOG_2_E` (api/common.h:31)
FLASHMLA_LOG2E = 1.44269504

#: DeepSeek-V4-Pro at TP8 on the H200: 16 of the 64 padded heads per rank, NoPE 448 + RoPE 64, one scale per 64 NoPE elements,
#: 64-row blocks, softmax_scale 512^-0.5
FLASHMLA_DSV4 = {"NH": 16, "DN": 448, "DR": 64, "QT": 64, "BN": 64, "DOT": HopperBF16WgmmaDot16, "SCALE": 512 ** -0.5}


def scale_log2_bits(scale: float) -> int:
    """`sm_scale * LOG_2_E` in double (`LOG_2_E` the float 1.44269504f), narrowed to f32 (api/sparse_decode.h:414)."""
    return int(np.float32(float(scale) * float(np.float32(FLASHMLA_LOG2E))).view(np.uint32))


def _f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


def _nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


@primitive("F32Geu", 1, [("a", F32), ("b", F32)], BIT, conformance=_PTX)
def F32Geu(a: int, b: int) -> int:
    """`setp.geu.f32`: a >= b, or either a NaN (unordered); +0 and -0 equal, subnormals as they are."""
    return 1 if _nan(a) or _nan(b) or _f32(a) >= _f32(b) else 0


@primitive("U32Ge", 1, [("a", I32), ("b", I32)], BIT, conformance=_PTX)
def U32Ge(a: int, b: int) -> int:
    """`setp.ge.u32`: a >= b as unsigned 32-bit integers."""
    return 1 if (a & M32) >= (b & M32) else 0


def e8m0_to_bf16(x: int) -> int:
    """`__nv_cvt_e8m0_to_bf16raw` (cuda_fp8.hpp; no instruction on sm_90): 0xFF (the UE8M0 NaN) -> 0x7FFF, 0 (2^-127) -> the
    subnormal 0x0040, else the bf16 2^(x - 127), x << 7."""
    x &= 0xFF
    return 0x7FFF if x == 0xFF else 0x0040 if x == 0 else x << 7


@primitive("E8m0ToBf16", 1, [("x", E8M0)], BF16, conformance=_CONF)
def E8m0ToBf16(x: int) -> int:
    """A UE8M0 scale byte as the bf16 the decode multiplies by (`e8m0_to_bf16`, splitkv_mla.cuh:560)."""
    return e8m0_to_bf16(x)


FLASHMLA_PRIMITIVES = [F32Geu, U32Ge, E8m0ToBf16]


class Types:
    """The leaf types of a body's values: words here, `boolean_flashmla`'s bit arrays there."""

    def __init__(self, bf16, f32, bit, e4m3, e8m0, i32):
        self.bf16, self.f32, self.bit, self.e4m3, self.e8m0, self.i32 = bf16, f32, bit, e4m3, e8m0, i32


WORD_T = Types(BF16, F32, BIT, E4M3, E8M0, I32)


class Ops(D.Ops):
    """`dsa.Ops` with ``half(bits)``, a constant bf16 (each once), and ``T`` the leaf types, ``leaf`` the leaves of one bf16 (1 on
    words, 16 on bits)."""

    def __init__(self, B, defs: dict, word, neg, half, T: Types, leaf: int):
        super().__init__(B, defs, word, neg)
        self._half, self._halves, self.T, self.leaf = half, {}, T, leaf

    def half(self, bits: int):
        if bits not in self._halves:
            self._halves[bits] = self._half(bits)
        return self._halves[bits]


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x), lambda bits: B.call(const(16, bits)),
               WORD_T, 1)


# ---- the CUDA libm pieces ---------------------------------------------------------------------------------------------------

def exp2f(o: Ops, x):
    """`NvExp2f_v1`'s body."""
    c = o.geu(x, o.word(F_NEG126))
    e = o.ex2(o.sel(c, x, o.mul(x, o.word(F_HALF))))
    return o.sel(c, e, o.mul(e, e))


def log2f(o: Ops, x):
    """`NvLog2f_v1`'s body."""
    p0 = o.geu(x, o.word(F_2M126))
    xs = o.sel(p0, x, o.mul(x, o.word(F_2P23)))
    e = o.iand(o.iadd(xs, o.word(D.LOG2F_OFFSET)), o.word(D.LOG2F_MASK))
    f = o.add(o.isub(xs, e), o.word(F_NEG1))
    p = o.fma(f, o.word(D.LOG2F_POLY[0]), o.word(D.LOG2F_POLY[1]))
    for c in D.LOG2F_POLY[2:]:
        p = o.fma(f, p, o.word(c))
    ef = o.fma(o.i2f(e), o.word(F_2M23), o.sel(p0, o.word(0), o.word(F_NEG23)))
    y = o.add(ef, o.fma(f, o.word(L2E), o.mul(f, o.mul(f, p))))
    y = o.sel(o.uge(xs, o.word(F_PINF)), o.fma(xs, o.word(F_PINF), o.word(F_PINF)), y)
    return o.sel(o.eq(xs, o.word(0)), o.word(F_NINF), y)


def fdividef_rcp(o: Ops, x):
    """`NvFdividefRcp_v1`'s body."""
    p = o.geu(o.fabs(x), o.word(F_2M126))
    r = o.rcp(o.sel(p, x, o.mul(x, o.word(F_2P24))))
    return o.mul(r, o.sel(p, o.word(F_ONE), o.word(F_2P24)))


_X = lambda S: ((("x", F32),), F32)  # noqa: E731


@composite("NvExp2f", 1, [], _X, conformance=_CONF)
def NvExp2f(B, S, x):
    """CUDA `exp2f` (non-fast-math) as the SASS has it: below -126 (setp.geu, so a NaN takes MUFU directly) MUFU.EX2 of x / 2,
    squared; else MUFU.EX2 of x."""
    return exp2f(word_ops(B), x)


@composite("NvLog2f", 1, [], _X, conformance=_CONF)
def NvLog2f(B, S, x):
    """CUDA `log2f` (non-fast-math) as the SASS has it: `dsa.ue8m0_scale`'s inlined `__nv_log2f` with both branches: the prescale
    of x below 2^-126 (setp.geu: negatives, zeros, subnormals; x 2^23, exponent addend -23), x >= 0x7F800000 unsigned (+inf, NaN
    and every negative) FFMA(x, +inf, +inf), and x == 0 -> -inf."""
    return log2f(word_ops(B), x)


@composite("NvFdividefRcp", 1, [], _X, conformance=_CONF)
def NvFdividefRcp(B, S, x):
    """CUDA `__fdividef(1.0f, x)` as the SASS has it: below 2^-126 in magnitude (setp.geu on |x|) MUFU.RCP of x 2^24, times 2^24;
    else MUFU.RCP of x."""
    return fdividef_rcp(word_ops(B), x)


# ---- the rows ---------------------------------------------------------------------------------------------------------------

@composite("FlashMlaRowValid", 1, [], lambda S: ((("idx", I32), ("len", I32), ("pos", I32)), BIT), conformance=_CONF)
def FlashMlaRowValid(B, S, idx, n, pos):
    """The row at position pos of a segment (window or extra) whose length is len is valid iff its index is not -1 and
    pos < len, signed (`is_index_valid`, splitkv_mla.cuh:630-636)."""
    o = word_ops(B)
    return o.band(o.bnot(o.ieq(idx, o.word(M32))), o.bnot(o.ile(n, pos)))


def _staged(D_: int, T: Types):
    """A staged row: its D bf16 elements and its validity bit."""
    return Tuple(Array(D_, T.bf16), T.bit)


def check_row_statics(DN: int, DR: int, QT: int) -> None:
    if DN < 1 or DR < 0 or QT < 1 or DN % QT:
        raise ValueError(f"FlashMlaKvRow: QT={QT} must divide DN={DN} (one scale per QT NoPE elements), DR={DR} >= 0")


def kv_row(o: Ops, S, nope, scale, rope, valid):
    """`FlashMlaKvRow_v1`'s body: an invalid row is the zero row, as the kernel's zeroed bytes dequantise to +0."""
    DN, DR, QT = S.DN, S.DR, S.QT
    check_row_statics(DN, DR, QT)
    z = o.half(0)
    sc = [o.e8m0(scale[j]) for j in range(DN // QT)]
    x = [o.bsel(valid, o.bmul(o.cvt(o.e4m3(nope[d])), sc[d // QT]), z) for d in range(DN)]
    return array_of(x + [o.bsel(valid, rope[d], z) for d in range(DR)])


def _row_type(S, T: Types = WORD_T):
    return ((("nope", Array(S.DN, T.e4m3)), ("scale", Array(S.DN // S.QT, T.e8m0)), ("rope", Array(S.DR, T.bf16)),
             ("valid", T.bit)), Array(S.DN + S.DR, T.bf16))


@composite("FlashMlaKvRow", 1, ["DN", "DR", "QT"], _row_type, conformance=_CONF)
def FlashMlaKvRow(B, S, nope, scale, rope, valid):
    """One cache row as the decode stages it (splitkv_mla.cuh:538-623): NoPE element d `Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])),
    E8m0ToBf16(scale[d / QT]))` (dequant.h:21-34), then the RoPE elements as stored; every element +0 when the row is invalid."""
    return kv_row(word_ops(B), S, nope, scale, rope, valid)


# ---- one block, one split -------------------------------------------------------------------------------------------------------

def check_block_statics(D_: int, BN: int) -> None:
    if D_ % 16 or BN % 16 or BN < 16:
        raise ValueError(f"FlashMlaBlock: D={D_} and BN={BN} must be multiples of 16")


def block(o: Ops, S, dot, q, k, m_prev=None, l_prev=None, o_prev=None):
    """`FlashMlaBlock_v1`'s body over ``o``'s steps, ``dot`` the score chain (`DotBf16{K = D, DOT}`)."""
    B, D_, BN, FIRST, T, L = o.B, S.D, S.BN, S.FIRST, o.T, o.leaf
    check_block_statics(D_, BN)
    RS = D_ * L + 1                               # a staged row's leaves: its elements', then its validity bit
    scale = o.word(scale_log2_bits(S.SCALE))
    ninf = o.word(F_NINF)
    s = [o.sel(k[c][1], B.call(dot, q, k[c][0]), ninf) for c in range(BN)]
    bmax = ninf
    for c in range(BN):
        bmax = o.fmax(bmax, s[c])
    m_old = o.word(F_MAX_INIT) if FIRST else m_prev
    m_new = o.fmax(o.mul(bmax, scale), m_old)
    nm = o.neg(m_new)
    # From the first block's constant state, r = exp2f(-1e30 - m') is finite in [0, 1] (m' >= -1e30), and a lane's sum of
    # exp2f words is never -0: FFMA(0, r, sum) is sum and FMUL(0, r) is +0 for every input, so that block takes no r.
    r = None if FIRST else o.exp2(o.add(m_old, nm))
    p = [o.exp2(o.fma(s[c], scale, nm)) for c in range(BN)]
    lanes = []
    for t in range(4):
        cols = [c for c in range(BN) if _lane(c) == t]
        acc = p[cols[0]]
        for c in cols[1:]:
            acc = o.add(acc, p[c])
        lanes.append(acc if FIRST else o.fma(l_prev[t], r, acc))
    pb = [o.cvt(x) for x in p]
    out = [o.word(0)] * D_ if FIRST else list(B.batch(o._defs["mul"], o_prev, r, axes=(0, None)))
    for ks in range(BN // 16):
        k0 = 16 * ks
        prow = Coll(Array(16, T.bf16), concat([pb[k0 + j].refs for j in range(16)]))
        vt = Coll(Array(D_, Array(16, T.bf16)), concat([strided_view(k, k0 * RS + d * L, L, RS, 1, 16 * L) for d in range(D_)]))
        out = list(B.batch(S.DOT, array_of(out), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(out))


def _block_type(S, T: Types = WORD_T):
    state = () if S.FIRST else (("m_prev", T.f32), ("l_prev", Array(4, T.f32)), ("o_prev", Array(S.D, T.f32)))
    return ((("q", Array(S.D, T.bf16)), ("k", Array(S.BN, _staged(S.D, T)))) + state,
            Tuple(T.f32, Array(4, T.f32), Array(S.D, T.f32)))


@composite("FlashMlaBlock", 1, ["D", "BN", "FIRST", "DOT", "SCALE"], _block_type, conformance=_CONF)
def FlashMlaBlock(B, S, q, k, m_prev=None, l_prev=None, o_prev=None):
    """One BN-row block of the decode's online softmax for one head (`scale_softmax`, splitkv_mla.cuh:32-84, and the two P V
    wgmmas, :253-258 and :398-404): the module docstring's step 2.  The first block of a split (FIRST) starts from the constant
    state m = -1e30, l = 0, o = 0 and takes no state."""
    return block(word_ops(B), S, bind(DotBf16V2, K=S.D, DOT=S.DOT), q, k, m_prev, l_prev, o_prev)


def check_split_statics(NB: int) -> None:
    if NB < 1:
        raise ValueError(f"FlashMlaSplit: NB={NB} blocks")


def split(o: Ops, S, block_of, q, k, sink=None):
    """`FlashMlaSplit_v1`'s body, ``block_of(first)`` the bound block."""
    B, NB, BN = o.B, S.NB, S.BN
    check_split_statics(NB)
    m = l = acc = None
    for b in range(NB):
        kb = k[b * BN:(b + 1) * BN]
        res = B.call(block_of(True), q, kb) if b == 0 else B.call(block_of(False), q, kb, m, l, acc)
        m, l, acc = res[0], res[1], res[2]
    total = o.add(o.add(l[0], l[1]), o.add(l[2], l[3]))
    empty = o.eq(total, o.word(0))
    mul = o._defs["mul"]
    if not S.PARTIAL:
        den = o.add(total, o.exp2(o.add(o.mul(sink, o.word(L2E)), o.neg(m))))
        scale = o.sel(empty, o.word(0), o.rcp1(den))
        return B.batch(o._defs["cvt"], B.batch(mul, acc, scale, axes=(0, None)), axes=(0,))
    scale = o.sel(empty, o.word(0), o.rcp1(total))
    lse = o.sel(empty, o.word(F_NINF), o.add(o.log2(total), m))
    return tuple_of(B.batch(mul, acc, scale, axes=(0, None)), lse)


def _split_type(S, T: Types = WORD_T):
    N = S.NB * S.BN
    args = (("q", Array(S.D, T.bf16)), ("k", Array(N, _staged(S.D, T))))
    if S.PARTIAL:
        return args, Tuple(Array(S.D, T.f32), T.f32)
    return args + (("sink", T.f32),), Array(S.D, T.bf16)


@composite("FlashMlaSplit", 1, ["NB", "D", "BN", "PARTIAL", "DOT", "SCALE"], _split_type, conformance=_CONF)
def FlashMlaSplit(B, S, q, k, sink=None):
    """One split of one head: NB `FlashMlaBlock_v1`s in ascending block order from the constant state, then the module docstring's
    step 3: the bf16 head output with the sink folded in, or (PARTIAL, one of several splits) the f32 partial and its lse."""
    def block_of(first):
        return bind(FlashMlaBlock, D=S.D, BN=S.BN, FIRST=first, DOT=S.DOT, SCALE=S.SCALE)
    return split(word_ops(B), S, block_of, q, k, sink)


# ---- the combine and the decode ---------------------------------------------------------------------------------------------

def check_combine_statics(NS: int) -> None:
    if NS < 2:
        raise ValueError(f"FlashMlaCombine: NS={NS} splits (one split is written by the decode itself)")


def combine(o: Ops, S, lse, oacc, sink):
    """`FlashMlaCombine_v1`'s body."""
    B, NS = o.B, S.NS
    check_combine_statics(NS)
    mx = o.word(F_NINF)
    for s in range(NS):
        mx = o.fmax(mx, lse[s])
    mx = o.guard(mx)
    nmx = o.neg(mx)
    e = [o.exp2(o.add(lse[s], nmx)) for s in range(NS)]
    lanes = []
    for ln in range(32):
        acc = None
        for s in range(ln, NS, 32):
            acc = e[s] if acc is None else o.add(acc, e[s])
        lanes.append(acc)
    total = D.halving_tree(lanes, lambda a, b: a if b is None else b if a is None else o.add(a, b))
    g = o.sel(o.eq(total, o.word(0)), o.word(F_PINF), o.add(o.log2(total), mx))
    sl2e = o.word(L2E)
    g_sink = o.add(g, o.log2(o.add(o.word(F_ONE), o.exp2(o.fma(sink, sl2e, o.neg(g))))))
    g_none = o.sel(o.eq(sink, o.word(F_NINF)), o.word(F_PINF), o.mul(sink, sl2e))
    g = o.sel(o.eq(g, o.word(F_PINF)), g_none, g_sink)
    ng = o.neg(g)
    fma = o._defs["fma"]
    res = o.word(0)
    for s in range(NS):
        w = o.exp2(o.add(lse[s], ng))
        res = B.batch(fma, w, oacc[s], res, axes=(None, 0, None if s == 0 else 0))
    return B.batch(o._defs["cvt"], res, axes=(0,))


def _combine_type(S, T: Types = WORD_T):
    return ((("lse", Array(S.NS, T.f32)), ("oacc", Array(S.NS, Array(S.D, T.f32))), ("sink", T.f32)), Array(S.D, T.bf16))


@composite("FlashMlaCombine", 1, ["NS", "D"], _combine_type, conformance=_CONF)
def FlashMlaCombine(B, S, lse, oacc, sink):
    """`flash_fwd_mla_combine_kernel` for one head of a request of NS splits (combine.cu:19-160): the module docstring's step 4."""
    return combine(word_ops(B), S, lse, oacc, sink)


def check_splits(SPLITS) -> int:
    """The number of blocks of a request whose splits SPLITS bound: (0, b1, ..., NB), strictly increasing."""
    sp = tuple(SPLITS)
    if len(sp) < 2 or sp[0] != 0 or any(b <= a for a, b in zip(sp, sp[1:])):
        raise ValueError(f"FlashMlaDecodeHead: SPLITS={sp} must be (0, ..., NB), strictly increasing")
    return sp[-1]


def decode_head(o: Ops, S, split_of, combine_of, q, k, sink):
    """`FlashMlaDecodeHead_v1`'s body, ``split_of(nb, partial)`` the bound split, ``combine_of(ns)`` the bound combine."""
    B, BN, sp = o.B, S.BN, tuple(S.SPLITS)
    check_splits(sp)
    if len(sp) == 2:
        return B.call(split_of(sp[1], False), q, k, sink)
    parts = [B.call(split_of(b - a, True), q, k[a * BN:b * BN]) for a, b in zip(sp, sp[1:])]
    oacc = Coll(Array(len(parts), Array(S.D, o.T.f32)), concat([p[0].refs for p in parts]))
    return B.call(combine_of(len(parts)), array_of([p[1] for p in parts]), oacc, sink)


def _head_type(S, T: Types = WORD_T):
    N = check_splits(S.SPLITS) * S.BN
    return ((("q", Array(S.D, T.bf16)), ("k", Array(N, _staged(S.D, T))), ("sink", T.f32)), Array(S.D, T.bf16))


@composite("FlashMlaDecodeHead", 1, ["SPLITS", "D", "BN", "DOT", "SCALE"], _head_type, conformance=_CONF)
def FlashMlaDecodeHead(B, S, q, k, sink):
    """One head of one decode token over its N staged rows: one `FlashMlaSplit_v1` writing bf16, or one partial split per span
    of SPLITS and `FlashMlaCombine_v1` over them."""
    def split_of(nb, partial):
        return bind(FlashMlaSplit, NB=nb, D=S.D, BN=S.BN, PARTIAL=partial, DOT=S.DOT, SCALE=S.SCALE)
    return decode_head(word_ops(B), S, split_of, lambda ns: bind(FlashMlaCombine, NS=ns, D=S.D), q, k, sink)


def check_decode_statics(N: int, NW: int, BN: int) -> None:
    if NW % BN or not BN <= NW <= N:
        raise ValueError(f"FlashMlaDecode: the window rows NW={NW} must be a multiple of BN={BN} in [BN, N={N}]")


def decode(o: Ops, S, row, valid_of, head, q, nope, scale, rope, idx, lens, sink):
    """`FlashMlaDecode_v1`'s body: each row's validity (window rows against ``lens[0]``, extra rows against ``lens[1]`` from
    position 0) and its staging once for every head."""
    B, NW, T = o.B, S.NW, o.T
    N = check_splits(S.SPLITS) * S.BN
    check_decode_statics(N, NW, S.BN)
    valid = [B.call(valid_of, idx[r], lens[0], o.word(r)) if r < NW else B.call(valid_of, idx[r], lens[1], o.word(r - NW))
             for r in range(N)]
    rows = B.batch(row, nope, scale, rope, Coll(Array(N, T.bit), concat([v.refs for v in valid])), axes=(0, 0, 0, 0))
    staged = Coll(Array(N, _staged(S.DN + S.DR, T)), concat([x for r in range(N) for x in (rows[r].refs, valid[r].refs)]))
    return B.batch(head, q, staged, sink, axes=(0, None, 0))


def _decode_type(S, T: Types = WORD_T):
    N, Dm = check_splits(S.SPLITS) * S.BN, S.DN + S.DR
    lens = (("swa_len", T.i32),) + ((("extra_len", T.i32),) if N > S.NW else ())
    return ((("q", Array(S.NH, Array(Dm, T.bf16))), ("nope", Array(N, Array(S.DN, T.e4m3))),
             ("scale", Array(N, Array(S.DN // S.QT, T.e8m0))), ("rope", Array(N, Array(S.DR, T.bf16))), ("idx", Array(N, T.i32)))
            + lens + (("sink", Array(S.NH, T.f32)),), Array(S.NH, Array(Dm, T.bf16)))


@composite("FlashMlaDecode", 1, ["NH", "SPLITS", "NW", "DN", "DR", "QT", "BN", "DOT", "SCALE"], _decode_type, conformance=_CONF)
def FlashMlaDecode(B, S, q, nope, scale, rope, idx, *rest):
    """One decode token's NH heads over its N cache rows in kernel order (the window's NW, then the extra rows), given each row's
    cache bytes and index: each row `FlashMlaRowValid_v1` and `FlashMlaKvRow_v1` once, then head h `FlashMlaDecodeHead_v1` over
    q[h] and sink[h]."""
    row = bind(FlashMlaKvRow, DN=S.DN, DR=S.DR, QT=S.QT)
    head = bind(FlashMlaDecodeHead, SPLITS=tuple(S.SPLITS), D=S.DN + S.DR, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return decode(word_ops(B), S, row, FlashMlaRowValid, head, q, nope, scale, rope, idx, rest[:-1], rest[-1])


WORD = dict(add=F32AddV2, mul=F32MulV2, fma=F32FmaV2, fmax=F32Fmaxf, eq=F32Eq, sel=SelectF32, bsel=SelectBf16, guard=GuardNegInfZero,
            ex2=MufuEx2Ftz, rcp=MufuRcpFtz, fabs=F32Fabs, geu=F32Geu, uge=U32Ge, iadd=I32Add, iand=D.U32And, isub=D.I32Sub,
            i2f=I32ToF32, ieq=I32Eq, ile=I32Le, bnot=BitNot, band=BitAnd, cvt=F2fpBf16, e4m3=E4m3ToF32, e8m0=E8m0ToBf16,
            bmul=Bf16MulBf16, exp2=NvExp2f, log2=NvLog2f, rcp1=NvFdividefRcp)

FLASHMLA_COMPOSITES = [NvExp2f, NvLog2f, NvFdividefRcp, FlashMlaRowValid, FlashMlaKvRow, FlashMlaBlock, FlashMlaSplit,
                       FlashMlaCombine, FlashMlaDecodeHead, FlashMlaDecode]

flashmla_twin.register_twin({d.id: d for d in FLASHMLA_COMPOSITES})
