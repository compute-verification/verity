"""DeepSeek-V2 MLA's decode-side kernels as vLLM d9105ea80 runs them on sm_120 under `VLLM_BATCH_INVARIANT` (GLM-4.7-Flash),
on bits: Triton's MLA decode attention, the absorb `torch.bmm`s, and the routed/shared-expert combine. Every word is its bits
(bf16 `Array<16, Value<1>>`, f32 `Array<32, Value<1>>`); each Definition is Boolean and its arithmetic is `mla_check`'s numpy
model, which equals the hardware on every captured call (`mla_capture`).

    TritonExpSub_v1                       (x, m: F32) -> F32      EXP(x - m): ex2.approx.f32(RN(RN(x - m) * 0x3FB8AA3B))
    F32MulConst_v1{K}                     (x: F32) -> F32         RN(x * K), the f32 word K folded
    TritonMlaScore_v1{K, SCALE, DOT}      (q, k: Array<K, BF16>) -> F32
                                          RN(DotBf16_v3{K, DOT}(q, k) * SCALE): K/16 DOT steps ascending from zero
    TritonMlaBlock_v1{DL, DR, NVIS, BN, SCALE, DOT}
                                          (q: Array<DL+DR, BF16>, kv: Array<NVIS, Array<DL+DR, BF16>>, e_max, e_sum: F32,
                                           acc: Array<DL, F32>) -> (F32, F32, Array<DL, F32>)
    TritonMlaGate_v1                      (e_max, e_sum: F32) -> Value<1>
    TritonMlaOut_v1                       (acc, e_sum, rcp: F32, gate: Value<1>) -> BF16
    TritonMlaDecodeHead_v1{T, DL, DR, BN, SPLITS, SCALE, DOT}
                                          (q: Array<DL+DR, BF16>, kv: Array<T, Array<DL+DR, BF16>>) -> Array<DL, BF16>
    TritonMlaDecode_v1{H, T, DL, DR, BN, SPLITS, SCALE, DOT}
                                          (q: Array<H, Array<DL+DR, BF16>>, kv: Array<T, Array<DL+DR, BF16>>)
                                          -> Array<H, Array<DL, BF16>>
    MlaBmm_v1{NHEADS, K, N, DOT}          (x: Array<NHEADS, Array<K, BF16>>, w: Array<NHEADS, Array<N, Array<K, BF16>>>)
                                          -> Array<NHEADS, Array<N, BF16>>
    MoeSharedCombineElement_v1{RSF}       (routed, shared: BF16) -> BF16
    MoeSharedCombine_v1{H, RSF}           (routed, shared: Array<H, BF16>) -> Array<H, BF16>

`TritonMlaDecode` is `decode_attention_fwd(..., is_mla=True)` as `TritonMLAImpl.forward_mqa` calls it for one query token:
``q = [ql_nope (DL) | q_pe (DR)]``, the request's T cache rows ``kv = [kv_c (DL) | k_pe (DR)]`` in position order (the
block-table gather is a view), the values ``kv[:, 0:DL]``, ``SCALE`` sm_scale's f32 word. Under `VLLM_BATCH_INVARIANT` the
kernel runs one KV split (SPLITS = 1, the only value on bits so far). `_fwd_grouped_kernel_stage1` visits blocks of BN = 32
keys in ascending order from ``(e_max, e_sum, acc) = (-inf, 0, 0)`` (`TritonMlaBlock`):

    s_j   = RN(DOT chain of q . kv_j * SCALE)                     a key past T is -inf, its value row +0
    n     = max.f32(s_0, ..., e_max)                              (with -inf once when NVIS < BN)
    re    = EXP(e_max - n);  p_j = EXP(s_j - n)
    acc   = DOT(DOT(RN(acc * re), bf16(p[0:16]), v[0:16]), bf16(p[16:32]), v[16:32])
    e_sum = fma(e_sum, re, ((w0 + w2) + (w1 + w3)))               w_k = ((c0+c1)+(c4+c5)) + ((c2+c3)+(c6+c7)) over warp k's
                                                                  eight columns
`EXP` is Triton's ``tl.exp``, which ptxas lowers to ``t < -126 ? MUFU.EX2(t * 0.5)^2 : MUFU.EX2(t)`` on ``t = RN(x * log2 e)``
(`ex2_approx`; MUFU.EX2 is core's `MufuEx2Ftz`, measured equal on sm_120). The split writes ``tv = div.full(acc, e_sum)`` and
``tl = e_max + logf(e_sum)`` (libdevice's FTZ `__nv_logf`), and `_fwd_kernel_stage2`'s merge of one split is
``o = cvt.rn.bf16(div.full(fma(1, tv, 0 * 0), 1))`` when tl is finite and NaN (0x7FFF) when it is not, since then ``n``, ``old``
or ``el`` is NaN. tl is finite exactly when e_max is finite and e_sum a positive normal (`TritonMlaGate`; FTZ logf is -inf at a
subnormal, the sum cannot overflow); ``div.full(x, 1) = x`` (MUFU.RCP(1) = 1), so ``o = gate ? F2fp(tv + 0) : 0x7FFF``
(`TritonMlaOut`, which also forms tv from ``rcp = DivFullRcp(e_sum)``, read once per row).

`MlaBmm` is vLLM's batch-invariant `bmm_kernel` (`tl.dot` over K tiles of 64 chained from zero, then cvt.rn.bf16.f32), the q
absorb (``ql_nope = q_nope x W_UK_T``) and the V up-projection (``attn_out x W_UV``): per head `Gemm_v3{K, N, DOT}`.
`MoeSharedCombine` is ``shared_output + fused_output * routed_scaling_factor`` on CUDA bf16 tensors (ATen, F2FP's NaN 0x7FFF):
``bf16(f32(shared) + f32(bf16(f32(routed) * f32(rsf))))``; ``RSF`` is ``f32(routed_scaling_factor)``'s word. At RSF = 1.0, where
vLLM skips the multiply, the multiply is the identity on every word that reaches the output.

DOT is the k16 step: `HopperBF16WgmmaDot16_v2`, the sm_120 mma.sync.m16n8k16 bf16 model (`sm120.mma.m16n8k16.bf16`).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as BS
from verity.ml.circuits.boolean.forms import NOT, and_reduce, mux, or_reduce, const_bits
from verity.ml.circuits.ops.boolean.softmax import f32_gt
from verity.ml.circuits.boolean.trace import definition
from verity.ml.definitions.registry.boolean_attention import _Bits

__all__ = ["F32MulConst", "MlaBmm", "MoeSharedCombine", "MoeSharedCombineElement", "TritonExpSub", "TritonMlaBlock",
           "TritonMlaDecode", "TritonMlaDecodeHead", "TritonMlaGate", "TritonMlaOut", "TritonMlaScore", "GLM47_FLASH"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
A16 = Array(16, BF16)
NEG_INF = 0xFF800000
LOG2E = 0x3FB8AA3B
HALF = 0x3F000000
MINUS_126 = 0xC2FC0000
NAN_F2FP = BG.NAN_F2FP
BLOCK_N = 32

#: GLM-4.7-Flash's bindings at TP1 (20 heads; 5 per rank at TP4): sm_scale = 256^-0.5, routed_scaling_factor 1.8
GLM47_FLASH = {"H": 20, "DL": 512, "DR": 64, "BN": BLOCK_N, "SPLITS": 1, "SCALE": 0x3D800000, "DOT": BG.HopperBF16WgmmaDot16,
               "RSF": 0x3FE66666, "HIDDEN": 2048, "QK_NOPE": 192, "V_HEAD": 256}


def ex2_approx(C, t):
    """``ex2.approx.f32`` (not .ftz) as ptxas 12.9 lowers it on sm_120: ``t < -126`` (ordered) halves t before MUFU.EX2 and
    squares the result (FMUL, subnormals kept)."""
    small = f32_gt(C, fp.f32_const(MINUS_126), t)[0]
    y = BM.mufu_ex2(C, mux(C, small, fp.f32_mul(C, t, fp.f32_const(HALF)), list(t)))
    return mux(C, small, fp.f32_mul(C, y, y), y)


def exp_sub(C, x, m):
    """Triton's ``tl.exp(x - m)`` on f32: ``ex2.approx.f32(RN(RN(x - m) * log2(e)))``."""
    return ex2_approx(C, fp.f32_mul(C, fp.f32_sub(C, x, m), fp.f32_const(LOG2E)))


def merge_gate(C, e_max, e_sum):
    """One split's ``tl = e_max + logf(e_sum)`` is finite: e_max finite and e_sum a positive normal (FTZ logf)."""
    e = e_sum[23:31]
    finite = NOT(and_reduce(C, e_max[23:31]))
    normal = C.AND(NOT(and_reduce(C, e)), or_reduce(C, e))
    return [C.AND(C.AND(finite, normal), NOT(e_sum[31]))]


def mla_out(C, acc, e_sum, rcp, gate):
    """``gate ? F2fp(div.full(acc, e_sum) + 0) : 0x7FFF``, with ``rcp = DivFullRcp(e_sum)``."""
    t = fp.f32_mul(C, BM.div_full_scale_a(C, acc, e_sum), rcp)
    zero = NOT(or_reduce(C, t[:31]))
    t = list(t[:31]) + [C.AND(t[31], NOT(zero))]
    return mux(C, gate[0], fp.f32_to_bf16(C, t, NAN_F2FP), const_bits(NAN_F2FP, 16))


TritonExpSub = definition("TritonExpSub", 1, exp_sub, (("x", F32), ("m", F32)), F32,
                          doc="Triton's tl.exp(x - m): ex2.approx.f32(RN(RN(x - m) * 0x3FB8AA3B)), ptxas's lowering")
TritonMlaGate = definition("TritonMlaGate", 1, merge_gate, (("e_max", F32), ("e_sum", F32)), BL.BIT,
                           doc="one split's e_max + logf(e_sum) is finite: e_max finite, e_sum a positive normal")
TritonMlaOut = definition("TritonMlaOut", 1, mla_out, (("acc", F32), ("e_sum", F32), ("rcp", F32), ("gate", BL.BIT)), BF16,
                          doc="one output word of a one-split Triton MLA decode: gate ? F2fp(div.full(acc, e_sum) + 0) : 0x7FFF")
F32MulConst = EW.specialized("F32MulConst", 1, ["K"], lambda S: lambda C, x: fp.f32_mul(C, x, const_bits(S.K, 32)),
                             lambda S: ((("x", F32),), F32), doc="RN(x * K) for the f32 word K, folded (FMUL, subnormals kept)")


def _score(B, S, q, k):
    """``RN(DotBf16_v3{K, DOT}(q, k) * SCALE)``: one key's scaled score."""
    return B.call(bind(F32MulConst, K=S.SCALE), B.call(bind(BG.DotBf16, K=S.K, DOT=S.DOT), q, k))


TritonMlaScore = CompositeDefinition("TritonMlaScore", 1, ("K", "SCALE", "DOT"),
                                     lambda S: ((("q", Array(S.K, BF16)), ("k", Array(S.K, BF16))), F32), _score,
                                     doc=_score.__doc__, conformance="exact-model-tested")


class _Memo:
    """Calls by callee and operand objects, so a body computes no value twice (the masked columns' sums)."""

    def __init__(self, B):
        self.B, self.done = B, {}

    def __call__(self, fn, *args):
        key = (fn.id, *map(id, args))
        if key not in self.done:
            self.done[key] = (self.B.call(fn, *args), args)
        return self.done[key][0]


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _block_type(S):
    D = S.DL + S.DR
    return ((("q", Array(D, BF16)), ("kv", Array(S.NVIS, Array(D, BF16))), ("e_max", F32), ("e_sum", F32),
             ("acc", Array(S.DL, F32))), Tuple(F32, F32, Array(S.DL, F32)))


def _block(B, S, q, kv, e_max, e_sum, acc):
    """One BN-key block of `_fwd_grouped_kernel_stage1` for one head, NVIS visible keys (the rest masked: score -inf, value
    row +0): ``n = max(scores, e_max)``, ``re = EXP(e_max - n)``, ``p = EXP(s - n)``, ``acc = DOT steps over RN(acc * re)``,
    ``e_sum = fma(e_sum, re, rowsum(p))``; returns ``(n, e_sum, acc)``."""
    DL, D, NVIS, BN = S.DL, S.DL + S.DR, S.NVIS, S.BN
    _check("TritonMlaBlock_v1", BN_is_32=BN == BLOCK_N, NVIS_in_1_BN=1 <= NVIS <= BN, DL_DR_multiples_of_16=DL % 16 == 0 == S.DR % 16)
    c, call = _Bits(B), _Memo(B)
    s = B.batch(bind(TritonMlaScore, K=D, SCALE=S.SCALE, DOT=S.DOT), q, kv, axes=(None, 0))
    m = s[0]
    for j in range(1, NVIS):
        m = B.call(BS.F32Fmaxf, m, s[j])
    ninf = c.word(NEG_INF, 32)
    if NVIS < BN:
        m = B.call(BS.F32Fmaxf, m, ninf)
    n = B.call(BS.F32Fmaxf, m, e_max)
    re = B.call(TritonExpSub, e_max, n)
    p = B.batch(TritonExpSub, s, n, axes=(0, None))
    pm = B.call(TritonExpSub, ninf, n) if NVIS < BN else None
    cols = [p[j] for j in range(NVIS)] + [pm] * (BN - NVIS)

    def add(a, b):
        return call(EW.F32Add, a, b)

    w = []
    for k in range(4):
        x = cols[8 * k:8 * k + 8]
        w.append(add(add(add(x[0], x[1]), add(x[4], x[5])), add(add(x[2], x[3]), add(x[6], x[7]))))
    e_sum = B.call(EW.F32Fma, e_sum, re, add(add(w[0], w[2]), add(w[1], w[3])))
    acc = B.batch(EW.F32Mul, acc, re, axes=(0, None))
    pb = B.batch(BG.F2fpBf16, p, axes=(0,))
    pmb = B.call(BG.F2fpBf16, pm) if pm is not None else None
    zero16 = c.zeros(BF16)
    for g in range(BN // 16):
        k0 = 16 * g
        prow = Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else pmb).refs for j in range(16)]))
        if k0 + 16 <= NVIS:
            vcols = [strided_view(kv, 16 * (k0 * D + d), 16, 16 * D, 1, 256) for d in range(DL)]
        else:
            vcols = [concat([(kv[k0 + j][d] if k0 + j < NVIS else zero16).refs for j in range(16)]) for d in range(DL)]
        acc = B.batch(S.DOT, acc, prow, Coll(Array(DL, A16), concat(vcols)), axes=(0, None, 0))
    return tuple_of(n, e_sum, acc)


TritonMlaBlock = CompositeDefinition("TritonMlaBlock", 1, ("DL", "DR", "NVIS", "BN", "SCALE", "DOT"), _block_type, _block,
                                     doc=_block.__doc__, conformance="exact-model-tested")


def _head_type(S):
    D = S.DL + S.DR
    return ((("q", Array(D, BF16)), ("kv", Array(S.T, Array(D, BF16)))), Array(S.DL, BF16))


def _head(B, S, q, kv):
    """One head of a one-split Triton MLA decode over T keys: ceil(T / BN) `TritonMlaBlock_v1` in ascending order from
    ``(-inf, 0, 0)``, then `TritonMlaOut_v1` on each coordinate with ``rcp = DivFullRcp_v2(e_sum)`` and the gate
    `TritonMlaGate_v1(e_max, e_sum)`."""
    T, BN = S.T, S.BN
    _check("TritonMlaDecodeHead_v1", SPLITS_is_1=S.SPLITS == 1, BN_is_32=BN == BLOCK_N, T_positive=T >= 1)
    c = _Bits(B)
    e_max, e_sum, acc = c.word(NEG_INF, 32), c.zeros(F32), c.zeros(Array(S.DL, F32))
    for lo in range(0, T, BN):
        hi = min(T, lo + BN)
        blk = bind(TritonMlaBlock, DL=S.DL, DR=S.DR, NVIS=hi - lo, BN=BN, SCALE=S.SCALE, DOT=S.DOT)
        st = B.call(blk, q, kv[lo:hi], e_max, e_sum, acc)
        e_max, e_sum, acc = st[0], st[1], st[2]
    rcp = B.call(BM.DivFullRcp, e_sum)
    gate = B.call(TritonMlaGate, e_max, e_sum)
    return B.batch(TritonMlaOut, acc, e_sum, rcp, gate, axes=(0, None, None, None))


TritonMlaDecodeHead = CompositeDefinition("TritonMlaDecodeHead", 1, ("T", "DL", "DR", "BN", "SPLITS", "SCALE", "DOT"), _head_type,
                                          _head, doc=_head.__doc__, conformance="exact-model-tested")


def _decode_type(S):
    D = S.DL + S.DR
    return ((("q", Array(S.H, Array(D, BF16))), ("kv", Array(S.T, Array(D, BF16)))), Array(S.H, Array(S.DL, BF16)))


def _decode(B, S, q, kv):
    """`TritonMLAImpl.forward_mqa`'s attention for one query token: `TritonMlaDecodeHead_v1` on each of the H heads, the
    request's cache rows shared (one latent KV head)."""
    head = bind(TritonMlaDecodeHead, T=S.T, DL=S.DL, DR=S.DR, BN=S.BN, SPLITS=S.SPLITS, SCALE=S.SCALE, DOT=S.DOT)
    return B.batch(head, q, kv, axes=(0, None))


TritonMlaDecode = CompositeDefinition("TritonMlaDecode", 1, ("H", "T", "DL", "DR", "BN", "SPLITS", "SCALE", "DOT"), _decode_type,
                                      _decode, doc=_decode.__doc__, conformance="exact-model-tested")


def _bmm(B, S, x, w):
    """vLLM's batch-invariant `bmm_kernel` for one token row: `Gemm_v3{K, N, DOT}` of ``x[h]`` against ``w[h]`` on each head,
    ``w[h][n][k]`` the right operand at ``[h, k, n]``."""
    return B.batch(bind(BG.Gemm, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))


MlaBmm = CompositeDefinition("MlaBmm", 1, ("NHEADS", "K", "N", "DOT"),
                             lambda S: ((("x", Array(S.NHEADS, Array(S.K, BF16))),
                                         ("w", Array(S.NHEADS, Array(S.N, Array(S.K, BF16))))), Array(S.NHEADS, Array(S.N, BF16))),
                             _bmm, doc=_bmm.__doc__, conformance="exact-model-tested")


def _combine(S):
    def build(C, routed, shared):
        return EW.bf16_add_f2fp(C, shared, EW.bf16_mul_f32(C, routed, const_bits(S.RSF, 32)))
    return build


MoeSharedCombineElement = EW.specialized(
    "MoeSharedCombineElement", 1, ["RSF"], _combine, lambda S: ((("routed", BF16), ("shared", BF16)), BF16),
    doc="bf16(f32(shared) + f32(bf16(f32(routed) * RSF))) for the f32 word RSF, folded; every NaN 0x7FFF")


def _shared_combine(B, S, routed, shared):
    """The routed experts' output scaled and the shared expert's added, ATen on CUDA bf16: `MoeSharedCombineElement_v1{RSF}`
    on each of the H elements."""
    return B.batch(bind(MoeSharedCombineElement, RSF=S.RSF), routed, shared, axes=(0, 0))


MoeSharedCombine = CompositeDefinition("MoeSharedCombine", 1, ("H", "RSF"),
                                       lambda S: ((("routed", Array(S.H, BF16)), ("shared", Array(S.H, BF16))), Array(S.H, BF16)),
                                       _shared_combine, doc=_shared_combine.__doc__, conformance="exact-model-tested")
