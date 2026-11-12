"""[GLM-5.3] The MTP draft layer's own launch, `_fused_eh_norm_kernel`, as one Definition; its sidecar binds the draft step's
other kernels at the statics a draft runs them at.

vLLM d9105ea80's `DeepseekV32MultiTokenPredictorLayer.forward` (`models/deepseek_v32/nvidia/mtp.py`) normalises the token's
embedding with `enorm` and the previous hidden state with `hnorm`, both `RMSNorm(hidden_size, eps=rms_norm_eps)`, in one launch
when the embedding table is sharded (`fused_eh_norm`, `models/deepseek_v32/common/kernels.py`: `_rms_norm` at 25-31, the kernel
and its launch at 987-1036).  One program per token, grid (tokens,), `BLOCK = next_power_of_2(H)` and Triton's default 4 warps:

    pos     tl.load(pos_ptr + tok), the token's int64 position
    e'      tl.where(pos == 0, 0.0, e.to(float32)): the embedding row, every element +0 at position 0
    out     [ _rms_norm(e', enorm.weight) | _rms_norm(p, hnorm.weight) ], 2H bf16 words

"The PTX" is Triton 3.7.1's sm_90 output for the specialisation the H200 runs (every pointer and stride divisible by 16) at H = 6144,
BLOCK = 8192: `.reqntid 128`, `setp.eq.b64` of the whole position against 0 and `selp.f32 0f00000000` on each of e's 64 f32
registers a thread, then each `_rms_norm` as `dsa_norm.TritonBlockRMSNorm_v1{N=6144, BLOCK=8192, WARPS=4}` states it: thread t's
registers in eight runs of 8 elements 1024 apart (`ld.global.v4.b32`), the last two runs past 6144 (`mov.pred %p2, 0`, so their
loads leave +0), `mul.f32 x1*x1` then `fma.rn.f32` over x0, x2, .., x63; each warp's butterfly 16..1 and the four warp sums'
butterfly 2, 1 through shared memory; `div.full.f32` by 6144.0 and `add.f32 eps`, which ptxas contracts to
`FFMA sum, 0x392AAAAB, eps` (`DivFullRcp(6144)`; `DivFullScaleA` leaves every sum unchanged at that divisor) before `MUFU.RSQ`;
`mul.f32 x*r` then `*w`, `F2FP.BF16.F32.PACK_AB`, and no store past 6144.  The two norms are the same program on different rows
and weights.  The SASS selects +0 before the chain (`FSEL R, R, RZ, P0`), so selecting e's bf16 word or +0 before its
`cvt.f32.bf16`, as this body does, gives the same f32 word.

Registered: `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}` (pos: i64, e: H bf16, p: H bf16, ew: H bf16, hw: H bf16) -> 2H bf16, one token's
row of the launch: `MtpEhZero_v1{H}` (pos: i64, e: H bf16) -> H bf16, the position select, then each half's
`TritonBlockRMSNorm_v1`.  A whole row of 6144 is past a Flock proof block, so it is staged as its cut (`CUTS`, `eh_cut`): the
select, each half's scale `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}` (x: N bf16) -> f32, the one word every normed
element reads, and N / `APPLY_G` of `dsa_norm.DsaRMSNormApply_v1{G}` a half, whose `bf16(r * x * f32(w))` is the block norm's
too.  `fused_embed_eh_norm` (`model_executor/layers/fused_embed_norm.py`), which folds the embedding gather in, is not stated: it
needs the whole table on a rank (tensor parallel size 1).
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BitAnd, SelectBf16
from verity.definitions.registry import dsa as D
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry.prims import BF16, F32
from verity.definitions.registry.sampling import I64
from verity.definitions.registry.torch_words import I64Le

_CONF = ("model-untested: read from vLLM d9105ea80's _fused_eh_norm_kernel and the cc 9.0 PTX and SASS Triton 3.7.1 compiles from "
         "it (the module docstring); adapters/vllm/tests/properties/glm53_mtp_eh_norm_capture_gpu.py is its hardware check")
_CUT_CONF = ("model-untested: MtpEhNorm_v1 cut at the select and at each half's scale; the units return its words "
             "(adapters/vllm/tests/program/test_mtp_draft.py)")

#: the columns of a half one apply unit takes, `dsa_norm`'s bound `DsaRMSNormApply_v1` width
APPLY_G = 64


def _sig(S):
    row = Array(S.H, BF16)
    return (("pos", I64), ("e", row), ("p", row), ("ew", row), ("hw", row)), Array(2 * S.H, BF16)


@composite("MtpEhZero", 1, ["H"], lambda S: ((("pos", I64), ("e", Array(S.H, BF16))), Array(S.H, BF16)), conformance=_CUT_CONF)
def MtpEhZero(B, S, pos, e):
    """e with every word +0 where pos is 0, the whole int64 compared (signed `pos <= 0` and `0 <= pos`)."""
    zero = B.call(const(64, 0))
    first = B.call(BitAnd, B.call(I64Le, pos, zero), B.call(I64Le, zero, pos))
    return B.batch(SelectBf16, first, B.call(const(16, 0)), e, axes=(None, None, 0))


@composite("TritonBlockRMSNormSumSq", 1, ["N", "BLOCK", "WARPS", "EPS"], lambda S: ((("x", Array(S.N, BF16)),), F32),
           conformance=_CUT_CONF)
def TritonBlockRMSNormSumSq(B, S, x):
    """`TritonBlockRMSNorm_v1`'s scale on its own: ``r = rsqrt.approx.ftz(fma(DivFullScaleA(sum, N), DivFullRcp(N), eps))`` over
    its masked block, the one f32 word each output element reads."""
    DN.check_block(S.N, S.BLOCK, S.WARPS, S.EPS)
    o = D.word_ops(B)
    return DN.rms_scale(o, S, [o.cvt(x[i]) for i in range(S.N)], DN.WARP * S.WARPS, S.BLOCK, fused_eps=True)


@composite("MtpEhNorm", 1, ["H", "BLOCK", "WARPS", "EPS"], _sig, conformance=_CONF)
def MtpEhNorm(B, S, pos, e, p, ew, hw):
    """One token's row of `_fused_eh_norm_kernel`: `MtpEhZero_v1{H}` of e, normed by `TritonBlockRMSNorm_v1{H, BLOCK, WARPS, EPS}`
    with ew, then p normed by the same Definition with hw."""
    e0 = B.call(bind(MtpEhZero, H=S.H), pos, e)
    norm = bind(DN.TritonBlockRMSNorm, N=S.H, BLOCK=S.BLOCK, WARPS=S.WARPS, EPS=S.EPS)
    en, pn = B.call(norm, e0, ew), B.call(norm, p, hw)
    return Coll(Array(2 * S.H, BF16), concat([en.refs, pn.refs]))


def eh_cut(row) -> list[tuple[object, int]]:
    """The units an `MtpEhNorm_v1` binding ``row`` is staged as, each bound with how many of it: the select, each half's scale and
    its H / `APPLY_G` apply units."""
    if getattr(row, "defn", None) is not MtpEhNorm:
        raise ValueError(f"eh_cut cuts an MtpEhNorm_v1 binding, not {row!r}")
    S = row.bindings
    H = int(S["H"])
    if H % APPLY_G:
        raise ValueError(f"eh_cut: H = {H} is not whole apply units of {APPLY_G}")
    return [(bind(MtpEhZero, H=H), 1),
            (bind(TritonBlockRMSNormSumSq, N=H, BLOCK=S["BLOCK"], WARPS=S["WARPS"], EPS=S["EPS"]), 2),
            (bind(DN.DsaRMSNormApply, G=APPLY_G), 2 * H // APPLY_G)]


CUTS = {"MtpEhNorm_v1": eh_cut}

MTP_DRAFT_COMPOSITES = [MtpEhZero, TritonBlockRMSNormSumSq, MtpEhNorm]
