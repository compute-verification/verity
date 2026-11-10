"""[GLM-5.3] vLLM's `merge_attn_states` as the H200 runs it: the merge of a dense-MHA prefill's two partial attentions, the one over
the context (the prefix, `MLACommonImpl._compute_prefill_context`) and the one over the chunk's own rows (the suffix), each with
its log-sum-exp (`MLACommonImpl.forward_mha`, vllm/model_executor/layers/attention/mla_attention.py).  The kernel is vLLM
d9105ea80's `merge_attn_states_kernel<__nv_bfloat16, __nv_bfloat16, 128, false>` (csrc/libtorch_stable/attention/
merge_attn_states.cu, built into `_C_stable_libtorch` without `--use_fast_math`); "the SASS" is that instance's sm_90 SASS in the
wheel.  `forward_mha` passes no `prefill_tokens_with_context` and no `output_lse`, so every token takes the merge path and the
kernel's `logf` never runs.

Per token and head, with p_lse and s_lse the prefix's and the suffix's lse (f32) and p, s their outputs (D bf16 each):
    p_lse, s_lse = (|x| == inf ? -inf : x)       FSETP.NEU |x|, +INF; FSEL (`std::isinf`)
    m = fmaxf(p_lse, s_lse)                       FMNMX (`F32Fmaxf_v1`)
    |m| == inf: out = p                           the prefix's words, copied (`std::isinf(max_lse)`)
    else:
      p_se = expf(p_lse - m), s_se = expf(s_lse - m)   FADD, then libdevice `__nv_expf` (`NvExpf_v1`'s sequence: FFMA.SAT,
                                                       FFMA.RM, FADD, two FFMAs, MUFU.EX2, FMUL)
      tot = p_se + s_se                                FADD
      ps = p_se / tot, ss = s_se / tot                 `div.rn.f32` (MUFU.RCP, two Newton FFMAs, FCHK and the slow path)
      out[d] = bf16(fma(p[d], ps, s[d] * ss))          FMUL of the suffix's term, FFMA of the prefix's, F2FP.BF16.F32.PACK_AB
The arithmetic is IEEE round-to-nearest without flush-to-zero (no fast-math), and every NaN result is the GPU's 0x7FFFFFFF (the
`_v2` FP32 primitives).  A token without context reaches the kernel as p = +0 with p_lse = -inf (`init_mla_context_partial`), so
ps = +0, ss = 1 and out = s word for word, but for -0, which comes back +0 (+0 + -0), and a NaN, which comes back 0x7FFF.

Registered functions (both `_v1`):
  MergeAttnStatesHead{D}    (p: D bf16, p_lse: f32, s: D bf16, s_lse: f32) -> D bf16
  MergeAttnStates{NH, D}    (p: NH x D bf16, p_lse: NH f32, s: NH x D bf16, s_lse: NH f32) -> NH x D bf16, one token's heads
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import F32Eq, F32Fabs, F32Fmaxf
from verity.circuits.types import Array
from verity.definitions.registry import prims as P
from verity.definitions.registry.moe import NvExpf
from verity.definitions.registry.prims import BF16, F32

INF, NEG_INF = 0x7F800000, 0xFF800000
#: bf16 elements per 128-bit pack: the kernel requires the head size to be a multiple of it
PACK = 8

_CONF = ("model-untested: read from the SASS of vLLM d9105ea80's merge_attn_states_kernel<__nv_bfloat16, __nv_bfloat16, 128, "
         "false> at cc 9.0 (the module docstring) and held word for word to that reading's CPU twin "
         "(verity/definitions/tests/test_merge_attn_states.py); the check that would make it exact is the kernel's own words at "
         "cc 9.0 for that test's edge and random inputs")


def _refuse(statics: dict) -> str | None:
    NH, D = statics.get("NH", 1), statics.get("D", PACK)
    if not (isinstance(NH, int) and NH >= 1):
        return f"NH={NH!r}: a token has at least one head"
    if not (isinstance(D, int) and D >= PACK and D % PACK == 0):
        return f"D={D!r}: the kernel takes D a positive multiple of {PACK} (one 128-bit bf16 pack a thread)"
    return None


def _head_type(S, bf16=BF16, f32=F32):
    return ((("p", Array(S.D, bf16)), ("p_lse", f32), ("s", Array(S.D, bf16)), ("s_lse", f32)), Array(S.D, bf16))


def _type(S, bf16=BF16, f32=F32):
    return ((("p", Array(S.NH, Array(S.D, bf16))), ("p_lse", Array(S.NH, f32)), ("s", Array(S.NH, Array(S.D, bf16))),
             ("s_lse", Array(S.NH, f32))), Array(S.NH, Array(S.D, bf16)))


@composite("MergeAttnStatesHead", 1, ["D"], _head_type, conformance=_CONF, refuse=_refuse)
def MergeAttnStatesHead(B, S, p, p_lse, s, s_lse):
    """One head of one token: the infinite lse words to -inf, m = fmaxf, the prefix's words where |m| is infinite, else
    bf16(fma(p, p_se / tot, s * (s_se / tot))) with the weights from `NvExpf_v1` (the module docstring)."""
    inf, ninf = B.call(const(32, INF)), B.call(const(32, NEG_INF))

    def finite_or_neg_inf(x):
        return B.call(P.SelectF32, B.call(F32Eq, B.call(F32Fabs, x), inf), ninf, x)

    pl, sl = finite_or_neg_inf(p_lse), finite_or_neg_inf(s_lse)
    m = B.call(F32Fmaxf, pl, sl)
    empty = B.call(F32Eq, B.call(F32Fabs, m), inf)
    p_se = B.call(NvExpf, B.call(P.F32SubV2, pl, m))
    s_se = B.call(NvExpf, B.call(P.F32SubV2, sl, m))
    tot = B.call(P.F32AddV2, p_se, s_se)
    ps, ss = B.call(P.F32DivV2, p_se, tot), B.call(P.F32DivV2, s_se, tot)
    s_term = B.batch(P.F32MulV2, B.batch(P.Bf16ToF32, s, axes=(0,)), ss, axes=(0, None))
    merged = B.batch(P.F32FmaV2, B.batch(P.Bf16ToF32, p, axes=(0,)), ps, s_term, axes=(0, None, 0))
    return B.batch(P.SelectBf16, empty, p, B.batch(P.F2fpBf16, merged, axes=(0,)), axes=(None, 0, 0))


@composite("MergeAttnStates", 1, ["NH", "D"], _type, conformance=_CONF, refuse=_refuse)
def MergeAttnStates(B, S, p, p_lse, s, s_lse):
    """One token's NH heads, each `MergeAttnStatesHead_v1{D}` (the kernel's threads of a head never read another head)."""
    return B.batch(bind(MergeAttnStatesHead, D=S.D), p, p_lse, s, s_lse, axes=(0, 0, 0, 0))


MERGE_ATTN_STATES_COMPOSITES = [MergeAttnStatesHead, MergeAttnStates]

__all__ = ["MERGE_ATTN_STATES_COMPOSITES", "MergeAttnStates", "MergeAttnStatesHead"]
