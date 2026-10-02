"""The B1 Definitions with a program value as an input rather than a constant: the norms' epsilon and the attention softmax scale.

Each takes the f32 word the kernel reads (an `eps` argument; FA2's `scale_softmax_log2`) as its last parameter, which a
`registered_values` program registers; the body is the constant Definition's (`b1`) word for word otherwise.
"""
from __future__ import annotations

from verity.ir.defs import composite
from verity.ir.types import Array, Tuple
from verity_vllm.program.registry import b1
from verity_vllm.program.registry.prims import BF16, F32


@composite("RMSNormTriton", 3, ["N"],
           lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16)), ("eps", F32)), Array(S.N, BF16)),
           conformance="exact-model-tested: RMSNormTriton_v1's body with eps an input")
def RMSNormTritonV3(B, S, x, w, eps):
    """`RMSNormTriton_v1{N, EPS}` with the epsilon an input (the f32 word the kernel reads, `eps` of `_rms_norm_kernel`) rather than a
    constant of the Definition: the program registers it."""
    return b1.RMSNormTriton.body(B, S, x, w, eps)


@composite("RMSNormFusedCuda", 4, ["N"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16)), ("eps", F32)),
                      Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance="exact-model-tested: RMSNormFusedCuda_v2's body with eps an input")
def RMSNormFusedCudaV4(B, S, x, res, w, eps):
    """`RMSNormFusedCuda_v2{N, EPS}` with the epsilon an input (the f32 word `fused_add_rms_norm` passes the kernel) rather than a
    constant of the Definition: the program registers it."""
    return b1.RMSNormFusedCuda.body(B, S, x, res, w, eps)


# Attention with the softmax scale an input: `scale_log2` is the f32 word FA2's `set_params_fprop` hands the kernel
# (`scale_softmax_log2`, `_scale_log2_bits(D)` on vLLM's `D**-0.5`), which the program registers; the chain is v3's otherwise.


@composite("AttnBlock", 10, ["D", "NVIS", "FIRST", "BN"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)), ("scale_log2", F32))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="exact-model-tested: AttnBlock_v3's body with the scale an input")
def AttnBlockV10(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2):
    """`AttnBlock_v3{D, NVIS, FIRST, BN}` with `scale_log2` an input."""
    return b1.AttnBlockV3.body(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2)


@composite("AttentionHead", 10, ["T", "D", "BN"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16))),
                       ("scale_log2", F32)), Array(S.D, BF16)),
           conformance="exact-model-tested: AttentionHead_v3's body over AttnBlock_v10")
def AttentionHeadV10(B, S, q, k, v, scale_log2):
    """`AttentionHead_v3{T, D, BN}` over `AttnBlock_v10`, every block reading the one `scale_log2`."""
    return b1.AttentionHeadV3.body(B, S, q, k, v, scale_log2, block=AttnBlockV10)


@composite("Attention", 10, ["T", "NH", "KVH", "D", "BN"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))), ("scale_log2", F32)),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="exact-model-tested: Attention_v3's body over AttentionHead_v10")
def AttentionV10(B, S, q, kc, vc, scale_log2):
    """`Attention_v3{T, NH, KVH, D, BN}` over `AttentionHead_v10`, every head reading the one `scale_log2`."""
    return b1.AttentionV3.body(B, S, q, kc, vc, scale_log2, head=AttentionHeadV10)
