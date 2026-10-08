"""The Definitions with a program value as an input rather than a constant: the norms' epsilon (the served and reference norms, and
`layer_norm`'s `LayerNormAten_v2`), the attention softmax scale, the attention softcap's two words, and the scalars of the elementwise kinds (a logits
softcap's `1 / C` and `C`, `variance + eps`).

Each takes the f32 word the kernel reads (an `eps` argument; FA2's `scale_softmax_log2`) as its last parameter(s), which a
`registered_values` program registers; the body is the constant Definition's word for word otherwise.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.registry import b1, dense as DN, fa2_softcap as F2S, fa3_check_inf as F3, ref_prims as RP
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


@composite("RMSNormRef", 2, ["N", "DT"],
           lambda S: ((("x", Array(S.N, RP._dt(S.DT))), ("w", Array(S.N, RP._dt(S.DT))), ("eps", F32)), Array(S.N, RP._dt(S.DT))),
           conformance="reference-defined: RMSNormRef_v1's body with eps an input")
def RMSNormRefV2(B, S, x, w, eps):
    """`RMSNormRef_v1{N, EPS, DT}` with the epsilon an input (`f32(eps)`)."""
    return RP.RMSNormRef.body(B, S, x, w, eps)


@composite("LayerNormRef", 2, ["N", "DT"],
           lambda S: ((("x", Array(S.N, RP._dt(S.DT))), ("g", Array(S.N, RP._dt(S.DT))), ("b", Array(S.N, RP._dt(S.DT))), ("eps", F32)),
                      Array(S.N, RP._dt(S.DT))),
           conformance="reference-defined: LayerNormRef_v1's body with eps an input")
def LayerNormRefV2(B, S, x, g, b, eps):
    """`LayerNormRef_v1{N, EPS, DT}` with the epsilon an input (`f32(eps)`)."""
    return RP.LayerNormRef.body(B, S, x, g, b, eps)


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


def _block_sig(S, *values):
    return (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
              ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)), *((n, F32) for n in values))),
            Tuple(F32, Array(4, F32), Array(S.D, F32)))


def _head_sig(S, *values):
    return ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16))),
             *((n, F32) for n in values)), Array(S.D, BF16))


def _attention_sig(S, *values):
    return ((("q", Array(S.NH, Array(S.D, BF16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
             ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))), *((n, F32) for n in values)), Array(S.NH, Array(S.D, BF16)))


# Attention_v2 (a target's k16 step, no per-iteration Check_inf) and Attention_v4 (FA3's per-iteration Check_inf) with `scale_log2` an input.


@composite("AttnBlock", 12, ["D", "NVIS", "FIRST", "BN", "DOT"], lambda S: _block_sig(S, "scale_log2"),
           conformance="exact-model-tested: AttnBlock_v2's body with the scale an input")
def AttnBlockV12(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2):
    """`AttnBlock_v2{D, NVIS, FIRST, BN, DOT}` with `scale_log2` an input."""
    return b1.AttnBlockV2.body(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2)


@composite("AttentionHead", 12, ["T", "D", "BN", "DOT", "INV"], lambda S: _head_sig(S, "scale_log2"),
           conformance="exact-model-tested: AttentionHead_v2's body over AttnBlock_v12")
def AttentionHeadV12(B, S, q, k, v, scale_log2):
    """`AttentionHead_v2{T, D, BN, DOT, INV}` over `AttnBlock_v12`, every block reading the one `scale_log2`."""
    return b1.AttentionHeadV2.body(B, S, q, k, v, scale_log2, block=AttnBlockV12)


@composite("Attention", 12, ["T", "NH", "KVH", "D", "BN", "DOT", "INV"], lambda S: _attention_sig(S, "scale_log2"),
           conformance="exact-model-tested: Attention_v2's body over AttentionHead_v12")
def AttentionV12(B, S, q, kc, vc, scale_log2):
    """`Attention_v2{T, NH, KVH, D, BN, DOT, INV}` over `AttentionHead_v12`, every head reading the one `scale_log2`."""
    return b1.AttentionV2.body(B, S, q, kc, vc, scale_log2, head=AttentionHeadV12)


@composite("AttnBlock", 13, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"], lambda S: _block_sig(S, "scale_log2"),
           conformance="not-yet-established: AttnBlock_v4's body with the scale an input")
def AttnBlockV13(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2):
    """`AttnBlock_v4{D, NVIS, FIRST, BN, DOT, CHECK}` with `scale_log2` an input."""
    return F3.AttnBlockV4.body(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2)


@composite("AttentionHead", 13, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"], lambda S: _head_sig(S, "scale_log2"),
           conformance="not-yet-established: AttentionHead_v4's body over AttnBlock_v13")
def AttentionHeadV13(B, S, q, k, v, scale_log2):
    """`AttentionHead_v4{T, D, BN, DOT, INV, MASKED_FROM}` over `AttnBlock_v13`, every block reading the one `scale_log2`."""
    return F3.AttentionHeadV4.body(B, S, q, k, v, scale_log2, block=AttnBlockV13)


@composite("Attention", 13, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"], lambda S: _attention_sig(S, "scale_log2"),
           conformance="not-yet-established: Attention_v4's body over AttentionHead_v13")
def AttentionV13(B, S, q, kc, vc, scale_log2):
    """`Attention_v4{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v13`, every head reading the one `scale_log2`."""
    return F3.AttentionV4.body(B, S, q, kc, vc, scale_log2, head=AttentionHeadV13)


# The softcapped chains with FA2's two words inputs, so `CAP` is no static: `softcap_mul` (`params.softcap = softmax_scale / CAP`,
# `dense.softcap_mul_bits`) and `scale_log2` (`params.scale_softmax_log2 = CAP * log2e`, `dense.softcap_scale_log2_bits`).


@composite("AttnBlockSoftcap", 4, ["D", "NVIS", "FIRST", "BN"], lambda S: _block_sig(S, "softcap_mul", "scale_log2"),
           conformance="exact-model-tested: AttnBlockSoftcap_v1's body with its two words inputs")
def AttnBlockSoftcapV4(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul, scale_log2):
    """`AttnBlockSoftcap_v1{D, NVIS, FIRST, BN, CAP}` with `softcap_mul` and `scale_log2` inputs."""
    return DN.AttnBlockSoftcap.body(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul, scale_log2)


@composite("AttentionHeadSoftcap", 4, ["T", "D", "BN"], lambda S: _head_sig(S, "softcap_mul", "scale_log2"),
           conformance="exact-model-tested: AttentionHeadSoftcap_v1's body over AttnBlockSoftcap_v4")
def AttentionHeadSoftcapV4(B, S, q, k, v, softcap_mul, scale_log2):
    """`AttentionHeadSoftcap_v1{T, D, BN, CAP}` over `AttnBlockSoftcap_v4`."""
    return DN.AttentionHeadSoftcap.body(B, S, q, k, v, softcap_mul, scale_log2, block=AttnBlockSoftcapV4)


@composite("AttentionSoftcap", 4, ["T", "NH", "KVH", "D", "BN"], lambda S: _attention_sig(S, "softcap_mul", "scale_log2"),
           conformance="exact-model-tested: AttentionSoftcap_v1's body over AttentionHeadSoftcap_v4")
def AttentionSoftcapV4(B, S, q, kc, vc, softcap_mul, scale_log2):
    """`AttentionSoftcap_v1{T, NH, KVH, D, BN, CAP}` over `AttentionHeadSoftcap_v4`."""
    return DN.AttentionSoftcap.body(B, S, q, kc, vc, softcap_mul, scale_log2, head=AttentionHeadSoftcapV4)


@composite("AttnBlockSoftcap", 5, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"], lambda S: _block_sig(S, "softcap_mul", "scale_log2"),
           conformance="not-yet-established: AttnBlockSoftcap_v2's body with its two words inputs")
def AttnBlockSoftcapV5(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul, scale_log2):
    """`AttnBlockSoftcap_v2{D, NVIS, FIRST, BN, CAP, DOT, CHECK}` with `softcap_mul` and `scale_log2` inputs."""
    return F2S.AttnBlockSoftcapV2.body(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul, scale_log2)


@composite("AttentionHeadSoftcap", 5, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"], lambda S: _head_sig(S, "softcap_mul", "scale_log2"),
           conformance="not-yet-established: AttentionHeadSoftcap_v2's body over AttnBlockSoftcap_v5")
def AttentionHeadSoftcapV5(B, S, q, k, v, softcap_mul, scale_log2):
    """`AttentionHeadSoftcap_v2{T, D, BN, CAP, DOT, INV, MASKED_FROM}` over `AttnBlockSoftcap_v5`."""
    return F2S.AttentionHeadSoftcapV2.body(B, S, q, k, v, softcap_mul, scale_log2, block=AttnBlockSoftcapV5)


@composite("AttentionSoftcap", 5, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: _attention_sig(S, "softcap_mul", "scale_log2"),
           conformance="not-yet-established: AttentionSoftcap_v2's body over AttentionHeadSoftcap_v5")
def AttentionSoftcapV5(B, S, q, kc, vc, softcap_mul, scale_log2):
    """`AttentionSoftcap_v2{T, NH, KVH, D, BN, CAP, DOT, INV, MASKED_FROM}` over `AttentionHeadSoftcap_v5`."""
    return F2S.AttentionSoftcapV2.body(B, S, q, kc, vc, softcap_mul, scale_log2, head=AttentionHeadSoftcapV5)


# The elementwise kinds' scalar as an input: the f32 word the ATen kernel reads (`f32(C)`; the division's `RN32(1 / C)`).


@composite("Bf16DivScalar", 3, ["N"], lambda S: ((("x", Array(S.N, BF16)), ("inv", F32)), Array(S.N, BF16)),
           conformance="exact-model-tested: Bf16DivScalar_v1's body with RN32(1 / C) an input")
def Bf16DivScalarV3(B, S, x, inv):
    """`Bf16DivScalar_v1{N, C}` with `inv = RN32(1 / C)` an input."""
    return DN.Bf16DivScalar.body(B, S, x, inv)


@composite("Bf16MulScalar", 3, ["N"], lambda S: ((("x", Array(S.N, BF16)), ("c", F32)), Array(S.N, BF16)),
           conformance="exact-model-tested: Bf16MulScalar_v1's body with f32(C) an input")
def Bf16MulScalarV3(B, S, x, c):
    """`Bf16MulScalar_v1{N, C}` with `c = f32(C)` an input."""
    return DN.Bf16MulScalar.body(B, S, x, c)


@composite("AddScalarF32", 3, ["N"], lambda S: ((("x", Array(S.N, F32)), ("c", F32)), Array(S.N, F32)),
           conformance="exact-model-tested: AddScalarF32_v1's body with f32(C) an input")
def AddScalarF32V3(B, S, x, c):
    """`AddScalarF32_v1{N, C}` with `c = f32(C)` an input."""
    return DN.AddScalarF32.body(B, S, x, c)


@composite("AddScalarBf16", 3, ["N"], lambda S: ((("w", Array(S.N, BF16)), ("c", F32)), Array(S.N, F32)),
           conformance="exact-model-tested: AddScalarBf16_v1's body with f32(C) an input")
def AddScalarBf16V3(B, S, w, c):
    """`AddScalarBf16_v1{N, C}` with `c = f32(C)` an input."""
    return DN.AddScalarBf16.body(B, S, w, c)
