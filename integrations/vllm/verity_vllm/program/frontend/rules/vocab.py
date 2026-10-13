"""Kind vocabularies: the registered definitions a derivation mode binds semantic roles to (GUIDANCE §43.2).

A rule never names a kind directly; it asks `ctx.kind(role, **statics)`. The vocabulary of the (mode, profile)
answers with a bound registered definition or `None` (-> `Unsupported(numerics_unregistered)`). Provenance records
the vocabulary's name and version, so "which kernels does this program claim" is pinned per derivation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

from verity.ml.circuits.defs import bind as ir_bind
from verity_vllm.program.registry import act as _act
from verity_vllm.program.registry import b1, gemm_bias as _gemm_bias
from verity_vllm.program.registry import dsa as _dsa, dsa_norm as _dsa_norm
from verity_vllm.program.registry import boolean_rope as _rope
from verity_vllm.program.registry import cublas_bf16 as _cublas_bf16, ll_bf16 as _ll_bf16
from verity_vllm.program.registry import sparse_mla as _sparse_mla, sparse_mla_split as _sparse_mla_split
from verity_vllm.program.registry import tp_allreduce as _tp_allreduce
from verity_vllm.program.registry import fa2_check_inf as _f2
from verity_vllm.program.registry import fa3_check_inf as _f3
from verity_vllm.program.registry import fp8 as _fp8
from verity_vllm.program.registry import fp8_vocab as _fp8_vocab
from verity_vllm.program.registry import moe as _moe
from verity_vllm.program.registry import pouw_rows as _pouw_rows
from verity_vllm.program.frontend import targets as _targets
from verity_vllm.program.registry import value_inputs as _vi
from verity_vllm.program.frontend.rules import ref_kinds as R


@dataclass
class Vocabulary:
    name: str
    mode: str
    profile: str | None
    kinds: dict[str, Callable[..., object]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def version(self) -> str:
        rows = []
        for role in sorted(self.kinds):
            rows.append(f"{role}:{self.notes.get(role, '')}")
        return hashlib.sha256("\n".join([self.name, self.notes.get("_pin", "")] + rows).encode()).hexdigest()[:16]

    def kind(self, role: str, **statics):
        binder = self.kinds.get(role)
        if binder is None:
            return None
        try:
            return binder(**statics)
        except TypeError:
            return None

    def has(self, role: str) -> bool:
        return role in self.kinds

    def to_json(self) -> dict:
        d = {"name": self.name, "version": self.version, "mode": self.mode, "profile": self.profile,
             "roles": {r: self.notes.get(r, "") for r in sorted(self.kinds)}}
        if "_pin" in self.notes:
            d["pin"] = self.notes["_pin"]
        return d


def _attention_dot_value(T, NH, KVH, D, BN, DOT, INV, FA2_MASKED_FROM):
    """Off the accepted target the scale is an input only under FA2's per-iteration Check_inf (Attention_v11): without
    `FA2_MASKED_FROM` the kind does not bind (`Vocabulary.kind`)."""
    from verity_vllm.program.registry import fa2_check_inf as _f2v  # noqa: PLC0415
    return ir_bind(_f2v.AttentionV11, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV, MASKED_FROM=FA2_MASKED_FROM)


# -- the kinds whose program values are inputs (`TargetProfile.registered_values`): the value is the kind's last operand, a field of
# the registered `values` parameter (`Ctx.supplied`); the rules ask for them instead of the constant-holding kinds under the switch
_VALUE_KINDS = {
    "rmsnorm_value": lambda N: ir_bind(_vi.RMSNormTritonV3, N=N),
    "rmsnorm_fused_value": lambda N, TARGET: _targets.rmsnorm_fused_spec(TARGET, N, eps_input=True),
    "attention_value": lambda T, NH, KVH, D: ir_bind(_vi.AttentionV10, T=T, NH=NH, KVH=KVH, D=D, BN=b1.fa2_kblock_n(D)),
    "attention_dot_value": _attention_dot_value,
    "attention_dot_plain_value": lambda T, NH, KVH, D, BN, DOT, INV: ir_bind(_vi.AttentionV12, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV),
    "attention_dot_fa3_value": lambda T, NH, KVH, D, BN, DOT, INV, MASKED_FROM:
        ir_bind(_vi.AttentionV13, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV, MASKED_FROM=MASKED_FROM),
}
_VALUE_NOTES = {
    "rmsnorm_value": "RMSNormTriton_v3{N}(x, w, eps): RMSNormTriton_v1's body with eps an input",
    "rmsnorm_fused_value": "RMSNormFusedCuda_v4{N}(x, res, w, eps): RMSNormFusedCuda_v2's body with eps an input",
    "attention_value": "Attention_v10{T,NH,KVH,D,BN=fa2_kblock_n(D)}(q, kc, vc, scale_log2): Attention_v3's body with the scale an input",
    "attention_dot_value": "Attention_v11{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}(q, kc, vc, scale_log2): Attention_v5's body with the scale an "
                           "input (FA2's per-iteration Check_inf); no kind elsewhere",
    "attention_dot_plain_value": "Attention_v12{T,NH,KVH,D,BN,DOT,INV}(q, kc, vc, scale_log2): Attention_v2's body with the scale an input",
    "attention_dot_fa3_value": "Attention_v13{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}(q, kc, vc, scale_log2): Attention_v4's body with the scale an "
                               "input (FA3's per-iteration Check_inf)",
}

# -- profile "b1-eager": the B0/B1 vLLM eager profile (TA1's registered composites, v1) --------------------------
PROFILE_B1_EAGER = Vocabulary(
    "b1-eager", "profile", "b1-eager",
    kinds={
        "gemm": lambda K, N: ir_bind(b1.Gemm, K=K, N=N),
        "pouw_linear_row": lambda K, N: ir_bind(_pouw_rows.NcpLinearRow, K=K, N=N),
        "pouw_linear_row_bias": lambda K, N: ir_bind(_pouw_rows.NcpLinearRowBias, K=K, N=N),
        "bias_add": lambda N: ir_bind(b1.BiasAdd, N=N),
        "gemm_bias": lambda K, N: ir_bind(_gemm_bias.GemmBias, K=K, N=N),
        "rmsnorm": lambda N, EPS: ir_bind(b1.RMSNormTriton, N=N, EPS=EPS),
        # `_C.fused_add_rms_norm` by the declared target (`frontend.targets.rmsnorm_fused_spec`, which the fold calls too): the rule
        # states `TARGET`, and without it the role has no kind
        "rmsnorm_fused": lambda N, EPS, TARGET: _targets.rmsnorm_fused_spec(TARGET, N, EPS),
        "rope": lambda NHEADS, D: ir_bind(b1.RoPE, NHEADS=NHEADS, D=D),
        "attention": lambda T, NH, KVH, D: ir_bind(b1.Attention, T=T, NH=NH, KVH=KVH, D=D),
        "silu_mul": lambda I: ir_bind(b1.SiluMul, I=I),
        "embedding": lambda V, H: ir_bind(b1.Embedding, V=V, H=H),
        "token_select": lambda V: ir_bind(b1.TokenSelect, V=V),
        "residual_add": lambda N: ir_bind(b1.ResidualAdd, N=N),
        **_VALUE_KINDS,
    },
    notes={
        **_VALUE_NOTES,
        "gemm": "Gemm_v1{K,N}: vLLM batch-invariant Triton matmul (AmpereBF16TcDot16 chains), bf16 in/out",
        "pouw_linear_row": "NcpLinearRow_v1{K,N}: one row of a pouw:ncp-v2 linear's served y (quantize, exact int7 dot, NcpDequant)",
        "pouw_linear_row_bias": "NcpLinearRowBias_v1{K,N}: one row of a biased pouw:ncp-v2 linear's served y (NcpDequant adds the bias)",
        "bias_add": "BiasAdd_v1{N}: aten.add(bf16, bf16)",
        "gemm_bias": "GemmBias_v1{K,N}: Gemm_v1 then BiasAdd_v1 per output word (the batch-invariant linear with a bias)",
        "rmsnorm": "RMSNormTriton_v1{N,EPS}: vLLM batch-invariant Triton _rms_norm_kernel",
        "rmsnorm_fused": "RMSNormFusedCuda_v2{N,EPS}: vLLM CUDA fused_add_rms_norm (generic kernel)",
        "rope": "RoPE_v1{NHEADS,D}: vLLM _C.rotary_embedding NeoX, bf16 cos/sin cache row as Input",
        "attention": "Attention_v1{T,NH,KVH,D}: FA2 causal GQA over [T][KVH][D] caches",
        "silu_mul": "SiluMul_v1{I}: vLLM _C.silu_and_mul",
        "embedding": "Embedding_v1{V,H}: row gather",
        "token_select": "TokenSelect_v1{V}: greedy argmax, lowest index on ties",
        "residual_add": "ResidualAdd_v1{N}",
    })

# -- reference: NUMERIC's reference kinds (`verity_vllm/program/registry/ref_prims.py`, vocabulary `verity-ref-vocab/1`) ----------
# The reference kinds define the arithmetic (sequential f32 chains, one DT rounding at stated points); agreement with torch
# CPU is measured over a tested domain (docs/data/ref-prims/), never assumed. The vocabulary version pins NUMERIC's
# `REF_VOCAB_VERSION` and `ref_vocab_digest()` (a hash over every reference kind's evaluator/body source).
from verity_vllm.program.registry import ref_prims as RP


def _ref_doc(d) -> str:
    return (d.doc or "").split("\n")[0].strip()


REFERENCE_CPU_F32 = Vocabulary(
    "reference-ref-prims", "reference", None,
    kinds={
        "linear": lambda K, N, DT, BIAS: ir_bind(RP.LinearRef, K=K, N=N, DT=DT, BIAS=bool(BIAS)),
        "add": lambda N, DT: ir_bind(RP.AddRef, N=N, DT=DT),
        "mul": lambda N, DT: ir_bind(RP.MulRef, N=N, DT=DT),
        "relu": lambda N, DT: ir_bind(RP.ReluRef, N=N, DT=DT),
        "silu": lambda N, DT: ir_bind(RP.SiluRef, N=N, DT=DT),
        "gelu": lambda N, DT, APPROX: ir_bind(RP.GeluRef, N=N, DT=DT, APPROX=APPROX),
        "layer_norm": lambda N, EPS, DT: ir_bind(RP.LayerNormRef, N=N, EPS=EPS, DT=DT),
        "rmsnorm": lambda N, EPS, DT: ir_bind(RP.RMSNormRef, N=N, EPS=EPS, DT=DT),
        "layer_norm_value": lambda N, DT: ir_bind(_vi.LayerNormRefV2, N=N, DT=DT),
        "rmsnorm_value": lambda N, DT: ir_bind(_vi.RMSNormRefV2, N=N, DT=DT),
        "softmax": lambda N, DT: ir_bind(RP.SoftmaxRef, N=N, DT=DT),
        "embedding": lambda V, H, DT: ir_bind(RP.EmbeddingRef, V=V, H=H, DT=DT),
        "argmax": lambda V, DT: ir_bind(RP.ArgmaxRef, V=V, DT=DT),
        "i64_to_i32": lambda: RP.I64ToI32,
        "eager_attention": lambda T, NH, KVH, D, DT, SCALE_OP: ir_bind(RP.EagerAttentionRef, T=T, NH=NH, KVH=KVH, D=D, DT=DT, SCALE_OP=SCALE_OP),
        "rope_heads": lambda NHEADS, D, DT: ir_bind(RP.RoPEHeadsRef, NHEADS=NHEADS, D=D, DT=DT),
        "bf16_to_f32": lambda N: ir_bind(R.DeriveRefBf16ToF32, N=N),
        "f32_to_bf16": lambda N: ir_bind(R.DeriveRefF32ToBf16Rn, N=N),
    },
    notes={
        "linear": "LinearRef_v1{K,N,DT,BIAS}: " + _ref_doc(RP.LinearRef),
        "add": "AddRef_v1{N,DT}: " + _ref_doc(RP.AddRef), "mul": "MulRef_v1{N,DT}: " + _ref_doc(RP.MulRef),
        "relu": "ReluRef_v1{N,DT}: " + _ref_doc(RP.ReluRef), "silu": "SiluRef_v1{N,DT}: " + _ref_doc(RP.SiluRef),
        "gelu": "GeluRef_v1{N,DT,APPROX}: " + _ref_doc(RP.GeluRef), "layer_norm": "LayerNormRef_v1{N,EPS,DT}: " + _ref_doc(RP.LayerNormRef),
        "rmsnorm": "RMSNormRef_v1{N,EPS,DT}: " + _ref_doc(RP.RMSNormRef),
        "layer_norm_value": "LayerNormRef_v2{N,DT}(x, g, b, eps): LayerNormRef_v1's body with eps an input",
        "rmsnorm_value": "RMSNormRef_v2{N,DT}(x, w, eps): RMSNormRef_v1's body with eps an input", "softmax": "SoftmaxRef_v1{N,DT}: " + _ref_doc(RP.SoftmaxRef),
        "embedding": "EmbeddingRef_v1{V,H,DT}: " + _ref_doc(RP.EmbeddingRef), "argmax": "ArgmaxRef_v1{V,DT}: " + _ref_doc(RP.ArgmaxRef),
        "i64_to_i32": "I64ToI32_v1: " + _ref_doc(RP.I64ToI32),
        "eager_attention": "EagerAttentionRef_v1{T,NH,KVH,D,DT,SCALE_OP}: " + _ref_doc(RP.EagerAttentionRef),
        "rope_heads": "RoPEHeadsRef_v1{NHEADS,D,DT}: " + _ref_doc(RP.RoPEHeadsRef),
        "bf16_to_f32": "batched Bf16ToF32_v1 (exact widening; derive-lane composite over the registered primitive)",
        "f32_to_bf16": "batched F32ToBf16Rn_v1 (RNE narrowing; derive-lane composite over the registered primitive)",
        "_pin": f"{RP.REF_VOCAB_VERSION} digest {RP.ref_vocab_digest()}",
    })

# -- profile "b1-eager-v2": the same Ampere eager profile over the v2 skeleton (`Serve_v2`/`Serve_v3`: `Gemm_v2{K,N,DOT}`,
# `Attention_v2{..,BN,DOT,INV}`, the `NORM` bundle) + the `pad-serve@1` kinds of the padded circuit (`serve3.py`) ------------
from verity_vllm.program.registry import b1_tp2 as TP2
from verity_vllm.program.registry import pad_prims as PP
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import sampling as _sampling  # [R15 stoch]
from verity_vllm.program.registry import sampling_vocab
from verity_vllm.program.registry import topp_words as _topp_words  # the keep word as word gates at a constant S
from verity_vllm.program.registry import layer_norm as _LN


def _norm_v2(role: str):
    def bind_norm(N, EPS):
        rec = b1.norm_bundle({"H": N, "EPS": EPS})
        return rec[role]
    return bind_norm


PROFILE_B1_EAGER_V2 = Vocabulary(
    "b1-eager-v2", "profile", "b1-eager-v2",
    kinds={
        "gemm": lambda K, N: ir_bind(b1.GemmV2, K=K, N=N, DOT=b1.GEMM_AMPERE["DOT"]),
        "pouw_linear_row": lambda K, N: ir_bind(_pouw_rows.NcpLinearRow, K=K, N=N),
        "pouw_linear_row_bias": lambda K, N: ir_bind(_pouw_rows.NcpLinearRowBias, K=K, N=N),
        "bias_add": lambda N: ir_bind(b1.BiasAdd, N=N),
        "rmsnorm": _norm_v2("L0"),
        "rmsnorm_fused": lambda N, EPS, TARGET: _targets.rmsnorm_fused_spec(TARGET, N, EPS),
        "rope": lambda NHEADS, D: ir_bind(b1.RoPE, NHEADS=NHEADS, D=D),
        "attention": lambda T, NH, KVH, D: ir_bind(b1.AttentionV2, T=T, NH=NH, KVH=KVH, D=D, **b1.ATTENTION_FA2),
        "silu_mul": lambda I: ir_bind(b1.SiluMul, I=I),
        "embedding": lambda V, H: ir_bind(b1.Embedding, V=V, H=H),
        "token_select": lambda V: ir_bind(b1.TokenSelect, V=V),
        "residual_add": lambda N: ir_bind(b1.ResidualAdd, N=N),
        # pad-serve@1 (spec §6/§7): the liveness chain and the muxes, as `serve3.py` uses them
        "const": lambda W, BITS: b1.const(W, BITS),
        "i32_eq": lambda: P.I32Eq,
        "bit_not": lambda: PP.BitNot,
        "bit_and": lambda: PP.BitAnd,
        "select_i32": lambda: P.SelectI32,
        "select_bf16": lambda: P.SelectBf16,
    },
    notes={
        "gemm": "Gemm_v2{K,N,DOT=AmpereBF16TcDot16_v2}: batch(GemmCoordinate_v2) — Serve_v2/v3's linear",
        "pouw_linear_row": "NcpLinearRow_v1{K,N}: one row of a pouw:ncp-v2 linear's served y (quantize, exact int7 dot, NcpDequant)",
        "pouw_linear_row_bias": "NcpLinearRowBias_v1{K,N}: one row of a biased pouw:ncp-v2 linear's served y (NcpDequant adds the bias)",
        "bias_add": "BiasAdd_v1{N}", "rmsnorm": "norm_bundle(C)['L0'] = RMSNormTriton_v1{N,EPS}",
        "rmsnorm_fused": "norm_bundle(C)['FUSED'] = RMSNormFusedCuda (b1's default fused norm){N,EPS}",
        "rope": "RoPE_v1{NHEADS,D}", "attention": "Attention_v2{T,NH,KVH,D,BN=64,DOT=AmpereBF16TcDot16_v2,INV=Fa2InvSum_v1} (FA2)",
        "silu_mul": "SiluMul_v1{I}", "embedding": "Embedding_v1{V,H}", "token_select": "TokenSelect_v1{V}", "residual_add": "ResidualAdd_v1{N}",
        "const": "Const<W>[bits]: a constant gate (b1.const)", "i32_eq": "I32Eq_v1", "bit_not": "BitNot_v1 (pad_prims)",
        "bit_and": "BitAnd_v1 (pad_prims; absorbing)", "select_i32": "SelectI32_v1", "select_bf16": "SelectBf16_v1",
    })


# -- the generic elementwise / activation / softcap kinds [R13 dense2 head-to-head; R15 dense F-dense-05]: the operations vLLM runs
# UNFUSED under the eager batch-invariant profile when a model's norm is `forward_native` (`ir.ops.rms_norm` / `fused_add_rms_norm` as
# an ATen chain on an f32 copy), its activation is GeGLU, its attention has a logit soft cap, or its logits are soft-capped. Every kind
# is op-shaped (an ATen CUDA kernel on a bf16 / f32 row, a Triton reduction, a softcapped FA2 head) — nothing here names a model; the
# Definitions live in `verity_vllm.program.registry.dense` (promoted from `quarantine.dense`, the X-08 pattern: same name / version / body, so the
# specialization ids the accepted `GEMMA2_2B` record resolved to are unchanged). `rules/vllm_bindings.py` asks for them. ----------------
def _dense_kinds() -> tuple[dict, dict]:
    from verity_vllm.program.registry import dense as DN
    from verity_vllm.program.registry import tables as _tables
    kinds = {
        "scale_tensor": lambda N: ir_bind(b1.Bf16MulScalarTensor, N=N),
        "gelu_tanh_mul": lambda I: ir_bind(DN.GeluTanhMul, I=I),
        "gelu_tanh_mul_table": lambda I: ir_bind(_tables.GeluTanhMulV3, I=I),
        "attention_softcap": lambda T, NH, KVH, D, CAP: ir_bind(DN.AttentionSoftcap, T=T, NH=NH, KVH=KVH, D=D, BN=b1.fa2_kblock_n(D), CAP=CAP),
        "bf16_div_scalar": lambda N, C: ir_bind(DN.Bf16DivScalar, N=N, C=C),
        "bf16_mul_scalar": lambda N, C: ir_bind(DN.Bf16MulScalar, N=N, C=C),
        "bf16_tanh": lambda N: ir_bind(DN.Bf16Tanh, N=N),
        "square_bf16": lambda N: ir_bind(DN.SquareBf16, N=N),
        "square_f32": lambda N: ir_bind(DN.SquareF32, N=N),
        "mean_triton": lambda N: ir_bind(DN.MeanTriton, N=N),
        "rsqrt_f32": lambda N: ir_bind(DN.RsqrtF32, N=N),
        "add_scalar_f32": lambda N, C: ir_bind(DN.AddScalarF32, N=N, C=C),
        "add_scalar_bf16": lambda N, C: ir_bind(DN.AddScalarBf16, N=N, C=C),
        "add_widened_bf16": lambda N: ir_bind(DN.AddWidenedBf16, N=N),
        "scale_row_f32": lambda N: ir_bind(DN.ScaleRowF32, N=N),
        "scale_row_bf16": lambda N: ir_bind(DN.ScaleRowBf16, N=N),
        "mul_vec_f32": lambda N: ir_bind(DN.MulVecF32, N=N),
        "narrow_f32_bf16": lambda N: ir_bind(DN.NarrowF32ToBf16, N=N),
        # the value-input twins (`TargetProfile.registered_values`): the scalar or the softcap's words a registered value
        "attention_softcap_value": lambda T, NH, KVH, D: ir_bind(_vi.AttentionSoftcapV4, T=T, NH=NH, KVH=KVH, D=D, BN=b1.fa2_kblock_n(D)),
        "attention_softcap_dot_value": lambda T, NH, KVH, D, BN, DOT, INV, FA2_MASKED_FROM:
            ir_bind(_vi.AttentionSoftcapV5, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV, MASKED_FROM=FA2_MASKED_FROM),
        "bf16_div_scalar_value": lambda N: ir_bind(_vi.Bf16DivScalarV3, N=N),
        "bf16_mul_scalar_value": lambda N: ir_bind(_vi.Bf16MulScalarV3, N=N),
        "add_scalar_f32_value": lambda N: ir_bind(_vi.AddScalarF32V3, N=N),
        "add_scalar_bf16_value": lambda N: ir_bind(_vi.AddScalarBf16V3, N=N),
    }
    notes = {
        "scale_tensor": "Bf16MulScalarTensor_v1{N}: aten.mul.Tensor(bf16 row, 0-d bf16 tensor) — f32 multiply, cvt.rn.bf16.f32",
        "gelu_tanh_mul": "GeluTanhMul_v1{I}: vLLM _C.gelu_tanh_and_mul (act_and_mul_kernel<bf16, gelu_tanh_kernel, act_first>)",
        "gelu_tanh_mul_table": "GeluTanhMul_v3{I}(gu, gelu_tanh_bf16): GeluTanhMul_v1 with the activation read from the registered table",
        "attention_softcap": "AttentionSoftcap_v1{T,NH,KVH,D,BN=fa2_kblock_n(D),CAP}: FA2 causal GQA with Is_softcap (MUFU.TANH on the scores)",
        "bf16_div_scalar": "Bf16DivScalar_v1{N,C}: aten.div.Tensor(bf16, C) — f32(x) * RN32(1/C), cvt.rn.bf16.f32",
        "bf16_mul_scalar": "Bf16MulScalar_v1{N,C}: aten.mul.Tensor(bf16, C) — f32 multiply, cvt.rn.bf16.f32",
        "bf16_tanh": "Bf16Tanh_v1{N}: aten.tanh(bf16) — tanhf in f32, cvt.rn.bf16.f32",
        "square_bf16": "SquareBf16_v1{N}: aten.pow(x.to(f32), 2) on a widened bf16 row — one FMUL", "square_f32": "SquareF32_v1{N}: aten.pow(x f32, 2) — one FMUL",
        "mean_triton": "MeanTriton_v1{N}: aten.mean.dim under VLLM_BATCH_INVARIANT (batch_invariant.mean_kernel, BLOCK_SIZE=1024, 4 warps)",
        "rsqrt_f32": "RsqrtF32_v1{N}: aten.rsqrt(f32) — rsqrt.approx.f32 (MUFU.RSQ + range scaling)",
        "add_scalar_f32": "AddScalarF32_v1{N,C}: aten.add.Tensor(f32, C) — one FADD", "add_scalar_bf16": "AddScalarBf16_v1{N,C}: aten.add.Tensor(w.to(f32), C) — one FADD",
        "add_widened_bf16": "AddWidenedBf16_v1{N}: aten.add.Tensor(x.to(f32), r.to(f32)) — one FADD",
        "scale_row_f32": "ScaleRowF32_v1{N}: aten.mul.Tensor(x f32 [M,N], s f32 [M,1]) — one FMUL",
        "scale_row_bf16": "ScaleRowBf16_v1{N}: aten.mul.Tensor(x.to(f32) [M,N], s f32 [M,1]) — one FMUL",
        "mul_vec_f32": "MulVecF32_v1{N}: aten.mul.Tensor(x f32 [M,N], w f32 [N]) — one FMUL",
        "narrow_f32_bf16": "NarrowF32ToBf16_v1{N}: aten.to.dtype(f32 -> bf16) — cvt.rn.bf16.f32 (NaN -> 0x7FFF)",
        "attention_softcap_value": "AttentionSoftcap_v4{T,NH,KVH,D,BN=fa2_kblock_n(D)}(q, kc, vc, softcap_mul, scale_log2): AttentionSoftcap_v1's "
                                   "body with FA2's two softcap words inputs",
        "attention_softcap_dot_value": "AttentionSoftcap_v5{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}(q, kc, vc, softcap_mul, scale_log2): "
                                       "AttentionSoftcap_v2's body with the two words inputs (FA2's per-iteration Check_inf)",
        "bf16_div_scalar_value": "Bf16DivScalar_v3{N}(x, inv): Bf16DivScalar_v1's body with RN32(1/C) an input",
        "bf16_mul_scalar_value": "Bf16MulScalar_v3{N}(x, c): Bf16MulScalar_v1's body with f32(C) an input",
        "add_scalar_f32_value": "AddScalarF32_v3{N}(x, c): AddScalarF32_v1's body with f32(C) an input",
        "add_scalar_bf16_value": "AddScalarBf16_v3{N}(w, c): AddScalarBf16_v1's body with f32(C) an input",
    }
    return kinds, notes


_DNK, _DNN = _dense_kinds()


def _moe_ffn_fixed(E: int, TOPK: int, VPT: int, H: int, I: int):
    """[R17-42 route A (3)] `MoeFfnFixedL_v1{E,TOPK,VPT,H,I}` (registry.moe_pad), imported on first use: the padded registry (pad's lifted
    registry + the hand-lift hook) is not loaded by a dense derivation that never asks for the kind."""
    from verity_vllm.program.registry import moe_pad as _mp  # noqa: PLC0415
    return ir_bind(_mp.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)


def _silu_mul_fp8_block_quant(I: int, G: int):
    from verity_vllm.program.registry import fp8_moe as _fm  # noqa: PLC0415
    return ir_bind(_fm.SiluMulFp8BlockQuant, I=I, G=G)


def _moe_expert_dot(routed, E, K, N, DOT, G, KB, SWAP):
    if G is None and KB is None and SWAP is None:
        return ir_bind(_moe.MoeExpertGemmWV2 if routed else _moe.MoeExpertGemmV2, E=E, K=K, N=N, DOT=DOT)
    if G is None or KB is None or SWAP is None:
        raise TypeError("the block-FP8 pair forms take G, KB and SWAP together")
    from verity_vllm.program.registry import fp8_moe_experts as _x  # noqa: PLC0415
    return ir_bind(_x.MoeExpertFp8GemmW if routed else _x.MoeExpertFp8Gemm, E=E, K=K, N=N, G=G, KB=KB, SWAP=SWAP, DOT=DOT)


def _attention_softcap_dot(T, NH, KVH, D, BN, DOT, INV, FA2_MASKED_FROM, CAP):
    if FA2_MASKED_FROM is None:
        raise TypeError("a softcap off the accepted target is registered only under FA2's per-iteration Check_inf")
    from verity_vllm.program.registry import fa2_softcap as _f2s  # noqa: PLC0415
    return ir_bind(_f2s.AttentionSoftcapV2, T=T, NH=NH, KVH=KVH, D=D, BN=BN, CAP=CAP, DOT=DOT, INV=INV, MASKED_FROM=FA2_MASKED_FROM)

# -- profile "b1-eager-v3" [CB-A 2026-09-18]: the admitted `vllm_d9105ea80_sm89_eager` profile as the accepted records bind it —
# TA1's v1 kinds with the attention chain `Attention_v3{T,NH,KVH,D,BN}` (`b1.AttentionV3`, `BN = fa2_kblock_n(D)`: FA2's kBlockN at
# this head dim, D54/D57). Used by the export-of-the-real-vLLM-class frontend (`rules/vllm_bindings.py`). ---------------------------
_PAD_KINDS = {k: PROFILE_B1_EAGER_V2.kinds[k] for k in ("const", "i32_eq", "bit_not", "bit_and", "select_i32", "select_bf16")}
_PAD_NOTES = {k: PROFILE_B1_EAGER_V2.notes[k] for k in _PAD_KINDS}

PROFILE_B1_EAGER_V3 = Vocabulary(
    "b1-eager-v3", "profile", "b1-eager-v3",
    kinds={**PROFILE_B1_EAGER.kinds, **_PAD_KINDS, **_DNK,
           "attention": lambda T, NH, KVH, D: ir_bind(b1.AttentionV3, T=T, NH=NH, KVH=KVH, D=D, BN=b1.fa2_kblock_n(D)),
           # `_C.silu_and_mul` as `SiluMul_v2` (registry.act): `SiluMul_v1` on every word but the red-team's edge words (expf's overflow,
           # up = -0, gate -inf / -0, NaN 0x7FFF); the older vocabularies keep `SiluMul_v1`
           "silu_mul": lambda I: ir_bind(_act.SiluMul, I=I),
           # [R13 hopper] the target-keyed variants (`verity_vllm.program.frontend.targets`): the same bodies with the k16 tensor-core step and (for
           # attention) the block size / epilogue as declared statics — bound by the vLLM rules when the active TargetProfile is not the
           # accepted records' (Ampere-family step under FA2); those keep `gemm` / `attention` above, so their digests do not move.
           "gemm_dot": lambda K, N, DOT: ir_bind(b1.GemmV2, K=K, N=N, DOT=DOT),
           "gemm_bias_dot": lambda K, N, DOT: ir_bind(_gemm_bias.GemmBiasV2, K=K, N=N, DOT=DOT),
           # a `MASKED_FROM` binds FA3's per-iteration Check_inf chain instead (`Attention_v4`, `TargetProfile.fa3_construction`), an
           # `FA2_MASKED_FROM` FA2's (`Attention_v5`, `targets.fa2_check_inf`); optional statics, so the role and its note -- and with them
           # this vocabulary's version -- stay as they were; a `CAP` with an `FA2_MASKED_FROM` binds the softcapped chain
           # (`AttentionSoftcap_v2`), and a `CAP` without one has no kind
           "attention_dot": lambda T, NH, KVH, D, BN, DOT, INV, MASKED_FROM=None, FA2_MASKED_FROM=None, CAP=None:
               _attention_softcap_dot(T, NH, KVH, D, BN, DOT, INV, FA2_MASKED_FROM, CAP) if CAP is not None
               else ir_bind(_f2.AttentionV5, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV, MASKED_FROM=FA2_MASKED_FROM)
               if FA2_MASKED_FROM is not None
               else ir_bind(b1.AttentionV2, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV) if MASKED_FROM is None
               else ir_bind(_f3.AttentionV4, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=DOT, INV=INV, MASKED_FROM=MASKED_FROM),
           # [R13 X-08] `scale_tensor` (generic bf16 row x 0-d bf16 tensor) and [R15 dense] the other op-shaped elementwise / activation /
           # softcap kinds come from `_dense_kinds()` above (promoted from quarantine.dense; Gemma-2's ops were the first instances)
           # [R13 MOE-01] the vLLM fused-MoE block as four kinds (`registry.moe`, promoted from quarantine.ov_moe): the router's f32 softmax +
           # TOPK-round argmax per token row; the pair-major expert GEMMs with the expert index as an I32 VALUE operand (up: plain; down: the
           # routed weight multiplied into the f32 accumulator); the routed sum over a token's TOPK slots. `rules/vllm_moe.py` asks for them.
           "moe_router_topk": lambda E, TOPK, VPT: ir_bind(_moe.MoeRouterTopK, E=E, TOPK=TOPK, VPT=VPT),
           "moe_router_topk_norm": lambda E, TOPK, VPT: ir_bind(_moe.MoeRouterTopKNorm, E=E, TOPK=TOPK, VPT=VPT),
           # the same router in its selection order over one softmax (`moe_construction = indexed-read-ordered`)
           "moe_router_topk_ordered": lambda E, TOPK, VPT: ir_bind(_moe.MoeRouterTopKOrdered, E=E, TOPK=TOPK, VPT=VPT),
           "moe_router_topk_ordered_norm": lambda E, TOPK, VPT: ir_bind(_moe.MoeRouterTopKOrderedNorm, E=E, TOPK=TOPK, VPT=VPT),
           "moe_expert_gemm": lambda E, K, N: ir_bind(_moe.MoeExpertGemm, E=E, K=K, N=N),
           "moe_expert_gemm_w": lambda E, K, N: ir_bind(_moe.MoeExpertGemmW, E=E, K=K, N=N),
           # the same launches with the k16 step as a static, bound when the active target's fused_moe_kernel runs another step than
           # the Ampere/Ada one (`frontend.targets.moe_expert_dot`); those above keep the `_v1` records' digests
           # a `G` (with the launch's `KB`, `SWAP`) binds the block-FP8 pair forms with DOT the e4m3 step (`rules/vllm_moe_fp8.py`); optional
           # statics, so the roles, their notes and this vocabulary's version stay as they were
           "moe_expert_gemm_dot": lambda E, K, N, DOT, G=None, KB=None, SWAP=None: _moe_expert_dot(False, E, K, N, DOT, G, KB, SWAP),
           "moe_expert_gemm_w_dot": lambda E, K, N, DOT, G=None, KB=None, SWAP=None: _moe_expert_dot(True, E, K, N, DOT, G, KB, SWAP),
           "moe_sum": lambda TOPK, H: ir_bind(_moe.MoeSum, TOPK=TOPK, H=H),
           # the one-group sigmoid router (`_moe_C.grouped_topk`, DeepSeek-V3 / GLM `noaux_tc`) and the block-FP8 experts' fused activation
           # and w2-input quant (`_C.silu_and_mul_per_block_quant`): `rules/vllm_moe_router.py`, `rules/vllm_moe_fp8.py`
           "moe_router_sigmoid_bias": lambda E, TOPK, RSF, KERNEL: ir_bind(_moe.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=RSF, KERNEL=KERNEL),
           "silu_mul_fp8_block_quant": lambda I, G: _silu_mul_fp8_block_quant(I, G),
           # [R17-42 route A (3)] the fused-MoE block under `TargetProfile.moe_construction = "padded"` as ONE kind per token: the fixed-slot
           # construction from the router logits (`registry.moe_pad.MoeFfnFixedL_v1`: ALL E expert bodies on their own weights, the TOPK slot
           # outputs selected by the router ids, `MoeSum_v1`); its lift of record is the hand-authored `LMoeFfnL_v1` (unselected slots DISABLED).
           # `rules/vllm_moe.py`'s padded rules ask for it at `_moe_C.moe_sum` (the block's last kernel); the registry is imported lazily.
           "moe_ffn_fixed": lambda E, TOPK, VPT, H, I: _moe_ffn_fixed(E, TOPK, VPT, H, I),
           # [R13 tp, P4] tensor-parallel collectives (registry.b1_tp2; exact-model-tested B1 TP2, re-confirmed on Llama TP2 by the step-1 probe):
           # world size 2 — `AllReduce2` is one Bf16Add per element (order-free for two ranks), `AllGather2` a concatenation, `EmbeddingShard` one
           # rank's masked vocab-shard gather.  A TP>2 world needs an `AllReduceN` whose NCCL tree order is part of the Definition.
           "all_reduce2": lambda N: ir_bind(TP2.AllReduce2, N=N),
           "all_gather2": lambda N: ir_bind(TP2.AllGather2, N=N),
           # [lane vllm-tp-n] the WORLD-parametric families for a world > 2 (identities `AllReduce_v2{WORLD,N}` / `AllGather_v1{WORLD,N}`; world 2
           # keeps the `*2` identities above -- `b1_tp2.all_reduce(world, N)` is the selector)
           "all_reduce": lambda WORLD, N: ir_bind(TP2.AllReduce, WORLD=WORLD, N=N),
           "all_reduce_mnnvl": lambda WORLD, N, SHOT: ir_bind(_tp_allreduce.AllReduceMnnvl, WORLD=WORLD, N=N, SHOT=SHOT),
           "all_reduce_multimem": lambda WORLD, SIZE, TREES, FTZ:
               ir_bind(_tp_allreduce.AllReduceMultimem, WORLD=WORLD, SIZE=SIZE, TREES=TREES, FTZ=FTZ),
           "all_reduce_custom": lambda WORLD, SIZE, STAGE:
               ir_bind(_tp_allreduce.AllReduceCustom, WORLD=WORLD, SIZE=SIZE, STAGE=STAGE),
           "all_reduce_rmsnorm_mnnvl": lambda WORLD, N, SHOT, CLUSTER, EPS, FMA:
               ir_bind(_tp_allreduce.AllReduceRmsNormMnnvl, WORLD=WORLD, N=N, SHOT=SHOT, CLUSTER=CLUSTER, EPS=EPS, FMA=FMA),
           "all_gather": lambda WORLD, N: ir_bind(TP2.AllGather, WORLD=WORLD, N=N),
           "embedding_shard": lambda VS, H, START: ir_bind(TP2.EmbeddingShard, VS=VS, H=H, START=START),
           # [R15 stoch] the seeded temperature + top-p token select (registry.sampling: TemperatureScale -> TopPMask{V} -> GumbelSelectF32,
           # exact to vLLM d9105ea80 `_temperature_kernel` -> `_apply_topp_split` (S splits) -> `_gumbel_sample_kernel` + argmax; the seed, pos,
           # temperature, top_p and [R16, coord M-0169] the split count S (= SplitsFor_v1(|live|, num_SMs), a derived value of the workload the
           # request Program reads from its `splits` Input) are OPERANDS).  `rules/vllm_sampling.py` asks for it.  Greedy rows keep `token_select`.
           # `CONSTRUCTION` = the target's `sampler_construction` (`registry.sampling_vocab.select_for`; the role and its note stay as they were)
           "token_select_stochastic": lambda V, CONSTRUCTION: ir_bind(sampling_vocab.select_for(_sampling.GumbelTopPTokenSelect, CONSTRUCTION), V=V),
           # the same select at a constant split count S (a single-request workload): its top-p keep word as word gates
           # (registry.topp_words, `GumbelTopPTokenSelect_v2{V,S}`, the same six operands), so a partition query cuts it
           "token_select_stochastic_split": lambda V, S: ir_bind(_topp_words.GumbelTopPTokenSelectV2, V=V, S=S),
           # [R15 fp8] the block-scaled FP8 linear of a pre-quantised checkpoint on Hopper (registry.fp8, pinned bit-exact on the H100 against
           # _C.per_token_group_fp8_quant and the blockwise _C.cutlass_scaled_mm): the dynamic per-128-group activation quant (both outputs
           # computed) and the scaled GEMM over e4m3 weights with f32 128x128 block scales as Inputs. `rules/vllm_bindings.py` asks for them.
           # SCALE is `TargetProfile.effective_fp8_scale_construction()`; FP8_DOT, given on a target whose blockwise step is not Hopper's
           # (`frontend.targets.fp8_block_dot`), binds the version 3 forms over it (the role and its note stay as they were).
           "fp8_group_quant": lambda K, G: ir_bind(_fp8.Fp8GroupQuant, K=K, G=G),
           "scaled_mm_fp8_block": lambda K, N, G, SCALE, FP8_DOT=None: _fp8_vocab.block_gemm(K, N, G, SCALE, FP8_DOT),
           # GPT-NeoX's nn.LayerNorm (weight + bias): torch's vectorized CUDA kernel, the Definition the eager Match fold binds
           # (`registry/layer_norm.py`, `engine/profiles/gen_ln_patterns.py`).  `rules/vllm_bindings/layer_norm.py` asks for it.
           "layer_norm_aten": lambda N, EPS: ir_bind(_LN.LayerNormAten, N=N, EPS=EPS),
           "layer_norm_aten_value": lambda N: ir_bind(_LN.LayerNormAtenV2, N=N),
           # [GLM-5.3] DeepSeek sparse attention on the H200 (registry.dsa, registry.sparse_mla): the indexer's K slot and query, DeepGEMM's
           # MQA logits, the top-k of a row of at most TOPK keys, and FA3's sparse MLA over the selected rows. `rules/vllm_dsa.py` asks for them.
           "dsa_indexer_k_row": lambda D, ROPE, EPS: ir_bind(_dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=EPS),
           "dsa_indexer_q": lambda H, D, ROPE: ir_bind(_dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE),
           "dsa_mqa_logits": lambda N, H, D: ir_bind(_dsa.DsaMqaLogitsV3, N=N, H=H, D=D),
           "dsa_topk_short_row": lambda TOPK: ir_bind(_dsa.DsaTopKShortRow, TOPK=TOPK),
           "dsa_rmsnorm": lambda N, EPS: ir_bind(_dsa_norm.DsaRMSNorm, N=N, EPS=EPS),
           "dsa_kv_rope": lambda D: ir_bind(_rope.RoPEHeadGptJV2, D=D),
           "dsa_q_rope": lambda D: ir_bind(_rope.RoPEHeadGptJV2, D=D),
           "cublas_splitk_gemm": lambda K, N, SPLIT_K, DOT:
               ir_bind(_cublas_bf16.CublasSplitKGemmWord, K=K, N=N, SPLIT_K=SPLIT_K, DOT=DOT),
           "batched_gemm": lambda B, K, N, DOT: ir_bind(_cublas_bf16.BatchedGemmWord, B=B, K=K, N=N, DOT=DOT),
           "ll_bf16_dotprod": lambda N, K, BS, VEC, FMA:
               ir_bind(_ll_bf16.LlBf16DotprodGemmWord, N=N, K=K, BS=BS, VEC=VEC, FMA=FMA),
           "ll_bf16_splitk": lambda N, K, SPLIT_K, TILE_K, WARPS, DOT:
               ir_bind(_ll_bf16.LlBf16SplitKGemmWord, N=N, K=K, SPLIT_K=SPLIT_K, TILE_K=TILE_K, WARPS=WARPS, DOT=DOT),
           "sparse_mla": lambda T, NH, DR, DV, BN, DOT, INV, SM_SCALE:
               ir_bind(_sparse_mla.SparseMla, T=T, NH=NH, DR=DR, DV=DV, BN=BN, DOT=DOT, INV=INV, SCALE=SM_SCALE),
           "sparse_mla_split": lambda T, NS, NH, DR, DV, BN, DOT, INV, SM_SCALE:
               ir_bind(_sparse_mla_split.SparseMlaSplit, T=T, NS=NS, NH=NH, DR=DR, DV=DV, BN=BN, DOT=DOT, INV=INV, SCALE=SM_SCALE)},
    notes={**PROFILE_B1_EAGER.notes, **_PAD_NOTES, **_DNN,
           "token_select_stochastic_split": "GumbelTopPTokenSelect_v2{V,S}: GumbelTopPTokenSelect_v1{V} at the constant split count S, "
                                            "its keep word TopPKeepWord_v1{V,S} as word gates (registry.topp_words); equal to v1 at splits = S",
           "token_select_stochastic": "GumbelTopPTokenSelect_v1{V}(logits bf16[V], top_p f32, seed i64, pos i64, temp f32, splits i32) -> i32: f32 "
                                      "widening, div.full.f32 temperature, the S-split top-p mask (_topp_sb_{stats,step,mask}; S = the splits operand "
                                      "= SplitsFor_v1(|live|, num_SMs)), Gumbel-max with key tl.randint(seed, pos), first global maximum "
                                      "(registry.sampling; difftest 72/72 bit-exact on L40S; R16 GPU fixture at S = 1..32)",
           "attention": "Attention_v3{T,NH,KVH,D,BN=fa2_kblock_n(D)}: FA2 causal GQA, kBlockN as a declared static (b1.AttentionV3)",
           "silu_mul": "SiluMul_v2{I}: vLLM _C.silu_and_mul (act_and_mul_kernel<bf16, silu_kernel>) -- SiluMulBf16_v2 per element: "
                       "expf's overflow (gate <= -89 -> -0), up + 0.0f, gate -0 kept, NaN 0x7FFF (registry.act)",
           "gemm_dot": "Gemm_v2{K,N,DOT}: the batch-invariant Triton GEMM with the target's registered k16 step (registry.targets.gemm_spec)",
           "gemm_bias_dot": "GemmBias_v2{K,N,DOT}: Gemm_v2 then BiasAdd_v1 per output word, the biased linear on a target whose record says "
                            "gemm_bias_dot (registry.targets.gemm_bias_spec)",
           "attention_dot": "Attention_v2{T,NH,KVH,D,BN,DOT,INV}: FA2/FA3 causal GQA with the target's step, block size and epilogue "
                            "(registry.targets.attention_spec)",
           "attention": "Attention_v3{T,NH,KVH,D,BN=fa2_kblock_n(D)}: FA2 causal GQA, kBlockN as a declared static (b1.AttentionV3)",
           "moe_router_topk": "MoeRouterTopK_v1{E,TOPK,VPT}: _moe_C.topk_softmax(renormalize=False) per token row — f32 softmax (NvExpf, kernel "
                              "reduction geometry E/VPT threads x VPT), TOPK rounds of lowest-index argmax; -> (weights f32[TOPK], ids i32[TOPK])",
           "moe_router_topk_norm": "MoeRouterTopKNorm_v1{E,TOPK,VPT}: the same kernel with renormalize=True -- the TOPK probabilities summed in slot order, "
                                   "denom = sum > 0 ? sum : 1, one div.rn.f32 reciprocal, one f32 mul per slot (Qwen3-MoE / Mixtral norm_topk_prob)",
           "moe_router_topk_ordered": "MoeRouterTopKOrdered_v1{E,TOPK,VPT}: MoeRouterTopK_v1's function over ONE softmax: round k takes the expert "
                                      "after round k-1's in the kernel's order (p descending, index ascending), each weight p[id] "
                                      "(moe_construction = indexed-read-ordered)",
           "moe_router_topk_ordered_norm": "MoeRouterTopKOrderedNorm_v1{E,TOPK,VPT}: MoeRouterTopKNorm_v1's function, stated as MoeRouterTopKOrdered_v1",
           "moe_expert_gemm": "MoeExpertGemm_v1{E,K,N}: fused_moe_kernel(MUL_ROUTED_WEIGHT=False) for one (token, slot) pair — bf16_rne(DotBf16(x, W[e][n,:])), "
                              "e an I32 value operand (gather over the expert axis inside the body)",
           "moe_expert_gemm_w": "MoeExpertGemmW_v1{E,K,N}: fused_moe_kernel(MUL_ROUTED_WEIGHT=True) — bf16_rne(f32mul(DotBf16(x, W[e][n,:]), g)), g the pair's routed weight",
           "moe_expert_gemm_dot": "MoeExpertGemm_v2{E,K,N,DOT}: MoeExpertGemm_v1 with the target's k16 step DOT — bf16_rne(DotBf16_v2{K,DOT}(x, W[e][n,:]))",
           "moe_expert_gemm_w_dot": "MoeExpertGemmW_v2{E,K,N,DOT}: MoeExpertGemmW_v1 with the target's k16 step DOT — "
                                    "bf16_rne(f32mul(DotBf16_v2{K,DOT}(x, W[e][n,:]), g))",
           "moe_sum": "MoeSum_v1{TOPK,H}: _moe_C.moe_sum — per token, ascending-k f32 sum of the TOPK weighted expert rows from +0, one bf16 RNE",
           "moe_router_sigmoid_bias": "MoeRouterSigmoidBias_v1{E,TOPK,SCALE=RSF,KERNEL}: _moe_C.grouped_topk(n_group=1, topk_group=1, renormalize=True, "
                                      "SIGMOID) per token row of f32 logits with the f32 selection bias -> (weights f32[TOPK], ids i32[TOPK]); "
                                      "KERNEL the launch (block / warp kernel, registry.moe)",
           "silu_mul_fp8_block_quant": "SiluMulFp8BlockQuant_v1{I,G}: _C.silu_and_mul_per_block_quant(group_size=G, scale_ub=None, scales not "
                                       "transposed) per (token, slot) row [gate || up] bf16[2I] -> (q e4m3[I], scale f32[I/G]): f32 SiLU x up, "
                                       "the group's fmaxf absmax, scale = max(absmax / 448, 1 / (448 * 512)), q = e4m3(clamp(result / scale)) "
                                       "(registry.fp8_moe)",
           "moe_ffn_fixed": "MoeFfnFixedL_v1{E,TOPK,VPT,H,I}: [R17-42 padded] the whole fused block of one token from its router logits -- "
                            "MoeRouterTopK_v1 -> ALL E MoeExpertBody_v1{H,I} on their own weights (g_e the routed weight of the slot the router "
                            "picked, 0.0f otherwise) -> the TOPK slot outputs selected by the ids -> MoeSum_v1; value-equal to the indexed-read "
                            "reading of the same launches; lift of record LMoeFfnL_v1 (hand-authored, unselected slots DISABLED)",
           "all_reduce2": "AllReduce2_v1{N}: pynccl ncclSum bf16 of two ranks' partials = bf16(f32(a)+f32(b)) per element (b1_tp2)",
           "all_gather2": "AllGather2_v1{N}: vllm.all_gather of two [N] shards along the trailing axis = the [2N] concatenation (b1_tp2)",
           "all_reduce": "AllReduce_v2{WORLD,N}: pynccl ncclSum bf16 over WORLD>2 ranks' partials as NCCL's intra-node tree chain of one-rounding bf16 adds (b1_tp2.allreduce_order)",
           "all_reduce_mnnvl": "AllReduceMnnvl_v1{WORLD,N,SHOT}: FlashInfer MNNVL all-reduce on a captured route",
           "all_reduce_multimem": "AllReduceMultimem_v1{WORLD,SIZE,TREES,FTZ}: torch symmetric-memory multimem all-reduce on captured trees",
           "all_reduce_custom": "AllReduceCustom_v1{WORLD,SIZE,STAGE}: vLLM custom all-reduce on a captured route",
           "all_reduce_rmsnorm_mnnvl": "AllReduceRmsNormMnnvl_v1{WORLD,N,SHOT,CLUSTER,EPS,FMA}: FlashInfer fused all-reduce + RMS norm",
           "all_gather": "AllGather_v1{WORLD,N}: vllm.all_gather of WORLD>2 [N] shards along the trailing axis = the [WORLD*N] concatenation (b1_tp2)",
           "embedding_shard": "EmbeddingShard_v1{VS,H,START}: one rank of VocabParallelEmbedding — shard[id-START] if START<=id<START+VS else 0 (b1_tp2)",
           "fp8_group_quant": "Fp8GroupQuant_v1{K,G}: _C.per_token_group_fp8_quant on one bf16 row -> (x_q e4m3[K], x_s f32[K/G]); per group absmax fmaxf-fold "
                              "from eps 1e-10, s = absmax/448 (div.rn), q = e4m3(clamp(f32(x)/s, ±448)); both outputs computed",
           "scaled_mm_fp8_block": "ScaledMmFp8BlockSharedScale_v1{K,N,G} (SCALE shared-per-block) or ScaledMmFp8Block_v1{K,N,G} (per-coordinate): blockwise "
                                  "_C.cutlass_scaled_mm on one token row -- per coordinate, ascending 128-tiles: temp = 4x HopperE4m3QgmmaDot32 from +0, "
                                  "acc = FFMA(temp, FMUL(x_s[kb], Ws[n//G][kb]), acc); cvt.rn.bf16.f32; shared-per-block forms each weight block's FMULs "
                                  "once; W e4m3 [N,K] + Ws f32 [N/G,K/G] Inputs",
           "layer_norm_aten": "LayerNormAten_v1{N,EPS}: aten.layer_norm on a bf16 row with bf16 weight and bias -- torch 2.13 "
                              "vectorized_layer_norm_kernel<bf16, float>: Welford over 128 threads / 4 warps, rsqrtf(var + eps), "
                              "bf16(w * (rstd * (x - mean)) + b)",
           "layer_norm_aten_value": "LayerNormAten_v2{N}(x, w, b, eps): LayerNormAten_v1's body with eps an input",
           "dsa_indexer_k_row": "DsaIndexerKRow_v1{D,ROPE,EPS}: one index-K cache slot of _fused_norm_rope_kernel -- f32 layer norm (w, b f32), "
                                "the interleaved RoPE of the first ROPE elements, DsaUe8m0Quant_v1{D} -> (e4m3[D], f32 scale)",
           "dsa_indexer_q": "DsaIndexerQ_v1{H,D,ROPE}: _fused_q_kernel's indexer outputs for one token -- per head the interleaved RoPE, "
                            "DsaUe8m0Quant_v1{D} and the weight scale(q) * D^-0.5 * H^-0.5 -> (e4m3[H,D], f32[H])",
           "dsa_mqa_logits": "DsaMqaLogits_v3{N,H,D}: DeepGEMM's Hopper fp8 MQA logits of one query over N index-K slots -- per head D/32 "
                             "HopperE4m3QgmmaDot32 steps, relu x weight summed per quad lane (the first two heads contracted), x the slot's scale",
           "dsa_topk_short_row": "DsaTopKShortRow_v1{TOPK}: the top-k of a row of at most TOPK keys (top_k_per_row_prefill, cooperative_topk) "
                                 "-- slot i = i < len ? i : -1",
           "dsa_rmsnorm": "DsaRMSNorm_v1{N,EPS}: _fused_norm_rope_kernel's cc 9.0 PTX reduction",
           "dsa_kv_rope": "RoPEHeadGptJ_v2{D}: _fused_norm_rope_kernel's interleaved k_pe head",
           "dsa_q_rope": "RoPEHeadGptJ_v2{D}: _fused_q_kernel's interleaved q_pe head",
           "cublas_splitk_gemm": "CublasSplitKGemm_v1{K,N,SPLIT_K,DOT}: vLLM-default cuBLAS BF16 linear at a captured split count",
           "batched_gemm": "BatchedGemm_v1{B,K,N,DOT}: vLLM-default cuBLAS batched BF16 MLA absorption",
           "ll_bf16_dotprod": "LlBf16DotprodGemm_v1{N,K,BS,VEC,FMA}: vLLM-default cuteDSL MoE gate for M <= 4",
           "ll_bf16_splitk": "LlBf16SplitKGemm_v1{N,K,SPLIT_K,TILE_K,WARPS,DOT}: vLLM-default cuteDSL MoE gate for 5 <= M <= 16",
           "sparse_mla": "SparseMla_v1{T,NH,DR,DV,BN,DOT,INV,SCALE=SM_SCALE}: FA3's sparse MLA (_vllm_fa3_C.fwd, FLASH_ATTN_MLA_SPARSE) of one "
                         "query token's NH heads over its T selected [kv_c | k_pe] rows -- the k16 wgmma chain over [q_rope | q_nope], "
                         "AttnBlock_v4's online softmax over BN-key blocks last to first, AttentionHead_v4's epilogue",
           "sparse_mla_split": "SparseMlaSplit_v1{T,NS,NH,DR,DV,BN,DOT,INV,SCALE=SM_SCALE}: FA3 sparse MLA at the token's default split count"})


def profile_records(vocab: Vocabulary) -> dict | None:
    """The `GEMM` / `ATTENTION` / `NORM`-maker records a v2-skeleton family (`Serve_v2`, `Serve_v3`) is bound with."""
    if vocab.name == "b1-eager-v2":
        return {"GEMM": b1.GEMM_AMPERE, "ATTENTION": b1.ATTENTION_FA2, "NORM": b1.norm_bundle}
    return None


# -- profile "dense2-eager-v1" [CB-A 2026-09-18, Gemma-2 head-to-head]: b1-eager-v3 + the generic elementwise / activation / softcap
# kinds the eager `gen_dense2_gemma2_2b` profile binds. [R15 dense] Those kinds are now part of `b1-eager-v3` itself (`_dense_kinds`):
# this name is kept for the R13 head-to-head artifacts and binds exactly the same kinds. --------------------------------------------
PROFILE_DENSE2_EAGER_V1 = Vocabulary("dense2-eager-v1", "profile", "dense2-eager-v1",
                                     kinds=dict(PROFILE_B1_EAGER_V3.kinds), notes=dict(PROFILE_B1_EAGER_V3.notes))


PROFILES = {"b1-eager": PROFILE_B1_EAGER, "b1-eager-v2": PROFILE_B1_EAGER_V2, "b1-eager-v3": PROFILE_B1_EAGER_V3,
            "dense2-eager-v1": PROFILE_DENSE2_EAGER_V1, None: PROFILE_B1_EAGER}


def vocabulary_for(mode: str, profile: str | None) -> Vocabulary:
    if mode == "reference":
        return REFERENCE_CPU_F32
    v = PROFILES.get(profile)
    if v is None:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(k for k in PROFILES if k)}")
    return v


__all__ = ["Vocabulary", "PROFILE_B1_EAGER", "REFERENCE_CPU_F32", "vocabulary_for", "PROFILES"]
