"""BuildProgram(φ) -> the Program the Build derives for a row, constructed ahead of time from a hand-written vocabulary.

The Build (`pipeline.build`) traces vLLM's model class on meta with torch.export and lets the binding rules translate each node;
this module writes the same Program down directly from φ (`phi.Phi`: the checkpoint's `config.json`, the declared target, the
wrapper shape and the sampling policy), with the same IR calls the derive makes: one `Builder` over the wrapper's user inputs and
the weights record (vLLM's parameters in module order, then its buffers), one call per row of each launch, the vocabulary's kinds
(`rules.vocab.PROFILE_B1_EAGER_V3.kind`) with the statics the rules select through `frontend.targets`, the flat body wrapped as
`Derived[<wrapper>[<model class>]]_v1{SRC}` and inlined by `Program`.  Nothing here traces, loads vLLM or reads a Build output.

Vocabulary (v1, Llama-family decoders, dense or fused-MoE, under the bi-eager profile): Embedding, RMSNormTriton (layer-0 input
norm, and the q/k norm: per head, or over all heads), Gemm (target-keyed; the q/k/v projection's bias, when the family's rule gives
it one, fused as the target's GemmBias where its record says so, else a BiasAdd per row), RoPE (all q rows, then all k rows),
Attention (target-keyed; one call per query row over the layer's history), RMSNormFusedCuda (post-attention / next-layer / final
norm), SiluMul, the fused-MoE block in its indexed-read construction (router Gemm, MoeRouterTopK per row, MoeExpertGemm / SiluMul /
MoeExpertGemmW per (token, slot) pair, MoeSum per row); for Gemma-2, the embedding's normalizer scale, GemmaRMSNorm's unfused (1 +
w) chain (every norm, the two extra MLP norms included), AttentionSoftcap, GeluTanhMul and the final logits' tanh soft-cap; and the
token select: greedy TokenSelect, or the seeded temperature + top-p GumbelTopPTokenSelect with its per-event constants (the split
count a `splits` Input entry, or the single request's constant S). Anything φ asks for outside it raises `Unexpressible` naming the
fact (a result, never a silent approximation).

Restored lineage: `verity_capture.experimental.aot.build` (`aot-build/eager/v8`, deleted in 528f9d21f) built the capture fold's
`Captured_v1{PROFILE, WORKLOAD}` Program; the Build of record now derives `Derived[vLLM*]` Programs, so the emitter targets those.

This module is NEVER on a record path: a predicted Program is not written into a Build row, not published as a Build Attempt and
not read by the Build, Match or Commit (`program.frontend.inputs_trace` lists `verity_vllm/predict/` as a forbidden input).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, concat
from verity.ml.circuits.types import Array, Record, Type, Value

from .phi import Phi

BUILDER_ID = "predict-build/derived/v1"
I32 = Value(32)
BF16 = Value(16)

#: the wrapper kinds of `vllm_meta.make_wrapper` per sampling kind: DERIVE_SOURCE (the SRC class) and the derive name
WRAPPERS = {("step", "greedy"): ("verity-vllm/engine-step/v1", "vLLMStep"),
            ("request", "greedy"): ("verity-vllm/greedy-request/v1", "vLLMRequest"),
            ("step", "stochastic"): ("verity-vllm/stochastic-engine-step/v1", "vLLMStep"),
            ("request", "stochastic"): ("verity-vllm/stochastic-request/v1", "vLLMRequest")}
I64 = Value(64)


#: the family facts (`engine.profiles.family_facts`) `model_shape` reads; a family with any other refuses
_FACTS_READ = ("QK_NORM", "QK_NORM_EPS", "RENORMALIZE", "CLIP_QKV", "SHARED_EXPERTS", "tie_word_embeddings", "attn_logit_softcapping",
               "final_logit_softcapping", "query_pre_attn_scalar", "sliding_window", "hidden_act", "embed_normalizer", "rmsnorm_plus_one",
               "layer_norms", "extra_layer_norms")
#: the activations of the gated MLP: the `*Mul` kind's role per `hidden_act`
_ACTS = {"silu": "silu_mul", "gelu_pytorch_tanh": "gelu_tanh_mul"}


class Unexpressible(Exception):
    """φ asks for a structure the vocabulary cannot express.  The message names the fact precisely."""


@dataclass(frozen=True)
class ModelShape:
    arch: str
    model_class: str
    H: int
    V: int
    NL: int
    NH: int
    KVH: int
    D: int
    I: int
    EPS: float
    tie: bool
    max_pos: int
    rot: int
    qkv_bias: bool = False
    window: int | None = None
    qk_norm: str | None = None
    E: int = 0
    TOPK: int = 0
    renorm: bool = False
    act: str = "silu"
    norm_chain: bool = False
    extra_norms: bool = False
    embed_scale: bool = False
    attn_cap: float | None = None
    final_cap: float | None = None

    @property
    def QKV(self) -> int:
        return (self.NH + 2 * self.KVH) * self.D


def model_shape(hf: dict) -> ModelShape:
    """The shape facts of an HF `config.json` the vocabulary reads, through the two declared homes of model facts: the family facts
    table (`engine.profiles.family_facts`: vLLM's class and constructor kind per `model_type`) and `program.model.config_of` (the
    static configuration, with the family's bias rule and the checkpoint's quantization).  The vocabulary is the dense and the
    fused-MoE (indexed-read) blocks with no per-family facts beyond the q/k RMSNorm (per head, or over all heads as `qk_norm` "all"),
    the router's renormalization and Gemma-2's (the embedding normalizer, the (1 + w) norm chain with the two extra MLP norms, the
    attention and final logit soft-caps, the tanh-GELU MLP, `query_pre_attn_scalar` = head_dim): a family with any other (another norm
    variant, LayerNorm, shared experts), a linear bias other than the q/k/v one a family's rule gives, an activation other than SiLU
    or a family-declared one, a partial rotary width or a rotary scaling without an analytic table raises `Unexpressible`.  A
    sliding window is kept as `window` (the keys FA2's `window_size_left` admits) for the families whose vLLM code hands it to FA2
    (the `dense2` patterns); `build_program` refuses a history longer than it."""
    from verity_vllm.engine.profiles.family_facts import FAMILY_FACTS, resolve_facts
    from verity_vllm.engine.profiles.generic import effective_sliding_window
    from verity_vllm.program import model as analytic
    family = str(hf.get("model_type", "")).lower()
    ff = FAMILY_FACTS.get(family)
    if ff is None:
        raise Unexpressible(f"model_type {family!r}: not in the family facts table")
    rules = dict(ff.get("rules") or {})
    facts = []
    arch = ff.get("arch")
    if arch not in ("dense", "moe"):
        facts.append(f"family {family!r} constructor {arch!r} (the dense and MoE blocks only)")
    other = sorted(set(ff.get("facts") or {}) - set(_FACTS_READ))
    if other or rules.get("qk_norm") not in (None, "head"):
        facts.append(f"family {family!r} per-family facts {other}, q/k norm {rules.get('qk_norm')}")
    try:
        C, _ = analytic.config_of(hf, family_rules={family: rules})
        fv = {} if other else resolve_facts(hf, C)[0]
    except (KeyError, ValueError) as e:
        raise Unexpressible(f"config.json: {e}") from None
    qk_norm = "head" if rules.get("qk_norm") == "head" else "all" if fv.get("QK_NORM") else None
    if fv.get("QK_NORM") and float(fv.get("QK_NORM_EPS", C["EPS"])) != float(C["EPS"]):
        facts.append(f"a q/k norm epsilon {fv['QK_NORM_EPS']} other than the layer norms' {C['EPS']}")
    if fv.get("CLIP_QKV") is not None:
        facts.append(f"clip_qkv {fv['CLIP_QKV']}")
    if int(fv.get("SHARED_EXPERTS") or 0):
        facts.append(f"shared experts (shared_expert_intermediate_size {fv['SHARED_EXPERTS']})")
    E = TOPK = 0
    I = int(C["I"])
    if arch == "moe":
        from verity_vllm.program.frontend.rules.vllm_moe import TOPK_SOFTMAX_E_TESTED
        E, TOPK = int(C.get("NUM_EXPERTS") or 0), int(C.get("TOPK") or 0)
        if ff.get("moe_width_key"):
            I = int(hf[ff["moe_width_key"]])
        if E not in TOPK_SOFTMAX_E_TESTED or not 1 <= TOPK <= E:
            facts.append(f"{E} experts, top-{TOPK} (the router is difftested for E in {sorted(TOPK_SOFTMAX_E_TESTED)})")
        if hf.get("mlp_only_layers") or int(hf.get("decoder_sparse_step") or 1) != 1:
            facts.append("dense MLP layers among the MoE ones (mlp_only_layers / decoder_sparse_step)")
    qkv_bias = bool(C.get("BIAS"))
    if qkv_bias and "qkv_bias" not in rules:
        facts.append("attention_bias (q/k/v and o_proj biases)")
    if hf.get("mlp_bias"):
        facts.append("MLP biases")
    window = effective_sliding_window(hf)
    if window and "dense2" not in (ff.get("patterns") or ()):
        facts.append(f"sliding window {window} (family {family!r} has no inert-window patterns)")
    act = fv.get("hidden_act") or hf.get("hidden_act", hf.get("hidden_activation", "silu"))
    if act not in _ACTS or (act != "silu" and "hidden_act" not in fv):
        facts.append(f"hidden_act {act!r} (SiluMul, or GeluTanhMul where the family declares it)")
    norm_chain = "unfused-norm-chains" in (ff.get("patterns") or ())
    if norm_chain != bool(fv.get("rmsnorm_plus_one")):
        facts.append("a norm other than the fused kernels or the unfused (1 + w) chain of the families that declare it")
    extra = tuple(fv.get("extra_layer_norms") or ())
    if extra not in ((), ("ln_pre_mlp", "ln_post_mlp")) or int(fv.get("layer_norms") or 2) != 2 + len(extra) or (extra and not norm_chain):
        facts.append(f"layer norms {fv.get('layer_norms')} {list(extra)} (input + post-attention, or with the pre/post MLP pair of the unfused norm chain)")
    qpas = fv.get("query_pre_attn_scalar")
    if qpas is not None and float(qpas) != float(C["D"]):
        facts.append(f"query_pre_attn_scalar {qpas} (the attention kinds fold the scale head_dim^-0.5, head_dim {C['D']})")
    caps = [float(fv[k]) if fv.get(k) else None for k in ("attn_logit_softcapping", "final_logit_softcapping")]
    if C.get("FP8_BLOCK"):
        facts.append(f"block-scaled FP8 checkpoint (G={C['FP8_BLOCK']})")
    if analytic.rotary_dim(C) != C["D"]:
        facts.append(f"partial rotary width {analytic.rotary_dim(C)} of head_dim {C['D']}")
    try:
        analytic._inv_freq(C)
    except NotImplementedError as e:
        facts.append(str(e))
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))
    cls = re.match(r"\w+", ff["vllm_class"]).group(0)
    return ModelShape(arch=family, model_class=cls, H=int(C["H"]), V=int(C["V"]), NL=int(C["NL"]), NH=int(C["NH"]), KVH=int(C["KVH"]),
                      D=int(C["D"]), I=I, EPS=float(C["EPS"]), tie=bool(fv.get("tie_word_embeddings", C["tie_word_embeddings"])),
                      max_pos=int(C["MAX_POS"]), rot=int(C["D"]), qkv_bias=qkv_bias, window=int(window) - 1 if window else None,
                      qk_norm=qk_norm, E=E, TOPK=TOPK, renorm=bool(fv.get("RENORMALIZE")), act=act, norm_chain=norm_chain,
                      extra_norms=bool(extra), embed_scale="embed_normalizer" in fv, attn_cap=caps[0], final_cap=caps[1])


#: vLLM's `DEFAULT_VOCAB_PADDING_SIZE`: the vocabulary-parallel embedding and LM head pad the vocabulary to a multiple of it
VOCAB_PADDING = 64


def rank_shape(C: ModelShape, world: int) -> ModelShape:
    """One rank's slice of C under vLLM's tensor parallelism: the query and key/value heads and the MLP width split `world` ways
    (`QKVParallelLinear`, `MergedColumnParallelLinear` then `RowParallelLinear`; under the fused MoE each expert's width, the router
    and the experts whole), the hidden size, the norms and the rotary table whole.  V stays the full vocabulary: the embedding and the LM head take V / world rows each (`VocabParallelEmbedding`,
    `ParallelLMHead`) and the token select reads the gathered row.  A split vLLM pads or replicates is `Unexpressible`."""
    if world == 1:
        return C
    facts = []
    replicated_kv = C.KVH < world and world % C.KVH == 0          # QKVParallelLinear: a key/value head on world // KVH ranks
    if C.NH % world or (C.KVH % world and not replicated_kv):
        facts.append(f"{C.NH} query and {C.KVH} key/value heads over {world} ranks (an even split, or each key/value head on "
                     f"world // KVH ranks)")
    if C.I % world:
        facts.append(f"an MLP width of {C.I} over {world} ranks")
    if C.V % VOCAB_PADDING or C.V % world:
        facts.append(f"a vocabulary of {C.V} over {world} ranks (padded to a multiple of {VOCAB_PADDING}: the padding rows' trim)")
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))
    return replace(C, NH=C.NH // world, KVH=1 if replicated_kv else C.KVH // world, I=C.I // world)


def _arr(shape: tuple[int, ...], w: int) -> Type:
    t: Type = Value(w)
    for n in reversed(shape):
        t = Array(int(n), t)
    return t


def _arr_shape(t: Type) -> tuple[int, ...]:
    shape = []
    while isinstance(t, Array):
        shape.append(t.n)
        t = t.elem
    return tuple(shape)


def weight_leaves(C: ModelShape, world: int = 1) -> list[tuple[str, Type]]:
    """The weights record of the model class on meta, as `derive._weights_of` lists it: the distinct parameters in module order
    (the tied `lm_head.weight` is the embedding's leaf), then the distinct buffers (the embedding's 0-d bf16 `normalizer` where the
    family scales it, the one shared rotary `cos_sin_cache` [max_pos, rot] bf16, each attention layer's four f32 scalar scales).  The
    field names are informational: the descriptor carries types only.  Under `world` ranks each is one rank's shard (`rank_shape`)."""
    bf = lambda *s: _arr(s, 16)  # noqa: E731
    V = C.V // world
    C = rank_shape(C, world)
    out: list[tuple[str, Type]] = [("model.model.embed_tokens.weight", bf(V, C.H))]
    for L in range(C.NL):
        p = f"model.model.layers.{L}."
        out.append((p + "self_attn.qkv_proj.weight", bf(C.QKV, C.H)))
        if C.qkv_bias:
            out.append((p + "self_attn.qkv_proj.bias", bf(C.QKV)))
        if C.qk_norm == "all":
            out += [(p + "self_attn.q_norm.weight", bf(C.NH * C.D * world)), (p + "self_attn.k_norm.weight", bf(C.KVH * C.D * world))]
        out.append((p + "self_attn.o_proj.weight", bf(C.H, C.NH * C.D)))
        if C.qk_norm == "head":
            out += [(p + "self_attn.q_norm.weight", bf(C.D)), (p + "self_attn.k_norm.weight", bf(C.D))]
        if C.E:
            out += [(p + "mlp.gate.weight", bf(C.E, C.H)), (p + "mlp.experts.w13_weight", bf(C.E, 2 * C.I, C.H)),
                    (p + "mlp.experts.w2_weight", bf(C.E, C.H, C.I))]
        else:
            out += [(p + "mlp.gate_up_proj.weight", bf(2 * C.I, C.H)), (p + "mlp.down_proj.weight", bf(C.H, C.I))]
        out += [(p + "input_layernorm.weight", bf(C.H)), (p + "post_attention_layernorm.weight", bf(C.H))]
        if C.extra_norms:
            out += [(p + "pre_feedforward_layernorm.weight", bf(C.H)), (p + "post_feedforward_layernorm.weight", bf(C.H))]
    out.append(("model.model.norm.weight", bf(C.H)))
    if not C.tie:
        out.append(("model.lm_head.weight", bf(V, C.H)))
    if C.embed_scale:
        out.append(("model.model.normalizer", bf(1)))
    out.append(("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache", bf(C.max_pos, C.rot)))
    for L in range(C.NL):
        out += [(f"model.model.layers.{L}.self_attn.attn.{s}", Value(32)) for s in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")]
    return out


def _ptypes(spec) -> tuple:
    params, _ = spec.signature()
    return tuple(t for _, t in params)


class _Specs:
    """The vocabulary's kinds for C on the declared target, bound once each (the same objects the rules' `ctx.kind` returns)."""

    def __init__(self, C: ModelShape, target, world: int = 1, rank: int = 0, M: int = 1) -> None:
        from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as VOC
        from verity_vllm.program.frontend import targets as TG
        VS, C = C.V // world, rank_shape(C, world)
        self.C, self.tp, self.TG, self.voc = C, target, TG, VOC
        self.accepted = TG.uses_accepted_kinds(target)
        self.dot = TG.tensor_core_dot(target)
        if world == 1:
            self.embedding = self.kind("embedding", V=C.V, H=C.H)
            self.all_reduce = self.all_gather = None
        else:
            self.embedding = self.kind("embedding_shard", VS=VS, H=C.H, START=rank * VS)
            self.all_reduce, self.all_gather = self.kind("all_reduce2", N=C.H), self.kind("all_gather2", N=VS)
        self.norm0 = self.kind("rmsnorm", N=C.H, EPS=C.EPS, TARGET=target, M=M)
        self.fused = self.kind("rmsnorm_fused", N=C.H, EPS=C.EPS, TARGET=target, M=M)
        self.qkv, self.o = self.gemm(C.H, C.QKV), self.gemm(C.NH * C.D, C.H)
        self.qkv_bias = self.qkv_fused = None
        if C.qkv_bias:
            role = self.gemm_bias_role()
            if role is None:
                self.qkv_bias = self.kind("bias_add", N=C.QKV)
            else:
                self.qkv = self.qkv_fused = self.kind(role[0], K=C.H, N=C.QKV, **role[1])
        self.head_norm = self.kind("rmsnorm", N=C.D, EPS=C.EPS, TARGET=target, M=M * C.NH) if C.qk_norm == "head" else None
        if C.qk_norm == "all":
            self.q_norm, self.k_norm = (
                self.kind("rmsnorm", N=n * C.D * world, EPS=C.EPS, TARGET=target, M=M) for n in (C.NH, C.KVH)
            )
            self.q_gather, self.k_gather = (self.kind("all_gather2", N=n * C.D) for n in (C.NH, C.KVH)) if world > 1 else (None, None)
        self.gate_up, self.down, self.logits = self.gemm(C.H, 2 * C.I), self.gemm(C.I, C.H), self.gemm(C.H, VS)
        if C.E:
            self.moe()
        self.rope_q, self.rope_k = self.kind("rope", NHEADS=C.NH, D=C.D), self.kind("rope", NHEADS=C.KVH, D=C.D)
        self.silu = self.kind(_ACTS[C.act], I=C.I)
        self.token_select = self.kind("token_select", V=C.V)
        if C.norm_chain:
            self.chain = {r: self.kind(r, N=C.H) for r in ("square_bf16", "square_f32", "mean_triton", "add_widened_bf16", "scale_row_bf16",
                                                           "scale_row_f32", "mul_vec_f32", "narrow_f32_bf16")}
            self.chain.update(plus_one=self.kind("add_scalar_bf16", N=C.H, C=1.0), eps=self.kind("add_scalar_f32", N=1, C=C.EPS),
                              rsqrt=self.kind("rsqrt_f32", N=1))
            self.weight_once = target.effective_weight_only_calls() == "once"
        if C.embed_scale:
            self.embed_scale = self.kind("scale_tensor", N=C.H)
        if C.final_cap is not None:
            self.cap = [self.kind("bf16_div_scalar", N=C.V, C=C.final_cap), self.kind("bf16_tanh", N=C.V),
                        self.kind("bf16_mul_scalar", N=C.V, C=C.final_cap)]
        self._attn: dict[tuple, Any] = {}
        self._const: dict[tuple, Any] = {}

    def moe(self) -> None:
        """The fused-MoE block's kinds under the target's declared construction (`rules.vllm_moe`): the router GEMM, the top-k
        softmax (in selection order under `indexed-read-ordered`; the renormalizing form when the family renormalizes), the per-pair
        expert GEMMs (the target's k16 step as `DOT` where `frontend.targets.moe_expert_dot` names one) and the routed sum."""
        from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, INDEXED_READ_ORDERED, TOPK_SOFTMAX_VPT
        import torch
        C = self.C
        c = self.tp.effective_moe_construction()
        if c not in (INDEXED_READ, INDEXED_READ_ORDERED):
            raise Unexpressible(f"vocabulary cannot express: moe_construction {c!r} (indexed-read only)")
        self.router = self.gemm(C.H, C.E)
        self.topk = self.kind("moe_router_topk" + ("_ordered" if c == INDEXED_READ_ORDERED else "") + ("_norm" if C.renorm else ""),
                              E=C.E, TOPK=C.TOPK, VPT=TOPK_SOFTMAX_VPT[torch.bfloat16])
        try:
            dot = self.TG.moe_expert_dot(self.tp)
        except self.TG.UnregisteredTarget as e:
            raise Unexpressible(f"numerics_unregistered: {e}") from None
        extra = {} if dot is None else {"DOT": dot}
        sfx = "" if dot is None else "_dot"
        self.expert_up = self.kind("moe_expert_gemm" + sfx, E=C.E, K=C.H, N=2 * C.I, **extra)
        self.expert_down = self.kind("moe_expert_gemm_w" + sfx, E=C.E, K=C.I, N=C.H, **extra)
        self.moe_sum = self.kind("moe_sum", TOPK=C.TOPK, H=C.H)

    def stochastic_select(self, S):
        """`GumbelTopPTokenSelect[vllm]`'s kind: the select over a derived `splits` operand, or at the single request's constant S."""
        if S == "derived":
            return self.kind("token_select_stochastic", V=self.C.V, CONSTRUCTION=self.tp.effective_sampler_construction())
        return self.kind("token_select_stochastic_split", V=self.C.V, S=int(S))

    def const(self, W: int, BITS: int):
        key = (W, BITS)
        if key not in self._const:
            self._const[key] = self.kind("const", W=W, BITS=BITS)
        return self._const[key]

    def kind(self, role: str, **statics):
        try:
            fn = self.voc.kind(role, **statics)
        except self.TG.UnregisteredTarget as e:
            raise Unexpressible(f"numerics_unregistered: {e}") from None
        if fn is None:
            raise Unexpressible(f"vocabulary {self.voc.name!r} has no kind for role {role!r} with {statics}")
        return fn

    def gemm(self, K: int, N: int):
        return self.kind("gemm", K=K, N=N) if self.accepted else self.kind("gemm_dot", K=K, N=N, DOT=self.dot)

    def gemm_bias_role(self) -> tuple[str, dict] | None:
        """The biased linear's one-Call kind as `TritonGemmRule` picks it (the bias add the GEMM's only reader): `GemmBias_v1` under
        the accepted kinds, `GemmBias_v2{DOT}` where the target's record names a biased-linear step (`frontend.targets.gemm_bias_dot`,
        on the trees that have it), else None: the GEMM, then `BiasAdd_v1` per row."""
        if self.accepted:
            return "gemm_bias", {}
        dot = getattr(self.TG, "gemm_bias_dot", lambda tp: None)(self.tp)
        return None if dot is None else ("gemm_bias_dot", {"DOT": dot})

    def attention(self, P: int, row: int, own_rows: int, launch_sq: int):
        """`UnifiedAttentionWithOutput[vllm]`'s kind for the query row at history length P (row = its index among the sequence's
        query rows in the launch, own_rows = their number, launch_sq = the launch's max_seqlen_q)."""
        C, TG, tp = self.C, self.TG, self.tp
        key = (P, row, own_rows, launch_sq)
        fn = self._attn.get(key)
        if fn is not None:
            return fn
        cap = {} if C.attn_cap is None else {"CAP": C.attn_cap}
        if self.accepted:
            fn = self.kind("attention_softcap" if cap else "attention", T=P, NH=C.NH, KVH=C.KVH, D=C.D, **cap)
        else:
            if cap and not TG.fa2_check_inf(tp):
                raise Unexpressible(f"vocabulary cannot express: a soft-capped attention on {tp.tensor_core_family} without FA2's "
                                    "per-iteration Check_inf (the only registered target-keyed softcap chain)")
            g = TG.attention_geometry(tp, C.D, launch_sq, C.NH, C.KVH)
            extra = dict(cap)
            if TG.fa3_check_inf(tp):
                extra["MASKED_FROM"] = TG.fa3_masked_from(tp, P, C.NH, C.KVH, C.D, launch_sq, row)
            if TG.fa2_check_inf(tp):
                extra["FA2_MASKED_FROM"] = TG.fa2_masked_from(tp, P, C.D, launch_sq, row, own_rows)
            fn = self.kind("attention_dot", T=P, NH=C.NH, KVH=C.KVH, D=C.D, BN=g["BN"], DOT=g["DOT"], INV=g["INV"], **extra)
        self._attn[key] = fn
        return fn


@dataclass
class BuildResult:
    wrapper: str
    program: Any
    digest: str
    desc: dict | None
    function: str
    n_root_nodes: int
    timings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"wrapper": self.wrapper, "function": self.function, "digest": self.digest,
                "n_root_nodes": self.n_root_nodes, "gates": getattr(self.program, "gates", None), "timings": self.timings}


class _Model:
    """The forward of the model class over one call's rows, emitted into `B` with the per-layer K/V history carried across calls
    (the implicit KV cache the attention rule makes explicit).  C is one rank's shape (`rank_shape`); a tensor-parallel rank's
    collectives read `peers`, the other rank's partials, one Input per collective in call order."""

    def __init__(self, B: Builder, specs: _Specs, W: Coll, C: ModelShape, peers: list[Coll] = (), rank: int = 0) -> None:
        self.B, self.S, self.W, self.C = B, specs, W, C
        self.peers, self.rank, self.n_peer = list(peers), rank, 0
        n = 0
        self.embed = W[n]
        n += 1
        self.layers = []
        for _ in range(C.NL):
            names = ["qkv"] + (["qkv_b"] if C.qkv_bias else []) + (["q_norm", "k_norm"] if C.qk_norm == "all" else []) + ["o"] + \
                (["q_norm", "k_norm"] if C.qk_norm == "head" else []) + (["gate_w", "w13", "w2"] if C.E else ["gate_up", "down"]) + \
                ["ln1", "ln2"] + (["pre_ff", "post_ff"] if C.extra_norms else [])
            self.layers.append({k: W[n + i] for i, k in enumerate(names)})
            n += len(names)
        self.norm = W[n]
        n += 1
        self.lm_head = self.embed if C.tie else W[n]
        n += 0 if C.tie else 1
        self.normalizer = W[n] if C.embed_scale else None
        n += 1 if C.embed_scale else 0
        self.cos_sin = W[n]
        self.hist: list[tuple[list, list]] = [([], []) for _ in range(C.NL)]
        self.lp_of: dict[int, int] = {}
        self.plus_one: dict[int, Coll] = {}

    def call(self, fn, *args) -> Coll:
        return self.B.call(fn, *args)

    def collective(self, fn, own: list[Coll]) -> list[Coll]:
        """`fn` (`AllReduce2` / `AllGather2`) per row over this rank's partial and the next peer Input's row, in rank order
        (`rules.vllm_bindings.collectives`), or the rows themselves on one rank."""
        if not self.peers:
            return own
        peer, n = self.peers[self.n_peer], own[0].type.leaves
        self.n_peer += 1
        p = _ptypes(fn)[0]
        parts = [(x.refs, peer.refs.slice(i * n, (i + 1) * n)) for i, x in enumerate(own)]
        return [self.call(fn, Coll(p, concat(list(a if self.rank == 0 else a[::-1])))) for a in parts]

    def norm_all_heads(self, parts: list[tuple]) -> list[list]:
        """The q and k norms over all heads (`OlmoeAttention._apply_qk_norm`), each part (rows, width, gather, norm, weight): each
        row normed whole, which under tensor parallelism is both parts' rows all-gathered across the ranks, both normed, then this
        rank's slice of each normed row."""
        rows = [[c.refs for c in self.collective(g, [Coll(_arr((n,), 16), r) for r in x])] if g else x for x, n, g, _, _ in parts]
        normed = [[self.call(fn, Coll(_ptypes(fn)[0], r), Coll(_ptypes(fn)[1], w.refs)).refs for r in x]
                  for x, (_, _, _, fn, w) in zip(rows, parts)]
        return [[r.slice(self.rank * n, (self.rank + 1) * n) for r in x] if g else x for x, (_, n, g, _, _) in zip(normed, parts)]

    def forward(self, toks: list[Coll], pos0: int) -> list[Coll]:
        """Token rows (I32 values) at positions pos0.. -> the final-normed hidden rows."""
        S, C, M = self.S, self.C, len(toks)
        pe = _ptypes(S.embedding)
        pq, po, pgu, pd = _ptypes(S.qkv), _ptypes(S.o), _ptypes(S.gate_up), _ptypes(S.down)
        prq, prk, ps = _ptypes(S.rope_q), _ptypes(S.rope_k), _ptypes(S.silu)
        q_w, k_w, v_w = C.NH * C.D, C.KVH * C.D, C.KVH * C.D
        h = self.collective(S.all_reduce, [self.call(S.embedding, Coll(pe[0], t.refs), Coll(pe[1], self.embed.refs)) for t in toks])
        if C.embed_scale:
            pz = _ptypes(S.embed_scale)
            h = [self.call(S.embed_scale, Coll(pz[0], h[i].refs), Coll(pz[1], self.normalizer.refs)) for i in range(M)]
        x, _ = self.rms(h, self.layers[0]["ln1"])
        res = [h[i].refs for i in range(M)]
        d = None
        for L in range(C.NL):
            lw = self.layers[L]
            if L > 0:
                x, res = self.rms(d, lw["ln1"], res)
            if S.qkv_fused is not None:
                qkv = [self.call(S.qkv, Coll(pq[0], x[i].refs), Coll(pq[1], lw["qkv"].refs), Coll(pq[2], lw["qkv_b"].refs)) for i in range(M)]
            else:
                qkv = [self.call(S.qkv, Coll(pq[0], x[i].refs), Coll(pq[1], lw["qkv"].refs)) for i in range(M)]
            if S.qkv_bias is not None:
                pb = _ptypes(S.qkv_bias)
                qkv = [self.call(S.qkv_bias, Coll(pb[0], qkv[i].refs), Coll(pb[1], lw["qkv_b"].refs)) for i in range(M)]
            cs = [Coll(prq[1], self.cos_sin.refs.slice((pos0 + i) * C.rot, (pos0 + i + 1) * C.rot)) for i in range(M)]
            q_in = [qkv[i].refs.slice(0, q_w) for i in range(M)]
            k_in = [qkv[i].refs.slice(q_w, q_w + k_w) for i in range(M)]
            if C.qk_norm == "head":
                q_in, k_in = self.head_norms(q_in, C.NH, lw["q_norm"]), self.head_norms(k_in, C.KVH, lw["k_norm"])
            elif C.qk_norm == "all":
                q_in, k_in = self.norm_all_heads([(q_in, q_w, S.q_gather, S.q_norm, lw["q_norm"]),
                                                  (k_in, k_w, S.k_gather, S.k_norm, lw["k_norm"])])
            q = [self.call(S.rope_q, Coll(prq[0], q_in[i]), cs[i]) for i in range(M)]
            k = [self.call(S.rope_k, Coll(prk[0], k_in[i]), Coll(prk[1], cs[i].refs)) for i in range(M)]
            kh, vh = self.hist[L]
            P0 = len(kh)
            if P0 == 0:
                self.lp_of[L] = M
            own_rows = M if P0 == 0 else 1
            launch_sq = own_rows
            for i in range(M):
                kh.append(k[i].refs)
                vh.append(qkv[i].refs.slice(q_w + k_w, q_w + k_w + v_w))
            attn = []
            for i in range(M):
                P = P0 + i + 1
                fn = S.attention(P, i if P0 == 0 else 0, own_rows, launch_sq)
                pa = _ptypes(fn)
                attn.append(self.call(fn, Coll(pa[0], q[i].refs), Coll(pa[1], concat(kh[:P])), Coll(pa[2], concat(vh[:P]))))
            o = self.collective(S.all_reduce, [self.call(S.o, Coll(po[0], attn[i].refs), Coll(po[1], lw["o"].refs)) for i in range(M)])
            if C.extra_norms:
                o, _ = self.rms(o, lw["ln2"])
                x, res = self.rms(o, lw["pre_ff"], res)
            else:
                x, res = self.rms(o, lw["ln2"], res)
            if C.E:
                d = self.collective(S.all_reduce, self.moe(x, lw))
            else:
                gu = [self.call(S.gate_up, Coll(pgu[0], x[i].refs), Coll(pgu[1], lw["gate_up"].refs)) for i in range(M)]
                a = [self.call(S.silu, Coll(ps[0], gu[i].refs)) for i in range(M)]
                d = self.collective(S.all_reduce, [self.call(S.down, Coll(pd[0], a[i].refs), Coll(pd[1], lw["down"].refs)) for i in range(M)])
            if C.extra_norms:
                d, _ = self.rms(d, lw["post_ff"])
        return self.rms(d, self.norm, res)[0]

    def rms(self, xs: list[Coll], w: Coll, res: list | None = None) -> tuple[list[Coll], list | None]:
        """The RMSNorm over the rows `xs` with weight `w`, after the residual add when `res` is given -> (the normed rows, the new
        residual rows or None): the fused kernels (`RMSNormTriton` alone, `RMSNormFusedCuda` with the residual), or GemmaRMSNorm's
        `forward_native` as `rules.vllm_bindings.norm_chain` binds it: (1 + w) (once per program under the target's
        `weight_only_calls = once`), [the widened residual add and its narrowing,] x², the mean, + eps, rsqrt, the row scale, the weight
        multiply and the narrowing, each over every row."""
        S, M = self.S, len(xs)
        if not self.C.norm_chain:
            if res is None:
                p = _ptypes(S.norm0)
                return [self.call(S.norm0, Coll(p[0], x.refs), Coll(p[1], w.refs)) for x in xs], None
            p = _ptypes(S.fused)
            f = [self.call(S.fused, Coll(p[0], xs[i].refs), Coll(p[1], res[i]), Coll(p[2], w.refs)) for i in range(M)]
            return [f[i][0] for i in range(M)], [f[i][1].refs for i in range(M)]
        K = S.chain
        call = lambda fn, *a: self.call(fn, *(Coll(t, r) for t, r in zip(_ptypes(fn), a)))  # noqa: E731
        w1 = self.plus_one.get(id(w))
        if w1 is None:
            w1 = call(K["plus_one"], w.refs)
            if S.weight_once:
                self.plus_one[id(w)] = w1
        s, new_res = [x.refs for x in xs], None
        if res is not None:
            s = [call(K["add_widened_bf16"], s[i], res[i]).refs for i in range(M)]
            new_res = [call(K["narrow_f32_bf16"], s[i]).refs for i in range(M)]
        sq, sc = (K["square_bf16"], K["scale_row_bf16"]) if res is None else (K["square_f32"], K["scale_row_f32"])
        v = [call(sq, s[i]) for i in range(M)]
        m = [call(K["mean_triton"], v[i].refs) for i in range(M)]
        e = [call(K["eps"], m[i].refs) for i in range(M)]
        r = [call(K["rsqrt"], e[i].refs) for i in range(M)]
        y = [call(sc, s[i], r[i].refs) for i in range(M)]
        y = [call(K["mul_vec_f32"], y[i].refs, w1.refs) for i in range(M)]
        return [call(K["narrow_f32_bf16"], y[i].refs) for i in range(M)], new_res

    def moe(self, x: list[Coll], lw: dict) -> list[Coll]:
        """The fused-MoE block over the rows `x` as `rules.vllm_moe` binds its launches: each row's router logits and top-k, then
        each (token, slot) pair p = t * TOPK + k in pair order: the w13 expert GEMM (expert id ids[t][k]), the activation, the w2
        expert GEMM with the routed weight w[t][k]; then each row's sum of its TOPK down rows."""
        S, C, M, K = self.S, self.C, len(x), self.C.TOPK
        pr, pt, pu, pa, pw, pm = (_ptypes(fn) for fn in (S.router, S.topk, S.expert_up, S.silu, S.expert_down, S.moe_sum))
        lg = [self.call(S.router, Coll(pr[0], x[t].refs), Coll(pr[1], lw["gate_w"].refs)) for t in range(M)]
        r = [self.call(S.topk, Coll(pt[0], lg[t].refs)) for t in range(M)]
        ids = [r[p // K][1].refs.slice(p % K, p % K + 1) for p in range(M * K)]
        ws = [r[p // K][0].refs.slice(p % K, p % K + 1) for p in range(M * K)]
        up = [self.call(S.expert_up, Coll(pu[0], x[p // K].refs), Coll(pu[1], ids[p]), Coll(pu[2], lw["w13"].refs)) for p in range(M * K)]
        act = [self.call(S.silu, Coll(pa[0], up[p].refs)) for p in range(M * K)]
        down = [self.call(S.expert_down, Coll(pw[0], act[p].refs), Coll(pw[1], ids[p]), Coll(pw[2], ws[p]), Coll(pw[3], lw["w2"].refs))
                for p in range(M * K)]
        return [self.call(S.moe_sum, Coll(pm[0], concat([down[t * K + k].refs for k in range(K)]))) for t in range(M)]

    def head_norms(self, rows: list, heads: int, weight: Coll) -> list:
        """The per-head q/k RMSNorm over every row's heads, rows major (one launch over [rows, heads, D]) -> each row's normed heads."""
        fn, D = self.S.head_norm, self.C.D
        ph = _ptypes(fn)
        out = [self.call(fn, Coll(ph[0], r.slice(h * D, (h + 1) * D)), Coll(ph[1], weight.refs)) for r in rows for h in range(heads)]
        return [concat([out[i * heads + h].refs for h in range(heads)]) for i in range(len(rows))]

    def sample(self, hidden_row: Coll, event: int, pos: int, policy: "_Stochastic | None" = None) -> Coll:
        """The token of sampling event `event` from the logits of `hidden_row` (at position `pos`): greedy `TokenSelect`, or the seeded
        temperature + top-p select with its per-row constants in operand order (top_p, pos, temperature[, S])."""
        S = self.S
        pl = _ptypes(S.logits)
        lg = self.collective(S.all_gather, [self.call(S.logits, Coll(pl[0], hidden_row.refs), Coll(pl[1], self.lm_head.refs))])[0]
        for fn in S.cap if self.C.final_cap is not None else ():
            lg = self.call(fn, Coll(_ptypes(fn)[0], lg.refs))
        if policy is None:
            return self.call(S.token_select, Coll(_ptypes(S.token_select)[0], lg.refs))
        fn = S.stochastic_select(policy.S)
        pt = _ptypes(fn)
        top_p = self.call(S.const(32, policy.top_p_bits))
        pos_c = self.call(S.const(64, int(pos) & 0xFFFFFFFFFFFFFFFF))
        temp = self.call(S.const(32, policy.temperature_bits))
        splits = policy.splits.refs.slice(event, event + 1) if policy.S == "derived" else self.call(S.const(32, int(policy.S))).refs
        return self.call(fn, Coll(pt[0], lg.refs), Coll(pt[1], top_p.refs), Coll(pt[2], policy.seed.refs), Coll(pt[3], pos_c.refs),
                         Coll(pt[4], temp.refs), Coll(pt[5], splits))


@dataclass(frozen=True)
class _Stochastic:
    """The row's stochastic policy as the select reads it: the seed (and, for several requests, the `splits`) Inputs, the f32 bits of
    temperature and top_p, and S: "derived" (an Input entry per event) or the single request's constant split count."""
    seed: Coll
    splits: Coll | None
    S: Any
    temperature_bits: int
    top_p_bits: int


def _check_phi(phi: Phi) -> None:
    facts = []
    s = phi.sampling or {}
    if s.get("kind", "greedy") == "stochastic":
        if int(s.get("top_k", 0) or 0) != 0 or not s.get("seeded", False):
            facts.append(f"stochastic sampling with top_k {s.get('top_k')!r}, seeded {s.get('seeded')!r} (seeded top-p only)")
        if s.get("splits") != "derived" and not isinstance(s.get("splits"), int):
            facts.append(f"stochastic sampling without a split count (splits {s.get('splits')!r})")
    elif s.get("kind", "greedy") != "greedy":
        facts.append(f"sampling {s.get('kind')!r} (greedy or seeded top-p)")
    world = int((phi.tp or {}).get("world", 1))
    if world not in (1, 2):
        facts.append(f"tensor parallel world {world} (TP1 and TP2, the worlds a Build has traced)")
    elif world == 2 and s.get("kind", "greedy") != "greedy":
        facts.append("stochastic sampling under tensor parallelism (the with-peers wrapper takes the prompt and the peers only)")
    if phi.profile != "b1-eager-v3":
        facts.append(f"numerics profile {phi.profile!r}")
    if phi.execution != "bi-eager":
        facts.append(f"execution {phi.execution!r}")
    if (phi.model or {}).get("dtype", "bfloat16") != "bfloat16":
        facts.append(f"checkpoint dtype {phi.model.get('dtype')!r} (bf16 only)")
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))


def build_program(phi: Phi, wrapper: str, *, LP: int | None = None, T: int | None = None, M: int = 1, pos0: int = 8,
                  with_desc: bool = True) -> BuildResult:
    """The Program of the Build's `wrapper` for φ: `step` (one engine step, M token rows at pos0.., the last row sampled) or
    `request` (prefill of LP rows, then T decode steps each fed the previous greedy sample; returns the T+1 samples)."""
    t0 = time.time()
    _check_phi(phi)
    kind = (phi.sampling or {}).get("kind", "greedy")
    if (wrapper, kind) not in WRAPPERS:
        raise Unexpressible(f"wrapper {wrapper!r} (have {sorted({w for w, _ in WRAPPERS})})")
    C = model_shape(phi.hf_config)
    history = pos0 + M if wrapper == "step" else int(LP) + int(T)
    if C.window is not None and history > C.window:
        raise Unexpressible(f"vocabulary cannot express: a history of {history} keys under the sliding window of {C.window} keys")
    target = phi.target_profile()
    world, rank = int(phi.tp.get("world", 1)), int(phi.tp.get("rank", 0))
    specs = _Specs(C, target, world, rank, M)
    leaves = weight_leaves(C, world)
    wrec = Record(*leaves)
    n = M if wrapper == "step" else int(LP)
    events = 1 if wrapper == "step" else int(T) + 1
    params: list[tuple[str, Type]] = [("input_ids" if wrapper == "step" and world == 1 else "prompt", Array(n, I32))]
    if kind == "stochastic":
        params.append(("seed", Array(1, I64)))
        if phi.sampling["splits"] == "derived":
            params.append(("splits", Array(events, I32)))
    n_user = len(params)
    if world > 1:
        # `vllm_meta.WithPeers`: per forward pass the embedding's all-reduce, each layer's q / k all-gathers (a norm over all heads)
        # and o / down all-reduces over its rows, then the sampled row's logits all-gather
        R, passes = specs.C, [n] + [1] * (events - 1)
        layer = lambda r: ([(r, R.NH * R.D), (r, R.KVH * R.D)] if R.qk_norm == "all" else []) + [(r, C.H)] * 2  # noqa: E731
        shapes = [s for r in passes for s in [(r, C.H)] + layer(r) * C.NL + [(1, C.V // world)]]
        params += [(f"peers_{k}", _arr(s, 16)) for k, s in enumerate(shapes)]
    params.append(("weights", wrec))
    params = tuple(params)
    B = Builder(params, Value(1))
    ids = Coll(params[0][1], Affine("p", 0, 0, n))
    W = Coll(wrec, Affine("p", len(params) - 1, 0, wrec.leaves))
    peers = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params[n_user:-1], start=n_user)]
    policy = None
    if kind == "stochastic":
        from verity_vllm.program.frontend.rules.vllm_sampling import f32_bits
        s = phi.sampling
        policy = _Stochastic(seed=Coll(I64, Affine("p", 1, 0, 1)), splits=Coll(params[2][1], Affine("p", 2, 0, events)) if s["splits"] == "derived" else None,
                             S=s["splits"], temperature_bits=f32_bits(s["temperature"]), top_p_bits=f32_bits(s.get("top_p", 1.0)))
    model = _Model(B, specs, W, specs.C, peers, rank)
    rows = [Coll(I32, ids.refs.slice(i, i + 1)) for i in range(n)]
    if wrapper == "step":
        hid = model.forward(rows, pos0)
        tok = model.sample(hid[-1], 0, pos0 + M - 1, policy)
        out = Coll(Array(1, I32), tok.refs)
    else:
        hid = model.forward(rows, 0)
        toks = [model.sample(hid[-1], 0, int(LP) - 1, policy)]
        for s in range(int(T)):
            hid = model.forward([Coll(I32, toks[-1].refs)], int(LP) + s)
            toks.append(model.sample(hid[-1], s + 1, int(LP) + s, policy))
        out = Coll(Array(len(toks), I32), concat([t.refs for t in toks]))
    if model.n_peer != len(peers):
        raise AssertionError(f"{model.n_peer} collectives read {len(peers)} peer Inputs")
    B.ret_type = out.type
    body = B.finish(out)
    t_body = time.time()
    src_class, derive_name = WRAPPERS[(wrapper, kind)]
    if world > 1:
        src_class = f"verity-vllm/with-peers/v1[{src_class}]"
    name = f"{derive_name}[{C.model_class}]"
    src_static = {"class": src_class, "mode": "profile", "profile": phi.profile, "inputs": [list(_arr_shape(t)) for _, t in params[:-1]],
                  "config_sha256": hashlib.sha256(canonical_json({})).hexdigest()[:16]}
    from verity_vllm.program.replay import replayer
    defn = CompositeDefinition(f"Derived[{name}]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (tuple(_p), _r), replayer(body),
                               doc=f"predicted ahead of time by verity_vllm.predict.build ({BUILDER_ID})", conformance="predicted: not a record",
                               register=False)
    sd = bind(defn, SRC=src_static)
    sd.set_body(body)
    program = Program(sd)
    timings = {"body_s": round(t_body - t0, 3), "program_s": round(time.time() - t_body, 3)}
    desc = digest = None
    if with_desc:
        t2 = time.time()
        desc = encode_program(program)
        digest = program_digest(desc)
        timings["encode_digest_s"] = round(time.time() - t2, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    return BuildResult(wrapper=wrapper, program=program, digest=digest or "", desc=desc, function=sd.id,
                       n_root_nodes=len(program.fn.body.nodes) if hasattr(program.fn, "body") else -1, timings=timings)
