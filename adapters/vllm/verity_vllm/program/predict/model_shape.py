"""The shape facts of a checkpoint's `config.json` that the predicted Programs are built from, and its weights record.

`model_shape` reads an HF config through the two declared homes of model facts (`engine.profiles.family_facts` and
`program.model.config_of`) into a `ModelShape`, refusing as `Unexpressible` whatever the vocabulary of `build` can't express;
`rank_shape` is one rank's slice of it under vLLM's tensor parallelism, and `weight_leaves` the weights record of the model
class on meta, as the derive lists it.  Nothing here builds a Program: `build`, `batch`, `steps` and the family decoders
read a shape from here (`build` hands the names on, so `from .build import model_shape` holds).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from verity.circuits.types import Type, Value

from verity_vllm.engine.profiles.family_facts import FAMILY_FACTS, resolve_facts

from .common import Unexpressible, _arr, served_class

#: the family facts (`engine.profiles.family_facts`) `model_shape` reads; a family with any other refuses
_FACTS_READ = ("QK_NORM", "QK_NORM_EPS", "RENORMALIZE", "CLIP_QKV", "SHARED_EXPERTS", "tie_word_embeddings", "attn_logit_softcapping",
               "final_logit_softcapping", "query_pre_attn_scalar", "sliding_window", "hidden_act", "embed_normalizer", "rmsnorm_plus_one",
               "layer_norms", "extra_layer_norms")
#: the activations of the gated MLP: the `*Mul` kind's role per `hidden_act`
_ACTS = {"silu": "silu_mul", "gelu_pytorch_tanh": "gelu_tanh_mul"}


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
    attn_softcap: float | None = None
    final_softcap: float | None = None
    int8: bool = False

    @property
    def QKV(self) -> int:
        return (self.NH + 2 * self.KVH) * self.D

    @property
    def nodes_per_row_layer(self) -> int:
        """About how many root nodes one row adds per layer: 14, plus the q/k/v bias, the q/k RMSNorms, the MoE block's three per pair,
        Gemma-2's four unfused norm chains' thirty and the int8 linears' four quantizers."""
        qk: dict[str | None, int] = {"head": self.NH + self.KVH, "all": 2}
        return 14 + self.qkv_bias + qk.get(self.qk_norm, 0) + (3 * self.TOPK + 2 if self.E else 0) + \
            30 * self.norm_chain + 4 * self.int8


def model_shape(hf: dict) -> ModelShape:
    """The shape facts of an HF `config.json` the vocabulary reads, through the two declared homes of model facts: the family facts
    table (`engine.profiles.family_facts`: vLLM's class and constructor kind per `model_type`) and `program.model.config_of` (the
    static configuration, with the family's bias rule and the checkpoint's quantization).  The vocabulary is the dense and the
    fused-MoE (indexed-read) blocks with no per-family facts beyond the q/k RMSNorm (per head, or over all heads as `qk_norm` "all"),
    the router's renormalization and Gemma-2's (the embedding normalizer, the (1 + w) norm chain with the two extra MLP norms, the
    attention and final logit softcapping, the tanh-GELU MLP, `query_pre_attn_scalar` = head_dim): a family with any other (another norm
    variant, LayerNorm, shared experts), a linear bias other than the q/k/v one a family's rule gives, an activation other than SiLU
    or a family-declared one, a partial rotary width, a rotary scaling without an analytic table, a block-FP8 checkpoint or an
    exact-int8 one with an int8 LM head or a biased linear raises `Unexpressible` (an exact-int8 checkpoint's linears are `int8`).  A
    sliding window is kept as `window` (the keys FA2's `window_size_left` admits) for the families whose vLLM code hands it to FA2
    (the `dense2` patterns); `build_program` refuses a history longer than it."""
    from verity_vllm.observe.fold.profiles.generic import effective_sliding_window
    from verity_vllm.program import model as analytic
    family = str(hf.get("model_type", "")).lower()
    ff = FAMILY_FACTS.get(family)
    if ff is None:
        raise Unexpressible(f"model_type {family!r}: not in the family facts table")
    stated_rules = ff.get("rules")
    rules = dict(stated_rules) if stated_rules else {}    # a family that states no rules has none
    facts = []
    arch = ff.get("arch")
    if arch not in ("dense", "moe"):
        facts.append(f"family {family!r} constructor {arch!r} (the dense and MoE blocks only)")
    stated_facts = ff.get("facts")
    other = sorted(set(stated_facts if stated_facts else {}) - set(_FACTS_READ))    # and one that states no facts, none
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
    inter = int(C["I"])
    if arch == "moe":
        from verity_vllm.program.frontend.rules.vllm_moe import TOPK_SOFTMAX_E_TESTED
        E, TOPK = int(C.get("NUM_EXPERTS") or 0), int(C.get("TOPK") or 0)
        if ff.get("moe_width_key"):
            inter = int(hf[ff["moe_width_key"]])
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
    softcaps = [float(fv[k]) if fv.get(k) else None for k in ("attn_logit_softcapping", "final_logit_softcapping")]
    if C.get("FP8_BLOCK"):
        facts.append(f"block-scaled FP8 checkpoint (G={C['FP8_BLOCK']})")
    int8 = bool(C.get("INT8_EXACT"))
    if C.get("INT8_LM_HEAD"):
        facts.append("an int8 LM head (verity-int8-channel127/v2; the token select reads a bf16 head's logits)")
    if int8 and qkv_bias:
        facts.append("an int8 linear with a bias (Int8PlainLinearMethod takes none)")
    if analytic.rotary_dim(C) != C["D"]:
        facts.append(f"partial rotary width {analytic.rotary_dim(C)} of head_dim {C['D']}")
    try:
        analytic._inv_freq(C)
    except NotImplementedError as e:
        facts.append(str(e))
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))
    return ModelShape(arch=family, model_class=served_class(ff["vllm_class"]), H=int(C["H"]), V=int(C["V"]), NL=int(C["NL"]), NH=int(C["NH"]), KVH=int(C["KVH"]),
                      D=int(C["D"]), I=inter, EPS=float(C["EPS"]), tie=bool(fv.get("tie_word_embeddings", C["tie_word_embeddings"])),
                      max_pos=int(C["MAX_POS"]), rot=int(C["D"]), qkv_bias=qkv_bias, window=int(window) - 1 if window else None,
                      qk_norm=qk_norm, E=E, TOPK=TOPK, renorm=bool(fv.get("RENORMALIZE")), act=act, norm_chain=norm_chain,
                      extra_norms=bool(extra), embed_scale="embed_normalizer" in fv, attn_softcap=softcaps[0], final_softcap=softcaps[1], int8=int8)


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
    replicated_kv = world > C.KVH and world % C.KVH == 0          # QKVParallelLinear: a key/value head on world // KVH ranks
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


def weight_leaves(C: ModelShape, world: int = 1) -> list[tuple[str, Type]]:
    """The weights record of the model class on meta, as `derive._weights_of` lists it: the distinct parameters in module order
    (the tied `lm_head.weight` is the embedding's leaf), then the distinct buffers (the embedding's 0-d bf16 `normalizer` where the
    family scales it, the one shared rotary `cos_sin_cache` [max_pos, rot] bf16, each attention layer's four f32 scalar scales).  The
    field names are informational (the descriptor carries types only) and, as every decoder's (`predict.dsa`, `predict.dsv4`), the
    vLLM model's own (`model.layers.0.…`, `lm_head.weight`), never the prefix of the module a trace wraps it in.  Under `world`
    ranks each is one rank's shard (`rank_shape`)."""
    def bf(*s: int) -> Type:
        return _arr(s, 16)

    V = C.V // world
    C = rank_shape(C, world)

    def linear(name: str, N: int, K: int) -> list[tuple[str, Type]]:
        """A decoder linear's parameters: its bf16 weight, or the exact-int8 checkpoint's s8 weight and f32 row scales
        (`engine.int8_vllm.Int8PlainLinearMethod`'s `weight` [N, K] and `weight_scale` [N, 1], in that order)."""
        if not C.int8:
            return [(name + ".weight", bf(N, K))]
        return [(name + ".weight", _arr((N, K), 8)), (name + ".weight_scale", _arr((N, 1), 32))]

    out: list[tuple[str, Type]] = [("model.embed_tokens.weight", bf(V, C.H))]
    for L in range(C.NL):
        p = f"model.layers.{L}."
        out += linear(p + "self_attn.qkv_proj", C.QKV, C.H)
        if C.qkv_bias:
            out.append((p + "self_attn.qkv_proj.bias", bf(C.QKV)))
        if C.qk_norm == "all":
            out += [(p + "self_attn.q_norm.weight", bf(C.NH * C.D * world)), (p + "self_attn.k_norm.weight", bf(C.KVH * C.D * world))]
        out += linear(p + "self_attn.o_proj", C.H, C.NH * C.D)
        if C.qk_norm == "head":
            out += [(p + "self_attn.q_norm.weight", bf(C.D)), (p + "self_attn.k_norm.weight", bf(C.D))]
        if C.E:
            out += [(p + "mlp.gate.weight", bf(C.E, C.H)), (p + "mlp.experts.w13_weight", bf(C.E, 2 * C.I, C.H)),
                    (p + "mlp.experts.w2_weight", bf(C.E, C.H, C.I))]
        else:
            out += linear(p + "mlp.gate_up_proj", 2 * C.I, C.H) + linear(p + "mlp.down_proj", C.H, C.I)
        out += [(p + "input_layernorm.weight", bf(C.H)), (p + "post_attention_layernorm.weight", bf(C.H))]
        if C.extra_norms:
            out += [(p + "pre_feedforward_layernorm.weight", bf(C.H)), (p + "post_feedforward_layernorm.weight", bf(C.H))]
    out.append(("model.norm.weight", bf(C.H)))
    if not C.tie:
        out.append(("lm_head.weight", bf(V, C.H)))
    if C.embed_scale:
        out.append(("model.normalizer", bf(1)))
    out.append(("model.layers.0.self_attn.rotary_emb.cos_sin_cache", bf(C.max_pos, C.rot)))
    for L in range(C.NL):
        out += [(f"model.layers.{L}.self_attn.attn.{s}", Value(32)) for s in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")]
    return out
