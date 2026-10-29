"""Pinned capture profile κ: vLLM `d9105ea80` (0.28.1rc1.dev472), torch 2.13.0+cu129, Triton 3.7.1, RTX 4090 (sm_89),
eager, `VLLM_BATCH_INVARIANT=1`, bf16, SmolLM2-135M (`b1.B0_CONFIG`).  See DESIGN.md §1 (κ), §6 (patterns), §7 (weights).

`PROFILE.patterns` is the ordered pattern list of DESIGN §6 (specific kernels first, the transparent aten set last).
The constexpr / launch-geometry values the patterns accept are pinned below as module constants; a launch that
differs is `Unsupported`, never adapted to.

Root classification for the fold (DESIGN §7): `weight_field(name)` maps a `Root.name` to its `b1.weights_type(C)`
Record path; `root_class(name)` says why a name maps to None (`alias-of-cos_sin`: the shared rotary module's
per-layer buffer names; `ignored`: KV-scale placeholders that carry no value under `kv_cache_dtype=auto`).

[M6] Everything that depends on the model is a function of the static configuration `C` (`b1.<CASE>_CONFIG`):
`build_profile(C, profile_id=...)` returns the Profile for another dense Qwen2/Llama-architecture checkpoint under the
same κ (same vLLM / torch / Triton / GPU pins); `profiles/vllm_d9105ea80_sm89_eager_qwen15.py` is the one-line module
for Qwen2.5-1.5B (`b1.B1_CONFIG`).  The pattern list, the kernel pins and the weight-name map are shared — a second
model is a configuration record, not a second profile implementation.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity.ml.definitions.registry import b1

from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import (AttentionScope, BiasAdd, Embedding, FusedAddRMSNorm, GemmLaunch, KVCacheUpdate, LogitsGather,
                        RMSNormTritonLaunch, RotaryEmbedding, SiluAndMul, TokenSelectArgmax, TokenSelectGumbelTopP, TokenSelectGumbelTwoStage,
                        TransparentAten, TransparentTritonMetadata, fuse_bias_adds)
from verity_vllm.observe.fold.resolver import Profile

PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16"

# ---------------------------------------------------------------------------------------------------------
# static configuration C (b1.B0_CONFIG; b1.py is not edited)
# ---------------------------------------------------------------------------------------------------------


def with_geometry(config: dict[str, Any]) -> dict[str, Any]:
    """`C` plus the derived FA2 static: `BN = b1.fa2_kblock_n(D)` (128 at head dim 64 — SmolLM2; 64 at head dim 128 —
    Qwen2.5-1.5B)."""
    C = dict(config)
    C["BN"] = b1.fa2_kblock_n(C["D"])
    return C


C: dict[str, Any] = with_geometry(b1.B0_CONFIG)

# ---------------------------------------------------------------------------------------------------------
# pinned launch constexprs / geometry
# ---------------------------------------------------------------------------------------------------------

#: `vllm/model_executor/determinism/batch_invariant.py::matmul_persistent` bf16 config (the inlined copy in
#: `fixtures/results/B0-535f2cb2cd26-20260907T1815Z/bundle/kernels/86e1fd7cb50926e7.py` has the same constants).  The
#: registered `Gemm_v1` body is one f32 accumulator per output coordinate advanced by K/16 `AmpereBF16TcDot16` steps
#: in ascending k from zero: the kernel's `accumulator = tl.dot(a, b, accumulator)` over `BLOCK_SIZE_K = 64` tiles
#: (mma.sync m16n8k16 chained inside each tile), K a multiple of 64 for every b1 GEMM (576, 1536).  `HAS_BIAS=False`:
#: the bias epilogue (`accumulator += bias` in f32 before the bf16 store) is a different numeric than `Gemm` + `BiasAdd`.
GEMM_CONSTEXPRS: dict[str, Any] = {
    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8,
    "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False,
}
GEMM_NUM_WARPS = 8
GEMM_NUM_STAGES = 3
#: `NUM_SMS` is baked into the kernel (tile schedule only, no numeric effect).  [X-01] It is a TARGET fact, not a κ pin: the
#: matcher pins each launch to the record target's SM count (`patterns.pinned_num_sms`, from the run header's device document);
#: this value is only the profile's own default (the RTX 4090's 128 SMs) for headers that carry no device — one source.
GEMM_NUM_SMS = DEFAULT_TARGET.num_sms

#: `batch_invariant.py::_rms_norm_kernel` as `rms_norm_batch_invariant` launches it (W11's model, `verity/ml/definitions/numerics/
#: rms_triton_model.py`): one CTA of 4 warps per row, 8 columns per thread per 1024-column chunk, warp butterfly
#: xor 16,8,4,2,1, CTA `(w0 + w2) + (w1 + w3)` — the reduction tree `RMSNormTriton_v1` writes out.
RMS_CONSTEXPRS: dict[str, Any] = {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True}
RMS_NUM_WARPS = 4

#: FA2 (`vllm-project/flash-attention 506341a`, `_vllm_fa2_C.varlen_fwd`): under `VLLM_BATCH_INVARIANT=1`
#: `num_splits = 1` selects `run_mha_fwd_splitkv_align` → `Flash_fwd_kernel_traits<64, 64, 128, 4>`: kBlockM 64,
#: kBlockN 128 (measured on the 4090, `out/curve/e10/KERNEL-GEOMETRY.md` §A).  `C["BN"]` is the `Attention_v3` static `BN`.
ATTN_NUM_SPLITS = 1

#: KV cache of the FlashAttention backend in κ (M1 log): the per-layer cache is one bf16 buffer
#: `[num_blocks, KVH, block_size, 2*D]` viewed as key `[num_blocks, block_size, KVH, D]` strides `(bs*KVH*2D, KVH*2D, 2D, 1)`
#: = `(6144, 384, 128, 1)` and value the same view at `+D`; `kv_cache_dtype=auto` (bf16); block_size 16
#: (`StepStart.block_size` is the fact used for the slot arithmetic).  `patterns.as_slot_rows` accepts any uniformly
#: strided `[blocks, block_size, KVH, D]` cache (fails loudly otherwise).
KV_BLOCK_SIZE = 16
KV_CACHE_DTYPE = "auto"

#: vLLM V2 sampler (`vllm/v1/worker/gpu/sample/gumbel.py`): greedy `gumbel_sample` = `_gumbel_sample_kernel` with
#: `BLOCK_SIZE=1024` (blockwise max/argmax over the bf16 logits, `ceil(V/1024) = 48` blocks) + `aten.argmax` over the
#: block maxima + `aten.gather` of the block argmax → the first global maximum = `TokenSelect_v1{V}`.
SAMPLER_BLOCK_SIZE = 1024

# ---------------------------------------------------------------------------------------------------------
# weights (DESIGN §7): vLLM parameter / buffer names -> b1.weights_type(C) Record paths
# ---------------------------------------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LAYER_FIELDS = {
    "input_layernorm.weight": "ln1",
    "self_attn.qkv_proj.weight": "qkv_w",
    "self_attn.qkv_proj.bias": "qkv_b",
    "self_attn.o_proj.weight": "o_w",
    "post_attention_layernorm.weight": "ln2",
    "mlp.gate_up_proj.weight": "gate_up_w",
    "mlp.down_proj.weight": "down_w",
}
COS_SIN_BUFFER = "self_attn.rotary_emb.cos_sin_cache"
#: `get_rope` caches one `RotaryEmbedding` module for all layers; `named_buffers()` names its `cos_sin_cache` after the
#: first layer (deduplicated) — or after every layer when duplicates are kept, all aliasing one allocation.  Layer 0's
#: name is the Record field `cos_sin`; the others are aliases (`root_class` → "alias-of-cos_sin").
ALIAS_OF_COS_SIN = re.compile(r"^model\.layers\.[1-9]\d*\.self_attn\.rotary_emb\.cos_sin_cache$")
#: Roots with no field: the attention layers' KV/Q scale placeholders (`kv_cache_dtype=auto`: never read by a kernel
#: we model; `reshape_and_cache_flash` receives them and ignores them for bf16 caches).
IGNORED_ROOTS = (
    re.compile(r"^model\.layers\.\d+\.self_attn\.attn\.(k_scale|v_scale|q_scale|prob_scale|_k_scale|_v_scale|_q_scale|_prob_scale)$"),
)


def make_weight_field(C: dict[str, Any]) -> Callable[[str], tuple[str | int, ...] | None]:
    """The `weight_field` of a static configuration: `Root.name` -> path into `analytic.weights_type(C)` (`("embed",)`,
    `("layers", L, "qkv_w")`, `("norm",)`, `("cos_sin",)`, `("lm_head",)` when untied), or None (see `root_class`).
    `qkv_b` is a field only when `C["BIAS"]` (Qwen2's fused QKV projection carries a bias; Llama-architecture models do
    not); `lm_head` only when `C["tie_word_embeddings"]` is False."""
    bias = bool(C.get("BIAS", False))
    tied = bool(C.get("tie_word_embeddings", True))
    # family / quantization facts of the configuration (analytic.dense_layer_weights_type): the per-head q/k norm weights
    # (Qwen3) and the f32 block scales of the FP8 linears exist as fields only when C says so; on any other configuration the same
    # parameter names stay unknown roots (never silently mapped)
    layer_fields = dict(_LAYER_FIELDS)
    if C.get("QK_NORM") == "head":
        layer_fields.update({"self_attn.q_norm.weight": "q_norm", "self_attn.k_norm.weight": "k_norm"})
    if C.get("FP8_BLOCK"):
        layer_fields.update({"self_attn.qkv_proj.weight_scale_inv": "qkv_ws", "self_attn.o_proj.weight_scale_inv": "o_ws",
                             "mlp.gate_up_proj.weight_scale_inv": "gate_up_ws", "mlp.down_proj.weight_scale_inv": "down_ws"})
    if C.get("INT8_EXACT"):
        # the exact-int8 linears' f32 per-row scales (`program.model._int8_fields`)
        layer_fields.update({"self_attn.qkv_proj.weight_scale": "qkv_ws", "self_attn.o_proj.weight_scale": "o_ws",
                             "mlp.gate_up_proj.weight_scale": "gate_up_ws", "mlp.down_proj.weight_scale": "down_ws"})

    def weight_field(name: str) -> tuple[str | int, ...] | None:
        if name == "model.embed_tokens.weight":
            return ("embed",)
        if name == "lm_head.weight":
            # tied checkpoints (SmolLM2, Qwen2.5 ≤ 3B): lm_head.weight aliases embed (a fact, DESIGN §3);
            # untied ones (TinyLlama, Qwen2.5-7B, Llama-3.1-8B): a field of its own (`analytic.weights_type`)
            return ("embed",) if tied else ("lm_head",)
        if name == "lm_head.weight_scale" and C.get("INT8_LM_HEAD") and not tied:
            return ("lm_head_ws",)
        if name == "model.norm.weight":
            return ("norm",)
        m = _LAYER_RE.match(name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest == COS_SIN_BUFFER:
                return ("cos_sin",) if layer == 0 else None
            f = layer_fields.get(rest)
            if f is not None:
                if f == "qkv_b" and not bias:
                    return None
                return ("layers", layer, f)
        return None

    return weight_field


def make_root_class(weight_field: Callable[[str], Any]) -> Callable[[str], str]:
    """'weight' (has a Record field), 'alias-of-cos_sin' (another layer's name for the shared rotary table — the fold
    treats reads of it as reads of `cos_sin`), 'ignored' (no value under κ), 'unknown' (must not happen on an accepted
    run)."""

    def root_class(name: str) -> str:
        if weight_field(name) is not None:
            return "weight"
        if ALIAS_OF_COS_SIN.match(name):
            return "alias-of-cos_sin"
        if any(r.match(name) for r in IGNORED_ROOTS):
            return "ignored"
        return "unknown"

    return root_class


weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)


# ---------------------------------------------------------------------------------------------------------
# header consistency
# ---------------------------------------------------------------------------------------------------------

_HF_KEYS = {"hidden_size": "H", "num_hidden_layers": "NL", "num_attention_heads": "NH", "num_key_value_heads": "KVH",
            "intermediate_size": "I", "vocab_size": "V", "rms_norm_eps": "EPS", "rope_theta": "ROPE_THETA",
            "max_position_embeddings": "MAX_POS"}


def make_check_header(C: dict[str, Any]) -> Callable[[RunHeader], list[str]]:
    def check_header(header: RunHeader) -> list[str]:
        """Mismatches between the run's recorded HF config and C (empty when consistent or when the header carries no config)."""
        cfg = (header.model or {}).get("config") or {}
        out = []
        for hf, key in _HF_KEYS.items():
            if hf in cfg and cfg[hf] != C[key]:
                out.append(f"{hf}={cfg[hf]!r} != C[{key}]={C[key]!r}")
        if "head_dim" in cfg and cfg["head_dim"] != C["D"]:
            out.append(f"head_dim={cfg['head_dim']!r} != C[D]={C['D']!r}")
        if "attention_bias" in cfg and bool(cfg["attention_bias"]) != bool(C.get("BIAS", False)):
            out.append(f"attention_bias={cfg['attention_bias']!r} != C[BIAS]={C.get('BIAS', False)!r}")
        # a fact compared to C (make_weight_field maps lm_head.weight onto embed iff tied, onto its own field iff untied)
        if "tie_word_embeddings" in cfg and bool(cfg["tie_word_embeddings"]) != bool(C.get("tie_word_embeddings", True)):
            out.append(f"tie_word_embeddings={cfg['tie_word_embeddings']!r} != C[tie_word_embeddings]={C.get('tie_word_embeddings', True)!r}")
        return out

    return check_header


check_header = make_check_header(C)


# ---------------------------------------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------------------------------------


def make_patterns(C: dict[str, Any]) -> list:
    """DESIGN §6's ordered pattern list for a static configuration: the kernel pins are κ's (shared), the model
    enters only through `EPS` (norm epsilons), `V` (the sampler's logits width) and `BN` (FA2 kBlockN at `D`)."""
    fp8: list = []
    if C.get("FP8_BLOCK"):
        # the block-scaled FP8 linear (vLLM Fp8LinearMethod -> CutlassFp8BlockScaledMMKernel, VLLM_USE_DEEP_GEMM=0): the
        # dynamic per-token-group activation quant and the blockwise CUTLASS scaled GEMM, each ONE registered family (registry.fp8)
        from verity_vllm.observe.fold.patterns.fp8 import CutlassScaledMmFp8Block, PerTokenGroupFp8Quant
        fp8 = [PerTokenGroupFp8Quant(group=int(C["FP8_BLOCK"])), CutlassScaledMmFp8Block(group=int(C["FP8_BLOCK"]))]
    if C.get("INT8_EXACT"):
        # the exact-int8 profile's named ops (engine.int8_ops), each one registered Definition (registry.int8_exact); the attention
        # scope first, so a layer on the int8 backend never reaches the FA2 scope
        from verity_vllm.observe.fold.patterns.int8 import Int8AttentionScope, Int8RowQuant, Int8ScaledMm
        fp8 = [Int8RowQuant(), Int8ScaledMm(), Int8AttentionScope()]
    return [
        *fp8,
        GemmLaunch(GEMM_CONSTEXPRS, num_warps=GEMM_NUM_WARPS, num_stages=GEMM_NUM_STAGES, num_sms=GEMM_NUM_SMS),
        RMSNormTritonLaunch(RMS_CONSTEXPRS, num_warps=RMS_NUM_WARPS, eps=C["EPS"]),
        FusedAddRMSNorm(eps=C["EPS"]),
        RotaryEmbedding(),
        KVCacheUpdate(),
        AttentionScope(bn=C["BN"], num_splits=ATTN_NUM_SPLITS),
        SiluAndMul(),
        Embedding(),
        # the seeded temperature + top-p chain (returns None under a greedy header; first in line so the greedy
        # recognizer's `outside greedy sampling` refusal is not what a stochastic row sees)
        TokenSelectGumbelTopP(vocab=C["V"], block_size=SAMPLER_BLOCK_SIZE),
        TokenSelectGumbelTwoStage(vocab=C["V"], block_size=SAMPLER_BLOCK_SIZE),
        TokenSelectArgmax(vocab=C["V"]),
        LogitsGather(),
        BiasAdd(),
        TransparentAten(),
        TransparentTritonMetadata(),
    ]


PATTERNS = make_patterns(C)

# the census allowlist is PROTECTED (verity_vllm/properties/kernel_allowlist.py); profiles only reference it
from verity_vllm.properties.kernel_allowlist import KERNEL_ALLOWLIST  # noqa: E402


def build_profile(config: dict[str, Any], *, profile_id: str, case: str, model_note: str = "") -> Profile:
    """The κ profile for a static configuration (`b1.<case>_CONFIG`, `BN` derived if absent).  Shared: the κ pins,
    the pattern list, the weight-name map, the kernel allowlist.  Per model: `C` (and so EPS / V / BN / BIAS), the
    Record the fold's `weights` parameter has, and the header consistency check."""
    C = config if "BN" in config else with_geometry(config)
    wf = make_weight_field(C)
    return Profile(
        id=profile_id,
        patterns=make_patterns(C),
        config=C,
        weight_field=wf,
        accepted=False,
        kernel_allowlist=KERNEL_ALLOWLIST,
        step_passes=(fuse_bias_adds,),
        notes={
            "case": case,
            "model": {"repo": C.get("model"), "note": model_note},
            "kappa": {"vllm": "d9105ea80 (0.28.1rc1.dev472+gd9105ea80.cu129)", "torch": "2.13.0+cu129", "triton": "3.7.1",
                      "gpu": "RTX 4090 sm_89", "env": ["VLLM_BATCH_INVARIANT=1", "VLLM_USE_V2_MODEL_RUNNER=1",
                                                       "VLLM_ENABLE_V1_MULTIPROCESSING=0", "VLLM_USE_FLASHINFER_SAMPLER=0"],
                      "engine": "eager, bf16, max_model_len 4096, enable_prefix_caching False, greedy"},
            "pins": {"gemm": {**GEMM_CONSTEXPRS, "NUM_SMS": GEMM_NUM_SMS, "num_warps": GEMM_NUM_WARPS, "num_stages": GEMM_NUM_STAGES},
                     "rms_norm": {**RMS_CONSTEXPRS, "num_warps": RMS_NUM_WARPS},
                     "attention": {"kernel": "flash_fwd_splitkv_kernel", "kBlockM": 64, "kBlockN": C["BN"], "num_splits": ATTN_NUM_SPLITS},
                     "kv_cache": {"dtype": KV_CACHE_DTYPE, "block_size": KV_BLOCK_SIZE}},
            "alias_of_cos_sin": ALIAS_OF_COS_SIN.pattern,
            "ignored_roots": [r.pattern for r in IGNORED_ROOTS],
            "root_class": "profiles.vllm_d9105ea80_sm89_eager.make_root_class(profile.weight_field)(name)",
            "grouped_resolutions": "patterns.ComputeGroup / TransparentGroup (.before / .also); patterns.members(r) is the fold order",
        },
    )


PROFILE = build_profile(C, profile_id=PROFILE_ID, case="B0",
                        model_note="SmolLM2-135M @ 93efa2f0 (b1.B0_CONFIG): H=576 NL=30 NH=9 KVH=3 D=64 I=1536 V=49152, no QKV bias, FA2 kBlockN 128")
