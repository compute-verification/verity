"""[R19 qwen3, row #75] Profile QWEN3_30B_A3B -- `Qwen/Qwen3-30B-A3B` (`Qwen3MoeForCausalLM`) on the pinned κ (vLLM d9105ea80, sm89, eager,
bf16) as a Match-grade CONSTRUCTIVE profile: the 48 MoE blocks fold to pair-major Computes over the registered MoE Definitions
(`verity_vllm.registry.moe`) per step per rank, exactly as `gen_ov_moe_olmoe` does for OLMoE, instead of the wave-1 `MoEBlockScope`
(recognised Unsupported BY DESIGN, which is why R17/R18 left the row's Match cell at collective level, `tp2_match` only).

`PROFILE` is the RANK profile of the row of record (tensor_parallel_size=2, `WORLD = 2`): what ONE rank's aten stream computes between the
collectives, the profile `tp2_fold_match --profile gen_ov_moe_qwen3` folds a rank log (`tp2_capture --tp 2 --record-collectives` ->
`log.rank<r>.jsonl.gz`) with -- the same construction as `vllm_d9105ea80_sm89_eager_qwen15_tp2rank` / `gen_llama_llama32_1b_tp2rank`
(`EmbeddingShardCuda`, `TpCollective(world=2)`, the sampler over the all-gathered vocabulary) plus the MoE block.  `build_profile(world=1)`
is the TP1 profile of the same checkpoint (no row of record uses it; kept so the case has one module for every world).

The rank-local configuration (vLLM's parallel layers at tp_size 2; every shape confirmed against the Build of record's per-rank parameters,
veritor `lane/r17-tp` .../r17int9/build_summary.json `tp.sharded_params`):
  * attention: NH/2 = 16 heads, KVH/2 = 2 kv heads, D = 128 -> QKV = (16 + 4) * 128 = 2560 (`QKVParallelLinear weight(2560, 2048)`), o_proj K =
    16 * 128 = 2048 (`RowParallelLinear weight(2048, 2048)`), no QKV bias; the per-head q/k RMSNorm weights are ONE `[D] = [128]` vector each
    (`RMSNorm weight(128,)` x 96) -- Qwen3's `qk_norm = "head"` family fact (`analytic.FAMILY_RULES["qwen3"]`, applied here to `qwen3_moe`);
  * MoE: the router `mlp.gate` is REPLICATED (`ReplicatedLinear weight(128, 2048)`: every rank computes the full E = 128 logits and the same
    top-8 route); the expert stacks are sharded on the INTERMEDIATE axis -- `w13 [E, 2 * I_r, H] = (128, 768, 2048)`, `w2 [E, H, I_r] =
    (128, 2048, 384)` with I_r = moe_intermediate_size / 2 = 384 (`RoutedExperts`); `moe_sum` is rank-local, then ONE all-reduce of the
    [M, H] routed sum per layer (97 all_reduce(M, 2048) per step: 48 o_proj + 48 MoE + 1 embedding; 1 all_gather(M, 75968) of the logits);
  * vocab: V/2 = 75,968 rows of the embedding shard and of the UNTIED `lm_head` (`ParallelLMHead weight(75968, 2048)`); H = 2048, NL = 48,
    EPS = 1e-6, rope theta 1e6 and the norms replicated.

What the profile pins beyond the shard configuration (the κ facts the wave-1 note and R17 named as the missing "Match-grade" pieces):
  * `renormalize=True` (config.json `norm_topk_prob: true`; vLLM `Qwen3MoeSparseMoeBlock` builds its FusedMoE with `renormalize=
    config.norm_topk_prob`) -> the router is `MoeRouterTopKNorm_v1{E=128,TOPK=8,VPT=8}` (`gen_ov_moe_patterns.MoEFusedExperts(renormalize=
    True)`), the frontend's `moe_router_topk_norm` kind, so the fold and the Build compare instance by instance;
  * the router's reduction geometry VPT = 8 at E = 128 for bf16 logits (`ROUTER_VPT_BY_E`; the frontend's `TOPK_SOFTMAX_VPT[bf16]`);
  * the fused_moe_kernel tiling `FUSED_MOE_PINS` (64/64/32/8, 4 warps, 3 stages): vLLM's batch-invariant default, the same at every M
    (frontend `MOE_GEMM_PINS` comment; measured on the L40S for E=128 / I=768 by the moe-xcheck `qwen3_shaped_block_e128_i768_renorm`,
    veritor `lane/r17-tp` out/gen/sweep/evidence/moe-xcheck/).  Injectable (`build_profile(fused_moe_pins=..., fused_moe_num_warps=...,
    fused_moe_num_stages=...)`) so a rank log that shows another tiling at the rank shape (N = 768 / 384) is pinned from the capture, not
    silently accepted: the pattern refuses with the observed values (F-r19-qwen3-1 is the finding to post if it does);
  * the in-scope all-reduce: vLLM issues the MoE block's tensor-parallel all-reduce either INSIDE `vllm.moe_forward` (`FusedMoE.forward_impl`,
    `reduce_results=True`) or after it in the sparse block (`reduce_results=False`); the rank log decides, not this module.  Both fold to the same
    Program: `MoEFusedExperts.scope_patterns = (TpCollective(2),)` resolves a nested `vllm.all_reduce` as `AllReduce2_v1{N=2048}` after
    `MoeSum_v1`, and the outer `TpCollective` handles the other placement.

Not this profile's business: the padded construction (`moe_construction = padded` refuses under renormalize=True -- `MoeFfnFixedL_v1` states the
unnormalised router), shared experts (Qwen3-30B-A3B has none; a `shared_experts_input` that is not None refuses through the scope glue),
expert parallelism (`expert_map`: `moe_sum` with an expert_map refuses by name).  `accepted=False` until the row's fold Match is in.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .. import analytic
from ..events import RunHeader
from ..patterns import TokenSelectArgmax, TokenSelectGumbelTopP, TokenSelectGumbelTwoStage
from ..resolver import Profile
from ..tp.collective_pattern import TpCollective
from . import vllm_d9105ea80_sm89_eager as base
from .gen_llama_llama32_1b_tp2rank import EmbeddingShardCuda
from .gen_moe_patterns import GemmLaunchTunedAda
from .gen_ov_moe_patterns import (FUSED_MOE_NUM_STAGES, FUSED_MOE_NUM_WARPS, FUSED_MOE_NUMERIC_PINS, FUSED_MOE_PINS, ROUTER_VPT_BY_E,
                                  MOE_CONSTRUCTION_PATTERNS)

REPO = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"     # workloads/qwen3-30b-a3b__*.json tokenizer_revision (manifests/checkpoints.json)
CASE = "QWEN3_30B_A3B"
#: the row of record is TP2: `PROFILE` is the rank profile
WORLD = 2

#: `analytic.FAMILY_RULES` has `qwen3` (dense); `qwen3_moe` is the same attention (vLLM `Qwen3MoeAttention`: per-head `RMSNorm(head_dim)` on q
#: and k before RoPE, `attention_bias` False, rope theta from config.json) -- passed as the lane's family rules, no edit of analytic.py.
FAMILY_RULES = {"qwen3_moe": {"qkv_bias_key": "attention_bias", "default_theta": 1_000_000.0, "qk_norm": "head"}}

#: Family facts of vLLM's `Qwen3MoeForCausalLM` (d9105ea80) that config.json does not state.
HAND = {
    "RENORMALIZE": True,      # Qwen3MoeSparseMoeBlock: FusedMoE(..., renormalize=config.norm_topk_prob) -> True for this checkpoint
    "SHARED_EXPERTS": 0,      # no shared experts (the model has none; vLLM passes shared_experts_input=None)
}


def _find_config() -> Path:
    """`data/hf_configs/QWEN3_30B_A3B.config.json` (tracked), else the seeded HF snapshot's `config.json` (`HF_HOME`)."""
    cands = [REPO / "data" / "hf_configs" / f"{CASE}.config.json"]
    for home in (os.environ.get("HF_HOME"), Path.home() / ".cache" / "huggingface"):
        if home:
            cands.append(Path(home) / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION / "config.json")
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Qwen3-30B-A3B config.json: none of {[str(c) for c in cands]}")


QWEN3_CONFIG = _find_config()
HF_CONFIG: dict[str, Any] = json.loads(QWEN3_CONFIG.read_text())

#: the HF keys vLLM's parallel layers divide by the world (heads / kv heads / expert intermediate / vocab); H, NL, the norms, the router
#: (`ReplicatedLinear`) and the rotary table are replicated.  `intermediate_size` (6144) is the DENSE MLP width Qwen3-MoE never uses here
#: (`mlp_only_layers` is empty, `decoder_sparse_step` 1): the experts' width is `moe_intermediate_size`.
_SHARDED_HF_KEYS = ("num_attention_heads", "num_key_value_heads", "moe_intermediate_size", "vocab_size")


def rank_hf_config(cfg: dict[str, Any], world: int) -> dict[str, Any]:
    """The rank-local HF configuration: the sharded keys divided by `world`, `intermediate_size` := the (sharded) expert width."""
    out = dict(cfg)
    for k in _SHARDED_HF_KEYS:
        if cfg[k] % world:
            raise ValueError(f"{k}={cfg[k]} is not divisible by the world {world}")
        out[k] = cfg[k] // world
    out["intermediate_size"] = out["moe_intermediate_size"]
    return out


def derive_config(cfg: dict[str, Any], world: int) -> tuple[dict[str, Any], dict[str, str]]:
    """`analytic.config_of` over the rank-local HF configuration, with the family rules of `qwen3_moe` and the hand facts."""
    if cfg.get("model_type") != "qwen3_moe":
        raise ValueError(f"model_type {cfg.get('model_type')!r} is not qwen3_moe")
    if cfg.get("mlp_only_layers"):
        raise ValueError(f"mlp_only_layers={cfg['mlp_only_layers']!r}: this profile states every layer as a fused-MoE layer")
    if int(cfg.get("decoder_sparse_step", 1)) != 1:
        raise ValueError(f"decoder_sparse_step={cfg['decoder_sparse_step']!r}: this profile states every layer as a fused-MoE layer")
    if not cfg.get("norm_topk_prob"):
        raise ValueError("norm_topk_prob is not true: this profile states the renormalising router (MoeRouterTopKNorm_v1)")
    if not cfg.get("head_dim"):
        raise ValueError("qwen3_moe without head_dim in config.json: the per-head q/k norm width is not derivable")
    C, prov = analytic.config_of(rank_hf_config(cfg, world), model=MODEL, revision=REVISION, hand=HAND, family_rules=FAMILY_RULES)
    C = base.with_geometry(C)
    C["WORLD"], prov["WORLD"] = world, "hand"
    C["MOE_I_FULL"], prov["MOE_I_FULL"] = int(cfg["moe_intermediate_size"]), "hf-config"
    if C["H"] % 64 or C["I"] % 32:
        raise ValueError(f"H={C['H']} I={C['I']}: the pinned GEMM tilings need K a multiple of 64 (dense) / 32 (fused_moe_kernel)")
    C["ROUTER_VPT"], prov["ROUTER_VPT"] = router_vpt_bf16(C["NUM_EXPERTS"]), "derived"
    return C, prov


def router_vpt_bf16(E: int) -> int:
    """`topkGatingSoftmax<T=bf16, EXPERTS=E>`'s values per thread (vLLM csrc/moe/topk_softmax_kernels.cu): 16-byte loads = `ELTS_PER_LDG =
    min(16, sizeof(T) * E) / sizeof(T)` bf16 words, `VECs_PER_THREAD = max(1, E / (ELTS_PER_LDG * 32))`, `VPT = ELTS_PER_LDG * VECs_PER_THREAD`.
    8 at E = 64 (OLMoE, bit-exact on the recorded cases) and at E = 128 (Qwen3-30B-A3B, the Build's binding); `ROUTER_VPT_BY_E` where it has
    the entry -- the two must agree (asserted), the rule covers the tiny test configurations."""
    if E <= 0 or E & (E - 1):
        raise ValueError(f"E={E}: the router kernel is instantiated for powers of two only")
    elts = min(8, E)
    vpt = elts * max(1, E // (elts * 32))
    if E in ROUTER_VPT_BY_E and ROUTER_VPT_BY_E[E] != vpt:
        raise ValueError(f"E={E}: router_vpt_bf16 gives {vpt}, ROUTER_VPT_BY_E pins {ROUTER_VPT_BY_E[E]}")
    return vpt


C, PROV = derive_config(HF_CONFIG, WORLD)
assert C["family"] == "qwen3_moe" and C["QK_NORM"] == "head" and C["BIAS"] is False and C["tie_word_embeddings"] is False, C
assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"], C["QKV"], C["NUM_EXPERTS"], C["TOPK"]) == \
    (2048, 48, 16, 2, 128, 384, 75968, 2560, 128, 8), C
assert C["BN"] == 64 and C["NORM_TOPK_PROB"] is True and C["EPS"] == 1e-6, C

# --- the weights Record ----------------------------------------------------------------------------------------------------------------
# `analytic.weights_type(C)`: `NUM_EXPERTS` -> `moe_layer_weights_type` (gate_w [E][H], w13 [E][2I][H], w2 [E][H][I]); `QK_NORM == "head"` ->
# q_norm / k_norm `[D]` (R19: the per-head width, not OLMoE's all-heads one); untied -> `lm_head [V][H]`.  All at the RANK shapes.

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
#: Qwen3-MoE's per-layer weight names the base map does not know -> the layer Record fields (vLLM d9105ea80 `FusedMoE.routed_experts`).
MOE_LAYER_FIELDS = {
    "mlp.gate.weight": "gate_w",                            # [E, H] bf16 -- the router (ReplicatedLinear, no bias; full E on every rank)
    "mlp.experts.routed_experts.w13_weight": "w13",         # [E, 2*I_r, H] bf16 -- gate|up of every expert, fused, intermediate rows sharded
    "mlp.experts.routed_experts.w2_weight": "w2",           # [E, H, I_r] bf16 -- down of every expert, K sharded
}
#: Base per-layer names Qwen3-MoE has no parameter for in these layers (dense MLP / QKV bias); a Root with one of these is a profile error.
_DENSE_ONLY = ("mlp.gate_up_proj.weight", "mlp.down_proj.weight", "self_attn.qkv_proj.bias")
#: parameters of MoE variants this profile does not state (shared experts, expert-parallel maps, router bias)
_REFUSED = re.compile(r"^model\.layers\.\d+\.mlp\.(shared_expert|shared_experts|shared_expert_gate|gate\.e_score_correction_bias|experts\.expert_map)")


def make_weight_field(C: dict[str, Any]) -> Callable[[str], tuple[str | int, ...] | None]:
    base_wf = base.make_weight_field(C)
    layer_type = analytic.weights_type(C).fields[analytic.weights_type(C).index("layers")][1].elem
    layer_field_names = frozenset(k for k, _ in layer_type.fields)

    def weight_field(name: str) -> tuple[str | int, ...] | None:
        m = _LAYER_RE.match(name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest in _DENSE_ONLY:
                return None
            f = MOE_LAYER_FIELDS.get(rest)
            if f is not None:
                return ("layers", layer, f) if f in layer_field_names else None
        return base_wf(name)

    return weight_field


def make_root_class(weight_field: Callable[[str], Any]) -> Callable[[str], str]:
    base_rc = base.make_root_class(weight_field)

    def root_class(name: str) -> str:
        cls = base_rc(name)
        if cls == "unknown":
            if _REFUSED.match(name):
                return "unmodelled-moe-variant"
            m = _LAYER_RE.match(name)
            if m and m.group(2) in MOE_LAYER_FIELDS:
                return "unbound-moe-weight"
        return cls

    return root_class


# --- header consistency ----------------------------------------------------------------------------------------------------------------

_MOE_HF_KEYS = {"num_experts": "NUM_EXPERTS", "num_experts_per_tok": "TOPK", "norm_topk_prob": "NORM_TOPK_PROB"}


def make_check_header(C: dict[str, Any]) -> Callable[[RunHeader], list[str]]:
    world = int(C["WORLD"])

    def check_header(header: RunHeader) -> list[str]:
        """The base check over the recorded (FULL) HF config re-sharded to this rank, minus the dense `intermediate_size`, plus the MoE keys."""
        cfg = (header.model or {}).get("config") or {}
        out: list[str] = []
        if cfg:
            try:
                rk = rank_hf_config(cfg, world)
            except (KeyError, ValueError) as e:
                return [f"recorded config cannot be sharded by {world}: {e}"]
            out += base.make_check_header(C)(dataclasses.replace(header, model={**(header.model or {}), "config": rk}))
            for hf, key in _MOE_HF_KEYS.items():
                if hf in cfg and cfg[hf] != C.get(key):
                    out.append(f"{hf}={cfg[hf]!r} != C[{key}]={C.get(key)!r}")
            if cfg.get("moe_intermediate_size") is not None and cfg["moe_intermediate_size"] != C["MOE_I_FULL"]:
                out.append(f"moe_intermediate_size={cfg['moe_intermediate_size']!r} != C[MOE_I_FULL]={C['MOE_I_FULL']!r}")
            if cfg.get("mlp_only_layers"):
                out.append(f"mlp_only_layers={cfg['mlp_only_layers']!r}: this profile states every layer as a fused-MoE layer")
        tp = (header.model or {}).get("tp") or (header.model or {}).get("tensor_parallel_size")
        if tp is not None and int(tp) != world:
            out.append(f"tensor_parallel_size={tp!r} != WORLD={world}")
        return out

    return check_header


# --- the profile -----------------------------------------------------------------------------------------------------------------------

def profile_id(world: int) -> str:
    return f"vllm-d9105ea80-sm89-eager-bf16-qwen3-30b-a3b-ovmoe" + (f"-tp{world}-rank" if world > 1 else "")


def build_profile(world: int = WORLD, moe_construction: str | None = None, *, fused_moe_pins: dict[str, Any] | None = None,
                  fused_moe_num_warps: int | None = None, fused_moe_num_stages: int | None = None) -> Profile:
    """The constructive Qwen3-30B-A3B profile for ONE rank of a `world`-way tensor-parallel engine (`world=1`: the whole model).

    `moe_construction`: None / `indexed-read` = the MOE-01 pairwise fold (`MoEFusedExperts`); `padded` is refused at fold time (the
    fixed-slot Definition states the unnormalised router).  `fused_moe_pins` / `fused_moe_num_warps` / `fused_moe_num_stages`: the
    fused_moe_kernel tiling to pin instead of `gen_ov_moe_patterns.FUSED_MOE_PINS` (a κ fact of the capture; see the module note)."""
    Cw, prov = (C, PROV) if world == WORLD else derive_config(HF_CONFIG, world)
    key = moe_construction or "indexed-read"
    if key not in MOE_CONSTRUCTION_PATTERNS:
        raise ValueError(f"moe_construction {moe_construction!r} is not one of {sorted(MOE_CONSTRUCTION_PATTERNS)}")
    wf = make_weight_field(Cw)
    p = base.build_profile(Cw, profile_id=profile_id(world), case=CASE,
                           model_note=f"Qwen3-30B-A3B @ {REVISION[:8]}" + (f" ONE RANK of TP{world} (rank-local shard configuration)" if world > 1 else "")
                                      + f": H={Cw['H']} NL={Cw['NL']} NH={Cw['NH']} KVH={Cw['KVH']} D={Cw['D']} per-head q/k RMSNorm, "
                                        f"E={Cw['NUM_EXPERTS']} top-{Cw['TOPK']} softmax RENORMALISED (norm_topk_prob), replicated router, "
                                        f"SwiGLU experts I={Cw['I']} (of {Cw['MOE_I_FULL']}), untied lm_head, V={Cw['V']}, FA2 kBlockN {Cw['BN']}")
    coll = TpCollective(world=world) if world > 1 else None
    pins = dict(FUSED_MOE_PINS if fused_moe_pins is None else fused_moe_pins)
    nw = FUSED_MOE_NUM_WARPS if fused_moe_num_warps is None else int(fused_moe_num_warps)
    ns = FUSED_MOE_NUM_STAGES if fused_moe_num_stages is None else int(fused_moe_num_stages)
    moe = MOE_CONSTRUCTION_PATTERNS[key](Cw, router_vpt={**ROUTER_VPT_BY_E, Cw["NUM_EXPERTS"]: Cw["ROUTER_VPT"]}, renormalize=True,
                                         scope_patterns=(coll,) if coll else (), fused_pins=pins, fused_num_warps=nw, fused_num_stages=ns)
    lane = [GemmLaunchTunedAda(base_constexprs=base.GEMM_CONSTEXPRS, num_sms=base.GEMM_NUM_SMS), moe]
    rank = [EmbeddingShardCuda(), coll] if coll else []
    patterns = rank + lane + list(p.patterns)
    if world > 1:
        # the sampler sees the all-gathered logits: the full vocabulary, not the rank's V/world shard
        for pt in patterns:
            if isinstance(pt, (TokenSelectGumbelTopP, TokenSelectGumbelTwoStage, TokenSelectArgmax)):
                pt.vocab = int(HF_CONFIG["vocab_size"])
    layer_fields = [n for n, _ in analytic.weights_type(Cw).fields[analytic.weights_type(Cw).index("layers")][1].elem.fields]
    notes = dict(p.notes)
    notes.update({
        "lane": "qwen3 (R19)",
        "hand_facts": {k: {"value": v, "source": "vllm/model_executor/models/qwen3_moe.py @ d9105ea80"} for k, v in HAND.items()},
        "family_rules": FAMILY_RULES,
        "config_provenance": prov,
        "moe": {"status": "constructive: Computes over registry.moe definitions per step per rank (R19 qwen3; replaces the wave-1 MoEBlockScope "
                          "Unsupported-by-design reading that kept #75's Match at collective level)",
                "scope_pattern": f"gen_ov_moe_patterns.{type(moe).__name__} (vllm.moe_forward*), renormalize=True",
                "definitions": [f"MoeRouterTopKNorm_v1{{E={Cw['NUM_EXPERTS']},TOPK={Cw['TOPK']},VPT={Cw['ROUTER_VPT']}}}",
                                f"MoeExpertGemm_v1{{E={Cw['NUM_EXPERTS']},K={Cw['H']},N={2 * Cw['I']}}}", f"SiluMul_v1{{I={Cw['I']}}} (base)",
                                f"MoeExpertGemmW_v1{{E={Cw['NUM_EXPERTS']},K={Cw['I']},N={Cw['H']}}}", f"MoeSum_v1{{TOPK={Cw['TOPK']},H={Cw['H']}}}"]
                               + ([f"AllReduce2_v1{{N={Cw['H']}}} (the block's TP all-reduce, in or after the scope)"] if world > 1 else []),
                "pins": {"fused_moe_kernel": {**FUSED_MOE_NUMERIC_PINS, **pins, "num_warps": nw, "num_stages": ns,
                                              "source": "gen_ov_moe_patterns.FUSED_MOE_PINS (vLLM BI default, M-invariant; moe-xcheck qwen3_shaped_block_e128_i768_renorm on L40S)"
                                                        if fused_moe_pins is None else "injected (build_profile(fused_moe_pins=...))"},
                         "router_vpt_by_E": {Cw["NUM_EXPERTS"]: Cw["ROUTER_VPT"]},
                         "alignment": "moe_align_block_size is Transparent(noop): a pair->expert-block schedule with no numerical effect"},
                "sharding": {"router": "replicated (full E logits on every rank)", "experts": f"intermediate axis / {world}: w13 [E, 2*{Cw['I']}, H], w2 [E, H, {Cw['I']}]",
                             "reduce": "moe_sum rank-local, then AllReduce2_v1{N=H} of the routed sum"} if world > 1 else "none (TP1)",
                "construction": key,
                "record_fields_present": sorted(f for f in ("q_norm", "k_norm", "gate_w", "w13", "w2") if f in layer_fields)},
        "root_class": "profiles.gen_ov_moe_qwen3.make_root_class(profile.weight_field)(name)",
    })
    if world > 1:
        notes["tp_rank"] = {"world": world, "rank_config_divided": list(_SHARDED_HF_KEYS), "intermediate_size_is": "moe_intermediate_size / world",
                            "recomputed": {"QKV": Cw["QKV"], "GU": Cw["GU"]}, "sampler_vocab": int(HF_CONFIG["vocab_size"]),
                            "own_patterns": [EmbeddingShardCuda.name, TpCollective.name],
                            "base": "gen_llama_llama32_1b_tp2rank (the rank facts) / gen_ov_moe_olmoe (the constructive MoE fold)",
                            "collectives": "[R17 D90] vllm.all_reduce / vllm.all_gather -> AllReduce2_v1{N} / AllGather2_v1{N} per row over (own partial, "
                                           "peer partial) in rank order; the peer partial is the fold's `peers_<request>` root parameter (root class "
                                           "collective_received), linked to the other rank's committed partial by the cross-rank leg (tp2_rank_match); "
                                           "the MoE block's all-reduce is resolved wherever vLLM issues it (inside vllm.moe_forward via "
                                           "MoEFusedExperts.scope_patterns, or after it by the outer TpCollective)"}
    return dataclasses.replace(p, id=profile_id(world), patterns=patterns, weight_field=wf, notes=notes, accepted=False)


def with_moe_construction(profile: Profile, moe_construction: str | None) -> Profile:
    """The fold's hook (run_config `fold_summary_result`): this profile under the workload's DECLARED MoE construction."""
    if moe_construction in (None, "indexed-read"):
        return profile
    return build_profile(int(profile.config["WORLD"]), moe_construction)


PROFILE = build_profile(WORLD)
weight_field = PROFILE.weight_field
root_class = make_root_class(weight_field)
check_header = make_check_header(C)
PROFILE_ID = PROFILE.id
