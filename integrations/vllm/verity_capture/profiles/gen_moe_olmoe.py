"""Profile OLMOE — `allenai/OLMoE-1B-7B-0924` (`OlmoeForCausalLM`) on the pinned κ (vLLM d9105ea80, sm89, eager, bf16).

Lane `moe` of the generalization sweep (`out/gen/LANE_BRIEF.md`).  The dense parts of OLMoE are Llama-shaped and reuse
the base sm89 module unchanged: `analytic.config_of` derives `C` from the checkpoint's `config.json` (H=2048, NL=16,
NH=KVH=16, D=128, I=1024 per expert, V=50304, MAX_POS=4096, EPS=1e-5 as the family's "hand" fact, NUM_EXPERTS=64,
TOPK=8, NORM_TOPK_PROB=False, untied lm_head), the κ pins / kernel allowlist / base pattern list come from
`vllm_d9105ea80_sm89_eager.build_profile`.

What this module adds (additively, nothing of the base module is edited):
  * the family facts vLLM hard-codes in `vllm/model_executor/models/olmoe.py` and that `config.json` does not carry
    (`hand`): `QK_NORM=True` (`OlmoeAttention` applies `RMSNorm(NH*D, eps=1e-5)` to q and to k over ALL heads at once,
    before RoPE), `RENORMALIZE=False` (`OlmoeMoE` builds its `FusedMoE` with `renormalize=False`, i.e. the top-8 softmax
    weights are NOT renormalised — consistent with `norm_topk_prob=false`), `CLIP_QKV=None`;
  * the `MoEBlockScope` pattern (`gen_moe_patterns.py`): the `vllm.moe_forward` custom-op scope, resolved `Unsupported`
    BY DESIGN with the complete tensor / kernel metadata of its body (mechanism note `out/gen/lanes/moe/MECHANISM_MOE.md`);
  * the `GemmLaunchTunedAda` specialization (`gen_moe_patterns.py`): `o_proj` (N=K=2048) is the first GEMM of the sweep
    whose `(N, K)` is in vLLM's tuned sm89 batch-invariant matmul table, so its launches carry 32x128 (decode) / 16x128
    (32-row prefill) tiles with 5 pipeline stages instead of the base 128x128 / 3; same kernel, same k-ascending body
    `Gemm_v1{K=2048,N=2048}` (checked by the tier-b replay on the captured operands);
  * `weight_field` entries for OLMoE's extra weights — `self_attn.{q,k}_norm.weight`, `mlp.gate.weight`,
    `mlp.experts.{w13,w2}_weight` — bound ONLY when `analytic.weights_type(C)` has the corresponding layer field
    (`q_norm`, `k_norm`, `gate_w`, `w13`, `w2`: the SHARED-CODE proposal `PROPOSED: analytic.weights_type MoE / qk-norm
    fields`).  Under the frozen Record they are reported as `unbound root` notes by the fold and as class
    "unbound-moe-weight" by `root_class` (never silently dropped, never turned into inputs);
  * a `check_header` that accepts the untied output projection (the base check flags it because the base profile maps
    `lm_head.weight` onto `embed`) and checks the MoE keys of the recorded HF config against `C`.

Expected resolution on the pinned κ (see out/gen/lanes/moe/REPORT.md): every dense event resolves with base patterns;
exactly one `Unsupported` per layer per step (`vllm.moe_forward.default`), whose census residue is the MoE kernel list.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Callable

from .. import analytic
from ..events import RunHeader
from ..resolver import Profile
from . import vllm_d9105ea80_sm89_eager as base
from .gen_moe_patterns import moe_patterns

REPO = Path(__file__).resolve().parents[2]
MODEL = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"     # manifests/checkpoints.json OLMOE
PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-olmoe-1b-7b"


def _find_config() -> Path:
    """`data/hf_configs/OLMOE.config.json` (tracked; shipped to pods by `record_v5/ship.sh` since the int-9 ship roots and in the
    R17 sparse worktrees since `out/gen/r17/sparse-patterns.txt` re-includes it), else the seeded HF snapshot's `config.json`
    (`HF_HOME=/workspace/hf`; the only path on pod trees shipped before that)."""
    import os
    cands = [REPO / "data" / "hf_configs" / "OLMOE.config.json"]
    for home in (os.environ.get("HF_HOME"), Path.home() / ".cache" / "huggingface"):
        if home:
            cands.append(Path(home) / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION / "config.json")
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"OLMoE config.json: none of {[str(c) for c in cands]}")


OLMOE_CONFIG = _find_config()

#: Family facts of vLLM's `OlmoeForCausalLM` (d9105ea80) that config.json does not state.  Ground truth: the pinned
#: `vllm/model_executor/models/olmoe.py` (read on the pod; quoted in out/gen/lanes/moe/MECHANISM_MOE.md §1).
HAND = {
    "QK_NORM": True,          # OlmoeAttention: self.q_norm = RMSNorm(hidden_size, eps=1e-5); self.k_norm likewise (all heads)
    "QK_NORM_EPS": 1e-5,      # same hard-coded epsilon as the layer norms
    "RENORMALIZE": False,     # OlmoeMoE: FusedMoE(..., renormalize=False) -> top-k softmax weights used as they are
    "CLIP_QKV": None,         # config.json clip_qkv=null -> no clamp on the QKV projection output
}

C, PROV = analytic.config_of(OLMOE_CONFIG, model=MODEL, revision=REVISION, hand=HAND)
C = base.with_geometry(C)

# --- the weights Record: MoE / qk-norm fields exist only when the SHARED-CODE proposal is applied ----------------------

WEIGHTS_TYPE = analytic.weights_type(C)
_LAYER_TYPE = WEIGHTS_TYPE.fields[WEIGHTS_TYPE.index("layers")][1].elem
LAYER_FIELD_NAMES = frozenset(k for k, _ in _LAYER_TYPE.fields)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
#: OLMoE's per-layer weight names that the base map does not know -> the PROPOSED layer fields.
MOE_LAYER_FIELDS = {
    "self_attn.q_norm.weight": "q_norm",                    # [NH*D] bf16
    "self_attn.k_norm.weight": "k_norm",                    # [KVH*D] bf16
    "mlp.gate.weight": "gate_w",                            # [E, H] bf16 — the router (ReplicatedLinear, no bias)
    "mlp.experts.routed_experts.w13_weight": "w13",         # [E, 2*I, H] bf16 — gate|up of every expert, fused (vLLM d9105ea80
    "mlp.experts.routed_experts.w2_weight": "w2",           # [E, H, I] bf16 — down of every expert; `FusedMoE.routed_experts`)
}
#: Base per-layer names that OLMoE has no parameter for (dense MLP); a Root with one of these is a profile error.
_DENSE_ONLY = ("mlp.gate_up_proj.weight", "mlp.down_proj.weight", "self_attn.qkv_proj.bias")


def make_weight_field(C: dict[str, Any]) -> Callable[[str], tuple[str | int, ...] | None]:
    base_wf = base.make_weight_field(C)

    def weight_field(name: str) -> tuple[str | int, ...] | None:
        m = _LAYER_RE.match(name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest in _DENSE_ONLY:
                return None
            f = MOE_LAYER_FIELDS.get(rest)
            if f is not None:
                return ("layers", layer, f) if f in LAYER_FIELD_NAMES else None
        return base_wf(name)

    return weight_field


weight_field = make_weight_field(C)
_base_root_class = base.make_root_class(weight_field)


def root_class(name: str) -> str:
    """Base classes plus "unbound-moe-weight": an OLMoE weight the frozen Record has no field for (would be "weight"
    once `PROPOSED: analytic.weights_type MoE / qk-norm fields` is applied)."""
    cls = _base_root_class(name)
    if cls == "unknown":
        m = _LAYER_RE.match(name)
        if m and m.group(2) in MOE_LAYER_FIELDS:
            return "unbound-moe-weight"
    return cls


# --- header consistency ------------------------------------------------------------------------------------------

_MOE_HF_KEYS = {"num_experts": "NUM_EXPERTS", "num_experts_per_tok": "TOPK", "norm_topk_prob": "NORM_TOPK_PROB"}


def check_header(header: RunHeader) -> list[str]:
    """The base check minus its tied-embedding requirement (OLMoE's lm_head is a field of its own), plus the MoE keys."""
    out = [m for m in base.make_check_header(C)(header) if not m.startswith("tie_word_embeddings=False")]
    cfg = (header.model or {}).get("config") or {}
    for hf, key in _MOE_HF_KEYS.items():
        if hf in cfg and cfg[hf] != C.get(key):
            out.append(f"{hf}={cfg[hf]!r} != C[{key}]={C.get(key)!r}")
    if cfg.get("clip_qkv") is not None:
        out.append(f"clip_qkv={cfg['clip_qkv']!r}: this profile assumes no QKV clamp (C[CLIP_QKV]=None)")
    return out


# --- the profile ---------------------------------------------------------------------------------------------------

def build_profile() -> Profile:
    p = base.build_profile(C, profile_id=PROFILE_ID, case="OLMOE",
                           model_note="OLMoE-1B-7B-0924: 16 layers, H=2048, 16 heads x 128 (no GQA), qk-norm, 64 experts "
                                      "top-8 (softmax, no renormalisation), SwiGLU experts I=1024, untied lm_head, V=50304")
    notes = dict(p.notes)
    notes.update({
        "lane": "moe",
        "hand_facts": {k: {"value": v, "source": "vllm/model_executor/models/olmoe.py @ d9105ea80"} for k, v in HAND.items()},
        "config_provenance": PROV,
        "moe": {"status": "Unsupported by design (mechanism problem, RULES §6)", "note": "out/gen/lanes/moe/MECHANISM_MOE.md",
                "scope_pattern": "gen_moe_patterns.MoEBlockScope (vllm.moe_forward*)",
                "record_fields_present": sorted(f for f in ("q_norm", "k_norm", "gate_w", "w13", "w2") if f in LAYER_FIELD_NAMES),
                "record_fields_proposed": ["q_norm", "k_norm", "gate_w", "w13", "w2"]},
        "root_class": "profiles.gen_moe_olmoe.root_class(name)",
    })
    lane = moe_patterns(C, gemm_constexprs=base.GEMM_CONSTEXPRS, gemm_num_sms=base.GEMM_NUM_SMS)
    return dataclasses.replace(p, patterns=lane + list(p.patterns), weight_field=weight_field, notes=notes)


PROFILE = build_profile()
