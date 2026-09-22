"""Profile OLMOE (lane ov-moe, swarm 2026-09-17) — `allenai/OLMoE-1B-7B-0924` on the pinned κ with the MoE block as
pair-major Computes over the quarantine definitions `verity_vllm.registry.quarantine.ov_moe.moe`.

Everything but the pattern list is wave 1's `gen_moe_olmoe` (C from the checkpoint's config.json + the family's hand
facts, the weights Record with the `gate_w` / `w13` / `w2` / `q_norm` / `k_norm` fields, the untied-lm_head header check,
`root_class`).  The pattern list replaces `MoEBlockScope` (recognised Unsupported) by `MoEFusedExperts`
(`gen_ov_moe_patterns`): router -> alignment (schedule, transparent) -> pair GEMM -> silu -> weighted pair GEMM -> sum.

Select explicitly (`run_config.py --profile gen_ov_moe_olmoe --case OLMOE`): two modules declare case OLMOE and the
case scan of `profiles.profile_for_case` returns the first in module order (wave 1's).

Status of the numerical claim: `MoeRouterTopK_v1`'s exponential is `NvExpf_v1` — the CUDA math-library `expf` sequence
(cov-moe, 2026-09-17; bit-exact on the recorded `topk_softmax` cases; lane ov-moe's correctly rounded stand-in matched
160/192 weight words); `MoeExpertGemm_v1` / `MoeExpertGemmW_v1` / `MoeSum_v1` are the dense bodies re-used per (token, slot)
pair and are `not-yet-established` until Tier A compares them on the captured operands.  The profile stays NOT accepted
(`accepted=False`) until the coverage cells' Tier A evidence is in (out/gen/COVERAGE-20260917.md, OLMoE row).
"""

from __future__ import annotations

import dataclasses

from ..resolver import Profile
from . import gen_moe_olmoe as w1
from . import vllm_d9105ea80_sm89_eager as base
from .gen_ov_moe_patterns import FUSED_MOE_NUMERIC_PINS, FUSED_MOE_NUM_STAGES, FUSED_MOE_NUM_WARPS, FUSED_MOE_PINS, ROUTER_VPT_BY_E, ov_moe_patterns

MODEL = w1.MODEL
REVISION = w1.REVISION
PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-olmoe-1b-7b-ovmoe"
C = w1.C
weight_field = w1.weight_field
root_class = w1.root_class
check_header = w1.check_header


def build_profile(moe_construction: str | None = None) -> Profile:
    """`moe_construction` (R17-42 route A (4)): None / `indexed-read` = the MOE-01 pairwise fold (`MoEFusedExperts`); `padded` = one
    `MoeFfnFixedL_v1` Compute per token (`MoEFusedExpertsPadded`).  The workload's DECLARED construction, passed down by run_config."""
    p = w1.build_profile()
    notes = dict(p.notes)
    notes.update({
        "lane": "ov-moe",
        "wave1_profile": w1.PROFILE_ID,
        "moe": {"status": "Computes over quarantine definitions (ov-moe/MoeExpertGemm@v1); router exp = NvExpf_v1 (the CUDA expf "
                          "sequence, bit-exact on the recorded topk_softmax cases; cov-moe 2026-09-17)",
                "note": "out/gen/lanes/ov-moe/REPORT.md",
                "scope_pattern": "gen_ov_moe_patterns.MoEFusedExperts (vllm.moe_forward*)",
                "definitions": ["MoeRouterTopK_v1{E,TOPK,VPT} (NvExpf_v1)", "MoeExpertGemm_v1{E,K,N}", "MoeExpertGemmW_v1{E,K,N}",
                                "MoeSum_v1{TOPK,H}", "SiluMul_v1{I} (base)"],
                "pins": {"fused_moe_kernel": {**FUSED_MOE_NUMERIC_PINS, **FUSED_MOE_PINS, "num_warps": FUSED_MOE_NUM_WARPS,
                                              "num_stages": FUSED_MOE_NUM_STAGES}, "router_vpt_by_E": ROUTER_VPT_BY_E,
                         "alignment": "moe_align_block_size is Transparent(noop): a pair->expert-block schedule with no numerical effect"},
                "record_fields_present": notes.get("moe", {}).get("record_fields_present")},
        "root_class": "profiles.gen_ov_moe_olmoe.root_class(name)",
    })
    if moe_construction not in (None, "indexed-read"):
        notes["moe"] = {**notes["moe"], "construction": moe_construction,
                        "scope_pattern": "gen_ov_moe_patterns.MoEFusedExpertsPadded (vllm.moe_forward*): ONE MoeFfnFixedL_v1{E,TOPK,VPT,H,I} per token",
                        "definitions": ["MoeFfnFixedL_v1{E,TOPK,VPT,H,I} (registry/moe_pad; router + all E fixed expert slots + MoeSum inside)"],
                        "declared_by": "the workload manifest's target.moe_construction (a Program static of record) -- never read off the trace"}
    # drop wave 1's two lane patterns (GemmLaunchTunedAda is re-added by ov_moe_patterns; MoEBlockScope is replaced)
    base_patterns = [pt for pt in p.patterns if type(pt).__name__ not in ("GemmLaunchTunedAda", "MoEBlockScope")]
    lane = ov_moe_patterns(C, gemm_constexprs=base.GEMM_CONSTEXPRS, gemm_num_sms=base.GEMM_NUM_SMS, moe_construction=moe_construction)
    return dataclasses.replace(p, id=PROFILE_ID, patterns=lane + base_patterns, notes=notes, accepted=False)


def with_moe_construction(profile: Profile, moe_construction: str | None) -> Profile:
    """The fold's hook (run_config `fold_summary_result`): this profile under the workload's DECLARED MoE construction.  The base profile
    (`PROFILE`) is the undeclared = `indexed-read` one; `padded` swaps the scope pattern and nothing else (same pins, same base patterns)."""
    if moe_construction in (None, "indexed-read"):
        return profile
    return build_profile(moe_construction)


PROFILE = build_profile()
