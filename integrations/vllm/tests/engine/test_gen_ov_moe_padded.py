"""The FOLD-side twin of the padded MoE construction: under the workload's DECLARED
`target.moe_construction = padded`, the record's six MoE kernels (`vllm.moe_forward*`: topk_softmax -> align -> fused_moe_kernel x2
-> silu_and_mul -> moe_sum) fold to ONE `MoeFfnFixedL_v1{E,TOPK,VPT,H,I}` Compute per token -- the Definition the frontend's
`PaddedMoeSumRule` binds, so the global match compares Build and record in one vocabulary.  The construction is selected by the declaration
(profile hook `with_moe_construction`, run_config `--moe-construction` from the manifest), never inferred from the trace; the
pairwise (MOE-01) reading's refusals are the padded reading's refusals, verbatim (same pins, same shape checks)."""

from __future__ import annotations

import importlib

import json
import sys
import types

import pytest

from verity_vllm.observe.fold.profiles.gen_ov_moe_patterns import MOE_CONSTRUCTION_PATTERNS, MoEFusedExperts, MoEFusedExpertsPadded, ov_moe_patterns

OV = importlib.import_module("verity_vllm.observe.fold.profiles.derived_OLMOE")
from verity_vllm.observe.fold.resolver import Compute, Group, StepContext, Transparent, Unsupported, resolve_step
from tests.engine.test_gen_ov_moe import C, E, H, I, TOPK, _ctx, _start, moe_scope_events
from verity_vllm.observe.fold.tree import step_events
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import moe_pad as MP

PADDED = OV.with_moe_construction(OV.PROFILE, "padded")
KERNEL_ORDER = OV.with_moe_construction(OV.PROFILE, "indexed-read")
FIXED_ID = f"MoeFfnFixedL_v1{{E={E},TOPK={TOPK},VPT=8,H={H},I={I}}}"


def _resolve_padded(M: int, profile=PADDED, **kw):
    start_kw = {k: kw.pop(k) for k in ("padded",) if k in kw}
    evs = moe_scope_events(M, **kw)
    start = _start(M, **start_kw)
    se = step_events([start, *evs])
    c = _ctx(start)
    ctx = StepContext(c.header, c.start, c.sample, c.links, c.roots, c.config, profile)    # the same facts, the PADDED profile
    return resolve_step(se, ctx), se


@pytest.mark.parametrize("M,naive", [(3, False), (32, False), (1, True)])
def test_padded_scope_folds_to_one_moe_ffn_fixed_compute_per_token(M, naive):
    res, se = _resolve_padded(M, naive=naive)
    assert len(res) == 1 and isinstance(res[0], Group), res
    g = res[0]
    assert g.pattern == "moe.fused_experts.padded" and g.consumed == tuple(sorted(se.roots[0].ids()))       # the whole scope is claimed
    computes = [p for p in g.parts if isinstance(p, Compute)]
    assert len(computes) == 1 and computes[0] is g.parts[-1]                                                  # ONE Compute, after the glue
    one = computes[0]
    assert one.spec.id == FIXED_ID and one.rows == M and one.pattern == "moe.fused_experts.padded"
    assert [type(o).__name__ for o in one.operands] == ["PerRow", "PerRow", "Shared", "Shared"]
    x, logits, w13, w2 = one.operands
    assert x.view.alloc == 295 and x.view.shape == (M, H) and x.view.dtype == "bfloat16"                     # hidden_states rows
    assert logits.view.alloc == 296 and logits.view.shape == (M, E) and logits.view.dtype == "bfloat16"      # router_logits rows (the gate GEMM's output)
    assert w13.view.shape == (E, 2 * I, H) and w2.view.shape == (E, H, I)                                    # the whole expert tables
    assert [type(o).__name__ for o in one.outputs] == ["PerRow"] and one.outputs[0].view.alloc == 300 and one.outputs[0].view.shape == (M, H)
    assert one.roots == (None, None, "model.layers.0.mlp.experts.routed_experts.w13_weight", "model.layers.0.mlp.experts.routed_experts.w2_weight")
    assert len(one.attribution) == M                                                                          # per-row (request, position)
    # kernel census: every launch of the scope is claimed exactly as the pairwise reading claims it
    kernels = [k for p in g.parts for k in p.kernels]
    assert "topkGating" in kernels and "fused_moe_kernel" in kernels and "act_and_mul_kernel" in kernels and "moe_sum_vec_kernel" in kernels
    aligns = [p for p in g.parts if isinstance(p, Transparent) and any("moe_align_block_size" in k for k in p.kernels)]
    assert len(aligns) == (0 if naive else 1) and all(t.effect == "noop" for t in aligns)
    # interior values are NOT record instances: no router / pair-GEMM / silu / sum Compute anywhere in the group
    assert not [p for p in g.parts if isinstance(p, Compute) and p.spec.name in ("MoeRouterTopK", "MoeExpertGemm", "MoeExpertGemmW", "SiluMul", "MoeSum")]


def test_padded_compute_is_the_frontend_definition_and_its_operand_leaf_counts_are_the_views():
    """The fold's spec IS the Definition the padded frontend rule binds (vocab kind `moe_ffn_fixed`): one id, one digest, one lift of
    record (`LMoeFfnL_v1`, hand-registered) -- the global match compares Build and record in that vocabulary."""
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V
    from verity.ml.definitions.registry import lifted as L
    fold_spec = bind(MP.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=8, H=H, I=I)
    front_spec = V.kinds["moe_ffn_fixed"](E, TOPK, 8, H, I)
    assert fold_spec.id == front_spec.id == FIXED_ID and fold_spec.defn is front_spec.defn and fold_spec.bindings == front_spec.bindings
    assert L.lift_callee(fold_spec).name == "LMoeFfnL"
    sig, out = fold_spec.signature()
    assert [t.leaves for _, t in sig] == [H, E, E * 2 * I * H, E * H * I] and out.leaves == H                 # = the operand views' numel


@pytest.mark.parametrize("kw,needle", [
    (dict(renormalize=True), "renormalize=False"),
    (dict(consts_up={"use_fp8_w8a8": True}), "numeric constexprs"),
    (dict(consts_down={"SPLIT_K": 2}), "numeric constexprs"),
    (dict(consts_up={"BLOCK_SIZE_M": 16}), "tiling"),
    (dict(num_stages=4), "tiling"),
    (dict(swap_launches=True), "MUL_ROUTED_WEIGHT"),
    (dict(padded=8), "padding rows"),
])
def test_padded_scope_declines_exactly_as_the_pairwise_reading(kw, needle):
    res, _ = _resolve_padded(3, **kw)
    assert len(res) == 1 and isinstance(res[0], Unsupported), res
    assert needle in res[0].reason, res[0].reason
    pair, _ = _resolve_padded(3, profile=KERNEL_ORDER, **kw)
    assert pair[0].reason == res[0].reason                                              # the same refusal as the kernel-order router's, verbatim


def test_the_construction_is_declared_and_selected_by_name_never_inferred(tmp_path):
    # the profile hook: undeclared / the default = the profile as is; padded swaps the scope pattern and nothing else
    assert OV.with_moe_construction(OV.PROFILE, None) is OV.PROFILE and OV.with_moe_construction(OV.PROFILE, "indexed-read-ordered") is OV.PROFILE
    assert isinstance(OV.PROFILE.patterns[1], MoEFusedExperts) and not isinstance(OV.PROFILE.patterns[1], MoEFusedExpertsPadded)
    assert isinstance(PADDED.patterns[1], MoEFusedExpertsPadded) and PADDED.patterns[1].name == "moe.fused_experts.padded"
    assert [type(p).__name__ for p in PADDED.patterns[2:]] == [type(p).__name__ for p in OV.PROFILE.patterns[2:]]
    assert PADDED.id == OV.PROFILE.id and PADDED.kernel_allowlist == OV.PROFILE.kernel_allowlist and PADDED.notes["moe"]["construction"] == "padded"
    assert set(MOE_CONSTRUCTION_PATTERNS) == {"indexed-read", "padded", "indexed-read-ordered"}
    with pytest.raises(ValueError):
        ov_moe_patterns(C, gemm_constexprs={}, gemm_num_sms=1, moe_construction="fixed-slot")
    # run_config reads the declaration from the row file (engine.target.moe_construction) and validates it by the frontend's table
    from verity_vllm.pipeline.match import _workload_moe_construction
    base = {"schema": "x", "requests": [], "sampling": {}}
    for name, target, want in (("none", None, None), ("pad", {"moe_construction": "padded"}, "padded"), ("ir", {"moe_construction": "indexed-read"}, "indexed-read")):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({**base, **({"engine": {"target": target}} if target else {})}))
        assert _workload_moe_construction(str(p)) == want
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({**base, "engine": {"target": {"moe_construction": "MOE-01"}}}))
    with pytest.raises(SystemExit):
        _workload_moe_construction(str(p))


def test_fold_refuses_a_declared_construction_the_profile_cannot_fold():
    """A `padded` declaration on a profile without the hook FAILS by name before the log is read: the MOE-01 fold is never a silent stand-in."""
    from verity_vllm.pipeline.match import fold_summary_result
    fake = types.ModuleType("verity_vllm.observe.fold.profiles._moe_test_nohook")
    fake.PROFILE = OV.PROFILE
    sys.modules["verity_vllm.observe.fold.profiles._moe_test_nohook"] = fake
    try:
        with pytest.raises(SystemExit, match="with_moe_construction"):
            fold_summary_result("/nonexistent/log.jsonl.gz", "_moe_test_nohook", moe_construction="padded")
    finally:
        del sys.modules["verity_vllm.observe.fold.profiles._moe_test_nohook"]
