"""The block-FP8 shared-expert MoE scope on RECORDED events: GLM-5.3 FP8 at TP8 on 8x H200.

`fixtures/glm53_tp8_moe_l3.rank0.jsonl.gz` is layer 3 (the first MoE layer) of rank 0's observer log of run r20261013-001654-3e77
(vy-glm53-h8, B1 i256 o32 greedy, vLLM d9105ea80, eager, `VLLM_BATCH_INVARIANT=0`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, the H200's
tuned fused-MoE configs), cut by `tests/engine/moe_scope_cut.py` (its `moe_scope_cut` Note names the source log and its sha256): the
RunHeader, the layer's weight Roots, and for the prefill step (256 rows) and the first decode step (1 row) the `vllm.moe_forward_shared`
scope and the add of its two outputs after it.  The rank's widths come from the checkpoint's config.json
(`tests/check/fixtures/glm_moe_dsa/config.json`, whose sha256 is the header's `config_sha256`) through `generic.rank_hf_config` and
`family_facts`, as the Build reads them.

The recorded header lacks two facts the scope reads, because `rank_worker.tp2_install` builds its `Engine` without the workload's
declared target: `profile.target` and `backends.moe_experts_class` (written only under a declared `moe_backend`).
  * As recorded, every scope refuses by name for the missing experts class.  With the class alone, the GPU document's fallback target
    reads `batch_invariant` True against the engine's `VLLM_BATCH_INVARIANT=0`, so the tuned tile refuses against the pinned kernel.
  * With both supplied (`TARGET`: the row's declared target with `moe_backend` "triton", which the run's live inventory backs: every
    FusedMoE layer's `fp8_backend` is `Fp8MoeBackend.TRITON`, mapped to `TritonExperts` by `fused_moe/oracle/fp8.py:189-194`), each step
    folds to the gate (cuBLAS `mm.dtype` at prefill, the cuteDSL tier at decode, bound by the empty it fills), the shared expert, the
    sigmoid router, the routed input quant, both pair forms on the served wgmma tiles (BLOCK_SIZE_K 128, so KB=128, with SWAP_AB), the
    fused SiLU quant and `moe_sum`, every event of the scope claimed, and the add after it is `ResidualAdd_v1`.
"""
from __future__ import annotations

import dataclasses
import functools
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from tests.engine.moe_scope_cut import cut, write
from tests.engine.test_gen_shared_moe_fp8 import GateStandIn, _fits, gate_spec
from verity.ml.circuits.defs import bind
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.engine.profiles import generic
from verity_vllm.engine.profiles.gen_ln_patterns import ResidualAddPattern
from verity_vllm.engine.profiles.gen_ov_moe_patterns import K_FUSED, TRITON_EXPERTS
from verity_vllm.engine.profiles.gen_shared_moe_patterns import GATE_CUBLAS, GATE_CUTEDSL, MoEFusedSharedExpertsFp8
from verity_vllm.observe.events import Note, Root, RunHeader
from verity_vllm.observe.fold.patterns import triton_name
from verity_vllm.observe.fold.resolve_log import collect_roots, make_context
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Shared, Transparent, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.log import read_log, split_steps
from verity_vllm.program.frontend import targets as T
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import moe as MOE

FIXTURE = Path(__file__).parent / "fixtures" / "glm53_tp8_moe_l3.rank0.jsonl.gz"
CONFIG = Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json"
RUN, SOURCE_SHA256 = "r20261013-001654-3e77", "1df7817918cd0db7aabcdbc80b615d508c7d1d73a09645024b5689afcb514f2a"
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
WORLD, LAYER = 8, 3
PRE = f"model.layers.{LAYER}.mlp"
#: engine step -> (rows M, gate tier, whether the scope aligns, the served (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M))
STEPS = {0: (256, GATE_CUBLAS, True, (16, 128, 128, 32)), 1: (1, GATE_CUTEDSL, False, (16, 64, 128, 1))}
TARGET = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, fp8_block_gemm="cutlass", moe_backend="triton")
DOT = "HopperE4m3QgmmaDot32_v1"
WEIGHTS = {"gate": f"{PRE}.gate.weight", "bias": f"{PRE}.gate.e_score_correction_bias",
           "gate_up": f"{PRE}.shared_experts.gate_up_proj.weight", "down": f"{PRE}.shared_experts.down_proj.weight",
           "w13": f"{PRE}.experts.routed_experts.w13_weight", "w2": f"{PRE}.experts.routed_experts.w2_weight"}
Q3 = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN3_30B_A3B_tp2")


@functools.lru_cache(maxsize=None)
def config() -> tuple[dict, int]:
    """The scope's config and block G for one TP8 rank."""
    cfg = json.loads(CONFIG.read_text())
    rank = generic.rank_hf_config(cfg, WORLD)
    facts, _ = FF.resolve_facts(rank, {})
    C = {"NUM_EXPERTS": facts["NUM_EXPERTS"], "TOPK": rank["num_experts_per_tok"], "H": rank["hidden_size"], "I": rank["moe_intermediate_size"],
         "SHARED_I": facts["SHARED_I"], "ROUTED_SCALING_FACTOR": facts["ROUTED_SCALING_FACTOR"]}
    return C, cfg["quantization_config"]["weight_block_size"][0]


C, G = config()
H, I, SI, E, TOPK, RSF = C["H"], C["I"], C["SHARED_I"], C["NUM_EXPERTS"], C["TOPK"], C["ROUTED_SCALING_FACTOR"]


def profile():
    return dataclasses.replace(Q3.PROFILE, config=C, patterns=[MoEFusedSharedExpertsFp8(C, group=G, gate_spec=gate_spec), ResidualAddPattern()])


def _supplied(header: RunHeader, *, target: bool, experts: bool) -> RunHeader:
    prof = json.loads(json.dumps(header.profile))
    if target:
        prof["target"] = TARGET.to_json()
    if experts:
        prof["backends"]["moe_experts_class"] = TRITON_EXPERTS
    return dataclasses.replace(header, profile=prof)


@functools.lru_cache(maxsize=None)
def _resolved(target: bool = True, experts: bool = True):
    """step -> (StepEvents, resolutions) under the recorded header with the facts named supplied."""
    header, roots, out = None, {}, {}
    for pre, step in split_steps(read_log(FIXTURE)):
        header = next((e for e in pre if isinstance(e, RunHeader)), header)
        collect_roots(pre, roots)
        if step:
            se = step_events(step)
            out[se.start.step] = (se, resolve_step(se, make_context(_supplied(header, target=target, experts=experts), se, roots, profile())))
    return out


def test_fixture_is_layer_3_of_rank_0_of_the_h200_capture():
    events = list(read_log(FIXTURE))
    p = next(e for e in events if isinstance(e, RunHeader)).profile
    assert (H, I, SI, E, TOPK, RSF, G) == (6144, 2048 // WORLD, 2048 // WORLD, 256, 8, 2.5, 128)
    assert p["case"] == "GLM53" and (p["model"]["repo"], p["model"]["revision"]) == ("zai-org/GLM-5.3", REVISION)
    assert p["model"]["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert p["vllm"]["commit"] == "d9105ea80" and p["parallelism"]["tensor_parallel_size"] == WORLD
    assert (p["gpu"]["name"], p["gpu"]["compute_capability"], p["gpu"]["multi_processor_count"]) == ("NVIDIA H200", "9.0", 132)
    assert p["engine_flags"]["VLLM_BATCH_INVARIANT"] == "0" and p["engine_flags"]["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
    assert p["backends"]["fp8_block_gemm_kernel"] == "CutlassFp8BlockScaledMMKernel" and p["backends"]["moe_gate_ll_bf16_gemm"] is True
    assert "target" not in p and "moe_experts_class" not in p["backends"]
    (note,) = [e for e in events if isinstance(e, Note)]
    assert note.text == "moe_scope_cut" and note.data["sha256"] == SOURCE_SHA256 and note.data["origin"] == f"{RUN}:out/log.rank0.jsonl.gz"
    assert (note.data["layer"], note.data["steps"], note.data["module"]) == (LAYER, [0, 1], f"{PRE}.experts")
    want = [n + s for k, n in WEIGHTS.items() for s in (("",) if k in ("gate", "bias") else ("", "_scale_inv"))]
    assert sorted(e.name for e in events if isinstance(e, Root)) == sorted(want)
    assert {s: (se.start.phase_hint, len(se.start.rows)) for s, (se, _) in _resolved().items()} == {0: ("prefill", 256), 1: ("decode", 1)}


def test_the_fixture_is_a_fixed_point_of_the_cutter(tmp_path):
    events = list(read_log(FIXTURE))
    note = next(e for e in events if isinstance(e, Note))
    write(cut(events, LAYER, set(STEPS), {k: note.data[k] for k in ("origin", "sha256")}), tmp_path / "recut.jsonl.gz")
    assert gzip.decompress((tmp_path / "recut.jsonl.gz").read_bytes()) == gzip.decompress(FIXTURE.read_bytes())


@pytest.mark.parametrize("step", sorted(STEPS))
def test_as_recorded_every_scope_refuses_by_name_for_the_facts_the_tp_capture_omits(step):
    _, res = _resolved(target=False, experts=False)[step]
    assert isinstance(res[0], Unsupported) and f"does not attest {TRITON_EXPERTS}" in res[0].reason, res[0]
    _, res = _resolved(target=False, experts=True)[step]
    assert isinstance(res[0], Unsupported) and "fused_moe_kernel numeric constexprs differ" in res[0].reason, res[0]
    assert res[0].detail["mismatches"]["SWAP_AB"] == {"observed": True, "pinned": False}


@pytest.mark.parametrize("step", sorted(STEPS))
def test_recorded_shared_scope_folds_on_the_served_wgmma_tiles_with_the_declared_target(step):
    se, res = _resolved()[step]
    M, tier, aligned, tile = STEPS[step]
    P = M * TOPK
    scope = se.roots[0]
    assert len(se.roots) == 2 and len(res) == 2 and isinstance(res[0], Group), [getattr(r, "reason", r) for r in res]
    g = res[0]
    assert g.consumed == tuple(sorted(scope.ids())) and g.pattern == "moe.fused_shared_experts.pairwise.fp8"
    launches = [n for n in scope.walk() if n.is_triton and triton_name(n) == K_FUSED]
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")
    assert [tuple(n.enter.constexprs[k] for k in keys) for n in launches] == [tile] * 2
    assert [n.enter.constexprs["SWAP_AB"] for n in launches] == [True, True]
    computes = [p for p in g.parts if isinstance(p, Compute)]
    fp8 = f"G={G},KB={tile[2]},SWAP=True,DOT={DOT}"
    assert [c.spec.id for c in computes] == [
        bind(GateStandIn, K=H, N=E, TIER=tier).id,
        f"Fp8GroupQuant_v1{{K={H},G={G}}}", f"ScaledMmFp8BlockSharedScale_v1{{K={H},N={2 * SI},G={G}}}", f"SiluMul_v2{{I={SI}}}",
        f"Fp8GroupQuant_v1{{K={SI},G={G}}}", f"ScaledMmFp8BlockSharedScale_v1{{K={SI},N={H},G={G}}}",
        bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=RSF, KERNEL="block").id,
        f"Fp8GroupQuant_v1{{K={H},G={G}}}", f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},{fp8}}}", f"SiluMulFp8BlockQuant_v1{{I={I},G={G}}}",
        f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},{fp8}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    assert computes[2].spec.id == T.scaled_mm_fp8_block_spec(TARGET, H, 2 * SI, G).id
    assert [c.rows for c in computes] == [M] * 8 + [P, P, P, M]
    assert all(_fits(c) for c in computes)
    gc, q1, m1, silu, q2, m2, router, qr, up, sq, down, msum = computes
    hidden, (shared_out, routed_out) = scope.enter.args[0], scope.output_views()
    assert gc.operands[0].view == hidden and gc.roots == (None, WEIGHTS["gate"]) and gc.kernels == (f"{tier}_gate_kernel",)
    assert router.operands[0].view == gc.outputs[0].view and router.roots == (None, WEIGHTS["bias"])
    assert q1.operands[0].view == hidden and m1.operands[0].view == q1.outputs[0].view and silu.operands[0].view == m1.outputs[0].view
    assert q2.operands[0].view == silu.outputs[0].view and m2.operands[0].view == q2.outputs[0].view
    assert m2.outputs[0].view.alloc == shared_out.alloc and msum.outputs[0].view.alloc == routed_out.alloc
    assert m1.roots[2:] == (WEIGHTS["gate_up"], WEIGHTS["gate_up"] + "_scale_inv") and m2.roots[2:] == (WEIGHTS["down"], WEIGHTS["down"] + "_scale_inv")
    assert qr.operands[0].view == hidden and [type(o) for o in up.operands] == [Gather, Gather, PerRow, Shared, Shared]
    assert up.operands[0].indices == tuple(p // TOPK for p in range(P)) and up.operands[2].view.alloc == router.outputs[1].view.alloc
    assert (down.operands[0].view, down.operands[1].view) == (sq.outputs[0].view, sq.outputs[1].view)
    assert down.operands[3].view.alloc == router.outputs[0].view.alloc
    assert up.roots[3:] == (WEIGHTS["w13"], WEIGHTS["w13"] + "_scale_inv") and down.roots[4:] == (WEIGHTS["w2"], WEIGHTS["w2"] + "_scale_inv")
    assert sum(isinstance(p, Transparent) and p.effect == "noop" for p in g.parts) == int(aligned)
    add = res[1]
    assert isinstance(add, Compute) and add.spec.id == f"ResidualAdd_v1{{N={H}}}" and add.rows == M
    assert [o.view.alloc for o in add.operands] == [shared_out.alloc, routed_out.alloc]
