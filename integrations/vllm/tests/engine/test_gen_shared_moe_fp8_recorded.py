"""The block-FP8 shared-expert MoE scope on RECORDED events: GLM-5.3 FP8 at TP8 on 8x H200, vLLM's default kernels.

`fixtures/glm53_tp8_moe_l3.rank0.jsonl.gz` is layer 3 (the first MoE layer) of rank 0's observer log of the Match capture
`sweep-auto/20261015T085245Z` on the 8x H200 node (B1 i256 o32 greedy, vLLM d9105ea80, eager, `VLLM_BATCH_INVARIANT=0`,
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, Build tree b2bf72032; the source log is art:1964424e2964937273c9fb727864e025f02b3326fa6a63c13ef8630afc69f931),
cut by `tests/engine/moe_scope_cut.py` (its `moe_scope_cut` Note names the source log and its sha256): the RunHeader, the layer's
weight Roots, and for the prefill step (256 rows) and the first decode step (1 row) the `vllm.moe_forward_shared` scope and the add
of its two outputs after it.  The rank's widths come from the checkpoint's config.json (`tests/check/fixtures/glm_moe_dsa/config.json`,
whose sha256 is the header's `config_sha256`) through `generic.rank_hf_config` and `family_facts`, as the Build reads them.

The header attests the declared target and `TritonExperts`, so each step folds as recorded, with no `gate_spec`, to the Build's gate
(cuBLAS split-K `mm.dtype` at prefill, the cuteDSL dotprod tier at decode, bound by the empty it fills), the shared expert's dynamic
block-FP8 linears and SiLU, the sigmoid router, the routed input quant, both pair forms on the served tiles, the routed SiLU and its
quant and `moe_sum`, every event of the scope claimed, and the add after it is `ResidualAdd_v1`.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import dataclasses
import functools
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from tests.engine.moe_scope_cut import cut, write
from tests.engine.test_gen_shared_moe_fp8 import _fits
from verity.ml.circuits.defs import bind
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.observe.fold.profiles import generic
from verity_vllm.observe.fold.profiles.gen_ln_patterns import ResidualAddPattern
from verity_vllm.observe.fold.profiles.gen_ov_moe_patterns import K_FUSED, TRITON_EXPERTS
from verity_vllm.observe.fold.profiles.gen_shared_moe_patterns import MoEFusedSharedExpertsFp8
from verity_vllm.observe.events import Note, Root, RunHeader
from verity_vllm.observe.fold.patterns import triton_name
from verity_vllm.observe.fold.resolve_log import collect_roots, make_context
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Shared, Transparent, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.log import read_log, split_steps
from verity.ml.definitions.registry import moe as MOE

FIXTURE = Path(__file__).parent / "fixtures" / "glm53_tp8_moe_l3.rank0.jsonl.gz"
CONFIG = Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json"
SOURCE_SHA256 = "1f98896722b560489077bb3e1b0c07ba22d51cd5039e6c2cc22db1f0c6a177df"
ORIGIN = ("vy-glm53-h8:/workspace/fs/glm53-failfast/sweep-auto/20261015T085245Z/glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager/"
          "match/capture/log.rank0.jsonl.gz")
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
WORLD, LAYER = 8, 3
PRE = f"model.layers.{LAYER}.mlp"
#: engine step -> (rows M, the gate's Definition, whether the scope aligns, the served (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
#: GROUP_SIZE_M), SWAP_AB, the tile's e4m3 step)
STEPS = {0: (256, "CublasSplitKGemmF32Out_v1{K=6144,N=256,SPLIT_K=3,DOT=HopperBF16WgmmaDot16_v1}", True, (16, 128, 128, 32), True,
             "HopperE4m3QgmmaDot32_v1"),
         1: (1, "LlBf16DotprodGemm_v1{N=256,K=6144,BS=128,VEC=8,FMA=1}", False, (8, 32, 128, 1), False, "HopperE4m3MmaV2Dot32_v1")}
WEIGHTS = {"gate": f"{PRE}.gate.weight", "bias": f"{PRE}.gate.e_score_correction_bias",
           "gate_up": f"{PRE}.shared_experts.gate_up_proj.weight", "down": f"{PRE}.shared_experts.down_proj.weight",
           "w13": f"{PRE}.experts.routed_experts.w13_weight", "w2": f"{PRE}.experts.routed_experts.w2_weight"}
Q3 = importlib.import_module("verity_vllm.observe.fold.profiles.derived_QWEN3_30B_A3B_tp2")


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
    return dataclasses.replace(Q3.PROFILE, config=C, patterns=[MoEFusedSharedExpertsFp8(C, group=G), ResidualAddPattern()])


@functools.lru_cache(maxsize=None)
def _resolved():
    """step -> (StepEvents, resolutions) under the recorded header."""
    header, roots, out = None, {}, {}
    for pre, step in split_steps(read_log(FIXTURE)):
        header = next((e for e in pre if isinstance(e, RunHeader)), header)
        collect_roots(pre, roots)
        if step:
            se = step_events(step)
            out[se.start.step] = (se, resolve_step(se, make_context(header, se, roots, profile())))
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
    assert p["backends"]["fp8_block_gemm_kernel"] == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel" and p["backends"]["moe_gate_ll_bf16_gemm"] is True
    assert p["backends"]["moe_experts_class"] == TRITON_EXPERTS
    assert (p["target"]["compute_capability"], p["target"]["batch_invariant"], p["target"]["moe_backend"]) == ([9, 0], False, "triton")
    (note,) = [e for e in events if isinstance(e, Note)]
    assert note.text == "moe_scope_cut" and note.data["sha256"] == SOURCE_SHA256 and note.data["origin"] == ORIGIN
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
def test_recorded_shared_scope_folds_as_recorded_on_the_served_tiles(step):
    se, res = _resolved()[step]
    M, gate, aligned, tile, swap, dot = STEPS[step]
    P = M * TOPK
    scope = se.roots[0]
    assert len(se.roots) == 2 and len(res) == 2 and isinstance(res[0], Group), [getattr(r, "reason", r) for r in res]
    g = res[0]
    assert g.consumed == tuple(sorted(scope.ids())) and g.pattern == "moe.fused_shared_experts.pairwise.fp8"
    launches = [n for n in scope.walk() if n.is_triton and triton_name(n) == K_FUSED]
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")
    assert [tuple(n.enter.constexprs[k] for k in keys) for n in launches] == [tile] * 2
    assert [n.enter.constexprs["SWAP_AB"] for n in launches] == [swap, swap]
    computes = [p for p in g.parts if isinstance(p, Compute)]
    fp8 = f"G={G},KB={tile[2]},SWAP={swap},DOT={dot}"
    linear = "FlashInferFp8Linear_v1" if M < 32 else "DeepGemmFp8Linear_v1"
    assert [c.spec.id for c in computes] == [
        gate, f"{linear}{{K={H},N={2 * SI}}}", f"SiluMul_v2{{I={SI}}}", f"{linear}{{K={SI},N={H}}}",
        bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=RSF, KERNEL="block").id,
        f"Fp8GroupQuantUe8m0_v1{{K={H},G={G}}}", f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},{fp8}}}", f"SiluMul_v2{{I={I}}}",
        f"Fp8GroupQuantUe8m0_v1{{K={I},G={G}}}", f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},{fp8}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    assert [c.rows for c in computes] == [M] * 6 + [P, P, P, P, M]
    assert all(_fits(c) for c in computes)
    gc, m1, silu, m2, router, qr, up, rs, rq, down, msum = computes
    hidden, (shared_out, routed_out) = scope.enter.args[0], scope.output_views()
    assert gc.operands[0].view == hidden and gc.roots == (None, WEIGHTS["gate"])
    assert router.operands[0].view == gc.outputs[0].view and router.roots == (None, WEIGHTS["bias"])
    out = [next(o for o in m.outputs if o is not None).view for m in (m1, m2)]
    assert (m1.operands[0].view.alloc, m1.operands[0].view.shape) == (hidden.alloc, hidden.shape) and silu.operands[0].view.alloc == out[0].alloc
    assert m2.operands[0].view == silu.outputs[0].view and out[1].alloc == shared_out.alloc and msum.outputs[0].view.alloc == routed_out.alloc
    assert m1.roots[1:] == (WEIGHTS["gate_up"], WEIGHTS["gate_up"] + "_scale_inv") and m2.roots[1:] == (WEIGHTS["down"], WEIGHTS["down"] + "_scale_inv")
    assert qr.operands[0].view == hidden and [type(o) for o in up.operands] == [Gather, Gather, PerRow, Shared, Shared]
    assert up.operands[0].indices == tuple(p // TOPK for p in range(P)) and up.operands[2].view.alloc == router.outputs[1].view.alloc
    assert rq.operands[0].view == rs.outputs[0].view and (down.operands[0].view, down.operands[1].view) == (rq.outputs[0].view, rq.outputs[1].view)
    assert down.operands[3].view.alloc == router.outputs[0].view.alloc
    assert up.roots[3:] == (WEIGHTS["w13"], WEIGHTS["w13"] + "_scale_inv") and down.roots[4:] == (WEIGHTS["w2"], WEIGHTS["w2"] + "_scale_inv")
    assert sum(isinstance(p, Transparent) and p.effect == "noop" for p in g.parts) == int(aligned)
    add = res[1]
    assert isinstance(add, Compute) and add.spec.id == f"ResidualAdd_v1{{N={H}}}" and add.rows == M
    assert [o.view.alloc for o in add.operands] == [shared_out.alloc, routed_out.alloc]
