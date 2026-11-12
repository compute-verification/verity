"""The Qwen3-MoE fold scope on RECORDED events, backing test_gen_ov_moe_qwen3.py's synthetic scopes.

`fixtures/qwen3_30b_a3b_tp2_moe_l24.rank0.jsonl.gz` is one decoder layer's `vllm.moe_forward` scope cut by `tests/engine/moe_scope_cut.py`
from rank 0 of the Qwen3-30B-A3B BF16 TP2 B1 i256 o32 capture (RTX PRO 6000, vLLM d9105ea80, batch-invariant eager; its `moe_scope_cut`
Note names the source log and its sha256): the RunHeader, layer 24's weight Roots (`mlp.gate`, `w13`, `w2`), and for the prefill step
(256 rows) and the first decode step (1 row) every event of the scope plus the block's all-reduce, which this serve issues right after
the scope.  Folded with `resolve_log` and the rank profile, as `verity-vllm resolve-log` folds a whole log:
  * the router's gate GEMM (`mlp.gate`'s `matmul_kernel_persistent`) launches inside the scope and resolves, ahead of the router, as the
    profile's GEMM for the record's target (cc 12.0, `blackwell_consumer`: `Gemm_v2{K=H,N=E,DOT=HopperBF16WgmmaDot16_v1}`; the
    synthetic header records no GPU, hence its `Gemm_v1`), writing the logits the router reads;
  * the activation binds as `SiluMul_v2{I}`, the version the MoE constructions declare;
  * the block's TP2 all-reduce resolves as `AllReduce2_v1{N=H}` over the scope's routed sum;
  * the recorded scope edited so the gate is not the router's logits launch (a second gate launch, the gate writing another buffer or
    following the router, another gate tiling) is refused by name.
"""
from __future__ import annotations

import dataclasses
import gzip
import importlib
from pathlib import Path

import pytest

from tests.engine.moe_scope_cut import cut, write
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Note, StepEnd, StepStart, TritonLaunch, ViewDesc, views_in
from verity_vllm.observe.fold.resolve_log import resolve_log
from verity_vllm.observe.fold.resolver import Compute, Group, Transparent
from verity_vllm.observe.log import LogWriter, read_log
from verity_vllm.program.frontend.target_profile import TargetProfile

Q3 = importlib.import_module("verity_vllm.observe.fold.profiles.derived_QWEN3_30B_A3B_tp2")
C = Q3.C
H, I, E, TOPK = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"]

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_30b_a3b_tp2_moe_l24.rank0.jsonl.gz"
SOURCE_SHA256 = "4f6fb584e7236ffd6b349a8aebe4a40fa245a523037cf0b9b6ff2a12ca1bebc6"
LAYER = 24
MODULE = f"model.layers.{LAYER}.mlp.experts"
GATE = "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent"
#: engine step -> (rows M, whether the scope aligns: the naive block assignment skips moe_align_block_size for M <= 2)
STEPS = {0: (256, True), 1: (1, False)}

DOT = "HopperBF16WgmmaDot16_v1"
EXPECTED_SPECS = [f"Gemm_v2{{K={H},N={E},DOT={DOT}}}", f"MoeRouterTopKOrderedNorm_v1{{E={E},TOPK={TOPK},VPT=8}}",
                  f"MoeExpertGemm_v2{{E={E},K={H},N={2 * I},DOT={DOT}}}", f"SiluMul_v2{{I={I}}}",
                  f"MoeExpertGemmW_v2{{E={E},K={I},N={H},DOT={DOT}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]


def _fold(path: Path):
    header, results, roots, notes = resolve_log(path, Q3.PROFILE)
    return header, {sr.step: sr for sr in results}, roots, notes


def test_fixture_is_the_recorded_layer_scope_of_the_tp2_capture():
    header, steps, roots, notes = _fold(FIXTURE)
    assert Q3.check_header(header) == []
    assert header.profile["vllm"]["commit"] == "d9105ea80" and header.profile["parallelism"]["tensor_parallel_size"] == 2
    assert header.profile["gpu"]["compute_capability"] == "12.0" and header.profile["engine_flags"]["VLLM_BATCH_INVARIANT"] == "1"
    (note,) = notes
    assert note.text == "moe_scope_cut" and note.data["sha256"] == SOURCE_SHA256 and note.data["layer"] == LAYER and note.data["steps"] == [0, 1]
    assert sorted(r.name for rs in roots.values() for r in rs) == [f"model.layers.{LAYER}.mlp.experts.routed_experts.w13_weight",
                                                                   f"model.layers.{LAYER}.mlp.experts.routed_experts.w2_weight",
                                                                   f"model.layers.{LAYER}.mlp.gate.weight"]
    assert {s: (sr.phase_hint, sr.rows) for s, sr in steps.items()} == {0: ("prefill", 256), 1: ("decode", 1)}


def test_the_fixture_is_a_fixed_point_of_the_cutter(tmp_path):
    """Cutting the fixture again (same layer, steps and source) gives the fixture's events: what is checked in is the cutter's output."""
    events = list(read_log(FIXTURE))
    note = next(e for e in events if isinstance(e, Note))
    source = {k: note.data[k] for k in ("origin", "sha256")}
    write(cut(events, LAYER, set(STEPS), source), tmp_path / "recut.jsonl.gz")
    assert gzip.decompress((tmp_path / "recut.jsonl.gz").read_bytes()) == gzip.decompress(FIXTURE.read_bytes())


@pytest.mark.parametrize("step", sorted(STEPS))
def test_recorded_scope_resolves_the_gate_gemm_ahead_of_the_router_and_the_block_all_reduce(step):
    _, steps, roots, _ = _fold(FIXTURE)
    sr = steps[step]
    M, aligned = STEPS[step]
    assert sr.error is None and not sr.unsupported, (sr.error, [u.reason for u in sr.unsupported])
    group, ar = sr.resolutions
    assert isinstance(group, Group) and isinstance(ar, Compute)
    computes = [p for p in group.parts if isinstance(p, Compute)]
    assert [c.spec.id for c in computes] == EXPECTED_SPECS
    assert [c.rows for c in computes] == [M, M, M * TOPK, M * TOPK, M * TOPK, M]
    gemm, router, up, silu, down, msum = computes
    assert sr.gemm_target["target"] == "blackwell_consumer" and sr.gemm_target["compute_capability"] == [12, 0]

    # the gate GEMM: the profile's own persistent GEMM over the scope's hidden states and the layer's gate weight, writing the logits
    gate_w = next(r for rs in roots.values() for r in rs if r.name == f"model.layers.{LAYER}.mlp.gate.weight")
    assert gemm.kernels == ("matmul_kernel_persistent",) and gemm.pattern.startswith("gemm.matmul_kernel_persistent")
    assert gemm.hint == f"model.layers.{LAYER}.mlp.gate/{gate_w.name}"
    assert gemm.operands[0].view == up.operands[0].view and gemm.operands[0].view.shape == (M, H)
    assert gemm.operands[1].view == gate_w.view and gemm.operands[1].view.shape == (E, H)
    assert gemm.outputs[0].view == router.operands[0].view and gemm.outputs[0].view.shape == (M, E)
    assert group.parts.index(gemm) < group.parts.index(router)

    # the activation: the version TargetProfile.MOE_CONSTRUCTIONS declares for the profile's construction
    declared = [d for d in TargetProfile.MOE_CONSTRUCTIONS[TargetProfile.DEFAULT_MOE_CONSTRUCTION]["definitions"] if d.startswith("SiluMul_")]
    assert [d.split("{")[0] for d in declared] == [silu.spec.id.split("{")[0]] == ["SiluMul_v2"]
    assert silu.operands[0].view == up.outputs[0].view and down.operands[0].view == silu.outputs[0].view
    aligns = [p for p in group.parts if isinstance(p, Transparent) and any("moe_align_block_size" in k for k in p.kernels)]
    assert len(aligns) == int(aligned)

    # the block's TP2 all-reduce: one AllReduce2 over the rank's routed sum, issued after the scope (a sibling, not a scope child)
    assert ar.spec.id == f"AllReduce2_v1{{N={H}}}" and ar.rows == M and ar.pattern == "tp.collective"
    (own,) = ar.operands
    assert own.kind == "all_reduce" and own.site == MODULE and own.own == msum.outputs[0].view
    assert ar.outputs[0].view.shape == (M, H) and ar.outputs[0].view.alloc != own.own.alloc


# --- the recorded scope, edited so the gate GEMM is not the router's logits launch -----------------------------------------------------

def _gate(events) -> TritonLaunch:
    (g,) = [e for e in events if isinstance(e, TritonLaunch) and e.fn == GATE]
    return g


def _with_launch(g: TritonLaunch, **set_args) -> TritonLaunch:
    return dataclasses.replace(g, args={**g.args, **set_args}, constexprs={**g.constexprs, **{k: v for k, v in set_args.items() if k in g.constexprs}})


def _edit(step_events: list, kind: str, fresh_id: int, fresh_alloc: int) -> list:
    g = _gate(step_events)
    k = step_events.index(g)
    if kind == "twice":
        return step_events[:k + 1] + [dataclasses.replace(g, id=fresh_id)] + step_events[k + 1:]
    if kind == "elsewhere":
        c = g.args["c_ptr"]
        return step_events[:k] + [_with_launch(g, c_ptr=ViewDesc(fresh_alloc, c.offset, c.shape, c.strides, c.dtype, c.device))] + step_events[k + 1:]
    if kind == "after-router":
        rest = step_events[:k] + step_events[k + 1:]
        router = next(e for e in rest if getattr(e, "op", "") == "_moe_C.topk_softmax.default")
        j = next(i for i, e in enumerate(rest) if isinstance(e, DispatchExit) and e.id == router.id)
        return rest[:j + 1] + [dataclasses.replace(g, id=fresh_id)] + rest[j + 1:]
    if kind == "tiling":
        return step_events[:k] + [_with_launch(g, BLOCK_SIZE_K=g.constexprs["BLOCK_SIZE_K"] // 2)] + step_events[k + 1:]
    raise AssertionError(kind)


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("kind,needle", [
    ("twice", "more than one matmul_kernel_persistent launch"),
    ("elsewhere", "does not write the router's gating logits"),
    ("after-router", "follows topk_softmax: it is not the router's gate GEMM"),
    ("tiling", "refused the router's gate GEMM"),          # the gate's tiling is the profile's pin for the record's target
])
def test_recorded_scope_refuses_a_gate_that_is_not_the_router_s_logits_launch(tmp_path, step, kind, needle):
    events = list(read_log(FIXTURE))
    fresh_id = 1 + max(e.id for e in events if isinstance(e, (DispatchEnter, DispatchExit, TritonLaunch)))
    fresh_alloc = 1 + max(v.alloc for e in events for f in dataclasses.fields(e) for v in views_in(getattr(e, f.name)))
    lo = next(i for i, e in enumerate(events) if isinstance(e, StepStart) and e.step == step)
    hi = next(i for i, e in enumerate(events) if i > lo and isinstance(e, StepEnd))
    edited = events[:lo] + _edit(events[lo:hi + 1], kind, fresh_id, fresh_alloc) + events[hi + 1:]
    with LogWriter(tmp_path / "edited.jsonl") as w:
        for e in edited:
            w.write(e)
    _, steps, _, _ = _fold(tmp_path / "edited.jsonl")
    sr = steps[step]
    assert sr.error is None and not [r for r in sr.resolutions if isinstance(r, Group)], sr.resolutions
    (bad,) = sr.unsupported
    assert needle in bad.reason and bad.op == "vllm.moe_forward.default", bad.reason
    if kind == "tiling":
        assert bad.detail["child_detail"]["mismatches"] == {"BLOCK_SIZE_K": {"observed": 32, "pinned": 64}}, bad.detail
    assert all(not o.unsupported and isinstance(o.resolutions[0], Group) for s, o in steps.items() if s != step)   # the untouched step folds
