"""Without batch invariance a request Program is built under its launch BATCH: every token-count-dependent kernel choice reads the
whole engine step its rows run in (`targets.parse_launch_context`'s batch form, `pipeline.launch_context`).

The batch is declared per request step as [rows, start] per scheduled sequence; a node traced inside a request step carries its
`launch_region` tag, which the bindings read (`launch_tokens_of`, `launch_keys_of`) and trace-time dispatch reads through
`traced_launch_tokens`.  The covered all-reduce routes take a call of the request's own rows only where an element's reduction does
not depend on its offset in the launch (`tp_allreduce.offset_free`).  A row with batch dispatch builds no request Program without
its batch."""
from __future__ import annotations

import json
import os
import sys
import types

import pytest
import torch
import torch.fx.traceback as fxt

from verity_vllm.pipeline import launch_context as LC
from verity_vllm.pipeline.build import applicability_conditions, run
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.outcomes import Refuse
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import observations as OB
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.registry import tp_allreduce as TA

#: request steps 0 and 1 of the 65-row prompt of GLM-5.3's B8 row: its prefill beside two decoders, then six sequences
BATCH = [[[1, 17], [1, 231], [65, 0]], [[1, 18], [1, 65], [187, 0]]]
GLM_B8 = os.path.join(os.path.dirname(__file__), "..", "..", "workloads",
                      "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager.json")
DSV4_B8 = os.path.join(os.path.dirname(__file__), "..", "..", "workloads",
                       "dsv4-pro__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager.json")


@pytest.fixture
def declared():
    saved = dict(VB._OBSERVED)
    VB.observe_launch_context(TG.parse_launch_context({"batch": BATCH}))
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _wl(reqs, *, batch_invariant):
    return {"requests": [{"prompt_len": lp, "max_tokens": c, "arrive_step": a} for lp, c, a in reqs],
            "engine": {"target": {"batch_invariant": batch_invariant}}}


def _node(tag=None):
    n = torch.fx.Graph().placeholder("x")
    if tag is not None:
        n.meta["custom"] = tag
    return n


def test_the_batch_form_parses_and_carries_max_seqlen_q():
    ctx = TG.parse_launch_context(json.dumps({"batch": BATCH}))
    assert ctx["batch_by_step"] == {0: [(1, 17), (1, 231), (65, 0)], 1: [(1, 18), (1, 65), (187, 0)]}
    assert ctx["max_seqlen_q_by_step"] == {0: 65, 1: 187} and ctx["conflict"] is False
    for bad in ([], [[]], [[[0, 3]]], [[[1, -1]]]):
        with pytest.raises(ValueError):
            TG.parse_launch_context({"batch": bad})


def test_launch_tokens_and_keys_read_the_step_and_refuse_another_requests_batch():
    ctx = TG.parse_launch_context({"batch": BATCH})
    assert TG.launch_tokens(65, 0, ctx) == 67 and TG.launch_tokens(1, 1, ctx) == 189
    assert TG.launch_tokens(1, 0, ctx, sampled=True) == 3 and TG.launch_tokens(1, 1, ctx, sampled=True) == 3
    assert TG.launch_keys(0, ctx) == [18, 232, *range(1, 66)]
    assert TG.launch_tokens(65, 0, None) == 65 and TG.launch_keys(0, None) is None
    with pytest.raises(TG.LaunchContextInvalid, match="runs step 2"):
        TG.launch_tokens(1, 2, ctx)
    with pytest.raises(TG.LaunchContextInvalid, match="another request's context"):
        TG.launch_tokens(100, 0, ctx)
    with pytest.raises(TG.LaunchContextConflict):
        TG.launch_tokens(1, 0, TG.parse_launch_context("conflict"))


def test_batch_dispatch_is_a_multi_request_row_without_batch_invariance():
    reqs = [(64, 3, 0), (300, 2, 1)]
    assert LC.batch_dispatch(_wl(reqs, batch_invariant=False))
    assert not LC.batch_dispatch(_wl(reqs, batch_invariant=True))
    assert not LC.batch_dispatch(_wl(reqs[:1], batch_invariant=False))
    ctx = LC.shape_contexts(_wl(reqs, batch_invariant=False))
    assert ctx[(64, 2)] == [[[64, 0]], [[1, 64], [300, 0]], [[1, 65], [1, 300]]]
    assert ctx[(300, 1)] == [[[1, 64], [300, 0]], [[1, 65], [1, 300]]]
    assert LC.spec_for(_wl(reqs, batch_invariant=False), 300, 1) == '{"batch":[[[1,64],[300,0]],[[1,65],[1,300]]]}'
    assert LC.spec_for(_wl(reqs, batch_invariant=True), 300, 1) == "[300,1]"
    assert LC.shape_contexts(_wl(reqs[:1], batch_invariant=False)) == {(64, 2): [64, 1, 1]}


def test_the_batch_lists_a_step_in_admission_order():
    """The order the scheduler keeps its running requests in and vLLM's runner its decodes (`vllm_dsv4._launch_decodes` reads a
    decode's index there): by arrival, ties in the workload's order, whatever order the workload lists them in."""
    reqs = LC._requests(_wl([(20, 4, 2), (10, 4, 0), (30, 4, 2)], batch_invariant=False))
    assert LC.request_launch_batch(reqs, reqs[0])[0] == [[1, 11], [20, 0], [30, 0]]


def test_glm53_b8_every_request_sits_in_its_declared_batch():
    w = json.load(open(GLM_B8))
    assert LC.batch_dispatch(w)
    reqs = LC._requests(w)
    for r in reqs:
        batch = LC.shape_contexts(w)[(r["prompt_len"], r["max_tokens"] - 1)]
        assert len(batch) == r["max_tokens"]
        for s, seqs in enumerate(batch):
            own = [r["prompt_len"], 0] if s == 0 else [1, r["prompt_len"] + s - 1]
            assert own in seqs
    steps = range(max(r["arrive_step"] + r["max_tokens"] for r in reqs))
    assert sorted({sum(LC.rows_at(q, e) for q in reqs) for e in steps}) == [1, 2, 3, 4, 6, 11, 12, 67, 189, 232, 393]


def test_dsv4_b8_every_request_sits_in_its_declared_batch():
    w = json.load(open(DSV4_B8))
    assert LC.batch_dispatch(w)
    reqs = LC._requests(w)
    for r in reqs:
        batch = LC.shape_contexts(w)[(r["prompt_len"], r["max_tokens"] - 1)]
        assert len(batch) == r["max_tokens"]
        for s, seqs in enumerate(batch):
            own = [r["prompt_len"], 0] if s == 0 else [1, r["prompt_len"] + s - 1]
            assert own in seqs
    steps = range(max(r["arrive_step"] + r["max_tokens"] for r in reqs))
    assert sorted({sum(LC.rows_at(q, e) for q in reqs) for e in steps}) == [1, 2, 3, 4, 5, 6, 7, 14, 55, 129, 140, 203, 290]


def test_launch_region_tags_the_traced_nodes_and_the_export_keeps_them(declared):
    class Step(torch.nn.Module):
        def forward(self, x, w):
            with OB.launch_region(1, "tokens"):
                assert OB.traced_launch_tokens(x.shape[0]) == 189
                h = torch.mm(x, w)
            with OB.launch_region(1, "sampled"):
                assert OB.traced_launch_tokens(1) == 3
                y = torch.mm(h[-1:], w)
            return torch.cat([h, y])

    assert OB.traced_launch_tokens(1) == 1
    with fxt.preserve_node_meta():
        ep = torch.export.export(Step(), (torch.randn(1, 4), torch.randn(4, 4)), strict=False)
    mms = [n for n in ep.graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.mm.default]
    assert [OB.launch_tokens_of(n, 1) for n in mms] == [189, 3]
    (cat,) = [n for n in ep.graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.cat.default]
    with pytest.raises(Refuse, match="outside any request step"):
        OB.launch_tokens_of(cat, 2)


def test_bindings_read_the_tagged_launch_and_refuse_what_it_does_not_hold(declared):
    prefill, decode = _node({"launch_step": 0, "launch_rows": "tokens"}), _node({"launch_step": 1, "launch_rows": "tokens"})
    assert OB.launch_tokens_of(prefill, 65) == 67 and OB.launch_tokens_of(decode, 1) == 189
    assert OB.launch_keys_of(decode, [66]) == [19, 66, *range(1, 188)]
    with pytest.raises(Refuse, match="do not hold"):
        OB.launch_keys_of(decode, [188])
    with pytest.raises(Refuse, match="request step 5"):
        OB.launch_tokens_of(_node({"launch_step": 5, "launch_rows": "tokens"}), 1)
    VB.observe_launch_context(None)
    assert OB.launch_tokens_of(_node(), 7) == 7 and OB.launch_keys_of(_node(), [3, 4]) == [3, 4]


def test_moe_config_and_gate_tier_are_read_at_the_launchs_tokens(declared, monkeypatch):
    from verity_vllm.engine import hooks
    from verity_vllm.program.frontend.export_compat import _moe_config_at_launch, moe_gate_leaf
    seen = []

    def config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
        seen.append(M)
        return {"BLOCK_SIZE_M": 16 if M <= 16 else 64}

    pkg = types.ModuleType("fake_fused_moe")
    pkg.__package__ = "fake_fused_moe"
    pkg.try_get_optimal_moe_config = config
    experts = types.ModuleType("fake_fused_moe.experts")
    experts.try_get_optimal_moe_config = config
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    monkeypatch.setitem(sys.modules, experts.__name__, experts)
    with hooks.Hooks() as h:
        _moe_config_at_launch(pkg, h)
        with OB.launch_region(1, "tokens"):
            pkg.try_get_optimal_moe_config((1,), (1,), 8, None, 1)
            experts.try_get_optimal_moe_config((1,), (1,), 8, None, 1)
    assert seen == [189, 189] and pkg.try_get_optimal_moe_config is config and experts.try_get_optimal_moe_config is config

    site = types.ModuleType("ll_bf16_site")
    site.ll_bf16_gemm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("the CUDA kernel ran"))

    class Gate(torch.nn.Module):
        def __init__(self, cublas: bool) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, 8, dtype=torch.bfloat16, device="meta"), requires_grad=False)
            self.allow_ll_bf16_gemm, self.allow_cublas_router_gemm = True, cublas

        def forward(self, x):
            with OB.launch_region(1, "tokens"):
                return site.ll_bf16_gemm(x, self.weight, torch.float32)

    x = torch.empty(1, 8, dtype=torch.bfloat16, device="meta")
    gate = Gate(True)
    with moe_gate_leaf(gate, site=site):
        ep = torch.export.export(gate, (x,), strict=False)
    calls = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert not any("ll_bf16" in str(n.target) for n in calls)
    (mm,) = [n for n in calls if n.target == torch.ops.aten.mm.dtype]
    assert mm.meta["val"].dtype == torch.float32 and tuple(mm.meta["val"].shape) == (1, 4)
    with moe_gate_leaf(Gate(False), site=site), pytest.raises(NotImplementedError, match="past 16 launch rows"):
        Gate(False)(x)


def test_route_binding_states_a_call_of_own_rows_only_for_an_offset_free_route():
    mnnvl, multimem, custom = (TA.covered("defaults", 6144, "fi_ar_comm", 3), TA.covered("defaults", 6144, "symm_mem_comm", 67),
                               TA.covered("switch", 6144, "ca_comm", 67))
    assert TA.offset_free(mnnvl, 8) and TA.offset_free(multimem, 8) and not TA.offset_free(custom, 8)
    assert TA.route_binding(mnnvl, 8, 3, 6144, 1)[1] == {"WORLD": 8, "N": 6144, "SHOT": TA.mnnvl_shot(3, 6144, 8)}
    assert TA.route_binding(multimem, 8, 67, 6144, 1)[1]["SIZE"] == 6144
    assert TA.route_binding(custom, 8, 67, 6144)[1]["SIZE"] == 67 * 6144
    with pytest.raises(ValueError, match="by its offset"):
        TA.route_binding(custom, 8, 67, 6144, 1)


def test_a_batch_dispatch_request_program_needs_its_batch(tmp_path):
    for launch in (None, TG.parse_launch_context([65, 187]), TG.parse_launch_context("conflict")):
        res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=65, T=1, launch=launch, batch_dispatch=True)
        assert res["ok"] is False and res["unsupported"]["reason"] == "launch_context", res.get("unsupported")
        assert "verity-vllm launch-context <workload> 65 1" in res["unsupported"]["hint"]


def test_the_batch_is_an_applicability_condition_only_when_declared():
    prof = TargetProfile.default()
    listed = applicability_conditions(prof, {}, {}, {}, None, launch=TG.parse_launch_context([65, 187]))
    batched = applicability_conditions(prof, {}, {}, {}, None, launch=TG.parse_launch_context({"batch": BATCH}))
    assert "batch_by_step" not in listed["launch_context"]
    assert batched["launch_context"]["batch_by_step"] == {"0": [[1, 17], [1, 231], [65, 0]], "1": [[1, 18], [1, 65], [187, 0]]}
    assert batched["launch_context"]["max_seqlen_q_by_step"] == listed["launch_context"]["max_seqlen_q_by_step"]
