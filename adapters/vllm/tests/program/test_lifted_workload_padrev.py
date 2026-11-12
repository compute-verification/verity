"""Adversarial tests for the MIXED BATCH Program side (`lifted_workload`, `lift_component`), from the attack list (A1 shared batch
dims, A5 post-hoc deletion, R1 one namespace) and T-1/T-4.  Padded baseline (experimental).  Reuses `test_lifted_mixed`'s TINY fixtures
and helpers.  The `component` fixture and the workload builders are `test_lifted_mixed`'s; the properties and the expected values are
this module's own."""

from __future__ import annotations

import pytest

from verity.circuits import Program
from verity_vllm.program import workload as W
from verity.circuits.codec import encode_program
from verity.circuits.defs import REGISTRY
from verity.definitions.registry import b1, lifted as L
from verity_vllm.program.lifting import lifted_compose
from tests.program import test_lifted_mixed as PM
from tests.program.test_lifted_mixed import BOUNDARY, CAP, EOS, LP, TINY, _inputs, _inputs_for, _prompt_b, _request_outs, _requests, _run
from verity.circuits.types import Array, Value

I32 = Value(32)


@pytest.fixture(scope="module")
def component():
    return PM.component.__wrapped__()


def _weights_type():
    from verity.definitions.registry.b1 import weights_type  # the TINY weights Record
    return weights_type(TINY)


# ---------------------------------------------------------------------------------------------------------
# R1 — one value namespace at the WORKLOAD root of the lifted Program (T-1 (i) on Lifted[Workload])
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 1])
def test_r1_the_lifted_workload_root_is_one_partitioned_namespace_and_every_guard_gate_sits_inside_its_own_request(component, lag):
    """Every gate of the lifted workload root has exactly one address: the Input intervals and the request intervals partition
    [0, gates) (no gap, no overlap, no empty interval); the shared lifted weights are ONE interval every request maps onto; and each
    request's guard gates (the lifting's only additions) lie inside THAT request's interval — a guard gate outside its request, or shared
    between two, would be a second identity / a cross-request edge."""
    wl = lifted_compose.lifted_workload(_requests(component, (0, 2)), EOS, lag=lag, boundary_prefix=BOUNDARY)
    am = wl.address_map()
    intervals = sorted([tuple(i["gates"]) for i in am["inputs"]] + [tuple(r["gates"]) for r in am["requests"]])
    assert intervals[0][0] == 0 and intervals[-1][1] == am["gates"] == wl.program.gates
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert a1 == b0, f"gap or overlap between {(a0, a1)} and {(b0, b1)}"
    assert all(a < b for a, b in intervals)
    sh = am["shared"]
    assert sh["group"] == "weights"
    for r in am["requests"]:
        w = [i["gates"] for i in r["inputs"] if i["global"].startswith("shared.")]
        assert w and all(sh["gates"][0] <= g0 < g1 <= sh["gates"][1] for g0, g1 in w)   # the same shared interval, never a copy
    # guard gates: inside the owning request's interval, and the two requests' guard gate sets are disjoint
    seen: set[int] = set()
    for r, q in zip(wl.requests, am["requests"]):
        c = r.component
        base = q["gates"][0]           # the call node's gate base: body-local offsets are global from here (address_map rule "instance")
        body = c.fn.body
        gates = set()
        for k in c.fn.guard_nodes:
            for g in range(body.offsets[k], body.offsets[k + 1]):
                gates.add(base + g)
        assert len(gates) == L.continuation_gate_count(CAP, len(EOS), lag), (r.request_id, len(gates))  # guard + Continuation_v1 controller + enable/one
        assert all(q["gates"][0] <= g < q["gates"][1] for g in gates), r.request_id
        assert not (gates & seen)
        seen |= gates
    pop = wl.population()
    assert pop["sum_equals_total"] and pop["components_partition_non_input_gates"] and pop["other_root_nodes"] == 0


# ---------------------------------------------------------------------------------------------------------
# A1 / A5 — A stops, B continues: B's words are the SAME whether or not A stops (the Program does not see the transcript)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_a1_a5_b_is_word_for_word_invariant_to_whether_a_stops_and_the_program_object_is_the_same(component):
    """`test_lifted_mixed`'s T-M2 compares B lifted vs B unlifted.  This runs the SAME lifted workload twice — A with the fixture prompt
    (stops at a_A = 4) and A with B's never-stopping prompt — and require every word of B's call range AND of the shared weights range to
    be identical.  Any ⊥ that crossed a batch dimension (A1) or any state B reads that depends on A's activity would show as a differing
    word.  The Program is one object: no structure of it depends on what will be served (A5)."""
    wl = lifted_compose.lifted_workload(_requests(component), EOS, lag=0, boundary_prefix=BOUNDARY)
    wl_u = W.WorkloadProgram(_requests(component), lag=0)  # lag is a required static of Workload_v1; wl_u is used for fn/inputs
    pa, pb = list(_inputs()[0]), _prompt_b()
    t_stop, o_stop = _run(wl.program, _inputs_for(wl_u.fn, {"A": pa, "B": pb}, True))
    t_go, o_go = _run(wl.program, _inputs_for(wl_u.fn, {"A": pb, "B": pb}, True))
    per_stop, per_go = _request_outs(wl, o_stop), _request_outs(wl, o_go)
    assert L.activity_prefix(per_stop["A"], EOS, CAP) == 4 and L.activity_prefix(per_go["A"], EOS, CAP) == CAP   # A stops / A runs to cap
    assert per_stop["B"] == per_go["B"] and all(L.is_active(32, x) for x in per_stop["B"])
    am = wl.address_map()
    qB = next(q for q in am["requests"] if q["request_id"] == "B")
    assert [t_stop[g] for g in range(*qB["gates"])] == [t_go[g] for g in range(*qB["gates"])]   # B's WHOLE call range, guards included
    assert [t_stop[g] for g in range(*am["shared"]["gates"])] == [t_go[g] for g in range(*am["shared"]["gates"])]
    # and A's range differs exactly from its stop on: the ⊥ words are A's, from a_A, and nowhere else
    qA = next(q for q in am["requests"] if q["request_id"] == "A")
    diff = [g for g in range(*qA["gates"]) if t_stop[g] != t_go[g]]
    assert diff and all(qA["gates"][0] <= g < qA["gates"][1] for g in diff)
    n_bot_stop = sum(1 for g in range(*qA["gates"]) if L.is_bot(32, t_stop[g]) or L.is_bot(16, t_stop[g]))
    n_bot_go = sum(1 for g in range(*qA["gates"]) if L.is_bot(32, t_go[g]) or L.is_bot(16, t_go[g]))
    assert n_bot_go == 1 and n_bot_stop > 100   # the never-stopping A holds only its bot32 constant


# ---------------------------------------------------------------------------------------------------------
# T-4 — a MIXED-CAP population: GP-01 composes it
# ---------------------------------------------------------------------------------------------------------

CAP_B = 4   # B declared with a smaller cap than A (GP-01's standard B2 fixture is unequal LP/T: R0 cap 3, R1 cap 2)


@pytest.fixture(scope="module")
def component_small():
    desc = encode_program(Program(b1.serve_v2(TINY, LP, CAP_B - 1)))
    return W.request_component(desc, REGISTRY, [("prompt", repr(Array(LP, I32))), ("weights", repr(_weights_type()))])


def _mixed_requests(component, component_small, arrivals=(0, 0)):
    return [W.WorkloadRequest("A", 0, LP, CAP, arrivals[0], component), W.WorkloadRequest("B", 1, LP, CAP_B, arrivals[1], component_small)]


def test_t4_a_mixed_cap_population_composes_with_every_request_programmed_for_its_own_cap(component, component_small):
    wl = lifted_compose.lifted_workload(_mixed_requests(component, component_small), EOS, lag=0, boundary_prefix=BOUNDARY)
    ch = wl.chronology()
    assert [r["programmed_steps"]["count"] for r in ch["requests"]] == [CAP, CAP_B] and ch["programmed_horizon_steps"] == CAP
    assert [len(e["members"]) for e in ch["engine_steps"]] == [2] * CAP_B + [1] * (CAP - CAP_B)
    for r in wl.requests:
        assert len(r.component.step_nodes) == r.cap and r.component.body_gates == W.WorkloadProgram(_mixed_requests(component, component_small), lag=0).requests[r.index].component.body_gates + L.continuation_gate_count(r.cap, len(EOS), 0)
