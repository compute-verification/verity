"""padrev — adversarial tests for pad's MIXED BATCH Program side (`lifted_workload`, `lift_component`, `live_count_hook`; lane/r16-pad
c02a25c93), from LIFTING-REVIEW §3 (A1 shared batch dims, A5 post-hoc deletion, A9 the S relation, R1 one namespace) and §1 T-1/T-4.
Padded baseline (experimental).  Reuses pad's TINY fixtures and helpers (`test_lifted_mixed`) and padrev's independent live reference
(`test_lifted_tiny_padrev.padrev_live_ref`, written before pad's LAG post).  The `component` fixture and the workload builders are pad's;
the properties and the expected values are padrev's."""

from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity_vllm.program import workload as W
from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import REGISTRY, bind
from verity_vllm.program.registry import b1, lifted as L
from verity_vllm.program.lifting import lifted_compose
from tests.program import test_lifted_mixed as PM
from tests.program.test_lifted_mixed import BOUNDARY, CAP, EOS, LP, SMS, TINY, _inputs, _inputs_for, _prompt_b, _request_outs, _requests, _run
from tests.program.test_lifted_tiny_padrev import padrev_live_ref
from tests.program.test_serve3 import Y_RAW
from verity.ml.circuits.types import Array, Value

I32 = Value(32)


@pytest.fixture(scope="module")
def component():
    return PM.component.__wrapped__()


def _weights_type():
    from verity_vllm.program.registry.b1 import weights_type  # the TINY weights Record
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
        assert len(gates) == L.continuation_gate_count(CAP, len(EOS), lag), (r.request_id, len(gates))  # guard + Continuation_v1 controller (R17-32) + enable/one (v0.6)
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
    """pad's T-M2 compares B lifted vs B unlifted.  The reviewer's angle: run the SAME lifted workload twice — A with the fixture prompt
    (stops at a_A = 4) and A with B's never-stopping prompt — and require every word of B's call range AND of the shared weights range to
    be identical.  Any ⊥ that crossed a batch dimension (A1) or any state B reads that depends on A's activity would show as a differing
    word.  The Program is one object: no structure of it depends on what will be served (A5)."""
    wl = lifted_compose.lifted_workload(_requests(component), EOS, lag=0, boundary_prefix=BOUNDARY)
    wl_u = W.WorkloadProgram(_requests(component), lag=0)  # R17-1: lag is a required static of Workload_v1 (moe a96a2071b); wl_u is used for fn/inputs
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
# A9 / P9 — the S hook under LAG=1 with staggered arrivals, against padrev's independent live reference
# ---------------------------------------------------------------------------------------------------------


def _hook(component, arrivals, prompts, lag, caps=None):
    return PM._hook_run(component, arrivals, prompts, lag)


def test_a9_the_hook_under_lag_1_with_staggered_arrivals_equals_the_independent_reference_per_request(component):
    """pad tests LAG 1 at arrivals (0, 0) and LAG 0 at (0, 2).  The reviewer runs LAG 1 at (0, 2): per request the live vector must equal
    padrev_live_ref (event 0 live; step s live iff no EOS at an event <= s-1-lag), and the root count at engine step t must be the sum of
    the members' reference values — A contributes its placeholder step (live at a_A under lag 1), B is live throughout its window."""
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    arrivals = (0, 2)
    wl, prog, t, outs, n_live, S = _hook(component, arrivals, prompts, 1)
    per = _request_outs(wl, outs)
    ref = {rid: padrev_live_ref(per[rid], EOS, CAP, 1)[0] for rid in ("A", "B")}
    a_A = padrev_live_ref(per["A"], EOS, CAP, 1)[1]
    assert a_A == 4 and ref["A"] == [1] * 5 + [0] * (CAP - 5) and ref["B"] == [1] * CAP
    H = max(a + CAP for a in arrivals)
    want = [sum(ref[rid][tt - a] for rid, a in zip(("A", "B"), arrivals) if 0 <= tt - a < CAP) for tt in range(H)]
    assert want == [1, 1, 2, 2, 2] + [1] * (H - 5)
    assert [L.payload(L.N_LIVE_BITS, x) for x in n_live] == want and all(L.is_active(L.N_LIVE_BITS, x) for x in n_live)
    assert [L.payload(L.SPLITS_BITS, x) for x in S] == [32] * H       # 142 SMs: 1 or 2 live -> 32
    # the placeholder step's token is A's own y_4 (a value of the execution), Active, then ⊥ -- in the RAW committed event (the boundary
    # node's gate).  The DESIGNATED output `slot=5/out` is ⊥ (R17-32 Continuation_v1 s5 C: served length 4, the discarded lag sample is
    # executed but not served; R17-1 "served length != executed-prefix length").
    fA = next(r for r in wl.requests if r.request_id == "A").component.fn
    base = PM._wl_call_base(wl, prog, "A")
    # the boundary node's OUTPUT gate = the unlifted request Program's return gate of that event, at the same offset inside the mapped node
    # (PAD-03: the node ranges correspond 1:1)
    pu = L.as_program(component.fn)
    n_in = sum(1 for n in pu.root.body.nodes if getattr(n.fn, "family", None) == "Input")
    off_u = pu.root.body.offsets[n_in:]  # the lowered root = the Input nodes then the function's nodes: function node b starts at off_u[b]
    raw_A = [t[base + fA.body.offsets[fA.node_map[b]] + (gu - off_u[b])] for b, gu in zip(fA.boundaries, pu.circuit.root_outputs())]
    assert raw_A[4] == Y_RAW[4] and L.is_active(32, raw_A[4]) and all(L.is_bot(32, x) for x in raw_A[5:]) and raw_A[:4] == per["A"][:4]
    assert all(L.is_bot(32, x) for x in per["A"][4:]) and L.served_count(per["A"]) == 4 and L.served_count(raw_A) == 5


# ---------------------------------------------------------------------------------------------------------
# T-4 / F-pad-19 — a MIXED-CAP population: GP-01 composes it; the hook must not silently miscount it
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


# F-pad-19 CLOSED by pad 850393f65 (live_count_hook takes one cap PER REQUEST; a served-length out_r refused by name): strict xfail
# flipped to PASS 07:08Z -- on the fixed hook the single-cap call below is REFUSED by name (the accepted outcome), see the per-request test after it
def test_f_pad_19_the_hook_on_a_mixed_cap_population_takes_each_requests_own_cap_or_refuses_by_name(component, component_small):
    from verity.ml.circuits.defs import Builder, composite
    from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
    from verity.ml.circuits.types import Tuple
    reqs = _mixed_requests(component, component_small)
    wl = lifted_compose.lifted_workload(reqs, EOS, lag=0, boundary_prefix=BOUNDARY)
    wl_u = W.WorkloadProgram(reqs, lag=0)
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}

    @composite("PadrevMixedS", 1, ["WL", "EOS", "CAP", "ARR"],
               lambda S: (S.WL.signature()[0], Tuple(S.WL.signature()[1], Array(S.CAP, Value(L.N_LIVE_BITS + 1)))), conformance="protocol-defined")
    def PadrevMixedS(B, S, *args):
        w = B.call(S.WL, *args, name="workload")
        eos_cols = [B.call(b1.const(33, e), name=f"eos[{i}]") for i, e in enumerate(S.EOS)]
        eos_set = coll_concat(Array(len(S.EOS), Value(33)), eos_cols)
        bot = B.call(L.l_bot(32), name="bot32")
        n_live, _ = L.live_count_hook(B, [w[0], w[1]], list(S.ARR), S.CAP, eos_set, bot, 0)
        return tuple_of(w, array_of(n_live))

    # with cap = the LONGER request's: the hook reads B's out[s-1] for s up to CAP-1 although B has CAP_B words
    try:
        prog = Program(bind(PadrevMixedS, WL=wl.fn, EOS=EOS, CAP=CAP, ARR=(0, 0)))
        _, outs = _run(prog, _inputs_for(wl_u.fn, prompts, True))
        n_live = outs[CAP + CAP_B:]
    except ValueError as e:                                   # a NAMED refusal is acceptable
        assert "cap" in str(e).lower()
        return
    except IndexError:
        pytest.fail("unnamed IndexError: the hook indexed past the shorter request's out (F-pad-19)")
    # if it built, the count must be the population's activity: A alone after B's cap (B never stops within its 4 events)
    per = _request_outs(wl, outs[:CAP + CAP_B])
    a_A = L.activity_prefix(per["A"], EOS, CAP)
    want = [(1 if s < a_A else 0) + (1 if s < CAP_B else 0) for s in range(CAP)]
    assert [L.payload(L.N_LIVE_BITS, x) for x in n_live] == want


def _padrev_mixed_s2():
    from verity.ml.circuits.defs import composite
    from verity.ml.circuits.refs import array_of, coll_concat, tuple_of
    from verity.ml.circuits.types import Tuple

    @composite("PadrevMixedS2", 1, ["WL", "EOS", "CAPS", "ARR", "LAG", "H"],
               lambda S: (S.WL.signature()[0], Tuple(S.WL.signature()[1], Array(S.H, Value(L.N_LIVE_BITS + 1)), Array(S.H, Value(L.SPLITS_BITS + 1)))),
               conformance="protocol-defined")
    def PadrevMixedS2(B, S, *args):
        w = B.call(S.WL, *args, name="workload")
        eos_cols = [B.call(b1.const(33, e), name=f"eos[{i}]") for i, e in enumerate(S.EOS)]
        eos_set = coll_concat(Array(len(S.EOS), Value(33)), eos_cols)
        bot = B.call(L.l_bot(32), name="bot32")
        n_live, S_t = L.live_count_hook(B, [w[0], w[1]], list(S.ARR), list(S.CAPS), eos_set, bot, S.LAG, num_sms=SMS)
        return tuple_of(w, array_of(n_live), array_of(S_t))

    return PadrevMixedS2


PadrevMixedS2 = _padrev_mixed_s2()


@pytest.mark.parametrize("lag,arrivals", [(0, (0, 0)), (1, (0, 0)), (0, (0, 3)), (1, (2, 0))])
def test_f_pad_19_per_request_caps_the_hook_equals_the_independent_reference_on_a_mixed_cap_staggered_population(component, component_small, lag, arrivals):
    """After pad's 850393f65: `live_count_hook(cap=[cap_A, cap_B])` on A (cap 9, stops at a_A = 4) + B (cap 4, never stops), under both lags
    and staggered arrivals.  Per engine step the count must equal the sum of padrev_live_ref over the requests programmed for it (B a member
    of [a_B, a_B + 4) ONLY -- a row of B outside its own cap would be an invented member), S = SplitsFor of that count and ⊥ exactly at 0;
    and the same call with ONE cap for both is refused by name in both directions (9 for B's 4 words, 4 for A's 9)."""
    reqs = _mixed_requests(component, component_small, arrivals)
    wl = lifted_compose.lifted_workload(reqs, EOS, lag=lag, boundary_prefix=BOUNDARY)
    wl_u = W.WorkloadProgram(reqs, lag=lag)
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    caps = [CAP, CAP_B]
    H = max(a + c for a, c in zip(arrivals, caps))
    prog = Program(bind(PadrevMixedS2, WL=wl.fn, EOS=EOS, CAPS=tuple(caps), ARR=arrivals, LAG=lag, H=H))
    _, outs = _run(prog, _inputs_for(wl_u.fn, prompts, True))
    per = _request_outs(wl, outs[:CAP + CAP_B])
    n_live, S_t = outs[CAP + CAP_B:CAP + CAP_B + H], outs[CAP + CAP_B + H:]
    ref = {"A": padrev_live_ref(per["A"], EOS, CAP, lag)[0], "B": padrev_live_ref(per["B"], EOS, CAP_B, lag)[0]}
    assert padrev_live_ref(per["A"], EOS, CAP, lag)[1] == 4 and ref["B"] == [1] * CAP_B and sum(ref["A"]) == min(4 + lag, CAP)
    want = [sum(ref[rid][t - a] for rid, a, c in zip(("A", "B"), arrivals, caps) if 0 <= t - a < c) for t in range(H)]
    assert [L.payload(L.N_LIVE_BITS, x) for x in n_live] == want and all(L.is_active(L.N_LIVE_BITS, x) for x in n_live)
    assert max(want) == 2 and 0 in want   # both live somewhere; A's window outlives both its stop and B's cap, so some step has no live request
    from verity_vllm.program.registry.topp_geometry import splits_for
    for t in range(H):
        if want[t] == 0:
            assert L.is_bot(L.SPLITS_BITS, S_t[t]), t
        else:
            assert L.payload(L.SPLITS_BITS, S_t[t]) == splits_for(want[t], SMS), t
    # one cap for both is refused by name in both directions (9 for B's 4 words, 4 for A's 9): the hook takes the fixed population only
    for one_cap in (CAP, CAP_B):
        with pytest.raises(ValueError, match="F-pad-19"):
            Program(bind(PadrevMixedS2, WL=wl.fn, EOS=EOS, CAPS=(one_cap, one_cap), ARR=arrivals, LAG=lag, H=max(a + one_cap for a in arrivals)))
