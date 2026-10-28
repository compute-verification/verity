"""Stage DENSE B1, the Program OF RECORD path — **padded baseline (experimental)**.

The Build's Program of record for a B1 workload is the DERIVED FLAT request Program (derive_step `--wrapper request`, `T = cap − 1`),
composed by GP-01 as a `WorkloadRequest_v1{program_digest, program_fn}` component whose body is the flat root with the prescribed
inputs as parameters.  `lifted.lift_request_component(component.fn, EOS, max_tokens=cap)` lifts THAT object by node replay and splices the input
guard (`LStepLive` + `LSelect32`) before the first reader of each event's token — the same construction as `LServe_v1{GUARD=in}`,
now on the flat body GP-01's `address_map.json` is emitted from.

On the laptop there is no derived descriptor (deriving needs the vLLM tree on a pod), so the component under test is GP-01's own
`request_component` applied to the frozen `Serve_v2{TINY, LP=4, STEPS=8}` Program — the same lowering (Input batches + a flat-at-root
body), the same descriptor codec, the same `TokenSelect_v1`-terminated step segmentation (here the boundary node is the `Final_v2`
call that returns the token).  What this establishes, gate for gate on TINY:

  * the lifted component's body is the unlifted one with exactly `2·(cap−1) + |E| + 1` guard gates spliced in; the deletion rule
    recovers the unlifted `(spec, lo, hi)` rows modulo `Lifted[…]` (address stability modulo the guard nodes, as a function);
  * every node reader of event `s−1`'s token is redirected to `tok_in_s`; the returned `out[]` reads the committed tokens;
  * on the fixture inputs with EOS = 17 (`a_r = 4`): events `< a_r` agree bit for bit with the unlifted Program at the mapped gate
    of EVERY node (the whole Active prefix, not only the tokens); from `a_r` on the token cone is ⊥ and `out[s] = ⊥` exactly for
    `s ≥ a_r`; with an EOS the request never emits every event is Active and every gate agrees;
  * the manifest fixes the Program: the digest depends on the EOS set and on nothing about a stop; refusals are by name.

What it does NOT establish: anything about a real record (that needs an early-stop run on a pod).
"""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program, bind
from verity.ml.circuits.codec import _spec_id, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY
from verity_vllm.program import workload as W
from verity.ml.definitions.registry import b1, lifted as L
from verity.ml.definitions.numerics import serve3_reference as ref
from verity.ml.definitions.registry.b1 import weights_type
from tests.program.test_serve3 import D_MAX, EOS_FOR, LP, TINY, Y_RAW, _inputs
from verity.ml.circuits.types import Array, Value

I32 = Value(32)
CAP = D_MAX + 1
BOUNDARY = "Final_v2"  # Serve_v2's root emits `t=<T>/final` = Final_v2{…} → the token; a derived flat body has TokenSelect_v1 at root


def tiny_component(steps: int):
    """GP-01's component of `Serve_v2{TINY, LP, steps}` under the request wrapper's parameter names, as every lifted test builds it: one
    descriptor is one cached component (`workload._REQUEST_CACHE`, `lifted._LREQUEST_CACHE`) whatever the names, so a build under
    other names leaves another test in the process a workload root without `shared.weights`."""
    desc = encode_program(Program(b1.serve_v2(TINY, LP, steps)))
    return W.request_component(desc, REGISTRY, [("prompt", repr(Array(LP, I32))), ("weights", repr(weights_type(TINY)))])


@pytest.fixture(scope="module")
def component():
    return tiny_component(D_MAX)


@pytest.fixture(scope="module")
def lifted(component):
    return L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)


def _inputs_per_param(params, lifted_words: bool):
    prompt, w = _inputs()
    flat = list(prompt) + list(w)
    out, pos = [], 0
    for _, t in params:
        n = t.leaves
        chunk = flat[pos:pos + n]
        pos += n
        out.append(L.lift_words(t, chunk) if lifted_words else chunk)
    assert pos == len(flat)
    return out


def _run(fn, inputs):
    prog = L.as_program(fn)
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _node_gate_ranges(prog):
    """body node k of the lowered function -> [lo, hi) root gate range.  The root body is the Input nodes (one per parameter batch)
    followed by the function's replayed nodes, so function node k is root node n_in + k and gate ranges are the root offsets."""
    body = prog.root.body
    n_in = sum(1 for n in body.nodes if getattr(n.fn, "family", None) == "Input")
    assert all(getattr(n.fn, "family", None) == "Input" for n in body.nodes[:n_in]) and body.offsets[n_in] == prog.input_gates
    return [(body.offsets[k], body.offsets[k + 1]) for k in range(n_in, len(body.nodes))]


def test_component_is_the_flat_program_and_the_lift_has_exactly_the_guard_nodes_more(component, lifted):
    body_u, body_l = component.fn.body, lifted.body
    assert component.n_nodes == len(body_u.nodes)
    assert lifted.cap == CAP and len(lifted.boundaries) == CAP
    # GP-01's own segmentation keys on TokenSelect_v1 at root (the derived flat body); on the Serve_v2 skeleton the boundary is the
    # Final_v2 call, so GP-01 sees one segment here while `request_boundaries` finds the cap events — on a derived body they coincide
    assert component.step_nodes == ((0, len(body_u.nodes)),) and L.request_boundaries(body_u) == []
    assert lifted.id.startswith("LiftedRequest_v2{") and lifted.bindings["EOS"] == (17,) and lifted.bindings["GUARD"] == "in"
    assert lifted.bindings["program_digest"] == component.program_digest and lifted.bindings["program_fn"] == component.program_fn
    # the activity guard: (live, tok_in) per guarded step + |E| + bot; Continuation_v1: is_eos + out per slot, emit per slot (t=1 a constant)
    assert len(body_l.nodes) == len(body_u.nodes) + 2 * (CAP - 1) + 1 + 2 + 3 * CAP
    assert body_l.gates == body_u.gates + 2 * (CAP - 1) + 1 + 2 + 3 * CAP == body_u.gates + L.continuation_gate_count(CAP, 1, 0)
    # G1: the component's parameters are PRESCRIBED inputs -- ordinary, unwidened -- unless the body needs one lifted; the return is lifted
    assert list(body_l.params) == [(n, L.class_type(t, c)) for (n, t), c in zip(body_u.params, lifted.param_classes)] and body_l.ret_type == L.lift_type(body_u.ret_type)
    assert set(lifted.param_classes) == {L.ORDINARY}, lifted.param_classes  # the derived flat request body: weights + prompt, never merged with lifted leaves
    # the deletion rule as a function: original node k ↦ lifted node nidx[k], monotone, same gate count, same spec modulo Lifted[…]
    nidx = lifted.node_map
    assert list(nidx) == sorted(nidx) and len(nidx) == len(body_u.nodes)
    for k, nu in enumerate(body_u.nodes):
        nl = body_l.nodes[nidx[k]]
        assert body_l.offsets[nidx[k] + 1] - body_l.offsets[nidx[k]] == body_u.offsets[k + 1] - body_u.offsets[k]
        assert L.unlifted_id(_spec_id(nl.fn)) == _spec_id(nu.fn)  # Lifted[X_vk]_v2{same bindings, ORD=m} / Lifted[prim]_v2{ORD=m}: the same id modulo wrapper + class vector (G1)
        assert nl.name == nu.name and nl.form == nu.form
    rows_u = [(_spec_id(n.fn), body_u.offsets[k], body_u.offsets[k + 1]) for k, n in enumerate(body_u.nodes)]
    rows_l = L.address_map_modulo_guards(lifted)
    assert [(lo, hi) for _, lo, hi in rows_l] == [(lo, hi) for _, lo, hi in rows_u]
    assert [L.unlifted_id(sl) for sl, _, _ in rows_l] == [_spec_id(n.fn) for n in body_u.nodes]
    # the guard nodes: eos consts + bot32 + t=1/emit at the head, then per slot t (right after its sampling event) the controller's
    # (slot=<t>/is_eos, slot=<t>/out, slot=<t+1>/emit) and (s=<s>/live, s=<s>/tok_in) immediately before each step's first token reader
    names = [body_l.nodes[k].name for k in lifted.guard_nodes]
    assert names[:4] == ["eos[0]", "bot32", "enable/one", "slot=1/emit"]  # the root enable constant after bot32
    ctrl = [n for t in range(1, CAP + 1) for n in ((f"slot={t}/is_eos", f"slot={t}/out") + ((f"slot={t + 1}/emit",) if t < CAP else ()))]
    assert sorted(names[4:]) == sorted(ctrl + [n for s in range(1, CAP) for n in (f"s={s}/live", f"s={s}/tok_in")])
    # G1: the EOS set is ordinary (`Const32`), and the members reading it say so on their id (`{ORD=2}`: the eos_set operand)
    assert {_spec_id(body_l.nodes[k].fn) for k in lifted.guard_nodes} == {"Const32[0x00000011]_v1", "LBot32_v1", "LStepLive1_v1{ORD=2}", "LSelect32_v1",
                                                                             "Const2[0x1]_v1", "LIsEOS1_v1{ORD=2}", "LEmitNext[H=1]_v1"}
    by_name = {body_l.nodes[k].name: k for k in lifted.guard_nodes}
    for s in range(1, CAP):
        b = lifted.boundaries[s - 1]
        live_k = by_name[f"s={s}/live"]
        tok_k = live_k + 1
        assert body_l.nodes[live_k].name == f"s={s}/live" and body_l.nodes[tok_k].name == f"s={s}/tok_in"
        # live reads the committed token of event s−1 and the eos set; tok_in reads (live, token, bot32)
        assert L._reads(body_l.nodes[live_k].args[0].refs) == {nidx[b]}
        assert L._reads(body_l.nodes[tok_k].args[1].refs) == {nidx[b]} and L._reads(body_l.nodes[tok_k].args[2].refs) == {1}
        # every ORIGINAL node reader of the token is now a reader of tok_in, and none reads the token directly
        readers_u = [k for k, n in enumerate(body_u.nodes) if any(b in L._reads(a.refs) for a in n.args)]
        assert readers_u and nidx[readers_u[0]] == tok_k + 1  # the guard sits immediately before the first reader
        for k in readers_u:
            nl = body_l.nodes[nidx[k]]
            assert any(tok_k in L._reads(a.refs) for a in nl.args) and not any(nidx[b] in L._reads(a.refs) for a in nl.args)
    # Continuation_v1: each slot's controller gates sit right after its sampling event and read (y_t, eos_set) / (emit_t, y_t, bot32) / (emit_t, is_eos_t)
    for t in range(1, CAP + 1):
        b = nidx[lifted.boundaries[t - 1]]
        eos_k, out_k = by_name[f"slot={t}/is_eos"], by_name[f"slot={t}/out"]
        assert (eos_k, out_k) == (b + 1, b + 2) and lifted.output_nodes[t - 1] == out_k
        assert L._reads(body_l.nodes[eos_k].args[0].refs) == {b} and L._reads(body_l.nodes[out_k].args[1].refs) == {b} and L._reads(body_l.nodes[out_k].args[2].refs) == {1}
        assert L._reads(body_l.nodes[out_k].args[0].refs) == {by_name[f"slot={t}/emit"]}
        if t < CAP:
            e_k = by_name[f"slot={t + 1}/emit"]
            assert e_k == b + 3 and L._reads(body_l.nodes[e_k].args[0].refs) == {by_name[f"slot={t}/emit"]} and L._reads(body_l.nodes[e_k].args[1].refs) == {eos_k}
    # the return reads the DESIGNATED outputs o_t = LSelect32(emit_t, y_t, bot) (spec s5 C), never the raw tokens nor tok_in; the raw
    # committed tokens (the Active EOS is stop evidence; the lag slot's discarded sample) stay body values
    assert L._reads(body_l.ret) == set(lifted.output_nodes) and lifted.continuation == "continuation-v1"


@pytest.mark.slow
def test_active_prefix_is_bit_exact_at_every_mapped_gate_and_the_rest_is_bot(component, lifted):
    pu, tu, out_u = _run(component.fn, _inputs_per_param(component.fn.body.params, False))
    pl, tl, out_l = _run(lifted, _inputs_per_param(component.fn.body.params, True))
    assert out_u == Y_RAW and pl.input_gates == pu.input_gates
    a_r = L.activity_prefix(out_l, (17,), CAP)
    # out_l is the DESIGNATED output o[]: under A = 0 it equals the raw committed events (served 4, then bot)
    assert a_r == 4 and out_l == Y_RAW[:4] + [L.enc_bot(32)] * (CAP - 4) == L.assign_inactive("tiny", Y_RAW[:4], (17,), CAP, lag=0)
    ru, rl = _node_gate_ranges(pu), _node_gate_ranges(pl)
    nidx = lifted.node_map
    # step segmentation (X-09's unit): event s's segment ends at boundary s (inclusive)
    segs = [((lifted.boundaries[s - 1] + 1) if s else 0, lifted.boundaries[s] + 1) for s in range(CAP)]
    assert segs[-1][1] == len(component.fn.body.nodes)
    active_gates = bot_gates = 0
    for s, (lo, hi) in enumerate(segs):
        for k in range(lo, hi):
            (ulo, uhi), (llo, lhi) = ru[k], rl[nidx[k]]
            assert uhi - ulo == lhi - llo
            widths = [pl.circuit.gate(g).prim.ret.w - 1 for g in range(llo, lhi)]
            if s < a_r:  # the Active prefix: every gate of every node agrees bit for bit (tag 0)
                for gu, gl, w in zip(range(ulo, uhi), range(llo, lhi), widths):
                    assert tl[gl] == tu[gu] and L.is_active(w, tl[gl]), (s, k, gu)
                active_gates += uhi - ulo
            else:  # from a_r on: the token cone is ⊥; the only Active gates are constants and weights-only cones (no token reaches them)
                for gl, w in zip(range(llo, lhi), widths):
                    x = tl[gl]
                    assert L.valid(w, x), (s, k, gl)
                    if L.is_bot(w, x):
                        bot_gates += 1
    assert active_gates > 0 and bot_gates > 0
    # the sampled token of every event s ≥ a_r is ⊥ (the boundary node's gate), and the guard values are the spec functions
    by_name = {lifted.body.nodes[k].name: k for k in lifted.guard_nodes}
    for s in range(1, CAP):
        live_k = by_name[f"s={s}/live"]
        live, tok_in = tl[rl[live_k][0]], tl[rl[live_k + 1][0]]
        assert live == (1 if s < a_r else 0) and tok_in == (out_l[s - 1] if s < a_r else L.enc_bot(32))
    # Continuation_v1 bits: emit_t = 1 for t <= a_r (slot 4 = the EOS, served), 0 after; is_eos_t = 1 exactly at t = 4; out_t = y_t iff emit_t
    assert [tl[rl[by_name[f"slot={t}/emit"]][0]] for t in range(1, CAP + 1)] == [1] * a_r + [0] * (CAP - a_r)
    assert [tl[rl[by_name[f"slot={t}/is_eos"]][0]] for t in range(1, CAP + 1)] == [0] * (a_r - 1) + [1] + [0] * (CAP - a_r)
    assert [tl[rl[by_name[f"slot={t}/out"]][0]] for t in range(1, CAP + 1)] == out_l
    # (the boundary nodes' outputs ARE `out_l` — `body.ret` reads exactly them, checked structurally above — so `out_l[s] = ⊥` for
    # `s ≥ a_r` is the boundary-token check; the gate-level loop above additionally shows every ⊥ gate is a valid representation)
    # the exact count on the returned interface: cap − a_r ⊥ events
    assert sum(1 for x in out_l if L.is_bot(32, x)) == CAP - a_r == L.expected_bot_return_leaves({"NL": 0, "KVH": 0, "D": 0}, CAP, a_r, lag=0)


def test_no_eos_means_every_event_active_and_every_gate_agrees(component):
    lf = L.lift_request_component(component.fn, (63,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)  # 63 is never sampled on the fixture
    pu, tu, out_u = _run(component.fn, _inputs_per_param(component.fn.body.params, False))
    pl, tl, out_l = _run(lf, _inputs_per_param(component.fn.body.params, True))
    assert out_l == out_u == Y_RAW and L.activity_prefix(out_l, (63,), CAP) == CAP
    ru, rl = _node_gate_ranges(pu), _node_gate_ranges(pl)
    for k in range(len(component.fn.body.nodes)):
        (ulo, uhi), (llo, lhi) = ru[k], rl[lf.node_map[k]]
        assert [tl[g] for g in range(llo, lhi)] == [tu[g] for g in range(ulo, uhi)]


def test_a_served_shape_component_is_refused_against_the_declared_cap(component):
    """A component derived at the SERVED shape (`build_request_LP<lp>_T<served>`, the eos-truncated derivation) has fewer sampling events than the declared `max_tokens`; lifting it would make the served length the "fixed population".
    Refused by name; the same component under its own (wrong) count is what the refusal prevents."""
    served = tiny_component(D_MAX - 3)  # served 6 of cap 9
    assert len(L.request_boundaries(served.fn.body, BOUNDARY)) == CAP - 3
    with pytest.raises(ValueError, match=r"has 6 sampling events but the manifest declares max_tokens=9.*F-pad-15"):
        L.lift_request_component(served.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(ValueError, match="declares max_tokens=10"):
        L.lift_request_component(component.fn, (17,), max_tokens=CAP + 1, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(TypeError):  # the declaration is not optional
        L.lift_request_component(component.fn, (17,), boundary_prefix=BOUNDARY)  # type: ignore[call-arg]


def test_the_lift_is_a_function_of_the_manifest_and_refuses_by_name(component, lifted):
    assert L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY) is lifted  # cached: one object per key
    other = L.lift_request_component(component.fn, (17, 63), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    assert other is not lifted and other.body.gates == lifted.body.gates + 1 and other.bindings["EOS"] == (17, 63)
    d1, d2 = program_digest(encode_program(L.as_program(lifted))), program_digest(encode_program(L.as_program(other)))
    assert d1 != d2 and d1 == program_digest(encode_program(L.as_program(lifted)))
    assert _spec_id(lifted) != _spec_id(component.fn) and 'EOS=[17],GUARD="in"' in _spec_id(lifted)
    du = program_digest(encode_program(L.as_program(component.fn)))  # the flat component is a DERIVED object: its own digest, not the record's
    assert du == program_digest(encode_program(L.as_program(component.fn))) and du not in (d1, d2, component.program_digest)
    with pytest.raises(ValueError, match="output guard .* refused by name"):
        L.lift_request_component(component.fn, (17,), guard="out", max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(ValueError, match="EOS set is empty"):
        L.lift_request_component(component.fn, (), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(L.BoundaryFailure):
        L.lift_request_component(component.fn, (L.enc_bot(32),), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)  # ⊥ is not an EOS id
    with pytest.raises(TypeError, match="not a GP-01 request component"):
        L.lift_request_component(b1.serve_v2(TINY, LP, D_MAX), (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(ValueError, match="no TokenSelect_v1 node"):
        L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0)  # Serve_v2's root has no TokenSelect at root: the derived body's boundary is not here


def test_boundary_detection_refuses_a_body_whose_steps_do_not_consume_the_token(component):
    """A body whose event tokens are returned but never read by a later node is not a request (nothing links the events)."""
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, Body, Node
    from verity.ml.circuits.refs import Coll, Affine, concat
    tsel = bind(b1.TokenSelect, V=4)
    d = CompositeDefinition("FakeReq", 1, ("program_digest", "program_fn"), lambda S: ((), None), lambda *_: None, register=False)
    sp = SpecializedDefinition(W._REQUEST_DEF, {"program_digest": "x", "program_fn": "y"}, (("program_digest", "x"), ("program_fn", "y")))
    params = (("logits", Array(4, Value(16))),)
    n0 = Node("call", tsel, (Coll(Array(4, Value(16)), Affine("p", 0, 0, 4)),), I32, name="e0")
    n1 = Node("call", tsel, (Coll(Array(4, Value(16)), Affine("p", 0, 0, 4)),), I32, name="e1")
    sp.set_body(Body(params, Array(2, I32), [n0, n1], concat([Affine("n", 0, 0, 1), Affine("n", 1, 0, 1)]), [0, tsel.gates, 2 * tsel.gates]))
    with pytest.raises(ValueError, match="read by no node: step 1 does not consume it"):
        L.lift_request_component(sp, (0,), max_tokens=2, lag=0)


def test_lag1_candidate_the_placeholder_step_is_active_on_the_eos_and_bit_exact_then_bot(component, lifted):
    """The candidate `LAG=1` (vLLM async scheduling runs the step after the
    EOS on the EOS token and discards its sample).  On the fixture (EOS 17, a_r = 4): steps 1..4 are Active — step 4 takes tok_in = 17 and
    its output `out[4]` equals the UNLIFTED Program's `y_4` bit for bit at every gate (the placeholder's discarded sample is a value of the
    execution, not invented); step 5 is the first ⊥ step (its guard reads tok_in_4 = 17 ∈ E), and every later step stays ⊥ even though
    `out[4]` is an ordinary token: the guard reads the FED token `tok_in_{s−1}` (⊥ from step 5 on), so `live_s = LIsActive(tok_in_s)` holds at
    every step (the P9 hook's frame); a guard on `out[s−2]` alone would report `live_6 = Active(1)` for a step whose input is ⊥.  Guard count:
    step 1 unguarded."""
    l1 = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=1, boundary_prefix=BOUNDARY)
    assert l1 is not lifted and l1.lag == 1 and l1.bindings["LAG"] == 1 and "LAG=1" in l1.id
    assert l1.body.gates == component.fn.body.gates + L.continuation_gate_count(CAP, 1, 1) == component.fn.body.gates + 2 * (CAP - 2) + 3 + 3 * CAP
    names = [component_name for component_name in (n.name for k, n in enumerate(l1.body.nodes) if k in l1.guard_nodes)]
    assert [n for n in names if n.startswith(("eos", "bot", "s="))] == ["eos[0]", "bot32"] + [n for s in range(2, CAP) for n in (f"s={s}/live", f"s={s}/tok_in")]
    pu, tu, out_u = _run(component.fn, _inputs_per_param(component.fn.body.params, False))
    pl, tl, out_l = _run(l1, _inputs_per_param(component.fn.body.params, True))
    ru, rl = _node_gate_ranges(pu), _node_gate_ranges(pl)
    # the RAW committed events (the boundary nodes): Active through the placeholder step -- out[4] = the placeholder's sample = y_4 of the
    # unlifted run -- then bot: the pre-challenge assignment of the events
    # (the unlifted return gates are the boundary nodes' output gates; the deletion rule maps them 1:1 into the lifted node ranges)
    raw_l = [tl[rl[l1.node_map[b]][0] + (gu - ru[b][0])] for b, gu in zip(l1.boundaries, pu.circuit.root_outputs())]
    assert raw_l == Y_RAW[:5] + [L.enc_bot(32)] * (CAP - 5) == L.assign_inactive("tiny", Y_RAW[:5], (17,), CAP, lag=1)
    # the DESIGNATED output (the return, Continuation_v1 s5 C): the served 4 (EOS included), then bot -- the discarded lag sample is NOT in it
    a_r = L.activity_prefix(raw_l, (17,), CAP)
    assert a_r == 4 and out_l == Y_RAW[:4] + [L.enc_bot(32)] * (CAP - 4) and L.served_count(out_l) == 4 and L.served_count(raw_l) == 5
    by_name = {l1.body.nodes[k].name: k for k in l1.guard_nodes}
    assert [tl[rl[by_name[f"slot={t}/emit"]][0]] for t in range(1, CAP + 1)] == [1, 1, 1, 1, 0, 0, 0, 0, 0]  # emit: served slots 1..4
    # run (r_1 = 1, r_2 = 1 unguarded, r_{s+1} = live_s): [1,1,1,1,1,0,0,0,0] -- the (e, r) = (0, 1) lag slot is t = 5
    assert [1, 1] + [tl[rl[by_name[f"s={s}/live"]][0]] for s in range(2, CAP)] == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    segs = [((l1.boundaries[s - 1] + 1) if s else 0, l1.boundaries[s] + 1) for s in range(CAP)]
    for s, (lo, hi) in enumerate(segs):
        for k in range(lo, hi):
            (ulo, uhi), (llo, lhi) = ru[k], rl[l1.node_map[k]]
            if s <= a_r:  # Active through the placeholder step, gate for gate
                assert [tl[g] for g in range(llo, lhi)] == [tu[g] for g in range(ulo, uhi)], (s, k)
    # guards: live_s for s = 2..8 is [1, 1, 1, 0, 0, 0, 0]; tok_in_4 = 17 (the EOS, fed forward), tok_in_5.. = bot
    lives, toks = [], []
    for s in range(2, CAP):
        live_k = by_name[f"s={s}/live"]
        lives.append(tl[rl[live_k][0]])
        toks.append(tl[rl[live_k + 1][0]])
    assert lives == [1, 1, 1, 0, 0, 0, 0] and toks[:3] == Y_RAW[1:4] and toks[2] == 17 and toks[3:] == [L.enc_bot(32)] * 4
    # the alternative form (guard on out[s-2] alone) would report live_6 = Active(1): out[4] = Y_RAW[4] is not an EOS
    assert Y_RAW[4] not in (17,) and L.l_step_live(1).evaluate(L.enc_active(32, Y_RAW[4]), L.enc_active(32, 17)) == 1
    with pytest.raises(ValueError, match="LAG=2"):
        L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=2, boundary_prefix=BOUNDARY)
    with pytest.raises(TypeError):  # the lag is declared, never defaulted
        L.lift_request_component(component.fn, (17,), max_tokens=CAP, boundary_prefix=BOUNDARY)  # type: ignore[call-arg]


@pytest.mark.slow
def test_continuation_v1_words_of_the_lifted_program_equal_the_rules_recurrence_under_both_profiles(component):
    """The controller words the LIFTED PROGRAM computes on the fixture's own
    transcript -- `slot=t/emit`, `slot=t/is_eos`, `slot=t/out` and `r_t` (= 1 at the unguarded slots, else `s=t-1/live`) -- EQUAL
    `continuation.recurrence` over the raw committed events, under A = 0 and A = 1: one mechanism, read off the circuit rather than restated
    (the two fixed lists in the tests above are what the rule gives on this transcript; here the rule itself is the oracle)."""
    from verity_vllm.program.lifting import continuation as K
    pu, tu, out_u = _run(component.fn, _inputs_per_param(component.fn.body.params, False))
    ru = _node_gate_ranges(pu)
    for lag in (0, 1):
        lf = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=lag, boundary_prefix=BOUNDARY)
        pl, tl, out_l = _run(lf, _inputs_per_param(component.fn.body.params, True))
        rl = _node_gate_ranges(pl)
        raw_l = [tl[rl[lf.node_map[b]][0] + (gu - ru[b][0])] for b, gu in zip(lf.boundaries, pu.circuit.root_outputs())]
        raw = [None if L.is_bot(32, x) else L.payload(32, x) for x in raw_l]
        slots = K.recurrence(raw, cap=CAP, A=lag, eos=(17,))
        by_name = {lf.body.nodes[k].name: k for k in lf.guard_nodes}
        emit = [tl[rl[by_name[f"slot={t}/emit"]][0]] for t in range(1, CAP + 1)]
        is_eos = [tl[rl[by_name[f"slot={t}/is_eos"]][0]] for t in range(1, CAP + 1)]
        out = [tl[rl[by_name[f"slot={t}/out"]][0]] for t in range(1, CAP + 1)]
        run = [1] + [tl[rl[by_name[f"s={t - 1}/live"]][0]] if f"s={t - 1}/live" in by_name else 1 for t in range(2, CAP + 1)]
        assert [(bool(e), bool(r)) for e, r in zip(emit, run)] == K.bits(slots), lag
        assert [bool(b) for b in is_eos] == [s.is_eos for s in slots], lag
        o = [None if L.is_bot(32, x) else L.payload(32, x) for x in out]  # o_t = y_t iff e_t, else bot (spec s5 C)
        assert [x for x in o if x is not None] == K.served_output(slots) and [x is not None for x in o] == [s.emit for s in slots], lag
        assert out == out_l and K.executed_slots(slots) == [t for t, r in enumerate(run, 1) if r] and K.lag_slots(slots) == ([5] if lag else [])
        assert K.invariants(slots, cap=CAP, A=lag) == {k: True for k in K.invariants(slots, cap=CAP, A=lag)}
