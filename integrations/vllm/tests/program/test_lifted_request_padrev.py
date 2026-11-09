"""Adversarial tests for the lifted Program OF RECORD (`lift_request_component`), built by node replay over GP-01's request
component.  Attack list items: A3 (post-hoc deletion disguised as "not required"),
A4 (`1‖nonzero` laundered into ⊥ = a corrupt committed word that makes a request look legitimately stopped), the redirect
completeness (an un-redirected reader of event `s−1`'s raw token lets an Active value flow into a ⊥ step), and "nothing about a
served length enters" (guard placement is a function of the body's dataflow and the declared cap only).

Uses `test_lifted_request.py`'s fixture shape (GP-01 `request_component` of the frozen `Serve_v2{TINY}` descriptor; boundary `Final_v2`); nothing here is
implementation."""
from __future__ import annotations

import pytest

from verity.ml.definitions.registry import lifted as L
from verity.ml.definitions.numerics import serve3_reference as ref
from tests.program.test_serve3 import D_MAX, EOS_FOR, TINY, _inputs
from tests.program.test_lifted_request import tiny_component
from verity.ml.circuits.types import Value

CAP = D_MAX + 1
BOUNDARY = "Final_v2"
NEVER = 0x7FFFFFFF


@pytest.fixture(scope="module")
def component():
    return tiny_component(D_MAX)


def _inputs_per_param(params, words):
    out, pos = [], 0
    for _, t in params:
        n = t.leaves
        out.append(L.lift_words(t, words[pos:pos + n]))
        pos += n
    assert pos == len(words)
    return out


def _run(fn, words):
    prog = L.as_program(fn)
    t = ref.materialize(prog, _inputs_per_param(fn.body.params, words))
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _reads_of(body, k):
    """The set of body-node indices node k reads (operand refs that point at nodes)."""
    out = set()
    for a in body.nodes[k].args:
        for r in L._reads(a.refs):
            out.add(r)
    return out


def test_guard_placement_is_the_same_for_every_declared_eos_set(component):
    """Nothing about a stop enters the construction: the guard nodes sit at the same body positions whether the declared set is
    {17} (a_r = 4 on the fixture), {NEVER} (no stop) or {17, NEVER}; only |E| changes the count.  A builder that placed guards by
    looking at where the transcript stops would fail this."""
    l17 = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    lnv = L.lift_request_component(component.fn, (NEVER,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    l2 = L.lift_request_component(component.fn, (17, NEVER), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    assert l17.node_map == lnv.node_map and l17.boundaries == lnv.boundaries == l2.boundaries
    # guard_nodes = the activity guard (eos[i], bot32, the root enable constant enable/one, s=<s>/live, s=<s>/tok_in) + the
    # Continuation_v1 controller (is_eos/out per slot, emit per slot)
    assert len(l17.guard_nodes) == len(lnv.guard_nodes) == L.continuation_gate_count(CAP, 1, 0) == 2 * (CAP - 1) + 1 + 2 + 3 * CAP
    assert len(l2.guard_nodes) == L.continuation_gate_count(CAP, 2, 0) == 2 * (CAP - 1) + 2 + 2 + 3 * CAP  # one more EOS constant, nothing else moves
    # the guard nodes are exactly the named ones — never an unnamed extra node
    names = sorted(l17.body.nodes[k].name for k in l17.guard_nodes)
    ctrl = [f"slot={t}/emit" for t in range(1, CAP + 1)] + [f"slot={t}/is_eos" for t in range(1, CAP + 1)] + [f"slot={t}/out" for t in range(1, CAP + 1)]
    expect = sorted(["eos[0]", "bot32", "enable/one"] + [f"s={s}/live" for s in range(1, CAP)] + [f"s={s}/tok_in" for s in range(1, CAP)] + ctrl)
    assert names == expect
    assert l17.id != lnv.id != l2.id  # the declared set is part of the identity


def test_every_reader_of_a_raw_event_token_is_a_guard_or_the_return(component):
    """Redirect completeness: in the lifted body the raw token gate of event s−1 (the mapped boundary node) is read ONLY by its own
    guard `s=<s>/live` and `s=<s>/tok_in`; every other reader was redirected to `tok_in`.  A missed reader would carry an Active
    token into a ⊥ step — an Active value inside a ⊥ step, which the spec's §4 forbids."""
    lifted = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    body = lifted.body
    by_name = {n.name: k for k, n in enumerate(body.nodes) if n.name}
    for s in range(1, CAP):
        raw = lifted.node_map[lifted.boundaries[s - 1]]
        readers = {k for k in range(len(body.nodes)) if raw in _reads_of(body, k)}
        # its guard and its own slot's controller (`slot=<s>/is_eos`, `slot=<s>/out` read the raw token y_s) — nothing else
        assert readers == {by_name[f"s={s}/live"], by_name[f"s={s}/tok_in"], by_name[f"slot={s}/is_eos"], by_name[f"slot={s}/out"]}, (s, readers)
        # and tok_in_s is read by at least one original (non-guard) node — the step consumes the guarded token
        tok_in = by_name[f"s={s}/tok_in"]
        consumers = {k for k in range(len(body.nodes)) if tok_in in _reads_of(body, k)} - set(lifted.guard_nodes)
        assert consumers, s
    # the last event's raw token is read by nobody (no step beyond the cap)
    last = lifted.node_map[lifted.boundaries[-1]]
    assert {k for k in range(len(body.nodes)) if last in _reads_of(body, k)} == {by_name[f"slot={CAP}/is_eos"], by_name[f"slot={CAP}/out"]}
    # the return reads the designated outputs `slot=<t>/out`, never tok_in and never a raw token directly
    ret_reads = L._reads(body.ret)
    assert ret_reads == {by_name[f"slot={t}/out"] for t in range(1, CAP + 1)}
    assert not any(by_name[f"s={s}/tok_in"] in ret_reads for s in range(1, CAP))


def test_a4_an_inv_prompt_word_is_never_laundered_into_bot(component):
    """A `1‖nonzero` word (INV) inside the prescribed prompt: the boundary refuses it by name; and if a materialisation bypasses the
    boundary, the whole token cone is INV — never ⊥ (which would read as a request that legitimately stopped: A3 via A4) and
    never Active (which would be a value invented from garbage)."""
    lifted = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    prompt, w = _inputs()
    words = list(prompt) + list(w)
    _, _, honest = _run(lifted, words)
    assert all(L.is_active(32, x) for x in honest[:4]) and all(x == L.enc_bot(32) for x in honest[4:])
    # boundary: refused by name
    with pytest.raises(L.BoundaryFailure):
        L.check_boundary(32, L.inv(32), what="prompt[1]", prescribed=True, index_range=TINY["V"])
    # bypass: materialise with prompt[1] = INV directly in the lifted words
    prog = L.as_program(lifted)
    ins = _inputs_per_param(lifted.body.params, words)
    ins[0][1] = L.inv(32)
    t = ref.materialize(prog, ins)
    outs = [t[g] for g in prog.circuit.root_outputs()]
    assert all(x == L.inv(32) for x in outs), [hex(x) for x in outs]
    assert not any(x == L.enc_bot(32) for x in outs)
    # the guard values themselves: live_s = INV_1 (not ⊥_1, not Active(0)), tok_in_s = INV_32
    by_name = {n.name: k for k, n in enumerate(lifted.body.nodes) if n.name}
    n_in = sum(1 for n in prog.root.body.nodes if getattr(n.fn, "family", None) == "Input")
    offs = prog.root.body.offsets
    for s in range(1, CAP):
        live = t[offs[n_in + by_name[f"s={s}/live"]]]
        tok_in = t[offs[n_in + by_name[f"s={s}/tok_in"]]]
        assert live == L.inv(1) and tok_in == L.inv(32), (s, live, tok_in)


def test_a3_a_bot_word_inside_the_prompt_is_refused_and_would_otherwise_delete_the_request(component):
    """⊥ is not a prescribed value: a ⊥ prompt word is refused at the boundary; bypassing it makes every event ⊥ from the prefill on
    — i.e. the whole request "inactive" with zero committed EOS — which the served-prefix rule then refuses (n_r must equal a_r,
    and a_r of an all-⊥ transcript is the cap, not 0)."""
    lifted = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(L.BoundaryFailure):
        L.check_boundary(32, L.enc_bot(32), what="prompt[0]", prescribed=True, index_range=TINY["V"])
    prompt, w = _inputs()
    prog = L.as_program(lifted)
    ins = _inputs_per_param(lifted.body.params, list(prompt) + list(w))
    ins[0][0] = L.enc_bot(32)
    t = ref.materialize(prog, ins)
    outs = [t[g] for g in prog.circuit.root_outputs()]
    assert all(x == L.enc_bot(32) for x in outs)  # strictness: one ⊥ prompt word deletes the request's every event
    # …and the served-prefix rule does not accept that as a stop: a_r of an all-⊥ transcript is the cap; a served length of 0 ≠ cap
    assert L.activity_prefix(outs, (17,), CAP) == CAP
    with pytest.raises(L.ServedPrefixFailure):
        L.check_served_prefix("r0", 0, CAP, lag=0)  # n_r = 0 served events < a_r = cap: "declared activity not executed"
    with pytest.raises((L.ServedPrefixFailure, ValueError)):
        L.assign_inactive("r0", [], (17,), CAP, lag=0)  # and the assignment rule refuses an empty served prefix likewise


def test_values_of_the_active_prefix_do_not_depend_on_the_declared_set(component):
    """A7: the lifting arithmetic changes no Active numeric — the Active prefix under {17} equals, gate for gate, the run under {NEVER}
    (whose every event is Active), on the mapped gates of every original node of events < a_r."""
    l17 = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    lnv = L.lift_request_component(component.fn, (NEVER,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    prompt, w = _inputs()
    words = list(prompt) + list(w)
    p17, t17, o17 = _run(l17, words)
    pnv, tnv, onv = _run(lnv, words)
    assert o17[:4] == onv[:4] and all(x == L.enc_bot(32) for x in o17[4:]) and all(L.is_active(32, x) for x in onv)
    n_in17 = sum(1 for n in p17.root.body.nodes if getattr(n.fn, "family", None) == "Input")
    n_innv = sum(1 for n in pnv.root.body.nodes if getattr(n.fn, "family", None) == "Input")
    o_17, o_nv = p17.root.body.offsets, pnv.root.body.offsets
    a_r_node = l17.boundaries[3]  # the node emitting event 3's token (the EOS): everything up to and including it is the Active prefix
    same = diff = 0
    for k in range(a_r_node + 1):
        k17, knv = l17.node_map[k], lnv.node_map[k]
        g17 = [t17[g] for g in range(o_17[n_in17 + k17], o_17[n_in17 + k17 + 1])]
        gnv = [tnv[g] for g in range(o_nv[n_innv + knv], o_nv[n_innv + knv + 1])]
        assert g17 == gnv, k
        same += len(g17)
    for k in range(a_r_node + 1, len(component.fn.body.nodes)):
        k17, knv = l17.node_map[k], lnv.node_map[k]
        g17 = [t17[g] for g in range(o_17[n_in17 + k17], o_17[n_in17 + k17 + 1])]
        gnv = [tnv[g] for g in range(o_nv[n_innv + knv], o_nv[n_innv + knv + 1])]
        diff += sum(1 for a, b in zip(g17, gnv) if a != b)
    assert same > 1000 and diff > 0


def test_a3_a_served_shape_component_is_refused_against_the_declared_cap():
    """A component derived at the SERVED shape (row_pod's eos-truncated build_request_LP32_T11) is refused against the declared cap: the
    population is never read from the body's boundary count alone, which would lift it with cap = served length."""
    served_steps = D_MAX - 3  # a request Program derived at T = the served length − 1, not at cap − 1
    comp = tiny_component(served_steps)
    with pytest.raises(ValueError):
        L.lift_request_component(comp.fn, (17,), boundary_prefix=BOUNDARY, max_tokens=CAP, lag=0)  # declared cap ≠ the body's population


def test_the_served_shape_component_lifts_only_under_its_own_cap_and_is_a_different_program(component):
    """Companion of the test above: the served-shape component lifts only when the caller declares ITS cap (max_tokens = served length) — a
    different Program (id) from the declared population's; the record tool must never be handed that component."""
    served_steps = D_MAX - 3
    comp = tiny_component(served_steps)
    lifted = L.lift_request_component(comp.fn, (17,), max_tokens=served_steps + 1, lag=0, boundary_prefix=BOUNDARY)
    assert lifted.cap == served_steps + 1 != CAP
    full = L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    assert full.cap == CAP and full.id != lifted.id
    for bad in (CAP - 1, CAP + 1, 0):
        with pytest.raises(ValueError):
            L.lift_request_component(component.fn, (17,), max_tokens=bad, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(ValueError):
        L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=2, boundary_prefix=BOUNDARY)  # lag ∉ {0, 1} refused by name
    with pytest.raises(TypeError):
        L.lift_request_component(component.fn, (17,), max_tokens=CAP, boundary_prefix=BOUNDARY)  # an omitted lag is not a default
