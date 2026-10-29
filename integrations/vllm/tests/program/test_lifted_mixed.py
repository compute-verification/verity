"""Stage MIXED BATCH, Program side on CPU — **padded baseline (experimental)**.

The lifted fixed WORKLOAD Program: GP-01's `WorkloadProgram` compose — unchanged — over lifted request components
(`lifted.lifted_workload`).  Two declared requests share ONE lifted weights Input; A stops early (its transcript emits the declared EOS
17 at its event 3, `a_A = 4`), B never stops (a shifted prompt; the cap decides).  What this establishes, gate for gate on TINY
(the frozen `Serve_v2{TINY, LP=4, STEPS=8}` as each request's component — the same lowering/codec/compose as a derived flat body):

  * the population is `Σ_r cap_r` sampling events fixed by the manifest, composed by the SAME code that composes the unlifted
    workload; per request the address map is the unlifted rows + that request's own guard rows; the chronology programs
    every request for `cap` steps whatever its transcript; the digest is a function of the manifest + EOS set + LAG, of nothing served;
  * NO CROSS-REQUEST POISONING: with A stopping, every gate of B's call equals B's gate in the UNLIFTED workload (all Active, bit for
    bit), and A's Active prefix equals A's unlifted gates; A's ⊥ is confined to A's own steps `≥ a_A` (per-request state: A's K/V rows
    after `a_A` are ⊥, B's never);
  * the NEGATIVE: a Program with a cross-request edge (B's tokens gated by A's activity — an injected wrong ⊥ propagation) is caught
    twice, by name: its digest/address map is not the Program of record's, and its values fail the interface check `out_B ==
    assign_inactive(B's committed tokens)` (B's events are committed Active where the poisoned Program says ⊥); a ⊥ smuggled into the
    shared weights is refused at the prescribed boundary, and bypassed it deactivates BOTH requests' event 0 — again refused by name
    against each request's own committed tokens (a corrupt input cannot look like a legitimate stop);
  * the S-migration hook (P9, the candidate wiring): `n_live_t` as Program nodes at the workload root equals
    `Σ_r LIsActive(tok_in_{r,t})` read off the components' OWN guard nodes at every engine step, drops by exactly one at `a_A`, follows
    arrivals, and `S_t = LSplitsForSMS<142>(n_live_t)` is Active while any request is live and ⊥ when none is; both `LAG ∈ {0, 1}`.

What it does NOT establish: anything about a real B2 record (the `mixedstop` workload waits on a Commit, like DENSE B1's record half).
"""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, bind, composite
from verity_vllm.program import workload as W
from verity.ml.circuits.refs import Coll, array_of, coll_concat, tuple_of
from verity.ml.definitions.registry import b1, lifted as L
from verity_vllm.program.lifting import lifted_compose
from verity.ml.definitions.numerics import serve3_reference as ref
from verity.ml.definitions.registry.b1 import weights_type
from tests.program.test_serve3 import D_MAX, LP, TINY, Y_RAW, _inputs
from verity.ml.circuits.types import Array, Tuple, Value

I32 = Value(32)
CAP = D_MAX + 1
EOS = (17,)
BOUNDARY = "Final_v2"
BOT32 = L.enc_bot(32)
SMS = 142
B_SHIFT = 1  # prompt_b[0] += 1: the TINY greedy transcript never emits 17 (checked in the fixture)


@pytest.fixture(scope="module")
def component():
    desc = encode_program(Program(b1.serve_v2(TINY, LP, D_MAX)))
    return W.request_component(desc, REGISTRY, [("prompt", repr(Array(LP, I32))), ("weights", repr(weights_type(TINY)))])


def _requests(component, arrivals=(0, 0)):
    return [W.WorkloadRequest("A", 0, LP, CAP, arrivals[0], component), W.WorkloadRequest("B", 1, LP, CAP, arrivals[1], component)]


@pytest.fixture(scope="module")
def wl_u(component):
    return W.WorkloadProgram(_requests(component), lag=0)  # lag is a required static of Workload_v1; the unlifted reference is used for fn/inputs/transcripts


@pytest.fixture(scope="module")
def wl_l(component):
    return lifted_compose.lifted_workload(_requests(component), EOS, lag=0, boundary_prefix=BOUNDARY)


def _prompt_b():
    p, _ = _inputs()
    pb = list(p)
    pb[0] = (pb[0] + B_SHIFT) % TINY["V"]
    return pb


def _inputs_for(fn_u, prompts: dict[str, list[int]], lifted_words: bool, poison: dict | None = None):
    """One flat word list per root parameter of the (un)lifted workload, from the fixture weights and per-request prompts."""
    _, w = _inputs()
    w = list(w)
    out, pos = [], 0
    for name, t in fn_u.signature()[0]:
        if name.startswith("shared."):
            chunk = w[pos:pos + t.leaves]
            pos += t.leaves
        else:
            rid = name[len("request["):name.index("]")]
            chunk = list(prompts[rid])
        words = L.lift_words(t, chunk) if lifted_words else chunk
        if poison and name in poison:
            words = list(words)
            for i, v in poison[name]:
                words[i] = v
        out.append(words)
    assert pos == len(w)
    return out


def _run(prog, inputs):
    t = ref.materialize(prog, inputs)
    return t, [t[g] for g in prog.circuit.root_outputs()]


def _call_gate_base(wl, rid):
    body = wl.program.root.body
    k = wl._K + [r.request_id for r in wl.requests].index(rid)
    return body.offsets[k]


def _request_outs(wl, outs):
    """The root output words split per request (the Workload return is the tuple of the requests' `out[]` arrays)."""
    res, pos = {}, 0
    for r in wl.requests:
        n = r.component.fn.body.ret_type.leaves
        res[r.request_id] = outs[pos:pos + n]
        pos += n
    assert pos == len(outs)
    return res


def _bots(words):
    return sum(1 for x in words if L.is_bot(32, x) or (x >> 16) == 1 and (x & 0xFFFF) == 0)


# ---------------------------------------------------------------------------------------------------------
# T-M1: the fixed population, composed by GP-01's code
# ---------------------------------------------------------------------------------------------------------


def test_the_lifted_workload_is_gp01s_compose_over_lifted_components_with_a_fixed_population(component, wl_u, wl_l):
    assert type(wl_l) is W.WorkloadProgram and wl_l.fn.defn is W.WORKLOAD  # the same compose, the same Definition
    root = wl_l.program.root.body
    calls = root.nodes[wl_l._K:]
    assert [n.name for n in calls] == ["request[A]", "request[B]"] and wl_l._K == wl_u._K  # one shared lifted weights Input set + 2 prompts
    gg = L.continuation_gate_count(CAP, len(EOS), 0)  # the activity guard + the Continuation_v1 controller (3·cap)
    for r_l, r_u in zip(wl_l.requests, wl_u.requests):
        c_l, c_u = r_l.component, r_u.component
        assert c_l.fn.id.startswith("LiftedRequest_v2{") and c_l.fn.bindings["LAG"] == 0 and c_l.fn.bindings["EOS"] == EOS
        assert c_l.program_digest == c_u.program_digest and c_l.program_fn == c_u.program_fn  # the identity is the unlifted component's
        assert c_l.groups == c_u.groups == ("prompt",) + ("weights",) * (c_u.groups.count("weights")) and c_l.input_gates == c_u.input_gates
        assert c_l.body_gates == c_u.body_gates + gg and c_l.n_nodes == c_u.n_nodes + 2 * (CAP - 1) + len(EOS) + 2 + 3 * CAP
        # every declared event is a step of the lifted body: exactly CAP segments (GP-01's `component_steps_equal_cap`).  Slot t's
        # Continuation_v1 controller nodes (is_eos_t, out_t, emit_{t+1}) follow its sampled token, so they fall into the NEXT step's
        # segment; the LAST slot's (is_eos, out; no transition) close the last step's segment (no (cap+1)-th segment)
        assert len(c_l.step_nodes) == CAP and c_l.step_gates[-1][1] == c_l.body_gates
        assert [n.name for n in c_l.fn.body.nodes[c_l.step_nodes[-1][1] - 2:c_l.step_nodes[-1][1]]] == [f"slot={CAP}/is_eos", f"slot={CAP}/out"]
        assert [(n, L.lift_type(t)) for n, t in c_u.params] == list(c_l.params)
    # address map: per request the gate interval is the unlifted one plus its guard gates; the shared group is ONE interval
    am_l, am_u = wl_l.address_map(), wl_u.address_map()
    assert am_l["shared"]["gates"][1] - am_l["shared"]["gates"][0] == am_u["shared"]["gates"][1] - am_u["shared"]["gates"][0]
    for q_l, q_u in zip(am_l["requests"], am_u["requests"]):
        assert q_l["gates"][1] - q_l["gates"][0] == q_u["gates"][1] - q_u["gates"][0] + gg
        assert q_l["program_digest"] == q_u["program_digest"] and q_l["specialization"] != q_u["specialization"]
    # chronology: every request programmed for cap steps, whatever its transcript will be
    ch = wl_l.chronology()
    assert ch["component_steps_equal_cap"] is True and ch["programmed_horizon_steps"] == CAP
    assert all(r["programmed_steps"]["count"] == CAP for r in ch["requests"])
    assert all(len(e["members"]) == 2 for e in ch["engine_steps"])
    # the digest: a function of the manifest, the EOS set and LAG — of nothing served
    d0 = wl_l.digest
    assert d0 != wl_u.digest
    assert lifted_compose.lifted_workload(_requests(component), EOS, lag=0, boundary_prefix=BOUNDARY).digest == d0
    assert lifted_compose.lifted_workload(_requests(component), (63,), lag=0, boundary_prefix=BOUNDARY).digest != d0
    assert lifted_compose.lifted_workload(_requests(component), EOS, lag=1, boundary_prefix=BOUNDARY).digest != d0
    assert lifted_compose.lifted_workload(_requests(component, (0, 2)), EOS, lag=0, boundary_prefix=BOUNDARY).digest != d0  # arrivals are declared
    with pytest.raises(ValueError, match="never from a served length"):
        lifted_compose.lifted_workload([W.WorkloadRequest("A", 0, LP, CAP, 0, component, T=3)], EOS, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(ValueError, match="F-pad-15"):  # a component of a smaller cap under a larger declaration
        lifted_compose.lifted_workload([W.WorkloadRequest("A", 0, LP, CAP + 1, 0, component)], EOS, lag=0, boundary_prefix=BOUNDARY)
    with pytest.raises(TypeError):
        lifted_compose.lifted_workload(_requests(component), EOS, boundary_prefix=BOUNDARY)  # type: ignore[call-arg]  # the lag is declared


# ---------------------------------------------------------------------------------------------------------
# T-M2/T-M3: A stops, B continues — nothing crosses; state is per request
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(wl_u, wl_l):
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    tu, ou = _run(wl_u.program, _inputs_for(wl_u.fn, prompts, False))
    tl, ol = _run(wl_l.program, _inputs_for(wl_u.fn, prompts, True))
    return tu, _request_outs(wl_u, ou), tl, _request_outs(wl_l, ol)


@pytest.mark.slow
def test_a_stops_early_and_b_continues_with_every_gate_of_b_bit_equal_to_the_unlifted_workload(wl_u, wl_l, runs):
    tu, ou, tl, ol = runs
    assert ou["A"] == Y_RAW and 17 not in ou["B"], "fixture: A emits the EOS at event 3, B never"
    a_A = L.activity_prefix(ol["A"], EOS, CAP)
    assert a_A == 4 and ol["A"] == Y_RAW[:4] + [BOT32] * (CAP - 4) == L.assign_inactive("A", Y_RAW[:4], EOS, CAP, lag=0)
    assert ol["B"] == ou["B"] == L.assign_inactive("B", ou["B"], EOS, CAP, lag=0)  # B: every event Active, the unlifted tokens
    # B's WHOLE call: every unlifted node's gates equal the lifted node's gates (the deletion rule's map), all Active
    for rid, a_r in (("B", CAP), ("A", a_A)):
        c_u = next(r for r in wl_u.requests if r.request_id == rid).component
        c_l = next(r for r in wl_l.requests if r.request_id == rid).component
        Gu, Gl = _call_gate_base(wl_u, rid), _call_gate_base(wl_l, rid)
        nidx = c_l.fn.node_map
        bu, bl = c_u.fn.body, c_l.fn.body
        seg_of = {}  # lifted node -> event index (segments of the lifted component)
        for s, (lo, hi) in enumerate(c_l.step_nodes):
            for k in range(lo, hi):
                seg_of[k] = s
        agree = 0
        for k in range(len(bu.nodes)):
            lu = [tu[Gu + g] for g in range(bu.offsets[k], bu.offsets[k + 1])]
            ll = [tl[Gl + g] for g in range(bl.offsets[nidx[k]], bl.offsets[nidx[k] + 1])]
            if seg_of[nidx[k]] < a_r:
                assert ll == lu, (rid, k)  # Active: the lifted word IS the unlifted word
                agree += len(lu)
        assert agree > 10_000
    # B's guard nodes: live_s = Active(1) and tok_in_s = B's own committed token at every step; the only ⊥ word in B's whole call range
    # is its `bot32` CONSTANT (the guard's unselected branch), and no bf16 word is ⊥
    GB = _call_gate_base(wl_l, "B")
    cB = next(r for r in wl_l.requests if r.request_id == "B").component
    bl = cB.fn.body
    for k in cB.fn.guard_nodes:
        nm = bl.nodes[k].name
        if nm.endswith("/live"):
            assert tl[GB + bl.offsets[k]] == 1, nm
        elif nm.endswith("/tok_in"):
            assert tl[GB + bl.offsets[k]] == ou["B"][int(nm[len("s="):nm.index("/")]) - 1], nm
    words_B = [tl[GB + g] for g in range(cB.body_gates)]
    assert sum(1 for x in words_B if L.is_bot(32, x)) == 1 and not any(L.is_bot(16, x) for x in words_B)
    sh = wl_l.address_map()["shared"]["gates"]
    assert [tl[g] for g in range(*sh)] == [tu[g] for g in range(*wl_u.address_map()["shared"]["gates"])]


@pytest.mark.slow
def test_a_s_bot_is_confined_to_its_own_steps_from_a_r_on_and_b_s_state_is_never_bot(wl_l, runs):
    _, _, tl, ol = runs
    a_A = L.activity_prefix(ol["A"], EOS, CAP)
    for rid, a_r in (("A", a_A), ("B", CAP)):
        c = next(r for r in wl_l.requests if r.request_id == rid).component
        G = _call_gate_base(wl_l, rid)
        bl = c.fn.body
        skip = {bl.offsets[k] for k in c.fn.guard_nodes if bl.nodes[k].name == "bot32"}  # the guard's ⊥ constant is not state
        # the Continuation_v1 controller words (`slot=<t>/…`) are the designated-output side, not the step's state; they are
        # counted separately below (they sit in the segment AFTER their token, so they would skew the per-step K/V growth constant)
        skip |= {bl.offsets[k] for k, n in enumerate(bl.nodes) if (n.name or "").startswith("slot=")}
        per_step = []
        for s, (lo, hi) in enumerate(c.step_gates):
            words = [tl[G + g] for g in range(lo, hi) if g not in skip]
            per_step.append(sum(1 for x in words if L.is_bot(32, x) or L.is_bot(16, x)))  # token cone (32) and K/V / activation (16) ⊥ words
        assert len(per_step) == CAP and [n > 0 for n in per_step] == [s >= a_r for s in range(CAP)], (rid, per_step)
        # the Continuation_v1 controller words are confined too -- the request's OWN `slot=<t>/out` is ⊥ exactly for the
        # non-emitted slots t > a_r (LAG 0: the EOS slot itself is emitted); `emit`/`is_eos` are Boolean words, never ⊥; B (no EOS) has none
        names = [n.name or "" for n in bl.nodes]
        outs = {t: tl[G + bl.offsets[names.index(f"slot={t}/out")]] for t in range(1, CAP + 1)}
        assert [t for t, x in outs.items() if L.is_bot(32, x)] == list(range(a_r + 1, CAP + 1)), (rid, a_r)
        for t in range(1, CAP + 1):
            for k in ("emit", "is_eos"):
                x = tl[G + bl.offsets[names.index(f"slot={t}/{k}")]]
                assert x in (0, 1), (rid, t, k, x)
        if rid == "A":
            assert min(per_step[a_r:]) > 100  # every inactive step carries its ⊥ K/V rows and activations, not only its token
            # the K/V-cache-like state: every inactive step APPENDS one ⊥ K/V row per layer that every later step reads (its score and
            # probability for that position are ⊥ per head per layer), so the ⊥ count grows by the same constant from step to step
            diffs = {b - a for a, b in zip(per_step[a_r:], per_step[a_r + 1:])}
            # measured constant on TINY: 28 words per additional ⊥ row = 24 (numerically 2·NL·NH·3, the K/V reads; the per-read decomposition
            # is not asserted) + 4 (the per-position constants a longer attention owns are ⊥ inside a disabled step too)
            assert len(diffs) == 1 and diffs.pop() == 28, per_step[a_r:]


# ---------------------------------------------------------------------------------------------------------
# T-M4: the negative — a wrong ⊥ propagation is caught
# ---------------------------------------------------------------------------------------------------------


@composite("PoisonedWorkload", 1, ["WL", "EOS", "CAP"],
           lambda S: (S.WL.signature()[0], S.WL.signature()[1]), conformance="protocol-defined")
def PoisonedWorkload(B, S, *args):
    """The WRONG Program: B's committed tokens gated by A's activity (a cross-request edge that no honest lift has)."""
    wl = B.call(S.WL, *args, name="workload")
    eos_cols = [B.call(b1.const(33, e), name=f"eos[{i}]") for i, e in enumerate(S.EOS)]
    eos_set = coll_concat(Array(len(S.EOS), Value(33)), eos_cols)
    bot = B.call(L.l_bot(32), name="bot32")
    out_a, out_b = wl[0], wl[1]
    gated = []
    for s in range(S.CAP):
        live_a = B.call(L.l_step_live(len(S.EOS)), out_a[s - 1] if s else out_a[0], eos_set, name=f"poison/live_a[{s}]")
        gated.append(B.call(L.l_select(32), live_a, out_b[s], bot, name=f"poison/out_b[{s}]"))
    return tuple_of(out_a, array_of(gated))


@pytest.mark.slow
def test_negative_a_cross_request_bot_edge_is_caught_by_identity_and_by_the_interface_check(wl_u, wl_l):
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    bad = Program(bind(PoisonedWorkload, WL=wl_l.fn, EOS=EOS, CAP=CAP))
    # identity: the poisoned Program is not the Program of record (digest, gate count, root shape)
    assert program_digest(encode_program(bad)) != wl_l.digest and bad.gates > wl_l.program.gates
    # values: A is untouched; B's events from a_A on are ⊥ in the poisoned Program while B's committed tokens are Active there
    _, outs = _run(bad, _inputs_for(wl_u.fn, prompts, True))
    out_a, out_b = outs[:CAP], outs[CAP:]
    assert out_a == Y_RAW[:4] + [BOT32] * (CAP - 4)
    committed_b = _prompt_b_transcript(wl_u, prompts)
    assert out_b[:4] == committed_b[:4] and out_b[4:] == [BOT32] * (CAP - 4)
    want = L.assign_inactive("B", committed_b, EOS, CAP, lag=0)  # the served-prefix rule on B's OWN committed tokens: all Active
    assert want == committed_b and want != out_b
    bad_events = [s for s in range(CAP) if out_b[s] != want[s]]
    assert bad_events == list(range(4, CAP))  # exactly the events A's stop was wrongly propagated to
    # the honest Program passes the same check on the same tokens
    _, ol = _run(wl_l.program, _inputs_for(wl_u.fn, prompts, True))
    assert _request_outs(wl_l, ol)["B"] == want


def _prompt_b_transcript(wl_u, prompts):
    _, ou = _run(wl_u.program, _inputs_for(wl_u.fn, prompts, False))
    return _request_outs(wl_u, ou)["B"]


@pytest.mark.slow
def test_negative_a_bot_smuggled_into_the_shared_weights_is_refused_at_the_boundary_and_bypassed_deactivates_no_request_legitimately(wl_u, wl_l):
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    name, t = next((n, t) for n, t in wl_u.fn.signature()[0] if n.startswith("shared."))
    w = t.leaf_widths()[0]
    with pytest.raises(L.BoundaryFailure):
        L.check_boundary(w, L.enc_bot(w), what=name, prescribed=True)  # a prescribed leaf is never ⊥: refused by name
    # bypassing the boundary: the ⊥ weight reaches every request through the shared Input — event 0 of BOTH goes ⊥ …
    _, outs = _run(wl_l.program, _inputs_for(wl_u.fn, prompts, True, poison={name: [(0, L.enc_bot(w))]}))
    per = _request_outs(wl_l, outs)
    assert all(L.is_bot(32, per[r][0]) for r in ("A", "B"))
    # … and no request can pass its own served-prefix check with its committed tokens: a corrupt input is never a legitimate stop
    for rid, tokens in (("A", Y_RAW[:4]), ("B", _prompt_b_transcript(wl_u, prompts))):
        want = L.assign_inactive(rid, tokens, EOS, CAP, lag=0)
        assert want[0] != per[rid][0] and L.is_active(32, want[0])
        assert all(L.is_bot(32, x) for x in per[rid])  # strictness: the whole token cone


# ---------------------------------------------------------------------------------------------------------
# T-M5: the S-migration hook as Program nodes (P9; the candidate wiring)
# ---------------------------------------------------------------------------------------------------------


def _caps(S):
    return [S.CAP] * len(S.ARR) if isinstance(S.CAP, int) else list(S.CAP)


def _horizon(S):
    return max(a + c for a, c in zip(S.ARR, _caps(S)))


@composite("LWorkloadS", 1, ["WL", "EOS", "LAG", "SMS", "ARR", "CAP"],
           lambda S: (S.WL.signature()[0],
                      Tuple(S.WL.signature()[1], Array(_horizon(S), Value(L.N_LIVE_BITS + 1)), Array(_horizon(S), Value(L.SPLITS_BITS + 1)))),
           conformance="protocol-defined")
def LWorkloadS(B, S, *args):
    wl = B.call(S.WL, *args, name="workload")
    eos_cols = [B.call(b1.const(33, e), name=f"eos[{i}]") for i, e in enumerate(S.EOS)]
    eos_set = coll_concat(Array(len(S.EOS), Value(33)), eos_cols)
    bot = B.call(L.l_bot(32), name="bot32")
    outs = [wl[r] for r in range(len(S.ARR))]
    n_live, Sx = L.live_count_hook(B, outs, list(S.ARR), S.CAP, eos_set, bot, S.LAG, num_sms=S.SMS)
    return tuple_of(wl, array_of(n_live), array_of(Sx))


def _hook_run(component, arrivals, prompts, lag, eos=EOS):
    wl = lifted_compose.lifted_workload(_requests(component, arrivals), eos, lag=lag, boundary_prefix=BOUNDARY)
    wl_u = W.WorkloadProgram(_requests(component, arrivals), lag=lag)
    prog = Program(bind(LWorkloadS, WL=wl.fn, EOS=eos, LAG=lag, SMS=SMS, ARR=tuple(arrivals), CAP=CAP))
    t, outs = _run(prog, _inputs_for(wl_u.fn, prompts, True))
    H = max(a + CAP for a in arrivals)
    n_out = 2 * CAP
    return wl, prog, t, outs[:n_out], outs[n_out:n_out + H], outs[n_out + H:]


def _wl_call_base(wl, prog, rid):
    body = prog.root.body
    k_wl = next(k for k, n in enumerate(body.nodes) if n.name == "workload")
    inner = wl.program.root.body
    k = wl._K + [r.request_id for r in wl.requests].index(rid)
    return body.offsets[k_wl] + inner.offsets[k] - inner.offsets[wl._K]


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.slow
def test_the_live_count_hook_equals_the_components_own_input_activity_and_drops_by_one_at_a_A(component, lag):
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    wl, prog, t, outs, n_live, S = _hook_run(component, (0, 0), prompts, lag)
    per = _request_outs(wl, outs)
    a_A = L.activity_prefix(per["A"], EOS, CAP)
    n_exec = L.expected_served(a_A, CAP, lag)
    assert a_A == 4 and per["B"] == _prompt_b_transcript(W.WorkloadProgram(_requests(component), lag=lag), prompts)
    # the hook's count per engine step
    counts = [L.payload(L.N_LIVE_BITS, x) for x in n_live]
    assert all(L.is_active(L.N_LIVE_BITS, x) for x in n_live)
    assert counts == [2] * n_exec + [1] * (CAP - n_exec)
    # == Σ_r LIsActive(tok_in_{r,t}) read off each component's OWN guard nodes (P9, the Program's values, not the hook's recomputation)
    acts = {}
    for rid in ("A", "B"):
        c = next(r for r in wl.requests if r.request_id == rid).component
        lifted, bl = c.fn, c.fn.body
        G = _wl_call_base(wl, prog, rid)
        ia = L.l_is_active(32)
        toks = {bl.nodes[k].name: k for k in lifted.guard_nodes if bl.nodes[k].name.endswith("/tok_in")}
        a = []
        for s in range(CAP):
            if s <= lag:  # unguarded steps read the raw committed token (event 0 = the prefill's output, always Active)
                a.append(1 if s == 0 else ia.evaluate(t[G + bl.offsets[lifted.node_map[lifted.boundaries[s - 1]]]]))
            else:
                a.append(ia.evaluate(t[G + bl.offsets[toks[f"s={s}/tok_in"]]]))
        acts[rid] = a
    assert acts["B"] == [1] * CAP and acts["A"] == [1] * n_exec + [0] * (CAP - n_exec)
    assert counts == [acts["A"][s] + acts["B"][s] for s in range(CAP)]
    # S_t: Active(32) at 142 SMs for n ∈ {1, 2} (vLLM's split table: 142 SMs, 4 → 32 is the first breakpoint), never ⊥ while B is live
    assert [L.payload(L.SPLITS_BITS, x) for x in S] == [32] * CAP and all(L.is_active(L.SPLITS_BITS, x) for x in S)


@pytest.mark.slow
def test_the_hook_follows_declared_arrivals_and_S_is_bot_when_no_request_is_live(component):
    # arrivals (0, 2): the horizon is 11 engine steps; A alone for t < 2, both until A stops, B alone after
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    wl, prog, t, outs, n_live, S = _hook_run(component, (0, 2), prompts, 0)
    per = _request_outs(wl, outs)
    a_A = L.activity_prefix(per["A"], EOS, CAP)
    counts = [L.payload(L.N_LIVE_BITS, x) for x in n_live]
    assert len(counts) == CAP + 2 and counts == [1, 1] + [2] * (a_A - 2) + [1] * (CAP + 2 - a_A)
    assert all(L.is_active(L.SPLITS_BITS, x) for x in S)
    # both requests stop (the same prompt twice): from a_r on no request is live — n_live = Active(0) and S = ⊥ (no sampler launch)
    same = {"A": list(_inputs()[0]), "B": list(_inputs()[0])}
    wl2, prog2, t2, outs2, n2, S2 = _hook_run(component, (0, 0), same, 0)
    per2 = _request_outs(wl2, outs2)
    assert per2["A"] == per2["B"] == Y_RAW[:4] + [BOT32] * (CAP - 4)
    assert [L.payload(L.N_LIVE_BITS, x) for x in n2] == [2] * 4 + [0] * (CAP - 4)
    assert all(L.is_active(L.N_LIVE_BITS, x) for x in n2)  # the COUNT is Active(0), never ⊥ (it is a value of the Program)
    assert [L.is_bot(L.SPLITS_BITS, x) for x in S2] == [False] * 4 + [True] * (CAP - 4)  # the SPLIT COUNT is ⊥: no event
    # structural refusals of the hook
    with pytest.raises(ValueError, match="requests"):
        L.live_count_prim(65)
    with pytest.raises(ValueError, match="LAG=3"):
        lifted_compose.lifted_workload(_requests(component), EOS, lag=3, boundary_prefix=BOUNDARY)


# GP-01's populations have UNEQUAL caps (its standard B2 fixture is cap 3 / cap 2). The hook takes one cap per request.

D_MAX_B = 3
CAP_B = D_MAX_B + 1


@pytest.fixture(scope="module")
def component_b():
    desc = encode_program(Program(b1.serve_v2(TINY, LP, D_MAX_B)))
    return W.request_component(desc, REGISTRY, [("prompt", repr(Array(LP, I32))), ("weights", repr(weights_type(TINY)))])


@pytest.mark.slow
def test_f_pad_19_the_hook_takes_one_cap_per_request_and_refuses_a_served_length_by_name(component, component_b):
    reqs = [W.WorkloadRequest("A", 0, LP, CAP, 0, component), W.WorkloadRequest("B", 1, LP, CAP_B, 0, component_b)]
    prompts = {"A": list(_inputs()[0]), "B": _prompt_b()}
    for lag in (0, 1):
        wl = lifted_compose.lifted_workload(reqs, EOS, lag=lag, boundary_prefix=BOUNDARY)
        wl_u = W.WorkloadProgram(reqs, lag=lag)
        # ONE cap for both is refused by name -- 9 would read B's out[4..8] (an UNNAMED IndexError), 4 would truncate A's schedule
        for one_cap in (CAP, CAP_B):
            with pytest.raises(ValueError, match="F-pad-19"):
                Program(bind(LWorkloadS, WL=wl.fn, EOS=EOS, LAG=lag, SMS=SMS, ARR=(0, 0), CAP=one_cap))
        with pytest.raises(ValueError, match="one cap per request"):
            Program(bind(LWorkloadS, WL=wl.fn, EOS=EOS, LAG=lag, SMS=SMS, ARR=(0, 0), CAP=(CAP,)))
        # per-request caps: horizon 9; B is a member of steps [0, 4) only, A of [0, 9)
        prog = Program(bind(LWorkloadS, WL=wl.fn, EOS=EOS, LAG=lag, SMS=SMS, ARR=(0, 0), CAP=(CAP, CAP_B)))
        t, outs = _run(prog, _inputs_for(wl_u.fn, prompts, True))
        n_out = CAP + CAP_B
        per = _request_outs(wl, outs[:n_out])
        n_live = outs[n_out:n_out + CAP]
        Sx = outs[n_out + CAP:]
        assert len(n_live) == CAP and len(Sx) == CAP
        a_A = L.activity_prefix(per["A"], EOS, CAP)
        assert a_A == 4 and L.activity_prefix(per["B"], EOS, CAP_B) == CAP_B  # A stops at 4 (the fixture), B runs to ITS cap (no EOS)
        live_A = [1] * min(a_A + lag, CAP) + [0] * (CAP - min(a_A + lag, CAP))
        live_B = [1] * CAP_B + [0] * (CAP - CAP_B)  # a non-member contributes nothing
        assert [L.payload(L.N_LIVE_BITS, x) for x in n_live] == [a + b for a, b in zip(live_A, live_B)]
        assert [L.is_bot(L.SPLITS_BITS, x) for x in Sx] == [c == 0 for c in (a + b for a, b in zip(live_A, live_B))]
        # the members per engine step are structural: 2 for t < CAP_B, 1 after -- read off the LiveCount widths
        counts = [n.fn.id for n in prog.root.body.nodes if n.name and n.name.startswith("S/n_live[")]
        assert counts == [f"Lifted[LiveCount2_v1]_v1"] * CAP_B + [f"Lifted[LiveCount1_v1]_v1"] * (CAP - CAP_B)
