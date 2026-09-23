"""`verity_vllm.lifting.record` — the record half's checker (R1–R6) exercised on a SYNTHETIC record built from the TINY fixture:
GP-01's request component of the frozen `Serve_v2{TINY, LP=4, STEPS=8}` descriptor (cap 9), the fixture's committed transcript
(raw greedy `[15, 23, 26, 17, 26, 17, 26, 17, 26]`, EOS 17 → `a_r = 4`, the record serves 4 tokens), and a binding map in
`CommittedStore`'s schema with identities at the served steps only.  **padded baseline (experimental)**; not a record of any row.

Positive: every check R1–R5 PASS, R6 "not evaluated"; the ⊥ population per event is the whole token cone of events 4..8 (every
original node of those segments — on this fixture no weights-only node exists inside a step), Active events exactly [0, 4).
Negatives (each one flips exactly the check it attacks): a step ≥ a_r committed (declared inactivity executed); a served step missing
from the map; a transcript that continues past the EOS (execution past EOS); a transcript shorter than the Active prefix (declared
activity not executed); a ⊥ word inside the served prefix; a wrong cap; a tampered closure (the lifted body's guard replaced) is
caught by R4's guard/live check.
"""
from __future__ import annotations

import copy
import json

import pytest

from verity.ir import Program
from verity_vllm.lifting import continuation as K
from verity_vllm.lifting import record as R
from verity_vllm.ir import workload as W
from verity.ir.defs import REGISTRY
from verity.ir.codec import encode_program
from verity_vllm.registry import b1, lifted as L
from tests.ir.test_serve3 import D_MAX, LP, TINY, Y_RAW
from verity.ir.types import Value

CAP = D_MAX + 1
BOUNDARY = "Final_v2"
EOS = (17,)
A_R = 4
SERVED = Y_RAW[:A_R]  # the record's committed transcript: prefill event + 3 decode events, the last is the EOS


@pytest.fixture(scope="module")
def desc():
    return encode_program(Program(b1.serve_v2(TINY, LP, D_MAX)))


def _bmap(steps, rid="r0"):
    """A binding map in CommittedStore's schema: one identity per (step, op) for the given engine steps."""
    entries = []
    for st in steps:
        for op, member in (("model.embed_tokens", "in"), ("model.layers.0.mlp.down_proj", "0"), ("sampler", "sampled_token_ids")):
            entries.append({"name": f"{op}/{member}@{st}", "identity": {"request_id": rid, "step": st, "op_path": op, "output_member": member,
                                                                       "element_range": [0, 1]}, "encoding": {"dtype": "i64"}, "element_offset": 0})
    return {"schema": "test", "entries": entries}


def _header(lag):
    """A synthetic RunHeader carrying the ONE attested field (RULING 21): the resolved SchedulerConfig.async_scheduling."""
    return {"profile": {"engine_kwargs": {"enforce_eager": True, "async_scheduling": bool(lag)}}}


def _check(desc, transcript=SERVED, eos=EOS, cap=CAP, steps=range(A_R), lag=0, bmap="steps", **kw):
    if lag in (0, 1) and "lag_source" not in kw:
        lag, kw["lag_source"] = L.lag_from_header(_header(lag))  # the lag is READ from the attested header, not declared by the test
    # the synthetic Commit run row carries the DRIVER's token count (a_r: the EOS in, a placeholder's sample out -- sys's real int-3 row:
    # 11 / 12 under LAG 1) and the engine's LOOP count (engine_steps == iterations == a_r + LAG whatever the stop reason -- the OLMoE
    # length-stop row: 32 / 33 under LAG 1)
    y = list(transcript)
    gen = next((i + 1 for i, t in enumerate(y) if t in eos), len(y)) if len(y) <= cap else len(y)
    return R.check_record(desc, transcript, eos, cap=cap, bmap=_bmap(steps) if bmap == "steps" else bmap, request_id="r0", boundary_prefix=BOUNDARY,
                          lag=lag, run_row={"generated_tokens": gen, "engine_steps": gen + (lag or 0), "iterations": gen + (lag or 0)}, **kw)


def test_f_pad_17_an_unattested_lag_or_a_missing_binding_map_is_never_a_record_pass(desc):
    """padrev M-0203 F-pad-17 / M-0168; coord RULING 21: `ok` is a verdict only with an attested lag AND an evaluated executed set."""
    good = _check(desc)
    assert good["ok"] is True and good["checks"]["R0_lag_attestation"]["ok"] and good["lag_source"] == L.LAG_FIELD and good["candidate"] == L.DP_PAD_02
    assert "NOT evaluated" in good["scope"]
    # (b) the same record with the lag declared by the caller and no attested field: FAIL by name, every other check untouched
    for src in ("unspecified", "", "cli"):
        bad = _check(desc, lag_source=src)
        assert bad["ok"] is False and bad["checks"]["R0_lag_attestation"]["ok"] is False and "no attested header field" in bad["checks"]["R0_lag_attestation"]["why"]
        assert all(bad["checks"][k]["ok"] is True for k in ("R1_population", "R2_activity", "R3_executed_set", "R4_assignment", "R5_interface"))
    # (a) no binding map: the Program's own closure passes but there is NO record verdict (None), and the scope says so
    prog_only = _check(desc, bmap=None)
    assert prog_only["ok"] is None and prog_only["checks"]["R3_executed_set"]["ok"] is None and "NO record verdict" in prog_only["scope"]
    assert _check(desc, bmap=None, transcript=Y_RAW[:A_R - 1])["ok"] is False  # a failing closure is still False without a map
    # the header reader: absent / unresolved (None) / non-boolean attestations are refused by name; True → 1, False → 0
    assert L.lag_from_header(_header(1)) == (1, L.LAG_FIELD) and L.lag_from_header(_header(0)) == (0, L.LAG_FIELD)
    for h in ({}, {"profile": {}}, {"profile": {"engine_kwargs": {}}}, {"profile": {"engine_kwargs": {"async_scheduling": None}}},
              {"profile": {"engine_kwargs": {"async_scheduling": 1}}}):
        with pytest.raises(ValueError, match="async_scheduling"):
            L.lag_from_header(h)


def test_positive_record_passes_every_disk_check_and_reports_the_bot_population(desc):
    rep = _check(desc)
    ck = rep["checks"]
    assert rep["ok"] and all(ck[k]["ok"] is True for k in ("R1_population", "R2_activity", "R3_executed_set", "R3b_run_row", "R4_assignment", "R5_interface"))
    assert ck["R6_values"]["ok"] is None and "not evaluated" in ck["R6_values"]["why"]
    # R17-32: the R17-1 guard (2 per guarded step + |E| + bot) + the Continuation_v1 controller (is_eos, out per slot; emit per slot)
    assert ck["R1_population"]["guard_gates"] == 2 * (CAP - 1) + 1 + 2 + 3 * CAP == L.continuation_gate_count(CAP, 1, 0)
    assert ck["R1_population"]["guard_gates_r17_1"] == 2 * (CAP - 1) + 1 + 2 and ck["R1_population"]["controller_gates_continuation_v1"] == 3 * CAP
    assert ck["R1_population"]["deletion_rule_ranges"] and ck["R1_population"]["deletion_rule_specs"]
    assert ck["R2_activity"]["a_r"] == A_R and ck["R2_activity"]["stopped_by_eos"] and ck["R2_activity"]["first_eos_index"] == A_R - 1
    r4 = ck["R4_assignment"]
    assert r4["active_events"] == A_R and r4["guards_live_ok"] and r4["active_prefix_all_active"] and r4["inactive_boundaries_bot"]
    assert r4["mixed_operand_nodes_in_active_steps"] == []
    # the population: every original node of events >= a_r is bot (the fixture has no weights-only node inside a step)
    rows = rep["assignment"]
    for s in range(CAP):
        orig = [r for r in rows if r["event"] == s and not r["guard"]]
        assert orig and all((r["tag"] == "A") == (s < A_R) for r in orig), s
    assert set(r4["bot_leaves_by_event"]) == {str(s) for s in range(A_R, CAP)}
    per = {int(k): v for k, v in r4["bot_leaves_by_event"].items()}
    assert len(set(per.values())) == 1 and per[A_R] > 1  # the same leaf count at every inactive decode step
    assert r4["bot_words_total"] >= r4["bot_leaves_total"] == sum(per.values())
    # R4' (LIFTING-SPEC v0.6 B9): the ENABLE closure -- padded-v1's meaning -- agrees with propagation on every step's activity, names the
    # disabled-but-Active class (constants / weights-only gates inside disabled steps) with counts, and never finds a bot on a taken path
    r4e = ck["R4e_enable_closure"]
    assert r4e["ok"] is True and r4e["rules_version"] == L.PADDED_RULES_VERSION == "padded-v1" and rep["program_rules_version"] == "padded-v1"
    assert r4e["run_by_event"] == [1] * A_R + [0] * (CAP - A_R) and r4e["enables_monotone"] and r4e["enabled_but_bot_n"] == 0
    assert r4e["bot_words_total_enable"] - r4e["bot_words_total_propagation"] == r4e["extra_bot_words_under_enable"] >= 0
    assert set(r4e["bot_words_by_event_enable"]) == {str(s) for s in range(A_R, CAP)}
    assert r4e["materialization_projection"]["projected_not_measured"] is True
    assert r4["guard_bot_leaves"] == r4["guard_bot_leaves_expected"] == 1 + (CAP - A_R) + (CAP - A_R)  # bot32 + tok_in_s for s in [a_r, cap) + out_t for t > a_r (R17-32)
    assert {r["spec"] for r in rows if r["guard"]} == {"Const32[0x00000011]_v1", "LBot32_v1", "LStepLive1_v1{ORD=2}", "LSelect32_v1", "Const2[0x1]_v1", "LIsEOS1_v1{ORD=2}",
                                                       "LEmitNext[H=1]_v1"}  # G1 (v0.7 A11): the EOS set ordinary; the members reading it carry {ORD=2}
    # the LSelect32 guard rows are of two kinds: s=<s>/tok_in (Active for s < a_r, bot after) and slot=<t>/out (Active for t <= a_r, bot after)
    sel = [r for r in rows if r["guard"] and r["spec"] == "LSelect32_v1"]
    assert sorted(r["tag"] for r in sel) == sorted(["A"] * (A_R - 1) + ["⊥"] * (CAP - A_R) + ["A"] * A_R + ["⊥"] * (CAP - A_R))
    assert ck["R5_interface"]["bot_events"] == CAP - A_R
    # R9 Continuation_v1: the Program's controller bits equal the rule's recurrence over the transcript; the designated output serves a_r
    r9 = ck["R9_continuation_v1"]
    # rev S-r17-29 n1 / n2: no capture (no served list of record) and an EOS set given at check time -> R9 is None BY NAME, never True;
    # the Program-side facts (bits == rule) are still reported
    assert r9["ok"] is None and r9["of_record"] is False and "not of record" in r9["why"] and "(n1)" in r9["why"] and "(n2)" in r9["why"]
    assert r9["program_bits_equal_rule"] == {"emit": True, "is_eos": True, "run": True, "out": True} and r9["A"] == 0
    assert r9["executed_by_run"] == list(range(1, A_R + 1)) and r9["lag_slots"] == [] and r9["designated_output_served_count"] == A_R == r9["raw_committed_active_count"]
    assert rep["continuation"]["bits"] == {"emit": [1] * A_R + [0] * (CAP - A_R), "run": [1] * A_R + [0] * (CAP - A_R)}
    assert rep["ok"] is True  # None is not False: the record verdict is untouched by an R9 that is merely not of record


def test_no_eos_record_runs_to_the_cap_with_no_bot(desc):
    rep = _check(desc, transcript=Y_RAW, eos=(63,), steps=range(CAP))
    ck = rep["checks"]
    assert rep["ok"] and ck["R2_activity"]["a_r"] == CAP and not ck["R2_activity"]["stopped_by_eos"]
    assert ck["R4_assignment"]["bot_leaves_total"] == 0 and ck["R5_interface"]["bot_events"] == 0


def test_declared_inactivity_executed_is_caught_by_the_executed_set_and_the_other_candidate_is_reported(desc):
    rep = _check(desc, steps=range(A_R + 1))  # the map commits step a_r: what a vLLM bi-eager engine does (exec M-0140)
    r3 = rep["checks"]["R3_executed_set"]
    assert not rep["ok"] and r3["ok"] is False and r3["declared_lag"] == 0
    assert r3["declared_inactivity_executed"] == [A_R]
    assert r3["matches"] == {"0": False, "1": True}  # LAG=1's prediction is what this record shows (DP-pad-02)
    assert r3["expected_by_lag"] == {"0": list(range(A_R)), "1": list(range(A_R + 1))}
    assert rep["checks"]["R2_activity"]["ok"] and rep["checks"]["R4_assignment"]["ok"]  # the other checks are untouched
    # the positive record under LAG=0 matches LAG=0 only; at the cap (no EOS) both predict the same set
    assert _check(desc)["checks"]["R3_executed_set"]["matches"] == {"0": True, "1": False}
    assert _check(desc, transcript=Y_RAW, eos=(63,), steps=range(CAP))["checks"]["R3_executed_set"]["matches"] == {"0": True, "1": True}
    with pytest.raises(ValueError, match="only .* is implemented"):
        _check(desc, guard="out")
    with pytest.raises(ValueError, match="LAG=2"):
        _check(desc, lag=2)


def test_lag1_record_the_engine_of_record_executes_the_placeholder_step(desc):
    """DP-pad-02 option (b) as a labelled candidate: under LAG=1 the honest record has a_r + 1 executed events -- the placeholder step
    ran on the EOS token and its sample (Y_RAW[a_r], the unlifted Program's own y_{a_r}) is committed but not part of the transcript."""
    served = Y_RAW[:A_R + 1]  # 4 transcript events + the placeholder's discarded sample
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1)
    ck = rep["checks"]
    assert rep["ok"], {k: v for k, v in ck.items() if v["ok"] is False}
    assert ck["R1_population"]["guard_gates"] == 2 * (CAP - 2) + 1 + 2 + 3 * CAP  # step 1 unguarded under LAG=1; + the controller (R17-32); + enable/one
    assert ck["R2_activity"]["a_r"] == A_R and ck["R2_activity"]["executed_expected"] == A_R + 1 and ck["R2_activity"]["placeholder_sample"] == Y_RAW[A_R]
    assert ck["R3_executed_set"]["matches"] == {"0": False, "1": True}
    r4 = ck["R4_assignment"]
    assert r4["active_events"] == A_R + 1 and set(r4["bot_leaves_by_event"]) == {str(s) for s in range(A_R + 1, CAP)}
    assert ck["R5_interface"]["bot_events"] == CAP - A_R - 1
    # the same record read under LAG=0 is 'execution past EOS' (the placeholder is not modelled), and the LAG=0 record under LAG=1 is
    # 'declared activity not executed' (the placeholder step is declared): the lag is a fact of the profile, not a free parameter
    bad = _check(desc, transcript=served, steps=range(A_R + 1), lag=0)
    assert not bad["ok"] and "execution past EOS" in bad["checks"]["R2_activity"]["why"]
    bad = _check(desc, transcript=SERVED, steps=range(A_R), lag=1)
    assert not bad["ok"] and "declared activity not executed" in bad["checks"]["R2_activity"]["why"]


def _w33(x):
    return format(int(x), "033b")


def test_r8_three_vus_lists_the_words_of_an_active_a_transition_and_an_inactive_vu_replayed_by_the_lifted_evaluators(desc):
    """SPEC v0.5.1 A7 (c): one VU of each kind with its words LISTED, the guard chain replayed by `LStepLive`/`LSelect`'s own evaluators on
    the committed transcript (independently of R4's tag propagation, and agreeing with it)."""
    bot = _w33(L.enc_bot(32))
    # LAG 0: the EOS-sampling step a_r - 1 is the last active; step a_r is the first inactive
    rep = _check(desc)
    r8, tv = rep["checks"]["R8_three_vus"], rep["three_vus"]
    assert r8["ok"] and r8["guard_replay_agrees"] and r8["replayed_by"][:2] == ["LStepLive1_v1", "LSelect32_v1"]
    assert tv["active"]["step"] == A_R - 1 and tv["active"]["ok"] and tv["active"]["body"]["tags"] == ["A"]
    g = tv["active"]["guard"]
    assert g["guarded"] and g["gating_token"] == f"out[{A_R - 2}]" and g["live"] == "01" and g["tok_in"] == _w33(Y_RAW[A_R - 2]) and g["out"] == _w33(EOS[0])
    assert all(r["form"].startswith("0‖v") and "words" not in r for r in tv["active"]["body"]["rows"])
    tr = tv["transition"]
    assert tr["from_step"] == A_R - 1 and tr["to_step"] == A_R and tr["ok"]
    fi = tr["first_inactive"]
    assert fi["guard"]["g"] == _w33(EOS[0]) and fi["guard"]["live"] == "00" and fi["guard"]["live_is"] == "Active(0)" and fi["guard"]["tok_in"] == bot == fi["guard"]["out"]
    assert fi["body"]["tags"] == ["⊥"] and all(r["words"] == {str(w): "1" + "0" * (w - 1) for w in r["w_lifted"]} for r in fi["body"]["rows"])
    assert tr["live_first_inactive_is_active_0"] and tr["tok_in_first_inactive_is_bot"] and tr["every_consumer_bot"]
    assert tv["inactive"]["step"] == CAP - 1 and tv["inactive"]["ok"] and tv["inactive"]["guard"]["g"] == bot and tv["inactive"]["guard"]["live"] == "00"
    assert tv["interface_out"] == [_w33(t) for t in SERVED] + [bot] * (CAP - A_R)
    # R17-32: the Continuation_v1 controller replayed by ITS evaluators (LIsEOS1_v1, LEmitNext[H=1]_v1) agrees with the closure; the three VUs
    # ARE the (e, r) pairs (1,1) / (0,0) -- under LAG 0 there is no (0,1) slot; the stop slot emits the EOS, the first inactive designates bot
    assert r8["controller_replay_agrees"] and r8["replayed_by"][3:] == ["LIsEOS1_v1", "LEmitNext[H=1]_v1"]
    assert tv["active"]["controller"]["pair_emit_run"] == "(1, 1)" and tv["active"]["controller"]["kind"] == "active" and tv["active"]["controller"]["is_eos"] == "01"
    assert tv["active"]["controller"]["out_is"] == f"Active({EOS[0]})" and r8["active_controller"] == "(1, 1)"
    assert tr["controller"]["lag_slot"] is None and tr["stop_slot_is_1_1_eos_emitted"] and tr["lag_slot_is_0_1_out_bot"] and tr["first_inactive_is_0_0_out_bot"]
    assert tr["controller"]["first_inactive_slot"]["pair_emit_run"] == "(0, 0)" and tr["controller"]["first_inactive_slot"]["out_is"] == "⊥"
    assert tv["inactive"]["controller"]["pair_emit_run"] == "(0, 0)" and tv["inactive"]["controller"]["is_eos"] == "00" and r8["inactive_controller"] == "(0, 0)"
    assert tv["designated_out"] == [_w33(t) for t in SERVED] + [bot] * (CAP - A_R) and tv["controller_pairs"] == ["(1, 1)"] * A_R + ["(0, 0)"] * (CAP - A_R)
    # LAG 1: the placeholder step a_r is fed Active(EOS) and is ACTIVE (its out is the committed discarded sample); a_r + 1 is the first inactive
    served = Y_RAW[:A_R + 1]
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1)
    r8, tv = rep["checks"]["R8_three_vus"], rep["three_vus"]
    assert r8["ok"] and r8["guard_replay_agrees"] and r8["controller_replay_agrees"]
    tr = tv["transition"]
    assert tr["from_step"] == A_R and tr["to_step"] == A_R + 1 and tr["tok_in_a_r_is_active_eos"] and tr["ok"]
    # the lag slot a_r + 1 is (0, 1): run, not emitted -- its real committed token is in interface_out and NOT in designated_out
    lagc = tr["controller"]["lag_slot"]
    assert lagc["slot"] == A_R + 1 and lagc["pair_emit_run"] == "(0, 1)" and lagc["kind"] == "lag" and lagc["out_is"] == "⊥"
    assert tr["controller"]["stop_slot"]["pair_emit_run"] == "(1, 1)" and tr["controller"]["stop_slot"]["is_eos"] == "01"
    assert tv["interface_out"][A_R] == _w33(Y_RAW[A_R]) and tv["designated_out"][A_R] == bot
    assert tv["designated_out"] == [_w33(t) for t in Y_RAW[:A_R]] + [bot] * (CAP - A_R)
    assert tv["controller_pairs"] == ["(1, 1)"] * A_R + ["(0, 1)"] + ["(0, 0)"] * (CAP - A_R - 1)
    la = tr["last_active"]["guard"]
    assert la["gating_token"] == f"tok_in[{A_R - 1}]" and la["tok_in"] == _w33(EOS[0]) and la["tok_in_is"] == f"Active({EOS[0]})" and la["out"] == _w33(Y_RAW[A_R])
    assert tr["last_active"]["body"]["tags"] == ["A"] and tr["first_inactive"]["body"]["tags"] == ["⊥"]
    assert tr["first_inactive"]["guard"]["g"] == _w33(EOS[0]) and tr["first_inactive"]["guard"]["tok_in"] == bot
    assert tv["interface_out"] == [_w33(t) for t in served] + [bot] * (CAP - A_R - 1)
    # a length stop has no transition and no inactive VU: they are None by name, the active VU alone decides
    rep = _check(desc, transcript=Y_RAW, eos=(63,), steps=range(CAP))
    r8, tv = rep["checks"]["R8_three_vus"], rep["three_vus"]
    assert r8["ok"] and r8["transition"] is None and r8["inactive_step"] is None and "length stop" in r8["note"]
    assert tv["active"]["step"] == CAP - 1 and tv["active"]["ok"] and bot not in tv["interface_out"]


def test_declared_activity_not_committed_is_caught(desc):
    rep = _check(desc, steps=[0, 1, 3])
    assert not rep["ok"] and rep["checks"]["R3_executed_set"]["declared_activity_not_committed"] == [2]


def test_execution_past_eos_and_a_short_transcript_fail_the_activity_check(desc):
    rep = _check(desc, transcript=Y_RAW[:A_R + 2], steps=range(A_R + 2))  # served past the EOS
    assert not rep["ok"] and rep["checks"]["R2_activity"]["ok"] is False and "execution past EOS" in rep["checks"]["R2_activity"]["why"]
    rep = _check(desc, transcript=Y_RAW[:A_R - 1], steps=range(A_R - 1))  # stopped before its EOS
    assert not rep["ok"] and "declared activity not executed" in rep["checks"]["R2_activity"]["why"]
    rep = _check(desc, transcript=Y_RAW[:2] + [L.enc_bot(32)] + [17], steps=range(A_R))  # a bot word inside the served prefix
    assert not rep["ok"] and "BoundaryFailure" in rep["checks"]["R2_activity"]["why"]
    rep = _check(desc, transcript=Y_RAW + [26], steps=range(CAP + 1))  # more tokens than the cap
    assert not rep["ok"] and "exceed the cap" in rep["checks"]["R2_activity"]["why"]


def test_a_wrong_cap_is_a_population_failure_not_a_rebuild(desc):
    rep = _check(desc, cap=CAP - 1)
    assert not rep["ok"] and rep["checks"]["R1_population"]["ok"] is False and "declares cap" in rep["checks"]["R1_population"]["why"]
    assert "R2_activity" not in rep["checks"]  # nothing downstream is evaluated on a Program that is not the manifest's


def test_closure_reads_the_guards_by_their_spec_functions_and_a_wrong_propagation_is_visible(desc):
    from verity.ir.defs import REGISTRY
    from verity_vllm.ir import workload as W
    comp = W.request_component(desc, REGISTRY, None)
    lifted = L.lift_request_component(comp.fn, EOS, max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    cl = R.activity_closure(lifted, SERVED, EOS)
    assert cl.guard_live == [1, 1, 1, 0, 0, 0, 0, 0] and cl.event_tag == ["A"] * A_R + ["⊥"] * (CAP - A_R)
    assert cl.event_token[:A_R] == SERVED and cl.event_token[A_R:] == [None] * (CAP - A_R)
    # an injected wrong propagation: pretend the EOS set does not contain the committed EOS -> the closure keeps every event Active,
    # and then it needs a committed token for event 4 that the record does not have: a named ServedPrefixFailure, never a fill
    with pytest.raises(L.ServedPrefixFailure, match="record has only 4 committed tokens"):
        R.activity_closure(lifted, SERVED, (63,))
    # the dense rule's mixed-operand bookkeeping: on this fixture a mixed node (Active cache rows + a bot row) appears only in bot steps
    assert cl.mixed_nodes and all(e >= A_R for _, _, e in cl.mixed_nodes)


def test_node_reads_is_run_level(desc):
    from verity.ir.refs import Affine, Concat, Explicit, Strided
    assert R.node_reads(Affine("p", 1, 0, 10**9)) == set() and R.node_reads(Affine("n", 3, 0, 10**9)) == {3}
    assert R.node_reads(Concat([Affine("n", 1, 0, 2), Strided("n", 2, 0, 1, 1, 1, 4), Explicit((("p", 0, 0), ("n", 5, 1)))])) == {1, 2, 5}


# --- the capture side (Match's match/capture/: RunHeader, StepStart rows, SampleEnd ids, tokens.json's measured lag) -------------------

def _capture(tmp_path, samples, *, lag, eos_at, header=None, extra_steps=(), placeholder_input=None, measured=None, rid="r0", served=None):
    """A synthetic capture dir: `samples[s]` is the sampled id of engine step s (the request occupies one row per decode step, LP rows at
    step 0); the placeholder step (index eos_at + 1, present when lag == 1) is fed `placeholder_input` (default: the EOS it follows)."""
    import gzip as _gz
    d = tmp_path / "capture"
    d.mkdir(parents=True)
    ev = []
    if header is not None:
        ev.append({"$e": "RunHeader", **header})
    steps = list(range(len(samples))) + list(extra_steps)
    for st in steps:
        rows = [[rid, i] for i in range(LP)] if st == 0 else [[rid, LP + st - 1]]
        fed = list(range(LP)) if st == 0 else [samples[st - 1] if st - 1 < len(samples) else 0]
        if st == eos_at + 1 and placeholder_input is not None:
            fed = [placeholder_input]
        ev.append({"$e": "StepStart", "step": st, "rows": rows, "input_token_ids": fed})
        ev.append({"$e": "SampleEnd", "step": st, "sampled_token_ids": [samples[st] if st < len(samples) else 0]})
    with _gz.open(d / "log.jsonl.gz", "wt") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    tk = {"timing": {"output_lag_measured": lag if measured is None else measured, "output_lag_source": "measured",
                     "engine_steps": len(samples) - lag, "step_iterations": len(steps)}}
    if served is not None:  # the driver's served list (tokens.json requests[].generated_token_ids): what rev's (b) is checked against
        tk["requests"] = [{"request_id": rid, "generated_token_ids": [int(t) for t in served]}]
    (d / "tokens.json").write_text(json.dumps(tk))
    return d


def test_capture_side_the_engines_step_log_confirms_the_lag1_prediction_and_the_placeholder_is_fed_the_eos(desc, tmp_path):
    """R3c/R3d/R0b/R2b on a synthetic Match capture of the lag-1 record: the engine ran a_r + 1 steps (StepStart rows), step a_r was fed
    the committed EOS and sampled the placeholder id, the harness measured lag 1, the SampleEnd array IS the transcript."""
    served = Y_RAW[:A_R + 1]
    hdr = _header(1)
    cap_dir = _capture(tmp_path, served, lag=1, eos_at=A_R - 1, header=hdr)
    cd = R.read_capture(cap_dir)
    assert cd["header"]["profile"]["engine_kwargs"]["async_scheduling"] is True and cd["lag_measured"] == 1
    cr = R.capture_request(cd, "r0")
    assert cr["executed"] == list(range(A_R + 1)) and [cr["sampled"][s] for s in cr["executed"]] == list(served) and cr["inputs"][A_R] == EOS[0]
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd)
    ck = rep["checks"]
    assert rep["ok"] is True
    assert ck["R3c_capture_executed_set"]["ok"] and ck["R3c_capture_executed_set"]["matches"] == {"0": False, "1": True}
    assert ck["R3d_placeholder_step"] == {"ok": True, "step": A_R, "input_token": EOS[0], "committed_eos": EOS[0], "placeholder_sample": Y_RAW[A_R],
                                          "transcript_at_a_r": Y_RAW[A_R]}
    assert ck["R0b_lag_measured_consistency"]["ok"] and ck["R2b_capture_samples"]["ok"]
    # the same capture WITHOUT a binding map: the capture confirms the engine's executed set, yet `ok` stays None -- R3c is evidence
    # beside R3, not a substitute (F-pad-17)
    rep = _check(desc, transcript=served, lag=1, bmap=None, capture=cd)
    assert rep["ok"] is None and rep["checks"]["R3_executed_set"]["ok"] is None and rep["checks"]["R3c_capture_executed_set"]["ok"] is True


def test_r9_is_of_record_only_with_the_drivers_served_list_and_the_builds_pinned_eos_set_rev_s_r17_29_n1_n2(desc, tmp_path):
    """rev S-r17-29 (M-0540) n1: a served list DERIVED from the transcript makes R9 None, never True; n2: an EOS set given at check time
    (`--eos`) is not the Program's pinned constant -> None by name.  Of record = generated_token_ids in the capture's tokens.json AND
    `eos_source = EOS_SOURCE_OF_RECORD` (workload_program.json stop_rule.eos_token_ids).  A False stays False under derived inputs."""
    served = Y_RAW[:A_R + 1]  # LAG 1: the lag slot's discarded sample is committed, not served
    # (1) of record: the driver's served list is the emit prefix, the EOS set is the Build's -> True, and the wrong profile fails by name
    cd = R.read_capture(_capture(tmp_path / "a", served, lag=1, eos_at=A_R - 1, header=_header(1), served=Y_RAW[:A_R]))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd, eos_source=R.EOS_SOURCE_OF_RECORD, declared=(1, "lag.value"))
    r9 = rep["checks"]["R9_continuation_v1"]
    assert r9["ok"] is True and r9["of_record"] is True and r9["why"] is None and r9["eos_source"] == R.EOS_SOURCE_OF_RECORD
    assert r9["lag_slots"] == [A_R + 1] and r9["match_side_ok"] and r9["slot_map_ok"] and r9["wrong_profile_fails_by_name"] is True
    # revb S-r17b-20: n1 the synthetic capture attests nothing about prefill -> the two sub-checks are named VACUOUS, not passed;
    # R5 the RNG field: a_r + 1 draws, one after the served EOS at slot a_r + 1, pos = prompt rows (LP, from the capture's own StepStart) + a_r
    assert r9["chunked_prefill_attested"] == "absent" and len(r9["slot_map_vacuous_by_absence"]) == 2
    assert r9["rng_advances_iff_run"] == {"draws": A_R + 1, "draws_after_served_eos": 1, "lag_draw_slot": A_R + 1, "lag_draw_pos": LP - 1 + A_R}  # the ledger's pos (stoch M-0749)
    assert rep["continuation"]["rng"]["statement"] == K.RNG_STATEMENT and "StepStart rows" in rep["continuation"]["rng"]["prompt_len_source"]
    assert rep["continuation"]["served_source"].startswith("tokens.json requests[].generated_token_ids")
    assert rep["continuation"]["match_side"]["checks"]["c_lag_token_not_served"]["lag_tokens"] == [Y_RAW[A_R]]
    # (2) n1: the same capture WITHOUT generated_token_ids -> the served list is derived -> None by name, the checks still computed
    cd = R.read_capture(_capture(tmp_path / "b", served, lag=1, eos_at=A_R - 1, header=_header(1)))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd, eos_source=R.EOS_SOURCE_OF_RECORD, declared=(1, "lag.value"))
    r9 = rep["checks"]["R9_continuation_v1"]
    assert r9["ok"] is None and r9["of_record"] is False and "(n1)" in r9["why"] and "(n2)" not in r9["why"] and r9["match_side_ok"] is True
    assert rep["continuation"]["served_source"].startswith("the transcript's first a_r tokens") and rep["ok"] is True
    # (3) n2: the driver's list present but the EOS set given at check time -> None by name
    cd = R.read_capture(_capture(tmp_path / "c", served, lag=1, eos_at=A_R - 1, header=_header(1), served=Y_RAW[:A_R]))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd, declared=(1, "lag.value"))  # eos_source defaults to EOS_SOURCE_GIVEN
    r9 = rep["checks"]["R9_continuation_v1"]
    assert r9["ok"] is None and "(n2)" in r9["why"] and "(n1)" not in r9["why"] and r9["eos_source"] == R.EOS_SOURCE_GIVEN
    # (4) a lag token SERVED (appended to the driver's list) is False -- of record or not, a named failure stays False
    cd = R.read_capture(_capture(tmp_path / "d", served, lag=1, eos_at=A_R - 1, header=_header(1), served=Y_RAW[:A_R + 1]))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd, eos_source=R.EOS_SOURCE_OF_RECORD, declared=(1, "lag.value"))
    r9 = rep["checks"]["R9_continuation_v1"]
    assert r9["ok"] is False and r9["match_side_ok"] is False and rep["ok"] is False and "R9_continuation_v1" in rep["failed_checks"]
    assert {"b_served_equals_emit_tokens", "c_lag_token_not_served"} <= set(rep["continuation"]["match_side"]["failed"])
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd)  # the same under a check-time EOS set: still False
    assert rep["checks"]["R9_continuation_v1"]["ok"] is False
    # (5) rev M-0677: NO Build declaration checked (a synthetic record, or a pre-R17-1 Program declaring no LAG) -> A is the attestation's,
    # not the Build's -> R9 None by name even with the driver's list and the pinned EOS set; lag_program_source says 'attested'
    cd = R.read_capture(_capture(tmp_path / "e", served, lag=1, eos_at=A_R - 1, header=_header(1), served=Y_RAW[:A_R]))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd, eos_source=R.EOS_SOURCE_OF_RECORD)
    r9 = rep["checks"]["R9_continuation_v1"]
    assert r9["ok"] is None and r9["of_record"] is False and "no Build declaration checked" in r9["why"] and "M-0677" in r9["why"]
    assert rep["lag_program_source"].startswith("attested (") and rep["lag_program"] == rep["lag_attested"] == 1


def test_capture_side_negatives_an_extra_engine_step_a_foreign_placeholder_input_and_a_contradicting_measured_lag(desc, tmp_path):
    served = Y_RAW[:A_R + 1]
    # (i) the engine ran one step MORE than the declared lag-1 set: R3c fails and names it (the Commit's R3 would too, if it committed it)
    cd = R.read_capture(_capture(tmp_path / "a", served, lag=1, eos_at=A_R - 1, header=_header(1), extra_steps=(A_R + 1,)))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd)
    r3c = rep["checks"]["R3c_capture_executed_set"]
    assert r3c["ok"] is False and r3c["engine_steps"] == list(range(A_R + 2)) and r3c["matches"] == {"0": False, "1": False}
    assert rep["ok"] is False and "R3c_capture_executed_set" in rep["failed_checks"]  # F-pad-22: an evaluated capture check that fails vetoes `ok`
    # (ii) the placeholder step was fed something other than the committed EOS: DP-pad-02's model of the engine does not hold -> R3d FAIL
    cd = R.read_capture(_capture(tmp_path / "b", served, lag=1, eos_at=A_R - 1, header=_header(1), placeholder_input=Y_RAW[0]))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd)
    assert rep["checks"]["R3d_placeholder_step"]["ok"] is False and rep["checks"]["R3d_placeholder_step"]["input_token"] == Y_RAW[0]
    # (iii) the harness MEASURED lag 0 while the header attests lag 1: R0b fails by name (two facts of the record disagree)
    cd = R.read_capture(_capture(tmp_path / "c", served, lag=1, eos_at=A_R - 1, header=_header(1), measured=0))
    rep = _check(desc, transcript=served, steps=range(A_R + 1), lag=1, capture=cd)
    assert rep["checks"]["R0b_lag_measured_consistency"]["ok"] is False and rep["checks"]["R0b_lag_measured_consistency"]["measured"] == 0


def test_cli_a_capture_without_the_attested_field_is_refused_and_runs_only_with_an_explicit_lag_as_a_named_failure(desc, tmp_path):
    """The real fixture's situation (exec M-0140, before RULING 21's attestation unit lands): the RunHeader has no async_scheduling."""
    served = Y_RAW[:A_R + 1]
    hdr = {"profile": {"engine_kwargs": {"enforce_eager": True}}}
    cap_dir = _capture(tmp_path, served, lag=1, eos_at=A_R - 1, header=hdr)
    br = tmp_path / "build_request"
    br.mkdir()
    import gzip as _gz
    with _gz.open(br / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    out = tmp_path / "rep.json"
    base = ["--build-request", str(br), "--capture", str(cap_dir), "--eos", str(EOS[0]), "--cap", str(CAP), "--boundary-prefix", BOUNDARY, "--out", str(out)]
    with pytest.raises(SystemExit):  # argparse error: the field is absent and no --lag was declared -> nothing runs, nothing is defaulted
        R.main(base)
    assert R.main(base + ["--lag", "1"]) == 1  # runs, but the report is a FAIL by name (R0)
    rep = json.loads(out.read_text())
    assert rep["ok"] is False and rep["checks"]["R0_lag_attestation"]["ok"] is False and rep["lag_source"].startswith("UNATTESTED")
    assert rep["transcript_source"].startswith("match/capture/log.jsonl.gz SampleEnd")
    ck = rep["checks"]
    assert ck["R1_population"]["ok"] and ck["R2_activity"]["ok"] and ck["R4_assignment"]["ok"] and ck["R5_interface"]["ok"]
    assert ck["R3_executed_set"]["ok"] is None and ck["R3c_capture_executed_set"]["matches"] == {"0": False, "1": True} and ck["R3d_placeholder_step"]["ok"]
    # with the field attested, the CLI reads it and refuses a contradicting --lag
    cap_dir2 = _capture(tmp_path / "x", served, lag=1, eos_at=A_R - 1, header=_header(1))
    with pytest.raises(SystemExit):
        R.main(["--build-request", str(br), "--capture", str(cap_dir2), "--eos", str(EOS[0]), "--cap", str(CAP), "--boundary-prefix", BOUNDARY,
                "--out", str(out), "--lag", "0"])


def test_the_engine_request_id_and_the_commits_positional_id_are_two_named_spaces_nothing_is_inferred(desc):
    """binding-map/1 keys identities by the Commit's positional `r<k>`; the capture keys rows by the engine's request uuid (the real cp-sys
    record: `5b7fee10-…` vs `r0`).  The caller names both; a wrong Commit id is an R3 FAIL that lists the ids the map does carry."""
    uuid = "5b7fee10-6a81-52b7-b730-133d6cc768da"
    lag, src = L.lag_from_header(_header(0))
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R), rid="r0"), request_id=uuid, commit_request_id="r0",
                         boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    assert rep["ok"] is True and rep["request_id"] == uuid and rep["commit_request_id"] == "r0"
    assert rep["checks"]["R3_executed_set"]["binding_map_request_ids"] == ["r0"]
    # the uuid used as the Commit id finds nothing: declared activity not committed, by name, with the map's own ids in the report
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R), rid="r0"), request_id=uuid, boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    r3 = rep["checks"]["R3_executed_set"]
    assert rep["ok"] is False and r3["ok"] is False and r3["commit_request_id"] == uuid and r3["committed_steps_n"] == 0
    assert r3["declared_activity_not_committed"] == list(range(A_R)) and r3["binding_map_request_ids"] == ["r0"]


def test_f_pad_06b_the_report_carries_the_lifted_layout_size_a_binding_map_must_record(desc):
    """Σ leaf_words(w) over every leaf of every event (Active and ⊥ alike): 2 words for a 32-bit leaf, 1 otherwise -- the size by which a
    one-word-per-leaf reader of a lifted store is refused (padrev's legacy-reader attack)."""
    rep = _check(desc)
    r = rep["checks"]["R4b_lifted_layout_words"]
    assert r["ok"] is True and set(r["leaves_by_event"]) == {str(s) for s in range(CAP)}
    # every event has at least its token leaf (32-bit) -> lifted words > one-word count in every event; the excess counts the 32-bit leaves
    lifted = L.lift_request_component(W.request_component(desc, REGISTRY, None).fn, EOS, max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)
    cl = R.activity_closure(lifted, SERVED, EOS)
    for s in range(CAP):
        w32 = sum(1 for k, n in enumerate(lifted.body.nodes) if cl.step_of_node[k] == s and k not in set(lifted.guard_nodes)
                  for w in R._leaf_widths(n.out) if w == 33)
        assert r["lifted_words_by_event"][str(s)] == r["one_word_per_leaf_by_event"][str(s)] + w32 and w32 >= 1
    assert r["lifted_words_total"] == sum(r["lifted_words_by_event"].values()) > r["one_word_per_leaf_total"] == r["leaves_total"]
    # the ⊥ words of R4 are a subset of the lifted words: every ⊥ event's bot_words == its lifted words (the whole event is ⊥ there)
    r4 = rep["checks"]["R4_assignment"]
    for s in range(A_R, CAP):
        assert r4["bot_words_by_event"][str(s)] == r["lifted_words_by_event"][str(s)]


# --- MIXED BATCH record half: two requests, synthetic record (TINY fixture) ------------------------------------------------------------

D_MAX_B = 3
CAP_B = D_MAX_B + 1
UUID_A, UUID_B = "aaaaaaaa-0000-5000-8000-000000000001", "bbbbbbbb-0000-5000-8000-000000000002"
B_TOKENS = [1, 2, 3, 4]  # B never emits the EOS 17: runs to ITS cap 4


@pytest.fixture(scope="module")
def desc_b():
    return encode_program(Program(b1.serve_v2(TINY, LP, D_MAX_B)))


def _bmap2(a_steps, b_steps):
    m = _bmap(a_steps, rid="r0")
    m["entries"] += _bmap(b_steps, rid="r1")["entries"]
    return m


def _capture2(tmp_path, reqs, *, lag, header, facts=False):
    """reqs: (uuid, samples, arrive) -- the request occupies LP rows at its first step and one row per later step; sampled ids in row order.
    `facts`: write the record's own pairing facts (tokens.json requests: prompt_len / arrive_step per uuid) -- W0's fact source (F-pad-25);
    the default False leaves the record without one (a pairing taken by hand; W0 None) -- padrev's suites import this helper and write their
    own fact sources beside it, so the default must not add one."""
    import gzip as _gz
    d = tmp_path / "capture"
    d.mkdir(parents=True)
    ev = [{"$e": "RunHeader", **header}]
    horizon = max(a + len(smp) for _, smp, a in reqs)
    for st in range(horizon):
        rows, fed, sampled = [], [], []
        for rid, smp, a in reqs:
            s = st - a
            if not (0 <= s < len(smp)):
                continue
            if s == 0:
                rows += [[rid, i] for i in range(LP)]
                fed += list(range(LP))
            else:
                rows.append([rid, LP + s - 1])
                fed.append(smp[s - 1])
            sampled.append(smp[s])
        if rows:
            ev.append({"$e": "StepStart", "step": st, "rows": rows, "input_token_ids": fed})
            ev.append({"$e": "SampleEnd", "step": st, "sampled_token_ids": sampled})
    with _gz.open(d / "log.jsonl.gz", "wt") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    tk = {"timing": {"output_lag_measured": lag, "output_lag_source": "measured"}}
    if facts:
        tk["requests"] = [{"request_id": rid, "prompt_len": LP, "arrive_step": a} for rid, _, a in reqs]
    (d / "tokens.json").write_text(json.dumps(tk))
    return d


def _reqs(desc, desc_b, a_tr, b_tr):
    from verity_vllm.registry.b1 import weights_type
    from verity.ir.types import Array, Value
    PARAMS = [("prompt", repr(Array(LP, Value(32)))), ("weights", repr(weights_type(TINY)))]
    return [{"request_id": UUID_A, "commit_request_id": "r0", "desc": desc, "cap": CAP, "arrive": 0, "transcript": a_tr, "params": PARAMS, "prompt_len": LP},
            {"request_id": UUID_B, "commit_request_id": "r1", "desc": desc_b, "cap": CAP_B, "arrive": 2, "transcript": b_tr, "params": PARAMS, "prompt_len": LP}]


@pytest.mark.parametrize("lag", [0, 1])
def test_mixed_record_a_stops_early_b_runs_to_its_cap_the_engine_the_commit_and_the_program_agree_step_by_step(desc, desc_b, tmp_path, lag):
    a_tr = Y_RAW[:A_R + lag]  # A: EOS at event 3, the placeholder's sample under LAG=1
    lg, src = L.lag_from_header(_header(lag))
    cd = R.read_capture(_capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=lag, header=_header(lag), facts=True))
    bm = _bmap2(range(A_R + lag), range(2, 2 + CAP_B))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is True, {k: v for k, v in rep["checks"].items() if v["ok"] is False}
    assert [p["ok"] for p in rep["requests"]] == [True, True]
    w1 = rep["checks"]["W1_population"]
    assert w1["events"] == CAP + CAP_B and w1["guard_gates"] == L.continuation_gate_count(CAP, 1, lag) + L.continuation_gate_count(CAP_B, 1, lag)
    w2 = rep["checks"]["W2_chronology"]
    # n_live_t: A alone at t<2, both until A stops at 4(+lag), B alone until 6, nobody after (the horizon runs to A's cap 9)
    live_a = [1] * (A_R + lag) + [0] * (CAP - A_R - lag)
    live_b = [0, 0] + [1] * CAP_B + [0] * (CAP - 2 - CAP_B)
    assert w2["n_live_by_step"] == {str(t): a + b for t, (a, b) in enumerate(zip(live_a, live_b))}
    assert w2["engine_vs_program_mismatch"] == {} and w2["commit_vs_program_mismatch"] == {} and w2["engine_steps_outside_population"] == {}
    assert rep["checks"]["W3_no_crossing"]["ok"] and rep["checks"]["W3_no_crossing"]["a_r_per_request"] == {UUID_A: A_R, UUID_B: CAP_B}
    # F-pad-17 at the workload level: no binding map -> None; an unattested lag -> False
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is None and rep["checks"]["W2_chronology"]["commit_evaluated"] is False and rep["checks"]["W2_chronology"]["ok"] is True
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source="cli", bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is False and all(p["checks"]["R0_lag_attestation"]["ok"] is False for p in rep["requests"])


def test_mixed_record_negatives_a_kept_alive_by_the_engine_after_its_eos_and_a_commit_of_a_bot_step_are_caught_at_both_levels(desc, desc_b, tmp_path):
    a_tr = Y_RAW[:A_R]
    lg, src = L.lag_from_header(_header(0))
    # (i) the ENGINE ran A at steps 4 and 5 (while B was live -- a cross-request keep-alive): the wrong rows carry A's tokens y_4, y_5
    bad_a = Y_RAW[:A_R + 2]
    cd = R.read_capture(_capture2(tmp_path / "i", [(UUID_A, bad_a, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True))
    bm = _bmap2(range(A_R), range(2, 2 + CAP_B))
    # the request's OWN transcript is what the engine sampled (6 events): its R2 fails 'execution past EOS' ...
    rep = R.check_workload_record(_reqs(desc, desc_b, None, None), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is False and "execution past EOS" in rep["requests"][0]["checks"]["R2_activity"]["why"]
    # ... and with the COMMITTED transcript (4 events) given, the engine rows contradict the Program at steps 4 and 5, naming A, at the workload level
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    w2 = rep["checks"]["W2_chronology"]
    assert rep["ok"] is False and w2["ok"] is False and w2["engine_vs_program_mismatch"] == {"4": [UUID_A], "5": [UUID_A]} and w2["commit_vs_program_mismatch"] == {}
    assert rep["requests"][0]["checks"]["R3c_capture_executed_set"]["ok"] is False  # the per-request view sees it too
    # (ii) the COMMIT committed A at step 4 (a ⊥ step of A's own guard) while the engine did not run it: caught by A's R3 and by W2's commit view
    cd = R.read_capture(_capture2(tmp_path / "ii", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True))
    bm = _bmap2(range(A_R + 1), range(2, 2 + CAP_B))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    w2 = rep["checks"]["W2_chronology"]
    assert rep["ok"] is False and w2["commit_vs_program_mismatch"] == {"4": [UUID_A]} and w2["engine_vs_program_mismatch"] == {}
    assert rep["requests"][0]["checks"]["R3_executed_set"]["declared_inactivity_executed"] == [4]
    # (iii) B's row missing at step 3 (the engine skipped a live request): the Program says live, the engine has no row -> named at step 3
    cd = R.read_capture(_capture2(tmp_path / "iii", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True))
    import gzip as _gz
    p = tmp_path / "iii" / "capture" / "log.jsonl.gz"
    ev = [json.loads(l) for l in _gz.open(p, "rt")]
    for e in ev:
        if e["$e"] == "StepStart" and e["step"] == 3:
            e["rows"] = [r for r in e["rows"] if r[0] != UUID_B]
            e["input_token_ids"] = e["input_token_ids"][:len(e["rows"])]
        if e["$e"] == "SampleEnd" and e["step"] == 3:
            e["sampled_token_ids"] = e["sampled_token_ids"][:1]
    with _gz.open(p, "wt") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    cd = R.read_capture(tmp_path / "iii" / "capture")
    bm = _bmap2(range(A_R), range(2, 2 + CAP_B))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is False and rep["checks"]["W2_chronology"]["engine_vs_program_mismatch"] == {"3": [UUID_B]}


def test_f_pad_22_a_commit_the_engines_own_step_log_refutes_never_reads_ok_true(desc, tmp_path):
    """padrev M-0352: attestation true + a binding map committing exactly the prediction (R3 PASS) + the engine log missing the placeholder
    step (R3c FAIL) -> ok False, the failing checks named.  Any evaluated check that is False vetoes `ok`; None stays None."""
    served = Y_RAW[:A_R + 1]  # LAG=1: the placeholder's sample is the 5th event
    lag, src = L.lag_from_header(_header(1))
    good = R.read_capture(_capture(tmp_path / "good", served, lag=1, eos_at=A_R - 1, header=_header(1)))
    rep = R.check_record(desc, served, EOS, cap=CAP, bmap=_bmap(range(A_R + 1)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, capture=good)
    assert rep["ok"] is True and rep["failed_checks"] == []
    # the same Commit, the engine log without step 4 (the placeholder never ran): R3 still PASS, R3c/R3d/R2b FAIL -> ok False by name
    bad = R.read_capture(_capture(tmp_path / "bad", served[:A_R], lag=1, eos_at=A_R - 1, header=_header(1)))
    rep = R.check_record(desc, served, EOS, cap=CAP, bmap=_bmap(range(A_R + 1)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, capture=bad)
    assert rep["checks"]["R3_executed_set"]["ok"] is True and rep["checks"]["R3c_capture_executed_set"]["ok"] is False
    assert rep["ok"] is False and "R3c_capture_executed_set" in rep["failed_checks"]
    # without a binding map the same refuted capture is False too (a capture failure is never None), and a clean capture without a map stays None
    rep = R.check_record(desc, served, EOS, cap=CAP, boundary_prefix=BOUNDARY, lag=lag, lag_source=src, capture=bad)
    assert rep["ok"] is False
    rep = R.check_record(desc, served, EOS, cap=CAP, boundary_prefix=BOUNDARY, lag=lag, lag_source=src, capture=good)
    assert rep["ok"] is None and rep["failed_checks"] == []


def test_cli_build_workload_reads_the_rows_own_manifest_facts_and_checks_every_declared_request(desc, desc_b, tmp_path):
    """`--build-workload <row>`: requests, caps, arrivals, the r<k> -> engine-uuid map and the EOS set come from the Build's
    workload_program.json; each component from inputs.derived['LP,T'] rebased onto the row; params from instances.json.gz.  Nothing is
    inferred: a request without a declared engine id is refused by name."""
    import gzip as _gz
    from verity_vllm.registry.b1 import weights_type
    from verity.ir.types import Array
    row = tmp_path / "row"
    for name, d in (("build_request_LP32_T8", desc), ("build_request_LP32_T3", desc_b)):
        (row / name).mkdir(parents=True)
        with _gz.open(row / name / "descriptor.json.gz", "wt") as f:
            json.dump(d, f)
        with _gz.open(row / name / "instances.json.gz", "wt") as f:
            json.dump({"params": [["prompt", repr(Array(LP, Value(32)))], ["weights", repr(weights_type(TINY))]]}, f)
    (row / "build_workload").mkdir()
    # the Build's OWN facts of the composed workload (what a real workload_program.json records): the tool's recompose must reproduce them
    PARAMS = [("prompt", repr(Array(LP, Value(32)))), ("weights", repr(weights_type(TINY)))]
    comp_a, comp_b = W.request_component(desc, REGISTRY, PARAMS), W.request_component(desc_b, REGISTRY, PARAMS)
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, LP, CAP, 0, comp_a), W.WorkloadRequest("r1", 1, LP, CAP_B, 2, comp_b)], lag=0)  # R17-1: the lag static IS in the Build's digest
    wp = {"workload_digest": wl.digest, "gates": wl.program.gates, "population": {"component_gates": wl.fn.body.gates}, "manifest_digest": "md",
          "stop_rule": {"eos_token_ids": list(EOS), "eos_lag_rows": 0},  # R17-1: the Build DECLARES its lag (GP-01's field; moe's lag.value / stop_rule.lag also read)
          "requests": [{"request_id": "r0", "index": 0, "prompt_len": LP, "cap": CAP, "arrive_step": 0, "component": {"program_digest": comp_a.program_digest}},
                       {"request_id": "r1", "index": 1, "prompt_len": LP, "cap": CAP_B, "arrive_step": 2, "component": {"program_digest": comp_b.program_digest}}],
          "request_ids": {"workload_request_ids": {"r0": UUID_A, "r1": UUID_B}},
          "inputs": {"derived": {f"{LP},{CAP - 1}": f"/pod/path/{row.name}/build_request_LP32_T8", f"{LP},{CAP_B - 1}": f"/pod/path/{row.name}/build_request_LP32_T3"}}}
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp))
    a_tr = Y_RAW[:A_R]
    cap_dir = _capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True)
    commit = tmp_path / "commit"
    commit.mkdir()
    with _gz.open(commit / "binding_map_p0.json.gz", "wt") as f:
        json.dump(_bmap2(range(A_R), range(2, 2 + CAP_B)), f)
    out = tmp_path / "rep.json"
    args = ["--build-workload", str(row), "--capture", str(cap_dir), "--commit", str(commit), "--boundary-prefix", BOUNDARY, "--out", str(out)]
    rc = R.main(args)
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["ok"] is True and [p["request_id"] for p in rep["requests"]] == [UUID_A, UUID_B]
    assert [p["commit_request_id"] for p in rep["requests"]] == ["r0", "r1"] and rep["row"]["stop_rule"]["eos_token_ids"] == list(EOS)
    assert rep["checks"]["W2_chronology"]["n_live_by_step"]["3"] == 2 and rep["checks"]["W1_population"]["events"] == CAP + CAP_B
    w1 = rep["checks"]["W1_population"]
    assert w1["pinned_to_row"] is True and w1["recomposed_workload_digest"] == wl.digest == w1["recorded_workload_digest"]
    assert w1["recomposed_gates"] == wl.program.gates and w1["recomposed_component_gates"] == wl.fn.body.gates
    assert {k: v["recomposed"] == v["recorded"] for k, v in w1["component_program_digests"].items()} == {"r0": True, "r1": True}
    # R17-1 (SPEC v0.5 A2.2): the row's DECLARED lag was read and equals the attestation; every report names the lifted rules version
    assert rep["lag_declared"] == 0 and rep["lag_declared_field"] == "stop_rule.eos_lag_rows" and rep["lag_declared_checked"] is True
    assert rep["rules_version"] == L.LIFTED_RULES_VERSION == "lifted-v1" and rep["record_verdict"]["lag_declared_equals_attested"] is True
    r0 = rep["requests"][0]["checks"]["R0_lag_attestation"]
    assert r0["ok"] is True and r0["declared_lag"] == 0 and r0["declared_equals_attested"] is True and r0["declared_checked"] is True
    # ... a Build that declared NO lag (a pre-R17-1 workload_program.json) is R0 False by name on every request, rc 1, never ok
    wp_nolag = json.loads(json.dumps(wp))
    del wp_nolag["stop_rule"]["eos_lag_rows"]
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp_nolag))
    assert R.main(args) == 1
    rep_nl = json.loads(out.read_text())
    assert rep_nl["ok"] is False and rep_nl["lag_declared"] is None and rep_nl["lag_declared_checked"] is True
    assert all(p["checks"]["R0_lag_attestation"]["ok"] is False and "declared NO lag" in p["checks"]["R0_lag_attestation"]["why"] for p in rep_nl["requests"])
    # ... and a Build that declared the OTHER lag (LAG=1 Program on an async_scheduling=False record) is R0 False by name: built for another profile
    wp_ol = json.loads(json.dumps(wp))
    wp_ol["lag"], wp_ol["stop_rule"]["lag"], wp_ol["stop_rule"]["eos_lag_rows"] = {"value": 1}, 1, 1  # moe's fields + GP-01's, consistent
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp_ol))
    assert R.main(args) == 1
    rep_ol = json.loads(out.read_text())
    assert rep_ol["lag_declared"] == 1 and rep_ol["lag_declared_field"] == "lag.value" and rep_ol["ok"] is False
    assert all("declared LAG=1" in p["checks"]["R0_lag_attestation"]["why"] for p in rep_ol["requests"])
    # ... and the Program of record is the BUILD's: Lift(P) is taken at the DECLARED lag (A fixed at Build, never re-fitted to the record's
    # profile), so beside R0 the record's early stop is named against the LAG=1 Program where it shows -- the missing lag step (R2: 'expected
    # a_r + 1 executed events under LAG=1'), exactly the shape exec's real wrong-profile record gave (r17_2208: GM-01 G1 + G6 by name)
    for p in rep_ol["requests"]:
        assert p["lag_attested"] == 0 and p["lag_program"] == 1 and p["lag_program_source"].startswith("declared (")
    early = next(p for p in rep_ol["requests"] if p["checks"]["R2_activity"].get("stopped_by_eos") is not False and p["checks"]["R2_activity"]["ok"] is False)
    assert "declared activity not executed" in early["checks"]["R2_activity"]["why"] and "under LAG=1" in early["checks"]["R2_activity"]["why"]
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp))
    # F-pad-26(a): the recompose must reproduce the row's OWN record -- a recorded workload_digest the recompose does not reproduce (the tool
    # composing a Program the Build never emitted, or a row whose record was edited) is W1 False by name, rc 1, ok False
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(dict(wp, workload_digest="0" * 64)))
    assert R.main(args) == 1
    rep = json.loads(out.read_text())
    assert rep["ok"] is False and rep["failed_checks"] == ["W1_population"] and "workload_digest" in rep["checks"]["W1_population"]["why"]
    # ... and a component whose recorded program_digest is another Program's (a served-shape derivation recorded under r1) names r1
    wp_bad = json.loads(json.dumps(wp))
    wp_bad["requests"][1]["component"]["program_digest"] = "1" * 64
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp_bad))
    assert R.main(args) == 1
    rep = json.loads(out.read_text())
    assert rep["ok"] is False and rep["checks"]["W1_population"]["why"].startswith("r1: component program_digest")
    # ... and recorded total gates that disagree with the recompose
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(dict(wp, gates=wl.program.gates + 1)))
    assert R.main(args) == 1 and "gates" in json.loads(out.read_text())["checks"]["W1_population"]["why"]
    # F-pad-26(b): a component the Build never declared in inputs.derived is NOT read by the dir-name convention (the dir exists!) -- refused
    wp_nd = json.loads(json.dumps(wp))
    del wp_nd["inputs"]["derived"][f"{LP},{CAP_B - 1}"]
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp_nd))
    assert (row / "build_request_LP32_T3" / "descriptor.json.gz").exists()
    with pytest.raises(SystemExit):
        R.main(args)
    # a request without a declared engine id: refused by name (argparse error -> SystemExit 2)
    del wp["request_ids"]["workload_request_ids"]["r1"]
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp))
    with pytest.raises(SystemExit):
        R.main(["--build-workload", str(row), "--capture", str(cap_dir), "--out", str(out)])


def test_r3a_names_which_program_the_commit_bound_a_served_shape_commit_is_executed_set_evidence_only(desc):
    """Today's early-stop Match/Commit binds the SERVED-SHAPE derivation, another Program: R3 (the executed set by step) still runs, R3a says by
    name that no identity address is comparable and the scope string carries it; a Commit of the fixed Program reads binds_fixed_program True."""
    from verity.ir.codec import program_digest
    lag, src = L.lag_from_header(_header(0))
    bm = dict(_bmap(range(A_R)), program_digest=program_digest(desc))  # a Commit OF the fixed Program
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=bm, boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    r3a = rep["checks"]["R3a_commit_program"]
    assert rep["ok"] is True and r3a["ok"] is True and r3a["binds_fixed_program"] is True
    bm2 = dict(bm, program_digest="served-shape-LP32_T3-" + "0" * 40)
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=bm2, boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    r3a = rep["checks"]["R3a_commit_program"]
    assert rep["checks"]["R3_executed_set"]["ok"] is True and r3a["ok"] is None and r3a["binds_fixed_program"] is False
    assert "executed-set evidence only" in rep["scope"] and rep["ok"] is True  # a verdict on what IS comparable, scoped by name
    # a B>=2 Commit binds the WORKLOAD Program (GP-01's compose of the fixed components): named `bound = workload`, still the fixed Program
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=bm2, boundary_prefix=BOUNDARY, lag=lag, lag_source=src, workload_digest=bm2["program_digest"])
    assert rep["checks"]["R3a_commit_program"]["bound"] == "workload" and rep["checks"]["R3a_commit_program"]["binds_fixed_program"] is True


def test_f_pad_23_the_verdict_says_what_it_is_about_padding_exercised_lag_discriminated_values_never(desc, desc_b, tmp_path):
    """padrev M-0398: `ok = True` on a length stop over-reads -- the fields beside it say the padding was trivial and the lag undiscriminated."""
    lag, src = L.lag_from_header(_header(1))
    # a length stop (a_r = cap): ok True, padding NOT exercised, lag NOT discriminated
    full = [1] * CAP
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    rv = rep["record_verdict"]
    assert rep["ok"] is True and rv["padding_exercised"] is False and rv["lag_discriminated"] is False and rv["values_evaluated"] is False
    assert "0 bot" in rv["statement"] and "lag not discriminated" in rv["statement"]
    assert rv["lag_discriminated_by_executed_set"] is False and rv["lag_discriminated_by_run_row"] is False
    # F-pad-28b (padrev M-0581): the SAME length stop WITH the Commit run row (32 / 33 under LAG 1): R3b's loop count discriminates LAG on a
    # length stop too -- lag_discriminated True BY THE RUN ROW, still not by the executed set; the two facts are named apart
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         run_row={"generated_tokens": CAP, "engine_steps": CAP + 1, "iterations": CAP + 1})
    rv = rep["record_verdict"]
    assert rep["ok"] is True and rv["lag_discriminated"] is True and rv["lag_discriminated_by_run_row"] is True and rv["lag_discriminated_by_executed_set"] is False
    assert rv["padding_exercised"] is False and "lag not discriminated" not in rv["statement"]
    # an early stop under LAG=1: padding exercised, lag discriminated (the two predictions differ), R6 still never
    served = Y_RAW[:A_R + 1]
    rep = R.check_record(desc, served, EOS, cap=CAP, bmap=_bmap(range(A_R + 1)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    rv = rep["record_verdict"]
    assert rep["ok"] is True and rv["padding_exercised"] is True and rv["lag_discriminated"] is True and rv["values_evaluated"] is False
    assert rv["lag_discriminated_by_executed_set"] is True and rv["lag_discriminated_by_run_row"] is False  # no run row given here
    # workload level: A padded, B not -> padding_exercised with A named; the statement names the label-swap limit
    a_tr = Y_RAW[:A_R]
    lag0, src0 = L.lag_from_header(_header(0))
    cd = R.read_capture(_capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lag0, lag_source=src0, bmap=_bmap2(range(A_R), range(2, 2 + CAP_B)),
                                  capture=cd, boundary_prefix=BOUNDARY)
    rv = rep["record_verdict"]
    assert rep["ok"] is True and rv["padding_exercised"] is True and rv["requests_padded"] == [UUID_A] and "label swap" in rv["statement"]
    # F-pad-28a (padrev M-0581): at B >= 2 the run row's loop count is the BATCH's (max_r(arrive_r + a_r) + LAG), never a request's -- R3b is
    # unevaluated per request and the workload verdict says so by name
    assert rv["run_row_evaluated"] is False and rv["lag_discriminated_by_run_row"] is False and "max_r" in rv["run_row_note"]
    assert all(p["checks"]["R3b_run_row"]["ok"] is None for p in rep["requests"])


def test_f_pad_24_the_uuid_to_rk_pairing_is_read_from_the_record_and_a_non_bijection_or_a_crossed_pair_is_refused_by_name(desc, desc_b, tmp_path):
    """padrev M-0398: a crossed pairing with equal windows passed R3/W2 silently.  Now: distinct ids both ways, every uuid the engine ran
    declared, and the record's own facts about a uuid (spec_tokens.json prompt_len / max_tokens / arrive_step) must match the manifest's for
    its r<k>; same-shape requests are named as indistinguishable (R6's job)."""
    import gzip as _gz
    a_tr = Y_RAW[:A_R]
    lag, src = L.lag_from_header(_header(0))
    cap_dir = _capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True)
    # the record's own facts about each uuid (what spec_tokens.json / tokens.json carry on a real row)
    spec = {"requests": [{"request_id": UUID_A, "prompt_len": LP, "max_tokens": CAP, "arrive_step": 0},
                         {"request_id": UUID_B, "prompt_len": LP, "max_tokens": CAP_B, "arrive_step": 2}]}
    (cap_dir / "spec_tokens.json").write_text(json.dumps(spec))
    (cap_dir.parent / "run.json").write_text(json.dumps({"realized": {"requests": [{"request_id": UUID_A}, {"request_id": UUID_B}]}}))
    cd = R.read_capture(cap_dir)
    bm = _bmap2(range(A_R), range(2, 2 + CAP_B))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    w0 = rep["checks"]["W0_pairing"]
    assert rep["ok"] is True and w0["ok"] is True and w0["pairs"] == {"r0": UUID_A, "r1": UUID_B} and w0["fact_mismatch"] == {}
    assert set(w0["sources_evaluated"]) == {"spec_tokens.json", "tokens.json", "run.json realized.requests"} and w0["indistinguishable_by_pairing_facts"] == {}
    assert w0["facts_verified"] is True
    # F-pad-25 (padrev M-0452): a record with NO pairing fact source (no spec_tokens.json, no tokens.json requests, no run.json) -- the pairing
    # is taken by hand: W0 ok None (not a pass over nothing), the workload verdict None with the scope named; the bijection refusals still apply
    bare = R.read_capture(_capture2(tmp_path / "bare", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=False))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lag, lag_source=src, bmap=bm, capture=bare, boundary_prefix=BOUNDARY)
    w0 = rep["checks"]["W0_pairing"]
    assert w0["ok"] is None and w0["sources_evaluated"] == [] and w0["facts_verified"] is False and "no pairing fact source" in w0["why"]
    assert rep["ok"] is None and "pairing unverified" in rep["scope"] and rep["checks"]["W2_chronology"]["ok"] is True
    reqs = _reqs(desc, desc_b, a_tr, B_TOKENS)
    reqs[1]["request_id"] = UUID_A
    with pytest.raises(ValueError, match="not injective"):
        R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=bare, boundary_prefix=BOUNDARY)
    # CROSSED: r0 <-> B's uuid, r1 <-> A's uuid.  Caps differ (9 vs 4), so the record's IDENTITY facts refute it by name (spec max_tokens,
    # run.json order): refused -- there is no pairing the later checks could be about
    reqs = _reqs(desc, desc_b, a_tr, B_TOKENS)
    reqs[0]["request_id"], reqs[1]["request_id"] = UUID_B, UUID_A
    reqs[0]["transcript"], reqs[1]["transcript"] = B_TOKENS, a_tr
    with pytest.raises(ValueError, match="W0 pairing refused.*do not match the manifest facts"):
        R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    # an EARLY ADMISSION (the engine ran B from step 1, declared arrive 2): the identity facts still say this uuid IS r1, so the pairing holds
    # and arrive_step is a CHRONOLOGY fact -- W0 False by name (arrive_step_disagreement, no identity_mismatch), the report continues: W2 names
    # the engine step outside B's window and the missing last step, B's R3c names its executed set -- all on ONE report
    early = R.read_capture(_capture2(tmp_path / "early", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 1)], lag=0, header=_header(0), facts=True))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lag, lag_source=src, bmap=bm, capture=early, boundary_prefix=BOUNDARY)
    w0, w2 = rep["checks"]["W0_pairing"], rep["checks"]["W2_chronology"]
    assert rep["ok"] is False and w0["ok"] is False and w0["fact_mismatch"]["r1"]["tokens.json"]["arrive_step"] == (2, 1)
    assert w0["identity_mismatch"] == {} and w0["arrive_step_disagreement"] == {"r1": {"tokens.json": (2, 1)}} and "chronology" in w0["why"]
    assert w2["ok"] is False and "5" in w2["engine_vs_program_mismatch"] and ("1" in w2["engine_vs_program_mismatch"] or "1" in w2["engine_steps_outside_population"])
    rb = next(p for p in rep["requests"] if p["request_id"] == UUID_B)
    assert rb["checks"]["R3c_capture_executed_set"]["ok"] is False and rb["checks"]["R3c_capture_executed_set"]["engine_steps"] == [1, 2, 3, 4]
    # NOT INJECTIVE: both r<k> declared with the same uuid
    reqs = _reqs(desc, desc_b, a_tr, B_TOKENS)
    reqs[1]["request_id"] = UUID_A
    with pytest.raises(ValueError, match="not injective"):
        R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    # an UNDECLARED uuid ran (a third request in the engine log): refused by name
    ev = [json.loads(l) for l in _gz.open(cap_dir / "log.jsonl.gz", "rt")]
    for e in ev:
        if e["$e"] == "StepStart" and e["step"] == 3:
            e["rows"].append(["cccccccc-0000-5000-8000-000000000003", 99])
            e["input_token_ids"].append(5)
    with _gz.open(cap_dir / "log.jsonl.gz", "wt") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    with pytest.raises(ValueError, match="undeclared"):
        R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lag, lag_source=src, bmap=bm, capture=R.read_capture(cap_dir), boundary_prefix=BOUNDARY)
    # the LIMIT, named: two requests of the same shape are listed as indistinguishable by pairing facts
    reqs = _reqs(desc, desc, a_tr, a_tr)
    reqs[1].update(cap=CAP, arrive=0)
    cd2 = R.read_capture(_capture2(tmp_path / "same", [(UUID_A, a_tr, 0), (UUID_B, a_tr, 0)], lag=0, header=_header(0), facts=True))
    rep = R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=_bmap2(range(A_R), range(A_R)), capture=cd2, boundary_prefix=BOUNDARY)
    assert rep["ok"] is True and list(rep["checks"]["W0_pairing"]["indistinguishable_by_pairing_facts"].values()) == [["r0", "r1"]]


def test_r3b_the_commit_run_row_counts_are_the_drivers_tokens_and_the_engines_steps_under_the_declared_lag(desc):
    """sys's real int-3 earlystop row: runs.jsonl generated_tokens 11 (= a_r: the EOS in, the placeholder's sample out), engine_steps 12
    (= a_r + LAG).  A run row that claims 12 generated tokens, or 11 engine steps, fails by name."""
    served = Y_RAW[:A_R + 1]
    lag, src = L.lag_from_header(_header(1))
    ok_row = {"pair": 0, "arm": "instrumented", "generated_tokens": A_R, "engine_steps": A_R + 1}
    rep = R.check_record(desc, served, EOS, cap=CAP, bmap=_bmap(range(A_R + 1)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, run_row=ok_row)
    assert rep["ok"] is True and rep["checks"]["R3b_run_row"]["ok"] is True
    for bad in ({"generated_tokens": A_R + 1, "engine_steps": A_R + 1}, {"generated_tokens": A_R, "engine_steps": A_R}):
        rep = R.check_record(desc, served, EOS, cap=CAP, bmap=_bmap(range(A_R + 1)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, run_row=bad)
        assert rep["ok"] is False and rep["failed_checks"] == ["R3b_run_row"]
    # LAG 0: both counts equal a_r
    lag0, src0 = L.lag_from_header(_header(0))
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag0, lag_source=src0,
                         run_row={"generated_tokens": A_R, "engine_steps": A_R})
    assert rep["ok"] is True and rep["checks"]["R3b_run_row"]["ok"] is True
    # a LENGTH stop under LAG 1 (the OLMoE B1 row: 32 / 33): the loop iterates once more than the executed steps -- the extra iteration schedules
    # nothing (R3c's StepStart rows stay = cap), so engine_steps == iterations == cap + 1 is the row of record and cap + 0 (the LAG-0 shape, and
    # the check's own first, refuted model `engine_steps == executed steps`) fails by name; the report names the excess beside the two counts
    full = ([t for t in Y_RAW if t not in EOS] * CAP)[:CAP]  # cap tokens, none an EOS: a length stop
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         run_row={"generated_tokens": CAP, "engine_steps": CAP + 1, "iterations": CAP + 1})
    r3b = rep["checks"]["R3b_run_row"]
    assert rep["ok"] is True and r3b["ok"] is True and r3b["expected_executed_steps"] == CAP and r3b["expected_iterations"] == CAP + 1
    assert r3b["iterations_exceed_executed_steps"] is True and "R3c_capture_executed_set" not in rep["checks"]  # no capture given here
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         run_row={"generated_tokens": CAP, "engine_steps": CAP, "iterations": CAP})
    assert rep["ok"] is False and rep["failed_checks"] == ["R3b_run_row"]
    # a run row whose two loop-count fields disagree with each other is refused too (iterations is the same fact under another name)
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         run_row={"generated_tokens": CAP, "engine_steps": CAP + 1, "iterations": CAP})
    assert rep["ok"] is False and rep["failed_checks"] == ["R3b_run_row"]


def test_r3e_a_commit_without_the_bot_population_is_named_beside_ok_and_one_with_it_defers_to_r6(desc):
    lag, src = L.lag_from_header(_header(0))
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    r3e = rep["checks"]["R3e_bot_population"]
    assert rep["ok"] is True and r3e["ok"] is None and r3e["bot_steps_n"] == CAP - A_R and r3e["bot_population_bound"] is False
    assert rep["record_verdict"]["bot_population_bound"] is False and "NOT bound" in rep["record_verdict"]["statement"]
    # a (hypothetical lifted) Commit that carries identities at every bot step: R3 must then NOT read them as 'declared inactivity executed' --
    # it does today (the binding map has no enc(bot) marker): the lifted binding-map schema is what makes the two distinguishable (F-pad-06)
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    assert rep["ok"] is False and rep["checks"]["R3_executed_set"]["declared_inactivity_executed"] == list(range(A_R, CAP))
    assert rep["checks"]["R3e_bot_population"]["bot_population_bound"] is True
    # a length stop: no bot steps -> R3e True, bot_population_bound None (nothing to bind)
    rep = R.check_record(desc, [1] * CAP, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    assert rep["ok"] is True and rep["checks"]["R3e_bot_population"]["ok"] is True and rep["record_verdict"]["bot_population_bound"] is None


def _verdict(program_digest, cap, bot_steps, *, per_step=(("input_token_ids", 1), ("instance_outputs", 363), ("fa2_hidden_m1_stream", 30), ("sampled_token_ids", 1)),
             weights=303, request="r0", steps=None, missing_override=None, extra_missing_steps=()):
    """A Commit verdict.json in the schema sys's int-3 earlystop row wrote: required_manifest.{program_digest, populations, coverage[]} --
    a Commit of the EXECUTED steps only leaves unbound per-step-population x |bot steps| identities of every per-step family, none of `weights`."""
    steps = cap if steps is None else steps
    by_fam = {f: n * steps for f, n in per_step} | {"weights": weights}
    swm = sorted(set(bot_steps) | set(extra_missing_steps))
    miss = {f: n * len(swm) for f, n in per_step} if swm else {}
    if missing_override:
        miss.update(missing_override)
    cov = {"ok": not swm, "manifest_digest": "m" * 64, "checked": sum(by_fam.values()) - by_fam["weights"], "missing_n": sum(miss.values()),
           "missing_by_family": miss, "checked_by_family": {f: v for f, v in by_fam.items() if f != "weights"}, "families_missing_entirely": [],
           "steps_with_missing": swm, "requests_with_missing": [request] if swm else []}
    return {"commit_pass": not swm, "required_manifest": {"program_digest": program_digest, "manifest_digest": "m" * 64,
                                                          "populations": {"identities": sum(by_fam.values()), "by_family": by_fam, "steps": steps},
                                                          "coverage": [cov, cov, cov]}}


def test_r7_the_commits_required_manifest_traverses_the_fixed_population_and_what_is_unbound_is_exactly_the_bot_population(desc):
    """reqval coordination (BRIEF s7B 'required logical values remain assigned before sampling'): the Commit verdict's required manifest is the
    FIXED Program's (cap-shape) -- D_MAX x requests with no post-hoc rebuild -- and a Commit of the executed steps only leaves unbound exactly
    the bot population (sys's int-3 row: 7,900 of 12,943 = 20 bot steps x 395/step).  Positive and negative, as measured."""
    lag, src = L.lag_from_header(_header(0))
    pd = W.request_component(desc, REGISTRY).program_digest
    bot = list(range(A_R, CAP))
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, bot))
    r7 = rep["checks"]["R7_required_population"]
    assert rep["ok"] is True and r7["ok"] is True and r7["manifest_of_fixed_program"] is True and r7["unbound_is_exactly_bot_population"] is True
    assert r7["steps_with_missing"] == bot and r7["missing_n"] == 395 * len(bot) and r7["families"]["instance_outputs"]["per_event"] == 363
    assert r7["families"]["weights"] == {"population": 303, "per_event": None, "unbound": 0, "unbound_expected": 0, "checked": None}
    assert r7["manifest_of"] == "unlifted" and rep["checks"]["R3a_commit_program"]["bound"] == "other"  # _bmap carries no program_digest
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=dict(_bmap(range(A_R)), program_digest=pd), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, bot))
    assert rep["checks"]["R3a_commit_program"]["bound"] == "component" and rep["checks"]["R3a_commit_program"]["binds_lifted"] is False
    # F-pad-30 (padrev M-0638): a required manifest / a binding map AGAINST Lift(P) -- program_digest = sha256(lifted.id), the digest
    # lifted_instances_emit writes -- is the FIXED Program too, named 'lifted' (never 'another Program'); an unrelated digest still reads None
    ld = rep["checks"]["R1_population"]["lifted_program_digest"]
    assert ld == R.lifted_program_digest(L.lift_request_component(W.request_component(desc, REGISTRY).fn, EOS, max_tokens=CAP, lag=lag, boundary_prefix=BOUNDARY))
    assert ld != pd and len(ld) == 64
    bm_l = dict(_bmap(range(A_R)), program_digest=ld)
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=bm_l, boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=_verdict(ld, CAP, bot))
    r7, r3a = rep["checks"]["R7_required_population"], rep["checks"]["R3a_commit_program"]
    assert rep["ok"] is True and r7["ok"] is True and r7["manifest_of_fixed_program"] is True and r7["manifest_of"] == "lifted"
    assert r3a["ok"] is True and r3a["bound"] == "lifted_component" and r3a["binds_fixed_program"] is True and r3a["binds_lifted"] is True
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=dict(_bmap(range(A_R)), program_digest="f" * 64), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict("f" * 64, CAP, bot))
    assert rep["checks"]["R7_required_population"]["ok"] is None and rep["checks"]["R7_required_population"]["manifest_of"] is None
    assert rep["checks"]["R3a_commit_program"]["ok"] is None and rep["checks"]["R3a_commit_program"]["bound"] == "other"
    assert rep["record_verdict"]["fixed_population_traversed"] is None
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=_verdict(pd, CAP, bot))
    assert rep["record_verdict"]["fixed_population_traversed"] is True
    # NEGATIVE: an ACTIVE step with an unbound identity (a coverage hole of the executed population) -- False by name, ok False (F-pad-22)
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, bot, extra_missing_steps=[1]))
    assert rep["ok"] is False and rep["failed_checks"] == ["R7_required_population"] and "ACTIVE steps" in rep["checks"]["R7_required_population"]["why"]
    # NEGATIVE: a family's unbound count that is not the bot stratum's (some bot identities bound, or extra ones missing) -- False by name
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, bot, missing_override={"instance_outputs": 363 * len(bot) - 1}))
    assert rep["ok"] is False and rep["checks"]["R7_required_population"]["why"].startswith("instance_outputs:")
    # NEGATIVE: a manifest whose step count is not the cap (a population that is not D_MAX x requests) -- False by name
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, bot, steps=CAP + 1))
    assert rep["ok"] is False and "manifest steps" in rep["checks"]["R7_required_population"]["why"]
    # a served-shape Commit (ruling 35: the manifest traverses ANOTHER Program): None with the scope named -- nothing about the bot population
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict("f" * 64, A_R, []))
    r7 = rep["checks"]["R7_required_population"]
    assert rep["ok"] is True and r7["ok"] is None and r7["manifest_of_fixed_program"] is False and "ANOTHER Program" in r7["why"]
    assert rep["record_verdict"]["fixed_population_traversed"] is None
    # a (hypothetical lifted) Commit that binds the WHOLE fixed population: coverage complete, nothing unbound -> True, fully bound
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src,
                         commit_verdict=_verdict(pd, CAP, []))
    r7 = rep["checks"]["R7_required_population"]
    assert r7["ok"] is True and r7["fixed_population_fully_bound"] is True and r7["unbound_is_exactly_bot_population"] is False
    # F-pad-31 (padrev M-0698): a LENGTH stop (0 bot) with a complete coverage -- the 'unbound is exactly the bot population' property is NOT
    # exercised (empty == empty is not a finding): None, never False; the fully-bound fact stands; padding_exercised carries the 'trivial' word
    full = ([t for t in Y_RAW if t not in EOS] * CAP)[:CAP]
    rep = R.check_record(desc, full, EOS, cap=CAP, bmap=_bmap(range(CAP)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=_verdict(pd, CAP, []))
    r7 = rep["checks"]["R7_required_population"]
    assert rep["ok"] is True and r7["ok"] is True and r7["unbound_is_exactly_bot_population"] is None and r7["fixed_population_fully_bound"] is True
    assert rep["record_verdict"]["padding_exercised"] is False
    # no verdict: None by name
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src)
    assert rep["checks"]["R7_required_population"]["ok"] is None and rep["record_verdict"]["fixed_population_traversed"] is None
    # F-pad-29 (padrev M-0608): the Commit's coverage RUNS must agree -- a row whose run 1 has a hole run 0 does not (or vice versa) is refused
    # by name, never read from run 0 alone; the agreeing case reports coverage_runs_agree True with the run count
    v = _verdict(pd, CAP, bot)
    assert v["required_manifest"]["coverage"][0] == v["required_manifest"]["coverage"][2]
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=v)
    r7 = rep["checks"]["R7_required_population"]
    assert r7["ok"] is True and r7["coverage_runs_agree"] is True and r7["coverage_runs"] == 3
    v["required_manifest"]["coverage"][1] = dict(v["required_manifest"]["coverage"][1], steps_with_missing=bot + [1], missing_n=395 * len(bot) + 1)
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=v)
    r7 = rep["checks"]["R7_required_population"]
    assert rep["ok"] is False and r7["ok"] is False and r7["coverage_runs_agree"] is False and r7["runs_disagreeing_with_run_0"] == [1] and "DISAGREE" in r7["why"]
    # ... and a verdict whose FIRST run is the complete one while a later run has the hole is refused the same way (no run is privileged)
    v2 = _verdict(pd, CAP, [])
    v2["required_manifest"]["coverage"][2] = _verdict(pd, CAP, bot)["required_manifest"]["coverage"][0]
    rep = R.check_record(desc, SERVED, EOS, cap=CAP, bmap=_bmap(range(A_R)), boundary_prefix=BOUNDARY, lag=lag, lag_source=src, commit_verdict=v2)
    assert rep["ok"] is False and rep["checks"]["R7_required_population"]["runs_disagreeing_with_run_0"] == [2]


def test_w5_the_workloads_required_manifest_leaves_unbound_exactly_the_union_of_the_requests_bot_events(desc, desc_b, tmp_path):
    """W5 = R7 at the workload level: A stops early (bot events at engine steps A_R..CAP-1), B runs to its cap (no bot event) -- a Commit of
    the executed steps leaves unbound A's bot steps only, per-event populations x A's bot events; B's steps are all bound."""
    lag, src = L.lag_from_header(_header(0))
    a_tr = Y_RAW[:A_R]
    cd = R.read_capture(_capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0), facts=True))
    bm = _bmap2(range(A_R), range(2, 2 + CAP_B))
    reqs = _reqs(desc, desc_b, a_tr, B_TOKENS)
    events = CAP + CAP_B
    bot = list(range(A_R, CAP))
    comps = [W.request_component(desc, REGISTRY), W.request_component(desc_b, REGISTRY)]
    per = (("input_token_ids", 1), ("instance_outputs", 7), ("sampled_token_ids", 1))
    v = _verdict(comps[0].program_digest, events, bot, per_step=per, request="r0")
    rep = R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY, commit_verdict=v)
    w5 = rep["checks"]["W5_required_population"]
    assert rep["ok"] is True and w5["ok"] is True and w5["requests_with_bot_events"] == ["r0"] and w5["steps_with_missing"] == bot
    assert w5["families"]["instance_outputs"] == {"population": 7 * events, "per_event": 7, "unbound": 7 * len(bot), "unbound_expected": 7 * len(bot), "checked": 7 * events}
    assert rep["record_verdict"]["fixed_population_traversed"] is True
    # per request: A's R7 sees the SAME verdict (the row's) only through the workload -- the per-request R7 is None (no per-request verdict)
    assert all(p["checks"]["R7_required_population"]["ok"] is None for p in rep["requests"])
    # NEGATIVE: the verdict names B among the requests with unbound identities (B has no bot event) -- False by name
    v_b = _verdict(comps[0].program_digest, events, bot, per_step=per, request="r1")
    rep = R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY, commit_verdict=v_b)
    assert rep["ok"] is False and "requests_with_missing" in rep["checks"]["W5_required_population"]["why"]
    # NEGATIVE: one of B's steps unbound (an Active hole in the executed population of another request) -- False by name
    v_h = _verdict(comps[0].program_digest, events, bot, per_step=per, extra_missing_steps=[3])
    rep = R.check_workload_record(reqs, EOS, lag=lag, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY, commit_verdict=v_h)
    assert rep["ok"] is False and "ACTIVE steps" in rep["checks"]["W5_required_population"]["why"]


def test_w5_reads_against_the_commits_own_population_rule_and_gates_on_the_match_lengths_f_r17b_21():
    """revb M-0342 / M-0336: W5 at B >= 2 reads against the rule the record's Commit APPLIED -- a pre-F-moe-r17-03 verdict keeps the cap-shaped
    reading (bot rows unbound); a verdict carrying the Commit's `executed_prefix_of_record` (moe aad4bd3ea) reads: bot cells = the Commit's
    NOT-EXECUTED class and never bound, active cells bound under the rule of record, no row past the stop, and the Commit's own per-request
    lengths EQUAL the Match record's (F-r17b-21) -- each False by name.  The rule used is recorded."""
    epr = {"available": True, "requests": {"r0": {"served_output_len": 11, "executed_prefix_len": 12, "r2_r3_expected_served": 12},
                                           "r1": {"served_output_len": 32, "executed_prefix_len": 32, "r2_r3_expected_served": 32}}}
    base = {"ok": True, "bot_identities": 6180, "bot_identities_unbound": 6180, "unbound_is_exactly_bot_population": True,
            "commit_rule_of_record": {"available": True, "coverage_ok": True, "programmed_not_executed_n": 6180, "programmed_not_executed_equals_bot_population": True,
                                      "rows_past_the_stop": {"ok": True}}}
    # pre-F-moe-r17-03 verdict (r17_1845's commit_attempt1_fail): the cap-shaped reading stands, nothing gates on lengths
    w = R.w5_under_commit_rule(dict(base), epr, {"required_manifest": {"coverage": [{"ok": False, "missing_n": 6180}]}}, has_cells=True)
    assert w["ok"] is True and w["commit_rule_used"] == R.RULE_CAP_SHAPE and w["commit_verdict_executed_prefix"] is None
    assert w["commit_lengths_equal_match_lengths"] == {"ok": None, "by_request": None, "rule": w["commit_lengths_equal_match_lengths"]["rule"], "gates": False}
    # a Commit on aad4bd3ea: its coverage block carries executed_prefix_of_record -- the Commit's lengths equal the Match record's -> ok, rule recorded
    good = {"required_manifest": {"coverage": [{"ok": True, "executed_prefix_of_record": {
        "requests": {"r0": {"served_output_len": 11, "executed_prefix_len": 12, "cap": 32, "eos_stop": True}, "r1": {"served_output_len": 32, "executed_prefix_len": 32, "cap": 32, "eos_stop": False}},
        "request_steps": {"r0": [0, 11], "r1": [0, 31]}, "ok": True, "lag": 1}, "rows_past_the_stop": {"ok": True, "entries_n": 0}}]}}
    w = R.w5_under_commit_rule(dict(base), epr, good, has_cells=True)
    assert w["ok"] is True and w["commit_rule_used"] == R.RULE_EXECUTED_PREFIX and w["commit_lengths_equal_match_lengths"]["ok"] is True and w["commit_lengths_equal_match_lengths"]["gates"] is True
    assert w["commit_lengths_equal_match_lengths"]["by_request"]["r0"] == {"commit_served": 11, "commit_executed": 12, "match_a_r": 11, "match_n_exec": 12, "equal": True}
    assert w["bot_cells_in_commit_not_executed_class"] is True and w["bot_identities_never_bound"] is True and w["active_cells_bound_under_rule_of_record"] is True
    assert w["commit_verdict_executed_prefix"]["rows_past_the_stop"]["ok"] is True
    # NEGATIVE (F-r17b-21): the Commit's own run stopped r0 one token earlier than the record (served 10 / executed 11) -> False by name
    bad = json.loads(json.dumps(good))
    bad["required_manifest"]["coverage"][0]["executed_prefix_of_record"]["requests"]["r0"].update(served_output_len=10, executed_prefix_len=11)
    w = R.w5_under_commit_rule(dict(base), epr, bad, has_cells=True)
    assert w["ok"] is False and w["commit_lengths_equal_match_lengths"]["ok"] is False and w["commit_lengths_equal_match_lengths"]["by_request"]["r0"]["equal"] is False
    # NEGATIVE: the Commit's not-executed class is not the bot population (a bot identity the Commit still required, or one it bound)
    w = R.w5_under_commit_rule({**base, "commit_rule_of_record": {**base["commit_rule_of_record"], "programmed_not_executed_equals_bot_population": False}}, epr, good, has_cells=True)
    assert w["ok"] is False and w["bot_cells_in_commit_not_executed_class"] is False
    w = R.w5_under_commit_rule({**base, "commit_rule_of_record": {**base["commit_rule_of_record"], "rows_past_the_stop": {"ok": False, "requests": {"r0": [12]}}}}, epr, good, has_cells=True)
    assert w["ok"] is False
    w = R.w5_under_commit_rule({**base, "commit_rule_of_record": {**base["commit_rule_of_record"], "coverage_ok": False}}, epr, good, has_cells=True)
    assert w["ok"] is False and w["active_cells_bound_under_rule_of_record"] is False
    # the merged W5 names F-r17b-21 in `why` and records the rule
    rep = {"checks": {"W5_required_population": {"ok": True, "coverage_ok": True, "checked": 1, "missing_n": 0, "note": ""}, "W2_chronology": {"ok": True}, "W0_pairing": {"ok": True}},
           "requests": [], "ok": True, "record_verdict": {"ok": True}}
    w_bad = R.w5_under_commit_rule(dict(base), epr, bad, has_cells=True)
    R.w5_from_binding_level(rep, w_bad)
    w5 = rep["checks"]["W5_required_population"]
    assert w5["ok"] is False and rep["ok"] is False and "F-r17b-21" in w5["why"] and w5["commit_rule_used"] == R.RULE_EXECUTED_PREFIX
    assert w5["verdict_manifest_coverage"]["label"].startswith("name-level: cannot attribute")
