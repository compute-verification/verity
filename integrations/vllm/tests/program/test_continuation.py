"""Continuation_v1 -- controller first.

The reference evaluator (`next_flags`), the INDEPENDENTLY written three-state machine, and the checker's recurrence over a raw
transcript agree on EVERY EOS pattern for small caps under both profiles; the named invariants hold; each edge case is one test; the lag
unit's guard chain (`LStepLive`/`LSelect`/`LBot`, simulated with the registered evaluators) computes exactly `r` / `u` of the rule;
the derived invariant reproduces the three real records' executed lengths; the wrong profile FAILS by name.  Everything labelled lifted-padded; nothing here reads a trace to set A.
"""
from __future__ import annotations

import itertools

import pytest

from verity_vllm.program.lifting import continuation as K
from verity.ml.definitions.registry import lifted as L
from verity_vllm.program.lifting import lifted_records

EOS = (17,)
ORD = 5  # an ordinary (non-EOS) token id


def pattern_tokens(is_eos: tuple[bool, ...]) -> list[int]:
    return [EOS[0] if b else ORD for b in is_eos]


def all_patterns(cap: int):
    return itertools.product((False, True), repeat=cap)


# ---------------------------------------------------------------------------------------------------------
# exhaustive agreement: reference evaluator == three-state machine == recurrence, both profiles, C <= 6
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("cap", [1, 2, 3, 4, 5, 6])
def test_reference_evaluator_three_state_machine_and_recurrence_agree_on_every_pattern(cap, A):
    n = 0
    for pat in all_patterns(cap):
        ref = K.three_state_run(pat, cap=cap, A=A)
        # the reference evaluator unrolled by hand
        e, r, ev = True, True, []
        for t in range(1, cap + 1):
            ev.append((e, r))
            e, r = K.next_flags(emit=e, token_is_eos=pat[t - 1], has_next_slot=t < cap, async_one_step=A == 1)
        assert ref == ev, (cap, A, pat)
        # the recurrence over the raw tokens the pattern would produce at the slots that run (an inactive slot produces none)
        raw = [pattern_tokens(pat)[t - 1] if ref[t - 1][1] else None for t in range(1, cap + 1)]
        slots = K.recurrence(raw, cap=cap, A=A, eos=EOS)
        assert K.bits(slots) == ref, (cap, A, pat)
        inv = K.invariants(slots, cap=cap, A=A)
        assert all(inv.values()), (cap, A, pat, {k: v for k, v in inv.items() if not v})
        n += 1
    assert n == 2 ** cap


@pytest.mark.parametrize("A", [0, 1])
def test_derived_invariant_n_executed_equals_the_run_count_on_every_pattern(A):
    for cap in range(1, 7):
        for pat in all_patterns(cap):
            ref = K.three_state_run(pat, cap=cap, A=A)
            n_served = sum(1 for e, _ in ref if e)
            n_exec = sum(1 for _, r in ref if r)
            assert n_exec == K.n_executed_derived(n_served, cap=cap, A=A), (cap, A, pat)
            # the lag rule's expected_served(a_r, cap, lag) is the same number with a_r = n_served (one rule, two names)
            assert n_exec == L.expected_served(n_served, cap, A)


# ---------------------------------------------------------------------------------------------------------
# the worked example and the edge cases
# ---------------------------------------------------------------------------------------------------------


def test_worked_example_c5_eos_at_slot_2_async_matches_the_spec_table():
    slots = K.recurrence([100, EOS[0], 7, None, None], cap=5, A=1, eos=EOS)
    assert [(s.t, int(s.emit), int(s.run), s.raw, s.out) for s in slots] == [(1, 1, 1, 100, 100), (2, 1, 1, 17, 17), (3, 0, 1, 7, None),
                                                                            (4, 0, 0, None, None), (5, 0, 0, None, None)]
    assert [s.kind for s in slots] == ["active", "active", "lag", "inactive", "inactive"]
    assert [s.next_in for s in slots] == [100, 17, None, None, None]  # slot 3 consumes the RAW EOS as its forward input; slot 4 gets bot
    assert K.served_output(slots) == [100, 17] and K.executed_slots(slots) == [1, 2, 3] and K.lag_slots(slots) == [3]
    assert [s.stop for s in slots] == [False, True, False, False, False]
    # synchronous on the same raw tokens: slot 3 already inactive
    sync = K.recurrence([100, EOS[0], None, None, None], cap=5, A=0, eos=EOS)
    assert [s.kind for s in sync] == ["active", "active", "inactive", "inactive", "inactive"] and K.executed_slots(sync) == [1, 2]


def test_edge_c_equals_1_one_slot_both_profiles_no_lag_slot():
    for A in (0, 1):
        for tok in (ORD, EOS[0]):
            s = K.recurrence([tok], cap=1, A=A, eos=EOS)
            assert K.bits(s) == [(True, True)] and not s[0].next_emit and not s[0].next_run and K.lag_slots(s) == []
            assert K.n_executed_derived(1, cap=1, A=A) == 1


def test_edge_no_eos_exactly_c_slots_both_profiles():
    for A in (0, 1):
        s = K.recurrence([ORD] * 6, cap=6, A=A, eos=EOS)
        assert K.executed_slots(s) == [1, 2, 3, 4, 5, 6] and K.served_output(s) == [ORD] * 6 and K.lag_slots(s) == []


def test_edge_eos_in_slot_1_sync_stops_async_one_more_if_room():
    sync = K.recurrence([EOS[0]], cap=4, A=0, eos=EOS)
    assert K.executed_slots(sync) == [1] and K.served_output(sync) == [EOS[0]]
    asy = K.recurrence([EOS[0], ORD], cap=4, A=1, eos=EOS)
    assert K.executed_slots(asy) == [1, 2] and K.served_output(asy) == [EOS[0]] and K.lag_slots(asy) == [2]
    # C = 1: no room -- the async profile executes nothing beyond the cap
    assert K.executed_slots(K.recurrence([EOS[0]], cap=1, A=1, eos=EOS)) == [1]


def test_edge_eos_in_penultimate_slot_async_executes_the_final_slot_but_does_not_serve_it():
    s = K.recurrence([ORD, ORD, ORD, EOS[0], ORD], cap=5, A=1, eos=EOS)
    assert K.executed_slots(s) == [1, 2, 3, 4, 5] and K.served_output(s) == [ORD, ORD, ORD, EOS[0]] and s[4].kind == "lag" and s[4].out is None
    assert K.n_executed_derived(4, cap=5, A=1) == 5


def test_edge_eos_in_final_slot_nothing_beyond_the_cap_either_profile():
    for A in (0, 1):
        s = K.recurrence([ORD] * 4 + [EOS[0]], cap=5, A=A, eos=EOS)
        assert K.executed_slots(s) == [1, 2, 3, 4, 5] and K.served_output(s) == [ORD] * 4 + [EOS[0]] and K.lag_slots(s) == []
        assert not s[-1].next_run and not s[-1].next_emit and K.n_executed_derived(5, cap=5, A=A) == 5


def test_edge_lag_token_eos_or_not_changes_nothing_no_restart_no_second_lag():
    a = K.recurrence([ORD, EOS[0], EOS[0], None, None, None], cap=6, A=1, eos=EOS)  # the lag token IS an EOS
    b = K.recurrence([ORD, EOS[0], ORD, None, None, None], cap=6, A=1, eos=EOS)     # the lag token is ordinary
    assert K.bits(a) == K.bits(b) and K.executed_slots(a) == [1, 2, 3] and K.lag_slots(a) == [3]
    # b_3 = IsEOS(y_3) IS computed on the real lag token (True for the EOS, False for the ordinary one) -- but stop_3 = e_3 AND b_3 = 0 either
    # way and e_4 = h AND e_3 AND NOT b_3 = 0 with e_3 already 0: the discarded slot's EOS never restarts / extends anything
    assert a[2].is_eos is True and b[2].is_eos is False
    assert a[2].stop is False and b[2].stop is False and K.bits(a)[3:] == [(False, False)] * 3
    # the raw lag EOS is still a real committed token (required value) -- recurrence keeps it as raw; it is not in the output
    assert a[2].raw == EOS[0] and a[2].out is None


def test_edge_two_requests_stopping_at_different_times_have_independent_states():
    r0 = K.recurrence([ORD, ORD, EOS[0], ORD, None, None, None, None], cap=8, A=1, eos=EOS)
    r1 = K.recurrence([ORD] * 8, cap=8, A=1, eos=EOS)
    assert K.executed_slots(r0) == [1, 2, 3, 4] and K.executed_slots(r1) == list(range(1, 9))
    # nothing of r0's state enters r1's: the function of r1's transcript alone
    assert K.bits(r1) == K.bits(K.recurrence([ORD] * 8, cap=8, A=1, eos=EOS))


def test_edge_unscheduled_request_is_pending_not_inactive_named_by_the_slot_map():
    ok = K.slot_map([3, 4, 5, 6], arrive_step=3)
    assert ok["ok"] and ok["slots"] == [1, 2, 3, 4]
    gap = K.slot_map([3, 4, 6, 7], arrive_step=3)  # engine step 5: the request was not scheduled
    assert not gap["ok"] and gap["slots"] is None and "not contiguous" in gap["problems"][0] and "pending, not inactive" in gap["problems"][0]
    late = K.slot_map([4, 5], arrive_step=3)
    assert not late["ok"] and "slot 1 is not at its arrival" in late["problems"][0]
    chunked = K.slot_map([0, 1, 2], arrive_step=0, prefill_steps=[0, 1], chunked_prefill_observed=True)
    assert not chunked["ok"] and any("chunked prefill" in p for p in chunked["problems"]) and any("prefill spans 2" in p for p in chunked["problems"])
    assert chunked["chunked_prefill_attested"] == "observed" and chunked["vacuous_by_absence"] == []
    # the two prefill sub-checks are VACUOUS when the capture attests nothing -- named, never counted as a pass
    assert ok["chunked_prefill_attested"] == "absent" and ok["prefill_steps_attested"] == "absent" and len(ok["vacuous_by_absence"]) == 2
    assert "the workload's applicability" in ok["positive_attestation"]
    attested = K.slot_map([3, 4, 5, 6], arrive_step=3, prefill_steps=[3], chunked_prefill_observed=False)
    assert attested["ok"] and attested["vacuous_by_absence"] == [] and attested["chunked_prefill_attested"] == "not-observed" and attested["prefill_steps_attested"] == [3]


def test_r5_rng_advances_iff_run_is_stated_and_the_lag_slot_is_the_one_post_eos_draw():
    """The Definition SAYS it (RNG_STATEMENT) and the rule's accounting names the draws: one per executed slot, exactly one after the
    served EOS under A = 1 (at pos = prompt_len - 1 + n, the ledger's query position), none under A = 0, none at an inactive slot."""
    assert "iff r_t" in K.RNG_STATEMENT and "one draw" in K.RNG_STATEMENT and "keyed by (seed, pos)" in K.RNG_STATEMENT
    asy = K.rng_accounting(K.recurrence([ORD, ORD, EOS[0], 9, None, None], cap=6, A=1, eos=EOS), prompt_len=32)
    assert asy["draws"] == 4 and asy["draw_slots"] == [1, 2, 3, 4] and asy["draws_after_served_eos"] == 1 and asy["lag_draw_slot"] == 4
    assert asy["lag_draw_pos"] == 32 - 1 + 3 and asy["inactive_slots_draw_nothing"] == [5, 6]  # pos = prompt_len - 1 + n, n = 3 served (slot 4 at L - 1 + 3)
    syn = K.rng_accounting(K.recurrence([ORD, ORD, EOS[0], None, None, None], cap=6, A=0, eos=EOS), prompt_len=32)
    assert syn["draws"] == 3 and syn["draws_after_served_eos"] == 0 and syn["lag_draw_slot"] is None and syn["lag_draw_pos"] is None
    full = K.rng_accounting(K.recurrence([ORD] * 6, cap=6, A=1, eos=EOS))
    assert full["draws"] == 6 and full["draws_after_served_eos"] == 0 and full["lag_draw_pos"] is None and full["inactive_slots_draw_nothing"] == []


def test_edge_wrong_profile_fails_by_name_never_a_fitted_explanation():
    # an A = 1 record: 4 served (EOS 4th), 5 executed (the lag slot's raw token 9), cap 8
    exe, raw, served = [1, 2, 3, 4, 5], {1: ORD, 2: ORD, 3: ORD, 4: EOS[0], 5: 9}, [ORD, ORD, ORD, EOS[0]]
    good = K.check_transcript(executed_slots_of_record=exe, raw_by_slot=raw, served_of_record=served, cap=8, A=1, eos=EOS)
    assert good["ok"] and good["failed"] == [] and good["checks"]["f_n_executed_derived_invariant"]["derived"] == 5
    bad = K.check_transcript(executed_slots_of_record=exe, raw_by_slot=raw, served_of_record=served, cap=8, A=0, eos=EOS)
    assert not bad["ok"] and "a_executed_set_equals_run_slots" in bad["failed"]
    assert bad["checks"]["a_executed_set_equals_run_slots"]["execution_past_the_stop"] == [5] and "execution past the stop" in bad["checks"]["a_executed_set_equals_run_slots"]["why"]
    neg = K.wrong_profile_negative(executed_slots_of_record=exe, raw_by_slot=raw, served_of_record=served, cap=8, A_attested=1, eos=EOS)
    assert neg["fails_by_name"] is True and neg["A_tried"] == 0 and neg["exercised"]
    # and the mirror: an A = 0 record (4 executed) read under A = 1 -> a DELETED step at slot 5
    exe0, raw0 = [1, 2, 3, 4], {1: ORD, 2: ORD, 3: ORD, 4: EOS[0]}
    bad1 = K.check_transcript(executed_slots_of_record=exe0, raw_by_slot=raw0, served_of_record=served, cap=8, A=1, eos=EOS)
    assert bad1["checks"]["a_executed_set_equals_run_slots"]["deleted_steps"] == [5] and "a deleted step" in bad1["checks"]["a_executed_set_equals_run_slots"]["why"]
    # a cap-exhausting record: both profiles agree, the negative is NOT exercised (stated, not claimed)
    full = K.wrong_profile_negative(executed_slots_of_record=list(range(1, 9)), raw_by_slot={t: ORD for t in range(1, 9)}, served_of_record=[ORD] * 8, cap=8, A_attested=1, eos=EOS)
    assert full["exercised"] is False and full["fails_by_name"] is None


def test_rev_m0445_negatives_lag_token_served_eos_set_extended_second_lag_row():
    exe, raw, served = [1, 2, 3, 4, 5], {1: ORD, 2: ORD, 3: ORD, 4: EOS[0], 5: 9}, [ORD, ORD, ORD, EOS[0]]
    # lag-slot token served: one entry past the emit prefix -> (b) and (c) by name
    r = K.check_transcript(executed_slots_of_record=exe, raw_by_slot=raw, served_of_record=served + [9], cap=8, A=1, eos=EOS)
    assert {"b_served_equals_emit_tokens", "c_lag_token_not_served"} <= set(r["failed"]) and r["checks"]["c_lag_token_not_served"]["served_entries_past_emit_prefix"] == 1
    # a served early stop on a token outside the pinned set (the record stopped on 9 with 9 not in EOS): (e), and the executed set disagrees
    r = K.check_transcript(executed_slots_of_record=[1, 2, 3, 4], raw_by_slot={1: ORD, 2: ORD, 3: ORD, 4: 9}, served_of_record=[ORD, ORD, ORD, 9], cap=8, A=0, eos=EOS)
    assert "e_stop_under_pinned_set" in r["failed"] and "never extend the set" in r["checks"]["e_stop_under_pinned_set"]["why"]
    # a second lag row (execution at slot 6 under A = 1): past the stop
    r = K.check_transcript(executed_slots_of_record=exe + [6], raw_by_slot={**raw, 6: 3}, served_of_record=served, cap=8, A=1, eos=EOS)
    assert r["checks"]["a_executed_set_equals_run_slots"]["execution_past_the_stop"] == [6]
    # the lag slot's raw token altered: the verdict does NOT change (it is discarded), the bits are the same
    r2 = K.check_transcript(executed_slots_of_record=exe, raw_by_slot={**raw, 5: 12345}, served_of_record=served, cap=8, A=1, eos=EOS)
    assert r2["ok"] and [s["emit"] for s in r2["slots"]] == [1, 1, 1, 1, 0, 0, 0, 0] and [s["run"] for s in r2["slots"]] == [1, 1, 1, 1, 1, 0, 0, 0]


def test_f_r17_14_the_rule_stops_at_the_first_eos_a_served_eos_x_eos_transcript_fails_by_name():
    """Continuation_v1 stops at the FIRST EOS by construction (`e_{t+1} = h AND e AND NOT b`), so served `[EOS, x, EOS]` (cap 5, A = 1, eos {7}) predicts served `[7]`,
    executed `[1, 2]`, and a record that served three tokens and executed four fails (a) (b) (c) (d) (f) by name -- the named negative."""
    slots = K.recurrence([7, 3, 7, 5, 5], cap=5, A=1, eos=(7,))
    assert K.served_output(slots) == [7] and K.executed_slots(slots) == [1, 2] and K.lag_slots(slots) == [2]
    r = K.check_transcript(executed_slots_of_record=[1, 2, 3, 4], raw_by_slot={1: 7, 2: 3, 3: 7, 4: 5}, served_of_record=[7, 3, 7], cap=5, A=1, eos=(7,))
    assert not r["ok"] and {"a_executed_set_equals_run_slots", "b_served_equals_emit_tokens", "c_lag_token_not_served", "d_inactive_suffix_has_no_rows",
                            "f_n_executed_derived_invariant"} <= set(r["failed"])
    assert r["checks"]["a_executed_set_equals_run_slots"]["execution_past_the_stop"] == [3, 4]
    assert r["checks"]["b_served_equals_emit_tokens"]["first_differing_position"] == 1  # position 1: the rule emits nothing after the first EOS
    # the same under A = 0: served [7], executed [1]; the record's rows 2..4 are past the stop
    r0 = K.check_transcript(executed_slots_of_record=[1, 2, 3, 4], raw_by_slot={1: 7, 2: 3, 3: 7, 4: 5}, served_of_record=[7, 3, 7], cap=5, A=0, eos=(7,))
    assert r0["checks"]["a_executed_set_equals_run_slots"]["execution_past_the_stop"] == [2, 3, 4]


def test_executed_prefix_by_rule_is_moes_block_read_from_the_recurrence_first_eos_and_the_real_records():
    """One mechanism: the executed prefix FROM the rule -- equal to the Commit's executed_prefix_of_record on every well-formed record
    (the three real ones + the length stop), and FAIL by name on [EOS, x, EOS], which a last-token test accepts."""
    for n_served, cap, n_exec, eos_id in ((11, 32, 12, 0), (8, 32, 9, 0), (11, 32, 12, 50279)):
        served = [5] * (n_served - 1) + [eos_id]
        r = K.executed_prefix_by_rule(served, cap=cap, A=1, eos=(eos_id,))
        assert r["ok"] and r["executed_prefix_len"] == n_exec == r["derived_invariant_n_executed"] and r["eos_stop"] and r["eos_step"] == n_served - 1
        assert r["lag_steps"] == [n_served] and r["cap_exhausted"] is False and r["served_past_first_eos"] == []
        r0 = K.executed_prefix_by_rule(served, cap=cap, A=0, eos=(eos_id,))
        assert r0["ok"] and r0["executed_prefix_len"] == n_served and r0["lag_steps"] == []
    full = K.executed_prefix_by_rule([5] * 32, cap=32, A=1, eos=(0,))
    assert full["ok"] and full["executed_prefix_len"] == 32 and full["cap_exhausted"] and full["eos_stop"] is False and full["lag_steps"] == []
    # [EOS, x, EOS] is a problem BY NAME (a last-token test says ok, executed 4)
    bad = K.executed_prefix_by_rule([7, 3, 7], cap=5, A=1, eos=(7,))
    assert not bad["ok"] and bad["first_eos_index"] == 0 and bad["served_past_first_eos"] == [1, 2] and "FIRST EOS" in bad["why"] and bad["eos_stop"] is None
    # an early stop on a token outside the pinned set; over the cap; nothing served
    assert "no served token is a declared EOS id" in K.executed_prefix_by_rule([5, 5, 9], cap=8, A=1, eos=(7,))["why"]
    assert "over the declared cap" in K.executed_prefix_by_rule([5] * 9, cap=8, A=0, eos=(7,))["why"]
    assert not K.executed_prefix_by_rule([], cap=8, A=0, eos=(7,))["ok"]


# ---------------------------------------------------------------------------------------------------------
# IsEOS total and the profile source
# ---------------------------------------------------------------------------------------------------------


def test_is_eos_is_total_over_the_lifted_encoding_bot_and_noncanonical_are_not_eos():
    assert K.is_eos_total(32, L.enc_active(32, 17), EOS) is True
    assert K.is_eos_total(32, L.enc_active(32, 18), EOS) is False
    assert K.is_eos_total(32, L.enc_bot(32), EOS) is False           # IsEOS(bot) = 0 by the tag gate, not a convention
    assert K.is_eos_total(32, L.enc_bot(32) | 17, EOS) is False      # 1||17: a noncanonical word, never normalized, never EOS
    assert K.is_eos_total(32, L.inv(32), EOS) is False
    # the recurrence over lifted words: a bot at a slot with r = 1 reads b = 0 (and check_transcript names the missing token)
    s = K.recurrence([L.enc_active(32, ORD), L.enc_bot(32)], cap=2, A=0, eos=EOS, lifted_words=True)
    assert s[1].raw is None and s[1].is_eos is False


def test_profile_a_is_the_resolved_header_field_through_the_one_lag_rule_never_defaulted():
    assert K.resolve_profile({"profile": {"engine_kwargs": {"async_scheduling": True}}}) == (1, K.PROFILE_FIELD)
    assert K.resolve_profile({"profile": {"engine_kwargs": {"async_scheduling": False}}}) == (0, K.PROFILE_FIELD)
    with pytest.raises(ValueError, match="not the resolved boolean"):
        K.resolve_profile({"profile": {"engine_kwargs": {"async_scheduling": None}}})
    with pytest.raises(ValueError, match="not attested"):
        K.resolve_profile({"profile": {"engine_kwargs": {}}})
    with pytest.raises(ValueError, match="DP-pad-02"):
        K.check_profile(2)
    assert K.declared_profile({"workload_fn": "Workload_v1{requests=(a,b),shared=weights,lag=1}"}) == (1, lifted_records.LAG_STATIC_FIELD + ":lag=")


# ---------------------------------------------------------------------------------------------------------
# the lag unit's guard chain computes exactly r / u of the rule
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("cap", [2, 3, 4, 5, 6])
def test_r17_1_guard_chain_is_run_and_next_input_of_continuation_v1_on_every_pattern(cap, lag):
    for pat in all_patterns(cap):
        raw_ids = pattern_tokens(pat)
        raw_words = [L.enc_active(32, t) for t in raw_ids]
        ch = K.chain_bits(raw_words, cap=cap, lag=lag, eos=EOS)
        sl = K.recurrence(raw_ids, cap=cap, A=lag, eos=EOS)
        assert ch["run"] == [s.run for s in sl], (cap, lag, pat)
        # the chain's tok_in_s (= in_{s+1}) is the recurrence's next_in of slot s: the raw token when r_{s+1} = 1, bot otherwise
        assert ch["in"][1:] == [L.enc_active(32, s.next_in) if s.next_in is not None else None for s in sl[:-1]], (cap, lag, pat)
        # the committed out[] of the chain is Active exactly on the run slots (strictness), bot after
        assert [w != L.enc_bot(32) for w in ch["out"]] == [s.run for s in sl], (cap, lag, pat)


# ---------------------------------------------------------------------------------------------------------
# the three real records (numbers of record)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("row, n_served, cap, n_exec", [("smollm2 earlystop r17_1921", 11, 32, 12), ("smollm2 earlystop2 r17_1921", 8, 32, 9),
                                                        ("olmoe mixedstop r17_1845 r0", 11, 32, 12), ("olmoe mixedstop r17_1845 r1 (cap)", 32, 32, 32)])
def test_derived_invariant_reproduces_the_real_records_executed_lengths_under_the_attested_async_profile(row, n_served, cap, n_exec):
    assert K.n_executed_derived(n_served, cap=cap, A=1) == n_exec, row
    # and the synchronous profile predicts n: the same transcript under A = 0 is a different executed set (the wrong-profile negative)
    assert K.n_executed_derived(n_served, cap=cap, A=0) == n_served


# ---------------------------------------------------------------------------------------------------------
# the rule's draw positions, and the ledger compared to them by name
# ---------------------------------------------------------------------------------------------------------


def test_r5_expected_draw_positions_are_the_run_slots_and_the_ledger_is_compared_by_name():
    L_ = 32
    # the earlystop shape: 11 served (EOS last), cap 32, A = 1 -> 12 draws at pos 31..42 (slot t at L - 1 + (t - 1), the ledger's pos fact),
    # the lag draw at 42 (= the lag_draw_pos of record)
    exp = K.expected_draw_positions([ORD] * 10 + [EOS[0]], cap=32, A=1, eos=EOS, prompt_len=L_)
    assert exp["positions"] == list(range(31, 43)) and exp["lag_positions"] == [42] and exp["n"] == 12 and exp["first_forbidden_position"] == 43
    assert exp["served_positions"] == list(range(31, 42))
    # A = 0: no lag draw
    exp0 = K.expected_draw_positions([ORD] * 10 + [EOS[0]], cap=32, A=0, eos=EOS, prompt_len=L_)
    assert exp0["positions"] == list(range(31, 42)) and exp0["lag_positions"] == []
    # a cap-exhausting request: cap draws, no lag position, nothing after the cap in either profile
    full = K.expected_draw_positions([ORD] * 8, cap=8, A=1, eos=EOS, prompt_len=L_)
    assert full["positions"] == list(range(31, 39)) and full["lag_positions"] == [] and full["first_forbidden_position"] == 39
    # EOS at the last slot: the cap bounds the lag slot (min(n + 1, C)); exactly cap draws
    at_cap = K.expected_draw_positions([ORD] * 7 + [EOS[0]], cap=8, A=1, eos=EOS, prompt_len=L_)
    assert at_cap["positions"] == list(range(31, 39)) and at_cap["lag_positions"] == []
    # the ledger: measured exactly -> ok
    assert K.compare_draw_positions(exp["positions"], list(range(31, 43)))["ok"]
    # over-cap draws with NO EOS are EXTRA positions by name (never silent)
    c = K.compare_draw_positions(full["positions"], list(range(31, 40)))
    assert not c["ok"] and c["extra"] == [39] and "without r_t" in c["problems"][0]
    # two draws after the served EOS (a second lag draw), a missing served draw, a repeated position, out of order: each named
    c2 = K.compare_draw_positions(exp["positions"], list(range(31, 44)))
    assert c2["extra"] == [43] and not c2["ok"]
    c3 = K.compare_draw_positions(exp["positions"], [p for p in range(31, 43) if p != 35])
    assert c3["missing"] == [35] and "without its sampling event" in c3["problems"][0]
    c4 = K.compare_draw_positions(exp["positions"], list(range(31, 43)) + [40])
    assert c4["repeated"] == [40] and c4["extra"] == [] and not c4["ok"]
    c5 = K.compare_draw_positions(exp["positions"], list(reversed(range(31, 43))))
    assert not c5["ok"] and "out of order" in c5["problems"][0]
    # A = 0 record with the lag draw present: the extra draw at 42 is named (the RNG advanced without r_t)
    c6 = K.compare_draw_positions(exp0["positions"], list(range(31, 43)))
    assert c6["extra"] == [42] and not c6["ok"]
    with pytest.raises(ValueError):
        K.expected_draw_positions([ORD], cap=4, A=2, eos=EOS, prompt_len=L_)


def test_r5_positions_are_the_ledgers_pos_fact_draw_k_at_prompt_len_minus_1_plus_k_stoch_m_0749():
    """One convention on the record: the draw ledger maps draw k (k from 0, slot t = k + 1) to pos = prompt_len - 1 + k, the query
    position whose logits are sampled and the `pos` fact the sampled rows carry (an 8-token prompt's first draw is at pos 7, the llama B8
    record).  `expected_draw_positions` gives the same numbers -- the ledger imports `recurrence`/`executed_slots` and a disagreement is
    'two mechanisms'."""
    L_, C = 8, 6
    exp = K.expected_draw_positions([ORD, ORD, EOS[0]], cap=C, A=1, eos=EOS, prompt_len=L_)
    ledger = [L_ - 1 + k for k in range(4)]  # 3 served + the lag draw = draws k = 0..3
    assert exp["positions"] == ledger == [7, 8, 9, 10] and exp["lag_positions"] == [10] and exp["first_forbidden_position"] == 11
    acc = K.rng_accounting(K.recurrence([ORD, ORD, EOS[0], 9, None, None], cap=C, A=1, eos=EOS), prompt_len=L_)
    assert acc["lag_draw_pos"] == 10 == exp["lag_positions"][0]
    # the ledger's own rule, restated (stoch_recompute.expected_executed_slots): first EOS at t < C -> slots 1..t + min(A, C - t)
    assert K.executed_slots(K.recurrence([ORD, ORD, EOS[0]], cap=C, A=1, eos=EOS)) == list(range(1, 3 + 1 + min(1, C - 3)))


def test_f_r17b_40_first_draw_is_at_the_last_prompt_position_qwen_b8_l_1024_first_pos_1023_last_1057_at_cap_35():
    """The record's device key -- qwen B8 request with L = 1024, cap 35, no EOS: first_pos 1023, last_pos 1057, 388/388
    device_facts_match on exactly those positions.  The map is pos = prompt_len - 2 + t (slot 1 at L - 1)."""
    exp = K.expected_draw_positions([ORD] * 35, cap=35, A=1, eos=EOS, prompt_len=1024)
    assert exp["positions"][0] == 1023 and exp["positions"][-1] == 1057 and exp["n"] == 35 and exp["lag_positions"] == []
    assert exp["positions"] == [1024 - 2 + t for t in range(1, 36)] and exp["first_forbidden_position"] == 1058
    # llama B8, an 8-token prompt: pos 7 first
    assert K.expected_draw_positions([ORD] * 4, cap=4, A=0, eos=EOS, prompt_len=8)["positions"] == [7, 8, 9, 10]
