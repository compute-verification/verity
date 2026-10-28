"""A stochastic row's selector (GumbelTopPTokenSelect_v1) has no registered evaluator, so the sampled
replay excludes its output VUs from the population under the text `no registered evaluator for GumbelTopPTokenSelect_v1` -- the SAME
why-class ('no-evaluator') as the ruled applicability domain (RoPE_v1, sampled_replay.NOT_YET), which `_complete_replay_population_gap`
did not gate (it gated plane-fed + interior-composed-v2 only).  The selector's output is THE value a stochastic row adds and is
reproducible from the committed logits only with the seed AND the top-p split count S (REQ-03), so until REQ-03 lands and a
derived_rows evaluator exists, form (A) cannot replay it.  On the first stochastic row of record form (B)
value-checked the 32 sampled ids (compared_by_family.sampled_token_ids 32 == 32 (step, request) pairs, 0 mismatch, attribution complete),
so C2 held through (B) + Match.  On a stochastic COMMIT-ONLY rerun WITHOUT --match-dir the same Commit PASSed with the 32 ids value-checked by
NEITHER form: C1 / C3 bind them, nothing computes them -- a vacuous pass path (not exercised live).

Behaviour of record (this file): a complete replay whose population excludes a TokenSelect family by `no registered evaluator` is a
form-(B)-REQUIRED gap with a FAMILY-TARGETED test -- the oracle present, its request attribution complete, `compared_by_family.sampled_token_ids`
== the (step, request) pairs (>= the excluded selector VUs), none not-compared, 0 mismatch -> PASS carrying the note; else FAIL by name.
NOT `_form_b_complete_over_values`: the stochastic row's oracle is value-partial on instance_outputs (the selector's own instance output
has no fold instance) and `result` is None there -- that partiality is not about the sampled ids.  The NOT_YET domain stays ungated; the
plane-fed FAIL still leads when both are present; greedy rows (TokenSelect_v1 HAS an evaluator) are byte-identical."""
from __future__ import annotations

import pytest

from verity_vllm.check.c2_rules import _complete_replay_population_gap, _form_b_selector_coverage
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.replay.replay_codes import NOT_YET, why_class
from verity.ml.definitions.numerics.rows import EVALUATORS


def _bnd(n_req):
    return {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


BASE = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
        "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
        "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
        "weights_pin": {"result": True, "compared": 163, "mismatched": [], "missing": []}}

K_ROPE = "no derived_rows evaluator for RoPE_v1 yet (v2)"
K_ATTN = "consumes the request's K/V history across steps (state-linked replay, v2)"
K_SEL = "no registered evaluator for GumbelTopPTokenSelect_v1"

# the first stochastic row of record's sampled_replay_p0.json population (llama32-1b L40S B1 256/32 stoch-t0.8-p0.95):
# the NOT_YET domain (RoPE 9184, Attention 4592) + the 32 selector outputs (one per step) with no evaluator; the Const* structure rows are
# counted by family only (no identity) and are not in not_evaluable
SR_STOCH = {"result": True, "grade": "complete", "why": None,
            "sample": {"evaluated": 919, "equal": 919, "mismatch_n": 0, "not_evaluated_n": 0},
            "strata_unevaluated": [], "unevaluated_strata": [],
            "population": {"vus": 32750, "not_evaluable": {K_ROPE: 9184, K_ATTN: 4592, K_SEL: 32},
                           "not_evaluable_by_family": {"RoPE_v1": 9184, "Attention_v3": 4592, "GumbelTopPTokenSelect_v1": 32}}}
# a greedy dense row (phi3 B8 256/32): the NOT_YET domain only -- TokenSelect_v1 has an evaluator
SR_GREEDY = {**SR_STOCH, "population": {"vus": 242048, "not_evaluable": {K_ROPE: 68480, K_ATTN: 34240}, "not_evaluable_by_family": {"RoPE_v1": 68480, "Attention_v3": 34240}}}

# the same row's oracle_compare (runs.jsonl commit/oracle_compare): result None / grade attribution-complete -- value-partial on
# instance_outputs (32 'no fold instance of this op at this step' = the selector's own instance output), the 32 sampled ids all compared
OC_STOCH = {"result": None, "grade": "attribution-complete", "complete": False, "compared": 6304, "mismatch_n": 0,
            "compared_by_family": {"input_token_ids": 32, "instance_outputs": 6240, "sampled_token_ids": 32},
            "not_compared_by_family": {"hidden_stream": 512, "instance_outputs": 32},
            "not_compared": {"no oracle family: hidden_stream": 512, "no fold instance of this op at this step": 32},
            "attribution": {"complete": True, "pairs": 32, "unverified": [], "unverified_n": 0, "ambiguous_steps": [], "requests": ["r0"]},
            "steps_not_snapshotted": []}


def test_the_class_label_is_the_domains_and_the_family_is_not_in_the_domain():
    """The finding's mechanism: the selector exclusion reads 'no-evaluator' like RoPE_v1 does, but GumbelTopPTokenSelect_v1 is neither in
    NOT_YET nor in EVALUATORS -- the population text is the only handle, and the gate reads it."""
    assert why_class(K_SEL) == "no-evaluator" == why_class(K_ROPE)
    assert "GumbelTopPTokenSelect_v1" not in NOT_YET                        # never the domain: replayed once the selector's evaluator is on the tree
    # (then the population carries no K_SEL and this gate is the safety net), gated by name before
    assert "TokenSelect_v1" in EVALUATORS                                   # greedy selectors are replayed; no gap on greedy rows


def test_stochastic_commit_only_rerun_without_form_b_fails_by_name():
    """The vacuous path: complete replay, 0 mismatch, C1 / C3 hold, no --match-dir oracle -> the 32 sampled ids are computed by nothing."""
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_STOCH}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 32 selector output VUs (GumbelTopPTokenSelect_v1 x32) not replayed"), why
    assert "form (B) absent" in why and "value-checked by NEITHER form" in why and "REQ-03" in why


def test_the_first_stochastic_row_of_record_passes_carrying_the_note():
    """C2 holds through (B) + Match; the PASS text says so instead of a bare PASS."""
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_STOCH, "oracle_compare": OC_STOCH}, True, min_openings=64)
    assert ok, why
    assert why.startswith("PASS (selector outputs x32 (GumbelTopPTokenSelect_v1 x32) not replayed (no evaluator; S not committed until REQ-03), value-checked by form (B) [sampled_token_ids compared 32/32 (step, request) pairs, not compared 0, 0 mismatch]"), why
    # value-partial on instance_outputs is NOT the test here (not _form_b_complete_over_values) -- result None passes the family test
    note, fail = _complete_replay_population_gap(SR_STOCH, OC_STOCH)
    assert fail is None and note and "sampled_token_ids compared 32/32" in note


@pytest.mark.parametrize("oc, expect", [
    ({**OC_STOCH, "mismatch_n": 1}, "form (B) 1 mismatch"),
    ({**OC_STOCH, "result": False, "grade": "mismatch", "why": "3781 identities differ"}, "form (B) result False (mismatch): 3781 identities differ"),
    ({**OC_STOCH, "attribution": {**OC_STOCH["attribution"], "complete": False, "unverified_n": 2}}, "form (B) request attribution not complete"),
    ({**OC_STOCH, "compared_by_family": {**OC_STOCH["compared_by_family"], "sampled_token_ids": 31}}, "sampled_token_ids compared 31/32"),
    ({**OC_STOCH, "compared_by_family": {**OC_STOCH["compared_by_family"], "sampled_token_ids": 31}, "not_compared_by_family": {"sampled_token_ids": 1}}, "not compared 1"),
    ({**OC_STOCH, "compared_by_family": {"input_token_ids": 32, "instance_outputs": 6240}}, "sampled_token_ids compared 0/32"),
])
def test_form_b_that_does_not_cover_every_sampled_id_fails_by_name(oc, expect):
    reason, _ = _form_b_selector_coverage(oc, 32)
    assert reason is not None and expect in reason, (reason, expect)
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_STOCH, "oracle_compare": oc}, True, min_openings=64)
    assert not ok, why
    if oc.get("result") is False:
        assert why == "3781 identities differ", why                          # the oracle's own FAIL leads (earlier rule, unchanged)
    else:
        assert why.startswith("C2 of record not established: ") and expect in why, why


def test_b8_needs_every_step_request_pair_not_just_the_selector_count():
    """At B>=2 the population's selector VUs are counted per request; the oracle must compare one sampled id per (step, request) pair --
    a compared count below the pairs (a request's ids missing) fails even when it is >= the excluded count of one request."""
    sr = {**SR_STOCH, "population": {**SR_STOCH["population"], "not_evaluable": {K_ROPE: 9184, K_ATTN: 4592, K_SEL: 32}}}
    oc = {**OC_STOCH, "compared_by_family": {**OC_STOCH["compared_by_family"], "sampled_token_ids": 200},
          "attribution": {**OC_STOCH["attribution"], "pairs": 256, "requests": [f"r{i}" for i in range(8)]}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(8), "sampled_replay": sr, "oracle_compare": oc}, True, min_openings=64)
    assert not ok and "sampled_token_ids compared 200/256" in why, why
    oc_ok = {**oc, "compared_by_family": {**oc["compared_by_family"], "sampled_token_ids": 256}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(8), "sampled_replay": sr, "oracle_compare": oc_ok}, True, min_openings=64)
    assert ok and "sampled_token_ids compared 256/256" in why, why


def test_greedy_rows_and_the_not_yet_domain_are_unchanged():
    """TokenSelect_v1 has an evaluator (no exclusion); RoPE / Attention exclusions are the applicability domain and are not gated here."""
    ok, why = commit_verdict({**BASE, "binding": _bnd(8), "sampled_replay": SR_GREEDY}, True, min_openings=64)
    assert ok and why.startswith("PASS"), why                                      # no oracle, no note about selectors
    assert "selector" not in why
    assert _complete_replay_population_gap(SR_GREEDY, None) == (None, None)
    # an OLDER record's RoPE / attention labels by name (K_ROPE, K_ATTN: RoPE_v1, Attention_v3 and Attention_v2 have
    # left NOT_YET; every replay on this tree evaluates those rows) are neither the domain nor a gated gap: the
    # verdict is the one its tree gave, a Commit-only rerun evaluates the rows
    assert not ({"RoPE_v1", "Attention_v3", "Attention_v2"} & set(NOT_YET))
    sr = {**SR_GREEDY, "population": {**SR_GREEDY["population"], "not_evaluable": {K_ROPE: 10, K_ATTN: 5}}}
    assert _complete_replay_population_gap(sr, None) == (None, None)
    # a would-be 'no registered evaluator for RoPE_v1' on a tree that had it in neither table WOULD be gated like Gemm_v2 was
    sr = {**SR_GREEDY, "population": {**SR_GREEDY["population"], "not_evaluable": {"no registered evaluator for RoPE_v1": 10}}}
    assert _complete_replay_population_gap(sr, None)[1] is not None


def test_earlier_fails_lead_the_plane_fed_fail_first():
    """The plane-fed FAIL (neither form covers the planes) still names first when a stochastic MoE row carries both gaps."""
    k_06f = "fed by moe_block_stream planes (SiluMul_v1@experts): row -> plane mapping not in this replay (MAN-06f, v2)"
    sr = {**SR_STOCH, "population": {**SR_STOCH["population"], "not_evaluable": {K_ROPE: 9184, k_06f: 64, K_SEL: 32}}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr, "oracle_compare": OC_STOCH}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs"), why
    # the weights pin FAIL (earlier in the verdict) is unchanged
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_STOCH, "oracle_compare": OC_STOCH,
                              "weights_pin": {"result": False, "why": "1 served weight(s) differ from the checkpoint of record: x"}}, True, min_openings=64)
    assert not ok and why.startswith("weights of record: 1 served weight(s) differ"), why


def test_a_selector_family_alias_is_read_by_the_token_select_rule_not_by_a_list():
    """Any *TokenSelect family without an evaluator is the same gap (GumbelTokenSelect_v1, ArgmaxTokenSelect_v1 if they ever lose theirs);
    a non-selector family without an evaluator is not this rule (it is not covered by the sampled_token_ids family)."""
    sr = {**SR_STOCH, "population": {**SR_STOCH["population"], "not_evaluable": {K_ROPE: 9184, "no registered evaluator for GumbelTokenSelect_v1": 32}}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr}, True, min_openings=64)
    assert not ok and "32 selector output VUs (GumbelTokenSelect_v1 x32)" in why, why
    sr2 = {**SR_STOCH, "population": {**SR_STOCH["population"], "not_evaluable": {K_ROPE: 9184, "no registered evaluator for Gemm_v2": 7}}}
    note, fail = _complete_replay_population_gap(sr2, None)
    assert fail is not None and "selector" not in fail   # the Hopper gate (test_commit_verdict_hopper_no_evaluator_gap.py)
