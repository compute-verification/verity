"""[r8 lane E-cleanup 2026-09-18] unit tests for the CANDIDATE chunked-prefill rules (`verity_capture.experimental.chunked_prefill_rules`).

Synthetic fixtures modelled on `B0-cov_b0_b1_c256_ck64` (one request, prompt in 3 chunks: partial-prefill steps 0 and 1,
first token at step 2, 15 served tokens at steps 2..16).  The rules must (i) reduce to the frozen ones on unchunked records,
(ii) drop exactly the discarded draws, (iii) still fail on every other disagreement, and (iv) never turn a missing artefact into a pass."""
from __future__ import annotations

from verity_capture.experimental.chunked_prefill_rules import candidate_G5, discarded_sample_rule, request_steps

RID = "97f3c5d4-16d8-5c2e-9ead-2d9396d2e8d6"
SERVED = [560, 1272, 30, 198, 198, 57, 457, 1129, 8679, 418, 650, 5710, 282, 5914, 28]
ATTR = {RID: {"partial_prefill_steps": [0, 1], "first_token_step": 2, "token_steps": list(range(2, 17))}}
DRAWS = {0: 2093, 1: 339, **{s: t for s, t in zip(range(2, 17), SERVED)}}      # (a)'s SampleEnd id per step (0, 1 discarded by the engine)


def _rows(n=17):
    return [{"step": s, "sampled_eq": False if s < 2 else True} for s in range(n)]


def _a(draws=DRAWS):
    return {s: [t] for s, t in draws.items()}, {s: [RID] for s in draws}


def _b(served=SERVED, first=2):
    return {s: {RID: [t]} for s, t in zip(range(first, first + len(served)), served)}


def test_chunked_record_discards_exactly_the_partial_prefill_draws_and_accounts_for_every_served_token():
    ids, rows = _a()
    out, problems, n, acc = discarded_sample_rule(_rows(), ids, rows, _b(), ATTR, {RID: SERVED})
    assert problems == [] and n == 2 and acc == {RID: True}
    assert [r["sampled_eq"] for r in out] == [True] * 17
    assert out[0]["discarded_samples"] == {RID: 2093} and out[1]["discarded_samples"] == {RID: 339} and out[2]["discarded_samples"] == {}


def test_unchunked_record_is_the_frozen_comparison():
    draws = {s: t for s, t in zip(range(15), SERVED)}
    ids, rows = _a(draws)
    attr = {RID: {"partial_prefill_steps": [], "first_token_step": 0, "token_steps": list(range(15))}}
    out, problems, n, acc = discarded_sample_rule(_rows(15), ids, rows, _b(first=0), attr, {RID: SERVED})
    assert problems == [] and n == 0 and acc == {RID: True} and all(r["sampled_eq"] for r in out)
    # and with no attribution at all (a record that predates chunk_attribution) the rule is inert too
    out, problems, n, acc = discarded_sample_rule(_rows(15), ids, rows, _b(first=0), {}, {RID: SERVED})
    assert problems == [] and n == 0 and all(r["sampled_eq"] for r in out)


def test_a_served_token_at_a_partial_prefill_step_is_a_problem():
    ids, rows = _a()
    b = _b()
    b[1] = {RID: [339]}                                          # (b) claims a served token where the attribution says the prompt was incomplete
    _, problems, _, _ = discarded_sample_rule(_rows(), ids, rows, b, ATTR, {RID: SERVED})
    assert any("served token in (b) at one of its partial-prefill steps" in p for p in problems)


def test_a_disagreement_at_a_served_step_still_fails():
    ids, rows = _a()
    b = _b()
    b[5] = {RID: [999]}
    out, problems, _, _ = discarded_sample_rule(_rows(), ids, rows, b, ATTR, {RID: SERVED})
    assert out[5]["sampled_eq"] is False and any(p.startswith("step 5: SampleEnd ids by request") for p in problems)


def test_kept_draws_must_be_exactly_the_served_sequence_at_the_attributed_steps():
    ids, rows = _a()
    _, problems, _, acc = discarded_sample_rule(_rows(), ids, rows, _b(), ATTR, {RID: SERVED[:-1] + [7]})
    assert acc == {RID: False} and any("!= (a) generated_token_ids" in p for p in problems)
    attr = {RID: {**ATTR[RID], "token_steps": list(range(3, 18))}}          # attribution disagrees with where the draws were kept
    _, problems, _, acc = discarded_sample_rule(_rows(), ids, rows, _b(), attr, {RID: SERVED})
    assert acc == {RID: False} and any("!= attributed token_steps" in p for p in problems)


def test_missing_request_names_leave_the_row_untouched():
    ids, rows = _a()
    rows = {}                                                     # positional (a) sample rows: the rule cannot apply
    out, problems, n, _ = discarded_sample_rule(_rows(), ids, rows, _b(), ATTR, {RID: SERVED})
    assert n == 0 and out[0]["sampled_eq"] is False and "discarded_samples" not in out[0] and problems == []


def test_request_steps_reads_only_attributed_requests():
    doc = {"timing": {"requests": {RID: {"partial_prefill_steps": [0, 1], "first_token_step": 2, "token_steps": [2, 3]},
                                   "legacy": {"arrive_step": 0}}}}
    assert request_steps(doc) == {RID: {"partial_prefill_steps": [0, 1], "first_token_step": 2, "token_steps": [2, 3]}}
    assert request_steps(None) == {} and request_steps({"timing": {}}) == {}


def _g5(reason):
    return {"id": "G5", "name": "composition exactness", "status": "fail" if reason else "pass", "reason": reason}


def test_G5_first_token_rule_replaces_only_the_step0_problem():
    rec = _g5("served tokens: step 0 (prefill) not checked")
    tokens = {"timing": {"requests": {RID: {"partial_prefill_steps": [0, 1], "first_token_step": 2, "token_steps": list(range(2, 17))}}}}
    assert candidate_G5(rec, tokens, {"tokens": {"steps": [2, 7, 12]}})["status"] == "pass"
    r = candidate_G5(rec, tokens, {"tokens": {"steps": [3, 7, 12]}})              # the completion step is not among the checked steps
    assert r["status"] == "fail" and "first_token_step not checked" in r["reason"]
    # other G5 problems survive untouched
    rec2 = _g5("served tokens: step 0 (prefill) not checked; tier A: 3/575 mismatches")
    r = candidate_G5(rec2, tokens, {"tokens": {"steps": [2, 7]}})
    assert r["status"] == "fail" and r["reason"] == "tier A: 3/575 mismatches"


def test_G5_without_attribution_keeps_the_recorded_frozen_result():
    rec = _g5("served tokens: step 0 (prefill) not checked")
    r = candidate_G5(rec, None, {"tokens": {"steps": [3, 7]}})
    assert (r["status"], r["reason"], r["candidate"]["applied"]) == ("fail", rec["reason"], False)
    assert candidate_G5(_g5(None), {"timing": {}}, None)["status"] == "pass"
