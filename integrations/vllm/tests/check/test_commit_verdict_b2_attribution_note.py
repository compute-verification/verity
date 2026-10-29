"""A PASS at B>=2 without a complete request attribution on the row's own record NAMES the gap.

Two B8 256/32 Commit-only reruns (llama L40S, llama A100) restored the Program but had no Match record, so form (B) was not armed;
commit_verdict said a bare 'PASS' and the report rendered a Commit checkmark.  Form (B) is NOT made mandatory: the verdict is unchanged;
the PASS text at B>=2 carries the attribution gap by name when the oracle compare is absent or its attribution block is not complete --
and carries nothing at B=1 (the class does not exist) or when attribution is complete (value-partial by snapshot gaps is not this).  A
FAIL of any earlier rule is unchanged.
"""
from __future__ import annotations

from verity_vllm.check.commit_pass import commit_verdict

WP_OK = {"result": True, "why": None, "root_of_record": "ab" * 32, "compared": 163, "mismatched": [], "missing": [], "unpinned": []}
VC_OK = {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True}


def _base(n_req: int) -> dict:
    bnd = {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}
    return {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "binding": bnd,
            "sampled_replay": {"result": True, "grade": "complete"}, "boundary_linkage": {"result": True, "pairs": 6, "linked": 6},
            "value_correspondence": VC_OK, "weights_pin": WP_OK}


def _oc(complete: bool, unverified: int = 0) -> dict:
    return {"result": True, "grade": "complete" if complete else "partial", "compared": 25216, "equal": 25216, "mismatch_n": 0, "complete": True,
            "attribution": {"complete": complete, "pairs": 128, "unverified_n": unverified, "ambiguous_steps": []}}


def test_b1_carries_no_attribution_note_with_or_without_form_b():
    ok, reason = commit_verdict(_base(1), True, min_openings=64)
    assert ok is True and reason == "PASS"
    ok, reason = commit_verdict({**_base(1), "oracle_compare": _oc(True)}, True, min_openings=64)
    assert ok is True and "attribution" not in reason


def test_b8_without_form_b_passes_with_the_gap_named():
    ok, reason = commit_verdict(_base(8), True, min_openings=64)
    assert ok is True, reason                                                          # not mandatory: still a PASS
    assert reason.startswith("PASS (") and "B=8: per-(step,request) attribution not verified on this record" in reason
    assert "form (B) not armed on this attempt (no --match-dir oracle compare: no Match record)" in reason
    assert "equal-shape request swap" in reason


def test_b2_with_complete_attribution_passes_clean_even_when_value_partial():
    oc = _oc(True); oc["complete"] = False; oc["grade"] = "partial"; oc["steps_not_snapshotted"] = 32       # the snapshot-gap class
    ok, reason = commit_verdict({**_base(2), "oracle_compare": oc}, True, min_openings=64)
    assert ok is True and "attribution" not in reason, reason


def test_b2_with_incomplete_attribution_passes_with_the_counts_named():
    ok, reason = commit_verdict({**_base(2), "oracle_compare": _oc(False, unverified=3)}, True, min_openings=64)
    assert ok is True and "B=2: per-(step,request) attribution not verified" in reason
    assert "form (B) request attribution not complete (pairs 128, unverified 3, ambiguous steps 0)" in reason


def test_earlier_fails_are_unchanged_and_named_first():
    rec = _base(8)
    rec["weights_pin"] = {**WP_OK, "result": False, "why": "1 served weight(s) differ from the checkpoint of record: model.layers.0.mlp.down_proj.weight"}
    ok, reason = commit_verdict(rec, True, min_openings=64)
    assert ok is False and reason.startswith("weights of record: ") and "attribution" not in reason
    rec = _base(8)
    rec["sampled_replay"] = {"result": None, "grade": "partial", "why": "replay not run", "sample": {"evaluated": 0, "mismatch_n": 0}}
    ok, reason = commit_verdict(rec, True, min_openings=64)
    assert ok is False and reason.startswith("C2 of record not established") and "attribution not verified" not in reason
    rec = _base(8)
    rec["boundary_linkage"] = {"result": False, "why": "1 boundary unlinked"}
    ok, reason = commit_verdict(rec, True, min_openings=64)
    assert ok is False and "boundary linkage" in reason and "attribution not verified" not in reason
