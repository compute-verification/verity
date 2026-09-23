"""[r8 lane E-cleanup 2026-09-18] unit tests for the CANDIDATE prefix-closure G3 rule (`verity_capture.experimental.prefix_g3_rules`).

The rule is a pure function of `patterns_prefix.obligations`' run summary and the recorded G3 result: it must be inert on cold runs and
on closed warm runs, add (never remove) problems on external-state / alias-violated / undecided linkages, and keep recorded problems."""
from __future__ import annotations

from verity_capture import patterns_prefix as P
from verity_capture.experimental.prefix_g3_rules import candidate_G3, prefix_closure_problems

CONS, PROD = "21c55add-consumer", "e1b4efc4-producer"


def _linkage(*, warm=True, writer=True, identical=True, decided=True, n=3):
    """A synthetic linkage record with one producer (cold) and one consumer reusing `n` slots (the schema patterns_prefix reads)."""
    slots = []
    for p in range(n):
        s = {"position": p, "slot": 100 + p, "token_identical": (identical if decided else None), "position_identical": True}
        s["writer"] = {"request_id": PROD, "step": 0, "row": p, "token": 7 + p} if writer else None
        slots.append(s)
    return {"requests": {PROD: {"warm": False, "prompt_rows_at_first_step": 40},
                         CONS: {"warm": warm, "first_step": 4, "num_computed_at_first_step": n if warm else 0, "prompt_rows_at_first_step": 40,
                                "declared_prompt_len": n + 40, "producers": [{"request_id": PROD, "step": 0}],
                                "reused_blocks": ([{"physical_block": 9, "slots": slots}] if warm else [])}}}


def _g3(reason=None):
    return {"id": "G3", "name": "root closure", "status": "fail" if reason else "pass", "reason": reason}


def test_cold_and_closed_runs_are_unchanged():
    for lk in (_linkage(warm=False), _linkage()):
        ob = P.obligations(lk)
        assert prefix_closure_problems(ob) == []
        r = candidate_G3(_g3(), ob)
        assert (r["status"], r["reason"]) == ("pass", None)
    assert candidate_G3(_g3(), P.obligations(_linkage(warm=False)))["candidate"]["applied"] is False
    assert candidate_G3(_g3(), P.obligations(_linkage()))["candidate"]["applied"] is True


def test_external_state_fails_G3():
    ob = P.obligations(_linkage(writer=False))
    assert ob["run"]["status"] == "external-state"
    r = candidate_G3(_g3(), ob)
    assert r["status"] == "fail" and "writer is outside the log" in r["reason"] and "cached_kv" in r["reason"]


def test_alias_violation_and_undecided_alias_fail_G3():
    r = candidate_G3(_g3(), P.obligations(_linkage(identical=False)))
    assert r["status"] == "fail" and "3 alias obligation(s) failed" in r["reason"]
    r = candidate_G3(_g3(), P.obligations(_linkage(decided=False)))
    assert r["status"] == "fail" and "undecidable" in r["reason"] and "unproven alias claims" in r["reason"]


def test_recorded_problems_are_kept_and_never_removed():
    rec = _g3("fold: 2 unknown roots: ['x', 'y']")
    r = candidate_G3(rec, P.obligations(_linkage()))                # closed: nothing added, the recorded failure stands
    assert (r["status"], r["reason"]) == ("fail", rec["reason"])
    r = candidate_G3(rec, P.obligations(_linkage(writer=False)))
    assert r["status"] == "fail" and r["reason"].startswith(rec["reason"] + "; prefix:")


def test_problem_texts_are_specific_to_the_defect():
    ob = P.obligations(_linkage(writer=False))
    assert prefix_closure_problems(ob) == ["prefix: 3 reused KV slot(s) whose writer is outside the log (no same-run state edge; a committed `cached_kv` Input is required)"]
    assert prefix_closure_problems({"run": {"status": "cold"}}) == [] and prefix_closure_problems({}) == []
