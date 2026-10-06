"""The outer gate's agreement harness (`verifier/firewall_outer_agree.py`) on hand-made transcripts: a check's outcome is
recomputed from its affine form over the tape's pads, and a case's expectation holds only when the firewall stopped where
the control says and the contract cut at that item for the control's reason. The transcripts themselves come from
`flock-circuit`'s recorder (`live/src/gate_transcript.rs`), run by the outer agreement run."""
from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "verifier"


def _harness():
    spec = importlib.util.spec_from_file_location("firewall_outer_agree", VERIFIER / "firewall_outer_agree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OA = _harness()


def h(v: int) -> str:
    return f"{v:032x}"


def transcript(items, refused=None, at=None, stopped=None, pads=(0x1234, 0xBEEF << 70)):
    return {"tape": {"pads": [h(p) for p in pads]}, "items": items,
            "firewall": {"gate": {"refused": refused, "at": at}, "stopped": stopped, "recorder_errors": []}}


def check(ok, c, a):
    return {"check": {"relation": "r", "pads": [0], "ok": ok, "form": {"c": h(c), "a": [[i, h(x)] for i, x in a]}}}


def test_gmul_is_the_ghash_polynomials():
    assert OA.gmul(1 << 127, 2) == 0x87
    assert OA.gmul(3, 3) == 5


def test_a_checks_outcome_is_recomputed_from_its_form():
    p0, p1 = 0x1234, 0xBEEF << 70
    a = 7
    holds = OA.gmul(a, p0) ^ p1
    t = transcript([check(True, holds, [(0, a), (1, 1)]), check(False, holds ^ 1, [(0, a), (1, 1)]), check(True, 1, [(0, a)])])
    assert OA.recheck(t) == (2, 3)


def test_a_control_must_stop_where_the_firewall_did_and_for_its_reason():
    why = "rep 0 algebraic phase round 0 frame 1: label: not the value the statement fixes"
    t = transcript([{"fixed": {}}] * 10, refused="GATE-REFUSED: algebraic phase, round 0, frame 1: label \"glock\"", at=9)
    assert OA.judge("label", t, {"accepted": False, "released": 9, "why": why}) == (True, "")
    assert not OA.judge("label", t, {"accepted": False, "released": 8, "why": why})[0]
    assert not OA.judge("label", t, {"accepted": False, "released": 9, "why": "the session ended without Finish"})[0]
    assert not OA.judge("label", t, {"accepted": True, "released": 10, "why": None})[0]
    assert not OA.judge("root", t, {"accepted": False, "released": 9, "why": why})[0]


def test_the_prover_stopping_is_judged_at_the_first_failing_check():
    t = transcript([{"masked": {}}, check(True, 0, []), check(False, 1, []), check(False, 1, [])],
                   stopped="PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)")
    assert OA.judge("algebra", t, {"accepted": False, "released": 2, "why": "the relation r fails"}) == (True, "")
    assert not OA.judge("algebra", t, {"accepted": False, "released": 3, "why": "the relation r fails"})[0]
    assert not OA.judge("honest", t, {"accepted": True, "released": 4, "why": None})[0]


def test_an_honest_session_is_accepted_with_nothing_stopped():
    t = transcript([{"finish": {}}])
    assert OA.judge("honest", t, {"accepted": True, "released": 1, "why": None}) == (True, "")
    assert not OA.judge("honest", t, {"accepted": False, "released": 0, "why": "x"})[0]
