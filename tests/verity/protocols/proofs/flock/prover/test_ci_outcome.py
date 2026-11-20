"""`ci.outcome`, the agreement job's one line: it says how many sessions the Lean verifier accepts beside the sessions both
verifiers accept and refuse as compared, so a run whose agreement is on refusals alone says so in that line."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

AGREEMENT = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "prover" / "agreement"


@pytest.fixture
def ci(monkeypatch):
    monkeypatch.syspath_prepend(str(AGREEMENT))
    import ci as mod

    return mod


def _set(i, sessions, passed, both_accept, both_refuse, upstream_alone):
    return {"set": i, "sessions": sessions, "agree": sessions, "disagree": [], "accepted": 0, "passed_every_check": passed,
            "both_accept": both_accept, "both_refuse": both_refuse, "upstream_alone_accepts": upstream_alone,
            "lean_alone_accepts": 0}


def test_the_line_says_no_session_is_accepted_beside_what_both_refuse(ci):
    results = [_set(0, 21, 3, 3, 17, 1), _set(1, 2, 1, 1, 0, 1), _set(2, 27, 3, 3, 22, 2),
               {"set": 3, "pr83": "x", "skipped": "no upstream binary for this circuit prover commit"}]
    assert ci.outcome(results) == (
        "ci: 3 of 4 sets agree: [0, 1, 2]: 0 sessions the Lean verifier accepts; as compared, 7 both accept (7 passed every "
        "Lean check and were refused last), 39 both refuse, 4 upstream alone accepts, 0 the Lean verifier alone accepts")


def test_agree_counts_sessions_by_the_two_compared_verdicts(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(AGREEMENT))
    import agree

    why = "refused: a session without a draw"
    sessions = {"honest": ({"accepted": False, "why": why + ": D12"}, True),
                "forged": ({"accepted": False, "why": "lincheck: ConsistencyFailed"}, False)}
    dirs = []
    for name in sessions:
        (tmp_path / name).mkdir()
        dirs.append(str(tmp_path / name))
    monkeypatch.setattr(agree, "verdicts", lambda a, slots: (
        {str(tmp_path / n): L for n, (L, _) in sessions.items()},
        {str(tmp_path / n): {"accepted": U, "why": ""} for n, (_, U) in sessions.items()}))
    out = tmp_path / "out"
    rc = agree.main(["--lean", "L", "--upstream", "U", "--circuit", "c", "--public", "p", "--program", "g",
                     "--lean-refuses-drawless", why, "--out", str(out), "--sessions", *dirs], quiet=True)
    r = json.loads((out / "result.json").read_text())
    assert rc == 0
    assert {k: r[k] for k in ("accepted", "passed_every_check", "both_accept", "both_refuse", "upstream_alone_accepts",
                              "lean_alone_accepts")} == {"accepted": 0, "passed_every_check": 1, "both_accept": 1,
                                                         "both_refuse": 1, "upstream_alone_accepts": 0,
                                                         "lean_alone_accepts": 0}
