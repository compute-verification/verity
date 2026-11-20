"""The agreement's counts behind check's outcome line (`agree.py`'s result, `ci.py`'s sets): the sessions by the two
verdicts as compared, the Lean verifier's refusals by the check the vectors name, the dangerous direction and a panic each
failing by name, and the live sets check's step runs beside the replayable ones."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

AGREEMENT = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "prover" / "agreement"


@pytest.fixture
def agree(monkeypatch):
    monkeypatch.syspath_prepend(str(AGREEMENT))
    import agree as mod

    return mod


def _run(agree, tmp_path, monkeypatch, sessions, expect=(), extra=()):
    """`agree.main` on stand-in verdicts: `sessions` maps a name to (Lean's verdict, upstream accepts)."""
    dirs = []
    for name in sessions:
        (tmp_path / name).mkdir()
        dirs.append(str(tmp_path / name))
    monkeypatch.setattr(agree, "verdicts", lambda a, slots, panics: (
        {str(tmp_path / n): L for n, (L, _) in sessions.items()},
        {str(tmp_path / n): {"accepted": U, "why": ""} for n, (_, U) in sessions.items()}))
    (tmp_path / "expect.json").write_text(json.dumps([{"name": n, "expect": e} for n, e in expect]))
    out = tmp_path / "out"
    rc = agree.main(["--lean", "L", "--upstream", "U", "--circuit", "c", "--public", "p", "--program", "g", *extra,
                     "--expect", str(tmp_path / "expect.json"), "--out", str(out), "--sessions", *dirs], quiet=True)
    return rc, json.loads((out / "result.json").read_text())


def test_agree_counts_sessions_by_the_two_compared_verdicts_and_refusals_by_their_target(agree, monkeypatch, tmp_path):
    why = "refused: a session without a draw"
    rc, r = _run(agree, tmp_path, monkeypatch, {
        "honest": ({"accepted": False, "why": why + ": D12"}, True),
        "forged": ({"accepted": False, "why": "lincheck: ConsistencyFailed"}, False),
        "named": ({"accepted": False, "why": "S12: retained bytes"}, False)},
        expect=[("named", {"accepted": False, "why": "S12:"})], extra=["--lean-refuses-drawless", why])
    assert rc == 0
    assert {k: r[k] for k in ("accepted", "passed_every_check", "both_accept", "both_refuse", "upstream_alone_accepts",
                              "lean_alone_accepts", "refused_at_target", "refused_elsewhere", "refused_untargeted",
                              "lean_panics")} == {
        "accepted": 0, "passed_every_check": 1, "both_accept": 1, "both_refuse": 2, "upstream_alone_accepts": 0,
        "lean_alone_accepts": 0, "refused_at_target": 1, "refused_elsewhere": 0, "refused_untargeted": 1, "lean_panics": []}


def test_the_lean_verifier_alone_accepting_fails_whatever_the_expectation_says(agree, monkeypatch, tmp_path):
    rc, r = _run(agree, tmp_path, monkeypatch, {"forged": ({"accepted": True, "why": ""}, False)},
                 expect=[("forged", {"accepted": True, "why": "", "upstream_accepted": False})])
    assert rc == 1 and r["disagree"] == ["forged"] and r["lean_alone_accepts"] == 1


def test_a_lean_verifier_that_panics_and_refuses_fails_by_name(agree, tmp_path):
    """A stand-in verifier that prints a panic's message, refuses and exits 0: its refusal is never an agreement on refusal."""
    lean = tmp_path / "fake-lean"
    lean.write_text(f"#!{sys.executable}\nimport json, sys\nprint('Error: index out of bounds', file=sys.stderr)\n"
                    "for s in sys.argv[sys.argv.index('--session') + 1:]:\n"
                    "    print('VERDICT ' + json.dumps({'session': s, 'accepted': False, 'why': 'S2: refused'}))\n")
    lean.chmod(0o755)
    s = tmp_path / "s"
    s.mkdir()
    (s / "case.json").write_text(json.dumps({"expect": "reject", "live": {"accepted": False, "why": "R2"}}))
    for f in ("c", "p", "g"):
        (tmp_path / f).write_text(f)
    out = tmp_path / "out"
    rc = agree.main(["--lean", str(lean), "--upstream", agree.LIVE, "--circuit", str(tmp_path / "c"), "--public",
                     str(tmp_path / "p"), "--program", str(tmp_path / "g"), "--out", str(out), "--sessions", str(s)],
                    quiet=True)
    r = json.loads((out / "result.json").read_text())
    assert rc == 1 and r["lean_panics"] == ["s"] and r["disagree"] == ["s"] and r["both_refuse"] == 0


def test_checks_run_adds_the_live_sets_marked_check_and_names_their_second_verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(AGREEMENT))
    import ci

    v = json.loads((AGREEMENT / "vectors.json").read_text())
    picked = ci.picked(v, SimpleNamespace(live=False, sets=None))
    live = [(i, s) for i, s, on in picked if on]
    assert [i for i, _s, on in picked if not on] == list(range(len(v["replayable"]["sets"])))
    assert live and all(str(i).startswith("live") and s.get("check") for i, s in live)
    assert all(not on for _i, _s, on in ci.picked(v, SimpleNamespace(live=False, sets="0")))
    assert [i for i, _s, _on in ci.picked(v, SimpleNamespace(live=True, sets="6"))] == [6]
    assert ci.SECOND[True] == "our live server's recorded verdicts" and ci.SECOND[False] == "upstream's pinned verifier"
