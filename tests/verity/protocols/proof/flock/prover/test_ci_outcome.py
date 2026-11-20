"""The fields behind check's outcome line (`agree.py`'s result, `ci.py`'s sets): what both refuse, what the second verifier
alone accepts with its reason, the Lean verifier's refusals by the check the vectors name, the dangerous direction and a Lean
run `lean_run` refuses each failing by name, and the live sets check's step runs beside the replayable ones."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

AGREEMENT = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "prover" / "agreement"


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
    monkeypatch.setattr(agree, "verdicts", lambda a, slots: (
        {str(tmp_path / n): L for n, (L, _) in sessions.items()},
        {str(tmp_path / n): {"accepted": U, "why": ""} for n, (_, U) in sessions.items()}))
    (tmp_path / "expect.json").write_text(json.dumps([{"name": n, "expect": e} for n, e in expect]))
    out = tmp_path / "out"
    rc = agree.main(["--lean", "L", "--upstream", "U", "--circuit", "c", "--public", "p", "--program", "g", *extra,
                     "--expect", str(tmp_path / "expect.json"), "--out", str(out), "--sessions", *dirs], quiet=True)
    return rc, json.loads((out / "result.json").read_text())


def test_agree_names_its_fields_and_counts_no_session_refused_last_as_an_accept(agree, monkeypatch, tmp_path):
    why = "refused: a session without a draw"
    rc, r = _run(agree, tmp_path, monkeypatch, {
        "honest": ({"accepted": False, "why": why + ": D12"}, True),
        "forged": ({"accepted": False, "why": "lincheck: ConsistencyFailed"}, False),
        "named": ({"accepted": False, "why": "S12: retained bytes"}, False),
        "root": ({"accepted": False, "why": "setup: registered reads: port cs"}, True)},
        expect=[("named", {"accepted": False, "why": "S12:"}),
                ("root", {"accepted": False, "why": "setup: registered reads", "upstream_accepted": True})],
        extra=["--lean-refuses-drawless", why, "--second-alone-reason", "the live server reads no registered roots"])
    assert rc == 0
    assert {k: r[k] for k in ("accepted", "passed_every_check", "both_refuse", "second_alone_accepts", "lean_alone_accepts",
                              "refused_at_named_check", "refused_at_another_check", "refused_where_none_named",
                              "lean_failed")} == {
        "accepted": 0, "passed_every_check": 1, "both_refuse": 2,
        "second_alone_accepts": [{"reason": "the live server reads no registered roots", "sessions": ["root"]}],
        "lean_alone_accepts": 0, "refused_at_named_check": 2, "refused_at_another_check": 0, "refused_where_none_named": 1,
        "lean_failed": []}
    assert "both_accept" not in r and "upstream_alone_accepts" not in r


def test_the_lean_verifier_alone_accepting_fails_whatever_the_expectation_says(agree, monkeypatch, tmp_path):
    rc, r = _run(agree, tmp_path, monkeypatch, {"forged": ({"accepted": True, "why": ""}, False)},
                 expect=[("forged", {"accepted": True, "why": "", "upstream_accepted": False})])
    assert rc == 1 and r["disagree"] == ["forged"] and r["lean_alone_accepts"] == 1


def test_a_lean_verifier_that_panics_and_refuses_fails_by_name(agree, tmp_path, capsys):
    """A stand-in verifier that prints a panic's line, refuses and exits 0: `lean_run` refuses its answer, so its refusal is
    never an agreement on refusal."""
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
    import lean_run as LR

    assert rc == 1 and r["lean_failed"] == [f"{LR.LEAN_PANIC}: Error: index out of bounds"] and r["disagree"] == ["s"]
    assert r["both_refuse"] == 0 and f"agree: FAILED: {LR.LEAN_PANIC}" in capsys.readouterr().err


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


def test_a_fetch_the_store_refuses_is_a_named_error(monkeypatch, tmp_path):
    """Without store access the step fails naming the fixture it could not fetch, never a skip."""
    monkeypatch.syspath_prepend(str(AGREEMENT))
    import ci

    monkeypatch.setattr(ci.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2, stderr="no store config\n"))
    with pytest.raises(SystemExit) as e:
        ci.fetch("art:70f12de7/vectors/honest", tmp_path)
    assert str(e.value) == "ci: error: cannot fetch art:70f12de7 from the evidence store (exit 2): no store config"
    assert not (tmp_path / "70f12de7").exists()
