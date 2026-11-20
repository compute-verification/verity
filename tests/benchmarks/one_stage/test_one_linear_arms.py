"""`one_linear_arms.py`, which `one_linear.sh` gates its one honest audit on: it passes only on an accept, a crash is an
ERROR wherever its log holds it, and a reject stays a reject, named at the record's failed check or at the checks a5
refused before the draw. `accept_line.py`
prints that refusal as its one line, and on an accept its lines without C."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ARMS = Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage" / "one_linear_arms.py"
WINDOW = 'STEP {"step": "window", "units": 288, "blocks": 144, "law": {"law": "subset", "k": 4}}\n'
TRACEBACK = ("Traceback (most recent call last):\n  File \"one_linear.py\", line 152, in window\n"
             "ValueError: a law stated as text is a rate, not 'stratified-1/72'\n")
NOT_OK = {"ok": False, "refused": None, "audit": None, "draw": None}

#: status, log, summary, then the line it prints and its exit
CASES = {
    "accept": (0, WINDOW, {"ok": True, "audit": {"accepted": True}, "draw": [3, 9, 40, 100]},
               "ARM honest ACCEPT status 0, summary ok", 0),
    "status 0, not ok": (0, WINDOW, NOT_OK, "ARM honest ERROR status 0 with a summary that is not ok", 1),
    "reject": (1, WINDOW, NOT_OK, "ARM honest REJECT status 1: ", 1),
    "refused at roots, a wrong port": (
        1, WINDOW, {"ok": False, "audit": {"accepted": False, "checks_failed": {"roots": {"ports": ["y"]}}}},
        'ARM honest REJECT status 1: at roots: {"roots": {"ports": ["y"]}}', 1),
    "refused before the draw, read off a5's line after the arm log's cut copy": (
        1, WINDOW + 'A5 {"refused_before_draw": {"registration": false}, "che\n'
        'A5 {"refused_before_draw": {"registration": false}, "checks": {"registration": {"ok": false, "codes": '
        '["R8-shared-input"]}, "source": {"ok": true}}}\n', {"ok": False, "checks_before_draw": {"registration": False}},
        'ARM honest REJECT before the draw at {"registration": ["R8-shared-input"]}', 1),
}


@pytest.mark.parametrize("case", list(CASES))
def test_only_an_accept_passes_and_a_reject_is_named(tmp_path, case):
    status, log, summary, line, code = CASES[case]
    (tmp_path / "arm.log").write_text(log)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    p = subprocess.run([sys.executable, str(ARMS), "honest", str(status), str(tmp_path / "arm.log"),
                        str(tmp_path / "summary.json")], capture_output=True, text=True, check=False)
    assert p.stdout.startswith(line), p.stdout + p.stderr
    assert p.returncode == code


def test_a_crash_is_an_error_even_past_the_logs_tail_and_with_status_0(tmp_path):
    """The arm's log holds the last 3000 bytes of its audit's stderr, which can start after a traceback's header: the
    judge reads the whole stderr it is given, so a crash whose header only the whole file holds is still an ERROR, and
    a crash is never an accept."""
    err = TRACEBACK + "x" * 4000 + "\n"
    (tmp_path / "arm.log").write_text(WINDOW + err[-3000:])
    (tmp_path / "audit.err").write_text(err)
    (tmp_path / "summary.json").write_text(json.dumps({"ok": True, "audit": {"accepted": True}}))
    argv = [sys.executable, str(ARMS), "honest", "0", str(tmp_path / "arm.log"), str(tmp_path / "summary.json")]
    tail = subprocess.run(argv, capture_output=True, text=True, check=False)
    whole = subprocess.run([*argv, str(tmp_path / "audit.err"), str(tmp_path / "missing.txt")], capture_output=True,
                           text=True, check=False)
    assert tail.stdout.startswith("ARM honest ACCEPT") and tail.returncode == 0
    assert whole.stdout.startswith("ARM honest ERROR a crash: Traceback (most recent call last):"), whole.stdout
    assert whole.returncode == 1


def test_the_accept_line_names_the_roots_refusal_and_states_an_accept_without_c():
    sys.path.insert(0, str(ARMS.parent))
    import accept_line as AL
    refused = {"ok": False, "audit": {"accepted": False, "checks_failed": {"roots": {"ports": ["y"]}}}}
    assert AL.lines(refused, None, None) == ["NO ACCEPT " + AL.refusal(refused)]
    assert AL.refusal(refused).startswith('REJECT at roots: {"roots": {"ports": ["y"]}}')
    accepted = {"ok": True, "draw": [45, 56, 63, 210], "audit": {
        "accepted": True, "members_lean": {"ran": True, "ok": True, "codes": [], "detail": {"lean": {"verdict": "accept"}}},
        "wrong_units": {"of": 288, "bound": 45, "delta": 0.5}}}
    got = AL.lines(accepted, None, "1/2")
    assert got[1] == ("ACCEPT one window, 288 units, 4 drawn: except with probability 1/2, at most 45 of its 288 units "
                      "are in blocks that are not correct (no C given, so no AND-gate figure)")
    assert "record = command" in got[2]
