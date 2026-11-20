"""`one_linear_arms.py`, which `one_linear.sh` gates each audit arm on: a crash, a panic or an undocumented status is an
ERROR and fails its arm, a negative passes only on its refusal by name, and the honest arm's reject stays a reject, named
at the record's failed check. `accept_line.py` prints that refusal as its one line, and on an accept its lines without C."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ARMS = Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage" / "one_linear_arms.py"
WINDOW = 'STEP {"step": "window", "units": 288, "blocks": 144, "law": {"law": "subset", "k": 4}}\n'
TRACEBACK = ("Traceback (most recent call last):\n  File \"one_linear.py\", line 152, in window\n"
             "ValueError: a law stated as text is a rate, not 'stratified-1/72'\n+ exit 5\n")
NOT_OK = {"ok": False, "refused": None, "audit": None, "draw": None}
SUBSET = ("the sampling design is stratified, which the one-statement audit's soundness statement doesn't cover: it "
          "covers a subset sampling design")

#: arm, status, log, summary (None: none written), then the line it prints and its exit
CASES = {
    "stratified crash (silent before)": ("stratified", 5, WINDOW + TRACEBACK, None,
                                         "ARM stratified ERROR a crash: Traceback (most recent call last):", 1),
    "forged-one crash, status 1": ("forged-one", 1, WINDOW + TRACEBACK.replace("+ exit 5\n", ""), NOT_OK,
                                   "ARM forged-one ERROR a crash: Traceback (most recent call last):", 1),
    "registered panic": ("registered", 1, WINDOW + "PANIC at Verity.x Verity.X:1:2: oops\n", NOT_OK,
                         "ARM registered ERROR a crash: PANIC at Verity.x", 1),
    "status 3": ("registered", 3, WINDOW, NOT_OK,
                 "ARM registered ERROR status 3, which records_audit.py audit does not document", 1),
    "no summary": ("forged-drawn", 1, WINDOW, None, "ARM forged-drawn ERROR status 1 and no summary.json", 1),
    "honest reject": ("honest", 1, WINDOW, NOT_OK, "ARM honest REJECT status 1, not ACCEPT by its name", 1),
    "honest refused at roots, a wrong port": (
        "honest", 1, WINDOW, {"ok": False, "audit": {"accepted": False, "checks_failed": {"roots": {"ports": ["y"]}}}},
        'ARM honest REJECT status 1, not ACCEPT by its name: at roots: {"roots": {"ports": ["y"]}}', 1),
    "honest accept": ("honest", 0, WINDOW, {"ok": True, "audit": {"accepted": True}, "draw": [3, 9, 40, 100]},
                      "ARM honest ACCEPT status 0, summary ok", 0),
    "registered refused by no name": ("registered", 1, WINDOW, NOT_OK,
                                      "ARM registered REJECT status 1, not REFUSED by its name", 1),
    "registered refused by the members check": (
        "registered", 1, WINDOW, {"ok": False, "audit": {"members_lean": {"ran": True, "ok": False, "reason": "port w"}}},
        "ARM registered REFUSED the members check: port w", 0),
    "registered refused by R8 before the draw": (
        "registered", 1, WINDOW, {"ok": False, "checks": {"registration": {
            "ok": False, "codes": ["R8-shared-input"], "detail": {"R8-shared-input": {"inputs": [0]}}}}},
        'ARM registered REFUSED before the draw, a completeness limit: R8-shared-input [0]', 0),
    "registered refused by R8, read off a5's line": (
        "registered", 1, WINDOW + 'A5 {"refused_before_draw": {"registration": false}, "checks": {"registration": '
        '{"ok": false, "codes": ["R8-shared-input"], "detail": {"R8-shared-input": {"inputs": [0]}}}}}\n',
        {"ok": False, "checks_before_draw": {"registration": False}},
        'ARM registered REFUSED before the draw, a completeness limit: R8-shared-input [0]', 0),
    "registered refused by R8 and every member's header, as a5 prints it, after the arm log's cut copy": (
        "registered", 1, WINDOW + 'A5 {"refused_before_draw": {"m0": false, "registration": false}, "che\n'
        'A5 {"refused_before_draw": {"m0": false, "registration": false}, "checks": {'
        '"m0": {"ok": false, "codes": ["header"], "detail": {"header": "public file: port w has no tree under the member"}}, '
        '"registration": {"ok": false, "codes": ["R8-shared-input"], "detail": {"R8-shared-input": {"inputs": [0]}}}, '
        '"source": {"ok": true}}}\n', {"ok": False, "checks_before_draw": {"m0": False, "registration": False}},
        "ARM registered REFUSED before the draw, a completeness limit: R8-shared-input [0]; 1 members at header: "
        "{'header': 'public file: port w has no tree under the member'}", 0),
    "registered refused before the draw by another check": (
        "registered", 1, WINDOW, {"ok": False, "checks": {"registration": {"ok": False, "codes": ["R3-root"]}}},
        "ARM registered REJECT status 1, not REFUSED by its name", 1),
    "stratified refused by SUBSET_ONLY": ("stratified", 1, WINDOW, {"ok": False, "refused": {"params": SUBSET}},
                                          "ARM stratified REFUSED SUBSET_ONLY: the sampling design is stratified", 0),
    "forged-drawn by forged_outcome": ("forged-drawn", 0, WINDOW,
                                       {"ok": True, "forged": {"ok": True, "passes_on": "no_proof_released"}},
                                       "ARM forged-drawn REFUSED forged_outcome passes on no_proof_released", 0),
    "forged-one not drawn": ("forged-one", 0, WINDOW, {"ok": True, "draw": [3, 9, 40, 100]},
                             "ARM forged-one ACCEPT the forged unit's block 0 is not drawn", 0),
    "forged-one accepted, drawn": ("forged-one", 0, WINDOW, {"ok": True, "draw": [1, 9, 40, 100]},
                                   "ARM forged-one FAIL accepted with the forged unit's block 0 drawn", 1),
    "forged-one rejected, drawn": ("forged-one", 1, WINDOW, {"ok": False, "audit": {"accepted": False, "why": ["m0"]},
                                                             "draw": [0, 9, 40, 100]},
                                   "ARM forged-one REFUSED the audit record rejects", 0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_crash_is_an_error_a_negative_passes_by_its_name_and_an_honest_reject_stays_a_reject(tmp_path, case):
    arm, status, log, summary, line, code = CASES[case]
    (tmp_path / "arm.log").write_text(log)
    if summary is not None:
        (tmp_path / "summary.json").write_text(json.dumps(summary))
    p = subprocess.run([sys.executable, str(ARMS), arm, str(status), str(tmp_path / "arm.log"),
                        str(tmp_path / "summary.json")], capture_output=True, text=True, check=False)
    assert p.stdout.startswith(line), p.stdout + p.stderr
    assert p.returncode == code


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


def test_a_traceback_past_the_logs_tail_is_found_in_the_whole_audit_err(tmp_path):
    """The arm's log holds the last 3000 bytes of its audit's stderr, which can start after a traceback's header: the
    judge reads the whole stderr it is given, so a crash whose header only the whole file holds is still an ERROR."""
    err = TRACEBACK.replace("+ exit 5\n", "") + "x" * 4000 + "\n"
    (tmp_path / "arm.log").write_text(WINDOW + err[-3000:])
    (tmp_path / "audit.err").write_text(err)
    (tmp_path / "summary.json").write_text(json.dumps(NOT_OK))
    argv = [sys.executable, str(ARMS), "forged-one", "1", str(tmp_path / "arm.log"), str(tmp_path / "summary.json")]
    tail = subprocess.run(argv, capture_output=True, text=True, check=False)
    whole = subprocess.run([*argv, str(tmp_path / "audit.err"), str(tmp_path / "missing.txt")], capture_output=True,
                           text=True, check=False)
    assert tail.stdout.startswith("ARM forged-one REJECT") and tail.returncode == 1
    assert whole.stdout.startswith("ARM forged-one ERROR a crash: Traceback (most recent call last):"), whole.stdout
    assert whole.returncode == 1
