"""PoUB's Lean answers are read only where the verifier wrote no panic and exited 0, and it starts with the runtime's panic
switch: the P2 case and the period case (`serve`), the session's verifier (`lean_verifier.Lean`) and the P2 party's
(`p2_lean.Lean`). A stand-in that answers accept beside a panic line is refused by name, and one that exits non-zero with
an accept on its output stream is refused."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity import lean_process as LP
from verity.protocols.resource_accounting.memory.poub.verifier import p2_case, p2_lean, period
from verity.protocols.resource_accounting.memory.poub.verifier.session import lean_verifier

PANIC_LINE = "PANIC at Poub.P2Protocol Poub.P2Protocol:1:2: index out of bounds"
ACCEPT = {"verdict": "accept"}


def _fake(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return exe


def _one_shot(panic: bool, exit_code: int = 0) -> str:
    return (f"print({PANIC_LINE!r}, file=sys.stderr)\n" if panic else "") + \
        f"print(json.dumps({ACCEPT!r}))\nsys.exit({exit_code})\n"


def _served(panic: bool) -> str:
    return f"""for line in sys.stdin:
    {'print(%r, file=sys.stderr, flush=True)' % PANIC_LINE if panic else 'pass'}
    print(json.dumps({{"answer": {ACCEPT!r}}}), flush=True)
"""


@pytest.mark.parametrize("serve", [p2_case.serve, period.serve], ids=["p2_case", "period"])
def test_an_accept_beside_a_panic_is_refused_by_name_whatever_the_exit(tmp_path, serve):
    for code in (0, 1):
        with pytest.raises(LP.LeanPanic) as e:
            serve(_fake(tmp_path, _one_shot(True, code)), [{}])
        assert str(e.value) == f"{LP.LEAN_PANIC}: {PANIC_LINE}"


@pytest.mark.parametrize("serve", [p2_case.serve, period.serve], ids=["p2_case", "period"])
def test_an_accept_from_a_process_that_exits_non_zero_is_refused(tmp_path, serve):
    with pytest.raises(subprocess.CalledProcessError):
        serve(_fake(tmp_path, _one_shot(False, 3)), [{}])


@pytest.mark.parametrize("serve", [p2_case.serve, period.serve], ids=["p2_case", "period"])
def test_the_case_is_answered_by_a_process_started_with_the_panic_switch(tmp_path, monkeypatch, serve):
    monkeypatch.setenv(LP.ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, "print(json.dumps({'switch': os.environ.get('LEAN_ABORT_ON_PANIC')}))\n")
    assert serve(exe, [{}]) == [{"switch": "1"}]


def test_the_sessions_verifier_refuses_an_answer_beside_a_panic_by_name(tmp_path):
    lean = lean_verifier.Lean(_fake(tmp_path, _served(True)), lambda _: None)
    try:
        with pytest.raises(lean_verifier.LeanError) as e:
            lean.call({"op": "x"})
        assert str(e.value) == f"{LP.LEAN_PANIC}: {PANIC_LINE}"
    finally:
        lean.close()


def test_the_p2_partys_verifier_refuses_an_answer_beside_a_panic_by_name(tmp_path):
    with p2_lean.Lean(_fake(tmp_path, _served(True))) as lean, pytest.raises(p2_lean.LeanError) as e:
        lean.call({"op": "x"})
    assert str(e.value) == f"{LP.LEAN_PANIC}: {PANIC_LINE}"


def test_both_verifiers_start_with_the_panic_switch_and_read_a_clean_answer(tmp_path, monkeypatch):
    monkeypatch.setenv(LP.ABORT_ON_PANIC, "0")
    body = """for line in sys.stdin:
    print(json.dumps({"answer": {"switch": os.environ.get("LEAN_ABORT_ON_PANIC")}}), flush=True)
"""
    exe = _fake(tmp_path, body)
    lean = lean_verifier.Lean(exe, lambda _: None)
    try:
        assert lean.call({"op": "x"}) == {"switch": "1"}
    finally:
        lean.close()
    with p2_lean.Lean(exe) as lean:
        assert lean.call({"op": "x"}) == {"switch": "1"}
