"""PoUS's Lean answers (`period.serve`, `timed_audit.run_lean`) are read only where the verifier wrote no panic and exited
0 (`run_lean`: 0 or 1, its verdict's own), and it starts with the runtime's panic switch. A stand-in that answers accept
beside a panic line is refused by name, and one that exits non-zero with an accept on its output stream is refused."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity import lean_process as LP
from verity.protocols.resource_accounting.memory.pous.attester import timed_audit as TA
from verity.protocols.resource_accounting.memory.pous.verifier import period

PANIC_LINE = "PANIC at PousVerifier.RunProtocol PousVerifier.RunProtocol:1:2: index out of bounds"
ACCEPT = json.dumps({"verdict": "accept"})


def _fake(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport os, sys\n{body}")
    exe.chmod(0o755)
    return exe


def _accepting(panic: bool, exit_code: int = 0) -> str:
    return (f"print({PANIC_LINE!r}, file=sys.stderr)\n" if panic else "") + f"print({ACCEPT!r})\nsys.exit({exit_code})\n"


def test_an_accept_beside_a_panic_is_refused_by_name_whatever_the_exit(tmp_path):
    for code in (0, 1):
        exe = _fake(tmp_path, _accepting(True, code))
        with pytest.raises(LP.LeanPanic) as e:
            period.serve(exe, [{"commitment": "", "evidence": "", "params": {}}])
        assert str(e.value) == f"{LP.LEAN_PANIC}: {PANIC_LINE}"


def test_an_accept_from_a_process_that_exits_non_zero_is_refused(tmp_path):
    exe = _fake(tmp_path, _accepting(False, 3))
    with pytest.raises(LP.LeanExit, match="^lean-exit: .*status 3"):
        period.serve(exe, [{}])


def test_the_verdict_by_the_timed_audit_is_refused_beside_a_panic_and_after_a_failure(tmp_path):
    with pytest.raises(LP.LeanPanic, match=f"^{LP.LEAN_PANIC}: {PANIC_LINE}$"):
        TA.run_lean(_fake(tmp_path, _accepting(True)), {}, b"", {})
    with pytest.raises(LP.LeanExit, match="^lean-exit: .*status 3"):
        TA.run_lean(_fake(tmp_path, _accepting(False, 3)), {}, b"", {})
    assert TA.run_lean(_fake(tmp_path, _accepting(False, 1)), {}, b"", {}) == {"verdict": "accept"}


def test_the_answers_are_read_from_a_process_started_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(LP.ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, "import json\nprint(json.dumps({'switch': os.environ.get('LEAN_ABORT_ON_PANIC')}))\n")
    assert LP.ABORT_ON_PANIC == "LEAN_ABORT_ON_PANIC"
    assert period.serve(exe, [{}]) == [{"switch": "1"}]
    assert TA.run_lean(exe, {}, b"", {}) == {"switch": "1"}


def test_a_warning_on_the_error_stream_is_no_panic(tmp_path):
    exe = _fake(tmp_path, "print('warning: slow', file=sys.stderr)\n" + _accepting(False))
    assert period.serve(exe, [{}]) == [{"verdict": "accept"}]
