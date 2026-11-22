"""PoUS's Lean answers (`period.serve`, `timed_audit.run_lean`) are read only where the verifier wrote no panic and exited
0 (`run_lean`: 0 or 1, its verdict's own), and it starts with the runtime's panic switch. A stand-in that answers accept
beside a panic line is refused by name, and one that exits non-zero with an accept on its output stream is refused."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity import lean_process as LP
from verity.protocols.accounting.memory.pous.attester import timed_audit as TA
from verity.protocols.accounting.memory.pous.verifier import period

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


#: the runtime's lines with no ``PANIC`` (`verity.lean_process.PANIC`)
OTHER_PANICS = ["Error: index out of bounds", "Error: invalid `String.Pos` at `String.get!`",
                "`Task.get` called from a `(sync := true)` task", "Stack overflow detected. Aborting.",
                "LEAN ASSERTION VIOLATION"]


@pytest.mark.parametrize("line", OTHER_PANICS)
def test_an_accept_beside_a_runtime_line_with_no_panic_word_is_refused_by_name_at_exit_0(tmp_path, line):
    exe = _fake(tmp_path, f"print({line!r}, file=sys.stderr)\nprint({ACCEPT!r})\n")
    with pytest.raises(LP.LeanPanic) as e:
        period.serve(exe, [{}])
    assert str(e.value) == f"{LP.LEAN_PANIC}: {line}"
    with pytest.raises(LP.LeanPanic, match=f"^{LP.LEAN_PANIC}: "):
        TA.run_lean(exe, {}, b"", {})


def test_an_accept_before_a_silent_exit_134_or_a_signal_is_refused_by_name(tmp_path):
    killed = f"print({ACCEPT!r}, flush=True)\nimport signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
    for serve in (period.serve, lambda exe, cases: TA.run_lean(exe, {}, b"", {})):
        with pytest.raises(LP.LeanExit, match=f"^{LP.LEAN_EXIT}: status 134 \\(abort\\)"):
            serve(_fake(tmp_path, _accepting(False, 134)), [{}])
        with pytest.raises(LP.LeanExit, match=f"^{LP.LEAN_EXIT}: killed by SIGKILL"):
            serve(_fake(tmp_path, killed), [{}])


def test_a_refusal_sentence_that_contains_the_runtimes_text_is_no_panic(tmp_path):
    sentence = "the case is refused: Error: index out of bounds is what its field says, not the runtime"
    exe = _fake(tmp_path, f"print({sentence!r}, file=sys.stderr)\n" + _accepting(False))
    assert period.serve(exe, [{}]) == [{"verdict": "accept"}]
    assert TA.run_lean(exe, {}, b"", {}) == {"verdict": "accept"}


def test_a_status_outside_the_documented_set_is_refused_whatever_it_printed(tmp_path):
    """`serve` documents 0 (`Cli.lean` `serveLoop`; 70 where an outcome isn't canonical), and the timed audit 0 and 1
    (`TimedAudit.run`; 2 where its verdict isn't canonical JSON)."""
    for code in (2, 70):
        with pytest.raises(LP.LeanExit, match=f"status {code}"):
            period.serve(_fake(tmp_path, _accepting(False, code)), [{}])
    with pytest.raises(LP.LeanExit, match="status 2"):
        TA.run_lean(_fake(tmp_path, _accepting(False, 2)), {}, b"", {})
