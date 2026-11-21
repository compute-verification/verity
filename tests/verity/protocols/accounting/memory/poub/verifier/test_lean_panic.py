"""PoUB's Lean answers are read only where the verifier wrote no panic and exited 0, and it starts with the runtime's panic
switch: the P2 case and the period case (`serve`), the session's verifier (`lean_verifier.Lean`) and the P2 party's
(`p2_lean.Lean`). A stand-in that answers accept beside a panic line is refused by name, and one that exits non-zero with
an accept on its output stream is refused."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity import lean_process as LP
from verity.protocols.accounting.memory.poub.verifier import p2_case, p2_lean, period
from verity.protocols.accounting.memory.poub.verifier.session import lean_verifier

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
    with pytest.raises(LP.LeanExit, match="^lean-exit: .*status 3"):
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
        lean.close(check=False)


def test_the_p2_partys_verifier_refuses_an_answer_beside_a_panic_by_name(tmp_path):
    lean = p2_lean.Lean(_fake(tmp_path, _served(True)))
    try:
        with pytest.raises(p2_lean.LeanError) as e:
            lean.call({"op": "x"})
        assert str(e.value) == f"{LP.LEAN_PANIC}: {PANIC_LINE}"
    finally:
        lean.close(check=False)


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
        lean.close(check=False)
    with p2_lean.Lean(exe) as lean:
        assert lean.call({"op": "x"}) == {"switch": "1"}


OTHER_PANICS = ["Error: index out of bounds", "Error: invalid `String.Pos` at `String.get!`",
                "`Task.get` called from a `(sync := true)` task", "Stack overflow detected. Aborting.",
                "LEAN ASSERTION VIOLATION"]


def _kept(stderr_line: str | None, then: str = "") -> str:
    """A kept process that answers accept to every request, with ``stderr_line`` before each reply and ``then`` after."""
    return (f"for line in sys.stdin:\n"
            f"    {'print(%r, file=sys.stderr, flush=True)' % stderr_line if stderr_line else 'pass'}\n"
            f"    print(json.dumps({{'answer': {ACCEPT!r}}}), flush=True)\n    {then or 'pass'}\n")


def _sessions(exe: Path):
    return lean_verifier.Lean(exe, lambda _: None)


def _party(exe: Path):
    return p2_lean.Lean(exe)


KEPT = [pytest.param(_sessions, lean_verifier.LeanError, id="sessions"), pytest.param(_party, p2_lean.LeanError, id="party")]


@pytest.mark.parametrize("line", OTHER_PANICS)
@pytest.mark.parametrize("make, error", KEPT)
def test_an_accept_beside_a_runtime_line_with_no_panic_word_is_refused_by_name(tmp_path, make, error, line):
    lean = make(_fake(tmp_path, _kept(line)))
    try:
        with pytest.raises(error) as e:
            lean.call({"op": "x"})
        assert str(e.value) == f"{LP.LEAN_PANIC}: {line}"
    finally:
        lean.close(check=False)


@pytest.mark.parametrize("make, error", KEPT)
def test_a_kept_process_that_ends_134_or_by_a_signal_after_an_accept_is_refused_by_name_at_its_next_request(
        tmp_path, make, error):
    for then, why in (("sys.exit(134)", "status 134 \\(abort\\)"),
                      ("import signal; os.kill(os.getpid(), signal.SIGKILL)", "killed by SIGKILL")):
        lean = make(_fake(tmp_path, _kept(None, then)))
        try:
            assert lean.call({"op": "x"}) == ACCEPT
            lean.proc.wait()
            with pytest.raises(error, match=f"^{LP.LEAN_EXIT}: {why}"):
                lean.call({"op": "x"})
        finally:
            lean.close(check=False)


@pytest.mark.parametrize("make, error", KEPT)
def test_a_refusal_sentence_that_contains_the_runtimes_text_is_no_panic(tmp_path, make, error):
    sentence = "the window is refused: Error: index out of bounds is what its field says, not the runtime"
    lean = make(_fake(tmp_path, _kept(sentence)))
    try:
        assert lean.call({"op": "x"}) == ACCEPT
    finally:
        lean.close(check=False)


#: what the stand-in does at end of input, after every answer: the answers it gave were accepts
AT_END = {"a panic line": f"print({PANIC_LINE!r}, file=sys.stderr, flush=True)",
          "exit 134": "sys.exit(134)",
          "a signal": "import signal; os.kill(os.getpid(), signal.SIGKILL)"}
REFUSED_AT_END = {"a panic line": f"^{LP.LEAN_PANIC}: {PANIC_LINE}$", "exit 134": f"^{LP.LEAN_EXIT}: status 134 \\(abort\\)$",
                  "a signal": f"^{LP.LEAN_EXIT}: killed by SIGKILL$"}


def _answering(at_end: str) -> str:
    return f"for line in sys.stdin:\n    print(json.dumps({{'answer': {ACCEPT!r}}}), flush=True)\n{at_end}\n"


@pytest.mark.parametrize("how", AT_END)
@pytest.mark.parametrize("make, error", KEPT)
def test_a_kept_process_that_answers_accept_and_ends_unclean_is_refused_at_its_close(tmp_path, make, error, how):
    lean = make(_fake(tmp_path, _answering(AT_END[how])))
    assert lean.call({"op": "x"}) == ACCEPT
    with pytest.raises(error, match=REFUSED_AT_END[how]):
        lean.close()


@pytest.mark.parametrize("make, error", KEPT)
def test_a_kept_process_that_ends_clean_passes_its_close(tmp_path, make, error):
    lean = make(_fake(tmp_path, _answering("pass")))
    assert lean.call({"op": "x"}) == ACCEPT
    lean.close()


def test_the_p2_partys_context_refuses_the_accept_it_returned_where_its_process_ended_unclean(tmp_path):
    def served():
        with p2_lean.Lean(_fake(tmp_path, _answering(AT_END["exit 134"]))) as lean:
            return lean.call({"op": "x"})

    with pytest.raises(p2_lean.LeanError, match=REFUSED_AT_END["exit 134"]):
        served()


def test_a_context_that_ends_on_an_error_keeps_that_error(tmp_path):
    with pytest.raises(KeyError), p2_lean.Lean(_fake(tmp_path, _answering(AT_END["exit 134"]))) as lean:
        lean.call({"op": "x"})
        raise KeyError("the session's own failure")


def test_a_session_accepted_by_a_verifier_that_ended_unclean_is_refused_by_name():
    from verity.protocols.accounting.memory.poub.verifier.session import verifier

    results = [{"accepted": True, "failure": None, "k": 0}, {"accepted": True, "failure": None, "k": 1}]
    verifier.refuse_unclean(results, REFUSED_AT_END["exit 134"])
    assert results == [{"accepted": False, "failure": REFUSED_AT_END["exit 134"], "k": 0},
                       {"accepted": False, "failure": REFUSED_AT_END["exit 134"], "k": 1}]
