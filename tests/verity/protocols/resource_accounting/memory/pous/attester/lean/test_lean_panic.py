"""A Lean verdict is read only where the verifier wrote no panic, and it starts with the runtime's panic switch: the
memory attester's verdict and the wipe's, through the difftest server both launch under. A stand-in that answers accept
beside a panic line is refused by name; the verifiers' Lean is not built here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import verdict as V
import verity_difftest as D
import wipe_verdict as WV

PANIC_LINE = "PANIC at MemoryChallenger.Judge MemoryChallenger.Judge:1:2: index out of bounds"
ACCEPT = {"accepted": True, "late": 0, "reasons": [], "wrong": 0}


def _fake(tmp_path: Path, body: str) -> D.Server:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return D.Server([str(exe)])


def _answer(output: dict, *, panic: bool, exit_code: int = 0) -> str:
    return f"""for line in sys.stdin:
    {'print(%r, file=sys.stderr, flush=True)' % PANIC_LINE if panic else 'pass'}
    print({json.dumps({"output": output}, sort_keys=True, separators=(",", ":"))!r}, flush=True)
    sys.exit({exit_code})
"""


def test_the_memory_attesters_accept_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    server = _fake(tmp_path, _answer(ACCEPT, panic=True))
    monkeypatch.setattr(V, "server", lambda: server)
    monkeypatch.setattr(V, "record", lambda *a, **k: {})
    with pytest.raises(D.ProtocolError) as e:
        V.verdict(None, None, [], lambda i: b"", trusted=set())
    assert str(e.value) == f"{D.LEAN_PANIC}: {PANIC_LINE}"


def test_the_wipes_accept_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    server = _fake(tmp_path, _answer({"accepted": True, "reasons": []}, panic=True))
    monkeypatch.setattr(WV, "server", lambda: server)
    monkeypatch.setattr(WV, "record", lambda *a, **k: {})
    with pytest.raises(D.ProtocolError) as e:
        WV.verdict(None, None, trusted=set())
    assert str(e.value) == f"{D.LEAN_PANIC}: {PANIC_LINE}"


def test_an_accept_from_a_process_that_exits_non_zero_is_refused(tmp_path):
    server = _fake(tmp_path, _answer(ACCEPT, panic=False, exit_code=3))
    try:
        assert server.call("verdict", {})[0] == "output"
        with pytest.raises(D.ProtocolError, match="ended on"):
            server.call("verdict", {})
    finally:
        server.close()


def test_the_server_starts_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(D.ABORT_ON_PANIC, "0")
    server = _fake(tmp_path, """for line in sys.stdin:
    print(json.dumps({"output": {"switch": os.environ.get("LEAN_ABORT_ON_PANIC")}}, sort_keys=True, separators=(",", ":")), flush=True)
""")
    try:
        assert D.ABORT_ON_PANIC == "LEAN_ABORT_ON_PANIC"
        assert json.loads(server.call("verdict", {})[1]) == {"switch": "1"}
    finally:
        server.close()


def test_a_warning_on_the_error_stream_is_no_panic(tmp_path):
    server = _fake(tmp_path, _answer(ACCEPT, panic=False).replace("    pass\n", "    print('warning: slow', file=sys.stderr, flush=True)\n", 1))
    try:
        assert json.loads(server.call("verdict", {})[1]) == ACCEPT
    finally:
        server.close()


#: the runtime's lines with no ``PANIC`` (`verity.lean_process.PANIC`): an index past the end, `String.get!` off a character,
#: `Task.get` in a sync task, a stack overflow and a debug build's assertion
OTHER_PANICS = ["Error: index out of bounds", "Error: invalid `String.Pos` at `String.get!`",
                "`Task.get` called from a `(sync := true)` task", "Stack overflow detected. Aborting.",
                "LEAN ASSERTION VIOLATION"]


def _replying(stderr_line: str | None, then: str = "") -> str:
    """A server that answers accept to every request, with ``stderr_line`` before each reply and ``then`` after it."""
    out = json.dumps({"output": ACCEPT}, sort_keys=True, separators=(",", ":"))
    return (f"for line in sys.stdin:\n"
            f"    {'print(%r, file=sys.stderr, flush=True)' % stderr_line if stderr_line else 'pass'}\n"
            f"    print({out!r}, flush=True)\n    {then or 'pass'}\n")


@pytest.mark.parametrize("line", OTHER_PANICS)
def test_an_accept_beside_a_runtime_line_with_no_panic_word_is_refused_by_name(tmp_path, line):
    server = _fake(tmp_path, _replying(line))
    try:
        with pytest.raises(D.ProtocolError) as e:
            server.call("verdict", {})
        assert str(e.value) == f"{D.LEAN_PANIC}: {line}"
    finally:
        server.close()


def test_a_server_that_ends_134_silently_after_an_accept_is_refused_by_name_at_its_next_request(tmp_path):
    server = _fake(tmp_path, _replying(None, "sys.exit(134)"))
    try:
        assert server.call("verdict", {})[0] == "output"
        with pytest.raises(D.ProtocolError, match=f"{D.LEAN_EXIT}: status 134 \\(abort\\)"):
            server.call("verdict", {})
    finally:
        server.close()


def test_a_server_killed_by_a_signal_after_an_accept_is_refused_by_name_at_its_next_request(tmp_path):
    server = _fake(tmp_path, _replying(None, "import signal; os.kill(os.getpid(), signal.SIGKILL)"))
    try:
        assert server.call("verdict", {})[0] == "output"
        with pytest.raises(D.ProtocolError, match=f"{D.LEAN_EXIT}: killed by SIGKILL"):
            server.call("verdict", {})
    finally:
        server.close()


def test_a_refusal_sentence_that_contains_the_runtimes_text_is_no_panic(tmp_path):
    sentence = "the record is refused: Error: index out of bounds is what its field says, not the runtime"
    server = _fake(tmp_path, _replying(sentence))
    try:
        assert json.loads(server.call("verdict", {})[1]) == ACCEPT
    finally:
        server.close()


#: what the stand-in does at end of input, after every reply: the verdict it gave was an accept
AT_END = {"a panic line": f"print({PANIC_LINE!r}, file=sys.stderr, flush=True)",
          "exit 134": "sys.exit(134)",
          "a signal": "import signal; os.kill(os.getpid(), signal.SIGKILL)"}
REFUSED_AT_END = {"a panic line": f"^{D.LEAN_PANIC}: {PANIC_LINE}$", "exit 134": f"^{D.LEAN_EXIT}: status 134 \\(abort\\)$",
                  "a signal": f"^{D.LEAN_EXIT}: killed by SIGKILL$"}


def _answering(at_end: str) -> str:
    out = json.dumps({"output": ACCEPT}, sort_keys=True, separators=(",", ":"))
    return f"for line in sys.stdin:\n    print({out!r}, flush=True)\n{at_end}\n"


@pytest.mark.parametrize("how", AT_END)
def test_a_server_that_answers_accept_and_ends_unclean_is_refused_at_its_checked_close(tmp_path, how):
    server = _fake(tmp_path, _answering(AT_END[how]))
    assert server.call("verdict", {})[0] == "output"
    with pytest.raises(D.ProtocolError, match=REFUSED_AT_END[how]):
        server.close_checked()


def test_a_server_that_ends_clean_passes_its_checked_close(tmp_path):
    server = _fake(tmp_path, _answering("pass"))
    assert server.call("verdict", {})[0] == "output"
    server.close_checked()
    server.close_checked()


@pytest.mark.parametrize("how", AT_END)
@pytest.mark.parametrize("module, call", [
    (V, lambda: V.verdict(None, None, [], lambda i: b"", trusted=set())),
    (WV, lambda: WV.verdict(None, None, trusted=set()))], ids=["memory_attester", "wipe"])
def test_a_verdict_is_refused_at_the_end_of_the_verdicts_where_its_server_ended_unclean(tmp_path, monkeypatch, how, module,
                                                                                        call):
    import functools

    monkeypatch.setattr(module, "record", lambda *a, **k: {})
    monkeypatch.setattr(module, "server", functools.cache(lambda: _fake(tmp_path, _answering(AT_END[how]))))
    call()
    with pytest.raises(D.ProtocolError, match=REFUSED_AT_END[how]):
        module.finish()
    module.finish()


def test_the_end_of_the_verdicts_is_nothing_where_no_server_started(monkeypatch):
    import functools

    monkeypatch.setattr(V, "server", functools.cache(lambda: pytest.fail("a server was started")))
    V.finish()
