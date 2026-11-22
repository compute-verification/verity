"""A Lean verifier's answer is read only where it wrote no panic and ended at a status its caller documents, and the door
starts it with the runtime's panic switch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity.protocols.proof.spot.verifier.one_stage import lean_child as LC, proofs as PF, verifier as V
from verity.protocols.proof.spot.verifier.one_stage.settings import ABORT_ON_PANIC

PANIC_LINE = "PANIC at Flock.verify Flock.Verify:1:2: index out of bounds"


def _fake(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return exe


def test_an_accept_beside_a_panic_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, f"print({PANIC_LINE!r}, file=sys.stderr)\nprint('VERDICT ' + json.dumps({{'accepted': True}}))\n")
    with pytest.raises(LC.LeanPanic) as e:
        LC.run([str(exe)])
    assert str(e.value) == f"{LC.LEAN_PANIC}: {PANIC_LINE}"


def test_a_served_answer_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    exe = _fake(tmp_path, f"""for line in sys.stdin:
    print({PANIC_LINE!r}, file=sys.stderr, flush=True)
    print(json.dumps({{"output": {{"accepted": True}}}}), flush=True)
""")
    monkeypatch.setattr(V, "executable", lambda *_: exe)
    server = V._Server("fake-lean", tmp_path)
    try:
        with pytest.raises(V.LeanError, match=f"^{LC.LEAN_PANIC}: {PANIC_LINE}$"):
            server.call("verify", {})
    finally:
        server.close()


def test_the_child_starts_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, f"print(os.environ.get({ABORT_ON_PANIC!r}))\n")
    assert LC.run([str(exe)]).stdout == "1\n"
    assert LC.run([str(exe)], env={"PATH": "/usr/bin"}).stdout == "1\n"
    assert LC.run([str(exe)], env=PF.verifier_env()).stdout == "1\n"
    assert PF.verifier_env({})[ABORT_ON_PANIC] == "1"


def test_a_clean_answer_is_read(tmp_path):
    exe = _fake(tmp_path, "print('warning: slow', file=sys.stderr)\nprint('VERDICT {}')\n")
    p = LC.run([str(exe)])
    assert (p.returncode, p.stdout) == (0, "VERDICT {}\n")


def test_a_refusal_at_a_status_its_caller_documents_is_read(tmp_path):
    exe = _fake(tmp_path, "print('{\"refused\": \"an error in the evidence\"}')\nsys.exit(1)\n")
    assert LC.run([str(exe)], ok_statuses=LC.V2_EXITS).returncode == 1
    with pytest.raises(LC.LeanExit, match=f"^{LC.LEAN_EXIT}: status 1$"):
        LC.run([str(exe)])


#: what Lean's runtime writes on a panic or an abort with no ``PANIC`` in it (`object.cpp`, `stack_overflow.cpp`, `debug.cpp`)
UNPREFIXED = ["Error: index out of bounds", "Error: invalid `String.Pos` at `String.get!`",
              "`Task.get` called from a `(sync := true)` task", "Stack overflow detected. Aborting.", "LEAN ASSERTION VIOLATION"]


@pytest.mark.parametrize("line", [*UNPREFIXED, "INTERNAL PANIC: out of memory", "PANIC: Promise.result!: dropped"])
def test_each_line_the_runtime_writes_on_a_panic_is_one(line):
    assert LC.panic_line(f"warning: slow\n{line}\nbacktrace:\n") == line


@pytest.mark.parametrize("text", ["refused: Error: index out of bounds", "an error: index out of bounds",
                                  "Error: index out of bounds of the table", "error: Stack overflow detected. Aborting."])
def test_a_refusal_that_names_those_words_is_no_panic(text):
    assert LC.panic_line(text) is None


def test_an_accept_beside_an_index_past_the_end_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, "print('Error: index out of bounds', file=sys.stderr)\nprint('VERDICT ' + json.dumps({'accepted': True}))\n")
    with pytest.raises(LC.LeanPanic, match=f"^{LC.LEAN_PANIC}: Error: index out of bounds$"):
        LC.run([str(exe)])


def test_a_served_answer_beside_an_index_past_the_end_is_refused_by_name(tmp_path, monkeypatch):
    exe = _fake(tmp_path, """for line in sys.stdin:
    print('Error: index out of bounds', file=sys.stderr, flush=True)
    print(json.dumps({"output": {"accepted": True}}), flush=True)
""")
    monkeypatch.setattr(V, "executable", lambda *_: exe)
    server = V._Server("fake-lean", tmp_path)
    try:
        with pytest.raises(V.LeanError, match=f"^{LC.LEAN_PANIC}: Error: index out of bounds$"):
            server.call("verify", {})
    finally:
        server.close()


def test_an_abort_that_wrote_nothing_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, "print('VERDICT ' + json.dumps({'accepted': True}))\nsys.exit(134)\n")
    with pytest.raises(LC.LeanExit, match=f"^{LC.LEAN_EXIT}: status 134 \\(abort\\)$"):
        LC.run([str(exe)], ok_statuses=range(256))


def test_a_process_killed_by_a_signal_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, "import signal\nprint('VERDICT ' + json.dumps({'accepted': True}), flush=True)\nos.kill(os.getpid(), signal.SIGKILL)\n")
    with pytest.raises(LC.LeanExit, match=f"^{LC.LEAN_EXIT}: killed by SIGKILL$"):
        LC.run([str(exe)], ok_statuses=range(256))


def test_a_served_process_that_dies_answers_no_call(tmp_path, monkeypatch):
    exe = _fake(tmp_path, "import signal\nsys.stdin.readline()\nos.kill(os.getpid(), signal.SIGABRT)\n")
    monkeypatch.setattr(V, "executable", lambda *_: exe)
    server = V._Server("fake-lean", tmp_path)
    monkeypatch.setattr(V, "_server", server)
    try:
        assert V.lean("verify", {})["answered"] is False
    finally:
        server.close()


ACCEPTS = """for line in sys.stdin:
    print(json.dumps({"output": {"accepted": True}}), flush=True)
"""


def _served(tmp_path, monkeypatch, body: str) -> V._Server:
    monkeypatch.setattr(V, "executable", lambda *_: _fake(tmp_path, body))
    return V._Server("fake-lean", tmp_path)


def test_a_served_process_that_ends_at_its_documented_status_closes(tmp_path, monkeypatch):
    server = _served(tmp_path, monkeypatch, ACCEPTS)
    assert server.call("verify", {}) == ("output", {"accepted": True})
    assert server.close() is None


@pytest.mark.parametrize("end, why", [("sys.exit(134)", f"{LC.LEAN_EXIT}: status 134 \\(abort\\)"),
                                      (f"print({PANIC_LINE!r}, file=sys.stderr)", f"{LC.LEAN_PANIC}: {PANIC_LINE}")])
def test_a_served_process_that_accepts_then_does_not_end_as_documented_is_refused_at_its_close(tmp_path, monkeypatch,
                                                                                                end, why):
    server = _served(tmp_path, monkeypatch, f"{ACCEPTS}{end}\n")
    assert server.call("verify", {}) == ("output", {"accepted": True})
    with pytest.raises(V.LeanError, match=f"^fake-lean did not end as documented at its close: {why}$"):
        server.close()


def test_a_served_process_that_accepts_then_aborts_refuses_the_next_call_by_name(tmp_path, monkeypatch):
    server = _served(tmp_path, monkeypatch, """print(json.dumps({"output": {"accepted": True}}), flush=True) if sys.stdin.readline() else None
sys.exit(134)
""")
    try:
        assert server.call("verify", {}) == ("output", {"accepted": True})
        server._proc.wait()
        with pytest.raises(V.LeanError, match=f"^fake-lean ended after its last answer, before verify: {LC.LEAN_EXIT}: "
                                              "status 134 \\(abort\\)$"):
            server.call("verify", {})
    finally:
        server.close()


def test_the_verifiers_exit_refuses_by_status_an_accept_whose_process_then_aborted(tmp_path, monkeypatch):
    server = _served(tmp_path, monkeypatch, f"{ACCEPTS}sys.exit(134)\n")
    monkeypatch.setattr(V, "_server", server)
    monkeypatch.setattr(V, "_caller_server", V._Server("unstarted", tmp_path))
    exited = []
    monkeypatch.setattr(V.os, "_exit", exited.append)
    assert V.lean("verify", {}) == {"accepted": True}
    V._close_at_exit()
    assert exited == [70]
