"""How an agreement harness starts the Lean executable whose answer it compares, with the standard library alone: in
`lean_env`, refused by name where it wrote a panic line, whatever its exit status and answer, and where it was killed by
a signal or ended with a status its caller does not document, whatever it wrote. The rule is the one-stage door's
`lean_child`'s, which `tests/test_lean_child_rule.py` holds this copy to."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Collection, Mapping, Sequence

#: the Lean runtime's switch: set, a panic ends the process with exit 1 where it would otherwise print its line, return a
#: default value and carry on
ABORT_ON_PANIC = "LEAN_ABORT_ON_PANIC"
#: a line Lean's runtime writes on a panic or an abort: ``PANIC at DECL MODULE:LINE:COL: MSG`` (`mkPanicMessageWithDecl`),
#: the C runtime's ``PANIC: MSG`` and ``INTERNAL PANIC: MSG``, and its lines with no ``PANIC``: an array index past the end
#: (`lean_array_get_panic`, `lean_array_set_panic`), `String.get!` off a character, `Task.get` in a sync task, a stack
#: overflow and a debug build's assertion
PANIC = re.compile(r"^(?:(?:INTERNAL )?PANIC\b.*|Error: index out of bounds|Error: invalid `String\.Pos` at `String\.get!`"
                   r"|`Task\.get` called from a `\(sync := true\)` task|Stack overflow detected\. Aborting\."
                   r"|LEAN ASSERTION VIOLATION)$", re.M)
LEAN_PANIC = "lean-panic: the Lean verifier wrote a panic: its answer is not read"
LEAN_EXIT = "lean-exit: the Lean verifier ended with a status it does not document: its answer is not read"
#: the statuses a Lean executable documents where its caller names none: 0, its answer
OK = frozenset({0})
#: 0 every file's answer, 1 a file refused: `unit-cut`, `cut-check`, `evaluate`, `qword-program`, `qcall-program` and
#: `flock-zk-server finish`
REFUSES = frozenset({0, 1})
#: `verity-verify-proof verify` and `agree` (`verifyCmdBy`), and `rec-compose`: 0 every session accepted, 1 one refused, 2
#: its inputs refused before any session
VERIFY_EXITS = frozenset({0, 1, 2})
#: `flock-zk-server contract` (`Flock.ZkServer.Contract.cmd`): 0 every file's verdict, 2 a file or the public input unread
CONTRACT_EXITS = frozenset({0, 2})
#: `abort()`'s status, as a shell or a scope wrapping the process reports a SIGABRT
ABORTED = 128 + signal.SIGABRT


class LeanPanic(RuntimeError):
    """A Lean process wrote a panic line: `LEAN_PANIC`, then the line."""


class LeanExit(LeanPanic):
    """A Lean process was killed by a signal, aborted, or ended with a status its caller does not document: `LEAN_EXIT`,
    then the status."""


def lean_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a Lean executable whose answer can become a verdict starts in: ``environ`` (this process's when
    None) with `ABORT_ON_PANIC` set, whatever ``environ`` holds."""
    return {**(os.environ if environ is None else environ), ABORT_ON_PANIC: "1"}


def panic_line(stderr: str | bytes | None) -> str | None:
    """The first panic line of ``stderr``; None where it has none."""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    m = PANIC.search(stderr or "")
    return None if m is None else m.group(0)


def refuse_panic(stderr: str | bytes | None) -> None:
    """`LeanPanic` where ``stderr`` holds a panic line."""
    if (line := panic_line(stderr)) is not None:
        raise LeanPanic(f"{LEAN_PANIC}: {line}")


def exit_why(returncode: int, ok_statuses: Collection[int] = OK) -> str | None:
    """Why a process that ended with ``returncode`` (negative: the signal that killed it) has no answer to read; None
    where ``returncode`` is one of ``ok_statuses``, which never takes an abort."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"{LEAN_EXIT}: killed by {name}"
    if returncode == ABORTED or returncode not in ok_statuses:
        return f"{LEAN_EXIT}: status {returncode}" + (" (abort)" if returncode == ABORTED else "")
    return None


def refuse_exit(returncode: int, ok_statuses: Collection[int] = OK) -> None:
    """`LeanExit` where ``returncode`` is not one of ``ok_statuses`` (`exit_why`)."""
    if (why := exit_why(returncode, ok_statuses)) is not None:
        raise LeanExit(why)


def why(p: subprocess.CompletedProcess, ok_statuses: Collection[int] = OK) -> str | None:
    """`run`'s refusal of ``p`` as a line, for a caller that records it rather than raising; None where it has none."""
    try:
        refuse_panic(p.stderr)
        refuse_exit(p.returncode, ok_statuses)
    except LeanPanic as e:
        return str(e)
    return None


def run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, ok_statuses: Collection[int] = OK,
        **kw) -> subprocess.CompletedProcess:
    """``subprocess.run`` of ``cmd``, its output captured as text, in `lean_env` of ``env``; `LeanPanic` where its
    standard error holds a panic line, else `LeanExit` where its status is not one of ``ok_statuses``, the statuses its
    executable documents for an answer (a refusal's among them)."""
    p = subprocess.run(list(cmd), capture_output=True, text=True, env=lean_env(env), **kw)
    refuse_panic(p.stderr)
    refuse_exit(p.returncode, ok_statuses)
    return p


def start(cmd: Sequence[str], **kw) -> subprocess.Popen:
    """``subprocess.Popen`` of ``cmd`` in `lean_env`, its standard error kept for `ended`."""
    err = tempfile.TemporaryFile()
    p = subprocess.Popen(list(cmd), stderr=err, env=lean_env(), **kw)
    p.lean_err = err
    return p


def ended(p: subprocess.Popen, ok_statuses: Collection[int] = OK) -> int:
    """``p``'s status (`start`'s process), once it ends, its standard error passed on; refused as `run` refuses."""
    rc = p.wait()
    p.lean_err.seek(0)
    err = p.lean_err.read().decode("utf-8", "replace")
    p.lean_err.close()
    sys.stderr.write(err)
    refuse_panic(err)
    refuse_exit(rc, ok_statuses)
    return rc


def main(argv: Sequence[str]) -> int:
    """``lean_run.py SET OUT -- CMD...``, for a shell job: CMD started as `run` starts it, SET the name of the statuses it
    documents (`VERIFY_EXITS`); its standard error to OUT.err, and its output to OUT.out unless it is refused, which
    leaves OUT.out empty, prints the refusal and exits 3."""
    if len(argv) < 4 or argv[2] != "--" or not isinstance(ok := globals().get(argv[0]), frozenset):
        print("usage: lean_run.py SET OUT -- CMD...  (SET: OK, REFUSES, VERIFY_EXITS or CONTRACT_EXITS)", file=sys.stderr)
        return 2
    p = subprocess.run(list(argv[3:]), capture_output=True, text=True, env=lean_env())
    refused = why(p, ok)
    with open(f"{argv[1]}.err", "w") as f:
        f.write(p.stderr)
    with open(f"{argv[1]}.out", "w") as f:
        f.write("" if refused else p.stdout)
    if refused:
        print(refused, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
