"""How a compiled Lean verifier whose answer can become a verdict is started, and when its answer is read: with the
runtime's panic switch set (`lean_env`), so a panic ends the process where it would print its line, return a default value
and carry on; refused by name where its standard error holds a panic line, whatever its exit status and answer; and refused
where it was killed by a signal or ended with a status its caller does not document, whatever it wrote. One place for every
launch in `verity` (`verity.protocols.proof.spot.verifier.one_stage.lean_child` re-exports it for the door).
`tests/test_lean_child_rule.py` holds the copies other distributions must keep equal to it."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from collections.abc import Collection, Mapping, Sequence

#: the Lean runtime's switch: set, a panic (an index past the end among them) ends the process with exit 1 where it would
#: otherwise print its line, return a default value and carry on
ABORT_ON_PANIC = "LEAN_ABORT_ON_PANIC"


def lean_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a Lean executable whose answer can become a verdict starts in: ``environ`` (this process's when
    None) with `ABORT_ON_PANIC` set, whatever ``environ`` holds."""
    return {**(os.environ if environ is None else environ), ABORT_ON_PANIC: "1"}


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
#: a protocol's `check` (`Core.Protocol.main`): 0 accept, 1 reject, 2 undecided
CHECK_EXITS = frozenset({0, 1, 2})
#: `verity-verify-proof verify` (`verifyCmdBy`): 0 every session accepted, 1 one refused, 2 its inputs refused before any
#: session; and `flock-lower-check compose`: 0 written, 1 its refusal, 2 an argument it can't read
VERIFY_EXITS = COMPOSE_EXITS = frozenset({0, 1, 2})
#: `verity-verify-proof ks-form` (`ksFormCmd`), and `spot-one-stage-zk members` and `verdicts`: 0 the answer, 1 the refusal
KS_FORM_EXITS = V2_EXITS = frozenset({0, 1})
#: `abort()`'s status, as a shell or a scope wrapping the process reports a SIGABRT
ABORTED = 128 + signal.SIGABRT


class LeanPanic(RuntimeError):
    """A Lean process wrote a panic line: `LEAN_PANIC`, then the line."""


class LeanExit(LeanPanic):
    """A Lean process was killed by a signal, aborted, or ended with a status its caller does not document: `LEAN_EXIT`,
    then the status."""


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


def run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, ok_statuses: Collection[int] = OK,
        **kw) -> subprocess.CompletedProcess:
    """``subprocess.run`` of ``cmd``, its output captured as text, in `lean_env` of ``env``; `LeanPanic` where its
    standard error holds a panic line, else `LeanExit` where its status is not one of ``ok_statuses``, the statuses its
    executable documents for an answer (a refusal's among them)."""
    p = subprocess.run(list(cmd), capture_output=True, text=True, env=lean_env(env), **kw)
    refuse_panic(p.stderr)
    refuse_exit(p.returncode, ok_statuses)
    return p
