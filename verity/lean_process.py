"""How a compiled Lean verifier whose answer can become a verdict is started, and when its answer is read: with the
runtime's panic switch set, so a panic ends the process where it would print its line, return a default value and carry
on, and refused by name where its standard error holds a panic line, whatever its exit status and answer. One place for
every launch in `verity` (`verity.protocols.proofs.spot.verifier.one_stage.lean_child` re-exports it for the door)."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence

#: the Lean runtime's switch: set, a panic (an index past the end among them) ends the process with exit 1 where it would
#: otherwise print its line, return a default value and carry on
ABORT_ON_PANIC = "LEAN_ABORT_ON_PANIC"

#: a line Lean's runtime writes on a panic: ``PANIC at DECL MODULE:LINE:COL: MSG`` (`mkPanicMessageWithDecl`), or the C
#: runtime's own ``PANIC: MSG`` and ``INTERNAL PANIC: MSG``
PANIC = re.compile(r"^(?:INTERNAL )?PANIC\b.*$", re.M)
LEAN_PANIC = "lean-panic: the Lean verifier wrote a panic: its answer is not read"


class LeanPanic(RuntimeError):
    """A Lean process wrote a panic line: `LEAN_PANIC`, then the line."""


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


def run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, **kw) -> subprocess.CompletedProcess:
    """``subprocess.run`` of ``cmd``, its output captured as text, in `lean_env` of ``env``; `LeanPanic` where its
    standard error holds a panic line."""
    p = subprocess.run(list(cmd), capture_output=True, text=True, env=lean_env(env), **kw)
    refuse_panic(p.stderr)
    return p
