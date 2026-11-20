"""How the door starts a Lean executable whose answer can become a verdict: in `settings.lean_env`, and refused by name
where it wrote a panic line, whatever its exit status and answer."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence

from .settings import lean_env

#: a line Lean's runtime writes on a panic: ``PANIC at DECL MODULE:LINE:COL: MSG`` (`mkPanicMessageWithDecl`), or the C
#: runtime's own ``PANIC: MSG`` and ``INTERNAL PANIC: MSG``
PANIC = re.compile(r"^(?:INTERNAL )?PANIC\b.*$", re.M)
LEAN_PANIC = "lean-panic: the Lean verifier wrote a panic: its answer is not read"


class LeanPanic(RuntimeError):
    """A Lean process wrote a panic line: `LEAN_PANIC`, then the line."""


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
