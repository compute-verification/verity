"""How Flock's distribution starts a Lean executable whose answer can become a verdict: `verity.lean_process`'s rule,
which every launch in `verity` uses."""

from __future__ import annotations

from verity.lean_process import (ABORT_ON_PANIC, ABORTED, LEAN_EXIT, LEAN_PANIC, OK, PANIC, LeanExit, LeanPanic,
                                 exit_why, lean_env, panic_line, refuse_exit, refuse_panic, run)

__all__ = ["ABORTED", "ABORT_ON_PANIC", "LEAN_EXIT", "LEAN_PANIC", "LeanExit", "LeanPanic", "OK", "PANIC", "exit_why",
           "lean_env", "panic_line", "refuse_exit", "refuse_panic", "run"]
