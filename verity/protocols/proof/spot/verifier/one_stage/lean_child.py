"""How the door starts a Lean executable whose answer can become a verdict: in `settings.lean_env`, refused by name where
it wrote a panic line, whatever its exit status and answer, and where it was killed by a signal or ended with a status
its caller does not document, whatever it wrote. The code is `verity.lean_process`'s, which every other launch in `verity`
uses too."""

from __future__ import annotations

from verity.lean_process import (ABORTED, CHECK_EXITS, COMPOSE_EXITS, KS_FORM_EXITS, LEAN_EXIT, LEAN_PANIC, OK, PANIC,
                                 V2_EXITS, VERIFY_EXITS, LeanExit, LeanPanic, exit_why, panic_line, refuse_exit,
                                 lean_env, refuse_panic, run)

__all__ = ['ABORTED', 'CHECK_EXITS', 'COMPOSE_EXITS', 'KS_FORM_EXITS', 'LEAN_EXIT', 'LEAN_PANIC', 'LeanExit', 'LeanPanic', 'OK', 'PANIC', 'V2_EXITS', 'VERIFY_EXITS', 'exit_why', 'lean_env', 'panic_line', 'refuse_exit', 'refuse_panic', 'run']
