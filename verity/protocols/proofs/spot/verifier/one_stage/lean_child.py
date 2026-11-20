"""How the door starts a Lean executable whose answer can become a verdict: in `settings.lean_env`, and refused by name
where it wrote a panic line, whatever its exit status and answer. The code is `verity.lean_process`'s, which every other
launch in `verity` uses too."""

from __future__ import annotations

from verity.lean_process import LEAN_PANIC, PANIC, LeanPanic, panic_line, refuse_panic, run

__all__ = ["LEAN_PANIC", "PANIC", "LeanPanic", "panic_line", "refuse_panic", "run"]
