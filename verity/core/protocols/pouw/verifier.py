"""PoUW's verifier of record. Lean is the definition and the verifier of record (Daniel, 18 Oct; AGENTS.md "Design
rules"), and PoUW has switched to it by the 8 Oct procedure (`.agents/skills/friction/SKILL.md`): the served window's checks
are its Lean's, `lean/check` (`pouw-check`), `lean/commit` (`pouw-commit`) and `lean/window` (`pouw-window`), whose
verdicts `benchmarks/pouw/served_zk/of_record.py` publishes. The Python beside them (`core_check`, `coins`, `window`) is kept
as the reference, and the difftests hold it to the Lean: a disagreement fails with the Lean's verdict.

`RECORD_VERDICT` is the switch, and setting it to ``"python"`` is the one line that undoes it. This module imports nothing,
so what decides a window never depends on `verity.ml`."""

RECORD_VERDICT = "lean"
