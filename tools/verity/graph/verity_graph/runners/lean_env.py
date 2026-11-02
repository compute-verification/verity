"""lean-env's runner: make the `lean` env a task names (`families/lean_env.py`), kernel-replay its modules (all of them, or
the closure of those the task's `modules` names, as its `scope` says), and write their declarations' axiom sets with the
env's digest (`lean_audit.deps_key`). Until it is written every such task is an error, which no executor takes for a
result, and every lean-module task, which reads its env's `axioms`, is cancelled."""
from __future__ import annotations

from pathlib import Path

#: the files of the Lean tool a task runs: inputs of every lean-env task
TOOLS = ("tools/verity/lean/Replay.lean",)


def run(task: dict, work: Path) -> dict:
    return {"status": "error", "detail": {"error": "the lean-env runner isn't written yet"}}
