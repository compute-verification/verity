"""The rule that refuses a Lean child's answer (its panic lines and the exits it refuses) is one rule in four homes,
which no one of them can load: `verity`'s (`verity.lean_process`, which the one-stage door and every other launch in
`verity` use), Flock's distribution's, the agreement scripts' and the difference-test server's (its panic line and its
refusal's name)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOMES = ("verity/lean_process.py",
         "verity/protocols/proofs/flock/prover/python/verity_flock/lean_child.py",
         "verity/protocols/proofs/flock/prover/agreement/lean_run.py")
SERVER = "tools/difftest/python/verity_difftest/__init__.py"
SERVER_RULE = ("PANIC", "LEAN_PANIC", "LEAN_EXIT")
RULE = ("PANIC", "LEAN_PANIC", "LEAN_EXIT", "OK", "ABORTED", "LeanPanic", "LeanExit", "panic_line", "refuse_panic",
        "exit_why", "refuse_exit", "run")


def rule(path: str) -> dict[str, str]:
    got = {}
    for node in ast.parse((ROOT / path).read_text()).body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign)
                 else [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [])
        got.update({n: ast.unparse(node) for n in names if n in RULE})
    return got


def test_the_three_homes_hold_one_rule():
    door, *others = (rule(p) for p in HOMES)
    assert sorted(door) == sorted(RULE), f"{HOMES[0]} lacks {sorted(set(RULE) - set(door))}"
    for path, got in zip(HOMES[1:], others):
        differ = [n for n in RULE if got.get(n) != door[n]]
        assert not differ, f"{path}'s {differ} differ from {HOMES[0]}'s: copy them from it exactly"


def test_the_difference_test_server_holds_the_rules_panic_line():
    door, server = rule(HOMES[0]), rule(SERVER)
    differ = [n for n in SERVER_RULE if server.get(n) != door[n]]
    assert not differ, f"{SERVER}'s {differ} differ from {HOMES[0]}'s: copy them from it exactly"
