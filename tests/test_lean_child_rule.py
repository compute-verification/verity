"""The rule that refuses a Lean child's answer (its panic lines and the exits it refuses) is one rule in three homes,
which no one of them can load: `verity`'s (`verity.lean_process`, which the one-stage door, Flock's distribution and
every other launch in `verity` use), the agreement scripts' and the difference-test server's (its panic line and its
refusal's name)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOMES = ("verity/lean_process.py",
         "verity/protocols/proof/flock/prover/agreement/lean_run.py")
#: the modules that re-export `verity.lean_process`'s rule, and so hold it by loading it
LOADERS = ("verity/protocols/proof/flock/prover/python/verity_flock/lean_child.py",)
SERVER = "tools/difftest/python/verity_difftest/__init__.py"
SERVER_RULE = ("ABORT_ON_PANIC", "PANIC", "LEAN_PANIC", "LEAN_EXIT")
RULE = ("ABORT_ON_PANIC", "lean_env", "PANIC", "LEAN_PANIC", "LEAN_EXIT", "OK", "ABORTED", "LeanPanic", "LeanExit",
        "panic_line", "refuse_panic", "exit_why", "refuse_exit", "run")


def rule(path: str) -> dict[str, str]:
    got = {}
    for node in ast.parse((ROOT / path).read_text()).body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign)
                 else [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [])
        got.update({n: ast.unparse(node) for n in names if n in RULE})
    return got


def test_the_loaders_take_the_whole_rule_from_its_home_and_define_none_of_it():
    for path in LOADERS:
        body = ast.parse((ROOT / path).read_text()).body
        taken = {a.name for n in body if isinstance(n, ast.ImportFrom) and n.module == "verity.lean_process"
                 for a in n.names}
        lacks = sorted(set(RULE) - taken)
        assert not rule(path) and not lacks, (f"{path} defines {sorted(rule(path))} and lacks {lacks}: import them from "
                                              "verity.lean_process")


def test_the_homes_hold_one_rule():
    door, *others = (rule(p) for p in HOMES)
    assert sorted(door) == sorted(RULE), f"{HOMES[0]} lacks {sorted(set(RULE) - set(door))}"
    for path, got in zip(HOMES[1:], others):
        differ = [n for n in RULE if got.get(n) != door[n]]
        assert not differ, f"{path}'s {differ} differ from {HOMES[0]}'s: copy them from it exactly"


def test_the_difference_test_server_holds_the_rules_panic_line():
    door, server = rule(HOMES[0]), rule(SERVER)
    differ = [n for n in SERVER_RULE if server.get(n) != door[n]]
    assert not differ, f"{SERVER}'s {differ} differ from {HOMES[0]}'s: copy them from it exactly"
