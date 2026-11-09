"""Every refusal of the check's runners that names a command to run has a test that starts from the refused state, runs that
advice and gets past the refusal (REMEDIATED), or is listed in KNOWN with its owner and why; KNOWN only shrinks.

A refusal is a `raise ...(message)` or a `print(message, file=...)` in a function of `SCANNED`, and what it names is each
backticked span of its message that holds a command (`uv`, `research`, `python`, `git` or one of the runners), an f-string's
fields read as `X`. Advice no test has run is how a refusal comes to name a command that is itself refused."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("tools/verity/check/suites.py", "tools/verity/check/check.py", "tools/verity/check/slot.py")
COMMAND = re.compile(r"`([^`]*\b(?:uv run|uv sync|research|python3?|git|suites\.py|check\.py|slot\.py)\b[^`]*)`")

#: (file, function, the command it names) -> the test, `path::name`, that runs it from the refused state
REMEDIATED = {
    ("tools/verity/check/suites.py", "changed_base", "--changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD)"):
        "tools/verity/check/tests/test_suites.py::test_the_changed_refusals_advice_run_in_a_stale_checkout_gets_past_it",
}

#: (file, function, the command it names) -> "owner: why", each untested when this test landed
KNOWN = {
    ("tools/verity/check/suites.py", "worktree", "git add -A"): "ci: names the step that failed, advises nothing",
    ("tools/verity/check/suites.py", "changed_base", "research run"): "ci: names where the tree came from, advises nothing",
    ("tools/verity/check/suites.py", "main", "research run --on"): "ci: a tree outside git; untested",
    ("tools/verity/check/suites.py", "main", "X && uv run tools/verity/check/suites.py"):
        "ci: syncs the tree's own environment before running there; no test runs it from the refused state",
    ("tools/verity/check/check.py", "fixtures_only", "research data export"): "ci: needs the store's write key; untested",
    ("tools/verity/check/check.py", "main", "research run"): "ci: names the launcher, advises nothing",
    ("tools/verity/check/check.py", "main", "check.py --record --cc-records --on M"): "ci: untested",
    ("tools/verity/check/check.py", "main", "X && uv run --extra torch-cpu python tools/verity/check/check.py"):
        "ci: syncs the tree's own environment before running there; no test runs it from the refused state",
    ("tools/verity/check/slot.py", "main", "research run --pool NAME"): "ci: untested",
    ("tools/verity/check/slot.py", "main", "check.py --slots"): "ci: lists the slots; untested",
}


def _text(node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _text(node.left) + _text(node.right)
    return ""


def advice() -> set[tuple[str, str, str]]:
    """(file, function, command) for each command a refusal in SCANNED names."""
    out = set()
    for f in SCANNED:
        for fn in ast.walk(ast.parse((ROOT / f).read_text())):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and n.exc.args:
                    message = n.exc.args[0]
                elif isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print" and n.args and \
                        any(k.arg == "file" for k in n.keywords):
                    message = n.args[0]
                else:
                    continue
                out |= {(f, fn.name, m[1]) for m in COMMAND.finditer(_text(message))}
    return out


def test_every_command_a_refusal_names_is_remediated_or_known():
    new = sorted(advice() - set(REMEDIATED) - set(KNOWN))
    assert not new, ("refusals that name a command no test runs from the refused state: add a test that does to REMEDIATED, "
                     "or an entry to KNOWN with its owner and why:\n" + "\n".join(f"  {f} {fn}(): `{c}`" for f, fn, c in new))


def test_the_lists_are_current():
    found = advice()
    stale = sorted((set(REMEDIATED) | set(KNOWN)) - found)
    assert not stale, "entries no refusal names any more (delete them):\n" + "\n".join(f"  {f} {fn}(): `{c}`" for f, fn, c in stale)
    gone = [t for t in REMEDIATED.values() if not re.search(rf"^def {t.split('::')[1]}\(", (ROOT / t.split("::")[0]).read_text(),
                                                            re.M)]
    assert not gone, f"REMEDIATED names tests that don't exist: {gone}"
