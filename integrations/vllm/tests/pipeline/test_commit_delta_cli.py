"""[R15 rel, rev-cb BLOCKER M-0662] commit_delta's argument parser must BUILD.

MAN-09 8432f36ef added a second `ap.add_argument("--checkpoints", ...)` to commit_delta.main(); argparse refuses duplicate option strings,
so every invocation of the r15-int-5 candidate 6227bde01 (positives, negatives, pod_ready's readiness collection) died with
`argparse.ArgumentError: conflicting option string: --checkpoints` before any engine work.  The unit suites never build that parser
(main() constructs it inline and the module imports vllm/torch lazily but heavily), so the defect was only found live (same class as M-0236).

This test builds the parser from the SOURCE, torch-free: it extracts every `ap.add_argument(...)` call of main() with `ast`, replays them
on a fresh argparse.ArgumentParser and asserts that (a) it builds, (b) no option string is declared twice, (c) the MAN-09 flags and the
pre-existing --checkpoints are all present exactly once.  Constant-only arguments are replayed verbatim; keyword values that reference
names (type=int, default=SOME_CONST, ...) are replaced by placeholders -- only the option strings and their multiplicity are under test.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / "commit.py"


def _add_argument_calls() -> list[ast.Call]:
    tree = ast.parse(SRC.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls: list[ast.Call] = []
    for node in ast.walk(main):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "ap"):
            calls.append(node)
    return calls


def _option_strings(call: ast.Call) -> list[str]:
    out = []
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            out.append(a.value)
    return out


def test_commit_delta_parser_builds_and_declares_every_option_once():
    calls = _add_argument_calls()
    assert len(calls) > 20, "commit_delta.main() add_argument calls not found -- did the parser move?"
    ap = argparse.ArgumentParser(prog="commit_delta", allow_abbrev=False)
    seen: Counter[str] = Counter()
    for c in calls:
        opts = _option_strings(c)
        assert opts, "add_argument without a literal option string at line %d" % c.lineno
        for o in opts:
            seen[o] += 1
        kw = {}
        for k in c.keywords:
            if k.arg in ("help", "metavar", "dest", "nargs", "choices", "required", "action") and isinstance(k.value, ast.Constant):
                kw[k.arg] = k.value.value
        if kw.get("action") in ("store_true", "store_false", "count"):
            kw.pop("nargs", None)
        # the option strings and their multiplicity are what is under test; argparse raises ArgumentError on a duplicate right here
        ap.add_argument(*opts, **kw)
    dups = sorted(o for o, n in seen.items() if n > 1)
    assert dups == [], "option strings declared twice in commit_delta.main(): %s (argparse.ArgumentError at parse time, M-0662)" % dups
    for must in ("--checkpoints", "--checkpoint-dir", "--weights-of-record", "--program-digest", "--program-dir", "--programs-root",
                 "--match-dir", "--replay-challenge"):
        assert seen[must] == 1, "%s declared %d times" % (must, seen[must])
    assert ap.format_usage().startswith("usage: commit_delta")   # the parser is whole: usage renders over every declared option
