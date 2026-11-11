"""Only a lane's own command line allows a Lean failure.

`lean_changed.py --records --allow CODE:ENTRY` has a lane's fast path take a failure as known, which is no pass. Nothing
that runs Lean's tools for the tree gives one: the check and its steps, the merge queue's loop, its records run and its
Lean tasks, the ready mark's preflight and the records gate, the `verity` command, and the Lean tools beside the two.
None of their sources so much as names the option, in a command it builds or in one it tells a lane to run. That
`check.py` has no such option and that the fast path hands it none is the Lean suite's
(`tools/verity/lean/tests/test_records_allow.py`)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: where the sources are that run `check.py` or `lean_changed.py`: the check, `research` (the queue, its loop, its
#: records run, the landing, the preflight), the task graph, the `verity` command, and the Lean tools themselves
RUNNERS = ("tools/verity/check", "tools/research/src/research", "tools/verity/graph/verity_graph", "tools/verity/cli",
           "tools/verity/lean")
#: the two files that may name it: the option's own, and the audit's, whose docstrings say it takes none
NAMES_IT = {"tools/verity/lean/lean_changed.py", "tools/verity/lean/check.py"}
#: callers the scan must reach, so that a move of one of them leaves the rule on it rather than on nothing
CALLERS = {"tools/verity/check/check.py", "tools/verity/check/train.sh", "tools/verity/lean/lean_audit.py",
           "tools/verity/lean/warm.py", "tools/research/src/research/vqueue_loop.py",
           "tools/research/src/research/vqueue_records.py", "tools/research/src/research/landing.py",
           "tools/research/src/research/queue.py", "tools/verity/cli/verity_cli/__init__.py",
           "tools/verity/graph/verity_graph/runners/lean_module.py"}
SOURCES = {".py", ".sh", ".toml"}
ALLOW = re.compile(r"(?<![\w-])--allow(?![\w-])")


def sources() -> list[str]:
    """The source files under `RUNNERS`, tests aside, as paths from the root."""
    found = (f for d in RUNNERS for f in (ROOT / d).rglob("*") if f.suffix in SOURCES and f.is_file())
    return sorted(r for f in found if "tests" not in (r := f.relative_to(ROOT).as_posix()).split("/"))


def test_nothing_that_runs_the_lean_check_the_merge_queue_or_the_preflight_names_allow():
    scanned = sources()
    assert CALLERS <= set(scanned), f"callers the scan no longer reaches: {sorted(CALLERS - set(scanned))}"
    named = {f: ns for f in scanned if f not in NAMES_IT
             if (ns := [n for n, line in enumerate((ROOT / f).read_text().splitlines(), 1) if ALLOW.search(line)])}
    assert not named, (f"these name --allow, which only a lane gives `lean_changed.py --records` by hand: {named}. The check, "
                       "the merge queue and the ready mark's preflight see every finding: drop the option from the command")


def test_the_scan_reads_the_option_and_not_another_that_starts_with_it():
    assert NAMES_IT <= set(sources()) and all(ALLOW.search((ROOT / f).read_text()) for f in NAMES_IT)
    assert ALLOW.search("lean_changed.py --records P --allow x:y") and ALLOW.search('"--allow",')
    assert not ALLOW.search("git commit --allow-empty") and not ALLOW.search("--allowed")
