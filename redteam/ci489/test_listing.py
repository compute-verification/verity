"""The graph a candidate is judged by is listed from the candidate's own tree, by the candidate's own listers.

Each test shows a tree edit that removes gating tasks from the listing without an error, where check.py on the same tree
runs (and fails) the step. Under graph2 an absent task is neither pending nor failed, so the candidate passes on fewer tasks.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from verity_graph import graph
from verity_graph.families import build

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("graph_tests_conftest", ROOT / "tools/verity/graph/tests/conftest.py")
toys = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = toys
_spec.loader.exec_module(toys)


def test_a_family_whose_lister_module_is_gone_lists_no_task_and_no_error(tmp_path):
    """FINDING (self-weakening): `families.listers()` is every module in `families/`, and `graph.listing` skips a [family.*]
    table no lister serves (`if name not in listers: continue`); only a lister without a table is refused. A PR that deletes
    `families/lean_lock.py` (or renames it) lists no lean-lock task, and the candidate passes without the lock check. Nothing
    under tools/verity/graph/ is in drive.toml's `lowrisk_excludes`, so that PR can land on its ready mark alone. FIX: listing
    refuses a table with no lister; add tools/verity/graph/ to `lowrisk_excludes`; W3 lists each candidate with main's
    listers (the deployed copy) or holds its task set to its base's."""
    root = toys.toy_tree(tmp_path / "tree")
    full = graph.listing(root, listers=toys.LISTERS)
    assert {t["kind"] for t in full["tasks"]} == {"toy", "toy-b", "other"}
    without = graph.listing(root, listers={"toy": toys.toy})
    assert {t["kind"] for t in without["tasks"]} == {"toy", "toy-b"}
    assert "other" in graph.spec(root)["family"]


def lean_tree(root: Path, targets: list[str]) -> list[str]:
    files = {f"{build.LEAN}/lakefile.toml": f'name = "flock"\ndefaultTargets = {json.dumps(targets)}\n\n'
                                            '[[lean_exe]]\nname = "flock-verify"\nroot = "Main"\n',
             f"{build.LEAN}/Main.lean": "def main : IO Unit := pure ()\n",
             build.UNIT_CUT: "print('ok')\n",
             f"{build.VERIFIER}/upstream.json": "{}\n",
             "pyproject.toml": '[project]\nname = "toy"\n'}
    for p, text in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
    return sorted(files)


def test_the_lean_build_family_follows_the_candidates_own_default_targets(tmp_path):
    """FINDING (coverage, self-weakening): the build family lists lean-build, lean-unit-cut, lean-agreement and lean-suites
    only for the `lean_exe`s the verifier's lakefile names in `defaultTargets`. A lakefile edit that drops flock-verify from
    `defaultTargets` lists none of them, so graph2 passes the candidate with the verifier unbuilt and untested, while
    check.py's `lean_steps` still runs lean-build and lean-unit-cut on that tree (it skips them only without the lakefile),
    and lean-unit-cut fails there for want of `.lake/build/bin/flock-verify`. lakefile.* is excluded from the low-risk lane,
    so this needs a reviewed PR: a slip, not an attack. FIX: W3 refuses a candidate whose listing lacks a task id its base
    lists, unless the PR deleted that task's files; the build family lists lean-build whenever the lakefile is on the tree."""
    on = tmp_path / "on"
    off = tmp_path / "off"
    assert {t["kind"] for t in build.tasks(on, lean_tree(on, ["flock-verify"]))} == {"lean-build", "lean-unit-cut",
                                                                                       "lean-agreement"}
    assert build.tasks(off, lean_tree(off, [])) == []
