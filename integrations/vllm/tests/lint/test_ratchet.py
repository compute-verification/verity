"""The ratchet itself: a key found more often than allowed is new, and an entry allowing more than is found is stale; and
every lint has the test the ratchet gate runs."""
from __future__ import annotations

import ast
import json
import sys

import pytest

from . import _ratchet as R


def _outside_imports(tree: ast.Module) -> list[str]:
    """What a module imports beyond the standard library, pytest and this directory."""
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level and n.module]
    return sorted({m for m in mods if m.split(".")[0] not in sys.stdlib_module_names | {"pytest"}})


def test_every_lint_is_one_the_ratchet_gate_can_run() -> None:
    """`tools/ratchet_gate.py` runs each lint's `test_no_new*` tests, as the base has them, on a change's tree, under drive's
    Python: the standard library and a stand-in for pytest's marks.  So each lint (a module here whose `NAME`, a string
    literal, names its allowlist) needs one that takes no fixture, and it and the helpers here import nothing else."""
    lints = 0
    for p in sorted(R.HERE.glob("test_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        names = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME"
                                                                                 for t in n.targets)]
        if not names:
            continue
        lints += 1
        assert isinstance(names[0], ast.Constant) and isinstance(names[0].value, str), \
            f"{p.name}: `NAME` must be a string literal, which the ratchet gate reads without importing the module"
        runnable = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_no_new")
                    and not (n.args.args or n.args.kwonlyargs or n.args.vararg or n.args.kwarg)]
        assert runnable, (f"{p.name}: no `test_no_new*` test without arguments, which the ratchet gate runs to judge a change "
                          "to this lint; name the test that refuses violations its allowlist doesn't cover `test_no_new_*`")
        bad = _outside_imports(tree)
        assert not bad, f"{p.name} imports {bad}, which drive's Python lacks, so the ratchet gate couldn't run it"
    assert lints
    for p in sorted(R.HERE.glob("_*.py")):
        bad = _outside_imports(ast.parse(p.read_text(encoding="utf-8")))
        assert not bad, f"{p.name}, which the lints import, imports {bad}, which drive's Python lacks"


def test_new_and_stale_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(R, "ALLOWLISTS", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"rule": "X", "entries": [
        {"file": "a.py", "kind": "k", "symbol": "f", "detail": "d", "count": 2},
        {"file": "b.py", "kind": "k", "symbol": "<module>", "detail": "gone"},
    ]}))
    found = [R.Violation("a.py", "k", "f", "d", 3), R.Violation("a.py", "k", "f", "d", 9),
             R.Violation("a.py", "k", "f", "d", 12), R.Violation("c.py", "k", "g", "new", 5)]
    assert R.new_violations("x", found) == ["  a.py:3,9,12 [k] f: d  (3 found, 2 allowed)", "  c.py:5 [k] g: new"]
    assert R.stale_entries("x", found[:1]) == [
        '  lower count to 1: {"file": "a.py", "kind": "k", "symbol": "f", "detail": "d", "count": 2}',
        '  delete {"file": "b.py", "kind": "k", "symbol": "<module>", "detail": "gone"}',
    ]
    assert R.new_violations("x", found[:2]) == [] and R.stale_entries("x", found[:2] + found[3:]) != []


def test_the_moved_definitions_are_scanned_under_their_repository_paths() -> None:
    p = R.DEFINITIONS / "registry" / "a.py"
    assert R.rel(p) == "verity/ml/definitions/registry/a.py" and R.module_name(p) == "verity.ml.definitions.registry.a"
    assert R.rel(R.PKG / "program" / "b.py") == "verity_vllm/program/b.py"
    assert not any(p.is_relative_to(R.DEFINITIONS) for p in R.library_files(".py", R.CORE_PKG))
    assert set(R.library_files()) >= set(R.library_files(".py", R.DEFINITIONS))


@pytest.mark.parametrize("entries", [
    [{"file": "a.py", "kind": "k", "symbol": "f"}],
    [{"file": "a.py", "kind": "k", "symbol": "f", "detail": "d", "count": 0}],
    [{"file": "a.py", "kind": "k", "symbol": "f", "detail": "d"}, {"file": "a.py", "kind": "k", "symbol": "f", "detail": "d"}],
])
def test_malformed_allowlists_are_refused(tmp_path, monkeypatch, entries) -> None:
    monkeypatch.setattr(R, "ALLOWLISTS", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"rule": "X", "entries": entries}))
    with pytest.raises(AssertionError):
        R.load_allowlist("x")
