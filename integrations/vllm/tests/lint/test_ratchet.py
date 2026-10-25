"""The ratchet itself: a key found more often than allowed is new, and an entry allowing more than is found is stale; and
every lint has the test the ratchet gate runs."""
from __future__ import annotations

import ast
import json

import pytest

from . import _ratchet as R


def test_every_lint_has_a_test_no_new_the_ratchet_gate_can_run() -> None:
    """`tools/ratchet_gate.py` runs each lint's `test_no_new*` tests, as the base has them, on a change's tree without pytest,
    so each lint (a module here naming its allowlist in `NAME`) needs one that takes no fixture."""
    lints = 0
    for p in sorted(R.HERE.glob("test_*.py")):
        body = ast.parse(p.read_text(encoding="utf-8")).body
        if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME" for t in n.targets) for n in body):
            continue
        lints += 1
        runnable = [n.name for n in body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_no_new")
                    and not (n.args.args or n.args.kwonlyargs or n.args.vararg or n.args.kwarg)]
        assert runnable, (f"{p.name}: no `test_no_new*` test without arguments, which the ratchet gate runs to judge a change "
                          "to this lint; name the test that refuses violations its allowlist doesn't cover `test_no_new_*`")
    assert lints


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
