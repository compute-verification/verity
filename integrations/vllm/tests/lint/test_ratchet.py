"""The ratchet itself: a key found more often than allowed is new, and an entry allowing more than is found is stale."""
from __future__ import annotations

import json

import pytest

from . import _ratchet as R


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
