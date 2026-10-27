"""`tools/verity/lean/trusted_size.py`: the per-area size of a package's trusted text, on a package of a few files and
facts written by hand (no Lean needed)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_trusted_size", HERE.parent / "trusted_size.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

FILES = {
    "Definitions/A/Basic.lean": "/-! A docstring\nover two lines. -/\n\ndef read (n : Nat) : Nat := n -- a comment\n\n"
                                "structure S where\n  x : Nat\n\ntheorem t : read 1 = 1 := rfl\n",
    "Definitions/A/Old.lean": "-- nothing reads this\ndef old : Nat := 0\n",
    "Definitions/B.lean": "/- a /- nested -/ comment -/\nabbrev b : Nat := 1\n",
    "Proofs/A.lean": "theorem p : True := trivial\n",
}
FACTS = {"pins": {}, "spec_definitions": [
    {"module": "Definitions.A.Basic", "name": "A.read", "kind": "def"},
    {"module": "Definitions.A.Basic", "name": "A.S", "kind": "inductive"},
    {"module": "Definitions.A.Basic", "name": "A.read.eq_1", "kind": "def"},
    {"module": "Definitions.A.Old", "name": "A.old", "kind": "def"},
    {"module": "Definitions.B", "name": "B.b", "kind": "def"},
]}
LOCK = {"roots": ["Definitions"], "guarantees": {"A.g": {"owner": "@lane"}}, "reads": {
    "Definitions.A.Basic": {"definitions": {"A.read": "h1", "A.S.x": "h2", "A.S": "h3"}, "guarantees": ["A.g"]},
    "Definitions.B": {"definitions": {"B.b": "h4"}, "guarantees": ["A.g"]},
}}


def package(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    (tmp_path / "lakefile.toml").write_text('name = "p"\n')
    (tmp_path / "lean-audit.json").write_text(json.dumps(LOCK))
    return tmp_path


def test_code_lines_skip_comments_and_docstrings():
    assert T.code_lines(FILES["Definitions/A/Basic.lean"]) == 4
    assert T.code_lines(FILES["Definitions/B.lean"]) == 1


def test_areas_and_unread_from_facts(tmp_path):
    m = T.measure(package(tmp_path), "Definitions", FACTS)
    a, b = m["areas"]["A"], m["areas"]["B"]
    assert m["exact"] and set(m["areas"]) == {"A", "B"}
    assert (a["files"], a["definitions"], a["read"], a["unread"], a["theorems"], a["guarantees"]) == (2, 3, 2, 1, 1, 1)
    assert (b["definitions"], b["read"], b["unread"]) == (1, 1, 0)
    assert m["unread"] == {"Definitions.A.Old": ["A.old"]}
    assert "| total | 3 |" in T.table(m)


def test_without_facts_counts_the_sources(tmp_path):
    m = T.measure(package(tmp_path), "Definitions")
    a = m["areas"]["A"]
    assert not m["exact"] and (a["definitions"], a["read"], a["unread"]) == (3, 3, None)
    assert "~3" in T.table(m)
