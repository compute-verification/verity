"""`tools/verity/lean/trusted_size.py`: the per-area size of a package's trusted text, on a package of a few files and
facts written by hand (no Lean needed)."""
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import tempfile
import types
from pathlib import Path

import pytest

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
FACTS = {"pins": {"A.g": {"kind": "theorem", "reads": [{"name": "A.read", "module": "Definitions.A.Basic"},
                                                      {"name": "B.b", "module": "Definitions.B"}]}}, "spec_definitions": [
    {"module": "Definitions.A.Basic", "name": "A.read", "kind": "def"},
    {"module": "Definitions.A.Basic", "name": "A.S", "kind": "inductive"},
    {"module": "Definitions.A.Basic", "name": "A.read.eq_1", "kind": "def"},
    {"module": "Definitions.A.Old", "name": "A.old", "kind": "def"},
    {"module": "Definitions.B", "name": "B.b", "kind": "def"},
]}
LOCK = {"roots": ["Definitions"], "guarantees": {"A.g": {"owner": "@lane"}}, "reads": {
    "Definitions.A.Basic": {"definitions": {"A.read": "h1", "A.S.x": "h2", "A.S": "h3"}},
    "Definitions.B": {"definitions": {"B.b": "h4"}},
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
    by = T.measure(package(tmp_path), "Definitions", FACTS, read_by={"Definitions.B": ["A.g", "B.h"]})
    assert (by["areas"]["A"]["guarantees"], by["areas"]["B"]["guarantees"]) == (0, 2), "a run's report.json `readers`"


def test_without_facts_counts_the_sources(tmp_path):
    m = T.measure(package(tmp_path), "Definitions")
    a = m["areas"]["A"]
    assert not m["exact"] and (a["definitions"], a["read"], a["unread"], a["guarantees"]) == (3, 3, None, None)
    assert "~3" in T.table(m) and T.table(m).splitlines()[-1].endswith("| - | - |"), "readers unknown without facts"


def test_the_gzipped_facts_and_an_older_run_s_plain_ones_measure_the_same(tmp_path, capsys):
    pkg = package(tmp_path / "pkg")
    data = json.dumps(FACTS).encode()
    (tmp_path / "facts.json.gz").write_bytes(gzip.compress(data))
    (tmp_path / "facts.json").write_bytes(data)
    printed = []
    for name in ("facts.json.gz", "facts.json"):
        assert T.main([str(pkg), "--facts", str(tmp_path / name), "--json"]) == 0
        printed.append(json.loads(capsys.readouterr().out))
    assert printed[0] == printed[1] == json.loads(json.dumps(T.measure(pkg, "Definitions", FACTS))) and printed[0]["exact"]


@pytest.mark.parametrize("kept", ["facts.json.gz", "facts.json"])
def test_a_run_s_facts_are_its_gzipped_ones_or_an_older_run_s_plain_ones(tmp_path, monkeypatch, kept):
    """`--run` asks the store for both names beside the lock and the report, and takes the one the run kept."""
    asked = []

    def run(argv, **kw):
        if argv[:3] == ["research", "data", "show"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"outputs": {"run_files": "art:tree"}}), "")
        to, globs = Path(argv[argv.index("--to") + 1]), argv[argv.index("--to") + 3::2]
        asked.extend(globs)
        for rel in (g for g in globs if g.endswith((kept, T.C.POLICY, "report.json"))):
            (to / rel).parent.mkdir(parents=True, exist_ok=True)
            (to / rel).write_text("{}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(T, "subprocess", types.SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    spec = "Security"
    facts, lock, report = T.fetch_run("r1", tmp_path / spec)
    assert asked == [f"lean-audit/audit-{spec}/{f}"
                     for f in ("facts.json.gz", "facts.json", "lean-audit.json", "report.json")]
    assert (facts.name, lock.name, report.name) == (kept, "lean-audit.json", "report.json") and facts.is_file()
    asked.clear()
    T.fetch_run("r1", T.ROOT / spec / "Properties")
    assert asked[0] == f"lean-audit/audit-{spec}_Properties/facts.json.gz", \
        "a nested package's reports are under its path's tag, as check.py names them"
