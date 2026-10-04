"""A module move leaves the circuits alone: every Definition id and canonical descriptor digest, circuit-check's task list
(target, statics, pin) and the frontend vocabularies' digests are those of the snapshot taken on `main` before the split
(`fixtures/migration/`, written by `migration_snapshot.py`).

A Definition, target or vocabulary added after the snapshot does not fail this; one that moved, changed its digest or
dropped out of what `circuit-check --all` checks does.  At the split itself the two snapshots are equal (the PR shows it).
The snapshot is taken in a fresh process because the order matters: tracing the catalog's roots registers lazy families'
members, and this suite's other modules trace them at import.

It takes about 95 s (one `suite()` and one more `catalog()` trace); no cheaper test catches a split that renames a
Definition by moving its body to another module, or a registry module moved out of what `load_registries` imports.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import migration_snapshot as M
import pytest

HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pair() -> tuple[dict, dict]:
    out = subprocess.run([sys.executable, str(HERE / "migration_snapshot.py")], check=True, capture_output=True, text=True).stdout
    return M.read(HERE / "fixtures" / "migration"), json.loads(out)


def test_every_definition_keeps_its_id_and_digest(pair):
    before, now = pair
    moved = {fid: (d, now["definitions"].get(fid)) for fid, d in before["definitions"].items() if now["definitions"].get(fid) != d}
    assert moved == {}, f"{len(moved)} Definitions whose descriptor digest is not the snapshot's (before, now): {dict(list(moved.items())[:20])}"
    assert sorted(set(before["registered"]) - set(now["registered"])) == []
    assert sorted(set(before["references"]) - set(now["references"])) == []


def test_circuit_check_keeps_every_task(pair):
    before, now = pair
    tasks = {(k, i): rest for k, i, rest in now["tasks"]}
    lost = [t for t in before["tasks"] if tasks.get((t[0], t[1]), object()) != t[2]]
    assert lost == [], f"{len(lost)} circuit-check tasks gone or changed (kind, target, pin or statics): {lost[:20]}"
    assert sorted(set(now["unresolved_pieces"]) - set(before["unresolved_pieces"])) == []


def test_every_vocabulary_keeps_its_digest(pair):
    before, now = pair
    assert {k: v for k, v in now["vocabularies"].items() if k in before["vocabularies"]} == before["vocabularies"]
