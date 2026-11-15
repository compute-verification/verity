"""The contract's invariants on this checkout's whole graph: the same bytes from two checkouts of one commit at different paths,
nothing opened outside the tree, no `owner` field, every key the rule's, and a task run's graph the listing's. Slow: each lists
the whole tree, which the toy checkout's tests (`test_graph.py`) can't stand in for, since every family lists here."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity_graph import ROOT, contract, graph

HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.slow


def _listing(root: Path, cwd: Path) -> bytes:
    p = subprocess.run([sys.executable, "-m", "verity_graph", "--json", "--root", str(root)], cwd=cwd, capture_output=True)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


@pytest.fixture(scope="module")
def audited(tmp_path_factory) -> tuple[dict, list[str]]:
    """This checkout's listing under an audit hook: (its graph, the paths it opened outside the tree)."""
    out = tmp_path_factory.mktemp("audit") / "outside.json"
    p = subprocess.run([sys.executable, str(HERE / "audit_listing.py"), str(ROOT), str(out)], cwd=ROOT, capture_output=True)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout), json.loads(out.read_text())


def test_two_checkouts_of_one_commit_at_different_paths_list_the_same_bytes(tmp_path, monkeypatch):
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    a, b = tmp_path / "a", tmp_path / "deeper" / "elsewhere" / "b"
    try:
        for w in (a, b):
            subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach", str(w), head], check=True)
        one = _listing(a, a)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("TZ", "Pacific/Kiritimati")
        monkeypatch.setenv("LC_ALL", "C")
        monkeypatch.setenv("VERITY_GRAPH_PROBE", "1")
        two = _listing(b, tmp_path)
    finally:
        for w in (a, b):
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(w)], capture_output=True)
    assert one == two, "the listing depends on where the checkout is, the working directory or the environment"
    assert json.loads(one)["commit"] == head


def test_the_listing_opens_nothing_outside_the_tree(audited):
    _doc, outside = audited
    assert outside == [], f"the listing read the machine, not the commit: {outside[:20]}"


def test_no_owner_field_anywhere(audited):
    doc, _outside = audited

    def walk(x, where):
        if isinstance(x, dict):
            assert graph.FORBIDDEN not in x, f"{where} has an `{graph.FORBIDDEN}` field"
            for k, v in x.items():
                walk(v, f"{where}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{where}[{i}]")
    walk(doc, "graph")
    assert graph.FORBIDDEN not in (ROOT / graph.SPEC).read_text()


def test_every_key_is_the_rule_s(audited):
    doc, _outside = audited
    assert doc["schema"] == contract.SCHEMA_GRAPH
    for t in doc["tasks"]:
        want = None if t["reads"] else contract.key(t, doc["files"], None, doc["envs"][t["env"]]["digest"])
        assert t["key"] == want, t["id"]
    for name, e in doc["envs"].items():
        assert e["digest"] == contract.env_digest(e["spec"], doc["files"]), name


def test_the_packed_listing_unpacks_to_the_listing_and_is_a_tenth_of_its_bytes_at_most(audited):
    doc, _outside = audited
    packed = graph.pack(doc)
    assert graph.unpack(packed) == doc
    assert 10 * len(contract.canonical(packed)) < len(contract.canonical(doc)), "the tasks' inputs and reads no longer pack"


def test_what_a_task_run_lists_is_the_listing_s(audited, tmp_path):
    """`verity graph run`'s graph for a task of each kind, listed (`graph.task_graph`): the listing's tasks, each with the
    priority of the chains that graph holds, its digests and envs; and handed its part of the listing (`graph.part`), taken here
    (`graph.given`)."""
    doc, _outside = audited
    by = {t["id"]: t for t in doc["tasks"]}
    for tid in sorted({t["kind"]: t["id"] for t in doc["tasks"]}.values()):
        part = graph.task_graph(ROOT, tid)
        assert [{**by[t["id"]], "priority": t["priority"]} for t in part["tasks"]] == part["tasks"], tid
        assert {t["id"]: t["priority"] for t in part["tasks"]} == graph.priorities(part["tasks"]), tid
        assert part["envs"] == doc["envs"] and part["files"] == {p: doc["files"][p] for p in part["files"]}, tid
        (tmp_path / "part.json").write_bytes(contract.canonical(graph.part(doc, tid, by)))
        assert graph.given(ROOT, tmp_path / "part.json", tid) == graph.part(doc, tid, by), tid
