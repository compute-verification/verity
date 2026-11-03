"""The contract's invariants on this checkout's whole graph: the same bytes from two checkouts of one commit at different paths,
nothing opened outside the tree, no `owner` field, every key the rule's, a task run's graph the listing's, and the listing
within a minute of one core, as it is and with its Lean modules ported to the module system. Slow: each lists the whole tree,
which the toy checkout's tests (`test_graph.py`) can't stand in for, since every family lists here."""
from __future__ import annotations

import json
import resource
import subprocess
import sys
from pathlib import Path

import pytest

from verity_graph import ROOT, contract, graph

HERE = Path(__file__).resolve().parent
#: the listing's budget on one core, in CPU seconds (the contract's invariant 5)
BUDGET_S = 60

pytestmark = pytest.mark.slow


def _listing(root: Path, cwd: Path) -> bytes:
    p = subprocess.run([sys.executable, "-m", "verity_graph", "--json", "--root", str(root)], cwd=cwd, capture_output=True)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


@pytest.fixture(scope="module")
def audited(tmp_path_factory) -> tuple[dict, list[str], float]:
    """This checkout's listing under an audit hook: (its graph, the paths it opened outside the tree, its CPU seconds)."""
    out = tmp_path_factory.mktemp("audit") / "outside.json"
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    p = subprocess.run([sys.executable, str(HERE / "audit_listing.py"), str(ROOT), str(out)], cwd=ROOT, capture_output=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return json.loads(p.stdout), json.loads(out.read_text()), cpu


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
    _doc, outside, _cpu = audited
    assert outside == [], f"the listing read the machine, not the commit: {outside[:20]}"


def test_no_owner_field_anywhere(audited):
    doc, _outside, _cpu = audited

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
    doc, _outside, _cpu = audited
    assert doc["schema"] == contract.SCHEMA_GRAPH
    for t in doc["tasks"]:
        want = None if t["reads"] else contract.key(t, doc["files"], None, doc["envs"][t["env"]]["digest"])
        assert t["key"] == want, t["id"]
    for name, e in doc["envs"].items():
        assert e["digest"] == contract.env_digest(e["spec"], doc["files"]), name


def test_what_a_task_run_lists_is_the_listing_s(audited, tmp_path):
    """`verity graph run`'s graph for a task of each kind, listed (`graph.task_graph`): the listing's tasks, digests and envs;
    and handed its part of the listing (`graph.part`), taken here (`graph.given`)."""
    doc, _outside, _cpu = audited
    by = {t["id"]: t for t in doc["tasks"]}
    for tid in sorted({t["kind"]: t["id"] for t in doc["tasks"]}.values()):
        part = graph.task_graph(ROOT, tid)
        assert [by[t["id"]] for t in part["tasks"]] == part["tasks"] and part["envs"] == doc["envs"], tid
        assert part["files"] == {p: doc["files"][p] for p in part["files"]}, tid
        (tmp_path / "part.json").write_bytes(contract.canonical(graph.part(doc, tid, by)))
        assert graph.given(ROOT, tmp_path / "part.json", tid) == graph.part(doc, tid, by), tid


def test_the_listing_takes_under_a_minute_of_one_core(audited):
    """CPU seconds of the listing's processes (getrusage of the children): what one core spends, whatever the machine's load."""
    doc, _outside, cpu = audited
    assert cpu < BUDGET_S, f"listing {len(doc['tasks'])} tasks took {cpu:.1f} CPU seconds, past the contract's {BUDGET_S}"


def test_the_listing_with_its_lean_modules_ported_takes_under_a_minute_of_one_core(tmp_path):
    """Invariant 5 with the Lean families at their size once the tree's modules are `module` files (`ported_listing.py`: two
    read entries per module a task reads, seven outputs per module). What it catches that the test above can't: a core or a
    Lean family that fits the budget today and won't after the port."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(tmp_path / "graph.json", "wb") as out:
        p = subprocess.run([sys.executable, str(HERE / "ported_listing.py"), str(ROOT)], cwd=ROOT, stdout=out,
                           stderr=subprocess.PIPE)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if p.returncode == 3:
        pytest.skip("no Lean family lists on this tree")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    assert cpu < BUDGET_S, (f"listing {p.stderr.decode().strip()} with the Lean modules ported took {cpu:.1f} CPU seconds, "
                            f"past the contract's {BUDGET_S}")
