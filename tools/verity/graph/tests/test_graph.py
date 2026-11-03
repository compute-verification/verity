"""The listing on a toy checkout: what the core adds to a family's tasks, what it refuses, and that its bytes depend on the
commit alone."""
from __future__ import annotations

import subprocess

import pytest
from conftest import LISTERS, SPEC, git, toy, toy_tree

from verity_graph import contract, graph


def test_the_listing_completes_digests_and_keys(tree):
    doc = graph.listing(tree, listers=LISTERS)
    assert doc["schema"] == "verity-graph/v2" and doc["commit"] == git(tree, "rev-parse", "HEAD")
    assert [t["id"] for t in doc["tasks"]] == ["other:x", "toy-b:b", "toy:a"]
    a = next(t for t in doc["tasks"] if t["id"] == "toy:a")
    assert list(a) == list(graph.TASK_FIELDS)
    assert a["argv"] == ["verity", "graph", "run", "toy:a"] and a["vars"] == {} and a["env"] == "py"
    assert a["inputs"] == sorted(["src/a.txt", *graph.runner_sources("toy")])
    assert a["about"] == "toy a" and a["timeout_s"] == 60 and a["resources"] == {"cores": 1, "mem_gb": 1, "disk_gb": 1}
    assert a["key"] == contract.key(a, doc["files"], None, doc["envs"]["py"]["digest"])
    b = next(t for t in doc["tasks"] if t["id"] == "toy-b:b")
    assert b["key"] is None, "a task that reads an artifact is keyed by the executor, once the read's digest exists"
    x = next(t for t in doc["tasks"] if t["id"] == "other:x")
    assert x["about"] == "other x with plain" and x["env"] == "lean"
    assert doc["envs"]["lean"]["spec"] == ["lib/lean-toolchain"] and doc["envs"]["py"]["spec"] == ["pyproject.toml", "uv.lock"]
    assert doc["envs"]["py"]["digest"] == contract.env_digest(["uv.lock", "pyproject.toml"], doc["files"])
    want = {p for t in doc["tasks"] for p in t["inputs"]} | {"lib/lean-toolchain", "pyproject.toml", "uv.lock"}
    assert set(doc["files"]) == want, "files holds every input of a task or an env, once, and nothing else"
    assert doc["files"]["src/a.txt"] == contract.digest(b"a\n")


def test_two_checkouts_of_one_commit_list_the_same_bytes(tree, tmp_path, monkeypatch):
    other = tmp_path / "deeper" / "elsewhere"
    subprocess.run(["git", "clone", "-q", str(tree), str(other)], check=True)
    one = contract.canonical(graph.listing(tree, listers=LISTERS))
    monkeypatch.chdir(other.parent)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VERITY_GRAPH_TOY", "1")
    assert contract.canonical(graph.listing(other, listers=LISTERS)) == one


def test_a_dirty_tree_names_no_commit(tree):
    (tree / "src/a.txt").write_text("changed\n")
    doc = graph.listing(tree, listers=LISTERS)
    assert doc["commit"] is None and doc["files"]["src/a.txt"] == contract.digest(b"changed\n")
    (tree / "src/a.txt").write_text("a\n")
    (tree / "new.txt").write_text("untracked\n")
    assert graph.listing(tree, listers=LISTERS)["commit"] is None


def test_a_family_and_what_it_reads(tree):
    doc = graph.listing(tree, {"other"}, listers=LISTERS)
    assert [t["id"] for t in doc["tasks"]] == ["other:x"]
    reader = {**LISTERS, "other": lambda root, files: [{"id": "other:y", "vars": {"flavor": "f"},
                                                         "reads": [{"task": "toy-b:b", "output": "o"}]}]}
    doc = graph.listing(tree, {"other"}, listers=reader)
    assert [t["id"] for t in doc["tasks"]] == ["other:y", "toy-b:b", "toy:a"]
    assert [t["id"] for t in graph.closure(doc, {"toy-b:b"})] == ["toy:a", "toy-b:b"]
    with pytest.raises(graph.GraphError, match="no family nope"):
        graph.listing(tree, {"nope"}, listers=LISTERS)


def test_a_task_s_graph_is_the_part_of_the_listing_that_running_it_needs(tree):
    doc = graph.listing(tree, listers=LISTERS)
    by = {t["id"]: t for t in doc["tasks"]}
    for tid, want in (("toy-b:b", ["toy-b:b", "toy:a"]), ("toy:a", ["toy:a"]), ("other:x", ["other:x"])):
        part = graph.task_graph(tree, tid, listers=LISTERS)
        assert [t["id"] for t in part["tasks"]] == want and [by[i] for i in want] == part["tasks"]
        assert part["envs"] == doc["envs"] and part["commit"] is None
        assert part["files"] == {p: doc["files"][p] for p in sorted({p for t in part["tasks"] for p in t["inputs"]} | {
            p for e in doc["envs"].values() for p in e["spec"]})}
    refused = {**LISTERS, "toy": lambda root, files: [*toy(root, files), {"id": "toy:c", "owner": "@ci"}]}
    assert graph.task_graph(tree, "toy:a", listers=refused)["tasks"] == [by["toy:a"]], \
        "the rest of the family is neither completed nor checked: the listing is"
    with pytest.raises(graph.GraphError, match="no task toy:zz on this tree"):
        graph.task_graph(tree, "toy:zz", listers=LISTERS)


def test_covered_files_follow_the_table(tree):
    assert graph.covered(tree, "toy", graph.tracked(tree)) == ["src/a.txt", "src/b.txt"]
    assert graph.covered(tree, "other", graph.tracked(tree)) == ["lib/lean-toolchain", "lib/x.lean"]
    assert graph.matches("lean-toolchain", "**/lean-toolchain") and graph.matches("a/b/lean-toolchain", "**/lean-toolchain")


def test_a_declared_family_without_a_lister_lists_nothing(tree):
    assert [t["id"] for t in graph.listing(tree, listers={"toy": toy})["tasks"]] == ["toy-b:b", "toy:a"]


def _one(**task):
    return {"toy": lambda root, files: [{"id": "toy:a", "inputs": ["src/a.txt"], **task}]}


@pytest.mark.parametrize("task, error", [
    ({"owner": "@ci"}, "`owner` field"),
    ({"detail": {"owner": "x"}}, "`owner` field"),
    ({"key": "sha512:00"}, "leaves `key` to the core"),
    ({"kind": "other"}, "isn't one of \\[family.toy\\]'s kinds"),
    ({"id": "toy-b:a"}, "a task's id is `<kind>:<name>`"),
    ({"id": "toy:"}, "a task's id is `<kind>:<name>`"),
    ({"env": "rust"}, "env 'rust' isn't one of"),
    ({"inputs": ["src/nope.txt"]}, "aren't files of the tree"),
    ({"inputs": ["src/a.txt", "src/a.txt"]}, "distinct paths"),
    ({"vars": {"n": 1}}, "vars is"),
    ({"argv": []}, "argv is a non-empty list"),
    ({"reads": [{"task": "toy:z", "output": "o"}]}, "isn't a task"),
    ({"outputs": [{"name": "o", "path": "../x"}]}, "not a relative POSIX path"),
    ({"outputs": [{"name": "o", "path": "out/x"}, {"name": "o", "path": "out/y"}]}, "distinct names"),
    ({"outputs": [{"name": "o", "path": "out"}, {"name": "p", "path": "out/y"}]}, "lands inside"),
    ({"outputs": [{"name": "o", "path": graph.LOG}]}, "both land at"),
    ({"resources": {"cores": 0}}, "resources are"),
    ({"timeout_s": 1.5}, "timeout_s is a positive integer"),
])
def test_the_core_refuses_what_is_not_graph_v2(tree, task, error):
    with pytest.raises(graph.GraphError, match=error):
        graph.listing(tree, listers=_one(**task))


def test_reads_that_cycle_or_collide_are_refused(tree):
    cycle = {"toy": lambda root, files: [
        {"id": "toy:a", "reads": [{"task": "toy:b", "output": "o"}], "outputs": [{"name": "o", "path": "out/a"}]},
        {"id": "toy:b", "reads": [{"task": "toy:a", "output": "o"}], "outputs": [{"name": "o", "path": "out/b"}]}]}
    with pytest.raises(graph.GraphError, match="cycle through toy:a, toy:b"):
        graph.listing(tree, listers=cycle)
    collide = {"toy": lambda root, files: [
        {"id": "toy:a", "outputs": [{"name": "o", "path": "out/a"}]},
        {"id": "toy:b", "reads": [{"task": "toy:a", "output": "o"}], "outputs": [{"name": "p", "path": "out/a"}]}]}
    with pytest.raises(graph.GraphError, match="both land at out/a"):
        graph.listing(tree, listers=collide)
    twice = {"toy": lambda root, files: [{"id": "toy:a"}, {"id": "toy:a"}]}
    with pytest.raises(graph.GraphError, match="listed twice"):
        graph.listing(tree, listers=twice)


def test_the_tables_are_checked(tmp_path):
    with pytest.raises(graph.GraphError, match="`owner` field"):
        graph.spec(toy_tree(tmp_path / "a", SPEC + '\n[family.x]\nowner = "@ci"\n'))
    with pytest.raises(graph.GraphError, match="needs `files`"):
        graph.spec(toy_tree(tmp_path / "b", SPEC + "\n[family.x]\nkinds = ['x']\n"))
    with pytest.raises(graph.GraphError, match="a lister with no \\[family.\\*\\] table"):
        graph.listing(toy_tree(tmp_path / "c"), listers={**LISTERS, "gone": toy})


def test_a_kind_needs_its_runner(tree):
    (tree / graph.RUNNERS / "toy_b.py").unlink()
    git(tree, "commit", "-qam", "no runner")
    with pytest.raises(graph.GraphError, match="kind toy-b has no runner"):
        graph.listing(tree, listers=LISTERS)


def test_the_real_tables_parse():
    """This checkout's tasks.toml: every family the contract names, with a kind each and envs py, lean and rust."""
    from verity_graph import ROOT, families

    s = graph.spec(ROOT)
    assert {"test-file", "lean-module", "lean-lock", "circuit-binding", "build"} <= set(s["family"])
    assert set(s["env"]) == {"py", "lean", "rust"}
    assert set(families.listers()) <= set(s["family"]), "every lister in families/ has its table"
