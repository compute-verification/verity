"""`tasks.py` (`verity tasks`): the task graph of a synthetic checkout, and one task run in it.  The checkout holds a copy of the
check's tooling (this directory, `tools/verity/lean`'s jobs and the task graph's contract they key with, and the handle registry
`ci.toml`'s owners are checked against) beside two suites and a Lake package, and `tasks.py` runs in a process of its own there,
as it runs on the repository's tree."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
FIELDS = {"id", "kind", "argv", "cwd", "reason", "inputs", "needs", "outputs", "deps", "beside", "resources", "cost_s", "key",
          "unkeyed"}
KINDS = {"suite", "lean-package", "circuit-check", "build", "step"}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    t = tmp_path / "tree"
    shutil.copytree(ROOT / "tools" / "verity" / "check", t / "tools" / "verity" / "check",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.md"))
    (t / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in (ROOT / "tools" / "verity" / "lean").glob("*.py"):
        shutil.copy2(f, t / "tools" / "verity" / "lean" / f.name)
    contract = Path("tools") / "verity" / "graph" / "verity_graph" / "contract.py"
    (t / contract).parent.mkdir(parents=True)
    shutil.copy2(ROOT / contract, t / contract)
    registry = Path(".agents") / "skills" / "using-slack" / "registry.json"
    (t / registry).parent.mkdir(parents=True)
    shutil.copy2(ROOT / registry, t / registry)
    (t / ".gitignore").write_text("__pycache__/\n")
    (t / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["a", "b"]\n')
    for name in ("a", "b"):
        (t / name / "tests").mkdir(parents=True)
        (t / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n\n[tool.uv]\npackage = false\n\n'
                                                 '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
        (t / name / "tests" / f"test_{name}.py").write_text(f"def test_{name}():\n    assert True\n")
    (t / "lean" / "P").mkdir(parents=True)
    (t / "lean" / "P" / "lakefile.toml").write_text('name = "P"\n')
    (t / "lean" / "P" / "lean-audit.json").write_text("{}\n")
    (t / "lean" / "P" / "P.lean").write_text("theorem t : True := trivial\n")
    # its own `verity`, without the twins `check.PREBUILD` loads, so a run's prebuild builds nothing
    (t / "verity").mkdir()
    (t / "verity" / "__init__.py").write_text("")
    _git(t, "init", "-q")
    _git(t, "add", "-A")
    _git(t, "commit", "-q", "-m", "tree")
    return t


def _tasks(tree: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(tree.parent / "home"), "CHECK_CACHE": str(tree.parent / "check-cache"),
           "VERITY_TEST_CACHE": str(tree.parent / "test-cache")}
    return subprocess.run([sys.executable, str(tree / "tools" / "verity" / "check" / "tasks.py"), *args], cwd=tree, env=env,
                          capture_output=True, text=True)


def _graph(tree: Path, *args: str) -> tuple[dict, str]:
    r = _tasks(tree, "list", "--json", *args)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), r.stdout


def _keys(tree: Path) -> dict[str, str | None]:
    return {t["id"]: t["key"] for t in _graph(tree)[0]["tasks"]}


def test_list_is_a_tasks_v1_graph_closed_under_its_deps_and_the_same_each_time(tree):
    doc, text = _graph(tree)
    assert doc["schema"] == "tasks/v1" and doc["kind"] == "after-landing" and doc["circuit_shards"] == 4
    ids = [t["id"] for t in doc["tasks"]]
    assert len(ids) == len(set(ids))
    made = {a for t in doc["tasks"] for a in t["outputs"]}
    for t in doc["tasks"]:
        assert set(t) == FIELDS and t["kind"] in KINDS, t["id"]
        assert set(t["deps"]) <= set(ids) and set(t["beside"]) <= set(ids) and set(t["needs"]) <= made, t["id"]
        assert (t["argv"] is None) == (t["reason"] is not None), t["id"]
        assert (t["key"] is None) == isinstance(t["unkeyed"], str), t["id"]
        assert t["key"] is None or re.fullmatch(r"[0-9a-f]{128}", t["key"]), t["id"]
        assert set(t["resources"]) == {"cores", "mem_gb", "disk_gb"} and t["cost_s"] is None
        assert isinstance(t["inputs"]["paths"], list) and t["inputs"]["paths"]
    by = {t["id"]: t for t in doc["tasks"]}
    assert {"pytest:a", "pytest:b", "lean-audit:lean/P", "lean-build", "flock-circuit-build", "ratchet-gate",
            *(f"circuit-check:{k}/4" for k in range(1, 5))} <= set(ids)
    a = by["pytest:a"]
    assert (a["kind"], a["cwd"], a["argv"][:3]) == ("suite", "<tree>", ["<python>", "<tree>/tools/verity/check/suites.py", "a"])
    assert a["key"] and "a" in a["inputs"]["paths"] and "b" not in a["inputs"]["paths"]
    assert by["flock-rust-tests"]["deps"] == ["flock-circuit-build"] and by["flock-rust-tests"]["needs"] == \
        by["flock-circuit-build"]["outputs"]
    assert by["lean-build"]["key"] is None and by["lean-build"]["kind"] == "build"
    assert str(tree) not in text, "a task names the tree's path, so its key is of where it is, not what it is"
    assert _graph(tree)[1] == text, "the same tree lists the same graph"

    preflight = {t["id"] for t in _graph(tree, "--kind", "preflight")[0]["tasks"]}
    assert "pytest:a" in preflight and not any(i.startswith(("circuit-check", "ratchet-gate")) for i in preflight), \
        "the jobs off the merge gate are in no preflight"
    assert [t["id"] for t in _graph(tree, "--circuit-shards", "2")[0]["tasks"] if t["kind"] == "circuit-check"] == \
        ["circuit-check:1/2", "circuit-check:2/2"]


def test_a_lean_audit_asks_the_large_slot_only_when_it_reads_security(tree):
    """Only an audit of `Security` or of a package requiring it (Mathlib, ArkLib) asks 32 cores and 128 GB: the first graph's 26
    others queued for the two large slots while 21 small ones sat idle."""
    for name, deps in (("Security", []), ("lean/Q", ["../../Security"])):
        (tree / name).mkdir(parents=True)
        (tree / name / "lakefile.toml").write_text(f'name = "{name.rsplit("/", 1)[-1]}"\n')
        (tree / name / "lean-audit.json").write_text("{}\n")
        (tree / name / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "dir": d} for d in deps]}))
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "security")
    by = {t["id"]: t["resources"] for t in _graph(tree)[0]["tasks"] if t["kind"] == "lean-package"}
    assert by == {"lean-audit:lean/P": {"cores": 16, "mem_gb": 64, "disk_gb": 60},
                  "lean-audit:Security": {"cores": 32, "mem_gb": 128, "disk_gb": 60},
                  "lean-audit:lean/Q": {"cores": 32, "mem_gb": 128, "disk_gb": 60}}, by


def test_a_suites_key_moves_with_its_inputs_and_with_nothing_else(tree):
    before = _keys(tree)
    (tree / "notes.txt").write_text("outside every suite\n")
    (tree / "a" / "README.md").write_text("prose, which no suite here reads\n")
    assert _keys(tree) == before
    (tree / "a" / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")      # uncommitted: it counts
    after = _keys(tree)
    assert after["pytest:a"] != before["pytest:a"]
    assert {i: k for i, k in after.items() if i != "pytest:a"} == {i: k for i, k in before.items() if i != "pytest:a"}


def test_changed_keeps_the_tasks_whose_inputs_the_change_touches(tree):
    assert _graph(tree, "--changed", "HEAD")[0]["tasks"] == [], "no change touches nothing"
    (tree / "b" / "tests" / "test_b.py").write_text("def test_b():\n    assert 1\n")
    doc = _graph(tree, "--changed", "HEAD")[0]
    ids = {t["id"] for t in doc["tasks"]}
    assert doc["changed"]["files"] == 1
    assert "pytest:b" in ids and "pytest:a" not in ids and "lean-audit:lean/P" not in ids
    assert "lean-changed" in ids, "a job that declares no inputs reads the whole tree, so every change touches it"
    every = {t["id"]: t for t in _graph(tree)[0]["tasks"]}
    assert all(set(every[i]["deps"]) <= ids for i in ids), "a kept task's deps are kept"


def test_run_of_a_suite_task_returns_its_verdict_and_its_listed_key(tree):
    listed = _keys(tree)
    out = tree.parent / "a.json"
    r = _tasks(tree, "run", "pytest:a", "--out", str(out), "--work", str(tree.parent / "work-a"))
    res = json.loads(out.read_text())
    assert r.returncode == 0, r.stderr
    assert (res["schema"], res["id"], res["verdict"], res["key"]) == ("tasks/v1", "pytest:a", "passed", listed["pytest:a"])
    assert [(s["step"], s["status"]) for s in res["steps"]] == [("pytest", "passed")] and res["seconds"] >= 0
    assert "suites.json" in res["outputs"]["files"]

    (tree / "b" / "tests" / "test_b.py").write_text("def test_b():\n    assert False\n")
    r = _tasks(tree, "run", "pytest:b", "--work", str(tree.parent / "work-b"))
    assert r.returncode == 1 and json.loads(r.stdout)["verdict"] == "failed"

    r = _tasks(tree, "run", "pytest:nothing")
    assert r.returncode == 2 and "no task pytest:nothing" in r.stderr


def test_a_suite_past_the_threshold_is_listed_in_parts_by_file_and_a_part_runs_under_its_listed_key(tree):
    for n in (2, 3):
        (tree / "a" / "tests" / f"test_a{n}.py").write_text(f"def test_a{n}():\n    assert True\n")
    costs = tree.parent / "costs.json"
    costs.write_text(json.dumps({"pytest:a": 2000}))
    doc, _ = _graph(tree, "--costs", str(costs), "--split-s", "900")
    by = {t["id"]: t for t in doc["tasks"]}
    ids = [f"pytest:a:{k}/3" for k in (1, 2, 3)]
    assert set(ids) <= set(by) and "pytest:a" not in by and "pytest:b" in by, "a whole suite under the threshold stays whole"
    split = doc["splits"]["a"]
    assert (doc["split_s"], split["m"], sorted(f for p in split["files"] for f in p)) == \
        (900, 3, ["a/tests/test_a.py", "a/tests/test_a2.py", "a/tests/test_a3.py"])
    assert [round(by[i]["cost_s"], 1) for i in ids] == [666.7] * 3
    assert len({by[i]["key"] for i in ids}) == 3
    for k, i in enumerate(ids, 1):
        assert f"a:{k}/3@{split['digest'][:12]}" in by[i]["argv"], "a part's argv pins the partition it was listed of"

    timings = tree.parent / "timings.json"
    timings.write_text(json.dumps({n: v["timings"] for n, v in doc["splits"].items()}))
    out = tree.parent / "a2.json"
    r = _tasks(tree, "run", "pytest:a:2/3", "--timings", str(timings), "--out", str(out), "--work", str(tree.parent / "work-a2"))
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert (res["verdict"], res["key"]) == ("passed", by["pytest:a:2/3"]["key"])
    (row,) = json.loads((tree.parent / "work-a2" / "suites.json").read_text())
    assert row["part"]["ks"] == [2] and row["counts"]["passed"] == 1, "one of three files: the part's own"
