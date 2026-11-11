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
    # its own `verity`, whose twins (`check.PREBUILD`) build nothing
    (t / "verity" / "definitions" / "numerics").mkdir(parents=True)
    for d in ("verity", "verity/definitions", "verity/definitions/numerics"):
        (t / d / "__init__.py").write_text("")
    (t / "verity/definitions/numerics/twins.py").write_text("def preload():\n    return {}\n")
    _git(t, "init", "-q")
    _git(t, "add", "-A")
    _git(t, "commit", "-q", "-m", "tree")
    return t


def _tasks(tree: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(tree.parent / "home"), "CHECK_CACHE": str(tree.parent / "check-cache"),
           "VERITY_TEST_CACHE": str(tree.parent / "test-cache"), **(env or {})}
    return subprocess.run([sys.executable, str(tree / "tools" / "verity" / "check" / "tasks.py"), *args], cwd=tree, env=env,
                          capture_output=True, text=True)


def _graph(tree: Path, *args: str, env: dict[str, str] | None = None) -> tuple[dict, str]:
    r = _tasks(tree, "list", "--json", *args, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), r.stdout


def _keys(tree: Path, env: dict[str, str] | None = None) -> dict[str, str | None]:
    return {t["id"]: t["key"] for t in _graph(tree, env=env)[0]["tasks"]}


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
    spec = "Security"
    for name, deps in ((spec, []), ("lean/Q", [f"../../{spec}"])):
        (tree / name).mkdir(parents=True)
        (tree / name / "lakefile.toml").write_text(f'name = "{name.rsplit("/", 1)[-1]}"\n')
        (tree / name / "lean-audit.json").write_text("{}\n")
        (tree / name / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "dir": d} for d in deps]}))
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "security")
    by = {t["id"]: t["resources"] for t in _graph(tree)[0]["tasks"] if t["kind"] == "lean-package"}
    assert by == {"lean-audit:lean/P": {"cores": 16, "mem_gb": 64, "disk_gb": 60},
                  f"lean-audit:{spec}": {"cores": 32, "mem_gb": 128, "disk_gb": 60},
                  "lean-audit:lean/Q": {"cores": 32, "mem_gb": 128, "disk_gb": 60}}, by


def test_a_lean_audit_fails_on_a_lock_record_difference_as_the_after_landing_run_does(tree, monkeypatch):
    """An audit task never regenerates lock records (`--no-regenerate`, drive's after-landing run), listed or run: with
    `--regenerate` it passed `Security` on a stale lock that the after-landing run of the same commit failed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("tasks_under_test", tree / "tools" / "verity" / "check" / "tasks.py")
    T = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(tree / "tools" / "verity" / "check"))
    spec.loader.exec_module(T)
    seen = []
    monkeypatch.setattr(T.C, "groups", lambda *a, **k: seen.append(k.get("regenerate", True)) or [])
    monkeypatch.setattr(T.C, "run", lambda *a, **k: seen.append(k.get("regenerate", True)) or (True, []))
    T.command("lean-audit:lean/P", 16)
    T.run("lean-audit:lean/P", tree / "work")
    assert len(seen) >= 2 and not any(seen), seen


def test_a_suites_key_moves_with_its_inputs_and_with_nothing_else(tree):
    before = _keys(tree)
    (tree / "notes.txt").write_text("outside every suite\n")
    (tree / "a" / "README.md").write_text("prose, which no suite here reads\n")
    assert _keys(tree) == before
    (tree / "a" / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")      # uncommitted: it counts
    after = _keys(tree)
    assert after["pytest:a"] != before["pytest:a"]
    assert {i: k for i, k in after.items() if i != "pytest:a"} == {i: k for i, k in before.items() if i != "pytest:a"}


def test_a_kept_steps_key_is_its_inputs_not_the_slot_or_the_toolchain_it_is_listed_on(tree):
    """A job run whole whose pass check keeps (`check.STEP_KEYS`) lists one key on every slot and host: keyed by its step key,
    which reads the slot's test cache, the kernel release and `rustc -vV`, `flock-rust-tests` passed on a worker's slot under
    another key than the lister's, so its verdict was never labelled."""
    flock = tree / "verity" / "protocols" / "flock" / "prover"
    (flock / "live").mkdir(parents=True)
    (flock / "check_build.sh").write_text("RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.0.0}\n")
    (flock / "live" / "lib.rs").write_text("pub fn f() {}\n")
    (flock / "flock-a-b684b12.patch").write_text("a\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "flock")

    def slot(name: str, rustc: str) -> dict[str, str]:
        home, tmp = tree.parent / f"home-{name}", tree.parent / f"tmp-{name}"
        (home / ".cargo" / "bin").mkdir(parents=True)
        tmp.mkdir()
        (home / ".cargo" / "bin" / "rustc").write_text(f"#!/bin/sh\necho 'rustc {rustc}'\n")
        (home / ".cargo" / "bin" / "rustc").chmod(0o755)
        return {"HOME": str(home), "VERITY_TEST_CACHE": str(tree.parent / f"tests-slot-{name}"), "TMPDIR": str(tmp)}

    key = _keys(tree, slot("x3", "1.0.0 (one build)"))["flock-rust-tests"]
    assert key and _keys(tree, slot("c4la0", "1.0.0 (another build)"))["flock-rust-tests"] == key, \
        "a lister and a worker on other slots, with other homes and toolchains, list one key"
    (tree / "a" / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
    assert _keys(tree)["flock-rust-tests"] == key, "a file outside its inputs leaves its key"
    (flock / "flock-a-b684b12.patch").write_text("b\n")
    assert _keys(tree)["flock-rust-tests"] != key, "a file its inputs' glob matches moves its key"


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


def test_run_with_the_graph_runs_any_id_it_lists_as_listed_and_refuses_another_commits_or_an_unlisted_id(tree):
    """A worker whose own record doesn't split a suite still runs the part another worker listed, of the listed partition and
    under the listed key: deciding splits from the runner's record failed ids listed elsewhere (`no task ... on this tree`)."""
    for n in (2, 3):
        (tree / "a" / "tests" / f"test_a{n}.py").write_text(f"def test_a{n}():\n    assert True\n")
    costs = tree.parent / "costs.json"
    costs.write_text(json.dumps({"pytest:a": 2000}))
    doc, text = _graph(tree, "--costs", str(costs), "--split-s", "900")
    by = {t["id"]: t for t in doc["tasks"]}
    assert "pytest:a:3/3" in by and "pytest:a" in _keys(tree), "this machine's own record lists a whole"
    graph = tree.parent / "graph.json"
    graph.write_text(text)

    out = tree.parent / "a3.json"
    r = _tasks(tree, "run", "pytest:a:3/3", "--graph", str(graph), "--out", str(out), "--work", str(tree.parent / "work-a3"))
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert (res["id"], res["verdict"], res["key"]) == ("pytest:a:3/3", "passed", by["pytest:a:3/3"]["key"])
    assert "differs from the graph's" not in r.stderr
    (row,) = json.loads((tree.parent / "work-a3" / "suites.json").read_text())
    assert row["part"]["ks"] == [3] and row["counts"]["passed"] == 1, "the listed part's one file"

    r = _tasks(tree, "run", "pytest:a", "--graph", str(graph))
    assert r.returncode == 2 and "no task pytest:a in the graph given" in r.stderr, r.stderr
    r = _tasks(tree, "run", "pytest:a:3/3", "--graph", str(graph), "--timings", str(costs))
    assert r.returncode == 2 and "no --timings or --circuit-shards" in r.stderr, r.stderr
    graph.write_text(json.dumps({**doc, "schema": "tasks/v0"}))
    r = _tasks(tree, "run", "pytest:a:3/3", "--graph", str(graph))
    assert r.returncode == 2 and "not a tasks/v1 graph" in r.stderr, r.stderr

    graph.write_text(text)
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "more tests")
    r = _tasks(tree, "run", "pytest:a:3/3", "--graph", str(graph))
    assert r.returncode == 2 and f"a graph of commit {doc['commit']}, not of this tree's HEAD" in r.stderr, r.stderr


def _files(tree: Path, n: int) -> None:
    for k in range(2, n + 1):
        (tree / "a" / "tests" / f"test_a{k}.py").write_text(f"def test_a{k}():\n    assert True\n")


def _split(doc: dict, suite: str) -> int:
    return doc["splits"][suite]["m"] if suite in doc["splits"] else 1


def test_executor_seconds_split_a_suite_in_proportionally_more_parts_than_the_listers_record(tree):
    """The lister's own record under-counts what a suite takes on the executor's worker slots: given the executor's seconds
    (3x the record here) the listing cuts it in 3x the parts, so each part fits `--split-s` on the executor."""
    _files(tree, 8)
    cache = tree.parent / "test-cache"
    cache.mkdir()
    (cache / "durations.json").write_text(json.dumps({"a": 1000}))
    assert _split(_graph(tree)[0], "a") == 2, "the tree's own record: ceil(1000 / 900)"
    costs = tree.parent / "costs.json"
    costs.write_text(json.dumps({"pytest:a": 3000}))
    doc, _ = _graph(tree, "--costs", str(costs))
    by = {t["id"]: t for t in doc["tasks"]}
    assert _split(doc, "a") == 4 and [f"pytest:a:{k}/4" for k in range(1, 5)] == [i for i in by if i.startswith("pytest:a:")]
    assert sum(by[f"pytest:a:{k}/4"]["cost_s"] for k in range(1, 5)) == pytest.approx(3000)
    costs.write_text(json.dumps({"pytest:a": 6000}))
    assert _split(_graph(tree, "--costs", str(costs))[0], "a") == 7


def test_circuit_checks_shard_count_is_sized_from_its_recorded_seconds_never_under_ci_tomls(tree):
    costs = tree.parent / "costs.json"

    def shards(*args: str) -> tuple[int, list[str]]:
        doc = _graph(tree, "--costs", str(costs), *args)[0]
        return doc["circuit_shards"], [t["id"] for t in doc["tasks"] if t["kind"] == "circuit-check"]

    costs.write_text(json.dumps({"circuit-check": 9000}))
    n, ids = shards()
    assert n == 10 and ids == [f"circuit-check:{k}/10" for k in range(1, 11)], "ceil(9000 / 900)"
    assert shards("--split-s", "3000")[0] == 4, "ceil(9000 / 3000) = 3, under ci.toml's 4"
    assert shards("--circuit-shards", "2")[0] == 2, "a count given wins"
    costs.write_text(json.dumps({"circuit-check": 100}))
    assert shards()[0] == 4


def test_costs_reads_the_executors_runs_and_folds_parts_back_to_whole_jobs(tree, tmp_path, monkeypatch):
    """`costs` prices each task the executor passed from the run its newest `pass` label names (the result's own steps, not a
    dep's build beside them; else the run's wall seconds), takes the queue's seconds over them, and sums a split suite's parts
    and circuit-check's slices back to the whole over the share they cover; `list --costs` then splits them anew."""
    import importlib.util

    import research.store.local as local
    from research.executor import keyed
    from research.store.model import Label

    spec = importlib.util.spec_from_file_location("tasks_costs_under_test", tree / "tools" / "verity" / "check" / "tasks.py")
    T = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(tree / "tools" / "verity" / "check"))
    spec.loader.exec_module(T)
    _files(tree, 6)
    doc = {"schema": "tasks/v1", "commit": "0" * 40, "kind": "after-landing",
           "splits": {"a": {"m": 2, "shares": [0.75, 0.25]}},
           "tasks": [{"id": "pytest:a:1/2", "key": "1" * 128}, {"id": "pytest:a:2/2", "key": "2" * 128},
                     {"id": "lean-suites:b", "key": "3" * 128}, {"id": "lean-audit:lean/P", "key": "4" * 128},
                     *({"id": f"circuit-check:{k}/4", "key": None} for k in range(1, 5))]}
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(doc))
    keys = {t["id"]: t["key"] for t in keyed(doc)["tasks"]}
    #: task id -> (verdict, its run's result steps or None for a run with no result, the run's wall seconds)
    ran = {"pytest:a:1/2": ("pass", [("pytest", 2400.0)], 2500.0), "pytest:a:2/2": ("pass", [("pytest", 600.0)], 700.0),
           "lean-suites:b": ("pass", [("lean-build", 90.0), ("lean-suites", 1200.0)], 1400.0),
           "lean-audit:lean/P": ("fail", [("lean-audit", 50.0)], 60.0),
           "circuit-check:1/4": ("pass", None, 3000.0), "circuit-check:2/4": ("pass", [("circuit-check", 2000.0)], 2100.0)}
    runs = {f"r20261101-000000-{i:04x}": (p, *v) for i, (p, v) in enumerate(ran.items())}
    by_key = {keys[p]: (rid, v) for rid, (p, v, _, _) in runs.items()}

    class Store:
        remote = None

        def labels(self, target, key=None):
            rid, v = by_key.get(target.removeprefix("task:"), (None, None))
            return [Label(target, "verdict", v, "executor", "2026-11-01T00:00:00Z", rid)] if rid else []

        def get_attempt(self, rid):
            _p, _v, steps, wall = runs[rid]
            return {"outputs": {"run_record": f"art:{rid}" if steps else None}, "execution": {"wall_s": wall}}

        def listing(self, art):
            return [{"path": "job.json"}, {"path": T.RESULT}]

        def fetch(self, art, to, paths=None):
            _p, _v, steps, _w = runs[art.removeprefix("art:")]
            (to / T.RESULT).parent.mkdir(parents=True, exist_ok=True)
            (to / T.RESULT).write_text(json.dumps({"seconds": sum(s for _, s in steps) + 5,
                                                   "steps": [{"step": n, "seconds": s} for n, s in steps]}))
            return to

    monkeypatch.setattr(local, "LocalStore", Store)
    got, notes = T.costs([str(graph)])
    assert got == {"pytest:a": 3000.0, "lean-suites:b": 1200.0, "circuit-check": 10000.0}, got
    assert "5 of 8 tasks priced (4 by their result's steps, 1 by their run's wall seconds, 0 given)" in notes[0], notes
    assert "none for lean-audit:lean/P" in notes[0], "a failed run prices nothing"

    got, notes = T.costs([str(graph)], {"pytest:a:2/2": 1000.0, "lean-audit:lean/P": 70.0, "pytest:elsewhere": 5.0})
    assert (got["pytest:a"], got["lean-audit:lean/P"], "pytest:elsewhere" in got) == (3400.0, 70.0, False), got
    assert "2 given" in notes[0]

    del ran["pytest:a:2/2"]
    runs.clear()
    runs.update({f"r20261101-000000-{i:04x}": (p, *v) for i, (p, v) in enumerate(ran.items())})
    by_key.clear()
    by_key.update({keys[p]: (rid, v) for rid, (p, v, _, _) in runs.items()})
    got, _ = T.costs([str(graph)])
    assert got["pytest:a"] == 3200.0, "part 1 alone, three quarters of the suite: 2400 / 0.75"
    assert T.costs([str(graph), str(graph)], {"pytest:a:1/2": 3000.0})[0]["pytest:a"] == 4000.0

    out = tmp_path / "costs.json"
    assert T.main(["costs", "--graph", str(graph), "--out", str(out)]) == 0
    listing = _graph(tree, "--costs", str(out))[0]
    assert (_split(listing, "a"), listing["circuit_shards"]) == (4, 12), "ceil(3200 / 900) parts, ceil(10000 / 900) shards"


def test_a_graph_given_by_art_id_is_fetched_from_the_store_and_must_be_a_tasks_graph(tree, tmp_path, monkeypatch):
    import importlib.util

    import research.store.local as local

    spec = importlib.util.spec_from_file_location("tasks_graph_under_test", tree / "tools" / "verity" / "check" / "tasks.py")
    T = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(tree / "tools" / "verity" / "check"))
    spec.loader.exec_module(T)
    head = _git(tree, "rev-parse", "HEAD").strip()
    doc = {"schema": "tasks/v1", "commit": head, "kind": "after-landing", "tasks": [{"id": "pytest:a"}]}
    (tmp_path / "g.json").write_text(json.dumps(doc))
    art, kinds = "art:" + "0" * 64, {"art:" + "0" * 64: "tasks-graph/v1", "art:" + "1" * 64: "fixture/v1"}

    class Store:
        def get_manifest(self, ref):
            if ref not in kinds:
                raise local.NotFound(f"{ref}: no manifest")
            return type("M", (), {"kind": kinds[ref]})()

        def fetch(self, ref):
            return tmp_path / "g.json"

    monkeypatch.setattr(local, "LocalStore", Store)
    assert T.listed(art) == doc
    with pytest.raises(ValueError, match="a fixture/v1 artifact, not a tasks-graph/v1"):
        T.listed("art:" + "1" * 64)
    with pytest.raises(ValueError, match="no manifest"):
        T.listed("art:" + "2" * 64)
