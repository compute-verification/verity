"""A suite in parts by test file (`suites.py SUITE:K/M`, `--files K/M`): the partition, a part's key, a run of a part, the join of
a suite's parts into its verdict (`suites.combine_parts`), and the planner's splitting of a long suite (`shards.py`)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                       # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


SH = _module("shards_for_suite_parts", HERE / "shards.py")


@pytest.fixture
def S(monkeypatch):
    mod = _module("suites_for_suite_parts", HERE / "suites.py")
    yield mod
    sys.modules.pop("suites_for_suite_parts", None)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


#: the synthetic suite's test files: test_d fails until fixed; none reaches another
TESTS = {"test_a.py": "def test_a():\n    pass\n\ndef test_a2():\n    pass\n",
         "test_b.py": "def test_b():\n    pass\n",
         "test_c.py": "from helpers import VALUE\n\ndef test_c():\n    assert VALUE == 1\n",
         "sub/test_d.py": "def test_d():\n    assert False\n",
         "sub/test_e.py": "def test_e():\n    pass\n"}
TIMES = {"pkg/tests/test_a.py": 10.0, "pkg/tests/test_b.py": 9.0, "pkg/tests/test_c.py": 1.0, "pkg/tests/sub/test_d.py": 1.0}


def _workspace(root: Path) -> None:
    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copy(HERE / "settings.py", root / "tools" / "verity" / "check" / "settings.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (root / "pkg" / "tests" / "sub").mkdir(parents=True)
    (root / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (root / "pkg" / "tests" / "conftest.py").write_text("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n")
    (root / "pkg" / "tests" / "helpers.py").write_text("VALUE = 1\n")
    for name, text in TESTS.items():
        (root / "pkg" / "tests" / name).write_text(text)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "tree")


def test_the_partition_packs_the_longest_first_and_holds_every_file_once_whatever_their_order():
    S = _module("suites_partition", HERE / "suites.py")
    files = ["a", "b", "c", "d", "e"]
    times = {"a": 10.0, "b": 9.0, "c": 1.0, "d": 1.0}             # e has none: the median of the others' (5 s)
    parts = S.partition(files, times, 2)
    assert parts == [["a", "c", "d"], ["b", "e"]]
    assert S.partition(list(reversed(files)), dict(reversed(times.items())), 2) == parts, "a function of its inputs alone"
    for m in (1, 2, 3, 5, 7):
        got = S.partition(files, times, m)
        flat = [f for p in got for f in p]
        assert len(got) == m and sorted(flat) == files and len(set(flat)) == len(flat), m
    assert S.partition(files, {"a": 10.04}, 2) == S.partition(files, {"a": 10.01}, 2), "jitter below two figures moves nothing"
    assert S.partition_digest(parts) != S.partition_digest(S.partition(files, {**times, "e": 20.0}, 2))
    with pytest.raises(ValueError):
        S.partition(files, times, 0)


def test_a_parts_key_moves_with_its_files_and_the_shared_inputs_and_not_with_another_parts(tmp_path, S, monkeypatch):
    root = tmp_path / "tree"
    _workspace(root)
    monkeypatch.setattr(S, "ROOT", root)
    (suite,) = [s for s in S.discover() if s.name == "pkg"]

    def keyed(k: int) -> tuple[str, tuple[str, ...]]:
        tree, pkgs = S.worktree(), S.members()
        paths = S.inputs_of(suite, pkgs)
        cut = S.dropped(suite, paths)
        ids = S.input_ids(tree, sorted(set(paths) | set(S.RUNNER)), S.markdown(suite), cut)
        part = S.part_of(suite, (k,), 2, tree, cut, TIMES)
        mods, own, keep = S.part_context(suite, part, tree, cut)
        return S.part_key(suite, part, keep, mods, ids, own, {}, [], {}), part.files

    (k1, f1), (k2, f2) = keyed(1), keyed(2)
    assert f1 == ("pkg/tests/sub/test_d.py", "pkg/tests/test_a.py", "pkg/tests/test_c.py")
    assert f2 == ("pkg/tests/sub/test_e.py", "pkg/tests/test_b.py") and k1 != k2
    tests = root / "pkg" / "tests"
    (tests / "test_b.py").write_text("def test_b():\n    assert 1\n")               # another part's file
    assert keyed(1)[0] == k1 and keyed(2)[0] != k2
    (tests / "test_a.py").write_text(TESTS["test_a.py"] + "\n")                     # its own
    assert keyed(1)[0] != k1
    k1 = keyed(1)[0]
    (tests / "conftest.py").write_text((tests / "conftest.py").read_text() + "\n")  # shared by every part
    assert keyed(1)[0] != k1
    k1 = keyed(1)[0]
    (tests / "helpers.py").write_text("VALUE = 1  # a helper is in every part's key\n")
    assert keyed(1)[0] != k1


def _run(root: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(RESEARCH_STORE=str(tmp_path / "pod-store"), RESEARCH_STORE_CONFIG=str(tmp_path / "absent.toml"),
               RESEARCH_FIXTURES_URL="", VERITY_TEST_CACHE=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1",
               CHECK_SLOT_MEM_GB="1000")
    return subprocess.run([sys.executable, str(root / "tools" / "verity" / "check" / "suites.py"), "--jobs", "1",
                           "--logs", str(tmp_path / "logs"), *args], cwd=root, env=env, capture_output=True, text=True)


def _ran(rows: list[dict]) -> int:
    return sum((r.get("counts") or {}).get(k, 0) for r in rows for k in ("passed", "failed"))


def test_the_parts_run_their_files_once_and_join_to_the_whole_suites_verdict(tmp_path, S):
    root = tmp_path / "tree"
    _workspace(root)
    timings = tmp_path / "timings.json"
    timings.write_text(json.dumps({"pkg": TIMES}))
    rows = {}
    for k in (1, 2):
        out = tmp_path / f"part{k}.json"
        p = _run(root, tmp_path, f"pkg:{k}/2", "--timings", str(timings), "--json", str(out))
        rows[k] = json.loads(out.read_text())
        assert p.returncode == (1 if k == 1 else 0), p.stdout + p.stderr
    whole = tmp_path / "whole.json"
    assert _run(root, tmp_path, "pkg", "--json", str(whole)).returncode == 1
    (w,) = json.loads(whole.read_text())
    assert _ran(rows[1]) + _ran(rows[2]) == _ran([w]) == 6, "every test runs in exactly one part"
    assert rows[1][0]["failed_tests"] == ["tests/sub/test_d.py::test_d"] and rows[1][0]["part"]["ks"] == [1]
    (joined,) = S.combine_parts(rows[1] + rows[2])
    assert (joined["status"], joined["failed_tests"]) == ("failed", w["failed_tests"]), "a failing file fails its suite"

    (root / "pkg" / "tests" / "sub" / "test_d.py").write_text("def test_d():\n    pass\n")
    for k in (1, 2):
        out = tmp_path / f"part{k}.json"
        p = _run(root, tmp_path, "pkg", "--files", f"{k}/2", "--timings", str(timings), "--json", str(out))
        assert p.returncode == 0, p.stdout + p.stderr
        rows[k] = json.loads(out.read_text())
    assert rows[2][0]["status"] == "cached", "part 2's files and shared inputs are as they were: its pass holds"
    (joined,) = S.combine_parts(rows[1] + rows[2])
    assert joined["status"] == "passed" and joined["why"] == []
    assert _run(root, tmp_path, "pkg").returncode == 0

    p = _run(root, tmp_path, "pkg:1,2/2", "--timings", str(timings), "--fresh")
    assert p.returncode == 0 and "6 passed" in p.stdout, "both parts in one run are the suite"
    p = _run(root, tmp_path, "pkg:1/2@0000000000", "--timings", str(timings))
    assert p.returncode == 2 and "their timing records differ" in p.stderr
    p = _run(root, tmp_path, "pkg:1/2", "--list")
    assert p.returncode == 0 and "part 1/2 of partition" in p.stdout, p.stdout


def test_the_join_fails_a_suite_whose_parts_are_missing_twice_or_of_two_partitions(S):
    def row(k: int, m: int = 3, digest: str = "d1", status: str = "passed") -> dict:
        return {"suite": "pkg", "name": "pkg", "status": status, "seconds": 1.0, "failed_tests": [],
                "part": {"ks": [k], "m": m, "digest": digest}}

    whole = {"suite": "other", "name": "other", "status": "passed"}
    out = {r["name"]: r for r in S.combine_parts([whole, row(1), row(2), row(3, status="cached")])}
    assert out["other"] == whole and out["pkg"]["status"] == "passed"
    assert "missing: [3]" in S.combine_parts([row(1), row(2)])[0]["why"][0]
    assert "twice: [2]" in S.combine_parts([row(1), row(2), row(2), row(3)])[0]["why"][0]
    assert "different partitions" in S.combine_parts([row(1), row(2, digest="d2"), row(3)])[0]["why"][0]
    assert S.combine_parts([row(1), row(2), row(3, status="failed")])[0]["status"] == "failed"


def test_the_planner_splits_a_suite_past_the_threshold_into_parts_priced_by_their_share():
    assert SH.split_m(None, 10) == SH.split_m(900, 10) == SH.split_m(5000, 1) == 1
    assert SH.split_m(901, 10) == 2 and SH.split_m(6608, 789) == 8 and SH.split_m(10875, 5) == 5
    assert SH.suite_part("pytest:verity/core/protocols/pouw:2/3") == ("pytest", "verity/core/protocols/pouw", "2/3")
    assert SH.suite_part("pytest:a") == ("pytest", "a", None)
    names = SH.parts(["pytest", "lean-suites", "lean-build"], ["a", "b", "f"], ["f"], True, {"a": 3, "f": 2})
    assert names == ["pytest:a:1/3", "pytest:a:2/3", "pytest:a:3/3", "pytest:b", "lean-suites:f:1/2", "lean-suites:f:2/2",
                     "lean-build"]
    got = SH.priced(names, {"a": 3000.0, "b": 10.0}, {}, {}, {"pytest:a:1/3": 0.5, "pytest:a:2/3": 0.3, "pytest:a:3/3": 0.2})
    assert got == {"pytest:a:1/3": 1500.0, "pytest:a:2/3": 900.0, "pytest:a:3/3": 600.0, "pytest:b": 10.0}
    ts = SH.tasks(names, got)
    split = {"a": {"m": 3, "digest": "d1"}}
    doc = SH.plan(ts, {}, 4, splits=split)
    assert doc["splits"] == split and SH.check(doc, doc["n"], names) is None
    assert doc["digest"] != SH.plan(ts, {}, 4, splits={"a": {"m": 3, "digest": "d2"}})["digest"], \
        "a plan's digest names its partitions"
