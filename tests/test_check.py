"""`tools/check/check.py`: its steps in order, the Lean hook before and after PR #85, and the result a recorded run publishes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_check", ROOT / "tools" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


def test_the_steps_are_pytest_then_circuit_check_then_the_lean_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    names = [(n, argv if isinstance(argv, str) else argv[1:4]) for n, argv, _cwd in CHECK.steps(tmp_path)]
    assert names[0] == ("pytest", ["-m", "pytest", "-q"]) and names[1] == ("circuit-check", ["-m", "circuit_check", "--all"])
    assert names[2] == ("lean", "skipped: PR #85 (backends/flock/verifier) is not on this tree")
    assert "-m" in CHECK.steps(tmp_path)[0][1] and "not circuit_suite" in CHECK.steps(tmp_path)[0][1]


def test_the_lean_step_is_required_once_the_verifier_is_on_the_tree(tmp_path, monkeypatch):
    (tmp_path / "verifier" / "lean").mkdir(parents=True)
    (tmp_path / "verifier" / "lean" / "lakefile.toml").write_text("")
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    monkeypatch.delenv("FLOCK_UPSTREAM", raising=False)
    (build, lake, _), (agree, why, _) = CHECK.lean_steps(tmp_path)
    assert (build, lake, agree) == ("lean-build", ["lake", "build"], "lean-agreement") and why.startswith("FLOCK_UPSTREAM is not set")
    monkeypatch.setenv("FLOCK_UPSTREAM", "abc=/bin/up1,def=/bin/up2")
    argv = CHECK.lean_steps(tmp_path)[1][1]
    assert argv[1].endswith("verifier/ci.py") and argv[argv.index("--upstream") + 1] == "abc=/bin/up1" and argv.count("--upstream") == 2


def test_a_recorded_run_writes_a_valid_result(tmp_path, monkeypatch):
    from research import result as RES

    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260927-010000-abcd")
    (tmp_path / "circuit-check.json").write_text("{}")
    done = [{"step": "pytest", "status": "passed", "rc": 0, "seconds": 1.0}, {"step": "circuit-check", "status": "failed", "rc": 1, "seconds": 2.0}]
    CHECK.write_result(tmp_path, False, done)
    r = RES.check(tmp_path, "r20260927-010000-abcd")
    assert r.valid, r.errors
    assert r.validation_status == "failed" and r.doc["kind"] == "check/v1" and r.doc["artifacts"] == ["circuit-check.json"]


def test_the_steps_see_the_trees_packages_only(monkeypatch):
    """On a machine the runner's PYTHONPATH starts with its own shipped `research`; the steps must import the tree's."""
    monkeypatch.setenv("PYTHONPATH", f"/opt/research-tool:{ROOT / 'tools/research/src'}:/elsewhere")
    assert CHECK.env()["PYTHONPATH"] == str(ROOT / "tools/research/src")
    monkeypatch.setenv("PYTHONPATH", "/opt/research-tool")
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "abc")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/workspace/research/cache/pycache")
    env = CHECK.env()
    assert "PYTHONPATH" not in env and "RESEARCH_SOURCE_SHA" not in env and "PYTHONPYCACHEPREFIX" not in env


def test_the_groups_split_the_cores_and_run_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    (pt,), (cc,), (lean,) = CHECK.groups(tmp_path, 16)
    assert pt[1][pt[1].index("-n") + 1] == "8" and cc[1][cc[1].index("--jobs") + 1] == "8" and "--cache" in cc[1]
    assert lean[0] == "lean" and "--cache" not in CHECK.groups(tmp_path, 16, cache=False)[1][0][1]


def test_the_pytest_cache_key_is_the_tree_the_environment_and_the_arguments(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a")
    d = CHECK.tree_digest(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert CHECK.tree_digest(tmp_path) == d                                        # bytecode is not the tree
    (tmp_path / "a.txt").write_text("b")
    assert CHECK.tree_digest(tmp_path) != d
    argv = ["py", "-m", "pytest", "-q", "-n", "8"]
    e = CHECK.env()
    k = CHECK.pytest_key(argv, e)
    assert CHECK.pytest_key(["py", "-m", "pytest", "-q", "-n", "2"], e) == k       # the worker count computes nothing
    assert CHECK.pytest_key(argv, {**e, "PWD": "/elsewhere"}) == k
    assert CHECK.pytest_key(argv, {**e, "VERITY_X": "1"}) != k and CHECK.pytest_key(argv + ["-k", "x"], e) != k
