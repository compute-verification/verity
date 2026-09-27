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


def test_the_lean_steps_once_the_verifier_is_on_the_tree(tmp_path, monkeypatch):
    """`lake build` and the unit-cut conformance are required; the upstream agreement runs when the run was sent its bundle."""
    (tmp_path / "verifier" / "lean").mkdir(parents=True)
    (tmp_path / "verifier" / "lean" / "lakefile.toml").write_text("")
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    (build, lake, _), (unit, cut, _), (agree, why, _) = CHECK.lean_steps(tmp_path)
    assert (build, unit, agree) == ("lean-build", "lean-unit-cut", "lean-agreement") and lake[-1].endswith("lake build")
    assert cut[1].endswith("verifier/unit_cut_agree.py") and cut[cut.index("--lean") + 1].endswith(".lake/build/bin/flock-verify")
    assert why.startswith("skipped: no upstream bundle sent")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "ci-bundle.tar.gz").write_bytes(b"")
    argv = CHECK.lean_steps(tmp_path)[2][1]
    assert argv[:2] == ["bash", "-c"] and "ci-pod.sh" in argv[2] and "ci-bundle.tar.gz" in argv[2]


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
