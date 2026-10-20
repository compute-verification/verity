"""`check.py --deferred STEP,...`, the post-landing run `research queue drive` launches on `main` (`train.sh --deferred`): only
those ci.toml jobs and what they need, inside a recorded run that is no check, so no gate reads it."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CHECK = _module("repo_check_for_deferred", ROOT / "tools" / "verity" / "check" / "check.py")


def _main(monkeypatch, tmp_path, argv):
    got = []
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261018-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)
    monkeypatch.setattr(CHECK, "run", lambda out, n, cache, keep_going, only, *packs: got.append(only) or (True, []))
    return CHECK.main(argv), got[0] if got else None


def test_a_deferred_run_runs_only_its_steps_and_what_they_need_inside_a_recorded_run(monkeypatch, tmp_path):
    rc, only = _main(monkeypatch, tmp_path, ["--deferred", "pytest,flock-rust-tests", "--keep-going", "--no-cache"])
    assert rc == 0 and {p.partition(":")[0] for p in only} == {"pytest", "flock-rust-tests", "flock-circuit-build"}
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["kind"] == CHECK.DEFERRED_KIND == "check-deferred/v1" and "shard" not in doc
    assert doc["deferred"]["deferred"] == ["pytest", "flock-rust-tests"]


def test_a_deferred_run_is_refused_with_a_part_or_a_step_ci_toml_doesnt_declare(monkeypatch, tmp_path, capsys):
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest", "tests"]) == (2, None)
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest", "--shard", "1/3"]) == (2, None)
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest,nope"]) == (2, None)
    assert "no job nope" in capsys.readouterr().err


def test_record_of_a_deferred_run_launches_it_without_the_check_tool_and_without_lean(monkeypatch):
    """A post-landing run recorded as `--tool check` would be a check of `main`'s tip that `research merge` reads."""
    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: (_ for _ in ()).throw(AssertionError("Lean dependencies sent")))
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: ["slot", *(["--priority"] if priority else [])])
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--priority", "--keep-going", "--deferred", "pytest,circuit-check"]) == 0
    [cmd] = launched
    head, tail = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert "--tool" not in head and head[head.index("--send") + 1].endswith(CHECK.SHARD_PARTS)
    assert tail[:2] == ["slot", "--priority"] and tail[-3:] == ["--keep-going", "--deferred", "pytest,circuit-check"]
