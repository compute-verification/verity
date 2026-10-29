"""`check.py --deferred STEP,...`, the post-landing run `research queue drive` launches on `main` (`train.sh --deferred`): only
those ci.toml jobs and what they need, inside a recorded run that is no check, so no gate reads it."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

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
    """A post-landing run recorded as `--tool check` would be a check of `main`'s tip that `research merge` reads.  It carries
    its base's commit, above which ratchet-gate judges every merge."""
    real, launched, sent = subprocess.run, [], {}

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            sent.update({Path(cmd[i + 1]).name: Path(cmd[i + 1]).read_text() for i, x in enumerate(cmd) if x == "--send"})
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
    assert sent[CHECK.BASE_FILE] == f"{CHECK._rev('HEAD')}\n"


def _pack(tmp_path: Path, monkeypatch) -> Path:
    """A pack of one circuit-check target's pass, one suite's and one Lean package's, as a recorded check's pack holds them."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "earned" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "earned" / "tests"))
    for p in (CHECK.cache_dir() / "circuit-check" / "v1" / "d" / "t.json", CHECK.cache_dir() / "lean-audit" / "k.json",
              CHECK.VERDICTS.roots()["tests"] / "suite" / "k.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    CHECK.VERDICTS.export(pack := tmp_path / "gathered.tar.gz")
    return pack


@pytest.mark.parametrize("circuit_cache", [False, True])
def test_the_after_landing_run_reuses_circuit_checks_passes_alone(monkeypatch, tmp_path, circuit_cache):
    """Drive's after-landing run (`train.sh --deferred STEPS --no-cache --circuit-cache`): its suites, Lean and steps reuse no
    pass, and circuit-check its per-target ones, imported from the gathered pack (its entries alone) and packed again as the run
    stores them; `--no-cache` alone, the nightly `audit-main`, imports and packs nothing."""
    pack = _pack(tmp_path, monkeypatch)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "here" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "here" / "tests"))
    run_dir, got = tmp_path / "run", {}
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261027-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)

    def run(out, n, cache, keep_going, only, packs, reused, regenerate, circuit_cache=None):
        got.update(cache=cache, circuit_cache=circuit_cache, packs=packs is not None, reused=reused is not None)
        return True, []

    monkeypatch.setattr(CHECK, "run", run)
    argv = ["--deferred", "pytest,circuit-check", "--keep-going", "--no-cache", *(["--circuit-cache"] if circuit_cache else []),
            "--no-regenerate", "--verdicts-in", str(pack)]
    assert CHECK.main(argv) == 0
    assert got == {"cache": False, "circuit_cache": circuit_cache, "packs": circuit_cache, "reused": circuit_cache}
    assert sorted(CHECK.VERDICTS.entries()) == (["check/circuit-check/v1/d/t.json"] if circuit_cache else [])
    doc = json.loads((run_dir / "result.json").read_text())
    assert ("only" in (doc.get("verdicts") or {}).get("in", {})) is circuit_cache
