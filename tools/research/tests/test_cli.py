"""`research run/status/inspect` over the promoted harness, in a temporary runs root."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from research import runs as R
from research.cli import main


def test_run_records_and_inspects(tmp_path, capsys):
    rc = main(["run", "--runs-dir", str(tmp_path), "--project", "t", "--stage", "smoke", "--no-sampler", "--",
               sys.executable, "-c", "print('hello'); import sys; sys.exit(3)"])
    assert rc == 3, "research run exits with the workload's status"
    runs = R.list_runs(tmp_path)
    assert len(runs) == 1
    s = runs[0]
    assert s["project"] == "t" and s["stage"] == "smoke" and s["rc"] == 3
    assert s["state"] in ("done", "failed")
    assert (tmp_path / s["id"] / "stdout.log").read_text().strip() == "hello"
    job = json.loads((tmp_path / s["id"] / "job.json").read_text())
    assert job["schema"].startswith("telemetry/job/")
    assert job["ownership"]["job"] == s["id"]

    assert main(["inspect", s["id"][:12], "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert s["id"] in out and "rc=3" in out

    assert main(["status", "--runs-dir", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"][0]["id"] == s["id"]


RESULT_OF_ENV = ("import os, json; json.dump({'foo': os.environ.get('RSEC_FOO'), 'tok': os.environ.get('RSEC_API_TOKEN'), 'cwd': os.getcwd()}, "
                 "open('result.json', 'w'))")


def test_local_env_is_applied_to_the_workload_recorded_and_scoped(tmp_path, monkeypatch):
    src = tmp_path / "checkout"
    src.mkdir()
    monkeypatch.chdir(src)
    monkeypatch.delenv("RSEC_FOO", raising=False)
    monkeypatch.setenv("RSEC_API_TOKEN", "callers")
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--env", "RSEC_FOO=bar", "--env", "RSEC_API_TOKEN=s3cret",
               "--", sys.executable, "-c", RESULT_OF_ENV])
    assert rc == 0
    run_dir = next((tmp_path / "runs").iterdir())
    seen = json.loads((run_dir / "result.json").read_text())
    assert seen["foo"] == "bar" and seen["tok"] == "s3cret", "the workload sees --env"
    assert Path(seen["cwd"]).resolve() == run_dir.resolve() and list(src.iterdir()) == [], "default cwd is the run dir: the checkout stays clean"
    assert "RSEC_FOO" not in os.environ and os.environ["RSEC_API_TOKEN"] == "callers", "--env is scoped to the workload"
    params = json.loads((run_dir / "job.json").read_text())["store"]["params"]
    assert params["env"] == {"RSEC_FOO": "bar", "RSEC_API_TOKEN": "<redacted>"}
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--env", "NOEQUALS", "--", "true"]) == 2


def test_cwd_overrides_the_run_dir_default(tmp_path, monkeypatch):
    src = tmp_path / "checkout"
    src.mkdir()
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--no-sampler", "--cwd", str(src), "--", sys.executable, "-c", RESULT_OF_ENV]) == 0
    assert Path(json.loads((src / "result.json").read_text())["cwd"]).resolve() == src.resolve()


def test_machine_side_records_the_requests_env(tmp_path):
    from research.cli import _request_env

    req = tmp_path / "request.json"
    req.write_text(json.dumps({"schema": "x", "env": {"A": "1", "B_TOKEN": "t"}}))
    assert _request_env(json.dumps({"machine": "m", "request": str(req)})) == {"A": "1", "B_TOKEN": "t"}
    assert _request_env(json.dumps({"machine": "m"})) == {} and _request_env(None) == {}
    assert _request_env(json.dumps({"request": str(tmp_path / "absent.json")})) == {}


def test_run_requires_separator_and_command(tmp_path):
    assert main(["run", "--runs-dir", str(tmp_path), "echo", "x"]) == 2
    assert main(["run", "--runs-dir", str(tmp_path), "--"]) == 2
    assert not list(tmp_path.iterdir())


def test_remote_requires_project_and_listed_machine(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    assert main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--", "true"]) == 2
    assert "--project is required" in capsys.readouterr().err
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--project", "verity", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "not in" in err and "refused" in err
    (tmp_path / "machines.toml").write_text('[machines.pod-x]\nprovider = "runpod"\npod_id = "p1"\nproject = "veritor"\n')
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--project", "verity", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "belongs to project(s) ['veritor']" in err, "a foreign pod is refused before any contact"
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 2 and all((r / "remote.json").exists() for r in runs), "refused launches are still recorded runs"


def test_resolve_prefix_and_ambiguity(tmp_path):
    for name in ("r20260921-100000-aaaa", "r20260921-100001-bbbb"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "job.json").write_text("{}")
    assert R.resolve(tmp_path, "r20260921-100000").name.endswith("aaaa")
    with pytest.raises(LookupError):
        R.resolve(tmp_path, "r20260921")
    with pytest.raises(FileNotFoundError):
        R.resolve(tmp_path, "nope")


def test_no_packages_import():
    """`research` must not depend on the Verity library, frontends, backends, or integrations (README: it treats
    the workload as opaque). Covers `verity` and any `verity_*` extension distribution."""
    import pathlib
    import re

    src = pathlib.Path(R.__file__).parent
    offenders = []
    for p in src.rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(from|import)\s+(verity\w*|packages|veritor|backends|frontends|integrations)\b", line):
                offenders.append(f"{p.relative_to(src)}: {line.strip()}")
    assert not offenders, offenders
