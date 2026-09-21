"""`research run/status/inspect` over the promoted harness, in a temporary runs root."""
from __future__ import annotations

import json
import sys

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


def test_run_requires_separator_and_command(tmp_path):
    assert main(["run", "--runs-dir", str(tmp_path), "echo", "x"]) == 2
    assert main(["run", "--runs-dir", str(tmp_path), "--"]) == 2
    assert not list(tmp_path.iterdir())


def test_remote_not_implemented_is_explicit(tmp_path, capsys):
    assert main(["run", "--runs-dir", str(tmp_path), "--on", "pod-x", "--", "true"]) == 2
    assert "not implemented" in capsys.readouterr().err
    assert not list(tmp_path.iterdir()), "no run directory is created for a refused run"


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
    """`research` must not depend on Verity's packages/; it runs commands and records outputs."""
    import pathlib
    import re

    src = pathlib.Path(R.__file__).parent
    offenders = []
    for p in src.rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(from|import)\s+(verity|packages|veritor|verity_ir|verity_vllm)\b", line):
                offenders.append(f"{p.relative_to(src)}: {line.strip()}")
    assert not offenders, offenders
