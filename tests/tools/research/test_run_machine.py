"""A run launched onto a named machine is handed the machine's name (RESEARCH_MACHINE) and its record's `[properties]`, as
inputs/NAME.toml, and nothing else of the record; a local run is handed neither.  The machine is a directory reached through
`sh -c`, and its `launch-request` is this package's own."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

from research import cli, remote
from research import runs as R
from waits import wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
A = "r20000101-000000-aaaa"
SSH = ["sh", "-c"]
PROPS = {"cards": {"value": 1, "read": "nvidia-smi -L | wc -l"}, "profiler": {"value": "unread"}}
SHOW = ("import os, pathlib; d = pathlib.Path(os.environ['RESEARCH_RUN_DIR']) / 'inputs'; "
        "print(os.environ.get('RESEARCH_MACHINE'), sorted(p.name for p in d.iterdir()) if d.is_dir() else [])")


@pytest.fixture
def record(tmp_path, monkeypatch):
    """Machine `m`'s record, an address and a key among its fields; the machine's root is tmp_path/m."""
    rec = {"host": "203.0.113.7", "port": 2222, "user": "research", "pod_id": "dummy-pod", "key": "/dummy/id_ed25519",
           "properties": PROPS}
    root = tmp_path / "m"
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": str(root), "python": PY, **rec})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "m", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: SSH)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: SRC)
    monkeypatch.setattr(remote, "launch_pause", lambda attempt: 0)
    return rec


def _launch(tmp_path: Path, sends: list[Path] = ()) -> Path:
    remote.launch(run_id=A, local_dir=tmp_path / "here" / A, machine_name="m", project="p", command=[PY, "-c", SHOW],
                  harness_args=["--stage", "t", "--no-sampler"], source=None, sends=list(sends), cwd=None, env={}, exclusive=False)
    run = tmp_path / "m" / "runs" / A
    wait_runner(run)
    assert R.summarize(run)["rc"] == 0
    return run


def test_a_launched_run_gets_the_machines_name_and_a_copy_of_its_properties_alone(tmp_path, record):
    """Rule: the run's copy holds the name and `[properties]`; weakening: the record whole, its address and key with it."""
    run = _launch(tmp_path)
    assert (run / "stdout.log").read_text() == "m ['m.toml']\n"
    text = (run / "inputs" / "m.toml").read_text()
    assert tomllib.loads(text) == {"machine": "m", "properties": PROPS}
    assert [v for v in ("203.0.113.7", "2222", "dummy-pod", "/dummy/id_ed25519") if v in text] == []
    assert json.loads((run / "job.json").read_text())["store"]["params"]["env"]["RESEARCH_MACHINE"] == "m"


def test_a_machine_without_properties_gets_its_name_and_no_file(tmp_path, record):
    del record["properties"]
    assert (_launch(tmp_path) / "stdout.log").read_text() == "m []\n"


def test_a_send_under_the_copys_name_is_refused(tmp_path, record):
    mine = tmp_path / "m.toml"
    mine.write_text("[properties]\n")
    with pytest.raises(remote.Refused, match=r"--send m\.toml: the launch puts m's \[properties\] in the run's inputs/"):
        remote.launch(run_id=A, local_dir=tmp_path / "here" / A, machine_name="m", project="p", command=[PY, "-c", "pass"],
                      harness_args=["--stage", "t", "--no-sampler"], source=None, sends=[mine], cwd=None, env={}, exclusive=False)


def test_a_local_run_gets_neither(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_MACHINE", raising=False)
    runs = tmp_path / "runs"
    assert cli.main(["run", "--runs-dir", str(runs), "--id", A, "--no-sampler", "--", PY, "-c", SHOW]) == 0
    assert (runs / A / "stdout.log").read_text() == "None []\n"
    assert "RESEARCH_MACHINE" not in (json.loads((runs / A / "job.json").read_text())["store"]["params"].get("env") or {})
