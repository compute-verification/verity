"""The machine-side half of --on, exercised locally: claim by run id, duplicate delivery, exclusive lock, refusal record."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from research import remote
from research import runs as R

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])   # .../src, so `-m research` resolves in the child


def _request(tmp_path, run_id, command, *, exclusive=False, env=None, harness=()):
    root = tmp_path / "root"
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": env or {}, "harness_args": ["--stage", "t", "--no-sampler", *harness], "command": command, "exclusive": exclusive,
           "cost": {"provider": "runpod", "pod_id": "p1", "rate_usd_per_hour": 0.06, "rate_observed_utc": "t0", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p1"}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p, root / "runs"


def _wait_terminal(run_dir, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = R.summarize(run_dir)
        if s["state"] in R.TERMINAL:
            return s
        time.sleep(0.1)
    pytest.fail(f"run did not finish: {R.summarize(run_dir)}")


def test_claim_starts_detached_runner_and_redelivery_is_idempotent(tmp_path):
    req, runs_root = _request(tmp_path, "r1", [PY, "-c", "import os, time; time.sleep(1.5); print(os.environ['RESEARCH_RUN_ID'])"])
    out = remote.launch_request(req)
    assert out["claimed"] and out["started"] and out["runner_pid"]
    again = remote.launch_request(req)                          # the ssh session "dropped" and the launcher re-delivered
    assert again["claimed"] is False and again["existing"] and again["runner_pid"] == out["runner_pid"]
    s = _wait_terminal(runs_root / "r1")
    assert s["state"] == "done" and s["rc"] == 0
    assert (runs_root / "r1" / "stdout.log").read_text().strip() == "r1"
    job = json.loads((runs_root / "r1" / "job.json").read_text())
    assert job["run"]["remote"]["pod_id"] == "p1" and job["cost"]["rate_usd_per_hour"] == 0.06 and job["cost"]["shared"] is False
    assert s["cost"]["usd"] is not None and s["cost"]["usd"] < 0.001
    launch = json.loads((runs_root / "r1" / "launch.json").read_text())
    assert launch["argv"][launch["argv"].index("--") + 1] == PY, "argv travels as a list; no shell string"


def test_env_and_spaces_reach_the_workload(tmp_path):
    req, runs_root = _request(tmp_path, "r2", [PY, "-c", "import os, sys; print(os.environ['FOO'], '|', sys.argv[1])", "two  words"], env={"FOO": "bar baz"})
    assert remote.launch_request(req)["started"]
    s = _wait_terminal(runs_root / "r2")
    assert (runs_root / "r2" / "stdout.log").read_text() == "bar baz | two  words\n"


def test_exclusive_refuses_a_live_holder_and_reclaims_a_dead_one(tmp_path):
    req1, runs_root = _request(tmp_path, "hold", [PY, "-c", "import time; time.sleep(4)"], exclusive=True)
    assert remote.launch_request(req1)["started"]
    time.sleep(0.5)
    req2, _ = _request(tmp_path, "second", [PY, "-c", "print('no')"], exclusive=True)
    out = remote.launch_request(req2)
    assert out["claimed"] and out["started"] is False and "exclusively held by run hold" in out["refused"]
    assert (runs_root / "second" / "refused.json").exists() and not (runs_root / "second" / "job.json").exists()
    s2 = R.summarize(runs_root / "second")
    assert s2["state"] in ("submitted", "unknown")                 # never invented as "failed" by the runner: it never ran
    s = _wait_terminal(runs_root / "hold")
    assert s["state"] == "done" and not (runs_root / ".exclusive.lock").exists(), "the runner releases its own lock"
    req3, _ = _request(tmp_path, "third", [PY, "-c", "print('yes')"], exclusive=True)
    assert remote.launch_request(req3)["started"]
    _wait_terminal(runs_root / "third")


def test_stale_lock_from_vanished_holder_is_reclaimed(tmp_path):
    _, runs_root = _request(tmp_path, "ghost", ["true"])
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / ".exclusive.lock").write_text("ghost\n")
    (runs_root / "ghost").mkdir()
    (runs_root / "ghost" / "launch.json").write_text(json.dumps({"runner_pid": 2**22 - 7}))   # no such pid
    req, _ = _request(tmp_path, "live", [PY, "-c", "print(1)"], exclusive=True)
    assert remote.launch_request(req)["started"]
    _wait_terminal(runs_root / "live")


def test_bad_schema_is_refused_without_side_effects(tmp_path):
    req, runs_root = _request(tmp_path, "bad", ["true"])
    doc = json.loads(req.read_text()); doc["schema"] = "nope"; req.write_text(json.dumps(doc))
    with pytest.raises(remote.Refused):
        remote.launch_request(req)
    assert not (runs_root / "bad").exists()


def test_tool_tarball_is_deterministic_and_excludes_caches():
    d1, s1 = remote.tool_tarball()
    d2, s2 = remote.tool_tarball()
    assert s1 == s2 and d1 == d2
    import io, tarfile
    names = tarfile.open(fileobj=io.BytesIO(d1)).getnames()
    assert "research/cli.py" in names and "research/telemetry/run.py" in names
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_machines_resolution(tmp_path, monkeypatch):
    f = tmp_path / "m.toml"
    f.write_text('[machines.a]\nprovider="runpod"\npod_id="p"\nprojects=["verity","x"]\n[machines.b]\nprovider="runpod"\npod_id="q"\nproject="veritor"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(f))
    m = remote.resolve_machine("a", "x")
    assert m["root"] == "/workspace/research" and m["python"] == "python3" and m["name"] == "a"
    with pytest.raises(remote.Refused):
        remote.resolve_machine("b", "verity")
    with pytest.raises(remote.Refused):
        remote.resolve_machine("zzz", "verity")
    with pytest.raises(remote.Refused):
        remote.resolve_machine("a", None)
