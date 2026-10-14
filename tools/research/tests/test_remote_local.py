"""The machine-side half of --on, exercised locally: claim by run id, duplicate delivery, exclusive lock, refusal record."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from research import remote
from research import runs as R
from waits import Gate, wait_runner

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


def _wait_terminal(run_dir):
    wait_runner(run_dir)
    return R.summarize(run_dir)


def test_claim_starts_detached_runner_and_redelivery_is_idempotent(tmp_path):
    gate = Gate(tmp_path / "gate")
    req, runs_root = _request(tmp_path, "r1", [PY, "-c", f"{Gate.WAIT}; import os; print(os.environ['RESEARCH_RUN_ID'])", str(gate.path)])
    try:
        out = remote.launch_request(req)
        assert out["claimed"] and out["started"] and out["runner_pid"]
        again = remote.launch_request(req)                      # the ssh session "dropped" and the launcher re-delivered, mid-run
        assert again["claimed"] is False and again["existing"] and again["runner_pid"] == out["runner_pid"]
        gate.release()
        s = _wait_terminal(runs_root / "r1")
    finally:
        gate.close()
    assert s["state"] == "done" and s["rc"] == 0
    assert (runs_root / "r1" / "stdout.log").read_text().strip() == "r1"
    job = json.loads((runs_root / "r1" / "job.json").read_text())
    assert job["run"]["remote"]["pod_id"] == "p1" and job["cost"]["rate_usd_per_hour"] == 0.06 and job["cost"]["shared"] is False
    assert s["cost"]["usd"] == round(0.06 * s["cost"]["wall_s"] / 3600, 4), "the request's rate times the run's wall time"
    launch = json.loads((runs_root / "r1" / "launch.json").read_text())
    assert launch["argv"][launch["argv"].index("--") + 1] == PY, "argv travels as a list; no shell string"


def test_a_claimed_run_needs_no_remote_of_its_own(tmp_path, monkeypatch):
    """Its launcher settled custody (--custody-r2, or --no-custody-r2 and `research fetch --all`): no local-only refusal here."""
    monkeypatch.delenv("RESEARCH_LOCAL_ONLY")
    req, runs_root = _request(tmp_path, "r0", [PY, "-c", "print('ran')"])
    assert remote.launch_request(req)["started"]
    s = _wait_terminal(runs_root / "r0")
    assert s["state"] == "done" and (runs_root / "r0" / "stdout.log").read_text() == "ran\n"
    assert "local_only" not in json.loads((runs_root / "r0" / "job.json").read_text())["store"]


def test_env_and_spaces_reach_the_workload(tmp_path):
    req, runs_root = _request(tmp_path, "r2", [PY, "-c", "import os, sys; print(os.environ['FOO'], '|', sys.argv[1])", "two  words"], env={"FOO": "bar baz"})
    assert remote.launch_request(req)["started"]
    s = _wait_terminal(runs_root / "r2")
    assert (runs_root / "r2" / "stdout.log").read_text() == "bar baz | two  words\n"


def test_exclusive_refuses_a_live_holder_and_reclaims_a_dead_one(tmp_path):
    gate = Gate(tmp_path / "gate")
    req1, runs_root = _request(tmp_path, "hold", [PY, "-c", Gate.WAIT, str(gate.path)], exclusive=True)
    try:
        assert remote.launch_request(req1)["started"]
        req2, _ = _request(tmp_path, "second", [PY, "-c", "print('no')"], exclusive=True)
        out = remote.launch_request(req2)
        assert out["claimed"] and out["started"] is False and "exclusively held by run hold" in out["refused"]
        assert (runs_root / "second" / "refused.json").exists() and not (runs_root / "second" / "job.json").exists()
        s2 = R.summarize(runs_root / "second")
        assert s2["state"] == "refused"                            # never invented as "failed" by the runner: it never ran
        gate.release()
        s = _wait_terminal(runs_root / "hold")
    finally:
        gate.close()
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


def test_a_queued_run_holds_its_cpu_allocation_through_custody_and_its_gpu_lease_around_the_workload_alone(tmp_path):
    """--queue: the runner runs inside the job's CPU allocation (a systemd scope on vy-nebius-2), which the stand-in wrapper
    holds until the runner exits: the run's status is already terminal then. A gpu phase's lease (gpu-lease) wraps the
    workload alone, so the GPU is not held through custody."""
    wrap = [PY, "-c", "import json, os, subprocess, sys; rc = subprocess.call(sys.argv[2:], env={**os.environ, 'HELD': 'yes'}); "
            "open(sys.argv[1] + '.at-release', 'w').write(json.load(open(sys.argv[1]))['state']); sys.exit(rc)",
            str(tmp_path / "root" / "runs" / "q1" / "status.json")]
    lease = [PY, "-c", "import os, subprocess, sys; sys.exit(subprocess.call(sys.argv[1:], env={**os.environ, 'LEASED': 'yes'}))"]
    req, runs_root = _request(tmp_path, "q1", [PY, "-c", "import os; print(os.environ['HELD'], os.environ['LEASED'])"])
    doc = json.loads(req.read_text())
    doc[remote.QUEUE_WRAP], doc[remote.QUEUE_WORKLOAD_WRAP] = wrap, lease
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    s = _wait_terminal(runs_root / "q1")
    assert out["started"] and s["state"] == "done" and (runs_root / "q1" / "stdout.log").read_text().strip() == "yes yes"
    assert (runs_root / "q1" / "status.json.at-release").read_text() == "done"
    argv = json.loads((runs_root / "q1" / "launch.json").read_text())["argv"]
    assert argv[: len(wrap)] == wrap and argv[len(wrap):][1:3] == ["-m", "research"]
    assert argv[argv.index("--", len(wrap)) + 1:][: len(lease)] == lease


def test_a_queued_run_with_a_disk_request_is_admitted_on_the_machine_ahead_of_its_allocation(tmp_path):
    req, runs_root = _request(tmp_path, "d1", [PY, "-c", "print('ok')"])
    doc = json.loads(req.read_text())
    doc["queue"] = {"disk_gb": 0.001}
    doc["env"] = {**(doc.get("env") or {}), "RESEARCH_DISK_HOLD_PCT": "100"}   # admitted however full this host's disk is
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    assert out["started"] and _wait_terminal(runs_root / "d1")["state"] == "done"
    argv = json.loads((runs_root / "d1" / "launch.json").read_text())["argv"]
    assert argv[1:5] == ["-m", "research", "disk-hold", "--gb"] and argv[argv.index("--") + 1:][1:4] == ["-m", "research", "run"]


def test_bad_schema_is_refused_without_side_effects(tmp_path):
    req, runs_root = _request(tmp_path, "bad", ["true"])
    doc = json.loads(req.read_text()); doc["schema"] = "nope"; req.write_text(json.dumps(doc))
    with pytest.raises(remote.Refused):
        remote.launch_request(req)
    assert not (runs_root / "bad").exists()


def test_a_machine_that_takes_runs_through_its_check_slots_refuses_any_other(tmp_path, monkeypatch):
    """/etc/vy/runs-via-slot (cpu-1's, deploy.toml): the launcher starts a run only when its command is slot.py's (`python3
    tools/verity/check/slot.py [OPTS] -- CMD`); three runs beside cpu-1's merge check took its slice from 586G to 647G (8 Oct)."""
    marker = tmp_path / "runs-via-slot"
    monkeypatch.setenv("RUNS_VIA_SLOT_FILE", str(marker))
    marker.write_text("")
    req, runs_root = _request(tmp_path, "off", [PY, "-c", "print('no')"])
    out = remote.launch_request(req)
    assert out["started"] is False and "only through its check slots" in out["refused"] and str(marker) in out["refused"]
    assert R.summarize(runs_root / "off")["state"] == "refused" and not (runs_root / "off" / "job.json").exists()
    req, _ = _request(tmp_path, "slot", ["python3", "tools/verity/check/slot.py", "--by", "t", "--", "true"])
    doc = json.loads(req.read_text()); doc["cwd"] = str(tmp_path); req.write_text(json.dumps(doc))   # no slot.py here: the run fails
    assert remote.launch_request(req)["started"]
    _wait_terminal(runs_root / "slot")


def test_the_machines_deployed_slot_rule_decides_not_the_shipped_trees(tmp_path, monkeypatch):
    """With /usr/local/lib/vy-monitoring/slot_only.py installed (deploy.toml, cpu-1), its `through_slot` judges the command: a lane
    whose branch predates a rule change (proofs' C3 branches and the `env CHECK_SLOTS=` pool, 13 Oct) is held to the deployed rule."""
    marker = tmp_path / "runs-via-slot"
    marker.write_text("")
    monkeypatch.setenv("RUNS_VIA_SLOT_FILE", str(marker))
    node = tmp_path / "slot_only.py"
    node.write_text("def through_slot(command):\n    return command[:1] == ['echo']\n\n"
                    "def refusal(marker):\n    return f'node rule ({marker})'\n")
    monkeypatch.setenv("NODE_SLOT_ONLY_FILE", str(node))
    req, runs_root = _request(tmp_path, "node-off", ["python3", "tools/verity/check/slot.py", "--", "true"])
    out = remote.launch_request(req)
    assert out["started"] is False and out["refused"] == f"node rule ({marker})"
    req, _ = _request(tmp_path, "node-on", ["echo", "ok"])
    assert remote.launch_request(req)["started"]
    _wait_terminal(runs_root / "node-on")
    node.unlink()
    assert remote.node_slot_only().through_slot(["python3", "tools/verity/check/slot.py", "--", "true"])


def test_tool_tarball_is_deterministic_and_excludes_caches():
    d1, s1 = remote.tool_tarball()
    d2, s2 = remote.tool_tarball()
    assert s1 == s2 and d1 == d2
    import io, tarfile
    names = tarfile.open(fileobj=io.BytesIO(d1)).getnames()
    assert "research/cli.py" in names and "research/telemetry/run.py" in names
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_source_tree_research_package_is_the_tool_that_ships(tmp_path, monkeypatch, capsys):
    """`research run --on M --source <tree>`: when the tree carries tools/research/src/research, THAT package is tarred as the tool
    (the machine executes it), not the launcher's own install; without one the launcher's package ships, and a `--tool` it does not
    know is refused before anything is shipped."""
    assert remote.source_tool_package(None) is None
    assert remote.source_tool_package(tmp_path) is None                       # no tools/research in this tree
    pkg = tmp_path / "tools" / "research" / "src" / "research"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "marker_only_in_the_tree.py").write_text("X = 1\n")
    assert remote.source_tool_package(tmp_path) == pkg
    import io, tarfile
    data, sha = remote.tool_tarball(pkg)
    names = tarfile.open(fileobj=io.BytesIO(data)).getnames()
    assert "research/marker_only_in_the_tree.py" in names and "research/cli.py" not in names
    assert sha != remote.tool_tarball()[1]                                    # a different package is a different tool identity
    # the refusal: the launcher's registry does not know `nope.tool` and no --source package would replace it
    from research import cli
    rc = cli.main(["run", "--on", "m", "--project", "p", "--tool", "nope.tool", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "launcher research package does not know tool 'nope.tool'" in err and "--source with tools/research" in err


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


def test_exclusive_refuses_when_any_run_is_live_and_vice_versa(tmp_path):
    first, second = Gate(tmp_path / "first"), Gate(tmp_path / "second")
    plain, runs_root = _request(tmp_path, "plain", [PY, "-c", Gate.WAIT, str(first.path)])
    try:
        assert remote.launch_request(plain)["started"]
        excl, _ = _request(tmp_path, "excl", [PY, "-c", "print(1)"], exclusive=True)
        out = remote.launch_request(excl)
        assert out["started"] is False and "other run(s) are live" in out["refused"] and "plain" in out["refused"]
        assert not (runs_root / ".exclusive.lock").exists(), "a refused exclusive launch leaves no lock behind"
        first.release()
        _wait_terminal(runs_root / "plain")
        holder, _ = _request(tmp_path, "holder", [PY, "-c", Gate.WAIT, str(second.path)], exclusive=True)
        assert remote.launch_request(holder)["started"]
        late, _ = _request(tmp_path, "late", [PY, "-c", "print(1)"])
        out = remote.launch_request(late)
        assert out["started"] is False and "exclusively held by run holder" in out["refused"], "a plain launch respects a live exclusive holder"
        second.release()
        _wait_terminal(runs_root / "holder")
    finally:
        first.close()
        second.close()


def test_snapshot_identity_reaches_the_record(tmp_path):
    req, runs_root = _request(tmp_path, "snap", [PY, "-c", "pass"])
    doc = json.loads(req.read_text()); doc["tool"] = str(Path(SRC)); req.write_text(json.dumps(doc))
    assert remote.launch_request(req)["started"]
    _wait_terminal(runs_root / "snap")
    job = json.loads((runs_root / "snap" / "job.json").read_text())
    assert job["tool"]["research_snapshot"] == Path(SRC).name


def test_source_sha_reaches_the_workload_as_research_source_sha(tmp_path):
    """[R19 tele, F-r17-tp-06] a `--source` run executes from <root>/src/<sha>/, a `git archive` export with no .git: the workload's
    source-identity precheck learns the tree's sha from RESEARCH_SOURCE_SHA (request.json remote.source_sha).  A run without a shipped
    tree does not carry the variable."""
    probe = "import os; print(os.environ.get('RESEARCH_SOURCE_SHA', '<unset>'))"
    req, runs_root = _request(tmp_path, "s1", [PY, "-c", probe])
    d = json.loads(req.read_text())
    d["remote"]["source_sha"] = "a" * 40
    req.write_text(json.dumps(d))
    src = runs_root.parent / "src" / ("a" * 40)                  # the shipped tree must be verified READY (test_remote_ship_source.py) to execute
    src.mkdir(parents=True)
    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": "a" * 40, "archive_sha256": "0" * 64}))
    assert remote.launch_request(req)["started"]
    s = _wait_terminal(runs_root / "s1")
    assert s["state"] == "done" and (runs_root / "s1" / "stdout.log").read_text().strip() == "a" * 40
    req2, _ = _request(tmp_path, "s2", [PY, "-c", probe])
    assert remote.launch_request(req2)["started"]
    _wait_terminal(runs_root / "s2")
    assert (runs_root / "s2" / "stdout.log").read_text().strip() == "<unset>"


def test_pod_target_rereads_a_running_pod_whose_mapping_flaps(monkeypatch):
    """RunPod's REST GET /pods/{id} intermittently answers a RUNNING pod with publicIp='' / portMappings=None (2026-09-22,
    s00kzu0nisihgr: alternating answers within seconds while ssh worked).  pod_target re-reads before refusing; a mapping that never
    appears is still refused by name."""
    from research.pods import runpod

    answers = iter([{"desiredStatus": "RUNNING", "publicIp": "", "portMappings": None, "name": "p"},
                    {"desiredStatus": "RUNNING", "publicIp": "", "portMappings": None, "name": "p"},
                    {"desiredStatus": "RUNNING", "publicIp": "10.0.0.1", "portMappings": {"22": 2222}, "name": "p", "costPerHr": 1.0}])
    monkeypatch.setattr(runpod, "get", lambda pid: next(answers))
    monkeypatch.setattr(remote, "POD_TARGET_RETRY_S", 0.0)
    tgt = remote.pod_target({"provider": "runpod", "pod_id": "x"})
    assert (tgt["host"], tgt["port"]) == ("10.0.0.1", 2222)

    monkeypatch.setattr(runpod, "get", lambda pid: {"desiredStatus": "RUNNING", "publicIp": "", "portMappings": None})
    with pytest.raises(remote.Refused, match="no ssh port mapped yet"):
        remote.pod_target({"provider": "runpod", "pod_id": "x"})
