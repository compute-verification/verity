"""The research package that runs on a machine is the one the launcher shipped (remote.tool_identity), or the run is refused with
ResearchVersionMismatch before anything starts -- by `launch-request --expect-tool` and again by the runner.  A package that
predates the check fails closed on the flag; a --source tree's older package is allowed, and said so."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from research import remote
from research import runs as R
from waits import wait_runner

PY = sys.executable
PKG = Path(remote.__file__).resolve().parent
SRC = str(PKG.parent)


def _copy_pkg(dst: Path) -> Path:
    shutil.copytree(PKG, dst / "research", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst / "research"


def test_identity_is_content_only(tmp_path):
    pkg = _copy_pkg(tmp_path / "a")
    assert remote.tool_identity(pkg) == remote.tool_identity()
    os.chmod(pkg / "mem.py", 0o600)
    (pkg / "__pycache__").mkdir(exist_ok=True)
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert remote.tool_identity(pkg) == remote.tool_identity(), "modes and bytecode do not enter"
    (pkg / "mem.py").write_text((pkg / "mem.py").read_text() + "\n# changed\n")
    assert remote.tool_identity(pkg) != remote.tool_identity()
    assert remote.knows_version_check(None) and remote.knows_version_check(pkg)
    (pkg / "remote.py").write_text("# an older package\n")
    assert not remote.knows_version_check(pkg)


def _request(root: Path, run_id: str, *, identity: str | None) -> Path:
    (root / "requests" / run_id).mkdir(parents=True)
    marker = root / f"ran-{run_id}"
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": str(root),
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler"], "command": [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"],
           "exclusive": False, "cost": {"provider": "runpod", "pod_id": "p", "rate_usd_per_hour": 0.1, "rate_observed_utc": "t", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p", "root": str(root), "tool": SRC, "tool_identity": identity}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p


def _launch_request(*args: str, pythonpath: str = SRC) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", "research", "launch-request", *args], capture_output=True, env={**os.environ, "PYTHONPATH": pythonpath})


def _settled(run_dir: Path) -> dict:
    wait_runner(run_dir)
    return R.summarize(run_dir)


def test_launch_request_refuses_a_package_that_is_not_the_shipped_one_before_claiming(tmp_path):
    root = tmp_path / "root"
    r = _launch_request("--expect-tool", "0" * 16, str(_request(root, "wrong", identity=remote.tool_identity())))
    out = json.loads(r.stdout.decode().splitlines()[-1])
    assert r.returncode == 3 and out["fault"] == "ResearchVersionMismatch" and f"({PKG}, identity {remote.tool_identity()})" in out["refused"]
    assert not (root / "runs" / "wrong").exists(), "nothing claimed, nothing started"
    err = remote.launch_request_failure(r.returncode, r.stdout, r.stderr)
    assert isinstance(err, remote.ResearchVersionMismatch) and "not the one the launcher shipped" in str(err)

    r = _launch_request("--expect-tool", remote.tool_identity(), str(_request(root, "right", identity=remote.tool_identity())))
    assert r.returncode == 0 and json.loads(r.stdout.decode().splitlines()[-1])["started"]
    run_dir = root / "runs" / "right"
    assert _settled(run_dir)["state"] == "done" and (root / "ran-right").exists()
    running = json.loads((run_dir / "job.json").read_text())["run"]["remote"]["tool_running"]
    assert running == {"identity": remote.tool_identity(), "path": str(PKG)}, "job.json records the package that actually ran"


def test_a_package_that_predates_the_check_fails_closed_and_the_launcher_names_it(tmp_path):
    old = _copy_pkg(tmp_path / "old")
    cli = (old / "cli.py").read_text()
    (old / "cli.py").write_text(cli.replace('ap.add_argument("--expect-tool"', 'ap.add_argument("--not-yet"'))
    root = tmp_path / "root"
    r = _launch_request("--expect-tool", remote.tool_identity(), str(_request(root, "old", identity=None)), pythonpath=str(old.parent))
    assert r.returncode == 2 and not (root / "runs" / "old").exists()
    err = remote.launch_request_failure(r.returncode, r.stdout, r.stderr)
    assert isinstance(err, remote.ResearchVersionMismatch) and "predates the version check" in str(err)


def test_the_runner_refuses_itself_when_it_is_not_the_shipped_package(tmp_path):
    root = tmp_path / "root"
    out = remote.launch_request(_request(root, "shadowed", identity="f" * 16))
    assert out["started"] is False and out["fault"] == "ResearchVersionMismatch", \
        "launch_request (called in-process here, no --expect-tool) starts the runner, which checks too and refuses before its first record"
    run_dir = root / "runs" / "shadowed"
    assert _settled(run_dir)["state"] == "refused"
    refused = json.loads((run_dir / "refused.json").read_text())
    assert out["refused"] == refused["reason"], "the launch answers with the runner's own refusal"
    assert refused["fault"] == "ResearchVersionMismatch" and "identity " + "f" * 16 in refused["reason"]
    assert not (root / "ran-shadowed").exists() and not (run_dir / "job.json").exists(), "the workload never ran"
