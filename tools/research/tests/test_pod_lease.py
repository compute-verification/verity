"""A lease on every pod (pods/lease.py, pods/sh/lease.sh): the dead-man ends the pod with its own key once the lease and its grace
have passed, a failed read never extends it, it is armed at boot and over ssh by `research pods create` (which terminates a pod it
could not arm, and creates nothing a budget line does not cover), the idle guard ends an expired pod whose runs have custody, and
`research run --on` extends it.  Fake clocks (LEASE_NOW, GUARD_NOW), fake terminate commands and a fake RunPod API; the pod is a
local directory."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from research import remote
from research.pods import lease as L
from research.pods import part, runpod, workproc

from test_pod_guard_daemon import POD, QUIET, FakeRunPod, _env, mark_fetched, run_dir, tick

NOW = 1_800_000_000


class Fifo:
    """A FIFO held open for reading and writing, so a writer never blocks on open and `readline` waits for the next line."""

    def __init__(self, path: Path):
        os.mkfifo(path)
        self.path, self.fd = path, os.open(path, os.O_RDWR)

    def readline(self) -> str:
        out = b""
        while not out.endswith(b"\n"):
            out += os.read(self.fd, 1)
        return out.decode()

    def close(self) -> None:
        os.close(self.fd)


def lease_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    environ = tmp_path / "environ"
    environ.write_bytes(f"RUNPOD_POD_ID={POD}\0RUNPOD_API_KEY=pod-scoped-secret\0".encode())
    rec = tmp_path / "terminate.sh"
    rec.write_text(f'#!/bin/sh\necho "$1" >> {tmp_path / "terminated"}\necho TERMINATED "$1"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LEASE_", "RESEARCH_LEASE"))}
    return {**env, "LEASE_DIR": str(tmp_path / "lease"), "LEASE_ENVIRON": str(environ), "LEASE_TERMINATE_CMD": f"sh {rec}",
            "LEASE_DEADLINE_FILE": str(tmp_path / "deadline"), **extra}


def lease_sh(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", str(L.LEASE_SH), *args], env=env, capture_output=True, text=True)


# ---- sh/lease.sh ---------------------------------------------------------------------------------------------------------------------------------

def test_extend_pushes_the_expiry_and_never_shortens_it(tmp_path):
    env = lease_env(tmp_path, LEASE_NOW=str(NOW))
    assert lease_sh(env, "extend", "3600", "create").stdout.split()[::2] == [str(NOW + 3600), "create"]
    assert lease_sh(env, "extend", "60", "someone").stdout.split()[0] == str(NOW + 3600), "a shorter extension changes nothing"
    assert lease_sh({**env, "LEASE_NOW": str(NOW + 7200)}, "extend", "600", "run-r1").stdout.split()[::2] == [str(NOW + 7800), "run-r1"]
    assert lease_sh(env, "show").stdout.startswith(f"{NOW + 7800} ")
    assert lease_sh(env, "extend", "1.5", "x").returncode == 2


def test_the_loop_ends_the_pod_only_after_the_expiry_and_its_grace(tmp_path):
    env = lease_env(tmp_path, LEASE_NOW=str(NOW), LEASE_ONCE="1")
    lease_sh(env, "extend", "600", "create")
    assert lease_sh({**env, "LEASE_NOW": str(NOW + 600 + 899)}).returncode == 1 and not (tmp_path / "terminated").exists()
    assert lease_sh({**env, "LEASE_NOW": str(NOW + 600 + 900)}).returncode == 0
    assert (tmp_path / "terminated").read_text().split() == [POD]
    log = (tmp_path / "lease" / "lease.log").read_text()
    assert "lease expired" in log and f"+ 900s grace: terminating {POD}" in log


def test_a_deadline_clamps_every_extension_and_ends_the_machine_without_grace(tmp_path):
    env = lease_env(tmp_path, LEASE_NOW=str(NOW), LEASE_ONCE="1")
    deadline = NOW + 10_000
    (tmp_path / "deadline").write_text(f"{deadline} 2026-10-07T15:00:00Z launch\n")
    cap = deadline - 300
    out = lease_sh(env, "extend", "86400", "run-r1").stdout.split()
    assert out[0] == str(cap) and out[2] == "run-r1(clamped-to-deadline)"
    (tmp_path / "lease" / "lease").write_text(f"{deadline + 99_999} x hand-edited\n")
    assert lease_sh(env, "extend", "60", "run-r2").stdout.split()[0] == str(cap), "an expiry past the cap is pulled back to it"
    (tmp_path / "lease" / "lease").write_text(f"{deadline + 99_999} x hand-edited\n")
    assert lease_sh({**env, "LEASE_NOW": str(cap - 1)}).returncode == 1 and not (tmp_path / "terminated").exists()
    assert lease_sh({**env, "LEASE_NOW": str(cap)}).returncode == 0, "the cap ends it with no grace, whatever the lease file says"
    (tmp_path / "deadline").write_text("garbage\n")
    assert lease_sh(env).returncode == 0, "a deadline that does not parse is a deadline of 0"


def test_a_lease_file_that_does_not_parse_keeps_the_expiry_it_had(tmp_path):
    env = lease_env(tmp_path, LEASE_ONCE="1", LEASE_NOW=str(NOW))
    (tmp_path / "lease").mkdir()
    (tmp_path / "lease" / "lease").write_text("garbage\n")
    assert lease_sh({**env, "RESEARCH_LEASE_EXPIRES": str(NOW + 3600)}).returncode == 1, "the boot expiry stands: garbage extends nothing"
    assert lease_sh({**env, "RESEARCH_LEASE_EXPIRES": str(NOW - 901)}).returncode == 0


def test_the_loop_terminates_with_the_pod_key_over_graphql_and_says_when_it_cannot(tmp_path):
    api = FakeRunPod()
    env = {**lease_env(tmp_path, LEASE_ONCE="1", LEASE_NOW=str(NOW), RESEARCH_LEASE_EXPIRES=str(NOW - 1000),
                       LEASE_API_GQL=f"{api.url}/graphql", LEASE_API_REST=f"{api.url}/v1"), "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}
    env.pop("LEASE_TERMINATE_CMD")
    (Path(env["LEASE_DIR"])).mkdir()
    shim = Path(env["LEASE_DIR"]) / "python3"
    shim.symlink_to(sys.executable)
    env["PATH"] = f"{shim.parent}:{env['PATH']}"
    assert lease_sh(env).returncode == 0
    method, path, auth, body = api.calls[0]
    assert (method, path, auth) == ("POST", "/graphql", "Bearer pod-scoped-secret") and f'podTerminate(input: {{podId: \\"{POD}\\"}})' in body
    refused = FakeRunPod(rest_status=403, gql_body='{"errors": [{"message": "Unauthorized"}]}')
    env.update(LEASE_API_GQL=f"{refused.url}/graphql", LEASE_API_REST=f"{refused.url}/v1")
    assert lease_sh(env).returncode == 2 and [c[0] for c in refused.calls] == ["POST", "DELETE"]
    assert "Unauthorized" in (Path(env["LEASE_DIR"]) / "lease.log").read_text()


def test_the_lease_loop_and_its_helper_are_not_work_for_the_idle_guard():
    assert not workproc.is_work("sh /root/.research/lease.sh loop")
    assert not workproc.is_work("python3 - research-pod-lease https://api.runpod.io/graphql https://rest.runpod.io/v1 pod1")


# ---- armed at boot, and over ssh --------------------------------------------------------------------------------------------------------------

def test_with_boot_lease_a_runpod_image_runs_the_lease_before_its_start_script_and_otherwise_carries_the_expiry_only(tmp_path):
    only_env = {"env": {"RESEARCH_LEASE_EXPIRES": str(NOW)}}
    assert L.boot_fields(runpod.DEFAULT_IMAGE, NOW) == only_env, "the boot start command is opt-in"
    assert L.boot_fields("nvcr.io/nvidia/pytorch:24.01", NOW, boot=True) == only_env
    ticks = Fifo(tmp_path / "ticks")
    rec = tmp_path / "rec.sh"
    rec.write_text(f'#!/bin/sh\necho "$1" > {ticks.path}\n')
    f = L.boot_fields(runpod.DEFAULT_IMAGE, 1000, boot=True, lease_dir=str(tmp_path / "lease"), start="echo started")
    assert f["dockerStartCmd"][:2] == ["bash", "-c"] and f["env"]["RESEARCH_LEASE_SH"] == L.LEASE_SH.read_text()
    env = {**lease_env(tmp_path, LEASE_TERMINATE_CMD=f"sh {rec}"), **f["env"]}
    r = subprocess.run(f["dockerStartCmd"], env=env, capture_output=True, text=True)
    try:
        assert r.returncode == 0 and r.stdout == "started\n", "the image's own start command still runs, last"
        assert ticks.readline() == f"{POD}\n", "the lease loop started at boot ended the pod: its expiry is long past"
    finally:
        ticks.close()
    assert (tmp_path / "lease" / "lease.sh").read_text() == L.LEASE_SH.read_text()


def _arm_locally(tmp_path: Path, seconds: int = 3600):
    env = lease_env(tmp_path)
    genv = _env(tmp_path, QUIET)
    extra = {k: env[k] for k in ("LEASE_ENVIRON", "LEASE_TERMINATE_CMD")} | {"LEASE_TICK_S": "3600", "GUARD_TICK_S": "3600"} \
        | {k: genv[k] for k in ("GUARD_PS", "GUARD_NVSMI", "GUARD_ENVIRON", "GUARD_TERMINATE_CMD")}

    def run_here(ssh, argv, stdin=None, timeout=None):
        assert ssh == ["ssh", "pod"]
        return subprocess.run(argv, input=stdin, capture_output=True)

    return L.arm(["ssh", "pod"], seconds, first_idle_min=20, idle_min=90, root=str(tmp_path / "root"), lease_dir=str(tmp_path / "lease"),
                 extra_env=extra, run=run_here)


def _kill(*pids):
    for pid in pids:
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def test_arming_over_ssh_writes_the_lease_starts_one_loop_and_the_idle_guard(tmp_path, reap):
    first = _arm_locally(tmp_path)
    try:
        assert first["lease_alive"] and first["guard_alive"] and first["expires_epoch"]
        line = (tmp_path / "lease" / "lease").read_text().split()
        assert int(line[0]) == first["expires_epoch"] and line[2] == "create"
        assert {p.name for p in (tmp_path / "lease").iterdir()} >= {"lease.sh", "pod_guard.sh", "lease", "lease.pid"}
        again = _arm_locally(tmp_path)
        assert again["lease_pid"] == first["lease_pid"], "a live lease loop is kept, not doubled"
        _kill(again["guard_pid"])
    finally:
        _kill(first["lease_pid"], first["guard_pid"])


def test_idle_min_zero_arms_the_lease_and_no_idle_guard(tmp_path, monkeypatch, reap):
    real = L.arm_script
    monkeypatch.setattr(L, "arm_script", lambda seconds, **kw: real(seconds, **{**kw, "idle_min": 0}))
    rep = _arm_locally(tmp_path)
    try:
        assert rep["lease_alive"] and rep["guard_pid"] is None and rep["guard_alive"] is False
        assert not (tmp_path / "root" / "guard" / "guard.pid").exists()
    finally:
        _kill(rep["lease_pid"])


def test_an_arm_that_fails_raises():
    with pytest.raises(RuntimeError, match="arming the lease failed"):
        L.arm(["ssh", "pod"], 60, run=lambda *a, **k: subprocess.CompletedProcess(a, 255, b"", b"ssh: connect refused"))


# ---- research pods create -----------------------------------------------------------------------------------------------------------------------

def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(name="vy-lane", allow_duplicate=False, max_hours=2.0, first_idle_min=20.0, idle_min=90.0, **kw)


@pytest.fixture
def account(monkeypatch):
    made: list[argparse.Namespace] = []
    monkeypatch.setattr(runpod, "_request", lambda method, path, body=None: [])
    monkeypatch.setattr(runpod, "create", lambda ns: made.append(ns) or {"id": "newpod"})
    return made


def test_create_refuses_what_no_budget_line_lets_and_creates_nothing(account, capsys):
    pod, rc = part.create_leased(_ns(), gate=lambda name, h: ("no budget line covers 'vy-lane-veritor-campaign'", None))
    assert (pod, rc) == (None, part.EXIT_UNBUDGETED) and account == []
    assert "REFUSED vy-lane: no budget line covers" in capsys.readouterr().err


def test_create_leases_the_pod_from_its_boot_and_arms_it_once_sshd_answers(account):
    seen = {}

    def arm(ssh, seconds, **kw):
        seen.update(ssh=ssh, seconds=seconds, **kw)
        return {"expires_epoch": NOW + 7200, "expires_utc": "t", "lease_pid": 1, "lease_alive": True, "guard_pid": 2, "guard_alive": True}

    pod, rc = part.create_leased(_ns(), gate=lambda name, h: (None, "vy-"), wait_ssh=lambda pid: ("1.2.3.4", 22), arm=arm,
                                 terminate=lambda pid: pytest.fail("an armed pod is not terminated"), clock=lambda: NOW)
    assert rc == 0 and pod["lease"]["lease_alive"] and account[0].lease_expires == NOW + 7200
    assert seen["seconds"] == 7200 and seen["first_idle_min"] == 20.0 and seen["ssh"][-1] == "root@1.2.3.4"


@pytest.mark.parametrize("stage", ["ssh", "arm"])
def test_create_terminates_a_pod_it_could_not_arm(account, stage, capsys):
    killed = []

    def no_sshd(pid):
        raise runpod.PodError("sshd did not answer within 1800s")

    def bad_arm(*a, **k):
        raise RuntimeError("arming the lease failed (rc 1)")

    pod, rc = part.create_leased(_ns(), gate=lambda name, h: (None, "vy-"), wait_ssh=no_sshd if stage == "ssh" else (lambda pid: ("h", 22)),
                                 arm=bad_arm, terminate=killed.append, clock=lambda: NOW)
    assert (pod, rc) == (None, part.EXIT_FAILED) and killed == ["newpod"]
    assert "could not be armed" in capsys.readouterr().err


def test_the_create_call_carries_the_lease_in_the_pods_environment_and_start_command(monkeypatch):
    bodies = []
    monkeypatch.setattr(runpod, "public_key", lambda: "ssh-ed25519 AAAA test")
    monkeypatch.setattr(runpod, "_request", lambda method, path, body=None: bodies.append(body) or {"id": "p"})
    ap = argparse.ArgumentParser()
    runpod.create_args(ap)
    ns = ap.parse_args(["--name", "vy-x", "--cpu", "cpu3c", "--vcpu", "2"])
    ns.lease_expires = NOW
    runpod.create(ns)
    b = bodies[0]
    assert b["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA test" and b["env"]["RESEARCH_LEASE_EXPIRES"] == str(NOW) and "dockerStartCmd" not in b
    ns.boot_lease = True
    runpod.create(ns)
    b = bodies[1]
    assert b["dockerStartCmd"][0] == "bash" and b["dockerStartCmd"][2].endswith("exec /start.sh") and "RESEARCH_LEASE_SH" in b["env"]


# ---- the idle guard and the lease ----------------------------------------------------------------------------------------------------------------

def _lease_file(tmp_path: Path, epoch: float) -> Path:
    f = tmp_path / "lease-file"
    f.write_text(f"{int(epoch)} t create\n")
    return f


def test_a_pod_that_never_gets_a_run_waits_only_the_first_run_window(tmp_path):
    root = tmp_path / "root"
    env = _env(tmp_path, QUIET, GUARD_FIRST_IDLE_MIN="20", GUARD_LEASE_FILE=str(tmp_path / "none"))
    tick(root, {**env, "GUARD_NOW": str(NOW - 21 * 60)})
    assert tick(root, {**env, "GUARD_NOW": str(NOW)})["verdict"] == "terminate"
    mark_fetched(run_dir(root, "r1"))
    assert tick(root, {**env, "GUARD_NOW": str(NOW)})["verdict"] == "idle", "after a first run the full idle limit applies"


def test_an_expired_lease_ends_a_pod_whose_runs_have_custody_busy_or_not_and_waits_for_one_that_lacks_it(tmp_path):
    root = tmp_path / "root"
    lf = _lease_file(tmp_path, NOW - 60)
    busy = _env(tmp_path, QUIET + ["python train.py"], now=NOW, GUARD_LEASE_FILE=str(lf))
    st = tick(root, busy)
    assert st["verdict"] == "expired" and st["lease_expires_epoch"] == NOW - 60
    run_dir(root, "r-open")
    st = tick(root, busy)
    assert st["verdict"] == "lease-wait" and "waiting for the custody of r-open" in st["refusal"]
    lf.write_text("not a number\n")
    assert tick(root, busy)["lease_expires_epoch"] == NOW - 60, "an unreadable lease keeps the expiry the guard last read"
    assert tick(root, {**busy, "GUARD_LEASE_FILE": str(_lease_file(tmp_path, NOW + 60))})["verdict"] == "busy"


def test_the_daemon_ends_an_expired_pod_on_its_first_tick(tmp_path):
    root = tmp_path / "root"
    mark_fetched(run_dir(root, "r-done"))
    (tmp_path / "preserve").write_text("art:not-preserved-anywhere\n")
    env = _env(tmp_path, QUIET, now=NOW, GUARD_LEASE_FILE=str(_lease_file(tmp_path, NOW - 1)), GUARD_PRESERVE_FILE=str(tmp_path / "preserve"),
               GUARD_RESEARCH="false")
    proc = subprocess.Popen(["bash", str(remote.GUARD_SCRIPT), "daemon", str(root), "90"], env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.wait() == 0
    assert (tmp_path / "terminated").read_text().split() == [POD]
    st = json.loads((root / "guard" / "state.json").read_text())
    assert st["verdict"] == "terminated" and "lease expired" in (root / "guard" / "guard.log").read_text()


# ---- research run --on extends it -----------------------------------------------------------------------------------------------------------



def test_launch_request_extends_the_lease_for_a_started_run(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_launch_request", lambda p: {"claimed": True, "started": True})
    seen = []
    monkeypatch.setattr(L, "machine_extend", lambda s, by: seen.append((s, by)) or {"leased": True, "extended": True, "expires_utc": "T"})
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"run_id": "r1", "runs_root": "/x/runs", "guard": None, "lease_extend_s": 4500.0}))
    assert remote.launch_request(req)["lease"]["extended"] and seen == [(4500.0, "run-r1")]
    req.write_text(json.dumps({"run_id": "r1", "runs_root": "/x/runs", "guard": None, "lease_extend_s": None}))
    assert "lease" not in remote.launch_request(req) and len(seen) == 1


def test_the_launcher_asks_for_the_runs_timeout_plus_its_margin_and_reports_the_answer():
    assert remote.lease_extend_s(["--stage", "run", "--timeout", "3600"]) == 3600 + L.RUN_MARGIN_S
    assert remote.lease_extend_s(["--timeout=60"]) == 60 + L.RUN_MARGIN_S and remote.lease_extend_s(["--stage", "run"]) is None
    assert "no --timeout" in remote.lease_line(None, None, "vy-x")
    assert "predates leases" in remote.lease_line(None, 100.0, "vy-x")
    assert "has no lease" in remote.lease_line({"leased": False, "reason": "no lease"}, 100.0, "vy-control")
    assert "no sooner than T" in remote.lease_line({"leased": True, "extended": True, "expires_utc": "T"}, 100.0, "vy-x")
    assert "could NOT be extended" in remote.lease_line({"leased": True, "extended": False, "error": "e"}, 100.0, "vy-x")
