"""The pod-side guard `research run --on` starts (pods/sh/pod_guard.sh `daemon`, research.remote.ensure_guard) and the `.fetched`
marker `research fetch --all` leaves on the machine.  Fake process tables, GPUs, clocks and a fake terminate command; the machine is
a local directory reached through `sh -c` instead of ssh."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import tracemalloc
from pathlib import Path

import pytest

from research import remote
from research.pods import workproc

GUARD = remote.GUARD_SCRIPT
POD = "podfake123"


def _env(tmp_path: Path, ps: list[str], *, gpu: list[str] | None = None, apps: int = 0, now: float | None = None,
         key: bool = True, **extra: str) -> dict[str, str]:
    (tmp_path / "ps.txt").write_text("".join(f"{line}\n" for line in ps))
    nv = tmp_path / "nvidia-smi"
    if gpu is None:
        nv.write_text("#!/bin/sh\nexit 127\n")                      # a CPU pod: no nvidia-smi answer
    else:
        body = "\n".join(gpu)
        nv.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu*) printf '%s\\n' '{body}';; --query-compute-apps*) "
                      + ("seq 1 %d;;" % apps if apps else "true;;") + " esac\n")
    nv.chmod(0o755)
    environ = tmp_path / "environ"
    environ.write_bytes(b"\0".join([b"HOME=/root", f"RUNPOD_POD_ID={POD}".encode()] + ([b"RUNPOD_API_KEY=pod-scoped-secret"] if key else [])) + b"\0")
    env = {**os.environ, **workproc.guard_env(), "GUARD_TERMINAL": " ".join(remote.TERMINAL), "GUARD_PS": f"cat {tmp_path / 'ps.txt'}",
           "GUARD_NVSMI": str(nv), "GUARD_ENVIRON": str(environ), "GUARD_TERMINATE_CMD": f"sh {_recorder(tmp_path)}"}
    env.pop("RUNPOD_POD_ID", None)
    if now is not None:
        env["GUARD_NOW"] = str(now)
    env.update(extra)
    return env


def _recorder(tmp_path: Path) -> Path:
    rec = tmp_path / "terminate.sh"
    rec.write_text(f"#!/bin/sh\necho \"$1\" >> {tmp_path / 'terminated'}\necho TERMINATED \"$1\"\n")
    return rec


def tick(root: Path, env: dict[str, str], idle_min: float = 90) -> dict:
    r = subprocess.run(["bash", "-c", f'. "{GUARD}"; guard_tick "$0" "$1"', str(root), str(idle_min)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("VERDICT"), r.stdout + r.stderr
    return json.loads((root / "guard" / "state.json").read_text())


def run_dir(root: Path, rid: str, state: str = "done", pid: int | None = None, age_s: float = 3600) -> Path:
    d = root / "runs" / rid
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "pid": pid}))
    (d / "launch.json").write_text(json.dumps({"run_id": rid, "runner_pid": pid}))
    (d / "result.json").write_text("{}")
    old = time.time() - age_s
    for p in d.iterdir():
        os.utime(p, (old, old))
    return d


def mark_fetched(d: Path, verified_from: float | None = None) -> None:
    (d / ".fetched").write_text(json.dumps({"schema": remote.FETCHED_SCHEMA, "verified_from": time.time() if verified_from is None else verified_from}))


def idle_since(root: Path, env: dict[str, str], minutes: float) -> dict:
    """A first tick seeds the idle clock; a second one `minutes` later decides."""
    now = time.time()
    tick(root, {**env, "GUARD_NOW": str(now - 60 * minutes)})
    return tick(root, {**env, "GUARD_NOW": str(now)})


QUIET = ["/usr/sbin/sshd -D", "sleep 60", "bash /workspace/research/tool/x/research/pods/sh/pod_guard.sh daemon /workspace/research 90",
         "python3 - research-pod-guard /workspace/research 90", "sh -c # research-notes-probe", "/usr/bin/python3 -m jupyter lab",
         "nginx: master process", "/start.sh"]


# ---- the activity definition (shared with the watcher's probe) ------------------------------------------------------------------------

def test_pod_services_the_probe_and_the_guard_itself_are_not_work(tmp_path):
    assert not any(workproc.is_work(a) for a in QUIET)
    st = tick(tmp_path / "root", _env(tmp_path, QUIET))
    assert st["busy"] is False and st["activity"]["work"] == 0 and st["verdict"] == "idle"


@pytest.mark.parametrize("line", ["python train.py --steps 100", "cargo build --release", "/root/.cargo/bin/rustc x.rs",
                                  "bash run_bench.sh", "ligero-prover --n 20"])
def test_a_work_process_is_activity(tmp_path, line):
    assert workproc.is_work(line)
    st = tick(tmp_path / "root", _env(tmp_path, QUIET + [line]))
    assert st["busy"] and st["verdict"] == "busy" and st["activity"]["work"] == 1 and line[:40] in st["activity"]["work_args"][0]


def test_a_live_verifier_gpu_use_and_gpu_apps_are_activity(tmp_path):
    root = tmp_path / "root"
    assert tick(root, _env(tmp_path, QUIET + ["/opt/verity live serve --port 9000"]))["activity"]["serve"] == 1
    st = tick(root, _env(tmp_path, QUIET, gpu=["0", "37"]))
    assert st["busy"] and st["activity"]["gpu_util"] == 37
    st = tick(root, _env(tmp_path, QUIET, gpu=["0", "0"], apps=2))
    assert st["busy"] and st["activity"]["gpu_apps"] == 2
    st = tick(root, _env(tmp_path, QUIET, gpu=["0", "0"]))
    assert not st["busy"] and st["activity"]["gpu_util"] == 0
    assert tick(root, _env(tmp_path, QUIET))["activity"]["gpu_util"] is None, "no nvidia-smi: a CPU pod"


def test_a_live_runner_is_activity_and_a_dead_or_finished_one_is_not(tmp_path):
    root = tmp_path / "root"
    run_dir(root, "r-live", state="running", pid=os.getpid())
    assert tick(root, _env(tmp_path, QUIET))["activity"]["runners"] == ["r-live"]
    (root / "runs" / "r-live" / "status.json").write_text(json.dumps({"state": "running", "pid": 99999999}))
    (root / "runs" / "r-live" / "launch.json").write_text(json.dumps({"runner_pid": 99999999}))
    assert tick(root, _env(tmp_path, QUIET))["activity"]["runners"] == [], "a crashed runner (dead pids) does not hold the pod"
    run_dir(root, "r-done", state="done", pid=os.getpid())
    assert tick(root, _env(tmp_path, QUIET))["activity"]["runners"] == []


def test_the_watcher_probe_uses_the_same_definition():
    from research import notes

    assert f"grep -E '{workproc.WORK_RE}'" in notes.POD_PROBE and f"grep -vE '{workproc.NOT_WORK_RE}'" in notes.POD_PROBE
    assert workproc.FIND_PRUNE in notes.POD_PROBE and "-name guard" in workproc.FIND_PRUNE, "the guard's own writes are not activity for either"
    assert notes.PROBE_MARK == workproc.PROBE_MARK


# ---- decisions -------------------------------------------------------------------------------------------------------------------------

def test_idle_below_the_limit_waits_and_busy_resets_the_clock(tmp_path):
    root = tmp_path / "root"
    env = _env(tmp_path, QUIET)
    st = idle_since(root, env, 30)
    assert st["verdict"] == "idle" and st["idle_min"] == 30.0 and st["idle_min_limit"] == 90
    st = tick(root, _env(tmp_path, QUIET + ["python x.py"]))
    assert st["verdict"] == "busy" and st["idle_min"] == 0


def test_idle_past_the_limit_refuses_while_a_run_is_unfetched_and_names_it(tmp_path):
    root = tmp_path / "root"
    run_dir(root, "r-fetched")
    mark_fetched(root / "runs" / "r-fetched")
    run_dir(root, "r-unfetched")
    st = idle_since(root, _env(tmp_path, QUIET), 91)
    assert st["verdict"] == "refuse" and [u["run"] for u in st["unfetched"]] == ["r-unfetched"]
    assert "r-unfetched" in st["refusal"] and "research fetch --all" in st["refusal"] and st["refused_since_utc"]


def test_a_run_changed_after_its_fetch_is_unfetched_again(tmp_path):
    root = tmp_path / "root"
    d = run_dir(root, "r1")
    mark_fetched(d)
    assert idle_since(root, _env(tmp_path, QUIET), 91)["verdict"] == "terminate"
    time.sleep(0.05)
    (d / "late.log").write_text("written after the verification began")
    st = tick(root, _env(tmp_path, QUIET))
    assert st["verdict"] == "refuse" and "changed after its fetch (late.log)" in st["unfetched"][0]["why"]


def test_every_run_fetched_and_idle_past_the_limit_terminates(tmp_path):
    root = tmp_path / "root"
    for rid in ("a", "b"):
        mark_fetched(run_dir(root, rid))
    st = idle_since(root, _env(tmp_path, QUIET), 90)
    assert st["verdict"] == "terminate" and st["unfetched"] == [] and st["runs"] == 2 and st["pod_id"] == POD


def test_no_runs_at_all_is_nothing_to_protect(tmp_path):
    assert idle_since(tmp_path / "root", _env(tmp_path, QUIET), 95)["verdict"] == "terminate"


# ---- the daemon ------------------------------------------------------------------------------------------------------------------------

def _daemon(root: Path, env: dict[str, str], idle_min: float) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(GUARD), "daemon", str(root), str(idle_min)], env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait(pred, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _state(root: Path) -> dict:
    try:
        return json.loads((root / "guard" / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def test_daemon_refuses_while_unfetched_then_terminates_after_the_fetch_and_exits(tmp_path):
    root = tmp_path / "root"
    d = run_dir(root, "r-e2e")
    env = _env(tmp_path, QUIET, GUARD_TICK_S="0.2", GUARD_RETRY_S="0.2")
    proc = _daemon(root, env, 0.01)                 # 0.6 s idle limit
    try:
        assert _wait(lambda: _state(root).get("verdict") == "refuse"), _state(root)
        time.sleep(0.6)
        assert not (tmp_path / "terminated").exists(), "refused: nothing terminated while r-e2e is unfetched"
        log = (root / "guard" / "guard.log").read_text()
        assert "VERDICT refuse" in log and "unfetched=r-e2e" in log and log.count("VERDICT refuse") == 1, "logged once per change, not per tick"
        mark_fetched(d)
        assert _wait(lambda: (tmp_path / "terminated").exists())
        assert proc.wait(timeout=20) == 0, "the daemon exits once termination is accepted"
        assert (tmp_path / "terminated").read_text().split() == [POD]
        st = _state(root)
        assert st["verdict"] == "terminated" and st["terminate"]["ok"] and st["terminate"]["rc"] == 0
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


def test_a_second_daemon_on_the_same_pod_exits_and_the_first_keeps_guarding(tmp_path):
    root = tmp_path / "root"
    env = _env(tmp_path, QUIET + ["python busy.py"], GUARD_TICK_S="0.2")
    first = _daemon(root, env, 90)
    try:
        assert _wait(lambda: (root / "guard" / "guard.pid").exists() and _state(root).get("verdict") == "busy")
        second = _daemon(root, env, 90)
        assert second.wait(timeout=20) == 0 and first.poll() is None
        assert int((root / "guard" / "guard.pid").read_text()) == first.pid
        assert "another guard holds" in (root / "guard" / "guard.log").read_text()
    finally:
        os.killpg(first.pid, signal.SIGKILL)


def test_the_daemon_never_uses_the_account_key_path(tmp_path):
    """No pod-scoped key (and no GUARD_TERMINATE_CMD): guard_terminate_self fails instead of reaching for /root/dm/runpod.py."""
    env = _env(tmp_path, QUIET, key=False)
    env.pop("GUARD_TERMINATE_CMD")
    r = subprocess.run(["bash", "-c", f'. "{GUARD}"; GUARD_POD_KEY_ONLY=1 GUARD_LOGF="$0" guard_terminate_self test; echo rc=$?', str(tmp_path / "g.log")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "rc=3" in r.stdout and "no pod-scoped RUNPOD_API_KEY" in (tmp_path / "g.log").read_text()


# ---- the real terminate path, against a fake RunPod API ----------------------------------------------------------------------------------

class FakeRunPod:
    """REST DELETE /pods/{id} answers `rest_status`; POST /graphql answers 200 with `gql_body`.  Records (method, path, auth, body)."""

    def __init__(self, rest_status: int = 200, gql_body: str = '{"data": {"podTerminate": null}}'):
        import http.server
        import threading

        self.calls: list[tuple[str, str, str, str]] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def _answer(self, status: int, body: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                outer.calls.append((self.command, self.path, self.headers.get("Authorization", ""), self.rfile.read(n).decode()))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def do_DELETE(self):  # noqa: N802
                self._answer(rest_status, "{}" if rest_status == 200 else '{"error": "forbidden"}')

            def do_POST(self):  # noqa: N802
                self._answer(200, gql_body)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def env(self) -> dict[str, str]:
        return {"GUARD_API_REST": f"{self.url}/v1", "GUARD_API_GQL": f"{self.url}/graphql", "PATH": "/usr/bin:/bin"}   # no runpodctl


def _terminate(tmp_path: Path, api: FakeRunPod, key: bool = True) -> subprocess.CompletedProcess:
    env = {**_env(tmp_path, QUIET, key=key), **api.env()}
    env.pop("GUARD_TERMINATE_CMD")
    import sys

    env["GUARD_PY"] = sys.executable
    return subprocess.run(["bash", "-c", f'. "{GUARD}"; guard_terminate_pod {POD}; echo rc=$?'], env=env, capture_output=True, text=True, timeout=60)


def test_terminate_uses_the_pod_key_from_the_environ_file_over_rest(tmp_path):
    api = FakeRunPod()
    r = _terminate(tmp_path, api)
    assert "rc=0" in r.stdout and f"TERMINATED {POD} (REST DELETE" in r.stdout, r.stdout + r.stderr
    assert api.calls == [("DELETE", f"/v1/pods/{POD}", "Bearer pod-scoped-secret", "")]


def test_terminate_falls_back_to_graphql_when_rest_refuses(tmp_path):
    api = FakeRunPod(rest_status=403)
    r = _terminate(tmp_path, api)
    assert "rc=0" in r.stdout and "GraphQL podTerminate" in r.stdout and "REST DELETE HTTP 403" in r.stdout, r.stdout + r.stderr
    assert [c[0] for c in api.calls] == ["DELETE", "POST"] and f'podTerminate(input: {{podId: \\"{POD}\\"}})' in api.calls[1][3]
    assert all(c[2] == "Bearer pod-scoped-secret" for c in api.calls)


def test_terminate_fails_when_both_apis_refuse_and_without_a_key_calls_nothing(tmp_path):
    api = FakeRunPod(rest_status=403, gql_body='{"errors": [{"message": "Unauthorized"}]}')
    r = _terminate(tmp_path, api)
    assert "rc=1" in r.stdout and "Unauthorized" in r.stdout
    api2 = FakeRunPod()
    r = _terminate(tmp_path, api2, key=False)
    assert "rc=3" in r.stdout and api2.calls == []


def test_daemon_terminates_through_the_real_path_with_the_pod_key(tmp_path):
    """The 2026-09-24 pod run: with GUARD_TERMINATE_CMD unset the daemon must reach the API with the key from the environ file."""
    import sys

    api = FakeRunPod()
    root = tmp_path / "root"
    mark_fetched(run_dir(root, "r-done"))
    env = {**_env(tmp_path, QUIET, GUARD_TICK_S="0.2", GUARD_RETRY_S="0.2"), **api.env(), "GUARD_PY": sys.executable}
    env.pop("GUARD_TERMINATE_CMD")
    proc = _daemon(root, env, 0.01)
    try:
        assert proc.wait(timeout=30) == 0
        assert api.calls[:1] == [("DELETE", f"/v1/pods/{POD}", "Bearer pod-scoped-secret", "")]
        st = _state(root)
        assert st["verdict"] == "terminated" and st["terminate"]["ok"] and "REST DELETE" in st["terminate"]["output"]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


# ---- ensure_guard (the machine side of `research run --on`) ------------------------------------------------------------------------------

def test_no_pod_scoped_key_means_no_guard_and_says_so(tmp_path, monkeypatch):
    env = _env(tmp_path, QUIET, key=False)
    out = remote.ensure_guard(tmp_path / "root", 90, pod_id=POD, environ=env["GUARD_ENVIRON"])
    assert out["installed"] is False and "no pod-scoped RUNPOD_API_KEY" in out["reason"] and "NOT installed" in out["reason"]
    assert not (tmp_path / "root" / "guard" / "guard.pid").exists()


def test_not_a_runpod_pod_or_another_pod_means_no_guard(tmp_path):
    (tmp_path / "environ").write_bytes(b"HOME=/root\0")
    assert "not a RunPod pod" in remote.ensure_guard(tmp_path / "root", 90, environ=str(tmp_path / "environ"))["reason"]
    env = _env(tmp_path, QUIET)
    out = remote.ensure_guard(tmp_path / "root", 90, pod_id="someotherpod", environ=env["GUARD_ENVIRON"])
    assert out["installed"] is False and "someotherpod" in out["reason"]


def test_ensure_guard_starts_one_daemon_and_finds_it_the_next_time(tmp_path, monkeypatch):
    env = _env(tmp_path, QUIET + ["python busy.py"])
    for k in ("GUARD_PS", "GUARD_NVSMI", "GUARD_TERMINATE_CMD"):
        monkeypatch.setenv(k, env[k])
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-reach-the-daemon")
    root = tmp_path / "root"
    first = remote.ensure_guard(root, 90, pod_id=POD, environ=env["GUARD_ENVIRON"])
    try:
        assert first["installed"] and first["started"] and first["pid"] and first["state"] == str(root / "guard" / "state.json")
        again = remote.ensure_guard(root, 90, pod_id=POD, environ=env["GUARD_ENVIRON"])
        assert again["installed"] and not again["started"] and again["pid"] == first["pid"]
        assert _wait(lambda: _state(root).get("verdict") == "busy")
        daemon_env = subprocess.run(["ps", "eww", "-o", "command=", "-p", str(first["pid"])], capture_output=True, text=True).stdout
        assert "must-not-reach-the-daemon" not in daemon_env
    finally:
        os.killpg(first["pid"], signal.SIGKILL)


def test_launch_request_carries_the_guard_answer_and_a_request_without_guard_starts_none(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_launch_request", lambda p: {"claimed": True, "started": True})
    seen = []
    monkeypatch.setattr(remote, "ensure_guard", lambda root, idle, pod_id=None: seen.append((root, idle, pod_id)) or {"installed": False, "reason": "x"})
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"runs_root": "/workspace/research/runs", "guard": {"idle_min": 3}, "remote": {"pod_id": POD}}))
    assert remote.launch_request(req)["guard"] == {"installed": False, "reason": "x"}
    assert seen == [(Path("/workspace/research"), 3.0, POD)]
    req.write_text(json.dumps({"runs_root": "/workspace/research/runs", "guard": None}))
    assert "guard" not in remote.launch_request(req) and len(seen) == 1


def test_guard_is_opt_in_per_machine():
    assert remote.guard_idle_min({"name": "vy-control"}) is None, "a machine without `guard` (the control pod) never gets one"
    assert remote.guard_idle_min({"guard": False}) is None
    assert remote.guard_idle_min({"guard": True}) == 90.0 and remote.guard_idle_min({"guard": 3}) == 3.0
    with pytest.raises(remote.Refused):
        remote.guard_idle_min({"guard": "yes"})


# ---- fetch --all writes .fetched on the machine ----------------------------------------------------------------------------------------

def _fake_machine(tmp_path: Path, monkeypatch, files: dict[str, bytes]) -> tuple[Path, Path]:
    root = tmp_path / "machine"
    d = run_dir(root, "r-fetch")
    for name, body in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(body)
    old = time.time() - 60
    for p in d.rglob("*"):
        os.utime(p, (old, old))
    local = tmp_path / "local" / "r-fetch"
    local.mkdir(parents=True)
    (local / "remote.json").write_text(json.dumps({"run_id": "r-fetch", "host": "h", "port": 22, "remote_run_dir": str(d)}))
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    return d, local


def test_fetch_all_verifies_then_writes_the_marker_the_guard_accepts(tmp_path, monkeypatch):
    d, local = _fake_machine(tmp_path, monkeypatch, {"out/proof.bin": b"p" * 1000, "events.jsonl": b"{}\n"})
    rec = remote.fetch(local, everything=True)
    assert rec["preserved"] and rec["fetched_marker"]["written"], rec
    marker = json.loads((d / ".fetched").read_text())
    assert marker["schema"] == remote.FETCHED_SCHEMA and marker["files"] == 5 and marker["verified_from"] <= time.time()
    assert (local / "out" / "proof.bin").read_bytes() == b"p" * 1000 and not (local / ".fetched").exists()
    assert json.loads((local / "preserved.json").read_text())["verified_from_machine_epoch"] == marker["verified_from"]
    env = _env(tmp_path, QUIET)
    assert idle_since(d.parent.parent, env, 91)["verdict"] == "terminate"
    rec = remote.fetch(local, everything=True)
    assert rec["preserved"] and rec["fetched_marker"]["written"], "fetching again verifies again (the marker is not part of the copy)"


def test_fetch_all_with_a_mismatch_writes_no_marker(tmp_path, monkeypatch):
    d, local = _fake_machine(tmp_path, monkeypatch, {"a.txt": b"a"})
    real = remote._file_sha256
    monkeypatch.setattr(remote, "_file_sha256", lambda p: "0" * 64 if p.name == "a.txt" else real(p))
    rec = remote.fetch(local, everything=True)
    assert rec["preserved"] is False and "fetched_marker" not in rec and not (d / ".fetched").exists()
    assert rec["preserve_error"]["mismatched_or_missing"] == ["a.txt"]


def test_fetch_all_streams_the_run_instead_of_holding_it(tmp_path, monkeypatch):
    blob = os.urandom(1 << 20) * 24                 # 24 MiB
    d, local = _fake_machine(tmp_path, monkeypatch, {"big.bin": blob})
    tracemalloc.start()
    try:
        rec = remote.fetch(local, everything=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert rec["preserved"] and (local / "big.bin").stat().st_size == len(blob)
    assert peak < 6 * (1 << 20), f"fetch --all held {peak / 2**20:.1f} MiB of a 24 MiB run in memory"
