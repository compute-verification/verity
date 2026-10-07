"""The write probe (`pods/nebius/write_probe.py`): which /workspace writers an allocation covers, on a synthetic /proc."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
_spec = importlib.util.spec_from_file_location("nebius_write_probe", NEBIUS / "write_probe.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

MB = 10**6
POD = "6a323904-1111-2222-3333-444455556666"


def proc_entry(proc: Path, pid: int, *, ppid: int = 1, start: int = 1000, written: int = 0, cgroup: str = "/user.slice",
               cwd: str = "/home/research", writes: tuple[str, ...] = (), reads: tuple[str, ...] = (), env: dict | None = None,
               cmd: str = "python3 job.py") -> None:
    d = proc / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "fdinfo").mkdir()
    (d / "stat").write_text(f"{pid} (py (thon)) S {ppid} " + " ".join(["0"] * 17) + f" {start} 0 0\n")
    (d / "io").write_text(f"rchar: 1\nwchar: {written}\nread_bytes: 0\nwrite_bytes: {written}\ncancelled_write_bytes: 0\n")
    (d / "task" / str(pid)).mkdir(parents=True)
    (d / "task" / str(pid) / "io").write_text(f"write_bytes: {written}\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "status").write_text("Name:\tpython3\nUid:\t0\t0\t0\t0\n")
    (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
    (d / "environ").write_bytes(b"".join(f"{k}={v}\0".encode() for k, v in (env or {}).items()))
    os.symlink(cwd, d / "cwd")
    for i, (path, flags) in enumerate([(p, "0100001") for p in writes] + [(p, "0100000") for p in reads]):
        os.symlink(path, d / "fd" / str(i + 3))
        (d / "fdinfo" / str(i + 3)).write_text(f"pos:\t0\nflags:\t{flags}\nmnt_id:\t30\n")


def set_written(proc: Path, pid: int, written: int, tid: int | None = None) -> None:
    t = proc / str(pid) / "task" / str(tid or pid)
    t.mkdir(parents=True, exist_ok=True)
    (t / "io").write_text(f"write_bytes: {written}\n")


def test_each_allocation_is_named_and_only_uncovered_workspace_writers_are_unmatched(tmp_path):
    proc = tmp_path / "proc"
    proc_entry(proc, 100, cgroup="/user.slice/user-1001.slice/session-17.scope", writes=("/workspace/scratch/big.bin",))
    proc_entry(proc, 200, cgroup=f"/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod{POD.replace('-', '_')}.slice/cri-containerd-ab.scope",
               writes=("/workspace/cp/row/bundle.npz",))
    proc_entry(proc, 300, cgroup="/user.slice/user-1001.slice/user@1001.service/app.slice/fill-verity-179.scope", cwd="/workspace/jobs/x")
    proc_entry(proc, 401, env={"RESEARCH_RUN_ID": "r20261001-030000-abcd"})
    proc_entry(proc, 400, ppid=401, cgroup="/user.slice/user-1001.slice/session-9.scope", writes=("/workspace/research/runs/r/out",))
    proc_entry(proc, 450, cgroup="/user.slice/user-1001.slice/user@1001.service/app.slice/cluster-r20261001-022442-80e9.scope",
               cwd="/workspace/research/runs/r20261001-022442-80e9")
    proc_entry(proc, 500, cgroup="/user.slice/user-1001.slice/session-17.scope", writes=("/tmp/elsewhere",), reads=("/workspace/in",))
    proc_entry(proc, 600, cgroup="/user.slice/user-1001.slice/session-17.scope", writes=("/workspace/slow",))
    proc_entry(proc, 700, cgroup="/system.slice/vy-cluster-agent.service", cwd="/workspace/cluster")
    proc_entry(proc, 800, cgroup="/user.slice/user-1001.slice/session-18653.scope", writes=("/workspace/research/store/catalog.sqlite",),
               cmd="python3 -m research run --runs-dir /workspace/research/runs --id r20261001-040538-950d --claimed")
    before = W.snapshot(str(proc))
    for pid in (100, 200, 300, 400, 450, 500, 700, 800):
        set_written(proc, pid, 1200 * MB)
    set_written(proc, 600, 500 * MB)
    rows = W.writers(before, W.snapshot(str(proc)), 60.0, 10.0, lambda: {POD: ("default", "nd-commit-0-x", "deployments-gpu")},
                     "vy-nebius-1", str(proc))
    got = {r["pid"]: (r["allocation"], r["matched"]) for r in rows}
    assert got == {
        100: ("none:/user.slice/user-1001.slice/session-17.scope", False),
        200: ("kueue:deployments-gpu/nd-commit-0-x", True),
        300: ("fill:fill-verity-179", True),
        400: ("run:r20261001-030000-abcd", True),
        450: ("run:r20261001-022442-80e9", True),
        700: ("system:vy-cluster-agent.service", True),
        800: ("run:r20261001-040538-950d", True),
    }
    r100 = next(r for r in rows if r["pid"] == 100)
    assert (r100["mb_per_s"], r100["paths"], r100["user"], r100["cmd"]) == (20.0, ["/workspace/scratch/big.bin"], "root", "python3 job.py")
    assert next(r for r in rows if r["pid"] == 300)["paths"] == ["/workspace/jobs/x"]


def test_a_pod_without_kueue_admission_or_unknown_to_kubectl_is_unmatched_and_kubectl_is_read_once(tmp_path):
    proc = tmp_path / "proc"
    other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    for pid, uid in ((10, POD), (11, other), (12, POD.replace("6a32", "7b43"))):
        proc_entry(proc, pid, cgroup=f"/kubepods/besteffort/pod{uid}/c1", writes=("/workspace/x",))
    before = W.snapshot(str(proc))
    for pid in (10, 11, 12):
        set_written(proc, pid, 900 * MB)
    calls = []

    def pods():
        calls.append(1)
        return {POD: ("default", "sky-jobs-controller", None), other: ("gpu-operator", "dcgm-exporter", None)}

    got = {r["pid"]: (r["allocation"], r["matched"]) for r in W.writers(before, W.snapshot(str(proc)), 60.0, 10.0, pods, "n1", str(proc))}
    assert got == {10: ("pod:default/sky-jobs-controller (no Kueue admission)", False),
                   11: ("system:gpu-operator/dcgm-exporter", True),
                   12: (f"pod:{POD.replace('6a32', '7b43')} (not in kubectl's list)", False)}
    assert calls == [1]


def test_a_runner_is_not_charged_for_the_children_it_reaped_and_its_jobs_are_fill_jobs(tmp_path):
    """/proc/<pid>/io adds each reaped child's bytes to the parent's: node 2's fill runner alerted at 26.7 MB/s while its
    log stayed empty. A CPU job without a named unit runs in an anonymous run-*.scope, so its lineage names it."""
    proc = tmp_path / "proc"
    proc_entry(proc, 50, cgroup="/user.slice/user-1001.slice/session-65.scope", writes=("/workspace/pouw/infra/logs/fill.err",),
               cmd="python3 /workspace/pouw/infra/bin/fill_runner.py")
    job = "/user.slice/user-1001.slice/user@1001.service/app.slice/run-rd021988fb33f4e9f89ecbd41e20cc1ef.scope"
    proc_entry(proc, 60, ppid=50, cgroup=job, cwd="/workspace/pouw/fill", cmd="bash /workspace/pouw/fill/running/fp4-gc-judge-388eeb55.sh")
    proc_entry(proc, 61, ppid=60, cgroup=job, writes=("/workspace/pouw/gpu7-fp4/kt/out/census.jsonl",), cmd="python census.py run")
    before = W.snapshot(str(proc))
    (proc / "50" / "io").write_text(f"write_bytes: {5000 * MB}\n")
    set_written(proc, 61, 1200 * MB)
    set_written(proc, 61, 600 * MB, tid=62)
    got = {r["pid"]: (r["allocation"], r["matched"], r["mb_per_s"])
           for r in W.writers(before, W.snapshot(str(proc)), 60.0, 10.0, dict, "vy-nebius-2", str(proc))}
    assert got == {61: ("fill:fp4-gc-judge-388eeb55", True, 30.0)}
    before = W.snapshot(str(proc))
    set_written(proc, 50, 1200 * MB)
    got = {r["pid"]: (r["allocation"], r["matched"])
           for r in W.writers(before, W.snapshot(str(proc)), 60.0, 10.0, dict, "vy-nebius-2", str(proc))}
    assert got == {50: ("fill:runner", True)}


def test_a_reused_pid_is_a_new_process_and_a_timed_window_pauses_the_probe(tmp_path):
    proc = tmp_path / "proc"
    proc_entry(proc, 100, start=1000, writes=("/workspace/x",))
    before = W.snapshot(str(proc))
    (proc / "100" / "stat").write_text("100 (python3) S 1 " + " ".join(["0"] * 17) + " 2000 0 0\n")
    set_written(proc, 100, 5000 * MB)
    assert W.writers(before, W.snapshot(str(proc)), 60.0, 10.0, dict, "n2", str(proc)) == []
    status = tmp_path / "status.txt"
    status.write_text("2026-10-01T03:47:17Z free GPUs 7/8, kept free -, waiters -, timed False, window waiting False, backfill False\n")
    assert not W.window_running(str(status))
    status.write_text("2026-10-01T03:47:17Z free GPUs 0/8, kept free 8, waiters -, timed True, window waiting False, backfill False\n")
    assert W.window_running(str(status))
    status.write_text("2026-10-01T03:47:17Z free GPUs 2/8, kept free 8, waiters -, timed False, window waiting True, backfill False\n")
    assert W.window_running(str(status))
    assert not W.window_running(str(tmp_path / "missing"))


def test_the_relay_keeps_its_place_through_an_unmounted_log_and_a_failed_post(tmp_path, monkeypatch):
    """Node 2's quota cutover (2026-10-01): while /workspace was unmounted the log read as missing, the relay reset its cursor to 0,
    and after the remount it posted hours-old rows; a Slack error after 5 posts lost the state, so every pass repeated them."""
    import argparse
    import inspect
    import json
    import re
    import subprocess

    from verity_ops import slack

    rows = [json.dumps({"key": f"k{i}", "node": "vy-nebius-2", "ts": f"t{i}", "mb_per_s": 20, "pid": i, "user": "research",
                        "cmd": "python3 job.py", "paths": ["/workspace/x"], "cgroup": "/user.slice", "allocation": "none"})
            for i in range(4)]
    log: dict[str, list[str] | None] = {"n2": rows[:2]}
    posted: list[str] = []
    fail_on: set[str] = set()

    def fake_run(argv, capture_output, text):
        lines = log[argv[-2]]
        if lines is None:   # what the remote `[ -f log ] || { echo ...; }` prints
            return subprocess.CompletedProcess(argv, 0, re.search(r"echo (\w+);", argv[-1]).group(1) + "\n", "")
        seen = int(re.search(r"tail -n \+(\d+)", argv[-1]).group(1)) - 1
        return subprocess.CompletedProcess(argv, 0, "\n".join([str(len(lines)), *lines[seen:]]) + "\n", "")

    real_post = inspect.signature(slack._post)

    def fake_post(*a, **k):
        args = real_post.bind(*a, **k).arguments   # the real signature, so a call that drifts from it fails here
        assert args.get("thread") is None, "an alert is a top-level post"
        ts = re.search(r"\((t\d)\)", args["text"]).group(1)
        if ts in fail_on:
            raise slack.SlackError("the Cursor identity socket the Slack broker needs is unavailable")
        posted.append(ts)

    monkeypatch.setattr(W.subprocess, "run", fake_run)
    monkeypatch.setattr(slack, "registry_path", lambda: None)
    monkeypatch.setattr(slack, "load_registry", lambda p: {})
    monkeypatch.setattr(slack, "channel_id_for", lambda reg, key: "C1")
    monkeypatch.setattr(slack, "Slack", lambda *a, **k: None)
    monkeypatch.setattr(slack, "_post", fake_post)
    state = tmp_path / "relay.json"
    ns = argparse.Namespace(nodes=["n2"], out="/workspace/research/write-probe", every=0, state=str(state), handle="infra",
                            dry_run=False, once=True)

    W.relay(ns)
    assert posted == ["t0", "t1"]
    log["n2"] = None                                   # unmounted
    W.relay(ns)
    assert json.loads(state.read_text())["lines"]["n2"] == 2
    log["n2"] = rows                                   # remounted, two new rows; the second post fails
    fail_on.add("t3")
    try:
        W.relay(ns)
    except slack.SlackError:
        pass
    assert posted == ["t0", "t1", "t2"]
    fail_on.clear()
    W.relay(ns)
    assert posted == ["t0", "t1", "t2", "t3"]
