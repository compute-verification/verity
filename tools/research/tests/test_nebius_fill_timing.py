"""Node 2's fill runner gives network accounting's timing chunks (`timing=1`) a GPU on NUMA 1 and whole cores of 124-191 of their
own, at nice 0, in a top-level slice, and never stops them; node_ops' Quiet never pauses them.  Their chunks kept 0 of 353 windows
on the shared 48-95 at nice 19 (art:88c665c3).  The node's sysfs and cgroups are files under tmp_path, laid out as node 2's."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from research.pods import registry

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
FILL_RUNNER, NODE_OPS = NEBIUS / "fill_runner.py", NEBIUS / "node_ops.py"
GPU_LEASE = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
NUMA1 = set(range(96, 192))


def cgroup(root: Path, name: str, cpus: str, populated: bool, own: str = "") -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "cgroup.events").write_text(f"populated {int(populated)}\nfrozen 0\n")
    (d / "cpuset.cpus").write_text(own + "\n")
    (d / "cpuset.cpus.effective").write_text(cpus + "\n")


@pytest.fixture
def node2(tmp_path, monkeypatch):
    """Node 2 as fill sees it: 192 CPUs in adjacent SMT pairs, NUMA 0 = 0-95 with GPUs 0-3 and NUMA 1 = 96-191 with GPUs 4-7,
    user.slice and system.slice confined to 0-123, PID 1's init.scope on every CPU, and pous.slice and provers.slice empty."""
    for c in range(192):
        t = tmp_path / "cpu" / f"cpu{c}" / "topology"
        t.mkdir(parents=True)
        (t / "thread_siblings_list").write_text(f"{c & ~1}-{c | 1}\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "cgroup.procs").write_text("")
    for name, cpus, populated, own in (("user.slice", "0-123", True, "0-123"), ("system.slice", "0-123", True, "0-123"),
                                       ("init.scope", "0-191", True, ""), ("pous.slice", "0-191", False, ""),
                                       ("provers.slice", "0-191", False, "")):
        cgroup(cg, name, cpus, populated, own)
    fill, gl = tmp_path / "fill", tmp_path / "gl"
    for d in ("queue", "running", "logs", "done", "failed"):
        (fill / d).mkdir(parents=True)
    (fill / "timing-owners").write_text("net  # network accounting's chunks\n")
    gl.mkdir()
    for k, v in {"FILL_DIR": fill, "GPU_LEASE_DIR": gl, "FILL_CGROUP_ROOT": cg, "FILL_CPU_ROOT": tmp_path / "cpu",
                 "FILL_YIELD_LOCK": tmp_path / "check-d.lock", "FILL_CPU_CLAIMS_DIR": tmp_path / "claims"}.items():
        monkeypatch.setenv(k, str(v))

    def load():
        spec = importlib.util.spec_from_file_location("fill_runner_timing", FILL_RUNNER)
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        monkeypatch.setattr(fr, "gpu_cpus", lambda i: set(range(96)) if i < 4 else set(NUMA1))
        return fr

    return load, fill, cg


def cpus_of(text: str) -> set[int]:
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def whole_cores(cpus: set[int]) -> bool:
    return all(c ^ 1 in cpus for c in cpus)


def fake_node(fr, monkeypatch, fill: Path):
    """tick() against leases, waiters and starts the test holds: a start records (job, GPU, its timing CPUs) and runs the job."""
    held: dict[int, str] = {}
    want: list[int] = []
    started, stopped = [], []
    monkeypatch.setattr(fr, "reap", lambda: None)
    monkeypatch.setattr(fr, "verity_pool", lambda queued: None)
    monkeypatch.setattr(fr, "lease", lambda i: held.get(i))
    monkeypatch.setattr(fr, "waiters", lambda: list(want))
    monkeypatch.setattr(fr, "repin", lambda f, fs: None)
    monkeypatch.setattr(fr, "active_s", lambda j: 0.0)
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: stopped.append((k, why)) or fr.running[k].update(stopping=True,
                                                                                                                   why=why))

    def start(p, h, backfill=False):
        started.append((p.name, h["on"], h.get("timing_cpus")))
        p.unlink()
        run(p.name, int(h["on"]), h)

    def run(name, gpu, h):
        held[gpu] = f"pid={100 + gpu} run=- who=fill:{h['owner']} since=2026-10-03T20:00:00Z cmd=x"
        fr.running[name] = {"pid": 100 + gpu, "t0": len(fr.running), "paused": False, "started": "-", "h": h}

    monkeypatch.setattr(fr, "start", start)
    return held, want, started, stopped, run


def queue(fill: Path, name: str, head: str) -> None:
    (fill / "queue" / name).write_text(f"#!/bin/bash\n# fill: {head}\n")


TIMING = "owner=net gpus=1 timing=1 cpus=16 max_min=60 mem_gb=48 project=verity"


def test_a_timing_job_gets_a_numa1_gpu_and_whole_cores_of_124_191_alone(node2, monkeypatch):
    """Four chunks at most: one per NUMA 1 GPU, each on 16 threads of 124-191, 8 whole SMT pairs no other job has, local to its
    GPU; the fifth waits, since 188-191 hold only 4. A pous job queued beside them takes a NUMA 0 GPU, which a chunk can't use,
    rather than the highest free one."""
    load, fill, _ = node2
    fr = load()
    queue(fill, "h.sh", "owner=net gpus=0 timing=1 cpus=7 max_min=90 project=verity")
    h = fr.header(fill / "queue" / "h.sh")
    assert (h["timing"], h["gpus"], h["cpus"], h["max_min"]) == (True, 1, 7, 60)
    queue(fill, "h.sh", "owner=net timing=1 cpus=40 max_min=50")
    assert fr.header(fill / "queue" / "h.sh")["cpus"] == 16
    queue(fill, "h.sh", "owner=net timing=yes max_min=60")
    assert fr.header(fill / "queue" / "h.sh")["timing"] is False and fr.header(fill / "queue" / "h.sh")["max_min"] == 30
    (fill / "queue" / "h.sh").unlink()

    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    for k in range(5):
        queue(fill, f"nw-chunk-{k}.sh", TIMING)
    queue(fill, "pous-a.sh", "owner=pm gpus=1 max_min=30")
    fr.tick(8)
    assert started == [("pous-a.sh", "3", None)]                     # pous ranks first; it leaves NUMA 1 to the chunks
    fr.tick(8)
    chunks = started[1:]
    assert [(name, gpu) for name, gpu, _ in chunks] == [(f"nw-chunk-{k}.sh", g) for k, g in enumerate("7654")]
    sets = [cpus_of(c) for _, _, c in chunks]
    assert all(len(s) == 16 and s <= set(range(124, 192)) and whole_cores(s) for s in sets), sets
    assert len(set().union(*sets)) == 64                              # no CPU in two chunks
    assert [c for _, _, c in chunks] == ["124-139", "140-155", "156-171", "172-187"]
    assert (fill / "queue" / "nw-chunk-4.sh").exists() and stopped == []
    assert fr.timing_free([]) == {188, 189, 190, 191} and fr.carve({188, 189, 190, 191}, NUMA1, 16) is None
    status = (fill / "status.txt").read_text()
    assert "running nw-chunk-0.sh owner=net gpus=1 since=- timing cpus=124-139" in status

    fr.running.pop("nw-chunk-1.sh"), held.pop(6)                     # a chunk ends: the next takes its GPU and cores
    fr.tick(8)
    assert started[-1] == ("nw-chunk-4.sh", "6", "140-155")


def test_other_jobs_leave_only_the_gpus_a_queued_timing_job_could_start_on(node2, monkeypatch):
    """Steering is for a chunk that can start: while user.slice reaches past 123 (after a reboot, before the deploy's
    set-property) no chunk can, and a pous job takes the highest free GPU as usual; it used to take NUMA 0's while a chunk sat
    queued with nowhere to go. A chunk pinned to GPU 6 leaves only GPU 6 to the rest last."""
    load, fill, cg = node2
    fr = load()
    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    (cg / "user.slice" / "cpuset.cpus.effective").write_text("0-191\n")
    queue(fill, "nw-chunk.sh", TIMING)
    queue(fill, "pous-a.sh", "owner=pm gpus=1 max_min=30")
    fr.tick(8)
    assert started == [("pous-a.sh", "7", None)] and (fill / "queue" / "nw-chunk.sh").exists()
    (cg / "user.slice" / "cpuset.cpus.effective").write_text("0-123\n")
    fr.running.clear(), held.clear(), started.clear()
    queue(fill, "nw-chunk.sh", TIMING + " on=6")
    queue(fill, "pous-b.sh", "owner=pm gpus=1 max_min=30")
    fr.tick(8)
    fr.tick(8)
    assert started == [("pous-b.sh", "7", None), ("nw-chunk.sh", "6", "124-139")]


def test_a_malformed_header_files_its_job_under_failed_and_the_tick_goes_on(node2, monkeypatch):
    """A value fill reads as a whole number that isn't one (gpus, max_min, prio; a CPU or timing job's mem_gb; a timing job's cpus)
    sends that job to failed/ with an event, and the rest start. Its ValueError used to end every tick before any start, and a
    running job's ended the runner at adoption. A GPU job's mem_gb, which fill never reads, is no reason."""
    load, fill, _ = node2
    fr = load()
    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    events = []
    monkeypatch.setattr(fr, "event", lambda kind, job, **kw: events.append((kind, job, kw.get("why"))))
    queue(fill, "pous-a.sh", "owner=pm gpus=1 max_min=30")
    queue(fill, "pous-m.sh", "owner=pm gpus=1 max_min=30 mem_gb=48G")
    bad = {"bad-gpus.sh": "owner=pm gpus=one", "bad-min.sh": "owner=pm max_min=1h", "bad-prio.sh": "owner=pm prio=high",
           "bad-mem.sh": "owner=pm gpus=0 mem_gb=48G", "bad-cpus.sh": "owner=net timing=1 cpus=8,9"}
    for name, head in bad.items():
        queue(fill, name, head)
    fr.tick(8)
    assert {name for name, _, _ in started} == {"pous-a.sh", "pous-m.sh"}
    assert sorted(p.name for p in (fill / "failed").iterdir()) == sorted(bad)
    whys = {job: why for kind, job, why in events if kind == "failed"}
    assert whys == {"bad-gpus.sh": "header: gpus=one", "bad-min.sh": "header: max_min=1h", "bad-prio.sh": "header: prio=high",
                    "bad-mem.sh": "header: mem_gb=48G", "bad-cpus.sh": "header: cpus=8,9"}
    assert "free GPUs" in (fill / "status.txt").read_text()

    (fill / "running" / "old.sh").write_text("#!/bin/bash\n# fill: owner=pm gpus=one\n")
    (fill / "running" / ".old.sh.pgid").write_text(f"{os.getpid()}\n")
    fr.adopt()
    assert fr.running["old.sh"]["h"]["gpus"] == 1


def test_no_exception_lifts_a_timing_job_past_its_cap_and_a_commit_is_never_one(node2):
    """Nothing stops a timing job before its max_min, so nothing lifts that past FILL_TIMING_MAX_MIN: not a FILL_DIR/max-min line
    (they used to give it the line's minutes), a fence that names it (up to 4 h) or the Commit prefix (120 min and rank 0, above
    every job). A Commit guest's timing=1 is ignored: it stays an ordinary, preemptible Commit."""
    load, fill, _ = node2
    fr = load()
    (fill / "max-min").write_text("nw-* 240 2099-01-01T00:00Z  # infra, a long sweep\n")
    queue(fill, "nw-chunk.sh", TIMING.replace("max_min=60", "max_min=240"))
    assert fr.header(fill / "queue" / "nw-chunk.sh")["max_min"] == 60
    (fill / "max-min").unlink()
    (fill / "fences").write_text("net 7 140-147 2099-01-01T00:00Z nw-* 240  # a fence naming the chunk\n")
    assert fr.header(fill / "queue" / "nw-chunk.sh", t=fr.utc("2098-12-31T00:00Z"))["max_min"] == 60
    (fill / "fences").unlink()
    queue(fill, "verity-commit-x.sh", "owner=c gpus=1 timing=1 cpus=16 max_min=120 project=verity")
    h = fr.header(fill / "queue" / "verity-commit-x.sh")
    assert (h["timing"], h["gpus"], h["max_min"]) == (False, 1, 120)


def test_a_timing_job_needs_an_owner_infra_lists_and_gets_at_most_a_gpus_memory_share(node2, monkeypatch):
    """timing=1 takes a GPU and whole cores nothing preempts, so only an owner FILL_DIR/timing-owners names (a glob) may set it;
    anyone else's job goes to failed/, and without the file everyone's does. Its scope's MemoryMax is its mem_gb up to
    gpu-lease's per-GPU share, which no other GPU job can exceed either."""
    load, fill, _ = node2
    monkeypatch.setenv("GPU_LEASE_MEM_GB_PER_GPU", "200")
    fr = load()
    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    events = []
    monkeypatch.setattr(fr, "event", lambda kind, job, **kw: events.append((kind, job, kw.get("why"))))
    (fill / "timing-owners").write_text("net  # network accounting\nnetwork-*  # its lanes\n")
    queue(fill, "pous-x.sh", "owner=pm gpus=1 timing=1 cpus=16 max_min=60 mem_gb=100000 project=pous")
    queue(fill, "nw-a.sh", TIMING.replace("mem_gb=48", "mem_gb=100000"))
    queue(fill, "nw-b.sh", TIMING.replace("owner=net", "owner=network-n3"))
    fr.tick(8)
    assert [(kind, job, why) for kind, job, why in events if kind == "failed"] == [
        ("failed", "pous-x.sh", "header: timing=1 (owner pm isn't in FILL_DIR/timing-owners)")]
    assert [name for name, _, _ in started] == ["nw-a.sh", "nw-b.sh"]
    assert (fr.running["nw-a.sh"]["h"]["mem_gb"], fr.running["nw-b.sh"]["h"]["mem_gb"]) == (200, 48)
    (fill / "timing-owners").unlink()
    queue(fill, "nw-c.sh", TIMING)
    fr.tick(8)
    assert (fill / "failed" / "nw-c.sh").exists() and len(started) == 2


def test_fill_never_stops_a_timing_job_and_a_whole_node_window_waits_for_it(node2, monkeypatch):
    """A timing chunk is not preemptible. A waiter, a timed lease, a Commit or a fence stops other fill jobs, never it; a window's
    `gpu-lease 8 --wait` waits for it to end (fill backfills meanwhile), and then stops the rest of fill. No chunk starts whose
    max_min, and the 2 min past it a chunk may hold its GPU, reach a booked window."""
    load, fill, _ = node2
    fr = load()
    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    run("nw-chunk.sh", 7, {"owner": "net", "gpus": 1, "project": "verity", "timing": True, "timing_cpus": "124-139",
                           "max_min": 10**9})
    run("pous-a.sh", 0, {"owner": "pm", "gpus": 1, "project": "pous", "max_min": 10**9})

    want[:] = [8]
    fr.tick(8)
    assert stopped == [] and "window waiting True, backfill True" in (fill / "status.txt").read_text()
    want[:] = [1] * 7
    fr.tick(8)
    assert stopped == [("pous-a.sh", "free 6 < needed 7")]
    fr.running["pous-a.sh"]["stopping"] = False
    stopped.clear()
    want.clear()
    held[1] = "pid=9 run=- who=ma since=2026-10-03T20:00:00Z timed=1 cmd=y"
    fr.tick(8)
    assert stopped == [("pous-a.sh", "timed window")]
    fr.running["pous-a.sh"]["stopping"] = False
    stopped.clear()
    held.pop(1)
    (fill / "fences").write_text("ma 7 116-123 2099-01-01T00:00Z  # a fence on the chunk's GPU\n")
    queue(fill, "verity-commit-c.sh", "owner=c gpus=1 project=verity")
    for i in range(1, 7):
        held[i] = f"pid={200 + i} run=- who=other since=2026-10-03T20:00:00Z cmd=z"
    fr.tick(8)
    assert stopped == [("pous-a.sh", "a Commit reclaims its GPU")]       # the chunk keeps GPU 7 through the fence and the Commit
    (fill / "fences").unlink()
    (fill / "queue" / "verity-commit-c.sh").unlink()
    for i in range(1, 7):
        held.pop(i)
    fr.running.pop("pous-a.sh"), held.pop(0)
    run("pous-b.sh", 0, {"owner": "pm", "gpus": 1, "project": "pous", "max_min": 10**9})
    stopped.clear()
    want[:] = [8]
    fr.tick(8)
    assert stopped == []
    fr.running.pop("nw-chunk.sh"), held.pop(7)                         # the chunk's max_min ends it: the window drains fill
    fr.tick(8)
    assert stopped == [("pous-b.sh", "timed window")]

    want.clear()
    fr.running.clear()
    held.clear()
    t = fr.utc("2026-10-03T23:00Z")
    (fill / "windows").write_text("2026-10-03T23:00Z 30 # a booked window\n")
    queue(fill, "nw-next.sh", TIMING)
    # 60 min and the 2 a chunk may overrun them (gpu-lease's kill, fill's reap of what outlives it) reach the window from 21:59Z
    monkeypatch.setattr(fr, "wall", lambda: t - 61 * 60)
    fr.tick(8)
    assert all(name != "nw-next.sh" for name, _, _ in started)
    monkeypatch.setattr(fr, "wall", lambda: t - 62 * 60)
    fr.tick(8)
    assert started[-1] == ("nw-next.sh", "7", "124-139")
    assert fr.clears({"max_min": 30}, fr.windows(), t - 30 * 60)       # a job a window stops needs no margin


def test_a_new_fence_moves_other_fill_jobs_threads_but_never_a_timing_jobs(node2, monkeypatch):
    mine = os.sched_getaffinity(0)
    if len(mine) < 2:
        pytest.skip("needs two CPUs")
    load, _, _ = node2
    fr = load()
    moves = []
    monkeypatch.setattr(fr.os, "sched_setaffinity", lambda pid, cpus: moves.append(pid))
    monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
    f = {"owner": "ma", "gpu": 7, "cpus": str(max(mine)), "jobs": None}
    for h, moved in (({"gpus": 1, "timing": True, "timing_cpus": str(max(mine))}, False), ({"gpus": 1}, True)):
        monkeypatch.setattr(fr, "running", {"job.sh": {"pid": os.getpgid(0), "unit": None, "h": h}})
        moves.clear()
        fr.repin(f, [f])
        assert bool(moves) == moved, h


def test_start_runs_a_timing_job_unpreemptible_at_nice_0_in_a_top_level_scope_and_a_restart_keeps_its_cores(node2, monkeypatch):
    """The command: vy-provers' form (`sudo -n systemd-run --scope --slice=...`) with the scope's AllowedCPUs, nice 0 and memory
    cap, then this user with this user's groups (setpriv), then gpu-lease without --preemptible or a scope of its own, inside
    the system scope, and taskset on the same CPUs; no `nice -n 19`. A stop kills the system scope through sudo. A restarted
    runner adopts the chunk with its cores, or with all of 124-191 (recorded for `fence`) when no record names them, and as a
    timing job by its unit, whatever its header or FILL_DIR/timing-owners says by then."""
    load, fill, cg = node2
    fr = load()
    cmds, calls = [], []
    monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: cmds.append((cmd, kw)) or type("P", (), {"pid": 4242})())
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(fr, "event", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(fr.os, "killpg", lambda pid, sig: None)
    queue(fill, "nw-chunk.sh", TIMING)
    p = fill / "queue" / "nw-chunk.sh"
    fr.start(p, {**fr.header(p), "on": "7", "timing_cpus": "124-139"})
    (cmd, kw), = cmds
    me = pwd.getpwuid(os.getuid())
    unit = fr.running["nw-chunk.sh"]["unit"]
    root, user = cmd[:cmd.index("setpriv")], cmd[cmd.index("setpriv"):]
    assert unit.startswith("fill-timing-") and root == [
        "sudo", "-n", "--preserve-env", "env", f"PATH={os.environ.get('PATH', '')}", f"HOME={me.pw_dir}", f"USER={me.pw_name}",
        f"LOGNAME={me.pw_name}", "systemd-run", "--quiet", "--scope", "--collect", "--slice=vy-timing.slice", "--unit", unit,
        "--nice=0", "-p", "AllowedCPUs=124-139", "-p", "MemoryMax=48G", "-p", "MemorySwapMax=0", "--"]
    assert user[:user.index("--")] == ["setpriv", f"--reuid={os.getuid()}", f"--regid={os.getgid()}", "--init-groups"]
    lease = user[user.index("--") + 1:]
    assert lease[:lease.index("--")] == [fr.GPU_LEASE, "1", "--on", "7", "--max-min", "60", "--mem-gb", "0"]
    workload = lease[lease.index("--") + 1:]
    assert workload == ["taskset", "-c", "124-139", "bash", "-c", fr.TIMING_WRAP, "fill-rc", str(fill / "running" / ".nw-chunk.sh.rc"),
                        str(fill / "running" / "nw-chunk.sh"), fr.KEEP, str(cg)]
    assert "--preemptible" not in cmd and "nice" not in cmd and kw["start_new_session"]
    assert kw["env"]["GPU_LEASE_WHO"] == "fill:net" and kw["env"]["FILL_JOB"] == "nw-chunk.sh"
    pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    assert {kw["env"][v] for v in pools} == {"8"}                    # one compute thread per core, as on node 1
    assert (fill / "running" / ".nw-chunk.sh.cpus").read_text() == "124-139\n"
    assert (fill / "running" / ".nw-chunk.sh.unit").read_text() == f"{unit}\n"
    assert any(a == ("start", "nw-chunk.sh") and k["timing_cpus"] == "124-139" for a, k in (c for c in calls if isinstance(c, tuple)))
    fr.stop("nw-chunk.sh", "max_min")
    assert ["sudo", "-n", "systemctl", "kill", "--signal=SIGTERM", f"{unit}.scope"] in calls
    assert not any(c[:2] == ["systemctl", "--user"] for c in calls if isinstance(c, list))

    pgid = os.getpid()                                                # a live process group stands in for the chunk's
    (fill / "running" / ".nw-chunk.sh.pgid").write_text(f"{pgid}\n")
    fr2 = load()
    monkeypatch.setattr(fr2, "event", lambda *a, **kw: None)
    fr2.adopt()
    j = fr2.running["nw-chunk.sh"]
    assert (j["unit"], j["h"]["timing_cpus"]) == (unit, "124-139")
    assert fr2.timing_free([]) == set(range(140, 192))
    (fill / "running" / ".nw-chunk.sh.cpus").unlink()
    fr3 = load()
    monkeypatch.setattr(fr3, "event", lambda *a, **kw: None)
    fr3.adopt()
    assert fr3.running["nw-chunk.sh"]["h"]["timing_cpus"] == "124-191" and fr3.timing_free([]) == set()
    assert (fill / "running" / ".nw-chunk.sh.cpus").read_text() == "124-191\n"      # which `fence` reads
    (fill / "timing-owners").unlink()                                 # nor does a later edit make it preemptible
    (fill / "running" / "nw-chunk.sh").write_text("#!/bin/bash\n# fill: owner=net gpus=1 max_min=60 project=verity\n")
    fr4 = load()
    monkeypatch.setattr(fr4, "event", lambda *a, **kw: None)
    fr4.adopt()
    j = fr4.running["nw-chunk.sh"]
    assert (j["h"]["timing"], j["h"]["gpus"], j["h"]["timing_cpus"], j["unit"]) == (True, 1, "124-191", unit)


needs_lease = pytest.mark.skipif(not all(map(shutil.which, ("bash", "flock", "setsid", "timeout"))) or
                                 not Path("/proc/self/cgroup").read_text().startswith("0::"),
                                 reason="gpu-lease and the keeper need bash, flock, setsid, timeout and cgroup v2")


@needs_lease
def test_the_keeper_holds_a_timing_jobs_lease_until_nothing_else_is_left_in_its_scope(node2, tmp_path):
    """gpu-lease drops its locks once its CMD exits, and `research run` starts its workload with none of them (Popen's
    close_fds), in a session of its own: the workload of a chunk that ended, or that gpu-lease's --max-min stopped, kept running
    on a GPU whose lease was gone. The keeper TIMING_WRAP leaves in the scope holds the locks until it is alone there. A file
    stands for the scope's cgroup.procs (under the root the keeper is given, at this process's cgroup): two lines while the
    workload lives, one once it is gone."""
    load, _, _ = node2
    fr = load()
    gl, root = tmp_path / "gl", tmp_path / "cgroot"
    procs = root / Path("/proc/self/cgroup").read_text().splitlines()[0].split("::", 1)[1].lstrip("/") / "cgroup.procs"
    procs.parent.mkdir(parents=True)
    procs.write_text("1\n2\n")
    job, rc = tmp_path / "job.sh", tmp_path / "rc"
    job.write_text("#!/bin/bash\nexit 0\n")
    env = {**{k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}, "GPU_LEASE_DIR": str(gl), "GPU_LEASE_COUNT": "1",
           "GPU_LEASE_UUID_TABLE": "GPU-a0", "GPU_LEASE_WHO": "fill:net", "GPU_LEASE_USAGE_LOG": str(tmp_path / "usage.jsonl"),
           "GPU_LEASE_UTIL_LOG": str(tmp_path / "util-*.jsonl")}

    def lease(*wrap: str) -> bool:
        """Run a chunk under gpu-lease as fill does; whether GPU 0's lease is still held once gpu-lease has exited."""
        with open(tmp_path / "lease.log", "w") as log:
            subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "0", "--max-min", "5", "--mem-gb", "0", "--", "bash", "-c", wrap[0],
                            "fill-rc", str(rc), str(job), *wrap[1:]], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert rc.read_text() == "0\n"
        with open(gl / "0.lock") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True

    assert lease(fr.RC_WRAP) is False                                   # the lease ended with the job, its workload still there
    try:
        assert lease(fr.TIMING_WRAP, fr.KEEP, str(root)) is True
        procs.write_text("1\n")                                         # the workload is gone: the keeper lets the lease go
        with open(gl / "0.lock") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
    finally:
        procs.write_text("1\n")


def test_a_timing_job_ends_only_once_its_scope_is_empty_and_a_failed_kill_is_logged(node2, monkeypatch):
    """KEEP holds a timing job's lease while anything is left in its scope, so the job isn't filed, nor its cores freed, while
    the scope reads `populated 1` (it was as soon as gpu-lease exited): a tick later what's left is stopped (SIGTERM, then
    SIGKILL), and a kill root's systemctl refuses while the scope is populated is logged, once. A scope fill can't read counts
    as populated."""
    load, fill, cg = node2
    fr = load()
    events, calls, ok = [], [], [True]
    monkeypatch.setattr(fr, "event", lambda kind, job, **kw: events.append((kind, job, kw)))
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0 if ok[0] else 1))

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(fr.os, "killpg", gone)
    cgroup(cg, "vy-timing.slice", "0-191", True, "124-191")
    cgroup(cg / "vy-timing.slice", "fill-timing-1.scope", "124-139", True, "124-139")
    queue(fill, "nw-chunk.sh", TIMING)
    p = fill / "queue" / "nw-chunk.sh"
    h = {**fr.header(p), "on": "7", "timing_cpus": "124-139"}
    shutil.move(str(p), fill / "running" / "nw-chunk.sh")
    (fill / "running" / ".nw-chunk.sh.cpus").write_text("124-139\n")
    fr.running["nw-chunk.sh"] = j = {"pid": 4242, "proc": type("P", (), {"poll": lambda self: 0})(), "h": h, "t0": 0.0,
                                     "started": "-", "paused": False, "unit": "fill-timing-1"}
    fr.reap()
    assert "nw-chunk.sh" in fr.running and events == [] and calls == []
    fr.reap()
    assert events == [("stop", "nw-chunk.sh", {"why": "a process outlived its job"})] and j["requeue"] is False
    assert ["sudo", "-n", "systemctl", "kill", "--signal=SIGTERM", "fill-timing-1.scope"] in calls
    ok[0], j["stop_at"] = False, -1e9                                    # 20 s on, and root's systemctl refuses
    fr.reap()
    fr.reap()
    assert events[1:] == [("scope-kill-failed", "nw-chunk.sh", {"unit": "fill-timing-1", "signal": "SIGKILL"})]
    assert calls[-1] == ["sudo", "-n", "systemctl", "kill", "--signal=SIGKILL", "fill-timing-1.scope"]
    assert "nw-chunk.sh" in fr.running and (fill / "running" / ".nw-chunk.sh.cpus").exists()
    (cg / "vy-timing.slice" / "fill-timing-1.scope" / "cgroup.events").write_text("populated 0\n")
    fr.reap()
    assert "nw-chunk.sh" not in fr.running and (fill / "done" / "nw-chunk.sh").exists()
    assert events[-1][0] == "done" and not (fill / "running" / ".nw-chunk.sh.cpus").exists()
    assert fr.scope_populated({"unit": "fill-timing-2"}) is False
    (cg / "vy-timing.slice" / "fill-timing-2.scope" / "cgroup.events").mkdir(parents=True)
    assert fr.scope_populated({"unit": "fill-timing-2"}) is True


def test_node_ops_alerts_on_a_timing_scope_fill_could_not_kill(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_DIR", str(tmp_path))
    monkeypatch.setenv("FILL_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("node_ops_kill_failed", NODE_OPS)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    (tmp_path / "events.jsonl").write_text(
        '{"t": "-", "event": "scope-kill-failed", "job": "nw-chunk.sh", "unit": "fill-timing-1", "signal": "SIGKILL"}\n'
        '{"t": "-", "event": "stop", "job": "pous-a.sh", "why": "max_min"}\n')
    ops.fill_alerts()
    a, = [json.loads(l) for l in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert (a["kind"], a["key"]) == ("fill-scope-kill-failed", "nw-chunk.sh") and "fill-timing-1.scope" in a["msg"]


def test_timing_cpus_leave_out_every_cpu_another_cgroup_or_a_fence_may_use_and_fail_closed(node2):
    """124-191 are a chunk's only while nothing else can run there: a prover or pous-quiet scope in its top-level slice, an orphaned
    timing scope, a fence, or user.slice widened past 123 takes its CPUs away, and a cgroup that can't be read takes them all.
    A core goes whole or not at all."""
    load, fill, cg = node2
    fr = load()
    assert fr.timing_free([]) == set(range(124, 192))
    cgroup(cg / "provers.slice", "run-u1.scope", "128-191", True, "128-191")
    (cg / "provers.slice" / "cgroup.events").write_text("populated 1\n")
    assert fr.timing_free([]) == {124, 125, 126, 127}
    assert fr.carve(fr.timing_free([]), NUMA1, 4) == "124-127" and fr.carve(fr.timing_free([]), NUMA1, 5) is None
    assert fr.carve({124, 126, 127}, NUMA1, 2) == "126-127" and fr.carve({124, 126, 127}, NUMA1, 3) is None
    assert fr.carve(set(range(124, 192)), set(range(96)), 2) is None
    assert fr.carve(set(range(124, 192)), NUMA1, 7) == "124-131"
    (cg / "provers.slice" / "cgroup.events").write_text("populated 0\n")
    cgroup(cg, "vy-timing.slice", "0-191", True, "124-191")
    cgroup(cg / "vy-timing.slice", "fill-timing-1.scope", "140-147", True, "140-147")
    assert fr.timing_free([]) == set(range(124, 192)) - set(range(140, 148))
    fence = {"cpus": "124-125,190-191"}
    assert fr.timing_free([fence]) == set(range(126, 190)) - set(range(140, 148))
    (cg / "user.slice" / "cpuset.cpus.effective").write_text("0-191\n")
    assert fr.timing_free([]) == set()
    (cg / "user.slice" / "cpuset.cpus.effective").write_text("0-123\n")
    (cg / "system.slice" / "cgroup.events").unlink()
    assert fr.timing_free([]) == set()


def test_no_timing_job_starts_while_a_lent_scope_holds_124_191_and_one_starts_once_it_ends(node2, monkeypatch):
    """Node 2 on 3 Oct, 18:30-20:10Z: circuits' 235B Commit in a scope of the top-level vy-lend.slice on 80-91,124-191, beside
    compute accounting's two zk runs in that slice on 48-79 and 92-123. Each scope counts on its own: the Commit's (or an
    AllowedCPUs on the slice itself) holds every chunk back, and once it is gone a chunk starts with the zk runs still there."""
    load, fill, cg = node2
    fr = load()
    held, want, started, stopped, run = fake_node(fr, monkeypatch, fill)
    cgroup(cg, "vy-lend.slice", "0-191", True)
    for name, cpus in (("circuits-tp8-commit5.scope", "80-91,124-191"), ("zk-a.scope", "48-79"), ("zk-b.scope", "92-123")):
        cgroup(cg / "vy-lend.slice", name, cpus, True, cpus)
    queue(fill, "nw-chunk.sh", TIMING)
    fr.tick(8)
    assert fr.timing_free([]) == set() and started == []
    (cg / "vy-lend.slice" / "circuits-tp8-commit5.scope" / "cgroup.events").write_text("populated 0\n")
    (cg / "vy-lend.slice" / "cpuset.cpus").write_text("80-91,124-191\n")
    (cg / "vy-lend.slice" / "cpuset.cpus.effective").write_text("80-91,124-191\n")
    fr.tick(8)
    assert started == []
    (cg / "vy-lend.slice" / "cpuset.cpus").write_text("\n")
    (cg / "vy-lend.slice" / "cpuset.cpus.effective").write_text("0-191\n")
    fr.tick(8)
    assert fr.timing_free([]) == set(range(140, 192)) and started == [("nw-chunk.sh", "7", "124-139")]


def test_a_fence_takes_no_core_a_running_timing_job_holds(node2, monkeypatch, tmp_path, capsys):
    load, fill, _ = node2
    monkeypatch.setenv("FILL_QUOTA_LOG", str(tmp_path / "quota-changes.log"))
    fr = load()
    monkeypatch.setattr(fr, "gpu_count", lambda: 8)
    (fill / "running" / ".nw-chunk.sh.cpus").write_text("124-139\n")
    t = fr.utc("2026-10-03T20:00Z")
    args = ["ma", "--gpu", "6", "--until", "2026-10-03T21:00Z", "--why", "honest latency"]
    assert fr.fence([*args, "--cpus", "138,139"], t=t) == 2
    assert "timing job nw-chunk.sh has cores 124-139" in capsys.readouterr().err
    assert fr.fence([*args, "--cpus", "140,141"], t=t) == 0
    assert fr.fences(t)[0]["cpus"] == "140-141"


def node_ops(tmp_path, monkeypatch, cgroups: dict[int, str]):
    """node_ops, with /proc/<pid>/cgroup of each pid in `cgroups` read from a file of that text."""
    monkeypatch.setenv("OPS_DIR", str(tmp_path))
    monkeypatch.delenv("QUIET_MATCH", raising=False)
    (tmp_path / "logs").mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("node_ops_timing", NODE_OPS)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    real = ops.Path
    for pid, text in cgroups.items():
        (tmp_path / f"cgroup.{pid}").write_text(text)
    monkeypatch.setattr(ops, "Path", lambda p, *a: real(p, *a) if str(p) not in {f"/proc/{pid}/cgroup" for pid in cgroups}
                        else tmp_path / f"cgroup.{str(p).split('/')[2]}")
    return ops


def test_node_ops_exempts_only_the_top_level_timing_slice(tmp_path, monkeypatch):
    """Quiet's exemption is the system manager's vy-timing.slice, which only root's systemd-run reaches; any user can make a
    vy-timing.slice of their own (`systemd-run --user --scope --slice=vy-timing.slice`), which used to exempt them too."""
    ops = node_ops(tmp_path, monkeypatch, {1: "0::/vy-timing.slice/fill-timing-1.scope\n",
                                           2: "0::/user.slice/user-1001.slice/user@1001.service/vy-timing.slice/run-r1.scope\n",
                                           3: "0::/user.slice/vy-timing.slice/x.scope\n"})
    assert [ops.in_timing_slice(pid) for pid in (1, 2, 3, 1 << 22)] == [True, False, False, False]


def test_node_ops_quiet_never_pauses_a_timing_job_for_a_timed_window(tmp_path, monkeypatch):
    """A stall over 1 s voids a whole chunk (benchmarks/network_traces' record.py), so while a timed window holds GPUs, Quiet
    pauses another CPU-heavy group but leaves one in the timing slice running. Each group's cgroup file says where it runs."""
    token = f"quiet-timing-test-{os.getpid()}"                         # Quiet touches no other process of this user
    busy = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()", "-m", "research.telemetry", "sample",
            token]
    procs = [subprocess.Popen(busy, stdout=subprocess.PIPE, text=True, start_new_session=True) for _ in range(2)]
    try:
        for p in procs:
            assert p.stdout.readline().strip() == "ready"
        chunk, other = procs
        ops = node_ops(tmp_path, monkeypatch, {chunk.pid: "0::/vy-timing.slice/fill-timing-1.scope\n",
                                               other.pid: "0::/user.slice/user-1000.slice/user@1000.service/vy-timing.slice/r.scope\n"})
        monkeypatch.setattr(ops, "QUIET_MATCH", re.compile(re.escape(token)))
        monkeypatch.setattr(ops.Quiet, "window", lambda self: 1 << 22)
        q = ops.Quiet()
        q.tick()
        assert set(q.paused) == {other.pid}
        assert os.WIFSTOPPED(os.waitpid(other.pid, os.WUNTRACED)[1])
        assert Path(f"/proc/{chunk.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "S"
    finally:
        for p in procs:
            os.killpg(p.pid, signal.SIGCONT)
            os.killpg(p.pid, signal.SIGTERM)
            p.wait()


def test_node_ops_quiet_spares_a_gpu_leases_jobs_and_their_own_groups(tmp_path, monkeypatch):
    """A GPU lease's JIT build (ninja gives each compile a group of its own) paused by a timed window held flashinfer's lock, and
    the timed lease's own job then waited on it until its lease ran out (n2, 24 Oct 07:0xZ). Quiet now leaves a lease's processes
    and every group they start running, and still pauses an ordinary CPU-heavy group."""
    token = f"quiet-lease-test-{os.getpid()}"
    busy = f"import signal; print('ready', flush=True); signal.pause()  # -m research.telemetry sample {token}"
    child = ("import os, subprocess, sys; p = subprocess.Popen([sys.executable, '-c', sys.argv[1]], start_new_session=True); "
             "print(p.pid, flush=True); p.wait()")
    lease = subprocess.Popen([sys.executable, "-c", child, busy], stdout=subprocess.PIPE, text=True, start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", busy], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        grandchild = int(lease.stdout.readline())
        assert lease.stdout.readline().strip() == "ready" and other.stdout.readline().strip() == "ready"
        ops = node_ops(tmp_path, monkeypatch, {lease.pid: "0::/user.slice/user-1001.slice/user@1001.service/app.slice/gpu-lease-7.scope\n"})
        monkeypatch.setattr(ops, "QUIET_MATCH", re.compile(re.escape(token)))
        monkeypatch.setattr(ops.Quiet, "window", lambda self: 1 << 22)
        q = ops.Quiet()
        q.tick()
        assert set(q.paused) == {other.pid}
        assert grandchild not in q.paused and os.getpgid(grandchild) == grandchild
    finally:
        for p in (lease, other):
            os.killpg(p.pid, signal.SIGCONT)
        for pg in (grandchild, lease.pid, other.pid):
            try:
                os.killpg(pg, signal.SIGTERM)
            except ProcessLookupError:
                pass
        lease.wait()
        other.wait()


def pool(node2, monkeypatch, slots: int = 6):
    """The Verity pool open, its disk empty, and a start that records (job, its cores, the claim's descriptor) and keeps the
    descriptor open, as the job it stands for does."""
    load, fill, _ = node2
    for k, v in {"FILL_VERITY_UNTIL": "9999-01-01T00:00:00+00:00", "FILL_VERITY_STOP": "9999-01-01T01:00:00+00:00",
                 "FILL_VERITY_CPU_SLOTS": str(slots), "FILL_VERITY_LEND": "1"}.items():
        monkeypatch.setenv(k, v)
    fr = load()
    monkeypatch.setattr(fr, "disk_pct", lambda path: 0.0)
    started, stopped = [], []

    def start(p, h, backfill=False, claim_fd=None):
        started.append((p.name, h["pool_cpus"], claim_fd))
        fr.running[p.name] = {"h": h, "t0": len(fr.running), "pid": 0}
        p.unlink()

    monkeypatch.setattr(fr, "start", start)
    monkeypatch.setattr(fr, "stop", lambda name, why, requeue=True: stopped.append((name, why)))

    def queued():
        return [(p, fr.header(p), 0.0) for p in sorted((fill / "queue").glob("*.sh"))]

    return fr, fill, started, stopped, queued


def test_the_verity_pool_gives_each_job_whole_cores_of_its_own_beside_the_queued_runs_and_a_job_that_doesnt_fit_waits(
        node2, monkeypatch, tmp_path):
    """Each job of the pool claims `cpus` CPUs of 48-95 in whole cores in the registry node 2's queued runs claim theirs in
    (research's cores.py), off the fences and every live claim; one that doesn't fit stays queued until a job's claim ends."""
    from research import cores

    fr, fill, started, stopped, queued = pool(node2, monkeypatch)
    claims = tmp_path / "claims"
    claims.mkdir()
    run, run_fd = cores.take(8, set(range(48, 96)), "r20261019-061310-4377", claims, tmp_path / "cpu")      # a queued run's 48-55
    (fill / "fences").write_text("ma 1 64-71 2099-01-01T00:00Z  # a fence on GPU 1's cores\n")
    queue(fill, "v-a.sh", "owner=a gpus=0 project=verity cpus=8 mem_gb=16")
    queue(fill, "v-b.sh", "owner=b gpus=0 project=verity cpus=3 mem_gb=16")
    queue(fill, "v-c.sh", "owner=c gpus=0 project=verity cpus=24 mem_gb=16")
    fr.verity_pool(queued())
    assert [(n, c) for n, c, _ in started] == [("v-a.sh", "56-63"), ("v-b.sh", "72-75")]      # 64-71 fenced; 3 CPUs: 2 cores
    assert (fill / "queue" / "v-c.sh").exists() and stopped == [], "76-95 hold 20 CPUs, not 24: it waits"
    assert sorted(cores.claimed(claims).values(), key=min) == [run, set(range(56, 64)), set(range(72, 76))]
    assert all(os.get_inheritable(fd) is False for _, _, fd in started), "the runner passes it to the job alone (pass_fds)"
    os.close(run_fd)                                                    # the queued run ended
    fr.verity_pool(queued())
    assert started[-1][:2] == ("v-c.sh", "48-55,76-91")
    for _, _, fd in started:
        os.close(fd)


def test_the_pool_starts_and_lends_nothing_while_a_queued_run_waits_for_cores(node2, monkeypatch, tmp_path):
    from research import cores

    fr, fill, started, stopped, queued = pool(node2, monkeypatch)
    claims = tmp_path / "claims"
    claims.mkdir()
    ticket = cores.locked(claims / "wait.00000000000000000001.r1", "48\n")
    queue(fill, "v-a.sh", "owner=a gpus=0 project=verity cpus=8")
    queue(fill, "p-a.sh", "owner=p gpus=0 cpus=4")
    fr.verity_pool(queued())
    assert started == [] and stopped == []
    os.close(ticket)                                                     # the run took its cores, or gave up
    fr.verity_pool(queued())
    assert [(n, c) for n, c, _ in started] == [("v-a.sh", "48-55"), ("p-a.sh", "56-59")]
    assert fr.running["p-a.sh"]["h"]["lent"] and fr.running["p-a.sh"]["h"]["pool_cpus"] == "56-59"
    queue(fill, "v-b.sh", "owner=b gpus=0 project=verity cpus=40")
    fr.verity_pool(queued())
    assert stopped == [("p-a.sh", "the Verity pool takes its slot back")], "a Verity job that finds no cores requeues a lent one"
    for _, _, fd in started:
        os.close(fd)


def test_a_pool_job_starts_at_nice_0_in_a_scope_holding_its_cores_and_inherits_its_claim(node2, monkeypatch, tmp_path):
    """No `nice -n 19` and the scope's AllowedCPUs are the claimed cores; the job gets the claim's descriptor and the runner keeps
    none, so the claim ends with the job. A Verity job's cpus is a whole number, at most the pool's 48 CPUs."""
    load, fill, _ = node2
    fr = load()
    cmds = []
    monkeypatch.setattr(fr.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: cmds.append((cmd, kw)) or type("P", (), {"pid": 4242})())
    monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
    queue(fill, "v-a.sh", "owner=a gpus=0 project=verity cpus=8 mem_gb=16")
    p = fill / "queue" / "v-a.sh"
    cpus, fd = fr.claim("v-a.sh", 8)
    fr.start(p, {**fr.header(p), "pool_cpus": cpus}, claim_fd=fd)
    (cmd, kw), = cmds
    assert cpus == "48-55" and kw["pass_fds"] == (fd,) and kw["env"]["OMP_NUM_THREADS"] == "8"
    assert "nice" not in cmd and "AllowedCPUs=48-55" in cmd and cmd[cmd.index("taskset"):][:3] == ["taskset", "-c", "48-55"]
    with pytest.raises(OSError):
        os.fstat(fd)                                                     # closed here once the job has it
    fr.running["v-a.sh"]["started"] = "-"
    fake_node(fr, monkeypatch, fill)
    fr.tick(8)
    assert "running v-a.sh owner=a gpus=0 since=- cpus=48-55" in (fill / "status.txt").read_text()
    for head, want in (("cpus=999", 48), ("cpus=0", 1)):
        queue(fill, "v-b.sh", f"owner=a gpus=0 project=verity {head}")
        assert fr.header(fill / "queue" / "v-b.sh")["cpus"] == want
    queue(fill, "v-b.sh", "owner=a gpus=0 project=verity cpus=eight")
    assert fr.header(fill / "queue" / "v-b.sh")["bad"] == "cpus=eight"
