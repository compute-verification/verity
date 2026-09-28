#!/usr/bin/env python3
"""Preemptible fill jobs on a shared GPU server: run them only on GPUs nobody else wants (stdlib only).

A fill job is an executable script in FILL_DIR/queue/.  Its header says what it needs:

    # fill: owner=bc-0de2d624 gpus=1 max_min=25 cpus=8 project=pous mem_gb=128
    #   gpus 0 or 1 (default 1); max_min <= 30 (default 25); cpus: threads it may use (OMP_NUM_THREADS; default 8); on: the GPU
#   index or indices the job may run on (3, 2-7 or 2,5; it waits for one of them); prio: higher starts first (default 0; give GPU-heavy jobs more than
#   ones that hold a GPU while mostly computing on the CPU); within a prio, owners take turns (the owner with the fewest
#   jobs of that kind running goes next), then the longest-queued job (a requeued job queues afresh);
    #   project pous (default) or verity (Verity's work, which runs only when no pous job is queued, except circuits' Commit
#   guests, verity-commit-*.sh, which rank above every other job); mem_gb: a CPU job's memory cap
#   (default 128; a GPU job's cap is gpu-lease's, 90% of RAM over the GPU count)

It must be restartable: it can be stopped (SIGTERM, then SIGKILL 20 s later) at any moment and started again from the top,
so it resumes from checkpoints in its own output directory.  Its exit status says what happened:
    0 done (moved to done/);  99 more chunks remain (requeued at once);  143 preempted by a waiter (requeued);
    anything else failed (failed/, after 2 tries).
It runs with FILL=1, FILL_JOB=<name>, GPU_LEASE_WHO=fill:<owner>; a GPU job through `gpu-lease 1 --preemptible`, so the
oldest waiter can stop it, and a CPU job in a systemd user scope with its memory cap.

Every POLL_S seconds the runner:
  - finds the free GPUs (gpu-lease's locks), the waiting `gpu-lease ... --wait` requests, and a timed window (one holder of
    every GPU, or a waiter for all of them);
  - stops GPU fill jobs, proofs' and other Verity jobs first, then pous, then Commits, newest first within each, until the free
    GPUs cover every waiter plus RESERVE (default 0: waiters with --wait preempt fill leases anyway), and all of them while
    a timed window waits or runs; pauses CPU fill jobs (SIGSTOP) during a timed window and resumes them after;
    while the node's scheduler agent holds agent.lock it stops nothing for waiters (the agent evicts), but still drains fill
    for a waiting window, since the agent's eviction signals only gpu-lease's pid;
  - stops a GPU job through its lease's scope (gpu-lease-<pid>) as well as its process group, and kills whatever is left in a
    job's scope once the job ends, since `research run` starts its workload in a session of its own that outlives both;
  - stops a job past its max_min (not counting time paused) and requeues it;
  - starts GPU jobs on every free GPU beyond RESERVE when nobody waits, except the GPUs listed in FILL_DIR/keep-free
    (e.g. `1`, or `0,1`, with an optional `# reason`), and none whose max_min would reach into a window booked in
    FILL_DIR/windows (`2026-10-01T10:00Z 30 # who`, one per line: start in UTC, minutes); Commit guests first, then pous,
    then the rest of Verity's (only with 2+ GPUs free), and a Commit that finds no free GPU stops the newest proofs or pous
    GPU job outside a window; and CPU jobs up to CPU_SLOTS, pinned
    to CPU_SET at nice 19; more queued CPU jobs borrow the Verity pool's idle slots while no Verity job waits;
  - runs the CPU jobs of an owner listed in FILL_DIR/cpu-sets (`<owner> <cpus> <slots> [mem_gb]  # why`, read each tick) on
    those CPUs instead, up to that many at a time, at nice 19 and ionice idle, with their memory cap lowered to mem_gb if
    given; and logs each scoped CPU job's peak memory (mem_peak_gb) when it exits.
Status: FILL_DIR/status.txt; every start, stop and exit: FILL_DIR/events.jsonl.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FILL = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill"))
LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease"))
GPU_LEASE = os.environ.get("FILL_GPU_LEASE", "/workspace/pouw/infra/bin/gpu-lease")
POLL_S = float(os.environ.get("FILL_POLL_S", "10"))
RESERVE = int(os.environ.get("FILL_RESERVE", "0"))
CPU_SLOTS = int(os.environ.get("FILL_CPU_SLOTS", "4"))
CPU_SET = os.environ.get("FILL_CPU_SET", "96-127")
CPU_MEM_GB = int(os.environ.get("FILL_CPU_MEM_GB", "128"))
# PoUW's terms for CPU fill on NUMA 0 (bc-2aa33ad8, 30 Sep): CPU jobs freeze from the moment a window waits, not only once it is
# timed, and pous and lent CPU jobs keep their memory off NUMA 0 (numactl --membind=<node>).
FREEZE_ON_WAIT = os.environ.get("FILL_FREEZE_ON_WAIT", "0") == "1"
CPU_MEMBIND = os.environ.get("FILL_CPU_MEMBIND", "")
MAX_TRIES = 2
# Verity's CPU-only jobs (project=verity, gpus=0) get their own pool, so they never take pous's CPU slots (the terms agreed with
# the nebius-infra steward): their CPU set, a job count, a per-job and a total memory cap, no start once the disk reaches
# VERITY_DISK_PCT, and none after VERITY_UNTIL; running ones are stopped at VERITY_STOP, before the node stops itself.
VERITY_CPU_SET = os.environ.get("FILL_VERITY_CPU_SET", "48-95")
VERITY_CPU_SLOTS = int(os.environ.get("FILL_VERITY_CPU_SLOTS", "6"))
VERITY_MEM_MAX_GB = int(os.environ.get("FILL_VERITY_MEM_MAX_GB", "256"))
VERITY_MEM_TOTAL_GB = int(os.environ.get("FILL_VERITY_MEM_TOTAL_GB", "1024"))
VERITY_DISK_PCT = float(os.environ.get("FILL_VERITY_DISK_PCT", "55"))
VERITY_UNTIL = datetime.fromisoformat(os.environ.get("FILL_VERITY_UNTIL", "2026-10-07T12:00:00+00:00"))
VERITY_STOP = datetime.fromisoformat(os.environ.get("FILL_VERITY_STOP", "2026-10-07T13:00:00+00:00"))
VERITY_DISK_PATH = os.environ.get("FILL_VERITY_DISK_PATH", "/workspace")
# the pool lends its idle slots to queued pous CPU jobs (Daniel 19:12Z: borrowing goes both ways); a Verity job that can't start
# takes a slot back by requeueing the newest lent job (143, as a preempted job)
VERITY_LEND = os.environ.get("FILL_VERITY_LEND", "1") == "1"
COMMIT_PREFIX = "verity-commit-"
CGROUP_ROOT = os.environ.get("FILL_CGROUP_ROOT", "/sys/fs/cgroup")

running: dict[str, dict] = {}
backoff_until = 0.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event(kind: str, job: str, **kw) -> None:
    with open(FILL / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": now(), "event": kind, "job": job, **kw}) + "\n")


# one-job exceptions to the 30-min GPU cap, each approved by infra by name
MAX_MIN_EXCEPTIONS: dict[str, int] = {}


def header(p: Path) -> dict:
    h = {"owner": "unknown", "gpus": 1, "max_min": 25, "cpus": 8, "project": "pous", "on": None, "prio": 0}
    for line in p.read_text(errors="replace").splitlines()[:20]:
        m = re.match(r"#\s*fill:\s*(.*)", line)
        if m:
            for k, v in re.findall(r"(\w+)=(\S+)", m.group(1)):
                h[k] = int(v) if v.isdigit() else v
    h["gpus"] = 1 if int(h["gpus"]) >= 1 else 0
    # a Verity CPU job has its own CPUs and is frozen through windows, so it may run a whole vLLM Build (2-50 min, some 1-4 h)
    cap = MAX_MIN_EXCEPTIONS.get(p.name, 360 if h["project"] == "verity" and not h["gpus"] else 30)
    h["max_min"] = min(int(h["max_min"]), cap)
    return h


def cpu_sets() -> dict[str, dict]:
    """FILL_DIR/cpu-sets: owners whose CPU jobs get CPUs of their own (verity-top, 1 Oct: compute accounting's verifies on 0-47)."""
    out = {}
    try:
        lines = (FILL / "cpu-sets").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        f = line.split("#")[0].split()
        if len(f) >= 3 and re.fullmatch(r"[\d,-]+", f[1]) and f[2].isdigit() and (len(f) < 4 or f[3].isdigit()):
            out[f[0]] = {"cpus": f[1], "slots": int(f[2]), "mem_gb": int(f[3]) if len(f) > 3 else None}
    return out


def mem_peak_gb(pid: int) -> float | None:
    """The peak memory of the cgroup `pid` runs in (its scope), in GB."""
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[1]
        return round(int((Path(CGROUP_ROOT) / cg.lstrip("/") / "memory.peak").read_text()) / 1e9, 2)
    except (OSError, IndexError, ValueError):
        return None


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l.startswith("GPU "))


def lease(i: int) -> str | None:
    lock = LEASE_DIR / f"{i}.lock"
    if not lock.exists() or subprocess.run(["flock", "-n", "-s", str(lock), "true"], capture_output=True).returncode == 0:
        return None
    try:
        return (LEASE_DIR / f"{i}.owner").read_text().strip() or "held"
    except OSError:
        return "held"


def agent_up() -> bool:
    """Whether the node's scheduler agent (`cluster agent --mode live`) holds LEASE_DIR/agent.lock."""
    try:
        fd = os.open(LEASE_DIR / "agent.lock", os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def waiters() -> list[int]:
    """How many GPUs each waiting `gpu-lease N --wait` wants, read from gpu-lease's own queue.  A request that can't start at once
    queues as LEASE_DIR/wait.<ns>.<pid>, holds that file locked while it waits, and removes it once it has its GPUs, so a lease
    that holds its GPUs (its argv still says --wait) is never a waiter.  A file nobody holds locked is a dead waiter's."""
    out = []
    for f in sorted(LEASE_DIR.glob("wait.*")):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                continue
            except BlockingIOError:
                pass
            m = re.search(r"\bn=(\d+)", os.read(fd, 4096).decode(errors="replace"))
            if m:
                out.append(int(m.group(1)))
        finally:
            os.close(fd)
    return out


def scope(j: dict, *args: str) -> bool:
    """systemctl --user on a job's scope (a Verity or lent CPU job's own, a GPU job's gpu-lease-<pid>): freeze, thaw and kill
    reach every process in it, including a workload that `research run` starts as its own process group, which killpg misses."""
    unit = j.get("unit")
    return bool(unit) and subprocess.run(["systemctl", "--user", *args, f"{unit}.scope"], capture_output=True).returncode == 0


def stop(name: str, why: str, requeue: bool = True) -> None:
    j = running[name]
    if j.get("stopping"):
        return
    try:
        scope(j, "thaw")
        os.killpg(j["pid"], signal.SIGCONT)
        os.killpg(j["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    scope(j, "kill", "--signal=SIGTERM")
    j["stopping"], j["stop_at"], j["requeue"], j["why"] = True, time.monotonic(), requeue, why
    event("stop", name, why=why)


def start(p: Path, h: dict, backfill: bool = False) -> None:
    name = p.name
    run = FILL / "running" / name
    shutil.move(str(p), run)
    log = open(FILL / "logs" / f"{name}.{datetime.now(timezone.utc):%H%M%S}.log", "w")
    env = {**os.environ, "FILL": "1", "FILL_JOB": name, "GPU_LEASE_WHO": f"fill:{h['owner']}",
           "OMP_NUM_THREADS": str(h["cpus"])}
    verity = h["project"] == "verity" and not h["gpus"]
    in_pool = verity or h.get("lent")
    unit_name = None
    own = h.get("cpuset") if not h["gpus"] else None
    cpus = own or (VERITY_CPU_SET if in_pool else CPU_SET)
    cmd = ["nice", "-n", "19", "taskset", "-c", cpus, "bash", str(run)]
    if own:
        cmd = ["ionice", "-c3", *cmd]
    if CPU_MEMBIND and not h["gpus"] and not verity:
        cmd = ["numactl", f"--membind={CPU_MEMBIND}", *cmd]
    if h["gpus"]:
        pin = ["--on", str(h["on"])] if h.get("on") is not None else []
        cmd = [GPU_LEASE, "1", *pin, "--preemptible", "--max-min", str(h["max_min"]), "--", *cmd]
    elif shutil.which("systemd-run"):
        kind = "cpuset" if own else "verity" if verity else "lent"
        unit_name = f"fill-{kind}-{int(time.time() * 1000)}" if in_pool or own else None
        unit = ["--unit", unit_name] if unit_name else []
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", *unit, "-p", f"MemoryMax={h.get('mem_gb', CPU_MEM_GB)}G",
               "-p", "MemorySwapMax=0", "--", *cmd]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(FILL), start_new_session=True)
    log.close()
    if h["gpus"]:
        unit_name = f"gpu-lease-{proc.pid}"         # gpu-lease runs CMD in the scope named by its own pid
    (FILL / "running" / f".{name}.pgid").write_text(f"{proc.pid}\n")
    if unit_name:
        (FILL / "running" / f".{name}.unit").write_text(f"{unit_name}\n")
    running[name] = {"pid": proc.pid, "proc": proc, "h": h, "t0": time.monotonic(), "started": now(), "paused": False,
                     "unit": unit_name}
    event("start", name, owner=h["owner"], gpus=h["gpus"], project=h["project"], backfill=backfill)


def gpu_set(on, n: int) -> set[int] | None:
    if on is None:
        return None
    out = set()
    for part in str(on).split(","):
        lo, _, hi = part.partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            out.update(i for i in range(int(lo), int(hi or lo) + 1) if i < n)
    return out


def keep_free(n: int) -> set[int]:
    try:
        return gpu_set((FILL / "keep-free").read_text().split("#")[0].strip() or None, n) or set()
    except OSError:
        return set()


def rank(name: str, h: dict) -> int:
    """Node 2's order (Daniel): circuits' Commit guests, then pous, then proofs and the rest of Verity's jobs."""
    return 0 if name.startswith(COMMIT_PREFIX) else 1 if h["project"] != "verity" else 2


def windows() -> list[tuple[float, float]]:
    """The timed windows booked in FILL_DIR/windows, as (start, end) in epoch seconds."""
    out = []
    try:
        lines = (FILL / "windows").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        f = line.split("#")[0].split()
        try:
            t = datetime.fromisoformat(f[0]).timestamp()
            out.append((t, t + 60 * float(f[1] if len(f) > 1 else 30)))
        except (IndexError, ValueError):
            continue
    return out


wall = time.time


def clears(h: dict, booked: list[tuple[float, float]], t: float) -> bool:
    """Whether a job started at `t` ends, at its max_min, before every booked window, or starts after it."""
    return all(t + 60 * h["max_min"] <= s or t >= e for s, e in booked)


def next_job(left: list, counts: Counter):
    return min(left, key=lambda x: (rank(x[0].name, x[1]), -int(x[1].get("prio") or 0), counts[x[1]["owner"]], x[2], x[0].name))


def active_s(j: dict) -> float:
    t = time.monotonic()
    return t - j["t0"] - j.get("paused_s", 0.0) - (t - j["paused_at"] if j["paused"] else 0.0)


def tries(name: str) -> int:
    p = FILL / "tries.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[name] = d.get(name, 0) + 1
    p.write_text(json.dumps(d))
    return d[name]


def reap() -> None:
    global backoff_until
    for name, j in list(running.items()):
        if j["proc"] is None:                       # adopted from an earlier runner: no exit status, only whether it lives
            try:
                os.killpg(j["pid"], 0)
                rc = None
            except ProcessLookupError:
                rc = "adopted-exit"
        else:
            rc = j["proc"].poll()
        if rc is None and j.get("unit") and not j["h"]["gpus"]:
            peak = mem_peak_gb(j["pid"])
            if peak is not None:
                j["mem_peak_gb"] = max(peak, j.get("mem_peak_gb") or 0.0)
        if rc is None:
            if j.get("stopping") and time.monotonic() - j["stop_at"] > 20:
                try:
                    os.killpg(j["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                scope(j, "kill", "--signal=SIGKILL")
            continue
        if scope(j, "kill", "--signal=SIGKILL"):
            event("scope-residue", name, unit=j["unit"])
        run = FILL / "running" / name
        mins = round((time.monotonic() - j["t0"]) / 60, 1)
        paused_min = round((j.get("paused_s", 0.0) + (time.monotonic() - j["paused_at"] if j["paused"] else 0.0)) / 60, 1)
        (FILL / "running" / f".{name}.pgid").unlink(missing_ok=True)
        (FILL / "running" / f".{name}.unit").unlink(missing_ok=True)
        if rc == "adopted-exit":
            dest, kind = "queue", "adopted-exit"      # its own checkpoint makes the rerun a no-op if it finished
        elif j.get("stopping") and j.get("requeue"):
            dest, kind = "queue", "requeued"
        elif rc == 0:
            dest, kind = "done", "done"
        elif rc in (99, 75, 124, 143, -15):
            dest, kind = "queue", {99: "more", 75: "no-gpu", 124: "max_min"}.get(rc, "preempted")
            if rc == 75:
                backoff_until = time.monotonic() + 15
        else:
            dest, kind = ("failed", "failed") if tries(name) >= MAX_TRIES else ("queue", "retry")
        if run.exists():
            shutil.move(str(run), FILL / dest / name)
            if dest == "queue":
                os.utime(FILL / dest / name)
        peak = {"mem_peak_gb": j["mem_peak_gb"]} if j.get("mem_peak_gb") is not None else {}
        event(kind, name, rc=rc, minutes=mins, paused_min=paused_min, owner=j["h"]["owner"], gpus=j["h"]["gpus"],
              why=j.get("why"), **peak)
        del running[name]


def disk_pct(path: str) -> float:
    s = os.statvfs(path)
    return 100.0 * (1 - s.f_bavail / s.f_blocks)


def verity_pool(queued: list) -> None:
    t = datetime.now(timezone.utc)
    jobs = {k: j for k, j in running.items() if not j["h"]["gpus"] and j["h"]["project"] == "verity"}
    lent = {k: j for k, j in running.items() if j["h"].get("lent") and not j.get("stopping")}
    if t >= VERITY_STOP:
        for k, j in jobs.items():
            if not j.get("stopping"):
                stop(k, "verity cutoff before the node stops")
        return
    if t >= VERITY_UNTIL or disk_pct(VERITY_DISK_PATH) >= VERITY_DISK_PCT:
        return
    mem = sum(int(j["h"].get("mem_gb", CPU_MEM_GB)) for j in [*jobs.values(), *lent.values()])
    left = [x for x in queued if not x[1]["gpus"] and x[1]["project"] == "verity"]
    counts = Counter(j["h"]["owner"] for j in jobs.values())
    slots = VERITY_CPU_SLOTS - len(jobs) - len(lent)
    held = False
    while left:
        best = next_job(left, counts)
        left.remove(best)
        p, h, _ = best
        need = min(int(h.get("mem_gb", CPU_MEM_GB)), VERITY_MEM_MAX_GB)
        if slots <= 0 or mem + need > VERITY_MEM_TOTAL_GB:
            held = True
            continue
        start(p, {**h, "mem_gb": need})
        counts[h["owner"]] += 1
        mem += need
        slots -= 1
    if held and lent:
        stop(max(lent, key=lambda k: lent[k]["t0"]), "the Verity pool takes its slot back")
        return
    if held or not VERITY_LEND:
        return
    borrow = [x for x in queued if not x[1]["gpus"] and x[1]["project"] != "verity" and x[0].exists()]
    counts = Counter(j["h"]["owner"] for j in lent.values())
    while borrow and slots > 0:
        best = next_job(borrow, counts)
        borrow.remove(best)
        p, h, _ = best
        need = min(int(h.get("mem_gb", CPU_MEM_GB)), VERITY_MEM_MAX_GB)
        if mem + need > VERITY_MEM_TOTAL_GB:
            continue
        start(p, {**h, "mem_gb": need, "lent": True})
        counts[h["owner"]] += 1
        mem += need
        slots -= 1


NODE_MEMINFO = os.environ.get("FILL_NODE_MEMINFO", "/sys/devices/system/node/node0/meminfo")


def numa0_gb() -> dict:
    """NUMA 0's reclaimable (MemFree + Inactive(file)) and anonymous memory in GB, logged beside each window's freeze."""
    kb = {}
    try:
        for line in Path(NODE_MEMINFO).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[2].rstrip(":") in ("MemFree", "Inactive(file)", "AnonPages"):
                kb[parts[2].rstrip(":")] = int(parts[3])
    except (OSError, ValueError):
        return {}
    return {"numa0_reclaimable_gb": round((kb.get("MemFree", 0) + kb.get("Inactive(file)", 0)) / 2**20, 1),
            "numa0_anon_gb": round(kb.get("AnonPages", 0) / 2**20, 1)}


def cpu_frozen(timed: bool, window: bool) -> bool:
    return timed or (FREEZE_ON_WAIT and window)


def tick(n: int) -> str:
    global backoff_until
    reap()
    leases = [lease(i) for i in range(n)]
    free = sum(1 for l in leases if l is None)
    want = waiters()
    holders = {l.split()[0] for l in leases if l}
    timed = (n > 0 and all(leases) and len(holders) == 1) or any(l and " timed=1 " in f" {l} " for l in leases)
    window = not timed and any(w >= n for w in want)
    gpu_jobs = sorted((j["t0"], k) for k, j in running.items() if j["h"]["gpus"] and not j.get("stopping"))
    blockers = (n - free) - len(gpu_jobs)
    need = RESERVE + sum(want)
    agent = agent_up()
    if timed or (window and blockers <= 0):
        # the window is running, or only fill stands between it and the GPUs
        for _, k in gpu_jobs:
            stop(k, "timed window")
    elif not window and not agent:
        order = sorted(gpu_jobs, key=lambda x: (-rank(x[1], running[x[1]]["h"]), -x[0]))
        deficit = need - free
        for _, k in order:
            if deficit <= 0:
                break
            stop(k, f"free {free} < needed {need}")
            deficit -= 1
    frozen = cpu_frozen(timed, window)
    if frozen and any(not j["h"]["gpus"] and not j.get("stopping") and not j["paused"] for j in running.values()):
        event("freeze-memory", "-", timed=timed, waiting=window, **numa0_gb())
    for k, j in running.items():
        if not j["h"]["gpus"] and not j.get("stopping"):
            try:
                if frozen and not j["paused"]:
                    scope(j, "freeze")
                    os.killpg(j["pid"], signal.SIGSTOP); j["paused"], j["paused_at"] = True, time.monotonic(); event("pause", k)
                elif not frozen and j["paused"]:
                    scope(j, "thaw")
                    os.killpg(j["pid"], signal.SIGCONT)
                    j["paused_s"] = j.get("paused_s", 0.0) + time.monotonic() - j["paused_at"]; j["paused"] = False
                    event("resume", k)
            except ProcessLookupError:
                pass
        if not j.get("stopping") and not j["paused"] and active_s(j) > 60 * j["h"]["max_min"]:
            stop(k, "max_min")
    queued = []
    for p in sorted((FILL / "queue").glob("*")):
        if p.is_file() and not p.name.startswith("."):
            try:
                queued.append((p, header(p), p.stat().st_mtime))
            except OSError:
                pass
    started = False
    # a window blocked by another lease: fill backfills, admitted by gpu-lease --max-min only if it ends before the window can start
    backfill = window and blockers > 0
    if not timed and (not want or backfill) and time.monotonic() >= backoff_until:
        # a job pinned with on=<index> waits for that GPU (a busy one would only bounce off gpu-lease)
        keep = keep_free(n)
        booked, t_now = windows(), wall()
        idle = [i for i, l in enumerate(leases) if l is None and i not in keep]
        ready = [x for x in queued if x[1]["gpus"] and idle and clears(x[1], booked, t_now) and
                 (gpu_set(x[1].get("on"), n) is None or any(i in idle for i in gpu_set(x[1]["on"], n)))]
        by_rank = [[x for x in ready if rank(x[0].name, x[1]) == r] for r in range(3)]
        left = by_rank[0] + (by_rank[1] or (by_rank[2] if free >= 2 else []))
        counts = Counter(j["h"]["owner"] for j in running.values() if j["h"]["gpus"])
        # each job gets its own free GPU (--on), so jobs started in one tick don't take each other's
        slots = max(0, len(idle) - RESERVE)
        while left and slots > 0:
            best = next_job(left, counts)
            left.remove(best)
            p, h, _ = best
            allowed = gpu_set(h.get("on"), n)
            want_gpu = next((i for i in reversed(idle) if allowed is None or i in allowed), None)
            if want_gpu is None or want_gpu not in idle:
                continue
            idle.remove(want_gpu)
            start(p, {**h, "on": want_gpu}, backfill=backfill)
            counts[h["owner"]] += 1
            slots -= 1
            started = True
        if not window:
            # a Commit that found no free GPU takes the newest proofs (then pous) fill's, one per Commit, counting GPUs already freeing
            waiting = sum(1 for p, h, _ in queued if h["gpus"] and rank(p.name, h) == 0 and p.exists() and clears(h, booked, t_now))
            freeing = sum(1 for j in running.values() if j["h"]["gpus"] and j.get("stopping"))
            victims = sorted((k for k, j in running.items() if j["h"]["gpus"] and not j.get("stopping") and rank(k, j["h"])),
                             key=lambda k: (-rank(k, running[k]["h"]), -running[k]["t0"]))
            for k in victims[:max(0, waiting - freeing)]:
                stop(k, "a Commit reclaims its GPU")
    if not timed:
        sets = cpu_sets()
        own = [x for x in queued if not x[1]["gpus"] and x[1]["project"] != "verity" and x[1]["owner"] in sets]
        rest = [x for x in queued if x not in own]
        for owner, s in sets.items():
            busy = sum(1 for j in running.values() if j["h"].get("cpuset") and j["h"]["owner"] == owner)
            mine = sorted((x for x in own if x[1]["owner"] == owner), key=lambda x: (x[2], x[0].name))
            for p, h, _ in mine[:max(0, s["slots"] - busy)]:
                cap = int(h.get("mem_gb", CPU_MEM_GB))
                start(p, {**h, "cpuset": s["cpus"], "mem_gb": min(cap, s["mem_gb"]) if s["mem_gb"] else cap})
        pous_cpu = [j for j in running.values() if not j["h"]["gpus"] and j["h"]["project"] != "verity"
                    and not j["h"].get("lent") and not j["h"].get("cpuset")]
        left = [x for x in rest if not x[1]["gpus"] and x[1]["project"] != "verity"]
        counts = Counter(j["h"]["owner"] for j in pous_cpu)
        cpu_running = len(pous_cpu)
        while left and cpu_running < CPU_SLOTS:
            best = next_job(left, counts)
            left.remove(best)
            start(best[0], best[1])
            counts[best[1]["owner"]] += 1
            cpu_running += 1
        verity_pool(rest)
    lines = [f"{now()} free GPUs {free}/{n}, kept free {sorted(keep_free(n)) or '-'}, waiters {want or '-'}, timed {timed}, "
             f"window waiting {window}, "
             f"backfill {backfill}, queued {len(queued)} "
             f"(gpu {sum(1 for x in queued if x[1]['gpus'])}, cpu {sum(1 for x in queued if not x[1]['gpus'])}), "
             f"done {sum(1 for _ in (FILL / 'done').glob('*'))}, failed {sum(1 for _ in (FILL / 'failed').glob('*'))}"]
    for k, j in running.items():
        lines.append(f"  running {k} owner={j['h']['owner']} gpus={j['h']['gpus']} since={j['started']}"
                     f"{' PAUSED' if j['paused'] else ''}{' stopping: ' + j['why'] if j.get('stopping') else ''}")
    (FILL / "status.txt").write_text("\n".join(lines) + "\n")
    return lines[0]


def main() -> None:
    for d in ("queue", "running", "done", "failed", "logs"):
        (FILL / d).mkdir(parents=True, exist_ok=True)
    for p in (FILL / "running").glob("*"):
        if p.name.startswith("."):
            continue
        side = FILL / "running" / f".{p.name}.pgid"
        try:
            pgid = int(side.read_text().strip())
            os.killpg(pgid, 0)
        except (OSError, ValueError):
            side.unlink(missing_ok=True)
            shutil.move(str(p), FILL / "queue" / p.name)
            continue
        try:
            stopped = Path(f"/proc/{pgid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "T"
        except OSError:
            stopped = False
        try:
            unit = (FILL / "running" / f".{p.name}.unit").read_text().strip() or None
        except OSError:
            unit = None
        h = header(p)
        if unit and unit.startswith("fill-lent-"):
            h["lent"] = True
        if unit and unit.startswith("fill-cpuset-"):
            h["cpuset"] = cpu_sets().get(h["owner"], {}).get("cpus") or "adopted"
        if h["gpus"] and not unit:
            unit = f"gpu-lease-{pgid}"
        running[p.name] = {"pid": pgid, "proc": None, "h": h, "t0": time.monotonic(), "started": now(),
                           "paused": stopped, "paused_at": time.monotonic(), "unit": unit}
        event("adopted", p.name, pgid=pgid)
    n = gpu_count()
    while True:
        try:
            tick(n)
        except Exception as e:  # noqa: BLE001
            event("runner-error", "-", error=str(e)[:300])
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
