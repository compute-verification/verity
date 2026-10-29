#!/usr/bin/env python3
"""Preemptible fill jobs on a shared GPU server: run them only on GPUs nobody else wants (stdlib only).

A fill job is an executable script in FILL_DIR/queue/.  Its header says what it needs:

    # fill: owner=bc-0de2d624 gpus=1 max_min=25 cpus=8 project=pous mem_gb=128
    #   gpus 0 or 1 (default 1), or 2 for a Commit guest (a TP2 row: two idle GPUs on one NUMA node); max_min <= 30 (default
    #   25), a Commit guest's <= FILL_COMMIT_MAX_MIN (120), a job FILL_DIR/max-min names <= what it allows (infra's exceptions,
    #   each with an end: `pous-honest-soak-* 540 2026-10-02T14:45Z  # who, why`); cpus: threads it may use (OMP_NUM_THREADS; default 8); on: the GPU
#   index or indices the job may run on (3, 2-7 or 2,5; it waits for one of them); prio: higher starts first (default 0; give GPU-heavy jobs more than
#   ones that hold a GPU while mostly computing on the CPU); within a prio, owners take turns (the owner with the fewest
#   jobs of that kind running goes next), then the longest-queued job (a requeued job queues afresh);
    #   project pous (default) or verity (Verity's work, which runs only when no pous job is queued, except circuits' Commit
#   guests, verity-commit-*.sh, which rank above every other job); mem_gb: a CPU job's memory cap
#   (default 128; a GPU job's cap is gpu-lease's, 90% of RAM over the GPU count, a timing job's its own scope's, default 128
#   and at most gpu-lease's);
#   timing=1: a timing chunk (network accounting's), from an owner FILL_DIR/timing-owners lists (one glob per line, infra's:
#   `network-accounting  # who, why`; anyone else's goes to failed/), on one GPU with cpus CPUs of its own (1 to 16, whole
#   cores; see below)
#   and its thread pools (OMP_NUM_THREADS, OPENBLAS_, MKL_, NUMEXPR_) one per core, max_min <= FILL_TIMING_MAX_MIN (60)
#   whatever max-min line or fence names it (a Commit guest's timing=1 is ignored);
#   a value fill reads as a whole number that isn't one sends the job to failed/ (event `failed`, why `header: gpus=one`)

It must be restartable: it can be stopped (SIGTERM, then SIGKILL 20 s later) at any moment and started again from the top,
so it resumes from checkpoints in its own output directory.  Its exit status says what happened:
    0 done (moved to done/);  99 more chunks remain (requeued at once);  143 preempted by a waiter (requeued);
    anything else failed (failed/, after 2 tries).
A wrapper in the job's own scope writes that status to running/.<job>.rc as the job exits, so a job that ends while a restarted
runner has adopted it is filed by it too.  A job whose status the runner can't learn goes to held-unknown-exit/, never back to the
queue: it may have finished, and rerunning a finished job can delete its result.  Exit 0 is done even when a stop raced it.
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
    (e.g. `1`, or `0,1`, with an optional `# reason`; a waiter pinned inside them doesn't count), and none whose max_min would reach into a window booked in
    FILL_DIR/windows (`2026-10-01T10:00Z 30 # who`, one per line: start in UTC, minutes; a timing job's max_min plus
    TIMING_MARGIN_MIN, 2); Commit guests first, then pous,
    then the rest of Verity's (only with 2+ GPUs free), and a Commit that finds no free GPU stops the newest proofs or pous
    GPU job outside a window; and CPU jobs up to CPU_SLOTS, pinned
    to CPU_SET at nice 19; more queued CPU jobs borrow the Verity pool's idle slots while no Verity job waits;
  - gives each job of the Verity pool (a lent one too) `cpus` CPUs of VERITY_CPU_SET off the fences in whole cores of its own,
    claimed in CLAIMS beside node 2's queued runs (research's cores.py), at nice 0 in a scope that holds them (AllowedCPUs);
    a job whose cores aren't free, or any while a queued run waits in the claims' line, stays queued;
  - runs the CPU jobs of an owner listed in FILL_DIR/cpu-sets (`<owner> <cpus> <slots> [mem_gb]  # why`, read each tick) on
    those CPUs instead, up to that many at a time, at nice 19 and ionice idle, with their memory cap lowered to mem_gb if
    given, and paused (none started) while a train check holds FILL_YIELD_LOCK if those CPUs meet FILL_YIELD_CPUS; and logs
    each scoped CPU job's peak memory (mem_peak_gb) when it exits;
  - keeps fill off each fence in FILL_DIR/fences (`<owner> <gpu> <cpus> <until UTC> [<job glob> <minutes>]  # why`, read each
    tick, a line past its until ignored), which `fill_runner.py fence` writes and logs, one per owner, at most 1 GPU, FENCE_MAX_CPUS
    cores on that GPU's NUMA node and FENCE_MAX_H hours, none a running timing job holds: it stops any other fill job on the GPU,
    moves running fill jobs' threads
    off the cores, keeps the GPU as if in keep-free and starts no job on the cores; a job the glob names runs on that GPU alone,
    pinned to those cores, with max_min up to the minutes (still stopped, like any fill job, for a window or a waiter);
  - starts a timing job only on a GPU whose NUMA node holds `cpus` free CPUs of TIMING_CPU_SET in whole cores (SMT siblings),
    and gives it those CPUs alone, at nice 0, in a scope of the top-level TIMING_SLICE (`sudo -n systemd-run`: user.slice and
    system.slice, where every other fill job, slot d and the Verity pool run, are confined below them); never a CPU a fence, a
    running timing job or any populated top-level cgroup but PID 1's may use.  Its lease isn't --preemptible and nothing here
    stops it before its max_min, for a waiter, a window, a Commit or a fence, nor moves its threads (booked windows still bound
    its start).  While one could start, other 1-GPU jobs take the GPUs it couldn't start on first.  The lease is taken
    inside the scope (`sudo -n systemd-run --scope ... -- setpriv ... -- gpu-lease 1 --on G --max-min M --mem-gb 0 -- taskset
    ...`), where a keeper holds it until nothing else is left there: the job is over once its scope is empty, what outlives it
    by a tick is stopped, and a kill of the scope that fails while it is populated is logged (`scope-kill-failed`).
Status: FILL_DIR/status.txt; every start, stop and exit: FILL_DIR/events.jsonl.
A bare `fill_runner.py` is the runner, which exits at once while another holds FILL_DIR/.runner.lock; any argument but `fence`
prints the usage.
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import functools
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
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
# slot d's train checks on 0-47 (top-level, 1 Oct 1:52 AM PDT) run in a login session's cgroup, beside fill's user scopes, so
# nice 19 can't make a CPU job yield to one: a cpu-sets job on those CPUs pauses while a check holds the slot's lock
YIELD_LOCK = Path(os.environ.get("FILL_YIELD_LOCK", "/workspace/research/locks/check-d.lock"))
YIELD_CPUS = os.environ.get("FILL_YIELD_CPUS", "0-47")
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
# the pool's jobs all ran at nice 19 on all of 48-95, where nice orders nothing between their scopes and node 2's queued runs'
# (one suite got 0.9 cores, 19 Oct): each now claims whole cores in the registry the queued runs claim theirs in (research's
# cores.py): CLAIMS/<name>.cpus, live while its job holds it flock'ed, taken under CLAIMS/.lock; CLAIMS/wait.* are the runs
# waiting for cores, which the pool lets go first
CLAIMS = Path(os.environ.get("FILL_CPU_CLAIMS_DIR", "/workspace/research/locks/cpu-claims"))
# the pool lends its idle slots to queued pous CPU jobs (borrowing goes both ways); a Verity job that can't start
# takes a slot back by requeueing the newest lent job (143, as a preempted job)
VERITY_LEND = os.environ.get("FILL_VERITY_LEND", "1") == "1"
COMMIT_PREFIX = "verity-commit-"
# a Commit reruns from the top when stopped, and a Gemma-2 Commit's CPU phases alone outlast 30 min; booked windows still bound it
COMMIT_MAX_MIN = int(os.environ.get("FILL_COMMIT_MAX_MIN", "120"))
CGROUP_ROOT = os.environ.get("FILL_CGROUP_ROOT", "/sys/fs/cgroup")
# what a fence may take, in the tool rather than a budget line: the GPU is paid for either way, so a fence costs fill's throughput
FENCE_MAX_CPUS, FENCE_MAX_H = 8, 4
# network accounting's timing chunks kept none of 353 windows on node 2's shared 48-95 at nice 19 (art:88c665c3): each gets
# whole cores here alone, which only a top-level slice reaches while user.slice and system.slice are confined to 0-123
TIMING_CPU_SET = os.environ.get("FILL_TIMING_CPU_SET", "124-191")
TIMING_SLICE = os.environ.get("FILL_TIMING_SLICE", "vy-timing.slice")
# a chunk is 55 min plus its startup (benchmarks/network_traces/run.sh leases --max-min 60); booked windows still bound it
TIMING_MAX_MIN = int(os.environ.get("FILL_TIMING_MAX_MIN", "60"))
# how far past its max_min a timing job may hold its GPU: gpu-lease's SIGKILL 20 s after it, a tick to see what outlived the
# job, and the stop's 20 s to SIGKILL
TIMING_MARGIN_MIN = 2
TIMING_MAX_CPUS = 16
CPU_ROOT = Path(os.environ.get("FILL_CPU_ROOT", "/sys/devices/system/cpu"))
QUOTA_LOG = Path(os.environ.get("FILL_QUOTA_LOG", "/workspace/verity-guest/kueue-backup/quota-changes.log"))
PCI_ROOT = Path(os.environ.get("FILL_PCI_ROOT", "/sys/bus/pci/devices"))
HELD = "held-unknown-exit"
# runs `bash JOB` and writes its exit status to RC; the trap (a handler, so the job doesn't inherit an ignored signal) keeps the
# wrapper alive through a stop until the job itself exits
RC_WRAP = 'trap : TERM INT HUP; bash "$2"; c=$?; echo "$c" > "$1"; exit "$c"'
# a timing job's lease keeper (`fill-keep CGROUP_ROOT`): it inherits gpu-lease's lock descriptors and holds them while anything
# else is left in its scope, since `research run` starts its workload in a session of its own that outlives the job and
# gpu-lease.  In a session of its own, ignoring TERM, so neither gpu-lease's --max-min nor a stop's SIGTERM ends it before the
# workload; a scope it can't read it holds on to, until fill kills the scope.
KEEP = ('trap "" TERM INT HUP; read -r cg < /proc/$$/cgroup; procs="$1${cg#0::}/cgroup.procs"; '
        'while ! mapfile -t ps < "$procs" || [ "${#ps[@]}" -gt 1 ]; do sleep 1; done')
TIMING_WRAP = 'setsid bash -c "$3" fill-keep "$4" < /dev/null > /dev/null 2>&1 & ' + RC_WRAP

running: dict[str, dict] = {}
backoff_until = 0.0
repinned: set[str] = set()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event(kind: str, job: str, **kw) -> None:
    with open(FILL / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": now(), "event": kind, "job": job, **kw}) + "\n")


def utc(s: str) -> float:
    """`2026-10-02T14:45Z` (UTC when it names no zone) in epoch seconds."""
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp()


def fences(t: float | None = None) -> list[dict]:
    """FILL_DIR/fences, written by `fill_runner.py fence`: `<owner> <gpu> <cpus> <until UTC> [<job glob> <minutes>]  # why`, one
    per owner.  A line past its until is ignored, so a fence ends on its own."""
    t = time.time() if t is None else t
    out = []
    try:
        lines = (FILL / "fences").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        f = line.split("#")[0].split()
        if len(f) not in (4, 6) or not f[1].isdigit() or not re.fullmatch(r"[\d,-]+", f[2]) or len(f) == 6 and not f[5].isdigit():
            continue
        try:
            until = utc(f[3])
        except ValueError:
            continue
        if until > t:
            out.append({"owner": f[0], "gpu": int(f[1]), "cpus": f[2], "until": until, "jobs": f[4] if len(f) == 6 else None,
                        "minutes": int(f[5]) if len(f) == 6 else 0, "line": line})
    return out


def fence_for(name: str, fs: list[dict]) -> dict | None:
    """The fence among `fs` whose job glob names job `name`."""
    return next((f for f in fs if f["jobs"] and fnmatch.fnmatchcase(name, f["jobs"])), None)


def max_min_exception(name: str, t: float | None = None) -> int | None:
    """FILL_DIR/max-min: exceptions to the 30-min GPU cap, each approved by infra, one per line, read at each start:
    `<job name glob> <minutes> <until UTC, 2026-10-02T14:45Z>  # who, why`.  A job it names may set max_min up to those minutes
    and never past the line's until, so the exception ends on its own; a line past its until is ignored.  A job a fence names
    gets the fence's minutes the same way."""
    t = time.time() if t is None else t
    f = fence_for(name, fences(t))
    if f and f["until"] - t >= 60:
        return min(f["minutes"], int((f["until"] - t) // 60))
    try:
        lines = (FILL / "max-min").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        f = line.split("#")[0].split()
        if len(f) != 3 or not f[1].isdigit() or not fnmatch.fnmatchcase(name, f[0]):
            continue
        try:
            left = int((utc(f[2]) - t) // 60)
        except ValueError:
            continue
        if left >= 1:
            return min(int(f[1]), left)
    return None


def timing_owners() -> list[str]:
    """FILL_DIR/timing-owners: the owners whose jobs may be timing jobs, one glob per line (`network-accounting  # who, why`),
    written by infra.  A timing job holds a GPU and whole cores that nothing preempts, so without the file nobody's may."""
    try:
        lines = (FILL / "timing-owners").read_text().splitlines()
    except OSError:
        return []
    return [f[0] for f in (line.split("#")[0].split() for line in lines) if f]


def mem_share(n: int) -> int:
    """gpu-lease's memory cap per GPU, in GB: GPU_LEASE_MEM_GB_PER_GPU, else 90% of MemTotal over the n GPUs."""
    per = os.environ.get("GPU_LEASE_MEM_GB_PER_GPU", "")
    if per.isdigit():
        return int(per)
    try:
        kb = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal:"))
    except (OSError, StopIteration, IndexError, ValueError):
        return CPU_MEM_GB
    return int(kb * 0.9 / 2**20 / max(n, 1))


HEADER = {"owner": "unknown", "gpus": 1, "max_min": 25, "cpus": 8, "project": "pous", "on": None, "prio": 0, "mem_gb": CPU_MEM_GB}


def header(p: Path, t: float | None = None) -> dict:
    """The job's header, normalized.  A value fill reads as a whole number that isn't one (gpus, max_min, prio; a CPU or timing
    job's mem_gb; a timing job's cpus) keeps its default and is named in h["bad"], as is timing=1 from an owner
    FILL_DIR/timing-owners doesn't list: the runner files a queued job so under failed/."""
    h = {k: v for k, v in HEADER.items() if k != "mem_gb"}
    for line in p.read_text(errors="replace").splitlines()[:20]:
        m = re.match(r"#\s*fill:\s*(.*)", line)
        if m:
            for k, v in re.findall(r"(\w+)=(\S+)", m.group(1)):
                h[k] = int(v) if v.isdigit() else v
    bad = []

    def whole(k: str) -> int:
        try:
            return int(h[k])
        except ValueError:
            bad.append(f"{k}={h[k]}")
            return HEADER[k]

    commit = p.name.startswith(COMMIT_PREFIX)
    h["timing"] = h.get("timing") == 1 and not commit
    if h["timing"] and not any(fnmatch.fnmatchcase(str(h["owner"]), g) for g in timing_owners()):
        bad.append(f"timing=1 (owner {h['owner']} isn't in FILL_DIR/timing-owners)")
        h["timing"] = False
    for k in ("gpus", "max_min", "prio"):
        h[k] = whole(k)
    h["gpus"] = 1 if h["timing"] else min(max(h["gpus"], 0), 2 if commit else 1)
    if "mem_gb" in h and (h["timing"] or not h["gpus"]):
        h["mem_gb"] = whole("mem_gb")
    if h["timing"]:
        h["cpus"] = min(max(whole("cpus"), 1), TIMING_MAX_CPUS)
    elif not h["gpus"] and h["project"] == "verity":
        # the pool's job gets that many CPUs of its own, in whole cores
        h["cpus"] = min(max(whole("cpus"), 1), len(gpu_set(VERITY_CPU_SET, 1 << 16) or ()) or 1)
    # a Verity CPU job has its own CPUs and is frozen through windows, so it may run a whole vLLM Build (2-50 min, some 1-4 h)
    cap = max_min_exception(p.name, t)
    if cap is None:
        cap = (COMMIT_MAX_MIN if commit and h["gpus"] else TIMING_MAX_MIN if h["timing"]
               else 360 if h["project"] == "verity" and not h["gpus"] else 30)
    if h["timing"]:
        # nothing stops a timing job before its max_min, so no exception lifts that
        cap = min(cap, TIMING_MAX_MIN)
    h["max_min"] = min(h["max_min"], cap)
    if bad:
        h["bad"] = " ".join(bad)
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


def held(lock: Path) -> bool:
    """Whether someone holds `lock` exclusively (flock)."""
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def agent_up() -> bool:
    """Whether the node's scheduler agent (`cluster agent --mode live`) holds LEASE_DIR/agent.lock."""
    return held(LEASE_DIR / "agent.lock")


def yields(cpus: str | None) -> bool:
    """Whether a cpu-sets job on `cpus` gives way to a train check: its CPUs meet YIELD_CPUS."""
    return bool(cpus) and bool(gpu_set(cpus, 1 << 16) & gpu_set(YIELD_CPUS, 1 << 16))


def waiters() -> list[int]:
    """How many GPUs each waiting `gpu-lease N --wait` wants, read from gpu-lease's own queue.  A request that can't start at once
    queues as LEASE_DIR/wait.<ns>.<pid>, holds that file locked while it waits, and removes it once it has its GPUs, so a lease
    that holds its GPUs (its argv still says --wait) is never a waiter.  A file nobody holds locked is a dead waiter's.  A waiter
    pinned (`on=`) inside FILL_DIR/keep-free waits for GPUs fill never takes, so it holds back no fill."""
    out = []
    keep = keep_free(1 << 16)
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
            text = os.read(fd, 4096).decode(errors="replace")
            m = re.search(r"\bn=(\d+)", text)
            on = re.search(r"\bon=(\S+)", text)
            pinned = gpu_set(on.group(1), 1 << 16) if on else None
            if pinned and pinned <= keep:
                continue
            if m:
                out.append(int(m.group(1)))
        finally:
            os.close(fd)
    return out


def scope(j: dict, *args: str) -> bool:
    """systemctl --user on a job's scope (a Verity or lent CPU job's own, a GPU job's gpu-lease-<pid>), or the system manager's
    on a timing job's fill-timing-*: freeze, thaw and kill reach every process in it, including a workload that `research run`
    starts as its own process group, which killpg misses."""
    unit = j.get("unit")
    systemctl = ["sudo", "-n", "systemctl"] if unit and unit.startswith("fill-timing-") else ["systemctl", "--user"]
    return bool(unit) and subprocess.run([*systemctl, *args, f"{unit}.scope"], capture_output=True).returncode == 0


def scope_populated(j: dict) -> bool:
    """Whether a timing job's scope still holds a process (its cgroup.events); one that can't be read counts as holding one."""
    try:
        return "populated 1" in (Path(CGROUP_ROOT) / TIMING_SLICE / f"{j['unit']}.scope" / "cgroup.events").read_text()
    except FileNotFoundError:
        return False
    except OSError:
        return True


def kill(name: str, j: dict, sig: str) -> None:
    """Signal everything in the job's scope.  Only root's systemctl reaches a timing job's, and what it misses keeps the GPU's
    lease (KEEP), so a kill that fails while that scope is populated is logged, once a job (`scope-kill-failed`)."""
    if not scope(j, "kill", f"--signal={sig}") and j["h"].get("timing") and scope_populated(j) and not j.get("kill_failed"):
        j["kill_failed"] = True
        event("scope-kill-failed", name, unit=j["unit"], signal=sig)


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
    kill(name, j, "SIGTERM")
    j["stopping"], j["stop_at"], j["requeue"], j["why"] = True, time.monotonic(), requeue, why
    event("stop", name, why=why)


def start(p: Path, h: dict, backfill: bool = False, claim_fd: int | None = None) -> None:
    """Start a queued job.  A pool job's h["pool_cpus"] are its claimed cores, whose claim `claim_fd` holds: the job inherits it,
    and the runner's own copy is closed once the job is started."""
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
    pool_cpus = h.get("pool_cpus") if in_pool else None
    cpus = own or pool_cpus or (VERITY_CPU_SET if in_pool else CPU_SET)
    fs = fences()
    mine = fence_for(name, fs)
    fenced = set().union(*(gpu_set(f["cpus"], 1 << 16) or set() for f in fs))
    # taskset needs a CPU: a set the fences would empty (a cpu-sets line of a few cores) keeps its own
    cpus = mine["cpus"] if mine else cpu_list((gpu_set(cpus, 1 << 16) or set()) - fenced) or cpus
    rc_file = FILL / "running" / f".{name}.rc"
    rc_file.unlink(missing_ok=True)
    cmd = [*([] if pool_cpus else ["nice", "-n", "19"]), "taskset", "-c", cpus, "bash", "-c", RC_WRAP, "fill-rc", str(rc_file), str(run)]
    if own:
        cmd = ["ionice", "-c3", *cmd]
    if CPU_MEMBIND and not h["gpus"] and not verity:
        cmd = ["numactl", f"--membind={CPU_MEMBIND}", *cmd]
    timing = h.get("timing_cpus")
    if timing:
        # root's systemd-run makes the scope in a top-level slice at nice 0 and runs gpu-lease in it as this user, with this user's
        # groups (setpriv; sudo leaves USER and LOGNAME root's): the lease's locks are taken inside the scope, and KEEP holds
        # them there until nothing else is left in it.  --mem-gb 0: the scope's MemoryMax is the cap, and a user scope of
        # gpu-lease's own would take the job out of this one.
        unit_name = f"fill-timing-{int(time.time() * 1000)}"
        # one compute thread per core in each pool `research run --queue --cpus 8` set for the chunks on node 1; the SMT siblings
        # are headroom
        cores = str(len({siblings(c) for c in gpu_set(timing, 1 << 16) or ()}) or h["cpus"])
        env.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), cores))
        me = pwd.getpwuid(os.getuid())
        cmd = ["sudo", "-n", "--preserve-env", "env", f"PATH={env.get('PATH', '')}", f"HOME={me.pw_dir}", f"USER={me.pw_name}",
               f"LOGNAME={me.pw_name}", "systemd-run", "--quiet", "--scope", "--collect", f"--slice={TIMING_SLICE}",
               "--unit", unit_name, "--nice=0", "-p", f"AllowedCPUs={timing}", "-p", f"MemoryMax={h.get('mem_gb', CPU_MEM_GB)}G",
               "-p", "MemorySwapMax=0", "--",
               "setpriv", f"--reuid={me.pw_uid}", f"--regid={os.getgid()}", "--init-groups", "--",
               GPU_LEASE, "1", "--on", str(h["on"]), "--max-min", str(h["max_min"]), "--mem-gb", "0", "--",
               "taskset", "-c", timing, "bash", "-c", TIMING_WRAP, "fill-rc", str(rc_file), str(run), KEEP, CGROUP_ROOT]
    elif h["gpus"]:
        pin = ["--on", str(h["on"])] if h.get("on") is not None else []
        cmd = [GPU_LEASE, str(h["gpus"]), *pin, "--preemptible", "--max-min", str(h["max_min"]), "--", *cmd]
    elif shutil.which("systemd-run"):
        kind = "cpuset" if own else "verity" if verity else "lent"
        unit_name = f"fill-{kind}-{int(time.time() * 1000)}" if in_pool or own else None
        unit = ["--unit", unit_name] if unit_name else []
        held_cpus = ["-p", f"AllowedCPUs={cpus}"] if pool_cpus else []
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", *unit, "-p", f"MemoryMax={h.get('mem_gb', CPU_MEM_GB)}G",
               "-p", "MemorySwapMax=0", *held_cpus, "--", *cmd]
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(FILL), start_new_session=True,
                                pass_fds=() if claim_fd is None else (claim_fd,))
    finally:
        if claim_fd is not None:
            os.close(claim_fd)
    log.close()
    if h["gpus"] and not timing:
        unit_name = f"gpu-lease-{proc.pid}"         # gpu-lease runs CMD in the scope named by its own pid
    (FILL / "running" / f".{name}.pgid").write_text(f"{proc.pid}\n")
    if unit_name:
        (FILL / "running" / f".{name}.unit").write_text(f"{unit_name}\n")
    if timing:
        (FILL / "running" / f".{name}.cpus").write_text(f"{timing}\n")
    running[name] = {"pid": proc.pid, "proc": proc, "h": h, "t0": time.monotonic(), "started": now(), "paused": False,
                     "unit": unit_name}
    event("start", name, owner=h["owner"], gpus=h["gpus"], project=h["project"], backfill=backfill,
          **({"timing_cpus": timing} if timing else {}), **({"cpus": cpus} if pool_cpus else {}))


def gpu_set(on, n: int) -> set[int] | None:
    if on is None:
        return None
    out = set()
    for part in str(on).split(","):
        lo, _, hi = part.partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            out.update(i for i in range(int(lo), int(hi or lo) + 1) if i < n)
    return out


def pick_gpus(idle: list[int], allowed: set[int] | None, g: int, n: int) -> list[int] | None:
    """g of the idle GPUs (highest index first) for one job: two from one NUMA node (0..n/2-1, n/2..n-1) when they can be."""
    ok = [i for i in reversed(idle) if allowed is None or i in allowed]
    if len(ok) < g:
        return None
    for upper in (True, False):
        same = [i for i in ok if (i >= n // 2) == upper]
        if len(same) >= g:
            return sorted(same[:g])
    return sorted(ok[:g])


def keep_free(n: int, fs: list[dict] | None = None) -> set[int]:
    """The GPUs fill keeps its jobs off: FILL_DIR/keep-free's, and those of the fences `fs` (by default every live one), where
    only a fence's own jobs run."""
    fs = fences() if fs is None else fs
    try:
        kept = gpu_set((FILL / "keep-free").read_text().split("#")[0].strip() or None, n) or set()
    except OSError:
        kept = set()
    return kept | {f["gpu"] for f in fs if f["gpu"] < n}


def cpu_list(cpus: set[int]) -> str:
    """{96, 97, 98, 100} as `96-98,100`."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and runs[-1][1] == c - 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def gpu_cpus(i: int) -> set[int]:
    """The CPUs on GPU i's NUMA node: its PCI device's local_cpulist."""
    bus = subprocess.run(["nvidia-smi", "-i", str(i), "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout.strip()
    domain, rest = bus.split(":", 1)
    return gpu_set((PCI_ROOT / f"{domain[-4:]}:{rest}".lower() / "local_cpulist").read_text().strip(), 1 << 16) or set()


_local: dict[int, set[int]] = {}


def local_cpus(i: int) -> set[int]:
    """gpu_cpus(i), read once a runner; none while it can't be read."""
    if i not in _local:
        try:
            _local[i] = gpu_cpus(i)
        except (OSError, ValueError, subprocess.SubprocessError):
            return set()
    return _local[i]


@functools.cache
def siblings(c: int) -> frozenset[int]:
    """CPU c's core: its SMT siblings (CPU_ROOT/cpu<c>/topology/thread_siblings_list); none when that can't be read."""
    try:
        return frozenset(gpu_set((CPU_ROOT / f"cpu{c}" / "topology" / "thread_siblings_list").read_text().strip(), 1 << 16) or ())
    except OSError:
        return frozenset()


def carve(free: set[int], local: set[int], k: int) -> str | None:
    """At least k CPUs of `free` on `local` (a GPU's NUMA node), lowest first, or None when they don't fit.  A core goes whole,
    all its SMT siblings or none, so no two jobs share one."""
    ok, got = free & local, set()
    for c in sorted(ok):
        core = siblings(c)
        if len(got) < k and c == min(core, default=-1) and core <= ok:
            got |= core
    return cpu_list(got) if len(got) >= k else None


def timing_free(fs: list[dict]) -> set[int]:
    """TIMING_CPU_SET less every CPU another process may run on: a fence's, a running timing job's, and each populated top-level
    cgroup's but PID 1's init.scope (user.slice and system.slice hold every other fill job, slot d and the Verity pool): a
    slice's own cpuset, else that of each populated cgroup in it, as in TIMING_SLICE.  A cgroup that can't be read takes all."""
    free = gpu_set(TIMING_CPU_SET, 1 << 16) or set()
    for cpus in [f["cpus"] for f in fs] + [j["h"].get("timing_cpus") for j in running.values()]:
        free -= gpu_set(cpus, 1 << 16) or set()

    def populated(d: Path) -> bool:
        return "populated 1" in (d / "cgroup.events").read_text()

    try:
        for d in Path(CGROUP_ROOT).iterdir():
            if not d.is_dir() or d.name == "init.scope" or not populated(d):
                continue
            own = d.name != TIMING_SLICE and (d / "cpuset.cpus").read_text().strip()
            for c in [d] if own else [c for c in d.iterdir() if c.is_dir() and populated(c)] or [d]:
                free -= gpu_set((c / "cpuset.cpus.effective").read_text().strip(), 1 << 16) or set()
    except OSError:
        return set()
    return free


def repin(f: dict, fs: list[dict]) -> None:
    """Move every running fill job but fence f's own and the timing jobs off f's cores: each thread in the job's process group or
    its scope (a GPU job's gpu-lease-<pid>, where `research run` starts its workload in a session of its own).  A thread on f's
    cores alone stays."""
    off = gpu_set(f["cpus"], 1 << 16) or set()
    jobs = [j for k, j in running.items() if fence_for(k, fs) is not f and not j["h"].get("timing")]
    groups = {j["pid"] for j in jobs if j.get("pid")}
    scopes = tuple(f"/{j['unit']}.scope" for j in jobs if j.get("unit"))
    moved = 0
    for d in Path("/proc").glob("[0-9]*"):
        try:
            pgid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[2])
            cg = (d / "cgroup").read_text().strip()
        except (OSError, IndexError, ValueError):
            continue
        if pgid not in groups and not (scopes and cg.endswith(scopes)):
            continue
        for task in d.glob("task/[0-9]*"):
            try:
                had = os.sched_getaffinity(int(task.name))
                if had & off and had - off:
                    os.sched_setaffinity(int(task.name), had - off)
                    moved += 1
            except OSError:
                pass
    event("fence", "-", owner=f["owner"], gpu=f["gpu"], cpus=f["cpus"], jobs=f["jobs"], threads_moved=moved)


def rank(name: str, h: dict) -> int:
    """Node 2's order: circuits' Commit guests, then pous, then proofs and the rest of Verity's jobs."""
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
    """Whether a job started at `t` ends, at its max_min (a timing job, which no window stops, TIMING_MARGIN_MIN after it),
    before every booked window, or starts after it."""
    end = t + 60 * (h["max_min"] + (TIMING_MARGIN_MIN if h.get("timing") else 0))
    return all(end <= s or t >= e for s, e in booked)


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


def exit_status(name: str) -> int | None:
    """The status a job's wrapper wrote as it exited (RC_WRAP), or None when there is none."""
    try:
        return int((FILL / "running" / f".{name}.rc").read_text().strip())
    except (OSError, ValueError):
        return None


def reap() -> None:
    global backoff_until
    for name, j in list(running.items()):
        if j["proc"] is None:                       # adopted from an earlier runner: its status is only what its wrapper wrote
            try:
                os.killpg(j["pid"], 0)
                rc = None
            except ProcessLookupError:
                rc = exit_status(name)
                rc = "unknown" if rc is None else rc
        else:
            rc = j["proc"].poll()
        if rc is not None and j["h"].get("timing") and scope_populated(j):
            # KEEP holds the GPU's lease while anything is left in the scope, so the job isn't over until it's empty; what is
            # still there a tick after the job ended is stopped
            rc = None
            j["left_ticks"] = j.get("left_ticks", 0) + 1
            if j["left_ticks"] >= 2:
                stop(name, "a process outlived its job", requeue=False)
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
                kill(name, j, "SIGKILL")
            continue
        if scope(j, "kill", "--signal=SIGKILL"):
            event("scope-residue", name, unit=j["unit"])
        run = FILL / "running" / name
        mins = round((time.monotonic() - j["t0"]) / 60, 1)
        paused_min = round((j.get("paused_s", 0.0) + (time.monotonic() - j["paused_at"] if j["paused"] else 0.0)) / 60, 1)
        for side in ("pgid", "unit", "rc", "cpus"):
            (FILL / "running" / f".{name}.{side}").unlink(missing_ok=True)
        if rc == 0:
            dest, kind = "done", "done"
        elif j.get("stopping") and j.get("requeue"):
            dest, kind = "queue", "requeued"
        elif rc == "unknown":
            dest, kind = HELD, "unknown-exit"
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


def claim(name: str, k: int) -> tuple[str, int] | None:
    """k CPUs of VERITY_CPU_SET in whole cores that no fence and no live claim in CLAIMS holds, claimed for job `name`: the CPUs
    and the descriptor that holds the claim (start passes it to the job).  None while they don't fit, or while a queued run waits
    for cores (a live CLAIMS/wait.*), which goes first; a claim or ticket nobody holds is a leftover, removed."""
    try:
        CLAIMS.mkdir(parents=True, exist_ok=True)
        with open(CLAIMS / ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            free = gpu_set(VERITY_CPU_SET, 1 << 16) or set()
            free -= set().union(*(gpu_set(f["cpus"], 1 << 16) or set() for f in fences()))
            waiting = False
            for f in sorted(CLAIMS.iterdir()):
                if not (f.name.startswith("wait.") or f.name.endswith(".cpus")):
                    continue
                if not held(f):
                    f.unlink(missing_ok=True)
                elif f.name.startswith("wait."):
                    waiting = True
                else:
                    free -= gpu_set(f.read_text().strip(), 1 << 16) or set()
            got = None if waiting else carve(free, free, k)
            if got is None:
                return None
            # written and locked under a name no reader looks at, then renamed into place, so nobody takes it for a leftover;
            # a name of its own each start, since a requeued job's last claim may still be held
            path = CLAIMS / f"fill-{time.time_ns()}-{name}.cpus"
            tmp = CLAIMS / f".tmp-{os.getpid()}-{path.name}"
            fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o664)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.write(fd, f"{got}\n".encode())
                os.replace(tmp, path)
            except OSError:
                os.close(fd)
                raise
            return got, fd
    except OSError as e:
        event("claim-failed", name, error=str(e)[:200])
        return None


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
        got = claim(p.name, int(h["cpus"])) if slots > 0 and mem + need <= VERITY_MEM_TOTAL_GB else None
        if got is None:
            held = True
            continue
        start(p, {**h, "mem_gb": need, "pool_cpus": got[0]}, claim_fd=got[1])
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
        got = claim(p.name, h["cpus"] if isinstance(h["cpus"], int) and h["cpus"] > 0 else HEADER["cpus"])
        if got is None:
            break
        start(p, {**h, "mem_gb": need, "lent": True, "pool_cpus": got[0]}, claim_fd=got[1])
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
    # a timing job isn't fill's to stop: until its max_min it holds its GPU as any other lease does
    gpu_jobs = sorted((j["t0"], k) for k, j in running.items()
                      if j["h"]["gpus"] and not j.get("stopping") and not j["h"].get("timing"))
    blockers = (n - free) - sum(running[k]["h"]["gpus"] for _, k in gpu_jobs)
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
            deficit -= running[k]["h"]["gpus"]
    fs = fences()
    for f in fs:
        if f["line"] not in repinned:
            repinned.add(f["line"])
            repin(f, fs)
        pid = re.search(r"\bpid=(\d+)", leases[f["gpu"]] or "") if f["gpu"] < n else None
        k = next((k for k, j in running.items() if pid and j.get("pid") == int(pid.group(1))), None)
        if k and not running[k].get("stopping") and fence_for(k, fs) is not f and not running[k]["h"].get("timing"):
            stop(k, f"GPU {f['gpu']} is fenced for {f['owner']}")
    frozen = cpu_frozen(timed, window)
    if frozen and any(not j["h"]["gpus"] and not j.get("stopping") and not j["paused"] for j in running.values()):
        event("freeze-memory", "-", timed=timed, waiting=window, **numa0_gb())
    checking = held(YIELD_LOCK)
    for k, j in running.items():
        if not j["h"]["gpus"] and not j.get("stopping"):
            hold = frozen or (checking and yields(j["h"].get("cpuset")))
            try:
                if hold and not j["paused"]:
                    scope(j, "freeze")
                    os.killpg(j["pid"], signal.SIGSTOP); j["paused"], j["paused_at"] = True, time.monotonic(); event("pause", k)
                elif not hold and j["paused"]:
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
                h = header(p)
                if h.get("bad"):
                    shutil.move(str(p), FILL / "failed" / p.name)
                    event("failed", p.name, owner=h["owner"], why=f"header: {h['bad']}")
                    continue
                queued.append((p, h, p.stat().st_mtime))
            except OSError:
                pass
    started = False
    # a window blocked by another lease: fill backfills, admitted by gpu-lease --max-min only if it ends before the window can start
    backfill = window and blockers > 0
    if not timed and (not want or backfill) and time.monotonic() >= backoff_until:
        # a job pinned with on=<index> waits for that GPU (a busy one would only bounce off gpu-lease)
        keep, fenced = keep_free(n, fs), {f["gpu"] for f in fs}
        booked, t_now = windows(), wall()
        idle = [i for i, l in enumerate(leases) if l is None and (i not in keep or i in fenced)]
        timing = any(x[1].get("timing") for x in queued)
        pool = timing_free(fs) if timing else set()

        def allowed(x) -> set[int]:
            f = fence_for(x[0].name, fs)
            gpus = {f["gpu"]} if f else set(range(n)) - fenced
            on = gpu_set(x[1].get("on"), n)
            gpus = gpus if on is None else gpus & on
            return {i for i in gpus if carve(pool, local_cpus(i), x[1]["cpus"])} if x[1].get("timing") else gpus

        ready = [x for x in queued if x[1]["gpus"] and clears(x[1], booked, t_now) and any(i in idle for i in allowed(x))]
        # the GPUs a timing job could start on now, which the rest take last
        near = set().union(*(allowed(x) for x in ready if x[1].get("timing")))
        # nothing else may use a fence's GPU, so its jobs go first
        theirs = [x for x in ready if fence_for(x[0].name, fs)]
        by_rank = [[x for x in ready if x not in theirs and rank(x[0].name, x[1]) == r] for r in range(3)]
        left = theirs + by_rank[0] + (by_rank[1] or (by_rank[2] if free >= 2 else []))
        counts = Counter(j["h"]["owner"] for j in running.values() if j["h"]["gpus"])
        # each job gets its own free GPUs (--on), so jobs started in one tick don't take each other's
        slots = max(0, len(idle) - RESERVE)
        while left and slots > 0:
            best = next_job(left, counts)
            left.remove(best)
            p, h, _ = best
            mine = allowed(best)
            # while a timing job could start, the rest take the GPUs it couldn't start on first
            first = mine - near if near and not h.get("timing") and h["gpus"] == 1 else mine
            got = (pick_gpus(idle, first, h["gpus"], n) or pick_gpus(idle, mine, h["gpus"], n)) if h["gpus"] <= slots else None
            if not got:
                continue
            for i in got:
                idle.remove(i)
            if h.get("timing"):
                h = {**h, "timing_cpus": carve(pool, local_cpus(got[0]), h["cpus"]),
                     "mem_gb": min(int(h.get("mem_gb", CPU_MEM_GB)), mem_share(n))}
                pool -= gpu_set(h["timing_cpus"], 1 << 16) or set()
            start(p, {**h, "on": ",".join(map(str, got))}, backfill=backfill)
            counts[h["owner"]] += 1
            slots -= h["gpus"]
            started = True
        if not window:
            # Commits that found too few free GPUs take the newest proofs (then pous) fills' GPUs, counting GPUs already freeing
            waiting = sum(h["gpus"] for p, h, _ in queued if h["gpus"] and rank(p.name, h) == 0 and p.exists() and clears(h, booked, t_now))
            short = (waiting - sum(1 for i in idle if i not in fenced)
                     - sum(j["h"]["gpus"] for j in running.values() if j["h"]["gpus"] and j.get("stopping")))
            victims = sorted((k for k, j in running.items() if j["h"]["gpus"] and not j.get("stopping") and rank(k, j["h"])
                              and not fence_for(k, fs) and not j["h"].get("timing")),
                             key=lambda k: (-rank(k, running[k]["h"]), -running[k]["t0"]))
            for k in victims:
                if short <= 0:
                    break
                stop(k, "a Commit reclaims its GPU")
                short -= running[k]["h"]["gpus"]
    if not timed:
        sets = cpu_sets()
        own = [x for x in queued if not x[1]["gpus"] and x[1]["project"] != "verity" and x[1]["owner"] in sets]
        rest = [x for x in queued if x not in own]
        for owner, s in sets.items():
            if checking and yields(s["cpus"]):
                continue
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
    lines = [f"{now()} free GPUs {free}/{n}, kept free {sorted(keep_free(n, fs)) or '-'}, waiters {want or '-'}, timed {timed}, "
             f"window waiting {window}, "
             f"backfill {backfill}, queued {len(queued)} "
             f"(gpu {sum(1 for x in queued if x[1]['gpus'])}, cpu {sum(1 for x in queued if not x[1]['gpus'])}), "
             f"done {sum(1 for _ in (FILL / 'done').glob('*'))}, failed {sum(1 for _ in (FILL / 'failed').glob('*'))}, "
             f"held for an unknown exit {sum(1 for _ in (FILL / HELD).glob('*'))}"]
    for k, j in running.items():
        lines.append(f"  running {k} owner={j['h']['owner']} gpus={j['h']['gpus']} since={j['started']}"
                     f"{' timing cpus=' + j['h']['timing_cpus'] if j['h'].get('timing_cpus') else ''}"
                     f"{' cpus=' + j['h']['pool_cpus'] if j['h'].get('pool_cpus') else ''}"
                     f"{' PAUSED' if j['paused'] else ''}{' stopping: ' + j['why'] if j.get('stopping') else ''}")
    lines.extend(f"  fence {f['line'].split('#')[0].strip()}" for f in fs)
    (FILL / "status.txt").write_text("\n".join(lines) + "\n")
    return lines[0]


def adopt() -> None:
    """Take over the jobs an earlier runner left in running/."""
    for p in (FILL / "running").glob("*"):
        if p.name.startswith("."):
            continue
        side = FILL / "running" / f".{p.name}.pgid"
        try:
            pgid = int(side.read_text().strip())
        except (OSError, ValueError):
            shutil.move(str(p), FILL / HELD / p.name)
            event("unknown-exit", p.name, why="no process group recorded")
            continue
        # a job that ended while no runner ran is adopted too: the first tick files it by its wrapper's status
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
        if h["timing"] or unit and unit.startswith("fill-timing-"):
            # its unit says what it is, whatever its header or FILL_DIR/timing-owners says now
            h["timing"], h["gpus"] = True, 1
            # CPUs it may hold that no record names are all taken, so no later job, nor a fence, shares them
            cpus = FILL / "running" / f".{p.name}.cpus"
            try:
                h["timing_cpus"] = cpus.read_text().strip() or TIMING_CPU_SET
            except OSError:
                h["timing_cpus"] = TIMING_CPU_SET
            cpus.write_text(f"{h['timing_cpus']}\n")
        if h["gpus"] and not unit and not h["timing"]:
            unit = f"gpu-lease-{pgid}"
        running[p.name] = {"pid": pgid, "proc": None, "h": h, "t0": time.monotonic(), "started": now(),
                           "paused": stopped, "paused_at": time.monotonic(), "unit": unit}
        event("adopted", p.name, pgid=pgid)


def fence(argv: list[str], t: float | None = None) -> int:
    """`fill_runner.py fence OWNER --gpu I --cpus LIST --until UTC [--jobs GLOB --max-min M] --why TEXT` writes OWNER's fence
    (replacing the one it has), `fence OWNER --release` removes it; either is logged in QUOTA_LOG, and the runner acts on it at
    its next tick."""
    ap = argparse.ArgumentParser(prog="fill_runner.py fence", description=fence.__doc__)
    ap.add_argument("owner")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--cpus", help=f"at most {FENCE_MAX_CPUS}, on the GPU's NUMA node: 116-123 or 116,118")
    ap.add_argument("--until", help=f"UTC, at most {FENCE_MAX_H} h from now: 2026-10-02T23:00Z")
    ap.add_argument("--jobs", help="a fill job name glob whose jobs run on the GPU and cores (fill jobs still: a window or a "
                                   "waiter stops them)")
    ap.add_argument("--max-min", type=int, help="the --jobs jobs' max_min cap, to --until at most")
    ap.add_argument("--why")
    ap.add_argument("--release", action="store_true")
    a = ap.parse_args(argv)
    t = time.time() if t is None else t

    def refuse(why: str) -> int:
        print(f"fill_runner.py fence: {why}", file=sys.stderr)
        return 2

    if not re.fullmatch(r"[^\s#]+", a.owner):
        return refuse("OWNER is one word without #")
    with open(FILL / ".fences.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = fences(t)
        others = [f for f in live if f["owner"] != a.owner]
        old = next((f for f in live if f["owner"] == a.owner), None)
        if a.release:
            if not old:
                return refuse(f"{a.owner} has no fence")
            line, note = None, f"fence released, was {old['line'].split('#')[0].strip()}"
        else:
            if None in (a.gpu, a.cpus, a.until, a.why):
                return refuse("--gpu, --cpus, --until and --why are required")
            if (a.jobs is None) != (a.max_min is None):
                return refuse("--jobs and --max-min go together")
            try:
                until = utc(a.until)
            except ValueError:
                return refuse(f"--until {a.until!r} is not a UTC time like 2026-10-02T23:00Z")
            if not t < until <= t + 3600 * FENCE_MAX_H:
                return refuse(f"--until must be in the next {FENCE_MAX_H} h")
            n = gpu_count()
            if not 0 <= a.gpu < n:
                return refuse(f"this node's GPUs are 0-{n - 1}")
            if a.gpu in keep_free(n, []):
                return refuse(f"GPU {a.gpu} is in FILL_DIR/keep-free")
            cpus = gpu_set(a.cpus, 1 << 16) if re.fullmatch(r"[\d,-]+", a.cpus) else None
            if not cpus or len(cpus) > FENCE_MAX_CPUS:
                return refuse(f"--cpus {a.cpus!r}: 1 to {FENCE_MAX_CPUS} cores")
            local = gpu_cpus(a.gpu)
            if not cpus <= local:
                return refuse(f"cores {cpu_list(cpus - local)} aren't on GPU {a.gpu}'s NUMA node ({cpu_list(local)})")
            for f in others:
                theirs = gpu_set(f["cpus"], 1 << 16) or set()
                if f["gpu"] == a.gpu or cpus & theirs:
                    return refuse(f"{f['owner']} has GPU {f['gpu']} and cores {f['cpus']} until "
                                  f"{datetime.fromtimestamp(f['until'], timezone.utc):%H:%MZ}")
            for p in (FILL / "running").glob(".*.cpus"):
                try:
                    theirs = gpu_set(p.read_text().strip(), 1 << 16) or set()
                except OSError:
                    continue
                if cpus & theirs:
                    return refuse(f"timing job {p.name[1:-len('.cpus')]} has cores {cpu_list(theirs)} until its max_min")
            if a.jobs is not None and (not re.fullmatch(r"[^\s#]+", a.jobs) or not 1 <= a.max_min <= 60 * FENCE_MAX_H):
                return refuse(f"--jobs is one glob without #, --max-min 1 to {60 * FENCE_MAX_H}")
            end = datetime.fromtimestamp(until, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            jobs = f" {a.jobs} {a.max_min}" if a.jobs is not None else ""
            line = f"{a.owner} {a.gpu} {cpu_list(cpus)} {end}{jobs}  # {' '.join(a.why.split())}"
            note = f"fence {'(replaces its last) ' if old else ''}{line}"
            for s, e in windows():
                if s < until and e > t:
                    print(f"note: the window booked at {datetime.fromtimestamp(s, timezone.utc):%H:%MZ} stops every fill job "
                          f"while it runs, the fence's too")
        with open(QUOTA_LOG, "a") as log:
            log.write(f"{now()} {a.owner} (fill_runner.py fence): {note}\n")
        tmp = FILL / ".fences.tmp"
        tmp.write_text("".join(f"{x}\n" for x in [*(f["line"] for f in others), *([line] if line else [])]))
        os.replace(tmp, FILL / "fences")
    print(f"{note}\nthe fill runner acts on it within {POLL_S:.0f} s")
    return 0


USAGE = """usage: fill_runner.py                 the runner: one per node, held by FILL_DIR/.runner.lock
       fill_runner.py fence OWNER ...   write or release OWNER's fence (`fill_runner.py fence --help`)"""


def main() -> None:
    FILL.mkdir(parents=True, exist_ok=True)
    # held for the process's life; not inherited (PEP 446), so jobs that outlive a runner leave the next one free to adopt them
    lock = open(FILL / ".runner.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"fill_runner.py: another runner holds {FILL / '.runner.lock'}; one runs per node")
    for d in ("queue", "running", "done", "failed", "logs", HELD):
        (FILL / d).mkdir(parents=True, exist_ok=True)
    adopt()
    n = gpu_count()
    while True:
        try:
            tick(n)
        except Exception as e:  # noqa: BLE001
            event("runner-error", "-", error=str(e)[:300])
        time.sleep(POLL_S)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fence"]:
        sys.exit(fence(sys.argv[2:]))
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(USAGE)
        sys.exit(0)
    if sys.argv[1:]:
        sys.exit(f"{USAGE}\nfill_runner.py: unknown arguments {' '.join(sys.argv[1:])!r}")
    main()
