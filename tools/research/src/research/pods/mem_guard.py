"""Laptop memory guard for orchestrated lanes (v4: jobs are charged as a whole, see lane_run.py).

Polls every few seconds. Lane processes (python/pytest/cargo launched from a /private/tmp
worktree or this repo) are charged to the lane_run job they belong to (process group, else
ppid ancestry); a job whose *total* RSS exceeds 1.25x its declaration or --per-proc GB is
killed as a whole. An unregistered lane process over --unmanaged GB is killed. When the
machine's available memory (free+inactive+speculative+purgeable) drops under --min-free GB
and memory_pressure agrees, the largest lane process is killed — this emergency rule stays
because 8 GB of declarations permit ~10 GB of actual use under the 1.25x rule, before Cursor
and other applications (24 GB machine). Cursor, browsers and other user apps are never
touched. Every action is appended to the log with the full command line so the owning lane
can be told what happened and why.

Run:  python -m research.pods.mem_guard --per-proc 3.5 --min-free 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

LANE_DIRS = ("/private/tmp/ctr-", "/private/tmp/lane-", os.path.expanduser("~/projects/veritor"), os.path.expanduser("~/projects/verity"))
LANE_CMDS = ("python", "pytest", "cargo", "rustc", "veritor-zk-host", "node")
PROTECT = ("budget_cap.py", "mem_guard.py", "runpod.py", "caffeinate", "lane_run.py")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def page_size() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    return int(m.group(1)) if m else 16384


def free_gb(ps: int) -> float:
    """Available memory the way macOS accounts it: free + inactive (file cache, reclaimable) + speculative + purgeable.

    The first version counted only ``Pages free`` and killed 0.5 GB test runs while 9.6 GB of file cache sat in
    ``inactive`` and ``memory_pressure`` reported 79 % free (2026-09-10 19:55-20:01Z, four false kills)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            vals[k.strip()] = int(v.strip().rstrip(".").replace(".", "") or 0)
    avail = (vals.get("Pages free", 0) + vals.get("Pages inactive", 0) + vals.get("Pages speculative", 0)
             + vals.get("Pages purgeable", 0))
    return avail * ps / 1e9


def pressure_free_pct() -> float | None:
    """`memory_pressure`'s system-wide free percentage (the kernel's own view); None if unavailable."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def cwd_of(pid: int) -> str:
    out = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("n"):
            return line[1:]
    return ""


PGID: dict[int, int] = {}  # pid -> pgid, refreshed by lane_procs() each poll


def lane_procs() -> list[tuple[int, float, str]]:
    out = subprocess.run(["ps", "-axo", "pid=,pgid=,rss=,args="], capture_output=True, text=True).stdout
    procs = []
    PGID.clear()
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, pgid, rss_kb, args = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
        PGID[pid] = pgid
        if pid == os.getpid() or any(p in args for p in PROTECT):
            continue
        exe = args.split()[0]
        if not any(c in os.path.basename(exe) for c in LANE_CMDS) and not any(c in args.split()[0] for c in LANE_CMDS):
            continue
        if "Cursor" in args or "ChatGPT" in args or "Codex" in args:
            continue
        in_lane = any(d in args for d in LANE_DIRS)
        if not in_lane and rss_kb > 512 * 1024:  # only pay for lsof on big ones
            in_lane = any(cwd_of(pid).startswith(d) for d in LANE_DIRS)
        if in_lane:
            procs.append((pid, rss_kb / 1048576, args[:300]))
    return procs


LAUNCH = os.environ.get("VERITOR_LAUNCH_DIR") or os.path.expanduser("~/.runpod/launch")


def ancestors(pid: int, depth: int = 4) -> list[int]:
    out = []
    for _ in range(depth):
        try:
            pp = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            pid = int(pp)
        except Exception:
            break
        if pid <= 1:
            break
        out.append(pid)
    return out


def slots() -> dict[int, dict]:
    """Live lane_run registrations: pid -> slot (declared gb, class, lane, command hash)."""
    d = os.path.join(LAUNCH, "slots")
    res: dict[int, dict] = {}
    if not os.path.isdir(d):
        return res
    for name in os.listdir(d):
        try:
            with open(os.path.join(d, name)) as f:
                s = json.load(f)
            res[int(s["pid"])] = s
        except Exception:
            continue
    return res


def managed_slot(pid: int, reg: dict[int, dict]) -> dict | None:
    """The lane_run slot a process belongs to: same process group as the job (lane_run v2 starts the job in its own
    session, so every descendant inherits ``pgid``), else the registrant / child pid itself or an ancestor of it."""
    pg = PGID.get(pid)
    if pg is not None:
        for s in reg.values():
            if s.get("pgid") is not None and int(s["pgid"]) == pg:
                return s
    if pid in reg:
        return reg[pid]
    by_child = {int(s["child_pid"]): s for s in reg.values() if s.get("child_pid")}
    if pid in by_child:
        return by_child[pid]
    for a in ancestors(pid):
        if a in reg:
            return reg[a]
        if a in by_child:
            return by_child[a]
    return None


def job_usage(procs: list[tuple[int, float, str]], reg: dict[int, dict]) -> tuple[dict[int, tuple[float, list[int]]], list[tuple[int, float, str]]]:
    """Charge every lane process to its job. Returns ({registrant pid: (total GB, [member pids])}, unmanaged procs)."""
    jobs: dict[int, tuple[float, list[int]]] = {}
    unmanaged: list[tuple[int, float, str]] = []
    for pid, gb, args in procs:
        s = managed_slot(pid, reg)
        if s is None:
            unmanaged.append((pid, gb, args))
            continue
        key = int(s["pid"])
        tot, members = jobs.get(key, (0.0, []))
        jobs[key] = (tot + gb, members + [pid])
    return jobs, unmanaged


def kill_job(slot: dict, members: list[int], why: str, log) -> None:
    """Kill the whole job (process group, then any member that left it) and record it so lane_run refuses a retry."""
    pg = slot.get("pgid")
    if pg:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(int(pg), sig)
            except ProcessLookupError:
                break
            time.sleep(2)
    for pid in members:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    args = " ".join(slot.get("cmd") or [])
    mark_killed(slot, args, why)
    log.write(f"{now()} KILLED-JOB registrant={slot.get('pid')} pgid={pg} members={members} {why} :: {args[:300]}\n")
    log.flush()


def mark_killed(slot: dict | None, args: str, why: str) -> None:
    """Record the kill so lane_run refuses an automatic retry of the same command."""
    try:
        os.makedirs(os.path.join(LAUNCH, "killed"), exist_ok=True)
        h = slot["hash"] if slot else hashlib.sha1(args.encode()).hexdigest()
        with open(os.path.join(LAUNCH, "killed", h), "w") as f:
            f.write(f"{now()} {why} :: {args[:200]}\n")
    except Exception:
        pass


def kill(pid: int, why: str, args: str, log, slot: dict | None = None) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        return
    mark_killed(slot, args, why)
    log.write(f"{now()} KILLED pid={pid} {why} :: {args}\n")
    log.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-proc", type=float, default=3.5, help="GB RSS per lane process")
    ap.add_argument("--min-free", type=float, default=2.0, help="GB available (free+inactive+speculative+purgeable) below which the largest lane process is killed")
    ap.add_argument("--min-pct", type=float, default=15.0, help="memory_pressure free percentage below which the min-free rule may fire")
    ap.add_argument("--min-kill", type=float, default=1.5, help="GB RSS a lane process must have to be killed under the min-free rule")
    ap.add_argument("--unmanaged", type=float, default=1.0, help="GB RSS above which a lane process without a lane_run slot is killed")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--log", default=os.path.expanduser("~/.runpod/mem-guard.log"))
    a = ap.parse_args()
    ps = page_size()
    with open(a.log, "a") as log:
        log.write(f"{now()} START per_proc={a.per_proc}GB min_free={a.min_free}GB(avail) min_pct={a.min_pct}% min_kill={a.min_kill}GB unmanaged={a.unmanaged}GB interval={a.interval}s pid={os.getpid()}\n")
        log.flush()
        while True:
            try:
                procs = lane_procs()
                reg = slots()
                jobs, unmanaged = job_usage(procs, reg)
                # (1) a job's processes are charged together: the job total, not any single member, is held to the
                #     declaration (1.25x) and to the per-process ceiling.
                for key, (tot, members) in jobs.items():
                    slot = reg[key]
                    declared = float(slot.get("gb", a.per_proc))
                    if tot > 1.25 * declared or tot > a.per_proc:
                        kill_job(slot, members, f"job rss={tot:.2f}GB over {len(members)} process(es) > "
                                 f"{'1.25x declared %.2fGB' % declared if tot > 1.25 * declared else 'per_proc=%.1fGB' % a.per_proc}"
                                 f" (lane {slot.get('lane')}, class {slot.get('class')})", log)
                # (2) processes nobody registered.
                for pid, gb, args in unmanaged:
                    if gb > a.per_proc:
                        kill(pid, f"rss={gb:.2f}GB > per_proc={a.per_proc}GB", args, log, None)
                    elif gb > a.unmanaged:
                        kill(pid, f"UNMANAGED rss={gb:.2f}GB > {a.unmanaged}GB without a lane_run slot (start it through zk/campaign/lane_run.py)", args, log, None)
                fr = free_gb(ps)
                pct = pressure_free_pct()
                # Both signals must say "pressure": available memory under the floor AND the kernel's free percentage
                # under --min-pct.  Then only a lane process that is actually large (>= --min-kill) is a candidate.
                if fr < a.min_free and (pct is None or pct < a.min_pct):
                    procs = [p for p in lane_procs() if p[1] >= a.min_kill]
                    if procs:
                        pid, gb, args = max(procs, key=lambda p: p[1])
                        kill(pid, f"avail={fr:.2f}GB < min_free={a.min_free}GB, pressure_free={pct}% (largest lane proc rss={gb:.2f}GB)", args, log, managed_slot(pid, slots()))
                    else:
                        log.write(f"{now()} LOW avail={fr:.2f}GB pressure_free={pct}%, no lane process >= {a.min_kill}GB to kill\n")
                        log.flush()
            except Exception as e:  # keep the guard alive
                log.write(f"{now()} ERROR {e!r}\n")
                log.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
