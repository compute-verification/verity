#!/usr/bin/env python3
"""`python3 oom_order.py [--runs DIR ...] [--ramlock DIR] [--log FILE] [--dry-run]`: order the kernel OOM killer's picks on a
shared node by each live run's memory use against its research.mem reservation, until reservations are caps (CI-98).

A reservation (`research run --mem-gb`, a check slot's mem=; research.mem's files in RAMLOCK) only gates admission: the kernel
picks its victim by size alone. T1R's replay r20261019-110314-3fd7 stayed inside its 200 GiB reservation and was OOM-killed as
the node's largest process, beside a run holding about 150 GB on a 64 GB reservation and one that had reserved nothing (INF-135).

Each pass, for every live run under the runs roots (status.json `running`), its processes are every pid whose environment names
the run dir as RESEARCH_RUN_DIR (what the runner exports to its workload, so its workers that leave the group count) and the
members of the run's process group (status.json's pgid, once a member with the variable is in it) that name no run dir, and every
process that names no run dir in a `.scope` cgroup holding the run's processes and no other run's: a run's workers that scrub its
environment and leave its group still sit in its session scope (24 Oct, cpu-3: suites.py's verity-circuit-check run e48b held 383 GB
in its scope with no reservation, read here at 0.1 GB, so the kernel took small runs and a reserved 80 GB verify first). Its use is their RssAnon + RssShmem (what reclaim can't free), its reservation the largest file in RAMLOCK naming its run
id (or a pid of its), else none. A run of several processes that reads over its reservation that way is read again from their
Pss_Anon + Pss_Shmem (smaps_rollup), which splits a page among the processes that share it: summed Rss counts a forked worker's
copy-on-write pages once per worker, and read each PRF-131 share (one ~278 GB parent, up to 9 forked workers) at 2,505 GB against
its 345 GB, so the share became the node's first pick (24 Oct, n2 00:09Z and 03:44Z, cpu-2 03:36Z). Pss only for those runs, and
only ones with a reservation: reading it walks the process's page tables (12 s a pass on n2 with every unreserved vLLM server read). Every process of the run gets one oom_score_adj:

  over (use > reservation, or no reservation)   OVER + round(SPAN * (use - reservation) / MemTotal), at most 1000
  within                                         WITHIN (0)
  either, under a "priority" reservation         PRIORITY (500) lower, at least -999

so a run over its reservation, or without one, is killed before a run within it, and the further over the sooner. WITHIN is 0, the
value each run's processes start at from its ssh login and could set again themselves: a root write also sets the process's floor
(oom_score_adj_min) to the value written, so a run once raised can't lower itself, and a run within its reservation is given
nothing below where it started. A process at -1000 (OOM-disabled by root on purpose) is left alone. Processes outside any run
keep theirs. The adj is written only where it differs, and each run whose processes changed is a line of LOG. vy-oom-order.timer
runs it every 15 s as root (reading another user's /proc/PID/environ and lowering an adj take root). Stdlib only: deploy.toml
installs this file alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNS = (Path("/workspace/research/runs"), Path("/workspace/jobs/runs"))
RAMLOCK = Path("/workspace/ramlock")
LOG = Path("/workspace/monitoring/oom-order.jsonl")
PROC = Path("/proc")
GB = 1e9                          # research.mem's GB
OVER, SPAN, WITHIN, MAX = 500, 500, 0, 1000
#: a run under a reservation marked "priority": true (a train check's slot, #3486) ranks this much lower, so a research job
#: growing back into its reservation is killed before a train (ci 24 Oct 05:47Z, Daniel's landing mandate)
PRIORITY = 500
DISABLED = -1000
#: a run's runner (`python3 -m research run`), in its run's scope but never one of its processes (nor its session's sshd)
RUNNER = b"-m\0research\0run\0"


def clock() -> datetime:
    return datetime.now(timezone.utc)


def _json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def live_runs(roots: list[Path]) -> dict[str, dict]:
    """Resolved run dir -> {"run", "pgid"} for each run under ``roots`` whose status.json says `running`."""
    out = {}
    for root in roots:
        for st_path in sorted(root.glob("*/status.json")):
            st = _json(st_path)
            if st.get("state") == "running":
                pgid = st.get("pgid")
                out[str(st_path.parent.resolve())] = {"run": st_path.parent.name, "pgid": pgid if isinstance(pgid, int) else None}
    return out


def reservations(ramlock: Path) -> list[dict]:
    """Each readable reservation in ``ramlock``: {"gb", "run_id", "pid"} (research.mem's format)."""
    out = []
    for f in sorted(ramlock.glob("*.json")):
        d = _json(f)
        try:
            gb = float(d["gb"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"file": f.name, "gb": gb, "run_id": d.get("run_id"), "pid": d.get("pid") if isinstance(d.get("pid"), int) else None,
                    "priority": d.get("priority") is True})
    return out


def _read(p: Path) -> bytes | None:
    try:
        return p.read_bytes()
    except OSError:
        return None


def scan(proc: Path = PROC) -> dict[int, dict]:
    """pid -> {"run_dir" (RESEARCH_RUN_DIR or None), "pgid", "bytes" (RssAnon + RssShmem), "adj"} for every process readable here."""
    out = {}
    for e in os.listdir(proc):
        if not e.isdigit():
            continue
        d = proc / e
        stat, status, adj = _read(d / "stat"), _read(d / "status"), _read(d / "oom_score_adj")
        if stat is None or status is None or adj is None:
            continue
        try:
            pgid = int(stat[stat.rindex(b")") + 2:].split()[2])
            cur = int(adj)
        except (ValueError, IndexError):
            continue
        rss = 0
        for ln in status.splitlines():
            if ln.startswith((b"RssAnon:", b"RssShmem:")):
                rss += int(ln.split()[1]) * 1024
        run_dir = None
        for kv in (_read(d / "environ") or b"").split(b"\0"):
            if kv.startswith(b"RESEARCH_RUN_DIR="):
                run_dir = kv.split(b"=", 1)[1].decode(errors="replace")
        cg = (_read(d / "cgroup") or b"").split(b"\n")[0]
        scope = cg.split(b"::", 1)[1].decode(errors="replace") if cg.startswith(b"0::") and cg.endswith(b".scope") else None
        out[int(e)] = {"run_dir": run_dir, "pgid": pgid, "bytes": rss, "adj": cur, "scope": scope}
    return out


def pss(pid: int, proc: Path = PROC) -> int | None:
    """pid's Pss_Anon + Pss_Shmem in bytes (/proc/PID/smaps_rollup), None when unreadable or without those fields."""
    text = _read(proc / str(pid) / "smaps_rollup")
    vals = [int(ln.split()[1]) * 1024 for ln in (text or b"").splitlines() if ln.startswith((b"Pss_Anon:", b"Pss_Shmem:"))]
    return sum(vals) if vals else None


def _runner(pid: int, proc: Path = PROC) -> bool:
    """Whether pid is a run's runner or its session's sshd (its cmdline, read only for a scope's candidates: reading it can wait
    on a large process's memory lock)."""
    cmd = _read(proc / str(pid) / "cmdline") or b""
    return RUNNER in cmd or cmd.startswith(b"sshd")


def target(use: int, reserved: float, limit: int, priority: bool = False) -> int:
    if reserved > 0 and use <= reserved:
        adj = WITHIN
    else:
        adj = min(MAX, OVER + round(SPAN * (use - reserved) / limit)) if limit > 0 else OVER
    return max(-999, adj - PRIORITY) if priority else adj


def plan(runs: dict[str, dict], res: list[dict], procs: dict[int, dict], limit: int, proc: Path = PROC) -> list[dict]:
    """Each live run with a process: {"run", "pids", "use", "reserved", "adj"}."""
    out = []
    scope_runs: dict[str, set[str]] = {}
    for x in procs.values():
        if x.get("scope") and x["run_dir"]:
            scope_runs.setdefault(x["scope"], set()).add(x["run_dir"])
    for run_dir, r in sorted(runs.items(), key=lambda kv: kv[1]["run"]):
        pids = {p for p, x in procs.items() if x["run_dir"] == run_dir}
        if r["pgid"] is not None and any(procs[p]["pgid"] == r["pgid"] for p in pids):
            pids |= {p for p, x in procs.items() if x["pgid"] == r["pgid"] and x["run_dir"] is None}
        own = {procs[p].get("scope") for p in pids} - {None}
        own = {s for s in own if scope_runs.get(s) == {run_dir}}
        if own:
            pids |= {p for p, x in procs.items() if x.get("scope") in own and x["run_dir"] is None and not _runner(p, proc)}
        if not pids:
            continue
        use = sum(procs[p]["bytes"] for p in pids)
        mine = [x for x in res if x["run_id"] == r["run"] or x["pid"] in pids]
        reserved = max([x["gb"] * GB for x in mine], default=0.0)
        priority = any(x.get("priority") for x in mine)
        if len(pids) > 1 and 0 < reserved < use:
            use = sum(v if (v := pss(p, proc)) is not None else procs[p]["bytes"] for p in pids)
        out.append({"run": r["run"], "pids": sorted(pids), "use": use, "reserved": reserved, "adj": target(use, reserved, limit, priority)})
    return out


def apply(plans: list[dict], procs: dict[int, dict], proc: Path = PROC, dry_run: bool = False) -> list[dict]:
    """Write each run's adj to its processes that differ; one record per run that changed."""
    out = []
    for p in plans:
        moved, errors = [], []
        for pid in p["pids"]:
            cur = procs[pid]["adj"]
            if cur in (p["adj"], DISABLED):
                continue
            try:
                if not dry_run:
                    (proc / str(pid) / "oom_score_adj").write_text(str(p["adj"]))
                moved.append([pid, cur])
            except (FileNotFoundError, ProcessLookupError):
                pass
            except OSError as e:
                errors.append(f"{pid}: {type(e).__name__}: {e}")
        if moved or errors:
            out.append({"at_utc": clock().strftime("%Y-%m-%dT%H:%M:%SZ"), "run": p["run"], "use_gb": round(p["use"] / GB, 1),
                        "reserved_gb": round(p["reserved"] / GB, 1), "adj": p["adj"], "set": moved, "dry_run": dry_run,
                        **({"errors": errors} if errors else {})})
    return out


def mem_total(proc: Path = PROC) -> int:
    for ln in (_read(proc / "meminfo") or b"").splitlines():
        if ln.startswith(b"MemTotal:"):
            return int(ln.split()[1]) * 1024
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oom_order.py", description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, action="append", default=None, metavar="DIR", help="runs root (repeatable; default: "
                    + ", ".join(map(str, RUNS)) + ")")
    ap.add_argument("--ramlock", type=Path, default=RAMLOCK)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--proc", type=Path, default=PROC, help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help="print what it would set, write nothing")
    a = ap.parse_args(argv)
    procs = scan(a.proc)
    recs = apply(plan(live_runs(a.runs or list(RUNS)), reservations(a.ramlock), procs, mem_total(a.proc), a.proc), procs, a.proc,
                 a.dry_run)
    for rec in recs:
        line = json.dumps(rec, sort_keys=True)
        print(line, flush=True)
        if not a.dry_run:
            a.log.parent.mkdir(parents=True, exist_ok=True)
            with a.log.open("a") as f:
                f.write(line + "\n")
    return 1 if any(r.get("errors") for r in recs) else 0


if __name__ == "__main__":
    sys.exit(main())
