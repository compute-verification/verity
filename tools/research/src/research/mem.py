"""Memory admission on a shared machine: `research run --on M --mem-gb N [--mem-wait S]` (machine side: remote.launch_request).

Every process on a pod shares one memory cgroup, and `free` / /proc/meminfo describe the HOST (vyv-v2cpu3 on 2026-09-23: MemTotal
946 GB, cgroup memory.max 256 GB, and 300 GB of hand-written reservations).  The reading, in GB (10^9 bytes):

    limit      memory.max (cgroup v2) / memory.limit_in_bytes (v1) of the cgroup with the least headroom (limit - in_use): the
               process's own, or a v2 ancestor's (a Nebius node puts each run in its own session scope under the user slice's
               cap, so the run's own empty scope says nothing about its neighbours); none finite -> MemTotal
    in_use     that cgroup's memory.current / memory.usage_in_bytes minus inactive_file (reclaimable page cache); meminfo: MemTotal
               - MemAvailable
    reserved   over the LIVE reservations in the ramlock dir (the fleet convention, /workspace/ramlock/<owner>-<job>.json with
               {"owner", "gb", "started", "pid"?, "note"}; machines.toml `ramlock` overrides the dir): the part not yet in use --
               gb minus the RSS of the named pid's process tree, or all of gb when the file names no pid (a running reserved
               job is already in `in_use`; counting it twice would refuse runs that fit)
    available  limit - in_use - reserved

A reservation is stale (listed, not counted) when its pid is gone or its run_dir is terminal (status.json done/failed/cancelled, or
refused.json).  An unreadable file is listed and not counted (its size is unknown).

admit() holds an flock on <ramlock>/.admission.lock from the reading through writing research-<run id>.json (same schema, plus
run_id / run_dir), so two launches cannot both take the last N GB.  Short of N it raises MemoryNotAvailable (refused.json names
it), or with wait_s polls until then.  The runner removes its file at exit, any status ($RESEARCH_MEM_RESERVATION, research.cli);
a runner that dies leaves a file whose pid is gone.  `research mem [--ramlock DIR] [--json]` prints the reading on the machine."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from .remote import MemoryNotAvailable
from .telemetry.cgroup import _int_or_kind, _kv_file, _read
from .telemetry.job import write_json_atomic

RAMLOCK_DIR = "/workspace/ramlock"
GB = 1e9
TERMINAL = ("done", "failed", "cancelled")


def _cgroup_dirs() -> list[Path]:
    if os.environ.get("RESEARCH_CGROUP_DIR"):
        return [Path(os.environ["RESEARCH_CGROUP_DIR"])]
    from .telemetry.cgroup import Cgroup

    cg = Cgroup("self")
    return [d for d in (cg.dir, cg.v1_dir) if d is not None]


def _with_ancestors(dirs: list[Path]) -> list[Path]:
    """Each dir, and each ancestor of a v2 one (memory.max beside it) that has a memory.max of its own."""
    out = []
    for d in dirs:
        out.append(d)
        if (d / "memory.max").exists():
            out += [p for p in d.parents if (p / "memory.max").exists()]
    return out


def _proc() -> Path:
    return Path(os.environ.get("RESEARCH_PROC_ROOT", "/proc"))


def reading() -> dict[str, Any]:
    """{"limit", "in_use" (bytes), "source", "cgroup": "v2" | "v1" | None}"""
    best: dict[str, Any] | None = None
    for d in _with_ancestors(_cgroup_dirs()):
        v2 = (d / "memory.max").exists()
        limit, _ = _int_or_kind(*_read(d / ("memory.max" if v2 else "memory.limit_in_bytes")))
        usage, _ = _int_or_kind(*_read(d / ("memory.current" if v2 else "memory.usage_in_bytes")))
        if limit is None or usage is None:
            continue
        inactive = _kv_file(_read(d / "memory.stat")[0]).get("inactive_file" if v2 else "total_inactive_file", 0)
        in_use = max(0, usage - inactive)
        if best is None or limit - in_use < best["limit"] - best["in_use"]:
            best = {"limit": limit, "in_use": in_use, "source": str(d), "cgroup": "v2" if v2 else "v1"}
    if best is not None:
        return best
    mi = {k.rstrip(":"): v * 1024 for k, v in _kv_file(_read(_proc() / "meminfo")[0]).items()}
    total = mi.get("MemTotal", 0)
    return {"limit": total, "in_use": max(0, total - mi.get("MemAvailable", total)), "source": str(_proc() / "meminfo"), "cgroup": None}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _rss_tree(pid: int) -> int | None:
    """Resident bytes of pid and its descendants (/proc/<n>/stat ppid, /proc/<n>/status VmRSS), None when pid is not in /proc."""
    parent: dict[int, int] = {}
    rss: dict[int, int] = {}
    try:
        entries = [e for e in os.listdir(_proc()) if e.isdigit()]
    except OSError:
        return None
    for e in entries:
        stat, _ = _read(_proc() / e / "stat")
        status, _ = _read(_proc() / e / "status")
        if not stat:
            continue
        try:
            parent[int(e)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        rss[int(e)] = _kv_file(status).get("VmRSS:", 0) * 1024
    if pid not in parent:
        return None
    tree, frontier = {pid}, [pid]
    while frontier:
        p = frontier.pop()
        for c, pp in parent.items():
            if pp == p and c not in tree:
                tree.add(c)
                frontier.append(c)
    return sum(rss.get(p, 0) for p in tree)


def _terminal(run_dir: str) -> bool:
    d = Path(run_dir)
    if (d / "refused.json").exists():
        return True
    try:
        return json.loads((d / "status.json").read_text()).get("state") in TERMINAL
    except (OSError, ValueError, AttributeError):
        return False


def reservations(ramlock_dir: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted(Path(ramlock_dir).glob("*.json")):
        row: dict[str, Any] = {"file": f.name}
        try:
            doc = json.loads(f.read_text())
            gb = float(doc["gb"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            out.append({**row, "live": False, "why": f"unreadable ({type(e).__name__}); not counted", "gb": None, "outstanding": 0})
            continue
        pid = doc.get("pid") if isinstance(doc.get("pid"), int) else None
        row.update(owner=doc.get("owner"), gb=gb, pid=pid, run_id=doc.get("run_id"), note=doc.get("note"))
        if doc.get("run_dir") and _terminal(str(doc["run_dir"])):
            row.update(live=False, why=f"run {doc.get('run_id')} ended", outstanding=0)
        elif pid is not None and not _alive(pid):
            row.update(live=False, why=f"pid {pid} gone", outstanding=0)
        else:
            used = _rss_tree(pid) if pid is not None else None
            row.update(live=True, why=f"pid {pid} holds {used / GB:.1f} GB" if used is not None else ("no pid" if pid is None else f"pid {pid}"),
                       outstanding=max(0.0, gb * GB - (min(used, gb * GB) if used is not None else 0)))
        out.append(row)
    return out


def snapshot(ramlock_dir: Path) -> dict[str, Any]:
    r = reading()
    res = reservations(ramlock_dir)
    reserved = sum(x["outstanding"] for x in res if x["live"])
    return {"limit_gb": round(r["limit"] / GB, 1), "in_use_gb": round(r["in_use"] / GB, 1), "reserved_gb": round(reserved / GB, 1),
            "available_gb": round((r["limit"] - r["in_use"] - reserved) / GB, 1), "source": r["source"], "cgroup": r["cgroup"],
            "ramlock_dir": str(ramlock_dir), "reservations": res}


def describe(s: dict[str, Any], asked: float | None = None) -> str:
    live = sorted((x for x in s["reservations"] if x["live"]), key=lambda x: -x["outstanding"])
    head = f"asked {asked:g} GB; " if asked is not None else ""
    text = (f"{head}available {s['available_gb']:g} GB = limit {s['limit_gb']:g} ({s['cgroup'] or 'meminfo'} {s['source']}) - in use "
            f"{s['in_use_gb']:g} - reserved {s['reserved_gb']:g} by {len(live)} live reservation(s) in {s['ramlock_dir']}")
    if live:
        text += ": " + ", ".join(f"{x['file']} {x['outstanding'] / GB:.0f}/{x['gb']:g} GB ({x['owner']}; {x['why']})" for x in live[:4])
        text += f" (+{len(live) - 4} more)" if len(live) > 4 else ""
    stale = [x for x in s["reservations"] if not x["live"]]
    if stale:
        text += "; not counted: " + ", ".join(f"{x['file']} ({x['why']})" for x in stale[:4])
    return text


@contextlib.contextmanager
def _locked(ramlock_dir: Path) -> Iterator[None]:
    with open(ramlock_dir / ".admission.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def admit(gb: float, *, ramlock_dir: Path, run_id: str, run_dir: Path, wait_s: float = 0.0, poll_s: float = 5.0, note: str = "") -> dict[str, Any]:
    """Reserve `gb` or raise MemoryNotAvailable.  -> the record for job.json run.remote.mem."""
    ramlock_dir = Path(ramlock_dir)
    ramlock_dir.mkdir(parents=True, exist_ok=True)
    path = ramlock_dir / f"research-{run_id}.json"
    t0 = time.monotonic()
    while True:
        with _locked(ramlock_dir):
            s = snapshot(ramlock_dir)
            if s["available_gb"] >= gb:
                write_json_atomic(path, {"owner": "research", "gb": gb, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": None,
                                         "run_id": run_id, "run_dir": str(run_dir), "note": note[:300]})
                return {"gb": gb, "reservation": str(path), "waited_s": round(time.monotonic() - t0, 1),
                        **{k: s[k] for k in ("limit_gb", "in_use_gb", "reserved_gb", "available_gb", "source", "cgroup")}}
        waited = time.monotonic() - t0
        if waited >= wait_s:
            raise MemoryNotAvailable(describe(s, gb) + (f"; waited {waited:.0f}s" if wait_s else "; --mem-wait S waits for it"))
        time.sleep(max(0.05, min(poll_s, wait_s - waited)))


def set_pid(path: str | Path, pid: int) -> None:
    p = Path(path)
    doc = json.loads(p.read_text())
    doc["pid"] = pid
    write_json_atomic(p, doc)


def release(path: str | Path | None, run_id: str | None = None) -> None:
    """Remove a reservation file -- only if it is this run's (a stale path must not delete someone else's)."""
    if not path:
        return
    p = Path(path)
    try:
        if run_id is None or json.loads(p.read_text()).get("run_id") == run_id:
            p.unlink()
    except (OSError, ValueError):
        pass


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research mem", description="this machine's memory reading for admission (research.mem)")
    ap.add_argument("--ramlock", default=os.environ.get("RESEARCH_RAMLOCK_DIR", RAMLOCK_DIR), metavar="DIR")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    s = snapshot(Path(ns.ramlock))
    print(json.dumps(s, indent=2, default=str) if ns.json else describe(s))
    return 0
