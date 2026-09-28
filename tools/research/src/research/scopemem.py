"""A live run's memory where it executes: the cgroups (systemd scopes and units) its workload runs in, read on its machine.

    where      the cgroup v2 path of every process in the workload's process group (status.json `pgid`, `pid`; a pid whose
               start time is not the recorded `starttime_ticks` was reused and is skipped), plus the `--job-user` unit
               (/system.slice/research-job-<run id>.service), which runs outside that group.  One run can sit in several: a
               queued GPU phase's runner in cluster-<run id>.scope and its workload in gpu-lease's gpu-lease-<pid>.scope.
               The namespace root `/` is left out: it is the whole visible machine, not the run's.
    current    memory.current; peak memory.peak (absent before Linux 5.19)
    cap        the tightest finite memory.max on the cgroup and its ancestors (a scope inside a capped vy-lend.slice is bound
               by the slice), with the cgroup that sets it; `max` everywhere is no cap
    growth     bytes/s of memory.current over the last GROWTH_WINDOW_S of the run's own sampler records (resources.jsonl
               `cgroup` records), only for the cgroup the sampler reads; none otherwise -- never two reads taken here
    near_cap   current within NEAR_CAP of the cap
    cpuset     each cgroup's own cpuset.cpus (a scope's AllowedCPUs; none when unset) and cpuset.cpus.effective, and the
               workload group's affinities (Cpus_allowed_list), which a `taskset` sets where no cgroup does

`node` reads the same for several runs on one machine, and that machine's state they may hold or wait on: every GPU_LEASE_DIRS
directory's live leases and live waiters as gpu-lease writes them (an `<i>.owner` line whose pid lives and whose `<i>.lock`
someone flocks; a `wait.*` file someone flocks), with node 1's `n1-lease.json` pool record and `n1-lease.limits`, and the
check and Lean slots (`tools/verity/check/slot.py` and `lean_slot.py`'s `--status --json` documents, when the caller loaded them).
gpu-lease's locks are read from /proc/locks, never probed; slot.py probes its line's tickets, as `check.py --slots` does.

Self-contained (stdlib, no package imports): `research status` and `research inspect` send this file to the run's machine
(`remote.NODE_PROGRAM`) over the ssh the run's remote.json names, one ssh per machine, and it prints the reading as JSON."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

NEAR_CAP = 0.10
GROWTH_WINDOW_S = 60.0
SAMPLES_TAIL = 1 << 20
JOB_UNIT = "/system.slice/research-job-{run_id}.service"   # research.jobuser.unit_name, in the system slice systemd-run puts it
GIB = 1 << 30
GPU_LEASE_DIRS = ("/run/gpu-lease", "/run/gpu-lease-circuits")


def _text(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _bytes(raw: str | None) -> int | None:
    return int(raw) if raw is not None and raw.isdigit() else None


def _stat(proc: Path, pid: int) -> list[str] | None:
    """/proc/<pid>/stat fields after `comm`: [state, ppid, pgrp, ...]; starttime is index 19."""
    raw = _text(proc / str(pid) / "stat")
    return raw.rsplit(")", 1)[1].split() if raw and ")" in raw else None


def _cgroup_of(proc: Path, pid: int) -> str | None:
    for line in (_text(proc / str(pid) / "cgroup") or "").splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or None
    return None


def workload(run_dir: Path, proc: Path) -> list[int]:
    """The run's pid (unless reused) and every live member of its workload's process group."""
    try:
        st = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        st = {}
    pid, pgid, start = st.get("pid"), st.get("pgid"), st.get("starttime_ticks")
    pids: list[int] = []
    if isinstance(pid, int):
        f = _stat(proc, pid)
        if f is not None and (start is None or f[19:20] == [str(start)]):
            pids.append(pid)
    if isinstance(pgid, int) and (pids or not isinstance(pid, int)):
        try:
            entries = sorted(int(e) for e in os.listdir(proc) if e.isdigit())
        except OSError:
            entries = []
        pids += [p for p in entries if p not in pids and (_stat(proc, p) or [None] * 3)[2] == str(pgid)]
    return pids


def locate(run_dir: Path, proc: Path, cg_root: Path, pids: list[int] | None = None) -> list[str]:
    paths: list[str] = []
    for p in workload(run_dir, proc) if pids is None else pids:
        c = _cgroup_of(proc, p)
        if c and c != "/" and c not in paths:
            paths.append(c)
    unit = JOB_UNIT.format(run_id=run_dir.name)
    if unit not in paths and (cg_root / unit.lstrip("/")).is_dir():
        paths.append(unit)
    return paths


def samples(run_dir: Path) -> list[dict[str, Any]]:
    """The cgroup records in the last SAMPLES_TAIL bytes of the run's resources.jsonl, oldest first."""
    try:
        with open(run_dir / "resources.jsonl", "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - SAMPLES_TAIL))
            lines = f.read().decode("utf-8", "replace").splitlines()[1 if size > SAMPLES_TAIL else 0:]
    except OSError:
        return []
    out = []
    for line in lines:
        if '"kind":"cgroup"' not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec.get("cgroup"), dict) and isinstance(rec.get("t_mono"), (int, float)):
            out.append(rec)
    return out


def growth(recs: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    """{"bytes_per_s", "over_s"} from the samples of `path`: the newest against the oldest within GROWTH_WINDOW_S of it."""
    pts = [(float(r["t_mono"]), r["cgroup"]["memory_current_bytes"]) for r in recs
           if r["cgroup"].get("path") == path and isinstance(r["cgroup"].get("memory_current_bytes"), int)]
    if len(pts) < 2:
        return None
    t1, b1 = pts[-1]
    window = [(t, b) for t, b in pts[:-1] if t >= t1 - GROWTH_WINDOW_S]
    t0, b0 = window[0] if window else pts[-2]
    return {"bytes_per_s": (b1 - b0) / (t1 - t0), "over_s": round(t1 - t0, 1)} if t1 > t0 else None


def read(cg_root: Path, path: str) -> dict[str, Any]:
    d = cg_root / path.lstrip("/")
    own = _text(d / "memory.max")
    cap = cap_at = None
    p = Path(path)
    while True:
        v = _bytes(_text(cg_root / str(p).lstrip("/") / "memory.max"))
        if v is not None and (cap is None or v < cap):
            cap, cap_at = v, str(p)
        if p == p.parent:
            break
        p = p.parent
    cur = _bytes(_text(d / "memory.current"))
    return {"cgroup": path, "current_bytes": cur, "peak_bytes": _bytes(_text(d / "memory.peak")), "memory_max": own,
            "cap_bytes": cap, "cap_at": cap_at, "near_cap": cur is not None and cap is not None and cur >= (1 - NEAR_CAP) * cap,
            "cpuset": _text(d / "cpuset.cpus") or None, "cpus_effective": _text(d / "cpuset.cpus.effective"),
            **({} if d.is_dir() else {"error": "gone"})}


def _roots(proc: Path | None, cg_root: Path | None) -> tuple[Path, Path]:
    return (Path(proc or os.environ.get("RESEARCH_PROC_ROOT", "/proc")),
            Path(cg_root or os.environ.get("RESEARCH_CGROUP_ROOT", "/sys/fs/cgroup")))


def reading(run_dir: Path, proc: Path | None = None, cg_root: Path | None = None, pids: list[int] | None = None) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    proc, cg_root = _roots(proc, cg_root)
    recs = samples(run_dir)
    return [{**read(cg_root, c), "growth": growth(recs, c)} for c in locate(run_dir, proc, cg_root, pids)]


def affinity(proc: Path, pids: list[int]) -> list[str]:
    """The distinct Cpus_allowed_list of `pids`, in pid order."""
    out: list[str] = []
    for p in pids:
        for line in (_text(proc / str(p) / "status") or "").splitlines():
            if line.startswith("Cpus_allowed_list:") and (v := line.split(":", 1)[1].strip()) not in out:
                out.append(v)
    return out


def fields(line: str) -> dict[str, str]:
    """A gpu-lease owner or wait line's `key=value` fields; `cmd` is the rest of the line after `cmd=` (cluster.nebius2.fields)."""
    head, sep, cmd = line.strip().partition(" cmd=")
    if not sep and head.startswith("cmd="):
        head, cmd = "", head[4:]
    out = dict(tok.split("=", 1) for tok in head.split() if "=" in tok)
    if sep or cmd:
        out["cmd"] = cmd[:120]
    return out


def flocks(proc: Path) -> set[tuple[int, int, int]]:
    """(major, minor, inode) of every file someone holds a flock on, from /proc/locks (a waiter's line has `->` and no pid)."""
    out = set()
    for line in (_text(proc / "locks") or "").splitlines():
        f = line.split()
        if len(f) > 5 and f[1] == "FLOCK" and f[5].count(":") == 2:
            maj, mn, ino = f[5].split(":")
            out.add((int(maj, 16), int(mn, 16), int(ino)))
    return out


def _locked(p: Path, held: set[tuple[int, int, int]]) -> bool:
    try:
        st = p.stat()
    except OSError:
        return False
    return (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino) in held


def leases(d: Path, proc: Path, held: set[tuple[int, int, int]]) -> dict[str, Any] | None:
    """gpu-lease's directory `d`: its GPUs, how many are locked (a lease, or on node 1 a fence), each live lease's owner line
    and each live waiter's wait line, oldest first, and node 1's pool record; None when `d` doesn't exist."""
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    gpus = sorted(int(n[:-5]) for n in names if n.endswith(".lock") and n[:-5].isdigit())
    owners = []
    for i in gpus:
        f = fields(_text(d / f"{i}.owner") or "")
        if _locked(d / f"{i}.lock", held) and f.get("pid", "").isdigit() and (proc / f["pid"]).is_dir():
            owners.append({"index": i, **f})
    waits = [{"file": n, **fields(_text(d / n) or "")} for n in names
             if n.startswith("wait.") and n.rsplit(".", 1)[-1].isdigit() and _locked(d / n, held)]
    try:
        pool = json.loads((d / "n1-lease.json").read_text())
        pool = {k: pool.get(k) for k in ("t", "pool", "pool_name", "blocked")} | {"pending_holders": len(pool.get("pending_holders") or [])}
    except (OSError, ValueError, AttributeError):
        pool = None
    return {"gpus": len(gpus), "locked": sum(_locked(d / f"{i}.lock", held) for i in gpus), "owners": owners, "waits": waits,
            "pool": pool, "limits": _text(d / "n1-lease.limits")}


def node(run_dirs: list[str], proc: Path | None = None, cg_root: Path | None = None, slot_mods: tuple[Any, Any] | None = None,
         now: Any = None) -> dict[str, Any]:
    """Each run's memory and CPUs (by run dir), this machine's gpu-lease directories (GPU_LEASE_DIRS, or $RESEARCH_GPU_LEASE_DIRS
    colon-separated) and its check and Lean slots (`slot_mods`: slot.py and lean_slot.py, loaded; each document an
    `{"error": ...}` when it can't be read)."""
    proc, cg_root = _roots(proc, cg_root)
    runs = {}
    for rd in run_dirs:
        pids = workload(Path(rd), proc)
        runs[rd] = {"memory": reading(Path(rd), proc, cg_root, pids), "affinity": affinity(proc, pids)}
    held = flocks(proc)
    dirs = os.environ.get("RESEARCH_GPU_LEASE_DIRS")
    lease = {d: r for d in (dirs.split(":") if dirs else GPU_LEASE_DIRS) if (r := leases(Path(d), proc, held)) is not None}
    slots = None
    if slot_mods is not None:
        from datetime import UTC, datetime

        now = now or datetime.now(UTC)
        slots = {}
        for k, m in zip(("check", "lean"), slot_mods):
            try:
                slots[k] = m.status_doc(now, proc)
            except Exception as e:  # noqa: BLE001
                slots[k] = {"error": f"{type(e).__name__}: {e}"}
    return {"runs": runs, "leases": lease, "slots": slots}


def _gib(b: int | None) -> str:
    return "?" if b is None else f"{b / GIB:.1f} GiB"


def render(r: dict[str, Any]) -> str:
    """One line: `<cgroup>  current .. peak .. cap .. growth ..`, ending NEAR CAP when current is within NEAR_CAP of the cap."""
    if r.get("error") == "gone":
        return f"{r['cgroup']}  gone"
    if r["cap_bytes"] is None:
        cap = "none (memory.max is max)"
    else:
        used = f", {100 * r['current_bytes'] / r['cap_bytes']:.0f}% used" if r["current_bytes"] is not None else ""
        cap = _gib(r["cap_bytes"]) + (f" (at {r['cap_at']}{used})" if r["cap_at"] != r["cgroup"] else f" ({used[2:]})" if used else "")
    g = r.get("growth")
    grow = "-" if g is None else f"{g['bytes_per_s'] * 60 / GIB:+.1f} GiB/min over {g['over_s']:g} s of samples"
    return (f"{r['cgroup']}  current {_gib(r['current_bytes'])}  peak {_gib(r['peak_bytes'])}  cap {cap}  growth {grow}"
            + ("  NEAR CAP" if r["near_cap"] else ""))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python3 scopemem.py RUN_DIR...", file=sys.stderr)
        return 2
    print(json.dumps(node(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
