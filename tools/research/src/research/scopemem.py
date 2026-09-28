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

Self-contained (stdlib, no package imports): `research status` sends this file to the run's machine as `python3 - RUN_DIR`
over the ssh the run's remote.json names, and it prints the reading as JSON."""
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


def locate(run_dir: Path, proc: Path, cg_root: Path) -> list[str]:
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
    paths: list[str] = []
    for p in pids:
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
            **({} if d.is_dir() else {"error": "gone"})}


def reading(run_dir: Path, proc: Path | None = None, cg_root: Path | None = None) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    proc = Path(proc or os.environ.get("RESEARCH_PROC_ROOT", "/proc"))
    cg_root = Path(cg_root or os.environ.get("RESEARCH_CGROUP_ROOT", "/sys/fs/cgroup"))
    recs = samples(run_dir)
    return [{**read(cg_root, c), "growth": growth(recs, c)} for c in locate(run_dir, proc, cg_root)]


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
    if len(argv) != 1:
        print("usage: python3 scopemem.py RUN_DIR", file=sys.stderr)
        return 2
    print(json.dumps(reading(Path(argv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
