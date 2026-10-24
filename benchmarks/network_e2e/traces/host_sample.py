"""`python host_sample.py OUT CPUS NAME=PID|NAME:PID...`: the host's load every `PERIOD` seconds, one JSON line each to
OUT, until SIGTERM, so that a warden trace's late ticks can be laid against what the machine was doing.

Each line: `t` (unix seconds), `load` (/proc/loadavg's 1-minute load, running and total tasks), `procs` (/proc/stat's
running and blocked), `busy` (the share of the period the whole host's CPUs were busy, and `steal`), `cpus` (the busy
share of each CPU in CPUS, the slot's), `psi` (/proc/pressure/cpu's `some` stall in the period, ms, when the kernel has
it), and per NAME the threads of PID and its descendants (``NAME=PID``) or of PID alone (``NAME:PID``): their CPU
seconds in the period, the CPUs they last ran on, their involuntary context switches in the period, their count, and the
main thread's state and nice. Stdlib only.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

PERIOD = 0.1
TICK = os.sysconf("SC_CLK_TCK")


def cpu_times(text: str) -> dict[str, list[int]]:
    """/proc/stat's `cpu` and `cpuN` lines: name -> its jiffies (user nice system idle iowait irq softirq steal ...)."""
    return {f[0]: [int(x) for x in f[1:]] for f in (line.split() for line in text.splitlines()) if f and f[0].startswith("cpu")}


def busy(a: list[int], b: list[int]) -> tuple[float | None, float | None]:
    """The busy and steal shares between two jiffy rows (idle and iowait are idle)."""
    d = [y - x for x, y in zip(a, b)]
    total = sum(d[:8])
    if total <= 0:
        return None, None
    return round(1 - (d[3] + d[4]) / total, 3), round(d[7] / total, 3)


def stat_field(text: str, key: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(key + " "):
            return int(line.split()[1])
    return None


def tree(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            for task in os.listdir(f"/proc/{p}/task"):
                todo += [int(c) for c in Path(f"/proc/{p}/task/{task}/children").read_text().split()]
        except OSError:
            continue
    return out


def threads(pid: int) -> list[tuple[int, int, int, str, int]]:
    """Each thread of `pid`: (utime + stime ticks, the CPU it last ran on, involuntary switches, state, nice)."""
    rows = []
    try:
        tasks = sorted(os.listdir(f"/proc/{pid}/task"), key=lambda x: (x != str(pid), int(x)))
    except OSError:
        return rows
    for task in tasks:
        try:
            st = Path(f"/proc/{pid}/task/{task}/stat").read_text()
            status = Path(f"/proc/{pid}/task/{task}/status").read_text()
        except OSError:
            continue
        f = st[st.rindex(")") + 2:].split()
        rows.append((int(f[11]) + int(f[12]), int(f[36]), stat_field(status, "nonvoluntary_ctxt_switches:") or 0, f[0],
                     int(f[16])))
    return rows


def watches(args: list[str]) -> dict[str, tuple[int, bool]]:
    """NAME=PID (with its descendants) and NAME:PID (alone) -> {NAME: (PID, descendants)}."""
    out = {}
    for a in args:
        i = min((a.find(c) for c in "=:" if c in a), default=-1)
        if i <= 0:
            raise SystemExit(f"{a!r}: NAME=PID or NAME:PID")
        out[a[:i]] = (int(a[i + 1:]), a[i] == "=")
    return out


def sample(cpus: list[int], watched: dict[str, tuple[int, bool]], prev: dict) -> dict:
    stat = Path("/proc/stat").read_text()
    times = cpu_times(stat)
    load = Path("/proc/loadavg").read_text().split()
    rec: dict = {"t": round(time.time(), 4), "load": [float(load[0]), *map(int, load[3].split("/"))],
                 "procs": [stat_field(stat, "procs_running"), stat_field(stat, "procs_blocked")]}
    if "times" in prev:
        rec["busy"] = busy(prev["times"]["cpu"], times["cpu"])
        rec["cpus"] = {c: busy(prev["times"][f"cpu{c}"], times[f"cpu{c}"])[0] for c in cpus
                       if f"cpu{c}" in times and f"cpu{c}" in prev["times"]}
    prev["times"] = times
    try:
        some = Path("/proc/pressure/cpu").read_text().split()
        total = int(some[4].split("=")[1])
        if "psi" in prev:
            rec["psi"] = round((total - prev["psi"]) / 1000, 3)
        prev["psi"] = total
    except (OSError, IndexError, ValueError):
        pass
    procs = {}
    for name, (pid, descendants) in watched.items():
        rows = [r for p in (tree(pid) if descendants else [pid]) for r in threads(p)]
        if not rows:
            continue
        cpu_ticks, invol = sum(r[0] for r in rows), sum(r[2] for r in rows)
        last = prev.setdefault("procs", {}).get(name)
        procs[name] = {"cpu_s": None if last is None else round((cpu_ticks - last[0]) / TICK, 3),
                       "invol": None if last is None else invol - last[1], "on": sorted({r[1] for r in rows}),
                       "state": rows[0][3], "nice": rows[0][4], "threads": len(rows)}
        prev["procs"][name] = (cpu_ticks, invol)
    rec["procs"] = procs
    return rec


def main(argv: list[str]) -> int:
    out, cpus = Path(argv[0]), [int(c) for c in argv[1].split(",") if c]
    watched = watches(argv[2:])
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    prev: dict = {}
    nxt = time.monotonic()
    with open(out, "w", encoding="utf-8") as f:
        while not stop:
            f.write(json.dumps(sample(cpus, watched, prev), separators=(",", ":")) + "\n")
            f.flush()
            nxt += PERIOD
            time.sleep(max(0.0, nxt - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
