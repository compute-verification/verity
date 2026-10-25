"""`python stall_probe.py --out FILE --origin T [--label TID=NAME]... [--samples N] NAME=PID...`: what blocks a tick.
Every thread of each watched process is sampled every `PERIOD_S` (1 ms) from a process of its own until SIGTERM (or N
samples), and each episode of a thread in uninterruptible sleep (state `D` of `/proc/PID/task/TID/stat`) is kept with
the kernel function it slept in at each of its samples (`wchan`), its kernel stack when that is readable (root only),
the page faults the thread took during it, and its memory cgroup's state at its start: a charge past the cgroup's
limit reclaims in the charging thread, which sleeps in D while it does.

It is a process of its own because a thread of the warden's process can't run while the ticker holds the interpreter
in the kernel (a page fault holds the GIL), which is when it is needed. Times are seconds after T, the warden's origin
on the monotonic clock, which every process of the host shares.

FILE has `threads` (each sampled thread: its process, label, samples by state, and its minor and major page faults
from the origin to its last sample), `episodes`, the sampler's own `interval_ms` quantiles and every `gap` of
`GAP_S` or more between its samples (a stall it would miss), and the cgroups' state at the start and the end. Stdlib
only.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PERIOD_S = 0.001
#: a pause of the sampler itself it records: a D episode shorter than it could fall inside one
GAP_S = 0.005
#: how often each process's thread list is read again
SCAN_S = 0.05
#: the sampler's intervals are kept as a histogram of 0.1 ms bins up to this many
BINS = 1000
CGROUP_FILES = ("memory.current", "memory.max", "memory.high", "memory.events", "memory.pressure", "io.pressure")


def stat_fields(raw: bytes) -> list[bytes]:
    """A `/proc/.../stat` line's fields from its state on (field 3, the state, is index 0)."""
    return raw[raw.rindex(b")") + 2:].split()


def faults(fields: list[bytes]) -> tuple[int, int]:
    """(minor, major) page faults of a stat line's fields (`stat_fields`)."""
    return int(fields[7]), int(fields[9])


def cgroup_path(pid: int, proc: Path, root: Path) -> Path | None:
    """The cgroup v2 directory of `pid`, or None."""
    try:
        for line in (proc / str(pid) / "cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return root / line[3:].lstrip("/")
    except OSError:
        pass
    return None


def cgroup_state(path: Path | None) -> dict | None:
    """`CGROUP_FILES` of a cgroup as read: numbers, `max`, or key -> number (events and pressure lines)."""
    if path is None:
        return None
    out: dict = {}
    for name in CGROUP_FILES:
        try:
            text = (path / name).read_text().strip()
        except OSError:
            continue
        if name in ("memory.events",):
            out[name] = {k: int(v) for k, v in (line.split() for line in text.splitlines())}
        elif name.endswith(".pressure"):
            out[name] = {line.split()[0]: int(line.rsplit("total=", 1)[1]) for line in text.splitlines()}
        else:
            out[name] = int(text) if text.isdigit() else text
    return out


class _Thread:
    def __init__(self, process: str, pid: int, tid: int, label: str, task: Path) -> None:
        self.process, self.pid, self.tid, self.label, self.task = process, pid, tid, label, task
        self.stat = os.open(task / "stat", os.O_RDONLY)
        try:
            self.wchan: int | None = os.open(task / "wchan", os.O_RDONLY)
        except OSError:
            self.wchan = None
        self.states: Counter = Counter()
        self.base: tuple[int, int] | None = None
        self.last: tuple[int, int] | None = None
        self.episode: dict | None = None

    def close(self) -> None:
        for fd in (self.stat, self.wchan):
            if fd is not None:
                os.close(fd)


class Sampler:
    """The sampling, a sample at a time (`sample(now)`), so that a test drives it on its own clock and /proc tree."""

    def __init__(self, watched: dict[str, int], origin: float, labels: dict[int, str] | None = None,
                 proc: Path = Path("/proc"), cgroups: Path = Path("/sys/fs/cgroup")) -> None:
        self.watched, self.origin, self.labels, self.proc = watched, origin, labels or {}, proc
        self.threads: dict[int, _Thread] = {}
        self.done: list[_Thread] = []
        self.episodes: list[dict] = []
        self.samples, self.last, self.next_scan = 0, None, None
        self.intervals = [0] * (BINS + 1)
        self.longest, self.gaps = 0.0, []
        self.stack_readable: bool | None = None
        self.cgroups = {name: cgroup_path(pid, proc, cgroups) for name, pid in watched.items()}
        self.cgroup_start = {name: cgroup_state(path) for name, path in self.cgroups.items()}

    def scan(self) -> None:
        for name, pid in self.watched.items():
            try:
                tids = [int(t) for t in os.listdir(self.proc / str(pid) / "task")]
            except OSError:
                continue
            for tid in tids:
                if tid not in self.threads:
                    try:
                        self.threads[tid] = _Thread(name, pid, tid, self.labels.get(tid, "thread"),
                                                    self.proc / str(pid) / "task" / str(tid))
                    except OSError:
                        continue

    def sample(self, now: float) -> None:
        if self.next_scan is None or now >= self.next_scan:
            self.scan()
            self.next_scan = now + SCAN_S
        if self.last is not None:
            gap = now - self.last
            self.intervals[min(BINS, int(gap * 10_000))] += 1
            self.longest = max(self.longest, gap)
            if gap >= GAP_S:
                self.gaps.append([round(self.last - self.origin, 4), round(gap * 1000, 3)])
        self.last = now
        self.samples += 1
        for tid, th in list(self.threads.items()):
            try:
                raw = os.pread(th.stat, 1024, 0)
            except OSError:
                raw = b""
            if not raw:
                self._end(th, now)
                del self.threads[tid]
                th.close()
                self.done.append(th)
                continue
            fields = stat_fields(raw)
            state = fields[0].decode()
            th.states[state] += 1
            th.last = faults(fields)
            if th.base is None and now >= self.origin:
                th.base = th.last
            if state == "D":
                self._in_d(th, now, fields)
            elif th.episode is not None:
                self._end(th, now)

    def _in_d(self, th: _Thread, now: float, fields: list[bytes]) -> None:
        if th.episode is None:
            th.episode = {"process": th.process, "pid": th.pid, "tid": th.tid, "label": th.label,
                          "start": round(now - self.origin, 4), "_start": now, "samples": 0, "wchan": Counter(),
                          "_faults": faults(fields), "stack": self._stack(th),
                          "cgroup": cgroup_state(self.cgroups.get(th.process))}
        th.episode["samples"] += 1
        if th.wchan is not None:
            try:
                where = os.pread(th.wchan, 128, 0).decode().strip()
            except OSError:
                where = ""
            if where and where != "0":
                th.episode["wchan"][where] += 1

    def _stack(self, th: _Thread) -> list[str] | None:
        if self.stack_readable is False:
            return None
        try:
            text = (th.task / "stack").read_text()
        except OSError:
            self.stack_readable = False
            return None
        self.stack_readable = True
        return [line.split("] ", 1)[-1] for line in text.splitlines()]

    def _end(self, th: _Thread, now: float) -> None:
        e = th.episode
        if e is None:
            return
        th.episode = None
        start, before = e.pop("_start"), e.pop("_faults")
        after = th.last or before
        e["ms"] = round((now - start) * 1000, 3)
        e["faults"] = [after[0] - before[0], after[1] - before[1]]
        e["wchan"] = dict(e["wchan"].most_common())
        self.episodes.append(e)

    def finish(self, now: float) -> None:
        for th in list(self.threads.values()):
            self._end(th, now)
            th.close()
            self.done.append(th)
        self.threads = {}

    def quantile(self, q: float) -> float | None:
        total = sum(self.intervals)
        if not total:
            return None
        k, seen = q * (total - 1), 0
        for i, n in enumerate(self.intervals):
            seen += n
            if seen > k:
                return round(i / 10, 1)
        return round(BINS / 10, 1)

    def doc(self) -> dict:
        threads = [{"process": th.process, "pid": th.pid, "tid": th.tid, "label": th.label,
                    "samples": sum(th.states.values()), "states": dict(th.states),
                    "faults": None if th.base is None or th.last is None else
                    [th.last[0] - th.base[0], th.last[1] - th.base[1]]}
                   for th in sorted(self.done, key=lambda t: (t.process, t.tid))]
        by_label: dict[str, dict] = {}
        for e in self.episodes:
            s = by_label.setdefault(e["label"], {"episodes": 0, "max_ms": 0.0, "wchan": Counter()})
            s["episodes"] += 1
            s["max_ms"] = max(s["max_ms"], e["ms"])
            s["wchan"].update(e["wchan"])
        return {"origin": self.origin, "period_ms": PERIOD_S * 1000, "samples": self.samples,
                "interval_ms": {"p50": self.quantile(0.5), "p99": self.quantile(0.99),
                                "max": round(self.longest * 1000, 3)},
                "gap_ms_at_least": GAP_S * 1000, "gaps": self.gaps,
                "stack": {None: "not read", True: "readable", False: "unreadable (root only)"}[self.stack_readable],
                "by_label": {k: v | {"wchan": dict(v["wchan"].most_common())} for k, v in by_label.items()},
                "threads": threads, "episodes": self.episodes,
                "cgroups": {name: {"path": str(path) if path else None, "start": self.cgroup_start[name],
                                   "end": cgroup_state(path)} for name, path in self.cgroups.items()}}


def summary(doc: dict) -> dict:
    """What `warden.json` keeps of FILE: each label's episodes and page faults from the origin, the sampler's pace and
    its gaps."""
    faults: dict[str, list[int]] = {}
    for th in doc["threads"]:
        if th["faults"] is not None:
            got = faults.setdefault(f"{th['process']} {th['label']}", [0, 0])
            got[0], got[1] = got[0] + th["faults"][0], got[1] + th["faults"][1]
    return {"by_label": doc["by_label"], "faults": faults, "samples": doc["samples"], "interval_ms": doc["interval_ms"],
            "gaps": len(doc["gaps"]), "stack": doc["stack"]}


def start(out: Path, origin: float, watched: dict[str, int], labels: dict[int, str]):
    """The sampler as a process beside the caller's, once it is sampling."""
    argv = [sys.executable, str(Path(__file__).resolve()), "--out", str(out), "--origin", repr(origin),
            *(f"--label={tid}={name}" for tid, name in labels.items()),
            *(f"{name}={pid}" for name, pid in watched.items())]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
    if "sampling" not in json.loads(proc.stdout.readline() or "{}"):
        proc.wait()
        raise RuntimeError(f"the stall probe exited ({proc.returncode}) before it sampled")
    return proc


def stop(proc, out: Path) -> dict:
    """Stop the sampler from `start` and give its `summary`."""
    proc.terminate()
    proc.wait()
    return summary(json.loads(out.read_text()))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="stall_probe.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--origin", required=True, type=float, help="the warden's origin on the monotonic clock")
    ap.add_argument("--label", action="append", default=[], help="TID=NAME: a thread's label (default: thread)")
    ap.add_argument("--samples", type=int, help="stop after this many samples")
    ap.add_argument("watched", nargs="+", help="NAME=PID: a process whose threads are sampled")
    args = ap.parse_args(argv)
    watched = {w.split("=", 1)[0]: int(w.split("=", 1)[1]) for w in args.watched}
    labels = {int(x.split("=", 1)[0]): x.split("=", 1)[1] for x in args.label}
    stop_ = []
    signal.signal(signal.SIGTERM, lambda *_: stop_.append(1))
    sampler = Sampler(watched, args.origin, labels)
    clock = time.monotonic
    sampler.sample(clock())
    print(json.dumps({"sampling": len(sampler.threads)}), flush=True)
    nxt = clock()
    while not stop_ and (args.samples is None or sampler.samples < args.samples):
        nxt += PERIOD_S
        delay = nxt - clock()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = clock()
        sampler.sample(clock())
        if not sampler.threads:
            break
    sampler.finish(clock())
    args.out.write_text(json.dumps(sampler.doc(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
