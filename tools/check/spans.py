"""trace/v1 spans over a check run: `spans.jsonl` beside `steps.jsonl`, the queryable summary layer the watcher and the
dashboard read.

One span is one JSON line.  The shape is frozen (a contract the watcher builds against; change no field):

    {"trace_id": "<run id>", "span_id": "...", "parent_span_id": null, "name": "...",
     "start_utc": "...", "end_utc": null, "status": null, "attributes": {"step": "pytest", "rc": 0}}

`parent_span_id` null marks the root; `end_utc`/`status` null while open; `status` is `ok`, `failed` or `cancelled`;
`attributes` is a flat string-to-scalar map (check steps carry `step`, `rc`, `cache_key`, `log`; a check in its own scope adds
`mem_peak_gb`, `mem_current_gb`, `mem_held_gb` and `cpu_s` to each step and to the root's completion, `check_memory`).

The file is append-only, so an open span is a line with `end_utc`/`status` null and its completion is a later line with
the same `span_id`: the last line per `span_id` wins.  The root span (the check itself) opens when the run's `Steps`
is made, so a watcher sees the check live; each step opens a child span as its process starts and completes it as
`steps.jsonl` gets its record (a step that never started, a skip, is one completed line); the root completes when the run
ends.  So the steps running now are the open children, each since its own start (`research status RUN` reads them).  `trace_id` is the recorded run's id (`RESEARCH_RUN_ID`), so the store's custody
ties the trace to the Attempt; outside a recorded run it is a local id, and the file is still written for humans.

A check in its own scope inside a `research run` also keeps, in the run's slot record, what a step's end can't show: its
busiest minute and its highest `anon` (`ScopeWatch`). systemd removes the scope with the check, and the run's own sampler
reads the cgroup it was started in, not the scope.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("spans_settings", Path(__file__).resolve().parent / "settings.py")
#: every environment variable check reads, declared once (`settings.py`)
SETTINGS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SETTINGS)
SPANS_LOG = "spans.jsonl"

#: a steps.jsonl status -> a span status (the frozen shape's three values)
STATUS = {"passed": "ok", "failed": "failed", "stopped": "cancelled", "skipped": "cancelled"}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool))


#: the check's own scope unit, set by slot.py on a `scope=1` slot (slot.SCOPE_ENV)
SCOPE_ENV = "CHECK_SCOPE_UNIT"
#: the run's slot record, relative to RESEARCH_RUN_DIR (slot.py's RUN_SLOT), which `ScopeWatch` adds `used` to
RUN_SLOT = "slot.json"
#: how often `ScopeWatch` reads the scope, and the span its busiest cores are averaged over and its record rewritten
WATCH_EVERY_S = 10
CORES_WINDOW_S = 60
GIB = 2 ** 30


def own_scope(root: Path = Path("/")) -> Path | None:
    """The cgroup this process runs in, when that is the check's own scope (SCOPE_ENV names it); else None, since a shared
    session scope's numbers are other runs' too."""
    unit = SETTINGS.get(SCOPE_ENV)
    if not unit or not (own := root / "proc/self/cgroup").is_file():
        return None
    if (line := next((ln for ln in own.read_text().splitlines() if ln.startswith("0::")), None)) is None:
        return None
    d = root / "sys/fs/cgroup" / line[3:].strip().strip("/")
    return d if d.name == f"{unit}.scope" else None


class ScopeWatch:
    """What the check's own scope used at its most, as `used` in the run's slot record (RUN_SLOT), which outlives the scope:

        cores_max        CPU seconds over seconds, at the busiest CORES_WINDOW_S
        anon_max_gib     `anon` of memory.stat at its highest: memory without the page cache
        samples, every_s how many readings, and how often one is taken

    A step's end in spans.jsonl shows the scope as it is then, after the step's busiest minute and its highest memory have
    passed, so these are read between: `sample` reads the scope every WATCH_EVERY_S on a thread of its own (`start`), and
    `write` puts the figures in the record every CORES_WINDOW_S and at `stop`, so a check that is killed keeps what it had
    used a minute before. GiB are 2^30 bytes, as a slot's mem= is. A reading that fails is counted (`tick`), and a record
    that can't be read or written is said on stderr and never stops the check."""

    def __init__(self, scope: Path, record: Path, clock=time.monotonic):
        self.scope, self.record, self.clock = scope, record, clock
        self.window: deque[tuple[float, float]] = deque()
        self.used: dict = {"cores_max": None, "anon_max_gib": 0.0, "samples": 0, "every_s": WATCH_EVERY_S}
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    @classmethod
    def start(cls, root: Path = Path("/")) -> ScopeWatch | None:
        """A watch of the check's own scope on its thread; None outside one or outside a `research run`."""
        if (scope := own_scope(root)) is None or not (run_dir := SETTINGS.RESEARCH_RUN_DIR):
            return None
        watch = cls(scope, Path(run_dir) / RUN_SLOT)
        watch.thread = threading.Thread(target=watch.run, name="scope-watch", daemon=True)
        watch.thread.start()
        return watch

    def sample(self) -> None:
        t = self.clock()
        anon = next(int(ln.split()[1]) for ln in (self.scope / "memory.stat").read_text().splitlines() if ln.startswith("anon "))
        cpu = int(next(ln.split()[1] for ln in (self.scope / "cpu.stat").read_text().splitlines()
                       if ln.startswith("usage_usec "))) / 1e6
        u = self.used
        self.window.append((t, cpu))
        while len(self.window) > 1 and t - self.window[1][0] >= CORES_WINDOW_S:
            self.window.popleft()
        t0, cpu0 = self.window[0]
        if t - t0 >= CORES_WINDOW_S:
            u["cores_max"] = max(u["cores_max"] or 0.0, round((cpu - cpu0) / (t - t0), 1))
        u.update(anon_max_gib=max(u["anon_max_gib"], round(anon / GIB, 1)), samples=u["samples"] + 1)

    def tick(self) -> None:
        """One `sample`; one that fails is counted in `used` (`missed`, and the last error as `unread`), not raised."""
        try:
            self.sample()
        except (OSError, ValueError, StopIteration) as e:
            self.used.update(missed=self.used.get("missed", 0) + 1, unread=f"{type(e).__name__}: {e}")

    def write(self) -> None:
        """`used` into the record, beside what slot.py wrote there."""
        doc = {}
        try:
            doc = json.loads(self.record.read_text())
        except (OSError, ValueError) as e:
            print(f"spans: the run's {RUN_SLOT} not read ({e}); `used` is written alone", file=sys.stderr, flush=True)
        tmp = self.record.with_name(f".{self.record.name}.{os.getpid()}")
        try:
            tmp.write_text(json.dumps({**(doc if isinstance(doc, dict) else {}), "used": dict(self.used)}) + "\n")
            os.replace(tmp, self.record)
        except OSError as e:
            print(f"spans: what the scope used not written to the run's {RUN_SLOT}: {e}", file=sys.stderr, flush=True)

    def run(self) -> None:
        written = self.clock()
        while not self.stopped.wait(WATCH_EVERY_S):
            self.tick()
            if self.clock() - written >= CORES_WINDOW_S:
                self.write()
                written = self.clock()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread:
            self.thread.join()
        self.tick()
        self.write()


def check_memory(root: Path = Path("/")) -> dict:
    """`mem_peak_gb` (the check's high-water mark so far, memory.peak) and `mem_current_gb` of the cgroup this process runs in,
    when that is the check's own scope (SCOPE_ENV names it); else nothing, since a shared session scope's numbers are other
    runs' too; with them `mem_held_gb`, memory.current less the page cache (`file - shmem` of memory.stat: what is left once
    the kernel has reclaimed what it can, research.memmark's `held`), and `cpu_s`, the CPU seconds the scope has used
    (cpu.stat's usage_usec), each where its file can be read. The scope is gone when the check ends, so the root's
    completion keeps what the check used: its peak with the page cache, its CPU seconds over the time it held its slot, and
    the most a step's end held without the cache, a floor on that peak. GB are 10^9 bytes, one decimal."""
    unit = SETTINGS.get(SCOPE_ENV)
    if not unit:
        return {}
    def stat(f: Path) -> dict[str, int]:
        if not f.is_file():
            return {}
        return {w[0]: int(w[1]) for ln in f.read_text().splitlines() if len(w := ln.split()) == 2 and w[1].isdigit()}

    try:
        line = next(ln for ln in (root / "proc/self/cgroup").read_text().splitlines() if ln.startswith("0::"))
        d = root / "sys/fs/cgroup" / line[3:].strip().strip("/")
        if d.name != f"{unit}.scope":
            return {}
        peak, current = (int((d / f"memory.{k}").read_text().strip()) for k in ("peak", "current"))
        mem, cpu = stat(d / "memory.stat"), stat(d / "cpu.stat")
    except (OSError, StopIteration, ValueError):
        return {}
    out = {"mem_peak_gb": round(peak / 1e9, 1), "mem_current_gb": round(current / 1e9, 1)}
    if "file" in mem:
        out["mem_held_gb"] = round(max(0, current - max(0, mem["file"] - mem.get("shmem", 0))) / 1e9, 1)
    if "usage_usec" in cpu:
        out["cpu_s"] = round(cpu["usage_usec"] / 1e6, 1)
    return out


class SpanLog:
    """The one writer of a run's `spans.jsonl`: the open root line at construction, a step's open child line as it starts
    (`open`) and its completion from its record (`step`), and the root's completion line at `close`.  Append-only; the
    caller (`Steps`) serializes writes."""

    def __init__(self, out: Path, trace_id: str | None = None):
        self.path = out / SPANS_LOG
        self.trace_id = trace_id or SETTINGS.RESEARCH_RUN_ID or f"local-{uuid.uuid4().hex[:12]}"
        self.root_id = uuid.uuid4().hex[:16]
        self.root_start = _now_utc()
        self.closed = False
        self.opened: dict[str, tuple[str, str]] = {}
        self.path.unlink(missing_ok=True)
        self._write(self.root_id, None, "check", self.root_start, None, None, {})
        self.watch = ScopeWatch.start()

    def open(self, name: str) -> None:
        """The step's open child span, as its process starts; `step` completes it."""
        span_id, start = uuid.uuid4().hex[:16], _now_utc()
        self.opened[name] = (span_id, start)
        self._write(span_id, self.root_id, name, start, None, None, {"step": name})

    def step(self, rec: dict) -> None:
        """The step's completed child span, from its `steps.jsonl` record: the span `open` started, else a new one whose start
        is its end minus its `seconds`."""
        ended = rec.get("ended_utc") or _now_utc()
        try:
            t1 = datetime.strptime(ended, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            start = (t1 - timedelta(seconds=float(rec.get("seconds") or 0.0))).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            start = ended
        span_id, start = self.opened.pop(str(rec.get("step")), (uuid.uuid4().hex[:16], start))
        cache = rec.get("cache") or {}
        attributes = {"step": rec.get("step"), "rc": rec.get("rc"), "log": rec.get("log"),
                      "cache_key": cache.get("hit") or cache.get("stored"), "detail": rec.get("detail")}
        attributes = {k: v for k, v in {**attributes, **check_memory()}.items() if v is not None and _scalar(v)}
        self._write(span_id, self.root_id, str(rec.get("step")), start, ended, STATUS.get(rec.get("status"), "failed"), attributes)

    def close(self, status: str) -> None:
        """The root's completion line (`ok`, `failed` or `cancelled`); once. In its own scope it carries the check's peak."""
        if not self.closed:
            self.closed = True
            self._write(self.root_id, None, "check", self.root_start, _now_utc(), status, check_memory())
            if self.watch:
                self.watch.stop()

    def _write(self, span_id: str, parent: str | None, name: str, start: str, end: str | None, status: str | None,
               attributes: dict) -> None:
        line = {"trace_id": self.trace_id, "span_id": span_id, "parent_span_id": parent, "name": name,
                "start_utc": start, "end_utc": end, "status": status, "attributes": attributes}
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")
