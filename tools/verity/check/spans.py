"""trace/v1 spans over a check run: `spans.jsonl` beside `steps.jsonl`, the queryable summary layer the watcher and the
dashboard read (the observability report's section 2; the first slice of the job schema).

One span is one JSON line.  The shape, FROZEN 14 Oct 2026 19:13Z (a contract infra's watcher builds against; change no
field):

    {"trace_id": "<run id>", "span_id": "...", "parent_span_id": null, "name": "...",
     "start_utc": "...", "end_utc": null, "status": null, "attributes": {"step": "pytest", "rc": 0}}

`parent_span_id` null marks the root; `end_utc`/`status` null while open; `status` is `ok`, `failed` or `cancelled`;
`attributes` is a flat string-to-scalar map (check steps carry `step`, `rc`, `cache_key`, `log`; a check in its own scope adds
`mem_peak_gb` and `mem_current_gb` to each step and to the root's completion, `check_memory`).

The file is append-only, so an open span is a line with `end_utc`/`status` null and its completion is a later line with
the same `span_id`: the last line per `span_id` wins.  The root span (the check itself) opens when the run's `Steps`
is made, so a watcher sees the check live; each step becomes a completed child span as `steps.jsonl` gets its record;
the root completes when the run ends.  `trace_id` is the recorded run's id (`RESEARCH_RUN_ID`), so the store's custody
ties the trace to the Attempt; outside a recorded run it is a local id, and the file is still written for humans.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

SPANS_LOG = "spans.jsonl"

#: a steps.jsonl status -> a span status (the frozen shape's three values)
STATUS = {"passed": "ok", "failed": "failed", "stopped": "cancelled", "skipped": "cancelled"}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool))


#: the check's own scope unit, set by slot.py on a `scope=1` slot (slot.SCOPE_ENV)
SCOPE_ENV = "CHECK_SCOPE_UNIT"


def check_memory(root: Path = Path("/")) -> dict:
    """`mem_peak_gb` (the check's high-water mark so far, memory.peak) and `mem_current_gb` of the cgroup this process runs in,
    when that is the check's own scope (SCOPE_ENV names it); else nothing, since a shared session scope's numbers are other
    runs' too. GB are 10^9 bytes, one decimal."""
    unit = os.environ.get(SCOPE_ENV)
    if not unit:
        return {}
    try:
        line = next(ln for ln in (root / "proc/self/cgroup").read_text().splitlines() if ln.startswith("0::"))
        d = root / "sys/fs/cgroup" / line[3:].strip().strip("/")
        if d.name != f"{unit}.scope":
            return {}
        return {f"mem_{k}_gb": round(int((d / f"memory.{k}").read_text().strip()) / 1e9, 1) for k in ("peak", "current")}
    except (OSError, StopIteration, ValueError):
        return {}


class SpanLog:
    """The one writer of a run's `spans.jsonl`: the open root line at construction, one completed child line per step
    record, and the root's completion line at `close`.  Append-only; the caller (`Steps`) serializes writes."""

    def __init__(self, out: Path, trace_id: str | None = None):
        self.path = out / SPANS_LOG
        self.trace_id = trace_id or os.environ.get("RESEARCH_RUN_ID") or f"local-{uuid.uuid4().hex[:12]}"
        self.root_id = uuid.uuid4().hex[:16]
        self.root_start = _now_utc()
        self.closed = False
        self.path.unlink(missing_ok=True)
        self._write(self.root_id, None, "check", self.root_start, None, None, {})

    def step(self, rec: dict) -> None:
        """The step's completed child span, from its `steps.jsonl` record (its end minus its `seconds` is its start)."""
        ended = rec.get("ended_utc") or _now_utc()
        try:
            t1 = datetime.strptime(ended, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            start = (t1 - timedelta(seconds=float(rec.get("seconds") or 0.0))).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            start = ended
        cache = rec.get("cache") or {}
        attributes = {"step": rec.get("step"), "rc": rec.get("rc"), "log": rec.get("log"),
                      "cache_key": cache.get("hit") or cache.get("stored"), "detail": rec.get("detail")}
        attributes = {k: v for k, v in {**attributes, **check_memory()}.items() if v is not None and _scalar(v)}
        self._write(uuid.uuid4().hex[:16], self.root_id, str(rec.get("step")), start, ended,
                    STATUS.get(rec.get("status"), "failed"), attributes)

    def close(self, status: str) -> None:
        """The root's completion line (`ok`, `failed` or `cancelled`); once. In its own scope it carries the check's peak."""
        if not self.closed:
            self.closed = True
            self._write(self.root_id, None, "check", self.root_start, _now_utc(), status, check_memory())

    def _write(self, span_id: str, parent: str | None, name: str, start: str, end: str | None, status: str | None,
               attributes: dict) -> None:
        line = {"trace_id": self.trace_id, "span_id": span_id, "parent_span_id": parent, "name": name,
                "start_utc": start, "end_utc": end, "status": status, "attributes": attributes}
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")
