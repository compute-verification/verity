"""events.jsonl -- the one append helper, per-process streams, tolerant readers, span/hold accounting (brief §3, §5).

Writing: `append(path, record)` serializes the COMPLETE record first and issues ONE os.write on an O_APPEND descriptor; POSIX appends of
a single write to a regular file do not interleave, so many children may share the file.  Records are bounded (MAX_RECORD_BYTES); a larger
record is replaced by a truncation stub so the line stays valid JSON.  Every record carries the writer's stream id
`<host>:<pid>:<starttime>` and a per-process seq; a second host writes its own `events.<suffix>.jsonl` (STREAM_SUFFIX) -- readers merge
`events*.jsonl`.  No network file is assumed to be an event bus.

Clocks: `t_utc` for navigation, `t_mono` (time.monotonic) for durations on ONE host.  `duration()` refuses to subtract monotonic
timestamps of different hosts.

Reading: `iter_records(path)` yields parsed records and reports a truncated / malformed LAST line as a `partial` note instead of raising;
malformed lines elsewhere are counted and skipped.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import EVENTS_SCHEMA, HOLD_REASONS

MAX_RECORD_BYTES = 4096
_HOST = socket.gethostname()
_SEQ = 0


def _starttime(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            s = f.read().decode("ascii", "replace")
        return int(s[s.rindex(")") + 2:].split()[19])   # field 22 (starttime, clock ticks since boot)
    except (OSError, ValueError, IndexError):
        return None


def stream_id(pid: int | None = None, host: str | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    st = _starttime(pid)
    return f"{host or _HOST}:{pid}:{st if st is not None else 'na'}"


def now() -> tuple[str, float]:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()


def utc_now_ms() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


def events_path(attempt_dir: str | Path, suffix: str | None = None) -> Path:
    suffix = suffix if suffix is not None else os.environ.get("TELEMETRY_STREAM_SUFFIX", "")
    return Path(attempt_dir) / (f"events.{suffix}.jsonl" if suffix else "events.jsonl")


KEEP_KEYS = ("schema", "t_utc", "t_mono", "host", "pid", "stream", "seq", "kind", "stage", "id", "parent", "outcome", "rc", "reason", "actor", "name", "proc")


def _truncate_fields(rec: dict[str, Any], original_bytes: int) -> dict[str, Any]:
    """Over-long record: drop/shorten FIELDS (largest first), never bytes -- the line stays complete, valid JSON with its identity keys
    (rev d7).  The writer's truncation is marked `truncated_fields`; it is not a reader-side `partial`."""
    out = {k: rec[k] for k in KEEP_KEYS if k in rec}
    rest = sorted(((k, v) for k, v in rec.items() if k not in out), key=lambda kv: len(json.dumps(kv[1], default=str)))
    dropped: list[str] = []
    budget = MAX_RECORD_BYTES - len(json.dumps(out, separators=(",", ":"), default=str)) - 160
    for k, v in rest:                       # smallest first, so as many fields as fit survive intact
        enc = json.dumps(v, default=str)
        if len(enc) + len(k) + 4 <= budget:
            out[k] = v
            budget -= len(enc) + len(k) + 4
        elif isinstance(v, str) and budget > 64:
            out[k] = v[: budget - 40] + "...[truncated]"
            budget = 0
            dropped.append(k)
        else:
            dropped.append(k)
    out["truncated_fields"] = dropped
    out["original_bytes"] = original_bytes
    return out


def append(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append ONE complete record (+ stream/seq/schema/time if missing) with a single write.  Returns the record as written."""
    global _SEQ
    _SEQ += 1
    rec = dict(record)
    rec.setdefault("schema", EVENTS_SCHEMA)
    if "t_utc" not in rec or "t_mono" not in rec:
        rec.setdefault("t_utc", utc_now_ms())
        rec.setdefault("t_mono", time.monotonic())
    rec.setdefault("host", _HOST)
    rec.setdefault("pid", os.getpid())
    rec.setdefault("stream", stream_id())
    rec.setdefault("seq", _SEQ)
    line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
    data = line.encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        rec = _truncate_fields(rec, len(data))
        data = (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        n = os.write(fd, data)
        if n != len(data):   # regular files: a single write either completes or fails; a short write is reported, not retried piecemeal
            raise OSError(f"short write {n}/{len(data)} to {path}")
    finally:
        os.close(fd)
    return rec


@dataclass
class ReadReport:
    records: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    partial: list[dict[str, Any]] = field(default_factory=list)     # truncated / malformed lines, by file
    malformed: int = 0


def iter_records(path: str | Path, report: ReadReport | None = None) -> Iterator[dict[str, Any]]:
    """Parse JSONL tolerantly: a malformed LAST line is a truncation (recorded as partial); malformed inner lines are counted and skipped."""
    p = Path(path)
    rep = report if report is not None else ReadReport()
    rep.files.append(str(p))
    try:
        data = p.read_bytes()
    except OSError as e:
        rep.partial.append({"file": str(p), "reason": f"unreadable: {e.strerror}"})
        return
    lines = data.split(b"\n")
    trailing_complete = data.endswith(b"\n")
    if trailing_complete:
        lines = lines[:-1]
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except ValueError as e:
            last = i == len(lines) - 1
            if last and not trailing_complete:
                rep.partial.append({"file": str(p), "line": i + 1, "reason": "truncated last line (no newline)", "bytes": len(raw)})
            else:
                rep.malformed += 1
                if len(rep.partial) < 10:
                    rep.partial.append({"file": str(p), "line": i + 1, "reason": f"malformed: {e}", "bytes": len(raw)})
            continue
        yield rec


def read_all(attempt_dir: str | Path, prefix: str = "events") -> ReadReport:
    """Merge every `<prefix>*.jsonl` stream of an attempt directory, ordered by (t_utc, stream, seq)."""
    rep = ReadReport()
    d = Path(attempt_dir)
    for p in sorted(d.glob(f"{prefix}*.jsonl")):
        for rec in iter_records(p, rep):
            rep.records.append(rec)
    rep.records.sort(key=lambda r: (str(r.get("t_utc", "")), str(r.get("stream", "")), int(r.get("seq", 0) or 0)))
    return rep


def duration(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Monotonic b - a when both come from the same host; else None (never subtract unrelated hosts' monotonic clocks)."""
    ha, hb = str(a.get("stream", "")).split(":")[0], str(b.get("stream", "")).split(":")[0]
    if not ha or ha != hb or a.get("t_mono") is None or b.get("t_mono") is None:
        return None
    return float(b["t_mono"]) - float(a["t_mono"])


# --------------------------------------------------------------------------------------------------------------------------------------
# emitter (spans, marks, holds, queue timeline)
# --------------------------------------------------------------------------------------------------------------------------------------

class Emitter:
    def __init__(self, attempt_dir: str | Path, stream_suffix: str | None = None, proc: str | None = None):
        self.dir = Path(attempt_dir)
        self.path = events_path(self.dir, stream_suffix)
        self.proc = proc or Path(sys.argv[0]).name or "python"
        self.stream = stream_id()
        self._n = 0

    def _id(self, stage: str) -> str:
        self._n += 1
        return f"{stage}#{self._n}@{self.stream}"

    def emit(self, kind: str, **kv: Any) -> dict[str, Any]:
        return append(self.path, {"kind": kind, "proc": self.proc, **kv})

    def span_start(self, stage: str, parent: str | None = None, span_id: str | None = None, **kv: Any) -> str:
        """`parent` defaults to $TELE_PARENT_SPAN: a runner exports the span id it opened (TELE_PARENT_SPAN=<id>, TELE_ATTEMPT_DIR=<D>) and every
        child process -- fold worker, GM-01, a forked evaluator -- attaches its own spans under it without being told (rev d5)."""
        sid = span_id or self._id(stage)
        if parent is None:
            parent = os.environ.get("TELE_PARENT_SPAN") or None
        self.emit("span_start", stage=stage, id=sid, parent=parent, **kv)
        return sid

    def span_end(self, stage: str, span_id: str, outcome: str = "ok", rc: int | None = None, refs: list[str] | None = None, **kv: Any) -> None:
        self.emit("span_end", stage=stage, id=span_id, outcome=outcome, rc=rc, refs=refs or [], **kv)

    def mark(self, stage: str, name: str, **kv: Any) -> None:
        self.emit("mark", stage=stage, name=name, **kv)

    def hold_start(self, reason: str, detail: str | None = None, hold_id: str | None = None, **kv: Any) -> str:
        if reason not in HOLD_REASONS:
            kv["reason_detail"] = reason
            reason = "other"
        hid = hold_id or self._id(f"hold.{reason}")
        self.emit("hold_start", reason=reason, detail=detail, id=hid, **kv)
        return hid

    def hold_end(self, hold_id: str, reason: str | None = None, **kv: Any) -> None:
        self.emit("hold_end", id=hold_id, reason=reason, **kv)

    def queue(self, what: str, **kv: Any) -> None:
        assert what in ("submitted", "admitted", "started")
        self.emit(what, **kv)


# --------------------------------------------------------------------------------------------------------------------------------------
# accounting: spans (inclusive / exclusive / union), holds (interval union), queue timeline
# --------------------------------------------------------------------------------------------------------------------------------------

def _union(intervals: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    cur: tuple[float, float] | None = None
    for a, b in sorted(intervals):
        if b < a:
            a, b = b, a
        if cur is None:
            cur = (a, b)
        elif a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            total += cur[1] - cur[0]
            cur = (a, b)
    if cur is not None:
        total += cur[1] - cur[0]
    return total


def spans(records: list[dict[str, Any]], last_observation_mono: dict[str, float] | None = None) -> dict[str, Any]:
    """Fold span_start/span_end into per-span rows.  Same-host monotonic durations only; open spans stay open (`interrupted`) with the last
    observation of their stream as the last known time -- never an invented end.  Returns {spans: [...], total_union_s, overlap_s, by_stage}."""
    starts: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    last_seen: dict[str, float] = dict(last_observation_mono or {})
    for r in records:
        if r.get("t_mono") is not None and r.get("stream"):
            last_seen[r["stream"]] = max(last_seen.get(r["stream"], float("-inf")), float(r["t_mono"]))
        k = r.get("kind")
        if k == "span_start":
            sid = r.get("id")
            starts[sid] = r
            rows[sid] = {"id": sid, "stage": r.get("stage"), "parent": r.get("parent"), "stream": r.get("stream"),
                         "start_utc": r.get("t_utc"), "start_mono": r.get("t_mono"), "end_utc": None, "end_mono": None,
                         "outcome": "open", "rc": None, "refs": [], "inclusive_s": None, "exclusive_s": None, "children": []}
            order.append(sid)
        elif k == "span_end":
            sid = r.get("id")
            row = rows.get(sid)
            if row is None:   # end without start (lost start record): keep it, mark unmatched
                row = rows[sid] = {"id": sid, "stage": r.get("stage"), "parent": r.get("parent"), "stream": r.get("stream"), "start_utc": None,
                                   "start_mono": None, "end_utc": r.get("t_utc"), "end_mono": r.get("t_mono"), "outcome": r.get("outcome", "ok"),
                                   "rc": r.get("rc"), "refs": r.get("refs") or [], "inclusive_s": None, "exclusive_s": None, "children": [],
                                   "unmatched_start": True}
                order.append(sid)
                continue
            row.update(end_utc=r.get("t_utc"), end_mono=r.get("t_mono"), outcome=r.get("outcome", "ok"), rc=r.get("rc"), refs=r.get("refs") or [])
            d = duration(starts[sid], r)
            row["inclusive_s"] = d
            if d is None and starts[sid].get("t_mono") is not None:
                row["duration_note"] = "start and end on different hosts: monotonic clocks not comparable"
    for sid in order:
        row = rows[sid]
        if row["outcome"] == "open":
            row["outcome"] = "interrupted"
            ls = last_seen.get(row["stream"])
            if ls is not None and row["start_mono"] is not None:
                row["last_observation_mono"] = ls
                row["observed_at_least_s"] = max(0.0, ls - float(row["start_mono"]))
        if row["parent"] in rows:
            rows[row["parent"]]["children"].append(sid)
    # exclusive = inclusive - union(children intervals ∩ own interval)
    for sid in order:
        row = rows[sid]
        if row["inclusive_s"] is None or row["start_mono"] is None:
            continue
        a, b = float(row["start_mono"]), float(row["end_mono"])
        kid_iv = []
        for c in row["children"]:
            k = rows[c]
            if k["start_mono"] is None or k["stream"].split(":")[0] != row["stream"].split(":")[0]:
                continue
            ke = float(k["end_mono"]) if k["end_mono"] is not None else k.get("last_observation_mono", b)
            kid_iv.append((max(a, float(k["start_mono"])), min(b, ke)))
        row["exclusive_s"] = row["inclusive_s"] - _union(kid_iv)
    roots = [rows[s] for s in order if rows[s]["parent"] not in rows]
    closed_roots = [(float(r["start_mono"]), float(r["end_mono"])) for r in roots if r["inclusive_s"] is not None]
    total_union = _union(closed_roots)
    summed = sum(b - a for a, b in closed_roots)
    by_stage: dict[str, dict[str, Any]] = {}
    for sid in order:
        r = rows[sid]
        s = by_stage.setdefault(r["stage"], {"calls": 0, "inclusive_sum_s": 0.0, "exclusive_sum_s": 0.0, "open": 0})
        s["calls"] += 1
        if r["inclusive_s"] is not None:
            s["inclusive_sum_s"] += r["inclusive_s"]
            s["exclusive_sum_s"] += r["exclusive_s"] or 0.0
        else:
            s["open"] += 1
    return {"spans": [rows[s] for s in order], "roots": [r["id"] for r in roots], "total_union_s": total_union,
            "overlap_s": max(0.0, summed - total_union), "by_stage": by_stage,
            "note": "inclusive per span; exclusive = inclusive minus the union of its children; total = union of closed root spans; "
                    "concurrent roots are reported as overlap_s and never summed into elapsed"}


def holds(records: list[dict[str, Any]], end_mono_default: float | None = None) -> dict[str, Any]:
    """Hold intervals by reason; elapsed waiting = union of intervals (overlapping gates are not summed)."""
    open_h: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for r in records:
        if r.get("kind") == "hold_start":
            open_h[r["id"]] = r
        elif r.get("kind") == "hold_end" and r.get("id") in open_h:
            s = open_h.pop(r["id"])
            rows.append({"id": r["id"], "reason": s.get("reason"), "detail": s.get("detail"), "start_utc": s.get("t_utc"), "end_utc": r.get("t_utc"),
                         "start_mono": s.get("t_mono"), "end_mono": r.get("t_mono"), "seconds": duration(s, r), "open": False})
    for hid, s in open_h.items():
        rows.append({"id": hid, "reason": s.get("reason"), "detail": s.get("detail"), "start_utc": s.get("t_utc"), "end_utc": None,
                     "start_mono": s.get("t_mono"), "end_mono": end_mono_default, "seconds": None, "open": True})
    iv = [(float(h["start_mono"]), float(h["end_mono"])) for h in rows if h["start_mono"] is not None and h["end_mono"] is not None]
    by_reason: dict[str, float] = {}
    for h in rows:
        if h["seconds"] is not None:
            by_reason[h["reason"]] = by_reason.get(h["reason"], 0.0) + h["seconds"]
    return {"holds": rows, "waiting_union_s": _union(iv), "by_reason_summed_s": by_reason,
            "note": "waiting_union_s is the union of hold intervals; by_reason sums overlap and must not be added"}


def queue_timeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """submission -> admission -> start, as monotonic gaps when on one host, else UTC strings only."""
    ev = {k: None for k in ("submitted", "admitted", "started")}
    for r in records:
        if r.get("kind") in ev and ev[r["kind"]] is None:
            ev[r["kind"]] = r
    out: dict[str, Any] = {k: (v.get("t_utc") if v else None) for k, v in ev.items()}
    out["submission_to_admission_s"] = duration(ev["submitted"], ev["admitted"]) if ev["submitted"] and ev["admitted"] else None
    out["admission_to_start_s"] = duration(ev["admitted"], ev["started"]) if ev["admitted"] and ev["started"] else None
    return out
