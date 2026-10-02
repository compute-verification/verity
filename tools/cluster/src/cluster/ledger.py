"""The allocation ledger: an append-only, hash-chained JSON-lines log of what each node's scheduler did.

Every record carries `seq`, `t` (epoch seconds), `kind`, its fields, `prev` (the previous record's hash) and `hash`
(sha256 of the record's canonical JSON without `hash`). Kinds: `submit` (a job enters a node's queue, with the `handle`
its request holds live), `withdraw` (a queued job left without starting: its handle dropped, or it gave up), `start` (an
allocation; `source` names who decided it when that wasn't this ledger's planner, as for a lease `gpu-lease` granted),
`freeze` and `evict` (with `by`), `thaw`, `expire`, `end` (with `rc`, the run id when it was a recorded run, and `gpus`:
each GPU's held, busy and sampled seconds, from `gpu_usage` or a `gpu-lease/usage/v1` record), `refuse` (a job and its
reason), `idle` (GPUs held without computing, a notice only), `admin-begin` / `admin-end` for a repair session on a
node (who and why), which no scheduler decided and every quiet report must see, and `agent` (a live agent took over the
chain: its pid and the tree it runs from), which changes no state.

A long-lived agent writes its ledger in segments (`<root>/<stamp>/ledger.jsonl`): each segment's first record continues the
last one's `seq` and `prev`, so the segments in order (`segments`) are one chain, and `read` of the root reads them all.

The ledger is the one source of lease and queue state: `state` replays it into the live allocations and the queue, and
the node agent writes that to `state.json` after every tick, so every reader (the fill runner, the quiet step, status
pages, usage reports) sees the same state, never one inferred from processes or file names. It is also the evidence
behind a run's placement: `history` replays it into each allocation's running and frozen intervals, and `quiet_report`
says, for one allocation, what else was active on its node while it ran, which is what a timed row's claim that "the node
was quiet" rests on.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import model
from .plan import Action, Evict, Expire, Freeze, Idle, Refuse, Start, Thaw

USAGE_V1 = "gpu-lease/usage/v1"

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != "hash"})).hexdigest()


class Ledger:
    """Appends to one ledger file, which continues `chain` (the earlier segments' files, in order) when given. Opening checks
    the whole chain, so a writer never extends a broken one."""

    def __init__(self, path: str | Path, chain: Iterable[str | Path] = ()):
        self.path = Path(path)
        files = [Path(p) for p in chain if Path(p) != self.path] + [self.path]
        records = [r for f in files if f.exists() for r in read(f)]
        problems = verify(records)
        if problems:
            raise ValueError(f"{self.path}: {problems[0]}")
        self.seq = records[-1]["seq"] if records else 0
        self.last = records[-1]["hash"] if records else GENESIS
        self.last_t = records[-1]["t"] if records else None

    def roll(self, path: str | Path) -> None:
        """Append from now on to `path`, a new segment's file, continuing this chain."""
        path = Path(path)
        if path.exists() and path.stat().st_size:
            raise ValueError(f"{path}: a new segment must start empty")
        self.path = path

    def append(self, t: int, kind: str, **fields: Any) -> dict[str, Any]:
        if self.last_t is not None and t < self.last_t:
            raise ValueError(f"record at {t} is older than the ledger's last, {self.last_t}")
        record = {"seq": self.seq + 1, "t": t, "kind": kind, **fields, "prev": self.last}
        record["hash"] = digest(record)
        with self.path.open("a") as f:
            f.write(canonical(record).decode() + "\n")
        self.seq, self.last, self.last_t = record["seq"], record["hash"], t
        return record

    def submit(self, t: int, job: model.Job, node: str, handle: str | None = None) -> dict[str, Any]:
        return self.append(t, "submit", job=model.job_to_dict(job), node=node, **({"handle": handle} if handle else {}))

    def withdraw(self, t: int, job: str, why: str) -> dict[str, Any]:
        return self.append(t, "withdraw", job=job, why=why)

    def end(self, t: int, alloc: str, rc: int | None = None, *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
        """An allocation's end, with its exit status and per-GPU use from its `gpu-lease/usage/v1` record when there is one."""
        return self.append(t, "end", alloc=alloc, **(end_fields(usage) if usage else {"rc": rc}))

    def actions(self, t: int, actions: Iterable[Action], source: str | None = None) -> list[dict[str, Any]]:
        """Record the actions an executor carried out at `t` (a Wait is not recorded: nothing happened)."""
        out = []
        for a in actions:
            if isinstance(a, Start):
                out.append(self.append(t, "start", alloc=model.allocation_to_dict(a.alloc), **({"source": source} if source else {})))
            elif isinstance(a, (Freeze, Evict)):
                out.append(self.append(t, "freeze" if isinstance(a, Freeze) else "evict", alloc=a.alloc, by=a.by))
            elif isinstance(a, (Thaw, Expire)):
                out.append(self.append(t, "thaw" if isinstance(a, Thaw) else "expire", alloc=a.alloc))
            elif isinstance(a, Refuse):
                out.append(self.append(t, "refuse", job=a.job, reason=a.reason))
            elif isinstance(a, Idle):
                out.append(self.append(t, "idle", alloc=a.alloc, submitter=a.submitter, idle_s=a.idle_s))
        return out


def read(path: str | Path) -> list[dict[str, Any]]:
    """A ledger file's records; for a directory, its own `ledger.jsonl`, else every segment's under it in order."""
    path = Path(path)
    if path.is_dir():
        if (path / "ledger.jsonl").exists():
            return read(path / "ledger.jsonl")
        return [r for seg in segments(path) for r in read(seg)]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def segments(root: str | Path) -> list[Path]:
    """The segment ledgers under `root` (`<root>/*/ledger.jsonl`) in chain order: by their first record's `seq`, then by
    name, with the empty ones last."""
    def first(p: Path) -> float:
        with p.open() as f:
            line = f.readline()
        return json.loads(line)["seq"] if line.strip() else float("inf")
    return sorted(Path(root).glob("*/ledger.jsonl"), key=lambda p: (first(p), p.parent.name))


@dataclass(frozen=True)
class Queued:
    job: model.Job
    node: str
    handle: str | None
    since: int


@dataclass(frozen=True)
class State:
    """Lease and queue state as the ledger has it at record `seq` (hash `head`): the live allocations and the queued jobs."""
    allocations: dict[str, model.Allocation]
    queue: dict[str, Queued]
    seq: int = 0
    head: str = GENESIS
    t: int | None = None

    def on(self, node: str) -> tuple[list[model.Allocation], list[model.Job]]:
        return ([a for a in self.allocations.values() if a.node == node],
                [q.job for q in self.queue.values() if q.node == node])


def state(records: Iterable[dict[str, Any]], start: State | None = None) -> State:
    """Replay the ledger (or the records after `start`, on top of it) into its live allocations and queue. A start takes its
    job out of the queue; an evict, expire or end removes the allocation, and a job that runs again is submitted again; a
    withdraw or a refusal removes a queued job."""
    allocs: dict[str, model.Allocation] = dict(start.allocations) if start else {}
    queue: dict[str, Queued] = dict(start.queue) if start else {}
    seq, head, t = (start.seq, start.head, start.t) if start else (0, GENESIS, None)
    for r in records:
        k, t = r["kind"], r["t"]
        seq, head = r["seq"], r["hash"]
        if k == "submit":
            job = model.job_from_dict(r["job"])
            queue[job.id] = Queued(job, r["node"], r.get("handle"), t)
        elif k in ("withdraw", "refuse"):
            queue.pop(r["job"], None)
        elif k == "start":
            a = model.allocation_from_dict(r["alloc"])
            allocs[a.id] = a
            queue.pop(a.job.id, None)
        elif k == "freeze":
            a = allocs[r["alloc"]]
            allocs[a.id] = replace(a, state="frozen", frozen_by=r["by"], frozen_at=a.frozen_at if a.frozen_at is not None else t)
        elif k == "thaw":
            a = allocs[r["alloc"]]
            allocs[a.id] = replace(a, state="running", frozen_by=None, frozen_at=None,
                                   frozen_s=a.frozen_s + t - (a.frozen_at if a.frozen_at is not None else t))
        elif k in ("evict", "expire", "end"):
            allocs.pop(r["alloc"], None)
    return State(allocs, queue, seq, head, t)


def state_to_dict(s: State) -> dict[str, Any]:
    return {"schema": "cluster/state/v1", "seq": s.seq, "head": s.head, "t": s.t,
            "allocations": [model.allocation_to_dict(a) for a in sorted(s.allocations.values(), key=lambda a: a.id)],
            "queue": [{"job": model.job_to_dict(q.job), "node": q.node, "handle": q.handle, "since": q.since}
                      for q in sorted(s.queue.values(), key=lambda q: (q.since, q.job.id))]}


def state_from_dict(d: dict[str, Any]) -> State:
    return State({a["id"]: model.allocation_from_dict(a) for a in d["allocations"]},
                 {q["job"]["id"]: Queued(model.job_from_dict(q["job"]), q["node"], q.get("handle"), q["since"]) for q in d["queue"]},
                 d["seq"], d["head"], d["t"])


def write_state(path: str | Path, s: State) -> None:
    """`state.json`, replaced whole (write, then rename), so a reader never sees half of it."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(state_to_dict(s), sort_keys=True) + "\n")
    os.replace(tmp, path)


def epoch(stamp: str) -> int:
    """`2026-09-30T18:00:00Z` as epoch seconds, the form `gpu-lease` writes its times in."""
    return int(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def end_fields(rec: dict[str, Any]) -> dict[str, Any]:
    """An `end` record's fields from a `gpu-lease/usage/v1` record: its `rc`, `run` and per-GPU `gpus`, in `gpu_usage`'s
    shape. Busy and sampled time are capped at the held time, since `gpu-lease` counts each reading a full cadence."""
    if rec.get("schema") != USAGE_V1:
        raise ValueError(f"not a {USAGE_V1} record: {rec.get('schema')!r}")
    gpus = []
    for g in rec["gpus"]:
        held = int(g["held_s"])
        gpus.append({"index": int(g["index"]), "uuid": g.get("uuid", "-"), "held_s": held,
                     "busy_s": min(int(g["busy_s"]), held), "sampled_s": min(int(g["sampled_s"]), held)})
    out: dict[str, Any] = {"rc": rec["rc"], "gpus": gpus}
    if rec.get("run"):
        out["run"] = rec["run"]
    return out


def usage_key(rec: dict[str, Any]) -> tuple[int, int]:
    """What ties a usage record to its lease: `gpu-lease`'s pid and start (both in the lease's owner file)."""
    return int(rec["pid"]), epoch(rec["since"])


def verify(records: list[dict[str, Any]]) -> list[str]:
    """Every break in the chain, by record: a gap in `seq`, a `prev` that isn't the last hash, a wrong hash, time going back."""
    out = []
    prev, last_t = GENESIS, None
    for i, r in enumerate(records, 1):
        if r.get("seq") != i:
            out.append(f"record {i}: seq is {r.get('seq')}")
        if r.get("prev") != prev:
            out.append(f"record {i}: prev does not match record {i - 1}'s hash")
        if r.get("hash") != digest(r):
            out.append(f"record {i}: hash does not match its content")
        if last_t is not None and r.get("t", 0) < last_t:
            out.append(f"record {i}: time goes back")
        prev, last_t = r.get("hash"), r.get("t", last_t)
    return out


@dataclass
class History:
    alloc: str
    node: str
    job: dict[str, Any]
    gpus: list[int]
    cpus: str
    intervals: list[list[Any]] = field(default_factory=list)  # [state, t0, t1 or None]
    ended: str | None = None

    def close(self, t: int) -> None:
        if self.intervals and self.intervals[-1][2] is None:
            self.intervals[-1][2] = t

    def open(self, state: str, t: int) -> None:
        self.close(t)
        self.intervals.append([state, t, None])


def history(records: Iterable[dict[str, Any]]) -> tuple[dict[str, History], list[dict[str, Any]]]:
    """Each allocation's intervals, and the admin sessions as `{node, who, why, t0, t1}`."""
    allocs: dict[str, History] = {}
    admin: list[dict[str, Any]] = []
    for r in records:
        k, t = r["kind"], r["t"]
        if k == "start":
            a = r["alloc"]
            h = allocs[a["id"]] = History(a["id"], a["node"], a["job"], a["gpus"], a["cpus"])
            h.open("running", t)
        elif k in ("freeze", "thaw"):
            allocs[r["alloc"]].open("frozen" if k == "freeze" else "running", t)
        elif k in ("evict", "expire", "end"):
            h = allocs[r["alloc"]]
            h.close(t)
            h.ended = k if k != "end" else f"end rc={r.get('rc')}"
        elif k == "admin-begin":
            admin.append({"node": r["node"], "who": r.get("who"), "why": r.get("why"), "t0": t, "t1": None})
        elif k == "admin-end":
            for s in admin:
                if s["node"] == r["node"] and s["t1"] is None:
                    s["t1"] = t
    return allocs, admin


def gpu_usage(t0: int, t1: int, samples: Iterable[tuple[int, bool]], cadence_s: int) -> dict[str, int]:
    """One GPU's use over a lease held in [t0, t1): `held_s`, `busy_s` and `sampled_s`, from the node's sampler readings
    `(t, busy)`, each covering [t, t + cadence_s). Time no sample covers (a timed window, where the sampler reads no NVML)
    is held but unsampled, never counted idle."""
    busy = sampled = 0
    for t, is_busy in samples:
        span = max(0, min(t + cadence_s, t1) - max(t, t0))
        sampled += span
        busy += span if is_busy else 0
    return {"held_s": t1 - t0, "busy_s": busy, "sampled_s": min(sampled, t1 - t0)}


def usage(records: Iterable[dict[str, Any]], since: int = 0) -> list[dict[str, Any]]:
    """Held against busy GPU time per submitter, from the `end` records' per-GPU `gpus` entries since `since`, lowest busy
    share first: which holders' leases sit idle. A report for panels and people; nothing acts on it."""
    starts: dict[str, dict[str, Any]] = {}
    per: dict[str, dict[str, Any]] = {}
    for r in records:
        if r["kind"] == "start":
            starts[r["alloc"]["id"]] = r["alloc"]
        if r["kind"] != "end" or r["t"] < since or not r.get("gpus"):
            continue
        who = starts.get(r["alloc"], {}).get("job", {}).get("submitter", "?")
        row = per.setdefault(who, {"submitter": who, "leases": 0, "held_s": 0, "busy_s": 0, "sampled_s": 0})
        row["leases"] += 1
        for g in r["gpus"]:
            for k in ("held_s", "busy_s", "sampled_s"):
                row[k] += g[k]
    rows = list(per.values())
    for row in rows:
        row["busy_share"] = round(row["busy_s"] / row["sampled_s"], 3) if row["sampled_s"] else None
    return sorted(rows, key=lambda r: (r["busy_share"] is None, r["busy_share"] or 0, r["submitter"]))


def failures(records: Iterable[dict[str, Any]], since: int = 0) -> list[dict[str, Any]]:
    """Everything since `since` that didn't end well, newest first: a nonzero exit, an expiry, an eviction, a refusal, an
    idle lease. One feed for the whole cluster, each line naming the job, its submitter and the node."""
    starts: dict[str, dict[str, Any]] = {}
    out = []
    for r in records:
        if r["kind"] == "start":
            starts[r["alloc"]["id"]] = r["alloc"]
        if r["t"] < since or r["kind"] == "start":
            continue
        a = starts.get(r["alloc"], {}) if isinstance(r.get("alloc"), str) else {}
        who = {"job": a.get("job", {}).get("id", r.get("job")), "submitter": a.get("job", {}).get("submitter", r.get("submitter")),
               "node": a.get("node", r.get("node"))}
        if r["kind"] == "end" and r.get("rc") not in (0, None):
            out.append({"t": r["t"], "what": f"exit {r['rc']}", "run": r.get("run"), **who})
        elif r["kind"] == "expire":
            out.append({"t": r["t"], "what": "hit max_s", **who})
        elif r["kind"] == "evict":
            out.append({"t": r["t"], "what": f"evicted by {r['by']}", **who})
        elif r["kind"] == "refuse":
            out.append({"t": r["t"], "what": f"refused: {r['reason']}", **who})
        elif r["kind"] == "idle":
            out.append({"t": r["t"], "what": f"GPUs idle {r['idle_s']} s", **who})
    return sorted(out, key=lambda x: -x["t"])


def quiet_report(records: list[dict[str, Any]], alloc: str, *, until: int | None = None) -> dict[str, Any]:
    """What else was active on `alloc`'s node while it ran. `quiet` is true when every overlap is a frozen allocation or a
    quiet-safe service, and no admin session overlapped. An interval still open counts up to `until` (default: open ends
    at the last record's time)."""
    allocs, admin = history(records)
    me = allocs[alloc]
    end = until if until is not None else records[-1]["t"]
    mine = [(t0, t1 if t1 is not None else end) for s, t0, t1 in me.intervals if s == "running"]
    overlaps = []
    for other in allocs.values():
        if other.alloc == alloc or other.node != me.node:
            continue
        for state, t0, t1 in other.intervals:
            t1 = t1 if t1 is not None else end
            for a0, a1 in mine:
                lo, hi = max(a0, t0), min(a1, t1)
                if lo < hi:
                    overlaps.append({"alloc": other.alloc, "state": state, "t0": lo, "t1": hi,
                                     "quiet_safe": bool(other.job.get("quiet_safe")), "who": other.job.get("submitter"),
                                     "gpus": other.gpus, "run": other.job.get("run"),
                                     "preempt": other.job.get("preempt", "requeue")})
    sessions = [s for s in admin if s["node"] == me.node
                and any(max(a0, s["t0"]) < min(a1, s["t1"] if s["t1"] is not None else end) for a0, a1 in mine)]
    active = [o for o in overlaps if o["state"] == "running" and not o["quiet_safe"]]
    return {"alloc": alloc, "node": me.node, "run": me.job.get("run"), "gpus": me.gpus, "running": mine, "overlaps": overlaps, "admin": sessions,
            "quiet": not active and not sessions}
