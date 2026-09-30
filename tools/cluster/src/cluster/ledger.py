"""The allocation ledger: an append-only, hash-chained JSON-lines log of what each node's scheduler did.

Every record carries `seq`, `t` (epoch seconds), `kind`, its fields, `prev` (the previous record's hash) and `hash`
(sha256 of the record's canonical JSON without `hash`). Kinds: `start` (an allocation), `freeze` and `evict` (with `by`),
`thaw`, `expire`, `end` (with `rc`, the run id when it was a recorded run, and `gpus`: each GPU's held, busy and sampled
seconds from `gpu_usage`), `refuse` (a job and its reason), `idle` (GPUs held without computing, a notice only), and
`admin-begin` / `admin-end` for a repair session on a node (who and why), which no scheduler decided and every quiet
report must see.

The ledger is the evidence behind a run's placement: `history` replays it into each allocation's running and frozen
intervals, and `quiet_report` says, for one allocation, what else was active on its node while it ran, which is what a
timed row's claim that "the node was quiet" rests on.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import model
from .plan import Action, Evict, Expire, Freeze, Idle, Refuse, Start, Thaw

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != "hash"})).hexdigest()


class Ledger:
    """Appends to one ledger file. Opening checks the chain, so a writer never extends a broken one."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        records = read(self.path) if self.path.exists() else []
        problems = verify(records)
        if problems:
            raise ValueError(f"{self.path}: {problems[0]}")
        self.seq = records[-1]["seq"] if records else 0
        self.last = records[-1]["hash"] if records else GENESIS
        self.last_t = records[-1]["t"] if records else None

    def append(self, t: int, kind: str, **fields: Any) -> dict[str, Any]:
        if self.last_t is not None and t < self.last_t:
            raise ValueError(f"record at {t} is older than the ledger's last, {self.last_t}")
        record = {"seq": self.seq + 1, "t": t, "kind": kind, **fields, "prev": self.last}
        record["hash"] = digest(record)
        with self.path.open("a") as f:
            f.write(canonical(record).decode() + "\n")
        self.seq, self.last, self.last_t = record["seq"], record["hash"], t
        return record

    def actions(self, t: int, actions: Iterable[Action]) -> list[dict[str, Any]]:
        """Record the actions an executor carried out at `t` (a Wait is not recorded: nothing happened)."""
        out = []
        for a in actions:
            if isinstance(a, Start):
                out.append(self.append(t, "start", alloc=model.allocation_to_dict(a.alloc)))
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
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
                                     "quiet_safe": bool(other.job.get("quiet_safe"))})
    sessions = [s for s in admin if s["node"] == me.node
                and any(max(a0, s["t0"]) < min(a1, s["t1"] if s["t1"] is not None else end) for a0, a1 in mine)]
    active = [o for o in overlaps if o["state"] == "running" and not o["quiet_safe"]]
    return {"alloc": alloc, "node": me.node, "running": mine, "overlaps": overlaps, "admin": sessions,
            "quiet": not active and not sessions}
