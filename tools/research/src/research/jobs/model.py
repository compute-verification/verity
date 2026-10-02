"""The Job queue's reference semantics: what Control's Job queue (the site's `/api/jobs`, on Neon) does, as an in-memory model.

The Job queue holds batch, unattended work on pods: a job is added, claimed by a worker under a lease, renewed while it runs,
and completed with an outcome.  An agent's interactive session on a pod is a Broker allocation, not a job.

This is the executable spec of the job-service design (§2, §3) and of the site spec.  `vectors.json` beside it pins every rule
as request/response cases; the site's TypeScript runs the same file, and `tests/test_jobs_vectors.py` runs it here.  The
dispatcher's tests use this model as their server.

`JobService.request(method, path, token, body, now)` is the HTTP contract: it returns `(status, body)`. `tick(now)` is the
expire-leases cron. Times are seconds since `epoch`, written in responses as ISO 8601 (`iso`).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

KINDS = ("merge-check", "lean-regen", "audit-main", "research-run")
SCOPES = ("jobs:enqueue", "jobs:work", "events:read")
STATES = ("queued", "running", "done", "failed", "dead", "cancelled")
OUTCOMES = ("passed", "failed", "error")
#: requirement -> the capability it reads: a bool needs the capability true, a `min_` number needs it at least that, a list
#: needs every name in the capability's list
REQUIREMENTS = {"avx512": "avx512", "min_mem_gb": "mem_gb", "min_cpus": "cpus", "min_disk_gb": "disk_gb", "min_lease_s": "lease_s",
                "tools": "tools"}
DEFAULT_LEASE_S = 900
#: (min, max) lease_s by kind, checked in code, not by the database, so the interactive kind can carry leases of hours
LEASE_BOUNDS = {kind: (60, 3600) for kind in KINDS}
DEFAULT_MAX_ATTEMPTS = 3
#: a budget line's prefix, as budgets.toml names them (`vy-coord-`)
LINE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}-")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class Token:
    name: str
    scopes: tuple[str, ...]
    revoked: bool = False


@dataclass
class Worker:
    id: str                            # a RunPod pod id, or 'dispatch:{name}' for a stage-1 dispatcher
    kind: str                          # pod | dispatcher
    token: str                         # the name of the one jobs:work token that may claim as this worker
    kinds: tuple[str, ...]             # the job kinds it may claim
    capabilities: dict = field(default_factory=dict)   # as its last claim reported them
    last_seen: float | None = None
    retired: bool = False


@dataclass
class Job:
    id: int
    kind: str
    payload: dict
    priority: int
    requirements: dict
    dedupe_key: str | None
    enqueued_by: str
    enqueued_at: float
    not_before: float
    max_attempts: int
    lease_s: int
    line: str | None = None            # the budget line the job's compute is charged to
    state: str = "queued"
    attempts: int = 0
    lease_worker: str | None = None
    lease_slot: str | None = None
    lease_hash: str | None = None
    lease_expires: float | None = None
    cancel_requested: bool = False
    progress: dict | None = None
    result: dict | None = None
    finished_at: float | None = None


@dataclass
class Attempt:
    job_id: int
    n: int
    worker: str
    slot: str | None
    claimed_at: float
    renewed_at: float | None = None
    ended_at: float | None = None
    outcome: str | None = None         # passed | failed | error | lease-lost | cancelled
    run_id: str | None = None
    detail: str | None = None


@dataclass
class JobService:
    epoch: str = "2026-09-29T00:00:00Z"
    tokens: dict[str, Token] = field(default_factory=dict)      # by the token's hash, as the site stores them
    workers: dict[str, Worker] = field(default_factory=dict)
    jobs: dict[int, Job] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    lease_tokens: int = 0

    # -- setup (the site's Console: minting tokens, registering workers)

    def mint(self, secret: str, name: str, scopes: list[str]) -> None:
        self.tokens[sha256(secret)] = Token(name, tuple(scopes))

    def register(self, worker_id: str, kind: str, token: str, kinds: list[str]) -> None:
        self.workers[worker_id] = Worker(worker_id, kind, token, tuple(kinds))

    def retire(self, worker_id: str) -> None:
        self.workers[worker_id].retired = True

    def revoke(self, name: str) -> None:
        for t in self.tokens.values():
            if t.name == name:
                t.revoked = True

    # -- helpers

    def iso(self, t: float | None) -> str | None:
        if t is None:
            return None
        base = dt.datetime.fromisoformat(self.epoch.replace("Z", "+00:00"))
        return (base + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def seconds(self, iso: str) -> float:
        base = dt.datetime.fromisoformat(self.epoch.replace("Z", "+00:00"))
        return (dt.datetime.fromisoformat(iso.replace("Z", "+00:00")) - base).total_seconds()

    def _event(self, now: float, kind: str, subject: str, actor: str, payload: dict) -> None:
        self.events.append({"id": len(self.events) + 1, "at": self.iso(now), "source": "jobs", "kind": kind, "subject": subject,
                            "actor": actor, "payload": payload})

    def _auth(self, token: str | None, scope: str) -> Token:
        t = self.tokens.get(sha256(token)) if token else None
        if t is None or t.revoked:
            raise Refused(401, "a valid token is needed")
        if scope not in t.scopes:
            raise Refused(403, f"the token lacks {scope}")
        return t

    def view(self, j: Job) -> dict:
        return {"id": j.id, "kind": j.kind, "state": j.state, "priority": j.priority, "payload": j.payload,
                "requirements": j.requirements, "dedupe_key": j.dedupe_key, "enqueued_by": j.enqueued_by,
                "enqueued_at": self.iso(j.enqueued_at), "not_before": self.iso(j.not_before), "attempts": j.attempts,
                "max_attempts": j.max_attempts, "lease_s": j.lease_s, "line": j.line, "lease_worker": j.lease_worker, "lease_slot": j.lease_slot,
                "lease_expires_at": self.iso(j.lease_expires), "cancel_requested": j.cancel_requested, "progress": j.progress,
                "result": j.result, "finished_at": self.iso(j.finished_at)}

    def history(self, j: Job) -> list[dict]:
        return [{"n": a.n, "worker": a.worker, "slot": a.slot, "claimed_at": self.iso(a.claimed_at),
                 "renewed_at": self.iso(a.renewed_at), "ended_at": self.iso(a.ended_at), "outcome": a.outcome, "run_id": a.run_id,
                 "detail": a.detail} for a in self.attempts if a.job_id == j.id]

    def _attempt(self, j: Job) -> Attempt:
        return next(a for a in reversed(self.attempts) if a.job_id == j.id and a.n == j.attempts)

    def _job(self, job_id: str) -> Job:
        j = self.jobs.get(int(job_id)) if job_id.isdigit() else None
        if j is None:
            raise Refused(404, "no such job")
        return j

    def _lease(self, j: Job, lease_token: Any) -> None:
        """A lease token that isn't this job's current one, whether lost to expiry, ended, or never held, is one answer: 409.
        The worker's rule is then always the same: stop, and report nothing more."""
        if not isinstance(lease_token, str) or j.lease_hash is None or sha256(lease_token) != j.lease_hash:
            raise Refused(409, f"the lease is not current (the job is {j.state})")

    # -- the API

    def request(self, method: str, path: str, token: str | None, body: dict | None, now: float) -> tuple[int, dict]:
        try:
            if body is not None and not isinstance(body, dict):
                raise Refused(400, "the body must be a JSON object")
            for k in ("slot", "run_id", "summary"):
                if (body or {}).get(k) is not None and not isinstance(body[k], str):
                    raise Refused(400, f"{k} must be a string")
            return self._route(method, path.split("?")[0], _query(path), token, body or {}, now)
        except Refused as r:
            return r.status, {"error": r.error}

    def _route(self, method, path, query, token, body, now):
        if method == "POST" and path == "/api/jobs":
            return self.add(self._auth(token, "jobs:enqueue"), body, now)
        if method == "POST" and path == "/api/jobs/claim":
            return self.claim(self._auth(token, "jobs:work"), body, now)
        if method == "GET" and path == "/api/jobs":
            self._auth_any(token)
            rows = [j for j in self.jobs.values() if (not query.get("state") or j.state == query["state"])
                    and (not query.get("kind") or j.kind == query["kind"])]
            return 200, {"jobs": [self.view(j) for j in sorted(rows, key=lambda j: -j.id)[:_limit(query)]]}
        if method == "GET" and path == "/api/events":
            self._auth(token, "events:read")
            if not (query.get("after") or "0").isdigit():
                raise Refused(400, "after must be an event id")
            after = int(query.get("after") or 0)
            rows = [e for e in self.events if e["id"] > after and (not query.get("kind") or e["kind"] == query["kind"])
                    and (not query.get("subject") or e["subject"] == query["subject"])][:_limit(query)]
            return 200, {"events": rows, "next": rows[-1]["id"] if rows else after}
        m = re.fullmatch(r"/api/jobs/([^/]+)(?:/(renew|complete|cancel))?", path)
        if m and method == "GET" and m.group(2) is None:
            self._auth_any(token)
            j = self._job(m.group(1))
            return 200, {**self.view(j), "history": self.history(j)}
        if m and method == "POST" and m.group(2) == "renew":
            return self.renew(self._auth(token, "jobs:work"), self._job(m.group(1)), body, now)
        if m and method == "POST" and m.group(2) == "complete":
            return self.complete(self._auth(token, "jobs:work"), self._job(m.group(1)), body, now)
        if m and method == "POST" and m.group(2) == "cancel":
            return self.cancel(self._auth(token, "jobs:enqueue"), self._job(m.group(1)), now)
        raise Refused(404, "no such route")

    def _auth_any(self, token: str | None) -> Token:
        t = self.tokens.get(sha256(token)) if token else None
        if t is None or t.revoked:
            raise Refused(401, "a valid token is needed")
        return t

    def add(self, t: Token, body: dict, now: float) -> tuple[int, dict]:
        kind, payload = body.get("kind"), body.get("payload")
        if kind not in KINDS:
            raise Refused(400, f"kind must be one of {', '.join(KINDS)}")
        if not isinstance(payload, dict):
            raise Refused(400, "payload must be an object")
        req = body.get("requirements") or {}
        if not isinstance(req, dict) or set(req) - set(REQUIREMENTS):
            raise Refused(400, f"requirements may only name {', '.join(REQUIREMENTS)}")
        for name, want in req.items():
            ok = (isinstance(want, bool) if name == "avx512" else
                  isinstance(want, list) and all(isinstance(x, str) for x in want) if name == "tools" else
                  _number(want) and want >= 0)
            if not ok:
                raise Refused(400, f"requirements.{name} must be "
                                   + {"avx512": "true or false", "tools": "a list of names"}.get(name, "a number of at least 0"))
        prio, lease_s = body.get("priority", 0), body.get("lease_s", DEFAULT_LEASE_S)
        max_attempts = body.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        if not (_int(prio) and -1000 <= prio <= 1000):
            raise Refused(400, "priority must be an integer from -1000 to 1000")
        lo, hi = LEASE_BOUNDS[kind]
        if not (_int(lease_s) and lo <= lease_s <= hi):
            raise Refused(400, f"lease_s must be an integer from {lo} to {hi} for {kind}")
        check_line(body.get("line"))
        if not (_int(max_attempts) and 1 <= max_attempts <= 10):
            raise Refused(400, "max_attempts must be an integer from 1 to 10")
        key = body.get("dedupe_key")
        if key is not None and not (isinstance(key, str) and 0 < len(key) <= 200):
            raise Refused(400, "dedupe_key must be a string of 1 to 200 characters")
        if key is not None:
            for j in self.jobs.values():
                if j.dedupe_key == key:
                    return 200, {"id": j.id, "created": False, "state": j.state}
        try:
            not_before = self.seconds(body["not_before"]) if body.get("not_before") is not None else now
        except (TypeError, AttributeError, ValueError):
            raise Refused(400, "not_before must be an ISO 8601 time") from None
        j = Job(len(self.jobs) + 1, kind, payload, prio, req, key, t.name, now, not_before, max_attempts, lease_s, body.get("line"))
        self.jobs[j.id] = j
        self._event(now, "job.added", f"job:{j.id}", t.name, {"kind": kind, "dedupe_key": key})
        return 201, {"id": j.id, "created": True, "state": j.state}

    def claim(self, t: Token, body: dict, now: float) -> tuple[int, dict]:
        w = self.workers.get(body.get("worker")) if isinstance(body.get("worker"), str) else None
        if w is None or w.token != t.name or w.retired:
            raise Refused(403, "the token may not claim as this worker")
        worker = w.id
        kinds = body.get("kinds") or list(w.kinds)
        if set(kinds) - set(w.kinds):
            raise Refused(403, f"this worker may claim only {', '.join(w.kinds)}")
        caps = body.get("capabilities") or {}
        if not isinstance(caps, dict):
            raise Refused(400, "capabilities must be an object")
        w.capabilities, w.last_seen = caps, now
        ready = [j for j in self.jobs.values() if j.state == "queued" and j.kind in kinds and j.not_before <= now
                 and meets(j.requirements, caps)]
        if not ready:
            return 204, {}
        j = min(ready, key=lambda j: (-j.priority, j.enqueued_at, j.id))
        self.lease_tokens += 1
        lease = f"lease-{j.id}-{j.attempts + 1}-{self.lease_tokens}"    # the site uses 32 random bytes; tokens are never stored
        j.state, j.attempts, j.lease_worker, j.lease_slot = "running", j.attempts + 1, worker, body.get("slot")
        j.lease_hash, j.lease_expires, j.progress = sha256(lease), now + j.lease_s, None
        self.attempts.append(Attempt(j.id, j.attempts, worker, body.get("slot"), now))
        self._event(now, "job.claimed", f"job:{j.id}", t.name, {"attempt": j.attempts, "worker": worker, "slot": body.get("slot")})
        return 200, {"job": {"id": j.id, "kind": j.kind, "payload": j.payload, "requirements": j.requirements},
                     "attempt": j.attempts, "lease_token": lease, "lease_s": j.lease_s,
                     "lease_expires_at": self.iso(j.lease_expires), "grants": {}}

    def renew(self, t: Token, j: Job, body: dict, now: float) -> tuple[int, dict]:
        self._lease(j, body.get("lease_token"))
        j.lease_expires = now + j.lease_s
        if isinstance(body.get("progress"), dict):
            j.progress = body["progress"]
        self._attempt(j).renewed_at = now
        return 200, {"lease_expires_at": self.iso(j.lease_expires), "cancel": j.cancel_requested}

    def complete(self, t: Token, j: Job, body: dict, now: float) -> tuple[int, dict]:
        self._lease(j, body.get("lease_token"))
        outcome = body.get("outcome")
        if outcome not in OUTCOMES:
            raise Refused(400, f"outcome must be one of {', '.join(OUTCOMES)}")
        a = self._attempt(j)
        a.ended_at, a.run_id, a.detail = now, body.get("run_id"), body.get("summary")
        a.outcome = "cancelled" if j.cancel_requested else outcome
        result = {"outcome": outcome, "run_id": body.get("run_id"), "summary": body.get("summary")}
        j.lease_hash = j.lease_worker = j.lease_slot = j.lease_expires = None
        if j.cancel_requested:
            j.state, j.result, j.finished_at = "cancelled", result, now
        elif outcome == "passed":
            j.state, j.result, j.finished_at = "done", result, now
        elif outcome == "failed":                                       # a failure is a verdict: final, never retried
            j.state, j.result, j.finished_at = "failed", result, now
        elif j.attempts < j.max_attempts:                               # an error is no verdict: another worker tries again
            j.state, j.result = "queued", result
        else:
            j.state, j.result, j.finished_at = "dead", result, now
        self._event(now, "job.finished", f"job:{j.id}", t.name, {"attempt": a.n, "outcome": outcome, "state": j.state,
                                                                  "run_id": body.get("run_id")})
        if j.state == "dead":
            self._event(now, "job.dead", f"job:{j.id}", t.name, {"attempts": j.attempts})
        return 200, {"state": j.state}

    def cancel(self, t: Token, j: Job, now: float) -> tuple[int, dict]:
        if j.enqueued_by != t.name:
            raise Refused(403, "only the token that added a job cancels it")
        if j.state == "queued":
            j.state, j.finished_at = "cancelled", now
            self._event(now, "job.cancelled", f"job:{j.id}", t.name, {"was": "queued"})
        elif j.state == "running":
            j.cancel_requested = True                                   # the worker learns it at its next renew
            self._event(now, "job.cancelled", f"job:{j.id}", t.name, {"was": "running"})
        else:
            raise Refused(409, f"the job is already {j.state}")
        return 200, {"state": j.state, "cancel_requested": j.cancel_requested}

    # -- the expire-leases cron

    def tick(self, now: float) -> list[int]:
        """Every running job whose lease has passed: `lease-lost`, then queued again, or dead after `max_attempts`, or
        cancelled when a cancel was asked. Returns the ids it moved."""
        moved = []
        for j in sorted(self.jobs.values(), key=lambda j: j.id):
            if j.state != "running" or j.lease_expires is None or j.lease_expires >= now:
                continue
            a = self._attempt(j)
            a.ended_at, a.outcome = now, "lease-lost"
            j.lease_hash = j.lease_worker = j.lease_slot = j.lease_expires = None
            if j.cancel_requested:
                j.state, j.finished_at = "cancelled", now
            elif j.attempts < j.max_attempts:
                j.state = "queued"
            else:
                j.state, j.finished_at = "dead", now
            self._event(now, "job.lease_lost", f"job:{j.id}", "cron", {"attempt": a.n, "state": j.state})
            if j.state == "dead":
                self._event(now, "job.dead", f"job:{j.id}", "cron", {"attempts": j.attempts})
            moved.append(j.id)
        return moved


class Refused(Exception):
    def __init__(self, status: int, error: str):
        super().__init__(error)
        self.status, self.error = status, error


def meets(requirements: dict, capabilities: dict) -> bool:
    """Whether a worker's capabilities meet a job's requirements (`REQUIREMENTS`)."""
    for req, want in requirements.items():
        have = capabilities.get(REQUIREMENTS[req])
        if isinstance(want, bool):
            if want and have is not True:
                return False
        elif isinstance(want, list):
            if not isinstance(have, list) or set(want) - set(have):
                return False
        elif not _number(have) or have < want:
            return False
    return True


def check_line(line: Any) -> None:
    if line is not None and not (isinstance(line, str) and LINE.fullmatch(line)):
        raise Refused(400, "line must be a budget line's prefix, such as vy-coord-")


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)          # JSON's true is no integer


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _limit(query: dict[str, str]) -> int:
    """`limit`: 1 to 1000, default 100."""
    raw = query.get("limit") or "100"
    if not raw.isdigit() or not 1 <= int(raw) <= 1000:
        raise Refused(400, "limit must be an integer from 1 to 1000")
    return int(raw)


def _query(path: str) -> dict[str, str]:
    if "?" not in path:
        return {}
    return dict(p.split("=", 1) for p in path.split("?", 1)[1].split("&") if "=" in p)
