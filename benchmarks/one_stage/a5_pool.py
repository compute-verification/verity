"""a5's session pool: an audit's sampled members' sessions and their Lean checks side by side, admitted by memory.

AGENTS.md's rule for outer proofs ("Live coins only") is independent sessions: each staged statement is fixed before any
session flips coins and gets exactly one session, with its own live coins in its own ZK server pass; no statement reads
another session's transcript or coins; no retries. The rule orders nothing among the sessions, so the pool runs them side by
side, and it keeps the rest by construction: it takes its jobs only once every statement is staged (`a5.audit`), starts each job
exactly once, never starts one again whatever its outcome, and lets a job's exception leave the pool (`run`), so a failed
session is a visible failure.

A job is a member's session or the Lean checks of a finished session, and needs memory: its template's measured peak plus
`HEADROOM` (`Needs`, from ``--pool-memory``). `Schedule` admits a waiting job when a slot of its kind is free
(``--session-jobs``, ``--lean-jobs``) and its need fits under the budget (``--memory-budget-gb``) beside the jobs running:

* Lean checks wait ahead of sessions, oldest first, so a check never queues behind the sessions still to run; sessions wait
  largest first (ties in the members' order), so the big ones are not left to run alone at the end.
* A job whose need alone is over the budget, or whose template has no measured peak, is exclusive: it is admitted only when
  nothing runs, and while it waits at its turn nothing behind it is admitted, so it can't be starved.
* With no budget only the slots bound the pool, and one slot of each kind is the members' order, one session at a time.

Side-by-side provers share the cores by `threads`.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: a job is admitted at its template's measured peak plus this share of it
HEADROOM = 0.10
SESSION, LEAN = "session", "lean"
KINDS = (SESSION, LEAN)


def jobs(s: str) -> int:
    """``--session-jobs`` and ``--lean-jobs``: how many run side by side, at least 1."""
    if not re.fullmatch(r"[1-9][0-9]*", s.strip()):
        raise argparse.ArgumentTypeError(f"{s}: a pool runs at least 1 job at a time; pass a positive whole number")
    return int(s)


def size_class(template: str, k_log: int | None) -> str | None:
    """A template's size class, ``<Definition name>@k_log=<K>``: what a measured peak is filed under when no run has measured
    the template itself."""
    name = re.match(r"[A-Za-z0-9_]+", template)
    return f"{name.group(0)}@k_log={k_log}" if name and k_log is not None else None


@dataclass(frozen=True)
class Needs:
    """Measured peak memory in GB by job kind, then by template or size class (`size_class`): ``--pool-memory``'s file,
    ``{"session": {KEY: GB}, "lean": {KEY: GB}}``. `need` is a job's admission need, None for a template nothing measured."""
    peaks: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path | None) -> Needs:
        if path is None:
            return cls()
        doc = json.loads(Path(path).read_text())
        bad = [k for k in doc if k not in KINDS] + [f"{k}/{t}" for k in KINDS for t, gb in doc.get(k, {}).items()
                                                   if isinstance(gb, bool) or not isinstance(gb, (int, float)) or gb <= 0]
        if bad:
            raise ValueError(f"{path}: a pool-memory file is {{\"session\": {{KEY: GB}}, \"lean\": {{KEY: GB}}}} with each GB a "
                             f"positive number; not {bad[:5]}")
        return cls({k: dict(doc.get(k, {})) for k in KINDS})

    def need(self, kind: str, template: str, k_log: int | None = None) -> float | None:
        table = self.peaks.get(kind, {})
        peak = table.get(template, table.get(size_class(template, k_log) or ""))
        return None if peak is None else peak * (1 + HEADROOM)


def threads(need_gb: float | None, budget_gb: float | None, session_jobs: int, cores: int) -> int:
    """A side-by-side prover's threads: the cores' share its memory need is of the budget, and at least an even share among
    ``session_jobs``, so a large member's prover is not held to a small one's threads while only a few of its size fit."""
    even = 1 / session_jobs
    share = max(even, need_gb / budget_gb) if need_gb and budget_gb else even
    return max(1, min(cores, round(cores * min(1.0, share))))


@dataclass
class Job:
    """One job of the pool: ``key`` names it, alone among the pool's jobs, and ``need_gb`` is its admission need (None:
    unmeasured)."""
    kind: str
    key: str
    need_gb: float | None
    run: Callable[[], Any]


class Schedule:
    """The pool's admission, with no thread or clock in it: jobs join as they become ready (`add`), `start` hands back the
    waiting jobs admitted now, and `finish` frees a running job's slot and memory. A job is handed back once and never again."""

    def __init__(self, limits: dict[str, int], budget_gb: float | None = None):
        if budget_gb is not None and budget_gb <= 0:
            raise ValueError(f"--memory-budget-gb {budget_gb}: a budget is a positive number of GB")
        self.limits, self.budget = dict(limits), budget_gb
        self.waiting: dict[str, list[Job]] = {k: [] for k in KINDS}
        self.running: list[Job] = []
        self.seen: set[str] = set()
        self.peak_gb, self.peak_jobs = 0.0, 0

    def exclusive(self, job: Job) -> bool:
        return self.budget is not None and (job.need_gb is None or job.need_gb > self.budget)

    def add(self, job: Job) -> None:
        if job.key in self.seen:
            raise RuntimeError(f"{job.key}: a job joins the pool once; nothing is rerun")
        self.seen.add(job.key)
        q = self.waiting[job.kind]
        if job.kind == SESSION and self.budget is not None:     # largest first, an unmeasured one as the largest; ties in order
            size = float("inf") if job.need_gb is None else job.need_gb
            at = next((i for i, w in enumerate(q) if (float("inf") if w.need_gb is None else w.need_gb) < size), len(q))
            q.insert(at, job)
        else:
            q.append(job)

    def used_gb(self) -> float:
        return sum(j.need_gb or 0.0 for j in self.running)

    def start(self) -> list[Job]:
        started: list[Job] = []
        for kind in (LEAN, SESSION):
            q = self.waiting[kind]
            i = 0
            while i < len(q):
                job = q[i]
                if sum(1 for j in self.running if j.kind == kind) >= self.limits[kind]:
                    break
                if self.exclusive(job):
                    if self.running:
                        return started      # its turn: nothing behind it starts until it has run alone
                    fits = True
                else:
                    fits = (self.budget is None or not self.running or
                            (not any(self.exclusive(j) for j in self.running) and self.used_gb() + job.need_gb <= self.budget))
                if fits:
                    self.running.append(q.pop(i))
                    started.append(job)
                    self.peak_gb, self.peak_jobs = max(self.peak_gb, self.used_gb()), max(self.peak_jobs, len(self.running))
                    if self.exclusive(job):
                        return started
                else:
                    i += 1
        return started

    def finish(self, job: Job) -> None:
        self.running.remove(job)

    def idle(self) -> bool:
        return not self.running and not any(self.waiting.values())


def run(sessions: Sequence[Job], lean_of: Callable[[Job, Any], Job], limits: dict[str, int],
        budget_gb: float | None = None) -> tuple[list[Any], dict]:
    """Run every session once and, after each, its Lean checks (``lean_of(session, its result)``) once, by `Schedule`;
    -> each session's Lean checks' result in ``sessions``' order, and the pool's record. Every job runs in its own thread. A
    job that raises is not run again: the pool starts nothing more, waits for the jobs already running, and raises the first
    exception."""
    sched = Schedule(limits, budget_gb)
    results: list[Any] = [None] * len(sessions)
    place = {j.key: i for i, j in enumerate(sessions)}
    cv = threading.Condition()
    finished: list[tuple[Job, Any, BaseException | None]] = []
    log: list[dict] = []
    failed: list[BaseException] = []

    def work(job: Job) -> None:
        try:
            out, err = job.run(), None
        except BaseException as e:      # noqa: BLE001 - handed to the pool's caller below
            out, err = None, e
        with cv:
            finished.append((job, out, err))
            cv.notify()

    for s in sessions:
        sched.add(s)
    of: dict[str, int] = {}     # a Lean job -> its session's place
    with cv:
        while not sched.idle():
            if not failed:
                for job in sched.start():
                    log.append({"start": job.key, "kind": job.kind, "need_gb": job.need_gb, "beside": len(sched.running) - 1,
                                "used_gb": round(sched.used_gb(), 1)})
                    threading.Thread(target=work, args=(job,), name=f"a5-pool-{job.key}", daemon=True).start()
            if not sched.running:
                break       # a failure left jobs waiting: they never start
            while not finished:
                cv.wait()
            for job, out, err in finished:
                sched.finish(job)
                if err is not None:
                    failed.append(err)
                elif job.kind == SESSION:
                    check = lean_of(job, out)
                    of[check.key] = place[job.key]
                    sched.add(check)
                else:
                    results[of[job.key]] = out
            finished.clear()
    if failed:
        raise failed[0]
    return results, {"session_jobs": limits[SESSION], "lean_jobs": limits[LEAN], "memory_budget_gb": budget_gb,
                     "headroom": HEADROOM, "peak_admitted_gb": round(sched.peak_gb, 1), "most_side_by_side": sched.peak_jobs,
                     "unmeasured": sorted(e["start"] for e in log if e["need_gb"] is None), "starts": log}
