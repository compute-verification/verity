"""The shadow: the node agent's planner run beside today's `gpu-lease` on vy-nebius-2, deciding and never acting.

Each `step` takes one observation of the node (`nebius2.read` live, `nebius2.from_sampler` in a replay) and the
`preempted.log` lines since the last one, and:

1. mirrors what `gpu-lease` did into the ledger: a waiter is a `submit` (its wait file the handle), a new lease a `start`
   with `source = "gpu-lease"`, a lease gone an `end`, or an `evict` by its requester when `preempted.log` stopped it, and a
   waiter gone without a lease a `withdraw`. So the ledger, and `state.json` from it, is the node's lease state as the agent
   would hold it after the switch;
2. plans the node from that state and the queue, and for each request records when the planner would first have started
   it and what it would have evicted, beside when `gpu-lease` decided (its first SIGTERM for the request, else the lease's
   start). A request `gpu-lease` granted before the planner would have is planned again without its own lease (a lease
   granted at once, never seen waiting, is planned only that way);
3. checks the plan with `invariants.check`: a rule the planned state breaks that the observed one doesn't is a *safety*
   divergence. So is a timed window the planner would have started more than 5 s after `gpu-lease` did, or a request
   left waiting over 10 minutes while enough GPUs sat free. Any other difference is a *design* divergence, listed for
   review. A rule the node already breaks (memory its leases overcommit) is *observed*, recorded once each.

`gpu-lease` takes a GPU's lock before it writes the owner line, so for an instant a held lock shows the last holder's
line. An owner line whose allocation has already ended is that: the GPU is planned as held by nobody the planner may move
(`nebius2.unowned`), and nothing is recorded until the new line is written. So is a lock held under a line naming a pid
that doesn't hold it (`Observation.held`: the take in progress, or a dead holder's child still holding the descriptor).

It reads no clock (the observation carries `t`) and writes only its output directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from . import invariants, ledger, nebius2
from .ledger import epoch
from .model import Allocation, Cluster, Job
from .plan import Evict, Start, Wait, Refuse, apply, plan

WINDOW_LAG_S = 5
GONE_S = 30
STARVE_S = 600
EARLY_S = 60


@dataclass
class Request:
    job: Job
    handle: str | None
    t_req: int
    seen_waiting: bool
    real: int | None = None
    granted: int | None = None
    shadow: int | None = None
    shadow_evicts: tuple[str, ...] = ()
    real_evicts: list[str] = field(default_factory=list)
    reason: str | None = None
    starved: bool = False
    planned: bool = False
    beside: tuple[str, ...] = ()
    gone: int | None = None

    def record(self, outcome: str) -> dict[str, Any]:
        lag = None if self.shadow is None or self.real is None else self.shadow - self.real
        return {"job": self.job.id, "who": self.job.submitter, "gpus": self.job.gpus, "class": self.job.priority,
                "outcome": outcome, "t_req": self.t_req, "seen_waiting": self.seen_waiting, "real": self.real,
                "granted": self.granted, "shadow": self.shadow, "lag_s": lag, "shadow_evicts": list(self.shadow_evicts),
                "real_evicts": self.real_evicts, "reason": self.reason, "beside": list(self.beside)}


class Shadow:
    """`chain`: the earlier segments' ledger files, in order, which this one's ledger continues (`ledger.segments`)."""

    def __init__(self, cluster: Cluster, out: Path, node: str = nebius2.NODE, chain: Iterable[Path] = ()):
        self.cluster, self.node, self.out = cluster, node, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        files = [Path(p) for p in chain] + [self.out / "ledger.jsonl"]
        self.ledger = ledger.Ledger(self.out / "ledger.jsonl", chain=files[:-1])
        records = [r for f in files if f.exists() for r in ledger.read(f)] if self.ledger.seq else []
        self.state = ledger.state(records) if records else ledger.State({}, {})
        self.requests: dict[str, Request] = {}
        self.sigterm: dict[str, str] = {}
        self.stopped_for: dict[str, tuple[int, list[str]]] = {}
        self.observed: set[str] = set()
        self.ended: set[str] = set(ledger.history(records)[0]) - set(self.state.allocations) if records else set()
        self.unowned: list[Allocation] = []
        self.ticks = 0
        self.last_t: int | None = None

    def _emit(self, name: str, rec: dict[str, Any]) -> None:
        with (self.out / name).open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _diverge(self, t: int, cls: str, msg: str, **extra: Any) -> None:
        self._emit("divergences.jsonl", {"t": t, "class": cls, "msg": msg, **extra})

    def _append(self, recs: Iterable[dict[str, Any]]) -> None:
        self.state = ledger.state(list(recs), self.state)

    def roll(self, out: Path) -> None:
        """Write from now on to `out`, a new segment: its ledger continues this one's chain."""
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger.roll(self.out / "ledger.jsonl")
        self._div_offset = 0

    def _request(self, job: Job, handle: str | None, t_req: int, seen_waiting: bool) -> Request:
        r = self.requests.get(job.id)
        if r is None:
            r = self.requests[job.id] = Request(job, handle, t_req, seen_waiting)
        if job.id in self.stopped_for:
            first, victims = self.stopped_for[job.id]
            r.real = first if r.real is None else min(r.real, first)
            r.real_evicts = victims
        return r

    def step(self, obs: nebius2.Observation, preemptions: Iterable[dict[str, Any]] = ()) -> None:
        t = max(obs.t, self.ledger.last_t or obs.t)
        self.ticks += 1
        self.last_t = t
        allocs, self.unowned = {}, []
        for a in nebius2.allocations(self.cluster, obs, self.node):
            if a.id in self.ended:
                self.unowned.append(nebius2.unowned(self.cluster, a.gpus, self.node))
            else:
                allocs[a.id] = a
        self.unowned += [nebius2.unowned(self.cluster, (i,), self.node) for i in obs.held]
        waiting = {j.id: (j, h) for j, h in nebius2.queue(self.cluster, obs, self.node)} if obs.waits_known else None
        for p in preemptions:
            f = p["victim"]
            if "pid" in f and "since" in f:
                victim = f"gl-{f['pid']}@{epoch(f['since'])}"
                by = f"gl-{p['by']}"
                self.sigterm.setdefault(victim, by)
                first, victims = self.stopped_for.setdefault(by, (p["t"], []))
                if victim not in victims:
                    victims.append(victim)
        for r in self.requests.values():
            self._request(r.job, r.handle, r.t_req, r.seen_waiting)
        self._mirror(t, allocs, waiting)
        self._plan(t, allocs)

    def _mirror(self, t: int, allocs: dict[str, Allocation], waiting: dict[str, tuple[Job, str]] | None) -> None:
        recs = []
        for aid in sorted(set(self.state.allocations) - set(allocs)):
            self.ended.add(aid)
            by = self.sigterm.pop(aid, None)
            recs.append(self.ledger.append(t, "evict", alloc=aid, by=by) if by else self.ledger.end(t, aid))
        started_jobs = {a.job.id for a in allocs.values()}
        if waiting is not None:
            for jid in sorted(set(self.state.queue) - set(waiting) - started_jobs):
                r = self.requests.get(jid)
                if r is not None and r.gone is None:
                    r.gone = t
                if r is None or t - r.gone >= GONE_S:
                    recs.append(self.ledger.withdraw(t, jid, "handle dropped"))
                    if self.requests.pop(jid, None) is not None:
                        self._emit("decisions.jsonl", r.record("withdrawn"))
                        self.stopped_for.pop(jid, None)
            for jid in set(waiting) & set(self.requests):
                self.requests[jid].gone = None
            for jid, (job, handle) in sorted(waiting.items()):
                if jid not in self.state.queue:
                    recs.append(self.ledger.submit(t, job, self.node, handle))
                    self._request(job, handle, job.submitted, True)
                elif jid not in self.requests and jid not in started_jobs:
                    # queued before this agent started (a restart, or a handover from another agent)
                    self._request(job, handle, job.submitted, True)
        new = [a for aid, a in sorted(allocs.items()) if aid not in self.state.allocations]
        for a in new:
            recs += self.ledger.actions(t, [Start(a)], source="gpu-lease")
        self._append(recs)
        for a in new:
            r = self._request(a.job, None, a.started, False)
            r.granted = a.started
            r.real = a.started if r.real is None else min(r.real, a.started)
            if not r.seen_waiting:
                r.job = replace(a.job, gpu_ids=())

    def _plan(self, t: int, allocs: dict[str, Allocation]) -> None:
        observed = list(allocs.values()) + self.unowned
        base = set(invariants.check(self.cluster, self.node, observed, t))
        for msg in sorted(base - self.observed):
            self._diverge(t, "observed", msg)
        self.observed |= base
        pending = [r for r in self.requests.values() if r.granted is None and r.gone is None]
        if pending:
            self._decide(t, observed, [r.job for r in pending], pending, base)
        for r in [r for r in self.requests.values() if r.granted is not None]:
            if r.shadow is None:
                own = [a for a in observed if a.job.id != r.job.id]
                # the request as it was asked, not as gpu-lease placed it: the planner picks its own GPUs
                r.beside = tuple(sorted(a.id for a in own if a.job.preempt != "requeue"))
                queue = [r.job] + [p.job for p in pending]
                self._decide(t, own, queue, [r], set(invariants.check(self.cluster, self.node, own, t)))
            self._close(t, r)

    def _decide(self, t: int, observed: list[Allocation], queue: list[Job], reqs: list[Request], base: set[str]) -> None:
        """Plan `queue` beside `observed` at `t`. A request planned for the first time is decided at its request time: the
        agent answers a request when it arrives, and the observation only shows it at the next tick."""
        actions = plan(self.cluster, self.node, observed, queue, t)
        after = apply(observed, actions, t)
        broke = sorted(set(invariants.check(self.cluster, self.node, after, t, actions=actions, before=observed)) - base)
        for msg in broke:
            self._diverge(t, "safety", f"the plan breaks a rule: {msg}")
        starts = {a.alloc.job.id: a.alloc for a in actions if isinstance(a, Start)}
        why = {a.job: a.reason for a in actions if isinstance(a, (Wait, Refuse))}
        refused = {a.job for a in actions if isinstance(a, Refuse)}
        for r in reqs:
            jid = r.job.id
            first, r.planned = not r.planned, True
            if jid in starts and r.shadow is None:
                r.shadow = min(r.t_req, t) if first else t
                r.shadow_evicts = tuple(sorted(e.alloc for e in actions if isinstance(e, Evict) and e.by == jid))
            elif jid not in starts:
                r.reason = why.get(jid, r.reason)
                free = len(self.cluster.nodes[self.node].gpus) - len({g for a in after for g in a.gpus})
                if (not r.starved and r.granted is None and jid not in refused and t - r.t_req > STARVE_S
                        and free >= r.job.gpus and not r.job.gpu_ids):
                    r.starved = True
                    held = (r.reason or "").startswith(("node is quiet", f"{self.node} is reserved for"))
                    self._diverge(t, "design" if held else "safety",
                                  f"{jid} waits {t - r.t_req} s with {free} GPUs free: {r.reason}", job=jid)

    def _close(self, t: int, r: Request) -> None:
        jid = r.job.id
        rec = r.record("granted")
        lag = rec["lag_s"]
        if r.job.quiet and lag is None and r.beside:
            self._diverge(t, "design", f"timed window {jid}: gpu-lease ran it beside {list(r.beside)}, which nobody "
                          f"preempts; the planner keeps a timed window alone, so it waits for them", job=jid)
        elif r.job.quiet and (lag is None or lag > WINDOW_LAG_S):
            self._diverge(t, "safety", f"timed window {jid}: the planner would start it "
                          + (f"{lag} s after gpu-lease" if lag is not None else f"later ({r.reason})"), job=jid)
        elif lag is None:
            self._diverge(t, "design", f"gpu-lease granted {jid}; the planner would wait: {r.reason}", job=jid)
        elif lag > WINDOW_LAG_S:
            self._diverge(t, "design", f"the planner would start {jid} {lag} s after gpu-lease", job=jid)
        elif lag < -EARLY_S:
            self._diverge(t, "design", f"the planner would start {jid} {-lag} s before gpu-lease", job=jid)
        if r.real_evicts and sorted(r.real_evicts) != sorted(r.shadow_evicts) and r.shadow is not None:
            self._diverge(t, "design", f"{jid}: gpu-lease stopped {sorted(r.real_evicts)}, the planner "
                          f"{list(r.shadow_evicts) or 'nothing'}", job=jid)
        self._emit("decisions.jsonl", rec)
        del self.requests[jid]
        self.stopped_for.pop(jid, None)

    def close(self, t: int, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        for r in list(self.requests.values()):
            self._emit("decisions.jsonl", r.record("open"))
        ledger.write_state(self.out / "state.json", self.state)
        out = {"schema": "cluster/shadow-summary/v1", "node": self.node, "ticks": self.ticks, "t_end": t,
               "ledger_seq": self.ledger.seq, **(summary or {})}
        (self.out / "summary.json").write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
        return out


def replay(cluster: Cluster, sampler: Iterable[dict[str, Any]], preempted: Iterable[str], out: Path) -> Shadow:
    """Today's node through the shadow: sampler lines in order, each with the `preempted.log` lines up to its time."""
    pre = sorted((p for p in map(nebius2.preemption, preempted) if p), key=lambda p: p["t"])
    sh = Shadow(cluster, out)
    k = 0
    last = None
    for rec in sampler:
        obs = nebius2.from_sampler(rec)
        batch = []
        while k < len(pre) and pre[k]["t"] <= obs.t:
            batch.append(pre[k])
            k += 1
        sh.step(obs, batch)
        last = obs.t
    if last is not None:
        sh.close(last, {"source": "replay"})
    return sh


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The timed windows in a ledger: each quiet allocation's pid, start and end, and its quiet report."""
    allocs, _ = ledger.history(records)
    started = {r["alloc"]["id"]: r["alloc"]["started"] for r in records if r["kind"] == "start"}
    out = []
    for h in allocs.values():
        if not h.job.get("quiet"):
            continue
        t0, t1 = min(started[h.alloc], h.intervals[0][1]), h.intervals[-1][2]
        rep = ledger.quiet_report(records, h.alloc)
        out.append({"alloc": h.alloc, "pid": int(h.alloc.split("@")[0][3:]), "t0": t0, "t1": t1, "gpus": len(h.gpus),
                    "quiet": rep["quiet"], "beside": sorted({o["alloc"] for o in rep["overlaps"] if o["state"] == "running"})})
    return sorted(out, key=lambda w: w["t0"])


def quiet_agreement(wins: list[dict[str, Any]], quiet_log: list[dict[str, Any]], tolerance_s: int) -> list[str]:
    """Where node_ops' quiet log (`quiet.jsonl`: `pause` naming the window's holder pid, `resume` "window over") and the
    ledger's windows disagree: a pause outside its holder's window, or a window's resume more than `tolerance_s` from its
    end."""
    out = []
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for w in wins:
        by_pid.setdefault(w["pid"], []).append(w)
    paused: dict[str, dict[str, Any]] = {}
    for e in quiet_log:
        t = epoch(e["t"])
        if e.get("event") == "pause":
            w = next((w for w in by_pid.get(int(e.get("holder", -1)), [])
                      if w["t0"] - tolerance_s <= t <= (w["t1"] or t) + tolerance_s), None)
            if w is None:
                out.append(f"node_ops paused pgid {e.get('pgid')} at {e['t']} for holder {e.get('holder')}, "
                           "outside any window the ledger has")
            else:
                paused[str(e.get("pgid"))] = w
        elif e.get("event") == "resume" and e.get("why") == "window over":
            w = paused.pop(str(e.get("pgid")), None)
            if w is not None and w["t1"] is not None and abs(t - w["t1"]) > tolerance_s:
                out.append(f"node_ops resumed pgid {e.get('pgid')} at {e['t']}, {t - w['t1']} s from the end of {w['alloc']}")
    return out


def sampler_windows(cluster: Cluster, rows: Iterable[dict[str, Any]], node: str = nebius2.NODE) -> dict[int, int]:
    """The timed windows the node's sampler saw: each window holder's pid, with the first time it was seen."""
    out: dict[int, int] = {}
    for rec in rows:
        obs = nebius2.from_sampler(rec)
        if nebius2.timed_now(cluster, obs, node):
            for a in nebius2.allocations(cluster, obs, node):
                if a.job.quiet:
                    out.setdefault(int(a.job.id[3:]), obs.t)
    return out


def evaluate(out: Path, quiet_log: list[dict[str, Any]] | None = None, *, tolerance_s: int = 30,
             since: int | None = None, until: int | None = None,
             sampled: dict[int, int] | None = None) -> dict[str, Any]:
    """The shadow's pass bar over its output directory: the ledger verifies; no safety divergence (a plan that breaks a
    rule, a timed window the planner would start over 5 s after `gpu-lease`, a request left waiting over 10 minutes beside
    free GPUs); every window agrees with node_ops' quiet log; the agent stayed under 1% of a core, made no NVML call and
    raised nothing. The evidence (hours covered, windows seen) is reported against the 3 h / 6-window guide, not
    required. Design divergences are listed for review. `sampled` (`sampler_windows` of the node's own log) lists the
    windows the agent never saw hold their GPUs (it learnt of them from `status.txt` alone), which it could not time."""
    out = Path(out)
    records = ledger.read(out / "ledger.jsonl") if (out / "ledger.jsonl").exists() else []
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    div = _jsonl(out / "divergences.jsonl")
    dec = _jsonl(out / "decisions.jsonl")
    wins = windows(records) if records else []
    t_lo = since if since is not None else (records[0]["t"] if records else 0)
    t_hi = until if until is not None else (records[-1]["t"] if records else 0)
    qlog = [e for e in (quiet_log or []) if t_lo <= epoch(e["t"]) <= t_hi]
    lags = [d["lag_s"] for d in dec if d["class"] == "timed" and d["outcome"] == "granted"]
    fp = summary.get("footprint") or {}
    checks = {
        "ledger verifies": not ledger.verify(records) and bool(records),
        "no safety divergence": not any(d["class"] == "safety" for d in div),
        "windows agree with node_ops' quiet log": not quiet_agreement(wins, qlog, tolerance_s),
        "under 1% of one core": fp.get("cpu_share") is None or fp["cpu_share"] < 0.01,
        "no NVML call": fp.get("nvml_calls", 0) == 0,
        "no exception": not summary.get("exception"),
    }
    by = {}
    for d in div:
        by.setdefault(d["class"], []).append(d)
    return {
        "schema": "cluster/shadow-evaluation/v1", "pass": all(checks.values()), "checks": checks,
        "evidence": {"hours": round((t_hi - t_lo) / 3600, 2), "windows": len(wins), "guide": "3 h and 6 windows",
                     "enough": (t_hi - t_lo) >= 3 * 3600 and len(wins) >= 6},
        "window_lag_s": {"max": max((x for x in lags if x is not None), default=None), "n": len(lags),
                         "not_started": sum(x is None for x in lags)},
        "requests": len(dec), "footprint": fp, "exception": summary.get("exception"),
        "safety": [d["msg"] for d in by.get("safety", [])],
        "design": [d["msg"] for d in by.get("design", [])],
        "observed": [d["msg"] for d in by.get("observed", [])],
        "quiet_disagreements": quiet_agreement(wins, qlog, tolerance_s),
        "windows_not_quiet": [{"alloc": w["alloc"], "beside": w["beside"]} for w in wins if not w["quiet"]],
        "windows_missed": sorted(f"gl-{pid} at {t}" for pid, t in (sampled or {}).items()
                                 if t_lo <= t <= t_hi and pid not in {w["pid"] for w in wins}),
    }


__all__ = ["Shadow", "Request", "replay", "evaluate", "windows", "sampler_windows", "quiet_agreement", "WINDOW_LAG_S", "STARVE_S"]
