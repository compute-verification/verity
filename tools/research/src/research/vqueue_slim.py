"""The run beside each candidate of the verified queue (`research.vqueue`) of what its graph2 check doesn't check: no graph/v2
task runs lean-agreement or the preflights.  It is drive's run beside the executor's check (`Driver._start_slim`):
`train.sh SHA MACHINES --deferred STEPS` of `SLIM_STEPS`, lean-agreement left out when a run of every step would skip it,
whose check runs the preflights before any step.  It is judged as `Driver._follow_slim` judges it, on drive's merge gate:
the preflights and lean-agreement count, policy.toml's `off_gate` (ratchet-gate) doesn't.  A candidate passes only once its
graph2 check's gating tasks and this run both passed (`combine`).

The loop calls `start` once per candidate and keeps the record it returns with the candidate.  `judge` reads that record each
tick and fills it in: the run ids once a detached launch printed them, and the verdict once the run ended or was given up,
which it returns from then on without reading the run again.  So the loop saves the record after each `judge`: a launch
read once is gone (`Driver._resolve` removes its directory).  The loop's Driver outlives its ticks, so `judge` reads the
runs' states afresh each call, never through `Driver._states`, which reads each run once a tick.  Nothing here writes
drive's state."""
from __future__ import annotations

from .drive import (ACTIVE, SIGTERM_RC, SLIM_JUDGED, SLIM_STEPS, Driver, implicated, killed, not_run, step_paths,
                    step_records)
from .vqueue import FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict

#: a candidate's state from its two verdicts' (`combine`): the first of these either has
ORDER = (FAIL, INFRA, PENDING, LIST_FAILED, PASS)


def start(driver: Driver, sha: str, base: str) -> dict | None:
    """Launch the run beside candidate ``sha``'s check, ``base`` the commit of `main` below it, as `Driver._start_slim` launches
    it beside the executor's check of ``sha`` with ``base`` verified: `SLIM_STEPS` but those a run of every step of ``sha``
    above ``base`` skips by name (lean-agreement, unless what it checks changed), with that run's keywords (ratchet-gate
    judges the merges above ``base``), on drive's machines (`Driver.on`).  The record to keep (`Driver._started`): ``ids``,
    none yet while a detached launch runs (its ``launch`` then); the candidate (``tip``), ``base``, when it launched
    (``at``), the ``steps`` it runs and those it skips (``skip``).  None, logged, when it registered no run, or when
    ``driver`` launches nothing (a dry run, or its queue's shadow), as `Driver._start_slim` does."""
    if not driver.on:
        raise ValueError("the run beside a candidate's check needs drive's machines: build the Driver with drive.toml's `on` "
                         "(`research.queue.drive_flags(q.workdir)[0].on`)")
    kw, skip, _, flags = driver._postland_args({"verified": base}, sha, False)
    skip = [s for s in skip if s in SLIM_STEPS]
    steps = [s for s in SLIM_STEPS if s not in skip]
    what = f"train.sh {sha[:12]} {driver.on} --deferred {','.join(steps)}{flags} beside candidate {sha[:12]}'s check"
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    rec, tail = driver._started(driver.launch_postland(driver.repo, sha, driver.on, ",".join(steps), **kw))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or 'launching'}")
    return {**rec, "tip": sha, "base": base, "at": driver.now(), "steps": steps, **({"skip": skip} if skip else {})}


def judge(driver: Driver, rec: dict | None) -> Verdict:
    """The verdict of the run `start` recorded in ``rec``, read as `Driver._follow_slim` reads it, on the steps it judges on
    the merge gate: `SLIM_JUDGED` but a step it skipped and policy.toml's `off_gate` at ``tip``.  Pending while it launches
    or runs.  Infra when it registered no run (``rec`` None, or a detached launch that printed no run ids), never ran or was
    stopped, or was lost or killed before it passed each of those steps; infra's one `Failed` says which.  Once it ended,
    pass when it passed each of them, fail with each of them it failed (a `Failed`: ``step NAME``, ``SHA:NAME``, the paths
    drive names that failure's culprit by, `step_paths`), and infra when it failed none but passed fewer (one skipped, or
    absent from its steps.jsonl).  Each verdict but pending is kept in ``rec`` (``verdict``) and read from there."""
    if rec is None:
        return _down(None, "registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    tip = rec["tip"]
    if (tail := driver._resolve(rec, f"the run of {', '.join(rec['steps'])} beside candidate {tip[:12]}'s check")) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    if "launch" in rec:
        return Verdict(PENDING)
    if not (ids := rec["ids"]):
        return _keep(rec, _down(rec, "registered no run"))
    states = driver.states(driver.repo, ids)
    if lost := driver._lost(rec, states):
        states = {r: ("lost", -1) for r in states}
    if any(st in ACTIVE for st, *_ in states.values()):
        return Verdict(PENDING)
    why = "lost" if lost else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values())
    if not lost and driver._never_ran(rec, states):
        return _keep(rec, _down(rec, f"never ran or was stopped ({why})"))
    off = driver._policy(tip).off_gate
    if not (missed := driver._unrun(ids, [s for s in SLIM_JUDGED if (s in rec["steps"] or s not in SLIM_STEPS)
                                          and s not in off], tip)):
        return _keep(rec, Verdict(PASS))
    dirs = [d for ds in driver._shards(rec) for d in ds]
    passed = driver._every_shard(rec, states, lambda rc: rc == 0) or (
        driver._every_shard(rec, states, lambda rc: rc in (0, 1)) and driver._only_allowed(rec))
    if not passed and (lost or any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                   for _, rc, *_ in states.values())
                       or any(implicated(d) is None and killed(d, why) for d in dirs)):
        return _keep(rec, _down(rec, "was lost" if lost else f"was killed ({why})"))
    paths: dict[str, set[str]] = {}
    for d in dirs:
        for st in step_records(d):
            if (s := st.get("step")) in missed and st.get("status") == "failed" and not not_run(st):
                paths.setdefault(s, set()).update(step_paths(d, st) or ())
    if paths:
        return _keep(rec, Verdict(FAIL, tuple(Failed(f"step {s}", f"{tip}:{s}", _inputs(paths[s])) for s in missed
                                              if s in paths)))
    return _keep(rec, _down(rec, f"ended ({why}) without passing {', '.join(missed)}: each skipped, or absent from its "
                                 "steps.jsonl"))


def combine(graph: Verdict, slim: Verdict) -> Verdict:
    """A candidate's verdict from its graph2 check's (``graph``: its gating tasks) and the run beside it's (``slim``,
    `judge`): fail when either failed; else infra when either ended in infra errors; else pending while either is; else
    list-failed when graph2's check failed to list; else pass, both passed.  It holds both verdicts' failures and infra
    errors."""
    if bad := sorted({graph.state, slim.state} - set(STATES)):
        raise ValueError(f"no verdict has state {', '.join(map(repr, bad))}: a state is one of {', '.join(STATES)}")
    state = next(s for s in ORDER if s in (graph.state, slim.state))
    return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)


def _inputs(paths: set[str]) -> frozenset[str]:
    """``paths`` as `vqueue.meets` reads them, meeting the files `drive.touches` matches them to: an absolute one (a traceback's
    frame) as each of its tails that has a directory, since a bare `conftest.py` would meet every conftest."""
    out: set[str] = set()
    for p in paths:
        parts = p.split("/")
        out.update(("/".join(parts[i:]) for i in range(1, len(parts) - 1)) if p.startswith("/") else (p,))
    return frozenset(out)


def _down(rec: dict | None, why: str) -> Verdict:
    """Infra: what became of the run ``rec`` records."""
    ids = " ".join(rec["ids"]) if rec and rec.get("ids") else ""
    return Verdict(INFRA, (), (Failed(f"the run beside its check{f' ({ids})' if ids else ''} {why}",
                                      f"{rec['tip'] if rec else ''}:{ids or 'no run'}", frozenset()),))


def _keep(rec: dict, v: Verdict) -> Verdict:
    rec["verdict"] = {"state": v.state, **{k: [{"task": f.task, "key": f.key, "inputs": sorted(f.inputs)} for f in fs]
                                           for k, fs in (("failed", v.failed), ("infra", v.infra)) if fs}}
    return v


def _kept(got: dict) -> Verdict:
    return Verdict(got["state"], *(tuple(Failed(f["task"], f["key"], frozenset(f["inputs"])) for f in got.get(k) or ())
                                   for k in ("failed", "infra")))
