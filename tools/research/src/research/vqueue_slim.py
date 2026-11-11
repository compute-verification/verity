"""The run beside each candidate of the verified queue (`research.vqueue`) of what its graph2 check doesn't check: no graph/v2
task runs lean-agreement or the preflights.  It is `train.sh SHA MACHINES --deferred STEPS` of `SLIM_STEPS`, lean-agreement
left out when a run of every step would skip it (`Landing.beside_args`), whose check runs the preflights before any step.  It
is judged on the merge gate: the preflights and lean-agreement count, policy.toml's `off_gate` (ratchet-gate) doesn't.  A candidate passes only once its
graph2 check's gating tasks and this run both passed (`combine`).

What the loop does.  It starts the run with each candidate's build, beside each candidate's check, so up to its `--depth`
of them at once take the queue's machines, or once for a candidate on `main`'s chain whose start its log doesn't hold; until
then the candidate's run verdict is pending (`Verdict(PENDING)`, not `judge` of None, which is infra), and a candidate dropped
while its run runs (pending, or a check that gates failed before it ended) cancels it (one still launching has no run ids to
cancel yet).  It calls `start` once a candidate, `judge` each tick, and saves the record after
every `judge`: `judge` fills it in (the run ids once a detached launch printed them, the verdict once the run ended or was
given up, which it returns from then on without reading the run again), and a launch read once is gone (`Landing._resolve`
removes its directory).  On
`vqueue.Retry` with this run's verdict infra, it calls `start` again and keeps the new record in place of the old.  It gives
the `Landing` the `on` of drive.toml at its `main` (`MAIN`): `start` refuses a `Landing` without machines.  Under `--dry-run` it
calls neither `start` nor `judge`.  The candidate's verdict for `vqueue.decide` is `combine` of graph2's and this one's.  The
loop's `Landing` outlives its ticks, so `judge` reads the runs' states afresh each call.

A failed step's `Failed.key` is its content key at the candidate (`key`): SHA-512 over the step's name and the files it
reads (`reads`) as git holds them in the candidate's tree; its `base_key` is the same at the base, computed from the base's
tree, with no run.  An equal key means the base fails the step too, as graph2's keys of its tasks by their inputs do, so
`vqueue.decide` takes the failure as inherited.  A step whose reads can't be bounded is keyed on the whole tree, so its
failure is never inherited by its key: preflight-lints (its lints read every file, `check.lints_key`) and ratchet-gate (it
judges the merges above the base, not a tree).  Each failed step's one test (`Failed.tests`) is the step's (`step_test`), so
the run beside the base's own check, which the loop keeps beside `main` (started on `main` against itself, so lean-agreement
never runs), says what the base fails (`based`): failed there too, a step's failure is inherited by `vqueue.decide`'s rule
of failing tests.

The check that gates (`research queue vqueue --gate check`).  `start` with ``shards`` launches a whole check of the
candidate in place of the run of `SLIM_STEPS`: every step of `AFTER_LANDING` as that many shards, one launch a shard on its
own machines (`shard_machines`), with the same keywords (`Landing.beside_args`).  ``shards`` `auto:N` (`AUTO_SHARDS`) is N
shards of an auto plan of the candidate's tree, which splits pytest by suite where ci.toml's plans can't split it: `start`
has `check.py --shard-plan --deferred STEPS --n N` make it in the background (`Landing.plan_shards`), and the first `judge`
that finds it made (`Landing.planned`) launches every shard `--shard auto:K/N` with its file, which is removed once each
shard has sent it with its run; pending until then, infra when it failed or took past `PLAN_WAIT_S`.  A candidate dropped
while its plan is made launches nothing (`abandon`).  Its record says so (``gate``), and `judge` then judges every step
`main` waits for at the candidate's base (`Landing._gating` there, the
preflights and a lean-agreement that ran, less policy.toml's `off_gate` and drive.toml's `warn_steps`), a step a shard holds
passed only once every shard holding it passed (`shard_passed`).  A failed step's tests are what its failure names
(`failing`), so `vqueue.decide` takes as inherited only a failure whose every failing test the run beside the base fails
too; what a failure doesn't name is one test no other key has (`unnamed`), so it is never inherited.  It fails as soon as a
shard has failed a step it judges, or a suite of one, while the others run (`_early`: a shard's failures so far, which its
check writes as each one fails), a failure read then as it would be at the end; a candidate on another candidate whose check
runs is compared with what that check has failed so far, the rest unknown until it ends (`based` with ``so_far``).  A step
it leaves out
still runs: `failures` reads what it failed, which the loop rings to who acts on it, but what the run beside the base's
check failed too (`Loop._off_gate`)."""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import json
import os
import re
import shutil
import signal
from dataclasses import replace
from pathlib import Path

from .landing import (ACTIVE, AFTER_LANDING, AUTO_PLAN, KILLED_RC, LAUNCH_LOST_S, LEAN_AUDIT_RE, PLAN_WAIT_S, PREFLIGHT,
                      SIGTERM_RC, SLIM_JUDGED, SLIM_STEPS, SUITE_STEPS, Landing, _epoch, _step_log, implicated, killed, not_run,
                      shard_machines, shard_passed, shard_steps, step_paths, step_records, suite_paths, suite_rows)
from .queue import _SUMMARY_RE, CHECK_STEPS, _git
from .reds import named
from .store.fixtures import REPLICA_MAP, entries
from .vqueue import FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict, meets

#: a candidate's state from its two verdicts' (`combine`): the first of these either has
ORDER = (FAIL, INFRA, PENDING, LIST_FAILED, PASS)
#: the train.sh a candidate's check launches with: the loop's own, deployed with its pin, so a candidate's tree doesn't decide
#: how its launch is waited for
OWN_TRAIN = Path(__file__).resolve().parents[4] / "tools" / "verity" / "check" / "train.sh"
#: a line of the loop's deselect list (`deselection`) that is a test's node id, its file's path from the repository's root
DESELECT_RE = re.compile(r"(?!/|\.\./)[\w./-]+\.py::\S+")
#: what a launch that registered no run says when it points at the machines, not the candidate's tree (`Failed.launch`):
#: train.sh stopped its launcher, still running, at the end of its wait; no run ids within `LAUNCH_LOST_S`; the pod lost its
#: process.  A launcher that exited without a run ran the candidate's own check.py, and says none of these
LAUNCH_ONLY = ("no run launched in the wait", "min after its launch", "its process is gone with no exit written")
#: what declares lean-agreement's reads: the graph build family, whose lean-agreement task reads `VERIFIER`'s files and
#: `CHECK_READS` (check.py, the modules it loads as it starts, and what they read), and the lean-build of `LEAN`, read from its
#: source at the commit keyed (`_declared`), never run
BUILD = "tools/verity/graph/verity_graph/families/build.py"
#: the environment every step runs in: the lock `uv run --locked` installs, and the interpreter it pins
ENV = ("uv.lock", ".python-version")
#: what `uv lock --check` (preflight-lock) reads: the lock and every file the workspace's metadata can come from
LOCK_READS = ("uv.lock", "uv.toml", ".python-version", "pyproject.toml", "**/pyproject.toml", "setup.py", "**/setup.py",
              "setup.cfg", "**/setup.cfg")
#: the replica map preflight-fixtures reads (`research.store.fixtures.load_map`), as a repository path
MAP = REPLICA_MAP.as_posix()
#: the code preflight-fixtures runs beside check.py: `check.fetchable_fixtures` imports `research.store` and what it imports
FIXTURES_CODE = "tools/research/src/research"
#: the tag a content key starts with (`key`)
KEY_TAG = "verity-vqueue-slim-step/v1"
#: a record's ``gate`` when its run is the check that gates (`start` with ``shards``)
CHECK = "check"
#: `start`'s ``shards`` naming N shards of an auto plan of the candidate's tree
AUTO_SHARDS = re.compile(r"auto:([1-9]\d*)")
#: lean-audit's own failures of no package, in its log
AUDIT_FAIL_RE = re.compile(r"^lean-audit: FAIL: .*$", re.M)
#: what the audit prints under a package's `AUDIT .../PKG: FAIL` that is no failure: counts and timings of its run
AUDIT_SAYS = ("closure:", "staged:", "trust gap:", "heartbeats:")
#: the audit's scratch tree, which each run names anew
SCRATCH_RE = re.compile(r"\S*?/lean-audit-scratch-[^/\s]+/")


def shard_count(shards: int | str) -> tuple[int, bool]:
    """How many shards `start`'s ``shards`` is, and whether of an auto plan: an int N, ci.toml's plan with N shards (1 or less:
    one whole run), or `auto:N` (`AUTO_SHARDS`).  ValueError for anything else."""
    if isinstance(shards, int):
        return shards, False
    if m := AUTO_SHARDS.fullmatch(str(shards)):
        return int(m[1]), True
    raise ValueError(f"shards {shards!r}: N, ci.toml's plan with N shards, or auto:N, N shards of an auto plan")


def deselection(driver: Landing, listed: Path | None) -> Path | None:
    """The file a candidate's check leaves out the tests of (`train.sh --deselect`), from the loop's list ``listed``, read at
    each launch: its node ids (`DESELECT_RE`; blank and `#` lines skipped), written beside it as ``<name>.sent``, each other
    line logged and left out, so a typo there drops no test and reds no candidate.  None while it is absent or names none."""
    if listed is None or not listed.is_file():
        return None
    lines = [s for line in listed.read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]
    ids = list(dict.fromkeys(s for s in lines if DESELECT_RE.fullmatch(s)))
    if bad := [s for s in lines if not DESELECT_RE.fullmatch(s)]:
        driver.q.log(f"research queue vqueue: {listed}: {len(bad)} line(s) not a test's node id (PATH.py::NAME, PATH from the "
                     f"repository's root), left out: {'; '.join(bad[:5])}")
    if not ids:
        return None
    sent = listed.with_name(f"{listed.name}.sent")
    part = sent.with_name(f"{sent.name}.{os.getpid()}")
    part.write_text("".join(f"{i}\n" for i in ids))
    os.replace(part, sent)
    return sent


def start(driver: Landing, sha: str, base: str, shards: int | str = 0, deselect: Path | None = None) -> dict | None:
    """Launch the run beside candidate ``sha``'s check, ``base`` the commit of `main` below it: `SLIM_STEPS` but those a run
    of every step of ``sha`` above ``base`` skips by name (lean-agreement, unless what it checks changed), with that run's
    keywords (`Landing.beside_args`: ratchet-gate judges the merges above ``base``), on the queue's machines (`Landing.on`).
    With ``shards``, the check that gates instead: `--deferred` every step of `AFTER_LANDING` (each check runs the
    preflights first), `--shard K/N` on shard K's machines (`shard_machines`) for each of that many, one whole run for 1;
    `auto:N` has the auto plan its shards launch by made first (`Landing.plan_shards`, ``plan``, which `judge` follows).
    The record to keep (`Landing._started`): ``ids``, none yet while a detached launch runs (its ``launch`` then, or each
    shard's in ``shards``, which `judge` makes ``groups``, each shard's ids); the candidate (``tip``), ``base``, when it
    launched (``at``), the ``steps`` it runs and those it skips (``skip``), and ``gate`` (`CHECK`) with ``shards``.  None,
    logged, when it registered no run (a shard that registered none stops the others), or when ``driver`` launches nothing (a
    dry run, or its queue's shadow).  Every launch leaves out the tests the loop's list ``deselect`` names (`deselection`), each
    shard's alike; the nightly uncached check, which reads no list, runs them."""
    if not driver.on:
        raise ValueError("the run beside a candidate's check needs the queue's machines: build the Landing with drive.toml's `on` "
                         "(`research.queue.flags_at(repo, main).on`)")
    n, auto = shard_count(shards)
    kw, skip, flags = driver.beside_args(sha, base)
    if left := deselection(driver, deselect):
        kw = {**kw, "deselect": str(left)}
        flags += f" --deselect {left} ({len(left.read_text().splitlines())} tests)"
    every = AFTER_LANDING if n else SLIM_STEPS
    skip = [s for s in skip if s in every]
    steps = [s for s in every if s not in skip]
    sent = ",".join(every if n else steps)
    what = (f"train.sh {sha[:12]} {driver.on} --deferred {sent}{flags}"
            + (f" as an auto plan's {n} shards" if auto else f" as {n} shards" if n > 1 else "")
            + (f" as candidate {sha[:12]}'s check" if n else f" beside candidate {sha[:12]}'s check"))
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    if auto:
        rec, tail = {"ids": [], "plan": {**driver.plan_shards(driver.repo, sha, sent, n=n), "n": n, "deferred": sent,
                                         "kw": kw}}, ""
    elif n > 1:
        rec, tail = _launch(driver, sha, sent, kw, n)
        if rec is not None and not any("launch" in r for r in rec["shards"]):
            _grouped(rec)
    else:
        rec, tail = driver._started(driver.launch_postland(driver.repo, sha, driver.on, sent, **kw, train_from=OWN_TRAIN))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or ('planning' if auto else 'launching')}")
    return {**rec, "tip": sha, "base": base, "at": driver.now(), "steps": steps, **({"skip": skip} if skip else {}),
            **({"gate": CHECK} if n else {})}


def _launch(driver: Landing, sha: str, sent: str, kw: dict, n: int, plan: str | None = None) -> tuple[dict | None, str]:
    """Each of ``n`` shards of candidate ``sha``'s check launched on its machines (`shard_machines`): `--shard K/N` of ci.toml's
    plan, or of the auto plan whose file is ``plan``, `--shard auto:K/N` with it.  The record of every shard's launch
    (``shards``); None with the tail of one that registered no run, the others' runs cancelled."""
    got = [driver._started(driver.launch_postland(driver.repo, sha, on, sent, **kw, shard=f"{'auto:' * bool(plan)}{k}/{n}",
                                                  **({"shard_plan": plan} if plan else {}), train_from=OWN_TRAIN))
           for k, on in enumerate(shard_machines(driver.on, n), 1)]
    if any(r is None for r, _ in got):
        if ids := [i for r, _ in got if r for i in r["ids"]]:
            driver.cancel(driver.repo, ids, f"vqueue: a shard of candidate {sha[:12]}'s check registered no run")
        return None, next(t for r, t in got if r is None)
    return {"ids": [], "shards": [r for r, _ in got]}, ""


def _grouped(rec: dict) -> dict:
    """``rec`` once each of its ``shards`` has its run ids: they are its ``groups``, and its ``ids`` every one's; each shard sent
    its auto plan's file with its run (`_unplan`)."""
    rec["groups"] = [list(r["ids"]) for r in rec.pop("shards")]
    rec["ids"] = [i for g in rec["groups"] for i in g]
    _unplan(rec)
    return rec


def _unplan(rec: dict) -> None:
    """Remove the directory of ``rec``'s auto plan (`start` with `auto:N`), which no shard of it sends any more."""
    if plan := rec.get("plan"):
        shutil.rmtree(plan["out"], ignore_errors=True)


def abandon(rec: dict) -> None:
    """Give up ``rec``'s auto plan before its shards launch (its candidate dropped, or the plan failed): its job's directories
    removed, so it writes no plan (`check.py --shard-plan` makes no directory) and no shard launches by it; a job still
    running ends on its own, its worktree with it."""
    if (plan := rec.get("plan")) and "made" not in plan:
        shutil.rmtree(plan["dir"], ignore_errors=True)
        _unplan(rec)


def _planned(driver: Landing, rec: dict, what: str) -> Verdict:
    """The auto plan ``rec``'s shards launch by (`start` with `auto:N`), followed: pending while `check.py --shard-plan` makes
    it, up to `PLAN_WAIT_S` after the start, then stopped; once made, every shard launched `--shard auto:K/N` with its file
    (`_launch`, ``plan``'s ``made``, ``at`` their launch), and pending.  Infra, kept, when it failed, has another N than asked,
    wasn't made in time, or a shard registered no run, its directories removed (`abandon`)."""
    plan = rec["plan"]
    doc, why = driver.planned(plan)
    if why is not None:
        why = f"failed: {why}"
    elif doc is None:
        if driver.now() - rec["at"] <= PLAN_WAIT_S:
            return Verdict(PENDING)
        with contextlib.suppress(ProcessLookupError, PermissionError, ValueError, TypeError, KeyError):
            os.killpg(int(plan["pid"]), signal.SIGTERM)
        why = f"was not made in {PLAN_WAIT_S // 60} min (stopped)"
    elif doc["n"] != plan["n"]:
        why = f"has {doc['n']} shards, not {plan['n']}"
    got, tail = (None, f"its auto plan {why}") if why is not None else _launch(
        driver, rec["tip"], plan["deferred"], plan["kw"], plan["n"], str(Path(plan["out"]) / AUTO_PLAN))
    if got is None:
        abandon(rec)
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    plan["made"] = {"digest": doc["digest"], "predicted_s": doc.get("predicted_s")}
    rec.update(got, at=driver.now())
    if not any("launch" in r for r in rec["shards"]):
        _grouped(rec)
    driver.q.log(f"research queue vqueue: {what}: auto plan {doc['digest'][:12]}, predicted wall "
                 f"{(doc.get('predicted_s') or 0) / 60:.0f} min: {' '.join(rec['ids']) or 'launching'}")
    return Verdict(PENDING)


def judge(driver: Landing, rec: dict | None) -> Verdict:
    """The verdict of the run `start` recorded in ``rec``, read on the steps it judges on the merge gate: `SLIM_JUDGED` but a
    step it skipped and policy.toml's `off_gate` at ``tip``; of the check that gates (``gate``), every step `main` waits for
    at ``base`` (`Landing._gating`), the preflights and a lean-agreement it ran, but policy.toml's `off_gate` there and
    drive.toml's `warn_steps` (`_gate`).  Pending while its auto plan is made, the call that finds it made launching the
    shards (`_planned`), and while it launches or runs, each shard's launch read as the run's own is; the check that gates,
    while it runs, fails once a shard has failed one of those steps (`_early`).
    Infra when it registered no run (``rec`` None, its auto plan failed, or a detached launch that printed no run ids,
    a shard's stopping the other shards; `Failed.launch` when it says one of `LAUNCH_ONLY`), never ran or was stopped, or was lost or killed before it passed each of those
    steps; infra's one `Failed` says which.  Once it ended, pass when it passed each of them (the check that gates: in
    every shard that holds it, `shard_passed`), a step that failed only on what an unexpired allowance of policy.toml at
    ``base`` covers counting as passed, so no candidate's allowance covers its own failure; fail with each of them it failed
    (a `Failed`: ``step NAME``, the step's content key at the candidate, the paths that failure names its culprit by,
    `step_paths`, its content key at ``base``, `key`, and the step's one test, `step_test`; the check that gates: the tests
    and paths its failure names, `failing`), and infra when it failed none but passed fewer (one skipped, or absent from its
    steps.jsonl).  Each verdict but pending is kept in ``rec`` (``verdict``, with the steps it passed) and read from there."""
    if rec is None:
        return _down(None, "registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    tip, check = rec["tip"], rec.get("gate") == CHECK
    what = f"the run of {', '.join(rec['steps'])} " + (f"as candidate {tip[:12]}'s check" if check else
                                                      f"beside candidate {tip[:12]}'s check")
    if "plan" in rec and "made" not in rec["plan"]:
        return _planned(driver, rec, what)
    shards = rec.get("shards") or ()
    # every shard's launch is read in one call, since a launch read once is gone: the record never holds some read, some not
    printed = [driver.launched(r["launch"])[0] for r in shards if "launch" in r]
    if None in printed and [] not in printed and driver.now() - rec["at"] <= LAUNCH_LOST_S:
        return Verdict(PENDING)
    for k, shard in enumerate(shards, 1):
        shard.setdefault("at", rec["at"])
        if (tail := driver._resolve(shard, f"shard {k}/{len(shards)} of {what}")) is not None:
            if got := [i for r in shards for i in r["ids"]]:
                driver.cancel(driver.repo, got, f"vqueue: shard {k}/{len(shards)} of candidate {tip[:12]}'s check registered "
                                                "no run")
            _unplan(rec)
            return _keep(rec, _down(rec, f"registered no run: shard {k}/{len(shards)}: {tail}"))
    if any("launch" in r for r in shards):
        return Verdict(PENDING)
    if shards:
        _grouped(rec)
    if (tail := driver._resolve(rec, what)) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    if "launch" in rec:
        return Verdict(PENDING)
    if not (ids := rec["ids"]):
        return _keep(rec, _down(rec, "registered no run"))
    states = driver.states(driver.repo, ids)
    if lost := driver._lost(rec, states):
        states = {r: ("lost", -1) for r in states}
    if any(st in ACTIVE for st, *_ in states.values()):
        return _early(driver, rec, states) if check else Verdict(PENDING)
    why = "lost" if lost else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values())
    if not lost and driver._never_ran(rec, states):
        return _keep(rec, _down(rec, f"never ran or was stopped ({why})"))
    base = rec.get("base")
    allow = base or tip
    if check:
        judged = _gate(driver, rec)
        view = shard_steps(driver._shards(rec), [any(states[r][1] == 0 for r in g) for g in rec.get("groups") or [ids]])
        passes = shard_passed(view, driver._allowed_steps(allow, [d for ds in driver._shards(rec) for d in ds]))
        missed = [s for s in judged if s not in passes]
    else:
        off = driver._policy(tip).off_gate
        judged = [s for s in SLIM_JUDGED if (s in rec["steps"] or s not in SLIM_STEPS) and s not in off]
        missed = driver._unrun(ids, judged, allow)
    if not missed:
        return _keep(rec, Verdict(PASS), judged)
    dirs = [d for ds in driver._shards(rec) for d in ds]
    ok = driver._every_shard(rec, states, lambda rc: rc == 0) or (
        driver._every_shard(rec, states, lambda rc: rc in (0, 1)) and driver._only_allowed(rec, allow))
    if not ok and (lost or any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                   for _, rc, *_ in states.values())
                       or any(implicated(d) is None and killed(d, why) for d in dirs)):
        return _keep(rec, _down(rec, "was lost" if lost else f"was killed ({why})"))
    paths: dict[str, set[str]] = {}
    tests: dict[str, set[str]] = {}
    keys: dict[str, str] = {}
    for d in dirs:
        for st in step_records(d):
            if (s := st.get("step")) in missed and st.get("status") == "failed" and not not_run(st):
                if s not in keys:
                    keys[s] = key(driver.repo, tip, s) or f"{tip}:{s}"
                at, named = failing(driver, allow, d, st, unnamed(keys[s])) if check else (step_paths(d, st), {step_test(s)})
                paths.setdefault(s, set()).update(at or ())
                tests.setdefault(s, set()).update(named)
    passed = [s for s in judged if s not in missed]
    if paths:
        return _keep(rec, Verdict(FAIL, tuple(Failed(f"step {s}", keys[s], _inputs(paths[s]),
                                                     key(driver.repo, base, s) if base else None, frozenset(tests[s]))
                                              for s in missed if s in paths), ended=True), passed)
    return _keep(rec, _down(rec, f"ended ({why}) without passing {', '.join(missed)}: each skipped, or absent from its "
                                 "steps.jsonl"), passed)


def _gate(driver: Landing, rec: dict) -> list[str]:
    """The steps the check that gates (``rec``) is judged on: every step `main` waits for at its base (`Landing._gating`), the
    preflights and a lean-agreement it ran, but policy.toml's `off_gate` there and drive.toml's `warn_steps`."""
    at = rec.get("base") or rec["tip"]
    gate = {*SLIM_JUDGED, *driver._gating(at)} - {*driver._policy(at).off_gate, *driver.warn_steps}
    return [s for s in CHECK_STEPS if s in gate and (s in rec["steps"] or s in PREFLIGHT)]


def _early(driver: Landing, rec: dict, states: dict) -> Verdict:
    """The verdict of the check that gates (``rec``) while a shard of it runs: pending until one of its runs says it failed
    (one running names its failures so far, `landing.failing`; one ended exited 1), then fail with each step it is judged on
    (`_gate`) that such a run failed, not by a signal, or a suite of which it failed a named test, uncovered by an allowance
    at its base, read as once every shard ended (`failing`), from their dirs fetched afresh (`landing.evidence`: the Landing's
    own cache holds what the end's reading reads).  A suite failed beyond its tests alone, as a stop marks each suite it
    cuts short, waits for the end.  Never kept, so not ``ended``: `vqueue.decide` reads a fail whose every failure is
    inherited as pending until the check ends.  ``rec`` holds it (``early``: its failures as `_keep` keeps a verdict's, with
    ``at``, when the first was written: a failed step's end, a failed suite's log's last write), which the check above reads
    (`based`)."""
    said = [r for r, s in states.items() if ((len(s) > 3 and s[3]) if s[0] in ACTIVE else s[1] == 1)]
    if not said:
        return Verdict(PENDING)
    tip, base = rec["tip"], rec.get("base")
    allow, dirs = base or tip, driver.evidence(driver.repo, said)
    allowed = driver._allowed_steps(allow, dirs)
    keys: dict[str, str] = {}
    paths: dict[str, set[str]] = {}
    tests: dict[str, set[str]] = {}
    at: list[float] = []
    for d in dirs:
        if implicated(d) is None and killed(d):
            continue                                     # a kill, as `judge` reads it at the end: infra, never an early red
        ended = {st.get("step"): st for st in step_records(d)}
        for s in _gate(driver, rec):
            if (st := ended.get(s)) is not None:
                if (st.get("status") != "failed" or not_run(st) or s in allowed
                        or isinstance(rc := st.get("rc"), int) and (rc < 0 or rc in KILLED_RC)):
                    continue
                at.append(_epoch(st.get("ended_utc")) or driver.now())
            elif s in SUITE_STEPS and (rows := [row for name, row in suite_rows(d).items() if row["status"] == "failed"
                                                 and row["step"] == s and row.get("failed_tests")
                                                 and not driver._allowed(allow, f"suite {name}", d, row).covered]):
                st = {"step": s}
                at += [_written(d / "suites" / Path(row["log"]).name, driver.now()) for row in rows if row.get("log")]
            else:
                continue
            if s not in keys:
                keys[s] = key(driver.repo, tip, s) or f"{tip}:{s}"
            got, named = failing(driver, allow, d, st, unnamed(keys[s]))
            paths.setdefault(s, set()).update(got or ())
            tests.setdefault(s, set()).update(named)
    if not tests:
        return Verdict(PENDING)
    v = Verdict(FAIL, tuple(Failed(f"step {s}", keys[s], _inputs(paths[s]), key(driver.repo, base, s) if base else None,
                                   frozenset(tests[s])) for s in _gate(driver, rec) if s in tests))
    first = [t for t in (*at, (rec.get("early") or {}).get("at")) if t is not None]
    rec["early"] = {**_doc(v), "at": min(first, default=driver.now())}
    return v


def _written(path: Path, otherwise: float) -> float:
    """When ``path`` was last written (a fetched file keeps its node's time), ``otherwise`` when it wasn't fetched."""
    return path.stat().st_mtime if path.is_file() else otherwise


def unnamed(key: str) -> str:
    """The one test of what a failure keyed ``key`` doesn't name: no failure under another key has it, so
    `vqueue.decide` never takes it as inherited by its tests."""
    return f"unnamed:{key}"


def failures(driver: Landing, rec: dict | None, step: str) -> dict[str, str] | None:
    """What the run ``rec`` records failed of ``step``, a step `judge` leaves out (policy.toml's `off_gate`, drive.toml's
    `warn_steps`), once its verdict is kept: each failure the step's logs name (`reds.named`: ratchet-gate's merges,
    circuit-check's failures), by its key with its line, else the step itself; {} when it passed.  A shard's twin that passed
    it answers for the shard (`shard_steps`).  None while the run runs, and when it didn't run the step to its end: no shard
    holds it, one skipped or stopped it, or a kill ended it (`KILLED_RC`, a signal)."""
    if rec is None or "verdict" not in rec or not rec.get("ids"):
        return None
    groups = driver._shards(rec)
    view = shard_steps(groups, [True] * len(groups)) or []
    if not (held := [k for k, v in enumerate(view) if step in v]) or any(view[k][step] not in ("passed", "failed") for k in held):
        return None
    out: dict[str, str] = {}
    for d in (d for k in held if view[k][step] == "failed" for d in groups[k]):
        for st in step_records(d):
            if st.get("step") != step or st.get("status") != "failed" or not_run(st):
                continue
            if st.get("rc") in KILLED_RC or isinstance(st.get("rc"), int) and st["rc"] < 0:
                return None
            out.update(named(step, _step_log(d, st)) or {step: f"{step} failed, its log naming no failure"})
    return out


def failing(driver: Landing, at: str, d: Path, st: dict, rest: str) -> tuple[set[str], set[str]]:
    """What failed step ``st`` of the check that gates names in its run dir ``d``: the paths its culprit
    is named by, and its failing tests.  A step that runs suites (`SUITE_STEPS`): of each suite of its that failed
    (`suite_rows`) uncovered by ``at``'s allowances (its base's), `suite_paths` and `SUITE::TEST` of each test it failed,
    and ``rest`` under the suite's name when it names none or failed for more than its tests (its row's `beyond_tests` isn't
    false).  Any other: `step_paths`, the test id of each `FAILED` and `ERROR` line of its log's short summary (all of it
    before ` - `, a parametrized id's spaces included), and of lean-audit each line under a failed package's
    `AUDIT .../PKG: FAIL` (`audit_failures`).  ``rest`` (`unnamed`) alone when it names no test."""
    step = st.get("step")
    if step not in SUITE_STEPS:
        log = _step_log(d, st)
        tests = {m[1] for m in _SUMMARY_RE.finditer(log)} | (audit_failures(log, rest) if step == "lean-audit" else set())
        return step_paths(d, st) or set(), tests or {rest}
    paths: set[str] = set()
    tests = set()
    for name, row in suite_rows(d).items():
        if row["status"] != "failed" or row["step"] != step or driver._allowed(at, f"suite {name}", d, row).covered:
            continue
        paths |= suite_paths(d, row) or set()
        tests |= {f"{name}::{t}" for t in row.get("failed_tests") or ()}
        if not row.get("failed_tests") or row.get("beyond_tests") is not False:
            tests.add(f"{name}::{rest}")
    return paths, tests or {rest}


def audit_failures(log: str, rest: str) -> set[str]:
    """The failures lean-audit's ``log`` names: `PKG::LINE` of each line the audit prints under a failed package's
    `AUDIT .../PKG: FAIL` (`LEAN_AUDIT_RE`) but its run's counts and timings (`AUDIT_SAYS`, with the lines under each), the
    scratch tree's name left out, ``rest`` under the name of a package it prints none under; and each `lean-audit: FAIL:`
    line."""
    out: dict[str, set[str]] = {}
    pkg, says = None, False
    for line in log.splitlines():
        if line.startswith("AUDIT"):
            pkg = m[1] if (m := LEAN_AUDIT_RE.match(line)) else None
            if pkg:
                out.setdefault(pkg, set())
        elif pkg and line.startswith(" ") and (text := line.strip()):
            if re.match(r"  \S", line):
                says = text.startswith(AUDIT_SAYS)
            if not says:
                out[pkg].add(SCRATCH_RE.sub("", text))
    return {f"{p}::{t}" for p, ts in out.items() for t in ts or {rest}} | set(AUDIT_FAIL_RE.findall(log))


def combine(graph: Verdict, slim: Verdict) -> Verdict:
    """A candidate's verdict from its graph2 check's (``graph``: its gating tasks) and the run beside it's (``slim``,
    `judge`): fail when either failed; else infra when either ended in infra errors; else pending while either is; else
    list-failed when graph2's check failed to list; else pass, both passed.  It holds both verdicts' failures and infra
    errors."""
    if bad := sorted({graph.state, slim.state} - set(STATES)):
        raise ValueError(f"no verdict has state {', '.join(map(repr, bad))}: a state is one of {', '.join(STATES)}")
    state = next(s for s in ORDER if s in (graph.state, slim.state))
    return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)


#: the steps whose reads can't be bounded, each keyed on the whole tree, and why
WHOLE = {"preflight-lints": "its lints read every file (`check.lints_key`)",
         "ratchet-gate": "it judges the merges above the base, not a tree"}


def reads(repo: Path, commit: str, step: str) -> frozenset[str] | str:
    """What ``step`` reads of ``commit``'s tree, as `vqueue.meets` reads paths, directories and globs; for a step read whole,
    why: one whose reads can't be bounded (`WHOLE`, or a step this doesn't know), or whose declaration ``commit`` doesn't hold
    as plain data.  Each runs under check.py, so each bounded one holds check's reads (`CHECK_READS` at ``commit``) and `ENV`:

    - lean-agreement: the graph build family's lean-agreement task's inputs, `VERIFIER`'s files beside check's, and the
      Lean package `LEAN` whose lean-build it reads, whole (its lock and nested packages too, which `package` cuts);
    - preflight-lock: `LOCK_READS`;
    - preflight-fixtures: the replica map (`MAP`), each path it registers (`research.store.fixtures.entries`), and the code
      its check imports (`FIXTURES_CODE`)."""
    if step in WHOLE:
        return WHOLE[step]
    if step not in ("lean-agreement", "preflight-lock", "preflight-fixtures"):
        return f"nothing declares what {step} reads"
    if isinstance(got := _declared(repo, commit), str):
        return got
    check = (*got["CHECK_READS"], *ENV)
    if step == "lean-agreement":
        return frozenset((got["VERIFIER"], got["LEAN"], *check))
    if step == "preflight-lock":
        return frozenset((*LOCK_READS, *check))
    if isinstance(doc := _map(repo, commit), str):
        return doc
    return frozenset((MAP, *entries(doc).values(), FIXTURES_CODE, *check))


def key(repo: Path, commit: str, step: str) -> str | None:
    """``step``'s content key at ``commit``: SHA-512 over `KEY_TAG`, the step's name and each file of ``commit``'s tree it
    reads (`reads`), as git holds it (mode, object and path), in path order; for a step read whole, over the tag, its name and
    the commit's tree.  Two commits with one key give the step the same files to read.  None when git can't read ``commit``."""
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False)
    if r.returncode:
        return None
    tree = r.stdout.strip()
    h = hashlib.sha512(f"{KEY_TAG}\0{step}\0".encode())
    if isinstance(want := reads(repo, tree, step), str):
        h.update(f"tree\0{tree}".encode())
        return h.hexdigest()
    for path, entry in _files(repo, tree):
        if meets((path,), want):
            h.update(f"{entry}\t{path}\0".encode(errors="surrogateescape"))
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def _files(repo: Path, tree: str) -> tuple[tuple[str, str], ...]:
    """(path, "MODE TYPE OBJECT") of each entry under ``tree`` (an object id, which never changes, so a listing is kept), in
    git's order, which is path order."""
    out = _git(repo, "ls-tree", "-r", "-z", "--full-tree", tree).stdout
    return tuple((path, entry) for entry, _, path in (line.partition("\t") for line in out.split("\0") if line))


#: the names `_declared` reads of `BUILD`
DECLARED = ("CHECK_READS", "VERIFIER", "LEAN")


def _declared(repo: Path, commit: str) -> dict | str:
    """`BUILD`'s `DECLARED` at ``commit``, read from its source without running it (`_data`): `VERIFIER` and `LEAN` strings,
    `CHECK_READS` a tuple of them; else why not."""
    r = _git(repo, "show", f"{commit}:{BUILD}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {BUILD}"
    try:
        body = ast.parse(r.stdout).body
    except SyntaxError as e:
        return f"{BUILD} at {commit[:12]} doesn't parse: {e}"
    got = {node.targets[0].id: _data(node.value, {}) for node in body
           if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in DECLARED}
    check = got.get("CHECK_READS")
    if (isinstance(got.get("VERIFIER"), str) and isinstance(got.get("LEAN"), str) and isinstance(check, tuple)
            and all(isinstance(p, str) for p in check)):
        return got
    return f"{BUILD} at {commit[:12]} doesn't declare {', '.join(DECLARED)} as plain data"


def _map(repo: Path, commit: str) -> dict | str:
    """The replica map at ``commit``, else why not."""
    r = _git(repo, "show", f"{commit}:{MAP}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {MAP}"
    try:
        doc = json.loads(r.stdout)
    except ValueError as e:
        return f"{MAP} at {commit[:12]} isn't JSON: {e}"
    return doc if isinstance(doc, dict) else f"{MAP} at {commit[:12]} isn't a JSON object"


def _data(node: ast.AST, names: dict[str, str]) -> str | tuple | None:
    """The value of ``node``, an expression of string literals, f-strings of the names a comprehension binds (``names``),
    tuples and lists with starred items, and comprehensions over those: what `BUILD` writes its declarations in.  None for
    anything else, which is never run."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [_data(v.value if isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None else v,
                       names) for v in node.values]
        return "".join(parts) if all(isinstance(p, str) for p in parts) else None
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list = []
        for e in node.elts:
            got = _data(e.value if isinstance(e, ast.Starred) else e, names)
            if not isinstance(e, ast.Starred):
                out.append(got)
            elif isinstance(got, tuple):
                out.extend(got)
            else:
                return None
        return None if None in out else tuple(out)
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)) and len(node.generators) == 1:
        g = node.generators[0]
        if isinstance(g.target, ast.Name) and not g.ifs and not g.is_async and isinstance(over := _data(g.iter, names), tuple):
            items = tuple(_data(node.elt, {**names, g.target.id: x}) for x in over)
            return None if None in items else items
    return None


def _inputs(paths: set[str]) -> frozenset[str]:
    """``paths`` as `vqueue.meets` reads them, meeting the files they name: an absolute one (a traceback's
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
                                      f"{rec['tip'] if rec else ''}:{ids or 'no run'}", frozenset(),
                                      launch=why.startswith("registered no run") and any(m in why for m in LAUNCH_ONLY)),))


def _keep(rec: dict, v: Verdict, passed: list[str] | None = None) -> Verdict:
    rec["verdict"] = {**_doc(v), **({"passed": passed} if passed is not None else {})}
    return v


def _doc(v: Verdict) -> dict:
    return {"state": v.state, **{k: [{"task": f.task, "key": f.key, "inputs": sorted(f.inputs),
                                      **({"base_key": f.base_key} if f.base_key is not None else {}),
                                      **({"tests": sorted(f.tests)} if f.tests else {}),
                                      **({"launch": True} if f.launch else {})} for f in fs]
                                 for k, fs in (("failed", v.failed), ("infra", v.infra)) if fs}}


def _kept(got: dict) -> Verdict:
    return Verdict(got["state"], *(tuple(Failed(f["task"], f["key"], frozenset(f["inputs"]), f.get("base_key"),
                                                frozenset(f.get("tests") or ()), launch=bool(f.get("launch")))
                                         for f in got.get(k) or ()) for k in ("failed", "infra")), ended=got["state"] == FAIL)


def step_test(step: str) -> str:
    """The one test id of ``step`` of the run beside a check: its failures are compared by it (`vqueue.Failed.tests`), as a
    step read whole (`WHOLE`) is never at its base's key."""
    return f"step:{step}"


def based(v: Verdict, base: dict | None, so_far: bool = False) -> Verdict:
    """``v`` with each failed step's `base_tests` read from the run beside the base's check (``base``, that run's record, as
    `judge` keeps it): the tests that run failed of it (the step's, `step_test`, where it kept none), none when it passed it,
    and None while it runs, or when it ended without judging it, so `vqueue.decide` takes a failure the base has too as
    inherited.  With ``so_far`` (a base that is itself a candidate), one whose check hasn't ended answers with what it has
    failed so far (its ``early``, `_early`) where that holds every test failed of the step, and None otherwise, since it
    may yet fail the rest: a candidate above another is failed on its own failures once its base's word on them is known."""
    got = (base or {}).get("verdict")
    if early := so_far and base is not None and not isinstance(got, dict):
        got = base.get("early") or {}
    if not isinstance(got, dict) or not v.failed:
        return v
    failed = {f.get("task"): f.get("tests") for f in got.get("failed") or () if isinstance(f, dict)}
    passed = set(got.get("passed") or ())

    def of(f: Failed) -> Failed:
        s = f.task.removeprefix("step ")
        if s == f.task:
            return f
        tests = frozenset(failed[f.task] or {step_test(s)}) if f.task in failed else frozenset() if s in passed else None
        return replace(f, base_tests=None if early and tests is not None and not f.tests <= tests else tests)

    return replace(v, failed=tuple(of(f) for f in v.failed))
