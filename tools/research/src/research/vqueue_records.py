"""The records run of a candidate of the verified queue (`research.vqueue`) whose own failures are all stale Lean lock records
(graph2's lean-lock names the command that regenerates them: `vqueue.Failed.records`), and the records commit made of it:
drive's lock run and lock commit, run on the candidate's tree before it lands rather than on `checked`'s after.

The run is drive's lock run (`Driver.launch_locks`: `train.sh SHA MACHINES --deferred lean-audit`, its check at
`POSTLAND_SLOT_RANK`, its lean-audit regenerating each lock record it finds stale, with `--bookkeeping-warn` where train.sh
takes it), on drive's machines (`Driver.on`), judged as `Driver._follow_locks` judges drive's (`Driver._lock_outcome`): one
its machine failed (lost, killed, out of disk or memory) runs again once on the other machines (`Driver._resend_locks`); one
that registered no run, never ran or was stopped is infra, which the loop launches again once; one that failed refuses.  A
passed run's Lean report (`merge.lean_report`) says the rest: it refuses when it regenerated no record of a package graph2
found stale, or when a package it regenerated changes what a guarantee says (the report's `statement_changes`: a guarantee's
statement, a definition one reads, a listed premise or listed compile-time code), a claim change for its owners: each
guarantee's and premise's `owner` in the candidate's lock that the review names (`_owners`), else `CI`.

A records commit is drive's lock commit (`Driver._put_locks`): the records the run regenerated (`Driver._locks`, from the
store, else from its run dir on its node), each at its path in the tree (`drive.lock_path`), committed on top of the candidate
with drive's message (`drive.lock_message`: `Lock records regenerated for #N...`, the bookkeeping the run repaired,
`Checked:` the run of `queue/<sha12>` and `Regenerated:` the files) as `queue.IDENTITY`, and vouched for by the merge gate
against the run (`Driver._unvouched`, `merge.regenerated_parent`), so post_train reads it as drive's and drive reverts it with
the PRs it names (`Driver._lock_commits`).  It is made in a scratch index, without a checkout or a ref: drive's clone stays as
drive keeps it.  Under a dry run or the queue's shadow nothing launches (`start`), as `vqueue_slim.start`.  The loop's
Driver outlives its ticks, so `judge` reads the runs' states afresh each call, never through `Driver._states`.  Nothing here
writes drive's state."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .drive import ACTIVE, CI, LEAN_STEP, Driver, lock_message, lock_path
from .merge import AUDIT, bookkeeping_warnings, lean_report
from .queue import IDENTITY, QueueError, _git, _out
from .vqueue import FAIL, INFRA, PASS, PENDING, RecordsRun

#: the most review lines a refusal names; the run's Lean report has them all (review.txt in each package's reports)
SHOWN = 8
#: a review line naming a guarantee or a listed premise (`check.py`'s `review` and `assumption_changes`), by the lock's
#: section that records its owner
NAMED = (("guarantees", re.compile(r"^guarantee (\S+): ")), ("assumptions", re.compile(r"^assumption (\S+): ")),
         ("domain", re.compile(r"^domain hypothesis (\S+): ")), ("open", re.compile(r"^open premise (\S+): ")))
#: a review line of a definition the build's guarantees read, which names them
READ_BY = re.compile(r"^definition .*, read by (.+): (?:new|gone|changed|in a changed module)$")


def start(driver: Driver, sha: str, packages: Iterable[str]) -> dict | None:
    """Launch candidate ``sha``'s records run, ``packages`` those graph2 found stale, as `Driver._start_locks` launches the
    lock run (`Driver.launch_locks`, on `Driver.on`).  The record to keep (`Driver._started`): ``ids``, none yet while a
    detached launch runs (its ``launch`` then); the candidate (``tip``), when it launched (``at``) and ``packages``.  None,
    logged, when it registered no run, or when ``driver`` launches nothing (a dry run, or its queue's shadow)."""
    if not driver.on:
        raise ValueError("a candidate's records run needs drive's machines: build the Driver with drive.toml's `on` "
                         "(`research.queue.drive_flags(q.workdir)[0].on`)")
    what = f"train.sh {sha[:12]} {driver.on} --deferred {LEAN_STEP}, the records run of candidate {sha[:12]}"
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    rec, tail = driver._started(driver.launch_locks(driver.repo, sha, driver.on))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or 'launching'}")
    return {**rec, "tip": sha, "at": driver.now(), "packages": sorted(packages)}


def judge(driver: Driver, rec: dict | None) -> RecordsRun:
    """The records run `start` recorded in ``rec``.  Pending while it launches or runs; infra when it registered no run
    (``rec`` None, or a detached launch that printed no run ids), never ran or was stopped; launched again once elsewhere
    when its machine failed it, and pending then; fail when it failed (or its resend registered no run), or passed with
    records that can't be committed (`_passed`); else pass.  A passed run whose Lean report can't be read yet is pending,
    with why.  Every verdict but pending is kept in ``rec`` (``verdict``, with the run that passed) and read from there."""
    if rec is None:
        return RecordsRun(INFRA, "it registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    tip = rec["tip"]
    what = f"the records run of candidate {tip[:12]}"
    if (tail := driver._resolve(rec, what)) is not None:
        return _keep(rec, RecordsRun(FAIL if rec.get("resend") else INFRA, f"it registered no run: {tail}"))
    if "launch" in rec:
        return RecordsRun()
    if not (ids := rec["ids"]):
        return _keep(rec, RecordsRun(INFRA, "it registered no run"))
    states = driver.states(driver.repo, ids)
    lost = driver._lost(rec, states)
    if not lost and any(st in ACTIVE for st, *_ in states.values()):
        return RecordsRun()
    outcome, why = driver._lock_outcome(rec, states, lost)
    if outcome == "fault":
        where, again, tail = driver._resend_locks(rec, states)
        if again is None:
            return _keep(rec, RecordsRun(FAIL, f"it was {why}, and its resend registered no run: {tail}"))
        driver.q.log(f"research queue vqueue: {what} was {why}, a machine's fault: again, once, on {where}: "
                     f"{' '.join(rec['ids']) or 'launching'}")
        return RecordsRun()
    if outcome == "unrun":
        return _keep(rec, RecordsRun(INFRA, f"it never ran or was stopped ({' '.join(ids)})"))
    if outcome == "failed":
        return _keep(rec, RecordsRun(FAIL, f"it failed: {why}"))
    return _passed(driver, rec, why)


def _passed(driver: Driver, rec: dict, rid: str) -> RecordsRun:
    """The verdict of records run ``rid``, passed, on its Lean report: fail when it regenerated no record of a package of
    ``rec["packages"]``, or a package it regenerated changes what a guarantee says (`statement_changes`), naming the review's
    lines and the owners of what they name; else pass.  Pending, with why, while its report can't be read."""
    try:
        report = lean_report(driver.q.store, rid, driver.fetch)
    except Exception as e:  # noqa: BLE001  (unread, it holds the commit: it is never taken as passed)
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read ({type(e).__name__})")
    if not isinstance(report, dict):
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read yet from the store or from its "
                                   "run dir on its node")
    regen = report.get("regenerated") if isinstance(report.get("regenerated"), dict) else {}
    if missing := [p for p in rec.get("packages") or () if p not in regen]:
        return _keep(rec, RecordsRun(FAIL, f"{rid} passed without regenerating the records of {', '.join(missing)}, which "
                                           "graph2 found stale"), rid)
    reports = {pkg: r for pkg, path in regen.items() for r in report.get("packages") or ()
               if isinstance(r, dict) and (r.get("package") == pkg or r.get("regenerated") == path)}
    if changed := sorted(pkg for pkg, r in reports.items() if r.get("statement_changes")):
        lines = [x for pkg in changed for x in reports[pkg].get("review") or () if isinstance(x, str) and x[:1] != " "]
        more = f"; and {len(lines) - SHOWN} more in its Lean report" if len(lines) > SHOWN else ""
        why = (f"regenerating the records of {', '.join(changed)} ({rid}) changes what a guarantee says, which the change's "
               f"own records must say for its owners to review: {'; '.join(lines[:SHOWN])}{more}")
        return _keep(rec, RecordsRun(FAIL, why, _owners(driver.repo, rec["tip"], {pkg: reports[pkg] for pkg in changed})),
                     rid)
    return _keep(rec, RecordsRun(PASS), rid)


def _keep(rec: dict, r: RecordsRun, run: str | None = None) -> RecordsRun:
    """``r``, kept in ``rec`` (``verdict``), with ``run``, the run that passed."""
    rec["verdict"] = {"state": r.state, "why": r.why, "owners": list(r.owners), **({"run": run} if run else {})}
    return r


def _kept(got: dict) -> RecordsRun:
    return RecordsRun(str(got.get("state") or INFRA), str(got.get("why") or ""), tuple(got.get("owners") or ()))


def _owners(repo: Path, tip: str, reports: dict[str, dict]) -> tuple[str, ...]:
    """The owners, in ``tip``'s locks of each package of ``reports``, of every guarantee and premise its review names; `CI`
    when none is found."""
    out: list[str] = []
    for pkg, r in reports.items():
        got = _git(repo, "show", f"{tip}:{lock_path(pkg)}", check=False)
        try:
            lock = json.loads(got.stdout) if not got.returncode else {}
        except ValueError:
            lock = {}
        for line in r.get("review") or ():
            if not isinstance(line, str):
                continue
            named = [(s, m[1]) for s, rx in NAMED if (m := rx.match(line))]
            if m := READ_BY.match(line):
                named += [("guarantees", g.strip()) for g in m[1].split(",")]
            for section, name in named:
                entry = (lock.get(section) if isinstance(lock.get(section), dict) else {}).get(name)
                if isinstance(entry, dict) and isinstance(owner := entry.get("owner"), str) and owner.startswith("@"):
                    out.append(owner)
    return tuple(dict.fromkeys(out)) or (CI,)


def commit(driver: Driver, sha: str, rid: str, prs: Iterable[int]) -> tuple[str, list[str]] | str:
    """Candidate ``sha``'s records commit of passed records run ``rid``, its members ``prs``: (the commit, its files), the
    records the run regenerated committed on top of ``sha`` in drive's lock commit's form and vouched for by the merge gate;
    why not, instead, while its records can't be read (`Driver._locks`), it regenerated none, they are the records ``sha``
    holds, or the gate refuses the commit."""
    repo, into = driver.repo, f"queue/{sha[:12]}"
    scratch = Path(tempfile.mkdtemp(prefix="vqueue-records-"))
    try:
        locks, held = driver._locks(rid, scratch)
        if held:
            return held
        if not locks:
            return f"{rid}'s Lean report names no record it regenerated"
        index = {"GIT_INDEX_FILE": str(scratch / "index")}
        _git(repo, "read-tree", sha, env=index)
        files = []
        for pkg, src in sorted(locks.items()):
            dst = lock_path(pkg)
            mode = (_out(repo, "ls-tree", sha, "--", dst).split() or ["100644"])[0]
            _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{_out(repo, 'hash-object', '-w', '--', str(src))},{dst}",
                 env=index)
            files.append(dst)
        tree = _git(repo, "write-tree", env=index).stdout.strip()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if tree == _out(repo, "rev-parse", f"{sha}^{{tree}}"):
        return f"the records {rid} regenerated are those {sha[:12]} holds"
    prs = " ".join(f"#{n}" for n in prs) or f"`{into}` {sha[:12]}"
    message = lock_message(prs, rid, into, sha, bookkeeping_warnings(driver.q.store, rid, driver.fetch), files)
    made = _git(repo, "commit-tree", tree, "-p", sha, "-m", message, env=IDENTITY).stdout.strip()
    if not made:
        raise QueueError(f"git commit-tree of {sha[:12]}'s records commit printed no commit")
    if why := driver._unvouched(rid, made):
        return why
    return made, files
