"""The records run of a candidate of the verified queue (`research.vqueue`) with a member that changes Lean, a lock or a spec,
which runs before its check (`vqueue.Build.records`), and the records commit made of it: drive's lock run and lock commit,
run on the candidate's tree before its check rather than on `checked`'s after it lands.

The run is drive's lock run (`Driver.launch_locks`: `train.sh SHA MACHINES --deferred lean-audit`, its check at
`POSTLAND_SLOT_RANK`, its lean-audit regenerating each lock record it finds stale, with `--bookkeeping-warn` where train.sh
takes it), on drive's machines (`Driver.on`), judged as `Driver._follow_locks` judges drive's (`Driver._lock_outcome`): one
its machine failed (lost, killed, out of disk or memory) runs again once on the other machines (`Driver._resend_locks`); one
that registered no run, never ran or was stopped is infra, which the loop launches again once; one that failed refuses.  A
passed run passes once its Lean report (`merge.lean_report`) can be read: every package it regenerated is committed, and a
change to what a guarantee says (the report's `statement_changes`: a guarantee's statement, a definition one reads, a listed
premise or listed compile-time code) refuses nothing.  The records commit names each such change (`changes`), and the spec DM
shows Daniel each one once it lands on `main` (`merge.caught_up`, on the loop's `on_main`): that DM is the check.

A records commit is drive's lock commit (`Driver._put_locks`): the records the run regenerated (`Driver._locks`, from the
store, else from its run dir on its node), each at its path in the tree (`drive.lock_path`), committed on top of the candidate
with drive's message (`drive.lock_message`: `Lock records regenerated for #N...`, each change to what a guarantee says with
the owners of what it names, the bookkeeping the run repaired, `Checked:` the run of `queue/<sha12>` and `Regenerated:` the
files) as `queue.IDENTITY`, and vouched for by the merge gate
against the run (`Driver._unvouched`, `merge.regenerated_parent`), so post_train reads it as drive's and drive reverts it with
the PRs it names (`Driver._lock_commits`); the loop tells the bookkeeping the run repaired as drive tells it
(`drive.bookkeeping_told`: to `lean` and each owner a warning names).  A run that regenerated nothing, or only the records the
candidate holds, makes none, and the candidate is checked as built, as drive's lock run that regenerated nothing leaves
`checked`'s tip as it is.  It is made in a scratch index, without a checkout or a ref: drive's clone stays as drive keeps it.
Under a dry run or the queue's shadow nothing launches (`start`), as `vqueue_slim.start`.  The loop's Driver outlives its
ticks, so `judge` reads the runs' states afresh each call, never through `Driver._states`.  Nothing here writes drive's
state."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .drive import ACTIVE, CI, LEAN_STEP, Driver, bookkeeping_told, lock_message, lock_path
from .merge import AUDIT, bookkeeping_warnings, lean_report
from .queue import IDENTITY, QueueError, _git, _out
from .vqueue import FAIL, INFRA, PASS, PENDING, RecordsRun

#: what is told of the bookkeeping a records run repaired, and to whom (`drive.bookkeeping_told`)
Told = tuple[str, tuple[str, ...]]
#: a review line naming a guarantee or a listed premise (`check.py`'s `review` and `assumption_changes`), by the lock's
#: section that records its owner
NAMED = (("guarantees", re.compile(r"^guarantee (\S+): ")), ("assumptions", re.compile(r"^assumption (\S+): ")),
         ("domain", re.compile(r"^domain hypothesis (\S+): ")), ("open", re.compile(r"^open premise (\S+): ")))
#: a review line of a definition the build's guarantees read, which names them
READ_BY = re.compile(r"^definition .*, read by (.+): (?:new|gone|changed|in a changed module)$")


def start(driver: Driver, sha: str) -> dict | None:
    """Launch candidate ``sha``'s records run, as `Driver._start_locks` launches the lock run (`Driver.launch_locks`, on
    `Driver.on`).  The record to keep (`Driver._started`): ``ids``, none yet while a detached launch runs (its ``launch``
    then); the candidate (``tip``) and when it launched (``at``).  None, logged, when it registered no run, or when ``driver``
    launches nothing (a dry run, or its queue's shadow)."""
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
    return {**rec, "tip": sha, "at": driver.now()}


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
    """The verdict of records run ``rid``, passed: pass once its Lean report can be read, whatever its records change of what a
    guarantee says (the records commit names it: `_changes`); pending, with why, while it can't be."""
    try:
        report = lean_report(driver.q.store, rid, driver.fetch)
    except Exception as e:  # noqa: BLE001  (unread, it holds the commit: it is never taken as passed)
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read ({type(e).__name__})")
    if not isinstance(report, dict):
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read yet from the store or from its "
                                   "run dir on its node")
    return _keep(rec, RecordsRun(PASS), rid)


def _keep(rec: dict, r: RecordsRun, run: str | None = None) -> RecordsRun:
    """``r``, kept in ``rec`` (``verdict``), with ``run``, the run that passed."""
    rec["verdict"] = {"state": r.state, "why": r.why, **({"run": run} if run else {})}
    return r


def _kept(got: dict) -> RecordsRun:
    return RecordsRun(str(got.get("state") or INFRA), str(got.get("why") or ""))


def _changes(driver: Driver, rid: str, tip: str) -> list[str]:
    """What the records passed records run ``rid`` regenerated change of what a guarantee says: each line of the review of each
    package it regenerated whose report says its records do (`statement_changes`), its details aside, under the package and
    with the owners of what it names in ``tip``'s lock (`_owners`).  Raises `QueueError` when its Lean report can't be read."""
    report = lean_report(driver.q.store, rid, driver.fetch)
    if not isinstance(report, dict):
        raise QueueError(f"{rid}'s Lean report ({AUDIT}) can't be read from the store or from its run dir on its node")
    regen = report.get("regenerated") if isinstance(report.get("regenerated"), dict) else {}
    out = []
    for pkg, path in sorted(regen.items()):
        r = next((r for r in report.get("packages") or () if isinstance(r, dict)
                  and (r.get("package") == pkg or r.get("regenerated") == path)), None)
        if r is None or not r.get("statement_changes"):
            continue
        got = _git(driver.repo, "show", f"{tip}:{lock_path(pkg)}", check=False)
        try:
            lock = json.loads(got.stdout) if not got.returncode else {}
        except ValueError:
            lock = {}
        out += [f"{pkg}: {line} ({', '.join(_owners(lock, line))})" for line in r.get("review") or ()
                if isinstance(line, str) and line[:1] != " "]
    return out


def _owners(lock: dict, line: str) -> tuple[str, ...]:
    """The owners, in ``lock``, of each guarantee and premise review line ``line`` names; `CI` when none is found."""
    named = [(s, m[1]) for s, rx in NAMED if (m := rx.match(line))]
    if m := READ_BY.match(line):
        named += [("guarantees", g.strip()) for g in m[1].split(",")]
    out = []
    for section, name in named:
        entry = (lock.get(section) if isinstance(lock.get(section), dict) else {}).get(name)
        if isinstance(entry, dict) and isinstance(owner := entry.get("owner"), str) and owner.startswith("@"):
            out.append(owner)
    return tuple(dict.fromkeys(out)) or (CI,)


def commit(driver: Driver, sha: str, rid: str, prs: Iterable[int]) -> tuple[str, list[str], Told | None] | str | None:
    """Candidate ``sha``'s records commit of passed records run ``rid``, its members ``prs``: (the commit, its files, what
    drive tells of the bookkeeping the run repaired and to whom, `drive.bookkeeping_told`, None when it repaired none), the
    records the run regenerated committed on top of ``sha`` in drive's lock commit's form, naming what they change of what a
    guarantee says (`_changes`), and vouched for by the merge gate.  None when there is none to make: the run regenerated
    nothing (`Driver._locks`), or only the records ``sha`` holds.  Why not, instead, while its records or its Lean report can't
    be read, or the gate refuses the commit."""
    repo, into = driver.repo, f"queue/{sha[:12]}"
    scratch = Path(tempfile.mkdtemp(prefix="vqueue-records-"))
    try:
        locks, held = driver._locks(rid, scratch)
        if held:                            # the loop rings a held commit once a why: one naming this call's dir would ring each tick
            return held.replace(str(scratch), "<scratch>")
        if not locks:
            return None
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
        return None
    try:
        changed = _changes(driver, rid, sha)
    except Exception as e:  # noqa: BLE001  (a commit naming none of the changes its records make never lands)
        return (f"{rid}'s Lean report ({AUDIT}) can't be read ({type(e).__name__}), so a records commit couldn't name what "
                "its records change of what a guarantee says")
    prs = " ".join(f"#{n}" for n in prs) or f"`{into}` {sha[:12]}"
    warned = bookkeeping_warnings(driver.q.store, rid, driver.fetch)
    message = lock_message(prs, rid, into, sha, warned, files, changed)
    made = _git(repo, "commit-tree", tree, "-p", sha, "-m", message, env=IDENTITY).stdout.strip()
    if not made:
        raise QueueError(f"git commit-tree of {sha[:12]}'s records commit printed no commit")
    if why := driver._unvouched(rid, made):
        return why
    return made, files, bookkeeping_told(rid, into, sha, prs, made, warned) if warned else None
