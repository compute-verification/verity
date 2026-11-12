"""The records run of a candidate of the verified queue (`research.vqueue`) with a member that changes Lean, a lock or a spec,
which runs before its check (`vqueue.Build.records`), and the records commit made of it.

The run is a lock run (`Landing.launch_locks`: `train.sh SHA MACHINES --deferred lean-audit`, its check at
`POSTLAND_SLOT_RANK`, its lean-audit regenerating each lock record it finds stale, with `--bookkeeping-warn` where train.sh
takes it), on the queue's machines (`Landing.on`), judged by `lock_outcome`: one its machine failed (lost, killed, out of disk
or memory: `machine_fault`) runs again once on the other machines (`resend_locks`); one that registered no run, never ran or was
stopped is infra, which the loop launches again once; one that failed refuses.  A passed run passes once its Lean report
(`merge.lean_report`) can be read: every package it regenerated is committed, and a change to what a guarantee says (the
report's `statement_changes`: a guarantee's statement, a definition one reads, a listed premise or listed compile-time code)
refuses nothing.  The records commit names each such change (`_changes`), and the spec DM shows each one once it lands on
`main` (`merge.caught_up`, on the loop's `on_main`): that DM is the check.

A records commit is a lock commit: the records the run regenerated (`regenerated_locks`, from the store, else from its run dir
on its node), each at its path in the tree (`lock_path`), committed on top of the candidate with a lock commit's message
(`lock_message`: `Lock records regenerated for #N...`, each change to what a guarantee says with the owners of what it names,
the bookkeeping the run repaired, `Checked:` the run of `queue/<sha12>` and `Regenerated:` the files) as the queue
(`Queue.identity`: the author its repository's rules name, else `queue.IDENTITY`), and
vouched for by the merge gate against the run (`unvouched`, `merge.regenerated_parent`), so post_train reads it as a lock
commit, and neither reads who made it; the loop tells the bookkeeping the run repaired (`bookkeeping_told`: to `lean` and
each owner a warning names).  A run
that regenerated nothing, or only the records the candidate holds, makes none, and the candidate is checked as built.  It is
made in a scratch index, without a checkout or a ref.  The candidate is a staged batch, off main's chain; when the chain's top
it goes onto is another base, its members are merged again there, and the records commit, made and vouched for on the staged
tree, is carried onto that merge, the same files byte for byte, only when the two trees' Lean inputs are the same
(`same_inputs`, `vqueue_slim.lean_key`), which are all the records depend on.  Under a dry run or the queue's shadow nothing launches (`start`), as
`vqueue_slim.start`.  The loop's `Landing` outlives its ticks, so `judge` reads the runs' states afresh each call."""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .landing import (ACTIVE, CI, LEAN_STEP, NOT_STARTED, PROGRESS, SIGTERM_RC, Landing, _step_log, failed_steps, killed,
                      step_records)
from .merge import AUDIT, bookkeeping_warnings, lean_report
from .queue import IDENTITY, QueueError, _git, _out
from .vqueue import FAIL, INFRA, PASS, PENDING, RecordsRun

#: what is told of the bookkeeping a records run repaired, and to whom (`bookkeeping_told`)
Told = tuple[str, tuple[str, ...]]
#: a review line naming a guarantee or a listed premise (`check.py`'s `review` and `assumption_changes`), by the lock's
#: section that records its owner
NAMED = (("guarantees", re.compile(r"^guarantee (\S+): ")), ("assumptions", re.compile(r"^assumption (\S+): ")),
         ("domain", re.compile(r"^domain hypothesis (\S+): ")), ("open", re.compile(r"^open premise (\S+): ")))
#: a review line of a definition the build's guarantees read, which names them
READ_BY = re.compile(r"^definition .*, read by (.+): (?:new|gone|changed|in a changed module)$")
#: a failed step's log saying its machine ran out of disk or memory: the machine's fault, not the change's (`machine_fault`)
FAULT_RE = re.compile(r"No space left on device|\bENOSPC\b|\[Errno 28\]|Cannot allocate memory|\[Errno 12\]|\bMemoryError\b"
                      r"|\b[Oo]ut of memory\b")
#: a Lake package's lock, which a lock run may regenerate (`merge.LOCK`)
LOCK = "lean-audit.json"
#: the subject a lock commit opens with (`lock_message`), before the PRs whose records it regenerated
LOCK_SUBJECT = "Lock records regenerated for "


def lock_path(pkg: str) -> str:
    """Where a record lock run's Lean report names under ``pkg`` (`merge.regenerated_records`) goes in the tree: the package's
    lock, or a file of `merge.REPAIRED` at its own path."""
    from .merge import REPAIRED

    return pkg if pkg in REPAIRED else f"{pkg}/{LOCK}" if pkg not in ("", ".") else LOCK


def lock_checked(rid: str, into: str, tip: str) -> str:
    """A lock commit's `Checked:` line, its parenthesis left open (`lock_message` closes it; a merge onto a moved branch says
    so first): lock run ``rid`` of ``into``'s ``tip``, whose parent the merge gate vouches for the commit against."""
    return f"Checked: {rid} (the lock run of `{into}` {tip[:12]}, passed by regenerating them"


def lock_message(prs: str, rid: str, into: str, tip: str, warned: list[str], files: list[str],
                 changed: Iterable[str] = (), onto: str | None = None) -> str:
    """A lock commit's message (`commit`), which the merge gate (`merge.regenerated_parent`), the loop (`LOCK_SUBJECT`) and
    post_train read: `Lock records regenerated for PRS`, what its records change of what a
    guarantee says (``changed``, which the spec DM shows once they land), the bookkeeping the lock run repaired (``warned``,
    each a warning for the next PR to fix), with ``onto`` the tree they were carried onto from ``tip``, its `Checked:` line
    (`lock_checked`) and `Regenerated: FILES`."""
    said = ("What these records change of what a guarantee says, which the spec DM shows once they land:\n" +
            "".join(f"  {c}\n" for c in changed) + "\n") if (changed := list(changed)) else ""
    repairs = ("Bookkeeping the lock run repaired, each a warning for the next PR to fix:\n" +
               "".join(f"  {w}\n" for w in warned) + "\n") if warned else ""
    carried = (f"Carried from {tip[:12]} onto {onto[:12]}, the same members merged onto another base, whose Lean inputs are "
               f"{tip[:12]}'s (`vqueue_slim.lean_key`): all these records depend on.\n\n") if onto else ""
    return (f"{LOCK_SUBJECT}{prs}\n\n{said}{repairs}{carried}{lock_checked(rid, into, tip)})\n"
            f"Regenerated: {', '.join(files)}\n")


#: how many bookkeeping warnings of a lock run its post lists (`bookkeeping_told`); the lock commit lists them all
BOOKKEEPING_SHOWN = 12


def bookkeeping_told(rid: str, into: str, tip: str, prs: str, lock: str, warned: list[str]) -> tuple[str, tuple[str, ...]]:
    """What is told of the bookkeeping lock run ``rid`` of ``into``'s ``tip`` repaired in lock commit ``lock`` (``warned``,
    `merge.bookkeeping_warnings`: each a warning for the next PR to fix), and to whom, as handles with no `@`: the Lean check's
    owner, `lean`, and each owner a warning names."""
    owners = sorted({h for w in warned for h in re.findall(r"\(@([\w-]+)\)", w)} - {"lean"})
    shown = "; ".join(warned[:BOOKKEEPING_SHOWN]) + (f"; and {len(warned) - BOOKKEEPING_SHOWN} more in the commit"
                                                    if len(warned) > BOOKKEEPING_SHOWN else "")
    return (f"the lock run {rid} of `{into}` {tip[:12]} repaired bookkeeping that {prs} brought, in {lock[:12]}, each a warning "
            f"for the next PR to fix: {shown}", ("lean", *owners))


def start(driver: Landing, sha: str) -> dict | None:
    """Launch candidate ``sha``'s records run (`Landing.launch_locks`, on `Landing.on`).  The record to keep
    (`Landing._started`): ``ids``, none yet while a detached launch runs (its ``launch`` then); the candidate (``tip``) and when it launched (``at``).  None, logged, when it registered no run, or when ``driver``
    launches nothing (a dry run, or its queue's shadow)."""
    if not driver.on:
        raise ValueError("a candidate's records run needs the queue's machines: build the Landing with drive.toml's `on` "
                         "(`research.queue.flags_at(repo, main).on`)")
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


def judge(driver: Landing, rec: dict | None) -> RecordsRun:
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
    outcome, why = lock_outcome(driver, rec, states, lost)
    if outcome == "fault":
        where, again, tail = resend_locks(driver, rec, states)
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


def _passed(driver: Landing, rec: dict, rid: str) -> RecordsRun:
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


def _changes(driver: Landing, rid: str, tip: str) -> list[str]:
    """What the records passed records run ``rid`` regenerated change of what a guarantee says: each line of the review of each
    package it regenerated whose report says its records do (`statement_changes`), its details aside, under the package and
    with the owners of what it names in ``tip``'s lock (`_owners`), which a package new at ``tip`` hasn't.  Raises when its Lean
    report, or a lock of ``tip`` it names, can't be read."""
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
        lock = json.loads(got.stdout) if not got.returncode else {}
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


def same_inputs(repo: Path, a: str, b: str) -> bool:
    """Whether commits ``a`` and ``b`` have the same Lean inputs (`vqueue_slim.lean_key`), so the same lock records."""
    from .vqueue_slim import lean_key
    return (key := lean_key(repo, a)) is not None and key == lean_key(repo, b)


def commit(driver: Landing, sha: str, rid: str, prs: Iterable[int],
           onto: str | None = None) -> tuple[str, list[str], Told | None] | str | None:
    """Staged batch ``sha``'s records commit of passed records run ``rid``, its members ``prs``: (the commit, its files, what
    is told of the bookkeeping the run repaired and to whom, `bookkeeping_told`, None when it repaired none), the records the
    run regenerated committed on top of ``sha`` as a lock commit (`lock_message`), naming what they change of what a guarantee
    says (`_changes`), and vouched for by the merge gate.  None when there is none to make: the run regenerated nothing
    (`regenerated_locks`), or only the records ``sha`` holds.  Why not, instead, while its records or its Lean report can't
    be read, or the gate refuses the commit.

    With ``onto``, ``prs`` merged again onto another base, that commit is made and vouched for as above, then carried onto
    ``onto``: the same files, byte for byte, committed on top of it, its message saying so.  Records depend on a tree's Lean
    inputs alone, so this raises unless ``onto``'s are ``sha``'s (`same_inputs`); None means ``onto`` is checked as built."""
    repo, into = driver.repo, f"queue/{sha[:12]}"
    if onto is not None and not same_inputs(repo, sha, onto):
        raise QueueError(f"{onto[:12]}'s Lean inputs aren't {sha[:12]}'s: the records {rid} regenerated on {sha[:12]} aren't "
                         "its records")
    scratch = Path(tempfile.mkdtemp(prefix="vqueue-records-"))
    try:
        locks, held = regenerated_locks(driver, rid, scratch)
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
        return (f"{rid}'s Lean report ({AUDIT}) or a lock it names can't be read ({type(e).__name__}), so a records commit "
                "couldn't name what its records change of what a guarantee says")
    prs = " ".join(f"#{n}" for n in prs) or f"`{into}` {sha[:12]}"
    warned = bookkeeping_warnings(driver.q.store, rid, driver.fetch)
    message, identity = lock_message(prs, rid, into, sha, warned, files, changed), driver.q.identity
    made = _git(repo, "commit-tree", tree, "-p", sha, "-m", message, env=identity).stdout.strip()
    if not made:
        raise QueueError(f"git commit-tree of {sha[:12]}'s records commit printed no commit")
    if why := unvouched(driver, rid, made):
        return why
    if onto is not None:
        made = _carry(repo, made, onto, files, lock_message(prs, rid, into, sha, warned, files, changed, onto), identity)
    return made, files, bookkeeping_told(rid, into, sha, prs, made, warned) if warned else None


def _carry(repo: Path, made: str, onto: str, files: list[str], message: str, identity: dict[str, str] = IDENTITY) -> str:
    """Records commit ``made``'s ``files``, as it holds them, committed on top of ``onto`` with ``message`` as ``identity``."""
    scratch = Path(tempfile.mkdtemp(prefix="vqueue-records-"))
    try:
        index = {"GIT_INDEX_FILE": str(scratch / "index")}
        _git(repo, "read-tree", onto, env=index)
        for f in files:
            mode, _, blob = _out(repo, "ls-tree", made, "--", f).partition("\t")[0].split()
            _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{f}", env=index)
        tree = _git(repo, "write-tree", env=index).stdout.strip()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    carried = _git(repo, "commit-tree", tree, "-p", onto, "-m", message, env=identity).stdout.strip()
    if not carried:
        raise QueueError(f"git commit-tree of {made[:12]}'s records carried onto {onto[:12]} printed no commit")
    return carried


def lock_outcome(driver: Landing, rec: dict, states: dict, lost: bool) -> tuple[str, str]:
    """How lock run ``rec`` came out once it ended or was ``lost`` (`Landing._lost`), its runs' ``states`` read: ("fault", what)
    when its machine failed it (lost, killed, out of disk or memory: `machine_fault`), and it runs again once elsewhere
    (`resend_locks`), never a resend nor in a dry run or the shadow; ("passed", the run that exited 0 with lean-audit
    passed in its steps.jsonl); ("unrun", "") when every twin exited 75 or was stopped; else ("failed", why)."""
    rcs = [rc for _, rc, *_ in states.values()]
    fault = None
    if rec.get("resend") or driver.dry_run or driver.q.shadow:
        pass
    elif lost:
        fault = "lost"
    elif 0 in rcs or all(rc in (NOT_STARTED, SIGTERM_RC) for rc in rcs):
        pass
    elif any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC)) for rc in rcs):
        fault = "killed"
    else:
        fault = machine_fault(driver, rec["ids"])
    if fault:
        return "fault", fault
    if lost:
        return "failed", f"{' '.join(rec['ids'])} lost on its resend"
    if ran := [rid for rid, (_, rc, *_) in states.items() if rc == 0]:
        if not driver._unrun(ran, [LEAN_STEP]):
            return "passed", ran[0]
        return "failed", f"{' '.join(ran)} exited 0 without passing {LEAN_STEP}"
    if all(rc in (NOT_STARTED, SIGTERM_RC) for rc in rcs):
        return "unrun", ""
    return "failed", ", ".join(f"{rid} {st} rc={rc}" for rid, (st, rc, *_) in states.items())


def resend_locks(driver: Landing, rec: dict, states: dict) -> tuple[str, dict | None, str]:
    """Launch lock run ``rec`` of ``rec["tip"]`` again, once, after its machine failed it (`lock_outcome`), on the machines
    of `on` its runs (``states``) didn't run on (`Landing._elsewhere`): (where, the new launch's fields, which ``rec`` then holds
    as a resend; None, with train.sh's output's tail, when it registered no run)."""
    where = driver._elsewhere(states)
    again, tail = driver._started(driver.launch_locks(driver.repo, rec["tip"], where))
    if again is not None:
        rec.update(**again, at=driver.now(), resend=True)
    return where, again, tail


def regenerated_locks(driver: Landing, rid: str, scratch: Path) -> tuple[dict[str, Path], str | None]:
    """({package: its regenerated lock, fetched under ``scratch`` or from its node}, why its records can't be committed
    instead) of passed lock run ``rid``; ({}, None) when it regenerated none.  Read from the run's files in the store, else
    from its run dir on its node (`merge.live_records`).  Held, never committed without them, when its Lean report names
    records neither holds, or can't be read at all unless its lean-audit step was skipped (`lean_audit_ran`)."""
    from .merge import AUDIT, live_records, regenerated_records

    try:
        recs = regenerated_records(driver.q.store, rid, scratch / rid)
        if recs is None:
            recs = live_records(rid, driver.fetch)
    except Exception as e:  # noqa: BLE001  (a guard never fails open: `merge.Unreadable`, or the store failing)
        return {}, f"its regenerated lock records can't be read: {type(e).__name__}: {e}"
    if recs is None and lean_audit_ran(driver, rid):
        return {}, (f"its {AUDIT}, which says whether it regenerated lock records, can't be read from the store or from its "
                    "run dir on its node")
    return {pkg: path for pkg, (_, path) in (recs or {}).items() if path is not None}, None


def unvouched(driver: Landing, rid: str, commit: str = "HEAD") -> str | None:
    """Why the merge gate refuses lock commit ``commit`` (`merge.regenerated_parent`, reading the records itself: from the
    store, else from ``rid``'s run dir on its node), or None when it vouches for it."""
    from .merge import declared_command, produced, regenerated_parent

    if regenerated_parent(driver.repo, commit, driver.q.store, tool="check", kinds=produced(driver.repo, "check"),
                          command=declared_command(driver.repo, "check"), deferred=rid, fetch=driver.fetch):
        return None
    return (f"the merge gate refuses its lock commit: its records aren't byte for byte the ones {rid} regenerated "
            "(`merge.regenerated_parent`, on the live run)")


def lean_audit_ran(driver: Landing, rid: str) -> bool:
    """Whether run ``rid``'s lean-audit step may have regenerated records: true unless its result in the store, else its
    steps.jsonl (`evidence`), says the step was skipped or never ran; a run neither can be read of may have."""
    from .merge import step_status

    try:
        stored = bool(((driver.q.store.get_attempt(rid) or {}).get("outputs") or {}).get("result"))
    except Exception:  # noqa: BLE001  (unknown: it may have)
        return True
    if stored:
        return step_status(driver.q.store, rid, "lean-audit") not in ("skipped", "absent")
    for d in driver.evidence(driver.repo, [rid]):
        with contextlib.suppress(OSError):
            for line in (d / PROGRESS[0]).read_text().splitlines():
                with contextlib.suppress(ValueError):
                    if (rec := json.loads(line)).get("step") == "lean-audit":
                        return rec.get("status") != "skipped"
    return True


def machine_fault(driver: Landing, ids: list[str]) -> str | None:
    """What says a lock run that failed failed by its machine rather than by the Lean it checked, or None: a step killed
    (`killed`), a failed step's log saying its machine ran out of disk or memory (`FAULT_RE`), or lean-audit passed in a run
    that failed anyway (a disk that fills after every audit passed is no failure of the change)."""
    for d in driver.evidence(driver.repo, ids):
        if d.name not in ids:
            continue
        if any(st.get("step") == LEAN_STEP and st.get("status") == "passed" for st in step_records(d)):
            return f"failed after its {LEAN_STEP} passed"
        if killed(d):
            return "killed"
        for st in failed_steps(d):
            if m := FAULT_RE.search(_step_log(d, st)):
                return f"failed in {st.get('step')} by its machine ({m[0]})"
    return None
