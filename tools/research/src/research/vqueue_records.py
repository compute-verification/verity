"""The records run of a verified-queue candidate whose member changes Lean, a lock or a spec."""
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

Told = tuple[str, tuple[str, ...]]
NAMED = (("guarantees", re.compile(r"^guarantee (\S+): ")), ("assumptions", re.compile(r"^assumption (\S+): ")),
         ("domain", re.compile(r"^domain hypothesis (\S+): ")), ("open", re.compile(r"^open premise (\S+): ")))
READ_BY = re.compile(r"^definition .*, read by (.+): (?:new|gone|changed|in a changed module)$")
FAULT_RE = re.compile(r"No space left on device|\bENOSPC\b|\[Errno 28\]|Cannot allocate memory|\[Errno 12\]|\bMemoryError\b"
                      r"|\b[Oo]ut of memory\b|could not create numbered dir with prefix \S+ in \S+ after \d+ tries"
                      r"|pool '[^']+' is closed on this machine")
LOCK = "lean-audit.json"
LOCK_SUBJECT = "Lock records regenerated for "


def lock_path(pkg: str) -> str:
    from .merge import REPAIRED

    return pkg if pkg in REPAIRED else f"{pkg}/{LOCK}" if pkg not in ("", ".") else LOCK


def lock_checked(rid: str, into: str, tip: str) -> str:
    return f"Checked: {rid} (the lock run of `{into}` {tip[:12]}, passed by regenerating them"


def lock_message(prs: str, rid: str, into: str, tip: str, warned: list[str], files: list[str],
                 changed: Iterable[str] = (), onto: str | None = None) -> str:
    said = ("What these records change of what a guarantee says, which the spec DM shows once they land:\n" +
            "".join(f"  {c}\n" for c in changed) + "\n") if (changed := list(changed)) else ""
    repairs = ("Bookkeeping the lock run repaired, each a warning for the next PR to fix:\n" +
               "".join(f"  {w}\n" for w in warned) + "\n") if warned else ""
    carried = (f"Carried from {tip[:12]} onto {onto[:12]}, the same members merged onto another base, whose Lean inputs are "
               f"{tip[:12]}'s (`vqueue_slim.lean_key`): all these records depend on.\n\n") if onto else ""
    return (f"{LOCK_SUBJECT}{prs}\n\n{said}{repairs}{carried}{lock_checked(rid, into, tip)})\n"
            f"Regenerated: {', '.join(files)}\n")


BOOKKEEPING_SHOWN = 12


def bookkeeping_told(rid: str, into: str, tip: str, prs: str, lock: str, warned: list[str]) -> tuple[str, tuple[str, ...]]:
    owners = sorted({h for w in warned for h in re.findall(r"\(@([\w-]+)\)", w)} - {"lean"})
    shown = "; ".join(warned[:BOOKKEEPING_SHOWN]) + (f"; and {len(warned) - BOOKKEEPING_SHOWN} more in the commit"
                                                    if len(warned) > BOOKKEEPING_SHOWN else "")
    return (f"the lock run {rid} of `{into}` {tip[:12]} repaired bookkeeping that {prs} brought, in {lock[:12]}, each a warning "
            f"for the next PR to fix: {shown}", ("lean", *owners))


def start(driver: Landing, sha: str, on: str = "") -> dict | None:
    if not (machines := on or driver.on):
        raise ValueError("a candidate's records run needs the queue's machines: build the Landing with drive.toml's `on` "
                         "(`research.queue.flags_at(repo, main).on`)")
    what = f"train.sh {sha[:12]} {machines} --deferred {LEAN_STEP}, the records run of candidate {sha[:12]}"
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    rec, tail = driver._started(driver.launch_locks(driver.repo, sha, machines))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or 'launching'}")
    return {**rec, "tip": sha, "at": driver.now(), **({"on": on} if on else {})}


def judge(driver: Landing, rec: dict | None) -> RecordsRun:
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
    try:
        report = lean_report(driver.q.store, rid, driver.fetch)
    except Exception as e:  # noqa: BLE001  (unread, it holds the commit: it is never taken as passed)
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read ({type(e).__name__})")
    if not isinstance(report, dict):
        return RecordsRun(PENDING, f"{rid} passed, but its Lean report ({AUDIT}) can't be read yet from the store or from its "
                                   "run dir on its node")
    return _keep(rec, RecordsRun(PASS), rid)


def _keep(rec: dict, r: RecordsRun, run: str | None = None) -> RecordsRun:
    rec["verdict"] = {"state": r.state, "why": r.why, **({"run": run} if run else {})}
    return r


def _kept(got: dict) -> RecordsRun:
    return RecordsRun(str(got.get("state") or INFRA), str(got.get("why") or ""))


def _changes(driver: Landing, rid: str, tip: str) -> list[str]:
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
    from .vqueue_slim import lean_key
    return (key := lean_key(repo, a)) is not None and key == lean_key(repo, b)


def commit(driver: Landing, sha: str, rid: str, prs: Iterable[int],
           onto: str | None = None) -> tuple[str, list[str], Told | None] | str | None:
    repo, into = driver.repo, f"queue/{sha[:12]}"
    if onto is not None and not same_inputs(repo, sha, onto):
        raise QueueError(f"{onto[:12]}'s Lean inputs aren't {sha[:12]}'s: the records {rid} regenerated on {sha[:12]} aren't "
                         "its records")
    scratch = Path(tempfile.mkdtemp(prefix="vqueue-records-"))
    try:
        locks, held = regenerated_locks(driver, rid, scratch)
        if held:
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
    where = driver._elsewhere(states, rec.get("on") or "")
    again, tail = driver._started(driver.launch_locks(driver.repo, rec["tip"], where))
    if again is not None:
        rec.update(**again, at=driver.now(), resend=True)
    return where, again, tail


def regenerated_locks(driver: Landing, rid: str, scratch: Path) -> tuple[dict[str, Path], str | None]:
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
    from .merge import declared_command, produced, regenerated_parent

    if regenerated_parent(driver.repo, commit, driver.q.store, tool="check", kinds=produced(driver.repo, "check"),
                          command=declared_command(driver.repo, "check"), deferred=rid, fetch=driver.fetch):
        return None
    return (f"the merge gate refuses its lock commit: its records aren't byte for byte the ones {rid} regenerated "
            "(`merge.regenerated_parent`, on the live run)")


def lean_audit_ran(driver: Landing, rid: str) -> bool:
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
