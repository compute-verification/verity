"""What a rehearsal of a kept window's audit keeps (rehearse_audit.sh). A rehearsal is no audit of record: it samples a
kept registration again on a local development session, and what this writes says so and is no verdict of a record.
From the audit's own directory and the kept audit's it keeps only: the warm's and the audit's exits and seconds, the
honest arm's outcome (the answer `served_integrity.answer` gives on it, whether its record accepted and why not, and
each check's verdict), its stages' seconds, the negatives' workers' record, the memo's counts, each negative's refusal
beside the kept audit's by name with its codes, and the most disk the job's directory held. A negative is listed as
refused only where a verdict said so: one whose code is ``disagree``, or that Lean gave no verdict line for, is listed as
not answered. For the memory: each sampled member's circuit, program and partition files' bytes, each command its session
logged with its exit and seconds, the line it owes by its verb and whether it gave one (of the sample's check, each
case's ``ok``), the job's cgroup's limit, kills and peak as the job read them, of the memory scope that limits it
the times it reached its limit and what its processes and its file cache hold, the node's MemAvailable at the
job's start and at its lowest with the stage that was seen in, and what the audit's stderr says of a fork with more than
one thread alive. Of a job that stopped itself: the stage it stopped in, why, and the sampled member's circuit file's
bytes. No count or index of the window is written, and no size but those files': of
each JSON read, only the keys named here, and under ``staging`` and ``timing`` only seconds; a reason is kept with every
number in it struck."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def seconds(x):
    """``x`` with only its leaves that are seconds: numbers under a key ending ``_s``."""
    if not isinstance(x, dict):
        return None
    kept = {}
    for k, v in x.items():
        if isinstance(v, dict):
            v = seconds(v)
        elif not (str(k).endswith("_s") and isinstance(v, (int, float)) and not isinstance(v, bool)):
            v = None
        if v is not None:
            kept[k] = v
    return kept or None


def refusals(audit: Path) -> dict[str, bool] | None:
    """Each negative's name -> whether it was refused, from a5's negatives file, else its summary."""
    neg = load(audit / "window" / "negatives.json")
    if isinstance(neg, dict):
        return {k: bool(v.get("refused")) if isinstance(v, dict) else bool(v) for k, v in neg.items()}
    summary = load(audit / "window" / "summary.json") or {}
    neg = summary.get("negatives")
    return {k: bool(v) for k, v in neg.items()} if isinstance(neg, dict) else None


def entries(audit: Path) -> dict:
    """a5's negatives file as it stands: each negative's name -> its entry."""
    neg = load(audit / "window" / "negatives.json")
    return neg if isinstance(neg, dict) else {}


def codes(v) -> list[str]:
    """An entry's codes: its own, Lean's under ``lean``, and the checks it lists as failed."""
    if not isinstance(v, dict):
        return []
    lean = v.get("lean") if isinstance(v.get("lean"), dict) else {}
    return [str(c) for part in (v.get("codes"), lean.get("codes"), v.get("failed")) if isinstance(part, list) for c in part]


def answer(v) -> str:
    """``refused``, ``not refused`` or ``not answered``. Not answered: the entry's code is ``disagree`` (the two verifiers
    did not agree, which is also what a Lean process that ended leaves), its ``agree`` is false, its ``lean`` holds no
    verdict, or its reason is a0's stand-in for no verdict line."""
    if not isinstance(v, dict):
        return "refused" if v else "not refused"
    why = v.get("why") if isinstance(v.get("why"), str) else ""
    if ("disagree" in codes(v) or v.get("agree") is False or ("lean" in v and not isinstance(v["lean"], dict))
            or why.startswith("no verdict")):
        return "not answered"
    return "refused" if v.get("refused") else "not refused"


def reason(v) -> str | None:
    """An entry's reason with every number and digest struck, cut short."""
    why = v.get("why") if isinstance(v, dict) else None
    if not isinstance(why, str):
        why = v.get("python") if isinstance(v, dict) and isinstance(v.get("python"), str) else None
    return re.sub(r"[0-9a-f]{16,}|[0-9]+", "N", why)[:160] if why else None


def size(p: Path) -> int | None:
    try:
        return p.stat().st_size
    except OSError:
        return None


#: the line each command a session's directory logs owes in its output, by the command's verb (`a0.run`'s first line):
#: Lean's check of a proof a verdict, the prover its session's outcome, Lean's check of the sample one line a case.
#: A command of another verb (a put into the archive) owes none
OWES = {"verify": "VERDICT ", "prove": "LIVE\t", "draw-test": "DRAWCHECK "}


def members(audit: Path) -> list[dict]:
    """Each sampled member (one with a session directory), in the directories' order and without its index: the bytes of
    the files its Lean check reads, and each command its session directory logged (`a0.run`'s lines: its verb, its exit
    and seconds; the line it owes by its verb (`OWES`) and whether its output held one; of the sample's check, each
    case's ``ok``)."""
    out = []
    for sdir in sorted((audit / "window").glob("session-m*")):
        w, logs = sdir.parent, []
        for log in sorted(sdir.glob("*.log"), key=lambda p: p.stat().st_mtime):
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            m = re.search(r"^# rc (\S+) in ([0-9.]+) s$", text, re.M)
            if m:
                verb = re.match(r"\$ \S+ ([a-z][a-z-]*)(?: |$)", text.split("\n", 1)[0])
                verb = verb.group(1) if verb else None
                owes = OWES.get(verb)
                said = [x for x in text.split("--- stderr", 1)[0].splitlines() if owes and x.startswith(owes)]
                entry = {"log": log.name, "command": verb, "exit": m.group(1), "seconds": float(m.group(2)),
                         "owes": owes.strip() if owes else None, "owed_line": bool(said) if owes else None,
                         "ended_at": round(log.stat().st_mtime, 1)}
                if verb == "draw-test":
                    entry["drawcheck_ok"] = [drawcheck_ok(x) for x in said]
                logs.append(entry)
        out.append({"circuit_bytes": size(w / sdir.name.removeprefix("session-") / "circuit.txt"),
                    "program_bytes": size(w / "program.json"), "partition_bytes": size(w / "partition.json"),
                    "registered_bytes": size(w / sdir.name.removeprefix("session-") / "registered.json"),
                    "lean_commands": logs})
    return out


def drawcheck_ok(line: str) -> bool | None:
    """A ``DRAWCHECK`` line's ``ok``, and nothing else of it; None where the line is not Lean's JSON."""
    try:
        ok = json.loads(line.split(" ", 1)[1]).get("ok")
    except (IndexError, ValueError, AttributeError):
        return None
    return ok if isinstance(ok, bool) else None


def owed(c: dict) -> str:
    """What a logged command's entry (`members`) says of the line it owes, for the output."""
    if c["owes"] is None:
        return "owes no line"
    if not c["owed_line"]:
        return f"NO {c['owes']} LINE"
    if "drawcheck_ok" not in c:
        return f"its {c['owes']} line"
    oks = c["drawcheck_ok"]
    return (f"{len(oks)} DRAWCHECK line(s), " + ("every ok true" if all(x is True for x in oks) else
                                                 "NOT EVERY OK TRUE: " + ", ".join(str(x).lower() for x in oks)))


def fork_lines(err: Path | None) -> dict | None:
    """What the audit's stderr says of a fork with more than one thread alive: Python's own warning (shown only under
    ``PYTHONWARNINGS=default::DeprecationWarning``; without it the stderr says nothing either way)."""
    if err is None:
        return None
    try:
        lines = [x for x in err.read_text(errors="replace").splitlines() if "multi-threaded" in x]
    except OSError:
        return {"read": False}
    return {"read": True, "lines": len(lines), "first": re.sub(r"pid=[0-9]+", "pid=N", lines[0])[-200:] if lines else None}


def mem_available(start_kb: int | None, low_kb: int | None, low_stage: str | None, stop_gb: int | None) -> dict | None:
    """The node's MemAvailable as the job read it every 5 s (/proc/meminfo's kB are 1024 bytes): at the job's start, at
    its lowest with the stage that was seen in, the most taken between the two, and the taking the job stops itself at."""
    if start_kb is None:
        return None
    low = start_kb if low_kb is None else low_kb
    return {"at_start_bytes": start_kb * 1024, "lowest_bytes": low * 1024, "lowest_stage": low_stage,
            "most_taken_bytes": (start_kb - low) * 1024, "stops_at_taken_bytes": None if stop_gb is None else stop_gb * 10 ** 9}


def stop_reason(stopped: int | None, stop_gb: int | None, said: str | None = None) -> str | None:
    """Why the job stopped itself, by its exit; ``said`` is the job's own reason where its stop wrote one: which of the
    two memory readings stopped it, the scope's or the node's, with both as they were then."""
    if stopped is None:
        return None
    if said and stopped == 72:
        return said
    return {71: "the disk ran low: less than the job's floor was free under its work directory",
            72: f"the memory ran low: MemAvailable fell more than {stop_gb} GB under its value at the job's start",
            143: "the job was told to stop"}.get(stopped, f"exit {stopped}")


def mem_line(m: dict | None) -> str:
    if m is None:
        return "REHEARSAL MemAvailable: not read"
    def g(b: int) -> str:
        return f"{b / 1e9:.1f} GB"
    stop = "" if m["stops_at_taken_bytes"] is None else f"; the job stops itself at more than {g(m['stops_at_taken_bytes'])} under"
    return (f"REHEARSAL MemAvailable: lowest {g(m['lowest_bytes'])}, seen in {m['lowest_stage']}; {g(m['at_start_bytes'])} at "
            f"the job's start, so {g(m['most_taken_bytes'])} under it at the most{stop}")


def workers_line(w) -> str:
    """The negatives' workers as a5 recorded them: how many at once, the stage's seconds, each task's own seconds
    (no task is named), and the most a worker held of its own at its end and its Lean process at its peak."""
    tasks = [t for t in (w.get("negatives") or {}).values() if isinstance(t, dict)] if isinstance(w, dict) else []
    if not tasks:
        return "REHEARSAL negatives' workers: no task's seconds recorded"
    own = sorted((t["own_s"] for t in tasks if isinstance(t.get("own_s"), (int, float))), reverse=True)
    def most(part: str, key: str) -> str:
        got = [t[part][key] for t in tasks if isinstance(t.get(part), dict) and isinstance(t[part].get(key), int)]
        return f"{max(got) / 1e9:.1f} GB" if got else "not recorded"
    return (f"REHEARSAL negatives' workers: {w.get('workers')} at once, {w.get('wall_s')} s; each task's own seconds, "
            f"longest first: {', '.join(f'{x:g}' for x in own)}; the most a worker held of its own at its end "
            f"{most('worker', 'private')}, the highest peak of a worker's Lean process {most('lean', 'peak')}")


WHAT = ("a rehearsal: a kept registration sampled again on a local development challenger session, whose key is its "
        "own, so its sample is no auditor's; no audit of record and no verdict of a record")


def verdicts(x):
    """A map of checks -> each one's verdict alone: a bool as it is, a dict's ``ok``."""
    if not isinstance(x, dict):
        return None
    return {k: (v.get("ok") if isinstance(v, dict) else v if isinstance(v, bool) else None) for k, v in x.items()}


#: the reader's exit where the honest arm's answer raised: no row of a record was computed of this audit, whatever
#: else rehearsal.json holds (1 is a rehearsal.json that is not whole)
ANSWER_RAISED = 3


def honest(audit: Path, tree: Path | None, registration: Path | None) -> dict:
    """The honest arm's outcome: the answer a record's row would be of it, and what a5 says of its record. An answer
    that raised is kept as its ``error`` (every number in it struck), which `main` prints in the answer's place and
    exits `ANSWER_RAISED` on."""
    a5 = load(audit / "window" / "summary.json") or {}
    rec = load(audit / "window" / "audit.json") or {}
    row = load(audit / "summary.json")
    said = {**a5, **(row or {})}  # the records audit's summary over a5's own, key by key
    fixed = said.get("fixed_before_draw")
    out = {"summary_written": row is not None, "ok": (row or {}).get("ok"), "refused": (row or {}).get("refused"),
           "accepted": (said.get("audit") or {}).get("accepted"), "why": (said.get("audit") or {}).get("why"),
           "record_accepted": rec.get("accepted"),
           "checks_before_draw": verdicts(said.get("checks_before_draw")),
           "checks_after_draw": verdicts(rec.get("checks_after_draw") or said.get("checks_after_draw")),
           "fixed_before_draw": {"ok": fixed.get("ok"), "codes": fixed.get("codes")} if isinstance(fixed, dict) else None,
           "sample_refused": ((said.get("checks") or {}).get("draw") or {}).get("codes"),
           "reached_its_sample": (audit / "window" / "receipt.json").exists()}
    if tree is not None and registration is not None:
        try:
            sys.path.insert(0, str(tree / "benchmarks" / "one_stage"))
            import served_integrity as SI
            got = SI.answer(audit, load(registration))
            out["answer"] = {"verdict": got.get("verdict"), "reason": got.get("reason")}
        except Exception as e:  # noqa: BLE001  the rest of rehearsal.json is written without it; `main` says it failed
            out["answer"] = {"error": re.sub(r"[0-9a-f]{16,}|[0-9]+", "N", f"{type(e).__name__}: {e}")[:300]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", type=Path)
    ap.add_argument("kept", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--exit", type=int, default=None, help="the audit's exit; none when no audit ended")
    ap.add_argument("--audit-seconds", type=int, default=None)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--warm-jobs", type=int, default=0)
    ap.add_argument("--warm-exit", type=int, default=None)
    ap.add_argument("--warm-seconds", type=int, default=None)
    ap.add_argument("--disk-gb", type=int, default=None, help="the most the job's directory held, as the job read it")
    ap.add_argument("--min-free-gb", type=int, default=None, help="the start's check: the job started only with that much free")
    ap.add_argument("--free-at-start-gb", type=int, default=None, help="what was free under the work directory at the start")
    ap.add_argument("--stopped", type=int, default=None, help="the exit the job stopped with, before its audit ended")
    ap.add_argument("--tree", type=Path, default=None)
    ap.add_argument("--tree-commit", default=None, help="where the tree audited was sent as an input: its commit")
    ap.add_argument("--tree-id", default=None, help="and the tree id the job held it to")
    ap.add_argument("--registration", type=Path, default=None)
    ap.add_argument("--audit-err", type=Path, default=None, help="the audit's stderr")
    ap.add_argument("--warnings", default=None, help="the audit's PYTHONWARNINGS, as the job set it")
    ap.add_argument("--memory", default=None,
                    help="JSON: the job's cgroup's limit, kills and peak, and the memory scope's max events and what its "
                         "processes and file cache hold, as the job read them at its start and now")
    ap.add_argument("--stage", default=None, help="the job's stage as this is written")
    ap.add_argument("--mem-start-kb", type=int, default=None, help="MemAvailable at the job's start, /proc/meminfo's kB")
    ap.add_argument("--mem-low-kb", type=int, default=None, help="the lowest MemAvailable the job read")
    ap.add_argument("--mem-low-stage", default=None, help="the stage the lowest was read in")
    ap.add_argument("--mem-stop-gb", type=int, default=None, help="the job stops itself once MemAvailable is more than this under its start")
    ap.add_argument("--stop-reason", default=None, help="the reason the job's own stop wrote, where it wrote one")
    a = ap.parse_args(argv)
    row, a5 = load(a.audit / "summary.json") or {}, load(a.audit / "window" / "summary.json") or {}
    kept_a5 = load(a.kept / "window" / "summary.json") or {}
    now, kept = refusals(a.audit), refusals(a.kept)
    names = sorted(set(now or {}) | set(kept or {}))
    mine = entries(a.audit)
    said = {n: answer(mine[n]) for n in mine}
    try:
        memory = json.loads(a.memory) if a.memory else None
    except ValueError:
        memory = {"unread": a.memory[:200]}
    sampled = members(a.audit)
    doc = {
        "rehearsal": True, "of_record": False, "what": WHAT, "session": "local",
        "tree_audited": None if a.tree_commit is None else {
            "commit": a.tree_commit, "tree_id": a.tree_id,
            "how": "sent as an input and held to this commit and tree id before the job ran in it; the run's source "
                   "is the launcher's and is not the tree audited"},
        "audit_ended": a.exit is not None, "stopped": a.stopped, "stage": a.stage,
        "stop_reason": stop_reason(a.stopped, a.mem_stop_gb, a.stop_reason),
        "sampled_circuit_bytes": [m["circuit_bytes"] for m in sampled],
        "mem_available": mem_available(a.mem_start_kb, a.mem_low_kb, a.mem_low_stage, a.mem_stop_gb),
        "exit": a.exit, "audit_seconds": a.audit_seconds, "workers_asked": a.workers, "warm_jobs": a.warm_jobs,
        "warm_exit": a.warm_exit, "warm_seconds": a.warm_seconds, "job_directory_gb": a.disk_gb,
        "start_check_gb": a.min_free_gb, "free_at_start_gb": a.free_at_start_gb,
        "honest": honest(a.audit, a.tree, a.registration) if a.exit is not None else None,
        "kept_accepted": (kept_a5.get("audit") or {}).get("accepted"),
        "negative_workers": a5.get("negative_workers") or row.get("negative_workers"),
        "verifier_checks": a5.get("verifier_checks") or row.get("verifier_checks"),
        "seconds": seconds({k: a5.get(k) for k in ("staging", "timing")}),
        "kept_seconds": seconds({k: kept_a5.get(k) for k in ("staging", "timing")}),
        "negatives": {n: {"refused": (now or {}).get(n), "kept_refused": (kept or {}).get(n), "answer": said.get(n),
                          "codes": codes(mine.get(n)), "reason": reason(mine.get(n))} for n in names},
        "not_answered": sorted(n for n, v in said.items() if v == "not answered"),
        "refused_by_a_verdict": sorted(n for n, v in said.items() if v == "refused"),
        "members": sampled, "memory": memory,
        "fork": {"warnings": a.warnings, **(fork_lines(a.audit_err) or {})},
        "negatives_read": {"now": now is not None, "kept": kept is not None},
        "same_names": now is not None and kept is not None and set(now) == set(kept),
        "same_refusals": now is not None and kept is not None and now == kept,
        "not_refused": sorted(n for n, v in (now or {}).items() if not v),
    }
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    if a.tree_commit is not None:
        print(f"REHEARSAL the tree audited: commit {a.tree_commit}, tree id {a.tree_id}, sent as an input; the run's "
              f"source is the launcher's and is not the tree audited")
    if a.exit is None:
        print(f"REHEARSAL (no audit of record): no audit ended; the warm's exit {a.warm_exit}, stopped {a.stopped}")
        if a.stopped is not None:
            size_ = (", ".join(f"{b} bytes" for b in doc["sampled_circuit_bytes"]) if doc["sampled_circuit_bytes"]
                     else "no member sampled yet")
            print(f"REHEARSAL stopped (exit {a.stopped}) in {a.stage}: {doc['stop_reason']}; the sampled member's "
                  f"circuit file: {size_}")
        print(mem_line(doc["mem_available"]))
        return 0
    h = doc["honest"]
    whole = now is not None and doc["negative_workers"] is not None
    raised = (h.get("answer") or {}).get("error")
    print(f"REHEARSAL (no audit of record) honest arm: accepted {h['accepted']}, summary ok {h['ok']}, "
          + (f"ANSWER NOT READ, it raised: {raised}" if raised else f"answer {(h.get('answer') or {}).get('verdict')}"))
    print(f"REHEARSAL negatives: {len(now or {})} now, {len(kept or {})} kept; same names {doc['same_names']}; "
          f"same refusals {doc['same_refusals']}; not refused now {len(doc['not_refused'])}")
    print(f"REHEARSAL negatives by their verdicts: {len(doc['refused_by_a_verdict'])} refused, "
          f"{sum(1 for v in said.values() if v == 'not refused')} not refused, {len(doc['not_answered'])} NOT ANSWERED "
          f"(the code disagree, or no verdict line from Lean: never counted as refused)")
    for n in mine:
        print(f"REHEARSAL negative {n}: {said[n].upper() if said[n] != 'refused' else said[n]}; codes {codes(mine[n])}")
    for i, m in enumerate(doc["members"], 1):
        print(f"REHEARSAL member {i} of {len(doc['members'])}: circuit file {m['circuit_bytes']} bytes, program file "
              f"{m['program_bytes']} bytes, partition file {m['partition_bytes']} bytes")
        for c in m["lean_commands"]:
            print(f"REHEARSAL member {i} logged command {c['log']} ({c['command'] or 'verb not read'}): exit {c['exit']}, "
                  f"{c['seconds']} s, {owed(c)}")
    print(workers_line(doc["negative_workers"]))
    if memory is not None:
        print(f"REHEARSAL memory: {json.dumps(memory, sort_keys=True)}")
    print(mem_line(doc["mem_available"]))
    f = doc["fork"]
    print("REHEARSAL fork: " + (
        "the audit's stderr was not read" if not f.get("read") else
        f"{f['lines']} line(s) of Python's warning of a fork with more than one thread alive"
        + (f", the first: {f['first']}" if f["lines"] else "")
        + ("" if a.warnings else "; PYTHONWARNINGS was not set, so Python hides that warning and none says nothing")))
    return ANSWER_RAISED if raised else 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
