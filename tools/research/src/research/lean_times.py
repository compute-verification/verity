"""`research lean-times [--hours 24] [--until T] [--over 15] [--json]`: how long each Lean task took, read from the store's attempts:
per kind and per package, the count and the p50, p95 and max of its wall time in minutes, and each task over `--over` minutes by
name, with its node.

A task is one of:

  records       a records run (`lean_changed.py --records PKG`, result kind `lean-fast/v1`), under PKG: its attempt's wall. Beside
                it: its Lean work, the sum of the `lean-records: P: check.py exit N in S s` lines its stdout.log prints, one for each
                package it audited (PKG, then its requirers), each timed from that package's sync, once the slots and the tree's lock
                are held, to check.py's exit; or with none (the `reused` path, a run that ended before check.py) its wall less its
                check slot's wait. Its wait is the wall outside the work: the check slot's (`slot.json`, took less
                since) and the rest, which is the node's Lean `build` slot (`lean_slot.py` logs that wait without times), the slot
                tree's lock and the run's start; its `path` (`slot`, `dev-cache` or `reused`); and how cold it ran: the modules Lake
                compiled, dependencies included (the distinct `M` of its `lean-audit/**/build*.log` lines `Built M (t)`), against
                the modules of the packages it audited (each package's `AUDIT P: ... in N modules` line, once), the rest of which it
                restored or replayed. A records run without that result (cancelled or killed) is unfinished: counted, not timed.
  lean-audit, lean-suites, lean-build, lean-changed
                one such step of a recorded check (`check.py`'s result: `check/v1`, a shard's or a deferred run's), its `seconds`,
                which for `lean-audit` include its wait for a Lean audit slot. One check.py audits a step's packages together, so a
                `lean-audit` step counts once in its kind and once under each package it audited (its result's `reuse`, the
                packages not reused); a `lean-suites` step is one task for each suite it ran (its result's `reuse`), under that
                suite, timed by that suite's own seconds, with the step's outcome; another step's package, and a step's that ran
                none, is the part of its name after `:`, else `-`. A skipped step is no task, a stopped one is unfinished.

The window is the tasks that ended in it: a records run at its attempt's end, a step at its `ended_utc`. The attempts are those
the store's remote holds that started up to `LOOKBACK_H` before the window; blobs are read from the remote and not kept, since a
day's build logs are gigabytes. Percentiles are nearest-rank, as `research queue stats`'s.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

from .queue import FAST_KIND, LEAN_AUDIT_STEP
from .queue_stats import ts

HOURS, OVER_MIN = 24.0, 15.0
#: how long before the window the attempts read may start: a task that ended in it started at most this long before it
LOOKBACK_H = 12.0
#: run records read at once
READ_JOBS = 32
CHECK = "tools/verity/check/check.py"
RECORDS = re.compile(r"lean_changed\.py --records (\S+)")
SUITES_STEP = "lean-suites"
STEPS = (LEAN_AUDIT_STEP, SUITES_STEP, "lean-build", "lean-changed")
KINDS = ("records", *STEPS)
WORK = re.compile(r"^lean-records: (\S+): check\.py (?:--dev-cache )?exit -?\d+ in (\d+) s", re.M)
AUDITED = re.compile(r"^AUDIT (\S+): (?:PASS|FAIL)\s+\d+ declarations in (\d+) modules", re.M)
#: Lake's line for a module it compiled (`✔ [3/118] Built Flock.Field (456ms)`; `:c.o` and the like are not modules)
BUILT = re.compile(rb"\bBuilt ([^\s:]+) \((?:\d+m ?)?[\d.]+m?s\)")
BUILD_LOG = re.compile(r"^lean-audit/(?:.+/)?build(?:-[a-z]+)?\.log$")
NONE = "-"
#: the package column's width: the longest Lake package path's
W = 44
ALL = "(all)"


def clock() -> datetime:
    return datetime.now(UTC)


def pct(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(q * (len(vals) - 1)))]


def _node(ex: dict) -> str | None:
    return ex.get("machine") or (ex.get("remote") or {}).get("machine") or ex.get("host")


def _since(slot: dict) -> float:
    """A check slot's wait (`slot.json`: took less since), 0 without one."""
    if not (isinstance(slot, dict) and slot.get("took") and slot.get("since")):
        return 0.0
    return max((ts(slot["took"]) - ts(slot["since"])).total_seconds(), 0.0)


def records_task(rid: str, doc: dict, meta: dict | None, paths: list[str], read: Callable[[str], bytes]) -> dict:
    """The records run ``rid`` (its attempt ``doc``, its result's ``meta``; ``read`` gives the bytes of each of its run record's
    ``paths``) as a task."""
    ex = doc.get("execution") or {}
    fp = (meta or {}).get("workload_fingerprint") or {}
    m = RECORDS.search(" ".join(doc.get("argv") or []))
    task = {"kind": "records", "packages": [fp.get("package") or (m[1] if m else NONE)], "run": rid, "node": _node(ex),
            "ended": ex.get("end_utc"), "minutes": None, "outcome": "unfinished"}
    if (meta or {}).get("kind") != FAST_KIND:
        return task
    out = read("stdout.log").decode(errors="replace") if "stdout.log" in paths else ""
    slot = {}
    if "slot.json" in paths:
        try:
            slot = json.loads(read("slot.json"))
        except ValueError as e:
            task["note"] = f"slot.json unreadable ({e}): its check-slot wait counts as 0"
    wall, check_wait = float(ex.get("wall_s") or 0.0), _since(slot)
    work = [int(s) for _, s in WORK.findall(out)]
    work_s = float(sum(work)) if work else max(wall - check_wait, 0.0)
    modules = {p: int(n) for p, n in AUDITED.findall(out)}
    built = {b for p in paths if BUILD_LOG.match(p) for b in BUILT.findall(read(p))}
    return {**task, "minutes": wall / 60, "outcome": (meta.get("validation") or {}).get("status") or "unknown",
            "work_min": work_s / 60, "wait_min": max(wall - work_s, 0.0) / 60, "check_slot_wait_min": check_wait / 60,
            "path": fp.get("path"), "built": len(built), "modules": sum(modules.values()) if modules else None}


def check_tasks(rid: str, doc: dict, meta: dict) -> list[dict]:
    """The Lean steps (`STEPS`) of the check ``rid`` (its attempt ``doc``, its result's ``meta``) as tasks."""
    ex = doc.get("execution") or {}
    reuse = (meta.get("reuse") or {}).get("steps") or {}
    audited = [p for p in (reuse.get(LEAN_AUDIT_STEP) or {}).get("reran") or [] if isinstance(p, str)]
    suites = [r for r in (reuse.get(SUITES_STEP) or {}).get("reran") or [] if isinstance(r, dict) and isinstance(r.get("suite"), str)]
    out = []
    for s in meta.get("steps") or []:
        if not isinstance(s, dict):
            continue
        kind, _, part = str(s.get("step") or "").partition(":")
        if kind not in STEPS or s.get("status") == "skipped":
            continue
        done = s.get("status") in ("passed", "failed") and isinstance(s.get("seconds"), (int, float))
        task = {"kind": kind, "packages": [part or NONE], "run": rid, "node": _node(ex), "ended": s.get("ended_utc"),
                "minutes": s["seconds"] / 60 if done else None, "outcome": s.get("status") if done else "unfinished",
                "cached": bool((s.get("cache") or {}).get("hit"))}
        if kind == SUITES_STEP and not part and suites:
            out += [{**task, "packages": [r["suite"]],
                     **({"minutes": r["seconds"] / 60} if done and isinstance(r.get("seconds"), (int, float)) else
                        {"minutes": None, "outcome": "unfinished"})} for r in suites]
            continue
        out.append({**task, "packages": audited} if kind == LEAN_AUDIT_STEP and not part and audited else task)
    return out


def within(tasks: list[dict], start: datetime, end: datetime) -> list[dict]:
    """The ``tasks`` that ended in [start, end]."""
    keep = []
    for t in tasks:
        if (e := _ended(t)) is not None and start <= e <= end:
            keep.append(t)
    return keep


def _ended(t: dict) -> datetime | None:
    """When the task ``t`` ended, None while it hasn't; an unparseable time is noted on ``t``, which no window then holds."""
    e = t.get("ended")
    if not isinstance(e, str):
        return None
    try:
        return ts(e)
    except ValueError as err:
        t["note"] = f"ended {e!r} doesn't parse ({err}): left out of every window"
        return None


def _stats(vals: list[float], pre: str = "") -> dict:
    if not vals:
        return {}
    return {f"{pre}p50_min": round(pct(vals, 0.5), 1), f"{pre}p95_min": round(pct(vals, 0.95), 1), f"{pre}max_min": round(max(vals), 1)}


def summary(tasks: list[dict], over: float = OVER_MIN) -> dict:
    """{kind: {package: row}}: each kind's tasks under `ALL` and under each of their packages. A row is the count of timed tasks
    (`n`) and of unfinished ones, the p50, p95 and max of their wall time in minutes and how many took over ``over``, and for
    records the same of their wait and their work, and how many worked over ``over``."""
    acc: dict[str, dict[str, list[dict]]] = {}
    for t in tasks:
        for key in (ALL, *t["packages"]):
            acc.setdefault(t["kind"], {}).setdefault(key, []).append(t)
    out: dict[str, dict[str, dict]] = {}
    for kind in [k for k in KINDS if k in acc] + sorted(set(acc) - set(KINDS)):
        rows = sorted(acc[kind].items(), key=lambda kv: (kv[0] != ALL, -len(kv[1]), kv[0]))
        out[kind] = {}
        for key, group in rows:
            timed = [t for t in group if t["minutes"] is not None]
            mins = [t["minutes"] for t in timed]
            row = {"n": len(mins), "unfinished": len(group) - len(mins), "over": sum(m > over for m in mins), **_stats(mins)}
            if kind == "records":
                work = [t["work_min"] for t in timed if t.get("work_min") is not None]
                row |= {**_stats([t["wait_min"] for t in timed if t.get("wait_min") is not None], "wait_"), **_stats(work, "work_"),
                        "work_over": sum(m > over for m in work)}
            out[kind][key] = row
    return out


def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def render(s: dict, tasks: list[dict], start: datetime, end: datetime, over: float = OVER_MIN) -> str:
    gt = f">{over:g}"
    lines = [f"Lean tasks that ended {start:%Y-%m-%d %H:%MZ} to {end:%Y-%m-%d %H:%MZ}: wall time in minutes, nearest-rank "
             "percentiles; for records runs also their wait (the wall outside their Lean work: the check and Lean slots, the start) "
             "and their work",
             f"{'kind':<12} {'package':<{W}} {'n':>5} {'unfin':>5} {'p50':>6} {'p95':>6} {'max':>6} {gt:>5} {'wait50':>6} {'wait95':>6} "
             f"{'work50':>6} {'work95':>6} {'workmx':>6} {'w' + gt:>5}"]
    for kind, rows in s.items():
        for key, r in rows.items():
            rec = kind == "records"
            lines.append(f"{kind:<12} {key[:W]:<{W}} {r['n']:>5} {r['unfinished']:>5} {_m(r.get('p50_min')):>6} "
                         f"{_m(r.get('p95_min')):>6} {_m(r.get('max_min')):>6} {r['over']:>5} {_m(r.get('wait_p50_min')):>6} "
                         f"{_m(r.get('wait_p95_min')):>6} {_m(r.get('work_p50_min')):>6} {_m(r.get('work_p95_min')):>6} "
                         f"{_m(r.get('work_max_min')):>6} {r['work_over'] if rec else '-':>5}")
    slow = sorted((t for t in tasks if t["minutes"] is not None and t["minutes"] > over), key=lambda t: (-t["minutes"], t["run"]))
    recs = [t for t in slow if t["kind"] == "records"]
    lines += ["", f"records runs over {over:g} min: {len(recs)}, {sum(t['work_min'] > over for t in recs)} of them over {over:g} min "
              "of Lean work; the longest first (chk-slot: the check slot's part of the wait; compiled: the modules Lake compiled, "
              "dependencies included, against the audited packages' modules)",
              f"{'run':<22} {'package':<{W}} {'node':<17} {'wall':>6} {'work':>6} {'wait':>6} {'chk-slot':>8} {'path':<9} "
              f"{'compiled':>11} outcome"]
    for t in recs:
        comp = f"{t['built']}/{t['modules'] if t['modules'] is not None else '?'}"
        lines.append(f"{t['run']:<22} {t['packages'][0][:W]:<{W}} {str(t['node'])[:17]:<17} {_m(t['minutes']):>6} "
                     f"{_m(t['work_min']):>6} {_m(t['wait_min']):>6} {_m(t['check_slot_wait_min']):>8} {str(t['path']):<9} "
                     f"{comp:>11} {t['outcome']}")
    steps = [t for t in slow if t["kind"] != "records"]
    lines += ["", f"check steps' tasks over {over:g} min: {len(steps)}, the longest first",
              f"{'run':<22} {'step':<12} {'node':<17} {'min':>6} {'outcome':<8} packages"]
    for t in steps:
        pk = ", ".join(t["packages"])
        if len(pk) > 80:
            pk = f"{len(t['packages'])} packages: {pk[:60]}..."
        lines.append(f"{t['run']:<22} {t['kind']:<12} {str(t['node'])[:17]:<17} {_m(t['minutes']):>6} {t['outcome']:<8} {pk}")
    return "\n".join(lines)


def _bytes(store, sha: str) -> bytes:
    """Blob ``sha``'s bytes, from this store's objects or else its remote, checked and not kept."""
    from .store.local import IntegrityError, object_key

    if (p := store.object_path(sha)).is_file():
        return p.read_bytes()
    with tempfile.TemporaryDirectory() as d:
        store.remote.get(object_key(sha), Path(d) / "blob")
        data = (Path(d) / "blob").read_bytes()
    if (got := hashlib.sha256(data).hexdigest()) != sha:
        raise IntegrityError(f"the remote's blob {sha} hashes to {got}")
    return data


def _record_files(store, art: str | None) -> dict[str, str]:
    """{path: sha256} of a run record's files (none without one)."""
    if not art:
        return {}
    m = store.get_manifest(art)
    return {f.path: f.sha256 for f in (m.payload.files or [] if m.payload is not None and m.payload.type == "tree" else [])}


def _tasks_of(store, rid: str, doc: dict) -> list[dict]:
    """The Lean tasks of attempt ``rid`` (its manifest ``doc``): a records run's, or a check's Lean steps."""
    argv, out = " ".join(doc.get("argv") or []), doc.get("outputs") or {}
    meta = store.get_manifest(out["result"]).meta if out.get("result") else None
    if RECORDS.search(argv):
        files = _record_files(store, out.get("run_record")) if (meta or {}).get("kind") == FAST_KIND else {}
        return [records_task(rid, doc, meta, list(files), lambda p: _bytes(store, files[p]))]
    if CHECK in argv and meta:
        return check_tasks(rid, doc, meta)
    return []


def collect(store, start: datetime, end: datetime, log: Callable[[str], None] = lambda s: print(s, file=sys.stderr)) -> list[dict]:
    """The Lean tasks that ended in [start, end], from the attempts the store's remote holds that started after `LOOKBACK_H`
    before ``start``."""
    if store.remote is None:
        raise RuntimeError("this machine's store has no remote, so the attempts other machines published aren't here (R2_ENDPOINT "
                           "and R2_BUCKET, or ~/.research/store.toml)")
    since = f"r{start - timedelta(hours=LOOKBACK_H):%Y%m%d-%H%M%S}"
    rep = store.refresh(attempts_only=True, since=since)
    if errs := rep.get("errors"):
        log(f"research lean-times: {len(errs)} attempt(s) on the remote couldn't be read, e.g. {errs[0]}")
    rows = [(rid, json.loads(doc)) for rid, doc in store.index.sql("SELECT id, manifest_json FROM attempts WHERE id > ? AND id <= ? "
                                                                    "ORDER BY id", (since, f"r{end:%Y%m%d-%H%M%S}~"))
            if doc]
    rows = [(rid, d) for rid, d in rows if RECORDS.search(a := " ".join(d.get("argv") or [])) or CHECK in a]
    log(f"research lean-times: {len(rows)} records run(s) and check(s) started from {since[1:]} on {store.remote.describe()}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_JOBS) as ex:
        each = list(ex.map(lambda r: _tasks_of(store, *r), rows))
    return within([t for got in each for t in got], start, end)


def main(argv: list[str], store=None) -> int:
    ap = argparse.ArgumentParser(prog="research lean-times", description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=HOURS, help="the window's length (default %(default)g)")
    ap.add_argument("--until", default=None, metavar="T", help="the window's end, an ISO time (default now)")
    ap.add_argument("--over", type=float, default=OVER_MIN, metavar="MIN",
                    help="name each task over this many minutes (default %(default)g)")
    ap.add_argument("--json", action="store_true", help="the window, the summary and every task, as one JSON object")
    a = ap.parse_args(argv)
    try:
        end = ts(a.until) if a.until else clock()
    except ValueError:
        ap.error(f"--until {a.until!r}: not an ISO time")
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(hours=a.hours)
    if store is None:
        from .store.local import LocalStore

        store = LocalStore()
    try:
        tasks = collect(store, start, end)
    except Exception as e:  # noqa: BLE001  (no remote, a failed listing: say so, never answer from what's here)
        print(f"research lean-times: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    s = summary(tasks, a.over)
    if a.json:
        print(json.dumps({"window": {"start": f"{start:%Y-%m-%dT%H:%M:%SZ}", "end": f"{end:%Y-%m-%dT%H:%M:%SZ}"}, "over_min": a.over,
                          "summary": s, "tasks": sorted(tasks, key=lambda t: (t["kind"], t["ended"] or "", t["run"]))}, indent=1))
    else:
        print(render(s, tasks, start, end, a.over))
    return 0
