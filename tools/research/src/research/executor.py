"""The task executor's tasks/v1 side: each task's verdict read by its key, and the shadow records of after-landing runs.

`tasks/v1` (`verity tasks list --json`, `tools/check/tasks.py`) is the graph schema.  A graph's keys hash the lister's
environment, so the executor lists a check on a worker, puts it in the store (`tasks-graph/v1`, `TASKS_GRAPH_KIND`) and labels
`commit:<sha> tasks_graph art:<id>` (`vocab.TASKS_GRAPH`).

A task's verdict comes from the `verdict` labels on `task:<key>` (`task_target`), written by the executor from its run of `verity
tasks run ID`: pass or fail by that run's result, or `ERROR` when no run gave one, which is pending (not run); a step that
skipped gets no label and stays pending.  A fail stands until an error label is written after it (a false fail relabelled), so
a rerun that passes never turns a fail green; past that, the newest of pass and error is the key's (`fold`).  A task with no key
of its own (tasks/v1's `unkeyed`) gets one of its check alone (`keyed`), so no verdict of it is reused across checks.

The shadow records (`executor_shadow` labels on `commit:<tip>`) compare an after-landing run's verdicts with the executor's
(`old_verdicts`, `compare`); `research queue shadow-report` prints the newest of them and whether `SWITCH_STREAK` tips in a row
agreed fully (`agrees`)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
#: the label of a task whose runs all ended without a result (the machine lost, the run refused or cancelled): no verdict of
#: the tree, so `fold` reads it as pending
ERROR = "error"
#: a shadow comparison's side with no counterpart: the run has no verdict of an executor-only task, and the executor has no
#: task of a part the run records (missing in executor)
NONE, MISSING = "none", "missing"
#: the store kind of a tasks/v1 graph the executor listed
TASKS_GRAPH_KIND = "tasks-graph/v1"
#: the newest tips in a row whose shadow records agree fully before the executor may gate
SWITCH_STREAK = 3
#: what a check's step and suite records say, as a task's verdict: a step skipped by name, or a pass reused, is a pass
OLD_PASS = ("passed", "skipped", "reused", "cached")
OLD_FAIL = ("failed",)
#: a suite's record that is no verdict of it: a step skips by name the suites another step runs (`pytest` skips the Lean suites)
SUITE_NO_VERDICT = ("skipped",)
#: a task of a suite split by test file (`pytest:SUITE:K/M`): compared through its suite when the run records it whole
SPLIT_RE = re.compile(r":\d+/\d+$")
#: a lean-audit log's verdict on one Lake package: its audit (`AUDIT PKG: PASS|FAIL`, PKG under the audit's scratch tree on a
#: check's run) or a pass reused (`lean-audit: PKG: PASS reused`)
AUDIT_RE = re.compile(r"^AUDIT (?:\S*?/lean-audit-scratch-[^/\s]+/)?(\S+?): (PASS|FAIL)\b", re.M)
REUSED_RE = re.compile(r"^lean-audit: (\S+?): PASS reused\b", re.M)


def task_target(key: str) -> str:
    """The store target of a task's key: `task:<128 hex>`, its key without a `sha512:`."""
    return f"task:{key.removeprefix('sha512:')}"


def check_id(graph: dict) -> str:
    """The check a graph is of: its kind and commit (`after-landing:SHA`)."""
    return f"{graph.get('kind') or 'after-landing'}:{graph['commit']}"


def _digest(doc) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def keyed(graph: dict) -> dict:
    """``graph`` with each task's `key` the one its verdict is labelled under: its own, or for an unkeyed task, SHA-512 over its
    check and id (its `unkeyed` reason kept), which no other check shares."""
    check = check_id(graph)
    return {**graph, "tasks": [t if t.get("key") else {**t, "key": _digest({"check": check, "id": t["id"]})}
                               for t in graph["tasks"]]}


def fold(values: Iterable[str]) -> str:
    """A key's verdict from its labels' values, oldest first: fail while a fail has no `ERROR` after it, else pass when the
    newest is pass, else pending (no label, or `ERROR` newest)."""
    vs = list(values)
    if FAIL in vs and ERROR not in vs[len(vs) - vs[::-1].index(FAIL):]:
        return FAIL
    return PASS if vs and vs[-1] == PASS else PENDING


# -- the shadow comparison: an after-landing run's verdicts beside the executor's, per task where the run records one, else per
#    job (a run records circuit-check, the builds and the other whole steps once, however many tasks the graph cuts them into)

def old_verdict(status: str | None) -> str | None:
    return PASS if status in OLD_PASS else FAIL if status in OLD_FAIL else None


def old_verdicts(run_dirs: Iterable[Path]) -> dict[str, str]:
    """{part: pass | fail} of an after-landing run's records, over its shards: each step (steps.jsonl), each suite as its task
    id (`pytest:NAME`, `lean-suites:NAME`), each Lake package its lean-audit log names (`lean-audit:PKG`).  A part failed in any
    shard fails; one stopped or never run is absent, and so are the preflight steps, which no after-landing graph holds
    (`tasks.py` runs a part without them), and a suite a step skipped by name (`SUITE_NO_VERDICT`)."""
    from .landing import PREFLIGHT, step_records, suite_rows

    got: dict[str, set[str]] = {}

    def put(part: str, v: str | None) -> None:
        if v is not None:
            got.setdefault(part, set()).add(v)

    for d in run_dirs:
        for st in step_records(d):
            if st.get("step") in PREFLIGHT:
                continue
            put(st.get("step") or "", old_verdict(st.get("status")))
            if st.get("step") == "lean-audit" and st.get("log"):
                try:
                    log = (d / Path(st["log"]).name).read_text(errors="replace")
                except OSError:
                    log = ""
                for pkg, said in AUDIT_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS if said == "PASS" else FAIL)
                for pkg in REUSED_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS)
        for name, s in suite_rows(d).items():
            if s.get("status") not in SUITE_NO_VERDICT:
                put(f"{s['step']}:{name}", old_verdict(s.get("status")))
    return {p: FAIL if FAIL in vs else PASS for p, vs in got.items() if p}


def compare(ids: dict[str, str], old: dict[str, str], new: dict[str, str]) -> dict:
    """How the run's verdicts (``old``, `old_verdicts`) and the executor's (``new``, by key, of the tasks ``ids`` {task id:
    key}) compare, over the union of both sides' parts:

    - each task the run records on its own (`pytest:SUITE`, `lean-audit:PKG`, a job run whole): its verdict beside the task's;
    - each job the run records as a step: the step's verdict beside its tasks' folded (fail when one fails, pending when one
      is, else pass), so a task of a job the run records only whole (a circuit-check slice) is compared through it;
    - each suite the run records whole that the graph splits by test file (`SPLIT_RE`): its verdict beside its parts' folded;
    - a part the run records that no task matches, by id or by job: "missing in executor" (`MISSING`), a disagreement;
    - a task the run has no counterpart of (its id unrecorded, and its job unrecorded or recorded part by part without it):
      "executor only", a disagreement whatever its verdict.

    So a graph smaller than the run never agrees fully, nor one that ran something the run didn't."""
    units: dict[str, tuple[str, str]] = {}
    jobs: dict[str, list[str]] = {}
    split: dict[str, list[str]] = {}
    only = []
    by_part = {p.partition(":")[0] for p in old if ":" in p}
    for tid, key in ids.items():
        v, job = new.get(key, PENDING), tid.partition(":")[0]
        jobs.setdefault(job, []).append(v)
        if tid in old:
            units[tid] = (old[tid], v)
        elif (whole := SPLIT_RE.sub("", tid)) != tid and whole != job and whole in old:
            split.setdefault(whole, []).append(v)
        elif tid == job or job in by_part or job not in old:
            units[tid] = (NONE, v)
            only.append(tid)

    def folded(vs: list[str]) -> str:
        return FAIL if FAIL in vs else PENDING if PENDING in vs else PASS

    for job, vs in jobs.items():
        if job in old:
            units[job] = (old[job], folded(vs))
    units.update({whole: (old[whole], folded(vs)) for whole, vs in split.items()})
    missing = sorted(p for p in old if p not in units)
    units.update({p: (old[p], MISSING) for p in missing})
    rows = sorted(units.items())
    return {"units": len(rows), "agree": sum(o == n for _, (o, n) in rows),
            "pending": sorted(u for u, (o, n) in rows if n == PENDING and o != NONE),
            "disagree": [{"unit": u, "old": o, "executor": n} for u, (o, n) in rows
                         if o != n and (n != PENDING or o == NONE)],
            "missing": missing, "executor_only": sorted(only)}


def agrees(rec: dict) -> bool:
    """A tip's shadow record agrees fully: every unit agrees, nothing pending, missing in the executor or executor only, and
    no error and nothing left incomplete.  A record without the coverage fields (`compare`'s "missing", "executor_only")
    never agrees."""
    return (not rec.get("error") and not rec.get("incomplete") and rec.get("units", 0) > 0
            and rec.get("agree") == rec.get("units") and not rec.get("pending") and not rec.get("disagree")
            and rec.get("missing") == [] and rec.get("executor_only") == [])


# -- `research queue shadow-report`

@dataclass
class Shadowed:
    tip: str
    ts: str
    rec: dict


def shadow_records(store, pull: bool = True, log: Callable[[str], None] = print) -> list[Shadowed]:
    """Every tip's newest `executor_shadow` record, newest first, its labels pulled from the store's remote first."""
    from .store import ids, labels_sync

    if pull and getattr(store, "remote", None) is not None:
        try:
            labels_sync.pull_prefix(store, "commit:")
        except Exception as e:  # noqa: BLE001  (the local labels still answer, said so)
            log(f"research queue shadow-report: the store's remote wasn't read ({type(e).__name__}: {e}); local records only")
    out = []
    for d in sorted((Path(store.root) / "labels").glob(f"{ids.target_dirname('commit:')}*")):
        tip = d.name.partition("_")[2]
        if labs := store.labels(f"commit:{tip}", key=vocab.EXECUTOR_SHADOW):
            out.append(Shadowed(tip, labs[-1].ts, labs[-1].value))
    return sorted(out, key=lambda s: s.ts, reverse=True)


def shadow_report(records: list[Shadowed], tips: int) -> tuple[str, bool]:
    """(the report of the newest ``tips`` shadowed tips, whether the switch gate holds: the newest `SWITCH_STREAK` agree fully)."""
    shown = records[:tips]
    streak = next((i for i, s in enumerate(records) if not agrees(s.rec)), len(records))
    lines = [f"executor shadow: the newest {len(shown)} of {len(records)} checked tips compared"]
    for s in shown:
        r = s.rec
        what = (f"error: {r['error']}" if r.get("error") else f"incomplete: {r['incomplete']}" if r.get("incomplete") else
                f"agree {r.get('agree', 0)}/{r.get('units', 0)}, disagree {len(r.get('disagree') or ())}"
                + (f", pending {len(r['pending'])}" if r.get("pending") else "")
                + ("" if "missing" in r else ", no coverage counts")
                + (f", missing in executor {len(r['missing'])}" if r.get("missing") else "")
                + (f", executor only {len(r['executor_only'])}" if r.get("executor_only") else ""))
        lines.append(f"{s.tip[:12]}  {s.ts}  {what}  (run {r.get('run') or '-'})")
        for x in r.get("disagree") or ():
            lines.append(f"    {x['unit']}: " + ("missing in executor (run " + x["old"] + ")" if x["executor"] == MISSING else
                                                 f"executor only (executor {x['executor']})" if x["old"] == NONE else
                                                 f"run {x['old']}, executor {x['executor']}"))
        lines += [f"    {u}: executor pending" for u in r.get("pending") or ()]
    ok = streak >= SWITCH_STREAK
    lines.append(f"switch gate ({SWITCH_STREAK} tips in a row agreeing fully, none missing in executor or executor only, none "
                 f"incomplete): {'met' if ok else 'not met'} "
                 f"(the newest {streak} agree)")
    return "\n".join(lines), ok

