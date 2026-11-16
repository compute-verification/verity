"""`research run-times`: how long each run waited to start, set up and ran, from the runs' own records.

Each run launched in the window (its id carries its launch time) is one row, read from its run directory on the machine that
holds it (`run_records.py`, sent there over ssh; `--runs DIR` reads a runs directory here instead):

  wait    launch -> its check slot taken (`slot.json`'s `took`); for a run that goes through no slot, launch -> its command
          starting (`status.json`'s `started_utc`). A run that got no slot counts with the time it waited: until it was
          cancelled or ended in the line, until slot.py gave up (`left`), or until now while it still waits.
  setup   slot taken -> the job's own command starting, where a record says when: `check-live.json`'s `started_utc`, which
          only check.py writes. slot.py starts the job's command line the moment it takes the slot, so a `uv sync` in that
          line is in the run's time, and a run without that record has no setup.
  run     the job's start (the slot's taking, or the command's start, where setup isn't recorded) -> the run's end.

The 50th, 90th and 95th percentiles (nearest rank) are given by kind of task (`run_records.kind`, from the command `job.json`
holds), by machine and by pool (the check slots the command asks for: a pool's by name, the machine's own as `check`, none as
`-`). A
refused launch (`refused.json`, which records no command: its kind and pool read `(not recorded)`), a run whose twin took the
slot and one that never joined a line are counted, not timed. Each run that waited over `--over` minutes is named with what it
waited behind, where a record says so: slot.json's `why` (for a record written before slot.py kept it past the slot's taking,
its headline in stderr.log) and, for a run cancelled in line, the cancel's reason.

A machine that doesn't answer, or whose answer was cut short, is named as unread: its runs are in no number here. Nothing
that is there and unreadable is read as absent: a record file that can't be read is named beside its run, whose row is made
without it; and a run with a time in its records that is no time is left out and named.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from . import run_records
from .queue_stats import pct, rid_time, ts

HOURS, OVER_MIN = 12.0, 10.0
#: the machines read by default: the registry's Nebius nodes
NODE = re.compile(r"vy-nebius-(cpu-)?\d+$")
PROJECT = "verity"
READ_TIMEOUT_S, READ_JOBS = 180, 16
#: a run no record gives a command for (a refused launch leaves `refused.json` alone): its kind and its pool
UNKNOWN = "(not recorded)"
KINDS = (*run_records.KINDS, UNKNOWN)
#: how a launch came out. The first four asked to start, and their wait is a wait for a start: `waiting` is still in line,
#: `left` is slot.py giving up at its --wait-max, `cancelled` ended in line (a cancel, the run's timeout). The rest are counted,
#: not timed: the machine refused the launch, the run's twin on another machine took the slot, it ended without joining a line
#: (`no slot`: the pool has no slots there), or it left no status
ASKED = ("started", "waiting", "left", "cancelled")
ELSE = ("twin", "no slot", "no record")
OUTCOMES = (*ASKED, "refused", *ELSE)
#: a machine's address in ssh's own message, which is printed under the machine's name instead
ADDRESS = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
#: what slot.py says of a slot it can't take (its `blocked` and `held_why`), each under the name it is counted by
CAUSES = (("slots busy", re.compile(r": busy\b")), ("memory short", re.compile(r"GB of memory free")),
          ("disk short", re.compile(r" has \d+ (inodes|GB) free, under ")), ("a window", re.compile(r"\bwindow .* is open or starts")),
          ("slots kept for candidates' checks", re.compile(r": for train checks\b")), ("slots under the memory asked", re.compile(r": under \d+ GB")))
W = 150


def clock() -> datetime:
    return datetime.now(UTC)


def slot_wait(slot: dict) -> float:
    """A check slot's wait in seconds (`slot.json`: took less since), 0 without one."""
    if not (isinstance(slot, dict) and slot.get("took") and slot.get("since")):
        return 0.0
    return max((ts(slot["took"]) - ts(slot["since"])).total_seconds(), 0.0)


def _t(s) -> datetime | None:
    """A record's time, None where it has none. One that is no ISO time raises ValueError."""
    return ts(s) if isinstance(s, str) else None


def _s(a: datetime | None, b: datetime | None) -> float | None:
    return max((b - a).total_seconds(), 0.0) if a and b else None


def cause(why: str | None) -> str:
    """What a wait's reason (slot.json's `why`) comes down to, for counting."""
    if not why:
        return "no record says"
    if "twin" in why:
        return "its twin took the slot"
    if "ahead in line" in why:
        return "others ahead in line"
    return " + ".join(name for name, said in CAUSES if said.search(why)) or "every slot taken"


def times(rec: dict, now: datetime) -> dict:
    """The run of record ``rec`` (`run_records.record`) as a row: what it is, when each thing happened, how it came out, its
    three times in seconds, what it waited behind and, cancelled in line, by what and why; and the record files that were
    there and couldn't be read (`unread`), which the row is made without. A run still in line has waited until ``now``.
    A time in the record that is no ISO time raises ValueError."""
    st, sl, rf = rec.get("status") or {}, rec.get("slot") or {}, rec.get("refused") or {}
    launched, started, ended = rid_time(rec["run"]), _t(st.get("started_utc")), _t(st.get("ended_utc"))
    since, took, job = _t(sl.get("since")), _t(sl.get("took")), _t(rec.get("job_started"))
    through, cancel = rec.get("pool", run_records.NO_POOL) != run_records.NO_POOL, rec.get("cancel") or {}
    behind = sl.get("why")
    if took or (started and not through and not sl):
        outcome, at = "started", took or started
    elif rf and not started:
        outcome, at, behind = "refused", _t(rf.get("at_utc")), f"{rf.get('fault')}: {rf.get('reason')}"
    elif sl.get("state") == "left":
        outcome, at = "twin" if "twin" in (behind or "") else "left", _t(sl.get("left")) or ended
    elif started and not ended and st.get("state") == "running":
        outcome, at = "waiting", now
    elif sl and ended:
        outcome, at = "cancelled", ended
    elif ended:
        outcome, at = "no slot", ended
    else:
        outcome, at = "no record", None
    begun = (job or at) if outcome == "started" else None
    return {"run": rec["run"], "machine": rec.get("machine"), "pool": rec.get("pool") or UNKNOWN, "kind": rec.get("kind") or UNKNOWN,
            "by": rec.get("by"), "outcome": outcome, "launched": f"{launched:%Y-%m-%dT%H:%M:%SZ}", "claimed": rec.get("claimed"),
            "joined": sl.get("since"), "took": sl.get("took"), "job_started": rec.get("job_started"),
            "ended": st.get("ended_utc"), "state": st.get("state"),
            "wait_s": _s(launched, at), "ship_s": _s(launched, _t(rec.get("claimed"))), "line_s": _s(since, at) if since else None,
            "setup_s": _s(took, job), "run_s": _s(begun, ended), "behind": behind,
            "cancel": f"{cancel.get('actor')}: {cancel['reason']}" if outcome == "cancelled" and cancel.get("reason") else None,
            "unread": rec.get("unread") or None}


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), **{f"p{round(q * 100)}_min": round(pct(vals, q) / 60, 1) for q in (0.5, 0.9, 0.95)}}


def _group(rows: list[dict], over_s: float) -> dict:
    waits = [r["wait_s"] for r in rows if r["outcome"] in ASKED and r["wait_s"] is not None]
    return {"runs": len(rows), **{o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
            "over": sum(w > over_s for w in waits), "wait": _stats(waits),
            "setup": _stats([r["setup_s"] for r in rows if r["setup_s"] is not None]),
            "run": _stats([r["run_s"] for r in rows if r["run_s"] is not None])}


def _order(by: str, keys: Iterable[str]) -> list[str]:
    """``keys`` in the order a table lists them: the runs' kinds as `KINDS` has them, anything else by name, cpu-2 before cpu-10."""
    if by == "kind" and set(keys) <= set(KINDS):
        return [k for k in KINDS if k in keys]
    num = lambda k: (re.sub(r"\d+$", "", k), int(m[0]) if (m := re.search(r"\d+$", k)) else 0)  # noqa: E731
    return sorted(keys, key=num)


def summary(rows: list[dict], over: float = OVER_MIN) -> dict:
    """{all, kind: {..}, machine: {..}, pool: {..}}: each group's count by outcome, its wait over the runs that asked to start
    (how many waited over ``over`` minutes among them), its setup where recorded and its run time, as percentiles in minutes;
    with the started runs' wait split in two (`ship`: launch to the machine's claim; `line`: joining the line to the slot), the
    refusals by fault and how long each took to be told, what the runs over ``over`` waited behind, and the runs with a record
    file that couldn't be read (`unread`: each one's files and why)."""
    over_s = over * 60
    out: dict = {"all": _group(rows, over_s)}
    for by in ("kind", "machine", "pool"):
        acc: dict[str, list[dict]] = {}
        for r in rows:
            acc.setdefault(str(r[by] or "?"), []).append(r)
        out[by] = {k: _group(acc[k], over_s) for k in _order(by, acc)}
    began = [r for r in rows if r["outcome"] == "started"]
    out["parts"] = {"ship": _stats([r["ship_s"] for r in began if r["ship_s"] is not None]),
                    "line": _stats([r["line_s"] for r in began if r["line_s"] is not None])}
    refused = [r for r in rows if r["outcome"] == "refused"]
    out["refused"] = {"faults": dict(Counter(str(r["behind"]).partition(":")[0] for r in refused).most_common()),
                      "told": _stats([r["wait_s"] for r in refused if r["wait_s"] is not None])}
    out["behind"] = dict(Counter(cause(r["behind"]) for r in slow(rows, over)).most_common())
    out["unread"] = {r["run"]: r["unread"] for r in rows if r.get("unread")}
    return out


def rows_of(recs: Iterable[dict], now: datetime) -> tuple[list[dict], dict[str, str]]:
    """Each record's row (`times`), by run id; and the runs left out of them, each with why: a time in its records is no time."""
    rows, out = [], {}
    for rec in recs:
        try:
            rows.append(times(rec, now))
        except ValueError as e:
            out[rec["run"]] = f"on {rec.get('machine')}, a time in its records is no time: {e}"
    return sorted(rows, key=lambda r: r["run"]), out


def slow(rows: list[dict], over: float = OVER_MIN) -> list[dict]:
    """The runs that asked to start and waited over ``over`` minutes, the longest first."""
    return sorted((r for r in rows if r["outcome"] in ASKED and (r["wait_s"] or 0) > over * 60), key=lambda r: (-r["wait_s"], r["run"]))


def nodes() -> list[str]:
    from . import remote

    return _order("machine", [n for n, m in remote.load_machines().items() if NODE.match(n) and m.get("provider") == "ssh"])


def _reach(name: str) -> tuple[list[str], str]:
    """The ssh argv that reaches machine ``name`` and its runs directory there."""
    from . import remote

    m = remote.resolve_machine(name, PROJECT)
    t = remote.pod_target(m)
    return remote.ssh_argv(t["host"], t["port"], user=t["user"]), f"{m['root']}/runs"


def _ssh(argv: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=data, capture_output=True, timeout=READ_TIMEOUT_S)


def parse(text: str) -> list[dict] | str:
    """`run_records.py`'s answer as its records, or why it isn't one: a line that isn't JSON, or no closing count that matches."""
    recs, read = [], None
    for line in text.splitlines():
        try:
            doc = json.loads(line)
        except ValueError:
            return f"its answer has a line that is not JSON: {line[:80]!r}"
        if isinstance(doc, dict) and "run" in doc:
            recs.append(doc)
        elif isinstance(doc, dict) and "read" in doc:
            read = doc["read"]
    return recs if read == len(recs) else f"its answer was cut short: {len(recs)} run(s) and no closing count that matches"


def read_nodes(names: list[str], first: str, last: str, *, reach: Callable | None = None, run: Callable | None = None) -> dict[str, list[dict] | str]:
    """Each machine's records of the runs with ids in [first, last], read at once, or why they couldn't be."""
    reach, run = reach or _reach, run or _ssh
    script = Path(run_records.__file__).read_bytes()

    def one(name: str) -> list[dict] | str:
        try:
            argv, runs = reach(name)
            r = run([*argv, f"python3 - {runs} {first} {last}"], script)
        except subprocess.TimeoutExpired:
            return f"no answer in {READ_TIMEOUT_S} s"
        except Exception as e:  # noqa: BLE001  (not listed, not reachable: named as unread, never read as no runs)
            return str(e) or type(e).__name__
        if r.returncode:
            err = r.stderr.decode(errors="replace").strip().splitlines()
            return f"exit {r.returncode}: {ADDRESS.sub(name, err[-1]) if err else 'no output'}"
        recs = parse(r.stdout.decode(errors="replace"))
        return recs if isinstance(recs, str) else [{**rec, "machine": name} for rec in recs]

    with ThreadPoolExecutor(max_workers=max(len(names), 1)) as ex:
        return dict(zip(names, ex.map(one, names)))


def _m(st: dict, q: int) -> str:
    v = st.get(f"p{q}_min")
    return "-" if v is None else f"{v:.1f}"


def _pcts(st: dict) -> str:
    return f"{_m(st, 50):>6} {_m(st, 90):>6} {_m(st, 95):>6}"


def render(s: dict, rows: list[dict], start: datetime, end: datetime, over: float, read: list[str], unread: dict[str, str],
           left_out: dict[str, str] | None = None) -> str:
    gt = f">{over:g}"
    a = s["all"]
    lines = [f"runs launched {start:%Y-%m-%d %H:%MZ} to {end:%Y-%m-%d %H:%MZ}: {a['runs']} on {len(read)} machine(s)"
             + (f"; UNREAD, in no number here: {'; '.join(f'{n} ({why})' for n, why in unread.items())}" if unread else "")
             + (f"; LEFT OUT, in no number here: {'; '.join(f'{run} ({why})' for run, why in left_out.items())}" if left_out else ""),
             "minutes, nearest-rank percentiles. w: the wait, launch -> check slot taken (no slot asked: -> command started); a run "
             "that got no slot counts with the time it waited (in line: still there; left: slot.py gave up at its --wait-max; "
             f"cancel: ended in line). {gt}: of those, waited over {over:g} min. s: setup, slot -> check.py started (n: the runs "
             "with that record). r: run, from there (else from the slot) to the end. refus: the machine refused the launch. else: "
             "its twin ran instead, it ended without joining a line, or it left no status",
             ""]
    head = (f"{'runs':>5} {'start':>5} {'in line':>7} {'left':>4} {'cancel':>6} {'refus':>5} {'else':>4}  {'w50':>6} {'w90':>6} {'w95':>6} "
            f"{gt:>5}  {'n':>4} {'s50':>6} {'s90':>6} {'s95':>6}  {'n':>5} {'r50':>6} {'r90':>6} {'r95':>6}")

    def line(name: str, g: dict) -> str:
        return (f"{name[:18]:<18} {g['runs']:>5} {g['started']:>5} {g['waiting']:>7} {g['left']:>4} {g['cancelled']:>6} {g['refused']:>5} "
                f"{sum(g[o] for o in ELSE):>4}  {_pcts(g['wait'])} {g['over']:>5}  {g['setup']['n']:>4} {_pcts(g['setup'])}  "
                f"{g['run']['n']:>5} {_pcts(g['run'])}")

    for by in ("kind", "machine", "pool"):
        lines += [f"{'by ' + by:<18} {head}", *(line(k, g) for k, g in s[by].items())]
        if by == "kind":
            lines.append(line("all", a))
        lines.append("")
    p, rf = s["parts"], s["refused"]
    lines += [f"the wait of the {a['started']} that started, in two parts: launch -> the machine's claim p50 {_m(p['ship'], 50)} p90 "
              f"{_m(p['ship'], 90)} p95 {_m(p['ship'], 95)}; joining the line -> slot p50 {_m(p['line'], 50)} p90 {_m(p['line'], 90)} "
              f"p95 {_m(p['line'], 95)} (n {p['line']['n']})",
              f"refused: {a['refused']}" + (f" ({', '.join(f'{k} {n}' for k, n in rf['faults'].items())}), told p50 {_m(rf['told'], 50)} "
                                            f"p95 {_m(rf['told'], 95)} min after launch" if a["refused"] else ""),
              *([f"{len(torn)} run(s) have a record file that is there and couldn't be read, and their rows are made without it: "
                 + "; ".join(f"{run} ({', '.join(f'{name}: {why}' for name, why in files.items())})" for run, files in torn.items())]
                if (torn := s["unread"]) else []),
              ""]
    late = slow(rows, over)
    lines += [f"runs that waited over {over:g} min: {len(late)} of the {sum(a[o] for o in ASKED)} that asked to start; behind: "
              + ("; ".join(f"{k} {n}" for k, n in s["behind"].items()) or "-"),
              f"{'run':<22} {'machine':<16} {'pool':<9} {'kind':<12} {'by':<18} {'wait':>6} {'outcome':<9} waited behind"]
    for r in late:
        lines.append(f"{r['run']:<22} {str(r['machine'])[:16]:<16} {r['pool'][:9]:<9} {r['kind']:<12} {str(r['by'] or '-')[:18]:<18} "
                     f"{r['wait_s'] / 60:>6.1f} {r['outcome']:<9} {str(r['behind'] or '-')[:W]}"
                     + (f"; cancelled ({r['cancel'][:W]})" if r["cancel"] else ""))
    return "\n".join(lines)


def main(argv: list[str], *, reach: Callable | None = None, run: Callable | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research run-times", description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=HOURS, help="the window's length (default %(default)g)")
    ap.add_argument("--until", default=None, metavar="T", help="the window's end, an ISO time (default now)")
    ap.add_argument("--over", type=float, default=OVER_MIN, metavar="MIN",
                    help="name each run that waited over this many minutes (default %(default)g)")
    ap.add_argument("--on", default=None, metavar="MACHINE[,MACHINE...]",
                    help="the machines to read (default: every Nebius node of the registry)")
    ap.add_argument("--runs", default=None, metavar="DIR", help="read this runs directory here instead of any machine")
    ap.add_argument("--json", action="store_true", help="the window, the summary, and every run's row, as one JSON object")
    a = ap.parse_args(argv)
    now = clock()
    try:
        end = ts(a.until) if a.until else now
    except ValueError:
        ap.error(f"--until {a.until!r}: not an ISO time")
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(hours=a.hours)
    first, last = f"r{start:%Y%m%d-%H%M%S}", f"r{end:%Y%m%d-%H%M%S}-ffff"
    if a.runs:
        got: dict[str, list[dict] | str] = {a.runs: [run_records.record(d) for d in run_records.window(Path(a.runs), first, last)]}
    else:
        got = read_nodes(a.on.split(",") if a.on else nodes(), first, last, reach=reach, run=run)
    read = [n for n, v in got.items() if not isinstance(v, str)]
    unread = {n: v for n, v in got.items() if isinstance(v, str)}
    rows, left_out = rows_of((rec for n in read for rec in got[n]), now)
    s = summary(rows, a.over)
    if a.json:
        print(json.dumps({"window": {"start": f"{start:%Y-%m-%dT%H:%M:%SZ}", "end": f"{end:%Y-%m-%dT%H:%M:%SZ}"}, "over_min": a.over,
                          "read": read, "unread": unread, "left_out": left_out, "summary": s, "runs": rows}, indent=1))
    else:
        print(render(s, rows, start, end, a.over, read, unread, left_out))
    if not read:
        print("research run-times: no machine was read", file=sys.stderr)
        return 1
    return 0
