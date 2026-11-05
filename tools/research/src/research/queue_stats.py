"""`research queue stats`: where a landed PR's time went, from its ready mark to `main`, over a window of drive's landings.

Each PR drive landed in the window is split into five steps, read from drive's log (its tick times, the `checking TIP (#..): RID`
and `landed #.. on main at TIP (check RID)` lines), the landing check's own records (`slot.json`'s `took`, `steps.jsonl`'s step
ends) and the PR's newest `ready` label at the head that landed:

  queue     ready mark -> the first check of a train with the PR in it, at that head
  rework    that first check -> the check the PR landed with (trains thrown away by rebuilds, failures, ejections)
  machine   that check's launch -> its check slot
  check     its slot -> the last of its steps that ended before the landing (a deferred step ends after it)
  land      that verdict -> the tick that landed it"""
from __future__ import annotations

import concurrent.futures
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

#: a tick's start, before its lines (drive's loop on an agent VM)
TICK_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$")
#: a tick's end, after its lines (drive.sh on the control pod, since 20 Oct): its landings take this time
TICK_END_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick rc=")
RID = r"r\d{8}-\d{6}-[0-9a-f]{4}"
CHECKING_RE = re.compile(rf"^research queue drive: checking ([0-9a-f]{{12}}) \(([^)]*)\): ({RID})")
LANDED_RE = re.compile(rf"^research queue drive: landed ((?:#\d+ ?)+) on `main` at ([0-9a-f]{{12}}) \(check ({RID})\)")
STEPS = ("queue", "rework", "machine", "check", "land")


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rid_time(rid: str) -> datetime:
    return datetime.strptime(rid[1:16], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


@dataclass
class Launch:
    at: datetime
    tip: str
    prs: frozenset[int]


@dataclass
class Landing:
    at: datetime
    prs: list[int]
    tip: str
    check: str


def parse(log: str) -> tuple[list[Launch], list[Landing]]:
    """Drive's launches and landings, each at its time: a launch at its run id's, a landing at its tick's (the start line before
    it, or on the control pod the end line after it; a landing whose tick hasn't ended yet isn't counted)."""
    tick, launches, landings, pending = None, [], [], []
    for line in log.splitlines():
        if m := TICK_RE.match(line):
            tick = ts(m[1])
        elif m := TICK_END_RE.match(line):
            landings += [Landing(ts(m[1]), *rest) for rest in pending]
            tick, pending = None, []
        elif m := CHECKING_RE.match(line):
            launches.append(Launch(rid_time(m[3]), m[1], frozenset(int(x) for x in re.findall(r"#(\d+)", m[2]))))
        elif m := LANDED_RE.match(line):
            rest = ([int(x) for x in re.findall(r"#(\d+)", m[1])], m[2], m[3])
            if tick:
                landings.append(Landing(tick, *rest))
            else:
                pending.append(rest)
    return launches, landings


def slot_took(run_dir: Path) -> datetime | None:
    try:
        took = json.loads((run_dir / "slot.json").read_text()).get("took")
    except (OSError, ValueError):
        return None
    return ts(took) if took else None


def verdict_at(run_dir: Path, landed: datetime) -> datetime | None:
    ends = []
    try:
        for line in (run_dir / "steps.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec.get("ended_utc") and (end := ts(rec["ended_utc"])) <= landed:
                ends.append(end)
    except (OSError, ValueError):
        return None
    return max(ends, default=None)


#: how many landings' check records and PRs' marks `rows` reads at once: each is a fetch over ssh or from the store (22 Oct:
#: one at a time, a day of 30 landings and 134 PRs took over 20 min)
READ_JOBS = 8


def _each(fn, items: list) -> dict:
    """{item: fn(item)} for each of ``items``, `READ_JOBS` at a time."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_JOBS) as ex:
        return dict(zip(items, ex.map(fn, items)))


def rows(launches: list[Launch], landings: list[Landing], start: datetime, end: datetime,
         marked: Callable[[int], datetime | None], run_dir: Callable[[str], Path]) -> list[dict]:
    """One row per PR landed in [start, end]: each step's start, and how many trains with the PR were checked."""
    out = []
    todo = [ld for ld in landings if start <= ld.at <= end]
    dirs = _each(run_dir, list(dict.fromkeys(ld.check for ld in todo)))
    marks = _each(marked, list(dict.fromkeys(pr for ld in todo for pr in ld.prs)))
    for ld in todo:
        d = dirs[ld.check]
        launch, took, verdict = rid_time(ld.check), slot_took(d), verdict_at(d, ld.at)
        for pr in ld.prs:
            mark = marks[pr]
            tried = [x for x in launches if pr in x.prs and (mark is None or x.at >= mark) and x.at <= launch]
            first = min((x.at for x in tried), default=launch)
            out.append({"pr": pr, "tip": ld.tip, "marked": mark, "first": first, "launch": launch, "took": took,
                        "verdict": verdict, "landed": ld.at, "trains": len({x.tip for x in tried}) or 1})
    return out


def _hours(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() / 3600 if a and b else None


def spans(row: dict) -> dict[str, float | None]:
    """The row's total and its five steps, in hours."""
    return {"total": _hours(row["marked"], row["landed"]), "queue": _hours(row["marked"], row["first"]),
            "rework": _hours(row["first"], row["launch"]), "machine": _hours(row["launch"], row["took"]),
            "check": _hours(row["took"], row["verdict"]), "land": _hours(row["verdict"], row["landed"])}


def p90(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(0.9 * (len(vals) - 1)))]


def summary(rs: list[dict]) -> dict:
    out: dict = {"prs": len(rs), "landings": len({r["tip"] for r in rs}), "steps": {}}
    all_spans = [spans(r) for r in rs]
    for k in ("total", *STEPS):
        vals = [s[k] for s in all_spans if s[k] is not None]
        out["steps"][k] = ({"n": len(vals), "median_h": statistics.median(vals), "p90_h": p90(vals), "pr_h": sum(vals)}
                           if vals else {"n": 0})
    trains = [r["trains"] for r in rs]
    out["trains_median"] = statistics.median(trains) if trains else None
    out["rechecked"] = sum(t > 1 for t in trains)
    return out


def render(s: dict, start: datetime, end: datetime) -> str:
    names = {"total": "ready mark -> landed", "queue": "  queue: mark -> first check with it",
             "rework": "  rework: first check -> the one it landed with", "machine": "  machine: that check's launch -> slot",
             "check": "  check: slot -> verdict", "land": "  land: verdict -> landed"}
    lines = [f"{start:%Y-%m-%d %H:%MZ} to {end:%Y-%m-%d %H:%MZ}: {s['prs']} PR(s) landed by drive in {s['landings']} landing(s)"]
    for k, name in names.items():
        v = s["steps"][k]
        lines.append(f"{name:48} " + (f"median {v['median_h']:5.2f} h  p90 {v['p90_h']:5.2f} h  {v['pr_h']:7.1f} PR-h  (n={v['n']})"
                                      if v["n"] else "no data"))
    if s["prs"]:
        lines.append(f"trains checked with each landed PR: median {s['trains_median']:g}; {s['rechecked']} PR(s) in more than one")
    return "\n".join(lines)


def main(log: Path, hours: float, until: str | None, marked: Callable[[int], datetime | None], run_dir: Callable[[str], Path],
         as_json: bool) -> int:
    end = ts(until) if until else datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    launches, landings = parse(log.read_text(errors="replace"))
    rs = rows(launches, landings, start, end, marked, run_dir)
    s = summary(rs)
    print(json.dumps(s, indent=1, default=str) if as_json else render(s, start, end))
    return 0
