"""`research queue stats`: where a landed PR's time went, from its ready mark to `main`, over a window of drive's landings.

Each PR drive landed in the window is split into five steps, read from drive's log (its tick times, the `checking TIP (#..): RID`
and `landed #.. on main at TIP (check RID)` lines), the landing check's own records (`slot.json`'s `took`, `steps.jsonl`'s step
ends) and the PR's newest `ready` label at the head that landed:

  queue     ready mark -> the first check of a train with the PR in it, at that head
  rework    that first check -> the check the PR landed with (trains thrown away by rebuilds, failures, ejections)
  machine   that check's launch -> its check slot
  check     its slot -> the last of its steps that ended before the landing (a deferred step, `--defer`, ends after it)
  land      that verdict -> the tick that landed it

`--per-pr` reads drive's per-PR shadow instead (`research.drive`, `--per-pr-shadow N`), from its records in drive's state file:
each PR it sampled in the window (its first check launched there), when its landing commit would have landed and how, and when
the trains landed it (`shadow_rows`).  Its `first check's slot -> would land` is the per-PR landing's own path from the moment a
check slot took its first check (`slot.json`'s `took`).  What comes before it isn't a per-PR drive's: `mark -> first launch` is the
shadow's queue for its N checks, and `first launch -> its slot` the wait of an ordinary job behind every priority check and in
turn with lane checks (the shadow never takes a priority slot, ci 21 Oct), where a per-PR drive's checks would have the trains'
priority."""
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


def _at(t: float | None) -> datetime | None:
    return datetime.fromtimestamp(t, timezone.utc) if t is not None else None


def shadow_rows(state: dict, start: datetime, end: datetime) -> list[dict]:
    """One row per PR the per-PR shadow sampled in [start, end] (`drive["per_pr"]`, ended or not): its times, what it ended in
    (`outcome`: would-land, failed, conflict or still going, and why its record ended), its checks and plans."""
    sh = (state.get("drive") or {}).get("per_pr") or {}
    out = []
    for rec in [*sh.get("done", []), *(sh.get("prs") or {}).values()]:
        sampled = _at(rec.get("sampled"))
        if sampled is None or not start <= sampled <= end:
            continue
        wl, ended = rec.get("would_land") or {}, rec.get("ended") or {}
        out.append({"pr": rec["pr"], "head": rec["head"], "outcome": rec["phase"], "ended": ended.get("why"),
                    "marked": _at(rec.get("marked")), "sampled": sampled, "took": _at(rec.get("took")),
                    "would_land": _at(wl.get("at")),
                    "full": _at(wl.get("full_at")), "via": wl.get("via"), "revert": wl.get("revert") or rec.get("revert"),
                    "landed": _at(ended.get("at")) if ended.get("why") == "landed" else None,
                    "checks": len(rec.get("checks") or []), "plans": [p.get("result") for p in rec.get("plans") or []],
                    "plan_whys": [p.get("why") for p in rec.get("plans") or [] if p.get("why")]})
    return out


SHADOW_SPANS = {"would_land": ("marked", "would_land", "ready mark -> would land"),
                "path": ("took", "would_land", "  first check's slot -> would land (drive's rule)"),
                "whole": ("took", "full", "  first check's slot -> whole check passed"),
                "queue": ("marked", "sampled", "  mark -> first launch (the shadow's own N checks)"),
                "slot": ("sampled", "took", "  first launch -> its slot (an ordinary job's wait)"),
                "trains": ("marked", "landed", "the same PRs: ready mark -> landed by the trains")}


def shadow_summary(rs: list[dict]) -> dict:
    out: dict = {"sampled": len(rs), "spans": {}}
    for k, (a, b, _) in SHADOW_SPANS.items():
        vals = [h for r in rs if (h := _hours(r[a], r[b])) is not None and r["would_land"] is not None]
        out["spans"][k] = {"n": len(vals), "median_h": statistics.median(vals), "p90_h": p90(vals)} if vals else {"n": 0}
    for k in ("would-land", "failed", "conflict", "queued", "checking", "planning"):
        out[k] = sum(r["outcome"] == k for r in rs)
    out["under_1h"] = sum(1 for r in rs if (h := _hours(r["took"], r["would_land"])) is not None and h < 1)
    out["rechecked"] = sum(r["checks"] > 1 for r in rs)
    out["would_revert"] = sum(bool(r["revert"]) for r in rs)
    plans = [x for r in rs for x in r["plans"]]
    out["plans"] = {x: plans.count(x) for x in ("passed", "refused", "error") if x in plans}
    out["plan_refusals_naming_lean_audit"] = sum("lean-audit" in w for r in rs for w in r["plan_whys"])
    return out


def shadow_render(s: dict, rs: list[dict], start: datetime, end: datetime) -> str:
    lines = [f"{start:%Y-%m-%d %H:%MZ} to {end:%Y-%m-%d %H:%MZ}: the per-PR shadow sampled {s['sampled']} PR(s): "
             f"{s['would-land']} would land, {s['failed']} failed, {s['conflict']} conflict with `main`, "
             f"{s['queued'] + s['checking'] + s['planning']} still going"]
    for k, (_, _, name) in SHADOW_SPANS.items():
        v = s["spans"][k]
        lines.append(f"{name:52} " + (f"median {v['median_h']:5.2f} h  p90 {v['p90_h']:5.2f} h  (n={v['n']})" if v["n"] else
                                      "no data"))
    lines.append(f"would land within 1 h of its first check's slot: {s['under_1h']}; checked more than once: {s['rechecked']}; a "
                 f"deferred step failed after it passed drive's rule (a revert): {s['would_revert']}")
    if s["plans"]:
        lines.append("plans on a moved `main`: " + ", ".join(f"{n} {k}" for k, n in s["plans"].items())
                     + f" ({s['plan_refusals_naming_lean_audit']} refusal(s) name a lean-audit part)")
    for r in sorted(rs, key=lambda r: r["sampled"]):
        h = _hours(r["took"], r["would_land"])
        lines.append(f"  #{r['pr']} {r['head'][:8]} {r['outcome']:10} " + (f"{h:5.2f} h " if h is not None else "      ")
                     + f"checks {r['checks']} plans {len(r['plans'])}" + (f" ended {r['ended']}" if r["ended"] else "")
                     + (f": {r['via']}" if r["via"] else ""))
    return "\n".join(lines)


def shadow_main(state_path: Path, hours: float, until: str | None, as_json: bool) -> int:
    end = ts(until) if until else datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError) as e:
        print(f"research queue stats --per-pr: {state_path}: {e}")
        return 1
    rs = shadow_rows(state, start, end)
    s = shadow_summary(rs)
    print(json.dumps({**s, "rows": rs}, indent=1, default=str) if as_json else shadow_render(s, rs, start, end))
    return 0


def main(log: Path, hours: float, until: str | None, marked: Callable[[int], datetime | None], run_dir: Callable[[str], Path],
         as_json: bool) -> int:
    end = ts(until) if until else datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    launches, landings = parse(log.read_text(errors="replace"))
    rs = rows(launches, landings, start, end, marked, run_dir)
    s = summary(rs)
    print(json.dumps(s, indent=1, default=str) if as_json else render(s, start, end))
    return 0
