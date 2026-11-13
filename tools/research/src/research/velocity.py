"""`research velocity`: how fast the plan moves, read from records. A reading, never a gate: it exits 0, and a row it can't
read says `not read` and what it needs, never a guess.

  research velocity [--since TIME] [--until TIME] [--from DUMP] [--to DUMP] [--baseline DUMP] [--agents FILE.json|N]
                    [--no-save] [--line | --json]

The window is the WINDOW_H hours to now; --since and --until, times with their offsets, give another. One table, each row
with its source:

  GOALS     two dumps of the Goals page (`research goals show --json`) compared, over the top-priority goals set by hand
            (priority 1, given by a person), every goal of priority 1, and every goal, as the later dump has them: finished
            (open in the earlier dump, done in the later), moved earlier and later (how many, and their days in total; a
            day with no hour is its end, Pacific), the net (days earlier less days later), open and off track now, and the
            stalls: a goal with a step past its hour and not updated since, or with no step due within STEP_H hours
            (`stalls`). Two columns: the window, and since the baseline.
  LANDING   the share of marked changes on `main` within two hours of the mark, and the mark-to-`main` times with their
            median, over the window: the table of `research queue stats`, its long window started at the reading's start.
            It gathers the hours to the window's end from the queue's loop, the store and GitHub, which takes over a
            minute, and a window longer than its own is not read.
  AGENTS    the agents running against LIMIT, by lane, from --agents FILE.json, the saved output of the cursor-cloud
            `list-cloud-agents` tool: a worker's name starts with its issue key, whose team's lane it is, a coordinator's
            is its handle, and the rest are `unnamed`. --agents N is their count alone, printed with no lanes: the
            tool gives it as `totalMatched` when asked for one agent at an offset past the last.
  MACHINES  the dollars spent and the idle dollars in the window, from the fleet's samples (pods/nebius/fleet.py's report).
  ASKS      people's asks off track now (`research msg linear audit --asks`).

A DUMP is a file or an `art:` id. The later one is the page as it is now, which the command puts in the store (a
`goals-dump/v1`, preserved) and names, so the next reading compares with a preserved copy; --to gives another, and with
--until, a past time, it is the newest known dump at or before then. The earlier one is the newest known dump at or before
--since (--from gives another, and the window then starts when it was taken). A known dump is one this machine's store
holds, one the hourly sweep saved on the control pod (infra/control/top_watch.py), or one `velocity.json` names; that file
names the baseline too. A dump taken up to SLACK_S after a time stands for it. --no-save puts nothing in the store.
--line prints the reading as one line for a chat, --json all of it.
"""
from __future__ import annotations

import argparse
import functools
import gzip
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import linear
from .goals import PACIFIC
from .goals_audit import _waits, due_at

WINDOW_H = 6
STEP_H = 24                 # a goal's next step is due within this
GRACE_H = 1                 # and a step is dead this long past its hour: a reading on the hour meets steps due that hour
SLACK_S = 900               # readings drift by minutes: a dump taken this long after a time still stands for it
LIMIT = 100                 # the agents that may run at once
KIND = "goals-dump/v1"
CONTROL = "vy-control-verity"
SWEEP_GOALS = "/data/watchers/goals"        # where the hourly sweep saves the page on the control pod
POD_FLEET = "/data/watchers/fleet"          # and where the pod keeps the fleet's samples
DATA = Path(__file__).with_name("velocity.json")
STAMP = "%Y%m%dT%H%MZ"
DUMP_NAME = re.compile(r"goals-(\d{8}T\d{4}Z)\.json(\.gz)?$")
ISSUE_KEY = re.compile(r"([A-Z][A-Z0-9]*)-\d+\b")
UNNAMED = "unnamed"
SCOPES = {"hand": "top priority, set by hand", "p1": "priority 1", "all": "every goal"}
SHARE, MEDIAN = "Share on `main` within 2 h of the mark", "Mark to `main`"      # the rows of `research queue stats` read


class Unread(Exception):
    """Why a row is not read, and what it needs."""


def when(text: str) -> datetime:
    """A time written with its offset (2026-11-11T12:00-08:00), or a dump's stamp (20261111T2002Z)."""
    if re.fullmatch(r"\d{8}T\d{4}Z", text):
        return datetime.strptime(text, STAMP).replace(tzinfo=UTC)
    t = datetime.fromisoformat(text)
    if t.tzinfo is None:
        raise ValueError(f"{text}: give its offset, as in 2026-11-11T12:00-08:00")
    return t


def pst(t: datetime) -> str:
    return t.astimezone(PACIFIC).strftime("%a %-d %b %-I:%M %p %Z")


def span(a: datetime, b: datetime) -> str:
    """`a` to `b` in Pacific time, the day said once when both are on it."""
    a, b = a.astimezone(PACIFIC), b.astimezone(PACIFIC)
    return f"{a:%a %-d %b %-I:%M %p} to {b:%-I:%M %p %Z}" if a.date() == b.date() else f"{a:%a %-d %b %-I:%M %p} to {pst(b)}"


def row(read: Callable[[], dict]) -> dict:
    """What `read` gives, or `{"not_read": why}`: a row that can't be read says so, and the others are still read."""
    try:
        return read()
    except Unread as e:
        return {"not_read": str(e)}
    except Exception as e:  # noqa: BLE001 (a reading never fails: the row says why it isn't read)
        return {"not_read": f"{type(e).__name__}: {e}"}


# ---- goals: pure functions of two dumps -------------------------------------------------------------------------------------

def in_scope(g: dict, scope: str) -> bool:
    return scope == "all" or g.get("priority") == 1 and (scope == "p1" or bool(g.get("prioritySet")))


def standing(goals: list[dict], scope: str) -> dict:
    """The scope's goals in one dump: how many, how many open, and how many of those off track."""
    mine = [g for g in goals if in_scope(g, scope)]
    return {"goals": len(mine), "open": sum(g["status"] != "done" for g in mine),
            "off_track": sum(g["status"] == "off-track" for g in mine)}


def moved(before: list[dict], after: list[dict], scope: str) -> dict:
    """What changed from one dump to a later one, over the goals the later dump has in the scope and the earlier had open:
    how many finished, and how many moved earlier and later with their days in total (a goal finished since counts with its
    move). Beside them, not in the score: the scope's goals done now that the earlier dump didn't hold (`new_done`), and
    the goals the earlier dump had open in the scope that the page no longer holds (`removed`)."""
    was, here = {g["id"]: g for g in before}, {g["id"] for g in after}
    mine = [g for g in after if in_scope(g, scope)]
    held = [g for g in mine if g["id"] in was and was[g["id"]]["status"] != "done"]
    days = [(due_at(g["due"]) - due_at(was[g["id"]]["due"])).total_seconds() / 86400 for g in held]
    earlier, later = round(-sum(d for d in days if d < 0), 1), round(sum(d for d in days if d > 0), 1)
    return {"finished": sum(g["status"] == "done" for g in held),
            "earlier": sum(d < 0 for d in days), "earlier_days": earlier, "later": sum(d > 0 for d in days), "later_days": later,
            "net_days": round(earlier - later, 1),
            "new_done": sum(g["status"] == "done" for g in mine if g["id"] not in was),
            "removed": sum(g["status"] != "done" and in_scope(g, scope) and g["id"] not in here for g in before)}


def stalls(goals: list[dict], now: datetime) -> list[dict]:
    """The open top-priority goals set by hand that are stalled at `now`, each with why. A goal's steps are itself and
    every open goal it waits on, through any number of links. A step is dead when it is GRACE_H hours past its hour and
    hasn't been updated since that hour. The goal is stalled when a step is dead, or when no step is due within STEP_H
    hours."""
    by = {g["id"]: g for g in goals}

    def dead(s: dict) -> bool:
        due = due_at(s["due"])
        return due + timedelta(hours=GRACE_H) < now and (not s.get("updatedAt") or when(s["updatedAt"]) <= due)
    out = []
    for g in goals:
        if not in_scope(g, "hand") or g["status"] == "done":
            continue
        steps, todo = {g["id"]: g}, _waits(g)
        while todo:
            k = todo.pop()
            if k in by and k not in steps:
                steps[k] = by[k]
                todo += _waits(by[k])
        open_ = [s for s in steps.values() if s["status"] != "done"]
        late = sorted(s["id"] for s in open_ if dead(s))
        if late:
            out.append({"goal": g["id"], "why": "past its hour and not updated since", "steps": late})
        elif not any(due_at(s["due"]) <= now + timedelta(hours=STEP_H) for s in open_):
            out.append({"goal": g["id"], "why": f"no step due within {STEP_H} hours", "steps": []})
    return out


# ---- dumps ------------------------------------------------------------------------------------------------------------------

def read_page() -> list[dict]:
    from .goals import Route

    return Route().call()


def open_store():
    from .store.local import LocalStore

    return LocalStore()


@functools.cache
def control_ssh() -> tuple[str, ...]:
    """The ssh command line to the control pod, as `research pods ssh` builds it: asked for once a reading."""
    from . import remote
    from .pods import connect

    try:
        return tuple(connect.ssh_base(CONTROL))
    except remote.Refused as e:
        raise Unread(f"{CONTROL} isn't reached from here ({' '.join(str(e).split())})") from None


def pod(cmd: str, timeout: float = 120) -> str:
    """What the shell command `cmd` prints on the control pod; run here when this machine is the pod."""
    argv = ["sh", "-c", cmd] if Path(SWEEP_GOALS).parent.is_dir() else [*control_ssh(), cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Unread(f"{CONTROL} gave no answer ({type(e).__name__})") from None
    if p.returncode != 0:
        raise Unread(f"{CONTROL} isn't reached from here ({' '.join(p.stderr.split())[-160:] or f'exit {p.returncode}'})")
    return p.stdout


def parse(raw: bytes | str, what: str) -> list[dict]:
    goals = json.loads(raw)
    if not isinstance(goals, list):
        raise Unread(f"{what}: expected a JSON list of goals, as `research goals show --json` prints")
    return goals


def load(spec: str) -> dict:
    """The dump `spec` names, `{goals, taken, source}`: an `art:` id, a file (`.gz` or not; taken at its name's stamp, else
    at its newest update), or `sweep:NAME`, one the hourly sweep saved."""
    if spec.startswith("art:"):
        st = open_store()
        return {"goals": parse(st.fetch(spec).read_bytes(), spec), "taken": when(st.get_manifest(spec).meta["taken"]),
                "source": spec}
    if spec.startswith("sweep:"):
        name = spec.removeprefix("sweep:")
        if not (named := DUMP_NAME.fullmatch(name)):
            raise Unread(f"{spec}: the sweep's dumps are named goals-YYYYMMDDTHHMMZ.json.gz")
        return {"goals": parse(pod(f"zcat {SWEEP_GOALS}/{name}"), spec), "taken": when(named[1]),
                "source": f"{CONTROL}:{SWEEP_GOALS}/{name}"}
    path = Path(spec)
    goals = parse(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes(), spec)
    named = DUMP_NAME.search(path.name)
    taken = when(named[1]) if named else max((when(g["updatedAt"]) for g in goals if g.get("updatedAt")), default=None)
    if taken is None:
        raise Unread(f"{spec}: when it was taken isn't known: name it goals-YYYYMMDDTHHMMZ.json")
    return {"goals": goals, "taken": taken, "source": spec}


def known_dumps(notes: list[str]) -> dict[datetime, str]:
    """Each known dump by when it was taken: the ones `velocity.json` names, the ones this machine's store holds, and the
    ones the hourly sweep saved. A source that can't be read is named in `notes`."""
    out = {when(taken): art for art, taken in json.loads(DATA.read_text())["dumps"].items()}
    for what, read in (("this machine's store", lambda: {when(a["manifest"]["meta"]["taken"]): a["id"]
                                                         for a in open_store().index.select_artifacts(kind=KIND)}),
                       ("the sweep's dumps", lambda: {when(m[1]): f"sweep:{m[0]}" for name in
                                                      pod(f"ls {SWEEP_GOALS} 2>/dev/null; true").split()
                                                      if (m := DUMP_NAME.fullmatch(name))})):
        got = row(read)
        if "not_read" in got:
            notes.append(f"{what} not read: {got['not_read']}")
        else:
            out.update(got)
    return out


def save(goals: list[dict], taken: datetime) -> tuple[str | None, str]:
    """The dump put in the store and pushed to its remote: its `art:` id, and why it isn't preserved there ('' when it is)."""
    from .store.local import StoreError

    stamp = f"{taken.astimezone(UTC):{STAMP}}"
    try:
        st = open_store()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / f"goals-{stamp}.json"
            f.write_text(json.dumps(goals, indent=1, ensure_ascii=False) + "\n")
            art = st.put_artifact(KIND, {"taken": stamp, "what": "research goals show --json", "by": "research velocity"}, file=f)
    except (StoreError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        rep = st.push(art, preserve=True)
    except (StoreError, OSError) as e:
        return art, f"{type(e).__name__}: {e}"
    return art, "" if rep["preserved"] else "; ".join(rep.get("errors") or ["the remote doesn't hold it"])


# ---- the other rows ---------------------------------------------------------------------------------------------------------

def landing(since: datetime, until: datetime, live: bool) -> dict:
    """The share on `main` within two hours of the mark and the mark-to-`main` times over the window, as the table of
    `research queue stats` gives them for its long window: its document is gathered to `until` (`live`: nothing after it
    is cut, for it is now) and its long window started at `since`. The document holds queue_table.WINDOW_H hours, so a
    longer window is not read, and neither is one whose document lacks the queue's loop."""
    from . import queue_table

    hours = (until - since).total_seconds() / 3600
    if hours > queue_table.WINDOW_H:
        raise Unread(f"`research queue stats` gathers the {queue_table.WINDOW_H} hours to a window's end, and this window "
                     f"is {hours:.1f} h: --since gives a shorter one")
    doc = queue_table.collect(**{"now" if live else "until": until.timestamp()})
    if not doc.get("control"):
        raise Unread("`research queue stats` didn't read the queue's loop (" + "; ".join(doc["errors"]) + ")")
    cells = {r["name"]: r["day"] for r in queue_table.table(doc | {"day": f"{since.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}"})}
    return {"share": cells[SHARE], "mark_to_main": cells[MEDIAN],       # a node's run directories are no part of these rows
            "unread": [e for e in doc["errors"] if e.split(":")[0] not in queue_table.CPU_NODES]}


def agents(text: str, reg: dict) -> dict:
    """The agents running, by lane, from the saved output of the cursor-cloud `list-cloud-agents` tool (a line of prose,
    then its JSON). A name that starts with an issue key is its team's lane's, one that is a handle (`verity: HANDLE`
    too) is that lane's, and any other is UNNAMED."""
    try:
        doc = json.loads(text[text.index("{"):])
        listed = doc["agents"]
    except (ValueError, KeyError, TypeError):
        raise Unread("--agents: not the saved output of the cursor-cloud `list-cloud-agents` tool (its JSON has `agents`)") from None
    owners, handles = linear.owners(reg), {h["handle"] for h in reg.get("handles", [])}
    by: Counter = Counter()
    for a in listed:
        if a.get("status") != "RUNNING":
            continue
        name = str(a.get("name") or "")
        key, plain = ISSUE_KEY.match(name), name.removeprefix("verity: ").strip()
        by[owners[key[1]] if key and key[1] in owners else plain if plain in handles else UNNAMED] += 1
    return {"running": sum(by.values()), "limit": LIMIT, "lanes": dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))),
            "more": bool(doc.get("hasMore"))}


def machines(since: datetime, until: datetime) -> dict:
    """The dollars spent and the idle dollars in the window, by fleet.py's report over the control pod's samples."""
    from .pods.nebius import fleet

    hours = (until - since).total_seconds() / 3600
    days = [since.astimezone(UTC).date() + timedelta(days=n) for n in range((until.astimezone(UTC).date() - since.astimezone(UTC).date()).days + 1)]
    try:
        text = pod("cat " + " ".join(f"{POD_FLEET}/samples-{d:%Y%m%d}.jsonl" for d in days) + " 2>/dev/null; true")
    except Unread as e:
        raise Unread(f"{e}; on the pod: `FLEET_DIR={POD_FLEET} python3.12 /data/watchers/src/tools/research/src/research/pods/"
                     f"nebius/fleet.py report --hours {hours:g} --now {until.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}`") from None
    rows = [r for r in map(json.loads, filter(str.strip, text.splitlines())) if since <= when(r["t"]) < until]
    if not rows:
        raise Unread(f"{CONTROL}:{POD_FLEET} holds no sample of the window")
    rep = fleet.report(rows, hours)
    usd, idle = fleet.totals(rep)
    return {"usd": round(usd), "idle_usd": round(idle), "vms": len(rep),
            "no_rate": sorted(r["name"] for r in rep if r["rate_usd_per_hour"] is None)}


def read_asks(now: float) -> list[dict]:
    from . import slack

    return linear.Linear().asks(linear.owners(slack.load_registry(slack.registry_path())), None, now)


def asks(rows: list[dict], now: float) -> dict:
    """People's asks not delivered, and those of them off track (`linear.ask_findings`), by kind."""
    found = linear.ask_findings(rows, now)
    return {"open": sum(map(linear.ask_open, rows)), "off_track": len({f["issue"] for f in found}),
            "kinds": {k: n for k in linear.ASK_KINDS if (n := sum(f["kind"] == k for f in found))},
            "issues": sorted({f["issue"] for f in found})}


# ---- the reading ------------------------------------------------------------------------------------------------------------

def read(ns: argparse.Namespace, since: datetime | None, until: datetime | None, now: datetime) -> dict:
    notes: list[str] = []
    known: dict[datetime, str] | None = None

    def pick(t: datetime) -> str:
        nonlocal known
        if known is None:
            known = known_dumps(notes)
        at = max((k for k in known if k <= t + timedelta(seconds=SLACK_S)), default=None)
        if at is None:
            raise Unread(f"no dump of the Goals page is known at or before {pst(t)}: name one with --from or --to")
        return known[at]

    def page() -> dict:
        d = {"goals": read_page(), "taken": now, "source": "the Goals page"}
        if not ns.no_save:
            d["art"], d["unpreserved"] = save(d["goals"], now)
        return d

    later = row(lambda: load(ns.later) if ns.later else load(pick(until)) if until else page())
    end = until or (later.get("taken") if ns.later else None) or now
    earlier = row(lambda: load(ns.earlier or pick(since or end - timedelta(hours=WINDOW_H))))
    start = since or (earlier.get("taken") if ns.earlier else None) or end - timedelta(hours=WINDOW_H)
    if "taken" in earlier and (ahead := (start - earlier["taken"]).total_seconds()) > SLACK_S:
        notes.append(f"the goals' first column starts {ahead / 3600:.1f} h before the window, at its dump: none nearer "
                     f"{pst(start)} is known")
    base = row(lambda: load(ns.baseline or json.loads(DATA.read_text())["baseline"]))

    def change(before: dict, scope: str) -> dict:
        if "not_read" in before:
            return before
        if before["taken"] >= later["taken"]:
            return {"not_read": f"its dump ({pst(before['taken'])}) is no older than the later one"}
        return moved(before["goals"], later["goals"], scope)
    out = {"since": start, "until": end, "hours": round((end - start).total_seconds() / 3600, 1),
           "dumps": {k: {x: v for x, v in d.items() if x != "goals"} for k, d in (("earlier", earlier), ("later", later), ("baseline", base))}}
    if "not_read" in later:
        out["goals"] = out["stalls"] = {"not_read": later["not_read"]}
    else:
        out["goals"] = {s: {"now": standing(later["goals"], s), "window": change(earlier, s), "baseline": change(base, s)} for s in SCOPES}
        out["stalls"] = row(lambda: {"at": later["taken"], "goals": stalls(later["goals"], later["taken"])})
    live = not until and not ns.later

    def agents_row() -> dict:
        from . import slack

        if not ns.agents:
            raise Unread("give --agents FILE.json, the saved output of the cursor-cloud `list-cloud-agents` tool with statuses "
                         "RUNNING, or --agents N, their count (the agent watch's status file counts only the agents "
                         "registered with it)")
        if ns.agents.isdigit():
            return {"running": int(ns.agents), "limit": LIMIT, "lanes": {}, "more": False, "source": "the count --agents gave"}
        return agents(Path(ns.agents).read_text(), slack.load_registry(slack.registry_path(), people=False)) | {"source": ns.agents}

    def asks_row() -> dict:
        if not live:
            raise Unread("Linear gives people's asks as they stand now, and this window ends in the past; "
                         "`research msg linear audit --asks` reads them now")
        return asks(read_asks(now.timestamp()), now.timestamp())
    out |= {"landing": row(lambda: landing(start, end, live)), "agents": row(agents_row),
            "machines": row(lambda: machines(start, end)), "asks": row(asks_row), "notes": notes}
    return out


def _saved(d: dict) -> str:
    """Where the page read now was saved, when it was to be: its `art:` id, and why it isn't preserved when it isn't."""
    if "art" not in d:
        return ""
    return (f"saved as {d['art']}" if d["art"] else "not saved") + (f" (NOT preserved: {d['unpreserved']})" if d["unpreserved"] else "")


def _dump(d: dict) -> str:
    return f"not read: {d['not_read']}" if "not_read" in d else f"{d['source']} ({pst(d['taken'])})" + (f", {_saved(d)}" if _saved(d) else "")


def _cells(m: dict) -> list[str]:
    if "not_read" in m:
        return ["not read"] * 4
    return [f"{m['finished']}" + (f" (and {m['new_done']} new and done)" if m["new_done"] else ""),
            f"{m['earlier']} ({m['earlier_days']:.1f} d)", f"{m['later']} ({m['later_days']:.1f} d)", f"{m['net_days']:+.1f}"]


def _rest(r: dict) -> list[tuple[str, str]]:
    """The rows below the goals, each `(name, what it read and from where)`."""
    def cell(key: str, text: Callable[[dict], str]) -> str:
        return f"not read: {r[key]['not_read']}" if "not_read" in r[key] else text(r[key])
    hours = f"the {r['hours']:g} h to {pst(r['until'])}"
    return [
        ("LANDING", cell("landing", lambda x: f"within 2 h of the mark, {x['share']}; mark to `main` {x['mark_to_main']} "
                                              f"(`research queue stats`, {hours})"
                                              + (f"; it didn't read: {'; '.join(x['unread'])}" if x["unread"] else ""))),
        ("AGENTS", cell("agents", lambda x: f"{x['running']}{' or more' if x['more'] else ''} running of {x['limit']}"
                                            + (": " + ", ".join(f"{k} {n}" for k, n in x["lanes"].items()) if x["lanes"] else "")
                                            + f" ({x['source']})")),
        ("MACHINES", cell("machines", lambda x: f"${x['usd']:,} spent, ${x['idle_usd']:,} of it on idle hours, {x['vms']} VMs, {hours}"
                                                + (f"; no rate registered for {', '.join(x['no_rate'])}" if x["no_rate"] else "")
                                                + f" (the fleet's samples on {CONTROL})")),
        ("ASKS", cell("asks", lambda x: f"{x['off_track']} off track of {x['open']} not delivered"
                                        + (": " + ", ".join(f"{k} {n}" for k, n in x["kinds"].items()) + f" ({', '.join(x['issues'])})"
                                           if x["issues"] else "") + " (Linear, now)")),
    ]


def render(r: dict) -> str:
    lines = [f"research velocity: {span(r['since'], r['until'])} ({r['hours']:g} h). A reading, not a gate.",
             *(f"  {k + ' dump':<14} {_dump(d)}" for k, d in r["dumps"].items()), ""]
    if "not_read" in r["goals"]:
        lines.append(f"{'GOALS':<10}not read: {r['goals']['not_read']}")
    else:
        first, second = (f"since {pst(d['taken'])}" if "taken" in d else "not read" for d in (r["dumps"]["earlier"], r["dumps"]["baseline"]))
        lines.append(f"{'GOALS':<36}{first:<32}{second} (the baseline)")
        lines += [f"  {name} not read: {m['not_read']}" for name, m in (("the window", r["goals"]["hand"]["window"]),
                                                                       ("since the baseline", r["goals"]["hand"]["baseline"]))
                  if "not_read" in m]
        for scope, title in SCOPES.items():
            g = r["goals"][scope]
            lines.append(f"{title}: {g['now']['goals']} goals, {g['now']['open']} open, {g['now']['off_track']} of them off track")
            lines += [f"  {name:<34}{a:<32}{b}" for name, a, b in zip(
                ("finished", "moved earlier", "moved later", "net days (earlier less later)"), _cells(g["window"]), _cells(g["baseline"]))]
            if scope == "hand":
                s = r["stalls"]
                dead = sum(bool(x["steps"]) for x in s.get("goals", []))
                lines.append(f"  {'stalled now':<34}" + (f"not read: {s['not_read']}" if "not_read" in s else (
                    f"{len(s['goals'])} of {g['now']['open']} open: {dead} with a step past its hour and not updated since, "
                    f"{len(s['goals']) - dead} with no step due within {STEP_H} hours")))
                lines += [f"    {x['goal']}: " + (f"{', '.join(x['steps'])} {x['why']}" if x["steps"] else x["why"])
                          for x in s.get("goals", [])]
    lines += ["", *(f"{name:<10}{text}" for name, text in _rest(r)), *(f"note: {n}" for n in r["notes"])]
    return "\n".join(lines)


def line(r: dict) -> str:
    """The reading as one line for a chat: the score of the top-priority goals set by hand, then the other rows in short."""
    def score(m: dict) -> str:
        return "not read" if "not_read" in m else (f"{m['finished']} finished, {m['earlier_days']:.1f} days earlier less "
                                                   f"{m['later_days']:.1f} later (net {m['net_days']:+.1f})")

    def short(key: str, text: Callable[[dict], str]) -> str:
        return f"{key} not read" if "not_read" in r[key] else text(r[key])
    if "not_read" in r["goals"]:
        goals = f"Goals not read ({r['goals']['not_read']})"
    else:
        g, s, d = r["goals"]["hand"], r["stalls"], r["dumps"]
        goals = (f"Top-priority goals set by hand, since {pst(d['earlier']['taken']) if 'taken' in d['earlier'] else 'the earlier dump'}: "
                 f"{score(g['window'])}; {g['now']['off_track']} of {g['now']['open']} open off track; "
                 f"{'stalls not read' if 'not_read' in s else str(len(s['goals'])) + ' stalled'}"
                 + ("" if "not_read" in g["baseline"] else f". Since the baseline ({pst(d['baseline']['taken'])}): {score(g['baseline'])}"))
    rest = [short("landing", lambda x: f"landing within 2 h of the mark: {x['share'].split(';')[0]}"),
            short("agents", lambda x: f"{x['running']}{' or more' if x['more'] else ''} of {x['limit']} agents running"),
            short("machines", lambda x: f"machines ${x['usd']:,}, ${x['idle_usd']:,} of it idle"),
            short("asks", lambda x: f"{x['off_track']} of {x['open']} asks off track")]
    return f"Velocity, {span(r['since'], r['until'])}. {goals}. {'; '.join(rest).capitalize()}."


def main(argv: list[str], now: datetime | None = None) -> int:
    p = argparse.ArgumentParser(prog="research velocity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", metavar="TIME", help=f"the window's start, with its offset (default: {WINDOW_H} hours before its end)")
    p.add_argument("--until", "--now", dest="until", metavar="TIME", help="the window's end, a past time with its offset (default: now)")
    p.add_argument("--from", dest="earlier", metavar="DUMP", help="the earlier dump of the Goals page: a file or an art: id")
    p.add_argument("--to", dest="later", metavar="DUMP", help="the later dump (default: the page now, or the newest known at --until)")
    p.add_argument("--baseline", metavar="DUMP", help="the dump the second column starts from (default: velocity.json's)")
    p.add_argument("--agents", metavar="FILE.json|N", help="the saved output of the cursor-cloud `list-cloud-agents` tool, "
                                                           "or the count of agents running")
    p.add_argument("--no-save", action="store_true", help="don't put the page read now in the store")
    form = p.add_mutually_exclusive_group()
    form.add_argument("--line", action="store_true", help="one line for a chat")
    form.add_argument("--json", action="store_true")
    ns = p.parse_args(argv)
    now = now or datetime.now(UTC)
    try:
        since, until = (when(t) if t else None for t in (ns.since, ns.until))
        if until and until > now:
            raise ValueError(f"--until {ns.until}: it is not in the past; without --until the window ends now")
        if since and since >= (until or now):
            raise ValueError(f"--since {ns.since}: it is not before the window's end")
    except ValueError as e:
        print(f"research velocity: {e}", file=sys.stderr)
        return 2
    r = read(ns, since, until, now)
    print(json.dumps(r, indent=1, ensure_ascii=False, default=str) if ns.json else line(r) if ns.line else render(r))
    if ns.line and _saved(r["dumps"]["later"]):
        print(f"research velocity: the page read is {_saved(r['dumps']['later'])}", file=sys.stderr)
    return 0
