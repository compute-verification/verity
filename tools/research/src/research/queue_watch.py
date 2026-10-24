"""`research queue watch`: landing's watchdog, which pages `@ci` the moment the landing SLO is breached (top, 24 Oct 05:30Z, after
`main` stood still 01:38Z-05:15Z with 70+ ready PRs and nothing alerted).  It runs beside drive, never inside it: a wedged drive, or
one 35 min into a tick, still gets paged.  The control pod's watchers supervisor runs it every 5 min (infra/control/watchers.toml,
`queue-watch`).

    research queue watch [--every S | --once] [--log FILE] [--drive-state FILE] [--drive-runs DIR] [--repo DIR] [--state FILE]
                         [--hours H] [--as HANDLE] [--dry-run]

The SLO: ready mark to landed p90 at most 2 h, and `main` moving at least hourly while ready PRs wait.  Each pass checks:

  main      ready PRs wait (members of `next` above `main`, else the count drive's last sync logged) and `main` hasn't moved for
            MAIN_STALL_S: since the later of its head's commit time and drive's landing of it (its tick's end in drive's log), or
            since the last pass that saw the head before it
  p90       the p90 of ready mark to landed over the PRs drive landed in the last --hours (6) and the PRs waiting on `next` now
            (counting a waiting PR's age), over P90_SLO_S; marks are the store's `ready` labels at the head that landed or waits
  tick      no drive tick has ended (drive.sh's `tick rc=` line) for TICK_STALE_S, naming a stopped drive
  stalled   a train check or its touched suites' run (drive's state, `drive.launched`) holds a check slot and has written
            nothing in its run dir for STALLED_S: stopped (n2's Quiet SIGSTOPs the workload and its sampler), frozen, or its machine gone; the newest write is
            the newest mtime `research status RUN --on MACHINE` fetched (its sampler's heartbeat, sampler.json, every <= 60 s)
  unstarted a train check has waited unstarted for UNSTARTED_S since drive launched it while a machine in drive's current `on`
            has a check slot with no holder and no window blocking it, on two passes running (slot.py's `--status --json`)

A breach is paged as one post in the merge thread, as --as (default infra: drive posts as ci, and a post never rings its own
sender's inbox), to `@ci`, which rings ci's inbox and so wakes it (the agent watch's doorbell rule).  Each pass's new breaches,
repeats (at most every REPEAT_S while one lasts) and resolved lines (for a breach that was paged) go out as one post; a post that
fails is retried next pass.  A reader that fails is a breach of its own once it has failed for UNREAD_S.  The state file keeps
what is open, when each key was last paged, `main`'s head as last seen and the marks read.

Every reader is injected (`Readers`), and every check takes `now`: the tests need no clock.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import queue_stats as qs

MAIN_STALL_S = 3600
P90_SLO_S = 7200
TICK_STALE_S = 1800
STALLED_S = 600
UNSTARTED_S = 1800
#: a free slot is a snapshot: a train waiting while one frees takes it within slot.py's poll (20 s), so the breach must hold this
#: long (two passes) before it pages
UNSTARTED_HOLD_S = 240
REPEAT_S = 1800
UNREAD_S = 1800
WINDOW_H = 6.0
EVERY_S = 300
#: how much of drive's log a pass reads: about a day of ticks
LOG_TAIL = 16 << 20
#: the run states in which a check may still hold or take a slot (research.runs.ACTIVE)
ACTIVE = ("submitted", "admitted", "waiting", "running", "preserving")
#: a run's read (`research status`, an ssh fetch and a memory reading) gives up after this; its last fetched records stand
RUN_READ_S = 120
GIT_S = 180
#: a machine's check slots file (slot.py's SLOTS)
SLOTS_FILE = "/workspace/research/locks/slots"
SLOT_PY = "tools/verity/check/slot.py"
#: drive's flags line, once a tick (`research queue drive: flags from ...: on=a,b max_running=...`)
FLAGS_RE = re.compile(r"^research queue drive: flags from .*?: on=(\S+)", re.M)
#: its sync's line: how many PRs `next` holds above `main`
NEXT_RE = re.compile(r"^research queue: `next` at [0-9a-f]+: (\d+) PR\(s\) above `main`", re.M)
#: drive.sh's line while /data/drive/stopped holds the loop
STOPPED_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) stopped: ", re.M)
TICK_RC_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick rc=(\d+)", re.M)


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t))


def hm(t: float) -> str:
    return time.strftime("%H:%MZ", time.gmtime(t))


def _ts(s: str) -> float:
    return qs.ts(s).timestamp()


@dataclass
class Breach:
    key: str
    title: str
    detail: str = ""
    #: paged only once the key has been found for this long
    hold_s: float = 0.0


@dataclass
class Repo:
    """What the watch's clone says, fetched this pass."""
    main: str
    main_time: float                                    # `main`'s head's commit time
    waiting: list = field(default_factory=list)         # `queue.Member`s on `next` above `main`, oldest first
    heads: dict[int, str] = field(default_factory=dict)  # each PR's head as it landed (`main`'s first-parent trailers)


@dataclass
class Run:
    """One check run as `research status` read it."""
    state: str
    slot: dict
    machine: str | None
    wrote: float | None                                 # the newest write in its run dir
    note: str = ""                                      # its memory and slot lines, or why it wasn't read


@dataclass
class Readers:
    log: Callable[[], tuple[str, float | None]]         # drive's log (its tail) and its mtime
    drive: Callable[[], dict]                           # drive's state (the queue's state.json)
    repo: Callable[[], Repo]
    marks: Callable[[list[tuple[int, str]]], dict[tuple[int, str], float]]   # each (PR, head)'s ready mark, when it has one
    runs: Callable[[list[str]], dict[str, Run | None]]
    slots: Callable[[list[str]], dict[str, list[dict] | str]]               # each machine's check slots, or why unread


# ---- the checks ----------------------------------------------------------------------------------------------------------------------

def main_moved(repo: Repo, landings: list[qs.Landing], seen: dict, now: float) -> dict:
    """`main`'s head and when it moved there (module doc), as the state keeps it."""
    if seen.get("sha") == repo.main:
        return {**seen, "checked": now}
    at = max([repo.main_time, *(ld.at.timestamp() for ld in landings if repo.main.startswith(ld.tip))])
    if seen.get("checked") is not None:
        at = max(at, seen["checked"])                   # it moved after the last pass that saw the head before it
    return {"sha": repo.main, "moved": min(at, now), "checked": now}


def main_stalled(main: dict, waiting: int, now: float) -> Breach | None:
    if waiting <= 0 or now - main["moved"] < MAIN_STALL_S:
        return None
    return Breach("main", f"`main` hasn't moved for {(now - main['moved']) / 60:.0f} min while {waiting} ready PR(s) wait on `next`",
                  f"at {main['sha'][:12]} since {iso(main['moved'])}")


def slow_landings(now: float, hours: float, landings: list[qs.Landing], repo: Repo,
                  marks: Callable[[list[tuple[int, str]]], dict[tuple[int, str], float]]) -> Breach | None:
    """The p90 of ready mark to landed over the PRs landed in the window and those waiting now, when it is over the SLO."""
    landed = [(pr, repo.heads[pr], ld.at.timestamp()) for ld in landings if now - ld.at.timestamp() <= hours * 3600
              for pr in ld.prs if pr in repo.heads]
    waiting = [(m.pr, m.head, None) for m in repo.waiting]
    got = marks([(pr, head) for pr, head, _ in landed + waiting])
    spans = [(pr, (end or now) - got[(pr, head)], end is None) for pr, head, end in landed + waiting if (pr, head) in got]
    if not spans:
        return None
    p = qs.p90([s for _, s, _ in spans])
    if p <= P90_SLO_S:
        return None
    worst = sorted(spans, key=lambda x: -x[1])[:3]
    nw = sum(w for *_, w in spans)
    return Breach("p90", f"ready-to-landed p90 is {p / 3600:.1f} h, over the {P90_SLO_S / 3600:g} h SLO",
                  f"over {len(spans)} PR(s): {len(spans) - nw} landed in the last {hours:g} h, {nw} waiting; the longest "
                  + ", ".join(f"#{pr} {s / 3600:.1f} h{' waiting' if w else ''}" for pr, s, w in worst))


def tick_stale(log: str, mtime: float | None, now: float) -> Breach | None:
    ends = TICK_RC_RE.findall(log)
    last = _ts(ends[-1][0]) if ends else None
    if last is not None and now - last < TICK_STALE_S:
        return None
    stopped = [s for s in STOPPED_RE.findall(log) if last is None or _ts(s) > last]
    said = (f"the last ended at {iso(last)} (rc={ends[-1][1]})" if last is not None else "drive's log holds no tick's end")
    return Breach("tick", (f"no drive tick has ended for {(now - last) / 60:.0f} min" if last is not None else
                           "no drive tick has ended in the tail of drive's log")
                  + (": drive is stopped (/data/drive/stopped)" if stopped else ""),
                  said + (f"; stopped since {iso(_ts(stopped[0]))}" if stopped else "")
                  + (f"; its log was last written at {iso(mtime)}" if mtime is not None else ""))


def trains_of(log: str) -> dict[str, frozenset[int]]:
    """Each launched train's PRs, by its commit's first 12 hex digits (drive's `checking` lines)."""
    return {x.tip: x.prs for x in qs.parse(log)[0]}


def _train(c: str, prs: dict[str, frozenset[int]]) -> str:
    got = prs.get(c[:12])
    return f"train {c[:12]}" + (f" ({len(got)} PR(s), up to #{max(got)})" if got else "")


def stalled_checks(launched: dict, runs: dict[str, Run | None], prs: dict[str, frozenset[int]], now: float) -> list[Breach]:
    out = []
    for c, rec in launched.items():
        for rid in [*(rec.get("ids") or []), *(rec.get("touched") or [])]:
            r = runs.get(rid)
            if r is None or r.state not in ACTIVE or (r.slot or {}).get("state") != "held":
                continue
            if r.wrote is not None and now - r.wrote < STALLED_S:
                continue
            quiet = f"for {(now - r.wrote) / 60:.0f} min (since {iso(r.wrote)})" if r.wrote is not None else "that this watch has seen"
            out.append(Breach(f"stalled:{rid}", f"{_train(c, prs)}'s check {rid} holds {r.machine or '?'}'s slot "
                                                f"{r.slot.get('slot', '?')} and has written nothing in its run dir {quiet}",
                              f"slot taken {r.slot.get('took', '?')}" + (f"; {r.note}" if r.note else "")))
    return out


def unstarted(launched: dict, runs: dict[str, Run | None], now: float) -> dict[str, list[Run]]:
    """The launched trains whose check has waited UNSTARTED_S with no twin in a slot: commit -> its live twins."""
    out = {}
    for c, rec in launched.items():
        live = [r for rid in rec.get("ids") or [] if (r := runs.get(rid)) is not None and r.state in ACTIVE]
        if live and now - rec.get("at", now) >= UNSTARTED_S and not any((r.slot or {}).get("state") == "held" for r in live):
            out[c] = live
    return out


def free_slots(docs: dict[str, list[dict] | str]) -> list[tuple[str, str]]:
    """(machine, slot) of each check slot no check holds and no window blocks: a priority train may take any of them."""
    return [(m, s.get("name", "?")) for m, doc in docs.items() if isinstance(doc, list) for s in doc
            if s.get("holder") is None and not s.get("blocked")]


def unstarted_checks(waits: dict[str, list[Run]], launched: dict, free: list[tuple[str, str]], prs: dict[str, frozenset[int]],
                     now: float) -> list[Breach]:
    if not free:
        return []
    out = []
    for c, live in waits.items():
        theirs = sorted({r.machine for r in live if r.machine})
        out.append(Breach(f"unstarted:{c}", f"{_train(c, prs)}'s check has waited {(now - launched[c]['at']) / 60:.0f} min unstarted "
                                            "while a check slot is free",
                          f"its twins wait on {', '.join(theirs) or '?'}; free now: "
                          + ", ".join(f"{m} {s}" + ("" if m in theirs else " (not one of its machines)") for m, s in free),
                          hold_s=UNSTARTED_HOLD_S))
    return out


def watch(readers: Readers, state: dict, now: float, hours: float = WINDOW_H) -> list[Breach]:
    """One pass: every breach found now.  A reader that raises is the breach `unread:NAME`, and only the checks it feeds are
    skipped."""
    out: list[Breach] = []

    def read(name: str, fn: Callable):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (one reader failing leaves the other checks)
            out.append(Breach(f"unread:{name}", f"the landing watch can't read {name}", f"{type(e).__name__}: {str(e)[-300:]}",
                              hold_s=UNREAD_S))
            return None

    got = read("drive's log", readers.log)
    log, mtime = got if got is not None else (None, None)
    landings = qs.parse(log)[1] if log is not None else []
    repo = read("the repository (main and next)", readers.repo)
    if repo is not None:
        state["main"] = main_moved(repo, landings, state.get("main") or {}, now)
        nexts = NEXT_RE.findall(log or "")
        if b := main_stalled(state["main"], len(repo.waiting) or (int(nexts[-1]) if nexts else 0), now):
            out.append(b)
        if log is not None and (b := read("the ready marks", lambda: slow_landings(now, hours, landings, repo, readers.marks))):
            out.append(b)
    if log is not None and (b := tick_stale(log, mtime, now)):
        out.append(b)
    drive = read("drive's state", readers.drive)
    launched = ((drive or {}).get("drive") or {}).get("launched") or {}
    if launched:
        prs = trains_of(log or "")
        runs = read("the train checks' runs", lambda: readers.runs([rid for rec in launched.values()
                                                                    for rid in [*(rec.get("ids") or []), *(rec.get("touched") or [])]]))
        if runs is not None:
            out += stalled_checks(launched, runs, prs, now)
            if waits := unstarted(launched, runs, now):
                on = FLAGS_RE.findall(log or "")
                machines = on[-1].split(",") if on else sorted({r.machine for live in waits.values() for r in live if r.machine})
                docs = read("the check slots", lambda: readers.slots(machines))
                if docs is not None:
                    out += unstarted_checks(waits, launched, free_slots(docs), prs, now)
    return out


# ---- paging -------------------------------------------------------------------------------------------------------------------------

HEADER = (f"landing watch (SLO: ready-to-landed p90 <= {P90_SLO_S / 3600:g} h, and `main` moving at least hourly while ready PRs "
          "wait):")


def page(state: dict, found: list[Breach], now: float, post: Callable[[str], bool]) -> str | None:
    """Post this pass's new and repeated breaches and the paged ones that resolved, as one post; the text posted, else None.
    A post that fails changes nothing, so the next pass sends it again."""
    opened, last = state.setdefault("open", {}), state.setdefault("paged", {})
    now_keys = {b.key: b for b in found}
    lines, paged, gone = [], [], {}
    for k, b in now_keys.items():
        rec = opened.setdefault(k, {"since": now})
        rec["title"] = b.title
        if now - rec["since"] < b.hold_s or now - last.get(k, float("-inf")) < REPEAT_S:
            continue
        lines.append(f"- {b.title}" + (f": {b.detail}" if b.detail else "")
                     + (f" (still, since {hm(rec['since'])})" if rec.get("paged") else ""))
        paged.append(k)
    for k in [k for k in opened if k not in now_keys]:
        rec = gone[k] = opened.pop(k)
        if rec.get("paged"):
            lines.append(f"- resolved after {(now - rec['since']) / 60:.0f} min: {rec['title']}")
    if not lines:
        return None
    text = "\n".join([HEADER, *lines])
    if not post(text):
        opened.update({k: r for k, r in gone.items() if r.get("paged")})
        return None
    for k in paged:
        opened[k]["paged"] = last[k] = now
    state["paged"] = {k: t for k, t in last.items() if now - t < 86400}
    return text


# ---- the readers on the control pod --------------------------------------------------------------------------------------------------

def read_log(path: Path) -> tuple[str, float | None]:
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - LOG_TAIL))
        text = f.read().decode(errors="replace")
    return (text.split("\n", 1)[-1] if size > LOG_TAIL else text), path.stat().st_mtime


def _git(repo: Path, *args: str) -> str:
    from .gates import git_env

    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=git_env(GIT_TERMINAL_PROMPT="0"), timeout=GIT_S)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def read_repo(path: Path) -> Repo:
    """Fetch `main` and `next` into the watch's own clone (made once with the deploy key: watchers.toml) and read them."""
    from .queue import TRAILER_RE, chain_of

    _git(path, "fetch", "-q", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main",
         "+refs/heads/next:refs/remotes/origin/next")
    main, t = _git(path, "log", "-1", "--format=%H %ct", "refs/remotes/origin/main").split()
    heads: dict[int, str] = {}
    for body in _git(path, "log", "--first-parent", "--format=%B%x1e", "-n", "3000", main).split("\x1e"):
        tr = dict(TRAILER_RE.findall(body))
        if "PR" in tr and "Head" in tr:
            heads.setdefault(int(tr["PR"].lstrip("#")), tr["Head"])
    return Repo(main, float(t), chain_of(path, "refs/remotes/origin/main", "refs/remotes/origin/next"), heads)


def read_marks(cache: dict[str, float], targets: list[tuple[int, str]]) -> dict[tuple[int, str], float]:
    """Each target's ready mark: the newest `ready` label at that head (as `research queue stats` reads it), pulled from the store's
    remote once and kept in ``cache`` (the state's `marks`)."""
    from .queue import StoreLabels
    from .store.local import LocalStore

    if new := [(pr, h) for pr, h in targets if f"{pr}@{h}" not in cache]:
        labels = StoreLabels(LocalStore(None), log=lambda s: None)
        labels.prefetch(new)
        for pr, h in new:
            labels.ready(pr, h)
            if at := labels.marked_at(pr, h):
                cache[f"{pr}@{h}"] = _ts(at)
    for k in set(cache) - {f"{pr}@{h}" for pr, h in targets}:
        del cache[k]
    return {(pr, h): cache[f"{pr}@{h}"] for pr, h in targets if f"{pr}@{h}" in cache}


def read_run(rid: str, drive_runs: Path, mine: Path) -> Run | None:
    """``rid`` read with `research status RID --on MACHINE` into this watch's own runs dir (a copy of drive's remote.json), so
    its fetches never write drive's run dirs; None for a run drive's runs dir doesn't hold."""
    from . import runs as R

    d = mine / rid
    if not (d / "remote.json").is_file():
        try:
            rec = json.loads((drive_runs / rid / "remote.json").read_text())
        except (OSError, ValueError):
            return None
        if not rec.get("remote_run_dir"):
            return None
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(drive_runs / rid / "remote.json", d / "remote.json")
    machine = (json.loads((d / "remote.json").read_text()) or {}).get("machine")
    note, row = "", None
    try:
        r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--runs-dir", str(mine), *(["--on", machine] if machine else []),
                            "--json", "--also", "sampler.json", "--also", "steps.jsonl"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=RUN_READ_S)
        row = json.loads(r.stdout)["runs"][0]
    except subprocess.TimeoutExpired:
        note = f"its machine didn't answer in {RUN_READ_S} s"
    except (ValueError, KeyError, IndexError):
        note = f"`research status` exited {r.returncode}: {(r.stderr or '').strip()[-200:]}"
    row = row or R.summarize(d)
    rr = row.get("remote_rec") or {}
    if err := (rr.get("last_fetch_error") or {}).get("stderr"):
        note = note or f"its last fetch failed: {err.strip()[-200:]}"
    if gone := (rr.get("runner_gone") or {}).get("why"):
        note = "; ".join(filter(None, [note, f"its runner is gone: {gone}"]))
    held = [t for k, t in row.get("holds") or [] if k in ("mem", "slot")]
    wrote = max((p.stat().st_mtime for p in d.iterdir() if p.is_file() and p.name not in ("remote.json", ".fetched")), default=None)
    return Run(row.get("state") or "unknown", row.get("slot") or {}, machine, wrote, "; ".join([*held, *([note] if note else [])])[:400])


def read_runs(ids: list[str], drive_runs: Path, mine: Path) -> dict[str, Run | None]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=qs.READ_JOBS) as ex:
        return dict(zip(ids, ex.map(lambda rid: read_run(rid, drive_runs, mine), ids)))


def read_slots(machines: list[str], repo: Path) -> dict[str, list[dict] | str]:
    """Each machine's check slots (`slot.py --status --json`'s `slots`, [] with no slots file), as `research run --on auto` reads a
    pool (research.auto), with `main`'s slot.py; or why it wasn't read."""
    from . import auto

    prog = (f"SHARE = None\nSLOT_SRC = {_git(repo, 'show', f'refs/remotes/origin/main:{SLOT_PY}')!r}\n"
            f"POOLS = {{'check': {SLOTS_FILE!r}}}\n" + auto.PROBE).encode()

    def one(m: str) -> list[dict] | str:
        try:
            r = auto._ssh(auto._reach(m), prog)
        except subprocess.TimeoutExpired:
            return f"no answer in {auto.READ_TIMEOUT_S} s"
        except Exception as e:  # noqa: BLE001  (an unknown machine is only unread)
            return str(e) or type(e).__name__
        try:
            doc = json.loads(r.stdout)["check"]
        except (ValueError, KeyError, TypeError):
            return f"exit {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}"
        return doc.get("error") or doc.get("slots") or []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(machines))) as ex:
        return dict(zip(machines, ex.map(one, machines)))


def post_page(sender: str) -> Callable[[str], bool]:
    def post(text: str) -> bool:
        from .drive import CI, merge_thread_post

        return merge_thread_post(text, (CI,), sender=sender)
    return post


def main(argv: list[str], now: Callable[[], float] = time.time) -> int:
    from . import runs as R

    home = Path.home() / ".research"
    ap = argparse.ArgumentParser(prog="research queue watch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--every", type=float, default=EVERY_S, help=f"seconds between passes (default {EVERY_S})")
    ap.add_argument("--once", action="store_true", help="one pass; exit 1 when a breach is open")
    ap.add_argument("--log", type=Path, default=Path("/data/drive/drive.log"), help="drive's log (default /data/drive/drive.log)")
    ap.add_argument("--drive-state", type=Path, default=home / "queue" / "verity.state.json", help="drive's state file")
    ap.add_argument("--drive-runs", type=Path, default=None, help="the runs dir drive launches from (default research's runs root)")
    ap.add_argument("--repo", type=Path, default=home / "queue-watch" / "repo", help="this watch's clone, fetched each pass")
    ap.add_argument("--state", type=Path, default=home / "queue-watch" / "state.json", help="this watch's state; its runs dir beside it")
    ap.add_argument("--hours", type=float, default=WINDOW_H, help=f"p90's window of landings (default {WINDOW_H:g})")
    ap.add_argument("--as", dest="sender", default="infra", help="the handle it posts as (default infra; never ci, whose own posts "
                                                                 "don't ring its inbox)")
    ap.add_argument("--dry-run", action="store_true", help="print the page instead of posting it, and save no state")
    a = ap.parse_args(argv)
    if a.sender == "ci":
        ap.error("--as ci: a post never rings its sender's own inbox, so ci would never be paged")
    drive_runs = a.drive_runs or R.runs_root()
    mine = a.state.parent / "runs"
    post = (lambda text: print(text, flush=True) or True) if a.dry_run else post_page(a.sender)
    while True:
        t = now()
        try:
            state = json.loads(a.state.read_text()) if a.state.is_file() else {}
        except (OSError, ValueError) as e:
            print(f"{iso(t)} queue watch: {a.state} unreadable ({e}); starting from nothing", flush=True)
            state = {}
        marks = state.setdefault("marks", {})
        readers = Readers(log=lambda: read_log(a.log), drive=lambda: json.loads(a.drive_state.read_text()),
                          repo=lambda: read_repo(a.repo), marks=lambda ts: read_marks(marks, ts),
                          runs=lambda ids: read_runs(ids, drive_runs, mine), slots=lambda ms: read_slots(ms, a.repo))
        found = watch(readers, state, t, a.hours)
        sent = page(state, found, t, post)
        print(f"{iso(t)} queue watch: " + ("; ".join(b.key for b in found) or "no breach")
              + (" (paged)" if sent else ""), flush=True)
        if not a.dry_run:
            a.state.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.state.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
            tmp.replace(a.state)
        if a.once:
            return 1 if found else 0
        time.sleep(max(1.0, a.every - (now() - t)))
