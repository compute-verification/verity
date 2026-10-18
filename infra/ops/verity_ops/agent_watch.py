"""The agent watch (INF-69): free a cloud agent whose run is held, and record what it sees, from the control pod.

    python3 agent_watch.py loop --log /data/agent-watch/events.jsonl     $CURSOR_API_KEY set; standard library only

An agent hears Slack, Linear and timer events only between runs, and a run can hang after a tool call returns: proofs from
05:29 to 15:33Z and circuits from 09:49Z on 18 Oct, three coordinators at once about 15:26Z on 15 Oct. Until now only Daniel
pressing Stop freed one. Every `POLL_S` the watch lists the team's agents (`GET /v1/agents`; ACTIVE ones, and IDLE ones changed
in the last `IDLE_RECENT_S`, since an ERROR leaves an agent IDLE). For each ACTIVE agent it lists its runs (`GET .../runs`),
page by page back to the newest that started: its `latestRunId` is often a run cancelled on arrival, created and cancelled
within half a second while the run doing the work is older (memory-accounting, 19:32Z 18 Oct; 52 such runs in 50 min by 20:40Z). It follows the event stream (`GET .../runs/{run}/stream`,
server-sent events) of the agent's newest run that is RUNNING or WAITING_FOR_BACKGROUND_WORK, until that run ends or another
takes its place. Events carry no time, so it times them as they arrive. A heartbeat counts as nothing, and so does the `status`
the API re-sends on every reconnection. Each simple event (`assistant`, `thinking`, `tool_call`) has an `interaction_update`
twin, which counts as activity but never as what the agent did last. A cloud subagent's `task` call stays `running` while the
subagent works and the parent goes on (the parent thinks or starts a step while it is open), so such a call counts as
background work, and its progress (`tool-call-delta`) as the subagent's, not the parent's; so does a `task` call, or in a
background wait any call, finishing whose start the watch didn't see (it lay before the stream's replay). A run is

- **held** when its last event is a finished tool call, or the end of a turn while it still reads RUNNING, with no tool call
  of its own open, and the parent has streamed nothing since for `HELD_S` (RUNNING) or `HELD_BG_S` (WAITING_FOR_BACKGROUND_WORK,
  counting its subagents' progress, and longer because cancelling it may take its background workers with it; a wait after its
  turn ended is the normal state, never held). The watch cancels it (`POST .../runs/{run}/cancel`) and starts a new run on the
  agent (`POST /v1/agents/{id}/runs`) that says from when it was held and to read its inbox. A refused new run (409 agent_busy)
  is tried again every sweep for `CONTINUE_RETRY_S`, then left for a person. A cancel refused with 409 is not asked again;
  a run still read as going `REFUSED_S` later is recorded for a person. It cancels at most once per agent every
  `RECANCEL_S`, so a run that hangs again is left for a person;
- **in a long call** when a tool call the parent waits on has been running for `LONG_CALL_S` (recorded, never cancelled: a
  build or a wait may be legitimate);
- **crashed** when the agent's latest run ended in ERROR: the watch starts a new run, at most once an hour and `MAX_RESTARTS`
  times in a row (vllm's runs errored within a minute, run after run, on 18 Oct), then records it as **erroring**, for a person.
  Only a latest run that FINISHED ends the row: a restarted run reads RUNNING for a while before it errors again.

A run among an agent's newest left in CREATING for 10 min by the API's own `updatedAt` with no newer run started (only ones
cancelled on arrival) and no run followed (a run queued behind a followed one waits its turn, and that one is judged), or a followed background wait with nothing at all streamed for 2 h, is recorded once as **stuck**: on 18 Oct proofs' and circuits' runs sat in CREATING behind a
held one while every later one was cancelled on arrival, network-accounting's behind a goal run from 18:27Z, and the API refuses
to cancel a CREATING run. A goal run (`run-goal-...`) can't be read or streamed on its own (400), so it is never followed; an
agent whose latest run the API won't read is recorded once as unreadable.

A coordinator holds a handle of `--registry` (the Slack registry): the holder of the handle's open inbox in `--routes` (the notes'
`routes.d`, which a handover moves first; on 18 Oct the registry still named vllm's predecessor), else the first agent id in the
handle's `holder`. It runs a 15-minute duty timer, and its inbox thread rings it for each doorbell. It is judged by the run the
sweep judges it by (the run followed, else its newest that started: a doorbell mid-turn leaves a run cancelled on arrival as its
`latestRunId`, vllm at 20:23Z 18 Oct). One whose run's turn is
over (FINISHED, EXPIRED, or WAITING_FOR_BACKGROUND_WORK, where coordinators mostly sit between wakes) is **ready to wake** when
nothing has run for `WAKE_S` since its last activity (its timer didn't fire), or when a doorbell for its handle rang after that
and `DOORBELL_S` has passed with nothing run (its subscription didn't). Its last activity is the later of the run's `updatedAt`
(read each sweep) and its last stream event: whether a wake starts a new run or resumes a background-waiting one, it moves. The
doorbells come from `doorbells.json` beside `--log` (`{"t", "inboxes": {handle: {"agent", "last"}}}`, each inbox's latest
doorbell as a Slack ts), which infra's relay writes, since only it reads Slack; one older than `BELLS_STALE_S` is not read. The
watch records it at most once per agent an hour, kept in `state.json`, and with `--wake` starts a run that says so; a wake
refused with 409 agent_busy is recorded and counts as the hour's (zombie CREATING runs kept memory-accounting busy for hours).
`--wake` stays off by default.

Each finding is a line of JSON in `--log`; each sweep ends by replacing `status.json` beside it atomically, in the schema
`verity/agent-watch-status/v1` (`snapshot`: its time `t`, `poll_s`, and per agent read its id, the run the watch judges,
statuses, watch state, since and restart count, in categories only), so a reader can tell a quiet watch from a dead one; a
sweep that fails leaves the last one. `state.json` beside it keeps what must outlive a restart (the watch restarts on every
deploy): when it last cancelled or restarted each agent, and recorded it ready to wake, the restart counts, the continue runs still to start and the runs
already recorded. `watch.lock` beside it keeps a second watch from running. The control pod holds no Slack credential, so
infra's relay reads the findings over ssh and posts each in #agent-alerts. Cancelling and restarting don't wait for it.
`--once --dry-run` runs one sweep that follows no stream and sends no POST (each is printed).
"""
from __future__ import annotations

import argparse
import base64
import calendar
import fcntl
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Iterator

API = "https://api.cursor.com"
POLL_S = 60
HELD_S = 15 * 60
HELD_BG_S = 60 * 60         # held while WAITING_FOR_BACKGROUND_WORK: cancelling it may take its local background workers with it
LONG_CALL_S = 45 * 60
RECANCEL_S = 3600
CONTINUE_RETRY_S = 30 * 60  # how long a refused continue run is tried again before it is left for a person
REFUSED_S = 5 * 60          # a run whose cancel got 409 still read as going this long after is left for a person
MAX_RESTARTS = 3            # ERROR runs in a row the watch restarts before leaving the agent for a person
IDLE_RECENT_S = 6 * 3600    # an IDLE agent is read only if it changed this recently (an ERROR leaves an agent IDLE)
RUNS_PAGE = 50              # an ACTIVE agent's runs are read a page at a time, back to the newest that started
RUNS_PAGES = 10             # (memory-accounting had 52 runs cancelled on arrival in 50 min, 18 Oct), at most this many pages
FOLLOW = {"RUNNING": HELD_S, "WAITING_FOR_BACKGROUND_WORK": HELD_BG_S}
TERMINAL = {"FINISHED", "ERROR", "CANCELLED", "EXPIRED"}
# recorded once, never cancelled: a run CREATING this long by its `updatedAt`, a followed background wait with nothing streamed
STUCK = {"CREATING": 10 * 60, "WAITING_FOR_BACKGROUND_WORK": 2 * 3600}
GOAL = "run-goal-"              # a goal run's id: the runs endpoint and its stream take only run-<uuid> (400)
SUBAGENT = {"tool-call-delta"}          # interaction_update types that report a subagent's progress, not the parent's
CONTINUE = ("continue: your run was held from {since} (the agent watch saw nothing for {mins} min after {after}, and "
            "cancelled it). Read your inbox and carry on from where you stopped.")
AFTER = {"tool_done": "a finished tool call", "turn_ended": "the end of your turn"}
AFTER_ERROR = ("continue: your last run ended in ERROR (the agent watch restarted you). Read your inbox and carry on from where "
               "you stopped.")
WAKE_S = 30 * 60            # twice a coordinator's 15-minute duty timer
DOORBELL_S = 10 * 60
BELLS_STALE_S = 30 * 60
# a run whose turn is over; coordinators mostly sit in background work between wakes (12 of 14 at 19:38Z, 18 Oct). CANCELLED is a
# person's stop or the watch's own; ERROR is `restart`'s
IDLE_DONE = {"FINISHED", "EXPIRED", "WAITING_FOR_BACKGROUND_WORK"}
AGENT_RE = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
WAKE = ("wake: your duty timer seems not to have fired ({why}; the agent watch started this run). Run your duties, read your "
        "doorbell (`research msg read --as {handle}`), and set your timer and subscriptions again.")
SCHEMA = "verity/agent-watch-status/v1"
AGENT_STATUSES = ("ACTIVE", "IDLE", "ARCHIVED", "unknown")
RUN_STATUSES = ("CREATING", "RUNNING", "WAITING_FOR_BACKGROUND_WORK", "FINISHED", "ERROR", "CANCELLED", "EXPIRED", "unreadable",
                "unknown")
WATCH_STATES = ("ok", "followed", "held", "held_again", "continued", "continue_failed", "cancel_refused", "long_call", "stuck",
                "crashed", "erroring", "unreadable", "ready_to_wake")


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def epoch(s: str | None) -> float | None:
    """The API's ISO time (2026-10-18T09:49:50.225Z) in epoch seconds; None when it doesn't parse."""
    try:
        return calendar.timegm(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def read_json(p: Path | None) -> dict:
    try:
        d = json.loads(p.read_text()) if p else {}
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def holders(registry: Path | None, routes: Path | None) -> dict[str, str]:
    """agent id -> handle, for each registry handle: its open inbox's holder (`routes/<handle>.json`, which a handover moves
    first), else the first agent id in its `holder` (later ones are lanes that post as it)."""
    out = {}
    for h in read_json(registry).get("handles") or []:
        r = read_json(routes / f"{h.get('handle')}.json") if routes else {}
        m = AGENT_RE.fullmatch(str(r.get("holder") or "")) if not r.get("closed") else None
        if m := m or AGENT_RE.search(str(h.get("holder", ""))):
            out[m[0]] = h["handle"]
    return out


def doorbells(path: Path, t: float) -> tuple[dict[str, float], str]:
    """handle -> its latest doorbell (a Slack ts), from the relay's file, and whether that was fresh, stale or missing."""
    d = read_json(path)
    if not d:
        return {}, "missing"
    if t - (epoch(d.get("t")) or 0) >= BELLS_STALE_S:
        return {}, "stale"
    out = {}
    for handle, i in (d.get("inboxes") or {}).items():
        try:
            out[str(handle)] = float(i["last"])
        except (KeyError, TypeError, ValueError):
            continue
    return out, "fresh"


def started(x: dict) -> bool:
    """Whether a run record ever ran: a run cancelled on arrival reads CANCELLED with no `durationMs`."""
    return x.get("status") not in ("CREATING", "CANCELLED") or bool(x.get("durationMs"))


class Api:
    """Cursor's Cloud Agents API, v1."""

    def __init__(self, key: str, base: str = API, timeout: float = 60):
        self.auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
        self.base, self.timeout = base, timeout

    def _req(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Authorization": self.auth, "Accept": "application/json", **(headers or {})}
        if data is not None:
            h["Content-Type"] = "application/json"
        return urllib.request.Request(self.base + path, data=data, method=method, headers=h)

    def get(self, path: str, **params) -> dict:
        q = ("?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})) if params else ""
        with urllib.request.urlopen(self._req("GET", path + q), timeout=self.timeout) as r:
            return json.load(r)

    def post(self, path: str, body: dict | None = None) -> dict:
        with urllib.request.urlopen(self._req("POST", path, body if body is not None else {}), timeout=self.timeout) as r:
            return json.load(r)

    def agents(self) -> Iterator[dict]:
        cursor = None
        while True:
            page = self.get("/v1/agents", limit=100, cursor=cursor, includeArchived="false")
            yield from page.get("items") or []
            cursor = page.get("nextCursor")
            if not cursor:
                return

    def stream(self, agent: str, run: str, stop: Callable[[], bool] = lambda: False) -> Iterator[tuple[str, dict]]:
        """(event, data) from the run's stream, until it ends or `stop()`; resumes from the last event id when the connection
        drops. Returns on 400, 404 or 410 (410 stream_expired past the retention window)."""
        last_id = None
        while not stop():
            headers = {"Accept": "text/event-stream", **({"Last-Event-ID": last_id} if last_id else {})}
            try:
                with urllib.request.urlopen(self._req("GET", f"/v1/agents/{agent}/runs/{run}/stream", headers=headers),
                                            timeout=10 * 60) as r:
                    event, data = "message", []
                    for raw in r:
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if not line:
                            if data:
                                try:
                                    payload = json.loads("\n".join(data))
                                except ValueError:
                                    payload = {"text": "\n".join(data)}
                                yield event, payload if isinstance(payload, dict) else {"value": payload}
                                if event == "done":
                                    return
                            event, data = "message", []
                        elif line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif line.startswith("id:"):
                            last_id = line[3:].strip()
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 410):
                    return
                raise
            except (TimeoutError, OSError):
                continue


@dataclass
class Run:
    agent: str
    run: str
    name: str = ""
    status: str = "RUNNING"
    last_kind: str = "start"            # start | text | call | tool_done | turn_ended
    last_t: float = 0.0                 # the parent's last event
    bg_t: float = 0.0                   # its subagents' last progress
    calls: dict[str, float] = field(default_factory=dict)   # open tool calls the parent waits on: callId -> started
    bg: set[str] = field(default_factory=set)               # open calls it went on past: background work
    told: set[str] = field(default_factory=set)
    ended: str | None = None
    streaming: bool = False
    refused_t: float | None = None                  # when its cancel got 409


def lock(path: Path) -> IO | None:
    """An exclusive lock on `path`, held while the returned file stays open; None when another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


class Watch:
    def __init__(self, api, log: Path, clock: Callable[[], float] = time.time, registry: Path | None = None,
                 routes: Path | None = None, wake: bool = False):
        self.api, self.log, self.clock, self.wake = api, log, clock, wake
        self.registry, self.routes = registry, routes
        self.doorbells = ""                         # whether doorbells.json was fresh, stale or missing at the last sweep
        self.runs: dict[str, Run] = {}
        self.states: dict[str, tuple[str, str, str]] = {}   # agent -> (run, its last finding's kind, since), for status.json
        self.status_of: dict[str, tuple[str, str]] = {}     # agent -> (run, its status when the watch last read it)
        self.lock = threading.Lock()
        self.state_path = log.with_name("state.json")
        try:
            s = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            s = {}
        self.freed: dict[str, float] = s.get("freed", {})         # agent -> when the watch last cancelled or restarted it
        self.errors: dict[str, int] = s.get("errors", {})         # agent -> ERROR runs in a row the watch restarted
        self.left: set[str] = set(s.get("left", []))              # agents past MAX_RESTARTS, recorded once
        self.pending: dict[str, dict] = s.get("pending", {})      # agent -> a continue run the API refused, tried again
        self.done: dict[str, float] = s.get("done", {})           # runs judged once (ERROR, stuck, unreadable) -> when
        self.woken: dict[str, float] = s.get("woken", {})         # agent -> when the watch last recorded it ready to wake
        self.final: dict[str, dict] = {}                          # run -> its terminal record, never read again (not saved)

    def save(self) -> None:
        t = self.clock()
        self.done = {k: v for k, v in self.done.items() if v >= t - 2 * 86400}
        self.woken = {k: v for k, v in self.woken.items() if v >= t - RECANCEL_S}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"freed": self.freed, "errors": self.errors, "left": sorted(self.left),
                                   "pending": self.pending, "done": self.done, "woken": self.woken}) + "\n")
        tmp.replace(self.state_path)

    def record(self, kind: str, r: Run, **detail) -> None:
        rec = {"t": iso(self.clock()), "kind": kind, "agent": r.agent, "name": r.name, "run": r.run, **detail}
        self.states[r.agent] = (r.run, kind, str(detail.get("since") or rec["t"]))
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def once(self, run: str, t: float) -> bool:
        """True the first time `run` is judged (an ERROR restart, a stuck or unreadable record), across restarts."""
        if run in self.done:
            return False
        self.done[run] = t
        self.save()
        return True

    def restart(self, a: dict, run: str, t: float) -> None:
        """A new run on an agent whose latest run ended in ERROR, at most once an hour and `MAX_RESTARTS` times in a row."""
        r = Run(a["id"], run, a.get("name") or "")
        n = self.errors.get(a["id"], 0)
        if run in self.done:                        # restarted already: only the next ERROR run counts
            return
        if n >= MAX_RESTARTS:
            if a["id"] not in self.left:
                self.left.add(a["id"])
                self.save()
                self.record("erroring", r, status="ERROR", restarts=n, action="none: left for a person")
            return
        if t - self.freed.get(a["id"], -RECANCEL_S) < RECANCEL_S or not self.once(run, t):
            return
        self.freed[a["id"]], self.errors[a["id"]] = t, n + 1
        self.save()
        try:
            new = self.api.post(f"/v1/agents/{a['id']}/runs", {"prompt": {"text": AFTER_ERROR}})
            self.record("crashed", r, status="ERROR", action=f"restarted ({n + 1} of {MAX_RESTARTS})",
                        new_run=((new or {}).get("run") or {}).get("id"))
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # agent_busy: one of its own events already started a run; not ours to count
                self.freed.pop(a["id"], None)
                self.errors[a["id"]] = n
                self.done.pop(run, None)
                self.save()
            else:
                self.record("crashed", r, status="ERROR", action=f"restart failed: HTTP {e.code}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.record("crashed", r, status="ERROR", action=f"restart failed: {type(e).__name__}: {e}")

    @staticmethod
    def ready(status: str | None, last: float | None, bell: float | None, t: float) -> str | None:
        """Why a coordinator whose latest run is in `status`, last active at `last`, is ready to wake. Only a doorbell after
        `last` counts: one rung during the turn may have been read in it, and the timer rule catches one that wasn't."""
        if status not in IDLE_DONE or last is None:
            return None
        if bell is not None and bell > last and t - bell >= DOORBELL_S:
            return f"a doorbell rang at {iso(bell)} and nothing has run since"
        if t - last >= WAKE_S:
            return f"nothing has run for {int((t - last) // 60)} min since its latest run went {status} at {iso(last)}"
        return None

    def wake_up(self, a: dict, handle: str, run: str, why: str, t: float) -> None:
        """Record a coordinator ready to wake, and with `wake` start a run on it; at most once per agent an hour."""
        if t - self.woken.get(a["id"], -RECANCEL_S) < RECANCEL_S:
            return
        self.woken[a["id"]], new, action = t, None, "recorded (--wake is off)"
        self.save()
        if self.wake:
            try:
                new = ((self.api.post(f"/v1/agents/{a['id']}/runs", {"prompt": {"text": WAKE.format(why=why, handle=handle)}})
                        or {}).get("run") or {}).get("id")
                action = "started a run"
            except urllib.error.HTTPError as e:     # 409 agent_busy counts too: zombie CREATING runs keep an agent busy for hours
                action = ("wake refused: HTTP 409 (agent busy); tried again in an hour" if e.code == 409 else
                          f"wake failed: HTTP {e.code}")
            except (urllib.error.URLError, OSError, ValueError) as e:
                action = f"wake failed: {type(e).__name__}: {e}"
        self.record("ready_to_wake", Run(a["id"], run, a.get("name") or ""), handle=handle, reason=why, action=action,
                    **({"new_run": new} if new else {}))

    def on_event(self, run_id: str, event: str, data: dict) -> None:
        with self.lock:
            r = self.runs.get(run_id)
            if r is None:
                return
            t = self.clock()
            kind = data.get("type") if event == "interaction_update" else None
            if event in ("status", "result"):
                if str(data.get("status")) in TERMINAL:
                    r.ended = str(data.get("status"))
                return
            if event not in ("tool_call", "assistant", "thinking", "interaction_update"):
                return                              # heartbeat, error, done
            cid = str(data.get("callId") or data.get("name") or "?") if event == "tool_call" else data.get("callId")
            if kind in SUBAGENT or (cid is not None and str(cid) in r.bg):
                r.bg_t = t                          # a subagent's progress, or its task call finishing: not the parent's doing
                return
            if event == "tool_call":
                if data.get("status") == "completed" and cid not in r.calls and (
                        data.get("name") == "task" or r.status == "WAITING_FOR_BACKGROUND_WORK"):
                    r.bg_t = t                      # a worker finishing, its start before the replay: the parent's turn is over
                    return
                if data.get("status") == "completed":
                    r.calls.pop(cid, None)
                    r.last_kind = "call" if r.calls else "tool_done"
                else:
                    r.calls.setdefault(cid, t)
                    r.last_kind = "call"
            elif event in ("assistant", "thinking") or kind in ("step-started", "turn-ended", "user-message-appended"):
                r.bg.update(r.calls)                # it went on past them: background work (a cloud subagent's task)
                r.calls.clear()
                if event in ("assistant", "thinking"):
                    r.last_kind = "text"
                elif kind == "turn-ended":
                    r.last_kind = "turn_ended"
                elif kind == "user-message-appended":
                    r.last_kind = "start"
            r.last_t = t

    def classify(self, r: Run, t: float) -> str | None:
        if r.ended:
            return "ended"
        if r.calls and t - min(r.calls.values()) >= LONG_CALL_S:
            return "long_call"
        waiting = r.status == "WAITING_FOR_BACKGROUND_WORK"
        quiet = t - (max(r.last_t, r.bg_t) if waiting else r.last_t)
        if not r.calls and quiet >= FOLLOW[r.status] and (r.last_kind == "tool_done" or (r.last_kind == "turn_ended" and not waiting)):
            return "held"
        if waiting and t - max(r.last_t, r.bg_t) >= STUCK[r.status]:
            return "stuck"
        return None

    def free(self, r: Run, t: float) -> None:
        since, mins = iso(r.last_t), int((t - r.last_t) // 60)
        if t - self.freed.get(r.agent, -RECANCEL_S) < RECANCEL_S:
            if "held-again" not in r.told:
                r.told.add("held-again")
                self.record("held_again", r, since=since, minutes=mins, action="none: freed once within the hour")
            return
        if r.refused_t is not None:
            if t - r.refused_t >= REFUSED_S and "refused" not in r.told:
                r.told.add("refused")
                self.record("cancel_refused", r, since=since, minutes=mins,
                            action=f"none: the API refused the cancel (409) and still reads it {r.status}")
            return
        try:
            self.api.post(f"/v1/agents/{r.agent}/runs/{r.run}/cancel")
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # run_not_cancellable: it ended meanwhile, and the runs list lags
                r.refused_t = t
                return
            self.freed[r.agent] = t
            self.save()
            self.record("held", r, since=since, minutes=mins, action=f"cancel failed: HTTP {e.code}")
            r.ended = "CANCELLED"
            return
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.freed[r.agent] = t
            self.save()
            self.record("held", r, since=since, minutes=mins, action=f"cancel failed: {type(e).__name__}: {e}")
            r.ended = "CANCELLED"
            return
        self.freed[r.agent] = t
        r.ended = "CANCELLED"
        text = CONTINUE.format(since=since, mins=mins, after=AFTER.get(r.last_kind, "a finished tool call"))
        new, err = self.cont(r.agent, text)
        if new is not None:
            self.save()
            self.record("held", r, since=since, minutes=mins, action="cancelled and continued", new_run=new)
        else:
            self.pending[r.agent] = {"text": text, "run": r.run, "name": r.name, "since": t}
            self.save()
            self.record("held", r, since=since, minutes=mins, action=f"cancelled; the continue run was refused ({err}), "
                        "tried again each sweep")

    def cont(self, agent: str, text: str) -> tuple[str | None, str]:
        """Start the continue run: (its id or "", "") once started, (None, why) when refused."""
        try:
            new = self.api.post(f"/v1/agents/{agent}/runs", {"prompt": {"text": text}})
            return ((new or {}).get("run") or {}).get("id") or "", ""
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return None, f"{type(e).__name__}: {e}"

    def retry(self, current: dict[str, tuple[dict, dict]], t: float) -> None:
        """Start each continue run the API refused, unless the agent has a run of its own going again."""
        going = {a["id"] for a, x in current.values() if x["id"] != self.pending.get(a["id"], {}).get("run")}
        for agent, p in list(self.pending.items()):
            r = Run(agent, p["run"], p.get("name", ""))
            if agent in going:
                self.pending.pop(agent)
            elif t - p["since"] >= CONTINUE_RETRY_S:
                self.pending.pop(agent)
                self.record("continue_failed", r, since=iso(p["since"]), action="none: left for a person")
            else:
                new, _ = self.cont(agent, p["text"])
                if new is None:
                    continue
                self.pending.pop(agent)
                self.record("continued", r, new_run=new)
            self.save()

    def read(self, a: dict, run: str) -> dict | None:
        """The run's record (a terminal one is read once); None, and recorded once, when the API won't read it."""
        if run in self.final:
            return self.final[run]
        try:
            info = {"id": run, **self.api.get(f"/v1/agents/{a['id']}/runs/{run}")}
        except urllib.error.HTTPError as e:         # a goal run's id (run-goal-...) isn't one the runs endpoint takes
            if self.once(run, self.clock()):
                self.record("unreadable", Run(a["id"], run, a.get("name") or ""), status=f"HTTP {e.code}")
            return None
        if info.get("status") in TERMINAL:
            self.final[run] = info
        return info

    def runs_of(self, a: dict, t: float) -> list[dict] | None:
        """The agent's newest runs, newest first, back to the newest that started: an ACTIVE agent's paged from the list, else
        its latest, paged behind only when that was cancelled on arrival. None when the list fails."""
        latest = a["latestRunId"]
        runs: list[dict] = []
        if a.get("status") != "ACTIVE":
            if not (info := self.read(a, latest)) or started(info):
                return [info] if info else []
            runs = [info]
        try:
            listed, cursor = [], None
            for _ in range(RUNS_PAGES):
                page = self.api.get(f"/v1/agents/{a['id']}/runs", limit=RUNS_PAGE, cursor=cursor)
                listed += page.get("items") or []
                cursor = page.get("nextCursor")
                if not cursor or any(started(x) for x in listed):
                    break
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(t)} runs of {a['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return runs or None
        if not runs and latest not in {x.get("id") for x in listed} and (info := self.read(a, latest)):
            runs = [info]
        return runs + [x for x in listed if x.get("id") not in {y["id"] for y in runs}]

    def sweep(self, start_stream: Callable[[Run], None]) -> None:
        """One pass: find each live agent's current run and follow it, record runs stuck in CREATING, restart an ERROR one,
        start refused continue runs, then judge every followed run."""
        t = self.clock()
        live, agents = {}, list(self.api.agents())
        for a in agents:
            recent = t - (epoch(a.get("updatedAt")) or 0) < IDLE_RECENT_S
            if a.get("latestRunId") and (a.get("status") == "ACTIVE" or (a.get("status") == "IDLE" and recent)):
                live[a["id"]] = a
        current: dict[str, tuple[dict, dict]] = {}      # run -> (agent, run record)
        judged: dict[str, dict] = {}                    # run -> agent: the run each agent read is judged by
        judges: dict[str, dict] = {}                    # agent -> the record of the run it is judged by
        for a in live.values():
            latest = a["latestRunId"]
            if (runs := self.runs_of(a, t)) is None:
                continue
            follow = next((x for x in runs if x.get("status") in FOLLOW and not str(x.get("id")).startswith(GOAL)), None)
            newer_started = False                       # a CREATING run with a newer one started after it blocks nothing,
            for x in runs if follow is None else ():    # and one behind a followed run waits for it, judged in its place
                since = epoch(x.get("updatedAt") or x.get("createdAt"))
                if (x.get("status") == "CREATING" and not newer_started and since is not None
                        and t - since >= STUCK["CREATING"] and self.once(x["id"], t)):
                    self.record("stuck", Run(a["id"], x["id"], a.get("name") or ""), status="CREATING", since=iso(since),
                                minutes=int((t - since) // 60))
                newer_started |= started(x)
            last = next((x for x in runs if x.get("id") == latest), None)
            if follow:
                current[follow["id"]] = (a, follow)
            judge = follow or next((x for x in runs if started(x) or x.get("status") == "CREATING"), None) or last
            judged[judge["id"] if judge else latest] = a
            if judge:
                judges[a["id"]] = judge
                self.status_of[a["id"]] = (judge["id"], str(judge.get("status")))
            if last and last.get("status") == "FINISHED":   # not a run going: a restart runs a minute before it errors
                if self.errors.pop(a["id"], None) is not None or a["id"] in self.left:
                    self.left.discard(a["id"])
                    self.save()
            elif last and last.get("status") == "ERROR":
                self.restart(a, latest, t)
        self.retry(current, t)
        coords, (bells, fresh) = holders(self.registry, self.routes), doorbells(self.log.with_name("doorbells.json"), t)
        if fresh != self.doorbells and (coords or self.doorbells):
            print(f"{iso(t)} doorbells.json: {fresh}; {len(coords)} coordinators", file=sys.stderr, flush=True)
        self.doorbells = fresh
        for a in agents:
            handle = coords.get(a["id"])
            if not handle or not a.get("latestRunId") or a.get("status") == "ARCHIVED":
                continue
            if (info := judges.get(a["id"])) is None:
                runs = [] if a["id"] in live else self.runs_of(a, t) or []
                if not (info := next((x for x in runs if started(x)), runs[0] if runs else None)):
                    continue
            run = info["id"]
            with self.lock:
                r = self.runs.get(run)
                heard = r.last_t if r and r.last_kind != "start" else None
            last = max((x for x in (epoch(info.get("updatedAt") or info.get("createdAt")), heard) if x is not None), default=None)
            if why := self.ready(info.get("status"), last, bells.get(handle), t):
                self.wake_up(a, handle, run, why, t)
        for run, (a, x) in current.items():
            with self.lock:
                r = self.runs.get(run)
                if r is None:
                    waiting = x["status"] == "WAITING_FOR_BACKGROUND_WORK"
                    since = min(t, epoch(x.get("updatedAt")) or t) if waiting else t
                    r = self.runs[run] = Run(a["id"], run, a.get("name") or "", last_t=since)
                r.status = x["status"]
                start = not r.streaming
                r.streaming = True
            if start:
                start_stream(r)
        with self.lock:
            for run in [run for run in self.runs if run not in current]:
                self.runs.pop(run)
            runs = list(self.runs.values())
        for r in runs:
            with self.lock:
                kind = self.classify(r, t)
                if kind == "held":
                    self.free(r, t)
                elif kind in ("long_call", "stuck") and kind not in r.told:
                    r.told.add(kind)
                    if kind == "long_call":
                        self.record(kind, r, since=iso(min(r.calls.values())))
                    elif self.once(r.run + "#stuck", t):
                        quiet = max(r.last_t, r.bg_t)
                        self.record(kind, r, status=r.status, since=iso(quiet), minutes=int((t - quiet) // 60))
                if r.ended:
                    self.runs.pop(r.run, None)
        with self.lock:
            write_atomic(self.log.with_name("status.json"), status_bytes(snapshot(self, judged, t)))


def snapshot(w: Watch, live: dict[str, dict], t: float) -> dict:
    """`status.json`, schema `verity/agent-watch-status/v1`: one entry per agent the sweep read, in categories only (no name,
    prompt, error text, branch or Slack identity), keyed in `live` by the run the watch judges it by (the run it follows, else
    its newest run that started or is CREATING, else its latest). `watch_state` is the last finding on that run (a long call
    only while it lasts), else `followed` or `ok`; `since` is when that finding began; `run_status` is the run's status when
    the watch last read it. A value outside the pinned enums raises, so the last valid file stays."""
    agents = []
    for run, a in live.items():
        agent, r = a["id"], w.runs.get(run)
        state_run, state, since = w.states.get(agent, ("", "", None))
        long_call = r is not None and bool(r.calls) and t - min(r.calls.values()) >= LONG_CALL_S
        if state_run != run or (state == "long_call" and not long_call):
            state, since = ("followed" if r is not None else "ok"), None
        read_run, read_status = w.status_of.get(agent, ("", ""))
        run_status = ("unreadable" if state == "unreadable" else
                      read_status if read_run == run and read_status in RUN_STATUSES else "unknown")
        agent_status = a.get("status") if a.get("status") in AGENT_STATUSES else "unknown"
        if state not in WATCH_STATES:
            raise ValueError(f"status.json: watch state {state!r} is not one of WATCH_STATES; add it there")
        agents.append({"agent_id": agent, "run_id": run, "agent_status": agent_status, "run_status": run_status,
                       "watch_state": state, "since": since, "restart_count": int(w.errors.get(agent, 0))})
    return {"schema": SCHEMA, "t": iso(t), "poll_s": POLL_S, "agents": sorted(agents, key=lambda e: e["agent_id"])}


def status_bytes(snap: dict) -> bytes:
    """The same snapshot always gives the same bytes."""
    return (json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_atomic(path: Path, data: bytes) -> None:
    """Replace `path` with `data` or leave it as it was: a temporary file in its directory, fsynced, renamed over it, then the
    directory fsynced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def pump(watch: Watch, r: Run) -> None:
    """Feed the run's stream to the watch until the run ends or the watch stops following it; then the next sweep may follow
    it again."""
    def stop() -> bool:
        return r.ended is not None or watch.runs.get(r.run) is not r
    try:
        for event, data in watch.api.stream(r.agent, r.run, stop=stop):
            watch.on_event(r.run, event, data)
            if stop():
                break
    except Exception as e:  # noqa: BLE001 (a stream that fails is followed again by the next sweep)
        print(f"{iso(time.time())} stream {r.run} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        r.streaming = False


def follow(watch: Watch, r: Run) -> None:
    threading.Thread(target=pump, args=(watch, r), daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_watch.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--log", type=Path, default=Path("/data/agent-watch/events.jsonl"))
    lp.add_argument("--registry", type=Path, default=None, help="the Slack registry (registry.json): the handles")
    lp.add_argument("--routes", type=Path, default=None, help="the notes' routes.d: who holds each handle's inbox")
    lp.add_argument("--wake", action="store_true", help="start a run on a coordinator ready to wake (default: record it only)")
    lp.add_argument("--once", action="store_true", help="one sweep, following no stream")
    lp.add_argument("--dry-run", action="store_true", help="send no POST; print each instead")
    ns = ap.parse_args(argv)
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        print("agent_watch: no CURSOR_API_KEY in the environment", file=sys.stderr)
        return 2
    held = lock(ns.log.with_name("watch.lock"))
    if held is None:
        print(f"{iso(time.time())} another watch holds {ns.log.with_name('watch.lock')}", file=sys.stderr, flush=True)
        return 0
    api = Api(key)
    if ns.dry_run:
        def post(path: str, body: dict | None = None) -> dict:
            print(f"dry-run: POST {path} {json.dumps(body)}", file=sys.stderr, flush=True)
            return {"run": {"id": "dry-run"}}
        api.post = post
    watch = Watch(api, ns.log, registry=ns.registry, routes=ns.routes, wake=ns.wake)
    if ns.once:
        watch.sweep(lambda r: None)
        return 0
    while True:
        try:
            watch.sweep(lambda r: follow(watch, r))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(time.time())} sweep failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
