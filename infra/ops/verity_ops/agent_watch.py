"""The agent watch (INF-69): free a cloud agent whose run is held, and record what it sees, from the control pod.

    python3 agent_watch.py loop --log /data/agent-watch/events.jsonl     $CURSOR_API_KEY set; standard library only

An agent hears Slack, Linear and timer events only between runs, and a run can hang after a tool call returns: proofs from
05:29 to 15:33Z and circuits from 09:49Z on 18 Oct, three coordinators at once about 15:26Z on 15 Oct. Until now only Daniel
pressing Stop freed one. Every `POLL_S` the watch lists the team's agents (`GET /v1/agents`; ACTIVE ones, and IDLE ones changed
in the last `IDLE_RECENT_S`, since an ERROR leaves an agent IDLE) and follows the event stream (`GET .../runs/{run}/stream`,
server-sent events) of each latest run that is RUNNING or WAITING_FOR_BACKGROUND_WORK. A held agent can read as either (proofs
and circuits read as the second, top 18:07Z). Events carry no time, so it times them as they arrive, and a heartbeat counts as
nothing. A run is

- **held** when its last event is a finished tool call and nothing has followed for `HELD_S` (RUNNING) or `HELD_BG_S`
  (WAITING_FOR_BACKGROUND_WORK, longer because cancelling it may take its local background workers with it). The watch cancels
  it (`POST .../runs/{run}/cancel`) and starts a new run on the agent (`POST /v1/agents/{id}/runs`) that says from when it was
  held and to read its inbox. It does so at most once per agent every `RECANCEL_S`, so a run that hangs again is left for a person;
- **in a long call** when a tool call has been running for `LONG_CALL_S` (recorded, never cancelled: a build or a wait may be
  legitimate);
- **crashed** when the agent's latest run ended in ERROR: the watch starts a new run, at most once an hour and `MAX_RESTARTS`
  times in a row (vllm's runs errored within a minute, run after run, on 18 Oct), then records it as **erroring**, for a person.

A latest run left in CREATING for 10 min, or in WAITING_FOR_BACKGROUND_WORK for 2 h with nothing held, by the API's own
`updatedAt`, is recorded once as **stuck**: on 18 Oct circuits' new runs sat in CREATING while every later one was cancelled on
arrival, and the API refuses to cancel a CREATING run. An agent whose latest run the runs endpoint won't read (a goal run's
`run-goal-...` id gets 400) is recorded once as unreadable and skipped.

A coordinator holds a handle of `--registry` (the Slack registry): the holder of the handle's open inbox in `--routes` (the notes'
`routes.d`, which a handover moves first; on 18 Oct the registry still named vllm's predecessor), else the first agent id in the
handle's `holder`. It runs a 15-minute duty timer, and its inbox thread rings it for each doorbell. One whose latest run's turn is
over (FINISHED, EXPIRED, or WAITING_FOR_BACKGROUND_WORK, where coordinators mostly sit between wakes) is **ready to wake** when
nothing has run for `WAKE_S` since its last activity (its timer didn't fire), or when a doorbell for its handle rang after that
and `DOORBELL_S` has passed with nothing run (its subscription didn't). Its last activity is the later of the run's `updatedAt`
(read each sweep) and its last stream event: whether a wake starts a new run or resumes a background-waiting one, it moves. The
doorbells come from `doorbells.json` beside `--log` (`{"t", "inboxes": {handle: {"agent", "last"}}}`, each inbox's latest
doorbell as a Slack ts), which infra's relay writes, since only it reads Slack; one older than `BELLS_STALE_S` is not read. The
watch records it at most once per agent an hour, and with `--wake` starts a run that says so (a 409 agent_busy isn't counted: a
run of its own just started). `--wake` stays off until a red team has attacked the rule.

Each finding is a line of JSON in `--log`; each sweep overwrites `status.json` beside it (its time, the agents read, the runs it
follows, the coordinators it knows and whether the doorbells are fresh), so a reader can tell a quiet watch from a dead one. The
control pod holds no Slack credential, so infra's relay reads that file over ssh and rings the agent's coordinator. Cancelling and
restarting don't wait for it. `--once --dry-run` runs one sweep that follows no stream and sends no POST (each is printed).
"""
from __future__ import annotations

import argparse
import base64
import calendar
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

API = "https://api.cursor.com"
POLL_S = 60
HELD_S = 15 * 60
HELD_BG_S = 60 * 60         # held while WAITING_FOR_BACKGROUND_WORK: cancelling it may take its local background workers with it
LONG_CALL_S = 45 * 60
RECANCEL_S = 3600
MAX_RESTARTS = 3            # ERROR runs in a row the watch restarts before leaving the agent for a person
IDLE_RECENT_S = 6 * 3600    # an IDLE agent is read only if it changed this recently (an ERROR leaves an agent IDLE)
FOLLOW = {"RUNNING": HELD_S, "WAITING_FOR_BACKGROUND_WORK": HELD_BG_S}
TERMINAL = {"FINISHED", "ERROR", "CANCELLED", "EXPIRED"}
# a latest run left this long in a non-terminal state that isn't RUNNING (18 Oct: circuits' runs sat in CREATING from 09:49Z and
# in WAITING_FOR_BACKGROUND_WORK from 09:49Z, and every later run was cancelled on arrival); recorded, never cancelled
STUCK = {"CREATING": 10 * 60, "WAITING_FOR_BACKGROUND_WORK": 2 * 3600}
CONTINUE = ("continue: your run was held from {since} (the agent watch saw nothing for {mins} min after a finished tool call, and "
            "cancelled it). Read your inbox and carry on from where you stopped.")
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

    def stream(self, agent: str, run: str) -> Iterator[tuple[str, dict]]:
        """(event, data) from the run's stream, until it ends; resumes from the last event id when the connection drops."""
        last_id = None
        while True:
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
    last_kind: str = "start"            # start | text | call | tool_done
    last_t: float = 0.0
    calls: dict[str, float] = field(default_factory=dict)   # open tool calls: callId -> started
    told: set[str] = field(default_factory=set)
    ended: str | None = None
    held_s: float = HELD_S


class Watch:
    def __init__(self, api, log: Path, clock: Callable[[], float] = time.time, registry: Path | None = None,
                 routes: Path | None = None, wake: bool = False):
        self.api, self.log, self.clock, self.wake = api, log, clock, wake
        self.registry, self.routes = registry, routes
        self.runs: dict[str, Run] = {}
        self.done: dict[str, dict] = {}             # terminal runs read: run -> its record (a terminal run doesn't change)
        self.woken: dict[str, float] = {}           # agent -> when the watch last recorded it ready to wake
        self.freed: dict[str, float] = {}           # agent -> when the watch last cancelled one of its runs
        self.crashed: set[str] = set()              # runs recorded as crashed
        self.errors: dict[str, int] = {}            # agent -> ERROR runs in a row the watch restarted
        self.left: set[str] = set()                 # agents past MAX_RESTARTS, recorded once
        self.lock = threading.Lock()

    def restart(self, a: dict, run: str, t: float) -> None:
        """A new run on an agent whose latest run ended in ERROR, at most once an hour and `MAX_RESTARTS` times in a row."""
        r = Run(a["id"], run, a.get("name") or "")
        n = self.errors.get(a["id"], 0)
        if n >= MAX_RESTARTS:
            if a["id"] not in self.left:
                self.left.add(a["id"])
                self.record("erroring", r, status="ERROR", restarts=n, action="none: left for a person")
            return
        if t - self.freed.get(a["id"], -RECANCEL_S) < RECANCEL_S:
            self.crashed.discard(run)              # judged again once the hour is up
            return
        self.freed[a["id"]], self.errors[a["id"]] = t, n + 1
        try:
            new = self.api.post(f"/v1/agents/{a['id']}/runs", {"prompt": {"text": AFTER_ERROR}})
            self.record("crashed", r, status="ERROR", action=f"restarted ({n + 1} of {MAX_RESTARTS})",
                        new_run=((new or {}).get("run") or {}).get("id"))
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # agent_busy: one of its own events already started a run; not ours to count
                self.freed.pop(a["id"], None)
                self.errors[a["id"]] = n
                self.crashed.discard(run)
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
        if self.wake:
            try:
                new = ((self.api.post(f"/v1/agents/{a['id']}/runs", {"prompt": {"text": WAKE.format(why=why, handle=handle)}})
                        or {}).get("run") or {}).get("id")
                action = "started a run"
            except urllib.error.HTTPError as e:
                if e.code == 409:                   # agent_busy: a run of its own has just started; nothing missed
                    self.woken.pop(a["id"], None)
                    return
                action = f"wake failed: HTTP {e.code}"
            except (urllib.error.URLError, OSError, ValueError) as e:
                action = f"wake failed: {type(e).__name__}: {e}"
        self.record("ready_to_wake", Run(a["id"], run, a.get("name") or ""), handle=handle, reason=why, action=action,
                    **({"new_run": new} if new else {}))

    def run_info(self, a: dict, run: str) -> dict | None:
        """The run's record (from `done` once terminal); None, recorded once as unreadable, when the API won't read it."""
        if run in self.done:
            return self.done[run]
        try:
            info = self.api.get(f"/v1/agents/{a['id']}/runs/{run}")
        except urllib.error.HTTPError as e:         # a goal run's id (run-goal-...) isn't one the runs endpoint takes
            if run not in self.crashed:
                self.crashed.add(run)
                self.record("unreadable", Run(a["id"], run, a.get("name") or ""), status=f"HTTP {e.code}")
            return None
        if info.get("status") in TERMINAL:
            self.done[run] = info
        return info

    def record(self, kind: str, r: Run, **detail) -> None:
        rec = {"t": iso(self.clock()), "kind": kind, "agent": r.agent, "name": r.name, "run": r.run, **detail}
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def on_event(self, run_id: str, event: str, data: dict) -> None:
        with self.lock:
            r = self.runs.get(run_id)
            if r is None:
                return
            t = self.clock()
            if event == "heartbeat":
                return
            if event == "tool_call":
                cid = str(data.get("callId") or data.get("name") or "?")
                if data.get("status") == "completed":
                    r.calls.pop(cid, None)
                    r.last_kind = "tool_done" if not r.calls else "call"
                else:
                    r.calls.setdefault(cid, t)
                    r.last_kind = "call"
            elif event in ("assistant", "thinking", "interaction_update"):
                r.last_kind = "text"
            elif event in ("status", "result") and str(data.get("status")) in TERMINAL:
                r.ended = str(data.get("status"))
            r.last_t = t

    def classify(self, r: Run, t: float) -> str | None:
        if r.ended:
            return "crashed" if r.ended == "ERROR" else "ended"
        if r.calls and t - min(r.calls.values()) >= LONG_CALL_S:
            return "long_call"
        if not r.calls and r.last_kind == "tool_done" and t - r.last_t >= r.held_s:
            return "held"
        return None

    def free(self, r: Run, t: float) -> None:
        since, mins = iso(r.last_t), int((t - r.last_t) // 60)
        if t - self.freed.get(r.agent, -RECANCEL_S) < RECANCEL_S:
            if "held-again" not in r.told:
                r.told.add("held-again")
                self.record("held_again", r, since=since, minutes=mins, action="none: freed once within the hour")
            return
        self.freed[r.agent] = t
        try:
            self.api.post(f"/v1/agents/{r.agent}/runs/{r.run}/cancel")
            new = self.api.post(f"/v1/agents/{r.agent}/runs", {"prompt": {"text": CONTINUE.format(since=since, mins=mins)}})
            self.record("held", r, since=since, minutes=mins, action="cancelled and continued",
                        new_run=((new or {}).get("run") or {}).get("id"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.record("held", r, since=since, minutes=mins, action=f"free failed: {type(e).__name__}: {e}")
        r.ended = "CANCELLED"

    def sweep(self, start_stream: Callable[[Run], None]) -> None:
        """One pass: follow every RUNNING or background-waiting run not yet followed, restart an ERROR one, then judge every
        followed run."""
        t = self.clock()
        live, agents = {}, list(self.api.agents())
        for a in agents:
            run = a.get("latestRunId")
            recent = t - (epoch(a.get("updatedAt")) or 0) < IDLE_RECENT_S
            if run and (a.get("status") == "ACTIVE" or (a.get("status") == "IDLE" and recent)):
                live[run] = a
        for run, a in live.items():
            with self.lock:
                known = run in self.runs
            if known:
                continue
            info = self.run_info(a, run)
            if info is None:
                continue
            status = info.get("status")
            if status in STUCK and run not in self.crashed:
                since = epoch(info.get("updatedAt") or info.get("createdAt"))
                if since is not None and t - since >= STUCK[status]:
                    self.crashed.add(run)
                    self.record("stuck", Run(a["id"], run, a.get("name") or ""), status=status, since=iso(since),
                                minutes=int((t - since) // 60))
            if status in FOLLOW:
                self.errors.pop(a["id"], None)
                self.left.discard(a["id"])
                r = Run(a["id"], run, a.get("name") or "", last_t=t, held_s=FOLLOW[status])
                with self.lock:
                    self.runs[run] = r
                start_stream(r)
            elif status == "FINISHED":
                self.errors.pop(a["id"], None)
            elif status == "ERROR" and run not in self.crashed:
                self.crashed.add(run)
                self.restart(a, run, t)
        coords, (bells, fresh) = holders(self.registry, self.routes), doorbells(self.log.with_name("doorbells.json"), t)
        for a in agents:
            run, handle = a.get("latestRunId"), coords.get(a["id"])
            if not handle or not run or a.get("status") == "ARCHIVED":
                continue
            if not (info := self.run_info(a, run)):
                continue
            with self.lock:
                r = self.runs.get(run)
                heard = r.last_t if r and r.last_kind != "start" else None
            last = max((x for x in (epoch(info.get("updatedAt") or info.get("createdAt")), heard) if x is not None), default=None)
            if why := self.ready(info.get("status"), last, bells.get(handle), t):
                self.wake_up(a, handle, run, why, t)
        with self.lock:
            runs = list(self.runs.values())
        status = self.log.with_name("status.json")
        status.write_text(json.dumps({"t": iso(t), "active": len(live), "followed": [r.run for r in runs],
                                      "coordinators": len(coords), "doorbells": fresh}) + "\n")
        for r in runs:
            with self.lock:
                kind = self.classify(r, t)
                if kind == "held":
                    self.free(r, t)
                elif kind in ("long_call", "crashed") and kind not in r.told:
                    r.told.add(kind)
                    self.record(kind, r, since=iso(min(r.calls.values()) if r.calls else r.last_t))
                if r.ended or (r.run not in live and kind != "held"):
                    self.runs.pop(r.run, None)


def follow(watch: Watch, r: Run) -> None:
    def go():
        try:
            for event, data in watch.api.stream(r.agent, r.run):
                watch.on_event(r.run, event, data)
        except Exception as e:  # noqa: BLE001 (a stream that fails is retried by the next sweep)
            print(f"{iso(time.time())} stream {r.run} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            with watch.lock:
                watch.runs.pop(r.run, None)
    threading.Thread(target=go, daemon=True).start()


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
