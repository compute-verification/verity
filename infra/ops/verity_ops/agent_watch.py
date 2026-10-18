"""The agent watch (INF-69): free a cloud agent whose run is held, and record what it sees, from the control pod.

    python3 agent_watch.py loop --log /data/agent-watch/events.jsonl     $CURSOR_API_KEY set; standard library only

An agent hears Slack, Linear and timer events only between runs, and a run can hang after a tool call returns: proofs from
05:29 to 15:33Z and circuits from 09:49Z on 18 Oct, three coordinators at once about 15:26Z on 15 Oct. Until now only Daniel
pressing Stop freed one. Every `POLL_S` the watch lists the team's agents (`GET /v1/agents`; ACTIVE ones, and IDLE ones changed
in the last `IDLE_RECENT_S`, since an ERROR leaves an agent IDLE). For each ACTIVE agent it lists its newest runs
(`GET .../runs`): its `latestRunId` is often a run cancelled on arrival, created and cancelled within half a second while the
run doing the work is older (memory-accounting, 19:32Z 18 Oct). It follows the event stream (`GET .../runs/{run}/stream`,
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
  is tried again every sweep for `CONTINUE_RETRY_S`, then left for a person. It cancels at most once per agent every
  `RECANCEL_S`, so a run that hangs again is left for a person;
- **in a long call** when a tool call the parent waits on has been running for `LONG_CALL_S` (recorded, never cancelled: a
  build or a wait may be legitimate);
- **crashed** when the agent's latest run ended in ERROR: the watch starts a new run, at most once an hour and `MAX_RESTARTS`
  times in a row (vllm's runs errored within a minute, run after run, on 18 Oct), then records it as **erroring**, for a person.
  Only a latest run that FINISHED ends the row: a restarted run reads RUNNING for a while before it errors again.

A run among an agent's newest left in CREATING for 10 min by the API's own `updatedAt` with no newer run started (only ones
cancelled on arrival), or a followed background wait with nothing at all streamed for 2 h, is recorded once as **stuck**: on 18 Oct proofs' and circuits' runs sat in CREATING behind a
held one while every later one was cancelled on arrival, network-accounting's behind a goal run from 18:27Z, and the API refuses
to cancel a CREATING run. A goal run (`run-goal-...`) can't be read or streamed on its own (400), so it is never followed; an
agent whose latest run the API won't read is recorded once as unreadable.

Each finding is a line of JSON in `--log`; each sweep overwrites `status.json` beside it (its time, the agents read, the runs it
follows), so a reader can tell a quiet watch from a dead one. `state.json` beside it keeps what must outlive a restart (the
watch restarts on every deploy): when it last cancelled or restarted each agent, the restart counts, the continue runs still to
start and the runs already recorded. `watch.lock` beside it keeps a second watch from running. The control pod holds no Slack
credential, so infra's relay reads the log over ssh and rings the agent's coordinator. Cancelling and restarting don't wait for it.
"""
from __future__ import annotations

import argparse
import base64
import calendar
import fcntl
import json
import os
import sys
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
MAX_RESTARTS = 3            # ERROR runs in a row the watch restarts before leaving the agent for a person
IDLE_RECENT_S = 6 * 3600    # an IDLE agent is read only if it changed this recently (an ERROR leaves an agent IDLE)
RUNS_PAGE = 50              # an ACTIVE agent's newest runs read each sweep (proofs had 25 runs cancelled on arrival in 7 h)
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


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def epoch(s: str | None) -> float | None:
    """The API's ISO time (2026-10-18T09:49:50.225Z) in epoch seconds; None when it doesn't parse."""
    try:
        return calendar.timegm(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


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
    def __init__(self, api, log: Path, clock: Callable[[], float] = time.time):
        self.api, self.log, self.clock = api, log, clock
        self.runs: dict[str, Run] = {}
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
        self.final: dict[str, str] = {}                           # run -> its terminal status, never read again

    def save(self) -> None:
        cutoff = self.clock() - 2 * 86400
        self.done = {k: v for k, v in self.done.items() if v >= cutoff}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"freed": self.freed, "errors": self.errors, "left": sorted(self.left),
                                   "pending": self.pending, "done": self.done}) + "\n")
        tmp.replace(self.state_path)

    def record(self, kind: str, r: Run, **detail) -> None:
        rec = {"t": iso(self.clock()), "kind": kind, "agent": r.agent, "name": r.name, "run": r.run, **detail}
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
        try:
            self.api.post(f"/v1/agents/{r.agent}/runs/{r.run}/cancel")
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # run_not_cancellable: it ended meanwhile; judged again next sweep
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
            return {"id": run, "status": self.final[run]}
        try:
            info = {"id": run, **self.api.get(f"/v1/agents/{a['id']}/runs/{run}")}
        except urllib.error.HTTPError as e:         # a goal run's id (run-goal-...) isn't one the runs endpoint takes
            if self.once(run, self.clock()):
                self.record("unreadable", Run(a["id"], run, a.get("name") or ""), status=f"HTTP {e.code}")
            return None
        if info.get("status") in TERMINAL:
            self.final[run] = info["status"]
        return info

    def sweep(self, start_stream: Callable[[Run], None]) -> None:
        """One pass: find each live agent's current run and follow it, record runs stuck in CREATING, restart an ERROR one,
        start refused continue runs, then judge every followed run."""
        t = self.clock()
        live = {}
        for a in self.api.agents():
            recent = t - (epoch(a.get("updatedAt")) or 0) < IDLE_RECENT_S
            if a.get("latestRunId") and (a.get("status") == "ACTIVE" or (a.get("status") == "IDLE" and recent)):
                live[a["id"]] = a
        current: dict[str, tuple[dict, dict]] = {}      # run -> (agent, run record)
        for a in live.values():
            latest = a["latestRunId"]
            if a.get("status") == "ACTIVE":
                try:
                    runs = list(self.api.get(f"/v1/agents/{a['id']}/runs", limit=RUNS_PAGE).get("items") or [])
                except (urllib.error.URLError, OSError, ValueError) as e:
                    print(f"{iso(t)} runs of {a['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    continue
                if latest not in {x.get("id") for x in runs}:
                    info = self.read(a, latest)
                    runs = ([info] if info else []) + runs
            else:
                info = self.read(a, latest)
                runs = [info] if info else []
            newer_started = False                       # a CREATING run with a newer one started after it blocks nothing
            for x in runs:
                since = epoch(x.get("updatedAt") or x.get("createdAt"))
                if (x.get("status") == "CREATING" and not newer_started and since is not None
                        and t - since >= STUCK["CREATING"] and self.once(x["id"], t)):
                    self.record("stuck", Run(a["id"], x["id"], a.get("name") or ""), status="CREATING", since=iso(since),
                                minutes=int((t - since) // 60))
                newer_started |= x.get("status") not in ("CREATING", "CANCELLED") or bool(x.get("durationMs"))
            follow = next((x for x in runs if x.get("status") in FOLLOW and not str(x.get("id")).startswith(GOAL)), None)
            last = next((x for x in runs if x.get("id") == latest), None)
            if follow:
                current[follow["id"]] = (a, follow)
            if last and last.get("status") == "FINISHED":   # not a run going: a restart runs a minute before it errors
                if self.errors.pop(a["id"], None) is not None or a["id"] in self.left:
                    self.left.discard(a["id"])
                    self.save()
            elif last and last.get("status") == "ERROR":
                self.restart(a, latest, t)
        self.retry(current, t)
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
        status = self.log.with_name("status.json")
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"t": iso(t), "active": len(live), "followed": [r.run for r in runs]}) + "\n")
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
    ns = ap.parse_args(argv)
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        print("agent_watch: no CURSOR_API_KEY in the environment", file=sys.stderr)
        return 2
    held = lock(ns.log.with_name("watch.lock"))
    if held is None:
        print(f"{iso(time.time())} another watch holds {ns.log.with_name('watch.lock')}", file=sys.stderr, flush=True)
        return 0
    watch = Watch(Api(key), ns.log)
    while True:
        try:
            watch.sweep(lambda r: follow(watch, r))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(time.time())} sweep failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
