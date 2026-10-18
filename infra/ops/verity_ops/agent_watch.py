"""The agent watch (INF-69): free a cloud agent whose run is held, and record what it sees, from the control pod.

    python3 agent_watch.py loop --log /data/agent-watch/events.jsonl     $CURSOR_API_KEY set; standard library only

An agent hears Slack, Linear and timer events only between runs, and a run can hang after a tool call returns: proofs from
05:29 to 15:33Z and circuits from 09:49Z on 18 Oct, three coordinators at once about 15:26Z on 15 Oct. Until now only Daniel
pressing Stop freed one. Every `POLL_S` the watch lists the team's agents (`GET /v1/agents`) and follows each RUNNING run's event
stream (`GET /v1/agents/{id}/runs/{run}/stream`, server-sent events). Events carry no time, so it times them as they arrive, and
a heartbeat counts as nothing. A run is

- **held** when its last event is a finished tool call and nothing has followed for `HELD_S`. The watch cancels it
  (`POST .../runs/{run}/cancel`) and starts a new run on the agent (`POST /v1/agents/{id}/runs`) that says from when it was held
  and to read its inbox. It does so at most once per agent every `RECANCEL_S`, so a run that hangs again is left for a person;
- **in a long call** when a tool call has been running for `LONG_CALL_S` (recorded, never cancelled: a build or a wait may be
  legitimate);
- **crashed** when the run ended in ERROR (recorded).

Only RUNNING is followed. A latest run left in CREATING for 10 min, or in WAITING_FOR_BACKGROUND_WORK (an agent waiting on its
background workers, where a long benchmark legitimately keeps it) for 2 h, by the API's own `updatedAt`, is recorded once as
**stuck** and never cancelled: on 18 Oct circuits' sat so from 09:49Z while every newer run was cancelled on arrival, and neither
cancelling nor a new run moved it. An agent whose latest run the runs endpoint won't read (a goal run's `run-goal-...` id gets 400)
is recorded once as unreadable and skipped.

Each finding is a line of JSON in `--log`; each sweep overwrites `status.json` beside it (its time, the ACTIVE agents, the runs it
follows), so a reader can tell a quiet watch from a dead one. The control pod holds no Slack credential, so infra's relay reads that file over ssh and
rings the agent's coordinator. Cancelling and restarting don't wait for it.
"""
from __future__ import annotations

import argparse
import base64
import calendar
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
from typing import Callable, Iterator

API = "https://api.cursor.com"
POLL_S = 60
HELD_S = 15 * 60
LONG_CALL_S = 45 * 60
RECANCEL_S = 3600
TERMINAL = {"FINISHED", "ERROR", "CANCELLED", "EXPIRED"}
# a latest run left this long in a non-terminal state that isn't RUNNING (18 Oct: circuits' runs sat in CREATING from 09:49Z and
# in WAITING_FOR_BACKGROUND_WORK from 09:49Z, and every later run was cancelled on arrival); recorded, never cancelled
STUCK = {"CREATING": 10 * 60, "WAITING_FOR_BACKGROUND_WORK": 2 * 3600}
CONTINUE = ("continue: your run was held from {since} (the agent watch saw nothing for {mins} min after a finished tool call, and "
            "cancelled it). Read your inbox and carry on from where you stopped.")


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


class Watch:
    def __init__(self, api, log: Path, clock: Callable[[], float] = time.time):
        self.api, self.log, self.clock = api, log, clock
        self.runs: dict[str, Run] = {}
        self.freed: dict[str, float] = {}           # agent -> when the watch last cancelled one of its runs
        self.crashed: set[str] = set()              # runs recorded as crashed
        self.lock = threading.Lock()

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
        if not r.calls and r.last_kind == "tool_done" and t - r.last_t >= HELD_S:
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
        """One pass: follow every RUNNING run not yet followed, then judge every followed run."""
        t = self.clock()
        live = {}
        for a in self.api.agents():
            run = a.get("latestRunId")
            if run and a.get("status") == "ACTIVE":
                live[run] = a
        for run, a in live.items():
            with self.lock:
                known = run in self.runs
            if known:
                continue
            try:
                info = self.api.get(f"/v1/agents/{a['id']}/runs/{run}")
            except urllib.error.HTTPError as e:     # a goal run's id (run-goal-...) isn't one the runs endpoint takes
                if run not in self.crashed:
                    self.crashed.add(run)
                    self.record("unreadable", Run(a["id"], run, a.get("name") or ""), status=f"HTTP {e.code}")
                continue
            status = info.get("status")
            if status == "RUNNING":
                r = Run(a["id"], run, a.get("name") or "", last_t=t)
                with self.lock:
                    self.runs[run] = r
                start_stream(r)
            elif status in STUCK and run not in self.crashed:
                since = epoch(info.get("updatedAt") or info.get("createdAt"))
                if since is not None and t - since >= STUCK[status]:
                    self.crashed.add(run)
                    self.record("stuck", Run(a["id"], run, a.get("name") or ""), status=status, since=iso(since),
                                minutes=int((t - since) // 60))
            elif info.get("status") == "ERROR" and run not in self.crashed:
                self.crashed.add(run)
                self.record("crashed", Run(a["id"], run, a.get("name") or ""), status="ERROR")
        with self.lock:
            runs = list(self.runs.values())
        status = self.log.with_name("status.json")
        status.write_text(json.dumps({"t": iso(t), "active": len(live), "followed": [r.run for r in runs]}) + "\n")
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
    ns = ap.parse_args(argv)
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        print("agent_watch: no CURSOR_API_KEY in the environment", file=sys.stderr)
        return 2
    watch = Watch(Api(key), ns.log)
    while True:
        try:
            watch.sweep(lambda r: follow(watch, r))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(time.time())} sweep failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
