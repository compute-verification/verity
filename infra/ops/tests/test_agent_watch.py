"""The agent watch frees a held run: RUNNING, its last event a finished tool call, then nothing but heartbeats for
15 minutes. It cancels the run, starts a 'continue' run on the agent, and records it; it never cancels a long tool call."""
import io
import json
import re
import uuid

import pytest

from verity_ops import agent_watch as W


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeApi:
    """`older[agent]` lists an agent's runs before its latest, newest first; `GET .../runs` lists the latest, then those.
    A goal run's id (run-goal-...) is listed but can't be read on its own, as on the API. `workers` (Task subagents) aren't
    listed but are read by id (`GET /v1/agents/{id}`; 404 for any other id); `gets` counts each read. `convs[agent]` is its
    conversation as `GET /v0/agents/{id}/conversation` gives it (user and assistant text only; 404 when absent). `v0[agent]`
    is its status as the legacy `GET /v0/agents/{id}` gives it, or the error that read raises; without one it is RUNNING
    for a worker v1 has ACTIVE and FINISHED for any other."""

    def __init__(self, agents, runs, older=None, workers=None, convs=None, v0=None):
        self._agents, self._runs, self.older, self.posts = agents, runs, older or {}, []
        self.workers, self.gets, self.convs = workers if workers is not None else {}, [], convs if convs is not None else {}
        self.v0 = v0 if v0 is not None else {}

    def agents(self):
        """Each agent as the list gives it, on Verity's repository unless it says otherwise (`repos`)."""
        return iter({"repos": [{"url": "https://github.com/compute-verification/verity"}], **a} for a in self._agents)

    def get(self, path, **params):
        parts = path.split("/")
        self.gets.append(path)
        if parts[-1] == "conversation":
            if parts[-2] not in self.convs:
                raise W.urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return {"id": parts[-2], "messages": [{"id": str(i), "type": k, "text": s}
                                                  for i, (k, s) in enumerate(self.convs[parts[-2]])]}
        if len(parts) == 4 and parts[1] == "v0":
            if isinstance(got := self.v0.get(parts[-1]), Exception):
                raise got
            if got is None and parts[-1] not in self.workers:
                raise W.urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return {"id": parts[-1], "status": got or ("RUNNING" if self.workers[parts[-1]].get("status") == "ACTIVE" else "FINISHED")}
        if len(parts) == 4:
            if parts[-1] not in self.workers:
                raise W.urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return dict(self.workers[parts[-1]])
        if parts[-1] == "runs":
            a = next(a for a in self._agents if a["id"] == parts[-2])
            ids = [a["latestRunId"], *self.older.get(a["id"], [])]
            items = [{"id": i, **self._runs[i]} for i in ids if i in self._runs]
            at, n = int(params.get("cursor") or 0), int(params.get("limit") or len(items))
            return {"items": items[at:at + n], **({"nextCursor": str(at + n)} if at + n < len(items) else {})}
        if parts[-1].startswith("run-goal"):
            raise W.urllib.error.HTTPError(path, 400, "Bad Request", {}, None)
        return self._runs[parts[-1]]

    def post(self, path, body=None):
        self.posts.append((path, body))
        return {"run": {"id": "run-new"}} if path.endswith("/runs") else {"id": "run-1"}


def held_tool_call(w, run="run-1", cid="c1"):
    """A tool call as the API streams it: the simple event, then its `interaction_update` twin (nearly always)."""
    w.on_event(run, "tool_call", {"callId": cid, "name": "run_terminal_cmd", "status": "running"})
    w.on_event(run, "interaction_update", {"type": "tool-call-started", "callId": cid})
    w.on_event(run, "tool_call", {"callId": cid, "name": "run_terminal_cmd", "status": "completed"})
    w.on_event(run, "interaction_update", {"type": "tool-call-completed", "callId": cid})


def setup(tmp_path, status="RUNNING"):
    api = FakeApi([{"id": "bc-a", "name": "proofs", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": status}})
    clock = Clock()
    w = W.Watch(api, tmp_path / "events.jsonl", clock)
    started = []
    w.sweep(started.append)
    return api, clock, w, started


def records(tmp_path):
    p = tmp_path / "events.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def status(tmp_path):
    """status.json's entries by agent, after checking its frame."""
    snap = json.loads((tmp_path / "status.json").read_text())
    assert snap["schema"] == W.SCHEMA and snap["poll_s"] == W.POLL_S
    ids = [e["agent_id"] for e in snap["agents"]]
    assert ids == sorted(ids)
    for e in snap["agents"]:
        assert (e["agent_status"] in W.AGENT_STATUSES and e["run_status"] in W.RUN_STATUSES
                and e["watch_state"] in W.WATCH_STATES and isinstance(e["restart_count"], int))
    return {e["agent_id"]: e for e in snap["agents"]}


def test_a_run_held_after_a_finished_tool_call_is_cancelled_and_continued_once(tmp_path):
    api, clock, w, started = setup(tmp_path)
    assert [r.run for r in started] == ["run-1"]
    assert json.loads((tmp_path / "status.json").read_text()) == {
        "schema": "verity/agent-watch-status/v1", "t": W.iso(clock.t), "poll_s": 60,
        "agents": [{"agent_id": "bc-a", "run_id": "run-1", "agent_status": "ACTIVE", "run_status": "RUNNING",
                    "watch_state": "followed", "since": None, "restart_count": 0}]}
    w.on_event("run-1", "tool_call", {"callId": "c1", "name": "Shell", "status": "running"})
    clock.t += 30
    w.on_event("run-1", "tool_call", {"callId": "c1", "name": "Shell", "status": "completed"})
    held_from = clock.t
    for _ in range(14):
        clock.t += 60
        w.on_event("run-1", "heartbeat", {})
        w.sweep(started.append)
    assert api.posts == [], "14 minutes of heartbeats is not yet held"
    clock.t = held_from + W.HELD_S
    w.sweep(started.append)
    (cancel, cont) = api.posts
    assert cancel == ("/v1/agents/bc-a/runs/run-1/cancel", None)
    assert cont[0] == "/v1/agents/bc-a/runs" and W.iso(held_from) in cont[1]["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert rec["kind"] == "held" and rec["agent"] == "bc-a" and rec["new_run"] == "run-new" and rec["since"] == W.iso(held_from)
    assert status(tmp_path)["bc-a"] == {"agent_id": "bc-a", "run_id": "run-1", "agent_status": "ACTIVE", "run_status": "RUNNING",
                                        "watch_state": "held", "since": W.iso(held_from), "restart_count": 0}


def test_a_long_tool_call_or_fresh_text_is_never_cancelled(tmp_path):
    """A long call while its agent's `updatedAt` moves is working, not stalled."""
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "c1", "name": "Shell", "status": "running"})
    call_from = clock.t
    for _ in range(W.LONG_CALL_S // 60 - 1):
        clock.t += 60
        api._agents[0]["updatedAt"] = W.iso(clock.t)
        w.sweep(started.append)
    assert status(tmp_path)["bc-a"]["watch_state"] == "followed"
    clock.t += 60
    api._agents[0]["updatedAt"] = W.iso(clock.t)
    w.sweep(started.append)
    w.sweep(started.append)
    assert api.posts == []
    assert [r["kind"] for r in records(tmp_path)] == ["long_call"], "recorded once, never cancelled"
    assert [status(tmp_path)["bc-a"][k] for k in ("watch_state", "since")] == ["long_call", W.iso(call_from)]
    w.on_event("run-1", "tool_call", {"callId": "c1", "status": "completed"})
    w.on_event("run-1", "assistant", {"text": "done with the build"})
    clock.t += W.HELD_S
    api._agents[0]["updatedAt"] = W.iso(clock.t)
    w.sweep(started.append)
    assert api.posts == [], "the last event was the agent's own text, not a finished tool call"
    assert [status(tmp_path)["bc-a"][k] for k in ("watch_state", "since")] == ["followed", None], "a long call only while it lasts"


def test_the_same_agent_is_freed_at_most_once_an_hour(tmp_path):
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "c1", "status": "completed"})
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert len(api.posts) == 2
    api._agents[0]["latestRunId"] = "run-2"
    api._runs["run-2"] = {"status": "RUNNING"}
    clock.t += 60
    w.sweep(started.append)
    w.on_event("run-2", "tool_call", {"callId": "c2", "status": "completed"})
    clock.t += W.HELD_S
    w.sweep(started.append)
    w.sweep(started.append)
    assert len(api.posts) == 2, "held again within the hour: left for a person"
    assert [r["kind"] for r in records(tmp_path)] == ["held", "held_again"]
    assert [status(tmp_path)["bc-a"][k] for k in ("run_id", "watch_state")] == ["run-2", "held_again"]


def test_an_unreadable_run_is_recorded_once_and_the_sweep_goes_on(tmp_path):
    """A goal run's id (run-goal-...) gets 400 from the runs endpoint, which must not stop the sweep."""
    api = FakeApi([{"id": "bc-g", "status": "ACTIVE", "latestRunId": "run-goal-x"},
                   {"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}})
    w, started = W.Watch(api, tmp_path / "events.jsonl", Clock()), []
    w.sweep(started.append)
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-1"]
    assert [(r["kind"], r["status"]) for r in records(tmp_path)] == [("unreadable", "HTTP 400")]
    st = status(tmp_path)
    assert {k: st["bc-g"][k] for k in ("run_id", "run_status", "watch_state", "since")} == {
        "run_id": "run-goal-x", "run_status": "unreadable", "watch_state": "unreadable", "since": W.iso(Clock().t)}
    assert st["bc-a"]["watch_state"] == "followed"


def test_a_run_unreadable_for_an_hour_is_recorded_again_once_for_a_person(tmp_path):
    """The first unreadable line is posted nowhere (a goal run is often brief); one still the latest an hour on is the relay's
    to ring top with."""
    api, clock = FakeApi([{"id": "bc-g", "status": "ACTIVE", "latestRunId": "run-goal-x"}], {}), Clock()
    w, first = W.Watch(api, tmp_path / "events.jsonl", clock), clock.t
    for _ in range(3 * 60):
        w.sweep(lambda r: None)
        clock.t += 60
    assert [(r["kind"], r.get("since"), r.get("minutes")) for r in records(tmp_path)] == [
        ("unreadable", None, None), ("unreadable", W.iso(first), 60)]
    assert status(tmp_path)["bc-g"]["since"] == W.iso(first)


def test_a_run_stuck_in_creating_or_background_work_is_recorded_once_never_cancelled(tmp_path):
    """A coordinator's latest runs can sit in CREATING and WAITING_FOR_BACKGROUND_WORK for hours."""
    clock = Clock(W.epoch("2026-10-18T18:10:00Z"))
    api = FakeApi([{"id": "bc-c", "name": "circuits", "status": "ACTIVE", "latestRunId": "run-c"},
                   {"id": "bc-b", "status": "ACTIVE", "latestRunId": "run-b"},
                   {"id": "bc-n", "status": "ACTIVE", "latestRunId": "run-n"}],
                  {"run-c": {"status": "CREATING", "createdAt": "2026-10-18T09:49:50.225Z", "updatedAt": "2026-10-18T09:49:50.225Z"},
                   "run-b": {"status": "WAITING_FOR_BACKGROUND_WORK", "updatedAt": "2026-10-18T16:20:00.000Z"},
                   "run-n": {"status": "CREATING", "updatedAt": "2026-10-18T18:05:00.000Z"}})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-b"] and api.posts == [], "a background wait is followed, a CREATING run isn't"
    assert [(r["kind"], r["agent"], r["status"], r["minutes"]) for r in records(tmp_path)] == [
        ("stuck", "bc-c", "CREATING", 500)], "background work 1 h 50 min and CREATING 5 min are not yet stuck"
    st = status(tmp_path)
    assert [(st[a]["run_status"], st[a]["watch_state"], st[a]["since"]) for a in ("bc-c", "bc-b", "bc-n")] == [
        ("CREATING", "stuck", "2026-10-18T09:49:50Z"), ("WAITING_FOR_BACKGROUND_WORK", "followed", None), ("CREATING", "ok", None)]


def test_a_background_wait_past_two_hours_shows_stuck_while_it_is_followed(tmp_path):
    clock = Clock(W.epoch("2026-10-18T18:10:00Z"))
    api = FakeApi([{"id": "bc-b", "status": "ACTIVE", "latestRunId": "run-b", "updatedAt": "2026-10-18T18:09:00.000Z"}],
                  {"run-b": {"status": "WAITING_FOR_BACKGROUND_WORK", "updatedAt": "2026-10-18T16:05:00.000Z"}})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-b"] and api.posts == []
    assert [(r["kind"], r["minutes"]) for r in records(tmp_path)] == [("stuck", 125)]
    e = status(tmp_path)["bc-b"]
    assert (e["run_status"], e["watch_state"], e["since"]) == ("WAITING_FOR_BACKGROUND_WORK", "stuck", "2026-10-18T16:05:00Z")


def test_a_background_wait_is_held_only_after_an_hour(tmp_path):
    """top, 18:07Z: a held agent can read WAITING_FOR_BACKGROUND_WORK. Cancelling one may take its local workers, so it waits 1 h."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    assert [(r.run, W.FOLLOW[r.status]) for r in started] == [("run-1", W.HELD_BG_S)]
    held_tool_call(w)
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert api.posts == []
    clock.t += W.HELD_BG_S - W.HELD_S
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]
    e = status(tmp_path)["bc-a"]
    assert (e["run_status"], e["watch_state"], e["since"]) == ("WAITING_FOR_BACKGROUND_WORK", "held", W.iso(clock.t - W.HELD_BG_S))


def test_a_busy_agent_is_not_counted_as_restarted(tmp_path):
    """18:21Z: vllm's restart got 409 agent_busy, as one of its own events had just started a run."""
    api = FakeApi([{"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "ERROR"}})
    real_post, clock = api.post, Clock()
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []

    def busy(path, body=None):
        raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
    api.post = busy
    w.sweep(started.append)
    assert records(tmp_path) == [] and "bc-a" not in w.freed and w.errors["bc-a"] == 0
    api.post = real_post
    clock.t += 60
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs"] and w.errors["bc-a"] == 1, "tried again at once, counted once"


def test_an_error_run_is_restarted_once_an_hour_and_three_times_in_a_row_at_most(tmp_path):
    """Run after run can end in ERROR within a minute."""
    api, clock, w, started = setup(tmp_path, status="ERROR")
    w.sweep(started.append)
    assert started == [] and [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs"], "restarted once, not twice in the hour"
    assert "ended in ERROR" in api.posts[0][1]["prompt"]["text"]
    e = status(tmp_path)["bc-a"]
    assert (e["run_status"], e["watch_state"], e["since"], e["restart_count"]) == ("ERROR", "crashed", W.iso(clock.t), 1)
    for n in (2, 3, 4):
        clock.t += W.RECANCEL_S
        api._agents[0]["latestRunId"] = f"run-e{n}"
        api._runs[f"run-e{n}"] = {"status": "ERROR"}
        w.sweep(started.append)
        w.sweep(started.append)
    assert len(api.posts) == 3
    assert [(r["kind"], r.get("action", "")[:9]) for r in records(tmp_path)] == [
        ("crashed", "restarted"), ("crashed", "restarted"), ("crashed", "restarted"), ("erroring", "none: lef")]
    assert status(tmp_path)["bc-a"] == {"agent_id": "bc-a", "run_id": "run-e4", "agent_status": "ACTIVE", "run_status": "ERROR",
                                        "watch_state": "erroring", "since": W.iso(clock.t), "restart_count": 3}
    api._agents[0].update({"status": "IDLE", "updatedAt": "1970-01-01T00:00:00Z", "latestRunId": "run-e5"})
    api._runs["run-e5"] = {"status": "ERROR"}
    clock.t += W.RECANCEL_S
    w.sweep(started.append)
    assert len(api.posts) == 3, "an IDLE agent not changed in 6 h isn't read"


ASKED = [("user_message", "Review the first change."), ("assistant_message", "Reviewed: one finding, in the thread.")]


def test_an_idle_agent_whose_message_never_became_a_run_is_recorded_dropped_once_for_its_owner(tmp_path):
    """A message reached a reviewer after its last run FINISHED and never became a run: the API's latest run reads FINISHED,
    so no restart sees it, and its conversation ends on that message. An agent idle since its run ended, or changed within
    `DROPPED_S`, isn't dropped and isn't asked for its conversation; one whose conversation ends with its own text ran a
    turn the API lists no run for."""
    clock = Clock(W.epoch("2026-11-06T05:30:00Z"))
    api = FakeApi([{"id": "bc-r", "name": "Reviewer review-2", "status": "IDLE", "latestRunId": "run-r",
                    "updatedAt": "2026-11-06T01:45:11Z"},
                   {"id": "bc-q", "name": "quiet", "status": "IDLE", "latestRunId": "run-q",
                    "updatedAt": "2026-11-06T01:33:00Z"},
                   {"id": "bc-t", "name": "Reviewer review-3", "status": "IDLE", "latestRunId": "run-t",
                    "updatedAt": "2026-11-06T01:45:30Z"}],
                  {"run-r": {"status": "FINISHED", "updatedAt": "2026-11-06T01:32:53Z"},
                   "run-q": {"status": "FINISHED", "updatedAt": "2026-11-06T01:32:00Z"},
                   "run-t": {"status": "FINISHED", "updatedAt": "2026-11-06T01:32:10Z"}},
                  convs={"bc-r": [*ASKED, ("user_message", "Review the second change.")],
                         "bc-q": ASKED,
                         "bc-t": [*ASKED, ("user_message", "Review the second change."), ("assistant_message", "No finding.")]})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.workers["bc-r"] = {"lane": "ci", "handle": "ci"}
    w.sweep(started.append)
    clock.t += 60
    w.sweep(started.append)
    assert [(r["kind"], r["agent"], r["since"], r["ended"], r.get("owner")) for r in records(tmp_path)] == [
        ("dropped", "bc-r", "2026-11-06T01:45:11Z", "2026-11-06T01:32:53Z", "ci")]
    assert api.posts == [], "nothing is restarted: its owner resumes it or relaunches it"
    assert [g for g in api.gets if g.endswith("/conversation")] == [f"/v0/agents/{a}/conversation" for a in ("bc-r", "bc-t")], \
        "each read once, and the one that changed within `DROPPED_S` of its run's end never"
    api._agents[0]["updatedAt"] = "2026-11-06T05:20:00Z"
    w.sweep(started.append)
    assert [r["kind"] for r in records(tmp_path)] == ["dropped", "dropped"], "another dropped message is recorded again"


def test_since_when_a_conversation_went_unread_outlives_a_restart(tmp_path):
    """The watch restarts on every deploy: a restart neither asks again at once for a conversation the API wouldn't read
    nor starts its window again."""
    clock = Clock(W.epoch("2026-11-06T02:00:00Z"))
    api = FakeApi([{"id": "bc-r", "name": "Reviewer review-2", "status": "IDLE", "latestRunId": "run-r",
                    "updatedAt": "2026-11-06T01:45:11Z"}], {"run-r": {"status": "FINISHED", "updatedAt": "2026-11-06T01:32:53Z"}})
    W.Watch(api, tmp_path / "events.jsonl", clock).sweep(lambda r: None)
    clock.t += 60
    W.Watch(api, tmp_path / "events.jsonl", clock).sweep(lambda r: None)
    assert records(tmp_path) == [] and api.gets.count("/v0/agents/bc-r/conversation") == 1
    clock.t += W.DROPPED_UNREAD_S - 60
    W.Watch(api, tmp_path / "events.jsonl", clock).sweep(lambda r: None)
    assert [(r["kind"], r["unread_minutes"]) for r in records(tmp_path)] == [("dropped", W.DROPPED_UNREAD_S // 60)]
    assert api.gets.count("/v0/agents/bc-r/conversation") == 2


def test_a_restart_seen_running_before_it_errors_still_counts_toward_the_limit(tmp_path):
    """A restarted run reads CREATING, then RUNNING for a minute before it ends in ERROR again; a sweep that saw it running
    reset the count, so a crashing agent was restarted every hour for good."""
    api, clock, w, started = setup(tmp_path, status="ERROR")
    for n in (2, 3, 4, 5):
        clock.t += W.RECANCEL_S
        api._agents[0]["latestRunId"] = f"run-e{n}"
        api._runs[f"run-e{n}"] = {"status": "RUNNING"}
        w.sweep(started.append)
        api._runs[f"run-e{n}"] = {"status": "ERROR"}
        w.runs[f"run-e{n}"].ended = "ERROR"
        clock.t += 60
        w.sweep(started.append)
    assert len(api.posts) == W.MAX_RESTARTS
    assert [r["kind"] for r in records(tmp_path)][-1] == "erroring"
    api._agents[0]["latestRunId"] = "run-ok"
    api._runs["run-ok"] = {"status": "FINISHED"}
    w.sweep(started.append)
    assert "bc-a" not in w.errors, "a run that finished ends the streak"


def test_a_run_the_api_wont_cancel_is_not_cancelled_every_sweep(tmp_path):
    """409 run_not_cancellable while the runs list still reads RUNNING (it lags the run's end): asked once, not every minute."""
    api, clock, w, started = setup(tmp_path)
    held_tool_call(w)
    calls = []

    def refuse(path, body=None):
        calls.append(path)
        raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
    api.post = refuse
    clock.t += W.HELD_S
    for _ in range(5):
        w.sweep(started.append)
        clock.t += 60
    assert calls == ["/v1/agents/bc-a/runs/run-1/cancel"] and "bc-a" not in w.freed and records(tmp_path) == []
    clock.t += W.REFUSED_S
    w.sweep(started.append)
    w.sweep(started.append)
    assert len(calls) == 1
    assert [(r["kind"], r["action"]) for r in records(tmp_path)] == [
        ("cancel_refused", "none: the API refused the cancel (409) and still reads it RUNNING")]


def test_the_stream_parses_server_sent_events_until_done(monkeypatch):
    body = (b"event: status\ndata: {\"runId\": \"run-1\", \"status\": \"RUNNING\"}\n\n"
            b"id: 1-0\nevent: tool_call\ndata: {\"callId\": \"c1\", \"name\": \"Shell\",\n"
            b"data: \"status\": \"completed\"}\n\n"
            b"event: heartbeat\ndata: {}\n\n"
            b"event: done\ndata: {}\n\n"
            b"event: assistant\ndata: {\"text\": \"never read\"}\n\n")

    class Resp:
        def __enter__(self):
            return iter(body.splitlines(keepends=True))

        def __exit__(self, *a):
            return False

    seen = []
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda req, timeout=None: seen.append(req) or Resp())
    events = list(W.Api("k").stream("bc-a", "run-1"))
    assert [e for e, _ in events] == ["status", "tool_call", "heartbeat", "done"]
    assert events[1][1] == {"callId": "c1", "name": "Shell", "status": "completed"}
    assert seen[0].full_url.endswith("/v1/agents/bc-a/runs/run-1/stream")
    assert seen[0].get_header("Authorization").startswith("Basic ")


def test_a_sweeps_reads_stop_at_its_budget_and_its_actions_still_go(monkeypatch):
    """With the API answering near its 60 s timeout, one sweep's serial reads could take 24 min, past the cloud watch's alert.
    A read past `reads_until` raises TimeoutError without asking, one before it waits at most what is left; a POST always goes."""
    class Resp:
        def __enter__(self):
            return io.BytesIO(b"{}")

        def __exit__(self, *a):
            return False

    asked = []
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda req, timeout=None: asked.append((req.get_method(), timeout)) or Resp())
    clock = Clock(100.0)
    api = W.Api("k", clock=clock)
    api.get("/v1/agents")
    api.reads_until = 130.0
    api.get("/v1/agents")
    clock.t = 130.0
    with pytest.raises(TimeoutError):
        api.get("/v1/agents")
    api.post("/v1/agents/bc-a/runs/run-1/cancel")
    assert asked == [("GET", 60), ("GET", 30.0), ("POST", 60)]


def test_the_agents_listing_halves_its_page_on_each_503_down_to_its_floor(monkeypatch):
    """GET /v1/agents can 503 at a fixed page of 25 for hours while 10 answers, which would fail every sweep."""
    class Resp:
        def __init__(self, doc):
            self.doc = doc

        def __enter__(self):
            return io.BytesIO(json.dumps(self.doc).encode())

        def __exit__(self, *a):
            return False

    limits = []

    def urlopen(req, timeout=None):
        q = W.urllib.parse.parse_qs(W.urllib.parse.urlparse(req.full_url).query)
        limits.append(n := int(q["limit"][0]))
        if n > 3:
            raise W.urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b""))
        return Resp({"items": [{"id": f"bc-{len(limits)}"}], "nextCursor": None if "cursor" in q else "c1"})

    monkeypatch.setattr(W.urllib.request, "urlopen", urlopen)
    api = W.Api("k")
    assert [a["id"] for a in api.agents()] == ["bc-3", "bc-4"]
    assert limits == [W.AGENTS_PAGE, W.AGENTS_PAGE // 2, 2, 2] and api.page == 2
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(
        W.urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b""))))
    with pytest.raises(W.urllib.error.HTTPError):
        list(api.agents())


def test_a_run_read_that_times_out_skips_its_agent_and_the_sweep_goes_on(tmp_path):
    """An IDLE agent's run read (`Watch.read`) caught only HTTP errors, so a timeout there, or a spent read budget, failed the
    whole sweep and left status.json stale."""
    class SlowApi(FakeApi):
        def get(self, path, **params):
            if path.endswith("/runs/run-i"):
                raise TimeoutError("The read operation timed out")
            return super().get(path, **params)

    clock = Clock()
    api = SlowApi([{"id": "bc-i", "status": "IDLE", "latestRunId": "run-i", "updatedAt": W.iso(clock.t)},
                   {"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-1"]
    assert status(tmp_path)["bc-a"]["watch_state"] == "followed"
    assert records(tmp_path) == [], "a timeout is not an unreadable run"


def test_a_conversation_read_that_times_out_is_short_and_not_retried_for_a_while(tmp_path):
    """A coordinator's long transcript timed out at the API's 60 s, and a few such reads spent the sweep's read budget."""
    class SlowConvApi(FakeApi):
        timeouts: list = []

        def get(self, path, timeout=None, **params):
            if path.endswith("/conversation"):
                self.timeouts.append(timeout)
                self.gets.append(path)
                raise TimeoutError("The read operation timed out")
            return super().get(path, **params)

    clock = Clock()
    api = SlowConvApi([], {})
    w = W.Watch(api, tmp_path / "events.jsonl", clock)
    assert w.conversation("bc-big") is None and api.timeouts == [W.CONV_TIMEOUT_S]
    clock.t += W.CONV_SKIP_S - 1
    assert w.wake_of("bc-big", clock.t) == {} and len(api.timeouts) == 1, "skipped while it's slow"
    clock.t += 2
    assert w.conversation("bc-big") is None and len(api.timeouts) == 2, "read again once the skip ends"


def test_api_get_takes_a_shorter_timeout_but_never_a_longer_one(monkeypatch):
    seen = []
    monkeypatch.setattr(W.urllib.request, "urlopen", lambda req, timeout=None: seen.append(timeout) or io.BytesIO(b"{}"))
    api = W.Api("k")
    api.get("/v1/x"), api.get("/v1/x", timeout=15), api.get("/v1/x", timeout=600)
    assert seen == [60, 15, 60]


def test_a_tool_calls_interaction_update_twin_doesnt_hide_a_held_run(tmp_path):
    """In live streams nearly every `tool_call` completed is followed within milliseconds by an `interaction_update`
    `tool-call-completed`. Taken for the agent's own text, it would keep any run from ever being held."""
    api, clock, w, started = setup(tmp_path)
    held_tool_call(w)
    w.on_event("run-1", "interaction_update", {"type": "token-delta", "tokens": 3})
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]


def test_the_status_resent_on_every_reconnect_is_not_activity(tmp_path):
    """The API re-sends a run's `status`, with no id, at the top of every (re)connection; it says nothing about the agent."""
    api, clock, w, started = setup(tmp_path)
    held_tool_call(w)
    held_from = clock.t
    for _ in range(3):
        clock.t += 5 * 60
        w.on_event("run-1", "status", {"runId": "run-1", "status": "RUNNING"})
    w.sweep(started.append)
    assert len(api.posts) == 2 and records(tmp_path)[0]["since"] == W.iso(held_from)


def test_background_subagents_neither_hide_a_held_run_nor_count_as_a_long_call(tmp_path):
    """A cloud `task` call stays `running` in the parent's stream while its subagent works (a coordinator often
    has several, thinking and calling tools meanwhile), and the subagent's progress streams as `tool-call-delta`. Read as the
    parent's own, no coordinator could be held, and each would be recorded as a 45-minute tool call."""
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    w.on_event("run-1", "interaction_update", {"type": "step-started", "stepId": "s2"})
    w.on_event("run-1", "thinking", {"text": "the worker is off; now the build"})
    held_tool_call(w, cid="c2")
    held_from = clock.t
    for _ in range(W.HELD_S // 60 - 1):
        clock.t += 60
        w.on_event("run-1", "interaction_update", {"type": "tool-call-delta", "callId": "t1", "taskUpdate": {}})
        w.sweep(started.append)
    assert api.posts == []
    clock.t = held_from + W.LONG_CALL_S
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]
    assert [r["kind"] for r in records(tmp_path)] == ["held"]


def test_a_subagent_the_parent_waits_on_is_a_long_call_never_held(tmp_path):
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    for _ in range(W.LONG_CALL_S // 60 + 1):
        clock.t += 60
        w.on_event("run-1", "interaction_update", {"type": "tool-call-delta", "callId": "t1", "taskUpdate": {}})
        w.sweep(started.append)
    assert api.posts == [] and [r["kind"] for r in records(tmp_path)] == ["long_call"]


def test_a_turn_that_ended_is_held_only_while_the_run_still_reads_running(tmp_path):
    """A turn ends (`turn-ended`) and the run turns FINISHED or WAITING_FOR_BACKGROUND_WORK within seconds (proofs, 19:13Z: 8 s).
    Waiting on background work after it is the normal state; still RUNNING 15 min later is a hold."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    held_tool_call(w)
    w.on_event("run-1", "interaction_update", {"type": "turn-ended"})
    clock.t += 2 * W.HELD_BG_S
    api._agents[0]["updatedAt"] = W.iso(clock.t)
    w.sweep(started.append)
    assert api.posts == []
    api2, clock2, w2, started2 = setup(tmp_path / "running")
    held_tool_call(w2)
    w2.on_event("run-1", "interaction_update", {"type": "turn-ended"})
    clock2.t += W.HELD_S
    w2.sweep(started2.append)
    assert [p for p, _ in api2.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]


def test_a_background_wait_with_working_subagents_is_left_alone_and_a_silent_one_recorded_stuck(tmp_path):
    """A coordinator's run waits on its workers, whose progress streams as `tool-call-delta`: cancelling it may take them. A wait
    with nothing at all streamed for 2 h, while its agent's `updatedAt` moves, is recorded as stuck, once, and never cancelled."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    w.on_event("run-1", "interaction_update", {"type": "step-started", "stepId": "s2"})
    held_tool_call(w, cid="c2")
    w.on_event("run-1", "interaction_update", {"type": "turn-ended"})
    for _ in range(3 * 6):
        clock.t += 10 * 60
        w.on_event("run-1", "interaction_update", {"type": "tool-call-delta", "callId": "t1", "taskUpdate": {}})
        w.sweep(started.append)
    assert api.posts == [] and records(tmp_path) == []
    clock.t += W.STUCK["WAITING_FOR_BACKGROUND_WORK"]
    api._agents[0]["updatedAt"] = W.iso(clock.t - 60)
    w.sweep(started.append)
    w.sweep(started.append)
    assert [(r["kind"], r["status"], r["minutes"]) for r in records(tmp_path)] == [("stuck", "WAITING_FOR_BACKGROUND_WORK", 120)]
    assert api.posts == []


def test_a_background_wait_on_a_suspended_vm_is_stalled_cancelled_and_continued(tmp_path):
    """A lane can sit WAITING_FOR_BACKGROUND_WORK for most of a day with its VM
    suspended: nothing streamed and its agent's `updatedAt` still. Its local subagents are gone with the VM, so the wait is
    cancelled and continued after 2 h, once an hour at most; 2 h with nothing streamed while the agent moves stays recorded only."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    api._agents[0]["updatedAt"] = W.iso(clock.t)
    held_tool_call(w)
    w.on_event("run-1", "interaction_update", {"type": "turn-ended"})
    quiet_from = clock.t
    clock.t += W.STUCK["WAITING_FOR_BACKGROUND_WORK"] - 60
    w.sweep(started.append)
    assert api.posts == [] and records(tmp_path) == [], "1 h 59 min is not yet stalled"
    clock.t += 60
    w.sweep(started.append)
    (cancel, cont) = api.posts
    assert cancel == ("/v1/agents/bc-a/runs/run-1/cancel", None)
    assert cont[0] == "/v1/agents/bc-a/runs" and "WAITING_FOR_BACKGROUND_WORK" in cont[1]["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["status"], rec["since"], rec["minutes"], rec["new_run"]) == (
        "stalled", "WAITING_FOR_BACKGROUND_WORK", W.iso(quiet_from), 120, "run-new")


def test_a_worker_finishing_is_not_the_parent_finishing_a_tool_call(tmp_path):
    """One of a waiting coordinator's workers finishes, its `task` call completing, while another goes on quietly."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    for cid in ("t1", "t2"):
        w.on_event("run-1", "tool_call", {"callId": cid, "name": "task", "status": "running"})
    w.on_event("run-1", "assistant", {"text": "two workers are off; I'll wait"})
    clock.t += 30 * 60
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "completed"})
    w.on_event("run-1", "interaction_update", {"type": "tool-call-completed", "callId": "t1"})
    clock.t += W.HELD_BG_S
    w.sweep(started.append)
    assert api.posts == [] and records(tmp_path) == []


def test_a_call_finishing_that_started_before_the_watch_saw_it_is_background_work_in_a_wait(tmp_path):
    """A background wait can read as after a finished tool call, a worker's `task`
    call whose start lay before the stream's replay; an hour without its workers' progress and the watch would cancel it."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "completed"})
    w.on_event("run-1", "tool_call", {"callId": "c9", "name": "run_terminal_cmd", "status": "completed"})
    clock.t += W.HELD_BG_S + 60
    w.sweep(started.append)
    assert api.posts == []
    api2, clock2, w2, started2 = setup(tmp_path / "running")
    w2.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "completed"})
    clock2.t += W.HELD_S
    w2.sweep(started2.append)
    assert api2.posts == [], "a worker finishing is not the parent's tool call, whatever the run reads"


def test_a_run_behind_one_cancelled_on_arrival_is_followed_and_freed(tmp_path):
    """`latestRunId` can be a run created and cancelled within 0.5 s, with the working run RUNNING behind it: the watch reads
    past the latest, and keeps a followed run when a newer one appears."""
    api = FakeApi([{"id": "bc-m", "name": "memory-accounting", "status": "ACTIVE", "latestRunId": "run-1"}],
                  {"run-1": {"status": "RUNNING"}})
    clock, started = Clock(), []
    w = W.Watch(api, tmp_path / "events.jsonl", clock)
    w.sweep(started.append)
    held_tool_call(w)
    clock.t += 60
    api._agents[0]["latestRunId"] = "run-x"
    api._runs["run-x"] = {"status": "CANCELLED"}
    api.older["bc-m"] = ["run-1"]
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-1"]
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-m/runs/run-1/cancel", "/v1/agents/bc-m/runs"]
    w2, started2 = W.Watch(api, tmp_path / "fresh" / "events.jsonl", clock), []
    w2.sweep(started2.append)
    assert [r.run for r in started2] == ["run-1"], "a fresh watch finds it too"


def test_a_run_creating_behind_the_latest_is_recorded_stuck(tmp_path):
    """A goal run RUNNING, a run CREATING behind it, and every later run cancelled on arrival: the latest is always a
    cancelled one, and the CREATING run must still be recorded as stuck."""
    clock = Clock(W.epoch("2026-10-18T19:52:00Z"))
    api = FakeApi([{"id": "bc-n", "name": "network-accounting", "status": "ACTIVE", "latestRunId": "run-z"}],
                  {"run-z": {"status": "CANCELLED"}, "run-y": {"status": "CANCELLED"},
                   "run-q": {"status": "CREATING", "createdAt": "2026-10-18T18:27:29.896Z", "updatedAt": "2026-10-18T18:27:29.896Z"},
                   "run-goal-g": {"status": "RUNNING", "updatedAt": "2026-10-18T18:27:23.430Z"}},
                  older={"bc-n": ["run-y", "run-q", "run-goal-g"]})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    w.sweep(started.append)
    assert [(r["kind"], r["run"], r["status"], r["minutes"]) for r in records(tmp_path)] == [("stuck", "run-q", "CREATING", 84)]
    assert started == [] and api.posts == [], "a goal run's stream can't be read, and the API refuses to cancel a CREATING run"


def test_a_run_left_creating_with_a_newer_one_started_is_not_stuck(tmp_path):
    """A run CREATING while one created seconds after it started and waits on background work: the CREATING one blocks
    nothing."""
    clock = Clock(W.epoch("2026-10-18T20:05:00Z"))
    api = FakeApi([{"id": "bc-m", "name": "memory-accounting", "status": "ACTIVE", "latestRunId": "run-z"}],
                  {"run-z": {"status": "CANCELLED"},
                   "run-w": {"status": "WAITING_FOR_BACKGROUND_WORK", "updatedAt": "2026-10-18T19:52:04.724Z"},
                   "run-c": {"status": "CREATING", "updatedAt": "2026-10-18T19:50:43.000Z"},
                   "run-s": {"status": "CANCELLED", "durationMs": 36567983},
                   "run-q": {"status": "CREATING", "updatedAt": "2026-10-18T08:26:05.281Z"}},
                  older={"bc-m": ["run-w", "run-c", "run-s", "run-q"]})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert records(tmp_path) == [] and [r.run for r in started] == ["run-w"]


def test_a_run_creating_behind_a_followed_run_is_queued_not_stuck(tmp_path):
    """A run CREATING behind a followed RUNNING one, every later run cancelled on arrival. The CREATING one waits its turn; the watch judges the run ahead of it."""
    clock = Clock(W.epoch("2026-10-18T20:14:00Z"))
    api = FakeApi([{"id": "bc-i", "name": "verity: infra", "status": "ACTIVE", "latestRunId": "run-z"}],
                  {"run-z": {"status": "CANCELLED"},
                   "run-c": {"status": "CREATING", "updatedAt": "2026-10-18T19:56:42.021Z"},
                   "run-r": {"status": "RUNNING", "updatedAt": "2026-10-18T19:55:58.419Z"}},
                  older={"bc-i": ["run-c", "run-r"]})
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert records(tmp_path) == [] and [r.run for r in started] == ["run-r"]


def test_a_run_behind_a_full_page_of_runs_cancelled_on_arrival_is_still_followed(tmp_path):
    """A waiting run can sit on the second page of its runs, behind 49 runs cancelled on arrival and a CREATING one: read
    one page only, the watch would lose the run and record the CREATING one stuck."""
    clock = Clock(W.epoch("2026-10-18T20:40:30Z"))
    arrivals = [f"run-x{i}" for i in range(60)]
    runs = {i: {"status": "CANCELLED", "updatedAt": "2026-10-18T20:30:00.000Z"} for i in arrivals}
    runs.update({"run-c": {"status": "CREATING", "updatedAt": "2026-10-18T20:29:28.614Z"},
                 "run-w": {"status": "WAITING_FOR_BACKGROUND_WORK", "updatedAt": "2026-10-18T20:39:00.000Z"},
                 "run-f": {"status": "FINISHED", "durationMs": 32532}, "run-old": {"status": "RUNNING"}})
    api = FakeApi([{"id": "bc-m", "name": "memory-accounting", "status": "ACTIVE", "latestRunId": arrivals[0]}], runs,
                  older={"bc-m": [*arrivals[1:30], "run-c", *arrivals[30:], "run-w", "run-f", "run-old"]})
    reads = []
    get = api.get
    api.get = lambda path, **p: reads.append(p.get("cursor")) or get(path, **p)
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-w"] and records(tmp_path) == []
    assert reads == [None, "50"], "it pages until a run that started, and no further"


#: (run, made, started, ended), in seconds after the list's first hour, newest first: follow-ups that wait in CREATING for the
#: run going, each starting when the one ahead ends, and two short runs (run-f, run-p) that started the second they were made,
#: ahead of the ones waiting, and so sit in front of the run that was going by then
QUEUED = [("run-d", 5190, None, None), ("run-f", 4586, 4586, 4907), ("run-c", 3938, None, None), ("run-b", 1904, 6477, None),
          ("run-a", 280, 4907, 6476), ("run-p", -1200, -1200, -900), ("run-0", -3600, -900, 4586)]
#: how many runs cancelled on arrival were made after each of QUEUED and before the next one up (the newest 40 by second
#: 6900), which the list holds in front of it
ARRIVALS = [40, 25, 0, 50, 50, 20, 60]


def queue_at(api, t0, s):
    """Give `api`'s one agent the runs it has `s` seconds after `t0`: those of QUEUED and ARRIVALS made by then, each in the
    state it is in then."""
    runs, ids, newer = {}, [], 6900
    for (rid, made, start, end), n in zip(QUEUED, ARRIVALS):
        for i in range(n):
            at = newer - (i + 1) * (newer - made) / (n + 1)
            if at <= s:
                runs[f"{rid}-x{i}"] = {"status": "CANCELLED", "createdAt": W.iso(t0 + at), "updatedAt": W.iso(t0 + at)}
                ids.append(f"{rid}-x{i}")
        if made <= s:
            if start is None or s < start:
                runs[rid] = {"status": "CREATING", "createdAt": W.iso(t0 + made), "updatedAt": W.iso(t0 + made)}
            elif end is None or s < end:
                runs[rid] = {"status": "RUNNING", "createdAt": W.iso(t0 + made), "updatedAt": W.iso(t0 + start)}
            else:
                runs[rid] = {"status": "FINISHED", "createdAt": W.iso(t0 + made), "updatedAt": W.iso(t0 + end),
                             "durationMs": (end - start) * 1000}
            ids.append(rid)
        newer = made
    api._runs, api._agents[0]["latestRunId"], api.older = runs, ids[0], {api._agents[0]["id"]: ids[1:]}


def test_a_follow_up_waiting_behind_a_run_further_back_than_the_newest_that_started_is_not_stuck(tmp_path):
    """A follow-up made while a run is going waits in CREATING for it, hundreds more are cancelled on arrival, and a short
    run that started ahead of the queue finishes in front of the run going. Reading back only to that finished run, the
    watch never saw the run going, followed nothing, and recorded each follow-up as stuck once it had waited 10 min. Before
    it records one, it reads on for a run to follow, and judges that."""
    clock = Clock()
    t0 = clock.t
    api = FakeApi([{"id": "bc-v", "name": "a coordinator", "status": "ACTIVE", "latestRunId": ""}], {})
    get, reads = api.get, []
    api.get = lambda path, **p: reads.append(p.get("cursor")) or get(path, **p)
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []

    def sweep_at(s):
        clock.t = t0 + s
        queue_at(api, t0, s)
        reads.clear()
        w.sweep(started.append)
        return list(reads)

    assert sweep_at(600) == [None], "no run CREATING for 10 min yet: back to the newest that started, and no further"
    assert sweep_at(2382) == [None, "50", "100"] and [r.run for r in started] == ["run-0"], \
        "run-a has waited 35 min behind run-0, which lies behind the finished run-p: read on to, and followed"
    assert sweep_at(3127) == [None, "50", "100", "150"] and [r.run for r in started] == ["run-0"], "run-b too, 20 min"
    assert sweep_at(6317) == [None, "50", "100", "150"] and [r.run for r in started] == ["run-0", "run-a"], \
        "run-d has waited 18 min; run-a, its turn come, runs at place 157, behind the finished run-f at place 54"
    assert sweep_at(6900) == [None, "50", "100"] and [r.run for r in started] == ["run-0", "run-a", "run-b"], \
        "run-b, its turn come, runs behind run-f"
    assert [(i, x["id"], x["status"]) for i, x in enumerate(api.get("/v1/agents/bc-v/runs")["items"], 1)
            if x["status"] != "CANCELLED"][:5] == [(41, "run-d", "CREATING"), (67, "run-f", "FINISHED"), (68, "run-c", "CREATING"),
                                                   (119, "run-b", "RUNNING"), (170, "run-a", "FINISHED")]
    assert records(tmp_path) == [] and api.posts == [], "nothing recorded at any of them: each waits its turn"


def test_a_run_creating_with_no_run_going_in_what_the_watch_may_read_is_still_recorded_stuck_once(tmp_path):
    """A zombie CREATING run, which the API refuses to cancel: nothing runs or waits on background work behind it, only runs
    that finished and hundreds cancelled on arrival. The watch reads on for a run to follow, `RUNS_PAGES` pages at most,
    records the run once, and from then on reads back to the newest that started as before."""
    clock = Clock()
    made = W.iso(clock.t - 20 * 60)
    arrivals = [f"run-x{i}" for i in range(W.RUNS_PAGE * W.RUNS_PAGES + 60)]
    runs = {i: {"status": "CANCELLED", "updatedAt": made} for i in arrivals}
    runs.update({"run-c": {"status": "CREATING", "createdAt": made, "updatedAt": made},
                 "run-f": {"status": "FINISHED", "durationMs": 32532}, "run-far": {"status": "RUNNING"}})
    api = FakeApi([{"id": "bc-m", "name": "a coordinator", "status": "ACTIVE", "latestRunId": arrivals[0]}], runs,
                  older={"bc-m": [*arrivals[1:3], "run-c", *arrivals[3:63], "run-f", *arrivals[63:], "run-far"]})
    get, reads = api.get, []
    api.get = lambda path, **p: reads.append(p.get("cursor")) or get(path, **p)
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    assert reads == [None, *(str(W.RUNS_PAGE * i) for i in range(1, W.RUNS_PAGES))], "every page it may read, and no more"
    assert [(r["kind"], r["run"], r["status"], r["minutes"]) for r in records(tmp_path)] == [("stuck", "run-c", "CREATING", 20)]
    assert started == [], "a run going further back than the watch may read is not one it finds"
    reads.clear()
    clock.t += 60
    w.sweep(started.append)
    assert reads == [None, "50"] and len(records(tmp_path)) == 1, "recorded once, and read on for only once"


def test_a_stream_that_ends_is_followed_again(tmp_path):
    """The stream returns on 400, 404 or 410 (410 stream_expired past the retention window); the run stayed followed with
    nothing feeding it and was never judged again."""
    api, clock, w, started = setup(tmp_path)
    (r,) = started
    api.stream = lambda agent, run, stop=None: iter(())
    W.pump(w, r)
    w.sweep(started.append)
    assert [x.run for x in started] == ["run-1", "run-1"]


def test_the_stream_of_an_ended_run_is_closed(tmp_path):
    """A run's stream carries the agent's later runs and sends no `done` once it is cancelled (proofs' run-891ea386, cancelled
    15:32Z, streamed events to 19:31Z). The watch read every such stream forever: 67 threads for 18 runs at 19:29Z, 74 at 19:51Z."""
    api, clock, w, started = setup(tmp_path)
    (r,) = started
    read = []

    def stream(agent, run, stop=lambda: False):
        yield "status", {"runId": run, "status": "CANCELLED"}
        while True:
            read.append(1)
            assert len(read) < 5, "still reading a cancelled run's stream"
            yield "assistant", {"text": "a later run's text"}
    api.stream = stream
    W.pump(w, r)
    assert read == [] and not r.streaming


def test_the_hourly_memory_survives_a_restart_of_the_watch(tmp_path):
    """The watch restarts on every deploy (several times an hour on a busy day). Its once-an-hour memory and its restart
    counts outlive a restart, so a run that holds again isn't cancelled again after each one."""
    api, clock, w, started = setup(tmp_path)
    held_tool_call(w)
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert len(api.posts) == 2
    api._agents[0]["latestRunId"] = "run-2"
    api._runs["run-2"] = {"status": "RUNNING"}
    w2, started2 = W.Watch(api, tmp_path / "events.jsonl", clock), []
    clock.t += 60
    w2.sweep(started2.append)
    held_tool_call(w2, run="run-2", cid="c2")
    clock.t += W.HELD_S
    w2.sweep(started2.append)
    assert len(api.posts) == 2 and [r["kind"] for r in records(tmp_path)] == ["held", "held_again"]


def test_a_continue_refused_after_the_cancel_is_tried_again(tmp_path):
    """Cancelling succeeds but starting the continue run gets 409 agent_busy (a run CREATING behind it, as memory-accounting's
    at 19:28Z): the agent was left cancelled, with one 'free failed' line for a person."""
    api, clock, w, started = setup(tmp_path)
    real = api.post

    def busy(path, body=None):
        if path.endswith("/runs"):
            raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
        return real(path, body)
    api.post = busy
    held_tool_call(w)
    clock.t += W.HELD_S
    w.sweep(started.append)
    api._runs["run-1"] = {"status": "CANCELLED"}
    clock.t += 60
    w.sweep(started.append)
    api.post = real
    clock.t += 60
    w.sweep(started.append)
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]
    assert [(r["kind"], r.get("new_run")) for r in records(tmp_path)] == [("held", None), ("continued", "run-new")]


def test_a_continue_refused_for_half_an_hour_is_left_for_a_person(tmp_path):
    api, clock, w, started = setup(tmp_path)
    real = api.post

    def busy(path, body=None):
        if path.endswith("/runs"):
            raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
        return real(path, body)
    api.post = busy
    held_tool_call(w)
    clock.t += W.HELD_S
    w.sweep(started.append)
    api._runs["run-1"] = {"status": "CANCELLED"}
    for _ in range(W.CONTINUE_RETRY_S // 60 + 2):
        clock.t += 60
        w.sweep(started.append)
    assert [r["kind"] for r in records(tmp_path)] == ["held", "continue_failed"]


def test_an_error_seen_in_the_stream_is_recorded_once(tmp_path):
    """vllm, 18:07Z: a `crashed` line without an action from the stream, then another from the restart."""
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "status", {"runId": "run-1", "status": "ERROR"})
    api._runs["run-1"] = {"status": "ERROR"}
    api._agents[0].update({"status": "IDLE", "updatedAt": W.iso(clock.t)})
    w.sweep(started.append)
    w.sweep(started.append)
    assert [(r["kind"], r.get("action", "")[:9]) for r in records(tmp_path)] == [("crashed", "restarted")]


def test_a_second_watch_on_the_same_directory_stops(tmp_path):
    """agent_watch.sh holds its flock but its Python child doesn't (dash opens it close-on-exec): a watch left running after
    its shell dies, or started by hand, ran beside another, each with its own once-an-hour memory."""
    first = W.lock(tmp_path / "watch.lock")
    assert first is not None and W.lock(tmp_path / "watch.lock") is None


def test_a_failing_sweep_leaves_the_last_status_byte_identical(tmp_path, monkeypatch):
    """Console reads the last valid snapshot; a sweep that raises, before or while writing, neither truncates nor replaces it."""
    api, clock, w, started = setup(tmp_path)
    before = (tmp_path / "status.json").read_bytes()

    def broken_listing():
        yield {"id": "bc-z", "status": "ACTIVE", "latestRunId": "run-z"}
        raise W.urllib.error.URLError("listing dropped")
    clock.t += 60
    api.agents = broken_listing
    with pytest.raises(W.urllib.error.URLError):
        w.sweep(started.append)
    assert (tmp_path / "status.json").read_bytes() == before

    api.agents = lambda: iter(api._agents)
    clock.t += 60

    def no_replace(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(W.os, "replace", no_replace)
    with pytest.raises(OSError):
        w.sweep(started.append)
    assert (tmp_path / "status.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"], "no temporary file left"
    monkeypatch.undo()
    w.sweep(started.append)
    assert json.loads((tmp_path / "status.json").read_text())["t"] == W.iso(clock.t)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"]


def test_a_watch_state_outside_the_enum_is_refused_and_leaves_the_last_status(tmp_path):
    api, clock, w, started = setup(tmp_path)
    before = (tmp_path / "status.json").read_bytes()
    w.states["bc-a"] = ("run-1", "sleepwalking", W.iso(clock.t))
    clock.t += 60
    with pytest.raises(ValueError, match="not one of WATCH_STATES"):
        w.sweep(started.append)
    assert (tmp_path / "status.json").read_bytes() == before


def test_every_kind_the_watch_records_is_a_watch_state():
    """A new finding (a new `self.record("...")`) must join WATCH_STATES, or status.json refuses it in production."""
    kinds = set(re.findall(r'self\.record\("(\w+)"', open(W.__file__).read()))
    assert kinds and kinds <= set(W.WATCH_STATES), sorted(kinds - set(W.WATCH_STATES))


def test_unknown_api_statuses_read_as_unknown(tmp_path):
    api = FakeApi([{"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "QUEUED"}})
    w = W.Watch(api, tmp_path / "events.jsonl", Clock())
    w.sweep(lambda r: None)
    e = status(tmp_path)["bc-a"]
    assert (e["agent_status"], e["run_status"], e["watch_state"]) == ("ACTIVE", "unknown", "ok")
    (e,) = W.snapshot(w, {"run-1": {"id": "bc-a", "status": "PAUSED"}}, Clock().t)["agents"]
    assert e["agent_status"] == "unknown"


def two_agents(tmp_path, order):
    agents = {"bc-b": {"id": "bc-b", "name": "circuits", "status": "ACTIVE", "latestRunId": "run-b"},
              "bc-a": {"id": "bc-a", "name": "proofs", "status": "IDLE", "updatedAt": "2027-01-15T08:00:00Z",
                       "latestRunId": "run-a"}}
    api = FakeApi([agents[k] for k in order], {"run-a": {"status": "FINISHED"}, "run-b": {"status": "RUNNING"}})
    w = W.Watch(api, tmp_path / "events.jsonl", Clock(W.epoch("2027-01-15T08:30:00Z")))
    w.sweep(lambda r: None)
    return w


def test_the_same_state_gives_the_same_bytes_with_agents_sorted(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    w1, w2 = two_agents(tmp_path / "x", ["bc-b", "bc-a"]), two_agents(tmp_path / "y", ["bc-a", "bc-b"])
    a, b = (tmp_path / "x" / "status.json").read_bytes(), (tmp_path / "y" / "status.json").read_bytes()
    assert a == b, "the listing's order doesn't reach the file"
    assert [e["agent_id"] for e in json.loads(a)["agents"]] == ["bc-a", "bc-b"]
    live = {a_["latestRunId"]: a_ for a_ in w1.api._agents}
    t = W.epoch("2027-01-15T08:30:00Z")
    assert W.status_bytes(W.snapshot(w1, live, t)) == W.status_bytes(W.snapshot(w1, live, t)) == a
    assert a == W.status_bytes(json.loads(a)) and a.endswith(b"\n") and b" " not in a


SAMPLE = ('{"agents":[{"agent_id":"bc-a","agent_status":"IDLE","restart_count":0,"run_id":"run-a","run_status":"FINISHED",'
          '"since":null,"watch_state":"ok"},{"agent_id":"bc-b","agent_status":"ACTIVE","restart_count":0,"run_id":"run-b",'
          '"run_status":"RUNNING","since":null,"watch_state":"followed"}],"poll_s":60,"schema":"verity/agent-watch-status/v1",'
          '"t":"2027-01-15T08:30:00Z"}\n')


def test_the_sample_status_file_is_pinned(tmp_path):
    two_agents(tmp_path, ["bc-b", "bc-a"])
    assert (tmp_path / "status.json").read_text() == SAMPLE


ALLOWED_KEYS = {"schema", "t", "poll_s", "agents", "agent_id", "run_id", "agent_status", "run_status", "watch_state", "since",
                "restart_count"}


def walk(v):
    """Every key and every scalar value in a JSON document."""
    if isinstance(v, dict):
        for k, x in v.items():
            yield "key", k
            yield from walk(x)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x)
    else:
        yield "value", v


def test_status_carries_no_prompt_error_text_name_branch_slack_or_key(tmp_path):
    key = "dummy-" "watch-key-w7"
    secret_bits = ["dummy-watch-key-w7", W.Api(key).auth.split()[1], "proofs-coordinator", "cursor/secret-branch", "<@U0SLACK>",
                   "the relay said hi", "connection reset by sekrit-host", "Bad Request run-goal sekrit", "read your inbox",
                   "Read your inbox", "ended in ERROR"]
    clock = Clock()
    api = FakeApi([{"id": "bc-h", "name": "proofs-coordinator", "status": "ACTIVE", "latestRunId": "run-h",
                    "source": {"repository": "github.com/x/y", "ref": "cursor/secret-branch"},
                    "summary": "<@U0SLACK> the relay said hi", "env": {"CURSOR_API_KEY": key}},
                   {"id": "bc-e", "name": "proofs-coordinator", "status": "ACTIVE", "latestRunId": "run-e"},
                   {"id": "bc-u", "name": "proofs-coordinator", "status": "ACTIVE", "latestRunId": "run-goal-u"}],
                  {"run-h": {"status": "RUNNING", "prompt": {"text": "read your inbox"}}, "run-e": {"status": "ERROR"}})
    api.key = key

    def get(path, **params):
        if "run-goal" in path:
            raise W.urllib.error.HTTPError(path, 400, "Bad Request run-goal sekrit", {}, None)
        if path.endswith("/runs"):
            run = next(a for a in api._agents if a["id"] == path.split("/")[-2])["latestRunId"]
            return {"items": [{"id": run, **api._runs[run]}] if run in api._runs else []}
        return api._runs[path.rsplit("/", 1)[1]]

    def post(path, body=None):
        api.posts.append((path, body))
        if path == "/v1/agents/bc-e/runs":
            raise W.urllib.error.URLError("connection reset by sekrit-host")
        return {"run": {"id": "run-new"}}
    api.get, api.post = get, post
    w, started = W.Watch(api, tmp_path / "events.jsonl", clock), []
    w.sweep(started.append)
    w.on_event("run-h", "tool_call", {"callId": "c1", "name": "Shell", "args": {"command": "echo hi"}, "status": "completed"})
    clock.t += W.HELD_S
    w.sweep(started.append)
    log = (tmp_path / "events.jsonl").read_text()
    assert "connection reset by sekrit-host" in log and "proofs-coordinator" in log, "the fixture does reach the event log"
    assert any("Read your inbox" in (b or {}).get("prompt", {}).get("text", "") for _, b in api.posts)
    raw = (tmp_path / "status.json").read_text()
    st = status(tmp_path)
    assert {a: e["watch_state"] for a, e in st.items()} == {"bc-e": "crashed", "bc-h": "held", "bc-u": "unreadable"}
    for kind, v in walk(json.loads(raw)):
        if kind == "key":
            assert v in ALLOWED_KEYS, v
        else:
            assert v is None or isinstance(v, (int, str)), v
    for bit in secret_bits:
        assert bit not in raw, bit


PSQL = r"""#!/usr/bin/env python3
import json, os, pathlib, re, sys
out = pathlib.Path(os.environ["PSQL_OUT"])
n, script = len(list(out.glob("*.sql"))), sys.stdin.read()
(out / f"{n}.sql").write_text(script)
env_ok = bool(os.environ.get("PGPASSWORD")) and os.environ.get("PGHOST") == "db.example" and os.environ.get("PGSSLMODE") == "require"
(out / f"{n}.args").write_text(" ".join(sys.argv[1:]) + "\n" + ("env-ok\n" if env_ok else ""))
if (out.parent / "fail").exists():
    sys.exit(print("psql: error: connection refused", file=sys.stderr) or 2)
db = out.parent / "rows.json"
rows = json.loads(db.read_text()) if db.exists() else {}
lit = "'((?:[^']|'')*)'"
t, body = re.search(rf"VALUES \({lit}, {lit}\) ON CONFLICT \(t\) DO NOTHING;", script).groups()
rows.setdefault(t, body.replace("''", "'"))
sel = re.search(rf"SELECT CASE WHEN body = {lit} THEN 'same' ELSE 'differs' END FROM \S+ WHERE t = {lit};", script)
print("same" if rows[sel[2]] == sel[1].replace("''", "'") else "differs")
db.write_text(json.dumps(dict(sorted(rows.items())[-int(re.search(r"LIMIT (\d+)\);", script)[1]):])))
"""
PG_URL = "postgresql://watch:dummy-not-a-secret@db.example/store?sslmode=require"


@pytest.fixture
def psql(tmp_path, monkeypatch):
    """A fake `psql` first on PATH: each call's script as `out/N.sql`, its argv (and whether libpq's env was set) as `N.args`;
    it fails while `fail` exists. Otherwise it keeps the rows (`rows.json`, t -> body) as the script's INSERT ... ON CONFLICT
    DO NOTHING, SELECT and pruning DELETE would, and prints the SELECT's `same` or `differs` as `psql -A -t` does."""
    bin_, out = tmp_path / "bin", tmp_path / "psql-out"
    bin_.mkdir()
    out.mkdir()
    (bin_ / "psql").write_text(PSQL)
    (bin_ / "psql").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{W.os.environ['PATH']}")
    monkeypatch.setenv("PSQL_OUT", str(out))
    return out


def calls(out):
    return [((out / f"{i}.sql").read_text(), (out / f"{i}.args").read_text()) for i in range(len(list(out.glob("*.sql"))))]


def test_each_sweeps_status_bytes_go_to_neon_the_table_created_once(tmp_path, psql):
    api, clock = FakeApi([{"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}}), Clock()
    w = W.Watch(api, tmp_path / "events.jsonl", clock, publish=W.Publisher(PG_URL))
    sent = []
    for _ in range(2):
        w.sweep(lambda r: None)
        sent.append((tmp_path / "status.json").read_bytes())
        clock.t += 60
    (first, args), (second, _) = calls(psql)
    assert first.startswith(W.PUBLISH_DDL) and "CREATE" not in second
    for script, body, t in ((first, sent[0], clock.t - 120), (second, sent[1], clock.t - 60)):
        assert f"VALUES ('{W.iso(t)}', '{body.decode()}') ON CONFLICT (t) DO NOTHING;" in script, "the exact v1 bytes, by t"
        assert f"ORDER BY t DESC LIMIT {W.STATUS_ROWS})" in script
    assert "-1" in args.split() and "env-ok" in args and "dummy-not-a-secret" not in args, "the URL only in libpq's env"


def test_a_published_t_is_never_overwritten_and_a_different_body_there_is_a_fault(tmp_path, psql, capsys, monkeypatch):
    """console reads a row as final once written: a second publish at the same t keeps the first body, and one
    that differs is printed as a FAULT, once per t; the same body again is silent."""
    pub = W.Publisher(PG_URL)
    first, other = b'{"schema":"verity/agent-watch-status/v1","n":1}\n', b'{"schema":"verity/agent-watch-status/v1","n":2}\n'
    pub(first, 0.0)
    pub(first, 0.0)
    assert capsys.readouterr().err == "", "the same body twice is silent"
    pub(other, 0.0)
    pub(other, 0.0)
    rows = json.loads((tmp_path / "rows.json").read_text())
    assert rows == {W.iso(0.0): first.decode()}, "the first body stays"
    (fault,) = [ln for ln in capsys.readouterr().err.splitlines()]
    assert f"FAULT: {W.STATUS_TABLE} already holds a different body at t={W.iso(0.0)}" in fault
    assert all("UPDATE" not in s for s, _ in calls(psql))
    monkeypatch.setattr(W, "STATUS_ROWS", 3)
    for i in range(5):
        pub(first, 60.0 * (i + 1))
    assert list(json.loads((tmp_path / "rows.json").read_text())) == [W.iso(60.0 * i) for i in (3, 4, 5)], "the newest kept"
    assert capsys.readouterr().err == ""


def test_a_quote_in_the_status_is_doubled(monkeypatch):
    seen = []
    monkeypatch.setattr(W.subprocess, "run",
                        lambda argv, input, **k: seen.append(input) or W.subprocess.CompletedProcess(argv, 0, "same\n"))
    W.Publisher(PG_URL)(b'{"x":"it\'s"}\n', 0.0)
    assert "'{\"x\":\"it''s\"}\n')" in seen[0]


def test_a_failed_publish_is_said_once_an_hour_and_never_stops_a_sweep(tmp_path, psql, capsys):
    (tmp_path / "fail").touch()
    api, clock = FakeApi([{"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}}), Clock()
    w = W.Watch(api, tmp_path / "events.jsonl", clock, publish=W.Publisher(PG_URL))
    for _ in range(3):
        w.sweep(lambda r: None)
        assert status(tmp_path)["bc-a"]["watch_state"] == "followed"
        clock.t += W.RECANCEL_S / 2
        w.on_event("run-1", "thinking", {"text": "still at it"})
    said = [ln for ln in capsys.readouterr().err.splitlines() if "not published" in ln]
    assert len(said) == 2 and "connection refused" in said[0] and "dummy-not-a-secret" not in "".join(said)
    assert all(s.startswith(W.PUBLISH_DDL) for s, _ in calls(psql)), "the table is created until a publish succeeds"
    (tmp_path / "fail").unlink()
    w.sweep(lambda r: None)
    w.sweep(lambda r: None)
    assert [s.startswith(W.PUBLISH_DDL) for s, _ in calls(psql)[-2:]] == [True, False]


def test_no_psql_or_a_bad_url_is_said_and_the_sweep_goes_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))
    api, clock = FakeApi([{"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}}), Clock()
    for url in (PG_URL, "mysql://x@y/z"):
        W.Watch(api, tmp_path / "events.jsonl", clock, publish=W.Publisher(url)).sweep(lambda r: None)
    err = capsys.readouterr().err
    assert "FileNotFoundError" in err and "ValueError: not a postgres URL" in err and (tmp_path / "status.json").exists()


T0 = W.epoch("2026-10-18T20:00:00Z")
CO, LANE, OLD, NEW = (f"bc-{c * 8}-{c * 4}-{c * 4}-{c * 4}-{c * 12}" for c in "cbde")
FINISHED = {"status": "FINISHED", "createdAt": "2026-10-18T19:10:00Z", "updatedAt": "2026-10-18T19:20:00Z"}


def coordinators(tmp_path, agents, runs, wake=False, bells=None, older=None, workers=None, timers=..., **kw):
    """The registry: CO holds circuits (its lane LANE posts as it), and vllm still names OLD, though NEW holds vllm's inbox
    (routes.d), as after a handover. Both declare a 15-minute duty timer (`timers`; None writes no file). The watch over `agents`
    (`older` as in FakeApi)."""
    reg, routes = tmp_path / "registry.json", tmp_path / "routes.d"
    reg.write_text(json.dumps({"handles": [
        {"handle": "circuits", "holder": f"the circuits coordinator ({CO}); its lane ({LANE}) posts as it too"},
        {"handle": "vllm", "holder": f"the vLLM coordinator ({OLD})"},
        {"handle": "console", "holder": "Daniel's desktop agent (f7772a12-d61f-4bbb-8c4f-e79e8d6d30dc)"}]}))
    routes.mkdir()
    (routes / "vllm.json").write_text(json.dumps({"name": "vllm", "holder": NEW, "closed": None}))
    (routes / "circuits.json").write_text(json.dumps({"name": "circuits", "holder": OLD, "closed": "2026-10-17T22:05Z"}))
    (routes / "kernels.json").write_text(json.dumps({"name": "kernels", "holder": LANE, "closed": None}))
    if bells is not None:
        (tmp_path / "doorbells.json").write_text(json.dumps(bells))
    if timers is not None:
        (tmp_path / "timers.json").write_text(json.dumps({"circuits": 900, "vllm": 900} if timers is ... else timers))
    api, clock = FakeApi(agents, runs, older, workers), Clock(T0)
    return api, clock, W.Watch(api, tmp_path / "events.jsonl", clock, registry=reg, routes=routes, wake=wake,
                               timers=tmp_path / "timers.json", **kw)


def test_a_handle_is_held_by_its_open_inbox_else_by_the_registry(tmp_path):
    coordinators(tmp_path, [], {})
    assert W.holders(tmp_path / "registry.json", tmp_path / "routes.d") == {CO: "circuits", NEW: "vllm"}, \
        "circuits' route is closed (the registry's first id holds it), vllm's moved; kernels isn't a registry handle"
    assert W.holders(tmp_path / "registry.json", None) == {CO: "circuits", OLD: "vllm"}
    assert W.holders(tmp_path / "missing.json", tmp_path / "routes.d") == {}


def test_a_coordinator_with_no_run_for_30_min_is_recorded_ready_to_wake_once_an_hour(tmp_path):
    """A coordinator's 15-minute duty timer didn't fire: its latest run FINISHED at 19:20Z and nothing ran until 20:00Z."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "circuits", "status": "IDLE", "latestRunId": "run-f",
                                             "updatedAt": "2026-10-18T19:20:00Z"}], {"run-f": dict(FINISHED)})
    w.sweep(lambda r: None)
    clock.t += W.RECANCEL_S - 60
    w.sweep(lambda r: None)
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["agent"], rec["handle"], rec["run"]) == ("ready_to_wake", CO, "circuits", "run-f")
    assert rec["reason"] == ("nothing has run for 40 min since its latest run went FINISHED at 2026-10-18T19:20:00Z "
                             "(its timer is every 15 min)")
    assert rec["action"] == "recorded (--wake is off)" and api.posts == [], "the flag is off: recorded, never a POST"
    assert status(tmp_path)[CO]["watch_state"] == "ready_to_wake"
    clock.t += 60
    w.sweep(lambda r: None)
    assert [r["kind"] for r in records(tmp_path)] == ["ready_to_wake"] * 2, "again once the hour is up"


def test_ready_to_wake_needs_a_holder_a_finished_turn_and_30_min(tmp_path):
    agents = [{"id": CO, "status": "IDLE", "latestRunId": "run-recent"},          # ended 25 min ago: its timer may yet fire
              {"id": "bc-x", "status": "IDLE", "latestRunId": "run-f"},            # holds no handle
              {"id": LANE, "status": "IDLE", "latestRunId": "run-f2"},             # a lane that posts as the handle
              {"id": OLD, "status": "IDLE", "latestRunId": "run-f3"}]              # the registry's stale vllm holder
    runs = {"run-recent": {**FINISHED, "updatedAt": "2026-10-18T19:35:00Z"}, "run-f": dict(FINISHED), "run-f2": dict(FINISHED),
            "run-f3": dict(FINISHED)}
    api, clock, w = coordinators(tmp_path, agents, runs)
    w.sweep(lambda r: None)
    assert records(tmp_path) == []
    api._agents[0]["latestRunId"], runs["run-c"] = "run-c", {"status": "CANCELLED", "updatedAt": "2026-10-18T19:00:00Z"}
    w.sweep(lambda r: None)
    assert records(tmp_path) == [], "a cancelled run is a person's stop (or the watch's own), not a missed timer"
    api._agents[0].update(status="ACTIVE", latestRunId="run-r")
    runs["run-r"] = {"status": "RUNNING", "createdAt": "2026-10-18T19:00:00Z", "updatedAt": "2026-10-18T19:00:00Z"}
    started = []
    w.sweep(started.append)
    clock.t += W.HELD_S - 60
    w.sweep(started.append)
    assert records(tmp_path) == [] and [r.run for r in started] == ["run-r"], "a run in the window: followed, not woken"


def test_an_archived_coordinator_is_never_woken(tmp_path):
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "ARCHIVED", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 wake=True)
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and api.posts == []


def test_a_coordinator_left_in_background_work_is_ready_to_wake(tmp_path):
    """Coordinators mostly sit in WAITING_FOR_BACKGROUND_WORK between wakes, so a missed timer leaves that."""
    run = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": "2026-10-18T19:00:00Z", "updatedAt": "2026-10-18T19:40:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-b"}], {"run-b": run})
    started = []
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-b"] and records(tmp_path) == [], "20 min since it went to background work"
    clock.t += 5 * 60
    w.on_event("run-b", "assistant", {"text": "a background shell said something"})
    clock.t += 2 * 900 - 60
    w.sweep(started.append)
    assert records(tmp_path) == [], "an event 29 min ago: counted from it, not from updatedAt"
    clock.t += 60
    w.sweep(started.append)
    (rec,) = records(tmp_path)
    assert (rec["handle"], rec["agent"], rec["run"]) == ("vllm", NEW, "run-b")
    assert rec["reason"] == ("nothing has run for 30 min since its latest run went WAITING_FOR_BACKGROUND_WORK at "
                             "2026-10-18T20:05:00Z (its timer is every 15 min)")


def test_a_background_run_a_wake_resumed_is_not_ready(tmp_path):
    """A wake may resume a background-waiting run rather than start one: its updatedAt moves, though its stream says nothing."""
    run = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": "2026-10-18T17:00:00Z", "updatedAt": "2026-10-18T19:40:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-b"}], {"run-b": run})
    for _ in range(4):
        clock.t += 15 * 60
        run["updatedAt"] = W.iso(clock.t - 60)
        w.sweep(lambda r: None)
    assert records(tmp_path) == []


def test_a_pending_doorbell_marks_an_idle_coordinator_ready_to_wake(tmp_path):
    """Its inbox rang at 19:45Z, after its latest run started; 10 min on, nothing has run. The relay writes the doorbells."""
    rang = W.epoch("2026-10-18T19:45:00Z") + 0.123
    run = {"status": "FINISHED", "createdAt": "2026-10-18T19:40:00Z", "updatedAt": "2026-10-18T19:44:00Z"}
    bells = {"t": W.iso(T0 - 60), "inboxes": {"vllm": {"agent": NEW, "last": f"{rang:.6f}"},
                                              "circuits": {"agent": CO, "last": 0.0}}}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "IDLE", "latestRunId": "run-v"},
                                            {"id": CO, "status": "IDLE", "latestRunId": "run-c"}],
                                 {"run-v": dict(run), "run-c": dict(run)}, bells=bells)
    clock.t = rang + W.DOORBELL_S - 1
    w.sweep(lambda r: None)
    assert records(tmp_path) == [], "the subscription has 10 min to start a run"
    clock.t += 1
    w.sweep(lambda r: None)
    assert [(r["handle"], r["reason"]) for r in records(tmp_path)] == [
        ("vllm", "a doorbell rang at 2026-10-18T19:45:00Z and nothing has run since")], "circuits has had no doorbell"
    assert w.doorbells == "fresh"


def test_a_doorbell_before_the_run_or_a_stale_file_marks_nothing(tmp_path):
    run = {"status": "FINISHED", "createdAt": "2026-10-18T19:50:00Z", "updatedAt": "2026-10-18T19:51:00Z"}
    before = {"t": W.iso(T0), "inboxes": {"vllm": {"agent": NEW, "last": str(W.epoch("2026-10-18T19:45:00Z"))}}}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "IDLE", "latestRunId": "run-v"}], {"run-v": run}, bells=before)
    w.sweep(lambda r: None)
    assert records(tmp_path) == [], "the run that started at 19:50Z answered the 19:45Z doorbell"
    during = {"t": W.iso(T0), "inboxes": {"vllm": {"agent": NEW, "last": str(W.epoch("2026-10-18T19:50:30Z"))}}}
    (tmp_path / "doorbells.json").write_text(json.dumps(during))
    w.sweep(lambda r: None)
    assert records(tmp_path) == [], "one rung during the turn may have been read in it (the timer rule catches one that wasn't)"
    after = {"t": W.iso(T0 - W.BELLS_STALE_S), "inboxes": {"vllm": {"agent": NEW, "last": str(W.epoch("2026-10-18T19:50:30Z"))}}}
    (tmp_path / "doorbells.json").write_text(json.dumps(after))
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and w.doorbells == "stale"


def test_wake_starts_one_run_an_hour(tmp_path):
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 wake=True)
    w.sweep(lambda r: None)
    clock.t += 60
    w.sweep(lambda r: None)
    ((path, body),) = api.posts
    assert path == f"/v1/agents/{CO}/runs" and "seems not to have fired" in body["prompt"]["text"]
    assert "research msg read --as circuits" in body["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["action"], rec["new_run"]) == ("ready_to_wake", "started a run", "run-new")
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert len(api.posts) == 2, "still idle an hour later: woken again"


def test_a_coordinator_whose_latest_run_was_cancelled_on_arrival_is_judged_by_the_run_behind_it(tmp_path):
    """A run FINISHED, then a doorbell mid-turn made a run cancelled within 0.5 s, and that is its latestRunId (a third
    of live agents read so). Judged by latestRunId, a coordinator whose timer then fails is never woken."""
    on_arrival = {"status": "CANCELLED", "createdAt": "2026-10-18T19:19:59.500Z", "updatedAt": "2026-10-18T19:20:00.000Z"}
    agents = [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-x"},
              {"id": CO, "status": "IDLE", "latestRunId": "run-y"}]                 # IDLE for long: the sweep reads only latest
    runs = {"run-x": dict(on_arrival), "run-f": dict(FINISHED), "run-y": dict(on_arrival), "run-g": dict(FINISHED)}
    api, clock, w = coordinators(tmp_path, agents, runs, older={NEW: ["run-f"], CO: ["run-g"]})
    w.sweep(lambda r: None)
    assert sorted((r["handle"], r["run"]) for r in records(tmp_path)) == [("circuits", "run-g"), ("vllm", "run-f")], \
        "40 min since each one's last run finished"
    assert {k: status(tmp_path)[NEW][k] for k in ("run_id", "run_status", "watch_state")} == {
        "run_id": "run-f", "run_status": "FINISHED", "watch_state": "ready_to_wake"}
    runs["run-x"]["durationMs"] = 60_000
    runs["run-x"].update(status="CANCELLED", updatedAt=W.iso(clock.t))
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert [r["handle"] for r in records(tmp_path)].count("vllm") == 1, "a person stopped a run that ran: not a missed timer"


def test_a_wake_refused_with_409_is_recorded_once_and_tried_again_in_an_hour(tmp_path):
    """Zombie CREATING runs make the API call an agent busy for hours. A wake refused with 409 counts as the hour's, so it
    is recorded, and not posted every sweep."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 wake=True)
    tries = []

    def busy(path, body=None):
        tries.append(path)
        raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
    api.post = busy
    for _ in range(30):
        w.sweep(lambda r: None)
        clock.t += 60
    assert len(tries) == 1, f"{len(tries)} wake POSTs in 30 min"
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["action"]) == ("ready_to_wake", "wake refused: HTTP 409 (agent busy); tried again in an hour")
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert len(tries) == 2 and len(records(tmp_path)) == 2


def test_a_restarted_watch_remembers_the_hours_wakes(tmp_path):
    """W1-3. The watch restarts on every deploy; the hour's memory of wakes lived only in memory."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 wake=True)
    w.sweep(lambda r: None)
    clock.t += 10 * 60
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                    wake=True, timers=tmp_path / "timers.json")
    again.sweep(lambda r: None)
    assert len(api.posts) == 1, "woken again 10 min later by the restarted watch"
    clock.t += W.RECANCEL_S
    again.sweep(lambda r: None)
    assert len(api.posts) == 2


def test_the_terminal_run_cache_and_the_runs_already_judged_are_kept_apart(tmp_path):
    """W1-6. W1's watch cached terminal run records in `done`, which here is the runs judged once (state.json: run -> when). One
    dict for both: an ERROR run read is never restarted, state.json fills with records, and a cached record that dropped
    `updatedAt` reads as never active."""
    agents = [{"id": CO, "status": "IDLE", "latestRunId": "run-f", "updatedAt": FINISHED["updatedAt"]},
              {"id": "bc-e", "status": "IDLE", "latestRunId": "run-e", "updatedAt": W.iso(T0)}]
    api, clock, w = coordinators(tmp_path, agents, {"run-f": dict(FINISHED), "run-e": {"status": "ERROR"}}, wake=True)
    w.sweep(lambda r: None)
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert [r["kind"] for r in records(tmp_path)] == ["crashed", "ready_to_wake", "ready_to_wake"]
    assert len({r["reason"] for r in records(tmp_path) if r["kind"] == "ready_to_wake"}) == 2, "read from the cached record"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["done"] == {"run-e": T0} and state["woken"] == {CO: T0 + W.RECANCEL_S}
    assert W.Watch(api, tmp_path / "events.jsonl", clock).done == {"run-e": T0}


def test_a_terminal_run_is_read_once(tmp_path):
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f", "updatedAt": W.iso(T0)}],
                                 {"run-f": dict(FINISHED)})
    reads = []
    get = api.get
    api.get = lambda path, **p: reads.append(path) or get(path, **p)
    for _ in range(3):
        w.sweep(lambda r: None)
    assert reads == [f"/v1/agents/{CO}/runs/run-f"]


W1_, W2_, W3_, W4_, W5_, W6_ = (f"bc-{c * 8}-{c * 4}-{c * 4}-{c * 4}-{c * 12}" for c in "123456")


def note(notes, lane, name, origin, age_s=60, lane_field=True):
    """A note in the notes clone, `age_s` old by its mtime."""
    p = notes / "lanes" / lane / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nid: {name[:-3]}\n" + (f"lane: {lane}\n" if lane_field else "") + f"kind: report\norigin: {origin}\n"
                 f"---\n\n# a note\n\norigin: {W4_} in the body is not front-matter\n")
    W.os.utime(p, (T0 - age_s, T0 - age_s))
    return p


def worker(status, updated, name="ZKS10"):
    return {"id": "?", "name": name, "status": status, "createdAt": "2026-10-18T12:00:00Z", "updatedAt": updated}


def test_workers_are_found_in_recent_notes_and_workers_d_minus_the_coordinators(tmp_path):
    """Tonight's proofs workers wrote notes whose origin names them (`ZKS10, bc-...; for LOR1 (bc-...)`); a coordinator's own
    notes, a note older than 12 h, and an id in a note's body name no worker."""
    coordinators(tmp_path, [], {})
    notes = tmp_path / "notes"
    note(notes, "proofs", "a.md", f"ZKS10, {W1_}, for LOR1 ({W2_}); code cursor/zk-frame-lean-decides-c7a9")
    note(notes, "hash", "b.md", f"red-team-hash (started by circuits {CO}), {W3_}")
    note(notes, "circuits", "c.md", f"circuits (@circuits, {CO})")
    note(notes, "proofs", "d.md", f"proofs worker ({W4_})", age_s=W.WORKER_NOTES_S)
    note(notes, "zk", "e.md", f"https://cursor.com/agents/{W6_}", lane_field=False)
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "w5.json").write_text(json.dumps({"agent_id": W5_, "lane": "zk-port", "launched_by": NEW}))
    (tmp_path / "workers.d" / "co.json").write_text(json.dumps({"agent_id": CO, "lane": "circuits", "launched_by": CO}))
    coords = W.holders(tmp_path / "registry.json", tmp_path / "routes.d")
    found = W.workers(notes, tmp_path / "workers.d", tmp_path / "registry.json", coords, T0)
    assert {a: (d["lane"], d["handle"]) for a, d in found.items()} == {
        W1_: ("proofs", None), W2_: ("proofs", None), W3_: ("hash", "circuits"), W5_: ("zk-port", "vllm"), W6_: ("zk", None)}, \
        "proofs isn't a handle of this registry; hash's worker goes to the coordinator that started it; W5 to its launcher's; " \
        "a note with no lane: its directory's"
    assert found[W1_]["heard"] == T0 - 60


def test_a_note_in_a_lanes_subdirectory_is_its_agents_activity_and_its_lanes(tmp_path):
    """ci's pool reviewers write `lanes/ci/reviewers/review-1.md`; read only one level deep, review-1 was called silent."""
    notes = tmp_path / "notes"
    note(notes, "ci", "reviewers/review-1.md", f"review-1 ({W1_})", lane_field=False)
    coordinators(tmp_path, [], {})
    coords = W.holders(tmp_path / "registry.json", tmp_path / "routes.d")
    found = W.workers(notes, tmp_path / "workers.d", tmp_path / "registry.json", coords, T0)
    assert found[W1_]["lane"] == "ci", "a note with no lane: the lane directory it is under, not its own directory"
    heard, _, lanes = W.note_times(notes, T0)
    assert heard[W1_] == T0 - 60 and lanes == {"ci": T0 - 60}


def test_a_worker_whose_message_never_became_a_run_is_recorded_dropped_for_its_lane(tmp_path):
    """A pool reviewer is a worker the list doesn't show: read by id, it is IDLE and moved 12 min after its latest run
    FINISHED, its conversation ending on the message. Its lane's handle is rung, once per `updatedAt`."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits reviewer ({W1_})")
    idle = {**worker("IDLE", "2026-10-18T19:45:00Z", name="Reviewer review-2"), "latestRunId": "run-w"}
    api, clock, w = coordinators(tmp_path, [], {"run-w": {"status": "FINISHED", "updatedAt": "2026-10-18T19:32:53Z"}},
                                 workers={W1_: idle}, notes=notes)
    api.convs[W1_] = [*ASKED, ("user_message", "Review the second change.")]
    w.sweep(lambda r: None)
    clock.t += 60
    w.sweep(lambda r: None)
    assert [(r["kind"], r["agent"], r["run"], r["since"], r["ended"], r.get("owner"), r.get("role")) for r in records(tmp_path)
            if r["kind"] == "dropped"] == [
        ("dropped", W1_, "run-w", "2026-10-18T19:45:00Z", "2026-10-18T19:32:53Z", "circuits", "worker")]


def test_a_task_worker_whose_latest_run_id_never_moves_is_never_called_dropped(tmp_path):
    """A worker resumed by Task kept a week-old latestRunId through ten runs, and each IDLE after its work read as a message that
    never became a run. Seen working with its latest API run finished, it is a Task worker."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker ({W1_})")
    st = {**worker("ACTIVE", "2026-10-18T19:40:00Z", name="CC-KEY"), "latestRunId": "run-w"}
    api, clock, w = coordinators(tmp_path, [], {"run-w": {"status": "FINISHED", "updatedAt": "2026-10-18T19:00:00Z"}},
                                 workers={W1_: st}, notes=notes)
    for update in ({"status": "ACTIVE"}, {"status": "IDLE", "updatedAt": "2026-10-18T19:45:00Z"},
                   {"updatedAt": "2026-10-18T20:05:00Z"}):
        st.update(update)
        w.sweep(lambda r: None)
        clock.t += 60
    assert [r for r in records(tmp_path) if r["kind"] == "dropped"] == []


def test_a_worker_with_api_runs_seen_active_then_idle_is_a_run_that_ended_and_a_later_message_is_dropped(tmp_path):
    """A pool reviewer's runs are the API's: its IDLE right after an ACTIVE read is the run it was doing ending; a message
    that moves it later, with no run, is dropped."""
    notes = tmp_path / "notes"
    note(notes, "ci", "a.md", f"ci reviewer ({W1_})")
    st = {**worker("ACTIVE", "2026-10-18T19:40:00Z", name="review-1"), "latestRunId": "run-w"}
    runs = {"run-w": {"status": "RUNNING", "updatedAt": "2026-10-18T19:40:00Z"}}
    api, clock, w = coordinators(tmp_path, [], runs, workers={W1_: st}, notes=notes)
    w.sweep(lambda r: None)
    runs["run-w"].update(status="FINISHED", updatedAt="2026-10-18T19:44:00Z")
    st.update(status="IDLE", updatedAt="2026-10-18T19:50:00Z")
    clock.t += 60
    w.sweep(lambda r: None)
    assert [r for r in records(tmp_path) if r["kind"] == "dropped"] == [], "it ran and stopped: nothing was dropped"
    st.update(updatedAt="2026-10-18T20:05:00Z")
    api.convs[W1_] = [*ASKED, ("user_message", "Review the second change.")]
    clock.t = W.epoch("2026-10-18T20:11:00Z")
    w.sweep(lambda r: None)
    assert [r["since"] for r in records(tmp_path) if r["kind"] == "dropped"] == ["2026-10-18T20:05:00Z"]


def drops(tmp_path):
    return [r for r in records(tmp_path) if r["kind"] == "dropped"]


def conv_reads(api, agent):
    return api.gets.count(f"/v0/agents/{agent}/conversation")


def seat(tmp_path, launched_by=CO, listed=False):
    """A seat (W1_) registered as `launched_by`'s, whose one run the API lists ended at T0 with its report, and the watch
    after its first read of it, a minute later. The API gives it by id alone, as a Task subagent, or, `listed`, on the
    agent list, as an agent launched there and resumed with Task since."""
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "seat.json").write_text(json.dumps({"agent_id": W1_, "lane": "circuits",
                                                                  "launched_by": launched_by}))
    st = {**worker("IDLE", W.iso(T0), name="SEAT-A"), "latestRunId": "run-w", **({"id": W1_} if listed else {})}
    api, clock, w = coordinators(tmp_path, [st] if listed else [], {"run-w": {"status": "FINISHED", "updatedAt": W.iso(T0)}},
                                 workers={} if listed else {W1_: st})
    api.convs[W1_] = [("user_message", "You are SEAT-A: fill the table and report back."),
                      ("assistant_message", "Filled: the table is in the note.")]
    clock.t += 60
    w.sweep(lambda r: None)
    return api, clock, w, st


@pytest.mark.parametrize(("resumed", "turn"), [(242, 136), (1461, 447)], ids=["a-short-turn", "a-long-turn"])
@pytest.mark.parametrize("seen", [True, False], ids=["its-coordinators-call-seen", "a-workers-call-never-seen"])
@pytest.mark.parametrize("listed", [False, True], ids=["read-by-id", "on-the-list"])
def test_a_turn_a_task_resume_ran_to_its_end_is_no_dropped_message(tmp_path, resumed, turn, seen, listed):
    """A seat resumed with Task ran a turn the API lists no run for: it ended more than `DROPPED_S` after the seat's one
    listed run, moved its `updatedAt`, and no read of the seat fell inside it, so the seat, IDLE and with its report back,
    was rung as a message that never became a run. Its conversation ends with its own text. That holds whoever resumed
    it (its coordinator, whose Task call the watch saw, or a worker, whose Task calls the watch never sees) and however
    the watch reads it (by id, or on the agent list, where no read of it ACTIVE would have told either)."""
    api, clock, w, st = seat(tmp_path, launched_by=CO if seen else W2_, listed=listed)
    clock.t = T0 + resumed
    if seen:
        w.child(CO, {"args": {"agentId": W1_, "description": "SEAT-A"}}, clock.t)
    clock.t += turn
    api.convs[W1_] += [("user_message", "Check the second table against the first."),
                       ("assistant_message", "Checked: they agree. Report: nothing to change.")]
    st["updatedAt"] = W.iso(clock.t)
    if seen:
        w.child(CO, {"args": {"agentId": W1_, "description": "SEAT-A"}, "status": "completed"}, clock.t)
    sweeps(w, clock, 20, step=1)
    if not listed:
        assert w.workers[W1_]["updated"] == T0 + resumed + turn and not w.workers[W1_].get("no_runs"), "read IDLE, never ACTIVE"
    assert drops(tmp_path) == []
    assert conv_reads(api, W1_) == 1, "asked once for the change, and not again"


@pytest.mark.parametrize("listed", [False, True], ids=["read-by-id", "on-the-list"])
def test_a_message_nothing_answered_is_still_dropped_once_for_its_owner(tmp_path, listed):
    """The fault the ring is for: a message reached an idle seat and became no run and no turn. Its `updatedAt` moved past
    its one run's end and nothing of its own follows the message in its conversation."""
    api, clock, w, st = seat(tmp_path, listed=listed)
    clock.t = T0 + 600
    api.convs[W1_] += [("user_message", "Check the second table against the first.")]
    st["updatedAt"] = W.iso(clock.t)
    sweeps(w, clock, 20, step=1)
    assert [(r["agent"], r["run"], r["since"], r["ended"], r["owner"], r["status"]) for r in drops(tmp_path)] == [
        (W1_, "run-w", W.iso(T0 + 600), W.iso(T0), "circuits", "IDLE")]


def test_a_dropped_message_is_judged_once_it_has_sat_and_by_one_read_of_the_conversation(tmp_path):
    """The conversation is the sign, so it is read only for an agent about to be recorded, once per change: not while the
    change is younger than `DROPPED_S` (a turn just starting has written nothing yet), and not again once judged. The
    record says what the read showed."""
    api, clock, w, st = seat(tmp_path)
    assert conv_reads(api, W1_) == 0, "idle since its run ended: never asked"
    clock.t = T0 + 600
    api.convs[W1_] += [("user_message", "Check the second table against the first.")]
    st["updatedAt"] = W.iso(clock.t)
    sweeps(w, clock, W.DROPPED_S // 60 - 1, step=1)
    assert drops(tmp_path) == [] and conv_reads(api, W1_) == 0, "four minutes on: not yet a message nothing answered"
    sweeps(w, clock, 1, step=1)
    assert [(r["status"], r.get("unanswered"), r.get("unread_minutes")) for r in drops(tmp_path)] == [("IDLE", True, None)]
    sweeps(w, clock, 30, step=1)
    assert len(drops(tmp_path)) == 1 and conv_reads(api, W1_) == 1


def test_a_conversation_the_api_wont_read_neither_drops_by_itself_nor_hides_a_dropped_message(tmp_path):
    """With the conversation unread the times alone are no dropped message: nothing is recorded, and it is read again
    every `DROPPED_RETRY_S`. One unread at every try for `DROPPED_UNREAD_S` is recorded all the same, saying for how long
    it went unread in place of what it showed, so a conversation the API never gives hides no one; one that reads in
    time is judged by what it shows."""
    api, clock, w, st = seat(tmp_path)
    clock.t = T0 + 600
    del api.convs[W1_]
    st["updatedAt"] = W.iso(clock.t)
    sweeps(w, clock, W.DROPPED_S // 60 + W.DROPPED_UNREAD_S // 60 - 1, step=1)
    assert drops(tmp_path) == [] and conv_reads(api, W1_) == W.DROPPED_UNREAD_S // W.DROPPED_RETRY_S
    sweeps(w, clock, 30, step=1)
    assert [(r["since"], r.get("unanswered"), r["unread_minutes"]) for r in drops(tmp_path)] == [
        (W.iso(T0 + 600), None, W.DROPPED_UNREAD_S // 60)]
    assert conv_reads(api, W1_) == W.DROPPED_UNREAD_S // W.DROPPED_RETRY_S + 1
    st["updatedAt"] = W.iso(clock.t)
    sweeps(w, clock, W.DROPPED_S // 60 + 2 * W.DROPPED_RETRY_S // 60, step=1)
    api.convs[W1_] = [("user_message", "Check the second table."), ("assistant_message", "Checked: they agree.")]
    sweeps(w, clock, 2 * W.DROPPED_UNREAD_S // 60, step=1)
    assert len(drops(tmp_path)) == 1, "read at a later try, it ends with the seat's own text: a turn ran"


def test_a_conversation_is_unanswered_when_nothing_of_its_agents_follows_its_last_message():
    assert W.unanswered([]) is None and W.unanswered([{"type": "assistant_message", "text": " "}]) is None
    asked, said = {"type": "user_message", "text": "do it"}, {"type": "assistant_message", "text": "done"}
    assert W.unanswered([asked]) is True and W.unanswered([said, asked]) is True
    assert W.unanswered([asked, said]) is False and W.unanswered([asked, said, {"type": "assistant_message", "text": ""}]) is False
    assert W.unanswered([asked, {"type": "assistant_message", "text": ""}, {"type": "tool_result"}]) is True, \
        "an assistant message with no text is nothing of its own"


def test_an_agent_the_platform_reports_in_error_is_crashed_and_never_dropped(tmp_path):
    """One fact, one record: an agent in error stopped mid-work, which `crashed` says, and is no message that never became
    a turn, whatever its conversation ends on. Its own status ERROR, should v1 give one, needs no further read."""
    api, clock, w, st = seat(tmp_path)
    api.workers[W2_] = {**worker("IDLE", W.iso(T0), name="SEAT-B"), "latestRunId": "run-w"}
    w.workers[W2_] = {"heard": T0, "lane": "circuits", "handle": "circuits"}
    clock.t = T0 + 600
    for agent in (W1_, W2_):
        api.convs[agent] = [("user_message", "You are a seat."), ("assistant_message", "Filled."), ("user_message", "Check it.")]
        api.workers[agent]["updatedAt"] = W.iso(clock.t)
    st["status"] = "ERROR"
    before = len(api.gets)
    sweeps(w, clock, 15, step=1)
    assert {r["agent"]: (r["kind"], r["status"]) for r in records(tmp_path)} == {W1_: ("crashed", "ERROR"), W2_: ("dropped", "IDLE")}
    assert f"/v0/agents/{W1_}" not in api.gets[before:] and conv_reads(api, W1_) == 0


def test_the_workers_heard_most_recently_are_read_first_when_more_are_new_than_a_sweep_reads(tmp_path, monkeypatch):
    """A backlog of workers heard days ago, sorted ahead by id, kept a reviewer heard today from being read for hours."""
    monkeypatch.setattr(W, "WORKERS_NEW", 1)
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("IDLE", "2026-10-18T19:00:00Z"),
                                                           W2_: worker("IDLE", "2026-10-18T19:00:00Z")})
    w.workers = {W1_: {"heard": T0 - 5 * 86400 + 60, "lane": "x"}, W2_: {"heard": T0 - 60, "lane": "x"}}
    w.sweep(lambda r: None)
    assert f"/v1/agents/{W2_}" in api.gets and f"/v1/agents/{W1_}" not in api.gets


def test_a_worker_that_goes_idle_is_recorded_once_per_transition(tmp_path):
    """A lane's workers can all stop mid-step unnoticed. A worker's updatedAt
    moves on a status change, so the IDLE read gives when it stopped."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_})")
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", "2026-10-18T19:31:00Z")}, notes=notes)
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and status(tmp_path)[W1_] == {
        "agent_id": W1_, "run_id": "", "agent_status": "ACTIVE", "run_status": "unknown", "watch_state": "ok", "since": None,
        "restart_count": 0}
    api.workers[W1_] = worker("IDLE", "2026-10-18T20:06:00Z")
    for _ in range(W.BURST_QUIET_S // 60):
        clock.t += 60
        w.sweep(lambda r: None)
    assert records(tmp_path) == [], "held a sweep or so, for others of its parent's to join"
    clock.t += 60
    w.sweep(lambda r: None)
    (rec,) = records(tmp_path)
    assert {k: rec[k] for k in ("kind", "agent", "name", "run", "lane", "handle", "at")} == {
        "kind": "worker_idle", "agent": W1_, "name": "ZKS10", "run": "", "lane": "circuits", "handle": "circuits",
        "at": "2026-10-18T20:06:00Z"}
    assert {k: status(tmp_path)[W1_][k] for k in ("agent_status", "watch_state", "since")} == {
        "agent_status": "IDLE", "watch_state": "worker_idle", "since": "2026-10-18T20:06:00Z"}
    api.workers[W1_] = worker("ACTIVE", "2026-10-18T20:10:00Z")
    w.sweep(lambda r: None)
    assert status(tmp_path)[W1_]["watch_state"] == "ok", "resumed with Task"
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                    notes=notes)
    api.workers[W1_] = worker("IDLE", "2026-10-18T20:30:00Z")
    again.sweep(lambda r: None)
    clock.t += W.BURST_QUIET_S
    again.sweep(lambda r: None)
    assert [r["at"] for r in records(tmp_path)] == ["2026-10-18T20:06:00Z", "2026-10-18T20:30:00Z"], \
        "the ACTIVE read before a restart of the watch counts (state.json)"


def test_a_worker_whose_parent_got_its_task_result_is_not_recorded_idle_and_one_resumed_later_is(tmp_path):
    """A worker that ended its turn by reporting back (its parent's Task call completed as it stopped) needs no one,
    whether the result streams in before the IDLE read or while the record is held; resumed (its Task call running again) and
    stopped without a result, it is recorded as before."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_})")
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", "2026-10-18T19:31:00Z")}, notes=notes)
    w.sweep(lambda r: None)
    clock.t += 6 * 60
    api.workers[W1_] = worker("IDLE", W.iso(clock.t - 30))
    w.child(CO, {"name": "task", "status": "completed", "args": {"agentId": W1_}}, clock.t)
    for _ in range(W.BURST_QUIET_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)
    assert records(tmp_path) == [], "reported back: nobody must act"
    w.child(CO, {"name": "task", "status": "running", "args": {"agentId": W1_}}, clock.t)
    api.workers[W1_] = worker("ACTIVE", W.iso(clock.t))
    w.sweep(lambda r: None)
    clock.t += 10 * 60
    api.workers[W1_] = worker("IDLE", W.iso(clock.t - 30))
    w.sweep(lambda r: None)
    clock.t += 60
    w.child(CO, {"name": "task", "status": "completed", "args": {"agentId": W1_}}, clock.t)
    for _ in range(W.BURST_QUIET_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)
    assert records(tmp_path) == [], "a result that streams in while the record is held drops it"
    w.child(CO, {"name": "task", "status": "running", "args": {"agentId": W1_}}, clock.t)
    api.workers[W1_] = worker("ACTIVE", W.iso(clock.t))
    w.sweep(lambda r: None)
    clock.t += 10 * 60
    api.workers[W1_] = worker("IDLE", stopped := W.iso(clock.t - 30))
    for _ in range(W.BURST_QUIET_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)
    assert [(r["kind"], r["at"]) for r in records(tmp_path)] == [("worker_idle", stopped)], "no result: rung as before"


def test_a_worker_first_seen_idle_is_not_recorded(tmp_path):
    """Idle before the watch knew it: it may have reported back hours ago."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker ({W1_})")
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("IDLE", "2026-10-18T19:00:00Z")}, notes=notes)
    w.sweep(lambda r: None)
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and status(tmp_path)[W1_]["watch_state"] == "ok"


def test_a_worker_active_past_the_limit_is_recorded_once(tmp_path):
    """W3 read ACTIVE at 23:30Z with updatedAt 19:31Z: its updatedAt doesn't move while it works."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_})")
    since = W.iso(T0 - 6 * 3600 + 120)
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", since)}, notes=notes)
    w.sweep(lambda r: None)
    clock.t += 60
    w.sweep(lambda r: None)
    assert records(tmp_path) == []
    for _ in range(5):
        clock.t += 60
        w.sweep(lambda r: None)
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["since"], rec["hours"], rec["handle"]) == ("worker_long", since, 6, "circuits")
    assert {k: status(tmp_path)[W1_][k] for k in ("watch_state", "since")} == {"watch_state": "worker_long", "since": since}
    short = W.Watch(api, tmp_path / "other" / "events.jsonl", clock, registry=tmp_path / "registry.json",
                    routes=tmp_path / "routes.d", notes=notes, worker_long_s=3600)
    (tmp_path / "other").mkdir()
    short.sweep(lambda r: None)
    assert [r["kind"] for r in records(tmp_path / "other")] == ["worker_long"], "--worker-long-h"


def test_a_worker_the_api_answers_404_is_not_read_again_and_the_sweep_goes_on(tmp_path):
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits workers ({W1_}, {W2_})")
    api, clock, w = coordinators(tmp_path, [], {}, workers={W2_: worker("ACTIVE", W.iso(T0))}, notes=notes)
    for _ in range(3):
        w.sweep(lambda r: None)
        clock.t += 60
    assert api.gets.count(f"/v1/agents/{W1_}") == 1 and api.gets.count(f"/v1/agents/{W2_}") == 3
    assert set(status(tmp_path)) == {W2_}


def test_worker_reads_are_bounded_and_a_long_idle_worker_is_read_only_when_a_note_of_its_changes(tmp_path):
    notes = tmp_path / "notes"
    ids = [f"bc-{i:08x}-0000-0000-0000-{i:012x}" for i in range(50)]
    for i, a in enumerate(ids):
        note(notes, "circuits", f"{i}.md", f"circuits worker ({a})")
    long_idle = W.iso(T0 - W.WORKER_IDLE_S)
    api, clock, w = coordinators(tmp_path, [{"id": ids[0], "status": "IDLE", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 workers={a: worker("IDLE", long_idle) for a in ids}, notes=notes)
    reads, stops = [], []
    for _ in range(4):
        before = len(api.gets)
        w.sweep(lambda r: None)
        reads.append([g for g in api.gets[before:] if g.count("/") == 3 and g.startswith("/v1/")])
        stops.append([g for g in api.gets[before:] if g.count("/") == 3 and g.startswith("/v0/")])
        clock.t += 60
    assert [len(r) for r in reads] == [W.WORKERS_NEW, W.WORKERS_NEW, 9, 0], \
        "20 first reads a sweep; a listed agent is no worker; one IDLE for 12 h isn't read again"
    assert all(len(set(r)) == len(r) for r in reads), "at most one read per worker a sweep"
    assert [[g[4:] for g in r] for r in stops] == [[g[4:] for g in r] for r in reads], \
        "and one read of how it stopped (its v0 status), with its first read: FINISHED, so never again"
    p = note(notes, "circuits", "7.md", f"circuits worker ({ids[7]}), back at work", age_s=0)
    W.os.utime(p, (clock.t, clock.t))
    w.sweep(lambda r: None)
    assert [g for g in api.gets if g.count("/") == 3][-2:] == [f"/v1/agents/{ids[7]}", f"/v0/agents/{ids[7]}"]
    clock.t += 60
    before = len(api.gets)
    w.sweep(lambda r: None)
    assert api.gets[before:] == [], "read once for the note"


def test_idle_time_wakes_only_a_handle_that_declares_a_timer_and_a_doorbell_wakes_any(tmp_path):
    """A handle with no recurring timer by design (comms wakes on its inbox, issue events and one-shot timers) is never
    woken for 'nothing has run for 31 min' when every doorbell has been handled. Here vllm plays comms
    (no timer declared) and circuits plays infra (900 s)."""
    finished = {"status": "FINISHED", "createdAt": "2026-10-18T18:55:00Z", "updatedAt": "2026-10-18T19:00:00Z"}
    agents = [{"id": NEW, "status": "IDLE", "latestRunId": "run-v"}, {"id": CO, "status": "IDLE", "latestRunId": "run-c"}]
    runs = {"run-v": dict(finished), "run-c": {**finished, "updatedAt": "2026-10-18T19:29:00Z"}}
    api, clock, w = coordinators(tmp_path, agents, runs, timers={"circuits": 900})
    w.sweep(lambda r: None)
    assert [(r["handle"], r["reason"]) for r in records(tmp_path)] == [
        ("circuits", "nothing has run for 31 min since its latest run went FINISHED at 2026-10-18T19:29:00Z "
                     "(its timer is every 15 min)")], "vllm idle 60 min, with no timer and no doorbell: nothing"
    rang = W.epoch("2026-10-18T19:40:00Z")
    (tmp_path / "doorbells.json").write_text(json.dumps({"t": W.iso(T0 - 60), "inboxes": {"vllm": {"agent": NEW, "last": rang}}}))
    clock.t += 60
    w.sweep(lambda r: None)
    assert [(r["handle"], r["reason"]) for r in records(tmp_path)][1:] == [
        ("vllm", "a doorbell rang at 2026-10-18T19:40:00Z and nothing has run since")], "a doorbell newer than its last run"


def test_with_no_timers_file_only_doorbells_wake(tmp_path):
    agents = [{"id": NEW, "status": "IDLE", "latestRunId": "run-v"}, {"id": CO, "status": "IDLE", "latestRunId": "run-c"}]
    bells = {"t": W.iso(T0 - 60), "inboxes": {"circuits": {"agent": CO, "last": W.epoch("2026-10-18T19:40:00Z")}}}
    api, clock, w = coordinators(tmp_path, agents, {"run-v": dict(FINISHED), "run-c": dict(FINISHED)}, bells=bells, timers=None)
    for _ in range(10):
        w.sweep(lambda r: None)
        clock.t += 60
    assert [(r["handle"], r["reason"][:16]) for r in records(tmp_path)] == [("circuits", "a doorbell rang ")], \
        "both idle 40-50 min: only the doorbell wakes"
    (tmp_path / "timers.json").write_text('{"vllm": "900", "circuits": -1, "infra": true}')
    assert W.timers(tmp_path / "timers.json") == {} and W.timers(None) == {}, "only positive numbers declare a timer"


def test_every_declared_timer_is_a_registry_handle_with_a_positive_interval():
    """A lane opts in to the idle rule by adding `{handle: interval_s}` to infra/control/agent_watch_timers.json."""
    root = W.Path(__file__).resolve().parents[3]
    raw = json.loads((root / "infra/control/agent_watch_timers.json").read_text())
    handles = {h["handle"] for h in json.loads((root / ".agents/skills/using-slack/registry.json").read_text())["handles"]}
    assert raw and set(raw) <= handles, f"not registry handles: {sorted(set(raw) - handles)}"
    assert W.timers(root / "infra/control/agent_watch_timers.json") == {h: float(v) for h, v in raw.items()}, \
        "each interval a positive number of seconds"


def test_an_agent_not_on_veritys_repository_is_never_acted_on_or_shown(tmp_path):
    """The key sees every agent on the account, including a person's own (repos null). One on another repository, or none, is never streamed, cancelled, restarted or woken, and isn't a status row."""
    other = [{"url": "https://github.com/someone/calendly-helper"}]
    agents = [{"id": "bc-run", "status": "ACTIVE", "latestRunId": "run-1", "repos": other},
              {"id": "bc-err", "status": "IDLE", "latestRunId": "run-e", "updatedAt": W.iso(T0), "repos": None},
              {"id": CO, "status": "IDLE", "latestRunId": "run-f", "updatedAt": "2026-10-18T19:20:00Z",
               "repos": [{"url": "https://github.com/compute-verification/research-notes"}]},
              {"id": "bc-ok", "status": "ACTIVE", "latestRunId": "run-2"}]
    runs = {"run-1": {"status": "RUNNING"}, "run-e": {"status": "ERROR"}, "run-f": dict(FINISHED), "run-2": {"status": "RUNNING"}}
    api, clock, w = coordinators(tmp_path, agents, runs, wake=True)
    started = []
    w.sweep(started.append)
    for run in ("run-1", "run-2"):
        held_tool_call(w, run)
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-2"]
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-ok/runs/run-2/cancel", "/v1/agents/bc-ok/runs"], \
        "the Verity agent alone is freed: nothing restarted, cancelled or woken on the others"
    assert [r["agent"] for r in records(tmp_path)] == ["bc-ok"] and set(status(tmp_path)) == {"bc-ok"}
    assert not any(g.startswith(("/v1/agents/bc-run", "/v1/agents/bc-err", f"/v1/agents/{CO}")) for g in api.gets)


def test_a_worker_whose_repos_arent_veritys_is_never_acted_on(tmp_path):
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits workers ({W1_}, {W2_}, {W3_})")
    listed_elsewhere = {"id": W3_, "status": "ACTIVE", "latestRunId": "run-x", "repos": None}
    api, clock, w = coordinators(tmp_path, [listed_elsewhere], {"run-x": {"status": "RUNNING"}}, notes=notes, workers={
        W1_: {**worker("ACTIVE", W.iso(T0 - 7 * 3600)), "repos": [{"url": "https://github.com/someone/else"}]},
        W2_: worker("ACTIVE", W.iso(T0 - 7 * 3600))}, worker_long_s=3600)
    for _ in range(3):
        w.sweep(lambda r: None)
        clock.t += 60
    assert api.gets.count(f"/v1/agents/{W1_}") == 1 and f"/v1/agents/{W3_}" not in api.gets, \
        "read once, then never; an agent the list has (even not Verity's) is no worker"
    assert [(r["kind"], r["agent"]) for r in records(tmp_path)] == [("worker_long", W2_)] and set(status(tmp_path)) == {W2_}


class FakeSh:
    """`git ls-remote --heads` answers `heads` (branch -> sha), failing while `fail` holds it; ssh to a node runs RUNS_READER here,
    on the runs root its machines.d entry names (a tmp directory), as `python3 - ROOT SINCE` would there. `calls` counts each."""

    def __init__(self, heads=None):
        self.heads, self.fail, self.calls = dict(heads or {}), set(), []

    def __call__(self, argv, stdin, timeout):
        self.calls.append(argv)
        if argv[:2] == ["git", "ls-remote"]:
            if "git" in self.fail:
                return 128, "", "fatal: Could not read from remote repository."
            return 0, "".join(f"{sha}\trefs/heads/{b}\n" for b, sha in sorted(self.heads.items())), ""
        host = argv[-2].split("@", 1)[1]
        if host in self.fail:
            return 255, "", f"ssh: connect to host {host} port 22: Connection timed out"
        _, dash, root, since = W.shlex.split(argv[-1])
        r = W.subprocess.run([W.sys.executable, dash, root, since], input=stdin, capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr


REMOTE = "git@github.com:compute-verification/verity.git"


def sweeps(w, clock, minutes, step=5, beat=True):
    """A sweep every `step` minutes for `minutes`, each followed run's stream giving a heartbeat before it (`beat`): the
    stream is being received, as on the pod."""
    for _ in range(minutes // step):
        clock.t += step * 60
        for run in list(w.runs) if beat else ():
            w.on_event(run, "heartbeat", {})
        w.sweep(lambda r: None)


def silent(tmp_path):
    return [r for r in records(tmp_path) if r["kind"] == "silent"]


def test_a_silent_worker_is_rung_once_then_once_more_a_window_later_then_not_until_it_shows_activity(tmp_path):
    """Agents go quiet and nobody notices. A worker ACTIVE 90 min with no note of
    its and no push to its branch rings its coordinator's handle; a second silent window rings again; then nothing until a
    note or a push."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    sh = FakeSh({"cursor/w3-branch-c7a9": "a" * 40, "main": "m" * 40})
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    sweeps(w, clock, 85)
    assert silent(tmp_path) == [], "a fresh watch counts from its first heads read: nobody is silent for 90 min"
    sweeps(w, clock, 5)
    (rec,) = silent(tmp_path)
    assert {k: rec[k] for k in ("agent", "name", "run", "handle", "lane", "since", "minutes", "active_minutes", "window",
                                "branches")} == {
        "agent": W1_, "name": "ZKS10", "run": "", "handle": "circuits", "lane": "circuits", "since": W.iso(T0), "minutes": 90,
        "active_minutes": 270, "window": 1, "branches": ["cursor/w3-branch-c7a9"]}
    sweeps(w, clock, 90)
    assert [(r["window"], r["minutes"]) for r in silent(tmp_path)] == [(1, 90), (2, 180)]
    sweeps(w, clock, 180)
    assert len(silent(tmp_path)) == 2, "two rings a spell"
    p = note(notes, "circuits", "b.md", f"circuits worker W3 ({W1_}), checkpoint", age_s=0)
    W.os.utime(p, (clock.t, clock.t))
    sweeps(w, clock, 85)
    assert len(silent(tmp_path)) == 2, "a new note is activity"
    sweeps(w, clock, 5)
    assert [r["window"] for r in silent(tmp_path)] == [1, 2, 1] and silent(tmp_path)[-1]["since"] == W.iso(p.stat().st_mtime)


def test_a_note_naming_an_agent_by_its_short_id_is_its_activity(tmp_path):
    """Notes often sign with the short id (`ROLE (bc-xxxxxxxx)`): a full-id-only match rang an agent silent though its
    checkpoint was half an hour old."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    sh = FakeSh({"cursor/w3-branch-c7a9": "a" * 40, "main": "m" * 40})
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    sweeps(w, clock, 90)
    assert [r["window"] for r in silent(tmp_path)] == [1]
    p = note(notes, "circuits", "b.md", f"circuits worker W3 ({W.short_id(W1_)}), checkpoint", age_s=0)
    W.os.utime(p, (clock.t, clock.t))
    sweeps(w, clock, 85)
    assert len(silent(tmp_path)) == 1, "a note naming the short id is activity"
    sweeps(w, clock, 5)
    assert [r["window"] for r in silent(tmp_path)] == [1, 1] and silent(tmp_path)[-1]["since"] == W.iso(p.stat().st_mtime)


def test_a_push_to_the_branch_a_worker_moved_its_task_to_is_its_activity(tmp_path):
    """A worker that moved from its first branch to a successor no note or record names was rung as pushing nothing. A head
    keeping its branch's stem and suffix is its push; a lane-only stem, or another task's branch, is not."""
    notes = tmp_path / "notes"
    note(notes, "lean", "a.md", f"lean worker ONE18 ({W1_}); code `cursor/lean-one18-fc8d`", age_s=2 * 3600)
    first, moved = "cursor/lean-one18-fc8d", "cursor/lean-one18-v2-fc8d"
    sh = FakeSh({first: "a" * 40, moved: "b" * 40, "cursor/lean-other-fc8d": "c" * 40})
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    for _ in range(6):
        sweeps(w, clock, 60)
        sh.heads[moved] = f"{clock.t:040.0f}"
    assert silent(tmp_path) == [], "a push to its moved branch every hour is activity"
    for _ in range(2):
        sweeps(w, clock, 60)
        sh.heads["cursor/lean-other-fc8d"] = f"{clock.t:040.0f}"
    (rec,) = silent(tmp_path)
    assert rec["branches"] == [first, moved], "another task's branch is not its push"
    assert W.same_task(moved, first) and not W.same_task("cursor/lean-fc8d-x-fc8d", "cursor/lean-fc8d") \
        and not W.same_task("cursor/lean-one18-fc8d", "cursor/lean-one18-9f2a")


def test_a_research_run_a_worker_sent_is_its_activity_at_its_start_and_at_its_end(tmp_path):
    """lean and circuits workers waiting on two-hour checks were rung silent while their runs ran. The start of a run the
    worker sent, read while it runs, and its end are its activity; reading it while it runs doesn't use up its end's ring."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    (notes / "machines.d").mkdir(parents=True)
    root = tmp_path / "vy-nebius-1"
    (notes / "machines.d" / "vy-nebius-1.toml").write_text(
        f'provider = "ssh"\nhost = "vy-nebius-1.example"\nuser = "research"\nproject = "verity"\nroot = "{root}"\n')
    (tmp_path / "nodes.json").write_text(json.dumps({"nodes": ["vy-nebius-1"], "every_s": 300, "ssh": ["ssh"]}))
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=FakeSh({"cursor/w3-branch-c7a9": "a" * 40}),
                                 nodes=tmp_path / "nodes.json", timers=None)
    w.sweep(lambda r: None)
    sweeps(w, clock, 60)
    rid, started = "r20261018-200000-0001", clock.t
    node_run(root, rid, started + 3600, by=f"https://cursor.com/agents/{W1_}", wall_s=3600)
    status_json = root / "runs" / rid / "status.json"
    status_json.write_text(json.dumps({"state": "running", "transitions": [{"state": "running", "t_utc": W.iso(started)}]}))
    W.os.utime(status_json, (started, started))
    sweeps(w, clock, 85)
    assert silent(tmp_path) == [] and ended(tmp_path) == [], "the run's start is its sender's activity, and not an end"
    sweeps(w, clock, 10)
    assert [r["since"] for r in silent(tmp_path)] == [W.iso(started)]
    status_json.write_text(json.dumps({"state": "done", "rc": 0, "wall_s": clock.t - started, "transitions": [
        {"state": "running", "t_utc": W.iso(started)}, {"state": "done", "t_utc": W.iso(clock.t)}]}))
    W.os.utime(status_json, (clock.t, clock.t))
    finished = clock.t
    sweeps(w, clock, 85)
    assert len(silent(tmp_path)) == 1 and [r["run"] for r in ended(tmp_path)] == [rid], "its end is activity, and rings once"
    sweeps(w, clock, 10)
    assert [r["since"] for r in silent(tmp_path)][-1] == W.iso(finished)


def test_a_push_to_a_pr_branch_whose_body_names_a_worker_is_its_activity(tmp_path):
    """lean's workers push to `cursor/lean-...-fc8d` branches, which match none of a worker's known branches, and were rung
    silent while pushing. A push to the branch of an open PR whose body names the worker, by full or short id, is its activity."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    sh = FakeSh({"cursor/w3-branch-c7a9": "a" * 40, "cursor/lean-records-fc8d": "b" * 40})
    prs = [{"branch": "cursor/lean-records-fc8d", "body": f"Lane lean\n\nRECORDS ({W.short_id(W1_)}) re-heads it."}]
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh, open_prs=lambda: prs)
    w.sweep(lambda r: None)
    sweeps(w, clock, 60)
    sh.heads["cursor/lean-records-fc8d"] = "c" * 40
    pushed = clock.t + 5 * 60
    sweeps(w, clock, 90)
    assert silent(tmp_path) == [], "the push to its PR's branch is activity"
    sweeps(w, clock, 5)
    (rec,) = silent(tmp_path)
    assert rec["since"] == W.iso(pushed) and "cursor/lean-records-fc8d" in rec["branches"]
    w.open_prs = lambda: (_ for _ in ()).throw(OSError("HTTP 502"))
    w.prs_read = -float("inf")
    w.read_prs(clock.t)
    assert w.pr_branches == {W.short_id(W1_): {"cursor/lean-records-fc8d"}}, "a failed read keeps the last one"


def test_a_worker_whose_branches_nothing_names_takes_its_lanes_pushes_as_its_activity(tmp_path):
    """A worker was archived as silent 40 min after pushing its PR's `cursor/<lane>-...` branch: a Task-launched worker has no
    `git.branches`, the PR body named only `Lane circuits`, and the branch's suffix isn't its id's. With no branch
    of its known, pushes to its lane's branches (`cursor/<lane>-*`, and open PRs whose body says `Lane <lane>`) count."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker MOEFUSE ({W1_})", age_s=2 * 3600)
    sh = FakeSh({"cursor/circuits-cir193-6f04": "a" * 40, "feature/served-rows": "b" * 40, "cursor/lean-x-2222": "c" * 40})
    prs = [{"branch": "feature/served-rows", "body": "Lane circuits\n\nServed sparse-MLA rows."}]
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh, open_prs=lambda: prs)
    w.sweep(lambda r: None)
    sweeps(w, clock, 60)
    sh.heads["cursor/circuits-cir193-6f04"] = "d" * 40
    sweeps(w, clock, 60)
    sh.heads["feature/served-rows"] = "e" * 40
    pushed = clock.t + 5 * 60
    sweeps(w, clock, 60)
    sh.heads["cursor/lean-x-2222"] = "f" * 40
    sweeps(w, clock, 20)
    assert silent(tmp_path) == [], "its lane's pushes are its activity; another lane's isn't"
    sweeps(w, clock, 15)
    (rec,) = silent(tmp_path)
    assert rec["since"] == W.iso(pushed) and set(rec["branches"]) == {"cursor/circuits-cir193-6f04", "feature/served-rows"}


def test_a_worker_whose_branches_nothing_names_takes_its_lanes_runs_as_its_activity(tmp_path):
    """MOEFUSE's runs went under its coordinator's id (`--by circuits`), so they were no activity of its: with no branch of its
    known, a run an agent of its lane sent counts as its activity."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker MOEFUSE ({W1_})", age_s=2 * 3600)
    note(notes, "circuits", "b.md", f"circuits worker W2 ({W2_}) on cursor/w2-x-c7a9", age_s=2 * 3600)
    (notes / "machines.d").mkdir(parents=True)
    root = tmp_path / "vy-nebius-1"
    (notes / "machines.d" / "vy-nebius-1.toml").write_text(
        f'provider = "ssh"\nhost = "vy-nebius-1.example"\nuser = "research"\nproject = "verity"\nroot = "{root}"\n')
    (tmp_path / "nodes.json").write_text(json.dumps({"nodes": ["vy-nebius-1"], "every_s": 300, "ssh": ["ssh"]}))
    active = worker("ACTIVE", W.iso(T0 - 3 * 3600))
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: active, W2_: active}, notes=notes, git_remote=REMOTE,
                                 sh=FakeSh({"cursor/w2-x-c7a9": "a" * 40}), nodes=tmp_path / "nodes.json", timers=None)
    w.sweep(lambda r: None)
    sweeps(w, clock, 60)
    rid, started = "r20261018-200000-0002", clock.t
    node_run(root, rid, started + 3600, by=f"https://cursor.com/agents/{W2_}", wall_s=3600)
    status_json = root / "runs" / rid / "status.json"
    status_json.write_text(json.dumps({"state": "running", "transitions": [{"state": "running", "t_utc": W.iso(started)}]}))
    W.os.utime(status_json, (started, started))
    sweeps(w, clock, 85)
    assert W1_ not in {r["agent"] for r in silent(tmp_path)}, "a run of its lane's is its activity"
    sweeps(w, clock, 10)
    assert [r["since"] for r in silent(tmp_path) if r["agent"] == W1_] == [W.iso(started)], \
        "its silence counts from its lane's run"


def test_github_prs_reads_every_page_of_open_prs_with_the_token_from_its_file(tmp_path, monkeypatch):
    env = tmp_path / "relay.env"
    env.write_text("SLACK_BOT_TOKEN=x\nGH_TOKEN=ghs_dummy_for_tests\n")
    seen = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.data).encode()

    def urlopen(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization")))
        page = int(req.full_url.rsplit("page=", 1)[1])
        return Resp([{"head": {"ref": f"b{page}-{i}"}, "body": "x"} for i in range(100 if page == 1 else 3)])

    monkeypatch.setattr(W.urllib.request, "urlopen", urlopen)
    got = W.github_prs(REMOTE, env)()
    assert len(got) == 103 and got[0] == {"branch": "b1-0", "body": "x"}
    assert [u for u, _ in seen] == [
        f"https://api.github.com/repos/compute-verification/verity/pulls?state=open&per_page=100&page={p}" for p in (1, 2)]
    assert {a for _, a in seen} == {"Bearer ghs_dummy_for_tests"}
    with pytest.raises(ValueError):
        W.github_prs("/srv/git/verity.git", env)


def test_a_silent_worker_still_showing_nothing_six_hours_on_is_one_no_step_for_top(tmp_path):
    """Workers can sit ACTIVE with no step for days, each rung twice as silent and then by no one."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    sh = FakeSh({"cursor/w3-branch-c7a9": "a" * 40})
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    sweeps(w, clock, 359)
    assert [r["kind"] for r in records(tmp_path) if r["kind"] in ("silent", "no_step", "archived")] == ["silent", "silent"]
    assert not [p for p, _ in api.posts if p.endswith("/archive")]
    sweeps(w, clock, 1)
    sweeps(w, clock, 600)
    got = [r for r in records(tmp_path) if r["kind"] in ("no_step", "archived")]
    assert [{k: r.get(k) for k in ("kind", "agent", "since", "hours", "handle", "owner", "role")} for r in got] == [
        {"kind": "archived", "agent": W1_, "since": W.iso(T0), "hours": 6, "handle": "circuits", "owner": None, "role": "worker"}]
    assert [p for p, _ in api.posts if p.endswith("/archive")] == [f"/v1/agents/{W1_}/archive"]


def test_a_coordinator_working_with_no_note_or_push_is_silent_but_never_no_step(tmp_path):
    """A coordinator working only in Slack makes no note or push for hours, and no_step must not tell top to archive it. Its agent's `updatedAt` moving is activity to no_step,
    not to silent; once that stops too, no_step counts from it."""
    run = {"status": "RUNNING", "createdAt": W.iso(T0), "updatedAt": W.iso(T0)}
    sh = FakeSh({"main": "m" * 40})
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-r"}],
                                 {"run-r": run}, notes=tmp_path / "notes", git_remote=REMOTE, sh=sh)
    (tmp_path / "notes").mkdir()
    sweep = w.sweep

    def working(start):
        api._agents[0]["updatedAt"] = W.iso(clock.t)
        sweep(start)
    w.sweep = working
    w.sweep(lambda r: None)
    sweeps(w, clock, 8 * 60)
    kinds = [r["kind"] for r in records(tmp_path) if r["kind"] in ("silent", "no_step")]
    assert kinds == ["silent", "silent"], kinds
    w.sweep = sweep
    stopped = clock.t
    sweeps(w, clock, 6 * 60)
    got = [r for r in records(tmp_path) if r["kind"] == "no_step"]
    assert [(r["agent"], r["hours"], r["owner"], r["role"]) for r in got] == [(CO, 6, "top", "coordinator")]
    assert W.epoch(got[0]["since"]) >= stopped - 5 * 60


def test_a_coordinators_lane_note_is_its_activity(tmp_path):
    """A coordinator syncing notes signed with its lane, not its id, is not silent. A note under `lanes/<handle>/` counts for that handle's coordinator."""
    run = {"status": "RUNNING", "createdAt": W.iso(T0), "updatedAt": W.iso(T0)}
    notes = tmp_path / "notes"
    sh = FakeSh({"main": "m" * 40})
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-r"}],
                                 {"run-r": run}, notes=notes, git_remote=REMOTE, sh=sh)
    (notes / "lanes").mkdir(parents=True)
    w.sweep(lambda r: None)
    for _ in range(12):
        sweeps(w, clock, 20)
        p = note(notes, "circuits", f"c{clock.t:.0f}.md", "circuits coordinator", age_s=0)
        W.os.utime(p, (clock.t, clock.t))
    assert [r for r in silent(tmp_path) if r["agent"] == CO] == [], "its lane's notes every 20 min are activity"
    sweeps(w, clock, 65)
    assert [r["agent"] for r in silent(tmp_path)] == [CO], "with no note for an hour it is silent again"


def test_a_push_to_an_agents_branch_is_activity(tmp_path):
    """A coordinator's branches: its runs' `git.branches` and every head ending in its id's last four (`cursor/...-cccc`)."""
    run = {"status": "RUNNING", "createdAt": W.iso(T0), "updatedAt": W.iso(T0),
           "git": {"branches": [{"repoUrl": "github.com/compute-verification/verity", "branch": "cursor/api-named"},
                                {"repoUrl": "github.com/someone/else", "branch": "cursor/not-veritys"}]}}
    sh = FakeSh({"cursor/api-named": "1" * 40, f"cursor/by-suffix-{CO[-4:]}": "2" * 40, "cursor/not-veritys": "3" * 40})
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-r"}],
                                 {"run-r": run}, notes=tmp_path / "notes", git_remote=REMOTE, sh=sh)
    (tmp_path / "notes").mkdir()
    sweep = w.sweep

    def working(start):                     # the agent works (its updatedAt moves) but writes no note and pushes nothing
        api._agents[0]["updatedAt"] = W.iso(clock.t)
        sweep(start)
    w.sweep = working
    w.sweep(lambda r: None)
    sweeps(w, clock, 50)
    sh.heads[f"cursor/by-suffix-{CO[-4:]}"] = "4" * 40
    sweeps(w, clock, 50)
    sh.heads["cursor/api-named"] = "5" * 40
    sweeps(w, clock, 50)
    sh.heads["cursor/not-veritys"] = "6" * 40
    assert silent(tmp_path) == []
    sweeps(w, clock, 60)
    (rec,) = silent(tmp_path)
    assert (rec["agent"], rec["handle"], rec["run"], rec["minutes"], rec["since"], rec["owner"]) == (
        CO, "circuits", "run-r", 60, W.iso(T0 + 105 * 60), "top"), \
        "its last push was seen at the heads read after it; a coordinator is silent after an hour, and top is rung"
    assert rec["branches"] == ["cursor/api-named", f"cursor/by-suffix-{CO[-4:]}"]
    assert status(tmp_path)[CO]["watch_state"] == "silent"


def test_stale_heads_or_no_notes_judge_nobody_and_an_agent_that_goes_idle_ends_its_spell(tmp_path):
    agents = [{"id": CO, "status": "ACTIVE", "latestRunId": "run-r"}]
    runs = {"run-r": {"status": "RUNNING", "updatedAt": W.iso(T0)}}
    sh = FakeSh()
    api, clock, w = coordinators(tmp_path, agents, runs, notes=tmp_path / "notes", git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    sh.fail.add("git")
    sweeps(w, clock, 180)
    assert silent(tmp_path) == [], "no push can be told from heads 3 h old"
    sh.fail.clear()
    sweeps(w, clock, 5)
    assert len(silent(tmp_path)) == 1, "read again: silent since the first read"
    agents[0]["status"], runs["run-r"]["status"] = "IDLE", "FINISHED"
    agents[0]["updatedAt"] = W.iso(clock.t)
    sweeps(w, clock, 5)
    assert w.silent == {} and w.active == {}
    agents[0]["status"], runs["run-r"]["status"] = "ACTIVE", "RUNNING"
    sweeps(w, clock, 55)
    assert len(silent(tmp_path)) == 1, "a new ACTIVE spell counts from its start, the sweep that saw it ACTIVE again"
    sweeps(w, clock, 10)
    assert [r["since"] for r in silent(tmp_path)][1:] == [W.iso(clock.t - W.COORD_SILENT_S)]
    (tmp_path / "no-notes").mkdir()
    api2, clock2, w2 = coordinators(tmp_path / "no-notes", agents, runs, git_remote=REMOTE, sh=FakeSh())
    sweeps(w2, clock2, 200)
    assert silent(tmp_path / "no-notes") == []


def test_a_restarted_watch_keeps_the_spells_rings(tmp_path):
    agents = [{"id": CO, "status": "ACTIVE", "latestRunId": "run-r"}]
    runs = {"run-r": {"status": "RUNNING", "updatedAt": W.iso(T0)}}
    (tmp_path / "notes").mkdir()
    sh = FakeSh({"main": "m" * 40})
    api, clock, w = coordinators(tmp_path, agents, runs, notes=tmp_path / "notes", git_remote=REMOTE, sh=sh)
    w.sweep(lambda r: None)
    sweeps(w, clock, 90)
    assert len(silent(tmp_path)) == 1
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                    notes=tmp_path / "notes", git_remote=REMOTE, sh=sh)
    sweeps(again, clock, 25)
    assert len(silent(tmp_path)) == 1, "neither the ring nor the spell's start is forgotten"
    sweeps(again, clock, 5)
    assert [r["window"] for r in silent(tmp_path)] == [1, 2]


def lead(tmp_path):
    """CO, circuits' coordinator, ACTIVE in the turn of run-r since the first sweep (T0), with no note of its lane and no
    branch of its own. It works: its agent's `updatedAt` moves at every sweep, so no run of its is stalled."""
    agents = [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-r"}]
    runs = {"run-r": {"status": "RUNNING", "createdAt": W.iso(T0), "updatedAt": W.iso(T0)}}
    (tmp_path / "notes").mkdir(parents=True)
    api, clock, w = coordinators(tmp_path, agents, runs, notes=tmp_path / "notes", git_remote=REMOTE,
                                 sh=FakeSh({"main": "m" * 40}))
    sweep = w.sweep

    def working(start):
        agents[0]["updatedAt"] = W.iso(clock.t)
        sweep(start)
    w.sweep = working
    w.sweep(lambda r: None)
    return api, clock, w, agents, runs


def reads(api, what):
    """How often the fake API was asked for CO's `runs` or `conversation`."""
    return api.gets.count(f"/{'v0' if what == 'conversation' else 'v1'}/agents/{CO}/{what}")


def test_a_lead_writing_assistant_text_with_no_note_or_push_is_not_silent_and_its_text_costs_no_read(tmp_path):
    """A lead works through Slack, Linear and its workers for hours with no note and no push, and was rung silent two minutes
    after it wrote. Its run's stream gives its assistant text, timed as it comes, for no read; an hour after its last text,
    still in its turn, it is silent, in a record that counts the turn and the text."""
    api, clock, w, agents, runs = lead(tmp_path)
    for _ in range(9):
        sweeps(w, clock, 20)
        w.on_event("run-r", "assistant", {"text": "Read the thread; asking the worker for its numbers."})
    said = clock.t
    sweeps(w, clock, 55)
    assert silent(tmp_path) == [], "text every 20 min for three hours, the last 55 min ago"
    assert reads(api, "conversation") == 0 and reads(api, "runs") == 1 + 235 // 5, "nothing read beyond each sweep's own"
    sweeps(w, clock, 5)
    (rec,) = silent(tmp_path)
    assert {k: rec.get(k) for k in ("agent", "run", "role", "owner", "since", "minutes", "active_minutes", "window")} == {
        "agent": CO, "run": "run-r", "role": "coordinator", "owner": "top", "since": W.iso(said), "minutes": 60,
        "active_minutes": 240, "window": 1}
    assert reads(api, "conversation") == 0 and reads(api, "runs") == 2 + 240 // 5, \
        "its runs are read once more, to see it still in its turn; its stream is received, so its conversation is not"


def test_a_lead_in_a_turn_with_nothing_written_for_an_hour_is_silent_and_its_record_counts_that_turn(tmp_path):
    """The fault the ring is for: a lead in a turn, its stream received, that writes nothing anywhere the watch reads; tool
    calls and thinking are no text. The record counts the turn it is in, not how long the list has read ACTIVE: a lead
    ACTIVE through two turns was rung an hour into the first's start, with the minutes of both."""
    api, clock, w, agents, runs = lead(tmp_path)
    sweeps(w, clock, 40)
    runs["run-r"]["status"] = "FINISHED"
    runs["run-s"] = {"status": "RUNNING", "createdAt": W.iso(clock.t), "updatedAt": W.iso(clock.t)}
    agents[0]["latestRunId"], api.older[CO] = "run-s", ["run-r"]
    began = clock.t + 5 * 60                # the sweep that first sees its second turn
    for i in range(12):
        sweeps(w, clock, 5)
        w.on_event("run-s", "tool_call", {"callId": f"c{i}", "name": "mcp", "status": "running"})
        w.on_event("run-s", "tool_call", {"callId": f"c{i}", "name": "mcp", "status": "completed"})
        w.on_event("run-s", "thinking", {"text": "Which worker has the numbers?"})
    assert silent(tmp_path) == [], "55 min into its second turn, 100 min ACTIVE"
    sweeps(w, clock, 5)
    (rec,) = silent(tmp_path)
    assert (rec["run"], rec["since"], rec["minutes"], rec["active_minutes"], rec["role"], rec["owner"]) == (
        "run-s", W.iso(began), 60, 60, "coordinator", "top")
    sweeps(w, clock, 60)
    assert [(r["window"], r["minutes"], r["active_minutes"]) for r in silent(tmp_path)] == [(1, 60, 60), (2, 120, 120)]


@pytest.mark.parametrize("status", ["FINISHED", "WAITING_FOR_BACKGROUND_WORK", "CREATING"])
def test_a_lead_between_turns_is_not_silent_whatever_the_list_says(tmp_path, status):
    """The list's ACTIVE lags a turn's end, and covers a background wait and a run about to start: a lead was rung "ACTIVE 83
    min" a quarter of an hour after its turn ended. One whose run is not RUNNING is between turns, however old its last
    note; its next turn counts from the sweep that sees it begin."""
    api, clock, w, agents, runs = lead(tmp_path)
    sweeps(w, clock, 45)
    runs["run-r"]["status"] = status
    sweeps(w, clock, 120)
    assert silent(tmp_path) == [] and w.active == {}, "ACTIVE on the list for 165 min, with no note and no push"
    runs["run-r"]["status"] = "FINISHED"
    runs["run-s"] = {"status": "RUNNING", "createdAt": W.iso(clock.t), "updatedAt": W.iso(clock.t)}
    agents[0]["latestRunId"], api.older[CO] = "run-s", ["run-r"]
    sweeps(w, clock, 60)
    assert silent(tmp_path) == []
    sweeps(w, clock, 5)
    assert [(r["run"], r["minutes"], r["active_minutes"]) for r in silent(tmp_path)] == [("run-s", 60, 60)]


def test_a_lead_whose_turn_ends_while_the_sweep_runs_is_not_recorded_silent(tmp_path):
    """A sweep's reads take minutes: a lead was rung silent at the minute its turn ended. Its runs are read again before it
    is recorded, and a turn over by then is no silence."""
    api, clock, w, agents, runs = lead(tmp_path)
    sweeps(w, clock, 55)
    get = api.get

    def ending(path, **params):             # its turn ends between the sweep's read of its runs and its judging of silence
        got = get(path, **params)
        if path.endswith("/runs"):
            runs["run-r"]["status"] = "FINISHED"
        return got
    api.get = ending
    before = reads(api, "runs")
    sweeps(w, clock, 5)
    assert silent(tmp_path) == [] and reads(api, "runs") == before + 2, "the sweep's read saw it RUNNING; the second, ended"
    api.get, agents[0]["status"] = get, "IDLE"
    sweeps(w, clock, 60)
    assert silent(tmp_path) == [] and w.active == {} and w.silent == {}


def test_a_lead_whose_turn_ends_as_its_six_hours_are_up_is_not_no_step(tmp_path):
    """no_step tells top to stop and archive a lead: one whose turn ended while the sweep ran has just done something."""
    api, clock, w, agents, runs = lead(tmp_path)
    sweeps(w, clock, 125)
    assert [r["window"] for r in silent(tmp_path)] == [1, 2]
    w.sweep = W.Watch.sweep.__get__(w)      # its agent's `updatedAt` stops moving: no_step counts from here
    get, due = api.get, clock.t + W.NO_STEP_S

    def ending(path, **params):
        got = get(path, **params)
        if path.endswith("/runs") and clock.t >= due:
            runs["run-r"]["status"] = "FINISHED"
        return got
    api.get = ending
    sweeps(w, clock, W.NO_STEP_S // 60 + 30)
    assert [r["kind"] for r in records(tmp_path) if r["kind"] in ("no_step", "archived")] == []


def test_a_leads_conversation_is_read_only_when_it_is_about_to_be_recorded_with_no_stream_received(tmp_path):
    """With no stream of its run received (a goal run, a stream past its retention, a broken one) a lead's conversation
    says whether it wrote: read when it is about to be recorded, never once a sweep. The conversation carries no times, so
    text the watch reads for the first time is life at that read, and the same text an hour on is silence since."""
    api, clock, w, agents, runs = lead(tmp_path)
    api.convs[CO] = [("user_message", "Go."), ("assistant_message", "On it.")]
    sweeps(w, clock, 55, beat=False)
    assert reads(api, "conversation") == 0 and silent(tmp_path) == []
    sweeps(w, clock, 5, beat=False)
    assert reads(api, "conversation") == 1 and silent(tmp_path) == [], "text read for the first time can't be dated"
    sweeps(w, clock, 30, beat=False)
    api.convs[CO].append(("assistant_message", "Asked the worker for its numbers."))
    sweeps(w, clock, 30, beat=False)
    assert reads(api, "conversation") == 2 and silent(tmp_path) == [], "it wrote since the last read"
    sweeps(w, clock, 55, beat=False)
    assert reads(api, "conversation") == 2 and silent(tmp_path) == []
    sweeps(w, clock, 5, beat=False)
    (rec,) = silent(tmp_path)
    assert reads(api, "conversation") == 3 and "unread_minutes" not in rec
    assert (rec["since"], rec["minutes"], rec["active_minutes"]) == (W.iso(T0 + 120 * 60), 60, 180), \
        "silent since the read that last found new text, in a turn three hours old"
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                    notes=tmp_path / "notes", git_remote=REMOTE, sh=w.sh)
    assert again.texts == w.texts and again.active == w.active, "a restart keeps the text last read and the turn"


def test_a_conversation_the_api_wont_read_neither_rings_by_itself_nor_hides_a_silent_lead(tmp_path):
    """A read that fails says nothing, so it records nothing and is made again ten minutes on, not every sweep. A
    conversation unread at every try for a further hour hides no silence: the lead is recorded, its record saying for how
    long its text went unread."""
    api, clock, w, agents, runs = lead(tmp_path)        # the API answers 404 for its conversation
    sweeps(w, clock, 115, beat=False)
    assert silent(tmp_path) == [] and reads(api, "conversation") == 6, "tried at 60 min and every ten after"
    sweeps(w, clock, 5, beat=False)
    (rec,) = silent(tmp_path)
    assert (rec["minutes"], rec["active_minutes"], rec["unread_minutes"], rec["window"]) == (120, 120, 60, 1)
    sweeps(w, clock, 60, beat=False)
    assert len(silent(tmp_path)) == 1, "the second window waits its own hour of tries"
    sweeps(w, clock, 5, beat=False)
    assert [(r["window"], r["unread_minutes"]) for r in silent(tmp_path)] == [(1, 60), (2, 60)]
    api.convs[CO] = [("assistant_message", "Still here.")]
    w.silent.clear()
    sweeps(w, clock, 10, beat=False)
    assert len(silent(tmp_path)) == 2 and w.blind == {}, "read at last: text the watch hadn't read, so life"
    sweeps(w, clock, 60, beat=False)
    assert "unread_minutes" not in silent(tmp_path)[-1] and silent(tmp_path)[-1]["minutes"] == 60


def test_a_failed_read_of_a_leads_runs_records_nothing_until_a_read_shows_it_in_its_turn(tmp_path):
    """Whether it is still in its turn is not known from a read that failed: nothing is recorded, the turn the watch last
    saw stands, and the next try, ten minutes on, decides."""
    api, clock, w, agents, runs = lead(tmp_path)
    get, asked, fail = api.get, [], {"from": 1}

    def failing(path, **params):            # each sweep's reads of its runs fail from the `fail["from"]`th on
        if path.endswith("/runs"):
            asked.append(clock.t)
            if asked.count(clock.t) >= fail["from"]:
                raise W.urllib.error.URLError("timed out")
        return get(path, **params)
    api.get = failing
    sweeps(w, clock, 5)
    fail["from"] = 2
    sweeps(w, clock, 55)
    assert silent(tmp_path) == [] and asked.count(clock.t) == 2, "the sweep's read, and the one before recording, which failed"
    assert w.active == {CO: {"run": "run-r", "since": T0}}, "the sweep that couldn't read its runs left it in its turn"
    sweeps(w, clock, 5)
    assert silent(tmp_path) == [] and asked.count(clock.t) == 1, "not judged again for ten minutes"
    fail["from"] = 3
    sweeps(w, clock, 5)
    assert [(r["minutes"], r["active_minutes"]) for r in silent(tmp_path)] == [(70, 70)]


def test_text_a_stream_started_again_replays_is_no_new_text(tmp_path):
    """A stream that fails is started again from the run's start, and gives the run's old text over again, timed as it
    arrives: counted as written then, every failure would hide an hour of silence."""
    api, clock, w, agents, runs = lead(tmp_path)
    sweeps(w, clock, 10)
    w.on_event("run-r", "assistant", {"text": "Read the thread; "})
    w.on_event("run-r", "assistant", {"text": "asking the worker."})
    said = clock.t
    sweeps(w, clock, 40)
    w.runs["run-r"].streaming = False       # its stream failed: the next sweep starts it again
    sweeps(w, clock, 5)
    w.on_event("run-r", "assistant", {"text": "Read the thread; asking the worker."})
    sweeps(w, clock, 10)
    assert silent(tmp_path) == [], "55 min after its text"
    sweeps(w, clock, 5)
    assert [r["since"] for r in silent(tmp_path)] == [W.iso(said)] and reads(api, "conversation") == 0
    w.on_event("run-r", "assistant", {"text": " It answered."})
    sweeps(w, clock, 60)
    assert [r["since"] for r in silent(tmp_path)] == [W.iso(said), W.iso(said + 60 * 60)], "text past the replay is new"


def test_a_stream_just_started_shows_nothing_yet_so_the_conversation_is_read(tmp_path):
    """A stream started in the last minutes may still be replaying: that it gave no new text proves none was written."""
    api, clock, w, agents, runs = lead(tmp_path)
    api.convs[CO] = [("assistant_message", "On it.")]
    sweeps(w, clock, 58, step=1)
    w.runs["run-r"].streaming = False
    sweeps(w, clock, 2, step=1)
    assert silent(tmp_path) == [] and reads(api, "conversation") == 1, "an hour in, its stream a minute old"
    sweeps(w, clock, 60, step=1)
    assert [r["minutes"] for r in silent(tmp_path)] == [60] and reads(api, "conversation") == 1, \
        "an hour after that read, its stream received again: no second read"


def node_run(root, rid, ended, *, by="", state="done", rc=0, wall_s=750.0, command=None, job="", lane="", tree=""):
    """A run dir as `research run` leaves it on a node: runs/<rid>/{status,job}.json and requests/<rid>/request.json."""
    d, q = root / "runs" / rid, root / "requests" / rid
    d.mkdir(parents=True)
    q.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "rc": rc, "wall_s": wall_s, "transitions": [
        {"state": "running", "t_utc": W.iso(ended - wall_s)}, {"state": state, "t_utc": W.iso(ended)[:-1] + ".250Z"}]}))
    (d / "job.json").write_text(json.dumps({"ownership": {"job": job, "lane": lane},
                                             "computation": {"stage": "", **({"tree_sha": tree} if tree else {})}}))
    (q / "request.json").write_text(json.dumps({"command": command or ["python", "-m", "bench", "--seed", "1"],
                                                **({"submitted_by": by} if by else {})}))
    W.os.utime(d / "status.json", (ended, ended))


def nodes_setup(tmp_path, **kw):
    """vy-nebius-1 and vy-nebius-2 in the notes' machines.d, each with its runs root under tmp_path; nodes.json names both."""
    notes = tmp_path / "notes"
    (notes / "machines.d").mkdir(parents=True)
    roots = {}
    for n in ("vy-nebius-1", "vy-nebius-2"):
        roots[n] = tmp_path / n
        (notes / "machines.d" / f"{n}.toml").write_text(
            f'provider = "ssh"\nhost = "{n}.example"\nuser = "research"\nproject = "verity"\nroot = "{roots[n]}"\n')
    (tmp_path / "nodes.json").write_text(json.dumps({"nodes": ["vy-nebius-1", "vy-nebius-2"], "every_s": 300,
                                                     "ssh": ["ssh", "-o", "BatchMode=yes"]}))
    sh = FakeSh()
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "IDLE", "latestRunId": "run-f",
                                             "updatedAt": "2026-10-18T19:20:00Z"}], {"run-f": dict(FINISHED)}, notes=notes,
                                 nodes=tmp_path / "nodes.json", sh=sh, timers=None, **kw)
    return api, clock, w, sh, roots


def ended(tmp_path):
    return [r for r in records(tmp_path) if r["kind"] == "run_ended"]


def test_a_run_that_ends_rings_its_senders_handle_once(tmp_path):
    """A worker that ends its turn to wait on a run never learns it ended: Cursor refuses a new run on a Task worker (400
    legacy workflow), so its coordinator is rung to resume it; a coordinator's own run rings its own handle."""
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "w1.json").write_text(json.dumps({"agent_id": W1_, "lane": "zk-port", "launched_by": NEW}))
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    api.workers[W1_] = worker("ACTIVE", W.iso(T0))
    node_run(roots["vy-nebius-1"], "r20261018-190000-0001", T0 - 3600, by=f"https://cursor.com/agents/{CO}")
    w.sweep(lambda r: None)
    assert ended(tmp_path) == [], "a run that ended before the first read is not rung"
    assert [a[-1] for a in sh.calls if a[0] == "ssh"] == [f"python3 - {roots[n]} {T0 - W.RUN_ENDS_SLACK_S:.0f}"
                                                          for n in ("vy-nebius-1", "vy-nebius-2")]
    assert all(a[-2] == f"research@{n}.example" for a, n in zip(sh.calls, ("vy-nebius-1", "vy-nebius-2")))
    clock.t += 60
    node_run(roots["vy-nebius-1"], "r20261018-195000-0002", clock.t, by=f"https://cursor.com/agents/{CO}", job="sweep-a",
             wall_s=4000)
    node_run(roots["vy-nebius-2"], "r20261018-195500-0003", clock.t, by=f"https://cursor.com/agents/{W1_}", state="failed", rc=3,
             command=["python", "bench/run.py", "--n", "8", "--quick"])
    node_run(roots["vy-nebius-2"], "r20261018-195500-0004", clock.t, command=["slot", "--by", "vllm", "--", "check"], rc=1,
             state="failed", wall_s=30)
    node_run(roots["vy-nebius-2"], "r20261018-195500-0005", clock.t, by="https://cursor.com/agents/bc-unknown")
    w.sweep(lambda r: None)
    assert ended(tmp_path) == [], "read every 5 min"
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    got = {r["run"]: {k: r.get(k) for k in ("agent", "name", "handle", "role", "node", "what", "rc", "state", "after", "lane")}
           for r in ended(tmp_path) if r["role"] != "unknown"}
    (unsent,) = [r for r in ended(tmp_path) if r["role"] == "unknown"]
    assert (unsent["handle"], unsent["count"], [(u["run"], u["sender"]) for u in unsent["runs"]]) == (
        None, 1, [("r20261018-195500-0005", "https://cursor.com/agents/bc-unknown")]), "the run nobody's handle owns, for @infra"
    assert got == {
        "r20261018-195000-0002": {"agent": CO, "name": "verity: circuits", "handle": "circuits", "role": "coordinator",
                                  "node": "vy-nebius-1", "what": "sweep-a", "rc": 0, "state": "done", "after": "1 h 06 min",
                                  "lane": ""},
        "r20261018-195500-0003": {"agent": W1_, "name": "ZKS10", "handle": "vllm", "role": "worker", "node": "vy-nebius-2",
                                  "what": "python bench/run.py --n 8", "rc": 3, "state": "failed", "after": "12 min",
                                  "lane": "zk-port"},
        "r20261018-195500-0004": {"agent": "", "name": "", "handle": "vllm", "role": "lane", "node": "vy-nebius-2",
                                  "what": "slot --by vllm --", "rc": 1, "state": "failed", "after": "30 s", "lane": "vllm"}}
    assert {r["run"]: r.get("sender_status") for r in ended(tmp_path) if r["role"] != "unknown"} == {
        "r20261018-195000-0002": "IDLE", "r20261018-195500-0003": "ACTIVE", "r20261018-195500-0004": "unknown"}, \
        "the sender's status at its run's end, for the relay to ring only a sender not in a turn"
    assert status(tmp_path)[CO]["watch_state"] != "run_ended", "a research run's end is not the agent's state"
    for _ in range(3):
        clock.t += 5 * 60
        w.sweep(lambda r: None)
    restarted = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                        notes=tmp_path / "notes", nodes=tmp_path / "nodes.json", sh=sh)
    restarted.sweep(lambda r: None)
    assert len(ended(tmp_path)) == 4, "each run rings once, across restarts"


def test_a_run_a_worker_sent_is_its_activity(tmp_path):
    """A worker whose only sign of life was the runs it sent (no note, no push) was rung silent and then no_step while its runs
    ended every hour. A run counts at its start and at its end."""
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_})", age_s=2 * 3600)
    api, clock, w, sh, roots = nodes_setup(tmp_path, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, git_remote=REMOTE)
    sh.heads["main"] = "m" * 40
    w.sweep(lambda r: None)
    for i in range(6):
        sweeps(w, clock, 60)
        node_run(roots["vy-nebius-1"], f"r20261018-2{i}0000-000{i}", clock.t, by=f"https://cursor.com/agents/{W1_}", wall_s=600)
    last_end = clock.t
    assert silent(tmp_path) == [], "a run sent every hour is activity"
    sweeps(w, clock, 120)
    (rec,) = silent(tmp_path)
    assert rec["agent"] == W1_ and rec["since"] == W.iso(last_end), "silence counts from its newest run's end"


def test_a_run_ends_sender_resolves_as_a_learned_worker_does_and_the_rest_are_one_record_for_infra(tmp_path):
    """Run ends whose sender is a worker learned from a coordinator's Task call, or a coordinator's older generation after
    its handover, ring their handle; runs of workers the watch never learned are, for each read, one record, rung to infra. A `submitted_by` that names a lane ($RESEARCH_AGENT_URL is free text) rings that lane."""
    url = "https://cursor.com/agents/{}".format
    censusand, circuits3, ca2 = ("bc-fe356a9d-81e7-54a8-a5da-fbf425245da5", "bc-21f3d3de-28db-5e5d-964b-d357187449aa",
                                 "bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac")
    aede, df15 = "bc-aede327f-c2cf-5f24-a40a-309e7bb815f4", "bc-df15df07-8eff-531b-97bc-f5cbbdeda23e"
    assert (W.generation("circuits", circuits3), W.generation("compute-accounting", ca2)) == (3, 2)
    api, clock, w, _, roots = nodes_setup(tmp_path)
    reg = json.loads((tmp_path / "registry.json").read_text())
    reg["handles"].append({"handle": "compute-accounting", "holder": f"the compute-accounting coordinator ({LANE})"})
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    w.sweep(lambda r: None)
    w.child(circuits3, {"args": {"agentId": censusand, "description": "CENSUSAND: merge #2765 fix into #2738"}}, clock.t)
    api.workers[censusand] = worker("ACTIVE", W.iso(clock.t), "CENSUSAND: merge #2765 fix into #2738")
    clock.t += 60
    node_run(roots["vy-nebius-2"], "r20261019-043737-f6de", clock.t, by=url(censusand), state="failed", rc=130, wall_s=3068.6)
    node_run(roots["vy-nebius-2"], "r20261019-132935-493f", clock.t, by=url(ca2), state="failed", rc=127, wall_s=0.47)
    node_run(roots["vy-nebius-2"], "r20261019-052718-24a9", clock.t, by=url(aede), state="failed", rc=8, wall_s=6.8)
    node_run(roots["vy-nebius-1"], "r20261019-192229-3606", clock.t, by=url(df15), state="failed", rc=143, wall_s=1200)
    node_run(roots["vy-nebius-1"], "r20261019-195000-0001", clock.t, by="circuits lane, Daniel's laptop")
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    got = {r["run"]: (r["agent"], r["name"], r["handle"], r["role"]) for r in ended(tmp_path) if r["role"] != "unknown"}
    assert got == {"r20261019-043737-f6de": (censusand, "CENSUSAND: merge #2765 fix into #2738", "circuits", "worker"),
                   "r20261019-132935-493f": (ca2, "", "compute-accounting", "coordinator"),
                   "r20261019-195000-0001": ("", "", "circuits", "lane")}
    (unsent,) = [r for r in ended(tmp_path) if r["role"] == "unknown"]
    assert (unsent["handle"], unsent["count"], unsent["senders"]) == (None, 2, sorted([url(aede), url(df15)]))
    assert sorted((u["run"], u["node"], u["rc"]) for u in unsent["runs"]) == [
        ("r20261019-052718-24a9", "vy-nebius-2", 8), ("r20261019-192229-3606", "vy-nebius-1", 143)]
    assert censusand in w.awaiting, "the worker's end starts its clock, for its coordinator to resume it"


def test_a_run_ends_unknown_sender_is_rung_to_the_handle_its_branch_names(tmp_path):
    """An unknown sender is traced from the run's tree to the branch that had it as its head to the lane. The branch's suffix in the registry's `branch_suffixes`, or a handle its name starts with, routes it;
    a sha no branch had stays infra's."""
    url = "https://cursor.com/agents/{}".format
    api, clock, w, _, roots = nodes_setup(tmp_path)
    reg = json.loads((tmp_path / "registry.json").read_text())
    reg["handles"].append({"handle": "kernels", "holder": "the kernels coordinator"})
    for h in reg["handles"]:
        if h["handle"] == "circuits":
            h["branch_suffixes"] = ["8c79"]
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    w.heads = {"t0": T0, "t": T0, "heads": {"cursor/kernels-kp2s8-8c40": [b, T0]},
               "shas": {a: ["cursor/circuits-demowl-8c79", T0], b: ["cursor/kernels-kp2s8-8c40", T0]}}
    w.sweep(lambda r: None)
    clock.t += 60
    node_run(roots["vy-nebius-1"], "r20261025-232710-0001", clock.t, by=url("bc-e7fea65b-bbd7-5c6c-a635-ebfb68240c07"), tree=a)
    node_run(roots["vy-nebius-1"], "r20261025-232710-0002", clock.t, by=url("bc-5525ca1e-435c-5c8a-823a-eb79a408f170"), tree=b)
    node_run(roots["vy-nebius-1"], "r20261025-232710-0003", clock.t, by=url("bc-9f0ed9de-4f13-5f31-9e6b-dab2b7a1ac12"), tree=c)
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    got = {r["run"]: (r["handle"], r["role"]) for r in ended(tmp_path) if r["role"] != "unknown"}
    assert got == {"r20261025-232710-0001": ("circuits", "lane"), "r20261025-232710-0002": ("kernels", "lane")}
    (unsent,) = [r for r in ended(tmp_path) if r["role"] == "unknown"]
    assert [u["run"] for u in unsent["runs"]] == ["r20261025-232710-0003"]


def test_read_heads_remembers_each_pushed_sha_s_branch_for_two_days(tmp_path):
    api, clock, w, sh, _ = nodes_setup(tmp_path)
    w.git_remote = "git@example:verity.git"
    w.heads = {"t0": T0, "t": T0, "heads": {}, "shas": {"d" * 40: ["cursor/old-1111", T0 - W.HEADS_SHA_KEEP_S - 1]}}
    sh.heads = {"cursor/lean-x-fc8d": "a" * 40}
    w.read_heads(clock.t)
    sh.heads = {"cursor/lean-x-fc8d": "b" * 40}
    w.read_heads(clock.t + W.HEADS_EVERY_S)
    assert (w.branch_of("a" * 40), w.branch_of("b" * 40), w.branch_of("d" * 40)) == (
        "cursor/lean-x-fc8d", "cursor/lean-x-fc8d", ""), "an older head of a branch still names it; a stale one is forgotten"


def test_a_quiet_handles_runs_are_not_rung(tmp_path):
    """Drive's train checks end every few minutes under `--by ci`, and drive reads them itself: ringing ci for each is noise."""
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    cfg = json.loads((tmp_path / "nodes.json").read_text())
    (tmp_path / "nodes.json").write_text(json.dumps({**cfg, "quiet_handles": ["vllm"]}))
    w.sweep(lambda r: None)
    clock.t += 60
    node_run(roots["vy-nebius-2"], "r20261018-195500-0004", clock.t, command=["slot", "--by", "vllm", "--", "check"])
    node_run(roots["vy-nebius-1"], "r20261018-195000-0002", clock.t, by=f"https://cursor.com/agents/{CO}")
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    assert [(r["run"], r["handle"]) for r in ended(tmp_path)] == [("r20261018-195000-0002", "circuits")]


def test_the_shipped_nodes_file_keeps_ci_quiet():
    cfg = json.loads((W.Path(__file__).resolve().parents[3] / "infra/control/agent_watch_nodes.json").read_text())
    assert cfg["quiet_handles"] == ["ci"] and cfg["nodes"] == ["vy-nebius-2", "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-4", "vy-nebius-cpu-5"]


def test_a_node_that_doesnt_answer_is_said_once_an_hour_and_caught_up_when_it_does(tmp_path, capsys):
    """The H200s stop after 2 h idle: a node read fails, and its runs are read from its last good read once it answers."""
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    w.sweep(lambda r: None)
    sh.fail.add("vy-nebius-2.example")
    for _ in range(4):
        clock.t += 5 * 60
        node_run(roots["vy-nebius-2"], f"r20261018-2000{_:02d}-000{_}", clock.t, by=f"https://cursor.com/agents/{CO}")
        w.sweep(lambda r: None)
    err = capsys.readouterr().err
    assert err.count("vy-nebius-2 unreadable") == 1 and "Connection timed out" in err
    sh.fail.clear()
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    assert len(ended(tmp_path)) == 4 and {r["node"] for r in ended(tmp_path)} == {"vy-nebius-2"}
    (tmp_path / "notes" / "machines.d" / "vy-nebius-1.toml").unlink()
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    assert "no host for vy-nebius-1" in capsys.readouterr().err


def test_the_deployed_nodes_file_names_the_nebius_nodes_and_an_ssh_argv():
    root = W.Path(__file__).resolve().parents[3]
    cfg = W.read_json(root / "infra/control/agent_watch_nodes.json")
    assert cfg["nodes"] == ["vy-nebius-2", "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-4", "vy-nebius-cpu-5"] and cfg["every_s"] >= 60
    assert cfg["ssh"][0] == "ssh" and all(isinstance(a, str) for a in cfg["ssh"]) and "BatchMode=yes" in cfg["ssh"]


def test_a_running_run_that_streams_nothing_for_30_min_is_stalled_cancelled_and_continued(tmp_path):
    """A run that reads RUNNING and streams nothing, while its agent's updatedAt stands still,
    is a hung call or a suspended VM, which the API doesn't show. Its owner (top, for a coordinator) is rung with the record."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-r",
                                             "updatedAt": W.iso(T0)}], {"run-r": {"status": "RUNNING"}}, timers=None)
    w.sweep(lambda r: None)
    w.on_event("run-r", "tool_call", {"callId": "c1", "name": "Shell", "status": "running"})
    quiet_from = clock.t
    for _ in range(W.STALL_S // 60 - 1):
        clock.t += 60
        w.on_event("run-r", "heartbeat", {})
        w.sweep(lambda r: None)
    assert api.posts == [] and records(tmp_path) == [], "29 min of nothing is not yet a stall"
    clock.t += 60
    w.sweep(lambda r: None)
    (cancel, cont) = api.posts
    assert cancel == (f"/v1/agents/{CO}/runs/run-r/cancel", None)
    assert cont[0] == f"/v1/agents/{CO}/runs" and "streamed nothing for 30 min" in cont[1]["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert {k: rec[k] for k in ("kind", "since", "minutes", "action", "owner")} == {
        "kind": "stalled", "since": W.iso(quiet_from), "minutes": 30, "action": "cancelled and continued", "owner": "top"}
    assert status(tmp_path)[CO]["watch_state"] == "stalled"


def test_a_quiet_stream_while_the_agent_moves_or_its_subagents_report_is_no_stall(tmp_path):
    """The stream may break while the agent works (its updatedAt moves), or the parent may wait on a subagent that reports
    progress: neither is cancelled."""
    api, clock, w, started = setup(tmp_path)
    for _ in range(2 * W.STALL_S // 60):
        clock.t += 60
        api._agents[0]["updatedAt"] = W.iso(clock.t)
        w.sweep(started.append)
    api._agents[0]["updatedAt"] = W.iso(clock.t)
    w.on_event("run-1", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    for _ in range(2 * W.STALL_S // 60):
        clock.t += 60
        w.on_event("run-1", "interaction_update", {"type": "tool-call-delta", "callId": "t1", "taskUpdate": {}})
        w.sweep(started.append)
    assert api.posts == [] and [r["kind"] for r in records(tmp_path)] == ["long_call"]


def test_an_error_run_is_restarted_10_min_after_the_last_restart(tmp_path):
    api, clock, w, started = setup(tmp_path, status="ERROR")
    for n, dt in ((2, W.RESTART_S - 60), (2, 60), (3, W.RESTART_S)):
        clock.t += dt
        api._agents[0]["latestRunId"] = f"run-e{clock.t:.0f}"
        api._runs[f"run-e{clock.t:.0f}"] = {"status": "ERROR"}
        w.sweep(started.append)
    assert [r["action"] for r in records(tmp_path)] == ["restarted (1 of 3)", "restarted (2 of 3)", "restarted (3 of 3)"]


def run_end_for(tmp_path, roots, clock, w, agent, rid="r20261018-200100-0002"):
    """A `research run` sent by `agent` ends a minute on; the watch reads the node within 5 min. Returns when it ended."""
    w.sweep(lambda r: None)
    clock.t += 60
    node_run(roots["vy-nebius-1"], rid, clock.t, by=f"https://cursor.com/agents/{agent}", job="sweep-a")
    return clock.t


def test_a_coordinator_idle_after_its_run_ended_is_resumed_with_the_result_10_min_on(tmp_path):
    """Rule (b): a coordinator that ended its turn to wait on a run, and wasn't woken when it ended."""
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    ended = run_end_for(tmp_path, roots, clock, w, CO)
    while clock.t < ended + W.RESUME_S - 60:
        clock.t += 60
        w.sweep(lambda r: None)
    assert [r["kind"] for r in records(tmp_path)] == ["run_ended"] and api.posts == [], "9 min: its subscription may yet fire"
    clock.t += 60
    w.sweep(lambda r: None)
    ((path, body),) = api.posts
    assert path == f"/v1/agents/{CO}/runs"
    text = body["prompt"]["text"]
    assert "r20261018-200100-0002 (sweep-a) on vy-nebius-1: done rc=0" in text and "research msg read --as circuits" in text
    rec = records(tmp_path)[-1]
    assert {k: rec[k] for k in ("kind", "agent", "run", "role", "handle", "owner", "ended_run", "action", "new_run", "minutes")} == {
        "kind": "idle_after_run", "agent": CO, "run": "run-f", "role": "coordinator", "handle": "circuits", "owner": "top",
        "ended_run": "r20261018-200100-0002", "action": "resumed it with the run's result", "new_run": "run-new", "minutes": 10}
    for _ in range(30):
        clock.t += 60
        w.sweep(lambda r: None)
    assert len(api.posts) == 1, "resumed once"


def test_a_coordinator_that_ran_after_its_run_ended_is_not_resumed(tmp_path):
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    ended = run_end_for(tmp_path, roots, clock, w, CO)
    api._agents[0]["latestRunId"] = "run-g"
    api._runs["run-g"] = {"status": "FINISHED", "createdAt": W.iso(ended + 120), "updatedAt": W.iso(ended + 180)}
    for _ in range(30):
        clock.t += 60
        w.sweep(lambda r: None)
    assert api.posts == [] and [r["kind"] for r in records(tmp_path)] == ["run_ended"] and w.awaiting == {}


def busy_once(api):
    """The first new run POSTed is refused 409 (agent busy), as Cursor refuses one on an agent waiting on background work."""
    real, refused = api.post, []

    def post(path, body=None):
        if path.endswith("/runs") and not refused:
            refused.append(path)
            raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
        return real(path, body)
    api.post = post


def test_a_background_wait_that_refuses_the_resume_is_cancelled_first(tmp_path):
    """A wake gets 409 while the agent's run sits in WAITING_FOR_BACKGROUND_WORK, so that wait is cancelled first."""
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    api._agents[0].update(status="ACTIVE", latestRunId="run-b")
    api._runs["run-b"] = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": "2026-10-18T19:00:00Z",
                          "updatedAt": "2026-10-18T19:30:00Z"}
    ended = run_end_for(tmp_path, roots, clock, w, CO)
    busy_once(api)
    while clock.t < ended + W.RESUME_S:
        clock.t += 60
        w.sweep(lambda r: None)
    assert [p for p, _ in api.posts] == [f"/v1/agents/{CO}/runs/run-b/cancel", f"/v1/agents/{CO}/runs"]
    rec = records(tmp_path)[-1]
    assert (rec["kind"], rec["action"]) == ("idle_after_run", "cancelled its background wait (run-b) and resumed it with the "
                                                              "run's result")


def test_a_wake_refused_by_a_background_wait_cancels_it_and_starts_the_wake(tmp_path):
    run = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": "2026-10-18T19:00:00Z", "updatedAt": "2026-10-18T19:30:00Z"}
    bells = {"t": W.iso(T0 - 60), "inboxes": {"vllm": {"agent": NEW, "last": W.epoch("2026-10-18T19:45:00Z")}}}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-b"}], {"run-b": run}, wake=True,
                                 bells=bells, timers=None)
    busy_once(api)
    w.sweep(lambda r: None)
    assert [p for p, _ in api.posts] == [f"/v1/agents/{NEW}/runs/run-b/cancel", f"/v1/agents/{NEW}/runs"]
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["action"], rec["new_run"]) == (
        "ready_to_wake", "cancelled its background wait (run-b) and started a run", "run-new")


class FakeLaunchApi:
    def __init__(self):
        self.keys, self.posts = [], []

    def __call__(self, key):
        self.keys.append(key)
        return self

    def post(self, path, body=None):
        self.posts.append((path, body))
        return {"agent": {"id": "bc-77777777-7777-7777-7777-777777777777"}}


KEY = "dummy-" "launch-key-ff"


def test_a_worker_idle_before_its_run_ended_rings_its_coordinator_then_is_relaunched_only_with_the_key(tmp_path, capsys):
    """A Task worker can't be started (400 legacy workflow): its coordinator is rung at 10 min, and if it stays idle 20 min more,
    stopped mid-step, a fresh agent takes its place with the launch key (without one, recorded): named as its lane's worker,
    told its branch, how to report to its coordinator's inbox and that the watch wakes it after its runs, given its newest
    note, and written to workers.d as the lane's worker."""
    for keyed in (False, True):
        d = tmp_path / str(keyed)
        (d / "workers.d").mkdir(parents=True)
        (d / "workers.d" / "w1.json").write_text(json.dumps({"agent_id": W1_, "lane": "zk-port", "launched_by": NEW,
                                                             "branch": "cursor/zk-w1-c7a9"}))
        launch_api = FakeLaunchApi()
        api, clock, w, sh, roots = nodes_setup(d, launch_api=launch_api)
        note(d / "notes", "zk-port", "20261018T1950Z-report-w1.md", f"zk-port worker ({W1_}); code cursor/zk-w1-c7a9")
        api.workers[W1_] = worker("IDLE", W.iso(T0 - 600), "ZKS10: frame the openings")
        api.convs[W1_] = [("user_message", "You're ZKS10, a worker for the vllm lane."),
                          ("assistant_message", "The lock didn't change. Committing the fix and rerunning the check on cpu-3.")]
        if keyed:
            (d / "launch.env").write_text(f"CURSOR_API_KEY={KEY}\n")
        ended = run_end_for(d, roots, clock, w, W1_)
        while clock.t < ended + W.RESUME_S + W.BURST_QUIET_S:
            clock.t += 60
            w.sweep(lambda r: None)
        rec = records(d)[-1]
        assert {k: rec[k] for k in ("kind", "role", "handle", "owner", "lane", "ended_run")} == {
            "kind": "idle_after_run", "role": "worker", "handle": "vllm", "owner": "vllm", "lane": "zk-port",
            "ended_run": "r20261018-200100-0002"}
        assert rec["action"] == ("rang @vllm to resume it with Task; relaunched fresh in 20 min if it stays idle with its task "
                                 "unfinished and its coordinator isn't ACTIVE")
        for _ in range(W.WORKER_RELAUNCH_S // 60 + 5):
            clock.t += 60
            w.sweep(lambda r: None)
        relaunches = [r for r in records(d) if r["kind"] == "relaunch"]
        assert len(relaunches) == 1 and relaunches[0]["owner"] == "vllm" and api.posts == []
        if not keyed:
            assert relaunches[0]["action"].startswith("not relaunched: no launch key in") and launch_api.posts == []
            continue
        ((path, body),) = launch_api.posts
        assert launch_api.keys == [KEY] and path == "/v1/agents" and body["env"] == {"type": "cloud", "name": W.ENV_NAME}
        text = body["prompt"]["text"]
        assert "cursor/zk-w1-c7a9" in text and "20261018T1950Z-report-w1.md" in text
        assert body["name"] == "vllm worker ZKS10 (relaunch of 1111)", "a worker's name, never a coordinator's `verity: LANE`"
        assert "You are a worker, not a coordinator" in text and "`research msg send --as vllm --thread " in text
        assert "stopped mid-step (its last message: \u201cThe lock didn't change. Committing the fix" in text
        assert "the agent watch rings your coordinator's inbox if it failed (`research msg ring-run`)" in text
        assert relaunches[0]["new_agent"] == "bc-77777777-7777-7777-7777-777777777777"
        new = json.loads((d / "workers.d" / "vllm-bc-77777777-7777-7777-7777-777777777777.json").read_text())
        assert new == {"agent_id": "bc-77777777-7777-7777-7777-777777777777", "lane": "vllm", "launched_by": NEW,
                       "role": "worker", "name": "vllm worker ZKS10 (relaunch of 1111)", "relaunch_of": W1_,
                       "branch": "cursor/zk-w1-c7a9"}
        written = "".join(p.read_text() for p in d.rglob("*") if p.is_file() and p.name != "launch.env")
        assert KEY not in written and KEY not in "".join(capsys.readouterr()), "the key is never written or printed"


def test_a_relaunched_worker_with_no_recorded_branch_is_told_the_branches_its_newest_note_names(tmp_path):
    """A relaunch whose workers.d entry names no branch must not read "Your branches: none recorded" when its note opens
    with "Branch `cursor/ca-salt-wire-ea24`, stacked on ... (`cursor/net45-...`)". Most workers.d entries name no branch, so the note's branches are named, the first written as the new worker's."""
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "w1.json").write_text(json.dumps({"agent_id": W1_, "lane": "zk-port", "launched_by": NEW}))
    launch_api = FakeLaunchApi()
    api, clock, w, _, roots = nodes_setup(tmp_path, launch_api=launch_api)
    p = note(tmp_path / "notes", "zk-port", "20261019T1400Z-draft-salt-wire.md", W1_)
    p.write_text(p.read_text() + "Branch `cursor/ca-salt-wire-ea24`, stacked on #2817 (`cursor/net45-reveal-deadline-061a` @\n"
                 "77a862af94); rebased on `cursor/ca-salt-wire-ea24`'s base.\n")
    W.os.utime(p, (T0 - 60, T0 - 60))
    api.workers[W1_] = worker("IDLE", W.iso(T0 - 600))
    api.convs[W1_] = [("user_message", "brief"), ("assistant_message", "Now I'll rebase on #2817:")]
    (tmp_path / "launch.env").write_text(f"CURSOR_API_KEY={KEY}\n")
    ended = run_end_for(tmp_path, roots, clock, w, W1_)
    while clock.t < ended + W.RESUME_S + W.BURST_QUIET_S + W.WORKER_RELAUNCH_S + 300:
        clock.t += 60
        w.sweep(lambda r: None)
    ((_, body),) = launch_api.posts
    assert ("Your branches: none recorded; its newest note names cursor/ca-salt-wire-ea24, cursor/net45-reveal-deadline-061a; "
            "fetch the one you work on") in body["prompt"]["text"]
    new = json.loads((tmp_path / "workers.d" / "vllm-bc-77777777-7777-7777-7777-777777777777.json").read_text())
    assert new["branch"] == "cursor/ca-salt-wire-ea24"


def vllm_workers(d, *agents):
    """workers.d entries for `agents`, launched by NEW (vllm's coordinator), as a coordinator registers its workers."""
    (d / "workers.d").mkdir(parents=True, exist_ok=True)
    for a in agents:
        (d / "workers.d" / f"{a[3:7]}.json").write_text(json.dumps({"agent_id": a, "lane": "vllm", "launched_by": NEW}))


def test_a_worker_is_relaunched_only_unfinished_never_while_its_coordinator_is_active_and_once_per_idle(tmp_path):
    """A worker that finished and waits on its coordinator is not relaunched, since its copy could push over its original. A worker whose last message reads as a report back ("Nothing changed: ...",
    paragraphs ending "or you can file it"), or whose run passed and that didn't stop mid-step, is recorded quietly
    as not relaunched; one whose run failed is relaunched, but never while its coordinator is ACTIVE, never on a conversation
    the API won't read, and never twice for one idle."""
    strand, dsv4, failed = W1_, W2_, W3_
    vllm_workers(tmp_path, strand, dsv4, failed)
    launch_api = FakeLaunchApi()
    api, clock, w, _, roots = nodes_setup(tmp_path, launch_api=launch_api)
    api._agents.append({"id": NEW, "name": "verity: vllm", "status": "ACTIVE", "latestRunId": "run-v",
                        "updatedAt": W.iso(T0)})
    api._runs["run-v"] = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": W.iso(T0 - 600), "updatedAt": W.iso(T0)}
    for a, name in ((strand, "STRAND: land #2822 per proofs' review"), (dsv4, "DSV4LAND: CIR-54 indexer partition fix"),
                    (failed, "ZKS10: frame the openings")):
        api.workers[a] = worker("IDLE", W.iso(T0 - 600), name)
    api.convs[strand] = [("user_message", "brief"), ("assistant_message", (
        "Nothing changed: #2822's head stays at 1cac8bff78 and quick tier r20261019-201010-338e stands."))]
    api.convs[dsv4] = [("user_message", "brief"), ("assistant_message", (
        "The body is updated.\n\nI left the catalogue entries out of #2765. I can add them in a one-line PR once that commit "
        "lands, or you can file it."))]
    (tmp_path / "launch.env").write_text(f"CURSOR_API_KEY={KEY}\n")
    w.sweep(lambda r: None)
    clock.t += 60
    url = "https://cursor.com/agents/{}".format
    node_run(roots["vy-nebius-1"], "r20261019-191151-c1ad", clock.t, by=url(strand))
    node_run(roots["vy-nebius-1"], "r20261019-191719-8c94", clock.t, by=url(dsv4))
    node_run(roots["vy-nebius-2"], "r20261019-192229-3606", clock.t, by=url(failed), state="failed", rc=3)
    ended = clock.t
    while clock.t < ended + W.RESUME_S + W.BURST_QUIET_S + W.WORKER_RELAUNCH_S + 600:
        clock.t += 60
        w.sweep(lambda r: None)
    assert launch_api.posts == [] and set(w.awaiting) == {strand, dsv4, failed}, "its coordinator is ACTIVE: it may resume them"
    assert not [r for r in records(tmp_path) if r["kind"] in ("relaunch", "not_relaunched")]
    api._agents[-1]["status"] = "IDLE"
    clock.t += 60
    w.sweep(lambda r: None)
    declined = {r["agent"]: r for r in records(tmp_path) if r["kind"] == "not_relaunched"}
    assert set(declined) == {strand, dsv4} and all(r["quiet"] and r["handle"] == "vllm" for r in declined.values())
    assert "its last message reads as a report back" in declined[strand]["why"] and "Nothing changed" in declined[strand]["why"]
    assert "or you can file it" in declined[dsv4]["why"]
    assert launch_api.posts == [] and set(w.awaiting) == {failed}, "a conversation the API won't read is never a relaunch"
    api.convs[failed] = [("user_message", "brief"), ("assistant_message", "Rerunning the suites on cpu-3.")]
    clock.t += 60
    w.sweep(lambda r: None)
    ((_, body),) = launch_api.posts
    (rec,) = [r for r in records(tmp_path) if r["kind"] == "relaunch"]
    assert rec["agent"] == failed and "its run r20261019-192229-3606 ended failed rc=3" in rec["reason"]
    assert "its run r20261019-192229-3606 ended failed rc=3 and its last message reports nothing" in body["prompt"]["text"]
    node_run(roots["vy-nebius-2"], "r20261019-200000-0009", clock.t, by=url(failed), state="failed", rc=1)
    for _ in range(50):
        clock.t += 10 * 60
        w.sweep(lambda r: None)
    assert len(launch_api.posts) == 1 and len([r for r in records(tmp_path) if r["kind"] == "relaunch"]) == 1, \
        "never relaunched twice for one idle"
    assert [r["agent"] for r in records(tmp_path) if r["kind"] == "not_relaunched"] == [strand, dsv4], "each judged once"


@pytest.mark.parametrize("said, relaunch", [
    ("Nothing changed: the head stays at 1cac8bff78.", False),
    ("The report is `lanes/proofs/x.md`, status final, and it's synced.", False),
    ("Should I take items 1 to 4 next?", False),
    ("Head pushed.\n\n- tests pass", False),
    ("DONE", False),
    ("I set one wake-up, int7-verdict-2205, for 22:05Z. Then I'll read d798.", False),
    ("The lock didn't change. Committing the fix and rerunning the check on cpu-3.", True),
    ("One more direct query of the node:", True),
    ("Let me check the queue.", True),
    ("Everything passes on the node.", False),
])
def test_a_workers_last_message_reads_as_a_report_a_stop_mid_step_or_neither(said, relaunch):
    """From idle workers' last messages (`GET /v0/agents/{id}/conversation`): a report back runs to paragraphs,
    asks, or says final/report/nothing changed; a stop mid-step announces its next step; a wait it set itself is neither."""
    assert (not W.reported(said) and W.mid_step(said)) is relaunch


def cancelled(root, rid, reason):
    """The cancel_intent `research cancel` appends to a run's events.jsonl (remote.py's "(research cancel by BY)")."""
    with open(root / "runs" / rid / "events.jsonl", "a") as f:
        f.write(json.dumps({"kind": "sampler.tick", "pad": "x" * 200}) + "\n")
        f.write(json.dumps({"kind": "cancel_intent", "actor": "manual", "reason": reason, "signal": 15}) + "\n")


def test_a_run_its_own_lane_cancelled_is_posted_quietly_and_starts_no_clock(tmp_path):
    """A run that ended rc=143 after "superseded by ... at its final head ... (research cancel by ubuntu)" (no --by: the
    user's name) is not rung for resuming or relaunched. A cancel whose reason or BY names the sender, or whose BY names its handle or an agent of it, is the
    lane's own: recorded quietly (no ring) with nothing to resume. Another lane's cancel is a failure like any."""
    salt, mine, other = W1_, W2_, W3_
    vllm_workers(tmp_path, salt, mine, other)
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api.workers[salt] = worker("IDLE", W.iso(T0 - 600), "SALT-WIRE: salt tag, Proxy order, PR")
    api.workers[mine] = worker("IDLE", W.iso(T0 - 600), "STRAND: land #2822")
    api.workers[other] = worker("IDLE", W.iso(T0 - 600), "WIDTHS: template widths")
    w.sweep(lambda r: None)
    clock.t += 60
    url, root = "https://cursor.com/agents/{}".format, roots["vy-nebius-2"]
    for rid, a, why in (("r20261019-182846-7d10", salt, ("superseded by r20261019-194802-65ae at SALT-WIRE's final head "
                                                         "5ea25a5ae5; frees 64 GB (research cancel by ubuntu)")),
                        ("r20261019-182846-0002", mine, "rerun at the new head (research cancel by vllm)"),
                        ("r20261019-182846-0003", other, f"frees the node for {NEW[:11]}'s train (research cancel by circuits)")):
        node_run(root, rid, clock.t, by=url(a), state="failed", rc=143)
        cancelled(root, rid, why)
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    got = {r["agent"]: r for r in ended(tmp_path)}
    assert got[salt]["quiet"] and got[salt]["cancelled_by"] == "ubuntu" and "SALT-WIRE's final head" in got[salt]["cancel_reason"]
    assert got[mine]["quiet"] and got[mine]["cancelled_by"] == "vllm"
    assert "quiet" not in got[other] and got[other]["role"] == "worker", "circuits cancelled vllm's run: a failure, rung"
    assert set(w.awaiting) == {other}, "a run its own lane cancelled starts no clock"


def test_a_sender_no_route_knows_is_learned_from_its_brief(tmp_path):
    """A worker whose Task call came before the watch followed its coordinator runs under nobody's route unless its brief
    ("a cloud worker of the circuits coordinator (agent ...)", "a worker for the proofs lane") is read, and a learned worker
    must not be forgotten while it works. A sender no route knows is read for its
    brief (at most `LEARN_PER_READ` a node read, again only `LEARN_RETRY_S` later), and kept while it sends runs."""
    widths, zks10 = "bc-c8838f5a-e10e-54ad-be4f-bbad4534f555", "bc-5f0b2ba2-1013-5ed2-af26-b6fac7d3bc65"
    strangers = [f"bc-{c * 8}-{c * 4}-{c * 4}-{c * 4}-{c * 12}" for c in "789a0f"]
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api.workers[widths] = worker("IDLE", W.iso(T0 - 600), "WIDTHS: served-row template width inventory")
    api.workers[zks10] = worker("ACTIVE", W.iso(T0 - 600), "ZKS10: OpeningsRefused restatement")
    api.convs[widths] = [("user_message", (f"You are WIDTHS, a cloud worker of the circuits coordinator (agent {CO}, Slack "
                                           "@circuits, Linear team CIR). First reread AGENTS.md."))]
    api.convs[zks10] = [("user_message", "You're ZKS10, a worker for the vllm lane (@vllm) of Verity; ask @infra for nodes.")]
    w.sweep(lambda r: None)
    clock.t += 60
    url = "https://cursor.com/agents/{}".format
    node_run(roots["vy-nebius-1"], "r20261019-200004-8870", clock.t, by=url(widths), state="failed", rc=1)
    node_run(roots["vy-nebius-1"], "r20261019-213533-32a6", clock.t, by=url(zks10), command=["x", "--by", "vllm"])
    for i, s in enumerate(strangers):
        node_run(roots["vy-nebius-2"], f"r20261019-21000{i}-000{i}", clock.t, by=url(s))
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    got = {r["agent"]: (r["handle"], r["role"], r["name"], r.get("parent")) for r in ended(tmp_path) if r["role"] != "unknown"}
    assert got == {widths: ("circuits", "worker", "WIDTHS: served-row template width inventory", CO),
                   zks10: ("vllm", "worker", "ZKS10: OpeningsRefused restatement", NEW)}
    assert widths in w.awaiting, "a worker's run end starts its clock, not its coordinator's"
    def briefs():
        return [g for g in api.gets if g.endswith("/conversation") and g.split("/")[-2] in strangers]
    assert len(briefs()) == W.LEARN_PER_READ, "at most LEARN_PER_READ briefs a node read"
    clock.t += 3600
    for i, s in enumerate(strangers[:W.LEARN_PER_READ]):
        node_run(roots["vy-nebius-2"], f"r20261019-23000{i}-001{i}", clock.t, by=url(s))
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    assert len(briefs()) == W.LEARN_PER_READ, "a brief that named no coordinator is not read again soon"
    clock.t += W.WORKER_KEEP_S - 2 * 3600
    node_run(roots["vy-nebius-1"], "r20261021-190000-0001", clock.t, by=url(widths))
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    clock.t += 2 * 3600
    w.sweep(lambda r: None)
    assert widths in w.children, "a worker still sending runs is kept past WORKER_KEEP_S from its learning"


def test_a_sender_no_route_knows_is_watched_under_no_lane_until_a_route_claims_it(tmp_path):
    """A worker its coordinator never registered, whose brief names no coordinator, was rung to infra once at a run's end and
    never watched: pous-prf hung for hours unseen. Watched as infra's own worker, it then told infra to resume a worker only
    its launcher can, and each run of its rang infra again. Such a sender is a stray, watched under no lane: infra is rung
    for it once, at that run's end, and its idle and its later runs are recorded with no handle, to ring no one (never
    resumed or relaunched as anyone's). A workers.d entry claims it."""
    stray = "bc-abababab-abab-abab-abab-abababababab"
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api.workers[stray] = worker("ACTIVE", W.iso(T0 - 600), "pous-prf: hiding budget per leaf")
    api.convs[stray] = [("user_message", "You are pous-prf. Read the note and go on.")]
    w.sweep(lambda r: None)
    clock.t += 60
    node_run(roots["vy-nebius-1"], "r20261105-110000-0001", clock.t, by=f"https://cursor.com/agents/{stray}", state="failed", rc=1)
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    (rec,) = [r for r in ended(tmp_path) if r["role"] == "unknown"]
    assert "quiet" not in rec and W.stray(w.children[stray]) and stray not in w.awaiting, "rung to infra once, no clock started"
    clock.t += 60
    w.sweep(lambda r: None)
    assert (w.workers[stray]["handle"], w.workers[stray]["lane"]) == (None, ""), "nobody's: no lane is made up for it"
    api.workers[stray] = worker("IDLE", W.iso(clock.t), "pous-prf: hiding budget per leaf")
    for _ in range(W.BURST_S // 60 + 2):                    # a worker's idle is held to join its parent's others (`bursts`)
        clock.t += 60
        w.sweep(lambda r: None)
    (idle,) = [r for r in records(tmp_path) if r["kind"] in ("worker_idle", "workers_failed") and stray in json.dumps(r)]
    assert (idle["kind"], idle["lane"], idle.get("handle"), idle["quiet"]) == ("worker_idle", "", None, True)
    sends(roots, clock, w, stray, "r20261105-120000-0002")
    assert [r.get("quiet") for r in ended(tmp_path) if r["role"] == "unknown"] == [None, True], "its next run rings no one"
    assert not [r for r in records(tmp_path) if r["kind"] == "unplaced"], "infra was rung for it at its first run's end"
    (tmp_path / "workers.d").mkdir(exist_ok=True)
    (tmp_path / "workers.d" / f"{stray}.json").write_text(json.dumps({"agent_id": stray, "lane": "vllm", "launched_by": NEW}))
    clock.t += 60
    w.sweep(lambda r: None)
    assert stray not in w.children and w.workers[stray]["handle"] == "vllm"


def test_a_listed_stray_stays_unowned_and_is_never_resumed_as_infras(tmp_path):
    """A stray the agent list has (launched through the API) read as infra's owned agent: its next run routed as infra's worker
    and the watch resumed it. A stray has no owner: its runs stay unowned, and nothing resumes it."""
    stray = "bc-cdcdcdcd-cdcd-cdcd-cdcd-cdcdcdcdcdcd"
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api._agents.append({"id": stray, "name": "an API-launched agent", "status": "IDLE", "latestRunId": "run-s",
                        "updatedAt": W.iso(T0 - 1800)})
    api._runs["run-s"] = dict(FINISHED)
    api.convs[stray] = [("user_message", "Do the thing.")]
    w.sweep(lambda r: None)
    for i in range(2):
        clock.t += 60
        node_run(roots["vy-nebius-1"], f"r20261105-11000{i}-000{i}", clock.t, by=f"https://cursor.com/agents/{stray}")
        clock.t += 5 * 60
        w.sweep(lambda r: None)
    assert stray not in w.owners and stray not in w.awaiting
    assert all(r["role"] == "unknown" for r in ended(tmp_path)), "both runs are unowned"
    while clock.t < T0 + 3 * 3600:
        clock.t += 300
        w.sweep(lambda r: None)
    assert not [p for p, _ in api.posts if p == f"/v1/agents/{stray}/runs"], "never resumed"


def idles(tmp_path, agent):
    return [r for r in records(tmp_path) if r["kind"] == "worker_idle" and r["agent"] == agent]


def goes_idle(api, clock, w, agent, name):
    """`agent`, which the watch read ACTIVE, stops; its idle is posted once nothing has joined it (`bursts`)."""
    api.workers[agent] = worker("IDLE", W.iso(clock.t), name)
    for _ in range(W.BURST_QUIET_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)


def test_a_worker_is_its_launchers_by_the_task_call_then_its_registrations_then_its_briefs(tmp_path):
    """A lane's reviewers and workers were reported as another lane's, whose lead can't resume them, and their own lead got no
    ring: a worker the watch first met as a run's sender kept the lane that meeting gave it after the watch saw its
    coordinator's Task call. A worker is the lane's of the agent that launched or resumed it with a Task call the watch
    saw, whatever its registration (a note filed under the lane it reviews for) or its brief (which names the coordinator it
    is to ask) say; a registration places it over its brief; its brief only when nothing else has."""
    names = {W1_: "review-7", W2_: "FASTWARM: warm the cache for the new head", W3_: "ROWS: the table's rows"}
    notes = tmp_path / "notes"
    note(notes, "circuits", "reviewers/review-7.md", f"review-7 ({W1_}), reviewing for circuits", lane_field=False)
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "w4.json").write_text(json.dumps({"agent_id": W4_, "lane": "zk-port", "launched_by": NEW}))
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api.workers[W4_] = worker("ACTIVE", W.iso(T0 - 600), "PORT")
    for agent, name in names.items():
        api.workers[agent] = worker("ACTIVE", W.iso(T0 - 600), name)
    brief = f"You are a worker; ask the circuits coordinator (agent {CO}) for a node when you need one."
    api.convs[W2_] = api.convs[W3_] = [("user_message", brief)]
    w.sweep(lambda r: None)
    assert (w.workers[W1_]["lane"], w.workers[W1_]["handle"]) == ("circuits", "circuits"), "by its note, until its launcher is seen"
    sends(roots, clock, w, W2_, "r20261105-110000-0001")
    sends(roots, clock, w, W3_, "r20261105-110000-0002")
    assert [(r["agent"], r["handle"], r["role"]) for r in ended(tmp_path)] == [(W2_, "circuits", "worker"),
                                                                             (W3_, "circuits", "worker")], "by their brief"
    clock.t += 60
    w.sweep(lambda r: None)
    assert w.workers[W2_]["handle"] == w.workers[W3_]["handle"] == "circuits" and w.children[W2_]["brief"]
    for agent in (W1_, W2_, W4_):
        w.child(NEW, {"args": {"agentId": agent, "description": names.get(agent, "PORT")}}, clock.t)
    (tmp_path / "workers.d" / "w3.json").write_text(json.dumps({"agent_id": W3_, "lane": "vllm", "launched_by": NEW}))
    clock.t += 60
    w.sweep(lambda r: None)
    assert {a: (w.workers[a]["lane"], w.workers[a]["handle"]) for a in names} == dict.fromkeys(names, ("vllm", "vllm"))
    assert (w.workers[W4_]["lane"], w.workers[W4_]["handle"]) == ("zk-port", "vllm"), "its own lane under its launcher's handle"
    assert "brief" not in w.children[W2_], "seen launched or resumed: more than its brief's word"
    for agent, name in names.items():
        goes_idle(api, clock, w, agent, name)
    assert [(r["lane"], r["handle"], r.get("quiet")) for a in names for r in idles(tmp_path, a)] == [("vllm", "vllm", None)] * 3
    assert not [r for r in records(tmp_path) if r["kind"] == "unplaced" or (r["kind"] == "run_ended" and r["role"] == "unknown")]


def test_a_worker_nothing_places_rings_infra_once_and_its_alerts_ring_no_one(tmp_path):
    """A worker whose launcher the watch can't find was written down as infra's and reported so, each time it stopped. With
    no Task call by an agent a handle owns, no registration and no lane, it is nobody's: one `unplaced` record, for infra to
    find its launcher, across a restart too; its idle, alone or with its parent's others, is recorded with no handle or
    lane, to ring no one. Once a workers.d entry places it, its idle is its lane's."""
    person, kids = "bc-99999999-9999-9999-9999-999999999999", {W1_: "an agent a person's agent launched", W2_: "and another"}
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    for kid, name in kids.items():
        api.workers[kid] = worker("ACTIVE", W.iso(T0 - 600), name)
        w.child(person, {"args": {"agentId": kid, "description": name}}, clock.t)
    for _ in range(3):
        w.sweep(lambda r: None)
        clock.t += 60

    def unplaced():
        return [(r["agent"], r["name"], r["role"], r["lane"], r.get("handle"), r["status"], r["parent"])
                for r in records(tmp_path) if r["kind"] == "unplaced"]
    assert sorted(unplaced()) == [(kid, name, "worker", "", None, "ACTIVE", person) for kid, name in kids.items()], \
        "once each, however often it is read"
    assert {(w.workers[kid]["lane"], w.workers[kid]["handle"]) for kid in kids} == {("", None)}
    w = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                notes=tmp_path / "notes", nodes=tmp_path / "nodes.json", sh=sh)
    goes_idle(api, clock, w, W1_, kids[W1_])
    assert len(unplaced()) == 2, "nor again after a restart"
    assert [(r["lane"], r.get("handle"), r.get("quiet")) for r in idles(tmp_path, W1_)] == [("", None, True)]
    api.workers[W1_] = worker("ACTIVE", W.iso(clock.t), kids[W1_])
    clock.t += 60
    w.sweep(lambda r: None)
    for kid, name in kids.items():
        api.workers[kid] = worker("IDLE", W.iso(clock.t), name)
    goes_idle(api, clock, w, W1_, kids[W1_])
    assert [(r["lane"], r.get("handle"), r.get("quiet")) for r in idles(tmp_path, W1_)][1:] == [], "held with its parent's others"
    (both,) = [r for r in records(tmp_path) if r["kind"] == "workers_failed"]
    assert (both["agent"], both["lane"], both.get("handle"), both.get("owner"), both["quiet"]) == (person, "", None, None, True)
    assert sorted(m["agent"] for m in both["members"]) == sorted(kids)
    assert len(unplaced()) == 2 and api.posts == []
    (tmp_path / "workers.d").mkdir(exist_ok=True)
    (tmp_path / "workers.d" / "kid.json").write_text(json.dumps({"agent_id": W2_, "lane": "vllm", "launched_by": NEW}))
    api.workers[W2_] = worker("ACTIVE", W.iso(clock.t), kids[W2_])
    for _ in range(W.BURST_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)
    goes_idle(api, clock, w, W2_, kids[W2_])
    assert [(r["lane"], r.get("handle"), r.get("quiet")) for r in idles(tmp_path, W2_)] == [("vllm", "vllm", None)], \
        "placed by its registration: its lane's from then on"
    for kid, name in kids.items():
        api.workers[kid] = worker("ACTIVE", W.iso(clock.t), name)
    for _ in range(W.BURST_S // 60 + 2):
        clock.t += 60
        w.sweep(lambda r: None)
    for kid, name in kids.items():
        api.workers[kid] = worker("IDLE", W.iso(clock.t), name)
    goes_idle(api, clock, w, W1_, kids[W1_])
    mixed = [r for r in records(tmp_path) if r["kind"] == "workers_failed"][1:]
    assert [("quiet" in r, len(r["members"])) for r in mixed] == [(False, 2)], "one of them has a lane: its idle is never hidden"


def test_a_lane_an_older_watch_made_up_for_a_worker_is_dropped_for_what_places_it(tmp_path):
    """State an older watch wrote: a run's sender no handle owned was kept as infra's worker, and stayed infra's after the
    watch saw its launcher. Read back, a stray is nobody's (infra was rung for it at its run's end then: not again, nor at
    its next run), a worker with a launcher is its launcher's lane's, and a worker infra's coordinator registered is still
    infra's."""
    infra, stray, kid, own = "bc-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", W1_, W2_, W3_
    made_up = {"lane": "infra", "handle": "infra", "heard": T0 - 60, "read": T0 - 60, "status": "ACTIVE", "updated": T0 - 600,
               "since": T0 - 600, "long": False}
    (tmp_path / "state.json").write_text(json.dumps({
        "children": {stray: {"parent": "", "name": "", "t": T0 - 60, "owner": "infra"},
                     kid: {"parent": NEW, "name": "vllm-rows: the rows", "t": T0 - 60, "owner": "infra"}},
        "workers": {stray: {**made_up, "parent": "", "name": "review-9"},
                    kid: {**made_up, "parent": NEW, "name": "vllm-rows: the rows"},
                    own: {**made_up, "name": "a worker of infra's"}}}))
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / "own.json").write_text(json.dumps({"agent_id": own, "lane": "infra", "launched_by": infra}))
    api, clock, w, _, roots = nodes_setup(tmp_path)
    reg = json.loads((tmp_path / "registry.json").read_text())
    reg["handles"].append({"handle": "infra", "holder": f"infra's coordinator ({infra})"})
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    for agent in (stray, kid, own):
        api.workers[agent] = worker("ACTIVE", W.iso(T0 - 600), w.workers[agent]["name"])
    w.sweep(lambda r: None)
    assert {a: (w.workers[a]["lane"], w.workers[a]["handle"]) for a in (stray, kid, own)} == {
        stray: ("", None), kid: ("vllm", "vllm"), own: ("infra", "infra")}
    for agent in (stray, kid, own):
        goes_idle(api, clock, w, agent, w.workers[agent]["name"])
    assert {a: [(r["lane"], r.get("handle"), r.get("quiet")) for r in idles(tmp_path, a)] for a in (stray, kid, own)} == {
        stray: [("", None, True)], kid: [("vllm", "vllm", None)], own: [("infra", "infra", None)]}
    sends(roots, clock, w, stray, "r20261105-110000-0001")
    assert [(r["role"], r["quiet"]) for r in ended(tmp_path)] == [("unknown", True)]
    assert not [r for r in records(tmp_path) if r["kind"] == "unplaced"]


KEYED = "bc-efefefef-efef-efef-efef-efefefefefef"


def linear_teams(tmp_path):
    """The registry's lanes, each a handle with a Linear team that a cloud agent holds: circuits and vllm with a team of their
    own, two that share one (compute-accounting's own, which compute-research files in under its `linear_label`), one whose
    name extends vllm's, and kernels, renamed from another name. Three handles are no
    lane: console (no team, a desktop agent's), dashboard (a team, a desktop agent's) and steward (a cloud agent's, no team)."""
    reg = json.loads((tmp_path / "registry.json").read_text())
    for h in reg["handles"]:
        h.update({"linear": team} if (team := {"circuits": "CIR", "vllm": "VLLM"}.get(h["handle"])) else {})
    reg["handles"] += [
        {"handle": "compute-accounting", "holder": f"its coordinator ({W4_})", "linear": "CMP"},
        {"handle": "compute-research", "holder": f"its coordinator ({W5_})", "linear": "CMP",
         "linear_label": "lane:compute-research"},
        {"handle": "vllm-serve", "holder": f"its coordinator ({W6_})", "linear": "SRV"},
        {"handle": "kernels", "holder": f"its coordinator ({W3_})", "linear": "HASH", "charter": "renamed from @hash"},
        {"handle": "dashboard", "holder": "a desktop agent (4d92fa93-101a-42e1-8282-d4a998892829)", "linear": "CON"},
        {"handle": "steward", "holder": f"its agent ({W2_})"}]
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    return tmp_path / "registry.json"


def sends(roots, clock, w, agent, rid, **kw):
    """A `research run` sent by `agent` ends a minute on, and the watch reads the nodes 5 min later."""
    clock.t += 60
    node_run(roots["vy-nebius-1"], rid, clock.t, by=f"https://cursor.com/agents/{agent}", **kw)
    clock.t += 5 * 60
    w.sweep(lambda r: None)


def placed_files(tmp_path):
    return {p.name: json.loads(p.read_text()) for p in (tmp_path / "workers.d").glob("placed-*.json")}


def by_key(handle, key, **named):
    return handle, {"placed_by": "name", "issue": key, **named}


def test_an_issue_key_at_a_names_start_names_its_lane_only_when_one_lane_has_its_team(tmp_path):
    coordinators(tmp_path, [], {})
    reg = linear_teams(tmp_path)
    assert W.name_lane("VLLM-97 KFLAGS: the committer's flag, byte comparison", reg) == by_key("vllm", "VLLM-97")
    assert W.name_lane("CIR-206: one calibration", reg) == by_key("circuits", "CIR-206")
    assert W.name_lane("VLLM-7", reg) == by_key("vllm", "VLLM-7")
    assert W.name_lane("VLLM-97 KFLAGS, after CIR-3 landed", reg) == by_key("vllm", "VLLM-97"), \
        "the key it starts with is its issue"
    for name in ("VLLM-97x KFLAGS", "VLLM97 KFLAGS", "VLLM- KFLAGS", "XVLLM-97 KFLAGS", "Fix the audit", ""):
        assert W.name_lane(name, reg) is None, f"{name!r} holds no issue key of a team a lane has"
    assert W.name_lane("CMP-377 HARDNESS-SPEC", reg) is None, "two lanes file under this team: never a guess"
    assert W.name_lane("NOPE-12 a team no handle has", reg) is None
    assert W.name_lane("CON-12 the goals page", reg) is None, "its team's handle is a desktop agent's: no lane"
    assert W.name_lane("VLLM-97 KFLAGS", None) is None and W.name_lane("VLLM-97 KFLAGS", tmp_path / "absent.json") is None


def test_issue_keys_further_in_a_name_name_its_lane_when_they_are_all_one_teams(tmp_path):
    """A worker named with its label first (`LABEL: TEAM-7 step 2`) went to infra to place by hand, though anyone reading
    the name knew its issue. Keys further in are read when the name holds one team's only: with two teams' keys nothing
    says which is its issue, and what the name starts with, a key or a handle, is read before any of them."""
    coordinators(tmp_path, [], {})
    reg = linear_teams(tmp_path)
    for name, key in (("TRIM: VLLM-522 step 2", "VLLM-522"), ("KFORK: the forked committer (VLLM-61)", "VLLM-61"),
                      ("graph2 CIR-489 fixes", "CIR-489"), ("[VLLM-97] KFLAGS", "VLLM-97"), (" VLLM-97 KFLAGS", "VLLM-97"),
                      ("ROWS: VLLM-3, then VLLM-9", "VLLM-3"), ("FOLD: SHA-512 rows for VLLM-8", "VLLM-8"),
                      ("seat of NOPE-12: VLLM-8", "VLLM-8"), ("KFORK/VLLM-61-retry", "VLLM-61")):
        assert W.name_lane(name, reg) == by_key({"VLLM": "vllm", "CIR": "circuits"}[key.split("-")[0]], key), name
    for name in ("REVIEW: VLLM-61 against CIR-489", "ROWS: CIR-3 (was VLLM-9)", "PAGE: VLLM-8 for CON-12"):
        assert W.name_lane(name, reg) is None, f"{name!r} holds keys of two teams"
    for name in ("ROWS: xVLLM-9", "ROWS: pre-VLLM-9", "ROWS: VLLM-9x", "ROWS: VLLM_9", "ROWS: vllm-9", "ROWS: SEAT-VLLM-9",
                 "ROWS: NOPE-12", "PAGE: CON-12"):
        assert W.name_lane(name, reg) is None, f"{name!r} holds no whole issue key of a team a lane has"
    assert W.name_lane("SEAT-A: CMP-377", reg) is None, "two lanes file under this team: never a guess"
    assert W.name_lane("SEAT-A: CMP-377 (compute-research)", reg) == by_key(
        "compute-research", "CMP-377", named="compute-research"), "a shared team's key places by the one lane the name names"
    assert W.name_lane("vllm-metrics: the flat list for CIR-389", reg) == ("vllm", {"placed_by": "handle"}), \
        "the handle it starts with is its lane, and the key further in an issue it works with"
    assert W.name_lane("TRIM: VLLM-522 step 2", None) is None


def test_a_key_of_a_team_two_lanes_share_places_only_when_the_name_also_names_exactly_one_of_them(tmp_path):
    coordinators(tmp_path, [], {})
    reg = linear_teams(tmp_path)
    assert W.name_lane("CMP-377 HARDNESS-SPEC (compute-research)", reg) == by_key(
        "compute-research", "CMP-377", named="compute-research")
    assert W.name_lane("CMP-12 compute-accounting: the table", reg) == by_key(
        "compute-accounting", "CMP-12", named="compute-accounting")
    for name in ("CMP-12 compute-research answers compute-accounting", "CMP-12 compute-researcher", "CMP-12 compute-research-x",
                 "CMP-12 xcompute-research", "CMP-12 pre-compute-research", "CMP-12 vllm: its rows", "CMP-12 Compute-Research"):
        assert W.name_lane(name, reg) is None, f"{name!r} names both, or neither as a whole word"
    for name in ("VLLM-97 compute-research: its rows", "VLLM-97 vllm: its rows"):
        assert W.name_lane(name, reg) == by_key("vllm", "VLLM-97"), \
            "a team of one lane's own needs no second word, and a handle further in changes nothing"


def test_a_leading_handle_names_its_lane_whole_the_longest_first_and_only_a_lane_that_runs_workers(tmp_path):
    coordinators(tmp_path, [], {})
    reg = linear_teams(tmp_path)
    led = {"placed_by": "handle"}
    assert W.placing(reg) == {"circuits": "CIR", "vllm": "VLLM", "compute-accounting": "CMP", "compute-research": "CMP",
                              "vllm-serve": "SRV", "kernels": "HASH"}
    for name, handle in (("vllm-longrow-s1b: one served Count staged and measured", "vllm"), ("compute-research: x", "compute-research"),
                         ("compute-accounting-y", "compute-accounting"), ("vllm: x", "vllm"), ("vllm/x", "vllm"), ("vllm x", "vllm"),
                         ("vllm\tx", "vllm"), ("vllm-97 KFLAGS", "vllm"), ("vllm-metrics: the flat list for CIR-389", "vllm"),
                         ("vllm-serve-rows: x", "vllm-serve"), ("vllm-serve: x", "vllm-serve"), ("vllm-server: x", "vllm"),
                         ("kernels-fold: x", "kernels")):
        assert W.name_lane(name, reg) == (handle, led), name
    for name in ("computex: z", "compute: z", "compute-x: z", "vllm", "vllmx: y", "vllm_x", "vllm.x", "Vllm-x", "VLLM: x",
                 " vllm-x", "the vllm-x", "review-1: a reviewer"):
        assert W.name_lane(name, reg) is None, f"{name!r} starts with no whole handle"
    assert W.name_lane("hash-fold: x", reg) is None, "a retired name is no handle, whatever was renamed from it"
    for name in ("console-panels: x", "dashboard-goals: x", "steward-sync: x"):
        assert W.name_lane(name, reg) is None, f"{name!r} starts with a handle that is no lane"
    assert W.name_lane("CIR-206 vllm-rows: x", reg) == by_key("circuits", "CIR-206"), "an issue key at the start wins"
    assert W.name_lane("vllm-longrow-s1b: x", None) is None


@pytest.mark.parametrize("name", ["VLLM-97 KFLAGS: the committer's flag, byte comparison", "KFLAGS: VLLM-97 step 2"])
def test_an_unowned_sender_named_after_one_lanes_issue_is_placed_as_that_lanes_worker(tmp_path, name):
    """A worker its coordinator never registered, whose brief names no coordinator, went to infra to place by hand at its
    first run's end, though its name began with its lane's issue, or held it after its label. The watch places it: its
    workers.d entry, marked as inferred, and that run recorded for the lane with how it was found. From then on it is any
    worker of that lane."""
    api, clock, w, sh, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    api.workers[KEYED] = worker("IDLE", W.iso(T0 - 600), name)
    api.convs[KEYED] = [("user_message", "You are KFLAGS. Read the note and go on.")]
    w.sweep(lambda r: None)
    sends(roots, clock, w, KEYED, "r20261105-110000-0001", state="failed", rc=1)
    (rec,) = ended(tmp_path)
    assert {k: rec.get(k) for k in ("agent", "name", "handle", "role", "lane", "placed_by", "issue", "sender_status")} == {
        "agent": KEYED, "name": name, "handle": "vllm", "role": "worker", "lane": "vllm", "placed_by": "name",
        "issue": "VLLM-97", "sender_status": "IDLE"}
    assert placed_files(tmp_path) == {f"placed-{KEYED}.json": {
        "agent_id": KEYED, "lane": "vllm", "role": "worker", "name": name, "placed_by": "name", "issue": "VLLM-97"}}
    assert api.gets.count(f"/v1/agents/{KEYED}") == 1, "one read of a sender the watch meets for the first time"
    assert KEYED not in w.children and w.workers[KEYED]["handle"] == "vllm" and KEYED in w.awaiting, \
        "never a stray: its lane's worker, whose coordinator resumes it"
    sends(roots, clock, w, KEYED, "r20261105-120000-0002")
    later = ended(tmp_path)[-1]
    assert (later["run"], later["handle"], later["role"], later["lane"]) == ("r20261105-120000-0002", "vllm", "worker", "vllm")
    assert "placed_by" not in later and len(ended(tmp_path)) == 2, "placed once: its later runs are a worker's like any other"
    restarted = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                        notes=tmp_path / "notes", nodes=tmp_path / "nodes.json", sh=sh)
    sends(roots, clock, restarted, KEYED, "r20261105-130000-0003")
    assert [(r["handle"], r["role"], r.get("placed_by")) for r in ended(tmp_path)][-1] == ("vllm", "worker", None)
    assert not [r for r in ended(tmp_path) if r["role"] == "unknown"], "infra is never rung for it"


def test_a_name_that_names_no_one_lane_places_nobody_and_infra_is_rung_once(tmp_path):
    """No guess: a name with issue keys of two lanes, a key of a team no handle has, a key of a team two lanes share, a
    name that starts with part of a handle, with a retired name, or with a handle that is no lane leave the sender unowned,
    its runs one record for infra and nothing written; a later run of a sender infra was rung for rings no one."""
    senders = {"bc-a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1": "KFORK: the forked committer (VLLM-61, for CIR-9)",
               "bc-a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2": "NOPE-12 a team no handle has",
               "bc-a3a3a3a3-a3a3-a3a3-a3a3-a3a3a3a3a3a3": "CMP-377 HARDNESS-SPEC",
               "bc-a4a4a4a4-a4a4-a4a4-a4a4-a4a4a4a4a4a4": "computex: z",
               "bc-a5a5a5a5-a5a5-a5a5-a5a5-a5a5a5a5a5a5": "hash-fold: the retired name",
               "bc-a6a6a6a6-a6a6-a6a6-a6a6-a6a6a6a6a6a6": "dashboard-goals: a desktop agent's handle"}
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    for agent, name in senders.items():
        api.workers[agent] = worker("IDLE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    clock.t += 60
    for i, agent in enumerate(senders):
        node_run(roots["vy-nebius-1"], f"r20261105-11000{i}-000{i}", clock.t, by=f"https://cursor.com/agents/{agent}")
    clock.t += 5 * 60
    w.sweep(lambda r: None)
    (rec,) = ended(tmp_path)
    assert (rec["role"], rec["handle"], rec["count"]) == ("unknown", None, 6) and "placed_by" not in rec
    assert rec["senders"] == sorted(f"https://cursor.com/agents/{a}" for a in senders)
    assert all(W.stray(w.children[a]) for a in senders) and not w.awaiting and "quiet" not in rec
    assert placed_files(tmp_path) == {}
    clock.t += 60
    w.sweep(lambda r: None)
    sends(roots, clock, w, next(iter(senders)), "r20261105-120000-0009")
    assert [(r["role"], r.get("quiet")) for r in ended(tmp_path)] == [("unknown", None), ("unknown", True)] \
        and placed_files(tmp_path) == {}, "nor by the name the watch has read since"
    assert not [r for r in records(tmp_path) if r["kind"] == "unplaced"], "one ring for each: the first read's record"


def test_an_unowned_sender_whose_name_starts_with_its_lanes_handle_is_placed_as_that_lanes_worker(tmp_path):
    """A lane that names its workers with its handle first had each of them go to infra to place by hand. The handle places
    them as an issue key does, and the entry and the record say which it was; a key of a shared team places by the one
    sharing lane the name names."""
    led, shared = KEYED, "bc-a7a7a7a7-a7a7-a7a7-a7a7-a7a7a7a7a7a7"
    names = {led: "vllm-longrow-s1b: one served Count staged and measured", shared: "CMP-377 HARDNESS-SPEC (compute-research)"}
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    for agent, name in names.items():
        api.workers[agent] = worker("IDLE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    sends(roots, clock, w, led, "r20261105-110000-0001", state="failed", rc=1)
    sends(roots, clock, w, shared, "r20261105-110000-0002")
    first, second = ended(tmp_path)
    assert {k: first.get(k) for k in ("agent", "name", "handle", "role", "lane", "placed_by", "issue", "named")} == {
        "agent": led, "name": names[led], "handle": "vllm", "role": "worker", "lane": "vllm", "placed_by": "handle",
        "issue": None, "named": None}
    assert {k: second.get(k) for k in ("agent", "handle", "role", "lane", "placed_by", "issue", "named")} == {
        "agent": shared, "handle": "compute-research", "role": "worker", "lane": "compute-research", "placed_by": "name",
        "issue": "CMP-377", "named": "compute-research"}
    assert placed_files(tmp_path) == {
        f"placed-{led}.json": {"agent_id": led, "lane": "vllm", "role": "worker", "name": names[led], "placed_by": "handle"},
        f"placed-{shared}.json": {"agent_id": shared, "lane": "compute-research", "role": "worker", "name": names[shared],
                                  "placed_by": "name", "issue": "CMP-377", "named": "compute-research"}}
    assert w.workers[led]["handle"] == "vllm" and w.workers[shared]["handle"] == "compute-research" and not w.children
    sends(roots, clock, w, led, "r20261105-120000-0003")
    later = ended(tmp_path)[-1]
    assert (later["agent"], later["handle"], later["role"], later.get("placed_by")) == (led, "vllm", "worker", None)
    assert not [r for r in ended(tmp_path) if r["role"] == "unknown"], "infra is never rung for either"


def test_a_sender_whose_name_cant_be_read_is_nobodys_until_the_watch_has_its_name(tmp_path, capsys):
    """A failed read places nobody and is said in the log; the sender is a stray, read each sweep as any stray is, and the
    name that read gives places it at its next run's end, with no read of its own."""
    name = "VLLM-97 KFLAGS: the committer's flag, byte comparison"
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    api.workers[KEYED] = worker("ACTIVE", W.iso(T0 - 600), name)
    down, get = [True], api.get

    def flaky(path, **params):
        if down[0] and path == f"/v1/agents/{KEYED}":
            api.gets.append(path)
            raise TimeoutError("reads are past this sweep's budget")
        return get(path, **params)
    api.get = flaky
    w.sweep(lambda r: None)
    sends(roots, clock, w, KEYED, "r20261105-110000-0001", state="failed", rc=1)
    (rec,) = ended(tmp_path)
    assert rec["role"] == "unknown" and rec["senders"] == [f"https://cursor.com/agents/{KEYED}"]
    assert W.stray(w.children[KEYED]) and placed_files(tmp_path) == {}
    assert f"the name of {KEYED} was not read (TimeoutError: reads are past this sweep's budget)" in capsys.readouterr().err
    reads = api.gets.count(f"/v1/agents/{KEYED}")
    sends(roots, clock, w, KEYED, "r20261105-110000-0002")
    assert [r["role"] for r in ended(tmp_path)] == ["unknown", "unknown"], "still unread: still nobody's"
    assert api.gets.count(f"/v1/agents/{KEYED}") == reads + 1, "the sweep's read of a stray, and no second one for its name"
    down[0] = False
    clock.t += 60
    w.sweep(lambda r: None)
    assert w.workers[KEYED]["name"] == name and w.workers[KEYED]["handle"] is None, "read as a stray, nobody's so far"
    reads = api.gets.count(f"/v1/agents/{KEYED}")
    clock.t += 5 * 60
    node_run(roots["vy-nebius-1"], "r20261105-120000-0003", clock.t, by=f"https://cursor.com/agents/{KEYED}", rc=2, state="failed")
    w.run_ends(w.coords, clock.t)
    assert api.gets.count(f"/v1/agents/{KEYED}") == reads, "placed by the name the watch already has"
    rec = ended(tmp_path)[-1]
    assert (rec["handle"], rec["role"], rec["lane"], rec["placed_by"], rec["issue"]) == ("vllm", "worker", "vllm", "name", "VLLM-97")
    assert KEYED not in w.children and w.workers[KEYED]["handle"] == "vllm" and list(placed_files(tmp_path)) == [
        f"placed-{KEYED}.json"]


def test_a_coordinators_own_entry_always_wins_over_one_the_watch_inferred_from_a_name(tmp_path):
    """A sender its coordinator registered is never placed by its name, whatever the name says; and an entry a coordinator
    writes after the watch placed a sender moves it, whichever file sorts first."""
    name = "VLLM-97 KFLAGS: the committer's flag, byte comparison"
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / f"{W1_}.json").write_text(json.dumps({"agent_id": W1_, "lane": "circuits", "launched_by": CO}))
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    api.workers[W1_] = api.workers[KEYED] = worker("IDLE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    sends(roots, clock, w, W1_, "r20261105-110000-0001")
    sends(roots, clock, w, KEYED, "r20261105-110000-0002")
    assert [(r["agent"], r["handle"], r["role"], r.get("placed_by")) for r in ended(tmp_path)] == [
        (W1_, "circuits", "worker", None), (KEYED, "vllm", "worker", "name")]
    assert list(placed_files(tmp_path)) == [f"placed-{KEYED}.json"], "nothing is written for a registered sender"
    (tmp_path / "workers.d" / f"zz-{KEYED}.json").write_text(json.dumps({"agent_id": KEYED, "lane": "circuits", "launched_by": CO}))
    got = W.workers(None, tmp_path / "workers.d", tmp_path / "registry.json", w.coords, clock.t)[KEYED]
    assert (got["lane"], got["handle"], got["parent"]) == ("circuits", "circuits", CO)
    sends(roots, clock, w, KEYED, "r20261105-120000-0003")
    rec = ended(tmp_path)[-1]
    assert (rec["agent"], rec["handle"], rec["lane"], rec["role"], rec.get("placed_by")) == (KEYED, "circuits", "circuits", "worker", None)
    assert w.workers[KEYED]["handle"] == "circuits" and w.workers[KEYED]["parent"] == CO


def test_a_listed_sender_is_placed_by_the_name_the_list_gives_and_owned_by_its_lane(tmp_path):
    """An agent the list has is placed with no read of its own; it gets an owner, as one a workers.d entry names does."""
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    api._agents.append({"id": KEYED, "name": "CIR-206 CCALONE: one calibration", "status": "IDLE", "latestRunId": "run-s",
                        "updatedAt": W.iso(T0 - 1800)})
    api._runs["run-s"] = dict(FINISHED)
    api.convs[KEYED] = [("user_message", "Do the thing.")]
    w.sweep(lambda r: None)
    sends(roots, clock, w, KEYED, "r20261105-110000-0001")
    (rec,) = ended(tmp_path)
    assert (rec["handle"], rec["role"], rec["listed"], rec["placed_by"], rec["issue"]) == ("circuits", "worker", True, "name", "CIR-206")
    assert f"/v1/agents/{KEYED}" not in api.gets and w.owners[KEYED] == "circuits" and KEYED not in w.workers
    sends(roots, clock, w, KEYED, "r20261105-120000-0002")
    assert [(r["handle"], r["role"], r.get("placed_by")) for r in ended(tmp_path)][-1] == ("circuits", "worker", None)


MID, SEAT = "bc-0a0a0a0a-0a0a-0a0a-0a0a-0a0a0a0a0a0a", "bc-0b0b0b0b-0b0b-0b0b-0b0b-0b0b0b0b0b0b"
PERSON = "bc-99999999-9999-9999-9999-999999999999"


def unplaced(tmp_path):
    return [r["agent"] for r in records(tmp_path) if r["kind"] == "unplaced"]


def registers(tmp_path, agent, **entry):
    """A workers.d entry as a coordinator, or a worker for the workers it launches, writes it."""
    (tmp_path / "workers.d").mkdir(exist_ok=True)
    (tmp_path / "workers.d" / f"{agent}.json").write_text(json.dumps({"agent_id": agent, **entry}))


def launches(w, clock, parent, agent, name=""):
    """`parent`'s Task call for `agent`, as a run the watch follows streams it."""
    w.child(parent, {"args": {"agentId": agent, "description": name}}, clock.t)


@pytest.mark.parametrize("link", ["task", "registration", "brief"])
def test_a_sender_a_worker_launched_is_placed_under_that_workers_coordinator(tmp_path, link):
    """A swarm's seats, launched by its integrator, itself a worker, each went to infra to place by hand: a launcher's
    handle was read only where the launcher was a coordinator, or where an earlier sweep had written its handle down. A
    worker is its launcher's coordinator's however far up that is, by the one rule its run's end and its idle are both
    placed by, whether the watch knows its launcher from a Task call, its registration or its brief."""
    names = {MID: "INTEGRATE: the swarm's integrator", SEAT: "SEAT-A: the first seat"}
    api, clock, w, _, roots = nodes_setup(tmp_path)
    for agent, name in names.items():
        api.workers[agent] = worker("ACTIVE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    if link == "task":
        launches(w, clock, MID, SEAT, names[SEAT])          # seen before the call that launched its launcher
    elif link == "registration":
        registers(tmp_path, SEAT, launched_by=MID)
    else:
        api.convs[SEAT] = [("user_message", f"You are SEAT-A ({SEAT}), asked for by {PERSON}; report to the integrator "
                                            f"(agent {MID}) and to no one else.")]
    launches(w, clock, NEW, MID, names[MID])
    sends(roots, clock, w, SEAT, "r20261105-110000-0001", state="failed", rc=1)
    assert [(r["agent"], r["handle"], r["role"], r["name"], r.get("parent"), r.get("placed_by")) for r in ended(tmp_path)] == [
        (SEAT, "vllm", "worker", names[SEAT], MID, None)]
    clock.t += 60
    w.sweep(lambda r: None)
    assert {a: (w.workers[a]["lane"], w.workers[a]["handle"]) for a in names} == dict.fromkeys(names, ("vllm", "vllm"))
    goes_idle(api, clock, w, SEAT, names[SEAT])
    assert [(r["lane"], r["handle"], r.get("quiet")) for r in idles(tmp_path, SEAT)] == [("vllm", "vllm", None)]
    assert unplaced(tmp_path) == [] and placed_files(tmp_path) == {}, "infra is never rung for it, and no name placed it"


def test_a_registered_launcher_places_over_the_registrations_lane_and_a_worker_keeps_a_lane_of_its_own(tmp_path):
    """A registration's `launched_by` placed its worker only when it named a coordinator; with a worker there the entry's
    lane decided, or nothing. The launcher's coordinator places it, as a coordinator named there does: over a lane that is
    another handle's, and beside a lane that is no handle. A note's origin names whom its author worked for, not who
    launched it: a coordinator there never moves a worker out of its note's lane."""
    port, review, lone, noted = W1_, W2_, W3_, W4_
    registers(tmp_path, MID, launched_by=NEW)
    registers(tmp_path, port, lane="zk-port", launched_by=MID)
    registers(tmp_path, review, lane="circuits", launched_by=MID)
    registers(tmp_path, lone, launched_by=port)
    note(tmp_path / "notes", "vllm", "rows.md", f"ROWS ({noted}), asked for by circuits ({CO})")
    api, clock, w, _, roots = nodes_setup(tmp_path)
    for agent in (MID, port, review, lone, noted):
        api.workers[agent] = worker("ACTIVE", W.iso(T0 - 600), "a worker")
    w.sweep(lambda r: None)
    assert {a: (w.workers[a]["lane"], w.workers[a]["handle"]) for a in (port, review, lone, noted)} == {
        port: ("zk-port", "vllm"), review: ("vllm", "vllm"), lone: ("vllm", "vllm"), noted: ("vllm", "vllm")}
    sends(roots, clock, w, lone, "r20261105-110000-0001")
    assert [(r["agent"], r["handle"], r["role"], r["parent"]) for r in ended(tmp_path)] == [(lone, "vllm", "worker", port)]
    assert unplaced(tmp_path) == []


def test_a_sender_with_a_shared_teams_key_is_placed_by_its_launcher_and_one_with_no_launcher_rings_infra_once(tmp_path):
    """Two lanes file under one Linear team, so a seat named after its issue in that team is placed by no name. One whose
    launcher the watch knows is its launcher's lane's all the same; one nothing places is infra's to find, rung once: at
    its first run's end, or when the watch first read it as a worker whose launcher no handle owns."""
    lone, kid = "bc-0c0c0c0c-0c0c-0c0c-0c0c-0c0c0c0c0c0c", "bc-0d0d0d0d-0d0d-0d0d-0d0d-0d0d0d0d0d0d"
    seats = {SEAT: "CMP-41 SEAT-A", lone: "CMP-42 SEAT-B", kid: "CMP-43 SEAT-C"}
    api, clock, w, _, roots = nodes_setup(tmp_path)
    linear_teams(tmp_path)
    registers(tmp_path, MID, lane="compute-research", launched_by=W5_)
    for agent, name in {MID: "CMP-40 INTEGRATE", **seats}.items():
        api.workers[agent] = worker("IDLE", W.iso(T0 - 600), name)
        api.convs[agent] = [("user_message", f"You are a seat of the swarm. Report to the integrator (agent {MID})."
                             if agent == SEAT else "You are a seat of the swarm. Read the note and go on.")]
    launches(w, clock, PERSON, kid, seats[kid])
    w.sweep(lambda r: None)
    assert unplaced(tmp_path) == [kid], "a worker whose launcher no handle owns: rung once, as it is first read"
    for i, agent in enumerate(seats):
        sends(roots, clock, w, agent, f"r20261105-11000{i}-000{i}")
    by_seat = {r["agent"] or r["senders"][0][-len(SEAT):]: (r["handle"], r["role"], r.get("placed_by"), r.get("quiet"))
               for r in ended(tmp_path)}
    assert by_seat == {SEAT: ("compute-research", "worker", None, None), lone: (None, "unknown", None, None),
                       kid: (None, "unknown", None, True)}
    assert W.stray(w.children[lone]) and w.children[kid]["parent"] == PERSON and placed_files(tmp_path) == {}
    for i, agent in enumerate(seats):
        sends(roots, clock, w, agent, f"r20261105-12000{i}-001{i}")
    assert [(r["role"], r.get("quiet")) for r in ended(tmp_path)[3:]] == [("worker", None), ("unknown", True), ("unknown", True)]
    assert unplaced(tmp_path) == [kid], "one ring for each of the two nothing places, and none for the one its launcher does"


def test_launchers_that_name_each_other_place_nobody_and_a_chain_is_followed_only_so_far(tmp_path):
    """Following a launcher's launcher must end: a worker resumed by the worker it launched, or two registrations naming
    each other, are a loop with no coordinator on it, and place nobody; one of them placed some other way places both. A
    chain is followed `LAUNCHERS_MAX` agents up in one asking."""
    a, b = W1_, W2_
    api, clock, w, _, roots = nodes_setup(tmp_path)
    for agent in (a, b):
        api.workers[agent] = worker("ACTIVE", W.iso(T0 - 600), "one of a pair")
    launches(w, clock, a, b)
    launches(w, clock, b, a)
    registers(tmp_path, a, launched_by=b)
    registers(tmp_path, b, launched_by=a)
    api.convs[a] = [("user_message", f"Report to {b}, and to yourself ({a}).")]
    coords = W.holders(tmp_path / "registry.json", tmp_path / "routes.d")
    asked, placed = [], w.placed
    w.placed = lambda agent, *args, **kw: asked.append(agent) or placed(agent, *args, **kw)
    assert w.placed(a, coords) is None and len(asked) <= 7, "a loop is left where it closes, not walked as far as a chain is"
    assert w.placed(b, coords) is None
    w.sweep(lambda r: None)
    sends(roots, clock, w, a, "r20261105-110000-0001")
    assert sorted(unplaced(tmp_path)) == [a, b] and [(r["role"], r.get("quiet")) for r in ended(tmp_path)] == [("unknown", True)]
    assert {(w.workers[x]["lane"], w.workers[x]["handle"]) for x in (a, b)} == {("", None)}
    registers(tmp_path, b, lane="vllm")
    sends(roots, clock, w, a, "r20261105-120000-0002")
    assert [(r["agent"], r["handle"], r["role"]) for r in ended(tmp_path)][1:] == [(a, "vllm", "worker")]
    assert {(w.workers[x]["lane"], w.workers[x]["handle"]) for x in (a, b)} == {("vllm", "vllm")}
    chain = [f"bc-{i:08x}-0000-0000-0000-00000000000c" for i in range(W.LAUNCHERS_MAX + 1)]
    for kid, parent in zip(chain, [*chain[1:], NEW]):
        launches(w, clock, parent, kid)
    assert w.placed(chain[1], coords) == "vllm", "its coordinator is the last agent a chain this long reaches"
    assert w.placed(chain[0], coords) is None, "one agent further: not followed"


def test_a_relaunched_worker_routes_as_its_lanes_worker_and_the_watch_resumes_it(tmp_path):
    """A relaunched worker is an agent the list has, so `watch_workers` drops it from the workers it reads; its runs
    resolved as nobody's. Its workers.d entry (launched by its coordinator) makes it its lane's worker: its run's end routes to
    the lane's handle as a worker's, and, being listed, the watch resumes it with the result."""
    new = "bc-77777777-7777-7777-7777-777777777777"
    (tmp_path / "workers.d").mkdir()
    (tmp_path / "workers.d" / f"vllm-{new}.json").write_text(json.dumps({
        "agent_id": new, "lane": "vllm", "launched_by": NEW, "role": "worker", "name": "vllm worker ZKS10 (relaunch of 1111)",
        "relaunch_of": W1_}))
    api, clock, w, _, roots = nodes_setup(tmp_path)
    api._agents.append({"id": new, "name": "vllm worker ZKS10 (relaunch of 1111)", "status": "IDLE", "latestRunId": "run-n",
                        "updatedAt": W.iso(T0 - 1800)})
    api._runs["run-n"] = dict(FINISHED)
    ended_t = run_end_for(tmp_path, roots, clock, w, new)
    while clock.t < ended_t + W.RESUME_S + 300:
        clock.t += 60
        w.sweep(lambda r: None)
    (rec,) = ended(tmp_path)
    assert (rec["handle"], rec["role"], rec["name"], rec.get("listed")) == (
        "vllm", "worker", "vllm worker ZKS10 (relaunch of 1111)", True)
    resumed = [p for p, _ in api.posts if p == f"/v1/agents/{new}/runs"]
    assert resumed and [r["kind"] for r in records(tmp_path)][-1] == "idle_after_run"


def test_a_coordinator_past_its_restarts_is_relaunched_from_its_handover_note_with_the_key(tmp_path, capsys):
    gen3 = f"bc-{W.uuid.uuid5(W.COORDINATORS, 'circuits/3')}"
    assert W.generation("circuits", gen3) == 3 and W.generation("circuits", CO) is None
    launcher, launched = tmp_path / "launch_coordinator.py", []
    launcher.write_text("# stand-in\n")
    gen4 = f"bc-{W.uuid.uuid5(W.COORDINATORS, 'circuits/4')}"

    def launch(argv, env):
        launched.append((argv, env))
        return 0, f"circuits: launched {gen4}, https://cursor.com/agents/{gen4}\n", ""
    api, clock, w = coordinators(tmp_path, [{"id": gen3, "name": "verity: circuits", "status": "IDLE", "latestRunId": "run-e1",
                                             "updatedAt": W.iso(T0)}], {"run-e1": {"status": "ERROR"}}, timers=None,
                                 notes=tmp_path / "notes", launcher=launcher, launch=launch)
    (tmp_path / "registry.json").write_text(json.dumps({"handles": [{"handle": "circuits", "holder": f"circuits ({gen3})"}]}))
    (tmp_path / "routes.d" / "circuits.json").write_text(json.dumps({"name": "circuits", "holder": gen3, "closed": None,
                                                                     "inbox_ts": "1760000000.000300"}))
    note(tmp_path / "notes", "circuits", "20261018T1900Z-report-coordinator-handover.md", f"circuits ({gen3})")
    note(tmp_path / "notes", "circuits", "20261018T1950Z-reply-from-x-handover-ack.md", "x")
    note(tmp_path / "notes", "circuits", "20261018T1955Z-report-checkpoint.md", f"circuits ({gen3})")
    for n in range(W.MAX_RESTARTS + 1):
        api._agents[0]["latestRunId"] = f"run-e{n + 1}"
        api._runs[f"run-e{n + 1}"] = {"status": "ERROR"}
        w.sweep(lambda r: None)
        clock.t += W.RESTART_S
    kinds = [r["kind"] for r in records(tmp_path)]
    assert kinds == ["crashed"] * 3 + ["erroring", "relaunch"] and launched == []
    assert records(tmp_path)[-1]["action"].startswith("not relaunched: no launch key in"), "gated off until the key exists"
    assert records(tmp_path)[-2]["owner"] == records(tmp_path)[-1]["owner"] == "top"
    (tmp_path / "launch.env").write_text(f"export CURSOR_API_KEY='{KEY}'\n")
    w.left.discard(gen3)
    clock.t += W.RELAUNCH_EVERY_S
    api._agents[0].update(latestRunId="run-e9", updatedAt=W.iso(clock.t))
    api._runs["run-e9"] = {"status": "ERROR"}
    w.sweep(lambda r: None)
    ((argv, env),) = launched
    assert argv[1:] == [str(launcher), "launch", "circuits", "--predecessor", gen3, "--pairing-thread", "1760000000.000300",
                        "--generation", "4", "--text-file", str(tmp_path / "relaunch-circuits.md")]
    assert env == {"CURSOR_API_KEY": KEY, "RESEARCH_SLACK_REGISTRY": str(tmp_path / "registry.json")}
    brief = (tmp_path / "relaunch-circuits.md").read_text()
    assert "20261018T1900Z-report-coordinator-handover.md" in brief and f"origin: circuits ({gen3})" in brief
    rec = records(tmp_path)[-1]
    assert (rec["kind"], rec["new_agent"], rec["action"]) == (
        "relaunch", gen4, f"relaunched it from its handover note as {gen4} (generation 4)")
    written = "".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file() and p.name != "launch.env")
    assert KEY not in written and KEY not in "".join(capsys.readouterr())


# workers of one coordinator that stopped mid-step, each its last entry a tool call, a tool result
# or thinking, with no assistant turn after, at seconds after the first (16:49:23Z), from their transcripts
STOPS = [("REC1", 0), ("G2-1", 143), ("REC2", 423), ("G2-7", 632), ("ZKS10", 655), ("M18b", 655), ("G2-4", 670), ("INT7", 757),
         ("KSF1", 767), ("FIX2595", 774), ("G2-3", 795), ("G2-2", 815), ("G2-6", 836), ("G2-8", 862)]
KIDS = {name: f"bc-{i + 1:08x}-0000-0000-0000-{i + 1:012x}" for i, (name, _) in enumerate(STOPS)}


def task_call(w, run, cid, agent, description, completed=False):
    """A Task call as a followed run streams it: the worker's id in `args.agentId` (a launch or a resume), the result truncated."""
    w.on_event(run, "tool_call", {"callId": cid, "name": "task", "status": "completed" if completed else "running",
                                  "args": {"agentId": agent, "description": description, "resume": agent},
                                  **({"truncated": {"result": True}} if completed else {})})


def test_proofs_workers_that_stop_together_are_one_alert_to_their_parents_lane_naming_each_and_its_wake_up(tmp_path):
    """Many of a coordinator's workers, known only from its Task calls, stopping over a few minutes are watched as its workers, and their IDLE reads, a sweep every 5 min, make one
    workers_failed to its handle naming all 14, with the wake-ups two said they set (the API's conversation has no tool calls,
    only what an agent says). A worker of another coordinator that stops meanwhile is its own record, to its own handle."""
    other = "bc-0000000f-0000-0000-0000-00000000000f"
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-co",
                                             "updatedAt": W.iso(T0)},
                                            {"id": NEW, "name": "verity: vllm", "status": "ACTIVE", "latestRunId": "run-v",
                                             "updatedAt": W.iso(T0)}],
                                 {"run-co": {"status": "RUNNING"}, "run-v": {"status": "RUNNING"}}, timers=None)
    w.sweep(lambda r: None)
    for i, (name, agent) in enumerate(KIDS.items()):
        task_call(w, "run-co", f"t{i}", agent, f"{name}: continue", completed=i % 2 == 1)
        api.workers[agent] = worker("ACTIVE", W.iso(T0), f"{name}: continue")
    task_call(w, "run-v", "tv", other, "V1: serve")
    api.workers[other] = worker("ACTIVE", W.iso(T0), "V1: serve")
    api.convs[KIDS["INT7"]] = [("user_message", "Coordinator (proofs), 17:33Z. You went idle at about 17:05Z."),
                               ("assistant_message", "Everything is pushed and synced. Setting my wake-up timer before ending "
                                                     "the turn.")]
    api.convs[KIDS["G2-1"]] = [("user_message", "Continue."),
                               ("assistant_message", "It's still one job behind. I'll record the run in the note."),
                               ("assistant_message", "The note is synced.\n\nA one-off wake-up fires in 15 minutes. Then I'll "
                                                     "check the run.")]
    clock.t += 60
    w.sweep(lambda r: None)
    assert {a: (w.workers[a]["parent"], w.workers[a]["handle"]) for a in [*KIDS.values(), other]} == {
        **{a: (CO, "circuits") for a in KIDS.values()}, other: (NEW, "vllm")}, "each worker its parent's, from the Task call"
    first = clock.t + 30
    stops = {KIDS[name]: first + s for name, s in STOPS}
    posted_at = None
    while clock.t < first + 40 * 60 and posted_at is None:
        clock.t += 5 * 60
        for agent, at in stops.items():
            if at <= clock.t:
                api.workers[agent] = worker("IDLE", W.iso(at), api.workers[agent]["name"])
        if clock.t >= first + 300:
            api.workers[other] = worker("IDLE", W.iso(first + 300), "V1: serve")
        w.sweep(lambda r: None)
        if any(r["kind"] == "workers_failed" for r in records(tmp_path)):
            posted_at = clock.t
    assert posted_at is not None and posted_at - first <= W.BURST_S + 5 * 60, "rung within a sweep of the burst's end"
    (burst,) = [r for r in records(tmp_path) if r["kind"] == "workers_failed"]
    assert {k: burst[k] for k in ("agent", "name", "handle", "owner", "lane", "count", "role")} == {
        "agent": CO, "name": "verity: circuits", "handle": "circuits", "owner": "circuits", "lane": "circuits", "count": 14,
        "role": "workers"}
    assert [m["name"] for m in burst["members"]] == [f"{name}: continue" for name, _ in STOPS]
    assert {m["kind"] for m in burst["members"]} == {"worker_idle"} and burst["minutes"] == 14
    wakes = {m["name"]: (m.get("wake"), m.get("wake_due")) for m in burst["members"] if m.get("wake")}
    assert wakes == {"INT7: continue": ("Setting my wake-up timer before ending the turn.", None),
                     "G2-1: continue": ("A one-off wake-up fires in 15 minutes.", W.iso(stops[KIDS["G2-1"]] + 15 * 60))}
    rest = [r for r in records(tmp_path) if r["kind"] != "workers_failed"]
    assert [(r["kind"], r["agent"], r["handle"]) for r in rest] == [("worker_idle", other, "vllm")], \
        "no worker_idle of the 14 on its own; the other coordinator's worker alone, as itself"
    assert set(rest[0]) == {"t", "kind", "agent", "name", "run", "at", "since", "lane", "handle"}, "posted as before it was held"
    assert all("idle_t" not in m for m in burst["members"])
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d")
    assert again.bursts == {} and set(KIDS.values()) <= set(again.children), "nothing left held; the children kept"


def test_a_burst_held_when_the_watch_restarts_is_still_posted_once(tmp_path):
    """The watch restarts on every deploy: a burst held in state.json is posted by the next one, once."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "ACTIVE", "latestRunId": "run-co",
                                             "updatedAt": W.iso(T0)}], {"run-co": {"status": "RUNNING"}}, timers=None)
    w.sweep(lambda r: None)
    pair = [KIDS["G2-2"], KIDS["G2-6"]]
    for i, agent in enumerate(pair):
        task_call(w, "run-co", f"t{i}", agent, "G2: continue")
        api.workers[agent] = worker("ACTIVE", W.iso(T0))
    clock.t += 60
    w.sweep(lambda r: None)
    for agent in pair:
        api.workers[agent] = worker("IDLE", W.iso(clock.t))
    clock.t += 60
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and len(json.loads((tmp_path / "state.json").read_text())["bursts"][CO]["items"]) == 2
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d")
    for _ in range(2):
        clock.t += W.BURST_QUIET_S
        again.sweep(lambda r: None)
    assert [(r["kind"], r["count"]) for r in records(tmp_path)] == [("workers_failed", 2)]


def test_a_worker_learned_from_a_former_coordinators_task_call_inherits_that_coordinators_handle(tmp_path):
    """A worker learned from a Task call of a coordinator's older generation, after a newer one took the inbox (routes.d),
    has a parent `holders` no longer names, and must not be recorded with lane '' and no handle. Its
    parent's handle is the registry holder's, or the handle `launch_coordinator.py` made the parent a generation of."""
    reg, routes = tmp_path / "registry.json", tmp_path / "routes.d"
    api, clock, w = coordinators(tmp_path, [{"id": OLD, "name": "verity: vllm", "status": "ACTIVE", "latestRunId": "run-old",
                                             "updatedAt": W.iso(T0)}], {"run-old": {"status": "RUNNING"}}, timers=None)
    coords = W.holders(reg, routes)
    gen3 = f"bc-{uuid.uuid5(W.COORDINATORS, 'circuits/3')}"
    assert OLD not in coords and W.lineage(reg, coords, OLD) == "vllm" and W.lineage(reg, coords, NEW) == "vllm"
    assert W.lineage(reg, coords, gen3) == "circuits" and W.lineage(reg, coords, KIDS["REC1"]) is None
    w.sweep(lambda r: None)
    g2 = KIDS["G2-1"]
    task_call(w, "run-old", "t1", g2, "G2-1 → LiveEnd for FinExe")
    api.workers[g2] = worker("ACTIVE", W.iso(T0), "G2-1 → LiveEnd for FinExe")
    w.workers[g2] = {"heard": 0.0, "parent": OLD, "lane": "", "handle": None}    # what the deployed watch had learned
    clock.t += 60
    w.sweep(lambda r: None)
    assert (w.workers[g2]["lane"], w.workers[g2]["handle"]) == ("vllm", "vllm")
    api.workers[g2] = worker("IDLE", W.iso(clock.t), "G2-1 → LiveEnd for FinExe")
    for _ in range(3):
        clock.t += W.BURST_QUIET_S
        w.sweep(lambda r: None)
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["agent"], rec["lane"], rec["handle"]) == ("worker_idle", g2, "vllm", "vllm")


def test_a_lane_named_after_a_handle_belongs_to_that_handle():
    """kernels' workers run lanes like `kernels-hashproof`; they belong to kernels, not infra."""
    names = {"kernels", "proofs", "infra", "kernels-x"}
    assert W.lane_handle("kernels", names) == "kernels"
    assert W.lane_handle("kernels-hashproof", names) == "kernels"
    assert W.lane_handle("kernels-x-y", names) == "kernels-x"
    assert W.lane_handle("kernelsy", names) is None
    assert W.lane_handle("", names) is None


def stream_worker(w, clock, run, minutes, cid="t1"):
    """A worker the parent resumed in the foreground (a `task` call left open) streams its progress each minute for `minutes`."""
    for _ in range(minutes):
        clock.t += 60
        w.on_event(run, "interaction_update", {"type": "tool-call-delta", "callId": cid, "taskUpdate": {}})
        w.sweep(lambda r: None)


def test_every_holder_a_registry_handle_names_counts_for_held_unread(tmp_path):
    coordinators(tmp_path, [], {})
    assert W.all_holders(tmp_path / "registry.json", tmp_path / "routes.d") == {CO: "circuits", LANE: "circuits", OLD: "vllm",
                                                                                 NEW: "vllm"}


def test_a_lane_holder_waiting_on_a_streaming_worker_for_an_hour_is_cancelled_and_continued(tmp_path):
    """A lane that posts as a handle (the registry's second holder) can sit for hours in a resumed worker's stream, a long
    call to the watch and not a coordinator's run. Owner top; the worker runs on."""
    run = {"status": "RUNNING", "createdAt": "2026-10-18T19:59:00Z", "updatedAt": "2026-10-18T19:59:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": LANE, "status": "ACTIVE", "latestRunId": "run-l"}], {"run-l": run})
    w.sweep(lambda r: None)
    w.on_event("run-l", "tool_call", {"callId": "t1", "name": "task", "status": "running", "args": {"agentId": "bc-w"}})
    since = clock.t
    stream_worker(w, clock, "run-l", W.UNREAD_S // 60 - 1)
    assert api.posts == [], "59 minutes: not yet"
    stream_worker(w, clock, "run-l", 1)
    assert [p for p, _ in api.posts] == ["/v1/agents/" + LANE + "/runs/run-l/cancel", "/v1/agents/" + LANE + "/runs"]
    assert "while a worker it waits on streamed" in api.posts[1][1]["prompt"]["text"]
    rec = next(r for r in records(tmp_path) if r["kind"] == "held_unread")
    assert (rec["owner"], rec["handle"], rec["since"], rec["action"]) == ("top", "circuits", W.iso(since), "cancelled and continued")
    assert rec["reason"] == "a lane of @circuits: nothing of its own for 60 min"


def test_an_inbox_holder_is_held_unread_only_once_a_doorbell_waited_an_hour(tmp_path):
    run = {"status": "RUNNING", "createdAt": "2026-10-18T19:59:00Z", "updatedAt": "2026-10-18T19:59:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-v"}], {"run-v": run},
                                 timers={})
    w.sweep(lambda r: None)
    w.on_event("run-v", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    stream_worker(w, clock, "run-v", W.UNREAD_S // 60 + 30)
    assert api.posts == [], "no doorbell and no timer: nothing sits unread"
    rang = clock.t
    bells = lambda: {"t": W.iso(clock.t), "inboxes": {"vllm": {"agent": NEW, "last": f"{rang:.6f}"}}}
    (tmp_path / "doorbells.json").write_text(json.dumps(bells()))
    for _ in range(W.UNREAD_S // 60 - 1):
        stream_worker(w, clock, "run-v", 1)
        (tmp_path / "doorbells.json").write_text(json.dumps(bells()))
    assert api.posts == [], "the doorbell has waited 59 min"
    stream_worker(w, clock, "run-v", 1)
    assert [p for p, _ in api.posts][0] == "/v1/agents/" + NEW + "/runs/run-v/cancel"
    rec = next(r for r in records(tmp_path) if r["kind"] == "held_unread")
    assert rec["reason"] == f"a doorbell rang at {W.iso(rang)}" and rec["owner"] == "top"


def test_a_holder_whose_worker_went_quiet_is_not_held_unread(tmp_path):
    """No worker progress for `WORKER_LIVE_S`: the stall rules judge it, not this one."""
    run = {"status": "RUNNING", "createdAt": "2026-10-18T19:59:00Z", "updatedAt": "2026-10-18T19:59:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": LANE, "status": "ACTIVE", "latestRunId": "run-l"}], {"run-l": run})
    w.sweep(lambda r: None)
    w.on_event("run-l", "tool_call", {"callId": "t1", "name": "task", "status": "running"})
    stream_worker(w, clock, "run-l", 40)
    for _ in range(25):
        clock.t += 60
        api._agents[0]["updatedAt"] = W.iso(clock.t)
        w.sweep(lambda r: None)
    assert api.posts == [] and "held_unread" not in [r["kind"] for r in records(tmp_path)]


def test_a_platform_outage_is_retried_uncapped_rings_a_person_and_clears_the_cap_once_coordinators_are_back(tmp_path):
    """Most coordinators ended in ERROR over a few hours while every restart failed. The watch spent each one's restarts within
    minutes and left them for a person, nobody was rung, and after a ring revived them they stayed 'left for a person' until
    state.json was cleared by hand. Replayed: one coordinator down every 20 minutes."""
    ids = [f"bc-{i:08x}-0000-0000-0000-{i:012x}" for i in range(1, 14)]
    reg, routes = tmp_path / "registry.json", tmp_path / "routes.d"
    routes.mkdir()
    reg.write_text(json.dumps({"handles": [{"handle": f"h{i}", "holder": f"the h{i} coordinator ({a})"} for i, a in enumerate(ids)]}))
    api = FakeApi([{"id": a, "name": f"verity: h{i}", "status": "ACTIVE", "latestRunId": f"ok-{i}"} for i, a in enumerate(ids)],
                  {f"ok-{i}": {"status": "FINISHED"} for i in range(len(ids))})
    clock = Clock(T0)
    w = W.Watch(api, tmp_path / "events.jsonl", clock, registry=reg, routes=routes)
    w.errors[ids[0]], w.left = W.MAX_RESTARTS, {ids[0]}          # its cap already spent
    refused = []

    def down(path, body=None):
        refused.append((path, clock.t))
        raise W.urllib.error.HTTPError(path, 503, "Service Unavailable", {}, None)
    api.post = down
    for minute in range(0, 4 * 60 + 1, 5):
        if minute % 20 == 0 and minute // 20 < len(ids):
            i = minute // 20
            api._agents[i]["latestRunId"] = f"err-{i}"
            api._runs[f"err-{i}"] = {"status": "ERROR"}
        clock.t = T0 + minute * 60
        w.sweep(lambda r: None)
    outages = [r for r in records(tmp_path) if r["kind"] == "outage"]
    assert [r["t"] for r in outages] == [W.iso(T0 + m * 60) for m in (110, 170, 230)], \
        "rung once 5 have been down 30 min (the 5th fell at 80 min), then hourly"
    assert outages[0]["count"] == 5 and outages[0]["owner"] == "top" and outages[0]["handles"] == [f"h{i}" for i in range(5)]
    tries = [t for path, t in refused if path == f"/v1/agents/{ids[0]}/runs"]
    outage_tries = [t for t in tries if t >= T0 + 40 * 60]
    assert len(outage_tries) >= 12 and all(b - a >= W.OUTAGE_RETRY_S for a, b in zip(outage_tries, outage_tries[1:])), \
        "in the outage the first coordinator is still tried every 15 min, its spent cap notwithstanding"
    assert not any(r["kind"] == "relaunch" for r in records(tmp_path) if r["agent"] == ids[0] and r["t"] >= W.iso(T0 + 2400))
    assert all(w.errors.get(a, 0) == 0 for a in ids[2:]), "a restart in an outage is not counted"
    assert not any(r["kind"] in ("erroring", "relaunch") for r in records(tmp_path) if r["agent"] in ids[2:])
    api.post = lambda path, body=None: {"run": {"id": "run-new"}}
    for i, a in enumerate(ids):
        api._agents[i]["latestRunId"] = f"back-{i}"
        api._runs[f"back-{i}"] = {"status": "RUNNING"}
    clock.t += 60
    w.sweep(lambda r: None)
    assert w.errors[ids[0]] and w.down, "seeing a restart running is not enough: one reads RUNNING a minute before it errors"
    clock.t += W.STICK_S
    w.sweep(lambda r: None)
    assert not w.left and not any(w.errors.values()) and not w.down, "back STICK_S: every cap cleared, no person needed"
    api._agents[3]["latestRunId"] = "err-again"
    api._runs["err-again"] = {"status": "ERROR"}
    clock.t += 60
    w.sweep(lambda r: None)
    assert w.errors[ids[3]] == 1 and records(tmp_path)[-1]["action"].startswith("restarted (1 of"), \
        "one coordinator down alone is restarted and counted as before"


def test_a_sweep_refused_with_429_waits_retry_after_or_doubles_to_a_cap():
    def refused(retry_after=None):
        return W.urllib.error.HTTPError("/v1/agents", 429, "Too Many Requests",
                                        {"Retry-After": retry_after} if retry_after else {}, None)
    assert W.rate_wait(refused("300"), 0) == 300
    assert W.rate_wait(refused(), 0) == W.RATE_WAIT_S[0]
    assert W.rate_wait(refused(), 240) == 480
    assert W.rate_wait(refused(), 800) == W.RATE_WAIT_S[1] == W.rate_wait(refused("99999"), 0)


def test_a_silent_worker_the_api_wont_archive_is_a_no_step_for_top_with_why(tmp_path):
    notes = tmp_path / "notes"
    note(notes, "circuits", "a.md", f"circuits worker W3 ({W1_}); code cursor/w3-branch-c7a9", age_s=2 * 3600)
    sh = FakeSh({"cursor/w3-branch-c7a9": "a" * 40})
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("ACTIVE", W.iso(T0 - 3 * 3600))}, notes=notes,
                                 git_remote=REMOTE, sh=sh)

    def refuse(path, body=None):
        if path.endswith("/archive"):
            raise W.urllib.error.HTTPError(path, 403, "Forbidden", {}, None)
        return {}
    api.post = refuse
    w.sweep(lambda r: None)
    sweeps(w, clock, 960)
    got = [r for r in records(tmp_path) if r["kind"] in ("no_step", "archived")]
    assert [{k: r.get(k) for k in ("kind", "owner", "role", "archive_failed")} for r in got] == [
        {"kind": "no_step", "owner": "top", "role": "worker", "archive_failed": "HTTP 403"}]


W7_, W8_ = (f"bc-{c * 8}-{c * 4}-{c * 4}-{c * 4}-{c * 12}" for c in "78")


def registered(tmp_path, clock, agent=W1_, lane="circuits", by=CO, **more):
    """A workers.d entry for `agent`, written at the clock's time: an entry's mtime is when the watch last heard of it."""
    d = tmp_path / "workers.d"
    d.mkdir(exist_ok=True)
    p = d / f"{lane}-{agent}.json"
    p.write_text(json.dumps({"agent_id": agent, "lane": lane, "launched_by": by, **more}))
    W.os.utime(p, (clock.t, clock.t))
    return p


def integrator(tmp_path, v0="ERROR", name="CIR-7 INTEGRATOR", status="IDLE", agents=(), runs=None, **kw):
    """A swarm's integrator (W1_), a Task subagent the circuits coordinator launched and registered, named after its issue:
    no run the API lists, IDLE on v1 since T0 however its turn ended, and `v0` what the legacy status says of it. The
    watch, before its first sweep."""
    api, clock, w = coordinators(tmp_path, list(agents), runs or {}, workers={W1_: worker(status, W.iso(T0), name=name)}, **kw)
    linear_teams(tmp_path)
    registered(tmp_path, clock)
    api.v0[W1_] = v0
    return api, clock, w


def stops(api, agent=W1_, since=0):
    return api.gets[since:].count(f"/v0/agents/{agent}")


RESUME_IT = "can't start a run on a worker: resume it with Task, which only the agent that launched it can, or relaunch it"


def test_a_worker_stopped_with_an_error_is_recorded_once_for_its_lane_naming_it_and_its_issue(tmp_path):
    """A swarm's integrator the platform cuts mid-turn (a billing stop) reads IDLE on v1 as after any turn, and has no
    run to read ERROR. Its v0 status says ERROR: one record for its lane, whose coordinator alone can resume it, and
    no second however long it sits so."""
    api, clock, w = integrator(tmp_path)
    sweeps(w, clock, 3 * 60, step=1)
    (rec,) = records(tmp_path)
    assert {k: v for k, v in rec.items() if k not in ("t", "since")} == {
        "kind": "crashed", "agent": W1_, "name": "CIR-7 INTEGRATOR", "run": "", "role": "worker", "status": "ERROR",
        "lane": "circuits", "handle": "circuits", "issue": "CIR-7", "action": RESUME_IT}
    assert W.epoch(rec["since"]) == T0 + W.DROPPED_S, "once its last change is `DROPPED_S` old: a turn would have begun by then"
    assert status(tmp_path)[W1_]["watch_state"] == "crashed"
    assert stops(api) == 2 and api.posts == [], "read at its first read and once past `DROPPED_S`; nothing is started on it"
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d")
    sweeps(again, clock, 60, step=1)
    assert len(records(tmp_path)) == 1 and stops(api) == 2, "nor after the watch restarts, as it does on every deploy"


def test_a_worker_in_error_that_runs_again_and_stops_again_rings_again(tmp_path):
    """Once per stop. Its launcher resumes it with Task: the message moves its `updatedAt`, v1 still reads IDLE, and v0 has
    it in a turn, read every `STOP_S` until that ends. A turn that ends well clears the error and rings nobody; one that
    ends in ERROR is a new stop."""
    api, clock, w = integrator(tmp_path)
    sweeps(w, clock, 20, step=1)
    for end, kinds in (("ERROR", ["crashed"] * 2), ("FINISHED", ["crashed"] * 2), ("ERROR", ["crashed"] * 3)):
        api.workers[W1_]["updatedAt"], api.v0[W1_], before = W.iso(clock.t), "RUNNING", len(api.gets)
        sweeps(w, clock, 20, step=1)
        assert stops(api, since=before) == 4, "at the change, then every `STOP_S` of its turn: not every sweep"
        assert status(tmp_path)[W1_]["watch_state"] == "ok", "in a turn again: no longer shown in error"
        api.v0[W1_], before = end, len(api.gets)
        sweeps(w, clock, 20, step=1)
        assert [r["kind"] for r in records(tmp_path)] == kinds, end
        assert stops(api, since=before) == 1, "its end is read once"
    assert [r["agent"] for r in records(tmp_path)] == [W1_] * 3


def test_a_worker_seen_to_stop_in_error_is_one_crashed_record_and_no_worker_idle(tmp_path):
    """One record of one stop. A worker v1 shows ACTIVE and then IDLE went idle, which rings its lane as `worker_idle`
    when it finished; one v0 has in ERROR at that read is `crashed` at once, in its place."""
    for d, end, kinds in ((tmp_path / "error", "ERROR", ["crashed"]), (tmp_path / "done", "FINISHED", ["worker_idle"])):
        d.mkdir()
        api, clock, w = integrator(d, v0="RUNNING", status="ACTIVE")
        sweeps(w, clock, 10, step=1)
        assert stops(api) == 0, "v1 has it in a turn: nothing to ask"
        api.workers[W1_].update(status="IDLE", updatedAt=W.iso(clock.t))
        api.v0[W1_] = end
        sweeps(w, clock, 30, step=1)
        assert [r["kind"] for r in records(d)] == kinds
        assert W.epoch(records(d)[0]["since"]) <= clock.t - 29 * 60, "from the first read after it stopped"
        api.workers[W1_].update(status="ACTIVE", updatedAt=W.iso(clock.t))
        api.v0[W1_] = "RUNNING"
        sweeps(w, clock, 10, step=1)
        api.workers[W1_].update(status="IDLE", updatedAt=W.iso(clock.t))
        api.v0[W1_] = "ERROR"
        sweeps(w, clock, 10, step=1)
        assert [r["kind"] for r in records(d)] == [*kinds, "crashed"], "it ran again and stopped again, in error"


def test_workers_of_one_coordinator_stopped_in_error_together_are_one_record_naming_each(tmp_path):
    """A billing stop cuts every worker of a swarm at once: their coordinator is rung once, as for any workers that stop
    together, each member saying it is in error and its issue. No conversation is read for a wake-up."""
    api, clock, w = integrator(tmp_path)
    api.workers[W7_] = worker("IDLE", W.iso(T0), name="CIR-8 SEAT-A")
    api.v0[W7_] = "ERROR"
    registered(tmp_path, clock, W7_)
    sweeps(w, clock, 60, step=1)
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["agent"], rec["handle"], rec["owner"], rec["count"]) == ("workers_failed", CO, "circuits", "circuits", 2)
    assert [(m["kind"], m["agent"], m["status"], m["issue"]) for m in rec["members"]] == [
        ("crashed", W1_, "ERROR", "CIR-7"), ("crashed", W7_, "ERROR", "CIR-8")]
    assert not [g for g in api.gets if g.endswith("/conversation")]


def test_a_v0_status_the_api_wont_give_records_nothing_and_is_read_again(tmp_path, capsys):
    api, clock, w = integrator(tmp_path, v0=W.urllib.error.HTTPError("/v0/agents/x", 500, "Server Error", {}, None))
    sweeps(w, clock, 30, step=1)
    assert records(tmp_path) == [] and stops(api) == 30 * 60 // W.STOP_S, "every `STOP_S`, not every sweep"
    assert capsys.readouterr().err.count("v0 status went unread") == 1, "said once an hour"
    api.v0[W1_] = "ERROR"
    sweeps(w, clock, 10, step=1)
    assert [(r["kind"], r["agent"]) for r in records(tmp_path)] == [("crashed", W1_)]


def test_a_worker_nothing_places_is_recorded_in_error_quietly(tmp_path):
    """Only a worker the watch places rings a lane: one it can't place rang infra once as unplaced, and its error, which
    nobody the watch knows could act on, is posted and rings no one."""
    api, clock, w = coordinators(tmp_path, [], {}, workers={W1_: worker("IDLE", W.iso(T0), name="INTEGRATOR")})
    api.v0[W1_] = "ERROR"
    w.workers[W1_] = {"heard": T0, "lane": "", "handle": None}
    sweeps(w, clock, 20, step=1)
    assert [(r["kind"], r.get("quiet"), r.get("handle"), r.get("issue")) for r in records(tmp_path)] == [
        ("unplaced", None, None, None), ("crashed", True, None, None)]


def test_a_listed_agent_with_no_run_is_read_as_a_worker_by_id(tmp_path):
    """The sweep judges a listed agent by its latest run and leaves one without a run alone, and the worker read skipped
    every agent the list has: an agent on the list with no run was read by neither."""
    listed = {"id": W1_, "name": "CIR-7 INTEGRATOR", "status": "IDLE", "updatedAt": W.iso(T0)}
    api, clock, w = integrator(tmp_path, agents=[listed])
    sweeps(w, clock, 20, step=1)
    assert [(r["kind"], r["agent"], r["handle"], r["issue"]) for r in records(tmp_path)] == [("crashed", W1_, "circuits", "CIR-7")]
    assert status(tmp_path)[W1_]["run_id"] == "", "one row, a worker's"


def test_a_listed_worker_in_error_is_restarted_and_each_record_names_its_lead_and_its_issue(tmp_path):
    """An agent on the list has runs the watch can start: one whose latest run ended in ERROR, or whose own status the
    list gives as ERROR, is restarted, and its coordinator rung with each try and when they are spent."""
    by_run = {"id": W1_, "name": "CIR-7 INTEGRATOR", "status": "IDLE", "latestRunId": "run-e1", "updatedAt": W.iso(T0)}
    by_status = {"id": W7_, "name": "CIR-8 SEAT-A", "status": "ERROR", "latestRunId": "run-f", "updatedAt": W.iso(T0)}
    api, clock, w = integrator(tmp_path, agents=[by_run, by_status], runs={"run-e1": {"status": "ERROR"}, "run-f": dict(FINISHED)})
    registered(tmp_path, clock, W7_)
    w.sweep(lambda r: None)
    w.sweep(lambda r: None)
    assert [(r["kind"], r["agent"], r["owner"], r["issue"], r["action"]) for r in records(tmp_path)] == [
        ("crashed", W1_, "circuits", "CIR-7", "restarted (1 of 3)"), ("crashed", W7_, "circuits", "CIR-8", "restarted (1 of 3)")]
    assert [p for p, _ in api.posts] == [f"/v1/agents/{W1_}/runs", f"/v1/agents/{W7_}/runs"]
    assert status(tmp_path)[W7_]["watch_state"] == "crashed" and f"/v0/agents/{W1_}" not in api.gets
    for n in (2, 3, 4):
        clock.t += W.RESTART_S
        by_run["latestRunId"] = f"run-e{n}"
        api._runs[f"run-e{n}"] = {"status": "ERROR"}
        w.sweep(lambda r: None)
    mine = [r for r in records(tmp_path) if r["agent"] == W1_]
    assert [(r["kind"], r["owner"], r["issue"]) for r in mine] == [("crashed", "circuits", "CIR-7")] * 3 + [
        ("erroring", "circuits", "CIR-7")]
    assert not drops(tmp_path)


def test_a_restart_the_api_refuses_is_tried_again_and_ends_as_erroring_for_top(tmp_path):
    """A coordinator in error whose restart the API refused got one `crashed` line and was never tried again: its run
    stayed judged, so it was neither restarted, nor counted erroring, nor relaunched. Each refused try now counts toward
    the row and the next is made `RESTART_S` later; when they are spent it is erroring, which rings top, and relaunched."""
    api, clock, w = coordinators(tmp_path, [{"id": CO, "name": "verity: circuits", "status": "IDLE", "latestRunId": "run-e",
                                             "updatedAt": W.iso(T0)}], {"run-e": {"status": "ERROR"}})
    tried = []

    def refuse(path, body=None):
        tried.append((path, clock.t))
        raise W.urllib.error.HTTPError(path, 402, "Payment Required", {}, None)
    api.post = refuse
    sweeps(w, clock, 60, step=1)
    assert tried == [(f"/v1/agents/{CO}/runs", T0 + 60 + n * W.RESTART_S) for n in range(W.MAX_RESTARTS)]
    got = [(r["kind"], r["owner"], r["action"]) for r in records(tmp_path)]
    assert got[:4] == [
        ("crashed", "top", "got no restart from the API (HTTP 402): try 1 of 3, the next in 10 min"),
        ("crashed", "top", "got no restart from the API (HTTP 402): try 2 of 3, the next in 10 min"),
        ("crashed", "top", "got no restart from the API (HTTP 402): try 3 of 3, the last"),
        ("erroring", "top", "relaunching it")]
    assert [g[0] for g in got[4:]] == ["relaunch"] and got[4][2].startswith("not relaunched: no launch key")
    assert status(tmp_path)[CO]["watch_state"] in ("erroring", "relaunch") and status(tmp_path)[CO]["restart_count"] == 3


class FakeLinear:
    """Linear beside the fake API, as `W.Linear.issue` gives an issue: `issues[key]` holds its `comments` (`said` adds
    one, which a read gives from its time on by `clock`), `state` and `labels`; an issue it doesn't hold is one Linear
    answers it can't find. `fail` is the error every read raises while it is set. `calls` lists each read, `(key, the
    cursor it paged from)`."""

    def __init__(self, clock=None, **issues):
        self.issues, self.calls, self.fail = {k.replace("_", "-"): {"comments": [], **v} for k, v in issues.items()}, [], None
        self.clock = clock

    def said(self, key, t, by):
        """A comment on `key` at `t`, signed `by`; its id."""
        n = len(self.issues[key]["comments"]) + 1
        self.issues[key]["comments"].append({"id": f"{key}-c{n}", "t": float(t), "by": by})
        return f"{key}-c{n}"

    def issue(self, key, after=None):
        self.calls.append((key, after))
        if self.fail is not None:
            raise self.fail
        if key not in self.issues:
            raise ValueError("Linear: Entity not found: Issue - Could not find referenced Issue.")
        d, at = self.issues[key], int(after or 0)
        newest = sorted((c for c in d["comments"] if not self.clock or c["t"] <= self.clock.t), key=lambda c: -c["t"])
        return {"url": f"https://linear.example/issue/{key}", "state": d.get("state", "started"),
                "labels": list(d.get("labels", ())), "comments": [dict(c) for c in newest[at:at + W.HANDIN_PAGE]],
                "after": str(at + W.HANDIN_PAGE) if at + W.HANDIN_PAGE < len(newest) else None}


def swarm(tmp_path, name="CIR-7 INTEGRATOR", linear=..., **entry):
    """A swarm of the circuits lane: its integrator (W1_, as `integrator` makes it, its last turn ended well) registered
    by its lead as taking hand-ins on the issue it is named after, and Linear holding that issue with no comment yet."""
    lin = FakeLinear(CIR_7={})
    api, clock, w = integrator(tmp_path, v0="FINISHED", name=name, linear=lin if linear is ... else linear)
    lin.clock = clock
    registered(tmp_path, clock).unlink()
    registered(tmp_path, clock, integrates="CIR-7", **entry)
    return api, clock, w, lin


def handins(tmp_path):
    return [r for r in records(tmp_path) if r["kind"] == "handin_waiting"]


def read_at(w, clock, lin, minutes):
    """A sweep a minute for `minutes`: the minutes after T0 at which Linear was read, one entry a call."""
    out = []
    for _ in range(minutes):
        clock.t += 60
        n = len(lin.calls)
        w.sweep(lambda r: None)
        out += [int(clock.t - T0) // 60] * (len(lin.calls) - n)
    return out


def test_a_hand_in_waiting_two_hours_on_an_integrator_is_recorded_once_for_its_lead(tmp_path):
    """A swarm's integrator that stopped answers no hand-in, and nothing else tells its lead when its stop went unseen.
    A comment on the issue the integrator takes hand-ins on, with none of the integrator's after it, is one record for
    its lead once it is two hours old: not before, not again while it waits, not for the comments that wait behind it,
    and not after the watch restarts."""
    api, clock, w, lin = swarm(tmp_path)
    first = lin.said("CIR-7", T0 + 10 * 60, "circuits/SEAT-A")
    assert read_at(w, clock, lin, 129) == [1, 121] and handins(tmp_path) == [], "nothing until it is two hours old"
    lin.said("CIR-7", T0 + 100 * 60, "circuits/SEAT-B")
    assert read_at(w, clock, lin, 1) == [130]
    (rec,) = records(tmp_path)
    assert rec == {
        "t": W.iso(T0 + 130 * 60), "kind": "handin_waiting", "agent": W1_, "name": "CIR-7 INTEGRATOR", "run": "",
        "role": "worker", "issue": "CIR-7", "url": "https://linear.example/issue/CIR-7", "comment": first,
        "by": "circuits/SEAT-A", "since": W.iso(T0 + 10 * 60), "minutes": 120, "waiting": 2, "signs": "INTEGRATOR",
        "lane": "circuits", "handle": "circuits"}
    assert status(tmp_path)[W1_]["watch_state"] != "handin_waiting", "the integrator's row is judged by its own turns"
    lin.said("CIR-7", clock.t, "circuits/SEAT-C")
    assert read_at(w, clock, lin, 9 * 60) == [250, 370, 490, 610] and len(handins(tmp_path)) == 1
    again = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                    linear=lin)
    assert read_at(again, clock, lin, 5 * 60) == [671, 791, 911] and len(handins(tmp_path)) == 1


@pytest.mark.parametrize("answer, rings", [
    ("circuits/INTEGRATOR", False),
    ("vllm/integrator", False),         # signed under another lane's handle, in another case: still its name
    ("circuits", True),                 # its lead's own comment is no answer of the integrator's
    ("circuits/SEAT-B", True),
    ("INTEGRATOR", True),               # a handle's or a person's comment, never a worker's: a worker signs LANE/NAME
    ("", True)])
def test_a_hand_in_the_integrator_answered_rings_nobody_and_one_only_another_answered_does(tmp_path, answer, rings):
    """The team's agents all post through one app, so a comment's author doesn't say whose it is: its signature does.
    Only a comment signed as a worker of the integrator's name (`--worker NAME`: LANE/NAME) answers a hand-in."""
    api, clock, w, lin = swarm(tmp_path)
    first = lin.said("CIR-7", T0 + 10 * 60, "circuits/SEAT-A")
    lin.said("CIR-7", T0 + 30 * 60, answer)
    read_at(w, clock, lin, 5 * 60)
    assert [(r["comment"], r["by"], r["waiting"]) for r in handins(tmp_path)] == ([(first, "circuits/SEAT-A", 2)] if rings else [])


def test_each_hand_in_left_waiting_after_the_integrators_last_answer_rings_once(tmp_path):
    """Once per comment: an integrator that answered and then stopped leaves the next hand-in waiting, which rings when
    it is two hours old, and one it answers late stops counting."""
    api, clock, w, lin = swarm(tmp_path)
    first = lin.said("CIR-7", T0, "circuits/SEAT-A")
    read_at(w, clock, lin, 150)
    lin.said("CIR-7", clock.t, "circuits/INTEGRATOR")
    second = lin.said("CIR-7", clock.t + 20 * 60, "circuits/SEAT-B")
    read_at(w, clock, lin, 6 * 60)
    got = handins(tmp_path)
    assert [(r["comment"], r["by"], r["minutes"], r["waiting"]) for r in got] == [
        (first, "circuits/SEAT-A", 120, 1), (second, "circuits/SEAT-B", 120, 1)]
    assert [int(W.epoch(r["t"]) - T0) // 60 for r in got] == [120, 290], "each within a sweep of its two hours"


def test_a_hand_in_that_waits_for_days_is_still_recorded_once(tmp_path):
    """The watch forgets what it hasn't met for two days, whenever it saves its state. A hand-in it keeps meeting is not
    forgotten: however long it waits, it is one record."""
    api, clock, w, lin = swarm(tmp_path)
    lin.said("CIR-7", T0, "circuits/SEAT-A")
    for _ in range(40):
        clock.t += W.HANDIN_S
        w.sweep(lambda r: None)
        w.save()
    assert clock.t - T0 > 3 * 86400 and len(lin.calls) == 40 and len(handins(tmp_path)) == 1


def test_an_integrator_that_stopped_days_ago_still_has_a_hand_in_on_its_issue_recorded(tmp_path):
    """A worker idle for two days is forgotten, and its name with it. An integrator's name says which comments are its
    own, so it is kept while its entry stands: a hand-in days after its last turn is the one its lead most needs."""
    api, clock, w, lin = swarm(tmp_path)
    for _ in range(40):
        clock.t += W.HANDIN_S
        w.sweep(lambda r: None)
    assert W1_ in w.workers and handins(tmp_path) == []
    late = lin.said("CIR-7", clock.t, "circuits/SEAT-A")
    clock.t += W.HANDIN_S
    w.sweep(lambda r: None)
    assert [(r["comment"], r["name"], r["signs"]) for r in handins(tmp_path)] == [(late, "CIR-7 INTEGRATOR", "INTEGRATOR")]


@pytest.mark.parametrize("state", ["completed", "canceled"])
def test_a_closed_issue_takes_no_hand_in(tmp_path, state):
    """An issue that is done or cancelled has no integrator to answer on it: a comment left on it rings nobody."""
    api, clock, w, lin = swarm(tmp_path)
    lin.issues["CIR-7"]["state"] = state
    lin.said("CIR-7", T0, "circuits/SEAT-A")
    assert read_at(w, clock, lin, 5 * 60) == [1, 121, 241] and handins(tmp_path) == []
    lin.issues["CIR-7"]["state"] = "started"
    assert read_at(w, clock, lin, 2 * 60) == [361] and len(handins(tmp_path)) == 1, "reopened, it takes them again"


def test_a_linear_read_that_fails_records_nothing_and_is_made_again_at_the_next_sweep_once_per_issue(tmp_path, capsys):
    """A failed read is no finding: nothing is recorded from it, each issue due is read again at the next sweep and at
    most once a sweep, and the hand-in that waited through it still rings, once. An answer that Linear can't (no such
    issue) is asked again far less often."""
    api, clock, w, lin = swarm(tmp_path)
    registered(tmp_path, clock, W7_, integrates="CIR-8")
    api.workers[W7_] = worker("IDLE", W.iso(T0), name="CIR-8 MERGER")
    lin.issues["CIR-8"] = {"comments": []}
    lin.said("CIR-7", T0 - 3 * 3600, "circuits/SEAT-A")
    lin.said("CIR-8", T0 - 3 * 3600, "circuits/SEAT-B")
    lin.fail = W.urllib.error.URLError("timed out")
    assert read_at(w, clock, lin, 5) == [1, 2, 3, 4, 5] and [k for k, _ in lin.calls] == ["CIR-7"] * 5, \
        "one read a sweep while Linear is down: the second issue is not tried after the first fails"
    assert records(tmp_path) == []
    assert capsys.readouterr().err.count("Linear unread (URLError") == 1, "said once, not at every sweep"
    lin.fail = W.urllib.error.HTTPError(W.LINEAR_API, 400, "Bad Request", {}, None)
    assert read_at(w, clock, lin, 1) == [6] and records(tmp_path) == []
    lin.fail = None
    assert read_at(w, clock, lin, 30) == [7, 7]
    assert [(r["issue"], r["agent"], r["minutes"]) for r in handins(tmp_path)] == [("CIR-7", W1_, 187), ("CIR-8", W7_, 187)]
    del lin.issues["CIR-8"]
    lin.calls.clear()
    assert read_at(w, clock, lin, 3 * 60) == [127, 127, 137, 147, 157, 167, 177, 187, 197, 207], \
        "an issue Linear won't give is read every `HANDIN_RETRY_S`, and the other goes on being read"
    assert len(handins(tmp_path)) == 2 and "issue CIR-8 unread: Linear: Entity not found" in capsys.readouterr().err


def test_with_no_linear_credential_no_hand_in_is_recorded_and_the_log_says_so_once(tmp_path, capsys):
    """Without the Linear app's two names in the file the watch is handed, or with no file, there is no hand-in ring:
    nothing is read or recorded, and the log says so once, however many sweeps and issues."""
    env = tmp_path / "relay.env"
    env.write_text("GH_TOKEN=dummy-not-a-secret\nLINEAR_CLIENT_ID=\n")
    asked = []
    lin = W.Linear(env, tmp_path / "linear-token.json", http=lambda *a: asked.append(a))
    api, clock, w, _ = swarm(tmp_path, linear=lin)
    registered(tmp_path, clock, W7_, integrates="CIR-8")
    api.workers[W7_] = worker("IDLE", W.iso(T0), name="CIR-8 MERGER")

    def three_hours(watch):
        for _ in range(3 * 60):
            clock.t += 60
            watch.sweep(lambda r: None)
        return capsys.readouterr().err

    err = three_hours(w)
    assert records(tmp_path) == [] and asked == [] and not (tmp_path / "linear-token.json").exists()
    assert err.count("no Linear credential") == 1 and f"no LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET in {env}" in err
    bare = W.Watch(api, tmp_path / "bare" / "events.jsonl", clock, registry=tmp_path / "registry.json",
                   routes=tmp_path / "routes.d")
    (tmp_path / "bare" / "workers.d").mkdir(parents=True)
    (tmp_path / "bare" / "workers.d" / "i.json").write_text((tmp_path / "workers.d" / f"circuits-{W1_}.json").read_text())
    err = three_hours(bare)
    assert not (tmp_path / "bare" / "events.jsonl").exists()
    assert err.count("no Linear credential") == 1 and "started without --gh-env" in err


def test_an_integrators_issue_is_read_when_a_hand_in_may_be_due_and_never_at_every_sweep(tmp_path):
    """The calls are counted: an issue with nothing waiting is read once in two hours, one with a hand-in waiting once
    more when that turns two hours old, and a sweep reads an issue at most once. Two integrators of one issue (one
    relaunched, its old entry still there) are one read, and the record names the one registered last."""
    api, clock, w, lin = swarm(tmp_path)
    assert read_at(w, clock, lin, 24 * 60) == [1 + 120 * n for n in range(12)] and len(lin.calls) == 12
    clock.t += 60
    api.workers[W7_] = worker("IDLE", W.iso(clock.t), name="CIR-7 INTEGRATOR-2")
    registered(tmp_path, clock, W7_, integrates="CIR-7")
    lin.said("CIR-7", clock.t, "circuits/SEAT-A")
    lin.said("CIR-7", clock.t + 30 * 60, "circuits/INTEGRATOR")
    late = lin.said("CIR-7", clock.t + 40 * 60, "circuits/SEAT-A")
    lin.calls.clear()
    at = read_at(w, clock, lin, 4 * 60)
    assert at == [1442, 1441 + 120, 1481 + 120] and lin.calls == [("CIR-7", None)] * 3
    (rec,) = handins(tmp_path)
    assert (rec["agent"], rec["name"], rec["comment"], rec["signs"]) == (W7_, "CIR-7 INTEGRATOR-2", late, "INTEGRATOR-2")
    lin.issues["CIR-7"]["state"] = "completed"
    lin.said("CIR-7", clock.t, "circuits/SEAT-B")
    read_at(w, clock, lin, 6 * 60)
    assert len(handins(tmp_path)) == 1, "a closed issue takes no hand-in"


def test_the_integrators_last_answer_is_looked_for_pages_back_and_no_further(tmp_path):
    """An issue's comments come newest first, a page a call: the watch pages back only until it meets the integrator's
    last, and stops after `HANDIN_PAGES`, saying that older comments may wait too, once."""
    api, clock, w, lin = swarm(tmp_path)
    lin.said("CIR-7", T0 - 9 * 3600, "circuits/INTEGRATOR")
    for n in range(W.HANDIN_PAGE + 20):
        lin.said("CIR-7", T0 - 8 * 3600 + n, f"circuits/SEAT-{n % 5}")
    read_at(w, clock, lin, 1)
    assert lin.calls == [("CIR-7", None), ("CIR-7", str(W.HANDIN_PAGE))]
    (rec,) = handins(tmp_path)
    assert (rec["comment"], rec["waiting"], rec["minutes"], rec.get("older")) == ("CIR-7-c2", W.HANDIN_PAGE + 20, 481, None)
    lin.issues["CIR-7"]["comments"] = []
    for n in range(W.HANDIN_PAGE * W.HANDIN_PAGES + 50):
        lin.said("CIR-7", T0 - 8 * 3600 + n, "circuits/SEAT-A")
    lin.calls.clear()
    read_at(w, clock, lin, 5 * 60)
    assert lin.calls == [("CIR-7", None), ("CIR-7", "100"), ("CIR-7", "200")] * 2, "two reads in five hours, three pages each"
    assert [(r["waiting"], r.get("older")) for r in handins(tmp_path)[1:]] == [(W.HANDIN_PAGE * W.HANDIN_PAGES, True)]


def test_an_integrators_signature_is_its_entrys_else_the_word_after_its_names_issue_key(tmp_path, capsys):
    """Which comments are the integrator's is its entry's `signs`, else the NAME of `TEAM-7 NAME`; an integrator with
    neither is said in the log and no hand-in on its issue is recorded, since every answer of its would read as none."""
    assert [W.signer(n) for n in ("CIR-7 INTEGRATOR", "CIR-7: merge-1 takes the seats' rows", "INTEGRATOR (CIR-7)", "")] == [
        "INTEGRATOR", "merge-1", "", ""]
    api, clock, w, lin = swarm(tmp_path, name="the swarm's integrator", signs="MERGE")
    lin.said("CIR-7", T0, "circuits/SEAT-A")
    lin.said("CIR-7", T0 + 60, "circuits/merge")
    read_at(w, clock, lin, 3 * 60)
    assert handins(tmp_path) == []
    lin.said("CIR-7", clock.t, "circuits/SEAT-A")
    read_at(w, clock, lin, 3 * 60)
    assert [(r["signs"], r["name"]) for r in handins(tmp_path)] == [("MERGE", "the swarm's integrator")]
    registered(tmp_path, clock, integrates="CIR-7")
    lin.calls.clear()
    lin.said("CIR-7", clock.t, "circuits/MERGE")
    lin.said("CIR-7", clock.t + 60, "circuits/SEAT-A")
    capsys.readouterr()
    read_at(w, clock, lin, 3 * 60)
    assert len(handins(tmp_path)) == 1 and lin.calls == []
    assert capsys.readouterr().err.count("nothing says which comments are its own") == 3, "said, at most once an hour"


def test_a_hand_in_on_an_issue_whose_integrator_no_handle_owns_names_no_handle(tmp_path):
    """The lead rung is the one `placed` gives the integrator: its launcher's handle. An entry whose launcher and lane
    no handle owns gives none, and the record says so by carrying none, for the relay to ring infra."""
    api, clock, w, lin = swarm(tmp_path, lane="swarm-x", by=W8_)
    lin.said("CIR-7", T0 - 3 * 3600, "circuits/SEAT-A")
    read_at(w, clock, lin, 5)
    (rec,) = handins(tmp_path)
    assert rec["lane"] == "swarm-x" and "handle" not in rec and "quiet" not in rec


ISSUE = {"url": "https://linear.example/issue/CIR-7", "state": {"type": "started"},
         "labels": {"nodes": [{"name": "swarm"}, {"name": "lane:circuits"}]},
         "comments": {"pageInfo": {"hasNextPage": True, "endCursor": "cursor-2"}, "nodes": [
             {"id": "c-3", "createdAt": "2026-10-18T19:40:00.512Z", "botActor": None, "user": {"name": "A Person"}},
             {"id": "c-2", "createdAt": "2026-10-18T19:30:00.000Z", "botActor": {"userDisplayName": "circuits/INTEGRATOR"},
              "user": None},
             {"id": "c-1", "createdAt": "2026-10-18T19:20:00.000Z", "botActor": {"userDisplayName": "circuits"}, "user": None}]}}


class FakeHttp:
    """Linear's two endpoints as `W.Linear` calls them: `posts` lists each call (url, body, headers), `n` counts the
    tokens granted, `refuse` holds the bearer tokens the API answers 401 for, and `answer` is what it says of an issue
    (ISSUE, as its GraphQL gives one: the app's comments signed in `botActor`, a person's in `user`)."""

    def __init__(self):
        self.posts, self.n, self.refuse, self.answer = [], 0, set(), {"data": {"issue": ISSUE}}

    def __call__(self, url, data, headers):
        self.posts.append((url, data, headers))
        if url == W.LINEAR_TOKEN_URL:
            self.n += 1
            return {"access_token": f"dummy-token-{self.n}", "expires_in": 30 * 86400}
        if headers["Authorization"].removeprefix("Bearer ") in self.refuse:
            raise W.urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return self.answer


def test_the_linear_reader_gets_its_token_as_the_teams_client_does_and_keeps_it(tmp_path):
    """The watch can't import the team's Linear client, so it does what that client does: the app's client-credentials
    grant, always for the one scope set (a grant for another revokes the app's live tokens), a token kept 0600 and
    reused until an hour before it expires, fetched again once when Linear refuses it; one query an issue, with no
    comment's text asked for."""
    env, cache, clock = tmp_path / "relay.env", tmp_path / "state" / "linear-token.json", Clock(T0)
    env.write_text("SLACK_BOT_TOKEN=dummy-slack\nexport LINEAR_CLIENT_ID='dummy-client'\nLINEAR_CLIENT_SECRET=\"dummy-secret\"\n")
    http = FakeHttp()
    lin = W.Linear(env, cache, clock, http)
    assert lin.issue("CIR-7") == {
        "url": "https://linear.example/issue/CIR-7", "state": "started", "labels": ["swarm", "lane:circuits"], "after": "cursor-2",
        "comments": [{"id": "c-3", "t": T0 - 20 * 60, "by": "A Person"}, {"id": "c-2", "t": T0 - 30 * 60, "by": "circuits/INTEGRATOR"},
                     {"id": "c-1", "t": T0 - 40 * 60, "by": "circuits"}]}
    (url, form, head), (api_url, body, bearer) = http.posts
    assert url == "https://api.linear.app/oauth/token" and api_url == "https://api.linear.app/graphql"
    assert W.urllib.parse.parse_qs(form.decode()) == {"grant_type": ["client_credentials"], "scope": ["read,write"]}
    assert head == {"Authorization": "Basic " + W.base64.b64encode(b"dummy-client:dummy-secret").decode(),
                    "Content-Type": "application/x-www-form-urlencoded"}
    assert bearer == {"Authorization": "Bearer dummy-token-1", "Content-Type": "application/json"}
    sent = json.loads(body)
    assert sent["variables"] == {"id": "CIR-7", "n": W.HANDIN_PAGE, "after": None}
    assert sent["query"] == W.ISSUE_QUERY and "body" not in W.ISSUE_QUERY and "mutation" not in W.ISSUE_QUERY, \
        "a comment's text is never read, and nothing is written"
    assert cache.stat().st_mode & 0o777 == 0o600
    assert json.loads(cache.read_text()) == {
        "client": W.hashlib.sha256(b"dummy-client").hexdigest()[:16], "scope": "read,write", "access_token": "dummy-token-1",
        "expires_at": T0 + 30 * 86400}, "the team client's cache, entry for entry, and nothing of the secret"
    lin.issue("CIR-7", "cursor-2")
    assert len(http.posts) == 3 and json.loads(http.posts[-1][1])["variables"]["after"] == "cursor-2", "the token is reused"
    assert W.Linear(env, cache, clock, http).issue("CIR-7")["state"] == "started" and http.n == 1, "and by the next process"
    http.refuse.add("dummy-token-1")
    lin.issue("CIR-7")
    assert http.n == 2 and [h["Authorization"] for _, _, h in http.posts[-3:]] == [
        "Bearer dummy-token-1", head["Authorization"], "Bearer dummy-token-2"], "a refused token is fetched again, once"
    http.refuse.add("dummy-token-2")
    http.refuse.add("dummy-token-3")
    with pytest.raises(W.urllib.error.HTTPError):
        lin.issue("CIR-7")
    assert http.n == 3
    http.refuse.clear()
    clock.t += 30 * 86400 - W.LINEAR_EARLY_S
    lin.issue("CIR-7")
    assert http.n == 4, "replaced an hour before it expires"
    env.write_text("LINEAR_CLIENT_ID=dummy-other\nLINEAR_CLIENT_SECRET=dummy-secret\n")
    lin.issue("CIR-7")
    assert http.n == 5, "a token kept for another client is never used: a rotated credential needs no restart"
    for damaged in ("[not a cache", "[]", json.dumps({**json.loads(cache.read_text()), "expires_at": "soon"})):
        cache.write_text(damaged)
        lin.issue("CIR-7")
    assert http.n == 8 and json.loads(cache.read_text())["access_token"] == "dummy-token-8", "a cache it can't use is replaced"


def test_the_linear_reader_says_what_it_cant_read_without_a_credentials_value(tmp_path):
    env, clock = tmp_path / "relay.env", Clock(T0)
    lin = W.Linear(env, tmp_path / "linear-token.json", clock, http := FakeHttp())
    for text in (None, "", "LINEAR_CLIENT_ID=dummy-client\n", "LINEAR_CLIENT_SECRET=dummy-secret\nLINEAR_CLIENT_ID=\n"):
        if text is not None:
            env.write_text(text)
        with pytest.raises(W.NoCredential) as e:
            lin.issue("CIR-7")
        assert "dummy" not in str(e.value) and http.posts == []
    env.write_text("LINEAR_CLIENT_ID=dummy-client\nLINEAR_CLIENT_SECRET=dummy-secret\n")
    http.answer = {"errors": [{"message": "Entity not found: Issue"}], "data": None}
    with pytest.raises(ValueError, match="Linear: Entity not found: Issue") as e:
        lin.issue("CIR-99")
    assert not isinstance(e.value, W.NoCredential) and "dummy" not in str(e.value)
    for answer, why in (({"data": {"issue": None}}, "no issue CIR-99"), ([], "no issue CIR-99"),
                        ({"data": {"issue": {**ISSUE, "comments": None}}}, "CIR-99 came in another shape"),
                        ({"data": {"issue": {**ISSUE, "labels": {"nodes": [None]}}}}, "CIR-99 came in another shape"),
                        ({"data": {"issue": {**ISSUE, "comments": {"pageInfo": {"hasNextPage": False}, "nodes": [
                            {"id": "c-1", "createdAt": None}]}}}}, "a comment with no time")):
        http.answer = answer
        with pytest.raises(ValueError, match=why):
            lin.issue("CIR-99")
    assert issubclass(W.urllib.error.HTTPError, OSError), "a refused grant or a 400 is a read that failed, made again"


def test_a_shared_teams_sender_is_placed_by_the_lane_its_issue_is_under_read_once(tmp_path, capsys):
    """Two lanes file under one Linear team, so a key of that team at a name's start names neither, and such a sender
    went to infra to place by hand. Which lane an issue is under is in Linear, as the team's client reads it: the lane
    its `lane:` label names, and with no such label the team's own lane, the one the registry gives no `linear_label`.
    The issue is read once for an unplaced sender and what it gave is kept. A label naming a lane that doesn't share the
    team places nobody and is not read again; a read that failed, or an issue Linear can't find, is read again at the
    sender's next run's end."""
    labelled, bare, wrong, gone, down = (f"bc-{c * 4}-{c * 2}-{c * 2}-{c * 2}-{c * 6}" for c in ("a3", "a4", "a5", "a6", "a7"))
    names = {labelled: "CMP-377 HARDNESS-SPEC", bare: "CMP-12 TABLES", wrong: "CMP-13 ROWS", gone: "CMP-14 GONE",
             down: "CMP-15 LATER"}
    lin = FakeLinear(CMP_377={"labels": ["swarm", "lane:compute-research"]}, CMP_12={"labels": ["swarm"]},
                     CMP_13={"labels": ["lane:vllm"]}, CMP_15={"labels": ["lane:compute-accounting"]})
    api, clock, w, _, roots = nodes_setup(tmp_path, linear=lin)
    linear_teams(tmp_path)
    for agent, name in names.items():
        api.workers[agent] = worker("IDLE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    for n, agent in enumerate((labelled, bare, wrong, gone)):
        sends(roots, clock, w, agent, f"r20261105-11000{n}-000{n}")
    first, second, *rest = ended(tmp_path)
    assert {k: first.get(k) for k in ("agent", "handle", "role", "lane", "placed_by", "issue")} == {
        "agent": labelled, "handle": "compute-research", "role": "worker", "lane": "compute-research", "placed_by": "label",
        "issue": "CMP-377"}
    assert {k: second.get(k) for k in ("agent", "handle", "role", "lane", "placed_by", "issue")} == {
        "agent": bare, "handle": "compute-accounting", "role": "worker", "lane": "compute-accounting", "placed_by": "team",
        "issue": "CMP-12"}
    assert placed_files(tmp_path) == {
        f"placed-{labelled}.json": {"agent_id": labelled, "lane": "compute-research", "role": "worker", "name": names[labelled],
                                    "placed_by": "label", "issue": "CMP-377"},
        f"placed-{bare}.json": {"agent_id": bare, "lane": "compute-accounting", "role": "worker", "name": names[bare],
                                "placed_by": "team", "issue": "CMP-12"}}
    assert [(r["role"], r.get("handle")) for r in rest] == [("unknown", None)] * 2, "no guess for the others"
    assert lin.calls == [("CMP-377", None), ("CMP-12", None), ("CMP-13", None), ("CMP-14", None)]
    err = capsys.readouterr().err
    assert "by the lane label of its name's issue CMP-377" in err
    assert "by its name's issue CMP-12, which has no lane label: its team's own lane" in err
    for n, agent in enumerate((labelled, bare, wrong, gone)):
        sends(roots, clock, w, agent, f"r20261105-12000{n}-000{n}")
    assert lin.calls[4:] == [("CMP-14", None)], "read once per sender: a placement is the worker's entry, and a label " \
        "that places nobody is remembered; only the issue Linear couldn't find is asked for again"
    later = [r for r in ended(tmp_path) if r["agent"] == labelled][-1]
    assert (later["handle"], later["role"], later.get("placed_by")) == ("compute-research", "worker", None)
    lin.fail = W.urllib.error.URLError("timed out")
    sends(roots, clock, w, down, "r20261105-130000-0009")
    assert lin.calls[5:] == [("CMP-15", None)] and f"placed-{down}.json" not in placed_files(tmp_path)
    lin.fail = None
    sends(roots, clock, w, down, "r20261105-140000-0009")
    assert lin.calls[6:] == [("CMP-15", None)] and placed_files(tmp_path)[f"placed-{down}.json"]["lane"] == "compute-accounting"
    restarted = W.Watch(api, tmp_path / "events.jsonl", clock, registry=tmp_path / "registry.json", routes=tmp_path / "routes.d",
                        notes=tmp_path / "notes", nodes=tmp_path / "nodes.json", sh=w.sh, linear=lin)
    restarted.sweep(lambda r: None)
    sends(roots, clock, restarted, wrong, "r20261105-150000-0001")
    assert len(lin.calls) == 7, "nor after the watch restarts"


def test_an_issue_with_no_lane_label_places_nobody_when_no_one_lane_owns_its_team(tmp_path):
    """The team's own lane is the one sharing lane with no `linear_label`: a registry that marks none, or leaves two
    unmarked, names no owner, and an unlabelled issue of that team places nobody."""
    agent = "bc-a3a3a3a3-a3a3-a3a3-a3a3-a3a3a3a3a3a3"
    lin = FakeLinear(CMP_12={"labels": []})
    api, clock, w, _, roots = nodes_setup(tmp_path, linear=lin)
    reg = json.loads(linear_teams(tmp_path).read_text())
    for h in reg["handles"]:
        h.pop("linear_label", None)
    (tmp_path / "registry.json").write_text(json.dumps(reg))
    api.workers[agent] = worker("IDLE", W.iso(T0 - 600), "CMP-12 TABLES")
    w.sweep(lambda r: None)
    sends(roots, clock, w, agent, "r20261105-110000-0001")
    sends(roots, clock, w, agent, "r20261105-120000-0002")
    assert [r["role"] for r in ended(tmp_path)] == ["unknown"] * 2 and placed_files(tmp_path) == {}
    assert lin.calls == [("CMP-12", None)]


def test_a_name_one_lane_already_places_costs_no_linear_read(tmp_path):
    """The label is read only for a key of a team several lanes share that the name alone doesn't place: a key of one
    lane's team, a shared team's key with its lane named, a leading handle and a name with no key read nothing."""
    names = {"bc-a3a3a3a3-a3a3-a3a3-a3a3-a3a3a3a3a3a3": "CIR-9 WIDTHS", "bc-a4a4a4a4-a4a4-a4a4-a4a4-a4a4a4a4a4a4": "no key at all",
             "bc-a5a5a5a5-a5a5-a5a5-a5a5-a5a5a5a5a5a5": "CMP-377 HARDNESS-SPEC (compute-research)",
             "bc-a6a6a6a6-a6a6-a6a6-a6a6-a6a6a6a6a6a6": "vllm-longrow: rows", "bc-a7a7a7a7-a7a7-a7a7-a7a7-a7a7a7a7a7a7": "SEAT: CMP-9"}
    lin = FakeLinear(CMP_377={"labels": ["lane:compute-accounting"]}, CMP_9={"labels": ["lane:compute-accounting"]})
    api, clock, w, _, roots = nodes_setup(tmp_path, linear=lin)
    linear_teams(tmp_path)
    for agent, name in names.items():
        api.workers[agent] = worker("IDLE", W.iso(T0 - 600), name)
    w.sweep(lambda r: None)
    for n, agent in enumerate(names):
        sends(roots, clock, w, agent, f"r20261105-11000{n}-000{n}")
    assert lin.calls == [] and sorted(d["placed_by"] for d in placed_files(tmp_path).values()) == ["handle", "name", "name"]


def test_the_pods_wrapper_hands_the_watch_the_file_its_github_and_linear_credentials_are_in():
    """Without `--gh-env` the watch reads no open pull request and no Linear issue, and says only that it has no
    credential. The wrapper passes the pod's relay.env as a path: its values never enter the watch's environment."""
    sh = (W.Path(__file__).resolve().parents[3] / "infra/control/agent_watch.sh").read_text()
    assert " --gh-env /root/.config/verity/relay.env " in sh
    assert not re.search(r"(^|[\s;(])(\.|source)\s+\S*relay\.env", sh), "never sourced"
