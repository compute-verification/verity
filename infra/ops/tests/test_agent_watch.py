"""The agent watch (INF-69) frees a held run: RUNNING, its last event a finished tool call, then nothing but heartbeats for
15 minutes. It cancels the run, starts a 'continue' run on the agent, and records it; it never cancels a long tool call."""
import json
import re

import pytest

from verity_ops import agent_watch as W


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeApi:
    """`older[agent]` lists an agent's runs before its latest, newest first; `GET .../runs` lists the latest, then those.
    A goal run's id (run-goal-...) is listed but can't be read on its own, as on the API."""

    def __init__(self, agents, runs, older=None):
        self._agents, self._runs, self.older, self.posts = agents, runs, older or {}, []

    def agents(self):
        return iter(self._agents)

    def get(self, path, **params):
        parts = path.split("/")
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
    """A tool call as the API streams it: the simple event, then its `interaction_update` twin (18 Oct: 220 of 240)."""
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
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "c1", "name": "Shell", "status": "running"})
    call_from = clock.t
    clock.t += W.LONG_CALL_S - 60
    w.sweep(started.append)
    assert status(tmp_path)["bc-a"]["watch_state"] == "followed"
    clock.t += 60
    w.sweep(started.append)
    w.sweep(started.append)
    assert api.posts == []
    assert [r["kind"] for r in records(tmp_path)] == ["long_call"], "recorded once, never cancelled"
    assert [status(tmp_path)["bc-a"][k] for k in ("watch_state", "since")] == ["long_call", W.iso(call_from)]
    w.on_event("run-1", "tool_call", {"callId": "c1", "status": "completed"})
    w.on_event("run-1", "assistant", {"text": "done with the build"})
    clock.t += W.HELD_S
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
    """18 Oct: a goal run's id (run-goal-...) got 400 from the runs endpoint and stopped every sweep."""
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
    to ring @top with."""
    api, clock = FakeApi([{"id": "bc-g", "status": "ACTIVE", "latestRunId": "run-goal-x"}], {}), Clock()
    w, first = W.Watch(api, tmp_path / "events.jsonl", clock), clock.t
    for _ in range(3 * 60):
        w.sweep(lambda r: None)
        clock.t += 60
    assert [(r["kind"], r.get("since"), r.get("minutes")) for r in records(tmp_path)] == [
        ("unreadable", None, None), ("unreadable", W.iso(first), 60)]
    assert status(tmp_path)["bc-g"]["since"] == W.iso(first)


def test_a_run_stuck_in_creating_or_background_work_is_recorded_once_never_cancelled(tmp_path):
    """18 Oct: circuits' latest runs sat in CREATING and WAITING_FOR_BACKGROUND_WORK from 09:49Z."""
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
    api = FakeApi([{"id": "bc-b", "status": "ACTIVE", "latestRunId": "run-b"}],
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
    """vllm, 18 Oct: run after run ended in ERROR within a minute."""
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


def test_a_tool_calls_interaction_update_twin_doesnt_hide_a_held_run(tmp_path):
    """18 Oct, live streams: every `tool_call` completed is followed within milliseconds by an `interaction_update`
    `tool-call-completed` (220 of 240 sampled). The watch took it for the agent's own text, so no run could ever be held."""
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
    """A cloud `task` call stays `running` in the parent's stream while its subagent works (benchmarks, lean and
    compute-accounting had four each on 18 Oct, the parent thinking and calling tools meanwhile), and the subagent's progress
    streams as `tool-call-delta`. A coordinator always has some, so none could be held, and each was recorded as a 45-minute
    tool call."""
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
    with nothing at all streamed for 2 h is recorded as stuck, once, and never cancelled."""
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
    w.sweep(started.append)
    w.sweep(started.append)
    assert [(r["kind"], r["status"], r["minutes"]) for r in records(tmp_path)] == [("stuck", "WAITING_FOR_BACKGROUND_WORK", 120)]
    assert api.posts == []


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
    """Shadow run, 20:06Z 18 Oct: memory-accounting's background wait read as after a finished tool call, a worker's `task`
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
    """memory-accounting, 19:32Z 18 Oct: `latestRunId` was a run created and cancelled within 0.5 s, with run-7bb5675e RUNNING
    behind it. The watch read only the latest, and dropped a followed run as soon as a newer one appeared."""
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
    """network-accounting, 18 Oct: a goal run RUNNING from 18:27:22Z, a run CREATING behind it from 18:27:29Z, and every later run
    cancelled on arrival. The latest was always a cancelled one, so 1.5 h passed with nothing recorded (proofs from 08:26Z too)."""
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
    """memory-accounting, 20:01Z 18 Oct: run-6fd1b6c4 CREATING from 19:50:43Z, but run-84157683, created 4 s after it, started
    and waits on background work: the CREATING one blocks nothing (so does proofs' from 08:26Z, after 15:32Z)."""
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
    """infra, 20:07Z 18 Oct: run-1fba26b9 CREATING from 19:56:42Z behind run-87bda329, RUNNING from 19:55:57Z and followed,
    every later run cancelled on arrival. The CREATING one waits its turn; the watch judges the run ahead of it."""
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
    """memory-accounting, 20:40Z 18 Oct: run-84157683 WAITING from 19:50:47Z sat third on the second page of its runs, behind 49
    runs cancelled on arrival and a CREATING one; the watch read one page, lost the run and recorded the CREATING one stuck."""
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
    """The watch restarts on every deploy (six times from 17:58 to 18:24Z on 18 Oct). Its once-an-hour memory and its restart
    counts were in memory only, so a run that held again was cancelled again after each restart."""
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
    """console reads a row as final once written (20:52Z, 18 Oct): a second publish at the same t keeps the first body, and one
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


def coordinators(tmp_path, agents, runs, wake=False, bells=None, older=None):
    """The registry: CO holds @circuits (its lane LANE posts as it), and @vllm still names OLD, though NEW holds vllm's inbox
    (routes.d), as on 18 Oct. The watch over `agents` (`older` as in FakeApi)."""
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
    api, clock = FakeApi(agents, runs, older), Clock(T0)
    return api, clock, W.Watch(api, tmp_path / "events.jsonl", clock, registry=reg, routes=routes, wake=wake)


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
    assert rec["reason"] == "nothing has run for 40 min since its latest run went FINISHED at 2026-10-18T19:20:00Z"
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
    """19:38Z, 18 Oct: 12 of 14 coordinators sat in WAITING_FOR_BACKGROUND_WORK between wakes, so a missed timer leaves that."""
    run = {"status": "WAITING_FOR_BACKGROUND_WORK", "createdAt": "2026-10-18T19:00:00Z", "updatedAt": "2026-10-18T19:40:00Z"}
    api, clock, w = coordinators(tmp_path, [{"id": NEW, "status": "ACTIVE", "latestRunId": "run-b"}], {"run-b": run})
    started = []
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-b"] and records(tmp_path) == [], "20 min since it went to background work"
    clock.t += 5 * 60
    w.on_event("run-b", "assistant", {"text": "a background shell said something"})
    clock.t += W.WAKE_S - 60
    w.sweep(started.append)
    assert records(tmp_path) == [], "an event 29 min ago: counted from it, not from updatedAt"
    clock.t += 60
    w.sweep(started.append)
    (rec,) = records(tmp_path)
    assert (rec["handle"], rec["agent"], rec["run"]) == ("vllm", NEW, "run-b")
    assert rec["reason"] == "nothing has run for 30 min since its latest run went WAITING_FOR_BACKGROUND_WORK at 2026-10-18T20:05:00Z"


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
    assert path == f"/v1/agents/{CO}/runs" and "duty timer" in body["prompt"]["text"]
    assert "research msg read --as circuits" in body["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["action"], rec["new_run"]) == ("ready_to_wake", "started a run", "run-new")
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert len(api.posts) == 2, "still idle an hour later: woken again"


def test_a_coordinator_whose_latest_run_was_cancelled_on_arrival_is_judged_by_the_run_behind_it(tmp_path):
    """W1-1. vllm, 20:23Z 18 Oct: its run FINISHED, then a doorbell mid-turn made a run cancelled within 0.5 s, and that is its
    latestRunId (7 of 21 live agents read so). Judged by latestRunId, a coordinator whose timer then fails is never woken."""
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
    """W1-2. memory-accounting, 18 Oct: zombie CREATING runs from 19:50Z, so the API called it busy for hours. A wake whose 409
    dropped the hour's memory was posted every sweep, and nothing was recorded."""
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
                    wake=True)
    again.sweep(lambda r: None)
    assert len(api.posts) == 1, "woken again 10 min later by the restarted watch"
    clock.t += W.RECANCEL_S
    again.sweep(lambda r: None)
    assert len(api.posts) == 2


def test_the_terminal_run_cache_and_the_runs_already_judged_are_kept_apart(tmp_path):
    """W1-6. W1's watch cached terminal run records in `done`, which here is the runs judged once (state.json: run -> when). One
    dict for both: an ERROR run read is never restarted, state.json fills with records, and a cached record that dropped
    `updatedAt` reads as never active."""
    agents = [{"id": CO, "status": "IDLE", "latestRunId": "run-f", "updatedAt": W.iso(T0)},
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
