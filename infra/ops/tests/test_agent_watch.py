"""The agent watch (INF-69) frees a held run: RUNNING, its last event a finished tool call, then nothing but heartbeats for
15 minutes. It cancels the run, starts a 'continue' run on the agent, and records it; it never cancels a long tool call."""
import json

from verity_ops import agent_watch as W


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeApi:
    def __init__(self, agents, runs):
        self._agents, self._runs, self.posts = agents, runs, []

    def agents(self):
        return iter(self._agents)

    def get(self, path, **params):
        return self._runs[path.rsplit("/", 1)[1]]

    def post(self, path, body=None):
        self.posts.append((path, body))
        return {"run": {"id": "run-new"}} if path.endswith("/runs") else {"id": "run-1"}


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


def test_a_run_held_after_a_finished_tool_call_is_cancelled_and_continued_once(tmp_path):
    api, clock, w, started = setup(tmp_path)
    assert [r.run for r in started] == ["run-1"]
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


def test_a_long_tool_call_or_fresh_text_is_never_cancelled(tmp_path):
    api, clock, w, started = setup(tmp_path)
    w.on_event("run-1", "tool_call", {"callId": "c1", "name": "Shell", "status": "running"})
    clock.t += W.LONG_CALL_S
    w.sweep(started.append)
    w.sweep(started.append)
    assert api.posts == []
    assert [r["kind"] for r in records(tmp_path)] == ["long_call"], "recorded once, never cancelled"
    w.on_event("run-1", "tool_call", {"callId": "c1", "status": "completed"})
    w.on_event("run-1", "assistant", {"text": "done with the build"})
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert api.posts == [], "the last event was the agent's own text, not a finished tool call"


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


def test_a_crashed_run_is_recorded_once(tmp_path):
    api, clock, w, started = setup(tmp_path, status="ERROR")
    w.sweep(started.append)
    assert started == [] and api.posts == []
    assert [r["kind"] for r in records(tmp_path)] == ["crashed"]


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
