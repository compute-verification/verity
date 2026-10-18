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
    assert json.loads((tmp_path / "status.json").read_text()) == {"t": W.iso(clock.t), "active": 1, "followed": ["run-1"],
                                                                   "coordinators": 0, "doorbells": "missing"}
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


def test_an_unreadable_run_is_recorded_once_and_the_sweep_goes_on(tmp_path):
    """18 Oct: a goal run's id (run-goal-...) got 400 from the runs endpoint and stopped every sweep."""
    api = FakeApi([{"id": "bc-g", "status": "ACTIVE", "latestRunId": "run-goal-x"},
                   {"id": "bc-a", "status": "ACTIVE", "latestRunId": "run-1"}], {"run-1": {"status": "RUNNING"}})

    def get(path, **params):
        if "run-goal" in path:
            raise W.urllib.error.HTTPError(path, 400, "Bad Request", {}, None)
        return api._runs[path.rsplit("/", 1)[1]]
    api.get = get
    w, started = W.Watch(api, tmp_path / "events.jsonl", Clock()), []
    w.sweep(started.append)
    w.sweep(started.append)
    assert [r.run for r in started] == ["run-1"]
    assert [(r["kind"], r["status"]) for r in records(tmp_path)] == [("unreadable", "HTTP 400")]


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


def test_a_background_wait_is_held_only_after_an_hour(tmp_path):
    """top, 18:07Z: a held agent can read WAITING_FOR_BACKGROUND_WORK. Cancelling one may take its local workers, so it waits 1 h."""
    api, clock, w, started = setup(tmp_path, status="WAITING_FOR_BACKGROUND_WORK")
    assert [(r.run, r.held_s) for r in started] == [("run-1", W.HELD_BG_S)]
    w.on_event("run-1", "tool_call", {"callId": "c1", "status": "completed"})
    clock.t += W.HELD_S
    w.sweep(started.append)
    assert api.posts == []
    clock.t += W.HELD_BG_S - W.HELD_S
    w.sweep(started.append)
    assert [p for p, _ in api.posts] == ["/v1/agents/bc-a/runs/run-1/cancel", "/v1/agents/bc-a/runs"]


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
    for n in (2, 3, 4):
        clock.t += W.RECANCEL_S
        api._agents[0]["latestRunId"] = f"run-e{n}"
        api._runs[f"run-e{n}"] = {"status": "ERROR"}
        w.sweep(started.append)
        w.sweep(started.append)
    assert len(api.posts) == 3
    assert [(r["kind"], r.get("action", "")[:9]) for r in records(tmp_path)] == [
        ("crashed", "restarted"), ("crashed", "restarted"), ("crashed", "restarted"), ("erroring", "none: lef")]
    api._agents[0].update({"status": "IDLE", "updatedAt": "1970-01-01T00:00:00Z", "latestRunId": "run-e5"})
    api._runs["run-e5"] = {"status": "ERROR"}
    clock.t += W.RECANCEL_S
    w.sweep(started.append)
    assert len(api.posts) == 3, "an IDLE agent not changed in 6 h isn't read"


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



T0 = W.epoch("2026-10-18T20:00:00Z")
CO, LANE, OLD, NEW = (f"bc-{c * 8}-{c * 4}-{c * 4}-{c * 4}-{c * 12}" for c in "cbde")
FINISHED = {"status": "FINISHED", "createdAt": "2026-10-18T19:10:00Z", "updatedAt": "2026-10-18T19:20:00Z"}


def coordinators(tmp_path, agents, runs, wake=False, bells=None):
    """The registry: CO holds @circuits (its lane LANE posts as it), and @vllm still names OLD, though NEW holds vllm's inbox
    (routes.d), as on 18 Oct. The watch over `agents`."""
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
    api, clock = FakeApi(agents, runs), Clock(T0)
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
    assert json.loads((tmp_path / "status.json").read_text())["coordinators"] == 2
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
    assert json.loads((tmp_path / "status.json").read_text())["doorbells"] == "fresh"


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
    assert records(tmp_path) == [] and json.loads((tmp_path / "status.json").read_text())["doorbells"] == "stale"


def test_wake_starts_one_run_an_hour_and_a_busy_agent_is_not_counted(tmp_path):
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f"}], {"run-f": dict(FINISHED)},
                                 wake=True)
    real_post = api.post

    def busy(path, body=None):
        raise W.urllib.error.HTTPError(path, 409, "Conflict", {}, None)
    api.post = busy
    w.sweep(lambda r: None)
    assert records(tmp_path) == [] and CO not in w.woken, "409 agent_busy: a run of its own started; not counted"
    api.post = real_post
    clock.t += 60
    w.sweep(lambda r: None)
    w.sweep(lambda r: None)
    ((path, body),) = api.posts
    assert path == f"/v1/agents/{CO}/runs" and "duty timer" in body["prompt"]["text"]
    assert "research msg read --as circuits" in body["prompt"]["text"]
    (rec,) = records(tmp_path)
    assert (rec["kind"], rec["action"], rec["new_run"]) == ("ready_to_wake", "started a run", "run-new")
    clock.t += W.RECANCEL_S
    w.sweep(lambda r: None)
    assert len(api.posts) == 2, "still idle an hour later: woken again"


def test_a_terminal_run_is_read_once(tmp_path):
    api, clock, w = coordinators(tmp_path, [{"id": CO, "status": "IDLE", "latestRunId": "run-f", "updatedAt": W.iso(T0)}],
                                 {"run-f": dict(FINISHED)})
    reads = []
    get = api.get
    api.get = lambda path, **p: reads.append(path) or get(path, **p)
    for _ in range(3):
        w.sweep(lambda r: None)
    assert reads == [f"/v1/agents/{CO}/runs/run-f"]
