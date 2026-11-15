"""The cut at each window edge (`cut_plugin`) through the engine core's seam (`engine.core_gate`), on stand-ins for vLLM
d9105ea80's Scheduler, which admits, decodes and preempts as it does (a step's tokens are in flight until its output is
processed), and its EngineCore, stepping one step at a time or two deep through its batch queue, its input loop before
each step as vLLM's busy loop runs it."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import queue
import threading
import tomllib
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from verity_vllm.engine import core_gate as cg
from verity_vllm.pipeline import cut_plugin as CP


class Queue(deque):
    """`FCFSRequestQueue`'s methods the scheduler and the cut call."""

    def add_request(self, r):
        self.append(r)

    def prepend_request(self, r):
        self.appendleft(r)

    def remove_requests(self, rs):
        out = {id(r) for r in rs}
        keep = [r for r in self if id(r) not in out]
        self.clear()
        self.extend(keep)


class Req:
    def __init__(self, i: int, prompt: int = 6, max_tokens: int = 9, **params):
        self.request_id, self.num_prompt_tokens, self.max_tokens = f"req-{i}", prompt, max_tokens
        self.num_output_tokens = self.num_computed_tokens = self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.status, self.sampling_params = "WAITING", NS(**params)
        self.preempted: list[bool] = []


class Sched:
    """Admits waiting requests in queue order up to ``most`` running, prefilling each whole, then decodes one token a
    step; a scheduled token is in flight until `update` processes its step. `_preempt_request` is vLLM's: it frees the
    request's blocks, resets its computed tokens and puts it at the waiting queue's front. ``requests`` is its request
    table, in the order `add_request` received them."""
    block_size = 4

    def __init__(self, most: int = 3, policy: str = "fcfs", connector=None, dp: int = 1):
        self.scheduler_config, self.connector, self.most = NS(policy=policy), connector, most
        self.vllm_config = NS(parallel_config=NS(data_parallel_size=dp))
        self.waiting, self.skipped_waiting, self.current_step = Queue(), Queue(), 0
        self.running: list[Req] = []
        self.requests: dict[str, Req] = {}
        self.prev_step_scheduled_req_ids: set[str] = set()
        self.reset_preempted_req_ids: set[str] = set()
        self.held: dict[str, int] = {}
        self.kv_cache_manager = NS(get_block_ids=lambda rid: ([0] * self.held.get(rid, 0),))
        self.steps: list[dict] = []

    def add_request(self, r):
        self.requests[r.request_id] = r
        self.waiting.add_request(r)

    def has_requests(self) -> bool:
        return bool(self.running or self.waiting)

    def _preempt_request(self, r, timestamp, drop_stale_output=False):
        assert r.status == "RUNNING" and r.num_in_flight_tokens == 0
        self.held.pop(r.request_id, None)
        r.status, r.num_computed_tokens = "PREEMPTED", 0
        r.preempted.append(drop_stale_output)
        self.waiting.prepend_request(r)
        self.reset_preempted_req_ids.add(r.request_id)

    def schedule(self):
        self.current_step += 1
        step: dict[str, Any] = {"prev": set(self.prev_step_scheduled_req_ids), "admitted": [], "scheduled": {}}
        for r in self.running:
            step["scheduled"][r.request_id] = 1
        while self.waiting and len(self.running) < self.most:
            r = self.waiting.popleft()
            r.status = "RUNNING"
            self.running.append(r)
            step["admitted"].append(r.request_id)
            step["scheduled"][r.request_id] = r.num_prompt_tokens + r.num_output_tokens - r.num_computed_tokens
        for r in self.running:
            n = step["scheduled"][r.request_id]
            r.num_computed_tokens += n
            r.num_in_flight_tokens += n
            self.held[r.request_id] = -(-r.num_computed_tokens // self.block_size)
        step["preempted"], self.reset_preempted_req_ids = self.reset_preempted_req_ids, set()
        self.prev_step_scheduled_req_ids = set(step["scheduled"])
        self.steps.append(step)
        return step

    def update(self, step):
        for rid, n in step["scheduled"].items():
            r = next((r for r in self.running if r.request_id == rid), None)
            if r is None:
                continue
            r.num_in_flight_tokens -= n
            r.num_output_tokens += 1
            if r.num_output_tokens >= r.max_tokens:
                self.running.remove(r)
                self.held.pop(rid, None)
                self.requests.pop(rid)
                r.status = "FINISHED"


class Inputs:
    """The engine core's input queue, which no client request reaches: each wait on it is its timeout on the test's clock
    (``wait``), after which it is empty."""

    def __init__(self, wait) -> None:
        self.wait = wait

    def empty(self) -> bool:
        return True

    def get(self, timeout=None):
        assert timeout is not None, "the input loop waited for a client request that never comes"
        self.wait(timeout)
        raise queue.Empty


class Core:
    """vLLM's EngineCore as the seam drives it: `_process_input_queue` returns once it has work, `step` schedules, runs
    and processes one step, and `step_with_batch_queue` keeps two in flight, as with async scheduling (its batch queue two
    deep). ``wait`` is the test's clock passing while the input loop waits on its queue."""

    def __init__(self, sched: Sched, async_scheduling: bool = False, wait=lambda dt: None) -> None:
        self.scheduler = sched
        self.batch_queue: deque | None = deque() if async_scheduling else None
        self.input_queue, self.aborts_queue = Inputs(wait), NS(mutex=threading.Lock(), queue=deque())

    def _process_input_queue(self) -> None:
        pass

    def is_running(self) -> bool:
        return True

    def _notify_idle_state_callbacks(self) -> None:
        pass

    def has_work(self) -> bool:
        return self.scheduler.has_requests() or bool(self.batch_queue)

    def step(self):
        if not self.scheduler.has_requests():
            return {}, False
        self.scheduler.update(self.scheduler.schedule())
        return {}, True

    def step_with_batch_queue(self):
        assert self.batch_queue is not None
        s = self.scheduler
        if s.has_requests():
            self.batch_queue.appendleft(s.schedule())
            if len(self.batch_queue) < 2:
                return None, True
        elif not self.batch_queue:
            return None, False
        s.update(self.batch_queue.pop())
        return {}, True


@pytest.fixture
def engine():
    """``engine(make)``: an EngineCore class of its own behind a seam of its own with the edge ``"cut"`` made by
    ``make``; the seams' wraps come off afterwards."""
    seams = []

    def build(make, clock=lambda: 0.0):
        cls = type("C", (Core,), {})
        seam = cg.EngineCoreGate(proc=cls, core=cls, clock=clock)
        seam.edge("cut", make)
        seams.append(seam)
        return cls
    yield build
    for seam in seams:
        seam.uninstall()


def serve(cls, sched, reqs, *, arrive=None, async_scheduling=False, wait=lambda dt: None, limit=200):
    """Run an engine core of class ``cls`` over ``sched`` until every request finishes, its input loop then one step at
    a time; ``arrive`` maps a step call to the requests added before it (by default all before the first)."""
    core = cls(sched, async_scheduling, wait)
    step = core.step_with_batch_queue if async_scheduling else core.step
    arrive = arrive or {0: reqs}
    for k in range(limit):
        for r in arrive.get(k, ()):
            sched.add_request(r)
        if not core.has_work() and k > max(arrive):
            break
        core._process_input_queue()
        step()
    return sched.steps


def test_without_an_edge_the_scheduler_is_untouched(engine) -> None:
    plain, cut = Sched(), Sched()
    serve(Core, plain, [Req(i) for i in range(5)])
    steps = serve(engine(lambda core: CP.Cut(every_steps=10_000)), cut, [Req(i) for i in range(5)])
    assert [s["scheduled"] for s in steps] == [s["scheduled"] for s in plain.steps]


def test_a_sync_edge_preempts_every_running_request_and_requeues_by_release_order(engine) -> None:
    """Every 4 steps the running requests are preempted from the running list's tail, with their stale output dropped;
    the scheduler's step ids are cleared; the waiting queue holds them in the order they were added; and the edge's own
    step admits window w + 1's first requests from it, each recomputing its prompt and outputs. Released first, the
    continuations run ahead of the requests still waiting, which start once they finish."""
    sched, cut = Sched(), CP.Cut(every_steps=4)
    reqs = [Req(i) for i in (3, 1, 4, 0, 2)]
    steps = serve(engine(lambda core: cut), sched, reqs)
    e0 = cut.edges[0]
    assert e0["step"] == 4 and e0["drain_calls"] == 0 and e0["waits"] == 0 and not e0["over"]
    assert [c["id"] for c in e0["cut"]] == ["req-4", "req-1", "req-3"]
    assert [c["seq"] for c in e0["cut"]] == [2, 1, 0]
    assert all(c["n"] == c["e"] == c["a_O"] == 4 and c["P"] == 6 and c["computed"] == 9 and c["blocks"] == 3
               for c in e0["cut"])
    assert e0["requeued"] == ["req-3", "req-1", "req-4", "req-0", "req-2"]
    after = steps[4]
    assert after["prev"] == set() and after["preempted"] == {"req-3", "req-1", "req-4"}
    assert after["admitted"] == ["req-3", "req-1", "req-4"]
    assert after["scheduled"] == {"req-3": 10, "req-1": 10, "req-4": 10}
    assert all(r.preempted and all(r.preempted) for r in reqs[:3])
    assert [(e["step"], [(c["id"], c["n"], c["a_O"]) for c in e["cut"]]) for e in cut.edges[1:]] == [
        (8, [("req-4", 8, 4), ("req-1", 8, 4), ("req-3", 8, 4)]), (12, [("req-2", 3, 3), ("req-0", 3, 3)]),
        (16, [("req-2", 7, 4), ("req-0", 7, 4)])]
    assert len(steps) == 18 and all(r.status == "FINISHED" for r in reqs)


def test_an_async_edge_is_cut_once_the_seam_has_drained_the_step_in_flight(engine) -> None:
    """With a step in flight at the edge, the seam's next step only processes it; the cut comes at the one after, once
    its outputs are out, so no preempted request has tokens in flight, and the step after the cut schedules the
    continuations."""
    sched, cut = Sched(), CP.Cut(every_steps=4)
    steps = serve(engine(lambda core: cut), sched, [Req(i) for i in range(4)], async_scheduling=True)
    e0 = cut.edges[0]
    assert e0["drain_calls"] == 1 and e0["step"] == 4
    assert steps[4]["preempted"] == {"req-0", "req-1", "req-2"} and steps[4]["prev"] == set()
    assert steps[4]["admitted"] == ["req-0", "req-1", "req-2"]


def test_the_edge_holds_each_open_request_at_its_released_count(engine) -> None:
    """A request whose emitted outputs the network certifier hasn't released holds the cut: the seam's input loop asks
    `Cut.ready` once each poll_s it waits on its queue until they are, and nothing is scheduled meanwhile. Past hold_s
    the cut goes ahead and lists the request in ``over``."""
    t = [0.0]
    behind = {"req-1": 2}

    def wait(dt):
        t[0] += dt
        if t[0] >= 3:
            behind.pop("req-1", None)

    def released(r: Any) -> int:
        return r.num_output_tokens - behind.get(r.request_id, 0)

    sched, cut = Sched(), CP.Cut(every_steps=4, hold_s=5, released=released, clock=lambda: t[0], poll_s=1.0)
    steps = serve(engine(lambda core: cut), sched, [Req(0), Req(1)], wait=wait)
    e0 = cut.edges[0]
    assert e0["waits"] == 3 and e0["wait_s"] == 3.0 and not e0["over"] and e0["step"] == 4
    assert [c["e"] for c in e0["cut"]] == [4, 4] and len(steps) == 9

    behind["req-0"] = 1
    sched2, cut2 = Sched(), CP.Cut(every_steps=1, hold_s=5, released=released, clock=lambda: t[0], poll_s=1.0)
    serve(engine(lambda core: cut2), sched2, [Req(0, max_tokens=2)], wait=wait)
    e0 = cut2.edges[0]
    assert e0["waits"] == 5 and e0["over"] == ["req-0"] and e0["cut"][0]["e"] == 0


def test_the_cut_records_the_skipped_queue_and_leaves_it(engine) -> None:
    sched, cut = Sched(), CP.Cut(every_steps=1)
    a, b = Req(0), Req(1)
    sched.add_request(a)
    sched.requests[b.request_id] = b
    sched.skipped_waiting.add_request(b)
    core = engine(lambda core: cut)(sched)
    core.step()
    core.step()
    assert cut.edges[0]["skipped"] == ["req-1"] and list(sched.skipped_waiting) == [b]


def test_a_timed_edge_that_falls_while_idle_cuts_once(engine) -> None:
    t = [0.0]
    sched, cut = Sched(), CP.Cut(every_s=10, clock=lambda: t[0])
    sched.add_request(Req(0, max_tokens=50))
    core = engine(lambda core: cut)(sched)
    for now in (0.0, 4.0, 35.0, 36.0, 39.0, 40.0):
        t[0] = now
        core.step()
    assert [(e["due"], e["at"]) for e in cut.edges] == [(10.0, 35.0), (40.0, 40.0)]


def test_reads_outputs_and_the_pool_export_are_recorded(engine, tmp_path) -> None:
    """A request whose sampling reads its outputs is listed; the export runs after the cut, when no block is held."""
    seen = []

    def export(sched, k):
        seen.append(dict(sched.held))
        return {"nb": 8, "root": f"r{k}"}
    log = tmp_path / "cut.jsonl"
    sched, cut = Sched(), CP.Cut(every_steps=3, export=export, log=log)
    serve(engine(lambda core: cut), sched, [Req(0, frequency_penalty=0.5), Req(1)])
    assert seen and all(h == {} for h in seen)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines == cut.edges and lines[0]["reads_outputs"] == ["req-0"]
    assert [x["pool_row"]["root"] for x in lines] == [f"r{k}" for k in range(len(lines))]


def test_make_refuses_what_it_cannot_cut_and_register_adds_the_edge_to_the_seam(engine, monkeypatch, tmp_path) -> None:
    cfg = CP.cut_config({CP.CUT_ENV: json.dumps({"every_steps": 2, "hold_s": 1, "log": str(tmp_path)})})
    for sched, why in ((Sched(policy="priority"), "needs the FCFS scheduler"), (Sched(connector=object()), "KV connector"),
                       (Sched(dp=2), "one of 2 data-parallel engines")):
        with pytest.raises(RuntimeError, match=why):
            CP.make(cfg)(Core(sched))
    cut = CP.make(cfg)(Core(Sched()))
    assert (cut.every_steps, cut.hold_s, cut.file.name) == (2, 1.0, str(tmp_path / f"cut.{os.getpid()}.jsonl"))

    cls = type("C", (Core,), {})
    seam = cg.EngineCoreGate(proc=cls, core=cls)
    monkeypatch.setattr(cg, "SEAM", seam)
    monkeypatch.setattr(CP, "_REGISTERED", [])
    monkeypatch.setenv(CP.CUT_ENV, json.dumps({"every_steps": 2, "log": str(tmp_path / "r")}))
    try:
        CP.register()
        CP.register()
        sched = Sched()
        serve(cls, sched, [Req(0)])
        (log,) = (tmp_path / "r").glob("cut.*.jsonl")
        assert [json.loads(x)["step"] for x in log.read_text().splitlines()] == [2, 4, 6, 8]
    finally:
        seam.uninstall()


def test_the_distribution_declares_register_as_a_vllm_general_plugin() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]
    module, _, attr = project["entry-points"]["vllm.general_plugins"]["verity_cut"].partition(":")
    assert importlib.import_module(module) is CP and getattr(CP, attr) is CP.register


@pytest.mark.parametrize("raw, why", [
    ('{"every_s": 70, "every_steps": 4}', "one of every_s and every_steps"),
    ('{"hold_s": 1}', "one of every_s and every_steps"),
    ('{"every_s": 0}', "every_s is a positive number"),
    ('{"every_steps": 2.5}', "every_steps is a positive integer"),
    ('{"every_s": 70, "hold_s": -1}', "hold_s is a number of seconds"),
    ('{"every_s": 70, "at": 1}', "unknown keys"),
])
def test_the_setting_is_one_edge_period_and_a_hold(raw, why) -> None:
    with pytest.raises(ValueError, match=why):
        CP.cut_config({"VERITY_CUT": raw})


def test_the_setting_parses() -> None:
    assert CP.cut_config({}) is None
    assert CP.cut_config({"VERITY_CUT": '{"every_s": 70, "log": "/x"}'}) == {
        "every_s": 70.0, "every_steps": None, "hold_s": 30.0, "log": "/x", "feed": None}


def _stream(app, path: str, headers: dict[str, str]) -> list[dict]:
    scope = {"type": "http", "path": path, "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m: dict) -> None:
        sent.append(m)

    asyncio.run(app(scope, receive, send))
    return sent


def _event(token_ids: list[int], first: bool = False) -> bytes:
    choice = {"index": 0, "text": "x", "token_ids": token_ids, **({"prompt_token_ids": [1, 2, 3]} if first else {})}
    return b"data: " + json.dumps({"id": "cmpl-r", "choices": [choice]}).encode() + b"\n\n"


BODY = [_event([5, 6], first=True), _event([7])[:30], _event([7])[30:] + _event([]), _event([8, 9, 10]),
        b'data: {"id": "cmpl-r", "choices": [], "usage": {"completion_tokens": 6}}\n\ndata: [DONE]\n\n']


class Streams:
    async def __call__(self, scope: dict, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for i, chunk in enumerate(BODY):
            await send({"type": "http.response.body", "body": chunk, "more_body": i + 1 < len(BODY)})


def test_the_sent_feed_counts_each_streamed_response_s_output_ids_as_they_leave(tmp_path, monkeypatch) -> None:
    """The prompt's ids and an event split across two body messages count once each, and a request without an
    X-Request-Id or off the sampled routes is passed unrecorded; the engine core reads the counts by engine request id."""
    feed = tmp_path / "sent.jsonl"
    monkeypatch.setenv("VERITY_CUT", json.dumps({"every_steps": 8, "feed": str(feed)}))
    app = CP.SentFeed(Streams())
    out = _stream(app, "/v1/completions", {"X-Request-Id": "req-00007"})
    assert b"".join(m.get("body", b"") for m in out) == b"".join(BODY)
    _stream(app, "/v1/completions", {})
    _stream(app, "/health", {"X-Request-Id": "req-00008"})
    assert [json.loads(x) for x in feed.read_text().splitlines()] == [
        {"key": "req-00007", "sent": 2}, {"key": "req-00007", "sent": 3}, {"key": "req-00007", "sent": 6}]

    sent = CP.Sent(CP.FeedReader(feed))
    assert sent(NS(request_id="cmpl-req-00007-0-0123abcd")) == 0
    sent.poll()
    assert sent(NS(request_id="cmpl-req-00007-0-0123abcd")) == 6 and sent(NS(request_id="cmpl-req-00008-0-0123abcd")) == 0

    monkeypatch.delenv("VERITY_CUT")
    assert CP.SentFeed(Streams()).fd is None


def test_the_cut_polls_the_sent_counts_and_holds_at_them(engine) -> None:
    """The engine runs ahead of what the server has sent: the cut waits until each open request's sent count, read by
    ``poll`` on each ask of the seam's input loop, reaches its emitted one."""
    recs: list[list[dict]] = []
    feed = deque([[{"key": "req-00000", "sent": 2}], [{"key": "req-00000", "sent": 4}]])
    sent = CP.Sent(lambda: recs.pop() if recs else [])
    sched = Sched()
    cut = CP.Cut(every_steps=4, hold_s=60, released=sent, poll=sent.poll)
    r = Req(0)
    r.request_id = "cmpl-req-00000-0-0123abcd"
    core = engine(lambda core: cut)(sched, False, lambda dt: recs.append(feed.popleft()))
    sched.add_request(r)
    for _ in range(5):
        core._process_input_queue()
        core.step()
    e0 = cut.edges[0]
    assert e0["cut"][0]["n"] == e0["cut"][0]["e"] == 4 and e0["waits"] == 2 and sent.sent == {"req-00000": 4}
