"""The cut at each window edge (`cut_plugin`) on a stand-in for vLLM d9105ea80's Scheduler that admits, decodes and
preempts as it does (a step's tokens are in flight until its output is processed), driven by a sync loop and by the
async engine core's two-deep batch queue."""
from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace as NS

import pytest

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
        self.status, self.preempted, self.sampling_params = "WAITING", [], NS(**params)


class Sched:
    """Admits waiting requests in queue order up to ``cap`` running, prefilling each whole, then decodes one token a
    step; a scheduled token is in flight until `update` processes its step. `_preempt_request` is vLLM's: it frees the
    request's blocks, resets its computed tokens and puts it at the waiting queue's front."""
    block_size = 4

    def __init__(self, cap: int = 3, policy: str = "fcfs", connector=None):
        self.scheduler_config, self.connector, self.cap = NS(policy=policy), connector, cap
        self.waiting, self.skipped_waiting, self.running = Queue(), Queue(), []
        self.current_step, self.prev_step_scheduled_req_ids, self.reset_preempted_req_ids = 0, set(), set()
        self.held: dict[str, int] = {}
        self.kv_cache_manager = NS(get_block_ids=lambda rid: ([0] * self.held.get(rid, 0),))
        self.steps: list[dict] = []

    def add_request(self, r):
        self.waiting.add_request(r)

    def _preempt_request(self, r, timestamp, drop_stale_output=False):
        assert r.status == "RUNNING" and r.num_in_flight_tokens == 0
        self.held.pop(r.request_id, None)
        r.status, r.num_computed_tokens = "PREEMPTED", 0
        r.preempted.append(drop_stale_output)
        self.waiting.prepend_request(r)
        self.reset_preempted_req_ids.add(r.request_id)

    def schedule(self, throttle_prefills: bool = False):
        self.current_step += 1
        step = {"prev": set(self.prev_step_scheduled_req_ids), "admitted": [], "scheduled": {}}
        for r in self.running:
            step["scheduled"][r.request_id] = 1
        while self.waiting and len(self.running) < self.cap:
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
                r.status = "FINISHED"


def serve(sched, cut, reqs, *, arrive=None, async_scheduling=False, limit=200):
    """Drive ``sched`` behind ``cut`` (its `Cut.before` and `Cut.after` around each ``schedule()``) until every request
    finishes; ``arrive`` maps a step to the requests added before it (by default all before the first)."""
    arrive = arrive or {0: reqs}
    queue, k = deque(), 0
    while k < limit:
        for r in arrive.get(k, ()):
            cut.add(r)
            sched.add_request(r)
        if not (sched.running or sched.waiting or queue) and k > max(arrive):
            break
        held = cut.before(sched)
        try:
            out = sched.schedule()
        finally:
            cut.after(sched, held)
        queue.append(out)
        if not async_scheduling or len(queue) == 2:
            sched.update(queue.popleft())
        k += 1
    while queue:
        sched.update(queue.popleft())
    return sched.steps


def test_without_an_edge_the_scheduler_is_untouched() -> None:
    plain, cut = Sched(), Sched()
    serve(plain, CP.Cut(every_steps=10_000), [Req(i) for i in range(5)])
    steps = serve(cut, CP.Cut(every_steps=10_000), [Req(i) for i in range(5)])
    assert [s["scheduled"] for s in steps] == [s["scheduled"] for s in plain.steps]


def test_a_sync_edge_preempts_every_running_request_and_requeues_by_release_order() -> None:
    """Every 4 steps the running requests are preempted from the running list's tail, with their stale output dropped;
    the scheduler's step ids are cleared; the waiting queue holds them in the order they were added; and the edge's own
    step admits window w + 1's first requests from it, each recomputing its prompt and outputs. Released first, the
    continuations run ahead of the requests still waiting, which start once they finish."""
    sched, cut = Sched(), CP.Cut(every_steps=4)
    reqs = [Req(i) for i in (3, 1, 4, 0, 2)]
    steps = serve(sched, cut, reqs)
    e0 = cut.edges[0]
    assert e0["step"] == 4 and e0["held_steps"] == 0 and not e0["over"]
    assert [c["id"] for c in e0["cut"]] == ["req-4", "req-1", "req-3"]
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


def test_an_async_edge_holds_until_the_step_in_flight_is_processed() -> None:
    """With a step in flight at the edge, the edge's first schedule() schedules nothing; the cut comes at the next one,
    once that step's outputs are out, so no preempted request has tokens in flight."""
    sched, cut = Sched(), CP.Cut(every_steps=4)
    steps = serve(sched, cut, [Req(i) for i in range(4)], async_scheduling=True)
    e0 = cut.edges[0]
    assert e0["held_steps"] == 1 and e0["step"] == 5
    assert steps[4]["scheduled"] == {} and steps[4]["admitted"] == []
    assert steps[5]["preempted"] == {"req-0", "req-1", "req-2"} and steps[5]["prev"] == set()


def test_the_edge_holds_each_open_request_at_its_released_count() -> None:
    """A request whose emitted outputs the network certifier hasn't released holds the edge: no step is scheduled until
    they are. Past hold_s the cut goes ahead and lists the request in ``over``."""
    t = [0.0]
    behind = {"req-1": 2}
    released = lambda r: r.num_output_tokens - behind.get(r.request_id, 0)  # noqa: E731
    sched, cut = Sched(), CP.Cut(every_steps=4, hold_s=5, released=released, clock=lambda: t[0])
    for r in (Req(0), Req(1)):
        cut.add(r)
        sched.add_request(r)
    for _ in range(4):
        out = sched.schedule()
        sched.update(out)
    for k in range(3):
        t[0] = k
        held = cut.before(sched)
        sched.schedule()
        cut.after(sched, held)
        assert held is not None and not cut.edges
    behind.clear()
    t[0] = 3.5
    assert cut.before(sched) is None
    e0 = cut.edges[0]
    assert e0["held_steps"] == 3 and e0["held_s"] == 3.5 and not e0["over"]

    behind["req-0"] = 1
    sched2, cut2 = Sched(), CP.Cut(every_steps=1, hold_s=5, released=released, clock=lambda: t[0])
    r = Req(0)
    cut2.add(r)
    sched2.add_request(r)
    sched2.update(sched2.schedule())
    t[0] = 10.0
    held = cut2.before(sched2)
    assert held is not None
    cut2.after(sched2, held)
    t[0] = 15.0
    assert cut2.before(sched2) is None
    assert cut2.edges[0]["over"] == ["req-0"] and cut2.edges[0]["cut"][0]["e"] == 0


def test_a_held_edge_holds_the_skipped_queue_too() -> None:
    sched, cut = Sched(), CP.Cut(every_steps=1)
    a, b = Req(0), Req(1)
    for r in (a, b):
        cut.add(r)
    sched.add_request(a)
    sched.skipped_waiting.add_request(b)
    out = sched.schedule()
    held = cut.before(sched)
    assert held is not None and not sched.skipped_waiting and not sched.running
    cut.after(sched, held)
    assert list(sched.skipped_waiting) == [b] and sched.running == [a]
    sched.update(out)
    assert cut.before(sched) is None and cut.edges[0]["skipped"] == ["req-1"]


def test_a_timed_edge_that_falls_while_idle_cuts_once() -> None:
    t = [0.0]
    sched, cut = Sched(), CP.Cut(every_s=10, clock=lambda: t[0])
    r = Req(0, max_tokens=50)
    cut.add(r)
    sched.add_request(r)
    for now in (0.0, 4.0, 35.0, 36.0, 39.0, 40.0):
        t[0] = now
        held = cut.before(sched)
        out = sched.schedule()
        cut.after(sched, held)
        sched.update(out)
    assert [(e["due"], e["at"]) for e in cut.edges] == [(10.0, 35.0), (40.0, 40.0)]


def test_reads_outputs_and_the_pool_export_are_recorded(tmp_path) -> None:
    """A request whose sampling reads its outputs is listed; the export runs after the cut, when no block is held."""
    seen = []

    def export(sched, k):
        seen.append(dict(sched.held))
        return {"nb": 8, "root": f"r{k}"}
    log = tmp_path / "cut.jsonl"
    sched, cut = Sched(), CP.Cut(every_steps=3, export=export, log=log)
    serve(sched, cut, [Req(0, frequency_penalty=0.5), Req(1)])
    assert seen and all(h == {} for h in seen)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines == cut.edges and lines[0]["reads_outputs"] == ["req-0"]
    assert [x["pool_row"]["root"] for x in lines] == [f"r{k}" for k in range(len(lines))]


def test_install_wraps_every_scheduler_and_refuses_what_it_cannot_cut() -> None:
    class S(Sched):
        pass
    made = []

    def make(s):
        made.append(CP.Cut(every_steps=2))
        return made[-1]
    hooks = CP.install(S, make)
    try:
        s = S()
        assert s._verity_cut is made[0]
        r = Req(0)
        s.add_request(r)
        assert made[0].seq == {"req-0": 0}
        for _ in range(3):
            s.update(s.schedule())
        assert made[0].edges and made[0].edges[0]["cut"][0]["id"] == "req-0"
        with pytest.raises(RuntimeError, match="needs the FCFS scheduler"):
            S(policy="priority")
        with pytest.raises(RuntimeError, match="KV connector"):
            S(connector=object())
    finally:
        hooks.uninstall()
    assert not hasattr(S(), "_verity_cut")


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
        "every_s": 70.0, "every_steps": None, "hold_s": 30.0, "log": "/x"}
