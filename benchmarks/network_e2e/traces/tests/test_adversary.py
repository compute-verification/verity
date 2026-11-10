"""The hostile scheduler (`vllm_adversarial_plugin`) on a stand-in for vLLM's Scheduler that moves requests between its
waiting queue and running list as vLLM d9105ea80's does, and the tracer wrapped around it."""
from __future__ import annotations

import json
import re
from collections import deque
from types import SimpleNamespace as NS

import pytest


class Queue(deque):
    """`FCFSRequestQueue`'s methods the adversary calls."""

    def add_request(self, r):
        self.append(r)

    def prepend_request(self, r):
        self.appendleft(r)

    def remove_requests(self, rs):
        out = set(rs)
        keep = [r for r in self if r not in out]
        self.clear()
        self.extend(keep)


class Req:
    def __init__(self, i: int):
        self.request_id, self.status, self.num_preemptions = f"cmpl-req-{i:05d}-0-0123abcd", NS(name="WAITING"), 0
        self.num_output_tokens = 0


class Sched:
    """Admits waiting requests in queue order up to `cap` running, each with a 48-token prefix-cache hit and a 40-token
    prefill row; `_preempt_request` is vLLM's minus the block bookkeeping."""
    block_size = 16

    def __init__(self, n: int, cap: int = 8):
        self.waiting, self.running, self.current_step, self.cap = Queue(Req(i) for i in range(n)), [], 0, cap
        self.requests = {r.request_id: r for r in self.waiting}
        self.kv_cache_manager = NS(empty_kv_cache_blocks="empty", truncate_computed_blocks=lambda b, n: ("cut", b, n))
        self.reset_preempted_req_ids, self.admitted, self.cuts, self.hits = set(), [], [], []

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        assert request.status.name == "RUNNING"
        request.status, request.num_preemptions = NS(name="PREEMPTED"), request.num_preemptions + 1
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        return num_new_tokens

    def _get_local_prefix_cache_hit(self, request):
        return "blocks", 48, 0, False

    def add_request(self, request):
        self.waiting.add_request(request)

    def update_from_output(self, scheduler_output, model_runner_output):
        return {}

    def _free_request(self, request, delay_free_blocks: bool = False):
        return None, None

    def schedule(self, throttle_prefills: bool = False):
        self.current_step += 1
        admitted = []
        for r in list(self.running):
            assert self._reserve_prefill_lookahead(r, 41, 1) == 1
        while self.waiting and len(self.running) < self.cap:
            r = self.waiting.popleft()
            self.hits.append(self._get_local_prefix_cache_hit(r))
            self.cuts.append(self._reserve_prefill_lookahead(r, 0, 40))
            r.status = NS(name="RUNNING")
            self.running.append(r)
            admitted.append(r)
        self.admitted.append(admitted)
        preempted, self.reset_preempted_req_ids = self.reset_preempted_req_ids, set()
        return NS(scheduled_new_reqs=[NS(req_id=r.request_id, num_computed_tokens=0, prompt_len=40) for r in admitted],
                  scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[], num_output_tokens=[], resumed_req_ids=set()),
                  num_scheduled_tokens={r.request_id: 40 for r in admitted}, total_num_scheduled_tokens=40 * len(admitted),
                  preempted_req_ids=preempted, finished_req_ids=set())


def serve(tmp_path, monkeypatch, mod, cache: str, n: int = 60, seed: int = 7):
    """Serve n requests to the end, finishing the oldest running request after every call."""
    plugin = mod("vllm_adversarial_plugin")
    monkeypatch.setenv(plugin.SEED, str(seed))
    monkeypatch.setenv(plugin.CACHE, cache)
    monkeypatch.setenv(plugin.TRACE_DIR, str(tmp_path / "raw"))

    class S(Sched):
        pass

    plugin.install(S)
    plugin.install(S)
    s = S(n)
    finished, calls = [], 0
    while s.waiting or s.running:
        before = {r.request_id for r in [*s.waiting, *s.running]}
        out = s.schedule()
        calls += 1
        assert calls < 50 * n, "the hostile scheduler starved a request"
        after = [r.request_id for r in [*s.waiting, *s.running]]
        assert sorted(after) == sorted(before), "a request was lost or doubled"
        assert not {r.request_id for r in s.admitted[-1]} & out.preempted_req_ids, "a request preempted and resumed at once"
        if s.running:
            done = s.running.pop(0)
            done.status = NS(name="FINISHED")
            finished.append(done.request_id)
    assert sorted(finished) == sorted(s.requests)
    (log,) = (tmp_path / "raw").glob("adversary.*.jsonl")
    return plugin, s, [json.loads(x) for x in log.read_text().splitlines()]


def test_hostile_choices_keep_every_request_and_finish_them_all(tmp_path, monkeypatch, mod):
    plugin, s, calls = serve(tmp_path, monkeypatch, mod, "refuse")
    forced = [x for c in calls for x in c["preempted"]]
    assert forced and all(re.fullmatch(r"req-\d{5}", x) for x in forced)
    assert max(forced.count(x) for x in forced) <= plugin.MAX_FORCED
    assert all(s.requests[f"cmpl-{x}-0-0123abcd"].num_preemptions >= forced.count(x) for x in forced)
    assert any(c["held"] for c in calls) and all(len(a) <= c["admissible"] for a, c in zip(s.admitted, calls))
    orders = {tuple(r.request_id for r in a) for a in s.admitted if len(a) > 2}
    assert any(list(o) != sorted(o) for o in orders), "the waiting queue was never shuffled"
    assert [c["step"] for c in calls] == sorted(c["step"] for c in calls)
    splits = [x for c in calls for x in c["splits"]]
    assert splits and all(x[1] == 40 and 1 <= x[2] <= 39 for x in splits)
    assert sorted(s.cuts) != [40] * len(s.cuts) and 40 in s.cuts
    hits = [x for c in calls for x in c["hits"]]
    assert {x[2] for x in hits} == {0, 48} and {h[1] for h in s.hits} == {0, 48}
    assert not list((tmp_path / "raw").glob("adversary_errors.*"))


def test_shortest_cuts_every_hit_to_one_block_and_natural_leaves_it(tmp_path, monkeypatch, mod):
    _, s, calls = serve(tmp_path / "s", monkeypatch, mod, "shortest")
    assert {h for h in s.hits} == {(("cut", "blocks", 16), 16, 0, False)}
    assert {x[2] for c in calls for x in c["hits"]} == {16}
    _, s, calls = serve(tmp_path / "n", monkeypatch, mod, "natural", seed=8)
    assert {h for h in s.hits} == {("blocks", 48, 0, False)}


def test_an_error_in_a_choice_falls_back_to_the_honest_one(tmp_path, monkeypatch, mod):
    plugin = mod("vllm_adversarial_plugin")
    monkeypatch.setenv(plugin.SEED, "1")
    monkeypatch.setenv(plugin.CACHE, "shortest")
    monkeypatch.setenv(plugin.TRACE_DIR, str(tmp_path / "raw"))

    class S(Sched):
        pass

    plugin.install(S)
    s = S(4)
    s.kv_cache_manager = NS()
    s.schedule()
    assert {h for h in s.hits} == {("blocks", 48, 0, False)} and len(s.running) <= 4
    (err,) = (tmp_path / "raw").glob("adversary_errors.*.txt")
    assert "--- _get_local_prefix_cache_hit" in err.read_text()
    with pytest.raises(ValueError, match="VERITY_ADV_CACHE"):
        plugin.Adversary(1, "sometimes", None)


def test_the_tracer_wraps_the_adversary_and_sees_the_queue_put_back(tmp_path, monkeypatch, mod):
    corpus, adversary, tracer = mod("corpus"), mod("vllm_adversarial_plugin"), mod("vllm_trace_plugin")
    site = corpus.plugin_site(tmp_path / "site")
    (ep,) = site.glob("*.dist-info/entry_points.txt")
    assert ep.read_text().splitlines()[1:] == ["verity_adversary = vllm_adversarial_plugin:register",
                                               "verity_trace = vllm_trace_plugin:register"]
    monkeypatch.setenv(adversary.SEED, "3")
    monkeypatch.setenv(adversary.TRACE_DIR, str(tmp_path / "raw"))
    monkeypatch.setenv(tracer.TRACE_DIR, str(tmp_path / "raw"))

    class S(Sched):
        pass

    adversary.install(S)
    tracer.install(S)
    s = S(30)
    waiting = {}
    for _ in range(12):
        s.schedule()
        waiting[s.current_step] = len(s.waiting)
        if s.running:
            s.running.pop(0)
    (log,) = (tmp_path / "raw").glob("steps.*.jsonl")
    steps = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(steps) >= 6 and all(x["waiting"] == waiting[x["sched_step"]] for x in steps)
    (log,) = (tmp_path / "raw").glob("adversary.*.jsonl")
    assert sum(json.loads(x)["held"] for x in log.read_text().splitlines()) > 0


def test_register_without_a_seed_installs_nothing(monkeypatch, mod):
    plugin = mod("vllm_adversarial_plugin")
    monkeypatch.delenv(plugin.SEED, raising=False)
    plugin.register()
