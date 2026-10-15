"""`engine.arrivals.run_requests_arrivals`'s timing: the per-request durations (``request_timing``) and the per-step entries
(``step_timing``) the benchmarks harness reads, measured on the injected clock over a stub engine."""
from __future__ import annotations

from types import SimpleNamespace

from verity_vllm.engine.arrivals import run_requests_arrivals


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Engine:
    """Each ``step()`` decodes one token for every admitted request and returns it at once (no output lag), taking
    ``per_request`` seconds of the clock per request it runs."""

    def __init__(self, clock: _Clock, per_request: float) -> None:
        self.clock, self.per_request = clock, per_request
        self.live: dict[str, list[int]] = {}
        self.want: dict[str, int] = {}

    def add_request(self, request_id, prompt, params) -> None:
        self.live[request_id], self.want[request_id] = [], params.max_tokens

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def step(self):
        self.clock.now += self.per_request * len(self.live)
        outs = []
        for rid, toks in list(self.live.items()):
            toks.append(7)
            done = len(toks) == self.want[rid]
            outs.append(SimpleNamespace(request_id=rid, finished=done,
                                        outputs=[SimpleNamespace(token_ids=list(toks), finish_reason="length" if done else None)]))
            if done:
                del self.live[rid]
        return outs


def _run(**kw):
    clock = _Clock()
    llm = SimpleNamespace(llm_engine=_Engine(clock, per_request=0.5))
    reqs = [SimpleNamespace(request_id="a", prompt_token_ids=[1, 2], max_tokens=3, arrive_step=0),
            SimpleNamespace(request_id="b", prompt_token_ids=[3], max_tokens=1, arrive_step=1)]
    return run_requests_arrivals(llm, reqs, lambda r: SimpleNamespace(max_tokens=r.max_tokens), clock=clock, **kw)


def test_step_timing_records_each_steps_duration_outputs_and_forward_index():
    finished, timing = _run(step_timing=True)
    assert sorted(finished) == ["a", "b"]
    assert timing["steps"] == [
        {"iter": 0, "s": 0.5, "outputs": 1, "forward": True, "forward_index": 0},
        {"iter": 1, "s": 1.0, "outputs": 2, "forward": True, "forward_index": 1},
        {"iter": 2, "s": 0.5, "outputs": 1, "forward": True, "forward_index": 2},
    ]
    assert sum(s["s"] for s in timing["steps"]) == timing["wall_s"] == 2.0
    assert timing["engine_steps"] == 3


def test_request_timing_measures_from_admission_on_the_injected_clock():
    _, timing = _run(request_timing=True)
    a, b = timing["requests"]["a"], timing["requests"]["b"]
    assert (a["arrival_s"], a["first_output_s"], a["time_to_first_token_s"]) == (0.0, 0.5, 0.5)
    assert (b["arrival_s"], b["first_output_s"], b["time_to_first_token_s"]) == (0.5, 1.5, 1.0)


def test_timing_off_leaves_the_record_as_it_was():
    _, timing = _run()
    assert "steps" not in timing
    assert "time_to_first_token_s" not in timing["requests"]["a"]
