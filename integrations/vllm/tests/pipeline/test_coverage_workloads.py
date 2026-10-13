"""Coverage campaign harness (D87): seeded workload draws, the step-indexed arrival loop, the control-run step expectation."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from verity_vllm.pipeline import coverage_workloads as gw

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------------------
# draws
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", gw.PROFILES)
@pytest.mark.parametrize("ctx", sorted(gw.CTX))
def test_draws_respect_the_context_class(profile, ctx):
    c = gw.CTX[ctx]
    d = gw.draw_lengths(profile, ctx, 0)
    assert len(d) == (1 if profile == "B1" else 8)
    for r in d:
        assert c["lo"] <= r["prompt_len"] <= c["cap"]
        assert r["max_tokens"] == max(c["min_out"], r["prompt_len"] // 10)
        assert r["max_tokens"] <= c["cap"] // 10 + 1
    if profile == "B8-mixed":
        assert d[0]["arrive_step"] == 0 and any(r["arrive_step"] > 0 for r in d[1:])
        assert [r["arrive_step"] for r in d] == sorted(r["arrive_step"] for r in d)
        sim = gw.simulate_schedule(d)
        assert sim["membership_changes"] >= 2 and sim["steps_with_mixed_prefill_decode"] >= 1
    else:
        assert all(r["arrive_step"] == 0 for r in d)
        assert gw.simulate_schedule(d)["expected_engine_steps"] == max(r["max_tokens"] for r in d)


def test_draws_are_seeded_and_model_independent():
    assert gw.draw_lengths("B8-static", "C256", 0) == gw.draw_lengths("B8-static", "C256", 0)
    assert gw.draw_lengths("B8-static", "C256", 0) != gw.draw_lengths("B8-static", "C256", 1)
    a = gw.draw_lengths("B8-static", "C256", 0)
    assert len({r["prompt_len"] for r in a}) > 1, "heterogeneous lengths within the batch"


def test_generated_b0_workloads_match_their_draws():
    for prof in gw.PROFILES:
        for ctx in sorted(gw.CTX):
            p = REPO / "verity_vllm" / "engine" / "profiles" / f"workload_cov_b0_{prof.lower()}_{ctx.lower()}.json"
            if not p.exists():
                pytest.skip(f"{p.name} not generated in this checkout")
            doc = json.loads(p.read_text())
            draws = gw.draw_lengths(prof, ctx, doc["length_seed"])
            assert [(r["prompt_len"], r["max_tokens"], r["arrive_step"]) for r in doc["requests"]] == \
                   [(d["prompt_len"], d["max_tokens"], d["arrive_step"]) for d in draws]
            for r in doc["requests"]:
                assert len(r["prompt_token_ids"]) == r["prompt_len"]
            assert doc["expected_engine_steps"] == gw.simulate_schedule(draws)["expected_engine_steps"]
            assert doc["serving_profile"] == prof and doc["context_class"] == ctx
            if prof != "B1":
                assert doc["engine_args_required"]["max_num_batched_tokens"] >= 8 * gw.CTX[ctx]["cap"]


# ---------------------------------------------------------------------------------------------------------
# the arrival loop (vllm_adapter.run_requests_arrivals) on a stub engine
# ---------------------------------------------------------------------------------------------------------


class _Out:
    def __init__(self, rid: str, toks: list[int], finished: bool) -> None:
        self.request_id = rid
        self.finished = finished
        self.outputs = [types.SimpleNamespace(index=0, token_ids=list(toks), finish_reason="length" if finished else None)]


class _StubEngine:
    """Every active request generates one token per step and finishes after max_tokens tokens."""

    def __init__(self) -> None:
        self.active: dict[str, dict] = {}
        self.added_at: dict[str, int] = {}
        self.batches: list[list[str]] = []
        self.step_no = 0

    def add_request(self, *, request_id: str, prompt: dict, params) -> None:
        self.active[request_id] = {"toks": [], "max": params.max_tokens}
        self.added_at[request_id] = self.step_no

    def has_unfinished_requests(self) -> bool:
        return bool(self.active)

    def step(self):
        self.batches.append(sorted(self.active))
        outs = []
        for rid in list(self.active):
            st = self.active[rid]
            st["toks"].append(len(st["toks"]))
            done = len(st["toks"]) >= st["max"]
            outs.append(_Out(rid, st["toks"], done))
            if done:
                del self.active[rid]
        self.step_no += 1
        return outs


def _load_adapter():
    """The arrival loop lives in the torch-free `verity_vllm.engine.arrivals` (vllm_adapter re-exports it)."""
    from verity_vllm.engine import arrivals
    return arrivals


def test_arrival_loop_admits_by_engine_step_and_records_realized_steps():
    va = _load_adapter()
    R = lambda rid, n, a: types.SimpleNamespace(request_id=rid, prompt_token_ids=[1, 2, 3], max_tokens=n, arrive_step=a, _order=0)  # noqa: E731
    reqs = [R("r0", 5, 0), R("r1", 3, 2), R("r2", 2, 2), R("r3", 4, 9)]     # r3 arrives after everything else has finished
    eng = types.SimpleNamespace(llm_engine=_StubEngine())
    seen_steps = []
    outs, timing = va.run_requests_arrivals(eng, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens),
                                            on_step=lambda o: seen_steps.append(len(o)))
    assert set(outs) == {"r0", "r1", "r2", "r3"}
    st = eng.llm_engine
    assert st.added_at == {"r0": 0, "r1": 2, "r2": 2, "r3": 5}, "r3's planned step 9 is brought forward to the idle step 5"
    assert st.batches[:5] == [["r0"], ["r0"], ["r0", "r1", "r2"], ["r0", "r1", "r2"], ["r0", "r1"]]
    rec = timing["requests"]
    assert rec["r1"] == {"arrive_step": 2, "arrive_step_realized": 2, "first_step": 2, "finish_step": 4, "finish_reason": "length", "n_generated": 3,
                         "admit_iter": 2, "finish_iter": 4}
    assert rec["r3"]["arrive_step"] == 9 and rec["r3"]["arrive_step_realized"] == 5 and rec["r3"]["finish_step"] == 8
    assert timing["engine_steps"] == 9 == len(seen_steps) and timing["arrivals"] is True


def test_arrival_loop_reports_request_latency_from_the_injected_clock():
    va = _load_adapter()
    request = types.SimpleNamespace(request_id="r0", prompt_token_ids=[1], max_tokens=1, arrive_step=0)
    ticks = iter((10.0, 11.0, 12.0, 13.0, 14.0))
    _, timing = va.run_requests_arrivals(
        types.SimpleNamespace(llm_engine=_StubEngine()),
        [request],
        lambda r: types.SimpleNamespace(max_tokens=r.max_tokens),
        clock=lambda: next(ticks),
        request_timing=True,
    )
    assert timing["requests"]["r0"]["finish_reason"] == "length"
    assert {
        key: timing["requests"]["r0"][key]
        for key in ("arrival_s", "first_output_s", "time_to_first_token_s")
    } == {"arrival_s": 1.0, "first_output_s": 3.0, "time_to_first_token_s": 2.0}


class _LaggedStubEngine(_StubEngine):
    """vLLM V1's shape: `step()` runs a forward for the active requests but RETURNS the previous forward's outputs (one-iteration
    output lag); a request stays 'unfinished' for the client until its finish output has been returned; a step with nothing active
    runs no forward and only drains the buffered outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.buffered: list = []
        self.unfinished_client: set[str] = set()
        self.forwards = 0

    def add_request(self, *, request_id: str, prompt: dict, params) -> None:
        super().add_request(request_id=request_id, prompt=prompt, params=params)
        self.unfinished_client.add(request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self.unfinished_client)

    def step(self):
        out = self.buffered
        if self.active:
            self.batches.append(sorted(self.active))
            self.forwards += 1
            new = []
            for rid in list(self.active):
                st = self.active[rid]
                st["toks"].append(len(st["toks"]))
                done = len(st["toks"]) >= st["max"]
                new.append(_Out(rid, st["toks"], done))
                if done:
                    del self.active[rid]
            self.buffered = new
        else:
            self.buffered = []
        for o in out:
            if o.finished:
                self.unfinished_client.discard(o.request_id)
        self.step_no += 1
        return out


def test_arrival_loop_counts_forwards_not_iterations_under_the_output_lag():
    va = _load_adapter()
    R = lambda rid, n, a: types.SimpleNamespace(request_id=rid, prompt_token_ids=[1], max_tokens=n, arrive_step=a, _order=0)  # noqa: E731
    eng = types.SimpleNamespace(llm_engine=_LaggedStubEngine())
    # one request, 15 tokens: 16 step() iterations (empty first return + one drain) but 15 forwards -- the observer's count
    outs, timing = va.run_requests_arrivals(eng, [R("x", 15, 0)], lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
    assert eng.llm_engine.forwards == 15
    assert timing["engine_steps"] == 15 and timing["step_iterations"] == 16 and timing["output_lag_steps"] == 1
    rec = timing["requests"]["x"]
    assert (rec["first_step"], rec["finish_step"], rec["n_generated"]) == (0, 14, 15) and rec["finish_iter"] == 15
    # staggered arrivals: forward indices still match the engine's forwards
    eng = types.SimpleNamespace(llm_engine=_LaggedStubEngine())
    reqs = [R("r0", 5, 0), R("r1", 3, 2), R("r3", 9, 9)]
    outs, timing = va.run_requests_arrivals(eng, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
    assert timing["engine_steps"] == eng.llm_engine.forwards
    rec = timing["requests"]
    assert rec["r1"]["first_step"] == 2 and rec["r1"]["finish_step"] == 4         # forwards 2,3,4 generate its 3 tokens
    assert rec["r0"]["finish_step"] == 4 and rec["r3"]["first_step"] == rec["r3"]["arrive_step_realized"]
    assert rec["r3"]["finish_step"] == rec["r3"]["first_step"] + 9 - 1 == timing["engine_steps"] - 1


class _ChunkedStubEngine(_LaggedStubEngine):
    """V1 with chunked prefill at `budget` tokens per forward: prompts are prefilled in chunks (one per forward, admission order,
    sharing the budget), a chunk that leaves the prompt incomplete produces NO output (the engine discards its sample), the chunk
    that completes it yields token 0.  `engine_core.scheduler.requests[rid].num_computed_tokens` mirrors the V1 scheduler state
    (advanced by the schedule, freed when the finish output is processed); `scheduler=False` hides it (multiprocess core)."""

    def __init__(self, budget: int = 64, scheduler: bool = True) -> None:
        super().__init__()
        self.budget = budget
        self.sched = types.SimpleNamespace(requests={})
        self.engine_core = types.SimpleNamespace(scheduler=self.sched) if scheduler else types.SimpleNamespace()

    def add_request(self, *, request_id: str, prompt: dict, params) -> None:
        super().add_request(request_id=request_id, prompt=prompt, params=params)
        n = len(prompt["prompt_token_ids"])
        self.active[request_id].update(prompt=n, computed=0)
        self.sched.requests[request_id] = types.SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=n)

    def step(self):
        out = self.buffered
        new, batch, budget = [], [], self.budget
        for rid in list(self.active):
            st = self.active[rid]
            if st["computed"] < st["prompt"]:
                take = min(st["prompt"] - st["computed"], budget)
                if take <= 0:
                    continue                              # waits: budget exhausted by an earlier prefill
                budget -= take
                st["computed"] += take
                batch.append(rid)
                self.sched.requests[rid].num_computed_tokens = st["computed"]
                if st["computed"] < st["prompt"]:
                    continue                              # partial prefill: a forward, but its sample is discarded
            else:
                if budget <= 0:
                    continue
                budget -= 1
                st["computed"] += 1
                batch.append(rid)
                self.sched.requests[rid].num_computed_tokens = st["computed"]
            st["toks"].append(len(st["toks"]))
            done = len(st["toks"]) >= st["max"]
            new.append(_Out(rid, st["toks"], done))
            if done:
                del self.active[rid]
        if batch:
            self.forwards += 1
            self.batches.append(sorted(batch))
        self.buffered = new
        for o in out:
            if o.finished:
                self.unfinished_client.discard(o.request_id)
                self.sched.requests.pop(o.request_id, None)
        self.step_no += 1
        return out


def test_arrival_loop_counts_partial_prefill_forwards_under_chunked_prefill():
    """The chunked owner's finding (lead's request, 2026-09-17): a 155-token prompt at budget 64 = 3 chunks, the first two return
    nothing; the old inference measured lag 3 and counted 15 forwards for 17.  Both sources now give 17 / lag 1."""
    va = _load_adapter()
    R = lambda rid, plen, n, a: types.SimpleNamespace(request_id=rid, prompt_token_ids=[1] * plen, max_tokens=n, arrive_step=a, _order=0)  # noqa: E731
    for scheduler, source in ((True, "scheduler"), (False, "clamped")):
        eng = types.SimpleNamespace(llm_engine=_ChunkedStubEngine(budget=64, scheduler=scheduler))
        outs, timing = va.run_requests_arrivals(eng, [R("x", 155, 15, 0)], lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
        assert eng.llm_engine.forwards == 17 and eng.llm_engine.batches[:3] == [["x"], ["x"], ["x"]]
        assert timing["engine_steps"] == 17 and timing["step_iterations"] == 18, (scheduler, timing["engine_steps"])
        assert timing["output_lag_steps"] == 1 and timing["output_lag_measured"] == 3 and timing["output_lag_source"] == source
        rec = timing["requests"]["x"]
        assert (rec["first_step"], rec["finish_step"], rec["n_generated"], rec["finish_iter"]) == (0, 16, 15, 17)
        assert len(outs["x"].outputs[0].token_ids) == 15
    # the old rule, for the record: forward at j <=> output at j + measured lag  ->  15
    nonempty = [False] * 3 + [True] * 15
    assert sum(j + 3 < len(nonempty) and nonempty[j + 3] for j in range(len(nonempty))) == 15
    # an idle gap (nothing to forward while the next arrival is not due) is still no forward, and a chunked prompt arriving after
    # it is counted from its first chunk -- with the scheduler and without (the first prefill is unchunked -> lag measured 1)
    for scheduler, source in ((True, "scheduler"), (False, "measured")):
        eng = types.SimpleNamespace(llm_engine=_ChunkedStubEngine(budget=64, scheduler=scheduler))
        reqs = [R("a", 10, 3, 0), R("b", 130, 4, 9)]
        outs, timing = va.run_requests_arrivals(eng, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
        assert eng.llm_engine.forwards == 9 == timing["engine_steps"] and timing["step_iterations"] == 11
        assert timing["output_lag_steps"] == 1 and timing["output_lag_source"] == source
        rec = timing["requests"]
        assert (rec["a"]["first_step"], rec["a"]["finish_step"]) == (0, 2)
        assert rec["b"]["admit_iter"] == 4 and (rec["b"]["arrive_step_realized"], rec["b"]["first_step"], rec["b"]["finish_step"]) == (3, 3, 8)
    # an unknown engine (no engine_core / model_executor) keeps the measured lag and says so
    eng = types.SimpleNamespace(llm_engine=_LaggedStubEngine())
    _, timing = va.run_requests_arrivals(eng, [R("x", 1, 4, 0)], lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
    assert timing["output_lag_source"] == "measured" and timing["output_lag_steps"] == timing["output_lag_measured"] == 1
    assert timing["engine_steps"] == 4 == eng.llm_engine.forwards
    eng = types.SimpleNamespace(llm_engine=_StubEngine())
    _, timing = va.run_requests_arrivals(eng, [R("x", 1, 4, 0)], lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
    assert timing["output_lag_source"] == "measured" and timing["output_lag_steps"] == 0 and timing["engine_steps"] == 4
    assert va.structural_output_lag(types.SimpleNamespace(engine_core=object())) == 1
    assert va.structural_output_lag(types.SimpleNamespace(model_executor=object())) == 0
    assert va.structural_output_lag(types.SimpleNamespace()) is None


def test_arrival_loop_all_at_zero_equals_the_driver_shape():
    va = _load_adapter()
    R = lambda rid, n: types.SimpleNamespace(request_id=rid, prompt_token_ids=[1], max_tokens=n, arrive_step=0, _order=0)  # noqa: E731
    eng = types.SimpleNamespace(llm_engine=_StubEngine())
    outs, timing = va.run_requests_arrivals(eng, [R("a", 2), R("b", 4)], lambda r: types.SimpleNamespace(max_tokens=r.max_tokens))
    assert eng.llm_engine.batches == [["a", "b"], ["a", "b"], ["b"], ["b"]]
    assert timing["engine_steps"] == 4 and timing["generated_tokens"] == 6 and timing["prompt_tokens"] == 2


# ---------------------------------------------------------------------------------------------------------
# gates: the control run's realized step count is the expectation
# ---------------------------------------------------------------------------------------------------------


def test_expected_steps_prefers_the_control_run(tmp_path):
    from verity_vllm.check import gates as G

    run = tmp_path
    (run / G.ARTIFACTS["run"]).write_text(json.dumps({"workload": {"requests": [{"request_id": "x", "prompt_len": 5, "max_tokens": 16}]}}))
    ra = G.RunArtifacts(run)
    assert ra.expected_steps() == 16, "fallback: max(max_tokens)"
    ctl = run / G.ARTIFACTS["control_tokens"]
    ctl.parent.mkdir(parents=True, exist_ok=True)
    ctl.write_text(json.dumps({"requests": [{"request_id": "x", "prompt_token_ids": [1] * 5, "max_tokens": 16}], "timing": {"engine_steps": 21}}))
    assert G.RunArtifacts(run).expected_steps() == 21, "arrivals / chunking / EOS: the control run's realized steps"
