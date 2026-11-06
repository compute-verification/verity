"""Chunked-prefill column: the `--variant chunked=<N>` workload variant, its V1 schedule simulation, and the
prefill-chunk / token-step attribution of `verity_vllm.engine.chunks` on stub engines with vLLM V1's shape (one-iteration
output lag, no output at a partial-prefill step, token budget shared running-first).  Torch-free."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from verity_vllm.engine import arrivals
from verity_vllm.engine import chunks as CA
from verity_vllm.pipeline import coverage_workloads as gw
from tests.pipeline.test_coverage_workloads import _LaggedStubEngine, _Out

REPO = Path(__file__).resolve().parents[2]
PROF = REPO / "workloads"
# which `run_requests_arrivals` is present: the original loop measures the output lag from the first returned output (wrong under
# chunked prefill; `attribute()` corrects it), the current one derives the forwards from the scheduler / clamps
# the lag and exposes `scheduler_of`; the attribution's RESULT is the same on both, only the `corrections` record differs
_OLD_LOOP = not hasattr(arrivals, "scheduler_of")


# ---------------------------------------------------------------------------------------------------------
# generator: variant parsing + the chunked schedule simulation + the generated cell
# ---------------------------------------------------------------------------------------------------------


def test_parse_variant_accepts_sampling_names_and_feature_pairs():
    assert gw.parse_variant("greedy") == ("greedy", {})
    assert gw.parse_variant("t1") == ("t1", {})
    assert gw.parse_variant("chunked=64") == ("greedy", {"chunked": 64})
    assert gw.parse_variant("t1,chunked=32") == ("t1", {"chunked": 32})
    assert gw.variant_suffix("greedy", {}) == "" and gw.variant_suffix("t1", {}) == "_t1"
    assert gw.variant_suffix("greedy", {"chunked": 64}) == "_ck64" and gw.variant_suffix("t1", {"chunked": 64}) == "_t1_ck64"
    with pytest.raises(KeyError):
        gw.parse_variant("bogus")
    with pytest.raises(KeyError):
        gw.parse_variant("unknown=1")


def test_chunked_schedule_of_the_b1_c256_draw():
    d = gw.draw_lengths("B1", "C256", 0)
    assert [(r["prompt_len"], r["max_tokens"]) for r in d] == [(155, 15)]
    sim = gw.simulate_schedule_chunked(d, 64)
    assert sim["prefill_chunks"] == [[64, 64, 27]] and sim["partial_prefill_steps"] == [[0, 1]]
    assert sim["first_token_step"] == [2] and sim["finish_step"] == [16]
    assert sim["expected_engine_steps"] == 17 == 3 + 15 - 1          # ceil(155/64) chunks + max_tokens - 1 decodes
    assert sim["steps_with_partial_prefill"] == 2 and sim["peak_concurrent"] == 1
    # an unchunking budget reproduces the plain simulation
    assert gw.simulate_schedule_chunked(d, 2048)["expected_engine_steps"] == gw.simulate_schedule(d)["expected_engine_steps"] == 15


def test_chunked_schedule_reproduces_the_ov_batch_chunked_200p_shape():
    """b0_chunked_200p (a real run): request 0 (7 tokens, 8 out) decodes while request 1's 200-token prompt is prefilled
    in chunks 57 + 63 + 63 + 17; request 1's first token at step 3; 15 engine steps."""
    reqs = [{"prompt_len": 7, "max_tokens": 8, "arrive_step": 0}, {"prompt_len": 200, "max_tokens": 12, "arrive_step": 0}]
    sim = gw.simulate_schedule_chunked(reqs, 64)
    assert sim["prefill_chunks"] == [[7], [57, 63, 63, 17]] and sim["first_token_step"] == [0, 3]
    assert sim["finish_step"] == [7, 14] and sim["expected_engine_steps"] == 15
    assert sim["steps_with_mixed_prefill_decode"] == 3 and sim["peak_concurrent"] == 2


def test_chunked_schedule_makes_a_request_wait_when_the_budget_is_spent():
    reqs = [{"prompt_len": 100, "max_tokens": 4, "arrive_step": 0}, {"prompt_len": 100, "max_tokens": 4, "arrive_step": 0}]
    sim = gw.simulate_schedule_chunked(reqs, 64)
    assert sim["prefill_chunks"] == [[64, 36], [28, 63, 9]]          # request 1 gets nothing at step 0
    assert sim["first_token_step"] == [1, 3] and sim["expected_engine_steps"] == 7
    # B8-mixed arrivals under a small budget: never stalls, every request finishes
    d = gw.draw_lengths("B8-mixed", "C256", 0)
    sim8 = gw.simulate_schedule_chunked(d, 64)
    assert sim8["expected_engine_steps"] >= gw.simulate_schedule(d)["expected_engine_steps"]
    assert all(f is not None for f in sim8["finish_step"]) and max(len(c) for c in sim8["prefill_chunks"]) >= 2


def test_generated_ck64_cell_is_the_base_cell_with_a_chunking_budget():
    base, ck = PROF / "workload_cov_b0_b1_c256.json", PROF / "workload_cov_b0_b1_c256_ck64.json"
    if not ck.exists():
        pytest.skip("workload_cov_b0_b1_c256_ck64.json not generated in this checkout")
    a, b = json.loads(base.read_text()), json.loads(ck.read_text())
    key = lambda r: (r["request_id"], r["prompt_token_ids"], r["max_tokens"], r["arrive_step"])  # noqa: E731
    assert [key(r) for r in a["requests"]] == [key(r) for r in b["requests"]], "same request ids, prompts, lengths, arrivals"
    assert b["engine"]["args"] == {"max_num_batched_tokens": 64, "enable_chunked_prefill": True}
    assert b["feature_variant"] == {"chunked": 64} and b["sampling"]["variant"] == "greedy"
    assert b["expected_engine_steps"] == 17 and b["schedule_if_full_length"]["prefill_chunks"] == [[64, 64, 27]]
    assert b["expected_engine_steps"] >= a["expected_engine_steps"], "the snapshot-step range (run_config: range(expected)) must cover the run"
    assert b["case"] == a["case"] == "B0" and b["model"] == a["model"]


# ---------------------------------------------------------------------------------------------------------
# a stub engine with vLLM V1's chunked-prefill shape
# ---------------------------------------------------------------------------------------------------------


class _Req:
    def __init__(self, rid: str, plen: int, max_tokens: int) -> None:
        self.request_id = rid
        self.num_prompt_tokens = plen
        self.num_computed_tokens = 0
        self.max_tokens = max_tokens
        self.toks: list[int] = []


class _ChunkedLaggedStubEngine:
    """vLLM V1 shape: one token budget per forward, RUNNING requests first (1 decode token, or the remaining prompt of a prefill in
    progress), then WAITING requests in arrival order with the remaining budget; the scheduler advances `num_computed_tokens` when
    it schedules (before the forward); a request samples its first token at the forward that completes its prompt and yields NO
    output at a partial-prefill forward; `step()` returns the previous forward's outputs (lag 1); a finished request leaves the
    scheduler's `requests` when its finish output is returned.  Exposes `engine_core.scheduler.requests` like the in-process core."""

    def __init__(self, budget: int, max_num_seqs: int = 8, randomize_ids: bool = True) -> None:
        self.budget = budget
        self.max_num_seqs = max_num_seqs
        self.scheduler = types.SimpleNamespace(requests={}, waiting=[], running=[])
        self.engine_core = types.SimpleNamespace(scheduler=self.scheduler)
        self.buffered: list = []
        self.unfinished_client: set[str] = set()
        self.forwards = 0
        self.batches: list[dict[str, int]] = []
        # the pinned vLLM (input_processor.assign_request_id) keys the scheduler by f"{submitted}-{8 hex}" and returns outputs under
        # the submitted id (seen in run 0917T213719-880581: `...-e8d6-98207b44`); the stub does the same unless told not to
        self.randomize_ids = randomize_ids
        self._n_added = 0

    def add_request(self, *, request_id: str, prompt: dict, params) -> None:
        self._n_added += 1
        internal = f"{request_id}-{self._n_added * 0x1234567:08x}"[:len(request_id) + 9] if self.randomize_ids else request_id
        r = _Req(internal, len(prompt["prompt_token_ids"]), params.max_tokens)
        r.external_id = request_id
        self.scheduler.requests[internal] = r
        self.scheduler.waiting.append(r)
        self.unfinished_client.add(request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self.unfinished_client)

    def step(self):
        out = self.buffered
        s = self.scheduler
        left = self.budget
        sched: list[tuple[_Req, int]] = []
        for r in list(s.running):
            if left <= 0:
                break
            t = min(r.num_prompt_tokens - r.num_computed_tokens, left) if r.num_computed_tokens < r.num_prompt_tokens else 1
            sched.append((r, t))
            left -= t
        for r in list(s.waiting):
            if left <= 0 or len(s.running) >= self.max_num_seqs:
                break
            t = min(r.num_prompt_tokens, left)
            s.waiting.remove(r)
            s.running.append(r)
            sched.append((r, t))
            left -= t
        new = []
        if sched:
            self.forwards += 1
            self.batches.append({r.external_id: t for r, t in sched})
            for r, t in sched:
                before = r.num_computed_tokens
                r.num_computed_tokens += t                         # the scheduler advances it at schedule time
                if before < r.num_prompt_tokens and r.num_computed_tokens < r.num_prompt_tokens:
                    continue                                       # partial prefill: sampled and discarded, no output
                r.toks.append(len(r.toks))
                done = len(r.toks) >= r.max_tokens
                new.append(_Out(r.external_id, r.toks, done))     # outputs carry the SUBMITTED id (output_processor)
                if done:
                    s.running.remove(r)
        self.buffered = new
        for o in out:
            if o.finished:
                self.unfinished_client.discard(o.request_id)
                for k, r in list(s.requests.items()):
                    if r.external_id == o.request_id:
                        s.requests.pop(k)
        return out


def _R(rid: str, plen: int, n: int, a: int = 0):
    return types.SimpleNamespace(request_id=rid, prompt_token_ids=list(range(plen)), max_tokens=n, arrive_step=a, _order=0)


def _run(eng, reqs, *, with_scheduler: bool = True):
    llm = types.SimpleNamespace(llm_engine=eng)
    sched = CA.scheduler_of(llm) if with_scheduler else None
    col = CA.Collector(sched)
    outs, timing = arrivals.run_requests_arrivals(llm, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens), on_step=col.on_step)
    rep = CA.attribute(timing, col, {r.request_id: len(r.prompt_token_ids) for r in reqs})
    return outs, timing, rep


def test_scheduler_of_finds_the_in_process_scheduler_or_none():
    eng = _ChunkedLaggedStubEngine(64)
    assert CA.scheduler_of(types.SimpleNamespace(llm_engine=eng)) is eng.scheduler
    nested = types.SimpleNamespace(llm_engine=types.SimpleNamespace(engine_core=types.SimpleNamespace(engine_core=eng.engine_core)))
    assert CA.scheduler_of(nested) is eng.scheduler
    assert CA.scheduler_of(types.SimpleNamespace(llm_engine=_LaggedStubEngine())) is None
    assert CA.scheduler_of(types.SimpleNamespace()) is None


def test_b1_chunked_prefill_three_chunks_and_the_lag_correction():
    """The window's cell: prompt 155, max_tokens 15, budget 64 -> forwards 0/1/2 are chunks 64/64/27, token 0 comes from forward 2,
    17 forwards in 18 iterations.  The arrival loop alone mis-measures the lag (3: nothing returned at iterations 0-2) and reports 15
    forwards; the scheduler-state source repairs engine_steps / lag / finish_step and records the loop's values."""
    eng = _ChunkedLaggedStubEngine(64)
    outs, timing, rep = _run(eng, [_R("x", 155, 15)])
    assert eng.forwards == 17 and eng.batches[:3] == [{"x": 64}, {"x": 64}, {"x": 27}] and all(b == {"x": 1} for b in eng.batches[3:])
    assert rep["source"] == "scheduler-state" and rep["consistent"], rep["problems"]
    assert timing["engine_steps"] == 17 and timing["step_iterations"] == 18 and timing["output_lag_steps"] == 1
    rec = timing["requests"]["x"]
    assert rec["prefill_steps"] == [0, 1, 2] and rec["n_prefill_chunks"] == 3 and rec["partial_prefill_steps"] == [0, 1]
    assert [c["tokens"] for c in rec["prefill_chunks"]] == [64, 64, 27]
    assert [c["computed_before"] for c in rec["prefill_chunks"]] == [0, 64, 128]
    assert [c["prompt_complete"] for c in rec["prefill_chunks"]] == [False, False, True]
    assert rec["first_token_step"] == 2 and rec["token_steps"] == list(range(2, 17)) and rec["n_generated"] == 15
    assert (rec["first_step"], rec["finish_step"], rec["arrive_step_realized"]) == (0, 16, 0)
    assert rep["steps_scheduled_tokens"][:3] == [{"x": 64}, {"x": 64}, {"x": 27}] and len(rep["steps_scheduled_tokens"]) == 17
    assert rep["chunked_prefill_observed"] and rep["max_prefill_chunks"] == 3 and rep["partial_prefill_steps_total"] == 2
    c = rep["corrections"]
    if _OLD_LOOP:
        # the loop's own inference is on record: lag 3, 15 forwards, finish at 14 (what the control run 0917T213719-880581 said)
        assert c["loop"] == {"engine_steps": 15, "output_lag_steps": 3, "requests": {"x": {"arrive_step_realized": 0, "first_step": 0, "finish_step": 14}}}
        assert c["corrected"] == {"engine_steps": 17, "output_lag_steps": 1}
    else:
        assert c is None, c                    # the current loop derives the forwards itself and agrees: nothing to correct
    # every existing key is still there
    for k in ("arrive_step", "arrive_step_realized", "first_step", "finish_step", "finish_reason", "n_generated", "admit_iter", "finish_iter"):
        assert k in rec


def test_outputs_fallback_without_a_scheduler_infers_the_chunks_from_first_step_to_first_token():
    eng = _ChunkedLaggedStubEngine(64)
    outs, timing, rep = _run(eng, [_R("x", 155, 15)], with_scheduler=False)
    assert rep["source"] == "outputs" and rep["corrections"] is None
    if _OLD_LOOP:
        # nothing to correct with: the loop's mis-measured lag stands, and the record says what it can (documented limitation)
        assert timing["engine_steps"] == 15 and timing["output_lag_steps"] == 3
    else:
        assert timing["engine_steps"] == 17 and timing["output_lag_steps"] == 1    # the loop clamps / reads the scheduler itself
    rec = timing["requests"]["x"]
    assert rec["prefill_chunks"] is None and rec["first_token_step"] == rec["prefill_steps"][-1] and rec["token_steps"][0] == rec["first_token_step"]
    assert len(rec["token_steps"]) == 15


def test_two_requests_reproduce_the_chunked_200p_record():
    eng = _ChunkedLaggedStubEngine(64)
    outs, timing, rep = _run(eng, [_R("r0", 7, 8), _R("r1", 200, 12)])
    assert rep["consistent"], rep["problems"]
    assert timing["engine_steps"] == 15 and rep["corrections"] is None            # r0 serves a token at forward 0: the lag was right
    r0, r1 = timing["requests"]["r0"], timing["requests"]["r1"]
    assert r0["prefill_steps"] == [0] and r0["n_prefill_chunks"] == 1 and r0["partial_prefill_steps"] == [] and r0["token_steps"] == list(range(8))
    assert [c["tokens"] for c in r1["prefill_chunks"]] == [57, 63, 63, 17] and r1["prefill_steps"] == [0, 1, 2, 3]
    assert r1["partial_prefill_steps"] == [0, 1, 2] and r1["first_token_step"] == 3 and r1["token_steps"] == list(range(3, 15))
    assert r1["finish_step"] == 14 and r0["finish_step"] == 7
    assert rep["steps_scheduled_tokens"][0] == {"r0": 7, "r1": 57} and rep["steps_scheduled_tokens"][1] == {"r0": 1, "r1": 63}


def test_a_waiting_request_is_not_credited_with_a_chunk_by_the_scheduler_source():
    reqs = [_R("a", 100, 4), _R("b", 100, 4)]
    eng = _ChunkedLaggedStubEngine(64)
    outs, timing, rep = _run(eng, reqs)
    assert rep["consistent"], rep["problems"]
    a, b = timing["requests"]["a"], timing["requests"]["b"]
    assert [c["tokens"] for c in a["prefill_chunks"]] == [64, 36] and a["prefill_steps"] == [0, 1]
    assert [c["tokens"] for c in b["prefill_chunks"]] == [28, 63, 9] and b["prefill_steps"] == [1, 2, 3], "b waited at forward 0"
    assert b["first_step"] == 0, "the arrival loop's first_step is the admission forward, not the first chunk (kept as is)"
    assert timing["engine_steps"] == 7
    # the outputs-only inference cannot see the wait (and here also mis-measures the lag: nothing is returned at iterations 0-1) --
    # the documented limitation of that source; it starts b's chunks at its admission forward and has nothing to correct with
    eng2 = _ChunkedLaggedStubEngine(64)
    _, timing2, rep2 = _run(eng2, [_R("a", 100, 4), _R("b", 100, 4)], with_scheduler=False)
    b2 = timing2["requests"]["b"]
    assert rep2["source"] == "outputs" and rep2["corrections"] is None and b2["prefill_chunks"] is None
    assert b2["prefill_steps"][0] == b2["first_step"] == 0 and b2["prefill_steps"] != [1, 2, 3]


def test_unchunked_lagged_engine_records_one_chunk_per_request_and_consecutive_token_steps():
    eng = _LaggedStubEngine()
    llm = types.SimpleNamespace(llm_engine=eng)
    col = CA.Collector(CA.scheduler_of(llm))
    reqs = [_R("r0", 3, 5, 0), _R("r1", 3, 3, 2), _R("r3", 3, 9, 9)]
    outs, timing = arrivals.run_requests_arrivals(llm, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens), on_step=col.on_step)
    before = json.dumps(timing, sort_keys=True, default=str)
    rep = CA.attribute(timing, col, {r.request_id: 3 for r in reqs})
    assert rep["source"] == "outputs" and rep["consistent"] and not rep["chunked_prefill_observed"] and rep["corrections"] is None
    for rid, rec in timing["requests"].items():
        assert rec["prefill_steps"] == [rec["first_step"]] and rec["n_prefill_chunks"] == 1 and rec["partial_prefill_steps"] == []
        assert rec["token_steps"] == list(range(rec["first_step"], rec["finish_step"] + 1)) and rec["first_token_step"] == rec["first_step"]
    # additive: every pre-existing value is unchanged
    after = json.loads(json.dumps(timing, sort_keys=True, default=str))
    for k, v in json.loads(before).items():
        if k == "requests":
            for rid, rec in v.items():
                assert {kk: after["requests"][rid][kk] for kk in rec} == rec
        else:
            assert after[k] == v


def test_an_idle_gap_between_arrivals_does_not_inflate_the_corrected_lag():
    """With staggered arrivals the loop drains the finished batch (a no-forward iteration), then admits the
    next arrival at once; `iterations - forwards` counts every drain and would say lag 2 here.  The lag is per request
    finish iteration - last advancing iteration (1), and first/finish steps come from the advancing forwards themselves."""
    eng = _ChunkedLaggedStubEngine(64)
    _, timing, rep = _run(eng, [_R("a", 7, 3, 0), _R("b", 155, 4, 20)])
    # a: forwards 0-2 (prefill + 2 decodes), finish returned at iteration 3 (drain); b admitted in the idle gap: chunks at
    # iterations 4-6 = forwards 3-5, decodes 6-8, finish returned at iteration 10 -> 11 iterations, 9 forwards, lag 1
    assert timing["step_iterations"] == 11 and eng.forwards == 9
    assert timing["engine_steps"] == 9 and timing["output_lag_steps"] == 1
    assert rep["consistent"] and rep["problems"] == [], rep["problems"]
    a, b = timing["requests"]["a"], timing["requests"]["b"]
    assert (a["first_step"], a["finish_step"], a["token_steps"]) == (0, 2, [0, 1, 2])
    assert (b["arrive_step_realized"], b["first_step"], b["finish_step"]) == (3, 3, 8)
    assert b["prefill_steps"] == [3, 4, 5] and b["partial_prefill_steps"] == [3, 4] and b["token_steps"] == [5, 6, 7, 8]
    if _OLD_LOOP:
        assert rep["corrections"]["loop"]["engine_steps"] == 7        # the loop missed b's two partial chunks
    else:
        assert rep["corrections"] is None, rep["corrections"]


def test_external_id_strips_vllms_internal_suffix_only_for_ids_we_know():
    known = {"97f3c5d4-16d8-5c2e-9ead-2d9396d2e8d6", "b"}
    assert CA.external_id("97f3c5d4-16d8-5c2e-9ead-2d9396d2e8d6-98207b44", known) == "97f3c5d4-16d8-5c2e-9ead-2d9396d2e8d6"
    assert CA.external_id("b-0a1b2c3d", known) == "b" and CA.external_id("b", known) == "b"
    assert CA.external_id("1_b-0a1b2c3d", known) == "1_b"                    # parallel-sampling child of a known parent
    assert CA.external_id("zzz-0a1b2c3d", known) == "zzz-0a1b2c3d"           # unknown: never guessed
    assert CA.external_id("b-0a1b2c3G", known) == "b-0a1b2c3G"               # not a hex suffix


def test_randomized_scheduler_ids_are_mapped_back_to_the_submitted_ids():
    """Run 0917T213719-880581: the scheduler keyed the request as `<id>-98207b44`, the outputs as `<id>`; the per-forward table was
    right but every per-request chunk fell back to the outputs inference ("no prefill chunk seen in the scheduler state")."""
    eng = _ChunkedLaggedStubEngine(64, randomize_ids=True)
    rid = "97f3c5d4-16d8-5c2e-9ead-2d9396d2e8d6"
    _, timing, rep = _run(eng, [_R(rid, 155, 15)])
    assert list(eng.scheduler.requests) == [] and eng.batches[0] == {rid: 64}
    assert rep["source"] == "scheduler-state" and rep["consistent"] and rep["problems"] == []
    assert rep["steps_scheduled_tokens"][:4] == [{rid: 64}, {rid: 64}, {rid: 27}, {rid: 1}]
    assert rep["internal_request_ids"] == {rid: f"{rid}-{0x1234567:08x}"}
    rec = timing["requests"][rid]
    assert rec["prefill_steps"] == [0, 1, 2] and [c["tokens"] for c in rec["prefill_chunks"]] == [64, 64, 27]
    assert rec["partial_prefill_steps"] == [0, 1] and rec["first_token_step"] == 2 and rec["token_steps"] == list(range(2, 17))
    # without randomization (VLLM_DISABLE_REQUEST_ID_RANDOMIZATION) the record is identical and no map is reported
    _, timing2, rep2 = _run(_ChunkedLaggedStubEngine(64, randomize_ids=False), [_R(rid, 155, 15)])
    assert rep2["internal_request_ids"] is None and timing2["requests"][rid] == rec


# ---------------------------------------------------------------------------------------------------------
# the adapter wiring (needs torch + vllm importable: runs in the pod's venv-cu129, skipped on the laptop)
# ---------------------------------------------------------------------------------------------------------


def test_vllm_adapter_run_requests_carries_the_chunk_record():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from verity_vllm.engine import vllm_adapter as va

    eng = _ChunkedLaggedStubEngine(64)
    engine = types.SimpleNamespace(llm=types.SimpleNamespace(llm_engine=eng))
    reqs = [{"request_id": "x", "prompt_token_ids": list(range(155)), "max_tokens": 15, "arrive_step": 0}]
    tokens, timing = va.run_requests(engine, reqs, {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "n": 1})
    assert tokens["x"] == list(range(15)) and timing["engine_steps"] == 17
    rec = timing["requests"]["x"]
    assert rec["prefill_steps"] == [0, 1, 2] and [c["tokens"] for c in rec["prefill_chunks"]] == [64, 64, 27]
    assert timing["chunk_attribution"]["source"] == "scheduler-state" and timing["chunk_attribution"]["consistent"]


# ----------------------------------------------------------------------------------------------------------------------------------
# async_scheduling=False (LAG 0): step() returns THIS forward's outputs and frees the finishing request inside
# step(), BEFORE the collector's post-step scheduler snapshot -- its last forward is invisible to the schedule deltas, last_adv one early,
# the derived lag 1 for a true 0.  exec r17_2216 (the first sync record): attribute() rewrote engine_steps 11 -> 10, output_lag_steps
# 0 -> 1, finish_step 10 -> 9 and GM-01 G6 failed the record against its own tokens.  The loop's values must stand.
class _SyncStubEngine(_ChunkedLaggedStubEngine):
    """async_scheduling=False: no output buffering (the forward's outputs return from the same step()); a finished request leaves
    `scheduler.requests` inside that step()."""

    def step(self):
        s = self.scheduler
        left = self.budget
        sched: list[tuple[_Req, int]] = []
        for r in list(s.running):
            if left <= 0:
                break
            t = min(r.num_prompt_tokens - r.num_computed_tokens, left) if r.num_computed_tokens < r.num_prompt_tokens else 1
            sched.append((r, t))
            left -= t
        for r in list(s.waiting):
            if left <= 0 or len(s.running) >= self.max_num_seqs:
                break
            t = min(r.num_prompt_tokens, left)
            s.waiting.remove(r)
            s.running.append(r)
            sched.append((r, t))
            left -= t
        out = []
        if sched:
            self.forwards += 1
            self.batches.append({r.external_id: t for r, t in sched})
            for r, t in sched:
                before = r.num_computed_tokens
                r.num_computed_tokens += t
                if before < r.num_prompt_tokens and r.num_computed_tokens < r.num_prompt_tokens:
                    continue
                r.toks.append(len(r.toks))
                done = len(r.toks) >= r.max_tokens
                out.append(_Out(r.external_id, r.toks, done))
                if done:
                    s.running.remove(r)
                    self.unfinished_client.discard(r.external_id)
                    for k, rr in list(s.requests.items()):             # freed INSIDE step(): invisible to the post-step snapshot
                        if rr is r:
                            s.requests.pop(k)
        return out


def _run_sync(reqs, *, resolved_lag):
    eng = _SyncStubEngine(64)
    llm = types.SimpleNamespace(llm_engine=eng)
    col = CA.Collector(CA.scheduler_of(llm))
    outs, timing = arrivals.run_requests_arrivals(llm, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens), on_step=col.on_step)
    loop = {"engine_steps": timing["engine_steps"], "output_lag_steps": timing["output_lag_steps"],
            "requests": {rid: {k: rec.get(k) for k in ("arrive_step_realized", "first_step", "finish_step")} for rid, rec in timing["requests"].items()}}
    rep = CA.attribute(timing, col, {r.request_id: len(r.prompt_token_ids) for r in reqs}, resolved_lag=resolved_lag)
    return eng, timing, rep, loop


def test_f_r17_16_a_sync_record_keeps_the_loops_finish_step_when_the_derived_lag_contradicts_the_resolved_lag():
    eng, timing, rep, loop = _run_sync([_R("r0", 5, 11)], resolved_lag=0)
    # the loop measured the sync engine right: 11 forwards, lag 0, the EOS forward = 10
    assert eng.forwards == 11 and loop == {"engine_steps": 11, "output_lag_steps": 0, "requests": {"r0": {"arrive_step_realized": 0, "first_step": 0, "finish_step": 10}}}
    # the schedule deltas missed the last forward (freed inside step()): derived lag 1, 10 forwards, finish 9 -- NOT applied
    c = rep["corrections"]
    assert c is not None and c["applied"] is False and c["corrected"]["output_lag_steps"] == 1 and c["corrected"]["engine_steps"] == 10
    assert c["corrected"]["requests"]["r0"]["finish_step"] == 9 and "NOT APPLIED" in c["reason"]
    assert timing["engine_steps"] == 11 and timing["output_lag_steps"] == 0 and timing["requests"]["r0"]["finish_step"] == 10
    assert not rep["consistent"] and any("contradicts the resolved async_scheduling LAG 0" in p and p.endswith("finish_step values stand") for p in rep["problems"])
    # the served tokens are attributed with the loop's forward map: 11 token-producing forwards 0..10, the last = finish_step
    assert timing["requests"]["r0"]["token_steps"] == list(range(11)) and timing["requests"]["r0"]["first_token_step"] == 0


def test_f_r17_16_without_a_resolved_lag_the_same_record_reverts_the_correction_that_fails_its_own_check():
    eng, timing, rep, loop = _run_sync([_R("r0", 5, 11)], resolved_lag=None)
    c = rep["corrections"]
    assert c is not None and c["applied"] is False and "REVERTED" in c["reason"]
    assert timing["engine_steps"] == 11 and timing["output_lag_steps"] == 0 and timing["requests"]["r0"]["finish_step"] == 10
    assert not rep["consistent"] and rep["problems"][0].startswith("scheduler-state correction reverted") and rep["problems"][0].endswith("step fields stand")


def test_f_r17_16_an_async_record_with_the_matching_resolved_lag_is_unchanged():
    eng = _ChunkedLaggedStubEngine(64)
    llm = types.SimpleNamespace(llm_engine=eng)
    col = CA.Collector(CA.scheduler_of(llm))
    reqs = [_R("x", 155, 15)]
    outs, timing = arrivals.run_requests_arrivals(llm, reqs, lambda r: types.SimpleNamespace(max_tokens=r.max_tokens), on_step=col.on_step)
    rep = CA.attribute(timing, col, {"x": 155}, resolved_lag=1)
    assert rep["consistent"], rep["problems"]
    assert timing["engine_steps"] == 17 and timing["output_lag_steps"] == 1 and timing["requests"]["x"]["finish_step"] == 16
    assert rep["corrections"] is None or rep["corrections"]["applied"] is True
