"""Synthetic tests of the AOT builder's predictor / hint encoder (lane C-builder, R8 2026-09-18).

No corpus, no IR: a synthetic φ with three requests exercises admission (`max_num_seqs`), chunked prefill
(`max_num_batched_tokens`), the input-preparation row order (predictor v2), the served rule, and the predict-then-correct
closure: `encode_hints(φ, observed)` is empty when observed == predicted, and when it is not, feeding the hints back into
`predict_schedule` reproduces the observed schedule exactly (steps overrides, served lengths, discarded samples).
"""
from __future__ import annotations

import json

import pytest

from verity_capture.experimental.aot.encode import Observed, encode_hints
from verity_capture.experimental.aot.hints import HINTS_SCHEMA, Hints, ServedLength
from verity_capture.experimental.aot.phi import Phi
from verity_capture.experimental.aot.predict import PREDICTOR_ID, PREDICTORS, StepPlan, predict_schedule


def _phi(prompt_lens=(5, 7, 3), max_tokens=(3, 3, 3), arrive=(0, 0, 0), max_num_seqs=8, budget=1 << 20, chunked=True) -> Phi:
    reqs = [{"request_id": f"r{i}", "prompt_token_ids": list(range(100, 100 + n)), "max_tokens": m, "seed": None, "n": 1, "arrive_step": a}
            for i, (n, m, a) in enumerate(zip(prompt_lens, max_tokens, arrive))]
    return Phi(config={"H": 8}, profile={"module": "synthetic", "id": "0" * 12, "pins": {}},
               engine={"block_size": 16, "max_num_seqs": max_num_seqs, "max_num_batched_tokens": budget, "enable_chunked_prefill": chunked,
                       "enable_prefix_caching": False},
               workload={"requests": reqs, "sampling": {"temperature": 0.0}})


def _observed(plans: list[StepPlan]) -> Observed:
    return Observed(plans=[StepPlan(p.step, list(p.rows), list(p.logits_rows), list(p.served), p.phase) for p in plans],
                    request_ids=[], ended=True)


def test_hints_json_roundtrip_and_empty_default():
    h = Hints(predictor=PREDICTOR_ID)
    d = h.to_json()
    assert d["schema"] == HINTS_SCHEMA and d["served"] == [] and d["steps"] == [] and d["discarded"] == [] and d["extensions"] == {}
    assert Hints.from_json(json.loads(json.dumps(d))) == h
    assert h.naive_bits(max_tokens=15, n_requests=1, max_position=200) == 0


def test_predictor_single_request_is_one_prefill_plus_served_minus_one_decodes():
    phi = _phi(prompt_lens=(155,), max_tokens=(15,))
    plans = predict_schedule(phi)
    assert len(plans) == 15 and plans[0].phase == "prefill" and all(p.phase == "decode" for p in plans[1:])
    assert plans[0].rows == [(0, i) for i in range(155)] and plans[0].logits_rows == [(0, 154)] and plans[0].served == [True]
    assert plans[-1].rows == [(0, 168)]
    assert sum(sum(p.served) for p in plans) == 15


def test_predictor_v2_sorts_mixed_step_by_scheduled_tokens_and_v1_does_not():
    # request 0 runs; requests 1 (7 tokens) and 2 (3 tokens) arrive at step 1 together
    phi = _phi(prompt_lens=(5, 7, 3), max_tokens=(4, 2, 2), arrive=(0, 1, 1))
    v2 = predict_schedule(phi)[1]
    assert v2.phase == "mixed"
    groups = [r for r, _ in v2.rows]
    # v2: decode row (1 token) first, then the 3-token prompt, then the 7-token prompt
    assert groups == [0] + [2] * 3 + [1] * 7
    v1 = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v1"))[1]
    assert [r for r, _ in v1.rows] == [0] + [1] * 7 + [2] * 3          # scheduler order: admission order
    assert set(PREDICTORS) == {"dense-eager-fcfs/v1", "dense-eager-fcfs/v2", "dense-eager-fcfs/x-legacy-batch", "dense-eager-fcfs/v3-kv", "dense-eager-fcfs/v4-pool", "dense-eager-fcfs/v5-seat"}
    with pytest.raises(ValueError):
        predict_schedule(phi, Hints(predictor="no-such-predictor"))


def test_predictor_admission_limit_and_chunked_prefill():
    phi = _phi(prompt_lens=(10, 10, 10), max_tokens=(2, 2, 2), max_num_seqs=2, budget=6)
    plans = predict_schedule(phi)
    # step 0: two requests admitted (max_num_seqs), 6-token budget → request 0 gets 6, request 1 nothing yet
    assert [r for r, _ in plans[0].rows] == [0] * 6 and plans[0].served == [False]     # partial chunk: sample discarded
    # step 1: request 0's remaining 4 tokens + request 1's first 2 (budget 6); the runner sort puts the 2-token group first.
    # (This pins the modelled rule; the corpus had 0 partial-prefill chunks, so this case is unmeasured against vLLM.)
    assert [r for r, _ in plans[1].rows] == [1, 1, 0, 0, 0, 0]
    assert plans[1].logits_rows == [(1, 1), (0, 9)] and plans[1].served == [False, True]
    served = {}
    for p in plans:
        for (r, _), s in zip(p.logits_rows, p.served):
            served[r] = served.get(r, 0) + int(s)
    assert served == {0: 2, 1: 2, 2: 2}


def test_encode_is_empty_when_observed_equals_predicted():
    phi = _phi(prompt_lens=(5, 7, 3), max_tokens=(4, 2, 2), arrive=(0, 1, 1))
    hints, rep = encode_hints(phi, _observed(predict_schedule(phi)))
    assert hints == Hints(predictor=PREDICTOR_ID) and rep["corrections"] == []


def test_encode_served_hint_when_a_request_stops_early():
    phi = _phi(prompt_lens=(5,), max_tokens=(6,))
    obs = predict_schedule(phi, Hints(predictor=PREDICTOR_ID, served=[ServedLength(0, 4)]))
    assert len(obs) == 4
    hints, rep = encode_hints(phi, _observed(obs))
    assert hints.served == [ServedLength(0, 4)] and hints.steps == [] and hints.discarded == []
    assert [c["kind"] for c in rep["corrections"]] == ["served"]
    again = predict_schedule(phi, hints)
    assert [(p.rows, p.logits_rows, p.served) for p in again] == [(p.rows, p.logits_rows, p.served) for p in obs]


def test_encode_served_hint_with_admission_shift():
    # three requests, room for two: request 0 stopping early (after 2 of 6) frees the slot one step sooner, so request 2 is
    # admitted earlier — the whole schedule shifts, yet the residual is exactly one served entry
    phi = _phi(prompt_lens=(5, 5, 5), max_tokens=(6, 3, 3), max_num_seqs=2)
    obs = predict_schedule(phi, Hints(predictor=PREDICTOR_ID, served=[ServedLength(0, 2)]))
    base = predict_schedule(phi)
    assert len(obs) < len(base)
    hints, rep = encode_hints(phi, _observed(obs))
    assert hints.served == [ServedLength(0, 2)] and hints.steps == [] and hints.discarded == []
    again = predict_schedule(phi, hints)
    assert [(p.rows, p.logits_rows, p.served) for p in again] == [(p.rows, p.logits_rows, p.served) for p in obs]


def test_encode_steps_hints_express_a_preemption_like_gap_and_recompute():
    # an engine that drops request 1 for one step and then re-feeds its prompt + generated tokens from position 0
    phi = _phi(prompt_lens=(4, 3), max_tokens=(5, 4))
    base = predict_schedule(phi)
    obs = [StepPlan(p.step, list(p.rows), list(p.logits_rows), list(p.served), p.phase) for p in base]
    gap = 2
    obs[gap] = StepPlan(gap, [rp for rp in obs[gap].rows if rp[0] != 1], [], [])
    obs[gap].logits_rows = obs[gap].default_logits_rows()
    obs[gap].served = [True] * len(obs[gap].logits_rows)
    # recompute step: request 1 re-feeds positions 0..4 (3 prompt + 2 generated so far) after request 0's decode row
    recompute = [(0, obs[gap + 1].rows[0][1])] + [(1, q) for q in range(5)]
    obs[gap + 1] = StepPlan(gap + 1, recompute, [], [])
    obs[gap + 1].logits_rows = obs[gap + 1].default_logits_rows()
    obs[gap + 1].served = [True, True]
    # request 1 lost one step, so it still needs its 4th token: one more decode row at position 5 in the last step
    assert len(obs) == gap + 3
    last = obs[gap + 2]
    obs[gap + 2] = StepPlan(last.step, list(last.rows) + [(1, 5)], [], [])
    obs[gap + 2].logits_rows = obs[gap + 2].default_logits_rows()
    obs[gap + 2].served = [True, True]
    hints, rep = encode_hints(phi, _observed(obs))
    assert hints.served == [] and {h.step for h in hints.steps} == {gap, gap + 1}      # the tail re-synchronises by itself
    assert rep["corrections"][1]["observed_runs"] == [{"request": 0, "first": 6, "count": 1}, {"request": 1, "first": 0, "count": 5}]
    again = predict_schedule(phi, hints)
    assert [(p.rows, p.logits_rows, p.served) for p in again] == [(p.rows, p.logits_rows, p.served) for p in obs]


def test_predictor_v3_kv_models_preemption_and_recompute_from_the_block_budget():
    # block_size 4, 4 blocks of which block 0 is the null block -> 3 usable.  Two 4-token prompts fit (1 block each); the first
    # decode step needs a second block for each: request 0 gets the last free block, request 1 (the last running one) is
    # preempted; it is re-admitted (recompute of prompt + 1 generated token, 2 blocks) only after request 0 finished.
    phi = _phi(prompt_lens=(4, 4), max_tokens=(3, 3))
    phi.engine.update({"block_size": 4, "num_gpu_blocks": 4})
    plans = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))
    layout = [[(r, p) for r, p in s.rows] for s in plans]
    assert layout == [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
                      [(0, 4)], [(0, 5)],
                      [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)], [(1, 5)]]
    assert [s.served for s in plans] == [[True, True], [True], [True], [True], [True]]
    # without the KV capacity in φ, v3-kv is v2: no preemption, both requests decode side by side
    phi.engine.pop("num_gpu_blocks")
    assert [len(s.rows) for s in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))] == [8, 2, 2]
    # and the v2 residual of the preempting engine: every step request 1 is missing (1, 2) plus the recompute step (3)
    hints, rep = encode_hints(phi, _observed(plans))
    assert hints.served == [] and {h.step for h in hints.steps} == {1, 2, 3}
    assert [(p.rows, p.logits_rows, p.served) for p in predict_schedule(phi, hints)] == [(p.rows, p.logits_rows, p.served) for p in plans]


def test_predictor_v3_kv_prefix_caching_hits_full_blocks_including_in_flight_ones():
    # block_size 4.  Request 0: prompt P (8 tokens) computed at step 0 -> its two full blocks are cached at scheduling time.
    # Requests 1 and 2 arrive at step 2 with prompts P+X (12) and P+X+Y (16): request 1 hits 2 blocks (starts at 8) and, since
    # its own third block is cached when it is scheduled, request 2 admitted in the SAME step hits 3 blocks (starts at 12).
    # Request 3 (prompt == P exactly, arrive 3) may hit at most (8-1)//4 = 1 block: the last token is always recomputed.
    P, X, Y = list(range(100, 108)), list(range(200, 204)), list(range(300, 304))
    reqs = [{"request_id": f"r{i}", "prompt_token_ids": t, "max_tokens": 2, "seed": None, "n": 1, "arrive_step": a}
            for i, (t, a) in enumerate([(P, 0), (P + X, 2), (P + X + Y, 2), (P, 3)])]
    phi = Phi(config={"H": 8}, profile={"module": "synthetic", "id": "0" * 12, "pins": {}},
              engine={"block_size": 4, "max_num_seqs": 8, "max_num_batched_tokens": 1 << 20, "enable_chunked_prefill": True,
                      "enable_prefix_caching": True, "num_gpu_blocks": 1000},
              workload={"requests": reqs, "sampling": {"temperature": 0.0}})
    plans = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))
    runs = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in plans}
    assert runs[0] == [(0, 0, 8)]
    assert runs[2] == [(1, 8, 4), (2, 12, 4)]                      # both hits; the 4-token chunks tie -> scheduler order
    assert runs[3] == [(1, 12, 1), (2, 16, 1), (3, 4, 4)]           # request 3: 1 block hit, recomputes tokens 4..7
    assert all(all(p.served) for p in plans)
    # caching off: the same φ prefills everything
    phi.engine["enable_prefix_caching"] = False
    runs_off = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))}
    assert runs_off[2] == [(1, 0, 12), (2, 0, 16)]


def test_predictor_v4_pool_counts_shared_prefix_blocks_once_under_kv_pressure():
    # block_size 4, 6 blocks (5 usable).  Three prompts share an 8-token prefix (2 full blocks) + 2 own tokens; max_tokens 2 -> each
    # needs 3 blocks by itself.  v3-kv counts 3 per request (9 > 5) and admits only request 0 at step 0; the pool holds the shared
    # blocks ONCE (3 + 1 + 1 = 5) and admits all three (C-codec's pp4 mechanism).
    P = list(range(100, 108))
    reqs = [{"request_id": f"r{i}", "prompt_token_ids": P + [200 + 2 * i, 201 + 2 * i], "max_tokens": 2, "seed": None, "n": 1, "arrive_step": 0}
            for i in range(3)]
    phi = Phi(config={"H": 8}, profile={"module": "synthetic", "id": "0" * 12, "pins": {}},
              engine={"block_size": 4, "max_num_seqs": 8, "max_num_batched_tokens": 1 << 20, "enable_chunked_prefill": True,
                      "enable_prefix_caching": True, "num_gpu_blocks": 6},
              workload={"requests": reqs, "sampling": {"temperature": 0.0}})
    v4 = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v4-pool"))
    runs = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in v4}
    assert runs[0] == [(1, 8, 2), (2, 8, 2), (0, 0, 10)]                 # hits at 8, the 10-token prompt sorts last
    assert v4[0].hits == [(1, 0, 4, 0), (1, 4, 4, 0), (2, 0, 4, 0), (2, 4, 4, 0)]   # producer of every hit block: request 0
    assert len(v4) == 2 and runs[1] == [(0, 10, 1), (1, 10, 1), (2, 10, 1)]
    v3 = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))
    assert [(r.request, r.first, r.count) for r in v3[0].runs()] == [(0, 0, 10)]   # the per-request count refuses 1 and 2
    # the pool residual of the real mechanism against v3-kv is charged; against v4-pool it is empty
    hints, _ = encode_hints(phi, _observed(v4), predictor="dense-eager-fcfs/v3-kv")
    assert len(hints.steps) >= 1
    hints4, _ = encode_hints(phi, _observed(v4), predictor="dense-eager-fcfs/v4-pool")
    assert hints4.empty


def test_predictor_v4_pool_holds_a_max_tokens_stop_through_the_next_schedule():
    # block_size 4, 5 blocks (4 usable), caching off.  Requests 0 (P 4, max 3) and 1 (P 4, max 2) fill the pool by step 1; request 1
    # stops at max_tokens at step 1.  The finish is processed after schedule(2), so request 2 (arrive 2) is admitted at step 3 under
    # v4-pool (MEASURED pre6 120->122) but at step 2 under v3-kv's end-of-step free.
    phi = _phi(prompt_lens=(4, 4, 4), max_tokens=(3, 2, 1), arrive=(0, 0, 2))
    phi.engine.update({"block_size": 4, "num_gpu_blocks": 5})
    v4 = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v4-pool"))}
    assert v4[2] == [(0, 5, 1)] and v4[3] == [(2, 0, 4)]
    v3 = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v3-kv"))}
    assert v3[2] == [(0, 5, 1), (2, 0, 4)]
    # if that schedule(2) would have been empty (request 0 stopped at step 1 too), the retry after the free IS step 2 (preempt4 159->160)
    phi2 = _phi(prompt_lens=(4, 4, 4), max_tokens=(2, 2, 1), arrive=(0, 0, 2))
    phi2.engine.update({"block_size": 4, "num_gpu_blocks": 5})
    v4b = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi2, Hints(predictor="dense-eager-fcfs/v4-pool"))}
    assert v4b[2] == [(2, 0, 4)]


def test_predictor_v5_seat_holds_a_max_tokens_stop_seat_through_the_next_schedule():
    # max_num_seqs 2, pool unlimited (100 blocks).  Requests 0 (max 3) and 1 (max 2) take both seats at step 0; request 2 waits.
    # Request 1 stops at max_tokens at step 1.  scheduler.py:590-603 / :816-819: at schedule(2) it is still in `running` (skipped,
    # no rows) and counts against max_num_seqs, so request 2 is admitted at step 3 under v5-seat; v4-pool (seat freed at the end of
    # step 1) admits it at step 2 — the mix16 residual (steps 36/46/58).  An EOS stop (served < max_tokens) leaves after its placeholder.
    phi = _phi(prompt_lens=(4, 4, 4), max_tokens=(4, 2, 1), arrive=(0, 0, 0), max_num_seqs=2)
    phi.engine.update({"block_size": 4, "num_gpu_blocks": 100})
    v5 = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v5-seat"))}
    assert v5[1] == [(0, 4, 1), (1, 4, 1)] and v5[2] == [(0, 5, 1)] and v5[3] == [(0, 6, 1), (2, 0, 4)] and max(v5) == 3
    v4 = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v4-pool"))}
    assert v4[2] == [(0, 5, 1), (2, 0, 4)]
    # EOS stop of request 1 after 1 token (served 2 = token + placeholder, < max_tokens 3): seat leaves after the placeholder step 1
    phi_eos = _phi(prompt_lens=(4, 4, 4), max_tokens=(4, 3, 1), arrive=(0, 0, 0), max_num_seqs=2)
    phi_eos.engine.update({"block_size": 4, "num_gpu_blocks": 100})
    h = Hints(predictor="dense-eager-fcfs/v5-seat", served=[ServedLength(request=1, served=2)])
    v5e = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi_eos, h)}
    assert v5e[1] == [(0, 4, 1), (1, 4, 1)] and v5e[2] == [(0, 5, 1), (2, 0, 4)]
    # if the held seat would leave schedule(2) empty (request 0 stopped at step 1 too), the engine drains first: step 2 admits request 2
    phi2 = _phi(prompt_lens=(4, 4, 4), max_tokens=(2, 2, 1), arrive=(0, 0, 0), max_num_seqs=2)
    phi2.engine.update({"block_size": 4, "num_gpu_blocks": 100})
    v5b = {p.step: [(r.request, r.first, r.count) for r in p.runs()] for p in predict_schedule(phi2, Hints(predictor="dense-eager-fcfs/v5-seat"))}
    assert v5b[2] == [(2, 0, 4)] and max(v5b) == 2
    # the 26-record corpus shape (all requests admitted at once, no seat cap binding) is unchanged: v5-seat == v4-pool == v2
    phi3 = _phi(prompt_lens=(5, 7, 3), max_tokens=(3, 4, 2))
    phi3.engine.update({"num_gpu_blocks": 100})
    a = [p.runs() for p in predict_schedule(phi3, Hints(predictor="dense-eager-fcfs/v5-seat"))]
    b = [p.runs() for p in predict_schedule(phi3, Hints(predictor="dense-eager-fcfs/v2"))]
    assert a == b


def test_encode_steps_hint_reproduces_an_unpredicted_row_order():
    phi = _phi(prompt_lens=(5, 7, 3), max_tokens=(4, 2, 2), arrive=(0, 1, 1))
    obs = predict_schedule(phi, Hints(predictor="dense-eager-fcfs/v1"))     # an engine that kept scheduler order
    hints, rep = encode_hints(phi, _observed(obs))                          # encoded against v2
    assert hints.predictor == PREDICTOR_ID and len(hints.steps) == 1 and hints.steps[0].step == 1 and hints.served == []
    runs = [(r.request, r.first, r.count) for r in hints.steps[0].rows]
    assert runs == [(0, 5, 1), (1, 0, 7), (2, 0, 3)]
    assert hints.naive_bits(max_tokens=4, n_requests=3, max_position=16) > 0
    again = predict_schedule(phi, hints)
    assert [(p.rows, p.logits_rows, p.served) for p in again] == [(p.rows, p.logits_rows, p.served) for p in obs]
    # and the hints survive the JSON interchange
    assert Hints.from_json(json.loads(json.dumps(hints.to_json()))) == hints


def test_n_greater_than_one_forks_into_children_by_the_parallel_sampling_rule():
    # vllm/v1/engine/parallel_sampling.py: child id f"{index}_{request_id}", seed + index, added in index order (llm_engine.py:289)
    phi = _phi(prompt_lens=(5, 4), max_tokens=(3, 2))
    phi.workload["requests"][0].update(n=2, seed=7)
    lr = phi.logical_requests()
    assert [r["request_id"] for r in lr] == ["0_r0", "1_r0", "r1"]
    assert [r["seed"] for r in lr] == [7, 8, None] and all(r["n"] == 1 for r in lr)
    assert [(r["_parent"], r["_child"]) for r in lr] == [(0, 0), (0, 1), (1, 0)]
    plans = predict_schedule(phi)
    # both children prefill in step 0 (prefix caching off: each computes its own prompt), sorted by scheduled tokens: r1 (4) first
    assert plans[0].rows == [(2, p) for p in range(4)] + [(0, p) for p in range(5)] + [(1, p) for p in range(5)]
    assert sum(sum(p.served) for p in plans) == 3 + 3 + 2
    # the observer maps child ids to the logical indices; a header `children` list must agree with the rule
    obs = _observed(plans)
    hints, rep = encode_hints(phi, obs)
    assert hints.served == [] and hints.steps == [] and rep["corrections"] == []
    # no fork: logical_requests is the header list
    assert [r["request_id"] for r in _phi().logical_requests()] == ["r0", "r1", "r2"]
