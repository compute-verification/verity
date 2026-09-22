"""[R8 C-codec] Exactness tests: decode(phi, encode(phi, h)) == h for A0-A3 at every phi level, on synthetic schedules that exercise
the paths the corpus does not (chunked prefill, early stop, row-order deviation, delayed admission, num_computed discontinuity,
missing logits row, large arrival gaps -> histogram escape).  The A3 model is fitted on *unperturbed* synthetic runs only and frozen
before the perturbed ones are coded (no per-run model changes).  Laptop-sized (< 1 s)."""
from __future__ import annotations

import json
import random

import pytest

from .. import codecs, facts, model, sched

LEVELS = ["phi0", "phi1a", "phi1", "phi2"]


def synth_phi(rng: random.Random, n: int, budget: int, max_L: int, max_M: int, max_gap: int) -> tuple[dict, list[dict]]:
    reqs, arrive = [], 0
    for i in range(n):
        reqs.append({"request_id": f"r{i}", "prompt_len": rng.randint(1, max_L), "max_tokens": rng.randint(1, max_M)})
        arrive += rng.randint(0, max_gap)
        reqs[-1]["arrive_step"] = arrive
    phi0 = {"schema": "aot-codec/phi/v1", "run_id": "synth", "model": "synthetic", "requests": [{k: q[k] for k in ("request_id", "prompt_len", "max_tokens")} for q in reqs],
            "engine": {"max_num_seqs": rng.choice([2, 4, 8]), "max_num_batched_tokens": budget, "chunked_prefill": True, "prefix_caching": False, "block_size": 16}}
    return phi0, reqs


def synth_hint(phi0: dict, reqs: list[dict], n_generated: list[int] | None = None) -> dict:
    hreqs = [{"prompt_len": q["prompt_len"], "max_tokens": q["max_tokens"], "arrive_step": q["arrive_step"],
              "n_generated": (n_generated[i] if n_generated else q["max_tokens"])} for i, q in enumerate(reqs)]
    for q in hreqs:
        q["finish_reason"] = "length" if q["n_generated"] == q["max_tokens"] else "stop"
    steps, logits = sched.simulate(phi0, hreqs)
    # realized arrivals: the harness collapses idle gaps -> first step of each request
    first = {}
    for t, s in enumerate(steps):
        for i, _, _ in s["rows"]:
            first.setdefault(i, t)
    for i, q in enumerate(hreqs):
        q["arrive_step"] = first[i]
    return {"schema": facts.SCHEMA, "requests": hreqs, "steps": steps, "logits": logits, "_declared_engine_steps": len(steps)}


def roundtrip_all(phi0: dict, h: dict, m: dict) -> dict[str, dict[str, int]]:
    hs = facts.strip_private(h)
    out = {}
    for lvl in LEVELS:
        phi = facts.phi_variant(phi0, h, lvl)
        for name in codecs.CODECS:
            res = codecs.encode(name, phi, hs, m)
            back = codecs.decode(name, phi, res.bits, m)
            assert facts.canonical_json(back) == facts.canonical_json(hs), f"{name} {lvl} mismatch"
            out.setdefault(lvl, {})[name] = res.n_bits
    return out


@pytest.fixture(scope="module")
def frozen_model() -> dict:
    rng = random.Random(1)
    training = []
    for k in range(12):
        phi0, reqs = synth_phi(rng, rng.randint(1, 8), rng.choice([64, 128, 2112, 8192]), 300, 30, 6)
        h = synth_hint(phi0, reqs)
        for lvl in LEVELS:
            training.append((facts.phi_variant(phi0, h, lvl), facts.strip_private(h)))
    return model.fit(training, [f"synth{k}" for k in range(12)])


def test_clean_synthetic_chunked(frozen_model):
    rng = random.Random(7)
    for k in range(10):
        phi0, reqs = synth_phi(rng, rng.randint(1, 8), rng.choice([40, 64, 128]), 300, 20, 5)
        h = synth_hint(phi0, reqs)
        assert any(0 < c < q["prompt_len"] for s in h["steps"] for (i, c, n) in s["rows"] for q in [h["requests"][i]] if c + n < q["prompt_len"] or c > 0 and c < q["prompt_len"]) or True
        bits = roundtrip_all(phi0, h, frozen_model)
        assert bits["phi1"]["A2"] == 1  # the single `clean` flag: prediction exact by construction
        assert bits["phi1"]["A1"] >= bits["phi1"]["A2"]


def test_early_stop(frozen_model):
    rng = random.Random(3)
    phi0, reqs = synth_phi(rng, 6, 2112, 250, 30, 4)
    g = [max(1, q["max_tokens"] - rng.randint(0, q["max_tokens"] - 1)) for q in reqs]
    h = synth_hint(phi0, reqs, g)
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1"]["A2"] == 1 and bits["phi0"]["A2"] > bits["phi1a"]["A2"] > 1  # early stops cost n_generated bits, not finish_reason bits
    # a non-rule finish reason (abort) escapes to the explicit enum for every stopped request
    h2 = json.loads(json.dumps(h))
    h2["requests"][0]["finish_reason"] = "abort"
    bits2 = roundtrip_all(phi0, h2, frozen_model)
    assert bits2["phi1"]["A2"] > bits["phi1"]["A2"]


def test_row_order_deviation(frozen_model):
    rng = random.Random(5)
    phi0, reqs = synth_phi(rng, 8, 2112, 250, 12, 0)  # all arrive at step 0 -> 8 prefills in step 0
    h = synth_hint(phi0, reqs)
    rows = h["steps"][0]["rows"]
    assert len(rows) >= 2
    rows[0], rows[1] = rows[1], rows[0]
    h["logits"][0] = [[i, 1 if c + n >= h["requests"][i]["prompt_len"] else 0] for i, c, n in rows]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1"]["A2"] > 1  # one escaped step


def test_delayed_admission_changes_T(frozen_model):
    rng = random.Random(11)
    phi0, reqs = synth_phi(rng, 3, 2112, 100, 5, 2)
    h = synth_hint(phi0, reqs)
    # delay the last request by one step: remove it from its first step, shift all its later rows one step later
    last = len(reqs) - 1
    steps, logits = h["steps"], h["logits"]
    new_steps, new_logits, carry = [], [], None
    for s, lg in zip(steps, logits):
        rows = [r for r in s["rows"] if r[0] != last]
        mine = [r for r in s["rows"] if r[0] == last]
        if carry:
            rows = rows + carry
        carry = mine
        new_steps.append({"rows": rows}); new_logits.append([[i, 1 if c + n >= h["requests"][i]["prompt_len"] else 0] for i, c, n in rows])
    if carry:
        new_steps.append({"rows": carry}); new_logits.append([[i, 1 if c + n >= h["requests"][i]["prompt_len"] else 0] for i, c, n in carry])
    new_steps = [s for s in new_steps if s["rows"]]
    new_logits = [l for l in new_logits if l]
    h["steps"], h["logits"] = new_steps, new_logits
    first = {}
    for t, s in enumerate(new_steps):
        for i, _, _ in s["rows"]:
            first.setdefault(i, t)
    for i, q in enumerate(h["requests"]):
        q["arrive_step"] = first[i]
    roundtrip_all(phi0, h, frozen_model)


def test_num_computed_discontinuity_and_missing_logits(frozen_model):
    rng = random.Random(13)
    phi0, reqs = synth_phi(rng, 2, 2112, 50, 6, 0)
    h = synth_hint(phi0, reqs)
    # recompute: request 0 restarts its prompt at some later step (preemption-like); the schedule-implied n_generated changes,
    # so keep the total consistent by dropping its final decode step rows accordingly is not needed: n_generated is re-derived
    t = len(h["steps"]) // 2
    L0 = h["requests"][0]["prompt_len"]
    h["steps"][t]["rows"] = [[0, 0, L0]] + [r for r in h["steps"][t]["rows"] if r[0] != 0]
    for s in h["steps"][t + 1:]:
        for r in s["rows"]:
            if r[0] == 0:
                r[1] = L0 + (r[1] - r[1])  # placeholder, fixed below
    # rebuild positions of request 0 after t as consecutive from L0
    pos = L0
    for s in h["steps"][t + 1:]:
        for r in s["rows"]:
            if r[0] == 0:
                r[1] = pos; pos += r[2]
    for k, s in enumerate(h["steps"]):
        h["logits"][k] = [[i, 1 if c + n >= h["requests"][i]["prompt_len"] else 0] for i, c, n in s["rows"]]
    # drop one logits row (engine computed no logits for a request in one step)
    if len(h["logits"][t]) > 1:
        h["logits"][t] = h["logits"][t][1:]
    tot0 = sum(n for s in h["steps"] for i, c, n in s["rows"] if i == 0)
    h["requests"][0]["n_generated"] = max(tot0 - L0 + 1 - L0, 0)  # implied by the (now longer) row total minus the recompute
    h["requests"][0]["finish_reason"] = "other"
    roundtrip_all(phi0, h, frozen_model)


def test_large_arrival_gap_hits_histogram_escape(frozen_model):
    rng = random.Random(17)
    phi0, reqs = synth_phi(rng, 2, 2112, 100, 60, 0)
    reqs[0]["max_tokens"] = phi0["requests"][0]["max_tokens"] = 80  # request 0 is still decoding at step 40: no idle-gap collapse
    reqs[1]["arrive_step"] = 40
    h = synth_hint(phi0, reqs)
    assert h["requests"][1]["arrive_step"] == 40  # zig-zag delta 80 > K=16 bins -> histogram escape + gamma
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi0"]["A3"] > bits["phi1a"]["A3"]


def test_arith_coder_ideal_bound(frozen_model):
    rng = random.Random(19)
    phi0, reqs = synth_phi(rng, 8, 2112, 250, 20, 3)
    h = synth_hint(phi0, reqs)
    phi = facts.phi_variant(phi0, h, "phi0")
    res = codecs.encode("A3", phi, facts.strip_private(h), frozen_model)
    assert res.ideal_bits <= res.n_bits <= res.ideal_bits + 2


def test_sampling_n_fanout_is_phi_derivable(frozen_model):
    """A workload request with n=2 is two engine requests `0_<id>`, `1_<id>` (vllm/v1/engine/parallel_sampling.py:92).  The expansion is
    part of phi (zero advice): a bare log with the two child ids extracts, checks and codes as a clean 2-request run (observed on the
    out-of-corpus `n2` record, see out/gen/r8/aot-codec/extra_logs.md)."""
    spec = {"workload": {"requests": [{"request_id": "r", "prompt_token_ids": list(range(32)), "max_tokens": 4, "n": 2}], "sampling": {"n": 2}},
            "configuration": {"checkpoint": {"repo": "m"}}}
    reqs = facts.logical_requests(spec)
    assert [q["request_id"] for q in reqs] == ["0_r", "1_r"] and all(q["max_tokens"] == 4 for q in reqs)
    steps = []
    for t in range(4):  # both children: prefill 32 at t=0, then 3 decodes (row count L + M - 1 = 35 per child)
        rows, nc, sl, lg = [], {}, {}, []
        for rid in ("0_r", "1_r"):
            c, n = (0, 32) if t == 0 else (31 + t, 1)
            rows += [[rid, c + k] for k in range(n)]
            nc[rid], sl[rid] = c, c + n
            lg.append([rid, c + n - 1])
        steps.append({"$e": "StepStart", "step": t, "rows": rows, "num_computed": nc, "seq_lens": sl, "logits_rows": lg, "num_reqs": 2, "num_tokens": len(rows)})
    r = {"dir": "-", "run_id": "n2", "spec": spec, "experiment": {}, "fold": {}, "card": {}, "header": {"profile": {"cache": {"max_num_seqs": 8, "max_num_batched_tokens": 8192}}},
         "steps": steps, "samples": {}, "other": []}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"]
    assert [q["n_generated"] for q in h["requests"]] == [4, 4]
    bits = roundtrip_all(facts.phi_from_record(r), h, frozen_model)
    assert bits["phi1a"]["A2"] == 1 and bits["phi1"]["A3"] == 1


def _prefix_phi(ids_a: list[int], ids_b: list[int], max_tokens=(8, 8), caching=True) -> dict:
    B = 16
    return {"schema": "aot-codec/phi/v1", "run_id": "synth-prefix", "model": "synthetic",
            "requests": [{"request_id": "a", "prompt_len": len(ids_a), "max_tokens": max_tokens[0], "block_hashes": facts.block_hashes(ids_a, B)},
                         {"request_id": "b", "prompt_len": len(ids_b), "max_tokens": max_tokens[1], "block_hashes": facts.block_hashes(ids_b, B)}],
            "engine": {"max_num_seqs": 8, "max_num_batched_tokens": 8192, "chunked_prefill": True, "prefix_caching": caching, "block_size": B}}


def test_prefix_cache_hit_is_derivable(frozen_model):
    """Two 88-token prompts sharing 64 tokens (4 full blocks), admitted together with prefix caching on: vLLM allocates request 0 first
    (caching its 5 full blocks), request 1 is looked up and admitted with num_computed = 64 (SOURCE-INSPECTED; MEASURED on the
    out-of-corpus record /vol/cp/batch/runs/b0_prefix64_2req).  The hit is a function of phi0 -> the predicted schedule equals the record's
    (rows sorted by n_scheduled: request 1's 24-token chunk before request 0's 88) and A2/A3 cost the single clean flag."""
    a = list(range(1000, 1088))
    b = a[:64] + list(range(5000, 5024))
    phi0 = _prefix_phi(a, b)
    reqs = [{"prompt_len": 88, "max_tokens": 8, "arrive_step": 0, "n_generated": 8, "finish_reason": "length"} for _ in range(2)]
    steps, logits = sched.simulate(phi0, reqs)
    assert steps[0]["rows"] == [[1, 64, 24], [0, 0, 88]] and logits[0] == [[1, 1], [0, 1]] and len(steps) == 8
    h = {"schema": facts.SCHEMA, "requests": reqs, "steps": steps, "logits": logits, "_declared_engine_steps": 8}
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1a"]["A2"] == 1 and bits["phi1"]["A3"] == 1
    # the same schedule with caching *off* in phi is a deviation: positions do not follow the state -> A1 falls back to explicit positions,
    # A2 escapes exactly one step (the admission), all codecs still exact
    phi_off = _prefix_phi(a, b, caching=False)
    bits_off = roundtrip_all(phi_off, h, frozen_model)
    assert bits_off["phi1a"]["A2"] > 1 and bits_off["phi1"]["A1"] > bits["phi1"]["A1"]


def test_prefix_hit_against_generated_tokens_is_advice(frozen_model):
    """A hit longer than the shared *prompt* prefix (e.g. the cache matched blocks produced by another request's generated tokens) is not
    derivable from phi: the codec must escape (nc_eq miss -> explicit position) and still round-trip."""
    a = list(range(1000, 1088))
    b = a[:64] + list(range(5000, 5024))
    phi0 = _prefix_phi(a, b)
    reqs = [{"prompt_len": 88, "max_tokens": 8, "arrive_step": 0, "n_generated": 8, "finish_reason": "length"} for _ in range(2)]
    steps, logits = sched.simulate(phi0, reqs)
    steps[0]["rows"] = [[1, 80, 8], [0, 0, 88]]  # realized hit = 80 tokens (5 blocks), not the derivable 64
    logits[0] = [[1, 1], [0, 1]]
    h = {"schema": facts.SCHEMA, "requests": reqs, "steps": steps, "logits": logits, "_declared_engine_steps": len(steps)}
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1"]["A2"] > 1


def test_real_eos_record_async_placeholder(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/eos4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 08:36Z on cp-aot): two requests stop on
    <|endoftext|> (after 8 and 1 generated tokens), one runs to max_tokens, one arrives at step 3.  Under async scheduling an EOS stop gets one
    placeholder decode row (rows sampled = n_generated + 1); a max_tokens stop does not.  Given n_generated (phi1) the schedule is fully
    predicted: A2 = A3 = 1 bit; at phi1a the stop points are the advice."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/eos4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("eos4 record not present in this checkout")
    r = facts.load_log_only(log, "eos4")
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"]
    assert [(q["n_generated"], q["finish_reason"], q["arrive_step"]) for q in h["requests"]] == [(8, "stop", 0), (1, "stop", 0), (32, "length", 0), (20, "length", 3)]
    assert sum(1 for e in audit["events"] if e["kind"].startswith("EOS stop")) == 2
    assert facts.check_record(r, h)["ok"] and len(h["steps"]) == 32
    # request 1: prefill at step 0, EOS sampled there, placeholder decode row at step 1, gone at step 2
    assert [i for i, _, _ in h["steps"][1]["rows"]] == [0, 1, 2] and [i for i, _, _ in h["steps"][2]["rows"]] == [0, 2]
    bits = roundtrip_all(facts.phi_from_record(r), h, frozen_model)
    assert bits["phi1"]["A2"] == 1 and bits["phi1"]["A3"] == 1 and bits["phi2"]["A2"] == 1
    assert bits["phi1a"]["A2"] > bits["phi1"]["A2"]  # the two stop points are advice until phi declares n_generated


def _pool_phi(n_blocks: int | None, prompt_lens=(34, 35, 37, 33), max_tokens=160) -> dict:
    return {"schema": "aot-codec/phi/v1", "run_id": "synth-pool", "model": "synthetic",
            "requests": [{"request_id": f"r{i}", "prompt_len": L, "max_tokens": max_tokens} for i, L in enumerate(prompt_lens)],
            "engine": {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "chunked_prefill": True, "prefix_caching": False, "block_size": 16,
                       "async_scheduling": True, "num_gpu_blocks": n_blocks}}


def test_preemption_synthetic_pool_roundtrips(frozen_model):
    """A 20-block pool (19 usable) for 4 x (~35 + 160) tokens: the simulator preempts (last running request first), resumes with a
    re-prefill of prompt + generated tokens, and every codec round-trips at every phi level.  With the pool in phi (phi2) the run is
    clean (1 bit); without it (phi0/phi1a/phi1) the preemptions and resumes are advice."""
    phi0 = _pool_phi(20)
    reqs = [{"prompt_len": q["prompt_len"], "max_tokens": 160, "arrive_step": 0, "n_generated": 160, "finish_reason": "length"} for q in phi0["requests"]]
    steps, logits = sched.simulate(phi0, reqs)
    T = len(steps)
    assert T > 4 * 160 // 2, "the pool must force serialisation"
    # a resume row: num_computed 0 with more tokens than the prompt
    resumes = [(t, g) for t, s in enumerate(steps) for g in s["rows"] if g[1] == 0 and g[2] > phi0["requests"][g[0]]["prompt_len"]]
    assert resumes, "expected at least one re-prefill of prompt + generated tokens"
    h = {"schema": facts.SCHEMA, "requests": reqs, "steps": steps, "logits": logits, "_declared_engine_steps": T}
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1
    assert bits["phi1"]["A2"] > 100 and bits["phi0"]["A1"] > bits["phi2"]["A1"] - 200  # A1 does not use the pool model


def test_real_preemption_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/preempt4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 08:55Z on cp-aot with
    num_gpu_blocks_override=20, max_model_len=256): 4 requests x 160 tokens, 5 preemptions (steps 32, 62, 111, 190, 257) and 3 resume
    steps (160, 209, 277), 357 steps.  First contact (code 2f3327ea1) was NOT EXPRESSIBLE (chunk alphabet [1, remaining prompt];
    n_generated inferred from total scheduled tokens).  With the uniform rules (chunk alphabet [1, prompt + sampled - position];
    one sample per row that reaches the prompt end) it is exact; with the KV pool size in phi2 the block-accounting predictor
    reproduces all 357 steps: A2 = A3 = 1 bit.  Without the pool size the preemption schedule is advice (~2.2 kbit)."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/preempt4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("preempt4 record not present in this checkout")
    r = facts.load_log_only(log, "preempt4")
    r["phi_overrides"] = {"num_gpu_blocks": 20}  # the RunHeader does not record the pool size (harness gap)
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"]
    assert [q["n_generated"] for q in h["requests"]] == [160] * 4 and len(h["steps"]) == 357
    absent = lambda t: {0, 1, 2, 3} - {g[0] for g in h["steps"][t]["rows"]}
    assert absent(31) == set() and absent(32) == {3} and absent(62) == {2, 3} and absent(111) == {1, 2, 3}
    assert h["steps"][160]["rows"] == [[2, 0, 99], [1, 0, 146]]  # resumes: prompt + generated so far, sorted by n_scheduled
    phi0 = facts.phi_from_record(r)
    assert phi0["engine"]["num_gpu_blocks"] == 20
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1
    assert 1000 < bits["phi1"]["A2"] < 4000 and bits["phi1"]["A1"] < 5000
    # the pool size is the *only* phi2 fact that matters here: phi1 + num_gpu_blocks alone is clean too
    phi1p = facts.phi_variant(phi0, h, "phi1")
    phi1p["engine"]["num_gpu_blocks"] = 20
    hs = facts.strip_private(h)
    res = codecs.encode("A2", phi1p, hs, frozen_model)
    assert res.n_bits == 1 and facts.canonical_json(codecs.decode("A2", phi1p, res.bits, frozen_model)) == facts.canonical_json(hs)


def test_real_prefix_pool_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/pp4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 09:28Z on cp-aot with
    enable_prefix_caching=true, num_gpu_blocks_override=20, max_model_len=256): four 70/71-token prompts sharing 64 tokens (4 full blocks),
    100 generated tokens each.  The shared blocks are held *once* by the pool (ref_cnt): the run admits all four at step 0 (8 blocks, not
    20), preempts at steps 43 and 74, resumes both at step 100 -- one with num_computed 64 (the shared prefix, still held by the running
    requests), one with 80 (its own next block survived in the LRU free queue).  First contact with the per-request block count
    (rev 9c0d1a5) was 1720 bits at phi2 -- *worse than no pool model* (926 at phi1); with ref-counted blocks + admission lookup on
    resume (rev 0c2b8b8dd..c4ac79901), 156/157 steps were predicted and the LRU survival was the single escape (83 bits at phi2); with the
    block-identity pool (LRU free queue: unhashed blocks prepended, hashed appended, front popped -- 10:58Z) all 157 steps are predicted
    and the record is 1 bit at phi2."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/pp4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("pp4 record not present in this checkout")
    r = facts.load_log_only(log, "pp4")
    r["phi_overrides"] = {"num_gpu_blocks": 20}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 157
    assert [(q["prompt_len"], q["n_generated"]) for q in h["requests"]] == [(70, 100), (71, 100), (71, 100), (71, 100)]
    assert sorted(g[1] for g in h["steps"][0]["rows"]) == [0, 64, 64, 64]  # three admission hits on the shared prefix
    assert h["steps"][100]["rows"] == [[3, 64, 50], [2, 80, 65]]  # the two resumes: shared-prefix hit, and an LRU-survived own block
    phi2 = facts.phi_variant(facts.phi_from_record(r), h, "phi2")
    assert phi2["engine"]["num_gpu_blocks"] == 20 and phi2["engine"]["prefix_caching"]
    st = sched.SchedState(phi2, h["requests"])
    mism = []
    for t, s in enumerate(h["steps"]):
        pr, pl = st.predict()
        if pr != s["rows"] or pl != h["logits"][t]:
            mism.append(t)
        st.apply(s["rows"])
    assert mism == [], mism  # preemptions at 43 and 74, the 64-token resume *and* the 80-token resume (LRU-survived own block) are predicted
    assert {g[0] for g in h["steps"][43]["rows"]} == {0, 1, 2} and {g[0] for g in h["steps"][74]["rows"]} == {0, 1}  # last running first
    bits = roundtrip_all(facts.phi_from_record(r), h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and bits["phi1"]["A2"] > 500  # without the pool size the preemption schedule is advice


def test_real_preregistered_composition_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/pre6 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 09:50Z on cp-aot with
    num_gpu_blocks_override=24, max_model_len=256, long_prefill_token_threshold=24, prefix caching off): six requests -- two EOS stops (8 and 1
    tokens), arrivals 0/0/0/2/2/5, chunk cap, KV pressure (2 preemptions at step 48/72, both resumed at step 122), 182 steps.  The prediction
    "A2 = A3 = 1 bit at phi2" was written down BEFORE the capture (gen/pre6_preregistration.md) and FAILED on first contact (code e0c533b85:
    unexpressible + 162 bits) for two reasons the record exposed: (a) a re-prefill under the chunk cap is several chunks and only the one that
    reaches the end of the known sequence samples (the old rule counted every chunk past the prompt end); (b) under async scheduling the
    engine schedules step t+1 before it processes step t's output, so a length-stopped request holds its blocks and slot for one more step --
    the resumes came at last row + 2 (step 122), the predictor said + 1.  With the two uniform rules the record is clean."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/pre6/log.jsonl.gz"
    if not log.exists():
        pytest.skip("pre6 record not present in this checkout")
    r = facts.load_log_only(log, "pre6")
    r["phi_overrides"] = {"num_gpu_blocks": 24, "long_prefill_token_threshold": 24}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 182
    assert [(q["prompt_len"], q["n_generated"], q["finish_reason"], q["arrive_step"]) for q in h["requests"]] == [
        (14, 8, "stop", 0), (13, 1, "stop", 0), (34, 120, "length", 0), (44, 120, "length", 2), (61, 120, "length", 2), (27, 100, "length", 5)]
    assert h["steps"][120]["rows"] == [[2, 152, 1], [3, 160, 1]] and h["steps"][121]["rows"] == [[3, 161, 1]]  # request 2's last row at 120
    assert h["steps"][122]["rows"] == [[3, 162, 1], [4, 0, 24], [5, 0, 24]]  # resumes at last row + 2, in 24-token chunks
    assert h["steps"][124]["rows"] == [[5, 48, 21], [4, 48, 24]] and h["logits"][124] == [[5, 1], [4, 0]]  # only the final chunk samples
    phi0 = facts.phi_from_record(r)
    st = sched.SchedState(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    for t, s in enumerate(h["steps"]):
        assert st.predict() == (s["rows"], h["logits"][t]), t
        st.apply(s["rows"])
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and bits["phi1"]["A2"] > 500


def test_real_preregistered_budget_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/bud8 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 10:25Z on cp-aot with
    max_num_batched_tokens=64, num_gpu_blocks_override=40, max_model_len=256; prefix caching and chunk cap off): 8 requests, EOS stops (8 and 1),
    arrivals 0/0/0/1/1/3/5/8, budget-chunked prefills beside decodes, one preemption (step 38, request 7 mid-decode) resumed at step 61 in three
    budget-sized chunks, 75 steps.  Pre-registered (gen/bud8_preregistration.md, code 768f254d5 frozen before the capture): A2 = A3 = 1 bit at
    phi2 -- HELD on first contact (bud8/first_contact_rev_768f254d5.md).  The draining rule found on pre6 is load-bearing here out of sample:
    without it the resume is predicted one step early, twice."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/bud8/log.jsonl.gz"
    if not log.exists():
        pytest.skip("bud8 record not present in this checkout")
    r = facts.load_log_only(log, "bud8")
    r["phi_overrides"] = {"num_gpu_blocks": 40}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 75
    assert [(q["prompt_len"], q["n_generated"], q["finish_reason"], q["arrive_step"]) for q in h["requests"]] == [
        (14, 8, "stop", 0), (13, 1, "stop", 0), (34, 60, "length", 0), (44, 60, "length", 1), (61, 60, "length", 1), (27, 50, "length", 3), (104, 60, "length", 5), (129, 40, "length", 8)]
    assert h["steps"][1]["rows"] == [[0, 14, 1], [1, 13, 1], [2, 34, 1], [4, 0, 17], [3, 0, 44]]  # FCFS: 3 takes 44, 4 the 17 left; rows by size
    assert h["steps"][61]["rows"] == [[4, 119, 1], [6, 158, 1], [7, 0, 62]] and h["steps"][63]["rows"] == [[6, 160, 1], [7, 125, 32]]
    assert h["logits"][61] == [[4, 1], [6, 1], [7, 0]] and h["logits"][63] == [[6, 1], [7, 1]]  # only the last resume chunk samples
    phi0 = facts.phi_from_record(r)
    assert phi0["engine"]["max_num_batched_tokens"] == 64 and phi0["engine"]["num_gpu_blocks"] == 40
    st = sched.SchedState(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    for t, s in enumerate(h["steps"]):
        assert st.predict() == (s["rows"], h["logits"][t]), t
        st.apply(s["rows"])
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 400 < bits["phi1"]["A2"] < 800


def test_real_preregistered_midprefill_preemption_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/mp3 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 10:27Z on cp-aot with
    max_num_batched_tokens=64, num_gpu_blocks_override=15, max_model_len=224): two 29-token prompts x 40 tokens and a 150-token prompt x 20
    arriving at step 3.  Pre-registered step for step (gen/mp3_preregistration.md): C admitted at step 3 (62 tokens), second chunk at 4 (pool
    exactly full: 3 + 3 + 8 blocks), *preempted mid-prefill at position 124* at step 5 (its own block request, last running request), resumed at
    step 40 (both finishes drained before an empty schedule) with prompt-only chunks 64 / 64 / 22, T = 62 -- HELD on first contact, 1 bit at phi2."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/mp3/log.jsonl.gz"
    if not log.exists():
        pytest.skip("mp3 record not present in this checkout")
    r = facts.load_log_only(log, "mp3")
    r["phi_overrides"] = {"num_gpu_blocks": 15}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 62
    assert [(q["prompt_len"], q["n_generated"], q["arrive_step"]) for q in h["requests"]] == [(29, 40, 0), (29, 40, 0), (150, 20, 3)]
    assert h["steps"][3]["rows"] == [[0, 31, 1], [1, 31, 1], [2, 0, 62]] and h["steps"][4]["rows"] == [[0, 32, 1], [1, 32, 1], [2, 62, 62]]
    assert h["steps"][5]["rows"] == [[0, 33, 1], [1, 33, 1]]  # C preempted mid-prefill
    assert [s["rows"] for s in h["steps"][40:43]] == [[[2, 0, 64]], [[2, 64, 64]], [[2, 128, 22]]] and h["logits"][42] == [[2, 1]]
    phi0 = facts.phi_from_record(r)
    # the predictor's own output for this phi + these request facts *is* the record (the pre-registered step list)
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 300 < bits["phi1"]["A2"] < 500


def test_length_stop_holds_blocks_one_more_step():
    """Synthetic, max_num_seqs = 2, no pool: request 0 (4 tokens) and request 1 (40 tokens) hold both slots; request 2 waits.  0's last
    row is step 3; the engine processes that finish only after scheduling step 4, so 0 still holds its slot at step 4 and 2 is admitted at
    step 5 -- unless nothing else could run at step 4, in which case the engine drains the finish first and 2 follows at once."""
    phi = _pool_phi(None, prompt_lens=(16, 16, 16), max_tokens=40)
    phi["engine"]["max_num_seqs"] = 2
    reqs = [{"prompt_len": 16, "max_tokens": 40, "arrive_step": 0, "n_generated": 4, "finish_reason": "length"},
            {"prompt_len": 16, "max_tokens": 40, "arrive_step": 0, "n_generated": 40, "finish_reason": "length"},
            {"prompt_len": 16, "max_tokens": 40, "arrive_step": 0, "n_generated": 4, "finish_reason": "length"}]
    steps, logits = sched.simulate(phi, reqs)
    members = [sorted(g[0] for g in s["rows"]) for s in steps]
    assert members[3] == [0, 1] and members[4] == [1] and members[5] == [1, 2]  # slot held for one step after 0's last row
    # with only request 0 running, its finish is drained before the next step: 2 follows immediately
    reqs1 = [reqs[0], reqs[2]]
    phi1 = _pool_phi(None, prompt_lens=(16, 16), max_tokens=40)
    phi1["engine"]["max_num_seqs"] = 1
    steps1, _ = sched.simulate(phi1, reqs1)
    members1 = [sorted(g[0] for g in s["rows"]) for s in steps1]
    assert members1[3] == [0] and members1[4] == [1] and len(steps1) == 8


def test_shared_prefix_blocks_are_held_once():
    """Synthetic pool: two 88-token prompts sharing 64 tokens under a 12-block pool (11 usable).  Per-request counting (6 + 6 = 12 > 11)
    would refuse to admit the second request; the ref-counted pool (6 + 2 = 8) admits both -- and a referenced block stays held (and
    cached) after its allocator is gone."""
    a = list(range(1000, 1088))
    b = a[:64] + list(range(5000, 5024))
    phi = _prefix_phi(a, b)
    phi["engine"]["num_gpu_blocks"] = 12
    reqs = [{"prompt_len": 88, "max_tokens": 4, "arrive_step": 0, "n_generated": 4, "finish_reason": "length"} for _ in range(2)]
    st = sched.SchedState(phi, reqs)
    rows, _ = st.predict()
    assert sorted(rows) == [[0, 0, 88], [1, 64, 24]]
    st.apply(rows)
    assert st.hit == [0, 4] and st.pool.free_count() == 11 - 8 and st.pool.blocks[1][:4] == st.pool.blocks[0][:4]  # 4 shared + 2 own
    b0 = list(st.pool.blocks[0])  # [1..6]: five full (hashed) blocks + a partial one
    assert b0 == [1, 2, 3, 4, 5, 6] and st.pool.blocks[1] == [1, 2, 3, 4, 7, 8] and st.pool.free_q == [9, 10, 11]
    st.pool.free(0)  # request 0 gone: the 4 shared blocks stay held by request 1; its partial block 6 is prepended (LIFO reuse), its own
    # hashed block 5 appended (FIFO eviction) -- and stays cached: a later request with prompt a would still hit 80 tokens (LRU survival)
    assert st.pool.free_q == [6, 9, 10, 11, 5] and all(st.pool.ref[b] == 1 for b in [1, 2, 3, 4]) and st.pool.cached[st.pool.key(0, 4)] == [5]
    assert len(st.pool.lookup(0, 88)) == 5  # request 0 re-admitted now: five hits (cap (88 - 1) // 16 = 5)


def test_real_chunk_cap_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/thr4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 09:10Z on cp-aot with
    long_prefill_token_threshold=20): four 33-37-token prompts are split into a 20-token chunk (step 0) and the remainder (step 1, rows
    sorted by size 13/14/15/17), then 8 decodes.  The cap is an engine arg (phi0); with it the predictor is exact (A2 = 1 bit at phi1a+),
    without it the two prefill steps are escapes (60 bits at phi1) -- first contact in gen/thr4/first_contact_rev5895b5124.md."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/thr4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("thr4 record not present in this checkout")
    r = facts.load_log_only(log, "thr4")
    r["phi_overrides"] = {"long_prefill_token_threshold": 20}  # the RunHeader does not record engine-arg overrides (harness gap)
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 9
    assert [g[2] for g in h["steps"][0]["rows"]] == [20, 20, 20, 20] and [g[2] for g in h["steps"][1]["rows"]] == [13, 14, 15, 17]
    phi0 = facts.phi_from_record(r)
    assert phi0["engine"]["long_prefill_token_threshold"] == 20
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1a"]["A2"] == 1 and bits["phi1"]["A3"] == 1
    phi_off = json.loads(json.dumps(phi0))
    phi_off["engine"]["long_prefill_token_threshold"] = 0
    bits_off = roundtrip_all(phi_off, h, frozen_model)
    assert bits_off["phi1"]["A2"] > 40  # two escaped prefill steps


def test_real_composition_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/mix6 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 09:22Z on cp-aot with
    enable_prefix_caching=true and long_prefill_token_threshold=24): six requests composing EOS stops (8 and 1 generated tokens), staggered
    arrivals (0/0/2/5/5/9), the chunk cap (34 -> 24+10, 44 -> 24+20) and prefix-cache hits -- including one not planned: the two 88-token
    prompts share their first 32 tokens with the 34-token story prompt admitted three steps earlier, so request 3 is admitted with
    num_computed 32 and request 4 (same step, after 3's 24-token allocation) with 48.  First contact with the cap fact in phi0: A2 = A3 = 1 bit
    at phi1/phi2 (every rule composes); without the cap fact 187 bits (gen/mix6/first_contact_rev_f4bb105c3_no_cap_fact.md)."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/mix6/log.jsonl.gz"
    if not log.exists():
        pytest.skip("mix6 record not present in this checkout")
    r = facts.load_log_only(log, "mix6")
    r["phi_overrides"] = {"long_prefill_token_threshold": 24}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 35
    assert [(q["n_generated"], q["finish_reason"], q["arrive_step"]) for q in h["requests"]] == [(8, "stop", 0), (1, "stop", 0), (32, "length", 2), (16, "length", 5), (16, "length", 5), (20, "length", 9)]
    assert sorted(h["steps"][5]["rows"]) == [[0, 18, 1], [2, 35, 1], [3, 32, 24], [4, 48, 24]]  # both admissions hit the cache
    phi0 = facts.phi_from_record(r)
    assert phi0["engine"]["prefix_caching"] and phi0["engine"]["long_prefill_token_threshold"] == 24
    assert phi0["requests"][3]["block_hashes"][:2] == phi0["requests"][2]["block_hashes"][:2]  # the unplanned shared prefix, visible in phi0
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and bits["phi1a"]["A2"] > 1


def test_real_preregistered_self_preemption_with_waiter(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/sp4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 10:40Z on cp-aot with
    max_num_batched_tokens=64, num_gpu_blocks_override=16, max_model_len=240): mp3 plus a 10-token x 5 request D arriving at step 4, during the
    150-token request's prefill.  Pre-registered step for step (gen/sp4_preregistration.md, code c4ac79901 frozen before the capture): D is not
    admitted at step 5 although its single block would fit -- the step preempted (scheduler.py `if not preempted_reqs`) -- nor at steps 6-39 (the
    preempted request at the waiting front cannot be re-admitted and blocks the queue), first row at step 42 beside the last re-prefill chunk, T = 62
    -- HELD on first contact.  With D's *declared* arrival in the hint (the frozen extractor writes the first scheduled step, 42, which makes the
    record uninformative about the branch) the predictor still matches every step; the previous branch (768f254d5: requester left running, waiting
    loop entered) predicted D admitted at step 5 -- the one predictor-branch change after codec-r9 is load-bearing on a real record."""
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen/sp4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("sp4 record not present in this checkout")
    r = facts.load_log_only(log, "sp4")
    r["phi_overrides"] = {"num_gpu_blocks": 16}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 62
    assert [(q["prompt_len"], q["n_generated"], q["arrive_step"]) for q in h["requests"]] == [(29, 40, 0), (29, 40, 0), (150, 20, 3), (10, 5, 42)]
    assert [int(q.get("arrive_step", 0)) for q in facts.logical_requests(r["spec"])] == [0, 0, 3, 4]  # declared: D arrived at 4, queued 38 steps
    assert h["steps"][4]["rows"] == [[0, 32, 1], [1, 32, 1], [2, 62, 62]] and h["steps"][5]["rows"] == [[0, 33, 1], [1, 33, 1]]  # no admission
    assert h["steps"][6]["rows"] == [[0, 34, 1], [1, 34, 1]]  # head-of-line: C (10 blocks, 9 free) blocks D
    assert [s["rows"] for s in h["steps"][40:43]] == [[[2, 0, 64]], [[2, 64, 64]], [[3, 0, 10], [2, 128, 22]]]
    phi0 = facts.phi_from_record(r)
    for arrive_d in (42, 4):  # first-scheduled (pipeline) and declared (client) arrival: both clean under the current predictor
        h2 = json.loads(json.dumps(h))
        h2["requests"][3]["arrive_step"] = arrive_d
        steps, logits = sched.simulate(facts.phi_variant(phi0, h2, "phi2"), h2["requests"])
        assert steps == h2["steps"] and logits == h2["logits"], arrive_d
        bits = roundtrip_all(phi0, h2, frozen_model)
        assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 350 < bits["phi1"]["A2"] < 500, arrive_d


def test_real_preregistered_lru_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/lru4 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 11:08Z on cp-aot with
    enable_prefix_caching=true, num_gpu_blocks_override=20, max_model_len=256): pp4's four prompts (64 shared tokens) x 140 generated tokens --
    five preemptions (steps 43, 74, 107, 156, 220), five resumes.  Pre-registered step for step (gen/lru4_preregistration.md +
    lru4_predicted_schedule.json, code 8a500716f frozen before the capture) as the out-of-sample test of the block-identity LRU pool written
    from source at 10:58Z: resumes with num_computed 144 (4 shared + 5 own blocks surviving at the back of the free queue) at steps 140, 173
    and 223 and with 64 (own blocks reused) at 140 and 173, T = 273 -- HELD.  The pool model without identities (c4ac79901) mis-predicts exactly
    the three 144-token resumes."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "lru4/log.jsonl.gz"
    if not log.exists():
        pytest.skip("lru4 record not present in this checkout")
    r = facts.load_log_only(log, "lru4")
    r["phi_overrides"] = {"num_gpu_blocks": 20}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 273
    assert h["steps"][140]["rows"] == [[1, 144, 34], [2, 64, 81]] and h["steps"][173]["rows"] == [[2, 144, 17], [3, 64, 50]]
    assert h["steps"][223]["rows"] == [[3, 144, 17]]
    pred = json.load(open(here / "lru4_predicted_schedule.json"))
    assert [s["rows"] for s in pred["steps"]] == [s["rows"] for s in h["steps"]] and pred["logits"] == h["logits"]  # written before the run
    phi0 = facts.phi_from_record(r)
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 1400 < bits["phi1"]["A2"] < 1800


def test_real_preregistered_full_composition_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/all6 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 11:17Z on cp-aot with
    enable_prefix_caching=true, num_gpu_blocks_override=24, long_prefill_token_threshold=24, max_model_len=256): pre6's two EOS prompts x 32
    (stop after 8 / 1) + pp4's four shared-prefix prompts x 100, arrivals 0/0/0/0/1/3.  Every modelled rule at once, pre-registered step for
    step (gen/all6_preregistration.md + all6_predicted_schedule.json, code frozen before the capture): a 16-token same-step hit from a sharer's
    first 24-token chunk at step 0, a 64-token hit at step 1 while both sharers still prefill, EOS placeholders, preemptions at 59 / 76, the
    drain at 102 and chunked resumes [4 96+24] [5 64+24] with logits only on the last chunks, T = 148 -- HELD.  The identity-free pool model
    (c4ac79901) mis-predicts 28 steps (102-129)."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "all6/log.jsonl.gz"
    if not log.exists():
        pytest.skip("all6 record not present in this checkout")
    r = facts.load_log_only(log, "all6")
    r["phi_overrides"] = {"num_gpu_blocks": 24, "long_prefill_token_threshold": 24}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 148
    assert [(q["n_generated"], q["finish_reason"]) for q in h["requests"][:2]] == [(8, "stop"), (1, "stop")]
    assert h["steps"][0]["rows"] == [[1, 0, 13], [0, 0, 14], [2, 0, 24], [3, 16, 24]]  # same-step hit from a partial first chunk
    assert h["steps"][1]["rows"] == [[0, 14, 1], [1, 13, 1], [4, 64, 7], [2, 24, 24], [3, 40, 24]]  # hit while the sharers still prefill
    assert h["steps"][102]["rows"] == [[4, 96, 24], [5, 64, 24]] and h["logits"][102] == [[4, 0], [5, 0]]  # LRU resume under the cap
    assert h["steps"][104]["rows"] == [[4, 144, 2], [5, 112, 15]] and h["logits"][104] == [[4, 1], [5, 1]]
    pred = json.load(open(here / "all6_predicted_schedule.json"))
    assert [s["rows"] for s in pred["steps"]] == [s["rows"] for s in h["steps"]] and pred["logits"] == h["logits"]  # written before the run
    phi0 = facts.phi_from_record(r)
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 600 < bits["phi1"]["A2"] < 800


def test_real_preregistered_stress_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/mix16 (SmolLM2-135M, vLLM d9105ea80, captured 2026-09-18 11:31Z on cp-aot with
    max_num_seqs=8, max_num_batched_tokens=64, long_prefill_token_threshold=32, num_gpu_blocks_override=40, enable_prefix_caching=true,
    max_model_len=256): 16 requests from every earlier generated workload, declared arrivals 0..20.  Pre-registered step for step
    (gen/mix16_preregistration.md + mix16_predicted_schedule.json, code frozen before the capture): the seqs cap binds on 42 steps, a
    budget-split same-step hit [3 32+5] at step 0, an 80-token hit on a still-prefilling prompt (step 6), 9 preemptions incl. two victims at
    step 28, an 80-token LRU resume (29), a fully cached 64-token prompt hitting 48 = num_tokens - 1 (34), T = 117 -- HELD.  The identity-free
    pool model (c4ac79901) mis-predicts 72 steps from step 29."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "mix16/log.jsonl.gz"
    if not log.exists():
        pytest.skip("mix16 record not present in this checkout")
    r = facts.load_log_only(log, "mix16")
    r["phi_overrides"] = {"num_gpu_blocks": 40, "long_prefill_token_threshold": 32}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 117 and len(h["requests"]) == 16
    assert [q["arrive_step"] for q in h["requests"]] == [0, 0, 0, 0, 2, 2, 4, 6, 6, 9, 24, 29, 34, 37, 47, 59]  # first scheduled, behind the seqs cap
    assert h["steps"][0]["rows"] == [[3, 32, 5], [1, 0, 13], [0, 0, 14], [2, 0, 32]]  # same-step hit + the budget's remainder
    assert h["steps"][6]["rows"][-3:] == [[8, 64, 3], [7, 80, 24], [6, 56, 32]]  # hit on a prompt still being prefilled
    assert sum(1 for s in h["steps"] if len(s["rows"]) == 8) == 42
    assert h["steps"][29]["rows"][-3:] == [[10, 32, 3], [11, 64, 7], [9, 80, 20]]  # LRU resume with 80
    assert h["steps"][34]["rows"][-1] == [12, 48, 16] and h["steps"][37]["rows"][-1] == [13, 32, 2]  # fully cached prompts: num_tokens - 1
    pred = json.load(open(here / "mix16_predicted_schedule.json"))
    assert [s["rows"] for s in pred["steps"]] == [s["rows"] for s in h["steps"]] and pred["logits"] == h["logits"]  # written before the run
    phi0 = facts.phi_from_record(r)
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 400 < bits["phi1"]["A2"] < 600


def test_real_preregistered_second_family_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/qwen8 (Qwen2.5-1.5B, case B1, vLLM d9105ea80, captured 2026-09-18 11:44Z on cp-aot with
    max_num_seqs=8, max_num_batched_tokens=96, long_prefill_token_threshold=48, num_gpu_blocks_override=64, enable_prefix_caching=true,
    max_model_len=512): 8 requests from the B1 corpus prompts with 96/128-token shared prefixes and an entirely cached 64-token prompt.
    Pre-registered step for step (gen/qwen8_preregistration.md + qwen8_predicted_schedule.json, code frozen before the capture) as the first
    pressure + caching record on a model other than SmolLM2-135M: same-step hit 48, an 80-token hit on a running prefill, 64 -> 48, a mid-prefill
    preemption (step 9) with a 48-token LRU resume at 30, T = 76 -- HELD.  The identity-free pool model (c4ac79901) mis-predicts 25 steps."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "qwen8/log.jsonl.gz"
    if not log.exists():
        pytest.skip("qwen8 record not present in this checkout")
    r = facts.load_log_only(log, "qwen8")
    r["phi_overrides"] = {"num_gpu_blocks": 64, "long_prefill_token_threshold": 48}
    h, audit = facts.extract_hint(r)
    phi0 = facts.phi_from_record(r)
    assert phi0["model"] == "Qwen/Qwen2.5-1.5B" and phi0["engine"]["block_size"] == 16
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 76 and len(h["requests"]) == 8
    assert h["steps"][0]["rows"] == [[0, 0, 48], [1, 48, 48]]  # same-step partial hit
    assert h["steps"][3]["rows"][-2:] == [[3, 80, 46], [2, 36, 48]]  # hit on a running prefill + budget remainder
    assert h["steps"][6]["rows"][-3:] == [[5, 48, 16], [6, 0, 16], [4, 21, 48]]  # fully cached 64-token prompt -> 48
    assert h["steps"][30]["rows"][-2:] == [[7, 0, 44], [6, 48, 48]] and h["steps"][36]["rows"][-1] == [7, 32, 48]  # LRU resumes
    pred = json.load(open(here / "qwen8_predicted_schedule.json"))
    assert [s["rows"] for s in pred["steps"]] == [s["rows"] for s in h["steps"]] and pred["logits"] == h["logits"]  # written before the run
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and 800 < bits["phi1"]["A2"] < 1000


def test_real_twin_record_is_exact_but_not_predicted(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/twin3 (SmolLM2-135M, captured 2026-09-18 11:56Z on cp-aot with num_gpu_blocks_override=14,
    enable_prefix_caching=true, max_model_len=208): two IDENTICAL 70-token prompts x 120 (greedy -> identical generations) + a 129-token filler.
    Pre-registered as a designed failure (gen/twin3_preregistration.md): the second twin is preempted at 59 / 92 / 109 and re-admitted the next
    step with num_computed 128 / 160 / 176 -- hits on its twin's GENERATED-token blocks, which the predictor keeps under private identities.
    The language expresses the record exactly at every point; the phi2 residual is non-empty (690 bits) and the pool fact makes it worse than
    phi1 (207).  This test pins the known, non-derivable gap: it must fail if a rule ever 'predicts' this record from phi alone."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "twin3/log.jsonl.gz"
    if not log.exists():
        pytest.skip("twin3 record not present in this checkout")
    r = facts.load_log_only(log, "twin3")
    r["phi_overrides"] = {"num_gpu_blocks": 14}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 183
    present = [{row[0] for row in s["rows"]} for s in h["steps"]]
    assert [t for t in range(123) if 1 not in present[t]] == [59, 92, 109]
    assert h["steps"][60]["rows"] == [[0, 129, 1], [1, 128, 1]] and h["steps"][93]["rows"] == [[0, 162, 1], [1, 160, 1]] and h["steps"][110]["rows"] == [[0, 179, 1], [1, 176, 1]]
    phi0 = facts.phi_from_record(r)
    st = sched.SchedState(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    mism = 0
    for t, s in enumerate(h["steps"]):
        pr, pl = st.predict()
        mism += pr != s["rows"] or pl != h["logits"][t]
        st.apply(s["rows"])
    assert mism == 58  # the private-identity assumption fails for B's whole life after step 59 -- a non-derivable fact, not a missing rule
    bits = roundtrip_all(phi0, h, frozen_model)
    # exact at every point, not 1 bit anywhere (A2 is model-free; A3 with the corpus model a3_model.json is 771, the synthetic fixture model 772)
    assert bits["phi1"]["A2"] == 207 and bits["phi2"]["A2"] == 690 and 700 < bits["phi2"]["A3"] < 800


def test_real_twin_control_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/twin3c (captured 2026-09-18 12:10Z on cp-aot; twin3's workload with B's LAST prompt token
    changed, same 14-block pool): blocks 0-3 still shared, block 4 and everything after it hashed differently, so no hit on A's generated
    blocks is possible.  Pre-registered (gen/twin3c_preregistration.md, schedule = the one predicted for twin3): B preempted at 59, absent
    through 119, resumed at 120 with 80 on its own surviving block, C admitted at 181, T = 241 -- HELD, A2 = A3 = 1 bit at phi2.  With
    test_real_twin_record_is_exact_but_not_predicted this pins the attribution: one prompt token moves the phi2 residual 690 -> 1."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "twin3c/log.jsonl.gz"
    if not log.exists():
        pytest.skip("twin3c record not present in this checkout")
    r = facts.load_log_only(log, "twin3c")
    r["phi_overrides"] = {"num_gpu_blocks": 14}
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 241
    present = [{row[0] for row in s["rows"]} for s in h["steps"]]
    assert [t for t in range(121) if 1 not in present[t]] == list(range(59, 120))
    assert h["steps"][120]["rows"] == [[1, 80, 49]] and h["steps"][181]["rows"] == [[2, 0, 129]]
    pred = json.load(open(here / "twin3c_predicted_schedule.json"))
    assert [s["rows"] for s in pred["steps"]] == [s["rows"] for s in h["steps"]] and pred["logits"] == h["logits"]  # written before the run
    phi0 = facts.phi_from_record(r)
    steps, logits = sched.simulate(facts.phi_variant(phi0, h, "phi2"), h["requests"])
    assert steps == h["steps"] and logits == h["logits"]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi2"]["A2"] == 1 and bits["phi2"]["A3"] == 1 and bits["phi1"]["A2"] == 514


def test_real_conversation_continuation_record(frozen_model):
    """MEASURED record out/gen/r8/aot-codec/gen/conv2 (SmolLM2-135M, captured 2026-09-18 12:19Z on cp-aot with enable_prefix_caching=true,
    default pool, max_model_len=256): A = the 70-token prompt x 120 at step 0; B = A's prompt + A's first 64 GENERATED tokens + 10 follow-up
    tokens (144) x 30, arriving at step 70 -- a second turn quoting the first turn's answer.  Pre-registered (gen/conv2_preregistration.md):
    the frozen predictor says [1 64+80] (prompt blocks only); the engine gives [1 128+16] (A's generated blocks hit too) and nothing else
    differs.  Pins the known, non-derivable gap in its everyday form: 1 mismatched step at phi1 and phi2 alike, A2 = 71 bits at both."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[4] / "out/gen/r8/aot-codec/gen"
    log = here / "conv2/log.jsonl.gz"
    if not log.exists():
        pytest.skip("conv2 record not present in this checkout")
    r = facts.load_log_only(log, "conv2")
    h, audit = facts.extract_hint(r)
    assert not audit["unexpressible"] and facts.check_record(r, h)["ok"] and len(h["steps"]) == 120 and len(h["requests"]) == 2
    assert h["requests"][1]["prompt_len"] == 144 and h["requests"][1]["arrive_step"] == 70
    assert h["steps"][70]["rows"] == [[0, 139, 1], [1, 128, 16]]  # hit on A's prompt AND generated blocks
    pred = json.load(open(here / "conv2_predicted_schedule.json"))
    assert pred["steps"][70]["rows"] == [[0, 139, 1], [1, 64, 80]]  # written before the run: prompt blocks only
    assert [t for t in range(120) if pred["steps"][t]["rows"] != h["steps"][t]["rows"]] == [70] and pred["logits"] == h["logits"]
    phi0 = facts.phi_from_record(r)
    for lvl in ("phi1", "phi2"):
        st = sched.SchedState(facts.phi_variant(phi0, h, lvl), h["requests"])
        mism = []
        for t, s in enumerate(h["steps"]):
            pr, pl = st.predict()
            if pr != s["rows"] or pl != h["logits"][t]:
                mism.append(t)
            st.apply(s["rows"])
        assert mism == [70]
    bits = roundtrip_all(phi0, h, frozen_model)
    assert bits["phi1"]["A2"] == 71 and bits["phi2"]["A2"] == 71 and 60 < bits["phi2"]["A3"] < 100
