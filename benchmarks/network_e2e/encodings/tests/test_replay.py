"""The scheduler replay: the replay reproduces vLLM's own scheduler, and the residual advice decodes back to the trace.

The fixtures are step traces of vLLM d9105ea80's `Scheduler` and `AsyncScheduler` on a fake executor (`vllm_trace.py
tests/fixtures --requests 14 --seed 3`, the cases named by the file): preemption under KV pressure in both engine loops,
prefix caching, and a light load whose requests join part-full batches and an idle engine. Those of `tests/fixtures_draft`
are the same scheduler's with a draft of 1, 3 and 5 tokens (`vllm_trace.sh --draft`: the cases with a draft, at their own
size and seed 0)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


R = _load("replay")
CASES = sorted(p.stem for p in FIXTURES.glob("*.jsonl"))
DRAFT = Path(__file__).parent / "fixtures_draft"
DRAFT_CASES = sorted(p.stem for p in DRAFT.glob("*.jsonl"))


def trace(name: str, d: Path = FIXTURES):
    tok = d / f"{name}.tokens.json"
    return R.read_step_trace(d / f"{name}.jsonl", tokens=json.loads(tok.read_text()) if tok.exists() else None)


def test_fixtures_cover_the_served_paths():
    traces = {n: trace(n) for n in CASES}
    assert {t.engine.async_scheduling for t in traces.values()} == {False, True}
    assert any(t.engine.enable_prefix_caching and any(s.cache_hit for s in t.steps) for t in traces.values())
    for asy in (False, True):
        assert any(t.engine.async_scheduling == asy and any(s.preempted for s in t.steps) for t in traces.values())


@pytest.mark.parametrize("name", CASES)
def test_replay_matches_vllm_step_for_step(name):
    t = trace(name)
    rp = R.replay(t.engine, t.requests, admit=t.before_step)
    m = R.compare(rp.steps, t.steps)
    assert m.first_mismatch is None and m.ordered == m.steps == len(t.steps)
    assert [s.preempted for s in rp.steps] == [sorted(s.preempted) for s in t.steps]
    assert rp.finish == ["stop" if r.eos else "length" for r in t.requests]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("contexts", R.CONTEXTS)
def test_residual_decodes_to_the_trace(name, contexts):
    t = trace(name)
    code = R.encode(t, contexts=contexts)
    m = R.compare(R.decode(t.certified(code.bucket_s), code).steps, t.steps)
    assert m.first_mismatch is None and m.matched == len(t.steps)
    assert all(len(b) <= code.ideal[w] + 2 for w, b in code.bits.items())


def test_wrong_advice_changes_the_schedule():
    t = trace("async-light")
    code = R.encode(t)
    (w, bits), = code.bits.items()
    flipped = R.Code(code.contexts, code.bucket_s, code.window_buckets, {w: [1 - bits[0]] + bits[1:]}, code.ideal)
    assert R.compare(R.decode(t.certified(), flipped).steps, t.steps).first_mismatch is not None


def test_egress_tells_light_load_arrivals_apart():
    """At light load a request joins a running batch at an open step; its outputs' release buckets against a running
    request's say which step, where the ingress clock leaves about log2(steps per bucket) bits."""
    t = trace("async-light")
    bits = {c: sum(len(b) for b in R.encode(t, contexts=c).bits.values()) for c in R.CONTEXTS}
    assert bits["egress"] < bits["clocked"] < bits["plain"]


@pytest.mark.parametrize("name", CASES)
def test_whatif_is_the_served_run_for_the_request_and_those_ahead(name):
    """The egress context's premise: forked where a request is admitted, with nothing arriving after it, the replay
    emits that request's outputs in the steps the served run did, alongside the same outputs of every earlier request."""
    t = trace(name)
    first, kept = t.first_scheduled(), t.kept_steps()
    at: dict[int, dict[int, int]] = {}
    for j, ks in enumerate(kept):
        for m, s in enumerate(ks):
            at.setdefault(s, {})[j] = m
    checked = 0

    def decide(i, k, o):
        nonlocal checked
        if k < first[i]:
            return False
        for e in o.whatif(True, t.requests[i].output_len):
            if i in e:
                s = kept[i][e[i]]
                assert {j: m for j, m in e.items() if j <= i} == {j: m for j, m in at[s].items() if j <= i}
                checked += 1
        return True

    R.replay(t.engine, t.requests, decide=decide)
    assert checked


@pytest.mark.parametrize("name", CASES)
def test_kept_outputs_are_the_replays_emissions(name):
    t = trace(name)
    kept = t.kept_steps()
    assert [len(k) for k in kept] == [r.output_len for r in t.requests]
    assert all(t.emits[s] for k in kept for s in k)
    assert all(a < b for k in kept for a, b in zip(k, k[1:]))


def test_table_counts_every_request_and_step():
    t = trace("async-light")
    tb = R.table(t)
    assert sum(w["requests"] for w in tb["windows"]) == len(t.requests)
    assert sum(w["steps"] for w in tb["windows"]) == len(t.steps)
    assert all(v["first_mismatch"] is None for v in tb["decoded"].values())
    for w in tb["windows"]:
        assert w["adaptive_egress_bits"] < w["v0_bits"] // 100


def corpus_dir(t, d: Path) -> Path:
    """`t` written as one trace-corpus trace (`benchmarks/network_e2e/traces/summary.py`'s joined files), with a step
    that scheduled nothing, as the hook records one."""
    seen: set[int] = set()
    with open(d / "steps.jsonl", "w") as f:
        f.write(json.dumps({"step": 0, "total_num_scheduled_tokens": 0, "new": [], "cached": []}) + "\n")
        for k, s in enumerate(t.steps):
            new, cached = [], []
            for i, start, n in s.rows:
                row = {"id": t.ids[i], "num_computed_tokens": start, "num_scheduled_tokens": n, "resumed": i in s.resumed}
                (cached if i in seen else new).append(row)
                seen.add(i)
            f.write(json.dumps({"step": k + 1, "total_num_scheduled_tokens": sum(n for _, _, n in s.rows), "new": new,
                                "cached": cached, "batch": [t.ids[i] for i, _, _ in s.rows],
                                "preempted": [t.ids[i] for i in s.preempted], "sampled": [1] if t.emits[k] else [],
                                "t_output": 1e9 + t.release_t[k]}) + "\n")
    kept = t.kept_steps()
    with open(d / "requests.jsonl", "w") as f:
        for i, r in enumerate(t.requests):
            chunks = [[t.release_t[s] - t.ingress_t[i] + 0.001, 1] for s in kept[i]]
            for k in range(1, len(chunks) - 1, 7):          # the streamer merged two tokens into one chunk
                chunks[k] = [chunks[k + 1][0], 2]
                chunks[k + 1] = [chunks[k + 1][0], 0]
            f.write(json.dumps({"id": t.ids[i], "prompt_len": r.prompt_len, "max_tokens": r.max_tokens,
                                "output_len": r.output_len, "finish_reason": "stop" if r.eos else "length", "eos": r.eos,
                                "prompt_ids": [3] * r.prompt_len, "output_ids": [4] * r.output_len,
                                "send_unix": 1e9 + t.ingress_t[i], "chunks": [[0.0, 0]] + chunks}) + "\n")
    e = t.engine
    argv = ["--max-model-len", str(e.max_model_len), "--max-num-seqs", str(e.max_num_seqs), "--max-num-batched-tokens",
            str(e.max_num_batched_tokens), "--no-async-scheduling", "--enable-chunked-prefill"]
    (d / "meta.json").write_text(json.dumps({"server_argv": argv}))
    L, c = e.max_model_len, R.cdiv(e.max_model_len, e.block_size)
    (d / "server.log").write_text(f"INFO GPU KV cache size: {int(e.num_gpu_blocks / c * L):,} tokens, Maximum ...\n")
    return d


def test_reads_a_trace_corpus_trace(tmp_path):
    t = trace("sync-kv-pressure")
    d = corpus_dir(t, tmp_path)
    c = R.read_corpus_trace(d)
    assert c.engine == t.engine and c.ids == t.ids and len(c.steps) == len(t.steps)
    assert min(c.certified().ingress) == 0
    rp = R.replay(c.engine, c.requests, admit=c.before_step)
    assert R.compare(rp.steps, c.steps).ordered == len(t.steps)
    client = R.read_corpus_trace(d, egress="client")
    assert [len(ts) for ts in client.output_times()] == [r.output_len for r in t.requests]
    assert client.output_times() != c.output_times()
    for x in (c, client):
        assert R.compare(R.decode(x.certified(), R.encode(x)).steps, x.steps).first_mismatch is None


def test_the_kv_pool_is_the_hooks_record_of_the_schedulers(tmp_path):
    """`engine.json` (the trace hook's record of the scheduler's pool) gives the block size and count over `server.log`'s
    capacity line, which can't tell a 64-token block from a 16-token one; arguments win over both, and a pool of more
    than one KV cache group is refused by name."""
    t = trace("sync-kv-pressure")
    d = corpus_dir(t, tmp_path)
    e = t.engine
    (d / "server.log").write_text("")
    (d / "engine.json").write_text(json.dumps({"block_size": e.block_size, "num_gpu_blocks": e.num_gpu_blocks,
                                               "kv_cache_groups": 1, "scheduler": "Scheduler"}))
    c = R.read_corpus_trace(d)
    assert c.engine == e
    assert R.compare(R.replay(c.engine, c.requests, admit=c.before_step).steps, c.steps).ordered == len(t.steps)
    (d / "engine.json").write_text(json.dumps({"block_size": 64, "num_gpu_blocks": 7, "kv_cache_groups": 1}))
    pool = R.read_corpus_trace(d).engine
    assert (pool.block_size, pool.num_gpu_blocks) == (64, 7)
    assert R.read_corpus_trace(d, block_size=e.block_size, num_gpu_blocks=e.num_gpu_blocks).engine == e
    (d / "engine.json").write_text(json.dumps({"block_size": 64, "num_gpu_blocks": 7, "kv_cache_groups": 2}))
    with pytest.raises(ValueError, match="2 KV cache groups; the replay models one pool"):
        R.read_corpus_trace(d)


def test_warden_ingress_is_the_engines_arrival_order(tmp_path):
    """`ingress="warden"` reads the engine's own `arrival` stamp (the server's body read, the order vLLM schedules in):
    requests the client sent out of the engine's order cost order bits at the client's send and none there, the floor a
    server admitting in the warden's release order would give."""
    t = trace("sync-kv-pressure")
    d = corpus_dir(t, tmp_path)
    recs = [json.loads(line) for line in (d / "requests.jsonl").read_text().splitlines()]
    for r in recs:
        r["arrival"] = r["send_unix"]
    for k in range(4, len(recs), 5):
        recs[k]["send_unix"] = recs[k - 3]["arrival"] - 1e-3          # sent early, stamped late: a delay after the wire
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    client, warden = R.read_corpus_trace(d), R.read_corpus_trace(d, ingress="warden")
    n = len(t.requests)
    assert warden.ingress_t == sorted(warden.ingress_t) == [r["arrival"] for r in recs]
    assert R.order_bits(warden) <= 0.5 * R.math.log2(n) + 1 < R.order_bits(client)
    assert R.compare(R.decode(warden.certified(), R.encode(warden)).steps, warden.steps).first_mismatch is None
    with pytest.raises(ValueError, match="ingress"):
        R.read_corpus_trace(d, ingress="server")


def test_order_bits_charge_only_reordering():
    t = trace("sync-kv-pressure")
    assert sorted(t.ingress_t) == t.ingress_t
    n = len(t.requests)
    assert R.order_bits(t) <= 0.5 * R.math.log2(n) + 1
    t.ingress_t[0], t.ingress_t[-1] = t.ingress_t[-1], t.ingress_t[0]
    assert R.order_bits(t) > 0.5 * R.math.log2(n) + 1 + R.gamma_bits(n - 1)


def test_v0_advice_is_valid_advice_v0(tmp_path):
    v0 = R.load_v0(os.environ.get("VERITY_ENCODINGS_ADVICE_V0"))
    if v0 is None:
        pytest.skip(f"advice v0 (advice.py) isn't at {R.V0_PATH} (VERITY_ENCODINGS_ADVICE_V0: a path to its advice.py)")
    for name in CASES:
        t = trace(name)
        adv = R.v0_advice(t, v0)
        assert len(adv.steps) == len(t.steps) and len(adv.release) == sum(r.output_len for r in t.requests)
        assert sum(R.v0_parts(v0, adv).values()) == v0.total_bits(adv)
    c = R.read_corpus_trace(corpus_dir(trace("sync-kv-pressure"), tmp_path))
    assert sorted(R.v0_advice(c, v0).release) == sorted(b for bs in c.certified().release for b in bs)


def test_arrivals_into_a_quiet_engine_cost_nothing():
    e = R.Engine(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=64)
    reqs = [R.RequestIO(prompt_len=20, max_tokens=5, output_len=3, eos=True) for _ in range(3)]
    rp = R.replay(e, reqs, decide=lambda i, k, _: False)
    assert rp.forced == [True] * 3 and rp.asked[0] == [] and all(rp.asked[1:])
    assert R.gamma_values(rp) == [None] + [len(a) for a in rp.asked[1:]]


def test_admission_steps_round_trip_through_decide():
    """`replay(admit=r.admitted, passed=r.passed)` reproduces its decide run `r`, step for step, over 100 seeds of four
    small engines (24 requests, 4-token blocks), sync and async, prefix caching off and on. Under async scheduling the
    engine core can call `schedule` again, scheduling nothing, at the same step count while a step is in flight, and
    `admitted` alone can't say whether a request arrived before that call or after it: some of these async runs
    schedule differently without `passed`, which the test also requires."""
    engines = [(False, 16, 32, 4), (True, 16, 32, 4), (False, 24, 48, 4), (False, 16, 32, 2)]
    split = preempted = 0
    for seed in range(100):
        for asy in (False, True):
            for caching, blocks, budget, seqs in engines:
                rng = random.Random(seed)
                e = R.Engine(max_num_seqs=seqs, max_num_batched_tokens=budget, max_model_len=64, num_gpu_blocks=blocks,
                             block_size=4, enable_prefix_caching=caching, async_scheduling=asy)
                reqs = []
                for _ in range(24):
                    p, out, eos = rng.randint(2, 20), rng.randint(1, 12), rng.random() < 0.7
                    toks = tuple(rng.randint(0, 3) for _ in range(p + out)) if caching else None
                    reqs.append(R.RequestIO(prompt_len=p, max_tokens=out + (rng.randint(0, 2) if eos else 0),
                                            output_len=out, eos=eos, tokens=toks))
                a = R.replay(e, reqs, decide=lambda i, k, _: rng.random() < 0.4)
                b = R.replay(e, reqs, admit=a.admitted, passed=a.passed)
                assert b.steps == a.steps, (seed, asy, caching, blocks, budget, seqs)
                assert (b.admitted, b.passed, b.finish) == (a.admitted, a.passed, a.finish)
                preempted += any(s.preempted for s in a.steps)
                if asy and any(a.passed):
                    split += R.replay(e, reqs, admit=a.admitted).steps != a.steps
    assert split and preempted


def test_passed_is_checked():
    e = R.Engine(max_num_seqs=2, max_num_batched_tokens=32, max_model_len=64, num_gpu_blocks=16, block_size=4)
    reqs = [R.RequestIO(prompt_len=8, max_tokens=2, output_len=2, eos=False) for _ in range(2)]
    with pytest.raises(ValueError, match="only with admit"):
        R.replay(e, reqs, decide=lambda i, k, _: True, passed=[0, 0])
    with pytest.raises(ValueError, match="one int ≥ 0"):
        R.replay(e, reqs, admit=[0, 0], passed=[0])
    with pytest.raises(ValueError, match="nondecreasing"):
        R.replay(e, reqs, admit=[0, 0], passed=[1, 0])


def test_coder_round_trips_within_two_bits_of_ideal():
    rng = random.Random(11)
    for _ in range(100):
        n, p = rng.randint(0, 300), rng.random() ** 3
        bits = [rng.random() < p for _ in range(n)]
        ctxs = [rng.randint(0, 3) for _ in range(n)]
        model, enc, ideal = R._KT(), R._Encoder(), 0.0
        for b, c in zip(bits, ctxs):
            f0, f1 = model.freqs(c)
            ideal -= R.math.log2((f1 if b else f0) / (f0 + f1))
            enc.put(b, f0, f1)
            model.update(c, b)
        code = enc.finish()
        model, dec, out = R._KT(), R._Decoder(code), []
        for c in ctxs:
            b = dec.get(*model.freqs(c))
            model.update(c, b)
            out.append(b)
        assert out == bits and len(code) <= ideal + 2


def test_refuses_what_it_does_not_model():
    with pytest.raises(ValueError, match="FCFS"):
        R.Engine(max_num_seqs=1, max_num_batched_tokens=8, max_model_len=8, num_gpu_blocks=4, policy="priority")
    recs = [json.loads(line) for line in (FIXTURES / "async-light.jsonl").read_text().splitlines()]
    arrive = next(r for r in recs if r["kind"] == "arrive")
    arrive["sampling"]["stop"] = ["\n"]
    with pytest.raises(ValueError, match="stop strings"):
        R.load_step_trace(recs)
    recs = [json.loads(line) for line in (FIXTURES / "async-light.jsonl").read_text().splitlines()]
    recs.append({"kind": "finish", "request": arrive["request"], "reason": "abort", "outside": True})
    with pytest.raises(ValueError, match="abort"):
        R.load_step_trace(recs)


# --- with a draft ------------------------------------------------------------------------------------------------------


def test_draft_fixtures_cover_the_draft_paths():
    """The draft fixtures are vLLM's scheduler under `mtp`'s settings, and at every K they hold the rows the draft rules
    decide: every kept count, a row whose drafts the context limit cut and one whose drafts the budget cut, a new or
    resumed request's padded row in both loops, preemption and a prefix hit in both loops; and among them requests
    that arrived after an empty schedule."""
    traces = {n: trace(n, DRAFT) for n in DRAFT_CASES}
    got = {n: _load("vllm_trace").exercised(t) for n, t in traces.items()}

    def some(what, **where):
        return any(what(n) for n, t in traces.items() if all(getattr(t.engine, k) == v for k, v in where.items()))

    for n, t in traces.items():
        K, cfg = t.engine.num_speculative_tokens, t.header["config"]
        assert type(t.engine) is R.DraftEngine and cfg["speculative"]["method"] == "mtp"
        assert (cfg["num_lookahead_tokens"], cfg["num_prefill_lookahead"], cfg["use_eagle_block_drop"],
                cfg["draft_slots"]) == (K, 1, True, 0)
        assert all(got[n]["kept"].values()), n
        assert not t.engine.enable_prefix_caching or any(s.cache_hit for s in t.steps), n
    for K in (1, 3, 5):
        assert some(lambda n: got[n]["cut_budget"], num_speculative_tokens=K)
        for asy in (False, True):
            at = dict(num_speculative_tokens=K, async_scheduling=asy)
            assert some(lambda n: got[n]["padded"], **at), at
            for cache in (False, True):
                assert some(lambda n: got[n]["cut_context"], **at, enable_prefix_caching=cache), (at, cache)
                assert some(lambda n: any(s.preempted for s in traces[n].steps), **at, enable_prefix_caching=cache), (at, cache)
    assert some(lambda n: got[n]["after_empty"])


@pytest.mark.parametrize("name", DRAFT_CASES)
def test_replay_matches_vllm_with_a_draft_step_for_step(name):
    """Every step's rows in the scheduler's order, the drafts each row carried, and what it preempted and hit, from the
    engine, the requests with the drafts each emitting row kept, and when each request was added."""
    t = trace(name, DRAFT)
    rp = R.replay(t.engine, t.requests, admit=t.before_step, passed=t.passed)
    assert len(rp.steps) == len(t.steps)
    for k, (a, b) in enumerate(zip(rp.steps, t.steps)):
        # the trace records a new request's hit; a resumed request's shows in its row's start
        assert (a.rows, a.drafts, sorted(a.preempted), {i: h for i, h in a.cache_hit.items() if i in a.new}, sorted(a.new),
                sorted(a.resumed)) == (b.rows, b.drafts, sorted(b.preempted), b.cache_hit, sorted(b.new), sorted(b.resumed)), k
    assert rp.finish == ["stop" if r.eos else "length" for r in t.requests]


@pytest.mark.parametrize("name", DRAFT_CASES)
def test_kept_outputs_with_a_draft_are_the_traces_emissions(name):
    t = trace(name, DRAFT)
    assert [len(k) for k in t.kept_steps()] == [r.output_len for r in t.requests]
    assert [len(r.kept) + sum(r.kept) for r in t.requests] == [r.output_len for r in t.requests]


@pytest.mark.parametrize("name", DRAFT_CASES)
@pytest.mark.parametrize("contexts", R.CONTEXTS)
def test_residual_decodes_to_the_trace_with_a_draft(name, contexts):
    """Where the arrivals are not certified, the residual advice is still each request's admission alone: with the
    drafts each row kept beside the certified I/O, it decodes back to the trace's steps."""
    t = trace(name, DRAFT)
    code = R.encode(t, contexts=contexts)
    back = R.decode(t.certified(code.bucket_s), code)
    assert [(s.rows, s.drafts) for s in back.steps] == [(s.rows, s.drafts) for s in t.steps]
    assert all(len(b) <= code.ideal[w] + 2 for w, b in code.bits.items())


@pytest.mark.parametrize("name", DRAFT_CASES)
def test_kept_count_is_charged_at_its_code_or_derived(name):
    """The kept count's class, both ways: charged, the bits of the counts' code, which decode back to the trace's counts
    from K and the output lengths alone, and from which the replay gives the trace's steps; derived, nothing is sent.
    The entropy of the counts each alone, at most log2(K + 1) a row, is beside it and is not what is charged."""
    t = trace(name, DRAFT)
    K = t.engine.num_speculative_tokens
    # the trace's own record: a row's `drafts`, and `emitted` less the row's own token
    kept = [em[i] - 1 for st, em in zip(t.steps, t.emitted) for i, d in st.drafts.items() if d and em.get(i)]
    b = R.kept_bits(t)
    assert (b["K"], b["rows"], b["counts"]) == (K, len(kept), [kept.count(k) for k in range(K + 1)])
    assert b["derived"] == 0 and b["max"] == math.log2(K + 1)
    assert b["entropy"] == pytest.approx(-sum(c / len(kept) * math.log2(c / len(kept)) for c in b["counts"] if c))
    assert 0 < b["entropy"] <= b["max"] and b["alone"] == pytest.approx(b["rows"] * b["entropy"])
    bits, ideal = R.kept_encode(K, t.requests)
    assert b["charged"] == len(bits) <= ideal + 2 and b["ideal"] == ideal
    back = R.kept_decode(K, [q.output_len for q in t.requests], bits)
    assert back == [q.kept for q in t.requests]
    reqs = [dataclasses.replace(q, kept=k) for q, k in zip(t.requests, back)]
    again = R.replay(t.engine, reqs, admit=t.before_step, passed=t.passed)
    assert [(s.rows, s.drafts) for s in again.steps] == [(s.rows, s.drafts) for s in t.steps]
    d = R.draft_table(t, contexts=R.CONTEXTS)
    assert d["kept_count"] == b and not d["tick"] and (d["requests"], d["steps"]) == (len(t.requests), len(t.steps))
    assert d["arrivals"] == {c: sum(len(x) for x in R.encode(t, contexts=c).bits.values()) for c in R.CONTEXTS}


def test_kept_count_refuses_a_trace_it_cannot_read_it_from():
    t = trace(DRAFT_CASES[0], DRAFT)
    with pytest.raises(ValueError, match="no draft"):
        R.kept_bits(trace(CASES[0]))
    with pytest.raises(ValueError, match="emitted"):
        R.kept_bits(dataclasses.replace(t, emitted=None))
    other = dataclasses.replace(t, steps=[dataclasses.replace(t.steps[0], rows=t.steps[0].rows[:-1]), *t.steps[1:]])
    with pytest.raises(ValueError, match="the replay diverges from the trace at step 0"):
        R.kept_bits(other)


def test_kept_code_round_trips_and_pays_less_for_runs():
    """The kept count's code on counts no fixture has: outputs of every length from none, a request's last rows, whose
    counts its output bounds, and counts that come in runs, which cost less than the same counts shuffled."""
    rng = random.Random(7)

    def requests(K: int, stay: float, n: int = 40) -> list:
        out = []
        for _ in range(n):
            left, kept, c = rng.randint(0, 200), [], rng.randint(0, K)
            while left:
                c = c if rng.random() < stay else rng.randint(0, K)
                kept.append(min(c, left - 1))
                left -= 1 + kept[-1]
            out.append(R.RequestIO(prompt_len=1, max_tokens=256, output_len=len(kept) + sum(kept), eos=False,
                                   kept=tuple(kept)))
        return out

    for K in (1, 2, 5, 8):
        for stay in (0.0, 0.5, 0.95):
            reqs = requests(K, stay)
            bits, ideal = R.kept_encode(K, reqs)
            assert R.kept_decode(K, [q.output_len for q in reqs], bits) == [q.kept for q in reqs]
            assert len(bits) <= ideal + 2
    runs = requests(5, 0.95, n=200)
    shuffled = []
    for q in runs:
        kept = list(q.kept)
        rng.shuffle(kept)
        shuffled.append(dataclasses.replace(q, kept=tuple(kept)))
    assert len(R.kept_encode(5, runs)[0]) < 0.8 * len(R.kept_encode(5, shuffled)[0])
    none = [R.RequestIO(prompt_len=1, max_tokens=4, output_len=0, eos=False, kept=())]
    assert R.kept_encode(5, none) == ([], 0.0) and R.kept_decode(5, [0], []) == [()]
    with pytest.raises(ValueError, match="add up to 4 outputs, not its 3"):
        R.kept_encode(5, [R.RequestIO(prompt_len=1, max_tokens=4, output_len=3, eos=False, kept=(3,))])


def _second_serve(recs: list[dict], after: int, empty: int) -> list[dict]:
    """`recs`' serve once more by the same engine, from step `after` on and `empty` empty schedules after its last step:
    its requests renamed, its steps and arrivals moved."""
    f = "{}-again".format
    out = []
    for r in recs[1:]:
        if r["kind"] == "arrive":
            out.append(r | {"request": f(r["request"]), "before_step": r["before_step"] + after,
                            "after_empty": (r.get("after_empty") or 0) + (0 if r["before_step"] else empty)})
        else:
            assert r["kind"] == "step"
            out.append(r | {"step": r["step"] + after, "rows": [[f(i), a, n] for i, a, n in r["rows"]]}
                       | {k: [f(i) for i in r[k]] for k in ("new", "resumed", "preempted")}
                       | {k: {f(i): v for i, v in r[k].items()} for k in ("cache_hit", "emitted", "drafts", "finished")})
    return out


@pytest.mark.parametrize("name", ["async-draft3-prefix-cache", "sync-draft5", "async-draft1-light"])
def test_a_span_of_a_longer_serve_is_a_trace_of_its_own(name):
    """One load of a serve that ran others before it, begun with the engine drained and its prefix cache reset: its
    records alone, from step 0, load as the serve of those requests and the replay reproduces them. The empty schedules
    the engine made at the end of the load before are not the span's."""
    recs = R.records(DRAFT / f"{name}.jsonl")
    tok = DRAFT / f"{name}.tokens.json"
    toks = {f"{i}-again": v for i, v in json.loads(tok.read_text()).items()} if tok.exists() else None
    both = recs + _second_serve(recs, sum(r["kind"] == "step" for r in recs), 2)
    whole, t = trace(name, DRAFT), R.load_step_trace(R.span(both, lambda i: i.endswith("-again")), tokens=toks)
    assert t.ids == [f"{i}-again" for i in whole.ids]
    assert (t.requests, t.before_step, t.passed, t.emitted) == (whole.requests, whole.before_step, whole.passed, whole.emitted)
    assert [(s.rows, s.drafts, s.cache_hit) for s in t.steps] == [(s.rows, s.drafts, s.cache_hit) for s in whole.steps]
    assert R.kept_bits(t) == R.kept_bits(whole)
    first = R.load_step_trace(R.span(both, lambda i: not i.endswith("-again")))
    assert (first.ids, first.passed, [s.rows for s in first.steps]) == (whole.ids, whole.passed, [s.rows for s in whole.steps])


def test_the_table_reads_one_load_of_a_longer_serve(tmp_path):
    """`--span` reads each step trace as the span of the requests whose id holds it: the second load of a serve that
    ran two gives the table the load gives served alone."""
    name = "async-draft3-prefix-cache"
    recs = R.records(DRAFT / f"{name}.jsonl")
    toks = json.loads((DRAFT / f"{name}.tokens.json").read_text())
    both = recs + _second_serve(recs, sum(r["kind"] == "step" for r in recs), 2)
    (tmp_path / "both.jsonl").write_text("".join(json.dumps(r) + "\n" for r in both))
    (tmp_path / "both.tokens.json").write_text(json.dumps(toks | {f"{i}-again": v for i, v in toks.items()}))
    R.main([str(tmp_path / "both.jsonl"), "--span=-again", "--out", str(tmp_path / "span.json")])
    R.main([str(DRAFT / f"{name}.jsonl"), "--out", str(tmp_path / "alone.json")])
    table = json.loads((tmp_path / "span.json").read_text())["both"]
    assert table == json.loads((tmp_path / "alone.json").read_text())[name] and table["kept_count"]["charged"] > 0
    with pytest.raises(ValueError, match="not from a trace-corpus directory"):
        R.read_any(tmp_path, within="-again")


def test_a_span_that_is_not_a_serve_of_its_own_is_refused():
    recs = R.records(DRAFT / "sync-draft5.jsonl")
    ids = [r["request"] for r in recs if r["kind"] == "arrive"]
    with pytest.raises(ValueError, match="none of the trace's arrivals"):
        R.span(recs, lambda i: False)
    with pytest.raises(ValueError, match="holds requests outside the span"):
        R.span(recs, lambda i: i in ids[:20])
    with pytest.raises(ValueError, match="steps are not"):
        R.span([r for r in recs if (r["kind"], r.get("step")) != ("step", 3)], lambda i: True)


def test_a_draft_needs_what_each_row_kept():
    """The kept counts are an input the replay checks: without them, with a count over K, with counts that are not the
    request's outputs, and with a count over the drafts the row it schedules there carries, it refuses by name."""
    t = trace(DRAFT_CASES[0], DRAFT)
    K = t.engine.num_speculative_tokens
    how = dict(admit=t.before_step, passed=t.passed)
    first = t.first_scheduled()
    # a request with more than one emitting row whose first row carries no draft, so kept none
    i = next(i for i, r in enumerate(t.requests) if len(r.kept) > 1 and i not in t.steps[first[i]].drafts
             and t.emitted[first[i]].get(i))
    io = t.requests[i]
    assert io.kept[0] == 0

    def with_kept(kept):
        return [*t.requests[:i], R.RequestIO(io.prompt_len, io.max_tokens, io.output_len, io.eos, io.tokens, kept),
                *t.requests[i + 1:]]

    for bad in (None, (K + 1, *io.kept[1:]), io.kept[:-1], (*io.kept, 0)):
        with pytest.raises(ValueError, match=f"request {i}: the engine drafts {K} tokens a row"):
            R.replay(t.engine, with_kept(bad), **how)
    moved = (1, *io.kept[1:-1], *((io.kept[-1] - 1,) if io.kept[-1] else ()))     # one output from its last row to its first
    with pytest.raises(ValueError, match=f"request {i}: emitting row 0 kept 1 drafts, but the row the replay schedules"):
        R.replay(t.engine, with_kept(moved), **how)


def test_refuses_the_drafts_it_does_not_model():
    cfg = dict(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=64)
    mtp = {"method": "mtp", "num_speculative_tokens": 3}
    own = {"num_lookahead_tokens": 3, "num_prefill_lookahead": 1, "use_eagle_block_drop": True, "draft_slots": 0}
    for config in (cfg | {"speculative": mtp}, cfg | {"speculative": mtp | {"disable_eagle_block_drop": False}} | own):
        e = R.Engine.from_header(config)
        assert (type(e), e.num_speculative_tokens) == (R.DraftEngine, 3)
    plain = R.Engine.from_header(cfg)
    assert (type(plain), plain.num_speculative_tokens) == (R.Engine, 0)
    for spec, name in ((mtp | {"method": "ngram"}, "ngram"), (mtp | {"method": "draft_model"}, "draft_model"),
                       (mtp | {"parallel_drafting": True}, "parallel_drafting"),
                       (mtp | {"num_speculative_tokens_per_batch_size": [[1, 3]]}, "num_speculative_tokens_per_batch_size"),
                       (mtp | {"disable_padded_drafter_batch": True}, "disable_padded_drafter_batch"),
                       (mtp | {"disable_eagle_block_drop": True}, "disable_eagle_block_drop"),
                       (mtp | {"num_speculative_tokens": 0}, "num_speculative_tokens")):
        with pytest.raises(ValueError, match=name):
            R.Engine.from_header(cfg | {"speculative": spec})
    for name, value in (("num_lookahead_tokens", 0), ("num_prefill_lookahead", 2), ("use_eagle_block_drop", False),
                        ("draft_slots", 2)):
        with pytest.raises(ValueError, match=name):
            R.Engine.from_header(cfg | {"speculative": mtp} | own | {name: value})
    with pytest.raises(ValueError, match="long_prefill_token_threshold"):
        R.DraftEngine(**cfg, num_speculative_tokens=3, long_prefill_token_threshold=3)


def test_an_engine_without_a_draft_is_the_engine_class_it_was():
    """The certifier's engine class is `Engine`'s fields: a draft's settings are `DraftEngine`'s, so they join none of
    them."""
    e = R.Engine(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=64)
    assert sorted(dataclasses.asdict(e)) == sorted([
        "async_scheduling", "block_size", "enable_chunked_prefill", "enable_prefix_caching", "long_prefill_token_threshold",
        "max_model_len", "max_num_batched_tokens", "max_num_seqs", "num_gpu_blocks", "policy", "scheduler_reserve_full_isl",
        "watermark"])
