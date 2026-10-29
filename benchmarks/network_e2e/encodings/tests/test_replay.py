"""The scheduler replay: the replay reproduces vLLM's own scheduler, and the residual advice decodes back to the trace.

The fixtures are step traces of vLLM d9105ea80's `Scheduler` and `AsyncScheduler` on a fake executor (`vllm_trace.py
tests/fixtures --requests 14 --seed 3`, the cases named by the file): preemption under KV pressure in both engine loops,
prefix caching, and a light load whose requests join part-full batches and an idle engine."""

from __future__ import annotations

import importlib.util
import json
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


def trace(name: str):
    tok = FIXTURES / f"{name}.tokens.json"
    return R.read_step_trace(FIXTURES / f"{name}.jsonl", tokens=json.loads(tok.read_text()) if tok.exists() else None)


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
    rng = random.Random(7)
    for asy in (False, True):
        e = R.Engine(max_num_seqs=3, max_num_batched_tokens=48, max_model_len=256, num_gpu_blocks=12, async_scheduling=asy)
        reqs = []
        for _ in range(25):
            out = rng.randint(1, 30)
            reqs.append(R.RequestIO(prompt_len=rng.randint(4, 90), max_tokens=out + rng.randint(0, 2), output_len=out,
                                    eos=rng.random() < 0.7))
        reqs = [R.RequestIO(r.prompt_len, r.max_tokens, r.output_len if r.eos else r.max_tokens, r.eos) for r in reqs]
        a = R.replay(e, reqs, decide=lambda i, k, _: rng.random() < 0.4)
        b = R.replay(e, reqs, admit=a.admitted)
        assert R.compare(a.steps, b.steps).ordered == len(a.steps)
        assert any(s.preempted for s in a.steps)


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
