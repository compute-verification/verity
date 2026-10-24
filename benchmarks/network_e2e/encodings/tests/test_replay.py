"""ENC-REPLAY: the replay reproduces vLLM's own scheduler, and the residual advice decodes back to the trace.

The fixtures are step traces of vLLM d9105ea80's `Scheduler` and `AsyncScheduler` on a fake executor (`vllm_trace.py
tests/fixtures --requests 14 --seed 3`, the cases named by the file): preemption under KV pressure in both engine loops,
prefix caching, and a light load whose requests join part-full batches and an idle engine."""

from __future__ import annotations

import importlib.util
import json
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
@pytest.mark.parametrize("clocked", [False, True])
def test_residual_decodes_to_the_trace(name, clocked):
    t = trace(name)
    code = R.encode(t, clocked=clocked)
    m = R.compare(R.decode(t, code).steps, t.steps)
    assert m.first_mismatch is None and m.matched == len(t.steps)
    assert all(len(b) <= code.ideal[w] + 2 for w, b in code.bits.items())


def test_wrong_advice_changes_the_schedule():
    t = trace("async-light")
    code = R.encode(t, clocked=True)
    (w, bits), = code.bits.items()
    flipped = R.Code(code.clocked, code.bucket_s, code.window_buckets, {w: [1 - bits[0]] + bits[1:]}, code.ideal)
    assert R.compare(R.decode(t, flipped).steps, t.steps).first_mismatch is not None


def test_table_counts_every_request_and_step():
    t = trace("async-light")
    tb = R.table(t)
    assert sum(w["requests"] for w in tb["windows"]) == len(t.requests)
    assert sum(w["steps"] for w in tb["windows"]) == len(t.steps)
    assert all(v["first_mismatch"] is None for v in tb["decoded"].values())
    for w in tb["windows"]:
        assert w["adaptive_clocked_bits"] < w["v0_bits"] // 100


def test_arrivals_into_a_quiet_engine_cost_nothing():
    e = R.Engine(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=64)
    reqs = [R.RequestIO(prompt_len=20, max_tokens=5, output_len=3, eos=True) for _ in range(3)]
    rp = R.replay(e, reqs, decide=lambda i, k: False)
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
        a = R.replay(e, reqs, decide=lambda i, k: rng.random() < 0.4)
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
