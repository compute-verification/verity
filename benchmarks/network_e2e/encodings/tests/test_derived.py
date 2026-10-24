"""ENC-DERIVED's residual against real schedules: the fixtures are vLLM d9105ea8's own `Scheduler` and `AsyncScheduler`
driven on CPU by `sched_trace.py` (a 200-block pool, so one request is preempted), and the residual must give v0 back
exactly from the config and the tie's witness."""

from __future__ import annotations

import dataclasses
import importlib.util
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _load():
    spec = importlib.util.spec_from_file_location("enc_derived", HERE.parent / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load()
TRACES = ("sync-stress", "async-stress")


@pytest.fixture(scope="module", params=TRACES)
def case(request):
    trace = D.load_trace(HERE / "fixtures" / f"{request.param}.json.gz")
    cfg = D.config_of(trace)
    adv = D.advice_of(trace)
    return trace, cfg, adv, D.holdings_of(trace), D.residual(adv, cfg)


def _with_outputs(held, i, ids):
    return dataclasses.replace(held, outputs=held.outputs[:i] + (tuple(ids),) + held.outputs[i + 1:])


def test_the_residual_rebuilds_v0(case) -> None:
    _, cfg, adv, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    shaped = D.residual(adv, cfg, shaped=True)
    assert shaped.release is None
    assert D.rebuild(shaped, cfg, held, adv.release) == adv


def test_the_walk_samples_where_vllm_did(case) -> None:
    trace, cfg, adv, _, res = case
    walk = D.reconstruct(res, cfg)
    got, k = [], {}
    for step in walk.rows:
        for r in step:
            P = adv.requests[r.request].prompt_len
            got.append(r.start + r.n >= P + k.get(r.request, 0))
            k[r.request] = k.get(r.request, 0) + got[-1]
    assert got == [r["samples"] for s in trace["steps"] for r in s["rows"]]


def test_rows_in_scheduling_order_need_no_n(case) -> None:
    _, cfg, _, _, res = case
    assert res.ns == () and D.reconstruct(res, cfg).explicit_n == ()


def test_rows_out_of_order_send_n_and_rebuild(case) -> None:
    _, cfg, adv, held, _ = case
    rng = random.Random(7)
    steps = tuple(D.StepAdvice(tuple(rng.sample(st.rows, len(st.rows)))) for st in adv.steps)
    shuffled = dataclasses.replace(adv, steps=steps)
    loose = dataclasses.replace(cfg, rows_in_scheduling_order=False)
    res = D.residual(shuffled, loose)
    assert 0 < len(res.ns) == len(D.reconstruct(res, loose).explicit_n) < sum(len(st.rows) for st in steps)
    assert D.rebuild(res, loose, held, ()) == shuffled
    try:
        assert D.rebuild(D.residual(shuffled, cfg), cfg, held, ()) != shuffled
    except ValueError:
        pass


def test_a_preemption_sends_its_resume_and_none_can_be_dropped(case) -> None:
    _, cfg, adv, _, res = case
    assert res.resumes is not None and len(res.resumes) == sum(r.preempted for st in adv.steps for r in st.rows) == 1
    with pytest.raises(ValueError):
        D.reconstruct(dataclasses.replace(res, resumes=None), cfg)


def test_pool_bound() -> None:
    cfg = D.Config(block_size=16, num_gpu_blocks=5, max_num_batched_tokens=512, max_model_len=4096, eos_token_id=2,
                   stop_token_ids=())
    together = [[(0, 0, 48), (1, 0, 32)]]
    assert not D.pool_bound(together, cfg)
    assert D.pool_bound(together, dataclasses.replace(cfg, num_gpu_blocks=6))
    assert D.pool_bound([[(0, 0, 48)], [(1, 0, 32)]], cfg)
    assert not D.pool_bound([[(0, 0, 48)], [(1, 0, 32)], [(0, 48, 1)]], cfg)


def test_stop_rule() -> None:
    cfg = D.Config(block_size=16, num_gpu_blocks=8, max_num_batched_tokens=64, max_model_len=100, eos_token_id=2,
                   stop_token_ids=(3,))
    assert D.stop_rule([5, 6, 2], 10, 8, cfg) == "eos"
    assert D.stop_rule([5, 6, 3], 10, 8, cfg) == "eos"
    assert D.stop_rule([5, 6, 7], 10, 3, cfg) == "length"
    assert D.stop_rule([5, 6, 7], 97, 8, cfg) == "length"
    for bad in ([], [5, 2, 7], [5, 6, 7], [5, 6, 7, 8]):
        assert D.stop_rule(bad, 10, 3 if len(bad) == 4 else 8, cfg) is None


def test_the_tie_refuses_witness_that_disagrees(case) -> None:
    _, cfg, adv, held, res = case
    i = next(i for i, r in enumerate(adv.requests) if r.finish == "eos" and r.output_len >= 3)
    req = held.requests[i]
    short = dataclasses.replace(held, requests=held.requests[:i] + (dataclasses.replace(req, prompt_ids=req.prompt_ids[:-1]),)
                                + held.requests[i + 1:])
    with pytest.raises(ValueError, match="prompt_len"):
        D.rebuild(res, cfg, short, ())
    ids = held.outputs[i]
    with pytest.raises(ValueError, match="stop rule"):
        D.rebuild(res, cfg, _with_outputs(held, i, (ids[0], cfg.eos_token_id) + ids[2:]), ())
    with pytest.raises(ValueError, match="committed"):
        D.rebuild(res, cfg, _with_outputs(held, i, ids + (ids[0],)), ())


def test_a_flipped_lag_bit_is_refused(case) -> None:
    _, cfg, adv, held, res = case
    if cfg.scheduler != "async":
        assert res.lag is None
        return
    walk = D.reconstruct(res, cfg)
    assert any(res.lag) and not all(res.lag)
    for j, i in enumerate(walk.lag_requests):
        flipped = dataclasses.replace(res, lag=res.lag[:j] + (not res.lag[j],) + res.lag[j + 1:])
        ids = held.outputs[i]
        for witness in (held, _with_outputs(held, i, ids + (ids[-1],) if res.lag[j] else ids[:-1])):
            with pytest.raises(ValueError):
                D.rebuild(flipped, cfg, witness, ())


def test_fields_covers_every_i1_field() -> None:
    i1 = {f.name for cls in (D.RequestAdvice, D.Row) for f in dataclasses.fields(cls)} | {"release"}
    named = {f["field"].split(".")[-1].split(" ")[0] for f in D.FIELDS}
    assert i1 <= named, i1 - named


def test_derived_costs_less_than_v0_in_every_window(case) -> None:
    trace, *_ = case
    table = D.bits_table(trace, buckets=50)
    assert len(table["windows"]) > 1
    for w in table["windows"]:
        assert w["derived_shaped_bits"] <= w["derived_bits"] < w["v0_bits"], w
        assert w["forced_bits"] <= w["derived_bits"]
