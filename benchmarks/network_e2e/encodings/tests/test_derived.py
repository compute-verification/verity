"""ENC-DERIVED's residual against real schedules: the fixtures are vLLM d9105ea8's own `Scheduler` and `AsyncScheduler`
driven on CPU by `sched_trace.py` (`--seconds 8 --rate 12 --num-blocks 160 --max-num-batched-tokens 256`: about 20
preemptions and 77 steps the token budget binds), and the residual must give I1's advice back exactly from the config and
the tie's witness."""

from __future__ import annotations

import dataclasses
import importlib.util
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier import advice as I1

HERE = Path(__file__).parent


def _load():
    spec = importlib.util.spec_from_file_location("enc_derived", HERE.parent / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load()


@pytest.fixture(scope="module", params=("sync-crush", "async-crush"))
def case(request):
    trace = D.load_trace(HERE / "fixtures" / f"{request.param}.json.gz")
    cfg = D.config_of(trace)
    adv, order = D.advice_of(trace)
    return trace, cfg, adv, order, D.holdings_of(trace), D.residual(adv, cfg, order=order)


def _with_outputs(held, t, ids):
    return dataclasses.replace(held, outputs=held.outputs[:t] + (tuple(ids),) + held.outputs[t + 1:])


def _refused(f, want) -> bool:
    """`f` raises, or gives something other than `want`."""
    try:
        return f() != want
    except ValueError:
        return True


def test_the_residual_rebuilds_v0(case) -> None:
    _, cfg, adv, _, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    shaped = D.residual(adv, cfg, shaped=True)
    assert shaped.release is None
    assert D.rebuild(shaped, cfg, held, adv.release) == adv


def test_the_walk_samples_where_vllm_did(case) -> None:
    trace, cfg, adv, _, _, res = case
    got, k = [], {}
    for st in D.reconstruct(res, cfg).steps:
        for r in st.rows:
            got.append(r.start + r.n >= adv.requests[r.request].prompt_len + k.get(r.request, 0))
            k[r.request] = k.get(r.request, 0) + got[-1]
    assert got == [r["samples"] for s in trace["steps"] for r in s["rows"]]


def test_rows_in_scheduling_order_need_no_n(case) -> None:
    _, cfg, _, _, _, res = case
    assert res.ns == () and D.reconstruct(res, cfg).explicit_n == ()


def test_rows_out_of_order_send_n_and_rebuild(case) -> None:
    """Each step's last row moved first (where that keeps requests numbered by first row): n follows the order only where
    the budget binds, and those steps are where the order-blind walk must refuse it."""
    _, cfg, adv, _, held, _ = case
    seen, steps, bound = set(), [], 0
    for st in adv.steps:
        new = {r.request for r in st.rows} - seen
        seen |= new
        if len(new) <= 1 and len(st.rows) > 1:
            st = I1.StepAdvice(st.rows[-1:] + st.rows[:-1])
            bound += sum(r.n for r in st.rows) == cfg.max_num_batched_tokens
        steps.append(st)
    assert bound
    shuffled = I1.Advice(adv.requests, tuple(steps), adv.release)
    loose = dataclasses.replace(cfg, row_order="explicit")
    res = D.residual(shuffled, loose)
    assert 0 < len(res.ns) == len(D.reconstruct(res, loose).explicit_n) < sum(len(st.rows) for st in steps)
    assert D.rebuild(res, loose, held, ()) == shuffled
    assert _refused(lambda: D.rebuild(D.residual(shuffled, cfg), cfg, held, ()), shuffled)


def _batch_order(adv, rng):
    """Each step's rows shuffled as a runner's batch might hold them, new requests kept in first-row order."""
    seen, steps = set(), []
    for st in adv.steps:
        rows = rng.sample(st.rows, len(st.rows))
        slots = [j for j, r in enumerate(rows) if r.request not in seen]
        for j, r in zip(slots, sorted((r for r in rows if r.request not in seen), key=lambda r: r.request)):
            rows[j] = r
        seen |= {r.request for r in rows}
        steps.append(I1.StepAdvice(tuple(rows)))
    return I1.Advice(adv.requests, tuple(steps), adv.release)


def test_rows_in_batch_order_take_vllms_scheduling_order_from_history(case) -> None:
    """vLLM's own scheduling order is what the walk derives from admissions alone (rows in scheduling order rebuild under
    `row_order` "batch", budget-bound steps and the resume included), so a batch order costs no n."""
    _, cfg, adv, _, held, _ = case
    batch = dataclasses.replace(cfg, row_order="batch")
    assert D.rebuild(D.residual(adv, batch), batch, held, ()) == adv
    for seed in range(3):
        shuffled = _batch_order(adv, random.Random(seed))
        assert shuffled != adv
        res = D.residual(shuffled, batch)
        assert res.ns == () and D.rebuild(res, batch, held, ()) == shuffled


def test_a_resume_is_sent_and_checked(case) -> None:
    _, cfg, adv, _, held, res = case
    assert res.resumes is not None and len(res.resumes) == sum(r.preempted for st in adv.steps for r in st.rows) > 10
    r = res.resumes
    for bad in (None, (), r[:-1], r[1:], r + (r[-1] + 1,), (r[0] + 1,) + r[1:], tuple(x + 1 for x in r)):
        assert _refused(lambda: D.rebuild(dataclasses.replace(res, resumes=bad), cfg, held, ()), adv)


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
    _, cfg, adv, order, held, res = case
    i = next(i for i, q in enumerate(adv.requests) if q.finish == "eos" and q.output_len >= 3)
    t = order[i]
    req = held.requests[t]
    short = dataclasses.replace(held, requests=held.requests[:t] + (dataclasses.replace(req, prompt_ids=req.prompt_ids[:-1]),)
                                + held.requests[t + 1:])
    with pytest.raises(ValueError, match="prompt_len"):
        D.rebuild(res, cfg, short, ())
    ids = held.outputs[t]
    with pytest.raises(ValueError, match="stop rule"):
        D.rebuild(res, cfg, _with_outputs(held, t, (ids[0], cfg.eos_token_id) + ids[2:]), ())
    with pytest.raises(ValueError, match="committed"):
        D.rebuild(res, cfg, _with_outputs(held, t, ids + (ids[0],)), ())


def test_requests_out_of_ingress_order_send_it(case) -> None:
    _, cfg, adv, order, held, _ = case
    perm = list(range(len(held.requests)))
    random.Random(3).shuffle(perm)
    moved = D.Holdings(tuple(held.requests[perm.index(t)] for t in range(len(perm))),
                       tuple(held.outputs[perm.index(t)] for t in range(len(perm))))
    res = D.residual(adv, cfg, order=[perm[t] for t in order])
    assert res.order is not None and D.rebuild(res, cfg, moved, ()) == adv
    assert _refused(lambda: D.rebuild(dataclasses.replace(res, order=None), cfg, moved, ()), adv)


def test_a_flipped_lag_bit_is_refused(case) -> None:
    _, cfg, _, order, held, res = case
    if cfg.scheduler != "async":
        assert res.lag is None
        return
    walk = D.reconstruct(res, cfg)
    assert any(res.lag) and not all(res.lag)
    some = [j for lag in (True, False) for j in [j for j, x in enumerate(res.lag) if x == lag][:6]]
    for j, i in ((j, walk.lag_requests[j]) for j in some):
        flipped = dataclasses.replace(res, lag=res.lag[:j] + (not res.lag[j],) + res.lag[j + 1:])
        ids = held.outputs[order[i]]
        for witness in (held, _with_outputs(held, order[i], ids + (ids[-1],) if res.lag[j] else ids[:-1])):
            with pytest.raises(ValueError):
                D.rebuild(flipped, cfg, witness, ())


def test_a_residual_with_values_left_over_or_missing_is_refused(case) -> None:
    _, cfg, _, _, _, res = case
    for bad in (dict(hits=res.hits + (0,)), dict(hits=res.hits[:-1]), dict(sources=res.sources[:-1]), dict(ns=(4,))):
        with pytest.raises(ValueError):
            D.reconstruct(dataclasses.replace(res, **bad), cfg)


def test_fields_covers_every_i1_field() -> None:
    i1 = {f.name for cls in (I1.RequestAdvice, I1.Row) for f in dataclasses.fields(cls)} | {"release"}
    named = {f["field"].split(".")[-1].split(" ")[0] for f in D.FIELDS}
    assert i1 <= named, i1 - named


@pytest.mark.parametrize("T", (50, 600))
def test_v0_records_are_i1s_layout_and_charge(case, T) -> None:
    _, _, adv, *_ = case
    rec = D.v0_records(adv)
    assert sum(rec.header.values()) + sum(sum(d.values()) for d in (*rec.requests, *rec.steps)) \
        + rec.release * len(adv.release) == I1.total_bits(adv)
    assert {w: sum(d.values()) for w, d in D.charge(adv, T, rec).items()} == I1.bits_by_window(adv, T)


def test_derived_costs_less_than_v0_in_every_window(case) -> None:
    trace, *_ = case
    table = D.bits_table(trace, T=50)
    assert len(table["windows"]) > 1
    for w in table["windows"]:
        assert w["forced_bits"] < w["derived_shaped_bits"] < w["derived_bits"] < w["v0_bits"], w
