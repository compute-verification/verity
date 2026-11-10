"""The derived rules' residual against real schedules: the fixtures are vLLM d9105ea8's own `Scheduler` and
`AsyncScheduler` driven on CPU by `sched_trace.py`
(`--seconds 8 --rate 12 --num-blocks 160 --max-num-batched-tokens 256`: about 20 preemptions and 77 steps the token
budget binds), and the residual must give I1's advice back exactly from the config and the tie's witness."""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import random
import shutil
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
    held = D.holdings_of(trace)
    return trace, cfg, adv, order, held, D.residual(adv, cfg, held, order=order)


def _with_outputs(held, t, ids):
    return dataclasses.replace(held, outputs=held.outputs[:t] + (tuple(ids),) + held.outputs[t + 1:])


def _refused(f, want) -> bool:
    """`f` raises, or gives something other than `want`."""
    try:
        return f() != want
    except ValueError:
        return True


def test_the_residual_rebuilds_v0(case) -> None:
    _, cfg, adv, order, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    shaped = D.residual(adv, cfg, held, order=order, shaped=True)
    assert shaped.release is None
    assert D.rebuild(shaped, cfg, held, adv.release) == adv


def test_the_walk_samples_where_vllm_did(case) -> None:
    trace, cfg, adv, _, held, res = case
    got, k = [], {}
    for st in D.reconstruct(res, cfg, held).steps:
        for r in st.rows:
            got.append(r.start + r.n >= adv.requests[r.request].prompt_len + k.get(r.request, 0))
            k[r.request] = k.get(r.request, 0) + got[-1]
    assert got == [r["samples"] for s in trace["steps"] for r in s["rows"]]


def test_rows_in_scheduling_order_need_no_n(case) -> None:
    _, cfg, _, _, held, res = case
    assert res.ns == () and D.reconstruct(res, cfg, held).explicit_n == ()


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
    res = D.residual(shuffled, loose, held)
    assert 0 < len(res.ns) == len(D.reconstruct(res, loose, held).explicit_n) < sum(len(st.rows) for st in steps)
    assert D.rebuild(res, loose, held, ()) == shuffled
    assert _refused(lambda: D.rebuild(D.residual(shuffled, cfg, held), cfg, held, ()), shuffled)


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
    assert D.rebuild(D.residual(adv, batch, held), batch, held, ()) == adv
    for seed in range(3):
        shuffled = _batch_order(adv, random.Random(seed))
        assert shuffled != adv
        res = D.residual(shuffled, batch, held)
        assert res.ns == () and D.rebuild(res, batch, held, ()) == shuffled


def _tokens(adv, held, order):
    return D.tokens_of(held, order, [q.prompt_len for q in adv.requests])


def _with_row(adv, s, k, row):
    st = adv.steps[s]
    return I1.Advice(adv.requests, adv.steps[:s] + (I1.StepAdvice(st.rows[:k] + (row,) + st.rows[k + 1:]),) + adv.steps[s + 1:],
                     adv.release)


def _mid_block(adv, fine, held, order):
    """`adv` with one hit moved half a block on, as a hybrid model's finer hash lets it end inside a block: an admitting
    row whose next half block some request computed from the same tokens (a resumed one's own, say), so that only its
    start, n and sources change and the walk at hit unit `fine.unit` gives it back."""
    half, tokens = fine.unit, _tokens(adv, held, order)
    for s, st in enumerate(adv.steps):
        for k, r in enumerate(st.rows):
            if not r.cache_hit or r.n <= half + 1:
                continue
            src = r.sources[:-1] + (I1.Source(r.sources[-1].request, r.sources[-1].n + half),)
            row = dataclasses.replace(r, start=r.start + half, n=r.n - half, cache_hit=r.cache_hit + half, sources=src)
            try:
                moved = D.canonical_sources(_with_row(adv, s, k, row), tokens, fine.unit)
                D.residual(moved, fine, held, order=order)
                return moved
            except ValueError:
                continue
    raise AssertionError("no hit to move")


def test_a_hit_unit_finer_than_a_block_takes_hits_inside_one(case) -> None:
    """vLLM's hit lands on its hash_block_size: the block for one KV cache group, finer in a hybrid model (the tail
    copied on write). The config's unit is public, so a sent hit inside a block costs a bit of width, not a refusal."""
    _, cfg, adv, order, held, _ = case
    sent = dataclasses.replace(cfg, num_gpu_blocks=0)
    fine = dataclasses.replace(sent, hit_unit=cfg.block_size // 2)
    res, halves = D.residual(adv, sent, held, order=order), D.residual(adv, fine, held, order=order)
    assert res.hits and halves.hits == tuple(2 * h for h in res.hits) and D.rebuild(halves, fine, held, ()) == adv
    moved = _mid_block(adv, fine, held, order)
    with pytest.raises(ValueError, match="hit units"):
        D.residual(moved, sent, held, order=order)
    assert D.rebuild(D.residual(moved, fine, held, order=order), fine, held, ()) == moved


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
    """The engine taking requests out of the certified admission order: the steps whose new requests the walk mispredicts
    are sent, and nothing else."""
    _, cfg, adv, order, held, res = case
    assert res.order == ()
    perm = list(range(len(held.requests)))
    random.Random(3).shuffle(perm)
    moved = D.Holdings(tuple(held.requests[perm.index(t)] for t in range(len(perm))),
                       tuple(held.outputs[perm.index(t)] for t in range(len(perm))))
    told = D.residual(adv, cfg, moved, order=[perm[t] for t in order])
    assert told.order and D.rebuild(told, cfg, moved, ()) == adv
    assert _refused(lambda: D.rebuild(dataclasses.replace(told, order=()), cfg, moved, ()), adv)
    assert _refused(lambda: D.rebuild(dataclasses.replace(told, order=told.order[1:]), cfg, moved, ()), adv)


def test_an_order_the_walk_predicts_or_cant_take_is_refused(case) -> None:
    """A sent step whose new requests are the walk's prediction (a second string for the same advice), and one naming a
    request already admitted, are refused."""
    _, cfg, adv, order, held, res = case
    seen: set[int] = set()
    for s, st in enumerate(adv.steps):
        new = [r.request for r in st.rows if r.request not in seen]
        seen |= {r.request for r in st.rows}
        if s and new:
            break
    with pytest.raises(ValueError, match="order entry"):
        D.reconstruct(dataclasses.replace(res, order=((s, tuple(order[i] for i in new)),)), cfg, held)
    with pytest.raises(ValueError, match="not yet admitted"):
        D.reconstruct(dataclasses.replace(res, order=((s, (order[0],) + tuple(order[i] for i in new[1:])),)), cfg, held)


def test_a_flipped_lag_bit_is_refused(case) -> None:
    _, cfg, _, order, held, res = case
    if cfg.scheduler != "async":
        assert res.lag is None
        return
    walk = D.reconstruct(res, cfg, held)
    assert any(res.lag) and not all(res.lag)
    some = [j for lag in (True, False) for j in [j for j, x in enumerate(res.lag) if x == lag][:6]]
    for j, i in ((j, walk.lag_requests[j]) for j in some):
        flipped = dataclasses.replace(res, lag=res.lag[:j] + (not res.lag[j],) + res.lag[j + 1:])
        ids = held.outputs[order[i]]
        for witness in (held, _with_outputs(held, order[i], ids + (ids[-1],) if res.lag[j] else ids[:-1])):
            with pytest.raises(ValueError):
                D.rebuild(flipped, cfg, witness, ())


def test_a_residual_with_values_left_over_or_missing_is_refused(case) -> None:
    _, cfg, _, _, held, res = case
    assert res.hits
    for bad in (dict(hits=res.hits + (0,)), dict(hits=res.hits[:-1]), dict(ns=(4,))):
        with pytest.raises(ValueError):
            D.reconstruct(dataclasses.replace(res, **bad), cfg, held)


def _admitting(adv):
    """(step, row) of each admitting row: a request's first, or a resumed one."""
    seen, out = set(), []
    for s, st in enumerate(adv.steps):
        for k, r in enumerate(st.rows):
            if r.request not in seen or r.preempted:
                out.append((s, k))
            seen.add(r.request)
    return out


def _derived_hits(adv, cfg, held, order):
    """The admitting rows whose hit the walk derives, those before the no-eviction bound fails."""
    res = D.residual(adv, cfg, held, order=order)
    sent = len(D.reconstruct(res, cfg, held).sent_hits)
    rows = _admitting(adv)
    return rows[:len(rows) - sent] if cfg.scheduler == "sync" else []


def _evicted(adv, cfg, tokens):
    """The steps of the admitting rows, in scheduling order, whose hit is shorter than the longest prefix rows before
    them computed: where vLLM had evicted."""
    p, seen, samples, out = D.Prefixes(tokens, cfg.unit), set(), {}, []
    for s, st in enumerate(adv.steps):
        for r in st.rows:
            K = adv.requests[r.request].prompt_len + samples.get(r.request, 0)
            if (r.request not in seen or r.preempted) and r.cache_hit < p.cached(r.request, K):
                out.append(s)
            seen.add(r.request)
            p.add(r.request, r.start, r.start + r.n)
            samples[r.request] = samples.get(r.request, 0) + (r.start + r.n >= K)
    return out


def test_a_hit_is_the_longest_cached_prefix_while_nothing_can_be_evicted(case) -> None:
    """While the unshared blocks of every request admitted so far fit the pool, vLLM has evicted nothing, so a hit is
    the longest prefix rows scheduled before it computed: derived, 0 bits. A shorter one (its start and n moved to match)
    is refused. The bound is what makes that sound: vLLM does evict here, and every hit eviction shortened is past it."""
    _, cfg, adv, order, held, res = case
    if cfg.scheduler != "sync":
        assert len(res.hits) == len(_admitting(adv))
        return
    derived = _derived_hits(adv, cfg, held, order)
    assert 0 < len(derived) and len(res.hits) == len(_admitting(adv)) - len(derived)
    evicted = _evicted(adv, cfg, _tokens(adv, held, order))
    assert evicted and min(evicted) > derived[-1][0]
    short = 0
    for s, k in derived:
        r = adv.steps[s].rows[k]
        if not r.cache_hit or sum(x.n for x in adv.steps[s].rows) + cfg.unit > cfg.max_num_batched_tokens:
            continue
        row = dataclasses.replace(r, start=r.start - cfg.unit, n=r.n + cfg.unit, cache_hit=r.cache_hit - cfg.unit,
                                  sources=r.sources[:-1] + ((I1.Source(r.sources[-1].request, r.sources[-1].n - cfg.unit),)
                                                            if r.sources[-1].n > cfg.unit else ()))
        with pytest.raises(ValueError, match="longest cached"):
            D.residual(_with_row(adv, s, k, row), cfg, held, order=order)
        short += 1
    assert short


def _not_earliest(adv, tokens):
    """`adv` with one hit's first source run named for a later request that also computed KV there by that step from
    the same tokens (one I1 accepts), or None."""
    for s, st in enumerate(adv.steps):
        for k, r in enumerate(st.rows):
            if not r.cache_hit:
                continue
            a, n = r.sources[0].request, r.sources[0].n
            for b in range(a + 1, len(adv.requests)):
                if b == r.request or tokens[b][:n] != tokens[r.request][:n]:
                    continue
                try:
                    return _with_row(adv, s, k, dataclasses.replace(r, sources=(I1.Source(b, n),) + r.sources[1:]))
                except I1.MalformedAdvice:
                    continue
    return None


def test_a_source_is_the_earliest_admitted_request(case) -> None:
    """A hit's KV comes from the earliest-admitted request that computed it from the same tokens: canonical, 0 bits, and
    any other request I1 would accept is refused."""
    _, cfg, adv, order, held, res = case
    tokens = _tokens(adv, held, order)
    assert D.canonical_sources(adv, tokens, cfg.unit) == adv
    bad = _not_earliest(adv, tokens)
    assert bad is not None
    with pytest.raises(ValueError, match="sources"):
        D.residual(bad, cfg, held, order=order)
    assert D.rebuild(res, cfg, held, ()) != bad


CORPUS = HERE / "fixtures" / "corpus-smallkv"


def _corpus(d=CORPUS, **kw):
    return D.load_corpus_trace(d, eos_token_id=128009, stop_token_ids=(128001, 128008), **kw)


def _corpus_with(tmp_path, change):
    """The corpus fixture with `change` applied to each joined step record."""
    steps = [json.loads(line) for line in gzip.decompress((CORPUS / "steps.jsonl.gz").read_bytes()).decode().splitlines()]
    for s in steps:
        change(s)
    for f in ("requests.jsonl.gz", "meta.json", "server.log"):
        shutil.copy(CORPUS / f, tmp_path / f)
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    return _corpus(tmp_path)


def test_a_trace_corpus_trace_rebuilds_from_the_runners_batch_order(tmp_path) -> None:
    """The trace corpus's format from its own hook (`benchmarks/network_e2e/traces/vllm_trace_plugin.py`) and join
    (`summary.join_trace`) on vLLM d9105ea8's scheduler laid out as for Model Runner V2 (resumed requests among the new),
    driven by `sched_trace.py` (4 s at 12/s, 100 blocks, a 256-token budget), each step's batch put decodes first as V2's
    runner does; `server.log` is vLLM's capacity line for those 100 blocks. Decodes first is vLLM's scheduling order here
    (a request mid-prefill is the running list's newest), so the batches are also rotated, last row first where no two
    requests are new: from either order the walk derives every n, budget-bound steps and resumes included, at the bits it
    costs from vLLM's scheduling order. V2 itself sorts the step by scheduled tokens, stably (`_v2`), which the walk
    predicts; the fixture's decodes-first batch puts some steps' new requests in arrival order instead, and those steps'
    order is sent."""
    def rotate(s):
        if s.get("batch") and sum(not r["resumed"] for r in s["new"]) <= 1:
            s["batch"] = s["batch"][-1:] + s["batch"][:-1]

    sched, batch = _corpus(row_order="scheduling"), _corpus()
    v2, rotated = _corpus_with(tmp_path, _v2), _corpus_with(tmp_path, lambda s: (_v2(s), rotate(s)))
    assert sched["config"]["num_gpu_blocks"] == 100
    budget = sched["config"]["max_num_batched_tokens"]
    assert sum([r["request"] for r in a["rows"]] != [r["request"] for r in b["rows"]] and sum(r["n"] for r in a["rows"]) == budget
               for a, b in zip(sched["steps"], rotated["steps"])) > 10
    tables = [D.bits_table(t, T=600) for t in (sched, v2, rotated, batch)]
    for t in tables:
        assert t["explicit_n"] == 0 and t["resumed_rows"] > 0 and t["sources"] > 0
        assert 0 < t["hits_derived"] and 0 < t["hits_sent"]
        assert t["totals"]["v0_bits"] == tables[0]["totals"]["v0_bits"]
    for t in tables[:3]:
        assert t["order_sent"] == 0 and t["by_field"] == tables[0]["by_field"]
    told = tables[3]["by_field"]["derived"]
    assert 0 < tables[3]["order_sent"] < tables[3]["new_steps"] and told["order"] > 0
    assert {f: b for f, b in told.items() if f not in ("order", "header")} == \
        {f: b for f, b in tables[0]["by_field"]["derived"].items() if f != "header"}


def _v2(s):
    """Model Runner V2's batch: the step's requests in scheduling order, sorted stably by scheduled tokens."""
    if s.get("batch"):
        sched = ([r for r in s["cached"] if not r["resumed"]] + [r for r in s["cached"] + s["new"] if r["resumed"]]
                 + [r for r in s["new"] if not r["resumed"]])
        s["batch"] = [r["id"] for r in sorted((r for r in sched if r["id"] in set(s["batch"])),
                                              key=lambda r: r["num_scheduled_tokens"])]


def test_prefills_by_size_take_the_budget_in_arrival_order(tmp_path) -> None:
    """Model Runner V2's batch on the GPU (the trace corpus's): decodes first, then the prefill
    rows by size, so a step's new requests leave arrival order. vLLM admits them in arrival order, the certified ingress
    order, so the walk takes them so, only the last arrival is cut short, and their I1 numbers follow from their n, at no
    cost. A trace whose new requests of a step arrived the other way round is refused."""
    def by_size(s):
        if s.get("batch"):
            rows = {r["id"]: r for r in s["new"] + s["cached"]}
            s["batch"] = sorted(s["batch"], key=lambda i: (rows[i]["kind"] != "decode", rows[i]["num_scheduled_tokens"]))

    trace = _corpus_with(tmp_path, by_size)
    t = D.bits_table(trace, T=600)
    assert t["explicit_n"] == 0 and t["order_sent"] == 0 and "order" not in t["by_field"]["derived"]
    cfg, held = D.config_of(trace), D.holdings_of(trace)
    adv, order = D.advice_of(trace)
    cut = []
    for s, st in enumerate(adv.steps):
        new = [r for r in st.rows if r.request not in {x.request for y in adv.steps[:s] for x in y.rows}]
        short = [r for r in new if r.start + r.n < adv.requests[r.request].prompt_len]
        if short and new[-1] is not short[0]:
            cut.append((s, short[0].request, [r.request for r in new if r is not short[0]]))
    assert cut
    s, c, others = cut[0]
    swap = {order[c]: order[others[0]], order[others[0]]: order[c]}
    moved = D.Holdings(tuple(held.requests[swap.get(t, t)] for t in range(len(order))),
                       tuple(held.outputs[swap.get(t, t)] for t in range(len(order))))
    with pytest.raises(ValueError, match="budget left"):
        D.residual(adv, cfg, moved, order=[swap.get(t, t) for t in order])


def test_a_trace_corpus_trace_off_the_config_is_refused() -> None:
    with pytest.raises(ValueError, match="block"):
        _corpus(block_size=32)
    with pytest.raises(ValueError, match="stop rule"):
        D.bits_table(D.load_corpus_trace(CORPUS), T=600)


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
