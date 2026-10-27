"""The transformer's sampling plan (`transformer.py`): its DAG's units are their Definitions' and counted right, a unit is
credited its ANDs whatever the cut, the frontier prices every serve-time commitment and is monotone, the chosen plan's
certificate recomputes through core's `Stratified`, the pooled bound is IntegrityHorizon's in the uniform case, the
hand-off's work law draws the plan's k and its windows are IntegrityPeriodLog's, and the large model fills the card and
meets the target share."""

from __future__ import annotations

import json
from dataclasses import replace
from math import ceil, log
from pathlib import Path
from types import SimpleNamespace

import pytest

import transformer as T
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k
from verity.core.protocols.sampled_proofs.optimize import Candidate
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

CENSUS = Path(__file__).resolve().parents[2] / "census" / "hardware.json"
CIR_100 = ("Argmax", "EmbedSelect", "Quant", "Relu", "Rms", "Softmax")
#: units of several instances of one Definition, which no Definition of their own composes yet
GROUPS = ("Bf16AddRow", "EmbedRow", "PvHead", "QkHead", "S8Tile")


def _plan(m: T.Model, s: T.Serving, **kw) -> SimpleNamespace:
    dag, cm, objective, cands, _, front = T.plans(m, s, T.DELTAS[0], **kw)
    plan = T.chosen(cands, cm, objective, front, T.TARGET_SHARE * dag.work)
    return SimpleNamespace(m=m, s=s, dag=dag, cm=cm, objective=objective, cands=cands, front=front, plan=plan)


@pytest.fixture(scope="module")
def small():
    return _plan(T.SMALL, T.Serving(audit_store=T.SMALL.weight_bytes), grid=31, cut_grid=21)


def test_matmuls_are_their_definitions_and_the_rest_named_stubs(small):
    s8 = small.dag.s8(128)
    assert s8.ands == 22_017 and s8.out_bits <= 16
    assert small.dag.ids[s8.id] == descriptor_id(bind(G.ScaledMmS8Coordinate, K=128, **G.GEMM_SM120_S8))
    h = T.handoff(small.plan, small.cands, small.dag)
    real = {t for t, _ in h["record"]["templates"]} - set(h["stubs"])
    assert real and real <= set(small.dag.ids.values())
    assert h["stubs"] and all(s.startswith("stub:") and s[5:].startswith(CIR_100 + GROUPS) for s in h["stubs"])


def test_units_are_counted_per_block(small):
    m, counts = small.m, small.dag.unit_counts()
    s8 = m.requests * (m.positions * m.layers * (4 * m.width + m.mlp) + m.generate * m.vocab)
    assert counts[small.dag.s8(m.width).id] == s8 == 278_528
    assert counts[small.dag.real(T.F.Bf16Add).id] == m.requests * m.positions * (2 * m.layers + 1) * m.width
    assert small.dag.units_dense == sum(counts.values())


def test_a_unit_is_credited_its_ands_whatever_the_cut(small):
    dag, m = small.dag, small.m
    coord, tile = dag.s8(m.width), dag.tile(m.width, m.positions)
    assert coord.work == coord.ands and tile.work == m.positions * T.TILE_COLS * coord.work
    rms = dag.rmsnorm(m.width)
    ways = dict(rms.ways)
    (inv, one), (coord_, n) = ways["shared"]
    ((fused, n_),) = ways["recomputed"]
    assert rms.work == inv.work * one + coord_.work * n and fused.work == coord_.work and n_ == n
    assert dag.work == sum(s.work * n for s, n in dag.ops.values())


def test_frontier_prices_every_serve_time_commitment_and_is_monotone(small):
    assert {c.scale for c in small.cm.serve} == {"requests", "pouw:h1", "bandwidth"}
    assert {c.scale for c in T.costs(replace(T.LARGE, inflight=1), small.s).serve} >= {"carry"}
    serve = small.cm.serve_cost()
    assert serve > 0 and all(p.cost["serve"] == pytest.approx(serve) for p in small.front)
    pairs = [(p.cost["total"], p.certificate.harm) for p in small.front]
    assert all(c0 <= c1 and h0 >= h1 for (c0, h0), (c1, h1) in zip(pairs, pairs[1:]))
    assert pairs[-1][1] == 0


@pytest.mark.parametrize("delta", T.DELTAS)
def test_chosen_plan_recertifies_through_stratified(small, delta):
    plan = small.plan
    assert plan is not None and plan.certificate.harm <= T.TARGET_SHARE * small.dag.work
    r = T.recertify(plan, small.cands, delta)
    assert r["count_bound"] <= r["units"] + 1e-9
    assert r["ands"] <= r["density_bound"] * (1 + 1e-9) and r["certify"] == pytest.approx(r["ands"], rel=1e-9)
    law = T.law(plan, small.cands)
    assert law.miss(r["count_bound"]) >= delta > law.miss(r["count_bound"] + 1)
    if delta == T.DELTAS[0]:
        assert r["ands"] <= plan.certificate.harm * (1 + 1e-9)


def test_pooled_is_integrity_horizon_for_one_uniform_stratum():
    cand = Candidate("op", "t", 1024, 1, 16)
    plan = SimpleNamespace(sizes={"t": 64})
    for delta, horizon in ((1e-2, 74), (1e-6, 222)):
        assert T.horizon(plan, [cand], delta) == pytest.approx(log(1 / delta) * 1024 / 64)
        for blocks in (1, 1008, 525_600_000):
            got = T.pooled(plan, [cand], delta, blocks)
            assert got == pytest.approx(log(1 / delta) * 1024 / 64, rel=2 ** -18) and ceil(got) == horizon


def test_handoff_work_law_draws_the_plans_k_and_its_windows_are_integrity_period_logs(small):
    h = T.handoff(small.plan, small.cands, small.dag)
    rec = h["record"]
    assert rec["law"] == "work:1" and [t for t, _ in rec["templates"]] == list(rec["work_table"])
    assert rec["cap"] == min(h["statement_units"].values())
    total = sum(rec["work_table"][t]["work"] * n for t, n in rec["templates"])
    ks = {small.dag.template_id(c): small.plan.sizes[c.template] for c in small.cands if c.template in small.plan.sizes}
    for t, n in rec["templates"]:
        v = rec["work_table"][t]
        assert work_k(1, v["work"], n, total, v["floor"]) == ks[t] >= 1
    assert [w["templates"] for w in h["windows"]] == [[t] for t in rec["templates"]]
    for w in h["windows"]:
        ((t, n),), k = w["templates"], int(w["law"].removeprefix("subset:"))
        assert k == ks[t] and w["cap"] == h["statement_units"][t]
        for d in T.DELTAS:
            q = k / n
            assert w["K0"][str(d)] == min(ceil(log(len(h["windows"]) / d) / q), n - k + 1)


def test_handoff_merges_one_definition_cut_from_two_ops_at_the_higher_rate():
    cands = [Candidate("a", "a#x:F", 100, 5, 16), Candidate("b", "b#y:F", 300, 5, 16)]
    plan = SimpleNamespace(sizes={"a#x:F": 10, "b#y:F": 15}, statement_size={"a#x:F": 4, "b#y:F": 8})
    h = T.handoff(plan, cands, SimpleNamespace(template_id=lambda c: "F"))
    assert h["record"]["templates"] == [["F", 400]] and h["windows"][0]["law"] == "subset:40"
    assert h["record"]["work_table"] == {"F": {"work": 5, "floor": 40}} and h["windows"][0]["cap"] == 4


def test_large_model_fills_the_card():
    card = next(h for h in json.loads(CENSUS.read_text()) if h["id"] == "rtx-pro-6000-server/e4m3")["memory_bytes"]
    m = T.LARGE
    assert 0.9 * card < m.weight_bytes + m.kv_resident <= card and m.linear_params == 70 * 12 * 8192 ** 2 + 256 * 8192
    assert m.requests == round(T.BLOCK_S / T.LARGE_STEP_S * T.LARGE_SEQUENCES / m.positions)


def test_large_model_meets_the_target_share():
    big = _plan(T.LARGE, T.Serving(audit_store=T.LARGE.weight_bytes, replay=T.LARGE_REPLAY_S), grid=31, cut_grid=21)
    assert big.plan is not None
    r = T.recertify(big.plan, big.cands, T.DELTAS[0], exact_count=False)
    assert r["ands"] <= T.TARGET_SHARE * big.dag.work * (1 + 1e-9)
    assert {c.scale for c in big.cm.serve} >= {"requests", "pouw:h1", "bandwidth"} and big.cm.replay["request"].units \
        == T.LARGE.requests
