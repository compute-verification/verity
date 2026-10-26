"""The transformer's sampling plan (`transformer.py`): its DAG's units are their Definitions' and counted right, the
frontier prices every serve-time commitment and is monotone, the chosen plan's certificate recomputes through core's
`Stratified`, the pooled bound is IntegrityHorizon's in the uniform case, the hand-off's work law draws the plan's k,
and the large model fills the card."""

from __future__ import annotations

import json
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


@pytest.fixture(scope="module")
def small():
    m, s = T.SMALL, T.Serving(audit_store=T.SMALL.weight_bytes)
    dag, cm, objective, cands, _, front = T.plans(m, s, T.DELTAS[0], grid=31, cut_grid=21)
    plan = T.chosen(cands, cm, objective, front, T.TARGET * dag.units_dense)
    return SimpleNamespace(m=m, s=s, dag=dag, cm=cm, objective=objective, cands=cands, front=front, plan=plan)


def test_matmuls_are_their_definitions_and_the_rest_named_stubs(small):
    s8 = small.dag.s8(128)
    assert s8.ands == 22_017 and s8.out_bits <= 16
    assert small.dag.ids[s8.id] == descriptor_id(bind(G.ScaledMmS8Coordinate, K=128, **G.GEMM_SM120_S8))
    h = T.handoff(small.plan, small.cands, small.dag)
    real = {t for t, _ in h["templates"]} - set(h["stubs"])
    assert real and real <= set(small.dag.ids.values())
    assert h["stubs"] and all(s.startswith("stub:") and s[5:].startswith(CIR_100) for s in h["stubs"])


def test_units_are_counted_per_block(small):
    m, counts = small.m, small.dag.unit_counts()
    s8 = m.requests * (m.context * m.layers * (4 * m.width + m.mlp) + m.generate * m.vocab)
    assert counts[small.dag.s8(m.width).id] == s8 == 286_720
    assert counts[small.dag.real(T.F.Bf16Add).id] == m.requests * m.context * m.layers * 2 * m.width
    assert small.dag.units_dense == sum(counts.values())


def test_frontier_prices_every_serve_time_commitment_and_is_monotone(small):
    assert {c.scale for c in small.cm.serve} == {"requests", "pouw:h1", "bandwidth"}
    assert {c.scale for c in T.costs(T.LARGE, small.s).serve} >= {"carry"}
    serve = small.cm.serve_cost()
    assert serve > 0 and all(p.cost["serve"] == pytest.approx(serve) for p in small.front)
    pairs = [(p.cost["total"], p.certificate.harm) for p in small.front]
    assert all(c0 <= c1 and h0 >= h1 for (c0, h0), (c1, h1) in zip(pairs, pairs[1:]))
    assert pairs[-1][1] == 0


@pytest.mark.parametrize("delta", T.DELTAS)
def test_chosen_plan_recertifies_through_stratified(small, delta):
    plan = small.plan
    assert plan is not None and plan.certificate.harm <= T.TARGET * small.dag.units_dense
    r = T.recertify(plan, small.cands, delta)
    assert r["count_bound"] <= r["harm_bound"] + 1e-9 <= r["density_bound"] * (1 + 1e-9) + 1e-9
    assert r["certify"] == pytest.approx(r["harm_bound"], rel=1e-9)
    law = T.law(plan, small.cands)
    assert law.miss(r["count_bound"]) >= delta > law.miss(r["count_bound"] + 1)
    if delta == T.DELTAS[0]:
        assert r["harm_bound"] <= plan.certificate.harm * (1 + 1e-9)


def test_pooled_is_integrity_horizon_for_one_uniform_stratum():
    cand = Candidate("op", "t", 1024, 1, 16)
    plan = SimpleNamespace(sizes={"t": 64})
    for delta, horizon in ((1e-2, 74), (1e-6, 222)):
        for blocks in (1, 1008, 525_600_000):
            got = T.pooled(plan, [cand], delta, blocks)
            assert got == pytest.approx(log(1 / delta) * 1024 / 64) and ceil(got) == horizon


def test_handoff_work_law_draws_the_plans_k(small):
    h = T.handoff(small.plan, small.cands, small.dag)
    assert h["law"] == "work:1" and [t for t, _ in h["templates"]] == list(h["work_table"])
    total = sum(h["work_table"][t]["work"] * n for t, n in h["templates"])
    ks = {small.dag.template_id(c): small.plan.sizes[c.template] for c in small.cands if c.template in small.plan.sizes}
    for t, n in h["templates"]:
        v = h["work_table"][t]
        assert work_k(1, v["work"], n, total, v["floor"]) == ks[t] >= 1


def test_large_model_fills_the_card():
    card = next(h for h in json.loads(CENSUS.read_text()) if h["id"] == "rtx-pro-6000-server/e4m3")["memory_bytes"]
    m = T.LARGE
    used = m.weight_bytes + m.inflight * m.context * m.kv_bytes
    assert 0.9 * card < used <= card and 75e9 < m.linear_params < 85e9
