"""Demo A's frontier (`demo_a.py`): the Definition map's subcircuits at their per-token counts and staged k_log, each unit's work
its own ANDs and every unit its output bits, statements shared, core's risk wherever core computes it, the certificate
`certify`'s where core's exact count is cheap and its location term never below it, at each budget the plan under the
exchange rate no worse than either half's best alone, within the budget, the least verification phase the first whose share of its
GPU time affords a plan, PoUW's lanes their traced chunks and combine sharing statements, and M3's uniform draw and the
tile strata's epsilon as core and PoUW state them."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import floor, fsum, log1p

import pytest

import demo_a as D
from verity.core.protocols.pouw.window import k_prime
from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import location_bits
from verity.core.protocols.sampled_proofs.optimize import Objective, certify

TINY = replace(D.SHAPE, batch=4, step_s=0.01)

#: the Definition map's table: units a token and a generated token, by Definition
FAMILIES = {"ScaledMmS8CoordinateRows_v1{K=16384": (5_118_464, 256), "ScaledMmS8CoordinateRows_v1{K=65536": (851_968, 0),
            "AllGather_v1{WORLD=64,": (13_416, 0), "AllGather_v1{WORLD=128,": (40_064, 2),
            "AllGather_v1{WORLD=2,": (6_656, 0), "RowAbsMax_v1{N=16384,RELU=True": (156, 0),
            "RowAbsMax_v1{N=16387": (52, 0), "QuantChunkS8_v1{N=8192,RELU=True": (416, 0),
            "RowAbsMax_v1{N=16384,RELU=False": (156, 1), "QuantChunkS8_v1{N=8192,RELU=False": (312, 2),
            "RmsSumSqPart_v1": (13_312, 128), "RmsNormOutSums_v1": (13_312, 128), "Bf16AddRow_v2": (13_312, 0),
            "EmbeddingCoordinate_v1": (16_384, 0), "Argmax_v2": (0, 1)}


def test_the_partition_is_families_table_within_the_cap():
    m, fs = D.SHAPE, D.families()
    by = {t.id: t for t in fs}
    for prefix, (token, gen) in FAMILIES.items():
        (t,) = [t for i, t in by.items() if i.startswith(prefix)]
        assert t.units == m.batch * (m.positions * token + m.generate * gen), prefix
    for n in range(1, m.positions + 1):
        qk = next(t for t in fs if t.id.startswith(f"QkSoftmaxRow_v1{{T={n},"))
        pv = next(t for t in fs if t.id.startswith(f"PvRow_v1{{T={n},"))
        assert qk.units == pv.units == m.batch * 13_312 and qk.out_bits == 16 * n and pv.out_bits == 1024
    assert [D.qk_ands(t) for t in (1, 8, 16, 32)] == [39_924, 303_754, 605_290, 1_208_362]
    assert [D.pv_ands(t) for t in (1, 8, 31)] == [111_104, 321_024, 1_009_152]
    assert [D.attention_k_log(t) for t in (6, 7, 14, 15, 30, 31)] == [22, 23, 23, 24, 24, 25]
    def k_of(t):
        return int(t.id.split("K=")[1].split(",")[0])

    tiles = [t for t in fs if t.id.startswith(D.TILE)]
    s8_macs = sum(t.units * k_of(t) for t in fs if t.id.startswith(D.S8))
    assert sum(t.units * D.TILE_OUTPUTS * k_of(t) for t in tiles) == s8_macs
    assert len(tiles) == len(D.TILE_TRACED) and len(D.families(tiles=False)) == len(fs) - len(tiles)
    chunk, combine = D.tile_parts()
    for t in tiles:
        ands, statements = D.TILE_TRACED[k_of(t)]
        assert t.ands == ands == t.chunks * chunk + combine and t.out_bits == statements == t.chunks + 1
        assert t.chunks * D.TILE_CHUNK_MACS == D.TILE_OUTPUTS * k_of(t)
        assert t.cap * t.chunks <= 2 ** (35 - t.k_log) * t.groups and (t.cap + 1) * t.chunks > 2 ** (35 - t.k_log)
    assert [(t.cap, t.groups) for t in tiles] == [(2, 1), (1, 2)]
    assert max(t.k_log for t in fs) <= D.K_LOG_MAX
    assert all(t.cap == min(2 ** (35 - t.k_log), 2 ** 20) for t in fs if not t.chunks)
    one, week = D.candidates(), D.candidates(blocks=7)
    by_c = {t.id: t for t in fs}
    for c, w in zip(one, week):
        t = by_c[c.template]
        commit = D.TILE_COMMIT_BITS * t.chunks if t.chunks else c.out_bits
        assert c.work == c.ands and c.influence == 0 and c.commit_bits == commit and c.out_bits > 0 \
            and w.units == 7 * c.units
    cm, rates = D.costs(), replace(D.costs(), templates={})
    assert cm.shared and cm.commit_per_bit == pytest.approx(1 / (8 * 225e9))
    assert all(cm.statement(c) == rates.statement(c) and cm.unit(c) == rates.unit(c)
               for c in one if not by_c[c.template].chunks)
    proved = chunk + sum(D.row_ands(b) for b in D.TILE_ROWS)
    for c in one:
        t = by_c[c.template]
        if t.chunks:
            a0 = t.groups * (cm.statement_fixed + cm.statement_per_and * proved) + cm.statement_fixed \
                + cm.statement_per_and * combine
            assert cm.statement(c) == (pytest.approx(a0), 0.0) and cm.unit(c) == rates.unit(c)
    assert D.costs(blocks=7).replay["request"].units == 7 * m.batch
    assert all(D.costs(blocks=7).replay[t.id].units == 7 * t.units for t in tiles)
    assert m.block_s == pytest.approx(32 * 6.35)
    g = D.ands_against_gpu()
    assert sum(d["ands_share"] for d in g["definitions"].values()) == pytest.approx(1)


def test_the_risk_is_cores_where_core_computes_it():
    for s in (Stratum("a", 400, 30), Stratum("b", 2 ** 40, 300), Stratum("c", 2 ** 20, 20_000)):
        memo = D._Memo(s.name, s.size, s.k)
        for m in (0, 1, 17, s.size - s.k, s.size - s.k + 1):
            assert memo._risk(m) == s._risk(m)
    huge = Stratum("d", 2 ** 60, 3)
    with pytest.raises(ValueError):
        huge._risk(huge.size - 3)
    exact = Stratified((huge,)).rho({"d": huge.size - 3})
    assert D._Memo("d", huge.size, 3)._risk(huge.size - 3) == pytest.approx(exact, rel=1e-12)
    wide = D._Memo("e", 2 ** 57, 20_000)
    assert wide._risk(10 ** 6) == pytest.approx(fsum(-log1p(-10 ** 6 / (2 ** 57 - i)) for i in range(20_000)),
                                                rel=1e-12)


def test_location_is_location_bits_at_the_exact_count():
    law = Stratified((Stratum("a", 400, 30), Stratum("b", 90, 9)))
    profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, 0.01)
    assert D.location(law.population, law.count_bound(0.01)) == location_bits(profile)
    assert floor(law.harm_bound({"a": 1.0, "b": 1.0}, 0.01)) >= law.count_bound(0.01)


def test_the_certificate_is_certifys_on_small_strata():
    cands = [replace(c, units=60 + 7 * i, replay="", per_replay=0) for i, c in enumerate(D.candidates()[:6])]
    sizes = {c.template: 40 for c in cands}
    objective = Objective(0.01, work=1e-9, bits=1e-3)
    ours, core = D.certificate(sizes, cands, objective), certify(sizes, cands, objective)
    assert ours.free_compute == core.free_compute and ours.harm == core.harm
    assert ours.leak >= core.leak and ours.leak == pytest.approx(core.leak, rel=1e-6)


def test_each_budget_certifies_no_less_than_either_half_alone():
    cands, cm = D.candidates(TINY), D.costs(TINY)
    s = D.sweep(cands, cm, TINY, budgets=(1e-3, 3e4))
    assert s["skipped"] == [1e-3 * TINY.block_s]
    (p,) = s["points"]
    assert p.plan.cost["total"] <= p.budget + s["commit"] * (1 + 1e-9)
    at = p.at["0.01"]
    assert at["free_compute"] >= p.h_work * (1 - 1e-9) and at["leak"] >= p.h_bits * (1 - 1e-9)
    assert p.at["1e-06"]["free_compute"] >= at["free_compute"] and p.at["1e-06"]["leak"] >= at["leak"]


def test_the_tile_epsilon_is_the_least_whose_k_prime_the_draw_meets():
    cands = D.candidates()
    tiles = [c for c in cands if c.template.startswith(D.TILE)]
    by = D.tile_eps({c.template: 1760 for c in tiles}, cands)
    assert set(by) == {c.template for c in tiles}
    for tile in tiles:
        eps = by[tile.template]
        for d, e in eps.items():
            f = Fraction(e).limit_denominator(2 ** 32)
            n, w = tile.units, tile.ands
            assert k_prime(f, Fraction(float(d)), n * w, n, w) <= 1760 < k_prime(f - Fraction(1, 2 ** 32),
                                                                                 Fraction(float(d)), n * w, n, w)
        assert eps["1e-06"] > eps["0.01"]
    assert D.tile_eps_max(by) == {d: max(e[d] for e in by.values()) for d in ("0.01", "1e-06")}
    assert D.tile_eps({}, cands) is None and D.tile_eps_max(None) == {}


def test_the_least_sampling_block_is_the_first_that_affords_a_plan():
    n = D.least_blocks()
    assert n > 1
    assert D.cheapest(D.candidates(blocks=n), D.costs(blocks=n)) <= D.BUDGET_SHARE * n * D.SHAPE.block_s
    assert D.cheapest(D.candidates(blocks=n - 1), D.costs(blocks=n - 1)) > D.BUDGET_SHARE * (n - 1) * D.SHAPE.block_s
    one = D.plan_at(D.SHAPE, "one block", 1)
    assert not one["affordable"] and "at" not in one
    assert D.spans(periods={"week": 7 * 86400})["week"] == floor(7 * 86400 / D.SHAPE.block_s)


def test_m3_draws_a_uniform_eight_and_certifies_the_heaviest_units():
    o = D.m3()
    assert o["of_block"] == o["cost"] / D.SHAPE.block_s > 0
    assert o["of_budget"] == pytest.approx(o["of_block"] / D.BUDGET_SHARE)
    law = Stratified.uniform(o["units"], D.M3_UNITS)
    for d in D.DELTAS:
        at = o["at"][str(d)]
        assert at["wrong_units"] == floor(law.harm_bound({"unit": 1.0}, d))
        assert at["wrong_units"] == pytest.approx(o["units"] * (1 - d ** (1 / D.M3_UNITS)), rel=3e-6)
        assert 0 < at["work_share"] < 1 and 0 < at["bits_share"] <= 1
    assert o["at"]["1e-06"]["free_compute"] > o["at"]["0.01"]["free_compute"]
    assert D.m3(tiles=True)["units"] > o["units"]
    lanes = {k: D.m3(only=f"{D.TILE}{{H1=v5,K={k},") for k in sorted(D.TILE_TRACED)}
    cands, cm = D.candidates(), D.costs()
    for k, x in lanes.items():
        (c,) = [c for c in cands if c.template.startswith(f"{D.TILE}{{H1=v5,K={k},")]
        a0, a1 = cm.statement(c)
        drawn = D.M3_UNITS * (a1 + cm.unit(c) + cm.of(c).replay + D.TILE_REQUESTS * D.SHAPE.block_s / D.SHAPE.batch)
        assert x["cost"] == pytest.approx(-(-D.M3_UNITS // c.cap) * a0 + drawn, rel=1e-12)
    both = D.m3(only=D.TILE)
    assert lanes[16_384]["cost"] < both["cost"] < lanes[65_536]["cost"]


def test_the_knee_is_where_the_certified_shares_stop_falling_fast():
    def at(share):
        return {"0.01": {"free_compute": share, "leak": share}}

    pts = [D.Point(b, 1.0, 1.0, None, at(s)) for b, s in ((1, 1.0), (2, 1e-3), (4, 5e-4), (8, 2.5e-4))]
    assert D.knee(pts, 1.0, 1.0) == 1
