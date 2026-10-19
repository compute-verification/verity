"""The budget calculator's terms are NOTATION's and cost-model/v1's, its sweeps match its families and OPT2's frontier,
and its allocation is the least ε within each budget."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_budget", Path(__file__).parents[1] / "budget.py")
B = importlib.util.module_from_spec(spec)
sys.modules["pouw_budget"] = B
spec.loader.exec_module(B)

FAM = {f.name: f for f in B.FAMILIES}
#: γ at y = 0, the machine model's on each family's measured kernel (budget_fill.json's first point)
GAMMA = {"sm_120": 2.1868, "H200": 2.3435, "B200": 2.5611, "L40S": 3.0144, "A100": 4.2263}


def test_luck_is_the_drawn_units_deficient_branch_and_the_sum_bounds_the_mix():
    assert B.DEFICIT == pytest.approx(0.002368, rel=1e-3)
    for f in B.FAMILIES:
        t = B.today(f)
        assert t.eps_luck == pytest.approx(t.eps_sample * t.sigma_w / (1 - B.P_PASS))
        assert t.eps == pytest.approx(t.core + t.eps_sample)
        mix = t.eps_skip + t.eps_served + t.eps_machine + t.eps_sample * max(t.sigma_w / (1 - B.P_PASS), 1)
        assert t.eps >= mix
    h = B.today(FAM["H200"])
    assert h.sigma_w > 1 - B.P_PASS and h.eps_luck > h.eps_sample
    two_bit = B.point(FAM["H200"], h.config, h.B_sample, p=7 / 16)
    assert two_bit.eps_luck == pytest.approx(h.eps_luck * 0.25 / (9 / 16))
    assert two_bit.eps_sample == h.eps_sample


def test_today_on_sm_120_is_notations_terms_and_b_its_parts():
    t = B.today(FAM["sm_120"])
    assert t.config == B.Config("int", 0.0, False, "plain")
    assert round(100 * t.eps_skip, 3) == 0.362
    assert round(100 * t.eps_machine, 3) == 0.125
    assert t.eps_served == 0
    assert t.draws == pytest.approx(B.TODAY_DRAWS, rel=1e-6)
    assert t.eps_sample == pytest.approx(B.LN_INV_DELTA / B.TODAY_DRAWS, rel=0.02)
    assert t.B == pytest.approx(t.B_transform + t.B_hash + t.B_sample)
    assert t.B_transform == pytest.approx(GAMMA["sm_120"] - 1, abs=1e-4)
    assert t.B_hash == pytest.approx(0.060 + 0.016 + 0.019)
    assert t.B_sample == pytest.approx(B.spend_for(B.TODAY_DRAWS)) and 0.3 < t.B_sample < 0.4


def test_sampling_data_is_opt2s_frontier_on_the_grid():
    doc = json.loads(B.SAMPLING_DATA.read_text())
    assert set(doc["regimes"]) == set(B.REGIMES) and "2^-41" in doc["source"]
    for name, (commit, statement) in B.REGIMES.items():
        r = doc["regimes"][name]
        assert (r["commit"], r["statement"]) == (commit, statement)
        xs, es, ds, us, cuts = B.curve(name)
        assert set(xs) <= set(B.SPEND_GRID) and len(xs) >= len(B.SPEND_GRID) - 1
        assert all(a > b for a, b in zip(es, es[1:])) and all(a < b for a, b in zip(ds, ds[1:]))
        assert all(0 < u < 1 for u in us)
    plain = dict(zip(B.curve("plain")[0], B.curve("plain")[1]))
    for x, e in plain.items():
        if 0.01 <= x <= 1:
            assert e * x == pytest.approx(1.0e-4 * 41 / 40, rel=0.03)
            assert B.curve("plain")[4][B.curve("plain")[0].index(x)] == "coord"
    for name in ("sessions/256", "prefix, sessions/256"):
        for x, e in zip(*B.curve(name)[:2]):
            assert e < plain[x] / 50


def test_spend_and_draws_interpolate_through_the_points():
    xs, es, ds, _, _ = B.curve("plain")
    for x, e, d in list(zip(xs, es, ds))[5:-5]:
        s = B.sample(FAM["sm_120"], x)
        assert s.eps == pytest.approx(e) and s.draws == pytest.approx(d)
        assert B.spend_for(d) == pytest.approx(x)
    assert B.sample(FAM["sm_120"], xs[0] / 2) is None


def test_a_family_buys_provers_at_the_price_ratio():
    sm, h200 = FAM["sm_120"], FAM["H200"]
    assert h200.prover == pytest.approx(1 / 3)
    for b in (0.01, 0.1, 1.0):
        assert B.sample(h200, b).eps == pytest.approx(B.sample(sm, 3 * b).eps)
    assert B.today(h200).B_sample == pytest.approx(B.today(sm).B_sample / 3)


def test_filler_grows_each_drawn_units_cost():
    f = FAM["sm_120"]
    s0, s16 = B.sample(f, 0.3), B.sample(f, 0.3, y=16)
    u = B.curve("plain")[3][B.curve("plain")[0].index(0.316228) - 1]
    assert s16.spend == pytest.approx(s0.spend / (1 + u))
    assert s16.eps > s0.eps


def test_hash_leaves_runs_through_commit_gpus_points():
    assert B.hash_leaves(64) == pytest.approx(0.101)
    assert B.hash_leaves(128) == pytest.approx(0.060)
    assert B.hash_leaves(64) > B.hash_leaves(128) > B.hash_leaves(256) > 0


def test_fill_data_is_the_families_on_the_grid():
    doc = json.loads(B.FILL_DATA.read_text())
    assert set(doc["families"]) == set(FAM)
    for name, f in FAM.items():
        d = doc["families"][name]
        assert d["device"] == f.device and d["design"] == f.design_name
        assert "int" in f.fill and set(f.fill) <= set(B.FILL_MIXES)
        if "ffma" in f.fill:
            assert f.fill["ffma"] != f.fill["int"]
        for mix, pts in f.fill.items():
            assert tuple(p[0] for p in pts) == B.FILL_GRID
            assert pts[0][1] == pytest.approx(GAMMA[name], abs=1e-4)
            assert all(p[1] >= pts[0][1] for p in pts)
            assert all(min(p[2:]) >= 0 for p in pts)
        assert f.fill["int"][0] == f.fill.get("ffma", f.fill["int"])[0]


def test_served_check_is_free_and_takes_its_share_of_skip():
    for f in B.FAMILIES:
        t = B.today(f)
        r = next(r for r in B.levers(f) if r["lever"].startswith("check P·V's served output"))
        assert r["d_B"] == 0
        assert r["d_eps"] == pytest.approx(-f.served * t.eps_skip)
        assert r["d_eps"] < 0


def test_batching_frees_most_of_todays_sampling_spend():
    for f in B.FAMILIES:
        t = B.today(f)
        r = next(r for r in B.levers(f) if r["lever"] == "today's draws in regime sessions/256")
        assert -r["d_B"] > 0.99 * t.B_sample
        assert abs(r["d_eps"]) < 1e-3 * t.eps


@pytest.mark.parametrize("regime", B.ALLOCATED)
def test_allocation_fits_each_budget_and_eps_falls_with_it(regime):
    for f in B.FAMILIES:
        t = B.today(f)
        prev = math.inf
        for m, q, _ in B.allocation(f, regime=regime):
            assert q.config.regime == regime and q.config.served
            assert q.B == pytest.approx(m * t.B)
            assert q.eps <= prev
            prev = q.eps
        assert B.allocation(f, (1,), regime=regime)[0][1].eps < t.eps


@pytest.mark.parametrize("regime", B.ALLOCATED)
def test_optimum_has_no_better_step_within_its_budget(regime):
    """No marginal step the optimum reports lowers ε and still fits the budget, and every step down costs ε."""
    for f in B.FAMILIES:
        t = B.today(f)
        for m, q, mg in B.allocation(f, (2, 5, 10), regime=regime):
            assert mg
            for name, de, db in mg:
                if de < 0:
                    assert q.B + db > m * t.B, (f.name, m, name)
                if db < 0:
                    assert de >= 0, (f.name, m, name)


def test_integer_filler_never_beats_the_unrestricted_optimum():
    for f in B.FAMILIES:
        t = B.today(f)
        for regime in B.ALLOCATED:
            for m in (2, 5, 10):
                free = B.optimum(f, m * t.B, regime=regime)
                exact = B.optimum(f, m * t.B, B.unconditional(f), regime)
                assert exact.config.mix not in B.CONDITIONAL
                assert free.eps <= exact.eps


def test_main_writes_every_familys_tables(tmp_path, capsys):
    md, js = tmp_path / "budget.md", tmp_path / "budget.json"
    assert B.main(["--md", str(md), "--json", str(js)]) == 0
    text = md.read_text()
    assert capsys.readouterr().out.strip() == text.strip()
    assert "## Headline: ε = core + sampling" in text
    for name in FAM:
        assert f"## {name}:" in text
    assert "integer filler" in text and "Regime `sessions/256`" in text
    doc = json.loads(js.read_text())
    assert set(doc) == set(FAM)
    for name, d in doc.items():
        assert set(d["allocation"]) == set(B.ALLOCATED)
        for regime in B.ALLOCATED:
            assert [a["mult"] for a in d["allocation"][regime]] == [1, 2, 5, 10]
            assert all(a["point"]["config"]["mix"] not in B.CONDITIONAL
                       for a in d["allocation_integer_filler"][regime])
