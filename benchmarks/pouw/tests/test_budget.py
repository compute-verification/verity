"""The budget calculator's terms are NOTATION's and PRF-57's, its sweep matches its families, and its allocation is the
least ε within each budget."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_budget", Path(__file__).parents[1] / "budget.py")
B = importlib.util.module_from_spec(spec)
sys.modules["pouw_budget"] = B
spec.loader.exec_module(B)

FAM = {f.name: f for f in B.FAMILIES}
#: γ at y = 0, the machine model's on each family's measured kernel (budget_fill.json's first point)
GAMMA = {"sm_120": 2.1868, "H200": 1.7171, "B200": 4.0704, "L40S": 3.0144, "A100": 4.2263}


def test_eps_samp_is_the_worse_of_wrong_and_deficient_units():
    unit = B.LN_INV_DELTA / B.ROUNDS
    assert B.eps_samp(2, 0.218) == pytest.approx(unit / 2)
    assert B.eps_samp(2, 0.308) == pytest.approx(unit / 2 * 0.308 / 0.25)
    assert B.eps_samp(8, 0.308) == pytest.approx(B.eps_samp(2, 0.308) / 4)
    assert B.eps_samp(2, 0.308, p=7 / 16) == pytest.approx(unit / 2)
    assert round(100 * B.eps_samp(2, 0.0), 4) == 0.0296
    assert B.DEFICIT == pytest.approx(0.002368, rel=1e-3)


def test_today_on_sm_120_is_notations_terms_and_b_its_parts():
    t = B.today(FAM["sm_120"])
    assert t.config == B.Config("int", 0.0, 2, 2, False)
    assert round(100 * t.eps_skip, 3) == 0.362
    assert round(100 * t.eps_machine, 3) == 0.125
    assert t.eps_samp == B.eps_samp(2, 0.0) and t.sigma_w < 1 - B.P_PASS
    assert t.eps_served == 0
    assert t.eps == pytest.approx(t.eps_skip + t.eps_machine + t.eps_samp)
    assert t.B == pytest.approx(t.B_transform + t.B_hash + t.B_prove + t.B_verify)
    assert t.B_transform == pytest.approx(GAMMA["sm_120"] - 1, abs=1e-4)
    assert t.B_hash == pytest.approx(0.060 + 0.016 + 0.019)
    cpu = B.rate(FAM["sm_120"]) * B.ROUND_S
    assert t.B_prove == pytest.approx((12.13 * 32 / 2 + 2 * 1.62 * 0.1167 * 32) / cpu)


def test_h200s_deficient_branch_binds_today():
    t = B.today(FAM["H200"])
    assert t.sigma_w > 1 - B.P_PASS
    assert t.eps_samp > B.eps_samp(2, 0.0)


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


def test_allocation_fits_each_budget_and_eps_falls_with_it():
    for f in B.FAMILIES:
        t = B.today(f)
        prev = t.eps
        for m, p, _ in B.allocation(f):
            assert p.B <= m * t.B
            assert B.feasible(p.config)
            assert p.config.s * p.config.b * B.UNIT_4K <= B.SESSION_CAP
            assert p.eps <= prev
            prev = p.eps
        assert B.allocation(f, (1,))[0][1].eps < t.eps


def test_optimum_has_no_better_step_within_its_budget():
    """No marginal step the optimum reports lowers ε and still fits the budget, and every step down costs ε."""
    for f in B.FAMILIES:
        t = B.today(f)
        for m, p, mg in B.allocation(f, (2, 5, 10)):
            assert mg
            for name, de, db in mg:
                if de < 0:
                    assert p.B + db > m * t.B, (f.name, m, name)
                if db < 0:
                    assert de >= 0, (f.name, m, name)


def test_integer_filler_never_beats_the_unrestricted_optimum():
    for f in B.FAMILIES:
        t = B.today(f)
        for m in (2, 5, 10):
            free = B.optimum(f, m * t.B)
            exact = B.optimum(f, m * t.B, B.unconditional(f))
            assert exact.config.mix not in B.CONDITIONAL
            assert free.eps <= exact.eps


def test_cost_variants_move_only_their_part():
    f = FAM["sm_120"]
    t = B.today(f)
    work, large = B.today(f, fixed="work"), B.today(f, verify="large")
    assert work.B_prove < t.B_prove and work.eps == t.eps
    assert large.B_verify > t.B_verify and large.B_prove == t.B_prove
    eight = B.today(f, gpus=8)
    assert eight.B_prove < t.B_prove and eight.B_transform == t.B_transform


def test_main_writes_every_familys_tables(tmp_path, capsys):
    md, js = tmp_path / "budget.md", tmp_path / "budget.json"
    assert B.main(["--md", str(md), "--json", str(js)]) == 0
    text = md.read_text()
    assert capsys.readouterr().out.strip() == text.strip()
    for name in FAM:
        assert f"## {name}:" in text
    assert "integer filler" in text
    doc = json.loads(js.read_text())
    assert set(doc) == set(FAM)
    for name, d in doc.items():
        assert [a["mult"] for a in d["allocation"]] == [1, 2, 5, 10]
        assert all(a["point"]["config"]["mix"] not in B.CONDITIONAL for a in d["allocation_integer_filler"])
