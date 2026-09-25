"""Driver / calibration / analytic-model tests for the F/G coarse sweep (``sweeps/calibration.py``, ``sweeps/analytic.py``,
``sweeps/coarse_run.py``, ``sweeps/coarse_report.py``).  Nothing here builds a model larger than tiny2."""

from __future__ import annotations

import json
import math

import pytest

from accumulation.algorithms.registry import build
from accumulation.bounds.adw import adw
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.sweeps import analytic
from accumulation.sweeps.calibration import calibrate_cell, calibration_alg
from accumulation.sweeps.coarse_report import _knee, k_fits
from accumulation.sweeps.coarse_run import (AXES, Cell, DEFAULT_POLICY, finalize_row, new_row, preset_cells, row_key, sweep, too_heavy)


# ------------------------------------------------------------------------------------------------ calibration

def test_calibration_is_checker_work_of_inference_dense():
    r = calibrate_cell("tiny2", 8, log=lambda *a: None)
    assert r["error"] is None
    assert r["calibration_alg"] == calibration_alg("tiny2") == "inference-dense"
    a = adw(extract(build("inference-dense", MODELS["tiny2"], Workload(8, 8))))
    assert r["fwd_work"] == a.all_work and r["fwd_macs"] == a.all_macs
    assert r["fwd_work"] > r["fwd_macs"]                         # all ops, not only MACs
    assert r["session_over_fwd"] <= 1.0                          # the honest session is one RU at F_hat = G_hat = 1
    assert r["policy_points"]["F_hat=1"] == r["fwd_work"] and r["policy_points"]["G_hat=1"] == r["fwd_work"]
    assert r["w_tl"] * 8 * MODELS["tiny2"].layers == pytest.approx(r["step_all_work"])
    assert r["w_c"] * 8 * MODELS["tiny2"].layers == pytest.approx(r["step_credited_macs"])


def test_calibration_work_equals_single_unit_plan_work():
    from accumulation.bounds import coarse
    bp = build("inference-dense", MODELS["tiny2"], Workload(8, 8))
    g = extract(bp)
    a = adw(g)
    plan = coarse.upper_coarse(g, F=a.all_work, G=a.all_work, program=bp.program)
    st = plan.ru_stats()
    assert st["n_ru"] == 1 and st["work_max"] == a.all_work


# --------------------------------------------------------------------------------------------------- analytic

EIGHT_B = dict(d=4096, P_l=2.18e8, layers=32, w_tl=1.165e9, w_c=9.38e8, P=1.606e10)   # llama3-8b, Q_inf = 8192 calibration
FWD_8B = 7.965e13


def test_steady_state_matches_user_numbers_at_8b_default():
    ss = analytic.steady_state(F=1.5 * FWD_8B, G=4 * FWD_8B, Q_step=131072, **EIGHT_B)
    assert ss["A"] == pytest.approx(1.5 * 1.5 * FWD_8B / EIGHT_B["w_tl"], rel=1e-6)          # F-limited: G_hat 4 > 1.5 F_hat
    assert 1.4e5 < ss["A"] < 1.6e5
    assert "F-limited" in ss["regime"]
    ss8 = analytic.steady_state(F=1.5 * FWD_8B, G=4 * FWD_8B, Q_step=8192, **EIGHT_B)
    assert ss8["c_W"] == 2 and 2.2 < ss8["m_star"] < 2.6 and 6e4 < ss8["u_star"] < 7e4
    assert 0.42e6 < ss8["b_train_est"] < 0.46e6                                              # ~0.44 MB/token
    assert ss8["b_train_est"] == pytest.approx(ss8["b_closed"], rel=0.02)                    # scan agrees with the closed form
    assert ss8["b_train_est"] >= ss8["row_floor"]
    assert not ss8["weight_floor_applies"]                                                   # L Q_step = 262k > A
    assert 0.72e6 < ss["b_subunit_est"] < 0.78e6                                             # c_W = 6 branch at Q_step = 131072: ~0.75


def test_g_irrelevant_above_1p5_f():
    a = analytic.steady_state(F=1.5 * FWD_8B, G=4 * FWD_8B, Q_step=8192, **EIGHT_B)
    b = analytic.steady_state(F=1.5 * FWD_8B, G=100 * FWD_8B, Q_step=8192, **EIGHT_B)
    c = analytic.steady_state(F=1.5 * FWD_8B, G=1.0 * FWD_8B, Q_step=8192, **EIGHT_B)
    assert a["b_train_est"] == b["b_train_est"]
    assert c["b_train_est"] > a["b_train_est"] and "G-limited" in c["regime"]


def test_transient_and_knee():
    kw = dict(d=4096, layers=32, w_tl=1.165e9, F=1.5 * FWD_8B, G=4 * FWD_8B, Q_step=8192)
    t2 = analytic.transient(K=2, **kw)
    assert t2["m_t"] > 2 and t2["b_transient"] < 4 * 4096 * 32 / 2                          # more than a 2-layer chunk fits at K = 2
    assert 4 * 4096 * 32 / 2 == pytest.approx(0.262e6, rel=0.01)                            # the measured K = 2, 4 value
    kn = analytic.knee(b_ss=0.4376e6, **kw)
    assert 8 <= kn["K_m2"] <= 12 and 12 <= kn["K_knee"] <= 18
    curve = analytic.k_curve(b_ss=0.4376e6, P=1.606e10, Ks=(2, 4, 8, 16, 32), **kw)
    assert [c["regime"] for c in curve] == ["transient"] * 3 + ["steady"] * 2
    assert all(c["b_marginal"] <= 0.4376e6 + 1e-6 for c in curve)


def test_single_step_block_is_labelled_alternative():
    e = analytic.estimate_point("llama3-8b", Q_inf=8192, F_hat=1.5, G_hat=4, Q_step=131072, calib={})
    assert e["source"] == "analytic MACs" and e["b_train_est"] and e["b_single_est"]
    assert e["penalty_est"] > 1e3 and e["b_inf"] == 4.0


def test_measured_knee_detection():
    Q = 8192
    pts = [(K, 1.6e10 + 0.26e6 * Q * K) for K in (2, 4, 8)] + [(K, 1.6e10 + 0.26e6 * Q * 8 + 0.44e6 * Q * (K - 8)) for K in (12, 16, 24)]
    knee, marg = _knee(pts, Q)
    assert knee == 8
    assert marg[0] == pytest.approx(0.26e6) and marg[-1] == pytest.approx(0.44e6)
    assert _knee(pts[:3], Q)[0] is None


def test_k_fits_use_post_knee_segment():
    Q = 8192
    rows = []
    for K in (2, 4, 8, 12, 16, 24):
        I = 1.6e10 + 0.26e6 * Q * min(K, 8) + 0.44e6 * Q * max(K - 8, 0)
        r = new_row(Cell("llama3-8b", "local-sgd", 8192, 1.5, 4.0, Q, K, 8192))
        r.update(U=I, L=I / 2, M=1e12 * K)
        rows.append(finalize_row(r))
    idx = {row_key(r): r for r in rows}
    f = k_fits(idx, "llama3-8b", 8192, 1.5, 4.0, (Q,), Ks=(2, 4, 8, 12, 16, 24))[0]
    fu = f["fits"]["U"]
    assert fu["knee_K"] == 8 and fu["segment"] == "post-knee" and fu["Ks"] == [8, 12, 16, 24]
    assert fu["b_per_token"] == pytest.approx(0.44e6, rel=1e-6)
    assert fu["transient_b_per_token"] == pytest.approx(0.26e6, rel=1e-6)
    assert f["knee_pred"] is not None


# ------------------------------------------------------------------------------------------------------ driver

def test_ksweep_presets_extend_past_the_knee():
    ks = preset_cells("ksweep")
    assert sorted(c.K for c in ks if c.Q_step == 8192 and c.K) == sorted(AXES["ksweep"]) and max(AXES["ksweep"]) >= 48
    assert sorted(c.K for c in ks if c.Q_step == 32768 and c.K) == [2, 3, 4, 6, 8]
    assert all(too_heavy(c) == (c.tokens > 140000) for c in ks if c.K)


def test_too_heavy_marks_pod_cells():
    assert too_heavy(Cell("llama3-70b", "local-sgd", 8192, 1.5, 4.0, 131072, 2, 8192))
    assert too_heavy(Cell("llama3-8b", "local-sgd", 8192, 1.5, 4.0, 131072, 2, 8192))
    assert too_heavy(Cell("llama3-8b", "local-sgd", 131072, 1.5, 4.0, 131072, 1, 131072))
    assert not too_heavy(Cell("llama3-8b", "local-sgd", 8192, 1.5, 4.0, 8192, 16, 8192))
    assert not too_heavy(Cell("llama3-1b", "local-sgd", 8192, 1.5, 4.0, 8192, 48, 8192))
    assert not too_heavy(Cell("llama3-70b", "inference-session", 8192, 1.5, 4.0, None, None, 8192))
    assert too_heavy(Cell("llama3-405b", "inference-session", 131072, 1.5, 4.0, None, None, 131072))


def test_sweep_marks_pod_and_redoes_stale_basis(tmp_path):
    out = tmp_path / "coarse.jsonl"
    cal = {"model": "tiny2", "Q_inf": 8, "fwd_work": 7464, "fwd_macs": 3584, "w_tl": 1.0, "error": None}
    (tmp_path / "calibration.json").write_text(json.dumps({"rows": [cal]}))
    stale = new_row(Cell("tiny2", "inference-session", 8, 1.0, 1.0, None, None, 8))
    stale.update(F=3584, G=3584, fwd_macs=3584, U=32)                     # MAC-unit basis -> must be redone
    out.write_text(json.dumps(finalize_row(stale)) + "\n")
    logs = []
    heavy = Cell("llama3-70b", "local-sgd", 8192, 1.5, 4.0, 131072, 2, 8192)
    rows = sweep([Cell("tiny2", "inference-session", 8, 1.0, 1.0, None, None, 8), heavy], out, dry_run=True, log=logs.append)
    assert any("stale F/G basis" in m for m in logs)
    by = {row_key(r): r for r in rows}
    assert by[heavy.key()]["status"] == "pod" and by[heavy.key()]["error"] is None
    fresh = by[Cell("tiny2", "inference-session", 8, 1.0, 1.0, None, None, 8).key()]
    assert fresh.get("U") is None                                          # stale row replaced, to be re-run


def test_default_policy_unchanged():
    assert DEFAULT_POLICY == {"Q_inf": 8192, "F_hat": 1.5, "G_hat": 4, "Q_step_mult": 16}


# ------------------------------------------------------------------------------------------- planner memory

def test_planner_memo_caps_do_not_change_the_plan(monkeypatch):
    """``_Ctx._res_cache`` / ``_reads_cache`` are pure memos (bounded since they grew ~13 MB per unit evaluation at 65k
    tokens, 3.7 GB per cell); clearing them at any size must leave U unchanged."""
    from accumulation.bounds import coarse
    cfg = MODELS["tiny2"]
    bp = build("local-sgd", cfg, Workload(16, 8, local_steps=2))
    g = extract(bp)
    fwd = adw(extract(build("inference-dense", cfg, Workload(8, 8)))).all_work
    ref = coarse.upper_coarse(g, F=int(1.5 * fwd), G=int(4 * fwd), program=bp.program)
    monkeypatch.setattr(coarse, "_RES_CACHE_KEYS", 5)
    monkeypatch.setattr(coarse, "_RES_CACHE_HARD", 7)
    monkeypatch.setattr(coarse, "_READS_CACHE_BOXES", 3)
    capped = coarse.upper_coarse(g, F=int(1.5 * fwd), G=int(4 * fwd), program=bp.program)
    assert capped.total == ref.total and capped.n_units == ref.n_units
    coarse.check_coarse_plan(g, capped, int(1.5 * fwd), int(4 * fwd), program=bp.program)
