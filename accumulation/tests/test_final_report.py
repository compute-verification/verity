"""DELIVERABLES tables (`sweeps/final_report.py`): §0 objects, byte formatting with the inference ratio, status column, the
lower-module hash / `stale L` flag, the K-marginal steady state, and the end-to-end build on a synthetic coarse.jsonl."""

from __future__ import annotations

import json

from accumulation.sweeps import final_report as fr
from accumulation.sweeps.coarse_report import _index
from accumulation.sweeps.coarse_run import finalize_row, module_versions

Q, FH, GH = fr.Q, fr.FH, fr.GH


def _row(model, circuit, *, Q_inf=Q, F_hat=FH, G_hat=GH, Q_step=None, K=None, seq=None, tokens, U=None, L=None, credited=None, all_macs=None,
         lower="cur", status=None, P=16.06e9):
    cur = module_versions()
    r = {"model": model, "circuit": circuit, "Q_inf": Q_inf, "F_hat": F_hat, "G_hat": G_hat, "Q_step": Q_step, "K": K, "seq": seq if seq is not None else Q_inf,
         "tokens": tokens, "U": U, "L": L, "credited_macs": credited, "all_macs": all_macs, "M": credited, "token_bytes": 4 * tokens, "P": P, "status": status,
         "error": None, "lower_error": None, "upper_error": None, "U_n_ru": 1 if circuit in fr.INFERENCE else 8,
         "versions": {**cur, "lower": cur["lower"] if lower == "cur" else lower},
         "stage_versions": {"lower": cur["lower"] if lower == "cur" else lower, "coarse": cur["coarse"], "upper": cur["upper"]},
         "U_detail": None if circuit in fr.INFERENCE else {"family": "convex-block", "acyclic": True, "kf": 1, "fb": "convex"}}
    return finalize_row(r)


def test_bytes_and_ratio_formatting():
    assert fr.bpt(4.0) == "4.00 B/tok (1.00x inf)"
    assert fr.bpt(560_000.0, 2) == "560 KB/tok (140,000x inf)"
    assert fr._bytes(1.6e5) == "160 KB"
    assert fr._bytes(2.5e9) == "2.50 GB"
    assert fr.bpt(None) == "[ ]"


def test_penalties_follow_section_0():
    # c_L = (L/Q)/(M/Q), c_inf = 4 B / (M_inf/Q); certified = c_L / c_inf
    pen = fr.penalties(L_b=8.0, U_b=5e5, M_tok=3e10, M_inf_tok=1e10)
    assert abs(pen["c_inf"] - 4e-10) < 1e-20
    assert abs(pen["cert"] - (8.0 / 3e10) / 4e-10) < 1e-9          # 0.667x
    assert abs(pen["ach"] - (5e5 / 3e10) / 4e-10) < 1e-6           # 41,667x
    ro = fr.penalties(L_b=None, U_b=120_000.0, M_tok=1e10, M_inf_tok=1e10)   # rollout: same MACs -> (U/Q) / 4 B
    assert ro["cert"] is None and abs(ro["ach"] - 30_000) < 1e-9


def test_stale_l_flag_and_status():
    cur = module_versions()
    fresh = _row("llama3-8b", "local-sgd", Q_step=Q, K=2, tokens=2 * Q, U=1e9, L=8 * 2 * Q, credited=1e15)
    stale = _row("llama3-8b", "local-sgd", Q_step=Q, K=4, tokens=4 * Q, U=2e9, L=8 * 4 * Q, credited=2e15, lower="deadbeef00")
    assert fr.l_tag(fresh, cur) == f" `[L:{cur['lower']}]`"
    assert "stale(deadbeef00)" in fr.l_tag(stale, cur)
    assert fr.status_of(fresh) == "measured"
    assert fr.status_of(_row("llama3-8b", "local-sgd", Q_step=Q, K=2, tokens=2 * Q, U=1e9)) == "measured U"
    assert fr.status_of({"status": "pod", "U": None, "L": None}) == "pod"
    assert fr.status_of(None) == "pod" and fr.status_of(None, in_matrix=False) == "[ ]"


def test_k_marginal_steady_state_uses_two_largest_k():
    rows = [_row("llama3-8b", "local-sgd", Q_step=Q, K=K, tokens=K * Q, U=1e9 + 5e5 * K * Q, L=8 * K * Q + 4, credited=3e10 * K * Q) for K in (2, 4, 8)]
    st = fr.steady_training(_index(rows), "llama3-8b", Q_step=Q)
    assert st["K_pair"] == (4, 8)
    assert abs(st["U_b"] - 5e5) < 1e-6 and abs(st["L_b"] - 8.0) < 1e-9 and abs(st["M_tok"] - 3e10) < 1e-3
    assert st["status"] == "measured"


def test_superseded_cyclic_u_and_prefix_l_are_never_shown():
    from accumulation.sweeps.coarse_report import sanitize
    cyc = _row("llama3-8b", "local-sgd", Q_step=Q, K=2, tokens=2 * Q, U=1e9, L=8 * 2 * Q, credited=1e15)
    cyc["U_detail"] = {"family": "block", "kf": 2, "fb": "joint"}
    cyc["stage_versions"]["lower"] = "08d17eaed3"                    # pre-fix lower module (old lower.py-only hash)
    fwd = _row("llama3-8b", "forward-nonfixed", Q_step=Q, K=1, seq=Q, tokens=Q, U=1e9, credited=1e13)
    fwd["U_detail"] = {"family": "block"}                            # forward-only: unaffected by acyclicity
    out = sanitize([cyc, fwd])
    assert out[0]["U"] is None and out[0]["U_superseded"] == 1e9 and out[0]["L"] is None and out[0]["L_withheld"] == 8 * 2 * Q
    assert out[1]["U"] == 1e9
    assert "superseded" in fr.status_of(out[0]) and "withheld" in fr.status_of(out[0])


def test_build_emits_deliverables_layout(tmp_path):
    rows = [_row("llama3-8b", "inference-session", tokens=Q, U=4 * Q, all_macs=9.4e9 * Q),
            _row("llama3-8b", "inference-dense", tokens=Q, U=4 * Q, all_macs=9.4e9 * Q)]
    rows += [_row("llama3-8b", "local-sgd", Q_step=Q, K=K, tokens=K * Q, U=1e9 + 5e5 * K * Q, L=8 * K * Q + 4, credited=3e10 * K * Q) for K in (2, 4, 8, 12)]
    rows += [_row("llama3-8b", "forward-nonfixed", Q_step=q, K=1, seq=min(q, Q), tokens=q, U=16e9 + 1.2e5 * q, L=4 * q, credited=9.4e9 * q) for q in (8192, 131072)]
    (tmp_path / "coarse.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "plots").mkdir()
    out = fr.build(tmp_path, log=lambda *a: None, figures=False)
    t1 = (tmp_path / "TABLE1.md").read_text()
    t2 = (tmp_path / "TABLE2.md").read_text()
    rep = (tmp_path / "REPORT.md").read_text()
    assert "| Model | Workload | Useful MAC/token | L/Q | U/Q | U/L | Certified penalty | Best-attack penalty | Status |" in t1
    assert "analytic" not in "\n".join(l for l in t1.splitlines() if l.startswith("| ")), "no analytic values in Table 1"
    roll = next(x for x in out["table1"] if x["model"] == "llama3-8b" and x["workload"] == "rollout")
    assert roll["Q_roll"] == 131072 and abs(roll["ach"] - (roll["U_b"] / 4)) < 1e-9 and abs(roll["cert"] - 1.0) < 1e-9
    pre = next(x for x in out["table1"] if x["model"] == "llama3-8b" and x["workload"] == "pretraining")
    assert pre["K_pair"] == (8, 12) and abs(pre["U_b"] - 5e5) < 1e-6 and "measured" == pre["status"]
    assert "## Table 2b. Policy table" in t2 and "## K sweep detail" in t2
    assert f"[L:{module_versions()['lower']}]" in t1
    assert "## 6. Validation" in rep and "Worst-RU diagnostic" in rep


def test_rollout_tiers_and_fit():
    from accumulation.sweeps import rollout_tables as rt
    from accumulation.sweeps.analytic import rollout_tiers
    t = rollout_tiers("llama3-8b", Q_roll=1048576)
    assert t["b1"] < t["b2"] <= t["b3"]                       # closing routes only raises the estimate
    assert 3 <= t["m_star"] <= 6 and 2e5 <= t["t_star"] <= 4e5    # G-bound tiling: ~4 layers x 250k tokens (per-sequence Up, F does not bind)
    assert abs(t["A_token_layers"] - 4 * 8192 * 32) < 1e-6 * t["A_token_layers"] and not t["F_binds"]
    assert 0.10e6 <= t["b2"] <= 0.14e6                                # ~0.125 MB/tok, the redo's measured 8B U at 1M
    one = rollout_tiers("llama3-8b", Q_roll=1048576, G_hat=1000)        # whole rollout in one RU: I* = P/Q + 4 B exactly
    assert abs(one["b2"] - (one["P"] / 1048576 + 4.0)) < 1e-6 * one["b2"]
    assert rollout_tiers("llama3-8b", Q_roll=8192, F_hat=0.5)["F_binds"]   # F binds only when one sequence x m layers exceeds it
    f = rt.fit_r([(8192, 1e12), (131072, 1e9 + 5e4 * 131072), (1048576, 1e9 + 5e4 * 1048576)])
    assert f["n"] == 2 and abs(f["r"] - 5e4) < 1e-3 and 8192 not in f["Qs"]   # 8K is transient, excluded
    cur = module_versions()
    stale = {"L": 1e6, "L_per_token": 100.0, "L_stale": True, "stage_versions": {"lower": "deadbeef00"}}
    cell, v = rt.l_cell(stale, cur)
    assert "stale(`deadbeef00`)" in cell and "gray" in cell and v == 100.0
    fresh = {"L": 1e6, "L_per_token": 100.0, "L_stale": False, "L_gap": 0.02, "stage_versions": {"lower": cur["lower"]}}
    cell, _ = rt.l_cell(fresh, cur)
    assert cell.startswith("**") and "gap 2.0%" in cell
    unc = {**fresh, "L_gap": 0.0, "L_certified": False, "L_detail": {"recheck": "time limit (bound not closed)"}}
    assert "recheck: time limit" in rt.l_cell(unc, cur)[0] and "gap" not in rt.l_cell(unc, cur)[0]
    rec = {"L": 100.0, "L_recurring": 80.0, "L_root_read": 15.0, "token_bytes": 5.0, "P": 1.0}
    assert rt.recurring_share(rec) == "80%"
    assert rt.l_cell(None, cur)[0] == "pod-pending"


def test_provisional_moe_rules():
    from accumulation.sweeps import rollout_tables as rt
    band = {"U": 1e9, "U_per_token": 1e3, "Q_step": 1048576, "Q_inf": 8192, "tokens": 1048576, "U_detail": {"family": "bands"}, "P": 9e10, "token_bytes": 4e6, "L": 2e10}
    assert rt.u_provisional(band) == "planner fallback: bands"
    assert rt.att_cell(band, 1e3) == "[ ] provisional" and "provisional" in rt.u_cell(band)[0]
    single_ok = {**band, "Q_step": 8192, "U_detail": {"family": "single"}}
    assert rt.u_provisional(single_ok) is None                      # honest whole-model RU while Q_roll <= Q_inf
    assert rt.u_provisional({**band, "U_detail": {"family": "single"}}) == "planner fallback: single"
    whole = {**band, "U_detail": {"family": "single", "anatomy": {"n_ru": 1, "dominant": "whole-model"}}}
    assert rt.u_provisional(whole) is None and rt.new_planner(whole)  # checked whole-rollout RU (Work <= G) is admissible
    assert rt.u_provisional({**band, "U_detail": {"family": "convex-block"}}) is None
    assert rt.l_provisional(band) and abs(rt.l_floor(band) - (9e10 + 4e6)) < 1
    assert rt.cert_cell(band, 2e10 / 1048576).startswith("[ ] provisional; floor ")
    assert not rt.l_provisional({**band, "L": 9.5e10})
