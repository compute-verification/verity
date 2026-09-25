"""Smoke tests for ``accumulation.sweeps`` on the exact-solver-scale configs (tiny2 / tiny-moe).

The bound modules are optional here: when ``accumulation.bounds.lower`` / ``upper`` are missing the sweep must still
run and record ``None``; when present their values are checked for basic sanity (``L <= U``, ``L >= P``)."""

from __future__ import annotations

import json

import pytest

from accumulation.algorithms.registry import ALGORITHMS, build
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.sweeps import characterize as C
from accumulation.sweeps import run as R
from accumulation.sweeps import report as REP
from accumulation.sweeps.util import read_jsonl, replace_between


def test_characterize_tiny2_pretrain():
    row = C.characterize("pretrain-dense", "tiny2", Workload(tokens=8, seq=4))
    assert row["error"] is None
    assert row["Q"] == 8 and row["n_ops"] > 0 and row["P"] > 0
    assert row["adw_matmul_macs"] > 0 and 0 < row["adw_fraction"] <= 1
    assert row["adw_per_token"] == pytest.approx(row["adw_total"] / 8)
    assert row["bytes_by_role"]["accumulated"] + row["bytes_by_role"]["carried"] == row["P"]
    # the IR-faithful analytic model reproduces the extracted MAC count exactly
    assert row["ratio_ir_over_model"] == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("alg", sorted(ALGORITHMS))
def test_analytic_model_matches_ir_on_tiny(alg):
    model = "tiny-moe" if alg in ("inference-moe", "pretrain-moe") else "tiny2"
    wl = {"es": Workload(tokens=8, seq=4, population=2), "local-sgd": Workload(tokens=8, seq=4, local_steps=2),
          "lora": Workload(tokens=8, seq=4, rank=2), "hidden-adapter": Workload(tokens=8, seq=4, rank=2)}.get(alg, Workload(tokens=8, seq=4))
    r = C.audit_case(alg, model, wl)
    assert r["error"] is None, r["error"]
    assert r["ratio_vs_model"] == pytest.approx(1.0, rel=1e-9), (alg, r["ir_macs_per_token"], r["analytic_ir_model"])


def test_breakdown_classes_and_scopes():
    bp = build("pretrain-dense", MODELS["tiny2"], Workload(tokens=8, seq=4))
    g = extract(bp)
    bd = C.breakdown(g, bp)
    assert bd["totals"]["macs"] == sum(r["macs"] for r in bd["matmul"]) == sum(o.macs() for o in g.ops)
    classes = set(bd["by_class"])
    assert {"wt-fwd", "wt-fwd-recompute", "wt-dgrad", "wt-wgrad", "attn-fwd", "attn-fwd-recompute", "attn-bwd"} <= classes
    # fwd, dgrad and wgrad of the weights are the same MAC count; the recompute skips the Wd product
    assert bd["by_class"]["wt-fwd"] == bd["by_class"]["wt-dgrad"] == bd["by_class"]["wt-wgrad"]
    assert bd["by_class"]["wt-fwd-recompute"] < bd["by_class"]["wt-fwd"]
    labels = {r["label"] for r in bd["matmul"]}
    assert "wt-fwd[Wq]" in labels and "wt-wgrad[Wq]" in labels and "wt-dgrad[Wh]" in labels
    scopes = {r["scope"] for r in bd["matmul"]}
    assert any(s.startswith("BlockBwd") for s in scopes) and any("AttnSeqBwd" in s for s in scopes)
    text = C.format_breakdown(bd, 8, "t")
    assert "wt-fwd[Wq]" in text and "By class" in text


def test_lora_adw_excludes_frozen_prefix():
    # hidden-adapter with carried adapters: state floor is the adapter bytes only, but ADW covers the whole network
    row = C.characterize("hidden-adapter", "tiny2", Workload(tokens=8, seq=4, rank=2))
    assert row["error"] is None
    assert row["bytes_by_role"]["carried"] == row["P"] > 0 and row["bytes_by_role"]["accumulated"] == 0
    assert row["adw_fraction"] > 0.5


def test_smoke_sweep_and_report(tmp_path):
    out = tmp_path / "sweep_smoke.jsonl"
    cases = [c for c in R.smoke_cases() if c.model == "tiny2"]
    rows = R.sweep(cases, [1 << 20, 1 << 40], [64, 4096], out, log=lambda *a: None)
    assert len(rows) == len(cases) * 4
    lines = read_jsonl(out)
    assert len(lines) == len(rows)
    for r in rows:
        assert r["error"] is None, r["error"]
        assert r["all_macs"] > 0 and r["P"] >= 0 and r["n_ops"] > 0
        if r["alg"] in ("inference-dense", "inference-moe"):
            assert r["ADW_matmul_macs"] == 0 and r["P"] == 0   # fixed weights: nothing is accumulation-dependent
        else:
            assert r["ADW_matmul_macs"] > 0 and r["P"] > 0
        assert r["F"] in (1 << 20, 1 << 40) and r["X"] in (64, 4096)
    bounds = R.load_bounds()
    if bounds.lower is None:
        assert all(r["L"] is None and r["lower_error"] for r in rows)
    else:
        good = [r for r in rows if r["L"] is not None]
        assert good, [r["lower_error"] for r in rows]
        for r in good:
            assert r["L"] >= r["P"], "L must dominate the state floor (L_source includes accumulated/carried bytes)"
            assert r["kappa_L"] == pytest.approx(r["ADW_matmul_macs"] / r["L"])
    if bounds.upper is not None:
        for r in rows:
            if r["U"] is not None and r["L"] is not None:
                assert r["U"] >= r["L"], "U must be >= L on every graph (SPEC §3)"
    # resume: nothing left to do
    again = R.sweep(cases, [1 << 20, 1 << 40], [64, 4096], out, resume=True, log=lambda *a: None)
    assert again == []
    # report renders the table between markers and is idempotent
    md = tmp_path / "RESULTS.md"
    md.write_text("# R\n\n<!-- BEGIN:smoke -->\nold\n<!-- END:smoke -->\n")
    REP.render(tmp_path, md, log=lambda *a: None)
    text = md.read_text()
    assert "old" not in text and "| alg |" in text and text.count("<!-- BEGIN:smoke -->") == 1
    REP.render(tmp_path, md, log=lambda *a: None)
    assert md.read_text() == text


def test_p3_table_is_schema_tolerant():
    res = {"assumption": "H100 roofline", "kappa_native": 147.6,
           "rows": [{"alg": "a", "model": "m", "tokens": 8, "F": 1, "X": 1, "kappa_L": 10.0, "eta": 0.1, "extra_col": 3}]}
    md = REP.p3_table(res)
    assert "H100 roofline" in md and "| kappa_L |" in md and "extra_col" in md
    assert REP.p3_table({}) .startswith("_no rows")


def test_finalize_row_upgrades_legacy_keys():
    r = {"alg": "a", "model": "m", "tokens": 8, "seq": 4, "F": 1, "X": 1, "L": 100, "U": 200, "ADW_matmul_macs": 1000,
         "ADW_credited_macs": 800, "all_work": 4000, "t_extract": 1.5}
    R.finalize_row(r)
    assert r["kappa_L"] == pytest.approx(8.0) and r["kappa_U"] == pytest.approx(4.0) and r["extract_s"] == 1.5
    assert r["kappa_L_matmul"] == pytest.approx(10.0) and r["inv_eta_honest"] == pytest.approx(5.0)


def test_limits_gate_unsafe_workloads_and_exempt_tiny():
    from accumulation.sweeps.guard import DRIVER_LIMITS, LEGACY_LIMITS, check_limits, sequences_per_forward
    # current (post-fix) envelope: realistic workloads pass
    assert check_limits("pretrain-dense", "llama3-8b", Workload(4096, 4096), DRIVER_LIMITS) is None
    assert check_limits("pretrain-dense", "llama3-405b", Workload(4_194_304, 8192), DRIVER_LIMITS) is None
    assert check_limits("es", "llama3-8b", Workload(1024 * 4096, 4096, population=1024), DRIVER_LIMITS) is None
    assert "tokens" in check_limits("pretrain-dense", "llama3-8b", Workload(1 << 30, 4096), DRIVER_LIMITS)
    # pre-fix envelope kept for reference
    assert "1 sequences" not in (check_limits("pretrain-dense", "llama3-8b", Workload(4096, 4096), LEGACY_LIMITS) or "")
    assert "2 sequences" in check_limits("pretrain-dense", "llama3-8b", Workload(8192, 4096), LEGACY_LIMITS)
    assert "llama3-405b" in check_limits("pretrain-dense", "llama3-405b", Workload(8192, 4096), LEGACY_LIMITS)
    assert sequences_per_forward("es", Workload(8192, 1024, population=8)) == 1
    assert check_limits("pretrain-dense", "tiny2", Workload(8, 4), LEGACY_LIMITS) is None   # toy configs are exempt
    # the sweep driver records out-of-envelope cases as skipped rows instead of extracting them
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rows = R.sweep([R.Case("pretrain-dense", "llama3-8b", Workload(1 << 30, 4096))], [1], [1], Path(d) / "s.jsonl",
                       guarded=False, log=lambda *a: None)
        assert len(rows) == 1 and rows[0]["error"].startswith("skipped: outside extractor safe limits")


def test_guarded_case_matches_in_process():
    """A guarded child produces the same rows as the in-process path (tiny2, real bounds if present)."""
    from pathlib import Path
    import tempfile
    case = R.Case("forward-nonfixed", "tiny2", Workload(8, 4))
    with tempfile.TemporaryDirectory() as d:
        a = R.sweep([case], [1 << 20], [4096], Path(d) / "a.jsonl", guarded=False, log=lambda *a: None)
        b = R.sweep([case], [1 << 20], [4096], Path(d) / "b.jsonl", guarded=True, log=lambda *a: None)
    assert len(a) == len(b) == 1
    for k in ("L", "U", "ADW_matmul_macs", "P", "n_ops", "error"):
        assert a[0].get(k) == b[0].get(k), k
    assert "peak_rss_mb" in b[0]


def test_run_guarded_timeout_and_rss():
    import sys
    from accumulation.sweeps.guard import run_guarded
    r = run_guarded([sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=1.0)
    assert r.status == "timeout"
    r = run_guarded([sys.executable, "-c", "import time; b = bytearray(200 << 20); time.sleep(5)"], timeout_s=10.0, max_rss_mb=64)
    assert r.status == "oom-guard" and r.peak_rss_mb > 64
    r = run_guarded([sys.executable, "-c", "print('hi')"], timeout_s=10.0)
    assert r.status == "ok" and r.stdout.strip() == "hi"


def test_replace_between_appends_when_missing():
    t = replace_between("# x\n", "tag", "body")
    assert "<!-- BEGIN:tag -->\nbody\n<!-- END:tag -->" in t
    assert replace_between(t, "tag", "new").count("body") == 0


# --- driver hardening: resume / redo / kill recovery (tiny2 only) -----------------------------------------

def _row(**kw):
    r = {"alg": "pretrain-dense", "model": "tiny2", "tokens": 8, "seq": 4, "F": 1, "X": 1}
    if "ADW_matmul_macs" in kw:                      # the graph stage is complete only with the credited split
        r.update(ADW_credited_macs=kw["ADW_matmul_macs"], ADW_recompute_macs=0)
    r.update(kw)
    return r


def test_needed_stages_resume_and_redo():
    b = R.Bounds(lambda: None, lambda: None, None, None)
    v = {"lower": "v-new", "upper": "v-up", "adw": "v-adw"}
    kw = dict(retry_errors=False, versions=v, bounds=b)
    # nothing yet -> everything
    assert R.needed_stages(None, redo=set(), resume=False, **kw) == {"adw", "lower", "upper"}
    done = _row(ADW_matmul_macs=1, ADW_credited_macs=1, ADW_recompute_macs=0, L=5, U=9, lower_src="v-old", upper_src="v-up")
    assert R.needed_stages(done, redo=set(), resume=True, **kw) == set()
    assert R.needed_stages(done, redo=set(), resume=False, **kw) == {"adw", "lower", "upper"}
    # redo lower keeps upper; with resume it skips rows already computed by the current lower.py
    assert R.needed_stages(done, redo={"lower"}, resume=False, **kw) == {"lower"}
    assert R.needed_stages(done, redo={"lower"}, resume=True, **kw) == {"lower"}
    assert R.needed_stages(dict(done, lower_src="v-new"), redo={"lower"}, resume=True, **kw) == set()
    assert R.needed_stages(dict(done, lower_src="v-new"), redo={"lower"}, resume=False, **kw) == {"lower"}
    # a recorded deterministic error is complete under resume; a kill is retried only with --retry-errors
    err = _row(ADW_matmul_macs=1, L=5, upper_error="ValueError: no legal tile")
    assert R.needed_stages(err, redo=set(), resume=True, **kw) == set()
    kill = _row(ADW_matmul_macs=1, L=5, upper_error="oom-guard: RSS 3300 MB > 3072 MB")
    assert R.needed_stages(kill, redo=set(), resume=True, **kw) == set()
    assert R.needed_stages(kill, redo=set(), resume=True, retry_errors=True, versions=v, bounds=b) == {"upper"}
    hard = _row(error="timeout: exceeded 900s")
    assert R.needed_stages(hard, redo=set(), resume=True, **kw) == set()
    assert R.needed_stages(hard, redo=set(), resume=True, retry_errors=True, versions=v, bounds=b) == {"adw", "lower", "upper"}
    # a bound module that is not importable is never scheduled
    nb = R.Bounds(None, lambda: None, "ImportError", None)
    assert R.needed_stages(None, redo=set(), resume=False, retry_errors=False, versions=v, bounds=nb) == {"adw", "upper"}


def test_redo_lower_keeps_upper_and_stamps_version(tmp_path):
    bounds = R.load_bounds()
    if bounds.lower is None or bounds.upper is None:
        pytest.skip("needs both bound modules")
    out = tmp_path / "s.jsonl"
    case = R.Case("pretrain-dense", "tiny2", Workload(8, 4))
    first = R.sweep([case], [1 << 20], [64, 4096], out, guarded=False, log=lambda *a: None)
    assert len(first) == 2
    if any(r["lower_error"] or r["upper_error"] for r in first):   # a bound module is being rewritten right now
        pytest.skip(f"bound call failed: {[r['errors'] for r in first]}")
    assert all(r["L"] is not None and r["U"] is not None for r in first)
    ver = R.module_versions()
    assert all(r["lower_src"] == ver["lower"] and r["upper_src"] == ver["upper"] for r in first)
    # poison the stored L and pretend it came from an older lower.py; --redo lower must repair exactly that
    rows = read_jsonl(out)
    for r in rows:
        r["L"], r["lower_src"], r["U"] = 1, "stale", r["U"] + 7   # U is deliberately left wrong: redo lower must not touch it
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    redone = R.sweep([case], [1 << 20], [64, 4096], out, redo={"lower"}, resume=True, guarded=False, log=lambda *a: None)
    by_x = {r["X"]: r for r in redone}
    for r0 in first:
        r1 = by_x[r0["X"]]
        assert r1["L"] == r0["L"] and r1["lower_src"] == ver["lower"]
        assert r1["U"] == r0["U"] + 7 and r1["upper_src"] == ver["upper"]
        assert r1["kappa_U"] == pytest.approx(r1["ADW_credited_macs"] / r1["U"])  # derived columns are recomputed
    assert len(read_jsonl(out)) == 2   # compacted: one line per (case, F, X)
    # the same redo again with --resume is a no-op (versions match)
    assert R.sweep([case], [1 << 20], [64, 4096], out, redo={"lower"}, resume=True, guarded=False, log=lambda *a: None) == []


def test_guarded_child_kill_is_isolated_to_one_row(tmp_path, monkeypatch):
    """Killing the child inside the upper_bound call of one (F, X) loses only that call: L of that row is recovered
    from the checkpoint line, the other (F, X) are computed by a re-spawned child, and --resume --retry-errors
    recomputes just the killed stage."""
    if R.load_bounds().upper is None:
        pytest.skip("needs upper_bound")
    monkeypatch.setenv("ACCSWEEP_TEST_ABORT", f"{1 << 20},64,upper")
    out = tmp_path / "k.jsonl"
    case = R.Case("pretrain-dense", "tiny2", Workload(8, 4))
    rows = R.sweep([case], [1 << 20, 1 << 40], [64, 4096], out, log=lambda *a: None)
    assert len(rows) == 4 and len(read_jsonl(out)) == 4
    by = {(r["F"], r["X"]): r for r in rows}
    victim = by[(1 << 20, 64)]
    assert victim["U"] is None and "killed during upper" in victim["upper_error"] and victim["error"] is None
    if R.load_bounds().lower is not None:
        assert victim["L"] is not None                       # recovered from the checkpoint written after lower_bound
    for k, r in by.items():
        if k != (1 << 20, 64):
            assert r["U"] is not None and r["errors"] is None, r["errors"]
    monkeypatch.delenv("ACCSWEEP_TEST_ABORT")
    again = R.sweep([case], [1 << 20, 1 << 40], [64, 4096], out, resume=True, log=lambda *a: None)
    assert again == []                                        # a recorded kill is complete under plain --resume
    fixed = R.sweep([case], [1 << 20, 1 << 40], [64, 4096], out, resume=True, retry_errors=True, log=lambda *a: None)
    assert len(fixed) == 1 and fixed[0]["X"] == 64 and fixed[0]["U"] is not None and fixed[0]["errors"] is None
    assert fixed[0]["U"] == by[(1 << 40, 64)]["U"]           # F = 2^20 vs 2^40 do not matter at this size


def test_presets_are_cheapest_first_and_supported():
    from accumulation.sweeps.guard import DRIVER_LIMITS, check_limits
    p1 = R.p1_cases()
    assert [c.wl.tokens for c in p1] == sorted(c.wl.tokens for c in p1) and len(p1) == 15
    assert {c.alg for c in p1} == {"pretrain-dense", "lora", "inference-dense"}
    assert all(c.wl.rank == 16 and c.wl.adapted == R.ALL_ADAPTED for c in p1 if c.alg == "lora")
    costs = [R.case_cost(c) for c in p1]
    assert costs == sorted(costs)
    for c in R.p2_cases() + R.p2_8b_point_cases() + p1:
        assert ALGORITHMS[c.alg].supports(MODELS[c.model], c.wl) is None, c
        assert check_limits(c.alg, c.model, c.wl, DRIVER_LIMITS) is None, c
    assert {c.alg for c in R.p2_8b_point_cases()} == set(R.DENSE_ALGS)


# --- analysis: P1 fits and the P3 translation -------------------------------------------------------------

def test_fit_linear_recovers_a_line():
    from accumulation.sweeps.analysis import fit_linear
    f = fit_linear([10, 100, 1000], [1000 + 3 * q for q in (10, 100, 1000)])
    assert f["slope"] == pytest.approx(3.0) and f["intercept"] == pytest.approx(1000.0)
    assert f["r2"] == pytest.approx(1.0) and f["resid_rel_max"] == pytest.approx(0.0, abs=1e-12)
    assert f["secant_slope"] == pytest.approx(3.0) and f["n"] == 3
    assert fit_linear([5, 5], [1, 2]) is None


def _p3_rows():
    tr = lambda Q: R.finalize_row({"alg": "pretrain-dense", "model": "m", "tokens": Q, "seq": 4, "F": 7, "X": 3, "P": 1000,
                                   "ADW_matmul_macs": 80 * Q, "ADW_credited_macs": 50 * Q, "all_macs": 90 * Q, "all_work": 100 * Q,
                                   "L": 1000 + 2 * Q, "U": 1000 + 20 * Q})
    inf = lambda Q: R.finalize_row({"alg": "inference-dense", "model": "m", "tokens": Q, "seq": 4, "F": 7, "X": 3, "P": 0,
                                    "ADW_matmul_macs": 0, "all_macs": 20 * Q, "L": 4 * Q, "U": 100 + 5 * Q})
    return [tr(10), tr(100), tr(1000), inf(10), inf(100), inf(1000)]


def test_p1_fits_and_p3_definitions():
    from accumulation.sweeps.analysis import p1_fits, p3_from_rows
    rows = _p3_rows()
    fits = p1_fits(rows)
    byk = {(f["alg"], f["bound"]): f for f in fits["fits"]}
    assert byk[("pretrain-dense", "L")]["slope"] == pytest.approx(2.0) and byk[("pretrain-dense", "L")]["provisional"]
    assert byk[("pretrain-dense", "U")]["slope"] == pytest.approx(20.0) and not byk[("pretrain-dense", "U")]["provisional"]
    assert byk[("pretrain-dense", "U")]["intercept_over_P"] == pytest.approx(1.0)
    assert byk[("pretrain-dense", "U")]["kappa_inf"] == pytest.approx(50 / 20)          # credited MACs / r_U
    assert byk[("pretrain-dense", "U")]["kappa_inf_matmul"] == pytest.approx(80 / 20)   # recomputes included
    assert byk[("pretrain-dense", "U")]["inv_eta_honest"] == pytest.approx(100 / 50)
    assert byk[("inference-dense", "U")]["kappa_inf_allmacs"] == pytest.approx(20 / 5)
    p3 = p3_from_rows(rows, fits)
    rr = {(r["basis"]): r for r in p3["rows"] if r["alg"] == "pretrain-dense"}
    q = rr["Q=100"]
    kL, kU, kUi = 5000 / 1200, 5000 / 3000, 2000 / 600
    assert q["kappa_L_train"] == pytest.approx(kL) and q["kappa_U_train"] == pytest.approx(kU)
    assert q["kappa_U_inf"] == pytest.approx(kUi) and q["kappa_L_inf"] == pytest.approx(2000 / 400)
    assert q["rho_max"] == pytest.approx(kL / kUi) and q["rho_ach"] == pytest.approx(kU / kUi)
    assert q["rho_max_matmul"] == pytest.approx((8000 / 1200) / kUi) and q["rho_ach_matmul"] == pytest.approx((8000 / 3000) / kUi)
    assert q["inv_eta_honest"] == pytest.approx(2.0)
    assert q["gap_max"] == pytest.approx(kL / kU) == pytest.approx(3000 / 1200) and q["gap_ach"] == 1.0   # = U/L
    assert q["Q_inf"] == 100 and q["inference_alg"] == "inference-dense" and q["L_provisional"] is True
    a = rr["asymptotic"]
    assert a["kappa_L_train"] == pytest.approx(50 / 2) and a["kappa_U_train"] == pytest.approx(50 / 20)
    assert a["kappa_U_inf"] == pytest.approx(20 / 5) and a["rho_max"] == pytest.approx((50 / 2) / (20 / 5))
    assert a["rho_ach"] == pytest.approx((50 / 20) / (20 / 5)) and a["gap_max"] == pytest.approx(20 / 2)
    assert a["rho_ach_matmul"] == pytest.approx((80 / 20) / (20 / 5))
    assert "asymptotic" not in {r["basis"] for r in p3["rows"] if r["alg"] == "inference-dense"}
    # rendering
    assert "rho_max" in REP.p3_table(p3) and "r (B/token)" in REP.p1_fit_table(fits)
    # without an inference baseline at that (model, F, X) the row is kept with a note and no eta
    lone = p3_from_rows([r for r in rows if r["alg"] != "inference-dense"], None)["rows"]
    assert all(r["rho_max"] is None and "no inference-dense row" in r["notes"] for r in lone)


def test_plots_skip_gracefully(tmp_path):
    pytest.importorskip("matplotlib")
    from accumulation.sweeps import plots as PL
    made = PL.make_all(tmp_path, tmp_path / "plots")
    assert made == []  # no JSONL -> nothing to plot, no exception
    # rows without bounds are skipped too
    (tmp_path / "sweep_p2.jsonl").write_text(json.dumps({"alg": "pretrain-dense", "model": "llama3-8b", "tokens": 4096, "seq": 4096,
                                                          "F": 1e12, "X": 5e6, "L": None, "U": None, "params_total": 1}) + "\n")
    assert PL.make_all(tmp_path, tmp_path / "plots") == []


# --- policy points (F, G, X): driver, frontier, tables, Q scaling -------------------------------------------

def test_row_key_and_new_row_carry_G():
    c = R.Case("pretrain-dense", "tiny2", Workload(8, 4))
    assert R.case_key(c, 1, 2, None) != R.case_key(c, 1, 2, 4096)
    assert R.case_key(c, 1, 2) == R.case_key(c, 1, 2, None)          # legacy rows (no G) are the G = None point
    r = R.new_row(c, 1, 2, 4096)
    assert r["G"] == 4096 and R.parse_G("none") is None and R.parse_G("1e8") == 100_000_000
    inf = R.finalize_row({"alg": "inference-dense", "model": "m", "tokens": 8, "seq": 4, "F": 1, "X": 1, "G": None, "L": 10, "U": 20,
                          "ADW_matmul_macs": 0, "ADW_credited_macs": 0, "all_macs": 4000, "U_n_ru": 16})
    assert inf["M"] == 4000 and inf["kappa_low"] == 200 and inf["kappa_high"] == 400 and inf["U_n_ru_per_token"] == 2
    tr = R.finalize_row({"alg": "pretrain-dense", "model": "m", "tokens": 8, "seq": 4, "F": 1, "X": 1, "G": 5, "L": 10, "U": 20,
                         "ADW_matmul_macs": 5000, "ADW_credited_macs": 4000, "all_macs": 6000})
    assert tr["M"] == 4000 and tr["kappa_low"] == 200            # credited MACs, not matmul / all MACs


def test_smoke_G_sweep_in_process_and_frontier(tmp_path):
    from accumulation.sweeps import frontier as FR
    spec = R.PRESETS["smoke-G"]
    rows = R.sweep(spec["cases"](), spec["F"], spec["X"], tmp_path / "s.jsonl", G_values=spec["G"], guarded=False)
    assert len(rows) == 6 and all(r["errors"] is None for r in rows)
    byg = {(r["alg"], r["G"]): r for r in rows}
    for alg in ("inference-dense", "pretrain-dense"):
        u_none, u_4k, u_256 = (byg[(alg, g)]["U"] for g in (None, 4096, 256))
        assert u_none <= u_4k <= u_256                             # a tighter work cap can only cost more input
        assert all(byg[(alg, g)]["U_n_ru"] >= 1 and byg[(alg, g)]["U_ru_work_max"] <= (g or 1 << 62) for g in (4096, 256))
        assert byg[(alg, 256)]["L_at_G"] in ("G=3e+02", "G=None") and byg[(alg, 256)]["U_by_class"]
    res = FR.build_frontier(rows, model="tiny2", X=4096, out_dir=tmp_path, stem="fr", log=lambda *a: None)
    assert len(res["points"]) == 3 and all(p["rho_cert"] and p["rho_ach"] and p["rho_ach"] <= p["rho_cert"] + 1e-12 for p in res["points"])
    assert res["frontier"] and all(v for v in res["operating_points"].values())
    assert (tmp_path / "fr.md").exists() and (tmp_path / "fr.json").exists()
    # tables + qscaling + report run on the same rows (single Q -> qscaling pending, tables filled)
    from accumulation.sweeps import final_report as FRP
    from accumulation.sweeps import qscaling as QS
    from accumulation.sweeps import tables as T
    ops = {k: v for k, v in res["operating_points"].items()}
    (tmp_path / "frontier_8b.json").write_text(json.dumps(res))
    (tmp_path / "scale.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = T.build_tables(tmp_path, ops=ops, X=4096, tokens=8, log=lambda *a: None)
    assert t and all(len(v["rows"]) == 2 for v in t["main"]["points"].values())
    assert all(v["rows"][0]["status"] in ("measured", "L@G=None") for v in t["security"]["points"].values())
    q = QS.build_qscaling(tmp_path, ops=ops, log=lambda *a: None)
    assert all(f["status_L"].startswith("pending") for f in q["fits"])
    (tmp_path / "validation.json").write_text(json.dumps({"cases": [{"name": "t", "I_star": 10, "L": 8, "U": 12}],
                                                          "fuzz": {"violations": 0, "n": 5}}))
    rep = FRP.build_report(tmp_path, X=4096, tokens=8, log=lambda *a: None)
    txt = rep.read_text()
    assert "Operating points" in txt and "L/I*" in txt and "violations" in txt


def test_pareto_and_operating_points():
    from accumulation.sweeps import frontier as FR
    pts = [dict(F=1, G=None, F_label="1", G_label="None", U_inf_per_token=1.0, penalty=0.5, rho_cert=2.0, rho_ach=1.0, n_RU_inf_per_token=1),
           dict(F=1, G=10, F_label="1", G_label="1e+01", U_inf_per_token=2.0, penalty=2.0, rho_cert=0.5, rho_ach=0.3, n_RU_inf_per_token=2),
           dict(F=1, G=7, F_label="1", G_label="7e+00", U_inf_per_token=2.5, penalty=5.0, rho_cert=0.2, rho_ach=0.15, n_RU_inf_per_token=3),
           dict(F=1, G=5, F_label="1", G_label="5e+00", U_inf_per_token=4.0, penalty=8.0, rho_cert=0.125, rho_ach=0.1, n_RU_inf_per_token=4),
           dict(F=2, G=5, F_label="2", G_label="5e+00", U_inf_per_token=4.0, penalty=3.0, rho_cert=1 / 3, rho_ach=0.1, n_RU_inf_per_token=4),
           dict(F=2, G=7, F_label="2", G_label="7e+00", U_inf_per_token=2.5, penalty=5.0, rho_cert=0.2, rho_ach=0.15, n_RU_inf_per_token=3),
           dict(F=1, G=1, F_label="1", G_label="1e+00", U_inf_per_token=9.0, penalty=8.0, rho_cert=0.125, rho_ach=0.1, n_RU_inf_per_token=9,
                degenerate=True)]
    front = FR.pareto(pts)
    assert [p["penalty"] for p in front] == [0.5, 2.0, 5.0, 8.0]    # (4.0, 3.0) dominated; the degenerate point excluded
    assert [p["F"] for p in front][2] == 2                          # tie at (2.5, 5.0): the least restrictive F is kept
    ops = FR.choose_operating_points(front)
    assert ops["permissive"]["penalty"] == 2.0 and ops["aggressive"]["penalty"] == 8.0 and ops["balanced"]["penalty"] == 5.0


def test_qscaling_fits_three_largest_Q():
    from accumulation.sweeps import qscaling as QS
    rows = []
    for Q in (4096, 16384, 65536, 262144):
        rows.append(R.finalize_row({"alg": "pretrain-dense", "model": "m", "tokens": Q, "seq": 4096, "F": 7, "X": 3, "G": 100, "P": 1000,
                                    "ADW_credited_macs": 50 * Q, "ADW_matmul_macs": 60 * Q, "all_macs": 70 * Q, "all_work": 80 * Q,
                                    "L": 1000 + 2 * Q + (5000 if Q == 4096 else 0), "U": 1000 + 20 * Q, "L_at_G": "G=1e+02"}))
    f = QS.fits(rows)[0]
    assert f["tokens_used_L"] == [16384, 65536, 262144]            # the smallest Q (with its bump) is excluded
    assert f["r_L"] == pytest.approx(2.0) and f["P_L"] == pytest.approx(1000) and f["r_U"] == pytest.approx(20.0)
    assert f["status_L"] == "measured" and f["kappa_r_low"] == pytest.approx(50 / 20)
