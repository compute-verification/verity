"""admission_planner (COORD R17-29): the five terms, the named dominant contributor, the calibrated decision, per-record calibration.
Torch-free; fixtures = the admission fixtures (stoch) + the fp8 B1 sizing / B8 workload of record + the planner calibration table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import admission_planner as P
from verity_vllm.pipeline import cli

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "admission"
REPO = HERE.parents[1]
MiB = 1 << 20
WL = REPO / "workloads"
B1 = "qwen3-4b-fp8__fp8__h100__tp1__b1__i1024__o128__mixed__greedy__bi-eager"
B8 = "qwen3-4b-fp8__fp8__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
CGROUP_H100 = 250999996416          # cp-h100 memory.max (239,372 MiB)


def _sizing_fp8():
    """The B1 FP8 Commit's arena_sizing (per-module widths + model geometry) -- the sizing record both FP8 admissions of record used."""
    p = FIX / "fp8_b1_sizing.json"
    if not p.exists():
        pytest.skip("fp8 B1 sizing fixture missing")
    return json.loads(p.read_text())


def _wl(row):
    p = WL / f"{row}.json"
    if not p.exists():
        pytest.skip(f"workload {row} not in this checkout")
    wl = json.loads(p.read_text())
    return wl, (wl.get("engine_args_required") or {}).get("max_num_batched_tokens")


def _cal():
    return [json.loads(l) for l in P.CALIBRATION.read_text().splitlines() if l.strip()]


def test_b8_fp8_bounded_plan_names_terms_and_dominant_contributor():
    s, (wl, mnbt) = _sizing_fp8(), _wl(B8)
    p = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": True, "retain_exclude": ["fa3_hidden_m1"], "open_after_release": True, "pairs": 3},
               engine_rss_bytes=2193 * MiB, host_cap_bytes=CGROUP_H100, max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"])
    t = p["terms"]
    assert set(t) == set(P.TERMS) and all(int(t[k]["bytes"]) >= 0 for k in t)
    assert p["host_bound_bytes"] == sum(int(t[k]["bytes"]) for k in P.HOST_TERMS)                      # the host bound is the sum of the host terms
    assert t["gpu_staging"]["bytes"] == 8 * 256 * MiB + 256 * MiB and t["gpu_staging"]["where"] == "device"   # the window ring, not a step
    assert t["hash_backlog"]["bytes"] == 4 << 30                                                    # learn budget: bounded hash work is finite
    parts = t["host_staging"]["parts"]
    assert parts["pairs_extra"] == parts["levels"] + parts["kept"] + t["hash_backlog"]["bytes"]        # the 3-pair term = one more frozen pool
    assert t["opening_data"]["bytes"] == parts["levels"] + t["hash_backlog"]["bytes"] + (3 * 64 + 64) * 256
    d = p["dominant"]
    lm = d["largest_materialization"]
    assert lm["step"] == 0 and lm["tokens"] == 3537 and "3,537 prompt tokens" in lm["kind"] and "prefill step 0" in lm["kind"]
    assert lm["hidden_bytes_padded"] > 2 * lm["hidden_bytes_per_sequence"]                             # FA3 multi-prompt: the two geometries differ 2x
    assert lm["geometry_used"] == "padded" and lm["hidden_bytes"] == lm["hidden_bytes_padded"]           # FA3 uncalibrated -> the larger is used
    assert "excluded from host" in lm["where"] and "opened by replay" in lm["where"]
    assert "hidden M1 stream (fa3, 36 layers)" in d["text"] and "prefill step 0" in d["text"] and "predicted OOM" not in d["text"]
    assert d["host_term"] == "host_staging" and 0.5 < d["host_term_share"] < 0.8
    # the R17 B8 admission of record (M-0328: 193,483 MiB, 3 pairs) used the same terms on a per-step model with K = the prompt; the planner's
    # K reaches prompt + cap + lag (stoch) -> a larger, still-fitting bound; the decision is WARN (no FA3 B8 record yet), never a silent ADMIT
    assert 190_000 * MiB < p["host_bound_bytes"] < CGROUP_H100 * 0.9 and p["decision"] == "WARN" and p["admitted"]
    assert "0 calibration record(s) for bounded:host:exclude:pN" in p["why"] and "DOMINANT" in p["why"]


def test_b1_bounded_arms_admit_on_the_calibrated_bound_and_calibrate_against_their_records():
    s, (wl, mnbt) = _sizing_fp8(), _wl(B1)
    cal = _cal()
    px = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": True, "retain_exclude": ["fa3_hidden_m1"], "open_after_release": True},
                engine_rss_bytes=2189 * MiB, host_cap_bytes=CGROUP_H100, max_num_batched_tokens=mnbt, calibration=cal)
    pn = P.plan(s, wl["requests"], mode={"retain": "none", "bounded": True, "open_after_release": True},
                engine_rss_bytes=2189 * MiB, host_cap_bytes=CGROUP_H100, max_num_batched_tokens=mnbt, calibration=cal)
    assert px["mode_key"] == "bounded:host:exclude:p1" and pn["mode_key"] == "bounded:none:p1"
    # [revb n-fp8-plan-1] both modes' records are UNDER their derivations (ratios 0.65 / 0.72): that confirms the derivation as an upper
    # bound and lowers nothing -- factor 1.0, the calibrated bound IS the derivation, the worst ratio is on the record
    assert px["calibration"]["calibrated"] and px["calibration"]["entries"] == 2 and px["calibration"]["factor"] == 1.0
    assert px["calibration"]["worst_ratio"] == max(e["ratio"] for e in cal if e["mode"] == "bounded:host:exclude:p1") < 1.0
    assert px["decision"] == "ADMIT" and pn["decision"] == "ADMIT"
    assert px["calibrated_bound_bytes"] == px["host_bound_bytes"] and pn["calibrated_bound_bytes"] == pn["host_bound_bytes"]
    # the fixture's predicted bytes ARE this planner's numbers for these rows (a planner change re-derives the table by name)
    for e in cal:
        if e["mode"] == "bounded:host:exclude:p1":
            assert e["predicted_bytes"] == px["host_bound_bytes"], (e["row"], e["predicted_bytes"], px["host_bound_bytes"])
        if e["mode"] == "bounded:none:p1":
            assert e["predicted_bytes"] == pn["host_bound_bytes"]
    # both measured peaks (process VmHWM, try 5) are under the planner's bound: the bound is conservative and the record says by how much
    assert all(e["measured_bytes"] < e["predicted_bytes"] for e in cal if e["mode"].startswith("bounded"))
    # per-step soundness through the record's steps.jsonl when present beside the test (the try-5 record's per-step n_bytes)
    rec = FIX / "fp8_b1_try5_bounded_none"
    if rec.exists():
        c = P.calibrate(pn, rec, row="b1 bounded-none try5")
        assert c["measured_by"] == "process VmHWM" and abs(c["ratio"] - 0.7279) < 0.001
        assert c["soundness_per_step"]["sound"] and c["soundness_per_step"]["steps_under_predicted"] == 0
        assert c["entry"]["mode"] == "bounded:none:p1" and c["entry"]["measured_bytes"] == 24671784960


def test_r17_35_record_says_kind_scope_overlaps_and_the_gap_accounting_names_where_the_bound_is_loose(tmp_path):
    """R17-35 (Daniel s7): per run the record carries predicted/observed GPU and host peaks, the terms' contributions in the same scope,
    the lifetimes the bound assumes to overlap, the dominant contributor, and bound-vs-forecast -- and the gap is traced to a term, not
    absorbed into a multiplier."""
    s, (wl, mnbt) = _sizing_fp8(), _wl(B1)
    p = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": True, "retain_exclude": ["fa3_hidden_m1"], "open_after_release": True},
               engine_rss_bytes=2189 * MiB, host_cap_bytes=CGROUP_H100, max_num_batched_tokens=mnbt, calibration=_cal())
    # every number says what kind it is and in what scope
    assert p["kind"]["host_bound_bytes"].startswith("conservative_bound") and p["kind"]["calibrated_bound_bytes"].startswith("empirical_forecast")
    assert all(p["terms"][t]["scope"] == P.TERM_SCOPE[t] and p["terms"][t]["kind"].startswith("conservative_bound") for t in P.TERMS)
    assert "transient" in p["terms"]["gpu_staging"]["scope"] and "whole-run retained" in p["terms"]["host_staging"]["scope"]
    ov = p["lifetime_overlaps_assumed"]
    assert any("replay pool" in " ".join(o["lifetimes"]) for o in ov) and any("hidden M1" in " ".join(o["lifetimes"]) and "NOT co-resident" in o["assumed"] for o in ov)
    assert not any("pair N" in " ".join(o["lifetimes"]) for o in ov)                        # 1 pair: no pair overlap is assumed
    assert p["gpu_predicted"]["device_used_peak_bytes"] == (s["device_total_bytes"] - s["device_free_bytes"]) + p["terms"]["gpu_staging"]["bytes"]
    # a record: the try-5 verdict (VmHWM) + a footprint whose pool froze below the pool terms, the process baseline at the peak, and a device curve
    rec = FIX / "fp8_b1_try5_bounded_none"
    if not rec.exists():
        pytest.skip("try-5 record fixture missing")
    d = tmp_path / "commit"
    d.mkdir()
    (d / "verdict.json").write_text((rec / "verdict.json").read_text())
    vm = json.loads((rec / "verdict.json").read_text())["host_peak"]["process_vmhwm_bytes"]
    pool_pred = p["terms"]["host_staging"]["bytes"] + p["terms"]["hash_backlog"]["bytes"]
    # the try-5 shape (FP8-STAGING s5.4): run segment -- learn windows peak at 10 GiB then the pool freezes at 8 GiB (levels + kept) with the
    # baseline grown 2.3 -> 5.3 GiB; release_staging; replay segment -- a FRESH pool of 6 GiB beside the frozen one, the RSS peak = VmHWM
    frozen, run_pool_max, replay_pool, base0, base_end = 8 << 30, 10 << 30, 6 << 30, 2300 * MiB, 5300 * MiB
    rem = vm - (base_end + frozen + replay_pool)                      # what no term predicts at the peak (the warm-up pool in torch's pinned cache)
    tot, free0 = s["device_total_bytes"], s["device_free_bytes"]
    cu0 = {"reserved": tot - free0 - (2 << 30), "device_total": tot, "device_free": free0}
    cu1 = {"reserved": cu0["reserved"] + 1200 * MiB, "device_total": tot, "device_free": free0 - 1300 * MiB}
    ev = [{"t": 0.1, "event": "end_step", "step": 0, "staged_bytes": 900 * MiB, "rss_bytes": base0 + (4 << 30), "pinned_pool_bytes": 4 << 30, "pinned_pool_in_use": 0, "cuda": cu0},
          {"t": 5.0, "event": "end_step", "step": 1, "staged_bytes": 3 * MiB, "rss_bytes": base_end + run_pool_max, "pinned_pool_bytes": run_pool_max, "pinned_pool_in_use": run_pool_max, "cuda": cu1},
          {"t": 9.0, "event": "end_step", "step": 127, "staged_bytes": 3 * MiB, "rss_bytes": base_end + frozen, "pinned_pool_bytes": frozen, "pinned_pool_in_use": frozen, "cuda": cu1},
          {"t": 9.5, "event": "release_staging", "raw_device_bytes": 0, "cuda": cu1},
          {"t": 12.0, "event": "end_step", "step": 0, "staged_bytes": 900 * MiB, "rss_bytes": vm - MiB, "pinned_pool_bytes": replay_pool - MiB, "pinned_pool_in_use": replay_pool - MiB, "cuda": cu1},
          {"t": 15.0, "event": "end_step", "step": 127, "staged_bytes": 3 * MiB, "rss_bytes": vm, "pinned_pool_bytes": replay_pool, "pinned_pool_in_use": replay_pool, "cuda": cu1}]
    (d / "footprint.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    c = P.calibrate(p, d, row="synthetic b1")
    assert c["measured_bytes"] == vm and c["measured_by"] == "process VmHWM"
    ap = c["measured"]["at_peak"]
    assert ap["segments"] == 2 and ap["segment_kind"] == "replay" and ap["frozen_pool_bytes"] == frozen and ap["run_pool_max_bytes"] == run_pool_max
    assert ap["replay_pool_bytes"] == replay_pool and ap["baseline_end_of_run_bytes"] == base_end and ap["baseline_first_step_bytes"] == base0 and ap["remainder_bytes"] == rem
    g = {r["term"].split(" (")[0]: r for r in c["gap_accounting"]}
    assert g["TOTAL"]["predicted_bytes"] == p["host_bound_bytes"] and g["TOTAL"]["observed_bytes"] == vm
    fzr = g["host_staging + hash_backlog"]
    assert fzr["observed_bytes"] == frozen and fzr["predicted_bytes"] == pool_pred and fzr["gap_bytes"] == pool_pred - frozen and fzr["reason"].startswith("capacity_not_filled")
    assert g["headroom"]["observed_bytes"] == base_end and g["headroom"]["predicted_bytes"] == 2189 * MiB + 6 * (1 << 30)
    assert g["opening_data"]["observed_bytes"] == replay_pool and "OVERLAP with the run's frozen pool IS realized" in g["opening_data"]["reason"]
    assert g["released pools at the peak"]["observed_bytes"] == rem and g["released pools at the peak"]["predicted_bytes"] == 0 and "unmodelled" in g["released pools at the peak"]["reason"]
    assert ap["pair_of_peak"] == 0 and ap["pairs_seen"] == 1
    # the observed rows sum to the peak and the predicted rows to the bound (nothing is lost in the accounting); the miss is NAMED, not scaled away
    rows = [r for r in c["gap_accounting"] if r["term"] != "TOTAL"]
    assert sum(r["observed_bytes"] for r in rows) == vm and sum(r["predicted_bytes"] for r in rows) == p["host_bound_bytes"]
    assert c["bound_vs_forecast"]["conservative_bound_bytes"] == p["host_bound_bytes"] and c["bound_vs_forecast"]["forecast_is_calibrated"] and c["bound_vs_forecast"]["observed_bytes"] == vm
    # GPU: device scope and collector scope, each against its own prediction
    assert c["gpu"]["observed_device_used_peak_bytes"] == tot - (free0 - 1300 * MiB) and c["gpu"]["observed_torch_reserved_delta_bytes"] == 1200 * MiB
    assert c["gpu"]["observed_device_delta_bytes"] == 1300 * MiB and c["gpu"]["predicted_collector_staging_bytes"] == p["terms"]["gpu_staging"]["bytes"]
    assert c["gpu"]["observed_device_delta_since_admission_bytes"] == 1300 * MiB and 0 < c["gpu"]["ratio_device_delta_over_staging"] < 1 and 0 < c["gpu"]["ratio_device"] <= 1.05
    # the CLI prints it (the pod's one-command answer to 'where is the bound loose?')
    wlp = WL / f"{B1}.json"
    rc = cli.main(["admission-planner", "--workload", str(wlp), "--sizing", str(FIX / "fp8_b1_sizing.json"), "--bounded", "--exclude", "fa3_hidden_m1", "--open-after-release",
                 "--engine-rss-mb", "2189", "--cap-bytes", str(CGROUP_H100), "--record", str(d), "--json"])
    assert rc == 0


def test_unbounded_fa2_rows_use_the_per_sequence_geometry_and_the_calibration_carries_the_b16_miss():
    d = json.loads((FIX / "b8greedy_exec.json").read_text())
    wl, mnbt = _wl(d["row_id"])
    s = d["sizing"]
    cal = _cal()
    p = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": False, "pairs": 3}, engine_rss_bytes=2200 * MiB, host_cap_bytes=119209 * MiB,
               max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"], calibration=cal)
    lm = p["dominant"]["largest_materialization"]
    assert lm["geometry_used"] == "per_sequence" and lm["hidden_bytes"] == lm["hidden_bytes_per_sequence"]      # FA2: the calibrated geometry
    assert p["terms"]["hash_backlog"]["bytes"] == p["schedule"]["peak_step_bytes"] and "F-fp8-B8MEM" in p["terms"]["hash_backlog"]["how"]
    assert p["terms"]["host_staging"]["bytes"] == int(p["schedule"]["run_total_used"] * 1.25)
    # the F-dA-15 envelope refused this row 1.83x over the cgroup (218,163 MiB > 119,209) and it ran at ~65 GB; the planner's own bound
    # (84,644 MiB) fits with 29 % to spare.  Both unbounded:host:pN records are board-quoted APPROXIMATE RSS figures (no VmHWM): the B16
    # cp-big 1.298 miss (an unbounded 3-pair pool grew past its raw bytes) is carried on the record as a CAUTION and never as a factor
    # (revb n-fp8-plan-1 (a)) -> factor 1.0, the derivation is the bound, and the decision is WARN (not a plain ADMIT) naming that record.
    assert p["calibration"]["calibrated"] and p["calibration"]["factor"] == 1.0 and p["calibration"]["exact_entries"] == 0
    assert [r["ratio"] for r in p["calibration"]["approx_over_records"]] == [1.2983]
    assert p["host_bound_bytes"] < 119209 * MiB * 0.9 and p["calibrated_bound_bytes"] == p["host_bound_bytes"]
    assert p["decision"] == "WARN" and p["admitted"] and "b16 1024/128 mixed-arrivals" in p["why"] and "x1.298" in p["why"] and "APPROXIMATE" in p["why"]
    # an EXACT over-record raises the factor and the bound with it: the same row with a VmHWM-measured 1.2983 miss is REFUSED at the 10 %
    # margin (109,868 MiB = 92 % of the cgroup) and the refusal names the record; a 5 % margin admits it.  Not the envelope's 3x.
    exact = [({**e, "approx": False} if e["mode"] == "unbounded:host:pN" else e) for e in cal]
    pe = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": False, "pairs": 3}, engine_rss_bytes=2200 * MiB, host_cap_bytes=119209 * MiB,
                max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"], calibration=exact)
    assert abs(pe["calibration"]["factor"] - 1.2983) < 1e-6 and pe["calibrated_bound_bytes"] == int(pe["host_bound_bytes"] * 1.2983)
    assert pe["decision"] == "REFUSE" and "b16 1024/128 mixed-arrivals" in pe["why"] and "factor 1.298" in pe["why"]
    assert pe["calibrated_bound_bytes"] < 119209 * MiB                                     # over the margin, not over the cgroup
    p5 = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": False, "pairs": 3}, engine_rss_bytes=2200 * MiB, host_cap_bytes=119209 * MiB,
                max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"], calibration=exact, margin=0.05)
    assert p5["decision"] == "ADMIT"
    # the fixture's predicted bytes are this planner's numbers
    e = next(e for e in cal if e["row"].startswith("llama32-1b l40s b8 1024/128 greedy"))
    assert e["predicted_bytes"] == p["host_bound_bytes"]


def test_refusal_names_the_dominant_contributor_and_the_co_tenant_term():
    s, (wl, mnbt) = _sizing_fp8(), _wl(B8)
    # the R16 B8 launch as made: unbounded, retain host, 3 pairs, cgroup 239,372 MiB -> refused BY NAME before the run
    p = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": False}, engine_rss_bytes=2193 * MiB, host_cap_bytes=CGROUP_H100,
               max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"])
    assert p["decision"] == "REFUSE" and not p["admitted"]
    assert "DOMINANT" in p["why"] and "hidden M1 stream (fa3, 36 layers)" in p["why"] and "prefill step 0" in p["why"] and "3,537 prompt tokens" in p["why"]
    assert "D2H pinned copy" in p["dominant"]["largest_materialization"]["where"]
    # a bounded plan that fits alone is refused when co-tenants hold the difference: the effective cap is memory.max - memory.current(others)
    q = P.plan(s, wl["requests"], mode={"retain": "host", "bounded": True, "retain_exclude": ["fa3_hidden_m1"], "open_after_release": True, "pairs": 3},
               engine_rss_bytes=2193 * MiB, host_cap_bytes=CGROUP_H100, others_current_bytes=40 << 30, max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"])
    assert q["decision"] == "REFUSE" and q["effective_cap_bytes"] == CGROUP_H100 - (40 << 30) and "co-tenants 40960" in q["why"]
    # no cap readable -> WARN with the bound on the record, never a refusal and never a silent admit
    r = P.plan(s, wl["requests"], mode={"retain": "none", "bounded": True, "open_after_release": True}, engine_rss_bytes=2193 * MiB, host_cap_bytes=None,
               max_num_batched_tokens=mnbt or s["envelope"]["max_num_batched_tokens"])
    assert r["decision"] == "WARN" and r["threshold_bytes"] is None


def test_calibration_factor_rules():
    cal = _cal()
    # [revb n-fp8-plan-1] calibration only RAISES: a mode whose records all sit under their derivations keeps factor 1.0 (the worst ratio
    # stays on the record); an approximate record never sets the factor (both unbounded:host:pN entries are approx -> 1.0, with the 1.2983
    # over-record listed as a caution); an EXACT over-record does set it, and the worst one governs, not the mean
    fx = P.calibration_factor(cal, "bounded:host:exclude:p1")
    assert fx["factor"] == 1.0 and fx["calibrated"] and fx["worst_ratio"] == 0.6466 and fx["exact_entries"] == 2 and fx["approx_over_records"] == []
    fu = P.calibration_factor(cal, "unbounded:host:pN")
    assert fu["factor"] == 1.0 and fu["exact_entries"] == 0 and [r["ratio"] for r in fu["approx_over_records"]] == [1.2983]
    exact = [{**e, "approx": False} for e in cal if e["mode"] == "unbounded:host:pN"]
    assert P.calibration_factor(exact, "unbounded:host:pN")["factor"] == 1.2983 and P.calibration_factor(exact, "unbounded:host:pN")["approx_over_records"] == []
    mixed = exact + [{**exact[0], "row": "approx-worse", "ratio": 1.9, "approx": True}]
    assert P.calibration_factor(mixed, "unbounded:host:pN")["factor"] == 1.2983          # the approx 1.9 is a caution, never the factor
    one = [e for e in cal if e["mode"] == "bounded:none:p1"][:1]
    f = P.calibration_factor(one, "bounded:none:p1")
    assert f["factor"] == 1.0 and not f["calibrated"] and "1 calibration record(s)" in f["why"]      # < 2 records: the derivation stands
    assert P.calibration_factor(None, "bounded:sample:p1")["factor"] == 1.0
