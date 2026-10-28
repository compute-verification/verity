"""verity_vllm.pipeline.telemetry.admission — synthetic-interval unit tests + recorded attempts as a regression fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline.telemetry import admission as adm
from verity_vllm.pipeline import cli

FIXTURE = Path(__file__).resolve().parents[2] / "verity_vllm" / "observe" / "out" / "gen" / "sweep" / "evidence" / "admit" / "r17_fixture"
MiB = adm.MiB


# ---- sweep-line ----------------------------------------------------------------------------------------------------
def test_peak_overlap_sequential_reuse_is_not_simultaneous():
    # three pairs staging the same 10 GB one after another (the FP8 3-pair result): peak is one pair, not three
    iv = [("pair0", 0, 1, 10), ("pair1", 1, 2, 10), ("pair2", 2, 3, 10)]
    pk = adm.peak_overlap(iv)
    assert pk["peak_bytes"] == 10 and set(pk["live"]) in ({"pair0"}, {"pair1"}, {"pair2"})


def test_peak_overlap_sums_simultaneous_and_names_them():
    iv = [("pool", 0, 10, 100), ("oracle", 3, 6, 20), ("programs", 5, 8, 30), ("workers", 5, 8, 7)]
    pk = adm.peak_overlap(iv)
    assert pk["peak_bytes"] == 157 and pk["t"] == 5
    assert pk["live"] == {"pool": 100, "oracle": 20, "programs": 30, "workers": 7}


def test_peak_overlap_touching_intervals_do_not_overlap_and_reversed_bounds_are_fixed():
    assert adm.peak_overlap([("a", 0, 5, 1), ("b", 5, 9, 1)])["peak_bytes"] == 1
    assert adm.peak_overlap([("a", 5, 0, 4)])["peak_bytes"] == 4
    assert adm.peak_overlap([])["peak_bytes"] == 0


def test_same_name_twice_is_two_allocations():
    pk = adm.peak_overlap([("x", 0, 2, 5), ("x", 1, 3, 5)])
    assert pk["peak_bytes"] == 10 and pk["live"] == {"x": 10}


# ---- the model ---------------------------------------------------------------------------------------------------------
def _cfg(**kw):
    base = {"model": "llama32-1b", "requests": 8, "prompt_tokens": 3819, "decode_rows": 494, "steps": 129, "max_context": 1152, "pairs": 3, "host_limit_bytes": 119_209 * MiB}
    base.update(kw)
    return adm.RowConfig(**base)


def test_predict_names_every_term_and_has_no_multiplier():
    p = adm.predict(_cfg())
    for stage in ("build", "match", "commit"):
        s = p[stage]
        assert s["reserved_bytes"] == sum(s["live_at_peak"].values()) * MiB + sum(s["margin"].values()) * MiB or abs(
            s["reserved_bytes"] - (sum(s["live_at_peak"].values()) + sum(s["margin"].values())) * MiB) < len(s["live_at_peak"]) * MiB + 2 * MiB
        names = {t["name"] for t in s["terms"]}
        assert names <= set(adm.INVENTORY) | {"process.python_base", "margin.pod_cotenants", "margin.kernel_unreclaimable", "commit.c2.largest_vu_eval"}
        for t in s["terms"]:
            assert t["basis"], t


def test_commit_peak_is_pair0_replay_and_pool_persists_under_retain_host():
    p = adm.predict(_cfg())["commit"]
    assert p["peak_point"] == "p0.replay"
    pool = next(t for t in p["terms"] if t["name"] == "commit.pinned_pool")
    assert pool["live"] == ["warmup", "exit"]
    # --retain none on the unbounded windowed path allocates the same pool for the whole run; it is named on the term, never 0 MiB /
    # released at the pair's end by name
    q = adm.predict(_cfg(retain="none"))["commit"]
    pool_n = next(t for t in q["terms"] if t["name"] == "commit.pinned_pool")
    assert pool_n["live"] == ["warmup", "exit"] and pool_n["bytes"] == pool["bytes"] and "regardless of retention" in pool_n["basis"]


def test_heap_residual_only_with_more_than_one_pair_and_programs_once_under_replay_cache():
    one = adm.predict(_cfg(pairs=1))["commit"]
    assert not [t for t in one["terms"] if t["name"] == "commit.c2.heap_residual"]
    three = adm.predict(_cfg(pairs=3))["commit"]
    assert len([t for t in three["terms"] if t["name"] == "commit.c2.programs"]) == 1
    nocache = adm.predict(_cfg(pairs=3, replay_cache=False))["commit"]
    assert len([t for t in nocache["terms"] if t["name"] == "commit.c2.programs"]) == 3


def test_replay_workers_term_f_r17_24():
    """The forked pool without gc.freeze (fork_gc_freeze=False, as the recorded attempts ran)."""
    # 32 forked workers: two may hold the largest VU (2.9 x largest tensor) + 30 x 250 MiB
    p = adm.predict(_cfg(replay_workers=32, fork_gc_freeze=False))["commit"]
    w = next(t for t in p["terms"] if t["name"] == "commit.c2.replay_workers")
    lt = _cfg().largest_tensor()
    assert w["bytes"] == 2 * int(2.9 * lt) + 30 * 250 * MiB
    # serial replay: no worker term, the parent holds the largest VU itself
    s = adm.predict(_cfg(replay_workers=1, fork_gc_freeze=False))["commit"]
    assert not [t for t in s["terms"] if t["name"] == "commit.c2.replay_workers"]
    assert next(t for t in s["terms"] if t["name"] == "commit.c2.largest_vu_eval")["bytes"] == int(2.9 * lt)
    # largest strata in the parent: forked workers are 250 MiB workers
    f = adm.predict(_cfg(replay_workers=32, largest_vu_in_parent=True, fork_gc_freeze=False))["commit"]
    assert next(t for t in f["terms"] if t["name"] == "commit.c2.replay_workers")["bytes"] == 32 * 250 * MiB
    assert f["replay_workers_admissible"] >= p["replay_workers_admissible"]


def test_r19_fork_gc_freeze_is_the_default_and_makes_every_worker_a_250_mib_worker():
    """The shipped code freezes the parent's heap before the fork; the planner's default models it: 32
    workers x 250 MiB, no largest-VU child, no parent largest-VU term (a VU's working set is its own rows), the note names the mechanism."""
    p = adm.predict(_cfg(replay_workers=32))["commit"]
    assert p["fork_gc_freeze"] is True
    w = next(t for t in p["terms"] if t["name"] == "commit.c2.replay_workers")
    assert w["bytes"] == 32 * 250 * MiB and "gc.freeze" in w["basis"]
    assert not [t for t in p["terms"] if t["name"] == "commit.c2.largest_vu_eval"]
    assert "gc.freeze" in p["replay_workers_note"] and p["replay_workers_admissible"] == 32
    old = adm.predict(_cfg(replay_workers=32, fork_gc_freeze=False))["commit"]
    assert old["reserved_bytes"] - p["reserved_bytes"] == 2 * (int(2.9 * _cfg().largest_tensor()) - 250 * MiB)


def test_r19_retain_windowed_pool_terms():
    """Bounded staging + --retain host --retain-exclude hidden_m1: the pool is (raw - hidden M1) + raw x 0.25 (levels over every leaf);
    bounded + a non-host retention keeps the levels only; bounded without the exclusion (or an unrelated family) keeps the full pool."""
    full = adm.predict(_cfg())["commit"]
    raw, hid = adm.predict(_cfg())["derived"]["raw_bytes"], adm.predict(_cfg())["derived"]["hidden_bytes"]
    assert 0 < hid < raw and hid > 0.5 * raw
    kept = adm.predict(_cfg(bounded_staging=True, retain_exclude=["fa2_hidden_m1", "fa3_hidden_m1"]))["commit"]
    assert kept["pinned_pool"]["bytes"] == raw - hid + int(raw * 0.25) == next(t["bytes"] for t in kept["terms"] if t["name"] == "commit.pinned_pool")
    assert kept["pinned_pool"]["kept_raw_bytes"] == raw - hid and "- hidden M1" in kept["pinned_pool"]["basis"]
    assert kept["reserved_bytes"] < full["reserved_bytes"] and kept["pinned_pool"]["retain_exclude"] == ["fa2_hidden_m1", "fa3_hidden_m1"]
    assert adm.predict(_cfg(bounded_staging=True))["commit"]["pinned_pool"]["bytes"] == full["pinned_pool"]["bytes"]
    assert adm.predict(_cfg(bounded_staging=True, retain_exclude=["gemm_qkv"]))["commit"]["pinned_pool"]["bytes"] == full["pinned_pool"]["bytes"]
    lv = adm.predict(_cfg(bounded_staging=True, retain="none"))["commit"]["pinned_pool"]
    assert lv["bytes"] == int(raw * 0.25) and "levels only" in lv["basis"]
    # the alias
    assert adm.retain_exclude_families("hidden_m1") == ["fa2_hidden_m1", "fa3_hidden_m1"]
    assert adm.retain_exclude_families("fa3_hidden_m1_stream,gemm_qkv") == ["fa3_hidden_m1", "gemm_qkv"]
    assert adm.retain_exclude_families(None) == []


def test_largest_tensor_law_matches_the_ledger():
    # 'largest tensor bound': llama B1 392, B8 3140, B32 12560, B64 25120 MiB; r11 (LP 4096, T 512) 6274 MiB
    for b, mib in ((1, 392), (8, 3140), (32, 12560), (64, 25120)):
        c = adm.RowConfig(model="llama32-1b", requests=b, prompt_tokens=1024 * b, decode_rows=128 * b, max_context=1152, largest_request_rows=128, largest_request_ctx=1152)
        assert abs(c.largest_tensor() / MiB - mib) / mib < 0.05
    c = adm.RowConfig(model="llama32-1b", requests=1, prompt_tokens=4096, decode_rows=512, max_context=4608)
    assert abs(c.largest_tensor() / MiB - 6274) / 6274 < 0.05


def test_instances_law():
    assert abs(_cfg().instances() - 698_398) / 698_398 < 0.01
    assert abs(adm.RowConfig(model="phi3-mini", requests=8, prompt_tokens=3630, decode_rows=460).instances() - 1_315_324) / 1_315_324 < 0.01
    assert abs(adm.RowConfig(model="olmoe-1b-7b", requests=32, prompt_tokens=8659, decode_rows=1308).instances() - 5_745_046) / 5_745_046 < 0.01


def test_admission_decision_flips_on_the_limit():
    ok = adm.predict(_cfg(host_limit_bytes=200_000 * MiB))["commit"]
    no = adm.predict(_cfg(host_limit_bytes=60_000 * MiB))["commit"]
    assert ok["admit"] is True and no["admit"] is False and no["spare_mib"] < 0
    # the refusal names what does not fit, in bytes and by term
    assert ok["decision"]["admit"] is True and ok["decision"]["spare_bytes"] == ok["spare_mib"] * MiB + (200_000 * MiB - ok["reserved_bytes"]) % MiB
    d = no["decision"]
    assert d["admit"] is False and d["short_by_bytes"] == no["reserved_bytes"] - 60_000 * MiB
    assert next(iter(d["live_at_peak_mib"])) == "commit.pinned_pool" and "commit.pinned_pool" in d["reason"]
    assert adm.predict(_cfg(host_limit_bytes=None))["commit"]["decision"]["admit"] is None


def test_under_fitted_capture_models_are_flagged_not_silently_admitted():
    llama = adm.predict(_cfg())["match"]
    mistral = adm.predict(adm.RowConfig(model="mistral-7b", requests=8, prompt_tokens=3070, decode_rows=688, max_context=1152, host_limit_bytes=359_535 * MiB))["match"]
    assert llama["lower_bound_terms"] == []
    assert mistral["lower_bound_terms"] == ["match.capture.hidden_values", "match.capture.instance_records", "match.capture.token_values"]
    assert any("#60" in t["basis"] for t in mistral["terms"] if t["name"].startswith("match.capture."))


# ---- increment A path ------------------------------------------------------------------------------------------------
def test_observe_increment_a_named_allocs(tmp_path):
    ev = [
        {"kind": "span_start", "stage": "stage.commit", "id": "c", "t_mono": 0.0, "t_utc": "2026-09-21T00:00:00.000Z", "stream": "s", "seq": 1},
        {"kind": "mark", "name": "alloc", "obj": "commit.pinned_pool", "bytes": 100 * MiB, "alloc_id": "pool", "stage": "stage.commit", "t_mono": 1.0, "t_utc": "2026-09-21T00:00:01.000Z", "stream": "s", "seq": 2},
        {"kind": "mark", "name": "alloc", "obj": "commit.c2.match_oracle", "bytes": 20 * MiB, "alloc_id": "o0", "stage": "stage.commit", "t_mono": 2.0, "t_utc": "2026-09-21T00:00:02.000Z", "stream": "s", "seq": 3},
        {"kind": "mark", "name": "free", "obj": "commit.c2.match_oracle", "alloc_id": "o0", "stage": "stage.commit", "t_mono": 3.0, "t_utc": "2026-09-21T00:00:03.000Z", "stream": "s", "seq": 4},
        {"kind": "mark", "name": "alloc", "obj": "commit.c2.programs", "bytes": 30 * MiB, "alloc_id": "p0", "stage": "stage.commit", "t_mono": 3.5, "t_utc": "2026-09-21T00:00:03.500Z", "stream": "s", "seq": 5},
        {"kind": "span_end", "stage": "stage.commit", "id": "c", "t_mono": 5.0, "t_utc": "2026-09-21T00:00:05.000Z", "stream": "s", "seq": 6},
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev))
    res = [{"kind": "cgroup", "t_mono": 3.6, "cgroup": {"memory_current_bytes": 140 * MiB, "memory_stat": {"anon": 130 * MiB, "shmem": 0}}},
           {"kind": "process", "t_mono": 3.7, "processes": [{"pid": 1, "rss_bytes": 120 * MiB, "rss_anon_bytes": 110 * MiB}, {"pid": 2, "rss_bytes": 5 * MiB, "rss_anon_bytes": 5 * MiB}]}]
    (tmp_path / "resources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
    obs = adm.observe_increment_a(str(tmp_path))
    c = obs["stages"]["commit"]
    assert c["named_peak_mib"] == 130 and c["live_at_peak"] == {"commit.pinned_pool": 100, "commit.c2.programs": 30}
    assert c["cgroup_mib"] == 140 and c["anon_shmem_mib"] == 130 and c["rss_max_mib"] == 120 and c["rss_sum_mib"] == 115
    assert any("never freed" in n for n in obs["notes"])


# ---- recorded attempts as a regression fixture -----------------------------------------------------------------------------
EXPECT = {
    # attempt: (stage, predicted admit, outcome the model must reproduce, |residual_parent| bound in % when the observation is a peak)
    "b1_llama_l40s": ("commit", True, "PASS", 20),
    "llama_b8_l40s": ("commit", True, "PASS", 15),
    "phi3_b8": ("commit", True, "PASS", 10),
    "r22_b32": ("commit", True, "PASS", 10),
    "r11_llama_4096": ("commit", False, "FAIL", None),     # killed: observed is a lower bound; the plan must refuse
    "r23_b64_commit2": ("commit", False, "FAIL", None),
    "r67_olmoe_b32": ("commit", False, "FAIL", None),
}


@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
@pytest.mark.parametrize("attempt", sorted(EXPECT))
def test_r17_regression(attempt):
    stage, admit, outcome, tol = EXPECT[attempt]
    r = adm.compare(str(FIXTURE / attempt))
    s = r["stages"][stage]
    assert s["predicted_admit"] is admit, (attempt, s)
    assert (s["verdict"] or outcome) == outcome
    if tol is not None:
        assert abs(s["residual_parent_pct"]) <= tol, (attempt, s["residual_parent_pct"])
    else:
        # a refused, killed run: the reservation is at least the kill-time observation and over the limit
        assert s["predicted_mib"] >= s["observed_mib"] and s["predicted_mib"] > s["limit_mib"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
def test_r17_f_r17_24_explains_23_and_67():
    r23 = adm.compare(str(FIXTURE / "r23_b64_commit2"))
    pts = r23["points"]["commit/p0.replay"]
    assert abs(pts["residual_parent_mib"]) < 0.15 * pts["observed_rss_mib"]          # parent 293 GB predicted vs 257 GB at the kill (+14%, the pool 0.95 and programs still loading)
    assert pts["live"]["commit.c2.replay_workers"] > 100_000                          # >= 102.8 GB in the children at the kill
    assert r23["replay_workers"]["admissible"] < r23["replay_workers"]["requested"]
    r67 = adm.compare(str(FIXTURE / "r67_olmoe_b32"))
    s = r67["stages"]["commit"]
    assert s["predicted_admit"] is False and s["predicted_mib"] > s["limit_mib"]
    # the pool alone was 89 GB; the model's parent at the kill is within 20% of the recorded 154 GB
    assert abs(s["residual_parent_pct"]) <= 20


# ---- a lower-bound reservation never reads as a plain admit ---------------------------------------------------------------
@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
def test_c1_lower_bound_stage_is_not_admitted_on_the_60_fixture():
    d = FIXTURE / "r60_mistral_b8"
    r = adm.compare(str(d))
    m = r["stages"]["match"]
    assert m["predicted_admit"] is None, m
    pm = adm.predict(adm.row_config_from_attempt(str(d)))["match"]
    assert pm["verdict"] == "LOWER_BOUND" and pm["decision"]["admit"] is None
    assert "match.capture.hidden_values" in pm["decision"]["reason"] and "#60" in pm["decision"]["reason"]
    # over the limit a lower bound is still a refusal, and says so
    cfg = adm.row_config_from_attempt(str(d))
    cfg.host_limit_bytes = 50_000 * MiB
    pr = adm.predict(cfg)["match"]
    assert pr["verdict"] == "REFUSE" and pr["decision"]["admit"] is False and "LOWER BOUND" in pr["decision"]["reason"]
    # a clean model still admits
    assert adm.predict(_cfg())["match"]["verdict"] == "ADMIT"


# ---- the runner's advisory hook (row_pod.sh `plan`) -----------------------------------------------------------------------
WL = Path(__file__).resolve().parents[2] / "verity_vllm" / "observe" / "out/gen/sweep/workloads"


@pytest.mark.skipif(not WL.exists(), reason="sweep workloads not in this checkout")
def test_plan_writes_admission_json_and_never_fails(tmp_path, capsys):
    wl = WL / "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json"
    out = tmp_path / "admission.json"
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(wl), "--host-limit-bytes", str(188_000_000_000), "--snapshot-max-bytes", "33554432", "--snapshot-steps", "0,1", "--out", str(out)])
    line = capsys.readouterr().out.strip()
    assert rc == 0 and out.exists()
    doc = json.loads(out.read_text())
    assert doc["schema"] == adm.ADMISSION_SCHEMA and doc["advisory"] is True and doc["model"] == "mistral-7b"
    assert doc["config"] == {**doc["config"], "requests": 8, "prompt_tokens": 3250, "decode_rows": 516, "prefill_rows": 3258, "snapshot_max_bytes": 33554432}
    assert set(doc["stages"]) == {"build", "match", "commit"}
    assert doc["stages"]["commit"]["verdict"] == "REFUSE" and "commit" in doc["would_refuse"] and doc["stages"]["commit"]["decision"]["short_by_bytes"] > 0
    assert doc["stages"]["match"]["verdict"] == "LOWER_BOUND" and "match" in doc["lower_bound"]
    assert line.startswith("ADMISSION advisory WARN would-refuse commit") and "commit=REFUSE" in line and line == doc["line"]
    # unknown model: dense default, named, still a document and exit 0
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(WL / "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager.json"), "--host-limit-bytes", str(200_000_000_000), "--out", str(tmp_path / "a2.json")])
    line = capsys.readouterr().out.strip()
    assert rc == 0 and "no calibration for model 'smollm2-135m'" in line and "WARN" not in line
    # a broken input is reported in the line, exit 0 (advisory: the row must never stop here)
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(tmp_path / "missing.json")])
    assert rc == 0 and capsys.readouterr().out.startswith("ADMISSION advisory unavailable")
