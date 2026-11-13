"""verity_vllm.pipeline.telemetry.admission — synthetic-interval unit tests of the sweep and the model."""
from __future__ import annotations

from verity_vllm.pipeline.telemetry import admission as adm

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
    return adm.RowScale(**base)


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


def test_largest_tensor_law_matches_the_ledger():
    # 'largest tensor bound': llama B1 392, B8 3140, B32 12560, B64 25120 MiB; r11 (LP 4096, T 512) 6274 MiB
    for b, mib in ((1, 392), (8, 3140), (32, 12560), (64, 25120)):
        c = adm.RowScale(model="llama32-1b", requests=b, prompt_tokens=1024 * b, decode_rows=128 * b, max_context=1152, largest_request_rows=128, largest_request_ctx=1152)
        assert abs(c.largest_tensor() / MiB - mib) / mib < 0.05
    c = adm.RowScale(model="llama32-1b", requests=1, prompt_tokens=4096, decode_rows=512, max_context=4608)
    assert abs(c.largest_tensor() / MiB - 6274) / 6274 < 0.05


def test_instances_law():
    assert abs(_cfg().instances() - 698_398) / 698_398 < 0.01
    assert abs(adm.RowScale(model="phi3-mini", requests=8, prompt_tokens=3630, decode_rows=460).instances() - 1_315_324) / 1_315_324 < 0.01
    assert abs(adm.RowScale(model="olmoe-1b-7b", requests=32, prompt_tokens=8659, decode_rows=1308).instances() - 5_745_046) / 5_745_046 < 0.01


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
    mistral = adm.predict(adm.RowScale(model="mistral-7b", requests=8, prompt_tokens=3070, decode_rows=688, max_context=1152, host_limit_bytes=359_535 * MiB))["match"]
    assert llama["lower_bound_terms"] == []
    assert mistral["lower_bound_terms"] == ["match.capture.hidden_values", "match.capture.instance_records", "match.capture.token_values"]
    assert any("#60" in t["basis"] for t in mistral["terms"] if t["name"].startswith("match.capture."))
