"""Commit admission: the scheduled host-retention bound is SOUND (>= every observed step) and TIGHT (within 15 % of the
run total) on three recorded runs with per-step `n_bytes`, refuses by name over the cgroup, and is torch-free."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import admission_bound as ADM

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures" / "admission"
CAP_CP_BIG = 376_999_997_440          # a node's /sys/fs/cgroup/memory.max (= 359,535 MiB)


def _fixture(name: str):
    d = json.loads((FIX / f"{name}.json").read_text())
    wl = json.loads((REPO / "workloads" / f"{d['row_id']}.json").read_text())
    obs = {int(k): int(v) for k, v in d["observed_n_bytes_pair0_instrumented"].items()}
    return d["sizing"], wl, obs


@pytest.mark.parametrize("name,max_ratio", [("b16", 1.06), ("b4", 1.12), ("phi3b1", 1.16)])
def test_scheduled_bound_is_sound_and_tight_on_the_r16_records(name, max_ratio):
    sizing, wl, obs = _fixture(name)
    mnbt = wl["engine"]["args"]["max_num_batched_tokens"]
    sched = ADM.scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=mnbt, lag=1)
    chk = ADM.compare_observed(sched, obs)
    assert chk["sound"], f"{name}: under-predicted steps {chk['under']}"
    assert chk["steps_under_predicted"] == 0
    assert 1.0 <= chk["ratio_pred_over_obs"] <= max_ratio, chk
    # every observed step is predicted (the lag row is one extra scheduled step beyond the record's last step)
    assert sched["steps"] >= len(obs)


def test_b16_record_the_per_step_geometry_reproduces_the_record_to_the_percent():
    """The load-bearing fact: mixed prefill+decode steps (an arriving prompt beside 1-7 decodes) reproduce the record to within 0.6 % on
    11 of 15, the two-prompt step exactly (each arriving prompt is its own compacted stream, not a padded launch); the rest are over."""
    sizing, wl, obs = _fixture("b16")
    sched = ADM.scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=5348, lag=1)
    pred = {x["step"]: x for x in sched["per_step"]}
    prefill_steps = (0, 11, 18, 20, 26, 34, 36, 37, 40, 53, 56, 61, 65, 70, 76)
    ratios = {s: pred[s]["bytes"] / obs[s] for s in prefill_steps}
    assert all(pred[s]["prefills"] >= 1 for s in prefill_steps)
    assert all(1.0 <= r <= 1.11 for r in ratios.values()), ratios
    assert sum(1 for r in ratios.values() if r <= 1.006) >= 11, ratios
    assert pred[20]["prefills"] == 2 and ratios[20] <= 1.002
    # steps 36 / 53: a request's REACHABLE context (prompt + cap + lag) is one key block beyond the kv the record had at that step -> NB 9
    # vs 8 = +8-10 % on that step's stream, the price of a bound that does not depend on when within its life a request is scheduled
    assert 1.08 <= ratios[36] <= 1.11 and 1.07 <= ratios[53] <= 1.09, (ratios[36], ratios[53])
    for s in (5, 80, 120, 160):                                                              # pure decode steps: +3 % (NB+1); the 1-row step +12 %
        assert pred[s]["prefills"] == 0
        assert 1.0 <= pred[s]["bytes"] / obs[s] <= 1.12, (s, pred[s]["bytes"], obs[s])
    assert sched["peak_step"] == 36                                                          # the 1024-token prompt beside 4 decodes


def test_envelope_bound_was_the_loose_one_and_the_scheduled_bound_admits_the_cp_big_rows():
    """The recorded numbers: envelope x1.25 = 559,384 MiB for llama B16 mixed-arrivals against 359,535 MiB; the
    run bound 42.5 GiB.  The scheduled bound admits it (55.9 GB) and names the envelope beside it."""
    sizing, wl, obs = _fixture("b16")
    rec = ADM.admission(sizing, wl["requests"], CAP_CP_BIG, retain="host", max_num_batched_tokens=5348, lag=1,
                        envelope_bound_bytes=sizing["retained_bytes_bound"])
    assert rec["envelope_predicted_host_bytes"] >> 20 == 559_384
    assert rec["admitted"] is True
    assert rec["scheduled_raw_bytes"] >= sum(obs.values())
    assert rec["predicted_host_bytes"] < 60 << 30
    assert "envelope bound 559384 MiB was over it" in rec["why"]
    # phi3 B8 1024/128 from the phi3 widths (its envelope bound, 582,111 MiB, is over the cap)
    sizing3, _wl3, _ = _fixture("phi3b1")
    wl8 = json.loads((REPO / "workloads" / "phi3-mini__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json").read_text())
    rec8 = ADM.admission(sizing3, wl8["requests"], CAP_CP_BIG, retain="host", max_num_batched_tokens=3638, lag=1)
    assert rec8["admitted"] is True and (100 << 30) < rec8["predicted_host_bytes"] < (120 << 30), rec8["predicted_host_bytes"]
    assert rec8["scheduled"]["peak_step"] == 0 and rec8["scheduled"]["steps"] == 129        # 8 concurrent prefills; caps + 1 lag row


def test_refuses_by_name_over_the_cap_and_a_non_host_retention_is_predicted_not_refused():
    """the refusal domain is `--retain host` unbounded; a non-host
    retention on the windowed path is not 'predicted host 0 MiB' -- the unbounded windowed collector copies every window to the
    pinned pool regardless of --retain (a `--retain none` run allocates the 112 GiB pool), so the pool is predicted,
    the excess is named as a WARNING on the record, and the launch is still not refused here."""
    sizing, wl, _ = _fixture("b16")
    small = 32 << 30
    rec = ADM.admission(sizing, wl["requests"], small, retain="host", max_num_batched_tokens=5348, lag=1)
    assert rec["admitted"] is False
    assert "exceeds the container memory limit" in rec["why"] and "not launched unchanged" in rec["why"]
    assert rec["predicted_host_bytes"] > small
    for mode in ("none", "sample", "device"):
        r2 = ADM.admission(sizing, wl["requests"], small, retain=mode, max_num_batched_tokens=5348, lag=1)
        assert r2["admitted"] is True and mode in r2["why"]
        assert r2["predicted_host_bytes"] == rec["predicted_host_bytes"] and r2["would_exceed_by_bytes"] == rec["predicted_host_bytes"] - small
        assert "WARNING" in r2["why"] and "regardless of --retain" in r2["pool_rule"] and r2["staging"] == {"windowed": True, "bounded": False, "retain_exclude": []}
        # the pre-P0 per-step-block path copies nothing under a non-host retention: 0 by that rule, named
        r2b = ADM.admission(sizing, wl["requests"], small, retain=mode, max_num_batched_tokens=5348, lag=1, windowed=False)
        assert r2b["predicted_host_bytes"] == 0 and "per-step blocks" in r2b["pool_rule"] and "WARNING" not in r2b["why"]
    r3 = ADM.admission(sizing, wl["requests"], None, retain="host", max_num_batched_tokens=5348, lag=1)
    assert r3["admitted"] is True and "no container memory limit" in r3["why"] and r3["would_exceed_by_bytes"] is None


def test_bounded_staging_retain_exclude_hidden_predicts_the_kept_pool():
    """the RETAIN-windowed pool: under --bounded-staging --retain host --retain-exclude hidden_m1 the pool
    is (raw - hidden M1) + the levels over every leaf; the hidden stream is the bulk of the raw bytes at 1024/128; bounded + a non-host retention keeps the
    levels only.  Bounded runs are governed by the bounded HOST BOUND: recorded here, never refused."""
    sizing, wl, _ = _fixture("b16")
    small = 32 << 30
    full = ADM.admission(sizing, wl["requests"], small, retain="host", max_num_batched_tokens=5348, lag=1)
    raw, hidden = full["scheduled_raw_bytes"], full["scheduled_hidden_bytes"]
    assert 0 < hidden < raw and hidden > 0.5 * raw, (raw, hidden)                 # the M1 stream dominates the stream at this geometry
    kept = ADM.admission(sizing, wl["requests"], small, retain="host", max_num_batched_tokens=5348, lag=1, bounded=True, retain_exclude={"fa2_hidden_m1", "fa3_hidden_m1"})
    assert kept["predicted_host_bytes"] == raw - hidden + int(raw * (full["levels_factor"] - 1.0)) < full["predicted_host_bytes"]
    assert kept["admitted"] is True and "bounded HOST BOUND governs" in kept["why"] and kept["staging"]["retain_exclude"] == ["fa2_hidden_m1", "fa3_hidden_m1"]
    # bounded without the exclusion keeps everything; excluding an unrelated family keeps everything too (the rule is by hidden family)
    b_all = ADM.admission(sizing, wl["requests"], small, retain="host", max_num_batched_tokens=5348, lag=1, bounded=True)
    assert b_all["predicted_host_bytes"] == full["predicted_host_bytes"]
    b_other = ADM.admission(sizing, wl["requests"], small, retain="host", max_num_batched_tokens=5348, lag=1, bounded=True, retain_exclude={"gemm_qkv"})
    assert b_other["predicted_host_bytes"] == full["predicted_host_bytes"]
    lv = ADM.admission(sizing, wl["requests"], small, retain="none", max_num_batched_tokens=5348, lag=1, bounded=True)
    assert lv["predicted_host_bytes"] == int(raw * (full["levels_factor"] - 1.0)) and "levels only" in lv["pool_rule"]
    # the CLI alias hidden_m1 names both tap families
    rule = ADM.predicted_pool_bytes(1000, 600, retain="host", levels_factor=1.25, windowed=True, bounded=True, retain_exclude={"fa2_hidden_m1"})
    assert rule[0] == 650 and "hidden M1" in rule[1]


def test_lag_rows_chunking_and_the_rope_correction_are_named_and_monotone():
    sizing, wl, _ = _fixture("b16")
    s0 = ADM.scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=5348, lag=0)
    s1 = ADM.scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=5348, lag=1)
    assert s1["total_bytes"] > s0["total_bytes"] and s1["decode_rows"] == s0["decode_rows"] + len(wl["requests"])
    assert s1["horizon"] == s0["horizon"] + 1
    # a token budget below the longest prompt adds chunk padding (per_step per extra chunk), never fewer bytes
    sc = ADM.scheduled_step_bytes(sizing, wl["requests"], max_num_batched_tokens=512, lag=1)
    assert sc["total_bytes"] > s1["total_bytes"]
    assert max(x["chunks"] for x in sc["per_step"]) >= 2
    assert sc["total_bytes"] - s1["total_bytes"] == sum((x["chunks"] - 1) for x in sc["per_step"]) * sizing["per_step_bytes"]
    # an older sizing record counts the shared RotaryEmbedding once: corrected by name; a record that says rope_per_layer is taken as is
    pt, note = ADM.per_token_bytes_of(sizing)
    assert pt == sizing["per_token_bytes"] + sizing["widths"]["RotaryEmbedding"] * (sizing["model"]["layers"] - 1) and "RotaryEmbedding" in note
    assert any("RotaryEmbedding" in a for a in s1["assumptions"])
    pt2, note2 = ADM.per_token_bytes_of(dict(sizing, rope_per_layer=True))
    assert pt2 == sizing["per_token_bytes"] and note2 is None
    with pytest.raises(ValueError):
        ADM.scheduled_step_bytes(sizing, [{"prompt_len": 0, "max_tokens": 1, "arrive_step": 0}])


def test_cli_is_torch_free_and_reports_the_observed_check(tmp_path):
    sizing, wl, obs = _fixture("b16")
    sz = tmp_path / "sizing.json"; sz.write_text(json.dumps(sizing))
    st = tmp_path / "steps.jsonl"
    st.write_text("".join(json.dumps({"pair": 0, "arm": "instrumented", "step": s, "n_bytes": b}) + "\n" for s, b in obs.items()))
    cmd = [sys.executable, "-c", "import sys; sys.modules['torch'] = None; from verity_vllm.pipeline import cli; sys.exit(cli.main(['admission-bound', *sys.argv[1:]]))",
           "--workload", str(REPO / "workloads" / f"{wl['sweep']['row_id']}.json"), "--sizing", str(sz),
           "--cap-bytes", str(CAP_CP_BIG), "--observed", str(st)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env={"PYTHONPATH": f"{REPO}:{REPO / 'vllm-poc'}", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert "ADMITTED" in p.stdout and "SOUND" in p.stdout and "under-predicted 0" in p.stdout
    p2 = subprocess.run(cmd[:-2] + ["--cap-bytes", str(16 << 30)], cwd=REPO, capture_output=True, text=True,
                        env={"PYTHONPATH": f"{REPO}:{REPO / 'vllm-poc'}", "PATH": "/usr/bin:/bin"})
    assert p2.returncode == 3 and "REFUSED" in p2.stdout


def test_host_cap_names_its_source_and_a_bare_host_is_still_a_bound(monkeypatch, tmp_path):
    """no visible cgroup limit -> MemTotal, named; a cgroup limit wins when present; the record carries a domain."""
    import verity_vllm.pipeline.admission_bound as M
    real = M.Path.read_text

    def fake(self, *a, **k):
        p = str(self)
        if p.startswith("/sys/fs/cgroup"):
            raise OSError(p)
        if p == "/proc/meminfo":
            return "MemTotal:       16384000 kB\nMemFree: 1 kB\n"
        return real(self, *a, **k)

    monkeypatch.setattr(M.Path, "read_text", fake)
    cap, src = M.host_cap()
    assert cap == 16384000 * 1024 and "MemTotal" in src
    sizing, wl, _ = _fixture("b4")
    rec = M.admission(sizing, wl["requests"], cap)
    assert rec["admitted"] is False and rec["domain"].startswith("--retain host")

    def fake2(self, *a, **k):
        if str(self) == "/sys/fs/cgroup/memory.max":
            return "377000000000\n"
        return fake(self, *a, **k)

    monkeypatch.setattr(M.Path, "read_text", fake2)
    assert M.host_cap() == (377000000000, "/sys/fs/cgroup/memory.max")


def test_the_commit_admission_counts_the_declared_lag():
    """The host admission's lag rows are the manifest target's declared LAG, never a fallback of 1 when an import misses."""
    pytest.importorskip("torch")
    import inspect

    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as C
    from verity_vllm.pipeline import global_program as GP

    wl = va.load_workload(str(REPO / "workloads" / "olmoe-1b-7b__bf16__l40s__tp1__b32__i1024__o128__mixed__greedy__bi-eager.json"), None)
    assert C.admission_lag(wl) == GP.workload_target(wl["target"]).lag()
    for declared, lag in ((False, 0), (True, 1)):
        assert C.admission_lag(dict(wl, target={**(wl["target"] or {}), "async_scheduling": declared})) == lag
    assert "lag=admission_lag(wl)" in inspect.getsource(C.main)


def test_the_resident_bytes_add_to_the_pool_and_can_refuse():
    """The Commit's peak is the pinned pool plus what it holds beside it; a pool that fits alone can still be refused."""
    sizing, wl, _ = _fixture("b16")
    base = ADM.admission(sizing, wl["requests"], None, retain="host", max_num_batched_tokens=5348, lag=1)
    pool = base["predicted_host_bytes"]
    cap = pool + (4 << 30)
    alone = ADM.admission(sizing, wl["requests"], cap, retain="host", max_num_batched_tokens=5348, lag=1)
    both = ADM.admission(sizing, wl["requests"], cap, retain="host", max_num_batched_tokens=5348, lag=1, resident={"bytes": 8 << 30})
    assert alone["admitted"] is True and alone["pool_bytes"] == pool
    assert both["pool_bytes"] == pool and both["predicted_host_bytes"] == pool + (8 << 30)
    assert both["admitted"] is False and "resident beside it 8192 MiB" in both["why"]


@pytest.mark.parametrize("row", ["23", "67"])
def test_the_commit_admission_covers_the_recorded_peaks(row):
    """#23 OOM'd at the 239,372 MiB limit with its 182,765 MiB pool admitted; #67 completed.  Pool + resident covers both peaks, refuses
    #23 and admits #67 within 25 % of its peak; the pool alone is under both."""
    peaks = json.loads((FIX / "commit_peaks.json").read_text())
    r, limit = peaks["rows"][row], peaks["limit_bytes"]
    wl = json.loads((REPO / "workloads" / f"{r['row_id']}.json").read_text())
    res = ADM.committer_resident(wl, limit, pairs=r["pairs"], replay_workers=r["replay_workers"])
    pool, peak = r["pool_mib"] << 20, r["peak_anon_shmem_mib"] << 20
    predicted = pool + res["bytes"]
    assert pool < peak <= predicted
    assert res["peak_point"] == "p0.replay" and "commit.c2.replay_workers" in res["live_at_peak_mib"]
    assert (predicted > limit) is (row == "23")
    if not r["peak_is_lower_bound"]:
        assert predicted <= 1.25 * peak


def test_commit_main_passes_the_resident_bytes():
    src = (REPO / "verity_vllm" / "pipeline" / "commit.py").read_text()
    assert "resident=ADM.committer_resident(wl, host_cap, pairs=args.pairs, replay_workers=args.replay_workers)" in src


def test_row74_both_staging_modes():
    """#74 (Qwen3-4B-FP8, H100): unbounded, the pool of every window (152,257 MiB raw x 1.25) + resident is over the 251 GB limit and at or
    above what the run held when it was stopped; bounded with the FA3 hidden stream excluded it admits, and its pool counts the Merkle
    levels over every leaf (the warm-up primed 59,246 MiB pinned; kept raw x 1.25 alone was 16,058 MiB)."""
    peaks = json.loads((FIX / "commit_peaks.json").read_text())
    r, limit, M = peaks["rows"]["74"], peaks["limit_bytes"], 1 << 20
    wl = json.loads((REPO / "workloads" / f"{r['row_id']}.json").read_text())
    res = ADM.committer_resident(wl, limit, pairs=r["pairs"], replay_workers=r["replay_workers"])["bytes"]
    raw, hid = r["scheduled_raw_mib"] * M, r["scheduled_hidden_mib"] * M
    unb, _ = ADM.predicted_pool_bytes(raw, hid, retain="host", levels_factor=ADM.LEVELS_FACTOR, windowed=True, bounded=False, retain_exclude=None)
    assert unb + res > limit and unb + res >= r["unbounded"]["peak_anon_shmem_mib"] * M
    bnd, rule = ADM.predicted_pool_bytes(raw, hid, retain="host", levels_factor=ADM.LEVELS_FACTOR, windowed=True, bounded=True,
                                         retain_exclude={"fa3_hidden_m1"})
    assert bnd + res < limit and "levels over every leaf" in rule
    assert 0.8 <= bnd / (r["bounded_exclude_fa3_hidden_m1"]["warmup_pinned_mib"] * M) <= 1.0
