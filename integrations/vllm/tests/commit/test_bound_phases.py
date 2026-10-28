"""The bounded host bound is the max over the process's phases, with the C2 Programs and the pinned
pow2 rounding as named terms -- and it reconstructs a measured B8 miss (VmHWM 223,776 MiB vs the 193,827 MiB sum) BEFORE the run.

That run: 93,052 (pool 59,220 + engine 7,100 + pow2 rounding 21,152 + ~5,600) | + C2 Programs 70,300 (never freed) | + replay
pool 54,800 = 223,776 MiB, with the 3-pair term unused.  The same derivation with the exact pool, the Programs freed after C2 and the
replay cache is what the next B8 run is admitted on.
"""
from __future__ import annotations

import gzip
import json

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.pipeline.commit import c2_programs_forecast  # noqa: E402
from verity_vllm.commit.committer.native_host import bounded_host_bound, schedule_stream_bound  # noqa: E402

MiB = 1 << 20
ASZ = {"per_token_bytes": 2900996, "per_seq_bytes": 607744, "per_step_bytes": 93184,
       "model": {"hidden": 2560, "heads": 32, "kv_heads": 8, "head_dim": 128, "layers": 36, "hidden_m1": True, "attn_backend": "fa3"},
       "prefill_step_bytes": 20203618816, "prefill_hidden_bytes": 17232297984, "decode_step_bytes": 22531588, "decode_hidden_bytes": 18929664,
       "retained_bytes_bound": 23087662080}
REQS8 = [{"prompt_token_ids": list(range(n)), "arrive_step": 0, "max_tokens": c}
         for n, c in zip((365, 633, 11, 1024, 817, 449, 165, 73), (128, 113, 95, 35, 121, 3, 9, 2))]
COMMON = dict(learn_budget_bytes=4096 * MiB, openings=64, chunk_bytes=256, window_bytes=256 * MiB, bounded=True, open_after_release=True,
              host_cap_bytes=239372 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"})
C2_B8 = 70300 * MiB            # measured: RM._programs_for on the 8 Programs of record (7,853,122 rows)


def _s8():
    return schedule_stream_bound(ASZ, REQS8, {"max_tokens": 128})


def test_attempt_1_is_reconstructed_by_the_phase_bound_before_the_run():
    """torch pool (pow2 rounding), Programs never freed, 3 pairs -- the derivation attempt 1 should have had: its bound covers the observed
    223,776 MiB peak and names the openings-after-release phase (run + C2 Programs + replay pool) as the peak."""
    b = bounded_host_bound(_s8(), engine_rss_bytes=2536 * MiB, pairs=3, c2_programs_bytes=C2_B8, c2_freed_before_replay=False, pool_exact=False, **COMMON)
    assert b["host_peak_bound_bytes"] > 223776 * MiB, b["host_peak_bound_bytes"] // MiB
    assert b["peak_phase"] == "openings_after_release"
    ph = b["phases_bytes"]
    assert ph["openings_after_release"] == ph["run"] + b["replay_bytes"] + C2_B8 and ph["c2_sampled_replay"] == ph["run"] + C2_B8
    assert b["pool_rounding_bytes"] == b["levels_bytes"] + b["raw_kept_bytes"] + b["learn_budget_bytes"] > 21152 * MiB      # covers the measured 21,152
    assert b["host_peak_bound_bytes"] > 239372 * MiB, "the a97746261 launch was not admissible under this derivation"
    assert b["dominant_contributor"]["peak_phase"] == "openings_after_release" and "peak phase" in b["dominant_contributor"]["text"]
    # the 193,827 MiB sum of record = the same terms without c2_programs and pool_rounding (and the observed peak beat it by 29,949)
    legacy = b["engine_rss_bytes"] + b["levels_bytes"] + b["raw_kept_bytes"] + b["learn_budget_bytes"] + b["replay_bytes"] + b["slack_bytes"] + b["pairs_extra_bytes"]
    assert abs(legacy // MiB - 193827) < 400 and 223776 * MiB > legacy


def test_attempt_2_derivation_exact_pool_programs_freed_replay_cache_admits_b8_with_margin():
    """exact pool (RSS == bytes_total, freed on release -> no rounding, no pairs stacking), Programs freed before the replay pool, the
    replay cache: the peak phase is the C2 load (run + Programs) and the 3-pair B8 row sits under the cgroup with margin."""
    b = bounded_host_bound(_s8(), engine_rss_bytes=2536 * MiB, pairs=3, c2_programs_bytes=C2_B8, c2_freed_before_replay=True, pool_exact=True,
                           replay_cache=True, **COMMON)
    ph = b["phases_bytes"]
    assert b["pool_rounding_bytes"] == 0 and b["pairs_extra_bytes"] == 0 and b["pairs"] == 3
    assert b["peak_phase"] == "c2_sampled_replay" and ph["c2_sampled_replay"] > ph["openings_after_release"] > ph["run"]
    assert b["host_peak_bound_bytes"] == ph["c2_sampled_replay"] + b["slack_bytes"]
    assert b["host_peak_bound_bytes"] < 239372 * MiB * 0.9 and b["margin_frac"] > 0.30, (b["host_peak_bound_bytes"] // MiB, b["margin_frac"])
    assert b["dominant_contributor"]["host_term"] == "c2_programs"
    # every term says what kind of number it is
    assert set(b["term_kinds"]) >= {"c2_programs", "pool_rounding", "pairs", "replay", "levels"} and "forecast" in b["term_kinds"]["c2_programs"]
    assert b["c2_freed_before_replay"] is True and b["replay_cache"] is True
    # the observed attempt-1 phases, re-scored under this derivation: run 93,052 - rounding 21,152 = 71,900 < run + slack; run + programs
    # 142,200 < c2 phase + slack -- the numbers the run measures (C2 PROGRAMS FREED, pinned_pool_mode) are what calibrates it
    assert ph["run"] + b["slack_bytes"] > (93052 - 21152) * MiB and ph["c2_sampled_replay"] + b["slack_bytes"] > (93052 - 21152 + 70300) * MiB


def test_c2_programs_forecast_counts_rows_across_chunk_boundaries_and_names_absence(tmp_path):
    root = tmp_path / "programs"
    n_rows = {"LP7_T1": 3, "LP9_T2": 5}
    js = 0
    for name, n in n_rows.items():
        d = root / f"build_request_{name}"
        d.mkdir(parents=True)
        rows = [{"i": i, "spec": "Embedding_v1{V=8,H=4}", "form": "call", "name": f"m/e[{i}]", "args": [[["p", 0, 0, 1]]]} for i in range(n)]
        text = json.dumps({"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": [], "rows": rows}).encode()
        js += len(text)
        with gzip.open(d / "instances.json.gz", "wb") as f:
            f.write(text)
    fc = c2_programs_forecast(str(root), None)
    assert fc["rows"] == 8 and fc["json_bytes"] == js and fc["bytes"] == int(js * 15.0) and len(fc["files"]) == 2 and fc["why"] is None
    assert fc["kind"] == "empirical_forecast" and "B8" in fc["calibration"]
    # a single Program dir (B=1 rows pass --program-dir)
    fc1 = c2_programs_forecast(None, str(root / "build_request_LP9_T2"))
    assert fc1["rows"] == 5 and len(fc1["files"]) == 1
    # absence is named, never a silent 0
    fc0 = c2_programs_forecast(str(tmp_path / "nowhere"), None)
    assert fc0["bytes"] == 0 and fc0["rows"] == 0 and "no instance files" in fc0["why"]


def test_row_counting_is_exact_when_the_needle_straddles_a_read_boundary(tmp_path, monkeypatch):
    """the streamed counter keeps a (len(needle) - 1)-byte tail between 64 MiB reads; force tiny reads to exercise the seam."""
    import verity_vllm.pipeline.commit as CD

    d = tmp_path / "build_request_LP1_T1"
    d.mkdir()
    rows = [{"i": i, "spec": "s", "form": "call", "name": "n", "args": []} for i in range(1000)]
    text = json.dumps({"rows": rows}).encode()
    with gzip.open(d / "instances.json.gz", "wb") as f:
        f.write(text)
    real_open = gzip.open

    class _Small:
        def __init__(self, fh):
            self.fh = fh

        def read(self, n):
            return self.fh.read(7)          # 7-byte reads: the 6-byte needle straddles most seams

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

    monkeypatch.setattr(gzip, "open", lambda p, m="rb": _Small(real_open(p, m)))
    fc = CD.c2_programs_forecast(str(tmp_path), None)
    assert fc["rows"] == 1000 and fc["json_bytes"] == len(text)


def test_fork_children_is_a_named_cgroup_term_of_the_c2_phase():
    """The C2 evaluator children's private pages are charged to the cgroup, never to the parent's VmHWM: the term sits in
    the c2 phase, says its kind and scope, and attempt 3's derivation (exact SHARED pool, Programs freed, replay cache, 8 GiB children)
    keeps the B8 3-pair workload under the cgroup with margin."""
    b0 = bounded_host_bound(_s8(), engine_rss_bytes=2522 * MiB, pairs=3, c2_programs_bytes=70439 * MiB, c2_freed_before_replay=True, pool_exact=True,
                            replay_cache=True, **COMMON)
    b = bounded_host_bound(_s8(), engine_rss_bytes=2522 * MiB, pairs=3, c2_programs_bytes=70439 * MiB, c2_freed_before_replay=True, pool_exact=True,
                           replay_cache=True, fork_children_bytes=8 << 30, **COMMON)
    assert b["fork_children_bytes"] == 8 << 30 and b0["fork_children_bytes"] == 0
    assert b["phases_bytes"]["c2_sampled_replay"] == b0["phases_bytes"]["c2_sampled_replay"] + (8 << 30)
    assert b["phases_bytes"]["run"] == b0["phases_bytes"]["run"] and b["phases_bytes"]["openings_after_release"] == b0["phases_bytes"]["openings_after_release"]
    assert b["peak_phase"] == "c2_sampled_replay" and b["host_peak_bound_bytes"] == b0["host_peak_bound_bytes"] + (8 << 30)
    assert "cgroup scope" in b["term_kinds"]["fork_children"] and "fork_children" in b["scope"]["cgroup_only"] and "fork_children" not in b["scope"]["parent_vmhwm"]
    assert "fork_children" in b["derivation"]
    # attempt 2's measured c2 phase (VmHWM 133,531 MiB at the end of the Program load) is under b0's c2 phase + slack; attempt 3's bound
    # with the children term is 148,467 + 8,192 = 156,659 -> 34.6 pct margin under the 239,372 MiB cgroup
    assert b0["phases_bytes"]["c2_sampled_replay"] + b0["slack_bytes"] > 133531 * MiB
    assert abs(b["host_peak_bound_bytes"] // MiB - 156659) < 300, b["host_peak_bound_bytes"] // MiB
    assert b["margin_frac"] > 0.30


def test_c2_residual_and_pool_carry_are_named_terms_that_rescore_attempt_3():
    """Attempt 3 (B8, 3 pairs, exact shared pool, Programs freed, replay cache, 8 GiB children) held its
    156,654 MiB bound (VmHWM 150,935) but the peak was pair 2's replay phase (151,097), not the predicted c2 phase: the freed Programs left
    31,460 MiB resident and pair 1 carried 20,047 MiB of pair 0's pool pages.  With the two terms the later-pair replay phase is the peak, the
    observed phases sit under their predictions, and the workload is still admissible with margin."""
    common = dict(engine_rss_bytes=2517 * MiB, pairs=3, c2_programs_bytes=70439 * MiB, c2_freed_before_replay=True, pool_exact=True, replay_cache=True,
                  fork_children_bytes=8 << 30, **COMMON)
    b0 = bounded_host_bound(_s8(), **common)
    b = bounded_host_bound(_s8(), c2_residual_bytes=int(70439 * MiB * 0.45), pool_carry_bytes=20 << 30, **common)
    assert b0["c2_residual_bytes"] == 0 and b0["pool_carry_bytes"] == 0 and "later_pair_replay" in b0["phases_bytes"]
    assert b0["phases_bytes"]["later_pair_replay"] == b0["phases_bytes"]["openings_after_release"]
    ph = b["phases_bytes"]
    assert ph["run"] == b0["phases_bytes"]["run"] and ph["c2_sampled_replay"] == b0["phases_bytes"]["c2_sampled_replay"]
    assert ph["openings_after_release"] == ph["run"] + b["replay_bytes"] + b["c2_residual_bytes"]
    assert ph["later_pair_replay"] == ph["openings_after_release"] + (20 << 30)
    assert b["peak_phase"] == "later_pair_replay" and b["host_peak_bound_bytes"] == ph["later_pair_replay"] + b["slack_bytes"]
    # attempt 3 observed (footprint rss, MiB): pair 0 run 62,082 | c2 load end 133,527 | pair 0 replay 130,895 | pair 1 run 112,851 | pair 2 replay 151,097
    assert ph["run"] > 62082 * MiB and ph["c2_sampled_replay"] > 133527 * MiB and ph["openings_after_release"] > 130895 * MiB
    assert ph["later_pair_replay"] > 151097 * MiB and ph["later_pair_replay"] - 151097 * MiB < 25000 * MiB      # under, not by a mile
    # the TERMS are the bound (c2_residual 0.45 x 70,439 = 31,698; pool_carry 20 GiB = 20,480), not the measured 31,460 / 20,047 they were
    # read from: run 71,878 + replay 46,425 + 31,698 + 20,480 + slack 6,144 = 176,624 MiB -> 26.2 pct margin under the 239,372 MiB cgroup
    assert b["margin_frac"] > 0.20 and abs(b["host_peak_bound_bytes"] // MiB - 176624) < 300, b["host_peak_bound_bytes"] // MiB
    assert "c2_residual" in b["term_kinds"] and "pool_carry" in b["term_kinds"] and "c2_residual" in b["scope"]["parent_vmhwm"]
    assert "later_pair_replay" in b["derivation"]
    # the residual is a term only when the Programs are freed first (else c2_programs itself is in the replay phase)
    bn = bounded_host_bound(_s8(), **{**common, "c2_freed_before_replay": False}, c2_residual_bytes=30000 * MiB)
    assert bn["c2_residual_bytes"] == 0 and bn["phases_bytes"]["openings_after_release"] == bn["phases_bytes"]["run"] + bn["replay_bytes"] + 70439 * MiB
    # one pair: no carry, no later-pair phase
    b1 = bounded_host_bound(_s8(), **{**common, "pairs": 1}, c2_residual_bytes=30000 * MiB, pool_carry_bytes=20 << 30)
    assert b1["pool_carry_bytes"] == 0 and "later_pair_replay" not in b1["phases_bytes"]
