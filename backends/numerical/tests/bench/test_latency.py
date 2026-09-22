"""Pinned tests of the latency model (``verity_numerical.bench.latency``)."""

from __future__ import annotations

import math

import pytest
from verity_numerical.bench import latency as L

A = L.Design(name="A", prove_s=260.0, verify_s=26.0, depth=355, bytes_pv=70e6, mem_per_batch_bytes=34e9, mem_capacity_bytes=256e9,
             B=4096, hardware="cpu", proof_class="NON_ZK_PROOF_DIAGNOSTIC", source="test")
B = L.Design(name="B", prove_s=0.25, verify_s=0.4, depth=3, bytes_pv=240e6, mem_per_batch_bytes=22e9, mem_capacity_bytes=24e9,
             B=4096, hardware="gpu", proof_class="ARITHMETIC_DIAGNOSTIC", source="test")


def test_e2e_is_prove_plus_verify_plus_depth_rtt_plus_transfer():
    e = L.e2e(A, 50.0, 25.0)
    assert e["network"] == pytest.approx(355 * 0.050)
    assert e["transfer"] == pytest.approx(70e6 * 8 / 25e9)
    assert e["total"] == pytest.approx(260 + 26 + 17.75 + 0.0224)
    assert L.e2e(A, 0.0)["total"] == 286.0
    with pytest.raises(ValueError):
        L.e2e(A, -1.0)


def test_pinned_serial_numbers_for_the_brief():
    # A depth ~400 at 50 ms is ~20 s of waiting; B-Ligero depth 3-4 is 0.15-0.2 s
    assert L.e2e(L.replace(A, depth=400), 50.0)["network"] == pytest.approx(20.0)
    assert L.e2e(L.replace(B, depth=4), 50.0)["network"] == pytest.approx(0.2)


def test_n_saturate_and_pipeline_throughput():
    # zero RTT: one batch in flight already saturates a prover whose verifier is faster than it
    p = L.pipeline(A, 0.0)
    assert p["n_saturate"] == 2  # ceil((260 + 26) / 260)
    assert p["batches_per_s"] == pytest.approx(1 / 260)
    # 50 ms RTT: e2e 303.75 s / 260 s -> 2 in flight; fits (68 GB of 256)
    p = L.pipeline(A, 50.0)
    assert p["n_saturate"] == 2 and p["memory_fits"]
    assert p["binding"] == "prover compute"
    # B on a 24 GB card holds one batch: at 50 ms its e2e is 0.8 s so N_sat = 4 but N_mem = 1 -> memory binds
    p = L.pipeline(B, 50.0)
    assert p["n_saturate"] == 4 and p["n_memory_max"] == 1 and p["n_inflight"] == 1
    assert p["binding"].startswith("memory")
    assert p["batches_per_s"] == pytest.approx(1 / 0.8)
    assert p["vu_per_s"] == pytest.approx(4096 / 0.8)


def test_explicit_n_inflight_is_capped_by_prover_and_verifier():
    p = L.pipeline(B, 50.0, n_inflight=100)
    assert p["batches_per_s"] == pytest.approx(1 / 0.4)  # verifier-bound
    assert p["binding"] == "verifier compute"
    assert not p["memory_fits"]


def test_memory_binding_rtt():
    # B: N_mem = 1, so memory binds as soon as e2e > prove: (1 x 0.25 - 0.65) / 3 < 0 -> 0 ms (already at RTT 0)
    assert L.memory_binding_rtt(B) == 0.0
    # A: N_mem = 7: e2e > 7 x 260 = 1820 s needs rtt > (1820 - 286) / 355 = 4.32 s
    assert L.memory_binding_rtt(A) == pytest.approx((7 * 260 - 286) / 355 * 1000)
    assert L.memory_binding_rtt(L.replace(A, depth=0)) is None


def test_crossover_rtt():
    slow_shallow = L.Design(name="S", prove_s=300.0, verify_s=0.0, depth=3, bytes_pv=0, mem_per_batch_bytes=1, mem_capacity_bytes=1,
                            B=4096, hardware="x", proof_class="x", source="x")
    # 286 + 355 r = 300 + 3 r  ->  r = 14 / 352 s = 39.77 ms
    assert L.crossover_rtt(A, slow_shallow) == pytest.approx(14 / 352 * 1000)
    assert L.crossover_rtt(A, B) is None  # B already faster at RTT 0
    assert L.crossover_rtt(A, L.replace(A, prove_s=1.0)) is None  # same depth: never crosses


def test_bandwidth_for_compute_bound():
    c = L.CATALOG["C QuickSilver GPU model"]
    need = L.bandwidth_for_compute_bound(c)
    assert need > 1000  # Tb/s class: bandwidth-bound at every link speed of the report
    assert L.bandwidth_for_compute_bound(A) == pytest.approx(70e6 * 8 / 260 / 1e9)


def test_sweep_shape_and_report_runs():
    rows = L.sweep([A, B], (0.0, 50.0), (10.0, 100.0))
    assert len(rows) == 8
    md, doc = L.report([A, B], measured=[{"run_id": "r", "design": "A", "B": 4096, "rtt_ms": 10, "depth": 355, "t_total": 260, "verify": 26,
                                          "sleep": 3.55, "e2e_measured": 289.6, "e2e_model": 289.55}])
    assert "## 1. Serial end-to-end latency" in md and "## 5. Measured vs model" in md
    assert doc["crossovers"][0]["deeper"] == "A"
    assert len(doc["sweep"]) == 2 * 5 * 3


def test_from_envelope():
    env = {"run_id": "r1", "workload_fingerprint": {"B": 64, "K": 1536, "proof_class": "NON_ZK_PROOF_DIAGNOSTIC", "hardware": {"gpu": None, "cpu": {"model": "EPYC"}}},
           "measurements": [{"name": "t.total", "value": 6.0, "unit": "s"}, {"name": "verifier.seconds", "value": 0.5, "unit": "s"},
                            {"name": "rounds.sequential_depth", "value": 234, "unit": "count"}, {"name": "rounds.protocol", "value": 1974, "unit": "count"},
                            {"name": "transcript.bytes", "value": 1_638_920, "unit": "B"}, {"name": "mem.peak_host_bytes", "value": 587e6, "unit": "B"}]}
    d = L.from_envelope("A64", env, mem_capacity_bytes=256e9)
    assert (d.prove_s, d.verify_s, d.depth, d.B, d.hardware) == (6.0, 0.5, 234, 64, "EPYC")
    assert d.bytes_vp == 32 * 1974
    assert L.e2e(d, 50.0)["total"] == pytest.approx(6.5 + 234 * 0.05)
    with pytest.raises(ValueError):
        L.from_envelope("x", {"measurements": []}, mem_capacity_bytes=1)


def test_catalog_entries_are_consistent():
    for d in L.CATALOG.values():
        assert d.prove_s > 0 and d.verify_s >= 0 and d.depth >= 1 and d.bytes_pv > 0
        assert math.isfinite(L.pipeline(d, 50.0, 25.0)["vu_per_s"])
    # SP1 is the depth-1 baseline; B-Ligero and C are the shallow interactive designs; A is the deep one
    assert L.CATALOG["SP1 (FS, B=1)"].depth == 1 and not L.CATALOG["SP1 (FS, B=1)"].interactive
    assert L.CATALOG["B-Ligero GPU re-priced (4090)"].depth < 5 < L.CATALOG["A-CPU Goldilocks^3"].depth


def test_depth_chains_match_the_measured_runs():
    # A at B = 64 (r20260922-073131-bfd0): lookup chain 211 (n = 20), checker chain 167 (b = 13, s = 7), depth 214
    assert L.lookup_chain(20) == 211
    assert L.checker_chain(((13, 7),) * 6) == 167
    assert 1 + max(L.lookup_chain(20), L.checker_chain(((13, 7),) * 6)) + 2 == 214
    # B = 4096: n = 26 -> 352; the checker at b = 19 is below it
    assert L.lookup_chain(L.A_R16_LEVELS_B4096) == 352
    assert L.checker_chain(L.A_CHECKER_LAYERS_B4096) == 203
    # the levers: sharding removes levels, c halves rounds, cuts take the longest piece
    assert L.lookup_chain(21) == 232
    assert L.lookup_chain(26, 2) == 196
    assert L.lookup_chain(26, 1, (16, 21)) == 137
    assert L.checker_chain(L.A_CHECKER_LAYERS_B4096, 1, (2,)) == 4 * 33 + 3


def test_a_depth_options_rows():
    rows = L.a_depth_options(26, L.A_CHECKER_LAYERS_B4096, L.A_COMMITTED_B4096, L.CATALOG["A-CPU Goldilocks^3"], L.CATALOG["A-GPU re-priced (4090, fused SIMT)"])
    by = {r["option"].split(" [")[0].split(" (")[0]: r for r in rows}
    assert by["baseline"]["depth"] == 355
    assert by["R16 sharding s = 5"]["depth"] == 235 and by["R16 sharding s = 5"]["committed_pct"] == pytest.approx(1.10, abs=0.01)
    assert by["c = 2 variables per message"]["depth"] == 199 and by["c = 2 variables per message"]["prover_work_x"] > 1.5
    assert next(r for r in rows if r["option"].startswith("cut the tree at [16, 21] + commit"))["depth"] == 140
    assert all(r["depth"] >= 355 for r in rows if r["option"].startswith("checkpoints")) and all("never" in r["wins_above_rtt_ms_gpu"] for r in rows if r["option"].startswith("checkpoints"))
    assert by["hint-free branch"]["depth"] > 355 and by["hint-free branch"]["prover_work_x"] < 1


def test_measured_row():
    env = {"run_id": "r", "workload_fingerprint": {"B": 64, "latency": {"rtt_ms": 10}, "software": {"backend": {"name": "verity-gkr"}}},
           "measurements": [{"name": "t.total", "value": 6.0}, {"name": "verifier.seconds", "value": 0.5}, {"name": "rounds.sequential_depth", "value": 214},
                            {"name": "t.e2e.rtt_10ms", "value": 8.65}]}
    row = L.measured_row(env)
    assert row["e2e_model"] == pytest.approx(6.5 + 2.14) and row["sleep"] == pytest.approx(2.15) and row["depth"] == 214
