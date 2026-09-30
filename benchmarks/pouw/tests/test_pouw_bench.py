"""CPU tests for `pouw_bench`: its shapes, W1's γ and Ω against the scheme's certificate, samples per bit, its variants and
their builds' occupancy, and the metrics it derives from a shape's timings."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_bench", Path(__file__).parents[1] / "pouw_bench.py")
B = importlib.util.module_from_spec(spec)
sys.modules["pouw_bench"] = B
spec.loader.exec_module(B)

SPEC = json.loads(B.SHAPES.read_text())
QWEN15 = "vllm/qwen25-15b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager"


@pytest.fixture(scope="module")
def workloads():
    return json.loads(B.CENSUS.read_text())


def test_census_shapes_are_the_workloads_gemms_within_the_kernels_n(workloads):
    assert B.census_gemms(workloads, QWEN15) == [(1536, 151936), (1536, 1536), (1536, 17920), (8960, 1536)]
    census = [s for s in B.bench_shapes(["census"], SPEC, workloads) if s["source"] == QWEN15]
    assert {(s["k"], s["n"]) for s in census} == {(1536, 1536), (1536, 17920), (8960, 1536)}      # the lm_head's n is over 2^16
    assert {s["m"] for s in census} == {16, 2048}
    with pytest.raises(KeyError, match="no census workload"):
        B.census_gemms(workloads, "vllm/no-such-row")


def test_the_baseline_is_389_and_435s_qwen05_linears_at_their_rows():
    shapes = B.bench_shapes(["baseline"], SPEC, None)
    assert len(shapes) == 16 and {(s["k"], s["n"]) for s in shapes} == {(896, 1152), (896, 896), (896, 9728), (4864, 896)}
    assert sorted({s["m"] for s in shapes}) == [16, 32, 256, 2048]
    assert SPEC["figures"]["#435"]["prefill_forward_2048_rows_s"] == 2.04


def test_only_shapes_outside_the_baseline_get_their_own_gate_call(workloads):
    shapes = B.bench_shapes(["baseline", "square"], SPEC, workloads)
    assert B.gate_shapes(shapes, SPEC) == [(4096, 4096), (8192, 8192)]


def test_w1_is_the_certificates_at_the_domains_edge_and_absent_outside_it():
    cert = B.certificate()
    assert (cert["gamma0"], cert["omega_star"]) == ("1/200", "201/200")
    edge = B.w1(16, 4864, 1600)
    assert edge["omega"] == 1.005 and edge["gamma"] == pytest.approx(cert["gamma"])
    assert round(B.w1(4096, 4096, 4096)["gamma"] * 100, 3) == 0.694
    outside = B.w1(16, 896, 1152)
    assert outside["gamma"] is None and "outside" in outside["outside"]


def test_a_calls_hashed_bytes_are_about_three_quarters_of_a_byte_per_mac():
    assert 0.75 < B.hashed_bytes_per_mac(2048, 4096, 4096) < 0.76


def test_samples_per_bit():
    for eps in B.EPSILONS:
        t = B.samples_per_bit(eps)
        assert math.isclose((1 - eps) ** t, 0.5)
    assert round(B.samples_per_bit(0.01), 2) == 68.97
    row = [r for r in B.sampling() if r["eps_s"] == 0.01][0]
    assert row["draws_for_128_bits"] == 8828


def test_variants_parse_as_ncp2_gpu_bench_takes_them():
    assert B.parse_variants(B.VARIANTS) == [
        ("rows=0,sha=0", ("-DNCP2_TILE_REG_ROWS=0", "-DNCP2_TILE_SHA=0")),
        ("rows=8,unroll=2,sha=1", ("-DNCP2_TILE_REG_ROWS=8", "-DNCP2_TILE_SHA=1", "-DNCP2_TILE_REG_UNROLL=2"))]
    s2 = "rows=8,unroll=2,sha=1;rows=8,unroll=1,sha=1;rows=8,unroll=8,sha=1;rows=8,unroll=2,sha=0;rows=0"
    assert [n for n, _f in B.parse_variants(s2)] == s2.split(";")
    assert len(B.parse_variants("rows=0;rows=0")) == 1
    with pytest.raises(KeyError):
        B.parse_variants("regs=8")


def test_change_3_doubles_the_resident_chains():
    before, after = B.tile_block(("-DNCP2_TILE_REG_ROWS=0",)), B.tile_block(())
    assert (before["threads"], before["smem"], after["threads"], after["smem"]) == (32, 32768, 64, 32768)
    assert B.occupancy(128, 32, 32768) == {"blocks": 3, "warps": 3, "chains": 96, "limited_by": ["shared memory"]}
    assert B.occupancy(255, 64, 32768) == {"blocks": 3, "warps": 6, "chains": 192, "limited_by": ["shared memory"]}


PTXAS = """\
ptxas info    : Compiling entry function '_ZN4ncp28kf_tilesILi1EEEvimiiiPKhS2_PKjPhPj' for 'sm_89'
ptxas info    : Function properties for _ZN4ncp28kf_tilesILi1EEEvimiiiPKhS2_PKjPhPj
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 255 registers, used 0 barriers, 424 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function '_Z11kf_key_leafPKhiPh' for 'sm_89'
ptxas info    : Function properties for _Z11kf_key_leafPKhiPh
    144 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 0 barriers, 376 bytes cmem[0]
"""


def test_the_resource_usage_is_read_for_the_tile_kernels_only():
    assert B.ptxas_usage(PTXAS) == {"kf_tiles<1>": {"stack": 0, "spill_stores": 0, "spill_loads": 0, "registers": 255}}


def test_the_derived_metrics_report_hashing_apart_from_the_arithmetic():
    row = {"m": 2048, "k": 896, "n": 9728, "int8_ms": 0.5, "fp8_ms": 1.0, "bf16_ms": 2.0, "nohash_ms": 5.0, "sync_ms": 45.0,
           "sync_tile_ms": 40.0, "async_ms_per_call": 30.0}
    d = B.derive(row)
    assert (d["hashing_ms"], d["prover_x_int8"], d["prover_x_fp8"], d["arithmetic_x_int8"], d["hashing_x_int8"]) == \
        (40.0, 90.0, 45.0, 10.0, 80.0)
    assert d["hashing_share"] == pytest.approx(8 / 9) and d["async_x_int8"] == 60.0
    assert d["tile_word_rate_bytes_per_s"] > 0


def test_forwards_sum_the_baseline_linears_over_24_layers():
    rows = [{"variant": "v", "set": "baseline", "m": 16, "k": k, "n": n, "int8_ms": 1.0, "sync_ms": 10.0}
            for k, n in SPEC["sets"]["baseline"]["shapes"]]
    f = B.forwards(rows, SPEC)["v"]["16"]
    assert (f["calls"], f["int8_ms"], f["sync_ms"], f["prover_x_int8"]) == (4, 96.0, 960.0, 10.0)


@pytest.mark.slow
def test_the_twin_gate_holds_each_default_variant_to_pouw_native():
    """Each variant's flags reach the host build (a build per variant), and its twin commits `pouw_native`'s leaves."""
    calls = [c for c in B.PD.gate_calls() if c[0] == "five-strips"]
    refs = {name: B.PN.commit_call(0, B.SCHEME, call) for name, call in calls}
    digests = set()
    try:
        for _name, flags in B.parse_variants(B.VARIANTS):
            B.use(flags)
            digests.add(B.PD.source_digest(flags=B.PD.CFLAGS))
            assert [g["difference"] for g in B.twin_gate(calls, refs)] == [None]
    finally:
        B.use(())
    assert len(digests) == 2


@pytest.mark.slow
def test_the_dry_run_writes_its_cpu_results(tmp_path, monkeypatch):
    out = tmp_path / "dry.json"
    monkeypatch.setattr(sys, "argv", ["pouw_bench.py", "--dry-run", "--sets", "baseline", "--tile-variants", "rows=0,sha=0",
                                      "--cpu-rows", "0", "--out", str(out)])
    monkeypatch.setattr(B.PD, "nvcc", lambda: None)
    B.main()
    res = json.loads(out.read_text())
    assert res["dry_run"] and res["certificate"]["gamma0"] == "1/200" and len(res["sampling"]) == len(B.EPSILONS)
    assert len(res["shapes"]) == 16 and all(s["gamma"] is None for s in res["shapes"])     # k = 896 or n = 896: outside
    v = res["variants"]["rows=0,sha=0"]
    assert v["resource_usage"] is None and {g["difference"] for g in v["twin_gate"]} == {None}
    assert "timings" not in res
