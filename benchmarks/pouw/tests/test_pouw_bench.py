"""CPU tests for `pouw_bench`: its shapes, W1's γ, Ω and W_ref/mkn against the scheme's certificate, samples per bit and the
verifier's cost per draw, its variants and their builds' occupancy, its timing rules (operand sets past the L2, a weight id
per set, the arms' rotation), and the metrics and consistency gate it derives from a shape's timings."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_beside_omega_is_the_slowdown_w1_charges_against_the_plain_product():
    for m, k, n in ((16, 4864, 1600), (8192, 8192, 8192)):
        got = B.w1(m, k, n)
        assert got["w1_x_mkn"] == pytest.approx(3 * got["omega"]) and 3 < got["w1_x_mkn"] < 3.02


def test_a_calls_hashed_bytes_are_its_tile_words_three_quarters_of_a_byte_per_mac_and_the_rest():
    """The tile words are 3/4 of a byte per MAC; the dequantization leaves (a block per output, 64/k per MAC) and the units
    tree (a node per unit, about 128/k) add the rest."""
    m, k, n = 2048, 4096, 4096
    got = B.hashed_bytes_per_mac(m, k, n)
    assert 0.75 + 64 / k + 128 / k < got < 0.75 + 64 / k + 128 / k + 0.01


def test_samples_per_bit():
    for eps in B.EPSILONS:
        t = B.samples_per_bit(eps)
        assert math.isclose((1 - eps) ** t, 0.5)
    assert round(B.samples_per_bit(0.01), 2) == 68.97
    row = [r for r in B.draws() if r["eps_s"] == 0.01][0]
    assert row["draws_for_128_bits"] == 8828


def test_samples_per_bit_come_with_the_verifiers_cost_per_draw_and_are_labelled_ours():
    shapes = [{"m": 16, "k": 896, "n": 896}, {"m": 2048, "k": 896, "n": 896}]
    s = B.sampling(shapes)
    assert s["definition"].startswith("ours") and len(s["verifier_draws"]) == 1 and len(s["verifier_draws"][0]["raw_s"]) == 3
    small, big = s["shapes"]
    assert small["tile_s"] == big["tile_s"] and big["opened_bytes"] > small["opened_bytes"] > 2 * 16 * 896
    at = {p["eps_s"]: p for p in small["per_bit"]}[0.01]
    assert at["verifier_s"] == pytest.approx(B.samples_per_bit(0.01) * small["tile_s"])
    assert at["opened_bytes"] == pytest.approx(B.samples_per_bit(0.01) * small["opened_bytes"])


def test_a_tiles_opening_is_its_rows_their_scales_and_ids_its_leaf_and_its_path():
    population = 3 + 1 + 256                                    # one strip, one column strip: 16 x 16 outputs
    assert B.opened_bytes(16, 896, 16) == 2 * 16 * 896 + 4 * 16 * 4 + 32 + 32 * math.ceil(math.log2(population))


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


ROW = {"m": 2048, "k": 896, "n": 9728, "int8_ms": 0.5, "int8_tuned_ms": 0.4, "serve_ms": 0.6, "serve_tuned_ms": 0.5,
       "fp8_ms": 1.0, "fp8_tuned_ms": 1.1, "bf16_ms": 2.0, "nohash_ms": 5.0, "sync_ms": 45.0, "sync_tile_ms": 40.0,
       "async_ms": 30.0}


def test_the_headline_divides_by_the_fastest_served_plain_path_each_at_its_better_call():
    d = B.derive(ROW)
    assert (d["int8_best"], d["int8_best_ms"], d["fp8_best"], d["fp8_best_ms"]) == ("int8_tuned", 0.4, "fp8", 1.0)
    assert (d["plain_path"], d["plain_ms"]) == ("serve", 0.5)          # int8 alone lacks the dequantization
    assert (d["prover_x_plain"], d["arithmetic_x_plain"], d["hashing_x_plain"], d["async_x_plain"]) == (90.0, 10.0, 80.0, 60.0)
    assert (d["prover_x_int8"], d["prover_x_fp8"]) == (112.5, 45.0)
    assert d["hashing_ms"] == 40.0 and d["hashing_share"] == pytest.approx(8 / 9) and d["tile_word_rate_bytes_per_s"] > 0
    assert B.derive({k: v for k, v in ROW.items() if k != "serve_tuned_ms"})["plain_path"] == "serve"
    assert B.derive(ROW | {"fp8_ms": 0.3})["plain_path"] == "fp8"


def test_the_consistency_gate_fails_a_nohash_call_under_w1s_plain_products():
    row = ROW | B.w1(2048, 896, 9728) | B.derive(ROW)
    ok = B.consistency(row)
    assert ok["ok"] and ok["floor_ms"] == pytest.approx(row["w1_x_mkn"] * 0.4)
    assert not B.consistency(row | {"nohash_ms": 1.0})["ok"]              # under 3 plain int8 GEMMs: work skipped
    assert B.consistency(row | {"m": 16}) is None                          # not compute-bound: no gate


def test_a_shapes_operand_sets_pass_twice_the_l2_and_each_has_its_own_weight_id():
    l2 = 72 << 20
    for m, k, n in [(16, 896, 896), (2048, 896, 9728), (8192, 8192, 8192), (2048, 4096, 28672)]:
        sets = B.set_count(m, k, n, l2)
        assert 2 <= sets <= B.MAX_SETS and (sets == B.MAX_SETS or sets * (m + n) * k >= 2 * l2)
    assert B.set_count(16, 896, 896, l2) * (16 + 896) * 896 > l2         # capped, and still past the L2
    ids = [B.weight_id(s, i) for s in range(40) for i in range(B.MAX_SETS)]
    assert len(set(ids)) == len(ids) and min(ids) >= 1 << 32              # nor any gate call's small ids


def test_each_arm_runs_once_in_each_place_over_a_rotation():
    names = ["int8", "serve", "fp8", "bf16", "nohash", "sync", "async"]
    orders = [B.rotation(names, r) for r in range(len(names))]
    assert all(sorted(o) == sorted(names) for o in orders)
    assert all(sorted(o[i] for o in orders) == sorted(names) for i in range(len(names)))


def test_the_sensors_find_the_device_by_its_uuid_not_an_index():
    """CUDA's device order and NVML's can differ on a multi-GPU box: the UUID names the one the bench runs on."""
    for raw in ("GPU-5b3d8f2a-0000-1111-2222-333344445555", "5b3d8f2a-0000-1111-2222-333344445555"):
        props = SimpleNamespace(uuid=raw)
        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 3, get_device_properties=lambda i: props))
        assert B.device_uuid(torch) == "GPU-5b3d8f2a-0000-1111-2222-333344445555"


def test_every_prover_arm_is_timed_at_the_flags_its_gate_checks():
    assert B.ARM_FLAGS == {"nohash": B.PD.TIME_TILES, "sync": B.PD.HASH | B.PD.TIME_TILES | B.PD.CACHE_Y,
                           "async": B.PD.HASH | B.PD.CACHE_Y}
    assert not any(f & B.PD.LOWBYTE for f in B.ARM_FLAGS.values())


def test_forwards_are_synthetic_sums_of_the_baseline_linears_over_24_layers():
    rows = [{"variant": "v", "set": "baseline", "m": 16, "k": k, "n": n, "plain_ms": 1.0, "sync_ms": 10.0}
            for k, n in SPEC["sets"]["baseline"]["shapes"]]
    f = B.forwards(rows, SPEC)["v"]["16"]
    assert (f["calls"], f["plain_ms"], f["sync_ms"], f["prover_x_plain"], f["synthetic"]) == (4, 96.0, 960.0, 10.0, True)


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
    assert res["dry_run"] and res["certificate"]["gamma0"] == "1/200" and len(res["sampling"]["draws"]) == len(B.EPSILONS)
    assert len(res["sampling"]["shapes"]) == 16 and res["schema"] == "verity-pouw/pouw-bench/v2"
    assert len(res["shapes"]) == 16 and all(s["gamma"] is None for s in res["shapes"])     # k = 896 or n = 896: outside
    v = res["variants"]["rows=0,sha=0"]
    assert v["resource_usage"] is None and {g["difference"] for g in v["twin_gate"]} == {None}
    assert "timings" not in res
