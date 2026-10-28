"""native_peak: the declaration and the torch-free helpers run anywhere; the measurement itself only where torch + CUDA exist."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "research" / "src"))

from research.store import derivation as D  # noqa: E402
from research.store import tools_registry  # noqa: E402
from research import result as R  # noqa: E402


def _measure():
    spec = importlib.util.spec_from_file_location("native_peak_measure", REPO / "benchmarks" / "native_peak" / "measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declaration_is_registered_and_keys_what_is_measured():
    t = tools_registry.load("native_peak", REPO)
    assert t.name == "native_peak" and t.spec == "benchmarks.native_peak.tool:NATIVE_PEAK" and "hw.family" in t.key_conditions
    assert "native-peak/v1" in t.produces
    argv = ["/workspace/venv312/bin/python", "benchmarks/native_peak/measure.py", "--dtype", "bf16", "--sizes", "16384,8192", "--out", "/run/dir", "--device", "cuda:0"]
    params, key = t.resolve_params(argv, REPO)
    assert key == {"dtype": "bf16", "sizes": "8192,16384", "warmup": 3, "reps": 5, "sustain_s": 2.0}
    assert params["out"] == "/run/dir" and "out" not in key and "device" not in key
    _, key2 = t.resolve_params(argv[:3] + ["e4m3"], REPO)
    conds = {"hw.family": "rtx4090"}
    assert D.derivation_id(t, {}, key, conds, t.key_conditions) != D.derivation_id(t, {}, key2, conds, t.key_conditions)
    assert D.derivation_id(t, {}, key, conds, t.key_conditions) != D.derivation_id(t, {}, key, {"hw.family": "a100"}, t.key_conditions)


def test_spec_table_and_size_parsing():
    m = _measure()
    assert m.spec_peak("NVIDIA GeForce RTX 4090", "bf16") == (165.2e12, m.SPEC_PEAKS[0][2])
    assert m.spec_peak("NVIDIA GeForce RTX 4090", "e4m3")[0] == 330.3e12
    assert m.spec_peak("NVIDIA A100-SXM4-80GB", "bf16")[0] == 312.0e12
    assert m.spec_peak("NVIDIA A100-SXM4-80GB", "e4m3") == (None, m.SPEC_PEAKS[1][2])
    assert m.spec_peak("NVIDIA A100 80GB PCIe", "bf16")[0] == 312.0e12
    assert m.spec_peak("NVIDIA H100 80GB HBM3", "bf16")[0] == 989.4e12
    assert m.spec_peak("NVIDIA H100 80GB HBM3", "e4m3")[0] == 1978.9e12
    assert m.spec_peak("NVIDIA H100 80GB HBM3", "nvfp4")[0] is None
    assert m.spec_peak("NVIDIA H100 PCIe", "bf16")[0] == 756.0e12
    assert m.spec_peak("NVIDIA A10G", "bf16") == (None, None)
    # RTX 5090: the whitepaper's dense FP4-with-FP32-accumulate figure, not the sparse headline 3352
    assert m.spec_peak("NVIDIA GeForce RTX 5090", "nvfp4")[0] == 1676.0e12 and m.spec_peak("NVIDIA GeForce RTX 5090", "e4m3")[0] == 419.0e12
    assert m.spec_peak("NVIDIA GeForce RTX 4090", "nvfp4")[0] is None and "nvfp4" in m.DTYPES
    assert m.parse_sizes("8192, 16384") == [8192, 16384]
    with pytest.raises(ValueError):
        m.parse_sizes("100")


def test_smi_samples_parse_and_summarise():
    m = _measure()
    rows = m.parse_smi_csv("2026/09/22 21:00:00.100, 2520, 10501, 412.3, 61, 0x0000000000000004\n2026/09/22 21:00:00.200, 2505, 10501, 449.9, 62, 0x0000000000000004\nbad line\n")
    assert len(rows) == 2 and rows[0]["clocks_sm_mhz"] == 2520 and rows[1]["power_w"] == 449.9 and rows[0]["t"] is not None and rows[1]["t"] > rows[0]["t"]
    s = m.summarize_samples(rows)
    assert s["n_samples"] == 2 and s["clocks_sm_mhz"]["median"] == 2512.5 and s["power_w"]["max"] == 449.9 and s["throttle_reasons"] == ["0x0000000000000004"]
    assert m.summarize_samples([])["clocks_sm_mhz"] is None


def test_result_document_is_a_valid_envelope_with_headline_fields(tmp_path):
    m = _measure()
    envd = {"device": "NVIDIA GeForce RTX 4090", "sm": "8.9", "driver": "570.1", "cuda": "12.4", "torch": "2.4.1+cu124"}
    size = {"n": 8192, "m": 8192, "k": 8192, "flop": 2.0 * 8192 ** 3, "warmup": 3, "reps": 5, "seconds_all": [0.0080, 0.0079, 0.0081, 0.0080, 0.0082],
            "seconds_median": 0.0080, "seconds_min": 0.0079, "seconds_max": 0.0082, "flop_per_second": 2.0 * 8192 ** 3 / 0.0080,
            "sustain": {"seconds": 2.0, "iterations": 240, "flop_per_second": 2.0 * 8192 ** 3 * 240 / 2.0}, "smi_window": {"t0": 0, "t1": 1},
            "check": {"max_abs_err": 0.5, "ref_abs_max": 400.0, "max_rel_err": 1.25e-3, "tolerance": 2e-2, "ok": True, "block": 256},
            "kernels": ["sm89_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32"], "peak_device_bytes": 1 << 30,
            "smi": {"n_samples": 20, "clocks_sm_mhz": {"median": 2520.0, "min": 2505.0, "max": 2520.0, "n": 20}, "clocks_mem_mhz": None,
                    "power_w": {"median": 430.0, "min": 400.0, "max": 449.0, "n": 20}, "temperature_c": None, "throttle_reasons": []}}
    (tmp_path / "native_peak.json").write_text("{}")
    doc = m.build_result("r20260922-000000-abcd", "bf16", envd, [size], {"sass": "not captured", "error": "ncu not on PATH"}, {"n_samples": 20}, None, 12.0,
                         ["native_peak.json"])
    chk = R.check_doc(tmp_path, doc, "r20260922-000000-abcd")
    assert chk.valid, chk.errors
    assert doc["kind"] == "native-peak/v1" and doc["device"] == envd["device"] and doc["dtype"] == "bf16" and doc["accumulate"] == "fp32"
    assert doc["spec_flop_per_second"] == 165.2e12 and abs(doc["fraction_of_spec"] - doc["flop_per_second"] / 165.2e12) < 1e-12
    assert doc["sass"] == "not captured" and doc["clocks_mhz"] == 2520.0 and doc["power_w"] == 430.0 and "8192^3" in doc["sizes"]
    assert doc["method"].startswith("cublas dense gemm 8192^3, median of 5")
    assert doc["validation"]["status"] == "passed" and {x["name"] for x in doc["measurements"]} >= {"flop_per_second", "fraction_of_spec", "gemm_8192_seconds"}
    json.dumps(doc)   # serialisable as written
    skipped = m.build_result("r20260922-000000-abcd", "e4m3", {"device": "NVIDIA A100-SXM4-80GB"}, [], {"sass": "not captured"}, {"n_samples": 0},
                             "FP8 tensor cores need sm_89+", 1.0, [])
    assert R.check_doc(tmp_path, skipped, "r20260922-000000-abcd").valid
    assert skipped["validation"]["status"] == "not_run" and skipped["flop_per_second"] is None and skipped["spec_flop_per_second"] is None


def test_nvfp4_instruction_result_document(tmp_path):
    """--dtype nvfp4 records the instruction microbenchmark as the headline (label instead of n^3), the SASS from cuobjdump,
    and the cuBLASLt GEMM attempt beside it; the envelope stays valid."""
    m = _measure()
    envd = {"device": "NVIDIA GeForce RTX 5090", "sm": "12.0", "driver": "570.195.03", "cuda": "12.8", "torch": "2.8.0+cu128"}
    grid = {"sms": 170, "blocks": 680, "warps_per_block": 4, "warps": 2720, "mmas_per_iteration": 32, "iterations": 100000, "accumulator_chains": 8,
            "operand_sets": 4}
    flop = 2720.0 * 32 * 100000 * m.NVFP4_MMA_FLOP
    sass = {"sass": "OMMA.16864.F32.E2M1.E2M1.UE4M3", "opcodes": ["OMMA.16864.F32.E2M1.E2M1.UE4M3"], "mma_count": 32, "per_opcode": {}, "error": None}
    rec = {"n": None, "m": 16, "k": 64, "label": "mma_m16n8k64_x100000", "instruction": "mma.sync.aligned.kind::mxf4nvf4...ue4m3", "grid": grid,
           "cc": "12.0", "nvcc": "12.8", "so": "x.so", "flop": flop, "warmup": 3, "reps": 5, "seconds_all": [1.0] * 5, "seconds_median": 1.0,
           "seconds_min": 1.0, "seconds_max": 1.0, "flop_per_second": flop, "sustain": {"seconds": 2.0, "iterations": 2, "flop_per_second": flop},
           "smi_window": {"t0": 0, "t1": 1},
           "check": {"ok": True, "finite": True, "deterministic": True, "sass_mma_count": 32, "expected_mma_count": 32, "sass_body_copies": 1, "opcodes": sass["opcodes"],
                     "tolerance": 0.0, "max_rel_err": 0.0}, "kernels": ["nvfp4_peak_kernel"], "sass": sass, "peak_device_bytes": 1 << 20,
           "smi": {"n_samples": 20, "clocks_sm_mhz": {"median": 2400.0, "min": 2300.0, "max": 2400.0, "n": 20}, "clocks_mem_mhz": None,
                   "power_w": {"median": 500.0, "min": 400.0, "max": 575.0, "n": 20}, "temperature_c": None, "throttle_reasons": []}}
    gemm = {"attempted": True, "n": 8192, "supported": False, "reason": "torch has no float4_e2m1fn_x2", "flop_per_second": None}
    (tmp_path / "native_peak.json").write_text("{}")
    doc = m.build_result("r20260923-000000-abcd", "nvfp4", envd, [rec], sass, {"n_samples": 20}, None, 30.0, ["native_peak.json"], gemm)
    chk = R.check_doc(tmp_path, doc, "r20260923-000000-abcd")
    assert chk.valid, chk.errors
    assert doc["dtype"] == "nvfp4" and doc["spec_flop_per_second"] == 1676.0e12 and doc["flop_per_second"] == flop
    assert abs(doc["fraction_of_spec"] - flop / 1676.0e12) < 1e-12 and doc["sass"] == sass["sass"] and doc["gemm_attempt"] is gemm
    assert doc["method"].startswith("instruction microbenchmark: mma.sync") and "mma_m16n8k64_x100000" in doc["sizes"]
    assert doc["validation"]["status"] == "passed" and "sass 32/32" in doc["validation"]["detail"]
    assert {x["name"] for x in doc["measurements"]} >= {"flop_per_second", "fraction_of_spec", "mma_m16n8k64_x100000_seconds"}
    json.dumps(doc)


def test_measure_bf16_small_on_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m = _measure()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260922-000000-cuda")
    rc = m.main(["--dtype", "bf16", "--sizes", "1024", "--warmup", "1", "--reps", "3", "--sustain-s", "0.2", "--out", str(tmp_path), "--no-sass"])
    assert rc == 0
    doc = json.loads((tmp_path / "result.json").read_text())
    assert R.check_doc(tmp_path, doc, "r20260922-000000-cuda").valid and doc["validation"]["status"] == "passed"
    assert doc["flop_per_second"] > 0 and "1024^3" in doc["sizes"] and doc["sizes"]["1024^3"]["check_ok"]


def test_spec_peaks_agree_with_every_census_line():
    """The pod-side table stays standalone (nothing to read on a pod); it must quote benchmarks/census/hardware.json's peak for every line."""
    m = _measure()
    hardware = json.loads((REPO / "benchmarks" / "census" / "hardware.json").read_text(encoding="utf-8"))
    smi = {"NVIDIA A100 SXM4 80GB": "NVIDIA A100-SXM4-80GB", "NVIDIA H100 SXM5 80GB": "NVIDIA H100 80GB HBM3", "NVIDIA L40S 48GB": "NVIDIA L40S"}
    dtype = {"bf16": "bf16", "e4m3": "e4m3", "e2m1": "nvfp4"}
    for h in hardware:
        assert m.spec_peak(smi.get(h["device"], h["device"]), dtype[h["dtype"]])[0] == h["flop_per_second"], h["id"]
