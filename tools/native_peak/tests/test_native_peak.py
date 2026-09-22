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
    spec = importlib.util.spec_from_file_location("native_peak_measure", REPO / "tools" / "native_peak" / "measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declaration_is_registered_and_keys_what_is_measured():
    t = tools_registry.load("native_peak", REPO)
    assert t.name == "native_peak" and t.spec == "tools.native_peak.tool:NATIVE_PEAK" and "hw.family" in t.key_conditions
    assert "native-peak/v1" in t.produces
    argv = ["/workspace/venv312/bin/python", "tools/native_peak/measure.py", "--dtype", "bf16", "--sizes", "16384,8192", "--out", "/run/dir", "--device", "cuda:0"]
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
    assert m.spec_peak("NVIDIA H100 80GB HBM3", "bf16") == (None, None)
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
