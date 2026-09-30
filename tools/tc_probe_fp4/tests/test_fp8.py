"""``tools/tc_probe_fp4/fp8.py``, the sm_120 FP8 rows, on fake silicon (the row's primary model standing in for the device).

Every row passes its gates and reproduces its primary; one flipped bit in the capture fails the run; the candidate grid still
refutes every neighbour (so a capture that matched one of them would be caught); the row arithmetic equals the core batch kernel on
the registered sm_120 models; the cast and add references equal the core cast and pinned words.  With nvcc on the PATH, the SASS
gate is checked on a real build.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fp8  # noqa: E402
from verity.ml.tc import models as TCM  # noqa: E402
from verity.ml.tc.cast import f32_to_e4m3_sat_word  # noqa: E402

ROWS = {
    "step_e4m3": ["--row", "step", "--instruction", "f8_e4m3", "--n-random", "32"],
    "step_e4m3e5m2": ["--row", "step", "--instruction", "f8_e4m3e5m2", "--n-random", "32"],
    "step_mxf8": ["--row", "step", "--instruction", "mxf8_e4m3", "--n-random", "32"],
    "chain": ["--row", "chain", "--steps", "32", "--chain-tiles", "2"],
    "mixed": ["--row", "mixed", "--chain-tiles", "1"],
    "fadd": ["--row", "fadd"],
}


@pytest.fixture
def no_nvcc(monkeypatch):
    monkeypatch.setattr(fp8.P4, "nvcc_bin", lambda: "/nonexistent/nvcc")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)


def _run(tmp_path, argv):
    args = fp8.parser().parse_args([*argv, "--out", str(tmp_path), "--fake-silicon"])
    return fp8.run(args)


SLOW_ROWS = {"step_e4m3", "step_mxf8"}  # 8-13 s each on fake silicon


@pytest.mark.parametrize("row", [pytest.param(r, marks=pytest.mark.slow) if r in SLOW_ROWS else r for r in sorted(ROWS)])
def test_fake_row_passes_and_corruption_fails(tmp_path, no_nvcc, row):
    code, res = _run(tmp_path / "ok", ROWS[row])
    assert code == 0 and res["validation"] == "passed", res["failures"]
    cap = res["capture"]
    assert all(cap["agrees"].values()) if row == "fadd" else cap["total_mismatches"][cap["primary"]] == 0
    assert json.loads((tmp_path / "ok" / "probe_results.json").read_text())["validation"] == "passed"
    assert not (tmp_path / "ok" / "result.json").exists()
    code, res = _run(tmp_path / "bad", [*ROWS[row], "--corrupt-word", "5"])
    assert code == 1 and res["validation"] == "failed"
    assert any(f.startswith("primary") for f in res["failures"]), res["failures"]


@pytest.mark.parametrize("instr", ["f8_e4m3e5m2", pytest.param("mxf8_e4m3", marks=pytest.mark.slow)])
def test_candidate_grid_refutes_every_neighbour(tmp_path, no_nvcc, instr):
    code, res = _run(tmp_path, ["--row", "step", "--instruction", instr, "--n-random", "32"])
    assert code == 0
    totals = res["capture"]["total_mismatches"]
    primary = res["capture"]["primary"]
    refuted = {c: n for c, n in totals.items() if c != primary}
    assert refuted and all(n > 0 for n in refuted.values()), refuted


def test_row_arithmetic_equals_core_kernel():
    rng = np.random.default_rng(3)
    for instr, core in (("f8_e4m3", TCM.BLACKWELL_SM120_E4M3_M16N8K32), ("f8_e5m2", TCM.BLACKWELL_SM120_E5M2_M16N8K32)):
        for name, A, Bt, C, SFA, SFB in fp8.step_families(rng, instr, 16):
            want = fp8.step_model(core, A, Bt, C)
            got = fp8.evaluate(instr, fp8.PRIMARY[instr][1], A, Bt, C, SFA, SFB)
            assert np.array_equal(got, want), (instr, name)


def test_cvt_ref_equals_core_cast():
    rng = np.random.default_rng(4)
    X = np.concatenate([rng.integers(0, 1 << 32, 4000, dtype=np.uint64).astype(np.uint32),
                        np.array([0, 0x80000000, 0x43E00000, 0x43E80000, 0x7F800000, 0xFF800000, 0x3B000000, 0x3B800000], np.uint32)])
    X = X[(X & 0x7FFFFFFF) <= 0x7F800000]
    want = np.array([f32_to_e4m3_sat_word(int(x)) for x in X], np.uint8)
    assert np.array_equal(fp8.cvt_ref("e4m3", X), want)


def test_cvt_ref_pinned_words():
    x = np.array([0x3F800000, 0x43E00000, 0x7F800000, 0xC3F00000, 0x7FC00000, 0x80000000], np.uint32)  # 1, 448, +inf, -480, NaN, -0
    assert fp8.cvt_ref("e4m3", x).tolist() == [0x38, 0x7E, 0x7E, 0xFE, 0x7F, 0x80]
    assert fp8.cvt_ref("e5m2", np.array([0x3F800000, 0x7F800000], np.uint32)).tolist() == [0x3C, 0x7B]


def test_fadd_ref_ftz_and_nan():
    sub, neg_sub = np.uint32(1), np.uint32(0x80000001)
    x = np.array([sub, neg_sub, 0x7F800000, 0x3F800000], np.uint32)
    y = np.array([0, 0x80000000, 0xFF800000, 0x3F800000], np.uint32)  # -sub + -0: FTZ keeps the sign; -0 + +0 would be +0
    assert fp8.fadd_ref(x, y, ftz=False).tolist() == [1, 0x80000001, 0x7FFFFFFF, 0x40000000]
    assert fp8.fadd_ref(x, y, ftz=True).tolist() == [0, 0x80000000, 0x7FFFFFFF, 0x40000000]


def test_identity_row_stops_on_a_wrong_uuid(tmp_path, no_nvcc, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    code, res = _run(tmp_path / "ok", ["--row", "identity", "--expect-gpu-uuid", "GPU-fake"])
    assert code == 0 and "capture" not in res and "layout" not in res
    code, res = _run(tmp_path / "bad", ["--row", "identity", "--expect-gpu-uuid", "GPU-other"])
    assert code == 1 and res["failures"] == ["device UUID GPU-fake != --expect-gpu-uuid GPU-other"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    code, res = _run(tmp_path / "two", ["--row", "identity"])
    assert code == 1 and "is not a single device" in res["failures"][0]


def test_identity_gate_bootstrap_checks(monkeypatch):
    class Dev:
        def info(self):
            return {"device": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0", "sms": 170, "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"}

    smi = "NVIDIA GeForce RTX 5090, 570.153.02, 98.02, 00000000:01:00.0, 2000, 14000, 3000, 40, 100.0, 0x0"
    monkeypatch.setattr(fp8.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 0, "stdout": smi})())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    info, fails = fp8.identity_gate(Dev(), None)
    assert fails == ["device 'NVIDIA GeForce RTX 5090' is not an RTX PRO 6000", "170 SMs, not 188", "driver 570.153.02 is older than 575"]
    assert info["driver"] == "570.153.02" and info["uuid_checked"] is False


def test_smi_query_survives_an_unknown_field(monkeypatch):
    def run(argv, **k):
        f = argv[2].split("=", 1)[1]
        ok = "clocks_event_reasons" not in f
        return type("P", (), {"returncode": 0 if ok else 2, "stdout": ", ".join(x.upper() for x in f.split(",")) if ok else ""})()

    monkeypatch.setattr(fp8.subprocess, "run", run)
    got = fp8.smi_query("GPU-x", "driver_version,clocks.sm,clocks_event_reasons.active")
    assert got == {"driver_version": "DRIVER_VERSION", "clocks.sm": "CLOCKS.SM", "clocks_event_reasons.active": None}


def test_fake_silicon_refused_under_research_run(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    assert fp8.main(["--row", "fadd", "--out", str(tmp_path), "--fake-silicon"]) == 2


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("nvcc") is None or shutil.which("cuobjdump") is None, reason="needs nvcc and cuobjdump (CUDA 12.9)")
def test_sass_gate_on_a_real_build(tmp_path):
    so, _ = fp8.P4.build(fp8.ARCH, tmp_path, fp8.SOURCE)
    gate = fp8.sass_gate(so)
    assert gate["ok"], {k: r for k, r in gate["kernels"].items() if not r["ok"]}
