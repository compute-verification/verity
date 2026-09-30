"""``tools/tc_probe_fp4/w1.py``, the W1 price microbenchmarks, on fake silicon and on synthetic SASS.

The fake run passes every gate and turns its rates into prices in the unit; a flipped gate word fails the run before anything is
timed; the SASS loop gate rejects a loop with a missing or an extra watched instruction; the cast gate's packing (hi operand in
the upper byte) is pinned; the tool declarations resolve their key params.  With nvcc on the PATH, the real build's loops are
checked.
"""
from __future__ import annotations

import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import w1  # noqa: E402


@pytest.fixture
def no_nvcc(monkeypatch):
    monkeypatch.setattr(w1.P4, "nvcc_bin", lambda: "/nonexistent/nvcc")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)


def _run(tmp_path, *extra):
    return w1.run(w1.parser().parse_args(["--out", str(tmp_path), "--fake-silicon", "--reps", "2", *extra]))


def test_fake_run_prices(tmp_path, no_nvcc):
    code, res = _run(tmp_path)
    assert code == 0, res["failures"]
    assert res["numeric_gate"]["ok"] and res["clocks"] == "unlocked"
    p = res["prices"]["price_units"]
    assert p["qmma_e4m3_f32 per MAC"] == 1.0
    assert p["fadd per register"] == pytest.approx(1024 / 128, rel=1e-6)
    assert p["f2fp_e4m3x2 per instruction (2 codes)"] == pytest.approx(2 * p["f2fp_e4m3x2 per code"])


def test_corrupt_gate_word_fails_before_timing(tmp_path, no_nvcc):
    code, res = _run(tmp_path, "--corrupt-word", "3")
    assert code == 1 and res["failures"] == ["numeric gate: qmma_e4m3_f32"]
    assert "timing" not in res and "prices" not in res


def test_gate_words_are_not_vacuous():
    rows = w1.timing_rows(8, np.random.default_rng(w1.F8.SEED_DEFAULT))
    assert w1.expected_words(8, w1.GATE_ITERS, rows)[:, : w1.NCHAIN].all(axis=1).mean() > 0.5


def test_vacuous_gate_fails(tmp_path, no_nvcc, monkeypatch):
    monkeypatch.setattr(w1, "expected_words", lambda op, iters, rows: np.zeros((w1.THREADS, 16), np.uint32))
    code, res = _run(tmp_path)
    assert code == 1 and res["numeric_gate"]["ops"]["fadd"]["vacuous"] and "prices" not in res


def _sass(body: list[str]) -> str:
    lines = [f"        /*{16 * i:04x}*/                   {op} R0, R1 ;" for i, op in enumerate(body)]
    lines.append(f"        /*{16 * len(body):04x}*/                   BRA.U UP0, 0x0 ;")
    return "\n\tFunction : _ZN12_GLOBAL__N_19fadd_rateEPKjPjPxPii\n" + "\n".join(lines) + "\n"


def test_loop_bodies_counts_one_loop():
    body = w1.loop_bodies(_sass(["FADD"] * 64 + ["UIADD3"]))["fadd_rate"]
    assert body == collections.Counter({"FADD": 64, "UIADD3": 1, "BRA.U": 1})


@pytest.mark.parametrize("body", [["FADD"] * 63, ["FADD"] * 64 + ["FFMA"], ["FADD"] * 64 + ["UFADD"]])
def test_sass_gate_rejects_a_wrong_loop(monkeypatch, body):
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body)})())
    gate = w1.sass_gate(Path("/nonexistent.so"), None)
    assert not gate["kernels"]["fadd"]["ok"]


def test_f2fp_packing_pinned():
    rows = np.zeros((32, 8), np.uint32)
    rows[:, 0] = np.float32(1.0).view(np.uint32)
    rows[:, 1] = np.float32(2.0).view(np.uint32)
    h = w1.f2fp_model(1, rows)
    f = int(w1.F8.cvt_ref("e4m3", np.array([0x3F800000], np.uint32))[0])
    assert f == 0x38
    xs = w1.F8.cvt_ref("e4m3", rows[0, 1] + (np.arange(16, dtype=np.uint32) << 20)).astype(int)
    assert xs.tolist() == list(range(0x40, 0x50))
    g = int(w1.F8.cvt_ref("e4m3", np.array([0x3F800000 + (1 << 21)], np.uint32))[0])
    assert g == 0x3A
    want = [0] * 8
    for k in range(16):
        want[k % 8] += ((f << 8) | xs[k]) | (((xs[k] << 8) | f) << 16)
        want[k % 8] += ((g << 8) | xs[k]) | (((xs[k] << 8) | g) << 16)
    assert h[0].tolist() == [w % 2**32 for w in want] and all(want)


def test_tool_declarations_resolve():
    from tool_fp8 import TC_PROBE_FP8, W1_PRICES

    _, k = TC_PROBE_FP8.resolve_params(["python", "tools/tc_probe_fp4/fp8.py", "--row", "chain", "--out", "/x", "--expect-gpu-uuid", "GPU-1"])
    assert k["row"] == "chain" and k["instruction"] is None and k["steps"] == 2048
    _, k = W1_PRICES.resolve_params(["python", "tools/tc_probe_fp4/w1.py", "--out", "/x", "--reps", "12"])
    assert k["reps"] == 12 and k["clocks"] == "unlocked"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("nvcc") is None or shutil.which("cuobjdump") is None, reason="needs nvcc and cuobjdump (CUDA 12.9)")
def test_sass_loops_on_a_real_build(tmp_path):
    so, _ = w1.P4.build(w1.F8.ARCH, tmp_path, w1.SOURCE)
    gate = w1.sass_gate(so, None)
    assert gate["ok"], {k: r["loop"] for k, r in gate["kernels"].items() if not r["ok"]}
