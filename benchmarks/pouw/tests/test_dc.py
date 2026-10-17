"""dc: wgmma.cuh is gen_wgmma.py's output, the receipt twin (dgemm_ref.py) against a per-word loop and against
dgemm.cu's host reference (pinned), and analyze.py's slot arithmetic and pass lines on synthetic rows. The kernels
themselves need an H200 or a B200 (dgemm gate, run.sh)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

DC = Path(__file__).resolve().parents[1] / "dc"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_wgmma_header_is_current():
    gen = _load("gen_wgmma")
    assert (DC / "wgmma.cuh").read_text() == gen.render(), "run python benchmarks/pouw/dc/gen_wgmma.py"


def _loop_receipts(X, W, seed, bn):
    """The receipt word by word, as dgemm.cu's header states it."""
    ref = _load("dgemm_ref")
    m, k = X.shape
    n = W.shape[0]
    tiles_n = n // bn
    M = 0xFFFFFFFF
    R = np.zeros((m, tiles_n, 16), dtype=np.uint32)
    T = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        st = {}
        for tn in range(tiles_n):
            for c in range(16):
                y = int(ref.mix32(np.uint32(seed) ^ ref.mix32(np.uint32((i * tiles_n + tn) * 16 + c))))
                st[tn, c] = [y, int(ref.mix32(np.uint32((y + 0x9E3779B9) & M)))]
        for s in range(k // 32):
            T[i] += X[i, 32 * s:32 * s + 32].astype(np.int64) @ W[:, 32 * s:32 * s + 32].T.astype(np.int64)
            for tn in range(tiles_n):
                for j in range(bn):
                    y, z = st[tn, j % 16]
                    t = (j // 16) % 16
                    p = int(T[i, tn * bn + j]) & M
                    u = y ^ p ^ int(ref.KEY_XOR[t])
                    y = (u * int(ref.KEY_MUL[t]) + z) & M
                    rr = ref.ROT[t]
                    z = ((((u << rr) | (u >> (32 - rr))) & M) + y) & M
                    st[tn, j % 16] = [y, z]
        for (tn, c), (y, z) in st.items():
            R[i, tn, c] = y ^ (((z << 16) | (z >> 16)) & M)
    return T, R


def test_receipt_twin_matches_word_loop():
    ref = _load("dgemm_ref")
    rng = np.random.default_rng(3)
    X = rng.integers(-127, 128, size=(3, 96), dtype=np.int8)
    W = rng.integers(-127, 128, size=(64, 96), dtype=np.int8)
    T, R = ref.receipts(X, W, 11, 32)
    T2, R2 = _loop_receipts(X, W, 11, 32)
    assert (T == T2).all() and (R == R2).all()


def _code(seed: int, which: int, n: int) -> np.ndarray:
    """dgemm.cu's code(): the s8 inputs every mode makes."""
    ref = _load("dgemm_ref")
    base = ref.mix32(np.uint32(seed ^ (which << 28)))
    with np.errstate(over="ignore"):
        x = ref.mix32(base + np.arange(n, dtype=np.uint32))
    return ((x % 255).astype(np.int64) - 127).astype(np.int8)


def test_receipt_twin_matches_host_reference():
    """`dgemm cpuref DIR 128 256 256 7 90` (dgemm.cu's host_ref, CUDA 13.0) wrote these R and C."""
    ref = _load("dgemm_ref")
    m, n, k = 128, 256, 256
    X = _code(7, 1, m * k).reshape(m, k)
    W = _code(7, 2, n * k).reshape(n, k)
    T, R = ref.receipts(X, W, 7, 128)
    assert hashlib.sha256(T.astype(np.int32).tobytes()).hexdigest() == \
        "25fc61f6afb0f6eee278ae91b6a50e1db948e15d9cbbf27e1d7f424071fa198e"
    assert hashlib.sha256(R.tobytes()).hexdigest() == "427f9cfb6776df8a476f3b230fac55b47e2ddf0a08605a1829ea6f746f0f74ba"


def _rows(d: Path, rows: dict[str, dict[str, float]]):
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.jsonl").write_text("".join(json.dumps({"probe": p, "rates": r, "sm_mhz": 1755.0, "tags": []}) + "\n"
                                          for p, r in rows.items()))


def test_analyze_sm100_pass_line(tmp_path):
    an = _load("analyze")
    t5 = {"0:UTCIMMA*": 1 / 128}   # 256 slots a SM a clock at 128 x 256
    base = {"tc_i8_m128_n256": t5, "ld_32x32b_x32": {"0:LDTM.x32": 0.3}, "imma_s8_k16": {"0:IMMA.16816.S8.S8": 0.5},
            "s_duo": {"0:LOP3.LUT": 2.0}}
    _rows(tmp_path / "a", {**base, "ldmma_w8": {**t5, "1:LDTM.x32": 0.25}})
    out = an.analyze("sm100", tmp_path / "a")
    assert out["M4"]["rho_T5"] == 256 and out["M4"]["rho_ld"] == 256 and out["M4"]["rho_sync"] == 64
    assert out["M4"]["pass"] and out["fold_headroom"] == 64 / 256
    _rows(tmp_path / "b", {**base, "ldmma_w8": {**t5, "1:LDTM.x32": 0.2}})
    assert not an.analyze("sm100", tmp_path / "b")["M4"]["pass"]


def test_analyze_sm90_pass_line(tmp_path):
    an = _load("analyze")
    wg = {"0:IGMMA.64x256x32.S8.S8*": 128 / 4096}   # 128 slots a SM a clock
    _rows(tmp_path / "a", {"wg_s8_n256_a1": wg, "wg_e4m3_n256_a1": {"0:QGMMA.64x256x32.F32.E4M3.E4M3*": 128 / 4096},
                           "imma_s8_k16": {"0:IMMA.16816.S8.S8": 1.0},
                           "wgmix_imad_r64": {**wg, "0:IMAD": 0.5}, "s_imad": {"0:IMAD": 2.0}})
    out = an.analyze("sm90", tmp_path / "a")
    assert out["M3"]["honest_slots"] == 128 and out["M3"]["pass"]
    j = out["joints"]["wgmix_imad_r64"]
    assert j["lane_fraction"] == 0.25 and j["a"] == (2.0 - 0.5) / (128 / 4096) and j["tensor_fraction"] == 1.0
    _rows(tmp_path / "b", {"wg_s8_n256_a1": wg, "imma_s8_k16": {"0:IMMA.16816.S8.S8": 1.1}})
    assert not an.analyze("sm90", tmp_path / "b")["M3"]["pass"]
