"""lowbit_dc: each emulated route's k32 word step as egemm.cu states it (k3z and k4s on tcgen05, k3h on wgmma E4M3),
modelled in FP32 against the int64 word on every int8 pair and on the limb extremes, with the adder conditions fpx
measured (B200: exact iff representable; H200: a 14-bit window); and analyze.py's tables on synthetic rows. emu8's
tests cover the k3 and k4 identities themselves. `test_build_selftest` builds egemm.cu with nvcc; it skips without
CUDA 13's nvcc. The kernels need a B200 or an H200 (`run.sh gemm`)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

POUW = Path(__file__).resolve().parents[1]
HERE = POUW / "lowbit_dc"
CUDA = Path("/usr/local/cuda-13.0/bin")
MAGIC = 1.5 * 2**23
MAGIC_BITS = 0x4B400000


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


emu = _load("emu8_emu8", POUW / "emu8" / "emu8.py")
an = _load("lowbit_dc_analyze", HERE / "analyze.py")


def planes(x, route: str, side: int) -> list[np.ndarray]:
    """egemm.cu's plane_val."""
    h, l = emu.limbs(x)
    if route == "k3z":
        return [4 * h, h + l, l if side else -l]
    if route == "k4s":
        return [16 * h, l]
    return [h, h + l, l]


def f32(v: np.ndarray) -> np.ndarray:
    """One FP32 rounding of exact float64 values (an FFMA of integers whose exact result float64 holds)."""
    return v.astype(np.float32)


def bits(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32).view(np.uint32).astype(np.int64)


def word_step(A: np.ndarray, B: np.ndarray, route: str) -> np.ndarray:
    """One k32 step's increment of T, for A (rows x 32) and B (cols x 32), as egemm.cu computes it: each accumulator
    exact (asserted against the adder fpx measured), then the extraction in FP32 and the bits difference."""
    pa, pb = planes(A, route, 0), planes(B, route, 1)
    dot = lambda p, q: p @ q.T  # noqa: E731  (int64 exact)
    if route == "k3z":
        acc0, acc1 = dot(pa[0], pb[0]) + dot(pa[2], pb[2]), dot(pa[1], pb[1])
        assert np.abs(acc0).max() < 2**16 and np.abs(acc1).max() < 2**16
        inner = f32(MAGIC + 15.0 * acc0)
        assert (inner.astype(np.float64) == MAGIC + 15 * acc0).all()
        return bits(f32(16.0 * acc1 + inner.astype(np.float64))) - MAGIC_BITS
    if route == "k4s":
        acc = np.full((A.shape[0], B.shape[0]), MAGIC)
        for i in (0, 1):
            for j in (0, 1):
                prod = pa[i][:, None, :] * pb[j][None, :, :]
                for t in range(32):  # every partial sum in [2^23, 2^24), so tcgen05's adder is exact (gap <= 23)
                    acc = acc + prod[:, :, t]
                    assert (np.abs(acc - MAGIC) < 2**22).all()
        return bits(f32(acc)) - MAGIC_BITS
    g = [dot(pa[p], pb[p]) for p in range(3)]
    for p in range(3):  # a 14-bit window keeps every integer below 2^13: each term and each partial sum
        prod = pa[p][:, None, :] * pb[p][None, :, :]
        assert np.abs(prod).max() < 2**13 and np.abs(np.cumsum(prod, axis=2)).max() < 2**13
    v = f32(MAGIC + 240.0 * g[0]).astype(np.float64)
    v = f32(16.0 * g[1] + v).astype(np.float64)
    return bits(f32(-15.0 * g[2] + v)) - MAGIC_BITS


@pytest.mark.parametrize("route", ["k3z", "k4s", "k3h"])
def test_planes_are_e4m3_integers(route):
    x = np.arange(-128, 128)
    for side in (0, 1):
        for p in planes(x, route, side):
            emu.codes(np.unique(p))


@pytest.mark.parametrize("route", ["k3z", "k4s", "k3h"])
def test_word_step_every_pair_and_extremes(route):
    x = np.arange(-128, 128, dtype=np.int64)
    a = np.repeat(x, 32).reshape(256, 32)
    assert (word_step(a, a, route) == a @ a.T).all()
    rng = np.random.default_rng(5)
    edge = np.array([-128, 127, 119, 120, -120, -121, 0, 1], np.int64)
    for _ in range(4):
        A, B = rng.choice(edge, (16, 32)), rng.choice(edge, (24, 32))
        assert (word_step(A, B, route) == A @ B.T).all()
    worst = np.full((2, 32), -128, np.int64)
    worst[1] = 127
    assert (word_step(worst, worst, route) == worst @ worst.T).all()


def test_k3z_acc0_breaks_hoppers_window():
    """Why H200 runs k3h: k3z's first MMA leaves acc0 at up to 2^15, and the second adds (-l)(l') terms as small as 1, a
    gap of 15 bits that Hopper's 14-bit window (exact at gap <= 13) rounds away."""
    x = np.arange(-128, 128)
    pa, pb = planes(x, "k3z", 0), planes(x, "k3z", 1)
    first = 32 * np.abs(np.outer(pa[0], pb[0])).max()
    small = np.abs(np.outer(pa[2], pb[2]))
    assert first == 2**15 and small[small > 0].min() == 1


def _gate(arm: str, ok: bool = True) -> dict:
    return {"arm": arm, "exact": ok, "control_rejected": True, "split_bad": 0}


def _bench(arch: str, shape=(4096, 4096, 4096)) -> dict:
    m, n, k = shape
    b = {"m": m, "n": n, "k": k, "arch": arch, "sms": 148, "clock_khz": 1_965_000, "split_x_ms": 0.05,
         "split_x_k4s_ms": 0.04}
    for arm, ms in (("i8_bn64", 1.2), ("i8_bn128", 1.0), ("i8_bn256", 0.9), ("k3z_bn128", 2.0), ("k4s_bn128", 1.6),
                    ("k4s_bn64", 1.8)):
        for sfx, f in (("", 1.0), ("_res", 0.5)):
            b[f"{arm}_fold{sfx}_ms"] = ms * f
            b[f"{arm}_nofold{sfx}_ms"] = ms * f * 0.8
    return b


def test_gemm_summary_ratios():
    s = an.gemm_summary([_gate("i8_bn128"), _gate("k4s_bn128")], [_bench("sm_100a")],
                        [{"m": 4096, "n": 4096, "k": 4096, "cublaslt_ms": 0.1}])
    assert s["gates"] == 2 and s["gates_exact"] == 2 and s["gate_arms"] == ["i8_bn128", "k4s_bn128"]
    rows = {r["emu"]: r for r in s["bench"]}
    assert set(rows) == {"k3z_bn128", "k4s_bn128", "k4s_bn64"}
    k4 = rows["k4s_bn128"]
    assert k4["i8_same_tile"] == "i8_bn128" and k4["i8_dgemm_tile"] == "i8_bn256"
    assert k4["streamed"]["fold_ratio_same_tile"] == pytest.approx(1.6)
    assert k4["resident"]["fold_ratio_dgemm_tile"] == pytest.approx(1.6 / 0.9)
    assert k4["split_x_over_i8_fold"] == pytest.approx(0.04) and k4["emu_fold_over_cublaslt"] == pytest.approx(16)
    words = 4096**3 / 32
    assert k4["streamed"]["emu_words_sm_clk"] == pytest.approx(words / (1.6e-3 * 148 * 1.965e9))
    assert rows["k4s_bn64"]["streamed"]["fold_ratio_same_tile"] == pytest.approx(1.5)
    assert an.gemm_summary([], [_bench("sm_90a")], [])["bench"][0]["i8_dgemm_tile"] == "i8_bn128"


def test_gemm_main_exit(tmp_path, capsys):
    def write(gates):
        (tmp_path / "gate.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gates))
        (tmp_path / "bench.jsonl").write_text(json.dumps(_bench("sm_100a")) + "\n")
    write([_gate("k3z_bn128"), _gate("k4s_bn128")])
    assert an.gemm_main([str(tmp_path), "--json", str(tmp_path / "s.json")]) == 0
    assert "gates exact (route, receipts, control rejected, split): 2 of 2" in capsys.readouterr().out
    assert json.loads((tmp_path / "s.json").read_text())["gates_exact"] == 2
    write([_gate("k3z_bn128"), _gate("k4s_bn128", ok=False)])
    assert an.gemm_main([str(tmp_path)]) == 4
    write([])
    assert an.gemm_main([str(tmp_path)]) == 4


def _rec(cfg: str, K: int, **kw) -> dict:
    r = {"cfg": cfg, "fmt": "e4m3", "grid": "int", "s": 1, "dist": "uniform", "c0": "zero", "c0bits": "0x0", "sa": 0,
         "sb": 0, "unit_num": 1, "unit_e": 0, "control": False, "K": K, "n": 10, "rep": 10, "inexact": 0, "bad_rep": 0,
         "rz": 10, "rn": 10, "negzero": 0, "bad_word": 0, "bad_acc": 0, "maxerr": 0}
    return r | kw


def test_per_config_and_chains():
    recs = [_rec("a", 32), _rec("a", 64, rep=9, inexact=1, rz=9, maxerr=1.0), _rec("b", 32, dist="runs"),
            _rec("b", 64, dist="runs")]
    rows = an.per_config(recs)
    a = next(r for r in rows if r["cfg"] == "a")
    assert a["first_nonrep"] == 64 and a["first_inexact"] == 64 and a["bad_rep"] == 0 and not a["rz_all"]
    (c,) = an.group_chains(rows)
    assert c["configs"] == 2 and c["first_inexact"] == 64 and c["rz_frac"] == pytest.approx(19 / 20)


def test_windows_largest_exact_gap():
    w = [{"fmt": "e4m3", "test": "big+small", "gap": g, "n": 4, "exact": 4 if g <= 13 else 1, "rz": 1, "rn": 0,
          "dropped": 3 if g > 13 else 0, "other": 0} for g in range(10, 17)]
    (r,) = an.windows(w)
    assert r["max_exact_gap"] == 13 and r["first_bad_gap"] == 14 and r["past"]["dropped"] == 9


def _tool(name: str) -> str | None:
    return str(CUDA / name) if (CUDA / name).exists() else shutil.which(name)


@pytest.mark.slow
@pytest.mark.skipif(not _tool("nvcc"), reason="needs nvcc (CUDA 13)")
def test_build_selftest(tmp_path):
    """What the CPU model can't: egemm.cu's own plane_val and e4m3_int round-trip every limb and recombine every int8
    pair for all three routes, and both families' kernels compile with no spills."""
    for arch in ("100a", "90a"):
        binary = tmp_path / f"egemm{arch}"
        r = subprocess.run([_tool("nvcc"), "-std=c++17", "-O3", "-Xptxas", "-v", "-gencode",
                            f"arch=compute_{arch},code=sm_{arch}", "-o", str(binary), str(HERE / "egemm.cu")],
                           capture_output=True, text=True, check=True)
        spills = [ln for ln in r.stderr.splitlines() if "spill" in ln and not ln.strip().startswith("0 bytes")]
        assert all(" 0 bytes spill stores, 0 bytes spill loads" in ln for ln in spills), spills
        out = json.loads(subprocess.run([str(binary), "selftest"], capture_output=True, text=True, check=True).stdout)
        assert out["encode_bad"] == 0 and out["identity_bad"] == 0 and out["identity_pairs"] == 3 * 65536
