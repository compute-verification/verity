"""emu8: the limb identities over every int8 pair, the binade words and k3's constant-free recombination, the host's copy
of the generator, the int64 prefix words, the silicon-model replay, the recheck and its refusals, and the timing summary.
`test_build_hostdraw_and_sass` builds emu8.cu with nvcc; it skips without CUDA 13's nvcc.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

POUW = Path(__file__).resolve().parents[1]
HERE = POUW / "emu8"
CUDA = Path("/usr/local/cuda-13.0/bin")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


emu = _load("emu8_emu8", HERE / "emu8.py")


def f32(w: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", w))[0])


def bits(x: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_identities_every_int8_pair():
    res = emu.identities()
    for arm in ("k3", "k4"):
        assert res[arm]["pairs"] == 65536 and res[arm]["wrong"] == 0
    assert res["k3"]["max_abs_T_a_step"] == {"acc1": 32768, "acc2": 7200}
    assert res["k4"]["max_abs_T_a_step"] == {"acc": 524288}


def test_planes_are_e4m3_integers_and_limbs_balanced():
    x = np.arange(-128, 128)
    h, l = emu.limbs(x)
    assert (16 * h + l == x).all() and l.min() == -8 and l.max() == 7 and h.min() == -8 and h.max() == 8
    assert (h + l).min() == -15 and (h + l).max() == 14
    for arm in ("k3", "k4"):
        for side in (0, 1):
            for p in emu.planes(x, arm, side):
                c = emu.codes(p)
                assert [emu.e4m3_value(int(b)) for b in c] == [float(v) for v in p]
    with pytest.raises(KeyError):
        emu.codes(np.array([17]))  # 17 is not an E4M3 integer


def test_k3_binade_words_recombine_without_a_constant():
    """C1 and C2 sit in binades of ulp 1 and 1/8 with room for every step's T, the FP32 word is C + T there, and
    15 bits(C1 + T1) + 2 bits(C2 + T2) = 15 T1 + 16 T2 mod 2^32 at the extremes of T."""
    M = 1 << 32
    assert (15 * emu.C1_BITS + 2 * emu.C2_BITS) % M == 0
    assert emu.C1_BITS >> 23 == 127 + 23 and emu.C2_BITS >> 23 == 127 + 20 and emu.C0_BITS >> 23 == 127 + 23
    c1, c2, c0 = f32(emu.C1_BITS), f32(emu.C2_BITS), f32(emu.C0_BITS)
    for t1 in (-32768, -1, 0, 1, 32768):
        for t2 in (-6720, -1, 0, 1, 7200):
            b1, b2 = bits(c1 + np.float32(t1)), bits(c2 + np.float32(t2))
            assert (15 * b1 + 2 * b2) % M == (15 * t1 + 16 * t2) % M
            assert float(c1 + np.float32(t1)) == float(c1) + t1 and float(c2 + np.float32(t2)) == float(c2) + t2
    for t in (-(1 << 19), 0, 1 << 19):
        assert bits(c0 + np.float32(t)) - emu.C0_BITS == t
    # the binades' edges are past the extremes: one binade further and the bits stop being linear
    assert 2.0 ** 23 < float(c1) - 32768 and float(c1) + 32768 < 2.0 ** 24
    assert 2.0 ** 20 < float(c2) - 6720 and float(c2) + 7200 < 2.0 ** 21


def test_draw_ranges():
    for d in emu.DISTS:
        A, B = emu.draw(d, emu.GATE_SEED, 0, 6, 2048), emu.draw(d, emu.GATE_SEED, 1, 6, 2048)
        assert A.dtype == np.int8 and A.shape == (6, 2048) and B.shape == (6, 2048)
    A, B = emu.draw("dither", 1, 0, 64, 4096), emu.draw("dither", 1, 1, 64, 4096)
    assert A.min() >= -127 and A.max() <= 127 and B.min() >= -127 and B.max() <= 127
    Z, S = emu.draw("zero_a", 1, 0, 64, 4096), emu.draw("zero_a", 1, 1, 64, 4096)
    assert set(np.unique(Z)) == {0, 1} and 0 not in S and np.abs(S.astype(int)).max() == 64
    assert (S[:, :32] == S[:, 32:64]).all()  # B̃ = S, S[t, j] = U[j, t mod 32]
    assert set(np.unique(emu.draw("samesign", 1, 0, 8, 1024))) <= {-120, -127, -128}


def test_exact_words_against_a_loop():
    rng = np.random.default_rng(3)
    A = rng.integers(-128, 128, (3, 128)).astype(np.int8)
    B = rng.integers(-128, 128, (5, 128)).astype(np.int8)
    w = emu.exact_words(A, B)
    for l in range(4):
        for i in range(3):
            for j in range(5):
                assert w[l, i, j] == int((A[i, :32 * (l + 1)].astype(int) * B[j, :32 * (l + 1)]).sum())


@pytest.mark.parametrize("dist", emu.DISTS)
def test_replay_through_the_silicon_model(dist):
    A, B = emu.draw(dist, emu.GATE_SEED, 0, 8, 1024), emu.draw(dist, emu.GATE_SEED, 1, 8, 1024)
    r = emu.replay(A, B, 12)
    assert r["steps"] == r["k3_agree"] == r["k4_agree"] == 12


def _case(tmp_path, dist="maxlimb", M=4, N=3, K=128):
    d = tmp_path / f"c00-{dist}-{M}x{N}x{K}"
    d.mkdir()
    A, B = emu.draw(dist, emu.GATE_SEED, 0, M, K), emu.draw(dist, emu.GATE_SEED, 1, N, K)
    A.tofile(d / "A.bin")
    B.tofile(d / "B.bin")
    w = emu.exact_words(A, B)
    for arm in emu.ARMS:
        w.tofile(d / f"{arm}.bin")
    return d, w


def test_recheck_accepts_exact_words(tmp_path):
    d, _ = _case(tmp_path)
    v = emu.recheck(tmp_path, samples=4)
    assert len(v) == 1 and v[0]["refusals"] == [] and v[0]["bit_identical"] and v[0]["replay"]["k3_agree"] == 4


def test_recheck_refuses_a_wrong_word_poison_and_foreign_operands(tmp_path):
    d, w = _case(tmp_path)
    bad = w.copy()
    bad[1, 2, 0] += 1
    bad.tofile(d / "k3.bin")
    v = emu.recheck_case(d, 0)
    assert "k3: 1 words differ from the int64 words" in v["refusals"] and "the arms' words differ" in v["refusals"]
    p = w.copy().view(np.uint32)
    p[0, 0, :] = 0xA5A5A5A5
    p.view(np.int32).tofile(d / "k4.bin")
    v = emu.recheck_case(d, 0)
    assert v["k4"]["poisoned"] == 3 and any(r.startswith("k4:") for r in v["refusals"])
    A = np.fromfile(d / "A.bin", np.int8)
    A[0] ^= 1
    A.tofile(d / "A.bin")
    assert "the device's operands aren't the counter hash's" in emu.recheck_case(d, 0)["refusals"]


def test_recheck_refuses_short_words_and_a_failed_device_gate(tmp_path):
    d, w = _case(tmp_path)
    w.ravel()[:-1].tofile(d / "imma.bin")
    assert any("words," in r for r in emu.recheck_case(d, 0)["refusals"])
    log = tmp_path / "gate.jsonl"
    log.write_text('{"emu8": "gate", "ok": false, "accepted": true, "control_rejected": false}\n')
    assert emu.main(["recheck", str(tmp_path), "--samples", "0", "--gate-log", str(log)]) == 9


def test_summarize(tmp_path):
    t = tmp_path / "time.jsonl"
    t.write_text('{"emu8": "shape", "M": 128, "N": 128, "K": 256, "steady": true, "warmup_ms": 1.0, "fp8_lt": true}\n'
                 "GK arm=cublaslt_i8 reps=3 ms=0.010000 sm_mhz=2100 clock_gap=0.00%\n"
                 "GK arm=imma reps=3 ms=0.015000 sm_mhz=2100 clock_gap=0.00%\n"
                 "GK arm=k3 reps=3 ms=0.045000 sm_mhz=2100 clock_gap=0.10% [throttled]\n")
    s = emu.summarize(t)
    assert s[0]["arms"]["k3"]["vs_imma"] == pytest.approx(3.0) and s[0]["arms"]["k3"]["vs_cublaslt_i8"] == pytest.approx(4.5)
    assert s[0]["arms"]["k3"]["tags"] == "[throttled]" and "| 128x128x256 | k3 |" in emu.table(s)


acc = _load("emu8_accounting", HERE / "accounting.py")


@pytest.mark.parametrize("scheme,urange,need", [("k3", "±1..±64", 2), ("k3", "±1..±16", 2), ("k3", "one limb [-8, 7]", 1),
                                                ("k4", "±1..±64", 2), ("k4", "one limb [-8, 7]", 1)])
def test_cheater_needs_with_a_zero(scheme, urange, need):
    r = acc.needed(scheme, urange)
    assert r["needed"] == need and r["honest"] == len(acc.SCHEMES[scheme])
    assert r["S_is_e4m3"] == (urange != "±1..±64")
    one_limb_a = acc.needed(scheme, "±1..±64", a_values=tuple(range(-8, 8)))
    assert one_limb_a["needed"] == 2


def test_fp4_route_and_epsilon():
    r = acc.fp4_route(blocks=200)
    assert r["global_best_covered"] == 118 and r["targets"] == 128
    assert 0.8 < r["block_covered"] < 0.9 and 2.2 < r["fp4_products_a_mac"] < 2.35
    assert acc.epsilon(3.0, 2.0) == 1.5


def _tool(name: str) -> str | None:
    return str(CUDA / name) if (CUDA / name).exists() else shutil.which(name)


@pytest.mark.slow
@pytest.mark.skipif(not _tool("nvcc"), reason="needs nvcc (CUDA 13)")
def test_build_hostdraw_and_sass(tmp_path):
    """What the CPU tests can't: emu8.cu's own generator draws what the host regenerates, and today's nvcc compiles each
    timed mainloop to its declared MMA count (the run's SASS gate), with no spills."""
    binary = tmp_path / "emu8"
    subprocess.run([_tool("nvcc"), "-std=c++17", "-O3", "-gencode", "arch=compute_120a,code=sm_120a", "-o", str(binary),
                    str(HERE / "emu8.cu"), "-lcublasLt"], check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(Path(_tool("nvcc")).parent.parent / "lib64"))
    for i, d in enumerate(emu.DISTS):
        for side in (0, 1):
            out = subprocess.run([str(binary), "hostdraw", str(i), str(side), "3", "2080"], capture_output=True, check=True,
                                 env=env).stdout
            assert np.array_equal(np.frombuffer(out, np.int8).reshape(3, 2080), emu.draw(d, emu.GATE_SEED, side, 3, 2080)), d
    if not _tool("cuobjdump"):
        pytest.skip("needs cuobjdump for the SASS gate")
    env["PATH"] = f"{Path(_tool('cuobjdump')).parent}{os.pathsep}{os.environ['PATH']}"
    for arm, expect in ((0, "IMMA*=32"), (1, "QMMA*=48"), (2, "QMMA*=64")):
        for fold in (0, 1):
            r = subprocess.run([sys.executable, str(POUW / "gpu_kit" / "sass_gate.py"), str(binary), "--kernels",
                                f"_Z4gemmILi{arm}ELb{fold}ELb0E*", "--expect", expect], capture_output=True, text=True,
                               env=env)
            assert r.returncode == 0, r.stdout + r.stderr
