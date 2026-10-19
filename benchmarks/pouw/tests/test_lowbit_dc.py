"""lowbit_dc: each emulated route's k32 word step as egemm.cu states it (k3z and k4s on tcgen05, k3h on wgmma E4M3),
modelled in FP32 against the int64 word on every int8 pair and on the limb extremes, with the adder conditions fpx
measured (B200: exact iff representable; H200: a 14-bit window); and analyze.py's tables on synthetic rows. emu8's
tests cover the k3 and k4 identities themselves. `test_build_selftest` builds egemm.cu and
`test_pgemm_build_selftest_issue` pgemm.cu with nvcc; they skip without CUDA 13's nvcc. The kernels need a B200 or an
H200 (`run.sh gemm`; pgemm a B200, `run.sh pgemm`)."""
from __future__ import annotations

import importlib.util
import json
import re
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
    for arm, ms in (("i8_bn64", 1.2), ("i8_bn128", 1.0), ("i8_bn256", 0.9), ("i8r_bn128", 1.1), ("k3z_bn128", 2.0),
                    ("k4s_bn128", 1.6), ("k4s_bn64", 1.8)):
        for sfx, f in (("", 1.0), ("_res", 0.5)) + ((("_pipe", 0.25),) if not arm.startswith("i8_") else ()):
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
    pipe = k4["pipelined"]
    assert pipe["base"] == "i8r_bn128" and pipe["fold_ratio_same_tile"] == pytest.approx(1.6 / 1.1)
    assert pipe["fold_ratio_dgemm_tile"] == pytest.approx(1.6 * 0.25 / (0.9 * 0.5))
    assert "pipelined" not in rows["k4s_bn64"]  # no i8r at BN = 64 in this bench
    (nat,) = s["i8r"]
    assert nat["arm"] == "i8r_bn128" and nat["pipe_over_i8_res"] == pytest.approx(1.1 * 0.25 / 0.5)
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


def _pg_gate(arm: str, ok: bool = True) -> dict:
    return _gate(arm, ok) | {"m": 128, "n": 128, "k": 128, "dist": "uniform"}


def _pg_bench() -> dict:
    """pgemm bench's line: at both spacings every Dither route takes FULL 1.0 ms at FW 2 (k4s 2.0, i8c 1.0 at FW 1;
    k4t only FW 1, 1.2), and the split arms (k4t's at FW 1) MMARES 0.3, MMA 0.4, LD 0.5, EXT 0.8, FOLDONLY 0.9 ms
    (k4s's fold alone 1.9), FULLRES 0.8 (k4s 1.5)."""
    b = {"mode": "bench", "arch": "sm_100a", "m": 4096, "n": 4096, "k": 4096, "reps": 20, "sms": 148,
         "clock_khz": 1965000, "cublaslt_f8_out": "f32", "pf8_ms": 0.2, "pi8_ms": 0.25, "cublaslt_f8_ms": 0.125,
         "cublaslt_i8_ms": 0.16, "pf8_res_ms": 0.15}
    for L in (32, 64):
        for r in an.PG_DITHER:
            if r != "k4t":
                b[f"{r}_L{L}_fw2_full_ms"] = 2.0 if r == "k4s" else 1.0
            if r != "k4p":
                b[f"{r}_L{L}_fw1_full_ms"] = 1.6 if r == "k4s" else 1.2
            for md, ms in (("mmares", 0.3), ("mma", 0.4), ("ld", 0.5), ("ext", 0.8),
                           ("foldonly", 1.9 if r == "k4s" else 0.9), ("fullres", 1.5 if r == "k4s" else 0.8)):
                b[f"{r}_L{L}_fw{an.PG_SPLIT_FW.get(r, 2)}_{md}_ms"] = ms
    return b


def _pg_sass(arm: tuple[int, int, int, int], word: list[str], extra: int = 0, bnr: int | None = None) -> str:
    """A cuobjdump kernel whose word loop is BNR copies of `word` (and `extra` lone MOVs) closed by a backward BRA."""
    r, md, L, fw = arm
    ops = ["S2R R0, SR_TID.X"] + [w for _ in range(bnr or 128 // fw) for w in word] + ["MOV R1, R2"] * extra
    lines = [f"        /*{16 * i:04x}*/                   {op} ;" for i, op in enumerate(ops)]
    lines.append(f"        /*{16 * len(ops):04x}*/                   BRA 0x10 ;")
    return f"\n\tFunction : _Z5pg100ILi{r}ELi{md}ELi{L}ELi{fw}EEvPKhS1_iiijPiPjiPKj\n" + "\n".join(lines) + "\n"


def test_fold_mix_word_loop():
    duo = ["LOP3.LUT R3, R3, R4, R5, 0x96, !PT", "IMAD R6, R3, R7, R8", "LEA.HI R8, R3, R6, R3, 0x5"]
    sass = (_pg_sass((0, 0, 32, 2), ["IADD3 R9, R9, R10, -R11"] + duo, extra=64)
            + _pg_sass((5, 0, 32, 2), ["IMAD.IADD R9, R9, 0x1, R10"] + duo)
            + _pg_sass((1, 4, 64, 1), duo) + _pg_sass((0, 3, 32, 2), ["SYNCS.ARRIVE.TRANS64 RZ, [UR4], RZ"])
            + _pg_sass((7, 0, 64, 2), ["IMAD.IADD R9, R10, 0x1, R11"] + duo, bnr=128))
    mix = an.fold_mix(sass)
    k4t = mix.pop("k4t_L64_fw1_full")  # template FW 2, one warpgroup a 128-column receipt tile: 128 words a trip
    assert k4t["alu_word"] == 2 and k4t["imad_word"] == 2 and k4t["instrs_word"] == pytest.approx(4 + 1 / 128)
    k4s, k4p = mix["k4s_L32_fw2_full"], mix["k4p_L32_fw2_full"]
    assert k4s["alu_word"] == 3 and k4s["imad_word"] == 1 and k4s["instrs_word"] == pytest.approx(5 + 1 / 64)
    assert k4s["model_words_sm_clk"] == pytest.approx(64 / 3)  # the ALU pipe binds: 3 ops at 2 warp instrs a clock
    assert k4p["alu_word"] == 2 and k4p["imad_word"] == 2  # balanced, so issue binds: 4 a clock over 4 + 1/64
    assert k4p["model_words_sm_clk"] == pytest.approx(128 / (4 + 1 / 64))
    assert mix["i8c_L64_fw1_foldonly"]["model_words_sm_clk"] == pytest.approx(32)
    assert mix["k4s_L32_fw2_mma"]["instrs_word"] == pytest.approx(1 + 1 / 64)
    assert set(mix) == {"k4s_L32_fw2_full", "k4p_L32_fw2_full", "i8c_L64_fw1_foldonly", "k4s_L32_fw2_mma"}


def test_pgemm_summary_gamma_split_binding():
    s = an.pgemm_summary([_pg_gate("k4s_L32_fw2_full"), _pg_gate("i8c_L32_fw2_full", ok=False)], [_pg_bench()],
                         {"k4s_L32_fw2_full": {"alu_word": 3.0}})
    assert s["gates"] == 2 and s["gates_exact"] == 1 and len(s["bench"]) == 2
    row = next(r for r in s["bench"] if r["L"] == 32)
    k4s, i8c, k4p = (row["routes"][r] for r in ("k4s", "i8c", "k4p"))
    assert k4s["fw"] == 1 and k4s["full_ms"] == 1.6 and i8c["fw"] == 2 and k4p["fw"] == 2
    assert k4s["gamma_cublaslt_f8"] == pytest.approx(12.8) and k4s["gamma_pf8"] == pytest.approx(8)
    assert k4s["gamma_cublaslt_i8"] == pytest.approx(10) and k4s["gamma_pi8"] == pytest.approx(6.4)
    assert row["int8_dither"] in ("i8c", "i8r") and k4s["over_int8_dither"] == pytest.approx(1.6)
    sp = k4s["split_ms"]  # at FW 2, whose FULL is 2.0
    assert [sp[x] for x in ("tensor", "operand_feed", "tensor_memory_loads", "extraction", "fold")] == \
        pytest.approx([0.3, 0.1, 0.1, 0.3, 1.2])
    assert k4s["binding"] == "fold" and k4s["binding_of_full"] == pytest.approx(0.95)
    assert k4s["gamma_resident_cublaslt_f8"] == pytest.approx(12) and k4s["gamma_resident_pf8_res"] == pytest.approx(10)
    assert "gamma_pf8_res" not in k4s
    assert k4s["issue"] == {"alu_word": 3.0} and i8c["issue"] is None
    words = 4096**3 / 32
    assert k4s["words_sm_clk"] == pytest.approx(words / (1.6e-3 * 148 * 1.965e9))
    assert row["plain_words_sm_clk"]["cublaslt_f8"] == pytest.approx(words / (0.125e-3 * 148 * 1.965e9))
    row64 = next(r for r in s["bench"] if r["L"] == 64)
    assert row64["routes"]["k4s"]["words_sm_clk"] == pytest.approx(k4s["words_sm_clk"] / 2)
    timed = an.pgemm_summary([], [_pg_bench() | {"k4s_L32_fw1_full_mhz": 1500.0}], {})["bench"][0]["routes"]["k4s"]
    assert timed["mhz"] == 1500.0 and timed["words_sm_clk"] == pytest.approx(words / (1.6e-3 * 148 * 1.5e9))


def _dc_rate(arm: int, L: int, wpc: float, mhz: float = 1800.0, nb: int = 4) -> dict:
    return {"mode": "rate", "nb": nb, "lw": 2, "mi": 0, "arm": arm, "L": L, "committed_words_per_clk": wpc, "mhz": mhz}


def test_pgemm_summary_against_dc():
    """The CTA rate is a tile's words over the median CTA's clocks; DC's best honest (arm 2) row at each L sets the
    per-clock and per-second ratios and DC's own γ against cuBLASLt FP8 at the shape."""
    b = _pg_bench() | {"k4s_L32_fw1_full_mhz": 1600.0, "k4s_L32_fw1_full_cta_clk": 128 * 128 * 4096 / 32 / 15.0}
    dc = [_dc_rate(2, 32, 25.0), _dc_rate(2, 32, 30.0, nb=2), _dc_rate(0, 32, 90.0), _dc_rate(2, 64, 29.0)]
    s = an.pgemm_summary([], [b], {}, dc)
    assert s["dc_int8"][32]["committed_words_per_clk"] == 30.0 and s["dc_int8"][32]["nb"] == 2
    row = next(r for r in s["bench"] if r["L"] == 32)
    k4s = row["routes"]["k4s"]
    assert k4s["cta_words_sm_clk"] == pytest.approx(15.0) and k4s["gamma_fp8_peak_clk"] == pytest.approx(256 / 15)
    assert row["dc_int8"]["gamma_fp8_peak_clk"] == pytest.approx(256 / 30)
    assert next(r for r in s["bench"] if r["L"] == 64)["dc_int8"]["gamma_fp8_peak_clk"] == pytest.approx(128 / 29)
    assert k4s["over_dc_int8_clk"] == pytest.approx(2.0) and k4s["over_dc_int8_s"] == pytest.approx(2.0 * 1800 / 1600)
    assert "over_dc_int8_clk" not in row["routes"]["i8c"] and row["routes"]["i8c"]["cta_words_sm_clk"] is None
    dc_ms = 4096**3 / 32 / (30.0 * 148 * 1.8e9) * 1e3
    assert row["dc_int8"]["ms_at_shape"] == pytest.approx(dc_ms)
    assert row["dc_int8"]["gamma_cublaslt_f8"] == pytest.approx(dc_ms / 0.125)
    assert an.pgemm_summary([], [b], {})["bench"][0]["dc_int8"] is None


def test_pgemm_summary_k4t():
    """k4t's CTA is 128 x 256, and its kernel of record, time split and issue cost are its FW 1 arms."""
    b = _pg_bench() | {"k4t_L32_fw1_full_cta_clk": 128 * 256 * 4096 / 32 / 28.0}
    s = an.pgemm_summary([], [b], {"k4t_L32_fw1_full": {"alu_word": 2.1}}, [_dc_rate(2, 32, 29.4)])
    k4t = next(r for r in s["bench"] if r["L"] == 32)["routes"]["k4t"]
    assert k4t["fw"] == 1 and k4t["full_ms"] == 1.2 and k4t["cta_words_sm_clk"] == pytest.approx(28.0)
    assert k4t["split_ms"]["fold"] == pytest.approx(0.4) and k4t["binding_of_full"] == pytest.approx(0.75)
    assert k4t["issue"] == {"alu_word": 2.1} and k4t["over_dc_int8_clk"] == pytest.approx(1.05)


def test_pgemm_main_exit(tmp_path, capsys):
    (tmp_path / "bench.jsonl").write_text(json.dumps(_pg_bench()) + "\n")
    (tmp_path / "gate.jsonl").write_text(json.dumps(_pg_gate("k4s_L32_fw2_full")) + "\n")
    assert an.pgemm_main([str(tmp_path), "--json", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "gates exact (C, receipts, control rejected): 1 of 1" in out and "| 4096x4096x4096 | 32 | k4s | 1 | 1.6000 | - |" in out
    assert json.loads((tmp_path / "s.json").read_text())["bench"][0]["routes"]["k4s"]["binding"] == "fold"
    assert "Against DC's" not in out
    (tmp_path / "dc-rate.jsonl").write_text(json.dumps(_dc_rate(2, 32, 30.0)) + "\n")
    assert an.pgemm_main([str(tmp_path)]) == 0
    assert "| 4096x4096x4096 | 32 | k4s | - | 30.0 (1800) | - | - | " in capsys.readouterr().out
    (tmp_path / "gate.jsonl").write_text(json.dumps(_pg_gate("k4s_L32_fw2_full", ok=False)) + "\n")
    assert an.pgemm_main([str(tmp_path)]) == 4
    (tmp_path / "bench.jsonl").write_text("")
    (tmp_path / "gate.jsonl").write_text(json.dumps(_pg_gate("k4s_L32_fw2_full")) + "\n")
    assert an.pgemm_main([str(tmp_path)]) == 4


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
    pair for all three routes, and both families' kernels compile without spills, except ptxas's 4 bytes in the
    serial i8r BN = 64 arm with fold and write (the pipelined i8r arm, the one the price compares against, has none)."""
    for arch in ("100a", "90a"):
        binary = tmp_path / f"egemm{arch}"
        r = subprocess.run([_tool("nvcc"), "-std=c++17", "-O3", "-Xptxas", "-v", "-gencode",
                            f"arch=compute_{arch},code=sm_{arch}", "-o", str(binary), str(HERE / "egemm.cu")],
                           capture_output=True, text=True, check=True)
        spilled, fn = {}, None
        for ln in r.stderr.splitlines():
            if m := re.search(r"Function properties for (\S+)", ln):
                fn = m.group(1)
            elif (m := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)) and m.groups() != ("0", "0"):
                spilled[fn] = int(m.group(1))
        assert all(re.fullmatch(r"_Z5eg100ILi3ELi64ELb1ELb1ELb[01]ELb0E.*", f) and b <= 4 for f, b in spilled.items()), \
            spilled
        out = json.loads(subprocess.run([str(binary), "selftest"], capture_output=True, text=True, check=True).stdout)
        assert out["encode_bad"] == 0 and out["identity_bad"] == 0 and out["identity_pairs"] == 3 * 65536


@pytest.mark.slow
@pytest.mark.skipif(not (_tool("nvcc") and _tool("cuobjdump")), reason="needs nvcc and cuobjdump (CUDA 13)")
def test_pgemm_build_selftest_issue(tmp_path):
    """What only the compiler shows: pgemm.cu's host encoding is exact (selftest), every Dither route's FULL kernel at
    FW 2 compiles without spills (FW 1's k4s and k4p spill, FOLDONLY a few bytes), and the word loop's issue cost is
    the route's: k4s 3 ALU-pipe ops a word (64/3 words/SM/clk), k4p and i8r 2 ALU plus 2 IMAD, k4b those and a VIADD,
    i8c duo alone, k4t 2 ALU plus 2 IMAD (its extraction an IMAD-pipe add; its plain words, the rebase word apart). A
    rematerialized tcgen05.st constant, which once cost k4s a move a word, fails the last."""
    binary = tmp_path / "pgemm"
    r = subprocess.run([_tool("nvcc"), "-std=c++17", "-O3", "-Xptxas", "-v", "-gencode",
                        "arch=compute_100a,code=sm_100a", "-o", str(binary), str(HERE / "pgemm.cu")],
                       capture_output=True, text=True, check=True)
    spilled, fn = {}, None
    for ln in r.stderr.splitlines():
        if m := re.search(r"Compiling entry function '(\S+)'", ln):
            fn = m.group(1)
        elif (m := re.search(r"(\d+) bytes spill stores", ln)) and m.group(1) != "0":
            spilled[fn] = int(m.group(1))
    assert not [f for f in spilled if re.match(r"_Z5pg100ILi\dELi0ELi\d+ELi2E", f)], spilled
    out = json.loads(subprocess.run([str(binary), "selftest"], capture_output=True, text=True, check=True).stdout)
    assert out == {"mode": "selftest", "identity_pairs": 65536, "identity_bad": 0, "encode_bad": 0, "tile_bad": 0}
    mix = an.fold_mix(subprocess.run([_tool("cuobjdump"), "-sass", str(binary)], capture_output=True, text=True,
                                     check=True).stdout)
    for L in (32, 64):
        word = {r: mix[f"{r}_L{L}_fw{an.PG_SPLIT_FW.get(r, 2)}_full"] for r in an.PG_DITHER}
        assert word["k4t"]["alu_word"] == pytest.approx(2, abs=0.15) and word["k4t"]["imad_word"] == pytest.approx(2, abs=0.1)
        assert word["k4t"]["instrs_word"] < 4.4
        assert word["k4s"]["alu_word"] == pytest.approx(3, abs=0.1) and word["k4s"]["imad_word"] < 1.1
        for r in ("k4p", "i8r"):
            assert word[r]["alu_word"] == pytest.approx(2, abs=0.1) and word[r]["imad_word"] == pytest.approx(2, abs=0.1)
        assert word["i8c"]["alu_word"] == pytest.approx(2, abs=0.1) and word["i8c"]["imad_word"] < 1.1
        assert word["k4b"]["alu_word"] < 2.3 and word["k4b"]["imad_word"] == pytest.approx(2, abs=0.1)
        assert all(w["instrs_word"] < (5.7 if r == "k4b" else 4.7) for r, w in word.items()), word


@pytest.mark.slow
@pytest.mark.skipif(not (_tool("nvcc") and _tool("cuobjdump")), reason="needs nvcc and cuobjdump (CUDA 13)")
def test_fpx_builds_for_b300(tmp_path):
    """fpx is B300's exactness run: on sm_103a its tcgen05 kernels must compile to the same MMAs as on sm_100a, not to
    the empty bodies an sm_100-only guard leaves."""
    counts = {}
    for arch in ("100a", "103a"):
        binary = tmp_path / f"fpx{arch}"
        subprocess.run([_tool("nvcc"), "-std=c++17", "-O3", "-gencode", f"arch=compute_{arch},code=sm_{arch}", "-o",
                        str(binary), str(HERE / "fpx.cu")], capture_output=True, check=True)
        sass = subprocess.run([_tool("cuobjdump"), "-sass", str(binary)], capture_output=True, text=True, check=True).stdout
        counts[arch] = (len(re.findall(r"\bUTCQMMA\b", sass)), len(re.findall(r"\bUTCOMMA\b", sass)))
    assert counts["103a"] == counts["100a"] and counts["103a"][0] > 0 and counts["103a"][1] == 2, counts
