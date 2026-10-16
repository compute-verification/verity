"""fp_exact: the grids' codes, the host's copy of the generator, the representability test, the recheck and its refusals,
the silicon-model replay, the verdicts and the MMA rates. `test_build_dump_and_sass` builds fp_exact.cu with nvcc; it
skips without CUDA 13's nvcc and cuobjdump.
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

from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_NVF4

POUW = Path(__file__).resolve().parents[1]
HERE = POUW / "fp_exact"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load("fp_exact_fp_exact", HERE / "fp_exact.py")


def f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def cfg(fmt="e4m3", grid="e4m3_int16", dist="U", c0="zero", K=64, nrec=1, seed=12345) -> "fx.Config":
    w = {**fx.c0_words(fx.GRIDS[fx.GRID[grid]]), "negzero": 0x80000000}[c0]
    return fx.Config(fmt, grid, dist, c0, w, K, 128, seed, nrec)


def records(c, corrupt=None):
    """What the device writes for the config's recorded tiles when every output is the exact sum's word."""
    s_tot = 2 * fx.GRIDS[fx.GRID[c.grid]].s
    units = fx.word_units(c.c0_word, s_tot)
    out = np.zeros(c.nrec * 128, dtype=fx.REC)
    for t in range(c.nrec):
        A, B = fx.values(c, t)
        n = c.K // fx.KSTEP[c.fmt]
        part = np.einsum("isk,jsk->ijs", A.reshape(16, n, -1), B.reshape(8, n, -1)).cumsum(axis=2) + units
        ref = part[:, :, -1]
        r = out[t * 128:(t + 1) * 128].reshape(16, 8)
        r["d"] = fx.exact_words(ref, s_tot)[:, :, None].astype(np.uint32)
        r["ref"] = ref
        r["flags"] = fx.TAG | fx.representable(part).all(axis=2) | (fx.representable(ref) << 1)
    if corrupt:
        corrupt(out)
    return out


# ---- the grids ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("g", fx.GRIDS, ids=lambda g: g.name)
def test_grid_codes_multiply_to_their_values_on_the_silicon_model(g):
    cs = fx.codes(g, g.vals)
    for va, ca in zip(g.vals, cs):
        for vb, cb in zip(g.vals, cs):
            if g.e2m1:
                one = [0x38] * 4
                w = BLACKWELL_SM120_NVF4.step(0, [ca] + [0] * 63 + one, [cb] + [0] * 63 + one)
            else:
                w = BLACKWELL_SM120_E4M3_M16N8K32.step(0, [ca] + [0] * 31, [cb] + [0] * 31)
            assert f32(w) == va * vb * 2.0 ** (-2 * g.s), (g.name, va, vb)


def test_c0_words_sit_where_their_names_say():
    g = fx.GRIDS[fx.GRID["e4m3_quarter4"]]
    w = fx.c0_words(g)
    assert f32(w["binade"]) == 1.5 * 2 ** 23 * 2 ** -4 and f32(w["over"]) == 1.5 * 2 ** 24 * 2 ** -4
    assert w["nobits"] == 0x4B400000 and w["zero"] == 0
    assert "nobits" not in fx.c0_words(fx.GRIDS[fx.GRID["e4m3_int16"]])


def test_sweep_spans_k32_to_16384_and_beyond_for_every_format():
    cs = fx.configs()
    for fmt in fx.FMTS:
        k = {c.K for c in cs if c.fmt == fmt}
        assert sorted(k) == [fx.KSTEP[fmt] << j for j in range(len(k))] and max(k) == 262144
    assert len({c.name for c in cs}) == len(cs) and len({c.seed for c in cs}) == len(cs)
    assert {c.name for c in fx.configs(quick=True)} < {c.name for c in cs}


# ---- the generator -----------------------------------------------------------------------------------------------------


def test_mix64_is_splitmix64():
    assert int(fx.mix64(np.uint64(0))) == 0xE220A8397B1DCDAF
    assert int(fx.pick(6, np.uint64(0xFFFFFFFFFFFFFFFF))) == 5 and int(fx.pick(6, np.uint64(0))) == 0


def test_cancel_draws_sum_to_zero():
    A, B = fx.values(cfg(dist="CANCEL", K=256), 3)
    assert not (A @ B.T).any() and A.any()


def test_ramp_puts_the_large_products_first():
    c = cfg(dist="RAMP", K=128)
    A, B = fx.values(c, 0)
    assert (A[:, :64] == 16).all() and (B[:, :64] == 16).all() and (np.abs(A[:, 64:]) == 1).all()


def test_edge_firings_after_the_maximal_half_are_odd_about_half_the_time():
    """RAMP's +-1 products sum to an even number in every firing, so its partials never land on an odd value past 2^24;
    EDGE's 0-or-1 products do."""
    A, B = fx.values(cfg(dist="EDGE", K=4096), 0)
    assert (A[:, :2048] == 16).all() and set(np.unique(A[:, 2048:])) == {0, 1} and (B[:, 2048:] == 1).all()
    firing = (A[:, 2048:, None] * B.T[None, 2048:, :]).reshape(16, -1, 32, 8).sum(axis=2)
    assert 0.4 < (firing % 2).mean() < 0.6


@pytest.mark.parametrize("fmt", sorted(fx.SPARSE))
def test_sparse_operands_keep_two_pairs_of_every_four(fmt):
    A, _ = fx.values(cfg(fmt=fmt, grid="e2m1_int", K=1024), 0)
    pairs = (A.reshape(16, -1, 4, 2) != 0).any(axis=3)
    assert (pairs.sum(axis=2) <= 2).all() and (pairs.sum(axis=2) == 2).mean() > 0.5


# ---- representability, the recheck and its refusals --------------------------------------------------------------------


def test_representable_is_an_odd_part_under_2_24():
    v = [0, 1, -1, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 2, -((1 << 25) + 2), 3 << 40, ((1 << 24) + 1) << 3]
    assert fx.representable(v).tolist() == [True, True, True, True, True, False, True, False, True, False]


@pytest.mark.parametrize("c", [cfg(), cfg(grid="e4m3_half8", c0="binade", K=128),
                               cfg(fmt="nvf4", grid="e2m1_x2", dist="MAXSIGN", c0="nobits", K=128),
                               cfg(fmt="sp_mxf4", grid="e2m1_x2", dist="CANCEL", c0="negzero", K=256, nrec=2)],
                         ids=lambda c: c.name)
def test_recheck_accepts_exact_records(c):
    units = fx.word_units(c.c0_word, 2 * fx.GRIDS[fx.GRID[c.grid]].s)
    h = fx.recheck(c, units, records(c))
    assert h["outputs"] == 128 * c.nrec and not (h["ref_disagree"] or h["flag_disagree"] or h["bad_rep"])
    assert h["order_diff"] == 0 and h["unwritten"] == 0


def test_recheck_counts_a_wrong_word_and_an_order_that_disagrees():
    c = cfg(grid="e4m3_half8", c0="binade", K=128)
    units = fx.word_units(c.c0_word, 2)

    def flip(r):
        r["d"][5, 0] ^= 1
        r["d"][9, 2] ^= 1
    h = fx.recheck(c, units, records(c, flip))
    assert h["bad_rep"] == 1 and h["order_diff"] == 2 and h["exact"] == 127
    assert h["in_binade"] == 128 and h["binade_linear"] == 127


def test_recheck_sees_a_device_reference_that_is_not_the_hosts():
    c = cfg()

    def bump(r):
        r["ref"][7] += 1
    assert fx.recheck(c, 0, records(c, bump))["ref_disagree"] == 1


def test_refusal_names_the_unwritten_tag_count_and_records():
    c = cfg()
    row = {"tag": fx.TAG | 0, "out": 128 * 128, "c0_units": 0}
    assert fx.refusal(c, row, records(c)) == ""
    assert "didn't write" in fx.refusal(c, {**row, "tag": 0xA5A5A5A5}, records(c))
    assert "expected" in fx.refusal(c, {**row, "out": 5}, records(c))
    assert "started from" in fx.refusal(c, {**row, "c0_units": 1}, records(c))

    def poison(r):
        r["flags"][3] = 0xA5A5A5A5
    assert "poisoned" in fx.refusal(c, row, records(c, poison))


def test_sass_refusal_wants_one_expected_mma_a_kernel():
    good = {f"_Z12exact_kernelILi{i}EEv6ParamsP6CountsP3Rec": [fx.EXPECT_MMA[f]] for i, f in enumerate(fx.FMTS)}
    assert fx.sass_refusal(good) == ""
    two = {**good, "_Z12exact_kernelILi4EEv6ParamsP6CountsP3Rec": [fx.EXPECT_MMA["nvf4"]] * 2}
    assert "nvf4" in fx.sass_refusal(two)
    assert "e4m3_kind" in fx.sass_refusal({**good, "_Z12exact_kernelILi1EEv6ParamsP6CountsP3Rec": ["HMMA.16816.F32"]})


# ---- the silicon-model replay ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c", [cfg(K=64), cfg(fmt="mxf4", grid="e2m1_x2", K=128),
                               cfg(fmt="sp_nvf4", grid="e2m1_x2", dist="MAX", K=128)], ids=lambda c: c.name)
def test_replay_agrees_with_exact_records(c):
    r = fx.replay(c, records(c))
    assert r["outputs"] == r["agree"] == 128 and r["inexact"] == 0


def test_replay_counts_inexact_words_the_model_predicts():
    """E4M3 with C0 one binade over: the adder holds the integer sum, FP32 truncates its odd values (RZ)."""
    c = cfg(c0="over", K=32)
    A, B = fx.values(c, 0)
    ref = A @ B.T + fx.word_units(c.c0_word, 0)
    rz = np.array([[fx.word(float(v)) if v % 2 == 0 else fx.word(float(v - 1)) for v in row] for row in ref])

    def truncate(r):
        r.reshape(16, 8)["d"] = rz[:, :, None]
    r = fx.replay(c, records(c, truncate))
    assert r["agree"] == 128 and 0 < r["inexact"] == r["inexact_agree"] == int((ref % 2).sum())


# ---- verdicts and rates ------------------------------------------------------------------------------------------------


def _row(c, exact, **kw):
    base = {"out": 128 * c.tiles, "exact": exact, "rep_all": exact, "bad_rep": 0, "unrep": 128 * c.tiles - exact,
            "bad_unrep": 128 * c.tiles - exact, "rz_final": 0, "rn_final": 0, "order_diff": 0, "max_partial": c.K,
            "zero": 1, "negzero": 0, "refused": "", "replay": None,
            "host": {"outputs": 128, "bad_rep": 0, "in_binade": 0, "binade_linear": 0}}
    return base | kw


def test_verdicts_find_the_last_exact_k_and_the_first_inexact_one():
    cs = [cfg(K=k) for k in (32, 64, 128, 256)]
    rows = [_row(cs[0], 128 * 128), _row(cs[1], 128 * 128), _row(cs[2], 100), _row(cs[3], 128 * 128)]
    v = fx.verdicts(rows, cs)["e4m3/e4m3_int16"]
    s = v["series"]["U/zero"]
    assert s["exact_through_K"] == 64 and s["first_inexact_K"] == 128 and v["refused"] == []
    rows[1]["refused"] = "the kernel didn't write its counts"
    v = fx.verdicts(rows, cs)["e4m3/e4m3_int16"]
    assert v["refused"] == [cs[1].name] and v["series"]["U/zero"]["exact_through_K"] == 32


def test_limits_take_the_worst_draw_for_each_c0():
    cs = [cfg(dist=d, K=k) for d in ("U", "EDGE") for k in (32, 64, 128)] + [cfg(c0="over", K=32)]
    exact = {("EDGE", 128), ("U", 32), ("U", 64), ("U", 128), ("EDGE", 32), ("EDGE", 64)}
    rows = [_row(c, 128 * 128 if (c.dist, c.K) in exact and c.c0 == "zero" else 1) for c in cs]
    rows[4]["exact"] = 1
    t = fx.limits(fx.verdicts(rows, cs))
    assert "| e4m3/e4m3_int16 | zero | 32 | 64 (EDGE) |" in t and "| e4m3/e4m3_int16 | over | 0 | 32 (U) |" in t


def test_rates_are_macs_per_sm_clock_against_imma():
    rows = [{"probe": "imma_s8", "rate": 0.25, "sm_mhz": 2000, "distinct_sms": 188},
            {"probe": "omma_sp_nvf4", "rate": 0.2, "sm_mhz": 2000, "distinct_sms": 188},
            {"probe": "lds32", "rate": 1.0}]
    r = fx.rates(rows)
    assert set(r) == {"imma_s8", "omma_sp_nvf4"}
    assert r["imma_s8"]["macs_sm_clk"] == 1024 and r["omma_sp_nvf4"]["vs_imma"] == pytest.approx(3.2)
    assert r["imma_s8"]["tmacs"] == pytest.approx(1024 * 188 * 2000e6 / 1e12)
    with pytest.raises(SystemExit):
        fx.rates(rows[1:])


# ---- the binary --------------------------------------------------------------------------------------------------------


CUDA = Path("/usr/local/cuda-13.0/bin")


def _tool(name: str) -> str | None:
    return str(CUDA / name) if (CUDA / name).exists() else shutil.which(name)


@pytest.mark.slow
@pytest.mark.skipif(not (_tool("nvcc") and _tool("cuobjdump")), reason="needs nvcc and cuobjdump (CUDA 13)")
def test_build_dump_and_sass(tmp_path, monkeypatch):
    """What the CPU tests can't: fp_exact.cu's own generator draws what the host regenerates, and today's nvcc still
    compiles each format's chain to its one MMA."""
    binary = tmp_path / "fp_exact"
    subprocess.run([_tool("nvcc"), "-std=c++20", "-O2", "-gencode", "arch=compute_120a,code=sm_120a", "-o", str(binary),
                    str(HERE / "fp_exact.cu")], check=True)
    for c in (cfg(dist="RAMP", K=64), cfg(dist="EDGE", K=128), cfg(grid="e4m3_quarter4", dist="CANCEL", c0="negzero", K=128),
              cfg(fmt="sp_nvf4", grid="e2m1_x2", K=256), cfg(fmt="mxf4nv", grid="e2m1_int", dist="MAXSIGN", K=128)):
        (tmp_path / "c.txt").write_text(c.line() + "\n")
        subprocess.run([str(binary), str(tmp_path / "c.txt"), "--dump", str(tmp_path / "d.bin")], check=True)
        d = np.fromfile(tmp_path / "d.bin", dtype="<i4")
        A, B = fx.values(c, 0)
        assert (d[:16 * c.K].reshape(16, c.K) == A).all() and (d[16 * c.K:].reshape(8, c.K) == B).all(), c.name
    monkeypatch.setenv("PATH", f"{Path(_tool('cuobjdump')).parent}{os.pathsep}{os.environ['PATH']}")
    assert fx.sass_refusal(fx.sass_mmas(binary)) == ""
