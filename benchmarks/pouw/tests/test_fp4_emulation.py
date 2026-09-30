"""fp4_emulation.py's vectorised atom and BF16 step are bit-exact against verity.ml.tc, the hot accumulator separates the
emulation routes as claimed, and fp4_emulation_gpu.py packs what its kernels read."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
import fp4_emulation as E  # noqa: E402
import fp4_emulation_gpu as GPU  # noqa: E402


def test_vectorised_atom_and_bf16_step_match_the_pinned_models():
    rep = E.selftest(argparse.Namespace(seed=20260930))
    assert rep.pop("ok") is True
    for name, r in rep.items():
        assert r["native_mismatches"] == 0 and r["bf16_fold_mismatches"] == 0, (name, r)


def test_e2m1_and_ue4m3_round_to_nearest_even():
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0])
    assert (E.halves(E.e2m1_rn(ties)) / 2).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0]
    assert E.ue4m3_rn(np.array([1.0, 448.0, 1000.0, 2.0 ** -9, 1.0625, 1.1875])).tolist() == [0x38, 0x7E, 0x7E, 0x01, 0x38, 0x3A]


def test_hot_accumulator_breaks_the_exact_routes_and_keeps_the_step_floor():
    p = E.Problem("gaussian", 8, 8, 1024, 0.25, 5)
    zero = np.zeros((8, 8), np.uint32)
    native = p.native(zero)
    assert (p.exact()[0] == native).all() and (p.bf16_fold(zero) == native).all()
    hot = E.hot_accumulator(p, 10)
    native_hot = p.native(hot)
    assert (p.step_floor(hot) == native_hot).all()
    assert (p.exact(hot)[0] != native_hot).mean() > 0.9 and (p.bf16_fold(hot) != native_hot).mean() > 0.9


def test_gpu_fragments_have_the_kernels_word_counts_and_integer_arithmetic():
    pool = GPU.make_pool("gaussian", 3)
    for route in GPU.ROUTES:
        frags, lanec = GPU.pack(route, pool)
        assert frags.shape == (GPU.TILES, 32, *GPU.SHAPE[route]) and lanec.shape == (GPU.TILES, 32, 8)
        assert GPU.dry_run(route, pool, frags, lanec, 4) in (0, -1), route


def _cuda_bin() -> Path | None:
    nvcc = GPU.nvcc_bin()
    return Path(nvcc).parent if nvcc else None


@pytest.mark.skipif(_cuda_bin() is None or not (_cuda_bin() / "nvdisasm").is_file(),
                    reason="needs nvcc 12.8+ with cuobjdump and nvdisasm beside it")
def test_sass_gate(tmp_path):
    b = GPU.build(tmp_path)
    assert b["sass"]["ok"], b["sass"]
