"""Pearl-C4's block-scale step: the cheaper reciprocals equal RN_f32(1/s) on every valid byte, and the integer encode equals
``ue4m3_rn`` on a spread of inputs (``ue4m3_scale.py`` runs the exhaustive 2^31 check)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import ue4m3_scale as U  # noqa: E402


def test_reciprocals_exact_on_every_byte():
    for f in (lambda b: U.LUT[b], U.exp_mant, U.newton):
        assert [b for b in U.BYTES if U.bits(f(b)) != U.bits(U.REF[b])] == []


def test_lut_floor_entry():
    assert U.bits(U.LUT[0]) == U.bits(U.REF[1]) == 0x44000000        # 2^-9's reciprocal, 512


def test_int_encode_matches_ue4m3_rn():
    rng = np.random.default_rng(20260930)
    u = np.concatenate([rng.integers(0, 0x7F800000, 1 << 18, dtype=np.uint32),
                        np.arange(0x3B800000, 0x3D000000, 1 << 8, dtype=np.uint32),    # around 2^-9 .. 2^-5, ties included
                        np.arange(0x43D00000, 0x43F00000, 1 << 8, dtype=np.uint32)])   # around 448 .. 480, saturation
    ref = np.asarray(U.ue4m3_rn(U.from_bits_arr(u).astype(np.float64)), np.int64)
    assert int((U.int_encode(u).astype(np.int64) != ref).sum()) == 0
