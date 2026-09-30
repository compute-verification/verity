"""Pearl-C's attack pricing: the limb counts, the 2:4 test and the FP16 test on hand-built codes, and a formed family."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("pouw_pearlc_attacks", Path(__file__).parents[1] / "pearlc_attacks.py")
A = importlib.util.module_from_spec(spec)
sys.modules["pouw_pearlc_attacks"] = A
spec.loader.exec_module(A)


def test_limbs_follow_the_bit_span():
    ones = np.full((1, 32), 0x38, np.uint8)                     # 1.0 everywhere: one bit, one limb of each kind
    assert A.int8_limbs(ones).tolist() == [[1]] and A.fp4_limbs(ones).tolist() == [[1, 1]]
    wide = ones.copy()
    wide[0, 0], wide[0, 1] = 0x7E, 0x01                          # 448 = 0b1110 * 2^5 and 2^-9: bits 8 down to -9
    assert A.int8_limbs(wide).tolist() == [[3]]                 # 18 bits and a sign in signed int8 limbs
    assert A.fp4_limbs(wide).tolist() == [[3, 1]]               # windows at -9, 0 and 6: three E2M1 limbs


def test_sparse_and_fp16_tests():
    z = np.zeros((16, 64), np.uint8)
    assert A.sparse24_fragments(z) == 1.0
    z[0, 0:3] = 0x38
    assert A.sparse24_fragments(z) == 0.0
    assert A._fp16_exact(np.array([2048.0, 2049.0, 2050.0, 65504.0, 65505.0, 2.0 ** -24, 3 * 2.0 ** -25])).tolist() == \
        [True, False, True, True, False, True, False]


def test_no_route_undercuts_the_atom_at_public_rates():
    A.C.set_atom("sm120-e4m3-k32")
    try:
        m = A.measure("gaussian", 1024, 16, 16, run=1, line_norm=16.0)
    finally:
        A.C.set_atom("hopper-e4m3-k32")
    priced = A.price(m, A.PRICES["sm120-public"])
    assert m["rules_ok"] and m["int8_limbs_A"] > 1.5 and m["fp4_limbs_A"] > 2
    assert all(v["saving_ub"] == 0.0 for v in priced.values() if "saving_ub" in v)
