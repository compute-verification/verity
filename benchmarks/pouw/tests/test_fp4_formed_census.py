"""fp4_formed_census.py's own helpers by hand; its forming is GPU 5's twin, which that branch's tests hold to the reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp4_formed_census as FC  # noqa: E402


def test_frag_cost_skips_passes_once_or_pays_dense():
    live = np.zeros((32, 128), bool)
    live[16:, 64:] = np.tile([True, False, True, False], (16, 16))
    live[:16, 64:] = np.tile([True, True, True, False], (16, 16))
    c = FC.frag_cost(live)
    assert (c["frag_skip"], c["frag_one_pass"]) == (0.5, 0.25)
    assert c["cost_per_mac"] == (0 + 0 + 1.0 + 0.5) / 4
    live[:16, 64:] = False
    live[3, 70] = True                                                  # one change in 1,024: 8.46/1,024 beats a pass
    c = FC.frag_cost(live, fp32_op=8.46)
    assert c["cost_per_mac"] == 1.0 / 4 and c["cost_per_mac_with_cuda_core"] == (8.46 / 1024 + 0.5) / 4
    assert c["frag_cuda_core_cheaper"] == 0.25


def test_bf16_words_round_to_nearest_even():
    x = np.array([[1.0, 1 + 2**-8, 1 + 3 * 2**-8, -2.5]], np.float32)
    assert FC.bf16_words(x).tolist() == [[0x3F800000, 0x3F800000, 0x3F820000, 0xC0200000]]
