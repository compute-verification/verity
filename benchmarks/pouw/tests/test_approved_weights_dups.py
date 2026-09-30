"""The registration near-duplicate check compares codes regardless of scale (docs/pouw/approved-weights.md §8j).

A row that is a power-of-two multiple of another survives the keyed rotation and requantization bit for bit, with only
its per-row scale changed, so `structure-free/rot-*` leaves such copies to the near-duplicate rule. These tests pin that
the census (`approved_weights/aw_census.py`, the verifier's registration check) catches them in FP8 and NVFP4: the codes
it compares carry no row scale and no block-scale bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "approved_weights"))
import aw_census as C  # noqa: E402


def _rows(seed=0, n=48, k=256):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    w[5] = 2.0 * w[3]
    w[9] = -2.0 * w[7]
    w[11] = 3.0 * w[1]
    return w


def _rotate(w, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(w.shape[1], w.shape[1], generator=g))
    return w @ (q * torch.sign(torch.diagonal(r)))


@pytest.mark.parametrize("rotated", [False, True])
def test_fp8_power_of_two_copies_are_duplicates(rotated):
    w = _rows()
    w = _rotate(w) if rotated else w
    c = C.fp8_codes(w)
    cn = C.canon(c, False)
    assert torch.equal(cn[5], cn[3]) and torch.equal(cn[9], cn[7])
    assert cn.shape[0] - torch.unique(cn, dim=0).shape[0] >= 2
    best, flagged = C.near_dups(c, 128, 15, False)
    assert bool((best[5] == 1.0).all()) and bool((best[9] == 1.0).all())
    assert bool((best[11] > 0).all())            # 3x is not bit exact, but its groups are near-duplicates


@pytest.mark.parametrize("rotated", [False, True])
def test_nvfp4_power_of_two_copies_are_duplicates(rotated):
    w = _rows()
    w = _rotate(w) if rotated else w
    c = C.nvfp4_codes(w)
    cn = C.canon(c, True)
    assert torch.equal(cn[5], cn[3]) and torch.equal(cn[9], cn[7])
    k = c.shape[1]
    best, flagged = C.near_dups(c, k, int(k / (2 * C.FADD)), True)
    assert float(best[5, 0]) == 1.0 and float(best[9, 0]) == 1.0
    assert float(best[11, 0]) > 0.0
