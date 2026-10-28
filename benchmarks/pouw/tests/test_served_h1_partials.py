"""Kernels' prover side of option A (`served_h1/partials.py`, #3095/#3101) held to the order the service states
(`pouw.leaf_lanes`): row l of a drawn leaf's partials is lane l at the stated rows, columns and key block, its words in
`leaf_lanes.word_order`, each `h1zy.lane_partial` (the row PouwH1LaneZy_v1 checks), and a dead leaf's rows are zero. Skips
until served_h1/partials.py is on the tree."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.ml.protocols.pouw.circuit import h1zy as Z

SERVED = Path(__file__).resolve().parents[1] / "served_h1"
RC, K = (32, 32), 64


def _partials():
    if not (SERVED / "partials.py").exists():
        pytest.skip("served_h1/partials.py (#3101) is not on the tree")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(SERVED))
    spec = importlib.util.spec_from_file_location("served_h1_partials", SERVED / "partials.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m, torch


def test_kernels_partials_are_the_stated_lanes_in_the_stated_word_order():
    PT, torch = _partials()
    g = torch.Generator().manual_seed(20261021)
    a = torch.randint(-128, 128, (2 * RC[0], K), generator=g, dtype=torch.int8)
    b = torch.randint(-128, 128, (RC[1], K), generator=g, dtype=torch.int8)
    salt, ra, rb = bytes(range(64)), b"\x03" * 64, b"\x04" * 64
    win = PT.Window([b], [rb], [0], 2, {0: PT.Step([a], [ra], RC[0], salt)})
    live = PT.lane_partials(win, PT.Leaf(0, 0, 0), "cpu")
    assert live.shape == (LL.lanes(*RC), LL.words(K))
    au, bu = a.view(torch.uint8).tolist(), b.view(torch.uint8).tolist()
    for ln in range(LL.lanes(*RC)):
        want = Z.lane_partial([au[i] for i in LL.lane_rows(*RC, ln)], [bu[j] for j in LL.lane_cols(*RC, ln)], [1] * RC[0],
                              salt, 0, (ra, rb), ln, RC)
        assert live[ln].tolist() == want, f"lane {ln}"
    dead = PT.lane_partials(win, PT.Leaf(0, 0, 1), "cpu")
    assert dead.shape == live.shape and not dead.any(), "a dead leaf's partials are zero"
    assert np.array_equal(PT.combine(live), np.bitwise_xor.reduce(live, axis=0))
