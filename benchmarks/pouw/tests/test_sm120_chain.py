"""The sm_120 chain run's CPU side: its twin is the census's chain on the sm_120 atom, and its SASS gate reads cuobjdump."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("pouw_sm120_chain_run", Path(__file__).parents[1] / "sm120_chain" / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["pouw_sm120_chain_run"] = R
spec.loader.exec_module(R)


def test_twin_is_the_census_chain_and_int8_is_exact():
    R.census.set_atom(R.ATOM)
    try:
        a, b = R.standin(4, 256, 1), R.standin(3, 256, 2)
        ii, jj = np.array([0, 1, 3, 2]), np.array([0, 2, 1, 1])
        w = R.twin(a, b, ii, jj)
        for p, (i, j) in enumerate(zip(ii, jj)):
            want = np.float32(R.census.chain(a[i:i + 1], b[j:j + 1], 4)[4][0]).view(np.uint32)
            assert w["honest"][p] == want
            assert w["int8"][p] == np.uint32(np.int32(int(a[i].view(np.int8).astype(np.int64) @ b[j].view(np.int8).astype(np.int64))))
    finally:
        R.census.set_atom("hopper-e4m3-k32")


def test_same_words_takes_signed_zeros_as_equal():
    x = np.array([0x80000000, 0, 0x3F800000], np.uint32)
    y = np.array([0, 0, 0x3F800001], np.uint32)
    assert R.same_words(x, y).tolist() == [True, True, False]
