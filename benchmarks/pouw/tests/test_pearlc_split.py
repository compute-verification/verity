"""Pearl-C's channel split is one pinned rule: ties in the offence count go to the lower channel, and S stays in budget."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("pouw_pearlc_headline", Path(__file__).parents[1] / "pearlc_headline.py")
H = importlib.util.module_from_spec(spec)
sys.modules["pouw_pearlc_headline"] = H
spec.loader.exec_module(H)

SPIKES = [3, 5, 11, 13, 19, 21]


def rows() -> tuple[np.ndarray, np.ndarray]:
    k = 128
    base = np.where(np.arange(k) % 2 == 0, 1.0, -1.0).astype(np.float32)
    X = np.tile(base, (8, 1))
    X[:, SPIKES] = 20.0
    return X, np.tile(base, (4, 1))


def test_split_breaks_ties_to_the_lower_channel():
    X, W = rows()
    S, _ = H.split_channels(X, W, 1.0, budget=4)
    assert S == SPIKES[:4]


def test_split_never_exceeds_its_budget():
    X, W = rows()
    S, _ = H.split_channels(X, W, 1.0, budget=3)
    assert S == SPIKES[:3]
    S, trace = H.split_channels(X, W, 1.0, budget=64)
    assert sorted(S) == SPIKES and trace[-1]["failing_rows"] == 0
