"""fp4_cov70b_gpu.py's port against `fp4_f1_pricing.linear` itself, on CPU torch, on the same draws: synthetic linears
with honest rows, heavy-tailed ones near the cap (term 4), the census's cheaper-computation families and witness-dead
rows (R1), so the tile reject and `volunteer` act; at every weight band and at sampled ones, over several of `linear`'s
draw chunks and device batches. Every count must be equal and every figure within RTOL. Needs the census's sources
(bc-f5bf55c8's fill job, sha256-pinned in SOURCES) in FP4_COV70B_SRC."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = os.environ.get("FP4_COV70B_SRC", "")
pytestmark = pytest.mark.skipif(not SRC, reason="needs the census's sources in FP4_COV70B_SRC")


def _linear(F, m, n, k, rng):
    import torch
    X = rng.standard_normal((m, k)) * np.exp(rng.standard_normal(k) * 0.8)[None, :]
    X[:, rng.choice(k, 4, replace=False)] *= 40
    X[:64] = rng.standard_normal((64, k)) * np.exp(rng.standard_normal(k) * 3)[None, :]
    X[192:256] = rng.standard_normal((64, k)) * np.exp(rng.standard_normal(k) * 4)[None, :]
    X[64:128] = F.spike_rows(64, k, 8.9, rng)
    X[128:150] = F.parked_rows(22, k, 8.9, rng)
    X[150:160] = F.stride_rows(10, k, rng, 9.48)
    X[200:204] = np.sign(rng.standard_normal((4, k))) * 2.0
    X[5:8] = rng.standard_normal((3, k)) * np.exp(rng.standard_normal(k) * 6)[None, :]
    W = rng.standard_normal((n, k)) * 0.02
    W[rng.integers(0, n, 30), rng.integers(0, k, 30)] *= 30
    W[64:128] = 0.01 * F.spike_rows(64, k, 8.9, rng)
    W[192:210] = 0.01 * F.parked_rows(18, k, 8.9, rng)
    W[256:320] = 0.01 * F.narrow_cell(64, k, rng)
    for r in rng.choice(n, 6, replace=False):
        W[r] = np.sign(rng.standard_normal(k)) * 0.05

    def dead(R, s):
        x = np.sign(rng.standard_normal((R, k))) * s
        x[:, ::8] *= 1e-3
        return x
    X[64:67] = dead(3, 1.0)
    X[160:180] = dead(20, 1.0)
    W[128:192] = dead(64, 0.03)
    W[320:331] = dead(11, 0.03)
    W[576:578] = dead(2, 0.03)
    return X.astype(np.float32), torch.from_numpy(W.astype(np.float32)).to(torch.bfloat16)


@pytest.mark.parametrize("max_cols,chunk", [(0, 4_000_000), (0, 150_000), (4, 70_000)])
def test_the_port_agrees_with_the_census(max_cols, chunk):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import fp4_cov70b_gpu as G
    F = G.load(SRC)
    K = G.Port("cpu", batch_elems=300_000)
    X, W = _linear(F, 256, 640, 512, np.random.default_rng(7))
    ref = F.linear(X, W, np.random.default_rng([1, 2, 3]), "certified", max_cols, 0, "nv", False, chunk)
    got = G.linear(K, X, W, np.random.default_rng([1, 2, 3]), max_cols, chunk)
    bad, counts, figures, worst = G.compare(json.loads(json.dumps(ref)), json.loads(json.dumps(got)))
    assert not bad and counts >= 50 and figures >= 60 and worst < G.RTOL
    assert ref["norm"]["tiles_rejected"] > 0 and ref["norm"]["volunteer_a"] > 0 and ref["norm"]["volunteer_b"] > 0
    assert ref["r1"]["rows_excluded_a"] > 0 and got["vol_tiles"] > 0
