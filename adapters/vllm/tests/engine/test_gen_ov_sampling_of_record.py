"""`test_gen_ov_sampling`'s Gumbel lane noise against the GPU probe's dump (`out/gen/lanes/ov-sampling/runs/probe`, written on a
pod, which no tracked file holds). Skipped where it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_gumbel_lane_noise_is_bit_exact_against_the_gpu_probe():
    """The pod probe dumped `_gumbel_sample_kernel`'s noise (tl.rand -> log1pf/logf) for seeds/positions of the B1 runs;
    the lane's NumPy transcription and the Definition's per-lane primitive must reproduce every word."""
    import numpy as np
    from verity.definitions.registry.sampling import GumbelNoiseLane, GumbelStreamKey

    from tests.engine.gumbel_oracle import gumbel_noise_exact
    probe = Path(__file__).resolve().parents[2] / "out/gen/lanes/ov-sampling/runs/probe"
    cmp = probe / "compare.json"
    if not cmp.exists():
        pytest.skip("probe artifacts not fetched")
    rep = json.loads(cmp.read_text())
    assert rep["all_bit_exact"] is True, rep
    dumps = sorted(probe.glob("*.npz"))
    if not dumps:
        pytest.skip("probe .npz not fetched (compare.json is the recorded verdict)")
    d = np.load(dumps[0])
    seed, pos = int(d["seed"]), int(d["pos"])
    gpu = np.asarray(d["gumbel"], dtype=np.float32)
    ours = gumbel_noise_exact(seed, pos, gpu.shape[0])
    assert np.array_equal(gpu.view(np.uint32), ours.view(np.uint32))
    key = GumbelStreamKey.evaluate(seed & ((1 << 64) - 1), pos & ((1 << 64) - 1))
    for v in (0, 1, 7, gpu.shape[0] - 1):
        assert GumbelNoiseLane.evaluate(key, v) == int(gpu.view(np.uint32)[v])
