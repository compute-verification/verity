"""fp4_merge_gpu.py: the torch port of the census's forming and merge rate equals the numpy census bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp4_census as C  # noqa: E402
import fp4_merge_gpu as G  # noqa: E402


def test_selftest_on_cpu():
    assert G.selftest("cpu", rows=16, k=256)["selftest"] == "pass"


def test_device_noise_has_the_census_law():
    x = torch.as_tensor(np.random.default_rng(0).standard_normal((64, 1024)))
    gen = torch.Generator().manual_seed(1)
    z = (G.noisy_row(x, 0.25, gen) - x) / (0.25 * torch.sqrt((x ** 2).mean(1, keepdim=True)))
    assert torch.allclose((z ** 2).mean(1), torch.ones(64, dtype=torch.float64))


def test_full_partner_search_is_the_census_at_every_block():
    rng = np.random.default_rng(3)
    codes, sc, _ = C.quantize(C.noisy(C.family("outlier-in-block", 12, 512, rng)[0], "mxf4", "row", 0.25, rng), "mxf4")
    N, E = G.block_ints(torch.as_tensor(codes), torch.as_tensor(sc), "mxf4")
    got = G.partners(N, E, N.abs().amax(-1) > 0, "mxf4", chunk=64)
    Nn, En = C.block_ints(codes, sc, "mxf4")
    assert got == C.merge_partners(Nn, En, np.abs(Nn).max(-1) > 0, "mxf4", max_blocks=10 ** 9)
