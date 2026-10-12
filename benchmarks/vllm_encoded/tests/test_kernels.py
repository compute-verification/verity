import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kernels as K


def sketched(nbs: int, nchunks: int, sb: int, t: int, seed: int) -> K.TensorLinear:
    """A TensorLinear's sketch geometry over a random store on the CPU (no library: only `sketch_vectors` reads it)."""
    tlib = object.__new__(K.TensorLib)
    tlib.sb, tlib.sketch = sb, {"t": t}
    lin = object.__new__(K.TensorLinear)
    lin.tlib, lin.n, lin.nchunks = tlib, 16 * nbs, nchunks
    g = torch.Generator().manual_seed(seed)
    lin.store = torch.randint(0, 256, (nbs * nchunks * sb,), generator=g, dtype=torch.int64).to(torch.uint8).view(
        torch.int16)
    return lin


@pytest.mark.parametrize("nbs,nchunks,sb,t", [(5, 2, 8256, 2), (7, 1, 8192, 3), (3, 3, 600, 1)])
def test_sketch_vectors_are_sketchings(nbs, nchunks, sb, t):
    skt, lin = K._sketching(), sketched(nbs, nchunks, sb, t, nbs * sb + t)
    for groups in sorted({1, 2, 3, nbs}):
        ref = skt.vectors(lin.bytes.numpy(), nbs, nchunks, sb, groups, t)
        assert ref.shape[0] == 16 * groups * nchunks * skt.tiles_per_cta(nbs, groups, t)
        vs = list(range(ref.shape[0]))
        assert np.array_equal(lin.sketch_vectors(vs, groups), ref)


@pytest.mark.parametrize("n,kd,t", [(10240, 8192, 8), (8192, 28672, 8), (57344, 8192, 4), (16, 512, 8)])
def test_tag_vectors_bound_every_launch(n, kd, t):
    skt, tlib = K._sketching(), object.__new__(K.TensorLib)
    tlib.sb, tlib.sketch = 8256, {"t": t}
    nbs, nch = n // 16, kd // 512
    most = max(16 * groups * nch * skt.tiles_per_cta(nbs, groups, t) for groups in range(1, nbs + 1))
    assert tlib.tag_vectors(n, kd) == most
