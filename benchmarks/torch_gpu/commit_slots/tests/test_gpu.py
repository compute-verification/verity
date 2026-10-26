"""On a CUDA device: the call on the engine's device build gives the reference's roots (the recorded runs are check.py's,
at the frozen shape)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

import slots as S
import slots_gpu as G

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                reason="needs a CUDA device and nvcc")

CTX = bytes(range(64))
KEY = bytes(range(100, 132))


def _cuda(out: list[list[np.ndarray]]) -> list[list[object]]:
    return [[torch.from_numpy(a).cuda() for a in tensors] for tensors in out]


@pytest.mark.parametrize("slots", [1, 5, 300])
def test_the_edge_layout(slots: int) -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, slots, np.random.default_rng(slots))
    assert G.commit(G.Plan(CTX, layout), 1, KEY, _cuda(out), slots) == G.commit(G.Plan(CTX, layout, "host"), 1, KEY, out, slots)


def test_two_h200_layers() -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(3))
    assert G.commit(G.Plan(CTX, layout), 2, KEY, _cuda(out), 3) == S.commit(CTX, layout, 2, KEY, out, 3)
