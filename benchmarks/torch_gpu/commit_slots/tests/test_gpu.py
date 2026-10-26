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


@pytest.mark.parametrize("leaf_minb", [1, 2])
def test_either_leaf_kernel_gives_the_reference_s_roots(leaf_minb: int) -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(5))
    plan = G.Plan(CTX, layout, leaf_minb=leaf_minb)
    assert G.commit(plan, 3, KEY, _cuda(out), 3) == S.commit(CTX, layout, 3, KEY, out, 3)


@pytest.mark.parametrize("temp_bytes", [1, 4 * 64 * 25 * 2], ids=["1-slot-passes", "2-slot-passes"])
def test_passes_on_the_device_give_the_reference_s_roots(temp_bytes: int) -> None:
    """Passes over a few slots at a time (each pass's row table moved on its first slot, its salts' nonces from there) and the
    frontier folded and closed on the device: three H200 layers (75 units in groups of 25 rows) at 5 slots."""
    layout = S.h200_layout(layers=3)
    out = S.random_outputs(layout, 5, np.random.default_rng(4))
    plan = G.Plan(CTX, layout, temp_bytes=temp_bytes)
    assert plan.per_pass(plan.groups[0].shape, 5) == (1 if temp_bytes == 1 else 2)
    assert G.commit(plan, 6, KEY, _cuda(out), 5) == S.commit(CTX, layout, 6, KEY, out, 5)
