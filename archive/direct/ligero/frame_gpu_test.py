"""The GPU committer (``frame_gpu.py``) byte for byte against the host tree builders of ``auth.py`` / ``hashauth.py`` and
the keyed-BLAKE3 leaf's ``carry_values`` / ``native`` (skipped without a CUDA device)."""
from __future__ import annotations

import numpy as np
import pytest

from . import auth, frame_gpu, hashauth
from .leaf import blake3 as b3
from .leaf import registry

pytestmark = pytest.mark.skipif(not frame_gpu.available(), reason="needs the GPU committer (cupy, a CUDA device)")

BIND = {n: bytes([i + 1]) * 32 for i, n in enumerate(auth.TREE_NAMES)}


@pytest.fixture
def host(monkeypatch):
    """Run the host builders in the block (the GPU path off)."""
    class _Off:
        def __enter__(self):
            monkeypatch.setattr(frame_gpu, "ENABLED", False)

        def __exit__(self, *a):
            monkeypatch.setattr(frame_gpu, "ENABLED", True)
    return _Off()


def _same(t_gpu, t_host):
    assert t_gpu.ref == t_host.ref
    assert len(t_gpu.levels) == len(t_host.levels)
    for a, b in zip(t_gpu.levels, t_host.levels):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize("count", [1, 3, 4096, 70001])
def test_u16_word_tree_is_the_host_tree(count, host):
    words = np.random.default_rng(count).integers(0, 1 << 16, count).astype(np.uint16)
    t = auth.build_word_tree("a", BIND["a"], words)
    with host:
        h = auth.build_word_tree("a", BIND["a"], words, procs=1)
    _same(t, h)


@pytest.mark.parametrize("y_bits", [16, 32])
def test_output_word_tree_is_the_host_tree(y_bits, host):
    words = np.random.default_rng(y_bits).integers(0, 1 << y_bits, 4096, dtype=np.int64)
    t = hashauth.build_word_tree("y", BIND["y"], words, y_bits)
    with host:
        h = hashauth.build_word_tree("y", BIND["y"], words, y_bits)
    _same(t, h)


@pytest.mark.parametrize("word_bits,dtype", [(8, np.uint8), (16, np.uint16), (16, np.int64)])
def test_blake3_row_tree_is_the_host_tree(word_bits, dtype, host):
    scheme = registry.get("blake3")
    rows = np.random.default_rng(word_bits).integers(0, 1 << word_bits, (300, b3.K_VU)).astype(dtype)
    for name, role in (("a", 1), ("b", 2)):
        t, dg = hashauth.build_row_tree(name, BIND[name], rows, word_bits, role, scheme)
        with host:
            h, dh = hashauth.build_row_tree(name, BIND[name], rows, word_bits, role, scheme)
        _same(t, h)
        assert np.array_equal(dg, dh)
        with pytest.raises(ValueError, match="chain digests"):
            bad = dh.copy()
            bad[5, 3] ^= 1
            hashauth.build_row_tree(name, BIND[name], rows, word_bits, role, scheme, digests=bad)


@pytest.mark.parametrize("word_bits,steps", [(8, 24), (8, 12), (8, 48), (16, 48), (16, 24), (16, 96)])
def test_blake3_chain_is_carry_values(word_bits, steps):
    import torch
    scheme = registry.get("blake3")
    n = 37
    rows = np.random.default_rng(steps * word_bits).integers(0, 1 << word_bits, (n, b3.K_VU))
    words = torch.as_tensor(rows, device="cuda").reshape(n * steps, b3.K_VU // steps)
    for role in (1, 2):
        car, dg = scheme.row_sponges(words, word_bits=word_bits, role=role, row_words=b3.K_VU, n=n, steps=steps)
        assert np.array_equal(car.cpu().numpy(), scheme.carry_values(rows, word_bits=word_bits, role=role, steps=steps))
        assert np.array_equal(dg.cpu().numpy(), scheme.native(rows, word_bits=word_bits, role=role))


def test_device_tree_levels_come_to_the_host_once():
    words = np.arange(1 << 18, dtype=np.uint16)
    t = auth.build_word_tree("b", BIND["b"], words)
    assert t._levels is None
    root = t.root
    sib = auth.multiproof_siblings(t, np.arange(100, 300))
    assert t._levels is not None and t.root == root and len(sib) > 0


@pytest.mark.parametrize("word_bits,dt", [(8, np.uint8), (16, np.uint16)])
def test_committed_rows_pinned_reach_the_device_intact(word_bits, dt):
    import torch
    from .relchain import _rows_to_device, narrow_rows
    rows = [np.random.default_rng(r).integers(0, 1 << word_bits, 96) for r in range(40)]
    host = narrow_rows(rows, word_bits, pin=True)
    assert host.dtype == dt and torch.from_numpy(host.view(np.uint8)).is_pinned()
    narrow: dict = {}
    wide = _rows_to_device(host, "cuda", narrow, "a")
    assert np.array_equal(wide.cpu().numpy(), np.stack(rows)) and "a" in narrow
    with pytest.raises(ValueError, match="out of range"):
        narrow_rows([np.array([0, 1 << word_bits])], word_bits)
