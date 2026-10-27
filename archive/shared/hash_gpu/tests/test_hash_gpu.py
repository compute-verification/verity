"""Bit-exact tests of the GPU Merkle builders against the CPU references (tiny inputs; GPU parts skip on a laptop)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # kernels

from hash_gpu import poseidon2_ref as p2
from hash_gpu import reference

try:
    import cupy as cp

    GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    cp, GPU = None, False

try:
    import blake3  # noqa: F401

    HAVE_BLAKE3 = True
except ImportError:
    HAVE_BLAKE3 = False

P = p2.P


def test_poseidon2_matches_plonky3_vectors():
    p2.check_vectors()


def test_poseidon2_grain_constants_shape_and_range():
    for w in (16, 24):
        inst = p2.grain_constants(w)
        assert len(inst.ext_init) == 4 and len(inst.ext_final) == 4
        assert len(inst.internal) == p2.ROUNDS[w][1]
        assert all(0 <= c < P for r in inst.ext_init + inst.ext_final for c in r)
        out = inst.permute([1] * w)
        assert len(out) == w and out != [1] * w


def test_external_matrix_is_plonky3_layout():
    m = p2.external_matrix(16)
    assert m[0][:8] == [4, 6, 2, 2, 2, 3, 1, 1]
    s = list(range(16))
    dense = [sum(m[i][j] * s[j] for j in range(16)) % P for i in range(16)]
    assert dense == p2.mds_external(s)


def test_reference_sha256_chunked_equals_plain_for_short_columns():
    m = np.arange(24 * 3, dtype=np.uint32).reshape(24, 3) % P
    for j in range(3):
        assert reference.leaf(m, j, "sha256-chunked") == reference.leaf(m, j, "sha256")


def test_reference_path_verifies():
    m = (np.arange(40 * 5, dtype=np.uint64) * 2654435761 % P).astype(np.uint32).reshape(40, 5)
    for h in ("sha256", "sha256-chunked"):
        lv = reference.tree(m, h)
        for j in range(5):
            path, idx = [], j
            for lvl in lv[:-1]:
                sib = idx ^ 1
                path.append(lvl[sib] if sib < len(lvl) else None)
                idx >>= 1
            assert reference.verify_path(lv[0][j], j, path, lv[-1][0], h)


def _matrix(R, N, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, P, size=(R, N), dtype=np.uint32)


@pytest.mark.skipif(not GPU, reason="no CUDA device")
@pytest.mark.parametrize("hash", ["sha256", "sha256-chunked", "blake3"])
@pytest.mark.parametrize("R,N", [(1, 1), (5, 3), (16, 7), (300, 6), (1200, 5), (2100, 3)])
def test_gpu_tree_matches_reference(hash, R, N):
    if hash == "blake3" and not HAVE_BLAKE3:
        pytest.skip("blake3 package missing")
    from hash_gpu import merkle

    m = _matrix(R, N)
    t = merkle.commit(cp.asarray(m), hash)
    ref = reference.tree(m, hash)
    assert len(t.levels) == len(ref)
    for lvl, rl in zip(t.levels, ref):
        got = [merkle.digest_bytes(lvl[i], hash) for i in range(lvl.shape[0])]
        assert got == rl
    paths = t.open(range(N))
    for j in range(N):
        path = [None if s is None else merkle.digest_bytes(s, hash) for s in paths[j]]
        assert reference.verify_path(ref[0][j], j, path, ref[-1][0], hash)


@pytest.mark.skipif(not GPU, reason="no CUDA device")
@pytest.mark.parametrize("width", [16, 24])
@pytest.mark.parametrize("mds", ["blocks", "fp64", "int8"])
def test_poseidon2_torch_permutation(width, mds):
    import torch

    from hash_gpu.poseidon2_torch import Poseidon2Torch

    inst = p2.rng_constants(width)
    eng = Poseidon2Torch(inst, mds=mds)
    inp, exp = p2.PLONKY3_VECTORS[width]
    rng = np.random.default_rng(3)
    extra = rng.integers(0, P, size=(31, width), dtype=np.int64)
    states = torch.tensor(np.vstack([np.array(inp, dtype=np.int64)[None], extra]), device="cuda")
    out = eng.permute(states).cpu().numpy()
    assert out[0].tolist() == exp
    for k in range(1, 4):
        assert out[k].tolist() == inst.permute(extra[k - 1].tolist())


@pytest.mark.skipif(not GPU, reason="no CUDA device")
@pytest.mark.parametrize("hash", ["poseidon2-16", "poseidon2-24"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("R,N", [(1, 1), (8, 2), (9, 3), (40, 5), (200, 3), (24, 9)])
def test_poseidon2_tree_matches_reference(hash, backend, R, N):
    import torch

    from hash_gpu import merkle

    m = _matrix(R, N, seed=7)
    if backend == "torch":
        t = merkle.commit(torch.tensor(m.astype(np.int64), device="cuda"), hash, backend="torch", mds="int8")
    else:
        eng = merkle.poseidon2_engine(hash, "cuda")
        eng.leaf_tree = lambda mm, _f=eng.leaf_tree: _f(mm, col_block=4)      # exercise the column blocking
        t = merkle.commit(cp.asarray(m), hash, engine=eng)
    ref = reference.tree(m, hash)
    assert len(t.levels) == len(ref)
    for lvl, rl in zip(t.levels, ref):
        got = [merkle.digest_bytes(lvl[i], hash) for i in range(lvl.shape[0])]
        assert got == rl


def test_sha256_reference_is_hashlib_of_column_bytes():
    m = _matrix(10, 2)
    assert reference.leaf(m, 1, "sha256") == hashlib.sha256(m[:, 1].astype("<u4").tobytes()).digest()


@pytest.mark.skipif(not GPU, reason="no CUDA device")
@pytest.mark.parametrize("width", [16, 24])
def test_poseidon2_cuda_permutation_and_tree(width):
    from hash_gpu.poseidon2_cuda import Poseidon2Cuda

    inst = p2.rng_constants(width)
    eng = Poseidon2Cuda(inst)
    inp, exp = p2.PLONKY3_VECTORS[width]
    rng = np.random.default_rng(5)
    extra = rng.integers(0, P, size=(300, width), dtype=np.int64)
    states = np.vstack([np.array(inp, dtype=np.int64)[None], extra]).astype(np.uint32)
    out = eng.permute(cp.asarray(states)).get()
    assert out[0].tolist() == exp
    for k in range(1, 4):
        assert out[k].tolist() == inst.permute(extra[k - 1].tolist())
    hash = f"poseidon2-{width}"
    for R, N in [(1, 1), (9, 3), (40, 5), (200, 3)]:
        m = _matrix(R, N, seed=11)
        ref = reference.tree(m, hash, p2inst=inst)
        lv = eng.leaf_tree(cp.asarray(m))
        assert [row.astype("<u4").tobytes() for row in lv.get()] == ref[0]
        levels = [lv]
        while levels[-1].shape[0] > 1:
            levels.append(eng.merkle_level(levels[-1]))
        for got, rl in zip(levels, ref):
            assert [row.astype("<u4").tobytes() for row in got.get()] == rl
