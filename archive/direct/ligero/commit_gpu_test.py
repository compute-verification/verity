"""``commit_gpu`` (the protocol-specialised Blake3 Merkle kernels) against ``verity/ml/kernels/hash_gpu`` and the
``blake3`` package: every level, every leaf, the root, and the openings ``merkle.open_device`` produces -- at the
prover's shapes (bf16-hopper 3328 x 65536, fp8-hopper 3432, bf16-ampere 3552, fp4 1619 x 16384) and at the edges
(one chunk, a partial last chunk, tiny trees)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from . import merkle
from .field import P

cupy = pytest.importorskip("cupy")
if not torch.cuda.is_available() or not merkle.gpu_available():
    pytest.skip("CUDA + cupy + hash_gpu required", allow_module_level=True)

from . import commit_gpu  # noqa: E402
from hash_gpu import merkle as gm  # noqa: E402

DEV = torch.device("cuda")


def _matrix(R: int, N: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, P, (R, N), generator=g, dtype=torch.int64).to(torch.int32).to(DEV)


@pytest.mark.parametrize("R,N", [(3328, 65536), (3432, 65536), (3552, 65536), (1619, 16384), (100, 1024), (256, 1024),
                                 (257, 1024), (16, 2), (4096, 4096), (3, 4), (300, 8)])
def test_levels_leaves_root_equal_reference(R, N):
    import blake3
    U = _matrix(R, N, R + N)
    ref = gm.commit(U, "blake3")
    t = commit_gpu.commit(U)
    assert len(ref.levels) == len(t.levels)
    for a, b in zip(ref.levels, t.levels):
        assert a.shape == b.shape and bool(cupy.all(a == b))
    Uc = U.cpu().numpy().astype(np.uint32)
    for c in (0, 1, N // 2, N - 1):
        assert blake3.blake3(np.ascontiguousarray(Uc[:, c]).tobytes()).digest() == gm.digest_bytes(t.levels[0][c], "blake3")
    assert gm.digest_bytes(ref.root, "blake3") == gm.digest_bytes(t.root, "blake3")


def test_commit_device_and_openings_verify():
    U = _matrix(3328, 65536, 7)
    root, tree = merkle.commit_device(U)
    assert isinstance(tree, commit_gpu.Tree)
    cols = [0, 1, 2, 65535, 32768, 12345, 4097]
    paths = merkle.open_device(tree, cols)
    Uc = U.cpu().numpy().astype(np.uint32)
    for c, path in zip(cols, paths):
        assert len(path) == 16 and all(len(s) == 32 for s in path)
        assert merkle.verify_path(root, c, Uc[:, c], path)
    bad = list(paths[0]); bad[3] = bytes(32)
    assert not merkle.verify_path(root, cols[0], Uc[:, cols[0]], bad)
    # the shared kernels and the host path agree with the fused one
    merkle.USE_FUSED_COMMIT = False
    try:
        root2, _ = merkle.commit_device(U)
    finally:
        merkle.USE_FUSED_COMMIT = True
    assert root2 == root
    assert merkle.commit(Uc[:, :64].T.copy())[0] == merkle.commit_device(U[:, :64].contiguous())[0]


def test_stream_argument():
    U = _matrix(512, 2048, 11)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        root_s, tree_s = merkle.commit_device(U, stream=s)
    s.synchronize()
    root, _ = merkle.commit_device(U)
    assert root == root_s
