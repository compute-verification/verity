"""The harness's numpy route-U shift of the kernel's unbiased checkpoints equals the scheme's (Lean `checkedU_eq`)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.ml.protocols.pouw.schemes import ncp

spec = importlib.util.spec_from_file_location("pouw_kernels", Path(__file__).parents[1] / "kernels" / "__init__.py")
K = importlib.util.module_from_spec(spec)
sys.modules["pouw_kernels"] = K
spec.loader.exec_module(K)


def test_numpy_shift_equals_the_reference():
    m, k, n = 16, 1088, 32
    s, rng = ncp.NCP(n_min=16), random.Random(3)
    ctx = Context(b"\x01" * 32, Matmul(0, Shape(m, k, n), 9), b"\x02" * 32, b"\x03" * 32)
    a = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(m)]
    b = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(n)]
    ops = s.encode(ctx, s.noise(ctx), a, b, Tile(tuple(range(m)), tuple(range(n))))
    unbiased = np.array(ncp.checked_unbiased(ops), dtype=np.uint32)
    got = K.route_u_checkpoints(unbiased, np.array(ops.y, dtype=np.int64), s.beta)
    assert got.tolist() == s.checked(ctx, ops)
