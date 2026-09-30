"""CPU contract tests for the native route-U tile wrapper (ordering and framing, no CUDA required)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("pouw_native_kernels", Path(__file__).parents[1] / "kernels" / "__init__.py")
K = importlib.util.module_from_spec(spec)
sys.modules["pouw_native_kernels"] = K
spec.loader.exec_module(K)


class Tensor:
    def __init__(self, a):
        self.a = np.asarray(a)
        self.ndim, self.shape = self.a.ndim, self.a.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Ext:
    def __init__(self, c, ck):
        self.c, self.ck, self.cfg = Tensor(c), Tensor(ck), None

    def ncp_route_u(self, x, y, cfg):
        self.cfg = cfg
        return self.c, self.ck


def test_step_major_digest_and_tile_order():
    c = np.arange(16 * 32, dtype=np.int32).reshape(16, 32) - 200
    ck = np.arange(16 * 32 * 3, dtype=np.uint32).reshape(16, 32, 3)
    ext = Ext(c, ck)
    got = K.ncp_route_u_tiles(ext, Tensor(np.zeros((16, 48), np.uint8)), Tensor(np.zeros((32, 48), np.uint8)))
    assert ext.cfg == 2 and len(got) == 2
    for tile, j in zip(got, (0, 16)):
        block = np.ascontiguousarray(ck[:, j:j + 16].transpose(2, 0, 1)).astype("<u4").tobytes()
        assert tile[0] == hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + block).digest(32)
        assert tile[1] == c[:, j:j + 16].astype(np.int64).tolist()


def test_shape_rejected_before_extension():
    ext = Ext(np.zeros((16, 16), np.int32), np.zeros((16, 16, 3), np.uint32))
    try:
        K.ncp_route_u_tiles(ext, Tensor(np.zeros((8, 48), np.uint8)), Tensor(np.zeros((16, 48), np.uint8)))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid route-U tile shape was accepted")


if __name__ == "__main__":
    test_step_major_digest_and_tile_order()
    test_shape_rejected_before_extension()
