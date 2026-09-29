"""CPU contract tests for `ncp2_gpu_bench`'s accounting: the bytes each leaf hash absorbs per Qwen2.5-0.5B forward."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("ncp2_gpu_bench", Path(__file__).parents[1] / "ncp2_gpu_bench.py")
B = importlib.util.module_from_spec(spec)
sys.modules["ncp2_gpu_bench"] = B
spec.loader.exec_module(B)


def test_padding_and_the_tree():
    assert [B.absorbed("sha256", n) for n in (0, 55, 56, 119, 120)] == [64, 64, 128, 128, 192]
    assert [B.absorbed("sha512", n) for n in (111, 112)] == [128, 256]
    assert [B.absorbed("turboshake128", n) for n in (167, 168)] == [168, 336]
    assert [B.tree_nodes(p) for p in (1, 2, 3, 5, 8)] == [0, 1, 2, 4, 7]


def test_a_tiles_words_are_three_quarters_of_a_byte_per_useful_mac():
    for m in B.ROWS:
        acc = B.forward_accounting(m)
        words = sum(B.LAYERS * (m // 16) * (n // 16) * 3 * k // 16 * 256 * 4 for k, n in B.PD.QWEN05)
        assert words / acc["macs"] == 0.75
        assert 0.75 < acc["tile_bytes_per_mac"] < 0.76


def test_at_sixteen_rows_the_per_call_y_strips_hash_as_much_as_the_tile_words():
    ab = B.forward_accounting(16)["absorbed"]["sha256"]
    assert abs(ab["form_y strips (committed per call)"] / ab["tile words"] - 1) < 1e-3
    assert round(ab["total"] / 1e9, 2) == 9.57


class _T:
    """A numpy array standing in for a torch tensor: the calls the bench makes."""

    def __init__(self, a):
        self.a = a

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    @property
    def T(self):
        return _T(self.a.T)


class _Torch:
    bfloat16, uint8 = "bf16", "u8"

    class cuda:
        @staticmethod
        def empty_cache():
            pass

    @staticmethod
    def from_numpy(a):
        return _T(a)

    @staticmethod
    def randn(*shape, **_kw):
        return _T(np.zeros(shape, np.float32))

    @staticmethod
    def randint(lo, hi, shape, **_kw):
        return _T(np.zeros(shape, np.uint8))

    @staticmethod
    def mm(a, b):
        return _T(a.a @ b.a)


class _RouteU:
    """#218's `ncp_route_u` contract in numpy: C = the last checkpoint, and every 16-deep checkpoint [M, N, K/16] from
    c0 = −128·colsum(Y)."""

    @staticmethod
    def ncp_route_u(xu, ys, _cfg):
        x, y = xu.a.astype(np.int64), ys.a.view(np.int8).astype(np.int64)
        steps = x.shape[1] // 16
        per = np.einsum("mtd,ntd->mnt", x.reshape(x.shape[0], steps, 16), y.reshape(y.shape[0], steps, 16))
        ck = (np.cumsum(per, axis=2) - 128 * y.sum(axis=1)[None, :, None]) & 0xFFFFFFFF
        ck = ck.astype(np.uint32).view(np.int32)
        return _T(ck[:, :, -1].copy()), _T(ck)


def test_the_route_u_gate_reads_the_kernels_checkpoint_layout(monkeypatch):
    small = [("qwen05 small", c) for n, c in B.PD.gate_calls() if n == "five-strips"]
    monkeypatch.setattr(B.PD, "gate_calls", lambda qwen=False: small)
    assert B.gate_route_u(_RouteU, _Torch) == [{"call": "qwen05 small", "checkpoints_equal": True}]


def test_the_committer_and_hash_gates_run_through_the_host_build(monkeypatch):
    calls = B.PD.gate_calls()
    monkeypatch.setattr(B.PD, "gate_calls", lambda qwen=False: calls)
    monkeypatch.setattr(B.PD, "commit_call_timed", lambda i, s, c: (B.PD.commit_call(i, s, c), 0.0))
    host = B.PD.hash_many
    monkeypatch.setattr(B.PD, "hash_many", lambda kind, msgs, out=None, ds=0x1F, on="host": host(kind, msgs, out, ds))
    assert all(r["difference"] is None for r in B.gate_committer())
    assert B.gate_hashes() == dict.fromkeys(B.PD.HASHES, "pass")


class _Dev:
    """The device library's timing entries, writing a time linear in the bytes into their out-parameter."""

    @staticmethod
    def ncp2_ceiling(code, grid, threads, iters, ref):
        ref._obj.value = 1e12
        return 0

    @staticmethod
    def ncp2_leaf_time(code, units, words, reps, ref):
        ref._obj.value = units * (27 + 4 * words) / 5e8
        return 0

    @staticmethod
    def ncp2_f1_time(n, k, reps, ref):
        ref._obj.value = n * k / 5e8
        return 0


class _Timed:
    @staticmethod
    def ncp_route_u(xu, ys, _cfg):
        return None


def test_the_timings_and_projection_cover_every_linear_and_hash(monkeypatch):
    monkeypatch.setattr(B.PD, "device", lambda: _Dev)
    monkeypatch.setattr(_Torch, "mm", staticmethod(lambda a, b: None))
    monkeypatch.setitem(sys.modules, "gemm_bench", type(sys)("gemm_bench"))
    sys.modules["gemm_bench"].timed = lambda fn, reps: (fn(), 0.5)[1]
    t = B.timings(_Timed, _Torch, 2)
    assert len(t["linears"]) == len(B.ROWS) * len(B.PD.QWEN05)
    assert set(t["from_memory_bytes_per_s"]) == set(B.HASH) and abs(t["from_memory_bytes_per_s"]["sha256"] / 5e11 - 1) < 0.01
    p = B.projection(t)
    assert set(p) == set(B.ROWS) and all(p[m][h]["unfused_x_bf16"] > p[m][h]["fused_x_bf16"] > 0 for m in B.ROWS for h in B.HASH)
