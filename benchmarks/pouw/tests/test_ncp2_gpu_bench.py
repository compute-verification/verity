"""CPU contract tests for `ncp2_gpu_bench`'s accounting: the bytes each leaf hash absorbs per Qwen2.5-0.5B forward."""
from __future__ import annotations

import dataclasses
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
        words = sum(B.LAYERS * (m // 16) * (n // 16) * 3 * k // 16 * 256 * 4 for k, n in B.QWEN05)
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


def _check_tile(m, n, cfg):
    bm, bn = B.ROUTE_U_TILES[cfg]
    if m % bm or n % bn:
        raise RuntimeError(f"shape ({m}, ?, {n}) is not a multiple of the tile {bm} x {bn} x 64")


class _RouteU:
    """#218's `ncp_route_u` contract in numpy: C = the last checkpoint, and every 16-deep checkpoint [M, N, K/16] from
    c0 = −128·colsum(Y)."""

    @staticmethod
    def ncp_route_u(xu, ys, cfg):
        _check_tile(xu.a.shape[0], ys.a.shape[0], cfg)
        x, y = xu.a.astype(np.int64), ys.a.view(np.int8).astype(np.int64)
        steps = x.shape[1] // 16
        per = np.einsum("mtd,ntd->mnt", x.reshape(x.shape[0], steps, 16), y.reshape(y.shape[0], steps, 16))
        ck = (np.cumsum(per, axis=2) - 128 * y.sum(axis=1)[None, :, None]) & 0xFFFFFFFF
        ck = ck.astype(np.uint32).view(np.int32)
        return _T(ck[:, :, -1].copy()), _T(ck)


def test_the_route_u_gate_reads_the_kernels_checkpoint_layout_under_every_config_it_times(monkeypatch):
    small = [("linear small", c) for n, c in B.PD.gate_calls() if n == "five-strips"]
    monkeypatch.setattr(B.PD, "gate_calls", lambda shapes=(): small)
    monkeypatch.setattr(B, "CFG0_GATE", (128, 64, 256))
    got = B.gate_route_u(_RouteU, _Torch)
    assert [(r["call"], r["cfg"]) for r in got] == [("linear small", 2), ("linear k=64 n=256 m=128", 0),
                                                    ("linear k=64 n=256 m=128", 1), ("linear k=64 n=256 m=128", 2)]
    assert all(r["checkpoints_equal"] for r in got)
    timed = {B._cfgs(m, n)[0] for m in B.ROWS for _k, n in B.QWEN05}
    gated = {c for m in (16, 128) for _k, n in B.QWEN05 for c in B._cfgs(m, n)}
    assert timed <= gated


def test_the_committer_and_hash_gates_run_through_the_host_build(monkeypatch):
    calls = B.PD.gate_calls()
    monkeypatch.setattr(B.PD, "gate_calls", lambda shapes=(): calls)
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
    def ncp_route_u(xu, ys, cfg):
        _check_tile(xu.a.shape[0], ys.a.shape[0], cfg)


def test_the_timings_and_projection_cover_every_linear_and_hash(monkeypatch):
    monkeypatch.setattr(B.PD, "device", lambda: _Dev)
    monkeypatch.setattr(_Torch, "mm", staticmethod(lambda a, b: None))
    monkeypatch.setitem(sys.modules, "gemm_bench", type(sys)("gemm_bench"))
    sys.modules["gemm_bench"].timed = lambda fn, reps: (fn(), 0.5)[1]
    t = B.timings(_Timed, _Torch, 2)
    assert len(t["linears"]) == len(B.ROWS) * len(B.QWEN05)
    assert set(t["from_memory_bytes_per_s"]) == set(B.HASH) and abs(t["from_memory_bytes_per_s"]["sha256"] / 5e11 - 1) < 0.01
    p = B.projection(t)
    assert set(p) == set(B.ROWS) and all(p[m][h]["unfused_x_bf16"] > p[m][h]["fused_x_bf16"] > 0 for m in B.ROWS for h in B.HASH)


def test_a_linear_that_fails_is_recorded_and_the_rest_still_run(monkeypatch):
    monkeypatch.setattr(B.PD, "device", lambda: _Dev)
    monkeypatch.setattr(_Torch, "mm", staticmethod(lambda a, b: None))
    monkeypatch.setitem(sys.modules, "gemm_bench", type(sys)("gemm_bench"))
    sys.modules["gemm_bench"].timed = lambda fn, reps: (fn(), 0.5)[1]
    class Failing:
        @staticmethod
        def ncp_route_u(xu, ys, cfg):
            if (xu.a.shape[0], ys.a.shape[0]) == (16, 9728):
                raise RuntimeError("shape (16, 2688, 9728) is not a multiple of the tile")

    saved = []
    t = B.timings(Failing, _Torch, 2, save=lambda part: saved.append(len(part.get("linears", []))))
    bad = [r for r in t["linears"] if "error" in r]
    assert bad and all(r["n"] == 9728 and r["m"] == 16 for r in bad) and len(t["linears"]) == len(B.ROWS) * len(B.QWEN05)
    assert saved[0] == 0 and saved[-1] == len(t["linears"])
    assert 16 not in B.projection(t) and 256 in B.projection(t)


def test_main_on_the_pod_path_writes_every_piece(monkeypatch, tmp_path, capsys):
    """`main` end to end with torch, #218's extension and the device library faked, the committer and hashes on the host
    build: what the pod runs, but for the GPU."""
    calls = [c for c in B.PD.gate_calls() if c[0] == "five-strips"]
    monkeypatch.setattr(B.PD, "gate_calls", lambda shapes=(): calls + [("linear small", calls[0][1])])
    monkeypatch.setattr(B, "CFG0_GATE", (128, 64, 256))
    monkeypatch.setattr(B.PD, "commit_call_timed", lambda i, s, c: (B.PD.commit_call(i, s, c), 0.0))
    host = B.PD.hash_many
    monkeypatch.setattr(B.PD, "hash_many", lambda kind, msgs, out=None, ds=0x1F, on="host": host(kind, msgs, out, ds))
    monkeypatch.setattr(B.PD, "device", lambda: _Dev)
    torch = type(sys)("torch")
    for name in ("bfloat16", "uint8", "from_numpy", "randn", "randint"):
        setattr(torch, name, getattr(_Torch, name))
    torch.mm = lambda a, b: None
    torch.cuda = type("cuda", (), {"empty_cache": staticmethod(lambda: None), "get_device_name": staticmethod(lambda: "fake")})
    monkeypatch.setitem(sys.modules, "torch", torch)
    kernels = type(sys)("kernels")
    kernels.load = lambda: type("ext", (), {"ncp_route_u": staticmethod(lambda xu, ys, cfg: _RouteU.ncp_route_u(xu, ys, cfg)
                                                                          if xu.a.shape[1] <= 3 * 64 else _Timed.ncp_route_u(xu, ys, cfg))})
    monkeypatch.setitem(sys.modules, "kernels", kernels)
    monkeypatch.setitem(sys.modules, "gemm_bench", type(sys)("gemm_bench"))
    sys.modules["gemm_bench"].timed = lambda fn, reps: (fn(), 0.5)[1]
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["ncp2_gpu_bench.py", "--reps", "2", "--out", str(out)])
    B.main()
    import json

    res = json.loads(out.read_text())
    assert set(res) == {"accounting", "host_gate", "device", "gates", "timings", "projection"}
    assert set(res["gates"]) == {"hashes", "committer", "route_u"} and len(res["timings"]["linears"]) == 12
    assert "GATES " in capsys.readouterr().out


def test_the_fused_gate_runs_through_the_cpu_twin(monkeypatch):
    """`gate_fused` with the device swapped for `ncp2_fast.cuh` on the CPU (the low-byte tiles from `lowbyte_leaf`)."""
    calls = [c for c in B.PD.gate_calls() if c[0] in ("five-strips", "dequant-edges")]
    monkeypatch.setattr(B.PD, "gate_calls", lambda shapes=(): calls)
    PC = B.PD.PC

    def timed(i, s, call, on="device", flags=B.PD.HASH):
        rec = B.PD.commit_call(i, s, call, on="twin")
        if flags & B.PD.LOWBYTE:
            nat = B.PN.Native(call, "ncp-v2")
            lo, hi = nat.base["tile"], nat.base["dequant"]
            leaves = rec.leaves[:32 * lo] + b"".join(B.PD.lowbyte_leaf(rec.opening(u)) for u in range(lo, hi)) + rec.leaves[32 * hi:]
            rec = dataclasses.replace(rec, leaves=leaves, root=PC.units_root(leaves))
        return rec, 1.0

    monkeypatch.setattr(B.PD, "commit_call_timed", timed)
    got = B.gate_fused()
    assert [r["call"] for r in got] == ["five-strips", "dequant-edges"]
    assert all(r["difference"] is None and r["lowbyte_difference"] is None for r in got)


class _NT:
    """A numpy array standing in for the torch tensors `fused_timings` makes."""

    def __init__(self, a):
        self.a = a

    def view(self, _dtype):
        return self


class _Ev:
    def record(self):
        pass

    def elapsed_time(self, _other):
        return 10.0


class _FT:
    int8, int16, int32, float32 = np.int8, np.int16, np.int32, np.float32

    class Generator:
        def __init__(self, device=None):
            pass

        def manual_seed(self, _s):
            return self

    class cuda:
        Event = staticmethod(lambda enable_timing=True: _Ev())
        synchronize = staticmethod(lambda: None)
        empty_cache = staticmethod(lambda: None)

    @staticmethod
    def randint(lo, hi, shape, dtype=None, device=None, generator=None):
        return _NT(np.zeros(shape, dtype))

    @staticmethod
    def full(shape, v, dtype=None, device=None):
        return _NT(np.full(shape, v, dtype))

    @staticmethod
    def zeros(n, dtype=None, device=None):
        return _NT(np.zeros(n, dtype))

    @staticmethod
    def arange(n, dtype=None, device=None):
        return _NT(np.arange(n, dtype=dtype))

    @staticmethod
    def equal(a, b):
        return a is b or np.array_equal(a.a, b.a)


def test_the_fused_timings_cover_every_shape_and_mode(monkeypatch):
    class Committer:
        def __init__(self, scheme, salt, flags):
            self.flags, self.n = flags, 0

        def __call__(self, a, s, b, t, bias, forward, position, weight, rowids, colids):
            self.n += 1
            return _NT(np.zeros((a.a.shape[0], b.a.shape[0]), np.int16))

        def tile_ms(self):
            n, self.n = self.n, 0
            return 2.0 * n, n

    monkeypatch.setattr(B.PD, "FusedCommitter", Committer)
    out = B.fused_timings(_FT, 3)
    assert len(out["calls"]) == len(B.FUSED_ROWS) * len(B.QWEN05)
    assert all(r["y_equal"] and r["word_tile_ms"] == 2.0 and r["nohash_ms"] == 10.0 / 3 for r in out["calls"])
    assert set(out["forward_m2048"]) == {"nohash", "word", "lowbyte", "nohash_tile", "word_tile", "lowbyte_tile"}


def test_the_vllm_split_adds_up_to_the_slowdown():
    """`vllm_bench.split`: ncp2's time is the control's tile kernel (GEMM) + (ncp2 − control) (hashing) + the rest."""
    spec_v = importlib.util.spec_from_file_location("vllm_bench", Path(__file__).parents[1] / "vllm_bench.py")
    V = importlib.util.module_from_spec(spec_v)
    spec_v.loader.exec_module(V)
    bf16 = {"prefill_2048_s": 0.02, "decode_b32_s": 1.0}
    nohash = {"prefill_2048_s": 0.5, "prefill_2048_tile_ms": 300.0, "decode_b32_s": 4.0, "decode_b32_tile_ms": 1000.0}
    full = {"prefill_2048_s": 2.0, "prefill_2048_tile_ms": 1500.0, "decode_b32_s": 30.0}
    low = {"prefill_2048_s": 1.0, "decode_b32_s": 12.0}
    out = V.split(bf16, full, nohash, low)
    p = out["prefill_2048"]
    assert abs(p["gemm_s"] + p["hashing_s"] + p["else_s"] - p["ncp2_s"]) < 1e-12 and p["slowdown"] == 100.0
    assert abs(p["gemm_x_bf16"] + p["hashing_x_bf16"] + p["else_x_bf16"] - p["slowdown"]) < 1e-9
    assert (p["gemm_s"], p["hashing_s"]) == (0.3, 1.5) and p["lowbyte_slowdown"] == 50.0
    assert p["lowbyte_status"] == "gamma unproved: not a result"
    assert out["decode_b32"]["slowdown"] == 30.0 and abs(out["decode_b32"]["hashing_x_bf16"] - 26.0) < 1e-12
