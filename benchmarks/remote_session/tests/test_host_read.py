"""The library entry `DeviceStore.host_pass` takes for a host segment: `rs_pass` with the arguments a shard's pass
gives it, at the store's receipt CTAs, and `host_read.py --unbound`. The library is a stub that records its calls."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import device  # noqa: E402


class Tensor:
    """What `host_pass` reads of a tensor: its address; `[slot]` is another tensor."""

    def __init__(self, at: int) -> None:
        self.at = at

    def data_ptr(self) -> int:
        return self.at

    def __getitem__(self, k: int) -> Tensor:
        return Tensor(self.at + 1000 * (k + 1))


class Library:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name: str):
        def call(*args):
            self.calls.append((name, args))
            return 0
        return call


def store(host_ctas: int):
    """A store with one shard (segment 0) and one host segment (segment 1) on one GPU, as far as `host_pass` reads."""
    s = object.__new__(device.DeviceStore)
    s.lib = SimpleNamespace(so=Library())
    s.gpus = [SimpleNamespace(host_main=SimpleNamespace(cuda_stream=71), host_side=SimpleNamespace(cuda_stream=72))]
    s.seg_gpu = [0, 0]
    s.seg_ring = [None, {k: Tensor(a) for k, a in (("ctr", 10), ("ys", 20), ("dig", 30), ("up", 40), ("kept", 50))}]
    s.data_g, s.n, s.base = [None, Tensor(5)], [1 << 20, 1 << 16], [0, 1 << 20]
    s.h0, s.chunk, s.host_ctas, s.host_consumers = 6, 2048, host_ctas, 8
    return s


def test_a_host_pass_is_rs_pass_as_a_shard_calls_it():
    for ctas in (16, 32):
        s = store(ctas)
        assert s.host_pass(1, Tensor(7), 2) == 0
        assert s.lib.so.calls == [("rs_pass", (5, 1 << 16, 1 << 20, 7, 10, 20, 30, 40, 3050, 6, 2048, ctas, 0, 0, None,
                                               None, 71, 72, 8))]
    assert len(device.SESSION["rs_pass"]) == 19


def test_an_unbound_tier_sets_no_numa_policy(monkeypatch):
    """`--unbound` is for a machine that refuses set_mempolicy: after it `numa_bound` makes no call at all."""
    import ctypes

    import host_read

    monkeypatch.setattr(device, "numa_bound", device.numa_bound)   # put back when the test ends
    host_read.unbind()

    def refused(*a, **k):
        raise AssertionError("an unbound tier loaded libc for a policy")

    monkeypatch.setattr(ctypes, "CDLL", refused)
    with device.numa_bound(1):
        pass
