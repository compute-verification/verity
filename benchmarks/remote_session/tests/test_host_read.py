"""A host segment's read by name (`device.host_read_plan`) and the library entry `DeviceStore.host_pass` takes for it:
`pass` is `rs_pass` with the arguments a shard's pass gives it, every other name `rs_host_pass` with the name's terms,
and a name outside the grammar is refused when the store is built, before a pass. The library is a stub that records
its calls."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import device  # noqa: E402
from device import HOST_FORMS, HostRead, host_read_plan  # noqa: E402


@pytest.mark.parametrize("name, plan", [
    ("ahead1", HostRead(1, HOST_FORMS["load"], 0, 0)),
    ("ahead3-keep", HostRead(3, HOST_FORMS["keep"], 0, 0)),
    ("ahead4", HostRead(4, HOST_FORMS["load"], 0, 0)),
    ("prefetch-ahead2", HostRead(2, HOST_FORMS["prefetch"], 0, 0)),
    ("feed4", HostRead(0, HOST_FORMS["load"], 4, 128)),
    ("feed2-lead8-keep", HostRead(0, HOST_FORMS["keep"], 2, 8)),
    ("ahead1-feed4-lead32", HostRead(1, HOST_FORMS["load"], 4, 32)),
])
def test_a_name_is_its_terms(name, plan):
    assert host_read_plan(name) == plan


def test_pass_is_no_plan():
    assert host_read_plan("pass") is None


@pytest.mark.parametrize("name", [
    "", "ahead", "ahead0", "ahead5", "ahead2-ahead1", "feed0", "feed", "lead8", "ahead1-lead8", "keep", "prefetch",
    "feed4-prefetch", "keep-prefetch-ahead1", "pass-ahead1", "ahead1-", "ahead-1", "ahead--1", "ahead1+keep",
    "ahead1.keep", "ahead1 keep", "feed4-lead0", "Ahead1", "keep2-ahead1",
])
def test_a_name_outside_the_grammar_is_refused(name):
    with pytest.raises(ValueError, match="host read"):
        host_read_plan(name)


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


def store(host_read: str):
    """A store with one shard (segment 0) and one host segment (segment 1) on one GPU, as far as `host_pass` reads."""
    s = object.__new__(device.DeviceStore)
    s.lib = SimpleNamespace(so=Library())
    s.gpus = [SimpleNamespace(host_main=SimpleNamespace(cuda_stream=71), host_side=SimpleNamespace(cuda_stream=72),
                              host_feed=SimpleNamespace(cuda_stream=73))]
    s.seg_gpu = [0, 0]
    s.seg_ring = [None, {k: Tensor(a) for k, a in
                         (("ctr", 10), ("ys", 20), ("dig", 30), ("up", 40), ("kept", 50), ("feed", 60))}]
    s.data_g, s.n, s.base = [None, Tensor(5)], [1 << 20, 1 << 16], [0, 1 << 20]
    s.h0, s.chunk, s.host_ctas, s.host_consumers, s.host_read = 6, 2048, 16, 8, host_read
    return s


def test_pass_is_rs_pass_as_a_shard_calls_it():
    s = store("pass")
    assert s.host_pass(1, Tensor(7), 2) == 0
    assert s.lib.so.calls == [("rs_pass", (5, 1 << 16, 1 << 20, 7, 10, 20, 30, 40, 3050, 6, 2048, 16, 0, 0, None, None,
                                           71, 72, 8))]
    assert len(device.SESSION["rs_pass"]) == 19


@pytest.mark.parametrize("name, tail", [
    ("ahead2", (2, HOST_FORMS["load"], 60, 0, 0)),
    ("ahead1-prefetch", (1, HOST_FORMS["prefetch"], 60, 0, 0)),
    ("feed4-lead8-keep", (0, HOST_FORMS["keep"], 60, 4, 8)),
    ("ahead3-feed2", (3, HOST_FORMS["load"], 60, 2, 128)),
])
def test_a_plan_is_rs_host_pass_with_its_terms(name, tail):
    s = store(name)
    assert s.host_pass(1, Tensor(7), 0) == 0
    assert s.lib.so.calls == [("rs_host_pass", (5, 1 << 16, 1 << 20, 7, 10, 20, 1050, 6, 2048, 16, 8, *tail, 71, 72,
                                                73))]
    assert len(s.lib.so.calls[0][1]) == len(device.SESSION["rs_host_pass"])


def test_the_forms_are_session_cu_s():
    """`HOST_FORMS` and `HOST_AHEAD_MAX` are session.cu's enum and constant, in its order."""
    text = (HERE.parent / "session.cu").read_text()
    assert "enum { HOST_LOAD, HOST_KEEP, HOST_PREFETCH, HOST_FORMS };" in text
    assert [k for k, _ in sorted(HOST_FORMS.items(), key=lambda kv: kv[1])] == ["load", "keep", "prefetch"]
    assert f"constexpr int HOST_AHEAD_MAX = {device.HOST_AHEAD_MAX};" in text


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
