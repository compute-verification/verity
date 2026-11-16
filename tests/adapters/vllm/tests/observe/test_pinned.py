"""`verity_vllm.observe.pinned` on a machine with no accelerator: how the CUDA runtime is looked for, what `device_backed` says
without one, the arena's one buffer and what does not fit in it, and the bytes a view spans.  Pinned memory needs an accelerator,
so the arena's allocations here are host memory, with what was asked of `torch.empty` recorded."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.observe import pinned


def _no_runtime(name: str):
    raise OSError(f"{name}: cannot open shared object file")


def test_the_cuda_runtime_is_looked_for_by_soname_the_versioned_one_first_and_only_once(monkeypatch):
    tried: list[str] = []

    def only_unversioned(name: str):
        tried.append(name)
        return "the runtime" if name == "libcudart.so" else _no_runtime(name)

    monkeypatch.setattr(pinned, "_cudart", None)
    monkeypatch.setattr(pinned.ctypes, "CDLL", only_unversioned)
    assert pinned._runtime() == "the runtime" and tried == ["libcudart.so.12", "libcudart.so"]
    assert pinned._runtime() == "the runtime" and len(tried) == 2, "the runtime found is kept"

    def loads(name: str) -> str:
        tried.append(name)
        return f"loaded {name}"

    monkeypatch.setattr(pinned, "_cudart", None)
    monkeypatch.setattr(pinned.ctypes, "CDLL", loads)
    assert pinned._runtime() == "loaded libcudart.so.12" and tried[2:] == ["libcudart.so.12"]


def test_a_machine_with_no_cuda_runtime_is_asked_once_and_device_backed_is_false_there(monkeypatch):
    tried: list[str] = []

    def loads_nothing(name: str):
        tried.append(name)
        _no_runtime(name)

    monkeypatch.setattr(pinned, "_cudart", None)
    monkeypatch.setattr(pinned.ctypes, "CDLL", loads_nothing)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pinned._runtime() is None and tried == ["libcudart.so.12", "libcudart.so"]
    assert pinned.device_backed(torch.zeros(2)) is False
    assert pinned._runtime() is None and len(tried) == 2, "a runtime that did not load is not looked for again"


def test_device_backed_asks_the_runtime_about_a_cpu_tensor_alone_and_only_where_cuda_is(monkeypatch):
    monkeypatch.setattr(pinned, "_runtime", lambda: pytest.fail("the runtime was asked"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pinned.device_backed(torch.zeros(2, device="meta")) is False, "a tensor that is not on the CPU"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pinned.device_backed(torch.zeros(2)) is False, "a CPU tensor on a machine without CUDA"


@pytest.fixture
def asked(monkeypatch):
    """What `torch.empty` is asked for, with host memory given where pinned memory is asked."""
    calls, empty = [], torch.empty

    def host(*size, pin_memory=False, **kw):
        calls.append((size, pin_memory, kw))
        return empty(*size, **kw)

    monkeypatch.setattr(pinned.torch, "empty", host)
    return calls


def test_an_arena_is_one_pinned_buffer_of_bytes_taken_at_256_byte_steps(asked):
    arena = pinned.PinnedArena(600)
    assert asked == [((600,), True, {"dtype": torch.uint8})]
    assert (arena.buf.numel(), arena.buf.dtype, arena.off, arena.overflow) == (600, torch.uint8, 0, 0)
    first, second = arena.take(100), arena.take(300)
    base = arena.buf.data_ptr()
    assert (first.data_ptr(), first.numel(), second.data_ptr(), second.numel()) == (base, 100, base + 256, 300)
    assert arena.off == 556 and arena.overflow == 0 and len(asked) == 1
    arena.reset()
    assert arena.off == 0 and arena.fits(600) and not arena.fits(601)
    assert arena.take(600).data_ptr() == base and arena.off == 600


def test_what_does_not_fit_in_the_arena_is_a_fresh_pinned_tensor_and_is_counted(asked):
    arena = pinned.PinnedArena(600)
    arena.take(300)
    assert not arena.fits(89) and arena.fits(88), "the next take starts at byte 512"
    over = arena.take(89)
    assert asked[1:] == [((89,), True, {"dtype": torch.uint8})]
    assert (over.numel(), over.dtype, arena.overflow, arena.off) == (89, torch.uint8, 1, 300)
    assert not arena.buf.data_ptr() <= over.data_ptr() < arena.buf.data_ptr() + 600
    assert arena.take(88).data_ptr() == arena.buf.data_ptr() + 512 and (arena.overflow, arena.off) == (1, 600)
    arena.reset()
    assert (arena.off, arena.overflow) == (0, 1), "a reset keeps the count"


def test_the_bytes_a_view_spans_are_its_offset_and_its_last_elements_end_and_none_for_an_empty_view():
    base = torch.arange(24, dtype=torch.int16)
    assert pinned.spanned_bytes(base) == (0, 48)
    assert pinned.spanned_bytes(base[4:4]) == (8, 0) and pinned.spanned_bytes(torch.empty(0, dtype=torch.float64)) == (0, 0)
    assert pinned.spanned_bytes(base[2:9:3]) == (4, 14), "elements 2, 5 and 8: seven elements from the first to the last"
    grid = base.reshape(4, 6)
    assert pinned.spanned_bytes(grid[1:3, 2:5]) == (16, 18), "from element 8 to element 16"
    assert pinned.spanned_bytes(grid[:, 5:5]) == (10, 0)


def test_a_byte_view_of_a_storage_is_those_bytes_without_a_copy():
    base = torch.arange(24, dtype=torch.int16)
    off, nbytes = pinned.spanned_bytes(base.reshape(4, 6)[1:3, 2:5])
    view = pinned.storage_bytes_view(base, off, nbytes)
    assert (view.dtype, view.shape, view.device) == (torch.uint8, (18,), base.device)
    assert bytes(view.tolist()) == base.numpy().tobytes()[16:34]
    assert view.data_ptr() == base.data_ptr() + 16
    view[0] = 255
    assert base[8].item() == 255, "the view writes through to the storage"
