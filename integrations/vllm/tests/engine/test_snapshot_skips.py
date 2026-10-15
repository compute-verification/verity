"""The capture's snapshots never copy what can't hold a new value or can't be read on the host (VLLM-8):

* an identity alias's call (``aten.detach``, which ``torch::from_blob`` dispatches) lists its tensors as skipped;
* a CPU tensor whose storage is device memory (`pinned.device_backed`) is listed as skipped, wherever it shows up;
* every other tensor is still copied;
* `device_backed` answers from the CUDA runtime's pointer attributes, and False when it can't tell.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from verity_vllm.engine import capture as C
from verity_vllm.engine import pinned


class _View:
    alloc, dtype, shape, strides = 1, "uint8", [4], [1]

    def to_json(self):
        return {"alloc": 1}


class _Arena:
    def __init__(self):
        self.taken = []

    def take(self, n):
        self.taken.append(n)
        return torch.empty(n, dtype=torch.uint8)


def _capture():
    cap = C.Capture.__new__(C.Capture)
    cap.cfg = SimpleNamespace(snapshot_steps=frozenset({3}), snapshot_max_bytes=1 << 20, snapshot_max_bytes_op=None)
    cap._cur = SimpleNamespace(step=3, n_snap=0)
    cap.storage = SimpleNamespace(view_of=lambda t: _View())
    cap.root_allocs, cap._snap_index, cap._snap_pending, cap.errors = set(), [], [], []
    cap._arena, cap._kv_num_blocks = _Arena(), None
    return cap


def test_an_identity_alias_call_is_listed_as_skipped_and_copies_nothing():
    cap, t = _capture(), torch.arange(4, dtype=torch.uint8)
    cap._after_call(7, "aten.detach.default", (t,), {}, t)
    assert cap.errors == [] and cap._arena.taken == [] and cap._snap_pending == []
    assert [(e["event"], e["path"]) for e in cap._snap_index] == [(7, "args[0]"), (7, "outs")]
    assert all("identity alias" in e["skipped"] for e in cap._snap_index)


def test_a_device_backed_cpu_tensor_is_listed_as_skipped(monkeypatch):
    cap, t = _capture(), torch.arange(4, dtype=torch.uint8)
    monkeypatch.setattr(C, "device_backed", lambda x: x is t)
    other = torch.ones(4, dtype=torch.uint8)
    cap._after_call(8, "aten.add.Tensor", (t, other), {}, other)
    assert [e["path"] for e in cap._snap_index] == ["args[0]"] and "device memory" in cap._snap_index[0]["skipped"]
    assert cap._arena.taken == [4, 4] and [p["entry"]["path"] for p in cap._snap_pending] == ["args[1]", "outs"]


def test_other_calls_are_copied():
    cap, t = _capture(), torch.arange(4, dtype=torch.uint8)
    cap._after_call(9, "aten.mul.Tensor", (t, t), {}, t)
    assert cap._snap_index == [] and len(cap._snap_pending) == 3 and cap.errors == []


class _Runtime:
    def __init__(self, rc, kind):
        self.rc, self.kind, self.cleared = rc, kind, 0

    def cudaPointerGetAttributes(self, ref, ptr):
        ctypes.cast(ref, ctypes.POINTER(pinned._PointerAttributes)).contents.type = self.kind
        return self.rc

    def cudaGetLastError(self):
        self.cleared += 1


@pytest.mark.parametrize("rc, kind, want", [(0, 2, True), (0, 0, False), (0, 1, False), (1, 2, False)])
def test_device_backed_reads_the_runtimes_pointer_type(monkeypatch, rc, kind, want):
    rt = _Runtime(rc, kind)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pinned, "_runtime", lambda: rt)
    assert pinned.device_backed(torch.zeros(2)) is want
    assert rt.cleared == (1 if rc else 0)


def test_device_backed_is_false_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pinned.device_backed(torch.zeros(2)) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_backed_on_cuda_tells_a_cpu_label_over_device_memory_from_host_memory():
    x = torch.arange(1 << 20, dtype=torch.uint8, device="cuda")
    storage = torch._C._construct_storage_from_data_pointer(x.data_ptr(), torch.device("cpu"), x.numel())
    over_device = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (x.numel(),), (1,))
    assert over_device.device.type == "cpu" and pinned.device_backed(over_device)
    assert not pinned.device_backed(torch.zeros(16)) and not pinned.device_backed(torch.zeros(16).pin_memory())
    assert not pinned.device_backed(x)
