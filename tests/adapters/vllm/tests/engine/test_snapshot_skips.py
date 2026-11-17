"""The capture's snapshots never copy what can't hold a new value or can't be read on the host:

* an identity alias's call (``aten.detach``, which ``torch::from_blob`` dispatches) is not snapshotted at all;
* a CPU tensor whose storage is device memory (`pinned.device_backed`) is listed as skipped, wherever it shows up;
* every other tensor is still copied;
* `device_backed` answers from the CUDA runtime's pointer attributes, and False when it can't tell;
* a step whose copies fill the pinned arena writes them out and reuses it, the files and index those of one flush.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from verity_vllm.observe import capture as C
from verity_vllm.observe import pinned


class _View:
    alloc, dtype, shape, strides = 1, "uint8", [4], [1]

    def to_json(self):
        return {"alloc": 1}


class _Arena:
    def __init__(self):
        self.taken = []

    def fits(self, n):
        return True

    def take(self, n):
        self.taken.append(n)
        return torch.empty(n, dtype=torch.uint8)


def _capture():
    capture = C.Capture.__new__(C.Capture)
    capture.cfg = SimpleNamespace(snapshot_steps=frozenset({3}), snapshot_max_bytes=1 << 20, snapshot_max_bytes_op=None)
    capture._cur = SimpleNamespace(step=3, n_snap=0)
    capture.storage = SimpleNamespace(view_of=lambda t: _View())
    capture.root_allocs, capture._snap_index, capture._snap_pending, capture.errors = set(), [], [], []
    capture._arena, capture._kv_num_blocks = _Arena(), None
    return capture


def test_an_identity_alias_call_is_not_snapshotted(monkeypatch):
    capture, t = _capture(), torch.arange(4, dtype=torch.uint8)
    monkeypatch.setattr(C, "device_backed", lambda x: pytest.fail("an alias call's tensors were inspected"))
    for op in pinned.IDENTITY_ALIAS_OPS:
        capture._after_call(7, op, (t,), {}, t)
    assert capture.errors == [] and capture._arena.taken == [] and capture._snap_pending == [] and capture._snap_index == []


def test_a_device_backed_cpu_tensor_is_listed_as_skipped(monkeypatch):
    capture, t = _capture(), torch.arange(4, dtype=torch.uint8)
    monkeypatch.setattr(C, "device_backed", lambda x: x is t)
    other = torch.ones(4, dtype=torch.uint8)
    capture._after_call(8, "aten.add.Tensor", (t, other), {}, other)
    assert [e["path"] for e in capture._snap_index] == ["args[0]"] and capture._snap_index[0]["skipped"] == pinned.DEVICE_BACKED
    assert capture._arena.taken == [4, 4] and [p["entry"]["path"] for p in capture._snap_pending] == ["args[1]", "outs"]


def test_other_calls_are_copied():
    capture, t = _capture(), torch.arange(4, dtype=torch.uint8)
    capture._after_call(9, "aten.mul.Tensor", (t, t), {}, t)
    assert capture._snap_index == [] and len(capture._snap_pending) == 3 and capture.errors == []


def test_a_step_that_fills_the_arena_writes_it_out_and_reuses_it(monkeypatch, tmp_path):
    capture, syncs = _capture(), []
    arena = pinned.PinnedArena.__new__(pinned.PinnedArena)
    arena.buf, arena.off, arena.overflow = torch.empty(512, dtype=torch.uint8), 0, 0
    capture._arena, capture.cfg.snapshot_dir = arena, tmp_path
    capture._snap_files, capture._snap_bytes, capture.timing = {}, 0, {"flush_s": 0.0, "snapshot_s": 0.0, "sync_s": 0.0}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(len(capture._snap_pending)))
    ts = [torch.full((200,), i, dtype=torch.uint8) for i in range(5)]
    for i, t in enumerate(ts):
        capture._after_call(10 + i, "aten.mul.Tensor", (t,), {}, None)
    assert syncs == [2, 2] and arena.overflow == 0 and len(capture._snap_pending) == 1
    capture._flush_snapshots(capture._cur)
    assert [e["event"] for e in capture._snap_index] == [10, 11, 12, 13, 14] and capture.errors == []
    assert [(tmp_path / e["file"]).read_bytes() for e in capture._snap_index] == [t.numpy().tobytes() for t in ts]
    assert {e["file"].split("/")[1] for e in capture._snap_index} == {"3"} and capture._snap_bytes == 1000


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
