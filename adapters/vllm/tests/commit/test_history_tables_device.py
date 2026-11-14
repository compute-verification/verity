"""`history_tables_device`: a history table's tree on the GPU and its CPU twin, each held bit for bit to `history.HistoryTable`
over the same rows under the same salts: the root after each append (one row, several, up to the capacity), every
row's string, paths at held and placeholder positions, and the entry. The rows cover a segment's edges (one word; 512 words,
exactly one segment; 576, a DSA latent row; 4096, the most). The twin needs a C++ compiler; the device tests need CUDA and
nvcc, and skip elsewhere. The device table's Python runs without either, over stand-ins for torch and for the device's library
whose three kernels are the twin's: what it hands each kernel, in which order, and that the table it keeps is the reference's."""
from __future__ import annotations

import contextlib
import ctypes
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from verity.commitments import history as HT

from verity_vllm.commit import history_tables as H
from verity_vllm.commit import history_tables_device as HD

PROGRAM, VALUE = "ab" * 64, H.name("r0", 3)


def _rows(seed: int, k: int, words: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, (k, words), dtype=np.uint16)


def _same(t: HD.Table, ref: HT.HistoryTable, positions: list[int]) -> None:
    assert t.root == ref.root and t.bcs == ref.bcs and t.entry() == ref.entry()
    assert t.paths(positions) == b"".join(ref.path(p) for p in positions)
    assert H.check(t, t.bcs)


def _appended(device, words: int, capacity: int, batches: list[int]) -> None:
    t = HD.Table(PROGRAM, VALUE, words, capacity, device=device)
    ref = HT.HistoryTable(PROGRAM, VALUE, words, capacity)
    assert t.root == ref.root
    for i, k in enumerate(batches):
        rows = _rows(i, k, words)
        assert t.append(rows) == range(len(ref.rows), len(ref.rows) + k)
        ref.append(rows, t.salts[len(ref.rows):])
        _same(t, ref, sorted({0, len(ref.rows) - 1, len(ref.rows) % capacity, capacity - 1}))


@pytest.mark.parametrize("words, capacity, batches", [(576, HT.CAPACITY, [1, 5, 3]), (1, 8, [1, 7]), (512, 16, [4, 1]),
                                                      (4096, 4, [2, 2])], ids=["dsa", "one-word", "one-segment", "most"])
def test_the_cpu_twin_is_history_table(words, capacity, batches):
    _appended(None, words, capacity, batches)


def _built(rows: np.ndarray, salts: list[bytes] | None = None, device=None) -> HD.Table:
    t = HD.Table(PROGRAM, VALUE, rows.shape[1], 8, device=device)
    t.append(rows, salts)
    return t


def test_two_builds_of_one_table_share_no_salt():
    """A table built twice over the same rows, under the same program and name, samples each row's 192-byte salt afresh:
    the two builds share no salt, so no string, and their roots differ. Nothing the builder holds fixes a salt."""
    rows = _rows(9, 3, 4)
    a, b = _built(rows), _built(rows)
    assert all(len(s) == 192 for s in a.salts + b.salts) and len(set(a.salts + b.salts)) == 6
    assert not set(a.bcs) & set(b.bcs) and a.root != b.root


def test_it_takes_salts_given_as_history_table_does():
    """Under another table's salts it is that table: its strings and root. A salt is refused when it is not 192 bytes, not
    one a row, used twice in one append, or one the table has used, given or sampled."""
    rows = _rows(9, 3, 4)
    a = _built(rows)
    b = _built(rows, a.salts)
    assert (b.salts, b.bcs, b.root) == (a.salts, a.bcs, a.root)
    for salts, why in [(a.salts[:2], "one 192-byte salt per row"), ([y[:191] for y in a.salts], "one 192-byte salt per row"),
                       ([a.salts[0]] * 3, "never reused"), (a.salts, "never reused")]:
        with pytest.raises(ValueError, match=why):
            a.append(rows, salts)
    sampled = _built(rows)
    with pytest.raises(ValueError, match="never reused"):
        sampled.append(rows[:1], sampled.salts[:1])
    assert len(a.rows) == len(sampled.rows) == 3


def test_it_refuses_what_history_table_refuses():
    t = HD.Table(PROGRAM, VALUE, 4, 4)
    t.append(_rows(1, 3, 4))
    with pytest.raises(ValueError, match="capacity"):
        t.append(_rows(2, 2, 4))
    with pytest.raises(ValueError, match="rows of 4"):
        t.append(_rows(3, 1, 5))
    with pytest.raises(ValueError, match="positions"):
        t.paths([4])
    for words, capacity in [(4, 6), (4, 1), (4097, 4), (0, 4)]:
        with pytest.raises(ValueError):
            HD.Table(PROGRAM, VALUE, words, capacity)


def test_histories_take_the_twin_as_their_tables():
    """`Histories` with the twin's tables registers the roots `history.check` recomputes from the strings they commit."""
    hs = H.Histories(PROGRAM, capacity=16, make=HD.Table)
    hs.append("r0", 3, 0, _rows(4, 5, H.WORDS))
    hs.append("r0", 3, 5, _rows(5, 2, H.WORDS))
    t = hs.table("r0", 3)
    assert isinstance(t, HD.Table) and H.check(t, [hs.strings("r0", 3, [p]) for p in range(7)])


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or HD.D.nvcc() is None:
        pytest.skip("needs CUDA and nvcc")
    return torch


@pytest.mark.parametrize("words, capacity, batches", [(576, HT.CAPACITY, [1, 64, 3]), (1, 8, [1, 7]), (4096, 4, [2, 2])],
                         ids=["dsa", "one-word", "most"])
def test_the_device_table_is_history_table(words, capacity, batches):
    _cuda()
    _appended(0, words, capacity, batches)


def test_the_device_table_appends_rows_already_on_the_device():
    """The twin under the device table's salts is the device table; a second device build samples salts of its own."""
    torch = _cuda()
    t = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, device=0)
    twin = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY)
    rows = _rows(6, 9, H.WORDS)
    t.append(torch.from_numpy(rows.view(np.int16)).cuda().view(torch.bfloat16))
    twin.append(rows, t.salts)
    assert t.root == twin.root and t.bcs == twin.bcs and t.paths([0, 8, 9]) == twin.paths([0, 8, 9])
    again = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, device=0)
    again.append(rows)
    assert not set(again.salts) & set(t.salts) and again.root != t.root


def test_the_device_table_takes_salts_given():
    _cuda()
    rows = _rows(9, 3, 4)
    twin = _built(rows)
    t = _built(rows, twin.salts, device=0)
    assert (t.salts, t.bcs, t.root) == (twin.salts, twin.bcs, twin.root)


def _declared(name: str) -> list[Any]:
    """The ctypes of ``name``'s parameters as `row_table.cu` declares them: `Hist`'s pointer for a ``const Hist*``, a void pointer
    for any other pointer, and the integer of the declared width."""
    (parameters,) = re.findall(rf'extern "C" int {name}\(([^)]*)\)', (HD.D.CUDA_DIR / "row_table.cu").read_text())
    kinds = [" ".join(p.split()).rsplit(" ", 1)[0] for p in parameters.split(",")]
    integers = {"uint64_t": ctypes.c_uint64, "uint32_t": ctypes.c_uint32}
    return [HD._hp if k == "const Hist*" else ctypes.c_void_p if k.endswith("*") else integers[k] for k in kinds]


@pytest.mark.parametrize("name", ["rt_hist_init", "rt_hist_append", "rt_hist_paths"])
def test_the_device_kernels_are_called_as_their_source_declares_them(name):
    """`_device` gives each ``rt_hist_*`` kernel the arguments `row_table.cu` declares for it and an ``int`` to return. A stand-in
    holds what it assigns, so no build is needed and the test runs without CUDA."""
    lib: Any = SimpleNamespace(rt_hist_init=SimpleNamespace(), rt_hist_append=SimpleNamespace(), rt_hist_paths=SimpleNamespace())
    assert HD._device(lib) is lib
    kernel = getattr(lib, name)
    assert list(kernel.argtypes) == _declared(name) and kernel.restype is ctypes.c_int


class _Tensor:
    """A stand-in for a tensor: a numpy array on the host or on the device. One on the device is held by the address a kernel
    is handed for it, and a kernel finds no other."""

    def __init__(self, array: np.ndarray, device: str, held: dict[int, np.ndarray]) -> None:
        self.array, self.device, self.held, self.shape, self.is_cuda = array, device, held, array.shape, device == DEVICE
        if self.is_cuda:
            held[self.data_ptr()] = array

    def data_ptr(self) -> int:
        return int(self.array.ctypes.data)

    def to(self, device: str) -> _Tensor:
        return _Tensor(self.array.copy(), device, self.held)

    def cpu(self) -> _Tensor:
        return self.to("the host")

    def view(self, dtype: Any) -> _Tensor:
        return _Tensor(self.array.view(dtype), self.device, self.held)

    def __getitem__(self, at: Any) -> _Tensor:
        return _Tensor(self.array[at], self.device, self.held)

    def numpy(self) -> np.ndarray:
        assert not self.is_cuda
        return self.array

    def element_size(self) -> int:
        return int(self.array.itemsize)

    def is_contiguous(self) -> bool:
        return bool(self.array.flags.c_contiguous)


class _Kernel:
    """A stand-in for one ``rt_hist_*`` kernel of the device's library: the twin's (``twin``, called without the stream), after
    it has checked that it runs on the table's stream and that its arguments are ones the types `_device` gave it take. It
    returns ``refuses`` instead where that is set."""

    argtypes: Any = None
    restype: Any = None
    refuses = 0

    def __init__(self, name: str, twin: Any, it: SimpleNamespace) -> None:
        self.name, self.twin, self.it = name, twin, it

    def __call__(self, *args: Any) -> int:
        assert self.it.log[-1:] == ["on the table's stream"] and args[0] == STREAM
        assert len(args) == len(self.argtypes) and self.restype is ctypes.c_int
        for kind, arg in zip(self.argtypes, args, strict=True):
            kind.from_param(arg)
        self.it.log.append(self.name)
        self.it.calls.append((self.name, args))
        return self.refuses or int(self.twin(*args[1:]))


DEVICE, STREAM, CURRENT_DEVICE = "the device", 0x5EED, 3


def _stand_ins(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """torch and the device's library as the table's device branch reads them, with no device: a tensor is a `_Tensor`, and
    each ``rt_hist_*`` kernel is the twin's over the arrays on the device at the addresses it is handed, once it has seen that
    the byte tables and midstates are the library's. -> ``log`` (what ran, in order), ``calls`` (each kernel's arguments),
    ``asked`` (the device indices a library was asked for), ``on`` (an array as a tensor on the device), ``at`` (the array on
    the device at an address, as a type) and ``lib`` (the kernels)."""
    held: dict[int, np.ndarray] = {}
    it = SimpleNamespace(log=[], calls=[], asked=[], on=lambda array: _Tensor(array, DEVICE, held),
                         at=lambda pointer, dtype: held[pointer].view(dtype))

    def init(h: Any, tree: int) -> int:
        return HD._host().rt_hist_init_host(h, it.at(tree, np.uint64))

    def append(h: Any, start: int, k: int, rows: int, hm: int, ms: int, tree: int, bcs: int, salts: int) -> int:
        assert (hm, ms) == (library.hm.data_ptr(), library.ms.data_ptr()) and it.at(rows, np.uint16).size == k * h._obj.L // 2
        return HD._host().rt_hist_append_host(h, start, k, rows, HD.D._hm(), HD.D._prefixes(), it.at(tree, np.uint64),
                                              it.at(bcs, np.uint64), it.at(salts, np.uint32))

    def paths(h: Any, tree: int, positions: int, n: int, out: int) -> int:
        return HD._host().rt_hist_paths_host(h, it.at(tree, np.uint64), it.at(positions, np.uint32), n, it.at(out, np.uint64))

    class Event:
        def record(self, stream: Any) -> None:
            self.on = stream

    @contextlib.contextmanager
    def stream(s: Any) -> Any:
        assert s is library.stream
        it.log.append("on the table's stream")
        yield
        it.log.append("off it")

    def of(torch: Any, index: int) -> Any:
        assert torch is sys.modules["torch"]
        it.asked.append(index)
        return library

    it.lib = SimpleNamespace(rt_hist_init=_Kernel("rt_hist_init", init, it), rt_hist_append=_Kernel("rt_hist_append", append, it),
                             rt_hist_paths=_Kernel("rt_hist_paths", paths, it))
    library = SimpleNamespace(
        hm=it.on(HD.D._hm().view(np.int64)), ms=it.on(np.zeros(16, dtype=np.int64)), lib=it.lib,
        stream=SimpleNamespace(cuda_stream=STREAM, synchronize=lambda: it.log.append("synchronized"),
                               wait_event=lambda event: it.log.append(f"the table's stream waits for an event on {event.on}")))
    cuda = SimpleNamespace(Event=Event, stream=stream, current_device=lambda: CURRENT_DEVICE,
                           current_stream=lambda device: f"the caller's stream on {device}")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=_Tensor, int16=np.int16, int64=np.int64, cuda=cuda, from_numpy=lambda array: _Tensor(array, "the host", held),
        empty=lambda n, dtype, device: _Tensor(np.zeros(n, dtype=dtype), device, held),
        device=lambda kind, index=None: SimpleNamespace(
            index=index if index is not None else int(kind.split(":")[1]) if ":" in kind else None)))
    monkeypatch.setattr(HD.D.DeviceLibrary, "of", staticmethod(of))
    return it


@pytest.mark.parametrize("words, capacity, batches", [(576, HT.CAPACITY, [1, 5, 3]), (1, 8, [1, 7]), (512, 16, [4, 1]),
                                                      (4096, 4, [2, 2])], ids=["dsa", "one-word", "one-segment", "most"])
def test_the_device_tables_python_keeps_history_table_over_the_twins_kernels(monkeypatch, words, capacity, batches):
    """The device branch of `Table`, its build, appends, root and paths, over the stand-ins: the table is the reference's after
    every append, by one ``rt_hist_init``, an ``rt_hist_append`` an append and an ``rt_hist_paths`` each time paths are read."""
    it = _stand_ins(monkeypatch)
    _appended(0, words, capacity, batches)
    ran = [name for name, _ in it.calls]
    assert it.asked == [0] and ran[0] == "rt_hist_init" and ran.count("rt_hist_init") == 1
    assert ran.count("rt_hist_append") == len(batches) and ran.count("rt_hist_paths") == len(batches)


def test_a_device_append_waits_for_the_callers_stream_and_hands_the_kernel_the_tables_own(monkeypatch):
    """Each kernel runs on the table's stream, which is synchronized after it. An append first has that stream wait for an
    event on the caller's, so rows the caller wrote on the device are there when the kernel reads them. The kernel is handed the
    stream's handle, the table's header, the position and the count, the rows, the library's byte tables and midstates, the
    table's tree, room for the strings, and the salts, each on the device: numpy rows as a copy there, rows already there as
    they are."""
    it = _stand_ins(monkeypatch)
    t = HD.Table(PROGRAM, VALUE, 4, 8, device=0)
    ran = ["on the table's stream", "KERNEL", "off it", "synchronized"]
    assert it.log == [w.replace("KERNEL", "rt_hist_init") for w in ran]
    waited = ["the table's stream waits for an event on the caller's stream on the device"]
    rows, there = _rows(1, 3, 4), it.on(_rows(2, 2, 4).view(np.int16))
    for given, at in [(rows, 0), (there, 3)]:
        del it.log[:]
        t.append(given)
        assert it.log == waited + [w.replace("KERNEL", "rt_hist_append") for w in ran]
        name, (stream, header, start, k, src, hm, ms, tree, bcs, salts) = it.calls[-1]
        assert (name, stream, start, k) == ("rt_hist_append", STREAM, at, given.shape[0])
        assert ctypes.addressof(header._obj) == ctypes.addressof(t._h) and tree == t._tree.data_ptr()
        assert (src == there.data_ptr()) == (given is there) and src != rows.ctypes.data
        assert it.at(src, np.uint16).tobytes() == np.asarray(t.rows[at:at + k]).tobytes()
        assert it.at(salts, np.uint32).tobytes() == b"".join(t.salts[at:at + k])
        assert it.at(bcs, np.uint64).astype(">u8").tobytes() == b"".join(t.bcs[at:at + k])
        assert len({src, hm, ms, tree, bcs, salts}) == 6
    del it.log[:]
    opened = t.paths([0, 4, 7])
    assert it.log == [w.replace("KERNEL", "rt_hist_paths") for w in ran]
    name, (stream, header, tree, positions, n, out) = it.calls[-1]
    assert (name, stream, n, tree) == ("rt_hist_paths", STREAM, 3, t._tree.data_ptr())
    assert it.at(positions, np.uint32).tolist() == [0, 4, 7] and it.at(out, np.uint64).astype(">u8").tobytes() == opened
    twin = HD.Table(PROGRAM, VALUE, 4, 8)
    twin.append(np.concatenate([rows, there.cpu().numpy().view(np.uint16)]), t.salts)
    assert (t.root, t.bcs, opened) == (twin.root, twin.bcs, twin.paths([0, 4, 7])) and len(t.rows) == 5


def test_a_device_table_refuses_what_its_kernels_refuse_and_rows_on_the_device_it_cannot_read(monkeypatch):
    """A kernel that returns other than nought is an error that names it, the table and what it returned: at the build, at an
    append, which then holds no row of it, and where paths are read. Rows on the device that are not 16-bit, or not contiguous,
    are refused before any kernel runs."""
    it = _stand_ins(monkeypatch)
    it.lib.rt_hist_init.refuses = 5
    with pytest.raises(RuntimeError, match=rf"rt_hist_init refused table {re.escape(VALUE)} \(5\)"):
        HD.Table(PROGRAM, VALUE, 4, 8, device=0)
    it.lib.rt_hist_init.refuses = 0
    t = HD.Table(PROGRAM, VALUE, 4, 8, device=0)
    t.append(_rows(1, 2, 4))
    it.lib.rt_hist_append.refuses, it.lib.rt_hist_paths.refuses = 7, 9
    with pytest.raises(RuntimeError, match=rf"rt_hist_append refused table {re.escape(VALUE)} \(7\)"):
        t.append(_rows(2, 1, 4))
    with pytest.raises(RuntimeError, match=rf"rt_hist_paths refused table {re.escape(VALUE)} \(9\)"):
        t.paths([0])
    assert (len(t.rows), len(t.salts), len(t.bcs)) == (2, 2, 2)
    ran = len(it.calls)
    wide, strided = it.on(np.zeros((2, 4), dtype=np.int32)), it.on(np.zeros((2, 8), dtype=np.int16))[:, ::2]
    assert strided.shape == (2, 4) and not strided.is_contiguous()
    for rows in (wide, strided):
        with pytest.raises(ValueError, match="rows of 4 contiguous 16-bit words"):
            t.append(rows)
    assert len(it.calls) == ran and len(t.rows) == 2


@pytest.mark.parametrize("device, index", [(0, 0), (1, 1), ("cuda:2", 2), ("cuda", CURRENT_DEVICE)])
def test_a_device_table_asks_for_the_library_of_its_device(monkeypatch, device, index):
    """The library is the one of the device's index: the index given, the one a device string names, or torch's current
    device where the string names none."""
    it = _stand_ins(monkeypatch)
    HD.Table(PROGRAM, VALUE, 4, 8, device=device)
    assert it.asked == [index]
