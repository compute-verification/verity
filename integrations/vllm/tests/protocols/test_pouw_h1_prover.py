"""`pouw_h1_prover`'s kept call files and the `Writer` that writes them off the serving thread (serve-openai's
``--h1-prover``): each file is the bytes ``np.savez`` wrote before the writer, so what the prover reads (`open_call`,
`kept`) is unchanged; a caller waits only while the writer's ring is full, so at most its bound is in flight; and a call
whose file wasn't written is raised by its window's `Writer.wait`, the other windows unaffected."""
from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.protocols.pouw.h1 import prover as PV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import records as ZR  # noqa: E402


def _today(path, a, s_a, records):
    """A call's file as `keep_call` wrote it before the writer: ``np.savez`` of its arrays."""
    with open(path, "wb") as f:
        np.savez(f, a=np.asarray(a, np.int8), s_a=np.asarray(s_a, "<f4").reshape(-1),
                 records=np.zeros((0, 0), np.uint8) if records is None else np.asarray(records))
    return path.read_bytes()


def _call(seed, m=5, k=64, units=3, records=True):
    rng = np.random.default_rng(seed)
    return (rng.integers(-128, 128, (m, k), dtype=np.int8), rng.random((m, 1), dtype=np.float32),
            rng.integers(0, 256, (units, ZR.ZY_BYTES), dtype=np.uint8) if records else None)


def _size(a, s_a, records):
    return PV._up(a.nbytes) + PV._up(s_a.nbytes) + (0 if records is None else PV._up(records.nbytes))


@pytest.mark.parametrize("records", [True, False])
def test_each_kept_file_is_the_bytes_np_savez_wrote(tmp_path, records):
    """`keep_call` in the call and the writer's thread, from arrays or tensors, each write ``np.savez``'s bytes."""
    a, s_a, r = _call(1, records=records)
    want = _today(tmp_path / "today.npz", a, s_a, r)
    assert PV.keep_call(tmp_path / "inline", 0, a, s_a, r).read_bytes() == want
    w, d = PV.Writer(2 * _size(a, s_a, r)), tmp_path / "off"
    w.open(d)
    assert PV.writer_of(d) is w and PV.writer_of(tmp_path / "inline") is None
    w.keep(d, 0, a, s_a, r)
    w.keep(d, 1, torch.from_numpy(a), torch.from_numpy(s_a), None if r is None else torch.from_numpy(r))
    nbytes = a.nbytes + s_a.nbytes + (0 if r is None else r.nbytes)
    assert w.wait(d) == {"calls": 2, "bytes": 2 * nbytes, "waits": 0} and PV.writer_of(d) is None
    assert (d / PV.CALLS / "0.npz").read_bytes() == (d / PV.CALLS / "1.npz").read_bytes() == want
    w.close()


def test_a_file_of_arrays_in_any_layout_is_np_savez_s(tmp_path):
    """Arrays `keep_call` takes in another layout (Fortran order, strided, one row, no rows of records) are written as
    ``np.savez`` writes them."""
    a, s_a, r = _call(2, m=4, k=96, units=2)
    for i, (x, s, rec) in enumerate([(np.asfortranarray(a), s_a, r), (a[:, ::2], s_a[:, 0], r[:, ::3]), (a[:1], s_a[:1], r),
                                     (a, s_a, np.zeros((0, ZR.ZY_BYTES), np.uint8))]):
        assert PV.keep_call(tmp_path, i, x, s, rec).read_bytes() == _today(tmp_path / f"{i}.today", x, s, rec), i


def test_savez_writes_np_savezs_bytes_for_many_small_arrays_of_any_dtype(tmp_path):
    """`savez` of the live dump's kinds of arrays (one-row u16 words, u8 salts viewed out of one buffer, a value's and a
    path's strings, an i64 position, a 0-d unit) and of empty and Fortran-ordered ones is ``np.savez``'s file, byte for
    byte."""
    salts = np.arange(4 * 192, dtype=np.uint64).astype(np.uint8).reshape(4, 1, 192)
    arrays = {"units": np.array([3]), "zero": np.array(7, np.int32), "empty": np.zeros((0, 3), "<f4"),
              "fortran": np.asfortranarray(np.arange(12, dtype=np.int16).reshape(3, 4))}
    for i in range(400):
        p = f"slot_{i}"
        arrays |= {f"w_{p}": np.asarray([[i, i >> 16]], np.uint16), f"s_{p}": salts[i % 4],
                   f"v_{p}": np.array([f"h1/slots@{i % 3}"]), f"r_{p}": np.array([i], np.int64),
                   f"p_{p}": np.array([bytes(range(i % 64, i % 64 + 64)).hex() * 3])}
    with open(tmp_path / "want.npz", "wb") as f:
        np.savez(f, **arrays)
    assert PV.savez(tmp_path / "got.npz", arrays).read_bytes() == (tmp_path / "want.npz").read_bytes()
    assert sorted(np.load(tmp_path / "got.npz").files) == sorted(arrays)


def test_the_caller_waits_only_while_the_ring_is_full(tmp_path, monkeypatch):
    """A ring of two calls' bytes takes two calls while the first is still being written, neither caller waiting; a third
    waits for room until the first is written, so no more than the bound is ever in flight."""
    started, gate, savez = threading.Event(), threading.Event(), PV._savez

    def held(path, *arrays):
        started.set()
        gate.wait()
        return savez(path, *arrays)
    monkeypatch.setattr(PV, "_savez", held)
    a, s_a, r = _call(3)
    w, d = PV.Writer(2 * _size(a, s_a, r)), tmp_path / "w0"
    w.open(d)
    w.keep(d, 0, a, s_a, r)
    w.keep(d, 1, a, s_a, r)
    started.wait()
    third = threading.Thread(target=w.keep, args=(d, 2, a, s_a, r))
    third.start()
    with w._cv:
        w._cv.wait_for(lambda: w._dirs[str(d)]["waits"] > 0)
        assert len(w._jobs) == 2 and list((d / PV.CALLS).iterdir()) == []
    gate.set()
    third.join()
    assert w.wait(d) == {"calls": 3, "bytes": 3 * (a.nbytes + s_a.nbytes + r.nbytes), "waits": 1}
    assert sorted(p.name for p in (d / PV.CALLS).iterdir()) == ["0.npz", "1.npz", "2.npz"]
    w.close()


def test_a_call_not_written_fails_its_window_alone(tmp_path, monkeypatch):
    """A write that fails is its window's: `wait` raises OSError naming the call once the window's other calls are
    written, and the next window on the writer is written whole."""
    savez = PV._savez

    def full(path, *arrays):
        if path.parent.parent.name == "w0" and path.name == "1.npz":
            raise OSError(28, "No space left on device")
        return savez(path, *arrays)
    monkeypatch.setattr(PV, "_savez", full)
    a, s_a, r = _call(4)
    w = PV.Writer(_size(a, s_a, r))
    for window in ("w0", "w1"):
        w.open(tmp_path / window)
        for i in range(3):
            w.keep(tmp_path / window, i, a, s_a, r)
        if window == "w0":
            with pytest.raises(OSError, match=r"1 of its 3 calls' prover state wasn't written: call 1: OSError: .*No space"):
                w.wait(tmp_path / window)
            assert sorted(p.name for p in (tmp_path / window / PV.CALLS).iterdir()) == ["0.npz", "2.npz"]
        else:
            assert w.wait(tmp_path / window)["calls"] == 3
    w.close()
    with pytest.raises(RuntimeError, match="the writer is closed"):
        w.keep(tmp_path / "w1", 3, a, s_a, r)


def test_a_call_past_the_bound_is_refused_and_a_directory_opens_on_one_writer(tmp_path):
    a, s_a, r = _call(5)
    w = PV.Writer(_size(a, s_a, r) - PV.ALIGN)
    w.open(tmp_path)
    with pytest.raises(ValueError, match="past the writer's bound"):
        w.keep(tmp_path, 0, a, s_a, r)
    other = PV.Writer(PV.ALIGN)
    with pytest.raises(ValueError, match="already open on a writer"):
        other.open(tmp_path)
    assert w.wait(tmp_path) == {"calls": 0, "bytes": 0, "waits": 0}
    w.close()
    other.close()
