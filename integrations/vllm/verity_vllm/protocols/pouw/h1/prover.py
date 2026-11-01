"""An H1 window's private prover state (DRAWN1): what opening a sampled proof block's committed values needs after the
window's close, kept in a directory of the prover's own (`pouw_h1.H1Run`'s ``prover``), owner-only (0700, each file 0600),
never under a run's ``$OUT`` or any published path. The serving node keeps it (`research keep`), and its holder alone reads it.

* **What it keeps.** `keep_key`: the run's hm96 salt key (`KEY`, 32 bytes), from which the salt of every hiding leaf of
  the window is sampled: its request, slot, A and z tables' (`pouw_h1_requests.hiding_rows`) and its served trees' rows
  (`audit.hm96_salt` at `pouw_h1_served.nonce`). `keep_call`: each served call's m s8 rows, their f32 scales and its z/y
  records (`pouw_h1_records`, one a proof block), as ``calls/<index>.npz``. An empty call keeps nothing: its A table is T zero
  rows and its z table zero records. A handed-off window (`pouw_h1_handoff`) also keeps its saved state in `STATE`: its
  round log, request entries, slot owners, and each call's A codes and each linear's weight codes (the window's ``a/<K>``
  and ``b/<K>`` rows).
* **Opening a sampled proof block.** Proof block u of the window is proof block u - base of the call whose manifest row has base ≤ u < base +
  units (`locate`). `open_call` rebuilds that call's A table and z table from the kept rows, scales and records under the
  key, the Program salt (from the record's salt coin, `pouw_h1_challenger.program_salt_of`) and the call's
  nonces, and refuses either whose root isn't the one its manifest row carries, which its round's root commits. Then
  ``z.open(u - base)`` is the proof block's record (z then y) with its salt and path, and ``a.open(i)`` slot i's A row and scale
  (the proof block's row tile is slots 32·r .. 32·r + 31). Its ``a/<K>`` rows are its call's block of the window's A value
  (`pouw_h1_served.layout`), its codes the A table's padded to the Program's slot, and its ``b/<K>`` rows its linear's weight
  rows, each salted at `pouw_h1_served.nonce`; their trees are the ones the record's ``served`` entries name.
* **Off the serving thread** (`Writer`, serve-openai's ``--h1-prover``). A window's directory is open on the server's writer
  from the window's open to its release (`writer_of`): each call's rows, scales and records are copied into the writer's
  ring of host memory, bounded by a window's bytes (`kept_bytes`), from the device on its current stream, and the writer's
  threads write the calls' files, the bytes `keep_call` writes, each taking the next call in the order kept. The window's
  release waits for all its files (`Writer.wait`), and a file not written fails the window.
"""
from __future__ import annotations

import functools
import io
import os
import struct
import threading
import zipfile
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from verity_vllm.protocols.pouw.h1 import records as ZR

KEY = "key"
CALLS = "calls"
STATE = "state"
#: where each array of a call starts in a `Writer`'s ring: a multiple of this many bytes
ALIGN = 64


def private(directory: str | Path) -> Path:
    """`directory`, created if missing and made owner-only (0700)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def _write(path: Path, data: bytes) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(data)


def keep_key(directory: str | Path, key: bytes) -> Path:
    """The run's salt key, kept as `KEY` (0600) in the prover's directory."""
    path = private(directory) / KEY
    _write(path, bytes(key))
    return path


def key_of(directory: str | Path) -> bytes:
    return (Path(directory) / KEY).read_bytes()


#: the zip records ``np.savez`` writes through `zipfile`: a member's local header, its ZIP64 extra field (``np.savez`` forces
#: one on every member), its central directory entry, and the end record
_LOCAL, _ZIP64, _CENTRAL, _END = (struct.Struct(f) for f in ("<4s2B4HL2L2H", "<HHQQ", "<4s4B4HL2L5H2L", "<4s4H2LH"))
#: `zipfile.ZipInfo`'s default date, 1980-01-01 00:00, as a DOS date, and its creating system
_DOS_DATE, _SYSTEM = 1 << 5 | 1, zipfile.ZipInfo().create_system


@functools.lru_cache(maxsize=1024)
def _npy_header(dtype: np.dtype, shape: tuple[int, ...], fortran: bool) -> bytes:
    out = io.BytesIO()
    np.lib.format.write_array_header_1_0(out, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": fortran,
                                               "shape": shape})
    return out.getvalue()


def savez(path: Path, arrays: Mapping[str, Any]) -> Path:
    """``path`` (0600) holding the bytes ``np.savez(path, **arrays)`` writes: each array a stored member ``<name>.npy``
    with ZIP64 sizes at zipfile's 1980 date. Each member is one header, built once per dtype and shape, and its data hashed
    and written from the array's own buffer (the GIL released for a large one), so a file of many small arrays costs a
    few struct packs a member, not a zipfile stream's seeks and rewritten header; past zip's 32-bit sizes and counts,
    ``np.savez`` writes it."""
    members = []
    for name, v in arrays.items():
        v = np.asanyarray(v)
        fortran = bool(v.flags.f_contiguous and not v.flags.c_contiguous)
        data = (v.T if fortran else np.ascontiguousarray(v)).reshape(-1).view(np.uint8)
        head = _npy_header(v.dtype, v.shape, fortran)
        members.append((f"{name}.npy".encode("ascii"), head, data, zlib.crc32(data, zlib.crc32(head))))
    end = sum(_LOCAL.size + _ZIP64.size + _CENTRAL.size + 2 * len(n) + len(h) + d.nbytes for n, h, d, _c in members)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        if len(members) > 0xFFFF or end + _END.size > zipfile.ZIP64_LIMIT:
            np.savez(f, **arrays)
            return path
        central, offset = [], 0
        for name, head, data, crc in members:
            size = len(head) + data.nbytes
            f.write(_LOCAL.pack(b"PK\x03\x04", zipfile.ZIP64_VERSION, 0, 0, zipfile.ZIP_STORED, 0, _DOS_DATE, crc, 0xFFFFFFFF,
                                0xFFFFFFFF, len(name), _ZIP64.size) + name + _ZIP64.pack(1, _ZIP64.size - 4, size, size) + head)
            f.write(data)
            central.append(_CENTRAL.pack(b"PK\x01\x02", zipfile.ZIP64_VERSION, _SYSTEM, zipfile.ZIP64_VERSION, 0, 0,
                                         zipfile.ZIP_STORED, 0, _DOS_DATE, crc, size, size, len(name), 0, 0, 0, 0, 0o600 << 16,
                                         offset) + name)
            offset += _LOCAL.size + len(name) + _ZIP64.size + size
        cd = b"".join(central)
        f.write(cd + _END.pack(b"PK\x05\x06", 0, 0, len(central), len(central), len(cd), offset, 0))
    return path


def _savez(path: Path, a: np.ndarray, s_a: np.ndarray, records: np.ndarray | None) -> Path:
    """``path`` (0600) as ``np.savez`` writes the call's arrays, byte for byte (`savez`)."""
    return savez(path, {"a": np.asarray(a, np.int8), "s_a": np.asarray(s_a, "<f4").reshape(-1),
                        "records": np.zeros((0, 0), np.uint8) if records is None else np.asarray(records)})


def keep_call(directory: str | Path, index: int, a: np.ndarray, s_a: np.ndarray, records: np.ndarray | None) -> Path:
    """Call `index`'s served A rows (int8 [m, k]), their f32 scales [m] and its z/y records (or none), as committed."""
    return _savez(private(Path(directory) / CALLS) / f"{int(index)}.npz", a, s_a, records)


def _up(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


def kept_bytes(rows: int, k: int, units: int) -> int:
    """The most a call of ``rows`` slots at contraction ``k`` with ``units`` z/y records takes in a `Writer`'s ring."""
    return _up(rows * k) + _up(4 * rows) + _up(units * ZR.ZY_BYTES)


def _bytes_of(x: Any) -> Any:
    """``x`` (a tensor or an array) as one flat run of its bytes in C order: a uint8 CUDA tensor where it is, else an array."""
    if hasattr(x, "detach"):
        import torch

        t = x.detach().contiguous().reshape(-1)
        return t.view(torch.uint8) if t.is_cuda else t.numpy().view(np.uint8)
    return np.ascontiguousarray(x).reshape(-1).view(np.uint8)


#: each window directory open on a `Writer` (`Writer.open`), by its path
_OPEN: dict[str, Writer] = {}


def writer_of(directory: str | Path | None) -> Writer | None:
    """The `Writer` ``directory`` is open on, else None (its calls are kept by `keep_call`, on the caller's thread)."""
    return None if directory is None else _OPEN.get(str(Path(directory)))


@dataclass(frozen=True)
class _Job:
    #: its place in the order kept
    seq: int
    directory: str
    index: int
    lo: int
    #: each array's (offset, bytes, dtype, shape) in the ring: a, s_a and, when the call has them, its records
    spans: tuple[tuple[int, int, Any, tuple[int, ...]], ...]
    #: the copies' event on the device's current stream (None: copied from the host before the job was queued)
    copied: Any


class Writer:
    """`keep_call` off the serving thread: `keep` copies a call's rows, scales and records into a ring of ``bound`` bytes of
    host memory (pinned when CUDA is up, so a CUDA tensor's copy is queued on its device's current stream, behind what the
    stream has enqueued, and the caller never waits on the device), and ``threads`` threads of the writer's own write the
    calls' files from the ring, the bytes `keep_call` writes, each taking the next call in the order kept (a file's CRC and
    write release the GIL, so they overlap). A call's bytes stay in the ring until it and every call kept before it are
    written. The caller waits only while the ring is full: at most ``bound`` bytes are in flight (serve-openai's is a
    window's, `kept_bytes` of each of its calls). A window's directory is `open` on the writer until `wait` returns: once
    every call kept there is written, raising OSError when any wasn't."""

    def __init__(self, bound: int, threads: int = 2) -> None:
        import torch

        if threads < 1:
            raise ValueError(f"a writer of {threads} threads: at least one")
        self.bound = _up(bound)
        self._pinned = torch.empty(self.bound, dtype=torch.uint8, pin_memory=True) if torch.cuda.is_available() else None
        self._ring = self._pinned.numpy() if self._pinned is not None else np.empty(self.bound, np.uint8)
        self._cv = threading.Condition()
        #: the calls whose bytes are in the ring, in the order kept; the first ``_taken`` of them are being or were written
        self._jobs: deque[_Job] = deque()
        self._taken, self._seq, self._written = 0, 0, set()
        self._head, self._closed = 0, False
        self._dirs: dict[str, dict[str, Any]] = {}
        self._threads = [threading.Thread(target=self._run, name=f"h1-keep-{i}", daemon=True) for i in range(threads)]
        for t in self._threads:
            t.start()

    def open(self, directory: str | Path) -> None:
        """Keep the calls of the window whose prover directory is ``directory`` here, until `wait`."""
        key = str(Path(directory))
        if key in _OPEN:
            raise ValueError(f"{key} is already open on a writer")
        with self._cv:
            self._dirs[key] = {"calls": 0, "bytes": 0, "waits": 0, "pending": 0, "failed": []}
        _OPEN[key] = self

    def _fit(self, n: int) -> int | None:
        """Where ``n`` bytes go in the ring, after every job still queued or being written, or None while they don't fit."""
        if not self._jobs:
            return 0
        lo, head = self._jobs[0].lo, self._head
        if head > lo:
            return head if self.bound - head >= n else 0 if lo >= n else None
        return head if lo - head >= n else None

    def keep(self, directory: str | Path, index: int, a: Any, s_a: Any, records: Any) -> None:
        """Call ``index``'s rows ``a`` (int8 [m, k]), scales ``s_a`` (f32, m) and records (uint8 [units, ZY_BYTES], or
        None), tensors or arrays, kept under ``directory`` (open here) as `keep_call` keeps them, by the writer's thread."""
        m = int(a.shape[0])
        parts = [(_bytes_of(a), np.int8, (m, -1)), (_bytes_of(s_a), np.dtype("<f4"), (-1,))]
        if records is not None:
            parts.append((_bytes_of(records), np.uint8, tuple(int(d) for d in records.shape)))
        n = sum(_up(int(b.shape[0])) for b, _dtype, _shape in parts)
        if n > self.bound:
            raise ValueError(f"call {index}'s {n} bytes are past the writer's bound, {self.bound}")
        key = str(Path(directory))
        with self._cv:
            if self._closed:
                raise RuntimeError(f"call {index}: the writer is closed")
            st = self._dirs[key]
            while (lo := self._fit(n)) is None:
                st["waits"] += 1
                self._cv.notify_all()
                self._cv.wait()
            at, spans, stream = lo, [], None
            for b, dtype, shape in parts:
                size = int(b.shape[0])
                if isinstance(b, np.ndarray):
                    self._ring[at:at + size] = b
                else:
                    import torch

                    stream = torch.cuda.current_stream(b.device)
                    self._pinned[at:at + size].copy_(b, non_blocking=True)
                    # the copy is still queued when this returns: the allocator gives its source to no other stream till then
                    b.record_stream(stream)
                spans.append((at, size, dtype, shape))
                at += _up(size)
            copied = None
            if stream is not None:
                import torch

                copied = torch.cuda.Event(blocking=True)
                copied.record(stream)
            self._jobs.append(_Job(self._seq, key, int(index), lo, tuple(spans), copied))
            self._seq += 1
            self._head = lo + n
            st["calls"] += 1
            st["bytes"] += sum(s for _at, s, _dtype, _shape in spans)
            st["pending"] += 1
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._taken == len(self._jobs) and not self._closed:
                    self._cv.wait()
                if self._taken == len(self._jobs):
                    return
                job = self._jobs[self._taken]
                self._taken += 1
            failed = None
            try:
                if job.copied is not None:
                    job.copied.synchronize()
                a, s_a, *records = (self._ring[at:at + n].view(dtype).reshape(shape) for at, n, dtype, shape in job.spans)
                _savez(private(Path(job.directory) / CALLS) / f"{job.index}.npz", a, s_a, records[0] if records else None)
            except Exception as e:  # noqa: BLE001 - any failure is the window's, raised by its `wait`
                failed = f"call {job.index}: {type(e).__name__}: {e}"
            with self._cv:
                self._written.add(job.seq)
                while self._jobs and self._jobs[0].seq in self._written:
                    self._written.remove(self._jobs.popleft().seq)
                    self._taken -= 1
                st = self._dirs[job.directory]
                st["pending"] -= 1
                if failed is not None:
                    st["failed"].append(failed)
                self._cv.notify_all()

    def wait(self, directory: str | Path) -> dict[str, int]:
        """Once every call kept under ``directory`` is written, the directory closed on the writer: its calls, their bytes
        and how often the ring was full when one came (``waits``). OSError naming the first call not written, if any."""
        key = str(Path(directory))
        _OPEN.pop(key, None)
        with self._cv:
            while self._dirs.get(key, {}).get("pending"):
                self._cv.wait()
            st = self._dirs.pop(key, None) or {"calls": 0, "bytes": 0, "waits": 0, "failed": []}
        if st["failed"]:
            raise OSError(f"{len(st['failed'])} of its {st['calls']} calls' prover state wasn't written: {st['failed'][0]}")
        return {k: st[k] for k in ("calls", "bytes", "waits")}

    def close(self) -> None:
        """Every call kept so far written, and the writer's threads ended."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        for t in self._threads:
            t.join()


def locate(rows: Sequence[Mapping[str, Any]], unit: int) -> tuple[int, int]:
    """Proof block `unit` of the window as (its call's index, its leaf in that call's z table), from the manifest `rows`."""
    for row in rows:
        if row["base"] <= unit < row["base"] + row["units"]:
            return int(row["index"]), unit - int(row["base"])
    raise ValueError(f"unit {unit} is no call's: the window's units are its calls' [base, base + units)")


@dataclass(frozen=True)
class Opened:
    """A call's A table and z table (`commitments.MerkleTree`s, None for a call that committed no z), rebuilt."""

    index: int
    a: Any
    z: Any


def open_call(directory: str | Path, record: Mapping[str, Any], index: int, salt: bytes) -> Opened:
    """Call `index` of the window `record` (`pouw_h1.CALLS_FILE`), its tables rebuilt from the prover's directory under
    the window's Program salt `salt` (`pouw_h1_challenger.program_salt_of` its commitment): refused (ValueError) when a
    rebuilt table's root isn't the one the call's manifest row carries."""
    from verity.core.protocols.pouw.protocol import tagged

    from verity_vllm.protocols.pouw.h1 import records as ZR
    from verity_vllm.protocols.pouw.h1 import requests as RQ

    row = record["calls"][index]
    key, t, k, m = key_of(directory), int(row["rows"]), int(row["k"]), int(row["served"])
    path = Path(directory) / CALLS / f"{index}.npz"
    if m:
        with np.load(path) as kept:
            a, s_a, records = kept["a"], kept["s_a"], kept["records"]
    else:
        a, s_a, records = np.zeros((0, k), np.int8), np.zeros(0, "<f4"), np.zeros((int(row["units"]), ZR.ZY_BYTES), np.uint8)

    def table(name: str, leaves: list[bytes]) -> Any:
        tree = RQ.hiding_rows(tagged(RQ.TABLE_TAGS[name], salt, index), index, leaves, key,
                              RQ.TABLES_NONCE + 3 * index + RQ.TABLE_NONCES[name])
        if tree.commitment.root.hex() != (row.get("tables") or {}).get(name):
            raise ValueError(f"call {index}'s {name} table, rebuilt from the prover's directory, is not the one its "
                             "manifest row carries")
        return tree

    at = table("a", RQ.table_leaves(t, k + 4, m, a.reshape(m, k), s_a.reshape(m, 1)))
    z = None if not records.size else table("z", [r.tobytes() for r in records])
    return Opened(index, at, z)


__all__ = ["ALIGN", "CALLS", "KEY", "Opened", "STATE", "Writer", "keep_call", "keep_key", "kept_bytes", "key_of", "locate",
           "open_call", "private", "savez", "writer_of"]
