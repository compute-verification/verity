"""The serve-time commit's hashes on the GPU (README.md): hm96-sha512 hiding leaves over sha512/row/v2 rows, and
frame-v3-sha512 leaves, nodes and pads a level at a time, bit for bit the reference's (``verity.ml.commitments``).

A digest is 8 u64 words, each the big-endian value of its 8 bytes (int64 tensors on a device, uint64 arrays on the host):
``digests(words)`` gives the bytes, ``words(digests)`` the words. ::

    E = Engine()                                         # the current CUDA device; Engine("host") is the host twin
    rows = E.rows([[(0, 0, 32768)], [(1, 0, 16384), (2, 0, 4)]], tensors)   # parts (tensor, byte offset, bytes), at most 2
    x = E.row_digests(rows, slots)                       # (G, S, 8): SHA-512(P(8 nbytes) || row g of slot s)
    lt = E.leaf_table(dids, did_of_row, rank, position, salt_row)   # each row's frame leaf and its salt's index
    leaves = E.leaves(lt, x, SaltKey(key))               # (G, S, 8): the hm96 leaves' frame leaves; or salts=(G, S, 192)
    ft = E.frame_table([slot_did], u0, units)            # each unit's slot-tree leaf (rank u0 + j)
    rest = E.leaves(lt, x, SaltKey(key), out, frames=ft, dest=dest, run=run)   # one-row units' slot-tree leaves into run
    up = E.level(children, dids, level, first)           # (T, w, 8) nodes first .. first + w - 1 over (T, 2w, 8) children
    roots = E.root(leaves_of_trees, dids)                # (T, 8): pads to a power of two, then every level

``tensors`` are (slots, ...) arrays whose slot rows are contiguous bytes (any dtype); a part reads ``nbytes`` at a byte offset
of a slot's row. The salt comes from the caller: ``SaltKey(key, nonce0)`` draws row ``salt_row``'s y in slot s as ChaCha20
chunks 3 salt_row .. 3 salt_row + 2 under ``key`` with nonce ``nonce0 + s`` (C-Flock's ``hm96_salt``, commit_slots'
``row_salts``), or ``salts`` gives every y. Calls run on the current torch stream and don't synchronize.

The host twin counts what it hashes (``counts()``: compressions, those over a constant schedule, ChaCha20 chunks), so the
tests hold each call to its compressions a digest.

Builds are cached by a hash of the sources, flags and compiler version under ``~/.cache/verity/sha512_rows``
(``$SHA512_ROWS_BUILD``); nvcc and g++ are the ones on the path. ``SHA512_ROWS_V`` (a variant name, e.g. ``2211``) picks the
compression of every kernel but the row kernel, which takes ``variant=``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity_config.settings import of

__all__ = ["DEFAULT_VARIANT", "Engine", "LeafTable", "RowTable", "SaltKey", "device_lib", "digests", "host_lib", "words"]

HERE = Path(__file__).resolve().parent
POUW_SHA = HERE.parents[1] / "pouw" / "commit_gpu" / "sha512.cuh"
DEVICE_SOURCES = (POUW_SHA, HERE / "sha512x.cuh", HERE / "rows.cuh", HERE / "rows.cu")
HOST_SOURCES = (HERE / "sha512x.cuh", HERE / "rows.cuh", HERE / "rows_host.cpp")
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", "-lineinfo", "-Xptxas", "-v")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
settings = of(__file__)
BUILD_DIR = Path(settings.SHA512_ROWS_BUILD or Path.home() / ".cache" / "verity" / "sha512_rows")
DEFAULT_VARIANT = settings.SHA512_ROWS_V

INST = hm96.SHA512
KEY = INST.default_key
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
ROOT_SCHEMA = "frame-v3-sha512/root/v1"
MAX_SCHEMA = 64

ROW_SPEC = np.dtype([("a", "<u8"), ("a_stride", "<u8"), ("b", "<u8"), ("b_stride", "<u8"), ("a_len", "<u4"), ("b_len", "<u4"),
                     ("nbytes", "<u4"), ("a16", "<u4"), ("mid", "<u8", (8,)), ("kw", "<u8", (80,))])
LEAF_ROW = np.dtype([("rank", "<u8"), ("position", "<u8"), ("did", "<u4"), ("salt_row", "<u4"), ("q", "<u4"), ("pad_", "<u4"),
                     ("mid", "<u8", (8,)), ("tail", "<u8", (16,))])
assert ROW_SPEC.itemsize == 752 and LEAF_ROW.itemsize == 224

u32, u64, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p


class LeafArgs(ctypes.Structure):
    _fields_ = [("rows", vp), ("x", vp), ("out", vp), ("T", vp), ("salts", vp), ("key", u32 * 8), ("nonce0", u64),
                ("ms_salt", u64 * 8), ("ms_leaf", u64 * 8), ("G", u32), ("S", u32), ("minb", u32), ("sms", u32),
                ("frames", vp), ("dest", vp), ("units_out", vp), ("units", u32), ("pad2_", u32)]


assert ctypes.sizeof(LeafArgs) == 256


def variant_flags(name: str) -> tuple[str, ...]:
    """``-DSX_V``/``-DSX_ROLL`` for a variant name: the modes of S0, S1, s0 and s1 (0 to 3), and ``s`` for straight-line."""
    digits, roll = (name[:-1], 0) if name.endswith("s") else (name, 1)
    if len(digits) != 4 or any(c not in "0123" for c in digits):
        raise ValueError(f"a variant is four digits 0-3 and an optional 's', not {name!r}")
    v = sum(int(c) << (2 * i) for i, c in enumerate(digits))
    return (f"-DSX_V={v}", f"-DSX_ROLL={roll}")


def _build(tool: str, sources: Sequence[Path], flags: Sequence[str], name: str) -> ctypes.CDLL:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        r = subprocess.run([tool, *flags, "-o", str(tmp), str(sources[-1])], capture_output=True, text=True, check=False)
        lib.with_suffix(".log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{tool} failed:\n{r.stderr[-4000:]}")
        tmp.replace(lib)
    return _abi(ctypes.CDLL(str(lib)), lib)


def _abi(dll: ctypes.CDLL, path: Path) -> ctypes.CDLL:
    dll.sxr_variants.restype = ctypes.c_int
    dll.sxr_variant_name.argtypes, dll.sxr_variant_name.restype = [ctypes.c_int], ctypes.c_char_p
    dll.sxr_default.restype = ctypes.c_char_p
    dll.sxr_midstate.argtypes, dll.sxr_midstate.restype = [vp, vp], None
    dll.sxr_row_setup.argtypes = [vp, vp, u32]
    dll.sxr_rows.argtypes = [vp, ctypes.c_int, vp, vp, u32, u32, vp]
    dll.sxr_leaf_setup.argtypes = [vp, vp, u32, vp, vp, u32]
    dll.sxr_leaves.argtypes = [vp, ctypes.POINTER(LeafArgs)]
    dll.sxr_frame_leaves.argtypes = [vp, vp, u32, vp, u32, u32, u64, vp, u32, vp]
    dll.sxr_frame_setup.argtypes = [vp, vp, u32, u64, vp, vp, u32]
    dll.sxr_frame_units.argtypes = [vp, vp, vp, vp, u32, u32, u32, vp]
    dll.sxr_node_mids.argtypes = [vp, vp, u32, u32, vp]
    dll.sxr_level.argtypes = [vp, vp, u32, vp, u64, vp, u64, u32, u32, u64, vp]
    dll.sxr_pads.argtypes = [vp, vp, u32, u32, u64, u32, vp]
    for f in ("sxr_row_setup", "sxr_rows", "sxr_leaf_setup", "sxr_leaves", "sxr_frame_leaves", "sxr_frame_setup",
              "sxr_frame_units", "sxr_node_mids", "sxr_level", "sxr_pads"):
        getattr(dll, f).restype = ctypes.c_int
    if hasattr(dll, "sxr_counts"):
        dll.sxr_counts.argtypes, dll.sxr_counts.restype = [vp, ctypes.c_int], None
    if hasattr(dll, "sxr_graph_rekey"):
        dll.sxr_graph_leaf_nodes.argtypes, dll.sxr_graph_leaf_nodes.restype = [vp, vp, u32], ctypes.c_int
        dll.sxr_graph_rekey.argtypes, dll.sxr_graph_rekey.restype = [vp, vp, u32, vp], ctypes.c_int
    dll.path = path  # type: ignore[attr-defined]
    return dll


@cache
def host_lib() -> ctypes.CDLL:
    """``rows_host.cpp`` built with the system C++ compiler."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    return _build(cxx, HOST_SOURCES, CFLAGS, "libsha512_rows_host.so")


@cache
def device_lib(arch: str, default: str = DEFAULT_VARIANT) -> ctypes.CDLL:
    """``rows.cu`` built with nvcc for ``sm_<arch>`` (``"90a"``: the H200), its kernels but the row kernel on ``default``."""
    tool = shutil.which("nvcc")
    if tool is None:
        raise RuntimeError("no nvcc on the path")
    flags = (*NVCC_FLAGS, *variant_flags(default), f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return _build(tool, DEVICE_SOURCES, flags, "libsha512_rows_cuda.so")


def words(digests: Sequence[bytes] | bytes) -> np.ndarray:
    """64-byte digests as (n, 8) uint64 words."""
    raw = digests if isinstance(digests, bytes) else b"".join(digests)
    if len(raw) % 64:
        raise ValueError("digests are 64 bytes each")
    return np.frombuffer(raw, dtype=">u8").astype(np.uint64).reshape(-1, 8)


def digests(w: Any) -> list[bytes]:
    """Words (any shape ending in 8, a tensor or an array) as their 64-byte digests, in order."""
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    raw = np.ascontiguousarray(np.asarray(w).view(np.uint64)).reshape(-1, 8).astype(">u8").tobytes()
    return [raw[64 * i:64 * (i + 1)] for i in range(len(raw) // 64)]


def hankel_table(key: bytes = KEY) -> np.ndarray:
    """hm96-sha512's Hankel table under ``key``, (256, 256) bytes: T_v is the xor of ``k >> s`` over the set bits s of v (k the
    key as a little-endian integer), 256 bytes little-endian, so bytes [p, p + 64) of T_v are ``byte_tables(key)[p][v]``
    (commit_gpu's and commit_slots' table; refused for a key where that fails)."""
    k = int.from_bytes(key, "little")
    rows = [0] * 256
    for v in range(1, 256):
        rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
    T = [r.to_bytes(256, "little") for r in rows]
    tables = INST.byte_tables(key)
    if any(T[v][p:p + 64] != tables[p][v].to_bytes(64, "little") for p in range(len(tables)) for v in range(256)):
        raise ValueError("this key's byte tables are not windows of one Hankel table")
    return np.frombuffer(b"".join(T), dtype=np.uint8).reshape(256, 256)


@dataclass(frozen=True)
class SaltKey:
    """Salts from a 32-byte key: row r's y in slot s is ChaCha20 chunks 3r .. 3r + 2 with nonce ``nonce0 + s``."""

    key: bytes
    nonce0: int = 0

    def __post_init__(self) -> None:
        if len(self.key) != 32 or not 0 <= self.nonce0 < 1 << 64:
            raise ValueError("a salt key is 32 bytes and its first nonce a u64")


@dataclass
class RowTable:
    """A bound row table: each row's parts at the tensors' addresses, on the engine's device (keep the tensors alive), and
    the order the row kernel starts them in (the most compressions first; the digests stay in table order)."""

    spec: Any
    G: int
    slots: int
    nbytes: tuple[int, ...]
    tensors: tuple[Any, ...]
    order: Any = None


@dataclass
class LeafTable:
    """Each row's frame leaf (its domain, rank and position) and its salt's row index, set up on the engine's device."""

    rows: Any
    dids: Any
    G: int


def _did_words(dids: Any) -> np.ndarray:
    if isinstance(dids, bytes):
        return words(dids)
    if isinstance(dids, (list, tuple)):
        return words(list(dids))
    a = np.asarray(dids).view(np.uint64)
    return np.ascontiguousarray(a.reshape(-1, 8))


class Engine:
    """The engine's calls on ``device``: a CUDA device (torch tensors), or ``"host"`` for the host twin (numpy arrays)."""

    def __init__(self, device: Any = "cuda", arch: str | None = None, key: bytes = KEY, default: str = DEFAULT_VARIANT) -> None:
        self._schemas: dict[str, tuple[Any, int]] = {}
        self.host = device == "host"
        if self.host:
            self.torch, self.device, self.lib = None, "host", host_lib()
        else:
            import torch
            self.torch, self.device = torch, torch.device(device)
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if arch is None:
                major, minor = torch.cuda.get_device_capability(self.device)
                arch = f"{major}{minor}" + ("a" if major in (9, 10, 12) else "")
            self.lib = device_lib(arch, default)
        self.variants = [self.lib.sxr_variant_name(i).decode() for i in range(self.lib.sxr_variants())]
        self.default = self.lib.sxr_default().decode()
        self.key = INST.check_key(key)
        self.table = self._up(hankel_table(self.key).reshape(-1))
        self.ms_salt = self.midstate(INST.salt_prefix)
        self.ms_leaf = self.midstate(INST.leaf_prefix(self.key))

    # --- memory ----------------------------------------------------------------------------------------------------------

    def _up(self, a: np.ndarray) -> Any:
        a = np.ascontiguousarray(a)
        if self.host:
            return a.copy()
        t = self.torch.from_numpy(a.view(np.uint8).reshape(-1).copy())
        return t.to(self.device, non_blocking=False)

    def _down(self, t: Any, dtype: Any) -> np.ndarray:
        return (t if self.host else t.cpu().numpy()).view(dtype)

    def _words(self, *shape: int) -> Any:
        if self.host:
            return np.zeros((*shape, 8), dtype=np.uint64)
        return self.torch.empty((*shape, 8), dtype=self.torch.int64, device=self.device)

    def _wordsof(self, a: Any) -> Any:
        """A digests argument on the engine's device: words as given there, or bytes / arrays uploaded."""
        if not self.host and hasattr(a, "data_ptr"):
            return a if a.dtype == self.torch.int64 else a.view(self.torch.int64)
        if self.host and isinstance(a, np.ndarray):
            return np.ascontiguousarray(a.view(np.uint64))
        w = _did_words(a)
        return w.copy() if self.host else self._up(w).view(self.torch.int64).reshape(-1, 8)

    @staticmethod
    def _ptr(a: Any) -> int:
        if a is None:
            return 0
        return a.ctypes.data if isinstance(a, np.ndarray) else a.data_ptr()

    def _stream(self) -> int:
        return 0 if self.host else self.torch.cuda.current_stream(self.device).cuda_stream

    def _ok(self, rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what}: CUDA error {rc}")

    def counts(self, reset: bool = False) -> dict[str, int]:
        """The host twin's hashing since its last reset: compressions, those over a constant schedule (``kw``) and ChaCha20
        chunks."""
        if not self.host:
            raise ValueError("only the host twin counts")
        out = np.zeros(3, dtype=np.uint64)
        self.lib.sxr_counts(out.ctypes.data, int(reset))
        return {"compressions": int(out[0]), "kw": int(out[1]), "chacha": int(out[2])}

    def midstate(self, block: bytes) -> np.ndarray:
        """The chaining value after one 128-byte chunk from the IV."""
        if len(block) != 128:
            raise ValueError("a midstate is of one 128-byte block")
        out = np.zeros(8, dtype=np.uint64)
        buf = ctypes.create_string_buffer(block, 128)
        self.lib.sxr_midstate(ctypes.addressof(buf), out.ctypes.data)
        return out

    # --- rows ------------------------------------------------------------------------------------------------------------

    def rows(self, rows: Sequence[Sequence[tuple[int, int, int]]], tensors: Sequence[Any]) -> RowTable:
        """Bind rows (each one or two parts (tensor, byte offset, bytes) of a slot's row) to ``tensors`` (slots, ...)."""
        if not tensors:
            raise ValueError("no tensors")
        slots = int(tensors[0].shape[0])
        base, stride, width = [], [], []
        for t in tensors:
            if int(t.shape[0]) != slots:
                raise ValueError("every tensor has the same slots")
            if self.host:
                a = np.asarray(t)
                row = a[0]
                if not row.flags.c_contiguous:
                    raise ValueError("a slot's row is contiguous")
                base.append(a.ctypes.data)
                stride.append(a.strides[0] if slots > 1 else row.nbytes)
                width.append(row.nbytes)
            else:
                if t.device != self.device or not t[0].is_contiguous():
                    raise ValueError("tensors are on the engine's device and a slot's row is contiguous")
                base.append(t.data_ptr())
                stride.append(t.stride(0) * t.element_size())
                width.append(t[0].numel() * t.element_size())
        spec = np.zeros(len(rows), dtype=ROW_SPEC)
        nbytes = []
        for g, parts in enumerate(rows):
            if not 1 <= len(parts) <= 2:
                raise ValueError("a row is one or two parts")
            for (src, off, n), (p, ps, pl) in zip(parts, (("a", "a_stride", "a_len"), ("b", "b_stride", "b_len")), strict=False):
                if not 0 <= src < len(tensors) or off < 0 or n < 0 or off + n > width[src]:
                    raise ValueError(f"row {g}'s part {(src, off, n)} is outside its tensor's {width[src]}-byte rows")
                spec[g][p], spec[g][ps], spec[g][pl] = base[src] + off, stride[src], n
            spec[g]["nbytes"] = int(spec[g]["a_len"]) + int(spec[g]["b_len"])
            spec[g]["a16"] = int(spec[g]["a"] % 16 == 0 and spec[g]["a_stride"] % 16 == 0)
            nbytes.append(int(spec[g]["nbytes"]))
        if not 0 < len(rows) < 1 << 32 or not 0 < slots < 1 << 32:
            raise ValueError("rows and slots are u32 and at least 1")
        d = self._up(spec)
        self._ok(self.lib.sxr_row_setup(self._stream(), self._ptr(d), len(rows)), "row setup")
        order = np.argsort(-((np.asarray(nbytes, dtype=np.int64) + 17 + 127) // 128), kind="stable").astype(np.uint32)
        return RowTable(d, len(rows), slots, tuple(nbytes), tuple(tensors), self._up(order))

    def row_digests(self, table: RowTable, variant: str | None = None, out: Any = None) -> Any:
        """(G, S, 8): x = SHA-512(P(8 nbytes) || row g's bytes in slot s), sha512/row/v2."""
        name = variant or self.default
        if self.host:
            name = "host"
        if name not in self.variants:
            raise ValueError(f"no variant {name!r}; built: {self.variants}")
        out = self._words(table.G, table.slots) if out is None else out
        rc = self.lib.sxr_rows(self._stream(), self.variants.index(name), self._ptr(table.spec), self._ptr(table.order), table.G,
                               table.slots, self._ptr(out))
        self._ok(rc, "rows")
        return out

    # --- hm96 leaves -----------------------------------------------------------------------------------------------------

    def leaf_table(self, dids: Any, did_of_row: Sequence[int], rank: Sequence[int], position: Sequence[int] | None = None,
                   salt_row: Sequence[int] | None = None, schema: str = ROW_SCHEMA) -> LeafTable:
        """Row g's frame leaf is domain ``dids[did_of_row[g]]``'s leaf at (rank, position) under ``schema``; its salt is the
        salt stream's row ``salt_row[g]`` (default g). Position defaults to the rank."""
        G = len(rank)
        d_schema, n_schema = self._schema(schema)
        dw = _did_words(dids)
        rows = np.zeros(G, dtype=LEAF_ROW)
        rows["did"] = np.asarray(did_of_row, dtype=np.uint32)
        rows["rank"] = np.asarray(rank, dtype=np.uint64)
        rows["position"] = np.asarray(rank if position is None else position, dtype=np.uint64)
        rows["salt_row"] = np.arange(G, dtype=np.uint32) if salt_row is None else np.asarray(salt_row, dtype=np.uint32)
        if len(did_of_row) != G or int(rows["did"].max(initial=0)) >= len(dw):
            raise ValueError("each row names one of the domains")
        d_rows, d_dids = self._up(rows), self._up(dw)
        rc = self.lib.sxr_leaf_setup(self._stream(), self._ptr(d_rows), G, self._ptr(d_dids), self._ptr(d_schema), n_schema)
        self._ok(rc, "leaf setup")
        return LeafTable(d_rows, (d_dids, d_schema), G)

    def _schema(self, schema: str) -> tuple[Any, int]:
        """The schema's bytes on the device, uploaded once."""
        if schema not in self._schemas:
            s = schema.encode()
            if len(s) > MAX_SCHEMA:
                raise ValueError(f"a schema is at most {MAX_SCHEMA} bytes")
            self._schemas[schema] = self._up(np.frombuffer(s.ljust(1, b"\0"), dtype=np.uint8)), len(s)
        return self._schemas[schema]

    def frame_table(self, dids: Any, rank0: int, n: int, schema: str = ROOT_SCHEMA) -> LeafTable:
        """n units' frame leaves in one domain (``dids``, one): unit j's is leaf (rank0 + j, rank0 + j, schema) over its root,
        set up on the device (no upload a unit). What ``leaves(frames=)`` and ``frame_units`` take."""
        dw = self._wordsof(dids)
        if int(dw.shape[0]) != 1 or not 0 <= rank0 or rank0 + n > 1 << 64 or not 0 <= n < 1 << 32:
            raise ValueError("one domain, u64 ranks and a u32 count")
        d_schema, n_schema = self._schema(schema)
        if self.host:
            rows = np.zeros(n, dtype=LEAF_ROW)
        else:
            rows = self.torch.empty(n * LEAF_ROW.itemsize, dtype=self.torch.uint8, device=self.device)
        rc = self.lib.sxr_frame_setup(self._stream(), self._ptr(rows), n, rank0, self._ptr(dw), self._ptr(d_schema), n_schema)
        self._ok(rc, "frame setup")
        return LeafTable(rows, (dw, d_schema), n)

    def frame_units(self, frames: LeafTable, units: Any, values: Any, run: Any) -> Any:
        """run[s][units[i]] = the frame leaf of values[i][s] under ``frames``' unit units[i] (``units`` int64 on the device,
        ``values`` (n, S, 8), ``run`` (S, frames.G, 8)); returns run."""
        n, S = int(values.shape[0]), int(values.shape[1])
        if len(units) != n or tuple(run.shape) != (S, frames.G, 8):
            raise ValueError(f"values are (units, S, 8) and run (S, {frames.G}, 8)")
        v = self._wordsof(values)
        rc = self.lib.sxr_frame_units(self._stream(), self._ptr(frames.rows), self._ptr(units), self._ptr(v), n, S, frames.G,
                                      self._ptr(run))
        self._ok(rc, "frame units")
        return run

    def leaves(self, lt: LeafTable, x: Any, salts: SaltKey | Any, out: Any = None, minb: int = 0, frames: LeafTable | None = None,
               dest: Any = None, run: Any = None, sms: int = 0) -> Any:
        """(G, S, 8): each row's frame leaf over hm96's tree leaf of x (``row_digests``) under salt y. ``minb`` is the leaf
        kernel's CTAs an SM (1 or 2; 0, the build's default, 1), and ``sms`` the SMs its persistent grid fills (0, the
        device's; a green context of fewer runs a grid of the device's in waves): the same bytes either way.

        With ``frames`` (``frame_table``, a unit each), ``dest`` (G,) uint32 on the device and ``run`` (S, units, 8): a row
        whose ``dest`` d is below the units is unit d alone, so its leaf is the unit's root, and the unit's frame leaf over it
        goes to run[s][d], in the same thread; any other row's leaf goes to out[d - units][s] (``out`` given, (rest, S, 8)).
        Returns ``out``."""
        G, S = int(x.shape[0]), int(x.shape[1])
        if G != lt.G:
            raise ValueError(f"x has {G} rows, the leaf table {lt.G}")
        if minb not in (0, 1, 2):
            raise ValueError("a leaf kernel runs 1 or 2 blocks an SM")
        if frames is not None and (dest is None or run is None or len(dest) != G or tuple(run.shape) != (S, frames.G, 8)):
            raise ValueError(f"frames take a dest a row and run (S, {frames.G}, 8)")
        out = self._words(G, S) if out is None and frames is None else out
        a = LeafArgs()
        a.rows, a.x, a.out, a.T = self._ptr(lt.rows), self._ptr(x), self._ptr(out), self._ptr(self.table)
        if isinstance(salts, SaltKey):
            a.salts, a.nonce0 = 0, salts.nonce0
            a.key[:] = list(np.frombuffer(salts.key, dtype="<u4"))
        else:
            if tuple(salts.shape) != (G, S, 192):
                raise ValueError(f"salts are (G, S, 192) bytes, not {tuple(salts.shape)}")
            a.salts = self._ptr(salts)
        a.ms_salt[:], a.ms_leaf[:] = [int(v) for v in self.ms_salt], [int(v) for v in self.ms_leaf]
        if sms < 0:
            raise ValueError("a leaf grid fills a positive number of SMs, or 0 for the device's")
        a.G, a.S, a.minb, a.sms = G, S, minb, sms
        if frames is not None:
            a.frames, a.dest, a.units_out, a.units = self._ptr(frames.rows), self._ptr(dest), self._ptr(run), frames.G
        self._ok(self.lib.sxr_leaves(self._stream(), ctypes.byref(a)), "leaves")
        return out

    def graph_leaf_nodes(self, graph: int, n: int) -> Any:
        """The ``n`` leaf launches (``leaves``) of a CUDA graph captured from this engine's calls (a ``cudaGraph_t``), for
        ``rekey``; refused if the graph holds another number of them."""
        nodes = (ctypes.c_void_p * max(1, n))()
        got = self.lib.sxr_graph_leaf_nodes(ctypes.c_void_p(graph), nodes, n)
        if got < 0:
            raise RuntimeError(f"graph leaf nodes: CUDA error {-got}")
        if got != n:
            raise RuntimeError(f"the graph holds {got} leaf launches, not {n}")
        return nodes

    def rekey(self, graph_exec: int, nodes: Any, n: int, key: bytes) -> None:
        """The graph's instantiation (a ``cudaGraphExec_t``) to salt its ``n`` leaf launches ``nodes`` under ``key`` (32
        bytes) from its next launch on, each launch's first nonce as captured."""
        if len(key) != 32:
            raise ValueError("a salt key is 32 bytes")
        k = (ctypes.c_uint32 * 8)(*np.frombuffer(key, dtype="<u4").tolist())
        self._ok(self.lib.sxr_graph_rekey(ctypes.c_void_p(graph_exec), nodes, n, k), "graph rekey")

    def u32s(self, a: Any) -> Any:
        """A uint32 array on the engine's device (int32 there): ``leaves``' ``dest``."""
        a = np.ascontiguousarray(a, dtype=np.uint32)
        return a.copy() if self.host else self._up(a).view(self.torch.int32)

    def commit_rows(self, table: RowTable, lt: LeafTable, salts: SaltKey | Any, variant: str | None = None) -> Any:
        """Rows in, leaf digests out: ``leaves(lt, row_digests(table), salts)``."""
        return self.leaves(lt, self.row_digests(table, variant), salts)

    # --- trees -----------------------------------------------------------------------------------------------------------

    def frame_leaves(self, values: Any, dids: Any, rank0: int = 0, schema: str = ROOT_SCHEMA) -> Any:
        """(T, n, 8): leaf (rank0 + j, rank0 + j, schema, values[t][j]) of tree t's domain (one domain, or one a tree)."""
        T, n = int(values.shape[0]), int(values.shape[1])
        dw = self._wordsof(dids)
        D = int(dw.shape[0])
        if D not in (1, T):
            raise ValueError("one domain, or one a tree")
        d_schema, n_schema = self._schema(schema)
        out = self._words(T, n)
        v = self._wordsof(values)
        rc = self.lib.sxr_frame_leaves(self._stream(), self._ptr(dw), D, self._ptr(v), T, n, rank0, self._ptr(d_schema), n_schema,
                                       self._ptr(out))
        self._ok(rc, "frame leaves")
        return out

    def node_mids(self, dids: Any, level: int) -> Any:
        """(D, 8): each domain's node midstate at ``level``, what ``level(..., mids=)`` takes."""
        if not 0 <= level < 256:
            raise ValueError("a level is below 256")
        dw = self._wordsof(dids)
        mids = self._words(int(dw.shape[0]))
        self._ok(self.lib.sxr_node_mids(self._stream(), self._ptr(dw), int(dw.shape[0]), level, self._ptr(mids)), "node midstates")
        return mids

    def level(self, children: Any, dids: Any, level: int, first: int = 0, mids: Any = None, pre: Any = None) -> Any:
        """(T, w, 8): node (level, first + i) over children [2i] and [2i + 1] of (T, 2w, 8) in tree t's domain (``mids``:
        ``node_mids(dids, level)``, if already computed). With ``pre`` (T, 8), tree t's children are pre[t] and then
        children[t] ((T, 2w - 1, 8)). Each tree's digests are read in place: any stride between trees (a slice of a wider
        array) and none copied."""
        (kids, k_stride), T, n = self._trees(children), int(children.shape[0]), int(children.shape[1])
        if pre is not None:
            if tuple(pre.shape) != (T, 8):
                raise ValueError("pre is a digest a tree")
            (pre, p_stride), n = self._trees(pre[:, None, :]), n + 1
        if n % 2 or not 0 <= level < 256 or not 0 <= first < 1 << 64:
            raise ValueError("a level has an even width and is below 256, and its first index is a u64")
        mids = self.node_mids(dids, level) if mids is None else mids
        D = int(mids.shape[0])
        if D not in (1, T):
            raise ValueError("one domain, or one a tree")
        out = self._words(T, n // 2)
        rc = self.lib.sxr_level(self._stream(), self._ptr(mids), D, self._ptr(pre), p_stride if pre is not None else 0,
                                self._ptr(kids), k_stride, T, n // 2, first, self._ptr(out))
        self._ok(rc, "level")
        return out

    def _trees(self, a: Any) -> tuple[Any, int]:
        """(T, n, 8) digest words with each tree's n digests contiguous, as given when they are (else a copy), and the words
        from one tree to the next."""
        if self.host and isinstance(a, np.ndarray):
            a = a.view(np.uint64)
            if a.strides[2] != 8 or (a.shape[1] > 1 and a.strides[1] != 64) or a.strides[0] % 8:
                a = np.ascontiguousarray(a)
            return a, a.strides[0] // 8
        if not self.host and hasattr(a, "data_ptr"):
            a = a if a.dtype == self.torch.int64 else a.view(self.torch.int64)
            if a.stride(2) != 1 or (a.shape[1] > 1 and a.stride(1) != 8):
                a = a.contiguous()
            return a, a.stride(0)
        shape = tuple(int(d) for d in a.shape)
        return self._wordsof(a).reshape(shape), 8 * shape[1]

    def pads(self, dids: Any, trees: int, first: int, n: int) -> Any:
        """(trees, n, 8): pad(first + j) of tree t's domain."""
        dw = self._wordsof(dids)
        out = self._words(trees, n)
        self._ok(self.lib.sxr_pads(self._stream(), self._ptr(dw), int(dw.shape[0]), trees, first, n, self._ptr(out)), "pads")
        return out

    def root(self, leaves: Any, dids: Any) -> Any:
        """(T, 8): each tree's root over its (T, n, 8) leaves, padded to a power of two (merkle.tree_levels)."""
        T, n = int(leaves.shape[0]), int(leaves.shape[1])
        depth = merkle_depth(n)
        dw = self._wordsof(dids)
        level = self._wordsof(leaves)
        if (1 << depth) > n:
            shared = int(dw.shape[0]) == 1
            p = self.pads(dw, 1 if shared else T, n, (1 << depth) - n)
            if self.host:
                level = np.concatenate([level, np.broadcast_to(p, (T, *p.shape[1:]))], axis=1)
            else:
                level = self.torch.cat([level, p.expand(T, -1, -1)], dim=1)
        for lv in range(depth):
            level = self.level(level, dw, lv)
        return level.reshape(T, 8)
