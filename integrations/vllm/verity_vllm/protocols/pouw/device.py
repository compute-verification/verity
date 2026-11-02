"""`ncp-v2` on the device: the unit functions under `pouw_cuda/` (`ncp2.cuh`), and the gate that holds any build of them to
`pouw_native`, leaf for leaf.

`ncp2.cuh` computes each unit's values as `pouw_native` does (the noise from SHAKE256, X and Y, route U's checkpoints, the
exact dequantization and each unit's SHA-256 leaf), in functions that compile for the host and the device. `host()` builds
them for the CPU with the system C++ compiler, keyed by what it compiles, and `commit_call` commits a call through that
build; `device()` builds `ncp2_kernels.cu` with nvcc, whose committer calls the same functions and is driven the same way.
Each build returns the call's unit leaves and its weight side's Y unit leaves; the first call on a weight makes the weight
side's record from them, and a later call's must be that record's. `first_difference` is the gate: the first unit, in the
partition's numbering, whose leaf a record gets wrong, named by role and place, the first wrong Y unit of its weight side,
or the first wrong served entry. A device committer passes when it names nothing on `gate_calls`.
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from verity.ml.protocols.pouw.circuit import ncp2

from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import native as PN

HERE = Path(__file__).with_name("pouw_cuda")
SOURCES = ("ncp2.cuh", "ncp2_fast.cuh", "ncp2_host.cpp")
CFLAGS = ("-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-fno-exceptions", "-fno-rtti", "-fPIC", "-shared")
CUDA_SOURCES = ("ncp2.cuh", "ncp2_kernels.cu")
FUSED_SOURCES = ("ncp2.cuh", "ncp2_fast.cuh", "ncp2_fused.cu")
#: `ncp2_fused`'s flags: hash (1), time the tile kernel (2), cache Y's leaves per weight (4), the tiles' low-byte leaf variant (8;
#: its γ is unproved, so it is timed only and never a result)
HASH, TIME_TILES, CACHE_Y, LOWBYTE = 1, 2, 4, 8
#: the low-byte variant's TurboSHAKE128 domain byte, and its layout: 128 low bytes (8 rows of 16) per 168-byte block
LOWBYTE_DS, LOWBYTE_BLOCK, TS128_RATE = 0x0B, 128, 168
#: no --use_fast_math: the dequantization's steps are IEEE binary32, with subnormals, and nothing is contracted
NVCC_FLAGS = ("-O3", "-std=c++17", "-fmad=false", "-Xcompiler", "-fPIC", "-shared")
HASHES = {"sha256": (0, 32), "sha512": (1, 64), "shake256": (2, None), "turboshake128": (3, None)}

_u8, _u16, _u32, _u64 = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.uint8, np.uint16, np.uint32, np.uint64))
_i8 = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
_COMMIT = [ctypes.c_int] * 3 + [_i8, _u32, _i8, _u32, _u16, _u8] + [ctypes.c_uint64] * 3 + [_u32] * 4 + [_u8, _u8, _u16]


@dataclass
class Toolchain:
    """What the builds use: a C++ compiler and nvcc (None: the first on the path), the CUDA arch (89, the RTX 4090) and the
    directory the libraries are built into. A command line sets it once with `configure`."""

    cxx: str | None = None
    nvcc: str | None = None
    arch: str = "89"
    build_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "verity" / "pouw_cuda")


TOOLCHAIN = Toolchain()


def configure(**changes) -> Toolchain:
    """Set `TOOLCHAIN`'s fields (``cxx``, ``nvcc``, ``arch``, ``build_dir``; None leaves one as it is) and forget every loaded
    library, so the next load builds with them."""
    for key, value in changes.items():
        if not hasattr(TOOLCHAIN, key):
            raise TypeError(f"no toolchain field {key!r}")
        if value is not None:
            setattr(TOOLCHAIN, key, Path(value) if key == "build_dir" else value)
    for loader in (host, device, fused):
        getattr(loader, "cache_clear", lambda: None)()
    return TOOLCHAIN


def compiler() -> str | None:
    """`TOOLCHAIN`'s C++ compiler, or the first of g++, c++ and clang++ on the path."""
    return TOOLCHAIN.cxx or next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)


def nvcc() -> str | None:
    """`TOOLCHAIN`'s nvcc, or nvcc on the path."""
    return TOOLCHAIN.nvcc or shutil.which("nvcc")


def _need(tool: str | None, what: str) -> str:
    if tool is None:
        raise RuntimeError(f"no {what} (on the path, or `configure`)")
    return tool


def cuda_flags() -> tuple[str, ...]:
    return (*NVCC_FLAGS, f"-gencode=arch=compute_{TOOLCHAIN.arch},code=sm_{TOOLCHAIN.arch}")


def source_digest(sources: Sequence[str] = SOURCES, flags: Sequence[str] = (), tool: str | None = None) -> str:
    """SHA-256 over the sources (name and bytes, in order), the flags (default `CFLAGS`) and the compiler's version."""
    h = hashlib.sha256()
    for name in sources:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    version = subprocess.run([tool or _need(compiler(), "C++ compiler"), "--version"], capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()
    h.update(" ".join(flags or CFLAGS).encode() + b"\0" + "|".join(version).encode())
    return h.hexdigest()


def _build(tool: str, sources: Sequence[str], flags: Sequence[str], name: str) -> ctypes.CDLL:
    """``sources[-1]`` built in `TOOLCHAIN`'s ``build_dir`` under the first 12 hex digits of `source_digest`, so a changed
    source never loads another's library."""
    lib = TOOLCHAIN.build_dir / source_digest(sources, flags, tool)[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([tool, *flags, "-o", str(tmp), str(HERE / sources[-1])], check=True)
        tmp.replace(lib)
    return ctypes.CDLL(str(lib))


def _bind_hash_commit(dll: ctypes.CDLL) -> None:
    dll.ncp2_hash.argtypes = [ctypes.c_int, ctypes.c_int, _u8, _u64, ctypes.c_uint64, ctypes.c_uint64, _u8]
    dll.ncp2_hash.restype = dll.ncp2_commit.restype = ctypes.c_int


@cache
def host() -> ctypes.CDLL:
    """`ncp2_host.cpp` built for this CPU with the system C++ compiler."""
    dll = _build(_need(compiler(), "C++ compiler"), SOURCES, CFLAGS, "libncp2_host.so")
    _bind_hash_commit(dll)
    dll.ncp2_commit.argtypes = dll.ncp2_commit_fast.argtypes = _COMMIT
    dll.ncp2_units_root_fast.argtypes = [_u8, ctypes.c_uint64, _u8]
    dll.ncp2_tile_lowbyte.argtypes = [_u8, _u8, _u32, ctypes.c_int, ctypes.c_int, _u8]
    dll.ncp2_commit_fast.restype = dll.ncp2_units_root_fast.restype = dll.ncp2_tile_lowbyte.restype = ctypes.c_int
    return dll


@cache
def device() -> ctypes.CDLL:
    """`ncp2_kernels.cu` built with nvcc for ``$VERITY_POUW_CUDA_ARCH`` (default 89, the RTX 4090)."""
    dll = _build(_need(nvcc(), "nvcc"), CUDA_SOURCES, cuda_flags(), "libncp2_cuda.so")
    _bind_hash_commit(dll)
    fp, dp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    dll.ncp2_commit.argtypes = _COMMIT + [_u8, fp]
    dll.ncp2_ceiling.argtypes = [ctypes.c_int] * 4 + [dp]
    dll.ncp2_leaf_time.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, fp]
    dll.ncp2_f1_time.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_int, fp]
    for f in (dll.ncp2_ceiling, dll.ncp2_leaf_time, dll.ncp2_f1_time):
        f.restype = ctypes.c_int
    return dll


@cache
def fused() -> ctypes.CDLL:
    """`ncp2_fused.cu`, the serving kernel, built with nvcc like `device`."""
    dll = _build(_need(nvcc(), "nvcc"), FUSED_SOURCES, cuda_flags(), "libncp2_fused.so")
    vp, fp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
    dll.ncp2_fused.argtypes = ([ctypes.c_int] * 3 + [vp] * 5 + [_u8] + [ctypes.c_uint64] * 3 + [vp] * 4
                               + [ctypes.c_int, ctypes.c_uint64, vp, vp, vp, vp])
    dll.ncp2_fused_host.argtypes = _COMMIT + [_u8, fp, ctypes.c_int]
    dll.ncp2_fused_tile_ms.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)]
    dll.ncp2_fused_reset_cache.argtypes = []
    for f in (dll.ncp2_fused, dll.ncp2_fused_host, dll.ncp2_fused_tile_ms, dll.ncp2_fused_reset_cache):
        f.restype = ctypes.c_int
    return dll


def _lib(on: str) -> ctypes.CDLL:
    return {"host": host, "twin": host, "device": device, "fused": fused}[on]()


def lowbyte_leaf(words: np.ndarray) -> bytes:
    """The low-byte variant's leaf (its γ is unproved: timed only, never a result) over a tile's checkpoints (a tile unit's outputs, in order): TurboSHAKE128 with domain byte
    `LOWBYTE_DS` over each word's low byte, `LOWBYTE_BLOCK` bytes then zeros to the rate per block, the last block unpadded."""
    low = (np.asarray(words, np.uint32) & 0xFF).astype(np.uint8).tobytes()
    blocks = [low[i:i + LOWBYTE_BLOCK] for i in range(0, len(low), LOWBYTE_BLOCK)]
    msg = b"".join(bl + bytes(TS128_RATE - LOWBYTE_BLOCK) for bl in blocks[:-1]) + blocks[-1]
    return hash_many("turboshake128", [msg], 32, LOWBYTE_DS)[0]


def tile_lowbyte_host(xs: np.ndarray, ys: np.ndarray, c0: np.ndarray, steps: int) -> bytes:
    """`tile_fused_lowbyte` on the CPU: X and Y strips (16 × 3k bytes each) and the strip's c₀."""
    out = np.zeros(32, np.uint8)
    host().ncp2_tile_lowbyte(np.ascontiguousarray(xs, np.uint8).reshape(-1), np.ascontiguousarray(ys, np.uint8).reshape(-1),
                             np.ascontiguousarray(c0, np.uint32), xs.shape[1], steps, out)
    return out.tobytes()


def units_root_twin(leaves: bytes) -> bytes:
    """`pouw_circuit.units_root` by the host build's fast node (`ncp2_fast.cuh` `units_node_fast`)."""
    root = np.zeros(32, np.uint8)
    host().ncp2_units_root_fast(np.frombuffer(leaves, np.uint8).copy(), len(leaves) // 32, root)
    return root.tobytes()


def hash_many(kind: str, msgs: Sequence[bytes], out: int | None = None, ds: int = 0x1F, on: str = "host") -> list[bytes]:
    """Each message hashed by the ``on`` build ("host" or "device"): ``kind`` one of `HASHES`; ``out`` bytes for the XOFs;
    ``ds`` their domain byte."""
    code, width = HASHES[kind]
    width = width or out or 32
    offsets = np.cumsum([0, *map(len, msgs)], dtype=np.uint64)
    data = np.frombuffer(b"".join(msgs) or b"\0", np.uint8).copy()
    res = np.zeros(len(msgs) * width, np.uint8)
    if rc := _lib(on).ncp2_hash(code, ds, data, offsets, len(msgs), width, res):
        raise ValueError(f"{kind}: {width}-byte output, or CUDA error {rc}")
    return [res[i * width:(i + 1) * width].tobytes() for i in range(len(msgs))]


def side_of(scheme: str, call: PC.Call, yleaves: bytes, b: np.ndarray) -> PC.WeightRecord:
    """The call's weight side's record from a build's Y unit leaves and the weight's rows ``b`` as read: its inputs' digest
    and what each call on it reads (`WeightRecord.reads`) computed on the host, once per run and weight."""
    nw = PN.NativeWeight(call.b, scheme, call.salt, call.weight, call.colids)
    rd = PC.RowsDigest(nw.n, nw.k)
    rd.update(b)
    return PC.WeightRecord(scheme, call.b, call.salt, call.weight, nw.colids,
                           PC.weight_inputs_digest(rd.digest(), call.salt, call.weight, nw.colids), yleaves,
                           PC.units_root(yleaves), nw.outputs, nw.reads())


def _commit(on: str, index: int, scheme: str, call: PC.Call, flags: int = HASH,
            side: PC.WeightRecord | None = None) -> tuple[PC.CallRecord, float | None]:
    sh = call.shape
    nw = PN.NativeWeight(call.b, scheme, call.salt, call.weight, call.colids)
    nat = PN.Native(call, scheme, side=nw)
    b, t = call.b.rows(0, call.b.n)
    leaves, yleaves = np.zeros(nat.population * 32, np.uint8), np.zeros(nw.population * 32, np.uint8)
    y = np.zeros((sh.m, sh.n), np.uint16)
    c = lambda v, dt: np.ascontiguousarray(v, dt)
    args = [sh.m, sh.k, sh.n, c(call.a, np.int8), c(call.s, np.uint32), c(b, np.int8), c(t, np.uint32), c(call.bias, np.uint16),
            np.frombuffer(call.salt, np.uint8).copy(), call.forward, call.position, call.weight, c(call.rowids, np.uint32),
            c(call.colids, np.uint32), c(nat.pi, np.uint32), c(nw.inv, np.uint32), leaves, yleaves, y]
    root, ms = np.zeros(32, np.uint8), ctypes.c_float(0)
    lib = _lib(on)
    if on == "fused":
        rc = lib.ncp2_fused_host(*args, root, ctypes.byref(ms), flags)
    else:
        commit = lib.ncp2_commit_fast if on == "twin" else lib.ncp2_commit
        rc = commit(*args, *((root, ctypes.byref(ms)) if on == "device" else ()))
    if rc:
        raise ValueError(f"{sh}: outside 16-row and 16-column strips" if rc == 1 else f"CUDA error {rc}")
    blob, ylv = leaves.tobytes(), yleaves.tobytes()
    root_b = root.tobytes() if on in ("device", "fused") else units_root_twin(blob) if on == "twin" else PC.units_root(blob)
    if side is None:
        side = side_of(scheme, call, ylv, b)
    elif ylv != side.leaves:
        raise ValueError(f"{sh}: the {on} build's Y unit leaves are not the weight side's")
    rec = PC.CallRecord(index, scheme, call, PC.inputs_digest(call, side.reads), blob, root_b, y, side, nat.outputs, nat.read)
    return rec, (ms.value if on in ("device", "fused") else None)


def commit_call(index: int, scheme: str, call: PC.Call, on: str = "host", flags: int = HASH,
                side: PC.WeightRecord | None = None) -> PC.CallRecord:
    """One call committed by the ``on`` build: "host" (`ncp2.cuh`), "twin" (`ncp2_fast.cuh`, the serving kernel's CPU twin),
    "device" (`ncp2_kernels.cu`) or "fused" (`ncp2_fused.cu`); the twin and device records carry their own units root. Its
    weight side is ``side``, whose Y unit leaves the build's must be, or made from the build's when None (`side_of`). A record
    opens units through `pouw_native`. ``flags`` are the fused build's (`ncp2_fused`)."""
    return _commit(on, index, scheme, call, flags, side)[0]


class FusedCommitter:
    """Linear calls of ``scheme`` committed on the device by `ncp2_fused` from torch tensors on the current stream: A (m × k
    int8, m a multiple of 16), the row scales' FP32 encodings (int32), the weight's int7 rows and scale encodings, the bias's
    BF16 encodings (int16), the row and column ids (int32); the served y comes back as BF16 encodings (int16, m × n).
    ``flags`` are `HASH`, `TIME_TILES` and `CACHE_Y`."""

    def __init__(self, scheme: str, salt: bytes, flags: int) -> None:
        self.lib, self.kind = fused(), PC.kind_of(scheme)
        self.salt, self.flags = np.frombuffer(salt, np.uint8).copy(), flags
        self._perms: dict[int, tuple] = {}

    def perm(self, k: int):
        import torch
        if k not in self._perms:
            pi = np.asarray(ncp2.perm(self.kind, k), np.int64)
            self._perms[k] = tuple(torch.tensor(v.astype(np.int32), device="cuda") for v in (pi, np.argsort(pi)))
        return self._perms[k]

    def __call__(self, a, s, b, t, bias, forward: int, position: int, weight: int, rowids, colids, root: np.ndarray | None = None,
                 yleaves: np.ndarray | None = None):
        import torch
        (m, k), n = a.shape, b.shape[0]
        y = torch.empty((m, n), dtype=torch.int16, device=a.device)
        pi, inv = self.perm(k)
        rc = self.lib.ncp2_fused(m, k, n, a.data_ptr(), s.data_ptr(), b.data_ptr(), t.data_ptr(), bias.data_ptr(), self.salt, forward,
                                 position, weight, rowids.data_ptr(), colids.data_ptr(), pi.data_ptr(), inv.data_ptr(), self.flags,
                                 torch.cuda.current_stream().cuda_stream, y.data_ptr(),
                                 None if root is None else root.ctypes.data, None,
                                 None if yleaves is None else yleaves.ctypes.data)
        if rc:
            raise RuntimeError(f"ncp2_fused {m}x{k}x{n}: " + ("outside the strips" if rc == 1 else f"CUDA error {rc}"))
        return y

    def tile_ms(self) -> tuple[float, int]:
        """The tile kernel's summed time over the calls since the last read (`TIME_TILES`), and their count."""
        total, calls = ctypes.c_double(0), ctypes.c_uint64(0)
        self.lib.ncp2_fused_tile_ms(ctypes.byref(total), ctypes.byref(calls))
        return total.value, calls.value


# ---- the gate ------------------------------------------------------------------------------------------------------------

def lowbyte_difference(ref: PC.CallRecord, got: PC.CallRecord) -> str | None:
    """`first_difference` for a record committed with `LOWBYTE`: its tile leaves against `lowbyte_leaf` of ``ref``'s tile
    words, every other leaf and the served y against ``ref``."""
    nat = PN.Native(ref.call, ref.scheme)
    lo, hi = nat.base["tile"], nat.base["dequant"]
    for u in range(lo, hi):
        if got.leaf(u) != lowbyte_leaf(ref.opening(u)):
            role, where = nat.locate(u)
            return f"unit {u} ({role} {where}) has the wrong low-byte leaf"
    patched = got.leaves[:32 * lo] + ref.leaves[32 * lo:32 * hi] + got.leaves[32 * hi:]
    return first_difference(ref, dataclasses.replace(got, leaves=patched, root=PC.units_root(patched)))


def first_difference(ref: PC.CallRecord, got: PC.CallRecord) -> str | None:
    """None when ``got`` commits what ``ref`` does: its weight side's Y unit leaves, inputs' digest and root, its inputs'
    digest, every unit's leaf, the root and the served y. Otherwise the first difference: the first wrong Y unit, the first
    wrong unit in the partition's numbering (its role and place, and for a dequantization unit its arguments and both
    values), or the first wrong served entry."""
    if got.shape != ref.shape:
        return f"shape {got.shape} is not {ref.shape}"
    if got.side.leaves != ref.side.leaves:
        a, b = (np.frombuffer(r.side.leaves, np.uint8).reshape(-1, 32) for r in (ref, got))
        return f"Y unit {int(np.flatnonzero((a != b).any(axis=1))[0])} of the weight side has the wrong leaf"
    if (got.side.inputs, got.side.root) != (ref.side.inputs, ref.side.root):
        return "the weight side's inputs' digest or root differs"
    if got.inputs != ref.inputs:
        return "the inputs' digest differs"
    if len(got.leaves) != len(ref.leaves):
        return f"{got.units} units, not {ref.units}"
    if got.leaves != ref.leaves:
        a, b = np.frombuffer(ref.leaves, np.uint8).reshape(-1, 32), np.frombuffer(got.leaves, np.uint8).reshape(-1, 32)
        u = int(np.flatnonzero((a != b).any(axis=1))[0])
        nat = PN.Native(ref.call, ref.scheme)
        role, where = nat.locate(u)
        msg = f"unit {u} ({role} {where}) has the wrong leaf"
        if role == "dequant":
            args = nat.args(u, nat.outputs)
            r, c = divmod(where[2], ncp2.D)
            row, col = ncp2.D * where[0] + r, ncp2.D * where[1] + c
            msg += (f": y[{row}, {col}] is {int(got.y[row, col]):#06x}, not {int(ref.y[row, col]):#06x}, from z, s, t, bias = "
                    + ", ".join(f"{int(v):#x}" for v in args))
        return msg
    if got.root != ref.root:
        return "the units root differs"
    if not np.array_equal(got.y, ref.y):
        row, col = (int(v) for v in np.argwhere(got.y != ref.y)[0])
        return f"served y[{row}, {col}] is {int(got.y[row, col]):#06x}, not {int(ref.y[row, col]):#06x}"
    return None


#: Special FP32 and BF16 encodings the dequantization must carry bit for bit: signed zeros, the smallest and largest
#: subnormals, one, the largest finite, infinities, and quiet and signalling NaNs with payloads.
F32_EDGES = (0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
             0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0x7FFFFFFF, 0x3F808000, 0x3F818000, 0x33800000, 0x00800000)
BF16_EDGES = (0x0000, 0x8000, 0x0001, 0x807F, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x4B00)


def _stream(tag: str, n: int) -> np.ndarray:
    """``n`` fixed bytes for the gate call named ``tag``: SHAKE256 over a label and the tag."""
    return np.frombuffer(hashlib.shake_256(b"verity/pouw-device/gate-call\0" + tag.encode()).digest(max(1, n)), np.uint8)[:n]


def _values(tag: str, m: int, k: int, n: int) -> dict[str, np.ndarray]:
    """A call's inputs from `_stream`: int7 A and B uniform on [−64, 63], positive row and column scales, a BF16 bias about
    0 and n distinct column ids."""
    sizes = {"a": m * k, "b": n * k, "s": 2 * m, "t": 2 * n, "bias": 2 * n, "salt": 32, "cols": 4 * 4 * n}
    raw, at, out = _stream(tag, sum(sizes.values())), 0, {}
    for key, size in sizes.items():
        out[key], at = raw[at:at + size], at + size
    scale = lambda u: (np.float32(1e-3) + u.view("<u2").astype(np.float32) * np.float32(0.01 / 65535)).view(np.uint32)
    bias = ((out["bias"].view("<u2").astype(np.float32) / np.float32(16384) - np.float32(2)).view(np.uint32) >> 16).astype(np.uint16)
    return {"a": ((out["a"] & 0x7F).astype(np.int16) - 64).astype(np.int8).reshape(m, k),
            "b": ((out["b"] & 0x7F).astype(np.int16) - 64).astype(np.int8).reshape(n, k),
            "s": scale(out["s"]), "t": scale(out["t"]), "bias": bias, "salt": out["salt"].tobytes(),
            "colids": np.argsort(out["cols"].view("<u4"), kind="stable")[:n].astype(np.uint32)}


def gate_call(tag: str, m: int, k: int, n: int, *, a=None, b=None, s=None, t=None, bias=None, forward: int = 3, layer: int = 2,
              site: int = 0, weight: int = 2) -> PC.Call:
    """One call of the gate: its inputs fixed by ``tag`` (`_values`), any of them given instead."""
    v = _values(tag, m, k, n)
    pick = lambda given, key, dt: v[key] if given is None else np.asarray(given, dt)
    return PC.Call(pick(a, "a", np.int8), pick(s, "s", np.uint32), PC.Quantized(pick(b, "b", np.int8), pick(t, "t", np.uint32)),
                   pick(bias, "bias", np.uint16), v["salt"], forward=forward, layer=layer, site=site, weight=weight,
                   rowids=np.arange(1000, 1000 + m, dtype=np.uint32), colids=v["colids"])


def gate_calls(shapes: Sequence[tuple[int, int]] = ()) -> list[tuple[str, PC.Call]]:
    """The calls a device committer must commit exactly:
    - a tree of five strips;
    - a column count off every block size;
    - the dequantization's special encodings on every row and column scale and bias;
    - Z past 2²⁴, where I32ToF32Rn rounds;
    - one call at 16 rows for each (k, n) of ``shapes`` (a model's linears, which the caller names)."""
    edges = np.array(F32_EDGES, np.uint32)
    out = [("five-strips", gate_call("five-strips", 80, 64, 48)),
           ("odd-columns", gate_call("odd-columns", 16, 64, 272, forward=7, layer=23, site=3, weight=95)),
           ("dequant-edges", gate_call("dequant-edges", 16 * 2, 64, 16 * 18, s=np.resize(edges, 32), t=np.repeat(edges, 16),
                                       bias=np.resize(np.array(BF16_EDGES, np.uint16), 16 * 18)))]
    k = 4864
    extreme = _stream("z-past-2^24", 2 * 16 * k).reshape(2, 16, k)
    a = np.where(extreme[0] < 253, -64, (extreme[0] & 0x7F).astype(np.int16) - 64).astype(np.int8)
    b = np.where(extreme[1] < 253, -64, (extreme[1] & 0x7F).astype(np.int16) - 64).astype(np.int8)
    out.append(("z-past-2^24", gate_call("z-past-2^24", 16, k, 16, a=a, b=b)))
    out += [(f"linear k={k} n={n}", gate_call(f"linear k={k} n={n}", 16, k, n, layer=11, site=i, weight=4 * 11 + i))
            for i, (k, n) in enumerate(shapes)]
    return out


__all__ = [
    "BF16_EDGES",
    "CACHE_Y",
    "CFLAGS",
    "CUDA_SOURCES",
    "F32_EDGES",
    "FUSED_SOURCES",
    "HASH",
    "HASHES",
    "LOWBYTE",
    "LOWBYTE_BLOCK",
    "LOWBYTE_DS",
    "NVCC_FLAGS",
    "SOURCES",
    "TIME_TILES",
    "TOOLCHAIN",
    "TS128_RATE",
    "FusedCommitter",
    "Toolchain",
    "commit_call",
    "compiler",
    "configure",
    "cuda_flags",
    "device",
    "first_difference",
    "fused",
    "gate_call",
    "gate_calls",
    "hash_many",
    "side_of",
    "host",
    "lowbyte_difference",
    "lowbyte_leaf",
    "nvcc",
    "source_digest",
    "tile_lowbyte_host",
    "units_root_twin",
]
