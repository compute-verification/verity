"""`ncp-v2` on the device: the unit functions under `pouw_cuda/` (`ncp2.cuh`), and the gate that holds any build of them to
`pouw_native`, leaf for leaf.

`ncp2.cuh` computes each unit's values as `pouw_native` does (the noise from SHAKE256, X and Y, route U's checkpoints, the
exact dequantization and each unit's SHA-256 leaf), in functions that compile for the host and the device. `host()` builds
them for the CPU with the system C++ compiler, keyed by what it compiles, and `commit_call` commits a call through that
build; `device()` builds `ncp2_kernels.cu` with nvcc, whose committer calls the same functions and is driven the same way.
`first_difference` is the gate: the first unit, in the partition's numbering, whose leaf a record gets wrong, named by role
and place, or the first wrong served entry. A device committer passes when it names nothing on `gate_calls`.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
from verity_pouw.circuit import ncp2

from verity_vllm.protocol_options import pouw_circuit as PC
from verity_vllm.protocol_options import pouw_native as PN

HERE = Path(__file__).with_name("pouw_cuda")
SOURCES = ("ncp2.cuh", "ncp2_host.cpp")
CFLAGS = ("-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-fno-exceptions", "-fno-rtti", "-fPIC", "-shared")
CUDA_SOURCES = ("ncp2.cuh", "ncp2_kernels.cu")
#: no --use_fast_math: the dequantization's steps are IEEE binary32, with subnormals, and nothing is contracted
NVCC_FLAGS = ("-O3", "-std=c++17", "-fmad=false", "-Xcompiler", "-fPIC", "-shared")
HASHES = {"sha256": (0, 32), "sha512": (1, 64), "shake256": (2, None), "turboshake128": (3, None)}

_u8, _u16, _u32, _u64 = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.uint8, np.uint16, np.uint32, np.uint64))
_i8 = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
_COMMIT = [ctypes.c_int] * 3 + [_i8, _u32, _i8, _u32, _u16, _u8] + [ctypes.c_uint64] * 3 + [_u32] * 4 + [_u8, _u16]


def compiler() -> str | None:
    """``$CXX``, or the first of g++, c++ and clang++ on the path."""
    return os.environ.get("CXX") or next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)


def nvcc() -> str | None:
    """``$NVCC``, nvcc on the path, or CUDA's default install."""
    return os.environ.get("NVCC") or shutil.which("nvcc") or next((p for p in ("/usr/local/cuda/bin/nvcc",) if os.path.exists(p)), None)


def _need(tool: str | None, what: str) -> str:
    if tool is None:
        raise RuntimeError(f"no {what}")
    return tool


def cuda_flags() -> tuple[str, ...]:
    arch = os.environ.get("VERITY_POUW_CUDA_ARCH", "89")
    return (*NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}")


def source_digest(sources: Sequence[str] = SOURCES, flags: Sequence[str] = (), tool: str | None = None) -> str:
    """SHA-256 over the sources (name and bytes, in order), the flags (default `CFLAGS`) and the compiler's version."""
    h = hashlib.sha256()
    for name in sources:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    version = subprocess.run([tool or _need(compiler(), "C++ compiler: set CXX"), "--version"], capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()
    h.update(" ".join(flags or CFLAGS).encode() + b"\0" + "|".join(version).encode())
    return h.hexdigest()


def _build(tool: str, sources: Sequence[str], flags: Sequence[str], name: str) -> ctypes.CDLL:
    """``sources[-1]`` built in ``$VERITY_POUW_CUDA_BUILD`` (default ``~/.cache/verity/pouw_cuda``) under the first 12 hex
    digits of `source_digest`, so a changed source never loads another's library."""
    root = Path(os.environ.get("VERITY_POUW_CUDA_BUILD", Path.home() / ".cache" / "verity" / "pouw_cuda"))
    lib = root / source_digest(sources, flags, tool)[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([tool, *flags, "-o", str(tmp), str(HERE / sources[-1])], check=True)
        tmp.replace(lib)
    dll = ctypes.CDLL(str(lib))
    dll.ncp2_hash.argtypes = [ctypes.c_int, ctypes.c_int, _u8, _u64, ctypes.c_uint64, ctypes.c_uint64, _u8]
    dll.ncp2_hash.restype = dll.ncp2_commit.restype = ctypes.c_int
    return dll


@cache
def host() -> ctypes.CDLL:
    """`ncp2_host.cpp` built for this CPU with the system C++ compiler."""
    dll = _build(_need(compiler(), "C++ compiler: set CXX"), SOURCES, CFLAGS, "libncp2_host.so")
    dll.ncp2_commit.argtypes = _COMMIT
    return dll


@cache
def device() -> ctypes.CDLL:
    """`ncp2_kernels.cu` built with nvcc for ``$VERITY_POUW_CUDA_ARCH`` (default 89, the RTX 4090)."""
    dll = _build(_need(nvcc(), "nvcc: set NVCC"), CUDA_SOURCES, cuda_flags(), "libncp2_cuda.so")
    fp, dp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    dll.ncp2_commit.argtypes = _COMMIT + [_u8, fp]
    dll.ncp2_ceiling.argtypes = [ctypes.c_int] * 4 + [dp]
    dll.ncp2_leaf_time.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, fp]
    dll.ncp2_f1_time.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_int, fp]
    for f in (dll.ncp2_ceiling, dll.ncp2_leaf_time, dll.ncp2_f1_time):
        f.restype = ctypes.c_int
    return dll


def _lib(on: str) -> ctypes.CDLL:
    return {"host": host, "device": device}[on]()


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


def _commit(on: str, index: int, scheme: str, call: PC.Call) -> tuple[PC.CallRecord, float | None]:
    sh, nat = call.shape, PN.Native(call)
    b, t = call.b.rows(0, call.b.n)
    leaves = np.zeros(nat.population * 32, np.uint8)
    y = np.zeros((sh.m, sh.n), np.uint16)
    c = lambda v, dt: np.ascontiguousarray(v, dt)
    args = [sh.m, sh.k, sh.n, c(call.a, np.int8), c(call.s, np.uint32), c(b, np.int8), c(t, np.uint32), c(call.bias, np.uint16),
            np.frombuffer(call.salt, np.uint8).copy(), call.forward, call.position, call.weight, c(call.rowids, np.uint32),
            c(call.colids, np.uint32), c(nat.pi, np.uint32), c(nat.inv, np.uint32), leaves, y]
    root, ms = np.zeros(32, np.uint8), ctypes.c_float(0)
    rc = _lib(on).ncp2_commit(*args, *((root, ctypes.byref(ms)) if on == "device" else ()))
    if rc:
        raise ValueError(f"{sh}: outside 16-row and 16-column strips" if rc == 1 else f"CUDA error {rc}")
    blob = leaves.tobytes()
    rec = PC.CallRecord(index, scheme, call, PC.inputs_digest(call, PC.weight_digest(call.b)), blob,
                        root.tobytes() if on == "device" else PC.units_root(blob), y, nat.outputs)
    return rec, (ms.value if on == "device" else None)


def commit_call(index: int, scheme: str, call: PC.Call, on: str = "host") -> PC.CallRecord:
    """One call committed by the ``on`` build of `ncp2.cuh`; the device's record carries the device's units root. Its record
    opens units through `pouw_native`."""
    return _commit(on, index, scheme, call)[0]


def commit_call_timed(index: int, scheme: str, call: PC.Call) -> tuple[PC.CallRecord, float]:
    """`commit_call` on the device, and its device time in ms from the first kernel to the root, the inputs resident."""
    rec, ms = _commit("device", index, scheme, call)
    return rec, float(ms)


# ---- the gate ------------------------------------------------------------------------------------------------------------

def first_difference(ref: PC.CallRecord, got: PC.CallRecord) -> str | None:
    """None when ``got`` commits what ``ref`` does: its inputs' digest, every unit's leaf, the root and the served y. Otherwise
    the first difference: the first wrong unit in the partition's numbering (its role and place, and for a dequantization
    unit its arguments and both values), or the first wrong served entry."""
    if got.shape != ref.shape:
        return f"shape {got.shape} is not {ref.shape}"
    if got.inputs != ref.inputs:
        return "the inputs' digest differs"
    if len(got.leaves) != len(ref.leaves):
        return f"{got.units} units, not {ref.units}"
    if got.leaves != ref.leaves:
        a, b = np.frombuffer(ref.leaves, np.uint8).reshape(-1, 32), np.frombuffer(got.leaves, np.uint8).reshape(-1, 32)
        u = int(np.flatnonzero((a != b).any(axis=1))[0])
        nat = PN.Native(ref.call)
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


def _call(rng: np.random.Generator, m: int, k: int, n: int, *, a=None, b=None, s=None, t=None, bias=None, forward: int = 3,
          layer: int = 2, site: int = 0, weight: int = 2) -> PC.Call:
    a = rng.integers(-64, 64, (m, k)).astype(np.int8) if a is None else a
    b = rng.integers(-64, 64, (n, k)).astype(np.int8) if b is None else b
    scale = lambda size: (np.abs(rng.standard_normal(size)).astype(np.float32) * 0.01 + 1e-3).view(np.uint32)
    s = scale(m) if s is None else np.asarray(s, np.uint32)
    t = scale(n) if t is None else np.asarray(t, np.uint32)
    bias = (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16) if bias is None else bias
    return PC.Call(a, s, PC.Quantized(b, t), np.asarray(bias, np.uint16), rng.bytes(32), forward=forward, layer=layer, site=site,
                   weight=weight, rowids=np.arange(1000, 1000 + m, dtype=np.uint32),
                   colids=rng.permutation(4 * n)[:n].astype(np.uint32))


#: Qwen2.5-0.5B's four linears, (k, n): qkv (with bias), o, gate_up, down.
QWEN05 = ((896, 1152), (896, 896), (896, 9728), (4864, 896))


def gate_calls(seed: int = 0, qwen: bool = False) -> list[tuple[str, PC.Call]]:
    """The calls a device committer must commit exactly: a tree of five strips, a column count off every block size, the
    dequantization's special encodings on every row and column scale and bias, Z past 2²⁴ (I32ToF32Rn rounds), and with
    ``qwen`` each of Qwen2.5-0.5B's linears at 16 rows."""
    rng = np.random.default_rng(seed)
    edges = np.array(F32_EDGES, np.uint32)
    out = [("five-strips", _call(rng, 80, 64, 48)),
           ("odd-columns", _call(rng, 16, 64, 272, forward=7, layer=23, site=3, weight=95)),
           ("dequant-edges", _call(rng, 16 * 2, 64, 16 * 18, s=np.resize(edges, 32), t=np.repeat(edges, 16),
                                   bias=np.resize(np.array(BF16_EDGES, np.uint16), 16 * 18)))]
    k = 4864
    a = np.where(rng.random((16, k)) < 0.99, -64, rng.integers(-64, 64, (16, k))).astype(np.int8)
    b = np.where(rng.random((16, k)) < 0.99, -64, rng.integers(-64, 64, (16, k))).astype(np.int8)
    out.append(("z-past-2^24", _call(rng, 16, k, 16, a=a, b=b)))
    if qwen:
        out += [(f"qwen05 k={k} n={n}", _call(rng, 16, k, n, layer=11, site=i, weight=4 * 11 + i)) for i, (k, n) in enumerate(QWEN05)]
    return out


__all__ = ["BF16_EDGES", "CUDA_SOURCES", "F32_EDGES", "HASHES", "QWEN05", "SOURCES", "commit_call", "commit_call_timed", "compiler", "cuda_flags",
           "device", "first_difference", "gate_calls", "hash_many", "host", "nvcc", "source_digest"]
