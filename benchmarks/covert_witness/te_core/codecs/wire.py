"""The encoding's bytes: a 4-byte header length, a compact JSON header, then each array's raw bytes in order. Every byte
counts in a codec's size, the header included."""
from __future__ import annotations

import json
import struct

import torch

DTYPES = {str(d).removeprefix("torch."): d for d in (
    torch.float64, torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.int32, torch.int16, torch.int8,
    torch.uint8, torch.bool)}


def dtype_name(d: torch.dtype) -> str:
    return str(d).removeprefix("torch.")


def raw(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def from_raw(b: bytes, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if not b:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).view(dtype).reshape(shape).clone()


def pack(header: dict, arrays: list[torch.Tensor] = ()) -> bytes:
    """`header` plus `arrays`; each array's dtype and shape go in the header."""
    header = header | {"a": [[dtype_name(a.dtype), list(a.shape)] for a in arrays]}
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    return struct.pack("<I", len(h)) + h + b"".join(raw(a) for a in arrays)


def unpack(b: bytes) -> tuple[dict, list[torch.Tensor]]:
    (n,) = struct.unpack_from("<I", b)
    header = json.loads(b[4:4 + n])
    off, arrays = 4 + n, []
    for name, shape in header["a"]:
        dt = DTYPES[name]
        size = torch.empty((), dtype=dt).element_size()
        count = 1
        for s in shape:
            count *= s
        arrays.append(from_raw(b[off:off + size * count], dt, tuple(shape)))
        off += size * count
    if off != len(b):
        raise ValueError(f"encoding has {len(b) - off} trailing bytes")
    return header, arrays


def pack_bits(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Non-negative integers below 2**bits, `bits` each, little-endian within a byte stream (uint8)."""
    v = idx.reshape(-1).to(torch.int64)
    shifts = torch.arange(bits, dtype=torch.int64)
    flat = ((v[:, None] >> shifts) & 1).reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)]).reshape(-1, 8)
    return (flat.to(torch.int64) << torch.arange(8, dtype=torch.int64)).sum(1).to(torch.uint8)


def unpack_bits(b: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    flat = ((b.to(torch.int64)[:, None] >> torch.arange(8, dtype=torch.int64)) & 1).reshape(-1)[: count * bits]
    return (flat.reshape(count, bits) << torch.arange(bits, dtype=torch.int64)).sum(1)
