"""The CPU reference the decode-to-staging kernel (`p2stage.cu`) is checked against, with no torch.

A tensor of W at W-offset `base`, `length` bytes long, lies in blocks `blocks_of(base, length)`; block j carries W bytes
[2054 j, 2054 j + 2054) as bytes 2 .. 2055 of its big-endian 2,056-byte row (`verity/core/protocols/pous/schemes/p2.py`,
`p2-16448/v4`). A block's decode is `p2_v1.reference.decode` (the P2 spec's map, pinned by its vectors) with the
registered scheme's keys (`schemes.p2.sha512_keys` under `schemes.p2.expand_prefix`), the pairing
`benchmarks/pous/bench.py`'s `p2-v3-gpu` arm gates its kernels by. `staged` is what the kernel writes: every block's low
2,054 bytes, codeword or not, and `refusals` what the registered reference refuses (`decode_segment`: c >= p, or a decode
wider than the payload), which the kernel flags.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from verity.core.protocols.pous.schemes import p2 as S

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pous"))
from p2_v1 import reference as R

PARAMS = S.P2Params(*S.POINT)
SPEC = R.Params(PARAMS.w, PARAMS.c0, PARAMS.payload_bytes, PARAMS.blocks_per_segment)
ROW = PARAMS.block_bytes          # 2,056
PAYLOAD = PARAMS.payload_bytes    # 2,054
FLAG_T = 1                        # t's first candidate >= p
FLAG_WIDE = 4                     # a decode wider than the payload
FLAG_C = 8                        # c >= p


def prefix(salt: bytes) -> bytes:
    """The registered scheme's key prefix for `salt` (24 bytes): whole 128-byte blocks, one midstate."""
    return S.expand_prefix(PARAMS, salt)


def blocks_of(base: int, length: int) -> range:
    """The blocks a tensor of `length` bytes at W-offset `base` overlaps."""
    if base < 0 or length < 0:
        raise ValueError("a tensor has a nonnegative offset and length")
    return range(base // PAYLOAD, -(-(base + length) // PAYLOAD)) if length else range(0)


def keys(pfx: bytes, j: int) -> tuple[int, int]:
    """Block j's (K, t), j a global block index."""
    s, b = divmod(j, PARAMS.blocks_per_segment)
    return S.sha512_keys(PARAMS, pfx, s, b)


def decode_row(pfx: bytes, j: int, c: bytes) -> tuple[bytes, int]:
    """Block j's decoded 2,056-byte row and its flags (`FLAG_*`) for any c below 2^w."""
    if len(c) != ROW:
        raise ValueError(f"a block is {ROW} bytes")
    v = int.from_bytes(c, "big")
    if v >= SPEC.p:
        return bytes(ROW), FLAG_C
    k, t = keys(pfx, j)
    m = R.decode(SPEC, v, k, t).to_bytes(ROW, "big")
    return m, (FLAG_WIDE if m[0] or m[1] else 0)


def staged(pfx: bytes, base: int, length: int, rows: Mapping[int, bytes]) -> dict[int, bytes]:
    """For each block j in `rows` (its c row), the bytes it writes into the tensor's staging buffer, keyed by their first
    index there: its decode's low 2,054 bytes clipped to [0, length). A c >= p writes nothing (the kernel's bytes there are
    unspecified and flagged)."""
    out = {}
    for j, c in rows.items():
        m, f = decode_row(pfx, j, c)
        if f & FLAG_C:
            continue
        at = PAYLOAD * j - base
        lo, hi = max(0, -at), min(PAYLOAD, length - at)
        if lo < hi:
            out[at + lo] = m[2 + lo:2 + hi]
    return out


def refusals(pfx: bytes, rows: Mapping[int, bytes]) -> int:
    """The flags of these blocks OR'd: nonzero exactly when the registered reference refuses one of them."""
    f = 0
    for j, c in rows.items():
        f |= decode_row(pfx, j, c)[1]
    return f


def sample(blocks: range, count: int) -> list[int]:
    """`count` blocks of `blocks` spread evenly, with both ends and their neighbours (where a tensor is clipped)."""
    if not blocks:
        return []
    picks = {blocks[0], blocks[-1], blocks[min(1, len(blocks) - 1)], blocks[max(0, len(blocks) - 2)]}
    step = max(1, len(blocks) // max(1, count))
    picks.update(blocks[i] for i in range(0, len(blocks), step))
    return sorted(picks)


def check_rows(pfx: bytes, base: int, length: int, rows: Mapping[int, bytes], got: bytes | bytearray | memoryview) -> list[int]:
    """The blocks of `rows` whose staged bytes differ from `got` (the kernel's staging buffer, or the part of it at
    `base`'s tensor)."""
    bad = []
    for j in sorted(rows):
        for at, want in staged(pfx, base, length, {j: rows[j]}).items():
            if bytes(got[at:at + len(want)]) != want:
                bad.append(j)
    return bad


def layer_tensors(d: int, kv: int, mlp: int, layers: int = 1, head: int = 0) -> list[tuple[str, int, int]]:
    """(name, N, K) of a decoder's linears as W holds them, per layer QKV (d + 2 kv rows), O, up and down, then the head."""
    out = []
    for layer in range(layers):
        out += [(f"l{layer}.qkv", d + 2 * kv, d), (f"l{layer}.o", d, d), (f"l{layer}.up", mlp, d), (f"l{layer}.down", d, mlp)]
    if head:
        out.append(("head", head, d))
    return out


def packed(tensors: Iterable[tuple[str, int, int]], start: int = 0) -> list[tuple[str, int, int, int, int]]:
    """(name, N, K, base, length) with the tensors back to back from W-offset `start` (`Layout.of`, align=False)."""
    out, at = [], start
    for name, n, k in tensors:
        out.append((name, n, k, at, n * k))
        at += n * k
    return out
