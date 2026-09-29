"""`ncp-v2`'s reference, computed natively: what `NcpLinear` must evaluate to, from hashlib and `schemes.ncp`.

The key's hashes are hashlib's SHA-512 and SHAKE256 over the byte layouts `ncp2`'s docstring states; the checked words are
`schemes.ncp.assemble` and `schemes.ncp.checked_route_u` on that noise (the MVP's own arithmetic, so the circuit is checked
against it, not against itself); y is the dequantization primitives' evaluators on Z.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from verity.ir.defs import REGISTRY

from ..schemes.ncp import assemble, checked_route_u
from . import ncp2
from .ncp2 import D, Shape
from .words import M32, i32_to_f32_rn, sx

M64 = (1 << 64) - 1


@dataclass(frozen=True)
class Call:
    """One linear call's inputs: A and B as int7 rows, the scales and bias as encodings, and the anchored fields."""

    a: tuple[tuple[int, ...], ...]          # m rows of k int7 values
    s: tuple[int, ...]                      # m FP32 encodings
    b: tuple[tuple[int, ...], ...]          # n weight rows of k int7 values
    t: tuple[int, ...]                      # n FP32 encodings
    bias: tuple[int, ...]                   # n BF16 encodings
    salt: bytes                             # 32 bytes
    forward: int
    layer: int
    site: int
    weight: int
    rowids: tuple[int, ...]                 # the global row index of each row (< 2^32)
    colids: tuple[int, ...]                 # the global column index of each weight row (< 2^32)

    @property
    def shape(self) -> Shape:
        return Shape(len(self.a), len(self.a[0]), len(self.b))

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    def args(self) -> list:
        """The flat input leaves of `NcpLinear`, in parameter order."""
        salt = [int.from_bytes(self.salt[4 * i:4 * i + 4], "little") for i in range(8)]
        lane = lambda v: [v & M32, (v >> 32) & M32]  # noqa: E731
        return [[x & 0x7F for row in self.a for x in row], list(self.s), [x & 0x7F for row in self.b for x in row], list(self.t),
                list(self.bias), salt, lane(self.forward), lane(self.position), lane(self.weight), list(self.rowids),
                list(self.colids)]


def _u64le(v: int) -> bytes:
    return (v & M64).to_bytes(8, "little")


def row_digest(row: Sequence[int]) -> bytes:
    return hashlib.sha512(ncp2.TAG_ROW + bytes(x & 0xFF for x in row)).digest()


def strip_digest(rows: Sequence[Sequence[int]]) -> bytes:
    return hashlib.sha512(ncp2.TAG_STRIP + b"".join(row_digest(r) for r in rows)).digest()


def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(ncp2.TAG_NODE + left + right).digest()


def root(a: Sequence[Sequence[int]]) -> bytes:
    """D_A: the RFC 6962 tree over the strips' digests."""
    digests = [strip_digest(a[D * i:D * (i + 1)]) for i in range(len(a) // D)]
    return ncp2.rfc6962(0, len(digests), lambda i: digests[i], node)


def _digest_lanes(dg: bytes) -> bytes:
    """A digest as Keccak lanes: each 8-byte word's big-endian value, written little-endian."""
    return b"".join(dg[i:i + 8][::-1] for i in range(0, len(dg), 8))


def key(call: Call, d_a: bytes) -> bytes:
    msg = ncp2.TAG_KEY + call.salt + _u64le(call.forward) + _u64le(call.position) + _u64le(call.weight) + _digest_lanes(d_a)
    return hashlib.shake_256(msg).digest(32)


def _index(i: int) -> bytes:
    return _u64le((i << 32) | i) * 5


def noise(xof: bytes) -> list[int]:
    return [(v & 63) - 31 for v in xof]


def e1_row(k_call: bytes, rowid: int, k: int) -> list[int]:
    return noise(hashlib.shake_256(_index(rowid) + ncp2.TAG_E1 + k_call).digest(k))


def f1_col(call: Call, colid: int, k: int) -> list[int]:
    return noise(hashlib.shake_256(_index(colid) + ncp2.TAG_F1 + call.salt + _u64le(call.weight)).digest(k))


@dataclass(frozen=True)
class Result:
    root: bytes
    key: bytes
    words: list[list[int]]                  # per tile (row strip, then column strip): the credited words, step-major
    z: list[list[int]]                      # m × n, as uint32 encodings
    y: list[list[int]]                      # m × n BF16 encodings


def dequant(z: int, s: int, t: int, bias: int) -> int:
    mul, add, cvt = REGISTRY["F32Mul_v2"], REGISTRY["Bf16Add_v1"], REGISTRY["F32ToBf16Rn_v1"]
    return add.evaluate(cvt.evaluate(mul.evaluate(mul.evaluate(s, t), i32_to_f32_rn(z))), bias)


def evaluate(call: Call, kind: str = "shift24") -> Result:
    """The reference outputs of one call."""
    sh = call.shape
    k, pi = sh.k, ncp2.perm(kind, sh.k)
    d_a = root(call.a)
    kc = key(call, d_a)
    e1 = [e1_row(kc, call.rowids[i], k) for i in range(sh.m)]
    f1 = [f1_col(call, call.colids[j], k) for j in range(sh.n)]
    ops = assemble([list(r) for r in call.a], [list(r) for r in call.b], e1, f1, pi, (1,) * k)
    checked = checked_route_u(ops)                         # [i][j][t], t = 1 … 3k/16
    words = []
    for rs in range(sh.m // D):
        for cs in range(sh.n // D):
            words.append([checked[D * rs + r][D * cs + c][t] for t in range(sh.steps - 1) for r in range(D) for c in range(D)])
    z = [[checked[i][j][-1] for j in range(sh.n)] for i in range(sh.m)]
    y = [[dequant(z[i][j], call.s[i], call.t[j], call.bias[j]) for j in range(sh.n)] for i in range(sh.m)]
    return Result(d_a, kc, words, z, y)


def product(call: Call) -> list[list[int]]:
    """A·Bᵀ over the integers, as int32 encodings (Z's value, `checkedU_final`)."""
    return [[sum(x * y for x, y in zip(ar, br)) & M32 for br in call.b] for ar in call.a]


def signed(v: int) -> int:
    return sx(v, 32)


__all__ = ["Call", "Result", "e1_row", "evaluate", "f1_col", "key", "node", "product", "root", "row_digest", "strip_digest"]
