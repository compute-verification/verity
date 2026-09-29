"""H-1T's reference, computed natively: what `h1t.linear` must evaluate to.

The salt stream is hashlib's SHA-512 and SHAKE256 over `h1t`'s byte layouts.  The forming is Track H's pinned map
(`h1_tagged.h1t_ref`) through `h1t_words`' code-level port, which `tests/vectors/h1t_ref.json` holds to Track H's own
`h1t_ref` and `chain` bit for bit.  The chain is `verity.ml.tc`'s E4M3 k32 step from +0 and IEEE FP32 addition, and y
is the dequantization primitives' evaluators on Z.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from verity.ir.defs import REGISTRY
from verity.ml.tc.total_fp8 import HOPPER_E4M3_WGMMA_K32, tc_dot_total_e4m3

from . import h1t
from . import h1t_words as HW
from .h1t import D, REAL, SLICE, WORDS_PER_SLICE, Shape, slices, tag_tuple

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


@dataclass(frozen=True)
class Call:
    """One linear call's inputs: A's and B's rows as E4M3 codes, the scales and bias as encodings, and the anchored
    fields."""

    a: tuple[tuple[int, ...], ...]          # m rows of k codes (X_q)
    s: tuple[int, ...]                      # m FP32 encodings
    b: tuple[tuple[int, ...], ...]          # n weight rows of k codes (B′'s real rows)
    t: tuple[int, ...]                      # n FP32 encodings
    bias: tuple[int, ...] | None            # n BF16 encodings, or None for a linear with no bias
    salt: bytes                             # 32 bytes
    forward: int
    layer: int
    site: int
    weight: int
    stripids: tuple[int, ...]               # each strip's global strip index
    rowids: tuple[int, ...]                 # each row's global row index

    @property
    def shape(self) -> Shape:
        return Shape(len(self.a), len(self.a[0]), len(self.b))

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    @property
    def tags(self) -> list[tuple[int, int, int]]:
        return [tag_tuple(j) for j in range(len(self.b))]

    def args(self) -> list:
        """The flat input leaves of `H1tLinear`, in parameter order."""
        salt = [int.from_bytes(self.salt[4 * i:4 * i + 4], "little") for i in range(8)]
        lane = lambda v: [v & M32, (v >> 32) & M32]  # noqa: E731
        return [[x for row in self.a for x in row], list(self.s), [x for row in self.b for x in row], list(self.t),
                *([list(self.bias)] if self.bias is not None else []), salt, lane(self.forward), lane(self.position),
                lane(self.weight), list(self.stripids),
                list(self.rowids), [c for tg in self.tags for c in tg]]


def _u64le(v: int) -> bytes:
    return (v & M64).to_bytes(8, "little")


def row_digest(row: Sequence[int]) -> bytes:
    return hashlib.sha512(h1t.TAG_ROW + bytes(x & 0xFF for x in row)).digest()


def strip_digest(rows: Sequence[Sequence[int]]) -> bytes:
    return hashlib.sha512(h1t.TAG_STRIP + b"".join(row_digest(r) for r in rows)).digest()


def _digest_lanes(dg: bytes) -> bytes:
    """A digest as Keccak lanes: each 8-byte word's big-endian value, written little-endian."""
    return b"".join(dg[i:i + 8][::-1] for i in range(0, len(dg), 8))


def _index(i: int) -> bytes:
    return _u64le((i << 32) | i) * 5


def strip_key(call: Call, s: int) -> bytes:
    rows = call.a[D * s:D * (s + 1)]
    msg = (_index(call.stripids[s]) + h1t.TAG_KEY + call.salt + _u64le(call.forward) + _u64le(call.position)
           + _u64le(call.weight) + _digest_lanes(strip_digest(rows)))
    return hashlib.shake_256(msg).digest(32)


def salt_words(key: bytes, rowid: int, k: int) -> list[list[int]]:
    """A row's salt words: per slice, r0 … r17."""
    T = slices(k)
    raw = hashlib.shake_256(_index(rowid) + h1t.TAG_SALT + key).digest(4 * WORDS_PER_SLICE * T)
    w = [int.from_bytes(raw[4 * j:4 * j + 4], "little") for j in range(WORDS_PER_SLICE * T)]
    return [w[WORDS_PER_SLICE * t:WORDS_PER_SLICE * (t + 1)] for t in range(T)]


def form_row(codes: Sequence[int], words: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    """Track H's map on one row: (x1, x2), 32T codes each."""
    T = len(words)
    x = list(codes) + [0] * (REAL * T - len(codes))
    x1, x2 = [], []
    for t in range(T):
        sl, r = x[REAL * t:REAL * (t + 1)], words[t]
        ef = HW.slice_exp(sl)
        for lane in range(REAL):
            s = 1 if (r[lane // 2] >> (31 if lane % 2 else 15)) & 1 else -1
            m_ = HW.mu(sl[lane], ef)
            x1.append(HW.enc(HW.val(sl[lane]) + s * m_))
            x2.append(HW.enc(-s * m_))
        for word, sel in ((r[17], 1), (r[16], 0), (r[16], 1)):
            v = HW.PouwH1tTag.evaluate(ef + HW.E_BIAS, word, sel)
            x1.append(v)
            x2.append(v ^ 0x80)
    return x1, x2


def weight_slices(row: Sequence[int], tags: Sequence[int], T: int) -> list[list[int]]:
    """B′'s column as T slices: 29 real codes (0 past k) and the tag tuple."""
    x = list(row) + [0] * (REAL * T - len(row))
    return [x[REAL * t:REAL * (t + 1)] + list(tags) for t in range(T)]


def cell_words(x1: Sequence[int], x2: Sequence[int], bs: Sequence[Sequence[int]]) -> tuple[list[int], int]:
    """(the credited words I_1 … I_2T, R_2 … R_(2T−2); Z = R_2T)."""
    T = len(bs)
    add = REGISTRY["F32Add_v1"].evaluate
    atoms = [tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, list(xb[SLICE * t:SLICE * (t + 1)]), bs[t])
             for xb in (x1, x2) for t in range(T)]
    run = [atoms[0]]
    for a in atoms[1:]:
        run.append(add(run[-1], a))
    return atoms + run[1:2 * T - 2], run[-1]


def dequant(z: int, s: int, t: int, bias: int | None) -> int:
    """vLLM's epilogue order (`h1t.H1tDequant`): m = RN(s·Z), then bf16(RN(t·m + f32(bias))) or bf16(RN(t·m))."""
    mul, fma = REGISTRY["F32Mul_v2"].evaluate, REGISTRY["F32Fma_v2"].evaluate
    widen, cast = REGISTRY["Bf16ToF32_v1"].evaluate, REGISTRY["F2fpBf16_v1"].evaluate
    m = mul(s, z)
    return cast(mul(t, m) if bias is None else fma(t, m, widen(bias)))


@dataclass(frozen=True)
class Result:
    keys: list[bytes]
    x1: list[list[int]]
    x2: list[list[int]]
    words: list[list[int]]                  # per tile (row strip, then column strip), per cell (row-major): 4T − 3
    z: list[list[int]]                      # m × n, FP32 encodings
    out: list[list[int]]                    # y, BF16 encodings


def evaluate(call: Call) -> Result:
    sh = call.shape
    T = slices(sh.k)
    keys = [strip_key(call, s) for s in range(sh.m // D)]
    x1, x2 = [], []
    for i, row in enumerate(call.a):
        a1, a2 = form_row(row, salt_words(keys[i // D], call.rowids[i], sh.k))
        x1.append(a1)
        x2.append(a2)
    bp = [weight_slices(call.b[j], tag_tuple(j), T) for j in range(sh.n)]
    z = [[0] * sh.n for _ in range(sh.m)]
    words = []
    for rs in range(sh.m // D):
        for cs in range(sh.n // D):
            tw = []
            for i in range(D * rs, D * rs + D):
                for j in range(D * cs, D * cs + D):
                    w, z[i][j] = cell_words(x1[i], x2[i], bp[j])
                    tw += w
            words.append(tw)
    out = [[dequant(z[i][j], call.s[i], call.t[j], None if call.bias is None else call.bias[j]) for j in range(sh.n)]
           for i in range(sh.m)]
    return Result(keys, x1, x2, words, z, out)


__all__ = ["Call", "Result", "cell_words", "dequant", "evaluate", "form_row", "row_digest", "salt_words", "strip_digest",
           "strip_key", "weight_slices"]
