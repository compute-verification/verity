"""The parameters of P3: `segments` segments, each two layers of `n` labels over the band graph `B(n, k)`.

A label is `n` components of `wc` bits (Lean `Lab n wc`), so its width is `m = n * wc`; component `v` of every lower
label feeds top label `v` (the transpose).  `W` is `segments * n` lower inputs, and the codeword `C` is the
`segments * n` top labels: `B = segments * n` labels of `m` bits, rate 1.
"""

from __future__ import annotations

from dataclasses import dataclass

SALT_BITS = 192
"""The salt, sampled by the verifier after `W` is committed, and published as `pp` (`docs/p3-instantiation.md` §1: 24 B;
Lean's `p3Scheme` has 128 bits)."""

TAG_BITS = 256
"""The segment tag `tag_s = salt || u64(s)`: the salt in bits 0..191, the segment index in bits 192..255 (Lean's
`tagOf salt` for segment 0)."""


@dataclass(frozen=True, slots=True)
class Params:
    n: int
    wc: int
    k: int
    segments: int = 1

    def __post_init__(self) -> None:
        for name in ("n", "wc", "segments"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.k) is not int or self.k < 0:
            raise ValueError("k must be a nonnegative integer")
        if self.segments > 1 << (TAG_BITS - SALT_BITS):
            raise ValueError("the segment index must fit in 64 bits")
        if self.n > 1 << 16:
            raise ValueError("node indices are 16 bits")

    @property
    def m(self) -> int:
        """The label width in bits."""
        return self.n * self.wc

    @property
    def blocks_per_segment(self) -> int:
        """`n`: one label per top label (`codec.Geometry`)."""
        return self.n

    @property
    def blocks(self) -> int:
        """`B`: the number of codeword labels, one top label each."""
        return self.segments * self.n

    @property
    def block_bytes(self) -> int:
        return (self.m + 7) // 8

    @property
    def w_bits(self) -> int:
        return self.segments * self.n * self.m

    @property
    def w_bytes(self) -> int:
        return (self.w_bits + 7) // 8

    def parents(self, v: int) -> range:
        """`(bandGraph n k).par v`: the spine parent `v - 1` and the extras `{u : u + 1 < v ≤ u + k}`, so the
        `max(k, 1)` nodes before `v` (Lean keeps the spine at `k = 0`)."""
        if not 0 <= v < self.n:
            raise IndexError(f"node {v} is outside 0..{self.n - 1}")
        return range(max(0, v - max(self.k, 1)), v)

    def children(self, v: int) -> range:
        if not 0 <= v < self.n:
            raise IndexError(f"node {v} is outside 0..{self.n - 1}")
        return range(v + 1, min(self.n, v + max(self.k, 1) + 1))

    @property
    def segment_bytes(self) -> int:
        """One segment of `W` (and of `C`): `n` labels."""
        if (self.n * self.m) % 8:
            raise ValueError("a segment must be a whole number of bytes")
        return self.n * self.m // 8

    def with_segments(self, segments: int) -> Params:
        return Params(self.n, self.wc, self.k, segments)

    @property
    def codec(self):
        """P3's two-layer codec (`scheme.P3`)."""
        from .scheme import P3

        return P3

    def manifest(self) -> dict[str, int]:
        return {"n": self.n, "wc": self.wc, "k": self.k, "segments": self.segments}


OPERATING_POINT = Params(n=220, wc=2384, k=12)
"""P3's operating point (`docs/p3-scheme.md` §2c, `docs/p3-instantiation.md` §1): `B(220, 12)`, labels of 220
components of 298 B, `m = 65560` B, so a segment is 14.4 MB.  P3 is the research codec; the secure-first default is
the band (`band.SECURE_FIRST`)."""

LEAN_P3_MEETS = Params(n=220, wc=37, k=12)
"""Lean's `p3Scheme`: one segment of `B(220, 12)`, labels of `220 * 37 = 8140` bits."""
