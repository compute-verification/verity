"""The dense single-layer scheme over the overwrite chain: Lean's `chainDense` (POUS
`lean/submissions/sponge-dense/SpongeModel.lean` §9), segment by segment, as a `codec.Codec`.

One layer on the complete DAG, no `P`, no transpose.  Segment `s` is `chainDense B ℓ 512 (dom tag_s)`, with Lean's
256-bit salt taken to be `tag_s = salt ‖ u64(s)`:

~~~text
dom(tag_s, i) = tag_s (256 bits) ‖ i in binary (256 bits)                  Lean `dom`, 64 B, little-endian
h_0           = 0^ℓ ‖ dom(tag_s, i)                                          Lean `iv`: the domain is the capacity
key_i         = r of the last call of (r, h) <- Π₂(B ‖ h) over the blocks  Lean `ovKey`
                C_{i-1}, ..., C_0, pad                                       (`absorbed`: newest first, then `padBlk`)
C_i           = key_i xor W_i                                                Lean `labelOv`
~~~

`Π₂` is a permutation of `2ℓ + 512` bits (the default is `docs/p3-instantiation.md`'s Feistel-SHAKE256, domain 0x48),
`pad` is `0x01, zeros, 0x80`.  Block `i` costs `i + 1` calls of `Π₂`, so a segment of `B` blocks costs `B(B + 1)/2`.
Decoding recomputes every key from `C`, the rows in parallel (Lean `chainDense.dec`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .codec import Codec
from .labels import salt_value, tag
from .oracle import KeyQuery, PiInvQuery, PiQuery, Prog, Query, parallel, run
from .params import SALT_BITS
from .primitives import (CAPACITY_BITS, FeistelPi, IdealPermutation, OverwriteStepper, Permutation, Primitives,
                         key_program, pi2_call_ns)


@dataclass(frozen=True, slots=True)
class DenseParams:
    """`segments` segments of `blocks_per_segment` blocks (`B`) of `m` bits (`ℓ`), rate 1."""

    blocks_per_segment: int
    m: int
    segments: int = 1

    def __post_init__(self) -> None:
        for name in ("blocks_per_segment", "m", "segments"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.blocks_per_segment > 1 << 16:
            raise ValueError("block indices within a segment are 16 bits (the verifier's leaves)")
        if self.segments > 1 << 64:
            raise ValueError("the segment index must fit in 64 bits")

    @property
    def block_bytes(self) -> int:
        return (self.m + 7) // 8

    @property
    def blocks(self) -> int:
        return self.segments * self.blocks_per_segment

    @property
    def segment_bytes(self) -> int:
        if (self.blocks_per_segment * self.m) % 8:
            raise ValueError("a segment must be a whole number of bytes")
        return self.blocks_per_segment * self.m // 8

    @property
    def w_bits(self) -> int:
        return self.blocks * self.m

    @property
    def w_bytes(self) -> int:
        return (self.w_bits + 7) // 8

    @property
    def codec(self) -> Codec:
        return DENSE

    @property
    def challenges(self) -> int:
        """`k = 111` for 1%: `Pous.SecurityProofs.ChainDenseMeets64` over the overwrite chain, with no hypothesis; the
        random-oracle `dense_meets_64` needs only 106."""
        return 111

    @property
    def certified_rounds(self) -> int:
        """`D = 5` calls of `Π₂` per answer."""
        return 5

    @property
    def call_ns(self) -> int:
        """The fastest whole call, a 64 KiB `Π₂` in a long chain: 4,830 Keccak-f, at least 241.5 µs
        (`primitives.pi2_keccak_f`), so in the ideal-permutation model `D = 5` covers `Δ + RTT < 1.449` ms.  At the
        reference Feistel-SHAKE256 `Π₂` it does not: an adversary answers without completing a call, from 1.53% of
        `C` in 481 Keccak-f (red team, 4 Oct 2026, art:7617c8017e80)."""
        return pi2_call_ns(POINT_64K.m)

    def parents(self, i: int) -> range:
        """`(Dense.completeDAG B).par i`: every earlier block."""
        if not 0 <= i < self.blocks_per_segment:
            raise IndexError(f"block {i} is outside 0..{self.blocks_per_segment - 1}")
        return range(i)

    def with_segments(self, segments: int) -> DenseParams:
        return DenseParams(self.blocks_per_segment, self.m, segments)

    def manifest(self) -> dict[str, int | str]:
        return {"scheme": "dense-chain", "blocks_per_segment": self.blocks_per_segment, "m": self.m,
                "segments": self.segments}


POINT_64K = DenseParams(512, 524_288)
"""The dense point: 512 blocks of 65,536 B (`ℓ = 524288` bits) per segment (32 MiB), audited at `D = 5` calls of at
least 241.5 µs with `k = 111`: the point `Pous.SecurityProofs.ChainDenseMeets64` certifies for this `H` over an ideal
`Π₂`, with no hypothesis, and `dense_meets_64` for a random oracle (`lean/submissions/dense/DenseReCert.lean`,
`k = 106`).  At the reference Feistel-SHAKE256 `Π₂` neither bounds an adversary, which rebuilds a label from 1.53% of
`C` in 481 Keccak-f (`call_ns`).  The secure-first default is now the band (`band.SECURE_FIRST`)."""


def dom(t: int, i: int) -> int:
    """Lean `dom salt i`: the 256-bit salt (here `tag_s`), then `i` in binary, 512 bits."""
    if not 0 <= t < 1 << 256 or not 0 <= i < 1 << 256:
        raise ValueError("dom takes a 256-bit salt and a 256-bit index")
    return t | (i << 256)


@dataclass(frozen=True, slots=True)
class DenseChain:
    """`H` of `chainDense`: the overwrite chain over `pi2` (`2ℓ + 512` bits) from `h_0 = iv(dom(tag_s, i))`."""

    params: DenseParams
    pi2: Permutation
    name: str = "chain-dense-H/v1"

    def __post_init__(self) -> None:
        if self.pi2.width != 2 * self.params.m + CAPACITY_BITS:
            raise ValueError(f"the chain needs a permutation of 2ℓ + {CAPACITY_BITS} bits")

    def initial(self, t: int, i: int) -> int:
        """Lean `iv (dom t i) = 0^ℓ ‖ dom t i`."""
        return dom(t, i) << self.params.m

    def stepper(self, t: int, layer: int, node: int, parents: Sequence[int]) -> OverwriteStepper:
        if layer != 0:
            raise ValueError("the dense scheme has one layer")
        return OverwriteStepper(self.params.m, self.initial(t, node), sorted(parents, reverse=True))

    def steps(self, parents: Sequence[int]) -> list[tuple[int, ...]]:
        return [(u,) for u in sorted(parents, reverse=True)] + [()]

    def answer(self, q: Query) -> int:
        if isinstance(q, PiQuery):
            return self.pi2.forward(q.x)
        if isinstance(q, PiInvQuery):
            return self.pi2.inverse(q.y)
        raise TypeError(f"the chain answers Π₂ and Π₂⁻¹ queries, not {type(q).__name__}")


def reference(params: DenseParams) -> Primitives:
    """`Π₂` the Feistel-SHAKE256 of `docs/p3-instantiation.md` §1; the dense scheme has no `P`."""
    return Primitives(DenseChain(params, FeistelPi(2 * params.m + CAPACITY_BITS)), None)


def ideal(params: DenseParams, seed: bytes) -> Primitives:
    """Lean's model: an ideal `Π₂` (`idealPerm (ℓ + (ℓ + c))`), sampled lazily."""
    return Primitives(DenseChain(params, IdealPermutation(2 * params.m + CAPACITY_BITS, seed, "pi2")), None)


def _query(params, t: int, i: int, labels: Sequence[int]) -> KeyQuery:
    """Row `i`'s key query: its parents `params.parents(i)` (every earlier block here, a band for `band.BandParams`)."""
    return KeyQuery(t, 0, i, tuple((u, labels[u]) for u in params.parents(i)))


def encode_segment_prog(params: DenseParams, h: DenseChain, t: int, w: Sequence[int],
                        count: int | None = None) -> Prog[list[int]]:
    """Lean `labelOv`: blocks `0 .. count - 1` (default all) of one segment, each after every earlier one."""
    c: list[int] = []
    for i in range(params.blocks_per_segment if count is None else count):
        key = yield from key_program(h, _query(params, t, i, c))
        c.append(key ^ w[i])
    return c


def decode_segment_prog(params: DenseParams, h: DenseChain, t: int, c: Sequence[int]) -> Prog[list[int]]:
    """Lean `chainDense.dec`: `W_i = C_i xor key(IV(dom i); C_{i-1}, ..., C_0, pad)`, every row in parallel."""
    keys = yield from parallel(key_program(h, _query(params, t, i, c)) for i in range(params.blocks_per_segment))
    return [c[i] ^ keys[i] for i in range(params.blocks_per_segment)]


def _check(params: DenseParams, salt: bytes, s: int) -> int:
    if params.m % 8:
        raise ValueError("the byte API needs whole-byte blocks; use the *_prog functions for other widths")
    if type(salt) is not bytes or len(salt) != SALT_BITS // 8:
        raise ValueError(f"the salt is {SALT_BITS // 8} bytes, drawn by the verifier after W is committed")
    if not 0 <= s < params.segments:
        raise ValueError(f"segment {s} is outside 0..{params.segments - 1}")
    return tag(salt_value(salt), s)


def encode_segment(params: DenseParams, prims: Primitives, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
    """Segment `s` of `C` from its `segment_bytes` of `W`."""
    t, mb = _check(params, salt, s), params.m // 8
    if len(w_seg) != params.segment_bytes:
        raise ValueError(f"segment {s} of W is {params.segment_bytes} bytes")
    w = [int.from_bytes(w_seg[i * mb:(i + 1) * mb], "little") for i in range(params.blocks_per_segment)]
    c, _ = run(encode_segment_prog(params, prims.h, t, w), prims)
    return [x.to_bytes(mb, "little") for x in c]


def decode_segment(params: DenseParams, prims: Primitives, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
    """Segment `s` of `W` from its `blocks_per_segment` blocks."""
    t, mb = _check(params, salt, s), params.m // 8
    if len(blocks) != params.blocks_per_segment or any(len(b) != mb for b in blocks):
        raise ValueError(f"segment {s} needs its {params.blocks_per_segment} blocks of {mb} bytes")
    w, _ = run(decode_segment_prog(params, prims.h, t, [int.from_bytes(b, "little") for b in blocks]), prims)
    return b"".join(x.to_bytes(mb, "little") for x in w)


DENSE = Codec("dense-chain/v1", encode_segment, decode_segment)
"""The dense scheme as a codec (a selectable alternative to the band, `band.BAND12`)."""
