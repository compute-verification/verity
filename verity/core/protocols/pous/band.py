"""The band scheme over the overwrite chain, the secure-first default: Lean's `chainBand` (POUS
`lean/submissions/band-chain/BandChainModel.lean`), segment by segment, as a `codec.Codec`.

`chainBand B d ℓ c dom` is `chainScheme (bandDAG B d) ℓ c dom`: one layer on the band DAG, where block `v`'s parents
are `v − d, …, v − 1` (every earlier block when `v < d`), with the dense scheme's overwrite chain `H` (`dense.DenseChain`):

~~~text
h_0   = 0^ℓ || dom(tag_s, v)                                         Lean `iv (dom v)`
for B in (C_{v-1}, ..., C_{max(0, v-d)}, pad):  (r, h) <- Π₂(B || h)   Lean `absorbedG`, `parentsDesc`: newest first
key_v = r of the pad call;  C_v = key_v xor W_v                         Lean `labelG`
~~~

Block `v` costs `min(v, d) + 1` calls, against dense's `v + 1`; with `d ≥ B − 1` the band is the complete DAG, dense.
Segment `s` is `chainBand B d ℓ 512 (dom tag_s)`.

**Deployment condition (the cliff, POUS `docs/band-vs-dense.md`).** The effective `D`, the sequential `Π₂` calls an
adversary completes within `Δ + RTT`, must not exceed `d`.  With `d ≥ D′` the band is exactly as hard as dense at
`D′` (`band_chain_hard`); at `D′ = d + 1` the gap attack (`band_gap_attack`, `band_cliff_d12`) frees every
`(d + 1)`-th block and the audit accepts with probability 1.  `audit.UnsignedRounds` refuses a deadline whose effective
`D` exceeds `d`.  The effective `D` counts whole `Π₂` calls, the ideal-permutation model's unit; at the reference
Feistel-SHAKE256 `Π₂` a prover answers without completing a call (`primitives.pi2_keccak_f`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import dense
from .codec import Codec
from .primitives import Primitives, pi2_call_ns


@dataclass(frozen=True, slots=True)
class BandParams:
    """`segments` segments of `blocks_per_segment` blocks (`B`) of `m` bits (`ℓ`), each block's parents the `d`
    blocks before it, rate 1."""

    blocks_per_segment: int
    m: int
    d: int
    segments: int = 1

    def __post_init__(self) -> None:
        for name in ("blocks_per_segment", "m", "d", "segments"):
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
        return codec_for(self.d)

    @property
    def challenges(self) -> int:
        """`k = 111` for 1%: `Pous.SecurityProofs.BandMultiMeetsFamily` at every `d ≥ 5`, with no hypothesis."""
        return 111

    @property
    def certified_rounds(self) -> int:
        """`D = 5` calls of `Π₂` per answer."""
        return 5

    @property
    def call_ns(self) -> int:
        """The fastest whole call, a 64 KiB `Π₂` in a long chain: 4,830 Keccak-f, at least 241.5 µs
        (`primitives.pi2_keccak_f`).  The ideal-permutation model's unit of time; at the reference Feistel-SHAKE256
        `Π₂` an adversary answers without completing a call, from 1.53% of `C` in 481 Keccak-f (the red team's
        resume attack)."""
        return pi2_call_ns(SECURE_FIRST.m)

    def parents(self, v: int) -> range:
        """`(bandDAG B d).par v`: `{u : u < v ∧ v ≤ u + d}`."""
        if not 0 <= v < self.blocks_per_segment:
            raise IndexError(f"block {v} is outside 0..{self.blocks_per_segment - 1}")
        return range(max(0, v - self.d), v)

    def with_segments(self, segments: int) -> BandParams:
        return BandParams(self.blocks_per_segment, self.m, self.d, segments)

    def manifest(self) -> dict[str, int | str]:
        return {"scheme": "band-chain", "blocks_per_segment": self.blocks_per_segment, "m": self.m, "d": self.d,
                "segments": self.segments}


SECURE_FIRST = BandParams(512, 524_288, 12)
"""The secure-first point (POUS `docs/band-vs-dense.md`): `B(512, 12)`, 65,536 B blocks (32 MiB segments), audited at
`D = 5` calls of at least 241.5 µs with `k = 111` (`Pous.SecurityProofs.BandChainMeets64D12`, with no hypothesis), so
in the ideal-permutation model `Δ + RTT < 1.449` ms.  The cliff is at an effective `D` of 13: 13 call-times, 3.1395 ms,
must exceed `Δ + RTT`.  Neither bounds an adversary at the reference Feistel-SHAKE256 `Π₂`, which rebuilds a label from
1.53% of `C` in 481 Keccak-f, about 24 µs (the red team's resume attack)."""


def _checked(d: int, fn):
    def call(params, prims, salt, s, data):
        if getattr(params, "d", None) != d:
            raise ValueError(f"the band codec for d = {d} needs parameters with d = {d}")
        return fn(params, prims, salt, s, data)

    return call


@lru_cache(maxsize=None)
def codec_for(d: int) -> Codec:
    """The band codec for in-degree `d`: the dense chain's encoder and decoder, with the band's parents."""
    if type(d) is not int or d < 1:
        raise ValueError("the in-degree is a positive integer")
    return Codec(f"band-chain/d{d}/v1", _checked(d, dense.encode_segment), _checked(d, dense.decode_segment))


BAND12 = codec_for(12)
"""The secure-first codec: the band with in-degree 12."""


def reference(params: BandParams) -> Primitives:
    """The dense chain's primitives: `Π₂` the Feistel-SHAKE256 of `docs/p3-instantiation.md` §1, no `P`."""
    return dense.reference(params)


def ideal(params: BandParams, seed: bytes) -> Primitives:
    """Lean's model: an ideal `Π₂`, sampled lazily."""
    return dense.ideal(params, seed)


def effective_rounds(params, deadline_ns: int, rtt_allowance_ns: int = 0) -> int:
    """The effective `D`: the `Π₂` calls of at least `params.call_ns` that fit in `Δ + RTT`."""
    return (deadline_ns + rtt_allowance_ns) // params.call_ns


def check_deployment(params, deadline_ns: int, rtt_allowance_ns: int = 0) -> None:
    """Raise unless the effective `D` stays at or below the in-degree `d` (the band's cliff)."""
    d = getattr(params, "d", None)
    if d is not None and effective_rounds(params, deadline_ns, rtt_allowance_ns) > d:
        raise ValueError(f"an effective D of {effective_rounds(params, deadline_ns, rtt_allowance_ns)} calls exceeds the "
                         f"band's in-degree d = {d}: past the cliff, the audit accepts a holding with every "
                         f"{d + 1}-th block dropped")
