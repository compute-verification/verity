"""`band-chain/d12/v1`, the secure-first scheme: one layer on the band DAG over the overwrite chain (Lean `chainBand B d
ℓ 512 (dom tag_s)` per segment), encoded and decoded by the reference codec `verity_pous.band`.

Its certificate at the frozen point (`band.SECURE_FIRST`: `B = 512`, 64 KiB blocks), for every `d ≥ 5`, with no
hypothesis (in the ideal-permutation model): `k = 111`, `D = 5` calls of at least 241.5 µs, `Q = 2^20`, with the cliff at
`d` (`band_cliff_d12`). It is `Pous.Pinned.BandMultiMeetsFamily`: `Meets` for the `N`-segment store
`chainBandSeg N (2^9) d` keyed by `segDom salt` (segment `s` by `tag_s = salt ‖ u64 s`, as the codec keys it), for
every `N` from 1 to `2^64`, which is every segment count `BandParams` allows. It is one `Meets` for the whole store:
challenges uniform over all its blocks, the state bound of the whole store, one permutation. At one segment it is also
`BandChainMeets64`, and at 448 (14 GB) `BandMultiMeets14`.

It needs the segment tags injective across segments (`segDom_inj`; `2^64` is where `u64 s` stops being injective); with
a shared domain, identical segments get identical labels and the audit accepts with probability 1
(`shared_domain_breaks`).
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import band as band_codec
from .. import dense as dense_codec
from ..primitives import Primitives, pi2_call_ns
from ..protocol import Certificate, Scheme

CHAIN_CLAIMS = ("ideal-permutation", "random-oracle", "feistel-indifferentiability", "sequential-step-floor",
                "perfect-isolation", "cr/sha-256", "live-verifier", "designated-verifier")
"""What the overwrite-chain certificates rest on: the ideal-permutation model for `Π₂`, instantiated by Feistel10 over
SHAKE256 (its round functions random oracles, the composition a named heuristic); the step floor of at least
241.5 µs per 64 KiB `Π₂` call in a chain (`primitives.pi2_keccak_f`); perfect isolation; SHA-256 collision resistance
for the commitment to `W` and for `vk`; and a live, designated verifier."""

POINT = (512, 524_288)
"""`B` and `ℓ` (bits) of the certified point: 32 MiB segments of 64 KiB blocks."""


FAMILY_SEGMENTS = 1 << 64
"""The largest segment count of `BandMultiMeetsFamily` (`N ≤ 2^64`), which is also the most `BandParams` allows."""


def chain_certificate(theorem: str, cliff: int | None, segments: int | None) -> Certificate:
    """The chain certificate at `POINT` (the band's `BandMultiMeetsFamily` and dense's `ChainDenseMeets64` share its
    terms)."""
    return Certificate(k=111, rounds=5, work=1 << 20, call_ns=pi2_call_ns(POINT[1]), theorem=theorem, cliff=cliff,
                       segments=segments, claims=CHAIN_CLAIMS)


class Band(Scheme):
    """The band scheme at in-degree `params.d`; registered at `d = 12`."""

    backend = staticmethod(band_codec.reference)

    def __init__(self, params: band_codec.BandParams = band_codec.SECURE_FIRST, prims: Primitives | None = None) -> None:
        if not isinstance(params, band_codec.BandParams):
            raise TypeError("the band scheme takes band.BandParams")
        super().__init__(params, prims)

    @property
    def name(self) -> str:
        return band_codec.codec_for(self.params.d).name

    @property
    def certificate(self) -> Certificate | None:
        p = self.params
        if (p.blocks_per_segment, p.m) != POINT or p.d < 5:
            return None
        return chain_certificate("Pous.Pinned.BandMultiMeetsFamily", p.d, p.segments)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense_codec.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense_codec.decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["CHAIN_CLAIMS", "FAMILY_SEGMENTS", "POINT", "Band", "chain_certificate"]
