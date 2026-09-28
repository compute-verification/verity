"""`band-chain/d12/v1`, the secure-first scheme: one layer on the band DAG over the overwrite chain (Lean `chainBand B d
ℓ 512 (dom tag_s)` per segment), encoded and decoded by the reference codec `verity_pous.band`.

Its certificate at the frozen point (`band.SECURE_FIRST`: `B = 512`, 64 KiB blocks) is `band_meets_64`, for one segment
and every `d ≥ 5`, given `ChainExPostFactoG`: `k = 111`, `D = 5` calls of at least 483 µs, `Q = 2^20`, with the cliff at
`d` (`band_cliff_d12`).
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import band as band_codec
from .. import dense as dense_codec
from ..primitives import Primitives
from ..protocol import Certificate, Scheme

CHAIN_CLAIMS = ("ideal-permutation", "random-oracle", "feistel-indifferentiability", "sequential-step-floor",
                "perfect-isolation", "cr/sha-256", "live-verifier", "designated-verifier")
"""What the overwrite-chain certificates rest on: the ideal-permutation model for `Π₂`, instantiated by Feistel10 over
SHAKE256 (its round functions random oracles, the composition a named heuristic); the step floor of at least 483 µs
per 64 KiB `Π₂` call; perfect isolation; SHA-256 collision resistance for the commitment to `W` and for `vk`; and a
live, designated verifier."""

POINT = (512, 524_288)
"""`B` and `ℓ` (bits) of the certified point: 32 MiB segments of 64 KiB blocks."""


def chain_certificate(theorem: str, hypothesis: str, cliff: int | None) -> Certificate:
    """The chain certificate at `POINT` (`band_meets_64` and dense's `chain_meets_64` share its terms)."""
    return Certificate(k=111, rounds=5, work=1 << 20, call_ns=483_000, theorem=theorem, cliff=cliff, segments=1,
                       hypotheses=(hypothesis,), claims=CHAIN_CLAIMS)


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
        return chain_certificate("band_meets_64", "ChainExPostFactoG", p.d)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense_codec.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense_codec.decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["CHAIN_CLAIMS", "POINT", "Band", "chain_certificate"]
