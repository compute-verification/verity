"""The schemes behind `protocol.Scheme`, and the registry of their ids.

Each implements only what differs between schemes, which is encoding and decoding one segment, and states its
certificate at its parameters:

- `Band` (`band-chain/d12/v1` at `band.SECURE_FIRST`), the secure-first scheme: one layer on the band DAG over the
  overwrite chain (Lean `chainBand`).  Certified by `band_meets_64` for one segment at `B = 512`, 64 KiB, every
  `d ≥ 5`, given `ChainExPostFactoG`, with its cliff at `d`.
- `Dense` (`dense-chain/v1` at `dense.POINT_64K`), selectable: the same on the complete DAG (Lean `chainDense`).
  Certified by `chain_meets_64`, given `ChainExPostFacto`, with no cliff.
- `P3` (`p3/v1` at `params.OPERATING_POINT`), research: two layers with a transpose and a tweakable `P`.  It has no
  certificate for the implemented `H`: Lean's `p3_meets_64_D10_final` is about a random-oracle `H`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from . import band, dense, primitives, scheme
from .params import OPERATING_POINT, Params
from .protocol import Certificate, Scheme

CHAIN_CLAIMS = ("ideal-permutation", "random-oracle", "feistel-indifferentiability", "sequential-step-floor",
                "perfect-isolation", "cr/sha-256", "live-verifier", "designated-verifier")
"""What the overwrite-chain certificates rest on: the ideal-permutation model for `Π₂`, instantiated by Feistel10 over
SHAKE256 (its round functions random oracles, the composition a named heuristic); the step floor of at least 483 µs
per 64 KiB `Π₂` call; perfect isolation; SHA-256 collision resistance for the commitment to `W` and for `vk`; and a
live, designated verifier."""

_POINT = (512, 524_288)
"""`B` and `ℓ` (bits) of the certified point: 32 MiB segments of 64 KiB blocks."""


def _chain_certificate(theorem: str, hypothesis: str, cliff: int | None) -> Certificate:
    return Certificate(k=111, rounds=5, work=1 << 20, call_ns=483_000, theorem=theorem, cliff=cliff, segments=1,
                       hypotheses=(hypothesis,), claims=CHAIN_CLAIMS)


class Band(Scheme):
    """The band scheme at in-degree `params.d` (Lean `chainBand B d ℓ 512 (dom tag_s)` per segment)."""

    backend = staticmethod(band.reference)

    def __init__(self, params: band.BandParams = band.SECURE_FIRST, prims: primitives.Primitives | None = None) -> None:
        if not isinstance(params, band.BandParams):
            raise TypeError("the band scheme takes band.BandParams")
        super().__init__(params, prims)

    @property
    def id(self) -> str:
        return band.codec_for(self.params.d).name

    @property
    def certificate(self) -> Certificate | None:
        p = self.params
        if (p.blocks_per_segment, p.m) != _POINT or p.d < 5:
            return None
        return _chain_certificate("band_meets_64", "ChainExPostFactoG", p.d)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense.decode_segment(self.params, self.prims, salt, s, blocks)


class Dense(Scheme):
    """The dense scheme (Lean `chainDense B ℓ 512 (dom tag_s)` per segment)."""

    id = dense.DENSE.name
    backend = staticmethod(dense.reference)

    def __init__(self, params: dense.DenseParams = dense.POINT_64K, prims: primitives.Primitives | None = None) -> None:
        if not isinstance(params, dense.DenseParams):
            raise TypeError("the dense scheme takes dense.DenseParams")
        super().__init__(params, prims)

    @property
    def certificate(self) -> Certificate | None:
        if (self.params.blocks_per_segment, self.params.m) != _POINT:
            return None
        return _chain_certificate("chain_meets_64", "ChainExPostFacto", None)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense.decode_segment(self.params, self.prims, salt, s, blocks)


class P3(Scheme):
    """P3, the two-layer research scheme (Lean `p3Scheme`, with the instantiation's `H` and tweakable `P`)."""

    id = scheme.P3.name
    backend = staticmethod(primitives.reference)

    def __init__(self, params: Params = OPERATING_POINT, prims: primitives.Primitives | None = None) -> None:
        if not isinstance(params, Params):
            raise TypeError("P3 takes params.Params")
        super().__init__(params, prims)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return scheme.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return scheme.decode_segment(self.params, self.prims, salt, s, blocks)


SCHEMES: dict[str, Callable[[int], Scheme]] = {
    band.BAND12.name: lambda segments: Band(band.SECURE_FIRST.with_segments(segments)),
    dense.DENSE.name: lambda segments: Dense(dense.POINT_64K.with_segments(segments)),
    scheme.P3.name: lambda segments: P3(OPERATING_POINT.with_segments(segments)),
}
"""Each registered id at its own parameters, for `W` of `segments` segments."""

DEFAULT = band.BAND12.name
"""The secure-first scheme."""


def get(id: str = DEFAULT, segments: int = 1) -> Scheme:
    """The registered scheme `id` at its parameters, for `segments` segments of `W`."""
    try:
        make = SCHEMES[id]
    except KeyError:
        raise KeyError(f"no POUS scheme {id!r}; registered: {', '.join(sorted(SCHEMES))}") from None
    return make(segments)


__all__ = ["CHAIN_CLAIMS", "DEFAULT", "SCHEMES", "Band", "Dense", "P3", "get"]
