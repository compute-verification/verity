"""`band-chain/d12/v1`, the secure-first scheme: one layer on the band DAG over the overwrite chain (Lean `chainBand B d
ℓ 512 (dom tag_s)` per segment), encoded and decoded by the reference codec `verity.core.protocols.pous.band`.

Its certificate at the frozen point (`band.SECURE_FIRST`: `B = 512`, 64 KiB blocks), for every `d ≥ 5`, with no
hypothesis (in the ideal-permutation model): `k = 111`, `D = 5` calls of at least 241.5 µs, `Q = 2^20`, with the cliff at
`d` (`band_cliff_d12`). It is `Pous.SecurityProofs.BandMultiMeetsFamily`: `Meets` for the `N`-segment store
`chainBandSeg N (2^9) d` keyed by `segDom salt` (segment `s` by `tag_s = salt ‖ u64 s`, as the codec keys it), for
every `N` from 1 to `2^64`, which is every segment count `BandParams` allows. It is one `Meets` for the whole store:
challenges uniform over all its blocks, the state bound of the whole store, one permutation. At one segment it is also
`BandChainMeets64`, and at 448 (14 GB) `BandMultiMeets14`.

The same statement at fewer rounds (`FAMILY_ROUNDS`) gives three more certificates, with a smaller `k` under a tighter
deadline: `BandMultiMeetsFamilyD2`, `k = 97` at `D = 2` (`Δ + RTT < 724.5 µs`) for every `d ≥ 2`,
`BandMultiMeetsFamilyD1`, `k = 93` at `D = 1` (`Δ + RTT < 483 µs`) for every `d ≥ 1`, and `BandMultiMeetsFamilyD0`,
`k = 89` at `D = 0` (`Δ + RTT < 241.5 µs`) for every `d` (at `D = 0` the responder makes no `Π₂` call, so the band
needs no bound on its in-degree). The memory challenger audits under them only when asked for that `k`
(`protocol.TimedAudit`); a band with `d < 5` has no `D = 5` certificate but has these.  Each deadline-to-`D` conversion
here counts whole `Π₂` calls and holds in the ideal-permutation model only (`CHAIN_CLAIMS`).

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
SHAKE256 (its round functions random oracles, the composition a named heuristic, now refuted:
`feistel-indifferentiability`); the step floor of at least 241.5 µs per 64 KiB `Π₂` call in a chain
(`primitives.pi2_keccak_f`), which bounds an adversary only in the ideal-permutation model; perfect isolation; SHA-256
collision resistance for the commitment to `W` and for `vk`; and a live, designated verifier.  At the reference
Feistel-SHAKE256 `Π₂` the certificates do not hold: a prover rebuilds a label from 1.53% of `C` in 481 Keccak-f, about
24 µs, with no chain call (red team, 4 Oct 2026, art:7617c8017e80)."""

POINT = (512, 524_288)
"""`B` and `ℓ` (bits) of the certified point: 32 MiB segments of 64 KiB blocks."""


FAMILY_SEGMENTS = 1 << 64
"""The largest segment count of `BandMultiMeetsFamily` (`N ≤ 2^64`), which is also the most `BandParams` allows."""


FAMILY_ROUNDS = {5: (111, "Pous.SecurityProofs.BandMultiMeetsFamily"),
                 2: (97, "Pous.SecurityProofs.BandMultiMeetsFamilyD2"),
                 1: (93, "Pous.SecurityProofs.BandMultiMeetsFamilyD1"),
                 0: (89, "Pous.SecurityProofs.BandMultiMeetsFamilyD0")}
"""The band's certificates at `POINT` by round budget `D`: `k` and the family pin, each for every `d ≥ D` and every
segment count from 1 to `2^64`. An audit at `D` needs `Δ + RTT < (D + 1)·call_ns`."""


def chain_certificate(theorem: str, cliff: int | None, segments: int | None, rounds: int = 5,
                      k: int = 111) -> Certificate:
    """The chain certificate at `POINT` (the band's `BandMultiMeetsFamily` and dense's `ChainDenseMeets64` share its
    terms at `D = 5`)."""
    return Certificate(k=k, rounds=rounds, work=1 << 20, call_ns=pi2_call_ns(POINT[1]), theorem=theorem, cliff=cliff,
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
        """The default certificate, at `D = 5` (`d ≥ 5`)."""
        return next((c for c in self.certificates if c.rounds == 5), None)

    @property
    def certificates(self) -> tuple[Certificate, ...]:
        """`FAMILY_ROUNDS` at every `D ≤ d`, most rounds (the default, when `d ≥ 5`) first."""
        p = self.params
        if (p.blocks_per_segment, p.m) != POINT:
            return ()
        return tuple(chain_certificate(theorem, p.d, p.segments, rounds, k)
                     for rounds, (k, theorem) in FAMILY_ROUNDS.items() if rounds <= p.d)

    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        return dense_codec.encode_segment(self.params, self.prims, salt, s, w_seg)

    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        return dense_codec.decode_segment(self.params, self.prims, salt, s, blocks)


__all__ = ["CHAIN_CLAIMS", "FAMILY_ROUNDS", "FAMILY_SEGMENTS", "POINT", "Band", "chain_certificate"]
