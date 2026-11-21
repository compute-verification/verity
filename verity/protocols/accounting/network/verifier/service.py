"""N2's link (`TransferBlocks`): the network attester's two attested records of each execution phase, egress and
ingress, on its one grid under its schedule's parameters digest, and the attestations' check on their public fields
(`certified`). `encode` is an attested record's leaf bytes.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

from verity.protocols.accounting.network.verifier.certificate import (DIGEST_BYTES, Certificate, CertifiedRecord,
                                                                               PublicKey)
from verity.protocols.accounting.network.verifier.grid import GridParams


def encode(record: CertifiedRecord) -> bytes:
    """The leaf's bytes: the record head (`RecordHead.encode`), the attester's salt (64 bytes), then the attestation,
    every field fixed-width or length-prefixed: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖
    u64be window ‖ signed digest (64) ‖ u32be len(signature) ‖ signature."""
    c = record.certificate
    link = c.link.encode("utf-8")
    return b"".join([record.head.encode(), record.salt, struct.pack(">H", len(c.key_id)), c.key_id,
                     struct.pack(">H", len(link)), link, struct.pack(">BQ", c.direction.code, c.window), c.digest,
                     struct.pack(">I", len(c.signature)), c.signature])


def grid_bytes(params: GridParams, parameters: bytes) -> bytes:
    """The declared grid as a record head encodes it: T, r, B and τ_b (u32 big-endian), then the parameters digest."""
    if len(parameters) != DIGEST_BYTES:
        raise ValueError(f"a parameters digest is {DIGEST_BYTES} bytes")
    return struct.pack(">IIII", params.buckets, params.allocation, params.frame_bytes, params.bucket_ms) + parameters


@dataclass(frozen=True)
class NetworkAttesterProfile:
    """A network attester link's profile: the link's windows and the network attester's parameters digest
    (`network_attester.NetworkAttesterParams.digest`), which both ends' heads bind, both fixed before the run."""

    windows: int
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError(f"a network attester link has one or more windows, not {self.windows!r}")
        if type(self.digest) is not bytes or len(self.digest) != DIGEST_BYTES:
            raise ValueError(f"a network certifier's parameters digest is {DIGEST_BYTES} bytes")


@dataclass(frozen=True)
class TransferBlocks:
    """One link, every field public: the link, its egress and ingress grids, the network attester's public key, the
    run, the attester session's id (the 32 bytes the challenger sampled for the attester's session, which every
    attestation signs), and the network attester's profile (`network_attester`)."""

    link: str
    params: GridParams
    key: PublicKey
    run: str
    certifier_session: bytes
    ingress: GridParams | None = None
    network_attester: NetworkAttesterProfile | None = None

    def __post_init__(self) -> None:
        if self.network_attester is None:
            raise ValueError("a link is the network attester's: TransferBlocks(network_attester=NetworkAttesterProfile(windows, digest))")
        if self.ingress_params != self.params:
            raise ValueError("a network certifier link's two ends are on the network certifier's one grid")

    @property
    def ingress_params(self) -> GridParams:
        return self.ingress or self.params

    def grid(self, direction: int) -> bytes:
        """The end's declared grid as its head encodes it (`grid_bytes`): the network attester's grid under its
        parameters digest (`NetworkAttesterProfile.digest`)."""
        return grid_bytes((self.params, self.ingress_params)[direction], self.network_attester.digest)

    @property
    def windows(self) -> int:
        return self.network_attester.windows

    def certified(self, window: int, certificates: Sequence[Certificate]) -> bool:
        """The attestations' check, on the host from their public fields alone: two attestations, each the attester's
        and verifying in its session over its fields and its signed digest (`Certificate.verify`), and each naming this
        link, its end (egress first) and the window."""
        return len(certificates) == 2 and all(
            isinstance(c, Certificate) and (c.link, c.direction.code, c.window) == (self.link, d, window)
            and c.verify(self.key, self.certifier_session) for d, c in enumerate(certificates))


__all__ = ["NetworkAttesterProfile", "TransferBlocks", "encode", "grid_bytes"]
