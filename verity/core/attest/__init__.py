"""A party's signed period roots and its liveness records: how an auditor device's signatures reach the verifier only
through the prover's zero-knowledge proofs.

An auditor party (a memory challenger, a network certifier) on a device holds a lower HSS key the device's identity
certified for it in the deployment (`verity.ml.crypto.lms.Identity.issue`, `PartyKey`). Over one certification period it
commits every sampling block's records, each a byte string, as the leaves of one frame-v3-sha512 tree (`root`):

* the domain is `verity.ml.commitments.merkle.CommitmentDomain(BINDING, -1, RangeIndexedDomain(0, count), "sha512")`,
  ``BINDING`` = SHA-512(``verity/period-records/v1``), and record ``r`` is leaf ``(rank r, position r, SCHEMA)``;
* the tree pads to ``2^depth`` leaves, ``depth = bit_length(count - 1)``, and a period with no records has the empty root.

`Period` is the party's side. `Period.block` appends a sampling block's records and returns the block's liveness record,
"block ``w`` of the period was certified" with no content (`lms.liveness_message`): the party's one public message per
block, which the verifier checks in the clear (`lms.verify_liveness`). `Period.close` signs the root once
(`lms.period_root_message`, binding the deployment's hash, the party, the attachment id the challenger drew, the link,
the period's index and the record count) and returns the `SignedPeriod`, which goes to the prover only, as witness. The
prover opens the records a proof reads (`SignedPeriod.read`), and the proof runs `MemoryChallenger.Attest.check` (Lean,
package ``core_memory_challenger``): two LMS verifications for the two HSS levels, then each read's path to the root.

Stdlib only, as a core unit with no ml edge is: the tree is written out here, and its test holds it to
`verity.ml.commitments.merkle`, the signer is injected (`Signer`; `lms.PartyKey` is one).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["BINDING", "SCHEMA", "Period", "Read", "SignedPeriod", "Signer", "domain_id", "leaf", "levels", "path", "root",
           "verify_opening"]

#: the period tree's binding, and every leaf's schema
BINDING = hashlib.sha512(b"verity/period-records/v1").digest()
SCHEMA = "verity/period-record/v1"
_FRAME = b"veritor/protocol/merkle/frame/v3\0"
_TAGGED = b"verity/tagged-sha512/v1\0"
_RANGE = b"veritor/indexed-domain/range/v1"


def _uint(n: int) -> bytes:
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


def _hash(tag: bytes, *parts: bytes) -> bytes:
    return hashlib.sha512(_FRAME + len(tag).to_bytes(4, "big") + tag
                          + b"".join(len(p).to_bytes(8, "big") + p for p in parts)).digest()


def domain_id(count: int) -> bytes:
    """The domain id of a period's tree of ``count`` records (``merkle.CommitmentDomain.domain_id``)."""
    payload = json.dumps({"start": 0, "step": 1, "stop": count}, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha512(_TAGGED + len(_RANGE).to_bytes(4, "big") + _RANGE + len(payload).to_bytes(8, "big")
                              + payload).digest()
    return _hash(b"domain", BINDING, _uint(-1 + 2), identity, _uint(count))


def leaf(did: bytes, position: int, record: bytes) -> bytes:
    return _hash(b"leaf", did, _uint(position), _uint(position), SCHEMA.encode(), bytes(record))


def _depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def levels(records: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The tree's levels, the padded leaves first and the root's last; none for no records."""
    if not records:
        return ()
    did = domain_id(len(records))
    level = [leaf(did, r, x) for r, x in enumerate(records)]
    level += [_hash(b"pad", did, _uint(r)) for r in range(len(records), 1 << _depth(len(records)))]
    out = [tuple(level)]
    for lvl in range(_depth(len(records))):
        level = [_hash(b"node", did, _uint(lvl), _uint(i), level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
        out.append(tuple(level))
    return tuple(out)


def root(records: Sequence[bytes]) -> bytes:
    """The period's root over its records, the empty root for none."""
    ls = levels(records)
    return ls[-1][0] if ls else _hash(b"empty", domain_id(0))


def path(records: Sequence[bytes], position: int) -> tuple[bytes, ...]:
    """The siblings of record ``position``'s leaf, the leaf's side first."""
    if not 0 <= position < len(records):
        raise IndexError(f"a period of {len(records)} records has no position {position}")
    return tuple(level[(position >> k) ^ 1] for k, level in enumerate(levels(records)[:-1]))


def verify_opening(root_: bytes, count: int, position: int, record: bytes, siblings: Sequence[bytes]) -> bool:
    """Whether ``record`` at ``position`` climbs ``siblings`` to ``root_`` in a period tree of ``count`` records
    (`MemoryChallenger.Attest.check`'s test of each read, ``FrameV3Sha512.verifyOpening``)."""
    if len(root_) != 64 or not 0 <= position < count or len(siblings) != _depth(count):
        return False
    if any(len(s) != 64 for s in siblings):
        return False
    did = domain_id(count)
    digest, cursor = leaf(did, position, record), position
    for lvl, sib in enumerate(siblings):
        pair = (digest, bytes(sib)) if cursor % 2 == 0 else (bytes(sib), digest)
        digest, cursor = _hash(b"node", did, _uint(lvl), _uint(cursor >> 1), *pair), cursor >> 1
    return digest == root_


class Signer(Protocol):
    """A party's lower key as `Period` uses it (`lms.PartyKey`): each call signs at its key's next unused leaf and returns
    the two-level signature, whose ``encode()`` is its bytes."""

    def period_root(self, attachment: bytes, link: str, period: int, count: int, root: bytes): ...

    def liveness(self, attachment: bytes, link: str, period: int, block: int): ...


@dataclass(frozen=True)
class Read:
    """One record a period's proof reads: its position, its bytes and its path to the root (`MemoryChallenger.Attest.Read`)."""

    position: int
    record: bytes
    path: tuple[bytes, ...]


@dataclass(frozen=True)
class SignedPeriod:
    """A closed period as the prover receives it, witness only: the period's attachment, link and index, its records in
    order, their root and the party's two-level signature of the root (bytes)."""

    attachment: bytes
    link: str
    period: int
    records: tuple[bytes, ...]
    root: bytes
    signature: bytes

    @property
    def count(self) -> int:
        return len(self.records)

    def read(self, position: int) -> Read:
        return Read(position, self.records[position], path(self.records, position))

    def to_json(self) -> dict:
        return {"attachment": self.attachment.hex(), "link": self.link, "period": self.period,
                "records": [r.hex() for r in self.records], "root": self.root.hex(), "signature": self.signature.hex()}

    @classmethod
    def from_json(cls, d: dict) -> "SignedPeriod":
        return cls(bytes.fromhex(d["attachment"]), d["link"], d["period"], tuple(bytes.fromhex(r) for r in d["records"]),
                   bytes.fromhex(d["root"]), bytes.fromhex(d["signature"]))


@dataclass
class Period:
    """One certification period of a party: its records, sampling block by sampling block, under the party's lower key
    ``signer``. Each block is certified once, in increasing order, and the period is closed once."""

    signer: Signer
    attachment: bytes
    link: str
    period: int
    records: list[bytes] = field(default_factory=list)
    _last: int = -1
    _closed: bool = False

    def block(self, block: int, records: Sequence[bytes]) -> bytes:
        """Append sampling block ``block``'s records and return its public liveness record (the signature's bytes)."""
        if self._closed:
            raise ValueError(f"period {self.period} is closed")
        if block <= self._last:
            raise ValueError(f"block {block} is not after block {self._last}, already certified")
        self.records.extend(bytes(r) for r in records)
        self._last = block
        return self.signer.liveness(self.attachment, self.link, self.period, block).encode()

    def close(self) -> SignedPeriod:
        """Sign the period's root once and return what goes to the prover."""
        if self._closed:
            raise ValueError(f"period {self.period} is already closed")
        self._closed = True
        records = tuple(self.records)
        r = root(records)
        sig = self.signer.period_root(self.attachment, self.link, self.period, len(records), r).encode()
        return SignedPeriod(self.attachment, self.link, self.period, records, r, sig)
