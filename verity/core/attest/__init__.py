"""A party's signed period roots and its liveness records: how an auditor device's signatures reach the verifier only
through the prover's zero-knowledge proofs.

An auditor party (a memory challenger, a network certifier) on a device holds a lower HSS key the device's identity
certified for it in the deployment (`verity.ml.crypto.lms.Identity.issue`, `PartyKey`). Over one certification period it
commits every sampling block's records, each a byte string, as the leaves of one frame-v3-sha512 tree (`root`):

* the domain is `verity.ml.commitments.merkle.CommitmentDomain(BINDING, -1, RangeU64Domain(0, count), "sha512")`,
  ``BINDING`` = SHA-512(``verity/period-records/v1``), and record ``r`` is leaf ``(rank r, position r, SCHEMA)``; its id
  frames the owner, the start and the count as u64be words (`merkle.range_u64_domain_message`), so a period's proof binds
  the count it signs to the tree its reads open in one compression (``AttestPeriod_v1``);
* the tree pads to ``2^depth`` leaves, ``depth = bit_length(count - 1)``, and a period with no records has the empty root.

`Period` is the party's side. `Period.block` appends a sampling block's records and returns the block's liveness record,
"block ``w`` of the period was certified" with no content (`lms.liveness_message`): the party's one public message per
block, which the verifier checks in the clear (`lms.verify_liveness`). `Period.close` signs the root once
(`lms.period_root_message`, binding the deployment's hash, the party, the attachment id the challenger drew, the link,
the period's index and the record count) and returns the `SignedPeriod`, which goes to the prover only, as witness. The
party's lower key and the identity's certificate of it are public: every liveness record shows them in the clear. The
prover opens the records a proof reads (`SignedPeriod.read`); the verifier checks the certificate in the clear
(`Attest.checkKey`), and the proof runs `Attest.check` (Lean, this directory's ``lean/``, package ``core_attest``): the
root's LMS signature under the lower key, then each read's path to the root. `Attest.accepts` is both. The period's unit,
the root's signature as a Boolean Definition, is ``AttestPeriod_v1`` (`verity.ml.commitments.gates.attest`).

The party signs for one attachment at a time: `Period` starts its attachment on the signer (`lms.PartyKey.start`, which
records it on the key's leaf state before any signature of it), so a restart within the attachment signs nothing more
of it, under no-state-rollback. `Period.block` signs the block's liveness record before it appends the block's records,
so every record a closed period holds is in a block whose liveness record is out.

A party key signs period roots and liveness records only. What a party signs per challenge outside its periods (the
network certifier's reveals) is no record of a period: it stays under the party's own key, and no statement about a
period reads it as the device's.

Stdlib only, as a core unit with no ml edge is: the tree is written out here, and its test holds it to
`verity.ml.commitments.merkle`, the signer is injected (`Signer`; `lms.PartyKey` is one).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["BINDING", "SCHEMA", "Period", "Read", "SignedPeriod", "Signer", "domain_id", "leaf", "levels", "path", "root",
           "verify_opening"]

#: the period tree's binding, and every leaf's schema
BINDING = hashlib.sha512(b"verity/period-records/v1").digest()
SCHEMA = "verity/period-record/v1"
_FRAME = b"veritor/protocol/merkle/frame/v3\0"


def _uint(n: int) -> bytes:
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


def _u64(n: int) -> bytes:
    return n.to_bytes(8, "big")


def _hash(tag: bytes, *parts: bytes) -> bytes:
    return hashlib.sha512(_FRAME + len(tag).to_bytes(4, "big") + tag
                          + b"".join(len(p).to_bytes(8, "big") + p for p in parts)).digest()


def domain_id(count: int) -> bytes:
    """The domain id of a period's tree of ``count`` records, a u64 (``merkle.CommitmentDomain.domain_id``)."""
    if not 0 <= count < 1 << 64:
        raise ValueError(f"a period's record count is a u64, not {count}")
    return _hash(b"domain/range-u64", BINDING, _u64(-1 + 2), _u64(0), _u64(count))


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
    (`Attest.check`'s test of each read, ``Flock.FV3.Domain.verifyOpening``)."""
    if len(root_) != 64 or not 0 <= position < count < 1 << 64 or len(siblings) != _depth(count):
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
    """A party's lower key as `Period` uses it (`lms.PartyKey`): `start` starts its run of an attachment, refused for one
    it has started before, and each signing call, for the started attachment only, signs at its key's next unused leaf
    and returns the two-level signature, whose ``encode()`` is its bytes."""

    def start(self, attachment: bytes) -> None: ...

    def period_root(self, attachment: bytes, link: str, period: int, count: int, root: bytes): ...

    def liveness(self, attachment: bytes, link: str, period: int, block: int): ...


@dataclass(frozen=True)
class Read:
    """One record a period's proof reads: its position, its bytes and its path to the root (`Attest.Read`)."""

    position: int
    record: bytes
    path: tuple[bytes, ...]


@dataclass(frozen=True)
class SignedPeriod:
    """A closed period as the prover receives it: the period's attachment, link and index, its records in order, their
    root and the party's two-level signature of the root (bytes). The signature's lower key and certificate are public
    (`Attest.PartyKey`); the rest of it, the records and the root are witness."""

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
    ``signer``, which it starts on the attachment (`Signer.start`). Each block is certified once, in increasing order,
    and the period is closed once."""

    signer: Signer
    attachment: bytes
    link: str
    period: int
    records: list[bytes] = field(default_factory=list)
    _last: int = -1
    _closed: bool = False

    def __post_init__(self) -> None:
        self.signer.start(self.attachment)

    def block(self, block: int, records: Sequence[bytes]) -> bytes:
        """Sign sampling block ``block``'s public liveness record, then append the block's records; the record's bytes.
        A refused signature appends nothing."""
        if self._closed:
            raise ValueError(f"period {self.period} is closed")
        if block <= self._last:
            raise ValueError(f"block {block} is not after block {self._last}, already certified")
        live = self.signer.liveness(self.attachment, self.link, self.period, block).encode()
        self.records.extend(bytes(r) for r in records)
        self._last = block
        return live

    def close(self) -> SignedPeriod:
        """Sign the period's root once and return what goes to the prover."""
        if self._closed:
            raise ValueError(f"period {self.period} is already closed")
        self._closed = True
        records = tuple(self.records)
        r = root(records)
        sig = self.signer.period_root(self.attachment, self.link, self.period, len(records), r).encode()
        return SignedPeriod(self.attachment, self.link, self.period, records, r, sig)
