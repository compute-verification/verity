"""`verity.core.attest`: the period tree held to `verity.ml.commitments.merkle`, and a party's period under a small HSS key
(H = 4): its liveness records verify in the clear, its signed root and each record's opening are what
`Attest.accepts` accepts (here the Python it is held to), every misuse is refused, a block's liveness record is signed
before its records are appended, and a restart within the attachment signs nothing more of it."""
from __future__ import annotations

import hashlib

import pytest

from verity.core import attest as A
from verity.ml.commitments.gates import attest as attest_gate
from verity.ml.commitments.indexed import RangeU64Domain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import (
    Commitment,
    CommitmentDomain,
    MerkleTree,
    Opening,
    verify_opening,
)
from verity.ml.crypto import lms

SMALL = lms.Params(64, 1, 4)
DEPLOYMENT = hashlib.sha512(b"a deployment configuration").digest()
ATTACHMENT = bytes(range(32))


def _records(n: int) -> list[bytes]:
    return [hashlib.sha512(b"record %d" % i).digest()[: 1 + i % 70] for i in range(n)]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 9, 17])
def test_the_tree_is_merkles_frame_v3_sha512(n: int) -> None:
    """Root, domain id and every opening are `merkle`'s over the range-u64 domain at `BINDING`, owner -1, schema `SCHEMA`."""
    records = _records(n)
    domain = CommitmentDomain(A.BINDING, -1, RangeU64Domain(0, n), hash="sha512")
    assert A.domain_id(n) == domain.domain_id
    tree = MerkleTree(domain, dict(enumerate(records)), lambda _: A.SCHEMA)
    assert A.root(records) == tree.commitment.root
    for r in range(n):
        opening = tree.open(r)
        assert A.path(records, r) == opening.path
        assert A.verify_opening(A.root(records), n, r, records[r], opening.path)
        assert verify_opening(domain, Commitment(A.root(records), n), Opening(r, records[r], A.path(records, r)), A.SCHEMA,
                              VerificationLimits())


def test_the_domain_id_frames_a_u64_count() -> None:
    """At the u64 bounds too, and at the period unit's binding (`AttestPeriod_v1` computes the id in its proof)."""
    assert attest_gate.BINDING == A.BINDING
    for n in (2**32, 2**64 - 1):
        assert A.domain_id(n) == CommitmentDomain(A.BINDING, -1, RangeU64Domain(0, n), hash="sha512").domain_id
    for n in (-1, 2**64):
        with pytest.raises(ValueError, match="u64"):
            A.domain_id(n)
    one = _records(1)
    assert A.verify_opening(A.root(one), 1, 0, one[0], ())
    assert not A.verify_opening(A.root(one), 2**64, 0, one[0], ())


def test_an_opening_is_refused_off_its_record_position_count_or_root() -> None:
    records = _records(5)
    root, p = A.root(records), A.path(records, 3)
    assert A.verify_opening(root, 5, 3, records[3], p)
    assert not A.verify_opening(root, 5, 3, records[2], p)
    assert not A.verify_opening(root, 5, 2, records[3], p)
    assert not A.verify_opening(root, 6, 3, records[3], p)
    assert not A.verify_opening(root, 5, 5, records[3], p)
    assert not A.verify_opening(root[::-1], 5, 3, records[3], p)
    assert not A.verify_opening(root, 5, 3, records[3], p[:-1])
    assert not A.verify_opening(root, 5, 3, records[3], (p[0][:-1], *p[1:]))
    with pytest.raises(IndexError):
        A.path(records, 5)


@pytest.fixture
def party(tmp_path) -> lms.PartyKey:
    return lms.Identity.init(tmp_path / "device", SMALL).issue(DEPLOYMENT, "memory-challenger")


def _accepts(party: lms.PartyKey, s: A.SignedPeriod, reads: list[A.Read]) -> bool:
    """`Attest.accepts`, in the Python it is held to: the lower key's certificate in the clear (`Attest.checkKey`), then
    the root's signature under the lower key and each read's opening (`Attest.check`)."""
    sig = lms.HssSignature.decode(SMALL, s.signature)
    if not lms.verify_certificate(SMALL, party.identity, DEPLOYMENT, "memory-challenger", sig.lower, sig.certificate):
        return False
    m = lms.period_root_message(sig.lower[:16], DEPLOYMENT, "memory-challenger", s.attachment, s.link, s.period, s.count,
                                s.root)
    return (lms.verify(SMALL, sig.lower, m, sig.signature)
            and all(A.verify_opening(s.root, s.count, r.position, r.record, r.path) for r in reads))


def test_a_period_signs_liveness_per_block_and_its_root_once(party) -> None:
    period = A.Period(party, ATTACHMENT, "server-3", 7)
    blocks = [_records(3), [], _records(2)]
    for w, records in enumerate(blocks):
        live = period.block(w, records)
        assert lms.verify_liveness(SMALL, party.identity, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, w,
                                   lms.HssSignature.decode(SMALL, live))
    s = period.close()
    assert s.records == tuple(r for b in blocks for r in b) and s.root == A.root(s.records)
    reads = [s.read(i) for i in range(s.count)]
    assert _accepts(party, s, reads)
    assert A.SignedPeriod.from_json(s.to_json()) == s
    with pytest.raises(ValueError, match="closed"):
        period.close()
    with pytest.raises(ValueError, match="closed"):
        period.block(3, [])


def test_a_read_or_root_the_party_did_not_sign_is_refused(party) -> None:
    period = A.Period(party, ATTACHMENT, "server-3", 7)
    period.block(0, _records(4))
    s = period.close()
    forged = A.Read(1, b"not the record", s.read(1).path)
    assert not _accepts(party, s, [s.read(0), forged])
    other = _records(5)
    moved = A.SignedPeriod(s.attachment, s.link, s.period, tuple(other), A.root(other), s.signature)
    assert not _accepts(party, moved, [moved.read(0)])
    for field, value in [("link", "server-4"), ("period", 8), ("attachment", ATTACHMENT[::-1])]:
        changed = A.SignedPeriod(**{**s.__dict__, field: value})
        assert not _accepts(party, changed, [changed.read(0)]), field


def test_blocks_are_certified_once_in_order(party) -> None:
    period = A.Period(party, ATTACHMENT, "server-3", 0)
    period.block(2, [])
    for w in (2, 1):
        with pytest.raises(ValueError, match="already certified"):
            period.block(w, [])


class _Refusing:
    """A signer that starts any attachment and refuses every signature."""

    def start(self, attachment: bytes) -> None:
        pass

    def period_root(self, *args):
        raise lms.StateError("refused")

    def liveness(self, *args):
        raise lms.StateError("refused")


def test_a_block_whose_liveness_record_is_refused_appends_nothing() -> None:
    period = A.Period(_Refusing(), ATTACHMENT, "server-3", 0)
    with pytest.raises(lms.StateError):
        period.block(0, _records(3))
    assert period.records == []


def test_a_restart_within_the_attachment_signs_nothing_more_of_it(party) -> None:
    """The party's key reopened from its directory, as a restarted party would, is refused the attachment its periods
    started, so it opens no period of it, and signs for another attachment."""
    period = A.Period(party, ATTACHMENT, "server-3", 7)
    period.block(0, _records(2))
    restarted = lms.PartyKey(party.directory, party.identity, DEPLOYMENT, "memory-challenger", SMALL)
    with pytest.raises(lms.StateError, match="already started"):
        A.Period(restarted, ATTACHMENT, "server-3", 7)
    with pytest.raises(lms.StateError, match="not the one this key started"):
        restarted.liveness(ATTACHMENT, "server-3", 7, 1)
    A.Period(party, ATTACHMENT, "server-3", 8).block(0, [])
    other = A.Period(restarted, ATTACHMENT[::-1], "server-3", 7)
    live = other.block(0, _records(1))
    assert lms.verify_liveness(SMALL, party.identity, DEPLOYMENT, "memory-challenger", ATTACHMENT[::-1], "server-3", 7, 0,
                               lms.HssSignature.decode(SMALL, live))
