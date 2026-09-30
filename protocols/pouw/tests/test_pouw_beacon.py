"""The beacon's three use conditions for `beacon-unpredictability` (the assessor's B, ratings.md), on recorded drand
quicknet rounds: each round verified against the pinned key, the salt's round after the weight roots, and the round rule
read from the verifier's clock, never from a lagging view."""

from __future__ import annotations

import pytest

import quicknet as Q
from verity_pouw import beacon as B
from verity_pouw import bls12_381 as bls
from test_pouw_pearl_c_work import SPLITS, S, lifecycle
from verity_pouw.audit import Epoch
from verity_pouw.protocol import DomainError
from verity_pouw.schemes import pearl_c_work as W

CHAIN = B.QUICKNET


def test_the_pins_are_quicknet_as_published():
    """The chain's hashes, scheme, period, genesis and 96-byte group key as api.drand.sh publishes them; the decompressed
    key lies on the twist, in G2's prime-order subgroup, and compresses to the published bytes; t = 12 of n = 22."""
    info = Q.DATA["chain"]
    assert (CHAIN.chain_hash, CHAIN.group_hash, CHAIN.scheme, CHAIN.period, CHAIN.genesis) == \
        (info["hash"], info["groupHash"], info["schemeID"], info["period"], info["genesis_time"])
    assert CHAIN.public_key.hex() == info["public_key"] and info["metadata"]["beaconID"] == CHAIN.name
    assert bls.g2_on_curve(CHAIN.key) and bls.g2_mul(CHAIN.key, bls.R) is None
    assert bls.g2_compress(CHAIN.key) == CHAIN.public_key
    assert (CHAIN.threshold, CHAIN.nodes) == (12, 22)


def test_hash_to_g1_is_rfc_9380s():
    """BLS12381G1_XMD:SHA-256_SSWU_RO_ against RFC 9380 Appendix J.9.1's vectors for "" and "abc"."""
    v = Q.DATA["rfc9380_g1_sha256_ro"]
    for msg, (x, y) in v["points"].items():
        assert bls.hash_to_g1(msg.encode(), v["dst"].encode()) == (int(x, 16), int(y, 16))


@pytest.mark.parametrize("number", sorted(Q.ROUNDS))
def test_a_recorded_round_verifies_and_its_value_is_the_published_randomness(number):
    rnd = Q.ROUNDS[number]
    row = next(r for r in Q.DATA["rounds"] if r["round"] == number)
    assert B.verify(CHAIN, rnd) and rnd.value.hex() == row["randomness"]


def test_a_bad_signature_does_not_verify():
    """A flipped bit, another round's signature, a signature not on the curve, the point at infinity, a wrong length, and
    round 0 all fail."""
    rnd = Q.LATEST
    flipped = bytearray(rnd.signature)
    flipped[20] ^= 0x04
    other = next(r for n, r in Q.ROUNDS.items() if n != rnd.round)
    infinity = bytes([0xC0]) + bytes(47)
    for bad in (bytes(flipped), other.signature, bytes([0x80]) + bytes(47), infinity, rnd.signature[:47], b""):
        assert not B.verify(CHAIN, B.Round(rnd.round, bad))
    assert not B.verify(CHAIN, B.Round(0, rnd.signature))


def test_the_schedule_is_drands():
    """Round 1 at genesis, round r at genesis + (r − 1)·3 s, and nothing before genesis."""
    assert B.round_at(CHAIN, CHAIN.genesis) == 1 and B.round_at(CHAIN, CHAIN.genesis - 1) == 0
    for r in (2, 1000, Q.LATEST.round):
        start = CHAIN.genesis + (r - 1) * CHAIN.period
        assert B.round_at(CHAIN, start) == r and B.round_at(CHAIN, start + 2.999) == r and B.round_at(CHAIN, start - 1) == r - 1


def test_the_salt_takes_a_verified_round_after_the_weight_roots():
    """`Epoch.start`: the salt's round must verify and be at least 2 rounds after the one due when the weight roots
    were recorded; the salt is keyed by the round's value."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    work, roots = epoch.workload, prover.weight_roots
    rnd, at = Q.LATEST, Q.recorded_before(Q.LATEST)
    assert Epoch.start(work, roots, rnd, at).salt == Epoch.unverified(work, roots, rnd.value).salt
    with pytest.raises(DomainError, match="not at least 2 rounds after"):
        Epoch.start(work, roots, rnd, at + CHAIN.period)                     # the round was due only 1 round later
    with pytest.raises(DomainError, match="does not verify"):
        Epoch.start(work, roots, B.Round(rnd.round, Q.ROUNDS[1].signature), at)


def test_the_draws_round_is_read_from_the_clock_not_a_lagging_view():
    """`pearl_c_work.audit` takes the verifier's clock at the commitment, not "the latest round seen".  A view lagging
    by 3 rounds would call round r "after" a commitment recorded when round r − 1 was due, so r could already be public;
    the clock-read rule refuses it, needs 2 rounds of margin, and verifies the round's signature."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    rnd, at = Q.LATEST, Q.recorded_before(Q.LATEST)
    assert W.audit(S, epoch, prover, tr, SPLITS, records, at, rnd, 4).audit.accepted
    lagging_at, lagging_view = at + CHAIN.period, rnd.round - 3
    assert W.audit_replay(S, epoch, prover, tr, SPLITS, records, lagging_view, W.Beacon(rnd.round, rnd.value),
                          4).audit.accepted
    out = W.audit(S, epoch, prover, tr, SPLITS, records, lagging_at, rnd, 4)
    assert not out.audit.accepted and "not at least 2 rounds after" in out.audit.verdicts[0].reason
    forged = W.audit(S, epoch, prover, tr, SPLITS, records, at, B.Round(rnd.round, Q.ROUNDS[2].signature), 4)
    assert not forged.audit.accepted and "does not verify" in forged.audit.verdicts[0].reason
