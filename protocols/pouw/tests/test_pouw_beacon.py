"""The beacon's three use conditions for `beacon-unpredictability` (the assessor's A for the verifier path, ratings.md
21:48Z), on recorded drand quicknet rounds: each round verified against the pinned key, the salt's round after the weight
roots, and the round rule read from the verifier's clock, never from a lagging view."""

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
    assert (CHAIN.threshold, CHAIN.nodes) == (12, "22 at launch, 23 per FIP-0063")


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


def torsion_point():
    """A point of E(Fp) of order dividing G1's cofactor, not the identity: r·Q for the first curve point Q from x = 1."""
    x = 1
    while True:
        y = bls.sqrt((x ** 3 + 4) % bls.P)
        if y is not None:
            t = bls.g1_mul((x, y), bls.R)
            if t is not None:
                return t
        x += 1


def test_a_torsion_shifted_signature_does_not_verify():
    """σ + T, with T of small order outside G1's prime-order subgroup: the point is on the curve, its encoding is
    well-formed, and decoding refuses it at the subgroup check, so the round fails."""
    rnd = Q.LATEST
    sig = bls.g1_decompress(rnd.signature)
    assert sig is not None and bls.g1_compress(sig) == rnd.signature
    shifted = bls.g1_add(sig, torsion_point())
    assert bls.g1_on_curve(shifted) and bls.g1_mul(shifted, bls.R) is not None
    assert bls.g1_decompress(bls.g1_compress(shifted)) is None
    assert not B.verify(CHAIN, B.Round(rnd.round, bls.g1_compress(shifted)))


def test_an_encoding_with_x_at_least_p_does_not_verify():
    """A compressed x of p itself, and a signature's x + p where that still fits the 381 bits: both refused, so no
    residue has a second encoding."""
    rnd = Q.LATEST
    at_p = bytearray(bls.P.to_bytes(48, "big"))
    at_p[0] |= 0x80
    assert bls.g1_decompress(bytes(at_p)) is None and not B.verify(CHAIN, B.Round(rnd.round, bytes(at_p)))
    for r in Q.ROUNDS.values():
        x = int.from_bytes(bytes([r.signature[0] & 0x1F]) + r.signature[1:], "big")
        if x + bls.P < 1 << 381:
            alias = bytearray((x + bls.P).to_bytes(48, "big"))
            alias[0] |= r.signature[0] & 0xE0
            assert bls.g1_decompress(bytes(alias)) is None and not B.verify(CHAIN, B.Round(r.round, bytes(alias)))


def test_the_verifier_agrees_with_py_ecc():
    """A cross-check against py_ecc, which is not a dependency: skipped where it is absent. Its hash to G1 and its
    pairing check agree with ours on every recorded round, good and with a flipped bit."""
    pytest.importorskip("py_ecc")
    import hashlib

    from py_ecc.bls.hash_to_curve import hash_to_G1
    from py_ecc.bls.point_compression import decompress_G1
    from py_ecc.optimized_bls12_381 import FQ2, FQ12, G2, final_exponentiate, neg, normalize
    from py_ecc.optimized_bls12_381.optimized_pairing import miller_loop

    (x0, x1), (y0, y1) = CHAIN.key
    pk = (FQ2([x0, x1]), FQ2([y0, y1]), FQ2.one())
    for rnd in Q.ROUNDS.values():
        h = normalize(hash_to_G1(B.message(rnd.round), B.DST, hashlib.sha256))
        assert bls.hash_to_g1(B.message(rnd.round), B.DST) == (int(h[0].n), int(h[1].n))
        flipped = bytearray(rnd.signature)
        flipped[30] ^= 0x01
        for sig in (rnd.signature, bytes(flipped)):
            try:
                point = decompress_G1(int.from_bytes(sig, "big"))
                theirs = final_exponentiate(miller_loop(G2, point) * miller_loop(pk, neg(hash_to_G1(
                    B.message(rnd.round), B.DST, hashlib.sha256)))) == FQ12.one()
            except Exception:
                theirs = False
            assert B.verify(CHAIN, B.Round(rnd.round, sig)) == theirs


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
