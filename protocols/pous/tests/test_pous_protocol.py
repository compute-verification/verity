"""The generic protocol (`verity_pous.protocol`) over every scheme (`verity_pous.schemes`): one conformance suite for
the lifecycle, the certificates the verifier enforces, the storage profile and the claims it cites."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import pytest

from verity.claims import get as core_claim
from verity_pous import band, dense, protocol, schemes
from verity_pous.codec import Geometry
from verity_pous.layout import Layout
from verity_pous.params import Params
from verity_pous.verifier import VerifierKey

SALT = bytes(range(24))
SMALL = {
    "band": lambda segments=3: schemes.Band(band.BandParams(16, 64, 4, segments)),
    "dense": lambda segments=3: schemes.Dense(dense.DenseParams(8, 64, segments)),
    "p3": lambda segments=2: schemes.P3(Params(8, 64, 3, segments)),
}


class Stream(protocol.Scheme):
    """A toy scheme (no security): each block xored with a SHAKE-256 stream.  It implements only the two methods."""

    id = "toy-stream/v0"
    backend = staticmethod(lambda params: None)

    def _pad(self, salt: bytes, s: int, v: int) -> bytes:
        return hashlib.shake_256(salt + s.to_bytes(8, "little") + v.to_bytes(2, "little")).digest(self.params.block_bytes)

    def encode_segment(self, salt, s, w_seg):
        mb = self.params.block_bytes
        return [bytes(a ^ b for a, b in zip(w_seg[v * mb:(v + 1) * mb], self._pad(salt, s, v)))
                for v in range(self.params.blocks_per_segment)]

    def decode_segment(self, salt, s, blocks):
        return b"".join(bytes(a ^ b for a, b in zip(c, self._pad(salt, s, v))) for v, c in enumerate(blocks))


SMALL["toy"] = lambda segments=2: Stream(dense.DenseParams(4, 64, segments))


def _w(params: Geometry, seed: int) -> bytes:
    return random.Random(seed).randbytes(params.w_bytes)


def _ticks(step: int = 1_000):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def _lifecycle(sch: protocol.Scheme, seed: int = 1):
    w = _w(sch.params, seed)
    com = protocol.commit(sch, w)
    salt = protocol.draw_salt(b"verifier secret", com)
    return w, com, salt, protocol.encode(sch, w, salt)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_round_trip_and_both_setups(name):
    sch = SMALL[name]()
    w, com, salt, c = _lifecycle(sch)
    assert protocol.decode(sch, salt, c) == w
    private = protocol.verifier_key(sch, com, salt, w=w)
    public = protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=c)
    assert private.root == public.root and private.leaves == public.leaves


@pytest.mark.parametrize("name", sorted(SMALL))
def test_setups_reject(name):
    sch = SMALL[name]()
    w, com, salt, c = _lifecycle(sch)
    flipped = [list(seg) for seg in c]
    flipped[-1][3] = bytes([flipped[-1][3][0] ^ 1]) + flipped[-1][3][1:]
    with pytest.raises(ValueError):
        protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=flipped)
    with pytest.raises(ValueError):
        protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=c[:-1])
    with pytest.raises(ValueError):
        protocol.verifier_key(sch, com, salt, w=_w(sch.params, 2))
    with pytest.raises(ValueError):
        protocol.verifier_key(sch, com, salt, mode="trusted", w=w)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_audit_through_the_interface(name):
    sch = SMALL[name]()
    w, com, salt, c = _lifecycle(sch)
    vk = protocol.verifier_key(sch, com, salt, mode=protocol.PUBLIC, encoded=c)
    blocks = [b for seg in c for b in seg]
    key = protocol.challenge_key(b"audit secret", vk)
    with pytest.raises(ValueError, match="no certificate"):
        protocol.Verifier(sch, vk, key, k=20)
    v = protocol.Verifier(sch, vk, key, k=20, research=True, clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    profile = v.profile()
    assert not profile.certified and profile.state_bound_bits is None and "no certificate" in profile.statement()
    wrong = protocol.Verifier(sch, vk, key, k=20, research=True, clock=_ticks())
    assert not wrong.run(lambda i: bytes(len(blocks[i]))).accepted
    with pytest.raises(ValueError, match="accepted"):
        wrong.profile()


@pytest.mark.parametrize("name", ["band", "dense", "p3"])
def test_decode_tensor(name):
    base = SMALL[name]()
    layout = Layout.of(base.params.with_segments(1), [("embed", 150), ("norm", 3), ("proj", 170)])
    sch = type(base)(layout.params)
    tensors = {"embed": random.Random(3).randbytes(150), "norm": b"\x01\x02\x03", "proj": bytes(range(170))}
    w, salt = layout.pack(tensors), SALT
    blocks = [b for seg in protocol.encode(sch, w, salt) for b in seg]
    for name_, data in tensors.items():
        assert protocol.decode_tensor(sch, layout, salt, name_, blocks.__getitem__) == data


def test_band_through_the_interface_is_the_pinned_band_codec():
    import pous_vectors

    small = schemes.Band(band.BandParams(16, 64, 4, 2))
    w, salt = pous_vectors._bytes("band-small-w", small.params.w_bytes), pous_vectors._bytes("salt", 24)
    pinned = json.loads((Path(__file__).parent / "fixtures" / "vectors.json").read_text())["band/small/blocks"]
    assert [b.hex() for seg in protocol.encode(small, w, salt) for b in seg] == pinned
    assert small.codec().name == band.codec_for(4).name == small.id


def test_registry():
    assert set(schemes.SCHEMES) == {"band-chain/d12/v1", "dense-chain/v1", "p3/v1"}
    assert schemes.DEFAULT == band.BAND12.name
    b = schemes.get()
    assert isinstance(b, schemes.Band) and b.params == band.SECURE_FIRST and b.id == "band-chain/d12/v1"
    assert schemes.get(segments=30).params.segments == 30
    assert schemes.get("dense-chain/v1").params == dense.POINT_64K
    with pytest.raises(KeyError, match="registered"):
        schemes.get("band-chain/d13/v1")
    with pytest.raises(TypeError):
        protocol.Scheme(band.SECURE_FIRST)
    with pytest.raises(TypeError):
        schemes.Band(dense.POINT_64K)


def test_certificates_at_their_points():
    b = schemes.get().certificate
    assert (b.k, b.rounds, b.work, b.call_ns, b.cliff, b.segments) == (111, 5, 1 << 20, 483_000, 12, 1)
    assert (b.theorem, b.hypotheses) == ("band_meets_64", ("ChainExPostFactoG",))
    d = schemes.get("dense-chain/v1").certificate
    assert (d.theorem, d.hypotheses, d.cliff, d.k, d.rounds) == ("chain_meets_64", ("ChainExPostFacto",), None, 111, 5)
    assert schemes.get("p3/v1").certificate is None
    assert schemes.Band(band.BandParams(512, 524_288, 4)).certificate is None      # band_meets_64 needs d ≥ 5
    assert schemes.Band(band.BandParams(512, 524_288, 5)).certificate.cliff == 5
    assert schemes.Band(band.BandParams(256, 524_288, 12)).certificate is None
    with pytest.raises(ValueError, match="cliff"):
        protocol.Certificate(111, 6, 1 << 20, 483_000, "x", cliff=5)
    with pytest.raises(KeyError):
        protocol.Certificate(111, 5, 1 << 20, 483_000, "x", claims=("no-such-claim",))


def _fake_vk(sch: protocol.Scheme) -> VerifierKey:
    return VerifierKey.from_leaves(sch.params, SALT, [hashlib.sha256(bytes([i % 251])).digest()
                                                     for i in range(sch.params.blocks)])


def test_the_verifier_enforces_the_certificate():
    """k = 111 is certified only for D = 5 calls of 483 µs: Δ + RTT < 2.898 ms, inside the cliff at 6.279 ms."""
    sch = schemes.get()
    vk, key = _fake_vk(sch), b"\x00" * 32
    k = protocol.challenge_key(key, vk)
    assert protocol.Verifier(sch, vk, k, rtt_allowance_ns=36_800).k == 111
    assert protocol.Verifier(sch, vk, k, rtt_allowance_ns=2_397_999).gaps == ()
    with pytest.raises(ValueError, match="covers 5"):
        protocol.Verifier(sch, vk, k, rtt_allowance_ns=2_398_000)
    with pytest.raises(ValueError, match="below"):
        protocol.Verifier(sch, vk, k, k=110)
    uncertified = protocol.Verifier(sch, vk, k, rtt_allowance_ns=3_000_000, research=True)
    assert any("covers 5" in g for g in uncertified.gaps)
    with pytest.raises(ValueError, match="cliff"):
        protocol.Verifier(sch, vk, k, rtt_allowance_ns=5_779_000, research=True)
    with pytest.raises(ValueError, match="parameters"):
        protocol.Verifier(schemes.get(segments=2), vk, k)


def test_the_dense_certificate_has_no_cliff_but_the_same_rounds():
    sch = schemes.get("dense-chain/v1")
    vk = _fake_vk(sch)
    k = protocol.challenge_key(b"\x00" * 32, vk)
    with pytest.raises(ValueError, match="covers 5"):
        protocol.Verifier(sch, vk, k, rtt_allowance_ns=2_398_000)
    far = protocol.Verifier(sch, vk, k, rtt_allowance_ns=10_000_000, research=True)
    assert not any("cliff" in g for g in far.gaps)


def test_certified_profile_of_one_band12_segment():
    """An accepted audit at the frozen point gives the certified profile.  The audit reads only `vk` and the answers,
    so random blocks stand in for an encoding here (encoding a 32 MiB segment takes about 25 s in Python)."""
    sch = schemes.get()
    rng = random.Random(7)
    blocks = [rng.randbytes(sch.params.block_bytes) for _ in range(sch.params.blocks)]
    vk = VerifierKey.from_segments(sch.params, SALT, [blocks])
    v = protocol.Verifier(sch, vk, protocol.challenge_key(b"audit secret", vk), rtt_allowance_ns=36_800, clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    p = v.profile()
    assert p.certified and p.gaps == () and p.effective_rounds == 1
    assert p.state_bound_bits == 254_307_274 == (512 * 524_288 * 18) // 19
    assert "band_meets_64, given ChainExPostFactoG" in p.statement()
    j = p.to_json()
    assert j["schema"] == "verity-pous/storage-profile/v1" and j["certified"] and j["certificate"]["k"] == 111
    late = protocol.Verifier(sch, vk, protocol.challenge_key(b"audit secret", vk), clock=_ticks(600_000))
    assert not late.run(protocol.responder(sch, blocks.__getitem__)).accepted


def test_many_segments_are_a_gap_not_a_refusal():
    """`band_meets_64` is about one segment; the 14 GB multi-segment form is the next item."""
    sch = schemes.get(segments=2)
    v = protocol.Verifier(sch, _fake_vk(sch), protocol.challenge_key(b"\x00" * 32, _fake_vk(sch)))
    assert v.gaps == ("W has 2 segments; band_meets_64 is about 1",)


def test_claims_resolve_and_local_ones_are_not_in_core():
    for sch in (schemes.get(), schemes.get("dense-chain/v1")):
        for c in sch.certificate.claims:
            assert protocol.claim(c).id == c
    for c in protocol.LOCAL_CLAIMS:
        with pytest.raises(KeyError):
            core_claim(c)
        assert "/" not in c and c == c.lower()
    assert protocol.claim("cr/sha-256").kind == "assumption"
