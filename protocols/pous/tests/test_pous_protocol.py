"""The generic protocol (`verity_pous.protocol`) over every scheme (`verity_pous.schemes`): one conformance suite for
the lifecycle, the certificates the verifier enforces, the storage profile and the claims it cites."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import pytest

from verity.claims import get as core_claim
from verity_pous import band, dense, protocol, schemes
from verity_pous.codec import Geometry
from verity_pous.layout import Layout
from verity_pous.params import Params
from verity_pous.schemes.p2 import EXPERIMENTAL, P2Params
from verity_pous.verifier import VerifierKey

SALT = bytes(range(24))
SMALL = {
    "band": lambda segments=3: schemes.Band(band.BandParams(16, 64, 4, segments)),
    "dense": lambda segments=3: schemes.Dense(dense.DenseParams(8, 64, segments)),
    "p3": lambda segments=2: schemes.P3(Params(8, 64, 3, segments)),
    "p2": lambda segments=2: schemes.P2(P2Params(128, 173, 8, segments)),   # 2^128 − 173 is prime, ≡ 3 mod 4
}


class Stream(protocol.Scheme):
    """A toy scheme (no security): each block xored with a SHAKE-256 stream.  It implements only the two methods."""

    name = "toy-stream/v0"
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
        protocol.Verifier(sch, vk, key, k=20, experimental=True)
    v = protocol.Verifier(sch, vk, key, k=20, research=True, experimental=True, clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    profile = v.profile()
    assert not profile.certified and profile.state_bound_bits is None and "no certificate" in profile.statement()
    wrong = protocol.Verifier(sch, vk, key, k=20, research=True, experimental=True, clock=_ticks())
    assert not wrong.run(lambda i: bytes(len(blocks[i]))).accepted
    with pytest.raises(ValueError, match="accepted"):
        wrong.profile()


@pytest.mark.parametrize("name", ["band", "dense", "p3", "p2"])
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
    assert small.codec().name == band.codec_for(4).name == small.name


def test_registry():
    assert set(schemes.REGISTRY) == {"band-chain/d12/v1", "dense-chain/v1", "p3/v1", "p2-16448/v1"}
    assert schemes.DEFAULT == band.BAND12.name
    assert schemes.EXPERIMENTAL == {"p2-16448/v1"} and schemes.DEFAULT not in schemes.EXPERIMENTAL
    assert schemes.get().experimental is None
    b = schemes.get()
    assert isinstance(b, schemes.Band) and b.params == band.SECURE_FIRST and b.name == "band-chain/d12/v1"
    assert schemes.get(segments=30).params.segments == 30
    assert schemes.get("dense-chain/v1").params == dense.POINT_64K
    assert schemes.get("band-chain/d12/v1", params=band.BandParams(16, 64, 12)).params.blocks_per_segment == 16
    with pytest.raises(ValueError, match="band-chain/d4/v1"):
        schemes.get("band-chain/d12/v1", params=band.BandParams(16, 64, 4))
    with pytest.raises(KeyError, match="registered"):
        schemes.get("band-chain/d13/v1")
    with pytest.raises(TypeError):
        protocol.Scheme(band.SECURE_FIRST)
    with pytest.raises(TypeError):
        schemes.Band(dense.POINT_64K)


@pytest.mark.parametrize("name", sorted(schemes.REGISTRY))
def test_pinned_scheme_vectors(name):
    """Each registered scheme's conformance vector (`fixtures/schemes/`, `tests/pous_scheme_vectors.py`): its
    certificate at the registered point, and a small instance's encoded segments and `vk` root in both setups."""
    import pous_scheme_vectors as sv

    pinned = json.loads(sv.path(name).read_text())
    slow = name in sv.SLOW_ENCODE and not os.environ.get("VERITY_POUS_SLOW")
    assert pinned["name"] == name and pinned == sv.vector(name, sv.pinned_blocks(pinned) if slow else None)
    small = schemes.get(name, params=sv.SMALL[name])
    w, salt = sv.small_inputs(name)
    c = sv.pinned_blocks(pinned) if slow else protocol.encode(small, w, salt)
    assert protocol.decode(small, salt, c) == w
    public = protocol.verifier_key(small, protocol.commit(small, w), salt, mode=protocol.PUBLIC, encoded=c)
    assert public.root.hex() == pinned["small"]["root"]


@pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="a whole segment at the registered point: VERITY_POUS_SLOW=1")
def test_band12_segment_at_the_registered_point():
    """One whole 32 MiB band12 segment through the interface equals `band/64kb/segment` (about 25 s)."""
    import pous_vectors

    w, salt = pous_vectors.band_segment_input()
    blocks = schemes.get().encode_segment(salt, 0, w)
    pinned = json.loads((Path(__file__).parent / "fixtures" / "vectors.json").read_text())["band/64kb/segment"]
    assert hashlib.sha256(b"".join(blocks)).hexdigest() == pinned


def test_segment_tags_are_salt_then_u64_segment():
    """`tag_s = salt ‖ u64 s` (Lean `segDom`, `PousDenseReCert.tag`): the 24-byte salt, then the segment index, little
    endian, injective in `s` over all 2^64 segments (`segDom_inj`).  The GPU path builds the same bytes
    (`verity_vllm.engine.pous.codec.segment_tags`; `benchmarks/pous` gates it against these)."""
    from verity_pous.labels import salt_value, tag

    seen = set()
    for s in (0, 1, 2, 511, 512, 2**32, 2**64 - 1):
        t = tag(salt_value(SALT), s).to_bytes(32, "little")
        assert t == SALT + s.to_bytes(8, "little")
        seen.add(t)
    assert len(seen) == 7
    with pytest.raises(ValueError):
        tag(salt_value(SALT), 2**64)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_identical_segments_encode_differently(name):
    """The hard requirement behind `band_meets_14` and `shared_domain_breaks`: segments with the same bytes of `W`
    must encode to different blocks, because each segment is keyed by its own tag."""
    base = SMALL[name]()
    sch = type(base)(base.params.with_segments(3))
    seg = random.Random(5).randbytes(sch.params.segment_bytes)
    c = protocol.encode(sch, seg * 3, SALT)
    for a, b in itertools.combinations(range(3), 2):
        assert all(x != y for x, y in zip(c[a], c[b])), f"segments {a} and {b} share a block"
    vk = protocol.verifier_key(sch, protocol.commit(sch, seg * 3), SALT, w=seg * 3)
    assert len(set(vk.leaves)) == len(vk.leaves)
    assert protocol.decode(sch, SALT, c) == seg * 3


@pytest.mark.parametrize("name", sorted(SMALL))
def test_a_shared_domain_breaks_the_audit(name):
    """`shared_domain_breaks`, replayed on real labels: encode every segment under segment 0's domain, and an
    adversary holding segment 0 alone answers every challenge from it; the audit accepts.  With the scheme's own
    tags the same holding fails."""
    base = SMALL[name]()
    sch = type(base)(base.params.with_segments(3))
    seg = random.Random(6).randbytes(sch.params.segment_bytes)
    b = sch.params.blocks_per_segment
    shared = [sch.encode_segment(SALT, 0, seg) for _ in range(3)]
    vk_shared = VerifierKey.from_segments(sch.params, SALT, shared)
    held = shared[0]
    v = protocol.Verifier(sch, vk_shared, protocol.challenge_key(b"audit", vk_shared), k=40, research=True,
                          experimental=True, clock=_ticks())
    assert v.run(lambda i: held[i % b]).accepted
    own = protocol.encode(sch, seg * 3, SALT)
    vk = VerifierKey.from_segments(sch.params, SALT, own)
    v = protocol.Verifier(sch, vk, protocol.challenge_key(b"audit", vk), k=40, research=True, experimental=True,
                          clock=_ticks())
    assert not v.run(lambda i: own[0][i % b]).accepted


def test_the_14_gb_certificate():
    """`band_meets_14`: one `Meets` for 448 segments (14 GB) at k = 111, the same hypothesis as one segment; other
    segment counts keep the gap."""
    sch = schemes.get(segments=448)
    cert = sch.certificate
    assert (cert.theorem, cert.segments, cert.k, cert.rounds, cert.hypotheses) == ("band_meets_14", 448, 111, 5,
                                                                                    ("ChainExPostFactoG",))
    vk = VerifierKey.from_leaves(sch.params, SALT, [bytes(32)] * sch.params.blocks)
    v = protocol.Verifier(sch, vk, protocol.challenge_key(b"\x00" * 32, vk))
    assert v.gaps == ()
    p = protocol.StorageProfile(sch.name, sch.params.manifest(), sch.params.blocks, sch.params.m, 111, 536_800, cert)
    assert p.certified and p.state_bound_bits == 113_929_658_799
    assert schemes.get(segments=30).certificate.theorem == "band_meets_64"
    v30 = protocol.Verifier(schemes.get(segments=30), _fake_vk(schemes.get(segments=30)), protocol.challenge_key(b"\x00" * 32, vk))
    assert v30.gaps == ("W has 30 segments; band_meets_64 is about 1",)


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
    for name in schemes.REGISTRY:
        cert = schemes.get(name).certificate
        for c in cert.claims if cert else ():
            assert protocol.claim(c).id == c
    for c in protocol.LOCAL_CLAIMS:
        with pytest.raises(KeyError):
            core_claim(c)
        assert "/" not in c and c == c.lower()
    assert protocol.claim("cr/sha-256").kind == "assumption"


def test_p2_is_experimental_and_says_so():
    """P2 (`p2-16448/v1`) is registered but experimental: the verifier refuses it unless asked, never makes it the
    default, and its profile states "abstract-M1 result; instantiation assumption pending Daniel"."""
    sch = schemes.get("p2-16448/v1", segments=128)
    assert sch.experimental == EXPERIMENTAL == "abstract-M1 result; instantiation assumption pending Daniel"
    cert = sch.certificate
    assert (cert.theorem, cert.k, cert.rounds, cert.work, cert.segments, cert.hypotheses) == ("P2MeetsM1Uncond", 88, 0,
                                                                                              1 << 20, 128, ())
    assert {"m1-ideal-permutations", "sms-m1-instantiation", "seq-root"} <= set(cert.claims)
    one = schemes.get("p2-16448/v1")
    vk = VerifierKey.from_leaves(one.params, SALT, [bytes(32)] * one.params.blocks)
    key = protocol.challenge_key(b"\x00" * 32, vk)
    with pytest.raises(ValueError, match="experimental"):
        protocol.Verifier(one, vk, key)
    v = protocol.Verifier(one, vk, key, experimental=True, rtt_allowance_ns=36_800)
    assert v.k == 88 and v.gaps == ("W has 1 segments; P2MeetsM1Uncond is about 128",)
    with pytest.raises(ValueError, match="covers 0"):
        protocol.Verifier(one, vk, key, experimental=True, rtt_allowance_ns=2_400_000)
    p = protocol.StorageProfile(sch.name, sch.params.manifest(), sch.params.blocks, sch.params.m, 88, 536_800, cert,
                                experimental=sch.experimental)
    assert p.certified and p.statement().startswith("EXPERIMENTAL (abstract-M1 result; instantiation assumption pending "
                                                    "Daniel). ")
    assert "P2MeetsM1Uncond" in p.statement() and p.to_json()["experimental"] == EXPERIMENTAL
    assert p.state_bound_bits == (1 << 23) * 16448 * 18 // 19


def test_p2_arithmetic():
    """sq inverts the parity-canonical root, σ is an involution on [0, p), and a block that is not a codeword is refused."""
    from verity_pous.schemes.p2 import _Field, block_keys

    f, rng = _Field(128, 173), random.Random(9)
    for _ in range(300):
        y, k = rng.randrange(f.p), rng.getrandbits(128)
        assert f.sq(f.rho(y)) == y and f.sigma(f.sigma(y, k), k) == y < f.p
    params = P2Params(128, 173, 8, 2)
    assert block_keys(params, SALT, 0, 1) != block_keys(params, SALT, 1, 1) != block_keys(params, SALT, 0, 2)
    sch = schemes.P2(params)
    c = sch.encode_segment(SALT, 1, bytes(range(112)))
    with pytest.raises(ValueError, match="below p"):
        sch.decode_segment(SALT, 1, [(f.p).to_bytes(16, "little")] + c[1:])
    with pytest.raises(ValueError, match="codeword"):
        sch.decode_segment(SALT, 1, [bytes([7]) + bytes(15)] + c[1:])
    assert sch.decode_segment(SALT, 1, c) == bytes(range(112))


def test_p2_keys_are_chacha8_from_a_segment_seed():
    """P2's per-block keys expand where they are used, on the device, from one 32-byte seed per segment (ChaCha8 with
    the block index as nonce), so no key-sized state is stored and the 1.05× space bound holds."""
    from verity_pous.schemes.p2 import chacha_block, expand_keys, segment_seed

    # the ChaCha block function against RFC 8439 §2.3.2 (20 rounds, the same word layout), ChaCha8 against the published
    # zero-key, zero-nonce vector
    assert chacha_block(bytes(range(32)), 1 | 0x09000000 << 32, 0x4A000000, 20)[:16].hex() == "10f1e7e4d13b5915500fdd1fa32071c4"
    assert chacha_block(bytes(32), 0, 0)[:32].hex() == "3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e"
    params = P2Params(16448, 21065, 4)
    seed = segment_seed(SALT, 3)
    assert len(seed) == 32 and seed != segment_seed(SALT, 2)
    k, t = expand_keys(params, seed, 5)
    z = b"".join(chacha_block(seed, i, 5) for i in range(65))
    assert k == int.from_bytes(z[:2056], "little") and t == int.from_bytes(z[2056:4112], "little") % params.p
    assert expand_keys(params, seed, 6) != (k, t)
