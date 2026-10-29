"""The generic protocol (`verity.core.protocols.pous.protocol`) over every scheme (`verity.core.protocols.pous.schemes`): one conformance suite for
the lifecycle, the certificates the memory challenger enforces, the storage profile and the claims it cites."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
from pathlib import Path

import pytest

from verity.core.claims import get as core_claim
from verity.core.protocols.pous import band, dense, primitives, protocol, schemes
from verity.core.protocols.pous.codec import Geometry
from verity.core.protocols.pous.layout import Layout
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.schemes.p2 import CHACHA8, CHACHA8_NONCONFORMING, EXPERIMENTAL, P2Params
from verity.core.protocols.pous.schemes.p2 import ROOT_CALL_NS, RTT_ALLOWANCE_CAP_NS
from verity.core.protocols.pous.verifier import VerifierKey

SALT = bytes(range(24))
SMALL = {
    "band": lambda segments=3: schemes.Band(band.BandParams(16, 64, 4, segments)),
    "dense": lambda segments=3: schemes.Dense(dense.DenseParams(8, 64, segments)),
    "p3": lambda segments=2: schemes.P3(Params(8, 64, 3, segments)),
    "p2": lambda segments=2: schemes.P2(P2Params(128, 173, 8, segments)),   # 2^128 − 173 is prime, ≡ 3 mod 4
    "p2v2": lambda segments=2: schemes.P2ChaCha(P2Params(128, 173, 8, segments, CHACHA8)),
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
        protocol.TimedAudit(sch, vk, key, k=20, experimental=True)
    v = protocol.TimedAudit(sch, vk, key, k=20, research=True, experimental=True, clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    profile = v.profile()
    assert not profile.certified and profile.state_bound_bits is None and "no certificate" in profile.statement()
    wrong = protocol.TimedAudit(sch, vk, key, k=20, research=True, experimental=True, clock=_ticks())
    assert not wrong.run(lambda i: bytes(len(blocks[i]))).accepted
    with pytest.raises(ValueError, match="accepted"):
        wrong.profile()


@pytest.mark.parametrize("name", ["band", "dense", "p3", "p2", "p2v2"])
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
    assert set(schemes.REGISTRY) == {"band-chain/d12/v1", "dense-chain/v1", "p3/v1", "p2-16448/v4", "p2-16448/v2"}
    assert schemes.DEFAULT == band.BAND12.name
    assert schemes.EXPERIMENTAL == {"p2-16448/v4", "p2-16448/v2"} and schemes.DEFAULT not in schemes.EXPERIMENTAL
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
    from verity.core.protocols.pous.labels import salt_value, tag

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
    """The hard requirement behind `BandMultiMeetsFamily` and `shared_domain_breaks`: segments with the same bytes of `W`
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
    v = protocol.TimedAudit(sch, vk_shared, protocol.challenge_key(b"audit", vk_shared), k=40, research=True,
                          experimental=True, clock=_ticks())
    assert v.run(lambda i: held[i % b]).accepted
    own = protocol.encode(sch, seg * 3, SALT)
    vk = VerifierKey.from_segments(sch.params, SALT, own)
    v = protocol.TimedAudit(sch, vk, protocol.challenge_key(b"audit", vk), k=40, research=True, experimental=True,
                          clock=_ticks())
    assert not v.run(lambda i: own[0][i % b]).accepted


def test_the_band_certificate_covers_every_segment_count():
    """`Pous.SecurityProofs.BandMultiMeetsFamily`: one `Meets` for the whole store at k = 111, for every segment count
    from 1 to 2^64 (`FAMILY_SEGMENTS`, all that `BandParams` allows), with no hypothesis.  The certificate used to be
    `band_meets_64` at one segment, `band_meets_14` at 448 and `band_meets_family` between, each given
    `ChainExPostFactoG`; `chainExPostFactoG_v6` proves that hypothesis, and the family pin covers every count."""
    for n in (1, 30, 448, 2240, 1 << 64):
        cert = schemes.Band(band.SECURE_FIRST.with_segments(n)).certificate
        assert (cert.theorem, cert.segments, cert.k, cert.rounds, cert.hypotheses) == (
            "Pous.SecurityProofs.BandMultiMeetsFamily", n, 111, 5, ())
    from verity.core.protocols.pous.schemes import band as band_scheme
    assert band_scheme.FAMILY_SEGMENTS == 1 << 64


def test_the_14_gb_certificate():
    """448 segments (14 GB, where the family is also `BandMultiMeets14`) and 30 segments are certified, with no gap."""
    sch = schemes.get(segments=448)
    vk = VerifierKey.from_leaves(sch.params, SALT, [bytes(64)] * sch.params.blocks)
    v = protocol.TimedAudit(sch, vk, protocol.challenge_key(b"\x00" * 32, vk))
    assert v.gaps == ()
    p = protocol.StorageProfile(sch.name, sch.params.manifest(), sch.params.blocks, sch.params.m, 111, 536_800,
                                sch.certificate)
    assert p.certified and p.state_bound_bits == 113_929_658_799
    s30 = schemes.get(segments=30)
    assert protocol.TimedAudit(s30, _fake_vk(s30), protocol.challenge_key(b"\x00" * 32, vk)).gaps == ()


def test_the_family_certificate_covers_a_70_gb_store():
    """A 70 GiB FP8 store (2,240 segments of 32 MiB) is certified whole by `BandMultiMeetsFamily`, as one `Meets` over
    its 1,146,880 blocks."""
    sch = schemes.get(segments=2240)
    cert = sch.certificate
    assert (cert.theorem, cert.segments, cert.k) == ("Pous.SecurityProofs.BandMultiMeetsFamily", 2240, 111)
    assert sch.params.blocks == 2240 * 512


def test_certificates_at_their_points():
    b = schemes.get().certificate
    assert (b.k, b.rounds, b.work, b.call_ns, b.cliff, b.segments) == (111, 5, 1 << 20, 241_500, 12, 1)
    assert (b.theorem, b.hypotheses) == ("Pous.SecurityProofs.BandMultiMeetsFamily", ())
    d = schemes.get("dense-chain/v1").certificate
    assert (d.theorem, d.hypotheses, d.cliff, d.k, d.rounds, d.segments) == (
        "Pous.SecurityProofs.ChainDenseMeets64", (), None, 111, 5, 1)
    assert schemes.get("p3/v1").certificate is None
    assert schemes.Band(band.BandParams(512, 524_288, 4)).certificate is None      # the D = 5 pin needs d ≥ 5
    assert schemes.Band(band.BandParams(512, 524_288, 5)).certificate.cliff == 5
    assert schemes.Band(band.BandParams(256, 524_288, 12)).certificate is None
    with pytest.raises(ValueError, match="cliff"):
        protocol.Certificate(111, 6, 1 << 20, 483_000, "x", cliff=5)
    with pytest.raises(KeyError):
        protocol.Certificate(111, 5, 1 << 20, 483_000, "x", claims=("no-such-claim",))


def test_certificates_name_proved_guarantees():
    """A certificate's `theorem` is the proof `Pous.SecurityProofs.X` it rests on, a guarantee in
    `Security/lean-audit.json`, of the statement `Pous.Guarantees.X` (`Definitions/Pous/Guarantees.lean` there and the
    files under `Definitions/Pous/Guarantees/`), experimental schemes' included."""
    security = Path(__file__).parents[5] / "Security"
    specs = security / "Definitions" / "Pous"
    texts = [(specs / "Guarantees.lean").read_text()] + [f.read_text() for f in sorted((specs / "Guarantees").glob("*.lean"))]
    statements = {x for t in texts for x in re.findall(r"^def (\w+) : Prop", t, re.M)}
    guarantees = json.loads((security / "lean-audit.json").read_text())["guarantees"]
    checked = 0
    for name in schemes.REGISTRY:
        sch = schemes.get(name)
        assert sch.certificates[:1] == (() if sch.certificate is None else (sch.certificate,)), name
        for cert in sch.certificates:
            x = cert.theorem.removeprefix("Pous.SecurityProofs.")
            assert cert.theorem == f"Pous.SecurityProofs.{x}" and x in statements, name
            assert cert.theorem in guarantees, name
            checked += 1
    assert checked == 7


def test_p3_has_no_certificate_because_its_chain_does_not_start_from_leans():
    """P3's operating point has the parameters of `Pous.SecurityProofs.P3ConcreteMeets64` (`n = 220`, `wc = 2384`,
    `B(220, 12)`, one segment) but not its `H`.  Lean's overwrite chain starts from `iv(domP3 tag L v)`: zeros, then
    the tag and `2v + L` in the 512-bit capacity part.  The code's starts from the instantiation's header in the low
    bytes, with the capacity part zero.  So P3 has no certificate; when the two agree, this fails and one is due."""
    sch = schemes.get("p3/v1")
    p = sch.params
    assert (p.n, p.wc, p.k, p.segments) == (220, 2384, 12, 1)
    tag, layer, v = int.from_bytes(SALT + (3).to_bytes(8, "little"), "little"), 1, 5
    lean_h0 = (tag | (2 * v + layer) << 256) << p.m
    h0 = sch.prims.h.initial(tag, layer, v, len(p.parents(v)))
    assert h0 != lean_h0 and h0 >> p.m == 0 and lean_h0 & ((1 << p.m) - 1) == 0
    assert sch.certificate is None


def _fake_vk(sch: protocol.Scheme) -> VerifierKey:
    return VerifierKey.from_leaves(sch.params, SALT, [hashlib.sha512(bytes([i % 251])).digest()
                                                     for i in range(sch.params.blocks)])


def test_the_call_floor_is_the_streamed_chain_count():
    """A 64 KiB `Π₂` call in a chain is 10 rounds of one absorb each: the prefix and a 65,568-byte half are 65,570
    bytes, 483 SHAKE256 blocks, so 4,830 Keccak-f, 241,500 ns at 50 ns.  Counting every round's 482 further squeeze
    blocks in sequence too gives 9,650 (482,500 ns), the 483 µs the certificates used to carry."""
    half = (2 * 524_288 + 512) // 16
    absorb = -(-(primitives.ROUND_PREFIX_BYTES + half) // primitives.SHAKE256_RATE)
    assert (half, absorb, primitives.FEISTEL_ROUNDS, primitives.KECCAK_F_NS) == (65_568, 483, 10, 50)
    assert primitives.pi2_keccak_f(524_288) == primitives.FEISTEL_ROUNDS * absorb == 4_830
    assert primitives.FEISTEL_ROUNDS * (absorb + -(-half // primitives.SHAKE256_RATE) - 1) == 9_650
    assert primitives.pi2_call_ns(524_288) == 241_500
    assert band.SECURE_FIRST.call_ns == dense.POINT_64K.call_ns == 241_500
    for name in ("band-chain/d12/v1", "dense-chain/v1"):
        cert = schemes.get(name).certificate
        assert cert.call_ns == 241_500 and (cert.rounds + 1) * cert.call_ns == 1_449_000
    assert (schemes.get().certificate.cliff + 1) * 241_500 == 3_139_500


def test_a_two_ms_deadline_is_refused_for_five_rounds():
    """Δ + RTT = 2.0 ms fits 8 calls of 241.5 µs, past the certificate's D = 5 (under the old 483 µs it fit 4)."""
    sch = schemes.get()
    vk = _fake_vk(sch)
    k = protocol.challenge_key(b"\x00" * 32, vk)
    with pytest.raises(ValueError, match="fits 8 calls of 241500 ns; the certificate covers 5"):
        protocol.TimedAudit(sch, vk, k, deadline_ns=500_000, rtt_allowance_ns=1_500_000)
    research = protocol.TimedAudit(sch, vk, k, deadline_ns=500_000, rtt_allowance_ns=1_500_000, research=True)
    assert any("covers 5" in g for g in research.gaps) and research.certificate.effective_rounds(research.limit_ns) == 8


def test_the_memory_challenger_enforces_the_certificate():
    """k = 111 is certified only for D = 5 calls of 241.5 µs: Δ + RTT < 1.449 ms, inside the cliff at 3.1395 ms.  A
    smaller k needs a deadline the `D = 2` certificate covers (`k ≥ 97`, under 724.5 µs)."""
    sch = schemes.get()
    vk, key = _fake_vk(sch), b"\x00" * 32
    k = protocol.challenge_key(key, vk)
    assert protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=36_800).k == 111
    assert protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=948_999).gaps == ()
    with pytest.raises(ValueError, match="covers 5"):
        protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=949_000)
    with pytest.raises(ValueError, match="below"):
        protocol.TimedAudit(sch, vk, k, k=110, rtt_allowance_ns=224_500)
    assert protocol.TimedAudit(sch, vk, k, k=110).certificate.theorem == "Pous.SecurityProofs.BandMultiMeetsFamilyD2"
    uncertified = protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=2_639_499, research=True)
    assert any("covers 5" in g for g in uncertified.gaps)
    with pytest.raises(ValueError, match="cliff"):
        protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=2_639_500, research=True)
    with pytest.raises(ValueError, match="parameters"):
        protocol.TimedAudit(schemes.get(segments=2), vk, k)


def test_the_band_has_certificates_at_fewer_rounds():
    """`BandMultiMeetsFamilyD2` (`k = 97` at `D = 2`, every `d ≥ 2`), `BandMultiMeetsFamilyD1` (`k = 93` at `D = 1`,
    every `d ≥ 1`) and `BandMultiMeetsFamilyD0` (`k = 89` at `D = 0`, every `d`) follow the default `D = 5`
    certificate, each for every segment count.  A certificate at `D` covers `Δ + RTT` up to `(D + 1)·call_ns`
    exclusive: 1.449 ms, 724.5 µs, 483 µs and 241.5 µs."""
    def rows(d: int) -> list[tuple[int, int, str, int | None]]:
        return [(c.rounds, c.k, c.theorem.removeprefix("Pous.SecurityProofs."), c.cliff)
                for c in schemes.Band(band.BandParams(512, 524_288, d)).certificates]

    d2, d1, d0 = (2, 97, "BandMultiMeetsFamilyD2"), (1, 93, "BandMultiMeetsFamilyD1"), (0, 89, "BandMultiMeetsFamilyD0")
    assert rows(12) == [(5, 111, "BandMultiMeetsFamily", 12), (*d2, 12), (*d1, 12), (*d0, 12)]
    assert rows(5) == [(5, 111, "BandMultiMeetsFamily", 5), (*d2, 5), (*d1, 5), (*d0, 5)]
    assert rows(4) == [(*d2, 4), (*d1, 4), (*d0, 4)] and rows(2) == [(*d2, 2), (*d1, 2), (*d0, 2)]
    assert rows(1) == [(*d1, 1), (*d0, 1)]
    assert schemes.Band(band.BandParams(256, 524_288, 12)).certificates == ()
    assert [c.segments for c in schemes.get(segments=448).certificates] == [448] * 4
    certs = schemes.get().certificates
    assert [(c.rounds + 1) * c.call_ns for c in certs] == [1_449_000, 724_500, 483_000, 241_500]
    for c in certs:
        tau = (c.rounds + 1) * c.call_ns
        assert c.refusals(c.k, tau - 1) == () and c.refusals(c.k, tau) != () and c.refusals(c.k - 1, 0) != ()
        assert (c.work, c.call_ns, c.hypotheses, c.claims) == (1 << 20, 241_500, (), certs[0].claims)
    assert schemes.get("dense-chain/v1").certificates == (schemes.get("dense-chain/v1").certificate,)


def test_the_memory_challenger_audits_at_two_or_one_rounds_when_asked():
    """With `k` unset the memory challenger stays at `k = 111` (`D = 5`), even under a deadline that fits fewer rounds.
    Asked for `k = 97` it audits under `BandMultiMeetsFamilyD2` within 724.5 µs, and for `k = 93` under
    `BandMultiMeetsFamilyD1` within 483 µs, which the default 500 µs deadline overruns.  A band of in-degree 2 or 1,
    which has no `D = 5` certificate, defaults to its own."""
    sch = schemes.get()
    vk = _fake_vk(sch)
    key = protocol.challenge_key(b"\x00" * 32, vk)
    assert protocol.TimedAudit(sch, vk, key).certificate.theorem == "Pous.SecurityProofs.BandMultiMeetsFamily"
    assert protocol.TimedAudit(sch, vk, key, deadline_ns=300_000).k == 111
    two = protocol.TimedAudit(sch, vk, key, k=97)
    assert (two.k, two.certificate.theorem, two.gaps) == (97, "Pous.SecurityProofs.BandMultiMeetsFamilyD2", ())
    assert protocol.TimedAudit(sch, vk, key, k=100, rtt_allowance_ns=224_499).certificate.rounds == 2
    with pytest.raises(ValueError, match="below the certificate's 111; under Pous.SecurityProofs.BandMultiMeetsFamilyD2"
                                         ": .* fits 3 calls of 241500 ns; the certificate covers 2"):
        protocol.TimedAudit(sch, vk, key, k=97, rtt_allowance_ns=224_500)
    with pytest.raises(ValueError, match="covers 1, so Δ \\+ RTT must stay under 483000 ns"):
        protocol.TimedAudit(sch, vk, key, k=93)
    one = protocol.TimedAudit(sch, vk, key, k=93, deadline_ns=482_999)
    assert (one.k, one.certificate.theorem, one.gaps) == (93, "Pous.SecurityProofs.BandMultiMeetsFamilyD1", ())
    with pytest.raises(ValueError, match="below the certificate's 93"):
        protocol.TimedAudit(sch, vk, key, k=92, deadline_ns=482_999)
    research = protocol.TimedAudit(sch, vk, key, k=92, deadline_ns=482_999, research=True)
    assert research.certificate.rounds == 5 and research.gaps == ("k = 92 is below the certificate's 111",)
    for d, k, deadline in ((4, 97, 500_000), (2, 97, 500_000), (1, 93, 482_999)):
        b = schemes.Band(band.BandParams(512, 524_288, d))
        v = protocol.TimedAudit(b, _fake_vk(b), key, deadline_ns=deadline)
        assert (v.k, v.gaps, v.certificate.cliff) == (k, (), d)
    b1 = schemes.Band(band.BandParams(512, 524_288, 1))
    with pytest.raises(ValueError, match="past the cliff at 1"):
        protocol.TimedAudit(b1, _fake_vk(b1), key)


def test_a_two_round_audit_gives_its_profile():
    """An accepted `k = 97` audit within 724.5 µs is certified by `BandMultiMeetsFamilyD2`, with the default's state
    bound (the same `rho`)."""
    sch = schemes.get()
    blocks = [bytes(sch.params.block_bytes)] * sch.params.blocks
    vk = VerifierKey.from_segments(sch.params, SALT, [blocks])
    v = protocol.TimedAudit(sch, vk, protocol.challenge_key(b"audit secret", vk), k=97, rtt_allowance_ns=224_499,
                          clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    p = v.profile()
    assert p.certified and p.k == 97 and p.effective_rounds == 2 and p.state_bound_bits == 254_307_274
    assert "passes 97 challenges within 724,499 ns" in p.statement()
    assert "(Pous.SecurityProofs.BandMultiMeetsFamilyD2; under ideal-permutation, " in p.statement()
    assert p.to_json()["certificate"]["rounds"] == 2


def test_the_dense_certificate_has_no_cliff_but_the_same_rounds():
    sch = schemes.get("dense-chain/v1")
    vk = _fake_vk(sch)
    k = protocol.challenge_key(b"\x00" * 32, vk)
    with pytest.raises(ValueError, match="covers 5"):
        protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=949_000)
    far = protocol.TimedAudit(sch, vk, k, rtt_allowance_ns=10_000_000, research=True)
    assert not any("cliff" in g for g in far.gaps)


def test_certified_profile_of_one_band12_segment():
    """An accepted audit at the frozen point gives the certified profile.  The audit reads only `vk` and the answers,
    so random blocks stand in for an encoding here (encoding a 32 MiB segment takes about 25 s in Python)."""
    sch = schemes.get()
    rng = random.Random(7)
    blocks = [rng.randbytes(sch.params.block_bytes) for _ in range(sch.params.blocks)]
    vk = VerifierKey.from_segments(sch.params, SALT, [blocks])
    v = protocol.TimedAudit(sch, vk, protocol.challenge_key(b"audit secret", vk), rtt_allowance_ns=36_800, clock=_ticks())
    assert v.run(protocol.responder(sch, blocks.__getitem__)).accepted
    p = v.profile()
    assert p.certified and p.gaps == () and p.effective_rounds == 2
    assert p.state_bound_bits == 254_307_274 == (512 * 524_288 * 18) // 19
    assert "(Pous.SecurityProofs.BandMultiMeetsFamily; under ideal-permutation, " in p.statement()
    assert "given" not in p.statement()
    j = p.to_json()
    assert j["schema"] == "verity-pous/storage-profile/v1" and j["certified"] and j["certificate"]["k"] == 111
    late = protocol.TimedAudit(sch, vk, protocol.challenge_key(b"audit secret", vk), clock=_ticks(600_000))
    assert not late.run(protocol.responder(sch, blocks.__getitem__)).accepted


def test_many_segments_are_a_gap_not_a_refusal():
    """A segment count its certificate doesn't state is a gap, not a refusal: `ChainDenseMeets64` is about one segment,
    and no guarantee covers the dense chain scheme at more.  The band's family pin covers every count, so two
    band segments have none."""
    band2 = schemes.get(segments=2)
    assert protocol.TimedAudit(band2, _fake_vk(band2), protocol.challenge_key(b"\x00" * 32, _fake_vk(band2))).gaps == ()
    sch = schemes.get("dense-chain/v1", segments=2)
    v = protocol.TimedAudit(sch, _fake_vk(sch), protocol.challenge_key(b"\x00" * 32, _fake_vk(sch)))
    assert v.gaps == ("W has 2 segments; Pous.SecurityProofs.ChainDenseMeets64 is about 1",)


def test_claims_resolve_and_local_ones_are_not_in_core():
    for name in schemes.REGISTRY:
        cert = schemes.get(name).certificate
        for c in cert.claims if cert else ():
            assert protocol.claim(c).id == c
    for c in protocol.LOCAL_CLAIMS:
        with pytest.raises(KeyError):
            core_claim(c)
        assert "/" not in c and c == c.lower()
    assert protocol.claim("cr/sha-512").kind == "assumption"
    assert protocol.claim("trusted-host").kind == "assumption"


def test_p2_is_experimental_and_says_so():
    """P2 (`p2-16448/v4`, and `p2-16448/v2` with ChaCha8 keys) is registered but experimental: the memory challenger
    refuses it unless asked, never makes it the default, and its profile states why.  The certificate is the pinned
    `P2SlackFamilyUncond` at the spec's k = 105, and the manifest names the payload profile.  Its call is
    `ROOT_CALL_NS`, half the fastest cooperating root: the memory challenger accepts spec v1.1's cap,
    `Δ + RTT = 854 µs`, and refuses a µs more, and v1's 1 ms."""
    for name in ("p2-16448/v4", "p2-16448/v2"):
        sch = schemes.get(name, segments=8)
        v2 = name.endswith("v2")
        assert sch.experimental == (f"{EXPERIMENTAL}; {CHACHA8_NONCONFORMING}" if v2 else EXPERIMENTAL)
        assert sch.params.profile == sch.params.manifest()["profile"] == "p2-2054B"
        assert ("expander" in sch.params.manifest()) is not v2
        cert = sch.certificate
        assert (cert.theorem, cert.k, cert.rounds, cert.work, cert.call_ns, cert.segments, cert.hypotheses) == (
            "Pous.SecurityProofs.P2SlackFamilyUncond", 105, 0, 1 << 20, ROOT_CALL_NS, 8, ())
        assert {"m1-ideal-permutations", "sms-m1-instantiation", "seq-root", "cr/sha-512", "trusted-host"} <= set(
            cert.claims) and not any("sha-256" in c for c in cert.claims)
        one = schemes.get(name)
        vk = VerifierKey.from_leaves(one.params, SALT, [bytes(64)] * one.params.blocks)
        key = protocol.challenge_key(b"\x00" * 32, vk)
        with pytest.raises(ValueError, match="experimental"):
            protocol.TimedAudit(one, vk, key)
        v = protocol.TimedAudit(one, vk, key, experimental=True, rtt_allowance_ns=36_800)
        assert v.k == 105 and v.gaps == ("W has 1 segments; Pous.SecurityProofs.P2SlackFamilyUncond is about 8",)
        capped = protocol.TimedAudit(one, vk, key, experimental=True, rtt_allowance_ns=RTT_ALLOWANCE_CAP_NS)
        assert capped.limit_ns == 854_000 and capped.k == 105 and capped.certificate == cert
        for rtt in (RTT_ALLOWANCE_CAP_NS + 1_000, 1_000_000):
            with pytest.raises(ValueError, match="covers 0, so Δ \\+ RTT must stay under 854029 ns"):
                protocol.TimedAudit(one, vk, key, experimental=True, rtt_allowance_ns=rtt)
        p = protocol.StorageProfile(sch.name, sch.params.manifest(), sch.params.blocks, sch.params.m, 105, 536_800, cert,
                                    experimental=sch.experimental)
        assert p.certified and p.statement().startswith(f"EXPERIMENTAL ({sch.experimental}). ")
        assert "P2SlackFamilyUncond" in p.statement() and p.to_json()["experimental"] == sch.experimental
        assert p.state_bound_bits == (1 << 19) * 16448 * 18 // 19
    with pytest.raises(ValueError, match="chacha8"):
        schemes.get("p2-16448/v4", params=P2Params(16448, 21065, 1 << 16, 1, CHACHA8))
    with pytest.raises(ValueError, match="expander"):
        P2Params(128, 173, 8, 1, "chacha20")


def test_p2_family_certificate_is_the_proved_block_range():
    """`p2.family_certificate` is `P2SlackFamilyUncond`'s `λ = 128` row at k = 105, the registered certificate's every
    field but its segment pin, for a v4 store of 2^8 to 2^23 blocks at the registered width and prime in any number of
    segments; there is none off that range, at another width, or for v2's ChaCha8 keys."""
    from verity.core.protocols.pous.schemes.p2 import CERT_FAMILY_BLOCKS, family_certificate

    reg = schemes.get("p2-16448/v4", segments=8).certificate
    assert CERT_FAMILY_BLOCKS == (1 << 8, 1 << 23)
    for bps, segments in ((256, 1), (128, 2), (1 << 16, 1), (1 << 16, 8), (1 << 16, 128)):
        cert = family_certificate(P2Params(16448, 21065, bps, segments))
        assert cert.to_json() == reg.to_json() | {"segments": None}, (bps, segments)
        assert cert.theorem == "Pous.SecurityProofs.P2SlackFamilyUncond" and cert.claims == reg.claims
    for q in (P2Params(16448, 21065, 255, 1), P2Params(16448, 21065, 1 << 16, 129), P2Params(128, 173, 256, 1),
              P2Params(16448, 21065, 256, 1, CHACHA8)):
        assert family_certificate(q) is None, q
    assert schemes.P2(P2Params(16448, 21065, 256, 1)).certificate is None


def test_p2_wide_certificate_is_period_held_at():
    """`p2.wide_certificate` is `Pous.Period.SecurityProofs.periodHeldAt`, the family's audit (k = 105, no root within
    `Δ + RTT`, `2^20` queries, `⌊ρ·|C|⌋` at `1% + 2^-128`) for a v4 store of 2^19 to 2^27 blocks at the registered width
    and prime in any number of segments, Demo B's 1038 segments among them, resting on `P2_CLAIMS` and a claim for each
    of `periodHeldAt`'s premises at `N`; there is none off that range, at another width, or for v2's ChaCha8 keys."""
    from verity.core.protocols.pous.schemes.p2 import CERT_WIDE_BLOCKS, P2_CLAIMS, WIDE_CLAIMS, family_certificate
    from verity.core.protocols.pous.schemes.p2 import wide_certificate

    fam = family_certificate(P2Params(16448, 21065, 256, 1)).to_json()
    assert CERT_WIDE_BLOCKS == (1 << 19, 1 << 27)
    assert WIDE_CLAIMS == (*P2_CLAIMS, "p2-ring-round-secure-wide", "pous-ring-device-at", "statistical-hiding")
    for bps, segments in ((1 << 16, 8), (1 << 12, 128), (1 << 16, 129), (1 << 16, 1038), (1 << 16, 1 << 11)):
        cert = wide_certificate(P2Params(16448, 21065, bps, segments))
        assert cert.to_json() == fam | {"theorem": "Pous.Period.SecurityProofs.periodHeldAt",
                                        "claims": list(WIDE_CLAIMS)}, (bps, segments)
        assert all(protocol.claim(c).id == c for c in cert.claims)
    for q in (P2Params(16448, 21065, (1 << 16) - 1, 8), P2Params(16448, 21065, 1 << 16, (1 << 11) + 1),
              P2Params(1024, 105, 1 << 16, 1038), P2Params(16448, 21065, 1 << 16, 1038, CHACHA8)):
        assert wide_certificate(q) is None, q


def test_p2_wide320_certificate_is_the_320_read_window():
    """`p2.wide320_certificate` is `Pous.SecurityProofs.ObligationWide320`: the wide certificate's audit and claims at
    k = 320, with `δ + 2^eps_log2` exactly `2^-20`, over the same 2^19 to 2^27 blocks; none elsewhere."""
    from fractions import Fraction

    from verity.core.protocols.pous.schemes.p2 import CERT_K_WIDE320, WIDE_CLAIMS, wide320_certificate, wide_certificate

    assert CERT_K_WIDE320 == 320
    for bps, segments in ((1 << 16, 8), (1 << 12, 128), (1 << 16, 1038), (1 << 16, 1 << 11)):
        q = P2Params(16448, 21065, bps, segments)
        cert = wide320_certificate(q)
        assert cert.to_json() == wide_certificate(q).to_json() | {
            "k": 320, "theorem": "Pous.SecurityProofs.ObligationWide320", "delta": f"{2 ** 108 - 1}/{2 ** 128}"}
        assert cert.delta + Fraction(1, 2 ** -cert.eps_log2) == Fraction(1, 1 << 20)
        assert cert.claims == WIDE_CLAIMS and cert.refusals(319, 0) == ("k = 319 is below the certificate's 320",)
    for q in (P2Params(16448, 21065, (1 << 16) - 1, 8), P2Params(16448, 21065, 1 << 16, (1 << 11) + 1),
              P2Params(1024, 105, 1 << 16, 1038), P2Params(16448, 21065, 1 << 16, 1038, CHACHA8)):
        assert wide320_certificate(q) is None, q


def test_p2_encode_blocks_are_slices_of_the_segment():
    """`p2.encode_blocks` from block `j0` of a segment gives those blocks of `encode_segment`, and refuses payloads that
    are not whole blocks inside the segment."""
    from verity.core.protocols.pous.schemes.p2 import encode_blocks, encode_segment

    params = P2Params(128, 173, 8, 2)
    w = bytes(range(112))
    c = encode_segment(params, None, SALT, 1, w)
    assert encode_blocks(params, SALT, 1, 0, w) == c and encode_blocks(params, SALT, 1, 8, b"") == []
    assert encode_blocks(params, SALT, 1, 3, w[3 * 14:6 * 14]) == c[3:6] != encode_blocks(params, SALT, 0, 3, w[42:84])
    for j0, payloads in ((0, w[:13]), (6, w[:42]), (-1, w[:14])):
        with pytest.raises(ValueError, match="whole blocks"):
            encode_blocks(params, SALT, 1, j0, payloads)


@pytest.mark.parametrize("expander", ["sha512-ctr-v4", CHACHA8])
def test_p2_arithmetic(expander):
    """sq inverts the parity-canonical root, σ is an involution on [0, p), blocks and payloads are big-endian integers
    (the spec's wire format), and a block that is not a codeword is refused."""
    from verity.core.protocols.pous.schemes.p2 import _Field, block_keys

    f, rng = _Field(128, 173), random.Random(9)
    for _ in range(300):
        y, k = rng.randrange(f.p), rng.getrandbits(128)
        assert f.sq(f.rho(y)) == y and f.sigma(f.sigma(y, k), k) == y < f.p
    params = P2Params(128, 173, 8, 2, expander)
    assert block_keys(params, SALT, 0, 1) != block_keys(params, SALT, 1, 1) != block_keys(params, SALT, 0, 2)
    sch = (schemes.P2ChaCha if expander == CHACHA8 else schemes.P2)(params)
    w_seg = bytes(range(112))
    c = sch.encode_segment(SALT, 1, w_seg)
    for j in (0, 5):
        k, t = block_keys(params, SALT, 1, j)
        m = (f.sq(f.sigma(f.sq(int.from_bytes(c[j], "big")), k)) - t) % f.p
        assert m == int.from_bytes(w_seg[14 * j:14 * (j + 1)], "big")
    with pytest.raises(ValueError, match="below p"):
        sch.decode_segment(SALT, 1, [(f.p).to_bytes(16, "big")] + c[1:])
    with pytest.raises(ValueError, match="codeword"):
        sch.decode_segment(SALT, 1, [bytes([7]) + bytes(15)] + c[1:])
    assert sch.decode_segment(SALT, 1, c) == w_seg


@pytest.mark.parametrize("w,c0", [(24, 20001), (24, 40001), (24, 65533), (128, 173)])
def test_p2_square_is_exact_for_every_accepted_c0(w, c0):
    """At a small width the largest accepted c0 leaves the two folds above 2p; the squaring is still a² mod p."""
    from verity.core.protocols.pous.schemes.p2 import _Field

    f, rng = _Field(w, c0), random.Random(w * c0)
    for a in [0, 1, f.p - 1, *(rng.randrange(f.p) for _ in range(2000))]:
        assert f.square(a) == a * a % f.p


def test_p2_call_is_half_the_cooperating_floor_root_and_the_cap_the_largest_us_under_it():
    """Spec v1.1 (§12): `ROOT_CALL_NS` is half the cooperating root in ns, rounded down,
    from the sources' own numbers: the red team's floor of 2,060.845 cycles at 5.7 GHz per squaring (`regate.py` at
    w = 16,448), 16,486 squaring-equivalents per root, and the measured 8-core speedup 2,716.7 / 778.5.  The
    RTT cap is the largest whole µs with `Δ + cap < ROOT_CALL_NS`, so the certificate
    admits the cap and refuses every whole µs past it; what it admits past the cap is under 1 µs."""
    from fractions import Fraction

    half = Fraction(16_486) * Fraction("2060.845") / Fraction("5.7") * Fraction("778.5") / Fraction("2716.7") / 2
    assert ROOT_CALL_NS == int(half) == 854_029
    delta = 500_000
    assert RTT_ALLOWANCE_CAP_NS == max(us * 1_000 for us in range(1_000) if delta + us * 1_000 < ROOT_CALL_NS) == 354_000
    one = schemes.get("p2-16448/v4")
    vk = VerifierKey.from_leaves(one.params, SALT, [bytes(64)] * one.params.blocks)
    key = protocol.challenge_key(b"\x00" * 32, vk)
    edge = ROOT_CALL_NS - 1 - delta
    assert edge - RTT_ALLOWANCE_CAP_NS == 28
    assert one.certificate.refusals(105, delta + edge) == ()
    with pytest.raises(ValueError, match="must stay under 854029 ns"):
        protocol.TimedAudit(one, vk, key, deadline_ns=delta, experimental=True, rtt_allowance_ns=edge + 1)
    cooperating_root = 2 * half
    assert cooperating_root / (delta + RTT_ALLOWANCE_CAP_NS) >= 2 > cooperating_root / (delta + RTT_ALLOWANCE_CAP_NS + 1_000)


def test_p2_refuses_an_rtt_past_the_cap_or_a_delta_other_than_the_spec_s_even_inside_the_certificate():
    """Spec v1.1 sets Δ = 0.5 ms and caps the measured RTT allowance at 354 µs ("Sequential reveal and deadlines", §6).
    The certificate reads only `Δ + RTT`, so Δ = 0 with RTT = 854,028 ns, the certificate's edge 28 ns past the cap, or
    a Δ above 0.5 ms are inside it; the verifier refuses each, and `research=True` runs one with the refusal as a gap."""
    one = schemes.get("p2-16448/v4")
    vk = VerifierKey.from_leaves(one.params, SALT, [bytes(64)] * one.params.blocks)
    key = protocol.challenge_key(b"\x00" * 32, vk)
    for delta, rtt, why in ((0, ROOT_CALL_NS - 1, "Δ = 500000 ns, not 0"),
                            (500_000, ROOT_CALL_NS - 1 - 500_000, "354028 ns is past the spec's cap of 354000 ns"),
                            (500_028, RTT_ALLOWANCE_CAP_NS, "Δ = 500000 ns, not 500028")):
        assert one.certificate.refusals(105, delta + rtt) == ()
        with pytest.raises(ValueError, match=re.escape(why)):
            protocol.TimedAudit(one, vk, key, deadline_ns=delta, rtt_allowance_ns=rtt, experimental=True)
    research = protocol.TimedAudit(one, vk, key, deadline_ns=0, rtt_allowance_ns=ROOT_CALL_NS - 1, research=True,
                                         experimental=True)
    assert research.gaps[0] == "the spec's operating point sets Δ = 500000 ns, not 0"
    assert protocol.TimedAudit(one, vk, key, rtt_allowance_ns=RTT_ALLOWANCE_CAP_NS, experimental=True).limit_ns == 854_000


def test_p2_v4_keys_are_the_specs():
    """v4's keys are the spec's §4 counter-mode SHA-512 streams: given the spec's prefix, `sha512_keys` gives every pinned
    K and t of the spec's `vectors.json` (`fixtures/p2-v4-keys.json`, copied with that file's hash), at w = 64, 96, 128
    and 16,448.  The protocol's prefix frames its 24-byte salt where the spec frames the model commitment and its 32-byte
    setup salt, and nothing else differs; both are zero-padded to whole 128-byte blocks."""
    from verity.ml.randomness import frame

    from verity.core.protocols.pous.schemes.p2 import KEYS_TAG, PROTOCOL_ID, expand_prefix, sha512_keys, stream

    doc = json.loads((Path(__file__).parent / "fixtures" / "p2-v4-keys.json").read_text())
    assert doc["source_sha512"] == "959911fd371111fe20c4cee6fc532ff205bbf377962c73ac539a4d2de9b22121287e6e996d9ae4bfbef9f26fcd67ac78620873702c8b2899bc41a21bdd76e853"
    for v in doc["vectors"]:
        params = P2Params(v["w"], v["c0"], 1 << 16, 8)
        spec = (KEYS_TAG + frame("p2/v4") + frame(f"p2-p{v['w']}-c{v['c0']}")
                + frame(bytes.fromhex(v["model_commitment"])) + frame(bytes.fromhex(v["setup_salt"])))
        spec += bytes(-len(spec) % 128)
        k, t = sha512_keys(params, spec, v["segment"], v["block"])
        n = params.block_bytes
        assert (k.to_bytes(n, "big").hex(), t.to_bytes(n, "big").hex()) == (v["K"], v["t"]), v["id"]
    assert len(doc["vectors"]) == 4 and {v["w"] for v in doc["vectors"]} == {64, 96, 128, 16448}
    params = P2Params(16448, 21065, 4)
    assert (KEYS_TAG, PROTOCOL_ID, params.prime_id) == (b"verity/pous/p2/keys/v4\0", "p2/v4", "p2-p16448-c21065")
    prefix = expand_prefix(params, SALT)
    body = KEYS_TAG + frame(PROTOCOL_ID) + frame(params.prime_id) + frame(SALT)
    assert len(body) == 95 and prefix == body + bytes(33) and len(prefix) == 128
    base = prefix + (3).to_bytes(8, "big") + (9).to_bytes(8, "big")
    assert stream(base, b"K", 130) == b"".join(hashlib.sha512(base + b"K" + i.to_bytes(4, "big")).digest() for i in range(3))[:130]
    assert len(base) % 128 + 1 + 4 + 17 <= 128                       # a chunk is one compression after the midstate
    with pytest.raises(ValueError, match="24 bytes"):
        expand_prefix(params, bytes(32))


def test_p2_v2_keys_are_chacha8_from_a_segment_seed():
    """v2's per-block keys expand where they are used, on the device, from one 32-byte seed per segment (ChaCha8 with
    the block index as nonce), so no key-sized state is stored and the 1.05× space bound holds."""
    from verity.core.protocols.pous.schemes.p2 import chacha_block, chacha_keys, segment_seed

    # the ChaCha block function against RFC 8439 §2.3.2 (20 rounds, the same word layout), ChaCha8 against the published
    # zero-key, zero-nonce vector
    assert chacha_block(bytes(range(32)), 1 | 0x09000000 << 32, 0x4A000000, 20)[:16].hex() == "10f1e7e4d13b5915500fdd1fa32071c4"
    assert chacha_block(bytes(32), 0, 0)[:32].hex() == "3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e"
    params = P2Params(16448, 21065, 4, 1, CHACHA8)
    seed = segment_seed(SALT, 3)
    assert len(seed) == 32 and seed != segment_seed(SALT, 2)
    k, t = chacha_keys(params, seed, 5)
    z = b"".join(chacha_block(seed, i, 5) for i in range(65))
    assert k == int.from_bytes(z[:2056], "big") and t == int.from_bytes(z[2056:4112], "big") < params.p
    assert chacha_keys(params, seed, 6) != (k, t)


def test_p2_tweak_is_rejection_sampled():
    """A tweak candidate at or above p is skipped for the next one, as the spec requires; it is never reduced by one
    subtraction of p, which would give the values below c0 twice the weight of the rest.  Both expanders."""
    from verity.core.protocols.pous.schemes import p2

    params = P2Params(128, 173, 4, 1, CHACHA8)
    assert p2.below(params.p, iter([params.p, params.p + 5, (1 << 128) - 1, 7])) == 7
    assert p2.below(params.p, iter([params.p - 1])) == params.p - 1
    real = p2.chacha_block

    def first_candidate_high(key: bytes, counter: int, nonce: int, rounds: int = 8) -> bytes:
        out = real(key, counter, nonce, rounds)
        return b"\xff" * 32 + out[32:] if counter == 0 else out     # K and the first t candidate are 2^128 − 1 ≥ p

    try:
        p2.chacha_block = first_candidate_high
        k, t = p2.chacha_keys(params, bytes(32), 0)
    finally:
        p2.chacha_block = real
    assert k == (1 << 128) - 1 and t == int.from_bytes(real(bytes(32), 0, 0)[32:48], "big") < params.p
    assert t != (1 << 128) - 1 - params.p

    # v4 at w = 24, c0 = 65,533, where a candidate is ≥ p about once in 256: the first block whose first one is
    small = P2Params(24, 65533, 4)
    prefix = p2.expand_prefix(small, SALT)
    j = next(j for j in range(10_000) if int.from_bytes(hashlib.sha512(
        prefix + bytes(8) + j.to_bytes(8, "big") + b"t" + bytes(4)).digest()[:3], "big") >= small.p)
    words = hashlib.sha512(prefix + bytes(8) + j.to_bytes(8, "big") + b"t" + bytes(4)).digest()[:9]
    first, second, third = (int.from_bytes(words[i:i + 3], "big") for i in (0, 3, 6))
    assert p2.sha512_keys(small, prefix, 0, j)[1] == (second if second < small.p else third) != first - small.p
