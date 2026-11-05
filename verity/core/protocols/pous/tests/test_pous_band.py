"""The band scheme (`verity.core.protocols.pous.band`, Lean's `chainBand`): round trips, the construction recomputed from the Lean
definitions' text, its costs, the secure-first parameters, the cliff, and the whole flow through setup, per-tensor
decode and the timed audit with the parameters' own codec."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest

from verity.core.protocols.pous import band, dense, setup
from verity.core.protocols.pous.audit import HonestResponder, UnsignedRounds, challenge_key, run_timed, simulate
from verity.core.protocols.pous.codec import encode_segments, split
from verity.core.protocols.pous.labels import salt_value, tag
from verity.core.protocols.pous.layout import Layout, decode_tensor
from verity.core.protocols.pous.oracle import KeyQuery, run
from verity.core.protocols.pous.primitives import Primitives, key_program
from verity.core.protocols.pous.scheme import Encoding
from verity.core.protocols.pous.verifier import VerifierKey

PARAMS = band.BandParams(16, 64, 4, 3)
SALT = bytes(range(24))


def _w(params, seed):
    return random.Random(seed).randbytes(params.w_bytes)


@pytest.mark.parametrize("backend", ["reference", "ideal"])
def test_round_trip(backend):
    prims = band.reference(PARAMS) if backend == "reference" else band.ideal(PARAMS, b"band-seed")
    w = _w(PARAMS, 1)
    for s, seg in enumerate(split(PARAMS, w)):
        blocks = PARAMS.codec.encode_segment(PARAMS, prims, SALT, s, seg)
        assert len(blocks) == 16 and PARAMS.codec.decode_segment(PARAMS, prims, SALT, s, blocks) == seg
    blocks = [b for seg in encode_segments(PARAMS.codec, PARAMS, prims, w, SALT) for b in seg]
    assert len(set(blocks)) == len(blocks)


def test_construction_is_lean_chain_band():
    """C_v = r of the pad call of the overwrite chain from h_0 = 0^ℓ ‖ tag_s ‖ u256(v), absorbing C_{v-1}, ...,
    C_{max(0, v-d)} (Lean `labelG`, `absorbedG`, `parentsDesc (bandDAG B d)`), xor W_v."""
    prims = band.reference(PARAMS)
    mb, d, s = PARAMS.m // 8, PARAMS.d, 2
    seg = list(split(PARAMS, _w(PARAMS, 2)))[s]
    tag_s = tag(salt_value(SALT), s).to_bytes(32, "little")
    pad = b"\x01" + bytes(mb - 2) + b"\x80"
    c: list[bytes] = []
    for v in range(16):
        h = bytes(mb) + tag_s + v.to_bytes(32, "little")
        for b in [c[u] for u in range(v - 1, max(0, v - d) - 1, -1)] + [pad]:
            out = prims.h.pi2.forward(int.from_bytes(b + h, "little")).to_bytes(2 * mb + 64, "little")
            r, h = out[:mb], out[mb:]
        c.append(bytes(x ^ y for x, y in zip(r, seg[v * mb:(v + 1) * mb])))
    assert PARAMS.codec.encode_segment(PARAMS, prims, SALT, s, seg) == c


def test_costs():
    """Label v costs min(v, d) + 1 calls; decode runs the rows in parallel: d + 1 rounds."""
    params = band.BandParams(16, 64, 4)
    prims = band.reference(params)
    t = tag(salt_value(SALT), 0)
    w = [random.Random(3).getrandbits(64) for _ in range(16)]
    calls = sum(min(v, 4) + 1 for v in range(16))
    c, cost = run(dense.encode_segment_prog(params, prims.h, t, w), prims)
    assert cost.depth == cost.work == calls == 70
    back, cost = run(dense.decode_segment_prog(params, prims.h, t, c), prims)
    assert back == w and cost.depth == 5 and cost.work == calls


def test_wide_band_is_dense():
    """With d ≥ B − 1 the band DAG is the complete DAG: the same encoding as dense."""
    wide, full = band.BandParams(8, 64, 7), dense.DenseParams(8, 64)
    w = _w(wide, 5)
    assert band.codec_for(7).encode_segment(wide, band.reference(wide), SALT, 0, w) == \
        dense.DENSE.encode_segment(full, dense.reference(full), SALT, 0, w)


def test_secure_first_parameters():
    p = band.SECURE_FIRST
    assert (p.blocks_per_segment, p.m, p.d, p.segment_bytes) == (512, 524_288, 12, 1 << 25)
    assert (p.challenges, p.certified_rounds, p.call_ns) == (111, 5, 241_500) and p.codec is band.BAND12
    assert list(p.parents(0)) == [] and list(p.parents(5)) == [0, 1, 2, 3, 4] and list(p.parents(100)) == list(range(88, 100))
    assert sum(len(p.parents(v)) + 1 for v in range(512)) == 6_578
    with pytest.raises(ValueError):
        band.BAND12.encode_segment(band.BandParams(16, 64, 4), None, SALT, 0, bytes(128))


def test_cliff_is_enforced():
    """The effective D (calls of at least 241.5 µs within Δ + RTT) must not exceed d; 13 calls need 3.1395 ms."""
    p = band.SECURE_FIRST
    vk = VerifierKey(p, SALT, bytes(32), (bytes(32),) * p.blocks)
    key = challenge_key(b"audit", vk)
    UnsignedRounds(vk, key, deadline_ns=500_000, rtt_allowance_ns=2_639_499)
    assert band.effective_rounds(p, 500_000, 2_639_499) == 12 and band.effective_rounds(p, 500_000, 2_639_500) == 13
    with pytest.raises(ValueError, match="cliff"):
        UnsignedRounds(vk, key, deadline_ns=500_000, rtt_allowance_ns=2_639_500)
    UnsignedRounds(VerifierKey(dense.POINT_64K, SALT, bytes(32), (bytes(32),) * 512), key, rtt_allowance_ns=10 ** 9)


class GapResponder:
    """The gap attack (`band_gap_attack`): hold every label except the positive multiples of d + 1; answer a dropped
    label by its own row, whose parents are all held, in min(v, d) + 1 = d + 1 rounds."""

    def __init__(self, params, prims, salt, w, blocks):
        self.params, self.prims, self.salt, self.w = params, prims, salt, w
        n, d = params.blocks_per_segment, params.d
        self.held = {i: b for i, b in enumerate(blocks) if (i % n) == 0 or (i % n) % (d + 1)}

    def answer(self, j, i):
        if i in self.held:
            return int.from_bytes(self.held[i], "little")
        n, mb = self.params.blocks_per_segment, self.params.block_bytes
        s, v = divmod(i, n)
        labels = tuple((u, int.from_bytes(self.held[s * n + u], "little")) for u in self.params.parents(v))
        key = yield from key_program(self.prims.h, KeyQuery(tag(salt_value(self.salt), s), 0, v, labels))
        return key ^ int.from_bytes(self.w[i * mb:(i + 1) * mb], "little")


def test_gap_attack_at_the_cliff():
    """With D′ = d + 1 the holding with every (d + 1)-th label dropped passes the audit; with D′ = d it does not."""
    params = band.BandParams(16, 64, 4, 2)
    prims = band.reference(params)
    w = _w(params, 7)
    blocks = [b for seg in encode_segments(params.codec, params, prims, w, SALT) for b in seg]
    vk = VerifierKey.from_segments(params, SALT, [blocks[:16], blocks[16:]])
    gap = GapResponder(params, prims, SALT, w, blocks)
    assert len(gap.held) == 32 - 2 * (15 // 5)
    key = challenge_key(b"audit", vk)
    assert simulate(vk, gap, prims, key, k=200, rounds=5).accepted
    result = simulate(vk, gap, prims, key, k=200, rounds=4)
    assert not result.accepted and all(a.correct for a in result.answers)


def test_end_to_end_with_the_parameters_codec():
    """Setup, per-tensor decode, the timed audit (k = 111) and the simulator, every codec taken from the parameters."""
    layout = Layout.of(band.BandParams(16, 64, 4), [("embed", 200), ("norm", 3), ("proj", 300)])
    params = layout.params
    prims = band.reference(params)
    data = {p.name: random.Random(p.size).randbytes(p.size) for p in layout.tensors}
    w = layout.pack(data)
    commitment = setup.commit(params, w)
    salt = setup.draw_salt(b"verifier-secret", commitment)
    server = [params.codec.encode_segment(params, prims, salt, s, g) for s, g in enumerate(split(params, w))]
    vk = setup.verifier_key(params, prims, w, commitment, salt)
    assert vk == VerifierKey.from_segments(params, salt, server)
    blocks = [b for seg in server for b in seg]
    assert all(decode_tensor(layout, prims, salt, name, blocks.__getitem__) == data[name] for name in data)
    key = challenge_key(b"audit", vk)
    timed = UnsignedRounds(vk, key, clock=itertools.count(0, 1000).__next__)
    assert timed.k == 111 and run_timed(timed, blocks.__getitem__).accepted
    enc = Encoding(params, salt, tuple(int.from_bytes(b, "little") for b in blocks))
    assert simulate(vk, HonestResponder(enc), prims, key).accepted


class ToyF:
    """Lean `toyF`: `x ↦ a·x + b mod 2^w`."""

    name = "lean-toy-f"

    def __init__(self, width):
        self.width = width

    def forward(self, x):
        return (11400714819323198485 * x + 12345 + self.width) % (1 << self.width)

    def inverse(self, y):
        a = 11400714819323198485 % (1 << self.width)
        return ((y - 12345 - self.width) * pow(a, -1, 1 << self.width)) % (1 << self.width)


LEAN = json.loads((Path(__file__).parent / "fixtures" / "lean_band_vectors.json").read_text())


@pytest.mark.parametrize("case", LEAN["parents"], ids=lambda c: f"B{c['B']}-d{c['d']}")
def test_parents_match_lean(case):
    params = band.BandParams(case["B"], 8, case["d"])
    assert [list(reversed(params.parents(v))) for v in range(case["B"])] == case["desc"]


@pytest.mark.parametrize("case", LEAN["labels"], ids=lambda c: f"B{c['B']}-d{c['d']}-l{c['l']}")
def test_labels_match_lean(case):
    from test_pous_dense import SmallCapacity

    b, d, ell = case["B"], case["d"], case["l"]
    params = band.BandParams(b, ell, d)
    h = SmallCapacity(params, ToyF(2 * ell + 4))
    prims = Primitives(h, None)
    salt, w = int(case["salt"]), int(case["w"])
    blocks = [(w >> (i * ell)) & ((1 << ell) - 1) for i in range(b)]
    c, _ = run(dense.encode_segment_prog(params, h, salt, blocks), prims)
    assert [str(x) for x in c] == case["blocks"]
    back, _ = run(dense.decode_segment_prog(params, h, salt, c), prims)
    assert str(sum(x << (i * ell) for i, x in enumerate(back))) == case["decode"] == case["w"]


@pytest.mark.slow
@pytest.mark.skipif(not __import__("os").environ.get("VERITY_POUS_SLOW"), reason="about a minute: set VERITY_POUS_SLOW=1")
def test_secure_first_segment():
    """One whole 32 MiB secure-first segment: its pinned digest and a bit-exact decode."""
    import hashlib

    import pous_vectors

    pinned = json.loads((Path(__file__).parent / "fixtures" / "vectors.json").read_text())
    p, prims = band.SECURE_FIRST, band.reference(band.SECURE_FIRST)
    w, salt = pous_vectors.band_segment_input()
    blocks = p.codec.encode_segment(p, prims, salt, 0, w)
    assert hashlib.sha256(b"".join(blocks)).hexdigest() == pinned["band/64kb/segment"]
    assert p.codec.decode_segment(p, prims, salt, 0, blocks) == w
