"""Milestone 1 of the public encoder (POUS `docs/public-encoder/protocol.md`): the server encodes and sends `C*` segment
by segment; the verifier decodes it, checks `commit(Dec(C*))` against the commitment, and builds `vk` from the bytes it
decoded.  Tests 1–7 are the spec's, on the band codec (the secure-first default) and on dense; the rest are negative
controls, the setup flag, and the codec-generic path."""

from __future__ import annotations

import inspect
import itertools
import json
import random
from pathlib import Path

import pytest

from verity.core.protocols.pous import band, dense, setup
from verity.core.protocols.pous.audit import UnsignedRounds, challenge_key, run_timed
from verity.core.protocols.pous.codec import Codec, split
from verity.core.protocols.pous.labels import salt_value, tag
from verity.core.protocols.pous.oracle import KeyQuery, run
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import Primitives, key_program, reference
from verity.core.protocols.pous.scheme import P3
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, segment_leaves

LEAN = json.loads((Path(__file__).parent / "fixtures" / "lean_dense_vectors.json").read_text())
LEAN_BAND = json.loads((Path(__file__).parent / "fixtures" / "lean_band_vectors.json").read_text())
SCHEMES = {"band": band.BandParams(16, 64, 4, 3), "dense": dense.DenseParams(8, 64, 3)}
PARAMS = SCHEMES["band"]


class World:
    """A committed W, the verifier's salt and the honest server's C* under one scheme."""

    def __init__(self, params):
        self.params = params
        self.prims = dense.reference(params)
        self.w = random.Random(21).randbytes(params.w_bytes)
        self.commitment = setup.commit(params, self.w)
        self.salt = setup.draw_salt(b"verifier-secret", self.commitment)
        self.honest = self.server()

    def server(self, w=None, salt=None):
        w, salt = w or self.w, salt or self.salt
        return [self.params.codec.encode_segment(self.params, self.prims, salt, s, seg)
                for s, seg in enumerate(split(self.params, w))]

    def accept(self, encoded, **kw):
        return setup.verifier_key_from_encoding(self.params, self.prims, self.commitment, self.salt, encoded, **kw)


WORLDS = {name: World(p) for name, p in SCHEMES.items()}
DEFAULT = WORLDS["band"]
PRIMS, W, COMMITMENT, SALT, HONEST = DEFAULT.prims, DEFAULT.w, DEFAULT.commitment, DEFAULT.salt, DEFAULT.honest


def _accept(encoded, **kw):
    return DEFAULT.accept(encoded, **kw)


def test_default_is_the_band():
    assert band.SECURE_FIRST.codec is band.BAND12 and PARAMS.codec is band.codec_for(4)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_1_honest(scheme):
    x = WORLDS[scheme]
    vk = x.accept(x.honest)
    reference_vk = setup.verifier_key(x.params, x.prims, x.w, x.commitment, x.salt)
    assert vk == reference_vk and vk.root == reference_vk.root
    blocks = [b for seg in x.honest for b in seg]
    assert run_timed(UnsignedRounds(vk, challenge_key(b"audit", vk), clock=itertools.count(0, 1000).__next__), blocks.__getitem__).accepted


def test_1_honest_details():
    vk = _accept(HONEST)
    reference_vk = setup.verifier_key(PARAMS, PRIMS, W, COMMITMENT, SALT)
    assert vk == reference_vk and vk.root == reference_vk.root
    assert setup.check_server_root(vk, MerkleTree([x for s, seg in enumerate(HONEST)
                                                   for x in segment_leaves(PARAMS, SALT, s, seg)]).root)
    blocks = [b for seg in HONEST for b in seg]
    assert run_timed(UnsignedRounds(vk, challenge_key(b"audit", vk), clock=itertools.count(0, 1000).__next__), blocks.__getitem__).accepted


@pytest.mark.parametrize("scheme", SCHEMES)
def test_2_one_flipped_byte_anywhere(scheme):
    x = WORLDS[scheme]
    for s in range(x.params.segments):
        for j in range(x.params.blocks_per_segment):
            for byte in (0, x.params.block_bytes - 1):
                bad = [list(seg) for seg in x.honest]
                block = bytearray(bad[s][j])
                block[byte] ^= 1 << (s + j) % 8
                bad[s][j] = bytes(block)
                with pytest.raises(ValueError, match="commitment"):
                    x.accept(bad)


def _corrupt_and_continue(x: World, s: int, j: int) -> list[bytes]:
    """Segment `s` with label `j` set to zeros and every later label encoded honestly on top of it."""
    p = x.params
    mb, t = p.block_bytes, tag(salt_value(x.salt), s)
    seg = list(split(p, x.w))[s]
    w = [int.from_bytes(seg[i * mb:(i + 1) * mb], "little") for i in range(p.blocks_per_segment)]
    c: list[int] = []
    for i in range(p.blocks_per_segment):
        if i == j:
            c.append(0)
            continue
        key, _ = run(key_program(x.prims.h, KeyQuery(t, 0, i, tuple((u, c[u]) for u in p.parents(i)))), x.prims)
        c.append(key ^ w[i])
    return [v.to_bytes(mb, "little") for v in c]


@pytest.mark.parametrize("scheme, s, j", [("band", 0, 0), ("band", 1, 3), ("band", 2, 15), ("dense", 0, 0),
                                          ("dense", 1, 3), ("dense", 2, 7)])
def test_3_corrupt_and_continue(scheme, s, j):
    x = WORLDS[scheme]
    bad = [list(seg) for seg in x.honest]
    bad[s] = _corrupt_and_continue(x, s, j)
    mb = x.params.block_bytes
    decoded = x.params.codec.decode_segment(x.params, x.prims, x.salt, s, bad[s])
    original = list(split(x.params, x.w))[s]
    differs = [i for i in range(x.params.blocks_per_segment)
               if decoded[i * mb:(i + 1) * mb] != original[i * mb:(i + 1) * mb]]
    assert differs == [j], "the corruption decodes to W' ≠ W at block j only"
    with pytest.raises(ValueError, match="commitment"):
        x.accept(bad)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_enc_dec(scheme):
    """`EncDec` (POUS `PubEncoderModel.lean`): every C re-encodes from its own decode, so `commit(Dec(C*)) = com_W`
    forces `C* = Enc(W)`.  Lean proves it for dense (`chainDense_encDec`) and the band (`chainBand_encDec`)."""
    x = WORLDS[scheme]
    p, rng = x.params, random.Random(9)
    for s in range(p.segments):
        c = [rng.randbytes(p.block_bytes) for _ in range(p.blocks_per_segment)]
        w = p.codec.decode_segment(p, x.prims, x.salt, s, c)
        assert p.codec.encode_segment(p, x.prims, x.salt, s, w) == c


@pytest.mark.parametrize("case", LEAN_BAND["encdec"], ids=lambda c: f"B{c['B']}-d{c['d']}-l{c['l']}")
def test_enc_dec_matches_lean(case):
    """Lean-evaluated encode after decode on codes that are not encodings (`labelG_dec`): the decode matches Lean's `W'`
    and re-encodes to the code."""
    from test_pous_dense import SmallCapacity, ToyF

    b, d, ell = case["B"], case["d"], case["l"]
    params = band.BandParams(b, ell, d)
    h = SmallCapacity(params, ToyF(2 * ell + 4))
    prims = Primitives(h, None)
    code = [int(x) for x in case["code"]]
    w, _ = run(dense.decode_segment_prog(params, h, int(case["salt"]), code), prims)
    assert str(sum(x << (i * ell) for i, x in enumerate(w))) == case["w"]
    again, _ = run(dense.encode_segment_prog(params, h, int(case["salt"]), w), prims)
    assert [str(x) for x in again] == case["reencode"] == case["code"]


def test_4_wrong_sizes():
    missing_block = [list(seg) for seg in HONEST]
    missing_block[1] = missing_block[1][:-1]
    extra_block = [list(seg) for seg in HONEST]
    extra_block[0] = extra_block[0] + [bytes(PARAMS.block_bytes)]
    short_block = [list(seg) for seg in HONEST]
    short_block[2][4] = short_block[2][4][:-1]
    long_block = [list(seg) for seg in HONEST]
    long_block[2][4] = long_block[2][4] + b"\x00"
    for bad in (missing_block, extra_block, short_block, long_block, HONEST[:-1], HONEST + [HONEST[0]], [],
                [[None] * PARAMS.blocks_per_segment] + HONEST[1:]):
        with pytest.raises(ValueError):
            _accept(bad)


def test_5_server_data_unused():
    """No parameter takes a root, a leaf list or a manifest from the server; the root is the decoded labels'."""
    assert set(inspect.signature(setup.verifier_key_from_encoding).parameters) == {
        "params", "prims", "commitment", "salt", "encoded", "codec"}

    class Package(list):
        root = b"\xff" * 32
        leaves = [b"\xee" * 32] * PARAMS.blocks

    vk = _accept(Package(HONEST))
    decoded = [x for s, seg in enumerate(HONEST) for x in segment_leaves(PARAMS, SALT, s, seg)]
    assert vk.leaves == tuple(decoded) and vk.root == MerkleTree(decoded).root != Package.root


def test_6_no_encoder(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the verifier called an encoder")

    reference_root = setup.verifier_key(PARAMS, PRIMS, W, COMMITMENT, SALT, encoded=HONEST).root
    patched = Codec(PARAMS.codec.name, boom, PARAMS.codec.decode_segment)
    monkeypatch.setattr(band, "codec_for", lambda d: patched)
    monkeypatch.setattr(dense, "encode_segment", boom)
    monkeypatch.setattr(dense, "encode_segment_prog", boom)
    assert PARAMS.codec.encode_segment is boom
    assert _accept(HONEST).root == reference_root
    dense_world = WORLDS["dense"]
    monkeypatch.setattr(dense, "DENSE", Codec(dense.DENSE.name, boom, dense.decode_segment))
    assert dense_world.accept(dense_world.honest).root == setup.verifier_key(
        dense_world.params, dense_world.prims, dense_world.w, dense_world.commitment, dense_world.salt,
        encoded=dense_world.honest).root




def _lean_codec(t: int) -> Codec:
    """The dense decoder under the Lean fixture's own 256-bit salt `t` (the fixture's salts do not fit the 24-byte
    salt of the byte API); the encoder refuses."""
    def decode(params, prims, salt, s, blocks):
        c = [int.from_bytes(b, "little") for b in blocks]
        w, _ = run(dense.decode_segment_prog(params, prims.h, t, c), prims)
        return b"".join(x.to_bytes(params.block_bytes, "little") for x in w)

    def encode(*args):
        raise AssertionError("the verifier called an encoder")

    return Codec("lean-toy-dense", encode, decode)


@pytest.mark.parametrize("group, chain", [("band", "small"), ("real", "real"), ("small", "small")])
def test_7_known_answer_vectors(group, chain):
    """The Lean-evaluated encodings with whole-byte blocks, through the verifier path: accepted against the
    commitment to the fixture's W, rejected against any other W.  `band` is `chainBand`'s (`d = 1`, where the band
    already cuts row 2's oldest parent); `real` and `small` are `chainDense`'s."""
    from test_pous_dense import SmallCapacity, ToyF

    cases = [dict(c, c=4) for c in LEAN_BAND["labels"]] if group == "band" else LEAN[group]
    cases = [c for c in cases if c["l"] % 8 == 0]
    assert cases
    for case in cases:
        params = (band.BandParams(case["B"], case["l"], case["d"]) if group == "band"
                  else dense.DenseParams(case["B"], case["l"]))
        cls = dense.DenseChain if chain == "real" else SmallCapacity
        prims = Primitives(cls(params, ToyF(2 * case["l"] + case["c"])), None)
        mb = params.block_bytes
        encoded = [[int(x).to_bytes(mb, "little") for x in case["blocks"]]]
        w = int(case["w"]).to_bytes(params.w_bytes, "little")
        codec = _lean_codec(int(case["salt"]))
        vk = setup.verifier_key_from_encoding(params, prims, setup.commit(params, w), SALT, encoded, codec)
        assert vk.leaves == tuple(segment_leaves(params, SALT, 0, encoded[0]))
        other = bytes(b ^ 1 for b in w)
        with pytest.raises(ValueError, match="commitment"):
            setup.verifier_key_from_encoding(params, prims, setup.commit(params, other), SALT, encoded, codec)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_negative_control_any_wrong_codeword(scheme):
    """A server submitting any C* with Dec(C*) ≠ W is rejected: random labels, the honest encoding of another W, or
    of W under another salt."""
    x = WORLDS[scheme]
    p = x.params
    rng = random.Random(4)
    noise = [[rng.randbytes(p.block_bytes) for _ in range(p.blocks_per_segment)] for _ in range(p.segments)]
    for bad in (noise, x.server(w=bytes(b ^ 0x80 for b in x.w)), x.server(salt=bytes(24))):
        with pytest.raises(ValueError, match="commitment"):
            x.accept(bad)


def test_setup_flag(monkeypatch):
    public = setup.verifier_setup(PARAMS, PRIMS, COMMITMENT, SALT, encoded=HONEST)
    private = setup.verifier_setup(PARAMS, PRIMS, COMMITMENT, SALT, w=W, public_encoder=False)
    assert setup.PUBLIC_ENCODER and public == private
    with pytest.raises(ValueError):
        setup.verifier_setup(PARAMS, PRIMS, COMMITMENT, SALT, w=W)
    monkeypatch.setattr(setup, "PUBLIC_ENCODER", False)
    assert setup.verifier_setup(PARAMS, PRIMS, COMMITMENT, SALT, w=W) == private
    with pytest.raises(ValueError):
        setup.verifier_setup(PARAMS, PRIMS, COMMITMENT, SALT, encoded=HONEST)


def test_codec_generic_p3():
    """The same path with P3's codec (a bijection, though its encode-after-decode lemma is not in Lean)."""
    params = Params(8, 64, 3, 2)
    prims = reference(params)
    w = random.Random(9).randbytes(params.w_bytes)
    commitment = setup.commit(params, w)
    encoded = [P3.encode_segment(params, prims, SALT, s, seg) for s, seg in enumerate(split(params, w))]
    vk = setup.verifier_key_from_encoding(params, prims, commitment, SALT, encoded)
    assert vk == setup.verifier_key(params, prims, w, commitment, SALT)
    encoded[1][2] = bytes(len(encoded[1][2]))
    with pytest.raises(ValueError):
        setup.verifier_key_from_encoding(params, prims, commitment, SALT, encoded)


def test_holds_one_segment_at_a_time():
    """The verifier reads C* lazily: a generator of segments, consumed in order."""
    seen = []

    def stream():
        for s, seg in enumerate(HONEST):
            seen.append(s)
            yield iter(seg)

    assert _accept(stream()) == VerifierKey.from_segments(PARAMS, SALT, HONEST) and seen == [0, 1, 2]
