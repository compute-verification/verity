"""Milestone 1 of the public encoder (POUS `docs/public-encoder/protocol.md`): the server encodes and sends `C*` segment
by segment; the verifier decodes it, checks `commit(Dec(C*))` against the commitment, and builds `vk` from the bytes it
decoded.  Tests 1–7 are the spec's; the rest are negative controls, the setup flag, and the codec-generic path."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path

import pytest

from verity_pous import dense, setup
from verity_pous.audit import TimedVerifier, challenge_key, run_timed
from verity_pous.codec import Codec, split
from verity_pous.labels import salt_value, tag
from verity_pous.oracle import KeyQuery, run
from verity_pous.params import Params
from verity_pous.primitives import Primitives, key_program, reference
from verity_pous.scheme import P3
from verity_pous.verifier import MerkleTree, VerifierKey, segment_leaves

PARAMS = dense.DenseParams(8, 64, 3)
PRIMS = dense.reference(PARAMS)
W = random.Random(21).randbytes(PARAMS.w_bytes)
COMMITMENT = setup.commit(PARAMS, W)
SALT = setup.draw_salt(b"verifier-secret", COMMITMENT)


def _server(params=PARAMS, prims=PRIMS, w=W, salt=SALT):
    """The honest server's C*, segment by segment."""
    return [params.codec.encode_segment(params, prims, salt, s, seg) for s, seg in enumerate(split(params, w))]


HONEST = _server()


def _accept(encoded, **kw):
    return setup.verifier_key_from_encoding(PARAMS, PRIMS, COMMITMENT, SALT, encoded, **kw)


def test_1_honest():
    vk = _accept(HONEST)
    reference_vk = setup.verifier_key(PARAMS, PRIMS, W, COMMITMENT, SALT)
    assert vk == reference_vk and vk.root == reference_vk.root
    assert setup.check_server_root(vk, MerkleTree([x for s, seg in enumerate(HONEST)
                                                   for x in segment_leaves(PARAMS, SALT, s, seg)]).root)
    blocks = [b for seg in HONEST for b in seg]
    assert run_timed(TimedVerifier(vk, challenge_key(b"audit", vk)), blocks.__getitem__).accepted


def test_2_one_flipped_byte_anywhere():
    for s in range(PARAMS.segments):
        for j in range(PARAMS.blocks_per_segment):
            for byte in (0, PARAMS.block_bytes - 1):
                bad = [list(seg) for seg in HONEST]
                block = bytearray(bad[s][j])
                block[byte] ^= 1 << (s + j) % 8
                bad[s][j] = bytes(block)
                with pytest.raises(ValueError, match="commitment"):
                    _accept(bad)


def _corrupt_and_continue(s: int, j: int) -> list[bytes]:
    """Segment `s` with block `j` set to zeros and every later block encoded honestly on top of it."""
    mb, t = PARAMS.block_bytes, tag(salt_value(SALT), s)
    seg = list(split(PARAMS, W))[s]
    w = [int.from_bytes(seg[i * mb:(i + 1) * mb], "little") for i in range(PARAMS.blocks_per_segment)]
    c: list[int] = []
    for i in range(PARAMS.blocks_per_segment):
        if i == j:
            c.append(0)
            continue
        key, _ = run(key_program(PRIMS.h, KeyQuery(t, 0, i, tuple((u, c[u]) for u in range(i)))), PRIMS)
        c.append(key ^ w[i])
    return [x.to_bytes(mb, "little") for x in c]


@pytest.mark.parametrize("s, j", [(0, 0), (1, 3), (2, 7)])
def test_3_corrupt_and_continue(s, j):
    bad = [list(seg) for seg in HONEST]
    bad[s] = _corrupt_and_continue(s, j)
    mb = PARAMS.block_bytes
    decoded = dense.decode_segment(PARAMS, PRIMS, SALT, s, bad[s])
    original = list(split(PARAMS, W))[s]
    differs = [i for i in range(PARAMS.blocks_per_segment) if decoded[i * mb:(i + 1) * mb] != original[i * mb:(i + 1) * mb]]
    assert differs == [j], "the corruption decodes to W' ≠ W at block j only"
    with pytest.raises(ValueError, match="commitment"):
        _accept(bad)


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
    """No parameter takes a root, a leaf list or a manifest from the server; the root is the decoded blocks'."""
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

    monkeypatch.setattr(dense, "DENSE", Codec(dense.DENSE.name, boom, dense.decode_segment))
    monkeypatch.setattr(dense, "encode_segment", boom)
    monkeypatch.setattr(dense, "encode_segment_prog", boom)
    assert PARAMS.codec.encode_segment is boom
    vk = _accept(HONEST)
    assert vk.root == setup.verifier_key(PARAMS, PRIMS, W, COMMITMENT, SALT, encoded=HONEST).root


LEAN = json.loads((Path(__file__).parent / "fixtures" / "lean_dense_vectors.json").read_text())


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


@pytest.mark.parametrize("group, chain", [("real", "real"), ("small", "small")])
def test_7_known_answer_vectors(group, chain):
    """The Lean-evaluated encodings with whole-byte blocks, through the verifier path: accepted against the
    commitment to the fixture's W, rejected against any other W."""
    from test_pous_dense import SmallCapacity, ToyF

    cases = [c for c in LEAN[group] if c["l"] % 8 == 0]
    assert cases
    for case in cases:
        params = dense.DenseParams(case["B"], case["l"])
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


def test_negative_control_any_wrong_codeword():
    """A server submitting any C* with Dec(C*) ≠ W is rejected: random blocks, or the honest encoding of another W."""
    rng = random.Random(4)
    noise = [[rng.randbytes(PARAMS.block_bytes) for _ in range(PARAMS.blocks_per_segment)] for _ in range(PARAMS.segments)]
    other_w = bytes(b ^ 0x80 for b in W)
    for bad in (noise, _server(w=other_w), _server(salt=bytes(24))):
        with pytest.raises(ValueError, match="commitment"):
            _accept(bad)


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
