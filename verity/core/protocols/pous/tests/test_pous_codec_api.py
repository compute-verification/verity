"""The per-segment byte API and the codec seam: everything above the codec works one segment at a time, with any codec."""

from __future__ import annotations

import itertools
import random

import pytest
from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.audit import HonestResponder, UnsignedRounds, challenge_key, run_timed
from verity.core.protocols.pous.codec import Codec, encode_segments, split
from verity.core.protocols.pous.layout import Layout, decode_tensor
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import reference
from verity.core.protocols.pous.scheme import P3, decode_segment, encode, encode_segment
from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash, segment_leaves

PARAMS = Params(8, 64, 3, 4)
PRIMS = reference(PARAMS)
W = random.Random(5).randbytes(PARAMS.w_bytes)
SALT = bytes(range(24))


def test_encode_segment_on_bytes():
    enc = encode(PARAMS, PRIMS, W, SALT)
    segs = list(split(PARAMS, W))
    assert len(segs) == 4 and b"".join(segs) == W
    for s, seg in enumerate(segs):
        blocks = encode_segment(PARAMS, PRIMS, SALT, s, seg)
        assert blocks == [enc.block(s * 8 + v) for v in range(8)]
        assert decode_segment(PARAMS, PRIMS, SALT, s, blocks) == seg
    assert encode(PARAMS, PRIMS, segs, SALT) == enc
    assert list(encode_segments(P3, PARAMS, PRIMS, W, SALT)) == [encode_segment(PARAMS, PRIMS, SALT, s, g)
                                                                  for s, g in enumerate(segs)]
    with pytest.raises(ValueError):
        encode_segment(PARAMS, PRIMS, SALT, 4, segs[0])
    with pytest.raises(ValueError):
        encode_segment(PARAMS, PRIMS, SALT, 0, segs[0][:-1])
    with pytest.raises(ValueError):
        list(split(PARAMS, segs[:3]))


def test_commit_streams_the_same_digest():
    manifest = canonical_json_bytes(PARAMS.manifest())
    whole = tagged_sha512(setup.COMMIT_TAG, len(manifest).to_bytes(8, "big") + manifest + W)
    assert setup.commit(PARAMS, W) == whole == setup.commit(PARAMS, list(split(PARAMS, W)))


def test_verifier_key_without_the_encoding():
    enc = encode(PARAMS, PRIMS, W, SALT)
    built = VerifierKey.build(enc)
    segments = [encode_segment(PARAMS, PRIMS, SALT, s, g) for s, g in enumerate(split(PARAMS, W))]
    leaves = [x for s, blocks in enumerate(segments) for x in segment_leaves(PARAMS, SALT, s, blocks)]
    assert VerifierKey.from_leaves(PARAMS, SALT, leaves) == built
    assert VerifierKey.from_segments(PARAMS, SALT, iter(segments)) == built
    block = enc.block(13)
    assert leaf_hash(PARAMS, SALT, 13, [block[:7], block[7:40], block[40:]]) == leaf_hash(PARAMS, SALT, 13, block)
    commitment = setup.commit(PARAMS, W)
    assert setup.verifier_key(PARAMS, PRIMS, W, commitment, SALT) == built
    assert setup.verifier_key(PARAMS, PRIMS, list(split(PARAMS, W)), commitment, SALT, encoded=segments) == built
    with pytest.raises(ValueError):
        VerifierKey.from_leaves(PARAMS, SALT, leaves[:-1])


def test_layout_of_ranges():
    sizes = [("a", 100), ("b", 1500), ("c", 7)]
    packed = Layout.of(PARAMS, sizes)
    ranges = Layout.of_ranges(PARAMS, [(p.name, p.offset, p.size) for p in reversed(packed.tensors)])
    assert ranges == packed
    gapped = Layout.of_ranges(PARAMS, [("x", 10, 5), ("y", 600, 3)])
    w = gapped.pack({"x": b"12345", "y": b"abc"})
    assert w[:10] == bytes(10) and w[10:15] == b"12345" and w[600:603] == b"abc" and gapped.params.segments == 2
    with pytest.raises(ValueError):
        Layout.of_ranges(PARAMS, [("x", 0, 10), ("y", 5, 10)])


def test_responder_reports_ride_on_answers():
    enc = encode(PARAMS, PRIMS, W, SALT)
    vk = VerifierKey.build(enc)
    responder = HonestResponder(enc)
    result = run_timed(UnsignedRounds(vk, challenge_key(b"audit", vk), k=5, clock=itertools.count(0, 1000).__next__),
                       lambda i: (responder.block(i), {"index": i, "device_ns": 7}))
    assert result.accepted and all(a.report == {"index": a.index, "device_ns": 7} for a in result.answers)


def _xor_encode(params, prims, salt, s, w_seg):
    mb = params.block_bytes
    pad = (salt * (mb // len(salt) + 1))[:mb]
    return [bytes(a ^ b ^ s for a, b in zip(w_seg[v * mb:(v + 1) * mb], pad)) for v in range(params.blocks_per_segment)]


def _xor_decode(params, prims, salt, s, blocks):
    return b"".join(_xor_encode(params, prims, salt, s, b"".join(blocks)))


def test_a_second_codec_plugs_in():
    """A toy one-layer codec (not a scheme) through setup, per-tensor decode and the timed audit: nothing above the
    codec reads P3's structure."""
    toy = Codec("toy-xor", _xor_encode, _xor_decode)
    layout = Layout.of(Params(8, 64, 3), [("t", 700), ("u", 900)])
    params = layout.params
    data = {"t": random.Random(1).randbytes(700), "u": random.Random(2).randbytes(900)}
    w = layout.pack(data)
    commitment = setup.commit(params, w)
    vk = setup.verifier_key(params, None, w, commitment, SALT, codec=toy)
    blocks = [b for seg in encode_segments(toy, params, None, w, SALT) for b in seg]
    assert all(decode_tensor(layout, None, SALT, name, blocks.__getitem__, codec=toy) == data[name] for name in data)
    assert run_timed(UnsignedRounds(vk, challenge_key(b"audit", vk), k=10, clock=itertools.count(0, 1000).__next__), blocks.__getitem__).accepted
