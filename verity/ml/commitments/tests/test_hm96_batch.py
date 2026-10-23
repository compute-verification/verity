"""`hm96.batch` against the plain reference, and `ml.crypto.chacha20`'s keystream against RFC 8439's block vector."""

import random

import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments.hm96 import batch
from verity.ml.crypto import chacha20


@pytest.mark.parametrize("inst", [hm96.SHA256, hm96.SHA512], ids=lambda i: i.name)
@pytest.mark.parametrize("n", [0, 1, 7])
def test_tree_leaves_are_the_references(inst, n):
    rng = random.Random(n)
    for key in (inst.default_key, inst.key_from_label(b"another key")):
        xs = [rng.randbytes(inst.digest_bytes) for _ in range(n)]
        ys = [rng.randbytes(inst.salt_bytes) for _ in range(n)]
        assert batch.masks(inst, key, b"".join(ys)) == b"".join(inst.mask(key, y) for y in ys)
        assert batch.tree_leaves(inst, key, b"".join(xs), b"".join(ys)) == [
            inst.tree_leaf(key, inst.commit_string(key, x, y)) for x, y in zip(xs, ys)]


def test_tree_leaves_refuse_unmatched_items():
    inst = hm96.SHA512
    with pytest.raises(ValueError, match="not as many"):
        batch.tree_leaves(inst, inst.default_key, bytes(64), bytes(2 * inst.salt_bytes))
    with pytest.raises(ValueError, match="whole"):
        batch.masks(inst, inst.default_key, bytes(inst.salt_bytes + 1))


def test_keystream_is_rfc8439s_block():
    """RFC 8439 section 2.3.2: block 1 under key 00..1f and nonce 000000090000004a00000000, its words 12-15 read as a 64-bit
    counter and a 64-bit nonce."""
    out = chacha20.keystream(bytes(range(32)), 0x4A000000, 1 | 0x09000000 << 32, 1)
    assert out.hex() == ("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
                         "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")


def test_keystream_blocks_are_consecutive_and_refuse_out_of_range():
    key = bytes(range(32, 64))
    many = chacha20.keystream(key, 5, 10, 4)
    assert many == b"".join(chacha20.keystream(key, 5, 10 + i, 1) for i in range(4))
    assert chacha20.keystream(key, 5, 10, 0) == b""
    for bad in ((key[:16], 0, 0, 1), (key, 1 << 64, 0, 1), (key, 0, (1 << 64) - 1, 2), (key, 0, -1, 1)):
        with pytest.raises(ValueError):
            chacha20.keystream(*bad)
