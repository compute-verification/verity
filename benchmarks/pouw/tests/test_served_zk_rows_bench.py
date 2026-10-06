"""served_zk's GPU row-commitment benchmark (`served_zk/rows_bench.py`), the host side its gates rest on: the salt stream is
ChaCha20 (RFC 7539's block vector, in sha512.cuh's counter and nonce layout), and the node and leaf heads it hands the
device are the messages merkle's frame-v3-sha512 tree hashes."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from verity.primitives.commitments import merkle as MK
from verity.primitives.commitments import rowleaf as RL
from verity.primitives.commitments.indexed import RangeIndexedDomain

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import rows_bench as RB  # noqa: E402


def test_chacha20_is_rfc7539s_block():
    key = bytes(range(32))
    # RFC 7539 2.3.2: counter 1, nonce 00:00:00:09:00:00:00:4a:00:00:00:00 = state words 12..15 (1, 0x09000000, 0x4a000000, 0)
    out = RB.chacha20_block(key, 1 | 0x09000000 << 32, 0x4A000000)
    assert out[:16].hex() == "10f1e7e4d13b5915500fdd1fa32071c4"
    assert out[-16:].hex() == "b5129cd1de164eb9cbd083e8a2503c4e"


def test_salts_are_three_blocks_a_leaf():
    assert RB.salt(5, 2) == b"".join(RB.chacha20_block(RB.SALT_KEY, 6 + j, 5) for j in range(3))
    assert len(RB.salt(0, 0)) == 192


def test_heads_are_merkles_messages():
    dom = MK.CommitmentDomain(hashlib.sha256(b"t").digest(), -1, RangeIndexedDomain(0, 300), hash="sha512")
    left, right = hashlib.sha512(b"l").digest(), hashlib.sha512(b"r").digest()
    for level, index in ((0, 0), (0, 255), (3, 1), (8, 0)):
        head = RB.node_head(dom, level, index)
        assert hashlib.sha512(head + left + (64).to_bytes(8, "big") + right).digest() == dom.node(level, index, left, right)
        assert len(head) <= RB.SLOT - 4
    v = hashlib.sha512(b"v").digest()
    for rank in (0, 1, 299):
        pre = dom.leaf_prefix(rank, rank, RL.SCHEMA_HM96_SHA512_ROW_SEG, 64)
        assert hashlib.sha512(pre + v).digest() == RL.row_leaf(dom, rank, v, RL.SCHEMA_HM96_SHA512_ROW_SEG)
        assert len(pre) <= RB.SLOT - 4
    slots = RB.slots([b"ab", b"c"])
    assert len(slots) == 2 * RB.SLOT and slots[:6] == b"\x02\0\0\0ab" and slots[RB.SLOT:RB.SLOT + 5] == b"\x01\0\0\0c"
