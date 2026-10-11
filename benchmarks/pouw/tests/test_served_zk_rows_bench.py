"""served_zk's GPU row commitments (`served_zk/rows_bench.py`, and the served path's `served_zk/statement.py`), the host side
their gates rest on: each commitment's salts are the OS's, 192 bytes a row, the ones its reference reads, the node and leaf heads it hands the device are the messages merkle's frame-v3-sha512 tree
hashes, and the statement's stages give each weight its own slot of its k's rows."""
from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

from verity.ml.commitments import merkle as MK
from verity.ml.commitments import rowleaf as RL
from verity.ml.commitments.indexed import RangeIndexedDomain

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import rows_bench as RB  # noqa: E402
import statement as ST  # noqa: E402


class _Host:
    """The device calls a `Commit` makes outside its launches, on host memory."""

    def __init__(self):
        self.mem, self.next, self.stream = {}, 4096, None

    def alloc(self, n):
        p, self.next = self.next, self.next + n + 4096
        self.mem[p] = bytearray(n)
        return p

    def put(self, data):
        p = self.alloc(len(data))
        self.mem[p][:] = data
        return p

    def launch(self, *a, **k):
        pass

    def ck(self, fn, dst, src, n, *stream):
        assert fn in ("cuMemcpyHtoD_v2", "cuMemcpyHtoDAsync_v2")
        base = max(p for p in self.mem if p <= dst.value)
        self.mem[base][dst.value - base:dst.value - base + n] = ctypes.string_at(src, n)


def test_salts_are_the_oss_per_row_and_the_references():
    h = _Host()
    cm = RB.Commit(h, h, {}, "t", 0, 5, 32, 32, RB.V2)
    first = cm.salts
    assert len(first) == 192 * 5 and bytes(h.mem[cm.d_salts]) == first
    cm.fresh_salts()
    assert cm.salts != first and bytes(h.mem[cm.d_salts]) == cm.salts
    assert RB.Commit(h, h, {}, "t", 0, 5, 32, 32, RB.V2).salts not in (first, cm.salts)
    want = cm.reference([bytes(32)] * 5)
    assert want["bc"] == b"".join(RB.INST.commit_string(RB.KEY, RL.sha512_row_v2_digest(bytes(32), 256), cm.salts[192 * i:192 * (i + 1)])
                                  for i in range(5))


def test_each_weight_has_its_own_slot_of_its_ks_stage():
    pos, count = ST.slots_of({9: 4096, 3: 14336, 5: 4096, 1: 4096})
    assert pos == {1: (4096, 0), 5: (4096, 1), 9: (4096, 2), 3: (14336, 0)} and count == {4096: 3, 14336: 1}


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
