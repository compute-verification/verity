"""`lead.record`: the pass's record keeps calls in order, each call's A root before its words, and is published once,
before any sample."""
from __future__ import annotations

import hashlib

import pytest
from verity_vllm.protocols.pouw.lead import record as LR
from verity_vllm.protocols.pouw.lead.feed import Call

SCHEME = "lead-blc4a-t64-rot"


def record() -> LR.PassRecord:
    return LR.PassRecord(SCHEME, 0, 1, 3, b"\x05" * 64, b"\x06" * 64, kernel="lead-h200/abc")


def test_the_record_publishes_its_calls_in_order():
    rec = record()
    calls = [Call(3, 0, 0, 1, 2048, 3072), Call(3, 1, 1, 1, 8192, 2048)]
    rec.a_root(calls[0], b"\x20" * 64)
    rec.words(calls[0], b"\x10" * 64)
    rec.a_root(calls[1], b"\x21" * 64)
    rec.words(calls[1], b"\x11" * 64)
    pc = rec.publish()
    sha = lambda b: hashlib.sha512(b).digest()  # noqa: E731
    hdr = lambda c: b"".join(v.to_bytes(8, "big") for v in (c.pass_index, c.index, c.linear, c.m, c.k, c.n))  # noqa: E731
    assert LR.header(calls[1]) == hdr(calls[1])
    assert rec.a_roots == [b"\x20" * 64, b"\x21" * 64]
    assert pc.activation_roots == sha(b"\x20" * 64 + b"\x21" * 64)
    assert pc.transcript_root == sha(b"\x10" * 64 + b"\x11" * 64)
    assert pc.records_root == sha(hdr(calls[0]) + hdr(calls[1]))
    assert (pc.units, pc.tiles) == (2, 1 * 384 * 74 + 1 * 256 * 293)
    assert (pc.scheme, pc.rank, pc.world, pc.pass_index, pc.salt, pc.kernel, pc.controls) == \
        (SCHEME, 0, 1, 3, b"\x05" * 64, "lead-h200/abc", ())
    assert pc.to_json()["activation_roots"] == pc.activation_roots.hex()
    assert rec.commitment() is pc


@pytest.mark.parametrize("m, k, n, want", [(1, 2048, 3072, 1 * 384 * 74), (17, 2048, 13, 2 * 2 * 74),
                                           (16, 21, 8, 1 * 1 * 1), (64, 22, 16, 4 * 2 * 2)])
def test_tiles_are_row_blocks_column_blocks_and_steps(m, k, n, want):
    assert LR.tiles(Call(0, 0, 0, m, k, n).shape) == want


def test_the_record_refuses_calls_out_of_order_and_after_publication():
    rec = record()
    c0, c1 = Call(3, 0, 0, 4, 32, 8), Call(3, 1, 0, 4, 32, 8)
    root = b"\x01" * 64
    with pytest.raises(LR.RecordError, match="before its A root"):
        rec.words(c0, root)
    with pytest.raises(LR.RecordError, match="next is call 0"):
        rec.a_root(c1, root)
    with pytest.raises(LR.RecordError, match="of pass 4 in pass 3"):
        rec.a_root(Call(4, 0, 0, 4, 32, 8), root)
    with pytest.raises(LR.RecordError, match="an A root of 32 bytes"):
        rec.a_root(c0, bytes(32))
    rec.a_root(c0, root)
    with pytest.raises(LR.RecordError, match="once, before its words"):
        rec.a_root(c0, root)
    with pytest.raises(LR.RecordError, match="A root and no words"):
        rec.publish()
    with pytest.raises(LR.RecordError, match="before its A root"):
        rec.words(c1, root)
    with pytest.raises(LR.RecordError, match="a words root of 32 bytes"):
        rec.words(c0, bytes(32))
    rec.words(c0, root)
    with pytest.raises(LR.RecordError, match="before its A root"):
        rec.words(c0, root)
    with pytest.raises(LR.RecordError, match="no sample before"):
        rec.commitment()
    rec.publish()
    with pytest.raises(LR.RecordError, match="after the pass's record was published"):
        rec.a_root(c1, root)
    with pytest.raises(LR.RecordError, match="published once"):
        rec.publish()
