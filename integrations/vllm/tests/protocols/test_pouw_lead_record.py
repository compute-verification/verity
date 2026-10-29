"""`lead.record`: a pass's record takes the weights' root, the key leaf and the window's call count before the window's
salt, then the calls in order below that count, each one's A root after the salt, then the forward's one words root. It
is published once, before any sample."""
from __future__ import annotations

import hashlib

import pytest
from verity_vllm.protocols.pouw.lead import record as LR
from verity_vllm.protocols.pouw.lead.feed import Call

SCHEME = "lead-blc4a-t64-rot"
WEIGHTS, LEAF, SALT, ROOT = b"\x06" * 64, b"\x07" * 64, b"\x05" * 64, b"\x01" * 64


def record(calls: int = 4) -> LR.PassRecord:
    return LR.PassRecord(SCHEME, 0, 1, 3, WEIGHTS, LEAF, calls, kernel="lead-h200/abc")


def salted(calls: int = 4) -> LR.PassRecord:
    rec = record(calls)
    rec.salted(SALT)
    return rec


def test_the_record_publishes_its_window_and_calls_in_order():
    rec = salted()
    calls = [Call(3, 0, 0, 1, 2048, 3072), Call(3, 1, 1, 1, 8192, 2048)]
    rec.a_root(calls[0], b"\x20" * 64)
    rec.a_root(calls[1], b"\x21" * 64)
    rec.words(b"\x10" * 64)
    pc = rec.publish()
    sha = lambda b: hashlib.sha512(b).digest()  # noqa: E731
    hdr = lambda c: b"".join(v.to_bytes(8, "big") for v in (c.pass_index, c.index, c.linear, c.m, c.k, c.n))  # noqa: E731
    assert LR.header(calls[1]) == hdr(calls[1])
    assert rec.a_roots == [b"\x20" * 64, b"\x21" * 64]
    assert pc.activation_roots == sha(b"\x20" * 64 + b"\x21" * 64)
    assert pc.transcript_root == b"\x10" * 64
    assert pc.records_root == sha(LEAF + (4).to_bytes(8, "big") + hdr(calls[0]) + hdr(calls[1]))
    assert (pc.units, pc.tiles) == (2, 1 * 384 * 74 + 1 * 256 * 293)
    assert (pc.scheme, pc.rank, pc.world, pc.pass_index, pc.salt, pc.weights_root, pc.kernel, pc.controls) == \
        (SCHEME, 0, 1, 3, SALT, WEIGHTS, "lead-h200/abc", ())
    assert pc.to_json()["activation_roots"] == pc.activation_roots.hex()
    assert rec.commitment() is pc


@pytest.mark.parametrize("m, k, n, want", [(1, 2048, 3072, 1 * 384 * 74), (17, 2048, 13, 2 * 2 * 74),
                                           (16, 21, 8, 1 * 1 * 1), (64, 22, 16, 4 * 2 * 2)])
def test_tiles_are_lean_rowblocks_colblocks_steps(m, k, n, want):
    assert LR.tiles(Call(0, 0, 0, m, k, n).shape) == want


def test_the_window_is_fixed_before_its_salt():
    """The weights' root, the key leaf and the call count are the record's from the start, each refused at a wrong size or
    count, and the salt comes once, before any A root."""
    for calls in (0, 2**63):
        with pytest.raises(LR.RecordError, match="a window of"):
            record(calls)
    with pytest.raises(LR.RecordError, match="a weights' root of 32 bytes"):
        LR.PassRecord(SCHEME, 0, 1, 3, bytes(32), LEAF, 4)
    with pytest.raises(LR.RecordError, match="a key leaf of 32 bytes"):
        LR.PassRecord(SCHEME, 0, 1, 3, WEIGHTS, bytes(32), 4)
    rec = record()
    with pytest.raises(LR.RecordError, match="A of call 0 before the window's salt"):
        rec.a_root(Call(3, 0, 0, 4, 32, 8), ROOT)
    with pytest.raises(LR.RecordError, match="the words root before any call's A root"):
        rec.words(ROOT)
    with pytest.raises(LR.RecordError, match="published before its forward's words root"):
        rec.publish()
    with pytest.raises(LR.RecordError, match="a salt of 32 bytes"):
        rec.salted(bytes(32))
    rec.salted(SALT)
    with pytest.raises(LR.RecordError, match="the window's salt comes once"):
        rec.salted(SALT)


def test_a_call_past_the_windows_count_is_refused():
    rec = salted(calls=2)
    rec.a_root(Call(3, 0, 0, 4, 32, 8), ROOT)
    rec.a_root(Call(3, 1, 0, 4, 32, 8), ROOT)
    with pytest.raises(LR.RecordError, match="past the window's 2 calls, fixed before its salt"):
        rec.a_root(Call(3, 2, 0, 4, 32, 8), ROOT)


def test_the_record_refuses_calls_out_of_order_and_after_the_words_root_or_publication():
    rec = salted()
    c0, c1 = Call(3, 0, 0, 4, 32, 8), Call(3, 1, 0, 4, 32, 8)
    with pytest.raises(LR.RecordError, match="the record's next is call 0"):
        rec.a_root(c1, ROOT)
    with pytest.raises(LR.RecordError, match="of pass 4 in pass 3"):
        rec.a_root(Call(4, 0, 0, 4, 32, 8), ROOT)
    with pytest.raises(LR.RecordError, match="a call's A root of 32 bytes"):
        rec.a_root(c0, bytes(32))
    rec.a_root(c0, ROOT)
    with pytest.raises(LR.RecordError, match="the record's next is call 1"):
        rec.a_root(c0, ROOT)
    with pytest.raises(LR.RecordError, match="published before its forward's words root"):
        rec.publish()
    with pytest.raises(LR.RecordError, match="a words root of 32 bytes"):
        rec.words(bytes(32))
    rec.words(ROOT)
    with pytest.raises(LR.RecordError, match="the words root comes once"):
        rec.words(ROOT)
    with pytest.raises(LR.RecordError, match="A of call 1 after the forward's words root"):
        rec.a_root(c1, ROOT)
    with pytest.raises(LR.RecordError, match="no sample before"):
        rec.commitment()
    rec.publish()
    with pytest.raises(LR.RecordError, match="after the pass's record was published"):
        rec.a_root(c1, ROOT)
    with pytest.raises(LR.RecordError, match="published once"):
        rec.publish()
