"""`lead.commit`: root(A_j) is the committer's rows-sha512 tree over the call's padded rows, and the pass's record keeps
calls in order, root before words, and is published once, before any sample."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from verity_vllm.commit import step_rows as SR
from verity_vllm.protocols.pouw.lead import commit as LC
from verity_vllm.protocols.pouw.lead.feed import Call

KP = bytes(range(64))
SCHEME = "lead-blc4a-t64-rot"


def rows_of(xq: np.ndarray, padded: int) -> list[bytes]:
    data = np.zeros((padded, xq.shape[1]), dtype=np.int8)
    data[:len(xq)] = xq
    b = data.tobytes()
    return [b[i:i + SR.ROW_BYTES].ljust(SR.ROW_BYTES, b"\0") for i in range(0, len(b), SR.ROW_BYTES)]


def xq_of(m: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(-127, 128, size=(m, k), dtype=np.int8)


@pytest.mark.parametrize("m, k, padded", [(1, 2048, 16), (17, 2048, 32), (1, 8192, 16), (40, 100, 48)])
def test_a_root_is_the_committers_tree_over_the_padded_rows(m, k, padded):
    call = Call(2, 5, 7, m, k, 64)
    xq = xq_of(m, k)
    rows = rows_of(xq, padded)
    ks = hashlib.sha512(LC.A_TAG.encode() + KP + (2).to_bytes(8, "big") + (5).to_bytes(8, "big")).digest()
    assert LC.call_key(KP, call) == ks
    dom = LC.a_domain(call, len(rows))
    want = SR.levels(dom, rows, ks, 8 * SR.ROW_BYTES)[-1][0]
    assert LC.a_root(xq, call, KP) == want and len(want) == LC.ROOT_BYTES


def test_a_root_binds_the_data_the_call_and_the_key():
    call = Call(0, 0, 0, 9, 64, 8)
    xq = xq_of(9, 64)
    root = LC.a_root(xq, call, KP)
    flipped = xq.copy()
    flipped[8, 63] ^= 1
    others = [LC.a_root(flipped, call, KP), LC.a_root(xq, Call(0, 1, 0, 9, 64, 8), KP),
              LC.a_root(xq, Call(0, 0, 1, 9, 64, 8), KP), LC.a_root(xq, Call(1, 0, 0, 9, 64, 8), KP),
              LC.a_root(xq, call, bytes(64))]
    assert root not in others and len(set(others)) == len(others)


def test_a_root_refuses_a_wrong_a():
    call = Call(0, 0, 0, 4, 32, 8)
    with pytest.raises(LC.RecordError, match="int8 \\(4, 32\\)"):
        LC.a_root(xq_of(4, 32).astype(np.int16), call, KP)
    with pytest.raises(LC.RecordError, match="int8 \\(4, 32\\)"):
        LC.a_root(xq_of(5, 32), call, KP)
    with pytest.raises(LC.RecordError, match="64 bytes"):
        LC.a_root(xq_of(4, 32), call, bytes(32))


def record() -> LC.PassRecord:
    return LC.PassRecord(SCHEME, 0, 1, 3, KP, b"\x05" * 64, b"\x06" * 64, kernel="lead-h200/abc")


def test_the_record_publishes_its_calls_in_order():
    rec = record()
    calls = [Call(3, 0, 0, 1, 2048, 3072), Call(3, 1, 1, 1, 8192, 2048)]
    a = [rec.a_root(c, xq_of(c.m, c.k, i)) for i, c in enumerate(calls[:1])]
    rec.words(calls[0], b"\x10" * 64)
    a.append(rec.a_root(calls[1], xq_of(1, 8192, 1)))
    rec.words(calls[1], b"\x11" * 64)
    pc = rec.publish()
    sha = lambda b: hashlib.sha512(b).digest()  # noqa: E731
    hdr = lambda c: b"".join(v.to_bytes(8, "big") for v in (c.pass_index, c.index, c.linear, c.m, c.k, c.n))  # noqa: E731
    assert pc.activation_roots == sha(a[0] + a[1])
    assert pc.transcript_root == sha(b"\x10" * 64 + b"\x11" * 64)
    assert pc.records_root == sha(hdr(calls[0]) + hdr(calls[1]))
    assert (pc.units, pc.tiles) == (2, 1 * 384 * 74 + 1 * 256 * 293)
    assert (pc.scheme, pc.rank, pc.world, pc.pass_index, pc.kernel, pc.controls) == (SCHEME, 0, 1, 3, "lead-h200/abc", ())
    assert pc.to_json()["activation_roots"] == pc.activation_roots.hex()
    assert rec.commitment() is pc


def test_the_record_refuses_calls_out_of_order_and_after_publication():
    rec = record()
    c0, c1 = Call(3, 0, 0, 4, 32, 8), Call(3, 1, 0, 4, 32, 8)
    with pytest.raises(LC.RecordError, match="before its A root"):
        rec.words(c0, bytes(64))
    with pytest.raises(LC.RecordError, match="next is call 0"):
        rec.a_root(c1, xq_of(4, 32))
    with pytest.raises(LC.RecordError, match="of pass 4 in pass 3"):
        rec.a_root(Call(4, 0, 0, 4, 32, 8), xq_of(4, 32))
    rec.a_root(c0, xq_of(4, 32))
    with pytest.raises(LC.RecordError, match="once, before its words"):
        rec.a_root(c0, xq_of(4, 32))
    with pytest.raises(LC.RecordError, match="A root and no words"):
        rec.publish()
    with pytest.raises(LC.RecordError, match="before its A root"):
        rec.words(c1, bytes(64))
    with pytest.raises(LC.RecordError, match="words root of 32 bytes"):
        rec.words(c0, bytes(32))
    rec.words(c0, bytes(64))
    with pytest.raises(LC.RecordError, match="before its A root"):
        rec.words(c0, bytes(64))
    with pytest.raises(LC.RecordError, match="no sample before"):
        rec.commitment()
    rec.publish()
    with pytest.raises(LC.RecordError, match="after the pass's record was published"):
        rec.a_root(c1, xq_of(4, 32))
    with pytest.raises(LC.RecordError, match="published once"):
        rec.publish()


def test_the_record_takes_a_64_byte_pass_key():
    with pytest.raises(LC.RecordError, match="64 bytes"):
        LC.PassRecord(SCHEME, 0, 1, 0, bytes(16), b"", b"")
