"""sha512/row-seg/v1 (frame_v3/PROTOCOL.md §6, segmented bit rows): prefixes and digests rebuilt by hand from the spec, the
segment cases (none, a short last one), domain separation from sha512/row/v2, the refusals, the hiding leaf, and the cost
claim of the design (77 in-circuit compressions and a serial chain of 14 for an 8 KiB row, against 65 and 65)."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments import rowleaf as R
from verity.ml.commitments.frame_v3 import vectors_sha512 as FV
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.indexed import RangeIndexedDomain

F = FV.load()
TAG = b"verity/sha512-row-seg/v1\0"
SEG_TAG = b"verity/sha512-row-seg/v1/segment\0"


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _outer(n_bits: int, s: int) -> bytes:
    return (TAG + n_bits.to_bytes(8, "big") + s.to_bytes(8, "big") + (1024).to_bytes(4, "big")).ljust(128, b"\0")


def _seg(j: int, s: int, m: int) -> bytes:
    return (SEG_TAG + j.to_bytes(8, "big") + s.to_bytes(8, "big") + m.to_bytes(8, "big")).ljust(128, b"\0")


def _by_hand(row: bytes, n_bits: int) -> bytes:
    s = -(-n_bits // 8192)
    ds = b""
    for j in range(s):
        m = min(8192, n_bits - 8192 * j)
        ds += hashlib.sha512(_seg(j, s, m) + row[1024 * j:1024 * j + (m + 7) // 8]).digest()
    return hashlib.sha512(_outer(n_bits, s) + ds).digest()


def _blocks(message_bytes: int) -> int:
    """SHA-512 compressions of a message of this many bytes (padding: 0x80, then 16 length bytes)."""
    return (message_bytes + 17 + 127) // 128


def test_the_tags_separate_the_outer_hash_the_segments_and_row_v2() -> None:
    assert (R.SHA512_ROW_SEG_TAG, R.SHA512_ROW_SEG_SEGMENT_TAG, R.ROW_SEG_BYTES) == (TAG, SEG_TAG, 1024)
    assert (R.SCHEMA_SHA512_ROW_SEG, R.SCHEMA_HM96_SHA512_ROW_SEG) == ("sha512/row-seg/v1", "hm96-sha512/row-seg/v1")
    tags = (TAG, SEG_TAG, R.SHA512_ROW_V2_TAG, R.SHA512_ROW_TAG)
    for a in tags:
        assert a.count(0) == 1 and a.endswith(b"\0")
        for b in tags:
            assert a == b or not b.startswith(a), "no tag is a prefix of another, so no two prefix blocks coincide"


@pytest.mark.parametrize("case", F["row_seg_leaves"], ids=lambda c: c["name"])
def test_row_seg_vectors_rebuilt_from_the_spec(case: dict) -> None:
    n_bits = case["n_bits"]
    if "words" in case:
        w = case["words"]
        words = FV.elements(w["label"], w["bits"], w["n"])
        row = R.row_bytes(words, w["bits"])
        assert n_bits == w["bits"] * w["n"] and R.bit_row((words, w["bits"])) == (row, n_bits)
        assert R.sha512_row_seg_word_digest(words, w["bits"]).hex() == case["digest"]
        assert R.sha512_row_digest(words, w["bits"], R.ROLE_X).hex() != case["digest"]
    else:
        row = _b(case["row_bytes"])
    s = len(case["segments"])
    assert case["schema"] == "sha512/row-seg/v1" and s == -(-n_bits // 8192) and len(row) == (n_bits + 7) // 8
    assert case["prefix"] == _outer(n_bits, s).hex() == R.sha512_row_seg_prefix(n_bits).hex()
    for j, seg in enumerate(case["segments"]):
        assert seg["n_bits"] == min(8192, n_bits - 8192 * j) and seg["prefix"] == _seg(j, s, seg["n_bits"]).hex()
    assert sum(seg["n_bits"] for seg in case["segments"]) == n_bits
    outer = hashlib.sha512(_b(case["prefix"]) + b"".join(_b(seg["digest"]) for seg in case["segments"])).hexdigest()
    assert case["digest"] == outer == _by_hand(row, n_bits).hex() == R.sha512_row_seg_digest(row, n_bits).hex()
    assert case["digest"] != R.sha512_row_v2_digest(row, n_bits).hex()


def test_the_vectors_cover_the_segment_cases() -> None:
    assert FV.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.frame_v3.vectors_sha512`"
    counts = {c["n_bits"]: [s["n_bits"] for s in c["segments"]] for c in F["row_seg_leaves"]}
    assert counts[0] == [] and counts[13] == [13] and counts[8192] == [8192] and counts[8193] == [8192, 1]
    assert counts[65536] == [8192] * 8 and counts[16 * 4096] == [8192] * 8 and counts[32 * 4096] == [8192] * 16
    assert {c["name"]: c["words"]["bits"] for c in F["row_seg_leaves"] if "words" in c} == {"bf16/k4096": 16, "fp32/k4096": 32, "bf16/k5": 16}
    assert any(m % 8 for ms in counts.values() for m in ms)


def test_the_empty_row_is_the_outer_prefix_alone() -> None:
    assert R.sha512_row_seg_segments(0) == []
    assert R.sha512_row_seg_digest(b"", 0) == hashlib.sha512(_outer(0, 0)).digest()
    assert len({R.sha512_row_seg_digest(b"", 0), R.sha512_row_seg_digest(b"\0", 1), R.sha512_row_seg_digest(b"\0", 8)}) == 3


@pytest.mark.parametrize(("call", "match"), [
    (lambda: R.sha512_row_seg_digest(b"\x00\x20", 13), "zero bits"),
    (lambda: R.sha512_row_seg_digest(b"\x00", 9), "is 2 bytes"),
    (lambda: R.sha512_row_seg_digest(b"", -1), "u64"),
    (lambda: R.sha512_row_seg_prefix(True), "u64"),  # type: ignore[arg-type]
    (lambda: R.sha512_row_seg_segment_prefix(2, 2, 8), "segment 2 of 2"),
    (lambda: R.sha512_row_seg_segment_prefix(0, 1, 0), "1 to 8192"),
    (lambda: R.sha512_row_seg_segment_prefix(0, 1, 8193), "1 to 8192"),
])
def test_refusals(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_the_hiding_leaf_wraps_the_digest_under_hm96_unchanged() -> None:
    case = next(c for c in F["row_seg_leaves"] if "hiding" in c)
    assert case["name"] == "bits/65536"
    h, inst = case["hiding"], hm96.SHA512
    assert h["schema"] == "hm96-sha512/row-seg/v1" and h["key_sha512"] == hashlib.sha512(inst.default_key).hexdigest()
    bc = inst.commit_string(inst.default_key, _b(case["digest"]), _b(h["salt"]))
    assert bc.hex() == h["bc"] and inst.tree_leaf(inst.default_key, bc).hex() == h["hm96_leaf"]
    doc = h["domain"]
    d = CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeIndexedDomain(*doc["range"]), hash=doc["hash"])
    assert R.row_leaf(d, 1, _b(h["hm96_leaf"]), h["schema"]).hex() == h["leaf"]
    assert R.row_leaf(d, 1, _b(h["hm96_leaf"]), R.SCHEMA_SHA512_ROW_SEG).hex() != h["leaf"]


def test_the_cost_claim_for_an_8_kib_row() -> None:
    """Every prefix is one constant block (a midstate), so a circuit compresses only what follows it."""
    n_bytes = 8192
    v1 = _blocks(128 + n_bytes) - 1
    segments = R.sha512_row_seg_segments(8 * n_bytes)
    per_segment = [_blocks(128 + (m + 7) // 8) - 1 for m in segments]
    outer = _blocks(128 + 64 * len(segments)) - 1
    assert (v1, len(segments), per_segment[0], outer) == (65, 8, 9, 5)
    assert (sum(per_segment) + outer, max(per_segment) + outer) == (77, 14)


def test_the_cost_for_pearl_c_fp32_and_nvfp4_p_rows() -> None:
    """Pearl-C's 16 KiB FP32 row (16 segments) and a 2.8 KiB nvfp4-p bit row (3 segments)."""
    for n_bytes, (total, serial) in ((16384, (153, 18)), (2880, (27, 11))):
        segments = R.sha512_row_seg_segments(8 * n_bytes)
        per = [_blocks(128 + (m + 7) // 8) - 1 for m in segments]
        outer = _blocks(128 + 64 * len(segments)) - 1
        assert (sum(per) + outer, max(per) + outer) == (total, serial), (n_bytes, sum(per) + outer, max(per) + outer)
