"""`verity.ml.commitments.gates.rowleaf`: the sha512/row-seg/v1 digest in gates reproduces
``frame_v3/vectors_sha512.json``'s ``row_seg_leaves`` (the whole at the short lengths, its segment part on every vector
segment and its outer part on every vector's segment digests) and `rowleaf.sha512_row_seg_digest` on random 262,144-bit
rows, a 64 x 64 Pearl-C tile's ``-h3`` digest; a flipped data bit, two segments swapped and a wrong length each change it;
it refuses malformed bindings and rows; and its parts cost the gates pinned here."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import compile as compile_bits
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import rowleaf as R
from verity.ml.commitments.frame_v3 import vectors_sha512 as FV
from verity.ml.commitments.gates import rowleaf as G
from verity.ml.commitments.gates.sha512 import midstate

CASES = [c for c in FV.load()["row_seg_leaves"] if c["n_bits"]]
#: (part, its static) -> (And, Xor, Not) gates of its traced circuit: a short segment, a full one, and the outer hash of one
#: segment and of a tile's 32
PARTS = {
    ("Sha512RowSegSegment", 13): (54140, 261300, 5348),
    ("Sha512RowSegSegment", 8192): (508253, 2489308, 47114),
    ("Sha512RowSegOuter", 13): (55592, 270549, 6070),
    ("Sha512RowSegOuter", 262144): (971308, 4777935, 87777),
}
#: N_BITS -> the digest's gates (`SpecializedDefinition.gates`: its parts' and the two constant bits its body writes)
WHOLE = {13: 653001, 262144: 103266622}
#: a 64 x 64 Pearl-C tile's row, in bits: 32 KiB, 32 full segments
TILE = 262144
#: the tile digest's And gates: 32 segments and the outer hash, 305 compressions
TILE_ANDS = 17235404


def _row(case: dict) -> bytes:
    if "words" in case:
        w = case["words"]
        return R.row_bytes(FV.elements(w["label"], w["bits"], w["n"]), w["bits"])
    return bytes.fromhex(case["row_bytes"])


def _cv(j: int, s: int, m: int) -> list[int]:
    """Segment j of s (m bits)'s chaining value, its prefix's midstate, as a `Digest`'s bits."""
    return [(v >> k) & 1 for v in midstate(R.sha512_row_seg_segment_prefix(j, s, m)) for k in range(64)]


def _digests(target, rows: list[list[int]]) -> list[bytes]:
    got = evaluate_bits(target, np.array(rows, dtype=np.uint8))
    return [G.digest_bytes(g.reshape(-1)) for g in got]


def _random_row(rng: random.Random, n_bits: int) -> bytes:
    row = bytearray(rng.randbytes((n_bits + 7) // 8))
    if n_bits % 8:
        row[-1] &= (1 << n_bits % 8) - 1
    return bytes(row)


def test_the_reference_is_rowleaf_on_the_vectors() -> None:
    assert sorted(c["n_bits"] for c in CASES) == [1, 13, 80, 8191, 8192, 8193, 24581, 65536, 65536, 131072]
    for case in CASES:
        row, n = _row(case), case["n_bits"]
        bits = G.arguments(row, n)
        assert len(bits) == n and (row, n) == R.bit_row((bits, 1))
        d = G.reference(row, n)
        assert d.hex() == case["digest"] and G.digest_bytes(G.digest_bits(d)) == d
        for j, seg in enumerate(case["segments"]):
            assert G.digest_bytes(G.digest_bits(bytes.fromhex(seg["digest"]))).hex() == seg["digest"], (case["name"], j)


def test_the_digest_is_typed_by_its_length() -> None:
    for n, s in ((1, 1), (13, 1), (8191, 1), (8192, 1), (8193, 2), (TILE, 32)):
        target = G.digest(n)
        assert target is G.digest(n)
        assert [t.leaves for _, t in target.signature()[0]] == [n] and target.signature()[1] == G.DIGEST
        assert [t.leaves for _, t in bind(G.Sha512RowSegOuter, N_BITS=n).signature()[0]] == [512 * s]
    assert G.DIGEST.leaves == 512 and G.DIGEST.elem.elem == BL.BIT
    for m in (1, 13, 8192):
        assert [t.leaves for _, t in bind(G.Sha512RowSegSegment, M=m).signature()[0]] == [512, m]


def test_the_cost_is_305_compressions_for_a_tile() -> None:
    """Every Q and S is one constant block, so a hash compresses only the blocks after it."""
    def blocks(n_bytes: int) -> int:
        return (128 + n_bytes + 17 + 127) // 128 - 1

    segments = R.sha512_row_seg_segments(TILE)
    assert segments == [8192] * 32
    assert [blocks((m + 7) // 8) for m in (1, 13, 8191, 8192)] == [1, 1, 9, 9] and blocks(64 * 32) == 17
    assert sum(blocks((m + 7) // 8) for m in segments) + blocks(64 * len(segments)) == 305


def test_malformed_bindings_and_rows_are_refused() -> None:
    for n in (0, -1, "8", 1 << 64):
        with pytest.raises(ValueError, match="N_BITS is an int in 1 ... 2"):
            G.digest(n).signature()  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="N_BITS is an int in 1 ... 2"):
            bind(G.Sha512RowSegOuter, N_BITS=n).signature()
    for m in (0, 8193, "8"):
        with pytest.raises(ValueError, match="M is an int in 1 ... 8192"):
            bind(G.Sha512RowSegSegment, M=m).signature()
    with pytest.raises(ValueError, match="2 bytes with zero fill"):
        G.arguments(b"\x00", 13)
    with pytest.raises(ValueError, match="2 bytes with zero fill"):
        G.arguments(b"\x00\x20", 13)
    with pytest.raises(ValueError, match="N_BITS is an int"):
        G.arguments(b"", 0)
    with pytest.raises(ValueError, match="64 bytes"):
        G.digest_bits(bytes(32))
    with pytest.raises(ValueError, match="512 bits"):
        G.digest_bytes([0] * 256)


@pytest.mark.slow
def test_the_digest_reproduces_the_short_vectors() -> None:
    """About four minutes, most of it tracing a full segment: the only check that the whole's wiring (each segment's slice of the
    row, its midstate from j, s and m, the digests' order into the outer hash) gives the vectors' digests, over one short
    segment, a full one with a one-bit last segment (8193) and three full ones with a five-bit last (24581)."""
    for case in CASES:
        n = case["n_bits"]
        if n in (1, 13, 80, 8193, 24581):
            assert _digests(G.digest(n), [G.arguments(_row(case), n)]) == [bytes.fromhex(case["digest"])], case["name"]


@pytest.mark.slow
def test_the_parts_reproduce_every_vector_segment_and_outer_hash() -> None:
    """About eight minutes, most of it tracing the segment part at M = 8191 and 8192: every segment digest of the vectors (37 full
    segments and the short ones of 1, 5, 13, 80 and 8191 bits) from its midstate, and every vector's outer hash from its
    segment digests, at every length the vectors take, among them PoUW's 8 KiB A row and Pearl-C's 16 KiB FP32 row."""
    by_m: dict[int, tuple[list, list, list]] = {}
    for case in CASES:
        row, n, s = _row(case), case["n_bits"], len(case["segments"])
        bits = G.arguments(row, n)
        for j, seg in enumerate(case["segments"]):
            m = seg["n_bits"]
            cvs, segs, want = by_m.setdefault(m, ([], [], []))
            cvs.append(_cv(j, s, m))
            segs.append(bits[G.SEG_BITS * j:G.SEG_BITS * j + m])
            want.append(bytes.fromhex(seg["digest"]))
        d = [b for seg in case["segments"] for b in G.digest_bits(bytes.fromhex(seg["digest"]))]
        assert _digests(bind(G.Sha512RowSegOuter, N_BITS=n), [d]) == [bytes.fromhex(case["digest"])], case["name"]
    assert {m: len(v[0]) for m, v in by_m.items()} == {1: 2, 13: 1, 8191: 1, 8192: 37, 5: 1, 80: 1}
    for m, (cvs, segs, want) in by_m.items():
        got = evaluate_bits(bind(G.Sha512RowSegSegment, M=m), np.array(cvs, dtype=np.uint8), np.array(segs, dtype=np.uint8))
        assert [G.digest_bytes(g.reshape(-1)) for g in got] == want, m


@pytest.mark.slow
def test_a_wrong_length_or_a_flipped_bit_changes_a_short_digest() -> None:
    """About two and a half minutes, the length controls: the same bits zero-extended by one bit, by a byte, and past a
    segment (8192 to 8193 bits) each give another digest, the reference's at that length; and one flipped bit does too."""
    rng = random.Random("row-seg-gates/length")
    row = _random_row(rng, 13)
    flipped = bytes([row[0] ^ 1, row[1]])
    got = [*_digests(G.digest(13), [G.arguments(row, 13), G.arguments(flipped, 13)]),
           *_digests(G.digest(14), [G.arguments(row, 14)]), *_digests(G.digest(21), [G.arguments(row + b"\0", 21)])]
    assert got == [G.reference(row, 13), G.reference(flipped, 13), G.reference(row, 14), G.reference(row + b"\0", 21)]
    case = next(c for c in CASES if c["n_bits"] == 8192)
    full = _row(case)
    assert _digests(G.digest(8193), [G.arguments(full + b"\0", 8193)]) == [G.reference(full + b"\0", 8193)]
    assert len({*got, bytes.fromhex(case["digest"]), G.reference(full + b"\0", 8193)}) == 6


@pytest.mark.slow
def test_the_deployed_tile_digest_and_its_controls() -> None:
    """About six minutes and 10 GB, most of it tracing the 17-compression outer hash: the whole at a 64 x 64 Pearl-C tile's
    262,144 bits (32 segments, the only binding that reaches that hash) against the reference on two random rows, then with a
    data bit flipped in the first and in the last segment and with segments 0 and 1, then 30 and 31, swapped: each changes
    the digest. The outer hash and the whole cost the pinned gates, and the whole's compiled counts are its parts'."""
    rng = random.Random("row-seg-gates/tile")
    rows = [_random_row(rng, TILE) for _ in range(2)]
    a = rows[0]
    first, last = bytes([a[0] ^ 1]) + a[1:], a[:-1] + bytes([a[-1] ^ 0x80])
    swap01 = a[1024:2048] + a[:1024] + a[2048:]
    swap3031 = a[:30 * 1024] + a[31 * 1024:] + a[30 * 1024:31 * 1024]
    cases = [*rows, first, last, swap01, swap3031]
    got = _digests(G.digest(TILE), [G.arguments(r, TILE) for r in cases])
    assert got == [G.reference(r, TILE) for r in cases]
    assert len(set(got)) == len(cases)
    seg, outer = PARTS[("Sha512RowSegSegment", 8192)], PARTS[("Sha512RowSegOuter", TILE)]
    assert G.Sha512RowSegOuter.traced(TILE).counts == dict(zip(("and", "xor", "not"), outer))
    assert compile_bits(G.digest(TILE)).counts == {k: 32 * x + y for k, x, y in zip(("and", "xor", "not"), seg, outer)}
    assert G.digest(TILE).gates == WHOLE[TILE] == 32 * sum(seg) + sum(outer) + 2 and 32 * seg[0] + outer[0] == TILE_ANDS


@pytest.mark.slow
def test_the_parts_and_the_digest_cost_the_pinned_gates() -> None:
    """About two minutes of tracing the full segment: the counts the module's docstring and a unit's size rest on, at the
    short bindings and the full segment (the tile's outer hash is pinned where the tile is evaluated)."""
    for (name, static), counts in PARTS.items():
        if static != TILE:
            assert getattr(G, name).traced(static).counts == dict(zip(("and", "xor", "not"), counts)), (name, static)
    seg13, outer13 = PARTS[("Sha512RowSegSegment", 13)], PARTS[("Sha512RowSegOuter", 13)]
    assert G.digest(13).gates == WHOLE[13] == sum(seg13) + sum(outer13) + 2
