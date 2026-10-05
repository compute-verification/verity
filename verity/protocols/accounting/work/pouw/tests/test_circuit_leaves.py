"""The tile checks' committed rows and tile digest (`verity.protocols.accounting.work.pouw.circuit.leaves`): the row schemas' words and commit strings
are the pinned vectors (``pouw_rows_vectors``), each row's bytes are the tensor's memory, a row's commit string is core's
over `sha512/row/v1` (role X) or `sha512/row/v2` at every padding edge, and the tile digest in gates is the pinned one and
`pearl_c.leaf`.  C-Flock's tests recompute the vectors through its own ports (``backends/flock/tests/test_pouw_rows.py``)."""
from __future__ import annotations

import json
import random
import types

import pytest

import pouw_rows_vectors as V
from verity.primitives.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, bit_row, nvfp4_row_bytes, row_bytes, sha512_row_digest,
                                        sha512_row_v2_digest)
from verity.primitives.circuits.evaluation import evaluate
from catalog.definitions.scalar import BitAnd, SelectBf16

from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.circuit import pc4, pc8
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C

PINNED = json.loads(V.OUT.read_text())


def test_pinned_row_vectors():
    assert V.vectors() == PINNED


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "tile"], ids=lambda c: c["case"])
def test_the_gates_compute_the_pinned_tile_digests(case):
    """`PearlTileDigest` gives the pinned tile digest from the case's C̃ and U."""
    tm, tn = case["k"]
    x = V.inputs(case["case"], "tile", (tm, tn))
    got = evaluate(L.tile_digest(tm, tn), [v for r in x["c"] for v in r], [v for r in x["u"] for v in r])
    assert L.digest_bytes(got).hex() == case["tile_digest"]


def test_the_rows_are_the_tensors_bytes():
    """Each row's bytes are the tensor's memory: bf16 and fp32 rows their words little-endian, nvfp4-p's codes and scales
    `rowleaf.nvfp4_row_bytes`' and fp8-p's codes the codes' bytes, each then P's BF16 words; and each packed row's bytes
    and bit count are `rowleaf.bit_row`'s of its elements (what `sha512/row/v2` hashes)."""
    rng, k = random.Random(3), 128
    x16, x32 = [rng.getrandbits(16) for _ in range(k)], [rng.getrandbits(32) for _ in range(k)]
    c4, c8 = [rng.getrandbits(4) for _ in range(k)], [rng.getrandbits(8) for _ in range(k)]
    scales, p = [rng.getrandbits(8) for _ in range(k // 16)], [rng.getrandbits(16) for _ in range(L.P_WORDS)]
    le = lambda vs, n: b"".join(v.to_bytes(n, "little") for v in vs)                 # noqa: E731
    rows = {"bf16": (L.words_bf16(x16), le(x16, 2)), "fp32": (L.words_fp32(x32), le(x32, 4)),
            "nvfp4-p": (L.words_nvfp4_p(c4, scales, p), nvfp4_row_bytes(c4, scales) + le(p, 2)),
            "fp8-p": (L.words_fp8_p(c8, p), bytes(c8) + le(p, 2))}
    for row, (words, memory) in rows.items():
        assert row_bytes(words, 16) == memory, row
        assert len(words) == L.ROW_WORDS[row](k), row
    assert bit_row((c4, 4), (scales, 8), (p, 16)) == (rows["nvfp4-p"][1], 16 * L.ROW_WORDS["nvfp4-p"](k))
    assert bit_row((c8, 8), (p, 16)) == (rows["fp8-p"][1], 16 * L.ROW_WORDS["fp8-p"](k))


@pytest.mark.parametrize("row, sizes", [("bf16", (64, 128, 192)), ("nvfp4-p", (2, 16, 54, 56, 118, 120)),
                                        ("tile", (16,))], ids=["v1", "v2", "tile"])
def test_a_rows_commit_string_is_cores_at_every_padding_edge(row, sizes):
    """`commit` is `hm96.SHA512.commit_string` of the row's `sha512/row/v1` digest (role X, whole blocks) or its
    `sha512/row/v2` digest: v2 rows that pad into one block after the prefix (54 words), into two (56), and across the next
    edge (118, 120)."""
    rng = random.Random(len(row))
    for n in sizes:
        words, y = [rng.getrandbits(16) for _ in range(n)], bytes(rng.getrandbits(8) for _ in range(192))
        x = (sha512_row_digest(words, 16, ROLE_X) if L.ROW_SCHEMA[row] == SCHEMA_HM96_SHA512_ROW
             else sha512_row_v2_digest(row_bytes(words, 16), 16 * n))
        assert L.row_digest(row, words) == x, n
        assert L.commit(row, words, y) == L.INST.commit_string(L.KEY, x, y), n


@pytest.mark.parametrize("tm, tn", [(1, 1), (2, 3), (3, 33), (2, 64)])
def test_the_tile_digest_is_pearl_cs_leaf(tm, tn):
    """`PearlTileDigest` is `pearl_c.leaf` (h0): a partial chunk, a row of two chunks, and one message per row."""
    rng = random.Random(tm * 100 + tn)
    c, u = ([[rng.getrandbits(32) for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    got = evaluate(L.tile_digest(tm, tn), [v for r in c for v in r], [v for r in u for v in r])
    assert L.digest_bytes(got) == C.leaf(types.SimpleNamespace(c=c, u=u))


def test_out_of_schema_rows_are_refused():
    for row, words in (("bf16", 32), ("fp32", 96)):
        with pytest.raises(ValueError, match="whole SHA-512 blocks"):
            L.commit(row, [0] * words, bytes(192))
    with pytest.raises(ValueError, match="positive number of words"):
        L.row_leaf("tile", 0)
    with pytest.raises(ValueError):
        L.words_nvfp4_p([0] * 64, [0] * 3, [0] * L.P_WORDS)
    with pytest.raises(ValueError):
        L.words_fp8_p([0] * 64, [0] * 63)
    with pytest.raises(ValueError):
        L.words_tile(bytes(31))
    with pytest.raises(ValueError):
        L.words_ok(2)


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "ok"], ids=lambda c: c["case"])
def test_an_ok_row_with_a_bit_set_above_bit_0_is_refused(case):
    """An ok row is one word with bits 1-15 zero, so one admission value has one leaf: the reference reads the pinned
    vector's word (`ok_of`) and refuses it with any one of bits 1-15 set, and so does `commit`, as it refuses a row of two
    words or none."""
    words, salt = V.words("ok", V.inputs(case["case"], "ok", case["k"])), V.shake(case["case"], "salt", L.INST.salt_bytes)
    assert L.ok_of(words) == words[0] and L.commit("ok", words, salt).hex() == case["bc"]
    for bad in [[words[0] | 1 << b] for b in range(1, 16)] + [[words[0], 0], []]:
        with pytest.raises(ValueError, match="an ok row is one word"):
            L.ok_of(bad)
        with pytest.raises(ValueError, match="an ok row is one word"):
            L.commit("ok", bad, salt)


ROW_UNITS = ([pc4.templates(64)[r] for r in ("row_a", "row_b")]
             + [pc8.templates(128, dev=d)[r] for d in ("h100", "sm120") for r in ("row_a", "row_b")])


@pytest.mark.parametrize("fn", ROW_UNITS, ids=lambda fn: fn.id)
def test_a_row_unit_fixes_its_ok_words_bits_1_to_15_in_gates(fn):
    """The circuit can't write an ok row `ok_of` refuses: a row unit's last output, its ok row's one word (`leaves.OK`), is
    `SelectBf16` of its admission bit between the constants 1 and 0 (`pc4.ok_word`), so every bit of the word is a gate's
    and bits 1-15 are zero whatever the bit, not free bits of the committed row."""
    b = fn.body
    _space, k, _leaf = b.ret[len(b.ret) - 1]
    gate = b.nodes[k]
    bit, one, zero = (b.nodes[a.refs[0][1]] for a in gate.args)
    assert fn.signature()[1].fields[-1] == L.OK
    assert gate.fn.id == SelectBf16.id and bit.fn.id == BitAnd.id
    assert [(c.fn.family, c.fn.evaluate()) for c in (one, zero)] == [("Const", 1), ("Const", 0)]
    assert [L.ok_of([gate.fn.evaluate(v, 1, 0)]) for v in (0, 1)] == [0, 1]
