"""The tile checks' committed rows and tile digest (`verity_pouw.circuit.leaves`): the row schemas' words and commit strings
are the pinned vectors (``pouw_rows_vectors``), each row's bytes are the tensor's memory, and the tile digest in gates is
the pinned one and `pearl_c.leaf`."""
from __future__ import annotations

import json
import random
import types

import pytest

import pouw_rows_vectors as V
from verity.commitments.rowleaf import nvfp4_row_bytes, row_bytes
from verity.evaluation import evaluate

from verity_pouw.circuit import leaves as L
from verity_pouw.schemes import pearl_c as C

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
    `rowleaf.nvfp4_row_bytes`' and fp8-p's codes the codes' bytes, each then P's BF16 words."""
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


@pytest.mark.parametrize("tm, tn", [(1, 1), (2, 3), (3, 33), (2, 64)])
def test_the_tile_digest_is_pearl_cs_leaf(tm, tn):
    """`PearlTileDigest` is `pearl_c.leaf` (h0): a partial chunk, a row of two chunks, and one message per row."""
    rng = random.Random(tm * 100 + tn)
    c, u = ([[rng.getrandbits(32) for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    got = evaluate(L.tile_digest(tm, tn), [v for r in c for v in r], [v for r in u for v in r])
    assert L.digest_bytes(got) == C.leaf(types.SimpleNamespace(c=c, u=u))


def test_out_of_schema_rows_are_refused():
    with pytest.raises(ValueError):
        L.words_nvfp4_p([0] * 64, [0] * 3, [0] * L.P_WORDS)
    with pytest.raises(ValueError):
        L.words_fp8_p([0] * 64, [0] * 63)
    with pytest.raises(ValueError):
        L.words_tile(bytes(31))
