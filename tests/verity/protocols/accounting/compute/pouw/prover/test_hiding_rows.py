"""`coins.requests.hiding_rows`: the reference hm96 commitment of its rows (`audit.commit_rows`), leaf for leaf, and the
roots it made before it took its frame from `merkle` (pinned)."""
from __future__ import annotations

import random

import pytest

from verity.protocols.accounting.compute.pouw.prover.serving.coins import requests as RQ
from verity.protocols.accounting.compute.pouw.verifier.audit import HM96, commit_rows

BINDING = bytes(range(64))
#: {(n, width): root hex} of `_rows(n, width)` under BINDING and owner 5, from the hand-built frame's `hiding_rows`
PINNED = {
    (1, 7): "fa4e2f2e1bb635448c81caf69634770944428ecfa60e48646d4689470173e379"
            "b162c9931ba86b68dd6395e01e1b50e6629ccd29bd57763c6721f6b026dace71",
    (64, 68): "b7bfc15a38171f00fa9a779f3b575df88cbbd89d6f28ae032dddfdc3b99a23f5"
              "792ee3fce6760afc627b48bb9cb05f74671aa88bf3198a770ae1f763fe6ad18c",
    (37, 6144): "0da5594c415816dbc6bfe5b8261c05ce599fdd2d5fbbc42078440125df1b62d1"
                "f23041ddb0cf4cbc6115c680b25ae60304246343c5cc05b7f3ee6df80e01aefd",
    (5, 0): "ee988515d5b1992a26ed2f14694f350a6de969c6ddb17ab2d1c04db4701b20bd"
            "459e75c99a46bf0bc1841189ff26b56563ea6955deffd3c79794d555c6d6decd",
}


def _rows(n: int, width: int) -> tuple[list[bytes], bytes]:
    rng = random.Random(1000 + n)
    return [rng.randbytes(width) for _ in range(n)], rng.randbytes(192 * n)


@pytest.mark.parametrize("n,width", sorted(PINNED))
def test_the_root_is_the_pinned_one(n, width):
    assert RQ.hiding_rows(BINDING, 5, *_rows(n, width)).commitment.root.hex() == PINNED[n, width]


@pytest.mark.parametrize("n,width", sorted(PINNED))
def test_every_leaf_and_the_root_are_the_reference_commitments(n, width):
    values, ys = _rows(n, width)
    want = commit_rows(BINDING, 5, values, hash=HM96, salts=[ys[192 * i:192 * (i + 1)] for i in range(n)])
    got = RQ.hiding_rows(BINDING, 5, values, ys)
    assert got.commitment == want.commitment and all(got.open(i) == want.open(i) for i in range(n))
