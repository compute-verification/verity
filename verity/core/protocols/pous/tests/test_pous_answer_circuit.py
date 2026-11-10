"""PoUS's answer opening (`pous.answer_circuit`) against `hashlib`'s SHA-512 and `hm96.SHA512.commit_string`."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as EB
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.sha512 import pad_tail
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
from verity.ml.protocols.pous import answer_circuit as A

RNG = random.Random(9)
INST = hm96.SHA512


def _bits(rows: list[bytes]) -> np.ndarray:
    return np.array([[(b >> k) & 1 for b in row for k in range(8)] for row in rows], dtype=np.uint8)


def _bytes(row) -> bytes:
    return bytes(int(sum(int(row[8 * i + k]) << k for k in range(8))) for i in range(len(row) // 8))


def _row_digest(block: bytes) -> bytes:
    return hashlib.sha512(sha512_row_v2_prefix(8 * len(block)) + block).digest()


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


def test_word_bit_is_a_permutation():
    assert sorted(A.word_bit(t) for t in range(1024)) == list(range(1024))
    assert A.from_words(A.to_words(list(range(1024)))) == list(range(1024))


@pytest.mark.slow
@pytest.mark.parametrize("N", [8, 136, 256])
def test_row_digest_is_sha512_row_v2(N):
    """A short row, a row with a tail, and a whole number of blocks (no ``rest``).  Slow: each end traces a
    compression (about 8 s)."""
    rows = [_rand(N) for _ in range(3)]
    out = EB.evaluate_bits(bind(A.PousRowDigest, N=N), _bits(rows))
    assert [_bytes(r) for r in out] == [_row_digest(b) for b in rows]


@pytest.mark.slow
def test_compress_is_sha512s():
    """One compression from SHA-512's IV over a padded one-block message is SHA-512 of it."""
    iv = (0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
          0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179)
    cv = np.array([[(w >> k) & 1 for w in iv for k in range(64)]], dtype=np.uint8)
    msgs = [_rand(n) for n in (0, 5, 111)]
    blocks = [m + pad_tail(len(m)) for m in msgs]
    out = EB.evaluate_bits(bind(A.PousSha512Compress), np.repeat(cv, len(msgs), axis=0), _bits(blocks))
    words = [[int(sum(int(r[64 * i + k]) << k for k in range(64))) for i in range(8)] for r in out]
    assert [b"".join(w.to_bytes(8, "big") for w in ws) for ws in words] == [hashlib.sha512(m).digest() for m in msgs]


@pytest.mark.slow
def test_answer_opens_is_the_leaf():
    """An honest answer opens; flipping one bit of the digest, the commit string, the label or the salt does not.
    Slow: hm96's 1,536 matrix rows take about a minute to trace."""
    N = 136
    cases, want = [], []
    for i in range(6):
        block, y = _rand(N), _rand(INST.salt_bytes)
        x = _row_digest(block)
        h, bc = x, INST.commit_string(INST.default_key, x, y)
        if i == 2:
            h = bytes([h[0] ^ 1]) + h[1:]
        if i == 3:
            bc = bc[:-1] + bytes([bc[-1] ^ 0x80])
        if i == 4:
            block = bytes([block[0] ^ 4]) + block[1:]
        if i == 5:
            y = y[:-1] + bytes([y[-1] ^ 2])
        cases.append((block, h, y, bc))
        want.append(int(i < 2))
    cols = [_bits([c[k] for c in cases]) for k in range(4)]
    assert EB.evaluate_bits(bind(A.PousAnswerOpens, N=N), *cols)[:, 0].tolist() == want
    x, y = cases[0][1], cases[0][2]
    out = EB.evaluate_bits(bind(A.PousHm96Commit), _bits([x]), _bits([y]))
    assert _bytes(out[0]) == INST.commit_string(INST.default_key, x, y)


SLOW = pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="minutes at P2's width: VERITY_POUS_SLOW=1")


@SLOW
def test_counts_at_p2():
    """The per-answer cost: 17 compressions for the 2,056-byte block's digest, hm96's 2, and 1,535 for the equality."""
    assert EB.compile(bind(A.PousRowDigest, N=2056)).counts["and"] == 981398
    assert EB.compile(bind(A.PousAnswerOpens, N=2056)).counts["and"] == 1096499
