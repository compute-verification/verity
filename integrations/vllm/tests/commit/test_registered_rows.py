"""`registered_rows.commit` is `Registered.commit` on an array: the same entry, row for row, for every width and row shape."""

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import hm96
from verity_vllm.commit import registered_rows as RR

PROGRAM = "ab" * 64
N = hm96.SHA512.salt_bytes


def _salts(rows: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, rows * N, dtype=np.uint8).tobytes()


@pytest.mark.parametrize("bits,dtype,word_bits,n,words", [
    (8, np.uint8, 16, 24, 8),        # an FP8 code a leaf, one u16 word each
    (16, np.uint16, 16, 130, 65),    # 65 words pad to two SHA-512 blocks
    (16, np.uint16, 16, 64, 64),     # exactly one block, no padding
    (32, np.uint32, 32, 120, 40),    # two u16 words a leaf, low first: 80 words pad to 128
    (1, np.uint8, 16, 7, 7),         # one row
])
def test_commit_is_registered_commit_on_its_rows(monkeypatch, bits, dtype, word_bits, n, words):
    monkeypatch.setattr(RR, "BLOCK_ROWS", 2)   # rows hashed across several numpy blocks
    leaves = np.random.default_rng(n).integers(0, 1 << bits, n, dtype=np.uint64).astype(dtype)
    salts = _salts(n // words, n)
    got = RR.commit(PROGRAM, "shared.weights[3]", leaves, word_bits, words, salts)
    want = G.Registered.commit(PROGRAM, "shared.weights[3]", leaves.reshape(-1, words).tolist(),
                               [salts[i * N:(i + 1) * N] for i in range(n // words)], word_bits=word_bits)
    assert got.entry == want.entry() and got.root == want.root.hex() and got.salts == salts


def test_default_rows_and_fresh_salts():
    leaves = np.arange(4096 * 3, dtype=np.uint16)
    a, b = RR.commit(PROGRAM, "v", leaves, 16), RR.commit(PROGRAM, "v", leaves, 16)
    assert a.entry["words"] == 4096 and a.entry["leaves"] == 3 and len(a.salts) == 3 * N
    assert a.salts != b.salts and a.root != b.root          # hiding: fresh salts, another root
    assert RR.commit(PROGRAM, "v", leaves, 16, salts=a.salts).entry == a.entry
    assert [RR.row_width(n) for n in (1, 7, 4096, 4097, 8192 * 3, 4099)] == [1, 7, 4096, 4097 // 17, 4096, 1]


def test_refusals():
    with pytest.raises(ValueError, match="does not fit 16 bits"):
        RR.commit(PROGRAM, "v", np.array([1 << 16], dtype=np.uint32), 16)
    with pytest.raises(ValueError, match="unsigned array"):
        RR.commit(PROGRAM, "v", np.array([1, 2], dtype=np.int16), 16)
    with pytest.raises(ValueError, match="not rows of 3"):
        RR.commit(PROGRAM, "v", np.arange(4, dtype=np.uint16), 16, words=3)
    with pytest.raises(ValueError, match=f"one {N}-byte salt per row"):
        RR.commit(PROGRAM, "v", np.arange(4, dtype=np.uint16), 16, words=2, salts=b"\0" * N)
    with pytest.raises(ValueError, match="never reused"):
        RR.commit(PROGRAM, "v", np.arange(4, dtype=np.uint16), 16, words=2, salts=b"\1" * (2 * N))
    with pytest.raises(ValueError):
        RR.commit(PROGRAM, "v", np.arange(4, dtype=np.uint16), 24)
