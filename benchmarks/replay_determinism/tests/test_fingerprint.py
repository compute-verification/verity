"""The row fingerprint (`replay_det.digest_rows`): one changed bit changes its row's digest and no other's, a row's digest
doesn't depend on the rows beside it, its length enters it, and a row longer than the keys folds its chunks."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_det as rd  # noqa: E402

KEYS = tuple(torch.from_numpy(k) for k in rd.fingerprint_keys())


def digest(t: "torch.Tensor", keys: tuple = KEYS) -> list[int]:
    return rd.digest_rows(t, t.shape[0], *keys).tolist()


def rows(n: int, nbytes: int, dtype: "torch.dtype" = torch.uint8) -> "torch.Tensor":
    g = np.random.default_rng(7)
    return torch.from_numpy(g.integers(0, 256, (n, nbytes), dtype=np.uint8)).view(dtype)


def flip(t: "torch.Tensor", row: int, byte: int, bit: int) -> "torch.Tensor":
    b = t.clone().view(torch.uint8)
    b[row, byte] ^= 1 << bit
    return b.view(t.dtype)


def test_keys_are_fixed_and_odd() -> None:
    a, b = rd.fingerprint_keys(), rd.fingerprint_keys()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a[0]) == rd.KEY_WORDS and all((k & 1).all() for k in a)


@pytest.mark.parametrize("dtype,nbytes", [(torch.float32, 24), (torch.bfloat16, 14), (torch.uint8, 7)],
                         ids=["32-bit words", "16-bit words", "bytes"])
def test_one_bit_changes_its_row_only(dtype: "torch.dtype", nbytes: int) -> None:
    t = rows(4, nbytes, dtype)
    base = digest(t)
    for byte in range(nbytes):
        for bit in range(8):
            d = digest(flip(t, 2, byte, bit))
            assert d[2] != base[2] and d[:2] + d[3:] == base[:2] + base[3:], (byte, bit)


def test_rows_independent_of_the_batch() -> None:
    t = rows(5, 32, torch.float32)
    base = digest(t)
    assert [digest(t[i:i + 1])[0] for i in range(5)] == base
    assert digest(t[[3, 0, 4]]) == [base[3], base[0], base[4]]


def test_bytes_and_length_enter() -> None:
    t = rows(2, 16, torch.float32)
    assert digest(t.view(torch.int32)) == digest(t)
    padded = torch.cat([t, torch.zeros(2, 1, dtype=t.dtype)], dim=1)
    assert all(x != y for x, y in zip(digest(padded), digest(t)))


def test_long_rows_fold_chunks() -> None:
    keys, chunk_keys = KEYS[0][:8], KEYS[1]
    t = rows(2, 80, torch.int32)
    want = []
    for r in t.to(torch.int64).tolist():
        acc = sum(sum(x * int(keys[j]) for j, x in enumerate(r[c * 8:c * 8 + 8])) * int(chunk_keys[c]) for c in range(3))
        acc = (acc + 80 * 0x9E3779B97F4A7C15 % (1 << 63)) % (1 << 64)
        want.append(acc - (1 << 64) if acc >= 1 << 63 else acc)
    assert digest(t, (keys, chunk_keys)) == want
    d = digest(flip(t, 1, 79, 7), (keys, chunk_keys))
    assert d[0] == want[0] and d[1] != want[1]


def test_digest_pinned() -> None:
    """A changed fingerprint makes phases fingerprinted before and after it incomparable."""
    x = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7) / 7
    assert digest(x) == [-580792813544784481, -5670171810079926710, -7542167291108679528]
    long = (torch.arange(2 * (rd.KEY_WORDS + 3), dtype=torch.int64) % 251).to(torch.int32).reshape(2, -1)
    assert digest(long) == [-3349483733805789527, -7491711547129569514]
