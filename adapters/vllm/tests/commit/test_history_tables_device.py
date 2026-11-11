"""`history_tables_device`: a history table's tree on the GPU and its CPU twin, each held bit for bit to `history.HistoryTable`
over the same rows under the table's own salts: the root after each append (one row, several, up to the capacity), every
row's string, paths at held and placeholder positions, and the entry. The rows cover a segment's edges (one word; 512 words,
exactly one segment; 576, a DSA latent row; 4096, the most). The twin needs a C++ compiler; the device tests need CUDA and
nvcc, and skip elsewhere."""
from __future__ import annotations

import numpy as np
import pytest

from verity.commitments import history as HT
from verity_vllm.commit import history_tables as H
from verity_vllm.commit import history_tables_device as HD

PROGRAM, VALUE, KEY = "ab" * 64, H.name("r0", 3), bytes(range(100, 132))


def _rows(seed: int, k: int, words: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, (k, words), dtype=np.uint16)


def _same(t: HD.Table, ref: HT.HistoryTable, positions: list[int]) -> None:
    assert t.root == ref.root and t.bcs == ref.bcs and t.entry() == ref.entry()
    assert t.paths(positions) == b"".join(ref.path(p) for p in positions)
    assert H.check(t, t.bcs)


def _appended(device, words: int, capacity: int, batches: list[int]) -> None:
    t = HD.Table(PROGRAM, VALUE, words, capacity, key=KEY, device=device)
    ref = HT.HistoryTable(PROGRAM, VALUE, words, capacity)
    assert t.root == ref.root
    for i, k in enumerate(batches):
        rows = _rows(i, k, words)
        assert t.append(rows) == range(len(ref.rows), len(ref.rows) + k)
        ref.append(rows, t.salts[len(ref.rows):])
        _same(t, ref, sorted({0, len(ref.rows) - 1, len(ref.rows) % capacity, capacity - 1}))


@pytest.mark.parametrize("words, capacity, batches", [(576, HT.CAPACITY, [1, 5, 3]), (1, 8, [1, 7]), (512, 16, [4, 1]),
                                                      (4096, 4, [2, 2])], ids=["dsa", "one-word", "one-segment", "most"])
def test_the_cpu_twin_is_history_table(words, capacity, batches):
    _appended(None, words, capacity, batches)


def test_its_salts_are_its_own_and_differ_by_table_and_position():
    """No salt is given, and each (table, position) has its own: the key's ChaCha20 stream under the table's domain id."""
    a = HD.Table(PROGRAM, VALUE, 4, 8, key=KEY)
    b = HD.Table(PROGRAM, H.name("r0", 4), 4, 8, key=KEY)
    rows = _rows(9, 3, 4)
    a.append(rows)
    b.append(rows)
    assert len(set(a.salts + b.salts)) == 6 and all(len(s) == 192 for s in a.salts)
    assert HD.salt_key(KEY, a.domain.domain_id) != HD.salt_key(KEY, b.domain.domain_id)
    with pytest.raises(ValueError, match="never given"):
        a.append(rows, a.salts)


def test_it_refuses_what_history_table_refuses():
    t = HD.Table(PROGRAM, VALUE, 4, 4, key=KEY)
    t.append(_rows(1, 3, 4))
    with pytest.raises(ValueError, match="capacity"):
        t.append(_rows(2, 2, 4))
    with pytest.raises(ValueError, match="rows of 4"):
        t.append(_rows(3, 1, 5))
    with pytest.raises(ValueError, match="positions"):
        t.paths([4])
    for words, capacity in [(4, 6), (4, 1), (4097, 4), (0, 4)]:
        with pytest.raises(ValueError):
            HD.Table(PROGRAM, VALUE, words, capacity, key=KEY)


def test_histories_take_the_twin_as_their_tables():
    """`Histories` with the twin's tables registers the roots `history.check` recomputes from the strings they commit."""
    hs = H.Histories(PROGRAM, capacity=16, make=lambda p, v, w, c: HD.Table(p, v, w, c, key=KEY))
    hs.append("r0", 3, 0, _rows(4, 5, H.WORDS))
    hs.append("r0", 3, 5, _rows(5, 2, H.WORDS))
    t = hs.table("r0", 3)
    assert isinstance(t, HD.Table) and H.check(t, [hs.strings("r0", 3, [p]) for p in range(7)])


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or HD.D.nvcc() is None:
        pytest.skip("needs CUDA and nvcc")
    return torch


@pytest.mark.parametrize("words, capacity, batches", [(576, HT.CAPACITY, [1, 64, 3]), (1, 8, [1, 7]), (4096, 4, [2, 2])],
                         ids=["dsa", "one-word", "most"])
def test_the_device_table_is_history_table(words, capacity, batches):
    _cuda()
    _appended(0, words, capacity, batches)


def test_the_device_table_appends_rows_already_on_the_device():
    torch = _cuda()
    t = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, key=KEY, device=0)
    twin = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, key=KEY)
    rows = _rows(6, 9, H.WORDS)
    t.append(torch.from_numpy(rows.view(np.int16)).cuda().view(torch.bfloat16))
    twin.append(rows)
    assert t.root == twin.root and t.salts == twin.salts and t.paths([0, 8, 9]) == twin.paths([0, 8, 9])
