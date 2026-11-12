"""`history_tables_device`: a history table's tree on the GPU and its CPU twin, each held bit for bit to `history.HistoryTable`
over the same rows under the same salts: the root after each append (one row, several, up to the capacity), every
row's string, paths at held and placeholder positions, and the entry. The rows cover a segment's edges (one word; 512 words,
exactly one segment; 576, a DSA latent row; 4096, the most). The twin needs a C++ compiler; the device tests need CUDA and
nvcc, and skip elsewhere."""
from __future__ import annotations

import numpy as np
import pytest

from verity.commitments import history as HT
from verity_vllm.commit import history_tables as H
from verity_vllm.commit import history_tables_device as HD

PROGRAM, VALUE = "ab" * 64, H.name("r0", 3)


def _rows(seed: int, k: int, words: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, (k, words), dtype=np.uint16)


def _same(t: HD.Table, ref: HT.HistoryTable, positions: list[int]) -> None:
    assert t.root == ref.root and t.bcs == ref.bcs and t.entry() == ref.entry()
    assert t.paths(positions) == b"".join(ref.path(p) for p in positions)
    assert H.check(t, t.bcs)


def _appended(device, words: int, capacity: int, batches: list[int]) -> None:
    t = HD.Table(PROGRAM, VALUE, words, capacity, device=device)
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


def _built(rows: np.ndarray, salts: list[bytes] | None = None, device=None) -> HD.Table:
    t = HD.Table(PROGRAM, VALUE, rows.shape[1], 8, device=device)
    t.append(rows, salts)
    return t


def test_two_builds_of_one_table_share_no_salt():
    """A table built twice over the same rows, under the same program and name, draws each row's 192-byte salt afresh:
    the two builds share no salt, so no string, and their roots differ. Nothing the builder holds fixes a salt."""
    rows = _rows(9, 3, 4)
    a, b = _built(rows), _built(rows)
    assert all(len(s) == 192 for s in a.salts + b.salts) and len(set(a.salts + b.salts)) == 6
    assert not set(a.bcs) & set(b.bcs) and a.root != b.root


def test_it_takes_salts_given_as_history_table_does():
    """Under another table's salts it is that table: its strings and root. A salt is refused when it is not 192 bytes, not
    one a row, used twice in one append, or one the table has used, given or drawn."""
    rows = _rows(9, 3, 4)
    a = _built(rows)
    b = _built(rows, a.salts)
    assert (b.salts, b.bcs, b.root) == (a.salts, a.bcs, a.root)
    for salts, why in [(a.salts[:2], "one 192-byte salt per row"), ([y[:191] for y in a.salts], "one 192-byte salt per row"),
                       ([a.salts[0]] * 3, "never reused"), (a.salts, "never reused")]:
        with pytest.raises(ValueError, match=why):
            a.append(rows, salts)
    drawn = _built(rows)
    with pytest.raises(ValueError, match="never reused"):
        drawn.append(rows[:1], drawn.salts[:1])
    assert len(a.rows) == len(drawn.rows) == 3


def test_it_refuses_what_history_table_refuses():
    t = HD.Table(PROGRAM, VALUE, 4, 4)
    t.append(_rows(1, 3, 4))
    with pytest.raises(ValueError, match="capacity"):
        t.append(_rows(2, 2, 4))
    with pytest.raises(ValueError, match="rows of 4"):
        t.append(_rows(3, 1, 5))
    with pytest.raises(ValueError, match="positions"):
        t.paths([4])
    for words, capacity in [(4, 6), (4, 1), (4097, 4), (0, 4)]:
        with pytest.raises(ValueError):
            HD.Table(PROGRAM, VALUE, words, capacity)


def test_histories_take_the_twin_as_their_tables():
    """`Histories` with the twin's tables registers the roots `history.check` recomputes from the strings they commit."""
    hs = H.Histories(PROGRAM, capacity=16, make=HD.Table)
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
    """The twin under the device table's salts is the device table; a second device build draws salts of its own."""
    torch = _cuda()
    t = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, device=0)
    twin = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY)
    rows = _rows(6, 9, H.WORDS)
    t.append(torch.from_numpy(rows.view(np.int16)).cuda().view(torch.bfloat16))
    twin.append(rows, t.salts)
    assert t.root == twin.root and t.bcs == twin.bcs and t.paths([0, 8, 9]) == twin.paths([0, 8, 9])
    again = HD.Table(PROGRAM, VALUE, H.WORDS, HT.CAPACITY, device=0)
    again.append(rows)
    assert not set(again.salts) & set(t.salts) and again.root != t.root


def test_the_device_table_takes_salts_given():
    _cuda()
    rows = _rows(9, 3, 4)
    twin = _built(rows)
    t = _built(rows, twin.salts, device=0)
    assert (t.salts, t.bcs, t.root) == (twin.salts, twin.bcs, twin.root)
