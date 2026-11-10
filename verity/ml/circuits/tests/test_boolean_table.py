"""`verity.ml.circuits.boolean.table`: a table read on bits is the entry at the index's low bits at every index, whatever the
index's high bits, in exactly (2^IB - 1) * 16 EW ANDs and no lookup; the Boolean form of a word read is Boolean, names the word,
takes its signature in bits and agrees with it through the word view; a word whose table does not fill its rows, or has more than
`MAX_WORDS` words, is refused."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import table as BT
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.types import Array, Value

#: (IB, EW, the index's width): one word an entry, two (as `exp2_lut127`'s 2^8), a narrow index and a 32-bit one
SHAPES = [(6, 1, 8), (5, 2, 16), (8, 2, 8), (7, 1, 32)]


def _word(ib: int, ew: int, iw: int) -> PrimitiveDefinition:
    """A word read as `verity.ml.definitions.registry.tables.table_read` makes one: (index, table) -> the entry at the index's low
    IB bits, its EW words joined low first."""
    mask, rows = (1 << ib) - 1, (ew << ib) // BT.ROW_WORDS

    def ev(idx: int, *words: int) -> int:
        e = (idx & mask) * ew
        return sum((words[e + k] & 0xFFFF) << (16 * k) for k in range(ew))

    return PrimitiveDefinition(f"BtRead{ib}x{ew}i{iw}", 1, (("index", Value(iw)), ("table", Array(rows, Array(64, Value(16))))),
                               Value(16 * ew), ev, register=False)


def _lanes(rng: np.random.Generator, ib: int, iw: int, n: int) -> np.ndarray:
    """Every index of IB bits, then random words of the index's width (high bits set)."""
    every = np.arange(1 << ib, dtype=np.uint64)
    return np.concatenate([every, rng.integers(0, 1 << iw, n, dtype=np.uint64)])


@pytest.mark.parametrize("ib,ew,iw", SHAPES)
def test_a_read_is_the_entry_at_the_index_low_bits(ib, ew, iw):
    rng = np.random.default_rng(ib * 131 + ew * 7 + iw)
    idx = _lanes(rng, ib, iw, 64)
    T, nwords = len(idx), (1 << ib) * ew
    table = rng.integers(0, 1 << 16, (nwords, T), dtype=np.uint64)
    C = Circuit()
    i = C.input("i", iw)
    words = [C.input(f"w{j}", 16) for j in range(nwords)]
    C.output("o", BT.read(C, i, words, ib, ew))
    assert (C.counts()["ands"], C.counts()["lookups"]) == (((1 << ib) - 1) * 16 * ew, 0)

    def lanes(col: np.ndarray, w: int) -> list[int]:
        return [sum(int((v >> b) & 1) << t for t, v in enumerate(col)) for b in range(w)]

    ins = {"i": lanes(idx, iw)} | {f"w{j}": lanes(table[j], 16) for j in range(nwords)}
    outs, _ = C.evaluate(ins, T)
    got = [sum(((outs["o"][b] >> t) & 1) << b for b in range(16 * ew)) for t in range(T)]
    e = (idx & np.uint64((1 << ib) - 1)).astype(np.int64) * ew
    want = [sum(int(table[e[t] + k, t]) << (16 * k) for k in range(ew)) for t in range(T)]
    assert got == want


@pytest.mark.parametrize("ib,ew,iw", SHAPES)
def test_the_boolean_form_is_its_word_read_on_bits(ib, ew, iw):
    word = _word(ib, ew, iw)
    d = BT.table_read(word, ib)
    assert BT.table_read(word, ib) is d and d.id == f"{word.name}_v2"
    b = bind(d)
    params, ret = b.signature()
    assert BL.is_boolean(b) and word_of(b) is word
    assert ([t for _, t in params], ret) == ([BL.bit_type(t) for _, t in word.params], BL.bit_type(word.ret))
    assert d.traced().counts["and"] == ((1 << ib) - 1) * 16 * ew
    rng = np.random.default_rng(ib * 17 + ew + iw)
    idx = _lanes(rng, ib, iw, 192)
    table = rng.integers(0, 1 << 16, (len(idx), (ew << ib) // BT.ROW_WORDS, BT.ROW_WORDS), dtype=np.uint64)
    got = np.asarray(evaluate_words(b, idx, table), dtype=np.uint64).reshape(len(idx), -1)
    want = np.asarray(evaluate_batch(word, idx, table), dtype=np.uint64).reshape(len(idx), -1)
    assert int((got != want).any(axis=1).sum()) == 0


def test_a_word_whose_table_does_not_fill_its_rows_is_refused():
    word = PrimitiveDefinition("BtReadBad", 1, (("index", Value(8)), ("table", Array(1, Array(64, Value(16))))), Value(16),
                               lambda i, *w: w[i & 31], register=False)
    with pytest.raises(ValueError, match="BtReadBad_v1: not a read of 2\\^5 entries"):
        BT.table_read(word, 5)
    with pytest.raises(ValueError, match="BtReadBad_v1: not a read of 2\\^9 entries"):
        BT.table_read(word, 9)


def test_a_table_past_max_words_has_no_form():
    BT.table_read(_word(11, 2, 16), 11)
    with pytest.raises(ValueError, match="BtRead12x2i16_v1: a table of 8192 words has no Boolean form"):
        BT.table_read(_word(12, 2, 16), 12)
