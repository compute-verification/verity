"""A table read (``gf2.Circuit.lookup``) laid out inline by ``ir_lower.layout`` as plain rows of the unit: the rows compute
``table[idx]`` on every index, value bits feed later gates, and every row is forced (a flipped row bit fails its lane)."""
from __future__ import annotations

import numpy as np
import pytest

from verity_flock import gf2
from verity_flock import ir_lower as IL

if not hasattr(gf2.Circuit, "lookup"):
    pytest.skip("gf2 has no table reads here", allow_module_level=True)


@pytest.fixture(params=[6, 8])
def read(request):
    n = request.param
    rng = np.random.default_rng(n)
    words = rng.integers(0, 1 << 31, size=1 << n, dtype=np.uint32)
    words[::5] = 0
    name = f"test-inline-{n}"
    gf2.TABLES[name] = lambda: words
    gf2._TABLE_WORDS.pop(name, None)
    C = gf2.Circuit()
    x, y = C.input("in0", n), C.input("in1", n)
    v = C.lookup(name, [x[i] ^ y[i] if i % 2 else x[i] for i in range(n)], 31)
    C.output("out0", v[:16])
    C.output("out1", v[16:31] + [C.AND(v[0], x[0])])
    lay = IL.layout(C, ["in0", "in1"], ["out0", "out1"])
    xs = np.arange(1 << n, dtype=np.uint64)
    ys = rng.integers(0, 1 << n, size=1 << n, dtype=np.uint64)
    yield n, words, lay, xs, ys
    gf2.TABLES.pop(name)
    gf2._TABLE_WORDS.pop(name, None)


def test_rows_compute_the_table(read):
    n, words, lay, xs, ys = read
    odd = np.uint64(sum(1 << i for i in range(1, n, 2)))
    want = words[(xs ^ (ys & odd)).astype(np.int64)].astype(np.uint64)
    (o0, o1), ok = IL.evaluate_layout(lay, [xs, ys], len(xs))
    assert ok == (1 << len(xs)) - 1
    assert np.array_equal(o0 | ((o1 & np.uint64(0x7FFF)) << np.uint64(16)), want)
    assert np.array_equal((o1 >> np.uint64(15)) & np.uint64(1), want & xs & np.uint64(1))
    assert lay.counts["reads"] == 1 and lay.counts["read_nnz_a"] == sum(bin(int(w)).count("1") for w in words)


def test_every_row_is_forced(read):
    n, _words, lay, xs, ys = read
    first = lay.in_words * IL.WORD
    for row in range(first, lay.useful):
        r = [lay.rows_a[row], lay.rows_b[row]]
        assert all(c < row or c == lay.const for x in r for c in IL._cols_of(x)), row
    for row in range(first, lay.useful, 7):
        if row != lay.const:
            _, ok = IL.evaluate_layout(lay, [xs, ys], len(xs), flip=(row, 3))
            assert not (ok >> 3) & 1, row
