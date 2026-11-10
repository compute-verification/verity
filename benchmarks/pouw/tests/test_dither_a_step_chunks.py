"""Dither's DITHER_A_STEP tables ``a/<K>@<s>`` at SPLITK's chunk rows (`dither_one.a_step_rows`, `a_step_fields`, what
`a_roots_cpu` commits and h1c's A-step tables read in place): past K = 4096 a row is `weights.chunks` (K) leaves, leaf
n i + c row i's chunk c (`h1zy.chunk_leaves`) ending in the row's scale, and at K <= 4096 the table is the one before
SPLITK byte for byte (its rows whole, its domain's fields without ``chunks``)."""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.protocols.pouw.circuit import h1zy as Z

DITHER = Path(__file__).resolve().parents[2] / "hash" / "dither"
KEY = bytes(range(32))


@functools.cache
def dither_one():
    """`benchmarks/hash/dither/dither_one.py`, imported as its scripts import it (its directory on sys.path). The suite's
    other tests import their own ``native`` and ``verify`` (the harness's), so dither's directory and the modules loaded
    from it leave sys.path and sys.modules once it's imported."""
    before = set(sys.modules)
    sys.path.insert(0, str(DITHER))
    try:
        import dither_one as DO
    finally:
        sys.path.remove(str(DITHER))
        for name in set(sys.modules) - before:
            if Path(getattr(sys.modules[name], "__file__", None) or "/").parent == DITHER:
                del sys.modules[name]
    return DO


def _win(k: int) -> SimpleNamespace:
    return SimpleNamespace(window="ab" * 32, a_calls={k: [0, 3]}, t=8)


def _rows(k: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(k)
    return rng.integers(-128, 128, (r, k), dtype=np.int8), rng.uniform(0.5, 2.0, r).astype(np.float32)


def _table(DO, fields: dict, q: np.ndarray, sc: np.ndarray, label: str):
    return DO.CM.s8_table(DO.CM.domain(fields, len(q), "a-table"), KEY, label, q, sc)


@pytest.mark.parametrize("k", [64, 2048, 4096])
def test_up_to_4096_the_a_step_table_is_the_one_before_splitk(monkeypatch, k):
    DO = dither_one()
    monkeypatch.delenv(DO.A_CHUNKS_ENV, raising=False)
    win, (q, sc) = _win(k), _rows(k, 16)
    before = {"window": win.window, "step": 3, "k": k, "linears": win.a_calls[k], "slots": win.t, "schema": DO.CM.SCHEMA}
    leaves, scales = DO.a_step_rows(q, sc)
    assert DO.a_chunks(k) == 1 and DO.a_step_fields(win, k, 3) == before
    assert leaves.tobytes() == q.tobytes() and scales.tobytes() == sc.tobytes()
    got = _table(DO, DO.a_step_fields(win, k, 3), leaves, scales, f"a/{k}@3")
    assert got.root == _table(DO, before, q, sc, f"a/{k}@3").root


@pytest.mark.parametrize("k, n", [(6144, 2), (14336, 4)])
def test_past_4096_a_row_is_its_chunk_leaves_each_ending_in_its_scale(monkeypatch, k, n):
    DO = dither_one()
    monkeypatch.delenv(DO.A_CHUNKS_ENV, raising=False)
    win, (q, sc) = _win(k), _rows(k, 4)
    leaves, scales = DO.a_step_rows(q, sc)
    assert DO.a_chunks(k) == n and leaves.shape == (4 * n, k // n) and DO.a_step_fields(win, k, 0)["chunks"] == n
    got, rows = DO.CM.s8_rows(leaves, scales), DO.CM.s8_rows(q, sc)
    assert [bytes(x) for x in got] == [leaf for r in rows for leaf in Z.chunk_leaves(bytes(r), k)]


def test_a_chunks_0_keeps_rows_whole_at_a_chunked_k(monkeypatch):
    """DITHER_A_CHUNKS=0 (the timing control) commits the rows whole, under fields without ``chunks``, so its root is
    another table's than the chunked one."""
    DO, k = dither_one(), 14336
    win, (q, sc) = _win(k), _rows(k, 2)
    monkeypatch.delenv(DO.A_CHUNKS_ENV, raising=False)
    chunked = _table(DO, DO.a_step_fields(win, k, 0), *DO.a_step_rows(q, sc), f"a/{k}@0")
    monkeypatch.setenv(DO.A_CHUNKS_ENV, "0")
    leaves, scales = DO.a_step_rows(q, sc)
    assert leaves.tobytes() == q.tobytes() and "chunks" not in DO.a_step_fields(win, k, 0)
    assert _table(DO, DO.a_step_fields(win, k, 0), leaves, scales, f"a/{k}@0").root != chunked.root
