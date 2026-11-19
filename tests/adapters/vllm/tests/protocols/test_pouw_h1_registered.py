"""`protocol_options.pouw_h1_registered` against core's `registered_value.Registered.commit`: over random rows of
16-bit words, one row to a few dozen, one word to a z/y record's 3072, under each domain binding and the same given
salts, the ``b || c`` strings, levels, entry, paths and statement are core's; a row read opens under
`registered.verify_read`. A window's rows aren't kept unless asked, and a read of one is refused. A small H1 window served
through `Served` registers the same entries, roots and manifest rows as with core's `Registered.commit` in its place."""
from __future__ import annotations

import os

import numpy as np
import pytest
from verity.commitments.periods import interface as I
from verity.commitments.periods import rows as CS
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import registered as NR
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import served as SV
from verity.protocols.proofs.spot.verifier.one_stage import registered as G
from verity.commitments import registered_value as RV

PROGRAM, WINDOW = "ab" * 64, "cd" * 64


def _ys(n: int) -> bytes:
    return os.urandom(192 * n)


def _core(program, value, words, row_salts, keep_rows=False, schema=G.SCHEMA_V2, later=False, rows=None, also=None,
          **binding):
    assert also is None, "core commits the value alone"
    words = NR._host(words)
    n = len(words)
    assert rows is None or np.array_equal(np.asarray(rows[:]), words)
    salts = ([row_salts[j] for j in range(n)] if isinstance(row_salts, NR.PublicSalts)
             else [row_salts[192 * j:192 * (j + 1)] for j in range(n)])
    return RV.Registered.commit(program, value, np.asarray(words).tolist(), salts, schema=schema, **binding)


@pytest.mark.parametrize("n,w,binding", [(1, 1, {}), (2, 2, {"window": WINDOW}), (3, 66, {}), (5, 1026, {"window": WINDOW}),
                                         (8, 3, {"produced": "ef" * 64}), (4, 5, {"session": "s-1"}),
                                         (33, 3072, {"window": WINDOW}), (64, 10, {})])
def test_a_value_is_what_core_commits(n, w, binding):
    """The same rows under the same salts: core's entry, root, levels, ``b || c`` strings, salts and rows, each row's path,
    a statement's table, and every row's read opens under core's `verify_read`."""
    rng = np.random.default_rng(n * 1000 + w)
    words = rng.integers(0, 1 << 16, (n, w), dtype=np.uint16)
    ys = _ys(n)
    ref, got = _core(PROGRAM, "h1/v", words, ys, **binding), NR.commit(PROGRAM, "h1/v", words, ys, True, **binding)
    assert isinstance(got, RV.Registered)
    assert got.entry() == ref.entry() and got.root == ref.root and got.domain == ref.domain
    assert got.levels == ref.levels and got.bcs == ref.bcs
    assert list(got.salts) == list(ref.salts) and list(got.rows) == list(ref.rows)
    assert [got.path(p) for p in range(n)] == [ref.path(p) for p in range(n)]
    ps = [n - 1, 0, n // 2]
    (rows, salts, head), want = got.statement(ps), ref.statement(ps)
    assert (rows, salts, head) == want
    assert got.salts.at(ps) == b"".join(want[1]) and got.rows.at(ps).tolist() == [list(r) for r in want[0]]
    for bad in ([n], [-1]):
        with pytest.raises(IndexError):
            got.salts.at(bad)
        with pytest.raises(IndexError):
            got.rows.at(bad)
    assert all(G.verify_read(got.domain, ref.root, p, got.bcs[p], got.path(p), G.SCHEMA_V2) for p in range(n))


def test_a_windows_reads_are_cores_statement():
    """A window's registered reads (`interface._Drawn`, `rows.value_salts` and `value_rows`): for this value and core's, at
    rows in any order and repeated, the salts, rows and header entry core's ``statement`` gives."""
    words = np.random.default_rng(4).integers(0, 1 << 16, (9, 40), dtype=np.uint16)
    ys = _ys(9)
    ref, got = _core(PROGRAM, "h1/v", words, ys), NR.commit(PROGRAM, "h1/v", words, ys, True)
    ps, more = [7, 0, 3, 7], [8, 3]
    drawn = I._Drawn({"core": ref, "h1": got}, {"core": [ps, more], "h1": [more, ps]})
    for name, v in (("core", ref), ("h1", got)):
        for at in (ps, more):
            rows, salts, head = ref.statement(at)
            assert drawn.salts(name, at) == CS.value_salts(v, at) == b"".join(salts) and drawn.entry(name, at) == head
            assert CS.value_rows(v, at).tolist() == [list(r) for r in rows]


@pytest.mark.parametrize("n,w", [(1, 1), (3, 511), (2, 512), (5, 513), (4, 2048), (3, 2100)])
def test_a_row_seg_value_is_what_core_commits(n, w):
    """An embedding table's ``row-seg`` rows, one segment's bytes, a whole segment, one word past it and several: core's
    entry, levels and ``b || c`` strings under the same salts, hashed with NumPy wherever `path` says, and every row's read
    opens under core's `verify_read`."""
    rng = np.random.default_rng(n * 7 + w)
    words = rng.integers(0, 1 << 16, (n, w), dtype=np.uint16)
    ys = _ys(n)
    ref = _core(PROGRAM, "weights/e@64/table", words, ys, schema=G.SCHEMA_SEG)
    got = NR.commit(PROGRAM, "weights/e@64/table", words, ys, True, schema=G.SCHEMA_SEG)
    assert got.entry() == ref.entry() and got.levels == ref.levels and got.bcs == ref.bcs and got.schema == G.SCHEMA_SEG
    assert all(G.verify_read(got.domain, ref.root, p, got.bcs[p], got.path(p), G.SCHEMA_SEG) for p in range(n))
    assert NR.commit(PROGRAM, "weights/e@64/table", words, ys).root != got.root
    with pytest.raises(ValueError, match="rows of"):
        NR.commit(PROGRAM, "v", words, ys, schema=RV.SCHEMA)


def test_given_salts_are_the_rows_salts_and_one_reused_is_refused():
    """The value is core's commit under its ``row_salts`` (here a public table's, `pouw_h1_served.Served._pubs`); a salt
    short or reused is refused."""
    words: np.ndarray = np.arange(15, dtype=np.uint16).reshape(3, 5)
    ys = b"".join(bytes([j + 1]) * 192 for j in range(3))
    got = NR.commit(PROGRAM, "h1/pub@0", words, ys, True, window=WINDOW)
    ref = RV.Registered.commit(PROGRAM, "h1/pub@0", words.tolist(), [ys[192 * j:192 * (j + 1)] for j in range(3)],
                              schema=G.SCHEMA_V2, window=WINDOW)
    assert got.entry() == ref.entry() and got.levels == ref.levels and got.bcs == ref.bcs
    assert list(got.salts) == list(ref.salts) and list(got.rows) == list(ref.rows)
    for bad in (ys[:-1], ys[:192] * 3):
        with pytest.raises(ValueError, match="never reused"):
            NR.commit(PROGRAM, "h1/pub@0", words, bad, True, window=WINDOW)


def test_a_windows_rows_arent_kept_unless_asked():
    """Without ``keep_rows`` the tree is the same and a read of a row is refused; the salts are still held."""
    words: np.ndarray = np.arange(12, dtype=np.uint16).reshape(3, 4)
    ys = _ys(3)
    got, ref = NR.commit(PROGRAM, "v", words, ys, window=WINDOW), _core(PROGRAM, "v", words, ys, window=WINDOW)
    assert got.entry() == ref.entry() and got.levels == ref.levels and len(got.rows) == 3
    with pytest.raises(LookupError, match="rebuilds"):
        got.rows[0]
    with pytest.raises(LookupError, match="rebuilds"):
        got.rows.at([0])
    with pytest.raises(LookupError):
        got.statement([0])
    assert got.salts[-1] == ref.salts[2] and got.salts[1:] == list(ref.salts[1:])


@pytest.mark.parametrize("words", [np.zeros((0, 4), np.uint16), np.zeros((2, 0), np.uint16), np.zeros((2, 3), np.int32),
                                   np.zeros(4, np.uint16)])
def test_a_value_of_no_rows_or_not_u16_words_is_refused(words):
    with pytest.raises(ValueError, match="16-bit words"):
        NR.commit(PROGRAM, "v", words, _ys(max(1, words.shape[0])))


def test_a_served_window_registers_what_core_would(monkeypatch):
    """A small H1 window (`test_pouw_h1_served._window`: two K, q and k on one input, a served and an empty step) and the
    period's weights: every registered value's entry, the window's served digest and each call's manifest row are the
    same as with core's `Registered.commit` registering them."""
    torch = pytest.importorskip("torch")
    from tests.protocols import test_pouw_h1_served as T

    def window():
        torch.manual_seed(0)
        weights = T._period(T._weights())
        run, _ws, _got = T._window(T.W1, weights=weights, served=1)
        return weights.entries(), run.served.entries(), run.served.digest(), [run.tables[c.index] for c in run.calls]

    got = window()
    monkeypatch.setattr(SV.NR, "commit", _core)
    want = window()
    assert got == want and len(got[1]) == 2 * len(T.SHAPES) * T.STEPS + (T.STEPS + 1) * len(T.KS)
