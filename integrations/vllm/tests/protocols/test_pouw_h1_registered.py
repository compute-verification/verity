"""`protocol_options.pouw_h1_registered` against core's `registered.Registered.commit` (VLLM-76): over random rows of
16-bit words, one row to a few dozen, one word to a z/y record's 3072, under each domain binding, the salts are
`audit.hm96_salt`'s, and the ``b || c`` strings, levels, entry, paths and statement are core's; a row read opens under
`registered.verify_read`. A window's rows aren't kept unless asked, and a read of one is refused. A small H1 window served
through `Served` registers the same entries, roots and manifest rows as with core's `Registered.commit` in its place."""
from __future__ import annotations

import numpy as np
import pytest

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity_vllm.protocols.pouw.h1 import registered as NR

PROGRAM, WINDOW = "ab" * 64, "cd" * 64
KEY = bytes(range(32))


def _core(program, value, words, key, nonce, keep_rows=False, row_salts=None, **binding):
    salts = ([hm96_salt(key, nonce, j) for j in range(len(words))] if row_salts is None
             else [row_salts[192 * j:192 * (j + 1)] for j in range(len(words))])
    return G.Registered.commit(program, value, np.asarray(words).tolist(), salts, schema=G.SCHEMA_V2, **binding)


def test_salts_are_hm96_salts_across_a_chacha_pass():
    """Row j's salt is `hm96_salt(key, nonce, j)`, a u64 nonce past 2^32 included, and rows past the first pass too."""
    for nonce in (0, 7, (1 << 42) + 3, (1 << 63) + 5):
        assert NR.salts(KEY, nonce, 3) == b"".join(hm96_salt(KEY, nonce, j) for j in range(3))
    n = NR._SALTED + 4
    got = NR.salts(KEY, 11, n)
    assert len(got) == 192 * n and all(got[192 * j:192 * (j + 1)] == hm96_salt(KEY, 11, j) for j in range(n - 8, n))


@pytest.mark.parametrize("n,w,binding", [(1, 1, {}), (2, 2, {"window": WINDOW}), (3, 66, {}), (5, 1026, {"window": WINDOW}),
                                         (8, 3, {"produced": "ef" * 64}), (4, 5, {"session": "s-1"}),
                                         (33, 3072, {"window": WINDOW}), (64, 10, {})])
def test_a_value_is_what_core_commits(n, w, binding):
    """The same rows under the same salts: core's entry, root, levels, ``b || c`` strings, salts and rows, each row's path,
    a statement's table, and every row's read opens under core's `verify_read`."""
    rng = np.random.default_rng(n * 1000 + w)
    words = rng.integers(0, 1 << 16, (n, w), dtype=np.uint16)
    nonce = (1 << 50) + n
    ref, got = _core(PROGRAM, "h1/v", words, KEY, nonce, **binding), NR.commit(PROGRAM, "h1/v", words, KEY, nonce, True,
                                                                              **binding)
    assert isinstance(got, G.Registered)
    assert got.entry() == ref.entry() and got.root == ref.root and got.domain == ref.domain
    assert got.levels == ref.levels and got.bcs == ref.bcs
    assert list(got.salts) == list(ref.salts) and list(got.rows) == list(ref.rows)
    assert [got.path(p) for p in range(n)] == [ref.path(p) for p in range(n)]
    ps = [n - 1, 0, n // 2]
    (rows, salts, head), want = got.statement(ps), ref.statement(ps)
    assert (rows, salts, head) == want
    assert all(G.verify_read(got.domain, ref.root, p, got.bcs[p], got.path(p), G.SCHEMA_V2) for p in range(n))


def test_given_salts_are_the_rows_salts_and_one_reused_is_refused():
    """With ``row_salts`` (a public table's, `pouw_h1_served.Served._pubs`) the value is core's commit under those salts,
    not the key's; a salt short or reused is refused."""
    words = np.arange(15, dtype=np.uint16).reshape(3, 5)
    ys = b"".join(bytes([j + 1]) * 192 for j in range(3))
    got = NR.commit(PROGRAM, "h1/pub@0", words, KEY, 0, True, ys, window=WINDOW)
    ref = G.Registered.commit(PROGRAM, "h1/pub@0", words.tolist(), [ys[192 * j:192 * (j + 1)] for j in range(3)],
                              schema=G.SCHEMA_V2, window=WINDOW)
    assert got.entry() == ref.entry() and got.levels == ref.levels and got.bcs == ref.bcs
    assert list(got.salts) == list(ref.salts) and list(got.rows) == list(ref.rows)
    for bad in (ys[:-1], ys[:192] * 3):
        with pytest.raises(ValueError, match="never reused"):
            NR.commit(PROGRAM, "h1/pub@0", words, KEY, 0, True, bad, window=WINDOW)


def test_a_windows_rows_arent_kept_unless_asked():
    """Without ``keep_rows`` the tree is the same and a read of a row is refused; the salts are still drawn on read."""
    words = np.arange(12, dtype=np.uint16).reshape(3, 4)
    got, ref = NR.commit(PROGRAM, "v", words, KEY, 9, window=WINDOW), _core(PROGRAM, "v", words, KEY, 9, window=WINDOW)
    assert got.entry() == ref.entry() and got.levels == ref.levels and len(got.rows) == 3
    with pytest.raises(LookupError, match="rebuilds"):
        got.rows[0]
    with pytest.raises(LookupError):
        got.statement([0])
    assert got.salts[-1] == ref.salts[2] and got.salts[1:] == list(ref.salts[1:])


@pytest.mark.parametrize("words", [np.zeros((0, 4), np.uint16), np.zeros((2, 0), np.uint16), np.zeros((2, 3), np.int32),
                                   np.zeros(4, np.uint16)])
def test_a_value_of_no_rows_or_not_u16_words_is_refused(words):
    with pytest.raises(ValueError, match="16-bit words"):
        NR.commit(PROGRAM, "v", words, KEY, 0)


def test_a_served_window_registers_what_core_would(monkeypatch):
    """A small H1 window (`test_pouw_h1_served._window`: two K, q and k on one input, a served and an empty step) and the
    period's weights: every registered value's entry, the window's served digest and each call's manifest row are the
    same as with core's `Registered.commit` registering them."""
    torch = pytest.importorskip("torch")
    from tests.protocols import test_pouw_h1_served as T
    from verity_vllm.protocols.pouw.h1 import served as SV

    def window():
        torch.manual_seed(0)
        weights = T._period(T._weights())
        run, _ws, _got = T._window(T.W1, weights=weights, served=1)
        return weights.entries(), run.served.entries(), run.served.digest(), [run.tables[c.index] for c in run.calls]

    got = window()
    monkeypatch.setattr(SV.NR, "commit", _core)
    want = window()
    assert got == want and len(got[1]) == 2 * len(T.SHAPES) * T.STEPS + (T.STEPS + 1) * len(T.KS)
