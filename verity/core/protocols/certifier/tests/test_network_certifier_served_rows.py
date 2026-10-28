"""The served rows (`served_rows`) on the compile slice's corpus-smallkv window: the server's rows as the model runner laid
them out give the slice's ``srv`` row, and with the row the Commit side registers there, never the prover's own, the slice
refuses six passes registered where user-1 was admitted at the sixth ask at ``rows``; the
honest run is accepted. The registration's row is bound by b‖c to its leaf, its run root and its index, so the six passes'
own row isn't the leaf, and the public record holds no row or salt."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR
from verity.ml.protocols.certifier import served_statics as SS
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
ROWS = 1 << CS.FLAGS.index("rows")
RUN_ROOT = bytes(range(64))


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    prompt = {a[0]: a[1] for a in fx["arrivals"]}
    ids = {f"user-{a[0]}-0a1b2c3d": a[0] for a in fx["arrivals"]}
    by_step = {}
    for u, step in enumerate(rows):   # the runner's layout of the fixture's rows: token rows [lo, hi) in slot order
        lo, got = 0, []
        for _slot, req, _kind, start, n in step:
            got.append((f"user-{req}-0a1b2c3d", lo, lo + n, start, prompt[req]))
            lo += n
        by_step[u] = got
    session = SR.windows(S, by_step, ids, fx["steps"])
    served = SR.register(S, session, RUN_ROOT)
    return SimpleNamespace(st=st, S=S, sl=sl, rows=rows, ids=ids, by_step=by_step, session=session, served=served,
                           record=served.record())


def test_the_runner_rows_are_the_slices(x):
    assert len(x.session) == 1 and list(x.session[0]) == x.rows
    assert x.served.registration.rows[0] == SR._row(CS.case(x.S, x.sl, BOOK).rows["srv"])[0]


def test_the_slice_refuses_six_passes_under_the_served_rows(x):
    """The honest run with the registered row in ``srv`` is accepted under the verifier's own ``pub``; the cheap run,
    six passes charged 17 bits against the honest 22, is accepted at 0x3F with its own rows in ``srv`` (a row that
    isn't the registered one, which the b‖c check refuses) and refused at ``rows`` with the registered row. These are
    the reference's verdicts: the corpus class is past what a test evaluates, and the circuit is its reference at the
    demonstration class (`test_network_certifier_compile_slice.py`)."""
    S, sl, reg = x.S, x.sl, x.served.registration
    honest = CS.case(S, sl, BOOK, served=x.session[0])
    assert CS.words(honest.rows["pub"]) == CS.pub(S, honest.length, BOOK)
    cheap_sl = CS.Slice(sl.arrivals, (0,) * 6, sl.steps, sl.tokens)
    own = CS.case(S, cheap_sl, BOOK)
    held = CS.case(S, cheap_sl, BOOK, served=x.session[0])
    assert own.length == held.length == 17 < honest.length == 22
    assert [SR._row(c.rows["srv"])[0] == reg.rows[0] for c in (honest, own, held)] == [True, False, True]
    assert [c.verdict for c in (honest, own, held)] == [CS.ACCEPT, CS.ACCEPT, CS.ACCEPT & ~ROWS]


def test_the_row_is_bound_by_bc_to_its_leaf_run_and_index(x):
    reg = x.served.registration
    path = reg.open(0).path
    assert SR.leaf_ok(x.record, RUN_ROOT, 0, x.served.bc(0), path)
    assert not SR.leaf_ok(x.record, bytes(64), 0, x.served.bc(0), path)
    assert not SR.leaf_ok(x.record, RUN_ROOT, 1, x.served.bc(0), path)
    cheap = CS.case(x.S, CS.Slice(x.sl.arrivals, (0,) * 6, x.sl.steps, x.sl.tokens), BOOK)
    forged = SR.register(x.S, [list(cheap.steps)], RUN_ROOT, salts=reg.salts)
    assert forged.registration.rows != reg.rows
    assert not SR.leaf_ok(x.record, RUN_ROOT, 0, forged.bc(0), path)


def test_two_windows_and_an_idle_tail(x):
    """Steps past the first window open a second, its idle steps zero; each window has its own leaf."""
    by_step = {**x.by_step, 8: x.by_step[2]}
    session = SR.windows(x.S, by_step, x.ids, 10)
    assert [len(w) for w in session] == [7, 3] and session[1] == [None, x.session[0][2], None]
    served = SR.register(x.S, session, RUN_ROOT)
    rec = served.record()
    assert rec["count"] == 2 and served.registration.rows[1] == SR._row(CS.srv_row(x.S, session[1]))[0]
    for w in (0, 1):
        assert SR.leaf_ok(rec, RUN_ROOT, w, served.bc(w), served.registration.open(w).path)


def test_the_slices_block_rows_are_the_registered_leaves(x):
    """At G < P (G = 1 on the corpus's P = 7) the slice's ``srv`` rows are the registration's rows of its blocks: unit w's
    row ``srv{j}`` is leaf w·(P / G) + j, held there by its b‖c and at no other leaf; the six passes with the registered
    rows in ``srv0`` … are refused at ``rows``, as at G = P."""
    G = 1
    st = CS.with_block(x.st, G)
    S = SimpleNamespace(**st)
    served = SR.register(x.S, x.session, RUN_ROOT, G=G)
    rec, reg = served.record(), served.registration
    honest = CS.case(S, x.sl, BOOK, served=x.session[0])
    names = CS.srv_names(S)
    assert rec["G"] == G and rec["count"] == len(names) == x.S.P
    for j, r in enumerate(names):
        i = SR.leaf(x.S, rec, 0, j)
        assert i == j and reg.rows[i] == SR._row(honest.rows[r])[0]
        assert SR.leaf_ok(rec, RUN_ROOT, i, served.bc(i), reg.open(i).path)
        assert not SR.leaf_ok(rec, RUN_ROOT, (i + 1) % len(names), served.bc(i), reg.open(i).path)
    held = CS.case(S, CS.Slice(x.sl.arrivals, (0,) * 6, x.sl.steps, x.sl.tokens), BOOK, served=x.session[0])
    assert honest.verdict == CS.ACCEPT and held.verdict == CS.ACCEPT & ~ROWS


@pytest.mark.parametrize("G", [0, 2, 3, 8])
def test_a_records_g_that_doesnt_divide_p_is_refused(x, G):
    """The verifier indexes a unit's rows by the public record's G, which `register` holds to divide
    P; a record whose G doesn't (2 or 3 on P = 7 floors to 3 or 2 blocks a window, 8 to none, 0 divides by zero) is
    refused there too, rather than read as another G's leaves. The tie's statics, which don't hold P, refuse that G
    too, so no unit is built for a block the registration never has."""
    with pytest.raises(ValueError, match="dividing P"):
        SR.leaf(x.S, {**x.record, "G": G}, 1, 0)
    with pytest.raises(ValueError, match="dividing P"):
        SS.statics(x.S, G=G)


def test_the_record_is_public_only(x):
    text = json.dumps(x.record)
    private = [*x.served.registration.salts, *x.served.registration.rows]
    assert not any(p.hex() in text for p in private)
    assert set(x.record) == {"schema", "binding", "root", "count", "n_bits", "G"}


@pytest.mark.parametrize("bad, why", [
    ({0: [("user-9-0a1b2c3d", 0, 1, 0, 1)]}, "no certified arrival"),
    ({0: [("user-0-0a1b2c3d", 0, 1, 1 << 20, 1)]}, "start"),
    ({0: [("user-0-0a1b2c3d", 0, 1, 0, 1)] * 3}, "over R"),
])
def test_rows_it_cant_pack_are_refused(x, bad, why):
    with pytest.raises(ValueError, match=why):
        SR.windows(x.S, bad, x.ids, 1)
