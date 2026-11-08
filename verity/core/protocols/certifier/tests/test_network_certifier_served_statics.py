"""The served rows held to their steps' Programs (`served_statics.ServedStaticsTie_v1`) on the compile slice's
corpus-smallkv window: the honest rows, registered a row a block, are accepted against the Programs the window ran, at
one block a window and at one step a block; the six-pass run's rows, which the slice accepts under their own
registered row, are refused; each field edit is refused by the flag that names it, and the ids and kinds are not the
tie's; the circuit is `check` on any rows; and each block's row is its leaf of the registration, as the Lean verifier's
registered reads open it."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR
from verity.ml.protocols.certifier import served_statics as SS, served_tie as ST
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.commitments import advice as ADV

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"


def _book() -> dict:
    """The registered book the compile slice's charge reads (as `test_network_certifier_compile_slice.book`)."""
    streams = json.loads((FIXTURE.parent / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


BOOK = _book()
COUNT, POS = 1 << SS.FLAGS.index("count"), 1 << SS.FLAGS.index("pos")
RUN_ROOT = bytes(range(64))


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                     K=fx["K"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(S=S, sl=sl, rows=rows, programs=SS.program_of(rows))


def _cases(x, window, G):
    st = SS.statics(x.S, G=G)
    T = SimpleNamespace(**st)
    return st, [SS.case(T, SR.block_bits(x.S, window, G, j), x.programs[j * G:(j + 1) * G]) for j in range(x.S.P // G)]


@pytest.mark.parametrize("G", [None, 1])
def test_the_honest_window_is_accepted_block_by_block(x, G):
    st, cases = _cases(x, x.rows, G or x.S.P)
    assert [c.verdict for c in cases] == SS.evaluate(st, cases) == [SS.ACCEPT] * (x.S.P // st["G"])


def test_six_passes_are_refused_against_the_programs_the_window_ran(x):
    """The six-pass run through the tie. The prover registers the six-pass run's rows (user-1 never admitted),
    consistently: their row is the registered one, so the slice accepts them, charged 17 bits under the registered
    book. Step 6's Program computed user-1's 29 prefill tokens in slot 1 beside user-0's decode; the cheap rows hold
    slot 1 empty, so the tie refuses at ``count``."""
    S, sl = x.S, x.sl
    cheap = CS.case(S, CS.Slice(sl.arrivals, (0,) * 6, sl.steps, sl.tokens), BOOK)
    assert cheap.verdict == CS.ACCEPT and cheap.length == 17
    served = SR.register(S, [list(cheap.steps)], RUN_ROOT, G=1)
    assert [u for u, (a, b) in enumerate(zip(cheap.steps, x.rows)) if a != b] == [6]
    st, cases = _cases(x, list(cheap.steps), 1)
    assert [SR._row(c.rows["srb"])[0] for c in cases] == list(served.registration.rows)
    assert [c.verdict for c in cases] == SS.evaluate(st, cases) == [SS.ACCEPT] * 6 + [SS.ACCEPT & ~COUNT]


def _edit(x, u, k, at, v):
    rows = [list(r) for r in x.rows[u]]
    rows[k][at] = v
    return [tuple(map(tuple, rows)) if i == u else s for i, s in enumerate(x.rows)]


@pytest.mark.parametrize("what, at, by, want", [
    ("start one on", 3, 1, SS.ACCEPT & ~POS),
    ("one token short", 4, -1, SS.ACCEPT & ~COUNT),
    ("another request", 1, 1, SS.ACCEPT),
    ("the other kind", 2, 1, SS.ACCEPT),
])
def test_each_edit_is_refused_by_its_flag(x, what, at, by, want):
    u, k = 1, 0
    window = _edit(x, u, k, at, x.rows[u][k][at] + by if at != 2 else 1 - x.rows[u][k][at])
    st, cases = _cases(x, window, 1)
    assert [c.verdict for c in cases] == SS.evaluate(st, cases)
    assert cases[u].verdict == want, what
    assert all(c.verdict == SS.ACCEPT for j, c in enumerate(cases) if j != u)


def test_a_row_dropped_or_moved_is_refused(x):
    u = 6
    dropped = [s if i != u else s[:1] for i, s in enumerate(x.rows)]
    swapped = [s if i != u else tuple((k, *r[1:]) for k, r in enumerate(reversed(s))) for i, s in enumerate(x.rows)]
    for window in (dropped, swapped):
        _st, cases = _cases(x, window, 1)
        assert cases[u].verdict != SS.ACCEPT


def test_the_circuit_is_check_on_any_rows(x):
    rnd = random.Random(7)
    st = SS.statics(x.S, G=1)
    T = SimpleNamespace(**st)
    out = []
    for j in range(x.S.P):
        c = SS.case(T, SR.block_bits(x.S, x.rows, 1, j), x.programs[j:j + 1])
        for _ in range(4):
            r = {k: list(v) for k, v in c.rows.items()}
            for _flip in range(rnd.randrange(1, 3)):
                key = rnd.choice(("pub", "srb"))
                r[key][rnd.randrange(len(r[key]))] ^= 1
            flags = SS.check(T, r)
            out.append(SS.Case(r, flags, SS.verdict(flags)))
    assert SS.evaluate(st, out) == [c.verdict for c in out]
    assert {c.verdict for c in out} >= {SS.ACCEPT, SS.ACCEPT & ~COUNT, SS.ACCEPT & ~POS}


def test_an_idle_step_is_held_zero(x):
    """An idle step's Program has no row, so its served rows must be zero: a row registered there is refused."""
    st = SS.statics(x.S, G=1)
    T = SimpleNamespace(**st)
    idle = SS.case(T, SR.block_bits(x.S, [None], 1, 0), [None])
    filled = SS.case(T, SR.block_bits(x.S, [x.rows[1]], 1, 0), [None])
    assert SS.evaluate(st, [idle, filled]) == [idle.verdict, filled.verdict] == [SS.ACCEPT, 0]


def test_a_block_is_its_leaf_of_the_registration(x):
    """A row a block: leaf w·(P / G) + j, its b‖c on its path to the root under the run root, and at G = P the window's
    row, the slice's ``srv``; the Lean verifier's port names the leaves, the rows' words and the domain."""
    S = x.S
    two = [x.rows, list(x.rows[:3]) + [None] * (S.P - 3)]
    window = SR.register(S, two, RUN_ROOT)
    assert window.registration.rows == SR.register(S, two, RUN_ROOT, G=S.P).registration.rows
    assert window.registration.rows[0] == SR._row(CS.case(S, x.sl, BOOK).rows["srv"])[0]
    served = SR.register(S, two, RUN_ROOT, G=1)
    rec = served.record()
    assert rec["count"] == 2 * S.P and rec["G"] == 1 and window.record()["G"] == S.P
    reg = served.registration
    for w, j in ((0, 0), (0, S.P - 1), (1, 2), (1, 3)):
        i = SR.leaf(S, rec, w, j)
        assert i == w * S.P + j and reg.rows[i] == SR._row(SR.block_bits(S, two[w], 1, j))[0]
        assert SR.leaf_ok(rec, RUN_ROOT, i, served.bc(i), reg.open(i).path)
        assert not SR.leaf_ok(rec, RUN_ROOT, i ^ 1, served.bc(i), reg.open(i).path)
    port = SR.reads_port(rec, RUN_ROOT, [0, None, 9], reads_from=4)
    assert port["leaves"] == 2 * S.P and port["words"] * 16 == rec["n_bits"] and port["reads"] == [0, None, 9]
    assert port["root"] == rec["root"] and port["row_schema"] == "hm96-sha512/row/v2" and port["reads_from"] == 4
    dom = ADV.domain(SR.binding(RUN_ROOT), rec["count"])
    assert dom.domain_id.hex() == port["domain"]
    for i in (0, 9, 2 * S.P - 1):
        path = b"".join(reg.open(i).path)
        assert REG.verify_read(dom, bytes.fromhex(port["root"]), i, served.bc(i), path, port["row_schema"])
        assert not REG.verify_read(dom, bytes.fromhex(port["root"]), i ^ 1, served.bc(i), path, port["row_schema"])
    with pytest.raises(ValueError, match="another run root"):
        SR.reads_port(rec, bytes(64), [0])
    with pytest.raises(ValueError, match="G dividing P"):
        SR.register(S, two, RUN_ROOT, G=2)


def test_statics_it_cant_pack_are_refused(x):
    T = SimpleNamespace(**SS.statics(x.S, G=1))
    with pytest.raises(ValueError, match="over R"):
        SS.statics_bits(T, [[(0, 1)] * (x.S.R + 1)])
    with pytest.raises(ValueError, match="doesn't fit"):
        SS.statics_bits(T, [[(1 << ST._step_widths(T).LW, 1)]])
    with pytest.raises(ValueError, match="over a block"):
        SS.statics_bits(T, [None, None])
