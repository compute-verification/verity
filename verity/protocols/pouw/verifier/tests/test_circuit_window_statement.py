"""`circuit.window_statement`: `PouwWindowStatement_v1{P, S}` on bits is `window_ok` on words, which is
`window.window_problem` on the entry rows at the public level and the public -h3 root: ok = 1 on an honest execution
phase's rows, 0 on rows broken in each way `window_problem` names, and the statement Program is its one unit."""
from __future__ import annotations

import random
import struct
from dataclasses import replace

import numpy as np
import pytest

from verity.circuits.boolean.forms import Circuit
from verity.circuits.evaluation import evaluate
from verity.protocols.pouw.prover.circuit import window_statement as WS
from verity.protocols.pouw.verifier import window as Wd
from verity.protocols.spot.verifier.one_stage import partition as PT

A_MAX, B_MAX = 1 << Wd.DEPTHS["a"], 1 << Wd.DEPTHS["b"]
TOP = 1 << Wd.T_BITS


def _d(n: int) -> bytes:
    return bytes([n]) * 64


def _root(d: bytes) -> list[int]:
    return list(struct.unpack(f"<{WS.ROOT_WORDS}H", d))


#: the pass's -h3 tile root every call's entry names
H3 = _d(60)
ROOT = _root(H3)


def _phase(calls, P, S=Wd.GRID_BITS, spare=0):
    """The honest execution phase of ``calls``, each ``(n_a, n_b, m, cols, tile0)``: their entries in order, the filler from
    the real tiles to the next grid point (its sides holding ``spare`` rows more than it needs), then empty entries up to
    P; with its level."""
    es, at = [], 0
    for c, (n_a, n_b, m, cols, tile0) in enumerate(calls):
        es.append(Wd.Entry("call", at, at + n_a * n_b, n_a, n_b, _d(2 * c + 1), _d(2 * c + 2), H3, _d(100 + c), tile0=tile0,
                           cols=cols, m=m))
        at += n_a * n_b
    work = Wd.bucket(at, S)
    f_b = min(work.n - at, B_MAX)
    es.append(Wd.Entry("filler", at, work.n, -(-(work.n - at) // f_b) + spare, f_b, _d(50), _d(51)))
    return es + [Wd.Entry("none")] * (P - len(es)), work.level


def _honest(rng, P, S=Wd.GRID_BITS):
    """A random honest execution phase of 1 to P - 1 calls, their sides, m, cols and pass tiles often at the layout's edges."""
    while True:
        calls, tile = [], rng.choice([0, 0, 7, TOP // 2])
        for _ in range(rng.randrange(1, P)):
            n_a = rng.choice([1, 1, 2, 3, 16, 17, 64, A_MAX, rng.randrange(1, A_MAX + 1)])
            n_b = rng.choice([1, 1, 2, 5, 64, 65, B_MAX, rng.randrange(1, B_MAX + 1)])
            m = min(A_MAX, Wd.ROW_STEP * (-(-n_a // Wd.ROW_STEP) + rng.choice([0, 0, 1, 5])))
            cols = min(B_MAX // Wd.TILE, -(-n_b // Wd.TILE) + rng.choice([0, 0, 1, 9]))
            tile += rng.choice([0, 0, 1, 1000])
            calls.append((n_a, n_b, m, cols, tile))
            tile += -(-m // Wd.TILE) * cols
        real = sum(c[0] * c[1] for c in calls)
        if Wd.bucket(real, S).n - real <= A_MAX * B_MAX // 2:
            return _phase(calls, P, S, spare=rng.choice([0, 0, 1]))


def _words(es):
    return [e.words() for e in es]


def _mutate(rng, rows, level, root):
    """One edit of an execution phase: of the level, of the public root, of a row's bit or kind, of the rows' order, or of
    one of an entry's integers, to a value beside its own or at one of the layout's edges."""
    rows, root = [list(r) for r in rows], list(root)
    e, k = rng.randrange(len(rows)), rng.randrange(10)
    if k == 0:
        level = rng.choice([min(level + 1, (1 << 16) - 1), max(level - 1, 0), rng.randrange(1 << 16)])
    elif k == 1:
        root[rng.randrange(len(root))] ^= 1 << rng.randrange(16)
    elif k == 2:
        rows[e][rng.randrange(Wd.ENTRY_WORDS)] ^= 1 << rng.randrange(16)
    elif k == 3:
        rows[e][0] = rng.randrange(8)
    elif k == 4:
        other = rng.randrange(len(rows))
        rows[e], rows[other] = rows[other], rows[e]
    elif k == 5:
        rows[e] = list(rows[rng.randrange(len(rows))])
    else:
        try:
            x = Wd.Entry.from_words(rows[e])
        except ValueError:
            return rows, level, root
        name = rng.choice(["lo", "hi", "n_a", "n_b", "tile0", "cols", "m"])
        v = getattr(x, name)
        v = rng.choice([0, v + 1, max(v - 1, 0), v + Wd.ROW_STEP, v ^ 1 << rng.randrange(64), TOP, TOP - 1, A_MAX + 1,
                        B_MAX + 1])
        rows[e] = replace(x, **{name: v % (1 << 64)}).words()
    return rows, level, root


def _lanes(x: int, T: int) -> np.ndarray:
    """A wire's T cases, one bit each."""
    return np.unpackbits(np.frombuffer(x.to_bytes(-(-T // 8), "little"), np.uint8), bitorder="little")[:T]


def _gates(P, S, cases):
    """Each case's ok from the unit's gate circuit (`traced`), every case evaluated at once: a wire's value is an integer
    holding one bit a case."""
    tr, T = WS.traced(P, S), len(cases)
    words = np.array([[w for r in rows for w in r] + [level] + list(root) for rows, level, root in cases], dtype="<u2")
    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")
    assert bits.shape[1] == tr.n_inputs
    ones = (1 << T) - 1
    val = [0, ones] + [int.from_bytes(np.packbits(bits[:, i], bitorder="little").tobytes(), "little")
                       for i in range(tr.n_inputs)]
    for kind, ops in tr.gates:
        a = val[ops[0]]
        val.append(a ^ ones if kind == "not" else a & val[ops[1]] if kind == "and" else a ^ val[ops[1]])
    (ok,) = tr.outputs
    return _lanes(val[ok], T).tolist()


def _word(P, S, cases):
    assert all(len(rows) == P for rows, _level, _root in cases)
    return [WS.window_ok(rows, level, root, S) for rows, level, root in cases]


def _b(v, w=16):
    return [(v >> i) & 1 for i in range(w)]


def _args(rows, level, root):
    return [x for r in rows for w in r for x in _b(w)], _b(level), [x for w in root for x in _b(w)]


@pytest.mark.parametrize("P, S", [(2, 3), (3, 3), (6, 3), (4, 0), (3, 8), (16, 3)])
def test_bits_are_the_word(P, S):
    """On honest execution phases and on ones a few edits from honest, the gates' ok is `window_problem`'s verdict. At
    16 entries, the served bound, this is the comparison with the reference: the unit is past the ANDs circuit-check
    lowers a whole Definition within, so it compares no realization there."""
    rng, cases = random.Random(100 * P + S), []
    for i in range(240):
        es, level = _honest(rng, P, S)
        case = (_words(es), level, ROOT)
        assert WS.window_ok(*case, S) == 1
        for _ in range(i % 4):
            case = _mutate(rng, *case)
        cases.append(case)
    want = _word(P, S, cases)
    assert _gates(P, S, cases) == want
    assert 60 <= sum(want) < len(want) - 60


def _base():
    """P = 5 at level `bucket` (20): calls of 3 x 4 and 2 x 4 tiles at pass tiles 0 and 1, the filler's two tiles, two
    empty entries."""
    es, level = _phase([(3, 4, 16, 1, 0), (2, 4, 16, 1, 1)], 5)
    assert (level, Wd.Work.at(level).n) == (Wd.bucket(20).level, 22)
    return es, level


def _raw(e, word, value):
    def edit(es, level, root):
        rows = _words(es)
        rows[e][word] = value
        return rows, level, root
    return edit


def _entry(e, **fields):
    return lambda es, level, root: (_words(es[:e] + [replace(es[e], **fields)] + es[e + 1:]), level, root)


def _these(build):
    return lambda es, level, root: (_words(build(es)), level, root)


def _level(to):
    return lambda es, level, root: (_words(es), to(level), root)


def _built(calls, keep=5):
    """The execution phase `_phase` makes of ``calls`` at its own level, its first ``keep`` entries: what the calls' own
    fields break is all that is broken."""
    def edit(es, level, root):
        made, at = _phase(calls, max(keep, len(calls) + 1))
        return _words(made[:keep]), at, root
    return edit


#: the largest call's pass tiles
FULL = (A_MAX // Wd.TILE) * (B_MAX // Wd.TILE)


#: an execution phase broken in each way `window_problem` names: the reason's words, and the edit of `_base` that reaches it
BREAKS = (
    ("not an entry row: kind 7", _raw(0, 0, 7)),
    ("not an entry row: kind 3", _raw(1, 0, 3)),
    ("not an entry row: kind 5", _raw(0, 0, 5)),
    ("not an entry row: kind 6", _raw(2, 0, 6)),
    ("not an entry row: kind 4", _raw(3, 0, 4)),
    ("not an entry row: kind 65536", _raw(4, 1, 1)),
    ("not an entry row: kind 65537", _raw(0, 1, 1)),
    (f"not an entry row: kind {2 + (1 << 63)}", _raw(2, 3, 1 << 15)),
    ("its bytes 40..48 are not zero", _raw(0, 20, 1)),
    ("its bytes 264..272 are not zero", _raw(1, 135, 1 << 15)),
    ("entry 2: not the layout: its range", _entry(2, hi=TOP)),
    ("entry 0: not the layout: its range", _entry(0, lo=1 << 63)),
    ("an empty entry is not the zero row", _entry(4, n_a=1)),
    ("an empty entry is not the zero row", _entry(3, record=_d(1))),
    ("side a holds 0 rows", _entry(0, n_a=0)),
    ("side a holds 0 rows", _built([(0, 4, 16, 1, 0), (5, 4, 16, 1, 1)])),                            # a call of no tiles
    ("side b holds 0 rows", _built([(5, 0, 16, 1, 0), (5, 4, 16, 1, 1)])),
    (f"side a holds {A_MAX + 1} rows", _entry(2, n_a=A_MAX + 1)),
    ("side b holds 0 rows", _entry(2, n_b=0)),
    (f"side b holds {B_MAX + 1} rows", _entry(1, n_b=B_MAX + 1)),
    (f"side b holds {(1 << 32) + 4} rows", _entry(1, n_b=(1 << 32) + 4)),
    ("the filler's -h3 root is not zero", _entry(2, root_h3=H3)),
    ("the filler's record is not zero", _entry(2, record=_d(9))),
    ("the filler's tile0, cols and m are not zero", _entry(2, tile0=1)),
    ("the filler's tile0, cols and m are not zero", _entry(2, cols=1)),
    ("the filler's tile0, cols and m are not zero", _entry(2, m=16)),
    ("its m 8 is not a multiple", _entry(0, m=8)),
    ("its m 0 is not a multiple", _entry(0, m=0)),
    ("its m 16 is not a multiple of 16 from n_a = 17", _entry(0, n_a=17)),
    ("its m 16 is not a multiple of 16 from n_a = 17", _built([(17, 4, 16, 1, 0)])),
    ("its m 8 is not a multiple", _built([(3, 4, 8, 1, 0)])),
    (f"its m {A_MAX + 16} is not a multiple", _built([(3, 4, A_MAX + 16, 1, 0)])),
    (f"its cols {B_MAX // 64 + 1} is not", _built([(3, 4, 16, B_MAX // 64 + 1, 0)])),
    ("with 64·cols >= n_b = 65", _built([(3, 65, 16, 1, 0)])),
    (f"its m {A_MAX + 16} is not a multiple", _entry(1, m=A_MAX + 16)),
    ("its cols 0 is not", _entry(0, cols=0)),
    (f"its cols {B_MAX // 64 + 1} is not", _entry(0, cols=B_MAX // 64 + 1)),
    ("with 64·cols >= n_b = 65", _entry(0, n_b=65)),
    (f"its pass tiles [{TOP - 1}, {TOP + 1})", _entry(1, tile0=TOP - 1, cols=2)),
    (f"its pass tiles [{1 << 63}", _entry(1, tile0=1 << 63)),
    ("the entries are the calls, then one filler", _these(lambda es: es[:3] + [es[0]] + es[4:])),     # a call after the filler
    ("the entries are the calls, then one filler",                                                    # ... from its end
     _these(lambda es: es[:3] + [replace(es[0], lo=22, hi=34, tile0=5)] + es[4:])),
    ("the entries are the calls, then one filler", _these(lambda es: es[:4] + [es[0]])),              # ... after an empty one
    ("the entries are the calls, then one filler", _these(lambda es: es[:3] + [es[2]] + es[4:])),     # two fillers
    ("the entries are the calls, then one filler",                                                    # a phase twice
     _these(lambda es: (lambda w: w + w[:2])(_phase([(5, 4, 16, 1, 0)], 3)[0]))),
    ("the entries are the calls, then one filler", _these(lambda es: [es[2]] + es[1:])),              # the filler first
    ("the entries are the calls, then one filler", _these(lambda es: es[:2] + es[3:] + es[3:4])),     # no filler
    ("the entries are the calls, then one filler", _these(lambda es: es[:2] + [es[3], es[2], es[4]])),  # an empty entry inside
    ("the entries are the calls, then one filler", _these(lambda es: es[:2] + [es[1]] * 3)),          # calls to the bound
    ("the entries are the calls, then one filler", _built([(1, 1, 16, 1, c) for c in range(5)])),     # ... in order
    ("the entries are the calls, then one filler", _these(lambda es: [es[3]] * 5)),                   # empty entries alone
    ("entry 1: its range does not start where the last ended", _entry(1, lo=13)),
    ("entry 0: its range does not start where the last ended", _entry(0, lo=1, hi=13)),
    ("entry 2: its range does not start where the last ended", _entry(2, lo=19)),
    ("entry 0: its range does not start where the last ended, or is empty", _entry(0, hi=0)),
    ("entry 2: its range does not start where the last ended, or is empty", _entry(2, hi=20)),
    ("entry 2: its range does not start where the last ended, or is empty",                           # the calls reach N
     _these(lambda es: [es[0], replace(es[1], hi=22, n_b=5), replace(es[2], lo=22)] + es[3:])),
    ("entry 2: its range does not start where the last ended, or is empty",                           # ... and pass it
     _these(lambda es: [es[0], replace(es[1], hi=24, n_a=3), replace(es[2], lo=24)] + es[3:])),
    ("entry 0: its range does not start where the last ended",                                        # the ranges from 2
     lambda es, level, root: (_words([replace(es[0], lo=2, hi=14), replace(es[1], lo=14, hi=22),
                                      replace(es[2], lo=22, hi=24)] + es[3:]), Wd.bucket(22).level, root)),
    ("its range is 11 tiles, not its 12 tiles", _entry(0, hi=11)),
    ("its range is 9 tiles, not its 8 tiles", _entry(1, hi=21)),
    ("entry 1: its -h3 root is not the pass's", _entry(1, root_h3=_d(61))),
    ("entry 0: its -h3 root is not the pass's", lambda es, level, root: (_words(es), level, _root(_d(61)))),
    ("its pass tiles start at 0, before the last call's end 1", _entry(1, tile0=0)),
    (f"its pass tiles start at 0, before the last call's end {TOP}",
     _built([(A_MAX, B_MAX, A_MAX, B_MAX // Wd.TILE, TOP - FULL), (1, 1, 16, 1, 0)])),
    ("the filler's range is 2 tiles, beyond its 1", _entry(2, n_a=1, n_b=1)),
    ("the ranges end at 22, not the padded count 24", _level(lambda level: level + 1)),
    ("the ranges end at 22, not the padded count 20", _level(lambda level: level - 1)),
    ("the ranges end at 22, not the padded count 2", _level(lambda level: 0)),
    ("the ranges end at 22, not the padded count", _level(lambda level: (1 << 16) - 1)),
    ("the ranges end at 21, not the padded count 22", _entry(2, hi=21)),
    ("16 real tiles are level", _these(lambda es: [es[0], replace(es[1], lo=12, hi=16, n_a=1),
                                                  replace(es[2], lo=16, n_a=2, n_b=3)] + es[3:])),
)
#: the reasons `window_problem` gives on entry rows alone, each by words only it holds
REASONS = ("not an entry row: kind", "40..48", "264..272", "not the layout: its range", "an empty entry", "side a holds",
           "side b holds", "the filler's -h3 root", "the filler's record", "the filler's tile0", "its m ", "its cols ",
           "its pass tiles [", "the entries are the calls", "does not start where", "tiles, not its",
           "its -h3 root is not the pass's", "its pass tiles start at", "the filler's range is", "the ranges end at",
           "real tiles are level")


def test_an_execution_phase_broken_in_each_way_the_statement_names_is_refused():
    """Each of `window_problem`'s reasons on entry rows is reached by an edit of an honest execution phase, and the gates
    refuse it."""
    es, level = _base()
    cases, got = [(_words(es), level, ROOT)], []
    for want, edit in BREAKS:
        rows, lvl, root = edit(es, level, ROOT)
        why = Wd.window_problem([struct.pack(f"<{Wd.ENTRY_WORDS}H", *r) for r in rows], Wd.Work.at(lvl), 5,
                                struct.pack(f"<{WS.ROOT_WORDS}H", *root))
        assert why and want in why, (want, why)
        got.append(why)
        cases.append((rows, lvl, root))
    assert all(any(r in why for why in got) for r in REASONS)
    assert _word(5, Wd.GRID_BITS, cases) == _gates(5, Wd.GRID_BITS, cases) == [1] + [0] * len(BREAKS)


def _at_the_edges():
    """Execution phases at the layout's edges and one past them: ``(P, entries, level, edit or None, each case's ok)``."""
    full, tiles = (A_MAX, B_MAX, A_MAX, B_MAX // Wd.TILE, 0), FULL
    on_point = _phase([(5, 4, 16, 1, 0)], 2)
    return (
        (2, *_phase([full], 2), None, [1]),                                               # the largest call, 2^30 tiles
        (2, *_phase([(*full[:4], TOP - tiles)], 2), _entry(0, tile0=TOP - tiles + 1), [1, 0]),   # pass tiles ending at 2^40
        (3, *_phase([full, (*full[:4], tiles)], 3), _entry(1, tile0=tiles - 1), [1, 0]),   # ... starting where the last ended
        (3, *_phase([full, (1, 1, 16, 1, tiles + 5)], 3), None, [1]),                      # ... and after a gap
        (2, *_phase([(16, 64, 16, 1, 9)], 2), _entry(0, n_b=65, hi=16 * 65), [1, 0]),      # n_a = m and n_b = 64·cols
        (2, *_phase([(3, 7, 16, 1, 0)], 2), None, [1]),                                    # 21 real tiles, the last before 22
        (2, *on_point, None, [1]),                                                         # 20 real tiles, on their grid point
        (2, *_phase([(5, 4, 16, 1, 0)], 2, spare=3), None, [1]),                           # a filler of rows to spare
        (2, on_point[0], on_point[1] - 1, None, [0]),                                      # the level below the real tiles'
        (2, on_point[0], on_point[1] + 1, None, [0]),                                      # ... and the one above
    )


def test_the_layouts_edges_are_inside_and_one_past_them_is_refused():
    for P, es, level, edit, want in _at_the_edges():
        cases = [(_words(es), level, ROOT)] + ([edit(es, level, ROOT)] if edit else [])
        assert _gates(P, Wd.GRID_BITS, cases) == _word(P, Wd.GRID_BITS, cases) == want


@pytest.mark.parametrize("S", [0, 1, Wd.GRID_BITS, WS.S_MAX])
def test_the_grid_point_in_gates_is_point_at_every_level(S):
    """`_point` is `window.point` at every 17-bit level (a u16 level ℓ, and ℓ + 1) whose point is below 2^40, and says so
    of every other."""
    C, bits = Circuit(), 17
    ell = C.input("ell", bits)
    value, fits = WS._point(C, ell, S)
    C.output("value", value)
    C.output("fits", [fits])
    T = 1 << bits
    levels = np.arange(T, dtype="<u4")
    lanes = np.unpackbits(levels.view(np.uint8).reshape(T, 4), axis=1, bitorder="little")
    out, _ = C.evaluate({"ell": [int.from_bytes(np.packbits(lanes[:, j], bitorder="little").tobytes(), "little")
                                 for j in range(bits)]}, T)
    got = sum(_lanes(x, T).astype(np.uint64) << np.uint64(k) for k, x in enumerate(out["value"]))
    points = (Wd.point(t, S) for t in range(T))
    want = np.array([p if p < TOP else TOP for p in points], dtype=np.uint64)
    fit = want < TOP
    assert (_lanes(out["fits"][0], T).astype(bool) == fit).all() and 0 < fit.sum() < T
    assert (got[fit] == want[fit]).all()


def test_the_definition_is_its_gates():
    """`PouwWindowStatement_v1{P, S}` evaluates to the word on an honest execution phase, at another level, under another
    root and with a row edited, and it reads the rows, the level and the root as `PORTS` names them."""
    es, level = _base()
    d = WS.template(5)
    assert d.id == "PouwWindowStatement_v1{P=5,S=3}"
    params, ret = d.signature()
    assert [n for n, _t in params] == list(WS.PORTS) and ret.leaves == 1
    assert [t.leaves for _n, t in params] == [5 * Wd.ENTRY_BITS, 16, 16 * WS.ROOT_WORDS]
    cases = [(_words(es), level, ROOT), (_words(es), level + 1, ROOT), (_words(es), level, _root(_d(61))),
             _entry(1, tile0=0)(es, level, ROOT)]
    assert [evaluate(d, *_args(*c))[0] for c in cases] == _word(5, Wd.GRID_BITS, cases) == [1, 0, 0, 0]


def test_an_entry_rows_fields_are_the_rows():
    """`FIELDS` tiles the entry row in order, and each field is where `window.Entry.row` writes it."""
    assert [at for _n, at, _s in WS.FIELDS] == [sum(s for _n, _at, s in WS.FIELDS[:i]) for i in range(len(WS.FIELDS))]
    assert sum(s for _n, _at, s in WS.FIELDS) == Wd.ENTRY_BYTES
    x = Wd.Entry("call", 3, 5, 7, 11, _d(1), _d(2), _d(3), _d(4), tile0=13, cols=17, m=19)
    row = x.row()
    got = {n: row[at:at + s] for n, at, s in WS.FIELDS}
    ints = {n: int.from_bytes(got[n], "little")
            for n in ("kind", "lo", "hi", "n_a", "n_b", "pad", "tile0", "cols", "m", "pad2")}
    assert ints == {"kind": Wd.KIND_CODE["call"], "lo": 3, "hi": 5, "n_a": 7, "n_b": 11, "pad": 0, "tile0": 13, "cols": 17,
                    "m": 19, "pad2": 0}
    assert [got[n] for n in ("root_a", "root_b", "root_h3", "record")] == [_d(1), _d(2), _d(3), _d(4)]


@pytest.mark.parametrize("P, S", [(1, 3), (0, 3), (True, 3), ("5", 3), (2.0, 3), (5, -1), (5, WS.S_MAX + 1), (5, True),
                                  (5, None)])
def test_p_is_at_least_a_call_and_the_filler_and_s_a_grid_the_gates_cover(P, S):
    with pytest.raises(ValueError, match="the execution phase's entries bound|the grid's bits"):
        WS.template(P, S)


def test_the_word_reads_entry_rows_and_a_root_of_their_words():
    es, level = _base()
    for rows, root in ((_words(es)[:4] + [[0] * 167], ROOT), (_words(es), ROOT[:31])):
        with pytest.raises(ValueError, match="u16 words"):
            WS.window_ok(rows, level, root)


@pytest.mark.parametrize("P, S", [(2, 3), (5, 3), (3, 1)])
def test_the_statement_program_is_its_one_unit(P, S):
    """The statement Program's partition is one unit, the template's one instance on the P rows, the level and the root,
    so it evaluates to the execution phase's ok."""
    prog = WS.program(P, S)
    params, ret = prog.fn.signature()
    assert [n for n, _t in params] == list(WS.PORTS) and ret.leaves == 1
    tp = PT.evaluate(prog, WS.query(P, S))
    assert tp.population == 1
    rng = random.Random(P + S)
    es, level = _honest(rng, P, S)
    for case in ((_words(es), level, ROOT), (_words(es), level + 1, ROOT)):
        assert list(evaluate(prog.fn, *_args(*case))) == [WS.window_ok(*case, S)]
    assert WS.window_ok(_words(es), level, ROOT, S) == 1


def test_a_served_execution_phases_sixteen_entries_are_one_unit():
    """At a served execution phase's bound of 16 entries (`served_commit.ENTRIES`) the unit accepts fifteen calls and the
    filler, and refuses them at the level below."""
    rng = random.Random(16)
    calls, tile = [], 0
    for _ in range(15):
        n_a, n_b = rng.randrange(1, 65), rng.randrange(1, 4097)
        calls.append((n_a, n_b, 64, 64, tile))
        tile += 64
    es, level = _phase(calls, 16)
    cases = [(_words(es), level, ROOT), (_words(es), level - 1, ROOT)]
    assert _gates(16, Wd.GRID_BITS, cases) == _word(16, Wd.GRID_BITS, cases) == [1, 0]
