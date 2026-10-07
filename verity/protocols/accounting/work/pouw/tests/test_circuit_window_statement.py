"""The window statement in gates (`circuit.window_statement`): the word primitive is `window.window_problem` on P entry rows,
ℓ and the pass's -h3 tile root, and its bit form equals it on an honest toy window and on one control per refusal of
`window_problem`, each a window the honest one becomes under one change."""
from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest

from verity.primitives.circuits.evaluation import evaluate
from verity.primitives.circuits.evaluation.bits import evaluate_words
from verity.protocols.accounting.work.pouw import window as Wd
from verity.protocols.accounting.work.pouw.circuit import window_statement as WS

P = WS.P


def _d(n: int) -> bytes:
    return bytes([n]) * 64


#: the pass's -h3 tile root every honest call names
H3 = _d(60)


def _window(real_a=(3, 2), rows_b=4, filler=(1, 2), m=16, cols=1):
    """Calls of m x k x 64·cols at pass tiles 0, ⌈m/64⌉·cols, ..., then the filler."""
    out, at = [], 0
    for c, a in enumerate(real_a):
        out.append(Wd.Entry("call", at, at + a * rows_b, a, rows_b, _d(2 * c + 1), _d(2 * c + 2), H3, _d(100 + c),
                            tile0=c * -(-m // 64) * cols, cols=cols, m=m))
        at += a * rows_b
    work = Wd.bucket(at)
    out.append(Wd.Entry("filler", at, work.n, *filler, _d(50), _d(51)))
    return [x.row() for x in out + [Wd.Entry("none")] * (P - len(out))], work.level


def _big():
    """15 calls of 2^13 x 2^17 tiles each, the most an entry holds, each of m = 2^13 and cols = 2^11 (2^18 pass tiles), the
    last ending at pass tile 2^40, and the filler: level 254, N = 2^34."""
    call, span = 1 << 30, 1 << 18
    out = [Wd.Entry("call", c * call, (c + 1) * call, 1 << 13, 1 << 17, _d(1), _d(2), H3, _d(3), tile0=(1 << 40) - (P - 1 - c) * span,
                    cols=1 << 11, m=1 << 13) for c in range(P - 1)]
    work = Wd.bucket((P - 1) * call)
    fill = work.n - (P - 1) * call
    out.append(Wd.Entry("filler", (P - 1) * call, work.n, 1 << 13, -(-fill // (1 << 13)), _d(4), _d(5)))
    return [x.row() for x in out], work.level


def _set(rows, e, **fields):
    return rows[:e] + [replace(Wd.Entry.from_row(rows[e]), **fields).row()] + rows[e + 1:]


def _bytes(rows, e, at, value: bytes):
    return rows[:e] + [rows[e][:at] + value + rows[e][at + len(value):]] + rows[e + 1:]


ROWS, LEVEL = _window()
NONE = Wd.Entry("none").row()
#: name: (rows, level, the refusal of `window_problem` it is a control of, None for a window the statement holds on[, the
#: pass's root when it is not H3])
CASES = {
    "honest": (ROWS, LEVEL, None),
    "big": (*_big(), None),
    "sides at 2^D": (_set(ROWS, 2, n_a=1 << 13, n_b=1 << 17), LEVEL, None),
    "kind 3": (_bytes(ROWS, 0, 0, b"\3"), LEVEL, "not an entry row: kind 3"),
    "pad word": (_bytes(ROWS, 1, 47, b"\x80"), LEVEL, "bytes 40..48 are not zero"),
    "second pad word": (_bytes(ROWS, 0, 271, b"\x01"), LEVEL, "bytes 264..272 are not zero"),
    "filler tile0": (_set(ROWS, 2, tile0=1), LEVEL, "the filler's tile0, cols and m are not zero"),
    "filler cols": (_set(ROWS, 2, cols=1), LEVEL, "the filler's tile0, cols and m are not zero"),
    "filler m": (_set(ROWS, 2, m=1 << 63), LEVEL, "the filler's tile0, cols and m are not zero"),
    "empty entry m": (_set(ROWS, P - 1, m=16), LEVEL, "an empty entry is not the zero row"),
    "m 0": (_set(ROWS, 0, m=0), LEVEL, "its m 0 is not"),
    "m 24": (_set(ROWS, 0, m=24), LEVEL, "its m 24 is not a multiple of 16"),
    "m 2^13 + 16": (_set(ROWS, 0, m=(1 << 13) + 16), LEVEL, "its m 8208"),
    "m 2^64 − 16": (_set(ROWS, 0, m=(1 << 64) - 16), LEVEL, "its m"),
    "m at n_a": (*_window(real_a=(16, 2), filler=(4, 8)), None),
    "m below n_a": (*_window(real_a=(17, 2), filler=(4, 8)), "from n_a = 17"),
    "cols 0": (_set(ROWS, 0, cols=0), LEVEL, "its cols 0 is not"),
    "cols 2^11 + 1": (_set(ROWS, 0, cols=(1 << 11) + 1), LEVEL, "its cols 2049"),
    "cols 2^64 − 1": (_set(ROWS, 0, cols=(1 << 64) - 1), LEVEL, "its cols"),
    "n_b at 64·cols": (*_window(rows_b=64, filler=(4, 8)), None),
    "n_b beyond 64·cols": (*_window(rows_b=65, filler=(4, 8)), "64·cols >= n_b = 65"),
    "n_b within 64·cols": (*_window(rows_b=65, cols=2, filler=(4, 8)), None),
    "pass tiles to 2^40": (_set(ROWS, 1, tile0=(1 << 40) - 1), LEVEL, None),
    "pass tiles past 2^40": (_set(ROWS, 1, tile0=1 << 40), LEVEL, "its pass tiles [1099511627776"),
    "tile0 2^64 − 1": (_set(ROWS, 1, tile0=(1 << 64) - 1), LEVEL, "its pass tiles"),
    "pass tiles overlap": (_set(ROWS, 1, tile0=0), LEVEL, "entry 1: its pass tiles start at 0, before the last call's end 1"),
    "pass tiles out of order": (_set(_set(ROWS, 0, tile0=5), 1, tile0=4), LEVEL, "entry 1: its pass tiles start at 4"),
    "pass tiles apart": (_set(ROWS, 1, tile0=7), LEVEL, None),
    "hi 2^40": (_set(ROWS, 2, hi=1 << 40), LEVEL, "is not of 40-bit indices"),
    "empty entry": (_set(ROWS, P - 1, root_b=_d(1)), LEVEL, "an empty entry is not the zero row"),
    "side a 0": (_set(ROWS, 0, n_a=0), LEVEL, "side a holds 0 rows"),
    "side a 2^13+1": (_set(ROWS, 2, n_a=(1 << 13) + 1), LEVEL, "side a holds 8193 rows"),
    "side b 2^17+1": (_set(ROWS, 2, n_b=(1 << 17) + 1), LEVEL, "side b holds 131073 rows"),
    "filler record": (_set(ROWS, 2, record=_d(1)), LEVEL, "the filler's record is not zero"),
    "filler root_h3": (_set(ROWS, 2, root_h3=H3), LEVEL, "the filler's -h3 root is not zero"),
    "empty entry root_h3": (_set(ROWS, P - 1, root_h3=H3), LEVEL, "an empty entry is not the zero row"),
    "call of another pass": (_set(ROWS, 1, root_h3=_d(61)), LEVEL, "entry 1: its -h3 root is not the pass's"),
    "another pass": (ROWS, LEVEL, "entry 0: its -h3 root is not the pass's", _d(61)),
    "no call": ([_set(ROWS, 2, lo=0)[2]] + [NONE] * (P - 1), 0, "the calls, then one filler"),
    "no filler": (ROWS[:2] + [NONE] * (P - 2), LEVEL, "the calls, then one filler"),
    "two fillers": (ROWS[:3] + [ROWS[2]] + ROWS[4:], LEVEL, "the calls, then one filler"),
    "call after filler": (ROWS[:3] + [ROWS[0]] + ROWS[4:], LEVEL, "the calls, then one filler"),
    "gap": (_set(ROWS, 1, lo=13), LEVEL, "entry 1: its range does not start where the last ended"),
    "empty range": (_set(ROWS, 0, hi=0), LEVEL, "entry 0: its range does not start where the last ended, or is empty"),
    "short call": (_set(ROWS, 0, hi=11), LEVEL, "its range is 11 tiles, not its 12 tiles"),
    "filler beyond": (_window(filler=(1, 1))[0], LEVEL, "the filler's range is 2 tiles, beyond its 1"),
    "level + 1": (ROWS, LEVEL + 1, "not the padded count 24"),
    "level 65535": (ROWS, 65535, "not the padded count"),
    "real below credited": (_set(_window(real_a=(2, 2))[0], 2, hi=22, n_a=2, n_b=3), LEVEL,
                            "16 real tiles are level 15, not the registered level 17"),
}


def _words(rows) -> list[int]:
    return [w for row in rows for w in struct.unpack(f"<{Wd.ENTRY_WORDS}H", row)]


def _root(case) -> bytes:
    return case[3] if len(case) > 3 else H3


def test_each_control_is_one_refusal_of_the_window_statement():
    for name, case in CASES.items():
        rows, level, why = case[:3]
        got = Wd.window_problem(rows, Wd.Work.at(level), P, _root(case))
        assert (got is None) if why is None else (got is not None and why in got), (name, got)


def test_the_word_primitive_is_the_window_statement():
    for name, case in CASES.items():
        rows, level, why = case[:3]
        root = list(struct.unpack("<32H", _root(case)))
        assert evaluate(WS.PouwWindowStatement, _words(rows), level, root) == (int(why is None),), name


def test_the_gates_are_the_word_primitive():
    """The bit form on every control, through its word view, in one bit-sliced pass."""
    names = list(CASES)
    words = np.array([np.array(_words(CASES[k][0]), dtype=np.uint16).reshape(P, Wd.ENTRY_WORDS) for k in names])
    levels = np.array([CASES[k][1] for k in names], dtype=np.uint16)
    roots = np.array([np.frombuffer(_root(CASES[k]), dtype="<u2") for k in names])
    got = evaluate_words(WS.PouwWindowStatementBits, words, levels, roots).reshape(-1)
    assert dict(zip(names, got.tolist())) == {k: int(CASES[k][2] is None) for k in names}


@pytest.mark.parametrize("level", [0, 13, 14, 15, 16, 22, 23, 254, 263, 264, 303, 304, 305])
def test_the_grid_points_in_gates_are_the_grids(level):
    """`point` in gates is `window.point` below 2^40, and says when a point is not."""
    from verity.primitives.circuits.boolean.forms import Circuit, const_bits

    C = Circuit()
    fits, bits = WS.point(C, const_bits(level, 17))
    value = sum(b.c << i for i, b in enumerate(bits))
    assert fits.c == (Wd.point(level) < 1 << Wd.T_BITS)
    if fits.c:
        assert value == Wd.point(level)
