"""The window statement (L4) in gates: `pouw.window.window_problem` on the window root's P entry rows and the registered
level ℓ, its one input that is not committed (the draw ranges over N = point(ℓ + 1) tiles and credits point(ℓ)).

    PouwWindowStatement_v1  (entries: Array<P, Array<120, Value<16>>>, level: Value<16>) -> Value<1>
                            1 iff `window_problem(rows, Work.at(level), P)` holds, each row its 120 u16 words
    PouwWindowStatement_v2  (entries: Array<P, Array<120, Array<16, Value<1>>>>, level: Array<16, Value<1>>) -> Value<1>
                            the same on bits, traced from `statement` (word view: PouwWindowStatement_v1)

P = 16, the served windows' bound.  An entry row's fields are its u64s at bits 64·f (PoUW's PROTOCOL.md, "The served
audit window").  The bit form decides each of `window_problem`'s conditions and returns their AND:

* each row an entry row of the layout: its kind 0, 1 or 2 (a 62-bit zero test and the two low bits), the zero word at
  bytes 40..48, lo and hi below 2^40, an empty entry the zero row (one OR over its 1,920 bits), a call's or the filler's
  sides 1..2^D rows (n − 1 below 2^D with no borrow), the filler's record zero;
* the kinds: entry 0 a call; entry e > 0 a call or the filler before the first filler, and empty after it; some filler.
  One bit of state per entry, so at most one filler;
* the ranges: each nonempty entry's lo the last hi (0 first) and hi > lo; a call's hi − lo its n_a·n_b (the product of
  n_a's low 14 bits and n_b's low 18, exact once the layout holds), the filler's at most that;
* the work: the last hi is point(ℓ + 1), and point(ℓ) is at most the filler's lo, the calls' tiles.  These two are
  `bucket(real) == Work.at(ℓ)`, since the filler's range is nonempty and so real < N.  A grid point (`point`) is x + 1
  below 15, else (8 + (r mod 8)) · 2^(r div 8 + 1) for r = x − 15; a point at or past 2^40 is no range's end, so a level
  whose N is one fails.

`tests/test_circuit_window_statement.py` holds the controls: an honest toy window and one window per refusal.
"""
from __future__ import annotations

import struct

from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits.boolean.forms import (NOT, ONE, OR, ZERO, add, and_reduce, const_bits, multiply, mux,
                                                      or_reduce, shl_window, sub)
from verity.primitives.circuits.boolean.trace import definition
from verity.primitives.circuits.defs import primitive
from verity.primitives.circuits.types import Array, Value

from .. import window as Wd

#: the served windows' public bound on entries (`benchmarks/pouw/served_zk`'s ``ENTRIES``)
P = 16
WORD = Value(16)
ROW = Array(Wd.ENTRY_WORDS, WORD)
LEVEL = Value(16)
_FIELD = 64
_HEAD = 6


@primitive("PouwWindowStatement", 1, [("entries", Array(P, ROW)), ("level", LEVEL)], Value(1), conformance="protocol-defined")
def PouwWindowStatement(*args: int) -> int:
    """1 iff the window statement holds on the P entry rows (each its 120 u16 words) at level ℓ."""
    *words, level = args
    rows = [struct.pack(f"<{Wd.ENTRY_WORDS}H", *words[e * Wd.ENTRY_WORDS:(e + 1) * Wd.ENTRY_WORDS]) for e in range(P)]
    return int(Wd.window_problem(rows, Wd.Work.at(level), P) is None)


def _eq(C, a, b):
    return and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])


def _zero(C, bits):
    return NOT(or_reduce(C, bits))


def _le(C, a, b, width):
    """a <= b, unsigned."""
    return NOT(sub(C, b, a, width)[1])


def _implies(C, a, b):
    return NOT(C.AND(a, NOT(b)))


def point(C, x):
    """(fits, the 64 low bits of the grid point at level x (`window.point`, s = 3)): fits iff the point is below 2^40."""
    r, below = sub(C, x, const_bits(15, len(x)), len(x))
    e = add(C, r[3:], [ONE], width=len(x) - 2)
    fits = _le(C, e, const_bits(Wd.T_BITS - 4, len(e)), len(e))
    big = shl_window(C, r[:3] + [ONE], e[:6], 0, _FIELD)
    small = add(C, x, [ONE], width=len(x)) + [ZERO] * (_FIELD - len(x))
    return OR(C, below, fits), mux(C, below, small, big)


def statement(C, *words):
    """The verdict bit on the P rows' words and ℓ's (the module docstring's conditions)."""
    *words, level = words
    n = Wd.ENTRY_WORDS
    rows = [[b for w in words[e * n:(e + 1) * n] for b in w] for e in range(P)]
    ok, seen, at, real = [], ZERO, const_bits(0, _FIELD), const_bits(0, _FIELD)
    for e, bits in enumerate(rows):
        kind, lo, hi, n_a, n_b, pad = (bits[_FIELD * f:_FIELD * (f + 1)] for f in range(_HEAD))
        record = bits[_FIELD * _HEAD + 2 * 512:]
        high = _zero(C, kind[2:])
        call = C.AND(high, C.AND(kind[0], NOT(kind[1])))
        filler = C.AND(high, C.AND(NOT(kind[0]), kind[1]))
        none = C.AND(high, NOT(OR(C, kind[0], kind[1])))
        active = call ^ filler
        ok += [high, NOT(C.AND(kind[0], kind[1])), _zero(C, pad), _zero(C, lo[Wd.T_BITS:] + hi[Wd.T_BITS:]),
               _implies(C, none, _zero(C, bits)), _implies(C, filler, _zero(C, record))]
        for rows_, d in ((n_a, Wd.DEPTHS["a"]), (n_b, Wd.DEPTHS["b"])):
            less, borrow = sub(C, rows_, const_bits(1, _FIELD), _FIELD)
            ok.append(_implies(C, active, C.AND(NOT(borrow), _zero(C, less[d:]))))
        ok.append(call if e == 0 else mux(C, seen, [none], [active])[0])
        size, under = sub(C, hi, lo, _FIELD)
        ok.append(_implies(C, active, C.AND(_eq(C, lo, at), C.AND(NOT(under), NOT(_zero(C, size))))))
        tiles = multiply(C, n_a[:Wd.DEPTHS["a"] + 1], n_b[:Wd.DEPTHS["b"] + 1])
        tiles += [ZERO] * (_FIELD - len(tiles))
        ok += [_implies(C, call, _eq(C, size, tiles)), _implies(C, filler, _le(C, size, tiles, _FIELD))]
        at, real, seen = mux(C, active, hi, at), mux(C, filler, lo, real), OR(C, seen, filler)
    level = list(level) + [ZERO]
    fits, padded = point(C, add(C, level, [ONE], width=len(level)))
    _, credited = point(C, level)
    ok += [seen, fits, _eq(C, at, padded), _le(C, credited, real, _FIELD)]
    return [and_reduce(C, ok)]


BIT = BL.BIT
PouwWindowStatementBits = definition(
    "PouwWindowStatement", 2, statement,
    (("entries", Array(P, Array(Wd.ENTRY_WORDS, Array(16, BIT)))), ("level", Array(16, BIT))),
    BIT, word=PouwWindowStatement, conformance=PouwWindowStatement.conformance,
    doc="The window statement (L4) on bits: PouwWindowStatement_v1")


__all__ = ["LEVEL", "P", "PouwWindowStatement", "PouwWindowStatementBits", "ROW", "point", "statement"]
