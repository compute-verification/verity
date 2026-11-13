"""`PouwWindowStatement_v1{P, S}`: an execution phase's statement on its committed entry rows (`window.window_problem`, L4) as
one unit, so the statement is a public Program's and not Python's alone.

    PouwWindowStatement_v1{P, S}   (rows: Array<P, Array<168, U16>>, level: U16, root_h3: Array<32, U16>) -> Value<1>

P is the execution phase's public entries bound and S its grid's bits (`window.GRID_BITS`). ``rows`` is the P entry rows as
they are committed, each its 168 u16 words (`window.Entry.words`): the leaves of the registered execution phase root at port
`window.PORT`, hiding leaves. ``level``, public, is the registered level ℓ (the `work` item). ``root_h3``, public, is
the pass's published -h3 tile root, its 64 bytes as 32 u16 words, low byte first. ok = 1 iff `window.window_problem`
holds on the rows at `window.Work.at` (ℓ), the bound P and that root:

* every row is an entry row of the layout (`window.Entry.from_row`, `window.layout_problem`): its kind one of none, call
  and filler, its pads zero, its range of 40-bit indices; an empty entry the zero row; a call's or the filler's sides 1
  to 2^D rows; the filler's -h3 root, record, tile0, cols and m zero; a call's m in whole 16s up to 2^13 and at least
  n_a, its cols 1 to 2^11 with 64·cols at least n_b, and its pass tiles [tile0, tile0 + ⌈m/64⌉·cols) within 2^40, so
  `window.tile_read`'s leaf is one of the call's own pass tiles and its word a place of that leaf;
* the rows are the calls (at least one), then one filler, then empty entries;
* the ranges tile [0, N) in order, N the grid point at ℓ + 1: the first starts at 0, each next where the last ended, a
  call's is its n_a·n_b tiles, and the filler's is not empty, within its n_a·n_b, and ends at N;
* every call's -h3 root is ``root_h3``;
* each call's pass tiles start at or past the call before's end, so the calls' pass tiles are in order and disjoint;
* the real tiles, where the filler starts, are at least the grid point at ℓ, so ℓ is `window.bucket` of them.

`window_ok` is the unit on words, and it is `window.window_problem` itself on the rows: the statement the committer, the
replay and the Lean port (`PouwWindow.Statement.windowProblem`) hold. `traced(P, S)` is its gate circuit.

It states the entry rows only. `window.window_problem` given the calls' records and the side rows' indices also holds
each call's record to its entry and each side row's index to its record's keep, which makes i -> I and j -> J injective
inside a call: a record is canonical JSON under a SHA-512 and the indices are leaves of the side trees, so that part is
not this unit's, and no Program states it yet.

**The Program** (`program`) is the one unit: `TemplateInstances_v1{template, N = 1}` over `template(P, S)`, its partition
`Q_template_instances` (`query`). It is proved for the execution phase, never sampled. No verifier requires its proof yet:
`core_check.opening` checks one opened entry for each sampled tile, and nothing there reads this Program.
"""
from __future__ import annotations

import functools
import struct
from types import SimpleNamespace
from typing import Any, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import (NOT, ONE, OR, ZERO, add, and_reduce, const_bits, multiply, mux, shl_window,
                                              sub)
from verity.circuits.boolean.trace import Traced, emit, trace
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.partition_object import template_instances_query
from verity.circuits.population import population_program
from verity.circuits.program import Program
from verity.circuits.types import Array, Type

from verity.protocols.pouw.verifier import window as Wd

U16 = Array(16, BL.BIT)
PORTS = ("rows", "level", "root_h3")
#: the public ports: the registered level and the pass's -h3 tile root
PUBLIC = ("level", "root_h3")
#: the -h3 root's u16 words
ROOT_WORDS = 32
#: the grid bits the level's gates cover: a level is a u16, and its point's shift is read in 6 bits
S_MAX = 8
#: an entry row's fields ``(name, first byte, bytes)``, `window.Entry.row`'s
FIELDS = (("kind", 0, 8), ("lo", 8, 8), ("hi", 16, 8), ("n_a", 24, 8), ("n_b", 32, 8), ("pad", 40, 8), ("root_a", 48, 64),
          ("root_b", 112, 64), ("root_h3", 176, 64), ("tile0", 240, 8), ("cols", 248, 8), ("m", 256, 8), ("pad2", 264, 8),
          ("record", 272, 64))
_T, _DA, _DB = Wd.T_BITS, Wd.DEPTHS["a"], Wd.DEPTHS["b"]
#: log2 of a pass tile's side and of the ticket subtiles' rows
_TILE, _STEP = Wd.TILE.bit_length() - 1, Wd.ROW_STEP.bit_length() - 1


def window_ok(rows: Sequence[Sequence[int]], level: int, root_h3: Sequence[int], bits: int = Wd.GRID_BITS) -> int:
    """The unit on words: 1 iff `window.window_problem` holds on the entry rows ``rows`` (each its 168 u16 words) at the
    work level ``level`` of the ``bits``-bit grid, the bound ``len(rows)`` and the pass's -h3 tile root ``root_h3`` (its 32
    u16 words). A level whose padded count is past the 40-bit indices is 0 without the call: no range ends there, and
    `window_problem`'s reason would print the count, which Python refuses past 4,300 digits."""
    if any(len(r) != Wd.ENTRY_WORDS for r in rows) or len(root_h3) != ROOT_WORDS:
        raise ValueError(f"the unit reads entry rows of {Wd.ENTRY_WORDS} u16 words and a root of {ROOT_WORDS}, not rows of "
                         f"{sorted({len(r) for r in rows})} and a root of {len(root_h3)}")
    work = Wd.Work.at(level, bits)
    if work.n >> Wd.T_BITS:
        return 0
    raw = [struct.pack(f"<{Wd.ENTRY_WORDS}H", *r) for r in rows]
    return int(Wd.window_problem(raw, work, len(rows), struct.pack(f"<{ROOT_WORDS}H", *root_h3), bits) is None)


def _types(P: int) -> tuple[list[Type], Type]:
    return [Array(P, Array(Wd.ENTRY_WORDS, U16)), U16, Array(ROOT_WORDS, U16)], BL.BIT


def _zero(C: Any, bits: Sequence[Any]) -> Any:
    return and_reduce(C, [NOT(b) for b in bits])


def _eq(C: Any, a: Sequence[Any], b: Sequence[Any]) -> Any:
    return and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b, strict=True)])


def _le(C: Any, a: Sequence[Any], b: Sequence[Any], width: int) -> Any:
    """a ≤ b, unsigned at ``width`` bits: b - a borrows not."""
    return NOT(sub(C, b, a, width)[1])


def _point(C: Any, ell: Sequence[Any], S: int) -> tuple[list, Any]:
    """`window.point` (ℓ) on the S-bit grid, as its low 40 bits, and whether it is below 2^40: ℓ + 1 below level
    2^(S+1) - 1, else (2^S + r mod 2^S) << (r div 2^S + 1) for r = ℓ - (2^(S+1) - 1). ``ell`` is ℓ's bits."""
    w = len(ell)
    r, small = sub(C, ell, const_bits((2 << S) - 1, w), w)
    shift = add(C, r[S:], [ONE], width=w - S + 1)
    fits = OR(C, small, _le(C, shift, const_bits(_T - 1 - S, len(shift)), len(shift)))
    above = shl_window(C, r[:S] + [ONE], shift[:6], 0, _T)
    below = add(C, ell[:S + 1], [ONE], width=S + 1) + [ZERO] * (_T - S - 1)
    return mux(C, small, below, above), fits


def _entry(C: Any, row: Sequence[Any], root: Sequence[Any], g: Sequence[Any], n: Sequence[Any], n_fits: Any) -> Any:
    """One entry row's gates: ``good``, the row is the zero row, a call of the layout whose range is its tiles and whose
    -h3 root is ``root``, or the filler of the layout whose range is not empty, within its tiles, from at least ``g`` to
    ``n``; with whether it is the zero row or of kind call, its range, whether that starts at 0, and a call's pass tiles'
    start and end."""
    f = {name: row[8 * at:8 * (at + size)] for name, at, size in FIELDS}

    def split(name: str, width: int) -> tuple[list, Any, Any]:
        """A u64 field's low ``width`` bits, whether they are zero, and whether its bits past them are."""
        x = f[name]
        return x[:width], _zero(C, x[:width]), _zero(C, x[width:])

    def count(x: Sequence[Any], none: Any, fits: Any, d: int) -> Any:
        """1 ≤ x ≤ 2^d, for a field whose low d + 1 bits are ``x``: not zero, nothing past them, and no more than 2^d."""
        return and_reduce(C, [NOT(none), fits, NOT(C.AND(x[d], NOT(_zero(C, x[:d]))))])

    kind, _, kind_fits = split("kind", 2)
    lo, lo_0, lo_fits = split("lo", _T)
    hi, hi_0, hi_fits = split("hi", _T)
    n_a, a_0, a_fits = split("n_a", _DA + 1)
    n_b, b_0, b_fits = split("n_b", _DB + 1)
    tile0, t_0, t_fits = split("tile0", _T)
    cols, c_0, c_fits = split("cols", _DB - _TILE + 1)
    m, m_0, m_fits = split("m", _DA + 1)
    pads_0 = C.AND(_zero(C, f["pad"]), _zero(C, f["pad2"]))
    sides_0 = C.AND(_zero(C, f["root_a"]), _zero(C, f["root_b"]))
    h3_0, record_0 = _zero(C, f["root_h3"]), _zero(C, f["record"])
    empty = and_reduce(C, [NOT(kind[0]), NOT(kind[1]), kind_fits, lo_0, lo_fits, hi_0, hi_fits, a_0, a_fits, b_0, b_fits,
                           pads_0, sides_0, h3_0, t_0, t_fits, c_0, c_fits, m_0, m_fits, record_0])
    call = and_reduce(C, [kind[0], NOT(kind[1]), kind_fits])
    filler = and_reduce(C, [NOT(kind[0]), kind[1], kind_fits])

    common = and_reduce(C, [pads_0, lo_fits, hi_fits, count(n_a, a_0, a_fits, _DA), count(n_b, b_0, b_fits, _DB)])
    top = add(C, lo, multiply(C, n_a, n_b), width=_T + 1)                  # lo + n_a·n_b
    tile_rows = add(C, m[_TILE:], [NOT(_zero(C, m[:_TILE]))], width=_DA + 1 - _TILE)   # ⌈m/64⌉
    end = add(C, tile0, multiply(C, tile_rows, cols), width=_T + 1)        # tile0 + ⌈m/64⌉·cols
    call_ok = and_reduce(C, [
        common,
        count(m, m_0, m_fits, _DA), _zero(C, m[:_STEP]), _le(C, n_a, m, _DA + 1),
        count(cols, c_0, c_fits, _DB - _TILE), _le(C, n_b, [ZERO] * _TILE + cols, _DB + 1),
        t_fits, NOT(C.AND(end[_T], NOT(_zero(C, end[:_T])))),
        _eq(C, f["root_h3"], root),
        _eq(C, hi + [ZERO], top)])
    filler_ok = and_reduce(C, [
        common, h3_0, record_0, t_0, t_fits, c_0, c_fits, m_0, m_fits,
        sub(C, lo, hi, _T)[1], _le(C, hi, top, _T + 1),
        n_fits, _eq(C, hi, n), _le(C, g, lo, _T)])
    good = empty ^ C.AND(call, call_ok) ^ C.AND(filler, filler_ok)
    return SimpleNamespace(good=good, empty=empty, call=call, lo=lo, starts=lo_0, hi=hi, tile0=tile0, end=end)


def _build(P: int, S: int) -> Any:
    def build(C: Any, *e: Any) -> list:
        words = Wd.ENTRY_WORDS
        rows = [[b for w in e[words * k:words * (k + 1)] for b in w] for k in range(P)]
        level, root = list(e[words * P]), [b for w in e[words * P + 1:] for b in w]
        g, _ = _point(C, level + [ZERO], S)
        n, n_fits = _point(C, add(C, level, [ONE], width=len(level) + 1), S)
        es = [_entry(C, row, root, g, n, n_fits) for row in rows]
        conds = [x.good for x in es] + [es[0].call, es[0].starts, NOT(es[-1].call)]
        for last, x in zip(es, es[1:]):
            conds.append(x.empty ^ last.call)                              # after a call a call or the filler, else empty
            conds.append(OR(C, x.empty, _eq(C, x.lo, last.hi)))
            conds.append(NOT(C.AND(x.call, sub(C, x.tile0, last.end, _T + 1)[1])))
        return [and_reduce(C, conds)]

    return build


@functools.cache
def traced(P: int, S: int = Wd.GRID_BITS) -> Traced:
    """The unit's gate circuit at P entries on the S-bit grid."""
    return trace(_build(P, S), *_types(P))


def _refuse(b: dict) -> str | None:
    p, s = b.get("P"), b.get("S")
    if not (isinstance(p, int) and not isinstance(p, bool) and p >= 2):
        return f"P is the execution phase's entries bound, an integer of at least 2 (a call and the filler), not {p!r}"
    if not (isinstance(s, int) and not isinstance(s, bool) and 0 <= s <= S_MAX):
        return f"S is the grid's bits, an integer from 0 to {S_MAX}, not {s!r}"
    return None


PouwWindowStatement = CompositeDefinition(
    "PouwWindowStatement", 1, ("P", "S"), lambda st: (tuple(zip(PORTS, _types(st.P)[0])), _types(st.P)[1]),
    lambda B, st, *c: emit(traced(st.P, st.S), *_types(st.P), B, *c), conformance="protocol-defined", refuse=_refuse,
    doc="An execution phase's statement on its committed entry rows (`circuit.window_statement`): the P entry rows (168 "
        "u16 words each), the public registered level and the public -h3 tile root of the pass (32 u16 words), to 1 iff "
        "every row is an entry row of the layout, the rows are the calls, then one filler, then zero rows, the ranges tile "
        "[0, N) in order for N the S-bit grid's point at level + 1, each call's range is its n_a n_b tiles and the "
        "filler's is within its own, every call's -h3 root is the public one, each call's pass tiles [tile0, tile0 + "
        "ceil(m / 64) cols) start at or past the call before's end, and the filler starts at or past the grid's point at "
        "level.")


def template(P: int, S: int = Wd.GRID_BITS) -> Any:
    """The statement Program's one template: `PouwWindowStatement_v1` at P entries on the S-bit grid."""
    return bind(PouwWindowStatement, P=P, S=S)


def program(P: int, S: int = Wd.GRID_BITS) -> Program:
    """The statement Program of an execution phase of P entries on the S-bit grid: its one unit, `template` (P, S) once
    (`TemplateInstances_v1`, each parameter an array over the one instance)."""
    return population_program(template(P, S), 1)


def query(P: int, S: int = Wd.GRID_BITS) -> dict[str, Any]:
    """The statement Program's partition query: its template's one instance, the unit."""
    return template_instances_query([template(P, S)])


__all__ = ["FIELDS", "PORTS", "PUBLIC", "PouwWindowStatement", "ROOT_WORDS", "S_MAX", "program", "query", "template", "traced",
           "window_ok"]
