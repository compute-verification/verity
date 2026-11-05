"""A window's admission from its committed request queue (PADCREDIT's admit): slot s of a call
at step t is selected iff the queue assigns it a request live at t, computed from the queue alone, and admitted iff it is
selected and its row's committed scale is nonzero, so a selected slot holding the canonical empty row (K zero bytes, scale 0)
is not admitted.

The **request queue** is an input committed before the window's first coin, two tables of hiding leaves:

    slot map      one u16 per (step t, slot s): q, the index of the request the scheduler put in slot s at step t, 0 for none
    requests      one (start, stop) pair of u16 per request index q >= 1: the steps [start, stop) the request is live in;
                  index 0 is the null request, (0, 0)

One admission unit (`PouwQueueAdmit`) reads a call's 16 slots: the step t, each slot's q from the slot map and the request
entry at q, and each slot's committed scale (the f32 that ends its row, as a u32), and outputs their 16 admit bits as one
word (bit b slot 16 g + b of group g), a proof block's outputs being at most 16 bits.  Bit b is
``q_b != 0 and start_b <= t < stop_b and scale_b != 0``.  The selection is the queue's: a row can withhold its slot's
admission, never grant it, so a row in a slot the queue leaves unassigned is never admitted, whatever it holds.

The unit reads a row's scale and none of its bytes, so an admit bit says nothing of the bytes: a selected row of zero bytes
under a nonzero scale (`scale_only`) is admitted, and so is a block of 32 of them, which is a dead tile to a tile that reads
A's bytes and masks its receipt by them.  Nothing here refuses such a
row, and no window reads this unit yet.  So an admit bit is not a live tile: credit must count only tiles live in-circuit,
and that is the served tally's to establish, not this unit's.  The unit is generic over the window: a window maps its call
to t and its slots to groups of 16, and nothing else of the window reaches it (served_h1's A tables of T slots, the static
window's side A of m slots).
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import primitive
from verity.ml.circuits.types import Array, Value

#: slots one unit admits, the 16-bit output limit
SLOTS = 16
U16 = Array(16, BL.BIT)
U32 = Array(32, BL.BIT)
M16 = (1 << 16) - 1
M32 = (1 << 32) - 1


def select_bit(t: int, q: int, start: int, stop: int) -> int:
    """1 iff the slot's request q is one (q != 0) and live at step t (start <= t < stop), every field a u16."""
    t, q, start, stop = t & M16, q & M16, start & M16, stop & M16
    return int(q != 0 and start <= t < stop)


def admit_bit(t: int, q: int, start: int, stop: int, scale: int) -> int:
    """1 iff the slot is selected (`select_bit`) and its committed scale, the u32 of its row's f32, is nonzero."""
    return int(select_bit(t, q, start, stop) and scale & M32 != 0)


def admit_word(t: int, q: Sequence[int], start: Sequence[int], stop: Sequence[int], scale: Sequence[int]) -> int:
    """The 16 slots' admit bits as a u16, bit b slot b."""
    if not len(q) == len(start) == len(stop) == len(scale) == SLOTS:
        raise ValueError(f"an admission unit reads {SLOTS} slots, not {len(q)}, {len(start)}, {len(stop)} and {len(scale)}")
    return sum(admit_bit(t, q[b], start[b], stop[b], scale[b]) << b for b in range(SLOTS))


@primitive("PouwQueueAdmit", 2, [("t", Value(16)), ("q", Array(SLOTS, Value(16))), ("start", Array(SLOTS, Value(16))),
                                  ("stop", Array(SLOTS, Value(16))), ("scale", Array(SLOTS, Value(32)))], Value(16),
           conformance="protocol-defined")
def PouwQueueAdmitWord(t: int, *e: int) -> int:
    """Bit b: slot b's request q_b is one and live at step t and its row's scale is nonzero,
    ``q_b != 0 and start_b <= t < stop_b and scale_b != 0``."""
    return admit_word(t, e[:SLOTS], e[SLOTS:2 * SLOTS], e[2 * SLOTS:3 * SLOTS], e[3 * SLOTS:])


def _bits(C_, t, *e):
    q, start, stop, scale = e[:SLOTS], e[SLOTS:2 * SLOTS], e[2 * SLOTS:3 * SLOTS], e[3 * SLOTS:]
    out = []
    for b in range(SLOTS):
        _, before_start = sub(C_, t, start[b], 16)
        _, before_stop = sub(C_, t, stop[b], 16)
        selected = C_.AND(or_reduce(C_, list(q[b])), C_.AND(NOT(before_start), before_stop))
        out.append(C_.AND(selected, or_reduce(C_, list(scale[b]))))
    return out


PouwQueueAdmit = definition(
    "PouwQueueAdmit", 3, _bits,
    (("t", U16), ("q", Array(SLOTS, U16)), ("start", Array(SLOTS, U16)), ("stop", Array(SLOTS, U16)),
     ("scale", Array(SLOTS, U32))),
    U16, word=PouwQueueAdmitWord, conformance="protocol-defined",
    doc="A call's 16 slots' admission from the committed request queue and their rows' scales, on bits: bit b = q_b != 0 and "
        "start_b <= t < stop_b and scale_b != 0 (t - start_b borrows not, t - stop_b borrows): PouwQueueAdmit_v2")

__all__ = ["M16", "M32", "PouwQueueAdmit", "PouwQueueAdmitWord", "SLOTS", "U16", "U32", "admit_bit", "admit_word",
           "select_bit"]
