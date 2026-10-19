"""A window's admission from its committed request queue (PADCREDIT's admit, coordinator 01:10Z on 19 Oct): slot s of a call
at step t is admitted iff the queue assigns it a request live at t, computed from the queue alone, never from the slot's row.

The **request queue** is an input committed before the window's first coin, two tables of hiding leaves:

    slot map      one u16 per (step t, slot s): q, the index of the request the scheduler put in slot s at step t, 0 for none
    requests      one (start, stop) pair of u16 per request index q >= 1: the steps [start, stop) the request is live in;
                  index 0 is the null request, (0, 0)

One admission unit (`PouwQueueAdmit`) reads a call's 16 slots: the step t, each slot's q from the slot map and the request
entry at q, and outputs their 16 admit bits as one word (bit b slot 16 g + b of group g), a proof unit's outputs being at most
16 bits.  Bit b is ``q_b != 0 and start_b <= t < stop_b``.  It reads no row, so a nonzero row in a slot the queue leaves
unassigned is never admitted: it credits nothing, and the window's check refuses it (a slot that is not admitted holds the
canonical empty row).  The unit is generic over the window: a window maps its call to t and its slots to groups of 16, and
nothing else of the window reaches it (served_h1's A tables of T slots, the static window's side A of m slots).
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
M16 = (1 << 16) - 1


def admit_bit(t: int, q: int, start: int, stop: int) -> int:
    """1 iff the slot's request q is one (q != 0) and live at step t (start <= t < stop), every field a u16."""
    t, q, start, stop = t & M16, q & M16, start & M16, stop & M16
    return int(q != 0 and start <= t < stop)


def admit_word(t: int, q: Sequence[int], start: Sequence[int], stop: Sequence[int]) -> int:
    """The 16 slots' admit bits as a u16, bit b slot b."""
    if not len(q) == len(start) == len(stop) == SLOTS:
        raise ValueError(f"an admission unit reads {SLOTS} slots, not {len(q)}, {len(start)} and {len(stop)}")
    return sum(admit_bit(t, q[b], start[b], stop[b]) << b for b in range(SLOTS))


@primitive("PouwQueueAdmit", 1, [("t", Value(16)), ("q", Array(SLOTS, Value(16))), ("start", Array(SLOTS, Value(16))),
                                  ("stop", Array(SLOTS, Value(16)))], Value(16), conformance="protocol-defined")
def PouwQueueAdmitWord(t: int, *e: int) -> int:
    """Bit b: slot b's request q_b is one and live at step t, ``q_b != 0 and start_b <= t < stop_b``."""
    return admit_word(t, e[:SLOTS], e[SLOTS:2 * SLOTS], e[2 * SLOTS:])


def _bits(C_, t, *e):
    q, start, stop = e[:SLOTS], e[SLOTS:2 * SLOTS], e[2 * SLOTS:]
    out = []
    for b in range(SLOTS):
        _, before_start = sub(C_, t, start[b], 16)
        _, before_stop = sub(C_, t, stop[b], 16)
        out.append(C_.AND(or_reduce(C_, list(q[b])), C_.AND(NOT(before_start), before_stop)))
    return out


PouwQueueAdmit = definition(
    "PouwQueueAdmit", 2, _bits, (("t", U16), ("q", Array(SLOTS, U16)), ("start", Array(SLOTS, U16)), ("stop", Array(SLOTS, U16))),
    U16, word=PouwQueueAdmitWord, conformance="protocol-defined",
    doc="A call's 16 slots' admission from the committed request queue, on bits: bit b = q_b != 0 and start_b <= t < "
        "stop_b (t - start_b borrows not, t - stop_b borrows): PouwQueueAdmit_v1")

__all__ = ["M16", "PouwQueueAdmit", "PouwQueueAdmitWord", "SLOTS", "U16", "admit_bit", "admit_word"]
