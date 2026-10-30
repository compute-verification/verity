"""`circuit.admit`: `PouwQueueAdmit_v3` is its `_v2` word, a slot is admitted iff the queue assigns it a request live at the
step and its row's scale is nonzero, so a selected slot holding the canonical empty row is not admitted, and the unit
reads no row byte, so a selected `scale_only` row (zero bytes, a nonzero scale) is admitted."""
from __future__ import annotations

import random
import struct

import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.types import Array
from verity.ml.protocols.pouw.circuit import admit as A

#: a live row's scale, as the u32 of its f32
LIVE = struct.unpack("<I", struct.pack("<f", 0.5))[0]


def _b(v, w=16):
    return [(v >> i) & 1 for i in range(w)]


def _bits(t, q, start, stop, scale):
    return list(evaluate(A.PouwQueueAdmit, _b(t), sum(map(_b, q), []), sum(map(_b, start), []), sum(map(_b, stop), []),
                         sum((_b(s, 32) for s in scale), [])))


def test_bits_are_the_word():
    rng = random.Random(19)
    for _ in range(48):
        t = rng.choice([0, 1, 17, 0xFFFE, 0xFFFF, rng.randrange(1 << 16)])
        near = [0, t, (t + 1) & A.M16, (t - 1) & A.M16, 0xFFFF]
        q = [rng.choice([0, 0, 1, rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        start = [rng.choice(near + [rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        stop = [rng.choice(near + [rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        scale = [rng.choice([0, 0, LIVE, 1, 1 << 31, A.M32, rng.randrange(1 << 32)]) for _ in range(A.SLOTS)]
        want = A.admit_word(t, q, start, stop, scale)
        assert _bits(t, q, start, stop, scale) == _b(want)
        assert evaluate(A.PouwQueueAdmitWord, t, *q, *start, *stop, *scale) == (want,)


@pytest.mark.parametrize("q, start, stop, t, scale, want", [
    (0, 0, 9, 3, LIVE, 0),      # unassigned: the null request, whatever its lifetime reads
    (5, 4, 9, 3, LIVE, 0),      # before the request starts
    (5, 4, 9, 4, LIVE, 1),      # its first step
    (5, 4, 9, 8, LIVE, 1),      # its last step
    (5, 4, 9, 9, LIVE, 0),      # past its stop
    (5, 4, 4, 4, LIVE, 0),      # an empty lifetime
    (5, 4, 9, 6, 0, 0),         # selected, but the row is the canonical empty row's scale 0 (RT-ALGO-4)
    (5, 4, 9, 6, 1 << 31, 1),   # -0.0 is not the canonical empty row's scale
])
def test_a_slot_is_admitted_iff_the_queue_assigns_it_a_live_request_and_its_row_a_scale(q, start, stop, t, scale, want):
    qs, st, sp, sc = [0] * A.SLOTS, [0] * A.SLOTS, [0] * A.SLOTS, [LIVE] * A.SLOTS
    qs[7], st[7], sp[7], sc[7] = q, start, stop, scale
    assert A.admit_word(t, qs, st, sp, sc) == want << 7
    assert _bits(t, qs, st, sp, sc) == _b(want << 7)


def test_a_selected_block_of_canonical_empty_rows_admits_nothing():
    """RT-ALGO-4's witness at the unit: all 16 slots selected and every row the canonical empty row (scale 0) admits no slot,
    so the tally credits none of them; the same slots with their rows' scales admit all 16."""
    q, start, stop = [3] * A.SLOTS, [0] * A.SLOTS, [8] * A.SLOTS
    assert A.select_bit(2, 3, 0, 8) == 1
    assert A.admit_word(2, q, start, stop, [0] * A.SLOTS) == 0
    assert _bits(2, q, start, stop, [0] * A.SLOTS) == [0] * 16
    assert A.admit_word(2, q, start, stop, [LIVE] * A.SLOTS) == A.M16


def test_the_unit_reads_no_row_byte():
    """Its ports are the step, the queue's 16 entries and the rows' 16 scales: no slot's s8 bytes reach it, so a selected
    `scale_only` row admits like a live one and an admit bit is not a live tile (RT4-LOGIC-4)."""
    params, ret = A.PouwQueueAdmit.signature(None)
    assert [n for n, _ in params] == ["t", "q", "start", "stop", "scale"] and ret == A.U16
    assert [t for n, t in params if n == "scale"] == [Array(A.SLOTS, A.U32)]
    assert A.PouwQueueAdmit.traced().counts["and"] == 81 * A.SLOTS
