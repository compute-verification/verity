"""`circuit.admit`: `PouwQueueAdmit_v2` is its `_v1` word, a slot is admitted iff the queue assigns it a request live at the
step, and the unit reads no row."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.core.protocols.pouw.circuit import admit as A


def _b16(v):
    return [(v >> i) & 1 for i in range(16)]


def _bits(t, q, start, stop):
    return list(evaluate(A.PouwQueueAdmit, _b16(t), sum(map(_b16, q), []), sum(map(_b16, start), []), sum(map(_b16, stop), [])))


def test_bits_are_the_word():
    rng = random.Random(19)
    for _ in range(48):
        t = rng.choice([0, 1, 17, 0xFFFE, 0xFFFF, rng.randrange(1 << 16)])
        near = [0, t, (t + 1) & A.M16, (t - 1) & A.M16, 0xFFFF]
        q = [rng.choice([0, 0, 1, rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        start = [rng.choice(near + [rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        stop = [rng.choice(near + [rng.randrange(1 << 16)]) for _ in range(A.SLOTS)]
        want = A.admit_word(t, q, start, stop)
        assert _bits(t, q, start, stop) == _b16(want)
        assert evaluate(A.PouwQueueAdmitWord, t, *q, *start, *stop) == (want,)


@pytest.mark.parametrize("q, start, stop, t, want", [
    (0, 0, 9, 3, 0),      # unassigned: the null request, whatever its lifetime reads
    (5, 4, 9, 3, 0),      # before the request starts
    (5, 4, 9, 4, 1),      # its first step
    (5, 4, 9, 8, 1),      # its last step
    (5, 4, 9, 9, 0),      # past its stop
    (5, 4, 4, 4, 0),      # an empty lifetime
])
def test_a_slot_is_admitted_iff_the_queue_assigns_it_a_live_request(q, start, stop, t, want):
    qs, st, sp = [0] * A.SLOTS, [0] * A.SLOTS, [0] * A.SLOTS
    qs[7], st[7], sp[7] = q, start, stop
    assert A.admit_word(t, qs, st, sp) == want << 7
    assert _bits(t, qs, st, sp) == _b16(want << 7)


def test_the_unit_reads_no_row():
    """Its ports are the step and the queue's 16 entries: no slot's row reaches it, so a junk row can't be admitted."""
    params, ret = A.PouwQueueAdmit.signature(None)
    assert [n for n, _ in params] == ["t", "q", "start", "stop"] and ret == A.U16
    assert A.PouwQueueAdmit.traced().counts["and"] == 49 * A.SLOTS
