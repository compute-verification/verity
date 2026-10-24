"""`circuit.live`: `PouwLiveSlots_v1{T}` on bits is `live_ok` on words, ok = 1 iff exactly the slots [0, m) name an admitted
request of a prefix-ordered request table, and then m is rule A's count (`admitted`)."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.core.protocols.pouw.circuit import h1window as W
from verity.core.protocols.pouw.circuit import live as L

M32 = (1 << 32) - 1


def _b(v, w=32):
    return [(v >> i) & 1 for i in range(w)]


def _bits(slot, req, m):
    (ok,) = evaluate(bind(L.PouwLiveSlots, T=len(slot)), sum(map(_b, slot), []), sum((_b(x) for h in req for x in h), []),
                     _b(m))
    return ok


def _valid(rng, T, n=None, m=None):
    """A round the unit accepts: n live requests at their own indices, m slots naming them, the rest empty."""
    n = rng.randrange(T + 1) if n is None else n
    m = (rng.randrange(T + 1) if n else 0) if m is None else m
    req = [[q, rng.randrange(1 << 32), rng.randrange(1 << 32), rng.randrange(1 << 32), rng.randrange(1, 1 << 32)]
           if q < n else [0] * L.HEADER for q in range(T)]
    slot = [rng.randrange(1, n + 1) if i < m else 0 for i in range(T)]
    return slot, req, m


def _mutate(rng, slot, req, m):
    slot, req = list(slot), [list(h) for h in req]
    T, k = len(slot), rng.randrange(6)
    i = rng.randrange(T)
    if k == 0:
        slot[i] = rng.choice([0, 1, T, T + 1, 1 << 31, M32, rng.randrange(1 << 32)])
    elif k == 1:
        req[i][L.PROMPT] = rng.choice([0, 1, M32])
    elif k == 2:
        req[i][L.INDEX] = rng.choice([i ^ 1, i + T, M32, rng.randrange(1 << 32)])
    elif k == 3:
        m = rng.choice([0, T, T + 1, 1 << T.bit_length(), M32, rng.randrange(1 << 32), rng.randrange(T + 1)])
    elif k == 4:
        req[i][rng.randrange(1, 4)] = rng.randrange(1 << 32)
    else:
        req[i] = [0] * L.HEADER
    return slot, req, m


@pytest.mark.parametrize("T", [1, 2, 5, 8])
def test_bits_are_the_word(T):
    rng = random.Random(T)
    for _ in range(40):
        case = _valid(rng, T)
        for _ in range(rng.randrange(3)):
            case = _mutate(rng, *case)
        assert _bits(*case) == L.live_ok(*case)


@pytest.mark.parametrize("T", [1, 4, 8])
def test_an_accepted_round_counts_m_admitted_slots(T):
    rng = random.Random(100 + T)
    for _ in range(40):
        slot, req, m = _valid(rng, T)
        assert L.live_ok(slot, req, m) == 1 and _bits(slot, req, m) == 1
        assert L.admitted(slot, req) == m


def _round():
    """T = 8: requests 0, 1 and 2 live, slots [0, 4) theirs."""
    req = [[q, 0, q, 16, 3 + q] for q in range(3)] + [[0] * L.HEADER] * 5
    return [1, 3, 3, 2, 0, 0, 0, 0], req, 4


@pytest.mark.parametrize("edit, want", [
    (lambda s, r, m: (s, r, m), 1),
    (lambda s, r, m: (s, r, 3), 0),                                          # slot 3 assigned, m says 3
    (lambda s, r, m: (s, r, 5), 0),                                          # slot 4 counted but empty (a pad row)
    (lambda s, r, m: (s[:2] + [0] + s[3:], r, m), 0),                        # an empty slot inside [0, m)
    (lambda s, r, m: (s[:2] + [4] + s[3:], r, m), 0),                        # slot names request 3, not an entry
    (lambda s, r, m: (s[:6] + [1, 0], r, m), 0),                             # an assigned slot past m
    (lambda s, r, m: (s[:2] + [3 + (1 << 31)] + s[3:], r, m), 0),             # a high bit on a named slot
    (lambda s, r, m: (s, r[:1] + [[0] * L.HEADER] + r[2:], m), 0),           # a gap in the request table
    (lambda s, r, m: (s, r[:2] + [[7] + r[2][1:]] + r[3:], m), 0),           # an entry at another index
    (lambda s, r, m: (s, r[:3] + [[3, 0, 0, 0, 1]] + r[4:], m), 1),          # one more live request, never named
    (lambda s, r, m: (s, r[:3] + [[9, 9, 9, 9, 0]] + r[4:], m), 1),          # a dead leaf's other words are not read
    (lambda s, r, m: (s, r, 9), 0),                                          # m > T
    (lambda s, r, m: (s, r, 4 + (1 << 16)), 0),                              # m's high bits
    (lambda s, r, m: ([0] * 8, r, 0), 1),                                    # m = 0: no slot assigned
    (lambda s, r, m: ([1] * 8, r, 8), 1),                                    # m = T
])
def test_ok_iff_exactly_the_first_m_slots_name_admitted_requests(edit, want):
    slot, req, m = edit(*_round())
    assert L.live_ok(slot, req, m) == want
    assert _bits(slot, req, m) == want
    if want:
        assert L.admitted(slot, req) == m


def test_m_reaches_the_leaf_count_not_the_window_slot():
    """T is the tables' leaf count (the pouw setting ``rows``), below the window's slot M when it isn't a power of two: at
    T = 96 (M = 128) every slot may be served, and an m past T is refused though it is within M."""
    assert W.slot(96, (32, 32)) == 128
    slot, req, _ = _valid(random.Random(96), 96, n=5, m=96)
    assert L.live_ok(slot, req, 96) == _bits(slot, req, 96) == 1
    for m in (97, 128):
        assert L.live_ok(slot, req, m) == _bits(slot, req, m) == 0


def test_ok_reads_every_slot_and_every_header_word_it_checks():
    """Its ports are T slots, T five-word headers and m, in that order, to one bit; its circuit grows linearly in T."""
    params, ret = bind(L.PouwLiveSlots, T=8).signature()
    assert [n for n, _ in params] == list(L.PORTS) and ret == L.BL.BIT
    assert [t.leaves for _, t in params] == [8 * 32, 8 * 5 * 32, 32]
    a8, a16 = L.traced(8).counts["and"], L.traced(16).counts["and"]
    assert a16 < 2 * a8 + 64


@pytest.mark.parametrize("T", [0, -1, "8", 2.5, None])
def test_t_is_a_positive_integer(T):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(L.PouwLiveSlots, T=T)
