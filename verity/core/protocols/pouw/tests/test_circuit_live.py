"""`circuit.live`: `PouwLiveSlots_v1{T}` on bits is `live_ok` on words, ok = 1 iff exactly the slots [0, m) name an admitted
request of a prefix-ordered request table, none past what its request can take in round s (`takes`), and then m is rule
A's count (`admitted`)."""
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


def _bits(slot, req, m, s):
    (ok,) = evaluate(bind(L.PouwLiveSlots, T=len(slot)), sum(map(_b, slot), []), sum((_b(x) for h in req for x in h), []),
                     _b(m), _b(s))
    return ok


def _valid(rng, T, n=None, s=None):
    """A round s the unit accepts: n live requests at their own indices, each admitted at or before s (some prefilling, some
    decoding, some past their cap), the first m slots naming them, none more often than `takes`, the rest empty."""
    n = rng.randrange(T + 1) if n is None else n
    s = rng.choice([0, 1, 2, 5, rng.randrange(1 << 32)]) if s is None else s
    req = []
    for q in range(T):
        if q >= n:
            req.append([0] * L.HEADER)
            continue
        r = rng.choice([s, s, max(s - 1, 0), rng.randrange(s + 1)])
        cap = min(M32, rng.choice([0, 1, 2, s - r, s - r + 1, s - r + 1, rng.randrange(1 << 32)]))
        req.append([q, r, rng.randrange(1 << 32), cap, rng.choice([1, 2, T, rng.randrange(1, 1 << 32)])])
    pool = [q + 1 for q, h in enumerate(req[:n]) for _ in range(min(L.takes(h, s), rng.randrange(T + 1)))]
    rng.shuffle(pool)
    m = rng.randrange(min(len(pool), T) + 1)
    return pool[:m] + [0] * (T - m), req, m, s


def _mutate(rng, slot, req, m, s):
    slot, req = list(slot), [list(h) for h in req]
    T, k = len(slot), rng.randrange(8)
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
        req[i][rng.randrange(1, 4)] = rng.choice([s, s + 1, max(s - 1, 0), rng.randrange(1 << 32)]) & M32
    elif k == 5:
        s = rng.choice([s + 1, max(s - 1, 0), s ^ (1 << 31), rng.randrange(1 << 32)]) & M32
    elif k == 6:
        slot[i] = slot[rng.randrange(T)]                                     # a request named once more
    else:
        req[i] = [0] * L.HEADER
    return slot, req, m, s


@pytest.mark.parametrize("T", [1, 2, 5, 8])
def test_bits_are_the_word(T):
    rng = random.Random(T)
    for _ in range(60):
        case = _valid(rng, T)
        for _ in range(rng.randrange(3)):
            case = _mutate(rng, *case)
        assert _bits(*case) == L.live_ok(*case)


@pytest.mark.parametrize("T", [1, 4, 8])
def test_an_accepted_round_counts_m_admitted_slots(T):
    rng = random.Random(100 + T)
    for _ in range(40):
        slot, req, m, s = _valid(rng, T)
        assert L.live_ok(slot, req, m, s) == 1 and _bits(slot, req, m, s) == 1
        assert L.admitted(slot, req) == m <= sum(L.takes(h, s) for h in req)


def _round():
    """T = 8, round 0: requests 0, 1 and 2 live, admitted there with prompts of 3, 4 and 5 tokens, slots [0, 4) theirs."""
    req = [[q, 0, q, 16, 3 + q] for q in range(3)] + [[0] * L.HEADER] * 5
    return [1, 3, 3, 2, 0, 0, 0, 0], req, 4, 0


@pytest.mark.parametrize("edit, want", [
    (lambda x, r, m, s: (x, r, m, s), 1),
    (lambda x, r, m, s: (x, r, 3, s), 0),                                     # slot 3 assigned, m says 3
    (lambda x, r, m, s: (x, r, 5, s), 0),                                     # slot 4 counted but empty (a pad row)
    (lambda x, r, m, s: (x[:2] + [0] + x[3:], r, m, s), 0),                   # an empty slot inside [0, m)
    (lambda x, r, m, s: (x[:2] + [4] + x[3:], r, m, s), 0),                   # slot names request 3, not an entry
    (lambda x, r, m, s: (x[:6] + [1, 0], r, m, s), 0),                        # an assigned slot past m
    (lambda x, r, m, s: (x[:2] + [3 + (1 << 31)] + x[3:], r, m, s), 0),        # a high bit on a named slot
    (lambda x, r, m, s: (x, r[:1] + [[0] * L.HEADER] + r[2:], m, s), 0),      # a gap in the request table
    (lambda x, r, m, s: (x, r[:2] + [[7] + r[2][1:]] + r[3:], m, s), 0),      # an entry at another index
    (lambda x, r, m, s: (x, r[:3] + [[3, 0, 0, 0, 1]] + r[4:], m, s), 1),     # one more live request, never named
    (lambda x, r, m, s: (x, r[:3] + [[9, 9, 9, 9, 0]] + r[4:], m, s), 1),     # a dead leaf's other words are not read
    (lambda x, r, m, s: (x, r, 9, s), 0),                                     # m > T
    (lambda x, r, m, s: (x, r, 4 + (1 << 16), s), 0),                         # m's high bits
    (lambda x, r, m, s: ([0] * 8, r, 0, s), 1),                               # m = 0: no slot assigned
    (lambda x, r, m, s: ([1, 1, 1, 2, 2, 2, 2, 3], r, 8, s), 1),              # m = T, each within its prompt
    # the per-request bound
    (lambda x, r, m, s: ([1] * 8, r, 8, s), 0),                               # one request in all T slots, prompt 3
    (lambda x, r, m, s: ([1] * 8, [[0, 0, 0, 16, 8]] + r[1:], 8, s), 1),       # ... whose prompt is T tokens
    (lambda x, r, m, s: ([1] * 8, [[0, 0, 0, 16, 8]] + r[1:], 8, 1), 0),       # ... decoding in round 1
    (lambda x, r, m, s: ([1, 1, 1, 3] + x[4:], r, m, s), 1),                  # request 0 prefills its 3 prompt tokens
    (lambda x, r, m, s: ([1, 1, 1, 1] + x[4:], r, m, s), 0),                  # ... and a 4th slot past its prompt
    (lambda x, r, m, s: (x, r, m, 1), 0),                                     # round 1: request 2 decodes on 2 slots
    (lambda x, r, m, s: ([1, 2, 3, 0, 0, 0, 0, 0], r, 3, 1), 1),              # round 1: one decode row each
    (lambda x, r, m, s: ([1, 2, 3, 0, 0, 0, 0, 0], r, 3, 15), 1),             # round 15, the cap's last decode
    (lambda x, r, m, s: ([1, 2, 3, 0, 0, 0, 0, 0], r, 3, 16), 0),             # round 16, past the cap of 16
    (lambda x, r, m, s: ([1, 2, 0, 0, 0, 0, 0, 0], r, 2, 1 + (1 << 31)), 0),  # s's high bit: past every cap
    (lambda x, r, m, s: (x, [h[:1] + [1] + h[2:] for h in r[:3]] + r[3:], m, s), 0),  # named in round 0, admitted in 1
])
def test_ok_iff_exactly_the_first_m_slots_name_admitted_requests(edit, want):
    slot, req, m, s = edit(*_round())
    assert L.live_ok(slot, req, m, s) == want
    assert _bits(slot, req, m, s) == want
    if want:
        assert L.admitted(slot, req) == m


def _mixed():
    """T = 8, round 2 of a per-round window: requests 0 and 1 admitted in round 0 (cap 4) and decoding, request 2 admitted in
    round 2 with a 4-token prompt and prefilling, request 3 admitted in round 0 with cap 2 and so done."""
    req = [[0, 0, 0, 4, 6], [1, 0, 6, 4, 2], [2, 2, 2, 8, 4], [3, 0, 8, 2, 1]] + [[0] * L.HEADER] * 4
    return [1, 2, 3, 3, 3, 3, 0, 0], req, 6, 2


@pytest.mark.parametrize("edit, want", [
    (lambda x, r, m, s: (x, r, m, s), 1),                                     # the honest mixed batch
    (lambda x, r, m, s: ([2, 1, 3, 3, 3, 3, 0, 0], r, m, s), 1),              # in another batch order
    (lambda x, r, m, s: ([1, 2, 3, 3, 3, 0, 0, 0], r, 5, s), 1),              # a prefix-cache hit: 3 of the 4 rows
    (lambda x, r, m, s: ([1, 2, 3, 3, 3, 3, 3, 0], r, 7, s), 0),              # a 5th prefill row past the prompt
    (lambda x, r, m, s: ([1, 1, 3, 3, 3, 3, 0, 0], r, m, s), 0),              # a decode request on 2 slots
    (lambda x, r, m, s: ([1, 2, 3, 3, 3, 3, 4, 0], r, 7, s), 0),              # request 3 past its cap
    (lambda x, r, m, s: ([1, 2, 3, 3, 3, 3, 0, 0], r, m, 1), 0),              # round 1: request 2 isn't admitted yet
])
def test_a_mixed_round_is_held_to_each_request_s_prefill_or_decode(edit, want):
    slot, req, m, s = edit(*_mixed())
    assert L.live_ok(slot, req, m, s) == want
    assert _bits(slot, req, m, s) == want


@pytest.mark.slow
def test_m_reaches_the_leaf_count_not_the_window_slot():
    """T is the tables' leaf count (the pouw setting ``rows``), below the window's slot M when it isn't a power of two: at
    T = 96 (M = 128) every slot may be served, and an m past T is refused though it is within M."""
    assert W.slot(96, (32, 32)) == 128
    req = [[q, 0, 20 * q, 16, 20] for q in range(5)] + [[0] * L.HEADER] * 91
    slot = [i // 20 + 1 for i in range(96)]
    assert L.live_ok(slot, req, 96, 0) == _bits(slot, req, 96, 0) == 1
    for m in (97, 128):
        assert L.live_ok(slot, req, m, 0) == _bits(slot, req, m, 0) == 0


def test_ok_reads_every_slot_and_every_header_word_it_checks():
    """Its ports are T slots, T five-word headers, m and s, in that order, to one bit; its circuit grows as T², a count of
    each request's slots."""
    params, ret = bind(L.PouwLiveSlots, T=8).signature()
    assert [n for n, _ in params] == list(L.PORTS) and ret == L.BL.BIT
    assert [t.leaves for _, t in params] == [8 * 32, 8 * 5 * 32, 32, 32]
    a8, a16 = L.traced(8).counts["and"], L.traced(16).counts["and"]
    assert a16 < 4 * a8


@pytest.mark.parametrize("T", [0, -1, "8", 2.5, None])
def test_t_is_a_positive_integer(T):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(L.PouwLiveSlots, T=T)
