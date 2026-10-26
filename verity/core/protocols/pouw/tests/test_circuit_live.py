"""`circuit.live`: `PouwLiveSlots_v1{T}` on bits is `live_ok` on words, ok = 1 iff exactly the slots [0, m) name an admitted
request of a prefix-ordered request table, none past what its request can take in round s (`takes`), and then m is rule
A's count (`admitted`). `PouwLiveSlots_v2{T}` on bits is `live_ok_v2`: v1's ok with each request's cap and round bound to
the public caps, so a request takes a decode row only within its public max_tokens."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.protocols.pouw.circuit import h1window as W
from verity.ml.protocols.pouw.circuit import live as L
from verity.core.protocols.pouw import live_reads as LR
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

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


@pytest.mark.parametrize("T, S", [(4, 1), (8, 3)])
def test_the_live_program_is_one_unit_a_round_of_its_one_template(T, S):
    """The live Program's partition is S units, unit s round s's `PouwLiveSlots_v1{T}` on its rows of the slots, m and round
    and on every header, so it evaluates to round s's ok at row s."""
    prog = L.program(T, S, "h1/")
    params, ret = prog.fn.signature()
    assert [n for n, _ in params] == [f"h1/{r}" for r in LR.ROLES] and ret.leaves == S
    tp = P.evaluate(prog, L.query(T))
    assert tp.population == S
    rng, n = random.Random(T * S), max(T // 2, 1)
    req = [[q, 0, q, S + 1, 1] if q < n else [0] * L.HEADER for q in range(T)]
    rounds = []
    for _s in range(S):
        m = rng.randrange(1, n + 1)
        rounds.append(([q + 1 for q in rng.sample(range(n), m)] + [0] * (T - m), m))
    if S > 1:
        rounds[-1] = (rounds[-1][0], rounds[-1][1] + 1)
    want = [L.live_ok(slot, req, m, s) for s, (slot, m) in enumerate(rounds)]
    assert want[0] == 1 and (S == 1 or want[-1] == 0)
    args = (sum((_b(v) for slot, _m in rounds for v in slot), []), sum((_b(x) for h in req for x in h), []),
            sum((_b(m) for _slot, m in rounds), []), sum((_b(s) for s in range(S)), []))
    assert list(evaluate(prog.fn, *args)) == want


@pytest.mark.parametrize("T, S", [(0, 1), (4, 0), (True, 2), (4, 2.0)])
def test_the_live_programs_t_and_s_are_positive_integers(T, S):
    with pytest.raises(ValueError, match="positive integers"):
        L.program(T, S)


def test_the_tables_rows_are_the_statements_port_rows():
    """A slot leaf's row (`slot_words`) and a header's (`header_words`) are the u16 words, low first, of the bits the
    statement's ``slot_<i>`` and ``req_<i>`` rows hold (`statement_rows`), and ``m``'s and ``s``'s rows their 32 bits. The headers are the
    one ``row-seg/v1`` rows (`REQUEST_SCHEMA`): a ``row/v2`` row this short has no compression before its last block, and a
    statement needs one."""
    assert {p[:3]: s for p, s, _b, _ps in LR.statement_rows(4)} == {"slo": G.SCHEMA_V2, "req": G.SCHEMA_SEG, "m": G.SCHEMA_V2,
                                                                  "s": G.SCHEMA_V2}
    rows = {p: (b, ps) for p, _s, b, ps in LR.statement_rows(4)}
    assert list(rows) == [f"slot_{i}" for i in range(4)] + [f"req_{i}" for i in range(4)] + ["m", "s"]
    assert rows["slot_2"] == (32, ["slot[2]"]) and rows["req_3"] == (160, ["req[3]"]) and rows["m"] == (32, ["m"])
    assert rows["s"] == (32, ["s"])
    words = lambda bits: [sum(b << j for j, b in enumerate(bits[16 * i:16 * i + 16])) for i in range(len(bits) // 16)]  # noqa: E731
    for v in (0, 1, 0x1234ABCD, M32):
        assert LR.slot_words([v]) == [words(_b(v))]
    h = [7, 0x10002, 3, 1 << 31, 9]
    assert LR.header_words([h]) == [words(sum(map(_b, h), []))] and len(LR.header_words([h])[0]) == LR.HEADER_WORDS
    assert G.row_words(LR.slot_words([5])[0], G.WORD_BITS, G.SCHEMA_V2) == LR.slot_words([5])[0]


def _window(T, S, rng):
    prog, window, session = "cd" * 64, "ab" * 64, "session-7"
    regs = {LR.table("h1/", LR.REQUESTS): G.Registered.commit(prog, LR.table("h1/", LR.REQUESTS),
                                                             LR.header_words(_valid(rng, T, n=2)[1]),
                                                             schema=LR.REQUEST_SCHEMA, session=session)}
    for s in range(S):
        v = LR.table("h1/", LR.SLOTS, s)
        regs[v] = G.Registered.commit(prog, v, LR.slot_words([1] * (T // 2) + [0] * (T - T // 2)), schema=G.SCHEMA_V2,
                                      window=window)
    return prog, window, session, regs


def test_a_rounds_live_file_reads_its_slot_table_the_request_table_and_the_known_m_round_and_ok_rows():
    """``live@<s>`` reads row i of round s's ``slots@<s>`` (bound to the open root) and of ``requests@0`` (bound to the
    session) at the domains those tables were registered under, m, s and ok at their known content, and nothing at another
    round's unit."""
    T, S, s = 4, 3, 1
    prog, window, session, regs = _window(T, S, random.Random(5))
    roots = {v: r.root.hex() for v, r in regs.items()}
    doc = LR.reads_file(prog, T, S, s, roots, window, session, 2, "h1/")
    ports = doc["ports"]
    assert doc["format"] == G.READS and sorted(ports) == sorted([p for p, *_ in LR.statement_rows(T)] + [LR.OK_PORT])
    for i in range(T):
        for port, v in ((f"slot_{i}", LR.table("h1/", LR.SLOTS, s)), (f"req_{i}", LR.table("h1/", LR.REQUESTS))):
            assert {k: ports[port][k] for k in G.ENTRY_KEYS} == {k: regs[v].entry()[k] for k in G.ENTRY_KEYS}
            assert ports[port]["reads"] == [None] * s + [i] and ports[port]["row_schema"] == regs[v].schema
    rows, salts, reg = LR.m_table(prog, "h1/", window, s, 2)
    assert ports["m"]["known"] == {"rows": [LR.slot_words([2])[0]], "salts": [x.hex() for x in salts]}
    assert ports["m"]["root"] == reg.root.hex() and ports["m"]["reads"] == [None] * s + [0]
    assert ports["m"]["root"] != LR.reads_file(prog, T, S, s, roots, window, session, 3, "h1/")["ports"]["m"]["root"]
    rows, salts, reg = LR.round_table(prog, "h1/", window, s)
    assert ports["s"]["known"] == {"rows": [LR.slot_words([s])[0]], "salts": [x.hex() for x in salts]}
    assert ports["s"]["root"] == reg.root.hex() and ports["s"]["reads"] == [None] * s + [0]
    assert ports["s"]["root"] != LR.reads_file(prog, T, S, 0, roots, window, session, 2, "h1/")["ports"]["s"]["root"]
    ok = ports[LR.OK_PORT]
    assert ok["leaves"] == S and ok["reads"] == [None] * s + [s] and ok["known"]["rows"] == [[1]] * S
    with pytest.raises(ValueError, match="none of the window's"):
        LR.reads_file(prog, T, S, S, roots, window, session, 2, "h1/")
    with pytest.raises(ValueError, match="doesn't register"):
        LR.reads_file(prog, T, S, 2, {k: r for k, r in roots.items() if not k.endswith("@2")}, window, session, 2, "h1/")


def test_the_live_tables_are_named_by_round():
    assert [LR.table("h1/", n, 2) for n in (LR.SLOTS, LR.M_ROWS, LR.ROUND_ROWS)] == ["h1/slots@2", "h1/m@2", "h1/round@2"]
    assert LR.table("h1/", LR.REQUESTS) == "h1/requests@0" and LR.table("", LR.OK_VALUE) == "live-ok"
    with pytest.raises(ValueError, match="requests@0"):
        LR.table("h1/", LR.REQUESTS, 1)


def _bits_v2(slot, req, m, s, cap):
    (ok,) = evaluate(bind(L.PouwLiveSlotsV2, T=len(slot)), sum(map(_b, slot), []),
                     sum((_b(x) for h in req for x in h), []), _b(m), _b(s), sum(map(_b, cap), []))
    return ok


def _admitted_in_round_0(rng, T, s=None):
    """A round s the v2 unit accepts: `_valid`'s with every live request admitted in round 0, and its public caps."""
    s = rng.choice([0, 1, 2, 5]) if s is None else s
    slot, req, m, _s = _valid(rng, T, s=s)
    req = [[h[0], 0, *h[2:]] if h[L.PROMPT] else h for h in req]
    pool = [q + 1 for q, h in enumerate(req) if h[L.PROMPT] for _ in range(min(L.takes(h, s), rng.randrange(T + 1)))]
    rng.shuffle(pool)
    m = rng.randrange(min(len(pool), T) + 1)
    cap = [h[L.CAP] if h[L.PROMPT] else 0 for h in req]
    if any(h[L.PROMPT] and h[L.CAP] == 0 for h in req):
        req = [[*h[:3], max(h[L.CAP], 1), h[4]] if h[L.PROMPT] else h for h in req]
        cap = [max(c, 1) if h[L.PROMPT] else 0 for c, h in zip(cap, req)]
    return pool[:m] + [0] * (T - m), req, m, s, cap


@pytest.mark.parametrize("T", [1, 2, 5, 8])
def test_v2_bits_are_the_word(T):
    rng = random.Random(1000 + T)
    for _ in range(60):
        slot, req, m, s, cap = _admitted_in_round_0(rng, T)
        assert L.live_ok_v2(slot, req, m, s, cap) == _bits_v2(slot, req, m, s, cap) == L.live_ok(slot, req, m, s) == 1
        for _ in range(rng.randrange(1, 4)):
            if rng.random() < 0.4:
                cap = list(cap)
                q = rng.randrange(T)
                cap[q] = rng.choice([0, 1, cap[q] + 1, max(cap[q] - 1, 0), M32, rng.randrange(1 << 32)])
            elif rng.random() < 0.3:
                req = [list(h) for h in req]
                q = rng.randrange(T)
                req[q][L.ROUND] = rng.choice([1, 2, 1 << 31, rng.randrange(1 << 32)])
            else:
                slot, req, m, s = _mutate(rng, slot, req, m, s)
        assert _bits_v2(slot, req, m, s, cap) == L.live_ok_v2(slot, req, m, s, cap)


def _fixed():
    """T = 8, a fixed length of max_tokens [3, 1, 2], every request admitted in round 0: round 0 their prompts (3, 4 and 5
    tokens), round 1 one decode row for requests 0 and 2 (max_tokens past 1), round 2 one for request 0."""
    req = [[q, 0, q, cap, 3 + q] for q, cap in enumerate((3, 1, 2))] + [[0] * L.HEADER] * 5
    return req, [3, 1, 2, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("s, slot", [(0, [1, 1, 1, 2, 2, 2, 2, 3]), (1, [1, 3, 0, 0, 0, 0, 0, 0]),
                                     (2, [1, 0, 0, 0, 0, 0, 0, 0])])
def test_an_honest_fixed_length_round_is_accepted(s, slot):
    req, cap = _fixed()
    m = sum(1 for x in slot if x)
    assert L.live_ok_v2(slot, req, m, s, cap) == _bits_v2(slot, req, m, s, cap) == 1


@pytest.mark.parametrize("s, slot, edit", [
    # request 1 takes request 0's decode row in round 1, its hidden cap swapped with request 0's: v1 accepts it
    (1, [2, 3, 0, 0, 0, 0, 0, 0], lambda r: [[0, 0, 0, 1, 3], [1, 0, 1, 3, 4]] + r[2:]),
    # ... and in round 2
    (2, [2, 0, 0, 0, 0, 0, 0, 0], lambda r: [[0, 0, 0, 1, 3], [1, 0, 1, 3, 4]] + r[2:]),
    # request 1 takes round 1's second row as a prefill, its hidden round 1
    (1, [1, 2, 0, 0, 0, 0, 0, 0], lambda r: r[:1] + [[1, 1, 1, 1, 4]] + r[2:]),
    # a fourth request the fixed length doesn't have
    (1, [1, 4, 0, 0, 0, 0, 0, 0], lambda r: r[:3] + [[3, 0, 3, 2, 1]] + r[4:]),
])
def test_a_swapped_request_to_row_assignment_is_refused(s, slot, edit):
    """The rows a round serves are the public schedule's count (#3590, #3827's `credit._length`), and v1 holds each
    request to its hidden header's cap and round. A prover who keeps the counts but rewrites those headers moves a decode
    row from one request to another, and v1 accepts it; v2 reads the public caps and refuses it."""
    req, cap = _fixed()
    req = edit(req)
    m = sum(1 for x in slot if x)
    assert L.live_ok(slot, req, m, s) == _bits(slot, req, m, s) == 1
    assert L.live_ok_v2(slot, req, m, s, cap) == _bits_v2(slot, req, m, s, cap) == 0


@pytest.mark.parametrize("cap", [[3, 1, 0, 0, 0, 0, 0, 0], [3, 1, 2, 1, 0, 0, 0, 0], [1, 3, 2, 0, 0, 0, 0, 0],
                                 [3, 1, 2, 0, 0, 0, 0, 1 << 31]])
def test_the_live_requests_are_exactly_the_public_caps(cap):
    """A leaf is live iff its public cap is nonzero, a live leaf's cap the public one: fewer public requests, one more, or
    caps in another order are refused at the honest round."""
    req, _cap = _fixed()
    slot = [1, 3, 0, 0, 0, 0, 0, 0]
    assert L.live_ok_v2(slot, req, 2, 1, cap) == _bits_v2(slot, req, 2, 1, cap) == 0


def test_v2_reads_v1s_ports_and_the_public_caps():
    """v2's ports are v1's and then T public caps, its one output the ok bit; v1's Definition is unchanged."""
    params, ret = bind(L.PouwLiveSlotsV2, T=8).signature()
    assert [n for n, _ in params] == list(L.PORTS_V2) == [*L.PORTS, "cap"] and ret == L.BL.BIT
    assert [t.leaves for _, t in params] == [8 * 32, 8 * 5 * 32, 32, 32, 8 * 32]
    assert bind(L.PouwLiveSlots, T=8).id == "PouwLiveSlots_v1{T=8}" and bind(L.PouwLiveSlotsV2, T=8).id == "PouwLiveSlots_v2{T=8}"
    assert [n for n, _ in bind(L.PouwLiveSlots, T=8).signature()[0]] == list(L.PORTS)
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(L.PouwLiveSlotsV2, T=0)


@pytest.mark.parametrize("T, S", [(4, 1), (8, 3)])
def test_the_v2_live_program_is_one_unit_a_round_that_reads_the_caps(T, S):
    prog = L.program_v2(T, S, "h1/")
    params, ret = prog.fn.signature()
    assert [n for n, _ in params] == [f"h1/{r}" for r in LR.ROLES_V2] and ret.leaves == S
    assert P.evaluate(prog, L.query_v2(T)).population == S
    n = max(T // 2, 1)
    req = [[q, 0, q, S + 1, 1] if q < n else [0] * L.HEADER for q in range(T)]
    cap = [S + 1 if q < n else 0 for q in range(T)]
    rounds = [([q + 1 for q in range(n)] + [0] * (T - n), n) for _s in range(S)]
    args = (sum((_b(v) for slot, _m in rounds for v in slot), []), sum((_b(x) for h in req for x in h), []),
            sum((_b(m) for _slot, m in rounds), []), sum((_b(s) for s in range(S)), []), sum(map(_b, cap), []))
    assert list(evaluate(prog.fn, *args)) == [L.live_ok_v2(slot, req, m, s, cap) for s, (slot, m) in enumerate(rounds)]
    assert list(evaluate(prog.fn, *args)) == [1] * S
    cap[0] += 1
    args = (*args[:4], sum(map(_b, cap), []))
    assert list(evaluate(prog.fn, *args)) == [0] * S


def test_a_v2_rounds_live_file_reads_the_known_public_caps():
    """With max_tokens, ``live@<s>`` also reads row i of the known ``<prefix>caps`` at each ``cap_<i>``: T rows, leaf q the
    public max_tokens of request q and 0 past them, under `core_check.public_row_salt` of the open root and q; other
    max_tokens give another root."""
    T, S, s = 4, 3, 1
    prog, window, session, regs = _window(T, S, random.Random(6))
    roots = {v: r.root.hex() for v, r in regs.items()}
    doc = LR.reads_file(prog, T, S, s, roots, window, session, 2, "h1/", [5, 2])
    ports = doc["ports"]
    assert sorted(ports) == sorted([p for p, *_ in LR.statement_rows(T, v2=True)] + [LR.OK_PORT])
    rows, salts, reg = LR.caps_table(prog, "h1/", window, T, [5, 2])
    assert rows == LR.slot_words([5, 2, 0, 0]) and LR.caps([5, 2, 7, 1, 9], T) == [5, 2, 7, 1]
    for i in range(T):
        assert ports[f"cap_{i}"]["value"] == "h1/caps" and ports[f"cap_{i}"]["root"] == reg.root.hex()
        assert ports[f"cap_{i}"]["reads"] == [None] * s + [i]
        assert ports[f"cap_{i}"]["known"] == {"rows": rows, "salts": [x.hex() for x in salts]}
    for other in ([2, 5], [5, 2, 1], [5], [5, 3]):
        assert LR.caps_table(prog, "h1/", window, T, other)[2].root != reg.root
    assert "cap_0" not in LR.reads_file(prog, T, S, s, roots, window, session, 2, "h1/")["ports"]
    rows = {p: (b, ps) for p, _s, b, ps in LR.statement_rows(4, v2=True)}
    assert list(rows)[-4:] == [f"cap_{i}" for i in range(4)] and rows["cap_1"] == (32, ["cap[1]"])
    assert LR.table("h1/", LR.CAPS) == "h1/caps"
