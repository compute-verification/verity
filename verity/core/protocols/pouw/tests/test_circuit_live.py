"""`circuit.live`: `PouwLiveSlots_v1{T}` on bits is `live_ok` on words, ok = 1 iff exactly the slots [0, m) name an admitted
request of a prefix-ordered request table, and then m is rule A's count (`admitted`)."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.core.protocols.pouw.circuit import h1window as W
from verity.core.protocols.pouw.circuit import live as L
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

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


@pytest.mark.parametrize("T, S", [(4, 1), (8, 3)])
def test_the_live_program_is_one_unit_a_round_of_its_one_template(T, S):
    """The live Program's partition is S units, unit s round s's `PouwLiveSlots_v1{T}` on its rows of the slots and m and on
    every header, so it evaluates to round s's ok at row s."""
    prog = L.program(T, S, "h1/")
    params, ret = prog.fn.signature()
    assert [n for n, _ in params] == [f"h1/{r}" for r in L.ROLES] and ret.leaves == S
    tp = P.evaluate(prog, L.query(T))
    assert tp.population == S
    rng = random.Random(T * S)
    rounds = [_valid(random.Random(T), T, n=T // 2, m=rng.randrange(1, T + 1)) for _ in range(S)]
    if S > 1:
        rounds[-1] = rounds[-1][:2] + (rounds[-1][2] + 1,)
    req = rounds[0][1]
    want = [L.live_ok(slot, req, m) for slot, _r, m in rounds]
    assert want[0] == 1 and (S == 1 or want[-1] == 0)
    args = (sum((_b(v) for slot, _r, _m in rounds for v in slot), []), sum((_b(x) for h in req for x in h), []),
            sum((_b(m) for _s, _r, m in rounds), []))
    assert list(evaluate(prog.fn, *args)) == want


@pytest.mark.parametrize("T, S", [(0, 1), (4, 0), (True, 2), (4, 2.0)])
def test_the_live_programs_t_and_s_are_positive_integers(T, S):
    with pytest.raises(ValueError, match="positive integers"):
        L.program(T, S)


def test_the_tables_rows_are_the_statements_port_rows():
    """A slot leaf's row (`slot_words`) and a header's (`header_words`) are the u16 words, low first, of the bits the
    statement's ``slot_<i>`` and ``req_<i>`` rows hold (`statement_rows`), and ``m``'s row its 32 bits."""
    rows = {p: (b, ps) for p, _s, b, ps in L.statement_rows(4)}
    assert list(rows) == [f"slot_{i}" for i in range(4)] + [f"req_{i}" for i in range(4)] + ["m"]
    assert rows["slot_2"] == (32, ["slot[2]"]) and rows["req_3"] == (160, ["req[3]"]) and rows["m"] == (32, ["m"])
    words = lambda bits: [sum(b << j for j, b in enumerate(bits[16 * i:16 * i + 16])) for i in range(len(bits) // 16)]  # noqa: E731
    for v in (0, 1, 0x1234ABCD, M32):
        assert L.slot_words([v]) == [words(_b(v))]
    h = [7, 0x10002, 3, 1 << 31, 9]
    assert L.header_words([h]) == [words(sum(map(_b, h), []))] and len(L.header_words([h])[0]) == L.HEADER_WORDS
    assert G.row_words(L.slot_words([5])[0], G.WORD_BITS, G.SCHEMA_V2) == L.slot_words([5])[0]


def _window(T, S, rng):
    prog, window, session = "cd" * 64, "ab" * 64, "session-7"
    regs = {L.table("h1/", L.REQUESTS): G.Registered.commit(prog, L.table("h1/", L.REQUESTS),
                                                             L.header_words(_valid(rng, T, n=2, m=0)[1]),
                                                             schema=G.SCHEMA_V2, session=session)}
    for s in range(S):
        v = L.table("h1/", L.SLOTS, s)
        regs[v] = G.Registered.commit(prog, v, L.slot_words([1] * (T // 2) + [0] * (T - T // 2)), schema=G.SCHEMA_V2,
                                      window=window)
    return prog, window, session, regs


def test_a_rounds_live_file_reads_its_slot_table_the_request_table_and_the_known_m_and_ok_rows():
    """``live@<s>`` reads row i of round s's ``slots@<s>`` (bound to the open root) and of ``requests@0`` (bound to the
    session) at the domains those tables were registered under, m and ok at their known content, and nothing at another
    round's unit."""
    T, S, s = 4, 3, 1
    prog, window, session, regs = _window(T, S, random.Random(5))
    roots = {v: r.root.hex() for v, r in regs.items()}
    doc = L.reads_file(prog, T, S, s, roots, window, session, 2, "h1/")
    ports = doc["ports"]
    assert doc["format"] == G.READS and sorted(ports) == sorted([p for p, *_ in L.statement_rows(T)] + [L.OK_PORT])
    for i in range(T):
        for port, v in ((f"slot_{i}", L.table("h1/", L.SLOTS, s)), (f"req_{i}", L.table("h1/", L.REQUESTS))):
            assert {k: ports[port][k] for k in G.ENTRY_KEYS} == {k: regs[v].entry()[k] for k in G.ENTRY_KEYS}
            assert ports[port]["reads"] == [None] * s + [i] and ports[port]["row_schema"] == G.SCHEMA_V2
    rows, salts, reg = L.m_table(prog, "h1/", window, s, 2)
    assert ports["m"]["known"] == {"rows": [L.slot_words([2])[0]], "salts": [x.hex() for x in salts]}
    assert ports["m"]["root"] == reg.root.hex() and ports["m"]["reads"] == [None] * s + [0]
    assert ports["m"]["root"] != L.reads_file(prog, T, S, s, roots, window, session, 3, "h1/")["ports"]["m"]["root"]
    ok = ports[L.OK_PORT]
    assert ok["leaves"] == S and ok["reads"] == [None] * s + [s] and ok["known"]["rows"] == [[1]] * S
    with pytest.raises(ValueError, match="none of the window's"):
        L.reads_file(prog, T, S, S, roots, window, session, 2, "h1/")
    with pytest.raises(ValueError, match="doesn't register"):
        L.reads_file(prog, T, S, 2, {k: r for k, r in roots.items() if not k.endswith("@2")}, window, session, 2, "h1/")


def test_the_live_tables_are_named_by_round():
    assert [L.table("h1/", n, 2) for n in (L.SLOTS, L.M_ROWS)] == ["h1/slots@2", "h1/m@2"]
    assert L.table("h1/", L.REQUESTS) == "h1/requests@0" and L.table("", L.OK_VALUE) == "live-ok"
    with pytest.raises(ValueError, match="requests@0"):
        L.table("h1/", L.REQUESTS, 1)
