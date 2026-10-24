"""The replay's state chain (`replay_chain`): the rows' layout and their refusals, `ReplayAdvance`, `ReplayLoad` and
`ReplayClose` against `advance`, `load` and `close` on random and edge cases, the front's log shifter against slicing,
the window's produced values opened at every unit's read, the carry and the empty start (`check_chain`), a two-window
session through all three parts, decisions a window leaves unread carried to the next and refused only at the
session's end, and the cost `size` reports."""
from __future__ import annotations

import random

import pytest

from verity.core.protocols.certifier import replay_chain as RC
from verity.core.protocols.certifier.head_check import ands
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.boolean.forms import Circuit

SMALL = RC.Shape(R=3, W=2, A=3, L=40, NB=20, IW=4, Q=5, D=6, P=3, K=3)
#: A a power of two (AC = 3 > DW = 1), one decision and one step a window
NARROW = RC.Shape(R=2, W=1, A=4, L=16, NB=8, IW=3, Q=6, D=1, P=1, K=2)
SHAPES = (SMALL, NARROW)
PROGRAM, REGISTRATION = "replay-chain-test", "ab" * 64


def _u(rng, w):
    return rng.randrange(1 << w)


def _rand_row(rng: random.Random, S, *, heads: bool = True) -> RC.Row:
    Z = RC.widths(S)
    return RC.Row(tuple((_u(rng, S.IW), _u(rng, Z.LW), _u(rng, Z.LW), _u(rng, Z.LW), _u(rng, Z.LW))
                        for _ in range(rng.randint(0, S.R))),
                  tuple((_u(rng, S.IW), _u(rng, Z.LW), _u(rng, Z.LW), _u(rng, Z.LW)) for _ in range(rng.randint(0, S.W))),
                  rng.randrange(S.NB), rng.randint(0, S.Q) if heads else 0, rng.randint(0, S.D) if heads else 0,
                  rng.randint(0, S.P) if heads else 0)


def _rand_arrivals(rng, S, n):
    Z = RC.widths(S)
    return tuple((_u(rng, S.IW), _u(rng, Z.LW), _u(rng, Z.LW)) for _ in range(n))


def _rand_window(rng: random.Random, S) -> RC.Window:
    return RC.Window(_rand_arrivals(rng, S, rng.randint(0, S.Q)), tuple(rng.randint(0, 1) for _ in range(rng.randint(0, S.D))))


def _flip(bits, rng) -> list[int]:
    out = list(bits)
    out[rng.randrange(len(out))] ^= 1
    return out


def test_the_rows_layout():
    S = SMALL
    assert [n for n, _c, _w in RC.state_fields(S)] == ["run_valid", "run_id", "run_P", "run_O", "run_out", "run_comp",
                                                       "wait_valid", "wait_id", "wait_P", "wait_O", "wait_out", "free"]
    assert RC.row_fields(S)[12:] == [("qh", 1, 3), ("dh", 1, 3), ("steps", 1, 2)]
    assert RC.window_fields(S) == [("q_valid", 5, 1), ("q_id", 5, 4), ("q_P", 5, 6), ("q_O", 5, 6), ("dec", 6, 1),
                                   ("left", 1, 3)]
    assert RC.front_fields(S) == [("arr_valid", 3, 1), ("arr_id", 3, 4), ("arr_P", 3, 6), ("arr_O", 3, 6)]
    used = 3 * (1 + 4 + 4 * 6) + 2 * (1 + 4 + 3 * 6) + 5 + 3 + 3 + 2
    assert RC.width(S) == S.width == 16 * -(-used // 16) and S.words == RC.width(S) // 16
    assert RC.window_width(S) == S.window_width == 16 * -(-(5 * (1 + 4 + 2 * 6) + 6 + 3) // 16)


def test_the_state_is_replay_steps():
    """The state row's state is ReplayStep_v1's state input and next-state output, field for field, and the front is its
    arrivals input."""
    RS = pytest.importorskip("verity.core.protocols.certifier.replay_step")
    from types import SimpleNamespace

    for S in SHAPES:
        st = SimpleNamespace(**RS.statics(S.R, S.W, S.A, {"max_num_seqs": S.R, "max_num_batched_tokens": 64,
                                                          "max_model_len": S.L, "num_gpu_blocks": S.NB}, S.IW))
        names = {n for n, _c, _w in RC.state_fields(S)}
        arr = {n for n, _c, _w in RC.front_fields(S)}
        assert [f for f in RS._in_fields(st) if f[0] in names] == RC.state_fields(S)
        assert [f for f in RS._out_fields(st) if f[0] in names] == RC.state_fields(S)
        assert [f for f in RS._in_fields(st) if f[0] in arr] == RC.front_fields(S)
        assert [f for f in RS._in_fields(st) if f[0] == "bits"] == [("bits", S.A, 1)]
        state = RS.State((RS.Req(1, 5, 3, 1, 6),), (RS.Req(2, 4, 2, 0, 0),), 11)
        arrivals = ((3, 7, 2), (4, 2, 1))
        args = RS.arguments(st, state, RS.Inputs(tuple(RS.Req(i, p, o, 0, 0) for i, p, o in arrivals)))
        row = RC.Row(((1, 5, 3, 1, 6),), ((2, 4, 2, 0),), 11)
        want = [b for (n, _c, _w), a in zip(RS._in_fields(st), args) if n in names for b in a]
        assert list(RC.state_bits(row, S)) == want
        assert list(RC.front_bits(S, arrivals)) == [b for (n, _c, _w), a in zip(RS._in_fields(st), args) if n in arr for b in a]


@pytest.mark.parametrize("S", SHAPES)
def test_round_trip(S):
    rng = random.Random(1)
    assert RC.decode(RC.encode(RC.empty(S), S), S) == RC.empty(S)
    empty = RC.encode(RC.empty(S), S)
    at = RC._bits_of(RC.state_fields(S)) - RC.widths(S).BW
    assert empty[at:at + RC.widths(S).BW] == tuple((S.NB - 1) >> j & 1 for j in range(RC.widths(S).BW))
    assert sum(empty) == bin(S.NB - 1).count("1")
    assert not any(RC.encode_window(RC.Window(), S))
    for _ in range(200):
        r, w = _rand_row(rng, S), _rand_window(rng, S)
        bits, wbits = RC.encode(r, S), RC.encode_window(w, S)
        assert len(bits) == S.width and RC.decode(bits, S) == r and RC.unrow(RC.row(bits)) == bits
        assert len(wbits) == S.window_width and RC.decode_window(wbits, S) == w and RC.unrow(RC.row(wbits)) == wbits


def _bits_with(S, encoded, fs, name, element, value):
    at = 0
    for n, c, w in fs:
        if n == name:
            bits = list(encoded)
            pos = at + element * w
            bits[pos:pos + w] = [(value >> j) & 1 for j in range(w)]
            return bits
        at += c * w
    raise KeyError(name)


def test_decode_refuses():
    S = SMALL
    r = RC.Row(((1, 5, 3, 1, 6), (2, 4, 2, 0, 4)), ((3, 7, 2, 0),), 11, 1, 2, 2)
    w = RC.Window(((4, 3, 3), (5, 2, 1)), (1, 0, 1))
    rb, wb = RC.encode(r, S), RC.encode_window(w, S)
    fs, wfs = RC.row_fields(S), RC.window_fields(S)
    bad = {
        "a gap in the running slots": _bits_with(S, rb, fs, "run_valid", 0, 0),
        "an empty slot with a bit set": _bits_with(S, rb, fs, "wait_id", 1, 5),
        "qh past Q": _bits_with(S, rb, fs, "qh", 0, 6),
        "dh past D": _bits_with(S, rb, fs, "dh", 0, 7),
        "nonzero padding": list(rb[:-1]) + [1],
        "a short row": list(rb[:-16]),
        "a non-bit": list(rb[:-1]) + [2],
    }
    for why, bits in bad.items():
        with pytest.raises(RC.MalformedRow):
            RC.decode(bits, S)
            pytest.fail(why)
    P2 = RC.Shape(**{**SMALL.statics(), "P": 2})
    with pytest.raises(RC.MalformedRow, match="steps"):
        RC.decode(_bits_with(P2, RC.encode(RC.Row(), P2), RC.row_fields(P2), "steps", 0, 3), P2)
    wbad = {
        "a gap in the queue": _bits_with(S, wb, wfs, "q_valid", 0, 0),
        "an empty queue slot with a bit set": _bits_with(S, wb, wfs, "q_O", 4, 1),
        "a decision past left": _bits_with(S, wb, wfs, "dec", 3, 1),
        "left past D": _bits_with(S, RC.encode_window(RC.Window(), S), wfs, "left", 0, 7),
        "nonzero padding": list(wb[:-1]) + [1],
        "a short row": list(wb[:-16]),
    }
    for why, bits in wbad.items():
        with pytest.raises(RC.MalformedRow):
            RC.decode_window(bits, S)
            pytest.fail(why)


def test_encode_refuses():
    S = SMALL
    for r in (RC.Row(running=((0, 0, 0, 0, 0),) * 4), RC.Row(qh=6), RC.Row(dh=7), RC.Row(steps=4), RC.Row(free=32),
              RC.Row(running=((16, 0, 0, 0, 0),))):
        with pytest.raises(RC.MalformedRow):
            RC.encode(r, S)
    for w in (RC.Window(queue=((0, 0, 0),) * 6), RC.Window(decisions=(1,) * 7), RC.Window(decisions=(2,))):
        with pytest.raises(RC.MalformedRow):
            RC.encode_window(w, S)
    with pytest.raises(RC.MalformedRow):
        RC.front_bits(S, ((0, 0, 0),) * 4)


@pytest.mark.parametrize("n", [1, 3, 4])
def test_front_is_a_slice(n):
    """`_front`'s pruned log shifter is xs[amt:amt + n] for every shift its bits express, past the end included."""
    rng = random.Random(7)
    for length in (1, 5, 8, 13):
        xs = [[rng.randint(0, 1) for _ in range(3)] for _ in range(length)]
        bits = max(1, length.bit_length())
        C = Circuit(cse=True)
        amt, flat = C.input("amt", bits), C.input("xs", 3 * length)
        out = RC._front(C, [flat[3 * k:3 * k + 3] for k in range(length)], amt, n)
        C.output("out", [b for e in out for b in e])
        for s in range(1 << bits):
            outs, _ = C.evaluate({"amt": [(s >> j) & 1 for j in range(bits)], "xs": [b for e in xs for b in e]}, 1)
            assert outs["out"] == [b for e in (xs[s:s + n] + [[0] * 3] * n)[:n] for b in e], (length, n, s)


def _advance_case(S, rng, k):
    win = _rand_window(rng, S)
    cur, stepped = _rand_row(rng, S), _rand_row(rng, S, heads=False)
    cur = RC.Row(cur.running, cur.waiting, cur.free, rng.randint(0, len(win.queue)), rng.randint(0, len(win.decisions)),
                 1 if k % 5 == 0 else cur.steps)
    arrivals, bits = RC.front(S, win, cur)
    top = (1 << RC.widths(S).AC) - 1
    taken = rng.randint(0, len(arrivals)) if rng.random() < 0.8 else rng.randint(0, top)
    rest = len(win.decisions) - cur.dh
    used = rng.randint(0, min(top, rest + 1))
    if k % 7 == 0 and cur.steps == 1 and rest <= top:
        used = rest
    if k % 11 == 0 and arrivals:
        arrivals = arrivals[:-1]
    if k % 13 == 0:
        bits = tuple(b ^ (j == 0) for j, b in enumerate(bits))
    return win, cur, rng.random() < 0.9, stepped, arrivals, bits, taken, used


def _advance_cases(S, rng, n):
    cases, want = [], []
    for k in range(n):
        win, cur, ok, stepped, arrivals, bits, taken, used = _advance_case(S, rng, k)
        ref = RC.advance(S, win, cur, ok, stepped, arrivals, bits, taken, used)
        good = RC.encode(ref, S) if ref is not None else RC.encode(cur, S)
        for nxt in (good, _flip(good, rng)):
            cases.append(RC.advance_arguments(S, win, cur, ok, stepped, arrivals, bits, taken, used, nxt))
            want.append((int(ref is not None and nxt == good), int(cur.steps > 0)))
    return cases, want


@pytest.mark.parametrize("S", SHAPES)
def test_advance_circuit_is_its_specification(S):
    rng = random.Random(2)
    cases, want = _advance_cases(S, rng, 300)
    assert RC.evaluate_advance(S, cases) == want
    assert {w for w in want} >= {(1, 1), (0, 1), (1, 0), (0, 0)}


def test_advance_refuses():
    S = SMALL
    win = RC.Window(((4, 3, 3), (5, 2, 1)), (1, 0, 1))
    cur = RC.Row(((1, 5, 3, 1, 6),), (), 11, 0, 0, 2)
    stepped = RC.Row(((1, 5, 3, 2, 7),), (), 10)
    arr, bits = RC.front(S, win, cur)
    assert arr == ((4, 3, 3), (5, 2, 1)) and bits == (1, 0, 1)
    assert RC.advance(S, win, cur, True, stepped, arr, bits, 1, 2) == RC.Row(((1, 5, 3, 2, 7),), (), 10, 1, 2, 1)
    at = RC.Row(cur.running, (), 11, 1, 2, 1)
    assert RC.front(S, win, at) == (((5, 2, 1),), (1, 0, 0)), "the front at the heads, zeros past left"
    last = RC.Row(cur.running, (), 11, 1, 1, 1)
    idle = RC.Row(cur.running, (), 11, 2, 3, 0)
    cases = {
        "a good step": (cur, True, arr, bits, 1, 2, True),
        "the step's ok": (cur, False, arr, bits, 1, 2, False),
        "an arrival that isn't the window's": (cur, True, (arr[1], arr[0]), bits, 1, 2, False),
        "an arrival left out": (cur, True, arr[:1], bits, 1, 2, False),
        "a decision bit that isn't the window's": (cur, True, arr, (0, 0, 1), 1, 2, False),
        "taken past what is left": (cur, True, arr, bits, 3, 2, False),
        "used past what is left": (at, True, *RC.front(S, win, at), 0, 2, False),
        "the window's last step leaves a decision, which carries": (last, True, *RC.front(S, win, last), 0, 1, True),
        "the window's last step reads the rest": (last, True, *RC.front(S, win, last), 0, 2, True),
        "an idle step leaves the row as it is": (idle, False, (), (0, 0, 0), 3, 3, True),
    }
    for why, (c, ok, a, b, t, u, accepts) in cases.items():
        ref = RC.advance(S, win, c, ok, stepped, a, b, t, u)
        assert (ref is not None) == accepts, why
        nxt = RC.encode(ref if ref is not None else c, S)
        assert RC.evaluate_advance(S, [RC.advance_arguments(S, win, c, ok, stepped, a, b, t, u, nxt)]) == [
            (int(accepts), int(c.steps > 0))], why
    assert RC.advance(S, win, idle, False, stepped, (), (0, 0, 0), 3, 3) == idle
    N = NARROW
    wide = RC.Window(tuple((k, 1, 1) for k in range(6)), (1,))
    full = RC.Row((), (), 5, 0, 0, 1)
    a, b = RC.front(N, wide, full)
    assert len(a) == N.A
    for t, u, accepts, why in ((4, 1, True, "all A taken"), (5, 1, False, "taken past A, with arrivals past A left"),
                               (4, 5, False, "used past A")):
        ref = RC.advance(N, wide, full, True, RC.Row(), a, b, t, u)
        assert (ref is not None) == accepts, why
        nxt = RC.encode(ref if ref is not None else full, N)
        assert RC.evaluate_advance(N, [RC.advance_arguments(N, wide, full, True, RC.Row(), a, b, t, u, nxt)]) == [
            (int(accepts), 1)], why


def _load_case(S, rng, k):
    prev = _rand_row(rng, S, heads=False)
    last = _rand_window(rng, S)
    qh = rng.randint(0, len(last.queue))
    prev = RC.Row(prev.running, prev.waiting, prev.free, qh, len(last.decisions), 0)
    if k % 6 == 0:
        prev = RC.Row(prev.running, prev.waiting, prev.free, qh, rng.randint(0, len(last.decisions)), 0)
    elif k % 6 == 1:
        prev = RC.Row(prev.running, prev.waiting, prev.free, qh, len(last.decisions), 1)
    elif k % 6 == 2 and len(last.decisions) < S.D:
        prev = RC.Row(prev.running, prev.waiting, prev.free, qh, len(last.decisions) + 1, 0)
    arrivals = _rand_arrivals(rng, S, rng.randint(0, S.K))
    decisions = tuple(rng.randint(0, 1) for _ in range(rng.randint(0, S.D)))
    steps = rng.randint(0, (1 << RC.widths(S).PW) - 1)
    return prev, last, arrivals, decisions, steps


def _load_cases(S, rng, n):
    cases, want = [], []
    for k in range(n):
        prev, last, arrivals, decisions, steps = _load_case(S, rng, k)
        ref = RC.load(S, prev, last, arrivals, decisions, steps)
        good = (RC.encode(ref[0], S), RC.encode_window(ref[1], S)) if ref is not None else (
            RC.encode(prev, S), RC.encode_window(last, S))
        for nxt, win in (good, (_flip(good[0], rng), good[1]), (good[0], _flip(good[1], rng))):
            cases.append(RC.load_arguments(S, prev, last, arrivals, decisions, len(decisions), steps, nxt, win))
            want.append(int(ref is not None and (nxt, win) == good))
    return cases, want


@pytest.mark.parametrize("S", SHAPES)
def test_load_circuit_is_its_specification(S):
    rng = random.Random(3)
    cases, want = _load_cases(S, rng, 300)
    assert RC.evaluate_load(S, cases) == want
    assert 0 < sum(want) < len(want)


def test_load_refuses():
    S = SMALL
    last = RC.Window(((3, 1, 1), (4, 3, 3), (5, 2, 1)), (1, 1))
    prev = RC.Row(((1, 5, 3, 1, 6),), (), 11, 1, 2, 0)
    arr = [(6, 3, 2), (7, 1, 1)]
    ok = RC.load(S, prev, last, arr, (1, 0), 2)
    assert ok == (RC.Row(prev.running, (), 11, 0, 0, 2), RC.Window(((4, 3, 3), (5, 2, 1), (6, 3, 2), (7, 1, 1)), (1, 0)))
    P2 = RC.Shape(**{**S.statics(), "P": 2})
    carried = RC.Row(prev.running, (), 11, 1, 1, 0)
    assert RC.load(S, carried, last, arr, (1, 0), 2) == (ok[0], RC.Window(ok[1].queue, (1, 1, 0))), "the unread carried"
    assert RC.load(S, prev, last, arr, (1,), 0) == (RC.Row(prev.running, (), 11, 0, 0, 0), RC.Window(ok[1].queue, (1,))), (
        "decisions in a window with no step, for the next to read")
    refused = {
        "the last window left a step": (S, RC.Row(prev.running, (), 11, 1, 2, 1), last, arr, (1, 0), 2),
        "the last window read past its decisions": (S, RC.Row(prev.running, (), 11, 1, 3, 0), last, arr, (1, 0), 2),
        "the queue overflows Q": (S, RC.Row(prev.running, (), 11, 0, 2, 0), last, arr * 2, (), 1),
        "the carried and the window's decisions overflow D": (S, RC.Row(prev.running, (), 11, 1, 0, 0), last, arr,
                                                               (1, 0, 1, 1, 0), 2),
        "more steps than P": (P2, prev, last, arr, (), 3),
    }
    for why, (shape, p, lw, a, d, s) in refused.items():
        assert RC.load(shape, p, lw, a, d, s) is None, why
        args = RC.load_arguments(shape, p, lw, a[:shape.K], d, len(d), s, RC.encode(p, shape), RC.encode_window(lw, shape))
        assert RC.evaluate_load(shape, [args]) == [0], why
    nxt, win = RC.encode(ok[0], S), RC.encode_window(ok[1], S)
    args = RC.load_arguments(S, prev, last, arr, (1, 0), 2, 2, nxt, win)
    assert RC.evaluate_load(S, [args]) == [1]
    past = RC.load_arguments(S, prev, last, arr, (1, 0, 1), 2, 2, nxt, win)
    assert RC.evaluate_load(S, [past]) == [0], "a decision past the count"
    over = RC.load_arguments(S, prev, last, arr, (1, 0), 7, 2, nxt, win)
    assert RC.evaluate_load(S, [over]) == [0], "a count past D"
    c0, c1 = RC.load(S, carried, last, arr, (1, 0), 2)
    cargs = RC.load_arguments(S, carried, last, arr, (1, 0), 2, 2, RC.encode(c0, S), RC.encode_window(c1, S))
    assert RC.evaluate_load(S, [cargs, list(args[:-1]) + [list(RC.encode_window(c1, S))]]) == [1, 0], (
        "the carried decisions written before the window's, and only when the last window left them")
    gap = list(args)
    gap[2] = list(gap[2])
    gap[2][0], gap[2][2] = 0, 1
    assert RC.evaluate_load(S, [gap]) == [0], "the arrivals' valid slots not a prefix"


def test_load_refuses_a_carry_it_cant_hold():
    """The two refusals the carry adds, each against the rows the circuit writes when that check alone is skipped: the
    carried and the window's decisions past D (2 + 5 of 6: the row cut to D, ``left`` 7), and a last window read past
    its decisions (``dh`` 5 of 2, which wraps the carried count to 5 and writes the window's decision at 5)."""
    S = SMALL
    last = RC.Window(((3, 1, 1),), (1, 1))
    over, past = RC.Row((), (), 11, 0, 0, 0), RC.Row((), (), 11, 0, 5, 0)
    five = (1, 0, 1, 1, 0)
    assert RC.load(S, over, last, (), five, 2) is None and RC.load(S, past, last, (), (1,), 1) is None
    wfs = RC.window_fields(S)
    cut = _bits_with(S, RC.encode_window(RC.Window(((3, 1, 1),), ((1, 1) + five)[:S.D]), S), wfs, "left", 0, 7)
    wrapped = RC.encode_window(RC.Window(((3, 1, 1),), (0, 0, 0, 0, 0, 1)), S)
    cases = [RC.load_arguments(S, over, last, (), five, len(five), 2, RC.encode(RC.Row((), (), 11, 0, 0, 2), S), cut),
             RC.load_arguments(S, past, last, (), (1,), 1, 1, RC.encode(RC.Row((), (), 11, 0, 0, 1), S), wrapped)]
    assert RC.evaluate_load(S, cases) == [0, 0]


def _close_cases(S, rng, n):
    cases, want = [], []
    for k in range(n):
        win, row = _rand_window(rng, S), _rand_row(rng, S)
        dh = len(win.decisions) if k % 2 else rng.randint(0, S.D)
        row = RC.Row(row.running, row.waiting, row.free, row.qh, dh, 0 if k % 3 else row.steps)
        cases.append(RC.close_arguments(S, row, win))
        want.append(int(RC.close(S, row, win)))
    return cases, want


@pytest.mark.parametrize("S", SHAPES)
def test_close_circuit_is_its_specification(S):
    cases, want = _close_cases(S, random.Random(7), 200)
    assert RC.evaluate_close(S, cases) == want
    assert 0 < sum(want) < len(want)


def test_a_window_leaves_decisions_to_the_next_and_the_session_reads_them_all():
    """A5 charges a decision up to two windows before its step's: window 0 brings three decisions and its one step reads
    one, window 1 brings none and reads one, window 2 brings one and reads the last two. Each load writes the window's
    decisions after the unread ones, each part's circuit accepts what its specification does, and `ReplayClose`
    accepts the session; ended after window 0 or 1, with decisions unread, it is refused."""
    S = SMALL
    cur, last, ends = RC.empty(S), RC.Window(), []
    for decisions, reads in (((1, 0, 1), [1]), ((), [1]), ((0,), [2])):
        loaded = RC.load(S, cur, last, (), decisions, len(reads))
        assert loaded is not None
        args = RC.load_arguments(S, cur, last, (), decisions, len(decisions), len(reads), RC.encode(loaded[0], S),
                                 RC.encode_window(loaded[1], S))
        assert RC.evaluate_load(S, [args]) == [1]
        cur, win = loaded
        for u in range(S.P):
            used = reads[u] if u < len(reads) else 0
            arr, bits = RC.front(S, win, cur)
            stepped = RC.Row(free=cur.free)
            nxt = RC.advance(S, win, cur, True, stepped, arr, bits, 0, used)
            assert nxt is not None
            args = RC.advance_arguments(S, win, cur, True, stepped, arr, bits, 0, used, RC.encode(nxt, S))
            assert RC.evaluate_advance(S, [args])[0][0] == 1
            cur = nxt
        ends.append((cur, win))
        last = win
    assert [w.decisions for _r, w in ends] == [(1, 0, 1), (0, 1), (1, 0)] and [r.dh for r, _w in ends] == [1, 1, 2]
    assert [RC.close(S, r, w) for r, w in ends] == [False, False, True]
    assert RC.evaluate_close(S, [RC.close_arguments(S, r, w) for r, w in ends]) == [0, 0, 1]


def _session(S, rng):
    """A two-window session of rows through `load` and `advance`, with made-up step outputs: each window's state rows,
    its window row and its units."""
    windows, cur, last, rid = [], RC.empty(S), RC.Window(), 0
    for w in range(2):
        n = S.P if w == 0 else 1
        arrivals = [(rid + k, 3 + k, 2) for k in range(2)]
        rid += 2
        decisions = (1, 0)
        rows = [cur]
        loaded = RC.load(S, cur, last, arrivals, decisions, n)
        assert loaded is not None
        cur, win = loaded
        rows.append(cur)
        units = [("load", rows[0], last, arrivals, decisions, n)]
        for u in range(S.P):
            stepped = _rand_row(rng, S, heads=False)
            taken, used = (1, 1) if u < n - 1 else (0, len(decisions) - cur.dh) if u == n - 1 else (0, 0)
            arr, bits = RC.front(S, win, cur)
            nxt = RC.advance(S, win, cur, True, stepped, arr, bits, taken, used)
            assert nxt is not None
            units.append(("step", cur, stepped, arr, bits, taken, used))
            cur = nxt
            rows.append(cur)
        windows.append((rows, win, units))
        last = win
    return windows


def test_a_session_through_the_chain():
    S, rng = SMALL, random.Random(4)
    values, carry = [], None
    for rows, win, units in _session(S, rng):
        v = RC.commit_window(PROGRAM, rows, win, S, REGISTRATION, carry=carry)
        assert v.state.leaves == S.P + 2 and v.state.words == S.words
        assert v.window.leaves == 2 and v.window.words == S.window_width // 16
        for unit, reads in zip(units, RC.reads(S)):
            got = {}
            for value, at in ((v.state, reads[RC.VALUE]), (v.window, reads[RC.WINDOW])):
                for name, p in at.items():
                    assert G.verify_read(value.domain, value.root, p, value.bcs[p], value.path(p), RC.SCHEMA)
                    got[name] = RC.unrow(value.rows[p])
            assert list(got["window"]) == list(RC.encode_window(win, S))
            if unit[0] == "load":
                _k, prev, last, arrivals, decisions, n = unit
                assert sorted(got) == ["last", "next", "prev", "window"]
                assert list(got["prev"]) == list(RC.encode(prev, S)) and list(got["last"]) == list(RC.encode_window(last, S))
                args = RC.load_arguments(S, prev, last, arrivals, decisions, len(decisions), n, got["next"], got["window"])
                assert RC.evaluate_load(S, [args]) == [1]
            else:
                _k, cur, stepped, arr, bits, taken, used = unit
                assert sorted(got) == ["next", "row", "window"] and list(got["row"]) == list(RC.encode(cur, S))
                args = RC.advance_arguments(S, win, cur, True, stepped, arr, bits, taken, used, got["next"])
                assert RC.evaluate_advance(S, [args])[0][0] == 1
        values.append(v)
        carry = v
    end = {}
    for value, at in ((v.state, RC.close_reads(S)[RC.VALUE]), (v.window, RC.close_reads(S)[RC.WINDOW])):
        for name, p in at.items():
            assert G.verify_read(value.domain, value.root, p, value.bcs[p], value.path(p), RC.SCHEMA)
            end[name] = RC.unrow(value.rows[p])
    assert RC.evaluate_close(S, [[list(end["row"]), list(end["window"])]]) == [1]
    ops = [RC.openings(v) for v in values]
    assert (ops[0][0][0], ops[0][1][0]) == RC.empty_openings(PROGRAM, S)
    assert ops[1][0][0] == ops[0][0][-1] and ops[1][1][0] == ops[0][1][-1]
    assert RC.check_chain(ops, PROGRAM, S) is None


def _fresh(S, value, rows):
    return G.Registered.commit(PROGRAM, value, [RC.row(r) for r in rows], word_bits=16, schema=RC.SCHEMA,
                               produced=REGISTRATION)


def test_check_chain_refuses():
    S, rng = SMALL, random.Random(5)
    (rows0, win0, _), (rows1, win1, _) = _session(S, rng)
    v0 = RC.commit_window(PROGRAM, rows0, win0, S, REGISTRATION)
    v1 = RC.commit_window(PROGRAM, rows1, win1, S, REGISTRATION, carry=v0)
    (s0, w0), (s1, w1) = RC.openings(v0), RC.openings(v1)
    assert RC.check_chain([(s0[:-1], w0), (s1, w1)], PROGRAM, S).startswith("C-rows")
    assert RC.check_chain([(s0, w0[:1]), (s1, w1)], PROGRAM, S).startswith("C-rows")
    fresh = list(_fresh(S, RC.VALUE, [RC.encode(r, S) for r in rows0]).bcs)
    assert RC.check_chain([(fresh, w0), (s1, w1)], PROGRAM, S).startswith("C-first"), "the empty state under a fresh salt"
    wfresh = list(_fresh(S, RC.WINDOW, [RC.encode_window(RC.Window(), S), RC.encode_window(win0, S)]).bcs)
    assert RC.check_chain([(s0, wfresh), (s1, w1)], PROGRAM, S).startswith("C-first"), "the empty window under a fresh salt"
    assert RC.check_chain([(s0, w0), (s1, w1)], "another-program", S).startswith("C-first"), "another program's public salts"
    other = list(_fresh(S, RC.VALUE, [RC.encode(r, S) for r in rows1]).bcs)
    assert RC.check_chain([(s0, w0), (other, w1)], PROGRAM, S).startswith("C-carry"), "the carried state under a fresh salt"
    wother = list(_fresh(S, RC.WINDOW, [RC.encode_window(win0, S), RC.encode_window(win1, S)]).bcs)
    assert RC.check_chain([(s0, w0), (s1, wother)], PROGRAM, S).startswith("C-carry"), "the carried window under a fresh salt"
    swapped = list(s1)
    swapped[0] = s0[-2]
    assert RC.check_chain([(s0, w0), (swapped, w1)], PROGRAM, S).startswith("C-carry"), "another row of the last window"
    assert RC.check_chain([], PROGRAM, S) is None


def test_commit_window_refuses():
    S, rng = SMALL, random.Random(6)
    (rows0, win0, _), (rows1, win1, _) = _session(S, rng)
    v0 = RC.commit_window(PROGRAM, rows0, win0, S, REGISTRATION)
    with pytest.raises(ValueError, match="rows"):
        RC.commit_window(PROGRAM, rows0[:-1], win0, S, REGISTRATION)
    with pytest.raises(ValueError, match="empty state"):
        RC.commit_window(PROGRAM, rows1, win1, S, REGISTRATION)
    with pytest.raises(ValueError, match="ended in"):
        RC.commit_window(PROGRAM, rows0, win0, S, REGISTRATION, carry=v0)
    with pytest.raises(ValueError, match="salts"):
        RC.commit_window(PROGRAM, rows1, win1, S, REGISTRATION, carry=v0, salts=[bytes(64)])


@pytest.mark.parametrize("S", SHAPES)
def test_size(S):
    got = RC.size(S)
    assert got["ands_per_step"] == ands(RC.advance_unit(S)) and got["ands_load"] == ands(RC.load_unit(S))
    assert got["ands_close"] == ands(RC.close_unit(S))
    assert got["ands_per_window"] == S.P * got["ands_per_step"] + got["ands_load"]
    assert got["width"] == S.width and got["window_width"] == S.window_width and got["rows_per_window"] == S.P + 2
    assert got["reads_per_window"] == 3 * S.P + 4 and got["record_bytes_per_window"] == 128 * (S.P + 4)
    assert got["path_depth"] == max(1, (S.P + 1).bit_length())
