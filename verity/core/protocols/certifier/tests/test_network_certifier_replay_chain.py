"""The replay's state chain (`replay_chain`): the row's layout and its refusals, `ReplayAdvance` and `ReplayLoad` against
`advance` and `load` on random and edge cases, the window's produced value opened at every unit's read, the carry and
the empty start (`check_chain`), a two-window session through both parts, and the cost `size` reports."""
from __future__ import annotations

import random

import pytest

from verity.core.protocols.certifier import replay_chain as RC
from verity.core.protocols.certifier.head_check import ands
from verity.core.protocols.sampled_proofs.one_stage import registered as G

SMALL = RC.Shape(R=3, W=2, A=3, L=40, NB=20, IW=4, Q=5, D=6, P=3, K=3)
#: A a power of two (AC = 3 > DW = 1), one decision and one step a window
NARROW = RC.Shape(R=2, W=1, A=4, L=16, NB=8, IW=3, Q=6, D=1, P=1, K=2)
SHAPES = (SMALL, NARROW)
PROGRAM, REGISTRATION = "replay-chain-test", "ab" * 64


def _rand_row(rng: random.Random, S, *, window: bool = True) -> RC.Row:
    Z, top = RC.widths(S), (1 << S.IW) - 1

    def u(w):
        return rng.randrange(1 << w)

    return RC.Row(tuple((u(S.IW), u(Z.LW), u(Z.LW), u(Z.LW), u(Z.LW)) for _ in range(rng.randint(0, S.R))),
                  tuple((u(S.IW), u(Z.LW), u(Z.LW), u(Z.LW)) for _ in range(rng.randint(0, S.W))), rng.randrange(S.NB),
                  tuple((rng.randint(0, top), u(Z.LW), u(Z.LW)) for _ in range(rng.randint(0, S.Q))) if window else (),
                  tuple(rng.randint(0, 1) for _ in range(rng.randint(0, S.D))) if window else (),
                  rng.randint(0, S.P) if window else 0)


def _flip(bits, rng) -> list[int]:
    out = list(bits)
    out[rng.randrange(len(out))] ^= 1
    return out


def test_the_row_layout():
    S = SMALL
    assert [n for n, _c, _w in RC.state_fields(S)] == ["run_valid", "run_id", "run_P", "run_O", "run_out", "run_comp",
                                                       "wait_valid", "wait_id", "wait_P", "wait_O", "wait_out", "free"]
    assert RC.row_fields(S)[12:] == [("q_valid", 5, 1), ("q_id", 5, 4), ("q_P", 5, 6), ("q_O", 5, 6), ("dec", 6, 1),
                                     ("left", 1, 3), ("steps", 1, 2)]
    used = 3 * (1 + 4 + 4 * 6) + 2 * (1 + 4 + 3 * 6) + 5 + 5 * (1 + 4 + 2 * 6) + 6 + 3 + 2
    assert RC.width(S) == SMALL.width == 16 * -(-used // 16) and SMALL.words == RC.width(S) // 16


def test_the_state_is_replay_steps():
    """The row's state is ReplayStep_v1's state input and next-state output, field for field."""
    RS = pytest.importorskip("verity.core.protocols.certifier.replay_step")
    from types import SimpleNamespace

    for S in SHAPES:
        st = SimpleNamespace(**RS.statics(S.R, S.W, S.A, {"max_num_seqs": S.R, "max_num_batched_tokens": 64,
                                                          "max_model_len": S.L, "num_gpu_blocks": S.NB}, S.IW))
        names = {n for n, _c, _w in RC.state_fields(S)}
        assert [f for f in RS._in_fields(st) if f[0] in names] == RC.state_fields(S)
        assert [f for f in RS._out_fields(st) if f[0] in names] == RC.state_fields(S)
        state = RS.State((RS.Req(1, 5, 3, 1, 6),), (RS.Req(2, 4, 2, 0, 0),), 11)
        args = RS.arguments(st, state, RS.Inputs(()))
        row = RC.Row(((1, 5, 3, 1, 6),), ((2, 4, 2, 0),), 11)
        want = [b for (n, _c, _w), a in zip(RS._in_fields(st), args) if n in names for b in a]
        assert list(RC.state_bits(row, S)) == want


@pytest.mark.parametrize("S", SHAPES)
def test_round_trip(S):
    rng = random.Random(1)
    assert RC.decode(RC.encode(RC.empty(S), S), S) == RC.empty(S)
    empty = RC.encode(RC.empty(S), S)
    at = RC._bits_of(RC.state_fields(S)) - RC.widths(S).BW
    assert empty[at:at + RC.widths(S).BW] == tuple((S.NB - 1) >> j & 1 for j in range(RC.widths(S).BW))
    assert sum(empty) == bin(S.NB - 1).count("1")
    for _ in range(200):
        r = _rand_row(rng, S)
        bits = RC.encode(r, S)
        assert len(bits) == S.width and RC.decode(bits, S) == r and RC.unrow(RC.row(bits)) == bits


def _bits_with(S, row, name, element, value):
    fs = RC.row_fields(S)
    at = 0
    for n, c, w in fs:
        if n == name:
            bits = list(RC.encode(row, S))
            pos = at + element * w
            bits[pos:pos + w] = [(value >> j) & 1 for j in range(w)]
            return bits
        at += c * w
    raise KeyError(name)


def test_decode_refuses():
    S = SMALL
    r = RC.Row(((1, 5, 3, 1, 6), (2, 4, 2, 0, 4)), ((3, 7, 2, 0),), 11, ((4, 3, 3),), (1, 0), 2)
    bad = {
        "a gap in the running slots": _bits_with(S, r, "run_valid", 0, 0),
        "an empty slot with a bit set": _bits_with(S, r, "wait_id", 1, 5),
        "an empty queue slot with a bit set": _bits_with(S, r, "q_O", 4, 1),
        "a decision past left": _bits_with(S, r, "dec", 3, 1),
        "left past D": _bits_with(S, RC.Row(), "left", 0, 7),
        "nonzero padding": list(RC.encode(r, S)[:-1]) + [1],
        "a short row": list(RC.encode(r, S)[:-16]),
        "a non-bit": list(RC.encode(r, S)[:-1]) + [2],
    }
    for why, bits in bad.items():
        with pytest.raises(RC.MalformedRow):
            RC.decode(bits, S)
    P2 = RC.Shape(**{**SMALL.statics(), "P": 2})
    with pytest.raises(RC.MalformedRow, match="steps"):
        RC.decode(_bits_with(P2, RC.Row(), "steps", 0, 3), P2)


def test_encode_refuses():
    S = SMALL
    for r in (RC.Row(running=((0, 0, 0, 0, 0),) * 4), RC.Row(queue=((0, 0, 0),) * 6), RC.Row(decisions=(1,) * 7),
              RC.Row(steps=4), RC.Row(free=32), RC.Row(running=((16, 0, 0, 0, 0),)), RC.Row(decisions=(2,))):
        with pytest.raises(RC.MalformedRow):
            RC.encode(r, S)


def _advance_cases(S, rng, n):
    cases, want = [], []
    for k in range(n):
        cur, stepped = _rand_row(rng, S), _rand_row(rng, S, window=False)
        if k % 5 == 0:
            cur = RC.Row(cur.running, cur.waiting, cur.free, cur.queue, cur.decisions, 1)
        top = (1 << RC.widths(S).AC) - 1
        taken, used, ok = rng.randint(0, top), rng.randint(0, min(top, len(cur.decisions) + 1)), rng.random() < 0.9
        if k % 7 == 0 and cur.steps == 1 and len(cur.decisions) <= top:
            used = len(cur.decisions)
        ref = RC.advance(S, cur, ok, stepped, taken, used)
        good = RC.encode(ref, S) if ref is not None else RC.encode(cur, S)
        for nxt in (good, _flip(good, rng)):
            cases.append(RC.advance_arguments(S, cur, ok, stepped, taken, used, nxt))
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
    cur = RC.Row(((1, 5, 3, 1, 6),), (), 11, ((4, 3, 3), (5, 2, 1)), (1, 0, 1), 2)
    stepped = RC.Row(((1, 5, 3, 2, 7),), (), 10)
    assert RC.advance(S, cur, True, stepped, 1, 2) == RC.Row(((1, 5, 3, 2, 7),), (), 10, ((5, 2, 1),), (1,), 1)
    assert RC.advance(S, cur, False, stepped, 1, 2) is None, "the step's ok"
    few = RC.Row(cur.running, (), 11, cur.queue, (1, 0), 2)
    assert RC.advance(S, few, True, stepped, 1, 3) is None, "more decisions read than are left"
    last = RC.Row(cur.running, (), 11, cur.queue, (1, 0, 1), 1)
    assert RC.advance(S, last, True, stepped, 0, 2) is None, "the window's last step leaves a decision"
    assert RC.advance(S, last, True, stepped, 0, 3).steps == 0
    idle = RC.Row(cur.running, (), 11, cur.queue)
    assert RC.advance(S, idle, False, stepped, 3, 3) == idle, "an idle step leaves the row as it is"
    for c, ok, st, t, u in ((cur, True, stepped, 1, 2), (cur, False, stepped, 1, 2), (few, True, stepped, 1, 3),
                            (last, True, stepped, 0, 2), (idle, False, stepped, 3, 3)):
        ref = RC.advance(S, c, ok, st, t, u)
        nxt = RC.encode(ref if ref is not None else c, S)
        assert RC.evaluate_advance(S, [RC.advance_arguments(S, c, ok, st, t, u, nxt)]) == [(int(ref is not None),
                                                                                             int(c.steps > 0))]


def _load_cases(S, rng, n):
    cases, want = [], []
    Z = RC.widths(S)
    for k in range(n):
        prev = _rand_row(rng, S, window=False)
        if k % 6 == 0:
            prev = RC.Row(prev.running, prev.waiting, prev.free, (), (1,) if k % 12 == 0 else (), 0 if k % 12 == 0 else 1)
        else:
            prev = RC.Row(prev.running, prev.waiting, prev.free, tuple((rng.randrange(1 << S.IW), rng.randrange(S.L), 1)
                                                                       for _ in range(rng.randint(0, S.Q))))
        arrivals = [(rng.randrange(1 << S.IW), rng.randrange(1 << Z.LW), rng.randrange(1 << Z.LW))
                    for _ in range(rng.randint(0, S.K))]
        decisions = tuple(rng.randint(0, 1) for _ in range(rng.randint(0, S.D)))
        steps = rng.randint(0, (1 << Z.PW) - 1)
        ref = RC.load(S, prev, arrivals, decisions, steps)
        good = RC.encode(ref, S) if ref is not None else RC.encode(prev, S)
        for nxt in (good, _flip(good, rng)):
            cases.append(RC.load_arguments(S, prev, arrivals, decisions, len(decisions), steps, nxt))
            want.append(int(ref is not None and nxt == good))
    return cases, want


@pytest.mark.parametrize("S", SHAPES)
def test_load_circuit_is_its_specification(S):
    rng = random.Random(3)
    cases, want = _load_cases(S, rng, 300)
    assert RC.evaluate_load(S, cases) == want
    assert 0 < sum(want) < len(want)


def test_load_refuses():
    S = SMALL
    prev = RC.Row(((1, 5, 3, 1, 6),), (), 11, ((4, 3, 3), (5, 2, 1)))
    arr = [(6, 3, 2), (7, 1, 1)]
    ok = RC.load(S, prev, arr, (1, 0), 2)
    assert ok == RC.Row(prev.running, (), 11, ((4, 3, 3), (5, 2, 1), (6, 3, 2), (7, 1, 1)), (1, 0), 2)
    P2 = RC.Shape(**{**S.statics(), "P": 2})
    refused = {
        "the last window left a decision": (S, RC.Row(prev.running, (), 11, prev.queue, (1,), 0), arr, (1, 0), 2),
        "the last window left a step": (S, RC.Row(prev.running, (), 11, prev.queue, (), 1), arr, (1, 0), 2),
        "the queue overflows Q": (S, RC.Row(prev.running, (), 11, prev.queue * 2), arr, (), 1),
        "more steps than P": (P2, prev, arr, (), 3),
        "decisions with no step": (S, prev, arr, (1,), 0),
    }
    for why, (shape, p, a, d, s) in refused.items():
        assert RC.load(shape, p, a, d, s) is None, why
        nxt = RC.encode(p, shape)
        assert RC.evaluate_load(shape, [RC.load_arguments(shape, p, a, d, len(d), s, nxt)]) == [0], why
    nxt = RC.encode(ok, S)
    args = RC.load_arguments(S, prev, arr, (1, 0), 2, 2, nxt)
    assert RC.evaluate_load(S, [args]) == [1]
    past = RC.load_arguments(S, prev, arr, (1, 0, 1), 2, 2, nxt)
    assert RC.evaluate_load(S, [past]) == [0], "a decision past the count"
    over = RC.load_arguments(S, prev, arr, (1, 0), 7, 2, nxt)
    assert RC.evaluate_load(S, [over]) == [0], "a count past D"
    gap = list(args)
    gap[1] = list(gap[1])
    gap[1][0], gap[1][2] = 0, 1
    assert RC.evaluate_load(S, [gap]) == [0], "the arrivals' valid slots not a prefix"


def _session(S, rng):
    """A two-window session of rows through `load` and `advance`, with made-up step outputs."""
    windows, cur, rid = [], RC.empty(S), 0
    for w in range(2):
        n = S.P if w == 0 else 1
        arrivals = [(rid + k, 3 + k, 2) for k in range(2)]
        rid += 2
        decisions = (1, 0)
        rows = [cur]
        cur = RC.load(S, cur, arrivals, decisions, n)
        assert cur is not None
        rows.append(cur)
        units = [("load", rows[0], arrivals, decisions, n, cur)]
        for u in range(S.P):
            stepped = _rand_row(rng, S, window=False)
            taken, used = (1, 1) if u < n - 1 else (0, len(cur.decisions)) if u == n - 1 else (0, 0)
            nxt = RC.advance(S, cur, True, stepped, taken, used)
            assert nxt is not None
            units.append(("step", cur, stepped, taken, used, nxt))
            cur = nxt
            rows.append(cur)
        windows.append((rows, units))
    return windows


def test_a_session_through_the_chain():
    S, rng = SMALL, random.Random(4)
    values, carry = [], None
    for rows, units in _session(S, rng):
        v = RC.commit_window(PROGRAM, rows, S, REGISTRATION, carry=carry)
        assert v.leaves == S.P + 2 and v.words == S.words
        for unit, reads in zip(units, RC.reads(S)):
            positions = list(reads.values())
            assert positions == [positions[0], positions[0] + 1]
            for p in positions:
                assert G.verify_read(v.domain, v.root, p, v.bcs[p], v.path(p), RC.SCHEMA)
                assert len(v.path(p)) == 64 * RC.size(S)["path_depth"]
            a, b = (RC.unrow(v.rows[p]) for p in positions)
            if unit[0] == "load":
                _k, prev, arrivals, decisions, n, _ = unit
                assert RC.evaluate_load(S, [RC.load_arguments(S, prev, arrivals, decisions, len(decisions), n, b)]) == [1]
                assert list(a) == list(RC.encode(prev, S))
            else:
                _k, cur, stepped, taken, used, _ = unit
                assert list(a) == list(RC.encode(cur, S))
                assert RC.evaluate_advance(S, [RC.advance_arguments(S, cur, True, stepped, taken, used, b)])[0][0] == 1
        values.append(v)
        carry = v
    ops = [RC.openings(v) for v in values]
    assert ops[0][0] == RC.empty_opening(PROGRAM, S)
    assert ops[1][0] == ops[0][-1]
    assert RC.check_chain(ops, PROGRAM, S) is None


def test_check_chain_refuses():
    S, rng = SMALL, random.Random(5)
    (rows0, _), (rows1, _) = _session(S, rng)
    v0 = RC.commit_window(PROGRAM, rows0, S, REGISTRATION)
    v1 = RC.commit_window(PROGRAM, rows1, S, REGISTRATION, carry=v0)
    ops0, ops1 = RC.openings(v0), RC.openings(v1)
    assert RC.check_chain([ops0[:-1], ops1], PROGRAM, S).startswith("C-rows")
    fresh = G.Registered.commit(PROGRAM, RC.VALUE, [RC.row(RC.encode(r, S)) for r in rows0], word_bits=16, schema=RC.SCHEMA,
                                produced=REGISTRATION)
    assert RC.check_chain([RC.openings(fresh), ops1], PROGRAM, S).startswith("C-first"), "the empty row under a fresh salt"
    assert RC.check_chain([ops0, ops1], "another-program", S).startswith("C-first"), "another program's public salt"
    v1_fresh = RC.commit_window(PROGRAM, rows1, S, REGISTRATION, carry=v0, salts=None)
    other = G.Registered.commit(PROGRAM, RC.VALUE, [RC.row(RC.encode(r, S)) for r in rows1], word_bits=16, schema=RC.SCHEMA,
                                produced=REGISTRATION)
    assert RC.check_chain([ops0, RC.openings(other)], PROGRAM, S).startswith("C-carry"), "the carried row under a fresh salt"
    assert RC.check_chain([ops0, RC.openings(v1_fresh)], PROGRAM, S) is None
    swapped = list(ops1)
    swapped[0] = ops0[-2]
    assert RC.check_chain([ops0, swapped], PROGRAM, S).startswith("C-carry"), "another row of the last window"
    assert RC.check_chain([], PROGRAM, S) is None


def test_commit_window_refuses():
    S, rng = SMALL, random.Random(6)
    (rows0, _), (rows1, _) = _session(S, rng)
    v0 = RC.commit_window(PROGRAM, rows0, S, REGISTRATION)
    with pytest.raises(ValueError, match="rows"):
        RC.commit_window(PROGRAM, rows0[:-1], S, REGISTRATION)
    with pytest.raises(ValueError, match="empty state"):
        RC.commit_window(PROGRAM, rows1, S, REGISTRATION)
    with pytest.raises(ValueError, match="ended in"):
        RC.commit_window(PROGRAM, rows0, S, REGISTRATION, carry=v0)
    with pytest.raises(ValueError, match="salts"):
        RC.commit_window(PROGRAM, rows1, S, REGISTRATION, carry=v0, salts=[bytes(64)])


@pytest.mark.parametrize("S", SHAPES)
def test_size(S):
    got = RC.size(S)
    assert got["ands_per_step"] == ands(RC.advance_unit(S)) and got["ands_load"] == ands(RC.load_unit(S))
    assert got["ands_per_window"] == S.P * got["ands_per_step"] + got["ands_load"]
    assert got["width"] == S.width and got["rows_per_window"] == S.P + 2 and got["reads_per_window"] == 2 * (S.P + 1)
    assert got["record_bytes_per_window"] == 128 * (S.P + 2)
    assert got["path_depth"] == max(1, (S.P + 1).bit_length())
