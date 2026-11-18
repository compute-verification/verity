"""The replay's state chain at the certified tick's rows (`replay_chain` with TAU): the state row is TickStep_v1's state
and clock and the front its arrivals and buckets; `TickAdvance_v1`, `TickLoad_v1` and `TickClose_v1` against
`advance`, `load` and `close` on random rows with clocks; a two-window chain of the replay's steps, each step
`tick_step.reference` on what the chain shows, carrying T as `Replay.clock` does; and the seam between two windows: one
that ends mid-bucket, one across which several buckets pass, and the chain refusing a window that starts at another
clock."""
from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest

from tests.compiler import test_network_attester_replay_chain as CT
from tests.compiler import test_network_attester_tick_step as TC
from tests.pipeline import test_tick_step as TT
from verity.protocols.proofs.spot.verifier.one_stage import produced
from verity.commitments.gates.bits import ands
from verity_vllm.compiler import replay_chain as RC
from verity_vllm.compiler import replay_step as RS
from verity_vllm.compiler import tick_step as TK
from verity_vllm.pipeline.advice_code import replay as RP
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.pipeline.advice_code import tick_step as TS

#: the replay chain's shapes with A ≥ R + 1, as TickStep_v1 needs
SMALL = RC.Shape(**{**CT.SMALL.statics(), "A": 4, "TAU": 5})
ASYNC = RC.Shape(**{**CT.ASYNC.statics(), "A": 4, "TAU": 100_000})
SERVED = RC.Shape(**{**CT.SERVED.statics(), "TAU": 10**8})
SHAPES = (SMALL, ASYNC, SERVED)


def _clocked(rng: random.Random, S: RC.Shape, row: RC.Row) -> RC.Row:
    return dataclasses.replace(row, clock=rng.randrange((1 << TK.QW) * S.TAU))


def test_the_rows_are_tick_steps():
    """At each shape the state row's state is TickStep_v1's state input with its clock and its next state's tail, and
    the front is its arrivals with their buckets, field for field; without TAU the rows and the parts are v1's."""
    for S in SHAPES:
        st = TK.statics(S.R, S.W, S.A, {"max_num_seqs": S.R, "max_num_batched_tokens": 64, "max_model_len": S.L,
                                        "num_gpu_blocks": S.NB, "async_scheduling": bool(S.AS)}, S.IW, EW=S.EW,
                        tau=S.TAU, sat=S.TAU // 2, free=S.TAU // 2)
        n, ins = len(RC.state_fields(S)), TK._in_fields(SimpleNamespace(**st))
        assert RC.state_fields(S) == ins[:n] == TK._out_fields(SimpleNamespace(**st))[-n:]
        assert RC.state_fields(S)[-2:] == [("tick_q", 1, 32), ("tick_r", 1, max((S.TAU - 1).bit_length(), 1))]
        assert RC.front_fields(S) == [f for f in ins if f[0].startswith("arr_")]
        assert RC.front_fields(S)[-1] == ("arr_bucket", S.A, 32) and RC.arrival_names(S)[-1] == "bucket"
        assert RC.arrival_fields(S)[-1] == ("a_bucket", S.K, 32) and RC.queue_fields(S)[-1] == ("q_bucket", S.Q, 32)
        v1 = RC.Shape(**{k: v for k, v in S.statics().items() if k != "TAU"})
        assert RC.state_fields(v1) == RC.state_fields(S)[:-2] and "bucket" not in RC.arrival_names(v1)
        assert [f(S).id.split("{")[0] for f in (RC.advance_unit, RC.load_unit, RC.close_unit)] == [
            "TickAdvance_v1", "TickLoad_v1", "TickClose_v1"]
        assert [f(v1).id.split("{")[0] for f in (RC.advance_unit, RC.load_unit, RC.close_unit)] == [
            "ReplayAdvance_v1", "ReplayLoad_v1", "ReplayClose_v1"]
    with pytest.raises(ValueError, match="TAU ≥ 1"):
        RC.TickAdvance.bind(**{**CT.SMALL.statics(), "TAU": 0}).signature()


def test_round_trip_and_refusals():
    S, rng = SMALL, random.Random(11)
    for _ in range(200):
        r = _clocked(rng, S, CT._rand_row(rng, S))
        assert RC.decode(RC.encode(r, S), S) == r
        w = CT._rand_window(rng, S)
        assert RC.decode_window(RC.encode_window(w, S), S) == w
    top = RC.Row(clock=(1 << TK.QW) * S.TAU - 1)
    assert RC.decode(RC.encode(top, S), S) == top
    with pytest.raises(RC.MalformedRow, match="tick_q"):
        RC.encode(RC.Row(clock=(1 << TK.QW) * S.TAU), S)
    with pytest.raises(RC.MalformedRow, match="without the tick"):
        RC.encode(RC.Row(clock=1), CT.SMALL)
    bits = list(RC.encode(RC.Row(clock=7), S))
    at = RC._bits_of(RC.state_fields(S)[:-1])
    bits[at:at + 3] = [1, 0, 1]
    with pytest.raises(RC.MalformedRow, match="remainder 5 isn't below TAU = 5"):
        RC.decode(bits, S)


@pytest.mark.parametrize("S", SHAPES[:2])
def test_the_parts_are_their_specifications(S):
    """The tick's parts on random rows with clocks, and each case with one bit of its next rows flipped: `TickAdvance_v1`
    takes the step's clock into the next row, `TickLoad_v1` state row 0's and an idle step the row's own."""
    rng = random.Random(12)
    cases, want = [], []
    for k in range(300):
        win, cur, ok, stepped, arrivals, bits, taken, used = CT._advance_case(S, rng, k)
        cur, stepped = _clocked(rng, S, cur), _clocked(rng, S, stepped)
        ref = RC.advance(S, win, cur, ok, stepped, arrivals, bits, taken, used)
        assert ref is None or ref.clock == (stepped.clock if cur.steps else cur.clock)
        good = RC.encode(ref if ref is not None else cur, S)
        for nxt in (good, CT._flip(good, rng)):
            cases.append(RC.advance_arguments(S, win, cur, ok, stepped, arrivals, bits, taken, used, nxt))
            want.append((int(ref is not None and nxt == good), int(cur.steps > 0)))
    assert RC.evaluate_advance(S, cases) == want and {(1, 1), (1, 0)} <= set(want)
    cases, loads = [], []
    for k in range(200):
        prev, last, arrivals, decisions, steps = CT._load_case(S, rng, k)
        prev = _clocked(rng, S, prev)
        ref = RC.load(S, prev, last, arrivals, decisions, steps)
        assert ref is None or ref[0].clock == prev.clock
        good = (RC.encode(ref[0], S), RC.encode_window(ref[1], S)) if ref is not None else (
            RC.encode(prev, S), RC.encode_window(last, S))
        for nxt, win in (good, (CT._flip(good[0], rng), good[1]), (good[0], CT._flip(good[1], rng))):
            cases.append(RC.load_arguments(S, prev, last, arrivals, decisions, len(decisions), steps, nxt, win))
            loads.append(int(ref is not None and (nxt, win) == good))
    assert RC.evaluate_load(S, cases) == loads and 0 < sum(loads) < len(loads)
    cases, closes = [], []
    for k in range(100):
        win, row = CT._rand_window(rng, S), _clocked(rng, S, CT._rand_row(rng, S))
        row = dataclasses.replace(row, dh=len(win.decisions) if k % 2 else row.dh, steps=0 if k % 3 else row.steps)
        cases.append(RC.close_arguments(S, row, win))
        closes.append(int(RC.close(S, row, win)))
    assert RC.evaluate_close(S, cases) == closes and 0 < sum(closes) < len(closes)


@pytest.mark.parametrize("S", SHAPES)
def test_size(S):
    got = RC.size(S)
    assert got["ands_per_step"] == ands(RC.advance_unit(S)) and got["ands_load"] == ands(RC.load_unit(S))
    assert got["ands_close"] == ands(RC.close_unit(S))
    v1 = RC.size(RC.Shape(**{k: v for k, v in S.statics().items() if k != "TAU"}))
    assert got["width"] > v1["width"] and got["ands_per_step"] > v1["ands_per_step"]


# --- the chain of the tick's steps ------------------------------------------------------------------------------------

def _state(Sc: RC.Shape, row: RC.Row) -> TS.State:
    return TS.State(RS.State(tuple(RS.Req(**dict(zip(RC.running_names(Sc), x))) for x in row.running),
                             tuple(RS.Req(**dict(zip(RC.waiting_names(Sc), x))) for x in row.waiting), row.free), row.clock)


def _row(Sc: RC.Shape, st: TS.State) -> RC.Row:
    return RC.Row(tuple(tuple(getattr(r, k) for k in RC.running_names(Sc)) for r in st.step.running),
                  tuple(tuple(getattr(r, k) for k in RC.waiting_names(Sc)) for r in st.step.waiting), st.step.free,
                  clock=st.clock)


def _arrival(Sc: RC.Shape, i: int, io: SC.RequestIO, bucket: int) -> RC.Arrival:
    return (i, io.prompt_len, io.output_len) + ((int(io.output_len >= io.max_tokens),) if Sc.AS else ()) + (bucket,)


@dataclasses.dataclass
class Window:
    rows: list[RC.Row]
    win: RC.QueueRow
    last: RC.QueueRow
    arrivals: tuple[RC.Arrival, ...]
    n: int
    steps: list[tuple[RC.Row, RC.Row, int, TS.Out, TS.Inputs | None]]    # row, next state, taken, out, inputs (None idle)


def run(Sc: RC.Shape, St: SimpleNamespace, rule: TS.Rule, windows: list[tuple[tuple[RC.Arrival, ...], int]],
        extra=None, start: RC.Row | None = None) -> list[Window]:
    """The prover's chain: each window loaded from the last one's last rows with its arrivals and step count, then each
    of its steps `tick_step.reference` on the state row and the arrivals the front shows at its head (`extra(k, cur,
    arrivals)`: the k-th step's hit, evict and ret), idle past the window's count. `start`: the second window's state row
    0 in place of the first one's last."""
    cur, last, out, k = RC.empty(Sc), RC.QueueRow(), [], 0
    for w, (arrivals, n) in enumerate(windows):
        if w == 1 and start is not None:
            cur = start
        loaded = RC.load(Sc, cur, last, arrivals, (), n)
        assert loaded is not None
        rows = [cur]
        cur, win = loaded
        rows.append(cur)
        steps = []
        for _u in range(Sc.P):
            arr, bits = RC.front(Sc, win, cur)
            if not cur.steps:
                steps.append((cur, RC.Row(), 0, TS.Out(False), None))
                rows.append(cur)
                continue
            reqs = tuple(RS.Req(*a[:3], cap=a[3] if Sc.AS else 0) for a in arr)
            inp = TS.Inputs(reqs, tuple(a[-1] for a in arr), *(extra(k, cur, reqs) if extra else ()))
            o = TS.reference(St, rule, _state(Sc, cur), inp)
            assert o.ok and o.state is not None
            stepped = _row(Sc, o.state)
            nxt = RC.advance(Sc, win, cur, True, stepped, arr, bits, o.taken, 0)
            assert nxt is not None
            steps.append((cur, stepped, o.taken, o, inp))
            rows.append(nxt)
            cur, k = nxt, k + 1
        out.append(Window(rows, win, last, tuple(arrivals), n, steps))
        last = win
    return out


def held(Sc: RC.Shape, st_: dict, windows: list[Window]) -> list[RC.Committed]:
    """Each window's load and steps held by their circuits (`TickLoad_v1`, `TickAdvance_v1` and each active step's
    `TickStep_v1` on the state row it reads), committed as the chain's produced values, and the session's openings
    through `check_chain`; the last rows through `TickClose_v1`."""
    loads, advances, ticks, want = [], [], [], []
    for w in windows:
        loads.append(RC.load_arguments(Sc, w.rows[0], w.last, w.arrivals, (), 0, w.n, RC.encode(w.rows[1], Sc),
                                       RC.encode_window(w.win, Sc)))
        for (cur, stepped, taken, o, inp), nxt in zip(w.steps, w.rows[2:]):
            arr, bits = RC.front(Sc, w.win, cur)
            advances.append(RC.advance_arguments(Sc, w.win, cur, o.ok, stepped, arr, bits, taken, 0, RC.encode(nxt, Sc)))
            if inp is not None:
                st = _state(Sc, cur)
                ticks.append((st.step, st.clock, RS.Inputs(inp.arrivals, (), inp.hit, inp.evict, inp.ret), inp.released))
                want.append((stepped, taken, o.rows))
    assert RC.evaluate_load(Sc, loads) == [1] * len(loads)
    assert [ok for ok, _ in RC.evaluate_advance(Sc, advances)] == [1] * len(advances)
    for (got, clock), (stepped, taken, rows) in zip(TK.evaluate(st_, ticks), want, strict=True):
        assert got.ok and got.state is not None and got.taken == taken and got.rows == rows
        assert _row(Sc, TS.State(got.state, clock)) == stepped
    end = windows[-1]
    assert RC.evaluate_close(Sc, [RC.close_arguments(Sc, end.rows[-1], end.win)]) == [1]
    values, carry = [], None
    for w in windows:
        carry = RC.commit_window(CT.PROGRAM, w.rows, w.win, Sc, CT.REGISTRATION, carry=carry)
        values.append(carry)
    assert RC.check_chain([RC.openings(v) for v in values], CT.PROGRAM, Sc) is None
    return values


def _shape(e: SC.Engine, n: int, R: int, W: int, A: int, rule: TS.Rule, P: int, K: int, Q: int) -> RC.Shape:
    return RC.Shape(R=R, W=W, A=A, L=e.max_model_len, NB=e.num_gpu_blocks, IW=max(n - 1, 1).bit_length(), Q=Q, D=1, P=P,
                    K=K, AS=int(e.async_scheduling), TAU=rule.bucket)


def _clocks(windows: list[Window]) -> list[int]:
    return [o.state.clock for w in windows for _c, _s, _t, o, _i in w.steps if o.rows and o.state is not None]


def _rows(windows: list[Window]) -> list[list[tuple[int, int, int]]]:
    return [[(i, start, n) for _, i, _, start, n in o.rows] for w in windows for _c, _s, _t, o, _i in w.steps if o.rows]


@pytest.mark.parametrize("name", ["chosen"])
@TT.BOTH
def test_a_two_window_chain_carries_the_replays_clock(name: str, asy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The replay of the plan under the rule, its executed steps cut into two windows at a step that starts mid-bucket:
    the first window loads every arrival its steps see (the next A at each step's cursor), the second the rest. Each step
    is `tick_step.reference` on the state row and the front at its head, with the replay's prefix counts, and gives the
    replay's arrivals, rows and T: the chain's T after each step with rows is `Replay.clock`. The proof blocks and the
    chain hold (`held`)."""
    rule = TC.RULES[name]
    e = TT.engine(asy)
    ios, released = TT.plan()
    rp, units = TT.record(monkeypatch, e, ios, released, rule)
    st_ = TC.binding(e, TT.N, 4, 4, 5, rule)
    St = SimpleNamespace(**st_)
    full = [u for u in units if not u.idle]
    scan, st, cursor = [], TS.empty(St), 0
    for u in full:
        out = TS.reference(St, rule, st, TT.inputs(St, st.step, cursor, ios, released, u))
        assert out.ok and out.state is not None
        scan.append((st, cursor))
        st, cursor = out.state, cursor + out.taken
    cut = next(j for j in range(len(scan) // 2, len(scan)) if scan[j][0].clock % rule.bucket)
    seen = min(TT.N, max(c + St.A for _s, c in scan[:cut]))
    Sc = _shape(e, TT.N, 4, 4, 5, rule, P=max(cut, len(scan) - cut), K=TT.N, Q=TT.N)
    arrivals = [_arrival(Sc, i, io, released[i]) for i, io in enumerate(ios)]
    cursors = [c for _s, c in scan]

    def extra(k: int, cur: RC.Row, reqs: tuple) -> tuple:
        inp = TT.inputs(St, _state(Sc, cur).step, cursors[k], ios, released, full[k])
        assert inp.arrivals == reqs, "the front shows the replay's arrivals"
        return inp.hit, inp.evict, inp.ret

    windows = run(Sc, St, rule, [(tuple(arrivals[:seen]), cut), (tuple(arrivals[seen:]), len(scan) - cut)], extra)
    assert windows[0].rows[-1].clock == scan[cut][0].clock and windows[0].rows[-1].clock % rule.bucket
    assert _clocks(windows) == rp.clock and _rows(windows) == [s.rows for s in rp.steps]
    held(Sc, st_, windows)


def _small(asy: bool, rule: TS.Rule, requests: list[tuple[int, int, bool]], released: list[int]):
    e = SC.Engine(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=64, num_gpu_blocks=16, block_size=16,
                  async_scheduling=asy)
    ios = [SC.RequestIO(prompt_len=p, max_tokens=o + 4 if eos else o, output_len=o, eos=eos) for p, o, eos in requests]
    st_ = TC.binding(e, len(ios), 2, 2, 3, rule)
    St = SimpleNamespace(**st_)
    steps = TC.scan(St, rule, ios, released)
    rp = RP.replay(e, ios, tick=rule, released=released)
    return e, ios, st_, St, steps, rp


def _split(Sc, St, rule, ios, released, cut: int, steps, first: int, start: RC.Row | None = None) -> list[Window]:
    arrivals = [_arrival(Sc, i, io, released[i]) for i, io in enumerate(ios)]
    return run(Sc, St, rule, [(tuple(arrivals[:first]), cut), (tuple(arrivals[first:]), len(steps) - cut)], start=start)


@TT.BOTH
def test_a_window_that_ends_mid_bucket_and_the_next_that_starts_there(asy: bool) -> None:
    """Request 0's last step takes T past bucket 1's close to 220,000 and leaves the engine drained; the first window
    ends there, 20,000 into bucket 2, with request 0's arrival alone. The second starts at that row and loads requests 1
    and 2: request 1, released in bucket 1, is given at once, and the two windows give the replay's rows and T."""
    rule = TC.RULES["today"]
    released = [0, 1, 2]
    e, ios, st_, St, steps, rp = _small(asy, rule, [(4, 2 if asy else 3, True), (4, 6, True), (4, 2, True)], released)
    cut = next(k for k, (st, _i, _o) in enumerate(steps) if st.step == RS.empty(St) and st.clock)
    Sc = _shape(e, 3, 2, 2, 3, rule, P=max(cut, len(steps) - cut), K=3, Q=3)
    windows = _split(Sc, St, rule, ios, released, cut, steps, 1)
    seam = windows[0].rows[-1]
    assert (seam.clock, seam.clock // rule.bucket, seam.clock % rule.bucket) == (220_000, 2, 20_000)
    assert windows[1].rows[0] == seam and windows[1].rows[1].clock == seam.clock
    assert _clocks(windows) == rp.clock and _rows(windows) == [s.rows for s in rp.steps]
    first = windows[1].steps[0][3]
    assert first.taken == 1 and first.state is not None and first.state.clock == seam.clock + rule.free
    held(Sc, st_, windows)


@TT.BOTH
def test_several_buckets_pass_on_a_window_boundary(asy: bool) -> None:
    """Two seams across which T passes several buckets. A jump: the first window drains request 0 in bucket 1; request
    1 is released in bucket 7, so the second window's first step jumps T to bucket 7's close, six buckets on. A step that
    carries several: under the several-bucket rule the first window ends on a step that carries two buckets, and the
    second goes on from its row with both requests running. Both give the replay's rows and T."""
    rule = TC.RULES["today"]
    released = [0, 7]
    e, ios, st_, St, steps, rp = _small(asy, rule, [(4, 2, True), (4, 2, True)], released)
    cut = next(k for k, (st, _i, _o) in enumerate(steps) if st.step == RS.empty(St) and st.clock)
    Sc = _shape(e, 2, 2, 2, 3, rule, P=max(cut, len(steps) - cut), K=2, Q=2)
    windows = _split(Sc, St, rule, ios, released, cut, steps, 1)
    seam, jumped = windows[0].rows[-1], windows[1].steps[0][3]
    assert jumped.state is not None and jumped.state.clock - rule.free == rule.close(7) == 8 * rule.bucket
    assert 8 - seam.clock // rule.bucket >= 6
    assert _clocks(windows) == rp.clock and _rows(windows) == [s.rows for s in rp.steps]
    held(Sc, st_, windows)

    rule = TC.RULES["several"]
    released = [0, 0]
    e, ios, st_, St, steps, rp = _small(asy, rule, [(10, 6, True), (8, 5, True)], released)
    cut = 1 + next(k for k, (st, _i, o) in enumerate(steps)
                   if o.rows and o.state is not None and o.state.clock // rule.bucket - st.clock // rule.bucket >= 2
                   and k + 1 < len(steps))
    Sc = _shape(e, 2, 2, 2, 3, rule, P=max(cut, len(steps) - cut), K=2, Q=2)
    windows = _split(Sc, St, rule, ios, released, cut, steps, 2)
    before, seam = windows[0].rows[cut], windows[0].rows[-1]
    assert seam.clock // rule.bucket - before.clock // rule.bucket >= 2 and seam.running
    assert windows[1].rows[1] == dataclasses.replace(seam, qh=0, steps=len(steps) - cut)
    assert _clocks(windows) == rp.clock and _rows(windows) == [s.rows for s in rp.steps]
    held(Sc, st_, windows)


def test_the_chain_refuses_a_window_that_starts_at_another_clock() -> None:
    """The first window of the mid-bucket seam ends at T = 220,000. A second window that starts at that row with T
    restarted at 0 runs on (request 2 a step late), but: `commit_window` refuses it as the carry, its state row 0
    committed under the carried row's salt opens to another b‖c, so `check_chain` refuses the session (C-carry), and
    `TickLoad_v1` refuses a state row 1 whose clock isn't row 0's, as `TickAdvance_v1` refuses a next row whose clock
    isn't the step's."""
    rule = TC.RULES["today"]
    released = [0, 1, 2]
    e, ios, st_, St, steps, _rp = _small(False, rule, [(4, 3, True), (4, 6, True), (4, 2, True)], released)
    cut = next(k for k, (st, _i, _o) in enumerate(steps) if st.step == RS.empty(St) and st.clock)
    Sc = _shape(e, 3, 2, 2, 3, rule, P=max(cut, len(steps) - cut) + 1, K=3, Q=3)
    honest = _split(Sc, St, rule, ios, released, cut, steps, 1)
    seam = honest[0].rows[-1]
    restarted = _split(Sc, St, rule, ios, released, cut, steps, 1, start=dataclasses.replace(seam, clock=0))
    assert restarted[1].rows[0].clock == 0 and _clocks(restarted) != _clocks(honest)
    v0 = RC.commit_window(CT.PROGRAM, honest[0].rows, honest[0].win, Sc, CT.REGISTRATION)
    RC.commit_window(CT.PROGRAM, honest[1].rows, honest[1].win, Sc, CT.REGISTRATION, carry=v0)
    with pytest.raises(ValueError, match="ended in"):
        RC.commit_window(CT.PROGRAM, restarted[1].rows, restarted[1].win, Sc, CT.REGISTRATION, carry=v0)
    n = len(v0.state.salts[-1])
    salts = [v0.state.salts[-1]] + [bytes([k + 1]) * n for k in range(len(restarted[1].rows) - 1)]
    state = produced.commit(CT.PROGRAM, RC.VALUE, [RC.row(RC.encode(r, Sc)) for r in restarted[1].rows],
                            CT.REGISTRATION, salts, RC.WORD_BITS, RC.SCHEMA)
    window = produced.commit(CT.PROGRAM, RC.WINDOW, [v0.window.rows[-1], RC.row(RC.encode_window(restarted[1].win, Sc))],
                             CT.REGISTRATION, [v0.window.salts[-1], bytes(n)], RC.WORD_BITS, RC.SCHEMA)
    forged = RC.Committed(state, window)
    assert bytes(forged.state.bcs[0]) != bytes(v0.state.bcs[-1]) and bytes(forged.window.bcs[0]) == bytes(v0.window.bcs[-1])
    assert RC.check_chain([RC.openings(v0), RC.openings(forged)], CT.PROGRAM, Sc).startswith("C-carry")
    w = honest[1]
    good = RC.load_arguments(Sc, seam, w.last, w.arrivals, (), 0, w.n, RC.encode(w.rows[1], Sc), RC.encode_window(w.win, Sc))
    off = RC.load_arguments(Sc, seam, w.last, w.arrivals, (), 0, w.n,
                            RC.encode(dataclasses.replace(w.rows[1], clock=0), Sc), RC.encode_window(w.win, Sc))
    assert RC.load(Sc, seam, w.last, w.arrivals, (), w.n) == (w.rows[1], w.win)
    assert RC.evaluate_load(Sc, [good, off]) == [1, 0]
    cur, stepped, taken, _o, _i = w.steps[0]
    arr, bits = RC.front(Sc, w.win, cur)
    late = dataclasses.replace(w.rows[2], clock=w.rows[2].clock + 1)
    assert RC.evaluate_advance(Sc, [RC.advance_arguments(Sc, w.win, cur, True, stepped, arr, bits, taken, 0, RC.encode(n, Sc))
                                    for n in (w.rows[2], late)]) == [(1, 1), (0, 1)]
