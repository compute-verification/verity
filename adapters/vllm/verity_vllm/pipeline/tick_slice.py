"""The compile slice under the certified tick: `compiler.compile_slice_v7`'s with each step `compiler.tick_step.TickStep_v1`
in place of `compiler.replay_step.ReplayStep_v1`, on the chain at the tick's rows (`compiler.replay_chain` with TAU).

    TickSlice_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CN, CTW, CU,
                 TAU, CSAT, CFREE, C1, TCU, CARRY}
        (pub, adv, cert, srv, cin, cout, wit) -> verdict[16]

**The rows** are CompileSlice_v7's (`compile_slice.fields`) at the tick's chain: ``cert``'s arrival slots each end with
its release bucket (``a_bucket``, 32 bits), ``wit``'s state rows with T = q·TAU + r and its arrivals with their buckets.
TickStep_v1 reads no decision, so ``adv`` holds none: ``load`` refuses a count other than 0, every step's ``bits`` are
the window row's, all 0, and the charge is the length terms alone.

**The circuit.** `replay_chain.TickLoad_v1` from the empty rows (T = 0) with ``cert``'s arrivals and buckets. Per step,
TickStep_v1 at (TAU, CSAT, CFREE, C1, CARRY, with TCU its CU) on S_u's state and T with ``arr[u]`` and their buckets, the
evictable and returned counts 0 (CompileSlice_v1's stand-in), and `replay_chain.TickAdvance_v1` holding them to W_1
at S_u's heads and S_{u+1} to the step's next state and T. The rows, the charge and the verdict are CompileSlice_v7's;
the slice has no prefix cache (PC = 0, so the cache's CN, CTW and CU are 0) and ``cache`` is 1.

**What holds the buckets and T.** The circuit reads a bucket only from ``cert``: the load writes each into W_1, and a
step reads W_1's at its head. ``cert`` is held to the certified record by its b‖c, CompileSlice_v7's stand-in, so a
bucket changed in ``cert`` is refused by ``rows`` where it changes the schedule and by that b‖c alone where it doesn't.
T is the chain's: the load starts it at 0 (the slice is CompileSlice_v1's *one window*), and each step's is TickStep_v1's.

**The reference** is `check` and `case`, CompileSlice_v7's with `pipeline.advice_code.tick_step.reference` for the step.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

from verity.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import SpecializedDefinition, StaticValue, bind
from verity.circuits.evaluation.bits import evaluate_bits

from verity_vllm.compiler import compile_slice as CS
from verity_vllm.compiler import compile_slice_v7 as V7
from verity_vllm.compiler import replay_chain as RC
from verity_vllm.compiler import replay_step as RS
from verity_vllm.compiler import tick_step as TK
from verity_vllm.pipeline.advice_code import tick_step as TS

STATICS = CS.STATICS + ("TAU", "CSAT", "CFREE", "C1", "TCU", "CARRY")
FLAGS, ACCEPT, CODE = CS.FLAGS, CS.ACCEPT, V7.CODE


def step_statics(S) -> SimpleNamespace:
    """TickStep_v1's statics: the step's, with TCU its CU."""
    return SimpleNamespace(**{k: getattr(S, k) for k in RS.STATICS}, TAU=S.TAU, CSAT=S.CSAT, CFREE=S.CFREE, C1=S.C1,
                           CU=S.TCU, CARRY=S.CARRY)


def rule(S) -> TS.Rule:
    return TS.Rule(S.TAU, S.CSAT, S.CFREE, S.C1)


def _sig(S):
    if S.PC:
        raise ValueError("the tick's slice has no prefix cache: PC = 0")
    if S.TCU:
        raise ValueError("the tick's slice has no catch-up while requests run: TCU = 0")
    TK.widths(step_statics(S))
    return CS._sig(S, CODE)


def _build(C, S, *ins):
    Sc, St = CS.chain(S), step_statics(S)
    F = CS.fields(S, CODE)
    split = {r: RC._split(x, F[r]) for x, r in zip(ins, CS.ROWS)}
    p, a, w = split["pub"], split["adv"], split["wit"]
    served = split["srv"]["rows"]
    states, win, count = w["state"], w["window"][0], a["count"][0]
    load_ok = RC._build_load(C, Sc, CS._consts(RC.encode(RC.empty(Sc), Sc)), CS._consts(RC.encode_window(RC.QueueRow(), Sc)),
                             RC._flat(split["cert"], RC.arrival_fields(Sc)), [e[0] for e in a["dec"]], count,
                             a["steps"][0], states[0], win)[0]
    steps_ok, rows, mism = [], [], []
    for u in range(S.P):
        cur, nxt, arr, bits = states[u], states[u + 1], w["arr"][u], w["bits"][u]
        r, fa = RC._split(cur, RC.row_fields(Sc)), RC._split(arr, RC.front_fields(Sc))
        elems = []
        for name, n, wd in TK._in_fields(St):
            elems += r[name] if name in r else fa[name] if name in fa else [[ZERO] * wd for _ in range(n)]
        out = RS._split(TK._out_fields(St), TK._build(C, St, *elems))
        stepped = [b for name, _n, _w in RC.state_fields(Sc) for e in out[name] for b in e]
        ok, active = RC._build_advance(C, Sc, out["ok"][0], cur, win, stepped, arr, bits, out["taken"][0],
                                       [ZERO] * RC.widths(Sc).AC, nxt)
        steps_ok.append(ok[0])
        mine = [C.AND(active[0], x) for name in CS.ROW_FIELDS for e in out[name] for x in e]
        if S.AS:
            mism += CS._build_map(C, S, CS._by_slot(S, mine), CS._by_slot(S, served[u]), w["map"][u * S.R:(u + 1) * S.R])
        else:
            rows += mine
    en = [RC._lt(C, const_bits(k, len(count)), count) for k in range(S.D)]
    code_ok, charged = CODE.link(C, S, split, en)
    rows_ok = RC._zero(C, mism) if S.AS else RC._eq(C, rows, [x for e in served for x in e])
    flags = [C.AND(load_ok, RC._zero(C, count)), and_reduce(C, steps_ok), code_ok, RC._lt(C, p["c"][0], charged).inv(),
             rows_ok, ONE]
    return [flags + [ZERO] * (16 - len(flags))]


TickSlice = _part(
    "TickSlice", STATICS, _sig, _build,
    "the compile slice's proof block under the certified tick, one window whole: TickLoad_v1 from the empty rows (T = 0) "
    "with the certified arrivals and their release buckets and no decision, then per step TickStep_v1 and TickAdvance_v1 "
    "on the same arrival wires, each step's T carried in its state row, the length terms charged by ChargeLink_v2{D, 0, 0, "
    "K} from the window's carry-in to its carry-out and held to the declared charge, and the step rows held to the "
    "committed served rows (verdict bits load, steps, code, charge, rows, cache)")


def statics(R: int, W: int, A: int, engine: dict, IW: int, Q: int, D: int, P: int, K: int, *, tau: int, sat: int,
            free: int, token: int = 0, carry: int = 1) -> dict:
    """`compile_slice_v7.statics` with the tick's amounts (`compiler.tick_step.statics`' ``tau``, ``sat``, ``free``, ``token``
    and ``carry``), refused as TickStep_v1 refuses them and where the engine caches prefixes."""
    st = {**V7.statics(R, W, A, engine, IW, Q, D, P, K), "TAU": tau, "CSAT": sat, "CFREE": free, "C1": token,
          "TCU": 0, "CARRY": carry}
    _sig(SimpleNamespace(**st))
    return st


def unit(**st: StaticValue) -> SpecializedDefinition:
    return bind(TickSlice, **st)


def demo(asy: bool = False) -> tuple[dict, Slice]:
    """The toy window's statics and slice: `compile_slice.DEMO_ENGINE` at two sequences without a prefix cache, sync or
    async, (R 2, W 1, A 3), six step rows (five steps sync, six async); three arrivals (id, P, O[, cap]) with max_tokens 4, 64 and 2, released in buckets
    0, 1 and 3 of 10 time units. T advances 6 after a step that saturates, else 4, plus 1 a token."""
    engine = {**CS.DEMO_ENGINE, "max_num_seqs": 2, "enable_prefix_caching": False, "async_scheduling": asy}
    st = statics(2, 1, 3, engine, 3, Q=3, D=1, P=6, K=3, tau=10, sat=6, free=4, token=1, carry=3)
    cap = ((0,), (0,), (1,)) if asy else ((), (), ())
    return st, Slice(tuple((i, P, O) + c for (i, P, O), c in zip(((0, 9, 3), (1, 6, 2), (2, 5, 2)), cap)), (),
                     6 if asy else 5, (), (4, 64, 2), (0, 1, 3))


@dataclass(frozen=True)
class Slice(V7.Slice):
    """`compile_slice_v7.Slice` with each arrival's release bucket, in the arrivals' order, and no decision."""

    buckets: tuple[int, ...] = ()


def arrivals(sl: Slice) -> tuple[RC.Arrival, ...]:
    """The slice's arrivals as the chain's, each ending with its release bucket."""
    if len(sl.buckets) != len(sl.arrivals):
        raise ValueError(f"{len(sl.buckets)} release buckets for {len(sl.arrivals)} arrivals")
    return tuple(tuple(a) + (b,) for a, b in zip(sl.arrivals, sl.buckets))


def _step(S, cur: RC.Row, arr: Sequence[RC.Arrival]) -> TS.Out:
    Sc = CS.chain(S)
    reqs = tuple(RS.Req(**dict(zip(RC.arrival_names(Sc)[:-1], a[:-1]))) for a in arr)
    return TS.reference(step_statics(S), rule(S), TS.State(CS._state(Sc, cur), cur.clock),
                        TS.Inputs(reqs, tuple(a[-1] for a in arr)))


def scan(S, sl: Slice, cert: Sequence[RC.Arrival] | None = None, *, clock: dict[int, int] | None = None) -> SimpleNamespace:
    """The window as a prover runs it from `sl`: TickLoad_v1's rows, then each step by `advice_code.tick_step.reference` on the
    state row and the window row's arrivals at its head. The prover's choices: `cert`, the arrivals with buckets it puts
    in ``cert`` (default `sl`'s); `clock`, step u -> the T it puts in the state row step u reads, the steps then going
    on from it."""
    Sc = CS.chain(S)
    got = tuple(map(tuple, cert if cert is not None else arrivals(sl)))
    loaded = RC.load(Sc, RC.empty(Sc), RC.QueueRow(), got, (), sl.steps)
    if loaded is None:
        raise ValueError("the load refuses the slice")
    cur, win = loaded
    states, fronts, steps = [], [], []
    for u in range(S.P):
        if u in (clock or {}):
            cur = replace(cur, clock=(clock or {})[u])
        states.append(cur)
        fronts.append(RC.front(Sc, win, cur))
        if not cur.steps:
            continue
        out = _step(S, cur, fronts[-1][0])
        if not out.ok or out.state is None:
            raise ValueError("the step refuses: the slice's class can't take it")
        cur = replace(CS._row(Sc, out.state.step), qh=cur.qh + out.taken, dh=cur.dh, steps=cur.steps - 1,
                      clock=out.state.clock)
        steps.append(out.rows)
    states.append(cur)
    return SimpleNamespace(states=states, window=win, fronts=fronts, steps=steps, arrivals=got)


def check(S, rows: dict) -> dict[str, bool | None]:
    """The circuit's flags on a proof block's rows by its pieces' references, CompileSlice_v7's (`compile_slice.check`) with
    `advice_code.tick_step.reference` for each active step and ``load`` refused at a count other than 0."""
    Sc = CS.chain(S)
    F = CS.fields(S, CODE)
    a, w = (RC._split(rows[r], F[r]) for r in ("adv", "wit"))
    states = [RC.decode(e, Sc) for e in w["state"]]
    win = RC.decode_window(w["window"][0], Sc)
    cert = RC._items(RC._unflatten(rows["cert"], RC.arrival_fields(Sc))[0], "a", RC.arrival_names(Sc))
    count, dec = CS._int(a["count"][0]), [e[0] for e in a["dec"]]
    if count > S.D or any(dec[count:]):
        raise RC.MalformedRow(f"the advice's count {count} with a decision past it")
    loaded = RC.load(Sc, RC.empty(Sc), RC.QueueRow(), cert, dec[:count], CS._int(a["steps"][0]))
    flags: dict[str, bool | None] = {"load": not count and loaded == (states[0], win), "cache": True}
    ok, length, carried = CODE.reference(S, rows, count, dec)
    flags.update(code=bool(ok) and tuple(int(b) for b in rows["cout"]) == tuple(carried),
                 charge=length <= CS._int(RC._split(rows["pub"], F["pub"])["c"][0]) if ok else None)
    steps_ok, out_rows = True, []
    for u in range(S.P):
        cur, nxt = states[u], states[u + 1]
        arr = RC._items(RC._unflatten(w["arr"][u], RC.front_fields(Sc))[0], "arr", RC.arrival_names(Sc))
        if not cur.steps:
            steps_ok &= nxt == cur
            out_rows.append(None)
            continue
        out = _step(S, cur, arr)
        if not out.ok or out.state is None:
            flags.update(steps=False, rows=None)
            return {f: flags[f] for f in FLAGS}
        stepped = replace(CS._row(Sc, out.state.step), clock=out.state.clock)
        want = RC.advance(Sc, win, cur, True, stepped, arr, [int(b) for b in w["bits"][u]], out.taken, 0)
        steps_ok &= want is not None and want == nxt
        out_rows.append(out.rows)
    flags.update(steps=steps_ok, rows=CS._rows_ok(S, out_rows, CS._served(S, rows), w.get("map")))
    return {f: flags[f] for f in FLAGS}


def case(S, sl: Slice, book: dict, *, cert: Sequence[RC.Arrival] | None = None, clock: dict[int, int] | None = None,
         served: Sequence | None = None, charge: int | None = None) -> CS.Case:
    """A proof block's rows for the prover's run of `sl` (`scan`'s, with `cert` and `clock`) under the registered `book`, from
    its priors, and `check`'s verdict on them: `served` the window's step rows it puts in ``srv`` (default the run's
    own), `charge` ``pub``'s (default the length terms' charge)."""
    Sc = CS.chain(S)
    vals, first = CODE.book(S, book)
    sc = scan(S, sl, cert, clock=clock)
    ok, length, carry = CODE.charge(S, book, first, sl, (), sc.arrivals)
    if not ok:
        raise ValueError("the charge refuses the slice from the book's priors")
    served = sc.steps if served is None else served
    width = CS.row_bits(S, CODE)
    rows = {"pub": CS._pack(S, "pub", {"c": [length if charge is None else charge], **vals}, CODE),
            "adv": CS._pack(S, "adv", {"dec": [0] * S.D, "count": [0], "steps": [sl.steps]}, CODE),
            **CS.srv_rows(S, served), "cin": list(CODE.encode(first)), "cout": list(CODE.encode(carry))}
    cert_bits = RC.load_arguments(Sc, RC.empty(Sc), RC.QueueRow(), sc.arrivals, (), 0, sl.steps, RC.encode(sc.states[0], Sc),
                                  RC.encode_window(sc.window, Sc))[2] + CODE.cert(S, sl, sc.arrivals)
    rows["cert"] = cert_bits + [0] * (width["cert"] - len(cert_bits))
    wit = [b for st in sc.states for b in RC.encode(st, Sc)] + list(RC.encode_window(sc.window, Sc))
    wit += [b for arr, _b in sc.fronts for b in RC.front_bits(Sc, arr)] + [b for _a, bits in sc.fronts for b in bits]
    if S.AS:
        mine, theirs = (list(x) + [None] * (S.P - len(x)) for x in (sc.steps, served))
        wit += [b for u in range(S.P) for r in CS.slot_map(S, mine[u], theirs[u]) for b in r]
    rows["wit"] = wit + [0] * (width["wit"] - len(wit))
    flags = check(S, rows)
    return CS.Case(rows, CS.verdict(flags), flags, tuple(sc.steps), (), length, carry)


def evaluate(st: dict, cases: Sequence[CS.Case]) -> list[int]:
    """The proof block's verdict word on each case, in one evaluation."""
    args = [[c.rows[r] for c in cases] for r in CS.ROWS]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


__all__ = ["ACCEPT", "CODE", "FLAGS", "STATICS", "Slice", "TickSlice", "arrivals", "case", "check", "demo", "evaluate",
           "rule", "scan", "statics", "step_statics", "unit"]
