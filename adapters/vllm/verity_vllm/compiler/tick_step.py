"""The certified tick's admission step as a Boolean Definition, on `replay_step`'s builder."""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from verity.circuits.boolean import BIT
from verity.circuits.boolean.forms import ONE, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import SpecializedDefinition, StaticValue, bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array

from verity_vllm.compiler import replay_step as RS

STATICS = RS.STATICS + ("TAU", "CSAT", "CFREE", "C1", "CU")
#: a bucket count's width: T is a bucket count q and a remainder r < TAU
QW = 32


def widths(S: SimpleNamespace) -> SimpleNamespace:
    """`replay_step.widths` with RW = bitlen(TAU − 1). Refuses amounts whose largest advance reaches a bucket, and a class
    that can't see the first arrival a step leaves (A ≥ min(M, R) + 1)."""
    Z = RS.widths(S)
    if min(S.TAU, S.CSAT, S.CFREE) < 1 or S.C1 < 0 or S.CU not in (0, 1):
        raise ValueError(f"tick statics TAU={S.TAU}, CSAT={S.CSAT}, CFREE={S.CFREE}, C1={S.C1}, CU={S.CU}: TAU, CSAT and "
                         "CFREE positive, C1 at least 0, CU 0 or 1")
    if (most := max(S.CSAT, S.CFREE) + S.C1 * S.T) >= S.TAU:
        raise ValueError(f"the largest advance max(CSAT, CFREE) + C1·T = {most} reaches the bucket TAU = {S.TAU}: the step "
                         "carries at most one bucket a step")
    if S.A < min(S.M, S.R) + 1:
        raise ValueError(f"A = {S.A} arrivals in view with up to {min(S.M, S.R)} taken a step: the tick step needs "
                         f"A ≥ min(M, R) + 1 = {min(S.M, S.R) + 1}, to see the first arrival it doesn't take")
    Z.RW = max((S.TAU - 1).bit_length(), 1)
    return Z


def _clock_fields(S) -> list[tuple[str, int, int]]:
    return [("tick_q", 1, QW), ("tick_r", 1, widths(S).RW)]


def _in_fields(S) -> list[tuple[str, int, int]]:
    state = RS._state_fields(S)
    rest = [f for f in RS._in_fields(S)[len(state):] if f[0] != "bits"]
    k = next(i for i, f in enumerate(rest) if f[0] == "hit")
    return state + _clock_fields(S) + rest[:k] + [("arr_bucket", S.A, QW)] + rest[k:]


def _out_fields(S) -> list[tuple[str, int, int]]:
    return [f for f in RS._out_fields(S) if f[0] != "used"] + _clock_fields(S)


def _build(C, S, *elements):
    Z = widths(S)
    ins = RS._split(_in_fields(S), elements)
    q, r, bucket = ins["tick_q"][0], ins["tick_r"][0], ins["arr_bucket"]
    valid = [e[0] for e in ins["arr_valid"]]
    waits = ins["wait_valid"][0][0]
    idle = waits.inv() if S.CU else C.AND(waits.inv(), or_reduce(C, [e[0] for e in ins["run_valid"]]).inv())
    a1 = add(C, list(bucket[0]), const_bits(1, QW + 1), width=QW + 1)
    jump = and_reduce(C, [idle, valid[0], RS._lt(C, q, a1)])
    bad = [C.AND(jump, a1[QW])]
    q = mux(C, jump, a1[:QW], q)
    r = RS._sel(C, jump.inv(), r)
    given, g = [], ONE
    for a in range(S.A):
        g = and_reduce(C, [g, valid[a], RS._lt(C, bucket[a], q)])
        given.append(g)

    inner = []
    for name, _n, _w in RS._in_fields(S):
        inner += ([[x] for x in given] if name == "arr_valid" else [[ONE] for _ in range(S.A)] if name == "bits"
                  else ins[name])
    tap: dict = {}
    out = RS._split(RS._out_fields(S), RS._build(C, S, *inner, tap=tap))

    rows = or_reduce(C, [e[0] for e in out["row_valid"]])
    if S.CSAT == S.CFREE:
        c = const_bits(S.CSAT, Z.RW)
    else:
        sat = or_reduce(C, [tap["preempted"], out["wait_valid"][0][0],
                            RS._lt(C, out["taken"][0], RS._popcount(C, given, Z.AC + 1))])
        c = mux(C, sat, const_bits(S.CSAT, Z.RW), const_bits(S.CFREE, Z.RW))
    w = Z.RW + 1
    n = RS._sub(C, const_bits(S.T, Z.TW), tap["budget"], Z.TW)
    d = RS._pad(RS._sel(C, rows, c), w)
    for j in range(S.C1.bit_length()):
        if S.C1 >> j & 1:
            d = add(C, d, RS._pad([ZERO] * j + n, w), width=w)
    s = add(C, RS._pad(r, w), d, width=w)
    over, under = sub(C, s, const_bits(S.TAU, w), w)
    wrap = under.inv()
    r = mux(C, wrap, over[:Z.RW], s[:Z.RW])
    qq = add(C, list(q), [wrap], width=QW, carry_out=True)
    bad.append(qq[QW])

    res = {**out, "ok": [[C.AND(out["ok"][0][0], or_reduce(C, bad).inv())]], "tick_q": [qq[:QW]], "tick_r": [r]}
    return [e for name, _n, _w in _out_fields(S) for e in res[name]]


def _sig(S):
    return (tuple((name, Array(n, Array(w, BIT))) for name, n, w in _in_fields(S)),
            RS._type(_out_fields(S)))


TickStep = _part(
    "TickStep", STATICS, _sig, _build,
    "one step of the certified tick (`ReplayStep_v1`'s step with every given arrival asked): the clock T = q·TAU + r jumps "
    "to the next arrival's bucket close when nothing runs or waits (CU = 0) or nothing waits (CU = 1); the step is given "
    "the leading arrivals whose buckets have closed; after a step with rows T advances by CSAT if it preempted, left a "
    "request waiting or left a given arrival, else CFREE, plus C1 for each token it schedules; 0 for ok to refuse what "
    "`ReplayStep_v1` refuses or a bucket count past 32 bits (`pipeline.advice_code.tick_step.reference`)")


def statics(R: int, W: int, A: int, engine: dict, IW: int, *, tau: int, sat: int, free: int, token: int = 0,
            quiet: bool = True, EW: int = 0) -> dict:
    """`replay_step.statics` with the tick's amounts in one unit of time (the bucket ``tau``, the advances ``sat``,
    ``free`` and ``token``) and its catch-up (``quiet``: only when nothing runs or waits), refused as `widths` refuses."""
    st = {**RS.statics(R, W, A, engine, IW, EW), "TAU": tau, "CSAT": sat, "CFREE": free, "C1": token, "CU": int(not quiet)}
    widths(SimpleNamespace(**st))
    return st


def unit(**st: StaticValue) -> SpecializedDefinition:
    return bind(TickStep, **st)


def arguments(S, st: RS.State, clock: int, inp: RS.Inputs, buckets: Sequence[int]) -> list[list[int]]:
    """The proof block's arguments: `replay_step.arguments` without its bits, with T and each arrival's release bucket."""
    Z = widths(S)
    vals = dict(zip((f[0] for f in RS._in_fields(S)), RS.arguments(S, st, inp)))
    vals["tick_q"], vals["tick_r"] = RS._bits(clock // S.TAU, QW), RS._bits(clock % S.TAU, Z.RW)
    vals["arr_bucket"] = [b for v in list(buckets) + [0] * (S.A - len(buckets)) for b in RS._bits(v, QW)]
    return [vals[name] for name, _n, _w in _in_fields(S)]


def decode(S, flat: Sequence[int]) -> tuple[RS.Out, int]:
    """The proof block's outputs as `replay_step.decode` gives them (``used`` 0) and T."""
    at, part = 0, {}
    for name, n, w in _out_fields(S):
        part[name] = list(flat[at:at + n * w])
        at += n * w
    inner = [b for name, n, w in RS._out_fields(S) for b in part.get(name, [0] * (n * w))]
    q, r = (sum(int(b) << j for j, b in enumerate(part[k])) for k in ("tick_q", "tick_r"))
    return RS.decode(S, inner), q * S.TAU + r


def evaluate(st_: dict, cases: Sequence[tuple[RS.State, int, RS.Inputs, Sequence[int]]]) -> list[tuple[RS.Out, int]]:
    """The proof block at statics `st_` on each (state, T, inputs, buckets), in one evaluation."""
    S = SimpleNamespace(**st_)
    args = [list(col) for col in zip(*(arguments(S, *c) for c in cases))]
    out = evaluate_bits(unit(**st_), *args)
    return [decode(S, row) for row in out.reshape(len(cases), -1).tolist()]


__all__ = ["QW", "STATICS", "TickStep", "arguments", "decode", "evaluate", "statics", "unit", "widths"]
