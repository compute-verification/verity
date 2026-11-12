"""Advice v1 to and from the replay's schedule: the entry for a serve I1 v0 can't say.

`codec.encode` and `codec.decode` speak I1 v0, whose every sampling row emits one output. Under a draft (the registered
engine's `num_speculative_tokens` > 0) a row emits its token and the drafts it kept, so no I1 v0 says what was served
and `codec` refuses one by name. This module is the same code's other entry. It carries a `Schedule`, what the replay
gives, in the record's ordinals:
- `arrival`: the requests in the order the engine took them;
- `steps`: each executed step's rows (request, first position, tokens), in the scheduler's order;
- `kept`: the drafts each request's emitting rows kept (`kept`), and without a draft one 0 an output.

`encode(schedule, record, registered) -> bytes` and `decode(bytes, record, registered) -> Schedule`. The string is
`codec`'s under the same registration: its windows, its length terms and, under a draft, its kept terms, which
`codec.bits_by_window` charges. Without a draft it is the string `codec.encode` gives the same serve, bit for bit.
`decode` is total as `codec.decode` is, and `decode(encode(s))` is s with each step's rows in the scheduler's order:
`encode` reads a step's rows as a set, since their order is derived and no bit of the string says it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

from verity_vllm.pipeline.advice_code import codec as V
from verity_vllm.pipeline.advice_code import kept as KP
from verity_vllm.pipeline.advice_code import replay as R
from verity_vllm.pipeline.advice_code import schedule as S
from verity_vllm.pipeline.advice_code import window_code as CV
from verity_vllm.pipeline.advice_code.codec import AdviceCodeError, Record, Registered

__all__ = ["decode", "encode", "Schedule"]


class Schedule(NamedTuple):
    """What a string carries under the record where no I1 v0 says it (the module's docstring), a request the ordinal
    of its `Record.requests` entry."""

    arrival: tuple[int, ...]
    steps: tuple[tuple[S.Row, ...], ...]
    kept: tuple[tuple[int, ...], ...]


def _schedule(record: Record, order: list[int], rp: R.Replay, kept: list[tuple[int, ...]] | None) -> Schedule:
    counts = kept if kept is not None else [(0,) * len(q.output) for q in record.requests]
    return Schedule(tuple(order), tuple(tuple((order[i], start, n) for i, start, n in st.rows) for st in rp.steps),
                    tuple(counts))


def encode(schedule: Schedule, record: Record, reg: Registered, *, check: bool = True) -> bytes:
    """The schedule's v1 string under the record; refuses a schedule v1 can't carry: one that isn't the replay's
    under its arrival order and kept counts. `check` decodes the string back and holds it to the schedule."""
    def run() -> bytes:
        if type(schedule) is not Schedule:
            raise AdviceCodeError(f"v1 encodes a Schedule here, not a {type(schedule).__name__}")
        n, order, engine = len(record.requests), list(schedule.arrival), reg.engine
        if sorted(order) != list(range(n)) or len(schedule.kept) != n:
            raise AdviceCodeError("the schedule's requests aren't the record's")
        V.check_record(record, reg, engine)
        if reg.tick is not None and order != V.release_order(record):
            raise AdviceCodeError("under the certified tick the engine admits in release order, and this schedule's "
                                  "arrival order isn't it")
        kept: list[tuple[int, ...]] | None = [tuple(c) for c in schedule.kept]
        if not engine.num_speculative_tokens:
            if kept != [(0,) * len(q.output) for q in record.requests]:
                raise AdviceCodeError("without a draft every row keeps none: a request's kept counts are one 0 an output")
            kept = None
        index = {k: e for e, k in enumerate(order)}
        theirs = [S.Step(rows=[(index.get(k, -1), start, m) for k, start, m in rows]) for rows in schedule.steps]
        first = [len(theirs)] * n
        for s in range(len(theirs) - 1, -1, -1):
            for e, _, _ in theirs[s].rows:
                if e < 0:
                    raise AdviceCodeError(f"step {s}: a row of a request the record doesn't have")
                first[e] = s
        if len(theirs) in first:
            raise AdviceCodeError(f"request {record.requests[order[first.index(len(theirs))]].id} has no row in the schedule")
        with KP.emitting() if kept is not None else nullcontext() as emitted:
            rp, decisions, looks = V.replay_record(record, reg, engine, order, lambda i, k: k >= first[i], kept)
        wins = V._residual(record, reg, order, rp, decisions, looks, V.windows(record, reg.T), emitted)
        terms = KP.tails(record.requests, V.kept_terms(record, reg), kept)
        data = CV.pack(V.string_of(wins, reg, V.length_tails(record, reg), terms)[0])[0]
        if check:
            m = R.compare(rp.steps, theirs)
            if m.first_mismatch is not None:
                raise AdviceCodeError(f"the schedule isn't the replay's from step {m.first_mismatch} "
                                      f"({m.matched}/{m.steps} steps match)")
            try:
                got = decode(data, record, reg)
            except AdviceCodeError as e:
                raise AdviceCodeError(f"the schedule is outside v1: {e}") from None
            if got != _schedule(record, order, rp, kept):
                raise AdviceCodeError("the schedule is outside v1: its string decodes to another schedule")
        return data
    return V._total(run)


def decode(data: bytes, record: Record, reg: Registered) -> Schedule:
    """The schedule the string `data` carries under the record, or `AdviceCodeError`."""
    def run() -> Schedule:
        wins, tails, kept = V._parse(data, reg, record)
        _, order, rp = V._replayed(wins, tails, kept, record, reg)
        return _schedule(record, order, rp, kept)
    return V._total(run)
