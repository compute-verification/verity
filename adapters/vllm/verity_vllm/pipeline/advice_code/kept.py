"""Advice v1's kept counts: with a draft, the drafts each emitting row kept, sent and charged.

With a draft (the registered engine's `num_speculative_tokens`, K > 0) a decode row carries up to K drafts after its
token, and what it emits is its token and the drafts it kept (`replay`'s docstring). How many it kept is the model's
accept rule's, run on the row's logits, and no proof yet holds that rule. So the count is the prover's choice, and v1
sends and charges it:
- each emitting row's count, in the order the replay reads them: the steps in the order they execute, a step's rows in
  the scheduler's order;
- in its step's window (`codec.charged`: the earliest window that releases an output at that step or later), the
  first window whose rows the count can change;
- a count c as the bits "c > j" for j = 0, 1, … until one is 0 or j is the most the row can keep: K, or the outputs
  the request still lacks after the row's own token. Each bit is coded under the window coder's adaptive estimate
  (`window_code.KT`) in the context (the request's count before, j) (`context`);
- as its window's second coded segment (`window_code.Encoder.segment`). The second segments follow the last window's
  tail, in the windows' order.

Which rows emit, and whose they are, is the replay's to say, and the replay runs on the counts. So the decoder reads
them while the replay runs, each as the replay asks for it (`Reader`, `replay.replay`'s ``kept``). Beside the string
it needs K and each request's certified output count and release buckets, and nothing else: a count's window is the
earliest that releases an output no row had emitted when its step began, which the counts before it say. The
decisions and the order, which the replay needs from its start, are in the first segments, all before any count.

What is sent is held to the rows by the replay, which refuses a count over the drafts the row it schedules there
carries, and one past where the request stops (`replay.Scheduler`).

`emitting` reads off the replay what each executed step's rows emitted, which under a draft is not one output a row.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline.advice_code import replay as R

__all__ = ["coder", "context", "Counts", "emitting", "Reader", "Terms", "terms"]


def context(before: int | None, j: int) -> str:
    """The estimate's context for the bit "c > j" of a count whose request's count before was `before` (None for the
    request's first emitting row): `window_code`'s bit symbol ``kept`` in it is the book's prior ``kept/<context>``."""
    return f"{'first' if before is None else before}.{j}"


class Terms(NamedTuple):
    """A string's kept terms: each request's certified output count, and each window's counts in the order the replay
    reads them, each as (request, the drafts its row kept). A request is its place in `outputs`."""

    outputs: tuple[int, ...]
    windows: tuple[tuple[tuple[int, int], ...], ...]

    def by_request(self) -> list[tuple[int, ...]]:
        """Each request's counts, one per row of it that emitted, in order."""
        out: list[list[int]] = [[] for _ in self.outputs]
        for w in self.windows:
            for i, c in w:
                out[i].append(c)
        return [tuple(c) for c in out]


def terms(emitted: R.Emitted, outputs: Sequence[int], window: Sequence[int], W: int) -> Terms:
    """The kept terms of a replay that ran to its end: `emitted` is what its executed steps emitted (`emitting`), and
    `window[s]` the window step s is charged to. A row's count is the outputs it emitted less its own token."""
    last = list(outputs)
    out: list[list[tuple[int, int]]] = [[] for _ in range(W)]
    for s in range(len(emitted) - 1, -1, -1):
        for i in reversed(emitted[s]):
            out[window[s]].append((i, last[i] - emitted[s][i] - 1))
            last[i] = emitted[s][i]
    return Terms(tuple(outputs), tuple(tuple(reversed(w)) for w in out))


class Counts:
    """The coder's state over one string's counts: the outputs each request has emitted, and its count before."""

    def __init__(self, K: int, outputs: Sequence[int]) -> None:
        self.K, self.outputs = K, tuple(outputs)
        self.done = [0] * len(self.outputs)
        self.before: list[int | None] = [None] * len(self.outputs)

    def _most(self, i: int) -> int:
        """The most drafts request i's next emitting row can keep."""
        most = min(self.K, self.outputs[i] - self.done[i] - 1)
        if most < 0:
            raise ValueError(f"request {i}: a row emits after its {self.outputs[i]} outputs")
        return most

    def put(self, e: Any, i: int, c: int) -> None:
        """Codes the count c of request i's next emitting row through the window coder `e` (`window_code.Encoder`).
        Refuses (ValueError) a count that isn't a row's."""
        most = self._most(i)
        if type(c) is not int or not 0 <= c <= most:
            raise ValueError(f"request {i}: a row kept {c!r} drafts where it can keep 0 to {most}: {self.done[i]} of its "
                             f"{self.outputs[i]} outputs are out, and a row drafts {self.K}")
        for j in range(min(c + 1, most)):
            e.bit("kept", c > j, context(self.before[i], j))
        self.done[i], self.before[i] = self.done[i] + 1 + c, c

    def get(self, d: Any, i: int) -> int:
        """`put`'s mirror through `window_code.Decoder`: the count of request i's next emitting row."""
        c, most = 0, self._most(i)
        while c < most and d.bit("kept", context(self.before[i], c)):
            c += 1
        self.done[i], self.before[i] = self.done[i] + 1 + c, c
        return c

    def end(self) -> None:
        """Refuses (ValueError) counts that with their rows don't add up to each request's outputs."""
        for i, (done, outputs) in enumerate(zip(self.done, self.outputs)):
            if done != outputs:
                raise ValueError(f"request {i}: its rows and the drafts they kept add up to {done} outputs, not its {outputs}")


def coder(K: int, kept: Terms) -> Callable[[Any, int], None]:
    """`window_code.encode`'s ``coded`` for a string's kept terms: codes window w's counts, and after the last window's
    refuses (ValueError) terms whose counts don't add up to each request's outputs."""
    counts = Counts(K, kept.outputs)

    def coded(e: Any, w: int) -> None:
        for i, c in kept.windows[w]:
            if type(i) is not int or not 0 <= i < len(counts.outputs):
                raise ValueError(f"window {w}: a kept count of request {i!r}, and there are {len(counts.outputs)}")
            counts.put(e, i, c)
        if w == len(kept.windows) - 1:
            counts.end()
    return coded


class Reader:
    """`replay.replay`'s ``kept`` off a string: each emitting row's count, read from its step's window's second segment
    when the replay asks for it. `d` is the string's `window_code.Decoder`, at its last window's tail's end;
    `requests` are the replay's, in its order, each with its certified `output` and `release`, whose buckets do not
    decrease (`codec.Request` refuses a record's that do); `emitted` is `emitting`'s list, open over the same replay,
    whose length is the step that is executing."""

    def __init__(self, d: Any, K: int, requests: Sequence[Any], T: int, W: int, emitted: R.Emitted) -> None:
        self.d, self.T, self.emitted = d, T, emitted
        self.counts = Counts(K, [len(q.output) for q in requests])
        self.release = [q.release for q in requests]
        #: each request's next output's window, with its outputs out when that was so: stale once it emits again
        self.next = [(q.release[0] // T, i, 0) for i, q in enumerate(requests) if q.release]
        heapq.heapify(self.next)
        self.windows: list[list[tuple[int, int]]] = [[] for _ in range(W)]
        self.w, self.step = 0, -1
        d.segment(0)

    def __call__(self, i: int, _row: int) -> int:
        done = self.counts.done
        if self.step != len(self.emitted):
            self.step = len(self.emitted)
            while self.next and self.next[0][2] != done[self.next[0][1]]:
                heapq.heappop(self.next)
            self._to(self.next[0][0] if self.next else len(self.windows) - 1)
        c = self.counts.get(self.d, i)
        self.windows[self.w].append((i, c))
        if done[i] < len(self.release[i]):
            heapq.heappush(self.next, (self.release[i][done[i]] // self.T, i, done[i]))
        return c

    def _to(self, w: int) -> None:
        while self.w < w:
            self.w += 1
            self.d.segment(self.w)

    def end(self) -> Terms:
        """The terms read, once the replay has run: closes the segments left and ends the string, at most 7 zero bits
        of padding after it. Refuses (ValueError) counts that don't add up to each request's outputs. They are the
        replay's own (`terms`, on `codec.charged`'s windows): each request's release buckets are in order, so the
        earliest window that releases an output not yet out is the earliest of each request's next output's."""
        self._to(len(self.windows) - 1)
        self.counts.end()
        self.d.finish(True)
        return Terms(self.counts.outputs, tuple(tuple(w) for w in self.windows))


@contextmanager
def emitting() -> Iterator[R.Emitted]:
    """What each executed step's rows emit while this is open, in the steps' order (`replay.Replay.steps`): request ->
    the index of the first output its row emitted there, a row that emitted nothing left out. A what-if run on a fork
    is not the replay's own and is not counted. It wraps the scheduler's `update` while it is open
    (`engine.hooks.Hooks`), as `codec.replay_record` wraps the scheduler for its lookups, reads what it computes and
    changes none of it."""
    steps: R.Emitted = []
    update0 = R.Scheduler.update

    def update(s: Any, step: Any, samples: Any) -> list[int]:
        before = {i: s.reqs[i].n_out for i, _, _ in step.rows}
        done = update0(s, step, samples)
        if step.rows and not s.whatif:
            steps.append({i: m for i, m in before.items() if s.reqs[i].n_out > m})
        return done

    with Hooks("kept.emitting") as hooks:
        hooks.patch(R.Scheduler, "update", update)
        yield steps
