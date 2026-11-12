"""Advice v1's kept counts: with a draft, the drafts each emitting row kept, sent and charged.

With a draft (the registered engine's `num_speculative_tokens`, K > 0) a decode row carries up to K drafts after its
token, and what it emits is its token and the drafts it kept (`replay`'s docstring). How many it kept is the model's
accept rule's, run on the row's logits, and no proof yet holds that rule. So the count is the prover's choice, and v1
sends and charges it:
- each request's counts, one per row of it that emitted, in order, in the window the request's body lands in, the
  window's requests in release order (`lengths.terms`, as its length term is);
- a count c as the bits "c > j" for j = 0, 1, … until one is 0 or j is the most the row can keep: K, or the outputs
  the request still lacks after the row's own token. Each bit is coded under the window coder's adaptive estimate
  (`window_code.KT`) in the context (the request's count before, j) (`context`);
- in the window's second coded segment, after its length terms (`window_code.Encoder.segment`).

The decoder needs K and each request's certified output count, and nothing else: a request's counts end where they and
its rows add up to its outputs, so the reader knows where the segment's message ends.

What is sent is held to the rows by the replay, which refuses a count over the drafts the row it schedules there
carries, and one past where the request stops (`replay.Scheduler`). A request's counts change no row before its
admission, so `lengths.check_admitted`, which `codec` runs under a draft whether or not lengths are sent, keeps the
charge no later than the first window whose rows a count changes.

`emitting` reads off the replay what each executed step's rows emitted, which under a draft is not one output a row.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline.advice_code import replay as R

__all__ = ["context", "emitting", "get", "put", "tails"]


def context(before: int | None, j: int) -> str:
    """The estimate's context for the bit "c > j" of a count whose request's count before was `before` (None for the
    request's first emitting row): `window_code`'s bit symbol ``kept`` in it is the book's prior ``kept/<context>``."""
    return f"{'first' if before is None else before}.{j}"


def _most(K: int, outputs: int, done: int) -> int:
    """The most drafts a row can keep where its request has emitted `done` of its `outputs`."""
    return min(K, outputs - done - 1)


def put(e: Any, K: int, outputs: int, counts: Sequence[int]) -> None:
    """Codes one request's counts through the window coder `e` (`window_code.Encoder`); `outputs` is its certified
    output count. Refuses (ValueError) counts that aren't a row's, or that with their rows don't add up to `outputs`."""
    done, before = 0, None
    for c in counts:
        most = _most(K, outputs, done)
        if most < 0:
            raise ValueError(f"a request's rows before its last add up to its {outputs} outputs")
        if type(c) is not int or not 0 <= c <= most:
            raise ValueError(f"a row kept {c!r} drafts where it can keep 0 to {most}: {done} of its request's "
                             f"{outputs} outputs are out, and a row drafts {K}")
        for j in range(min(c + 1, most)):
            e.bit("kept", c > j, context(before, j))
        done, before = done + 1 + c, c
    if done != outputs:
        raise ValueError(f"a request's rows and the drafts they kept add up to {done} outputs, not its {outputs}")


def get(d: Any, K: int, outputs: int) -> tuple[int, ...]:
    """`put`'s mirror through `window_code.Decoder`: one request's counts, read until they and their rows add up to
    `outputs`."""
    out: list[int] = []
    done, before = 0, None
    while done < outputs:
        c, most = 0, _most(K, outputs, done)
        while c < most and d.bit("kept", context(before, c)):
            c += 1
        out.append(c)
        done, before = done + 1 + c, c
    return tuple(out)


def tails(requests: Sequence[Any], ks: Sequence[Sequence[int]],
          kept: Sequence[Sequence[int]] | None) -> list[list[tuple[int, tuple[int, ...]]]] | None:
    """Each window's kept terms as `put` takes them: for each request of `ks` (`codec.kept_terms`), its output count
    and the drafts its emitting rows kept (`kept`, by ordinal). None for None: without a draft there are none."""
    if kept is None:
        return None
    return [[(len(requests[k].output), tuple(kept[k])) for k in w] for w in ks]


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
