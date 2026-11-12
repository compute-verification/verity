"""A prefix-cache hit's sources as the compiler reads them (`steps.plan_steps`): that they hold the request's own KV
(`check_sources`), the request that computed each cached block for a Program that reads the hit (`producers_of`), and the
served rows that computed each position for a Program that recomputes it (`view_of`, `hit_chunks`)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import verity.protocols.certifier.verifier.advice as I1

from .build import Unexpressible

Rows = list[tuple[int, int, I1.Row]]  # a request's rows as (step, index in the step, row), in step order


def check_sources(s: int, row: I1.Row, prompts: Mapping[int, tuple[int, ...]]) -> None:
    """A cache hit's sources hold the KV of the request's own tokens: through each run, the source's certified prompt equals the
    hitting request's (the KV at a position reads every token up to it). Refused otherwise, or where a run leaves a prompt. A
    resume's run of its own KV is its own by I1's replay (the request computed those positions), past its prompt too."""
    r, at = row.request, 0
    for src in row.sources:
        end, a, b = at + src.n, prompts.get(row.request), prompts.get(src.request)
        if src.request == r and row.preempted:
            at = end
            continue
        if a is None or b is None:
            raise Unexpressible(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV, and one of the two has no certified "
                                f"prompt")
        if end > min(len(a), len(b)):
            raise Unexpressible(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV at positions {at} … {end - 1}, past "
                                f"a certified prompt (v0 checks sources against prompts only)")
        if a[:end] != b[:end]:
            raise Unexpressible(f"request {r}: a forged cache hit at step {s}: request {src.request}'s certified prompt differs from its "
                                f"own before position {end}, so that KV isn't its own")
        at = end


def producers_of(i: int, s: int, row: I1.Row, bs: int) -> tuple[int, ...]:
    """The request that computed each cached block of `row`'s hit (`program.prefix_plan`'s producers), from its sources: whole blocks
    of the engine's `bs` tokens, as vLLM's prefix cache serves them."""
    out: list[int] = []
    for src in row.sources:
        if src.n % bs:
            raise Unexpressible(f"request {i}: step {s}'s cache hit reads {src.n} tokens of request {src.request}'s KV, not whole blocks of "
                                f"{bs} (vLLM's prefix cache serves whole blocks)")
        out += [src.request] * (src.n // bs)
    return tuple(out)


View = list[tuple[int, int, tuple[int, int]]]  # runs (lo, hi, (step, index in the step)) of the served row that computed each position


def _cut(view: View, lo: int, hi: int) -> View:
    return [(max(a, lo), min(b, hi), key) for a, b, key in view if max(a, lo) < min(b, hi)]


def _join(view: View) -> View:
    out: View = []
    for a, b, key in view:
        if out and out[-1][1] == a and out[-1][2] == key:
            a = out.pop()[0]
        out.append((a, b, key))
    return out


def view_of(i: int, hi: int, rows: Mapping[int, Rows], carried: int, seen: tuple[int, ...] = ()) -> View:
    """The served row that computed the KV request `i` holds at each position below `hi`: through its first row's hit, each
    source's own view, then its rows, each position the first that reached it. `Unexpressible` where no row of the window computed
    a position (a request the window carries in, indices below `carried`, computed its history before it), or a resume may have
    recomputed it."""
    if hi <= 0:
        return []
    if i < carried or i in seen or not rows.get(i):
        raise Unexpressible(f"request {i}'s KV at positions below {hi} has no row of the window that computed it" + (
            " (the window carries it in)" if i < carried else " (its hit reads its own KV)" if i in seen else ""))
    (_, _, first), at, out = rows[i][0], 0, []
    for src in first.sources:
        out += _cut(view_of(src.request, min(at + src.n, hi), rows, carried, (*seen, i)), at, hi)
        at += src.n
    reached = at
    for s, j, row in rows[i]:
        if reached >= hi:
            break
        if row.preempted or row.start > reached:
            raise Unexpressible(f"request {i}: step {s} runs its positions {row.start} … {row.start + row.n - 1}"
                                f"{' resumed' if row.preempted else ''} after it reached {reached}, so no one row computed its KV there")
        if (top := min(row.start + row.n, hi)) > reached:
            out.append((reached, top, (s, j)))
            reached = top
    if reached < hi:
        raise Unexpressible(f"request {i}'s rows reach position {reached}, not the {hi} a hit on its KV reads")
    return _join(out)


def hit_chunks(i: int, C: int, rows: Mapping[int, Rows], steps: Any, carried: int) -> list[tuple[int, int, int, int]]:
    """Request `i`'s hit of C positions as the prefill chunks that recompute it, each (step, start, rows, rows of its sequence in the
    step): each run of the rows that computed it in serving (`view_of`), at that row's launch, computing its first rows below C.
    That chain computes what serving did only where each such row started its run and its own history was the reader's, so
    `Unexpressible` otherwise (two requests computed one block in the same step), or where a row ran in the reader's step or later."""
    s0, view = rows[i][0][0], view_of(i, C, rows, carried)
    out = []
    for lo, hi, (s, j) in view:
        row = steps[s].rows[j]
        if s >= s0 or row.start != lo:
            raise Unexpressible(f"request {i}: its hit's positions {lo} … {hi - 1} are request {row.request}'s rows from position "
                                f"{row.start} at step {s}, " + (f"not before its own first step {s0}" if s >= s0 else
                                "a row that starts before them, over history the hit doesn't hold"))
        if (theirs := view_of(row.request, lo, rows, carried)) != _cut(view, 0, lo):
            raise Unexpressible(f"request {i}: its hit's positions {lo} … {hi - 1} are request {row.request}'s rows at step {s}, "
                                f"computed over a history of other rows than the hit holds ({theirs[:2]} vs {_cut(view, 0, lo)[:2]})")
        out.append((s, lo, hi - lo, int(row.n)))
    return out
