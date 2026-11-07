"""A class-G request's rows in a Commit window (`steps`): where its history ends and its Program begins (`segment_of`), and the
prefill chunks and launches its Program reads (`launch_of`)."""
from __future__ import annotations

from typing import Any

from verity.core.protocols.certifier import advice as I1
from verity_vllm.pipeline import window_advice as WA

from .build import Unexpressible

Rows = list[tuple[int, int, I1.Row]]  # a request's rows as (step, index in the step, row), in step order


def segment_of(i: int, LP: int, T: int, rows: Rows, carry: WA.Carried | None, left: bool) -> tuple[int, int]:
    """Request `i`'s (C, T) in class G, where its Program is its rows in the window alone: C its history, the positions it reads
    and doesn't compute (its prefix-cache hit, or where a request the window carries in continues: its carried `computed`), and
    positions to LP + T - 1 (its live decode rows; for a request the window leaves with outputs to keep, as far as its rows reach).
    A carried request and a cached one read their history as the boundary inputs (`build.build_program`'s C), never recompute it."""
    C = int(carry.computed) if carry is not None else int(rows[0][2].cache_hit)
    if left:
        end = max(row.start + row.n for _, _, row in rows)
        if end < LP:
            raise Unexpressible(f"request {i}: the window leaves it at position {end}, inside its {LP}-token prompt, so its Program "
                                f"here would sample nothing (a request carried out mid-prefill)")
        T = end - LP
    return C, T


def launch_of(i: int, LP: int, T: int, C: int, rows: Rows, steps: Any) -> tuple[tuple[int, ...], tuple]:
    """Request `i`'s prefill chunks and launch context in class G: when C < LP its prefill rows C..LP-1 in turn, each the rows one
    launch prefilled (one chunk for a prefill in one launch), and per step of its Program in the window (each chunk, then a decode
    a position up to LP + T - 1) every row of the advice step that row ran in as (n, start), in the step's batch order. Refused
    where its Program wouldn't hold every launch it ran in: a recompute (a row that runs positions its request had computed)."""
    chunks: list[int] = []
    for s, _, row in rows if C < LP else ():
        if (at := C + sum(chunks)) == LP:
            break
        if row.start != at or row.preempted or at + row.n > LP:
            raise Unexpressible(f"request {i}: step {s} runs its positions {row.start} … {row.start + row.n - 1}"
                                f"{' resumed' if row.preempted else ''} after its prefill reached {at}, a recompute without batch "
                                f"invariance (the chunked request wrapper prefills each prompt row once, in order)")
        chunks.append(row.n)
    k, decodes = len(chunks), LP + T - max(C, LP)
    for s, _, row in rows[k:k + decodes]:
        if row.n != 1 or row.preempted:
            raise Unexpressible(f"request {i}: step {s} runs {row.n} rows of it{' resumed' if row.preempted else ''}, a recompute "
                                f"without batch invariance (the request wrapper decodes one row a launch)")
    return tuple(chunks), tuple(tuple((r.n, r.start) for r in steps[s].rows) for s, _, _ in rows[:k + decodes])
