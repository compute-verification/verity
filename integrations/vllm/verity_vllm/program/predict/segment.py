"""A class-G request's rows in a Commit window (`steps`): where its history ends and its Program begins (`segment_of`), the
prefill chunks and launches its Program reads (`launch_of`), and what the window's edges carry (`edge_of`). A request's rows are
`steps.Rows`, (step, index in the step, advice row) in step order."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.workload import Edge

from .build import Unexpressible, decoder


def segment_of(i: int, LP: int, T: int, rows: Sequence[tuple[int, int, Any]], carry: WA.Carried | None, left: bool) -> tuple[int, int]:
    """Request `i`'s (C, T) in class G, where its Program is its rows in the window alone: C its history, the positions it reads
    and doesn't compute (its prefix-cache hit, or where a request the window carries in continues: its carried `computed`), and
    positions to LP + T - 1 (its live decode rows; for a request the window leaves with outputs to keep, as far as its rows reach,
    inside its prompt too, T < 0). A carried request and a cached one read their history as the boundary inputs
    (`build.build_program`'s C), never recompute it."""
    C = int(carry.computed) if carry is not None else int(rows[0][2].cache_hit)
    if left:
        T = max(row.start + row.n for _, _, row in rows) - LP
    return C, T


def launch_of(i: int, LP: int, T: int, C: int, rows: Sequence[tuple[int, int, Any]], steps: Any) -> tuple[tuple[int, ...], tuple]:
    """Request `i`'s prefill chunks and launch context in class G: when C < LP its prefill rows C..min(LP, LP + T)-1 in turn, each
    the rows one launch prefilled (one chunk for a prefill in one launch), and per step of its Program in the window (each chunk,
    then a decode a position up to LP + T - 1) every row of the advice step that row ran in as (n, start), in the step's batch
    order. Refused where its Program wouldn't hold every launch it ran in: a recompute (a row that runs positions its request had
    computed)."""
    chunks: list[int] = []
    for s, _, row in rows if C < LP else ():
        if (at := C + sum(chunks)) == LP:
            break
        if row.start != at or row.preempted or at + row.n > LP:
            raise Unexpressible(f"request {i}: step {s} runs its positions {row.start} … {row.start + row.n - 1}"
                                f"{' resumed' if row.preempted else ''} after its prefill reached {at}, a recompute without batch "
                                f"invariance (the chunked request wrapper prefills each prompt row once, in order)")
        chunks.append(row.n)
    k, decodes = len(chunks), max(LP + T - max(C, LP), 0)
    for s, _, row in rows[k:k + decodes]:
        if row.n != 1 or row.preempted:
            raise Unexpressible(f"request {i}: step {s} runs {row.n} rows of it{' resumed' if row.preempted else ''}, a recompute "
                                f"without batch invariance (the request wrapper decodes one row a launch)")
    return tuple(chunks), tuple(tuple((r.n, r.start) for r in steps[s].rows) for s, _, _ in rows[:k + decodes])


def edge_groups(net: Any, q: Any) -> tuple[str, ...]:
    """The history groups a window's edges carry of request `q` (its record, the same in every window: `window_advice.follows`):
    each one a row of its whole Program reads (`dsa.Net.edge_groups` at its last row's keys, its prompt and `output_len` - 1
    decodes), so each window computes and hands on a group's rows exactly where the request compiled whole computes them."""
    return net.edge_groups(int(q.prompt_len) + int(q.output_len) - 1)


def edge_of(phi: Any, advice: WA.Advice, units: Sequence[Any], idle: Sequence[int]) -> Edge | None:
    """What the window's edges carry in class G (`workload.Edge`): each request's rows [0, E) of every layer, E where it continues,
    of its history groups (`edge_groups`) as `Net.prefix_params(E)` lays them out, and from its prompt's end on its last kept
    output. In, each request the last window left with outputs to keep (carried here with outputs left, in that window's `close`
    order), E its carried `computed`; out, each one this window leaves so, in its `close` order, E its `computed` at the close,
    its unit's LP + T (an idle one's as it came in, its fields passed through). A request with no Program here (`partial` refused
    it) leaves no out member. None when neither edge carries one."""
    net = decoder(phi, 1, 0)

    def groups(i: int, E: int) -> tuple[tuple[str, int, int], ...]:
        q, carried = advice.requests[i], edge_groups(net, advice.requests[i])
        return (*((g, t.leaves, t.homogeneous_width()) for g, t in net.prefix_params(E) if g in carried),
                *((("last_token", 1, 32),) if E >= q.prompt_len else ()))

    ins = {i: groups(i, c.computed) for i, c in enumerate(advice.carried) if c.kept < advice.requests[i].output_len and c.computed}
    outs = []
    for c in WA.close(advice):
        if not (c.must and c.state.computed) or (u := units[c.request]) is None and c.request not in idle:
            continue
        if u is not None and u.LP + u.T != c.state.computed:
            raise Unexpressible(f"request {c.request}: the window leaves it to continue at {c.state.computed}, and its Program here "
                                f"reaches {u.LP + u.T}")
        outs.append((c.request, groups(c.request, c.state.computed)))
    return Edge(tuple(ins.items()), tuple(outs)) if ins or outs else None
