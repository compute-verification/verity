"""A node window's boundary reads (`node_window.node_window`): where the class's Definition's parameters and return meet
the window's units, as `interface.Read`s, each in 16-bit leaves of the parameter or the return it names: the circuit's
`Circuit.boundary_reads`. Two so far:

- **The prompt's read** (`prompt_reads`): each unit row that reads the Program's prompt reads its ids from `PROMPT`, at
  the leaves its stand-in rows name (`NodeWindow.stood`, built with ``stand_ins``). The prefill's embedding
  reads are block b of token t at unit ``a + b M + t`` (`node_window_embed.embedding_shard`), so one read of
  ``radix [blocks, M]`` and ``step [0, 2]``. Every rank reads the prompt.
- **The served ids** (`served_reads`): the leaves of the Program's first return field, the tokens, as the window writes
  them (the select combine's ``o0`` a token, `node_window_select`), at their leaves of `EGRESS`. Every rank computes the
  same tokens; rank 0 alone declares them, so over the ranks' windows each leaf of the return is written once
  (`written_once`). The return's row is O ids zero past the n served (`boundary_span.egress_row`), and the combines'
  ids are it at every n: each step's select writes the next step's live bit, and the steps past n write zeros.

A rank's window is built here (`rank_window`), the one place that states what a rank reads: its registered reads, its
window's own, and its boundary reads, these. The window declares no ingress, and the rows its reads fill leave its free
list. `node_window_join.join` joins these ranks and the served path commits each (`node_window_rows.commit_rank_window`),
so the two read the same build.

Each list is coalesced (`interface.coalesce`), so a read is a run of units, not a unit. Refused by name (`Unexpressible`):
a window built without stand-ins, a prompt row that isn't one whole stand-in row of the prompt, and a token not one
template's output field."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from verity.commitments.periods import interface as IF

from . import node_window_join as NJ
from .build import Unexpressible
from .node_window import _GRAIN, STAND_IN, Lowering, NodeWindow, _Free, node_window
from .node_window import PROMPT as PROGRAM_PROMPT
from .window_builder import Grain

#: the class's Definition's parameter holding the prompt's ids and its return's field holding the served ids
PROMPT, EGRESS = "b.prompt", "o0"


def prompt_reads(w: NodeWindow, param: str = PROMPT) -> list[IF.Read]:
    """Each unit row ``w`` leaves free under the Program's prompt, read from ``param`` at the leaves its stand-in row names."""
    if not w.stand_ins:
        raise Unexpressible(f"{w.name}: a window built without stand_ins keeps no rows for what it leaves free, so where "
                            "it reads the prompt isn't known")
    stood = f"{STAND_IN}p:{PROGRAM_PROMPT}"
    reads = []
    for tid, port, at, _count, rows, grid in w.stood:
        if not any(r.tid == stood for rs in grid for r in rs):
            continue
        if rows != (0, 1, 1) or any(len(rs) != 1 or rs[0].tid != stood for rs in grid):
            raise Unexpressible(f"{w.name}: {tid} {port} from unit {at}: a port reading the prompt is read whole from it")
        reads += [IF.Read(tid, port, param, at + i, (1,), int(rs[0].out.split(":")[0]), (0,)) for i, rs in enumerate(grid)]
    return IF.coalesce(reads)


def served_reads(w: NodeWindow, rank: int, param: str = EGRESS) -> list[IF.Read]:
    """The rows writing the Program's tokens (its first return field), each at its leaves of ``param``; none at a rank
    other than 0."""
    if rank != 0:
        return []
    body = w.fn.body
    tokens = (getattr(body.ret_type, "fields", None) or (body.ret_type,))[0]
    n = (tokens[1] if isinstance(tokens, tuple) else tokens).leaves
    try:
        rows = w.rows(body.ret.slice(0, n))
    except _Free as e:
        raise Unexpressible(f"{w.name}: its tokens are free ({e}); a served id is a unit's output") from None
    if any(r.tid.startswith(STAND_IN) for r in rows):
        raise Unexpressible(f"{w.name}: a token stands in for an unlowered node; a served id is a unit's output")
    base = np.cumsum([0, *(r.words for r in rows)]).tolist()
    return IF.coalesce([IF.Read(r.tid, r.out, param, r.unit, (1,), base[k], (0,)) for k, r in enumerate(rows)])


def boundary_reads(w: NodeWindow, rank: int) -> list[IF.Read]:
    """Rank ``rank``'s window's boundary reads: its prompt's and, at rank 0, the served ids."""
    return [*prompt_reads(w), *served_reads(w, rank)]


def rank_window(rank: int, fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN) -> NJ.Rank:
    """Rank ``rank``'s Program ``fn`` lowered into its window (`node_window.node_window`, with stand-ins), the frame
    dropped: its circuit declares the rank's `boundary_reads`, and the free inputs they fill are no longer free
    (`node_window_join.unfilled`)."""
    c, w = node_window(fn, families, grain, stand_ins=True)
    reads = tuple(boundary_reads(w, rank))
    c = replace(c, boundary_reads=reads)
    return NJ.Rank(rank, c, NJ.unfilled(w.free, reads), {t: tuple(s) for t, s in w.b.spans.items()}, w.steps, w.census(c))


def written_once(windows: Sequence[tuple[IF.Circuit, Sequence[IF.Read]]], leaves: int, param: str = EGRESS) -> None:
    """Refuse (ValueError) the windows' reads (each window's circuit and its reads) unless they write each of the
    return's ``leaves`` leaves of ``param`` exactly once: a read's row the words of its reader's output field from each
    source on."""
    hits = np.zeros(leaves, np.int64)
    for circuit, reads in windows:
        for r in (r for r in reads if r.param == param):
            words = dict(circuit.layouts[r.reader][0].outs)[r.port]
            at = (r.source(np.arange(r.units))[:, None] + np.arange(words)).reshape(-1)
            if (at >= leaves).any():
                raise ValueError(f"read {r.doc()}: leaves to {int(at.max())}, past the return's {leaves}")
            np.add.at(hits, at, 1)
    if (twice := np.flatnonzero(hits > 1)).size:
        raise ValueError(f"{param}: leaf {int(twice[0])} written by {int(hits[twice[0]])} reads; each is written once")
    if (none := np.flatnonzero(hits == 0)).size:
        raise ValueError(f"{param}: leaf {int(none[0])} written by none; each is written once")


__all__ = ["EGRESS", "PROMPT", "boundary_reads", "prompt_reads", "rank_window", "served_reads", "written_once"]
