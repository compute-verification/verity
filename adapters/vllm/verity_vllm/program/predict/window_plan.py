"""A served window's plan at its close: its units as `window_rows.Rows` fills them, made from its lowering and each slot's
served record with no value read, and the rows each served node feeds, laid out from the retained step blocks as the row
hash reads them (`stage`)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from verity.commitments.periods import interface as IF

from .window_circuit import Lowered
from .window_rows import COMPUTED, EVALUATED, SERVED, SOURCES, _fields, positions

__all__ = ["Feed", "Plan", "Span", "stage"]


class Span(NamedTuple):
    """A committed member's bytes [lo, hi) at engine step ``step`` of the window, by its layouts' name, in its committed
    ``dtype``, read as ``bits``-wide elements: the port's, recast where the dtype is another width (as `Slot` casts)."""
    step: int
    name: str
    lo: int
    hi: int
    dtype: str
    bits: int


@dataclass(frozen=True)
class Feed:
    """``rows`` output rows of ``words`` u16 words each, from unit ``unit`` of template ``tid`` on, of output ``out``: a served
    node's words, the concatenation of ``spans`` cut ``rows`` ways."""
    tid: str
    unit: int
    out: str
    rows: int
    words: int
    spans: tuple[Span, ...]


class Plan:
    """A served window's units as `Rows` fills them, made from its lowering and each slot's served record (`Served.where`)
    alone, so at the window's close with no value read: by template, each unit's every output's source (`SOURCES`;
    ``ports``, a unit's own the least of its outputs', `Rows.source`), and each run of output rows a served node fills, where
    the committer holds their words (`Feed`). A slot with no ``where`` is computed throughout (a stand-in's)."""

    def __init__(self, lowered: Lowered, served: Sequence[Any]) -> None:
        if lowered.requests is None:
            raise ValueError("a class circuit's plan: the committer fills a served window's circuit")
        if len(served) != len(lowered.programs):
            raise ValueError(f"{len(served)} slots' served records for {len(lowered.programs)} Programs")
        c = self.circuit = lowered.circuit
        self.fns = dict(zip(c.ids, (fn for fn, _ in c.templates)))
        self.n = dict(zip(c.ids, (k for _, k in c.templates)))
        self.lays = {tid: IF.layout(fn) for tid, fn in self.fns.items()}
        self.outs = {tid: [o for o, _ in lay.outs] for tid, lay in self.lays.items()}
        self.ports = {t: np.full((self.n[t], len(self.outs[t])), EVALUATED, dtype=np.int8) for t in self.lays}
        self.feeds: list[Feed] = []
        for P, S in zip(lowered.programs, served):
            self._slot(P, S)

    @property
    def source(self) -> dict[str, np.ndarray]:
        return {t: p.min(axis=1) if p.shape[1] else np.full(len(p), EVALUATED, dtype=np.int8) for t, p in self.ports.items()}

    def _slot(self, P: Any, S: Any) -> None:
        nodes, pos = P.fn.body.nodes, positions(P.fn)
        first: dict[int, int] = {}
        for j, p in enumerate(pos):
            first.setdefault(p, j)
        where = getattr(S, "where", None)
        for j, pieces in P.outs.items():
            node, p = nodes[j], pos[j]
            got = None if where is None else where(p, j - first[p], node)
            fields = _fields(node.fn)
            spans = None if got is None else _field_spans(got[1], fields, f"{P.name}'s node {j} at position {p}")
            total = sum(f.leaves * _bits(f) // 16 for f in fields)
            sums = [(total // 2 + at * r.words, r) for at, r in enumerate(P.norms.get(j, ()))]
            for lo, hi, r in [*pieces, *((lo, lo + r.words, r) for lo, r in sums)]:
                k, o = (hi - lo) // r.words, self.outs[r.tid].index(r.out)
                at = self.ports[r.tid][r.unit:r.unit + k, o]
                np.minimum(at, COMPUTED if spans is None else SERVED, out=at)
                if spans is not None:
                    self.feeds.append(Feed(r.tid, r.unit, r.out, k, r.words, _cut(spans, lo, hi)))

    def describe(self) -> dict[str, Any]:
        """The committer's order: each template in the circuit's order, its units, its input rows and output ports at their
        words and each port's units by source; the window's units by source, and how many runs of rows the served nodes feed."""
        src = self.source
        return {"templates": [{"id": t, "units": int(self.n[t]), "ins": [[p, int(w)] for p, w in self.lays[t].rows],
                               "outs": [[o, int(w), {s: int((self.ports[t][:, i] == k).sum()) for k, s in enumerate(SOURCES)}]
                                        for i, (o, w) in enumerate(self.lays[t].outs)]} for t in self.circuit.ids],
                "units": {s: int(sum((x == k).sum() for x in src.values())) for k, s in enumerate(SOURCES)},
                "feeds": len(self.feeds), "fed_rows": sum(f.rows for f in self.feeds)}


def _bits(ty: Any) -> int:
    (bits,) = ty.width_set()
    return int(bits)


def _field_spans(located: Sequence[Any], fields: Sequence[Any], what: str) -> list[tuple[int, int, Any, int]]:
    """Each return field's words [lo, hi) of the node's, its located member and its element width."""
    if len(located) != len(fields):
        raise ValueError(f"{what}: served as {len(located)} members, not its {len(fields)} return fields")
    out, at = [], 0
    for loc, ty in zip(located, fields):
        bits = _bits(ty)
        n = ty.leaves * bits // 16
        if (loc.hi - loc.lo) % ty.leaves:
            raise ValueError(f"{what}: {loc.hi - loc.lo} committed bytes for its field's {ty.leaves} elements")
        out.append((at, at + n, loc, bits))
        at += n
    return out


def _cut(spans: Sequence[tuple[int, int, Any, int]], lo: int, hi: int) -> tuple[Span, ...]:
    """The node's words [lo, hi) as the committed bytes that hold them, field by field."""
    out = []
    for a, b, loc, bits in spans:
        x, y = max(a, lo), min(b, hi)
        if x >= y:
            continue
        size = (loc.hi - loc.lo) // ((b - a) * 16 // bits)
        e0, e1 = (x - a) * 16 // bits, (y - a) * 16 // bits
        out.append(Span(int(loc.step), str(loc.name), loc.lo + e0 * size, loc.lo + e1 * size, str(loc.dtype), bits))
    return tuple(out)


def stage(feed: Feed, block: Callable[[int, str], tuple[Any, int]]) -> Any:
    """``feed``'s rows as the row hash reads them, a ``(rows, stride)`` uint8 tensor, the stride the row's ``2·words`` bytes
    rounded up to 16 and zero past them: each span's bytes sliced from its step's retained block (``block(step, name)``: the
    block, a uint8 tensor, and the member's offset in its step's stream), its elements, unsigned words of their committed
    width, taken at the port's ``bits``, on the block's device and with no copy to the host."""
    import torch

    from verity_vllm.check.replay.opening import _ITEMSIZE

    signed = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    parts = []
    for s in feed.spans:
        blk, off = block(s.step, s.name)
        b, size = blk[off + s.lo:off + s.hi], _ITEMSIZE[s.dtype]
        if size * 8 != s.bits:
            x = b.contiguous().view(signed[size]).to(signed[s.bits // 8])
            b = (x & ((1 << 8 * size) - 1) if size * 8 < s.bits else x).contiguous().view(torch.uint8)
        parts.append(b)
    n = 2 * feed.words
    flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)
    if flat.numel() != feed.rows * n:
        raise ValueError(f"{feed.tid} unit {feed.unit} {feed.out}: {flat.numel()} bytes for {feed.rows} rows of {n}")
    out = flat.new_zeros((feed.rows, -(-n // 16) * 16))
    out[:, :n] = flat.view(feed.rows, n)
    return out
