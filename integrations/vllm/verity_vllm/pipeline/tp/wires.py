"""A tensor-parallel window's collective sites at their producers' grain (`sites_circuit`): each request's W rank Programs'
producers of a partial, combines and the calls that read a combine's output as units, and each site's binding
(`program.tp_binding`) as wires, so that row p of every combine unit's parts, its parts cut W ways, is rank p's producer
unit's output row, and every read of a combine's output is the output row of the unit that computes it.

A wired row is committed under its writer's salt (`wiring.salts`), so a peer row and its producer's output are one public
string, and a peer row that isn't its producer's output is a wrong block of the window (`wiring.check`, and `check_drawn` on the
units a sample draws), never a free input: `interface.input_count` counts every peer row wired.

**Units.** A site's units are its producers' units, its combine units (`grain.lower`: one a word at coordinate grain, one a
row at row grain, none for a gather), the fused all-reduce's norm (one a rank) and its readers: a call of a rank Program
reading a combine's output is a unit of its own Definition, placed at the last site it reads. A read by another site's
combine (the fused all-reduce's residual) is that site's combine units' port. Each template's units are allocated in the
public order (request, then the site's place in its request Program's call order, its step then its layer, then stage:
producers, combines, norms, readers; then rank, then position), and `Sites.order` lists those ranges in it.

**Wires,** every one running forward in that order:
* each combine unit's parts, cut W ways, row p from rank p's producer unit at its position;
* the fused norm's r, cut N ways, from its rank's coordinate units;
* each read of a combine's output (NDAUDIT row 11), from the units computing those leaves (a gather's: the producers'),
  the reader's port cut so each row is one writer row. A read the grains don't fit (a row read a word at a time, a word
  read from part of a row, a port another read cuts otherwise) stays a free input, listed by name in `Sites.unwired`.

With ``ranks``, the window of those ranks alone: their units and the wires between them, every other rank's peer row a free
input, as in today's per-rank windows."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.pipeline.tp import grain as G
from verity_vllm.program import tp_binding as TB

from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as I

PRODUCER, COMBINE, NORM, CONSUMER = "producer", "combine", "norm", "consumer"
#: a range of one template's units: (descriptor id, first unit, units)
Range = tuple[str, int, int]
#: a writer row run of a combine's output at one rank: its leaves [lo, hi), a row's words, the writer's template, output and
#: first unit
Segment = tuple[int, int, int, str, str, int]


@dataclass(frozen=True)
class Sites:
    """`sites_circuit`'s circuit and where each unit came from: ``units[(request, site, stage, rank)]``, the stage's ranges in
    position order (``stage`` `PRODUCER`, `COMBINE`, `NORM` or `CONSUMER`); ``consumers[(request, rank, node)]``, the unit
    of a reader call; ``order``, every unit's range in the public order (module docstring), each template's tiling its
    units; ``unwired``, each read of a combine's output left a free input, and why."""
    circuit: I.Circuit
    units: Mapping[tuple[int, int, str, int], tuple[Range, ...]]
    consumers: Mapping[tuple[int, int, int], tuple[str, int]]
    order: tuple[Range, ...]
    unwired: tuple[str, ...]


def _pieces(ranges: Sequence[Range], q: int, m: int) -> Iterator[tuple[int, int, str, int, int]]:
    """Positions ``[q, q + m)`` of ``ranges`` (in position order), range by range: (position, range index, template, unit,
    positions)."""
    start = 0
    for k, (tid, at, n) in enumerate(ranges):
        lo, hi = max(q, start), min(q + m, start + n)
        if lo < hi:
            yield lo, k, tid, at + lo - start, hi - lo
        start += n


class _Window:
    def __init__(self, name: str, ranks: Sequence[int], world: int) -> None:
        self.name, self.ranks, self.world = name, list(ranks), world
        self.fns: dict[str, Any] = {}
        self.count: dict[str, int] = {}
        self.units: dict[tuple[int, int, str, int], list[Range]] = {}
        self.outs: dict[tuple[int, int, str, int], list[str]] = {}
        self.consumers: dict[tuple[int, int, int], tuple[str, int]] = {}
        self.order: list[Range] = []
        self.cuts: dict[str, dict[str, int]] = {}
        self.cut_of: dict[tuple[str, str], int] = {}
        self.wires: list[I.Wire] = []
        self.unwired: list[str] = []

    def alloc(self, key: tuple[int, int, str, int], fn: Any, n: int, out: str = "o0") -> tuple[str, int]:
        tid = descriptor_id(fn)
        self.fns.setdefault(tid, fn)
        at = self.count.get(tid, 0)
        self.count[tid] = at + n
        self.units.setdefault(key, []).append((tid, at, n))
        self.outs.setdefault(key, []).append(out)
        self.order.append((tid, at, n))
        return tid, at

    def cut(self, tid: str, port: str, k: int) -> int:
        """Cut ``tid``'s ``port`` ``k`` ways, unless it is already cut otherwise; -> its cut."""
        have = self.cut_of.setdefault((tid, port), k)
        if have == k and k > 1:
            self.cuts.setdefault(tid, {})[port] = k
        return have

    def words(self, tid: str, port: str) -> int:
        return dict(I.layout(self.fns[tid]).rows).get(port, 0)

    def site(self, i: int, j: int, s: TB.Site, low: G.Lowering) -> None:
        """Site ``j`` of request ``i``: its producers, combines and norm, and their parts and norm wires."""
        for p in self.ranks:
            for fn, out, n in low.producers[p]:
                self.alloc((i, j, PRODUCER, p), fn, n, out)
        for r in self.ranks:
            for fn, n in low.combines:
                self.alloc((i, j, COMBINE, r), fn, n, I.layout(fn).outs[0][0])
        for r in self.ranks if low.norm is not None else ():
            self.alloc((i, j, NORM, r), low.norm, 1)
        for r in self.ranks if low.combines else ():
            for q, _k, tc, uc, m in _pieces(self.units[i, j, COMBINE, r], 0, low.positions):
                parts = self.fns[tc].signature()[0][0][0]
                if self.cut(tc, parts, self.world) != self.world:
                    raise ValueError(f"{self.name}: {tc}'s {parts} is read otherwise than as {self.world} ranks' rows")
                for p in self.ranks:
                    for q2, k, tp, up, m2 in _pieces(self.units[i, j, PRODUCER, p], q, m):
                        self.wires.append(I.Wire(tc, parts, tp, self.outs[i, j, PRODUCER, p][k], at=uc + q2 - q, radix=(m2,),
                                                 base=up, step=(1,), cut=(p, 1)))
            if low.norm is not None:
                (tn, un, _), = self.units[i, j, NORM, r]
                port = low.norm.signature()[0][0][0]
                self.cut(tn, port, low.positions)
                for q, k, tc, uc, m in _pieces(self.units[i, j, COMBINE, r], 0, low.positions):
                    self.wires.append(I.Wire(tn, port, tc, self.outs[i, j, COMBINE, r][k], at=un, radix=(m,), base=uc,
                                             step=(1,), cut=(q, m)))

    def writers(self, i: int, j: int, r: int, low: G.Lowering) -> list[Segment]:
        """The writer rows of site ``j``'s combine output at rank ``r``, by its leaves."""
        w, segs, start = low.width, [], 0
        if not low.combines:
            for p in self.ranks:
                at = p * low.positions * w
                for q, k, tid, u, m in _pieces(self.units[i, j, PRODUCER, p], 0, low.positions):
                    segs.append((at + q * w, at + (q + m) * w, w, tid, self.outs[i, j, PRODUCER, p][k], u))
            return segs
        if low.norm is not None:
            (tn, un, _), = self.units[i, j, NORM, r]
            segs.append((0, low.positions, low.positions, tn, "o0", un))
            start = low.positions
        for (tid, at, n), out in zip(self.units[i, j, COMBINE, r], self.outs[i, j, COMBINE, r]):
            outs = I.layout(self.fns[tid]).outs
            if len(outs) > 1 and n == 1:
                for o, words in outs:
                    segs.append((start, start + words, words, tid, o, at))
                    start += words
            else:
                segs.append((start, start + n * outs[0][1], outs[0][1], tid, out, at))
                start += n * outs[0][1]
        return segs

    def read(self, what: str, target: tuple, off: int, lo: int, hi: int, segs: Sequence[Segment]) -> None:
        """Wire leaves ``[lo, hi)`` of a combine's output, read into ``target``'s port from its leaf ``off``, from the writer
        rows ``segs``; a run the grains don't fit is listed unwired."""
        for slo, shi, w, tw, out, u0 in segs:
            l1, h1 = max(lo, slo), min(hi, shi)
            if l1 >= h1:
                continue
            dest, m, uw = off + l1 - lo, (h1 - l1) // w, u0 + (l1 - slo) // w
            why = None
            if (l1 - slo) % w or (h1 - l1) % w:
                why = f"part of {w}-word rows of {tw}"
            elif target[0] == "row":
                _, tr, ur, port = target
                P = self.words(tr, port)
                if P % w or dest % w:
                    why = f"{w}-word rows of {tw} into its {P}-word {port} from word {dest}"
                elif (have := self.cut(tr, port, P // w)) != P // w:
                    why = f"{w}-word rows of {tw}, and {tr}'s {port} is cut {have} ways"
                else:
                    self.wires.append(I.Wire(tr, port, tw, out, at=ur, radix=(m,), base=uw, step=(1,), cut=(dest // w, m)))
            else:
                _, ranges, port = target
                if w != 1:
                    why = f"{w}-word rows of {tw} into a word a {ranges[0][0]} unit"
                else:
                    for q, _k, tr, ur, n in _pieces(ranges, dest, m):
                        if self.words(tr, port) != 1 or self.cut(tr, port, 1) != 1:
                            why = f"words of {tw} into {tr}'s {self.words(tr, port)}-word {port}"
                            break
                        self.wires.append(I.Wire(tr, port, tw, out, at=ur, radix=(n,), base=uw + q - dest, step=(1,)))
            if why is not None:
                self.unwired.append(f"{what} reads leaves [{l1}, {h1}): {why}")


def sites_circuit(bindings: Sequence[TB.RequestBinding | None], name: str, ranks: Sequence[int] | None = None) -> Sites:
    """The window of the collective sites of ``bindings`` (by advice request, None for a request no rank ran) at their
    producers' grain, of every rank or of ``ranks`` alone (module docstring); refused (ValueError) by name where a site has
    no grain (`grain.lower`)."""
    live = [b for b in bindings if b is not None]
    if not live:
        raise ValueError(f"{name}: no request's binding, so no collective site")
    world = live[0].world
    rs = list(range(world)) if ranks is None else sorted(set(ranks))
    if not rs or not set(rs) <= set(range(world)) or any(b.world != world for b in live):
        raise ValueError(f"{name}: ranks {rs} of bindings of worlds {sorted({b.world for b in live})}")
    win = _Window(name, rs, world)
    lowered: dict[tuple[int, int], G.Lowering] = {}
    for i, b in enumerate(bindings):
        if b is None:
            continue
        combines = [{s.combine[r][0]: j for j, s in enumerate(b.sites)} for r in range(world)]
        last: dict[tuple[int, int], list[int]] = {}
        for r in rs:
            at: dict[int, int] = {}
            for j, s in enumerate(b.sites):
                for x, *_ in s.readers[r] if s.readers else ():
                    if x not in combines[r]:
                        at[x] = j
            for x, j in at.items():
                last.setdefault((j, r), []).append(x)
        for j, s in enumerate(b.sites):
            where = f"{name}: request {i}'s collective {s.k} [{s.lo}, {s.hi})"
            low = lowered[i, j] = G.lower(b.fns[0][s.combine[0][0]],
                                          [[(b.fns[p][n], lo, hi) for n, lo, hi in s.own[p]] for p in range(world)], where)
            win.site(i, j, s, low)
            for r in rs:
                for x in sorted(last.get((j, r), ())):
                    win.consumers[i, r, x] = win.alloc((i, j, CONSUMER, r), b.fns[r][x], 1)
        for j, s in enumerate(b.sites):
            for r in rs:
                segs = win.writers(i, j, r, lowered[i, j])
                for x, a, off, lo, hi in s.readers[r] if s.readers else ():
                    xfn = b.fns[r][x]
                    port = xfn.signature()[0][a][0]
                    what = f"{name}: request {i}'s collective {s.k} [{s.lo}, {s.hi}): rank {r}'s node {x} ({xfn.id}) {port}"
                    if x in combines[r]:
                        j2 = combines[r][x]
                        low2, cs = lowered[i, j2], win.units.get((i, j2, COMBINE, r), [])
                        if low2.grain == G.COORD and cs:
                            target = ("coord", cs, port)
                        elif len(cs) == 1 and cs[0][2] == 1 and win.words(cs[0][0], port):
                            target = ("row", cs[0][0], cs[0][1], port)
                        else:
                            win.unwired.append(f"{what} reads leaves [{lo}, {hi}): no unit of collective {j2} reads them")
                            continue
                    else:
                        target = ("row", *win.consumers[i, r, x], port)
                    win.read(what, target, off, lo, hi, segs)
    templates = tuple((fn, win.count[tid]) for tid, fn in win.fns.items())
    circuit = I.Circuit(name, templates, wires=tuple(I.coalesce(win.wires)), cuts=win.cuts)
    return Sites(circuit, {k: tuple(v) for k, v in win.units.items()}, dict(win.consumers), tuple(win.order),
                 tuple(win.unwired))
