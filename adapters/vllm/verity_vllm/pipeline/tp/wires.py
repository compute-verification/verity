"""A tensor-parallel window's collective sites at a grain C-Flock proves (`sites_circuit`): each request's W rank Programs'
producers of a partial, the units combining them and the calls that read a combine's output, with each site's binding
(`program.tp_binding`) as wires. Row p of every combine unit's parts, cut W ways, is rank p's partial row there, and every
read of a combine's output is the output row of the unit that computes it.

A wired row is committed under its writer's salt (`wiring.salts`), so a peer row and its writer's output are one public
string. A peer row that isn't its writer's output is a wrong block of the window (`wiring.check`, and `check_drawn` on the
units a sample draws), never a free input: `interface.input_count` counts every peer row wired.

**Units** (`grain.lower`). A site's units are:
* its producers' units, per rank;
* at coordinate grain, each rank's gathers of its words into rows of K;
* its combine units (one a chunk at coordinate grain, one a row at row grain, none for a gather), the gathers of a token
  row's chunks, and the fused all-reduce's norm, one unit for every rank;
* its readers, per rank: a call of a rank Program reading a combine's output is a unit of its own Definition, placed at the
  last site it reads. A token select (`TokenSelect_v4{V}`) is instead the served window's split of it
  (`grain.GROUP`): V / C units `TokenSelectGroups_v1{C, GROUP}` reading its logits as their writers' rows, C
  `grain.select_width`, and one `TokenSelectCombine_v2{V, GROUP}` reading their bests and indices and ``x0``, logits[0]'s
  producer unit's word. A call producing a later site's partial (the LM head reading the last norm) is its producer units,
  each reading it, and a read by another site's combine (the fused all-reduce's residual) is that site's combine units' port.

Each template's units are allocated in the public order: request; then the site's place in its request Program's call
order, its step then its layer; then stage (`PRODUCER`, `GATHER`, `COMBINE`, `OUTPUT`, `NORM`, `CONSUMER`); then rank (none
for the stages every rank reads, `ALL`); then position. `Sites.order` lists those ranges in it, and is the circuit's
`Circuit.order`, since a template's units sit at several steps (the Gemm coordinates of one layer and the next).

**Wires**, every one running forward in that order:
* each gather's K one-word rows, from its rank's producer units in position order;
* each combine unit's parts, row p from rank p's partial row at its chunk (a gather's, else a producer unit's);
* each token row's chunks, from the combine units in order, and the norm's r, that row whole;
* each read of a combine's output, from the units computing those leaves (a gather's: the ranks' partial rows), so a value
  a unit reads is the output of the unit that computed it, never a free witness; the reader's port is cut so each row is one
  writer row. A read the grains don't fit (part of a writer row, a port
  another read cuts otherwise) stays a free input, listed by name in `Sites.unwired`.

With ``ranks``, the window of those ranks alone: their units, the units every rank reads, and the wires between them, every
other rank's partial row a free input, as in today's per-rank windows."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.pipeline.tp import grain as G
from verity_vllm.program import tp_binding as TB

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.definitions.registry import dead_free as DF
from verity.commitments.windows import interface as I

PRODUCER, GATHER, COMBINE, OUTPUT, NORM, CONSUMER = "producer", "gather", "combine", "output", "norm", "consumer"
#: the rank of the stages every rank reads: combines, output gathers and norms
ALL = -1
#: a range of one template's units: (descriptor id, first unit, units)
Range = tuple[str, int, int]
#: a writer row run of a site's output: its leaves [lo, hi), a row's words, the writer's template, output and first unit
Segment = tuple[int, int, int, str, str, int]


@dataclass(frozen=True)
class Sites:
    """`sites_circuit`'s circuit and where each unit came from: ``units[(request, site, stage, rank)]``, the stage's ranges in
    position order (rank `ALL` for the stages every rank reads); ``consumers[(request, rank, node)]``, the units of a reader
    call that produces no partial; ``order``, every unit's range in the public order (module docstring), each template's
    tiling its units; ``unwired``, each read of a combine's output left a free input, and why."""
    circuit: I.Circuit
    units: Mapping[tuple[int, int, str, int], tuple[Range, ...]]
    consumers: Mapping[tuple[int, int, int], tuple[Range, ...]]
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
        self.consumers: dict[tuple[int, int, int], tuple[Range, ...]] = {}
        #: a token select's group units and select, by (request, rank, node)
        self.selects: dict[tuple[int, int, int], tuple[Range, Range]] = {}
        self.order: list[Range] = []
        self.cuts: dict[str, dict[str, int]] = {}
        self.cut_of: dict[tuple[str, str], int] = {}
        self.wires: list[I.Wire] = []
        self.unwired: list[str] = []

    def alloc(self, key: tuple[int, int, str, int], fn: Any, n: int, out: str = "o0") -> Range:
        tid = descriptor_id(fn)
        self.fns.setdefault(tid, fn)
        at = self.count.get(tid, 0)
        self.count[tid] = at + n
        self.units.setdefault(key, []).append((tid, at, n))
        self.outs.setdefault(key, []).append(out)
        self.order.append((tid, at, n))
        return tid, at, n

    def cut(self, tid: str, port: str, k: int) -> int:
        """Cut ``tid``'s ``port`` ``k`` ways, unless it is already cut otherwise; -> its cut."""
        have = self.cut_of.setdefault((tid, port), k)
        if have == k and k > 1:
            self.cuts.setdefault(tid, {})[port] = k
        return have

    def words(self, tid: str, port: str) -> int:
        return dict(I.layout(self.fns[tid]).rows).get(port, 0)

    def rows(self, key: tuple[int, int, str, int]) -> Iterator[tuple[int, str, str, int, int]]:
        """The output rows of ``key``'s units in position order, range by range: (position, template, output, unit, units)."""
        for q, k, tid, u, m in _pieces(self.units.get(key, ()), 0, sum(n for *_, n in self.units.get(key, ()))):
            yield q, tid, self.outs[key][k], u, m

    def spread(self, ranges: Sequence[Range], port: str, k: int, x0: int, tw: str, out: str, u0: int, m: int) -> str | None:
        """Wire rows ``[x0, x0 + m)`` of ``ranges``' port ``port``, cut ``k`` ways and laid unit after unit (row x is unit
        x // k's row x % k), from writer ``tw``'s ``out`` rows of its units ``[u0, u0 + m)``; -> why not, or None."""
        new, x, end = [], x0, x0 + m
        while x < end:
            u, c = divmod(x, k)
            whole = (end - x) // k if c == 0 else 0
            span = whole * k if whole else min(k - c, end - x)
            for q, _i, tr, ur, nu in _pieces(ranges, u, max(whole, 1)):
                if (have := self.cut(tr, port, k)) != k:
                    return f"{tr}'s {port} is cut {have} ways, not {k}"
                base = u0 + x - x0 + (q - u) * k
                if whole:
                    new.append(I.Wire(tr, port, tw, out, at=ur, radix=(nu, k) if k > 1 else (nu,), base=base,
                                      step=(k, 1) if k > 1 else (1,), cut=(0, k)))
                else:
                    new.append(I.Wire(tr, port, tw, out, at=ur, radix=(span,), base=base, step=(1,), cut=(c, span)))
            x += span
        self.wires.extend(new)
        return None

    def gather_into(self, ranges: Sequence[Range], port: str, k: int, key: tuple[int, int, str, int]) -> None:
        """``ranges``' port ``port``, cut ``k`` ways, from ``key``'s units' output rows in position order."""
        for q, tw, out, u, m in self.rows(key):
            if (why := self.spread(ranges, port, k, q, tw, out, u, m)) is not None:
                raise ValueError(f"{self.name}: {why}")

    def site(self, i: int, j: int, low: G.Lowering) -> None:
        """Site ``j`` of request ``i``: its producers, gathers, combines, output gathers and norm, and the wires among them."""
        for p in self.ranks:
            for fn, out, n in low.producers[p]:
                self.alloc((i, j, PRODUCER, p), fn, n, out)
        if low.gather is not None:
            for p in self.ranks:
                g = self.alloc((i, j, GATHER, p), low.gather, low.rows)
                self.gather_into([g], low.gather.signature()[0][0][0], low.chunk, (i, j, PRODUCER, p))
        if not low.combines:
            return
        for fn, n in low.combines:
            self.alloc((i, j, COMBINE, ALL), fn, n, I.layout(fn).outs[0][0])
        for p in self.ranks:
            partial = (i, j, GATHER if low.gather is not None else PRODUCER, p)
            for q, _k, tc, uc, m in _pieces(self.units[i, j, COMBINE, ALL], 0, low.rows):
                parts = self.fns[tc].signature()[0][0][0]
                if self.cut(tc, parts, self.world) != self.world:
                    raise ValueError(f"{self.name}: {tc}'s {parts} is read otherwise than as {self.world} ranks' rows")
                for q2, k, tp, up, m2 in _pieces(self.units[partial], q, m):
                    self.wires.append(I.Wire(tc, parts, tp, self.outs[partial][k], at=uc + q2 - q, radix=(m2,), base=up,
                                             step=(1,), cut=(p, 1)))
        r_row = (i, j, COMBINE, ALL)
        if low.output is not None:
            o = self.alloc((i, j, OUTPUT, ALL), low.output, low.tokens)
            self.gather_into([o], low.output.signature()[0][0][0], low.row // low.chunk, r_row)
            r_row = (i, j, OUTPUT, ALL)
        if low.norm is not None:
            nm = self.alloc((i, j, NORM, ALL), low.norm, low.tokens)
            self.gather_into([nm], low.norm.signature()[0][0][0], 1, r_row)

    def writers(self, i: int, j: int, low: G.Lowering) -> list[Segment]:
        """The writer rows of site ``j``'s output, by its leaves."""
        segs: list[Segment] = []
        if not low.combines:
            for p in self.ranks:
                at = p * low.positions * low.width
                rows = (i, j, GATHER if low.gather is not None else PRODUCER, p)
                for q, tid, out, u, m in self.rows(rows):
                    segs.append((at + q * low.chunk, at + (q + m) * low.chunk, low.chunk, tid, out, u))
            return segs
        start = 0
        if low.norm is not None:
            for _q, tid, out, u, m in self.rows((i, j, NORM, ALL)):
                segs.append((start, start + m * low.row, low.row, tid, out, u))
                start += m * low.row
        if low.output is not None:
            for _q, tid, out, u, m in self.rows((i, j, OUTPUT, ALL)):
                segs.append((start, start + m * low.row, low.row, tid, out, u))
                start += m * low.row
            return segs
        for (tid, at, n), out in zip(self.units[i, j, COMBINE, ALL], self.outs[i, j, COMBINE, ALL]):
            outs = I.layout(self.fns[tid]).outs
            if len(outs) > 1 and n == 1:
                for o, words in outs:
                    segs.append((start, start + words, words, tid, o, at))
                    start += words
            else:
                segs.append((start, start + n * outs[0][1], outs[0][1], tid, out, at))
                start += n * outs[0][1]
        return segs

    def word(self, i: int, j: int, low: G.Lowering, leaf: int) -> tuple[str, str, int] | None:
        """The one-word unit computing leaf ``leaf`` of site ``j``'s output, a gather of words: (template, output, unit)."""
        p, q = divmod(leaf, low.positions)
        if low.combines or low.grain != G.COORD or p not in self.ranks:
            return None
        (_q, k, tid, u, _m), = _pieces(self.units[i, j, PRODUCER, p], q, 1)
        return tid, self.outs[i, j, PRODUCER, p][k], u

    def consumer(self, i: int, j: int, r: int, x: int, fn: Any, segs: Sequence[Segment]) -> None:
        """Rank ``r``'s reader call ``x`` (Definition ``fn``) placed at site ``j``: a unit of ``fn``, or a token select's
        group units and select (module docstring), where its logits' writer rows fit whole groups."""
        widths = {w for _lo, _hi, w, *_ in segs}
        C = None
        if fn.defn is DF.TokenSelectV4 and len(widths) == 1:
            C = G.select_width(fn.statics["V"], widths.pop(), G.GROUP)
        if C is None:
            self.consumers[i, r, x] = (self.alloc((i, j, CONSUMER, r), fn, 1),)
            return
        V = fn.statics["V"]
        groups = self.alloc((i, j, CONSUMER, r), bind(DF.TokenSelectGroupsV1, C=C, G=G.GROUP), V // C)
        select = self.alloc((i, j, CONSUMER, r), bind(DF.TokenSelectCombineV2, V=V, C=G.GROUP), 1)
        for port, out in (("best", "o0"), ("idx", "o1")):
            if (why := self.spread([select], port, V // C, 0, groups[0], out, groups[1], V // C)) is not None:
                raise ValueError(f"{self.name}: {why}")
        self.consumers[i, r, x] = self.selects[i, r, x] = (groups, select)

    def read(self, what: str, target: tuple, off: int, lo: int, hi: int, segs: Sequence[Segment]) -> None:
        """Wire leaves ``[lo, hi)`` of a site's output, read into ``target``'s port from its leaf ``off``, from the writer
        rows ``segs``: ``("spread", ranges, port, P)``, the port's P leaves a unit laid unit after unit;
        ``("same", ranges, port, P)``, each unit's port reading them alike. A run the grains don't fit is listed unwired."""
        kind, ranges, port, P = target
        for slo, shi, w, tw, out, u0 in segs:
            l1, h1 = max(lo, slo), min(hi, shi)
            if l1 >= h1:
                continue
            dest, m, uw = off + l1 - lo, (h1 - l1) // w, u0 + (l1 - slo) // w
            if (l1 - slo) % w or (h1 - l1) % w:
                why = f"part of {w}-word rows of {tw}"
            elif P % w or dest % w:
                why = f"{w}-word rows of {tw} into its {P}-word {port} from word {dest}"
            elif kind == "spread":
                why = self.spread(ranges, port, P // w, dest // w, tw, out, uw, m)
            else:
                why = None
                for tr, ur, n in ranges:
                    if (have := self.cut(tr, port, P // w)) != P // w:
                        why = f"{w}-word rows of {tw}, and {tr}'s {port} is cut {have} ways"
                        break
                    self.wires.append(I.Wire(tr, port, tw, out, at=ur, radix=(n, m), base=uw, step=(0, 1),
                                             cut=(dest // w, m)))
            if why is not None:
                self.unwired.append(f"{what} reads leaves [{l1}, {h1}): {why}")


def sites_circuit(bindings: Sequence[TB.RequestBinding | None], name: str, ranks: Sequence[int] | None = None) -> Sites:
    """The window of the collective sites of ``bindings`` (by advice request, None for a request no rank ran), of every rank
    or of ``ranks`` alone (module docstring); refused (ValueError) by name where a site has no grain (`grain.lower`)."""
    live = [b for b in bindings if b is not None]
    if not live:
        raise ValueError(f"{name}: no request's binding, so no collective site")
    world = live[0].world
    rs = list(range(world)) if ranks is None else sorted(set(ranks))
    if not rs or not set(rs) <= set(range(world)) or any(b.world != world for b in live):
        raise ValueError(f"{name}: ranks {rs} of bindings of worlds {sorted({b.world for b in live})}")
    win = _Window(name, rs, world)
    for i, b in enumerate(bindings):
        if b is None:
            continue
        combines = [{s.combine[r][0]: j for j, s in enumerate(b.sites)} for r in range(world)]
        producing = [{n for s in b.sites for n, _, _ in s.own[r]} for r in range(world)]
        last: dict[tuple[int, int], list[int]] = {}
        for r in rs:
            at: dict[int, int] = {}
            for j, s in enumerate(b.sites):
                for x, *_ in s.readers[r] if s.readers else ():
                    if x not in combines[r] and x not in producing[r]:
                        at[x] = j
            for x, j in at.items():
                last.setdefault((j, r), []).append(x)
        lowered: dict[int, G.Lowering] = {}
        segs: dict[int, list[Segment]] = {}
        #: (rank, producer node) -> its runs' (site, first position, positions)
        mine: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for j, s in enumerate(b.sites):
            where = f"{name}: request {i}'s collective {s.k} [{s.lo}, {s.hi})"
            reads = [(lo, hi) for rd in s.readers for *_, lo, hi in rd]
            low = lowered[j] = G.lower(b.fns[0][s.combine[0][0]],
                                       [[(b.fns[p][n], lo, hi) for n, lo, hi in s.own[p]] for p in range(world)], reads, where)
            win.site(i, j, low)
            segs[j] = win.writers(i, j, low)
            for r in rs:
                q = 0
                for (n, _lo, _hi), (*_, m) in zip(s.own[r], low.producers[r]):
                    mine.setdefault((r, n), []).append((j, q, m))
                    q += m
                for x in sorted(last.get((j, r), ())):
                    win.consumer(i, j, r, x, b.fns[r][x], segs[j])
        for j, s in enumerate(b.sites):
            for r in rs:
                for x, a, off, lo, hi in s.readers[r] if s.readers else ():
                    xfn = b.fns[r][x]
                    port = xfn.signature()[0][a][0]
                    what = f"{name}: request {i}'s collective {s.k} [{s.lo}, {s.hi}): rank {r}'s node {x} ({xfn.id}) {port}"
                    if x in combines[r]:
                        if r != rs[0]:
                            continue
                        cs = win.units.get((i, combines[r][x], COMBINE, ALL), [])
                        if not cs or not all(win.words(t, port) for t, *_ in cs):
                            win.unwired.append(f"{what} reads leaves [{lo}, {hi}): no unit of collective "
                                               f"{combines[r][x]} reads them")
                            continue
                        target = ("same", cs, port, win.words(cs[0][0], port))
                    elif (r, x) in mine:
                        P = dict(I.layout(xfn).rows)[port]
                        for j3, q0, m3 in mine[r, x]:
                            for _q, _k, tp, up, n in _pieces(win.units[i, j3, PRODUCER, r], q0, m3):
                                if win.words(tp, port) != P:
                                    win.unwired.append(f"{what} reads leaves [{lo}, {hi}): its producer unit {tp} has no "
                                                       f"{P}-word {port}")
                                else:
                                    win.read(what, ("same", [(tp, up, n)], port, P), off, lo, hi, segs[j])
                        continue
                    elif (i, r, x) in win.selects:
                        groups, select = win.selects[i, r, x]
                        win.read(what, ("spread", [groups], "x", win.words(groups[0], "x")), off, lo, hi, segs[j])
                        if off == 0:
                            if (src := win.word(i, j, lowered[j], lo)) is None:
                                win.unwired.append(f"{what} reads leaf {lo} as its select's x0: no one-word unit computes it")
                            else:
                                win.wires.append(I.Wire(select[0], "x0", src[0], src[1], at=select[1], radix=(1,),
                                                        base=src[2], step=(1,)))
                        continue
                    else:
                        (unit,) = win.consumers[i, r, x]
                        target = ("spread", [unit], port, win.words(unit[0], port))
                    win.read(what, target, off, lo, hi, segs[j])
    templates = tuple((fn, win.count[tid]) for tid, fn in win.fns.items())
    circuit = I.Circuit(name, templates, wires=tuple(I.coalesce(win.wires)), cuts=win.cuts, order=tuple(win.order))
    return Sites(circuit, {k: tuple(v) for k, v in win.units.items()}, dict(win.consumers), tuple(win.order),
                 tuple(win.unwired))
