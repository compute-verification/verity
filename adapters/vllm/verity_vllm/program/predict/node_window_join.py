"""The W ranks' windows of one request (`node_window.node_window`, each at its own TP rank) joined into one circuit (`join`):
each rank's units after the rank before's in every template, its registration renamed to its own, its ingress at its
units, and each of its peer inputs (``peers_*``) wired from what the writer rank's own window wires into the same reader
units and rows. What the joined record states of its ranks and its free rows, by proofs' design for it (PRF-618), is
`ranks_of` and `free_rows`; `interface.Circuit` holds neither field yet."""
from __future__ import annotations

import bisect
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from verity.circuits.codec import canonical_json
from verity.commitments.windows import interface as IF

from .build import Unexpressible
from .node_window import _GRAIN, NAME, Free, Lowering, _ingress_census, _peer, node_window
from .window_builder import WEIGHTS, Grain, inference


@dataclass(frozen=True)
class Rank:
    """Rank ``rank``'s window as `join` reads it, without the frame and its Program: its circuit, its free inputs, its
    units' runs (`_Builder.spans`), its steps, its census."""
    rank: int
    circuit: IF.Circuit
    free: tuple[Free, ...]
    spans: Mapping[str, tuple[tuple[int, int, int, int], ...]]
    steps: int
    census: Mapping[str, Any]


def rank_window(rank: int, fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN) -> Rank:
    """Rank ``rank``'s Program ``fn`` lowered into its window (`node_window`), the frame dropped."""
    c, w = node_window(fn, families, grain)
    return Rank(rank, c, tuple(w.free), {t: tuple(s) for t, s in w.b.spans.items()}, w.steps, w.census(c))


@dataclass(frozen=True)
class Joined:
    """The ranks' windows as one (`join`): ``circuit``; ``first[r][tid]``, rank r's first unit of each template; ``free``,
    the inputs still free, at the circuit's units, each ``name`` its rank's (``r<rank>/<name>``); ``wired``, the peer
    inputs `join` wired, as `Free`s of their reader rank at its own units; ``unwired``, each peer input left free, and why."""
    circuit: IF.Circuit
    first: Mapping[int, Mapping[str, int]]
    free: tuple[Free, ...]
    wired: tuple[tuple[int, Free], ...]
    unwired: tuple[str, ...]

    def census(self) -> dict[str, Any]:
        """The circuit's digest, units, templates and wire patterns; its free input ports by kind (ports, unit ports and
        words, as `NodeWindow.census`), and by template prefix, port and kind; its ingress; the peer inputs wired; the
        unwired."""
        c = self.circuit
        lay = {tid: dict(lo.rows) for tid, (lo, _n) in c.layouts.items()}
        free: dict[str, Counter[str]] = {}
        for f in self.free:
            k = free.setdefault("peers" if f.kind == "peer" else f.family, Counter())
            k["unit_ports"] += f.count
            k["words"] += f.count * lay[f.tid][f.port] * f.rows[1] // f.rows[2]
            k["ports"] += 1
        return {"digest": hashlib.sha512(IF.circuit_bytes(c.description())).hexdigest(), "units": sum(c.units.values()),
                "templates": len(c.ids), "wire_patterns": len(c.wires),
                "free": {k: dict(v) for k, v in sorted(free.items())},
                "free_ports": dict(sorted(Counter(f"{f.tid.split('{')[0]}/{f.port}: {f.kind} {f.family}"
                                                  for f in self.free).items())),
                "ingress": _ingress_census(c.ingress, lay),
                "wired": {"ports": len(self.wired), "unit_ports": sum(f.count for _r, f in self.wired)},
                "unwired": len(self.unwired)}


def _slice(radix: tuple[int, ...], step: tuple[int, ...], base: int, a: int, b: int) -> list[tuple[tuple[int, ...], int, int]]:
    """Indices [a, b) of a pattern of ``radix``, ``step`` and ``base`` (`interface._Pattern`) as patterns of the same
    digits' steps: each ``(radix, base, first index)``, the first digit's run cut where [a, b) cuts it."""
    n = math.prod(radix)
    if a >= b:
        return []
    if (a, b) == (0, n):
        return [(radix, base, 0)]
    M = math.prod(radix[1:])
    lo, hi = -(-a // M), b // M
    if lo > hi:
        d = a // M
        return [(r, s, i + d * M) for r, s, i in _slice(radix[1:], step[1:], base + d * step[0], a - d * M, b - d * M)]
    out = []
    if a < lo * M:
        d = lo - 1
        out += [(r, s, i + d * M) for r, s, i in _slice(radix[1:], step[1:], base + d * step[0], a - d * M, M)]
    if hi > lo:
        out.append(((hi - lo, *radix[1:]), base + lo * step[0], lo * M))
    if b > hi * M:
        out += [(r, s, i + hi * M) for r, s, i in _slice(radix[1:], step[1:], base + hi * step[0], 0, b - hi * M)]
    return out


def _within(w: IF.Wire, lo: int, hi: int, r0: int, r1: int) -> list[IF.Wire]:
    """``w``'s reads of reader units [lo, hi) as wires of their own, where its rows are among [r0, r1); none where they
    are none of them, and refused (`Unexpressible`) where its rows are some of them only."""
    c0, C = w.cut
    if c0 + C <= r0 or c0 >= r1:
        return []
    if c0 < r0 or c0 + C > r1:
        raise Unexpressible(f"wire {w.doc()}: rows [{c0}, {c0 + C}) of which [{r0}, {r1}) are some only")
    a, b = (max(lo, w.at) - w.at) * C, (min(hi, w.at + w.reader_units) - w.at) * C
    return [IF.Wire(w.reader, w.port, w.writer, w.out, at=w.at + i // C, radix=r, base=s, step=w.step[len(w.step) - len(r):],
                    cut=w.cut) for r, s, i in _slice(w.radix, w.step, w.base, a, b)]


def _renamed(reg: IF.Registered | None, rank: int, first: Mapping[str, int]) -> IF.Registered:
    """Rank ``rank``'s registration with every parameter and table named ``r<rank>/<name>`` and every reader at its
    units in the joined circuit (``first``); refused where it has histories or values, which no rank window declares."""
    reg = reg or IF.Registered({})
    if reg.histories or reg.values:
        raise Unexpressible(f"rank {rank}: a window's registration with histories or values, which `join` doesn't rename")
    p = f"r{rank}/"
    return IF.Registered(
        {p + k: v for k, v in reg.params.items()},
        tuple(IF.Table(p + t.name, p + t.param, t.offset, t.words, t.rows) for t in reg.tables),
        tuple(IF.Read(r.reader, r.port, p + r.param, at=r.at + first[r.reader], radix=r.radix, base=r.base, step=r.step,
                      cut=r.cut) for r in reg.reads),
        tuple(IF.TableRead(r.reader, r.port, p + r.table, r.of, r.at + first[r.reader], r.units) for r in reg.roots),
        tuple(IF.TableOpening(o.reader, p + o.table, o.key, o.ports, o.verdict, o.at + first[o.reader], o.units)
              for o in reg.openings),
        formats={p + k: v for k, v in reg.formats.items()})


def join(ranks: Sequence[Rank], name: str = f"{NAME}/ranks") -> Joined:
    """The windows of the W ranks of one request (rank r's at ``ranks[r]``) as one circuit. Rank r's units follow rank
    r - 1's in each template, its registration renamed to its own (`_renamed`), its ingress at its units. Each peer input of rank r (a ``peers_*``
    `Free`: input ``peers_<k(W-1)+j'>`` is rank j's partial at its k-th collective, j' rank j's place among the others) is
    wired from what rank j's own window wires into the same reader units and rows, its own partial there: rank j's wires
    cut to them (`_within`). Where they don't cover the rows, the input stays free, listed in ``unwired``. The circuit's
    order is made anew over every rank's runs (`interface.ordered`), its blocks from every rank's spans."""
    W = len(ranks)
    if [x.rank for x in ranks] != list(range(W)) or len({x.steps for x in ranks}) != 1:
        raise Unexpressible(f"{name}: ranks {[x.rank for x in ranks]} at steps {sorted({x.steps for x in ranks})}; the "
                            "ranks 0 .. W-1 of one request")
    fns: dict[str, Any] = {}
    total: Counter[str] = Counter()
    first: dict[int, dict[str, int]] = {}
    cuts: dict[str, dict[str, int]] = {}
    for x in ranks:
        first[x.rank] = {}
        for (fn, n), tid in zip(x.circuit.templates, x.circuit.ids):
            fns.setdefault(tid, fn)
            first[x.rank][tid] = total[tid]
            total[tid] += n
        for tid, kv in x.circuit.cuts.items():
            for port, k in kv.items():
                if cuts.setdefault(tid, {}).setdefault(port, k) != k:
                    raise Unexpressible(f"{name}: {tid} {port} cut {k} ways at rank {x.rank} and {cuts[tid][port]} before")
    #: each rank's wires by reader port, sorted by first unit, with their first units and the most reader units of one
    by: dict[int, dict[tuple[str, str], tuple[list[IF.Wire], list[int], int]]] = {}
    for x in ranks:
        idx: dict[tuple[str, str], list[IF.Wire]] = {}
        for w in x.circuit.wires:
            idx.setdefault((w.reader, w.port), []).append(w)
        by[x.rank] = {key: (ws, [w.at for w in ws], max(w.reader_units for w in ws))
                      for key, ws in ((key, sorted(ws, key=lambda w: w.at)) for key, ws in idx.items())}

    def shift(w: IF.Wire, r: int, j: int) -> IF.Wire:
        return IF.Wire(w.reader, w.port, w.writer, w.out, at=w.at + first[r][w.reader], radix=w.radix,
                       base=w.base + first[j][w.writer], step=w.step, cut=w.cut)

    wires: list[IF.Wire] = [shift(w, x.rank, x.rank) for x in ranks for w in x.circuit.wires]
    added: list[IF.Wire] = []
    free: list[Free] = []
    wired: list[tuple[int, Free]] = []
    unwired: list[str] = []
    for x in ranks:
        r = x.rank
        peers: dict[tuple[str, str, int, int, tuple[int, int, int]], list[Free]] = {}
        for f in x.free:
            if f.kind == "peer":
                peers.setdefault((f.tid, f.port, f.at, f.count, f.rows), []).append(f)
            else:
                free.append(replace(f, at=f.at + first[r][f.tid], name=f"r{r}/{f.name}"))
        for (tid, port, at, count, (i, n, k)), fs in peers.items():
            srcs = sorted({q + (q >= r) for f in fs for q in [_peer(f.name) % (W - 1)]})
            got, why = [], None
            try:
                for j in srcs:
                    ws, ats, most = by[j].get((tid, port), ([], [], 0))
                    for w in ws[bisect.bisect_left(ats, at - most + 1):bisect.bisect_left(ats, at + count)]:
                        if w.at + w.reader_units > at:
                            got += [(j, s) for s in _within(w, at, at + count, i, i + n)]
            except Unexpressible as e:
                why = str(e)
            covered = sum(s.reader_units * s.cut[1] for _j, s in got)
            if why is None and covered != count * n:
                why = f"ranks {srcs}' wires cover {covered} of its {count * n} unit rows"
            if why is not None:
                unwired.append(f"rank {r}: {tid} {port} units [{at}, {at + count}) rows [{i}, {i + n}) of {k} "
                               f"({', '.join(f.name for f in fs)}): {why}")
                free += [replace(f, at=f.at + first[r][f.tid], name=f"r{r}/{f.name}") for f in fs]
                continue
            added += [shift(s, r, j) for j, s in got]
            wired += [(r, f) for f in fs]
    reg = [_renamed(x.circuit.registered, x.rank, first[x.rank]) for x in ranks]
    registered = IF.Registered(
        {k: v for g in reg for k, v in g.params.items()}, tuple(t for g in reg for t in g.tables),
        tuple(r for g in reg for r in g.reads), tuple(r for g in reg for r in g.roots),
        tuple(o for g in reg for o in g.openings), formats={k: v for g in reg for k, v in g.formats.items()})
    spans: dict[str, list[tuple[int, int, int, int]]] = {}
    for x in ranks:
        for tid, ss in x.spans.items():
            spans.setdefault(tid, []).extend((at + first[x.rank][tid], n, slot, step) for at, n, slot, step in ss)
    starts = {tid: [at for at, *_ in ss] for tid, ss in spans.items()}
    wires += IF.coalesce(added)
    ingress = tuple(replace(g, at=g.at + first[x.rank][g.reader]) for x in ranks for g in x.circuit.ingress)
    circuit = IF.Circuit(name, tuple((fns[tid], total[tid]) for tid in fns), wires=tuple(wires), cuts=cuts,
                         registered=registered, ingress=ingress,
                         external=tuple(dict.fromkeys(e for x in ranks for e in x.circuit.external)), blocks=IF.Blocks(1, ranks[0].steps, tuple(inference(spans, 1))),
                         order=IF.ordered(dict(total), wires, starts))
    return Joined(circuit, first, tuple(free), tuple(wired), tuple(unwired))


def ranks_of(ranks: Sequence[Rank], joined: Joined, cap: int) -> dict[str, Any]:
    """The joined record's ``ranks`` (PRF-618): the prefix ``r<rank>/`` once, and per rank in rank order its ``window``
    (SHA-512 of the canonical JSON of its record less its run fields, `interface.descriptor_record` at ``cap``), its
    ``units`` per template and ``wires``, how many of the joined circuit's wires are its own. `join` puts each rank's own
    wires in turn ahead of the ones it adds across ranks."""
    if [x.rank for x in ranks] != list(range(len(ranks))) or sum(len(x.circuit.wires) for x in ranks) > len(joined.circuit.wires):
        raise ValueError(f"ranks {[x.rank for x in ranks]}: not the ranks 0 .. W-1 that `join` joined into {joined.circuit.name}")
    return {"prefix": "r<rank>/",
            "of": [{"window": hashlib.sha512(canonical_json(IF.descriptor_record(x.circuit, cap))).hexdigest(),
                    "units": dict(x.circuit.units), "wires": len(x.circuit.wires)} for x in ranks]}


def free_rows(joined: Joined) -> list[dict[str, Any]]:
    """The joined record's ``free`` (PRF-618): each reader row no declaration fills (``joined.free``), its reader units,
    its rows ``[first, first + rows)`` of the port cut ``of`` ways, and its kind: ``peer`` (another rank's partial) or
    ``weights`` (the weights record, read by no registered read); ``table`` and ``advice`` are kinds no window frees yet.
    Refused (`Unexpressible`) where a free row is of no such kind: an unlowered node's, or another input's."""
    out = []
    for f in joined.free:
        kind = "peer" if f.kind == "peer" else "weights" if (f.kind, f.name.split("/", 1)[-1]) == ("param", WEIGHTS) else None
        if kind is None:
            raise Unexpressible(f"{f.tid} {f.port} units [{f.at}, {f.at + f.count}): a free row of {f.kind} {f.name}, which "
                                "the record's free list has no kind for")
        first, rows, of = f.rows
        out.append({"reader": f.tid, "port": f.port, "at": f.at, "units": f.count, "rows": [first, rows], "of": of,
                    "kind": kind})
    return out


def cover_weights(joined: Joined) -> Joined:
    """``joined`` with every ``BatchedGemm_v1`` ``w`` row that reads a rank's ``/bnk`` entry (`node_window_sparse.entry`:
    an absorbed weight, computed from registered tensors and registered as no GPU registers it) covered by the registered
    weights it is computed from, so that no such row is left for `free_rows`."""
    raise NotImplementedError("circuits/request-built-as-proof-blocks: CIR-287")


__all__ = ["Joined", "Rank", "cover_weights", "free_rows", "join", "rank_window", "ranks_of"]
