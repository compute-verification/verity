"""A tensor-parallel window's collective sites at node grain (`sites_circuit`): every producer of a partial and every combine of
each request's W rank Programs as units, and each site's binding (`program.tp_binding`) as wires, so that row p of every rank's
combine parts, its parts cut W ways, is rank p's producer's output row.

A wired row is committed under its writer's salt (`wiring.salts`), so a peer row and its producer's output are one public
string, and a peer row that isn't its producer's output is a wrong block of the window (`wiring.check`, and `check_drawn` on the
units a sample draws), never a free input: `interface.input_count` counts every peer row wired. Each template's units are
node-major and rank-minor, a site's W combines side by side and its producers likewise where they share a template, so one wire
covers every reader rank and every cut of a site: its reader-rank digit has writer step 0 and its cut digit writer step 1. A
rank-specific producer (an embedding shard, its START its rank's) is a template of its own, one wire a cut.

At node grain a unit is one node of a rank Program, so a combine reads one row per rank only where every rank's partial is one
node's whole output field: a collective per row. A collective over several rows (the route all-reduce of a prefill's rows,
`_Peers.whole`) reads each rank's rows as one segment that no one unit writes, and is refused by name, as is a partial that is
part of a field. With ``ranks``, the window of those ranks alone: their units and the wires between them, every other rank's
peer row a free input, as in today's per-rank windows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.program import tp_binding as TB

from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as I

COMBINE, PRODUCER = "combine", "producer"


@dataclass(frozen=True)
class Sites:
    """`sites_circuit`'s circuit and where each unit came from: ``units[(request, site, rank, role)]``, the unit's template
    (descriptor id) and its index among that template's units, ``role`` `COMBINE` or `PRODUCER`."""
    circuit: I.Circuit
    units: Mapping[tuple[int, int, int, str], tuple[str, int]]


def _field(fn: Any, lo: int, hi: int) -> int | None:
    """The output field of ``fn`` whose leaves are exactly ``[lo, hi)``, else None."""
    ret, at = fn.signature()[1], 0
    for f, ty in enumerate(x[1] if isinstance(x, tuple) else x for x in (getattr(ret, "fields", None) or (ret,))):
        if (at, at + ty.leaves) == (lo, hi):
            return f
        at += ty.leaves
    return None


def sites_circuit(bindings: Sequence[TB.RequestBinding | None], name: str, ranks: Sequence[int] | None = None) -> Sites:
    """The window of the collective sites of ``bindings`` (by advice request, None for a request no rank ran) at node grain, of
    every rank or of ``ranks`` alone (module docstring); refused (ValueError) by name where a rank's partial isn't one node's
    whole output field."""
    live = [b for b in bindings if b is not None]
    if not live:
        raise ValueError(f"{name}: no request's binding, so no collective site")
    world = live[0].world
    rs = list(range(world)) if ranks is None else sorted(set(ranks))
    if not rs or not set(rs) <= set(range(world)) or any(b.world != world for b in live):
        raise ValueError(f"{name}: ranks {rs} of bindings of worlds {sorted({b.world for b in live})}")
    fns: dict[str, Any] = {}
    count: dict[str, int] = {}
    units: dict[tuple[int, int, int, str], tuple[str, int]] = {}
    cuts: dict[str, dict[str, int]] = {}
    wires: list[I.Wire] = []

    def unit(fn: Any, key: tuple[int, int, int, str]) -> tuple[str, int]:
        tid = descriptor_id(fn)
        fns.setdefault(tid, fn)
        units[key] = (tid, count.get(tid, 0))
        count[tid] = units[key][1] + 1
        return units[key]

    for i, b in enumerate(bindings):
        for j, s in enumerate(b.sites if b is not None else ()):
            where = f"{name}: request {i}'s collective {s.k} [{s.lo}, {s.hi})"
            writers = {}
            for p in rs:
                if len(s.own[p]) != 1:
                    raise ValueError(f"{where}: rank {p}'s partial is {len(s.own[p])} nodes' rows {[n for n, _, _ in s.own[p]]}, a "
                                     "collective over several rows, whose segment per rank no one unit writes at node grain; "
                                     "compile it per row")
                node, lo, hi = s.own[p][0]
                fn = b.fns[p][node]
                if (f := _field(fn, lo, hi)) is None:
                    raise ValueError(f"{where}: rank {p}'s partial is leaves [{lo}, {hi}) of node {node} ({fn.id}), not one whole "
                                     "output field")
                writers[p] = (*unit(fn, (i, j, p, PRODUCER)), f)
            for r in rs:
                tid, _ = unit(b.fns[r][s.combine[r][0]], (i, j, r, COMBINE))
                cuts.setdefault(tid, {})[s.combine[r][1]] = world
            tid, at = units[i, j, rs[0], COMBINE]
            port, ws = s.combine[rs[0]][1], [writers[p] for p in rs]
            if len(rs) == world and len({(t, f) for t, _, f in ws}) == 1 and [u for _, u, _ in ws] == list(range(ws[0][1], ws[0][1] + world)):
                wires.append(I.Wire(tid, port, ws[0][0], f"o{ws[0][2]}", at=at, radix=(world, world), base=ws[0][1], step=(0, 1),
                                    cut=(0, world)))
            else:
                wires += [I.Wire(tid, port, t, f"o{f}", at=at, radix=(len(rs),), base=u, step=(0,), cut=(p, 1))
                          for p, (t, u, f) in zip(rs, ws)]
    templates = tuple((fn, count[tid]) for tid, fn in fns.items())
    return Sites(I.Circuit(name, templates, wires=tuple(I.coalesce(wires)), cuts=cuts), units)
