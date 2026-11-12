"""The tensor-parallel compile: a node's advice compiled at every rank of φ.tp's world and composed into one TP workload
(`tp_workload.TpWorkload`).

`compile_steps` compiles one rank, the one φ.tp names. The advice is the node's, the same at every rank, so `compile_tp` compiles
it at each rank in turn (`at_rank`) and keeps of each only what the composition reads: its workload's digest, its rows and, per
request Program, its collectives (`tp_binding.rank_sites`), so no two ranks' Programs are held at once. It then pairs each
request's W rank Programs (`tp_binding.pair`) and composes the rank workloads with those bindings."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from verity_vllm.program import tp_binding as TB
from verity_vllm.program.tp_workload import TpWorkload

from .phi import Phi
from .steps import Registered, compile_steps


@dataclass(frozen=True)
class TpCompiled:
    """`compile_tp`'s result: per rank its workload's digest (``ranks[r]``); per rank, then per advice request, that rank's
    request Program's digest (``requests[r][i]``, None for a refused request); per advice request its binding (``bindings[i]``,
    None where no rank ran it); the composition."""
    ranks: tuple[str, ...]
    requests: tuple[tuple[str | None, ...], ...]
    bindings: tuple[TB.RequestBinding | None, ...]
    workload: TpWorkload


def at_rank(reg: Registered, rank: int) -> Registered:
    """``reg`` at rank ``rank`` of its φ.tp's world."""
    return replace(reg, phi=Phi.from_json({**reg.phi.to_json(), "tp": {**reg.phi.tp, "rank": int(rank)}}))


def compile_tp(reg: Registered, advice: Any, bodies: dict, **kw: Any) -> TpCompiled:
    """The advice compiled at every rank of ``reg``'s φ.tp world (`compile_steps` at each, ``kw`` its options) and composed;
    refused (ValueError) at world 1, when two ranks compile the advice to different rows or refuse different requests, and
    where a request's rank Programs don't pair (`tp_binding.pair`)."""
    world = int(reg.phi.tp.get("world", 1))
    if world < 2:
        raise ValueError(f"φ.tp's world is {world}: a TP compile composes two or more ranks")
    digests: list[str] = []
    units: list[tuple[str | None, ...]] = []
    sites: list[dict[str, TB.RankSites]] = []
    rows = None
    for r in range(world):
        c = compile_steps(at_rank(reg, r), advice, bodies, **kw)
        got = [[(x.request, x.start, x.n, x.samples) for x in step] for step in c.steps]
        mine = tuple(None if u is None else u.digest for u in c.requests)
        if rows is not None and got != rows:
            raise ValueError(f"rank {r} compiles the advice to other rows than rank 0")
        if units and [d is None for d in mine] != [d is None for d in units[0]]:
            raise ValueError(f"rank {r} refuses requests {[i for i, d in enumerate(mine) if d is None]}, rank 0 "
                             f"{[i for i, d in enumerate(units[0]) if d is None]}")
        rows = got
        digests.append(c.workload.digest)
        units.append(mine)
        sites.append({d: TB.rank_sites(c.programs[d].fn.body, world, r, d) for d in set(mine) - {None}})
        del c
    made: dict[tuple[str, ...], TB.RequestBinding] = {}
    bindings: list[TB.RequestBinding | None] = []
    for i in range(len(units[0])):
        key = tuple(u[i] for u in units)
        if key[0] is None:
            bindings.append(None)
            continue
        if key not in made:
            made[key] = TB.pair([sites[r][d] for r, d in enumerate(key)])
        bindings.append(made[key])
    return TpCompiled(tuple(digests), tuple(units), tuple(bindings), TpWorkload(digests, bindings))
