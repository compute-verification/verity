"""The binding of a tensor-parallel request's W rank Programs: for each collective, which rank's producer output each segment
of each rank's combine reads, derived from the rank Programs alone.

A rank Program reads what the other ranks sent at its k-th collective as the Inputs ``peers_<k·(W−1)+j'>``
(`predict.common._Peers`), j' rank j's place among the other ranks in rank order, and each collective node reads them in one
operand of W segments in rank order: at its own rank a producer node's output, its partial, and at rank j a slice of one of
those Inputs. A site is one collective at every rank, keyed by k and the slice of the peer Inputs its combine reads
(`rank_sites`). `pair` pairs each rank's segment j with rank j's own partial at the same site: by k and the slice, never by node
order, since dead-free compilation drops a dead collective with its producer at every rank. What doesn't pair is refused by
name: a collective live at some ranks only, ranks that combine a site with different Definitions, a peer segment that isn't
one slice of the Input its rank sends, a collective read inside a batch or a scan.

Each site also records, per rank, who reads its combine's output (``Site.readers``): every call reading leaves of it, which
the window wires from the combine units that compute those leaves: at a collective site too, every value a unit reads is the
output of the unit that computed it, never a free witness. A combine's output read inside a batch or a scan is refused by name.

The prover and the verifier derive the binding from the rank Programs of record alone, so it is part of the registered circuit
and never supplied; `pipeline.tp.wires` lowers it to a window's wires."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from verity.circuits.refs import runs

BINDING_TAG = "verity-vllm/tp-binding/v1"
PEERS = "peers_"

#: leaves ``[lo, hi)`` of node ``node``'s output
Run = tuple[int, int, int]
#: a site's key: the peer Inputs' collective index k and the slice ``[lo, hi)`` of them its combine reads
Key = tuple[int, int, int]
#: a read of a combine's output: the reading call node, its argument, the leaf of that argument the read fills from, and the
#: leaves ``[lo, hi)`` of the combine's output it reads
Reader = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class RankSites:
    """One rank's collectives in one request Program (`rank_sites`): by site, its combine node, the parameter its parts are,
    and its own partial's runs, and the reads of its combine's output (``readers``); ``fns`` the combine, producer and reader
    nodes' Definitions, by node."""
    world: int
    rank: int
    digest: str
    sites: Mapping[Key, tuple[int, str, tuple[Run, ...]]]
    fns: Mapping[int, Any] = field(compare=False, repr=False)
    readers: Mapping[Key, tuple[Reader, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Site:
    """One collective of a request at every rank: ``k`` and ``[lo, hi)`` its key, and per rank its combine (``combine[r]``: the
    node and its parts parameter), its own partial (``own[r]``: runs of producer outputs, one run of one node's whole
    output for a collective per row) and the reads of its combine's output (``readers[r]``, in node order). Rank r's
    segment p is rank p's own partial."""
    k: int
    lo: int
    hi: int
    combine: tuple[tuple[int, str], ...]
    own: tuple[tuple[Run, ...], ...]
    readers: tuple[tuple[Reader, ...], ...] = ()


@dataclass(frozen=True)
class RequestBinding:
    """A request's binding: ``world`` ranks, each one's request Program digest (``ranks``), its sites by key; ``fns`` each rank's
    combine and producer Definitions, by node (`RankSites.fns`)."""
    world: int
    ranks: tuple[str, ...]
    sites: tuple[Site, ...]
    fns: Mapping[int, Mapping[int, Any]] = field(compare=False, repr=False, default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"world": self.world, "ranks": list(self.ranks),
                "sites": [[s.k, s.lo, s.hi, [list(c) for c in s.combine], [[list(r) for r in o] for o in s.own],
                           [[list(r) for r in rs] for rs in s.readers]] for s in self.sites]}

    @cached_property
    def digest(self) -> str:
        """SHA-512 of `record`'s canonical JSON under `BINDING_TAG`."""
        body = json.dumps(self.record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha512(f"{BINDING_TAG}\n{body}".encode()).hexdigest()


def _segments(rs: list[tuple[str, int, int, int]], world: int, what: str) -> list[list[tuple[str, int, int, int]]]:
    """Runs ``rs`` of an operand cut into ``world`` equal segments, each its runs."""
    total = sum(hi - lo for *_, lo, hi in rs)
    if total % world:
        raise ValueError(f"{what}: its parts are {total} leaves, not {world} equal segments")
    n, out, at = total // world, [[] for _ in range(world)], 0
    for space, idx, lo, hi in rs:
        while lo < hi:
            seg, off = divmod(at, n)
            take = min(hi - lo, n - off)
            cur = out[seg]
            if cur and cur[-1][:2] == (space, idx) and cur[-1][3] == lo:
                cur[-1] = (space, idx, cur[-1][2], lo + take)
            else:
                cur.append((space, idx, lo, lo + take))
            lo, at = lo + take, at + take
    return out


def rank_sites(body: Any, world: int, rank: int, digest: str = "") -> RankSites:
    """Rank ``rank``'s collectives in request Program body ``body`` of a ``world``-rank TP request: every node reading a peer
    Input, each a call reading them in one operand of ``world`` segments in rank order, its own segment node outputs and every
    other segment j one slice of rank j's Input of one collective (module docstring); refused (ValueError) by name otherwise."""
    where = f"rank {rank}'s Program {digest[:16]}"
    if world < 2 or not 0 <= rank < world:
        raise ValueError(f"{where}: rank {rank} of a world of {world}, not a tensor-parallel rank")
    peer = {i: int(name[len(PEERS):]) for i, (name, _) in enumerate(body.params) if name.startswith(PEERS)}
    sites: dict[Key, tuple[int, str, tuple[Run, ...]]] = {}
    fns: dict[int, Any] = {}
    for at, node in enumerate(body.nodes):
        read = [a for a, arg in enumerate(node.args) if any(s == "p" and i in peer for s, i, _, _ in runs(arg.refs))]
        if not read:
            continue
        what = f"{where}: node {at} ({node.fn.id})"
        if node.form != "call" or len(read) != 1:
            raise ValueError(f"{what} reads peer Inputs in {len(read)} operands of a {node.form} node: a collective is a call reading "
                             "them in one operand")
        segs = _segments(runs(node.args[read[0]].refs), world, what)
        if any(s != "n" for s, *_ in segs[rank]):
            raise ValueError(f"{what}: its segment {rank}, its own partial, reads {segs[rank]}, not node outputs")
        keys = set()
        for s, seg in enumerate(segs):
            if s == rank:
                continue
            if len(seg) != 1 or seg[0][0] != "p" or seg[0][1] not in peer:
                raise ValueError(f"{what}: its segment {s} reads {seg}, not one slice of one peer Input")
            m = peer[seg[0][1]]
            if m % (world - 1) != (s if s < rank else s - 1):
                raise ValueError(f"{what}: its segment {s} reads {PEERS}{m}, another rank's Input than rank {s}'s")
            keys.add((m // (world - 1), seg[0][2], seg[0][3]))
        if len(keys) != 1:
            raise ValueError(f"{what}: its segments read collectives and slices {sorted(keys)}, not one")
        key = keys.pop()
        if key in sites:
            raise ValueError(f"{what}: reads collective {key[0]}'s [{key[1]}, {key[2]}), which node {sites[key][0]} reads too")
        sites[key] = (at, node.fn.signature()[0][read[0]][0], tuple((i, lo, hi) for _, i, lo, hi in segs[rank]))
        fns[at] = node.fn
        fns.update((i, body.nodes[i].fn) for _, i, _, _ in segs[rank])
    combines = {node: key for key, (node, _, _) in sites.items()}
    readers: dict[Key, list[Reader]] = {}
    for at, node in enumerate(body.nodes):
        for a, arg in enumerate(node.args):
            off = 0
            for s, i, lo, hi in runs(arg.refs):
                if s == "n" and i in combines:
                    if node.form != "call":
                        raise ValueError(f"{where}: node {at} ({node.fn.id}) reads collective node {i}'s output in a "
                                         f"{node.form} node: a combine's output is read by calls")
                    readers.setdefault(combines[i], []).append((at, a, off, lo, hi))
                    fns[at] = node.fn
                off += hi - lo
    return RankSites(world, rank, digest, sites, fns, {k: tuple(v) for k, v in readers.items()})


def pair(ranks: Sequence[RankSites]) -> RequestBinding:
    """The binding of one request's W rank Programs' collectives (`rank_sites` of each, in rank order); refused (ValueError) by
    name unless every site is live at every rank and combined there with one Definition."""
    world = len(ranks)
    if world < 2 or [r.rank for r in ranks] != list(range(world)) or any(r.world != world for r in ranks):
        raise ValueError(f"ranks {[(r.rank, r.world) for r in ranks]}: not each rank of one world once, in rank order")
    keys = set(ranks[0].sites)
    for r in ranks[1:]:
        if set(r.sites) != keys:
            k, lo, hi = min(set(r.sites) ^ keys)
            live = [x.rank for x in ranks if (k, lo, hi) in x.sites]
            raise ValueError(f"collective {k}'s [{lo}, {hi}) is read at ranks {live} only: a collective live at some ranks, dead at "
                             "others, pairs with nothing")
    sites = []
    for key in sorted(keys):
        ids = [x.fns[x.sites[key][0]].id for x in ranks]
        if len(set(ids)) != 1:
            r = next(r for r, i in enumerate(ids) if i != ids[0])
            raise ValueError(f"collective {key[0]}'s [{key[1]}, {key[2]}): rank 0 combines it with {ids[0]}, rank {r} with {ids[r]}")
        sites.append(Site(*key, tuple(x.sites[key][:2] for x in ranks), tuple(x.sites[key][2] for x in ranks),
                          tuple(x.readers.get(key, ()) for x in ranks)))
    return RequestBinding(world, tuple(x.digest for x in ranks), tuple(sites), {x.rank: x.fns for x in ranks})


def derive(bodies: Sequence[Any], digests: Sequence[str] | None = None) -> RequestBinding:
    """The binding of one request's rank Program bodies, in rank order (`rank_sites` of each, `pair`)."""
    ds = list(digests) if digests is not None else [""] * len(bodies)
    return pair([rank_sites(b, len(bodies), r, d) for r, (b, d) in enumerate(zip(bodies, ds, strict=True))])
