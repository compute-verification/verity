"""β(C): the min-cut bound on the stacked share of a circuit's dynamic memory (workstream S4 of
note:memory-accounting/20261026T0040Z-draft-stacking-safe-properties-program, whose frame this computes).

A stacker holds S_stack bits of other blocks' encodings in the dynamic memory a circuit writes (KV caches, latents, linear
attention's states). Take any cut (U, D) of the circuit's graph over a window, with every dynamic-memory write on D's side.
Everything D writes is a function of the bits that cross into it, b(U→D), and of D's own work, W(D). Under advice-bounded
incompressibility (an encoding of |c| bits takes about E units of work given much less than |c| bits of advice, the family
PoUS rests on), D's writes hold at most b(U→D) + |c|·W(D)/E bits of encodings, so

    S_stack ≤ min over (U, D) [ b(U→D) + |c|·W(D)/E ] + R_in·L,      β(C) = S_stack / M_dyn,

with M_dyn the bits of the writes. The cut D = {the writes} gives β ≤ 1 up to the input term.

The graph (`Graph`):

- **Nodes.** A node is a root Call of a Program (`from_program`), or a group of Calls (`coarsen`). Its work is its property/v1
  units (`verity.ml.circuits.stacking`): only gates with an operand that depends on a run input count. A node with a gate
  property/v1 refuses (unlisted, or a word read at another width) has unknown work and never sits in D, since charging it
  less would lower the bound below what it bounds.
- **Values.** A value is a run of a node's outputs (census/v1's values), with its bits and its readers. A value that no run
  input reaches carries 0 bits: registered words and constants are fixed before the seed, as in property/v1. A selection's
  candidates count as its reads.
- **Writes.** The dynamic-memory writes are the values a later step reads, so they are held across a step boundary (KV,
  read by every later step's attention). They are tied to t. Their reads are edges out of D and never cost anything,
  which is right: what U computes from D's values reaches D again only across the cut.
- **Run inputs.** The frame charges them as R_in·L (``inputs="rate"``, the default: their values cost nothing in the cut).
  ``inputs="window"`` instead holds the input nodes in U and charges the bits the window actually reads in the cut, with
  no R_in·L term.

**The min-cut** (`bound`). The flow network has an edge s→v of capacity |c|·w(v)/E for each node. Each value is a hyperedge
node: an edge producer→value at its bits, and value→reader edges of infinite capacity, so a value read across the cut is
paid once however many readers D holds. Each write has a value→t edge of infinite capacity. One max-flow (`max_flow`,
Dinic's, on exact integers: every capacity is scaled by the capacities' common denominator) gives the min, and the
residual graph gives the cut. `bound` recomputes the cut's cost from its two sides and holds it to the flow.

**Coarse graining** (`coarsen`). Nodes with one key become one node with their work summed. A value keeps its producer's
group and its readers' other groups, and values internal to a group drop out. Every cut of the coarse graph is a cut of the
fine graph at the same cost, so the coarse min is never lower. A coarse graph (layers or blocks as single nodes, with
aggregate work and boundary widths) gives a valid upper bound that is looser.

What property/v1 refuses but still tallies is charged as tallied and reported (`from_program`'s info): a gate reading more
than B operand bits, and an s8 core step whose window isn't proven. An encode built from such gates may cost less than E
of them, so a Program with either has no property/v1 verdict, and its β reads E in the basis property/v1 counts.

    python beta.py DESCRIPTOR.json[.gz] --out beta.json [--E 24504 --E 364497] [--L 600 --L 3600] [--m-dyn-node BYTES] \\
        [--chunk N]...                      # a Build's request Program: coarse bounds (by step, N Calls at a time), then the fine one
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from numbers import Rational
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

#: |c|: the bits of one encoded word
WORD_BITS = 1024
#: E, the property/v1 units one encode of a 1024-bit word costs: proved and conjectured
#: (note:memory-accounting/20261018T0651Z-report-stacked-share-int8)
E_PROVED = 24_504
E_CONJECTURED = 364_497
#: R_in's default, bytes a second (`stacking.R_IN`)
R_IN = 100_000
INPUTS = ("rate", "window")


def exact(x: Any, what: str) -> Fraction:
    """`x` as the rational it names: an int, a Fraction or a str such as '1/10'. A float is refused: `Fraction` reads it at its
    binary value."""
    if isinstance(x, float):
        raise ValueError(f"{what} is the float {x!r}, read at its binary value: give it exactly, as Fraction({str(x)!r})")
    if isinstance(x, bool) or not isinstance(x, (Rational, str)):
        raise ValueError(f"{what} is an int, a Fraction or a str such as '1/10', not {x!r}")
    return Fraction(x)


@dataclass(frozen=True)
class Value:
    """A run of `producer`'s outputs: its bits, the nodes that read it, whether a run input reaches it (else it carries 0 bits
    into the cut), whether it is a dynamic-memory write (tied to t), and whether it is a run input's (charged by R_in·L
    under ``inputs="rate"``)."""

    producer: int
    bits: int
    readers: tuple[int, ...] = ()
    dynamic: bool = True
    write: bool = False
    input: bool = False


@dataclass
class Graph:
    """Nodes with their work (property/v1 units, an int or a Fraction; None: unknown, never in D) and names; the values between
    them; the run-input nodes; and the nodes held on U's side (`fixed`: outside the window)."""

    work: list[Any]
    values: list[Value]
    names: list[str]
    inputs: frozenset[int] = frozenset()
    fixed: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        n = len(self.work)
        if len(self.names) != n:
            raise ValueError(f"{len(self.names)} names for {n} nodes")
        for w in self.work:
            if w is not None and (isinstance(w, (bool, float)) or not isinstance(w, Rational) or w < 0):
                raise ValueError(f"a node's work is a non-negative int or Fraction, or None, not {w!r}")
        for x in self.values:
            if not 0 <= x.producer < n or any(not 0 <= r < n for r in x.readers):
                raise ValueError(f"a value names a node outside the graph's {n}: {x}")
            if isinstance(x.bits, bool) or not isinstance(x.bits, int) or x.bits < 0:
                raise ValueError(f"a value's bits are an int >= 0, not {x.bits!r}")
        if any(not 0 <= v < n for v in self.inputs | self.fixed):
            raise ValueError(f"inputs and fixed name nodes of the graph's {n}")

    @property
    def m_dyn(self) -> int:
        """M_dyn: the bits of the writes."""
        return sum(x.bits for x in self.values if x.write)


def max_flow(n: int, edges: Sequence[tuple[int, int, int]], s: int, t: int,
             progress: Callable[[int, int], None] | None = None) -> tuple[int, list[bool]]:
    """The max flow from `s` to `t` over `edges` (u, v, capacity, an int >= 0) on nodes 0..n-1 (Dinic's), and which nodes the
    residual graph reaches from `s`: the min cut's source side. ``progress(phase, flow)`` is called after each phase."""
    to: list[int] = []
    cap: list[int] = []
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v, c in edges:
        if c < 0:
            raise ValueError(f"a capacity is an int >= 0, not {c!r}")
        adj[u].append(len(to))
        to.append(v)
        cap.append(c)
        adj[v].append(len(to))
        to.append(u)
        cap.append(0)
    flow = phase = 0
    while True:
        if progress is not None and phase:
            progress(phase, flow)
        phase += 1
        level = [-1] * n
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for e in adj[u]:
                if cap[e] > 0 and level[to[e]] < 0:
                    level[to[e]] = level[u] + 1
                    q.append(to[e])
        if level[t] < 0:
            break
        it = [0] * n
        path: list[int] = []
        u = s
        while True:
            if u == t:
                f = min(cap[e] for e in path)
                for e in path:
                    cap[e] -= f
                    cap[e ^ 1] += f
                flow += f
                cut = next(i for i, e in enumerate(path) if cap[e] == 0)
                u = to[path[cut] ^ 1]
                del path[cut:]
                continue
            lst = adj[u]
            while it[u] < len(lst):
                e = lst[it[u]]
                if cap[e] > 0 and level[to[e]] == level[u] + 1:
                    break
                it[u] += 1
            if it[u] < len(lst):
                e = lst[it[u]]
                path.append(e)
                u = to[e]
                continue
            if u == s:
                break
            level[u] = -1
            e = path.pop()
            u = to[e ^ 1]
            it[u] += 1
    seen = [False] * n
    seen[s] = True
    q = deque([s])
    while q:
        u = q.popleft()
        for e in adj[u]:
            if cap[e] > 0 and not seen[to[e]]:
                seen[to[e]] = True
                q.append(to[e])
    return flow, seen


@dataclass(frozen=True)
class Bound:
    """The bound at one (E, |c|, R_in, L): the min cut's cost (``cut``, bits) and its parts at the cut (``crossing`` b(U→D) in
    bits, ``work`` W(D) in units), the input term (bits), M_dyn, D's nodes and the parameters.

    ``m_node`` is the dynamic memory, in bits, the input term is shared over (default M_dyn). A graph of one request on a node
    that holds many: the cut grows with the requests and R_in·L doesn't, so β = cut/M_dyn + R_in·L/M_node."""

    cut: Fraction
    crossing: int
    work: Fraction
    input_bits: Fraction
    m_dyn: int
    d: tuple[int, ...]
    params: dict
    m_node: int | None = None

    @property
    def beta(self) -> Fraction | None:
        if not self.m_dyn:
            return None
        return Fraction(self.cut, self.m_dyn) + Fraction(self.input_bits, self.m_node or self.m_dyn)

    @property
    def s_stack(self) -> Fraction | None:
        """S_stack over the graph's writes, β·M_dyn."""
        return self.beta * self.m_dyn if self.m_dyn else self.cut + self.input_bits

    def to_json(self, names: Sequence[str] | None = None, show: int = 20) -> dict:
        out = {"beta": str(self.beta) if self.beta is not None else None,
               "beta_float": float(self.beta) if self.beta is not None else None,
               "s_stack_bits": str(self.s_stack), "cut_bits": str(self.cut), "crossing_bits": self.crossing,
               "d_work_units": str(self.work), "input_bits": str(self.input_bits), "m_dyn_bits": self.m_dyn,
               "m_node_bits": self.m_node or self.m_dyn, "d_nodes": len(self.d),
               **{k: str(v) if isinstance(v, Fraction) else v for k, v in self.params.items()}}
        if names is not None:
            out["d_names"] = [names[v] for v in self.d[:show]]
        return out


def bound(g: Graph, *, E: Any = E_PROVED, c: int = WORD_BITS, R_in: Any = R_IN, L: Any = 0, inputs: str = "rate",
          m_node: int | None = None, progress: Callable[[int, int], None] | None = None) -> Bound:
    """The min-cut bound on `g` at E units an encode of a `c`-bit word, R_in bytes a second over L seconds; ``inputs`` as the
    module says, ``m_node`` as `Bound` says. E, R_in and L are exact (`exact`)."""
    Eq, R, Lq = exact(E, "E"), exact(R_in, "R_in"), exact(L, "L")
    if Eq <= 0 or isinstance(c, bool) or not isinstance(c, int) or c < 1 or R < 0 or Lq < 0:
        raise ValueError(f"E > 0, an int c >= 1, R_in >= 0 and L >= 0, not E={E!r} c={c!r} R_in={R_in!r} L={L!r}")
    if inputs not in INPUTS:
        raise ValueError(f"inputs is one of {INPUTS}, not {inputs!r}")
    if m_node is not None and (isinstance(m_node, bool) or not isinstance(m_node, int) or m_node < g.m_dyn):
        raise ValueError(f"m_node is the bits of a node's dynamic memory, an int >= the graph's M_dyn {g.m_dyn}, not {m_node!r}")
    n = len(g.work)
    held = set(g.fixed) | (set(g.inputs) if inputs == "window" else set())
    S, T = n, n + 1
    work_cap: dict[int, Fraction] = {}
    for v, w in enumerate(g.work):
        if v not in held and w is not None and w:
            work_cap[v] = c * Fraction(w) / Eq
    den = lcm(1, *(f.denominator for f in work_cap.values()))
    edges: list[tuple[int, int, int]] = []
    infinite: list[int] = []
    nxt = n + 2

    def carried(x: Value) -> int:
        return x.bits if x.dynamic and not (inputs == "rate" and x.input) else 0

    for x in g.values:
        b = carried(x)
        if not b:
            continue
        if x.write:
            edges.append((x.producer, nxt, b * den))
            infinite.append(len(edges))
            edges.append((nxt, T, 0))
            nxt += 1
            continue
        rs = sorted({r for r in x.readers if r != x.producer})
        if len(rs) == 1:
            edges.append((x.producer, rs[0], b * den))
        elif rs:
            edges.append((x.producer, nxt, b * den))
            for r in rs:
                infinite.append(len(edges))
                edges.append((nxt, r, 0))
            nxt += 1
    for v, f in work_cap.items():
        edges.append((S, v, int(f * den)))
    for v in sorted(held) + [v for v, w in enumerate(g.work) if w is None and v not in held]:
        infinite.append(len(edges))
        edges.append((S, v, 0))
    big = sum(e[2] for e in edges) + 1
    for i in infinite:
        u, v, _ = edges[i]
        edges[i] = (u, v, big)
    flow, source_side = max_flow(nxt, edges, S, T, progress)
    if flow >= big:
        raise AssertionError("an infinite path from s to t: a write's producer is unbounded, which no graph built here has")
    d = tuple(v for v in range(n) if not source_side[v])
    in_d = set(d)
    crossing = sum(carried(x) for x in g.values if carried(x) and x.producer not in in_d
                   and (x.write or any(r in in_d for r in x.readers if r != x.producer)))
    work = sum((Fraction(g.work[v]) for v in d), Fraction(0))
    cut = Fraction(flow, den)
    if cut != crossing + c * work / Eq:
        raise AssertionError(f"the cut's cost {crossing} + {c}·{work}/{Eq} is not the flow's {cut}")
    input_bits = 8 * R * Lq if inputs == "rate" else Fraction(0)
    params = {"E": Eq, "c": c, "R_in": R, "L": Lq, "inputs": inputs}
    return Bound(cut, crossing, work, input_bits, g.m_dyn, d, params, m_node)


def coarsen(g: Graph, key: Callable[[int], Hashable]) -> Graph:
    """`g` with the nodes of one key as one node (in order of first appearance): work summed (unknown if any member's is),
    an input or fixed group if any member is, and values between groups (equal ones merged, bits summed)."""
    gid: dict[Hashable, int] = {}
    of = [gid.setdefault(key(v), len(gid)) for v in range(len(g.work))]
    work: list[Any] = [0] * len(gid)
    for v, w in enumerate(g.work):
        work[of[v]] = None if w is None or work[of[v]] is None else work[of[v]] + w
    merged: Counter = Counter()
    for x in g.values:
        p = of[x.producer]
        rs = () if x.write else tuple(sorted({of[r] for r in x.readers} - {p}))
        if x.write or rs:
            merged[(p, rs, x.dynamic, x.write, x.input)] += x.bits
    values = [Value(p, b, rs, dyn, w, inp) for (p, rs, dyn, w, inp), b in merged.items()]
    names = [str(k) for k in gid]
    return Graph(work, values, names, frozenset(of[v] for v in g.inputs), frozenset(of[v] for v in g.fixed))


def from_program(program: Any, ends: Sequence[int] | None = None, *, doc: Mapping | None = None,
                 window: range | None = None, writes: Callable[[int, int, Sequence[int]], bool] | None = None
                 ) -> tuple[Graph, dict]:
    """The graph of `program`'s root Calls over its census/v1 steps `ends` (`doc`: that census, else computed), and what
    property/v1 refused or flagged in it.

    ``window`` (a range of steps, default all): Calls before it are held in U, so the values they made reach the window as
    advice at their bits; Calls after it are dropped. ``writes(v, producer, readers)`` marks the dynamic-memory writes
    (default: a value a Call of the window makes that a later step reads)."""
    from verity.ml.circuits import stacking as P
    from verity.ml.circuits.census import census, step_of
    from verity.ml.circuits.program import RUN, input_id, is_input

    nodes = program.root.body.nodes
    doc = census(program, list(ends) if ends is not None else None) if doc is None else doc
    I = doc["inputs"]
    step = step_of(doc)
    S = len(doc["steps"]["ends"])
    win = range(S) if window is None else window
    if not isinstance(win, range) or win.step != 1 or not 0 <= win.start < win.stop <= S:
        raise ValueError(f"the window is a non-empty range of the census's {S} steps, not {window!r}")
    p = P._Pass()
    dyns: list[bool] = []
    work: list[Any] = [0] * len(nodes)
    refused: Counter = Counter()
    unproven: Counter = Counter()
    for k, t in p.walk(program.root.body, (), (), program.root.id, [], dyns, {}):
        if t.refused:
            work[k] = None
            for (code, fid), m in t.refused.items():
                refused[f"{code}:{fid}"] += m
        elif work[k] is not None:
            work[k] += sum(t.units.values())
        unproven.update(t.unproven)
    wide = {g: b for g, (cls, b) in sorted(p.shapes.items())
            if cls in P.BUDGETED and b > P.OPERAND_BITS and not P._column(g)}
    before = frozenset(k for k in range(I, len(nodes)) if step(k) < win.start)
    after = {k for k in range(I, len(nodes)) if step(k) >= win.stop}
    for k in after | before:
        work[k] = 0
    cand: dict[int, set[int]] = {}
    for k, vs in zip(doc["selects"]["call"], doc["selects"]["values"]):
        for v in vs:
            cand.setdefault(v, set()).add(k)
    vals = doc["values"]
    values: list[Value] = []
    for v, (j, bits, reads) in enumerate(zip(vals["producer"], vals["bits"], vals["reads"])):
        if j in after:
            continue
        rs = sorted((set(reads) | cand.get(v, set())) - after)
        allr = sorted(set(reads) | cand.get(v, set()))
        inp = j < I
        w = (writes(v, j, allr) if writes is not None else any(step(r) > step(j) for r in allr)) and not inp and j not in before
        values.append(Value(j, bits, tuple(rs), dyns[j], w, inp))
    names = [f"{k}:{nd.fn.id}" + (f"@{step(k)}" if k >= I else "") for k, nd in enumerate(nodes)]
    inputs = frozenset(k for k in range(I) if input_id(nodes[k].fn.id)[0] == RUN)  # type: ignore[index]
    info = {"nodes": len(nodes), "inputs": I, "steps": S, "window": [win.start, win.stop], "values": len(values),
            "writes": sum(x.write for x in values), "work_units": sum(w for w in work if w is not None),
            "unknown_work_nodes": sum(w is None for w in work), "refused": dict(refused),
            "operand_bits": {"B": P.OPERAND_BITS, "past_B": wide}, "s8_unproven": dict(unproven)}
    assert all(is_input(nodes[k].fn) for k in range(I))
    return Graph(work, values, names, inputs, before), info


def table(g: Graph, *, Es: Iterable[Any] = (E_PROVED, E_CONJECTURED), Ls: Iterable[Any] = (0,), R_in: Any = R_IN,
          c: int = WORD_BITS, inputs: str = "rate", m_node: int | None = None,
          progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """`bound` at each E and L (one max-flow an E: L moves only the input term)."""
    out = []
    for E in Es:
        b = bound(g, E=E, c=c, R_in=R_in, L=0, inputs=inputs, m_node=m_node, progress=progress)
        for L in Ls:
            extra = 8 * exact(R_in, "R_in") * exact(L, "L") if inputs == "rate" else Fraction(0)
            bl = Bound(b.cut, b.crossing, b.work, extra, b.m_dyn, b.d, {**b.params, "L": exact(L, "L")}, m_node)
            out.append(bl.to_json(g.names))
    return out


def report(program: Any, ends: Sequence[int], *, chunks: Iterable[int] = (), write: Callable[[dict], None] = lambda _d: None,
           log: Callable[[str], None] = lambda _m: None, **kw: Any) -> dict:
    """`program`'s bound over its steps `ends`: by step, by each of `chunks` Calls of a step at a time, then fine (`table`'s
    keywords in ``kw``). ``write(doc)`` is called after each bound, so a run cut short keeps the coarse ones."""
    from verity.ml.circuits.census import census

    g, info = from_program(program, doc=census(program, list(ends)))
    edges = sum(1 + len(x.readers) for x in g.values)
    log(f"graph: {len(g.work)} nodes, {len(g.values)} values, about {edges} edges, M_dyn {g.m_dyn} bits")
    doc: dict = {"schema": "verity-stacked-share/beta/v1", "graph": info, "graph_edges": edges,
                 "m_node_bits": kw.get("m_node"), "coarse": {}}
    I = info["inputs"]

    def step(v: int) -> str:
        return g.names[v].rsplit("@", 1)[-1]

    coarse: dict[str, Callable[[int], Hashable]] = {"by_step": lambda v: step(v) if v >= I else g.names[v]}
    for n in sorted(set(chunks), reverse=True):
        coarse[f"by_{n}_calls"] = lambda v, n=n: (step(v), (v - I) // n) if v >= I else g.names[v]

    def line(rows: list[dict]) -> str:
        return ", ".join(f"E={r['E']} L={r['L']} β={r['beta_float']:.4g}" for r in rows)

    for name, key in coarse.items():
        cg = coarsen(g, key)
        doc["coarse"][name] = {"nodes": len(cg.work), "values": len(cg.values), "bounds": table(cg, **kw)}
        write(doc)
        log(f"{name} ({len(cg.work)} nodes): {line(doc['coarse'][name]['bounds'])}")
    doc["bounds"] = table(g, **kw, progress=lambda ph, f: log(f"phase {ph}: flow {f}") if ph % 20 == 0 else None)
    write(doc)
    log(f"fine: {line(doc['bounds'])}")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("descriptor", type=Path, help="a Build's descriptor (.json or .json.gz), decoded through vLLM's registry")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", default="TokenSelect_v1", help="the family that ends a request Program's steps")
    ap.add_argument("--E", action="append", help=f"units an encode of a word costs (default {E_PROVED} and {E_CONJECTURED})")
    ap.add_argument("--L", action="append", help="seconds of lifetime (default 600 and 3600)")
    ap.add_argument("--R-in", default=str(R_IN), help="run-input bytes a second")
    ap.add_argument("-c", type=int, default=WORD_BITS, help="|c|, bits of an encoded word")
    ap.add_argument("--inputs", choices=INPUTS, default="rate")
    ap.add_argument("--m-dyn-node", help="bytes of the node's dynamic memory the input term is shared over (`Bound`; default "
                    "the Program's own writes)")
    ap.add_argument("--chunk", type=int, action="append", default=[],
                    help="also the coarse bound of each step's Calls taken N at a time, before the fine one")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from units import load  # noqa: E402

    from verity.ml.circuits.census import step_ends

    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[beta {time.monotonic() - t0:8.1f}s] {msg}", file=sys.stderr, flush=True)

    program = load(a.descriptor)
    ends = step_ends(program, [a.sample])
    log(f"loaded: {len(program.root.body.nodes)} root nodes, {len(ends)} steps")
    m_node = None if a.m_dyn_node is None else int(8 * exact(a.m_dyn_node, "--m-dyn-node"))

    def write(doc: dict) -> None:
        a.out.write_text(json.dumps({**doc, "descriptor": str(a.descriptor)}, indent=1, default=str))

    doc = report(program, ends, chunks=a.chunk, write=write, log=log, Es=a.E or (E_PROVED, E_CONJECTURED),
                 Ls=a.L or (600, 3600), R_in=a.R_in, c=a.c, inputs=a.inputs, m_node=m_node)
    print(json.dumps({"graph": doc["graph"], "bounds": [{k: r[k] for k in ("E", "L", "beta_float", "cut_bits", "m_dyn_bits")}
                                                        for r in doc["bounds"]]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
