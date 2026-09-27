"""One Call's computing graph and the unit width rule: the references a unit cut of the no-recompute partition is checked on.

`CallGraph(fn)` is the explicit gates of one activation of `fn` (its parameters are Input gates of a one-Call Program) that COMPUTE:
constants, values computed from constants alone, and wiring (`WIRING`: primitives each of whose output bits is a constant or a copy of
one input bit) are circuit structure, and a read through structure stands for reads of its source gates.  It is what a partition
checker derives from its own copy of the program rather than reads from a cut: each computing gate's declared width and primitive,
the reads between computing gates, the Call's parameter reads, the returned gates in return order, the pairs of gates computing the
same value, and the total and structure gate counts.

The width rule (`Q_word_v1{X,W,EXTRAS}`, today X = 16, W = 32, EXTRAS = 0) counts **bits**: a unit's boundary gates are its gates
another unit reads or the Call returns; with ob their summed declared widths and og their number, the unit passes when
ob <= X + EXTRAS, or when og == 1 and ob <= W + EXTRAS (one value of at most W bits).  `check_cut` applies it with
`partition.validate_unit_cut`, the invariant, to one Call's cut: failure code `unit-too-wide`, listing (unit, ob, og).

A cut is its owner per computing gate.  Its committed set is derived (`derived_committed`): a value is committed exactly when a unit
other than its producer's reads it, or the Call returns it; what serving commits is checked against that set.
"""
from __future__ import annotations

from array import array
from collections import Counter
from typing import Any, Iterable, Iterator, Sequence

from .layout import WIDE_FAN_IN, resolve
from .partition import INPUT_UNIT, CutVerdict, validate_unit_cut
from .refs import Affine

#: primitives that only rearrange bits (each output bit a constant or a copy of one input bit): wiring, like constants, is
#: computed in no unit, may be read by any, and is never committed
WIRING = frozenset({"Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"})

#: `Q_word_v1`'s width parameters (X, W, EXTRAS): at most X bits out of a unit, or exactly one value of at most W bits
WIDTH_RULE = (16, 32, 0)


def call_scope(fn: Any) -> tuple[Any, Any]:
    """(circuit, activation of `fn`) in a one-Call Program: the Call's parameters are Input gates, its returned leaves designated
    outputs.  A returned leaf a real Program does not read only shrinks out(S), so widths here bound those in any Program."""
    from .defs import CompositeDefinition
    from .program import Program
    params, ret = fn.signature()
    shell = CompositeDefinition(f"Word[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a), register=False)
    c = Program(shell).circuit
    rs = c.root_scope
    k = next(k for k, n in enumerate(rs.spec.body.nodes) if n.fn is fn)
    return c, rs.child(k)


def member_refs(a: Any, ax: int | None, m: int) -> Any:
    """The references of member `m` of a batch argument `a` over axis `ax` (None: broadcast)."""
    if ax is None:
        return a.refs
    if ax == 0:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.take(ax, m).refs


def _wide_operands(gate: Any, call: Any) -> Iterator[int]:
    """A wide fan-in gate's operands, skipping whole arguments that are the Call's own parameters (Input gates, outside the Call)."""
    s = gate.scope
    body = s.spec.body
    k = body.node_at(gate.index - s.offset)
    node = body.nodes[k]
    m = (gate.index - s.offset - body.offsets[k]) // node.fn.gates if node.form == "batch" else 0
    at_call = s.spec is call.spec and s.offset == call.offset
    for a, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
        refs = member_refs(a, ax, m)
        if at_call and isinstance(refs, Affine) and refs.space == "p":
            continue
        for j in range(len(refs)):
            yield resolve(s, refs[j])


class CallGraph:
    """The computing gates of one activation of `fn`, compacted in canonical (topological) order:

    * `n`, `keep` (computing gate -> the Call's gate index), `width` (declared bits), `prim` / `scope` (indices into `prim_names` /
      `scope_names`), `external` (whether it reads outside the Call, a wide gate included);
    * `src[e] -> dst[e]`: the reads between computing gates, each once per consumer, through structure;
    * `input_ids` (root gate id of each Call parameter leaf read, first-read order) and `input_reads` ((input index, gate) pairs);
    * `outputs`: the returned computing gates in return order (a returned wiring value: its source gates; a constant: nothing), and
      `returned_inputs`: the parameter leaves a returned wiring value reads (returned as they are);
    * `recomputed`: (gate, earlier gate) pairs computing the same value (the same primitive over the same operands by value);
    * `total`, `free` (structure gates), `free_by_prim`."""

    def __init__(self, fn: Any, wiring: frozenset[str] = WIRING) -> None:
        c, call = call_scope(fn)
        lo, n = call.offset, fn.gates
        width = array("q", bytes(8 * n))
        prim = array("q", bytes(8 * n))
        scope = array("q", bytes(8 * n))
        external = bytearray(n)
        free = bytearray(n)
        const = bytearray(n)
        self.prims: dict[str, int] = {}
        self.scopes: dict[str, int] = {}
        ops: list[list[int]] = []
        inputs: dict[int, int] = {}                                  # the Call's parameter gates it reads (root id -> index)
        wide: set[int] = set()
        for j in range(n):
            g = c.gate(lo + j)
            width[j] = g.prim.ret.w
            prim[j] = self.prims.setdefault(g.prim.id, len(self.prims))
            scope[j] = self.scopes.setdefault(g.scope.spec.id, len(self.scopes))
            const[j] = not g.prim.params
            free[j] = const[j] or g.prim.id in wiring
            it: Iterable[int] = g.operands
            ext = False
            if len(g.operands) > WIDE_FAN_IN:
                it = _wide_operands(g, call)
                ext = True
                wide.add(j)
            mine = []
            for o in it:
                if 0 <= o - lo < n:
                    mine.append(o - lo)
                else:
                    ext = True
                    if j not in wide:                                  # a wide gate's inputs: reads of committed inputs, not listed
                        mine.append(-1 - inputs.setdefault(o, len(inputs)))      # an input gate: -1 - its index
            external[j] = ext
            ops.append(mine)
        for j in range(n):                                          # a value computed from constants alone is a constant too
            if not free[j] and not external[j] and ops[j] and all(o >= 0 and const[o] for o in ops[j]):
                const[j] = free[j] = True
        # recomputes: a computation gate with the primitive and the operands (by value: constants, inputs, wiring, earlier canonical
        # gates) of an earlier one computes the same value again; a gate whose wide operands were not enumerated matches nothing
        tok = array("q", bytes(8 * n))                               # a value's token: equal tokens, equal values
        first: dict[int, int] = {}                                  # key hash -> the first gate with that key
        dup: list[tuple[int, int]] = []

        def key(j: int) -> tuple:
            return (int(prim[j]),) + tuple(hash(("i", -o)) if o < 0 else tok[o] for o in ops[j])   # -o >= 1: hash(-1) == hash(-2)

        for j in range(n):
            k = key(j)
            h = hash(k)
            if free[j]:
                tok[j] = hash(("s", h))
            elif j in wide:
                tok[j] = hash(("u", j))
            elif h in first and key(first[h]) == k:
                dup.append((j, first[h]))
                tok[j] = tok[first[h]]
            else:
                first.setdefault(h, j)
                tok[j] = hash(("g", j))
        srcs: dict[int, tuple[int, ...]] = {}
        for j in range(n):
            if free[j]:
                s: set[int] = set()
                for o in ops[j]:
                    s.update(srcs[o] if o >= 0 and free[o] else (o,))
                srcs[j] = tuple(sorted(s))
        keep = array("q", (j for j in range(n) if not free[j]))
        idx = array("q", [-1]) * n
        for i, j in enumerate(keep):
            idx[j] = i
        src, dst = array("q"), array("q")
        self.input_reads: list[tuple[int, int]] = []                 # (input index, computing gate)
        for j in keep:
            seen: set[int] = set()
            for o in ops[j]:
                for u in (srcs[o] if o >= 0 and free[o] else (o,)):
                    if u in seen:
                        continue
                    seen.add(u)
                    if u < 0:
                        self.input_reads.append((-1 - u, idx[j]))
                    else:
                        src.append(idx[u])
                        dst.append(idx[j])
        self.input_ids = sorted(inputs, key=inputs.__getitem__)       # root gate id of each input gate read
        self.total, self.free = n, sum(free)
        self.n = len(keep)
        self.keep = keep
        self.recomputed = [(idx[a], idx[b]) for a, b in dup]         # (gate, earlier gate computing the same value)
        self.width = array("q", (width[j] for j in keep))
        self.prim = array("q", (prim[j] for j in keep))
        self.scope = array("q", (scope[j] for j in keep))
        self.external = bytes(external[j] for j in keep)
        self.src, self.dst = src, dst
        outs: list[int] = []
        returned_inputs: set[int] = set()
        for g in (resolve(call, r) - lo for r in fn.body.ret):
            if 0 <= g < n:
                for u in (srcs[g] if free[g] else (g,)):
                    if u >= 0:
                        outs.append(idx[u])
                    else:
                        returned_inputs.add(-1 - u)
        self.outputs = outs
        self.returned_inputs = sorted(returned_inputs)
        self.prim_names = {v: k for k, v in self.prims.items()}
        self.scope_names = {v: k for k, v in self.scopes.items()}
        counts = Counter(prim[j] for j in range(n) if free[j])
        self.free_by_prim = Counter({self.prim_names[k]: counts[k] for k in sorted(counts)})


def fits(out_bits: Any, out_gates: Any, X: int = WIDTH_RULE[0], W: int = WIDTH_RULE[1], extras: int = WIDTH_RULE[2]) -> Any:
    """The width rule on (ob, og): ob <= X + extras, or og == 1 and ob <= W + extras.  Elementwise on numpy arrays as on ints."""
    return (out_bits <= X + extras) | ((out_gates == 1) & (out_bits <= W + extras))


def boundary_widths(graph: CallGraph, owner: Sequence[int], units: int | None = None) -> tuple[list[int], list[int]]:
    """(ob, og) per computing unit 0..units-1 of the cut `owner` (a unit per computing gate): the summed declared bits and the number
    of its boundary gates, those another unit reads or the Call returns."""
    U = (max(owner) + 1 if len(owner) else 0) if units is None else units
    boundary = bytearray(graph.n)
    for p, q in zip(graph.src, graph.dst):
        if owner[p] != owner[q]:
            boundary[p] = 1
    for g in graph.outputs:
        boundary[g] = 1
    ob, og = [0] * U, [0] * U
    for g in range(graph.n):
        if boundary[g]:
            ob[owner[g]] += graph.width[g]
            og[owner[g]] += 1
    return ob, og


def derived_committed(graph: CallGraph, owner: Sequence[int]) -> list[int]:
    """The computing gates the cut `owner` commits, ascending: a value is committed exactly when a unit other than its producer's reads
    it, or the Call returns it (the invariant's boundary set; a cut never stores it)."""
    c = {p for p, q in zip(graph.src, graph.dst) if owner[p] != owner[q]}
    c.update(graph.outputs)
    return sorted(c)


def check_cut(graph: CallGraph, owner: Sequence[int], committed: Iterable[int] | None = None,
              limits: tuple[int, int, int] | None = WIDTH_RULE) -> CutVerdict:
    """One Call's cut against its graph: `validate_unit_cut` (every computing gate in one unit, reads across units only of committed
    values, returned values committed, no value computed in two units), the Call's parameters standing for the input unit, and, unless
    `limits` is None, the width rule (`unit-too-wide`: (unit, ob, og) per failing unit).  `owner`: a unit per computing gate;
    `committed`: what serving commits, held to the derived set (`committed-unread`: served but not derived; `read-uncommitted` /
    `output-uncommitted`: derived but not served); None: the derived set (`derived_committed`)."""
    owner = [int(u) for u in owner]
    if len(owner) != graph.n:
        return CutVerdict(False, ("owner-length",), {"owner-length": {"n": 1, "gates": [len(owner), graph.n]}})
    if committed is None:
        committed = derived_committed(graph, owner)
    k = len(graph.input_ids)
    reads = list(zip(graph.src, graph.dst)) + [(graph.n + i, q) for i, q in graph.input_reads]
    cut = validate_unit_cut(graph.total + k, graph.free, owner + [INPUT_UNIT] * k, reads,
                            list(graph.outputs) + [graph.n + i for i in graph.returned_inputs],
                            sorted({int(g) for g in committed}) + list(range(graph.n, graph.n + k)), range(graph.n, graph.n + k),
                            graph.recomputed)
    if limits is None:
        return cut
    ob, og = boundary_widths(graph, owner)
    wide = [[u, b, c] for u, (b, c) in enumerate(zip(ob, og)) if not fits(b, c, *limits)]
    if not wide:
        return cut
    detail = dict(cut.detail)
    detail["unit-too-wide"] = {"n": len(wide), "gates": wide[:64], "limits": list(limits)}
    return CutVerdict(False, tuple(cut.codes) + ("unit-too-wide",), detail)


__all__ = ["WIRING", "WIDTH_RULE", "call_scope", "member_refs", "CallGraph", "fits", "boundary_widths", "derived_committed", "check_cut"]
