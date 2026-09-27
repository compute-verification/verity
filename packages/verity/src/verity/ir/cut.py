"""`Q_word` v1, the partition query, and the references a cut is checked on: each Call's computing graph, the cut, the width rule.

The algorithm, as a verifier re-implements it (`partition_object.evaluate`; any change to its behaviour bumps the query's version):

**Input.**  The program's descriptor, `codec.canonical_json(codec.encode_program(program))` minus `annotations` (schema
`verity-ir/descriptor/v1`), decoded by `codec.decode_program`.  The format, the canonical layout and reference resolution are
specified in `verity/ir/PROTOCOL.md` (pinned by `tests/ir/format_vectors.json`); its §8 restates the wide-gate rule of step 3.

0. **Applicability.**  Q_word v1 applies to a program that is topological -- every node of every body names only its body's
   parameters and earlier nodes, a body's `ret` any of its nodes (`partition_object.order_violation`, checked for every query) -- and
   whose root body's nodes other than its Input nodes are each of form `call` or `primitive`.  Otherwise the query is inapplicable:
   `evaluate` raises `QueryInapplicable` and `verify` refuses with `query-inapplicable` (where, and why).  `CallGraph`'s own check
   (step 3: an operand inside the Call comes before its gate) is a backstop the program-level rule already implies.
1. **Calls.**  The root body's nodes that are not Input nodes (callee id `Input<w>_v1`), in body order.  Each is cut as its Definition's activation
   in a one-Call Program (`call_scope`): the Definition's parameter leaves are Input gates `0..L-1` in parameter order, its own gates
   `L..L+gates-1`.  Activations of one Definition share its cut.  A primitive Call is one gate: it reads its inputs and is returned.
2. **Evaluate(fn, X).**  A primitive is one unit, or none when it has no parameters or its id is in `WIRING`.  A separable body
   (`separable`: no scan node, every node reads only parameters, the body returns every leaf each node produces) is its nodes'
   members' cuts, in node then member order.  Any other body is `cut_word(CallGraph(fn), X)`.
3. **CallGraph(fn)**, over the Call's gates `j = 0..n-1` in layout order:
   * operands: the gate's operand ids (one inside the Call must come before the gate, or the query is inapplicable: the Call is not
     topological); a gate with more than `WIDE_FAN_IN` (4096) operands is *wide*: its operands are
     enumerated per argument (`_wide_operands`), skipping an argument that is a whole parameter of the Call; an operand outside the
     Call is an input, numbered by first read, and marks the gate *external*; a wide gate's outside operands are not listed;
   * structure: a gate is a constant when its primitive has no parameters, wiring when its id is in `WIRING`; then, j ascending, a
     non-structure, non-external gate with at least one operand, all of them constants, is a constant too;
   * recomputes: `key(j) = (primitive, operand tokens)`, an input's token interned from `("i", input index)`; a structure gate's token
     is interned from `("s",) + key`, a wide gate's from `("u", j)`; a computing gate whose key an earlier computing gate `f` had is
     the recomputed pair `(j, f)` and takes `f`'s token, else it is interned from `("g", j)`.  Tokens are exact interned ids
     (`_Interner`), never hashes;
   * reads through structure: a structure gate's sources are its operands with every structure operand replaced by its sources;
     the computing gates, compacted in layout order, read each distinct source of their operands once, in operand order (an input:
     `input_reads`; a gate: an edge `src -> dst`);
   * outputs: each returned leaf in `body.ret` order (a primitive Call: its gate), resolved inside the Call: a computing gate, or a
     structure gate's computing sources (ascending) -- its input sources are `returned_inputs`.
4. **cut_word(graph, X).**  (i) `targets`: the returned computing gates, first occurrence in return order, packed greedily into runs
   of at most X declared bits (a wider gate is a run alone); each run is an *output* unit and heads its gates.  (ii) Repeat: every
   gate starts at its head's unit or -1; backward (`g = n-1..0`) a gate without a head takes the unit its consumers agree on
   (consumers in edge order; one consumer's unit, ignoring -1 and repeats; -2 when two differ).  Producers `p` of edges `(p, q)`
   with `own[p] = -2` and `own[q] >= 0`, ascending, become new *committed* units heading themselves; if there are none, gates still
   at -1 (ascending) join the one unit their uncommitted producers are all in (exactly one producer unit, and it is live), and every
   gate still at -1 heads its own *dead* unit; producers of edges across units that head nothing, ascending, then become committed
   units (the dead units dropped and the pass repeated), or, if there are none, the cut is final.  Units are numbered as created:
   outputs, committed in promotion order, dead last.
5. **Units of the Program.**  By Call, then within the Call's cut.  The input unit is not a unit.
6. **Checks** (`check_cut`).  The committed set is derived (`derived_committed`): a value is committed exactly when a unit other
   than its producer's reads it, or the Call returns it.  `partition.validate_unit_cut` holds the invariant (the Call's parameters
   standing for the input unit); the width rule counts **bits**: with ob the summed declared widths and og the number of a unit's
   boundary gates (read by another unit, or returned), a unit passes when ob <= X, or og == 1 and ob <= W; code `unit-too-wide`.
   What serving commits, when given, is held to the derived set (`committed-unread`, `read-uncommitted`, `output-uncommitted`).

**Constants of the version.**  `WIRING` and `WIDE_FAN_IN` are part of Q_word v1, like the algorithm: changing either is a new version.
`tests/ir/qword_vectors.json` pins both as data, beside vectors for every step above and a seeded random corpus.

**Cost.**  `owners` runs one backward pass per round of promotions, so O(n · (n + E)) for n computing gates and E edges in the worst
case.  `evaluate` shortcuts separable bodies, but `verify` builds the flat `CallGraph` of every distinct Definition, separable ones
included, to check the cut on it.
"""
from __future__ import annotations

from array import array
from collections import Counter
from typing import Any, Iterable, Iterator, Sequence

from .layout import resolve
from .partition import INPUT_UNIT, CutVerdict, validate_unit_cut
from .refs import Affine

#: primitives that only rearrange bits (each output bit a constant or a copy of one input bit): wiring, like constants, is
#: computed in no unit, may be read by any, and is never committed
WIRING = frozenset({"Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"})

#: a gate with more operands than this is *wide* (step 3); Q_word v1's own threshold, not the layout's lazy-operand one
WIDE_FAN_IN = 4096

#: `Q_word_v1`'s width parameters (X, W, EXTRAS): at most X bits out of a unit, or exactly one value of at most W bits
WIDTH_RULE = (16, 32, 0)


class QueryInapplicable(ValueError):
    """The query does not apply to the program (step 0); `detail` says where."""

    def __init__(self, message: str, detail: dict) -> None:
        super().__init__(message)
        self.detail = detail


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


class _Interner(dict):
    """Structural keys -> dense ids, in first-seen order."""

    def __call__(self, key: tuple) -> int:
        t = self.get(key)
        if t is None:
            t = self[key] = len(self)
        return t


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
                    if o - lo >= j:
                        raise QueryInapplicable(f"{fn.id}: gate {j} reads gate {o - lo}, not before it (the layout is not topological)",
                                                {"definition": fn.id, "gate": j, "reads": o - lo, "why": "not topological"})
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
        # gates) of an earlier one computes the same value again; a gate whose wide operands were not enumerated matches nothing.
        # Values are exact interned ids (`_Interner`), never hashes: equal ids, structurally equal values.
        ids = _Interner()
        tok = array("q", bytes(8 * n))
        first: dict[tuple, int] = {}                                # computing key -> the first gate with it
        dup: list[tuple[int, int]] = []
        for j in range(n):
            k = (prim[j],) + tuple(ids(("i", -1 - o)) if o < 0 else tok[o] for o in ops[j])
            if free[j]:
                tok[j] = ids(("s",) + k)
            elif j in wide:
                tok[j] = ids(("u", j))
            elif k in first:
                dup.append((j, first[k]))
                tok[j] = tok[first[k]]
            else:
                first[k] = j
                tok[j] = ids(("g", j))
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
        for g in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
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


def separable(fn: Any) -> list[tuple[Any, int]] | None:
    """[(callee, members)] per body node when `fn`'s body nodes read only its parameters and it returns every leaf they produce: no
    value crosses between nodes, so the cut is each node member's own.  None otherwise (a scan node included)."""
    from .intervals import norm, ref_intervals, total
    from .parts import n_members
    if not hasattr(fn, "body"):
        return None
    body = fn.body
    for n in body.nodes:
        if n.form == "scan":
            return None
        reads: dict = {}
        for a in n.args:
            ref_intervals(a.refs, reads)
        if any(space != "p" for space, _i in reads):
            return None
    ret: dict = {}
    ref_intervals(body.ret, ret)
    for k, n in enumerate(body.nodes):
        if total(norm(ret.get(("n", k), ()))) != n.out.leaves:
            return None
    return [(n.fn, n_members(n)) for n in body.nodes]


def targets(graph: CallGraph, X: int) -> list[list[int]]:
    """Step 1: the returned computing gates, first occurrence in return order, packed into runs of at most X declared bits (a gate
    wider than X is a run alone)."""
    seen, out, cur, bits = set(), [], [], 0
    width = graph.width.tolist()
    for g in graph.outputs:
        if g in seen:
            continue
        seen.add(g)
        w = width[g]
        if cur and bits + w > X:
            out.append(cur)
            cur, bits = [], 0
        cur.append(g)
        bits += w
    if cur:
        out.append(cur)
    return out


def owners(graph: CallGraph, head: list[int], kinds: list[str]) -> tuple[list[int], list[int]]:
    """Steps 2-3 (see the module docstring): owner per computing gate, and the heads, extended by the committed values."""
    n = graph.n
    consumers: list[list[int]] = [[] for _ in range(n)]
    producers: list[list[int]] = [[] for _ in range(n)]
    edges = list(zip(graph.src.tolist(), graph.dst.tolist()))
    for p, q in edges:
        consumers[p].append(q)
        producers[q].append(p)
    while True:
        own = list(head)
        for g in range(n - 1, -1, -1):
            if own[g] >= 0:
                continue
            o = -1
            for c in consumers[g]:
                oc = own[c]
                if oc == -1 or oc == o:
                    continue
                if o == -1 and oc >= 0:
                    o = oc
                else:
                    o = -2
                    break
            own[g] = o
        promote = sorted({p for p, q in edges if own[p] == -2 and own[q] >= 0})
        if not promote:
            live = len(kinds)
            for g in range(n):
                if own[g] != -1:
                    continue
                us = {own[p] for p in producers[g] if head[p] < 0}
                alive = {u for u in us if 0 <= u < live}
                own[g] = alive.pop() if len(alive) == 1 and len(us) == 1 else -1
            for g in range(n):
                if own[g] == -1:
                    own[g] = len(kinds)
                    kinds.append("dead")
            promote = sorted({p for p, q in edges if own[p] != own[q] and head[p] < 0})
            if not promote:
                return own, head
            del kinds[live:]
        for i, g in enumerate(promote):
            head[g] = len(kinds) + i
        kinds += ["committed"] * len(promote)


def cut_word(graph: CallGraph, X: int) -> tuple[list[int], list[str], list[int]]:
    """`Q_word`'s cut of one flat Call: (owner per computing gate, kind per unit: output / committed / dead, head per gate: its unit
    when the gate heads it, else -1)."""
    head = [-1] * graph.n
    kinds: list[str] = []
    for t in targets(graph, X):
        for g in t:
            head[g] = len(kinds)
        kinds.append("output")
    own, head = owners(graph, head, kinds)
    return own, kinds, head


class DefinitionCut:
    """`Q_word{X}`'s cut of one Definition specialization, as evaluated: `flat` (owner per computing gate, `kinds`) for a body cut as a
    whole, or `parts` [(child cut or None for a structure primitive, members)] for a separable body, whose units are its members' in
    node, then member, order.  `units`: how many; `owner()`: the owner per computing gate of the activation, materialised."""

    __slots__ = ("fn", "flat", "kinds", "parts", "units", "gates")

    def __init__(self, fn: Any, flat: list[int] | None = None, kinds: list[str] | None = None,
                 parts: list[tuple["DefinitionCut | None", int]] | None = None) -> None:
        self.fn, self.flat, self.kinds, self.parts = fn, flat, kinds, parts
        if parts is None:
            self.units, self.gates = len(kinds or ()), len(flat or ())
        else:
            self.units = sum(c.units * m for c, m in parts if c is not None)
            self.gates = sum(c.gates * m for c, m in parts if c is not None)

    def owner(self) -> list[int]:
        if self.parts is None:
            return list(self.flat)
        out: list[int] = []
        base = 0
        for c, m in self.parts:
            if c is None:
                continue
            sub = c.owner()
            for _ in range(m):
                out.extend(u + base for u in sub)
                base += c.units
        return out

    def locate(self, u: int) -> list[tuple[int, int, int]]:
        """Unit `u` of this cut as a path of (body node, member, unit within the member's cut) down to a flat cut."""
        if not 0 <= u < self.units:
            raise IndexError(f"unit {u} outside 0..{self.units - 1}")
        if self.parts is None:
            return []
        for k, (c, m) in enumerate(self.parts):
            if c is None:
                continue
            if u < c.units * m:
                return [(k, u // c.units, u % c.units)] + c.locate(u % c.units)
            u -= c.units * m
        raise AssertionError("unreachable")


def evaluate_definition(fn: Any, X: int, memo: dict | None = None, wiring: frozenset[str] = WIRING) -> DefinitionCut:
    """`Q_word{X}` on one activation of `fn`: a primitive is one unit (none when it is structure: a constant or wiring); a separable body
    is its members' cuts (`separable`); any other body is `cut_word` of its `CallGraph`.  Memoised by Definition id."""
    memo = {} if memo is None else memo
    key = ("Q_word", fn.id, X)
    got = memo.get(key)
    if got is not None:
        return got
    if not hasattr(fn, "body"):
        r = DefinitionCut(fn, [], []) if (not fn.params or fn.id in wiring) else DefinitionCut(fn, [0], ["output"])
    elif (sep := separable(fn)) is not None:
        parts = []
        for callee, m in sep:
            c = evaluate_definition(callee, X, memo, wiring)
            parts.append((c if c.gates else None, m))
        r = DefinitionCut(fn, parts=parts)
    else:
        own, kinds, _head = cut_word(CallGraph(fn, wiring), X)
        r = DefinitionCut(fn, own, kinds)
    memo[key] = r
    return r


__all__ = ["WIRING", "WIDE_FAN_IN", "WIDTH_RULE", "QueryInapplicable", "call_scope", "member_refs", "CallGraph", "fits", "boundary_widths", "derived_committed", "check_cut",
           "separable", "targets", "owners", "cut_word", "DefinitionCut", "evaluate_definition"]
