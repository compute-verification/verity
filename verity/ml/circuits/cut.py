"""`Q_word` v1 (and v2, a delta on it, below) and `Q_call` v1 (last), the partition queries, and the references a cut is checked on:
each Call's computing graph, the cut, the width rule.

The algorithm, as a verifier re-implements it (`partition_object.evaluate`; any change to its behaviour bumps the query's version):

**Input.**  The program's descriptor, `codec.canonical_json(codec.encode_program(program))` minus `annotations` (schema
`verity-ir/descriptor/v1`), decoded by `codec.decode_program`.  The format, the canonical layout and reference resolution are
specified in `verity/ml/circuits/PROTOCOL.md` (pinned by `verity/ml/circuits/tests/format_vectors.json`); its §8 restates the wide-gate rule of step 3.

0. **Applicability.**  Q_word v1 applies to a program that is topological -- every node of every body names only its body's
   parameters and earlier nodes, a body's `ret` any of its nodes (`partition_object.order_violation`, checked for every query) -- and
   whose root body's nodes other than its input nodes are each of form `call` or `primitive`.  Otherwise the query is inapplicable:
   `evaluate` raises `QueryInapplicable` and `verify` refuses with `query-inapplicable` (where, and why).  `CallGraph`'s own check
   (step 3: an operand inside the Call comes before its gate) is a backstop the program-level rule already implies.
1. **Calls.**  The root body's nodes that are not input nodes (callee id `Input<w>_v1` or `RegisteredInput<w>_v1`), in body order.  Each is cut as its Definition's activation
   in a one-Call Program (`call_scope`): the Definition's parameter leaves are Input gates `0..L-1` in parameter order, its own gates
   `L..L+gates-1`.  Activations of one Definition share its cut.  A primitive Call is one gate: it reads its inputs and is returned.
2. **Evaluate(fn, X).**  A primitive is one unit, or none when it has no parameters or its id is in `WIRING`.  A separable body
   (`separable`: no scan node, every node reads only parameters, the body returns every leaf each node produces) is its nodes'
   members' cuts, in node then member order.  Any other body is `cut_word(CallGraph(fn), X)`.
3. **CallGraph(fn)**, over the Call's gates `j = 0..n-1` in layout order:
   * operands: the gate's operand ids (one inside the Call must come before the gate, or the query is inapplicable: the Call is not
     topological); a gate with more than `WIDE_FAN_IN` (4096) operands is *wide*: its operands are
     enumerated per argument (`wide_operands`), skipping the Call's own parameters; an operand outside the
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
`verity/ml/circuits/tests/qword_vectors.json` pins both as data, beside vectors for every step above and a seeded random corpus.

**Q_word v2** is v1 with one change, in step 6: a recomputed pair `(j, f)` whose gates lie in two units is not a failure.  Steps 0-5,
`WIRING`, `WIDE_FAN_IN` and the width rule are v1's, so v2's Calls, units, owners, committed sets and widths are v1's on every program
(on one v1 refuses only for `gate-recomputed`, the cut v1 computed before refusing).  `check_cut` still computes the pairs: under v2
(`RECOMPUTE[2]`, "report") the pairs across units are `detail["recomputed_across"]` (count and first pairs) instead of the code
`gate-recomputed`; `redundant_gates` counts the pairs within one unit, as in v1; every other code is unchanged.  Each copy is its own
gate, certified by one unit from that unit's gates and committed values, so v2's proof blocks may compute a common value: a count of
them bounds the faulty ones' size but counts a shared value more than once.  `check_cut`'s default, and so `validate_unit_cut`'s, stays v1's refusal.

**Cost.**  `owners` runs one backward pass per round of promotions, so O(n · (n + E)) for n computing gates and E edges in the worst
case.  `evaluate` shortcuts separable bodies, but `verify` builds the flat `CallGraph` of every distinct Definition, separable ones
included, to check the cut on it.

**Q_call v1** (`evaluate_call`, `PROTOCOL.md` §11), a query of its own, with one parameter X (bits; 32 is the agreed value):

0. **Applicability.**  Q_word's step 0: topological, and every root Call a `call` or `primitive` node (`why: "root-form"`).  And no
   *wide gate*: a computing gate (below) whose declared width exceeds X makes the query inapplicable (`why: "gate-too-wide"`, with
   the Call, the Definition, the gate, its primitive and width).  An Input gate or a constant of any width is no computing gate.
1. **Calls.**  Q_word's step 1: each root Call is cut as its Definition's activation in a one-Call Program (`call_scope`), its
   parameter leaves Input gates `0..L-1`, its gates `0..n-1` (local, layout order), every returned leaf a designated output.
   Activations of one Definition share its cut.
2. **Gates** (`CallGates`).  A gate is *computing* when its primitive has parameters; Input gates and constants (no parameters)
   are inputs: in no unit, outside the committed set.  There is no wiring list, no constant folding and no wide fan-in rule: every operand
   of every gate is read.  `last[g]` is the last gate of the Call that reads computing gate `g`, or `n` when the Call returns `g`
   (a returned leaf resolving to a computing gate of the Call; one resolving to a parameter or a constant returns nothing of the
   Call's), or -1 when nothing reads it.
3. **Pieces.**  A piece is a gate interval `[lo, hi)` of the Call: the Call itself, a body node (`offsets[k]..offsets[k+1]` of
   its activation), or a member or iteration of a batch or scan node.  Its outputs are its computing gates `g` with
   `last[g] >= hi` (read after the piece, or returned): the layout is topological, so nothing before `lo` reads inside it, and
   this is `parts.out(S)` for S the piece's computing gates.  w_out is their summed declared widths.
4. **Cut**, the term `each(instances("@*"), frontier(X))` (`query_ast`'s `Frontier`; one Call's sets `frontier_pieces`), from
   the Call down, depth first in layout order: a piece without computing gates is no unit; one
   with w_out <= X is one unit; otherwise its children are the pieces, in order -- a composite activation's (the Call's, a call
   node's callee, a composite member's) body nodes; a batch or scan node's members or iterations.  A primitive piece is one
   gate of at most X bits, so it always fits, and the cut ends.  A unit is (`lo`, `hi`, path): the path is its steps from the
   Call's activation, `[k]` a body node, `[k, m]` member `m` of node `k`.
5. **Units of the Program.**  By Call, then layout order within it; the input unit is not a unit.
6. **Checks** (`check_call_cut`).  Owner per computing gate: its unit.  Committed: a computing gate another unit reads or the
   Call returns, i.e. `last[g] >= hi` of its unit, so a unit's committed bits are its w_out.  `validate_unit_cut` holds the
   invariant (its structure the constants: gates any unit may read, bound outside the partition; the Call's parameters its
   input unit), recomputes reported as under Q_word v2 (keys
   below), and the width rule is w_out <= X (`unit-too-wide`).  Served commits, numbered as `CallGates` compacts the computing
   gates, are held to the derived set.
   Recomputes: `key(j) = (primitive id, operand tokens)`, a parameter's token from `("i", leaf)`, a constant's (an operation's own
   bits) from `("c", primitive id)`; a computing gate whose key an earlier one `f` had is the pair `(j, f)` and takes `f`'s token,
   else its own, `("g", j)`.  A program's data reaches the Call through its parameters, from registered or run input gates, so it
   is compared by leaf, never by value.

**Cost.**  O(n) to build `CallGates` (one pass over the gates and their operands); the cut is structural, each Definition with a
given set of returns read resolved once, its pieces' w_out from the description, never gate by gate.
"""
from __future__ import annotations

from array import array
from collections import Counter
from typing import Any, Iterable, Iterator, Sequence

from .layout import resolve
from .partition import INPUT_UNIT, CutVerdict, validate_unit_cut
from .query_ast import frontier_pieces
from .refs import Affine, Strided

#: primitives that only rearrange bits (each output bit a constant or a copy of one input bit): wiring, like constants, is
#: computed in no unit, may be read by any, and is outside the committed set
WIRING = frozenset({"Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"})

#: a gate with more operands than this is *wide* (step 3); Q_word v1's own threshold, not the layout's lazy-operand one
WIDE_FAN_IN = 4096

#: `Q_word_v1`'s width parameters (X, W, EXTRAS): at most X bits out of a unit, or exactly one value of at most W bits
WIDTH_RULE = (16, 32, 0)

#: per `Q_word` version, `check_cut`'s rule for a value computed in two units (`partition.RECOMPUTE_RULES`): v2 reports what v1 refuses
RECOMPUTE = {1: "refuse", 2: "report"}


class QueryInapplicable(ValueError):
    """The query does not apply to the program (step 0); `detail` says where."""

    def __init__(self, message: str, detail: dict) -> None:
        super().__init__(message)
        self.detail = detail


def call_scope(fn: Any) -> tuple[Any, Any]:
    """(circuit, activation of `fn`) in a one-Call Program: the Call's parameters are Input gates, its returned leaves designated
    outputs.  A returned leaf a real Program does not read only shrinks out(S), so widths here bound those in any Program."""
    from .defs import CompositeDefinition
    from .ports import forward, harness
    from .program import Program
    params, ret = fn.signature()
    shell = harness(CompositeDefinition(f"Word[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                        register=False, ports=forward(fn)))
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


def wide_operands(gate: Any, call: Any) -> Iterator[int]:
    """A wide fan-in gate's operands, skipping those that are the Call's own parameters (Input gates, outside the Call, which a
    wide gate never lists) without resolving them."""
    s = gate.scope
    body = s.spec.body
    k = body.node_at(gate.index - s.offset)
    node = body.nodes[k]
    m = (gate.index - s.offset - body.offsets[k]) // node.fn.gates if node.form == "batch" else 0
    at_call = s.spec is call.spec and s.offset == call.offset
    for a, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
        refs = member_refs(a, ax, m)
        if at_call and isinstance(refs, (Affine, Strided)) and refs.space == "p":
            continue
        for j in range(len(refs)):
            r = refs[j]
            if at_call and r[0] == "p":
                continue
            yield resolve(s, r)


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
                it = wide_operands(g, call)
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
              limits: tuple[int, int, int] | None = WIDTH_RULE, recompute: str = RECOMPUTE[1]) -> CutVerdict:
    """One Call's cut against its graph: `validate_unit_cut` (every computing gate in one unit, reads across units only of committed
    values, returned values committed, no value computed in two units), the Call's parameters standing for the input unit, and, unless
    `limits` is None, the width rule (`unit-too-wide`: (unit, ob, og) per failing unit).  `owner`: a unit per computing gate;
    `committed`: what serving commits, held to the derived set (`committed-unread`: served but not derived; `read-uncommitted` /
    `output-uncommitted`: derived but not served); None: the derived set (`derived_committed`).  `recompute`: v1's "refuse" unless
    the query is `Q_word` v2 (`RECOMPUTE`)."""
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
                            graph.recomputed, recompute)
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


# --- Q_call v1 ------------------------------------------------------------------------------------------------------------------

#: `Q_call` v1's agreed bound: at most this many bits out of a unit
CALL_X = 32


class CallGates:
    """One activation of `fn` gate by gate (step 2): `n` local gates, `inputs` (the Call's parameter leaves, Input gates `0..L-1` of
    `call_scope`), per gate `computing`, `width`, `prim` (id), `last` (the last local reader; `n` when returned; -1 unread);
    `keep` (compact -> local) and `idx` (local -> compact, -1 for an input), the compact numbering `verify`'s served commits use;
    `reads`: (producer, consumer) compact pairs, each distinct producer once per consumer, a parameter leaf `i` as `nc + i`;
    `outputs`: the returned computing gates (compact, first occurrence in return order); `recomputed`: (gate, earlier) compact pairs."""

    def __init__(self, fn: Any) -> None:
        c, call = call_scope(fn)
        lo, n = call.offset, fn.gates
        self.fn, self.n, self.inputs = fn, n, lo
        computing = bytearray(n)
        width = [0] * n
        prim: list[str] = [""] * n
        last = [-1] * n
        ops: list[tuple[int, ...]] = [()] * n
        for j in range(n):
            g = c.gate(lo + j)
            p = g.prim
            width[j], prim[j] = p.ret.w, p.id
            if not p.params:
                continue
            computing[j] = 1
            mine = []
            for o in g.operands:
                o -= lo
                if o >= j:
                    raise QueryInapplicable(f"{fn.id}: gate {j} reads gate {o}, not before it (the layout is not topological)",
                                            {"definition": fn.id, "gate": j, "reads": o, "why": "not topological"})
                mine.append(o)                                       # < 0: parameter leaf lo + o
                if o >= 0 and computing[o] and last[o] < j:
                    last[o] = j
            ops[j] = tuple(mine)
        returned: list[int] = []
        for g in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
            if 0 <= g < n and computing[g]:
                returned.append(g)
                last[g] = n
        self.computing, self.width, self.prim, self.last = computing, width, prim, last
        self.keep = [j for j in range(n) if computing[j]]
        self.nc = len(self.keep)
        idx = [-1] * n
        for i, j in enumerate(self.keep):
            idx[j] = i
        self.idx = idx
        self.constants = n - self.nc
        seen_out: set[int] = set()
        self.outputs = [idx[g] for g in returned if not (g in seen_out or seen_out.add(g))]
        ids = _Interner()
        tok = [0] * n
        first: dict[tuple, int] = {}
        recomputed: list[tuple[int, int]] = []
        reads: list[tuple[int, int]] = []
        for j in range(n):
            if not computing[j]:
                tok[j] = ids(("c", prim[j]))
                continue
            k = (prim[j],) + tuple(ids(("i", lo + o)) if o < 0 else tok[o] for o in ops[j])
            if k in first:
                recomputed.append((idx[j], idx[first[k]]))
                tok[j] = tok[first[k]]
            else:
                first[k] = j
                tok[j] = ids(("g", j))
            q = idx[j]
            got: set[int] = set()
            for o in ops[j]:
                if o in got or (o >= 0 and not computing[o]):
                    continue
                got.add(o)
                reads.append((self.nc + lo + o if o < 0 else idx[o], q))
        self.recomputed, self.reads = recomputed, reads

    def widest(self, X: int) -> int | None:
        """The first computing gate wider than X bits (local), or None."""
        return next((j for j in self.keep if self.width[j] > X), None)

    def w_out(self, lo: int, hi: int, limit: int | None = None) -> int:
        """The summed widths of the piece `[lo, hi)`'s outputs (step 3), scanning down from `hi`; past `limit`, stops early."""
        w, comp, width, last = 0, self.computing, self.width, self.last
        for g in range(hi - 1, lo - 1, -1):
            if comp[g] and last[g] >= hi:
                w += width[g]
                if limit is not None and w > limit:
                    return w
        return w

    def any_computing(self, lo: int, hi: int) -> bool:
        return any(self.computing[lo:hi])


class CallCut:
    """`Q_call{X}`'s cut of one Definition: `units` [(lo, hi, path)] over its local gates (`CallGates`), and `owner()`, the unit of
    each computing gate in compact order."""

    __slots__ = ("fn", "units", "_owner")

    def __init__(self, fn: Any, units: list[tuple[int, int, tuple]], owner: list[int]) -> None:
        self.fn, self.units, self._owner = fn, units, owner

    def owner(self) -> list[int]:
        return list(self._owner)


def call_gates(fn: Any, memo: dict | None = None) -> CallGates:
    """`CallGates(fn)`, memoised by Definition id."""
    memo = {} if memo is None else memo
    key = ("Q_call/gates", fn.id)
    got = memo.get(key)
    if got is None:
        got = memo[key] = CallGates(fn)
    return got


def evaluate_call(fn: Any, X: int = CALL_X, memo: dict | None = None) -> CallCut:
    """`Q_call{X}` on one activation of `fn` (steps 0, 2-4): `QueryInapplicable` (`gate-too-wide`) when a computing gate is wider than
    X bits; the proof blocks are the term's sets (`query_ast.frontier_pieces`), each computing gate in exactly one.  Memoised by
    Definition id."""
    memo = {} if memo is None else memo
    key = ("Q_call", fn.id, X)
    got = memo.get(key)
    if got is not None:
        return got
    G = call_gates(fn, memo)
    j = G.widest(X)
    if j is not None:
        raise QueryInapplicable(f"{fn.id}: gate {j} ({G.prim[j]}) is {G.width[j]} bits, wider than Q_call's X = {X}",
                                {"definition": fn.id, "gate": j, "primitive": G.prim[j], "width": G.width[j], "X": X, "why": "gate-too-wide"})
    units = frontier_pieces(fn, X, memo)
    owner = [-1] * G.nc
    for u, (lo, hi, _path) in enumerate(units):
        for g in range(lo, hi):
            i = G.idx[g]
            if i >= 0:
                if owner[i] >= 0:
                    raise ValueError(f"{fn.id}: computing gate {g} is in units {owner[i]} and {u}")
                owner[i] = u
    gaps = [G.keep[i] for i, u in enumerate(owner) if u < 0]
    if gaps:
        raise ValueError(f"{fn.id}: computing gates {gaps[:8]} are in no unit ({len(gaps)} in all)")
    r = memo[key] = CallCut(fn, units, owner)
    return r


def call_committed(G: CallGates, cut: CallCut) -> list[int]:
    """The derived committed set (step 6), compact, ascending: computing gates read after their unit's interval, or returned."""
    out = []
    for lo, hi, _path in cut.units:
        for g in range(lo, hi):
            if G.computing[g] and G.last[g] >= hi:
                out.append(G.idx[g])
    return sorted(out)


def check_call_cut(G: CallGates, owner: Sequence[int], X: int, committed: Iterable[int] | None = None) -> CutVerdict:
    """One Call's `Q_call{X}` cut against its gates (step 6), from the reads alone: `validate_unit_cut` with recomputes reported, and
    `unit-too-wide` ((unit, out bits) per unit whose boundary -- gates another unit reads, or returned -- exceeds X bits).
    `committed`: what serving commits (compact numbering); None: the boundary the reads derive."""
    owner = [int(u) for u in owner]
    if len(owner) != G.nc:
        return CutVerdict(False, ("owner-length",), {"owner-length": {"n": 1, "gates": [len(owner), G.nc]}})
    k, nc = G.inputs, G.nc
    full = owner + [INPUT_UNIT] * k
    boundary = {p for p, q in G.reads if p < nc and full[p] != full[q]} | set(G.outputs)
    if committed is None:
        committed = boundary
    read = sorted({p for p, _q in G.reads if p >= nc})                     # the parameter leaves a unit reads, under the input unit's roots
    cut = validate_unit_cut(G.n + k, G.constants, full, G.reads, list(G.outputs),
                            sorted({int(g) for g in committed}) + read, range(nc, nc + k), G.recomputed, "report")
    ob: Counter = Counter()
    for g in boundary:
        ob[owner[g]] += G.width[G.keep[g]]
    wide = sorted([u, b] for u, b in ob.items() if b > X)
    if not wide:
        return cut
    detail = dict(cut.detail)
    detail["unit-too-wide"] = {"n": len(wide), "gates": wide[:64], "limits": [X]}
    return CutVerdict(False, tuple(cut.codes) + ("unit-too-wide",), detail)


__all__ = ["WIRING", "WIDE_FAN_IN", "WIDTH_RULE", "RECOMPUTE", "QueryInapplicable", "call_scope", "member_refs", "wide_operands", "CallGraph", "fits", "boundary_widths", "derived_committed", "check_cut",
           "separable", "targets", "owners", "cut_word", "DefinitionCut", "evaluate_definition",
           "CALL_X", "CallGates", "call_gates", "CallCut", "evaluate_call", "call_committed", "check_call_cut"]
