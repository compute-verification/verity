"""census/v1: what a `Program` computes, counted by class, and the values its root's Calls pass each other.

A census is a pure function of a Program's IR, written as canonical JSON (`codec.canonical_json`, RFC 8785 on its integer domain)
under a SHA-512 digest. Classes come from Definitions and primitives only: a class is a descriptor id (`codec.descriptor_id`, the
family `Name_vN` with its static bindings), never a kernel.

The root's body is the Program's step structure. Its first `inputs` nodes are the input nodes (`batch(Input<w>_v1 |
RegisteredInput<w>_v1, n)`), and every later node is a Call (an activation directly under the root), in execution order. A step is a
run of consecutive Calls: `ends` are the root-node indices that end the steps (exclusive), so step `s` is root nodes
`[ends[s - 1], ends[s])`, with `ends[-1]` read as `inputs`. An engine step's Program is one step (the default); a request Program's
steps end after each sampling event (`step_ends`). Inputs belong to no step.

    {"schema": "verity/census/v1",
     "program": <the Program's digest (`codec.program_digest`)> | null,
     "classes": {<descriptor id>: {
         "family": "Name_vN",
         "statics": {<static>: <its descriptor JSON>},           # {} for a primitive
         "kind": "composite" | "primitive" | "input",
         "binding": "run" | "registration" | null,              # an input's binding time
         "instances": <activations in the Program>,               # a primitive's: its gates
         "gates": <gates per instance>,
         "primitives": {<primitive id>: <gates per instance>},
         "served_macs": <per instance>, "tail_atoms": <per instance>, "receipt_folds": <per instance>,
         "served_macs_by_precision": {"mac:<precision>": <per instance>},
         "ports": <its descriptor entry's `ports` (`ports.declared`)>}},  # only on a class that declares ports
     "callees": [<descriptor id>],                                # the classes of the root's nodes
     "inputs": <the number of input nodes>,
     "nodes": {"callee": [<index into callees>], "n": [<activations: a batch's members, a scan's iterations, else 1>]},
     "steps": {"ends": [...], "gates": [...], "served_macs": [...], "tail_atoms": [...], "receipt_folds": [...],
               "served_macs_by_precision": [{"mac:<precision>": ...}], "select_gates": [...], "select_bits": [...]},
     "values": {"producer": [<root node>], "lo": [...], "hi": [...], "bits": [...], "reads": [[<root node>]], "output": [<bool>],
                "fact": [<index into facts>]},
     "facts": [{"interval": [a, b] | null, "origin": [<origin>], "sink": <bool>, "word": <word type> | null}],
     "selects": {"call": [<root node>], "gates": [...], "bits": [...], "values": [[<value>]]},
     "totals": {"gates", "computed_gates", "inputs": {"run" | "registration": {"gates", "bits"}}, "values", "value_bits",
                "classes", "served_macs", "tail_atoms", "receipt_folds", "served_macs_by_precision", "select_gates", "select_bits"},
     "approximate": <bool>,
     "digest": <SHA-512 of the canonical JSON of every other field>}

Every column of `nodes`, `steps`, `values` and `selects` has one entry per root node, step, value or selecting Call.

**Counts.** `instances` is top-down (a `batch` or `scan` node multiplies by its length); `gates`, `primitives` and the three
counters are bottom-up, per instance. An instance of a tensor-core step (a word primitive, or the Boolean Definition computing it)
performs `PRODUCTS[family]` products, and every one counts as a served MAC. A family named in `FAMILY_COUNTS` takes its
counters from its rule over its statics instead: that is how a matmul family states
its served MACs (m·n·k) apart from its tail atoms (k-steps appended to the same product) and a fold Definition its receipt folds; the
class's ancestors add up what the rule says. A class of a family `RULED` names is refused while `FAMILY_COUNTS` has no rule for it
(the family's module registers it when imported, which decoding a Program never does), and so is a Definition carrying a rule
(`.counts`) that `FAMILY_COUNTS` lacks: neither is counted by `PRODUCTS`. `served_macs_by_precision` splits the served MACs by the
operand format of the steps
performing them (`PRECISION[family]`), keyed by the accounting layer's kinds (`mac:bf16`, `mac:int8`, ...), and adds up to
`served_macs` in every class, step and the totals; a rule's served MACs take the one precision of the steps under it.

**Values.** A value is a run `[lo, hi)` of one root node's output leaves whose readers, candidacy, output status and fact are the
same throughout. `bits` is the sum of its leaves' widths. `reads` are the Calls with a gate that takes one of its leaves as an operand
(a direct read, which no value decides), and `output` says the root returns it. A value nobody reads is listed with no reads. A
leaf a node passes through (a Call that returns one of its arguments, a scan's carry) is not that node's value: whoever reads it
reads the value passed, and its producer's.

**Facts.** A value's fact is what `ports.facts(root)` knows of its leaves (PROTOCOL.md §12): an interval (null: it spans its
width), its origins, whether it is a sink, and its word type. An input's interval is its port's declared bound, which evaluation
doesn't enforce, so no reader takes it (or a transfer's from it) on trust; a constant's is its value; a computed value's is what
its producer declares or its transfer gives. `facts` lists each distinct fact once, in the order the
values first name it. A constant wider than the canonical JSON's integers is written with no interval (every declared interval
is within them). A Program in which no Definition declares ports has only the default facts.

**Selects.** A primitive with `evaluate_lazy` reads its scalar operands directly and selects among its array operands by a value
(a gather, a table read, an indexed expert): those array operands are candidates, not reads. Per Call with such gates, `gates`
counts them, `bits` sums the widths they return (what a selection moves), and `values` are the root's values among their candidates
(a selection among values computed inside the Call names none).

**Approximation.** A batch whose members read different parts of a mapped argument, too many to enumerate
(`liveness._MEMBER_LIMIT`), is taken to read all of it; `approximate` is then true and `reads` a superset.
"""

from __future__ import annotations

import bisect
import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

from . import ports
from .codec import MAX_SAFE_INT, _static_json, canonical_json, descriptor_id, jcs_domain_error
from .defs import Node, PrimitiveDefinition
from .intervals import Intervals, complement, contains, full, isect, norm, ref_intervals, total
from .layout import resolve
from .liveness import _MEMBER_LIMIT
from .program import input_id, is_input
from .refs import Affine, Concat, Explicit, Refs, concat
from .types import Array, Record, Tuple as TupleT, Type, Value

SCHEMA = "verity/census/v1"

#: products an instance of each tensor-core step family performs (its k): the word primitives, and the Boolean Definitions that
#: compute them (`ml.circuits.ops.boolean.gemm`, `ml.circuits.ops.boolean.s8`).  The s8 IMMA steps are one family in two arms,
#: s8 x s8 clamped to s32 and u8 x s8 wrapping, 32 products each; the s8 matmul Definitions over them name the arm in a static and
#: count by their `FAMILY_COUNTS` rule
PRODUCTS: dict[str, int] = {
    "AmpereBF16TcDot16_v1": 16, "AmpereBF16TcDot16_v2": 16, "HopperBF16WgmmaDot16_v1": 16, "HopperTf32WgmmaDot8_v1": 8,
    "HopperE4m3QgmmaDot32_v1": 32, "HopperE4m3MmaV2Dot32_v1": 32, "BlackwellE4m3QmmaDot32_v1": 32,
    "BlackwellE5m2QmmaDot32_v1": 32, "BlackwellNvf4OmmaDot64_v1": 64, "BlackwellMxf4OmmaDot64_v1": 64,
    "PouwDot16U8S8_v1": 16, "NcpDot16I7_v1": 16,
    "AmpereBF16TcDot16_v3": 16, "HopperBF16WgmmaDot16_v2": 16, "HopperE4m3QgmmaDot32_v2": 32, "HopperE4m3MmaV2Dot32_v2": 32,
    "BlackwellE4m3QmmaDot32_v2": 32, "BlackwellE5m2QmmaDot32_v2": 32, "BlackwellNvf4OmmaDot64_v2": 64,
    "BlackwellMxf4OmmaDot64_v2": 64,
    "BlackwellS8ImmaDot32_v1": 32, "BlackwellU8S8ImmaDot32_v1": 32, "BlackwellS8ImmaDot32_v2": 32, "BlackwellU8S8ImmaDot32_v2": 32,
}

#: the operand format of each `PRODUCTS` family's products, as the accounting layer names its multiply-add kinds (`mac:<precision>`):
#: fp8 is E4M3 and E5M2, fp4 NVFP4 and MXFP4, and int8 the s8 and u8 x s8 IMMA arms, PoUW's u8 x s8 step and NCP's i7 step
PRECISION: dict[str, str] = {
    **dict.fromkeys(("AmpereBF16TcDot16_v1", "AmpereBF16TcDot16_v2", "AmpereBF16TcDot16_v3", "HopperBF16WgmmaDot16_v1",
                     "HopperBF16WgmmaDot16_v2"), "bf16"),
    "HopperTf32WgmmaDot8_v1": "tf32",
    **dict.fromkeys(("HopperE4m3QgmmaDot32_v1", "HopperE4m3QgmmaDot32_v2", "HopperE4m3MmaV2Dot32_v1", "HopperE4m3MmaV2Dot32_v2",
                     "BlackwellE4m3QmmaDot32_v1", "BlackwellE4m3QmmaDot32_v2", "BlackwellE5m2QmmaDot32_v1",
                     "BlackwellE5m2QmmaDot32_v2"), "fp8"),
    **dict.fromkeys(("BlackwellNvf4OmmaDot64_v1", "BlackwellNvf4OmmaDot64_v2", "BlackwellMxf4OmmaDot64_v1",
                     "BlackwellMxf4OmmaDot64_v2"), "fp4"),
    **dict.fromkeys(("PouwDot16U8S8_v1", "NcpDot16I7_v1", "BlackwellS8ImmaDot32_v1", "BlackwellU8S8ImmaDot32_v1",
                     "BlackwellS8ImmaDot32_v2", "BlackwellU8S8ImmaDot32_v2"), "int8"),
}

COUNTERS = ("served_macs", "tail_atoms", "receipt_folds")

#: family `Name_vN` -> its counters per instance from its statics (a subset of `COUNTERS`; the rest stay bottom-up)
FAMILY_COUNTS: dict[str, Callable[[dict], dict[str, int]]] = {}

#: the families that count by a `FAMILY_COUNTS` rule -> the module that registers it when imported (the IR never imports a
#: semantics package, and decoding a Program imports none)
RULED: dict[str, str] = {"S8Dot_v1": "verity.ml.circuits.ops.s8", "S8Matmul_v1": "verity.ml.circuits.ops.s8",
                         "S8Dot_v2": "verity.ml.circuits.ops.boolean.s8"}


def family(fn: Any) -> str:
    """The family `Name_vN` of a Definition or primitive."""
    return fn.id if isinstance(fn, PrimitiveDefinition) else fn.base_id


def seal(doc: dict) -> dict:
    """`doc` with its `digest`: the SHA-512 of the canonical JSON of every other field (refused outside that JSON's domain)."""
    body = {k: v for k, v in doc.items() if k != "digest"}
    err = jcs_domain_error(body)
    if err is not None:
        raise ValueError(f"{doc.get('schema')}: outside the canonical JSON domain: {err}")
    return body | {"digest": hashlib.sha512(canonical_json(body)).hexdigest()}


def step_ends(program: Any, families: Iterable[str]) -> list[int]:
    """Step ends that close a step after each root Call of one of `families` (a request Program's sampling events), and after the
    last node."""
    fams = set(families)
    nodes = program.root.body.nodes
    ends = [k + 1 for k, nd in enumerate(nodes) if not is_input(nd.fn) and family(nd.fn) in fams]
    if nodes and (not ends or ends[-1] != len(nodes)):
        ends.append(len(nodes))
    return ends


# -- reads -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reads:
    """What one activation of a Definition reads of its parameters, per parameter."""
    direct: tuple[Intervals, ...]
    pool: tuple[Intervals, ...]  # candidates of its data-dependent gates
    select_gates: int
    select_bits: int
    approximate: bool


class _Marks:
    """Leaf intervals one body's nodes read, in one space of that body (`p` inside a Definition, `n` at the root)."""

    def __init__(self, space: str) -> None:
        self.space = space
        self.direct: dict[int, list] = {}
        self.pool: dict[int, list] = {}
        self.select_gates = self.select_bits = 0
        self.approximate = False

    def walk(self, refs: Refs, into: dict) -> None:
        if isinstance(refs, Concat):
            for p in refs.parts:
                self.walk(p, into)
        elif isinstance(refs, Explicit):
            for s, i, leaf in refs.items:
                if s == self.space:
                    into.setdefault(i, []).append((leaf, leaf + 1))
        elif refs.count and refs.space == self.space:
            t: dict = {}
            ref_intervals(refs, t)
            into.setdefault(refs.idx, []).extend(t[(refs.space, refs.idx)])

    def mark(self, refs: Refs, sub: Intervals | None, into: dict) -> None:
        if sub is None or total(sub) == refs.count:
            self.walk(refs, into)
        else:
            for lo, hi in sub:
                self.walk(refs.slice(lo, hi), into)

    def mapped(self, a, ax: int | None, n: int, sub: Intervals, into: dict) -> None:
        """Mark what `n` members (a batch's members, a scan's iterations) each reading `sub` of their part of `a` read of `a`."""
        if not sub or not n:
            return
        if ax is None:
            self.mark(a.refs, sub, into)
            return
        L = a.type.leaves // n
        if total(sub) == L:
            self.walk(a.refs, into)
        elif n * len(sub) > _MEMBER_LIMIT:
            self.walk(a.refs, into)
            self.approximate = True
        elif ax == 0:
            for m in range(n):
                for lo, hi in sub:
                    self.walk(a.refs.slice(m * L + lo, m * L + hi), into)
        else:
            for m in range(n):
                self.mark(a.take(ax, m).refs, sub, into)

    def reads(self, params: int) -> _Reads:
        return _Reads(tuple(norm(self.direct.get(i, ())) for i in range(params)),
                      tuple(norm(self.pool.get(i, ())) for i in range(params)), self.select_gates, self.select_bits, self.approximate)


def _flat(r: Refs) -> list[Refs]:
    return [q for p in r.parts for q in _flat(p)] if isinstance(r, Concat) else [r]


def _through(r: Refs, args) -> Refs:
    """Refs over a callee's parameters, as the caller's refs its arguments name."""
    parts: list[Refs] = []
    for p in _flat(r):
        if isinstance(p, Affine):
            parts.append(args[p.idx].refs.slice(p.base, p.base + p.count))
        else:
            parts.extend(args[i].refs.slice(leaf, leaf + 1) for _s, i, leaf in p)
    return concat(parts)


class _Aliases:
    """The output leaves of one body's nodes that are leaves of something else passed through (a callee returning its argument,
    a scan's carry), and what refs resolve to once every pass-through is followed."""

    def __init__(self, an: "_Analysis", body: Any) -> None:
        self.an, self.body = an, body
        self._nodes: dict[int, list[tuple[int, int, Refs]]] = {}

    def node(self, k: int) -> list[tuple[int, int, Refs]]:
        """Runs `[lo, hi)` of node `k`'s output leaves that pass through the refs (in this body's spaces) given with each, by `lo`."""
        hit = self._nodes.get(k)
        if hit is not None:
            return hit
        nd = self.body.nodes[k]
        sub = self.an.alias(nd.fn)
        runs: list[tuple[int, int, Refs]] = []
        if not sub:
            pass
        elif nd.form == "call":
            runs = [(lo, hi, _through(r, nd.args)) for lo, hi, r in sub]
        elif nd.form == "batch":
            L = nd.fn.out_leaves
            for m in range(nd.n):
                args = [a if ax is None else a.take(ax, m) for a, ax in zip(nd.args, nd.axes)]
                runs.extend((m * L + lo, m * L + hi, _through(r, args)) for lo, hi, r in sub)
        else:  # scan: iteration i's carry parameter is the init (i = 0) or the carry iteration i - 1 returned
            c, y = nd.out.fields[0].leaves, nd.out.fields[1].elem.leaves
            amap = {lo + t: r[t] for lo, hi, r in sub for t in range(hi - lo)}
            src: list = [nd.args[0].refs[t] for t in range(c)]
            for i in range(nd.n):
                def ref(pi: int, leaf: int, i: int = i, src: list = src):
                    if pi == 0:
                        return src[leaf]
                    a = nd.args[pi]
                    return a.refs[i * a.type.elem.leaves + leaf] if pi <= nd.n_xs else a.refs[leaf]
                for t in range(c, c + y):
                    if t in amap and (r := ref(amap[t][1], amap[t][2])) is not None:
                        runs.append((c + i * y + t - c, c + i * y + t - c + 1, Explicit((r,))))
                src = [ref(amap[t][1], amap[t][2]) if t in amap else None for t in range(c)]
            runs.extend((t, t + 1, Explicit((r,))) for t, r in enumerate(src) if r is not None)
            runs.sort(key=lambda x: x[0])
        self._nodes[k] = runs
        return runs

    def resolve(self, r: Refs, own: bool) -> list[tuple[int, Refs, bool]]:
        """`r` with every pass-through followed, as pieces `(position, refs, passed)`: `passed` pieces are the body's parameters,
        the others leaves its nodes produce (listed only when `own`)."""
        out: list[tuple[int, Refs, bool]] = []
        pos = 0
        for p in _flat(r):
            n = p.count
            if isinstance(p, Explicit):
                for t, (s, i, leaf) in enumerate(p.items):
                    self._leaf(s, i, leaf, pos + t, own, out)
            elif p.space == "p":
                if n:
                    out.append((pos, p, True))
            elif not (runs := self.node(p.idx)):
                if own and n:
                    out.append((pos, p, False))
            elif isinstance(p, Affine):
                b, e, cur = p.base, p.base + n, p.base
                for lo, hi, s in runs:
                    if hi <= cur or lo >= e:
                        continue
                    if lo > cur and own:
                        out.append((pos + cur - b, Affine("n", p.idx, cur, lo - cur), False))
                    a0, a1 = max(lo, cur), min(hi, e)
                    out.extend((pos + a0 - b + q, x, f) for q, x, f in self.resolve(s.slice(a0 - lo, a1 - lo), own))
                    cur = a1
                if cur < e and own:
                    out.append((pos + cur - b, Affine("n", p.idx, cur, e - cur), False))
            else:
                for t, (s, i, leaf) in enumerate(p):
                    self._leaf(s, i, leaf, pos + t, own, out)
            pos += n
        return out

    def _leaf(self, s: str, i: int, leaf: int, pos: int, own: bool, out: list) -> None:
        if s == "p":
            out.append((pos, Affine(s, i, leaf, 1), True))
            return
        runs = self.node(i)
        j = bisect.bisect_right(runs, leaf, key=lambda x: x[0]) - 1
        if j >= 0 and runs[j][0] <= leaf < runs[j][1]:
            lo, _, r = runs[j]
            out.extend((pos + q, x, f) for q, x, f in self.resolve(r.slice(leaf - lo, leaf - lo + 1), own))
        elif own:
            out.append((pos, Affine(s, i, leaf, 1), False))


class _Analysis:
    def __init__(self) -> None:
        self._reads: dict[int, tuple[Any, _Reads]] = {}  # id(fn) -> (fn, reads): the object keeps the id valid
        self._alias: dict[int, tuple[Any, tuple]] = {}
        self._repeats: dict[int, tuple[Any, bool]] = {}

    def alias(self, fn: Any) -> tuple[tuple[int, int, Refs], ...]:
        """Runs `[lo, hi)` of `fn`'s return leaves that are its parameters' leaves passed through, with those leaves (refs over
        `p`), by `lo`."""
        if isinstance(fn, PrimitiveDefinition):
            return ()
        hit = self._alias.get(id(fn))
        if hit is not None:
            return hit[1]
        got = tuple((q, q + x.count, x) for q, x, passed in _Aliases(self, fn.body).resolve(fn.body.ret, False) if passed)
        self._alias[id(fn)] = (fn, got)
        return got

    def repeats(self, fn: Any) -> bool:
        """Whether `fn`'s return names a gate it produces twice (a scan step's carry that is also its y)."""
        if isinstance(fn, PrimitiveDefinition):
            return False
        hit = self._repeats.get(id(fn))
        if hit is not None:
            return hit[1]
        t: dict = {}
        named: Counter = Counter()
        for _q, x, passed in _Aliases(self, fn.body).resolve(fn.body.ret, True):
            if not passed:
                ref_intervals(x, t)
                named[x.idx] += x.count
        got = any(named[k] > total(norm(t[("n", k)])) or self.repeats(fn.body.nodes[k].fn) for k in named)
        self._repeats[id(fn)] = (fn, got)
        return got

    def reads(self, fn: Any) -> _Reads:
        hit = self._reads.get(id(fn))
        if hit is not None:
            return hit[1]
        if isinstance(fn, PrimitiveDefinition):
            lazy = getattr(fn, "evaluate_lazy", None) is not None
            cand = [lazy and not isinstance(t, Value) for _, t in fn.params]
            r = _Reads(tuple(() if c else full(t.leaves) for c, (_, t) in zip(cand, fn.params)),
                       tuple(full(t.leaves) if c else () for c, (_, t) in zip(cand, fn.params)),
                       int(lazy), fn.ret.w if lazy else 0, False)
        else:
            m = _Marks("p")
            for node in fn.body.nodes:
                self.node(node, m)
            r = m.reads(len(fn.body.params))
        self._reads[id(fn)] = (fn, r)
        return r

    def node(self, node: Node, m: _Marks) -> None:
        r = self.reads(node.fn)
        n = node.n if node.form in ("batch", "scan") else 1
        m.select_gates += n * r.select_gates
        m.select_bits += n * r.select_bits
        m.approximate |= r.approximate
        for got, into in ((r.direct, m.direct), (r.pool, m.pool)):
            if node.form in ("primitive", "call"):
                for a, sub in zip(node.args, got):
                    m.mark(a.refs, sub, into)
            elif node.form == "batch":
                for a, ax, sub in zip(node.args, node.axes, got):
                    m.mapped(a, ax, node.n, sub, into)
            else:  # scan: iteration 0 reads the init as its carry; the others read the carry the one before returned
                k = node.n_xs
                m.mark(node.args[0].refs, got[0], into)
                for a, sub in zip(node.args[1:1 + k], got[1:1 + k]):
                    m.mapped(a, 0, node.n, sub, into)
                for a, sub in zip(node.args[1 + k:], got[1 + k:]):
                    m.mark(a.refs, sub, into)


# -- classes ---------------------------------------------------------------------------------------------------------


def _kind(fn: Any) -> tuple[str, str | None]:
    if isinstance(fn, PrimitiveDefinition):
        when = input_id(fn.id)
        return ("input", when[0]) if when else ("primitive", None)
    return "composite", None


def _rule(fn: Any) -> dict[str, int] | None:
    rule = FAMILY_COUNTS.get(family(fn))
    if rule is None:
        if family(fn) in RULED:
            raise ValueError(f"{descriptor_id(fn)} counts by its family's rule, which census.FAMILY_COUNTS lacks: import "
                             f"{RULED[family(fn)]}, which registers it, before taking the census")
        if hasattr(getattr(fn, "defn", None), "counts"):
            raise ValueError(f"{family(fn)} has a census rule (`.counts`) that census.FAMILY_COUNTS and census.RULED don't name: "
                             f"register it in both")
        return None
    got = rule(dict(fn.bindings))
    bad = {k: v for k, v in got.items() if k not in COUNTERS or not isinstance(v, int) or isinstance(v, bool) or v < 0}
    if bad:
        raise ValueError(f"FAMILY_COUNTS[{family(fn)!r}] gives {bad} for {descriptor_id(fn)}: a rule gives non-negative ints for "
                         f"{', '.join(COUNTERS)}")
    return got


def _step_macs(fam: str) -> Counter:
    """A tensor-core step's served MACs by precision (none for any other family)."""
    if fam not in PRODUCTS:
        return Counter()
    if fam not in PRECISION:
        raise ValueError(f"census.PRODUCTS names {fam} with no census.PRECISION: add its operand format")
    return Counter({f"mac:{PRECISION[fam]}": PRODUCTS[fam]})


def _ruled_macs(fn: Any, below: Counter, served: int) -> Counter:
    """A family rule's `served` MACs by precision: at the one precision of the steps under its class."""
    if sum(below.values()) == served:
        return below
    if served and len(below) != 1:
        raise ValueError(f"FAMILY_COUNTS[{family(fn)!r}] gives {served} served MACs for {descriptor_id(fn)} over steps at "
                         f"{sorted(below) or 'no precision'}: a rule's served MACs take the one precision of the steps under it")
    return Counter({q: served for q in below} if served else {})


def _classes(root: Any) -> tuple[dict[str, dict], dict[int, str], dict[int, tuple[int, ...]]]:
    """The classes under `root`: their entries, the descriptor id by `id(fn)`, and each class's counters per instance."""
    fns: dict[int, Any] = {}
    kids: dict[int, Counter] = {}
    post: list[Any] = []
    seen: set[int] = set()
    stack: list = [(root, None)]
    while stack:
        fn, it = stack.pop()
        if it is None:
            if id(fn) in seen:
                continue
            seen.add(id(fn))
            if isinstance(fn, PrimitiveDefinition):
                post.append(fn)
                continue
            c = kids[id(fn)] = Counter()
            for nd in fn.body.nodes:
                fns[id(nd.fn)] = nd.fn
                c[id(nd.fn)] += nd.n if nd.form in ("batch", "scan") else 1
            it = iter(list(c))
        nxt = next(it, None)
        if nxt is None:
            post.append(fn)
        else:
            stack.append((fn, it))
            stack.append((fns[nxt], None))

    ids: dict[int, str] = {}
    prims: dict[int, Counter] = {}
    counts: dict[int, tuple[int, ...]] = {}
    macs: dict[int, Counter] = {}
    for fn in post:
        i = id(fn)
        if isinstance(fn, PrimitiveDefinition):
            prims[i] = Counter({fn.id: 1})
            counts[i] = (PRODUCTS.get(fn.id, 0), 0, 0)
            macs[i] = _step_macs(fn.id)
        else:
            p: Counter = Counter()
            m: Counter = Counter()
            c = [0, 0, 0]
            for k, mult in kids[i].items():
                for pid, g in prims[k].items():
                    p[pid] += mult * g
                for q, n in macs[k].items():
                    m[q] += mult * n
                for j in range(3):
                    c[j] += mult * counts[k][j]
            if family(fn) in PRODUCTS:
                c[0], m = PRODUCTS[family(fn)], _step_macs(family(fn))
            rule = _rule(fn) if fn is not root else None
            if rule:
                c = [rule.get(name, c[j]) for j, name in enumerate(COUNTERS)]
                m = _ruled_macs(fn, m, c[0])
            prims[i], counts[i], macs[i] = p, tuple(c), m
        if fn is not root:
            ids[i] = descriptor_id(fn)

    inst: Counter = Counter({id(root): 1})
    for fn in reversed(post):
        for k, mult in kids.get(id(fn), {}).items():
            inst[k] += inst[id(fn)] * mult

    classes: dict[str, dict] = {}
    for fn in post:
        if fn is root:
            continue
        cid = ids[id(fn)]
        if cid in classes:  # two objects under one id are one definition (`E_IDENTITY` refuses anything else)
            classes[cid]["instances"] += inst[id(fn)]
            continue
        kind, binding = _kind(fn)
        statics = {} if isinstance(fn, PrimitiveDefinition) else {k: _static_json(fn.bindings[k]) for k in fn.defn.statics}
        classes[cid] = {"family": family(fn), "statics": statics, "kind": kind, "binding": binding, "instances": inst[id(fn)],
                        "gates": fn.gates, "primitives": dict(sorted(prims[id(fn)].items())),
                        **dict(zip(COUNTERS, counts[id(fn)])), "served_macs_by_precision": dict(sorted(macs[id(fn)].items()))}
        if (decl := ports.declared(fn)) is not None:
            classes[cid]["ports"] = decl
    return classes, ids, counts


# -- values ----------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _width(t: Type) -> int | None:
    return t.homogeneous_width()


@lru_cache(maxsize=None)
def _all_bits(t: Type) -> int:
    return _bits(t, 0, t.leaves)


def _bits(t: Type, lo: int, hi: int) -> int:
    """The summed widths of leaves `[lo, hi)` of `t`, computed structurally."""
    if hi <= lo:
        return 0
    w = _width(t)
    if w is not None:
        return (hi - lo) * w
    if isinstance(t, Array):
        L = t.elem.leaves
        a, b = lo // L, (hi - 1) // L
        if a == b:
            return _bits(t.elem, lo - a * L, hi - a * L)
        return _bits(t.elem, lo - a * L, L) + (b - a - 1) * _all_bits(t.elem) + _bits(t.elem, 0, hi - b * L)
    fields = t.fields if isinstance(t, TupleT) else [ft for _, ft in t.fields] if isinstance(t, Record) else None
    if fields is None:
        raise TypeError(t)
    out, pos = 0, 0
    for ft in fields:
        n = ft.leaves
        if pos + n > lo and pos < hi:
            out += _bits(ft, max(lo, pos) - pos, min(hi, pos + n) - pos)
        pos += n
    return out


def _minus(a: Intervals, b: list[tuple[int, int]]) -> Intervals:
    n = max((hi for _, hi in a), default=0)
    return isect(a, tuple(complement(norm((max(lo, 0), min(hi, n)) for lo, hi in b), n)))


def _segments(L: int, spans: list[tuple[int, int, tuple]]) -> Iterable[tuple[int, int, frozenset]]:
    """`[0, L)` cut into maximal runs over which the set of tags of the spans covering a leaf is the same."""
    ev: list[tuple[int, int, tuple]] = []
    for lo, hi, tag in spans:
        ev.append((lo, 1, tag))
        ev.append((hi, 0, tag))
    ev.sort(key=lambda e: (e[0], e[1]))
    active: Counter = Counter()
    pos, i, run = 0, 0, None
    while pos < L:
        while i < len(ev) and ev[i][0] <= pos:
            _, start, tag = ev[i]
            active[tag] += 1 if start else -1
            if not active[tag]:
                del active[tag]
            i += 1
        nxt = min(ev[i][0] if i < len(ev) else L, L)
        tags = frozenset(active)
        if run is not None and run[2] == tags:
            run = (run[0], nxt, tags)
        else:
            if run is not None:
                yield run
            run = (pos, nxt, tags)
        pos = nxt
    if run is not None:
        yield run


def _fact_json(f: ports.Fact) -> dict:
    j = f.to_json()
    if f.interval is not None and max(-f.interval[0], f.interval[1]) > MAX_SAFE_INT:
        j["interval"] = None
    return j


def census(program: Any, ends: list[int] | None = None, digest: str | None = None) -> dict:
    """The census/v1 of `program` (a `Program` or a `DecodedProgram`) over the steps `ends` (default: one step), naming the
    Program by `digest` (`codec.program_digest`) when given."""
    root = program.root
    nodes = root.body.nodes
    I = sum(1 for nd in nodes if is_input(nd.fn))
    if any(not is_input(nd.fn) for nd in nodes[:I]):
        raise ValueError("a root's input nodes come first (PROTOCOL.md §5.2)")
    ends = list(ends) if ends is not None else ([len(nodes)] if len(nodes) > I else [])
    if any(b <= a for a, b in zip([I] + ends, ends)) or (ends and ends[-1] != len(nodes)) or (not ends and len(nodes) > I):
        raise ValueError(f"step ends {ends[:8]} are not increasing root-node indices past the {I} input nodes ending at {len(nodes)}")

    classes, ids, counts = _classes(root)
    callees = sorted({ids[id(nd.fn)] for nd in nodes})
    at = {c: i for i, c in enumerate(callees)}
    mult = [nd.n if nd.form in ("batch", "scan") else 1 for nd in nodes]

    an = _Analysis()
    spans: list[list[tuple[int, int, tuple]]] = [[] for _ in nodes]
    sel: dict[int, tuple[int, int]] = {}
    approximate = False
    for k in range(I, len(nodes)):
        m = _Marks("n")
        an.node(nodes[k], m)
        approximate |= m.approximate
        for tag, got in ((0, m.direct), (1, m.pool)):
            for j, ivs in got.items():
                spans[j].extend((lo, hi, (tag, k)) for lo, hi in norm(ivs))
        if m.select_gates:
            sel[k] = (m.select_gates, m.select_bits)
    out = _Marks("n")
    out.walk(root.body.ret, out.direct)
    for j, ivs in out.direct.items():
        spans[j].extend((lo, hi, (2, 0)) for lo, hi in norm(ivs))
    al = _Aliases(an, root.body)
    for j in range(len(nodes) - 1, I - 1, -1):  # a leaf a node passes through is its producer's value: its readers read that
        runs = [(lo + q, lo + q + x.count, x) for lo, _hi, s in al.node(j) for q, x, _ in al.resolve(s, True)]
        if an.repeats(nodes[j].fn):  # and a leaf naming the gate an earlier one names is that one
            passed = norm((lo, hi) for lo, hi, _ in runs)
            first: dict[int, int] = {}
            for t in range(nodes[j].out.leaves):
                if not contains(passed, t):
                    g = first.setdefault(resolve(program.circuit.root_scope, ("n", j, t)), t)
                    if g != t:
                        runs.append((t, t + 1, Affine("n", j, g, 1)))
        if not runs:
            continue
        kept, moved = [], []
        for a, b, tag in spans[j]:
            for lo, hi, x in runs:
                if lo < b and hi > a:
                    m = _Marks("n")
                    m.walk(x.slice(max(a, lo) - lo, min(b, hi) - lo), m.direct)
                    moved.extend((p, u, v, tag) for p, ivs in m.direct.items() for u, v in norm(ivs))
            kept.extend((u + a, v + a, tag) for u, v in _minus(((0, b - a),), [(lo - a, hi - a) for lo, hi, _ in runs]))
        spans[j] = kept + [(lo, hi, (3, 0)) for lo, hi, _ in runs]
        for p, u, v, tag in moved:  # an earlier node's, or this one's own first naming of a gate
            spans[p].append((u, v, tag))

    fx = ports.facts(root)
    facts: dict[ports.Fact, int] = {}
    values: dict[str, list] = {"producer": [], "lo": [], "hi": [], "bits": [], "reads": [], "output": [], "fact": []}
    pools: dict[int, list[int]] = {k: [] for k in sel}
    for j, nd in enumerate(nodes):
        for lo, hi, tags in _segments(nd.out.leaves, spans[j] + [(a, b, (4, f)) for a, b, f in fx.node(j).segments()]):
            if (3, 0) in tags:
                continue
            v = len(values["producer"])
            values["producer"].append(j)
            values["lo"].append(lo)
            values["hi"].append(hi)
            values["bits"].append(_bits(nd.out, lo, hi))
            values["reads"].append(sorted(k for t, k in tags if t == 0))
            values["output"].append((2, 0) in tags)
            values["fact"].append(facts.setdefault(next(f for t, f in tags if t == 4), len(facts)))
            for t, k in tags:
                if t == 1:
                    pools.setdefault(k, []).append(v)
    ks = sorted(pools)
    selects = {"call": ks, "gates": [sel.get(k, (0, 0))[0] for k in ks], "bits": [sel.get(k, (0, 0))[1] for k in ks],
               "values": [pools[k] for k in ks]}

    by = [classes[ids[id(nd.fn)]]["served_macs_by_precision"] for nd in nodes]

    def by_precision(ks: range) -> dict[str, int]:
        out: Counter = Counter()
        for k in ks:
            for q, n in by[k].items():
                out[q] += mult[k] * n
        return dict(sorted(out.items()))

    steps: dict[str, list] = {"ends": ends, "gates": [], **{c: [] for c in COUNTERS}, "served_macs_by_precision": [],
                              "select_gates": [], "select_bits": []}
    for a, b in zip([I] + ends, ends):
        steps["gates"].append(sum(nodes[k].gates for k in range(a, b)))
        for j, c in enumerate(COUNTERS):
            steps[c].append(sum(mult[k] * counts[id(nodes[k].fn)][j] for k in range(a, b)))
        steps["served_macs_by_precision"].append(by_precision(range(a, b)))
        steps["select_gates"].append(sum(sel[k][0] for k in range(a, b) if k in sel))
        steps["select_bits"].append(sum(sel[k][1] for k in range(a, b) if k in sel))

    inputs = {when: {"gates": 0, "bits": 0} for when in ("run", "registration")}
    for nd in nodes[:I]:
        inputs[input_id(nd.fn.id)[0]]["gates"] += nd.gates
        inputs[input_id(nd.fn.id)[0]]["bits"] += _all_bits(nd.out)
    top = [sum(mult[k] * counts[id(nodes[k].fn)][j] for k in range(I, len(nodes))) for j in range(3)]
    totals = {"gates": root.gates, "computed_gates": root.gates - sum(x["gates"] for x in inputs.values()), "inputs": inputs,
              "value_bits": sum(b for p, b in zip(values["producer"], values["bits"]) if p >= I),
              "values": len(values["producer"]), "classes": len(classes), **dict(zip(COUNTERS, top)),
              "served_macs_by_precision": by_precision(range(I, len(nodes))),
              "select_gates": sum(g for g, _ in sel.values()), "select_bits": sum(b for _, b in sel.values())}
    return seal({"schema": SCHEMA, "program": digest, "classes": classes, "callees": callees, "inputs": I,
                 "nodes": {"callee": [at[ids[id(nd.fn)]] for nd in nodes], "n": mult}, "steps": steps, "values": values,
                 "facts": [_fact_json(f) for f in facts], "selects": selects, "totals": totals, "approximate": approximate})


def step_of(doc: dict) -> Callable[[int], int]:
    """Root node -> its step in a census (-1 for an input node)."""
    I, ends = doc["inputs"], doc["steps"]["ends"]
    return lambda k: -1 if k < I else bisect.bisect_right(ends, k)


__all__ = ["SCHEMA", "PRODUCTS", "PRECISION", "COUNTERS", "FAMILY_COUNTS", "RULED", "census", "family", "seal", "step_ends", "step_of"]
