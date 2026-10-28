"""The logical layer's objects: a Program, its values, its inputs, and the commitments the values sit under.

The circuit is a `verity.ml.circuits` `Program`, the format the rest of Verity reads: the sampled proofs partition it into
proof blocks, PoUW's compiler rewrites its matmuls into tile units, the census counts it and memory accounting derives its
requirement from it. It is Boolean, its semantics And, Xor and Not on bits, built constructively from registered
Definitions (`verity/ml/circuits/PROTOCOL.md`):

    linear = bind(Gemm, K=32, N=32, **GEMM_AMPERE)     # ops.boolean.gemm: x[32] @ W[32, 32]^T on the BF16 step, on bits
    def body(B, S, x, w1, w2):
        h = B.call(linear, x, w1, name="h")
        return B.call(linear, h, w2, name="y")
    fn = CompositeDefinition("TwoStage", 1, (), lambda S: ((("x", VEC), ("w1", MAT), ("w2", MAT)), VEC), body)
    program = Program(fn, registered=("w1", "w2"))

Its inputs, commitments and reads are in the word view (`boolean.word_type`): an *element* is a Boolean element's bits
(a BF16 word's 16) or a word leaf, and each is one word, so a Program and its word twin bind the same inputs and commit
the same rows.

A Program holds only gates. Its root body is its step structure: an input node for each parameter, bound per run or at
registration (`program.bindings`), then its Calls. The accounting layer reads it at that grain, the census's
(`verity.ml.circuits.census`):

- a *node* is a root node, named by its parameter or by the `name` its Call was given (`names`);
- a *value* (`Value`) is a census value: a run of one node's output leaves that the same Calls read, which is what moves
  between blocks when they sit apart;
- a node's *work* (`work`) is its gates by kind: a tensor-core step's products (`mac:<precision>`, the census's `PRODUCTS`),
  a Boolean gate (`cuda:gate`) or any other word operation (`cuda:op`). Inputs, constants and wiring (`cut.WIRING`) are no
  work. It times the work and is never credited;
- a node's *ANDs* (`ands`), what is credited: the census's Boolean work, Lean's `Circuit.andSize`.

`evaluate` computes the census values, each Call through `verity.ml.circuits.evaluation.evaluate_batch` (a kernel where
one is registered, a Boolean Definition's word Definition's among them; bit-sliced otherwise); `register` commits the
registered inputs before any run, bound to the occupancy profiles the developer may choose from (`profile_digest`), and
`commit` gives the hiding commitment a run's values sit under: the census values,
which are what can cross between devices (a Call's own wires are witness of its proof blocks). Both are hm96-sha512 rows,
one per element, in one frame-v3-sha512 tree (`verity.ml.commitments.advice`), bound to the Program's digest.
"""
from __future__ import annotations

import hashlib
import weakref
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import types as T
from verity.ml.circuits.census import PRECISION, PRODUCTS, census, family
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import WIRING
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.evaluation import evaluate as reference
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.batch import type_dtype
from verity.ml.circuits.layout import FunctionCall, resolve
from verity.ml.circuits.program import REGISTRATION, RUN, Program, input_id, is_input
from verity.ml.commitments import advice


@dataclass(frozen=True)
class Value:
    """A census value: output leaves [lo, hi) of root node `node`, `bits` wide, read by the Calls `reads` (a gather's
    candidates among them) and returned by the Program when `output`."""

    id: int
    node: int
    lo: int
    hi: int
    bits: int
    reads: tuple[int, ...] = ()
    output: bool = False
    name: str = ""

    def __repr__(self) -> str:
        return f"%{self.id}:{self.name}<{self.bits}>"


def names(program: Program) -> list[str]:
    """Each root node's name: its parameter's (an input node), the `name` its Call was given, else its family and index."""
    return [nd.name or f"{family(nd.fn)}#{k}" for k, nd in enumerate(program.root.body.nodes)]


def nodes(program: Program, name: str) -> list[int]:
    """The root nodes `name` names: the one of that name, else a parameter's input nodes (`x.a`, `x[0]`, ...)."""
    ns = names(program)
    exact = [k for k, n in enumerate(ns) if n == name]
    return exact or [k for k, n in enumerate(ns) if n.startswith((name + ".", name + "["))]


def binding(program: Program, k: int) -> str | None:
    """`run` or `registration` for an input node, None for a Call."""
    when = input_id(program.root.body.nodes[k].fn.id)
    return when[0] if when else None


def calls(program: Program) -> list[int]:
    """The Program's Calls: its root nodes that are not input nodes."""
    return [k for k, nd in enumerate(program.root.body.nodes) if not is_input(nd.fn)]


def parameters(program: Program) -> list[tuple[str, str]]:
    """Each parameter's name and binding time, in order."""
    return [(n, b) for (n, _), b in zip(program.fn.signature()[0], program.bindings)]


_census: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def census_of(program: Program, boolean: bool = False) -> dict:
    """The Program's census/v1 (one step), computed once; with `boolean` its Boolean work too, or `census.NoBooleanWork`."""
    memo = _census.setdefault(program, {})
    if boolean not in memo:
        from verity.ml.definitions.boolean import boolean_version
        memo[boolean] = census(program, digest=program_digest(program), boolean=boolean and boolean_version)
    return memo[boolean]


def values(program: Program) -> list[Value]:
    """The Program's census values, each named after its node (with its leaves when it is part of the node's output)."""
    doc, ns = census_of(program), names(program)
    v, sel = doc["values"], doc["selects"]
    candidates: dict[int, set[int]] = {}
    for k, vs in zip(sel["call"], sel["values"]):
        for i in vs:
            candidates.setdefault(i, set()).add(k)
    out = []
    for i, (k, lo, hi, bits) in enumerate(zip(v["producer"], v["lo"], v["hi"], v["bits"])):
        whole = lo == 0 and hi == program.root.body.nodes[k].out.leaves
        out.append(Value(i, k, lo, hi, bits, tuple(sorted(set(v["reads"][i]) | candidates.get(i, set()))), v["output"][i],
                         ns[k] if whole else f"{ns[k]}[{lo}:{hi}]"))
    return out


def gates(program: Program, v: Value) -> list[int]:
    """The gates value `v` is: its leaves resolved to the gates that produce them."""
    scope = program.circuit.root_scope
    return [resolve(scope, ("n", v.node, j)) for j in range(v.lo, v.hi)]


def layout(t: T.Type) -> list[tuple[int, ...]]:
    """Each element of type `t`, in leaf order, as its leaves' widths: a Boolean element's bits (`boolean.elements`), or
    one word leaf."""
    if isinstance(t, T.Value):
        return [(t.w,)]
    if isinstance(t, T.Array):
        return ([(1,) * t.n] if t.n else []) if t.elem == BL.BIT else layout(t.elem) * t.n
    if isinstance(t, T.Tuple):
        return [e for f in t.fields for e in layout(f)]
    if isinstance(t, T.Record):
        return [e for _, f in t.fields for e in layout(f)]
    raise TypeError(t)


def _node_type(program: Program, k: int) -> T.Type:
    """Root node `k`'s output type: its parameter's for an input node (whose batch of one-bit inputs is flat), else its own."""
    nd = program.root.body.nodes[k]
    if binding(program, k) is not None:
        for pi, (_, t) in enumerate(program.fn.signature()[0]):
            refs = program.input_coll(pi).refs
            if refs.count == nd.out.leaves and program.root.body.node_at(next(iter(refs))[2]) == k:
                return t
    return nd.out


def rows(program: Program, v: Value) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Value `v`'s elements, as the gates of each and their widths (the part of an element `v` covers, where it covers
    part)."""
    gs, out, at = gates(program, v), [], 0
    for widths in layout(_node_type(program, v.node)):
        lo, hi = max(at, v.lo), min(at + len(widths), v.hi)
        if lo < hi:
            out.append((tuple(gs[lo - v.lo:hi - v.lo]), widths[lo - at:hi - at]))
        at += len(widths)
    return out


def _word(transcript: dict[int, int], gs: Sequence[int], widths: Sequence[int]) -> int:
    word, at = 0, 0
    for g, w in zip(gs, widths):
        word |= transcript[g] << at
        at += w
    return word


def read(program: Program, transcript: dict[int, int], v: Value) -> list[int]:
    """Value `v`'s words in `transcript`, one per element."""
    return [_word(transcript, gs, ws) for gs, ws in rows(program, v)]


def _own(fn) -> dict[str, int] | None:
    """The work one activation of `fn` is when it is one thing (a tensor-core step, a primitive); None for any other
    composite, whose work is its body's."""
    fam = family(fn)
    if fam in PRODUCTS:
        return {f"mac:{PRECISION[fam]}": PRODUCTS[fam]}
    if not isinstance(fn, PrimitiveDefinition):
        return None
    if is_input(fn) or not fn.params or fn.id in WIRING:
        return {}
    return {"cuda:gate" if BL.is_gate(fn) else "cuda:op": 1}


def work(program: Program) -> dict[int, dict[str, int]]:
    """Each Call's work by kind, from its Definition's body (never expanded: a batch or scan multiplies its callee's)."""
    memo: dict[int, dict[str, int]] = {}

    def of(fn) -> dict[str, int]:
        if id(fn) not in memo:
            own = _own(fn)
            if own is None:
                total: Counter = Counter()
                for nd in fn.body.nodes:
                    for kind, n in of(nd.fn).items():
                        total[kind] += n * (nd.n if nd.form in ("batch", "scan") else 1)
                own = dict(total)
            memo[id(fn)] = own
        return memo[id(fn)]

    nodes_ = program.root.body.nodes
    return {k: dict(of(nodes_[k].fn)) for k in calls(program)}


def ands(program: Program) -> dict[int, int]:
    """Each Call's ANDs: its class's Boolean work in the census, every facet's, times its batch's n."""
    doc = census_of(program, boolean=True)
    classes, callees, nodes = doc["classes"], doc["callees"], doc["nodes"]
    return {k: nodes["n"][k] * sum(f["and"] for f in classes[callees[nodes["callee"][k]]]["work"].values())
            for k in calls(program)}


_reach: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def reach(program: Program, roots: frozenset[int]) -> dict[int, int]:
    """Each root Call's ANDs among the gates the root nodes `roots` reach, gate by gate (a word gate has none). Only the
    Calls their census values reach are expanded, and none past the explicit limit (then nothing is reached)."""
    memo = _reach.setdefault(program, {})
    if roots in memo:
        return memo[roots]
    readers: dict[int, set[int]] = {}
    for v in values(program):
        readers.setdefault(v.node, set()).update(v.reads)
    calls, todo = set(), list(roots)
    while todo:
        for r in readers.get(todo.pop(), ()):
            if r not in calls:
                calls.add(r)
                todo.append(r)
    c, body = program.circuit, program.root.body
    if sum(body.offsets[k + 1] - body.offsets[k] for k in calls) > c.explicit_limit:
        memo[roots] = {}
        return {}
    gate_readers: dict[int, list[int]] = {}
    shares: dict[int, int] = {}
    for k in calls:
        for gate in c.region(body.offsets[k], body.offsets[k + 1]):
            for o in set(gate.operands):
                gate_readers.setdefault(o, []).append(gate.index)
            if (kind := BL.gate_kind(gate.prim)) and kind[0] == "and":
                shares[gate.index] = kind[1] - 1
    seen = {g for k in roots for g in range(body.offsets[k], body.offsets[k + 1])}
    todo = list(seen)
    while todo:
        for r in gate_readers.get(todo.pop(), ()):
            if r not in seen:
                seen.add(r)
                todo.append(r)
    out: Counter = Counter()
    for g in seen & shares.keys():
        out[body.node_at(g)] += shares[g]
    memo[roots] = dict(out)
    return memo[roots]


def bind_inputs(program: Program, inputs: dict[str, int | Sequence[int]]) -> dict[int, int]:
    """The input gates' values: each parameter's words, one per element (an int for a parameter of one element)."""
    out: dict[int, int] = {}
    for pi, ((name, _), (_, t)) in enumerate(zip(parameters(program), program.fn.signature()[0])):
        if name not in inputs:
            raise ValueError(f"no value for parameter {name!r}")
        words = inputs[name]
        words = [words] if isinstance(words, int) else list(words)
        elements = layout(t)
        if len(words) != len(elements):
            raise ValueError(f"parameter {name!r} has {len(elements)} elements, given {len(words)} words")
        refs = iter(program.input_coll(pi).refs)
        for word, widths in zip(words, elements):
            if word >> sum(widths):
                raise ValueError(f"parameter {name!r}: the word {word:#x} is wider than its {sum(widths)} bits")
            for w in widths:
                out[next(refs)[2]] = word & ((1 << w) - 1)
                word >>= w
    return out


def evaluate(program: Program, inputs: dict[str, int | Sequence[int]]) -> dict[int, int]:
    """The values on `inputs` (by parameter) of the input gates and of what each root Call returns, by gate: every census
    value. Each root node is evaluated by `evaluate_batch` (a batch's members as one batch), never gate by gate."""
    c, t = program.circuit, bind_inputs(program, inputs)
    for k, nd in enumerate(program.root.body.nodes):
        if is_input(nd.fn):
            continue
        params = [pt for _, pt in nd.fn.signature()[0]]
        groups = [range(nd.n)] if nd.form != "scan" else [[m] for m in range(nd.n)]
        for members in groups:
            fcs = [FunctionCall(c, c.root_scope, k, m) for m in members]
            args = [[t[g] for g in fc.args] for fc in fcs]
            arrays, at = [], 0
            for pt in params:
                arrays.append(np.array([a[at:at + pt.leaves] for a in args], dtype=type_dtype(pt)))
                at += pt.leaves
            out = (np.asarray(evaluate_batch(nd.fn, *arrays)).reshape(len(fcs), -1) if params
                   else [reference(nd.fn)] * len(fcs))
            for fc, row in zip(fcs, out):
                t.update(zip(fc.returns, (int(x) for x in row)))
    return t


def _binding(label: bytes, program: Program, profiles: Sequence[bytes] = ()) -> bytes:
    return hashlib.sha512(label + b"\0" + bytes.fromhex(program_digest(program)) + b"".join(profiles)).digest()


def profile_digest(occupancy: Mapping[Any, Mapping[int, int]], origin: int = 0) -> bytes:
    """The digest of an occupancy profile (`binding.occupancy`: frames per certifier port and bucket), its buckets
    counted from `origin`, the first bucket of the epoch it is chosen for, so one registered profile fits every epoch."""
    h = hashlib.sha512(b"datacenter/occupancy/v1\0")
    for port in sorted(occupancy, key=repr):
        for b, n in sorted(occupancy[port].items()):
            if n:
                h.update(f"{port!r}\0{b - origin}\0{n}\0".encode())
    return h.digest()


def _rows(words: list[int], bits: int) -> list[bytes]:
    return [w.to_bytes((bits + 7) // 8, "little") for w in words]


@dataclass(frozen=True)
class Registration:
    """A Program's registered inputs and the occupancy profiles its developer may choose from, committed before any run:
    `root` is public, and so is the number of profiles; the values, salts and profiles are the developer's."""

    values: dict[str, int | tuple[int, ...]]
    rows: advice.Registration
    #: the digests of the occupancy profiles the developer may choose from (`profile_digest`), which the root binds;
    #: none: one profile, the placement's own
    profiles: tuple[bytes, ...] = ()

    @property
    def root(self) -> bytes:
        return self.rows.root

    @property
    def salts(self) -> tuple[bytes, ...]:
        return self.rows.salts


def register(program: Program, inputs: dict[str, int | Sequence[int]], salts: Sequence[bytes] | None = None,
             profiles: Sequence[Mapping[Any, Mapping[int, int]] | bytes] = ()) -> Registration:
    """Commit the values of `program`'s registered parameters, a row per element, under a binding to the Program's
    digest and to the occupancy `profiles` the developer may choose from (each with its buckets counted from its
    epoch's first, or its `profile_digest`)."""
    params = [(n, t) for (n, b), (_, t) in zip(parameters(program), program.fn.signature()[0]) if b == REGISTRATION]
    if not params:
        raise ValueError("the Program registers no parameter")
    regd = {n: inputs[n] if isinstance(inputs[n], int) else tuple(inputs[n]) for n, _ in params}
    words = [w for n in regd for w in ([regd[n]] if isinstance(regd[n], int) else regd[n])]
    width = max(sum(e) for _, t in params for e in layout(t))
    digests = tuple(o if isinstance(o, bytes) else profile_digest(o) for o in profiles)
    return Registration(regd, advice.register(_rows(words, width), width,
                                              _binding(b"datacenter/registration/v1", program, digests), salts), digests)


_committed: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def committed(program: Program) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The rows a run commits: the census values' elements (`rows`), what the Program's nodes pass each other, each once
    in value order. A Call's own wires are witness of its proof blocks, whose commitments are the proof system's."""
    if program not in _committed:
        seen: dict[tuple[int, ...], tuple[int, ...]] = {}
        for v in values(program):
            for gs, ws in rows(program, v):
                seen.setdefault(gs, ws)
        _committed[program] = list(seen.items())
    return _committed[program]


def commit(program: Program, transcript: dict[int, int], salts: Sequence[bytes] | None = None) -> advice.Registration:
    """The hiding commitment to a run's values (`committed`), a row per element: its `root` is public, its rows and salts
    witness."""
    rs = committed(program)
    width = max(sum(ws) for _, ws in rs)
    return advice.register(_rows([_word(transcript, gs, ws) for gs, ws in rs], width),
                           width, _binding(b"datacenter/values/v1", program), salts)


__all__ = ["REGISTRATION", "RUN", "Registration", "Value", "ands", "bind_inputs", "binding", "calls", "census_of",
           "commit", "committed", "evaluate", "gates", "layout", "names", "nodes", "parameters", "reach",
           "read", "register", "rows", "values", "work"]
