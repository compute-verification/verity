"""The logical layer's objects: a Program, its values, its inputs, and the commitments the values sit under.

The circuit is a `verity.ml.circuits` `Program`, the format the rest of Verity reads: the sampled proofs partition it into
proof units, PoUW's compiler rewrites its matmuls into tile units, the census counts it and memory accounting derives its
requirement from it. It is built constructively from registered Definitions (`verity/ml/circuits/PROTOCOL.md`):

    linear = bind(Gemm, K=32, N=32, **GEMM_AMPERE)     # x[32] @ W[32, 32]^T on the BF16 tensor-core step
    def body(B, S, x, w1, w2):
        h = B.call(linear, x, w1, name="h")
        return B.call(linear, h, w2, name="y")
    fn = CompositeDefinition("TwoStage", 1, (), lambda S: ((("x", VEC), ("w1", MAT), ("w2", MAT)), VEC), body)
    program = Program(fn, registered=("w1", "w2"))

A Program holds only gates. Its root body is its step structure: an input node for each parameter, bound per run or at
registration (`program.bindings`), then its Calls. The accounting layer reads it at that grain, the census's
(`verity.ml.circuits.census`):

- a *node* is a root node, named by its parameter or by the `name` its Call was given (`names`);
- a *value* (`Value`) is a census value: a run of one node's output leaves that the same Calls read, which is what moves
  between blocks when they sit apart;
- a node's *work* (`work`) is its gates by kind: a tensor-core step's products (`mac:<precision>`, the census's `PRODUCTS`),
  a Boolean gate (`cuda:gate`) or any other word operation (`cuda:op`). Inputs, constants and wiring (`cut.WIRING`) are no
  work.

`evaluate` is the reference semantics, gate by gate (`evaluate_region`, so a Program small enough to expand); `register`
commits the registered inputs before any run, and `commit` gives the hiding commitment every value of a run sits under.
Both are hm96-sha512 rows in one frame-v3-sha512 tree (`verity.ml.commitments.advice`), bound to the Program's digest.
"""
from __future__ import annotations

import hashlib
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.census import PRODUCTS, census, family
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import WIRING
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import Circuit, resolve
from verity.ml.circuits.program import REGISTRATION, RUN, Program, input_id, is_input
from verity.ml.commitments import advice

#: a tensor-core step family's operand format, by the first of these its name contains
PRECISION = (("BF16", "bf16"), ("Tf32", "tf32"), ("E4m3", "fp8"), ("E5m2", "fp8"), ("Nvf4", "fp4"), ("Mxf4", "fp4"),
             ("S8", "int8"), ("I7", "int8"))


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


def values(program: Program) -> list[Value]:
    """The Program's census values, each named after its node (with its leaves when it is part of the node's output)."""
    if program not in _census:
        _census[program] = census(program)
    doc, ns = _census[program], names(program)
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


def precision(fam: str) -> str:
    return next(p for key, p in PRECISION if key in fam)


def _own(fn) -> dict[str, int] | None:
    """The work one activation of `fn` is when it is one thing (a tensor-core step, a primitive); None for any other
    composite, whose work is its body's."""
    fam = family(fn)
    if fam in PRODUCTS:
        return {f"mac:{precision(fam)}": PRODUCTS[fam]}
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


def gate_work(circuit: Circuit, g: int) -> dict[str, int]:
    """Gate `g`'s share of `work`: a tensor-core step's products sit on its last gate, every other gate of it is none."""
    gate = circuit.gate(g)
    step, s = None, gate.scope
    while s.parent is not None:
        if family(s.spec) in PRODUCTS:
            step = s
        s = s.parent
    if step is None:
        return _own(gate.prim) or {}
    return _own(step.spec) if g == step.offset + step.spec.gates - 1 else {}


def bind_inputs(program: Program, inputs: dict[str, int | Sequence[int]]) -> dict[int, int]:
    """The input gates' values: each parameter's words (an int for a parameter of one leaf), in leaf order."""
    out: dict[int, int] = {}
    for pi, (name, _) in enumerate(parameters(program)):
        if name not in inputs:
            raise ValueError(f"no value for parameter {name!r}")
        words = inputs[name]
        words = [words] if isinstance(words, int) else list(words)
        refs = program.input_coll(pi).refs
        if len(words) != refs.count:
            raise ValueError(f"parameter {name!r} has {refs.count} leaves, given {len(words)} words")
        out.update((ref[2], w) for ref, w in zip(refs, words))
    return out


def evaluate(program: Program, inputs: dict[str, int | Sequence[int]]) -> dict[int, int]:
    """Every gate's value on `inputs` (by parameter): the reference semantics."""
    c = program.circuit
    return evaluate_region(c, c.region(0, c.size), bind_inputs(program, inputs))


def _binding(label: bytes, program: Program) -> bytes:
    return hashlib.sha512(label + b"\0" + bytes.fromhex(program_digest(program))).digest()


def _rows(words: list[int], bits: int) -> list[bytes]:
    return [w.to_bytes((bits + 7) // 8, "little") for w in words]


@dataclass(frozen=True)
class Registration:
    """A Program's registered inputs, committed before any run: `root` is public, the values and salts the developer's."""

    values: dict[str, int | tuple[int, ...]]
    rows: advice.Registration

    @property
    def root(self) -> bytes:
        return self.rows.root

    @property
    def salts(self) -> tuple[bytes, ...]:
        return self.rows.salts


def register(program: Program, inputs: dict[str, int | Sequence[int]], salts: Sequence[bytes] | None = None) -> Registration:
    """Commit the values of `program`'s registered parameters, a row per leaf, under a binding to the Program's digest."""
    regd = {n: inputs[n] if isinstance(inputs[n], int) else tuple(inputs[n])
            for n, b in parameters(program) if b == REGISTRATION}
    if not regd:
        raise ValueError("the Program registers no parameter")
    words = [w for n in regd for w in ([regd[n]] if isinstance(regd[n], int) else regd[n])]
    pis = [pi for pi, (_, b) in enumerate(parameters(program)) if b == REGISTRATION]
    width = max(program.circuit.gate(program.input_coll(pi).refs[0][2]).prim.ret.w for pi in pis)
    return Registration(regd, advice.register(_rows(words, width), width, _binding(b"datacenter/registration/v1", program),
                                              salts))


def commit(program: Program, transcript: dict[int, int], salts: Sequence[bytes] | None = None) -> advice.Registration:
    """The hiding commitment to every value of a run, a row per gate: its `root` is public, its rows and salts witness."""
    c = program.circuit
    width = max(g.prim.ret.w for g in c.region(0, c.size))
    return advice.register(_rows([transcript[g] for g in range(c.size)], width),
                           width, _binding(b"datacenter/values/v1", program), salts)


__all__ = ["PRECISION", "REGISTRATION", "RUN", "Registration", "Value", "bind_inputs", "binding", "calls", "commit",
           "evaluate", "gate_work", "gates", "names", "nodes", "parameters", "precision", "register", "values", "work"]
