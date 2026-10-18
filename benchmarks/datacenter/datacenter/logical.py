"""The logical layer's objects: a circuit, its values, its inputs, and the commitment every value sits under.

A circuit holds only gates. Everything a run brings or produces (each input, each intermediate, each output) is a
value: one word of a fixed number of bits, produced by exactly one gate. A circuit is written as an ordinary Python
function over words and traced, the way `jax.make_jaxpr` traces a function into a jaxpr:

    def layer(x, w):
        return (x * w) ^ 0x5A

    c = trace(layer, x=Input(8), w=Input(8, source="registered", digest=d))

Word gates stand for Boolean circuits of known size (`Gate.work`), so for everything that counts gates a circuit is a
Boolean circuit; the word view keeps the walkthrough readable. A circuit adapted from another format (verity's word
Definitions) brings each primitive's semantics and counts its work in that format's unit (`Circuit.unit`).
`Circuit.evaluate` is the reference semantics, and `commit` gives the hiding commitment the values sit under.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Callable, Literal

#: where an input's bits come from:
#: - "outside": the world (a request);
#: - "inside": the datacenter itself (telemetry, a clock reading, a sensor);
#: - "registered": bytes whose digest was declared before the run (weights, a container image);
#: - "trusted": a trusted device's fresh data (the challenger's noise), part of the verifier's coins.
Source = Literal["outside", "inside", "registered", "trusted"]

#: word operations, and the Boolean gates each stands for at width w (ripple-carry add, schoolbook multiply)
BASIS: dict[str, Callable[[int], int]] = {
    "input": lambda w: 0,
    "const": lambda w: 0,
    "add": lambda w: 5 * w,
    "mul": lambda w: 6 * w * w,
    "xor": lambda w: w,
    "and": lambda w: w,
    "or": lambda w: w,
    "not": lambda w: w,
}


@dataclass(frozen=True)
class Input:
    """An input: `bits` bits the circuit doesn't determine. A registered input carries the digest of its content,
    declared before the run."""

    bits: int
    source: Source = "outside"
    digest: bytes | None = None


@dataclass(frozen=True)
class Value:
    """A word of `bits` bits: the output of gate `id`. An input's value also says where its bits come from."""

    id: int
    bits: int
    name: str | None = None
    source: Source | None = None

    def __repr__(self) -> str:
        return f"%{self.id}" + (f":{self.name}" if self.name else "") + f"<{self.bits}>"


@dataclass(frozen=True)
class Gate:
    """One gate, producing one value of `bits` bits. A word operation of `BASIS`, or a primitive of a circuit format
    outside it (verity's word Definitions, say): then `fn` is its semantics and `size` its work. The work counts in
    `unit` (a kind of work: Boolean gates, multiply-adds at a precision), the circuit's unit when None."""

    op: str
    operands: tuple[int, ...]
    bits: int
    const: int | None = None
    input: Input | None = None
    fn: Callable[..., int] | None = field(default=None, compare=False)
    size: int | None = None
    unit: str | None = None

    @property
    def work(self) -> int:
        """The work this gate stands for: its `size`, or the Boolean gates of its word operation."""
        return self.size if self.size is not None else BASIS[self.op](self.bits)


def digest(value: int, bits: int) -> bytes:
    """The digest a registered input's content is declared by."""
    return hashlib.sha512(b"datacenter/registered/v1\0" + bits.to_bytes(2, "big") + value.to_bytes((bits + 7) // 8, "big")).digest()


@dataclass
class Circuit:
    """Gates in topological order: gate i produces value i. `outputs` are the values the circuit delivers. `unit` is
    what its gates' work counts unless a gate says otherwise: Boolean gates (`"gate"`) for a traced circuit, a format's
    own (a word operation, a multiply-add at a precision) for one adapted from it. `weights` is what an audit credits
    for one unit of each kind (PoUW's W_ref, a public constant of its scheme), 1 for a kind it doesn't list."""

    gates: list[Gate] = field(default_factory=list)
    outputs: tuple[int, ...] = ()
    names: dict[int, str] = field(default_factory=dict)
    unit: str = "gate"
    weights: dict[str, float] = field(default_factory=dict)

    def unit_of(self, g: Gate) -> str:
        return g.unit or self.unit

    def value(self, i: int) -> Value:
        g = self.gates[i]
        return Value(i, g.bits, self.names.get(i), g.input.source if g.input else None)

    def __getitem__(self, name: str) -> Value:
        return next(self.value(i) for i, n in self.names.items() if n == name)

    @property
    def values(self) -> list[Value]:
        return [self.value(i) for i in range(len(self.gates))]

    @property
    def inputs(self) -> list[Value]:
        return [self.value(i) for i, g in enumerate(self.gates) if g.op == "input"]

    @property
    def work(self) -> float:
        """What an audit credits for the whole circuit: each gate's work at its kind's weight. For a traced circuit, its
        size as a Boolean circuit."""
        return sum(self.weights.get(self.unit_of(g), 1.0) * g.work for g in self.gates)

    def evaluate(self, inputs: dict[str, int]) -> dict[int, int]:
        """Every value of the circuit on `inputs` (by input name): the reference semantics."""
        out: dict[int, int] = {}
        for i, g in enumerate(self.gates):
            mask = (1 << g.bits) - 1
            a = [out[j] for j in g.operands]
            if g.fn is not None:
                out[i] = g.fn(*a) & mask
                continue
            out[i] = {
                "input": lambda: inputs[self.names[i]] & mask,
                "const": lambda: g.const & mask,
                "add": lambda: (a[0] + a[1]) & mask,
                "mul": lambda: (a[0] * a[1]) & mask,
                "xor": lambda: a[0] ^ a[1],
                "and": lambda: a[0] & a[1],
                "or": lambda: a[0] | a[1],
                "not": lambda: ~a[0] & mask,
            }[g.op]()
        return out

    def __repr__(self) -> str:
        lines = [f"circuit ({len(self.gates)} word gates, "
                 + (f"{self.work:g} Boolean gates)" if self.unit == "gate" else f"{self.work:g} {self.unit})")]
        for i, g in enumerate(self.gates):
            rhs = (f"input {g.input.source}" if g.op == "input" else f"const {g.const:#x}" if g.op == "const"
                   else f"{g.op} " + ", ".join(repr(self.value(j)) for j in g.operands))
            lines.append(f"  {self.value(i)!r} = {rhs}")
        lines.append("  return " + ", ".join(repr(self.value(j)) for j in self.outputs))
        return "\n".join(lines)


class _Tracer:
    """A traced word: arithmetic on tracers appends gates to the circuit being traced."""

    def __init__(self, circuit: Circuit, i: int):
        self.circuit, self.i = circuit, i

    def _lift(self, other: _Tracer | int) -> int:
        if isinstance(other, _Tracer):
            return other.i
        self.circuit.gates.append(Gate("const", (), self.circuit.gates[self.i].bits, const=int(other)))
        return len(self.circuit.gates) - 1

    def _op(self, op: str, *others: _Tracer | int) -> _Tracer:
        operands = (self.i, *(self._lift(o) for o in others))
        self.circuit.gates.append(Gate(op, operands, self.circuit.gates[self.i].bits))
        return _Tracer(self.circuit, len(self.circuit.gates) - 1)

    def named(self, label: str) -> _Tracer:
        """Name this value, so a placement can refer to it. A name never changes what the circuit computes."""
        self.circuit.names[self.i] = label
        return self

    __add__ = __radd__ = lambda s, o: s._op("add", o)
    __mul__ = __rmul__ = lambda s, o: s._op("mul", o)
    __xor__ = __rxor__ = lambda s, o: s._op("xor", o)
    __and__ = __rand__ = lambda s, o: s._op("and", o)
    __or__ = __ror__ = lambda s, o: s._op("or", o)
    __invert__ = lambda s: s._op("not")


def trace(fn: Callable[..., object], **inputs: Input) -> Circuit:
    """The circuit `fn` computes, with one input gate per keyword argument, named after it. `fn` returns a word or a
    tuple of words: the circuit's outputs."""
    c = Circuit()
    args = {}
    for name in inspect.signature(fn).parameters:
        c.gates.append(Gate("input", (), inputs[name].bits, input=inputs[name]))
        c.names[len(c.gates) - 1] = name
        args[name] = _Tracer(c, len(c.gates) - 1)
    result = fn(**args)
    c.outputs = tuple(r.i for r in (result if isinstance(result, tuple) else (result,)))
    return c


@dataclass(frozen=True)
class Commitment:
    """A hiding commitment to every value of a run: SHA-512 over a fresh salt and the values. The digest is public; the
    salt and the values are witness."""

    digest: bytes
    count: int


def commit(c: Circuit, values: dict[int, int], salt: bytes | None = None) -> tuple[Commitment, bytes]:
    """The commitment to `values` (every value of `c`) and the salt that opens it."""
    salt = os.urandom(64) if salt is None else salt
    h = hashlib.sha512(b"datacenter/values/v1\0" + salt)
    for i, g in enumerate(c.gates):
        h.update(g.bits.to_bytes(2, "big") + values[i].to_bytes((g.bits + 7) // 8 or 1, "big"))
    return Commitment(h.digest(), len(c.gates)), salt
