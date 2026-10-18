"""The logical layer's objects: a circuit, its values, its inputs, and commitments to values.

A circuit holds only gates. Everything a run brings or produces (each input, each intermediate, each output) is a
value: one word of a fixed number of bits, produced by exactly one gate. A circuit is written as an ordinary Python
function over words and traced, the way `jax.make_jaxpr` traces a function into a jaxpr:

    def mlp(x, w):
        return (x * w) ^ 0x5A

    c = trace(mlp, x=Input(8), w=Input(8, source="inside"))

Word gates stand for Boolean circuits of known size (`Gate.boolean_size`), so a circuit is still a Boolean circuit for
everything that counts gates; the word view keeps the walkthrough readable. `Circuit.evaluate` is the reference
semantics, and `commit` gives the hiding commitment every value sits under.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Callable, Literal

#: where an input's bits come from: the outside world (requests), the datacenter itself (telemetry, clocks, sensors),
#: or a registered artifact whose digest is declared before the run (weights, an image)
Source = Literal["outside", "inside", "registered"]

#: word operations and the Boolean gates each costs at width w (ripple-carry add, schoolbook multiply)
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
    """An input: `bits` bits the circuit doesn't determine. Every input is charged: the accounting layer counts each
    input's bits as information entering the computation, unless its `source` is `registered` (its bytes are fixed by a
    digest declared before the run)."""

    bits: int
    source: Source = "outside"
    name: str | None = None


@dataclass(frozen=True)
class Value:
    """A word of `bits` bits: the output of gate `id` of its circuit."""

    id: int
    bits: int
    name: str | None = None

    def __repr__(self) -> str:
        return f"%{self.id}" + (f":{self.name}" if self.name else "") + f"<{self.bits}>"


@dataclass(frozen=True)
class Gate:
    op: str
    operands: tuple[int, ...]
    bits: int
    const: int | None = None
    input: Input | None = None

    @property
    def boolean_size(self) -> int:
        """The number of Boolean gates this word gate stands for."""
        return BASIS[self.op](self.bits)


@dataclass
class Circuit:
    """Gates in topological order; gate i produces value i. `outputs` are the values the circuit delivers."""

    gates: list[Gate] = field(default_factory=list)
    outputs: tuple[int, ...] = ()
    names: dict[int, str] = field(default_factory=dict)

    def value(self, i: int) -> Value:
        return Value(i, self.gates[i].bits, self.names.get(i))

    @property
    def values(self) -> list[Value]:
        return [self.value(i) for i in range(len(self.gates))]

    @property
    def inputs(self) -> list[Value]:
        return [self.value(i) for i, g in enumerate(self.gates) if g.op == "input"]

    @property
    def boolean_size(self) -> int:
        return sum(g.boolean_size for g in self.gates)

    def operands(self, v: Value | int) -> list[Value]:
        i = v.id if isinstance(v, Value) else v
        return [self.value(j) for j in self.gates[i].operands]

    def evaluate(self, inputs: dict[str, int]) -> dict[int, int]:
        """Every value of the circuit on `inputs` (by input name): the reference semantics."""
        out: dict[int, int] = {}
        for i, g in enumerate(self.gates):
            mask = (1 << g.bits) - 1
            a = [out[j] for j in g.operands]
            if g.op == "input":
                out[i] = inputs[self.names[i]] & mask
            elif g.op == "const":
                out[i] = g.const & mask
            elif g.op == "add":
                out[i] = (a[0] + a[1]) & mask
            elif g.op == "mul":
                out[i] = (a[0] * a[1]) & mask
            elif g.op == "xor":
                out[i] = a[0] ^ a[1]
            elif g.op == "and":
                out[i] = a[0] & a[1]
            elif g.op == "or":
                out[i] = a[0] | a[1]
            elif g.op == "not":
                out[i] = ~a[0] & mask
        return out

    def __repr__(self) -> str:
        lines = [f"circuit ({len(self.gates)} word gates, {self.boolean_size} Boolean gates)"]
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

    @property
    def bits(self) -> int:
        return self.circuit.gates[self.i].bits

    def _lift(self, other: _Tracer | int) -> int:
        if isinstance(other, _Tracer):
            return other.i
        self.circuit.gates.append(Gate("const", (), self.bits, const=int(other)))
        return len(self.circuit.gates) - 1

    def _op(self, op: str, *others: _Tracer | int) -> _Tracer:
        operands = (self.i, *(self._lift(o) for o in others))
        self.circuit.gates.append(Gate(op, operands, self.bits))
        return _Tracer(self.circuit, len(self.circuit.gates) - 1)

    __add__ = __radd__ = lambda s, o: s._op("add", o)
    __mul__ = __rmul__ = lambda s, o: s._op("mul", o)
    __xor__ = __rxor__ = lambda s, o: s._op("xor", o)
    __and__ = __rand__ = lambda s, o: s._op("and", o)
    __or__ = __ror__ = lambda s, o: s._op("or", o)
    __invert__ = lambda s: s._op("not")


def trace(fn: Callable[..., object], **inputs: Input) -> Circuit:
    """The circuit `fn` computes, with one input gate per keyword argument (named after it). `fn` returns a word or a
    tuple of words, the circuit's outputs."""
    c = Circuit()
    args = {}
    for name in inspect.signature(fn).parameters:
        spec = inputs[name]
        c.gates.append(Gate("input", (), spec.bits, input=Input(spec.bits, spec.source, name)))
        c.names[len(c.gates) - 1] = name
        args[name] = _Tracer(c, len(c.gates) - 1)
    result = fn(**args)
    result = result if isinstance(result, tuple) else (result,)
    c.outputs = tuple(r.i for r in result)
    return c


def name(circuit: Circuit, v: Value | int, label: str) -> Value:
    """Give value `v` a name (for printing and placement)."""
    i = v.id if isinstance(v, Value) else v
    circuit.names[i] = label
    return circuit.value(i)


@dataclass(frozen=True)
class Commitment:
    """A hiding commitment to a list of values: SHA-512 over a fresh salt and the values' encodings. The digest is
    public; the salt and the values are witness."""

    digest: bytes
    count: int


def commit(values: list[int], bits: list[int], salt: bytes | None = None) -> tuple[Commitment, bytes]:
    """A commitment to `values` (each of the given width) and the salt that opens it."""
    salt = os.urandom(64) if salt is None else salt
    h = hashlib.sha512(b"datacenter/values/v1\0" + salt)
    for v, w in zip(values, bits, strict=True):
        h.update(w.to_bytes(2, "big") + v.to_bytes((w + 7) // 8 or 1, "big"))
    return Commitment(h.digest(), len(values)), salt
