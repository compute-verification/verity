"""The universal unit: one public Boolean Definition that runs any private circuit of a size class.

    UniversalUnit_v1{G, N_IN, N_OUT}
        (op: Array<G, Array<2, Value<1>>>, src: Array<G, Array<2, Array<K, Value<1>>>>, out: Array<N_OUT, Array<K, Value<1>>>,
         x: Array<N_IN, Value<1>>) -> Array<N_OUT, Value<1>>
        W = N_IN + G wires, K = bit_length(W - 1) bits per wire index (at least 1)

The private circuit's gates and wiring are a value, a `Netlist`, that this Definition takes as input beside the data, so the
unit's structure is a function of the size class (G, N_IN, N_OUT) alone. It is not how a private circuit is hidden: circuit
hiding is recursion (C-Flock's ``--hide``).

**The private circuit's description** is in SSA form: wires 0 .. N_IN - 1 are the data inputs x, and wire N_IN + g is gate g's output. Gate g is
``(op, a, b)``; it reads wires a and b and computes

    op 0  XOR    a ^ b
    op 1  AND    a & b
    op 2  XNOR   ~(a ^ b)
    op 3  NAND   ~(a & b)

op's bit 0 picks AND over XOR and its bit 1 negates. With a = b they give the constants 0 (XOR) and 1 (XNOR), NOT a (NAND) and a
copy of a (AND), so the set is complete. Output i is wire ``out[i]``. Every bit string describes a private circuit: a source at or past the
reading gate's own wire (a forward read, or one past W) reads 0, as does an output past W, so the Definition is total and no
description has a cycle. A no-op gate is ``(XOR, 0, 0)``: the constant 0, which nothing reads. `pad` fits a smaller private circuit to a class
with no-op gates and unread inputs, and leaves its outputs unchanged; `interpret` is the reference the Definition agrees with.

**Routing.** Each source and each output is a multiplexer tree over the wires it may read (`gather.gather`, filling with 0):
gate g's two sources over its N_IN + g earlier wires, each output over all W. The gate's function is 2 ANDs (a & b, then op's bit
0 selecting it over a ^ b), and the tree over m wires by a K-bit index is T(m) ANDs, the sum of ceil(c / 2) over K halvings of
c = m. In all (`and_count`):

    ANDs = sum over g < G of (2 T(N_IN + g) + 2)  +  N_OUT T(W)  ~  G (2 N_IN + G) + N_OUT W

which is O(G W) with W = N_IN + G: about 2 N_IN + G ANDs per gate of the private circuit.

**Why this design.** It is the simplest universal unit, built from the multiplexer tree core already has.
- SSA, not a register file (W registers, each gate writing a destination): a register file pays a destination decoder and a
  W-wide conditional write per gate besides the two reads, about 4W ANDs a gate. SSA pays only the reads, and since a gate reads
  only earlier wires, every bit string describes an acyclic circuit with no validity check. A register file's advantage, W below
  N_IN + G for a long, narrow private circuit, is what a permutation network gives better.
- Multiplexers, not a permutation network: one Beneš or Waksman network from every wire to every gate input would read each
  gate's own output and later ones, a cycle, so routing in O(k log k) needs Valiant's edge-universal graphs over the k =
  N_IN + G + N_OUT nodes, about 4.5 k log2 k ANDs in their best known form (one AND per switch) and far more structure. That
  estimate is about these multiplexers' count at G = 16 and a third of it at G = 64; past that it wins, O(log k) ANDs per gate
  against O(W).

It is traced from `build` (`trace`): its body is And/Xor/Not gates with exactly the builder's live ANDs. It names no word
Definition; `interpret` is its reference, and the tests check that the two agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.types import Array

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.gather import gather

__all__ = ["AND", "NAND", "NOOP", "Netlist", "UniversalUnit", "XNOR", "XOR", "and_count", "arguments", "build", "index_bits",
           "interpret", "pad", "traced"]

BIT = BL.BIT
XOR, AND, XNOR, NAND = 0, 1, 2, 3
#: a gate nothing reads: the constant 0
NOOP = (XOR, 0, 0)


def index_bits(G: int, N_IN: int) -> int:
    """K: the bits of a wire index, enough for the W = N_IN + G wires."""
    return max(1, (N_IN + G - 1).bit_length())


def _check_class(G: int, N_IN: int, N_OUT: int) -> None:
    for k, v in (("G", G), ("N_IN", N_IN), ("N_OUT", N_OUT)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"a universal unit's {k} is an int >= 1, not {v!r}")


@dataclass(frozen=True)
class Netlist:
    """``gates[g] = (op, a, b)`` makes wire ``n_in + g``; ``outputs`` are wire indices. Any index is allowed: one the reading
    gate can't see, or an output past the last wire, reads 0."""

    n_in: int
    gates: tuple[tuple[int, int, int], ...]
    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(tuple(int(v) for v in g) for g in self.gates))
        object.__setattr__(self, "outputs", tuple(int(o) for o in self.outputs))
        if self.n_in < 0 or any(len(g) != 3 or not 0 <= g[0] <= 3 or min(g[1:]) < 0 for g in self.gates) or \
                any(o < 0 for o in self.outputs):
            raise ValueError("a netlist's gates are (op in 0..3, a >= 0, b >= 0) and its outputs indices >= 0")

    @property
    def wires(self) -> int:
        return self.n_in + len(self.gates)


def interpret(net: Netlist, x: Sequence[int]) -> list[int]:
    """The private circuit run gate by gate on the bits ``x``: the reference `UniversalUnit` agrees with."""
    if len(x) != net.n_in:
        raise ValueError(f"the netlist has {net.n_in} inputs, not {len(x)}")
    w = [b & 1 for b in x]
    for op, a, b in net.gates:
        va = w[a] if a < len(w) else 0
        vb = w[b] if b < len(w) else 0
        w.append(((va & vb) if op & 1 else (va ^ vb)) ^ (op >> 1))
    return [w[o] if o < len(w) else 0 for o in net.outputs]


def pad(net: Netlist, G: int, N_IN: int) -> Netlist:
    """``net`` in the class's G gates over N_IN inputs: unread inputs after its own (its gate wires shifted past them), then
    no-op gates. Every index of ``net`` must be in range (a source below its gate's own wire, an output below its wires)."""
    g0, n0 = len(net.gates), net.n_in
    if g0 > G or n0 > N_IN:
        raise ValueError(f"a netlist of {g0} gates over {n0} inputs does not fit G = {G}, N_IN = {N_IN}")
    if any(max(a, b) >= n0 + g for g, (_, a, b) in enumerate(net.gates)) or any(o >= net.wires for o in net.outputs):
        raise ValueError("pad takes a netlist whose every index is in range")
    shift = N_IN - n0

    def at(i: int) -> int:
        return i if i < n0 else i + shift

    gates = [(op, at(a), at(b)) for op, a, b in net.gates] + [NOOP] * (G - g0)
    return Netlist(N_IN, tuple(gates), tuple(at(o) for o in net.outputs))


def _bits(v: int, k: int) -> list[int]:
    return [(v >> j) & 1 for j in range(k)]


def arguments(net: Netlist, x: Sequence[int]) -> tuple[list[int], list[int], list[int], list[int]]:
    """``UniversalUnit_v1{G = len(gates), N_IN = n_in, N_OUT = len(outputs)}``'s four arguments, each a flat list of bits in leaf
    order (`verity.ml.circuits.evaluation.evaluate`'s form)."""
    k = index_bits(len(net.gates), net.n_in)
    if any(i >> k for g in net.gates for i in g[1:]) or any(o >> k for o in net.outputs):
        raise ValueError(f"an index does not fit the class's {k}-bit wire indices")
    if len(x) != net.n_in:
        raise ValueError(f"the netlist has {net.n_in} inputs, not {len(x)}")
    op = [b for g in net.gates for b in _bits(g[0], 2)]
    src = [b for g in net.gates for i in g[1:] for b in _bits(i, k)]
    out = [b for o in net.outputs for b in _bits(o, k)]
    return op, src, out, [b & 1 for b in x]


def _signature(S):
    _check_class(S.G, S.N_IN, S.N_OUT)
    idx = Array(index_bits(S.G, S.N_IN), BIT)
    return ((("op", Array(S.G, Array(2, BIT))), ("src", Array(S.G, Array(2, idx))), ("out", Array(S.N_OUT, idx)),
             ("x", Array(S.N_IN, BIT))), Array(S.N_OUT, BIT))


def build(C, G: int, N_IN: int, N_OUT: int, *elements):
    """The unit over `forms`: ``elements`` are G ops, 2G sources, N_OUT outputs and x, each a list of forms LSB first."""
    ops, src, outs, (x,) = elements[:G], elements[G:3 * G], elements[3 * G:3 * G + N_OUT], elements[3 * G + N_OUT:]
    wires = [[b] for b in x]
    for g in range(G):
        a = gather(C, src[2 * g], wires, 0)[0]
        b = gather(C, src[2 * g + 1], wires, 0)[0]
        t, n = ops[g]
        s = a ^ b
        wires.append([s ^ C.AND(t, C.AND(a, b) ^ s) ^ n])
    return [gather(C, idx, wires, 0)[0] for idx in outs]


def _types(G: int, N_IN: int, N_OUT: int):
    params, ret = _signature(SimpleNamespace(G=G, N_IN=N_IN, N_OUT=N_OUT))
    return [t for _, t in params], ret


@lru_cache(maxsize=None)
def traced(G: int, N_IN: int, N_OUT: int) -> TR.Traced:
    """The class's gate circuit (`trace.trace`)."""
    ps, ret = _types(G, N_IN, N_OUT)
    return TR.trace(lambda C, *e: build(C, G, N_IN, N_OUT, *e), ps, ret)


def _body(B, S, *colls):
    ps, ret = _types(S.G, S.N_IN, S.N_OUT)
    return TR.emit(traced(S.G, S.N_IN, S.N_OUT), ps, ret, B, *colls)


UniversalUnit = CompositeDefinition(
    "UniversalUnit", 1, ("G", "N_IN", "N_OUT"), _signature, _body, conformance="exact-model-tested",
    doc="The netlist (op, src, out) of G gates over N_IN inputs run on x: SSA wires (x, then each gate's output), each gate "
        "XOR / AND / XNOR / NAND of two earlier wires (a later or out-of-range one reads 0), output i wire out[i]")


def _tree(m: int, k: int) -> int:
    n = 0
    for _ in range(k):
        m = (m + 1) // 2
        n += m
    return n


def and_count(G: int, N_IN: int, N_OUT: int) -> int:
    """The unit's ANDs, in closed form: ``sum_g (2 T(N_IN + g) + 2) + N_OUT T(W)`` (the module docstring)."""
    _check_class(G, N_IN, N_OUT)
    k = index_bits(G, N_IN)
    return sum(2 * _tree(N_IN + g, k) + 2 for g in range(G)) + N_OUT * _tree(N_IN + G, k)
