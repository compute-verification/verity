"""A traced gate circuit (`verity.ml.circuits.boolean.trace.Traced`) cut by what its gates read, and replayed over forms; laid
out as `trace.definition` lays out its own (wires 0 and 1 the constants, then the inputs, then one wire per gate)."""
from __future__ import annotations

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO

__all__ = ["replay", "split", "sub"]


def sub(T: TR.Traced, keep: list[int], inputs: list[int], outputs: list[int]) -> TR.Traced:
    """The gates ``keep`` of ``T`` (indices, ascending) as a circuit of their own: its inputs the wires ``inputs`` of ``T`` in order,
    its outputs the wires ``outputs``. Every operand of a kept gate is a constant, one of ``inputs`` or a kept gate."""
    new = {0: 0, 1: 1, **{w: 2 + k for k, w in enumerate(inputs)}}
    gates: list[tuple[str, tuple[int, ...]]] = []
    for i in keep:
        kind, ops = T.gates[i]
        gates.append((kind, tuple(new[o] for o in ops)))
        new[T.first_gate + i] = 2 + len(inputs) + len(gates) - 1
    counts = {k: sum(1 for g, _ in gates if g == k) for k in ("and", "xor", "not")}
    return TR.Traced(len(inputs), gates, [new[o] for o in outputs], counts)


def split(T: TR.Traced, widths: list[int], shared: int, own: int | None) -> tuple[TR.Traced, TR.Traced, TR.Traced]:
    """``T``'s gates by the parameters they read (``widths``: bits per parameter): those reading only parameter ``shared``, those
    reading only parameter ``own`` (none when ``own`` is None), the rest. Returns the first two as circuits from their
    parameter's bits to their frontier (their gates the rest read), and the rest as a circuit from the parameters, each of the
    two followed by its frontier, to ``T``'s outputs."""
    starts = [2 + sum(widths[:p]) for p in range(len(widths))]
    wires = [list(range(s, s + n)) for s, n in zip(starts, widths)]
    sup = [0, 0] + [1 << p for p, n in enumerate(widths) for _ in range(n)]
    for _, ops in T.gates:
        m = 0
        for o in ops:
            m |= sup[o]
        sup.append(m)
    first, n = T.first_gate, len(T.gates)

    def part(p: int) -> tuple[list[int], list[int]]:
        keep = [i for i in range(n) if sup[first + i] == 1 << p]
        mine = {first + i for i in keep}
        read = {o for i, (_, ops) in enumerate(T.gates) if sup[first + i] != 1 << p for o in ops if o in mine}
        return keep, sorted(read | (mine & set(T.outputs)))

    (ks, fs), (ko, fo) = part(shared), part(own) if own is not None else ([], [])
    inputs = [w for p in range(len(widths)) for w in wires[p] + (fs if p == shared else fo if p == own else [])]
    drop = set(ks) | set(ko)
    rest = sub(T, [i for i in range(n) if i not in drop], inputs, T.outputs)
    return sub(T, ks, wires[shared], fs), sub(T, ko, wires[own] if own is not None else [], fo), rest


def replay(T: TR.Traced, C, bits: list) -> list:
    """``T``'s outputs over the forms ``bits`` (its inputs, in order), its ANDs made by ``C`` (so CSE shares those ``C`` has)."""
    w = [ZERO, ONE] + list(bits)
    for kind, ops in T.gates:
        a = w[ops[0]]
        w.append(C.AND(a, w[ops[1]]) if kind == "and" else a ^ w[ops[1]] if kind == "xor" else NOT(a))
    return [w[o] for o in T.outputs]
