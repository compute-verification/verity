"""Bit-sliced evaluation of Boolean Definitions (`verity.ir.boolean`): 64 instances per machine word.

`compile(target)` flattens a Boolean Definition once: each body is read once per specialization into local index arrays, calls,
batches and scans are inlined into one wire space, and the gates are grouped by depth and kind, so a group is one numpy
operation (``&``, ``^``, ``~``) over rows of ``uint64``. Wire 0 is the constant 0 and wire 1 the constant 1.

    evaluate_bits(target, *inputs)    the Definition's own signature: one 0/1 array per parameter, `evaluate_batch`'s shapes
    evaluate_words(target, *inputs)   its word view (`boolean.word_type`): element j of a port is one word, least significant
                                      bit first

`evaluate_batch` routes a Boolean composite here when no kernel is registered for it; the reference stays the oracle, and the
tests compare the two.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ir.types import Type

__all__ = ["Compiled", "compile", "evaluate_bits", "evaluate_words"]

_KIND_CODE = {"and": 0, "xor": 1, "not": 2}
_ROW_BUDGET = 1 << 28  # bytes of wire rows per chunk of instances


@dataclass
class _Local:
    """One body in its own wire space: parameter leaves first, then each node's output leaves."""

    n_params: int
    node_base: list[int]
    ops: list[tuple]
    ret: np.ndarray
    size: int


_LOCAL: "weakref.WeakKeyDictionary[SpecializedDefinition, _Local]" = weakref.WeakKeyDictionary()


def _local(spec: SpecializedDefinition) -> _Local:
    hit = _LOCAL.get(spec)
    if hit is not None:
        return hit
    body = spec.body
    p_base, k = [], 0
    for _, t in body.params:
        p_base.append(k)
        k += t.leaves
    n_params, node_base = k, []
    for nd in body.nodes:
        node_base.append(k)
        k += nd.out.leaves

    def idx(refs) -> np.ndarray:
        return np.fromiter(((p_base[i] if s == "p" else node_base[i]) + j for s, i, j in refs), dtype=np.int64,
                           count=refs.count)

    def member_refs(nd, m: int) -> list[np.ndarray]:
        out = []
        for a, ax in zip(nd.args, nd.axes):
            if ax is None:
                out.append(idx(a.refs))
            elif ax == 0:
                L = a.type.elem.leaves
                out.append(idx(a.refs.slice(m * L, (m + 1) * L)))
            else:
                out.append(idx(a.take(ax, m).refs))
        return out

    ops: list[tuple] = []
    for q, nd in enumerate(body.nodes):
        fn, base = nd.fn, node_base[q]
        if isinstance(fn, PrimitiveDefinition):
            if BL.is_constant_bit(fn):
                ops.append(("const", base, nd.n if nd.form == "batch" else 1, fn.evaluate()))
                continue
            if not BL.is_gate(fn):
                raise TypeError(f"{spec.id} node {q}: {fn.id} is not a Boolean gate or constant bit")
            kind, arity = BL.KINDS[fn.id]
            if nd.form == "primitive":
                operands = np.concatenate([idx(a.refs) for a in nd.args]).reshape(1, arity)
            else:
                operands = np.stack([np.concatenate(member_refs(nd, m)) for m in range(nd.n)]).reshape(nd.n, arity)
            ops.append(("gate", _KIND_CODE[kind], base, operands))
        elif nd.form == "call":
            ops.append(("call", fn, base, [np.concatenate([idx(a.refs) for a in nd.args])]))
        elif nd.form == "batch":
            ops.append(("call", fn, base, [np.concatenate(member_refs(nd, m)) for m in range(nd.n)]))
        else:
            c = nd.out.fields[0].leaves
            xs = [(idx(a.refs), a.type.elem.leaves) for a in nd.args[1:1 + nd.n_xs]]
            shared = [idx(a.refs) for a in nd.args[1 + nd.n_xs:]]
            ops.append(("scan", fn, base, idx(nd.args[0].refs), xs, shared, nd.n, c))
    hit = _LOCAL[spec] = _Local(n_params, node_base, ops, idx(body.ret), k)
    return hit


@dataclass
class Compiled:
    """A flattened Boolean Definition: ``groups`` of (kind, out wires, operand wires a, b), in an order where every operand is
    computed before it is read."""

    target: str
    params: tuple[Type, ...]
    ret: Type
    n_wires: int
    inputs: np.ndarray
    outputs: np.ndarray
    groups: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = field(repr=False)
    counts: dict[str, int] = field(default_factory=dict)


class _Flat:
    def __init__(self) -> None:
        self.n = 2
        self.kinds: list[int] = []
        self.outs: list[np.ndarray] = []
        self.ops: list[np.ndarray] = []

    def alloc(self, k: int) -> np.ndarray:
        out = np.arange(self.n, self.n + k, dtype=np.int64)
        self.n += k
        return out

    def inline(self, spec: SpecializedDefinition, args: np.ndarray) -> np.ndarray:
        L = _local(spec)
        g = np.empty(L.size, dtype=np.int64)
        g[:L.n_params] = args
        for op in L.ops:
            if op[0] == "gate":
                _, kind, base, operands = op
                n, arity = operands.shape
                out = self.alloc(n)
                g[base:base + n] = out
                a = g[operands]
                self.kinds.append(np.full(n, kind, dtype=np.int8))
                self.outs.append(out)
                self.ops.append(a if arity == 2 else np.concatenate([a, a], axis=1))
            elif op[0] == "const":
                _, base, n, v = op
                g[base:base + n] = v
            elif op[0] == "call":
                _, fn, base, members = op
                w = fn.out_leaves
                for m, refs in enumerate(members):
                    child = self.inline(fn, g[refs])
                    g[base + m * w:base + (m + 1) * w] = child
            else:
                _, fn, base, init, xs, shared, n, c = op
                carry = g[init]
                ys = []
                for m in range(n):
                    a = [carry] + [g[r[m * e:(m + 1) * e]] for r, e in xs] + [g[r] for r in shared]
                    r = self.inline(fn, np.concatenate(a) if a else np.empty(0, dtype=np.int64))
                    carry, y = r[:c], r[c:]
                    ys.append(y)
                g[base:base + c] = carry
                if ys:
                    g[base + c:base + c + sum(len(y) for y in ys)] = np.concatenate(ys)
        return g[L.ret]


_COMPILED: "weakref.WeakKeyDictionary[SpecializedDefinition, Compiled]" = weakref.WeakKeyDictionary()


def _spec(target: Any) -> SpecializedDefinition:
    if isinstance(target, CompositeDefinition):
        target = bind(target)
    if not isinstance(target, SpecializedDefinition) or not BL.is_boolean(target):
        raise TypeError(f"{getattr(target, 'id', target)!r} is not a Boolean composite Definition")
    return target


def compile(target: Any) -> Compiled:  # noqa: A001 (the module's verb; `builtins.compile` is not used here)
    spec = _spec(target)
    hit = _COMPILED.get(spec)
    if hit is not None:
        return hit
    params, ret = spec.signature()
    F = _Flat()
    inputs = F.alloc(sum(t.leaves for _, t in params))
    outputs = F.inline(spec, inputs)
    if F.kinds:
        kinds, outs, ops = np.concatenate(F.kinds), np.concatenate(F.outs), np.concatenate(F.ops)
    else:
        kinds, outs, ops = np.empty(0, np.int8), np.empty(0, np.int64), np.empty((0, 2), np.int64)
    depth = np.zeros(F.n, dtype=np.int64)
    for o, a in zip(F.outs, F.ops):
        depth[o] = np.maximum(depth[a[:, 0]], depth[a[:, 1]]) + 1
    order = np.lexsort((kinds, depth[outs]))
    kinds, outs, ops = kinds[order], outs[order], ops[order]
    key = depth[outs] * 4 + kinds
    cuts = np.flatnonzero(np.diff(key)) + 1 if len(key) else np.empty(0, np.int64)
    groups = [(int(kinds[s]), outs[s:e], ops[s:e, 0], ops[s:e, 1])
              for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(key)]) if e > s]
    counts = {k: int(np.count_nonzero(kinds == c)) for k, c in _KIND_CODE.items()}
    hit = _COMPILED[spec] = Compiled(spec.id, tuple(t for _, t in params), ret, F.n, inputs, outputs, groups, counts)
    return hit


def _run(c: Compiled, rows: np.ndarray) -> np.ndarray:
    """``rows``: (input leaves, words) of packed instances; returns (output leaves, words)."""
    W = np.zeros((c.n_wires, rows.shape[1]), dtype=np.uint64)
    W[1] = ~np.uint64(0)
    W[c.inputs] = rows
    for kind, out, a, b in c.groups:
        if kind == 0:
            W[out] = W[a] & W[b]
        elif kind == 1:
            W[out] = W[a] ^ W[b]
        else:
            W[out] = ~W[a]
    return W[c.outputs]


def _pack(bits: np.ndarray) -> np.ndarray:
    """(N, leaves) of 0/1 -> (leaves, ceil(N / 64)) uint64, instance i at bit i % 64 of word i // 64."""
    n = bits.shape[0]
    words = -(-n // 64)
    b = np.zeros((bits.shape[1], words * 64), dtype=np.uint8)
    b[:, :n] = bits.T
    return np.packbits(b, axis=1, bitorder="little").view("<u8").astype(np.uint64, copy=False)


def _unpack(rows: np.ndarray, n: int) -> np.ndarray:
    """(leaves, words) uint64 -> (N, leaves) uint8 of 0/1."""
    b = np.unpackbits(np.ascontiguousarray(rows.astype("<u8")).view(np.uint8), axis=1, bitorder="little")
    return b[:, :n].T


def _evaluate(c: Compiled, bits: np.ndarray) -> np.ndarray:
    n = bits.shape[0]
    chunk = 64 * max(1, _ROW_BUDGET // (8 * c.n_wires))
    out = np.empty((n, len(c.outputs)), dtype=np.uint8)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        out[lo:hi] = _unpack(_run(c, _pack(bits[lo:hi])), hi - lo)
    return out


def _leaf_bits(c: Compiled, inputs: Sequence[Any]) -> tuple[np.ndarray, int]:
    if len(inputs) != len(c.params):
        raise TypeError(f"{c.target}: expected {len(c.params)} input arrays, got {len(inputs)}")
    cols, n = [], None
    for j, (a, t) in enumerate(zip(inputs, c.params)):
        a = np.asarray(a)
        if a.ndim == 0 or int(np.prod(a.shape[1:], dtype=np.int64)) != t.leaves or (n is not None and a.shape[0] != n):
            raise ValueError(f"{c.target} argument {j}: shape {a.shape}, expected (N, {t.leaves} leaves)")
        n = a.shape[0]
        a = a.reshape(n, t.leaves)
        if a.size and (a.dtype.kind not in "uib" or a.min() < 0 or a.max() > 1):
            raise ValueError(f"{c.target} argument {j}: a Boolean port carries 0/1 leaves")
        cols.append(a.astype(np.uint8))
    n = 0 if n is None else n
    return (np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), np.uint8)), n


def evaluate_bits(target: Any, *inputs: Any) -> np.ndarray:
    """``N`` instances of a Boolean Definition in its own signature: inputs and output as `evaluate_batch` shapes them, every
    leaf 0 or 1."""
    from .batch import type_shape

    c = compile(target)
    bits, n = _leaf_bits(c, inputs)
    return _evaluate(c, bits).reshape((n, *type_shape(c.ret)))


def _to_bits(words: np.ndarray, widths: list[int]) -> np.ndarray:
    n = words.shape[0]
    out = np.empty((n, sum(widths)), dtype=np.uint8)
    k = 0
    for e, w in enumerate(widths):
        col = words[:, e].astype(np.uint64)
        for j in range(w):
            out[:, k + j] = (col >> np.uint64(j)) & np.uint64(1)
        k += w
    return out


def _to_words(bits: np.ndarray, widths: list[int]) -> np.ndarray:
    out = np.zeros((bits.shape[0], len(widths)), dtype=np.uint64)
    k = 0
    for e, w in enumerate(widths):
        for j in range(w):
            out[:, e] |= bits[:, k + j].astype(np.uint64) << np.uint64(j)
        k += w
    return out


def evaluate_words(target: Any, *inputs: Any) -> np.ndarray:
    """``N`` instances of a Boolean Definition through its word view: one array per parameter shaped as `evaluate_batch` shapes
    ``word_type(param)``, the output likewise (``word_type(ret)``)."""
    from .batch import type_dtype, type_shape

    c = compile(target)
    if len(inputs) != len(c.params):
        raise TypeError(f"{c.target}: expected {len(c.params)} input arrays, got {len(inputs)}")
    cols, n = [], None
    for j, (a, t) in enumerate(zip(inputs, c.params)):
        widths = BL.elements(t)
        if max(widths, default=1) > 64:
            raise ValueError(f"{c.target} argument {j}: elements wider than 64 bits")
        a = np.asarray(a)
        if a.ndim == 0 or (n is not None and a.shape[0] != n) or int(np.prod(a.shape[1:], dtype=np.int64)) != len(widths):
            raise ValueError(f"{c.target} argument {j}: shape {a.shape}, expected (N, {len(widths)} elements)")
        n = a.shape[0]
        a = a.reshape(n, len(widths))
        if a.size and a.dtype.kind == "i" and a.min() < 0:
            raise ValueError(f"{c.target} argument {j}: negative word")
        bits = _to_bits(a, widths)
        if a.size and np.any(_to_words(bits, widths) != a.astype(np.uint64)):
            raise ValueError(f"{c.target} argument {j}: a word wider than its element")
        cols.append(bits)
    n = 0 if n is None else n
    bits = np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), np.uint8)
    wret = BL.word_type(c.ret)
    out = _to_words(_evaluate(c, bits), BL.elements(c.ret))
    return out.astype(type_dtype(wret)).reshape((n, *type_shape(wret)))
