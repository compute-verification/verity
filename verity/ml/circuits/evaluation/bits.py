"""Bit-sliced evaluation of Boolean Definitions (`verity.ml.circuits.boolean`): 64 instances per machine word.

`compile(target)` reads each body once per specialization into local index arrays over the body's own wires (wire 0 the
constant 0, wire 1 the constant 1, then its parameter leaves, then each node's output leaves). A run of consecutive gate nodes
is grouped by depth and kind, so a group is one numpy operation (``&``, ``^``, ``~``) over rows of ``uint64``; a call runs its
callee's program on the rows of its arguments, the members of a batch of a composite run as one program over their rows side by
side (as many at a time as keep the callee's wires within `_ROW_BUDGET`), and a scan runs its step once per iteration. Memory is
one body's wires at a time, not the flattened circuit's.

    evaluate_bits(target, *inputs)    the Definition's own signature: one 0/1 array per parameter, `evaluate_batch`'s shapes
    evaluate_words(target, *inputs)   its word view (`boolean.word_type`): element j of a port is one word, least significant
                                      bit first
    peak_bytes(target, n)             a bound on the bytes `evaluate_bits` holds at once for n instances, its inputs aside

`evaluate_batch` routes a Boolean composite here when no kernel is registered for it; the reference stays the oracle, and the
tests compare the two.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.types import Type

from .reference import refuse_registered_tables

__all__ = ["Compiled", "compile", "evaluate_bits", "evaluate_words", "peak_bytes"]

_KINDS = ("and", "xor", "not")
_ROW_BUDGET = 1 << 28  # bytes of one body's wire rows per chunk of instances
_ARRAY = 256  # bytes of one ndarray's header, shape and strides (a scan keeps two per iteration)
_SLACK = 1 << 16  # bytes of numpy's iteration buffers and the frames' own objects


@dataclass
class Compiled:
    """One body: ``steps`` in order, each ``("gates", [(kind, out, a, b), ...])`` (local wire indices, every operand computed
    before its group), ``("call", callee, [member argument indices], out base, out leaves)`` or ``("scan", step, init, xs,
    shared, n, carry leaves, out base)``; an n-input gate is a balanced tree of two-input operations on scratch wires after the
    node outputs. ``counts`` are the two-input operations of the flattened circuit, by kind (an n-input gate counts n - 1)."""

    target: str
    params: tuple[Type, ...]
    ret: Type
    size: int
    n_params: int
    steps: list[tuple] = field(repr=False)
    ret_idx: np.ndarray = field(repr=False)
    counts: dict[str, int] = field(default_factory=dict)
    max_size: int = 0


_COMPILED: "weakref.WeakKeyDictionary[SpecializedDefinition, Compiled]" = weakref.WeakKeyDictionary()


def _spec(target: Any) -> SpecializedDefinition:
    if isinstance(target, CompositeDefinition):
        target = bind(target)
    if not isinstance(target, SpecializedDefinition) or not BL.is_boolean(target):
        raise TypeError(f"{getattr(target, 'id', target)!r} is not a Boolean composite Definition")
    return target


def _group(gates: list[tuple[int, int, np.ndarray, np.ndarray]], size: int) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """A run of gate nodes, (kind, out, a, b) each, as groups of equal depth and kind (depth counted within the run)."""
    depth = np.zeros(size, dtype=np.int64)
    for _, out, a, b in gates:
        depth[out] = np.maximum(depth[a], depth[b]) + 1
    kinds = np.concatenate([np.full(len(o), k, np.int8) for k, o, _, _ in gates])
    outs = np.concatenate([o for _, o, _, _ in gates])
    a = np.concatenate([x for _, _, x, _ in gates])
    b = np.concatenate([y for _, _, _, y in gates])
    order = np.lexsort((kinds, depth[outs]))
    kinds, outs, a, b = kinds[order], outs[order], a[order], b[order]
    key = depth[outs] * 4 + kinds
    cuts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    return [(int(kinds[s]), outs[s:e], a[s:e], b[s:e]) for s, e in zip(cuts[:-1], cuts[1:])]


def compile(target: Any) -> Compiled:  # noqa: A001 (the module's verb; `builtins.compile` is not used here)
    spec = _spec(target)
    hit = _COMPILED.get(spec)
    if hit is not None:
        return hit
    body = spec.body
    p_base, k = [], 2
    for _, t in body.params:
        p_base.append(k)
        k += t.leaves
    n_params, node_base, const = k - 2, [], {}
    for q, nd in enumerate(body.nodes):
        node_base.append(k)
        k += nd.out.leaves
        if isinstance(nd.fn, PrimitiveDefinition) and BL.is_constant_bit(nd.fn):
            const[q] = nd.fn.evaluate()

    def idx(refs) -> np.ndarray:
        return np.fromiter((p_base[i] + j if s == "p" else const[i] if i in const else node_base[i] + j for s, i, j in refs),
                           dtype=np.int64, count=refs.count)

    def member_refs(nd, m: int) -> np.ndarray:
        out = []
        for a, ax in zip(nd.args, nd.axes):
            if ax is None:
                out.append(idx(a.refs))
            elif ax == 0:
                L = a.type.elem.leaves
                out.append(idx(a.refs.slice(m * L, (m + 1) * L)))
            else:
                out.append(idx(a.take(ax, m).refs))
        return np.concatenate(out) if out else np.empty(0, np.int64)

    steps: list[tuple] = []
    run: list[tuple] = []
    counts = dict.fromkeys(_KINDS, 0)
    max_size = k

    def flush() -> None:
        if run:
            steps.append(("gates", _group(run, k)))
            run.clear()

    for q, nd in enumerate(body.nodes):
        fn, base = nd.fn, node_base[q]
        if isinstance(fn, PrimitiveDefinition):
            if q in const:
                continue
            gk = BL.gate_kind(fn)
            if gk is None:
                raise TypeError(f"{spec.id} node {q}: {fn.id} is not a Boolean gate or constant bit")
            kind, arity = gk
            if nd.form == "primitive":
                ops = np.concatenate([idx(a.refs) for a in nd.args]).reshape(1, arity)
            else:
                ops = np.stack([member_refs(nd, m) for m in range(nd.n)]).reshape(nd.n, arity)
            out = np.arange(base, base + len(ops), dtype=np.int64)
            ki, cols = _KINDS.index(kind), [ops[:, j] for j in range(arity)]
            while len(cols) > 2:
                pairs, cols = cols, []
                for j in range(0, len(pairs) - 1, 2):
                    tmp = np.arange(k, k + len(ops), dtype=np.int64)
                    k += len(ops)
                    run.append((ki, tmp, pairs[j], pairs[j + 1]))
                    cols.append(tmp)
                if len(pairs) % 2:
                    cols.append(pairs[-1])
            run.append((ki, out, cols[0], cols[-1]))
            counts[kind] += len(ops) * max(1, arity - 1)
            continue
        flush()
        child = compile(fn)
        max_size = max(max_size, child.max_size)
        if nd.form == "call":
            members = [np.concatenate([idx(a.refs) for a in nd.args]) if nd.args else np.empty(0, np.int64)]
        elif nd.form == "batch":
            members = [member_refs(nd, m) for m in range(nd.n)]
        else:
            xs = [(idx(a.refs), a.type.elem.leaves) for a in nd.args[1:1 + nd.n_xs]]
            shared = [idx(a.refs) for a in nd.args[1 + nd.n_xs:]]
            steps.append(("scan", child, idx(nd.args[0].refs), xs, shared, nd.n, nd.out.fields[0].leaves, base))
            for kk in _KINDS:
                counts[kk] += nd.n * child.counts[kk]
            continue
        steps.append(("call", child, np.stack(members), base, fn.out_leaves))
        for kk in _KINDS:
            counts[kk] += len(members) * child.counts[kk]
    flush()
    max_size = max(max_size, k)
    params, ret = spec.signature()
    hit = _COMPILED[spec] = Compiled(spec.id, tuple(t for _, t in params), ret, k, n_params, steps, idx(body.ret), counts,
                                     max_size)
    return hit


def _run(c: Compiled, args: np.ndarray) -> np.ndarray:
    """``args``: (parameter leaves, words) of packed instances; returns (return leaves, words)."""
    words = args.shape[1]
    W = np.empty((c.size, words), dtype=np.uint64)
    W[0] = 0
    W[1] = ~np.uint64(0)
    W[2:2 + c.n_params] = args
    for step in c.steps:
        if step[0] == "gates":
            for kind, out, a, b in step[1]:
                if kind == 0:
                    W[out] = W[a] & W[b]
                elif kind == 1:
                    W[out] = W[a] ^ W[b]
                else:
                    W[out] = ~W[a]
        elif step[0] == "call":
            _, child, members, base, w = step
            per = max(1, _ROW_BUDGET // (8 * child.size * words))
            for lo in range(0, len(members), per):
                m = members[lo:lo + per]
                n = len(m)
                rows = W[m].transpose(1, 0, 2).reshape(m.shape[1], n * words)
                out = _run(child, rows).reshape(w, n, words).transpose(1, 0, 2).reshape(n * w, words)
                W[base + lo * w:base + (lo + n) * w] = out
        else:
            _, child, init, xs, shared, n, cl, base = step
            carry = W[init]
            ys = []
            for m in range(n):
                parts = [carry] + [W[r[m * e:(m + 1) * e]] for r, e in xs] + [W[r] for r in shared]
                r = _run(child, np.concatenate(parts) if parts else np.empty((0, words), np.uint64))
                carry = r[:cl]
                ys.append(r[cl:])
            W[base:base + cl] = carry
            if ys:
                y = np.concatenate(ys)
                W[base + cl:base + cl + len(y)] = y
    return W[c.ret_idx]


def _pack(bits: np.ndarray) -> np.ndarray:
    """(N, leaves) of 0/1 -> (leaves, ceil(N / 64)) uint64, instance i at bit i % 64 of word i // 64."""
    n = bits.shape[0]
    b = np.zeros((bits.shape[1], -(-n // 64) * 8), dtype=np.uint8)
    b[:, :-(-n // 8)] = np.packbits(bits.T, axis=1, bitorder="little")
    return b.view("<u8").astype(np.uint64, copy=False)


def _unpack(rows: np.ndarray, n: int) -> np.ndarray:
    """(leaves, words) uint64 -> (N, leaves) uint8 of 0/1."""
    b = np.unpackbits(np.ascontiguousarray(rows.astype("<u8")).view(np.uint8), axis=1, bitorder="little")
    return b[:, :n].T


def _evaluate(c: Compiled, bits: np.ndarray) -> np.ndarray:
    n = bits.shape[0]
    chunk = 64 * max(1, _ROW_BUDGET // (8 * c.max_size))
    out = np.empty((n, len(c.ret_idx)), dtype=np.uint8)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        out[lo:hi] = _unpack(_run(c, _pack(bits[lo:hi])), hi - lo)
    return out


def _leaf_bits(c: Compiled, inputs: Sequence[Any]) -> tuple[np.ndarray, int]:
    refuse_registered_tables(c.target, inputs, "evaluate_bits")
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


def _run_peak(c: Compiled, words: int, memo: dict) -> int:
    """The bytes `_run(c, args)` holds at once beyond ``args``: its wire rows, and at each step what the step allocates (a gate
    group's operand copies; a call's gathered argument rows, its callee's run and the outputs it copies back, per chunk of members;
    a scan's iterates) on top of what earlier steps' locals still hold, and the returned rows."""
    key = (id(c), words)
    if key in memo:
        return memo[key]
    wb = 8 * words
    held_call = held_scan = top = 0
    for step in c.steps:
        if step[0] == "gates":
            top = max(top, held_call + held_scan + 3 * wb * max(len(g[1]) for g in step[1]))
        elif step[0] == "call":
            _, child, members, _, w = step
            k, m = min(len(members), max(1, _ROW_BUDGET // (8 * child.size * words))), members.shape[1]
            held_call = max(held_call, k * (m + w) * wb)
            own = max(2 * k * m * wb, k * m * wb + _run_peak(child, k * words, memo), k * (m + 2 * w) * wb)
            top = max(top, held_scan + held_call + own)
        else:
            _, child, _, _, _, n, cl, _ = step
            cr, cin = len(child.ret_idx), child.n_params
            after = (n * (2 * cr - cl) + cin) * wb + n * (2 * _ARRAY + 8)
            loop = (cl + n * cr + 2 * cin) * wb + n * (2 * _ARRAY + 8) + _run_peak(child, words, memo)
            top = max(top, held_call + held_scan + max(loop, after))
            held_scan = max(held_scan, after)
    top = max(top, held_call + held_scan + len(c.ret_idx) * wb)
    memo[key] = 8 * c.size * words + top
    return memo[key]


def peak_bytes(target: Any, n: int = 1) -> int:
    """An upper bound on the bytes `evaluate_bits(target, *inputs)` holds at once for ``n`` instances, the inputs aside:
    `_leaf_bits`' two copies of the leaves, `_evaluate`'s output, and in one chunk of instances the packed parameter words,
    `_run`'s rows (`_run_peak`) and `_unpack`'s copies, plus `_SLACK` for numpy's buffers. When the target's own body is wide (a GEMM's weight
    leaves) nearly all of it is 17 bytes per parameter leaf and instance word (the leaves, their packed words, their wire rows)
    and 8 per other wire; the transients are bounded by `_ROW_BUDGET` per level of nesting."""
    c = compile(target)
    P, R = c.n_params, len(c.ret_idx)
    nc = min(n, 64 * max(1, _ROW_BUDGET // (8 * c.max_size)))
    w = -(-nc // 64)
    held = n * (P + R)
    return _SLACK + max(2 * n * P, held + P * (-(-nc // 8) + 8 * w), held + 8 * P * w + _run_peak(c, w, {}), held + 80 * R * w)


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
    refuse_registered_tables(c.target, inputs, "evaluate_words")
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
