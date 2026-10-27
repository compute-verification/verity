"""A Boolean Program walked against a run's own words, call by call: the training step's check against its GPU dump.

`walk(root, values, models)` evaluates each node of the root's body on the words the run had at the node's inputs and compares
what it computes with the words the run had at its output, wherever ``values`` names them. A node reads the run's words for
every value they name and what the walk computed for the rest, so each named node is checked on the run's own inputs: a
mismatch is that call's, not one carried in from an earlier call.

``values`` maps names to words, each element of the value's word view (`boolean.word_type`) one word, in leaf order: a root
parameter by its name, a node by its path. A root node's path is its ``name`` (`Builder.batch(..., name=...)`), or ``#k`` for
the k-th node when it has none; a node inside a node the walk descends into is ``PATH/NAME``, and inside member or iteration m
of a batch or scan ``PATH[m]/NAME``. A tuple's words may come as one array or as a sequence of arrays, its fields in order.

A node is evaluated, in this order of preference:

- by its family's numpy model in ``models`` (`Model`), which the Definition's own tests hold to its bits;
- on bits (`bits.evaluate_bits`), its members side by side, when it takes a parameter (the evaluator counts members by them)
  and its gate evaluations and memory are under ``bits_max`` and ``mem_bytes``;
- else body by body: the walk descends into each member's body (at most ``descend_max`` members), whose nodes it checks the
  same way.

A node it cannot evaluate (more members than ``descend_max``, a primitive that is no gate) is `Walk.unchecked`, and so is every
unnamed node that reads it; a named one hands the run's words on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
from verity.ml.circuits.evaluation.bits import evaluate_bits, peak_bytes
from verity.ml.circuits.refs import runs

__all__ = ["BITS_MAX", "Check", "MEM_BYTES", "Model", "Walk", "family", "walk"]

#: a family's numpy model: ``model(statics, *args)``, each argument the words of its members (member axis first, then the word
#: view's shape) -> the output's words, member axis first, as one array or one a tuple field
Model = Callable[..., Any]
#: gate evaluations (gates x members) a node is evaluated on bits within
BITS_MAX = 1 << 33
#: bytes `bits.peak_bytes` may bound a node's bit evaluation by
MEM_BYTES = 8 << 30
_FIRST = 5
_BYTES = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}


@dataclass
class Check:
    """One named node: its words against the run's, ``mismatches`` of ``words``; ``first`` the first few as (member, element,
    computed word, the run's word)."""

    path: str
    family: str
    via: str
    members: int
    words: int
    mismatches: int
    first: list = field(default_factory=list)


@dataclass
class Walk:
    """``checks`` the named nodes the walk evaluated, ``computed`` the paths of the unnamed ones, ``unchecked`` (path, why) of
    those it could not evaluate."""

    checks: list[Check] = field(default_factory=list)
    computed: list[str] = field(default_factory=list)
    unchecked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def mismatches(self) -> int:
        return sum(c.mismatches for c in self.checks)

    def as_dict(self) -> dict:
        return {"checks": [asdict(c) for c in self.checks], "computed": len(self.computed), "unchecked": self.unchecked,
                "words": sum(c.words for c in self.checks), "mismatches": self.mismatches}


def family(fn: Any) -> str:
    """The Definition id without its statics: ``EagerAttentionTorch_v1``."""
    return getattr(fn, "base_id", None) or fn.id


def _uniform(t: Any) -> tuple[int, int] | None:
    """(width, elements) when every element of t is one byte-multiple width."""
    ws = BL.elements(t)
    return (ws[0], len(ws)) if ws and ws[0] in _BYTES and ws.count(ws[0]) == len(ws) else None


def _bits(words: np.ndarray, t: Any) -> np.ndarray:
    """(members, elements) words as (members, leaves) 0/1 bits, least significant first."""
    words = np.asarray(words).reshape(words.shape[0], -1)
    n, u = words.shape[0], _uniform(t)
    if u is not None:
        w, e = u
        raw = np.ascontiguousarray(words.astype(_BYTES[w])).view(np.uint8).reshape(n, e, w // 8)
        return np.unpackbits(raw, axis=2, bitorder="little").reshape(n, e * w)
    ws = np.asarray(BL.elements(t), dtype=np.int64)
    shift = np.arange(int(ws.sum()), dtype=np.int64) - np.repeat(np.r_[0, np.cumsum(ws)[:-1]], ws)
    rep = np.repeat(words.astype(np.uint64), ws, axis=1)
    return ((rep >> shift.astype(np.uint64)) & np.uint64(1)).astype(np.uint8)


def _words(bits: np.ndarray, t: Any) -> np.ndarray:
    """(members, leaves) bits as (members, elements) words."""
    n, u = bits.shape[0], _uniform(t)
    if u is not None:
        w, e = u
        raw = np.packbits(np.ascontiguousarray(bits, dtype=np.uint8).reshape(n, e, w), axis=2, bitorder="little")
        return np.ascontiguousarray(raw).view(_BYTES[w]).reshape(n, e).astype(np.uint64)
    ws = np.asarray(BL.elements(t), dtype=np.int64)
    if not ws.size:
        return np.zeros((n, 0), dtype=np.uint64)
    starts = np.r_[0, np.cumsum(ws)[:-1]]
    shift = np.arange(int(ws.sum()), dtype=np.int64) - np.repeat(starts, ws)
    return np.add.reduceat(bits.astype(np.uint64) << shift.astype(np.uint64), starts, axis=1)


def _given(v: Any, t: Any, path: str) -> np.ndarray:
    """``values``' words for one value of type t, as its leaves' bits (1-D)."""
    parts = v if isinstance(v, (list, tuple)) else (v,)
    words = np.concatenate([np.asarray(p).reshape(-1).astype(np.uint64) for p in parts]).reshape(1, -1)
    n = len(BL.elements(t))
    if words.size != n:
        raise ValueError(f"program_walk: {path}: {words.size} words given, its type has {n} elements")
    bits = _bits(words, t)
    if np.any(_words(bits, t) != words):
        raise ValueError(f"program_walk: {path}: a word wider than its element")
    return bits.reshape(-1)


class _Walker:
    def __init__(self, values: Mapping[str, Any], models: Mapping[str, Model], bits_max: int, mem_bytes: int,
                 descend_max: int) -> None:
        self.values, self.models = values, models
        self.bits_max, self.mem_bytes, self.descend_max = bits_max, mem_bytes, descend_max
        self.out = Walk()

    def body(self, spec: SpecializedDefinition, params: Sequence[np.ndarray], prefix: str) -> np.ndarray | None:
        """``spec``'s body walked on one member's parameter bits (1-D each): its return's bits, None if it reads an unchecked
        value."""
        b = spec.body
        env: dict[tuple[str, int], np.ndarray | None] = {("p", i): p for i, p in enumerate(params)}
        for k, node in enumerate(b.nodes):
            name = node.name or f"#{k}"
            env[("n", k)] = self.node(node, f"{prefix}/{name}" if prefix else name, env)
        return self.gather(b.ret, env)

    @staticmethod
    def gather(refs: Any, env: Mapping[tuple[str, int], np.ndarray | None]) -> np.ndarray | None:
        parts = []
        for space, idx, lo, hi in runs(refs):
            src = env[("p" if space == "p" else "n", idx)]
            if src is None:
                return None
            parts.append(src[lo:hi])
        return np.concatenate(parts) if parts else np.zeros(0, np.uint8)

    def members(self, node: Any, n: int, env: Mapping) -> list[np.ndarray] | None:
        """Per parameter of a call's or batch's callee, its members' bits (members, leaves)."""
        out = []
        for j, a in enumerate(node.args):
            ax = node.axes[j] if node.form == "batch" else None
            if ax is None or ax == 0:
                g = self.gather(a.refs, env)
                if g is None:
                    return None
                out.append(np.broadcast_to(g, (n, g.size)) if ax is None else g.reshape(n, -1))
            else:
                rows = [self.gather(a.take(ax, m).refs, env) for m in range(n)]
                if any(r is None for r in rows):
                    return None
                out.append(np.stack(rows))
        return out

    def node(self, node: Any, path: str, env: Mapping) -> np.ndarray | None:
        n = node.n if node.form in ("batch", "scan") else 1
        got, via = (self.scan(node, path, env), "scan") if node.form == "scan" else (None, "")
        if node.form != "scan":
            args = self.members(node, n, env)
            if args is None:
                self.out.unchecked.append((path, "reads a value the walk could not evaluate"))
            else:
                got, via = self.evaluate(node.fn, args, n, path)
        if path in self.values:
            want = _given(self.values[path], node.out, path)
            if got is not None:
                self.check(path, node.fn, via, n, node.out, got.reshape(-1), want)
            return want
        if got is None:
            return None
        self.out.computed.append(path)
        return got.reshape(-1)

    def scan(self, node: Any, path: str, env: Mapping) -> np.ndarray | None:
        init = self.gather(node.args[0].refs, env)
        xs = [self.gather(a.refs, env) for a in node.args[1:1 + node.n_xs]]
        shared = [self.gather(a.refs, env) for a in node.args[1 + node.n_xs:]]
        if init is None or any(x is None for x in xs + shared):
            self.out.unchecked.append((path, "reads a value the walk could not evaluate"))
            return None
        carry, ys = init, []
        for i in range(node.n):
            step = [carry] + [x.reshape(node.n, -1)[i] for x in xs] + shared
            got, _ = self.evaluate(node.fn, [s.reshape(1, -1) for s in step], 1, f"{path}[{i}]")
            if got is None:
                return None
            got = got.reshape(-1)
            carry, ys = got[:init.size], ys + [got[init.size:]]
        return np.concatenate([carry, *ys])

    def evaluate(self, fn: Any, args: list[np.ndarray], n: int, path: str) -> tuple[np.ndarray | None, str]:
        """``fn`` on n members' parameter bits: their outputs' bits (n, leaves), and how."""
        if isinstance(fn, PrimitiveDefinition):
            return self.gate(fn, args, n, path), "gate"
        params, ret = fn.signature()
        model = self.models.get(family(fn))
        if model is not None:
            words = [_words(a, t).astype(type_dtype(BL.word_type(t))).reshape(n, *type_shape(BL.word_type(t)))
                     for a, (_, t) in zip(args, params, strict=True)]
            res = model(dict(fn.bindings), *words)
            parts = res if isinstance(res, tuple) else (res,)
            out = np.concatenate([np.asarray(p).reshape(n, -1).astype(np.uint64) for p in parts], axis=1)
            return _bits(out, ret), "model"
        if args and BL.is_boolean(fn) and int(fn.gates) * n <= self.bits_max and peak_bytes(fn, n) <= self.mem_bytes:
            return np.asarray(evaluate_bits(fn, *args)).reshape(n, -1).astype(np.uint8), "bits"
        if n > self.descend_max:
            self.out.unchecked.append((path, f"{fn.id}: {n} members with no model, over the bit budget"))
            return None, ""
        rows = []
        for m in range(n):
            r = self.body(fn, [a[m] for a in args], f"{path}[{m}]" if n > 1 else path)
            if r is None:
                return None, ""
            rows.append(r)
        return np.stack(rows), "body"

    def gate(self, fn: PrimitiveDefinition, args: list[np.ndarray], n: int, path: str) -> np.ndarray | None:
        if BL.is_constant_bit(fn):
            return np.full((n, 1), fn.evaluate(), dtype=np.uint8)
        gk = BL.gate_kind(fn)
        if gk is None:
            self.out.unchecked.append((path, f"{fn.id} is not a gate or a constant bit"))
            return None
        ops = np.concatenate(args, axis=1) if args else np.zeros((n, 0), np.uint8)
        if gk[0] == "not":
            return (ops ^ 1).astype(np.uint8)
        return (np.bitwise_and if gk[0] == "and" else np.bitwise_xor).reduce(ops, axis=1, keepdims=True).astype(np.uint8)

    def check(self, path: str, fn: Any, via: str, n: int, t: Any, got: np.ndarray, want: np.ndarray) -> None:
        g, w = _words(got.reshape(1, -1), t)[0], _words(want.reshape(1, -1), t)[0]
        bad = np.flatnonzero(g != w)
        per = max(1, g.size // max(1, n))
        first = [(int(i // per), int(i % per), int(g[i]), int(w[i])) for i in bad[:_FIRST]]
        self.out.checks.append(Check(path, family(fn), via, n, int(g.size), int(bad.size), first))


def walk(root: SpecializedDefinition, values: Mapping[str, Any], models: Mapping[str, Model] | None = None, *,
         bits_max: int = BITS_MAX, mem_bytes: int = MEM_BYTES, descend_max: int = 64) -> Walk:
    """``root``'s body walked against ``values``, every root parameter among them (the module's docstring)."""
    params, _ = root.signature()
    missing = [name for name, _ in params if name not in values]
    if missing:
        raise ValueError(f"program_walk: no words for {root.id}'s parameters {missing}")
    w = _Walker(values, models or {}, bits_max, mem_bytes, descend_max)
    w.body(root, [_given(values[name], t, name) for name, t in params], "")
    return w.out
