"""The top-p keep word as word gates: `TopPKeepWord_v1{V,S}`, the split top-p pipeline at one split count S as a composite of
word-sized primitives, so a partition query (`Q_word`) cuts it into ordinary 16- and 32-bit units.

`TopPMaskWordx{V}_v1` is the same function as ONE primitive (a `Value<V>`, every split count, `splits` an operand): about 10^11
ANDs at V = 128256, a single word gate no query can cut. With `splits` a constant (the registered S) the pipeline
is fixed, and this module restates it at that S:

* `topp_keep_builder.keep_word` is the pipeline written once over vectors of words, each step an operation on whole vectors that an
  `Engine` applies (constants folded, the elements a constant operand decides wired, the rest instances);
* `topp_word_gates.GATES` is each operation's word gate;
* :class:`IREngine` runs the builder inside a composite body and emits each operation's instances as one batch of its word gate, so
  the Definition's gates are exactly the instances the circuit counts.

`TopPKeepWord_v1{V,S}` equals `topp_keep(x, p, S)` (and so `TopPMaskWordx{V}_v1` at `splits = S`) on every row: the test
evaluates the Definition with the circuit evaluator against `topp_split`. `TopPMask_v2{V,S}` applies it lane by lane with a select
(no `BitAt`), and `GumbelTopPTokenSelect_v2{V,S}` is `GumbelTopPTokenSelect_v1{V}` at the constant split count.
"""
from __future__ import annotations

import hashlib

import numpy as np

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite
from verity.ml.circuits.refs import Affine, Coll, Explicit, Strided, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.circuits.ops.scalar import BF16, BIT, BitAnd, F32, I32, I32Eq, SelectF32
from verity.ml.circuits.ops.prims import const
from verity.ml.definitions.registry.topp_keep_builder import NEG_INF, SPLIT_COUNTS, Engine, Vec, keep_word
from verity.ml.definitions.registry.topp_word_gates import CW, GATES, M32, SIG

I64 = Value(64)


# --- the keep word as a composite body ---------------------------------------------------------------------------------------

_TYPE = {1: BIT, 32: F32, CW: I32}


def _code(ref) -> int:
    space, idx, leaf = ref
    c = (idx << 32) | leaf
    return c if space == "n" else -1 - c


def _refs(codes: np.ndarray):
    """The IR references of ``codes`` (one per element), as affine / strided runs where they are."""
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    if n == 0:
        return Explicit(())
    neg = codes < 0
    key = np.where(neg, -1 - codes, codes)
    idx, leaf = key >> 32, key & M32
    same = (neg[1:] == neg[:-1]) & (idx[1:] == idx[:-1])
    st = leaf[1:] - leaf[:-1]
    # pair t (elements t, t+1) continues pair t - 1's run: the same array and the same step
    cont = np.zeros(max(n - 1, 0), bool)
    if n > 2:
        cont[1:] = same[1:] & same[:-1] & (st[1:] == st[:-1])
    # the first pair from t on that does not continue (n - 1: none)
    brk = np.where(~cont, np.arange(n - 1), n - 1)
    nxt = np.minimum.accumulate(brk[::-1])[::-1] if n > 1 else brk

    def ref(i):
        return ("p" if neg[i] else "n", int(idx[i]), int(leaf[i]))

    parts, i = [], 0
    while i < n:
        if i == n - 1 or not same[i]:
            parts.append(Explicit((ref(i),)))
            i += 1
            continue
        last = int(nxt[i + 1]) - 1 if i + 1 < n - 1 else i      # the run's last pair
        m = last - i + 2
        space, ix, base, step = ("p" if neg[i] else "n"), int(idx[i]), int(leaf[i]), int(st[i])
        parts.append(Affine(space, ix, base, m) if step == 1 else Strided(space, ix, base, 1, step, 0, m))
        i += m
    return concat(parts) if len(parts) > 1 else parts[0]


#: a call of at most this many elements is also value-numbered element by element
SMALL = 64


class _VCalls:
    """An expansion's calls on m elements at once (`IREngine.vcall`)."""

    def __init__(self, E, m):
        self.E, self.m = E, m

    def call(self, prim, *ops):
        return self.E.vcall(prim, list(ops), self.m)

    @staticmethod
    def c(v, w=32):
        return ("c", int(v), w)


class IREngine(Engine):
    """The engine inside a composite body: each operation's instances are one batch of its word gate, and a constant is a
    constant node."""

    def __init__(self, B):
        super().__init__(0)
        self.B = B
        self._consts: dict = {}
        #: (operation, constant operands, operand references) -> the gate that computes it: an operation the builder repeats on
        #: the same operands (a round's stop test) is computed once, so the partition finds no recomputed value
        self._made: dict = {}
        self._small: dict = {}

    def _const(self, w: int, v: int) -> Coll:
        key = (32 if w == CW else w, int(v))
        if key not in self._consts:
            self._consts[key] = self.B.call(const(*key))
        return self._consts[key]

    def _const_refs(self, kv, w):
        return np.array([_code(self._const(w, v).refs[0]) for v in np.asarray(kv).tolist()], np.int64)

    def param(self, coll: Coll, w: int) -> Vec:
        n = len(coll.refs)
        return Vec(w, np.zeros(n, bool), np.zeros(n, np.uint64), None, np.array([_code(r) for r in coll.refs], np.int64))

    def instances(self, name, key, args, sel, d, r):
        """Emit the instances ``sel`` of ``name``: its word gate as one batch, or its expansion as batches of word gates, each gate
        value-numbered (an earlier gate of the primitive on the same operands, a constant by its value, is reused), so no value is
        computed twice."""
        self.counts[(self.path, name, key)] += len(sel)
        ops = []
        for a, c, kd in zip(args, key, SIG[name][0]):
            w = 1 if kd == "b" else 32
            ops.append(("c", int(c), w) if c is not None else a.r[sel])
        fn, extra = GATES[name]
        ops += [("c", v, w) for v, w in extra]
        k = _VCalls(self, len(sel))
        out = k.call(fn, *ops) if isinstance(fn, PrimitiveDefinition) else fn(k, *ops)
        r[sel] = out if not isinstance(out, tuple) else self._const_refs([out[1]], out[2])[0]

    def vcall(self, prim: PrimitiveDefinition, ops: list, m: int):
        """``prim`` on m elements; an operand is ``("c", value, width)`` or an array of m references. Returns the m references, or a
        constant when every operand is one. The same call on the same operand arrays, and a repeated element within a call, reuse
        the gates made: a vector operation the builder repeats (a round's stop test), or two expansions share (the NaN test of
        numpy's max and min), is computed once."""
        if all(isinstance(o, tuple) for o in ops):
            v = int(prim.evaluate(*(o[1] for o in ops)))
            return ("c", v, prim.ret.w)
        cols = [np.full(m, -2 - o[1], np.int64) if isinstance(o, tuple) else np.asarray(o, np.int64) for o in ops]
        grid = np.ascontiguousarray(np.stack(cols, 1))
        key = (prim.id, hashlib.sha256(grid.tobytes()).digest())
        hit = self._made.get(key)
        if hit is not None:
            return hit.copy()
        uniq, first, inv = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        made = np.zeros(m, np.int64)
        old = np.zeros(m, bool)
        if m <= SMALL:                     # a scalar-sized call (a warp half's flags): elements an earlier small call made
            for t in first.tolist():
                hit = self._small.get((prim.id, tuple(grid[t].tolist())))
                if hit is not None:
                    made[t], old[t] = hit, True
        todo = np.sort(first[~old[first]])
        if not len(todo):
            out = made[first[inv]]
            self._made[key] = out
            return out.copy()
        colls, axes = [], []
        for o, (_n, t) in zip(ops, prim.params):
            if isinstance(o, tuple):
                colls.append(self._const(o[2], o[1]))
                axes.append(None)
            else:
                colls.append(Coll(Array(len(todo), t), _refs(np.asarray(o, np.int64)[todo])))
                axes.append(0)
        res = self.B.batch(prim, *colls, axes=tuple(axes))
        if isinstance(res.refs, Affine) and res.refs.space == "n":
            made[todo] = (np.int64(res.refs.idx) << 32) | (res.refs.base + np.arange(len(todo), dtype=np.int64))
        else:
            made[todo] = [_code(x) for x in res.refs]
        if m <= SMALL:
            for t in todo.tolist():
                self._small[(prim.id, tuple(grid[t].tolist()))] = made[t]
        out = made[first[inv]]
        self._made[key] = out
        return out.copy()

    def result(self, v: Vec, w: int) -> Coll:
        codes = v.r.copy()
        if v.k.any():
            codes[v.k] = self._const_refs(v.kv[v.k], w)
        return Coll(Array(len(v), _TYPE[w]), _refs(codes))


def keep_word_body(B, S: int, logits: Coll, top_p: Coll, parts_of=None) -> Coll:
    """The keep word's body at the split count S: `keep_word` in an :class:`IREngine` (``parts_of``: see
    `topp_keep_builder.pipeline`)."""
    E = IREngine(B)
    x, p = E.param(logits, 32), E.param(top_p, 32)
    return E.result(keep_word(E, x, p, E.const([S]), parts_of), 1)


@composite("TopPKeepWord", 1, ["V", "S"], lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32)), Array(st.V, BIT)),
           conformance="exact-model-tested")
def TopPKeepWord(B, st, logits, top_p):
    """`TopPMaskWordx{V}_v1`'s keep bits at the constant split count S, as word gates: the split top-p pipeline's operations
    (:func:`keep_word`), each a batch of its word gate over the lanes it touches. Equal to ``topp_keep(x, p, S)`` on every row."""
    assert int(st.S) in SPLIT_COUNTS, f"S = {st.S} is not a launched split count"
    return keep_word_body(B, int(st.S), logits, top_p)


@composite("TopPMask", 2, ["V", "S"],
           lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32), ("splits", I32)), Array(st.V, F32)),
           conformance="exact-model-tested")
def TopPMaskV2(B, st, logits, top_p, splits):
    """`TopPMask_v1{V}` at the split count S: each lane kept or set to -inf by `TopPKeepWord_v1{V,S}`'s bit, a select per lane, and
    no lane kept unless `splits == S`. Equal to `TopPMask_v1{V}` wherever `splits` is S or no launched split count (v1 masks every
    lane there too); a Program binds it only where `splits` is the constant S (a single-request workload)."""
    keep = B.call(bind(TopPKeepWord, V=st.V, S=st.S), logits, top_p)
    ok = B.call(I32Eq, splits, B.call(const(32, int(st.S))))
    keep = B.batch(BitAnd, keep, ok, axes=(0, None))
    return B.batch(SelectF32, keep, logits, B.call(const(32, NEG_INF)), axes=(0, 0, None))


@composite("GumbelTopPTokenSelect", 2, ["V", "S"],
           lambda st: ((("logits", Array(st.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectV2(B, st, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelect_v1{V}` with its keep word as word gates at the split count S (`TopPMask_v2{V,S}`), the same
    operands: the served token wherever `splits` is S, the constant a single-request workload's Program passes."""
    from verity.ml.definitions.registry import sampling as SM

    scaled = B.call(bind(SM.TemperatureScale, V=st.V), logits, temp)
    masked = B.call(bind(TopPMaskV2, V=st.V, S=st.S), scaled, top_p, splits)
    return B.call(bind(SM.GumbelSelectF32, V=st.V), masked, seed, pos, temp)


__all__ = ["IREngine", "keep_word_body", "TopPKeepWord", "TopPMaskV2", "GumbelTopPTokenSelectV2"]
