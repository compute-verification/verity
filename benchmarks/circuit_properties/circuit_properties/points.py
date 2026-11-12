"""The reading's distinct rows, tested at a point (`check`).

A charged product's operand that the reading takes as a weight with distinct rows (`Status.distinct`, and not
specific: `claims`) is claimed to have no two equal rows of its contraction's K
words, for generic weights. `check` evaluates the Program once at a point and refuses it (`work.RepeatedRows`) where
two rows of such an operand are equal there, naming the Definition, the node, the operand and the two rows. The point
is the claim's own worst case: whoever submits the circuit also picks the data. So every word of an input
(`Input<w>`) is one word, sampled generic, nonzero and, as a float, positive, repeated through every array it fills
(all tokens are one token), and every word of `θ` and of a registered input is sampled on its own: a finite float at its width (four
binades about 1, every sign and mantissa; `held` the bits that leaves it), or any nonzero word at a width no float
has. An input's word is positive because an input is data or a state the step carries in, which an honest step
reads under a square root, a logarithm or a divide: a positive word is inside each such domain, and one word repeated
says nothing about rows by its sign. `θ` keeps every sign, the generic part of the point. A NaN and equal rows are
refused, so the sign passes no claim; it decides only which honest circuits pass. The sample is seeded by SHA-512 of
the Program's digest, with no other randomness, so two reads agree.

Rows are equal only where they agree at `points` points, enough to hold 64 sampled bits a row: P = ceil(64 / (K b)),
b the bits its words hold at the point (`held`), at most those of the narrowest `θ` or registered input and of the
narrowest rounding the Program holds, so a narrow row (an e2m1 word after a rounding, a one-word row) is not refused
by chance. A row holding an f32 NaN is refused too: no comparison shows it distinct, and whoever submits the
circuit can make a NaN at the point.

The evaluation runs top-down along the reader's summaries (`Reader.summary`), a call's output at its caller's words,
so an identity written through a call is seen, and a view's words are its source's, so an argument passed twice or
an array beside its own view is one array. A batch's members are lanes; a scan runs the iterations the reader reads as
distinct contexts, the first of each, and a scan whose contexts repeat before its last iteration leaves its outputs
unevaluated: a claim that reads them is refused with the scan named. Inside a product, which the reader doesn't
summarize, every scan runs whole. The pass is about one evaluation of the step, where the reading is in the
description's size; one pass a Program, memoized by its digest.

Each primitive is computed by a numpy twin (`TWINS`), by the kernel `verity.circuits.evaluation.batch` registers,
or else by the reference a word at a time (`Tally.reference`), and a `DotRef` or an f32 `LinearRef` without a bias
by `dot`, its body's chain of multiply-adds on its operands as they are, so a batch's shared operand is never
repeated a member. Every twin is bit-exact for all words but the exponential's, which is held to the reference on the
twins test's words only. A lookup (`work.LOOKUPS`) reads its
index modulo its table's rows, so a sampled token picks a row: the point's rule, not the reference's (which gives a NaN
out of range).

What the point doesn't meet stays a premise (`work`'s docstring): rows equal only at values the sample never takes (W
plus z plus -1 times z, a one-word multiplier that is 0, inputs fitted to the weights, a circuit that tests its own
weights against a stated constant, rows equal only where an input's word is negative).
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.evaluation.batch import evaluate_batch, kernels
from verity.circuits.evaluation.reference import statics_of
from verity.circuits.program import is_input, is_registered
from verity.circuits.refs import Affine, Concat, Strided

from . import work as WK

#: the sampled bits a row must hold
ROW_BITS = 64
#: the bits a word of each rounding's format holds at the point: its sign, its mantissa and two of its exponent
ROUNDED = {"F32ToBf16Rn": 10, "F32ToF16Rn": 13, "F32ToE4m3Sat": 6, "F32ToE2m1Sat": 4}
F32_NAN, F32_NAN_PTX = 0x7FC00000, 0x7FFFFFFF


def held(w: int) -> int:
    """The bits a sampled word of width ``w`` holds: f32's, bf16's and e4m3's sign, mantissa and two exponent bits (four
    binades), every other width's own."""
    return {32: 26, 16: 10, 8: 6}.get(w, w)


def points(K: int, b: int) -> int:
    """P: the points at which rows of ``K`` words of ``b`` bits hold `ROW_BITS` sampled bits between them."""
    return max(1, math.ceil(ROW_BITS / (K * b)))


def sample(rng: np.random.Generator, w: int, shape, positive: bool = False) -> np.ndarray:
    """Words of width ``w``, each generic and nonzero: a finite float of four binades about 1 at 32, 16 and 8 bits
    (f32, bf16, e4m3), of every sign unless ``positive``, any nonzero word at every other width."""
    def bits(n: int) -> np.ndarray:
        return rng.integers(0, 1 << n, size=shape, dtype=np.uint64)

    floats = {32: (8, 23, 125), 16: (8, 7, 125), 8: (4, 3, 6)}
    if w in floats:
        e, m, lo = floats[w]
        exp = rng.integers(lo, lo + 4, size=shape, dtype=np.uint64)
        sign = bits(1) * np.uint64(not positive)
        return (sign << np.uint64(e + m)) | (exp << np.uint64(m)) | bits(m)
    return rng.integers(1, (1 << w) if w < 64 else (1 << 64) - 1, size=shape, dtype=np.uint64)


# ---- twins -----------------------------------------------------------------------------------------------------------

def _f(a: np.ndarray) -> np.ndarray:
    return a.astype(np.uint32).view(np.float32)


def _u(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)


def _nan(u: np.ndarray) -> np.ndarray:
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def _canon(u: np.ndarray, word: int = F32_NAN) -> np.ndarray:
    return np.where(_nan(u), np.uint64(word), u)


def _fma(a, b, c) -> np.ndarray:
    """RN(a·b + c), one rounding: the product is exact in float64, the sum's error exact (two-sum), and the sum
    rounded to odd in float64 rounds to f32 as the exact sum would; a NaN operand or an invalid one gives `F32_NAN`."""
    x, y, z = (_f(v).astype(np.float64) for v in (a, b, c))
    p = x * y
    s = p + z
    bp = s - z
    err = (p - bp) + (z - (s - bp))
    ok = np.isfinite(s) & (err != 0)
    odd = (s.view(np.uint64) & np.uint64(1)).astype(bool)
    s = np.where(ok & ~odd, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    out = _u(s.astype(np.float32))
    nan = _nan(a) | _nan(b) | _nan(c) | np.isnan(p + z)
    return np.where(nan, np.uint64(F32_NAN), out)


def dot(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`DotRef_v1`'s words, its body's chain of `F32Fma_v1` from +0 ascending, on ``x`` and ``w`` (..., K)
    broadcast: (...)."""
    acc = np.zeros(np.broadcast_shapes(x.shape[:-1], w.shape[:-1]), np.uint64)
    for k in range(x.shape[-1]):
        acc = _fma(x[..., k], w[..., k], acc)
    return acc


def _exp(a) -> np.ndarray:
    v = _f(a).astype(np.float64)
    out = _u(np.exp(v).astype(np.float32))
    out = np.where(v > 89, np.uint64(0x7F800000), np.where(v < -105, np.uint64(0), out))
    return _canon(np.where(_nan(a), np.uint64(F32_NAN), out))


def _sqrt(a) -> np.ndarray:
    return np.where(a == 0x80000000, a, _canon(_u(np.sqrt(_f(a)))))


def _max(a, b) -> np.ndarray:
    return np.where(_nan(a) | _nan(b), np.uint64(F32_NAN), np.where(_f(a) >= _f(b), a, b))


def _i64_to_i32(t) -> np.ndarray:
    v = t.astype(np.int64)
    return np.where((v >= -(1 << 31)) & (v < (1 << 31)), t & np.uint64(0xFFFFFFFF), np.uint64(0x80000000))


def _binop(op, nan: int | None = None, first: int = 0):
    """``op`` on float32 words; a NaN result ``nan`` where given, else the host's, as the reference's numpy float32
    scalars give it on x86-64: a NaN operand quieted, operand ``first``'s where both are, and an invalid operation's
    NaN."""
    def f(a, b):
        out = _u(op(_f(a), _f(b)))
        if nan is not None:
            return _canon(out, nan)
        quiet = np.uint64(0x400000)
        p, q = (a, b) if first == 0 else (b, a)
        return np.where(_nan(p), p | quiet, np.where(_nan(q), q | quiet, out))
    return f


def _v2(fn):
    return lambda *a: _canon(fn(*a), F32_NAN_PTX)


_add, _mul, _div = (_binop(np.add, first=1), _binop(np.multiply, first=1), _binop(np.divide))
#: numpy twins of the reference's word primitives, by id: each a function of its arguments' words (uint64 arrays of
#: one shape), bit-exact for all words but `F32ExpRn_v1`'s
TWINS = {
    "F32Add_v1": _add, "F32Mul_v1": _mul, "F32Div_v1": _div, "F32Fma_v1": _fma,
    "F32Add_v2": _v2(_add), "F32Mul_v2": _v2(_mul), "F32Div_v2": _v2(_div), "F32Fma_v2": _v2(_fma),
    "F32Sub_v1": _binop(np.subtract, F32_NAN), "F32Neg_v1": lambda a: a ^ np.uint64(0x80000000),
    "F32Sqrt_v1": _sqrt, "F32ExpRn_v1": _exp, "F32MaxRef_v1": _max,
    "F32Eq_v1": lambda a, b: ((_f(a) == _f(b)) & ~_nan(a) & ~_nan(b)).astype(np.uint64),
    "I32Eq_v1": lambda a, b: ((a & 0xFFFFFFFF) == (b & 0xFFFFFFFF)).astype(np.uint64),
    "I32Add_v1": lambda a, b: (a + b) & np.uint64(0xFFFFFFFF),
    "BitAnd_v1": lambda a, b: a & b & np.uint64(1),
    "SelectF32_v1": lambda c, a, b: np.where((c & np.uint64(1)).astype(bool), a, b),
    "I64ToI32_v1": _i64_to_i32,
}


# ---- the walk --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Unrun:
    """Values a scan leaves unevaluated: its Definition, the node and the iterations it runs of its length."""

    fn: str
    node: int
    ran: int
    n: int


@dataclass
class Tally:
    """What a pass evaluated: primitive words at one point, the points, each primitive's words the reference
    computed, the claims tested, and the claims that read values a scan left unevaluated."""

    words: int = 0
    points: int = 1
    reference: Counter = field(default_factory=Counter)
    claims: int = 0
    unrun: list = field(default_factory=list)


def claimed(st) -> bool:
    """Whether a charged operand of status ``st`` is charged for its distinct rows alone: `Status.distinct`, and not
    specific, whose charge the specific rule carries. A charged operand carries no gather's rows and, unless it is
    specific, reads no current input (`work.may_repeat`), so those need no test here."""
    return st.distinct and not st.specific


def claims(R, root, reasons) -> dict:
    """The charged products' operands whose rows the reading shows distinct and nothing else does (`claimed`), by
    summary and node: (argument, its parameter's name, K)."""
    out: dict = {}
    order = sorted(R.memo.values(), key=lambda s: -s.sid)
    count = {root.sid: 1}
    for s in order:
        for c, n, _b in s.children:
            count[c.sid] = count.get(c.sid, 0) + count.get(s.sid, 0) * n
    bodies = _bodies(root, R)
    for s in order:
        for i, site in enumerate(s.sites):
            if reasons.get((s.sid, i)) != "" or not count.get(s.sid, 0):
                continue
            nd = bodies[s.sid].body.nodes[site.node]
            names = [n for n, _t in nd.fn.signature()[0]]
            for st, name in ((site.x, site.product.x), (site.w, site.product.w)):
                if claimed(st):
                    out.setdefault(s.sid, {}).setdefault(site.node, []).append((names.index(name), name, site.k_dim))
    return out


def _bodies(root, R) -> dict:
    """Each summary's Definition, by its id: the root's, and each child's, its node's callee."""
    out = {root.sid: R.program.root}
    todo = [root]
    while todo:
        s = todo.pop()
        fn = out[s.sid]
        for (c, _n, _b), (node, _m) in zip(s.children, s.at, strict=True):
            if c.sid not in out:
                out[c.sid] = fn.body.nodes[node].fn
                todo.append(c)
    return out


_MEMO: dict = {}


def check(program, R, root, reasons) -> Tally:
    """Refuse ``program`` (`work.RepeatedRows`) where two rows of an operand of `claims` are equal at the point or one
    holds a NaN; one pass a Program, memoized by its digest."""
    cl = claims(R, root, reasons)
    if not cl:
        return Tally(points=0)
    from verity.circuits.codec import program_digest
    digest = program_digest(program)
    fns = _bodies(root, R)
    key = (digest, frozenset((fns[sid].id, node, i, K) for sid, nodes in cl.items() for node, cs in nodes.items()
                             for i, _n, K in cs))
    hit = _MEMO.get(key)
    if hit is None:
        try:
            hit = Walk(program, R, root, cl, digest).run()
        except WK.RepeatedRows as e:
            hit = e
        _MEMO[key] = hit
    if isinstance(hit, Exception):
        raise hit
    return hit


class Walk:
    def __init__(self, program, R, root, cl: dict, digest: str) -> None:
        self.program, self.R, self.root, self.claims = program, R, root, cl
        self.tally = Tally()
        self.seed = int.from_bytes(hashlib.sha512(b"circuit_properties/points\0" + bytes.fromhex(digest)).digest(), "big")
        self.fns = _bodies(root, R)
        theta = set(program.root.theta)
        b = min((held(_leaf_width(nd)) for k, nd in enumerate(program.root.body.nodes)
                 if is_input(nd.fn) and (is_registered(nd.fn) or k in theta)), default=64)
        b = min([b, *(ROUNDED[n] for n in _roundings(program.root))])
        self.P = max(points(K, min(b, held(_operand_width(self.fns[sid].body.nodes[node], i))))
                     for sid, nodes in cl.items() for node, cs in nodes.items() for i, _n, K in cs)
        self.tally.points = self.P
        self.last: dict = {}
        self.index: dict = {}

    def inputs(self) -> dict:
        """The point: each input node's words at each point, (points, leaves), by ("n", node)."""
        env: dict = {}
        rng = np.random.default_rng(self.seed)
        theta = set(self.program.root.theta)
        for k, nd in enumerate(self.program.root.body.nodes):
            if is_input(nd.fn):
                w = _leaf_width(nd)
                if is_registered(nd.fn) or k in theta:
                    env["n", k] = sample(rng, w, (self.P, nd.out.leaves))
                else:
                    env["n", k] = np.repeat(sample(rng, w, (self.P, 1), positive=True), nd.out.leaves, axis=1)
        return env

    def run(self) -> Tally:
        self.scans: list[Unrun] = []
        self.body(self.program.root, self.root, [], {k: (v, None) for k, v in self.inputs().items()}, self.P)
        if self.tally.unrun:
            sid, node, name, u = self.tally.unrun[0]
            raise WK.RepeatedRows(self.fns[sid].id, node, name, None, scan=u)
        return self.tally

    # -- references --------------------------------------------------------------------------------------------------
    # A value is (words, gone): words (lanes, leaves) as uint64, gone None or beside each word 0 where it is evaluated,
    # else 1 + the index of the scan (`Walk.scans`) that leaves it unevaluated.

    def take(self, refs, env: dict, L: int):
        """The value ``refs`` read, (lanes, count)."""
        if isinstance(refs, (Affine, Strided)):
            v, g = env[refs.space, refs.idx]
            at = slice(refs.base, refs.base + refs.count) if isinstance(refs, Affine) else self._strided(refs)
            return v[:, at], None if g is None else g[:, at]
        if isinstance(refs, Concat):
            parts = [self.take(p, env, L) for p in refs.parts if p.count]
            if not parts:
                return np.zeros((L, 0), np.uint64), None
            gone = None if all(g is None for _v, g in parts) else np.concatenate(
                [np.zeros(v.shape, np.int32) if g is None else g for v, g in parts], axis=1)
            return np.concatenate([v for v, _g in parts], axis=1), gone
        out, gone = np.zeros((L, refs.count), np.uint64), None
        for j, (space, idx, leaf) in enumerate(refs):
            v, g = env[space, idx]
            out[:, j] = v[:, leaf]
            if g is not None:
                gone = np.zeros(out.shape, np.int32) if gone is None else gone
                gone[:, j] = g[:, leaf]
        return out, gone

    def _strided(self, r) -> np.ndarray:
        key = (r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
        if key not in self.index:
            j = np.arange(r.count)
            self.index[key] = r.base + (j // r.inner) * r.outer_stride + (j % r.inner) * r.inner_stride
        return self.index[key]

    def _last(self, fn) -> dict:
        """Each node's last reader in ``fn``'s body (None: the body returns it)."""
        if fn.id not in self.last:
            last: dict = {}
            for k, nd in enumerate(fn.body.nodes):
                for a in nd.args:
                    for s in _sources(a.refs):
                        if s[0] == "n":
                            last[s[1]] = k
            for s in _sources(fn.body.ret):
                if s[0] == "n":
                    last[s[1]] = None
            self.last[fn.id] = last
        return self.last[fn.id]

    # -- bodies ------------------------------------------------------------------------------------------------------

    def body(self, fn, S, params: list, env: dict | None, L: int):
        """``fn``'s returned words at ``params`` (each (lanes, leaves)), ``S`` its summary or None inside a product."""
        env = {} if env is None else env
        for i, p in enumerate(params):
            env["p", i] = p
        nodes = fn.body.nodes
        kids: dict = {}
        if S is not None:
            for (c, _n, _b), (node, members) in zip(S.children, S.at, strict=True):
                kids.setdefault(node, []).append((c, members))
        mine = self.claims.get(S.sid, {}) if S is not None else {}
        last = self._last(fn)
        for k, nd in enumerate(nodes):
            if ("n", k) in env:
                continue
            args = [self.take(a.refs, env, L) for a in nd.args]
            f = nd.fn
            if isinstance(f, PrimitiveDefinition):
                env["n", k] = self.primitive(f, nd, args, L)
            elif _rows_of(f) is not None:
                env["n", k] = self.product(f, nd, args, L)
            elif nd.form == "call":
                env["n", k] = self.body(f, kids[k][0][0] if k in kids else None, args, None, L)
            elif nd.form == "batch":
                env["n", k] = self.batch(f, nd, args, kids[k][0][0] if k in kids else None, L)
            else:
                env["n", k] = self.scan(fn, k, nd, args, kids.get(k), L)
            for i, name, K in mine.get(k, ()):
                self.rows(S, k, nd, i, name, K, args[i], L)
            for a in nd.args:
                for s in _sources(a.refs):
                    if s[0] == "n" and last.get(s[1], -1) == k:
                        env.pop(s, None)
        return self.take(fn.body.ret, env, L)

    def primitive(self, f, nd, args: list, L: int):
        n = nd.n if nd.form == "batch" else 1
        self.tally.words += nd.out.leaves * (L // self.P)
        if not nd.args:
            return np.full((L, nd.out.leaves), WK._const(f), np.uint64), None
        params = [t for _n, t in f.signature()[0]]
        mapped = [n if nd.form == "batch" and nd.axes[i] is not None else 1 for i in range(len(params))]
        shaped = [v.reshape(L, m, t.leaves) for (v, _g), m, t in zip(args, mapped, params)]
        gone = None
        if any(g is not None for _v, g in args):
            gone = np.zeros((L, n), np.int32)
            for (_v, g), m, t in zip(args, mapped, params):
                if g is not None:
                    gone = np.maximum(gone, g.reshape(L, m, t.leaves).max(axis=2))
            gone = np.repeat(gone, nd.out.leaves // n, axis=1)
        lk = self.R.lookup(f)
        if lk is not None:
            names = [m for m, _t in f.params]
            tok, col = shaped[names.index(lk.index)], shaped[names.index(lk.table)]
            idx = (tok[..., 0] % np.uint64(col.shape[-1])).astype(np.int64)
            col = np.broadcast_to(col, (L, max(n, col.shape[1]), col.shape[-1]))
            idx = np.broadcast_to(idx, col.shape[:2])
            return np.take_along_axis(col, idx[..., None], axis=2).reshape(L, -1), gone
        twin = TWINS.get(f.id)
        if twin is not None and all(t.leaves == 1 for t in params):
            out = twin(*np.broadcast_arrays(*(s[..., 0] for s in shaped)))
            return np.broadcast_to(out, (L, n)).reshape(L, n).astype(np.uint64), gone
        if "numpy" not in kernels(f):
            self.tally.reference[f.id] += nd.out.leaves * (L // self.P)
        flat = [np.broadcast_to(s, (L, n, s.shape[-1])).reshape(L * n, s.shape[-1]) for s in shaped]
        try:
            out = evaluate_batch(f, *flat)
        except (ValueError, ArithmeticError, IndexError) as e:
            raise ValueError(f"the point's evaluation of {f.id}: {e}") from e
        return np.asarray(out, dtype=np.uint64).reshape(L, -1), gone

    def product(self, f, nd, args: list, L: int):
        """A `DotRef_v1` or an f32 `LinearRef_v1` without a bias (`_rows_of`), call or batch, by `dot` on its operands
        as they are, never a batch's shared operand repeated a member; its words are its body's."""
        n = nd.n if nd.form == "batch" else 1
        rows, K = _rows_of(f), f.signature()[0][0][1].leaves
        self.tally.words += self.flat(f) * n * (L // self.P)
        m = [n if nd.form == "batch" and nd.axes[i] is not None else 1 for i in (0, 1)]
        x, w = args[0][0].reshape(L, m[0], 1, K), args[1][0].reshape(L, m[1], rows, K)
        out = dot(x, w).reshape(L, n * rows)
        gone = None
        if any(g is not None for _v, g in args):
            gone = np.zeros((L, n), np.int32)
            for (_v, g), mi in zip(args, m):
                if g is not None:
                    gone = np.maximum(gone, g.reshape(L, mi, -1).max(axis=2))
            gone = np.repeat(gone, rows, axis=1)
        return out, gone

    def flat(self, fn) -> int:
        """The primitive words one activation of ``fn`` computes, through its calls, batches and scans."""
        key = ("flat", fn.id)
        if key not in self.last:
            self.last[key] = sum(nd.out.leaves if isinstance(nd.fn, PrimitiveDefinition) else
                                 (nd.n if nd.form in ("batch", "scan") else 1) * self.flat(nd.fn)
                                 for nd in fn.body.nodes if not is_input(nd.fn))
        return self.last[key]

    def batch(self, f, nd, args: list, S, L: int):
        n = nd.n
        params = [t for _n, t in f.signature()[0]]

        def spread(a, ax, t):
            return a.reshape(L * n, t.leaves) if ax is not None else np.repeat(a, n, axis=0)

        inner = [(spread(v, ax, t), None if g is None else spread(g, ax, t))
                 for (v, g), ax, t in zip(args, nd.axes, params)]
        v, g = self.body(f, S, inner, None, L * n)
        return v.reshape(L, -1), None if g is None else g.reshape(L, -1)

    def scan(self, fn, k: int, nd, args: list, kids, L: int):
        """Scan node ``k`` of ``fn``: its carry and each iteration's output, the iterations the reader reads as
        distinct contexts where ``kids`` (its summaries, in order) is given, each the first of its context, else all
        of them."""
        f, n, nx = nd.fn, nd.n, nd.n_xs
        params, ret = f.signature()
        cl = params[0][1].leaves
        runs = [c for c, _m in kids] if kids else [None] * n

        def cut(a, lo: int, hi: int):
            v, g = a
            return v[:, lo:hi], None if g is None else g[:, lo:hi]

        carry, ys = args[0], []
        for j, C in enumerate(runs):
            xs = [cut(args[1 + i], j * params[1 + i][1].leaves, (j + 1) * params[1 + i][1].leaves) for i in range(nx)]
            out = self.body(f, C, [carry, *xs, *args[1 + nx:]], None, L)
            carry = cut(out, 0, cl)
            ys.append(cut(out, cl, out[0].shape[1]))
        if len(runs) < n:
            self.scans.append(Unrun(fn.id, k, len(runs), n))
            leaves = nd.out.leaves
            return np.zeros((L, leaves), np.uint64), np.full((L, leaves), len(self.scans), np.int32)
        parts = [carry, *ys]
        gone = None if all(g is None for _v, g in parts) else np.concatenate(
            [np.zeros(v.shape, np.int32) if g is None else g for v, g in parts], axis=1)
        return np.concatenate([v for v, _g in parts], axis=1), gone

    # -- the check ---------------------------------------------------------------------------------------------------

    def rows(self, S, k: int, nd, i: int, name: str, K: int, value, L: int) -> None:
        """Refuse where two rows of ``K`` words of argument ``i`` of product node ``k`` are equal at every point, in
        one activation, or where a row holds an f32 NaN, which no comparison can show distinct."""
        words, gone = value
        if gone is not None and gone.any():
            self.tally.unrun.append((S.sid, k, name, self.scans[int(gone[gone > 0][0]) - 1]))
            return
        self.tally.claims += 1
        W = words.shape[1]
        if not K or W % K or W // K < 2:
            return
        P, A, R = self.P, L // self.P, W // K
        v = words.reshape(P, A, R, K).transpose(1, 2, 0, 3).reshape(A * R, P * K)
        if _operand_width(nd, i) == 32:
            bad = _nan(v).any(axis=1).reshape(A, R)
            if bad.any():
                a = int(np.flatnonzero(bad.any(axis=1))[0])
                raise WK.RepeatedRows(self.fns[S.sid].id, k, name, None, member=a if A > 1 else None,
                                      nan=(int(bad[a].sum()), R))
        keys = np.concatenate([np.repeat(np.arange(A, dtype=np.uint64), R)[:, None], v], axis=1)
        _keys, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        dup = np.flatnonzero(cnt > 1)
        if dup.size:
            r1, r2 = np.flatnonzero(inv.reshape(-1) == dup[0])[:2]
            a, (q1, q2) = int(r1 // R), (int(r1 % R), int(r2 % R))
            raise WK.RepeatedRows(self.fns[S.sid].id, k, name, (q1, q2), member=a if A > 1 else None)


def _rows_of(fn) -> int | None:
    """The rows of ``w`` a product `dot` computes reads: 1 for a `DotRef_v1`, N for an f32 `LinearRef_v1` without a
    bias; None for any other Definition."""
    defn = getattr(fn, "defn", None)
    if defn is None or defn.version != 1:
        return None
    st = statics_of(fn)
    if defn.name == "DotRef":
        return 1
    if defn.name == "LinearRef" and st.get("DT") == "f32" and st.get("BIAS") is False:
        return st["N"]
    return None


def _sources(refs) -> set:
    if isinstance(refs, (Affine, Strided)):
        return {(refs.space, refs.idx)} if refs.count else set()
    if isinstance(refs, Concat):
        return set().union(*(_sources(p) for p in refs.parts))
    return {(s, i) for s, i, _l in refs}


def _leaf_width(nd) -> int:
    return max(nd.out.width_set(), default=1)


def _operand_width(nd, i: int) -> int:
    return max(nd.fn.signature()[0][i][1].width_set(), default=1)


def _roundings(root) -> set:
    """The `ROUNDED` primitives' names in ``root``'s Definitions."""
    seen, out, todo = set(), set(), [root]
    while todo:
        fn = todo.pop()
        if fn.id in seen or isinstance(fn, PrimitiveDefinition):
            continue
        seen.add(fn.id)
        for nd in fn.body.nodes:
            if isinstance(nd.fn, PrimitiveDefinition):
                if nd.fn.name in ROUNDED:
                    out.add(nd.fn.name)
            else:
                todo.append(nd.fn)
    return out
