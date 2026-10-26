"""The split top-p pipeline over vectors of words, written once: :func:`keep_word`, the stats kernel, five step / combine rounds of
F = 8 pivots and the mask kernel as `topp_split` runs them, each step an operation of `topp_word_gates.SIG` on whole vectors
(:class:`Vec`).

An :class:`Engine` applies each operation: an element whose operands are constants is folded, one an operand's constant decides is
wired (the operation's word gate is a select, an AND, an OR or an XOR), and the rest are instances.  `verity_flock.topp_word` runs
the builder with the operations' `gf2` circuits (counting and evaluating them); `topp_words.IREngine` runs it inside a composite body.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import numpy as np

from verity.ml.circuits.ops.scalar import BitAnd, BitOr, SelectF32, SelectI32
from verity.ml.definitions.registry.topp_geometry import spt_for, z_group
from verity.ml.definitions.registry.topp_word_gates import CW, GATES, SIG, WIDTH, BitXor, SelectBit, U32Xor, f32_word, numeric

SPLIT_COUNTS = (1, 2, 4, 8, 16, 32)
F, R = 8, 5                      # topp_split.SPLIT_FANOUT, SPLIT_ROUNDS
STATS_BLOCK, STEP_BLOCK, MASK_BLOCK = 8192, 2048, 8192
NUM_WARPS = 8
NEG_INF, INF, ONE_F, NEG_ONE = 0xFF800000, 0x7F800000, 0x3F800000, 0xBF800000
NINE, TINY30 = f32_word(F + 1.0), f32_word(1e-30)
C6, C7 = f32_word(1.0 + 1e-6), f32_word(1.0 + 1e-7)


# --- vectors of words and the engine -----------------------------------------------------------------------------------------

class Vec:
    """n words of width w: element i is the constant kv[i] where k[i]; d (T, n) holds T test vectors' values (None when not
    evaluating); r (n,) each element's IR reference (None outside an IR body)."""

    __slots__ = ("d", "k", "kv", "w", "r")

    def __init__(self, w, k, kv, d, r=None):
        self.w, self.k, self.kv, self.d, self.r = w, k, kv, d, r

    def __len__(self):
        return len(self.k)

    def __getitem__(self, i):
        if not isinstance(i, slice):
            i = np.asarray(i, dtype=np.int64)
        return Vec(self.w, self.k[i], self.kv[i], None if self.d is None else self.d[:, i], None if self.r is None else self.r[i])


def cat(vs) -> Vec:
    vs = list(vs)
    d = None if vs[0].d is None else np.concatenate([v.d for v in vs], 1)
    r = None if vs[0].r is None else np.concatenate([v.r for v in vs])
    return Vec(vs[0].w, np.concatenate([v.k for v in vs]), np.concatenate([v.kv for v in vs]), d, r)


def put(v: Vec, idx, x: Vec) -> Vec:
    """v with elements idx replaced by x."""
    k, kv = v.k.copy(), v.kv.copy()
    k[idx], kv[idx] = x.k, x.kv
    d = r = None
    if v.d is not None:
        d = v.d.copy()
        d[:, idx] = x.d
    if v.r is not None:
        r = v.r.copy()
        r[idx] = x.r
    return Vec(v.w, k, kv, d, r)


#: distinct constant-operand variants one call lowers; the rest of the call takes the generic piece
VARIANTS = 16


def _folds(name, a):
    """(mask, source) pairs: elements an operand's known value decides, the source a Vec (wiring) or an int (a constant).  Read off
    the operation's word gate (`GATES`) on its operands and the constants it appends: a select's condition picks an operand, AND's
    0 and OR's 1 absorb, and AND's 1, OR's 0 and XOR's 0 pass the other operand."""
    fn, extra = GATES[name]
    a = [*a, *(v for v, _w in extra)]
    one = lambda v: v.k & (v.kv == 1)
    zero = lambda v: v.k & (v.kv == 0)
    if fn in (SelectF32, SelectI32, SelectBit):
        return [(one(a[0]), a[1]), (zero(a[0]), a[2])]
    if fn is BitAnd:
        return [(zero(a[0]), 0), (zero(a[1]), 0), (one(a[0]), a[1]), (one(a[1]), a[0])]
    if fn is BitOr:
        return [(one(a[0]), 1), (one(a[1]), 1), (zero(a[0]), a[1]), (zero(a[1]), a[0])]
    if fn in (BitXor, U32Xor):
        return [(zero(a[0]), a[1]), (zero(a[1]), a[0])]
    return []


class Engine:
    """Applies operations to vectors: an element whose operands are all constants is folded, one an operand's constant decides
    is wired, and the rest are instances of the operation's variant for their constant operands, counted in ``counts`` (key:
    the stage path, the operation, the constant operands) and handed to :meth:`instances`. :meth:`_run` computes an operation on
    arrays (the known elements; with T > 0, the instances): here on ints by its word gate, in `verity_flock.topp_word` by its
    circuit."""

    def __init__(self, T: int = 0):
        self.T = T
        self.counts: Counter = Counter()
        self.path: tuple = ()

    @contextmanager
    def at(self, label: str):
        old = self.path
        self.path = old + (label,)
        try:
            yield
        finally:
            self.path = old

    def const(self, vals, w: int = 32) -> Vec:
        kv = np.array([int(v) & ((1 << w) - 1) for v in np.atleast_1d(vals).tolist()], dtype=np.uint64)
        d = np.broadcast_to(kv, (self.T, len(kv))).copy() if self.T else None
        return Vec(w, np.ones(len(kv), bool), kv, d, self._const_refs(kv, w))

    def _const_refs(self, kv, w):
        return None

    def free(self, n: int, w: int = 32, data=None) -> Vec:
        """n unknown words (with T > 0, ``data`` (T, n) their test values)."""
        d = None
        if self.T:
            d = np.asarray(data, dtype=np.uint64).reshape(self.T, n)
        return Vec(w, np.zeros(n, bool), np.zeros(n, np.uint64), d)

    def _run(self, name, key, ins):
        """The operation's results on ``ins`` (per free operand a (T', m) array), constant operands from ``key``."""
        f = numeric(name)
        kinds = SIG[name][0]
        out_w = WIDTH[SIG[name][1]]
        free = iter(ins)
        cols = [next(free) if c is None else None for c in key]
        shape = next(x for x in cols if x is not None).shape if any(x is not None for x in cols) else (1, 1)
        size = int(np.prod(shape))
        grid = np.stack([np.full(size, int(key[j]), np.uint64) if cols[j] is None else cols[j].reshape(-1).astype(np.uint64)
                         for j in range(len(kinds))], 1)
        uniq, inv = np.unique(grid, axis=0, return_inverse=True)
        vals = np.array([f(*(int(v) for v in row)) & ((1 << out_w) - 1) for row in uniq], np.uint64)
        return vals[inv.reshape(-1)].reshape(shape)

    def instances(self, name, key, args, sel, d, r):
        self.counts[(self.path, name, key)] += len(sel)
        if d is not None:
            d[:, sel] = self._run(name, key, [a.d[:, sel] for a, c in zip(args, key) if c is None])

    def op(self, name: str, *args) -> Vec:
        kinds, out = SIG[name]
        args = [a if isinstance(a, Vec) else self.const([a], WIDTH[k]) for a, k in zip(args, kinds)]
        n = max(len(a) for a in args)
        args = [a if len(a) == n else a[np.zeros(n, np.int64)] for a in args]
        w = WIDTH[out]
        k, kv = np.zeros(n, bool), np.zeros(n, np.uint64)
        d = np.zeros((self.T, n), np.uint64) if self.T else None
        r = np.zeros(n, np.int64) if args[0].r is not None else None
        todo = np.ones(n, bool)
        for mask, src in _folds(name, args):
            sel = mask & todo
            if not sel.any():
                continue
            if isinstance(src, Vec):
                k[sel], kv[sel] = src.k[sel], src.kv[sel]
                if d is not None:
                    d[:, sel] = src.d[:, sel]
                if r is not None:
                    r[sel] = src.r[sel]
            else:
                k[sel], kv[sel] = True, src
                if d is not None:
                    d[:, sel] = src
            todo &= ~sel
        known = todo & np.logical_and.reduce([a.k for a in args])
        if known.any():
            idx = np.flatnonzero(known)
            vals = self._run(name, (None,) * len(args), [a.kv[idx][None, :] for a in args])[0]
            k[idx], kv[idx] = True, vals
            if d is not None:
                d[:, idx] = vals
            todo &= ~known
        rest = np.flatnonzero(todo)
        if len(rest):
            if not any(a.k[rest].any() for a in args):
                groups = [((None,) * len(args), rest)]
            else:
                K = np.stack([np.where(a.k[rest], a.kv[rest].astype(np.int64), -1) for a in args], 1)
                keys, inv = np.unique(K, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                if len(keys) > VARIANTS:
                    groups = [((None,) * len(args), rest)]
                else:
                    groups = [(tuple(None if c < 0 else int(c) for c in key), rest[inv == j]) for j, key in enumerate(keys)]
            for key, sel in groups:
                self.instances(name, key, args, sel, d, r)
        v = Vec(w, k, kv, d, r)
        if r is not None and k.any():
            v.r[k] = self._const_refs(kv[k], w)
        return v


# --- the pipeline ------------------------------------------------------------------------------------------------------------

def reduce(E: Engine, name: str, v: Vec, seg) -> Vec:
    """Each consecutive segment (lengths ``seg``) of v reduced by the associative piece ``name`` (any pairing)."""
    lens = np.asarray(seg, dtype=np.int64)
    assert (lens > 0).all()
    offs = np.cumsum(lens) - lens
    while (lens > 1).any():
        pairs, odd = lens // 2, lens % 2
        seg_rep = np.repeat(np.arange(len(lens)), pairs)
        within = np.arange(pairs.sum()) - np.repeat(np.cumsum(pairs) - pairs, pairs)
        a = offs[seg_rep] + 2 * within
        r = E.op(name, v[a], v[a + 1])
        oddi = np.flatnonzero(odd)
        new_lens = pairs + odd
        new_offs = np.cumsum(new_lens) - new_lens
        src = np.empty(new_lens.sum(), np.int64)
        src[new_offs[seg_rep] + within] = np.arange(len(a))
        src[new_offs[oddi] + pairs[oddi]] = len(a) + np.arange(len(oddi))
        v = cat([r, v[offs[oddi] + lens[oddi] - 1]])[src]
        lens, offs = new_lens, new_offs
    return v[offs]


def bfly(E: Engine, v: Vec, groups: int, lanes: int) -> Vec:
    """Lane 0 of an xor-butterfly sum (masks lanes/2 .. 1) of each group: at mask m, lane t < m is lane t plus lane t + m."""
    cur, m = lanes, lanes // 2
    while m >= 1:
        g = np.arange(groups * cur).reshape(groups, cur)
        v = E.op("add", v[g[:, :m].ravel()], v[g[:, m:2 * m].ravel()])
        cur, m = m, m // 2
    return v


def tree1d(E: Engine, v: Vec, nb: int, B: int, spt: int) -> Vec:
    """``topp_split.tree_sum_1d`` of each of nb consecutive [B] blocks: per thread sequential over its (rep, j) registers,
    warp butterfly, cross-warp butterfly."""
    reps = B // (32 * NUM_WARPS) // spt
    g = np.arange(nb * B).reshape(nb, reps, NUM_WARPS, 32, spt)
    acc = v[g[:, 0, :, :, 0].ravel()]
    for rep in range(reps):
        for j in range(spt):
            if rep or j:
                acc = E.op("add", acc, v[g[:, rep, :, :, j].ravel()])
    return bfly(E, bfly(E, acc, nb * NUM_WARPS, 32), nb, NUM_WARPS)


def pad(E: Engine, v: Vec, seg, B: int, fill: int) -> Vec:
    """The consecutive segments of v (lengths ``seg``) as [B] blocks, the lanes past each segment the constant ``fill``."""
    seg = list(seg)
    if all(n == B for n in seg):
        return v
    allv = cat([v, E.const([fill] * (len(seg) * B - len(v)), v.w)])
    idx = np.empty(len(seg) * B, np.int64)
    o, po = 0, len(v)
    for b, n in enumerate(seg):
        idx[b * B:b * B + n] = np.arange(o, o + n)
        idx[b * B + n:(b + 1) * B] = np.arange(po, po + B - n)
        o, po = o + n, po + B - n
    return allv[idx]


def scan(E: Engine, v: Vec, seg) -> Vec:
    """Inclusive prefix sums (count adds) within each consecutive segment."""
    pos = np.concatenate([np.arange(n) for n in seg])
    d = 1
    while d < max(seg):
        i = np.flatnonzero(pos >= d)
        v = put(v, i, E.op("iadd", v[i], v[i - d]))
        d *= 2
    return v


def last_true(E: Engine, flags: Vec) -> Vec:
    """One-hot of the last set flag (none set: all zero)."""
    n = len(flags)
    after = [E.const([0], 1)]
    for f in range(n - 1, 0, -1):
        after.append(E.op("or", flags[[f]], after[-1]))
    return E.op("and", flags, E.op("not", cat(after[::-1])))


def pick(E: Engine, oh: Vec, vals: Vec, groups: int, kind: str = "f") -> Vec:
    """Per group of len(oh) values, the one oh selects (0 when none)."""
    n = len(oh)
    g = E.op("gate" if kind == "f" else "gatei", oh[np.tile(np.arange(n), groups)], vals)
    return reduce(E, "xorf" if kind == "f" else "xori", g, [n] * groups)


# --- the pipeline ------------------------------------------------------------------------------------------------------------

def _blocks(bounds, B):
    return [[(i, min(B, e - i)) for i in range(s0, e, B)] for s0, e in bounds]


def stats_kernel(E, x, S, spt, bounds):
    """Per slice: the running max m, the rescaled exp sum, the minimum of the lanes > -inf; and whether any lane is > -inf."""
    blk = _blocks(bounds, STATS_BLOCK)
    fin = E.op("gt", x, NEG_INF)
    m, es, mn = E.const([NEG_INF] * S), E.const([0] * S), E.const([INF] * S)
    for t in range(max(map(len, blk))):
        act = np.array([s for s in range(S) if t < len(blk[s])])
        segs = [blk[s][t] for s in act]
        seg = [n for _, n in segs]
        lanes = np.concatenate([np.arange(i, i + n) for i, n in segs])
        sid = np.repeat(np.arange(len(act)), seg)
        xl, fl = x[lanes], fin[lanes]
        bm = reduce(E, "nanmax", xl, seg)
        bmin = reduce(E, "nanmin", E.op("sel", fl, xl, INF), seg)
        ma = m[act]
        nm = E.op("max", ma, bm)
        if ma.k.all() and (ma.kv == NEG_INF).all():
            rescale = E.const([0] * len(act))
        else:
            rescale = E.op("sel", E.op("eq", ma, NEG_INF), 0, E.op("exp", E.op("sub", ma, nm)))
        e = E.op("sel", fl, E.op("exp", E.op("sub", xl, nm[sid])), 0)
        se = tree1d(E, pad(E, e, seg, STATS_BLOCK, 0), len(act), STATS_BLOCK, spt)
        es = put(es, act, E.op("fma", es[act], rescale, se))
        m = put(m, act, nm)
        mn = put(mn, act, E.op("min", mn[act], bmin))
    return m, es, mn, reduce(E, "or", fin, [len(fin)])


def row_stats(E, m, es, mn, anyf, S):
    """M, the two warp halves' Z (and their reciprocals), the halves' max / min probabilities and early returns."""
    M = reduce(E, "nanmax", m, [S])
    w = E.op("sel", E.op("eq", m, NEG_INF), 0, E.op("exp", E.op("sub", m, M)))
    prod = E.op("mul", es, w)
    if S == 1:
        Zh = prod[[0, 0]]
    else:
        h = S // 2
        f = E.op("fma", es, w, prod[np.arange(S) ^ h])
        Zh = cat([bfly(E, f[np.arange(h)], 1, h), bfly(E, f[np.arange(h, S)], 1, h)])
    rZ = E.op("rcp", Zh)
    maxp = E.op("div", ONE_F, Zh, rZ)
    emn = E.op("exp", E.op("sub", reduce(E, "nanmin", mn, [S]), M))
    minp = E.op("max", E.op("div", emn, Zh, rZ), TINY30)
    early = E.op("or", E.op("or", E.op("ge", 0, Zh), E.op("not", anyf)), E.op("ge", E.op("mul", minp, C6), maxp))
    return M, Zh, rZ, maxp, minp, early


def ladder(E, lo, hi):
    """Each half's F pivots ``lo * exp(log(hi / lo) * (f + 1) / (F + 1))`` (h-major)."""
    ll = E.op("log", E.op("div", hi, lo, E.op("rcp", lo)))
    hs = np.repeat(np.arange(2), F)
    t = E.op("mul", ll[hs], E.const([f32_word(f + 1.0) for f in range(F)] * 2))
    u = E.op("div", t, NINE, E.op("rcp", NINE))
    return E.op("mul", lo[hs], E.op("exp", u))


def step_kernel(E, pr, piv, h2, blk, S):
    """One round's partials (mass, min, count at the min) per pivot and slice, f-major (index f * S + s)."""
    V = len(pr)
    order = [(s, t) for s in range(S) for t in range(len(blk[s]))]
    at = {st: i for i, st in enumerate(order)}
    seg = [n for s, t in order for n in [blk[s][t][1]]] * F
    nb = len(order)
    pv = pr[np.tile(np.arange(V), F)]
    above = E.op("gt", pv, piv[(h2[None, :] * F + np.arange(F)[:, None]).ravel()])
    contrib = E.op("gate", above, pv)
    mass_b = tree1d(E, pad(E, contrib, seg, STEP_BLOCK, 0), F * nb, STEP_BLOCK, 1)
    tmin_b = reduce(E, "nanmin", E.op("sel", above, pv, INF), seg)
    sid = np.repeat(np.arange(F * nb), seg)
    teq = E.op("and", above, E.op("near", pv, tmin_b[sid]))
    tcnt_b = reduce(E, "iadd", E.op("b2i", teq), seg)
    mass, minl, nmin = E.const([0] * F * S), E.const([ONE_F] * F * S), E.const([0] * F * S, CW)
    for t in range(max(map(len, blk))):
        ss = [s for s in range(S) if t < len(blk[s])]
        ai = np.array([f * S + s for f in range(F) for s in ss])
        bi = np.array([f * nb + at[(s, t)] for f in range(F) for s in ss])
        tm, tc, ml = tmin_b[bi], tcnt_b[bi], minl[ai]
        is_new, is_same = E.op("gt", ml, tm), E.op("near", tm, ml)
        mass = put(mass, ai, E.op("add", mass[ai], mass_b[bi]))
        nmin = put(nmin, ai, E.op("seli", is_new, tc, E.op("iadd", nmin[ai], E.op("gatei", is_same, tc))))
        minl = put(minl, ai, E.op("nanmin", ml, tm))
    return mass, minl, nmin


def _state(E, lo, hi):
    z = lambda v, w=32: E.const([v, v], w)
    return {"lo": lo, "hi": hi, "best": z(NEG_ONE), "best_mass": z(0), "best_minl": z(ONE_F), "best_nmin": z(0, CW),
            "done": z(0, 1), "nomask": z(0, 1), "pivot": z(0), "dup": z(ONE_F), "numdup": z(1, CW), "numkeep": z(1, CW),
            "fin": E.const([0] * 2 * R * F, 1), "best_fin": E.const([0] * 2 * R * F, 1)}


def combine(E, part, st, piv, p, Zh, S, r):
    """``topp_split.combine`` for round r in both halves (the chain skips a half once it is done)."""
    pm, pl, pn = part
    rows = min(4, S)
    mass = bfly(E, bfly(E, pm, F * (S // rows), rows), F, S // rows)
    gmin = reduce(E, "nanmin", pl, [S] * F)
    fs = np.repeat(np.arange(F), S)
    nmin = reduce(E, "iadd", E.op("gatei", E.op("near", pl, gmin[fs]), pn), [S] * F)
    lhs = E.op("sub", mass, E.op("mul", gmin, E.op("i2f", nmin)))
    ge = E.op("ge", mass, p)
    ok = E.op("and", ge, E.op("gt", p, lhs))
    ok_oh, ge_oh = last_true(E, ok), last_true(E, ge)
    any_ok, any_ge = reduce(E, "or", ok, [F])[[0, 0]], reduce(E, "or", ge, [F])[[0, 0]]
    piv_ok, bp = pick(E, ok_oh, piv, 2), pick(E, ge_oh, piv, 2)
    two = np.zeros(2, np.int64)
    mass_ok, gmin_ok, nmin_ok = pick(E, ok_oh, mass, 1)[two], pick(E, ok_oh, gmin, 1)[two], pick(E, ok_oh, nmin, 1, "i")[two]
    upd = E.op("and", any_ge, E.op("gt", bp, st["best"]))
    best = E.op("sel", upd, bp, st["best"])
    best_mass = E.op("sel", upd, pick(E, ge_oh, mass, 1)[two], st["best_mass"])
    best_minl = E.op("sel", upd, pick(E, ge_oh, gmin, 1)[two], st["best_minl"])
    best_nmin = E.op("seli", upd, pick(E, ge_oh, nmin, 1, "i")[two], st["best_nmin"])
    ge2 = ge[np.tile(np.arange(F), 2)]
    new_lo = E.op("max", st["lo"], reduce(E, "nanmax", E.op("sel", ge2, piv, NEG_INF), [F, F]))
    new_hi = E.op("min", st["hi"], reduce(E, "nanmin", E.op("sel", ge2, INF, piv), [F, F]))
    if r == R - 1:
        stop = E.const([1, 1], 1)
    else:
        stop = E.op("not", E.op("gt", new_hi, E.op("mul", new_lo, C7)))
    fb = E.op("and", E.op("not", any_ok), stop)
    bpos = E.op("gt", best, 0)
    usebest = E.op("and", fb, bpos)
    nomask0 = E.op("and", fb, E.op("not", bpos))
    done = E.op("or", any_ok, fb)
    pivot = E.op("sel", any_ok, piv_ok, E.op("gate", usebest, best))
    sel_mass = E.op("sel", any_ok, mass_ok, E.op("gate", usebest, best_mass))
    dup = E.op("sel", any_ok, gmin_ok, E.op("sel", usebest, best_minl, ONE_F))
    numdup = E.op("seli", any_ok, nmin_ok, E.op("seli", usebest, best_nmin, 1))
    lo = E.op("sel", any_ok, st["lo"], new_lo)
    hi = E.op("sel", any_ok, st["hi"], new_hi)
    fin_ok = E.op("and", done, E.op("not", nomask0))
    nomask = E.op("selb", fin_ok, E.op("ge", E.op("mul", pivot, Zh), ONE_F), nomask0)
    q = E.op("div", E.op("max", E.op("sub", sel_mass, p), 0), dup, E.op("rcp", dup))
    nk = E.op("seli", fin_ok, E.op("numkeep", numdup, q), 1)
    ex = E.op("not", st["done"])
    new = {"lo": lo, "hi": hi, "best": best, "best_mass": best_mass, "best_minl": best_minl, "best_nmin": best_nmin, "done": done,
           "nomask": nomask, "pivot": pivot, "dup": dup, "numdup": numdup, "numkeep": nk}
    out = {k: E.op({32: "sel", CW: "seli", 1: "selb"}[v.w], ex, v, st[k]) for k, v in new.items()}
    hs = np.repeat(np.arange(2), F)
    slots = (np.arange(2)[:, None] * R * F + r * F + np.arange(F)[None, :]).ravel()
    set_ok = E.op("and", E.op("and", ex, any_ok)[hs], ok_oh[np.tile(np.arange(F), 2)])
    out["fin"] = put(st["fin"], slots, set_ok)
    set_b = E.op("and", ex, upd)
    others = np.array([i for i in range(2 * R * F) if i not in set(slots.tolist())])
    bf = put(st["best_fin"], others, E.op("and", st["best_fin"][others], E.op("not", set_b)[others // (R * F)]))
    out["best_fin"] = put(bf, slots, E.op("and", set_b[hs], ge_oh[np.tile(np.arange(F), 2)]))
    return out


def mask_kernel(E, x, pr, st, parts, Zh, M, S, bounds, h1, stop):
    """The mask kernel's keep bit per lane, each lane following its own warp half (``sampling.topp_keep``): kept when its half
    stops (``stop``: p >= 1, the half's early return or its ``nomask``); else above its half's pivot logit, and among the ties at
    ``dup`` only the first ``numkeep`` of the row, counting the ties of lanes whose half goes on (from the slices before, within
    the lane's [MASK_BLOCK] block of its slice). Whether ties are resolved at all is the first going-on half's call."""
    V = len(x)
    L = -(-V // S)
    P = R * F
    plog = E.op("add", E.op("add", E.op("log", st["pivot"]), E.op("log", Zh)), M)
    tie_h = E.op("ilt", st["numkeep"], st["numdup"])
    ties = E.op("selb", stop[[0]], tie_h[[1]], tie_h[[0]])
    fin_any = reduce(E, "or", st["fin"], [P, P])
    foh = E.op("selb", fin_any[np.repeat(np.arange(2), P)], st["fin"], st["best_fin"])
    minl = cat([pt[1] for pt in parts])
    nmin = cat([pt[2] for pt in parts])
    hh, ss, pp = (a.ravel() for a in np.meshgrid(np.arange(2), np.arange(S), np.arange(P), indexing="ij"))
    g = foh[hh * P + pp]
    minl_sel = reduce(E, "xorf", E.op("gate", g, minl[pp * S + ss]), [P] * (2 * S))
    nmin_sel = reduce(E, "xori", E.op("gatei", g, nmin[pp * S + ss]), [P] * (2 * S))
    dupc = E.op("gatei", E.op("near", minl_sel, st["dup"][np.repeat(np.arange(2), S)]), nmin_sel)
    pre = [E.const([0, 0], CW)]
    for s in range(1, S):
        pre.append(E.op("iadd", pre[-1], dupc[[s - 1, S + s - 1]]))
    pre = cat(pre)
    kc = E.op("gt", x, plog[h1])
    dm = E.op("and", E.op("eq", pr, st["dup"][h1]), E.op("not", stop[h1]))
    c = scan(E, E.op("b2i", dm), [n for b in _blocks(bounds, MASK_BLOCK) for _, n in b])
    cum = E.op("iadd", c, pre[(np.arange(V) // L) * 2 + h1])
    kd = E.op("and", dm, E.op("ile", cum, st["numkeep"][h1]))
    kt = E.op("and", kc, E.op("or", E.op("not", dm), kd))
    return E.op("or", E.op("selb", ties, kt, kc), stop[h1])




def pipeline(E, x, p, S, p1, parts_of=None):
    """The keep bits of ``topp_split.topp_keep_row(x, p, S)`` (all ones where ``p1``). ``parts_of(E, r, S)``, when given, stands in
    for round r's step kernel (a test's constructed partials, as ``topp_split``'s own test substitutes them)."""
    V = len(x)
    spt = spt_for(V, S)
    L = -(-V // S)
    bounds = [(min(s * L, V), min(s * L + L, V)) for s in range(S)]
    off = np.arange(V) % L
    h1 = z_group((off // spt) % 32, S)
    h2 = z_group(off % 32, S)
    with E.at("stats kernel"):
        m, es, mn, anyf = stats_kernel(E, x, S, spt, bounds)
    with E.at("row stats"):
        M, Zh, rZ, maxp, minp, early = row_stats(E, m, es, mn, anyf, S)
    with E.at("probabilities exp(x - M) / Z"):
        pr = E.op("div", E.op("exp", E.op("sub", x, M)), Zh[h1], rZ[h1])
    st = _state(E, minp, maxp)
    blk = _blocks(bounds, STEP_BLOCK)
    parts = []
    for r in range(R):
        with E.at("ladders"):
            piv = ladder(E, st["lo"], st["hi"])
        with E.at("step kernel"):
            stop = E.op("or", E.op("or", early[[0]], early[[1]]), E.op("or", st["done"][[0]], st["done"][[1]]))
            if parts_of is None:
                pm, pl, pn = step_kernel(E, pr, piv, h2, blk, S)
                part = [E.op("sel", stop, 0, pm), E.op("sel", stop, 0, pl), E.op("seli", stop, 0, pn)]
            else:
                part = parts_of(E, r, S)
        parts.append(part)
        with E.at("combine"):
            st = combine(E, part, st, piv, p, Zh, S, r)
    with E.at("mask kernel"):
        stop = E.op("or", p1, E.op("or", early, st["nomask"]))
        return mask_kernel(E, x, pr, st, parts, Zh, M, S, bounds, h1, stop)


def keep_word(E: Engine, x: Vec, p: Vec, splits: Vec, parts_of=None) -> Vec:
    """``TopPMaskWordx{V}_v1``'s V keep bits: the pipeline of the split count ``splits`` names, none at any other value."""
    V = len(x)
    assert V < 1 << (CW - 1)
    with E.at("p >= 1"):
        p1 = E.op("ge", p, ONE_F)
    out = E.const([0] * V, 1)
    for S in SPLIT_COUNTS:
        with E.at("split count"):
            is_s = E.op("weq", splits, S)
        if is_s.k.all() and not is_s.kv.any():
            continue
        if p1.k.all() and p1.kv.all():
            keep = E.const([1] * V, 1)
        else:
            with E.at(f"pipeline S = {S}"):
                keep = pipeline(E, x, p, S, p1, parts_of)
        with E.at("split count"):
            out = E.op("xor", out, E.op("and", keep, is_s))
    return out


__all__ = ["SPLIT_COUNTS", "F", "R", "NEG_INF", "Vec", "cat", "put", "Engine", "pipeline", "keep_word"]
