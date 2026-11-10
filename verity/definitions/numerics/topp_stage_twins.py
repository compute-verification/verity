"""Every unit of `topp_stages.plan(V)` on one `GumbelTopPTokenSelect_v1{V}` row, and the served token: `stage_rows`.

The twin runs `topp_stages`' own stage functions, each kind of unit over all its blocks at once, in :class:`ArrayEngine` (the
builder's `Engine` with each word gate on numpy arrays), so a unit's outputs here are the words its Definition computes from the
same inputs: what the serving reader commits for a row cut into those units, and what the tests hold each unit Definition to.
The select chunks' carries (positions, tie counts, best and index) are read off the row's lanes at each chunk's end.
"""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.evaluation import evaluate_batch, kernels
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import sampling_rows as SR
from verity.definitions.registry import topp_keep_builder as K
from verity.definitions.registry import topp_stages as T
from verity.definitions.registry.topp_word_gates import GATES, SIG, WIDTH

_M32 = np.uint64(0xFFFFFFFF)


def _f(a):
    return (np.asarray(a, np.uint64) & _M32).astype(np.uint32).view(np.float32)


def _w(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)


def _ieee(op):
    """An f32 operation whose reference is numpy's on scalars: the same on arrays, a NaN operand's lane by the reference."""
    def fn(prim, a, b):
        with np.errstate(all="ignore"):
            out = _w(op(_f(a), _f(b)))
        nan = np.isnan(_f(a)) | np.isnan(_f(b))
        if nan.any():
            out[nan] = [prim.evaluate(int(x), int(y)) for x, y in zip(a[nan], b[nan])]
        return out
    return fn


def _sub(prim, a, b):
    with np.errstate(all="ignore"):
        r = _f(a) - _f(b)
    return np.where(np.isnan(r), np.uint64(0x7FC00000), _w(r))


def _i32(a):
    return (np.asarray(a, np.uint64) & _M32).astype(np.uint32).view(np.int32)


#: word gates on arrays, each equal to its primitive's reference on every input (`test_topp_stages`' fast-engine check)
FAST = {
    "F32Add_v1": _ieee(np.add),
    "F32Mul_v1": _ieee(np.multiply),
    "F32Sub_v1": _sub,
    "F32GtStrict_v1": lambda p, a, b: (_f(a) > _f(b)).astype(np.uint64),
    "F32Eq_v1": lambda p, a, b: (_f(a) == _f(b)).astype(np.uint64),
    "F32Fabs_v1": lambda p, a: a & np.uint64(0x7FFFFFFF),
    "SelectF32_v1": lambda p, c, a, b: np.where(c & np.uint64(1), a, b),
    "SelectI32_v1": lambda p, c, a, b: np.where(c & np.uint64(1), a, b),
    "SelectBit_v1": lambda p, c, a, b: np.where(c & np.uint64(1), a, b) & np.uint64(1),
    "BitAnd_v1": lambda p, a, b: a & b & np.uint64(1),
    "BitOr_v1": lambda p, a, b: (a | b) & np.uint64(1),
    "BitXor_v1": lambda p, a, b: (a ^ b) & np.uint64(1),
    "BitNot_v1": lambda p, a: (a ^ np.uint64(1)) & np.uint64(1),
    "U32Xor_v1": lambda p, a, b: (a ^ b) & _M32,
    "I32Add_v1": lambda p, a, b: (a + b) & _M32,
    "I32Le_v1": lambda p, a, b: (_i32(a) <= _i32(b)).astype(np.uint64),
    "I32Eq_v1": lambda p, a, b: ((a & _M32) == (b & _M32)).astype(np.uint64),
}


class _Arrays:
    """A word gate's calls on n lanes of arrays (the `GATES` bodies' ``k``)."""

    def __init__(self, n: int):
        self.n = n

    def call(self, prim, *ops):
        a = [np.broadcast_to(np.asarray(o, np.uint64), (self.n,)) for o in ops]
        fast = FAST.get(prim.id)
        if fast is not None:
            return np.asarray(fast(prim, *a), np.uint64)
        if "numpy" in kernels(prim):
            return np.asarray(evaluate_batch(prim, *a), np.uint64).reshape(self.n)
        grid = np.stack(a, 1)
        uniq, inv = np.unique(grid, axis=0, return_inverse=True)
        vals = np.array([prim.evaluate(*(int(v) for v in row)) for row in uniq], np.uint64)
        return vals[inv.reshape(-1)]

    @staticmethod
    def c(v, w=32):
        return np.uint64(int(v))


class ArrayEngine(K.Engine):
    """The builder's `Engine` on one row (T = 1), each operation's instances by its word gate on numpy arrays."""

    def __init__(self):
        super().__init__(1)

    def _run(self, name, key, ins):
        fn, extra = GATES[name]
        free = iter(ins)
        cols = [next(free) if c is None else None for c in key]
        shape = next((x.shape for x in cols if x is not None), (1, 1))
        n = int(np.prod(shape))
        ops = [np.full(n, int(key[j]), np.uint64) if cols[j] is None else cols[j].reshape(-1).astype(np.uint64)
               for j in range(len(key))]
        ops += [np.full(n, int(v), np.uint64) for v, _w in extra]
        k = _Arrays(n)
        out = k.call(fn, *ops) if isinstance(fn, PrimitiveDefinition) else fn(k, *ops)
        out = np.broadcast_to(np.asarray(out, np.uint64), (n,))
        return (out & np.uint64((1 << WIDTH[SIG[name][1]]) - 1)).reshape(shape)


def vals(v) -> np.ndarray:
    return np.where(v.k, v.kv, v.d[0]).astype(np.uint64)


def _words(*vs) -> np.ndarray:
    return np.concatenate([vals(v) for v in vs])


def _pipeline(E, geo: T.Geo, x, p, outs: dict):
    """Split count S's units' outputs into ``outs``; returns its keep bits and its tie counts (the mask kernel's scan)."""
    S, F, R = geo.S, T.F, T.R
    blk = {s: geo.blocks(s, K.STATS_BLOCK) for s in geo.slices}
    run, es_c = {}, {}
    for t in range(max(map(len, blk.values()))):
        act = [s for s in geo.slices if t < len(blk[s])]
        segs = [blk[s][t] for s in act]
        lanes = np.concatenate([np.arange(i, i + n) for i, n in segs])
        bm, bmin, anyb = T.stats_block(E, x[lanes], [n for _, n in segs])
        carry = None if t == 0 else tuple(K.cat([run[s][k] for s in act]) for k in range(3))
        nm, rescale, mn, af = T.stats_fold(E, bm, bmin, anyb, carry)
        for i, s in enumerate(act):
            run[s] = (nm[[i]], mn[[i]], af[[i]])
            outs[("head", S, s, t)] = [_words(nm[[i]], rescale[[i]], mn[[i]], T.flag_words(E, af[[i]]))]
        nparts = [-(-n // T.STATS_PART) for _, n in segs]
        acc = {}
        for a in range(T.PARTS):
            inn = [i for i in range(len(act)) if a < nparts[i]]
            if not inn:
                break
            seg = [min(T.STATS_PART, segs[i][1] - a * T.STATS_PART) for i in inn]
            pl = np.concatenate([np.arange(segs[i][0] + a * T.STATS_PART, segs[i][0] + a * T.STATS_PART + n) for i, n in zip(inn, seg)])
            got = T.exp_part(E, x[pl], nm[inn], seg, None if a == 0 else K.cat([acc[i] for i in inn]), geo.spt)
            for j, i in enumerate(inn):
                acc[i] = got[np.arange(j * T.THREADS, (j + 1) * T.THREADS)]
            ends = [i for i in inn if nparts[i] == a + 1]
            for i in inn:
                if i not in ends:
                    outs[("exp", S, act[i], t, a)] = [vals(acc[i])]
            if ends:
                es0 = E.const([0] * len(ends)) if t == 0 else K.cat([es_c[act[i]] for i in ends])
                r0 = E.const([0] * len(ends)) if t == 0 else rescale[ends]
                es = T.exp_end(E, K.cat([acc[i] for i in ends]), T.PARTS - 1 - a, geo.spt, es0, r0)
                for j, i in enumerate(ends):
                    es_c[act[i]] = es[[j]]
                    outs[("exp", S, act[i], t, a)] = [vals(es[[j]])]
    stats = {s: (run[s][0], es_c[s], run[s][1], run[s][2]) for s in geo.slices}
    M, Zh, rZ, early, st, piv = T.row_head(E, *T.slice_stats(E, geo, stats), S)
    off = np.arange(geo.V) % geo.L
    h1 = geo.h1(off)
    pr = T.probs(E, x, M, Zh, rZ, h1)
    steps = [(s, t, lo, n) for s in geo.slices for t, (lo, n) in enumerate(geo.blocks(s, K.STEP_BLOCK))]
    nb = len(steps)
    rw = T.pack_state(E, M, Zh, rZ, st, piv, early)
    for b, (s, t, lo, n) in enumerate(steps):
        outs[("probs", S, b)] = [vals(pr[np.arange(lo, lo + n)])] + ([vals(rw)] if b == 0 else [])
    h2 = T.Geo.h2(off, S)
    parts = []
    for r in range(R):
        mb, tm, tc = T.step_block(E, pr, piv, h2, [n for *_, n in steps])
        cas = {}
        for t in range(max(st_[1] for st_ in steps) + 1):
            bs = [b for b, st_ in enumerate(steps) if st_[1] == t]
            ss = [steps[b][0] for b in bs]
            idx = (np.arange(F)[:, None] * nb + np.array(bs)[None, :]).ravel()
            fm = (np.arange(len(ss))[None, :] * F + np.arange(F)[:, None]).ravel()
            carry = None if t == 0 else tuple(K.cat([cas[s][k] for s in ss])[fm] for k in range(3))
            new = T.step_fold(E, (mb[idx], tm[idx], tc[idx]), carry)
            for j, (b, s) in enumerate(zip(bs, ss)):
                cas[s] = tuple(v[np.arange(F) * len(ss) + j] for v in new)
                if b != nb - 1:
                    outs[("step", S, r, b)] = [vals(T.pack_ca(cas[s]))]
        pm, pl, pn = T.gather_parts(E, geo, cas)
        st, part = T.round_combine(E, pm, pl, pn, st, piv, p, Zh, early, S, r)
        if r < R - 1:
            piv = K.ladder(E, st["lo"], st["hi"])
            outs[("step", S, r, nb - 1)] = [_words(part[1], part[2]), vals(T.pack_state(E, M, Zh, rZ, st, piv, early))]
        parts.append(part)
    plog, ties, stop, pre = T.mask_head(E, st, parts, Zh, M, p, early, S)
    outs[("step", S, R - 1, nb - 1)] = [vals(T.pack_scalars(E, M, Zh, rZ, plog, st, pre, stop, ties))]
    dm = T.tie_lanes(E, pr, h1, st["dup"], stop)
    c = K.scan(E, E.op("b2i", dm), [n for s in geo.slices for _, n in geo.blocks(s, K.MASK_BLOCK)])
    cum = E.op("iadd", c, pre[(np.arange(geo.V) // geo.L) * 2 + h1])
    return T.keep_lanes(E, x, dm, cum, h1, plog, st["numkeep"], stop, ties), vals(c)


def _argmax_prefix(y: np.ndarray) -> np.ndarray:
    """The strict scan's index after each lane: the first lane of the greatest y so far (a NaN never wins; a NaN first lane
    is never beaten)."""
    yf = _f(y)
    n = len(yf)
    if np.isnan(yf[0]):
        return np.zeros(n, np.int64)
    cand = np.where(np.isnan(yf), np.float32(-np.inf), yf)
    run = np.maximum.accumulate(cand)
    up = np.ones(n, bool)
    up[1:] = cand[1:] > run[:-1]
    return np.maximum.accumulate(np.where(up, np.arange(n), 0))


def _scaled(E, V: int, logits, temp: int):
    lg = np.asarray(logits, np.uint16).reshape(-1)
    return T.temp_scale(E, E.free(V, 32, (lg.astype(np.uint64) << np.uint64(16))[None]), E.free(1, 32, [[int(temp)]]))


def keep_bits(V: int, logits, top_p: int, temp: int, S: int) -> np.ndarray:
    """Split count S's keep bits on the row (``top_p``, ``temp`` f32 bits), as its units compute them."""
    E = ArrayEngine()
    keep, _ = _pipeline(E, T.Geo(V, S), _scaled(E, V, logits, temp), E.free(1, 32, [[int(top_p)]]), {})
    return vals(keep).astype(bool)


def stage_rows(V: int, logits, top_p: int, seed: int, pos: int, temp: int, splits: int) -> tuple[int, dict]:
    """(token, outs): the served token of ``GumbelTopPTokenSelect_v1{V}`` on the row, and ``outs[key]`` every unit's output
    fields by its `topp_stages.plan(V)` key (each a u64 word array)."""
    E = ArrayEngine()
    x = _scaled(E, V, logits, temp)
    p = E.free(1, 32, [[int(top_p)]])
    outs: dict = {}
    keep, counts = np.zeros(V, bool), {}
    for S in K.SPLIT_COUNTS:
        k, counts[S] = _pipeline(E, T.Geo(V, S), x, p, outs)
        if S == int(splits):
            keep = vals(k).astype(bool)
    xs = vals(x)
    masked = np.where(keep, xs, np.uint64(K.NEG_INF))
    noisy = not SR.f32_eq(int(temp), 0)
    y = SR._u32(SR.add_ftz(SR._u32(masked), SR.noise_row(SA.stream_key(int(seed), int(pos)), V))).astype(np.uint64) if noisy \
        else masked
    bi = _argmax_prefix(y)
    units = [u for u in T.plan(V) if u.key[0] == "select"]
    for u in units[:-1]:
        hi = u.lanes[1]
        words = [y[bi[hi - 1]], bi[hi - 1], hi]
        for S in K.SPLIT_COUNTS:
            geo = T.Geo(V, S)
            sl, off = hi // geo.L, hi % geo.L
            c = counts[S][hi - 1]
            words += [off % K.MASK_BLOCK, c] if S == 1 else [off, off % K.MASK_BLOCK, off % geo.P, sl, c]
        outs[u.key] = [np.array(words, np.uint64)]
    outs[units[-1].key] = [np.array([bi[-1]], np.uint64)]
    return int(bi[-1]), outs


__all__ = ["ArrayEngine", "FAST", "keep_bits", "stage_rows", "vals"]
