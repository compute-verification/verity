"""vLLM's standalone top-p kernel path as word gates: `TopPStandaloneKeep_v1{V}`, `TopPStandaloneMask_v1{V}` and
`GumbelTopPStandaloneSelect_v1{V}`, the served token select of a sampler batch the split pipeline does not take (more than
`topp_geometry.SPLIT_MAX_BATCH` rows, top-k off).

`standalone_keep` is `topp_standalone.standalone_row` written over vectors of words with the split pipeline's builder
(`topp_keep_builder`: an `Engine` folds constants and wires what a constant decides, `topp_words.IREngine` emits the rest as
batches of `topp_word_gates.GATES`), pass by pass:

* the zeroth pass's avg, sum of squares (one fma chain a thread) and std (`MufuSqrtFtz`) of the first [BLOCK] lanes;
* the first pass's row max, finite min and per-tile exp sums, added tile by tile;
* the Gaussian cut: sigma by a one-hot of ``p * 200 >= k`` over the table's 200 cut values (constants);
* the second pass's probabilities, outlier bits and outlier mass;
* the compaction: each outlier moved down by the count of non-outliers before it, one stage per bit of that count (LSB first,
  so no two outliers ever meet), into [BLOCK_TRUNC] tiles padded with 0;
* both searches, over the compacted outliers in [BLOCK_TRUNC] tiles and over every lane in [BLOCK] tiles, each 18 rounds
  unrolled: a round's result is latched in the round its found test or its stop test first holds, and a later round changes
  nothing.  An all-zero tile adds nothing to a round (its lanes are never above a pivot >= 0), so the compacted search reads
  every tile the vocabulary could fill, whatever the outlier count;
* the result of the branch ``sum_outliers > p`` names, the pivot and duplicate logits, and the mask: a lane above the pivot
  logit, and of the lanes within 1e-9 of the duplicate logit only the first num_keep (an inclusive count) when num_keep is
  below the duplicate count.  ``num_keep = count - u32((mass - p) / min)`` is below the count exactly when 1 <= (mass - p) /
  min < count + 1, and is then 0 or `numkeep`'s value.

Every lane's keep bit equals `standalone_row`'s on every row (the test evaluates the Definitions with the circuit evaluator).
The body has no dead gate: the last round skips the found and stop tests a round 18 never reads, and the last compaction stage
skips the shift words no stage after it reads.
"""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import bind, composite
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BF16, BIT, F32, I32, SelectF32
from verity.circuits.refs import Coll
from verity.circuits.types import Array, Value
from verity.definitions.registry.topp_keep_builder import (INF, NEG_INF, ONE_F, Engine, Vec, bfly, cat, pad, put, reduce, scan,
                                                              tree1d)
from verity.definitions.registry.topp_split import fma32
from verity.definitions.registry.topp_standalone import BLOCK, BLOCK_TRUNC, MAX_ROUNDS, NORMAL_CDF_TO_SIGMA, NUM_WARPS, SPT
from verity.definitions.registry.topp_word_gates import CW, EPS9, HALF, f32_word
from verity.definitions.registry.topp_words import IREngine

I64 = Value(64)
SIGN = 0x80000000
TEN, TWO = f32_word(10.0), f32_word(2.0)
#: each table entry's cut, sigma + |sigma| * -0.25 (the product is exact, so the fma is one rounding)
SIGMA_CUT = tuple(f32_word(fma32(np.float32(abs(s)), np.float32(-0.25), np.float32(s))) for s in NORMAL_CDF_TO_SIGMA)


def _sumsq(E: Engine, v: Vec) -> Vec:
    """`tl.sum(x * x)` of one [BLOCK] block (`topp_standalone.tree_sumsq`)."""
    reps = BLOCK // (32 * NUM_WARPS) // SPT
    g = np.arange(BLOCK).reshape(reps, NUM_WARPS, 32, SPT)
    regs = [g[r, :, :, j].ravel() for r in range(reps) for j in range(SPT)]
    acc = E.op("fma", v[regs[0]], v[regs[0]], E.op("mul", v[regs[1]], v[regs[1]]))
    for r in regs[2:]:
        acc = E.op("fma", v[r], v[r], acc)
    return bfly(E, bfly(E, acc, NUM_WARPS, 32), 1, NUM_WARPS)


def _tiles_sum(E: Engine, v: Vec, B: int) -> Vec:
    """The tiles' `tl.sum`s added tile by tile from 0 (``s += tl.sum(tile)``; 0 + the first is the first)."""
    nt = len(v) // B
    t = tree1d(E, v, nt, B, SPT)
    s = t[[0]]
    for i in range(1, nt):
        s = E.op("add", s, t[[i]])
    return s


def _cut(E: Engine, p: Vec) -> Vec:
    """The table cut at ``clamp(int(p * 200), 0, 199)``: the index is k exactly when p * 200 >= k and not >= k + 1 (NaN: 0)."""
    n = len(SIGMA_CUT)
    q = E.op("mul", p, f32_word(200.0))
    ge = E.op("ge", q[np.zeros(n - 1, np.int64)], E.const([f32_word(float(k)) for k in range(1, n)]))
    lt = E.op("not", ge)
    oh = cat([lt[[0]], E.op("and", ge[np.arange(n - 2)], lt[np.arange(1, n - 1)]), ge[[n - 2]]])
    return reduce(E, "xorf", E.op("gate", oh, E.const(list(SIGMA_CUT))), [n])


def _round_stats(E: Engine, data: Vec, B: int, piv: Vec):
    """One round's (mass, min, count) over ``data`` in [B] tiles, the lanes past it 0, at the pivot: each tile's `tl.sum` of the
    lanes above it, least lane above it (1.0 if none) and lanes within 1e-9 of that, merged tile by tile as
    `_update_min_larger_stats` does.  A padding lane is never above the pivot, which is +0 or more or NaN (a probability, or a
    rounded point between two), so its terms are constants: 0 to the sum and the count, 1.0 to the min."""
    L = len(data)
    nt = -(-L // B)
    above = E.op("gt", data, piv[np.zeros(L, np.int64)])
    mass_t = tree1d(E, pad(E, E.op("gate", above, data), [L], nt * B, 0), nt, B, SPT)
    seg = [B] * nt
    tmin = reduce(E, "nanmin", pad(E, E.op("sel", above, data, ONE_F), [L], nt * B, ONE_F), seg)
    teq = E.op("and", above, E.op("near", data, tmin[np.arange(L) // B]))
    tcnt = reduce(E, "iadd", pad(E, E.op("b2i", teq), [L], nt * B, 0), seg)
    mass, ml, n = mass_t[[0]], E.const([ONE_F]), E.const([0], CW)
    for t in range(nt):
        tm, tc = tmin[[t]], tcnt[[t]]
        if t:
            mass = E.op("add", mass, mass_t[[t]])
        is_new, is_same = E.op("gt", ml, tm), E.op("near", tm, ml)
        n = E.op("seli", is_new, tc, E.op("iadd", n, E.op("gatei", is_same, tc)))
        ml = E.op("min", ml, tm)
    return mass, ml, n


def search(E: Engine, data: Vec, B: int, p: Vec, lo: Vec, hi: Vec):
    """`topp_standalone.search`: (pivot, min, count, mass) of the round the search ends in."""
    done = out = None
    for it in range(1, MAX_ROUNDS + 1):
        piv = E.op("fma", E.op("sub", hi, lo), HALF, lo)
        mass, ml, n = _round_stats(E, data, B, piv)
        lo = E.op("sel", E.op("gt", mass, p), piv, lo)
        hi = E.op("sel", E.op("gt", p, mass), piv, hi)
        mid = E.op("div", E.op("add", hi, lo), TWO, E.op("rcp", TWO))
        if it == MAX_ROUNDS:
            new = (mid, ml, n, mass)
            return tuple(E.op("seli" if v.w == CW else "sel", done, o, v) for o, v in zip(out, new))
        found = E.op("and", E.op("ge", mass, p), E.op("gt", p, E.op("fma", E.op("xorf", ml, SIGN), E.op("i2f", n), mass)))
        stop = E.op("gt", EPS9, E.op("sub", hi, lo))
        new = (E.op("sel", stop, mid, piv), ml, n, mass)
        end = E.op("or", found, stop)
        if out is None:
            out, done = new, end
            continue
        out = tuple(E.op("seli" if v.w == CW else "sel", done, o, v) for o, v in zip(out, new))
        done = E.op("or", done, end)
    raise AssertionError("unreachable")


def _bits(E: Engine, z: Vec, nb: int) -> list[Vec]:
    """Bits 0 .. nb - 1 of each count of z (below 2^nb): bit b is the top bit of the word z << (31 - b)."""
    y = z
    for _ in range(32 - nb):
        y = E.op("shl1", y)
    bits = [y] * nb
    for b in range(nb - 1, -1, -1):
        bits[b] = E.op("top", y)
        if b:
            y = E.op("shl1", y)
    return bits


def compact(E: Engine, v: Vec, flag: Vec) -> Vec:
    """The values whose flag is set, in lane order from lane 0, every lane after them 0: each flagged lane moves down by z, the
    count of unflagged lanes before it, by the stages 1, 2, 4, ...: stage b moves a lane down 2^b when z has bit b, and the
    bits still to come move with it.  After stage b a flagged lane j sits at j - (z mod 2^(b+1)), so flagged lanes stay
    distinct and in order."""
    V = len(v)
    nb = (V - 1).bit_length()
    bits = _bits(E, scan(E, E.op("b2i", E.op("not", flag)), [V]), nb) if nb else []
    valid = flag
    for b in range(nb):
        s = 1 << b
        j = np.arange(V - s)
        mv = E.op("and", valid[j + s], bits[b][j + s])
        stay = E.op("and", valid, E.op("not", bits[b]))
        v = put(v, j, E.op("sel", mv, v[j + s], v[j]))
        valid = put(stay, j, E.op("or", mv, stay[j]))
        for k in range(b + 1, nb):
            bits[k] = put(bits[k], j, E.op("selb", mv, bits[k][j + s], bits[k][j]))
    return E.op("gate", valid, v)


def standalone_keep(E: Engine, x: Vec, p: Vec) -> Vec:
    """The V keep bits of ``topp_standalone.standalone_row(x, p)``."""
    V = len(x)
    assert V < 1 << (CW - 1)
    with E.at("p < 1"):
        run = E.op("gt", ONE_F, p)
    with E.at("zeroth pass"):
        x0 = pad(E, x[np.arange(min(V, BLOCK))], [min(V, BLOCK)], BLOCK, NEG_INF)
        fin = E.op("gt", x0, NEG_INF)
        nf = reduce(E, "iadd", E.op("b2i", fin), [BLOCK])
        fl = E.op("gate", fin, x0)
        nz = E.op("not", E.op("ile", nf, 0))
        nff = E.op("i2f", nf)
        rn = E.op("rcp", nff)
        avg = E.op("gate", nz, E.op("div", tree1d(E, fl, 1, BLOCK, SPT), nff, rn))
        sq = E.op("gate", nz, E.op("div", _sumsq(E, fl), nff, rn))
        std = E.op("sqrt", E.op("max", E.op("fma", E.op("xorf", avg, SIGN), avg, sq), 0))
        ms = E.op("fma", std, TEN, avg)
    nt = -(-V // BLOCK)
    with E.at("first pass"):
        mx = reduce(E, "nanmax", x, [V])
        mn = E.op("min", reduce(E, "nanmin", E.op("sel", E.op("gt", x, NEG_INF), x, INF), [V]), mx)
        e = E.op("exp", E.op("sub", x, ms[np.zeros(V, np.int64)]))
        se = _tiles_sum(E, pad(E, e, [V], nt * BLOCK, 0), BLOCK)
        rse = E.op("rcp", se)
    with E.at("gaussian cut"):
        opiv = E.op("fma", std, _cut(E, p), avg)
        oprob = E.op("div", E.op("exp", E.op("sub", opiv, ms)), se, rse)
    with E.at("second pass"):
        zero = np.zeros(V, np.int64)
        probs = E.op("div", e, se[zero], rse[zero])
        out = E.op("gt", probs, oprob[zero])
        so = _tiles_sum(E, pad(E, E.op("gate", out, probs), [V], nt * BLOCK, 0), BLOCK)
        hi = E.op("div", E.op("exp", E.op("sub", mx, ms)), se, rse)
        lo = E.op("div", E.op("exp", E.op("sub", mn, ms)), se, rse)
    with E.at("compaction"):
        buf = compact(E, probs, out)
    with E.at("outlier search"):
        res_o = search(E, buf, BLOCK_TRUNC, p, oprob, hi)
    with E.at("full search"):
        res_f = search(E, probs, BLOCK, p, lo, hi)
    with E.at("mask"):
        ob = E.op("gt", so, p)
        piv, ml, n, mass = (E.op("seli" if a.w == CW else "sel", ob, a, b) for a, b in zip(res_o, res_f))
        dup = E.op("add", E.op("log", E.op("mul", ml, se)), ms)
        fpiv = E.op("add", E.op("log", E.op("mul", piv, se)), ms)
        q = E.op("div", E.op("sub", mass, p), ml, E.op("rcp", ml))
        trim = E.op("and", E.op("ge", q, ONE_F), E.op("gt", E.op("i2f", E.op("iadd", n, 1)), q))
        nk = E.op("seli", E.op("ge", q, E.op("i2f", n)), 0, E.op("numkeep", n, q))
        keep_all = E.op("or", E.op("not", run), E.op("or", E.op("not", E.op("gt", mx, fpiv)), E.op("eq", fpiv, NEG_INF)))
        d = E.op("near", x, dup[zero])
        rank = scan(E, E.op("b2i", d), [V])
        drop = E.op("and", E.op("and", d, trim[zero]), E.op("not", E.op("ile", rank, nk[zero])))
        keep = E.op("and", E.op("gt", x, fpiv[zero]), E.op("not", drop))
        return E.op("or", keep, keep_all[zero])


def standalone_keep_body(B, logits: Coll, top_p: Coll) -> Coll:
    E = IREngine(B)
    return E.result(standalone_keep(E, E.param(logits, 32), E.param(top_p, 32)), 1)


@composite("TopPStandaloneKeep", 1, ["V"], lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32)), Array(st.V, BIT)),
           conformance="exact-model-tested")
def TopPStandaloneKeep(B, st, logits, top_p):
    """The keep bits vLLM's standalone top-p kernel path leaves on one row (`topp_standalone.standalone_row`), as word gates:
    each operation of :func:`standalone_keep` a batch of its word gate over the lanes it touches."""
    return standalone_keep_body(B, logits, top_p)


@composite("TopPStandaloneMask", 1, ["V"], lambda st: ((("logits", Array(st.V, F32)), ("top_p", F32)), Array(st.V, F32)),
           conformance="exact-model-tested")
def TopPStandaloneMask(B, st, logits, top_p):
    """The row the kernel stores: each lane kept or set to -inf by `TopPStandaloneKeep_v1{V}`'s bit."""
    keep = B.call(bind(TopPStandaloneKeep, V=st.V), logits, top_p)
    return B.batch(SelectF32, keep, logits, B.call(const(32, NEG_INF)), axes=(0, 0, None))


@composite("GumbelTopPStandaloneSelect", 1, ["V"],
           lambda st: ((("logits", Array(st.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
           conformance="exact-model-tested")
def GumbelTopPStandaloneSelect(B, st, logits, top_p, seed, pos, temp):
    """`GumbelTopPTokenSelect_v3{V,S}` with the standalone path's mask in place of the split pipeline's: the served token of a row
    the sampler masks with `_topk_topp_kernel` (a batch over `topp_geometry.SPLIT_MAX_BATCH` rows, top-k off)."""
    from verity.definitions.registry import dead_free as DF
    from verity.definitions.registry import sampling as SA

    scaled = B.call(bind(SA.TemperatureScale, V=st.V), logits, temp)
    masked = B.call(bind(TopPStandaloneMask, V=st.V), scaled, top_p)
    return B.call(bind(DF.GumbelSelectF32V3, V=st.V), masked, seed, pos, temp)


__all__ = ["SIGMA_CUT", "compact", "search", "standalone_keep", "standalone_keep_body", "TopPStandaloneKeep", "TopPStandaloneMask",
           "GumbelTopPStandaloneSelect"]
