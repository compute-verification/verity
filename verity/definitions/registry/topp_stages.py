"""`GumbelTopPTokenSelect_v1{V}` (and `GumbelTopPTokenSelectSharedGreedy_v1{V}`, the same function) as units of at most a Flock
block each, ``splits`` still an operand: `plan` lists them with their wiring, `cut` counts them by Definition (`sampling.CUTS`).

The Call's keep word `TopPMaskWordx{V}_v1` runs the split top-p pipeline at every split count S and keeps the one ``splits``
names (`topp_keep_builder.keep_word`), so every S's pipeline is cut here, on its own slices and blocks (`Geo`).  A unit's body is
the builder's own operations on its lanes in an `IREngine` (each operation one batch of its word gate), with only the gates its
outputs read (`dead_free.live_replay`): the units compute the builder's words, and their composition is the Call word for word.
Every unit reads its own lanes of the bf16 logits and recomputes their temperature scale (`TemperatureLane_v1`'s gates).  Per S:

* ``TopPStatsHead_v1{N,FIRST}``: a stats block (8,192 lanes of a slice): its max, its finite lanes' min and whether any is finite,
  folded into the slice's running max ``m``, min and flag, with the rescale of the slice's exp sum (carried block to block);
* ``TopPStatsExp_v1{N,SPT,AT,LAST,FIRST}``: part AT (`STATS_PART` lanes) of a stats block: its lanes' exps added into the block's
  256 thread sums in `tree_sum_1d`'s order (carried part to part); the last part finishes the block's sum into the slice's exp sum;
* ``TopPProbs_v1{V,S,N,HEAD}``: a step block's probabilities; HEAD (the row's first block) first reduces every slice's stats into
  the row's (`row_stats`) and round 0's state and pivots;
* ``TopPStep_v1{V,S,N,FIRST,COMBINE}``: a round's step kernel on a step block (8 pivots' mass, min and count at the min), carried
  block to block within the slice; at COMBINE = r (the row's last block) round r's combine over every slice's partials, and after
  the last round the mask kernel's row scalars.

Then the row in chunks of N lanes, ``GumbelTopPSelect_v1{V,N,FIRST,LAST}``: every S's keep bits (its mask kernel, each lane's
slice, warp-half group and tie count from the positions carried chunk to chunk), those ``splits`` names, the -inf mask, the noise
and the select, whose best and index are carried too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.definitions.ops.fp32 import F32AddFtz
from verity.definitions.ops.prims import Bf16ToF32
from verity.definitions.ops.scalar import BF16, F32, I32
from verity.circuits.refs import Coll, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.registry import topp_keep_builder as K
from verity.definitions.registry.dead_free import live_replay
from verity.definitions.registry.sampling import GumbelNoiseLane, GumbelStreamKey
from verity.definitions.registry.topp_geometry import spt_for, z_group
from verity.definitions.registry.topp_word_gates import CW
from verity.definitions.registry.topp_words import IREngine

I64 = Value(64)
F, R = K.F, K.R
NEG_INF, INF, ONE_F = K.NEG_INF, K.INF, K.ONE_F
THREADS = 32 * K.NUM_WARPS
#: the lanes of a stats block one ``TopPStatsExp_v1`` unit adds (a stats block is `PARTS` of them)
STATS_PART = 4096
PARTS = K.STATS_BLOCK // STATS_PART
#: the lanes of a ``GumbelTopPSelect_v1`` chunk, at most (`Geo.chunk`)
SELECT_N = 256
_CONF = "exact-model-tested"


# --- geometry ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Geo:
    """Split count S's launch over a row of V: slices of L lanes (the last ones shorter or empty), each lane's warp-half group
    ``h1`` and the step kernel's ``h2`` (`topp_geometry.z_group`)."""
    V: int
    S: int

    @property
    def L(self) -> int:
        return -(-self.V // self.S)

    @property
    def spt(self) -> int:
        return spt_for(self.V, self.S)

    @property
    def P(self) -> int:
        """The period of ``h1`` along a slice: lane groups of spt * S."""
        return self.spt * self.S

    def bounds(self, s: int) -> tuple[int, int]:
        return min(s * self.L, self.V), min(s * self.L + self.L, self.V)

    @property
    def slices(self) -> list[int]:
        return [s for s in range(self.S) if self.bounds(s)[0] < self.bounds(s)[1]]

    def blocks(self, s: int, B: int) -> list[tuple[int, int]]:
        lo, hi = self.bounds(s)
        return [(i, min(B, hi - i)) for i in range(lo, hi, B)]

    def h1(self, off) -> np.ndarray:
        return z_group((np.asarray(off) // self.spt) % 32, self.S)

    @staticmethod
    def h2(off, S: int) -> np.ndarray:
        return z_group(np.asarray(off) % 32, S)


def chunk(V: int) -> int:
    """A select chunk's lanes: at most `SELECT_N`, and at most a slice at S = 32, so a chunk crosses into the next slice at most
    once at every S (`_positions`)."""
    return min(SELECT_N, -(-V // 32), V)


# --- the stages, on vectors of words (any number of blocks at once, so the twin runs a kind of unit over all of them) ------------

def temp_scale(E, xf, temp):
    """`TemperatureLane_v1` on widened lanes: x / T by `div.full.f32`, the widening itself when T is 0 or 1."""
    skip = E.op("or", E.op("eq", temp, 0), E.op("eq", temp, ONE_F))
    return E.op("sel", skip, xf, E.op("div", xf, temp, E.op("rcp", temp)))


def stats_block(E, x, seg):
    """Per stats block (lengths ``seg``): its max, the min of its lanes above -inf, and whether any is."""
    fin = E.op("gt", x, NEG_INF)
    return (K.reduce(E, "nanmax", x, seg), K.reduce(E, "nanmin", E.op("sel", fin, x, INF), seg), K.reduce(E, "or", fin, seg))


def stats_fold(E, bm, bmin, anyb, carry=None):
    """Blocks' stats folded into their slices' running (m, min, any) (``carry``; None: each block starts its slice): the new max,
    the rescale of the slice's exp sum so far, the new min and flag."""
    n = len(bm)
    if carry is None:
        nm = E.op("max", E.const([NEG_INF] * n), bm)
        return nm, E.const([0] * n), E.op("min", E.const([INF] * n), bmin), anyb
    ma, mn, af = carry
    nm = E.op("max", ma, bm)
    rescale = E.op("sel", E.op("eq", ma, NEG_INF), 0, E.op("exp", E.op("sub", ma, nm)))
    return nm, rescale, E.op("min", mn, bmin), E.op("or", af, anyb)


def thread_add(E, v, acc, nb: int, spt: int):
    """nb parts of `STATS_PART` lanes (v, padded) added into their blocks' 256 thread sums (``acc``; None: each part starts its
    block), each thread over its (rep, j) registers in `tree_sum_1d`'s order."""
    reps = STATS_PART // THREADS // spt
    g = np.arange(nb * STATS_PART).reshape(nb, reps, K.NUM_WARPS, 32, spt)
    for rep in range(reps):
        for j in range(spt):
            col = v[g[:, rep, :, :, j].ravel()]
            acc = col if acc is None else E.op("add", acc, col)
    return acc


def exp_part(E, x, nm, seg, acc, spt: int):
    """Parts (lengths ``seg``) of stats blocks of maxima ``nm``: each lane's exp(x - m) (0 at -inf) into the thread sums."""
    fin = E.op("gt", x, NEG_INF)
    e = E.op("sel", fin, E.op("exp", E.op("sub", x, nm[np.repeat(np.arange(len(seg)), seg)])), 0)
    return thread_add(E, K.pad(E, e, seg, STATS_PART, 0), acc, len(seg), spt)


def exp_end(E, acc, pad: int, spt: int, es, rescale):
    """The blocks' last parts: the ``pad`` parts past their lanes (zeros) added, the thread sums reduced (warp, then cross-warp
    butterflies) and the block's sum folded into its slice's exp sum: fma(es, rescale, sum)."""
    nb = len(acc) // THREADS
    for _ in range(pad):
        acc = thread_add(E, E.const([0] * (nb * STATS_PART)), acc, nb, spt)
    se = K.bfly(E, K.bfly(E, acc, nb * K.NUM_WARPS, 32), nb, K.NUM_WARPS)
    return E.op("fma", es, rescale, se)


def row_head(E, m, es, mn, anyf, S: int):
    """`row_stats` over the slices' stats, round 0's state (`_state`) and its pivots."""
    M, Zh, rZ, maxp, minp, early = K.row_stats(E, m, es, mn, anyf, S)
    st = K._state(E, minp, maxp)
    return M, Zh, rZ, early, st, K.ladder(E, st["lo"], st["hi"])


def pick(E, v2, h1):
    """Each lane's word of its warp-half group: ``v2[h1]`` for a known ``h1``, else a select."""
    if isinstance(h1, np.ndarray):
        return v2[h1]
    return E.op({32: "sel", CW: "seli", 1: "selb"}[v2.w], h1, v2[[1]], v2[[0]])


def probs(E, x, M, Zh, rZ, h1):
    return E.op("div", E.op("exp", E.op("sub", x, M)), pick(E, Zh, h1), pick(E, rZ, h1))


def step_block(E, pr, piv, h2, seg):
    """Per step block (lengths ``seg``) and pivot, f-major (f * nb + b): the mass above the pivot, the least probability above it
    and the count of lanes near that least (`step_kernel`'s per-block partials)."""
    nb, n = len(seg), len(pr)
    segF = list(seg) * F
    pv = pr[np.tile(np.arange(n), F)]
    above = E.op("gt", pv, piv[(h2[None, :] * F + np.arange(F)[:, None]).ravel()])
    mass = K.tree1d(E, K.pad(E, E.op("gate", above, pv), segF, K.STEP_BLOCK, 0), F * nb, K.STEP_BLOCK, 1)
    tmin = K.reduce(E, "nanmin", E.op("sel", above, pv, INF), segF)
    teq = E.op("and", above, E.op("near", pv, tmin[np.repeat(np.arange(F * nb), segF)]))
    return mass, tmin, K.reduce(E, "iadd", E.op("b2i", teq), segF)


def step_fold(E, part, carry=None):
    """Blocks' partials folded into their slices' (mass, least, count) (``carry``; None: each block starts its slice)."""
    mb, tm, tc = part
    n = len(mb)
    mass, minl, nmin = carry if carry is not None else (E.const([0] * n), E.const([ONE_F] * n), E.const([0] * n, CW))
    is_new, is_same = E.op("gt", minl, tm), E.op("near", tm, minl)
    return (E.op("add", mass, mb), E.op("nanmin", minl, tm),
            E.op("seli", is_new, tc, E.op("iadd", nmin, E.op("gatei", is_same, tc))))


def round_combine(E, pm, pl, pn, st, piv, p, Zh, early, S: int, r: int):
    """Round r's partials (f-major over the S slices) gated by the round's stop, and its combine: the new state and the gated
    partials (the mask kernel reads every round's)."""
    stop = E.op("or", E.op("or", early[[0]], early[[1]]), E.op("or", st["done"][[0]], st["done"][[1]]))
    part = [E.op("sel", stop, 0, pm), E.op("sel", stop, 0, pl), E.op("seli", stop, 0, pn)]
    return K.combine(E, part, st, piv, p, Zh, S, r), part


def mask_head(E, st, parts, Zh, M, p, early, S: int):
    """The mask kernel's row scalars (`mask_kernel` before its lanes): each half's pivot logit, whether ties are resolved, each
    half's stop and the ties of the slices before each (h-minor, ``pre[2 s + h]``)."""
    stop = E.op("or", E.op("ge", p, ONE_F), E.op("or", early, st["nomask"]))
    P = R * F
    plog = E.op("add", E.op("add", E.op("log", st["pivot"]), E.op("log", Zh)), M)
    tie_h = E.op("ilt", st["numkeep"], st["numdup"])
    ties = E.op("selb", stop[[0]], tie_h[[1]], tie_h[[0]])
    fin_any = K.reduce(E, "or", st["fin"], [P, P])
    foh = E.op("selb", fin_any[np.repeat(np.arange(2), P)], st["fin"], st["best_fin"])
    minl = K.cat([pt[1] for pt in parts])
    nmin = K.cat([pt[2] for pt in parts])
    hh, ss, pp = (a.ravel() for a in np.meshgrid(np.arange(2), np.arange(S), np.arange(P), indexing="ij"))
    g = foh[hh * P + pp]
    minl_sel = K.reduce(E, "xorf", E.op("gate", g, minl[pp * S + ss]), [P] * (2 * S))
    nmin_sel = K.reduce(E, "xori", E.op("gatei", g, nmin[pp * S + ss]), [P] * (2 * S))
    dupc = E.op("gatei", E.op("near", minl_sel, st["dup"][np.repeat(np.arange(2), S)]), nmin_sel)
    pre = [E.const([0, 0], CW)]
    for s in range(1, S):
        pre.append(E.op("iadd", pre[-1], dupc[[s - 1, S + s - 1]]))
    return plog, ties, stop, K.cat(pre)


def tie_lanes(E, pr, h1, dup, stop):
    """The lanes at their half's tie probability whose half goes on (the mask kernel's ``dm``)."""
    return E.op("and", E.op("eq", pr, pick(E, dup, h1)), E.op("not", pick(E, stop, h1)))


def keep_lanes(E, x, dm, cum, h1, plog, numkeep, stop, ties):
    """The mask kernel's keep bits from each lane's tie bit and its count of ties up to it in the row (``cum``)."""
    kc = E.op("gt", x, pick(E, plog, h1))
    kd = E.op("and", dm, E.op("ile", cum, pick(E, numkeep, h1)))
    kt = E.op("and", kc, E.op("or", E.op("not", dm), kd))
    return E.op("or", E.op("selb", ties, kt, kc), pick(E, stop, h1))


# --- the rows units pass ---------------------------------------------------------------------------------------------------------

#: a round state's words beside the row's M, Z and 1/Z (two halves each) and the round's 16 pivots
STATE_WORDS = ("lo", "hi", "best", "best_mass", "best_minl", "best_nmin", "pivot", "dup", "numdup", "numkeep")
_COUNTS = {"best_nmin", "numdup", "numkeep"}
#: a round state's flags, each a word of 0 or 1: early, done and nomask (two halves each), fin and best_fin (2 R F each)
STATE_FLAGS = 6 + 4 * R * F
RSW = 1 + 2 + 2 + 2 * len(STATE_WORDS) + 2 * F + STATE_FLAGS
CAW = 3 * F
CARRY = 3 + 2 + 5 * (len(K.SPLIT_COUNTS) - 1)


def scw(S: int) -> int:
    """The words of S's row scalars for the select: M, Z, 1/Z, pivot logit, dup, numkeep (two halves each), pre[2 S], and the
    flags stop (two halves) and ties."""
    return 14 + 2 * S


def _w(v, w: int):
    return K.Vec(w, v.k, v.kv, v.d, v.r)


def flag_words(E, bits):
    """Bits as words of 0 or 1: a serving row's elements are whole u16 words, so no unit's port is bits."""
    return _w(E.op("b2i", bits), 32)


def word_flags(E, words):
    return E.op("weq", words, 1)


def pack_state(E, M, Zh, rZ, st, piv, early):
    flags = K.cat([early, st["done"], st["nomask"], st["fin"], st["best_fin"]])
    words = K.cat([_w(v, 32) for v in (M, Zh, rZ, *(st[k] for k in STATE_WORDS), piv)] + [flag_words(E, flags)])
    assert len(words) == RSW
    return words


def unpack_state(E, rw, flags: bool = False):
    """(M, Zh, rZ, st, piv, early) from a round state's words; with ``flags``, st's flags and early too (else early is None)."""
    M, Zh, rZ = rw[[0]], rw[[1, 2]], rw[[3, 4]]
    st, o = {}, 5
    for k in STATE_WORDS:
        st[k] = _w(rw[[o, o + 1]], CW if k in _COUNTS else 32)
        o += 2
    piv, early = rw[np.arange(o, o + 2 * F)], None
    if flags:
        fb = word_flags(E, rw[np.arange(o + 2 * F, RSW)])
        early, st["done"], st["nomask"] = fb[[0, 1]], fb[[2, 3]], fb[[4, 5]]
        st["fin"], st["best_fin"] = fb[np.arange(6, 6 + 2 * R * F)], fb[np.arange(6 + 2 * R * F, 6 + 4 * R * F)]
    return M, Zh, rZ, st, piv, early


def pack_scalars(E, M, Zh, rZ, plog, st, pre, stop, ties):
    return K.cat([_w(v, 32) for v in (M, Zh, rZ, plog, st["dup"], st["numkeep"], pre)] + [flag_words(E, K.cat([stop, ties]))])


def unpack_scalars(E, sw, S: int):
    """(M, Zh, rZ, plog, dup, numkeep, pre, stop, ties) from S's row scalars."""
    fb = word_flags(E, sw[np.arange(11 + 2 * S, 14 + 2 * S)])
    return (sw[[0]], sw[[1, 2]], sw[[3, 4]], sw[[5, 6]], sw[[7, 8]], _w(sw[[9, 10]], CW), _w(sw[np.arange(11, 11 + 2 * S)], CW),
            fb[[0, 1]], fb[[2]])


def pack_ca(ca):
    return K.cat([_w(v, 32) for v in ca])


def unpack_ca(cw):
    return cw[np.arange(F)], cw[np.arange(F, 2 * F)], _w(cw[np.arange(2 * F, 3 * F)], CW)


# --- unit bodies ---------------------------------------------------------------------------------------------------------------

def _x(B, E, logits, temp):
    return temp_scale(E, E.param(B.batch(Bf16ToF32, logits, axes=(0,)), 32), E.param(temp, 32))


def _words(E, coll, w=32):
    return E.param(coll, w)


def _res(E, *vs):
    outs = [E.result(v, v.w if v.w == 1 else 32) for v in vs]
    return outs[0] if len(outs) == 1 else tuple_of(*outs)


def _head_sig(S):
    ps = [("logits", Array(S.N, BF16)), ("temp", F32)]
    if not S.FIRST:
        ps.append(("hw", Array(4, F32)))
    return tuple(ps), Array(4, F32)


def _head_body(B, S, logits, temp, hw=None):
    E = IREngine(B)
    x = _x(B, E, logits, temp)
    bm, bmin, anyb = stats_block(E, x, [S.N])
    carry = None
    if not S.FIRST:
        w = _words(E, hw)
        carry = (w[[0]], w[[2]], word_flags(E, w[[3]]))
    nm, rescale, mn, af = stats_fold(E, bm, bmin, anyb, carry)
    return _res(E, K.cat([nm, rescale, mn, flag_words(E, af)]))


def _exp_sig(S):
    ps = [("logits", Array(S.N, BF16)), ("temp", F32), ("hw", Array(4, F32))]
    if S.AT:
        ps.append(("acc", Array(THREADS, F32)))
    if S.LAST and not S.FIRST:
        ps.append(("es", Array(1, F32)))
    return tuple(ps), Array(1 if S.LAST else THREADS, F32)


def _exp_body(B, S, logits, temp, hw, *rest):
    E = IREngine(B)
    rest = list(rest)
    x = _x(B, E, logits, temp)
    w = _words(E, hw)
    acc = _words(E, rest.pop(0)) if S.AT else None
    acc = exp_part(E, x, w[[0]], [S.N], acc, S.SPT)
    if not S.LAST:
        return _res(E, acc)
    es, rescale = (E.const([0]), E.const([0])) if S.FIRST else (_words(E, rest.pop(0)), w[[1]])
    return _res(E, exp_end(E, acc, PARTS - 1 - S.AT, S.SPT, es, rescale))


def _probs_sig(S):
    ps = [("logits", Array(S.N, BF16)), ("temp", F32)]
    if not S.HEAD:
        return tuple(ps + [("rw", Array(RSW, F32))]), Array(S.N, F32)
    for s in Geo(S.V, S.S).slices:
        ps += [(f"hw{s}", Array(4, F32)), (f"es{s}", Array(1, F32))]
    return tuple(ps), Tuple(Array(S.N, F32), Array(RSW, F32))


def slice_stats(E, geo: Geo, stats: dict):
    """The S slices' (m, es, min) and the row's any-finite flag from the nonempty slices' ``stats[s]`` = (m, es, min, any), an
    empty slice's the stats kernel's initial constants."""
    m, es, mn = [], [], []
    for s in range(geo.S):
        if s in stats:
            m.append(stats[s][0]), es.append(stats[s][1]), mn.append(stats[s][2])
        else:
            m.append(E.const([NEG_INF])), es.append(E.const([0])), mn.append(E.const([INF]))
    anys = K.cat([stats[s][3] for s in geo.slices])
    return K.cat(m), K.cat(es), K.cat(mn), K.reduce(E, "or", anys, [len(anys)])


def _probs_body(B, S, logits, temp, *rest):
    E = IREngine(B)
    geo = Geo(S.V, S.S)
    x = _x(B, E, logits, temp)
    h1 = geo.h1(np.arange(S.N))
    if not S.HEAD:
        M, Zh, rZ, *_ = unpack_state(E, _words(E, rest[0]))
        return _res(E, probs(E, x, M, Zh, rZ, h1))
    stats = {}
    for i, s in enumerate(geo.slices):
        hw, es = rest[2 * i:2 * i + 2]
        w = _words(E, hw)
        stats[s] = (w[[0]], _words(E, es), w[[2]], word_flags(E, w[[3]]))
    M, Zh, rZ, early, st, piv = row_head(E, *slice_stats(E, geo, stats), S.S)
    return _res(E, probs(E, x, M, Zh, rZ, h1), pack_state(E, M, Zh, rZ, st, piv, early))


def _step_sig(S):
    geo = Geo(S.V, S.S)
    ps = [("pr", Array(S.N, F32)), ("rw", Array(RSW, F32))]
    if not S.FIRST:
        ps.append(("ca", Array(CAW, F32)))
    if S.COMBINE < 0:
        return tuple(ps), Array(CAW, F32)
    ps += [(f"ca{s}", Array(CAW, F32)) for s in geo.slices[:-1]] + [("top_p", F32)]
    if S.COMBINE < R - 1:
        return tuple(ps), Tuple(Array(2 * F * S.S, F32), Array(RSW, F32))
    ps += [(f"ph{r}", Array(2 * F * S.S, F32)) for r in range(R - 1)]
    return tuple(ps), Array(scw(S.S), F32)


def gather_parts(E, geo: Geo, cas: dict):
    """(mass, least, count) f-major over the S slices from the nonempty slices' ``cas[s]``, an empty slice's the initial ones."""
    out = []
    for i, init in enumerate((0, ONE_F, 0)):
        per = [cas[s][i] if s in cas else E.const([init] * F, CW if i == 2 else 32) for s in range(geo.S)]
        out.append(K.cat(per)[(np.arange(geo.S)[None, :] * F + np.arange(F)[:, None]).ravel()])
    return out


def _step_body(B, S, pr, rw, *rest):
    E = IREngine(B)
    geo = Geo(S.V, S.S)
    rest = list(rest)
    M, Zh, rZ, st, piv, early = unpack_state(E, _words(E, rw), S.COMBINE >= 0)
    carry = None if S.FIRST else unpack_ca(_words(E, rest.pop(0)))
    ca = step_fold(E, step_block(E, _words(E, pr), piv, Geo.h2(np.arange(S.N), S.S), [S.N]), carry)
    if S.COMBINE < 0:
        return _res(E, pack_ca(ca))
    cas = {s: unpack_ca(_words(E, rest.pop(0))) for s in geo.slices[:-1]}
    cas[geo.slices[-1]] = ca
    p = _words(E, rest.pop(0))
    pm, pl, pn = gather_parts(E, geo, cas)
    st, part = round_combine(E, pm, pl, pn, st, piv, p, Zh, early, S.S, S.COMBINE)
    if S.COMBINE < R - 1:
        return _res(E, K.cat([part[1], _w(part[2], 32)]), pack_state(E, M, Zh, rZ, st, K.ladder(E, st["lo"], st["hi"]), early))
    parts = []
    for r in range(R - 1):
        ph = _words(E, rest.pop(0))
        parts.append((None, ph[np.arange(F * S.S)], _w(ph[np.arange(F * S.S, 2 * F * S.S)], CW)))
    parts.append(part)
    plog, ties, stop, pre = mask_head(E, st, parts, Zh, M, p, early, S.S)
    return _res(E, pack_scalars(E, M, Zh, rZ, plog, st, pre, stop, ties))


def _select_sig(S):
    ps = [("logits", Array(S.N, BF16)), ("temp", F32), ("seed", I64), ("pos", I64), ("splits", I32)]
    ps += [(f"sw{s}", Array(scw(s), F32)) for s in K.SPLIT_COUNTS]
    if not S.FIRST:
        ps.append(("cw", Array(CARRY, F32)))
    return tuple(ps), I32 if S.LAST else Array(CARRY, F32)


def _wrap_bits(E, geo: Geo, n: int, off0):
    """For lanes 0 .. n of a chunk whose lane 0 is at offset ``off0`` of its slice: whether the lane is past the slice's end."""
    return E.op("ile", E.const([geo.L - j for j in range(n + 1)], CW), off0)


def _mod_from(E, base, d, P: int):
    """(base + d) mod P for base < P and d < P, as a count word, and the bits t >= P / 2, t >= P, t >= 3 P / 2 of t = base + d."""
    t = E.op("iadd", base, d)
    ge = [E.op("ile", E.const([c], CW), t) for c in (P // 2, P, 3 * P // 2)]
    return E.op("seli", ge[1], E.op("iadd", t, _neg(E, P)), t), ge


def _neg(E, v: int):
    """-v as an i32 constant (a count word's constants are 20 bits, an i32 add's operand all 32)."""
    return E.const([-v % (1 << 32)], 32)


def positions(E, geo: Geo, n: int, pos0):
    """Each lane j in 0 .. n - 1 of a chunk, and the lane past it, at split count ``geo.S``, from the chunk's first lane's position
    ``pos0`` = (off, pb, ph, sl): its offset in its slice (off), in its mask block (pb), in its h1 period (ph), and its slice.
    A chunk crosses into the next slice at most once (`chunk`), so lane j is past the slice's end iff off + j >= L, and from there
    its offsets are j + off - L; the period and the block wrap by the static offsets of j (and of j - L) modulo them.  Returns
    (h1 bits, reset bits, wrapped bits) for the n lanes and the position of lane n."""
    off, pb, ph, sl = pos0
    j = np.arange(n + 1)
    if geo.S == 1:
        pbn, _ = _mod_from(E, pb, E.const(j % K.MASK_BLOCK, CW), K.MASK_BLOCK)
        reset = E.op("weq", pbn, 0)
        return E.const([0] * n, 1), reset[np.arange(n)], E.const([0] * n, 1), (off, pbn[[n]], ph, sl)
    wr = _wrap_bits(E, geo, n, off)
    cB, cP = -geo.L % K.MASK_BLOCK, -geo.L % geo.P
    dB = E.op("seli", wr, E.const((j + cB) % K.MASK_BLOCK, CW), E.const(j % K.MASK_BLOCK, CW))
    dP = E.op("seli", wr, E.const((j + cP) % geo.P, CW), E.const(j % geo.P, CW))
    pbn, _ = _mod_from(E, pb, dB, K.MASK_BLOCK)
    phn, geP = _mod_from(E, ph, dP, geo.P)
    reset = E.op("weq", pbn, 0)
    h1 = E.op("xor", E.op("xor", geP[0], geP[1]), geP[2])
    offn = E.op("seli", wr[[n]], E.op("iadd", off, _neg(E, geo.L - n)), E.op("iadd", off, E.const([n], CW)))
    sln = E.op("seli", wr[[n]], E.op("iadd", sl, E.const([1], CW)), sl)
    lanes = np.arange(n)
    return h1[lanes], reset[lanes], wr[lanes], (offn, pbn[[n]], phn[[n]], sln)


def seg_scan(E, d, reset, c0):
    """Inclusive prefix sums of the counts ``d`` restarting at each set ``reset``, from the count ``c0`` before lane 0."""
    v, f = K.cat([c0, d]), K.cat([E.const([1], 1), reset])
    n, dd = len(v), 1
    while dd < n:
        i = np.arange(dd, n)
        v, f = K.put(v, i, E.op("seli", f[i], v[i], E.op("iadd", v[i], v[i - dd]))), K.put(f, i, E.op("or", f[i], f[i - dd]))
        dd *= 2
    return v[np.arange(1, n)]


def slice_pre(E, geo: Geo, pre, sl, wrapped, h1):
    """Each lane's ties of the slices before its own (``pre[2 s + h1]``), its slice ``sl`` or, past the wrap, sl + 1."""
    S = geo.S
    oh = E.op("weq", K.cat([sl] * S), E.const(list(range(S)), CW))
    nxt = np.minimum(np.arange(S) + 1, S - 1)
    cur = [K.reduce(E, "xori", E.op("gatei", oh, pre[2 * np.arange(S) + h]), [S]) for h in range(2)]
    after = [K.reduce(E, "xori", E.op("gatei", oh, pre[2 * nxt + h]), [S]) for h in range(2)]
    ph = [E.op("seli", wrapped, after[h], cur[h]) for h in range(2)]
    return E.op("seli", h1, ph[1], ph[0])


def _carry_split(cw):
    """(best, best_i, i) and each S's (off, pb, ph, sl, c) from a select carry's words (S = 1: pb and c)."""
    head = (cw[[0]], _w(cw[[1]], CW), _w(cw[[2]], CW))
    per, o = {}, 3
    for S in K.SPLIT_COUNTS:
        if S == 1:
            per[S] = (None, _w(cw[[o]], CW), None, None, _w(cw[[o + 1]], CW))
            o += 2
        else:
            per[S] = tuple(_w(cw[[o + k]], CW) for k in range(5))
            o += 5
    return head, per


def _select_body(B, S, logits, temp, seed, pos, splits, *rest):
    E = IREngine(B)
    rest, N = list(rest), S.N
    x = _x(B, E, logits, temp)
    tv, sp = E.param(temp, 32), E.param(splits, 32)
    scal = {s: unpack_scalars(E, _words(E, rest[i]), s) for i, s in enumerate(K.SPLIT_COUNTS)}
    zero = E.const([0], CW)
    if S.FIRST:
        head = (None, zero, zero)
        per = {s: (zero, zero, zero, zero, zero) for s in K.SPLIT_COUNTS}
    else:
        head, per = _carry_split(_words(E, rest[len(K.SPLIT_COUNTS)]))
    keep, out_pos = E.const([0] * N, 1), {}
    for s in K.SPLIT_COUNTS:
        geo = Geo(S.V, s)
        off, pb, ph, sl, c0 = per[s]
        h1, reset, wrapped, nxt = positions(E, geo, N, (off, pb, ph, sl))
        M, Zh, rZ, plog, dup, numkeep, pre, stop, ties = scal[s]
        pr = probs(E, x, M, Zh, rZ, h1)
        dm = tie_lanes(E, pr, h1, dup, stop)
        c = seg_scan(E, E.op("b2i", dm), reset, c0)
        lp = E.const([0] * N, CW) if s == 1 else slice_pre(E, geo, pre, sl, wrapped, h1)
        keep_s = keep_lanes(E, x, dm, E.op("iadd", c, lp), h1, plog, numkeep, stop, ties)
        keep = E.op("xor", keep, E.op("and", keep_s, E.op("weq", sp, s)))
        out_pos[s] = (*nxt, c[[N - 1]])
    masked = E.op("sel", keep, x, NEG_INF)
    idx = E.op("iadd", head[2], E.const(list(range(N)), CW))
    key = B.call(GumbelStreamKey, seed, pos)
    g = B.batch(GumbelNoiseLane, key, E.result(idx, CW), axes=(None, 0))
    noisy = E.op("not", E.op("eq", tv, 0))
    y = E.op("sel", noisy, E.param(B.batch(F32AddFtz, E.result(masked, 32), g, axes=(0, 0)), 32), masked)
    if S.FIRST:
        best, bi, start = y[[0]], zero, 1
    else:
        best, bi, start = head[0], head[1], 0
    for j in range(start, N):
        gt = E.op("gt", y[[j]], best)
        bi = E.op("seli", gt, idx[[j]], bi)
        if S.LAST and j == N - 1:
            break
        best = E.op("sel", gt, y[[j]], best)
    if S.LAST:
        return Coll(I32, E.result(_w(bi, 32), 32).refs)
    words = [best, bi, E.op("iadd", head[2], E.const([N], CW))]
    for s in K.SPLIT_COUNTS:
        offn, pbn, phn, sln, cn = out_pos[s]
        words += [pbn, cn] if s == 1 else [offn, pbn, phn, sln, cn]
    return _res(E, K.cat([_w(v, 32) for v in words]))


# --- the Definitions -------------------------------------------------------------------------------------------------------------

def _pair(name: str, statics: tuple[str, ...], sig, body, doc: str):
    """The unit's raw body (an uncatalogued Definition, every operation's lanes) and the registered one, its gates the outputs
    read (`live_replay`)."""
    raw = CompositeDefinition(name, 0, statics, sig, body, doc=doc, conformance=_CONF, register=False)

    def live(B, S, *args):
        return live_replay(B, bind(raw, **{k: getattr(S, k) for k in statics}), args)

    return raw, CompositeDefinition(name, 1, statics, sig, live, doc=doc, conformance=_CONF)


_HeadRaw, TopPStatsHead = _pair("TopPStatsHead", ("N", "FIRST"), _head_sig, _head_body,
                                "A stats block of a slice: its lanes' max, finite min and any-finite flag folded into the slice's "
                                "running (m, rescale, min) words and flag; FIRST: the slice's first block.")
_ExpRaw, TopPStatsExp = _pair("TopPStatsExp", ("N", "SPT", "AT", "LAST", "FIRST"), _exp_sig, _exp_body,
                              "Part AT of a stats block: its lanes' exp(x - m) into the block's 256 thread sums; LAST: then the "
                              "block's sum folded into the slice's exp sum (FIRST: the slice's first block).")
_ProbsRaw, TopPProbs = _pair("TopPProbs", ("V", "S", "N", "HEAD"), _probs_sig, _probs_body,
                             "A step block's probabilities exp(x - M) / Z; HEAD: first the row stats over every slice's, round "
                             "0's state and pivots.")
_StepRaw, TopPStep = _pair("TopPStep", ("V", "S", "N", "FIRST", "COMBINE"), _step_sig, _step_body,
                           "A round's step kernel on a step block, folded into its slice's partials; COMBINE = r: round r's "
                           "combine over every slice's, and after the last round the mask kernel's row scalars.")
_SelectRaw, GumbelTopPSelect = _pair("GumbelTopPSelect", ("V", "N", "FIRST", "LAST"), _select_sig, _select_body,
                                     "N lanes of the row: every split count's keep bits, those `splits` names, the -inf mask, "
                                     "the Gumbel noise and the select, carried chunk to chunk; LAST: the token.")
STAGE_DEFINITIONS = [TopPStatsHead, TopPStatsExp, TopPProbs, TopPStep, GumbelTopPSelect]


# --- the plan ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One unit of the cut: ``key`` names it, ``defn`` its Definition, ``lanes`` the logits it reads, ``wires`` each wired
    parameter's (parameter, writer key, writer's output index); its other parameters are the Call's operands."""
    key: tuple
    defn: SpecializedDefinition
    lanes: tuple[int, int]
    wires: tuple[tuple[str, tuple, int], ...] = field(default=())


@cache
def plan(V: int) -> tuple[Unit, ...]:
    """The units of a row of V in an order that writes each before its readers."""
    out: list[Unit] = []
    for S in K.SPLIT_COUNTS:
        out += _pipeline(Geo(V, S))
    N = chunk(V)
    starts = list(range(0, V, N))
    for k, lo in enumerate(starts):
        hi = min(lo + N, V)
        first, last = k == 0, k == len(starts) - 1
        w = []
        w += [(f"sw{S}", scalars_unit(V, S), 0) for S in K.SPLIT_COUNTS]
        if not first:
            w.append(("cw", ("select", k - 1), 0))
        out.append(Unit(("select", k), bind(GumbelTopPSelect, V=V, N=hi - lo, FIRST=first, LAST=last), (lo, hi), tuple(w)))
    return tuple(out)


def _pipeline(geo: Geo) -> list[Unit]:
    S, out, last_head, last_exp = geo.S, [], {}, {}
    for s in geo.slices:
        for t, (lo, n) in enumerate(geo.blocks(s, K.STATS_BLOCK)):
            hk = ("head", S, s, t)
            out.append(Unit(hk, bind(TopPStatsHead, N=n, FIRST=t == 0), (lo, lo + n),
                            () if t == 0 else (("hw", ("head", S, s, t - 1), 0),)))
            parts = -(-n // STATS_PART)
            for a in range(parts):
                plo, pn = lo + a * STATS_PART, min(STATS_PART, n - a * STATS_PART)
                last = a == parts - 1
                w = [("hw", hk, 0)]
                if a:
                    w.append(("acc", ("exp", S, s, t, a - 1), 0))
                if last and t:
                    w.append(("es", ("exp", S, s, t - 1, last_exp[(s, t - 1)]), 0))
                out.append(Unit(("exp", S, s, t, a), bind(TopPStatsExp, N=pn, SPT=geo.spt, AT=a, LAST=last, FIRST=last and t == 0),
                                (plo, plo + pn), tuple(w)))
            last_head[s], last_exp[(s, t)] = t, parts - 1
    steps = [(s, t, lo, n) for s in geo.slices for t, (lo, n) in enumerate(geo.blocks(s, K.STEP_BLOCK))]
    for b, (s, t, lo, n) in enumerate(steps):
        if b == 0:
            w = []
            for s2 in geo.slices:
                t2 = last_head[s2]
                w += [(f"hw{s2}", ("head", S, s2, t2), 0), (f"es{s2}", ("exp", S, s2, t2, last_exp[(s2, t2)]), 0)]
        else:
            w = [("rw", ("probs", S, 0), 1)]
        out.append(Unit(("probs", S, b), bind(TopPProbs, V=geo.V, S=S, N=n, HEAD=b == 0), (lo, lo + n), tuple(w)))
    last_step = {s: max(b for b, st in enumerate(steps) if st[0] == s) for s in geo.slices}
    for r in range(R):
        rs = ("probs", S, 0) if r == 0 else ("step", S, r - 1, len(steps) - 1)
        for b, (s, t, lo, n) in enumerate(steps):
            comb = r if b == len(steps) - 1 else -1
            w = [("pr", ("probs", S, b), 0), ("rw", rs, 1)]
            if t:
                w.append(("ca", ("step", S, r, b - 1), 0))
            if comb >= 0:
                w += [(f"ca{s2}", ("step", S, r, last_step[s2]), 0) for s2 in geo.slices[:-1]]
                if r == R - 1:
                    w += [(f"ph{r2}", ("step", S, r2, len(steps) - 1), 0) for r2 in range(R - 1)]
            out.append(Unit(("step", S, r, b), bind(TopPStep, V=geo.V, S=S, N=n, FIRST=t == 0, COMBINE=comb), (lo, lo + n),
                            tuple(w)))
    return out


def scalars_unit(V: int, S: int) -> tuple:
    """The key of the unit that writes S's row scalars: the last round's combining step."""
    geo = Geo(V, S)
    nb = sum(len(geo.blocks(s, K.STEP_BLOCK)) for s in geo.slices)
    return ("step", S, R - 1, nb - 1)


def cut(t) -> list[tuple[SpecializedDefinition, int]]:
    """The units a binding of `GumbelTopPTokenSelect_v1{V}` (or its shared-greedy form) is cut into, each Definition bound and how
    many of it (`sampling.CUTS`)."""
    from collections import Counter

    n = Counter(u.defn for u in plan(int(t.bindings["V"])))
    return list(n.items())


__all__ = ["Geo", "chunk", "plan", "cut", "Unit", "scalars_unit", "STAGE_DEFINITIONS", "TopPStatsHead", "TopPStatsExp",
           "TopPProbs", "TopPStep", "GumbelTopPSelect", "SELECT_N", "STATS_PART"]
