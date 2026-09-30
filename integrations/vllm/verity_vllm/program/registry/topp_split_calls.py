"""The split top-p select as Calls a word check cuts one by one: `GumbelTopPTokenSelect_v3{V,S}`.

`GumbelTopPTokenSelect_v2{V,S}` is one Call whose body is not separable (every stage reads the one before), so `Q_word_v1` cuts it as
one flat graph: about 800 gates per lane, over `word.MAX_GATES` from V = 27,996 at S = 32 (`V2_MAX_V`).  v3 is the same function
restated over the split kernels' own structure, each piece a Definition a Program calls at its root:

    TemperatureScale_v1{V}                                  the row / temperature
    TopPSplitStats_v1{N,SPT}             per slice          `_topp_sb_stats_kernel`: running max, rescaled exp sum, min, any finite
    TopPRowStats_v1{S}                   merge              `_topp_sb_row_stats`: M, the two warp halves' Z, early returns, round 0's pivots
    TopPSplitProbs_v1{N,SPT,S}           per slice          exp(x - M) / Z of the lane's half
    TopPSplitStep_v1{N,S}      x R       per slice          `_topp_sb_step_kernel`: per pivot mass, min, count at the min (0 once stopped)
    TopPCombine_v1{S,ROUND}    x R       merge              `topp_split.combine`: the state, the next round's pivots and stop
    TopPMaskSetup_v1{S}                  merge              the mask kernel's per-half scalars and the ties before each slice
    TopPSplitMask_v1{N,SPT,S}            per slice          `_topp_sb_mask_kernel`'s keep bit, the lane or -inf
    GumbelTopPSplitSelect_v1{V}                             `_gumbel_sample_kernel` over the masked row: the sampled id

Each piece runs `topp_keep_builder`'s own stage functions in an `IREngine`; a slice's stats and step are those functions at S = 1 on
the slice (no reduction there crosses a slice), so every Value is the one v2 computes.  The values between pieces are Call outputs,
committed under the query of record.  `GumbelTopPTokenSelect_v3{V,S}` is the chain as one composite with v2's six operands (the
equivalence test evaluates both); a Program calls its pieces (`chain`).
"""
from __future__ import annotations

import numpy as np

from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, coll_concat, tuple_of
from verity.ir.types import Array, Tuple
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import topp_keep_builder as K
from verity_vllm.program.registry.prims import BF16, BIT, F32, I32
from verity_vllm.program.registry.topp_word_gates import CW
from verity_vllm.program.registry.topp_words import I64, IREngine

F, R = K.F, K.R
P = R * F

#: the largest V whose `GumbelTopPTokenSelect_v2{V,S}` is within `word.MAX_GATES`, per split count S (measured: 23,999,904 gates at
#: V = 27,995, 24,000,658 at 27,996, S = 32); a Program binds v3 above it and v2 at or below, and v2 at a split count not listed
V2_MAX_V = {32: 27995}

# the state `topp_split.combine` carries between rounds, packed by width: F32 words, count words, bits (two halves each).  Its `fin`
# bits are not carried: a round only writes its own slots, so each combine returns those and the mask setup reads all of them.
_F32_KEYS = ("lo", "hi", "best", "best_mass", "best_minl", "pivot", "dup")
_CW_KEYS = ("best_nmin", "numdup", "numkeep")
_BIT_KEYS = ("done", "nomask")
STATE = Tuple(Array(2 * len(_F32_KEYS), F32), Array(2 * len(_CW_KEYS), I32), Array(2 * len(_BIT_KEYS) + 2 * P, BIT))
FIN = Array(2 * F, BIT)


def uses_v3(V: int, S: int) -> bool:
    return S in V2_MAX_V and V > V2_MAX_V[S]


def bounds(V: int, S: int) -> list[tuple[int, int]]:
    """The S slices' [start, end) (`topp_split`'s ceil(V / S) lanes each)."""
    L = -(-V // S)
    return [(min(s * L, V), min(s * L + L, V)) for s in range(S)]


def spt(V: int, S: int) -> int:
    return K._ts().spt_for(V, S)


def _h1(N, SPT, S):
    return K._ts().z_group((np.arange(N) // SPT) % 32, S)


def _h2(N, S):
    return K._ts().z_group(np.arange(N) % 32, S)


def _pack(E, st) -> Coll:
    bits = K.cat([st[k] for k in _BIT_KEYS] + [st["best_fin"]])
    return tuple_of(E.result(K.cat([st[k] for k in _F32_KEYS]), 32), E.result(K.cat([st[k] for k in _CW_KEYS]), CW), E.result(bits, 1))


def _unpack(E, state: Coll) -> dict:
    f, c, b = E.param(state[0], 32), E.param(state[1], CW), E.param(state[2], 1)
    st = {k: f[2 * i:2 * i + 2] for i, k in enumerate(_F32_KEYS)}
    st.update({k: c[2 * i:2 * i + 2] for i, k in enumerate(_CW_KEYS)})
    st.update({k: b[2 * i:2 * i + 2] for i, k in enumerate(_BIT_KEYS)})
    st["best_fin"] = b[2 * len(_BIT_KEYS):]
    return st


def _field(state: Coll, key: str) -> Coll:
    """A state field's two words, as references into the state (a Program's operand, no gate)."""
    port, i = (0, _F32_KEYS.index(key)) if key in _F32_KEYS else (1, _CW_KEYS.index(key))
    return state[port][2 * i:2 * i + 2]


def _fin_slots(r: int) -> np.ndarray:
    """Round r's slots of the `fin` bits (h * R * F + r * F + f), h-major."""
    return (np.arange(2)[:, None] * P + r * F + np.arange(F)[None, :]).ravel()


def _stop(E, early, st):
    """A round's step kernel returns zeros once either half returned early or is done."""
    return E.op("or", E.op("or", early[[0]], early[[1]]), E.op("or", st["done"][[0]], st["done"][[1]]))


@composite("TopPSplitStats", 1, ["N", "SPT"], lambda st: ((("x", Array(st.N, F32)),), Tuple(F32, F32, F32, BIT)),
           conformance="exact-model-tested")
def TopPSplitStats(B, st, x):
    """One slice's stats kernel: the running max, the rescaled exp sum, the minimum of the lanes > -inf, and whether any is."""
    E = IREngine(B)
    m, es, mn, anyf = K.stats_kernel(E, E.param(x, 32), 1, int(st.SPT), [(0, int(st.N))])
    return tuple_of(*(E.result(v, 32)[0] for v in (m, es, mn)), E.result(anyf, 1)[0])


@composite("TopPRowStats", 1, ["S"],
           lambda st: ((("m", Array(st.S, F32)), ("es", Array(st.S, F32)), ("mn", Array(st.S, F32)), ("anyf", Array(st.S, BIT))),
                       Tuple(F32, Array(2, F32), Array(2, F32), Array(2, BIT), STATE, Array(2 * F, F32), BIT)),
           conformance="exact-model-tested")
def TopPRowStats(B, st, m, es, mn, anyf):
    """The row stats over the S slices' (M, the halves' Z and 1/Z, early returns), the initial state, round 0's pivots and stop."""
    E, S = IREngine(B), int(st.S)
    anyv = K.reduce(E, "or", E.param(anyf, 1), [S])
    M, Zh, rZ, maxp, minp, early = K.row_stats(E, E.param(m, 32), E.param(es, 32), E.param(mn, 32), anyv, S)
    st0 = K._state(E, minp, maxp)
    piv = K.ladder(E, st0["lo"], st0["hi"])
    return tuple_of(E.result(M, 32)[0], E.result(Zh, 32), E.result(rZ, 32), E.result(early, 1), _pack(E, st0), E.result(piv, 32),
                    E.result(_stop(E, early, st0), 1)[0])


@composite("TopPSplitProbs", 1, ["N", "SPT", "S"],
           lambda st: ((("x", Array(st.N, F32)), ("M", F32), ("Zh", Array(2, F32)), ("rZ", Array(2, F32))), Array(st.N, F32)),
           conformance="exact-model-tested")
def TopPSplitProbs(B, st, x, M, Zh, rZ):
    """One slice's probabilities exp(x - M) / Z, Z the lane's warp half's."""
    E, h1 = IREngine(B), _h1(int(st.N), int(st.SPT), int(st.S))
    Zv, rv = E.param(Zh, 32), E.param(rZ, 32)
    return E.result(E.op("div", E.op("exp", E.op("sub", E.param(x, 32), E.param(M, 32))), Zv[h1], rv[h1]), 32)


@composite("TopPSplitStep", 1, ["N", "S"],
           lambda st: ((("pr", Array(st.N, F32)), ("piv", Array(2 * F, F32)), ("stop", BIT)),
                       Tuple(Array(F, F32), Array(F, F32), Array(F, I32))),
           conformance="exact-model-tested")
def TopPSplitStep(B, st, pr, piv, stop):
    """One slice's step kernel for one round: per pivot the mass above it, the minimum above it and the count there (zeros once
    ``stop``)."""
    E, N = IREngine(B), int(st.N)
    sv = E.param(stop, 1)
    pm, pl, pn = K.step_kernel(E, E.param(pr, 32), E.param(piv, 32), _h2(N, int(st.S)), K._blocks([(0, N)], K.STEP_BLOCK), 1)
    return tuple_of(E.result(E.op("sel", sv, 0, pm), 32), E.result(E.op("sel", sv, 0, pl), 32), E.result(E.op("seli", sv, 0, pn), CW))


def _combine_ret(st):
    return Tuple(STATE, FIN) if int(st.ROUND) == R - 1 else Tuple(STATE, FIN, Array(2 * F, F32), BIT)


@composite("TopPCombine", 1, ["S", "ROUND"],
           lambda st: ((("mass", Array(F * st.S, F32)), ("minl", Array(F * st.S, F32)), ("nmin", Array(F * st.S, I32)), ("state", STATE),
                        ("piv", Array(2 * F, F32)), ("top_p", F32), ("Zh", Array(2, F32)), ("early", Array(2, BIT))), _combine_ret(st)),
           conformance="exact-model-tested")
def TopPCombine(B, st, mass, minl, nmin, state, piv, top_p, Zh, early):
    """Round ROUND's combine of the S slices' partials (f-major, f * S + s) in both halves: the state and the round's `fin` bits;
    before the last round, the next round's pivots and stop too."""
    E, S, r = IREngine(B), int(st.S), int(st.ROUND)
    part = [E.param(mass, 32), E.param(minl, 32), E.param(nmin, CW)]
    old = dict(_unpack(E, state), fin=E.const([0] * 2 * P, 1))
    new = K.combine(E, part, old, E.param(piv, 32), E.param(top_p, 32), E.param(Zh, 32), S, r)
    head = (_pack(E, new), E.result(new["fin"][_fin_slots(r)], 1))
    if r == R - 1:
        return tuple_of(*head)
    return tuple_of(*head, E.result(K.ladder(E, new["lo"], new["hi"]), 32), E.result(_stop(E, E.param(early, 1), new), 1)[0])


@composite("TopPMaskSetup", 1, ["S"],
           lambda st: ((("state", STATE), ("fin", Array(2 * P, BIT)), ("minl", Array(P * st.S, F32)), ("nmin", Array(P * st.S, I32)),
                        ("Zh", Array(2, F32)), ("M", F32), ("early", Array(2, BIT)), ("top_p", F32), ("splits", I32)),
                       Tuple(Array(2, F32), Array(2, BIT), BIT, Array(2 * st.S, I32), BIT)),
           conformance="exact-model-tested")
def TopPMaskSetup(B, st, state, fin, minl, nmin, Zh, M, early, top_p, splits):
    """The mask kernel's scalars per warp half (the pivot logit, whether it stops), whether ties are resolved, the ties at dup in
    the slices before each slice (``pre[2 s + half]``), and whether ``splits`` is S (else no lane is kept)."""
    E, S = IREngine(B), int(st.S)
    s = dict(_unpack(E, state), fin=E.param(fin, 1))
    stop = E.op("or", E.op("ge", E.param(top_p, 32), K.ONE_F), E.op("or", E.param(early, 1), s["nomask"]))
    plog = E.op("add", E.op("add", E.op("log", s["pivot"]), E.op("log", E.param(Zh, 32))), E.param(M, 32))
    tie_h = E.op("ilt", s["numkeep"], s["numdup"])
    ties = E.op("selb", stop[[0]], tie_h[[1]], tie_h[[0]])
    fin_any = K.reduce(E, "or", s["fin"], [P, P])
    foh = E.op("selb", fin_any[np.repeat(np.arange(2), P)], s["fin"], s["best_fin"])
    hh, ss, pp = (a.ravel() for a in np.meshgrid(np.arange(2), np.arange(S), np.arange(P), indexing="ij"))
    g = foh[hh * P + pp]
    minl_sel = K.reduce(E, "xorf", E.op("gate", g, E.param(minl, 32)[pp * S + ss]), [P] * (2 * S))
    nmin_sel = K.reduce(E, "xori", E.op("gatei", g, E.param(nmin, CW)[pp * S + ss]), [P] * (2 * S))
    dupc = E.op("gatei", E.op("near", minl_sel, s["dup"][np.repeat(np.arange(2), S)]), nmin_sel)
    pre = [E.const([0, 0], CW)]
    for i in range(1, S):
        pre.append(E.op("iadd", pre[-1], dupc[[i - 1, S + i - 1]]))
    ok = E.op("weq", E.param(splits, 32), S)
    return tuple_of(E.result(plog, 32), E.result(stop, 1), E.result(ties, 1)[0], E.result(K.cat(pre), CW), E.result(ok, 1)[0])


@composite("TopPSplitMask", 1, ["N", "SPT", "S"],
           lambda st: ((("x", Array(st.N, F32)), ("pr", Array(st.N, F32)), ("plog", Array(2, F32)), ("dup", Array(2, F32)),
                        ("numkeep", Array(2, I32)), ("stop", Array(2, BIT)), ("ties", BIT), ("pre", Array(2, I32)), ("ok", BIT)),
                       Array(st.N, F32)),
           conformance="exact-model-tested")
def TopPSplitMask(B, st, x, pr, plog, dup, numkeep, stop, ties, pre, ok):
    """One slice's mask kernel: each lane kept (its logit) or -inf, by its warp half's pivot logit and, among the ties at dup, the
    first numkeep of the row (``pre``: the ties of the slices before, per half)."""
    E, N = IREngine(B), int(st.N)
    h1 = _h1(N, int(st.SPT), int(st.S))
    xv, sv = E.param(x, 32), E.param(stop, 1)
    kc = E.op("gt", xv, E.param(plog, 32)[h1])
    dm = E.op("and", E.op("eq", E.param(pr, 32), E.param(dup, 32)[h1]), E.op("not", sv[h1]))
    c = K.scan(E, E.op("b2i", dm), [n for _, n in K._blocks([(0, N)], K.MASK_BLOCK)[0]])
    cum = E.op("iadd", c, E.param(pre, CW)[h1])
    kd = E.op("and", dm, E.op("ile", cum, E.param(numkeep, CW)[h1]))
    kt = E.op("and", kc, E.op("or", E.op("not", dm), kd))
    keep = E.op("and", E.op("or", E.op("selb", E.param(ties, 1), kt, kc), sv[h1]), E.param(ok, 1))
    return E.result(E.op("sel", keep, xv, K.NEG_INF), 32)


@composite("GumbelTopPSplitSelect", 1, ["V"],
           lambda st: ((("masked", Array(st.V, F32)), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
           conformance="exact-model-tested")
def GumbelTopPSplitSelect(B, st, masked, seed, pos, temp):
    """The Gumbel-max select over the masked row: the split top-p chain's sampling event."""
    return B.call(bind(SM.GumbelSelectF32, V=st.V), masked, seed, pos, temp)


def _fmajor(steps, port, t):
    return coll_concat(Array(F * len(steps), t), [steps[s][port][f] for f in range(F) for s in range(len(steps))])


def _n(V: int, S: int, SLICE: int) -> int:
    s0, e = bounds(V, S)[SLICE]
    return e - s0


#: the pieces by the role a vocabulary names them under, bound to the select's (V, S) and a slice's or round's index
KINDS = {
    "topp_split_temperature": lambda V: bind(SM.TemperatureScale, V=V),
    "topp_split_stats": lambda V, S, SLICE=0: bind(TopPSplitStats, N=_n(V, S, SLICE), SPT=spt(V, S)),
    "topp_row_stats": lambda S: bind(TopPRowStats, S=S),
    "topp_split_probs": lambda V, S, SLICE=0: bind(TopPSplitProbs, N=_n(V, S, SLICE), SPT=spt(V, S), S=S),
    "topp_split_step": lambda V, S, SLICE=0: bind(TopPSplitStep, N=_n(V, S, SLICE), S=S),
    "topp_combine": lambda S, RND=0: bind(TopPCombine, S=S, ROUND=RND),
    "topp_mask_setup": lambda S: bind(TopPMaskSetup, S=S),
    "topp_split_mask": lambda V, S, SLICE=0: bind(TopPSplitMask, N=_n(V, S, SLICE), SPT=spt(V, S), S=S),
    "topp_split_select": lambda V: bind(GumbelTopPSplitSelect, V=V),
}


def chain(call, V: int, S: int, logits: Coll, top_p: Coll, seed: Coll, pos: Coll, temp: Coll, splits: Coll, kind=None) -> Coll:
    """`GumbelTopPTokenSelect_v2{V,S}` on the six operands as the pieces' Calls: ``kind(role, **statics)`` each piece's Definition
    (default `KINDS`; a rule passes its vocabulary's), ``call(definition, *operands) -> Coll`` its Call (a composite's `B.call`, a
    frontend rule's `ctx.call`)."""
    kind = kind or (lambda role, **st: KINDS[role](**st))
    sl = bounds(V, S)
    if any(e == s0 for s0, e in sl):
        raise ValueError(f"V = {V} leaves a slice of S = {S} empty")
    scaled = call(kind("topp_split_temperature", V=V), logits, temp)
    xs = [scaled[s0:e] for s0, e in sl]
    stats = [call(kind("topp_split_stats", V=V, S=S, SLICE=i), x) for i, x in enumerate(xs)]
    per = [coll_concat(Array(S, t), [o[i] for o in stats]) for i, t in enumerate((F32, F32, F32, BIT))]
    rs = call(kind("topp_row_stats", S=S), *per)
    M, Zh, rZ, early, state, piv, stop = (rs[i] for i in range(7))
    prs = [call(kind("topp_split_probs", V=V, S=S, SLICE=i), x, M, Zh, rZ) for i, x in enumerate(xs)]
    minls, nmins = [], []
    fin: list = [None] * (2 * P)
    for r in range(R):
        steps = [call(kind("topp_split_step", V=V, S=S, SLICE=i), pr, piv, stop) for i, pr in enumerate(prs)]
        mass, minl, nmin = _fmajor(steps, 0, F32), _fmajor(steps, 1, F32), _fmajor(steps, 2, I32)
        out = call(kind("topp_combine", S=S, RND=r), mass, minl, nmin, state, piv, top_p, Zh, early)
        state = out[0]
        for j, slot in enumerate(_fin_slots(r).tolist()):
            fin[slot] = out[1][j]
        if r < R - 1:
            piv, stop = out[2], out[3]
        minls.append(minl)
        nmins.append(nmin)
    setup = call(kind("topp_mask_setup", S=S), state, coll_concat(Array(2 * P, BIT), fin), coll_concat(Array(P * S, F32), minls),
                 coll_concat(Array(P * S, I32), nmins), Zh, M, early, top_p, splits)
    plog, stop, ties, pre, ok = (setup[i] for i in range(5))
    dup, numkeep = _field(state, "dup"), _field(state, "numkeep")
    masked = [call(kind("topp_split_mask", V=V, S=S, SLICE=i), x, pr, plog, dup, numkeep, stop, ties, pre[2 * i:2 * i + 2], ok)
              for i, (x, pr) in enumerate(zip(xs, prs))]
    return call(kind("topp_split_select", V=V), coll_concat(Array(V, F32), masked), seed, pos, temp)


@composite("GumbelTopPTokenSelect", 3, ["V", "S"],
           lambda st: ((("logits", Array(st.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectV3(B, st, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelect_v2{V,S}` as the split top-p chain (`chain`): the same six operands, the same token."""
    return chain(B.call, int(st.V), int(st.S), logits, top_p, seed, pos, temp, splits)


__all__ = ["V2_MAX_V", "STATE", "FIN", "KINDS", "uses_v3", "bounds", "spt", "chain", "TopPSplitStats", "TopPRowStats", "TopPSplitProbs", "TopPSplitStep",
           "TopPCombine", "TopPMaskSetup", "TopPSplitMask", "GumbelTopPSplitSelect", "GumbelTopPTokenSelectV3"]
