"""The public inputs of V[B] that the outer verifier derives from the inner coins (and B) alone.

Each inner check is affine in the inner messages once the coins are fixed. This module ports the Lean verifier's
coin-dependent computations (`Flock.bindAndZerocheck`, `lincheck`, `ringSwitch`, `ligerito`, `Stmt.fold`,
`Stmt.regionPoint` / `regionValue`) and turns each affine map into its coefficient vector, by evaluating the port on unit
vectors where the map is a state machine (the sumchecks, Ligerito's running claim). Nothing here reads a message, a query
answer, C or the data: `tables(frame, statement_public_parts, coins)` is what the outer verifier computes before it checks
the outer proof (Project store `docs/circuit-privacy.md` I.3, "coins are inputs of V[B]").
"""
from __future__ import annotations

import numpy as np

from . import gf
from .session import Layout, Session, fast100, named_coins
from .statement import PT_LOCAL, Statement
from .vb import Frame, R

K1 = (1, 0)


def _k(x):
    return (x, 0) if isinstance(x, int) else x


def _karr(xs) -> np.ndarray:
    out = np.zeros((len(xs), 2, 2), dtype=np.uint64)
    for i, (a, b) in enumerate(xs):
        out[i, 0] = gf.to_arr([a])[0]
        out[i, 1] = gf.to_arr([b])[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# §9–§10: zerocheck and lincheck
# ---------------------------------------------------------------------------------------------------------------------

def zerocheck_tables(m: int, co: dict) -> dict:
    r = gf.PINNED + co[("zc_rout",)]
    z = co[("zc_z",)][0]
    rho = [co[("zc_rho", i)][0] for i in range(m - 6)]
    wl = gf.lagrange(gf.NODES_L, gf.DEN6_INV, z)
    if z in gf.NODES_S:
        wcomb = [0] * 64
    elif z in gf.NODES_L:
        i = gf.NODES_L.index(z)
        wcomb = [1 if j == i else 0 for j in range(64)]
    else:
        zall = 1
        for nd in gf.NODES_S + gf.NODES_L:
            zall = gf.mul(zall, z ^ nd)
        cc = gf.mul(zall, gf.DEN7_INV)
        wcomb = [gf.mul(cc, gf.inv(z ^ gf.NODES_L[i])) for i in range(64)]
    A, Bc, C = [], [], []
    for i in range(m - 6):
        iv = gf.inv(1 ^ r[i])
        A.append(gf.mul(iv, 1 ^ rho[i]))
        Bc.append(gf.mul(gf.mul(r[i], iv), 1 ^ rho[i]) ^ rho[i])
        C.append(gf.mul(rho[i], 1 ^ rho[i]))
    suf = [1] * (m - 6)
    acc = 1
    for i in reversed(range(m - 6)):
        suf[i] = acc
        acc = gf.mul(acc, A[i])
    full = acc
    return dict(kab=[gf.mul(w, full) for w in wcomb], kc=[gf.mul(w ^ l, full) for w, l in zip(wcomb, wl)],
                kg1=[gf.mul(b, s) for b, s in zip(Bc, suf)], kginf=[gf.mul(c, s) for c, s in zip(C, suf)], wL=wl,
                z=z, rho=rho, r=r)


def lincheck_tables(k_log: int, co: dict) -> dict:
    alpha, beta = co[("lc_alpha",)][0], co[("lc_beta",)][0]
    ts = [co[("lc_t", j)][0] for j in range(k_log - 6)]
    suf, acc = [1] * len(ts), 1
    for j in reversed(range(len(ts))):
        suf[j] = acc
        acc = gf.mul(acc, 1 ^ ts[j])
    return dict(alpha=alpha, beta=beta, t=ts, kfa=gf.mul(alpha, acc), kfb=acc, kconst=gf.mul(beta, acc), kq1=suf,
                kqinf=[gf.mul(t ^ gf.mul(t, t), s) for t, s in zip(ts, suf)], rp=co[("lc_rp",)][0],
                wrp=gf.skip_weights(co[("lc_rp",)][0]))


def fold_tables(fr: Frame, st: Statement, zc: dict, lc: dict) -> dict:
    """The fold's public part and the coefficients of C's part (see `vb._private_fold`)."""
    B = fr.B
    alpha, beta = lc["alpha"], lc["beta"]
    lw = gf.skip_weights(zc["z"])
    rho_in = zc["rho"][:B.k_log - 6]
    r_in = list(reversed(lc["t"]))
    eqT = gf.to_arr(gf.eq_table(r_in))                                          # (2^(k_log-6),)
    comb = st.public_fold(alpha, lw, rho_in)
    comb[fr.pin] ^= gf.to_arr([beta])[0]
    ypub = np.bitwise_xor.reduce(gf.vmul(comb.reshape(-1, 64, 2), eqT[:, None, :]), axis=0)
    lo = B.unit_log - 6
    e_loc = st._tensor(lw, rho_in[:lo])
    e0 = np.concatenate([e_loc, np.zeros((1, 2), np.uint64)])
    erho_hi = gf.eq_table(rho_in[lo:])
    eqT_lo = gf.eq_table(r_in[:lo])
    eqT_hi = gf.eq_table(r_in[lo:])
    units = [fr.unit_pos0 + k for k in range(B.units)]
    su = 0
    for q in units:
        su ^= gf.mul(erho_hi[q], eqT_hi[q])
    s_erho = 0
    for q in units:
        s_erho ^= erho_hi[q]
    eqT_int = gf.from_arr(eqT)
    kcp = gf.mul(gf.mul(alpha ^ 1, eqT_int[fr.pin >> 6]), s_erho)
    src = fr.leaf_src.reshape(-1)
    eqTmsg = [eqT_int[c >> 6] for c in src]
    e_loc_i = gf.from_arr(e_loc)
    kleaf = []
    for g in range(B.g):
        for u in range(B.upv):
            eh = erho_hi[fr.unit_pos0 + g * B.upv + u]
            for j in range(B.leaves_per_unit_in):
                for t in range(16):
                    kleaf.append(gf.mul(gf.mul(alpha ^ 1, e_loc_i[16 * j + t]), eh))
    return dict(ypub=ypub, e0=e0, eqTlo=eqT_lo, su=su, kcp=kcp, eqTmsg=eqTmsg, kleaf=kleaf, alpha=alpha)


# ---------------------------------------------------------------------------------------------------------------------
# §11–§12, §16.2: claims, ring switch, regions
# ---------------------------------------------------------------------------------------------------------------------

def claim_points(fr: Frame, st: Statement, zc: dict, lc: dict, points: list[list[int]]) -> list[tuple[list[int], list[int]]]:
    """(skip weights, x) per claim: ab, c, then the regions (region-major, point 0 before point 1)."""
    kl, m = fr.B.k_log, fr.B.m
    r_in = list(reversed(lc["t"]))
    out = [(gf.skip_weights(lc["rp"]), r_in + zc["rho"][kl - 6:m - 6]), (gf.skip_weights(zc["z"]), list(zc["r"]))]
    for reg in fr.regions:
        for pt in points:
            pnt = st.region_point(reg, pt)
            out.append((gf.eq_table(pnt[:6]), pnt[6:]))
    return out


def ring_tables(claims, co: dict) -> dict:
    n = len(claims)
    gam = co[("rs_gamma",)]
    w, kT, sws = [], [], []
    for k, (skip, x) in enumerate(claims):
        x0 = x[0]
        w.append([gf.mul(skip[v % 64], x0 if v >= 64 else 1 ^ x0) for v in range(128)])
        rpp = co[("rs_rpp", k)]
        kT.append([gf.mul(gam[k], e) for e in gf.eq_table(rpp)])
        sws.append((gam[k], rpp, x[1:]))
    assert len(gam) == n
    return dict(w=w, kT=kT, sws=sws)


def region_tables(fr: Frame, st: Statement, points: list[list[int]]) -> dict:
    B = fr.B
    nb = 1 << B.nbl
    const = []
    out: dict = {}
    for ri, reg in enumerate(fr.regions):
        for pt, rv in enumerate(points):
            nf = len(reg.free)
            wt = gf.eq_table(rv[:nf])
            eb = gf.eq_table(rv[PT_LOCAL:PT_LOCAL + B.nbl])
            if reg.kind == "const":
                const.append(st.region_value(reg, rv, None, st.pub))
                continue
            acc = 0
            if reg.kind == "digest":
                coef = np.zeros((B.n, 256), dtype=object)
                rng = st.circ.ranges[reg.range_index]
                for b in range(nb):
                    for s_pos, x in enumerate(rng.slots):
                        if x is None:
                            pubbytes = st.dummy_run_cv(reg.port)
                        else:
                            i = st.inst_of(b, x[0])
                            if i is not None:
                                for bit in range(256):
                                    coef[i, bit] = gf.mul(eb[b], wt[256 * s_pos + bit])
                                continue
                            pubbytes = st.dummy_digests[reg.port]
                        for bi, byte in enumerate(pubbytes):
                            for t in range(8):
                                if (byte >> t) & 1:
                                    acc ^= gf.mul(eb[b], wt[256 * s_pos + 8 * bi + t])
                out[f"reg.dcoef.{ri}.{pt}"] = coef
            else:
                coef = []
                for b in range(nb):
                    for g in range(B.g):
                        for u in range(B.upv):
                            k = g * B.upv + u
                            for j in range(B.leaves_per_unit_out):
                                coef.append([gf.mul(eb[b], wt[128 * k + 16 * j + t]) for t in range(16)])
                out[f"reg.ocoef.{pt}"] = coef
            const.append(acc)
    out["reg.const"] = const
    return out


# ---------------------------------------------------------------------------------------------------------------------
# §12.4, §13: the basis, Ligerito's running claim and final check
# ---------------------------------------------------------------------------------------------------------------------

def _mul_columns(g: int) -> np.ndarray:
    """M[l][u] = bit l of g·x^u."""
    cols = []
    v = g
    for _ in range(128):
        cols.append(v)
        v = gf.mul(v, 2)
    a = gf.to_arr(cols)
    return np.unpackbits(a.view(np.uint8), axis=1, bitorder="little").T.astype(np.float32)


def _apply(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """S' = M S for S (128, 2, 2) K elements."""
    bits = np.unpackbits(np.ascontiguousarray(S).view(np.uint8).reshape(128, -1), axis=1, bitorder="little").astype(np.float32)
    out = (M @ bits).astype(np.int64) & 1
    return np.packbits(out.astype(np.uint8), axis=1, bitorder="little").view(np.uint64).reshape(128, 2, 2)


def basis_for_claim(sw, q: list, e: int) -> list:
    gamma, rpp, z = sw
    eqR = gf.to_arr(gf.eq_table(rpp))
    S = np.zeros((128, 2, 2), dtype=np.uint64)
    S[0, 0, 0] = 1
    for j, qj in enumerate(q):
        Mm = _apply(_mul_columns(1 ^ z[j]), S)
        S = Mm ^ gf.vkmul(np.broadcast_to(_karr([qj])[0], S.shape), S)
    pre = S
    ones = [_mul_columns(z[len(q) + i]) for i in range(e)]
    zeros = [_mul_columns(1 ^ z[len(q) + i]) for i in range(e)]
    out = []
    for y in range(2 ** e):
        t = pre
        for i in range(e):
            t = _apply(ones[i] if (y >> i) & 1 else zeros[i], t)
        v = np.bitwise_xor.reduce(gf.vmul(t, eqR[:, None, :]), axis=0)
        out.append((gf.from_arr(v[0])[0], gf.from_arr(v[1])[0]))
    return out


def _positions(lv, coins: list[int]) -> list[int]:
    out, j = [], 0
    for c in lv.strata:
        for s in range(2 ** c):
            out.append((s << (lv.d - c)) | ((coins[j] & ((1 << 64) - 1)) % (1 << (lv.d - c))))
            j += 1
    return out


def _sv_table(n: int) -> list[int]:
    sv = []
    for i in range(n):
        s = 1 << i
        for j in range(i):
            s = gf.mul(s, s) ^ gf.mul(sv[j], s)
        sv.append(s)
    return sv


def _whats(sv, sv_inv, q: int, n: int) -> list[int]:
    out, s = [], q
    for i in range(n):
        out.append(gf.mul(s, sv_inv[i]))
        s = gf.mul(s, s) ^ gf.mul(sv[i], s)
    return out


def _poly_msg(msg, s):
    u0, u2 = msg
    return [u0, gf.kadd(s, u2), u2]


def _poly_eval(q, r):
    return gf.kadd(gf.kadd(q[0], gf.kmul(q[1], r)), gf.kmul(gf.kmul(q[2], r), r))


def _poly_add(q, intro, b):
    return [gf.kadd(a, gf.ksmul(c, b)) for a, c in zip(q, intro)]


def ligerito_state(n: int, sch, co: dict, T, msgs, oods, es):
    """The running claim s at the final check (Flock.ligerito), as a function of the messages: affine, with the coins fixed."""
    Rl = len(sch.levels) - 1
    it = iter(range(len(msgs)))
    s = gf.kadd(_k(T), gf.ksmul(_k(oods[0]), co[("ood_beta", 0)][0]))
    q = _poly_msg(msgs[next(it)], s)
    chal = lambda lvl, c: (co[("chal", lvl, c)][0], co[("chal", lvl, c)][1])
    for c in range(sch.k0):
        s = _poly_eval(q, chal(0, c))
        q = _poly_msg(msgs[next(it)], s)
    oi = 1

    def ood_intro(q, s, oi):
        y = _k(oods[oi])
        intro = _poly_msg(msgs[next(it)], y)
        b = co[("ood_beta", oi)][0]
        return _poly_add(q, intro, b), gf.kadd(s, gf.ksmul(y, b)), oi + 1

    q, s, oi = ood_intro(q, s, oi)
    q, s, oi = ood_intro(q, s, oi)
    b0 = co[("cons_beta", 0)][0]
    q = _poly_add(q, _poly_msg(msgs[next(it)], es[0]), b0)
    s = gf.kadd(s, gf.ksmul(es[0], b0))
    for i in range(Rl):
        for c in range(4):
            s = _poly_eval(q, chal(i + 1, c))
            q = _poly_msg(msgs[next(it)], s)
        if i + 1 < Rl:
            q, s, oi = ood_intro(q, s, oi)
            q, s, oi = ood_intro(q, s, oi)
            bb = co[("cons_beta", i + 1)][0]
            q = _poly_add(q, _poly_msg(msgs[next(it)], es[i + 1]), bb)
            s = gf.kadd(s, gf.ksmul(es[i + 1], bb))
        else:
            bb = co[("cons_beta", Rl)][0]
            s = gf.kadd(s, gf.ksmul(es[Rl], bb))
    return s


def ligerito_tables(fr: Frame, co: dict, sws) -> dict:
    sh = fr.shape
    sch = sh.sched
    n = sch.n
    Rl = len(sch.levels) - 1
    nmsg = sch.k0 + 8 + 7 * (Rl - 1)
    nood = 1 + 2 * Rl
    nlev = Rl + 1
    zero = (0, 0)
    base = dict(T=0, msgs=[(zero, zero)] * nmsg, oods=[zero] * nood, es=[zero] * nlev)

    def probe(**kw):
        a = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
        a.update(kw)
        return ligerito_state(n, sch, co, a["T"], a["msgs"], a["oods"], a["es"])

    assert probe() == zero
    kT = probe(T=1)
    kmsg = []
    for i in range(nmsg):
        row = []
        for h in range(2):
            ms = list(base["msgs"])
            ms[i] = (K1, zero) if h == 0 else (zero, K1)
            row.append(probe(msgs=ms))
        kmsg.append(row)
    kood = []
    for i in range(nood):
        os_ = list(base["oods"])
        os_[i] = K1
        kood.append(probe(oods=os_))
    ke = []
    for i in range(nlev):
        es = list(base["es"])
        es[i] = K1
        ke.append(probe(es=es))
    # the final check's coefficients of yr (Flock.ligerito §13.6); coins only
    chal = lambda lvl, c: (co[("chal", lvl, c)][0], co[("chal", lvl, c)][1])
    init = [chal(0, c) for c in range(sch.k0)]
    levels = [[chal(i + 1, c) for c in range(4)] for i in range(Rl)]
    orig = init + [x for lv in levels for x in lv[1:]]
    ext = sch.ext
    assert len(orig) + ext == n
    heads = [lv[0] for lv in levels]
    cff = lambda c: gf.kadd(K1, gf.kmul(c, (1, 1)))

    def prod_from(a):
        acc = K1
        for c in heads[a:]:
            acc = gf.kmul(acc, cff(c))
        return acc

    pi0 = prod_from(0)
    bres = [zero] * 2 ** ext
    for sw in sws:
        b = basis_for_claim(sw, orig, ext)
        bres = [gf.kadd(x, gf.ksmul(y, sw[0])) for x, y in zip(bres, b)]
    res = [gf.kmul(x, pi0) for x in bres]
    sv = _sv_table(n + 1)
    sv_inv = [gf.inv(x) for x in sv]
    # out-of-domain contexts
    ctxs = [(co[("ood_z", 0)], co[("ood_beta", 0)][0], None)]
    oi = 1
    for split in [0, 0] + [s for i in range(Rl - 1) for s in (i + 1, i + 1)]:
        ctxs.append((co[("ood_z", oi)], co[("ood_beta", oi)][0], split))
        oi += 1
    for z, beta, split in ctxs:
        if split is None:
            fixed, scale = orig, pi0
        else:
            fixed = []
            for lv in levels[split:]:
                fixed = list(lv) if not fixed else fixed + lv[1:]
            scale = prod_from(split + 1)
        head = K1
        for t in range(len(fixed)):
            head = gf.kmul(head, gf.kadd(gf.kadd(K1, _k(z[t])), fixed[t]))
        cc = gf.ksmul(gf.kmul(head, scale), beta)
        for y in range(2 ** ext):
            tail = 1
            for j in range(ext):
                zj = z[len(fixed) + j]
                tail = gf.mul(tail, zj if (y >> j) & 1 else 1 ^ zj)
            res[y] = gf.kadd(res[y], gf.ksmul(cc, tail))
    # consistency contexts
    tabs = {}
    ext_at = n - sch.k0
    for lv_i in range(Rl + 1):
        L = sch.levels[lv_i]
        pos = _positions(L, co[("q_coins", lv_i)])
        al = co[("q_alpha", lv_i)]
        start = lv_i
        fixed = [x for lv in levels[start:] for x in lv[1:]]
        cols = len(fixed) + ext
        assert lv_i > 0 or cols == ext_at
        ea = gf.eq_table(al)
        scale = prod_from(start)
        beta = co[("cons_beta", lv_i)][0]
        add = [zero] * 2 ** ext
        for j, ps in enumerate(pos):
            wh = _whats(sv, sv_inv, ps, cols)
            pj = K1
            for i in range(len(fixed)):
                pj = gf.kmul(pj, gf.kadd(K1, gf.ksmul(fixed[i], 1 ^ wh[i])))
            coef = gf.ksmul(pj, ea[j])
            for y in range(2 ** ext):
                tj = 1
                for i in range(ext):
                    if (y >> i) & 1:
                        tj = gf.mul(tj, wh[len(fixed) + i])
                add[y] = gf.kadd(add[y], gf.ksmul(coef, tj))
        for y in range(2 ** ext):
            res[y] = gf.kadd(res[y], gf.ksmul(gf.kmul(add[y], scale), beta))
        # the per-level tables V[B] reads
        fold = init if lv_i == 0 else levels[lv_i - 1]
        tabs[f"lig.ea.{lv_i}"] = [[x] for x in ea[:L.q]]
        tabs[f"lig.ef.{lv_i}"] = gf.eq_table_k(fold)
        tabs[f"lig.dir.{lv_i}"] = [[(ps >> t) & 1 for t in range(L.d - L.cap)] for ps in pos]
        tabs[f"lig.sel.{lv_i}"] = [[1 if (ps >> (L.d - L.cap)) == e else 0 for e in range(2 ** L.cap)] for ps in pos]
    tabs.update({"lig.kT": kT, "lig.kmsg": kmsg, "lig.kood": kood, "lig.ke": ke, "lig.res": res})
    return tabs


# ---------------------------------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------------------------------

def _as_F(x) -> np.ndarray:
    """Nested lists of F ints, or of K pairs (which numpy expands into a trailing axis of 2), as (..., 2) uint64."""
    a = np.asarray(x, dtype=object)
    return gf.to_arr(a.reshape(-1)).reshape(a.shape + (2,))


def tables(fr: Frame, st: Statement, sess: Session) -> dict[str, np.ndarray]:
    """Every coin-derived public input of V[B], for both reps (leading axis R)."""
    per = []
    for rep in range(R):
        st_ = sess.streams[f"circuit/rep{rep}"]
        co = named_coins(fr.layouts[rep], st_)
        zc = zerocheck_tables(fr.B.m, co)
        lc = lincheck_tables(fr.B.k_log, co)
        fd = fold_tables(fr, st, zc, lc)
        cl = claim_points(fr, st, zc, lc, sess.points)
        rs = ring_tables(cl, co)
        rg = region_tables(fr, st, sess.points)
        lg = ligerito_tables(fr, co, rs["sws"])
        d = {"zc.kab": zc["kab"], "zc.kc": zc["kc"], "zc.kg1": zc["kg1"], "zc.kginf": zc["kginf"], "zc.wL": zc["wL"],
             "lc.kfa": lc["kfa"], "lc.kfb": lc["kfb"], "lc.kconst": lc["kconst"], "lc.kq1": lc["kq1"], "lc.kqinf": lc["kqinf"],
             "lc.wrp": lc["wrp"], "lc.alpha": fd["alpha"], "lc.su": fd["su"], "lc.kcp": fd["kcp"], "lc.eqTlo": fd["eqTlo"],
             "lc.eqTmsg": fd["eqTmsg"], "lc.kleaf": fd["kleaf"], "rs.w": rs["w"], "rs.kT": rs["kT"]}
        d = {k: _as_F(v) for k, v in d.items()}
        d["lc.ypub"] = fd["ypub"]
        d["lc.e0"] = fd["e0"]
        for k, v in rg.items():
            d[k] = _as_F(v)
        for k, v in lg.items():
            d[k] = np.asarray(v, dtype=np.uint8) if (k.startswith("lig.dir") or k.startswith("lig.sel")) else _as_F(v)
        per.append(d)
    return {k: np.stack([per[0][k], per[1][k]]) for k in per[0]}
