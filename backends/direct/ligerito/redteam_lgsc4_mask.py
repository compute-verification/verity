"""red-team-ligerito-3: LGSC0004's Libra masks (ligerito-sumcheck-3 c675bd5 ``ZkMask``) re-implemented over BabyBear in pure
Python and checked for arities 1..4, both kinds:

  1. sum-zero: the round check of a mask message returns the running constant ``K`` (``eq``: ``sum_b eq(tau, b) M(b) = K``;
     ``avg``: ``sum_b M(b) = 2^(n_rest + v) K``) for every coefficient vector, so the masked claim equals the unmasked one;
  2. binding: ``eval(M, r) = K + g(r) - base`` = the ``bind`` update (the next round's mask constant);
  3. hiding: the map ``c -> M - K ones`` has rank ``3^v - 1`` and its image is the kernel of the round check, so for every
     witness message ``q`` the masked message is uniform on ``{M : check(M) = check(q) + K}`` (no witness-dependent coordinate
     survives, cross terms included);
  4. contrast (red-team F3): the per-variable Libra mask ``sum_j g_j(X_j)`` spans only ``2v`` dimensions of the ``3^v - 1``;
  5. whole-mask brute force: ``sum_x eq(tau, x) ghat(x) = 0`` and ``ghat(r) = K_final`` on a toy hypercube with round groups.

Challenges are base-field elements (a special case of F_{p^6}; the rank statements are polynomial identities in tau).
Run: ``python -m backends.direct.ligerito.redteam_lgsc4_mask`` (< 5 s, no numpy/torch)."""
from __future__ import annotations

import itertools
import random

P = 2013265921


def inv(a: int) -> int:
    return pow(a % P, P - 2, P)


def idx3(v: int):
    return list(itertools.product(range(3), repeat=v))


def mono_to_mixed(c: dict, v: int) -> dict:
    """monomial coefficients c[e] (e in {0,1,2}^v) -> mixed representation M[a] (a in {0,1,inf=2}^v); per axis
    X^0 -> (1, 1, 0), X^1 -> (0, 1, 0), X^2 -> (0, 1, 1) (sumcheck.py ZkMask.message)."""
    M = dict(c)
    for ax in range(v):
        N = {}
        for a in idx3(v):
            if a[ax] == 0:
                N[a] = M[a]
            elif a[ax] == 1:
                a0, a2 = a[:ax] + (0,) + a[ax + 1:], a[:ax] + (2,) + a[ax + 1:]
                N[a] = (M[a0] + M[a] + M[a2]) % P
            else:
                N[a] = M[a]
        M = N
    return M


def contract(M: dict, ws: list, v: int) -> int:
    """sum_a M[a] prod_j ws[j][a_j]"""
    tot = 0
    for a in idx3(v):
        w = 1
        for j in range(v):
            w = w * ws[j][a[j]] % P
        tot = (tot + M[a] * w) % P
    return tot


def w_eval(r):
    return (1 - r) % P, r % P, r * (r - 1) % P


def w_eq(t):
    return (1 - t) % P, t % P, 0


W_SUM = (1, 1, 0)


def mono_weights(kind: str, t):
    if kind == "avg":
        h = inv(2)
        return (1, h, h)
    if kind == "eq":
        return (1, t % P, t % P)
    return (1, t % P, t * t % P)       # eval


def mask_message(c: dict, v: int, kind: str, taus, K: int, n_rest: int) -> dict:
    """ZkMask.message: mixed(g_c) + (K - base) ones, times 2^n_rest for avg."""
    M = mono_to_mixed(c, v)
    base = contract(c, [mono_weights(kind, taus[j] if kind == "eq" else None) for j in range(v)], v)
    ones = mono_to_mixed({a: (1 if a == (0,) * v else 0) for a in idx3(v)}, v)
    M = {a: (M[a] + ones[a] * (K - base)) % P for a in idx3(v)}
    if kind == "avg" and n_rest:
        s = pow(2, n_rest, P)
        M = {a: M[a] * s % P for a in M}
    return M


def bind(c: dict, v: int, kind: str, taus, r, K: int) -> int:
    ev = contract(c, [mono_weights("eval", r[j]) for j in range(v)], v)
    bs = contract(c, [mono_weights(kind, taus[j] if kind == "eq" else None) for j in range(v)], v)
    return (K + ev - bs) % P


def rank_mod_p(rows: list) -> int:
    A = [list(r) for r in rows]
    rk, ncol = 0, len(A[0]) if A else 0
    for col in range(ncol):
        piv = next((i for i in range(rk, len(A)) if A[i][col] % P), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        iv = inv(A[rk][col])
        A[rk] = [x * iv % P for x in A[rk]]
        for i in range(len(A)):
            if i != rk and A[i][col] % P:
                f = A[i][col]
                A[i] = [(x - f * y) % P for x, y in zip(A[i], A[rk])]
        rk += 1
    return rk


def rand_c(rng, v, only=None):
    c = {a: 0 for a in idx3(v)}
    for a in idx3(v):
        if a != (0,) * v and (only is None or only(a)):
            c[a] = rng.randrange(P)
    return c


def check_round(rng, v: int, kind: str) -> dict:
    taus = [rng.randrange(P) for _ in range(v)]
    r = [rng.randrange(P) for _ in range(v)]
    K, n_rest = rng.randrange(P), rng.randrange(0, 20)
    chk_w = [w_eq(t) for t in taus] if kind == "eq" else [W_SUM] * v
    scale = 1 if kind == "eq" else pow(2, n_rest + v, P)
    out = {"v": v, "kind": kind}
    # 1 + 2 on random coefficients
    ok1 = ok2 = True
    for _ in range(4):
        c = rand_c(rng, v)
        M = mask_message(c, v, kind, taus, K, n_rest)
        ok1 &= contract(M, chk_w, v) == K * scale % P
        want = bind(c, v, kind, taus, r, K)
        got = contract(M, [w_eval(x) for x in r], v)
        ok2 &= got == want * (1 if kind == "eq" else pow(2, n_rest, P)) % P
    out["sum_zero"], out["bind"] = ok1, ok2
    # 3: rank of c -> M - K ones (K = 0), image inside ker(check)
    basis = [a for a in idx3(v) if a != (0,) * v]
    cols = []
    for e in basis:
        c = {a: (1 if a == e else 0) for a in idx3(v)}
        M = mask_message(c, v, kind, taus, 0, n_rest)
        cols.append([M[a] for a in idx3(v)])
    out["rank"] = rank_mod_p(cols)
    out["full"] = 3 ** v - 1
    check_row = []
    for a in idx3(v):
        w = 1
        for j in range(v):
            w = w * chk_w[j][a[j]] % P
        check_row.append(w)
    out["image_is_kernel"] = out["rank"] == out["full"] and all(
        sum(x * y for x, y in zip(col, check_row)) % P == 0 for col in cols)
    # 4: per-variable Libra (monomials in one variable only)
    per_var = [e for e in basis if sum(1 for x in e if x) == 1]
    cols_pv = []
    for e in per_var:
        c = {a: (1 if a == e else 0) for a in idx3(v)}
        M = mask_message(c, v, kind, taus, 0, n_rest)
        cols_pv.append([M[a] for a in idx3(v)])
    out["per_variable_rank"] = rank_mod_p(cols_pv)
    return out


def eq1(t, x):
    return (t * x + (1 - t) * (1 - x)) % P


def whole_mask(rng, groups=(3, 2, 2), vf=2) -> dict:
    """Brute force on {0,1}^n, n = sum(groups) + vf (the final table round carries no group): ghat(x) = sum_R g_R(x_R) - base_R,
    variables bound top-first like sumcheck.py (group 0 = the top variables)."""
    n = sum(groups) + vf
    tau = [rng.randrange(P) for _ in range(n)]
    cs = [rand_c(rng, v) for v in groups]
    # variable order: x[n-1] is the top; group R takes the next v top variables, axis j <-> x[top - j]
    spans, top = [], n - 1
    for v in groups:
        spans.append([top - j for j in range(v)])
        top -= v

    def g_at(R, vals):
        v = groups[R]
        tot = 0
        for e in idx3(v):
            if cs[R][e]:
                m = cs[R][e]
                for j in range(v):
                    m = m * pow(vals[j], e[j], P) % P
                tot = (tot + m) % P
        return tot

    bases = [contract(cs[R], [mono_weights("eq", tau[spans[R][j]]) for j in range(groups[R])], groups[R])
             for R in range(len(groups))]

    def ghat(x):
        return sum(g_at(R, [x[i] for i in spans[R]]) - bases[R] for R in range(len(groups))) % P

    tot = 0
    for bits in itertools.product((0, 1), repeat=n):
        x = bits[::-1]
        e = 1
        for i in range(n):
            e = e * eq1(tau[i], x[i]) % P
        tot = (tot + e * ghat(x)) % P
    r = [rng.randrange(P) for _ in range(n)]
    K = 0
    for R, v in enumerate(groups):
        K = bind(cs[R], v, "eq", [tau[i] for i in spans[R]], [r[i] for i in spans[R]], K)
    xr = list(r)
    for i in range(vf):                           # the final table variables: ghat is constant K there
        xr[i] = rng.randrange(P)
    return {"n": n, "groups": groups, "eq_sum_ghat": tot, "K_final_eq_ghat_r": K == ghat(xr)}


def main() -> int:
    rng = random.Random(20260923)
    bad = 0
    print("arity kind sum_zero bind rank/full image=ker(check) per-variable-Libra-rank")
    for v in (1, 2, 3, 4):
        for kind in ("eq", "avg"):
            o = check_round(rng, v, kind)
            ok = o["sum_zero"] and o["bind"] and o["image_is_kernel"]
            bad += not ok
            print(f"{v:5d} {kind:4s} {o['sum_zero']!s:8s} {o['bind']!s:5s} {o['rank']:3d}/{o['full']:<3d} "
                  f"{o['image_is_kernel']!s:14s} {o['per_variable_rank']} (leaves {o['full'] - o['per_variable_rank']} "
                  f"witness coordinates clear)")
    w = whole_mask(rng)
    bad += not (w["eq_sum_ghat"] == 0 and w["K_final_eq_ghat_r"])
    print(f"whole mask n={w['n']} groups={w['groups']}: sum_x eq(tau,x) ghat(x) = {w['eq_sum_ghat']}, "
          f"ghat(r) == K_final: {w['K_final_eq_ghat_r']}")
    print("PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
