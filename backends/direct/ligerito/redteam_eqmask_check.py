"""red-team-ligerito: check ligerito-zk's F3 fix (``zk.EqSumcheckMask``, 318ec9b) against brute force.

(1) ``message_pair`` / ``message`` / ``zk_sum`` / ``final_value`` equal the direct sums of an independently evaluated
``g`` weighted by ``prod eq(tau_i, x_i)`` (n = 5, d = 2, F_{p^6}).  (2) The F3 distinguisher on the eq-weighted zero-check
``eq(tau, x) (A B - C + rho g)`` (n = 6): the bivariate opening message's 4 cross coefficients (a, b >= 1) are identical
across fresh ``(rho, g)`` without the cross block and vary with it.

    python -m backends.direct.ligerito.redteam_eqmask_check [--zk ~/projects/verity-main-wt/ligerito-zk]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

P, D = 2013265921, 6


def load_zk(root: Path):
    spec = importlib.util.spec_from_file_location("zk_under_test", root / "backends/direct/ligerito/zk.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["zk_under_test"] = m
    spec.loader.exec_module(m)
    return m


def mk_ops(zk):
    mul, one, zero, base = zk.ext_mul, zk.ext_one, zk.ext_zero, zk.ext_from_base
    add = lambda a, b: (np.asarray(a) + np.asarray(b)) % P          # noqa: E731
    sub = lambda a, b: (np.asarray(a) - np.asarray(b)) % P          # noqa: E731

    def eq1(t, x):
        return add(mul(t, x), mul(sub(one(), t), sub(one(), x)))

    def pw(x, e):
        r = one()
        for _ in range(e):
            r = mul(r, x)
        return r
    return mul, add, sub, one, zero, base, eq1, pw


def g_direct(ops, a0, a, c, x):
    mul, add, _, _, _, _, _, pw = ops
    acc = a0
    n, d = a.shape[0], a.shape[1]
    for i in range(n):
        for e in range(d):
            acc = add(acc, mul(a[i, e], pw(x[i], e + 1)))
    if c is not None:
        for p_, q_ in itertools.product(range(d), range(d)):
            acc = add(acc, mul(c[p_, q_], mul(pw(x[0], p_ + 1), pw(x[1], q_ + 1))))
    return acc


def check_impl(zk, rng, n=5, d=2):
    ops = mul, add, sub, one, zero, base, eq1, pw = mk_ops(zk)
    rnd = lambda *s: rng.integers(0, P, size=s + (D,), dtype=np.int64)   # noqa: E731
    coeffs, tau = rnd(zk.EqSumcheckMask.n_coeffs(n, d, True)), rnd(n)
    m = zk.EqSumcheckMask(n, d, coeffs, tau, cross=True)
    a0, a, c = coeffs[0], coeffs[1:1 + d * n].reshape(n, d, D), coeffs[1 + d * n:].reshape(d, d, D)
    g = lambda x: g_direct(ops, a0, a, c, x)                              # noqa: E731
    cube = list(itertools.product((0, 1), repeat=n))
    G = zero()
    for x in cube:
        w = one()
        for i in range(n):
            w = mul(w, eq1(tau[i], base(x[i])))
        G = add(G, mul(w, g([base(b) for b in x])))
    ok = {"zk_sum": np.array_equal(G, m.zk_sum())}
    xs = (0, 1, 2)
    mp = m.message_pair(xs)
    good = True
    for i, X1 in enumerate(xs):
        for j, X2 in enumerate(xs):
            s = zero()
            for rest in itertools.product((0, 1), repeat=n - 2):
                w = one()
                for k, b in enumerate(rest):
                    w = mul(w, eq1(tau[2 + k], base(b)))
                s = add(s, mul(w, g([base(X1), base(X2)] + [base(b) for b in rest])))
            good &= np.array_equal(s, mp[i, j])
    ok["message_pair"] = bool(good)
    r = rnd(n)
    for t in range(2, n):
        mt = m.message(t, r[:t], xs)
        good = True
        for k, X in enumerate(xs):
            s = zero()
            for rest in itertools.product((0, 1), repeat=n - t - 1):
                w = one()
                for kk, b in enumerate(rest):
                    w = mul(w, eq1(tau[t + 1 + kk], base(b)))
                s = add(s, mul(w, g(list(r[:t]) + [base(X)] + [base(b) for b in rest])))
            good &= np.array_equal(s, mt[k])
        ok[f"message t={t}"] = bool(good)
    ok["final_value"] = np.array_equal(m.final_value(r), g(list(r)))
    return ok


def interp3(v):
    """values at 0, 1, 2 (axis 0) -> coefficients c0, c1, c2."""
    inv2 = (P + 1) // 2
    c0 = v[0]
    c2 = ((v[2] - 2 * v[1] + v[0]) % P) * inv2 % P
    c1 = (v[1] - v[0] - c2) % P
    return np.stack([c0, c1, c2])


def opening_cross(zk, rng, tables, tau, cross: bool, n=6, d=2):
    """The 4 cross coefficients (X1^a X2^b, a, b >= 1) of the full opening message t(X1, X2) =
    sum_rest prod_{i>=2} eq(tau_i, x_i) (A B - C + rho g)(X1, X2, rest)."""
    ops = mul, add, sub, one, zero, base, eq1, pw = mk_ops(zk)
    A, B, C = tables
    rho = rng.integers(0, P, size=D, dtype=np.int64)
    coeffs = rng.integers(0, P, size=(zk.EqSumcheckMask.n_coeffs(n, d, cross), D), dtype=np.int64)
    m = zk.EqSumcheckMask(n, d, coeffs, tau, cross=cross)
    mp = m.message_pair((0, 1, 2))

    def mle(tab, x):
        acc = zero()
        for idx in range(1 << n):
            w = one()
            for j in range(n):
                w = mul(w, x[j] if (idx >> j) & 1 else sub(one(), x[j]))
            acc = add(acc, mul(w, base(tab[idx])))
        return acc
    vals = np.zeros((3, 3, D), dtype=np.int64)
    for i, X1 in enumerate((0, 1, 2)):
        for j, X2 in enumerate((0, 1, 2)):
            s = zero()
            for rest in itertools.product((0, 1), repeat=n - 2):
                x = [base(X1), base(X2)] + [base(b) for b in rest]
                w = one()
                for k, b in enumerate(rest):
                    w = mul(w, eq1(tau[2 + k], base(b)))
                q = sub(mul(mle(A, x), mle(B, x)), mle(C, x))
                s = add(s, mul(w, q))
            vals[i, j] = add(s, mul(rho, mp[i, j]))
    coef = interp3(np.stack([interp3(vals[:, j]) for j in range(3)], axis=1).transpose(1, 0, 2)).transpose(1, 0, 2)
    return coef[1:, 1:].copy()


def witness(rng, n=6):
    A = rng.integers(0, P, size=1 << n, dtype=np.int64)
    B = rng.integers(0, P, size=1 << n, dtype=np.int64)
    return A, B, (A * B) % P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zk", default=str(Path.home() / "projects/verity-main-wt/ligerito-zk"))
    zk = load_zk(Path(ap.parse_args().zk))
    rng = np.random.default_rng(1915)
    print("implementation vs brute force:", check_impl(zk, rng))
    tau = rng.integers(0, P, size=(6, D), dtype=np.int64)
    w1, w2 = witness(rng), witness(rng)
    for cross in (False, True):
        draws = [opening_cross(zk, rng, w1, tau, cross) for _ in range(3)]
        same = all(np.array_equal(draws[0], x) for x in draws[1:])
        other = opening_cross(zk, rng, w2, tau, cross)
        print(f"cross={cross}: opening cross coefficients identical across 3 fresh (rho, g): {same}; "
              f"second witness differs: {not np.array_equal(other, draws[0])}")


if __name__ == "__main__":
    main()
