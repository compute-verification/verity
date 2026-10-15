"""Partial completion against a k = 2 Sloth word with a dense XOR between its rounds: what attack is left?

The plain word (the word codec's `encode_word` at k = 2) is y = ρ(ρ(a) + c′), with a = m + c₀ + t(i) and
c′ = c₁ + t(i), and the red team's quartic completes its bottom b/10 bits with no root (`partial_root_k2.py`). The
masked word puts PoUS P2's σ_K between the rounds: y = ρ(z + c′) with z = σ_K(ρ(a)), where σ_K(r) = r ⊕ K if
r ⊕ K < p and r otherwise (an involution on [0, p), never reduced mod p) and K is a dense public b-bit constant. Its
decode is a = ±σ_K(±y² − c′)², two squarings and a b-bit XOR. Under the decode-free bound the adversary knows a, c′
and K. Each experiment prints one JSON line a (field, x), every completion checked against the encode:

- `quartic`: the red team's zero-root 4D lattice on y's top bits, K = 0 (the plain word, the control) and K dense.
- `advice`: the same lattice, with the XOR's carry term D = z ∧ K stored beside y's top bits. Since
  ρ(a) = z + K − 2D, a = s₁·(s₂·y² − (c′ − K + 2D))² is the plain word's quartic again; D costs wt(K) bits a word.
- `peel`: one root of a, then k = 1's 2D lattice on the last round (y² ≡ ±(z + c′), z known), at x = b/3 − 4.
- `intermediate`: store ρ(a)'s top bits instead of y's, complete them by the 2D lattice, then one root for y.
- `copper`: one root, then Howgrave-Graham's lattice for the last round's monic quadratic at m = 2, 3, 4, 6
  (dimension 2m + 1, x < b·m/(2m + 1): 0.4b, 0.43b, 0.44b, 0.46b; b/2 in the limit), reduced by fplll, with FLINT's
  integer root-finding of the short polynomial. `margin_floor` = b/(2x) is the margin of a free reduction.
- `wrap`: k = 1 between two cheap public maps, y = σ_K₂(ρ(σ_K₁(a))): the adversary keeps w = σ_K₂(y)'s top bits,
  completes w ≡ ±√σ_K₁(a) by the 2D lattice and outputs σ_K₂(w), so the wrap is absorbed.
- `floor`: GMP's extended GCD of (2·H·2^x, p), of which the 2D Gauss reduction is a half (`partial_floor.py`): the
  peel's floor is 1 + gcdext/2 roots against 2x/b, the wrap's gcdext/2 against x/b (`wrap_margin`).

A cost is the attack's time over one GMP root (ρ by `powmod`) in the same process; `claim_over_root` is the per-bit
price (2x/b at k = 2, x/b at k = 1); `margin` is cost over claim, and below 1 the attack beats the price.

`uv run --no-project --with gmpy2 --with python-flint --with fpylll --with cysignals python mask_attack.py
[WORDS [EXPERIMENT,... [FIELD ...]]]`
"""

from __future__ import annotations

import json
import random
import sys
import time

import flint
import gmpy2
from fpylll import LLL, IntegerMatrix
from gmpy2 import mpz

FIELDS = {"p512": (512, 569), "p640": (640, 305), "p768": (768, 825), "p1024": (1024, 105), "p1536": (1536, 3453),
          "p2048": (2048, 1557)}
EXPERIMENTS = ("quartic", "advice", "peel", "intermediate", "copper", "wrap", "floor")


def rho(a: mpz, p: mpz, e: mpz) -> mpz:
    """The word codec's ρ: the even root of a when a is a square, else the odd root of −a."""
    s = gmpy2.powmod(a, e, p)
    if s * s % p == a:
        return s if s % 2 == 0 else p - s
    return s if s % 2 == 1 else p - s


def sigma(r: mpz, k: mpz, p: mpz) -> mpz:
    v = r ^ k
    return v if v < p else r


def encode(a: mpz, cp: mpz, k: mpz, p: mpz, e: mpz) -> tuple[mpz, mpz]:
    """The masked k = 2 word and its mid-state z = σ_K(ρ(a)); K = 0 is the plain word."""
    z = sigma(rho(a, p, e), k, p)
    return rho((z + cp) % p, p, e), z


def decode(y: mpz, cp: mpz, k: mpz, p: mpz) -> mpz:
    z = (y * y - cp) % p if y % 2 == 0 else (-y * y - cp) % p
    r = sigma(z, k, p)
    return r * r % p if r % 2 == 0 else -r * r % p


def gauss(b1: tuple[mpz, mpz], b2: tuple[mpz, mpz]) -> tuple[tuple[mpz, mpz], tuple[mpz, mpz]]:
    n1, n2 = b1[0] * b1[0] + b1[1] * b1[1], b2[0] * b2[0] + b2[1] * b2[1]
    if n1 > n2:
        b1, b2, n1, n2 = b2, b1, n2, n1
    while True:
        q = gmpy2.f_div(2 * (b1[0] * b2[0] + b1[1] * b2[1]) + n1, 2 * n1)
        b2 = (b2[0] - q * b1[0], b2[1] - q * b1[1])
        n2 = b2[0] * b2[0] + b2[1] * b2[1]
        if n2 >= n1:
            return b1, b2
        b1, b2, n1, n2 = b2, b1, n2, n1


def complete_k1(h: mpz, x: int, t: mpz, p: mpz) -> mpz | None:
    """y = ρ(t) from its top bits h: `partial_root.py`'s 2D lattice (Gauss reduction, Babai rounding)."""
    y0, wgt = h << x, mpz(1) << x
    c = 2 * y0 % p
    for sign in (1, -1):
        r = (sign * t - y0 * y0) % p
        b1, b2 = gauss((wgt, -c % p), (mpz(0), p))
        tx, ty = wgt << (x - 1), (mpz(1) << (2 * x - 1)) - r
        det = b1[0] * b2[1] - b1[1] * b2[0]
        al = gmpy2.f_div(2 * (tx * b2[1] - ty * b2[0]) + det, 2 * det)
        be = gmpy2.f_div(2 * (b1[0] * ty - b1[1] * tx) + det, 2 * det)
        for da in (0, -1, 1):
            for db in (0, -1, 1):
                v0 = (al + da) * b1[0] + (be + db) * b2[0]
                v1 = (al + da) * b1[1] + (be + db) * b2[1]
                u = v0 // wgt
                if 0 <= u < wgt and r + v1 == u * u:
                    y = y0 + u
                    if (y * y - sign * t) % p == 0 and (y % 2 == 0) == (sign == 1):
                        return y
    return None


def complete_k2(h: mpz, x: int, a: mpz, cp: mpz, p: mpz) -> mpz | None:
    """y = ρ(ρ(a) + c′) from its top bits h: `partial_root_k2.py`'s 4D lattice (FLINT's LLL, Babai rounding)."""
    A = h << x
    w0 = mpz(1) << (3 * x)
    for s2 in (1, -1):
        f3 = 4 * A % p
        f2 = (6 * A * A - 2 * s2 * cp) % p
        f1 = (4 * A * A * A - 4 * s2 * cp * A) % p
        rows = [[1 << (3 * x), 0, 0, int(-f1 % p)], [0, 1 << (2 * x), 0, int(-f2 % p)], [0, 0, 1 << x, int(-f3 % p)],
                [0, 0, 0, int(p)]]
        red = flint.fmpz_mat(rows).lll()
        basis = [[mpz(int(red[i, j])) for j in range(4)] for i in range(4)]
        q = flint.fmpq_mat([[int(v) for v in r] for r in basis]).inv()
        last = [(mpz(int(q[3, i].p)), mpz(int(q[3, i].q))) for i in range(4)]
        for s1 in (1, -1):
            f0 = (A ** 4 - 2 * s2 * cp * A * A + cp * cp - s1 * a) % p
            coeffs = [(2 * f0 * num + den) // (2 * den) for num, den in last]
            v0 = sum(coeffs[i] * basis[i][0] for i in range(4))
            u = gmpy2.f_div(v0 + w0 // 2, w0)
            for du in (0, -1, 1):
                uu = u + du
                if not 0 <= uu < (mpz(1) << x):
                    continue
                y = A + uu
                y1 = (s2 * y * y - cp) % p
                if ((y % 2 == 0) == (s2 == 1) and (y1 % 2 == 0) == (s1 == 1)
                        and (s1 * y1 * y1 - a) % p == 0):
                    return y
    return None


def polymul(f: list[int], g: list[int]) -> list[int]:
    out = [0] * (len(f) + len(g) - 1)
    for i, fi in enumerate(f):
        for j, gj in enumerate(g):
            out[i + j] += fi * gj
    return out


def copper_rows(f: list[int], x: int, m: int, p: int) -> list[list[int]]:
    """Howgrave-Graham's basis for a monic quadratic f: u^j·p^(m−i)·f^i (i < m, j < 2) and f^m, scaled by X^deg."""
    n = 2 * m + 1
    powers = [[1]]
    for _ in range(m):
        powers.append(polymul(powers[-1], f))
    rows = []
    for i in range(m):
        for j in range(2):
            g = [0] * j + [c * p ** (m - i) for c in powers[i]]
            rows.append(g)
    rows.append(powers[m])
    out = []
    for g in rows:
        r = [0] * n
        for d, c in enumerate(g):
            r[d] = c << (x * d)
        out.append(r)
    return out


def complete_copper(h: mpz, x: int, t: mpz, s: int, m: int, p: mpz) -> mpz | None:
    """y = ρ(t) from its top bits h and its parity s: the short polynomial of the m-th lattice, its integer roots."""
    A = int(h << x)
    pi = int(p)
    f = [int((A * A - s * t) % p), 2 * A % pi, 1]
    mat = IntegerMatrix.from_matrix(copper_rows(f, x, m, pi))
    try:
        LLL.reduction(mat, method="fast", float_type="double")
    except Exception:  # fplll's fast mode fails on wide entries; its default mode doesn't
        mat = IntegerMatrix.from_matrix(copper_rows(f, x, m, pi))
        LLL.reduction(mat)
    n = 2 * m + 1
    for row in range(2):
        v = [int(mat[row, i]) for i in range(n)]
        if not any(v):
            continue
        poly = flint.fmpz_poly([v[i] >> (x * i) for i in range(n)])
        for root, _ in poly.roots():
            u = int(root)
            if 0 <= u < (1 << x):
                y = mpz(A + u)
                if (y * y - s * t) % p == 0:
                    return y
    return None


def dense(rng: random.Random, n: int) -> mpz:
    return mpz(rng.getrandbits(n))


def timed(fn, items) -> tuple[list, float]:
    t0 = time.perf_counter()
    out = [fn(*it) for it in items]
    return out, (time.perf_counter() - t0) / len(items) * 1e6


def line(**kw) -> None:
    for key in ("cost_over_root", "claim_over_root", "margin", "x_over_b", "root_us", "attack_us"):
        if key in kw and kw[key] is not None:
            kw[key] = round(kw[key], 4)
    print(json.dumps(kw), flush=True)


def run(name: str, words: int, experiments: list[str]) -> None:
    n, cc = FIELDS[name]
    p = mpz((1 << n) - cc)
    e = (p + 1) // 4
    rng = random.Random(f"codec-mix/{name}")
    K = dense(rng, n)
    batch = [(mpz(rng.randrange(int(p))), mpz(rng.randrange(int(p)))) for _ in range(words)]
    _, root_us = timed(lambda a: rho(a, p, e), [(a,) for a, _ in batch])
    masked = [encode(a, cp, K, p, e) for a, cp in batch]
    assert all(decode(y, cp, K, p) == a for (a, cp), (y, _) in zip(batch, masked)), "masked round trip"
    common = {"field": name, "bits": n, "words": words, "root_us": root_us}

    if "quartic" in experiments:
        x = n // 10 - 8
        plain = [encode(a, cp, mpz(0), p, e)[0] for a, cp in batch]
        for label, kk, ys in (("plain", 0, plain), ("dense", int(K).bit_count(), [y for y, _ in masked])):
            got, us = timed(lambda y, a, cp: complete_k2(y >> x, x, a, cp, p) == y,
                            [(y, a, cp) for y, (a, cp) in zip(ys, batch)])
            line(experiment="quartic", mask=label, mask_weight=kk, x=x, x_over_b=x / n, completed=sum(got),
                 attack_us=us, cost_over_root=us / root_us, claim_over_root=2 * x / n, stored_bits=n - x + 2, **common)

    if "advice" in experiments:
        x = n // 10 - 8
        cases = [(y, a, (cp - K + 2 * (z & K)) % p) for (a, cp), (y, z) in zip(batch, masked)]
        got, us = timed(lambda y, a, ce: complete_k2(y >> x, x, a, ce, p) == y, cases)
        w = int(K).bit_count()
        line(experiment="advice", mask="dense", mask_weight=w, x=x, x_over_b=x / n, completed=sum(got), attack_us=us,
             cost_over_root=us / root_us, claim_over_root=2 * x / n, stored_bits=n - x + 2 + w, **common)

    if "peel" in experiments:
        x = n // 3 - 4

        def peel(y: mpz, a: mpz, cp: mpz) -> bool:
            z = sigma(rho(a, p, e), K, p)
            return complete_k1(y >> x, x, (z + cp) % p, p) == y

        got, us = timed(peel, [(y, a, cp) for (a, cp), (y, _) in zip(batch, masked)])
        line(experiment="peel", x=x, x_over_b=x / n, completed=sum(got), attack_us=us, cost_over_root=us / root_us,
             claim_over_root=2 * x / n, margin=us / root_us / (2 * x / n), stored_bits=n - x, **common)

    if "intermediate" in experiments:
        x = n // 3 - 4

        def mid(y: mpz, a: mpz, cp: mpz, r: mpz) -> bool:
            r1 = complete_k1(r >> x, x, a, p)
            return r1 is not None and rho((sigma(r1, K, p) + cp) % p, p, e) == y

        firsts = [rho(a, p, e) for a, _ in batch]
        got, us = timed(mid, [(y, a, cp, r) for (a, cp), (y, _), r in zip(batch, masked, firsts)])
        line(experiment="intermediate", x=x, x_over_b=x / n, completed=sum(got), attack_us=us,
             cost_over_root=us / root_us, claim_over_root=2 * x / n, margin=us / root_us / (2 * x / n),
             stored_bits=n - x, **common)

    if "copper" in experiments:
        cases = [(y, a, cp) for (a, cp), (y, _) in zip(batch[:max(4, words // 4)], masked)]
        for m in (2, 3, 4, 6):
            x = int(n * m / (2 * m + 1)) - 2

            def cu(y: mpz, a: mpz, cp: mpz) -> bool:
                z = sigma(rho(a, p, e), K, p)
                return complete_copper(y >> x, x, (z + cp) % p, 1 if y % 2 == 0 else -1, m, p) == y

            cu(*cases[0])
            got, us = timed(cu, cases)
            line(experiment="copper", m=m, dim=2 * m + 1, x=x, x_over_b=x / n, x_bound_over_b=round(m / (2 * m + 1), 4),
                 completed=sum(got), tried=len(got), attack_us=us, cost_over_root=us / root_us,
                 claim_over_root=2 * x / n, margin=us / root_us / (2 * x / n), margin_floor=round(n / (2 * x), 4),
                 stored_bits=n - x + 1, **common)

    if "wrap" in experiments:
        x = n // 3 - 4
        K1, K2 = dense(rng, n), dense(rng, n)
        ys = [sigma(rho(sigma(a, K1, p), p, e), K2, p) for a, _ in batch]

        def wrap(y: mpz, a: mpz) -> bool:
            w = complete_k1(sigma(y, K2, p) >> x, x, sigma(a, K1, p), p)
            return w is not None and sigma(w, K2, p) == y

        got, us = timed(wrap, [(y, a) for y, (a, _) in zip(ys, batch)])
        line(experiment="wrap", k=1, x=x, x_over_b=x / n, completed=sum(got), attack_us=us, cost_over_root=us / root_us,
             claim_over_root=x / n, margin=us / root_us / (x / n), stored_bits=n - x, **common)

    if "floor" in experiments:
        x = n // 3 - 4
        cs = [2 * ((y >> x) << x) % p for y, _ in masked]
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            for c in cs:
                gmpy2.gcdext(c, p)
        g_us = (time.perf_counter() - t0) / reps / words * 1e6
        line(experiment="floor", x=x, x_over_b=x / n, gcdext_us=round(g_us, 3), attack_us=root_us + g_us / 2,
             cost_over_root=1 + g_us / 2 / root_us, claim_over_root=2 * x / n,
             margin=(1 + g_us / 2 / root_us) / (2 * x / n),
             wrap_margin=round(g_us / 2 / root_us / (x / n), 4), **common)


def main(argv: list[str]) -> None:
    words = int(argv[0]) if argv else 100
    experiments = argv[1].split(",") if len(argv) > 1 else list(EXPERIMENTS)
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"unknown experiments: {sorted(unknown)}")
    for name in argv[2:] or list(FIELDS):
        run(name, words, experiments)


if __name__ == "__main__":
    main(sys.argv[1:])
