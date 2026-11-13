"""The sampling tolerance of Pearl+'s credit, checked against an exact binomial tail.

`miss_mass` and `epsilon` below are `PouwCheck.PearlPlus.missMass` and `epsilon`
(`verity/protocols/pouw/verifier/lean/pearlplus/PouwPearlPlus/Credit.lean`) transcribed line for line: the same loop
bounds, the same integer update of the binomial coefficient, the same 31 halvings.  The probe checks, for each number of
draws t and each count x of draws on tiles over their bound:

* epsilon(t, x) is one grid step (1e-9) above the least k/1e9 at which t draws at that share land at most x times with
  chance at most delta: the one-sided Clopper-Pearson upper bound;
* with no draw (t = 0), or with every draw over the bound (x >= t), epsilon is above 1, so the credit is nothing;
* for a set of tiles holding the share p of the tickets, the chance that p exceeds epsilon at the count the draws give
  is at most delta: the statement the credit rests on.

It is a transcription, so it checks the algorithm as written and not the compiled Lean.  It runs in about a second:

    python benchmarks/pouw/rating_epsilon_probe.py
"""
from fractions import Fraction
from math import comb

G = 10 ** 9
DELTA = Fraction(1, 100)


def miss_mass(t: int, x: int, k: int, g: int) -> int:
    c, s = 1, 0
    for i in range(0, min(x, t) + 1):
        s = s + c * k ** i * (g - k) ** (t - i)
        c = c * (t - i) // (i + 1)
    return s


def epsilon(t: int, over: int, delta: Fraction) -> Fraction:
    def ok(k: int) -> bool:
        return miss_mass(t, over, k, G) * delta.denominator <= delta.numerator * G ** t

    lo, hi = 0, G
    for _ in range(31):
        if lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
    return Fraction(lo + 1, G)


def tail(t: int, x: int, p: Fraction) -> Fraction:
    a, b = p.numerator, p.denominator
    return Fraction(sum(comb(t, i) * a ** i * (b - a) ** (t - i) for i in range(0, min(x, t) + 1)), b ** t)


def main() -> int:
    failures = 0
    shares = [Fraction(1, 1000), Fraction(1, 100), Fraction(3, 100), Fraction(1, 10), Fraction(3, 10), Fraction(7, 10)]
    for t in (0, 1, 2, 5, 37, 120):
        eps = {x: epsilon(t, x, DELTA) for x in range(0, t + 1)}
        for x, e in eps.items():
            k = int(e * G) - 1
            if x >= t:
                if e <= 1:
                    failures += 1
                    print(f"FAIL t={t} x={x}: every draw over the bound, yet epsilon {e} is not above 1")
                continue
            if tail(t, x, Fraction(k, G)) > DELTA:
                failures += 1
                print(f"FAIL t={t} x={x}: the tail at the k found is above delta")
            if k > 0 and tail(t, x, Fraction(k - 1, G)) <= DELTA:
                failures += 1
                print(f"FAIL t={t} x={x}: the k found is not the least")
        for p in shares if t else []:
            a, b = p.numerator, p.denominator
            miss = Fraction(sum(comb(t, x) * a ** x * (b - a) ** (t - x) for x in range(0, t + 1) if p > eps[x]), b ** t)
            if miss > DELTA:
                failures += 1
                print(f"FAIL t={t} share={p}: the share exceeds epsilon with chance {float(miss)}")
        print(f"t={t}: epsilon at x = 0, 1, 2 is {[float(eps[x]) for x in sorted(eps)[:3]]}")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
