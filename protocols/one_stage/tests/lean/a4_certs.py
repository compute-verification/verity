"""Certificates for `ExfiltrationVectors.lean`'s A4 vectors (`gridA4`): for each of A4's floored laws and δ = 2^-e, the
count K, the greedy fills of K + 1 and K units as rows (N_s, k_s, c_s, c'_s), and each fill's last factor τ = p/q.

`certS` searches in the kernel one unit at a time, which A4's counts (up to 773,981 units) put out of reach, so this writes
the certificates instead. Exact integers only, and nothing of core's: a heap merge of the strata's falling escape factors
(N_s − c − k_s)/(N_s − c). The kernel checks every certificate with `meetsS` (`gridA4_meets`), so nothing here is trusted.
Prints the Lean list `gridA4`.

    python tests/lean/a4_certs.py
"""
from __future__ import annotations

import heapq
from fractions import Fraction
from math import comb, log

#: A4 P6's layer 0 (`tests/test_consumers.py`): one unit per template, and in proportion to output bits.
A4_SIZES = (287, 6_171_648, 11_480, 287, 287, 587_776)
LAWS = [("a4-floor", (1, 905, 2, 1, 1, 86), (10, 20, 40)), ("a4-by-bits", (10, 104, 12, 20, 39, 10), (20,))]


def factor(n: int, k: int, c: int) -> Fraction:
    return Fraction(max(0, n - c - k), n - c) if c < n else Fraction(0)


def push(heap, sizes, ks, a, s) -> None:
    if a[s] < sizes[s]:
        heapq.heappush(heap, (-factor(sizes[s], ks[s], a[s]), s))


def fills(sizes, ks, upto):
    """The greedy fill of every count 0..upto, taking the largest next factor (ties to the lowest stratum), with the last
    factor it took."""
    heap, a, out = [], [0] * len(sizes), [([0] * len(sizes), Fraction(1))]
    for s in range(len(sizes)):
        push(heap, sizes, ks, a, s)
    for _ in range(upto):
        f, s = heapq.heappop(heap)
        a[s] += 1
        push(heap, sizes, ks, a, s)
        out.append((list(a), -f))
    return out


def escape(sizes, ks, a) -> Fraction:
    num = den = 1
    for n, k, c in zip(sizes, ks, a):
        num, den = num * comb(n - c, k), den * comb(n, k)
    return Fraction(num, den)


def certificate(sizes, ks, e):
    """K and the fills of K + 1 and K: a float walk of the log-escape finds the crossing, exact escapes settle it."""
    heap, a, logm, walked = [], [0] * len(sizes), 0.0, 0
    for s in range(len(sizes)):
        push(heap, sizes, ks, a, s)
    while heap and heap[0][0] != 0 and logm + log(-heap[0][0]) >= -e * log(2) - 1e-6:
        f, s = heapq.heappop(heap)
        logm, a[s], walked = logm + log(-f), a[s] + 1, walked + 1
        push(heap, sizes, ks, a, s)
    table = fills(sizes, ks, walked + 4)
    K = max(c for c in range(max(0, walked - 3), walked + 4) if escape(sizes, ks, table[c][0]) * 2 ** e >= 1)
    assert escape(sizes, ks, table[K + 1][0]) * 2 ** e < 1
    return K, table[K + 1], table[K]


def main() -> None:
    rows = []
    for name, ks, es in LAWS:
        for e in es:
            K, (b, t), (a, u) = certificate(A4_SIZES, ks, e)
            r = ", ".join(f"({n}, {k}, {c}, {d})" for n, k, c, d in zip(A4_SIZES, ks, b, a))
            rows.append(f'  ⟨"{name}", {e}, {K}, [{r}],\n    ({t.numerator}, {t.denominator}), '
                        f'({u.numerator}, {u.denominator})⟩')
    print("def gridA4 : List A4Cert := [\n" + ",\n".join(rows) + "]")


if __name__ == "__main__":
    main()
