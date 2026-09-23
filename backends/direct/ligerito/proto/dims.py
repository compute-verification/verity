"""Round dimensions for Ligerito (paper Appendix A): minimise the opened-row bytes sum_i c_i d_i subject to
prod_i d_i = N over the ell dimensions d_i (d_i = 2^{k'_i} columns of round i < ell; d_ell = 2^{k_{ell-1}} = the
final vector length), relaxed optimum d_i* = (prod_j c_j N)^{1/ell} / c_i, rounded to powers of two with the paper's
fix-up (double the smallest / halve the largest until the product is N).  Costs: c_1 = |S| x 4 B (base rows),
c_i = |S| x 24 B (F_{p^6} rows), c_ell = 24 B.  ``choose`` grid-searches ell and also charges the Merkle paths
(|S| x depth x 32 B per committed round, no de-duplication) and the sumcheck messages when comparing.

Dimension constraints of this prototype: the first codeword length 4 x 2^{k_1} <= 2^24 (two-factor NTT with
factors <= 4096) and every column count C_i = 2^{k'_i} >= 8.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EXT_BYTES = 24
BASE_BYTES = 4
HASH_BYTES = 32
MAX_LOG_CODEWORD = 24          # for make/choose (the measured q192 configs); explicit Dims may go to 2^28 (NTT factors <= 16384)
MIN_LOG_COLS = 3


@dataclass
class Dims:
    n: int                              # log2 N
    kprime: list[int]                   # k'_i, i = 1..L  (L committed rounds)
    queries: list[int]                  # |S_i| per committed round
    rate_log2: int | list[int] = 2     # per committed round (an int = the same rate everywhere)

    @property
    def rates(self) -> list[int]:
        r = self.rate_log2
        return [r] * len(self.kprime) if isinstance(r, int) else list(r)

    @property
    def rounds(self) -> int:
        return len(self.kprime)

    @property
    def k(self) -> list[int]:
        """k_i = rows (log2) of M~_i: k_1 = n - k'_1, k_{i+1} = k_i - k'_{i+1}."""
        ks, cur = [], self.n
        for kp in self.kprime:
            cur -= kp
            ks.append(cur)
        return ks

    @property
    def final_log(self) -> int:
        return self.k[-1]

    def codeword_log(self, i: int) -> int:
        return self.k[i] + self.rates[i]

    def predicted_bytes(self) -> dict:
        ks = self.k
        rows = 0
        paths = 0
        for i, (kp, q) in enumerate(zip(self.kprime, self.queries)):
            eb = BASE_BYTES if i == 0 else EXT_BYTES
            rows += q * (1 << kp) * eb
            paths += q * self.codeword_log(i) * HASH_BYTES
        final = (1 << ks[-1]) * EXT_BYTES
        sumcheck = 3 * EXT_BYTES * sum(self.kprime)
        roots = HASH_BYTES * self.rounds
        return {"rows": rows, "paths_nodedup": paths, "final_vector": final, "sumcheck": sumcheck, "roots": roots,
                "total_nodedup": rows + paths + final + sumcheck + roots + EXT_BYTES}

    def describe(self) -> str:
        ks = self.k
        parts = [f"round {i + 1}: M~ 2^{ks[i]} x 2^{kp} (codeword 2^{self.codeword_log(i)}, rate 2^-{self.rates[i]}, |S| = {q})"
                 for i, (kp, q) in enumerate(zip(self.kprime, self.queries))]
        parts.append(f"final vector 2^{ks[-1]} ext")
        return "; ".join(parts)


def _round_pow2(x: float) -> int:
    import math
    return 1 << max(0, round(math.log2(max(x, 1.0))))


def relaxed(n: int, ell: int, queries: int) -> list[int]:
    """Appendix A: the ell dimensions (log2) for ell - 1 committed rounds plus the final vector."""
    import math
    c = [queries * BASE_BYTES] + [queries * EXT_BYTES] * (ell - 2) + [EXT_BYTES]
    N = 1 << n
    logprod = sum(math.log2(ci) for ci in c) + n
    d = [2 ** (logprod / ell - math.log2(ci)) for ci in c]
    d = [_round_pow2(x) for x in d]
    while True:
        prod = 1
        for x in d:
            prod *= x
        if prod == N:
            break
        if prod < N:
            i = min(range(ell), key=lambda j: d[j])
            d[i] *= 2
        else:
            i = max(range(ell), key=lambda j: d[j])
            d[i] //= 2
    return [x.bit_length() - 1 for x in d]


def make(n: int, ell: int, queries: int | list[int], rate_log2: int = 2) -> Dims:
    logs = relaxed(n, ell, queries if isinstance(queries, int) else queries[0])
    kprime = logs[:-1]
    # prototype constraints: 4 x 2^{k_1} <= 2^24  <=>  k'_1 >= n - (24 - rate_log2); C_i >= 8
    need = n - (MAX_LOG_CODEWORD - rate_log2)
    if kprime[0] < need:
        deficit = need - kprime[0]
        kprime[0] = need
        # take the deficit out of the final vector first, then the last rounds
        final = logs[-1]
        take = min(deficit, max(0, final - MIN_LOG_COLS))
        final -= take
        deficit -= take
        j = len(kprime) - 1
        while deficit > 0 and j >= 1:
            take = min(deficit, max(0, kprime[j] - MIN_LOG_COLS))
            kprime[j] -= take
            deficit -= take
            j -= 1
        assert deficit == 0, "cannot satisfy the codeword-length constraint"
        logs = kprime + [final]
    for kp in kprime:
        assert kp >= MIN_LOG_COLS, kprime
    assert sum(logs) == n
    qs = [queries] * len(kprime) if isinstance(queries, int) else list(queries)
    return Dims(n=n, kprime=kprime, queries=qs, rate_log2=rate_log2)


def choose(n: int, queries: int = 192, ells=(3, 4, 5, 6), rate_log2: int = 2) -> Dims:
    """The ell with the smallest predicted total (rows + non-deduplicated paths + final + messages)."""
    best = None
    for ell in ells:
        if ell - 1 > n // MIN_LOG_COLS:
            continue
        try:
            d = make(n, ell, queries, rate_log2)
        except AssertionError:
            continue
        tot = d.predicted_bytes()["total_nodedup"]
        if best is None or tot < best[0]:
            best = (tot, d)
    assert best is not None
    return best[1]


if __name__ == "__main__":
    import sys

    q = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    for n in (20, 24, 26, 28, 29, 30):
        for ell in (3, 4, 5):
            try:
                d = make(n, ell, q)
            except AssertionError as e:
                print(n, ell, "infeasible", e)
                continue
            pb = d.predicted_bytes()
            print(f"n={n} ell={ell} k'={d.kprime} k={d.k} total={pb['total_nodedup']/1024:.0f} KiB "
                  f"(rows {pb['rows']/1024:.0f}, paths {pb['paths_nodedup']/1024:.0f}, final {pb['final_vector']/1024:.0f})")
        print("  choose ->", choose(n, q).describe())
