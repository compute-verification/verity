"""J10's seeded-line numbers: what one noise seed a call does to GRIND's template moment (note:compute-accounting/
20261019T0825Z-draft-j10-shared-noise §2, §4).

Three checks at h1tile's template (a word of w <= 32 skipped products, dead with probability 2^-w over its row's
coins, otherwise lucky at most 1/2 over its column's, kappa = 4), on a call grid of m_c x n_c tiles of 32 x 32
(r = 32 m_c rows read a column's coins, n = 32 n_c columns a row's):

* `tileluck`: `TileLuck`'s condition (proved; dead events may correlate across rows) on the call, its largest r.theta
  against the one-tile point 1/256: the rate falls as 1/n_c.
* `gridluck`: `GridLuck`'s condition (stated; dead events independent across rows), exponent n/r in place of n, at the
  point r.theta = 1/(4 max(64, n/r)) (`GridTemplateCondition`), and the rate as a factor on today's theta_0 = 1/8192.
* `worst`: the exact moment of the most correlated instance the premises allow (a row's dead event shared by all its
  columns, a column's lucky event by all its rows), at `GridLuck`'s point, and how far past it the instance breaks.

``python benchmarks/rate_redteam/shared_noise_moment.py`` prints JSON lines.
"""
from __future__ import annotations

import json
import math

KAPPA = 4.0
THETA0 = 1 / (256 * 32)
W = range(1, 33)


def _lse(*xs: float) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def tile_log(x: float, n: float, w: int) -> float:
    """log of TileLuck's q e^{n x w} + (1 - q)(e^{xw}/2 + e^{-kappa x w}/2)^n."""
    q = 2.0 ** -w
    lpsi = math.log(0.5) + x * w + math.log1p(math.exp(-(1 + KAPPA) * x * w))
    return _lse(math.log(q) + n * x * w, math.log1p(-q) + n * lpsi)


def grid_log(x: float, e: float, w: int) -> float:
    """log of GridLuck's q e^{e x w} + (1 - q)(e^{xw}/2 + e^{-kappa x w}/2)^e, e = n / r, x = r theta."""
    return tile_log(x, e, w)


def largest(cond, *args) -> float:
    lo, hi = 0.0, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if all(cond(mid, *args, w) <= 0 for w in W) else (lo, mid)
    return lo


def _lbin(n: int, k: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def worst_log(theta: float, r: int, n: int, w: int) -> float:
    """log E e^{theta G} when every column shares its row's dead event and every row its column's lucky one: d dead
    rows (Binomial(r, 2^-w)) and l lucky columns (Binomial(n, 1/2)) make d n + (r - d) l units lucky."""
    q = 2.0 ** -w
    terms = []
    for d in range(min(r, 40) + 1):
        ld = _lbin(r, d, q)
        for l in range(n + 1):
            lucky = d * n + (r - d) * l
            terms.append(ld + _lbin(n, l, 0.5) + theta * w * ((1 + KAPPA) * lucky - KAPPA * r * n))
    return _lse(*terms)


def main() -> None:
    base = largest(tile_log, 32)
    for mc, nc in ((1, 1), (1, 4), (1, 16), (1, 32), (1, 64), (1, 128), (1, 448), (2, 128), (4, 4), (8, 8), (16, 1)):
        r, n = 32 * mc, 32 * nc
        x_tile = largest(tile_log, n)
        e = n / r
        point = 1 / (4 * max(64.0, e))
        ok = all(grid_log(point, e, w) <= 0 for w in W)
        theta = point / r
        print(json.dumps({"m_c": mc, "n_c": nc,
                          "tileluck": {"largest_x": round(x_tile, 6), "rate_loss": round(base / x_tile, 2)},
                          "gridluck": {"largest_x": round(largest(grid_log, e), 6), "point_x": point, "holds": ok,
                                       "premium_factor_vs_theta0": round(THETA0 / theta, 3)}}), flush=True)
    for mc, nc in ((1, 1), (1, 16), (1, 64), (1, 128), (1, 448), (4, 4), (2, 128)):
        r, n = 32 * mc, 32 * nc
        theta = 1 / (4 * max(64 * r, n))
        at = max(worst_log(theta, r, n, w) for w in (1, 2, 4, 8, 16, 32))
        lo, hi = 1.0, 64.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if max(worst_log(mid * theta, r, n, w) for w in (1, 2, 4, 8, 32)) <= 0 else (lo, mid)
        print(json.dumps({"worst": {"m_c": mc, "n_c": nc, "log_moment_at_point": round(at, 4), "holds": at <= 1e-12,
                                    "breaks_at_x_point": round(lo, 2)}}), flush=True)


if __name__ == "__main__":
    main()
