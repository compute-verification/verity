"""Lean's check of the planner's designs (`plancheck`, the `sampled-proofs-plancheck` executable): it accepts the
planner's one-stage leak and work designs at Figure 1's scales, certifying a little more than the planner's
`b K + loc_bits(n, K)` in exact arithmetic, and refuses each design with its rate a hair below the planner's rule, so
its acceptance is not vacuous; a two-stage point's bits bound its count by enumeration on a small design; a design it
can't read is an error."""
from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

import pytest

from verity.protocols.spot import plancheck as P
from verity.protocols.spot.verifier.one_stage.verifier import LeanError
from verity.protocols.spot.optimize import Cell, Grain, leak_count, two_stage_rankin

DELTA = 0.01
#: proof blocks, the widest one's output bits, and leak targets, from a thousand blocks to a world year's
LEAKS = [(1000, 8, 200.0), (10**6, 64, 1e5), (10**12, 4096, 1e9), (4 * 10**17, 65536, 1e15)]


def _exact_bound(n: int, b: int, k: int) -> Decimal:
    """The planner's `b K + loc_bits(n, K)` to 60 digits: `n H₂(K/n)`, or `n` once `2K ≥ n`."""
    with localcontext() as c:
        c.prec = 60
        if 2 * k >= n:
            return Decimal(b * k + n)
        n_, k_ = Decimal(n), Decimal(k)
        return b * k_ + (k_ * (n_ / k_).ln() + (n_ - k_) * (n_ / (n_ - k_)).ln()) / Decimal(2).ln()


def _lower(d: dict, by: Fraction) -> dict:
    """`d` with its rate (a leak design's) or its one stratum's rate (a work design's) times `1 - by`."""
    if d["kind"] == "leak":
        num, den = map(int, d["q"].split("/"))
        return d | {"q": f"{Fraction(num, den) * (1 - by)}"}
    (w, num), = d["strata"]
    return d | {"strata": [[w, str(int(num) * (2**20 - 1) // 2**20)]]}


def test_the_planners_leak_designs_certify_about_its_bound():
    designs = [P.leak_design(Grain("g", n, 1.0, b, 0.0, 1.0), ell, DELTA) for n, b, ell in LEAKS]
    for (n, b, ell), d, bound in zip(LEAKS, designs, P.check(designs)):
        k = leak_count(Grain("g", n, 1.0, b, 0.0, 1.0), ell)
        exact = _exact_bound(n, b, k)
        assert 0 < k < n and int(d["count"]) == k and bound is not None
        assert exact <= Decimal(bound.numerator) / bound.denominator <= exact * Decimal("1.01"), (n, b, ell)


def test_a_leak_design_below_the_planners_rate_is_refused():
    designs = [P.leak_design(Grain("g", n, 1.0, b, 0.0, 1.0), ell, DELTA) for n, b, ell in LEAKS]
    assert P.check([_lower(d, Fraction(1, 2**10)) for d in designs]) == [None] * len(designs)


def test_sampling_every_block_certifies_none():
    assert P.check([P.leak(64, 8, Fraction(1), 0, DELTA)]) == [Fraction(0)]


@pytest.mark.parametrize("n,work,target", [(1000, 1.0, 50.0), (10**9, 3.5e9, 1e17), (10**15, 2.8e10, 1e24)])
def test_the_planners_work_design_meets_its_target_and_no_lower_rate_is_checked(n, work, target):
    d = P.work_design(Grain("g", n, work, 1, 0.0, 1.0), target, DELTA)
    assert P.check([d, _lower(d, Fraction(1, 2**20))]) == [True, False]


def test_a_rate_rounds_up_by_the_slack_and_a_bound_up_to_a_float():
    assert P.rate_up(0.5) == Fraction(1, 2) * (1 + P.SLACK) and P.rate_up(1.0) == 1
    assert P.exact(0.01) == Fraction(1, 100)
    x = Fraction(1, 3)
    assert Fraction(P.upper(x)) >= x and math.nextafter(P.upper(x), 0) < x


def test_a_line_lean_cant_read_is_an_error():
    with pytest.raises(LeanError, match="can't read"):
        P.check([{"kind": "leak", "units": "ten"}])


def _admissible_count(cells: list[Cell], k: dict[int, int], replay_blocks: int, q: float, delta: float,
                      served: int | None = None, union: bool = True) -> int:
    """`CompiledTwo.Exfil`'s count by enumeration: Σ 2^{b(T)} over the sets T of proof blocks whose acceptance
    ∏_ρ (1 − q(1 − e_ρ)) is above δ, or under `served`, each replay block read by one request serving `served` bits,
    the sum over their patterns (`union`) or Σ 2^{Σ_ρ min(b(T_ρ), served)}, the capped sets' count."""
    blocks = [(c.stratum, int(c.bits)) for c in cells for _ in range(int(c.n))]
    n = {t: sum(1 for s, _ in blocks if s == t) for t in k}

    def esc(sub: tuple[int, ...]) -> float:
        m = {t: sum(x for x, (s, _) in zip(sub, blocks) if s == t) for t in k}
        return math.prod(math.comb(n[t] - m[t], k[t]) / math.comb(n[t], k[t]) for t in k)

    def bits(sub: tuple[int, ...]) -> int:
        return sum(b for x, (_, b) in zip(sub, blocks) if x)

    def light(sub: tuple[int, ...]) -> tuple:
        return () if not any(sub) else ("all",) if bits(sub) >= served else sub

    total, seen = 0, set()
    for T in product(product((0, 1), repeat=len(blocks)), repeat=replay_blocks):
        if math.prod(1 - q * (1 - esc(sub)) for sub in T) <= delta:
            continue
        if served is None:
            total += 2 ** sum(bits(sub) for sub in T)
        elif not union:
            total += 2 ** sum(min(bits(sub), served) for sub in T)
        elif (p := tuple(light(sub) for sub in T)) not in seen:
            seen.add(p)
            total += 2 ** sum(0 if not x else served if x == ("all",) else bits(x) for x in p)
    return total


SMALL = [([Cell(3, 2)], {0: 1}), ([Cell(2, 1), Cell(1, 4, stratum=1)], {0: 1, 1: 1})]


@pytest.mark.parametrize("cells,k", SMALL)
@pytest.mark.parametrize("served,union", [(None, False), (3, True), (3, False)])
def test_a_two_stage_points_bits_bound_its_count(cells, k, served, union):
    """Two replay blocks of a few proof blocks, sampled at q = 0.9 with δ = 0.3, which leaves fewer sets admissible
    than there are: the bits Lean certifies, at the θ it finds from the planner's, are at least log2 of the count by
    enumeration, in the sets', the patterns' and the capped sets' form; and no more than at θ = 0, which the search
    tries."""
    _, theta = two_stage_rankin(cells, k, 2, .9, .3, served, union)
    found, zero = P.check([P.two(cells, k, 2, .9, .3, served, union, theta or 1.0),
                           P.two(cells, k, 2, .9, .3, served, union, 0.0, search=None)])
    count = _admissible_count(cells, k, 2, .9, .3, served, union)
    assert count < _admissible_count(cells, k, 2, 0.0, .3, served, union)
    assert found[0] is not None and zero[0] is not None and found[1] > 0
    assert math.log2(count) <= found[0] <= zero[0] * (1 + 1e-9), (count, found, zero)


def test_a_two_stage_point_out_of_range_is_refused():
    d = P.two([Cell(3, 2)], {0: 1}, 2, .5, .1, theta=1.0, search=None)
    assert P.check([d | {"q": "3/2"}, d | {"delta": "0/1"}]) == [(None, 1), (None, 1)]
    assert P.check([d | {"form": "capped", "N": "1", "S": "3", "theta": "-1/1"}]) == [(None, -1)]
