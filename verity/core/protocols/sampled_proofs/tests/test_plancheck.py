"""Lean's check of the planner's one-stage designs (`plancheck`, the `sampled-proofs-plancheck` executable): it accepts
the planner's leak and work designs at Figure 1's scales, certifying at most a little more than the planner's
`b K + locBits(n, K)`, and refuses each design with its rate a hair below the planner's rule, so its acceptance is not
vacuous; a design it can't read is an error."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from verity.core.protocols.sampled_proofs import plancheck as P
from verity.core.protocols.sampled_proofs.one_stage.verifier import LeanError
from verity.core.protocols.sampled_proofs.optimize import Grain, leak_count, loc_bits

DELTA = 0.01
#: proof blocks, the widest one's output bits, and leak targets, from a thousand blocks to a world year's
LEAKS = [(1000, 8, 200.0), (10**6, 64, 1e5), (10**12, 4096, 1e9), (4 * 10**17, 65536, 1e15)]


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
        planner = b * k + loc_bits(n, k)
        assert 0 < k < n and int(d["count"]) == k
        assert bound is not None and planner <= bound <= planner * 1.01, (n, b, ell)


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
