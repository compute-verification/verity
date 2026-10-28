"""`prover.plan_change` is the set difference of the two plans' chunks as runs, whatever order each plan reads its runs
in and wherever in a store of up to 2^64 − 1 blocks they are, and the verifier's `Plan.changed` takes the plan sent
before, in store order, and that change to the chunks of the plan after."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from verity.core.protocols.poub import receipts as rb  # noqa: E402

import prover  # noqa: E402

TOP = (1 << 64) - 1


def reference(before: rb.Plan, after: rb.Plan):
    a, b = set(before), set(after)
    return prover.runs_of(sorted(b - a)), prover.runs_of(sorted(a - b))


def plan(rng: random.Random, blocks: int, lo: int, hi: int, shuffled: bool) -> rb.Plan:
    """A plan of some of chunks lo … hi − 1 in a store of `blocks`: in store order, or cut at random places (within a
    run as well as between them) and read in a random order."""
    order = sorted(rng.sample(range(lo, hi), rng.randrange(1, hi - lo + 1)))
    if not shuffled:
        return rb.Plan.of(order, blocks)
    cuts = sorted(rng.sample(range(1, len(order)), rng.randrange(0, len(order)))) if len(order) > 1 else []
    pieces = [order[a:b] for a, b in zip([0, *cuts], [*cuts, len(order)])]
    rng.shuffle(pieces)
    return rb.Plan.of([j for piece in pieces for j in piece], blocks)


def check(before: rb.Plan, after: rb.Plan) -> None:
    added, removed = prover.plan_change(before, after)
    assert (added, removed) == reference(before, after), (before, after)
    # `changed` takes a removed run only inside one of the plan's runs, so the plan sent before is in store order, as
    # every plan `changed` makes is.
    if list(before) == sorted(before):
        grown = before.changed(added, removed, after.blocks)
        assert set(grown) == set(after)
        if list(after) == sorted(after):
            assert grown == after


def test_a_plan_change_is_the_set_difference_of_plans_in_store_order():
    rng = random.Random(1)
    for _ in range(300):
        n, m = rng.randrange(1, 80), rng.randrange(0, 30)
        check(plan(rng, n, 0, n, False), plan(rng, n + m, 0, n + m, False))


def test_a_plan_change_is_the_set_difference_of_plans_read_in_any_order():
    rng = random.Random(2)
    for _ in range(300):
        n, m = rng.randrange(1, 80), rng.randrange(0, 30)
        check(plan(rng, n, 0, n, rng.random() < 0.5), plan(rng, n + m, 0, n + m, True))


def test_a_plan_change_holds_at_the_top_of_the_largest_store():
    rng = random.Random(3)
    for lo in (0, (1 << 63) - 40, TOP - 80):
        for _ in range(50):
            check(plan(rng, TOP, lo, lo + 80, rng.random() < 0.5), plan(rng, TOP, lo, lo + 80, rng.random() < 0.5))


def test_a_plan_unchanged_or_replaced_whole_is_an_empty_or_a_whole_change():
    whole = rb.Plan.whole(16)
    assert prover.plan_change(whole, whole) == ((), ())
    odd, even = rb.Plan.of(range(1, 16, 2), 16), rb.Plan.of(range(0, 16, 2), 16)
    assert prover.plan_change(odd, even) == (tuple((j, 1) for j in range(0, 16, 2)), tuple((j, 1) for j in range(1, 16, 2)))
    assert prover.plan_change(odd, whole) == (tuple((j, 1) for j in range(0, 16, 2)), ())
