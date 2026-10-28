"""The upstream certificate's checker (`circuit_properties.upstream`) on circuits small enough to decide by hand: a use that
is certified, the two ways a certificate fails (a required gate reaching the finishing set only through a gate outside
the region, and a finishing set reading no gate of the region), an output, the checks `Consumers` and `MulBlocks` make,
and overlap weights. The one-sweep reachability is checked against `FeedsWithin`'s definition on random wirings."""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction

import pytest
from circuit_properties import upstream as U

#: one token's norm and two products: sources 0, 1 (the previous layer), registered weights 2, 3; 4 = Σx², 5 and 6 the
#: normalized x0 and x1; products 7 = 5·2 and 8 = 6·3 (m(0, 0) and m(1, 0)), and 9 = 7 + 8
NORM = {0: (), 1: (), 2: (), 3: (), 4: (0, 1), 5: (0, 4), 6: (1, 4), 7: (2, 5), 8: (3, 6), 9: (7, 8)}
THETA = frozenset({2, 3})
BLOCKS = U.MulBlocks(P=((0, 0), (1, 0)), m={(0, 0): frozenset({7}), (1, 0): frozenset({8})}, o={(0, 0): 7, (1, 0): 8})
REGION = frozenset({4, 5, 6})


def test_norm_use_is_certified_by_its_shared_statistic():
    B = BLOCKS
    B.check(NORM)
    K = U.of_uses(B, [0], [REGION], [frozenset({4})], THETA, names=["x"])
    assert K.fins == ((frozenset({7}), frozenset({8})),)
    (v,) = U.upstream(NORM, K)
    assert v.holds and v.fins == 2 and v.classes == 2 and v.required == 1 and v.region == 3
    assert U.holds(NORM, K)
    assert U.required(NORM, REGION, [U.into(NORM, F) for F in K.fins[0]]) == {4}


def test_a_gate_one_product_never_reads_is_not_required():
    K = U.of_uses(BLOCKS, [0], [REGION], [frozenset({4, 5})], THETA)
    (v,) = U.upstream(NORM, K)
    assert not v.holds
    (bad,) = v.violations
    assert (bad.gate, bad.fin_into, bad.fins) == (5, frozenset({3, 6}), 1)
    assert "reaches no gate of into F ∩ A" in bad.why


def test_a_path_through_a_gate_outside_the_region_does_not_count():
    # 1 = a(0), 2 = b(1), 3 = c(2), 4 = c'(0), and the product 5 reads 3 and 4; gate 2 is not in the region
    reads = {0: (), 1: (0,), 2: (1,), 3: (2,), 4: (0,), 5: (3, 4)}
    A = frozenset({1, 3, 4})
    assert not U.feeds_within(reads, A, 1, 3) and U.feeds_within(reads, A | {2}, 1, 3)
    K = U.Consumers(A=(A,), G=(frozenset({1}),), theta=frozenset(), fins=((frozenset({5}),),))
    (v,) = U.upstream(reads, K)
    assert not v.holds and [b.gate for b in v.violations] == [1]
    assert U.required(reads, A, [U.into(reads, frozenset({5}))]) == {3, 4}
    K2 = U.Consumers(A=(A | {2},), G=(frozenset({1}),), theta=frozenset(), fins=((frozenset({5}),),))
    assert U.holds(reads, K2)


def test_a_finishing_set_reading_nothing_of_the_region_fails_every_required_gate():
    A = frozenset({4, 5})
    K = U.of_uses(BLOCKS, [0], [A], [frozenset({4})], THETA)
    (v,) = U.upstream(NORM, K)
    assert not v.holds
    (bad,) = v.violations
    assert bad.gate is None and bad.fin_into == {3, 6} and "reads no gate of the region" in bad.why
    assert U.required(NORM, A, [U.into(NORM, F) for F in K.fins[0]]) == frozenset()


def test_vacuous_without_required_gates_or_finishing_sets():
    K = U.Consumers(A=(frozenset({4}), REGION), G=(frozenset(), frozenset({4})), theta=THETA,
                    fins=((frozenset({8}),), ()))
    assert [v.holds for v in U.upstream(NORM, K)] == [True, True]


def test_a_gradient_output_and_its_partial_sums():
    # x0, x1, x2 (0..2), d0, d1, d2 (3..5); products 6, 7, 8; partial sums 9 = 6 + 7 and the output 10 = 9 + 8
    reads = {0: (), 1: (), 2: (), 3: (), 4: (), 5: (), 6: (0, 3), 7: (1, 4), 8: (2, 5), 9: (6, 7), 10: (9, 8)}
    A = frozenset({6, 7, 8, 9})
    K = U.of_outputs([10], [A], [A], frozenset())
    assert K.fins == ((frozenset({10}),),)
    assert U.holds(reads, K)
    assert U.required(reads, A, [U.into(reads, frozenset({10}))]) == A
    split = U.of_outputs([10], [frozenset({6, 7, 8})], [frozenset({6, 7, 8})], frozenset())
    (v,) = U.upstream(reads, split)
    assert not v.holds and [b.gate for b in v.violations] == [6, 7]


def test_finishing_sets_by_their_into_agree_with_the_sets():
    K = U.of_uses(BLOCKS, [0], [REGION], [frozenset({4})], THETA)
    counts = Counter(U.into(NORM, F) for F in K.fins[0])
    K2 = U.Consumers(A=K.A, G=K.G, theta=THETA, fin_into=(counts,))
    assert U.upstream(NORM, K) == U.upstream(NORM, K2)
    many = U.Consumers(A=K.A, G=K.G, theta=THETA, fin_into=(Counter({frozenset({2, 5}): 3, frozenset({3, 6}): 4}),))
    (v,) = U.upstream(NORM, many)
    assert v.holds and v.fins == 7 and v.classes == 2


def test_consumers_refuses_what_lean_would_not_build():
    with pytest.raises(U.NotConsumers, match="G_sub"):
        U.of_uses(BLOCKS, [0], [REGION], [frozenset({4, 7})], THETA)
    with pytest.raises(U.NotConsumers, match="A_θ"):
        U.of_uses(BLOCKS, [0], [REGION | {2}], [frozenset({4})], THETA)
    with pytest.raises(U.NotConsumers):
        U.Consumers(A=(REGION,), G=(frozenset(),), theta=THETA)
    with pytest.raises(U.NotConsumers, match="no reads for gate 99"):
        U.into(NORM, {99})


def test_mul_blocks_checks():
    U.check_wiring(NORM)
    with pytest.raises(ValueError, match="not an earlier one"):
        U.check_wiring({0: (), 1: (2,), 2: ()})
    with pytest.raises(U.NotMulBlocks, match="m_disjoint"):
        U.MulBlocks(P=((0, 0), (1, 0)), m={(0, 0): frozenset({7}), (1, 0): frozenset({7, 8})},
                    o={(0, 0): 7, (1, 0): 8}).check(NORM)
    with pytest.raises(U.NotMulBlocks, match="m_computed"):
        U.MulBlocks(P=((0, 0),), m={(0, 0): frozenset({2, 7})}, o={(0, 0): 7}).check(NORM)
    with pytest.raises(U.NotMulBlocks, match="o_mem"):
        U.MulBlocks(P=((0, 0),), m={(0, 0): frozenset({7})}, o={(0, 0): 8}).check(NORM)
    with pytest.raises(U.NotMulBlocks, match="reach"):
        U.MulBlocks(P=((0, 0),), m={(0, 0): frozenset({7, 8})}, o={(0, 0): 7}).check(NORM)


def test_overlap_weights():
    A = [frozenset({1, 2}), frozenset({2, 3}), frozenset({4})]
    assert U.multiplicity(A) == Counter({1: 1, 2: 2, 3: 1, 4: 1})
    assert U.overlap_weights(A, [Fraction(1)] * 3) == (False, 2)
    w = U.weights_by_multiplicity(A)
    assert w == [Fraction(1, 2), Fraction(1, 2), Fraction(1)]
    assert U.overlap_weights(A, w) == (True, None)
    assert U.overlap_weights(A, [Fraction(1), Fraction(0), Fraction(1)]) == (True, None)
    assert U.overlap_weights(A, [Fraction(-1), Fraction(1), Fraction(1)]) == (False, None)


def test_reaching_is_feeds_within():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 24)
        reads = {g: tuple(sorted(rng.sample(range(g), rng.randint(0, min(g, 3))))) for g in range(n)}
        R = frozenset(g for g in range(n) if rng.random() < 0.6)
        targets = [frozenset(rng.sample(range(n), rng.randint(0, min(n, 3)))) for _ in range(rng.randint(1, 3))]
        mask = U.reaching(reads, R, targets)
        for g in range(n):
            want = sum(1 << k for k, T in enumerate(targets) if any(U.feeds_within(reads, R, g, a) for a in T))
            assert mask.get(g, 0) == want, (reads, R, targets, g)
