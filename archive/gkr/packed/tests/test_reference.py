"""The packed prover emits the naive prover's transcript bit for bit, and the verifier accepts it (pure Python, < 30 s)."""
import random

import pytest
from packed import reference as ref


def _instance(n: int, seed: int):
    rng = random.Random(seed)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    g = [rng.randrange(ref.P) for _ in range(1 << n)]
    h = [rng.randrange(ref.P) for _ in range(1 << n)]
    return gammas, g, h


def test_field_basics():
    rng = random.Random(1)
    a, b, c = (ref.random_ext(rng) for _ in range(3))
    assert ref.mul(a, ref.ONE) == a
    assert ref.mul(a, b) == ref.mul(b, a)
    assert ref.mul(ref.mul(a, b), c) == ref.mul(a, ref.mul(b, c))
    assert ref.mul(a, ref.add(b, c)) == ref.add(ref.mul(a, b), ref.mul(a, c))
    x = rng.randrange(ref.P)
    assert ref.from_limbs(ref.limbs_balanced(x)) == x


@pytest.mark.parametrize("n,k", [(3, 0), (3, 1), (3, 2), (3, 3), (5, 0), (5, 2), (5, 3), (6, 4), (8, 3)])
def test_packed_matches_naive_and_verifies(n, k):
    gammas, g, h = _instance(n, 100 + n + k)
    f = ref.eq_table(gammas)
    gE, hE = [ref.base(v) for v in g], [ref.base(v) for v in h]
    rounds_naive, rs_naive, final_naive = ref.naive_prover(f, gE, hE, ref.seeded_challenger(7))
    rounds_packed, rs_packed, final_packed = ref.packed_prover(gammas, g, h, k, ref.seeded_challenger(7))
    assert rs_naive == rs_packed
    assert rounds_packed == rounds_naive
    assert final_packed == final_naive
    claim = ref.ZERO
    for a, b, c in zip(f, gE, hE):
        claim = ref.add(claim, ref.mul(ref.mul(a, b), c))
    assert ref.verify(claim, rounds_packed, rs_packed, final_packed)
    assert final_packed == (ref.eq_points(gammas, rs_packed), ref.multilinear_eval(gE, rs_packed), ref.multilinear_eval(hE, rs_packed))


def test_verifier_rejects_tampered_round():
    gammas, g, h = _instance(4, 3)
    rounds, rs, final = ref.packed_prover(gammas, g, h, 2, ref.seeded_challenger(9))
    f = ref.eq_table(gammas)
    claim = ref.ZERO
    for a, b, c in zip(f, g, h):
        claim = ref.add(claim, ref.scale(a, b * c % ref.P))
    bad = [list(S) for S in rounds]
    bad[1][2] = ref.add(bad[1][2], ref.ONE)
    assert not ref.verify(claim, bad, rs, final)
    assert not ref.verify(ref.add(claim, ref.ONE), rounds, rs, final)
