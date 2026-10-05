"""The logUp reference proves, verifies, rejects tampering, and rejects a wrong multiset (laptop, < 5 s)."""
import random

import pytest

from packed import logup_reference as lu
from packed import reference as ref


def _instance(n_wit: int, t_rows: int, seed: int):
    rng = random.Random(seed)
    table = list(range(t_rows))
    queries = [rng.randrange(t_rows) for _ in range(n_wit)]
    z = ref.random_ext(rng)
    mult = lu.multiplicities(queries, table)
    p, q, n = lu.leaves(queries, table, mult, z)
    return p, q, n


@pytest.mark.parametrize("n_wit,t_rows", [(5, 3), (48, 16), (100, 8)])
def test_prove_verify_and_leaf_claim(n_wit, t_rows):
    p, q, n = _instance(n_wit, t_rows, 7 + n_wit)
    tr, claim = lu.prove_table(p, q, n, lu.Coins(99))
    claim_v = lu.verify_table(n, tr, lu.Coins(99))
    assert claim == claim_v
    assert lu.check_leaves(p, q, claim)


def test_tampered_round_rejected():
    p, q, n = _instance(48, 16, 3)
    tr, _ = lu.prove_table(p, q, n, lu.Coins(5))
    tr.levels[3]["rounds"][1][0] = ref.add(tr.levels[3]["rounds"][1][0], ref.ONE)
    with pytest.raises(ValueError):
        lu.verify_table(n, tr, lu.Coins(5))


def test_wrong_multiplicity_has_nonzero_root():
    rng = random.Random(1)
    table = list(range(8))
    queries = [rng.randrange(8) for _ in range(20)]
    mult = lu.multiplicities(queries, table)
    mult[0] += 1
    p, q, n = lu.leaves(queries, table, mult, ref.random_ext(rng))
    with pytest.raises(ValueError):
        lu.prove_table(p, q, n, lu.Coins(2))


def test_multi_column_leaves_verify():
    rng = random.Random(11)
    table = [[i, (3 * i + 1) % 7, i * i % 5] for i in range(12)]
    queries = [table[rng.randrange(12)] for _ in range(30)]
    idx = {tuple(r): i for i, r in enumerate(table)}
    mult = [0] * 12
    for row in queries:
        mult[idx[tuple(row)]] += 1
    p, q, n = lu.leaves_multi(queries, table, mult, ref.random_ext(rng), ref.random_ext(rng))
    tr, claim = lu.prove_table(p, q, n, lu.Coins(4))
    assert lu.verify_table(n, tr, lu.Coins(4)) == claim
