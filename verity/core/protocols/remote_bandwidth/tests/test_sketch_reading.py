"""sketch-sha512's rank bound (`timing.sketch_reading`, `PROTOCOL.md` §4) against enumeration: the F2 count on every
small matrix over F2, the ±1 bound on every small ±1 matrix's rank over Q, its exactness at two rows, the readings and
thresholds the protocol states, and a grinding try's compressions against the reference's own hashes."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from fractions import Fraction

import pytest

from verity.core.protocols.remote_bandwidth import receipts, timing
from verity.ml.commitments import MerkleTree, scheme
from verity.ml.randomness import derive

PRIME = 1_000_003


def rank_mod(m: list[list[int]], p: int) -> int:
    m = [[x % p for x in row] for row in m]
    rank = 0
    for c in range(len(m[0])):
        pivot = next((i for i in range(rank, len(m)) if m[i][c]), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        inv = pow(m[rank][c], -1, p)
        for i in range(len(m)):
            if i != rank and m[i][c]:
                f = m[i][c] * inv % p
                m[i] = [(a - f * b) % p for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def at_most(rows: int, cols: int, values: tuple[int, ...], p: int) -> list[Fraction]:
    """P(rank ≤ r) mod p for r = 0 … min(rows, cols), over every rows × cols matrix of `values`."""
    counts = [0] * (min(rows, cols) + 1)
    for e in itertools.product(values, repeat=rows * cols):
        counts[rank_mod([list(e[i * cols:(i + 1) * cols]) for i in range(rows)], p)] += 1
    return [Fraction(sum(counts[:r + 1]), len(values) ** (rows * cols)) for r in range(len(counts))]


@pytest.mark.parametrize("rows, cols", [(1, 4), (2, 3), (3, 3), (3, 4)])
def test_the_f2_count_is_every_matrix_over_f2(rows: int, cols: int) -> None:
    want = at_most(rows, cols, (0, 1), 2)
    assert [timing.gf2_rank_at_most(rows, cols, r) for r in range(len(want))] == want
    assert timing.gf2_rank_at_most(rows, cols, -1) == 0 and timing.gf2_rank_at_most(rows, cols, len(want)) == 1


@pytest.mark.parametrize("rows, cols", [(2, 5), (3, 4), (3, 5), (4, 4)])
def test_the_sign_bound_holds_for_every_small_sign_matrix(rows: int, cols: int) -> None:
    # A ±1 matrix of at most 4 rows has every minor below 4^(4/2) = 16 in absolute value (Hadamard), so its rank mod
    # PRIME is its rank over Q.
    exact = at_most(rows, cols, (1, -1), PRIME)
    bound = [timing.sign_rank_at_most(rows, r, cols) for r in range(len(exact))]
    assert all(e <= b for e, b in zip(exact, bound))
    assert exact[0] == bound[0] == 0 and exact[-1] == bound[-1] == 1
    if rows == 2:
        assert exact == bound


@pytest.mark.parametrize("cols", [3, 5, timing.SECTOR_BYTES])
def test_the_bound_is_exact_at_two_rows(cols: int) -> None:
    """Two ±1 rows have rank 1 just when the second is ± the first."""
    assert timing.sign_rank_at_most(2, 1, cols) == Fraction(2, 2 ** cols)


def test_seeded_sampling_does_not_falsify_the_rank_tail() -> None:
    """Seeded sampling at small parameters against the formulas behind `Accounting.Assumptions.SketchRank`
    (art:d3c891e4 evaluates the deployed 16 × 4096 number, out of any sampler's reach): 10,000 draws a check at seed
    20261014.  The F2 law is exact, so the empirical CDF matches it two-sided; the ±1 bound and the union bound over
    sectors (the assumption's `∃ u` event) are upper bounds that sampling sits below, up to noise, and they are tight
    here (at 4 × 5 the bound on rank ≤ 2 is exactly 0.02588 by enumeration), so a wrong formula shows as systematic
    excess.  The margins come from a sweep of seeds 0..19 at this size: worst |emp − exact| 0.0025 (F2 at rank ≤ 6)
    and 0.0115 (≤ 7), worst excess over the bound 0.0007 (5 × 7 at rank ≤ 3) and 0.0066 (∃u over 3 sectors of 4 × 5),
    each at most about half its margin."""
    rng, n = random.Random(20261014), 10_000
    f2 = [0, 0]
    for _ in range(n):
        r = rank_mod([[rng.getrandbits(1) for _ in range(10)] for _ in range(8)], 2)
        f2[0] += r <= 6
        f2[1] += r <= 7
    assert abs(f2[0] / n - timing.gf2_rank_at_most(8, 10, 6)) < 0.005
    assert abs(f2[1] / n - timing.gf2_rank_at_most(8, 10, 7)) < 0.022
    # A ±1 matrix of at most 8 rows has every minor at most 8^4 (Hadamard), so its rank mod PRIME is its rank over Q.
    tails = [0, 0, 0]
    for _ in range(n):
        tails[0] += rank_mod([[rng.choice((1, -1)) for _ in range(7)] for _ in range(5)], PRIME) <= 3
        tails[1] += rank_mod([[rng.choice((1, -1)) for _ in range(10)] for _ in range(8)], PRIME) <= 6
        tails[2] += any(rank_mod([[rng.choice((1, -1)) for _ in range(5)] for _ in range(4)], PRIME) < 3
                        for _ in range(3))
    assert tails[0] / n <= timing.sign_rank_at_most(5, 3, 7) + 0.004
    assert tails[1] / n <= timing.sign_rank_at_most(8, 6, 10) + 0.004
    assert tails[2] / n <= 3 * timing.sign_rank_at_most(4, 2, 5) + 0.012


def lg(x: Fraction) -> float:
    return round(timing.log2(x), 2)


def test_the_readings_and_thresholds_the_protocol_states() -> None:
    assert timing.SECTORS == 128 and timing.sketch_reading() == timing.sketch_reading(receipts.SKETCH_ROWS)
    assert [lg(timing.sign_rank_at_most(16, r)) for r in range(15, 10, -1)] == [-16.0, -34.59, -55.39, -78.3, -103.25]
    assert [lg(timing.sign_rank_at_most(64, r)) for r in range(31, 25, -1)] == [-32.0, -66.58, -103.39, -142.3, -183.25,
                                                                                -226.23]
    for rows, star, rate, c40, c64 in ((16, 12, -95.12, 55.12, 31.12), (64, 27, -215.85, 175.85, 151.85)):
        r, f = timing.sketch_reading(rows)
        assert (r, lg(f)) == (star, rate)
        assert f == min(timing.sketch_rate(rows, x) for x in range(1, min(rows, timing.SECTOR_BYTES) + 1))
        assert f == Fraction(1, 2 ** (8 * r)) + 128 * timing.sign_rank_at_most(rows, r - 1)
        for bits, want in ((40, c40), (64, c64)):
            c = math.floor(Fraction(1, 2 ** bits) / f)
            assert timing.sketch_failure(c, 0, rows) <= Fraction(1, 2 ** bits) < timing.sketch_failure(c + 1, 0, rows)
            assert round(math.log2(c), 2) == want
    assert timing.sketch_rate(16, 11) == Fraction(1, 2 ** 88) + 128 * timing.sign_rank_at_most(16, 10)
    node_day, passes = round(timing.SHA512_PER_S * 86400), 86400 * 50
    assert timing.sketch_failure(node_day, passes) == timing.sketch_failure(node_day, passes, r=12)
    assert lg(timing.sketch_failure(node_day, passes)) == -43.02
    assert timing.sketch_failure(30 * node_day, 30 * passes) > Fraction(1, 2 ** 40)


class Counting:
    """hashlib's SHA-512, counting `⌈(bytes + 17)/128⌉` compressions a digest."""

    count = 0
    real = hashlib.sha512

    def __init__(self, data: bytes = b"", **_: object) -> None:
        self.h, self.n = Counting.real(data), len(data)

    def update(self, data: bytes) -> None:
        self.h.update(data)
        self.n += len(data)

    def copy(self) -> Counting:
        c = Counting.__new__(Counting)
        c.h, c.n = self.h.copy(), self.n
        return c

    def digest(self) -> bytes:
        Counting.count += (self.n + 17 + 127) // 128
        return self.h.digest()


def counted(f) -> int:
    Counting.count = 0
    f()
    return Counting.count


def test_a_grinding_try_costs_what_the_reference_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    seed, x, salt = bytes(range(64)), bytes(64), bytes(range(receipts.SALT_BYTES))
    domains = {n: receipts.domain(seed, receipts.Plan.whole(receipts.PER_LEAF * n)) for n in (1, 2, 4, 8)}
    monkeypatch.setattr(hashlib, "sha512", Counting)
    monkeypatch.setitem(scheme._SHA, "sha512", Counting)
    hm96, key = receipts.HM96, receipts.KEY
    hide = counted(lambda: hm96.tree_leaf(key, hm96.commit_string(key, x, salt)))
    def root(n: int) -> bytes:
        return MerkleTree(domains[n], {c: bytes(64) for c in range(n)}, lambda _c: receipts.SCHEMA).commitment.root

    tree = {n: counted(lambda: root(n)) for n in domains}
    leaf, node = tree[1], 3
    assert all(tree[n] == leaf * n + node * (n - 1) for n in tree)
    next_seed = counted(lambda: receipts.seed(bytes(receipts.CHALLENGE_BYTES), 1, seed))
    context = {"tag": receipts.SKETCH_SHA512, "k": receipts.SKETCH_ROWS, "n": timing.BLOCK_BYTES}
    signs = counted(lambda: derive(seed, receipts.SIGN_DOMAIN, context))
    assert (hide, leaf, next_seed, signs) == (9, 2, 2, 2)
    assert counted(lambda: next(derive(seed, receipts.SIGN_DOMAIN, context).blocks(("row", 0)))) == signs + 1
    for n in (1, 2, 4, 8, 2 ** 17):
        assert timing.grind_compressions(n) == hide + leaf + node * math.ceil(math.log2(n)) + next_seed + signs
    assert (timing.grind_compressions(1), timing.grind_compressions(2 ** 21 // receipts.PER_LEAF)) == (15, 66)
