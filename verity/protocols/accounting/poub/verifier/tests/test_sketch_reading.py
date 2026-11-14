"""sketch-sha512's rank bound (`timing.sketch_reading`, `PROTOCOL.md` §4) against enumeration: the F2 count on every
small matrix over F2, the ±1 bound on every small ±1 matrix's rank over Q, its exactness at two rows, the readings and
thresholds the protocol states, and a grinding try's compressions against the reference's own hashes."""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
import random
from fractions import Fraction

import pytest

from verity.commitments import MerkleTree, scheme
from verity.commitments.randomness import derive
from verity.protocols.accounting.poub.verifier import audit, ledger, receipts, timing

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
    (the deployed 16 × 4096 number is out of any sampler's reach): 10,000 samples a check at seed
    20261014.  The F2 formula is exact, so the empirical CDF matches it two-sided; the ±1 bound and the union bound over
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


def threshold(rows: int, bits: int) -> int:
    """The most compressions a session's sketch failure stays at most `2^-bits` at."""
    lo, hi = 0, 2 ** 256
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if timing.sketch_failure(0, rows, queries=mid) <= Fraction(1, 2 ** bits) else (lo, mid)
    return lo


def test_the_readings_and_thresholds_the_protocol_states() -> None:
    assert timing.SECTORS == 128 and timing.sketch_reading() == timing.sketch_reading(receipts.SKETCH_ROWS)
    assert [lg(timing.sign_rank_at_most(16, r)) for r in range(15, 10, -1)] == [-16.0, -34.59, -55.39, -78.3, -103.25]
    assert [lg(timing.sign_rank_at_most(64, r)) for r in range(31, 25, -1)] == [-32.0, -66.58, -103.39, -142.3, -183.25,
                                                                                -226.23]
    for rows, star, rate, c40, c64 in ((16, 12, -95.12, 55.12, 31.12), (64, 27, -215.85, 108.5, 96.5)):
        r, f = timing.sketch_reading(rows)
        assert (r, lg(f)) == (star, rate)
        assert f == min(timing.sketch_rate(rows, x) for x in range(1, min(rows, timing.SECTOR_BYTES) + 1))
        assert f == Fraction(1, 2 ** (8 * r)) + 128 * timing.sign_rank_at_most(rows, r - 1)
        for bits, want in ((40, c40), (64, c64)):
            c = threshold(rows, bits)
            assert (timing.sketch_failure(0, rows, queries=c) <= Fraction(1, 2 ** bits)
                    < timing.sketch_failure(0, rows, queries=c + 1))
            assert round(math.log2(c), 2) == want
            assert (c == math.floor(Fraction(1, 2 ** bits) / f)) == (rows == 16), "64 rows: the keys' birthday binds"
    assert timing.sketch_rate(16, 11) == Fraction(1, 2 ** 88) + 128 * timing.sign_rank_at_most(16, 10)
    node_day, passes = round(timing.SHA512_PER_S * 86400), 86400 * 50
    assert (timing.sketch_failure(passes, queries=node_day)
            == timing.sketch_failure(passes, r=12, queries=node_day))
    assert lg(timing.sketch_failure(passes, queries=node_day)) == -43.02
    assert timing.sketch_failure(30 * passes, queries=30 * node_day) > Fraction(1, 2 ** 40)
    blocks = 1 << 30
    guess = (timing.sketch_failure(passes, blocks=blocks, queries=node_day)
             - timing.sketch_failure(passes, queries=node_day))
    assert guess == (node_day + passes) * Fraction(blocks, 2 ** 512)
    assert lg(timing.sketch_failure(passes, blocks=blocks, queries=node_day)) == -43.02


def one_pass_session(blocks: int, passes: int, span_ns: int, samples: int) -> audit.Record:
    """What the memory line's δ reads of a paced session that reserved no append coins: `passes` one-pass windows over
    `blocks` blocks, `samples` openings a window at `audit.DELTA`, spread evenly over `span_ns`.  Its registration and
    coins are zeros, which no part of δ reads."""
    profile = audit.window_profile((blocks,), samples, audit.DELTA)
    at = [span_ns * t // passes for t in range(passes + 1)]
    return audit.Record(bytes(64), bytes(64), blocks, timing.BLOCK_BYTES, tuple(
        audit.WindowRecord(t, (blocks,), at[t], at[t + 1], at[t + 1], profile) for t in range(passes)),
        pace=audit.Pace(1), samples=samples, append_coins=0)


def test_the_sketch_charges_the_provers_budget() -> None:
    """Grinding reads no block, so `sketch_failure` charges the prover's budget, Q = 2^`audit.QUERIES_LOG2`
    compressions, whatever the node computes.  At a session of P = 2,118 one-pass windows over N = 97,517,568 blocks
    in 226.9 s, 48 openings a window, 16 rows read 2^-15.12 at Q = 2^80, and 64 rows 2^-97.00, where the keys' distance
    from uniform binds.  The memory line's δ of a record of those counts is the ledger's: every `Line.delta` of
    `ledger.memory` is the sum of `ledger.delta_parts`, the windows' profile δ at `audit.DELTA`, AuditRO's slack at
    `audit.SLACK_LOG2` a pass and the sketch at Q, the largest."""
    p, n, span_ns, samples = 2118, 97_517_568, 226_932_412_222, 48
    q = 2 ** audit.QUERIES_LOG2
    assert audit.QUERIES_LOG2 == 80
    assert timing.sketch_failure(p, blocks=n) == timing.sketch_failure(p, blocks=n, queries=q)
    assert timing.sketch_failure(p, blocks=n, queries=q // 2) < timing.sketch_failure(p, blocks=n)
    assert [lg(timing.sketch_failure(p, rows=rows, blocks=n)) for rows in (16, 64)] == [-15.12, -97.0]
    record, keys = one_pass_session(n, p, span_ns, samples), 64
    assert record.windows[0].profile.delta == audit.DELTA == 2.0 ** -101
    parts = ledger.delta_parts(record)
    assert tuple(parts) == ledger.DELTA_PARTS and max(parts, key=parts.get) == "sketch"
    assert parts["profile"] == audit.DELTA
    assert parts["slack"] == p * 2.0 ** -audit.SLACK_LOG2
    assert parts["sketch"] == ledger.sketch(record) == float(timing.sketch_failure(p, blocks=n))
    assert parts["hiding"] == 2 * 2.0 ** receipts.HM96.statistical_distance_log2(p, key_failure_log2=keys)
    total = 0.0
    for name in ledger.DELTA_PARTS:  # one float add a part in `memory`'s order, which `sum` doesn't promise
        total += parts[name]
    assert [x.delta for x in ledger.memory(record, ledger.Node.stated())] == [total] * 3
    assert round(math.log2(total), 2) == -15.12, "the ledger's, typed to check it"
    for bad in (0, -1, 1.5, 2.0 ** 80):
        with pytest.raises(ValueError, match="budget Q"):
            timing.sketch_failure(p, queries=bad)


def test_the_keys_distance_from_uniform_is_charged() -> None:
    """Two of the `Q + P` candidate seeds whose 256-bit sign keys collide share their signs, so the keys are
    within `(Q + P)²/2^257` of uniform, not uniform (`SketchHashRandom`'s ε), and `sketch_failure` charges it: 2^-97.00
    at the prover's budget of 2^80 compressions, 2^-169.94 at 12,708,215,084,432."""
    p, n = 2118, 97_517_568
    for q in (12_708_215_084_432, 2 ** audit.QUERIES_LOG2):
        rest = (q * Fraction(1, 2 ** 96)
                + (q + p) * (128 * timing.sign_rank_at_most(16, 11) + Fraction(n, 2 ** 512)))
        assert timing.sketch_failure(p, blocks=n, queries=q) - rest == Fraction((q + p) ** 2, 2 ** 257)
    assert [lg(Fraction((q + p) ** 2, 2 ** 257)) for q in (12_708_215_084_432, 2 ** 80)] == [-169.94, -97.0]


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

    tree = {n: counted(functools.partial(root, n)) for n in domains}
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
