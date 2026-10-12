"""The remote window's openings: the pooled draw, its bounds against v0's deployed point, the exact multiproof
expectation against `verity.ml.commitments.multiproof` by enumeration, and the cheapest-point searches."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from verity.core.protocols.bandwidth import audit
from verity.core.protocols.bandwidth import openings as o
from verity.ml.commitments.multiproof import multiproof_cost
from verity.ml.randomness import derive


def _key(tag: bytes = b"window"):
    return derive(b"verifier source", o.DOMAIN, {"window": tag})


def test_draw_maps_the_pooled_index_pass_major():
    counts = [3, 5, 2]
    key = _key()
    picks = o.draw(key, counts, 200)
    assert picks == o.draw(key, counts, 200)
    assert picks != o.draw(_key(b"other"), counts, 200)
    ends = list(itertools.accumulate(counts, initial=0))
    for t, (p, i) in enumerate(picks):
        assert 0 <= i < counts[p]
        assert ends[p] + i == key.uniform(sum(counts), ("opening", t))
    assert o.draw(key, counts, 0) == ()


def test_draw_weights_passes_by_their_receipts():
    picks = o.draw(_key(), [1, 3], 4000)
    first = sum(p == 0 for p, _ in picks)
    assert abs(first - 1000) < 4 * math.sqrt(4000 * 0.25 * 0.75)


@pytest.mark.parametrize("counts, s", [([], 1), ([0, 2], 1), ([2, True], 1), ([2], -1), ([2], 1.0)])
def test_draw_refuses_a_malformed_window(counts, s):
    with pytest.raises(ValueError):
        o.draw(_key(), counts, s)


@pytest.mark.parametrize("blocks", [1000, 2**10, 2**19])
@pytest.mark.parametrize("error_log2", [-20, -40, -42])
def test_samples_agree_with_v0(blocks, error_log2):
    assert o.samples_for(1 / 64, error_log2, blocks) == audit.samples_for(blocks, error_log2, Fraction(1, 64))
    assert o.samples_for(1 / 64, error_log2) >= o.samples_for(1 / 64, error_log2, blocks)


def test_window_error_is_v0s_sampling_term():
    n = 2**19
    k = audit.certified_blocks(n)
    assert o.window_error(n, 1 / 64, audit.SAMPLES) == pytest.approx(math.exp(audit.SAMPLES * math.log((k - 1) / n)))
    assert o.window_error(n, 1 / 64, audit.SAMPLES) <= (63 / 64) ** audit.SAMPLES


def test_crude_slack_is_v0s():
    n = 2**19
    k = audit.certified_blocks(n)
    crude = o.slack_bits(o.Slack.CRUDE, queries_log2=audit.QUERIES_LOG2, blocks=n, certified=k,
                         error_log2=-audit.SLACK_LOG2)
    assert crude == audit.QUERIES_LOG2 * k + audit.SLACK_LOG2


def test_exact_slack_bounds_the_binomial():
    k = 1000
    small = o.slack_bits(o.Slack.EXACT, queries_log2=20, blocks=2**30, certified=k, error_log2=-41)
    assert small == pytest.approx(math.log2(math.comb(2**20, k)) + 41, rel=1e-12)
    exact = math.comb(2**64, k)
    exact_bits = exact.bit_length() - 1 + math.log2(exact / 2 ** (exact.bit_length() - 1))
    big = o.slack_bits(o.Slack.EXACT, queries_log2=64, blocks=2**30, certified=k, error_log2=0)
    assert 0 <= big - exact_bits <= math.log2(2 * math.pi * k) / 2 + 1


def test_slack_forms_are_ordered():
    blocks = 2**27
    k = math.ceil((1 - 0.0015) * blocks)
    kw = dict(queries_log2=64, blocks=blocks, certified=k, error_log2=-41)
    named, exact, crude = (o.slack_bits(f, **kw) for f in (o.Slack.NAMED, o.Slack.EXACT, o.Slack.CRUDE))
    assert named < exact < crude
    assert named == pytest.approx(o._log2_choose(blocks, blocks - k) + 41)
    assert named / blocks < 0.0015 * math.log2(math.e / 0.0015) * 1.01


def test_horizon_bound_is_the_adaptive_optimum():
    s, d = 50, 0.02
    best = max((1 - d1) ** s * (1 - (2 * d - d1)) ** s for d1 in (i / 1000 for i in range(0, 41)))
    assert best == pytest.approx(2.0 ** o.horizon_error_log2(s, 2, d))
    assert o.horizon_error_log2(s, 1, d) == pytest.approx(math.log2((1 - d) ** s))


def test_horizon_counts_only_total_openings():
    assert o.horizon_deficit(100, 10, -40) == pytest.approx(o.horizon_deficit(1000, 1, -40))
    d = o.horizon_deficit(2760, 7, -40)
    assert o.horizon_error_log2(2760, 7, d) == pytest.approx(-40)
    assert o.horizon_deficit(2760, 1, -40) < 0.01 and o.horizon_deficit(2760, 7, -40) < 0.0015


def _rfc6962_siblings(count: int, ranks: set[int]) -> int:
    def walk(lo: int, hi: int) -> tuple[bool, int]:
        if hi - lo == 1:
            return lo in ranks, 0
        mid = lo + (1 << ((hi - lo - 1).bit_length() - 1))
        a, sa = walk(lo, mid)
        b, sb = walk(mid, hi)
        return a or b, sa + sb + (a != b)

    return walk(0, count)[1]


@pytest.mark.parametrize("leaves, universe, s", [(1, 3, 2), (2, 2, 3), (3, 5, 3), (5, 5, 3), (6, 9, 3), (8, 8, 3),
                                                  (7, 10, 3), (12, 12, 2)])
def test_multiproof_expectation_is_exact(leaves, universe, s):
    """Every draw of `s` from `universe` equally likely outcomes, `leaves` of them this tree's: the mean multiproof
    over all draws is `expected_siblings`, and the audit's RFC 6962 tree never needs more."""
    siblings = hashes = distinct = 0
    for picks in itertools.product(range(universe), repeat=s):
        ranks = {r for r in picks if r < leaves}
        if ranks:
            c = multiproof_cost(leaves, ranks)
            siblings += c.siblings
            hashes += c.node_hashes
            distinct += len(ranks)
            assert _rfc6962_siblings(leaves, ranks) <= c.siblings
    n = universe**s
    assert o.expected_siblings(leaves, 1 / universe, s) == pytest.approx(siblings / n)
    assert o.expected_node_hashes(leaves, 1 / universe, s) == pytest.approx(hashes / n)
    assert o.expected_known(leaves, 1 / universe, s)[0] == pytest.approx(distinct / n)


def test_one_opening_sends_one_path_per_tree():
    g = o.Geometry(stores=2, blocks=1024, passes=4, block_bytes=4096)
    c = o.opening_cost(g, 1)
    assert (c.block, c.store_path, c.leaf, c.receipt_path) == pytest.approx((4096, 10 * 64, 64, 10 * 64))
    row = o.compressions(o.ROW_PREFIX_BYTES + 4096) + o.compressions(o.STORE_LEAF_BYTES)
    nodes = 2 * 10 + 2 * 10
    receipt = o.compressions(o.RECEIPT_PREFIX_BYTES + 64) + o.compressions(o.RECEIPT_PREFIX_BYTES + 64 + 64)
    assert c.compressions == pytest.approx(row + nodes + receipt)
    assert c.squarings == pytest.approx(2 * 4096 // 64)
    assert c.sketch_ops == 16 * 4096


def test_multiproofs_share_paths():
    g = o.Geometry(stores=1, blocks=2**12, passes=2, block_bytes=4096)
    one, many = o.opening_cost(g, 1), o.opening_cost(g, 4096)
    assert many.store_path < one.store_path / 2
    assert many.block < one.block
    packed = o.opening_cost(o.Geometry(stores=1, blocks=2**12, passes=2, block_bytes=4096, receipts_per_leaf=16), 1)
    assert packed.leaf == pytest.approx(16 * 64) and packed.receipt_path == pytest.approx(8 * 64)


NODE = o.Geometry(stores=8, blocks=94 * 10**9 // 4096, passes=10, block_bytes=4096)


def test_point_shares_the_slack_error_over_passes_and_counts():
    g = o.Geometry(stores=2, blocks=1000, passes=5, block_bytes=4096)
    p = o.point(g, 500, slack=o.Slack.CRUDE)
    k = math.ceil((1 - p.mu_sample) * 2000)
    assert p.lam == pytest.approx((64 * k + 41 + math.log2(5 * 2001)) / (8 * g.store_bytes))


def test_cheapest_is_the_least_sufficient_point():
    p = o.cheapest(NODE, 0.0015)
    assert p is not None and p.total <= 0.0015
    assert o.point(NODE, p.s - 1).total > 0.0015
    assert p.s >= math.ceil(40 * math.log(2) / -math.log1p(-0.0015))
    hybrid = o.cheapest(NODE, 0.0015, windows=7)
    assert hybrid.s * 7 == pytest.approx(p.s, abs=7)


def test_cheapest_refuses_a_target_under_the_slack():
    v0_blocks = o.Geometry(stores=8, blocks=94 * 10**9 // 2056, passes=10, block_bytes=2056)
    assert o.cheapest(v0_blocks, 0.0015, slack=o.Slack.CRUDE) is None
    assert o.cheapest(v0_blocks, 0.0015, slack=o.Slack.NAMED) is not None


def test_achievable_spends_the_budget():
    rates = o.Rates(compression=2e-7, squaring=2e-7, sketch_op=1e-10)
    target = o.point(NODE, 5000)
    p = o.achievable(NODE, bandwidth=target.window_bytes, cores=1e6, rates=rates, window_s=1.0)
    assert p.s == 5000
    q = o.achievable(NODE, bandwidth=1e12, cores=target.window_core_seconds(rates), rates=rates, window_s=1.0)
    assert q.s == 5000
    assert o.achievable(NODE, bandwidth=100.0, cores=1.0, rates=rates, window_s=1.0) is None
