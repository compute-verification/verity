"""The lead's calls at the network certifier (`calls`): the shape a call's coins take, the coins flipped after its A root
(G exactly uniform by rejection), the stream released once, and the per-call count refused past the call's bits."""

import itertools
import math

import pytest

from verity.core.protocols.certifier.calls import (
    G_BOUND, STREAM_TAG, CallRefusal, Calls, CallShape, Coins, EgressCap, Verdict, flip, stream, timing_bits,
)
from verity.ml.commitments.identity import tagged_sha512

ROOT_A, ROOT_W = bytes(range(64)), bytes(range(64, 128))
#: m 16, n 8, S 1, one tile: 2 bytes of b, 8 of a₁, 32 of G (no rejection) and 24 of lanes.
SMALL = CallShape(1, 8, 1)
SMALL_ENTROPY = 2 + 8 + 32 + 24
#: rows · n · K = 8192, so a cap of 32 bits over 8192 MACs gives it exactly 32 bits.
TIGHT = CallShape(16, 8, 64)
#: The lead's cap and its forward's MACs a token.
LEAD = EgressCap(32, 973_078_528)


class Feed:
    """Entropy from fixed bytes; `read` counts the bytes taken."""

    def __init__(self, data: bytes) -> None:
        self.data, self.read = data, 0

    def __call__(self, k: int) -> bytes:
        if self.read + k > len(self.data):
            raise RuntimeError("the feed ran out")
        self.read += k
        return self.data[self.read - k:self.read]


def g_feed(words, *, lanes=bytes(24)) -> Feed:
    """A feed for SMALL: zero bits, then G's 32-bit words, then the lanes."""
    return Feed(bytes(10) + b"".join(w.to_bytes(4, "big") for w in words) + lanes)


def weights(coins: Coins) -> list[tuple[int, ...]]:
    return [tuple(coins.G(q, j) for q in range(4)) for j in range(coins.shape.n)]


def test_shape_pads_to_mma_tiles_and_counts_k32_steps():
    assert [(s.m, s.n, s.S, s.tiles) for s in (CallShape(1, 2048, 2048), CallShape(100, 1000, 4096),
                                               CallShape(17, 13, 64))] == [(16, 2048, 74, 1), (112, 1000, 147, 2),
                                                                           (32, 16, 3, 1)]
    assert [CallShape(1, 8, K).S for K in (1, 21, 22, 49, 50)] == [1, 1, 2, 2, 3]


@pytest.mark.parametrize("bad", [(0, 8, 8), (1, 0, 8), (1, 8, 0), (-1, 8, 8), (True, 8, 8), (1.0, 8, 8)])
def test_shape_refuses_a_dimension_that_isnt_a_positive_integer(bad):
    with pytest.raises(CallRefusal):
        CallShape(*bad)


def test_a_decode_call_takes_464096_coins():
    decode = CallShape(1, 2048, 2048)
    assert decode.coins == 16 * 74 + 16 * 4 + 4 * 2048 + 74 * 3 * 2048 == 464_096
    assert len(stream(flip(decode))) == 464_096


def test_g_is_the_base_7_digits_of_the_accepted_word_less_3():
    assert weights(flip(SMALL, g_feed([0] * 8))) == [(-3, -3, -3, -3)] * 8
    assert weights(flip(SMALL, g_feed([2400] * 8))) == [(3, 3, 3, 3)] * 8
    digits = [tuple(g + 3 for g in w) for w in weights(flip(SMALL, g_feed([7, 49, 343, 2406, 2401 * 9 + 1, 0, 0, 0])))]
    assert digits[:5] == [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (5, 0, 0, 0), (1, 0, 0, 0)]


def test_g_rejects_every_word_at_or_above_the_bound_and_takes_the_next():
    feed = g_feed([0xFFFF_FFFF, G_BOUND - 1, G_BOUND, 1, 0, 0, 0, 0, 0, 0])
    got = weights(flip(SMALL, feed))
    assert got[:2] == [(3, 3, 3, 3), (-2, -3, -3, -3)]
    assert feed.read == len(feed.data)


def test_g_is_exactly_uniform():
    """The accepted words are G_BOUND / 2401 for each residue mod 2401, and `flip` takes the 2401 residues one to one onto
    [−3, 3]⁴."""
    assert G_BOUND % 7**4 == 0 and 2**32 - G_BOUND == 872 < 7**4
    wide = CallShape(1, 2401, 1)
    n = wide.n
    coins = flip(wide, Feed(bytes(10) + b"".join(w.to_bytes(4, "big") for w in [*range(7**4), *[0] * (n - 7**4)])
                            + bytes(3 * n)))
    assert set(weights(coins)[:7**4]) == set(itertools.product(range(-3, 4), repeat=4))


def test_a_lane_is_its_bytes_low_7_bits_odd_and_each_value_from_two_bytes():
    shape = CallShape(1, 88, 1)
    coins = flip(shape, Feed(bytes(10) + bytes(4 * 88) + bytes(range(256)) + bytes(8)))
    s = [coins.s(0, 0, lane, j) for lane in range(3) for j in range(88)][:256]
    assert s == [2 * (x % 128) - 127 for x in range(256)]
    assert sorted(s) == sorted([*range(-127, 128, 2)] * 2)


def test_bits_are_read_least_significant_first():
    coins = flip(SMALL, Feed(bytes([0b0000_0001, 0b1000_0000, 0b0000_0110]) + bytes(7 + 32 + 24)))
    assert list(coins.b) == [1] + [0] * 14 + [1]
    assert list(coins.a1) == [0, 1, 1] + [0] * 61


def test_the_stream_is_one_signed_byte_a_coin_in_coin_index_order():
    shape = CallShape(17, 13, 64)
    coins = flip(shape)
    m, n, S = shape.m, shape.n, shape.S
    out = [x - 256 if x > 127 else x for x in stream(coins)]
    assert len(out) == shape.coins == m * S + 4 * m + 4 * n + S * 3 * n
    b, rest = out[:m * S], out[m * S:]
    a1, rest = rest[:4 * m], rest[4 * m:]
    G, s = rest[:4 * n], rest[4 * n:]
    assert b == [coins.b[i * S + t] for i in range(m) for t in range(S)] and set(b) | set(a1) <= {0, 1}
    assert a1 == [coins.a1[i * 4 + q] for i in range(m) for q in range(4)]
    assert G == [coins.G(q, j) for q in range(4) for j in range(n)] and set(G) <= set(range(-3, 4))
    assert s == [coins.s(t, 0, lane, j) for t in range(S) for lane in range(3) for j in range(n)]
    assert all(x % 2 == 1 and -127 <= x <= 127 for x in s)


def test_coins_hold_their_sizes_and_ranges():
    good = flip(SMALL, Feed(bytes(SMALL_ENTROPY)))
    for bad in ({"g": bytes([7]) + good.g[1:]}, {"u": good.u[:-1]}, {"b": bytes([2]) + good.b[1:]},
                {"a1": bytearray(good.a1)}):
        with pytest.raises(ValueError):
            Coins(**{"shape": SMALL, "b": good.b, "a1": good.a1, "g": good.g, "u": good.u, **bad})


def test_the_coins_follow_the_a_root_and_go_out_once():
    seen: list[list[str]] = []

    def entropy(k: int) -> bytes:
        seen.append([c["a_root"] for c in calls.summary()])
        return bytes(k)

    calls = Calls(LEAD, entropy)
    for refused in (lambda: calls.words_root(0, ROOT_W), lambda: calls.a_root(0, b"short", SMALL),
                    lambda: calls.a_root(-1, ROOT_A, SMALL), lambda: calls.a_root(True, ROOT_A, SMALL),
                    lambda: calls.a_root(0, ROOT_A, (1, 8, 1))):
        with pytest.raises(CallRefusal):
            refused()
    assert seen == [] and calls.summary() == []
    out = calls.a_root(0, ROOT_A, SMALL)
    assert len(out) == SMALL.coins and seen and all(ROOT_A.hex() in held for held in seen)
    reads = len(seen)
    with pytest.raises(CallRefusal, match="once"):
        calls.a_root(0, ROOT_A, SMALL)
    assert len(seen) == reads
    assert calls.summary()[0]["coins"] == tagged_sha512(STREAM_TAG, out).hex()


def test_a_call_whose_coins_arent_flipped_is_refused_and_never_opens():
    def broken(k: int) -> bytes:
        raise OSError("no entropy")

    calls = Calls(LEAD, broken)
    with pytest.raises(OSError):
        calls.a_root(0, ROOT_A, SMALL)
    with pytest.raises(CallRefusal):
        calls.words_root(0, ROOT_W)
    with pytest.raises(CallRefusal, match="once"):
        calls.a_root(0, ROOT_A, SMALL)
    short = Calls(LEAD, lambda k: bytes(k - 1))
    with pytest.raises(ValueError, match="entropy gave"):
        short.a_root(0, ROOT_A, SMALL)
    assert calls.close() == short.close() == {0: Verdict.NO_COINS}
    assert calls.summary()[0]["coins"] is None and calls.summary()[0]["verdict"] == Verdict.NO_COINS.value


def test_once_the_entropy_fails_the_window_flips_nothing_more():
    feed = Feed(bytes(SMALL_ENTROPY))
    calls = Calls(LEAD, feed)
    with pytest.raises(RuntimeError, match="ran out"):
        calls.a_root(0, ROOT_A, CallShape(64, 64, 64))
    read = feed.read
    with pytest.raises(ValueError, match="failed earlier"):
        calls.a_root(1, ROOT_A, SMALL)
    assert feed.read == read < SMALL_ENTROPY
    assert calls.close() == {0: Verdict.NO_COINS, 1: Verdict.NO_COINS}


def test_timing_bits_are_ceil_log2_of_r_plus_1():
    assert [timing_bits(r) for r in range(9)] == [0, 1, 2, 2, 3, 3, 3, 3, 4]
    assert all(timing_bits(r) == math.ceil(math.log2(r + 1)) for r in range(1, 5000))
    for bad in (-1, 1.0, True):
        with pytest.raises(CallRefusal):
            timing_bits(bad)


def test_a_call_is_refused_once_its_count_passes_its_bits():
    calls = Calls(EgressCap(32, 16 * 8 * 64))
    calls.a_root(0, ROOT_A, TIGHT)
    assert calls.bucket(32, [26]) == []
    assert calls.bucket(0) == []
    assert calls.bucket(1) == [0]
    assert calls.bucket(1) == []
    assert calls.words_root(0, ROOT_W) is Verdict.PAST_BITS


@pytest.mark.parametrize("count,verdict", [(10, Verdict.ACCEPTED), (11, Verdict.PAST_BITS)])
def test_a_share_that_isnt_an_integer_compares_exactly(count, verdict):
    calls = Calls(EgressCap(32, 3 * 16 * 8 * 64))
    calls.a_root(0, ROOT_A, TIGHT)
    calls.bucket(1, [count - 1])
    assert calls.words_root(0, ROOT_W) is verdict


def test_the_lead_gives_a_t512x64_call_8_bits_and_a_decode_call_none():
    assert not LEAD.passed(CallShape(64, 2048, 2048), 8) and LEAD.passed(CallShape(64, 2048, 2048), 9)
    held, leaky = Calls(LEAD), Calls(LEAD)
    for calls in (held, leaky):
        calls.a_root(0, ROOT_A, CallShape(1, 2048, 2048))
    for _ in range(600):
        assert held.bucket(0) == []
    assert leaky.bucket(1) == [0]
    assert held.words_root(0, ROOT_W) is Verdict.ACCEPTED and leaky.words_root(0, ROOT_W) is Verdict.PAST_BITS


def test_a_bucket_inside_two_spans_counts_against_both():
    calls = Calls(EgressCap(32, 16 * 8 * 64))
    calls.a_root(0, ROOT_A, TIGHT)
    calls.bucket(1, [20])
    calls.a_root(1, ROOT_A, TIGHT)
    assert calls.bucket(1, [11]) == [0]
    assert calls.bucket(1, [18]) == []
    assert calls.bucket(1, [0]) == []
    assert calls.bucket(1) == [1]
    assert calls.close() == {0: Verdict.PAST_BITS, 1: Verdict.PAST_BITS}


def test_buckets_outside_a_span_dont_count():
    calls = Calls(EgressCap(32, 16 * 8 * 64))
    calls.bucket(64, [10**6] * 64)
    calls.a_root(0, ROOT_A, TIGHT)
    calls.a_root(1, ROOT_A, TIGHT)
    assert calls.words_root(0, ROOT_W) is Verdict.ACCEPTED
    assert calls.bucket(64, [10**6]) == [1]
    assert calls.close() == {0: Verdict.ACCEPTED, 1: Verdict.PAST_BITS}


def test_a_malformed_bucket_is_refused():
    calls = Calls(LEAD)
    for bad in ((-1, []), (1, [8, 8]), (2, [-1]), (2, [1.5]), (2, [True])):
        with pytest.raises(CallRefusal):
            calls.bucket(*bad)


@pytest.mark.parametrize("cap,macs", [(-1, 1), (32, 0), (32.0, 1), (32, 1.5)])
def test_an_egress_cap_is_bits_over_macs(cap, macs):
    with pytest.raises(ValueError):
        EgressCap(cap, macs)


def test_a_call_open_when_the_window_closes_is_refused_and_the_window_takes_nothing_after():
    calls = Calls(LEAD)
    calls.a_root(0, ROOT_A, SMALL)
    calls.a_root(1, ROOT_A, SMALL)
    assert calls.words_root(1, ROOT_W) is Verdict.ACCEPTED
    with pytest.raises(CallRefusal, match="not open"):
        calls.words_root(1, ROOT_W)
    assert calls.close() == {0: Verdict.UNCLOSED, 1: Verdict.ACCEPTED}
    for late in (lambda: calls.a_root(2, ROOT_A, SMALL), lambda: calls.words_root(0, ROOT_W), lambda: calls.bucket(0),
                 calls.close):
        with pytest.raises(CallRefusal, match="closed"):
            late()


def test_the_record_carries_shape_roots_coins_digest_and_verdict_and_no_count():
    calls = Calls(EgressCap(32, 16 * 8 * 64))
    out = calls.a_root(3, ROOT_A, TIGHT)
    calls.bucket(1, [7])
    want = {"call": 3, "rows": 16, "cols": 8, "K": 64, "a_root": ROOT_A.hex(),
            "coins": tagged_sha512(STREAM_TAG, out).hex(), "words_root": None, "verdict": None}
    assert calls.summary() == [want]
    calls.words_root(3, ROOT_W)
    assert calls.summary() == [{**want, "words_root": ROOT_W.hex(), "verdict": "accepted"}]
