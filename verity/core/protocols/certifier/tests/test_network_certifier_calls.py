"""The lead's calls at the network certifier (`calls`): the shape a call's coins take, the coins flipped after its A root
(G exactly uniform by rejection), the release word for word as the lead's kernel reads it, each forward's pooled count,
the forwards serial, refused past the forward's bits, and the node's forwards over its PoUW windows and its control
channel."""

import asyncio
import itertools
import json
import math
import os
import random
import socket
import struct

import pytest

from verity.core.protocols.certifier.calls import (
    ADMITTED_TILES, CALLS_LINE_BYTES, G_BOUND, RELEASE_TAG, ROT_W, CallCoins, CallRefusal, Calls, CallsClient, CallShape,
    EgressCap, ForwardVerdict, NodeCalls, col_code, flip, release, serve_calls, timing_bits,
)
from verity.ml.commitments.identity import tagged_sha512

ROOT_A, ROOT_W = bytes(range(64)), bytes(range(64, 128))
#: m 16, n 8, S 1, H 1, one tile: 64 bytes of row words, 32 of G (no rejection) and 24 of lanes.
SMALL = CallShape(1, 8, 1)
SMALL_ENTROPY = 64 + 32 + 24
#: rows · n · K = 8192 MACs, so a cap of 32 bits over 8192 gives it exactly 32 bits.
TIGHT = CallShape(16, 8, 64)
#: The lead's cap and its forward's MACs a token.
LEAD = EgressCap(32, 973_078_528)
#: Llama-3.2-1B's linears, (K, n): qkv, o, gate_up and down.
LLAMA = ((2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048))


class Feed:
    """Entropy from fixed bytes; `reads` lists the sizes taken."""

    def __init__(self, data: bytes) -> None:
        self.data, self.read, self.reads = data, 0, []

    def __call__(self, k: int) -> bytes:
        if self.read + k > len(self.data):
            raise RuntimeError("the feed ran out")
        self.read += k
        self.reads.append(k)
        return self.data[self.read - k:self.read]


def g_feed(words, *, lanes=bytes(24)) -> Feed:
    """A feed for SMALL: zero row words, then G's 32-bit words, then the lanes."""
    return Feed(bytes(64) + b"".join(w.to_bytes(4, "little") for w in words) + lanes)


def weights(coins: CallCoins) -> list[tuple[int, ...]]:
    return [tuple(coins.G(q, j) for q in range(4)) for j in range(coins.shape.n)]


def words(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def kernel_feed(coins: CallCoins) -> list[int]:
    """The release as the lead's kernel lays it out, built a word at a time from the coins' values: row i's words with
    a₁[i, e] at bit e and b_t[i] at bit t + 3; column j's G word, byte e the int8 G[e, j]; then each tile's lane-coin
    words, (t·n + j), bytes 0–2 the int8 s_t[e, k, j] and byte 3 C[3, j] at step 0 and 1 later; and its column sources,
    (c·n + j), 0x4B400000 − σ."""
    sh = coins.shape
    out = []
    for i in range(sh.m):
        h = [0] * sh.H
        for e in range(4):
            h[0] |= coins.a1(i, e) << e
        for t in range(1, sh.S):
            h[(t + 3) // 32] |= coins.b(i, t) << (t + 3) % 32
        out += h
    out += [sum((coins.G(e, j) & 0xFF) << 8 * e for e in range(4)) for j in range(sh.n)]
    for k in range(sh.tiles):
        for t in range(sh.S):
            for j in range(sh.n):
                top = col_code(j, 3) & 0xFF if t == 0 else 1
                out.append(sum((coins.s(t, k, e, j) & 0xFF) << 8 * e for e in range(3)) | top << 24)
        for c in range(2):
            for j in range(sh.n):
                sigma = sum(ROT_W[c][e] * coins.s(t, k, e, j) for t in range(sh.S) for e in range(3))
                out.append((0x4B400000 - sigma) % 2**32)
    return out


def test_shape_pads_to_mma_tiles_and_counts_k32_steps():
    assert [(s.m, s.n, s.S, s.H, s.tiles) for s in (CallShape(1, 2048, 2048), CallShape(100, 1000, 4096),
                                                    CallShape(17, 13, 64), CallShape(1, 8, 8192))] == [
        (16, 2048, 74, 3, 1), (112, 1000, 147, 5, 2), (32, 16, 3, 1, 1), (16, 8, 293, 10, 1)]
    assert [CallShape(1, 8, K).S for K in (1, 21, 22, 49, 50)] == [1, 1, 2, 2, 3]
    assert [CallShape(1, 8, 21 + 28 * (S - 1)).H for S in (1, 29, 30, 61, 62)] == [1, 1, 2, 2, 3]


@pytest.mark.parametrize("bad", [(0, 8, 8), (1, 0, 8), (1, 8, 0), (-1, 8, 8), (True, 8, 8), (1.0, 8, 8)])
def test_shape_refuses_a_dimension_that_isnt_a_positive_integer(bad):
    with pytest.raises(CallRefusal):
        CallShape(*bad)


def test_shape_refuses_a_call_the_verifier_doesnt_admit():
    """`Layout.Admits`: rowBlocks · colBlocks · S ≤ 2⁴⁰."""
    K = 21 + 28 * (ADMITTED_TILES - 1)
    assert CallShape(16, 8, K).S == ADMITTED_TILES
    with pytest.raises(CallRefusal, match="admit"):
        CallShape(16, 8, K + 1)
    with pytest.raises(CallRefusal, match="admit"):
        CallShape(17, 8, K)
    assert CallShape(16, 16, 21 + 28 * (ADMITTED_TILES // 2 - 1)).S == ADMITTED_TILES // 2


def test_the_release_has_the_feeds_sizes_at_llama_3_2_1b():
    """Row words, G words, lane-coin words and column sources, in bytes, for each linear at one row, and its row words
    at 64 rows."""
    def parts(sh):
        return 4 * sh.m * sh.H, 4 * sh.n, 4 * sh.tiles * sh.S * sh.n, 8 * sh.tiles * sh.n

    decode = [CallShape(1, n, K) for K, n in LLAMA]
    assert [parts(sh) for sh in decode] == [(192, 12_288, 909_312, 24_576), (192, 8_192, 606_208, 16_384),
                                           (192, 65_536, 4_849_664, 131_072), (640, 8_192, 2_400_256, 16_384)]
    assert all(sum(parts(sh)) == sh.release_bytes for sh in decode)
    assert [parts(CallShape(64, n, K))[0] for K, n in LLAMA] == [768, 768, 768, 2_560]
    assert 16 * sum(sh.release_bytes for sh in decode) == 144_788_480


def test_a_decode_call_takes_464080_coins_and_releases_its_feed():
    decode = CallShape(1, 2048, 2048)
    assert decode.coins == 16 * (74 + 3) + 4 * 2048 + 3 * 74 * 2048 == 464_080
    assert len(release(flip(decode))) == decode.release_bytes == 630_976


def test_flip_reads_the_entropy_in_the_releases_order():
    shape = CallShape(70, 24, 120)
    feed = Feed(bytes(4 * 80 * 1 + 4 * 24 + 3 * 5 * 2 * 24))
    flip(shape, feed)
    assert feed.reads == [4 * shape.m * shape.H, 4 * shape.n, 3 * shape.S * shape.tiles * shape.n]


def test_row_words_hold_a1_then_the_tag_bits_from_step_1():
    shape = CallShape(1, 8, 64)
    coins = flip(shape, Feed(bytes([0b1010_0110, 0xFF, 0xFF, 0xFF]) + bytes(4 * 15 + 32 + 3 * 3 * 8)))
    assert words(coins.rows)[:2] == [0b10_0110, 0]
    assert [coins.a1(0, e) for e in range(4)] == [0, 1, 1, 0] and [coins.b(0, t) for t in (1, 2)] == [0, 1]
    with pytest.raises(IndexError):
        coins.b(0, 0)
    with pytest.raises(IndexError):
        coins.b(0, 3)
    wide = CallShape(1, 8, 2048)
    full = flip(wide, Feed(b"\xff" * (4 * 16 * 3) + bytes(32 + 3 * 74 * 8)))
    assert words(full.rows) == [0xFFFF_FFFF, 0xFFFF_FFFF, 0x1FFF] * 16
    assert all(full.b(i, t) == 1 for i in range(16) for t in range(1, 74))


def test_g_is_the_base_7_digits_of_the_accepted_word_less_3():
    assert weights(flip(SMALL, g_feed([0] * 8))) == [(-3, -3, -3, -3)] * 8
    assert weights(flip(SMALL, g_feed([2400] * 8))) == [(3, 3, 3, 3)] * 8
    digits = [tuple(g + 3 for g in w) for w in weights(flip(SMALL, g_feed([7, 49, 343, 2406, 2401 * 9 + 1, 0, 0, 0])))]
    assert digits[:5] == [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (5, 0, 0, 0), (1, 0, 0, 0)]
    assert words(flip(SMALL, g_feed([7 + 2 * 49] + [0] * 7)).g)[0] == 0xFD_FF_FE_FD


def test_g_rejects_every_word_at_or_above_the_bound_and_takes_the_next():
    feed = g_feed([0xFFFF_FFFF, G_BOUND - 1, G_BOUND, 1, 0, 0, 0, 0, 0, 0])
    got = weights(flip(SMALL, feed))
    assert got[:2] == [(3, 3, 3, 3), (-2, -3, -3, -3)]
    assert feed.read == len(feed.data) and feed.reads == [64, 32, 8, 24]


def test_g_is_exactly_uniform():
    """The accepted words are G_BOUND / 2401 for each residue mod 2401, and `flip` takes the 2401 residues one to one onto
    [−3, 3]⁴."""
    assert G_BOUND % 7**4 == 0 and 2**32 - G_BOUND == 872 < 7**4
    wide = CallShape(1, 2401, 1)
    n = wide.n
    coins = flip(wide, Feed(bytes(64) + b"".join(w.to_bytes(4, "little") for w in [*range(7**4), *[0] * (n - 7**4)])
                            + bytes(3 * n)))
    assert set(weights(coins)[:7**4]) == set(itertools.product(range(-3, 4), repeat=4))


def test_a_lane_is_its_bytes_low_7_bits_odd_and_each_value_from_two_bytes():
    shape = CallShape(1, 88, 1)
    coins = flip(shape, Feed(bytes(64 + 4 * 88) + bytes(range(256)) + bytes(8)))
    s = [coins.s(0, 0, lane, j) for j in range(88) for lane in range(3)][:256]
    assert s == [2 * (x % 128) - 127 for x in range(256)]
    assert sorted(s) == sorted([*range(-127, 128, 2)] * 2)


def test_col_code_is_balanced_base_13_of_j_plus_1():
    assert [col_code(0, q) for q in range(4)] == [1, 0, 0, 0]
    for j in range(14_280):
        C = [col_code(j, q) for q in range(4)]
        assert sum(c * 13**q for q, c in enumerate(C)) == j + 1 and all(-6 <= c <= 6 for c in C)


@pytest.mark.parametrize("shape", [CallShape(70, 24, 120), CallShape(1, 8, 1), CallShape(17, 13, 2048),
                                   CallShape(129, 16, 50)])
def test_the_release_is_the_kernels_feed_word_for_word(shape):
    coins = flip(shape)
    out = release(coins)
    assert len(out) == shape.release_bytes and words(out) == kernel_feed(coins)
    assert out.startswith(coins.rows + coins.g)


def test_a_column_source_read_as_fp32_is_the_bias_less_sigma():
    """0x4B400000 is 1.5 · 2²³, so its word less σ, read as an fp32, is 12,582,912 − σ exactly while |σ| < 2²²: what the
    epilogue adds to fold the lanes out."""
    shape = CallShape(64, 8, 8192)
    coins = flip(shape)
    w = words(release(coins))
    sources = w[shape.m * shape.H + shape.n + shape.S * shape.n:]
    assert len(sources) == 2 * shape.n
    for c, j in itertools.product(range(2), range(shape.n)):
        sigma = sum(ROT_W[c][e] * coins.s(t, 0, e, j) for t in range(shape.S) for e in range(3))
        assert abs(sigma) < 2**22
        assert struct.unpack("<f", struct.pack("<I", sources[c * shape.n + j]))[0] == 12_582_912 - sigma


def test_coins_hold_their_sizes_and_ranges():
    shape = CallShape(1, 8, 64)
    good = flip(shape, Feed(bytes(4 * 16 + 32 + 3 * 3 * 8)))
    for bad in ({"rows": bytes([0b100_0000]) + good.rows[1:]}, {"rows": good.rows[:-1]}, {"g": bytes([4]) + good.g[1:]},
                {"g": bytes([0xFC]) + good.g[1:]}, {"u": good.u[:-1]}, {"u": bytes([128]) + good.u[1:]},
                {"rows": bytearray(good.rows)}):
        with pytest.raises(ValueError):
            CallCoins(**{"shape": shape, "rows": good.rows, "g": good.g, "u": good.u, **bad})
    assert CallCoins(shape, bytes([0b11_1111]) + good.rows[1:], bytes([0xFD, 3]) + good.g[2:], bytes([127]) + good.u[1:])


def test_the_coins_follow_the_a_root_and_go_out_once():
    seen: list[list[str]] = []

    def entropy(k: int) -> bytes:
        seen.append([c["a_root"] for f in calls.summary() for c in f["calls"]])
        return bytes(k)

    calls = Calls(LEAD, entropy)
    for refused in (lambda: calls.words_root(0, ROOT_W), lambda: calls.a_root(0, 0, b"short", SMALL),
                    lambda: calls.a_root(-1, 0, ROOT_A, SMALL), lambda: calls.a_root(0, -1, ROOT_A, SMALL),
                    lambda: calls.a_root(True, 0, ROOT_A, SMALL), lambda: calls.a_root(0, True, ROOT_A, SMALL),
                    lambda: calls.a_root(0, 0, ROOT_A, (1, 8, 1))):
        with pytest.raises(CallRefusal):
            refused()
    assert seen == [] and calls.summary() == [] and calls.open_forward is None
    out = calls.a_root(0, 0, ROOT_A, SMALL)
    assert len(out) == SMALL.release_bytes and seen and all(ROOT_A.hex() in held for held in seen)
    reads = len(seen)
    with pytest.raises(CallRefusal, match="once"):
        calls.a_root(0, 0, ROOT_A, SMALL)
    assert len(seen) == reads
    assert calls.summary()[0]["calls"][0]["release"] == tagged_sha512(RELEASE_TAG, out).hex()


def test_a_call_whose_coins_arent_flipped_refuses_its_forward_which_stays_open():
    def broken(k: int) -> bytes:
        raise OSError("no entropy")

    calls = Calls(LEAD, broken)
    with pytest.raises(OSError):
        calls.a_root(0, 0, ROOT_A, SMALL)
    assert calls.open_forward == 0
    with pytest.raises(CallRefusal, match="once"):
        calls.a_root(0, 0, ROOT_A, SMALL)
    with pytest.raises(ValueError, match="failed earlier"):
        calls.a_root(0, 1, ROOT_A, SMALL)
    assert calls.words_root(0, ROOT_W) is ForwardVerdict.NO_COINS
    short = Calls(LEAD, lambda k: bytes(k - 1))
    with pytest.raises(ValueError, match="entropy gave"):
        short.a_root(0, 0, ROOT_A, SMALL)
    assert calls.close() == short.close() == {0: ForwardVerdict.NO_COINS}
    assert [c["release"] for c in calls.summary()[0]["calls"]] == [None, None]
    assert calls.summary()[0]["verdict"] == ForwardVerdict.NO_COINS.value


def test_once_the_entropy_fails_the_window_flips_nothing_more():
    feed = Feed(bytes(SMALL_ENTROPY))
    calls = Calls(LEAD, feed)
    with pytest.raises(RuntimeError, match="ran out"):
        calls.a_root(0, 0, ROOT_A, CallShape(64, 64, 64))
    assert calls.words_root(0, ROOT_W) is ForwardVerdict.NO_COINS
    read = feed.read
    with pytest.raises(ValueError, match="failed earlier"):
        calls.a_root(1, 0, ROOT_A, SMALL)
    assert feed.read == read < SMALL_ENTROPY
    assert calls.close() == {0: ForwardVerdict.NO_COINS, 1: ForwardVerdict.NO_COINS}


def test_timing_bits_are_ceil_log2_of_r_plus_1():
    assert [timing_bits(r) for r in range(9)] == [0, 1, 2, 2, 3, 3, 3, 3, 4]
    assert all(timing_bits(r) == math.ceil(math.log2(r + 1)) for r in range(1, 5000))
    for bad in (-1, 1.0, True):
        with pytest.raises(CallRefusal):
            timing_bits(bad)


@pytest.mark.parametrize("last,verdict", [(0, ForwardVerdict.ACCEPTED), (1, ForwardVerdict.PAST_BITS)])
def test_a_forward_is_refused_when_its_count_passes_its_bits(last, verdict):
    """TIGHT's 8192 MACs at a cap of 32 bits over 8192: 6 timing bits and 26 content bits, then one more bucket."""
    calls = Calls(EgressCap(32, TIGHT.macs))
    calls.a_root(0, 0, ROOT_A, TIGHT)
    calls.bucket(32, [26])
    calls.bucket(last)
    assert calls.words_root(0, ROOT_W) is verdict


@pytest.mark.parametrize("count,verdict", [(64, ForwardVerdict.ACCEPTED), (65, ForwardVerdict.PAST_BITS)])
def test_a_forward_pools_its_calls_bits_and_counts_each_bucket_once(count, verdict):
    """Two TIGHT calls bring 32 bits each; every bucket of the span counts once against their 64, never once a call."""
    calls = Calls(EgressCap(32, TIGHT.macs))
    calls.a_root(0, 0, ROOT_A, TIGHT)
    calls.bucket(1, [count // 2 - 1])
    calls.a_root(0, 1, ROOT_A, TIGHT)
    calls.bucket(1, [count - count // 2 - 1])
    assert calls.words_root(0, ROOT_W) is verdict


@pytest.mark.parametrize("count,verdict", [(10, ForwardVerdict.ACCEPTED), (11, ForwardVerdict.PAST_BITS)])
def test_a_share_that_isnt_an_integer_compares_exactly(count, verdict):
    calls = Calls(EgressCap(32, 3 * TIGHT.macs))
    calls.a_root(0, 0, ROOT_A, TIGHT)
    calls.bucket(1, [count - 1])
    assert calls.words_root(0, ROOT_W) is verdict


def test_the_lead_gives_a_forward_32_bits_a_row():
    """Llama-3.2-1B's 16 layers of linears are the lead's fwdMacs at one row, so a decode forward has 32 bits and a
    forward of 64 rows 2048. A forward the certifier holds at r = 0 counts nothing; one bucket a call at r = 1 passes."""
    decode = [CallShape(1, n, K) for K, n in LLAMA] * 16
    macs = sum(sh.macs for sh in decode)
    assert macs == LEAD.fwd_macs
    assert not LEAD.passed(macs, 32) and LEAD.passed(macs, 33)
    assert not LEAD.passed(64 * macs, 2048) and LEAD.passed(64 * macs, 2049)
    held, leaky = Calls(LEAD), Calls(LEAD)
    for calls, r in ((held, 0), (leaky, 1)):
        for j, sh in enumerate(decode):
            calls.open(0, j, ROOT_A, sh)
            calls.released(0, j, b"")
            calls.bucket(r)
    assert held.words_root(0, ROOT_W) is ForwardVerdict.ACCEPTED
    assert leaky.words_root(0, ROOT_W) is ForwardVerdict.PAST_BITS


@pytest.mark.parametrize("count,verdict", [(29, ForwardVerdict.ACCEPTED), (30, ForwardVerdict.PAST_BITS)])
def test_the_release_counts_once_a_forward_when_its_words_root_closes_it(count, verdict):
    calls = Calls(EgressCap(32, TIGHT.macs), release_bits=3)
    for forward in (0, 1):
        calls.a_root(forward, 0, ROOT_A, TIGHT)
        calls.bucket(1, [count - 1])
        assert calls.words_root(forward, ROOT_W) is verdict


@pytest.mark.parametrize("bad", [-1, 1.0, True])
def test_a_releases_bits_are_an_integer(bad):
    with pytest.raises(ValueError):
        Calls(LEAD, release_bits=bad)


def test_the_forwards_are_serial_and_a_closed_forward_takes_no_call():
    feed = Feed(bytes(3 * SMALL_ENTROPY))
    calls = Calls(LEAD, feed)
    calls.a_root(0, 0, ROOT_A, SMALL)
    read = feed.read
    with pytest.raises(CallRefusal, match="serial"):
        calls.a_root(1, 0, ROOT_A, SMALL)
    with pytest.raises(CallRefusal, match="not open"):
        calls.words_root(1, ROOT_W)
    assert feed.read == read and calls.open_forward == 0
    calls.a_root(0, 1, ROOT_A, SMALL)
    assert calls.words_root(0, ROOT_W) is ForwardVerdict.ACCEPTED and calls.open_forward is None
    for refused in (lambda: calls.a_root(0, 2, ROOT_A, SMALL), lambda: calls.words_root(0, ROOT_W)):
        with pytest.raises(CallRefusal, match="closed|not open"):
            refused()
    calls.a_root(1, 0, ROOT_A, SMALL)
    assert calls.open_forward == 1 and feed.read == 3 * SMALL_ENTROPY


def test_buckets_outside_a_forward_dont_count():
    calls = Calls(EgressCap(32, TIGHT.macs))
    calls.bucket(64, [10**6] * 64)
    calls.a_root(0, 0, ROOT_A, TIGHT)
    assert calls.words_root(0, ROOT_W) is ForwardVerdict.ACCEPTED
    calls.bucket(64, [10**6])
    calls.a_root(1, 0, ROOT_A, TIGHT)
    calls.bucket(64, [10**6])
    assert calls.words_root(1, ROOT_W) is ForwardVerdict.PAST_BITS


@pytest.mark.parametrize("bad", [(-1, []), (1, [8, 8]), (2, [-1]), (2, [1.5]), (2, [True])])
def test_a_malformed_bucket_is_refused_open_or_not(bad):
    calls = Calls(LEAD)
    with pytest.raises(CallRefusal):
        calls.bucket(*bad)
    calls.a_root(0, 0, ROOT_A, SMALL)
    with pytest.raises(CallRefusal):
        calls.bucket(*bad)


@pytest.mark.parametrize("cap,macs", [(-1, 1), (32, 0), (32.0, 1), (32, 1.5)])
def test_an_egress_cap_is_bits_over_macs(cap, macs):
    with pytest.raises(ValueError):
        EgressCap(cap, macs)


def test_a_forward_open_when_the_window_closes_is_refused_and_the_window_takes_nothing_after():
    calls = Calls(LEAD)
    calls.a_root(0, 0, ROOT_A, SMALL)
    assert calls.words_root(0, ROOT_W) is ForwardVerdict.ACCEPTED
    calls.a_root(1, 0, ROOT_A, SMALL)
    calls.open(1, 1, ROOT_A, SMALL)
    assert calls.close() == {0: ForwardVerdict.ACCEPTED, 1: ForwardVerdict.UNCLOSED}
    for late in (lambda: calls.a_root(2, 0, ROOT_A, SMALL), lambda: calls.open(1, 2, ROOT_A, SMALL),
                 lambda: calls.released(1, 1, b""), lambda: calls.words_root(1, ROOT_W), lambda: calls.bucket(0),
                 calls.close):
        with pytest.raises(CallRefusal, match="closed"):
            late()
    assert calls.summary()[1]["calls"][1]["release"] is None


def test_the_record_carries_each_forwards_calls_words_root_and_verdict_and_no_count():
    calls = Calls(EgressCap(32, TIGHT.macs))
    out1 = calls.a_root(3, 1, ROOT_A, TIGHT)
    out0 = calls.a_root(3, 0, ROOT_W, SMALL)
    calls.bucket(1, [7])
    want = {"forward": 3,
            "calls": [{"call": 0, "rows": 1, "cols": 8, "K": 1, "a_root": ROOT_W.hex(),
                       "release": tagged_sha512(RELEASE_TAG, out0).hex()},
                      {"call": 1, "rows": 16, "cols": 8, "K": 64, "a_root": ROOT_A.hex(),
                       "release": tagged_sha512(RELEASE_TAG, out1).hex()}],
            "words_root": None, "verdict": None}
    assert calls.summary() == [want]
    calls.words_root(3, ROOT_W)
    assert calls.summary() == [{**want, "words_root": ROOT_W.hex(), "verdict": "accepted"}]


# --- the node: its PoUW windows, its links' ticks and its control channel ---------------------------------------------

def verdicts(record: list[dict]) -> dict[int, str | None]:
    return {f["forward"]: f["verdict"] for f in record}


def test_a_node_begins_each_window_at_its_first_tick_and_publishes_the_window_before():
    node = NodeCalls(EgressCap(32, TIGHT.macs))
    with pytest.raises(CallRefusal, match="no PoUW window has begun"):
        node.a_root(0, 0, ROOT_A, TIGHT)
    node.tick(0, 0)
    assert node.records == {}
    node.a_root(0, 0, ROOT_A, TIGHT)
    assert node.words_root(0, ROOT_W) is ForwardVerdict.ACCEPTED
    node.a_root(1, 0, ROOT_A, TIGHT)
    node.tick(2, 0)
    assert node.window == 2
    assert verdicts(node.records[0]) == {0: "accepted", 1: "open when the window closed"}
    node.a_root(2, 0, ROOT_A, TIGHT)
    node.close()
    node.close()
    assert sorted(node.records) == [0, 2] and verdicts(node.records[2]) == {2: "open when the window closed"}
    for late in (lambda: node.tick(3, 0), lambda: node.a_root(3, 0, ROOT_A, TIGHT),
                 lambda: node.words_root(2, ROOT_W)):
        with pytest.raises(CallRefusal, match="the node's calls are closed"):
            late()


@pytest.mark.parametrize("last,verdict", [(0, ForwardVerdict.ACCEPTED), (1, ForwardVerdict.PAST_BITS)])
def test_a_tick_of_an_earlier_window_counts_against_the_forward_open_when_it_runs(last, verdict):
    node = NodeCalls(EgressCap(32, TIGHT.macs))
    node.tick(1, 0)
    node.a_root(0, 0, ROOT_A, TIGHT)
    node.tick(0, 32, [26])
    node.tick(0, last)
    assert node.window == 1 and node.words_root(0, ROOT_W) is verdict
    with pytest.raises(CallRefusal, match="PoUW window"):
        node.tick(-1, 0)


@pytest.mark.parametrize("count,verdict", [(29, ForwardVerdict.ACCEPTED), (30, ForwardVerdict.PAST_BITS)])
def test_the_nodes_release_counts_once_a_forward(count, verdict):
    node = NodeCalls(EgressCap(32, TIGHT.macs), release_bits=3)
    node.tick(0, 1, [10**6])
    for forward in (0, 1):
        node.a_root(forward, 0, ROOT_A, TIGHT)
        node.tick(0, 1, [count - 1])
        assert node.words_root(forward, ROOT_W) is verdict
    with pytest.raises(ValueError, match="release"):
        NodeCalls(LEAD, release_bits=-1)


def test_a_tick_while_a_calls_coins_are_flipped_counts_against_its_forward_and_doesnt_wait_for_the_flip():
    node, held, ticked = None, [], []

    def entropy(k: int) -> bytes:
        held.append(node._lock.locked())
        if not held[-1] and not ticked:
            node.tick(0, 32, [26])
            ticked.append(k)
        return os.urandom(k)

    node = NodeCalls(EgressCap(32, TIGHT.macs), entropy)
    node.tick(0, 0)
    assert len(node.a_root(0, 0, ROOT_A, TIGHT)) == TIGHT.release_bytes
    assert held and not any(held) and ticked
    node.tick(0, 1)
    assert node.words_root(0, ROOT_W) is ForwardVerdict.PAST_BITS


def test_coins_flipped_after_their_window_closed_dont_go_out():
    node, flips = None, []

    def entropy(k: int) -> bytes:
        if not flips:
            node.tick(1, 0)
            flips.append(k)
        return os.urandom(k)

    node = NodeCalls(LEAD, entropy)
    node.tick(0, 0)
    with pytest.raises(CallRefusal, match="the window is closed"):
        node.a_root(0, 0, ROOT_A, SMALL)
    assert [([c["release"] for c in f["calls"]], f["verdict"]) for f in node.records[0]] == [
        ([None], "open when the window closed")]
    assert len(node.a_root(0, 0, ROOT_A, SMALL)) == SMALL.release_bytes


def test_a_node_whose_entropy_fails_refuses_its_windows_forwards_and_flips_again_in_the_next():
    fails = [True]

    def entropy(k: int) -> bytes:
        if fails.pop() if fails else False:
            raise OSError("no entropy")
        return os.urandom(k)

    node = NodeCalls(LEAD, entropy)
    node.tick(0, 0)
    with pytest.raises(OSError):
        node.a_root(0, 0, ROOT_A, SMALL)
    with pytest.raises(ValueError, match="failed earlier"):
        node.a_root(0, 1, ROOT_A, SMALL)
    assert node.words_root(0, ROOT_W) is ForwardVerdict.NO_COINS
    with pytest.raises(ValueError, match="failed earlier"):
        node.a_root(1, 0, ROOT_A, SMALL)
    node.tick(1, 0)
    assert verdicts(node.records[0]) == {0: "a call's coins were not flipped", 1: "a call's coins were not flipped"}
    assert len(node.a_root(0, 0, ROOT_A, SMALL)) == SMALL.release_bytes


def control_channel(node: NodeCalls, prover):
    """`prover(host, port)` run off the loop against `serve_calls(node)` on loopback; returns what it returns."""
    async def main():
        server = await serve_calls(node, "127.0.0.1", 0)
        try:
            return await asyncio.to_thread(prover, *server.sockets[0].getsockname()[:2])
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(main())


def test_the_control_channel_answers_an_a_root_with_its_release_and_a_words_root_with_its_verdict():
    data = random.Random(7).randbytes(4096)
    node = NodeCalls(EgressCap(32, TIGHT.macs), Feed(data))
    node.tick(0, 0)

    def prover(host: str, port: int):
        with CallsClient(host, port) as client:
            out = [client.a_root(0, 0, ROOT_A, SMALL)]
            with pytest.raises(CallRefusal, match="registered already"):
                client.a_root(0, 0, ROOT_A, SMALL)
            with pytest.raises(CallRefusal, match="serial"):
                client.a_root(1, 0, ROOT_A, TIGHT)
            verdict = [client.words_root(0, ROOT_W)]
            out.append(client.a_root(1, 0, ROOT_A, TIGHT))
            node.tick(0, 32, [27])
            verdict.append(client.words_root(1, ROOT_W))
            with pytest.raises(CallRefusal, match="not open"):
                client.words_root(1, ROOT_W)
            with pytest.raises(CallRefusal, match="call 0 of forward 2's coins weren't flipped: the feed ran out"):
                client.a_root(2, 0, ROOT_A, CallShape(512, 64, 64))
        return out, verdict

    out, verdict = control_channel(node, prover)
    feed = Feed(data)
    assert out == [release(flip(SMALL, feed)), release(flip(TIGHT, feed))]
    assert verdict == [ForwardVerdict.ACCEPTED, ForwardVerdict.PAST_BITS]
    node.close()
    digests = [tagged_sha512(RELEASE_TAG, o).hex() for o in out]
    assert [([c["release"] for c in f["calls"]], f["verdict"]) for f in node.records[0]] == [
        ([digests[0]], "accepted"), ([digests[1]], "past its bits"), ([None], "a call's coins were not flipped")]


def test_the_control_channel_refuses_a_line_that_isnt_a_request():
    node = NodeCalls(LEAD)
    a, w = ROOT_A.hex().encode(), ROOT_W.hex().encode()
    lines = [b"not json", b"[0]", b"5", b'{"forward": 0}',
             b'{"forward": 0, "call": 0, "a_root": "zz", "rows": 1, "cols": 8, "K": 1}',
             b'{"forward": 0, "words_root": "%s", "extra": 1}' % w, b'{"forward": 0, "call": 0, "words_root": "%s"}' % w,
             b'{"forward": 0, "call": 5, "a_root": "%s", "rows": 0, "cols": 8, "K": 1}' % a,
             b'{"forward": 0, "call": 6, "a_root": "%s", "rows": 1, "cols": 8, "K": 1}' % a,
             b"x" * (CALLS_LINE_BYTES + 1)]

    def prover(host: str, port: int):
        with socket.create_connection((host, port)) as sock, sock.makefile("rwb") as stream:
            stream.write(b"\n".join(lines) + b"\n")
            stream.flush()
            sock.shutdown(socket.SHUT_WR)
            return [json.loads(line) for line in stream.read().splitlines()]

    answers = control_channel(node, prover)
    generic = {"refused": "a request is a JSON object of forward, call, a_root (hex), rows, cols and K, or of forward "
                          "and words_root (hex)"}
    assert answers[:7] == [generic] * 7
    assert answers[7] == {"forward": 0, "call": 5, "refused": "a call's rows is a positive integer, not 0"}
    assert answers[8] == {"forward": 0, "call": 6,
                          "refused": "no PoUW window has begun: a window begins at its first tick"}
    assert answers[9:] == [{"refused": f"a request is one line of at most {CALLS_LINE_BYTES} bytes"}]
