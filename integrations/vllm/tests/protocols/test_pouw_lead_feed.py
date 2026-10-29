"""`lead.feed`: a call's sizes, coins and feed against Lean's `Pouw.Lead` definitions (restated here from Layout.lean),
and the in-process certifier's refusals."""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
from verity_vllm.protocols.pouw.lead import feed as F

#: Llama-3.2-1B's four linears, (K, n)
LLAMA_1B = [(2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048)]


def lean_steps(k: int) -> int:
    return 1 + (max(k - 21, 0) + 27) // 28


def bal13(z: int, q: int) -> int:
    return (z + 14280) // 13**q % 13 - 6


def col_code(j: int) -> list[int]:
    return [bal13(j + 1, q) for q in range(4)]


def row_tag(i: int) -> list[int]:
    z = i // 2 + 1 if i % 2 == 0 else -(i // 2 + 1)
    return [bal13(z, q) for q in range(4)]


def rot_weight(i: int, ell: int) -> int:
    return 127 if ell == i % 2 else 1


def balanced(z: int) -> list[int]:
    """The low four digits of z's balanced base-13 expansion, digit by digit."""
    out = []
    for _ in range(4):
        d = (z % 13 + 6) % 13 - 6
        out.append(d)
        z = (z - d) // 13
    return out


def rand_of(seed: int) -> F.Rand:
    return random.Random(seed).randbytes


def sampled(m: int, k: int, n: int, seed: int = 0) -> tuple[F.Call, F.CallCoins, F.Feed]:
    call = F.Call(0, 0, 0, m, k, n)
    coins = F.CallCoins.sample(call.shape, rand_of(seed))
    return call, coins, F.Feed.of(call, coins)


@pytest.mark.parametrize("k, n", LLAMA_1B)
@pytest.mark.parametrize("m", [1, 16, 17, 64, 65, 8192])
def test_sizes_at_llama_shapes_are_leans(m, k, n):
    sh = F.Shape(m, k, n)
    assert sh.S == lean_steps(k) == {2048: 74, 8192: 293}[k]
    assert sh.H == {2048: 3, 8192: 10}[k]
    assert (sh.R, sh.C, sh.T) == (16 * -(-m // 16), n, -(-m // 64))
    assert sh.feed_words() == sh.R * sh.H + n
    assert (sh.lane_words(), sh.source_words()) == (sh.T * sh.S * n, 2 * sh.T * n)
    assert sh.computed(128) == sh.computed(256) == n
    assert sh.refusal() is None


def test_steps_is_stepsFor():
    assert [F.steps(k) for k in (1, 21, 22, 49, 50, 77, 78)] == [1, 1, 2, 2, 3, 3, 4]
    assert all(F.steps(k) == lean_steps(k) for k in range(1, 20_000))


def test_padding_is_whole_mma_tiles():
    sh = F.Shape(65, 100, 13)
    assert (sh.R, sh.C, sh.T, sh.computed(128)) == (80, 16, 2, 128)


def test_codes_are_colCode_and_distinct_below_28560():
    c = F.codes(28_560)
    for j in [*range(0, 28_560, 97), 14_279, 14_280, 28_559]:
        assert list(c[:, j]) == col_code(j) == balanced(j + 1)
    assert len({tuple(col) for col in c.T}) == 28_560
    assert not (c == 0).all(axis=0).any()


def test_row_words_hold_a1_and_the_step_bits_and_nothing_else():
    for k in (100, 1141, 2048):
        call, coins, feed = sampled(37, k, 8, seed=k)
        sh = call.shape
        assert feed.rows.shape == (sh.R, sh.H) and feed.rows.dtype == np.uint32
        for i in range(sh.R):
            want = sum(int(coins.a1[i, e]) << e for e in range(4))
            want += sum(int(coins.b[i, t]) << (t + 3) for t in range(1, sh.S))
            got = sum(int(w) << (32 * q) for q, w in enumerate(feed.rows[i]))
            assert got == want


def test_g_keep_and_its_digits():
    assert F.G_KEEP == 4_294_966_424
    assert F.g_digits(F.G_KEEP - 1) is not None
    assert F.g_digits(F.G_KEEP) is None and F.g_digits(2**32 - 1) is None
    assert {F.g_digits(v) for v in range(2401)} == set(itertools.product(range(7), repeat=4))
    assert F.g_digits(0) == (0, 0, 0, 0) and F.g_digits(2400) == (6, 6, 6, 6)
    assert F.g_digits(2401) == F.g_digits(0)


class Script:
    """A coin source that answers each request with the next scripted bytes, of exactly the length asked for."""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = list(chunks)
        self.asked: list[int] = []

    def __call__(self, n: int) -> bytes:
        self.asked.append(n)
        got = self.chunks.pop(0)
        assert len(got) == n
        return got


def test_g_is_sampled_by_rejection():
    sh = F.Shape(1, 1, 8)  # R 16, S 1, C 8, T 1
    u32 = lambda *ws: np.array(ws, dtype="<u4").tobytes()  # noqa: E731
    first = u32(F.G_KEEP, 2**32 - 1, 2400, 0, 1, 2, 3, 4)
    src = Script(bytes(16), bytes(64), first, u32(5, 6), bytes(3 * 8))
    coins = F.CallCoins.sample(sh, src)
    assert src.asked == [16, 64, 32, 8, 24]
    assert [F.g_digits(v) for v in (2400, 0, 1, 2, 3, 4, 5, 6)] == [tuple(col) for col in coins.g.T]
    feed = F.Feed.of(F.Call(0, 0, 0, 1, 1, 8), coins)
    assert feed.g[0] == 0x03030303 and feed.g[1] == 0xFDFDFDFD


def test_g_words_are_the_int8_tag_weights():
    _, coins, feed = sampled(16, 64, 40)
    G = coins.G()
    assert G.min() >= -3 and G.max() <= 3
    for j in range(40):
        assert [int(np.int8(np.uint8((int(feed.g[j]) >> (8 * e)) & 0xFF))) for e in range(4)] == list(G[:, j])


def test_lane_words_hold_the_coin_lanes_and_the_public_byte():
    call, coins, feed = sampled(100, 100, 12)
    sh, s = call.shape, coins.s()
    assert (s % 2 == 1).all() and np.abs(s).max() <= 127
    c3 = F.codes(sh.C)[3]
    assert feed.lanes_c.shape == (sh.T, sh.S, sh.C)
    for rt, t, j in itertools.product(range(sh.T), range(sh.S), range(sh.C)):
        w = int(feed.lanes_c[rt, t, j])
        assert [int(np.int8(np.uint8((w >> (8 * ell)) & 0xFF))) for ell in range(3)] == list(s[t, rt, :, j])
        assert int(np.int8(np.uint8(w >> 24))) == (c3[j] if t == 0 else 1)


def test_sigma_is_the_lane_terms_and_the_rest_is_leans_coin_term():
    """σ for a row's class is Σ_t Lean's `laneTerm`, so the word less σ keeps exactly step 0's T(i)·G + a₁·C and the later
    steps' 1 + b_t of Lean's `coinTerm`: what FOLD=1 serves beside A·B − β."""
    call, coins, feed = sampled(100, 100, 12, seed=3)
    sh, s, G = call.shape, coins.s(), coins.G()
    for i, j in itertools.product(range(sh.R), range(sh.C)):
        lane = [sum(rot_weight(i, ell) * int(s[t, i // 64, ell, j]) for ell in range(3)) for t in range(sh.S)]
        assert feed.sigma[i // 64, i % 2, j] == sum(lane)
        step0 = sum(row_tag(i)[q] * int(G[q, j]) for q in range(4)) + sum(int(coins.a1[i, q]) * col_code(j)[q] for q in range(4))
        coin_term = [(step0 if t == 0 else 1 + int(coins.b[i, t])) + lane[t] for t in range(sh.S)]
        assert sum(coin_term) - feed.sigma[i // 64, i % 2, j] == step0 + sum(1 + int(coins.b[i, t]) for t in range(1, sh.S))


def test_the_kernel_reads_lanes_and_sources_at_its_column_stride():
    call, _, feed = sampled(70, 60, 200)
    sh = call.shape
    n = sh.computed(128)
    assert (sh.C, n) == (200, 256)
    lanes, src = feed.lanes(n), feed.sources(n)
    assert (lanes.dtype, lanes.shape, src.dtype, src.shape) == (np.uint32, (sh.T * sh.S * n,), np.int32, (2 * sh.T * n,))
    for rt, t in itertools.product(range(sh.T), range(sh.S)):
        row = lanes[(rt * sh.S + t) * n:(rt * sh.S + t + 1) * n]
        assert (row[:200] == feed.lanes_c[rt, t]).all() and not row[200:].any()
    for rt, c in itertools.product(range(sh.T), range(2)):
        row = src[(rt * 2 + c) * n:(rt * 2 + c + 1) * n]
        assert (row[:200] == F.MAGIC - feed.sigma[rt, c]).all() and not row[200:].any()
    with pytest.raises(F.FeedError, match="stride 199"):
        feed.lanes(199)


def test_head_is_row_words_then_g_words():
    call, _, feed = sampled(20, 300, 24)
    head = feed.head()
    assert head.dtype == np.uint32 and len(head) == call.shape.feed_words()
    assert (head[:feed.rows.size] == feed.rows.reshape(-1)).all() and (head[feed.rows.size:] == feed.g).all()


def test_admits_is_leans():
    assert "K >= 1" in F.Shape(1, 0, 8).refusal()
    assert "no words" in F.Shape(0, 8, 8).refusal()
    assert F.Shape(16 * 2**20, 21, 8 * 2**20).refusal() is None  # exactly 2^40 tiles
    assert "2^40" in F.Shape(16 * 2**20, 22, 8 * 2**20).refusal()
    assert F.Shape(2**15, 2**16, 2**20).refusal() is None  # Lean's `admits_of_caps` corner


def test_coins_refuse_a_wrong_shape_dtype_or_range():
    call, coins, _ = sampled(16, 30, 8)
    ok = dict(shape=coins.shape, b=coins.b, a1=coins.a1, g=coins.g, u=coins.u)
    for name, bad in [("u", coins.u | 0x80), ("g", np.full_like(coins.g, 7)), ("b", coins.b.astype(np.int8)),
                      ("a1", coins.a1[:-1])]:
        with pytest.raises(F.FeedError, match=f"coins.{name}"):
            F.CallCoins(**{**ok, name: bad})
    with pytest.raises(F.FeedError, match="coins of"):
        F.Feed.of(F.Call(0, 0, 0, 17, 30, 8), coins)


def test_in_process_certifier_releases_once_after_the_root():
    asked: list[int] = []
    inner = rand_of(7)

    def rand(n: int) -> bytes:
        asked.append(n)
        return inner(n)

    cert = F.InProcessCertifier(rand)
    assert cert.source == F.IN_PROCESS == "in-process"
    for p in (3, 4):
        cert.open(p, b"\x07" * 64, b"\x06" * 64, 4)
    asked.clear()
    call = F.Call(3, 0, 5, 9, 2048, 3072)
    with pytest.raises(F.FeedError, match="root of 32 bytes"):
        cert.release(call, bytes(32))
    with pytest.raises(F.FeedError, match="K >= 1"):
        cert.release(F.Call(3, 1, 5, 9, 0, 3072), bytes(64))
    assert not asked and not cert.roots
    feed = cert.release(call, b"\x01" * 64)
    assert asked and feed.call == call and cert.roots == {(3, 0): b"\x01" * 64}
    with pytest.raises(F.FeedError, match="released already"):
        cert.release(call, b"\x02" * 64)
    assert cert.release(F.Call(4, 0, 5, 9, 2048, 3072), b"\x02" * 64).call.pass_index == 4


def test_in_process_certifier_samples_a_windows_salt_after_its_key_leaf_weights_root_and_call_count():
    """C3's order: a window's salt is sampled once, from the certifier's source, after it holds the window's key leaf,
    weights' root and call count. No coins are released before the salt, and none past the call count."""
    cert = F.InProcessCertifier(rand_of(5))
    call = F.Call(2, 0, 0, 4, 64, 8)
    with pytest.raises(F.FeedError, match="pass 2 call 0: before its window's salt"):
        cert.release(call, bytes(64))
    with pytest.raises(F.FeedError, match="a key leaf of 32 bytes and a weights' root of 64"):
        cert.open(2, bytes(32), bytes(64), 2)
    with pytest.raises(F.FeedError, match="a key leaf of 64 bytes and a weights' root of 32"):
        cert.open(2, bytes(64), bytes(32), 2)
    with pytest.raises(F.FeedError, match="a window of 0 calls"):
        cert.open(2, bytes(64), bytes(64), 0)
    assert not cert.windows
    salt = cert.open(2, b"\x07" * 64, b"\x06" * 64, 2)
    assert len(salt) == 64 and cert.windows[2] == F.WindowSalt(b"\x07" * 64, b"\x06" * 64, 2, salt)
    assert salt == bytes(rand_of(5)(64))
    with pytest.raises(F.FeedError, match="its window is open already; a window's salt is sampled once"):
        cert.open(2, b"\x07" * 64, b"\x06" * 64, 2)
    cert.release(call, bytes(64))
    cert.release(F.Call(2, 1, 0, 4, 64, 8), bytes(64))
    with pytest.raises(F.FeedError, match="pass 2 call 2: past the window's 2 calls"):
        cert.release(F.Call(2, 2, 0, 4, 64, 8), bytes(64))


def test_feed_is_a_function_of_the_coin_bytes():
    def release(seed: int) -> F.Feed:
        cert = F.InProcessCertifier(rand_of(seed))
        cert.open(0, bytes(64), bytes(64), 1)
        return cert.release(F.Call(0, 0, 0, 33, 200, 40), bytes(64))
    a, b, c = release(11), release(11), release(12)
    assert all((x == y).all() for x, y in zip((a.rows, a.g, a.lanes_c, a.sigma), (b.rows, b.g, b.lanes_c, b.sigma)))
    assert not (a.lanes_c == c.lanes_c).all()
