"""The word codec's reference (`codec.py`) and the kernel's arithmetic twin (`twin.py`) against it."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import codec as C  # noqa: E402
import twin as T  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/vectors").digest()
EDGES = [0, 1, 2, 3, C.P - 1, C.P - 2, (C.P - 1) // 2, (C.P + 1) // 2, 1 << 60, (1 << 56) - 1, 1 << 56, (1 << 32) - 1]


def test_constants():
    assert C.P == (1 << 61) - 1 and C.P % 4 == 3
    assert pow(2, C.P - 1, C.P) == 1 and C.ROOT_EXP == 1 << 59
    assert (C.LANE_BYTES, C.LANE_WORDS, C.TASK_WORDS, C.TASK_COLS) == (256, 37, 1184, 512)
    assert C.lane_words(60) == 35
    with pytest.raises(ValueError, match="payload bits"):
        C.lane_words(61)


def test_kernel_constants_match():
    src = (HERE.parent / "wc.cu").read_text()
    assert "constexpr int TASK_TILES = 16;" in src and "constexpr int LANES = 32;" in src
    assert re.search(r"__launch_bounds__\(GEMM_WARPS \* 32\)", src)


@pytest.mark.parametrize("p", [7, 11, 19, 23, 31, 43, 127, 8191])
def test_rho_is_a_permutation_with_the_parity_rule(p):
    images = [C.rho(a, p) for a in range(p)]
    assert sorted(images) == list(range(p))
    for a, y in enumerate(images):
        assert C.rho_inv(y, p) == a
        assert (y % 2 == 0) == (pow(a, (p - 1) // 2, p) != p - 1)


def test_rho_on_p61():
    rng = random.Random(1)
    for a in EDGES[:8] + [rng.randrange(C.P) for _ in range(200)]:
        y = C.rho(a)
        assert C.rho_inv(y) == a and C.rho(C.rho_inv(a)) == a
        assert (y % 2 == 0) == (pow(a, (C.P - 1) // 2, C.P) != C.P - 1)


def test_key_derivation():
    k = C.key(SEED, 3)
    assert k.rounds == 3 and all(0 <= c < C.P for c in k.c) and 0 < k.tau < C.P
    assert C.key(SEED, 5).c[:3] == k.c and C.key(SEED, 5).tau == k.tau
    with pytest.raises(ValueError, match="seed"):
        C.key(b"short", 1)
    with pytest.raises(ValueError, match="round"):
        C.key(SEED, 0)


@pytest.mark.parametrize("rounds", range(1, 9))
@pytest.mark.parametrize("bits", C.PACKINGS)
def test_round_trip(rounds, bits):
    k = C.key(SEED, rounds)
    rng = random.Random(rounds)
    for _ in range(40):
        m, i = rng.getrandbits(bits), rng.getrandbits(40)
        y = C.encode_word(m, i, k, bits)
        assert 0 <= y < C.P and C.decode_word(y, i, k, bits) == m
    with pytest.raises(ValueError, match="payload"):
        C.encode_word(1 << bits, 0, k, bits)


def test_pack_is_the_little_endian_bit_stream():
    rng = random.Random(9)
    for length in (0, 1, 7, 14, 15, 16, 100, 256):
        data = bytes(rng.getrandbits(8) for _ in range(length))
        for bits in C.PACKINGS:
            words = C.pack(data, bits)
            assert len(words) == -(-8 * length // bits) and C.unpack(words, length, bits) == data
            assert sum(m << (bits * i) for i, m in enumerate(words)) == int.from_bytes(data, "little")
        assert C.pack(data) == [int.from_bytes(data[o:o + 7].ljust(7, b"\0"), "little") for o in range(0, length, 7)]


def test_tweak_separates_indices():
    k = C.key(SEED, 2)
    assert len({C.encode_word(12345, i, k) for i in range(64)}) == 64


def test_decode_rejects_a_non_payload():
    k = C.key(SEED, 1)
    bad = next(y for y in range(1000, 2000) if (C.rho_inv(y) - k.c[0] - k.tweak(0)) % C.P >> 56)
    with pytest.raises(ValueError, match="decodes to no payload"):
        C.decode_word(bad, 0, k)
    with pytest.raises(ValueError, match="canonical"):
        C.decode_word(C.P, 0, k)


def test_pinned_vectors():
    v = json.loads((HERE / "vectors.json").read_text())
    assert bytes.fromhex(v["seed"]) == SEED and v["p"] == C.P
    for k, kv in v["keys"].items():
        key = C.key(SEED, int(k))
        assert list(key.c) == kv["c"] and key.tau == kv["tau"]
    keys = {int(k): C.key(SEED, int(k)) for k in v["keys"]}
    for w in v["words"]:
        assert C.encode_word(w["m"], w["i"], keys[w["k"]]) == w["y"]
        assert C.decode_word(w["y"], w["i"], keys[w["k"]]) == w["m"]


def test_flat_store_round_trip():
    k = C.key(SEED, 2)
    data = bytes(random.Random(3).getrandbits(8) for _ in range(100))
    store = C.encode_flat(data, k)
    assert len(store) == 8 * 15 and C.decode_flat(store, k, len(data)) == data


def test_fragment_offsets_follow_the_isa():
    # PTX ISA, mma.m16n8k32 with 8-bit A: a_i row groupID (+8 for i in 4..7, 12..15), column 4 threadID_in_group + i % 4 (+16
    # for i >= 8); register r holds a_(4r) .. a_(4r + 3).
    for lane in range(32):
        g, q = lane >> 2, lane & 3
        for i in range(16):
            r, b = divmod(i, 4)
            row = g + (8 if i in range(4, 8) or i >= 12 else 0)
            col = 4 * q + (i & 3) + (16 if i >= 8 else 0)
            assert C.fragment_offset(lane, r, b) == (row, col)


@pytest.mark.parametrize("bits", C.PACKINGS)
def test_fragment_layout_is_a_bijection(bits):
    n, kdim = 32, 1024
    rng = random.Random(4)
    w = bytes(rng.getrandbits(8) for _ in range(n * kdim))
    lanes = C.fragment_stream(w, n, kdim)
    assert len(lanes) == (n // 16) * (kdim // 512) * 32
    seen = sorted(b for s in lanes for b in s)
    assert seen == sorted(w)
    plain = C.plain_fragments(w, n, kdim)
    assert sorted(plain) == sorted(w)
    payloads = C.fragment_payloads(w, n, kdim, bits)
    lw = C.lane_words(bits)
    assert len(payloads) == len(lanes) * lw
    # each lane's payloads re-form its 64 registers, and the registers are its 16 tiles' fragments in plain order
    for idx in (0, 5, 37, len(lanes) - 1):
        task, lane = divmod(idx, 32)
        mine = [payloads[(task * lw + wi) * 32 + lane] for wi in range(lw)]
        regs = T.lane_registers(mine, bits)
        assert b"".join(r.to_bytes(4, "little") for r in regs) == lanes[idx]
        for tt in range(16):
            o = (task * 16 + tt) * 512 + 16 * lane
            assert plain[o:o + 16] == lanes[idx][16 * tt:16 * tt + 16]
    with pytest.raises(ValueError, match="multiple"):
        C.fragment_stream(bytes(16 * 500), 16, 500)


@pytest.mark.parametrize("bits", C.PACKINGS)
def test_blocks_are_a_lane_of_a_task(bits):
    lw = C.lane_words(bits)
    tasks = 3
    seen = sorted(i for j in range(32 * tasks) for i in C.block_indices(j, bits))
    assert seen == list(range(tasks * lw * 32))
    assert C.block_indices(32 * 2 + 5, bits) == [2 * lw * 32 + 32 * w + 5 for w in range(lw)]
    store = list(range(tasks * lw * 32))
    assert C.lane_block(store, 7, bits) == C.words_bytes([32 * w + 7 for w in range(lw)])


@pytest.mark.parametrize("bits", C.PACKINGS)
def test_twin_lane_tag_is_ro_blake3(bits):
    """The kernel's message (u64be j, then the words as u32 pairs), lengths and flags give ro-blake3/v1's tag."""
    from verity.core.protocols.bandwidth import tags
    from verity.ml.crypto import blake3

    scheme = tags.get("ro-blake3/v1")
    seed = hashlib.sha256(b"verity/word-codec/tests/tag-seed").digest()
    key = scheme.key(seed)
    keyed = scheme.keyed(seed)
    rng = random.Random(bits)
    for j in (0, 1, 31, 32 * 4097 + 17, (1 << 40) + 3, (1 << 64) - 1):
        words = [rng.randrange(C.P) for _ in range(C.lane_words(bits))]
        cv = tuple(int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8))
        msgs = T.b3_lane_messages(j, words)
        assert len(msgs) == 5 and msgs[-1][1] == (48 if bits == 56 else 32)
        for m, length, flags in msgs:
            cv = tuple(blake3.compress(cv, b"".join(v.to_bytes(4, "little") for v in m), 0, length, flags)[:8])
        assert b"".join(v.to_bytes(4, "little") for v in cv) == keyed(j, C.words_bytes(words))


@pytest.mark.parametrize("form", [0, 1])
def test_twin_squaring_at_the_edges_of_its_range(form):
    rng = random.Random(10 + form)
    lazy = [T.LAZY - 1 - d for d in range(8)] + [(1 << 61) + d for d in range(8)]
    for a in EDGES[:8] + lazy + [rng.randrange(T.LAZY) for _ in range(2000)]:
        if a < C.P or a < T.LAZY:
            T.sqr(a, form)


@pytest.mark.parametrize("form", [0, 1])
def test_twin_decode_and_encode_equal_the_reference(form):
    rng = random.Random(20 + form)
    for rounds in (1, 2, 5):
        k = C.key(SEED, rounds)
        for _ in range(12):
            m, i = rng.getrandbits(56), rng.getrandbits(40)
            t = k.tweak(i)
            y = T.encode_word(m, t, k.c, form)
            assert y == C.encode_word(m, i, k)
            assert T.decode_word(y, t, k.c, form) == m
        for y in EDGES[:8]:
            want = y
            for c in reversed(k.c):
                want = (C.rho_inv(want) - c - k.tweak(3)) % C.P
            assert T.decode_word(y, k.tweak(3), k.c, form) == want


def test_twin_canon_and_fold_ranges():
    for r in [0, 1, C.P - 1, C.P, C.P + 1, T.LAZY - 1]:
        T.canon(r)
    for z in [0, 1, T.U64, T.U64 - 7, 1 << 63, (1 << 61) - 1]:
        assert T.fold(z) < (1 << 61) + 8 and T.fold(z) % C.P == z % C.P


def test_twin_horner():
    rng = random.Random(30)
    for _ in range(1000):
        T.horner31(rng.getrandbits(32), rng.randrange(1 << 31), rng.getrandbits(32))
    T.horner31(T.U32, (1 << 31) - 1, T.U32)


def test_bound_reproduces_the_draft():
    b = C.bound(eps=0.5, r=58, tau=0.1, b_side=63, read_rate=1476)
    assert abs(b["rebuild"] - 0.4 / (0.5 * 57)) < 1e-12 and abs(b["side"] - 63 / 1476) < 1e-12
    assert abs(C.bound(0.5, 58, 0.1, 63, 1476, theta=0.1)["rebuild"] - 0.5 / (0.5 * 57)) < 1e-12
    with pytest.raises(ValueError):
        C.bound(0.0, 58, 0.1, 63, 1476)
