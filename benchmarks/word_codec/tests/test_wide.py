"""The wide word codecs' reference (`wide.py`): ρ over p = 3 mod 4 and p = 5 mod 8, round trips at each field, the
fused GEMM's store layout, the tags over the encoded words, and `wide.cu`'s constants."""

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
import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/wide/vectors").digest()
VECTORS = HERE / "wide_vectors.json"


def test_fields():
    for bits, f in W.FIELDS.items():
        assert f.p.bit_length() == bits and pow(3, f.p - 1, f.p) == 1
        if f.full_width(f.payload):
            assert 32 * f.limbs == bits == f.payload and f.payload_bound() == f.p
        else:
            assert 32 * f.limbs > bits >= 32 * (f.limbs - 1) and f.payload == bits - 1
            assert f.payload_bound() == 1 << f.payload
    assert [f.p % 8 for f in W.FIELDS.values()] == [7, 5, 7, 3, 7]
    assert [f.lane_words() for f in W.FIELDS.values()] == [17, 9, 4, 8, 4]
    assert [f.expansion() for f in W.FIELDS.values()] == [1.0625, 1.125, 1.0625, 1.0, 1.0]
    with pytest.raises(ValueError, match="payload"):
        W.FIELDS[127].check_payload(127)
    with pytest.raises(ValueError, match="payload"):
        W.FIELDS[127].check_payload(128)


@pytest.mark.parametrize("bits", [256, 512])
def test_a_full_width_word_holds_any_bytes_but_a_top_0xff(bits):
    f = W.FIELDS[bits]
    assert ((0xFE << (bits - 8)) | ((1 << (bits - 8)) - 1)) < f.p < 1 << bits
    assert W.root_split(f) == {256: (248, 6, 17), 512: (502, 8, 114)}[bits]
    k = W.key(f, SEED, 1)
    assert W.decode_word(W.encode_word(f.p - 1, 5, k), 5, k) == f.p - 1
    with pytest.raises(ValueError, match="under p"):
        W.encode_word(f.p, 5, k)


def small(p: int) -> W.Field:
    return W.Field("t", p.bit_length(), p, 1, 3)


@pytest.mark.parametrize("p", [7, 11, 19, 23, 127, 1019, 13, 29, 37, 53, 61, 101, 1013])
def test_rho_is_a_permutation_with_the_parity_rule(p):
    f = small(p)
    images = [W.rho(a, f) for a in range(p)]
    assert sorted(images) == list(range(p))
    for a, y in enumerate(images):
        assert W.rho_inv(y, f) == a
        assert (y % 2 == 0) == (W.legendre(a, p) != -1)
        if p % 8 == 5:
            assert W.rho_fast_5mod8(a, p) == y


def test_p255_constants_and_the_kernel_root():
    p = W.P255
    assert W.SQRT_M1 ** 2 % p == p - 1 and W.C1 ** 2 * 2 * W.SQRT_M1 % p == 1
    assert W.legendre(2, p) == -1
    f, rng = W.FIELDS[255], random.Random(2)
    for a in [0, 1, 2, p - 1, p - 2, (p - 1) // 2] + [rng.randrange(p) for _ in range(60)]:
        y = W.rho(a, f)
        assert W.rho_fast_5mod8(a) == y and W.rho_inv(y, f) == a


@pytest.mark.parametrize("bits", [127, 521, 256, 512])
def test_rho_three_mod_four(bits):
    f, rng = W.FIELDS[bits], random.Random(bits)
    for a in [0, 1, 2, f.p - 1, f.p - 2] + [rng.randrange(f.p) for _ in range(12)]:
        y = W.rho(a, f)
        assert W.rho_inv(y, f) == a and (y % 2 == 0) == (W.legendre(a, f.p) != -1)


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("bits", sorted(W.FIELDS))
def test_round_trip(bits, rounds):
    f = W.FIELDS[bits]
    k, rng = W.key(f, SEED, rounds), random.Random(bits * 10 + rounds)
    for m in [0, f.payload_bound() - 1] + [rng.getrandbits(f.payload) % f.payload_bound() for _ in range(3)]:
        i = rng.getrandbits(48)
        y = W.encode_word(m, i, k)
        assert 0 <= y < f.p and W.decode_word(y, i, k) == m == W.decode_raw(y, i, k)
    with pytest.raises(ValueError, match="payload"):
        W.encode_word(f.payload_bound(), 0, k)


def test_keys_and_tweaks():
    f = W.FIELDS[255]
    k = W.key(f, SEED, 3)
    assert W.key(f, SEED, 5).c[:3] == k.c and W.key(f, SEED, 5).tau == k.tau and 0 < k.tau < f.p
    assert W.key(W.FIELDS[127], SEED, 1).c[0] != k.c[0] % W.FIELDS[127].p
    assert len({W.encode_word(7, i, W.key(f, SEED, 1)) for i in range(16)}) == 16
    with pytest.raises(ValueError, match="seed"):
        W.key(f, b"short", 1)


def test_decode_rejects_a_non_payload():
    f = W.FIELDS[127]
    k = W.key(f, SEED, 1)
    rng = random.Random(3)
    bad = next(y for y in (rng.randrange(f.p) for _ in range(200)) if W.decode_raw(y, 0, k) >> f.payload)
    with pytest.raises(ValueError, match="decodes to no payload"):
        W.decode_word(bad, 0, k)
    with pytest.raises(ValueError, match="canonical"):
        W.decode_word(f.p, 0, k)


@pytest.mark.parametrize("bits", sorted(W.FIELDS))
def test_layout(bits):
    f = W.FIELDS[bits]
    n, kd = 16, 1024
    w = W.random_weights(n, kd, bits)
    lanes = W.fragment_payloads(w, n, kd, f)
    streams = C.fragment_stream(w, n, kd)
    assert len(lanes) == len(streams) == 2 * W.LANES
    for ms, s in zip(lanes, streams, strict=True):
        assert len(ms) == f.lane_words() and all(m >> f.payload == 0 for m in ms)
        assert W.unpack(ms, W.LANE_BYTES, f.payload) == s
    store = W.store_bytes(lanes, f)
    assert len(store) == 2 * f.record_bytes()
    lw = f.lane_words()
    rng = random.Random(1)
    for _ in range(20):
        q, lane, wi, e = rng.randrange(2), rng.randrange(32), rng.randrange(lw), rng.randrange(f.limbs)
        i = W.word_index(q, lane, wi, lw)
        at = 4 * (32 * f.limbs * (i >> 5) + 32 * e + (i & 31))
        assert store[at:at + 4] == W.limbs(lanes[32 * q + lane][wi], f)[e].to_bytes(4, "little")


def test_tags_are_ro_blake3_over_the_encoded_words():
    from verity.core.protocols.bandwidth import tags as T
    from verity.ml.crypto import blake3

    f = W.FIELDS[521]
    n, kd = 16, 512
    lanes = W.fragment_payloads(W.random_weights(n, kd, 4), n, kd, f)
    words = W.encode_store(lanes, W.key(f, SEED, 1))
    ys = W.tags_of(words, f, SEED, base=64)
    key = hashlib.sha512(T.prefix(T.TAG_DOMAIN, "ro-blake3/v1", SEED)).digest()[:32]
    for j in (0, 17, 31):
        block = W.block(words, j, f)
        assert len(block) == f.lane_words() * f.word_bytes
        assert ys[j] == blake3.keyed(key, (64 + j).to_bytes(8, "big") + block)
    assert len(W.leaves_of(ys, SEED)) == 2 and len(W.root_of(ys, SEED)) == 64


def test_batched_tags_cover_a_lane_of_adjacent_tasks():
    from verity.core.protocols.bandwidth import tags as T
    from verity.ml.crypto import blake3

    f = W.FIELDS[521]
    n, kd = 16, 1024
    words = W.encode_store(W.fragment_payloads(W.random_weights(n, kd, 5), n, kd, f), W.key(f, SEED, 1))
    ys = W.tags_of(words, f, SEED, base=32, tb=2)
    assert len(ys) == 32
    key = hashlib.sha512(T.prefix(T.TAG_DOMAIN, "ro-blake3/v1", SEED)).digest()[:32]
    for lane in (0, 13, 31):
        block = W.block(words, lane, f, tb=2)
        assert block == W.block(words, lane, f) + W.block(words, 32 + lane, f)
        assert len(block) == 552 - 8  # one BLAKE3 chunk: 9 compressions, the last over 40 bytes
        assert ys[lane] == blake3.keyed(key, (32 + lane).to_bytes(8, "big") + block)
    with pytest.raises(AssertionError):
        W.tags_of(words[:W.LANES], f, SEED, tb=2)


def test_kernel_constants_match():
    src = (HERE.parent / "wide.cu").read_text()
    for bits, f in W.FIELDS.items():
        m = re.search(rf"struct Fd<{bits}> {{\s*static constexpr int N = (\d+), M = (\d+), PB = (\d+);\s*"
                      rf"static constexpr uint32_t C = (\d+);", src)
        assert m and tuple(map(int, m.groups())) == (f.limbs, bits, f.payload, (1 << bits) - f.p)
    assert "constexpr int LANES = 32, TASK_TILES = 16, LANE_BYTES = 16 * TASK_TILES" in src
    assert W.root_squarings(W.FIELDS[255]) == 251 and W.root_squarings(W.FIELDS[521]) == 519
    assert W.root_chain(W.FIELDS[256]) == (257, 13) and W.root_chain(W.FIELDS[512]) == (515, 18)


def make_vectors() -> dict:
    out = {"seed": SEED.hex(), "domain": W.DOMAIN.hex(), "fields": {}}
    rng = random.Random(11)
    for bits, f in W.FIELDS.items():
        k2 = W.key(f, SEED, 2)
        words = []
        for k in (1, 2):
            key = W.key(f, SEED, k)
            for m in (0, f.payload_bound() - 1, rng.getrandbits(f.payload) % f.payload_bound()):
                i = rng.getrandbits(40)
                words.append({"k": k, "i": i, "m": hex(m), "y": hex(W.encode_word(m, i, key))})
        n, kd = 16, 512
        enc = W.encode_store(W.fragment_payloads(W.random_weights(n, kd, bits), n, kd, f), W.key(f, SEED, 1))
        ys = W.tags_of(enc, f, SEED)
        out["fields"][f.name] = {"p": hex(f.p), "c": [hex(c) for c in k2.c], "tau": hex(k2.tau), "words": words,
                                 "store_sha256": hashlib.sha256(W.store_bytes(enc, f)).hexdigest(),
                                 "tag0": ys[0].hex(), "root": W.root_of(ys, SEED).hex()}
    return out


def test_pinned_vectors():
    assert json.loads(VECTORS.read_text()) == make_vectors()


if __name__ == "__main__":
    VECTORS.write_text(json.dumps(make_vectors(), indent=1) + "\n")
