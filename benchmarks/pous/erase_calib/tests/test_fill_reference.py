"""CPU invariants of the erasure calibration's fill stream and digest: the ChaCha block function is RFC 8439's, every
layout is a bijection onto the buffer, the vectorized block hash is `hbm_audit/digest.py`'s, and the streams are pinned
so a change to their definition is caught without a GPU. The device kernels are checked bit-exactly against these
references at run time by `driver.py validate`, which aborts the job on a mismatch."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fill as F  # noqa: E402


def test_chacha20_block_is_rfc8439():
    """RFC 8439 section 2.3.2: key 00..1f, counter 1, nonce 00:00:00:09:00:00:00:4a:00:00:00:00."""
    key = [int.from_bytes(bytes(range(4 * i, 4 * i + 4)), "little") for i in range(8)]
    nonce = [0x09000000, 0x4A000000, 0x00000000]
    state = np.array([*F.SIGMA, *key, 1, *nonce], dtype=np.uint32)[:, None]
    want = [0xE4E7F110, 0x15593BD1, 0x1FDD0F50, 0xC47120A3, 0xC7F4D1C7, 0x0368C033, 0x9AAA2204, 0x4E6CD4C3,
            0x466482D2, 0x09AA9F07, 0x05D7C214, 0xA2028BD9, 0xD19C12B5, 0xB94E16DE, 0xE883D0CB, 0x4E3C50A2]
    assert F.chacha_block(state, 20)[:, 0].tolist() == want


def test_chacha_states_put_the_counter_after_the_key():
    key = tuple(range(100, 110))
    st = F.chacha_states(key, np.array([(7 << 32) | 5], dtype=np.uint64))[:, 0].tolist()
    assert st == [*F.SIGMA, *range(100, 108), 5, 7, 108, 109]


@pytest.mark.parametrize("layout", list(F.LAYOUTS))
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 64])
def test_every_layout_is_a_bijection(layout: str, n: int):
    phys = F.physical_index(layout, n)
    assert np.array_equal(np.sort(phys), np.arange(n * F.WORDS))


def test_blockstride_moves_whole_blocks_and_scatters_them():
    n = 64
    phys = F.physical_index("blockstride", n).reshape(n, F.WORDS)
    starts = phys[:, 0]
    assert np.array_equal(phys - starts[:, None], np.broadcast_to(np.arange(F.WORDS), (n, F.WORDS)))
    assert (starts % F.WORDS == 0).all() and len(set((starts // F.WORDS).tolist())) == n
    assert not np.array_equal(starts // F.WORDS, np.arange(n))


def test_stride_a_is_coprime():
    import math

    for n in [3, 4, 5, 8, 64, 1000, 1 << 20, 1_536_000]:
        a = F.stride_a(n)
        assert 0 < a < n and math.gcd(a, n) == 1


def test_powers_into_matches_pow():
    for base in [2, 3, 0x7FFFFFFE, 123456789]:
        for n in [1, 2, 7, 1024, 1025]:
            got = F.powers_into(np.empty(n, dtype=np.int64), base)
            assert got.tolist() == [pow(base, i, F.MOD) for i in range(n)]


def test_vectorized_block_hash_is_digest_pys():
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 1 << 32, size=(3, F.WORDS), dtype=np.uint64).astype(np.uint32)
    g, _, _ = F.digest_key(1, 2)
    got = F.reference_block_hashes(blocks, g)
    assert got.tolist() == [F.D.reference_block_hash(blocks[i], g) for i in range(3)]


def test_digest_depends_on_the_order_and_on_every_block():
    blocks = F.reference_fill("mix", F.fill_key(1, 1), 8, "seq")
    g, z, _ = F.digest_key(1, 1)
    idx = np.arange(8)
    base = F.reference_digest(blocks, idx, g, z)
    assert F.reference_digest(blocks, idx[::-1].copy(), g, z) != base
    for b in range(8):
        pert = blocks.copy()
        pert[b, 100] ^= np.uint32(1)
        assert F.reference_digest(pert, idx, g, z) != base


def test_chunked_stream_continues_the_whole_stream():
    key = F.fill_key(5, 5)
    for gen in F.GENERATORS:
        whole = F.logical_words(gen, key, 6)
        assert np.array_equal(F.logical_words(gen, key, 2, block_base=4), whole[4 * F.WORDS :])


def test_keys_are_fresh_per_run():
    assert F.fill_key(1, 1) != F.fill_key(1, 2) and F.fill_key(1, 1) != F.fill_key(2, 1)
    g, z, _ = F.digest_key(1, 1)
    assert 2 <= g < F.MOD and 2 <= z < F.MOD


# sha256 of the first two blocks of each stream under fill_key(20261004, 0), seq layout: the streams' definition
PINNED = {
    "mix": "42884dc643f0724e9c5b42199526330a6c99e590b44314abef9748bc52d29f06",
    "chacha8": "b75abd8f626ddb80d484a5bea6286480373d132c4c5f082708c703844c788e94",
    "chacha12": "8984cb47446a3232bbc8563b206f38a963edde00ea2008858198d3518d116c10",
    "chacha20": "6efa9013a5b7a40479ee94a3d81f7f70a11263507bd15260baaab9052d44cfd0",
}


def test_chacha20_stream_is_openssls():
    """Below 2^32 blocks the stream's ChaCha20 is RFC 8439's with nonce `(ctr_hi, nonce[0], nonce[1])`: lane `l` of
    block `b` is keystream block `b * LANES + l`."""
    algorithms = pytest.importorskip("cryptography.hazmat.primitives.ciphers.algorithms")
    from cryptography.hazmat.primitives.ciphers import Cipher

    key = F.fill_key(9, 9)
    kb = b"".join(w.to_bytes(4, "little") for w in key[:8])
    nonce = (0).to_bytes(4, "little") + key[8].to_bytes(4, "little") + key[9].to_bytes(4, "little")
    for b, l in [(0, 0), (0, 1), (3, 1023), (17, 512)]:
        ctr = b * F.LANES + l
        ks = Cipher(algorithms.ChaCha20(kb, ctr.to_bytes(4, "little") + nonce), mode=None).encryptor().update(bytes(64))
        words = F.logical_words("chacha20", key, b + 1).reshape(-1, F.CHACHA_WORDS, F.LANES)[b, :, l]
        assert words.astype("<u4").tobytes() == ks


@pytest.mark.parametrize("gen", list(F.GENERATORS))
def test_streams_are_pinned(gen: str):
    words = F.reference_fill(gen, F.fill_key(20261004, 0), 2, "seq")
    assert hashlib.sha256(words.tobytes()).hexdigest() == PINNED[gen]
