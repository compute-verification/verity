"""Sketch-then-hash's reference (`sketch.py`): the sign derivation from the window's seed, the vectors, the exact sketch,
the device's sign packing, and the tag, leaves and root under ro-sketch-blake3/v0. `vectors.json` pins the bytes
(`SKETCH_REGEN=1` rewrites it)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import sketch as S  # noqa: E402
from sketch import C  # noqa: E402

from verity.core.protocols.bandwidth import tags  # noqa: E402
from verity.ml.crypto import blake3  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402

SEED = hashlib.sha256(b"verity/sketch-tag/vectors").digest()
VECTORS = HERE / "vectors.json"


def weights(n: int, kd: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n * kd, dtype=np.uint8).tobytes()


def encoded(n: int, kd: int, bits: int, seed: int) -> bytes:
    """An encoded store's shape: L·32 canonical words a task (the sketch reads the words, whatever they decode to)."""
    words = (n // 16) * (kd // C.TASK_COLS) * C.lane_words(bits) * C.LANES
    return np.random.default_rng(seed).integers(0, C.P, words, dtype=np.uint64).astype("<u8").tobytes()


def test_sign_key_is_derive():
    key = S.sign_key(SEED, 16, 4096)
    assert key == derive(SEED, "verity/bandwidth/sketch-signs/v0", {"tag": "ro-sketch-blake3/v0", "k": 16, "n": 4096})
    assert S.sign_key(SEED, 16, 4096) != S.sign_key(SEED, 16, 4097) != S.sign_key(SEED, 8, 4096)
    with pytest.raises(ValueError):
        S.sign_key(SEED, 12, 4096)


def test_sign_rows_are_the_stream_bits():
    k, n = 8, 1000
    key, s = S.sign_key(SEED, k, n), S.signs(SEED, k, n)
    assert s.shape == (k, n) and s.dtype == np.int8 and set(np.unique(s)) == {-1, 1}
    for c in range(k):
        row = key.stream(("row", c), -(-n // 8))
        want = [-1 if row[p // 8] >> (p % 8) & 1 else 1 for p in range(n)]
        assert s[c].tolist() == want
    assert abs(int((s == -1).sum()) - k * n // 2) < 4 * np.sqrt(k * n)


def test_steps_and_lengths():
    assert S.steps(S.PLAIN) == 16 and S.steps(56) == 19 and S.steps(60) == 18
    assert [S.vector_bytes(S.PLAIN, t) for t in (2, 8, 32)] == [1024, 4096, 16384]
    assert S.vector_bytes(56, 7) == 4256 and S.vector_bytes(60, 7) == 4032


def test_plain_vectors_are_weight_rows():
    n, kd, t = 48, 1024, 2
    w = weights(n, kd, 1)
    vs = S.vectors(C.plain_fragments(w, n, kd), n, kd, S.PLAIN, t)
    wm = np.frombuffer(w, np.uint8).reshape(n, kd).view(np.int8)
    nch = kd // C.TASK_COLS
    assert vs.shape == (S.tiles(n, kd, t) * 16, 1024)
    for v in range(vs.shape[0]):
        (g, kc), r = divmod(v // 16, nch), v % 16
        for i in range(t):
            nb = g * t + i
            want = wm[16 * nb + r, 512 * kc:512 * kc + 512] if nb < n // 16 else np.zeros(512, np.int8)
            assert np.array_equal(vs[v, 512 * i:512 * i + 512], want)


@pytest.mark.parametrize("bits", [56, 60])
def test_encoded_vectors_follow_the_fragments(bits):
    """Position 32 (i m + s) + col of vector r is lane byte 16 s + 4 reg + b of the lane whose fragment holds (r, col),
    the lane's words little-endian, zero past them."""
    n, kd, t = 48, 1024, 2
    store = encoded(n, kd, bits, 2)
    vs = S.vectors(store, n, kd, bits, t)
    lw, m, nch = C.lane_words(bits), S.steps(bits), kd // C.TASK_COLS
    words = np.frombuffer(store, "<u8")
    where = {C.fragment_offset(lane, reg, b): (lane, 4 * reg + b) for lane in range(32) for reg in range(4)
             for b in range(4)}
    rng = random.Random(bits)
    for _ in range(400):
        v, p = rng.randrange(vs.shape[0]), rng.randrange(vs.shape[1])
        (g, kc), r = divmod(v // 16, nch), v % 16
        step, col = divmod(p, 32)
        i, s = divmod(step, m)
        nb = g * t + i
        lane, lb = where[(r, col)]
        byte = 16 * s + lb
        if nb >= n // 16 or byte >= 8 * lw:
            want = 0
        else:
            q = nb * nch + kc
            want = int(words[q * lw * 32 + (byte // 8) * 32 + lane]) >> (8 * (byte % 8)) & 0xFF
        assert int(vs[v, p]) & 0xFF == want


@pytest.mark.parametrize("bits", [S.PLAIN, 56, 60])
def test_vectors_partition_the_store(bits):
    n, kd, t = 48, 1024, 2
    shape = S.lane_steps(bytes(len(C.plain_fragments(bytes(n * kd), n, kd)) if bits == S.PLAIN
                               else len(encoded(n, kd, bits, 0))), n, kd, bits).shape
    idx = np.arange(1, int(np.prod(shape)) + 1, dtype=np.int64).reshape(shape)
    flat = S.arrange(idx, t).reshape(-1)
    nz = flat[flat != 0]
    assert sorted(nz.tolist()) == list(range(1, idx.size + 1))


def test_sketch_is_exact():
    rng = np.random.default_rng(3)
    vs = rng.integers(-128, 128, (24, 300), dtype=np.int64).astype(np.int8)
    s = S.signs(SEED, 8, 300)
    sk = S.sketch(vs, s)
    for v in range(24):
        for c in range(8):
            assert sk[v, c] == sum(int(vs[v, p]) * int(s[c, p]) for p in range(300))
    assert np.array_equal(S.sketch_f64(vs, s, rows=7), sk)
    extreme = np.full((1, 4096), -128, np.int8)
    assert np.abs(S.sketch(extreme, np.ones((16, 4096), np.int8))).max() == 128 * 4096 == 1 << 19


@pytest.mark.parametrize("bits,k", [(S.PLAIN, 8), (S.PLAIN, 16), (S.PLAIN, 32), (56, 16), (60, 32), (56, 8)])
def test_sign_fragments_are_the_mma_operand(bits, k):
    n, kd, t = 48, 1024, 2
    store = C.plain_fragments(weights(n, kd, 4), n, kd) if bits == S.PLAIN else encoded(n, kd, bits, 4)
    m = S.steps(bits)
    nn = 32 * m * t
    sb = S.sign_bits(SEED, k, nn)
    frag = S.sign_fragments(sb, m, t)
    gr = S.sign_groups(k, m)
    assert frag.size == t * gr * 32 * 16 and 24 * gr >= 2 * m * (k // 8) > 24 * (gr - 1)
    want = S.sketch(S.vectors(store, n, kd, bits, t), S.signs(SEED, k, nn))
    assert np.array_equal(S.sketch_from_fragments(store, n, kd, bits, t, frag, k), want)
    # the packing's formula, bit by bit, and the device's byte permute of it
    words = frag.view("<u4")
    assert not any(int(w) & 0x88888888 for w in words)
    lo, hi = (0xFF01FF01, 0xFFFF0101, 0x01010101), (0xFF01FF01, 0xFFFF0101, 0xFFFFFFFF)
    rng = random.Random(k + bits)
    for _ in range(200):
        i, s, lane, blk = rng.randrange(t), rng.randrange(m), rng.randrange(32), rng.randrange(k // 8)
        g, tq = lane >> 2, lane & 3
        for h in range(2):
            r = 2 * (s * (k // 8) + blk) + h
            u, c = divmod(r // 6, 4)
            word = int(words[((i * gr + u) * 32 + lane) * 4 + c])
            sel, e = word >> 16 * ((r % 6) // 3) & 0xFFFF, r % 3
            pool = lo[e].to_bytes(4, "little") + hi[e].to_bytes(4, "little")
            reg = bytes(pool[sel >> 4 * q & 7] for q in range(4))
            for q in range(4):
                bit = word >> (16 * ((r % 6) // 3) + 4 * q + e) & 1
                assert bit == sb[8 * blk + g, 32 * (i * m + s) + 16 * h + 4 * tq + q]
                assert reg[q] == (0xFF if bit else 0x01)


def test_tag_is_blake3_keyed_under_the_name():
    sk = np.array([[1, -2, 3, -4, 5, -6, 7, 1 << 30]], np.int64)
    (y,) = S.tags_of(SEED, sk, jbase=5)
    key = hashlib.sha512(tags.prefix(tags.TAG_DOMAIN, "ro-sketch-blake3/v0", SEED)).digest()[:32]
    assert y == blake3.keyed(key, (5).to_bytes(8, "big") + sk.astype("<i4").tobytes())
    assert y != tags.Blake3Tag().keyed(SEED)(5, sk.astype("<i4").tobytes())
    with pytest.raises(ValueError):
        S.sketch_words(np.array([[1 << 31]]))


def pinned() -> dict:
    out = {"seed": SEED.hex(), "cases": []}
    for bits, k, t in ((S.PLAIN, 16, 8), (56, 16, 2), (60, 8, 2), (S.PLAIN, 32, 2)):
        n, kd = 32, 1024
        store = C.plain_fragments(weights(n, kd, 9), n, kd) if bits == S.PLAIN else encoded(n, kd, bits, 9)
        nn = S.vector_bytes(bits, t)
        sk = S.sketch(S.vectors(store, n, kd, bits, t), S.signs(SEED, k, nn))
        ys = S.tags_of(SEED, sk)
        out["cases"].append({
            "bits": bits, "k": k, "T": t, "n": nn, "N": n, "K": kd,
            "sign_key": S.sign_key(SEED, k, nn).value.hex(),
            "signs_sha256": hashlib.sha256(S.sign_bits(SEED, k, nn).tobytes()).hexdigest(),
            "store_sha256": hashlib.sha256(store).hexdigest(),
            "sketch_first": sk[0].tolist(), "sketch_sha256": hashlib.sha256(S.sketch_words(sk).tobytes()).hexdigest(),
            "tag_first": ys[0].hex(), "root": S.root(SEED, ys).hex()})
    return out


def test_vectors_pinned():
    got = pinned()
    if os.environ.get("SKETCH_REGEN"):
        VECTORS.write_text(json.dumps(got, indent=1) + "\n")
    assert json.loads(VECTORS.read_text()) == got
