"""The sketch-then-hash tag over the wide store (`wide_sketch.py`): the vectors as the spec places each byte, the sign
matrix from its derivation, the fused GEMM's MMA arrangement against the plain product, and the tags, leaves and root."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import codec as C  # noqa: E402
import wide as W  # noqa: E402
import wide_sketch as S  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/wide/sketch-vectors").digest()


def random_store(f: W.Field, n_rows: int, kdim: int, seed: int) -> bytes:
    tasks = (n_rows // 16) * (kdim // W.TASK_COLS)
    return np.random.default_rng(seed).integers(0, 1 << 32, tasks * W.LANES * f.lane_words() * f.limbs,
                                                dtype=np.uint64).astype("<u4").tobytes()


def spec_byte(store: bytes, f: W.Field, kdim: int, v: int, pos: int) -> int:
    """Byte `pos` of vector `v` straight from the spec's wording."""
    nch, m, lw = kdim // W.TASK_COLS, S.steps(f), f.lane_words()
    tile, r = divmod(v, 16)
    G, kc = divmod(tile, nch)
    blk, col = divmod(pos, 32)
    i, s = divmod(blk, m)
    lane, reg, b = next((ln, rg, bb) for ln in range(32) for rg in range(4) for bb in range(4)
                        if C.fragment_offset(ln, rg, bb) == (r, col))
    q, x = (S.T * G + i) * nch + kc, 16 * s + 4 * reg + b
    if x >= 4 * lw * f.limbs:
        return 0
    rec = q * W.LANES * lw * f.limbs
    limb = int.from_bytes(store[4 * (rec + W.LANES * (x // 4) + lane):][:4], "little")
    return limb >> 8 * (x % 4) & 0xFF


def test_steps():
    assert [S.steps(f) for f in W.FIELDS.values()] == [17, 18, 17, 16, 16, 12]
    assert S.vector_bytes(W.FIELDS[512]) == 2048 and S.tiles(80, 1024) == 4
    src = (HERE.parent / "wide.cu").read_text()
    assert "constexpr int SK_K = 16, SK_M = (L * N + 3) / 4;" in src and "constexpr int GEMM_WARPS = 4;" in src
    assert f'SKETCH_NAME[] = "{S.NAME}"' in src and S.K == 16 and S.T == 4


@pytest.mark.parametrize("bits", [512, 521, 255])
def test_vectors_as_the_spec_places_each_byte(bits):
    f = W.FIELDS[bits]
    n_rows, kdim = 80, 1024  # five row blocks: the second tile row is short
    store = random_store(f, n_rows, kdim, bits)
    vs = S.vectors(store, n_rows, kdim, f)
    assert vs.shape == (16 * S.tiles(n_rows, kdim), S.vector_bytes(f))
    rng = np.random.default_rng(1)
    for v in [0, 17, vs.shape[0] - 1, *rng.integers(0, vs.shape[0], 6)]:
        for pos in [0, 31, 32 * S.steps(f) - 1, vs.shape[1] - 1, *rng.integers(0, vs.shape[1], 6)]:
            tile = v // 16
            G = tile // (kdim // W.TASK_COLS)
            i = pos // 32 // S.steps(f)
            want = 0 if S.T * G + i >= n_rows // 16 else spec_byte(store, f, kdim, int(v), int(pos))
            assert vs[v, pos].view(np.uint8) == want, (v, pos)
    # every byte of the store sits in exactly one vector
    total = np.frombuffer(store, np.uint8).astype(np.int64).sum()
    assert vs.view(np.uint8).astype(np.int64).sum() == total


def test_signs_from_the_derivation():
    from verity.ml.randomness import derive

    f = W.FIELDS[512]
    s = S.signs(SEED, f)
    n = S.vector_bytes(f)
    assert s.shape == (16, n) and set(np.unique(s)) == {-1, 1}
    key = derive(SEED, S.SIGN_DOMAIN, {"tag": S.NAME, "k": 16, "n": n})
    for c in (0, 7, 15):
        row = key.stream(("row", c), n // 8)
        for p in (0, 1, 9, n - 1):
            assert s[c, p] == (-1 if row[p // 8] >> (p % 8) & 1 else 1)


@pytest.mark.parametrize("bits", [512, 521])
def test_mma_arrangement_is_the_product(bits):
    f = W.FIELDS[bits]
    n_rows, kdim = 80, 1024
    store = random_store(f, n_rows, kdim, 7 + bits)
    s = S.signs(SEED, f)
    frag = S.sign_fragments(s, S.steps(f))
    assert frag.size == S.T * S.steps(f) * 32 * 16
    want = S.sketch(S.vectors(store, n_rows, kdim, f), s)
    assert np.array_equal(S.sketch_from_fragments(store, n_rows, kdim, f, frag), want)
    assert np.abs(want).max() < 128 * S.vector_bytes(f)


def test_tags_leaves_root():
    from verity.core.protocols.bandwidth import tags as TG
    from verity.ml.crypto import blake3

    f = W.FIELDS[512]
    store = random_store(f, 64, 1024, 3)
    sk = S.sketch(S.vectors(store, 64, 1024, f), S.signs(SEED, f))
    ys = S.tags_of(SEED, sk, jbase=1000)
    assert len(ys) == 32 and S.tags_of(SEED, sk, 1000, which=[5]) == [ys[5]]
    key = hashlib.sha512(TG.prefix(TG.TAG_DOMAIN, S.NAME, SEED)).digest()[:32]
    assert ys[5] == blake3.keyed(key, (1005).to_bytes(8, "big") + sk[5].astype("<i4").tobytes())
    lv = S.leaves(SEED, ys)
    assert len(lv) == 2
    assert S.root(SEED, ys) == hashlib.sha512(b"\x01" + lv[0] + lv[1]).digest()
    with pytest.raises(ValueError, match="int32"):
        S.tags_of(SEED, np.array([[1 << 31] + [0] * 15]))
