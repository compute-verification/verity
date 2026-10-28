"""`sketching`'s ro-sketch-blake3/v0 over `wt.cu`'s store: the MMA arithmetic the sketch warps run (A from a lane's
stored bytes, B from the prmt-expanded sign words) equals S · e_v over `vectors`, `vectors` puts every stored byte where
the module docstring says (and nowhere else), and the signs and tags are pinned."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketching as SK  # noqa: E402

SEED = bytes(range(32))


def _store(nbs: int, nchunks: int, sb: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, nbs * nchunks * sb, dtype=np.uint8)


@pytest.mark.parametrize("sb", [8192, 8256])
@pytest.mark.parametrize("k", [16, 64])
def test_fragments_equal_vectors(sb: int, k: int):
    nbs, nchunks, groups, t = 5, 2, 3, 2
    store, m = _store(nbs, nchunks, sb), SK.steps(sb)
    bits = SK.sign_bits(SEED, k, SK.vector_bytes(sb, t))
    want = SK.sketch(SK.vectors(store, nbs, nchunks, sb, groups, t), bits)
    got = SK.fragment_sketch(store, nbs, nchunks, sb, groups, t, SK.sign_fragments(bits, m, t), k)
    assert want.shape == (groups * nchunks * SK.tiles_per_cta(nbs, groups, t) * 16, k)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("t", [1, 2, 8])
def test_b_fragments_are_the_prmt_registers(t: int):
    """The sketch warps' fragments (wt_sign_frags: registers 0, 1, 2 the prmt of a word at e = 0, 1, 2, register 3 of
    the word >> 16 at e = 0) are the B registers the prmt expansion of `sign_fragments` gives, and they carry S."""
    m = SK.steps(8192)
    bits = SK.sign_bits(SEED, 16, SK.vector_bytes(8192, t))
    words, frags = SK.sign_fragments(bits, m, t), SK.b_fragments(bits, m, t)
    want = [SK.prmt(*SK.PRMT[r % 3], (int(w) >> (16 if r == 3 else 0)) & 0xFFFF) for w in words for r in range(4)]
    assert np.array_equal(frags, np.array(want, dtype=np.uint32))
    signs = frags.view(np.uint8).view(np.int8).reshape(t, m, 8, 4, 2, 2, 4).astype(np.int64)   # ii, s, g, tq, blk, h, q
    assert np.array_equal(signs.transpose(4, 2, 0, 1, 5, 3, 6).reshape(16, -1), 1 - 2 * bits.astype(np.int64))


@pytest.mark.parametrize("sb", [8192, 8256])
def test_vector_positions(sb: int):
    nbs, nchunks, groups, t = 7, 3, 3, 2
    idx = np.arange(1, nbs * nchunks * sb + 1, dtype=np.int64)       # 0 marks padding
    got = sum(SK.vectors(((idx >> 8 * b) & 0xFF).astype(np.uint8), nbs, nchunks, sb, groups, t).view(np.uint8)
              .astype(np.int64) << 8 * b for b in range(3))
    m, tpc = SK.steps(sb), SK.tiles_per_cta(nbs, groups, t)
    want = np.zeros_like(got)
    for g0 in range(groups):
        for kc in range(nchunks):
            b = g0 * nchunks + kc
            for i, G in enumerate(range(g0, nbs, groups)):
                j, ii = divmod(i, t)
                for r in range(16):
                    for s in range(m):
                        for kk in range(32):
                            lane, reg = 4 * (r % 8) + (kk % 16) // 4, 2 * (kk // 16) + r // 8
                            off = 512 * s + 16 * lane + 4 * reg + kk % 4
                            if off < sb:
                                want[16 * (b * tpc + j) + r, 32 * (ii * m + s) + kk] = (G * nchunks + kc) * sb + off + 1
    assert np.array_equal(got, want)
    assert np.array_equal(np.sort(got[got > 0]), idx)


@pytest.mark.parametrize("sb", [8192, 8256])
def test_torch_port(sb: int):
    torch = pytest.importorskip("torch")
    nbs, nchunks, groups, t = 7, 3, 3, 2
    store = _store(nbs, nchunks, sb, seed=4)
    vs = SK.vectors(store, nbs, nchunks, sb, groups, t)
    vt = SK.vectors_torch(torch, torch.from_numpy(store), nbs, nchunks, sb, groups, t)
    assert np.array_equal(vt.numpy(), vs)
    bits = SK.sign_bits(SEED, 16, vs.shape[1])
    assert np.array_equal(SK.sketch_torch(torch, vt, bits).numpy(), SK.sketch(vs, bits))


def test_pins():
    bits = SK.sign_bits(SEED, 16, 4096)
    assert np.packbits(bits[:2, :64], axis=1, bitorder="little").tobytes().hex() == PIN_SIGNS
    sk = np.arange(-8 * 64, 8 * 64, dtype=np.int64).reshape(16, 64) * 4099
    tags = SK.tags_of(SEED, sk, 5)
    assert tags[0].hex() == PIN_TAG0 and tags[15].hex() == PIN_TAG15
    fast = SK.tags_fast(SEED, sk, 5)
    assert fast is None or fast == tags


def test_pass_seed():
    """Pass j's seed binds σ, j and every byte of root_(j−1), so each pass's signs are fresh (red team F29)."""
    root = bytes(range(64))
    s = SK.pass_seed(SEED, 3, root)
    assert s == hashlib.sha512(b"verity/remote-bandwidth/seed/v0\0" + SEED + (3).to_bytes(4, "big") + root).digest()[:32]
    assert s.hex() == PIN_PASS_SEED
    others = {SK.pass_seed(SEED, 4, root), SK.pass_seed(SEED, 3, root[:63] + b"\xff"), SK.pass_seed(s, 3, root)}
    assert len(others | {s}) == 4
    assert not np.array_equal(SK.sign_bits(s, 16, 4096), SK.sign_bits(SK.pass_seed(SEED, 4, root), 16, 4096))
    with pytest.raises(ValueError):
        SK.pass_seed(SEED, 1 << 32, root)


# Equal to the sketch tag's `sketch.sign_bits` and `sketch.tags_of` (benchmarks/sketch_tag at 1b21bc8f7).
PIN_SIGNS = "b8d365fd851a03a15a362aacdf5279ae"
PIN_TAG0 = "adeb6d8a23d278c83f228c489fe3cba9c65e1734f16509c3a763fe68417fe773"
PIN_TAG15 = "cc570da7530a446ef5c196da1da0ea452237cc4ad57921584745487a129a5b42"
# The first 32 bytes of the remote driver's `receipts.seed(SEED, 3, bytes(range(64)))`.
PIN_PASS_SEED = "2a07234e952d7eba243f5e881fbbfba3ac2597accaaf9bbffe3e9a2a4978d97f"
