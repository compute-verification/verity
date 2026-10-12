"""ro-bound-sketch-blake3/v0's reference (`bound.py`): B's bytes where the spec places them, the unbound mode as
ro-sketch-blake3/v0 itself, the fused GEMM's MMA arrangement against the plain product, the tile share and the slice
snapshots from the spec's wording, the accumulators against FP32, the receipts and root, and pinned digests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import bound as BR  # noqa: E402

W, WS = BR.W, BR.WS
F = BR.FIELD
SEED = hashlib.sha256(b"verity/bound-receipt/vectors").digest()
N_ROWS, KDIM = 64, 1024  # one row group, two chunks: two tiles


def random_store(seed: int, n_rows: int = N_ROWS, kdim: int = KDIM) -> bytes:
    tasks = (n_rows // 16) * (kdim // W.TASK_COLS)
    return np.random.default_rng(seed).integers(0, 1 << 32, tasks * W.LANES * F.lane_words() * F.limbs,
                                                dtype=np.uint64).astype("<u4").tobytes()


def fp8(rng, shape) -> np.ndarray:
    """Finite E4M3 bytes (no NaN: 0x7F, 0xFF)."""
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    return np.where(b & 0x7F == 0x7F, b ^ 1, b).astype(np.uint8)


def spec_place(idx: int, b: int) -> tuple[int, int]:
    """(c, p) of byte b of u32 word idx of a fragment-order tile array, from the spec's wording."""
    rest, q = divmod(idx, 4)
    rest, lane = divmod(rest, 32)
    i, s = divmod(rest, 16)
    g, tq = divmod(lane, 4)
    c8, h = divmod(q, 2)
    return 8 * c8 + g, 32 * (16 * i + s) + 16 * h + 4 * tq + b


def test_shapes_and_the_kernel_constants():
    assert BR.TILE_WORDS == 8192 and WS.vector_bytes(F) == 2048 and WS.steps(F) == BR.STEPS
    assert [BR.batch_blocks(m) for m in (1, 8, 9, 16, 17, 32, 33, 64)] == [1, 1, 2, 2, 4, 4, 8, 8]
    with pytest.raises(ValueError):
        BR.batch_blocks(65)
    src = (HERE.parent / "bound.cu").read_text()
    assert f'BOUND_NAME[] = "{BR.NAME}"' in src
    assert "s4 = 4 * ((z4 + 4u * grid - 1) / (4u * grid));" in src and "(2048u * warp + 4u * lane) % s4, 128u % s4" in src
    assert "return Mb <= 8 ? 1 : Mb <= 16 ? 2 : Mb <= 32 ? 4 : 8;" in src


def test_from_fragments_inverts_sign_fragments():
    rng = np.random.default_rng(2)
    s = rng.integers(-128, 128, (16, 2048), dtype=np.int16).astype(np.int8)
    assert np.array_equal(BR.from_fragments(WS.sign_fragments(s, BR.STEPS)), s)


def test_b_bytes_where_the_spec_places_them():
    rng = np.random.default_rng(3)
    r = rng.integers(0, 1 << 32, (BR.T, 16, 32, 4), dtype=np.uint64).astype(np.uint32)
    masks = rng.integers(0, 2, BR.TILE_WORDS * 4, dtype=np.uint8) * 0xFF
    b = BR.b_matrix(r, masks).view(np.uint8)
    flat = r.reshape(-1)
    for idx in [0, 1, 3, 4, 127, 128, 2047, 2048, 8191, *rng.integers(0, 8192, 20)]:
        for byte in range(4):
            c, p = spec_place(int(idx), byte)
            assert b[c, p] == (int(flat[idx]) >> 8 * byte & 0xFF) ^ masks[4 * idx + byte], (idx, byte)


def test_masks_from_the_derivation():
    from verity.ml.randomness import derive

    m = BR.mask_bits(SEED)
    key = derive(SEED, WS.SIGN_DOMAIN, {"tag": BR.NAME, "k": 16, "n": 2048})
    for c in (0, 9, 15):
        row = key.stream(("row", c), 256)
        for p in (0, 5, 2047):
            assert m[c, p] == row[p // 8] >> (p % 8) & 1
    b = BR.from_fragments(BR.mask_fragments(SEED))
    assert np.array_equal(b, -m.astype(np.int8))
    assert not np.array_equal(m, WS.sign_bits(SEED, 16, 2048))  # its own derivation, not the unbound signs


def test_unbound_is_ro_sketch_blake3_v0():
    store = random_store(5)
    vs = WS.vectors(store, N_ROWS, KDIM, F)
    want = WS.sketch(vs, WS.signs(SEED, F))
    got = np.concatenate([BR.tile_sketch("unbound", vs[16 * u:16 * u + 16], BR.sign_fragments(SEED)) for u in range(2)])
    assert np.array_equal(got, want)
    assert BR.receipts_of(SEED, got, 7, name=WS.NAME) == WS.tags_of(SEED, want, 7)


@pytest.mark.parametrize("mode", ["tile", "slice32", "slice64", "slice128"])
def test_mma_arrangement_is_the_product(mode):
    """The kernel's fragments of B against its A fragments (`wide_sketch.sketch_from_fragments`, one mma.s8 at a time)
    give the plain product of the vectors and B."""
    rng = np.random.default_rng(11)
    store = random_store(6, N_ROWS, 512)
    acc = rng.integers(0, 1 << 32, (16, 64, 16), dtype=np.uint64).astype(np.uint32)
    zprev = rng.integers(0, 1 << 32, 3 * 64, dtype=np.uint64).astype(np.uint32)
    r = BR.r_words(mode, BR.z_share(zprev, 0, 1), acc)
    frag = np.ascontiguousarray(r, dtype="<u4").view(np.uint8).reshape(-1) ^ BR.mask_fragments(SEED)
    vs = WS.vectors(store, N_ROWS, 512, F)
    want = BR.sketch(vs, BR.b_matrix(r, BR.mask_fragments(SEED)))
    assert np.array_equal(WS.sketch_from_fragments(store, N_ROWS, 512, F, frag), want)
    assert np.abs(want).max() <= 128 * 128 * 2048


def test_tile_share():
    z = np.arange(1000, dtype=np.uint32)
    assert BR.share_words(1000, 3) == 336 and BR.share_words(1000, 1000) == 4
    s = BR.z_share(z, 2, 3)
    assert s.size == 336 and s[0] == 672 and s[327] == 999 and s[328] == 0
    r = BR.r_words("tile", BR.z_share(z, 1, 300))  # S4 = 4: every register group the same four words
    assert np.array_equal(r.reshape(-1, 4), np.broadcast_to(z[4:8], (2048, 4)))
    with pytest.raises(ValueError, match="32 KiB"):
        BR.share_words(8193 * 4, 4)
    assert BR.binding_ratio("tile", 1, 8192) == 4 / 8192


@pytest.mark.parametrize("j", [1, 2, 4, 8])
@pytest.mark.parametrize("w", [32, 64, 128])
def test_slice_snapshots_from_the_spec(w, j):
    rng = np.random.default_rng(100 * w + j)
    acc = rng.integers(0, 1 << 32, (16, 64, 8 * j), dtype=np.uint64).astype(np.uint32)
    z_u = rng.integers(0, 1 << 32, 64, dtype=np.uint64).astype(np.uint32)
    r = BR.r_words(f"slice{w}", z_u, acc)
    tile = BR.r_words("tile", z_u)
    sw = w // 32
    for i in range(4):
        for s in range(16):
            for lane in range(32):
                g, tq = divmod(lane, 4)
                for q in range(4):
                    if s < 4:
                        want = tile[i, s, lane, q]
                    else:
                        a, e = divmod(s, 4)
                        end = 4 * (a - 1) + sw * (e // sw) + sw - 1
                        assert end == (s - 4) * 32 // w * (w // 32) + w // 32 - 1  # the report's t(s)
                        e1 = e % sw

                        def acc_j(jj):
                            return int(acc[end, 16 * i + g + 8 * (q // 2), 8 * jj + 2 * tq + q % 2])
                        if j >= sw:
                            want = 0
                            for jj in range(e1, j, sw):
                                want ^= acc_j(jj)
                        else:
                            want = acc_j(e1 % j)
                    assert r[i, s, lane, q] == want, (i, s, lane, q)


def test_zero_z_is_a_subset_sketch():
    b = BR.b_matrix(BR.r_words("tile", np.zeros(4, np.uint32)), BR.mask_fragments(SEED))
    assert set(np.unique(b)) == {-1, 0}


def test_accumulators_are_the_fp32_product():
    """The model's chain after the last k32 step is the chunk's FP32 product up to the adder's truncation, and x's
    zero rows give +0."""
    rng = np.random.default_rng(4)
    w = fp8(rng, (64, 512))
    x = np.zeros((8, 512), np.uint8)
    x[:3] = fp8(rng, (3, 512))
    acc = BR.accumulators(w, x)
    assert acc.shape == (16, 64, 8) and not acc[:, :, 3:].any()

    def e4m3(b: np.ndarray) -> np.ndarray:
        e, mant = (b >> 3 & 15).astype(np.int64), (b & 7).astype(np.float64)
        mag = np.where(e == 0, mant * 2.0 ** -9, (8 + mant) * 2.0 ** (e - 10))
        return np.where(b & 0x80, -mag, mag)
    vw, vx = e4m3(w), e4m3(x)
    got = acc[-1].view(np.float32).astype(np.float64)
    ref = vw @ vx.T
    scale = np.abs(vw) @ np.abs(vx).T + 1e-30
    assert (np.abs(got - ref) / scale).max() < 1e-5


def test_receipts_leaves_root():
    from verity.core.protocols.bandwidth import tags as TG
    from verity.ml.crypto import blake3

    sk = np.random.default_rng(8).integers(-(1 << 25), 1 << 25, (32, 16))
    ys = BR.receipts_of(SEED, sk, jbase=1000)
    key = hashlib.sha512(TG.prefix(TG.TAG_DOMAIN, BR.NAME, SEED)).digest()[:32]
    assert ys[5] == blake3.keyed(key, (1005).to_bytes(8, "big") + sk[5].astype("<i4").tobytes())
    lv = BR.leaves(SEED, ys)
    assert len(lv) == 2 and BR.root(SEED, ys) == hashlib.sha512(b"\x01" + lv[0] + lv[1]).digest()
    with pytest.raises(ValueError, match="int32"):
        BR.receipts_of(SEED, np.array([[1 << 31] + [0] * 15]))


def pinned_inputs(m: int):
    rng = np.random.default_rng(1000 + m)
    j = BR.batch_blocks(m)
    store = random_store(9 + m, N_ROWS, 512)
    w = fp8(rng, (64, 512))
    x = np.zeros((8 * j, 512), np.uint8)
    x[:m] = fp8(rng, (m, 512))
    zprev = rng.standard_normal((m, 64)).astype(np.float32)
    return store, w, x, zprev


# sha256 over the 16 receipts of a one-tile matrix, each mode, M = 1 and 9: changing any rule changes these
PINNED = {
    (1, "unbound"): "eb3110f09b99182707538d8108f344995a2bb91156310bfd8be5e5150e8742d1",
    (1, "tile"): "0afeee9109bd221a3382ae25bd42fca1a54bc909c0b3297d728eba270ae63df0",
    (1, "slice32"): "45b72ac547b10050d5a70a7e5fe7a491fed99212abcfc8e3b55de837a1b619b7",
    (1, "slice64"): "45c9646d0deb0b96eba025350be5a53484f9b98a30bf912092cc307a6bd24600",
    (1, "slice128"): "6909c400af977cabb425a348d08f2b59a110ca90634b62bd252be6bd5e997952",
    (9, "unbound"): "6979f729f2a0549531f27b43728cd774363bf5787b54be4e90dbd54bd7f5d5d6",
    (9, "tile"): "4c4f1e28e053cbe63445356f5b7f988171eb04c6898c4eb8f1501ff17a026e1b",
    (9, "slice32"): "31bbb70f8979f46721b6fad31afe27106323d53759e3ea4ba8353e8b76d43a3e",
    (9, "slice64"): "fbef60834d71ea945865545ac5c1df1432a04e93238dbb77e5aa65de82e1921d",
    (9, "slice128"): "08781b7c75b184e6ce2df1e6c6831a401b965249222910b9995e1a7623f4fd84",
}


@pytest.mark.parametrize("m", [1, 9])
def test_pinned_receipts(m):
    store, w, x, zprev = pinned_inputs(m)
    vs = WS.vectors(store, N_ROWS, 512, F)
    acc = BR.accumulators(w, x)
    for mode in BR.MODES:
        masks = BR.sign_fragments(SEED) if mode == "unbound" else BR.mask_fragments(SEED)
        sk = BR.tile_sketch(mode, vs, masks, zprev, 0, 1, acc)
        ys = BR.receipts_of(SEED, sk, 0, name=WS.NAME if mode == "unbound" else BR.NAME)
        assert hashlib.sha256(b"".join(ys)).hexdigest() == PINNED[(m, mode)], (m, mode)
