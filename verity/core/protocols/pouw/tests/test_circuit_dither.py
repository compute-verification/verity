"""Dither's no-fold proof block for the filler (`circuit.dither`): the per-call pad, the 16-bit words, the leaves a kernel's
sampled cells are checked against, and `PouwDitherBlock_v1` on bits against the reference."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.core.protocols.pouw.circuit import dither as D

SALT = hashlib.sha512(b"window salt").digest()
COIN = hashlib.sha512(b"call coin").digest()
#: K = 192 at L = 128: two segments, the last of 64
SH = D.Shape(128, 256, 192, 32, 128)


def _weights(sh: D.Shape, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=(sh.n, sh.k))


def test_pad_is_fixed_by_the_coin_and_changes_with_it():
    rows = range(8)
    pad = np.array(D.pad_rows(D.call_seeds(SALT, COIN), 4096, rows))
    assert D.pad_rows(D.call_seeds(SALT, COIN), 4096, rows) == pad.tolist()
    other_coin = hashlib.sha512(b"another call's coin").digest()
    other_salt = hashlib.sha512(b"another window's salt").digest()
    for seeds in (D.call_seeds(SALT, other_coin), D.call_seeds(other_salt, COIN)):
        moved = np.array(D.pad_rows(seeds, 4096, rows))
        assert (moved != pad).mean() > 0.98
        assert all(seeds[x] != D.call_seeds(SALT, COIN)[x] for x in D.STREAMS)
    assert pad.min() == -128 and pad.max() == 127


def test_pad_and_seeds_pinned():
    """The derivation a kernel reproduces: seed_x = SHA-512(SEEDS_TAG ‖ salt ‖ coin ‖ x), then the pad stream's blocks."""
    seeds = D.call_seeds(SALT, COIN)
    assert seeds["pad"] == hashlib.sha512(D.SEEDS_TAG + SALT + COIN + b"pad\0\0\0\0\0").digest()
    k, i = 4096, 3
    blocks = b"".join(hashlib.sha512(seeds["pad"] + b"pad\0\0\0\0\0" + (i * 64 + b).to_bytes(8, "big")).digest()
                      for b in range(64))
    assert D.pad_rows(seeds, k, [i])[0] == np.frombuffer(blocks, dtype=np.int8).tolist()
    assert hashlib.sha512(b"".join(seeds[x] for x in D.STREAMS)).hexdigest()[:32] == "db8610d22ea9dc75058e0b5122c54691"
    pad = np.array(D.pad_rows(seeds, k, range(4)), dtype=np.int8).tobytes()
    assert hashlib.sha512(pad).hexdigest()[:32] == "c9259c0840d4afcace2b430c740baa8d"


def test_seeds_refuse_short_digests():
    with pytest.raises(ValueError, match="64-byte"):
        D.call_seeds(SALT[:32], COIN)


def test_noise_is_dithers_plus_hyb():
    seeds = D.call_seeds(SALT, COIN)
    na, nb = np.array(D.noise_a(SH, seeds, range(4))), np.array(D.noise_b(SH, seeds, range(4)))
    for n in (na, nb):
        assert n.min() >= -D.NOISE and n.max() <= D.NOISE + 1
    pi = D.perms(seeds, SH.k, SH.lp)
    assert all(sorted(p) == list(range(SH.lp)) for p in pi)
    lp = SH.lp
    assert set(np.unique(na[:, :lp] - na[:, lp:2 * lp]).tolist()) <= {-1, 0, 1}
    assert not set(np.unique(nb[:, :lp] - nb[:, lp:2 * lp]).tolist()) <= {-1, 0, 1}


def test_words_are_16_bit_prefix_sums():
    seeds, B = D.call_seeds(SALT, COIN), _weights(SH)
    T = np.array(D.words(SH, seeds, B))
    assert T.shape == (SH.S, SH.m, SH.n) and SH.S == 2 and 0 <= T.min() and T.max() < 1 << 16
    A = np.array(D.pad_rows(seeds, SH.k, range(SH.m)))
    X = D.wrap8(A + np.array(D.noise_a(SH, seeds, range(SH.m))))
    W = D.wrap8(B + np.array(D.noise_b(SH, seeds, range(SH.n))))
    rng = random.Random(1)
    for _ in range(20):
        i, j = rng.randrange(SH.m), rng.randrange(SH.n)
        full = [int(np.dot(X[i, :b], W[j, :b])) for b in (128, 192)]
        assert [int(T[s, i, j]) for s in range(2)] == [v % (1 << 16) for v in full]
        assert D.cell_words(SH, seeds, B[j], i, j) == [v % (1 << 16) for v in full]


def test_leaves_from_sampled_cells_and_openings():
    seeds, B = D.call_seeds(SALT, COIN), _weights(SH)
    T = D.words(SH, seeds, B)
    buf = D.words_buffer(SH, T)
    assert len(buf) == SH.leaves * SH.row == SH.tiles * SH.S * D.TILE_WORD_BYTES
    ks = hashlib.sha512(b"prover's salt key").digest()
    cm = D.commit(SH, T, lambda p: D.prf_salt(ks, p))
    rng = random.Random(2)
    for _ in range(6):
        i, j, s = rng.randrange(SH.m), rng.randrange(SH.n), rng.randrange(1, SH.S + 1)
        pos, off = SH.word_leaf(i, j, s)
        row = D.row_bytes(SH, buf, pos)
        assert D.leaf_row(SH, seeds, B, pos) == row
        assert int.from_bytes(row[off:off + 2], "little") == T[s - 1][i][j]
        assert (s, (i, j)) == (SH.leaf_cells(pos)[0], SH.leaf_cells(pos)[1][off // 2])
        assert D.opens(SH, cm.root, pos, row, cm.salt(pos), cm.path(pos))
        forged = bytes([row[0] ^ 1]) + row[1:]
        assert not D.opens(SH, cm.root, pos, forged, cm.salt(pos), cm.path(pos))
        assert not D.opens(SH, cm.root, pos, row, D.prf_salt(ks, pos + 1), cm.path(pos))


def _block(L: int, first: bool):
    return bind(D.PouwDitherBlock, L=L, FIRST=first)


def _opened(sh: D.Shape, cm: D.Committed, i: int, j: int, s: int) -> dict[int, bytes]:
    pos = [sh.word_leaf(i, j, x)[0] for x in range(max(1, s - 1), s + 1)]
    for p in pos:
        assert D.opens(sh, cm.root, p, D.row_bytes(sh, cm.buf, p), cm.salt(p), cm.path(p))
    return {p: D.row_bytes(sh, cm.buf, p) for p in pos}


def test_definition_passes_an_honest_block_on_bits():
    """Every segment of sampled cells, opened from the tree, through `PouwDitherBlock_v1` on bits; the last segment is
    shorter than L."""
    seeds, B = D.call_seeds(SALT, COIN), _weights(SH)
    cm = D.commit(SH, D.words(SH, seeds, B), lambda p: D.prf_salt(b"k" * 64, p))
    for i, j in ((0, 0), (77, 200), (127, 255)):
        for s in range(1, SH.S + 1):
            rows = _opened(SH, cm, i, j, s)
            ins = D.block_inputs(SH, seeds, B[j], i, j, s, D.word_at(SH, rows, i, j, s - 1), D.word_at(SH, rows, i, j, s))
            lo, hi = SH.seg(s)
            assert D.check_block(ins) == 1
            assert list(evaluate(_block(hi - lo, s == 1), *D.block_bits(ins))) == [1]


def test_definition_refuses_what_the_reference_refuses():
    """One bit of each port flipped, at both segments of a K = 96, L = 64 call (the last of 32)."""
    sh = D.Shape(128, 128, 96, 32, 64)
    seeds, B = D.call_seeds(SALT, COIN), _weights(sh)
    i, j = 5, 9
    T = D.cell_words(sh, seeds, B[j], i, j)
    rng = random.Random(3)
    for s in (1, 2):
        lo, hi = sh.seg(s)
        d = _block(hi - lo, s == 1)
        honest = D.block_inputs(sh, seeds, B[j], i, j, s, T[0] if s == 2 else 0, T[s - 1])
        for port in range(len(honest)):
            bad = [list(v) for v in honest]
            bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(8 if port < 4 else 16)
            want = D.check_block(bad)
            assert list(evaluate(d, *D.block_bits(bad))) == [want]
            if port >= 4:
                assert want == 0


def test_a_skipped_noise_tile_is_refused():
    """A prover that skips one L′ tile of segment 1 commits words the honest block can't pass: almost every cell's T_1 is
    wrong, and T_2 - T_1 is still right, so only segment 1's blocks refuse."""
    seeds, B = D.call_seeds(SALT, COIN), _weights(SH)
    A, na = np.array(D.pad_rows(seeds, SH.k, range(SH.m))), np.array(D.noise_a(SH, seeds, range(SH.m)))
    skipped = A.copy()
    skipped[:, 32:64] = -na[:, 32:64]
    assert not D.wrap8(skipped + na)[:, 32:64].any() and D.wrap8(A + na)[:, 32:64].any()
    T = np.array(D.words(SH, seeds, B, A=skipped))
    rng = random.Random(4)
    fails = {1: 0, 2: 0}
    for _ in range(16):
        i, j = rng.randrange(SH.m), rng.randrange(SH.n)
        for s in (1, 2):
            ins = D.block_inputs(SH, seeds, B[j], i, j, s, int(T[0, i, j]) if s == 2 else 0, int(T[s - 1, i, j]))
            fails[s] += 1 - D.check_block(ins)
    assert fails[1] >= 15 and fails[2] == 0


@pytest.mark.parametrize("bad", [dict(m=100), dict(lp=48), dict(L=48), dict(row=1000)])
def test_shape_refuses(bad):
    with pytest.raises(ValueError):
        D.Shape(**{**dict(m=128, n=128, k=256, lp=32, L=64), **bad})


def test_block_refuses_a_length_off_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        evaluate(bind(D.PouwDitherBlock, L=48, FIRST=True), *([[0] * 384] * 4), [0] * 16)
