"""Dither's C2-T filler form (`circuit.dither_c2t`): its shape rule beside the served unit's, the committed windows of each
segment's noised sum and their leaves, the Definition on bits against the reference, and each form's dispatch refusing the
other's shapes."""
from __future__ import annotations

import functools
import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.prims import BlackwellS8ImmaDot32  # noqa: F401  (registers the s8 steps' families)
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import dither_c2t as C

SEEDS = D.fill_seeds(hashlib.sha512(b"window salt").digest(), hashlib.sha512(b"call coin").digest())
KS = hashlib.sha512(b"prover's salt key").digest()
#: K = 128 at L = 64 (t = 2): two segments; 1 KiB rows
SH = C.C2TShape(128, 128, 128, lp=32, L=64, row=1024)


@functools.cache
def _call(sh: C.C2TShape = SH):
    A = D.pad_rows(SEEDS, sh.k, range(sh.m))
    B = np.random.default_rng(7).integers(-128, 128, size=(sh.n, sh.k)).tolist()
    V = C.values(sh, SEEDS, A, B)
    return A, B, V, C.commit(sh, C.values_buffer(sh, V), D.salts(KS, 1 << 33))


def _honest(sh: C.C2TShape, i: int, j0: int, s: int) -> list[list[int]]:
    A, B, V, _ = _call(sh)
    return C.block_inputs(sh, SEEDS, A[i], B[j0:j0 + 8], i, j0, s, V[s - 1][i][j0:j0 + 8])


def test_t_up_to_16_is_c2ts_rule_and_the_served_unit_keeps_8():
    """C2-T takes t = min(L, K) / L′ up to 16 at BYTE-CELL's K (8,192 and 16,384, one segment), and refuses 32; the served
    `dither.Shape` still refuses t = 16."""
    for k in (8192, 16384):
        sh = C.C2TShape(128, 128, k, lp=k // 16)
        assert (sh.t, sh.S, sh.seg(1)) == (16, 1, (0, k))
        with pytest.raises(ValueError, match="at most 8"):
            D.Shape(128, 128, k, lp=k // 16, L=k)
    assert C.C2TShape(128, 128, 16384, lp=512, L=8192).t == 16
    with pytest.raises(ValueError, match="at most 16"):
        C.C2TShape(128, 128, 16384, lp=512)


@pytest.mark.parametrize("r, w", [(9, 1), (8, 8), (2, 8), (9, 16), (0, 32)])
def test_a_window_covering_bit_9_is_admitted(r, w):
    sh = C.C2TShape(128, 128, 4096, lp=256, r=r, w=w)
    assert (sh.r, sh.w, sh.tile_bytes) == (r, w, 2048 * w)


@pytest.mark.parametrize("r, w", [(0, 8), (0, 9), (10, 1), (10, 8), (9, 24), (9, 0)])
def test_a_window_off_bit_9_or_past_32_bits_is_refused(r, w):
    """The low byte (C1's, which the orbit over Z/2^8 passes) and windows above bit 9 are outside SAKHigh-T's statement."""
    with pytest.raises(ValueError, match="covers bit 9"):
        C.C2TShape(128, 128, 4096, lp=256, r=r, w=w)


@pytest.mark.parametrize("bad", [dict(m=100), dict(lp=48), dict(L=96), dict(L=48), dict(row=1000), dict(k=D.K_LIMIT, lp=1024),
                                 dict(w=3, r=8, row=1024)])
def test_shape_refuses(bad):
    with pytest.raises(ValueError):
        C.C2TShape(**{**dict(m=128, n=128, k=128, lp=32, L=64, row=1024), **bad})


def test_window_is_floor_division_of_the_signed_sum():
    for x in (0, 511, 512, 1023, -1, -512, -513, -1024, 1 << 30, -(1 << 30)):
        assert C.window(x, 9, 1) == (x // 512) % 2 and C.window(x, 8, 8) == (x // 256) % 256


def test_windows_are_each_segments_own_noised_sum():
    """w_s(i, j) is the window of segment s's X W′ᵀ alone (not a prefix sum), X and W′ wrapped to int8 from the pad, B and
    the coins as `dither` forms them."""
    A, B, V, _ = _call()
    na, nb = np.array(D.noise_a(SH, SEEDS, range(SH.m))), np.array(D.noise_b(SH, SEEDS, range(SH.n)))
    X, W = D.wrap8(np.array(A) + na), D.wrap8(np.array(B) + nb)
    T = C.sums(SH, SEEDS, A, B)
    for s, (a, b) in enumerate(map(SH.seg, (1, 2))):
        full = X[:, a:b] @ W[:, a:b].T
        assert (np.array(T[s]) == full).all() and (np.array(V[s]) == (full >> 9) & 1).all()
    assert (np.array(T[0]) < 0).any() and (np.array(T[0]) >= 512).any()


def test_a_blocks_windows_are_one_rows_bytes_and_open():
    """At W = 1 a block's 8 windows are one byte, cells LSB first; at W = 8 eight bytes. Each opens from the value root under
    its salt, and a flipped byte or another salt doesn't."""
    _, _, V, cm = _call()
    assert (SH.tile_bytes, SH.leaves, len(cm.rows)) == (2048, 4, 4)
    rng = random.Random(1)
    for _ in range(8):
        i, j0, s = rng.randrange(SH.m), 8 * rng.randrange(SH.n // 8), rng.randrange(1, SH.S + 1)
        pos, off = SH.block(i, j0, s)
        row, salt, path = cm.open(pos)
        assert C.block_words(SH, row, off) == [V[s - 1][i][j] for j in range(j0, j0 + 8)]
        assert D.opens(cm.dom, cm.root, SH.leaves, pos, row, salt, path)
        assert not D.opens(cm.dom, cm.root, SH.leaves, pos, row[:off] + bytes([row[off] ^ 1]) + row[off + 1:], salt, path)
        assert not D.opens(cm.dom, cm.root, SH.leaves, pos, row, D.prf_salt(KS, pos), path)
    byte = C.C2TShape(128, 128, 128, lp=32, L=64, r=8, w=8, row=1024)
    A, B, *_ = _call()
    Vb = C.values(byte, SEEDS, A, B)
    buf = C.values_buffer(byte, Vb)
    pos, off = byte.block(77, 40, 2)
    assert (byte.tile_bytes, byte.leaves, off % 8) == (16384, 32, 0)
    assert C.block_words(byte, buf[pos * 1024:(pos + 1) * 1024], off) == Vb[1][77][40:48]
    assert byte.domain() != SH.domain()


def test_a_block_is_8_aligned_cells_of_one_row():
    with pytest.raises(ValueError, match="multiple of 8"):
        SH.block(5, 3, 1)
    with pytest.raises(ValueError, match="segment 3"):
        SH.block(5, 8, 3)


@pytest.mark.slow
@pytest.mark.parametrize("r, w", [(9, 1), (8, 8)])
def test_the_definition_on_bits_is_the_reference(r, w):
    """An honest block passes on bits; one bit of each port flipped, the Definition agrees with the reference in the clear,
    and a flipped committed window bit always fails."""
    sh = C.C2TShape(128, 128, 64, lp=32, r=r, w=w, row=1024)
    honest = _honest(sh, 77, 40, 1)
    d = C.block_definition(sh, 1)
    assert C.check_block(honest, r, w) == 1 and list(evaluate(d, *C.block_bits(honest, w))) == [1]
    rng = random.Random(3)
    for port, width in enumerate((8, 8, 8, 8, w)):
        bad = [list(v) for v in honest]
        bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(width)
        want = C.check_block(bad, r, w)
        assert list(evaluate(d, *C.block_bits(bad, w))) == [want]
        if port == 4:
            assert want == 0


def test_a_skipped_noise_tile_fails_almost_every_block():
    """A filler that skips segment 1's second L′ tile (its X W′ᵀ left out) commits windows that fail 12 of 12 sampled blocks of
    segment 1 (a block's free guess is about 2^-8), and segment 2's, computed honestly, pass."""
    A, B, V, _ = _call()
    skipped = [list(r) for r in A]
    na = D.noise_a(SH, SEEDS, range(SH.m))
    for i in range(SH.m):
        skipped[i][32:64] = [D.wrap8(-x) for x in na[i][32:64]]
    Vs = C.values(SH, SEEDS, skipped, B)
    rng = random.Random(5)
    fails = {1: 0, 2: 0}
    for _ in range(12):
        i, j0 = rng.randrange(SH.m), 8 * rng.randrange(SH.n // 8)
        for s in (1, 2):
            ins = C.block_inputs(SH, SEEDS, A[i], B[j0:j0 + 8], i, j0, s, Vs[s - 1][i][j0:j0 + 8])
            fails[s] += 1 - C.check_block(ins, SH.r, SH.w)
    assert fails == {1: 12, 2: 0}


def test_each_forms_dispatch_refuses_the_others_shape():
    """The verifier's dispatch: `dither.block_definition` takes only a served `Shape` and `dither_c2t.block_definition` only a
    `C2TShape`, which binds the form's 8 cells and window."""
    with pytest.raises(ValueError, match="C2-T filler call's are"):
        D.block_definition(SH, 1)
    with pytest.raises(ValueError, match="served call's are"):
        C.block_definition(D.Shape(128, 128, 128, lp=32, L=64, row=1024), 1)
    assert C.block_definition(SH, 2).bindings == {"L": 64, "CELLS": 8, "R": 9, "W": 1}
    assert D.block_definition(D.Shape(128, 128, 128, lp=32, L=64, row=1024), 1).bindings == {"L": 64, "FIRST": True}


@pytest.mark.parametrize("params", [dict(L=48, CELLS=8, R=9, W=1), dict(L=D.K_LIMIT, CELLS=8, R=9, W=1),
                                    dict(L=32, CELLS=0, R=9, W=1), dict(L=32, CELLS=8, R=30, W=3),
                                    dict(L=32, CELLS=8, R=9, W=True)])
def test_the_definition_refuses_bad_statics(params):
    with pytest.raises(ValueError):
        bind(C.PouwDitherFillC2T, **params).signature()


def test_each_word_primitive_is_its_definitions_word_view():
    for d in (C.PouwDitherBitEq, C.PouwDitherBitEqAnd):
        params, ret = d.signature({})
        assert ([BL.word_type(t) for _, t in params], BL.word_type(ret)) == ([t for _, t in d.word.params], d.word.ret)
        for v in range(1 << len(params)):
            bits = [[(v >> e) & 1] for e in range(len(params))]
            assert list(evaluate(d, *bits)) == [d.word.evaluate(*(b[0] for b in bits))]
