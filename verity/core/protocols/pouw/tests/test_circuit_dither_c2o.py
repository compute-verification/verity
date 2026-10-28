"""Dither's C2-O served form (`circuit.dither_c2o`): its shape rule, the committed windows of un-noised O and their leaves,
the Definition on bits against the reference, and each form's dispatch refusing the others' shapes."""
from __future__ import annotations

import functools
import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.prims import BlackwellS8ImmaDot32  # noqa: F401  (registers the s8 steps' families)
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import dither_c2o as O
from verity.core.protocols.pouw.circuit import dither_c2t as T

SEEDS = D.seeds(hashlib.sha512(b"call coin").digest(), hashlib.sha512(b"operand-root receipt").digest(), 0)
KS = hashlib.sha512(b"prover's salt key").digest()
SH = O.C2OShape(128, 128, 256, row=1024)


@functools.cache
def _call(sh: O.C2OShape = SH):
    rng = np.random.default_rng(7)
    A, B = rng.integers(-128, 128, size=(sh.m, sh.k)).tolist(), rng.integers(-128, 128, size=(sh.n, sh.k)).tolist()
    V = O.values(sh, SEEDS, A, B)
    return A, B, V, T.commit(sh, T.values_buffer(sh, V), D.salts(KS, 1 << 33))


def test_o_is_the_dithered_product_and_its_window_bit_9():
    """O = (A + d_A)(B + d_B)ᵀ exactly, with no noise, and the committed window is ⌊O/512⌋ mod 2."""
    A, B, V, _ = _call()
    da, db = np.array(D.dithers(SH, SEEDS, "A", range(SH.m))), np.array(D.dithers(SH, SEEDS, "B", range(SH.n)))
    full = (np.array(A) + da) @ (np.array(B) + db).T
    assert (np.array(O.outputs(SH, SEEDS, A, B)) == full).all()
    assert (np.array(V[0]) == (full >> 9) & 1).all() and (full < 0).any()


@pytest.mark.parametrize("r, w", [(0, 8), (10, 1), (9, 24), (9, 0)])
def test_a_window_off_bit_9_or_past_32_bits_is_refused(r, w):
    with pytest.raises(ValueError, match="covers bit 9"):
        O.C2OShape(128, 128, 4096, r=r, w=w)


def test_unpinned_weights_are_refused():
    with pytest.raises(ValueError, match="needs B pinned"):
        O.C2OShape(128, 128, 4096, pinned=False)


@pytest.mark.parametrize("bad", [dict(m=100), dict(k=48), dict(k=D.K_LIMIT), dict(row=1000), dict(w=3, r=8)])
def test_shape_refuses(bad):
    with pytest.raises(ValueError):
        O.C2OShape(**{**dict(m=128, n=128, k=256, row=1024), **bad})


def test_a_blocks_windows_are_one_rows_byte_and_open():
    _, _, V, cm = _call()
    assert (SH.tile_bytes, SH.leaves, len(cm.rows)) == (2048, 2, 2)
    rng = random.Random(1)
    for _ in range(8):
        i, j0 = rng.randrange(SH.m), 8 * rng.randrange(SH.n // 8)
        pos, off = SH.block(i, j0)
        row, salt, path = cm.open(pos)
        assert T.block_words(SH, row, off) == V[0][i][j0:j0 + 8]
        assert D.opens(cm.dom, cm.root, SH.leaves, pos, row, salt, path)
    with pytest.raises(ValueError, match="multiple of 8"):
        SH.block(5, 3)
    assert SH.domain() != T.C2TShape(128, 128, 256, lp=32, row=1024).domain()


@pytest.mark.slow
def test_the_definition_on_bits_is_the_reference():
    """An honest block passes on bits; one bit of each port flipped, the Definition agrees with the reference in the clear,
    and a flipped committed window bit always fails."""
    sh = O.C2OShape(128, 128, 32, row=1024)
    A, B, V, _ = _call(sh)
    honest = O.block_inputs(sh, SEEDS, A[77], B[40:48], 77, 40, V[0][77][40:48])
    d = O.block_definition(sh)
    assert O.check_block(honest, sh.r, sh.w) == 1 and list(evaluate(d, *O.block_bits(honest, sh.w))) == [1]
    rng = random.Random(3)
    for port, width in enumerate((8, 1, 8, 1, 1)):
        bad = [list(v) for v in honest]
        bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(width)
        want = O.check_block(bad, sh.r, sh.w)
        assert list(evaluate(d, *O.block_bits(bad, sh.w))) == [want]
        if port == 4:
            assert want == 0


def test_a_dropped_dither_fails_almost_every_block():
    """A prover that serves A Bᵀ, the product without the dithers, commits windows that fail 12 of 12 sampled blocks (a
    block's free guess is about 2^-8)."""
    A, B, *_ = _call()
    plain = (np.array(A) @ np.array(B).T >> 9) & 1
    rng = random.Random(5)
    fails = 0
    for _ in range(12):
        i, j0 = rng.randrange(SH.m), 8 * rng.randrange(SH.n // 8)
        ins = O.block_inputs(SH, SEEDS, A[i], B[j0:j0 + 8], i, j0, plain[i, j0:j0 + 8].tolist())
        fails += 1 - O.check_block(ins, SH.r, SH.w)
    assert fails == 12


def test_each_forms_dispatch_refuses_the_others_shapes():
    c2t, served = T.C2TShape(128, 128, 256, lp=32, row=1024), D.Shape(128, 128, 256, lp=32, L=256, row=1024)
    for other in (c2t, served):
        with pytest.raises(ValueError, match="C2-O's blocks are a C2OShape's"):
            O.block_definition(other)
    with pytest.raises(ValueError, match="C2-O call's"):
        D.block_definition(SH, 1)
    with pytest.raises(ValueError, match="for C2-O"):
        T.block_definition(SH, 1)
    assert O.block_definition(SH).bindings == {"K": 256, "CELLS": 8, "R": 9, "W": 1}


@pytest.mark.parametrize("params", [dict(K=48, CELLS=8, R=9, W=1), dict(K=D.K_LIMIT, CELLS=8, R=9, W=1),
                                    dict(K=32, CELLS=0, R=9, W=1), dict(K=32, CELLS=8, R=30, W=3)])
def test_the_definition_refuses_bad_statics(params):
    with pytest.raises(ValueError):
        bind(O.PouwDitherServedC2O, **params).signature()
