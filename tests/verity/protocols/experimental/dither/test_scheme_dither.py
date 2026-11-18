"""Dither as a named PoUW scheme (`dither.prover.scheme`): found by its name once the package is imported, its methods on
an honest tile at each named (L, L′) pair against the reference and the proof blocks' checks in the clear, a tampered
prefix sum refused, and the methods it has no code for refused with what is missing."""
from __future__ import annotations

import hashlib
from array import array

import numpy as np
import pytest

import verity.protocols.experimental.dither  # noqa: F401  (registers dither-v3.2)
from verity.protocols.experimental.dither.prover import dither as D
from verity.protocols.experimental.dither.prover import scheme as DS
from verity.protocols.resource_accounting.compute.pouw.prover import schemes
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import (
    Context, DomainError, Matmul, PoUWScheme, Shape, Tile)

SEEDS = D.seeds(hashlib.sha512(b"call coin").digest(), hashlib.sha512(b"operand-root receipt").digest(), 0)
#: each named pair at a K of three segments: a first, a middle and a last proof block
K = {"rtx-pro-6000-pinned": 3072, "rtx-pro-6000-chosen": 2304}
TILE = Tile((0, 77, 127), (5, 126))


def _ctx(k: int) -> Context:
    return Context(bytes(32), Matmul(0, Shape(128, k, 128), 0), bytes(64), bytes(64))


def _rows(n: int, k: int, seed: int) -> list[list[int]]:
    return np.random.default_rng(seed).integers(-128, 128, size=(n, k)).tolist()


def _tile(s: DS.Dither, k: int):
    ctx = _ctx(k)
    A, B = _rows(len(TILE.rows), k, 1), _rows(len(TILE.cols), k, 2)
    ops = s.encode(ctx, SEEDS, [array("b", r).tobytes() for r in A], [array("b", r).tobytes() for r in B], TILE)
    return ctx, A, B, ops, s.checked(ctx, ops)


def test_the_scheme_is_found_by_its_name_at_each_named_pair():
    s = schemes.get(DS.NAME)
    assert isinstance(s, PoUWScheme) and isinstance(s, DS.Dither) and s.name == "dither-v3.2"
    assert (s.L, s.lp) == (1024, 128) and dict(s.certificate.parameters) == {"L": 1024, "lp": 128}
    chosen = schemes.get(DS.NAME, parameters="rtx-pro-6000-chosen")
    assert (chosen.L, chosen.lp) == (768, 256)
    assert DS.NAME in schemes.EXPERIMENTAL and schemes.SCHEMES[DS.NAME] is not None
    assert s.certificate.gap is None and s.certificate.eps is None
    with pytest.raises(ValueError, match="no parameters 'h200'"):
        schemes.get(DS.NAME, parameters="h200")


@pytest.mark.parametrize("parameters", sorted(K))
def test_check_takes_the_references_domain(parameters):
    s = DS.Dither(parameters)
    s.check(Matmul(0, Shape(256, K[parameters], 384), 0))
    for bad in (Shape(100, K[parameters], 128), Shape(128, K[parameters] + 32, 128), Shape(128, 1 << 17, 128)):
        with pytest.raises(DomainError, match="dither-v3.2 at L = "):
            s.check(Matmul(0, bad, 0))


def test_tiles_are_the_value_roots_tiles():
    shape = Shape(256, 3072, 384)
    sh = DS.Dither().shape(shape)
    tiles = DS.Dither().tiles(shape)
    assert len(tiles) == sh.tiles == 6 and all(len(t.rows) == len(t.cols) == D.TILE for t in tiles)
    assert [sh.tile_of(t.rows[0], t.cols[0]) for t in tiles] == [(n, 0) for n in range(6)]


@pytest.mark.parametrize("parameters", sorted(K))
def test_an_honest_tile_passes_every_proof_block(parameters):
    s, k = DS.Dither(parameters), K[parameters]
    ctx, A, B, ops, checked = _tile(s, k)
    sh = s.shape(ctx.matmul.shape)
    assert sh.S == 3 and s.admissible(ctx, ops)
    da, db = D.dithers(sh, SEEDS, "A", TILE.rows), D.dithers(sh, SEEDS, "B", TILE.cols)
    for x, i in enumerate(TILE.rows):
        for y, j in enumerate(TILE.cols):
            T, O = checked[x][y]
            assert (T, O) == D.cell_values(sh, SEEDS, A[x], B[y], i, j)
            exact = sum((p + q) * (u + v) for p, q, u, v in zip(A[x], da[x], B[y], db[y]))
            assert s.useful(ctx, SEEDS, ops, checked)[x][y] == exact
            for seg in range(1, sh.S):
                ins = D.block_inputs(sh, SEEDS, A[x], B[y], i, j, seg, t0=T[seg - 2] if seg > 1 else 0, t1=T[seg - 1])
                assert D.check_block(ins) == 1
            last = D.block_inputs(sh, SEEDS, A[x], B[y], i, j, sh.S, t0=T[sh.S - 2], o=O)
            assert D.check_last(sh, SEEDS, i, j, last) == 1


@pytest.mark.parametrize("parameters", sorted(K))
def test_a_tampered_prefix_sum_is_refused(parameters):
    s, k = DS.Dither(parameters), K[parameters]
    ctx, A, B, _, checked = _tile(s, k)
    sh = s.shape(ctx.matmul.shape)
    x, y, i, j = 1, 0, TILE.rows[1], TILE.cols[0]
    T, O = checked[x][y]
    first = D.block_inputs(sh, SEEDS, A[x], B[y], i, j, 1, t1=T[0] + 1)
    middle = D.block_inputs(sh, SEEDS, A[x], B[y], i, j, 2, t0=T[0], t1=T[1] ^ 0x8000)
    assert D.check_block(first) == 0 and D.check_block(middle) == 0
    for t0, o in ((T[1] + 1, O), (T[1], O + 1)):
        assert D.check_last(sh, SEEDS, i, j, D.block_inputs(sh, SEEDS, A[x], B[y], i, j, 3, t0=t0, o=o)) == 0


def test_a_row_of_the_wrong_length_is_not_an_operand():
    s = DS.Dither()
    ctx = _ctx(3072)
    row = bytes(3072)
    assert s.encode(ctx, SEEDS, [row, row[:-1], row], [row, row], TILE) is None


@pytest.mark.parametrize(("method", "missing"), [
    ("noise", "a Context carries neither"),
    ("wref", "Dither has no W1 cost"),
    ("digest", "there is no transcript leaf per tile"),
])
def test_what_dither_has_no_code_for_is_refused(method, missing):
    s, ctx = DS.Dither(), _ctx(3072)
    args = {"noise": (ctx,), "wref": (ctx.matmul.shape,), "digest": (ctx, SEEDS, [])}[method]
    with pytest.raises(NotImplementedError, match=missing):
        getattr(s, method)(*args)
