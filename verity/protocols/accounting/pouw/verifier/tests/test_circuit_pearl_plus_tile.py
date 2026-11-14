"""Pearl+'s -h3 tile unit (`verity.protocols.accounting.pouw.prover.circuit.pearl_plus_tile`): its core on words equals the
scheme (`reference`), its σ equals its words, the tile leaf opens as `leaf_reference`, and the whole unit's ``ok`` is the
core's ``ok`` and the leaf's opening, on honest fixture tiles and tampered ones."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import evaluate_words
from verity.commitments.gates import frame_v3 as F3
from verity.commitments.gates.hm96 import reference as hm96_reference
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SCHEMA
from verity.protocols.accounting.pouw.prover.circuit import cap, pc8, pearl_plus_tile as T
from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.pouw.prover.schemes import pearl_c_u as CU
from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import DEVICES


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    out: list[int] = []
    for v in x:
        out += _flat(v)
    return out


def _words1(fn, *args) -> list[int]:
    """A Boolean Definition ``fn`` on one instance, each word argument as the σ word view (`evaluate_words`): its output
    words in leaf order."""
    out = evaluate_words(fn, *(np.asarray([_flat(a)], dtype=np.uint64) for a in args))
    return out.reshape(-1).tolist()


def _f32(rng: random.Random, n: int) -> list[int]:
    return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) for _ in range(n)]


class Tile:
    """An honest fixture tile built from the scheme (`pearl_c`): the circuit inputs, the reference's (digest, ok), and a leaf
    that opens the digest to a transcript root."""

    def __init__(self, dev: str, k: int, tm: int, tn: int, n: int, col: int, depth: int, seed: str):
        rng = random.Random(seed)
        d = DEVICES[dev]
        self.dev, self.k, self.tm, self.tn, self.n, self.col, self.depth = dev, k, tm, tn, n, col, depth
        self.a = [_f32(rng, k) for _ in range(tm)]
        self.b = [_f32(rng, k) for _ in range(tn)]
        self.rows, self.cols, self.midx = list(range(tm)), list(range(tn)), 7
        self.salt = bytes(rng.randrange(256) for _ in range(32))
        self.root = bytes(rng.randrange(256) for _ in range(64))
        scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=d, hashing="h3")
        self.scheme = scheme
        self.rho, self.rhor = cap._rho(dev, None), cap.row_rho(scheme, k)
        ctx = C.Context(self.salt, C.Matmul(self.midx, C.Shape(tm, k, tn), 0), bytes(64), self.root)
        ops = scheme.encode(ctx, scheme.noise(ctx), [C.f32_row(x) for x in self.a], [C.f32_row(x) for x in self.b],
                            C.Tile(tuple(self.rows), tuple(self.cols)))
        assert ops is not None, "the fixture tile must encode"
        checked = scheme.checked(ctx, ops)
        seed_b = scheme.noise(ctx).seed_b
        shape = pc8.Shape(tm, k, tn, tile=(tm, tn), dev=dev)
        self.eb = [list(ops.e_b[j]) for j in range(tn)]
        self.fa = [pc8.basis_fa(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)]
        self.fb = [pc8.basis_fb(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)]
        self.g = [pc8.gram_g(shape, seed_b)[p * 32:p * 32 + 32] for p in range(32)]
        _d, ng = cap._promoted(dev, k)
        self.cert = [[[list(cap.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
                      for j in range(tn)] for i in range(tm)]
        self.y = [[0] * n for _ in range(tm)]
        for i in range(tm):
            for j in range(tn):
                self.y[i][col * tn + j] = CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16
        self.digest, self.ok = T.reference(dev, k, n, self.a, self.b, self.y, self.cert, self.salt, self.root, self.midx,
                                           self.rows, self.cols, col, self.rho, self.rhor)
        # a transcript leaf that opens the honest digest
        self.hsalt = bytes(rng.randrange(256) for _ in range(192))
        self.tdomain = bytes(rng.randrange(256) for _ in range(64))
        self.tpos = rng.randrange(1 << depth)
        self.path = [bytes(rng.randrange(256) for _ in range(64)) for _ in range(depth)]
        payload = hm96_reference(self.digest, self.hsalt)[2]
        self.troot = F3.root_ref(self.tdomain, SCHEMA, payload, self.tpos, self.path)

    def core(self):
        return bind(T.PearlPlusTileCore, K=self.k, TM=self.tm, TN=self.tn, N=self.n, RHO=self.rho, RHOR=self.rhor,
                   DEV=self.dev)

    def core_v2(self):
        return bind(T.PearlPlusTileCoreV2, K=self.k, TM=self.tm, TN=self.tn, N=self.n, RHO=self.rho, RHOR=self.rhor,
                   DEV=self.dev)

    def core_args(self) -> list:
        return [self.a, self.b, self.y, self.cert, self.salt, self.root, self.midx, self.rows, self.col, self.eb, self.fa,
                self.fb, self.g]

    def tile_def(self):
        return T.tile(self.scheme, k=self.k, tm=self.tm, tn=self.tn, n=self.n, depth=self.depth)

    def tile_args(self) -> list:
        return [self.a, self.b, self.y, self.cert, self.hsalt, self.path, self.salt, self.root, self.midx, self.rows,
                self.col, self.eb, self.fa, self.fb, self.g, self.troot, self.tdomain, self.tpos]


SHAPES = [("h100", 256, 1, 1, 1, 0, 1), ("h100", 256, 2, 2, 4, 0, 2)]


@pytest.mark.parametrize("dev,k,tm,tn,n,col,depth", SHAPES)
def test_core_on_words_is_the_scheme(dev, k, tm, tn, n, col, depth):
    """The core on words is `reference`: the digest and ok = 1 on an honest tile."""
    t = Tile(dev, k, tm, tn, n, col, depth, f"core/{dev}/{k}/{tm}x{tn}")
    out = list(evaluate(t.core(), *(_flat(a) for a in t.core_args())))
    assert t.ok == 1
    assert bytes(out[:64]) == t.digest and out[64] == 1


@pytest.mark.slow
def test_core_sigma_is_its_words():
    """The σ core equals the word core on an honest 1 × 1 tile: same digest and ok.  Slow: lowering the core to bits is the
    only check that its σ (what the unit Calls) computes what its words do."""
    t = Tile("h100", 256, 1, 1, 1, 0, 1, "sigma")
    word = list(evaluate(t.core(), *(_flat(a) for a in t.core_args())))
    assert _words1(t.core_v2(), *t.core_args()) == word


@pytest.mark.slow
def test_leaf_opens_as_its_reference():
    """`PearlPlusTileLeaf_v1` is `leaf_reference`: 1 when the digest's hm96 leaf opens at tpos to troot, 0 under a wrong
    root.  Slow: the leaf is two SHA-512 compressions and the frame reads, lowered to bits."""
    t = Tile("h100", 256, 1, 1, 1, 0, 2, "leaf")
    leaf = bind(T.PearlPlusTileLeaf, D=t.depth)
    assert T.leaf_reference(t.digest, t.hsalt, t.troot, t.tdomain, t.tpos, t.path) == 1
    assert _words1(leaf, t.digest, t.hsalt, t.troot, t.tdomain, t.tpos, t.path) == [1]
    assert _words1(leaf, t.digest, t.hsalt, bytes(64), t.tdomain, t.tpos, t.path) == [0]


@pytest.mark.slow
def test_tile_unit_is_core_and_leaf():
    """The whole unit's ok is the core's ok and the leaf's opening: 1 on an honest 1 × 1 tile, 0 when the served y is
    tampered.  Slow: the whole unit (core σ and the leaf) lowered to bits, the only check it is their and."""
    t = Tile("h100", 256, 1, 1, 1, 0, 1, "unit")
    assert t.ok == 1
    assert _words1(t.tile_def(), *t.tile_args()) == [1]
    t.y[0][0] ^= 1 << 16
    assert _words1(t.tile_def(), *t.tile_args()) == [0]
