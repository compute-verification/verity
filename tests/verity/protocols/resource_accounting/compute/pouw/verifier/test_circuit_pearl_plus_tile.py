"""Pearl+'s -h3 tile unit (`verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile`): its core on words equals the
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
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8, pearl_plus_tile as T
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES


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
    that opens the digest to a transcript root.  Its first ``tr`` rows are real (every row when None) and the rest zero
    filler, as a call's A is padded to a multiple of 16; y holds the real rows."""

    def __init__(self, dev: str, k: int, tm: int, tn: int, n: int, col: int, depth: int, seed: str, tr: int | None = None):
        rng = random.Random(seed)
        d = DEVICES[dev]
        tr = tm if tr is None else tr
        self.dev, self.k, self.tm, self.tr, self.tn, self.n, self.col, self.depth = dev, k, tm, tr, tn, n, col, depth
        self.a = [_f32(rng, k) for _ in range(tr)] + [[0] * k for _ in range(tm - tr)]
        self.b = [_f32(rng, k) for _ in range(tn)]
        self.rows, self.cols, self.midx = list(range(tm)), list(range(tn)), 7
        self.salt = bytes(rng.randrange(256) for _ in range(32))
        self.root = bytes(rng.randrange(256) for _ in range(64))
        scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=d, hashing="h3")
        self.scheme = scheme
        self.rho, self.rhor = inert_bound._rho(dev, None), inert_bound.row_rho(scheme, k)
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
        _d, ng = inert_bound._promoted(dev, k)
        self.cert = [[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
                      for j in range(tn)] for i in range(tm)]
        self.y = [[0] * n for _ in range(tr)]
        for i in range(tr):
            for j in range(tn):
                self.y[i][col * tn + j] = CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16
        self.digest, self.ok = self.reference()
        # a transcript leaf that opens the honest digest
        self.hsalt = bytes(rng.randrange(256) for _ in range(192))
        self.tdomain = bytes(rng.randrange(256) for _ in range(64))
        self.tpos = rng.randrange(1 << depth)
        self.path = [bytes(rng.randrange(256) for _ in range(64)) for _ in range(depth)]
        payload = hm96_reference(self.digest, self.hsalt)[2]
        self.troot = F3.root_ref(self.tdomain, SCHEMA, payload, self.tpos, self.path)

    def reference(self) -> tuple[bytes | None, int]:
        return T.reference(self.dev, self.k, self.n, self.a, self.b, self.y, self.cert, self.salt, self.root, self.midx,
                           self.rows, self.cols, self.col, self.rho, self.rhor)

    def declare(self, tr: int) -> None:
        """Bind the tile at ``tr`` real rows, y its first ``tr`` rows, as a builder calling its later rows filler would."""
        self.tr, self.y = tr, self.y[:tr]

    def core(self):
        return bind(T.PearlPlusTileCore, K=self.k, TM=self.tm, TR=self.tr, TN=self.tn, N=self.n, RHO=self.rho,
                    RHOR=self.rhor, DEV=self.dev)

    def core_v2(self):
        return bind(T.PearlPlusTileCoreV2, K=self.k, TM=self.tm, TR=self.tr, TN=self.tn, N=self.n, RHO=self.rho,
                    RHOR=self.rhor, DEV=self.dev)

    def on_words(self) -> tuple[bytes, int]:
        out = list(evaluate(self.core(), *(_flat(a) for a in self.core_args())))
        return bytes(out[:64]), out[64]

    def core_args(self) -> list:
        return [self.a, self.b, self.y, self.cert, self.salt, self.root, self.midx, self.rows, self.col, self.eb, self.fa,
                self.fb, self.g]

    def tile_def(self):
        return T.tile(self.scheme, k=self.k, tm=self.tm, tn=self.tn, n=self.n, depth=self.depth, tr=self.tr)

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


#: (TM, TR): a 2-row tile with one filler row, and a 4-row call's one tile, 16 rows of which 12 are filler (slow: a
#: 16-row tile on words is about 35 s)
FILLERS = [pytest.param(2, 1, id="2x1-tr1"), pytest.param(16, 4, id="16x1-tr4", marks=pytest.mark.slow)]


def _filler(tm: int, tr: int, seed: str) -> Tile:
    return Tile("h100", 256, tm, 1, 1, 0, 1, f"{seed}/{tm}/{tr}", tr=tr)


@pytest.mark.parametrize("tm,tr", FILLERS)
def test_filler_rows_pass(tm, tr):
    """A tile whose rows past its real rows are zero filler passes, as Lean's tile rule passes it (`Rule.rowsFailure`'s
    filler kind, `UnitRecord.credited`): the core on words is `reference`, ok = 1, and y's port is its TR real rows."""
    t = _filler(tm, tr, "filler")
    assert t.ok == 1
    assert dict(t.core().signature()[0])["y"].n == tr
    assert t.on_words() == (t.digest, 1)


@pytest.mark.parametrize("tm,tr", FILLERS)
@pytest.mark.parametrize("word", [0x3F800000, 0x80000000])
def test_a_nonzero_filler_row_fails(tm, tr, word):
    """A filler row with one nonzero word fails, −0 among them (Lean's `filler row … is not zero` compares the FP32 words):
    the core on words is `reference`, ok = 0."""
    t = _filler(tm, tr, f"nonzero/{word:x}")
    t.a[tm - 1][3] = word
    digest, ok = t.reference()
    assert ok == 0 and digest is not None
    assert t.on_words() == (digest, 0)


@pytest.mark.parametrize("tm,tr", [pytest.param(2, 2, id="2x1-tr2"),
                                   pytest.param(16, 4, id="16x1-tr4", marks=pytest.mark.slow)])
def test_a_real_row_declared_filler_fails(tm, tr):
    """A live row bound as filler by a smaller TR fails: TR is the record's real rows, so a builder that called a real
    row filler gets ok = 0, never a pass that skips the row's admission, inert-work bound and y.  The digest still covers
    every row."""
    t = _filler(tm, tr, "declared")
    assert t.ok == 1
    t.declare(tr - 1)
    digest, ok = t.reference()
    assert ok == 0 and digest == t.digest
    assert t.on_words() == (t.digest, 0)


def test_tr_is_within_the_tile():
    """TR is 1 to TM: a tile has a real row, and no more real rows than rows."""
    for tr in (0, 2):
        with pytest.raises(ValueError, match="real rows are 1 to its TM rows"):
            bind(T.PearlPlusTileCore, TM=1, TR=tr)


@pytest.mark.parametrize("tm,tn,n,gates", [(1, 1, 1, 11_443), (2, 2, 4, 24_340)])
def test_a_full_tile_is_todays_core(tm, tn, n, gates):
    """At TR = TM the core is today's, gate for gate: the word core's gate count as it was before TR (one-circuit at
    c56dce91380, which the node-for-node comparison of the word core, its σ and the unit matched), and its y port TM rows."""
    t = Tile("h100", 256, tm, tn, n, 0, 1, f"full/{tm}x{tn}")
    assert t.core().gates == gates
    assert dict(t.core().signature()[0])["y"].n == tm


@pytest.mark.slow
def test_core_sigma_is_its_words():
    """The σ core equals the word core on an honest 1 × 1 tile: same digest and ok.  Slow: lowering the core to bits is the
    only check that its σ (what the unit Calls) computes what its words do."""
    t = Tile("h100", 256, 1, 1, 1, 0, 1, "sigma")
    word = list(evaluate(t.core(), *(_flat(a) for a in t.core_args())))
    assert _words1(t.core_v2(), *t.core_args()) == word


@pytest.mark.slow
def test_filler_sigma_is_its_words():
    """The σ core equals the word core on a 2 × 1 tile whose second row is filler, zero (ok 1) and with a nonzero word
    (ok 0), in one batch.  Slow: as `test_core_sigma_is_its_words`, and the only check that the σ of the real-row rule
    (the filler rows' zero test, their constant admission word in the inert-work bound and the real rows' y) computes
    what its words do."""
    t = Tile("h100", 256, 2, 1, 1, 0, 1, "sigma/filler", tr=1)
    honest = t.core_args()
    t.a = [t.a[0], [1] + t.a[1][1:]]
    tampered = t.core_args()
    word = [list(evaluate(t.core(), *(_flat(a) for a in args))) for args in (honest, tampered)]
    assert [w[64] for w in word] == [1, 0]
    got = evaluate_words(t.core_v2(), *(np.asarray([_flat(x), _flat(z)], dtype=np.uint64) for x, z in zip(honest, tampered)))
    assert got.reshape(2, -1).tolist() == word


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
