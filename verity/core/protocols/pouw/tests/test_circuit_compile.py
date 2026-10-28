"""`circuit.compile`: one pass over a served Program puts every registered matmul's dots into a scheme's tile units, and the
compiled Program's served outputs are the served Program's.

The scheme here (`compile-test`, tile 2 × 2; `compile-test-16`, 16 × 16) is a stand-in with a real scheme's shape,
``(a: r rows of A, b: c rows of B, salt, position, roots) -> {z, receipt}``: the unit derives its seed
from the salt, its position and its seed group's roots (`seeds.PouwSeed`), and from the seed one noise block a row of A; it
adds the noise to its rows, folds each cell's noised arm into the receipt (the s8 family's prefix arm, or for another kind its
own Definition on the noised row) and returns its cells' exact products as z. The one-stage layer is `TemplatePopulations` over
`ScaledMmS8Coordinate_v1` at K = 2048 and 8192, as `benchmarks/one_stage/a6.py` builds it."""
from __future__ import annotations

import dataclasses
import hashlib
import random
import struct
from functools import lru_cache

import pytest

from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import fp32 as FP
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import s8 as BS8
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Record, Tuple, Value
from verity.core.protocols.pouw import grind as GR
from verity.ml.protocols.pouw.circuit import compile as C
from verity.ml.protocols.pouw.circuit import hashes as H
from verity.ml.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.protocol import DomainError
from verity.core.protocols.sampled_proofs.one_stage import partition as OP

V1, V8, V16, V32 = Value(1), Value(8), Value(16), Value(32)
M32 = (1 << 32) - 1
LAYER = ((2048, 21504), (8192, 2048))
REDUCED = ((2048, 3), (8192, 2))
BOOLEAN = {"DOT": BG.BlackwellS8ImmaDot32, "TDOT": None}
PORTS = C._ports
ports = pytest.mark.skipif(PORTS is None, reason="port types (ml.circuits.ports) are not on this tree")
#: the coin rules' seed role (`ports._coin_role`, a coin port beside an `In(kind="position")` one): the salt's flow into the seed
#: is the seed's, and the Build check refuses some units before compile's own checks do (`UNIT_REFUSED`)
SEEDS = PORTS is not None and "position" in PORTS.KINDS
#: the coin rules before their seed role let a coin reach only a noise primitive's or a receipt fold's coin port, so it refuses the
#: salt's flow into the seed's gates
COIN_RULES = PORTS is not None and hasattr(PORTS, "_coin_role") and not SEEDS
compiles = pytest.mark.skipif(COIN_RULES, reason="COIN-RULES refuses the salt's flow into PouwSeed: E_COIN_FLOW")
SALT = hashlib.sha512(b"salt").digest()


def _ports(decl):
    """`decl` ({"params" | "ret": a function of `ports`}) as `Ports`."""
    return PORTS.Ports(**{k: v(PORTS) for k, v in decl.items()})


def _prim(name, params, ret, f, ports=None):
    kw = {} if PORTS is None or ports is None else {"ports": _ports(ports)}
    return PrimitiveDefinition(name, 1, params, ret, f, conformance="protocol-defined", register=False, **kw)


def _def(name, statics, signature, body, ports=None):
    kw = {} if PORTS is None or ports is None else {"ports": lambda S: _ports(ports)}
    return CompositeDefinition(name, 1, statics, signature, body, conformance="protocol-defined", register=False, **kw)


COIN = {"params": lambda p: {"e": p.In(kind="coin")}}
FOLD = {"params": lambda p: {"r": p.In(receipt=True)}, "ret": lambda p: p.Out(sink=True)}
UNIT_PORTS = {"params": lambda p: {"salt": p.In(kind="coin"), **({"position": p.In(kind="position")} if SEEDS else {})},
              "ret": lambda p: {"receipt": p.Out(sink=True)}}

ADD = {w: _prim(f"CompileTestAdd{w}", (("x", Value(w)), ("e", Value(w))), Value(w), lambda x, e, w=w: (x + e) % (1 << w))
       for w in (8, 16)}
XOR1 = _prim("CompileTestXor1", (("x", V1), ("e", V1)), V1, lambda x, e: x ^ e)
#: a noise word's w-bit pieces, least significant first
PIECES = {w: tuple(_prim(f"CompileTestPiece{w}x{s}", (("x", V32),), Value(w), lambda x, s=s, w=w: (x >> w * s) % (1 << w))
                   for s in range(32 // w)) for w in (8, 16)}
ADD8 = ADD[8]
#: a noise primitive (the coin rules' coin flow): it adds its coin to its operand
NOISED8 = _prim("CompileTestNoised8", (("x", V8), ("e", V8)), V8, lambda x, e: (x + e) & 0xFF, ports=COIN)
FOLD32 = _prim("CompileTestFold32", (("r", V32), ("t", V32)), V32, lambda r, t: (r * 0x9E3779B1 + t) & M32, ports=FOLD)
FOLD1 = _prim("CompileTestFold1", (("r", V1), ("t", V1)), V1, lambda r, t: r ^ t, ports=FOLD)
NEG8 = _prim("CompileTestNeg8", (("x", V8),), V8, lambda x: -x & 0xFF)
INC32 = _prim("CompileTestInc32", (("x", V32),), V32, lambda x: (x + 1) & M32)
ADD32 = _prim("CompileTestAdd32", (("x", V32), ("y", V32)), V32, lambda x, y: (x + y) & M32)


# -- the test scheme ---------------------------------------------------------------------------------------------------------

def _boolean(S) -> bool:
    return S.EXACT.signature()[0][0][1].homogeneous_width() == 1


def _tile_params(S):
    (_, a), (_, b) = S.EXACT.signature()[0][:2]
    D, Pt = SD.types(_boolean(S))
    return ("a", Array(S.R, a)), ("b", Array(S.C, b)), ("salt", D), ("position", Pt), ("roots", Array(2, D))


def _tile_ret(S, receipt):
    return Record(("z", Array(len(S.PAIRS), S.EXACT.signature()[1])), ("receipt", receipt))


def _unit_signature(S):
    return _tile_params(S), _tile_ret(S, V1 if _boolean(S) else V32)


def _noised(B, S, a, seed):
    """A's rows plus the seed's noise: on bits the stream's first bits, on words of w bits a noise block's 512 / w pieces a
    row, repeated along k."""
    row = a.type.elem
    if _boolean(S):
        flat = Array(a.type.leaves, V1)
        e = SD.stream_bits(B, seed, a.type.leaves)
        return Coll(a.type, B.batch(XOR1, Coll(flat, a.refs), e, axes=(0, 0)).refs)
    w = row.homogeneous_width()
    flat = Array(a.type.leaves, Value(w))
    parts = []
    for blk in SD.noise_blocks(B, seed, S.R, False):
        piece = concat([B.batch(f, blk, axes=(0,)).refs for f in PIECES[w]])
        parts.append(concat([piece] * -(-row.leaves // piece.count)).slice(0, row.leaves))
    return Coll(a.type, B.batch(ADD[w], Coll(flat, a.refs), Coll(flat, concat(parts)), axes=(0, 0)).refs)


def _fold(B, S, noised, b):
    """The receipt: each cell's noised arm folded, the s8 prefix words (`NOISED`) or the leaves of the kind's own Definition
    on the noised row."""
    fold, width = (FOLD1, V1) if _boolean(S) else (FOLD32, V32)
    r = None
    for i, j in S.PAIRS:
        y = B.call(S.EXACT if S.NOISED is None else S.NOISED, noised[i], b[j])
        for t in range(y.type.leaves):
            leaf = Coll(width, y.refs.slice(t, t + 1))
            r = leaf if r is None else B.call(fold, r, leaf)
    return r


def _z(B, S, a, b):
    return [B.call(S.EXACT, a[i], b[j]).refs for i, j in S.PAIRS]


def _unit_body(B, S, a, b, salt, position, roots):
    seed = SD.seed_call(B, salt, position, roots, _boolean(S))
    r = _fold(B, S, _noised(B, S, a, seed), b)
    return Coll(_unit_signature(S)[1], concat(_z(B, S, a, b) + [r.refs]))


TILE = ("EXACT", "NOISED", "R", "C", "PAIRS")
UNIT = _def("CompileTestUnit", TILE, _unit_signature, _unit_body, ports=UNIT_PORTS)


def _statics(dot, tile):
    noised = None
    if dot.a == Array(dot.k, V8):
        noised = bind(S8.S8Dot, K=dot.k, L=32, T0="zero", C=None, DOT=dot.step, A=(-128, 127), B=dot.b_bound or (-128, 127),
                      TAIL=(), TDOT=None, NOISE="P", LIMBS="none", COMMIT="prefix")
    return {"EXACT": dot.exact, "NOISED": noised, "R": tile.r, "C": tile.c, "PAIRS": tile.pairs}


def _unit(defn):
    return lambda dot, tile: bind(defn, **_statics(dot, tile))


SCHEME = C.register(C.Scheme("compile-test", (2, 2), {C.ANY: _unit(UNIT)})).name
SCHEME16 = C.register(C.Scheme("compile-test-16", (16, 16), {C.ANY: _unit(UNIT)})).name


def _recompute_body(B, S, a, b, salt, position, roots):
    seed = SD.seed_call(B, salt, position, roots, False)
    z = _z(B, S, a, b)
    r = B.call(FOLD32, B.call(FOLD32, B.call(S.EXACT, a[0], a[0]), Coll(V32, z[0])), seed[0])
    return Coll(_unit_signature(S)[1], concat(z + [r.refs]))


RECOMPUTING = C.register(C.Scheme("compile-test-recompute", (1, 1), {C.ANY: _unit(
    _def("CompileTestRecompute", TILE, _unit_signature, _recompute_body, ports=UNIT_PORTS))})).name
WRONG = C.register(C.Scheme("compile-test-wrong", (1, 1), {C.ANY: lambda d, t: d.exact})).name


def _no_seed(B, S, a, b, salt, position, roots):
    z = _z(B, S, a, b)
    return z, B.call(FOLD32, B.call(P.ZERO32), Coll(V32, z[0]))


def _two_seeds(B, S, a, b, salt, position, roots):
    s0, s1 = (SD.seed_call(B, salt, position, roots, False) for _ in range(2))
    return _z(B, S, a, b), B.call(FOLD32, _fold(B, S, _noised(B, S, a, s0), b), s1[0])


def _own_position(B, S, a, b, salt, position, roots):
    seed = SD.seed_call(B, salt, H.Emit(B).array([0, 7]), roots, False)
    return _z(B, S, a, b), _fold(B, S, _noised(B, S, a, seed), b)


def _own_roots(B, S, a, b, salt, position, roots):
    seed = SD.seed_call(B, salt, position, Coll(roots.type, concat([roots[0].refs, roots[0].refs])), False)
    return _z(B, S, a, b), _fold(B, S, _noised(B, S, a, seed), b)


def _salt_leak(B, S, a, b, salt, position, roots):
    seed = SD.seed_call(B, salt, position, roots, False)
    return _z(B, S, a, b), B.call(FOLD32, _fold(B, S, _noised(B, S, a, seed), b), salt[0])


def _noised_z(B, S, a, b, salt, position, roots):
    noised = _noised(B, S, a, SD.seed_call(B, salt, position, roots, False))
    return [B.call(S.EXACT, noised[i], b[j]).refs for i, j in S.PAIRS], _fold(B, S, noised, b)


def _bad(name, body):
    """A 1 × 1 scheme whose unit's body is `body(B, S, *params)` -> (its cells' z refs, its receipt)."""
    def build(B, S, *params):
        z, r = body(B, S, *params)
        return Coll(_unit_signature(S)[1], concat(z + [r.refs]))

    defn = _def("CompileTest" + name.title().replace("-", ""), TILE, _unit_signature, build, ports=UNIT_PORTS)
    return C.register(C.Scheme(f"compile-test-{name}", (1, 1), {C.ANY: _unit(defn)})).name


UNIT_REFUSED = [
    ("seed", "no-seed", _no_seed),
    ("E_COIN_REUSE" if SEEDS else "seed", "two-seeds", _two_seeds),
    ("E_PORT_KIND" if SEEDS else "position", "own-position", _own_position),
    ("position", "own-roots", _own_roots),
    ("E_COIN_FLOW" if SEEDS else "salt-flow", "salt-leak", _salt_leak),
    ("noise-in-z", "noised-z", _noised_z),
]
UNIT_REFUSED = [pytest.param(code, _bad(name, body), id=name) for code, name, body in UNIT_REFUSED]


# -- served Programs ---------------------------------------------------------------------------------------------------------

def _coord(K):
    return bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)


@lru_cache(maxsize=None)
def _layer(members):
    """One Program per population, built once: `mixed_population_program` makes a new `TemplatePopulations_v1` each call."""
    return OP.mixed_population_program([(_coord(K), n) for K, n in members])


def _acc(K):
    return bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM120_S8)


def _shared_x_body(B, S, x, w):
    xn = B.batch(NEG8, x, axes=(0,))
    return B.batch(_acc(S.K), xn, w, axes=(None, 0))


SHARED_X = _def("CompileTestSharedX", ("K", "N"),
                lambda S: ((("x", Array(S.K, V8)), ("w", Array(S.N, Array(S.K, V8)))), Array(S.N, V32)), _shared_x_body)


def _grid_body(B, S, x, w):
    n = S.N
    xs = Coll(Array(n * n, x.type.elem), concat([x[i].refs for i in range(n) for _j in range(n)]))
    ws = Coll(Array(n * n, w.type.elem), concat([w[j].refs for _i in range(n) for j in range(n)]))
    return B.batch(_acc(64), xs, ws, axes=(0, 0))


GRID = _def("CompileTestGrid", ("N",), lambda S: ((("x", Array(S.N, Array(64, V8))), ("w", Array(S.N, Array(64, V8)))),
                                                  Array(S.N * S.N, V32)), _grid_body)


def _matmul_inc_body(B, S, a, b):
    return B.batch(INC32, B.call(bind(S8.S8Matmul, N=S.N, **S8.SMOKE_PLAIN), a, b), axes=(0,))


MATMUL_INC = _def("CompileTestMatmulInc", ("N",),
                  lambda S: ((("a", Array(64, V8)), ("b", Array(S.N, Array(64, V8)))), Array(S.N, V32)), _matmul_inc_body)


def _twice_body(B, S, x, w):
    return Coll(Array(2, V32), concat([B.call(_acc(64), x, w).refs, B.call(_acc(64), x, w).refs]))


TWICE = _def("CompileTestTwice", (), lambda S: ((("x", Array(64, V8)), ("w", Array(64, V8))), Array(2, V32)), _twice_body)


def _chained_body(B, S, x, w):
    y = B.call(_acc(64), x, w)
    xb = Coll(Array(64, V8), concat([B.call(p, y).refs for p in PIECES[8]] * 16))
    return Coll(Array(2, V32), concat([y.refs, B.call(_acc(64), xb, w).refs]))


#: two calls, the second's operand computed from the first's output (a layer's and the next's)
CHAINED = _def("CompileTestChained", (), lambda S: ((("x", Array(64, V8)), ("w", Array(64, V8))), Array(2, V32)),
               _chained_body)

FAMILIES = {
    "s8dot": (lambda: Program(bind(S8.S8Dot, **S8.SMOKE_PLAIN)), [("S8Dot_v1", (1, 64, 1))]),
    "s8matmul": (lambda: Program(bind(MATMUL_INC, N=3)), [("S8Matmul_v1", (1, 64, 3))]),
    "gemmacc-shared-x": (lambda: Program(bind(SHARED_X, K=64, N=3)), [("GemmAccumulatorS8_v1", (1, 64, 3))]),
    "scaledmm-nested": (lambda: Program(bind(G.ScaledMmS8, K=64, N=3, **G.GEMM_SM120_S8)), [("GemmAccumulatorS8_v1", (1, 64, 3))]),
    "gemmacc-bits": (lambda: Program(bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)), [("GemmAccumulatorS8_v2", (1, 64, 1))]),
    "two-calls": (lambda: Program(bind(TWICE)), [("GemmAccumulatorS8_v1", (1, 64, 1))] * 2),
}


# -- helpers -----------------------------------------------------------------------------------------------------------------

def _leaf(rng, w):
    if w == 32:
        return struct.unpack("<I", struct.pack("<f", rng.uniform(1e-3, 1e-1)))[0]
    if w == 8:
        return rng.randrange(-127, 128) & 0xFF
    return rng.randrange(1 << w)


def _args(fn, rng):
    return [[_leaf(rng, w) for w in t.leaf_widths()] for _n, t in fn.signature()[0]]


def _roots(c):
    """A (root(A), root(B)) per seed group."""
    return [(hashlib.sha512(b"A%d" % g).digest(), hashlib.sha512(b"B%d" % g).digest()) for g in range(len(c.groups))]


def _run(c, args, salt=SALT, roots=None, base=0):
    return evaluate(c.program.fn, *args, *c.inputs(salt, _roots(c) if roots is None else roots, base))


def _outside_units(c):
    """Calls of a registered family, and matmul steps, anywhere in the compiled Program but inside a unit."""
    units = {id(g.unit) for g in c.plan.groups}
    found, todo, seen = [], [c.program.fn], set()
    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        for nd in f.body.nodes:
            if id(nd.fn) in units:
                continue
            if getattr(nd.fn, "base_id", None) in C.MATMULS or C.STEP.fullmatch(nd.fn.id):
                found.append(nd.fn.id)
            elif hasattr(nd.fn, "body"):
                todo.append(nd.fn)
    return found


def _refused(code, program, scheme=SCHEME):
    with pytest.raises(C.CompileError) as e:
        C.compile(program, scheme)
    assert e.value.code == code, str(e.value)
    return e.value


# -- the layer ---------------------------------------------------------------------------------------------------------------

@compiles
@pytest.mark.slow
def test_llama_layer_compiles():
    """The one-stage Llama-3.2-1B layer at full size, 16 × 16 tiles: one site per K, each 16 coordinates one unit, the coordinates'
    rest one `PouwRest` batch reading only the scales and z, a position slot a unit, and the verifier's rebuilt Program the
    one compiled. Its values are `test_reduced_layer_serves_the_same_outputs`'s; this one catches what only the full counts
    show (a site's tiling and packing, and the Program's digest, at 1472 units over 23552 dots)."""
    served = _layer(LAYER)
    c = C.compile(served, SCHEME16)
    assert [(s.dot.family, s.dot.k, s.count, s.shape) for s in c.sites] == [
        ("GemmAccumulatorS8_v1", 2048, 21504, (21504, 2048, 21504)), ("GemmAccumulatorS8_v1", 8192, 2048, (2048, 8192, 2048))]
    diagonal = C.Tile(16, 16, tuple((i, i) for i in range(16)))
    assert {u.tile for u in c.units} == {diagonal}
    assert [u.slot for u in c.units] == list(range(1344 + 128))
    assert (c.units[0].rows, c.units[0].cols) == (tuple(range(16)), tuple(range(16)))
    nodes = c.program.fn.body.nodes
    assert [nd.fn.base_id for nd in nodes] == ["CompileTestUnit_v1", "PouwRest_v1"] * 2
    assert [nd.n for nd in nodes] == [1344, 21504, 128, 2048]
    assert [n for n, _t in nodes[1].fn.signature()[0]] == ["sx", "sw", C.Z]
    assert [n for n, _t in c.program.fn.signature()[0]][-3:] == [C.SALT, C.POSITIONS, C.ROOTS]
    assert not _outside_units(c)
    assert C.rebuild(served, SCHEME16, encode_program(c.program)).digest == c.digest
    with pytest.raises(DomainError):
        c.check_descriptor(encode_program(C.compile(Program(bind(TWICE)), SCHEME16).program))


@compiles
def test_reduced_layer_serves_the_same_outputs():
    """The layer's two K's at 3 and 2 coordinates, 2 × 2 tiles: on random rows and scales the compiled Program's served outputs
    are the served Program's, its receipts follow the salt and the position range, the gate-level checks pass and its
    partition is one unit per tile and one per rest instance."""
    served = _layer(REDUCED)
    c = C.compile(served, SCHEME)
    two, one = C.Tile(2, 2, ((0, 0), (1, 1))), C.Tile(1, 1, ((0, 0),))
    assert [(u.site, u.slot, u.rows, u.tile) for u in c.units] == [(0, 0, (0, 1), two), (0, 1, (2,), one), (1, 2, (0, 1), two)]
    rng = random.Random(7)
    args = _args(served.fn, rng)
    want = evaluate(served.fn, *args)
    got = _run(c, args)
    assert len(want) == c.served_leaves == 5
    assert got[:5] == want
    assert len(got) == 5 + 3
    other = _run(c, args, salt=hashlib.sha512(b"other").digest())
    assert other[:5] == want and all(x != y for x, y in zip(other[5:], got[5:]))
    later = _run(c, args, base=len(c.units))
    assert later[:5] == want and all(x != y for x, y in zip(later[5:], got[5:]))
    gc = c.check_gates()
    assert gc.ok, gc
    assert (gc.reached, gc.z_gates, gc.z_bits) == (0, 5, 5 * 32)
    assert OP.units(c.program, c.query()) == 3 + 3 + 2


# -- every registered family -------------------------------------------------------------------------------------------------

@compiles
@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if "bits" in n else n for n in sorted(FAMILIES)])
def test_family_compiles_to_units(name):
    """Each registered family, at the root, batched, nested or on bits: its dots are the units' cells (one site per call of a
    kind), no registered Call or matmul step is left outside them, and the served outputs are unchanged. On bits the seed and
    the noise are SHA-512 in gates (three compressions), which is what makes it slow."""
    build, want_sites = FAMILIES[name]
    served = build()
    c = C.compile(served, SCHEME)
    assert [(s.dot.family, s.shape) for s in c.sites] == want_sites
    assert c.boolean == ("bits" in name)
    assert not _outside_units(c)
    rng = random.Random(name)
    args = _args(served.fn, rng)
    out = _run(c, args)
    assert out[:c.served_leaves] == evaluate(served.fn, *args)
    assert len(out) == c.served_leaves + sum(u.receipt.leaves for u in c.units)


@compiles
@pytest.mark.slow
def test_s8dot_on_bits_compiles():
    """The Boolean s8 family's unit: its salt, positions and roots are bits. Slow: building the seed's and the noise's
    SHA-512 gates."""
    c = C.compile(Program(bind(BS8.S8Dot, **S8.SMOKE_PLAIN | BOOLEAN)), SCHEME)
    assert [(s.dot.family, s.shape) for s in c.sites] == [("S8Dot_v2", (1, 64, 1))]
    assert [u.receipt for u in c.units] == [V1]
    assert [t for _n, t in c.program.fn.signature()[0]][-3:] == [SD.DIGEST_BITS, Array(1, SD.POS_BITS),
                                                                 Array(1, Array(2, SD.DIGEST_BITS))]
    assert not _outside_units(c)


@compiles
def test_shared_operand_is_one_row():
    """A's row shared by a batch is one committed row: the site's tiles pair it with B's rows, two to a tile."""
    c = C.compile(Program(bind(SHARED_X, K=64, N=3)), SCHEME)
    (s,) = c.sites
    assert s.pairs == ((0, 0), (0, 1), (0, 2))
    assert [(r.space, r.idx) for r in s.a_rows] == [("n", 0)]
    assert [(r.space, r.idx) for r in s.b_rows] == [("p", 1)] * 3
    assert [(u.rows, u.cols, u.tile) for u in c.units] == [((0,), (0, 1), C.Tile(1, 2, ((0, 0), (0, 1)))),
                                                           ((0,), (2,), C.Tile(1, 1, ((0, 0),)))]


@compiles
def test_grid_site_tiles_into_full_tiles():
    """One batch over a 4 × 4 grid of (row of A, row of B): four 2 × 2 tiles, each the whole grid of its rows and columns."""
    served = Program(bind(GRID, N=4))
    c = C.compile(served, SCHEME)
    (s,) = c.sites
    assert s.shape == (4, 64, 4)
    full = C.Tile(2, 2, ((0, 0), (0, 1), (1, 0), (1, 1)))
    assert [(u.rows, u.cols, u.tile) for u in c.units] == [((0, 1), (0, 1), full), ((0, 1), (2, 3), full),
                                                           ((2, 3), (0, 1), full), ((2, 3), (2, 3), full)]
    assert full.grid
    args = _args(served.fn, random.Random(3))
    assert _run(c, args)[:16] == evaluate(served.fn, *args)


# -- refusals ----------------------------------------------------------------------------------------------------------------

def _step_body(B, S, x, w):
    return Coll(Array(2, V32), concat([B.call(_acc(32), x, w).refs, B.call(P.BlackwellS8ImmaDot32, B.call(P.ZERO32), x, w).refs]))


def _inner_body(B, S, x, w):
    return B.call(_acc(64), B.batch(NEG8, x, axes=(0,)), w)


INNER = _def("CompileTestInner", (), lambda S: ((("x", Array(64, V8)), ("w", Array(64, V8))), V32), _inner_body)


def _scan_step(B, S, c, x, w):
    acc = B.call(_acc(64), x, w)
    return Coll(Tuple(V32, V32), concat([B.call(ADD32, c, acc).refs, acc.refs]))


SCAN_STEP = _def("CompileTestScanStep", (), lambda S: ((("c", V32), ("x", Array(64, V8)), ("w", Array(64, V8))),
                                                       Tuple(V32, V32)), _scan_step)


def _pair_body(B, S, z):
    one = B.call(INC32, z)
    return Coll(Array(2, V32), concat([one.refs, B.call(INC32, one).refs]))


PAIR = _def("CompileTestPair", (), lambda S: ((("z", V32),), Array(2, V32)), _pair_body)


def _seeding_body(B, S, x, w, salt, roots):
    seed = SD.seed_call(B, salt, H.Emit(B).array([0, 0]), roots, False)
    return B.call(ADD32, B.call(_acc(64), x, w), seed[0])


XW = (("x", Array(64, V8)), ("w", Array(64, V8)))
REFUSED = {
    "uncovered": lambda: _def("CompileTestBareStep", (), lambda S: ((("x", Array(32, V8)), ("w", Array(32, V8))), Array(2, V32)),
                              _step_body),
    "operand-inside": lambda: _def("CompileTestOuter", (), lambda S: (XW, V32), lambda B, S, x, w: B.call(INNER, x, w)),
    "scan": lambda: _def("CompileTestScan", (), lambda S: ((("c", V32), ("x", Array(3, Array(64, V8))), ("w", Array(3, Array(64, V8)))),
                                                           Tuple(V32, Array(3, V32))),
                         lambda B, S, c, x, w: B.scan(SCAN_STEP, c, xs=(x, w))),
    "width": lambda: _def("CompileTestWide", (), lambda S: (XW, Array(2, V32)), lambda B, S, x, w: B.call(PAIR, B.call(_acc(64), x, w))),
    "no-matmul": lambda: _def("CompileTestNoMatmul", (), lambda S: ((("z", V32),), Array(2, V32)), lambda B, S, z: B.call(PAIR, z)),
    "name": lambda: _def("CompileTestSaltName", (), lambda S: (((C.SALT, Array(64, V8)), ("w", Array(64, V8))), V32),
                         lambda B, S, x, w: B.call(_acc(64), x, w)),
    "coin-read": lambda: _def("CompileTestSeeding", (),
                              lambda S: ((*XW, ("salt", SD.DIGEST), ("roots", Array(2, SD.DIGEST))), V32), _seeding_body,
                              ports={"params": lambda p: {"salt": p.In(kind="coin")}}),
}


@compiles
@pytest.mark.parametrize("code", sorted(REFUSED))
def test_refused(code):
    """Each rule's refusal; ``coin-read``: a served Program that derives a seed itself (the salt enters only through compile),
    which the coin rules' seed role refuses as it is built: its seed's position is a constant."""
    if code == "coin-read" and SEEDS:
        with pytest.raises(PORTS.PortError) as e:
            Program(bind(REFUSED[code]()))
        assert e.value.code == "E_PORT_KIND", str(e.value)
        return
    _refused(code, Program(bind(REFUSED[code]())))


def test_edited_arm_refused():
    """A served Program computes the plain arm; noise enters only through the scheme's units."""
    _refused("edited-arm", Program(bind(S8.S8Dot, **S8.SMOKE_EDITED)))


@compiles
def test_compiled_program_reads_a_coin_and_is_refused():
    c = C.compile(Program(bind(TWICE)), SCHEME)
    _refused("coin-read", c.program)


def test_scheme_refused():
    _refused("scheme", Program(bind(TWICE)), "no-such-scheme")
    _refused("scheme", Program(bind(TWICE)), WRONG)
    with pytest.raises(ValueError):
        C.register(C.Scheme("compile-test-empty", (0, 2), {C.ANY: _unit(UNIT)}))
    with pytest.raises(ValueError):
        C.register(C.Scheme("compile-test-uncovering", (2, 2), {C.ANY: _unit(UNIT)}, cover=0))


# -- kinds and seed groups ---------------------------------------------------------------------------------------------------

BF = P.BF16
QK = bind(G.DotBf16, K=64, **G.GEMM_AMPERE)


def _scores_body(B, S, q, k):
    s = B.batch(QK, q, k, axes=(None, 0))
    m = s[0]
    for j in range(1, S.N):
        m = B.call(FP.F32Max, m, s[j])
    return m


SCORES = _def("CompileTestScores", ("N",), lambda S: ((("q", Array(64, BF)), ("k", Array(S.N, Array(64, BF)))), P.F32),
              _scores_body)
HEADS = _def("CompileTestHeads", ("H", "N"),
             lambda S: ((("q", Array(S.H, Array(64, BF))), ("k", Array(S.H, Array(S.N, Array(64, BF))))), Array(S.H, P.F32)),
             lambda B, S, q, k: B.batch(bind(SCORES, N=S.N), q, k, axes=(0, 0)))


def _attn_body(B, S, q, k, v):
    p = B.batch(P.F2fpBf16, B.batch(QK, q, k, axes=(None, 0)), axes=(0,))
    return B.call(P.AmpereBF16TcDot16, B.call(P.ZERO32), p, v)


ATTN = _def("CompileTestAttn", (), lambda S: ((("q", Array(64, BF)), ("k", Array(16, Array(64, BF))), ("v", P.A16)), P.F32),
            _attn_body)
ATTN_HEADS = _def("CompileTestAttnHeads", ("H",),
                  lambda S: ((("q", Array(S.H, Array(64, BF))), ("k", Array(S.H, Array(16, Array(64, BF)))),
                              ("v", Array(S.H, P.A16))), Array(S.H, P.F32)),
                  lambda B, S, q, k, v: B.batch(ATTN, q, k, v, axes=(0, 0, 0)))
GQA = _def("CompileTestGqa", ("H", "N"),
           lambda S: ((("q", Array(S.H, Array(64, BF))), ("k", Array(S.N, Array(64, BF)))), Array(S.H, P.F32)),
           lambda B, S, q, k: B.batch(bind(SCORES, N=S.N), q, k, axes=(0, None)))
KINDS = C.register(C.Scheme("compile-test-kinds", (2, 2), {"GemmAccumulatorS8_v1": _unit(UNIT)}))
COVER2 = C.register(C.Scheme("compile-test-cover", (2, 2), {C.ANY: _unit(UNIT)}, cover=2)).name
#: the test scheme with no row form for QKᵀ, so no grind term
QK_OPEN = C.register(C.Scheme("compile-test-qk-open", (2, 2), {C.ANY: _unit(UNIT)}, rows={})).name
#: a KV head's query heads × its keys, one tile
ATTN_TILE = C.register(C.Scheme("compile-test-attn", (4, 16), {C.ANY: _unit(UNIT)})).name

ROW = Array(128, BF)
QK128 = bind(G.DotBf16, K=128, **G.GEMM_AMPERE)
#: one query row's scores on its N keys
KEYS = _def("CompileTestKeys", ("N",), lambda S: ((("q", ROW), ("k", Array(S.N, ROW))), Array(S.N, P.F32)),
            lambda B, S, q, k: B.batch(QK128, q, k, axes=(None, 0)))
#: a KV head's G query heads on its N keys (GQA)
KV_DECODE = _def("CompileTestKvDecode", ("G", "N"),
                 lambda S: ((("q", Array(S.G, ROW)), ("k", Array(S.N, ROW))), Array(S.G, Array(S.N, P.F32))),
                 lambda B, S, q, k: B.batch(bind(KEYS, N=S.N), q, k, axes=(0, None)))
#: decode, one call: C (KV head, request) components
DECODE = _def("CompileTestDecode", ("C", "G", "N"),
              lambda S: ((("q", Array(S.C, Array(S.G, ROW))), ("k", Array(S.C, Array(S.N, ROW)))),
                         Array(S.C, Array(S.G, Array(S.N, P.F32)))),
              lambda B, S, q, k: B.batch(bind(KV_DECODE, G=S.G, N=S.N), q, k, axes=(0, 0)))


def _causal_body(B, S, q, k):
    parts = [B.batch(QK128, q[i], Coll(Array(i + 1, ROW), k.refs.slice(0, (i + 1) * ROW.leaves)), axes=(None, 0))
             for i in range(S.L)]
    return Coll(Array(S.L * (S.L + 1) // 2, P.F32), concat([p.refs for p in parts]))


#: one query head's causal scores over an L-token prompt: query i on keys 0 .. i
CAUSAL = _def("CompileTestCausal", ("L",), lambda S: ((("q", Array(S.L, ROW)), ("k", Array(S.L, ROW))),
                                                      Array(S.L * (S.L + 1) // 2, P.F32)), _causal_body)
#: a KV head's G query heads on the prompt's L keys
KV_PREFILL = _def("CompileTestKvPrefill", ("G", "L"),
                  lambda S: ((("q", Array(S.G, Array(S.L, ROW))), ("k", Array(S.L, ROW))),
                             Array(S.G, Array(S.L * (S.L + 1) // 2, P.F32))),
                  lambda B, S, q, k: B.batch(bind(CAUSAL, L=S.L), q, k, axes=(0, None)))
#: prefill, one call: C (KV head, request) components
PREFILL = _def("CompileTestPrefill", ("C", "G", "L"),
               lambda S: ((("q", Array(S.C, Array(S.G, Array(S.L, ROW)))), ("k", Array(S.C, Array(S.L, ROW)))),
                          Array(S.C, Array(S.G, Array(S.L * (S.L + 1) // 2, P.F32)))),
               lambda B, S, q, k: B.batch(bind(KV_PREFILL, G=S.G, L=S.L), q, k, axes=(0, 0)))


@compiles
def test_qk_heads_compile():
    """Attention's QKᵀ is credited: each head's scores, `DotBf16` of its query on its keys, are a site's dots,
    its query meeting only its own keys (a 1 × 2 tile a head), and the softmax max chain is the rest (under a scheme with
    no row form for QKᵀ, so no grind term: `test_qk_grind`)."""
    served = Program(bind(HEADS, H=4, N=2))
    c = C.compile(served, QK_OPEN)
    (s,) = c.sites
    assert (s.dot.family, s.shape, s.count) == ("DotBf16_v2", (4, 64, 8), 8)
    assert {u.tile for u in c.units} == {C.Tile(1, 2, ((0, 0), (0, 1)))}
    assert [(u.rows, u.cols) for u in c.units] == [((h,), (2 * h, 2 * h + 1)) for h in range(4)]
    assert [nd.fn.base_id for nd in c.program.fn.body.nodes] == ["CompileTestUnit_v1", "PouwRest_v1"]
    assert not _outside_units(c)
    args = _args(served.fn, random.Random(11))
    assert _run(c, args)[:4] == evaluate(served.fn, *args)


def _grind_pct(rows, dots):
    return f"{100 * GR.visible_premium(rows, GR.T_BITS) / dots:.4g}%"


@compiles
def test_qk_grind():
    """A seed group's QKᵀ carries GRIND's grind term, `grind.visible_premium` over its distinct rows (a Q row
    k + 320 bits read by its keys, a K row k bits read by the query heads on it) over its dots, and over 0.1% the group is
    refused, pointing to a live window's per-round coin. Four heads on their own keys are 4 Q rows and 8 K rows; four on shared keys
    (GQA) are 4 Q rows and 2 K rows read 4 times each, one row however many query heads read it."""
    e = _refused("cover", Program(bind(HEADS, H=4, N=2)))
    assert "grind term of its DotBf16_v2 dots is " + _grind_pct([(4, 384, 2), (8, 64, 1)], 8) in str(e)
    assert "a live per-round coin's, not a seed's" in str(e)
    e = _refused("cover", Program(bind(GQA, H=4, N=2)))
    assert "grind term of its DotBf16_v2 dots is " + _grind_pct([(4, 384, 2), (2, 64, 4)], 8) in str(e)
    assert C.compile(Program(bind(HEADS, H=4, N=2)), QK_OPEN).grind == ({},)
    assert C.SCHEMES[SCHEME].rows == C.ROWS and C.ROWS["DotBf16_v2"] == C.QK_ROWS
    assert C.QK_ROWS.bits(128) == (448, 128) and C.PV_ROWS.bits(2048) == (224, 2048)
    plugged = KINDS.plug("CompileTestPv_v1", _unit(UNIT), "compile-test-kinds-pv", rows=C.PV_ROWS)
    assert plugged.rows == {**C.ROWS, "CompileTestPv_v1": C.PV_ROWS}
    with pytest.raises(ValueError):
        C.register(C.Scheme("compile-test-one-sided", (2, 2), {C.ANY: _unit(UNIT)},
                            rows={"DotBf16_v2": C.RowNoise(a=(320, 0), b=(0, 0))}))


@compiles
def test_decode_grind_counts_every_query_head_pair():
    """At decode (Llama-3.1-8B: 8 KV heads of 4 query heads each, a head's query on its request's keys), one call's grind
    term counts every (query head, request) pair, its Q row, not the (KV head, request) components its K rows join: at 16
    keys a batch of 12 is over 0.1% (0.1002%) and 13 under it (0.0925%), where 13 requests are 104 components. Over few keys
    a KV head's K rows, read by all its query heads, bind instead: 70B's batch of 7 (8 query heads a KV head) is under 0.1%
    over 32 keys but not over 16, where GRIND's per-pair table, which assumes the Q rows bind, would pass it."""
    _refused("cover", Program(bind(DECODE, C=8 * 12, G=4, N=16)), ATTN_TILE)
    c = C.compile(Program(bind(DECODE, C=8 * 13, G=4, N=16)), ATTN_TILE)
    assert 9.2e-4 < c.grind[0]["DotBf16_v2"] <= GR.FLAG and len(c.units) == 8 * 13
    _refused("cover", Program(bind(DECODE, C=8 * 7, G=8, N=16)), ATTN_TILE)
    assert C.compile(Program(bind(DECODE, C=8 * 7, G=8, N=32)), ATTN_TILE).grind[0]["DotBf16_v2"] <= GR.FLAG


@compiles
def test_prefill_grind_counts_kv_components():
    """At prefill a KV head's K rows, read by its query heads on every later token, bind, so a call's grind term counts its
    (KV head, request) components: a 16-token causal prompt of 4 query heads a KV head needs 166 of them (165 is 0.10009%),
    far more than a decode's batch and far fewer than its rows. GRIND's table (`grind_price.prefill_cover`) says 171, since it
    counts every Q row's readers as all 16 keys and every K row's as 4 × 16; the rows' readers here are the causal ones."""
    _refused("cover", Program(bind(PREFILL, C=165, G=4, L=16)), ATTN_TILE)
    c = C.compile(Program(bind(PREFILL, C=166, G=4, L=16)), ATTN_TILE)
    assert 9.9e-4 < c.grind[0]["DotBf16_v2"] <= GR.FLAG


@compiles
def test_pv_inside_attention_is_uncovered():
    """P·V as attention serves it today, tensor-core steps on a P the block computes, is no Call of a registered family, so its
    dots can't be units: the pass refuses it rather than leave attention's second product uncredited."""
    e = _refused("uncovered", Program(bind(ATTN_HEADS, H=2)), QK_OPEN)
    assert "AmpereBF16TcDot16" in str(e)


@compiles
def test_kind_is_pluggable():
    """A scheme's unit, and so its noise form, is per kind: a kind it has none for is refused, and `plug` adds one."""
    served = Program(bind(S8.S8Dot, **S8.SMOKE_PLAIN))
    e = _refused("scheme", served, KINDS.name)
    assert "S8Dot_v1" in str(e)
    plugged = C.register(KINDS.plug("S8Dot_v1", _unit(UNIT), "compile-test-kinds-s8dot"))
    c = C.compile(served, plugged.name)
    assert [u.defn.base_id for u in c.units] == ["CompileTestUnit_v1"]
    assert set(plugged.units) == {"GemmAccumulatorS8_v1", "S8Dot_v1"} and set(KINDS.units) == {"GemmAccumulatorS8_v1"}
    _refused("scheme", Program(bind(HEADS, H=2, N=2)), plugged.name)


@compiles
def test_seed_group_covers_calls_committed_together():
    """Two calls committed together are one seed group: one slot of `pouw.roots` both units' seeds read, at
    their own positions, so their seeds still differ; keys are canonical, and a compiled Program is rebuilt only under its own
    groups."""
    served = Program(bind(TWICE))
    c = C.compile(served, SCHEME, groups=[0, 0])
    assert (c.group_of, c.groups, c.cover) == ((0, 0), ((0, 1),), (2,))
    assert c.program.fn.signature()[0][-1][1] == Array(1, Array(2, SD.DIGEST))
    root = (hashlib.sha512(b"a").digest(), hashlib.sha512(b"b").digest())
    s0, s1 = c.seeds(SALT, [root])
    assert s0 != s1
    with pytest.raises(ValueError):
        c.inputs(SALT, [root, root])
    args = _args(served.fn, random.Random(5))
    out = _run(c, args, roots=[root])
    assert out[:2] == evaluate(served.fn, *args) and out[2] != out[3]
    default = C.compile(served, SCHEME)
    assert (default.groups, default.cover) == (((0,), (1,)), (1, 1))
    assert C.compile(served, SCHEME, groups=["qk", "qk"]).digest == c.digest
    assert C.compile(served, SCHEME, groups=lambda s: s.dot.family).digest == c.digest
    assert C.rebuild(served, SCHEME, encode_program(c.program), groups=[0, 0]).digest == c.digest
    with pytest.raises(DomainError):
        C.rebuild(served, SCHEME, encode_program(c.program))
    with pytest.raises(ValueError):
        C.compile(served, SCHEME, groups=[0])


@compiles
def test_cover_is_the_least_seed_group():
    """A scheme's `cover` is the fewest units one seed group may cover: a call of one unit alone is refused, and committed
    with another it compiles. A grouped layer's units still compute no gate twice."""
    served = Program(bind(TWICE))
    e = _refused("cover", served, COVER2)
    assert "covers 1 units" in str(e)
    assert C.compile(served, COVER2, groups=lambda s: 0).cover == (2,)
    assert C.compile(Program(bind(SHARED_X, K=64, N=3)), COVER2).cover == (2,)
    layer = C.compile(_layer(REDUCED), SCHEME, groups=lambda s: 0)
    assert layer.cover == (3,)
    gc = layer.check_gates()
    assert gc.ok, gc


@compiles
def test_seed_group_never_spans_a_chain():
    """A seed binds its calls' operand roots, so one seed group can't hold a call and a call that reads its output (a step's
    layer ℓ and ℓ + 1): each alone compiles, with the served outputs unchanged, and together they are refused."""
    served = Program(bind(CHAINED))
    c = C.compile(served, SCHEME)
    assert c.groups == ((0,), (1,))
    args = _args(served.fn, random.Random(3))
    assert _run(c, args)[:2] == evaluate(served.fn, *args)
    with pytest.raises(C.CompileError) as e:
        C.compile(served, SCHEME, groups=[0, 0])
    assert e.value.code == "span" and "node 0" in str(e.value), str(e.value)


#: the floor's forms (`Scheme.noise`): the s8 accumulator's on both rows, S8Dot's with B's rows clean, none for S8Matmul
FLOOR = C.register(C.Scheme("compile-test-floor", (2, 2), {C.ANY: _unit(UNIT)}, rows={},
                            noise={"GemmAccumulatorS8_v1": C.RowNoise(a=(0, 1), b=(256, 1)),
                                   "S8Dot_v1": C.RowNoise(a=(0, 1), b=(0, 0))}))


def _floored_body(B, S, x, w):
    acc = B.batch(_acc(64), x, w, axes=(None, 0))
    dot = B.call(bind(S8.S8Dot, **S8.SMOKE_PLAIN), x, w[0])
    mm = B.call(bind(S8.S8Matmul, N=2, **S8.SMOKE_PLAIN), x, Coll(Array(2, w.type.elem), w.refs.slice(0, 2 * 64)))
    return Coll(Array(6, V32), concat([acc.refs, dot.refs, mm.refs]))


#: x against 3 rows of w on the accumulator, x · w[0] on S8Dot and x against w[0..1] on S8Matmul
FLOORED = _def("CompileTestFloored", (), lambda S: ((("x", Array(64, V8)), ("w", Array(3, Array(64, V8)))), Array(6, V32)),
               _floored_body)


@compiles
def test_floor_is_each_units_noised_multiply_adds():
    """`Compiled.floor`, PouwBlockCredit's W_b: a unit's dots whose kind's form noises both rows, k = 64 multiply-adds each,
    by slot and at its batch's node of the root. The accumulator's 3 dots on x's one row are a 1 × 2 tile (128) and a
    1 × 1 (64); S8Dot's form has B's rows clean and S8Matmul has none, so their units' floor is 0."""
    c = C.compile(Program(bind(FLOORED)), FLOOR.name)
    assert [(s.dot.family, s.count) for s in c.sites] == [("GemmAccumulatorS8_v1", 3), ("S8Dot_v1", 1), ("S8Matmul_v1", 2)]
    assert [(f.kind, f.dots, f.macs) for f in c.floor] == [
        ("GemmAccumulatorS8_v1", 2, 128), ("GemmAccumulatorS8_v1", 1, 64), ("S8Dot_v1", 0, 0), ("S8Matmul_v1", 0, 0)]
    assert [len(u.tile.pairs) for u in c.units] == [2, 1, 1, 2]
    root = c.program.root.body.nodes
    assert [root[f.node].fn for f in c.floor] == [u.defn for u in c.units]
    assert [root[f.node].name for f in c.floor] == [f"pouw[{i}]" for i in range(4)]
    assert all(f.macs == 0 for f in C.compile(Program(bind(FLOORED)), SCHEME).floor)
    assert FLOOR.plug("S8Matmul_v1", _unit(UNIT), "compile-test-floor-mm").noise == FLOOR.noise
    for bad in ({"DotBf16_v2": C.QK_ROWS}, {"S8Dot_v1": C.RowNoise(a=(0, 1), b=(-1, 1))}):
        with pytest.raises(ValueError):
            C.register(C.Scheme("compile-test-floor-bad", (2, 2), {C.ANY: _unit(UNIT)}, noise=bad))


@compiles
def test_recompute_across_units_is_found():
    """A unit computing a value another unit computes (here each 1 × 1 unit of a shared row folds a · a) fails the gate-level
    cut's no-recompute rule; the seeds, each reading its own position in every gate, don't."""
    c = C.compile(Program(bind(MATMUL_INC, N=3)), RECOMPUTING)
    gc = c.check_gates()
    assert not gc.ok and "gate-recomputed" in gc.cut.codes


@compiles
@pytest.mark.parametrize("code,scheme", UNIT_REFUSED)
def test_unit_refused(code, scheme):
    """A unit that derives no seed or two (``seed``), seeds from a position or roots of its own choosing (``position``: a unit
    can't choose its position), lets the salt reach anything but the seed (``salt-flow``), or returns a z the seed reaches
    (``noise-in-z``)."""
    _refused(code, Program(bind(TWICE)), scheme)


@compiles
def test_shared_position_slots_refused(monkeypatch):
    """Two unit batches given the same slots of `pouw.positions` (here the second batch reading the first's) would seed two
    units of a site alike: the root-level check refuses it, and the coin rules' seed role as the compiled Program is built."""
    real = C._plan

    def shared(served, name):
        p = real(served, name)
        groups = list(p.groups)
        groups[1] = dataclasses.replace(groups[1], first=0)
        return dataclasses.replace(p, groups=tuple(groups))

    monkeypatch.setattr(C, "_plan", shared)
    _refused("E_COIN_REUSE" if SEEDS else "position", Program(bind(SHARED_X, K=64, N=5)))


def _coin_served_body(B, S, x, w, u):
    return B.call(_acc(64), x, Coll(w.type, B.batch(NOISED8, w, u, axes=(0, 0)).refs))


@ports
def test_served_coin_parameter_refused():
    """A served Program that takes a coin is refused, even one whose coin flows as the coin rules allow (into a noise
    primitive): the salt enters only through compile."""
    defn = _def("CompileTestCoin", (), lambda S: ((*XW, ("u", Array(64, V8))), V32), _coin_served_body,
                ports={"params": lambda p: {"u": p.In(kind="coin")}})
    _refused("coin-read", Program(bind(defn)))


# -- seeds and the verifier's rebuild ----------------------------------------------------------------------------------------

@compiles
def test_same_roots_different_positions_different_seeds():
    """Two calls on the same operands, so the same roots: their positions differ, so do their seeds and receipts, while z is
    the same; the same (salt, position) under other roots is another seed too."""
    c = C.compile(Program(bind(TWICE)), SCHEME)
    assert [u.slot for u in c.units] == [0, 1]
    root = (hashlib.sha512(b"same-a").digest(), hashlib.sha512(b"same-b").digest())
    s0, s1 = c.seeds(SALT, [root, root])
    assert s0 != s1
    assert c.seeds(SALT, [root, root], base=2) != [s0, s1]
    assert c.seeds(SALT, [root, (root[1], root[0])])[1] != s1
    args = _args(c.served.fn, random.Random(1))
    out = _run(c, args, roots=[root, root])
    assert out[0] == out[1] and out[2] != out[3]


@compiles
def test_positions_and_inputs():
    c = C.compile(Program(bind(SHARED_X, K=64, N=3)), SCHEME)
    assert c.positions() == [0, 1] and c.positions(2) == [2, 3]
    with pytest.raises(ValueError):
        c.positions((1 << 64) - 1)
    salt, pos, roots = c.inputs(SALT, _roots(c), 6)
    assert salt == SD.halves(SALT) and pos == [0, 6, 0, 7] and roots == SD.halves(b"".join(_roots(c)[0]))
    with pytest.raises(ValueError):
        c.inputs(SALT, _roots(c) * 2)
    for short in (SALT[:32], SALT + b"\0", SALT.hex()):
        for f in (c.inputs, c.seeds):
            with pytest.raises(ValueError, match="64-byte salt"):
                f(short, _roots(c))


# -- the tiles as an annotation ----------------------------------------------------------------------------------------------

TILED = {
    **{n: (b, None) for n, (b, _s) in FAMILIES.items()},
    "two-calls-grouped": (lambda: Program(bind(TWICE)), [0, 0]),
    "reduced-layer-grouped": (lambda: _layer(REDUCED), lambda s: 0),
    "grid": (lambda: Program(bind(GRID, N=4)), None),
}


@compiles
@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if "bits" in n else n for n in sorted(TILED)])
def test_tiles_are_compiles_annotation(name):
    """`tiles` is `compile`'s Find and Tile alone: the same sites, units, seed groups, cover and grind, and the same seed
    inputs, with no compiled Program, and the served Program it reads is the one given."""
    build, groups = TILED[name]
    served = build()
    t, c = C.tiles(served, SCHEME, groups), C.compile(served, SCHEME, groups)
    assert type(t) is C.Tiles and not hasattr(t, "program") and t.served is served
    assert (t.sites, t.group_of, t.groups, t.cover, t.grind, t.units, t.boolean) == \
        (c.sites, c.group_of, c.groups, c.cover, c.grind, c.units, c.boolean)
    roots = _roots(c)
    assert t.positions(3) == c.positions(3)
    assert t.seeds(SALT, roots, 3) == c.seeds(SALT, roots, 3)
    assert t.inputs(SALT, roots, 3) == c.inputs(SALT, roots, 3)


def _tiles_refused(code, program, scheme=SCHEME, groups=None):
    with pytest.raises(C.CompileError) as e:
        C.tiles(program, scheme, groups)
    assert e.value.code == code, str(e.value)
    return e.value


TILES_REFUSED = {
    **{code: (lambda f=f: Program(bind(f())), SCHEME, None) for code, f in REFUSED.items() if code not in ("name", "width")},
    "edited-arm": (lambda: Program(bind(S8.S8Dot, **S8.SMOKE_EDITED)), SCHEME, None),
    "scheme": (lambda: Program(bind(TWICE)), "no-such-scheme", None),
    "span": (lambda: Program(bind(CHAINED)), SCHEME, [0, 0]),
}


@pytest.mark.parametrize("code", sorted(TILES_REFUSED))
def test_tiles_refused(code):
    """`tiles` refuses by the rules that read the served Program alone, with `compile`'s codes (`test_refused`)."""
    build, scheme, groups = TILES_REFUSED[code]
    if code == "coin-read" and SEEDS:
        with pytest.raises(PORTS.PortError):
            build()
        return
    _tiles_refused(code, build(), scheme, groups)


def test_tiles_refuse_what_reads_the_served_program():
    """The seed groups' rules and an uncovered matmul step read the served Program, so `tiles` refuses them as `compile`
    does: a unit factory's wrong signature, a group under the scheme's cover or over the grind term, and P·V's steps."""
    _tiles_refused("scheme", Program(bind(TWICE)), WRONG)
    assert "covers 1 units" in str(_tiles_refused("cover", Program(bind(TWICE)), COVER2))
    assert "grind term" in str(_tiles_refused("cover", Program(bind(HEADS, H=4, N=2))))
    assert "AmpereBF16TcDot16" in str(_tiles_refused("uncovered", Program(bind(ATTN_HEADS, H=2)), QK_OPEN))


@compiles
@pytest.mark.parametrize("code", ["name", "width"])
def test_tiles_take_what_only_replace_refuses(code):
    """A parameter named as a seed input and a rest past the width rule are refused by what Replace builds, so the served
    Program's tiles are still its annotation."""
    served = Program(bind(REFUSED[code]()))
    assert len(C.tiles(served, SCHEME).units) == 1
    _refused(code, served)


def test_seed_reads_every_field():
    """seed_j and σ_R each move with every field, and no σ_R is a seed_j (σ_R's tag is its own)."""
    d = [hashlib.sha512(bytes([i])).digest() for i in range(6)]
    base = (d[0], 0, d[1], d[2])
    out = []
    for f in (SD.seed, SD.receipt_seed):
        got = {f(*base)}
        for i, v in enumerate((d[3], 1, d[4], d[5])):
            got.add(f(*base[:i], v, *base[i + 1:]))
        got.add(f(d[0], 0, d[2], d[1]))
        assert len(got) == 6
        out.append(got)
    assert not out[0] & out[1]
    assert C.seed is SD.seed
    for f in (SD.seed, SD.receipt_seed):
        with pytest.raises(ValueError):
            f(b"short", 0, d[1], d[2])


def _seed_args(boolean, salt, position, roots):
    return [SD.leaves(salt, boolean), SD.position_leaves(position, boolean), SD.leaves(b"".join(roots), boolean)]


@compiles
def test_seed_in_gates_is_the_reference():
    """`PouwSeed_v1`, `PouwUnitSeeds_v1` and `PouwNoiseBlock_v1` on words are `seeds.seed`, [`seeds.seed`,
    `seeds.receipt_seed`] and `seeds.noise_block`."""
    roots = (hashlib.sha512(b"ra").digest(), hashlib.sha512(b"rb").digest())
    want = SD.seed(SALT, 5, *roots)
    assert list(evaluate(bind(SD.PouwSeed), *_seed_args(False, SALT, 5, roots))) == SD.halves(want)
    assert list(evaluate(bind(SD.PouwUnitSeeds), *_seed_args(False, SALT, 5, roots))) == \
        SD.halves(want) + SD.halves(SD.receipt_seed(SALT, 5, *roots))
    assert list(evaluate(bind(SD.PouwNoiseBlock), SD.halves(want), SD.position_leaves(3, False))) == \
        SD.halves(SD.noise_block(want, 3))


@compiles
@pytest.mark.slow
def test_seed_on_bits_is_the_reference():
    """`PouwSeed_v2` and `PouwNoiseBlock_v2` on bits (three SHA-512 compressions) are the reference."""
    roots = (hashlib.sha512(b"ra").digest(), hashlib.sha512(b"rb").digest())
    want = SD.seed(SALT, 1 << 40, *roots)
    assert list(evaluate(SD.PouwSeedBits, *_seed_args(True, SALT, 1 << 40, roots))) == SD.leaves(want, True)
    assert list(evaluate(SD.PouwNoiseBlockBits, SD.leaves(want, True), SD.position_leaves(2, True))) == \
        SD.leaves(SD.noise_block(want, 2), True)
    assert SD.stream(want, 600) == SD.leaves(SD.noise_block(want, 0), True) + SD.leaves(SD.noise_block(want, 1), True)[:88]


@compiles
@pytest.mark.slow
def test_unit_seeds_on_bits_is_the_reference():
    """`PouwUnitSeeds_v2` on bits (four SHA-512 compressions) is [seed_j, σ_R]: the only test of its gates, which share the
    masked salt and roots between the two hashes."""
    roots = (hashlib.sha512(b"ra").digest(), hashlib.sha512(b"rb").digest())
    got = list(evaluate(SD.PouwUnitSeedsBits, *_seed_args(True, SALT, 1 << 40, roots)))
    assert got == SD.leaves(SD.seed(SALT, 1 << 40, *roots), True) + SD.leaves(SD.receipt_seed(SALT, 1 << 40, *roots), True)


@compiles
def test_rebuild_holds_the_descriptor():
    served = Program(bind(TWICE))
    c = C.compile(served, SCHEME)
    desc = encode_program(c.program)
    assert C.rebuild(served, SCHEME, desc).digest == c.digest
    other = encode_program(C.compile(Program(bind(SHARED_X, K=64, N=3)), SCHEME).program)
    with pytest.raises(DomainError):
        C.rebuild(served, SCHEME, other)
    with pytest.raises(DomainError):
        c.check_descriptor({"not": "a program"})
