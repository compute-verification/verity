"""`circuit.compile`: one pass over a served Program puts every registered matmul's dots into a scheme's units, and the compiled
Program's served outputs are the served Program's.

The scheme here (`compile-test`) is a stand-in with a real scheme's shape, ``(a, b, noise) -> {z, receipt}``: the noise added to
A's row, the noised arm folded into the receipt (the s8 family's prefix arm, or for a Boolean family its own Definition on the
noised row) and z the exact product. #2365's layer is `TemplatePopulations` over `ScaledMmS8Coordinate_v1` at K = 2048 and
8192, as `benchmarks/one_stage/a6.py` builds it."""
from __future__ import annotations

import random
import struct
from functools import lru_cache

import pytest

from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import s8 as BS8
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Record, Tuple, Value
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.protocol import DomainError, tagged
from verity.core.protocols.sampled_proofs.one_stage import partition as OP

V1, V8, V32 = Value(1), Value(8), Value(32)
M32 = (1 << 32) - 1
LAYER = ((2048, 21504), (8192, 2048))
REDUCED = ((2048, 3), (8192, 2))
BOOLEAN = {"DOT": BG.BlackwellS8ImmaDot32, "TDOT": None}
PORTS = C._ports
ports = pytest.mark.skipif(PORTS is None, reason="port types (ml.circuits.ports, COIN-RULES) are not on this tree")
#: COIN-RULES' one use reads each noise leaf once, so a noised bit can't feed a Boolean multiplier's partial products
bits = pytest.mark.skipif(PORTS is not None, reason="a Boolean unit's noised bits fan out to the multiplier: E_COIN_REUSE")


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
UNIT_PORTS = {"params": lambda p: {"noise": p.In(kind="coin")}, "ret": lambda p: {"receipt": p.Out(sink=True)}}

ADD8 = _prim("CompileTestAdd8", (("x", V8), ("e", V8)), V8, lambda x, e: (x + e) & 0xFF, ports=COIN)
XOR1 = _prim("CompileTestXor1", (("x", V1), ("e", V1)), V1, lambda x, e: x ^ e, ports=COIN)
FOLD32 = _prim("CompileTestFold32", (("r", V32), ("t", V32)), V32, lambda r, t: (r * 0x9E3779B1 + t) & M32, ports=FOLD)
FOLD1 = _prim("CompileTestFold1", (("r", V1), ("t", V1)), V1, lambda r, t: r ^ t, ports=FOLD)
NEG8 = _prim("CompileTestNeg8", (("x", V8),), V8, lambda x: -x & 0xFF)
INC32 = _prim("CompileTestInc32", (("x", V32),), V32, lambda x: (x + 1) & M32)
ADD32 = _prim("CompileTestAdd32", (("x", V32), ("y", V32)), V32, lambda x, y: (x + y) & M32)


# -- the test scheme ---------------------------------------------------------------------------------------------------------

def _words(dot) -> bool:
    return dot.a == Array(dot.k, V8)


def _unit_signature(S):
    (_, a), (_, b) = S.EXACT.signature()[0][:2]
    return (("a", a), ("b", b), ("noise", a)), Record(("z", S.EXACT.signature()[1]),
                                                      ("receipt", V1 if S.NOISED is None else V32))


def _unit_body(B, S, a, b, noise):
    words = S.NOISED is not None
    flat = Array(a.type.leaves, V8 if words else V1)
    a2 = Coll(a.type, B.batch(ADD8 if words else XOR1, Coll(flat, a.refs), Coll(flat, noise.refs), axes=(0, 0)).refs)
    if words:
        T = B.call(S.NOISED, a2, b)
        r = B.call(P.ZERO32)
        for t in range(T.type.leaves):
            r = B.call(FOLD32, r, Coll(V32, T.refs.slice(t, t + 1)))
    else:
        y = B.call(S.EXACT, a2, b)
        r = Coll(V1, y.refs.slice(0, 1))
        for i in range(1, y.type.leaves):
            r = B.call(FOLD1, r, Coll(V1, y.refs.slice(i, i + 1)))
    z = B.call(S.EXACT, a, b)
    return Coll(_unit_signature(S)[1], concat([z.refs, r.refs]))


UNIT = _def("CompileTestUnit", ("EXACT", "NOISED"), _unit_signature, _unit_body, ports=UNIT_PORTS)


def _unit(dot):
    noised = None
    if _words(dot):
        noised = bind(S8.S8Dot, K=dot.k, L=32, T0="zero", C=None, DOT=dot.step, A=(-128, 127), B=dot.b_bound or (-128, 127),
                      TAIL=(), TDOT=None, NOISE="P", LIMBS="none", COMMIT="prefix")
    return bind(UNIT, EXACT=dot.exact, NOISED=noised)


def _noise(seed, dot, i, j):
    out, c = [], 0
    while len(out) < dot.a.leaves:
        out += tagged("verity/pouw/compile-test/noise", seed, i, j, c)
        c += 1
    return out[:dot.a.leaves] if _words(dot) else [v & 1 for v in out[:dot.a.leaves]]


SCHEME = C.register(C.Scheme("compile-test", _unit, _noise)).name


def _recompute_signature(S):
    (_, a), (_, b) = S.EXACT.signature()[0][:2]
    return (("a", a), ("b", b), ("noise", a)), Record(("z", S.EXACT.signature()[1]), ("receipt", V32))


def _recompute_body(B, S, a, b, noise):
    z = B.call(S.EXACT, a, b)
    r = B.call(FOLD32, B.call(S.EXACT, a, a), z)
    return Coll(_recompute_signature(S)[1], concat([z.refs, r.refs]))


RECOMPUTE = _def("CompileTestRecompute", ("EXACT",), _recompute_signature, _recompute_body, ports=UNIT_PORTS)
RECOMPUTING = C.register(C.Scheme("compile-test-recompute", lambda d: bind(RECOMPUTE, EXACT=d.exact),
                                  lambda seed, d, i, j: [0] * d.a.leaves)).name
WRONG = C.register(C.Scheme("compile-test-wrong", lambda d: d.exact, _noise)).name


# -- served Programs ---------------------------------------------------------------------------------------------------------

def _coord(K):
    return bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)


@lru_cache(maxsize=None)
def _layer(members):
    """One Program per population: `TemplatePopulations`' identity is its counts alone, so two of equal counts are one binding."""
    return OP.mixed_population_program([(_coord(K), n) for K, n in members])


def _acc(K):
    return bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM120_S8)


def _shared_x_body(B, S, x, w):
    xn = B.batch(NEG8, x, axes=(0,))
    return B.batch(_acc(S.K), xn, w, axes=(None, 0))


SHARED_X = _def("CompileTestSharedX", ("K", "N"),
                lambda S: ((("x", Array(S.K, V8)), ("w", Array(S.N, Array(S.K, V8)))), Array(S.N, V32)), _shared_x_body)


def _matmul_inc_body(B, S, a, b):
    return B.batch(INC32, B.call(bind(S8.S8Matmul, N=S.N, **S8.SMOKE_PLAIN), a, b), axes=(0,))


MATMUL_INC = _def("CompileTestMatmulInc", ("N",),
                  lambda S: ((("a", Array(64, V8)), ("b", Array(S.N, Array(64, V8)))), Array(S.N, V32)), _matmul_inc_body)


def _twice_body(B, S, x, w):
    return Coll(Array(2, V32), concat([B.call(_acc(64), x, w).refs, B.call(_acc(64), x, w).refs]))


TWICE = _def("CompileTestTwice", (), lambda S: ((("x", Array(64, V8)), ("w", Array(64, V8))), Array(2, V32)), _twice_body)

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
    return [(b"A%d" % s.position, b"B%d" % s.position) for s in c.sites]


def _run(c, args, salt=b"salt", forward=0):
    return evaluate(c.program.fn, *args, c.noise_input(salt, forward, _roots(c)))


def _outside_units(c):
    """Calls of a registered family, and matmul steps, anywhere in the compiled Program but inside a unit."""
    units = {id(s.unit) for s in c.sites}
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

@pytest.mark.skipif(C._ports is not None, reason="COIN-RULES' one-use check tracks every coin leaf: 61M at full size")
def test_llama_layer_compiles():
    """#2365's Llama-3.2-1B layer at full size: one site per K, every coordinate's dot a unit, the coordinates' rest one
    `PouwRest` batch reading only the scales and z, and the verifier's rebuilt Program the one compiled. Its values are
    `test_reduced_layer_serves_the_same_outputs`'s; this one catches what only the full counts show (a site's packing and the
    Program's digest at 23552 units)."""
    served = _layer(LAYER)
    c = C.compile(served, SCHEME)
    assert [(s.dot.family, s.dot.k, s.count, s.shape) for s in c.sites] == [
        ("GemmAccumulatorS8_v1", 2048, 21504, (21504, 2048, 21504)), ("GemmAccumulatorS8_v1", 8192, 2048, (2048, 8192, 2048))]
    assert all(s.pairs == tuple((u, u) for u in range(s.count)) for s in c.sites)
    nodes = c.program.fn.body.nodes
    assert [nd.fn.base_id for nd in nodes] == ["CompileTestUnit_v1", "PouwRest_v1"] * 2
    assert [nd.n for nd in nodes] == [21504, 21504, 2048, 2048]
    assert [n for n, _t in nodes[1].fn.signature()[0]] == ["sx", "sw", C.Z]
    assert not _outside_units(c)
    assert C.rebuild(served, SCHEME, encode_program(c.program)).digest == c.digest
    with pytest.raises(DomainError):
        c.check_descriptor(encode_program(C.compile(Program(bind(TWICE)), SCHEME).program))


def test_reduced_layer_serves_the_same_outputs():
    """The layer's two K's at 3 and 2 coordinates: on random rows and scales the compiled Program's served outputs are the
    served Program's, its receipts follow the salt, the gate-level checks pass and its partition is one unit per instance."""
    served = _layer(REDUCED)
    c = C.compile(served, SCHEME)
    rng = random.Random(7)
    args = _args(served.fn, rng)
    want = evaluate(served.fn, *args)
    got = _run(c, args)
    assert len(want) == c.served_leaves == 5
    assert got[:5] == want
    assert len(got) == 5 + 5
    assert _run(c, args, salt=b"other")[:5] == want
    assert _run(c, args, salt=b"other")[5:] != got[5:]
    gc = c.check_gates()
    assert gc.ok, gc
    assert (gc.reached, gc.z_gates, gc.z_bits) == (0, 5, 5 * 32)
    assert OP.units(c.program, c.query()) == 3 + 3 + 2 + 2


# -- every registered family -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [pytest.param(n, marks=bits) if "bits" in n else n for n in sorted(FAMILIES)])
def test_family_compiles_to_units(name):
    """Each registered family, at the root, batched, nested or on bits: its dots are the units (one site per call of a kind),
    no registered Call or matmul step is left outside them, and the served outputs are unchanged."""
    build, want_sites = FAMILIES[name]
    served = build()
    c = C.compile(served, SCHEME)
    assert [(s.dot.family, s.shape) for s in c.sites] == want_sites
    assert not _outside_units(c)
    rng = random.Random(name)
    args = _args(served.fn, rng)
    out = _run(c, args)
    assert out[:c.served_leaves] == evaluate(served.fn, *args)
    assert len(out) == c.served_leaves + sum(s.count * s.receipt.leaves for s in c.sites)


@bits
def test_s8dot_on_bits_compiles():
    c = C.compile(Program(bind(BS8.S8Dot, **S8.SMOKE_PLAIN | BOOLEAN)), SCHEME)
    assert [(s.dot.family, s.shape, s.receipt) for s in c.sites] == [("S8Dot_v2", (1, 64, 1), V1)]
    assert not _outside_units(c)


def test_shared_operand_is_one_row():
    """A's row shared by a batch is one committed row: the site's units pair it with each of B's rows."""
    c = C.compile(Program(bind(SHARED_X, K=64, N=3)), SCHEME)
    (s,) = c.sites
    assert s.pairs == ((0, 0), (0, 1), (0, 2))
    assert [(r.space, r.idx) for r in s.a_rows] == [("n", 0)]
    assert [(r.space, r.idx) for r in s.b_rows] == [("p", 1)] * 3


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
    "name": lambda: _def("CompileTestNoiseName", (), lambda S: (((C.NOISE, Array(64, V8)), ("w", Array(64, V8))), V32),
                         lambda B, S, x, w: B.call(_acc(64), x, w)),
}


@pytest.mark.parametrize("code", sorted(REFUSED))
def test_refused(code):
    _refused(code, Program(bind(REFUSED[code]())))


def test_edited_arm_refused():
    """A served Program computes the plain arm; noise enters only through the scheme's units."""
    _refused("edited-arm", Program(bind(S8.S8Dot, **S8.SMOKE_EDITED)))


def test_compiled_program_reads_a_coin_and_is_refused():
    c = C.compile(Program(bind(TWICE)), SCHEME)
    _refused("coin-read", c.program)


def test_scheme_refused():
    _refused("scheme", Program(bind(TWICE)), "no-such-scheme")
    _refused("scheme", Program(bind(TWICE)), WRONG)


def test_recompute_across_units_is_found():
    """A unit computing a value another unit computes (here each of a shared row's units folds a · a) fails the gate-level
    cut's no-recompute rule."""
    c = C.compile(Program(bind(MATMUL_INC, N=3)), RECOMPUTING)
    gc = c.check_gates()
    assert not gc.ok and "gate-recomputed" in gc.cut.codes


def _coin_served_body(B, S, x, w, u):
    return B.call(_acc(64), x, Coll(w.type, B.batch(ADD8, w, u, axes=(0, 0)).refs))


@ports
def test_served_coin_parameter_refused():
    """A served Program that takes a coin (here a noised weight, a valid Program under COIN-RULES) is refused: the noise enters
    only through compile."""
    defn = _def("CompileTestCoin", (), lambda S: ((*XW, ("u", Array(64, V8))), V32), _coin_served_body,
                ports={"params": lambda p: {"u": p.In(kind="coin")}})
    _refused("coin-read", Program(bind(defn)))


def _noise_twice_body(B, S, a, b, noise):
    flat = Array(a.type.leaves, V8)
    y = [B.call(S.EXACT, Coll(a.type, B.batch(ADD8, Coll(flat, a.refs), Coll(flat, noise.refs), axes=(0, 0)).refs), b)
         for _ in range(2)]
    z = B.call(S.EXACT, a, b)
    return Coll(_recompute_signature(S)[1], concat([z.refs, B.call(FOLD32, B.call(FOLD32, B.call(P.ZERO32), y[0]), y[1]).refs]))


NOISE_TWICE = _def("CompileTestNoiseTwice", ("EXACT",), _recompute_signature, _noise_twice_body, ports=UNIT_PORTS)


@ports
def test_noise_read_twice_refused():
    """A unit whose noise feeds two noise primitives is refused by COIN-RULES' one use."""
    name = C.register(C.Scheme("compile-test-noise-twice", lambda d: bind(NOISE_TWICE, EXACT=d.exact), _noise)).name
    _refused("E_COIN_REUSE", Program(bind(TWICE)), name)


# -- seeds and the verifier's rebuild ----------------------------------------------------------------------------------------

def test_same_roots_different_seeds():
    """Two calls on the same operands, so the same roots: their positions differ, so do their seeds, noise and receipts."""
    c = C.compile(Program(bind(TWICE)), SCHEME)
    a, b = c.sites
    root = (b"same-a", b"same-b")
    s0, s1 = c.seeds(b"salt", 3, [root, root])
    assert s0 != s1
    assert c.unit_noise(s0, a, 0) != c.unit_noise(s1, b, 0)
    args = _args(c.served.fn, random.Random(1))
    out = evaluate(c.program.fn, *args, [v for s, sd in zip(c.sites, (s0, s1)) for v in c.unit_noise(sd, s, 0)])
    assert out[0] == out[1] and out[2] != out[3]


def test_seed_reads_every_field():
    base = (b"salt", 0, 0, b"ra", b"rb")
    seeds = {C.seed(*base)}
    for i, v in enumerate((b"salt2", 1, 1, b"ra2", b"rb2")):
        seeds.add(C.seed(*base[:i], v, *base[i + 1:]))
    assert len(seeds) == 6
    assert C.seed(*base) == tagged(C.SEED_TAG, *base)


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
