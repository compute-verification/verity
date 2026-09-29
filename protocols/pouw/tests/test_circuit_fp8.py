"""NCP-FP8's circuit (`verity_pouw.circuit.fp8`, #295's D-3s under layout (a)): its primitives are #295's arithmetic, its
Program evaluates to #295's scheme bit for bit, its partition passes the whole-call `validate_unit_cut`, and every PoUW
unit is behind the pinned Z separator."""
from __future__ import annotations

import hashlib
import random
import struct

import pytest

from verity.commitments import merkle as MK
from verity.evaluation import evaluate
from verity.ir.cut import CallGraph, check_cut
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.refs import array_of
from verity.ir.types import Array
from verity.ml.tc import fp16, fp32
from verity.randomness import frame
from verity_pouw.circuit import fp8, fp8_reference as R
from verity_pouw.circuit import fp8_words as FW
from verity_pouw.circuit import hoist as H
from verity_pouw.circuit import partition as P
from verity_pouw.protocol import DomainError
from verity_pouw.protocol import Shape as PShape


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _code(rng: random.Random) -> int:
    while True:
        c = rng.getrandbits(8)
        if c & 0x7F != 0x7F:                                # E4M3's NaN codes are malformed rows
            return c


def call(m: int, k: int, n: int, *, sums: str = "f16", index: int = 3, weight: int = 11, seed: int = 7) -> R.Call:
    rng = random.Random(seed)
    a = tuple(tuple(_code(rng) for _ in range(k)) for _ in range(m))
    b = tuple(tuple(_code(rng) for _ in range(k)) for _ in range(n))
    return R.Call(fp8.Shape(m, k, n), a, tuple(_f32(rng.uniform(1e-3, 2e-2)) for _ in range(m)), b,
                  tuple(_f32(rng.uniform(1e-3, 2e-2)) for _ in range(n)), tuple(_f32(rng.uniform(-1, 1)) >> 16 for _ in range(n)),
                  bytes(rng.randrange(256) for _ in range(32)), index, weight, sums)


def run(c: R.Call) -> tuple[list[list[int]], list[list[int]]]:
    """The Program's outputs: y (m × n BF16) and each tile's words."""
    out = evaluate(fp8.linear(c.shape, c.sums, c.lengths), *c.args())
    m, n, per = c.shape.m, c.shape.n, fp8.D * fp8.D * fp8.tile_words(c.shape.k)
    tiles = (m // fp8.D) * (n // fp8.D)
    assert len(out) == m * n + per * tiles
    return ([list(out[i * n:(i + 1) * n]) for i in range(m)],
            [list(out[m * n + t * per:m * n + (t + 1) * per]) for t in range(tiles)])


# ---- the primitives -------------------------------------------------------------------------------------------------------

def _f16(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(16)
        if w & 0x7C00 != 0x7C00:
            return w


def test_the_primitives_are_295s_arithmetic():
    rng = random.Random(1)
    for _ in range(3000):
        a, b, c = _f16(rng), _f16(rng), _f16(rng)
        assert FW.PouwF16Add.evaluate(a, b) == fp16.add(a, b)
        assert FW.PouwF16Mul.evaluate(a, b) == fp16.mul(a, b)
        assert FW.PouwF16Fma.evaluate(a, b, c) == fp16.fma(a, b, c)
        assert FW.PouwF16Max.evaluate(a, b) == fp16.max_(a, b)
        assert FW.PouwF16Neg.evaluate(a) == fp16.neg(a) and FW.PouwF16Abs.evaluate(a) == a & 0x7FFF
        assert FW.PouwF16ToF32.evaluate(a) == fp16.to_f32(a) and FW.PouwF16ToE4m3.evaluate(a) == fp16.to_e4m3(a)
        x = fp16.to_f32(a) ^ (rng.getrandbits(13) if a & 0x7FFF else 0)
        assert FW.PouwF32ToF16Rn.evaluate(x) == fp16.round_to_f16(*fp32.decode(x))
        assert FW.PouwF32Sqrt.evaluate(x & 0x7FFF_FFFF) == fp32.sqrt(x & 0x7FFF_FFFF)
    for code in range(256):
        want = FW.F16_NAN if code & 0x7F == 0x7F else fp16.from_e4m3(code)
        assert FW.PouwF16FromE4m3.evaluate(code) == want
    # total on every encoding: the words #295's functions refuse get one fixed NaN
    assert FW.PouwF16Add.evaluate(0x7C00, 0x3C00) == FW.F16_NAN and FW.PouwF32Sqrt.evaluate(0x7F80_0001) == FW.F32_NAN


def test_the_hoisted_hashes_are_hashlibs_and_the_commitments():
    """SHA-256 and SHAKE256 over byte messages that mix constants, shared words and a row's own bytes, and
    `verity.commitments.merkle`'s framing, on the DAG the circuit is emitted from."""
    rng = random.Random(3)
    g = H.Graph()
    words, row, rid = [g.input(("w", i), True) for i in range(8)], [g.input(("b", i), False) for i in range(37)], g.input("r", False)
    out = H.sha256(g, H.const_bytes(b"abc\0xyz") + H.be_words(words) + [H.Byte8(x) for x in row] + H.u32_bytes(rid, 2))
    wv, bv, r = [rng.getrandbits(32) for _ in range(8)], [rng.getrandbits(8) for _ in range(37)], rng.getrandbits(16)
    env = {("w", i): wv[i] for i in range(8)} | {("b", i): bv[i] for i in range(37)} | {"r": r}
    data = b"abc\0xyz" + b"".join(x.to_bytes(4, "big") for x in wv) + bytes(bv) + r.to_bytes(2, "big")
    assert b"".join(x.to_bytes(4, "big") for x in H.evaluate(g, out, env)) == hashlib.sha256(data).digest()
    s = H.split(g, out)
    assert s.frontier and s.hoisted and not s.direct and all(not n.shared for n in s.local)

    g = H.Graph()
    salt, line = [g.input(("s", i), True) for i in range(32)], g.input("l", False)
    out = H.shake256(g, H.const_bytes(b"tag\0") + H.frame(b"b", [H.Byte8(x) for x in salt]) + H.frame(b"i", H.u32_bytes(line, 1)),
                     300)
    sv = bytes(rng.getrandbits(8) for _ in range(32))
    got = H.evaluate(g, out, {("s", i): sv[i] for i in range(32)} | {"l": 7})
    assert b"".join(x.to_bytes(4, "little") for x in got) == hashlib.shake_256(b"tag\0" + frame(sv) + frame(7)).digest(300)

    g = H.Graph()
    dom, rank = [g.input(("d", i), True) for i in range(8)], g.input("r", False)
    out = H.sha256(g, H.merkle_message(b"pad", H.be_words(dom), H.u32_bytes(rank, 1)))
    dv = bytes(rng.getrandbits(8) for _ in range(32))
    got = H.evaluate(g, out, {("d", i): int.from_bytes(dv[4 * i:4 * i + 4], "big") for i in range(8)} | {"r": 200})
    assert b"".join(x.to_bytes(4, "big") for x in got) == MK._hash(b"pad", dv, bytes([200]))


# ---- the Program against #295 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sums", fp8.SUMS)
def test_the_program_evaluates_to_295s_scheme(sums):
    """Every tile's checked words (I_1 … I_T, R_2 … R_(T−1)) and y from Z = R_T equal `FP8IS(D3s(sums))`'s, bit for bit:
    root_A as the protocol commits it, E₁ and F₁ from it, the D-3s forming, the atoms and the running sums."""
    c = call(16, 96, 16, sums=sums)
    ref = R.evaluate(c)
    y, words = run(c)
    assert words == ref.words and y == ref.out
    assert ref.root == R.root_a(c)


@pytest.mark.slow
def test_the_program_evaluates_to_295s_scheme_on_a_padded_tree_and_long_ids():
    """48 rows pad root_A's tree to 64 leaves (one pad strip, three nodes above the strips); a six-byte index and a
    three-byte weight id lengthen every stream's message."""
    c = call(48, 96, 32, index=(1 << 40) + 5, weight=70_000)
    assert c.lengths == fp8.Lengths(6, 6, 3)
    ref = R.evaluate(c)
    y, words = run(c)
    assert words == ref.words and y == ref.out


def test_the_owners_length_is_its_own():
    """The domain's owner is index + 2 (`CommitmentDomain`), which can take a byte more than the index."""
    c = call(32, 128, 16, index=254, weight=255)
    assert c.lengths == fp8.Lengths(1, 2, 1)
    ref = R.evaluate(c)
    y, words = run(c)
    assert words == ref.words and y == ref.out


def test_the_noise_reads_root_a():
    """Changing one code of one A row moves root_A, and with it every stream: every tile's words change."""
    c = call(16, 96, 16)
    a = [list(r) for r in c.a]
    a[5][7] ^= 1
    d = R.Call(c.shape, tuple(tuple(r) for r in a), c.s, c.b, c.t, c.bias, c.salt, c.index, c.weight, c.sums)
    assert R.root_a(d) != R.root_a(c)
    ref_c, ref_d = R.evaluate(c), R.evaluate(d)
    assert ref_d.x[0] != ref_c.x[0] and ref_d.y[0] != ref_c.y[0]         # row 0's and column 0's codes are unchanged


def test_the_domain_is_295s():
    with pytest.raises(DomainError):
        fp8.linear(fp8.Shape(16, 64, 16))                   # k = 64 fails the d = 32 rejection rule
    with pytest.raises(DomainError):
        fp8.linear(fp8.Shape(24, 96, 16))
    with pytest.raises(DomainError):
        fp8.linear(fp8.Shape(272, 96, 16))                  # rows past 255 take two bytes: not here yet


# ---- the partition --------------------------------------------------------------------------------------------------------

def _check(shape: fp8.Shape):
    c = P.fp8_circuit(shape)
    uc = P.circuit_cut(c)
    return c, uc


def test_the_whole_call_passes_validate_unit_cut():
    """Two strips of each kind: every computing gate certified once, reads across units only through committed values,
    and no value computed in two units, so the call-wide prefixes of the streams and of the tree are each computed once."""
    c, uc = _check(fp8.Shape(32, 96, 32))
    v = uc.verdict
    assert v.ok, v.codes
    roles = [u.role for u in uc.units]
    assert {r: roles.count(r) for r in set(roles)} == {"domain": 1, "digest": 2, "node": 1, "key": 1, "form_y": 2, "form_x": 2,
                                                       "tile": 4, "dequant": 1024}


def test_the_separator_is_z_and_the_width_rule_holds_with_the_pinned_port():
    c, uc = _check(fp8.Shape(32, 96, 32))
    sep = P.circuit_separator(c, uc)
    assert sep.ok and sep.per_tile_bits == 32 * fp8.D * fp8.D and sep.reached_pouw == 0 and sep.words_delivered == 0
    w = P.circuit_width_rule(c, uc=uc)
    assert w.ok and {uc.units[u].role for u in w.checked} == {"dequant"} and w.exempt == len(uc.units) - 1024


@pytest.mark.slow
def test_the_cut_holds_on_a_padded_tree():
    c, uc = _check(fp8.Shape(48, 96, 32))
    assert uc.verdict.ok
    assert P.circuit_separator(c, uc).ok and P.circuit_width_rule(c, uc=uc).ok


def test_delivered_words_fail_the_width_rule():
    """Were the tiles' words served beside y, the tiles would reach a served output other than through Z, and fail."""
    c, uc = _check(fp8.Shape(16, 96, 16))
    w = P.circuit_width_rule(c, served=len(uc.graph.outputs), uc=uc)
    assert not w.ok and {uc.units[u].role for u, _, _ in w.too_wide} >= {"tile"}


def test_without_hoisting_two_rows_recompute_the_streams_prefix():
    """The regression: two X̃ rows whose units each absorb the call-wide prefix (salt, index, root_A, weight id) compute the
    same gates of the same values, which `validate_unit_cut` rejects; hoisted, they do not."""
    k = 96

    def naive_build(g):
        salt, index, root, weight = ([g.input((nm, j), False) for j in range(w)] for nm, w in
                                     (("salt", 32), ("index", 2), ("root", 8), ("weight", 2)))
        line, codes = g.input("line", False), [g.input(("codes", j), False) for j in range(k)]
        return fp8._side(g, codes, fp8._stream(g, "E1", "f16", k, salt, index, root, weight, line, 1, 1), True, "f16"), \
            [*salt, *index, *root, *weight, line, *codes]

    g = H.Graph()
    outs, ins = naive_build(g)

    def naive_body(B, S, salt, index, root, weight, line, codes):
        leaves = [salt[j] for j in range(32)] + [index[j] for j in range(2)] + [root[j] for j in range(8)]
        leaves += [weight[j] for j in range(2)] + [line] + [codes[j] for j in range(k)]
        e = H.Emitter(B, {n.i: c for n, c in zip(ins, leaves)})
        e.emit([n for n in g.nodes if n.kind != "in"])
        return array_of([e.val(o, 8) for o in outs])

    row = bind(CompositeDefinition(
        "TestNaiveXRow", 1, (), lambda S: ((("salt", Array(32, fp8.U8)), ("index", fp8.PAIR), ("root", fp8.DIGEST),
                                            ("weight", fp8.PAIR), ("line", fp8.U32), ("codes", Array(k, fp8.U8))),
                                           Array(3 * k, fp8.U8)), naive_body, doc="test", conformance="protocol-defined"))
    two = bind(CompositeDefinition(
        "TestTwoNaiveRows", 1, (),
        lambda S: ((("salt", Array(32, fp8.U8)), ("index", fp8.PAIR), ("root", fp8.DIGEST), ("weight", fp8.PAIR),
                    ("lines", Array(2, fp8.U32)), ("rows", Array(2, Array(k, fp8.U8)))), Array(2, Array(3 * k, fp8.U8))),
        lambda B, S, salt, index, root, weight, lines, rows: B.batch(row, salt, index, root, weight, lines, rows,
                                                                    axes=(None, None, None, None, 0, 0)),
        doc="test", conformance="protocol-defined"))
    cg = CallGraph(two)
    body = two.body
    owner = [(cg.keep[q] - body.offsets[0]) // row.gates for q in range(cg.n)]
    v = check_cut(cg, owner, limits=None)
    assert not v.ok and "gate-recomputed" in v.codes
    # the shipped circuit's two strips, hoisted: none
    assert _check(fp8.Shape(32, 96, 32))[1].verdict.ok


def test_the_template_query_applies_and_numbers_units_as_core_does():
    from verity.ir import partition_object as PO
    from verity.ir.program import Program

    c = P.fp8_circuit(fp8.Shape(48, 96, 32))
    prog, q = Program(c.fn), P.query_of(c)
    part = PO.evaluate(prog, q)
    units = P.circuit_units(c)
    assert part.population == len(units)
    assert [(tid, lo, n) for _k, tid, lo, n in part.ranges] == [
        (u.template, u.index, sum(1 for v in units if v.node == u.node)) for u in units if u.member == 0]
    assert PO.verify(PO.build(prog, q), prog).ok


# ---- the draw -------------------------------------------------------------------------------------------------------------

def test_a_tiles_closure_adds_the_domain_and_the_pad_cone():
    """`ncp-v2`'s closure (the tile's strips, its strip's digest and path, the key), the domain unit, and every pad strip and
    all-pad node: root_A's tree is padded, and no tile lies beneath a pad."""
    from verity_pouw.circuit import plan as PL

    lay = PL.fp8_layout(fp8.Shape(80, 96, 16))
    us = lay.units
    ups = fp8.upper_nodes(fp8.Shape(80, 96, 16))
    for t, closure in lay.closure.items():
        rs, cs = us[t].where
        roles = [us[u].role for u in closure]
        assert roles[:4] == ["tile", "form_x", "form_y", "digest"] and {"key", "domain"} <= set(roles)
        assert roles.count("pad") == 3                          # strips 5, 6 and 7 of the 128-leaf tree
        nodes = [u for u in closure if us[u].role == "node"]
        assert len(nodes) == 3 + 1                              # its path (levels 4, 5, 6) and the all-pad node (4, 3)
    assert len(ups) == 7
    assert lay.tile_work() == -(-R.scheme().wref(PShape(80, 96, 16)) * fp8.D * fp8.D // (80 * 16))   # its cells' share


def test_traced_as_gives_the_fp8_circuit():
    from verity_pouw.circuit import SCHEMES, traced_as

    assert {"fp8-is-h100-d3s-v0-f16", "fp8-is-h100-d3s-v0"} <= set(SCHEMES)
    t = traced_as("fp8-is-h100-d3s-v0-f16", 32, 96, 32, index=5, weight=300)
    assert t.definition is fp8.linear(fp8.Shape(32, 96, 32), "f16", fp8.Lengths(1, 1, 2)) and t.served == 32 * 32
    assert t.query == P.query_of(P.fp8_circuit(fp8.Shape(32, 96, 32), "f16", fp8.Lengths(1, 1, 2)))
    assert len(t.layout().closure) == 4
    assert traced_as("fp8-is-h100-d3s-v0", 32, 96, 32).definition.S().SUMS == "f32"
