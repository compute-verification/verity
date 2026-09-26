"""C-Flock's generic IR lowering: the pieces against the IR primitives, the units read off each template's IR graph, the
lowered templates against the IR evaluator on captured #101 slices and synthetic sets, the pins, and the negatives."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from verity_flock import fp, gf2
from verity_flock import ir_lower as IL

P = pytest.importorskip("verity_vllm.program.registry.prims")
from verity_numerical.bench import lowerings  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402

SLICES = Path(__file__).resolve().parents[2] / "numerical" / "tests" / "bench" / "data" / "captured-101" / "slices"
REG = lowerings.registry("C-Flock")
LOWERED = {"rope-head": {"D": 64}, "silu-mul": {"I": 8192}}


def _isnan(w: int, bits: int = 32) -> bool:
    e, m = (0xFF, 23) if bits == 32 else (0xFF, 7)
    return (w >> m) & e == e and w & ((1 << m) - 1) != 0


def f32_words(rng: random.Random, n: int) -> list[int]:
    special = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x80000001, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x3F800000, 0xBF800000]
    out = []
    for _ in range(n):
        r, s = rng.random(), rng.getrandbits(1) << 31
        if r < 0.15:
            out.append(rng.choice(special))
        elif r < 0.3:
            out.append(s | rng.getrandbits(23))
        elif r < 0.45:
            out.append(s | (rng.randint(1, 30) << 23) | rng.getrandbits(23))
        elif r < 0.6:
            out.append(s | (rng.randint(220, 254) << 23) | rng.getrandbits(23))
        elif r < 0.8:
            out.append(s | (rng.randint(120, 134) << 23) | rng.getrandbits(23))
        else:
            out.append(rng.getrandbits(32))
    return out


def near(rng: random.Random, a: int) -> int:
    """An operand within two binades of a, of either sign, so additions cancel."""
    e = min(254, max(0, ((a >> 23) & 0xFF) + rng.randint(-2, 2)))
    return ((a ^ (rng.getrandbits(1) << 31)) & 0x807FFFFF & ~(0xFF << 23)) | (e << 23)


def bf16_words(rng: random.Random, n: int) -> list[int]:
    special = [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0x3F80, 0x7F7F, 0x0080]
    return [rng.choice(special) if rng.random() < 0.1 else
            (rng.getrandbits(1) << 15) | (rng.randint(100, 150) << 7) | rng.getrandbits(7) if rng.random() < 0.5 else rng.getrandbits(16)
            for _ in range(n)]


def run_piece(piece, widths, out_width, vectors):
    C = gf2.Circuit()
    ins = [C.input(f"i{k}", w) for k, w in enumerate(widths)]
    C.output("o", piece(C, *ins))
    T = len(vectors)
    outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vectors], w) for k, w in enumerate(widths)}, T)
    return [int(x) for x in IL.unslice_words(outs["o"], T)]


PIECES = [
    ("F32Add_v1", (32, 32), 32, lambda C, a, b: IL.PIECES["F32Add_v1"](C, a, b), P.F32Add.evaluate),
    ("F32Mul_v1", (32, 32), 32, lambda C, a, b: IL.PIECES["F32Mul_v1"](C, a, b), P.F32Mul.evaluate),
    ("F32Fma_v1", (32, 32, 32), 32, lambda C, a, b, c: IL.PIECES["F32Fma_v1"](C, a, b, c), P.F32Fma.evaluate),
    ("F2fpBf16_v1", (32,), 16, lambda C, a: IL.PIECES["F2fpBf16_v1"](C, a), P.F2fpBf16.evaluate),
    ("F32ToBf16Rn_v1", (32,), 16, lambda C, a: IL.PIECES["F32ToBf16Rn_v1"](C, a), P.F32ToBf16Rn.evaluate),
]


@pytest.mark.parametrize("name,widths,out,piece,ref", PIECES, ids=[p[0] for p in PIECES])
def test_pieces_match_the_ir_primitives(name, widths, out, piece, ref):
    rng = random.Random(20260926)
    vecs = []
    for _ in range(6000):
        v = f32_words(rng, len(widths))
        if len(v) >= 2 and rng.random() < 0.3:
            v[1] = near(rng, v[0])
        if len(v) == 3 and rng.random() < 0.3:
            v[2] = near(rng, P.f32_mul_bits(v[0], v[1]))
        vecs.append(v)
    got = run_piece(piece, widths, out, vecs)
    bad = [(v, g) for v, g in zip(vecs, got) if not (g == ref(*v) or (_isnan(g, out) and _isnan(ref(*v), out)))]
    assert bad == []


@pytest.mark.parametrize("prim", ["Bf16Add_v1", "Bf16AddF2fp_v1"])
def test_bf16_adds_match_the_ir_primitives(prim):
    rng = random.Random(7)
    vecs = [bf16_words(rng, 2) for _ in range(6000)]
    ref = {"Bf16Add_v1": P.Bf16Add, "Bf16AddF2fp_v1": P.Bf16AddF2fp}[prim].evaluate
    assert run_piece(IL.PIECES[prim], (16, 16), 16, vecs) == [ref(*v) for v in vecs]


def test_template_units_are_read_off_the_ir():
    rope = TM.subcircuit("rope-head", D=64)
    U = IL.units(rope.definition())
    assert U.n == 32 and U.in_widths == [16] * 4 and U.out_widths == [16, 16]
    assert U.inputs[5] == [5, 37, 69, 101] and U.outputs[5] == [5, 37]
    silu = TM.subcircuit("silu-mul", I=8192)
    U = IL.units(silu.definition())
    assert U.n == 8192 and U.inputs[3] == [3, 8195] and U.outputs[3] == [3]


def test_rope_unit_matches_its_primitives():
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    rng = random.Random(3)
    vecs = [bf16_words(rng, 4) for _ in range(8000)]
    (lo, hi), ok = IL.evaluate_layout(low.layout, [[v[j] for v in vecs] for j in range(4)], len(vecs))
    assert ok == (1 << len(vecs)) - 1
    assert [int(x) for x in lo] == [P.RopeOut.evaluate(*v) for v in vecs]
    assert [int(x) for x in hi] == [P.RopeOutAdd.evaluate(*v) for v in vecs]


def test_silu_unit_covers_the_whole_table():
    """u = 1.0 makes the unit's output its silu table entry, so every gate word g is checked; then random (g, u)."""
    low = REG.module("silu-mul").lowering(TM.subcircuit("silu-mul", I=4))
    g = list(range(1 << 16))
    (out,), ok = IL.evaluate_layout(low.layout, [g, [0x3F80] * len(g)], len(g))
    assert ok == (1 << len(g)) - 1
    assert [int(x) for x in out] == [P.SiluMulBf16.evaluate(w, 0x3F80) for w in g]
    rng = random.Random(5)
    vecs = [bf16_words(rng, 2) for _ in range(8000)]
    (out,), _ = IL.evaluate_layout(low.layout, [[v[0] for v in vecs], [v[1] for v in vecs]], len(vecs))
    assert [int(x) for x in out] == [P.SiluMulBf16.evaluate(*v) for v in vecs]


def _check_set(s: InputSet) -> int:
    sub = s.subcircuit
    low = REG.module(sub.template.id).lowering(sub)
    fin, want = IL.flat_ports(s, sub.inputs, 0, s.n), IL.flat_ports(s, sub.outputs, 0, s.n)
    got, ok = low.evaluate(fin)
    assert ok == (1 << (s.n * low.units.n)) - 1
    return int((got != want).any(axis=1).sum())


@pytest.mark.parametrize("name", ["rope-head-d64", "silumul-v1-i8192"])
def test_captured_slices(name):
    s = InputSet.open(SLICES / name)
    assert s.source == "captured" and s.n > 0
    assert _check_set(s) == 0


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_synthetic_sets(template, tmp_path):
    sub = TM.subcircuit(template, **LOWERED[template])
    n = 24 if template == "rope-head" else 3
    summary = generate(sub, n, 20260926, tmp_path)
    s = InputSet.open(summary["path"])
    assert s.source == "synthetic"
    assert _check_set(s) == 0


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_lowering_is_the_pinned_one(template):
    mod = REG.module(template)
    low = mod.lowering(TM.subcircuit(template, **LOWERED[template]))
    assert low.sha256 == mod.PIN
    head = low.text.split("\n", 1)[0].split()
    assert head[0] == IL.FORMAT and head[1] == mod.UNIT and int(head[2]) <= 1 << 13


def test_supports_says_what_is_missing():
    for template, params in LOWERED.items():
        why = REG.supports(TM.subcircuit(template, **params), "frame-v3/blake3-keyed")
        assert "C-Flock lowers" in why and "not wired yet" in why


def test_stage_writes_the_instance_file(tmp_path):
    s = InputSet.open(SLICES / "rope-head-d64")
    path = REG.module("rope-head").stage(s, 0, 2, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    import json

    h = json.loads(head)
    assert h["format"] == IL.INSTANCES_FORMAT and h["units"] == 64 and h["in_words"] == 1 and h["out_words"] == 2
    assert len(body) == 64 * (1 + 2) * 16
    assert (tmp_path / "net.txt").read_text().startswith(IL.FORMAT)
    first = int.from_bytes(body[:16], "little")
    x = IL.flat_ports(s, s.subcircuit.inputs, 0, 1)[0]
    assert first == int(x[0]) | int(x[32]) << 16 | int(x[64]) << 32 | int(x[96]) << 48
    y = IL.flat_ports(s, s.subcircuit.outputs, 0, 1)[0]
    outs = body[64 * 16:]
    assert [int.from_bytes(outs[16 * k:16 * k + 16], "little") for k in range(4)] == [int(y[0]), int(y[32]), int(y[1]), int(y[33])]


# --- negatives ---------------------------------------------------------------------------------------------------------------

def test_a_flipped_witness_bit_leaves_its_lane_unsatisfied():
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    x = np.arange(16, dtype=np.uint64)[None, :] * 0x101 + 0x3F00
    first_and = low.layout.in_words * IL.WORD
    for row in (first_and, first_and + 1000, low.layout.useful - 2):
        _, ok = low.evaluate(x, flip=(row, 2))
        assert ok == ((1 << 4) - 1) & ~(1 << 2)


def test_a_mutated_netlist_is_caught_by_the_ir_comparison():
    """Negating one output row, or dropping one term of one AND row, changes the unit's function on the stratified words. A
    few AND rows sit in logic this composition cannot reach (RoPE's fma addend is RN(y*s), never more than 16 significant
    bits; the pieces' NaN is already the torch word, so the cast's NaN select is a no-op), so mutating them is not
    observable by any input; the sample bound allows for them."""
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    rng = random.Random(11)
    vecs = [bf16_words(rng, 4) for _ in range(4000)]
    ports = [[v[j] for v in vecs] for j in range(4)]
    want = ([P.RopeOut.evaluate(*v) for v in vecs], [P.RopeOutAdd.evaluate(*v) for v in vecs])
    lay = low.layout
    first_and = lay.in_words * IL.WORD

    def caught_after(r, toggle_const=False):
        saved = lay.rows_a[r]
        if toggle_const:
            lay.rows_a[r] = saved[:-1] if saved[-1] == lay.const else saved + [lay.const]
        else:
            lay.rows_a[r] = saved[:-1] if len(saved) > 1 else saved + [lay.const]
        (lo, hi), _ = IL.evaluate_layout(lay, ports, len(vecs))
        lay.rows_a[r] = saved
        return [int(x) for x in lo] != want[0] or [int(x) for x in hi] != want[1]

    assert all(caught_after(lay.out_cols[j] * IL.WORD + b, toggle_const=True) for j in range(2) for b in (0, 7, 15))
    assert sum(caught_after(r) for r in rng.sample(range(first_and, first_and + lay.counts["ands"]), 12)) >= 8


def test_a_wrong_table_entry_is_caught(monkeypatch):
    table = list(IL.silu_table())
    table[0x3F90] ^= 1
    monkeypatch.setattr(IL, "silu_table", lambda: tuple(table))
    low = IL.lowering(TM.subcircuit("silu-mul", I=4).definition(), "tampered")
    (out,), _ = IL.evaluate_layout(low.layout, [[0x3F90, 0x3F91], [0x3F80, 0x3F80]], 2)
    assert int(out[0]) != P.SiluMulBf16.evaluate(0x3F90, 0x3F80)
    assert int(out[1]) == P.SiluMulBf16.evaluate(0x3F91, 0x3F80)


def test_a_reduction_is_one_component():
    assert IL.units(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5).definition()).n == 1


def test_units_refuse_components_of_different_shapes():
    from verity.ir.defs import CompositeDefinition, bind
    from verity.ir.refs import array_of
    from verity.ir.types import Array, Value

    mixed = CompositeDefinition("FlockLoweringMixed", 1, (), lambda S: ((("x", Array(4, Value(32))),), Array(2, Value(32))),
                                lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Mul, x[2], x[3])]),
                                register=False)
    with pytest.raises(ValueError, match="not all the same unit"):
        IL.units(bind(mixed))


def test_a_primitive_without_a_piece_is_named():
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.types import Value

    odd = PrimitiveDefinition("FlockLoweringTestOnly", 1, (("a", Value(8)),), Value(8), lambda a: a ^ 0x55, register=False)
    C = gf2.Circuit()
    with pytest.raises(IL.NoPiece, match="FlockLoweringTestOnly_v1"):
        IL.lower(C, odd, [C.input("a", 8)])


def test_piece_counts_are_what_the_design_says():
    """Constant propagation specializes the general pieces: a bf16 x bf16 product needs far fewer ANDs than f32 x f32."""
    def count(f, widths):
        C = gf2.Circuit()
        C.output("o", f(C, *[C.input(f"i{k}", w) for k, w in enumerate(widths)]))
        return C.counts()["ands"]

    general = count(fp.f32_mul, (32, 32))
    narrow = count(lambda C, a, b: fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), (16, 16))
    assert narrow * 3 < general
