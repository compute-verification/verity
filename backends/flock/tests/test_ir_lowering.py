"""C-Flock's generic IR lowering: the pieces against the IR primitives, the units read off each template's IR graph, the
lowered templates against the IR evaluator on captured #101 slices and synthetic sets, the pins, and the negatives."""
from __future__ import annotations

import os
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
    assert U.in_src[5] == [("leaf", 5), ("leaf", 37), ("leaf", 69), ("leaf", 101)] and U.out_src[5] == [("ret", 5), ("ret", 37)]
    silu = TM.subcircuit("silu-mul", I=8192)
    U = IL.units(silu.definition())
    assert U.n == 8192 and U.in_src[3] == [("leaf", 3), ("leaf", 8195)] and U.out_src[3] == [("ret", 3)]


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
    cuts = IL.cut_words(low, fin) if low.units.cut else None
    got, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << (s.n * low.units.n)) - 1
    bad = (got != want).any(axis=1)
    if cuts is not None:
        bad |= (cut_out != cuts).any(axis=1)
    return int(bad.sum())


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
    sub = TM.subcircuit(template, **LOWERED[template])
    (param,) = LOWERED[template].values()
    low = mod.lowering(sub)
    assert low.sha256 == mod.PINS[param]
    head = low.text.split("\n", 1)[0].split()
    assert head[0] == IL.FORMAT and head[1] == mod.UNIT and int(head[2]) <= 1 << 13
    if template == "silu-mul":
        assert mod.frame_lowering(sub).sha256 == mod.FRAME_PINS[param]


@pytest.mark.parametrize("template", sorted(LOWERED))
def test_pinned_rows_read_earlier_columns(template):
    """What the soundness package's `Rows` needs beyond the parser's checks (`soundness/README.md` §1.5): every computed
    row reads only earlier columns or the constant, and never an input padding column (`A = B = 0`, forced zero)."""
    mod = REG.module(template)
    sub = TM.subcircuit(template, **LOWERED[template])
    lows = [mod.lowering(sub)] + ([mod.frame_lowering(sub)] if template == "silu-mul" else [])
    for low in lows:
        lay = low.layout
        n_in = lay.in_words * IL.WORD
        padding = {c for c in range(n_in) if not lay.rows_a[c] and not lay.rows_b[c]}
        bad = [(r, c) for r in range(n_in, lay.const) for c in lay.rows_a[r] + lay.rows_b[r]
               if not ((c < r and c not in padding) or c == lay.const)]
        assert lay.const + 1 == lay.useful and not bad, (low.name, bad[:5])


def test_the_pin_covers_the_leaf_maps():
    """IR6: the netlist's LEAVES line places every unit on its instance's words (flat input and output leaves, each port's
    [leaves, bits]); another arrangement of the same unit is another pin with the same rows."""
    import json

    mod = REG.module("rope-head")
    a, b = mod.lowering(TM.subcircuit("rope-head", D=64)), mod.lowering(TM.subcircuit("rope-head", D=32))
    line = a.text.splitlines()[-1]
    assert line.startswith("LEAVES ") and json.loads(line.removeprefix("LEAVES ")) == a.leaves
    assert a.leaves["in"][1] == [1, 33, 65, 97] and a.leaves["out"][1] == [1, 33]
    assert a.leaves["in_ports"] == [[64, 16], [64, 16]] and a.leaves["out_ports"] == [[64, 16]]
    assert a.sha256 != b.sha256 and a.text.splitlines()[:-1] == b.text.splitlines()[:-1]


def test_supports_says_what_is_missing():
    for template, params in LOWERED.items():
        sub = TM.subcircuit(template, **params)
        assert REG.supports(sub, "frame-v3/blake3-keyed") is None and REG.relation(sub, "frame-v3/blake3-keyed") == f"irf:{template}"
        assert "not wired yet" in REG.supports(sub, "vllm-v1")
    for template in ("rmsnorm-fused-cuda", "rmsnorm-triton"):
        sub = TM.subcircuit(template, N=2048, EPS=1e-5)
        assert REG.supports(sub, "frame-v3/blake3-keyed") is None and REG.relation(sub, "frame-v3/blake3-keyed") == f"irf:{template}"
        assert "not wired yet" in REG.supports(sub, "vllm-v1")


def test_stage_writes_the_instance_file(tmp_path):
    s = InputSet.open(SLICES / "rope-head-d64")
    path = IL.stage(REG.module("rope-head").lowering(s.subcircuit), s, 0, 2, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    import json

    h = json.loads(head)
    assert h["format"] == IL.INSTANCES_FORMAT and h["units"] == 64 and h["in_words"] == 1 and h["out_words"] == 1
    assert len(body) == 64 * (1 + 1) * 16
    assert (tmp_path / "net.txt").read_text().startswith(IL.FORMAT)
    first = int.from_bytes(body[:16], "little")
    x = IL.flat_ports(s, s.subcircuit.inputs, 0, 1)[0]
    assert first == int(x[0]) | int(x[32]) << 16 | int(x[64]) << 32 | int(x[96]) << 48
    y = IL.flat_ports(s, s.subcircuit.outputs, 0, 1)[0]
    outs = body[64 * 16:]
    assert [int.from_bytes(outs[16 * k:16 * k + 16], "little") for k in range(2)] == [int(y[0]) | int(y[32]) << 16, int(y[1]) | int(y[33]) << 16]


# --- negatives ---------------------------------------------------------------------------------------------------------------

def test_a_flipped_witness_bit_leaves_its_lane_unsatisfied():
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=8))
    x = np.arange(16, dtype=np.uint64)[None, :] * 0x101 + 0x3F00
    first_and = low.layout.in_words * IL.WORD
    for row in (first_and, first_and + 1000, low.layout.useful - 2):
        _, _, ok = low.evaluate(x, flip=(row, 2))
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

    assert all(caught_after(lay.out_port_cols[j] + b, toggle_const=True) for j in range(2) for b in (0, 7, 15))
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


# --- RMSNorm: warps as units, the row tail native ---------------------------------------------------------------------------

SLOW = pytest.mark.skipif(not os.environ.get("FLOCK_IR_SLOW"), reason="builds a 2^19/2^21-row unit: set FLOCK_IR_SLOW=1")
RMS = {"rmsnorm-fused-cuda": "rmsnormfusedcuda-v2-n2048-00e4e4aa", "rmsnorm-triton": "rmsnormtriton-v1-n2048-00e4e4aa"}


def _tail(template, n):
    from collections import Counter

    sub = TM.subcircuit(template, N=n, EPS=1e-5)
    mod = REG.module(template)
    U = IL.units(sub.definition(), IL.reduction_cut(mod.width(n)))
    return U, Counter(g.prim.id for g in U.tail if not g.prim.id.startswith("Const"))


def test_rmsnorm_units_are_warps_with_a_native_tail():
    U, tail = _tail("rmsnorm-fused-cuda", 2048)
    assert U.n == 32 and len(U.cut) == 33
    assert tail == {"F32Add_v1": 32, "F32Div_v1": 1, "RsqrtApprox_v1": 1}
    assert U.in_src[0][-1] == ("cut", 32) and U.out_src[0][-1] == ("cut", 0)
    U, tail = _tail("rmsnorm-triton", 2048)
    assert U.n == 8 and len(U.cut) == 9
    assert tail == {"F32Add_v1": 8, "F32Fma_v1": 1, "MufuSqrtFtz_v1": 1, "DivFullRcp_v1": 2, "DivFullScaleA_v1": 1, "F32Mul_v1": 1}


def test_rmsnorm_widths_refuse_unequal_warps():
    fused, triton = REG.module("rmsnorm-fused-cuda"), REG.module("rmsnorm-triton")
    assert fused.width(2048) == 128 and fused.width(576) == 64 and fused.width(1536) is None
    assert triton.width(2048) == 256 and triton.width(1536) is None
    assert "not N = 1536" in REG.supports(TM.subcircuit("rmsnorm-fused-cuda", N=1536, EPS=1e-5), "frame-v3/blake3-keyed")


def test_rmsnorm_fused_small_row_against_the_ir(tmp_path):
    """N = 64: two warps of 32 one-element threads; spine synthetic rows, outputs and warp aggregates both compared."""
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 6, 20260926, tmp_path)["path"])
    assert REG.module("rmsnorm-fused-cuda").lowering(sub).units.n == 2
    assert _check_set(s) == 0


def test_a_wrong_row_scalar_changes_the_outputs(tmp_path):
    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 7, tmp_path)["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    fin = IL.flat_ports(s, sub.inputs, 0, s.n)
    cuts = IL.cut_words(low, fin)
    good, _, _ = low.evaluate(fin, cuts)
    cuts[:, -1] ^= 1 << 12
    bad, _, _ = low.evaluate(fin, cuts)
    assert (good != bad).any(axis=1).all()


def test_the_netlist_accounts_for_every_cut_word(tmp_path):
    """IR1 (the Python half; the Rust verifier refuses the rest, `flock-ir-block selftest`): the netlist's CUT line lists each
    unit's cut ports and the native tail; every cut word is computed exactly once, by a unit or by the tail, from words defined
    before. The instance header only counts them."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 2, 3, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    head = json.loads(IL.stage(low, s, 0, 2, tmp_path / "st").read_bytes().split(b"\n", 1)[0])
    cut = json.loads((tmp_path / "st" / "net.txt").read_text().splitlines()[-1].removeprefix("CUT "))
    assert cut == low.cut and "cut" not in head
    assert head["cut_words"] == cut["words"] == len(low.units.cut) == 3 and len(cut["in"]) == len(cut["out"]) == low.units.n == 2
    produced = [k for unit in cut["out"] for _, k in unit]
    tail_out = [op[0] for op in cut["tail"] if op[0] < cut["words"]]
    assert sorted(produced + tail_out) == list(range(cut["words"]))
    defined = set(produced)
    for op in cut["tail"]:
        assert set(op[2:] if op[1] != "Const" else []) <= defined
        defined.add(op[0])
    assert {op[1] for op in cut["tail"]} <= {"Const", *IL.TAIL_PRIMS}
    rs = InputSet.open(SLICES / "rope-head-d64")
    rope = REG.module("rope-head").lowering(rs.subcircuit)
    head = json.loads(IL.stage(rope, rs, 0, 1, tmp_path / "r").read_bytes().split(b"\n", 1)[0])
    assert rope.cut is None and head["cut_words"] == 0 and not rope.text.splitlines()[-1].startswith("CUT")


def test_the_pin_covers_the_cut_structure():
    """IR4: N and eps are tail constants in the netlist's CUT line, so another eps is another pin; the unit rows do not
    depend on eps (the row scalar is a cut word)."""
    import struct

    f32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731
    mod = REG.module("rmsnorm-fused-cuda")
    a = mod.lowering(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5))
    b = mod.lowering(TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-3))
    consts = {op[2] for op in a.cut["tail"] if op[1] == "Const"}
    assert {f32(1e-5), f32(64.0)} <= consts and f32(1e-3) not in consts
    assert a.sha256 != b.sha256 and a.text.splitlines()[1:-1] == b.text.splitlines()[1:-1]
    assert a.text.splitlines()[-1] != b.text.splitlines()[-1]


def test_a_producers_file_is_not_the_verifiers_own(tmp_path):
    """IR2: the verifier stages its own file; a producer's file with one output word changed differs from it."""
    s = InputSet.open(SLICES / "rope-head-d64")
    low = REG.module("rope-head").lowering(s.subcircuit)
    path = IL.stage(low, s, 0, 2, tmp_path)
    assert IL.check_staged(low, s, path) == []
    b = bytearray(path.read_bytes())
    b[-3] ^= 1
    path.write_bytes(bytes(b))
    assert IL.check_staged(low, s, path) == [f"body differs from the verifier's own staging of {s.name} [0, 2)"]


def test_the_tables_are_the_pinned_ones(tmp_path):
    assert IL.write_tables(tmp_path) == IL.TABLES
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ex2.u32", "rcp.u32", "rsq.u32", "sqrt.u32"]


@SLOW
@pytest.mark.parametrize("template", sorted(RMS))
def test_rmsnorm_captured_slices_and_pins(template):
    s = InputSet.open(SLICES / RMS[template])
    assert _check_set(s) == 0
    p = s.subcircuit.params
    assert REG.module(template).lowering(s.subcircuit).sha256 == REG.module(template).PINS[(int(p["N"]), float(p["EPS"]))]


#: The served workloads' other parameters (#73 Qwen3-4B on the H100, #60 Mistral-7B on the L40S): the same units read off the
#: IR, so RoPE and SiLU·mul keep the granted rows and only their leaf maps (the LEAVES line) differ; the RMSNorm warps widen.
WORKLOAD = [("rope-head", {"D": 128}), ("silu-mul", {"I": 9728}), ("silu-mul", {"I": 14336})]
WORKLOAD_RMS = [("rmsnorm-fused-cuda", {"N": 4096, "EPS": 1e-05}), ("rmsnorm-triton", {"N": 4096, "EPS": 1e-05}),
                ("rmsnorm-triton", {"N": 128, "EPS": 1e-06})]


@pytest.mark.parametrize("template,params", WORKLOAD)
def test_workload_pins_keep_the_granted_rows(template, params):
    mod = REG.module(template)
    (param,) = params.values()
    low, granted = mod.lowering(TM.subcircuit(template, **params)), mod.lowering(TM.subcircuit(template, **LOWERED[template]))
    assert low.sha256 == mod.PINS[param]
    if template == "silu-mul":
        assert mod.frame_lowering(TM.subcircuit(template, **params)).sha256 == mod.FRAME_PINS[param]
    rows = lambda t: [line for line in t.splitlines() if not line.startswith("LEAVES ")]  # noqa: E731
    assert rows(low.text) == rows(granted.text) and low.sha256 != granted.sha256


@SLOW
@pytest.mark.parametrize("template,params", WORKLOAD_RMS)
def test_workload_rmsnorm_pins(template, params):
    mod = REG.module(template)
    assert mod.lowering(TM.subcircuit(template, **params)).sha256 == mod.PINS[(params["N"], params["EPS"])]


def test_rmsnorm_warp_widths():
    """Identical warp units exist when every warp holds the same columns: Triton N a multiple of 1024 (8 per lane per chunk) or a
    power of two under 1024 (N/128 per lane); fused CUDA when min(N, 1024) threads divide N. N = 2560 is neither."""
    tr, fu = REG.module("rmsnorm-triton"), REG.module("rmsnorm-fused-cuda")
    assert [tr.width(n) for n in (96, 128, 256, 512, 1024, 2048, 2560, 4096)] == [None, 32, 64, 128, 256, 256, None, 256]
    assert (fu.width(4096), fu.width(2560)) == (256, None)
    assert REG.supports(TM.subcircuit("rmsnorm-triton", N=128, EPS=1e-06), "frame-v3/blake3-keyed") is None
    for template in ("rmsnorm-triton", "rmsnorm-fused-cuda"):
        assert "not N = 2560" in REG.supports(TM.subcircuit(template, N=2560, EPS=1e-06), "frame-v3/blake3-keyed")


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


# --- verity/flock-ir-frame/v3: the frame-v3 binding --------------------------------------------------------------------------

def test_frame_plans_take_the_largest_run_that_fits():
    """RoPE and SiLU·mul read whole chunks; an RMSNorm block holds runs of its chunks' compressions (fused: 4 of 16, its four
    warps' elements; Triton: 8 of 16, one warp's), within 2^22 bits."""
    from verity_flock import ir_frame as F

    rope = TM.subcircuit("rope-head", D=64)
    pl = F.plan(REG.module("rope-head").frame_lowering(rope), rope)
    assert (pl.G, pl.nb, pl.chunk_nb, pl.comp_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (2, 2, 2, 8, 64, 64, 20)
    assert pl.wiring[0] == [(0, 0), (0, 64), (1, 0), (1, 64)]
    silu = TM.subcircuit("silu-mul", I=8192)
    low = REG.module("silu-mul").frame_lowering(silu)
    pl = F.plan(low, silu)
    assert low.units.n == 4096 and (pl.G, pl.nb, pl.chunk_nb, pl.comp_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (1, 16, 16, 32, 256, 256, 22)
    assert pl.comps[3][0] == [(0, 3, 0), (0, 19, 0)]
    fused = TM.subcircuit("rmsnorm-fused-cuda", N=2048, EPS=1e-5)
    pl = F.plan(REG.module("rmsnorm-fused-cuda").lowering(fused), fused)
    assert (pl.nb, pl.chunk_nb, pl.G, pl.chunk_slots, pl.chunks_per_comp, pl.units_per_block, pl.k_log) == (4, 16, 1, 8, 6, 4, 22)
    assert pl.comps[1] == ([(p, c, 1) for p in range(3) for c in (0, 2)], [4, 5, 6, 7])
    triton = TM.subcircuit("rmsnorm-triton", N=2048, EPS=1e-5)
    pl = F.plan(REG.module("rmsnorm-triton").lowering(triton), triton)
    assert (pl.nb, pl.chunk_nb, pl.G, pl.chunk_slots, pl.units_per_block, pl.unit_pos0, pl.k_log) == (8, 16, 1, 2, 1, 1, 22)


def test_frame_staging_commits_rows_and_outputs(tmp_path):
    import json

    import blake3
    from verity.commitments.rowleaf import ROLE_X, blake3_row_key

    s = InputSet.open(SLICES / "rope-head-d64")
    path = REG.module("rope-head").stage(s, 0, 1, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["format"] == "flock-ir-frame/v3" and h["scheme"] == "frame-v3/blake3-keyed" and h["units"] == 32
    assert sorted(h["frame_v3"]["roots"]) == ["cs", "out", "x"] and h["frame_v3"]["schemas"]["x"] == "blake3-keyed/row/v2"
    x = IL.flat_ports(s, s.subcircuit.inputs, 0, 1)
    row0 = x[0, :64].astype("<u2").tobytes()
    assert body[:128] == row0
    assert body[256:256 + 32] == blake3.blake3(row0, key=blake3_row_key(ROLE_X)).digest()
    assert h["blocks"] == [[[[0, 0, 0, 0], [0, 1, 0, 0], None, None], list(range(32)) + [None] * 32]]


def test_frame_staging_carries_the_cut_words(tmp_path):
    """A cut template's frame file ends with every instance's cut words (the IR's), which the verifier checks against the
    netlist's pinned tail; the empty run slot of a three-port component block is padding."""
    import json

    sub = TM.subcircuit("rmsnorm-fused-cuda", N=64, EPS=1e-5)
    s = InputSet.open(generate(sub, 3, 5, tmp_path / "set")["path"])
    low = REG.module("rmsnorm-fused-cuda").lowering(sub)
    path = REG.module("rmsnorm-fused-cuda").stage(s, 0, 3, tmp_path / "st")
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["cut_words"] == 3 and "cut" not in h and h["layout"]["chunks_per_block"] == 4
    assert h["blocks"][0][0] == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], None]
    cuts = np.frombuffer(body[-3 * 3 * 4:], dtype="<u4").reshape(3, 3)
    assert (cuts == IL.cut_words(low, IL.flat_ports(s, sub.inputs, 0, 3))).all()


def test_non_gemm_results_carry_no_gemm_target():
    from verity_numerical.bench import contract

    base = dict(domain="finite", B=4, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
                security={"target": -128}, hardware={"gpu": None, "cpu": {"model": "x"}}, software={"backend": {"name": "b", "commit": None},
                "toolchain": {}}, preprocessing="none", retention="retained", instances=None)
    ok = contract.fingerprint(profile=None, K=None, subcircuit="rope-head/d64/neox-bf16", **base)
    assert contract.validate_fingerprint(ok) == []
    assert any("profile must be null" in p for p in contract.validate_fingerprint({**ok, "profile": contract.PROFILE}))
    assert any("profile must be a target" in p for p in contract.validate_fingerprint({**ok, "subcircuit": None}))
    rates = [contract.measurement(contract.OVERHEAD, 1.0)]
    assert contract._check_rates(ok, rates) and "GEMM rates" in contract._check_rates(ok, rates)[0]


def test_an_ir_cell_plans_through_the_spine():
    from verity_numerical.bench import cell as C

    two_machines = {"prover": {"machine_id": "m-p", "public_ip": "1.2.3.5"}, "verifier": {"machine_id": "m-v", "public_ip": "1.2.3.4"},
                    "link": {"peer": "1.2.3.4:7400", "peer_ip": "1.2.3.4"}}
    c = C.plan("C-interactive", "rope-head/d64/neox-bf16+frame-v3/blake3-keyed", str(SLICES / "rope-head-d64"), "vy-p", verifier="vy-v",
               verifier_addr="1.2.3.4:7400", rtt_target="1.2.3.4:22", lane="flock-ir-lowering", points="2 4", runs=5, placement=two_machines)
    assert c.relation == "irf:rope-head"
    v, p = c.jobs
    assert v["command"][:3] == ["bash", "backends/flock/pod/33-ir-cell.sh", "ROLE=verifier"] and "TEMPLATE=rope-head" in v["command"]
    assert "--send" in v["argv"] and v["argv"][v["argv"].index("--send") + 1] == p["argv"][p["argv"].index("--send") + 1]
    assert "VERIFIER=1.2.3.4:7400" in p["command"] and "LOOPBACK=1" in p["command"]


# --- attention: tensor-core k-steps as units, the softmax native -------------------------------------------------------------

def test_the_tensor_core_step_matches_the_ir_on_every_encoding():
    """``fp.tc_dot16`` = ``AmpereBF16TcDot16`` (the total model): attention-range words, full-range words (overflowing groups),
    NaN / inf / inf x 0 operands and accumulators, bit for bit."""
    rng = random.Random(11)
    specials16 = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 1, 0x8001, 0x7F7F, 0xFF7F]
    specials32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0xFF7FFFFF, 1]

    def bf(att):
        if att:
            return (rng.getrandbits(1) << 15) | (rng.randint(110, 135) << 7) | rng.getrandbits(7)
        r = rng.random()
        return rng.choice(specials16) if r < 0.05 else (rng.getrandbits(1) << 15) | (rng.randint(200, 254) << 7) | rng.getrandbits(7) \
            if r < 0.3 else rng.getrandbits(16)

    vecs = []
    for i in range(1500):
        att = i % 3 == 0
        acc = (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.getrandbits(23) if att else \
            rng.choice(specials32) if rng.random() < 0.1 else rng.getrandbits(32)
        vecs.append([acc] + [bf(att) for _ in range(32)])
    got = run_piece(lambda C, acc, *ab: IL.PIECES["AmpereBF16TcDot16_v1"](C, acc, *ab), [32] + [16] * 32, 32, vecs)
    assert [(v, g) for v, g in zip(vecs, got) if g != P.AmpereBF16TcDot16.evaluate(*v)] == []


def test_attention_units_are_tensor_core_steps():
    """Every AmpereBF16TcDot16 gate is a unit (b from K/V leaves or a zero word where the IR pads a masked key; a, acc and the
    result cut words; the query public), the rest is the tail, which returns the output; on the captured T = 129 head the laid-out
    units reproduce every tensor-core value of the IR."""
    s = InputSet.open(SLICES / "attention-head-fa2-d64-bn128")
    sub = s.subcircuit
    (T,) = {len(x) // 64 for x in s.port("k")}
    low = IL.lowering(sub.definition(T), f"attention-head/tc-step/t{T}", unit_fn=IL.tc_units)
    U = low.units
    assert (T, U.n, len(U.public), low.layout.useful) == (129, 4 * 129 + 64 * 9, 64, 9857)
    assert sum(k == "zero" for u in U.in_src for k, _ in u) == 64 * 15
    assert low.leaves["out"] == [[]] * U.n and low.cut["outputs"] and len(low.cut["outputs"]) == 64
    fin = np.stack([np.concatenate([s.port("q")[i], s.port("k")[i], s.port("v")[i]]).astype(np.uint64) for i in range(s.n)])
    cuts = IL.cut_words(low, fin)
    _res, cut_out, ok = low.evaluate(fin, cuts)
    assert ok == (1 << U.n) - 1 and (cut_out == cuts).all()


def test_attention_frame_places_a_block_per_16_keys(tmp_path):
    """A 16-key group per block: run slots [K keys | V keys] (one key = one run of two BLAKE3 blocks), unit slots [QK steps | PV
    columns]; keys past T are empty run slots, their QK steps empty unit slots, and a PV step's masked port reads the empty
    slot; the query is a public port and the tail computes the outputs, which the file carries."""
    import json

    from verity_flock import ir_frame as F

    s = InputSet.open(SLICES / "attention-head-fa2-d64-bn128")
    mod = REG.module("attention-head")
    assert mod.key_count(s) == 129
    low = mod.lowering_for_set(s)
    pl = F.plan_tc(low, s.subcircuit)
    assert (pl.nb, pl.chunk_slots, pl.units_per_block, pl.unit_pos0, pl.k_log, pl.public_ports, len(pl.comps)) == (2, 32, 128, 128, 22, [0], 9)
    runs, units = pl.comps[8]
    assert runs[0] == (1, 16, 0) and runs[1:16] == [None] * 15 and runs[16] == (2, 16, 0) and units[4:64] == [None] * 60
    assert pl.wiring[5] == [(1, 32 + 2 * t) for t in range(16)] and pl.wiring[64 + 7] == [(16 + t, 14) for t in range(16)]
    path = mod.stage(s, 0, 1, tmp_path)
    head, body = path.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    assert h["public_ports"] == [0] and h["outputs"] == "tail" and h["layout"]["ret_group"] is None and len(h["blocks"]) == 9
    out = np.frombuffer(body[-4 * h["cut_words"] - 128:-4 * h["cut_words"]], dtype="<u2")
    assert (out == s.port("out")[0]).all()


def test_attention_key_classes_and_the_class_manifest(tmp_path):
    """Key-count classes [1, 128], [129, 256], [257, 512], ...; a class manifest lists every T of its range with its v3 netlist's
    sha256 and the unit rows they share (the pin is the manifest's sha256); a class cell's sub-batches split at T boundaries."""
    import hashlib
    import json

    from verity_flock import ir_bench
    from verity_flock.templates import attention_head as A

    assert [A.key_class(t) for t in (1, 128, 129, 256, 257, 287, 512, 513)] == [(1, 128)] * 2 + [(129, 256)] * 2 + [(257, 512)] * 3 + [(513, 1024)]
    sub = TM.subcircuit("attention-head", D=64, BN=128)
    m = A.class_manifest(sub, 1, 3)
    d = json.loads(m)
    assert d["format"] == A.CLASS_FORMAT and d["T"] == [1, 3] and sorted(d["nets"]) == ["1", "2", "3"]
    assert m == (json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n").encode()
    for T in (1, 2, 3):
        low = A.cached_lowering(sub, T)
        assert d["nets"][str(T)] == low.sha256 == hashlib.sha256(low.text.encode()).hexdigest()
        assert A.unit_rows_sha256(low.text) == d["unit_rows"]

    class _K(ir_bench.KeyClass):
        def __init__(self, T):
            self.T = T

    assert _K([1, 1, 1, 2, 2, 5]).subs(6, 2) == [(0, 2), (2, 3), (3, 5), (5, 6)]
    assert _K([1, 1, 1, 2, 2, 5]).subs(4, 16) == [(0, 3), (3, 4)]


def test_an_ir_run_result_is_a_point_not_a_second_cell():
    """The runner publishes a run's result.json under its own kind: ir_bench marks it a bench-point/v1 (a known kind), so the
    cell's one bench-result/v1 is bench.cell register's, and verity_flock.register drops the kind it read."""
    from pathlib import Path

    from research.store.kinds import KINDS
    from verity_flock import ir_bench, register

    assert ir_bench.POINT_KIND == "bench-point/v1" and ir_bench.POINT_KIND in KINDS
    assert 'raw.pop("kind", None)' in Path(register.__file__).read_text()


def test_hopper_wgmma_piece_is_the_ir_primitive():
    """``PIECES["HopperBF16WgmmaDot16_v1"]`` (fp.tc_dot16 with HOPPER_BF16_WGMMA_K16's groups (16,), W 26, F -133) equals the IR
    primitive on special-heavy vectors (NaN payloads, both infinities, subnormals, cancellation)."""
    import random

    from verity.ml.prims import HopperBF16WgmmaDot16 as P

    rng = random.Random(7)
    spec16 = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x3F80]
    spec32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x7F7FFFFF, 0x3F800000]
    w16 = lambda: rng.choice(spec16) if rng.random() < 0.2 else rng.getrandbits(16)  # noqa: E731
    cases = []
    for _ in range(300):
        a, b = [w16() for _ in range(16)], [w16() for _ in range(16)]
        if rng.random() < 0.3:
            for j in range(0, 16, 2):
                a[j + 1], b[j + 1] = a[j] ^ 0x8000, b[j]
        cases.append((rng.choice(spec32) if rng.random() < 0.3 else rng.getrandbits(32), a, b))
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(32)] + [C.input("acc", 32)]
    C.output("d", IL.PIECES["HopperBF16WgmmaDot16_v1"](C, ins[32], *ins[:32]))
    vals = {f"in{j}": [sum((((c[1] + c[2])[j] >> i) & 1) << t for t, c in enumerate(cases)) for i in range(16)] for j in range(32)}
    vals["acc"] = [sum(((c[0] >> i) & 1) << t for t, c in enumerate(cases)) for i in range(32)]
    outs, _ = C.evaluate(vals, len(cases))
    got = [sum(((outs["d"][i] >> t) & 1) << i for i in range(32)) for t in range(len(cases))]
    assert got == [P.evaluate(c[0], *c[1], *c[2]) & 0xFFFFFFFF for c in cases]
