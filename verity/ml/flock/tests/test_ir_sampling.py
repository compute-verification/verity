"""C-Flock's lowering of GumbelTopPTokenSelect_v1 (``verity_flock.ir_sampling``): the pieces against the IR primitives, the lane
unit against the IR composites it is lowered from, a row's words against the IR evaluator on synthetic sets (a row whose last
BLAKE3 chunk is short included), the staged file's units, the native words, the pin, and the negatives."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import ir_sampling as S

P = pytest.importorskip("verity.ml.definitions.registry.prims")
from verity.ml.circuits.evaluation import evaluate  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import fp32 as F  # noqa: E402
from verity.ml.circuits.ops.scalar import SelectF32  # noqa: E402
from verity_numerical.bench import lowerings  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity.ml.definitions.registry import ref_prims as RP  # noqa: E402
from verity.ml.definitions.registry import sampling as SM  # noqa: E402

REG = lowerings.registry("C-Flock")
SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00123, 0x7F800001, 0xFFC00000, 0xFF800007, 1, 0x80000001,
           0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x3F800000, 0xBF800000, 0x3F4CCCCD, 0x7E800000, 0x7E800001, 0x00800001]


def f32_words(rng: random.Random, n: int) -> list[int]:
    out = []
    for _ in range(n):
        r, s = rng.random(), rng.getrandbits(1) << 31
        if r < 0.2:
            out.append(rng.choice(SPECIAL))
        elif r < 0.5:
            out.append(s | (rng.randint(100, 150) << 23) | rng.getrandbits(23))
        elif r < 0.6:
            out.append(s | rng.getrandbits(23))
        elif r < 0.7:
            out.append(s | (rng.randint(240, 254) << 23) | rng.getrandbits(23))
        else:
            out.append(rng.getrandbits(32))
    return out


def bf16_words(rng: random.Random, n: int) -> list[int]:
    special = [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3F80, 0x7F7F, 0x0080]
    return [rng.choice(special) if rng.random() < 0.15 else
            (rng.getrandbits(1) << 15) | (rng.randint(110, 140) << 7) | rng.getrandbits(7) if rng.random() < 0.6 else rng.getrandbits(16)
            for _ in range(n)]


def run_piece(piece, widths, vectors):
    C = gf2.Circuit()
    ins = [C.input(f"i{k}", w) for k, w in enumerate(widths)]
    C.output("o", piece(C, *ins))
    T = len(vectors)
    outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vectors], w) for k, w in enumerate(widths)}, T)
    return [int(x) for x in IL.unslice_words(outs["o"], T)]


PIECES = [
    ("F32Mul_v1", (32, 32), F.F32Mul.evaluate),
    ("F32AddFtz_v1", (32, 32), F.F32AddFtz.evaluate),
    ("DivFullScaleA_v1", (32, 32), P.DivFullScaleA.evaluate),
    ("F32GtStrict_v1", (32, 32), RP.F32GtStrict.evaluate),
]


@pytest.mark.parametrize("name,widths,ref", PIECES, ids=[p[0] for p in PIECES])
def test_pieces_match_the_ir_primitives(name, widths, ref):
    rng = random.Random(name)
    a = f32_words(rng, 6000)
    b = f32_words(rng, 6000)
    vecs = list(zip(a, b)) + [(x, y) for x in SPECIAL for y in SPECIAL]
    got = run_piece(S.PIECES[name], widths, vecs)
    want = [ref(x, y) for x, y in vecs]
    assert sum(g != w for g, w in zip(got, want)) == 0


def ir_walk(target, args, resolved):
    """The IR composite on ``args`` by its own gates and the primitives' evaluators, a gate whose primitive id is in
    ``resolved`` taking that value: the reference the lane unit is held to."""
    _prog, call = IL.standalone(target)
    t = dict(zip(call.args, args))
    for g in call:
        t[g.index] = resolved[g.prim.id] if g.prim.id in resolved else g.prim.evaluate(*[t[o] for o in g.operands])
    return call, t


def lane_reference(V, x, temp, rcp, skip, noisy, kbit, g, best, best_i, i):
    tl, ms, gs = S._composites(V)
    call, t = ir_walk(tl, [x, temp, rcp, skip], {})
    scaled = t[call.returns[0]]
    call, t = ir_walk(ms, [i, scaled, 0], {f"BitAtx{V}_v1": kbit})
    masked = t[call.returns[1]]
    call, t = ir_walk(gs, [best, best_i, i, masked, 0, noisy], {"GumbelNoiseLane_v1": g})
    nb, ni, nxt = (t[r] for r in call.returns[:3])
    if i == 0:                        # the scan's init: (y_0, 0, 1), y_0 the step's own y
        (gt,) = [x for x in call if x.prim.id == "F32GtStrict_v1"]
        nb, ni = t[gt.operands[0]], 0
    return [scaled, nb, ni, nxt]


def test_lane_unit_matches_the_ir_composites():
    V = 1024
    low = S.lane(V)
    rng = random.Random(7)
    vecs = []
    for _ in range(4000):
        temp = rng.choice([0, 0x80000000, 0x3F800000, 0x3F4CCCCD, 0x3E99999A, 0x40000000, 0x7F000000, 5, 0x7F800000, 0x7FC00000,
                           *f32_words(rng, 2)])
        rcp = P.DivFullRcp.evaluate(temp) if rng.random() < 0.7 else f32_words(rng, 1)[0]
        vecs.append((bf16_words(rng, 1)[0], temp, rcp, rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(1), f32_words(rng, 1)[0],
                     f32_words(rng, 1)[0], rng.getrandbits(32), rng.choice([0, 0, 1, 5, rng.getrandbits(32)])))
    outs, ok = low.evaluate([np.array([v[j] for v in vecs], dtype=np.uint64) for j in range(len(S.IN_PORTS))])
    assert ok == (1 << len(vecs)) - 1
    bad = sum([int(o[n]) for o in outs] != lane_reference(V, *v) for n, v in enumerate(vecs))
    assert bad == 0


def test_the_lane_is_the_pinned_one():
    low = S.lane(128256)
    assert S.pin_digest(low) == REG.module(S.TEMPLATE).PIN == S.PIN
    assert S.pin_digest(S.lane(64)) == S.PIN
    head = low.unit_text.split("\n", 1)[0].split()
    assert head[0] == IL.FORMAT and head[1] == S.UNIT and int(head[2]) <= 1 << 13


def _set(tmp_path, V, n=3):
    sub = TM.subcircuit(S.TEMPLATE, V=V)
    return InputSet.open(generate(sub, n, 20260926, tmp_path)["path"])


def _replay_scan(masked, g, noisy, V):
    """``GumbelSelectF32``'s scan replayed from its Definition with the IR primitives: every lane's carry out."""
    y0 = SelectF32.evaluate(noisy, F.F32AddFtz.evaluate(masked[0], g[0]), masked[0])
    carry = (y0, 0, 1)
    out = [carry]
    for v in range(1, V):
        call, t = ir_walk(SM.GumbelSelectStep, [*carry, masked[v], 0, noisy], {"GumbelNoiseLane_v1": g[v]})
        carry = tuple(t[r] for r in call.returns[:3])
        out.append(carry)
    return out


@pytest.mark.parametrize("V", [1056, 1280])
@pytest.mark.slow
def test_row_words_match_the_ir_evaluator(tmp_path, V):
    s = _set(tmp_path, V)
    logits, sc, outs = S.instances(s, 0, s.n)
    for i in range(s.n):
        args = [int(sc[p][i]) for p in S.PUBLIC_PORTS]
        cuts, tok = S.row_words(logits[i], *args)
        row = [int(x) for x in logits[i]]
        assert tok == evaluate(bind(SM.GumbelTopPTokenSelect, V=V), row, *args)[0] == int(outs[i])
        lane = cuts[len(S.SCALARS) + len(S.INIT):].reshape(V, len(S.LANE)).astype(np.int64)
        scaled = evaluate(bind(SM.TemperatureScale, V=V), row, args[3])
        assert list(lane[:, 2]) == list(scaled)
        masked = evaluate(bind(SM.TopPMask, V=V), list(scaled), args[0], args[4])
        assert list(np.where(lane[:, 0] == 1, lane[:, 2], S.NEG_INF)) == list(masked)
        keep = SM.topp_mask_word(V).evaluate(*scaled, args[0], args[4])
        assert [(keep >> v) & 1 for v in range(V)] == list(lane[:, 0])
        key = SM.GumbelStreamKey.evaluate(args[1], args[2])
        assert all(SM.GumbelNoiseLane.evaluate(key, v) == lane[v, 1] for v in range(0, V, 5))
        assert list(cuts[:4]) == S.row_scalars(args[3]) and list(cuts[4:7]) == [0, 0, 0]
        carries = _replay_scan(list(masked), list(lane[:, 1]), S.row_scalars(args[3])[3], V)
        assert [tuple(int(x) for x in lane[v, 3:6]) for v in range(V)] == carries


@pytest.mark.slow
def test_staged_rows_through_the_units(tmp_path):
    V = 1280
    s = _set(tmp_path, V, 4)
    low = S.lane(V)
    path = S.stage(low, s, 0, s.n, tmp_path / "st")
    head, logits, _d, _pub, cuts = S.read(path)
    assert head["format"] == S.FORMAT and head["layout"]["chunk_blocks"] == [16, 16, 8] and head["layout"]["nb"] == 4
    outs, ok = low.evaluate(S.unit_ports(cuts, logits))
    assert ok == (1 << (s.n * V)) - 1
    assert all((o == w).all() for o, w in zip(outs, S.unit_outputs(cuts, V)))
    assert S.check_native(path) == [] and S.check_staged(low, s, path) == []
    assert (tmp_path / "st" / "net.txt").read_text() == low.text
    assert S.serve_args(path) == ["--native-checked", S.public_sha256(path)]


def test_the_captured_vocabulary_plans_under_the_gpu_cap():
    pl = S.plan(S.lane(128256), 128256)
    assert pl["chunk_blocks"] == [16] * 250 + [8] and pl["nb"] == 4 and pl["k_log"] <= 22 and pl["units_per_block"] == 128


def test_supports_and_relation():
    sub = TM.subcircuit(S.TEMPLATE, V=128256)
    assert REG.supports(sub, "frame-v3/blake3-keyed") is None
    assert REG.relation(sub, "frame-v3/blake3-keyed") == "irf:gumbel-top-p-token-select"
    assert "not wired yet" in REG.supports(sub, "vllm-v1")


# --- negatives ---------------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_flipped_witness_bit_leaves_its_lane_unsatisfied(tmp_path):
    V = 1056
    s = _set(tmp_path, V, 2)
    low = S.lane(V)
    path = S.stage(low, s, 0, s.n, tmp_path / "st")
    _h, logits, _d, _p, cuts = S.read(path)
    row = low.layout.in_words * IL.WORD + 900
    _outs, ok = low.evaluate(S.unit_ports(cuts, logits), flip=(row, 37))
    assert ok == ((1 << (s.n * V)) - 1) & ~(1 << 37)


@pytest.mark.parametrize("word", ["kbit", "g"])
@pytest.mark.slow
def test_a_forged_native_word_is_refused(tmp_path, word):
    V = 1056
    s = _set(tmp_path, V, 2)
    path = S.stage(S.lane(V), s, 0, s.n, tmp_path / "st")
    b = bytearray(path.read_bytes())
    nl = b.index(b"\n")
    head, *_ = S.read(path)
    at = nl + 1 + 2 * s.n * V + 32 * s.n + 4 * s.n * sum(w for _, w in head["public_ports"]) + 4 * S.lane_word(33, word)
    b[at] ^= 1
    path.write_bytes(bytes(b))
    assert S.check_native(path) and S.check_staged(S.lane(V), s, path)
    with pytest.raises(ValueError):
        S.serve_args(path)


@pytest.mark.slow
def test_a_forged_unit_word_breaks_the_lane_relation(tmp_path):
    V = 1056
    s = _set(tmp_path, V, 2)
    low = S.lane(V)
    path = S.stage(low, s, 0, s.n, tmp_path / "st")
    _h, logits, _d, _p, cuts = S.read(path)
    cuts = cuts.copy()
    cuts[1, S.lane_word(100, "scaled")] ^= 1 << 5
    outs, _ = low.evaluate(S.unit_ports(cuts, logits))
    assert sum(int((o != w).sum()) for o, w in zip(outs, S.unit_outputs(cuts, V))) == 1


def test_a_mutated_netlist_is_caught_by_the_ir_comparison():
    V = 64
    low = S.lane(V)
    rng = random.Random(3)
    vecs = []
    for _ in range(1500):
        temp = rng.choice([0x3F4CCCCD, 0x7F000000, 5, *f32_words(rng, 1)])
        vecs.append((bf16_words(rng, 1)[0], temp, P.DivFullRcp.evaluate(temp), rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(1),
                     f32_words(rng, 1)[0], f32_words(rng, 1)[0], rng.getrandbits(32), rng.choice([0, rng.getrandbits(32)])))
    ports = [np.array([v[j] for v in vecs], dtype=np.uint64) for j in range(len(S.IN_PORTS))]
    want = [lane_reference(V, *v) for v in vecs]
    first = low.layout.in_words * IL.WORD
    ands = list(range(first + 1, first + low.layout.counts["ands"]))
    caught = 0
    for k in ands[::len(ands) // 48][:48]:
        lay = IL.Layout(list(low.layout.rows_a), list(low.layout.rows_b), low.layout.const, low.layout.in_groups, low.layout.out_groups,
                        low.layout.counts)
        lay.rows_b[k] = sorted(set(lay.rows_b[k]) ^ {k - 1})
        outs, _ = IL.evaluate_layout(lay, ports, len(vecs))
        caught += sum([int(o[n]) for o in outs] != w for n, w in enumerate(want)) > 0
    assert caught >= 16                               # a third of the spread-out mutations (the rest sit in rare paths)


CAPTURED = os.environ.get("VERITY_CAPTURED_101_GUMBEL")


@pytest.mark.skipif(not CAPTURED, reason="set VERITY_CAPTURED_101_GUMBEL to the captured #101 sampling set's directory")
def test_captured_rows_through_the_units(tmp_path):
    s = InputSet.open(Path(CAPTURED))
    low = S.lane(128256)
    path = S.stage(low, s, 0, s.n, tmp_path)
    _h, logits, _d, _p, cuts = S.read(path)
    outs, ok = low.evaluate(S.unit_ports(cuts, logits))
    assert ok == (1 << (s.n * 128256)) - 1
    assert all((o == w).all() for o, w in zip(outs, S.unit_outputs(cuts, 128256)))
    assert S.check_native(path) == []
