"""The CPU replay (`replay`): each template's function is circuits' semantics (a scalar composition of main's word
primitives' own evaluators, and main's Definition wherever one is importable), every template keys and parses at the frozen
shape's parameters, the draw is core's draw object from live coins, and a replay of seeded W8A8 coordinates, of a tiny instance
of the frozen architecture (`cpu_step` on the seeded weights) and of a bundle in the drawn-unit bundle's form matches unit for unit and
catches one flipped bit."""
from __future__ import annotations

import copy
import struct

import numpy as np
import pytest

import cpu_step as S
import replay as R
from verity.ml.protocols.pouw.circuit import words as PW
from verity.core.protocols.sampled_proofs.one_stage.draw import strata_problem
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.reference import evaluate as reference
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as SC

ADD, MUL, DIV = W.F32AddV2.evaluate, W.F32MulV2.evaluate, W.F32DivV2.evaluate
BF16, WIDE, I2F = P.F2fpBf16.evaluate, P.Bf16ToF32.evaluate, P.S32ToF32Rn.evaluate
SQRT, EX2, GT = PW.PouwF32SqrtRn.evaluate, MU.MufuEx2Ftz.evaluate, SC.Bf16GtStrict.evaluate
ONE = 0x3F800000

SPECIAL16 = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0,
             0xFFC0, 0x7F81, 0x4300, 0x42FE, 0xC2FF]
SPECIAL32 = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0x7F800000,
             0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF, 0x80000001, 0x42FE0000, 0x3727C5AC]


def words(rng: np.random.Generator, width: int, shape: tuple[int, ...], special: float = 0.2) -> np.ndarray:
    """Random words of ``width`` bits, a fraction ``special`` of them the IEEE corner cases (bf16 or f32)."""
    dt = {8: np.uint8, 16: np.uint16, 32: np.uint32}[width]
    a = rng.integers(0, 1 << width, size=shape, dtype=np.uint64).astype(dt)
    if width in (16, 32):
        pool = np.array(SPECIAL16 if width == 16 else SPECIAL32, dtype=dt)
        mask = rng.random(shape) < special
        a[mask] = rng.choice(pool, size=int(mask.sum()))
    return a


def bf16_values(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Finite bf16 words of moderate magnitude (what a row reaching a softmax or a norm holds), with a few corner cases."""
    v = (rng.standard_normal(shape) * 4).astype(np.float32).view(np.uint32) >> 16
    a = v.astype(np.uint16)
    mask = rng.random(shape) < 0.1
    a[mask] = rng.choice(np.array(SPECIAL16, np.uint16), size=int(mask.sum()))
    return a


def half(w: int, h: int) -> int:
    return (w >> (16 * h)) & 0xFFFF


def seq(ws) -> int:
    acc = 0
    for w in ws:
        acc = ADD(acc, int(w))
    return acc


def s8(w: int) -> int:
    return w - 256 if w & 0x80 else w


def s32(w: int) -> int:
    return w - (1 << 32) if w & 0x80000000 else w


def is_nan16(w: int) -> bool:
    return (w & 0x7F80) == 0x7F80 and (w & 0x7F) != 0


def relu16(w: int) -> int:
    return 0 if (w & 0x8000) and (w & 0x7FFF) and not is_nan16(w) else w


def to_s8_rni_sat(w: int) -> int:
    v = struct.unpack("<f", struct.pack("<I", w))[0]
    q = 0 if v != v else max(-128, min(127, round(v))) if abs(v) != float("inf") else (127 if v > 0 else -128)
    return q & 0xFF


def quant_scale(amax: int) -> int:
    a = WIDE(amax)
    return DIV(a, 0x42FE0000) if 1 <= s32(a) <= 0x7F800000 else ONE


def exp(x: int, m: int) -> int:
    return EX2(MUL(ADD(WIDE(x), WIDE(m) ^ 0x80000000), 0x3FB8AA3B))


def chunk(x) -> tuple[int, int]:
    best, idx = 0xFF80, 0
    for i, v in enumerate(x):
        if GT(int(v), best):
            best, idx = int(v), i
    return best, idx


def smax(x) -> int:
    if any(is_nan16(int(v)) for v in x):
        return 0x7FFF
    m = int(x[0])
    for v in x[1:]:
        if GT(int(v), m):
            m = int(v)
    return m


EPS = R.EPS_1E5
#: per template key: a scalar composition of main's word primitives, the inputs as Python ints (one word, or a list)
SCALAR = {
    "RmsSumSqPart_v1{C=8,HALF=0}": lambda x: half(seq(MUL(WIDE(v), WIDE(v)) for v in x), 0),
    "RmsSumSqPart_v1{C=8,HALF=1}": lambda x: half(seq(MUL(WIDE(v), WIDE(v)) for v in x), 1),
    "F32SumPart_v1{G=4,HALF=0}": lambda s: half(seq(s), 0),
    "F32SumPart_v1{G=4,HALF=1}": lambda s: half(seq(s), 1),
    f"RmsNormRecip_v1{{N=32,G=4,EPS={EPS},HALF=0}}": lambda s: half(DIV(ONE, SQRT(ADD(DIV(seq(s), I2F(32)), EPS))), 0),
    f"RmsNormRecip_v1{{N=32,G=4,EPS={EPS},HALF=1}}": lambda s: half(DIV(ONE, SQRT(ADD(DIV(seq(s), I2F(32)), EPS))), 1),
    "RmsNormOut_v1": lambda x, rs, w: BF16(MUL(WIDE(w), WIDE(BF16(MUL(WIDE(x), rs))))),
    "RowAbsMax_v1{N=8,RELU=false}": lambda x: max(v & 0x7FFF for v in x),
    "RowAbsMax_v1{N=8,RELU=true}": lambda x: max(relu16(v) & 0x7FFF for v in x),
    "QuantS8Pair_v1{RELU=false}": lambda x, a: [to_s8_rni_sat(DIV(WIDE(v), quant_scale(a))) for v in x],
    "QuantS8Pair_v1{RELU=true}": lambda x, a: [to_s8_rni_sat(DIV(WIDE(relu16(v)), quant_scale(a))) for v in x],
    "QuantS8Scale_v1{HALF=0}": lambda a: half(quant_scale(a), 0),
    "QuantS8Scale_v1{HALF=1}": lambda a: half(quant_scale(a), 1),
    "S8AccumulatorPart_v1{K=16,HALF=0}": lambda x, w: half(sum(s8(a) * s8(b) for a, b in zip(x, w)) & 0xFFFFFFFF, 0),
    "S8AccumulatorPart_v1{K=16,HALF=1}": lambda x, w: half(sum(s8(a) * s8(b) for a, b in zip(x, w)) & 0xFFFFFFFF, 1),
    "S32SumPart_v1{P=4,HALF=0}": lambda p: half(sum(map(s32, p)) & 0xFFFFFFFF, 0),
    "S32SumPart_v1{P=4,HALF=1}": lambda p: half(sum(map(s32, p)) & 0xFFFFFFFF, 1),
    "ScaledMmS8Epilogue_v1{P=4}": lambda p, sx, sw: BF16(MUL(MUL(I2F(sum(map(s32, p)) & 0xFFFFFFFF), sx), sw)),
    "SoftmaxMax_v1{N=5}": smax,
    "SoftmaxRecip_v1{N=5,HALF=0}": lambda x, m: half(DIV(ONE, seq(exp(v, m) for v in x)), 0),
    "SoftmaxRecip_v1{N=5,HALF=1}": lambda x, m: half(DIV(ONE, seq(exp(v, m) for v in x)), 1),
    "SoftmaxOut_v1": lambda x, m, r: BF16(MUL(exp(x, m), r)),
    "EmbeddingCoordinate_v1{V=24}": lambda tok, col, pos: W.Bf16Add.evaluate(col[tok], pos),
    "ArgmaxChunkBest_v1{C=8}": lambda x: chunk(x)[0],
    "ArgmaxChunkIdx_v1{C=8}": lambda x: chunk(x)[1],
    "ArgmaxCombine_v1{V=24,C=8}": None,
    "GemmCoordinatePad_v1{T=5,K=16,DOT=HopperBF16WgmmaDot16_v1}":
        lambda x, w: reference(bind(G.GemmCoordinate, K=16, **G.GEMM_HOPPER), list(x) + [0] * 11, list(w) + [0] * 11)[0],
    "Bf16Add_v2": lambda a, b: W.Bf16Add.evaluate(a, b),
}


def combine(x0: int, best, idx) -> int:
    run, run_i = x0, 0
    for c, (b, i) in enumerate(zip(best, idx)):
        if GT(int(b), run):
            run, run_i = int(b), 8 * c + int(i)
    return run_i & 0xFFFF


SCALAR["ArgmaxCombine_v1{V=24,C=8}"] = combine


def inputs(t: R.Template, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """``n`` instances of ``t``'s parameters: corner-case words, finite rows where a template reads a row of values, a
    token below V, s32 parts small enough that their sum is the frozen shape's."""
    out = []
    for name, width, count in t.params:
        shape = (n,) if count == 0 else (n, count)
        if t.name == "EmbeddingCoordinate_v1" and name == "tok":
            a = rng.integers(0, dict(t.statics)["V"], size=shape).astype(np.uint16)
        elif t.name == "ArgmaxCombine_v1" and name == "idx":
            a = rng.integers(0, dict(t.statics)["C"], size=shape).astype(np.uint16)
        elif name == "parts":
            a = rng.integers(-(1 << 28), 1 << 28, size=shape).astype(np.int32).view(np.uint32)
        elif t.name.startswith(("Softmax", "Rms")) and width == 16:
            a = bf16_values(rng, shape)
        else:
            a = words(rng, width, shape)
        out.append(a)
    return out


@pytest.mark.parametrize("key", sorted(SCALAR))
def test_each_reference_is_its_composition_of_mains_primitives(key: str) -> None:
    t = R.template(key)
    rng = np.random.default_rng(sum(map(ord, key)))
    args = inputs(t, rng, 96)
    got = t.reference(*args)
    for i in range(len(got)):
        row = [int(a[i]) if a.ndim == 1 else [int(v) for v in a[i]] for a in args]
        want = SCALAR[key](*row)
        assert list(map(int, got[i])) == (want if isinstance(want, list) else [want]), (key, row)


def test_softmax_and_rmsnorm_on_rows_with_nans_and_infinities() -> None:
    """The corner cases a row of values can't reach by chance: a NaN anywhere, an infinite max, an all-zero row."""
    py = [[0x3F80, 0x7FC0, 0, 0, 0], [0x7F80, 0x3F80, 0xFF80, 0, 0x8000], [0, 0x8000, 0, 0, 0], [0xFF80] * 5,
          [0x8000, 0, 0x8000, 0, 0]]
    rows = np.array(py, np.uint16)
    m = R.template("SoftmaxMax_v1{N=5}").reference(rows)[:, 0]
    assert list(m) == [smax(r) for r in py] == [0x7FFF, 0x7F80, 0, 0xFF80, 0x8000]
    for h in (0, 1):
        r = R.template(f"SoftmaxRecip_v1{{N=5,HALF={h}}}").reference(rows, m)[:, 0]
        assert list(r) == [half(DIV(ONE, seq(exp(v, int(mm)) for v in row)), h) for row, mm in zip(py, m)]
    z = np.zeros((2, 8), np.uint16)
    z[1, 3] = 0x7F80
    for h in (0, 1):
        assert list(R.template(f"RmsSumSqPart_v1{{C=8,HALF={h}}}").reference(z)[:, 0]) == [0, half(0x7F800000, h)]


def test_each_importable_definition_is_its_reference() -> None:
    """Wherever main has a template's Definition (today the W8A8 and BF16 coordinates and Bf16Add; circuits' quantizer ones as
    they land), the reference must be that Definition."""
    rng = np.random.default_rng(5)
    keys = [*SCALAR, "ScaledMmS8Coordinate_v1{K=32,DOT=BlackwellS8ImmaDot32_v1}", "GemmCoordinate_v2{K=16,DOT=HopperBF16WgmmaDot16_v1}"]
    seen = []
    for key in keys:
        t = R.template(key)
        target = t.target()
        if target is None:
            assert t.status == R.PENDING
            continue
        assert t.status == R.DEFINITION
        args = [R._as_param(a, p) for a, p in zip(inputs(t, rng, 256), t.params)]
        want = np.asarray(evaluate_batch(target, *args)).reshape(256, t.out[1])
        assert np.array_equal(t.reference(*args), want), key
        seen.append(t.name)
    assert {"ScaledMmS8Coordinate_v1", "GemmCoordinate_v2", "Bf16Add_v2"} <= set(seen)


FROZEN_KEYS = [
    "RmsSumSqPart_v1{C=128,HALF=0}", "F32SumPart_v1{G=64,HALF=1}", f"RmsNormRecip_v1{{N=16384,G=2,EPS={EPS},HALF=0}}",
    "RmsNormOut_v1", "RowAbsMax_v1{N=128,RELU=true}", "RowAbsMax_v1{N=4,RELU=false}", "QuantS8Scale_v1{HALF=1}",
    "QuantS8Pair_v1{RELU=false}", "S8AccumulatorPart_v1{K=256,HALF=0}", "S32SumPart_v1{P=64,HALF=1}",
    "ScaledMmS8Epilogue_v1{P=64}", "ScaledMmS8Epilogue_v1{P=4}", "GemmCoordinate_v2{K=64,DOT=HopperBF16WgmmaDot16_v1}",
    "GemmCoordinatePad_v1{T=32,K=32,DOT=HopperBF16WgmmaDot16_v1}", "SoftmaxMax_v1{N=32}", "SoftmaxRecip_v1{N=1,HALF=0}",
    "SoftmaxOut_v1", "Bf16Add_v2", "EmbeddingCoordinate_v1{V=256}", "ArgmaxChunkBest_v1{C=128}", "ArgmaxChunkIdx_v1{C=128}",
    "ArgmaxCombine_v1{V=256,C=128}", "ScaledMmS8Coordinate_v1{K=16384,DOT=BlackwellS8ImmaDot32_v1}",
]


def test_every_template_keys_parses_and_evaluates_at_the_frozen_shape() -> None:
    assert {k.split("{")[0] for k in FROZEN_KEYS} == set(R.templates())
    rng = np.random.default_rng(7)
    for key in FROZEN_KEYS:
        t = R.template(key)
        assert t.key == key and R.template(key) is t
        out = t.evaluate(*inputs(t, rng, 2))
        assert out.shape == (2, t.out[1])


@pytest.mark.parametrize("bad", ["Nope_v1", "RmsNormOut_v1{C=1}", "SoftmaxMax_v1", "RowAbsMax_v1{RELU=false,N=8}",
                                 "GemmCoordinatePad_v1{T=33,K=32,DOT=HopperBF16WgmmaDot16_v1}"])
def test_a_key_not_in_canonical_form_is_refused(bad: str) -> None:
    with pytest.raises((KeyError, ValueError, TypeError)):
        R.template(bad)


# ---- the draw ----------------------------------------------------------------------------------------------------------

POP = [("A_v1", 10), ("B_v1{K=2}", 1000), ("C_v1", 3)]
CTX = {"step_root": "00" * 64}


def test_the_draw_is_a_function_of_the_coins_and_the_context() -> None:
    a = R.draw(POP, "uniform:20", b"coins", CTX)
    assert a == R.draw(POP, "uniform:20", b"coins", CTX)
    assert a["units"] != R.draw(POP, "uniform:20", b"other coins", CTX)["units"]
    assert a["units"] != R.draw(POP, "uniform:20", b"coins", {"step_root": "11" * 64})["units"]
    assert a["law"] == "subset" and a["k"] == 20 and a["population"] == 1013
    assert a["units"] == sorted(set(a["units"])) and len(a["units"]) == 20 and all(0 <= u < 1013 for u in a["units"])


def test_a_stratified_draw_is_cores_stratified_law() -> None:
    d = R.draw(POP, "stratified:30", b"coins", CTX)
    assert strata_problem(d, d["population"]) is None
    for s in d["strata"]:
        lo, hi = s["units"][0]
        assert sum(lo <= u < hi for u in d["units"]) == s["k"] >= 1


def test_an_explicit_plan_draws_its_counts() -> None:
    d = R.draw(POP, {"A_v1": 2, "C_v1": 3}, b"coins", CTX)
    assert d["law"] == "plan" and d["k"] == 5
    assert sorted(u for u in d["units"] if u < 10).__len__() == 2 and [u for u in d["units"] if u >= 1010] == [1010, 1011, 1012]
    for plan in ("uniform:0", "uniform:2000", "bernoulli:3", {"D_v1": 1}, {"C_v1": 4}):
        with pytest.raises(ValueError):
            R.draw(POP, plan, b"coins", CTX)


# ---- end to end --------------------------------------------------------------------------------------------------------

def flip(units: list[dict], i: int, bit: int = 0) -> list[dict]:
    out = copy.deepcopy(units)
    out[i]["outputs"][0] ^= 1 << bit
    return out


S8 = "ScaledMmS8Coordinate_v1{K=128,DOT=BlackwellS8ImmaDot32_v1}"


def s8_units(n: int = 48, tamper: bool = False) -> list[dict]:
    """``n`` W8A8 coordinates on seeded s8 rows and positive scales, their outputs the template's numpy reference
    (``tamper``: every output's bit 0 flipped), as `replay`'s units."""
    t = R.template(S8)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (n, 128), dtype=np.int8).view(np.uint8) for _ in range(2))
    sx, sw = (rng.uniform(1e-3, 1e-2, n).astype(np.float32).view(np.uint32) for _ in range(2))
    outs = t.reference(x, sx, w, sw)[:, 0].astype(np.uint64) ^ np.uint64(tamper)
    return [{"unit": i, "template": S8, "label": None,
             "inputs": {"x": x[i].tolist(), "sx": [int(sx[i])], "wrow": w[i].tolist(), "sw": [int(sw[i])]},
             "outputs": [int(outs[i])]} for i in range(n)]


def test_seeded_w8a8_coordinates_replay_and_a_flipped_bit_is_caught() -> None:
    units = s8_units()
    pop = [(units[0]["template"], len(units))]
    d = R.draw(pop, "uniform:24", b"fixture coins", {"run": "s8-units"})
    drawn = [units[u] for u in d["units"]]
    rep = R.replay(drawn)
    assert R.ok(rep) and rep["matched"] == 24
    assert rep["templates"] == {"ScaledMmS8Coordinate_v1{K=128,DOT=BlackwellS8ImmaDot32_v1}":
                                {"status": R.DEFINITION, "units": 24, "mismatched": 0}}
    rep = R.replay(flip(drawn, 5, 9))
    assert not R.ok(rep) and [m["unit"] for m in rep["mismatched"]] == [drawn[5]["unit"]] and rep["matched"] == 23
    assert R.ok(R.replay(units))
    bad = R.replay(s8_units(tamper=True))
    assert bad["matched"] == 0 and len(bad["mismatched"]) == len(units)


def test_a_malformed_unit_is_reported_not_replayed() -> None:
    units = s8_units()[:3]
    units[0] = {**units[0], "inputs": {**units[0]["inputs"], "x": units[0]["inputs"]["x"][:-1]}}
    units[1] = {**units[1], "template": "Nope_v1"}
    rep = R.replay(units)
    assert rep["matched"] == 1 and {m["unit"] for m in rep["malformed"]} == {0, 1} and not R.ok(rep)


def iface_b(units: list[dict]) -> tuple[dict, bytes]:
    """``units`` as the drawn-unit bundle carries them (`verity/drawn-unit-bundle/v1`: per unit its template's name
    and binding, each port's row as u16 words little-endian in the payload); openings left out, `units_of` doesn't read them."""
    payload, out = bytearray(), []
    for u in units:
        t = R.template(u["template"])
        rows = []
        for (p, width, _), kind in [*((q, "in") for q in t.params), (("o0", t.out[0], 1), "out")]:
            w = u["inputs"][p] if kind == "in" else u["outputs"]
            row = np.asarray(w, dtype=f"<u{width // 8}").tobytes()
            rows.append({"port": p, "kind": kind, "row": [len(payload), len(row)], "salt": [len(payload) + len(row), 192]})
            payload += row + bytes(192)
        out.append({"unit": u["unit"], "coords": {"block": 0, "step": 0, "slot": 0, "layer": 0, "op": "x", "unit": u["unit"]},
                    "template": {"id": t.key.replace("DOT=BlackwellS8ImmaDot32_v1", 'DOT={"fn":"BlackwellS8ImmaDot32_v1"}'),
                                 "definition": t.name, "binding": {"DOT": {"fn": "BlackwellS8ImmaDot32_v1"}, "K": 128}},
                    "rows": rows})
    return {"format": R.IFACE_B, "units": out}, bytes(payload)


def test_an_iface_b_bundle_replays_and_a_flipped_payload_bit_is_caught() -> None:
    units = s8_units()[:6]
    doc, payload = iface_b(units)
    got = R.units_of(doc, payload)
    assert [(u["unit"], u["template"], u["inputs"], u["outputs"]) for u in got] == \
        [(u["unit"], u["template"], u["inputs"], u["outputs"]) for u in units]
    assert R.ok(R.replay(got))
    bad = bytearray(payload)
    o = next(r for r in doc["units"][4]["rows"] if r["kind"] == "out")
    bad[o["row"][0] + 1] ^= 0x40
    rep = R.replay(R.units_of(doc, bytes(bad)))
    assert [m["unit"] for m in rep["mismatched"]] == [units[4]["unit"]]
    with pytest.raises(ValueError):
        R.units_of(doc)


TINY_FORMS = S.Forms(rms_chunk=16, rms_group=2, amax_chunk=16, part_fanin=2, pv_k=16)


@pytest.fixture(scope="module")
def tiny() -> S.Step:
    """Two sequences through a tiny MQA decoder of the frozen architecture (width 64, 2 layers, 4 query heads of 16, MLP
    128, byte vocabulary, context 8, 16-code weight rows) on the test-seeded weights: 6 positions, ~159k units."""
    from e2e_h200 import served_weights as WG
    shape = WG.Shape("replay-tiny-mqa", 64, 2, 4, 16, 1, 128, 256, 8, 16)
    step = S.Step(shape, lambda n: WG.generate(shape, WG.TEST_SEED, n), batch=2, forms=TINY_FORMS)
    S.decode(step, [b"hi", b"ok"], 4)
    return step


def test_the_tiny_frozen_architecture_replays_every_unit(tiny) -> None:
    """Every unit of every template the architecture runs (RMSNorm's three levels, the quantizer's tree, the K-split W8A8
    linears, MQA attention with its softmax at every length 1..6, the residual adds, the embedding, argmax)."""
    rep = R.replay(tiny.rec.units())
    assert R.ok(rep), (rep["mismatched"][:3], rep["malformed"][:3])
    assert {k.split("{")[0] for k in rep["templates"]} == set(R.templates()) - {"ScaledMmS8Coordinate_v1"}


def test_a_drawn_tiny_step_replays_and_a_flipped_bit_is_caught_in_every_template(tiny) -> None:
    pop = tiny.rec.population()
    d = R.draw(pop, "stratified:200", b"fixture coins", {"run": "replay-tiny-mqa"})
    assert strata_problem(d, d["population"]) is None
    drawn = tiny.rec.units(d["units"])
    assert [u["unit"] for u in drawn] == d["units"] and R.ok(R.replay(drawn))
    first: dict[str, int] = {}
    for i, u in enumerate(drawn):
        first.setdefault(u["template"], i)
    assert set(first) == {k for k, _ in pop}
    for key, i in first.items():
        rep = R.replay(flip(drawn, i, 7))
        assert [m["unit"] for m in rep["mismatched"]] == [drawn[i]["unit"]], key
