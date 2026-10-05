"""`verity.ml.boolean.gemm`: the Boolean BF16 steps, cast, dot and coordinate agree with their word Definitions (`word_of`) on
sampled and special words and on every hardware capture the word Definitions replay, carry exactly their builders' live ANDs,
and are pinned by gate count and program digest."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verity.primitives.circuits import evaluation as E
from verity.primitives.circuits.evaluation import bits as BS
from verity.primitives.circuits.evaluation.reference import standalone_call
from verity.primitives.circuits.ir import boolean as BL
from verity.primitives.circuits.ir import codec
from verity.primitives.circuits.ir.defs import REGISTRY, bind
from verity.ml import kernels as K
from verity.ml import prims as P
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit

FIXTURES = Path(__file__).parent / "fixtures"

AMPERE_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE)
HOPPER_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER)
DOTS = [bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE), bind(BG.DotBf16, K=48, **BG.GEMM_HOPPER)]
STEPS = [bind(BG.AmpereBF16TcDot16), bind(BG.HopperBF16WgmmaDot16)]
GEMMS = [bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE), bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER)]
FP8_STEPS = [bind(BG.HopperE4m3QgmmaDot32), bind(BG.BlackwellE4m3QmmaDot32)]
FP8_COORDS = [bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3), bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3)]
FP4 = [(bind(BG.BlackwellNvf4OmmaDot64), K.BLACKWELL_SM120_NVF4), (bind(BG.BlackwellMxf4OmmaDot64), K.BLACKWELL_SM120_MXF4),
       (bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4), K.BLACKWELL_SM120_NVF4),
       (bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4), K.BLACKWELL_SM120_MXF4)]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F2fpBf16_v2": (66, 98, 15, "6db4bfff88f30864294527d1fbc78e9328edb598187535c7278b22d982f514a6"),
    "AmpereBF16TcDot16_v3": (8260, 15717, 860, "18031c3adb1a62ff7ac3af48bdb5b27f13f9afdcd70f519368faa0c27df0e2dc"),
    "HopperBF16WgmmaDot16_v2": (7872, 15049, 761, "4cef800bab64f8f7133d3e1ae3a36565f6de93553800e06023bc4225de1d12f0"),
    AMPERE_K64.id: (33106, 62966, 3455, "3547dde5e8fc8eaced08c40e120042eeae46922f17689c3112cb0cd552e65503"),
    HOPPER_K64.id: (31554, 60294, 3059, "8dcec5922c2388c3c1c13cba0be0023338d856d8068d84470b545df02d4024e7"),
    DOTS[0].id: (33040, 62868, 3440, "c71d572a2958f817da93a41fe847a5be35b6c1549f76e43aa48f43ca92efbc4d"),
    DOTS[1].id: (23616, 45147, 2283, "6350b7661f2b2a0e9d522371c3222f0bfc59fc9e486a68116011e14403e0fdf3"),
    GEMMS[0].id: (49758, 94596, 5205, "0a0a1dc213dbde84ec28e6db0f3033117e0416dbffc8edefe18ec7e95f358806"),
    GEMMS[1].id: (47430, 90588, 4611, "1c7cb1693d564049edac0136a85ee174313479c3e7325f493db87f6c126b5fa7"),
    "HopperE4m3QgmmaDot32_v2": (6799, 12201, 1018, "34552128143cbc0f016c782e6e0c561f13dd8bffe9cf5962606b464f980b3fa0"),
    "BlackwellE4m3QmmaDot32_v2": (9128, 17402, 1025, "833e9ca76a0e842775c1be9c31025d2ebf23760835c302cfa2d3c36b5a05a524"),
    FP8_COORDS[0].id: (13664, 24500, 2051, "503acd555cbc8477ad837275b04c6e4e17647a5d15f596dd99685c7403432618"),
    FP8_COORDS[1].id: (18322, 34902, 2065, "812ec8f6cc84a2ec16067426bc53c502ba3000a68ecd12b08cc2aa5a340a5f0c"),
    "BlackwellE5m2QmmaDot32_v2": (8931, 17231, 1251, "49e17b9ba15290f61ec48cf93281331c9848c752e877a1b79f330790a649052a"),
    "BlackwellNvf4OmmaDot64_v2": (6864, 13949, 616, "8c4b251d24cf87a9d5dec18c7a8212f6db2be096878f1eed71fdc6b67deff1ef"),
    "BlackwellMxf4OmmaDot64_v2": (5917, 12058, 605, "a2c3c1d8f2e0d31ac32461458d84877b7d57d2e384e68ac3989ac03d41bedfd6"),
    FP4[2][0].id: (13794, 27996, 1247, "ef4ba719d62b94aacad91b282c0137c2d4f15bad679a34365fff10a8fa7cc4c4"),
    FP4[3][0].id: (11900, 24214, 1225, "e75d20526b7ada2debc472d65568267d9a42e9f1f5d55b86cdf54c8344b4f7d4"),
}
TARGETS = [bind(BG.F2fpBf16), *STEPS, AMPERE_K64, HOPPER_K64, *DOTS, *GEMMS, *FP8_STEPS, *FP8_COORDS, bind(BG.BlackwellE5m2QmmaDot32),
           *(t for t, _ in FP4)]


@pytest.mark.parametrize("target", TARGETS, ids=[t.id for t in TARGETS])
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("d", [BG.F2fpBf16, BG.AmpereBF16TcDot16, BG.HopperBF16WgmmaDot16, BG.HopperE4m3QgmmaDot32,
                               BG.BlackwellE4m3QmmaDot32, BG.BlackwellE5m2QmmaDot32, BG.BlackwellNvf4OmmaDot64,
                               BG.BlackwellMxf4OmmaDot64], ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    """The traced circuit has one And gate per AND the builder keeps live, with common ANDs shared (`Circuit(cse=True)`)."""
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert T.counts["and"] == C.counts()["ands"] == PINS[f"{d.name}_v{d.version}"][0]


def _agree(target, inputs) -> None:
    word = TR.word_of(target)
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64)
    assert got.shape == ref.shape
    bad = np.flatnonzero(got.reshape(len(got), -1) != ref.reshape(len(ref), -1))
    assert not bad.size, f"{target.id}: {bad.size} instances differ from {word.id}, first {bad[:5]}"


def test_the_cast_agrees_with_f2fp_v1() -> None:
    rng = np.random.default_rng(11)
    hi = np.repeat(np.arange(1 << 16, dtype=np.uint32), 4) << np.uint32(16)
    lo = np.tile(np.array([0, 0x7FFF, 0x8000, 0x8001], np.uint32), 1 << 16)
    words = np.concatenate([hi | lo, rng.integers(0, 1 << 32, 200_000, dtype=np.uint64).astype(np.uint32)])
    _agree(bind(BG.F2fpBf16), [words])


@pytest.mark.parametrize("target", STEPS, ids=[t.id for t in STEPS])
def test_the_steps_agree_with_their_word_steps(target) -> None:
    params, _ = E.batch.signature_of(TR.word_of(target))
    _agree(target, K.sample_tc(np.random.default_rng(12), 20_000, params))


def _finite_e4m3(rng, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """``inputs`` with every E4M3 operand byte drawn uniformly from the finite bytes (0x7F and 0xFF are E4M3's NaNs)."""
    out = []
    for a in inputs:
        if a.dtype == np.uint8:
            a = rng.integers(0, 0x7F, a.shape, dtype=np.uint8) | (rng.integers(0, 2, a.shape, dtype=np.uint8) << np.uint8(7))
        out.append(a)
    return out


@pytest.mark.parametrize("target", FP8_STEPS + FP8_COORDS, ids=lambda t: t.id)
def test_the_fp8_steps_and_coordinates_agree_with_their_word_definitions(target) -> None:
    rng = np.random.default_rng(17)
    params, _ = E.batch.signature_of(TR.word_of(target))
    sampled = K.sample_tc(rng, 20_000, params)
    _agree(target, sampled)
    _agree(target, _finite_e4m3(rng, sampled))


def test_the_e5m2_step_agrees_with_its_word_step() -> None:
    """On `sample_tc_e5m2`'s instances, on uniform words, and on uniform finite E5M2 operands (exponent field not all ones)."""
    rng = np.random.default_rng(19)
    target = bind(BG.BlackwellE5m2QmmaDot32)
    params, _ = E.batch.signature_of(TR.word_of(target))
    sampled = K.sample_tc_e5m2(rng, 20_000, params)
    _agree(target, sampled)
    uniform = [rng.integers(0, 1 << (32 if a.dtype == np.uint32 else 8), a.shape, dtype=np.uint64).astype(a.dtype) for a in sampled]
    _agree(target, uniform)
    _agree(target, [a if a.dtype == np.uint32 else np.where((a & 0x7C) == 0x7C, a & 0xFB, a).astype(np.uint8) for a in uniform])


@pytest.mark.parametrize("target,model", FP4, ids=[t.id for t, _ in FP4])
def test_the_fp4_steps_and_coordinates_agree_with_their_word_definitions(target, model) -> None:
    """On the block-scaled sampler's instances (valid, zero, extreme and invalid scale bytes) and on uniform codes and bytes."""
    rng = np.random.default_rng(18)
    params, _ = E.batch.signature_of(TR.word_of(target))
    sampled = K._sample_block_scaled(model)(rng, 20_000, params)
    _agree(target, sampled)
    n_scales = 2
    _agree(target, [rng.integers(0, 1 << 32, a.shape, dtype=np.uint64).astype(np.uint32) if a.dtype == np.uint32 else
                    rng.integers(0, 256 if j >= len(sampled) - n_scales else 16, a.shape, dtype=np.uint64).astype(np.uint8)
                    for j, a in enumerate(sampled)])


@pytest.mark.parametrize("target", DOTS, ids=lambda t: t.id)
def test_the_dot_agrees_with_dot_bf16_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"DotBf16_v2{{K={target.bindings['K']},DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(word)
    _agree(target, K.sample_tc(np.random.default_rng(15), 4000, params))


@pytest.mark.parametrize("target", [AMPERE_K64, HOPPER_K64], ids=lambda t: t.id)
def test_the_coordinate_agrees_with_gemm_coordinate_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"GemmCoordinate_v2{{K=64,DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(word)
    _agree(target, K.sample_tc(np.random.default_rng(13), 4000, params))


@pytest.mark.parametrize("target", GEMMS, ids=lambda t: t.id)
def test_the_gemm_agrees_with_gemm_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"Gemm_v2{{K=32,N=3,DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(TR.word_of(bind(BG.GemmCoordinate, K=32, DOT=target.bindings["DOT"])))
    xs, ws = K.sample_tc(np.random.default_rng(16), 3 * 2000, params)
    _agree(target, [xs.reshape(2000, 3, 32)[:, 0], ws.reshape(2000, 3, 32)])


def test_the_bit_sliced_evaluator_is_the_ir_oracle_here() -> None:
    params, _ = E.batch.signature_of(P.AmpereBF16TcDot16)
    acc, a, b = K.sample_tc(np.random.default_rng(14), 6, params)
    spec = STEPS[0]
    bits = [BS._to_bits(np.asarray(x).reshape(6, -1), BL.elements(t)) for x, (_, t) in zip((acc, a, b), spec.signature()[0])]
    got = BS.evaluate_bits(spec, *bits)
    for i in range(6):
        assert tuple(got[i]) == E.evaluate(spec, *(list(map(int, x[i])) for x in bits))


# ---- the hardware captures ---------------------------------------------------------------------------------------------

def _u16(s: str) -> list[int]:
    return [int(s[i:i + 4], 16) for i in range(0, len(s), 4)]


def _replay_steps(target, cases) -> None:
    acc = np.array([int(c["acc"], 16) for c in cases], np.uint32)
    a = np.array([_u16(c["a"]) for c in cases], np.uint16)
    b = np.array([_u16(c["b"]) for c in cases], np.uint16)
    d = np.array([int(c["d"] if isinstance(c["d"], str) else c["d"][0], 16) for c in cases], np.uint32)
    for s in range(a.shape[1] // 16):
        acc = BS.evaluate_words(target, acc, a[:, 16 * s:16 * s + 16], b[:, 16 * s:16 * s + 16]).astype(np.uint32)
    assert np.array_equal(acc, d)


def test_the_ampere_step_replays_the_a100_and_rtx4090_special_value_records() -> None:
    files = sorted((FIXTURES / "tc-total-2026-09-07").glob("*/mma_special.reduced.json.gz"))
    assert files
    for path in files:
        with gzip.open(path, "rt") as f:
            doc = json.load(f)
        _replay_steps(STEPS[0], doc["cases"] + doc["pos"])


def test_the_hopper_step_replays_the_h100_records() -> None:
    with gzip.open(FIXTURES / "tc-hopper-2026-09-07" / "special.reduced.json.gz", "rt") as f:
        doc = json.load(f)
    _replay_steps(STEPS[1], [c for cases in doc["families"].values() for c in cases])
    with gzip.open(FIXTURES / "tc-hopper-wgmma-bf16-2026-09-26" / "records.json.gz", "rt") as f:
        records = json.load(f)["records"]
    by_steps: dict[int, list] = {}
    for r in records:
        by_steps.setdefault(r["steps"], []).append(r)
    for cases in by_steps.values():
        _replay_steps(STEPS[1], cases)


def test_the_coordinate_replays_the_rtx4090_gemm_captures() -> None:
    root = FIXTURES / "gemm-B1-b3eba618b1b5-20260907T1731Z"
    by_k: dict[int, list] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = json.loads((d / "fixture.json").read_text())
        x = np.frombuffer((d / "x.u16").read_bytes(), "<u2")
        if f["x_arg_cols"] is not None:
            x = x[slice(*f["x_arg_cols"])]
        by_k.setdefault(f["K"], []).append((x, np.frombuffer((d / "w_row.u16").read_bytes(), "<u2"), f["committed_word"]))
    assert sum(map(len, by_k.values())) == 30
    for k, rows in by_k.items():
        x, w = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
        got = BS.evaluate_words(bind(BG.GemmCoordinate, K=k, **BG.GEMM_AMPERE), x, w)
        assert got.tolist() == [r[2] for r in rows]
