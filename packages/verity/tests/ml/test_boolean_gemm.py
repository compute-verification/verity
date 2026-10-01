"""`verity.ml.boolean.gemm`: the Boolean BF16 steps, cast, dot and coordinate agree with their word Definitions (`word_of`) on
sampled and special words and on every hardware capture the word Definitions replay, carry exactly their builders' live ANDs,
and are pinned by gate count and program digest."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import REGISTRY, bind
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
}
TARGETS = [bind(BG.F2fpBf16), *STEPS, AMPERE_K64, HOPPER_K64, *DOTS, *GEMMS]


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


@pytest.mark.parametrize("d", [BG.F2fpBf16, BG.AmpereBF16TcDot16, BG.HopperBF16WgmmaDot16], ids=lambda d: d.name)
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
