"""`verity.ml.circuits.ops.boolean.gemm`: the Boolean BF16 steps, cast, dot and coordinate agree with their word Definitions (`word_of`) on
sampled and special words and on every hardware capture the word Definitions replay, carry exactly their builders' live ANDs,
and are pinned by gate count and program digest."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.kernels import tc as K
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit

FIXTURES = Path(__file__).parent / "fixtures"

AMPERE_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE)
HOPPER_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER)
DOTS = [bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE), bind(BG.DotBf16, K=48, **BG.GEMM_HOPPER)]
STEPS = [bind(BG.AmpereBF16TcDot16), bind(BG.HopperBF16WgmmaDot16)]
GEMMS = [bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE), bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER)]
FP8_STEPS = [bind(BG.HopperE4m3QgmmaDot32), bind(BG.BlackwellE4m3QmmaDot32), bind(BG.HopperE4m3MmaV2Dot32)]
FP8_COORDS = [bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3), bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3),
              bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3_MMAV2)]
FP4 = [(bind(BG.BlackwellNvf4OmmaDot64), K.BLACKWELL_SM120_NVF4), (bind(BG.BlackwellMxf4OmmaDot64), K.BLACKWELL_SM120_MXF4),
       (bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4), K.BLACKWELL_SM120_NVF4),
       (bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4), K.BLACKWELL_SM120_MXF4)]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F2fpBf16_v2": (66, 98, 15, "dc436e6bda068001621acaa798474d37bd955392fd5ae065c6ca6604141cef3a9cff1cfa08fdd43a8d3c3d2989e9acd69285c46e8b7333af6fe78fda5cb34105"),
    "AmpereBF16TcDot16_v3": (8260, 15717, 860, "e679c3424c6b0e1109eb340eb73b7558f1d5c221eaced8fb7b5db6f67c6b455d4474d90042e4ab83d1c048d2bbd635060e12a385a7fcd09a7a08f748bb2ab984"),
    "HopperBF16WgmmaDot16_v2": (7872, 15049, 761, "52c6a2d6ff6a794455edbdfe52486b0c462623883ba079daf68804655826747179940195147309470bef9823bdb1b634849cc849c96ee032f3c266aabbf40aee"),
    AMPERE_K64.id: (33106, 62966, 3455, "fa33a65f8a8bb6490c313f48ead1ec9429912630f4d8805706723238fc6abe52770f3be0e6530e4d4e5bc6391c79c514e47b54b58b3760475cd7567c0d179e3a"),
    HOPPER_K64.id: (31554, 60294, 3059, "d4d47a68284c26cc2768e6d75f6b7fb13bec5c66e96d3222a23983bbf71f0915c88c102d3fb23382209a745fcd3f3a6c4dd4b97e7f1d620fc2bf1182e4d448df"),
    DOTS[0].id: (33040, 62868, 3440, "58ea76fb86ac8d3886074f8063e0671e833c8ddd7123cb4a093408dcf8b718e0965d3753d7b26e67825ff7d6d8179131af090d231134c4fc5a0c26dea93d133d"),
    DOTS[1].id: (23616, 45147, 2283, "4920f9e88fbd049d2039735d3c0ec56f36b7f12e5234fb440d7f101f59d8bfea25425ef80fa009972ab1d6665af88b3e7b29060192edf052ceb22d936c21b8eb"),
    GEMMS[0].id: (49758, 94596, 5205, "ebf2f28fba384ac9bf0e5ebf34d2ec01630b80a47c85361b4f7c9a66de2f94974232b72daf5dcddffa90b60f90df4437bfd54e22f30eaef4a3f3232beb6ba426"),
    GEMMS[1].id: (47430, 90588, 4611, "88d503d3432bd9065d0b82042193c903d7ae9e46ab1a6c820cbdc79a0493252b9ec0802d8089721e24ae4689a79840b63630dd2d90d22223d0347784b3838f50"),
    "HopperE4m3QgmmaDot32_v2": (6799, 12201, 1018, "cd6e8e72b479cf266fc4b40dae6a5fcc0e7771e995fc22e293d73c4d283680e052143abc0d51434e68c766cd275d9532598ea788fa3e1feb62e8a21ab97b2a6b"),
    "BlackwellE4m3QmmaDot32_v2": (9128, 17402, 1025, "558f1f03cff362c3d6f1ba368f67e2a9f24ff62333f2dfc3f269564ffeb27a6f1dfe0d46e46ef465f3739a0d0b482c3a56d8983cd167ffa77f8db5707c3d634d"),
    FP8_COORDS[0].id: (13664, 24500, 2051, "6d1e77347ce33e2669a884c4ac07f73eb6e1e8117bb64d3bf5b734c7618d987cc460ed14200ce69dd23c1c9e703ce883cdb1f3ee81b60c9565b4ca7e38415d08"),
    FP8_COORDS[1].id: (18322, 34902, 2065, "1c55a01aceccd44d8f05082ccf98b593de36ad2a586caf4c05b21c763779dba8d8c9a67ea6ae85fccf55674eb78eac4bcf012f3fd1eadead7f3e05a362f772ef"),
    "HopperE4m3MmaV2Dot32_v2": (9710, 18390, 1105, "39c016593e48e4d8cc13401592ed9a7fc4f538d8011365c10e7f918ab5754562a49e3411a5c98381db4115c4d7c521f937cd29a9419dfaf635e3d79f36af6d87"),
    FP8_COORDS[2].id: (19486, 36878, 2225, "41613b7da3add51bbf8c3daf916732e888aee0a5ec08764a1ca22f0be069e1af1fe3ab78da3e3132f17529e40de6bd603e1f1b4fcf5325f0b6fc2cb36107a60d"),
    "BlackwellE5m2QmmaDot32_v2": (8931, 17231, 1251, "7d92a507ada52b4e9ec533bcd8572a18e18d8cff42a9b3f6f781b3143c53976cac2e80d72734f3d8d78c61b0d9e635cacd827caac6cfd96b8663fc676911bb9a"),
    "BlackwellNvf4OmmaDot64_v2": (6864, 13949, 616, "73168c623c4a783d34f98fee8600b11f5aab5db24730d2da5ce039bd2bcbbaedeef1ddb7f2ea7caef692029eef678f2f24d45a74948c74db541416db324e22a9"),
    "BlackwellMxf4OmmaDot64_v2": (5917, 12058, 605, "f067a7bc2f6718102469b2306b2110a7b3298d4625d47a71ed96af36cf1351b36b619c70df54e916640e6291d118d3d74495eb1e3ba701e311ce10875ea1b653"),
    FP4[2][0].id: (13794, 27996, 1247, "7c16a0433a10a7d45cefc550b28185239cba6387572e75f5a96f2934d0b2456c1e565fd65a89cc673c0cf5892b4854becbaeabeadc977d2990a57bb47c2d37ba"),
    FP4[3][0].id: (11900, 24214, 1225, "c32627d98e0ae22977797a995f9a29f42916bbc8c54255d9d4f64418650a4808fec676a7a26edc0b4fb4295d1c5ee231b17cf9a00f723dc7f55606d406c03f90"),
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
                               BG.HopperE4m3MmaV2Dot32, BG.BlackwellE4m3QmmaDot32, BG.BlackwellE5m2QmmaDot32, BG.BlackwellNvf4OmmaDot64,
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
