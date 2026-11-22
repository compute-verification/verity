"""`verity.circuits.ops.boolean.gemm`: the Boolean BF16 steps, cast, dot and coordinate agree with their word Definitions (`word_of`) on
sampled and special words and on every hardware capture the word Definitions replay, carry exactly their builders' live ANDs,
and are pinned by gate count and program digest.  The k16 step without its finite stand-ins (`HopperBF16WgmmaDot16_v3`, and
`fp.tc_dot16`'s ``stand_ins=False`` at two groups) is the step with them on every special class of each input."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import standalone_call
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits.defs import bind
from verity.circuits.kernels import tc as K
from verity.circuits.ops import prims as P
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import Circuit
from verity.circuits.silicon import fp

from _boolean_checks import assert_agrees, assert_pinned

FIXTURES = Path(__file__).parent / "fixtures"

AMPERE_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE)
HOPPER_K64 = bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER)
HOPPER_V3_K64 = bind(BG.GemmCoordinate, K=64, DOT=BG.HopperBF16WgmmaDot16V3)
DOTS = [bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE), bind(BG.DotBf16, K=48, **BG.GEMM_HOPPER)]
STEPS = [bind(BG.AmpereBF16TcDot16), bind(BG.HopperBF16WgmmaDot16), bind(BG.HopperBF16WgmmaDot16V3)]
GEMMS = [bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE), bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER),
         bind(BG.Gemm, K=32, N=3, DOT=BG.HopperBF16WgmmaDot16V3)]
FP8_STEPS = [bind(BG.HopperE4m3QgmmaDot32), bind(BG.BlackwellE4m3QmmaDot32), bind(BG.HopperE4m3MmaV2Dot32)]
FP8_COORDS = [bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3), bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3),
              bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3_MMAV2)]
FP4 = [(bind(BG.BlackwellNvf4OmmaDot64), K.BLACKWELL_SM120_NVF4), (bind(BG.BlackwellMxf4OmmaDot64), K.BLACKWELL_SM120_MXF4),
       (bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4), K.BLACKWELL_SM120_NVF4),
       (bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4), K.BLACKWELL_SM120_MXF4)]
S8 = [bind(BG.BlackwellS8ImmaDot32), bind(BG.S32ToF32Rn), bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8),
      bind(BG.ScaledMmS8Coordinate, K=64, **BG.GEMM_SM120_S8)]
PADS = [bind(BG.GemmCoordinatePad, T=1, K=32, **BG.GEMM_HOPPER), bind(BG.GemmCoordinatePad, T=17, K=32, **BG.GEMM_HOPPER)]
ROWS = bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8)
RESIDUAL = bind(BG.ScaledMmS8ResidualCoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8)
PVROWS = [bind(BG.PvRow, T=T, K=32, D=64, **BG.GEMM_HOPPER) for T in (1, 8, 32)]
PVROWPAD = bind(BG.PvRowPad, T=32, K=32, D=64, **BG.GEMM_HOPPER)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F2fpBf16_v2": (66, 98, 15, "7286b20c04f767b84d9312470f5763f838731bb47ada79fad0387aaa69bc6a81d91538c28e95df78a2ab4951196ca9f30100af21ef2026d74353491a1d84a1a1"),
    "AmpereBF16TcDot16_v3": (8260, 15717, 860, "84f08edf676a63e6e836a60a904fe5854e85837efbe55ac47d202db6d8f99b5851204510d7f417f24b19a70a5d3ab721105cb1811545596dfd7b0f5ff3bf86d8"),
    "HopperBF16WgmmaDot16_v2": (7872, 15049, 761, "61d4fe673017ab09d24b15b0c2a7410bc29377822c66b94940497a3b10dae4d93e817eb1a8714f00376ca12df8d27ffac2956b40eca87ab8c3065bc385fcf265"),
    "HopperBF16WgmmaDot16_v3": (6887, 14109, 728, "20342a309656a19e459a41c986af6194ea5264010a3f1a94055d712f43660152fb2686d6f7a957fc5808d1ecdf0ef9e8b2f160afd3100abbf37b0220cb2777c2"),
    HOPPER_V3_K64.id: (27614, 56534, 2927, "ef837c61d68e3278a3b623605dd0c0b7d9144671193ba451d2ec13f721de2cb9dfc4fa77c6ca6f5a7e95f846c6402b0bd88c49db89b1305e43b0ff5a4b5e027e"),
    GEMMS[2].id: (41520, 84948, 4413, "a4087af24980bf931cbaf287535854604a6c9eff68536997b0be1b988da6d052450016db34fd6af01676c48c3a7cc0e5885e8ba5a809a99291c91f98208e6636"),
    AMPERE_K64.id: (33106, 62966, 3455, "acb572ed9d49648c5962cc396dd27abd807a5ff3395e7803458d84ddd25df7a5b9b2d52794431dcf701de4f8fe7b9194ddfd296bfe2596a1df3c7980945e07d5"),
    HOPPER_K64.id: (31554, 60294, 3059, "0da0f1748779f731f022f1a61b13b0d2f89c0221df26616d441811bb23e7c95a0e3f9fc23e7b6823324d37063a4b43b5c9e8759656d77d78cf0ff05c58d981f9"),
    DOTS[0].id: (33040, 62868, 3440, "fab543f5257721d4e8f493449265fc63740b51c6cf04426a22154ee322a1c6eda895240203ba6a501fcced84bb55db9b2a11c959946208001c711c214c37c08d"),
    DOTS[1].id: (23616, 45147, 2283, "8970c534f52c7923bf4870ea29d2b26ed528690d75ead82f6aa108a578f250ec4901b385b81daa2afa00f5c10a3f85075c9828b9b60c2b771f9e898c808b4504"),
    GEMMS[0].id: (49758, 94596, 5205, "d52b243020f8fbd5555b35f666f4b30e33e5cba695e617d4d2741a3daec4177597fa29cfc2096f42a48598ea54fe759688c712c04c1b8ffe432a153f98b05193"),
    GEMMS[1].id: (47430, 90588, 4611, "0de6aaefefa4ba84c592d65a8097ae46ac9b7c062cfbd2a5cd2666fa5cff22382cafe7d7ce61eb3d56b3adfb931395b574cb450ec4479ae81328b2ef5ec7ff33"),
    "HopperE4m3QgmmaDot32_v2": (6799, 12201, 1018, "f7110ee66489a1d063bedbe083db2ff0f61b3520721e451a11a2a6466d0a3db4844eea72a1c3671d28eed7a23a2a30c87f29a4588c4cc4098fd5a909f5ae4c96"),
    "BlackwellE4m3QmmaDot32_v2": (9128, 17402, 1025, "ca1796d061e8d432124595c0de9fb05e1ccf9254e1d6290b1efca8f4159884d674c4e3197adb6302d3e45149b4e07b40955a74f7a8bf87fcfe38eac0fab472c6"),
    FP8_COORDS[0].id: (13664, 24500, 2051, "68098d46b5c0ca415531d98d92c4cefbe14a9e22794e6af22846ecd0b6e432960f353632ecef6f72f54cc65785defa03189bace28b6cbf78d2ec0c22d879f3ce"),
    FP8_COORDS[1].id: (18322, 34902, 2065, "28c3125ae650de6c3614fc58ba9ff53b8e2a613297939fbb9ba2a69e11326f3e731778718da1c6f3be004293692f9595c80a27ca5b89790cceedaa969bbd75c4"),
    "HopperE4m3MmaV2Dot32_v2": (9710, 18390, 1105, "16f54c7138654f4f0ee6478f8947c34c399c92f47cef23b1f95591ceb351d5a0b45b8cfedf5e9befc84dd3433ff11d9d09af7bdf260af4a9fbfb0f4bb9e0dcbd"),
    FP8_COORDS[2].id: (19486, 36878, 2225, "d7ee93d43a704f2fa575ef73ebb41d639d1d26c7e509143d03ed17a35220921481263c669cc48b80a84575b6a25eb065baa669569bc90813f3c8e5a41349ca77"),
    "BlackwellE5m2QmmaDot32_v2": (8931, 17231, 1251, "ab49e30386354bd2f3f1bf831eff62dd79d68e794976a5002266395eda9eb4ec5f1c5b6a19d3514bd7c1fafc758d65bc52e1645611f50c12fb20bd11ea94413e"),
    "BlackwellNvf4OmmaDot64_v2": (6864, 13949, 616, "45aea0b1dfbc81a54756e58bd8e6863ca164198717c67c84bfcf5623f6dc2609daeaf37c518d59948d3439239893f17ebb66a3a8d7bb32b48c84feb7bc28247c"),
    "BlackwellMxf4OmmaDot64_v2": (5917, 12058, 605, "3e1fdb45aece9d0264a47da9bb2bed28b037282608fd305a023e6ab2f49e71b0d031f16bd73b7ec22023266673b6d6a49d60580fed3be2c7e52a1338d6e8652a"),
    FP4[2][0].id: (13794, 27996, 1247, "7c83ea1fb19ac8ea6ac695839bc0532ee6a803fd874393494dfc49fc44db5c70fd7031bbf5522b5d6c3ca89ae2d8f6f90946e934cccf88297a895f1ebfba874a"),
    FP4[3][0].id: (11900, 24214, 1225, "4093b29969a5e813a85d11c44a5d1911b410785df43db723da9a2cee9a46864ba28460fc76b524a0e20f9775e6444a4213780dae7310c7301e634bf65a90608b"),
    "BlackwellS8ImmaDot32_v2": (4158, 10373, 474, "bef805a6cf434685fbe58329a1f2cc72c4296232fa0c8ac9235088805015abc8443b5a08f3b773d26022fda91f36678edf3151c604828871d1ba79ac2704e09f"),
    "S32ToF32Rn_v2": (368, 532, 43, "d031724edd4aed2bfef060cef3c82c5c163f17d47ae0cb177a4d02e490d183a8e79e1dbb87d6d1b5c93e0fd2d332b2f7f007b0a947659bed5159a9c3a2a3362f"),
    S8[2].id: (8316, 20746, 948, "688d73d055bd01cfac07a7b910a3f715c9c0e61d75a7eed09f577982d33ba968a29e14d7b61e6a5577eef6f7971df850cfe569a8f8051867d427c20dbd1821a7"),
    S8[3].id: (13546, 30500, 1286, "b2fac213998806e3b39337f6608eb49538583702fce34d2584b5afe1d13452aace869d895a155c3832b1e13dea4c351c6e02c28748422fa32f6f79e45f24be7c"),
    PADS[0].id: (15810, 30196, 1537, "b4f3ea478b5c10babe70ae67c8fa4b8fabbc9b1af0134df38bfcf5611ec8bf6b5406ced2a8daa23f920ee97117364ced20bfd7c611b5a6445ee3beca35e3a751"),
    PADS[1].id: (15810, 30196, 1537, "a7fa036a6826536abc2e777c25fc4700df7515b13f0520af89a131dd55421066f736e0a924d3f5942bec47e171dd528dfaef09f40f73dc57e8628025d3cbba4b"),
    ROWS.id: (13546, 30500, 1286, "46e492d5e13e6c042f9856e7630805f544f00f6d7b9fa882ce8ccffdee59bf3a279e3002d180054f1142ddb380675baf7b19cb37b7cc23b1e84a71b783b81996"),
    RESIDUAL.id: (14263, 31561, 1408, "4456abba14a66237ca8316bc2bca65b020b5b8672470fe710aa780bfd96c895efb2f43002fad386d2519efee08d8733e9f767b40daaddaaf45aef90f874c0612"),
    PVROWS[0].id: (1011840, 1932544, 98368, "1581b40c056c5bc056361494fa2d208f3cfeda807bf7b1b66d59627af80ad4330f9cd0f35d6b4fb3f8b7c9bad3f275ee072b9d82b70be496c825dc835c435a10"),
    PVROWS[1].id: (1011840, 1932544, 98368, "25ee9a04040590ab0c691b6f14b1edd7060d2669f51f9ba9df8c11ab54a4ad8cd8337d5dab6d9cc63d12dc25a956c8e13b660b9cba89a1804646e330a1e81c25"),
    PVROWS[2].id: (1011840, 1932544, 98368, "5915e3aa1bba09c6e8cd68765740d7c9942a3c21ef2d82372d110f3a32716129d855593e7db967ab66413f7cca36a2be0584b27b47a7b42aacab867813755e31"),
    PVROWPAD.id: (1046175, 2002233, 99424, "3d0cc938e92c3b646f49ddac1ad738119e0917feabdb3f9e84778f5d9810804ef61b1b732942d6161d5450f628cd34a5a34a95a0c29ae71db9fe4aad590d1115"),
}
TARGETS = [bind(BG.F2fpBf16), *STEPS, AMPERE_K64, HOPPER_K64, HOPPER_V3_K64, *DOTS, *GEMMS, *FP8_STEPS, *FP8_COORDS, bind(BG.BlackwellE5m2QmmaDot32),
           *(t for t, _ in FP4), *S8, *PADS, ROWS, RESIDUAL, *PVROWS, PVROWPAD]


@pytest.mark.parametrize("target", TARGETS, ids=[t.id for t in TARGETS])
def test_pinned_counts_and_digests(target) -> None:
    assert_pinned(target, PINS)


#: each Boolean GEMM composite, its step width k and the rest of a binding
FAMILIES = [(BG.DotBf16, 16, BG.GEMM_AMPERE), (BG.GemmCoordinate, 16, BG.GEMM_HOPPER), (BG.Gemm, 16, {"N": 3, **BG.GEMM_AMPERE}),
            (BG.GemmCoordinatePad, 16, {"T": 1, **BG.GEMM_HOPPER}), (BG.PvRow, 16, {"T": 1, "D": 2, **BG.GEMM_HOPPER}),
            (BG.PvRowPad, 16, {"T": 1, "D": 2, **BG.GEMM_HOPPER}),
            (BG.GemmCoordinateE4m3, 32, BG.GEMM_HOPPER_E4M3_MMAV2), (BG.GemmCoordinateNvf4, 64, BG.GEMM_SM120_NVF4),
            (BG.GemmCoordinateMxf4, 64, BG.GEMM_SM120_MXF4), (BG.GemmAccumulatorS8, 32, BG.GEMM_SM120_S8),
            (BG.ScaledMmS8Coordinate, 32, BG.GEMM_SM120_S8), (BG.ScaledMmS8CoordinateRows, 32, {"R": 1, **BG.GEMM_SM90_S8}),
            (BG.ScaledMmS8ResidualCoordinateRows, 32, {"R": 1, **BG.GEMM_SM90_S8})]


@pytest.mark.parametrize("defn,k,rest", FAMILIES, ids=[d.name for d, *_ in FAMILIES])
def test_each_composite_refuses_the_k_its_word_definition_refuses(defn, k, rest) -> None:
    """A K that isn't a multiple of the step, and for the s8 composites K = 2^17 (`test_gemm`, `test_integer_mma`)."""
    for bad in (k + 1, 3 * k // 2):
        with pytest.raises(ValueError, match=re.escape(f"{defn.id} refuses this binding: K={bad} is not a multiple of the "
                                                       f"{k}-wide step")):
            bind(defn, K=bad, **rest)
    if defn in (BG.GemmAccumulatorS8, BG.ScaledMmS8Coordinate, BG.ScaledMmS8CoordinateRows):
        with pytest.raises(ValueError, match=re.escape(f"{defn.id} refuses this binding: K=131072 is at or past 131072")):
            bind(defn, K=1 << 17, **rest)
    if defn is BG.ScaledMmS8ResidualCoordinateRows:
        with pytest.raises(ValueError, match=re.escape(f"{defn.id} refuses this binding: K=131072 is at or past 131072")):
            bind(defn, K=1 << 17, **rest)


#: the s8 composites at K = 2^17 - 32, the largest they admit: (gates, standalone program digest), recorded before `bind`
#: refused any K
S8_EDGE = {
    bind(BG.GemmAccumulatorS8, K=131040, **BG.GEMM_SM120_S8): (61445476, "a314e2ffeccfd23a3e675426f3efe0e81235dd62b23b0c05f83cec6159e3e669a11b5ca88bcf5dd7ebf5087dcbee63315a9b626919e5905870b794e1356a110c"),
    bind(BG.ScaledMmS8Coordinate, K=131040, **BG.GEMM_SM120_S8): (61460798, "ccdb59a278334781ac2c8a96ebb41a885acfb3c12f486e8b8111106f5afc130ef319c5718966f93c20d5a30b1918421ef57ce63ae922925f83daa67c6ebe2cc7"),
}


@pytest.mark.parametrize("spec", S8_EDGE, ids=[s.id for s in S8_EDGE])
def test_the_s8_composites_bind_their_largest_k_to_its_recorded_digest(spec) -> None:
    prog, _ = standalone_call(spec)
    assert (spec.gates, codec.program_digest(codec.encode_program(prog))) == S8_EDGE[spec]


@pytest.mark.parametrize("d", [BG.F2fpBf16, BG.AmpereBF16TcDot16, BG.HopperBF16WgmmaDot16, BG.HopperBF16WgmmaDot16V3,
                               BG.HopperE4m3QgmmaDot32, BG.HopperE4m3MmaV2Dot32, BG.BlackwellE4m3QmmaDot32, BG.BlackwellE5m2QmmaDot32,
                               BG.BlackwellNvf4OmmaDot64, BG.BlackwellMxf4OmmaDot64, BG.BlackwellS8ImmaDot32, BG.S32ToF32Rn],
                         ids=lambda d: f"{d.name}_v{d.version}")
def test_the_ands_are_the_builders_live_ands(d) -> None:
    """The traced circuit has one And gate per AND the builder keeps live, with common ANDs shared (`Circuit(cse=True)`),
    but for its commits (`Circuit.commit`, a row with B = 1 and no gate; `AND` folds every other constant operand)."""
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    commits = sum(k == "and" and v[1].is_const() for k, v in zip(C.kind, C.data))
    assert T.counts["and"] == C.counts()["ands"] - commits == PINS[f"{d.name}_v{d.version}"][0]


def test_the_cast_agrees_with_f2fp_v1() -> None:
    rng = np.random.default_rng(11)
    hi = np.repeat(np.arange(1 << 16, dtype=np.uint32), 4) << np.uint32(16)
    lo = np.tile(np.array([0, 0x7FFF, 0x8000, 0x8001], np.uint32), 1 << 16)
    words = np.concatenate([hi | lo, rng.integers(0, 1 << 32, 200_000, dtype=np.uint64).astype(np.uint32)])
    assert_agrees(bind(BG.F2fpBf16), [words])


@pytest.mark.parametrize("target", STEPS, ids=[t.id for t in STEPS])
def test_the_steps_agree_with_their_word_steps(target) -> None:
    params, _ = E.batch.signature_of(TR.word_of(target))
    assert_agrees(target, K.sample_tc(np.random.default_rng(12), 20_000, params))


#: one word of each special class in both signs: NaNs (quiet, signalling, all-ones payload), the infinities, the zeros, the least
#: and largest subnormals, and the largest finite word (the last two)
SPECIAL_BF16 = np.array([0x7FC0, 0x7F81, 0x7FFF, 0xFFC0, 0xFF81, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x007F, 0x8001,
                         0x807F, 0x7F7F, 0xFF7F], np.uint16)
SPECIAL_F32 = np.array([0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001, 0x7F800000, 0xFF800000, 0x00000000,
                        0x80000000, 0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
#: `fp.tc_dot16`'s ``stand_ins=False`` at the Ampere step's two groups (8, 8), where an earlier group's overflow meets a later
#: group's non-finite operand; not registered, it computes `AmpereBF16TcDot16_v3`'s function or this test fails
AMPERE_RAW = TR.definition("AmpereBF16TcDot16Raw", 1,
                           lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), stand_ins=False),
                           (("acc", BG.F32), ("a", BG.A16), ("b", BG.A16)), BG.F32, word=P.AmpereBF16TcDot16, register=False)


def _finite_words(rng, shape, bits: int) -> np.ndarray:
    """Finite words of ``bits`` (16 or 32): a random sign and mantissa, the exponent uniform over the finite ones for half the
    words and within 8 of the bias for the rest."""
    man = 7 if bits == 16 else 23
    e = np.where(rng.random(shape) < 0.5, rng.integers(0, 255, shape), rng.integers(119, 136, shape)).astype(np.uint64)
    w = (rng.integers(0, 2, shape).astype(np.uint64) << np.uint64(bits - 1)) | (e << np.uint64(man)) \
        | rng.integers(0, 1 << man, shape).astype(np.uint64)
    return w.astype(np.uint16 if bits == 16 else np.uint32)


def _backgrounds(rng, n: int, n_max: int) -> list[np.ndarray]:
    """[acc, a, b]: n rows of finite words, then n_max rows of the largest finite words in random signs, whose products
    overflow every group (an infinity from finite operands)."""
    return [np.concatenate([_finite_words(rng, (n, *s), bits), rng.choice(sp[-2:], (n_max, *s))])
            for s, bits, sp in (((), 32, SPECIAL_F32), ((16,), 16, SPECIAL_BF16), ((16,), 16, SPECIAL_BF16))]


def _special_cases(rng) -> list[np.ndarray]:
    """[acc, a, b] of the stand-in equivalence: each special word at each of the 33 inputs over 16 finite and 4 overflowing
    rows; every pair of special words at one lane (a[i], b[i]) for three lanes, at two lanes of a in different Ampere groups,
    and at the accumulator and a lane, each over 4 finite and 2 overflowing rows; then 25,600 rows whose words are each
    special at a rate of 0, 2, 10 or 30%, and otherwise uniform or finite."""
    rows: list[list[np.ndarray]] = []
    for s in SPECIAL_F32:
        acc, a, b = _backgrounds(rng, 16, 4)
        rows.append([np.full_like(acc, s), a, b])
    for pos in range(32):
        for s in SPECIAL_BF16:
            acc, a, b = _backgrounds(rng, 16, 4)
            (a if pos < 16 else b)[:, pos % 16] = s
            rows.append([acc, a, b])
    for s in SPECIAL_BF16:
        for t in SPECIAL_BF16:
            for i, j in ((0, 16), (9, 25), (15, 31), (2, 13)):
                acc, a, b = _backgrounds(rng, 4, 2)
                ab = np.concatenate([a, b], axis=1)
                ab[:, i], ab[:, j] = s, t
                rows.append([acc, ab[:, :16], ab[:, 16:]])
    for s in SPECIAL_F32:
        for t in SPECIAL_BF16:
            acc, a, b = _backgrounds(rng, 4, 2)
            acc[:], b[:, 5] = s, t
            rows.append([acc, a, b])
    n = 25_600
    rate = np.repeat([0.0, 0.02, 0.1, 0.3], n // 4)
    sampled = []
    for shape, bits, sp in (((n,), 32, SPECIAL_F32), ((n, 16), 16, SPECIAL_BF16), ((n, 16), 16, SPECIAL_BF16)):
        uniform = rng.integers(0, 1 << bits, shape, dtype=np.uint64).astype(sp.dtype)
        w = np.where(rng.random(shape) < 0.5, uniform, _finite_words(rng, shape, bits))
        sampled.append(np.where(rng.random(shape) < rate.reshape(-1, *([1] * (len(shape) - 1))), rng.choice(sp, shape), w))
    rows.append(sampled)
    return [np.concatenate([r[k] for r in rows]) for k in range(3)]


@pytest.mark.slow
@pytest.mark.parametrize("stand_in,raw", [(STEPS[1], STEPS[2]), (STEPS[0], bind(AMPERE_RAW))],
                         ids=["HopperBF16WgmmaDot16_v3", "tc_dot16-raw-at-groups-8-8"])
def test_the_step_without_its_stand_ins_is_the_step_on_every_special_class(stand_in, raw) -> None:
    """`fp.tc_dot16`'s argument that the finite stand-ins never reach the output, on words: without them the step equals the
    step with them, and its word step, on every special class of each input (`_special_cases`).  The cases reach each branch
    of the special select: the NaN, both infinities, an infinity from finite operands alone, and finite outputs."""
    acc, a, b = _special_cases(np.random.default_rng(41))
    got = BS.evaluate_words(raw, acc, a, b).astype(np.uint64)
    want = BS.evaluate_words(stand_in, acc, a, b).astype(np.uint64)
    bad = np.flatnonzero(got.reshape(len(acc)) != want.reshape(len(acc)))
    assert not bad.size, f"{raw.id}: {bad.size} of {len(acc)} differ from {stand_in.id}, first {bad[:5]}"
    assert_agrees(raw, [acc, a, b])
    out = got.reshape(len(acc)).astype(np.uint32)
    finite = ((acc & 0x7F800000) != 0x7F800000) & ((a & 0x7F80) != 0x7F80).all(1) & ((b & 0x7F80) != 0x7F80).all(1)
    inf = (out & 0x7FFFFFFF) == 0x7F800000
    assert {0x7FFFFFFF, 0x7F800000, 0xFF800000} <= set(out.tolist())
    assert (finite & inf).sum() > 5000 and (~inf & (out != 0x7FFFFFFF)).sum() > 3000


def _finite_e4m3(rng, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """``inputs`` with every E4M3 operand byte sampled uniformly from the finite bytes (0x7F and 0xFF are E4M3's NaNs)."""
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
    assert_agrees(target, sampled)
    assert_agrees(target, _finite_e4m3(rng, sampled))


def test_the_e5m2_step_agrees_with_its_word_step() -> None:
    """On `sample_tc_e5m2`'s instances, on uniform words, and on uniform finite E5M2 operands (exponent field not all ones)."""
    rng = np.random.default_rng(19)
    target = bind(BG.BlackwellE5m2QmmaDot32)
    params, _ = E.batch.signature_of(TR.word_of(target))
    sampled = K.sample_tc_e5m2(rng, 20_000, params)
    assert_agrees(target, sampled)
    uniform = [rng.integers(0, 1 << (32 if a.dtype == np.uint32 else 8), a.shape, dtype=np.uint64).astype(a.dtype) for a in sampled]
    assert_agrees(target, uniform)
    assert_agrees(target, [a if a.dtype == np.uint32 else np.where((a & 0x7C) == 0x7C, a & 0xFB, a).astype(np.uint8) for a in uniform])


@pytest.mark.parametrize("target,model", FP4, ids=[t.id for t, _ in FP4])
def test_the_fp4_steps_and_coordinates_agree_with_their_word_definitions(target, model) -> None:
    """On the block-scaled sampler's instances (valid, zero, extreme and invalid scale bytes) and on uniform codes and bytes."""
    rng = np.random.default_rng(18)
    params, _ = E.batch.signature_of(TR.word_of(target))
    sampled = K._sample_block_scaled(model)(rng, 20_000, params)
    assert_agrees(target, sampled)
    n_scales = 2
    assert_agrees(target, [rng.integers(0, 1 << 32, a.shape, dtype=np.uint64).astype(np.uint32) if a.dtype == np.uint32 else
                    rng.integers(0, 256 if j >= len(sampled) - n_scales else 16, a.shape, dtype=np.uint64).astype(np.uint8)
                    for j, a in enumerate(sampled)])


def _s8_operands(rng, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform s8 bytes; every fourth instance's operands are sampled from {-128, -127, 127} (the largest products)."""
    a = rng.integers(0, 256, (n, k), dtype=np.uint16).astype(np.uint8)
    b = rng.integers(0, 256, (n, k), dtype=np.uint16).astype(np.uint8)
    big = np.array([0x80, 0x81, 0x7F], np.uint8)
    a[::4], b[::4] = rng.choice(big, a[::4].shape), rng.choice(big, b[::4].shape)
    return a, b


def test_the_s8_step_agrees_with_its_word_step_and_clamps_at_d() -> None:
    """On uniform accumulators and on accumulators within 2^16 of either s32 edge, where the sum often leaves s32 (the clamp)."""
    rng = np.random.default_rng(21)
    n = 20_000
    acc = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    edge = rng.integers(0, 1 << 16, n // 2, dtype=np.uint64).astype(np.uint32)
    acc[: n // 4] = 0x7FFFFFFF - edge[: n // 4]
    acc[n // 4: n // 2] = 0x80000000 + edge[n // 4:]
    a, b = _s8_operands(rng, n, 32)
    assert_agrees(S8[0], [acc, a, b])
    exact = acc.view(np.int32).astype(np.int64) + (a.view(np.int8).astype(np.int64) * b.view(np.int8)).sum(1)
    got = BS.evaluate_words(S8[0], acc, a, b).astype(np.uint32).view(np.int32)
    assert np.array_equal(got, np.clip(exact, -(1 << 31), (1 << 31) - 1))
    assert ((exact > (1 << 31) - 1).sum() > 500) and ((exact < -(1 << 31)).sum() > 500)


def test_the_s32_cast_agrees_with_its_word_cast_and_numpy() -> None:
    rng = np.random.default_rng(22)
    words = np.concatenate([rng.integers(0, 1 << 32, 100_000, dtype=np.uint64).astype(np.uint32),
                            (np.uint32(1) << np.arange(32, dtype=np.uint32)) + np.uint32(1),
                            np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x01000001, 0x01000003], np.uint32)])
    assert_agrees(S8[1], [words])
    assert np.array_equal(BS.evaluate_words(S8[1], words).astype(np.uint32), words.view(np.int32).astype(np.float32).view(np.uint32))


def test_the_s8_accumulator_and_coordinate_agree_with_their_word_definitions() -> None:
    """On s8 rows, with per-token and per-channel scales of an int8 linear's range and with uniform f32 scale words."""
    rng = np.random.default_rng(23)
    n = 2000
    x, w = _s8_operands(rng, n, 64)
    assert_agrees(S8[2], [x, w])
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = (np.abs(rng.standard_normal(n)) * 0.02 / 127).astype(np.float32).view(np.uint32)
    assert_agrees(S8[3], [x, sx, w, sw])
    u = rng.integers(0, 1 << 32, (2, n), dtype=np.uint64).astype(np.uint32)
    assert_agrees(S8[3], [x, u[0], w, u[1]])


def test_the_row_split_s8_coordinate_agrees_with_its_word_definition_and_is_the_unsplit_coordinate() -> None:
    """`ScaledMmS8CoordinateRows_v2{K=64, R=2}` on x's and W's row's halves is `ScaledMmS8Coordinate_v2{K=64}` on the whole rows."""
    assert TR.word_of(ROWS).id == "ScaledMmS8CoordinateRows_v1{K=64,R=2,DOT=BlackwellS8ImmaDot32_v1}"
    rng = np.random.default_rng(26)
    n = 1000
    x, w = _s8_operands(rng, n, 64)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    parts = [x[:, :32], x[:, 32:], sx, w[:, :32], w[:, 32:], sw]
    assert_agrees(ROWS, parts)
    assert np.array_equal(BS.evaluate_words(ROWS, *parts), BS.evaluate_words(S8[3], x, sx, w, sw))


def test_the_residual_coordinate_agrees_with_its_word_definition_and_is_the_coordinate_then_the_add() -> None:
    """`ScaledMmS8ResidualCoordinateRows_v2{K=64, R=2}` is `Bf16AddF2fp_v2(r, ScaledMmS8CoordinateRows_v2{K=64, R=2})`, r
    random, special and the coordinate's word negated."""
    from verity.circuits.ops.boolean import elementwise as EW

    assert TR.word_of(RESIDUAL).id == "ScaledMmS8ResidualCoordinateRows_v1{K=64,R=2,DOT=BlackwellS8ImmaDot32_v1}"
    rng = np.random.default_rng(36)
    n = 1000
    x, w = _s8_operands(rng, n, 64)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = (np.abs(rng.standard_normal(n)) * 0.02 / 127).astype(np.float32).view(np.uint32)
    parts = [x[:, :32], x[:, 32:], sx, w[:, :32], w[:, 32:], sw]
    y = BS.evaluate_words(ROWS, *parts).reshape(n).astype(np.uint16)
    special = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81], np.uint16)
    r = rng.integers(0, 1 << 16, n, dtype=np.uint32).astype(np.uint16)
    r[:len(special)], r[len(special):2 * len(special)] = special, y[len(special):2 * len(special)] ^ np.uint16(0x8000)
    assert_agrees(RESIDUAL, parts + [r])
    assert np.array_equal(BS.evaluate_words(RESIDUAL, *parts, r).reshape(n), BS.evaluate_words(EW.Bf16AddF2fp, r, y).reshape(n))


def test_the_padded_coordinate_agrees_with_its_word_definition_and_is_the_coordinate_on_zero_padded_rows() -> None:
    """`GemmCoordinatePad_v2{T, K=32}` on T words is `GemmCoordinate_v3{K=32}` on them followed by K - T words of +0 bits."""
    coord = bind(BG.GemmCoordinate, K=32, **BG.GEMM_HOPPER)
    params, _ = E.batch.signature_of(TR.word_of(coord))
    xs, ws = K.sample_tc(np.random.default_rng(17), 2000, params)
    for T in (1, 5, 16, 17, 31, 32):
        target = bind(BG.GemmCoordinatePad, T=T, K=32, **BG.GEMM_HOPPER)
        assert TR.word_of(target).id == f"GemmCoordinatePad_v1{{T={T},K=32,DOT=HopperBF16WgmmaDot16_v1}}"
        assert_agrees(target, [xs[:, :T], ws[:, :T]])

        def pad(a):
            return np.concatenate([a[:, :T], np.zeros((len(a), 32 - T), a.dtype)], axis=1)

        assert np.array_equal(BS.evaluate_words(target, xs[:, :T], ws[:, :T]), BS.evaluate_words(coord, pad(xs), pad(ws)))


def test_the_pv_row_agrees_with_its_word_definition_and_is_the_padded_coordinate_on_each_column() -> None:
    """`PvRow_v2{T, K=32, D=64}` at T = 1, 8 and 32 agrees with `PvRow_v1`, and its dim d is `GemmCoordinatePad_v2{T, K=32}` of
    p and (v0[d], ..., v{T-1}[d])."""
    params, _ = E.batch.signature_of(TR.word_of(bind(BG.GemmCoordinate, K=32, **BG.GEMM_HOPPER)))
    n, D = 300, 64
    xs, ws = K.sample_tc(np.random.default_rng(27), n * D, params)
    for target in PVROWS:
        T = target.bindings["T"]
        assert TR.word_of(target).id == f"PvRow_v1{{T={T},K=32,D=64,DOT=HopperBF16WgmmaDot16_v1}}"
        p, v = xs[:n, :T], ws.reshape(n, D, 32)[:, :, :T].transpose(0, 2, 1)
        vs = [np.ascontiguousarray(v[:, t]) for t in range(T)]
        assert_agrees(target, [p, *vs])
        pad = bind(BG.GemmCoordinatePad, T=T, K=32, **BG.GEMM_HOPPER)
        cols = np.stack([BS.evaluate_words(pad, p, np.ascontiguousarray(v[:, :, d])).reshape(n) for d in range(D)], axis=1)
        assert np.array_equal(BS.evaluate_words(target, p, *vs).reshape(n, D), cols)


def test_the_padded_pv_row_agrees_with_its_word_definition_at_every_key_count() -> None:
    """`PvRowPad_v2{T=32, K=32, D=64}` agrees with `PvRowPad_v1` on 300 heads whose key counts n run over 0..33 and a few
    u16 words past them (0x7FFF, 0x8000, 0x8020 and 0xFFFF among them), with inf p words and inf and NaN V words past n; at
    n in {1, 2, 5, 17, 31, 32} its words are `PvRow_v2{T=n}`'s on the first n keys."""
    params, _ = E.batch.signature_of(TR.word_of(bind(BG.GemmCoordinate, K=32, **BG.GEMM_HOPPER)))
    heads, D, T = 300, 64, 32
    xs, ws = K.sample_tc(np.random.default_rng(28), heads * D, params)
    p, v = xs[:heads].copy(), ws.reshape(heads, D, 32).transpose(0, 2, 1).copy()
    counts = np.concatenate([np.arange(heads - 6) % (T + 2), [T + 1, 1 << 8, 0x7FFF, 0x8000, 0xFFFF, 0x8020]]).astype(np.uint16)
    past = np.arange(T)[None, :] >= counts[:, None]
    p[past & (np.arange(T) % 3 == 0)] = 0x7F80
    v[past, 0], v[past, 1] = 0x7F80, 0x7FC0
    vs = [np.ascontiguousarray(v[:, t]) for t in range(T)]
    assert_agrees(PVROWPAD, [p, *vs, counts])
    got = BS.evaluate_words(PVROWPAD, p, *vs, counts).reshape(heads, D)
    for n in (1, 2, 5, 17, 31, 32):
        at = np.flatnonzero(counts == n)
        row = bind(BG.PvRow, T=n, K=32, D=64, **BG.GEMM_HOPPER)
        assert np.array_equal(got[at], BS.evaluate_words(row, p[at, :n], *(x[at] for x in vs[:n])).reshape(len(at), D)), n


def test_the_scaled_s8_matmul_agrees_with_scaled_mm_s8_v1() -> None:
    """`ScaledMmS8_v2` at K = 64, N = 3 on s8 rows under an int8 linear's scales, and under uniform and special f32 scale words
    (±0, the least subnormal, the largest finite, ±inf, a quiet and a signalling NaN)."""
    target = bind(BG.ScaledMmS8, K=64, N=3, **BG.GEMM_SM120_S8)
    assert TR.word_of(target).id == "ScaledMmS8_v1{K=64,N=3,DOT=BlackwellS8ImmaDot32_v1}"
    rng = np.random.default_rng(24)
    n = 600
    x, w = _s8_operands(rng, n, 64)
    w = np.stack([w, np.roll(w, 1, axis=0), np.roll(w, 2, axis=0)], axis=1)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = (np.abs(rng.standard_normal((n, 3))) * 0.02 / 127).astype(np.float32).view(np.uint32)
    assert_agrees(target, [x, sx, w, sw])
    special = np.array([0, 0x80000000, 1, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001], np.uint32)
    assert_agrees(target, [x, rng.choice(special, n), w, rng.choice(special, (n, 3))])


@pytest.mark.parametrize("target", DOTS, ids=lambda t: t.id)
def test_the_dot_agrees_with_dot_bf16_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"DotBf16_v2{{K={target.bindings['K']},DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(word)
    assert_agrees(target, K.sample_tc(np.random.default_rng(15), 4000, params))


@pytest.mark.parametrize("target", [AMPERE_K64, HOPPER_K64, HOPPER_V3_K64], ids=lambda t: t.id)
def test_the_coordinate_agrees_with_gemm_coordinate_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"GemmCoordinate_v2{{K=64,DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(word)
    assert_agrees(target, K.sample_tc(np.random.default_rng(13), 4000, params))


@pytest.mark.parametrize("target", GEMMS, ids=lambda t: t.id)
def test_the_gemm_agrees_with_gemm_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"Gemm_v2{{K=32,N=3,DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(TR.word_of(bind(BG.GemmCoordinate, K=32, DOT=target.bindings["DOT"])))
    xs, ws = K.sample_tc(np.random.default_rng(16), 3 * 2000, params)
    assert_agrees(target, [xs.reshape(2000, 3, 32)[:, 0], ws.reshape(2000, 3, 32)])


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


@pytest.mark.parametrize("step", STEPS[1:], ids=lambda t: t.id)
def test_the_hopper_step_replays_the_h100_records(step) -> None:
    with gzip.open(FIXTURES / "tc-hopper-2026-09-07" / "special.reduced.json.gz", "rt") as f:
        doc = json.load(f)
    _replay_steps(step, [c for cases in doc["families"].values() for c in cases])
    with gzip.open(FIXTURES / "tc-hopper-wgmma-bf16-2026-09-26" / "records.json.gz", "rt") as f:
        records = json.load(f)["records"]
    by_steps: dict[int, list] = {}
    for r in records:
        by_steps.setdefault(r["steps"], []).append(r)
    for cases in by_steps.values():
        _replay_steps(step, cases)


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


def test_the_sm90_binding_on_bits_is_hoppers_step() -> None:
    """`BG.GEMM_SM90_S8` binds the Boolean step whose word step is `G.GEMM_SM90_S8`'s, and that step on bits is
    `HOPPER_SM90_S8_K32.step` on uniform accumulators and on accumulators within 2^16 of either s32 edge (clamps included);
    the accumulator on bits at K = 512, the least K H1 serves on sm_90, is that step chained."""
    from verity.circuits.ops import gemm as G
    from verity.circuits.silicon.integer import HOPPER_SM90_S8_K32

    assert BG.GEMM_SM90_S8 == BG.GEMM_SM120_S8 == BG.s8_gemm(HOPPER_SM90_S8_K32)
    assert TR.word_of(bind(BG.GEMM_SM90_S8["DOT"])).id == G.GEMM_SM90_S8["DOT"].id
    rng = np.random.default_rng(24)
    n = 4000
    acc = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    edge = rng.integers(0, 1 << 16, n // 2, dtype=np.uint64).astype(np.uint32)
    acc[: n // 4] = 0x7FFFFFFF - edge[: n // 4]
    acc[n // 4: n // 2] = 0x80000000 + edge[n // 4:]
    a, b = _s8_operands(rng, n, 32)
    got = BS.evaluate_words(bind(BG.GEMM_SM90_S8["DOT"]), acc, a, b).astype(np.uint32)
    want = [HOPPER_SM90_S8_K32.step(int(c), x.tolist(), y.tolist()) for c, x, y in zip(acc, a, b)]
    assert got.tolist() == want
    assert sum(w in (0x7FFFFFFF, 0x80000000) for w in want) > 200
    x, w = _s8_operands(rng, 200, 512)
    got = BS.evaluate_words(bind(BG.GemmAccumulatorS8, K=512, **BG.GEMM_SM90_S8), x, w).astype(np.uint32)
    for r in range(len(x)):
        c = 0
        for i in range(0, 512, 32):
            c = HOPPER_SM90_S8_K32.step(c, x[r, i:i + 32].tolist(), w[r, i:i + 32].tolist())
        assert int(got[r]) == c
