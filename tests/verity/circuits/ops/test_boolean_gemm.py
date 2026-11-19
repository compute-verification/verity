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
    "F2fpBf16_v2": (66, 98, 15, "72cc5a81dda9ea68111e29ad0033ab0b6dd8bc1a9a5e610c36f0ae1096db84f6e8a9067b47cd31c2763f918eb4fa7f28cadb7a767b6ed004e8257c2cbfff1c41"),
    "AmpereBF16TcDot16_v3": (8260, 15717, 860, "c6aa30933fb47184aa9c52074968fd202140717f37f4196c1c8c153679ff8efb45589a961624bee235677f59966e01e405bfdde52942ffa25b4113d9659ea335"),
    "HopperBF16WgmmaDot16_v2": (7872, 15049, 761, "a6049f968fe675724f2b0299f18072b23d9ba28bf4d0e368d8d64b140d59c55fed082c7fa83c8664274bbb38d8e2d8e6edc7477918291876a87c4dc505da3add"),
    "HopperBF16WgmmaDot16_v3": (6887, 14109, 728, "aeca699e48252096a3a53ca1a724696f0ae1405780f8ce1002d47056909a9493c9b3a27b2b8349f8d258b40922734f1e8e0833e74339591a6ff4c76f992f256b"),
    HOPPER_V3_K64.id: (27614, 56534, 2927, "91dc161845d5469b48e213c92f0a1b492a051a82430e1d725e4cdf560470f5de88b9669410babe0d6e1c0a40bb6eb9a552833c2284919a60a51db44e83d6e153"),
    GEMMS[2].id: (41520, 84948, 4413, "f8546cf1633110c21f6b8ccc6c86f3c589722d1608c0d0c5a1f621461d0727b7197f882205e025d3cd2c7a349bb1a9eea29e727949e68581d8f8965c6f13f1bd"),
    AMPERE_K64.id: (33106, 62966, 3455, "73df0772a8d30cf38441e9079e3a3e9a92f592692562c0fed9b7be61e59df795d31a35319987970ef46c6fc9e45db34d4fbc6f99a7919773cdf56c2ba8386979"),
    HOPPER_K64.id: (31554, 60294, 3059, "93d812a61e6d57227b7a722f7b10b6bbba6e4f5bdd51bf302e565ca4e24ddd81d284269769fcbf7281c734c6af26047981fd3e927583704ade7cddee1677e5e5"),
    DOTS[0].id: (33040, 62868, 3440, "341433c8da642266ed982a63e3f9290a152f5e849096089312f57596972818dc77fd1f0dd0f24cbad0f802db1884e201be5693aba5241ef09bac43a10d22d417"),
    DOTS[1].id: (23616, 45147, 2283, "411f6dced3f93e415c8cdf8bf77fd08aa5c5624b8abde3146bdb4a49543adf4382d8e00ae417e3062a6be5029fc1fc37dd6ebe9692bca53d46def85aba3e8b67"),
    GEMMS[0].id: (49758, 94596, 5205, "94e20c8605a7bd988d68bfa4a53dc686b6e25dbfde3249102ea2ee5f0ba4acf2c1cddbc7fe982613661e1c97d72451498ede0a98613f5be8318919e86ebd15cb"),
    GEMMS[1].id: (47430, 90588, 4611, "1c1f0860c31837ad070a97b97314df468cb29acd4f46a8fc4ed7264c3a2e5984039814b69141a0b186e618b9fd6a3b56fbcdb9f89a98e25bd933627d5121e7b3"),
    "HopperE4m3QgmmaDot32_v2": (6799, 12201, 1018, "1c8c15ec351d8e6acc909ac6ec93b3d9b1940ae3bfc6c14d926cf88b367b217d8f4ad344ced4cba80e3d09fe0274a01f4c49b0baf055d9fe9b062e1c630b8899"),
    "BlackwellE4m3QmmaDot32_v2": (9128, 17402, 1025, "2f6f609f8f24cff3eca7ddbcacf0ac6e8fbc08bdf8d69a1c7977d924fcebd24ac5bb9f0f39b9477ed2ad0657347785c12dcb2e7cba8b21f22d7076a8ec916088"),
    FP8_COORDS[0].id: (13664, 24500, 2051, "072ef37da9c2dcb0d9347cbf8e1ff827d2f7f750782bf9e61b53bf91024d669b9de28bd3ef8e0e9220714f343e87fd76c3f547232a48a5e1bb3664a2dbb695d8"),
    FP8_COORDS[1].id: (18322, 34902, 2065, "9a7c7b2b2bd89c852d40dc1bc400ec40697cdaae2a27aa017baf79a73ea8f129eb5a89b2cf034ee37fbb55197b48e596f10709ead2e69df727d164b198cdefc8"),
    "HopperE4m3MmaV2Dot32_v2": (9710, 18390, 1105, "53e5e1da974227c8000798f29cbff2c350f948703fd8a846ff65da3d54611ecdb261f870b5fa52010373f79b0b542070bec66f176d1e992366dbdb4c8ef3130b"),
    FP8_COORDS[2].id: (19486, 36878, 2225, "eb58b98e64ab4cd0c8d098c29c5d6a9db05a32038e2b7864b8db030748ff37af4ffcf3e0b84148181bb9073ceb199e0ab0409f1ea1e0deb25551fe224de18247"),
    "BlackwellE5m2QmmaDot32_v2": (8931, 17231, 1251, "dbb8b7c6f173336ca49b943d58d3c0fa7fcf979f39e734994bda10b8fdce79b33bbdc116c0a064d1c7e371d97594e91d668c7d0d43643b77a9c434a6a8daae97"),
    "BlackwellNvf4OmmaDot64_v2": (6864, 13949, 616, "ac8af4e948d86cafefe728a6d727820d8ac69323976b4a792f0c9333b1c8932c34de8e2af18ae3f9d2b37bc29aa7b50ca3086784c7e68b00113ff36d26a719cc"),
    "BlackwellMxf4OmmaDot64_v2": (5917, 12058, 605, "d42f357f5b865505efff0e6aff243e83b1ecace393ade385496a5b66a905f50d853d377d63bd61d49e1910682a2f77b9413d6f1a93c146401c832b0a15b3367a"),
    FP4[2][0].id: (13794, 27996, 1247, "7f1c5297d8704924a0429f185baf2d50087f649ec880546518eca4f5b7bd16f595dcc7f914946237b3b355cc968a9ee0f2d3060342e2f261e66b5bbcb62a44b3"),
    FP4[3][0].id: (11900, 24214, 1225, "f93f0169b4b153a15887ebe581ef3beb947b4e93f41325f4021252724ac3dccd5d3c571eb671cec34deff5aea942aa1a4d406806e948f74c1273a17fc02e9621"),
    "BlackwellS8ImmaDot32_v2": (4158, 10373, 474, "2378f82ebf3bae4cd8c8e06c4839e7150b8a43490d194c23e46c58986052e783d3e013a7cd85fbe472c3337d5ac9cd118601220aa4f3317453eba90ec139b702"),
    "S32ToF32Rn_v2": (368, 532, 43, "2c20577c64d3e9ffb541e614e4698e7aa2ed77739f83d8c38114695182b3ed7275b35729553fa709c1c29b7ed87e8614c36f13ef7ea2b184ba4415d312ff1d42"),
    S8[2].id: (8316, 20746, 948, "7b9c3283025a3ffa01f8b3fa8a9aa271fb6e79e8fc4eba956382afeb2bb67e249eac5e955e52ea5f60d402b8c4796bfecfaedf10fc5845c1bcabe6aebccefdd4"),
    S8[3].id: (13546, 30500, 1286, "f90eb247e74557606de78b930112a9a9e8740b284f94904612bc69c731c1a9ca70a0027ecab1ba035f8237415dd5ac795cc508a26aab6fdf49f2d46c42d721b3"),
    PADS[0].id: (15810, 30196, 1537, "a8f7af745cc259a5f2919ecd9bceba6d3c58c630682fdb5d3df7df511232e043c844f5737c607cda1473c3b580d6dd1dca9282708f2cd1f42045abf1d7274c64"),
    PADS[1].id: (15810, 30196, 1537, "5d4be0ab8a07f2b093dc93c7e67155808a6e31802ecf971c9b6e5eee83a00529f2590d2b886555f625c2054d05990c5f7130ab9edfe526eb41bc310c86d1feff"),
    ROWS.id: (13546, 30500, 1286, "a36501e9e91ff6411979b5343171c6cdedede822a1bdef4e6e3ab724061faec91af1c675742fc7da14f1d8e07463adaebb105bc3c4d96114799ae91c3780b053"),
    RESIDUAL.id: (14263, 31561, 1408, "8ee4456789c729dce382bc9086f4ee1cd26607ad78d9a6483b2e62ac94ccc576403211b87fe04c3b3c53f109c3987e45adcd4890824e9d43b0ef55bbfa0e08a3"),
    PVROWS[0].id: (1011840, 1932544, 98368, "c8a94f7f06d17f7d3f5e5d5e7483eb348aa7988c2a5042741037297cf25ceb3c3f26fb6a94647d1d33c3495d8715bca4a8b487107a3ab3df42b2657f6544e3f5"),
    PVROWS[1].id: (1011840, 1932544, 98368, "c73f7f5b0fe02297e6d0e8146d0b77c859e014660529aafa71c331bf72b7d6c1a79d4989a84b9a99543754b4921b3beb4239266ad362315488e3862b6713ef5a"),
    PVROWS[2].id: (1011840, 1932544, 98368, "cfb9cb910f6ad9412d6b3a2f154754c653925645787aa74426b8e5933bd49fc7096a85c124fc86abb71affd2b260e1cffe80d98570190e729cab03dc567cdc23"),
    PVROWPAD.id: (1046175, 2002233, 99424, "d615e7fc47ea054bbbcc2123fa665565037dbc3bcb4f1ffda84d0225495fa8ed435245b2f6418bcb4ab244be602b74bd70cb7c0317b5a20cca7d2a19eb0e9697"),
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
    bind(BG.GemmAccumulatorS8, K=131040, **BG.GEMM_SM120_S8): (61445476, "fe0bf2374ffd80b57b92ac0d6ecfc06c85809e05589965d654d4d5d893da17f5fa429da6e8354c26fe5547dad9d479075a2ff2b87f9b294bc7d4da923939649a"),
    bind(BG.ScaledMmS8Coordinate, K=131040, **BG.GEMM_SM120_S8): (61460798, "43c17d09742be0b7b19488b17267c6aa8df9a2ba1b0948bad9b1c8f66ec99d7f470632edd3ec658a9647b050f02f4619cf96ad284af9d2fe75eb511bc68dbe2b"),
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
