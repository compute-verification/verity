"""`verity.ml.circuits.ops.boolean.gemm`: the Boolean BF16 steps, cast, dot and coordinate agree with their word Definitions (`word_of`) on
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
from verity.ml.circuits.silicon import fp

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
    "F2fpBf16_v2": (66, 98, 15, "dc436e6bda068001621acaa798474d37bd955392fd5ae065c6ca6604141cef3a9cff1cfa08fdd43a8d3c3d2989e9acd69285c46e8b7333af6fe78fda5cb34105"),
    "AmpereBF16TcDot16_v3": (8260, 15717, 860, "e679c3424c6b0e1109eb340eb73b7558f1d5c221eaced8fb7b5db6f67c6b455d4474d90042e4ab83d1c048d2bbd635060e12a385a7fcd09a7a08f748bb2ab984"),
    "HopperBF16WgmmaDot16_v2": (7872, 15049, 761, "52c6a2d6ff6a794455edbdfe52486b0c462623883ba079daf68804655826747179940195147309470bef9823bdb1b634849cc849c96ee032f3c266aabbf40aee"),
    "HopperBF16WgmmaDot16_v3": (6887, 14109, 728, "324f525bdf51797bbcded8b90809e5987349bf7894fa3c10cd74dd574a5ba4fecbc72b36e10736cd9f5de3d0a750636ad1c3b2aea1a0775e2da76417a0b98e15"),
    HOPPER_V3_K64.id: (27614, 56534, 2927, "bd210a769d585a5e7458e29430fc3727e113c162bb6431fe5580ed2dc91d12c3fc1fa384bca6f59cf6f510a0b35b77d43bd30fcd8e6cb92e0b361ed8bc2959b5"),
    GEMMS[2].id: (41520, 84948, 4413, "588986891c25204d57c4dfba8b9d35429a3422864f3f1c4953811deca7e448177d7ef69e7703d27f41865cc85160bc7931978b6b8940ad617a3e5559431b2761"),
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
    "BlackwellS8ImmaDot32_v2": (4158, 10373, 474, "344df7d41dd450167b60d3696056942401f5559f029e746aaead23d03ffef532749da7135dd03c605e3491045b487669268bad46ea431a8d1aabe25037853dc5"),
    "S32ToF32Rn_v2": (368, 532, 43, "94fffccc7430d2311246e7ebe1a58f94679ad5873ca1c79a3df13092b990618c8b5984c9f497a18615302242f009b1f9f89ca91fa2cc4b6c7bfacffa7e89f848"),
    S8[2].id: (8316, 20746, 948, "cea9d903df4c752fd2eed0cad1182358692211e0b91b66ff1aa0786cfe29701a5b74c864b205ee329d67a5b7964d458ebc958f55c2112314fd552070ccd18139"),
    S8[3].id: (13546, 30500, 1286, "dcfc7b4b1c49380afa07a808563fbed0708769e10392614c43ee53429d8886c8dd9f4e263b2174784219a6f39c8e1d038aab290659b95a2ec1ec151d379f7fa2"),
    PADS[0].id: (15810, 30196, 1537, "5b7d01d00c96f657cc50610f32444b115d4eeef67f3bbc35b8b7d4797f92b9d42c8d5df9d4af5a7a29b9ad8e1189258001ede667860a887e60553279b54c5fa0"),
    PADS[1].id: (15810, 30196, 1537, "881b9ed3708fd486d94a3f476c27a71d34f4ff7c8b5dfc6a376f33e5e47ae0dc551a1ab1ca4a217265638dd9bc3ab998c678eb166fec4f648fd95c5a72860726"),
    ROWS.id: (13546, 30500, 1286, "a41ad1ee8f4e69ac0d97cd1ffaaaffdab0d665388f3447e517a8e611b898cfe206b1bd0259094bda0683d588c903f851529707467bbc0abacc85e717d2a77771"),
    RESIDUAL.id: (14263, 31561, 1408, "3c691bca7c9dd21686fd75d1c37096c0609d5bc05a1473efbad46e38c8fa921e0ba2624e92c9e673aa2f717ddf034995f4492de69f05675a878f7e13a35cb552"),
    PVROWS[0].id: (1011840, 1932544, 98368, "d1073d6d6cce761273e1431921924db345f441efd139b081eca21994371562c0fa2910abf53780ced933673655347a5cee79b6adceacecf15e3563e1d8498a67"),
    PVROWS[1].id: (1011840, 1932544, 98368, "404904a5f6b17d5010aa17c68f93cb69ce514c3719a0e5746431bda079e5cccc990d4172ea25769a06ceb9bc9c9d1f1c187c38dc46959737a11673f55a4a1568"),
    PVROWS[2].id: (1011840, 1932544, 98368, "52c7bce9fe1c63421d9a49037aa7585fb5e4a2c5fe93e1744ebdcfb28250da30ce33367174aa2489d4a4a660213c9466185d2a32d5c1998302a6f8aba3f3b55a"),
    PVROWPAD.id: (1046175, 2002233, 99424, "26bdff43d4ddbf9f867054015d45e79557aad78f57346b82e5b48785f517360c54af59f1a89268d75dcd5834f298bb45b0dd54c076944e43cf58d7c4d3610ee7"),
}
TARGETS = [bind(BG.F2fpBf16), *STEPS, AMPERE_K64, HOPPER_K64, HOPPER_V3_K64, *DOTS, *GEMMS, *FP8_STEPS, *FP8_COORDS, bind(BG.BlackwellE5m2QmmaDot32),
           *(t for t, _ in FP4), *S8, *PADS, ROWS, RESIDUAL, *PVROWS, PVROWPAD]


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
    bind(BG.GemmAccumulatorS8, K=131040, **BG.GEMM_SM120_S8): (61445476, "3f1ca8e5174475e107d370b1453f5f970b773b60c3d7c5d835f0a80ddcd9bdeb23aaafb84b157926776d691496dac53d937eb3dd05ab181ebcf7fb87720cc79a"),
    bind(BG.ScaledMmS8Coordinate, K=131040, **BG.GEMM_SM120_S8): (61460798, "6d2bf4fa1d2368702ed758d0bbee1127e95f86411826a6bc0874f65df276edae6877cd907addc57949c86b8dc7bd47ca65e20fab1d2a36dbda11fb9b793729e4"),
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
    drawn = []
    for shape, bits, sp in (((n,), 32, SPECIAL_F32), ((n, 16), 16, SPECIAL_BF16), ((n, 16), 16, SPECIAL_BF16)):
        uniform = rng.integers(0, 1 << bits, shape, dtype=np.uint64).astype(sp.dtype)
        w = np.where(rng.random(shape) < 0.5, uniform, _finite_words(rng, shape, bits))
        drawn.append(np.where(rng.random(shape) < rate.reshape(-1, *([1] * (len(shape) - 1))), rng.choice(sp, shape), w))
    rows.append(drawn)
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
    _agree(raw, [acc, a, b])
    out = got.reshape(len(acc)).astype(np.uint32)
    finite = ((acc & 0x7F800000) != 0x7F800000) & ((a & 0x7F80) != 0x7F80).all(1) & ((b & 0x7F80) != 0x7F80).all(1)
    inf = (out & 0x7FFFFFFF) == 0x7F800000
    assert {0x7FFFFFFF, 0x7F800000, 0xFF800000} <= set(out.tolist())
    assert (finite & inf).sum() > 5000 and (~inf & (out != 0x7FFFFFFF)).sum() > 3000


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


def _s8_operands(rng, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform s8 bytes; every fourth instance's operands are drawn from {-128, -127, 127} (the largest products)."""
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
    _agree(S8[0], [acc, a, b])
    exact = acc.view(np.int32).astype(np.int64) + (a.view(np.int8).astype(np.int64) * b.view(np.int8)).sum(1)
    got = BS.evaluate_words(S8[0], acc, a, b).astype(np.uint32).view(np.int32)
    assert np.array_equal(got, np.clip(exact, -(1 << 31), (1 << 31) - 1))
    assert ((exact > (1 << 31) - 1).sum() > 500) and ((exact < -(1 << 31)).sum() > 500)


def test_the_s32_cast_agrees_with_its_word_cast_and_numpy() -> None:
    rng = np.random.default_rng(22)
    words = np.concatenate([rng.integers(0, 1 << 32, 100_000, dtype=np.uint64).astype(np.uint32),
                            (np.uint32(1) << np.arange(32, dtype=np.uint32)) + np.uint32(1),
                            np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x01000001, 0x01000003], np.uint32)])
    _agree(S8[1], [words])
    assert np.array_equal(BS.evaluate_words(S8[1], words).astype(np.uint32), words.view(np.int32).astype(np.float32).view(np.uint32))


def test_the_s8_accumulator_and_coordinate_agree_with_their_word_definitions() -> None:
    """On s8 rows, with per-token and per-channel scales of an int8 linear's range and with uniform f32 scale words."""
    rng = np.random.default_rng(23)
    n = 2000
    x, w = _s8_operands(rng, n, 64)
    _agree(S8[2], [x, w])
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = (np.abs(rng.standard_normal(n)) * 0.02 / 127).astype(np.float32).view(np.uint32)
    _agree(S8[3], [x, sx, w, sw])
    u = rng.integers(0, 1 << 32, (2, n), dtype=np.uint64).astype(np.uint32)
    _agree(S8[3], [x, u[0], w, u[1]])


def test_the_row_split_s8_coordinate_agrees_with_its_word_definition_and_is_the_unsplit_coordinate() -> None:
    """`ScaledMmS8CoordinateRows_v2{K=64, R=2}` on x's and W's row's halves is `ScaledMmS8Coordinate_v2{K=64}` on the whole rows."""
    assert TR.word_of(ROWS).id == "ScaledMmS8CoordinateRows_v1{K=64,R=2,DOT=BlackwellS8ImmaDot32_v1}"
    rng = np.random.default_rng(26)
    n = 1000
    x, w = _s8_operands(rng, n, 64)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    parts = [x[:, :32], x[:, 32:], sx, w[:, :32], w[:, 32:], sw]
    _agree(ROWS, parts)
    assert np.array_equal(BS.evaluate_words(ROWS, *parts), BS.evaluate_words(S8[3], x, sx, w, sw))


def test_the_residual_coordinate_agrees_with_its_word_definition_and_is_the_coordinate_then_the_add() -> None:
    """`ScaledMmS8ResidualCoordinateRows_v2{K=64, R=2}` is `Bf16AddF2fp_v2(r, ScaledMmS8CoordinateRows_v2{K=64, R=2})`, r
    random, special and the coordinate's word negated."""
    from verity.ml.circuits.ops.boolean import elementwise as EW

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
    _agree(RESIDUAL, parts + [r])
    assert np.array_equal(BS.evaluate_words(RESIDUAL, *parts, r).reshape(n), BS.evaluate_words(EW.Bf16AddF2fp, r, y).reshape(n))


def test_the_padded_coordinate_agrees_with_its_word_definition_and_is_the_coordinate_on_zero_padded_rows() -> None:
    """`GemmCoordinatePad_v2{T, K=32}` on T words is `GemmCoordinate_v3{K=32}` on them followed by K - T words of +0 bits."""
    coord = bind(BG.GemmCoordinate, K=32, **BG.GEMM_HOPPER)
    params, _ = E.batch.signature_of(TR.word_of(coord))
    xs, ws = K.sample_tc(np.random.default_rng(17), 2000, params)
    for T in (1, 5, 16, 17, 31, 32):
        target = bind(BG.GemmCoordinatePad, T=T, K=32, **BG.GEMM_HOPPER)
        assert TR.word_of(target).id == f"GemmCoordinatePad_v1{{T={T},K=32,DOT=HopperBF16WgmmaDot16_v1}}"
        _agree(target, [xs[:, :T], ws[:, :T]])

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
        _agree(target, [p, *vs])
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
    _agree(PVROWPAD, [p, *vs, counts])
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
    _agree(target, [x, sx, w, sw])
    special = np.array([0, 0x80000000, 1, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001], np.uint32)
    _agree(target, [x, rng.choice(special, n), w, rng.choice(special, (n, 3))])


@pytest.mark.parametrize("target", DOTS, ids=lambda t: t.id)
def test_the_dot_agrees_with_dot_bf16_v2(target) -> None:
    word = TR.word_of(target)
    assert word.id == f"DotBf16_v2{{K={target.bindings['K']},DOT={TR.word_of(target.bindings['DOT']).id}}}"
    params, _ = E.batch.signature_of(word)
    _agree(target, K.sample_tc(np.random.default_rng(15), 4000, params))


@pytest.mark.parametrize("target", [AMPERE_K64, HOPPER_K64, HOPPER_V3_K64], ids=lambda t: t.id)
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
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.silicon.integer import HOPPER_SM90_S8_K32

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
