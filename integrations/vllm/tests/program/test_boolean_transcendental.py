"""`verity.ml.definitions.registry.boolean_transcendental`'s `F32ExpRn_v2`, `ref_prims`' `F32ExpRn_v1` on bits: it is Boolean,
names its word primitive and is its Boolean version, its gates and its Flock rows and nonzeros are pinned, and it agrees with
the word on edges, every binade's ends, the overflow and underflow thresholds, the words whose exp lies nearest a rounding
boundary, and random words. It is held on every one of the 2^32 words by `benchmarks/f32_exhaustive/run.py boolean F32ExpRn`."""
from __future__ import annotations

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.definitions.boolean import boolean_version
from verity.ml.definitions.registry import boolean_transcendental as BT
from verity.ml.definitions.registry import ref_prims as RP

#: (And, Xor, Not) gates, and the Flock unit's rows (ANDs and committed bits) and nonzeros in A and B
GATES, FLOCK = (6055, 45190, 192), (6484, 445332)
M32 = (1 << 32) - 1
F32_EDGES = [0, 1 << 31, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
             0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, M32]
#: the words x whose exp(x) lies within 2^-25 ulp of a rounding boundary, nearest first (-14.567090034484863, within 2^-28)
HARDEST = [0xC16912CD, 0xC2B2E798, 0xBBF0EDF1, 0xBAE0E25C, 0xB3000000, 0x377EFF81, 0x4288942B, 0x40315B33, 0x4001B249, 0x3D1A274E,
           0x39E5BB1D, 0x39C6BE5B, 0x38E69CC1, 0x383A3EF1, 0x36FDFFC1, 0x367BFFE1, 0x35F7FFF1, 0x356FFFF9, 0x34DFFFFD, 0x343FFFFF,
           0xC13D6631, 0xC0781533, 0xBCB8F40F, 0xBC2A461A, 0xBBB70EE8, 0x41CBF87B, 0x4034D02B, 0x3FE67199, 0x3CAFFE2E, 0x3C971AAA,
           0x3A7BCD08, 0x38AD9E29, 0x33800000, 0x337FFFFF]


def test_it_is_boolean_names_its_word_and_is_its_boolean_version() -> None:
    d = BT.F32ExpRn
    assert word_of(d) is RP.F32ExpRn and d.id == "F32ExpRn_v2"
    assert BL.is_boolean(d.bind()) and boolean_version(RP.F32ExpRn).id == d.bind().id
    assert BS.compile(d.bind()).counts == dict(zip(("and", "xor", "not"), GATES))
    C = Circuit(cse=True)
    C.output("y", BT.exp_rn(C, C.input("x", 32)))
    c = C.counts()
    assert (c["ands"], c["nnz_AB"]) == FLOCK


def test_it_agrees_with_the_word_on_edges_thresholds_the_hardest_words_and_random_words() -> None:
    """The edges and every binade's first and last word, both signs; each threshold and its neighbours (exp(x) past the largest
    finite word, below 2^-150, and the subnormal range's ends); the hardest words; random words over all 2^32 and dense in
    the range whose exp is neither 0, 1 nor inf."""
    binades = [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 0x7FFFFF)]
    thresholds = [t + k for t in (BT.OVERFLOW, BT.UNDERFLOW | 1 << 31, 0xC2AEAC50, 0xC2B17218) for k in (-2, -1, 0, 1, 2)]
    rng = np.random.default_rng(20261103)
    x = np.concatenate([np.array(F32_EDGES + binades + thresholds + HARDEST, dtype=np.uint64),
                        rng.integers(0, 1 << 32, 1 << 13, dtype=np.uint64),
                        rng.integers(0x33000000, BT.OVERFLOW, 1 << 13, dtype=np.uint64),
                        rng.integers(0xB3000000, BT.UNDERFLOW | 1 << 31, 1 << 13, dtype=np.uint64)])
    got = BS.evaluate_words(BT.F32ExpRn, x.reshape(-1, 1)).reshape(-1).tolist()
    want = [RP.F32ExpRn.evaluate(int(u)) for u in x]
    bad = [i for i in range(len(want)) if got[i] != want[i]]
    assert not bad, [(hex(int(x[i])), hex(got[i]), hex(want[i])) for i in bad[:5]]
