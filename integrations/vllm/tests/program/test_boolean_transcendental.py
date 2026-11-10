"""`verity.definitions.registry.boolean_transcendental`'s `F32ExpRn_v2` and `F32LogRn_v2`, `ref_prims`' `F32ExpRn_v1` and
`F32LogRn_v1` on bits: each is Boolean, names its word primitive and is its Boolean version, its gates and its Flock rows and
nonzeros are pinned, and it agrees with the word on edges, every binade's ends, its thresholds or the words near 1, the
words whose result lies nearest a rounding boundary, and random words. Each is held on every one of the 2^32 words by
`benchmarks/f32_exhaustive/run.py boolean F32ExpRn` and `... boolean F32LogRn`."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import Circuit
from verity.circuits.boolean.trace import word_of
from verity.circuits.evaluation import bits as BS
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import boolean_transcendental as BT
from verity.definitions.registry import ref_prims as RP

#: per Definition: (And, Xor, Not) gates, and the Flock unit's rows (ANDs and committed bits) and nonzeros in A and B
PINS = {"F32ExpRn": ((6055, 45190, 192), (6484, 445332)), "F32LogRn": ((3080, 27298, 293), (3541, 103111))}
M32 = (1 << 32) - 1
F32_EDGES = [0, 1 << 31, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
             0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, M32]
#: the words x whose exp(x) lies within 2^-25 ulp of a rounding boundary, nearest first (-14.567090034484863, within 2^-28)
HARDEST = [0xC16912CD, 0xC2B2E798, 0xBBF0EDF1, 0xBAE0E25C, 0xB3000000, 0x377EFF81, 0x4288942B, 0x40315B33, 0x4001B249, 0x3D1A274E,
           0x39E5BB1D, 0x39C6BE5B, 0x38E69CC1, 0x383A3EF1, 0x36FDFFC1, 0x367BFFE1, 0x35F7FFF1, 0x356FFFF9, 0x34DFFFFD, 0x343FFFFF,
           0xC13D6631, 0xC0781533, 0xBCB8F40F, 0xBC2A461A, 0xBBB70EE8, 0x41CBF87B, 0x4034D02B, 0x3FE67199, 0x3CAFFE2E, 0x3C971AAA,
           0x3A7BCD08, 0x38AD9E29, 0x33800000, 0x337FFFFF]
#: the words x whose ln(x) lies within 2^-27 ulp of a rounding boundary, nearest first (1.2783784e23, within 2^-34), and the
#: words within 2^-15 of 1 whose ln(x) lies within 2^-56 of one (1 - 2^-23, within 2^-70.5)
LOG_HARDEST = [0x65D890D3, 0x4C5D65A5, 0x4D604EBE, 0x66A8C860, 0x41178FEB, 0x1F116AB8, 0x6F31A8EC, 0x3C413D3A, 0x79E7EC37,
               0x5EE8984E, 0x4665A9A6, 0x3BF86EF0, 0x38DCBE38, 0x0DC8BBA4, 0x464D5B2B, 0x2C4C24B7, 0x1A8446CB, 0x111C87F8,
               0x6D1F23EB, 0x66ABBD63, 0x62B467BA, 0x5F64C24A, 0x5D800341, 0x54AF989D, 0x4E85F412, 0x2E492984, 0x29FD22F8,
               0x29E6126B, 0x28E3FA26, 0x2423C085, 0x16C5EE7A, 0x145CB6D4, 0x0F61FF63, 0x07C060FA, 0x064CB44B,
               0x3F7FFFFE, 0x3F800006, 0x3F7FFFF4, 0x3F800014, 0x3F7FFFD8, 0x3F80001C, 0x3F7FFFC8]


@pytest.mark.parametrize("name", sorted(PINS))
def test_it_is_boolean_names_its_word_and_is_its_boolean_version(name: str) -> None:
    d, word, construction = getattr(BT, name), getattr(RP, name), {"F32ExpRn": BT.exp_rn, "F32LogRn": BT.log_rn}[name]
    gates, flock = PINS[name]
    assert word_of(d) is word and d.id == f"{name}_v2"
    assert BL.is_boolean(d.bind()) and boolean_version(word).id == d.bind().id
    assert BS.compile(d.bind()).counts == dict(zip(("and", "xor", "not"), gates))
    C = Circuit(cse=True)
    C.output("y", construction(C, C.input("x", 32)))
    c = C.counts()
    assert (c["ands"], c["nnz_AB"]) == flock


def _agrees(d, word, x) -> None:
    got = BS.evaluate_words(d, x.reshape(-1, 1)).reshape(-1).tolist()
    want = [word.evaluate(int(u)) for u in x]
    bad = [i for i in range(len(want)) if got[i] != want[i]]
    assert not bad, [(hex(int(x[i])), hex(got[i]), hex(want[i])) for i in bad[:5]]


def test_exp_agrees_with_the_word_on_edges_thresholds_the_hardest_words_and_random_words() -> None:
    """The edges and every binade's first and last word, both signs; each threshold and its neighbours (exp(x) past the largest
    finite word, below 2^-150, and the subnormal range's ends); the hardest words; random words over all 2^32 and dense in
    the range whose exp is neither 0, 1 nor inf."""
    binades = [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 0x7FFFFF)]
    thresholds = [t + k for t in (BT.OVERFLOW, BT.UNDERFLOW | 1 << 31, 0xC2AEAC50, 0xC2B17218) for k in (-2, -1, 0, 1, 2)]
    rng = np.random.default_rng(20261103)
    _agrees(BT.F32ExpRn, RP.F32ExpRn,
            np.concatenate([np.array(F32_EDGES + binades + thresholds + HARDEST, dtype=np.uint64),
                            rng.integers(0, 1 << 32, 1 << 13, dtype=np.uint64),
                            rng.integers(0x33000000, BT.OVERFLOW, 1 << 13, dtype=np.uint64),
                            rng.integers(0xB3000000, BT.UNDERFLOW | 1 << 31, 1 << 13, dtype=np.uint64)]))


def test_log_agrees_with_the_word_on_edges_near_1_the_hardest_words_and_random_words() -> None:
    """The edges and every binade's first, second and last word, both signs; every word within 2^-13 of 1, where the three
    stages are exact and ln(x) comes nearest a boundary; the subnormal powers of two and all-ones significands; the hardest
    words; random words over all 2^32, over the positive finite words and over the subnormals."""
    binades = [s | e << 23 | m for s in (0, 1 << 31) for e in range(256) for m in (0, 1, 0x7FFFFF)]
    near_1 = list(range(0x3F800000 - (1 << 11), 0x3F800000 + (1 << 10)))
    subnormal = [1 << k for k in range(23)] + [(1 << k) - 1 for k in range(2, 24)]
    rng = np.random.default_rng(20261103)
    _agrees(BT.F32LogRn, RP.F32LogRn,
            np.concatenate([np.array(F32_EDGES + binades + near_1 + subnormal + LOG_HARDEST, dtype=np.uint64),
                            rng.integers(0, 1 << 32, 1 << 13, dtype=np.uint64),
                            rng.integers(0, 0x7F800000, 1 << 13, dtype=np.uint64),
                            rng.integers(1, 0x00800000, 1 << 11, dtype=np.uint64)]))
