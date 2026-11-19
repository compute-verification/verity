"""`verity.circuits.ops.boolean.mufu`: the Boolean MUFU operations agree with their word Definitions (`trace.word_of`) on special and random
words and, through every table index, on whole binades; each table's interpolator reproduces the measured words; and every
Definition is pinned by gate count and program digest."""
from __future__ import annotations

import copy
import itertools
import zlib

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits.defs import bind
from verity.circuits.ops import mufu as M
from verity.circuits.ops.boolean import mufu as BM
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import Circuit

from _boolean_checks import assert_live_ands, assert_pinned

DEFS = [BM.MufuEx2Ftz, BM.MufuRcpFtz, BM.MufuSqrtFtz, BM.RsqrtApprox, BM.DivFullRcp, BM.DivFullScaleA, BM.MufuTanh]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MufuEx2Ftz_v2": (1073, 4546, 98, "17224ff4f2c2acf5e204b3995768309ea29ff00d5c05035e51023481a9c8ea2f0a55cc89d1a01a28c8f1984cef8c87829f74740f426c23307e9a248478344083"),
    "MufuRcpFtz_v2": (902, 4810, 101, "2e29ec51721091977ad90e14888d4abbf0193a1dcf9099628ec09068f106b1e73ac87a013e7054d8f88977337e6a2c1bc6653a6d7cb12623d1a87ff79f6c9fc5"),
    "MufuSqrtFtz_v2": (879, 4840, 110, "9bd09e1412b8bc56759bfbbd2284812fe2aa86095332eea301ea75efff694facb40eee2db841e6ddf762460bf25b6c8a5ea935891e3c9bf17003aab5d0f1594a"),
    "RsqrtApprox_v2": (1134, 5269, 163, "c1c81cfcd7f457dd34145e0e87a116e10816393f1c747509b12f593732c444be58a8cc8ae6b50d14dd73ffda220b7e779cb4fa6500dee380fc77b8626e01f4a9"),
    "DivFullRcp_v2": (1152, 5172, 144, "246154abddd12ebcb1e7c0f30bde6170ee4a2dfa491b7c0dd3a07f989bf592bee4bb18d8eef4e5ba06aad7625f899a16e9a79cc8045da67ad4c77e5dd33fef00"),
    "DivFullScaleA_v3": (531, 860, 94, "ff538aa42d0a1268648ecf11716882acd49d4a3e1a8770b1433571a8c5aa47183b2218212c60257bd9d595c1c99c989543c16ede66a11bb8ddaa7fc8c9276633"),
    "MufuTanh_v2": (1049, 5408, 77, "456d541892ec291b2bd2abc1ae86cb721f2c7aa0c7163ff2ed1c696fc2407701c8b62b0f8818f55221d99a48ad487b70857083380b47ff02a9e2bbade2cb9e81"),
}

#: each table's words that are not its interpolant: (index, word - interpolant)
OVERRIDES = {"ex2": (), "rcp": (), "rsq": ((0, 1),), "sqrt": ()}

#: signed zeros, subnormals (the FTZ boundary), the normal range's ends, ones, the range-scaling thresholds 2^-126 and 2^126,
#: infinities, NaNs of both signs and kinds, ex2's saturation points, words just past a binade's ends, and the ends of tanh's
#: identity (|x| <= 2^-8) and saturation (|x| >= 8)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x3FFFFFFF, 0x40000000, 0x407FFFFF, 0x3E800000, 0x4B800000, 0x7E7FFFFF, 0x7E800000, 0x7E800001, 0x7F000000,
               0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x43000000, 0x42FE0000, 0x43160000, 0x33800000,
               0x34000000, 0x0C000000, 0x3B7FFFFF, 0x3B800000, 0x3B800001, 0x3BFFFFFF, 0x40FFFFFF, 0x41000000]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x30000000, 0x50000000, (n_params, 20_000), dtype=np.uint64)
            | (rng.integers(0, 2, (n_params, 20_000), dtype=np.uint64) << np.uint64(31))).astype(np.uint32)
    out = [np.concatenate([SPECIALS[g], r, x]) for g, r, x in zip(grid, rand, near)]
    # MufuTanh_v1 has no batch kernel and decodes a table block at each word it misses: sorted, it decodes each block once
    return [np.sort(out[0])] if n_params == 1 else out


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_pinned_counts_and_digests(d) -> None:
    assert_pinned(bind(d), PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    assert_live_ands(d, PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_agrees_with_its_word_primitive(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 1)
    inputs = _inputs(len(d.signature(None)[0]), np.random.default_rng(zlib.crc32(d.name.encode())))
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_the_interpolator_is_every_table_word(name) -> None:
    f = BM.fit(name)
    words = np.asarray(M.table(name), dtype=np.int64)
    q = f.interpolant(np.arange(len(words), dtype=np.int64))
    off = np.flatnonzero(q != words)
    assert tuple((int(i), int(words[i] - q[i])) for i in off) == BM.overrides(name) == OVERRIDES[name]


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_the_circuit_square_is_the_fit_square(name) -> None:
    f = BM.fit(name)
    C = Circuit()
    C.output("q", BM.truncated_square(C, C.input("u", f.nb), f.p))
    x = np.arange(1 << f.nb, dtype=np.int64)

    def sliced(bits):
        return int.from_bytes(np.packbits(bits.astype(np.uint8), bitorder="little").tobytes(), "little")

    def unsliced(v):
        return np.unpackbits(np.frombuffer(v.to_bytes(len(x) // 8, "little"), np.uint8), bitorder="little").astype(np.int64)

    outs, _ = C.evaluate({"u": [sliced((x >> k) & 1) for k in range(f.nb)]}, len(x))
    q = sum(unsliced(v) << j for j, v in enumerate(outs["q"]))
    assert (q == f.square(x)).all()


def _tanh_words(w: np.ndarray) -> np.ndarray:
    """``MufuTanh_v1`` on a word array as its evaluator answers: the rules (`M.MUFU_TANH_RULES`), else the measured table (a
    negative word by its odd symmetry), each block decoded once."""
    (s,) = M.mufu_tanh_shards()
    R = M.MUFU_TANH_RULES
    w = np.asarray(w, np.uint32)
    a, sign = w & 0x7FFFFFFF, w & 0x80000000
    out = np.where(a >= R["saturate_from"], 0x3F800000 | sign, w).astype(np.uint32)
    tab = np.flatnonzero((a >= R["identity_below"]) & (a < R["saturate_from"]))
    i = a[tab].astype(np.int64) - s.lo
    for k in np.unique(i // s.block_words):
        at = i // s.block_words == k
        out[tab[at]] = s.block(int(k))[i[at] % s.block_words] | sign[tab[at]]
    out[a > 0x7F800000] = R["nan_out"]
    return out


def test_the_tanh_reference_is_the_evaluator() -> None:
    w = _inputs(1, np.random.default_rng(zlib.crc32(b"MufuTanh/reference")))[0]
    assert (_tanh_words(w) == E.evaluate_batch(M.MufuTanh, w).astype(np.uint32).reshape(-1)).all()


@pytest.mark.slow
def test_the_tanh_interpolator_is_every_table_word() -> None:
    """Every word of the binades (about 10 s): the check the build makes (`_tanh_measured`), here against `_tanh_words`, the
    reference that agrees with the evaluator, rather than the blocks as the build reads them."""
    f = BM.tanh_fit()
    m = np.arange(1 << 23, dtype=np.int64)
    for e, _ in f.binades:
        assert (f.words(e, m) == _tanh_words(((e << 23) + m).astype(np.uint32))).all(), f"binade {e}"


def test_a_tanh_fit_that_is_not_the_table_fails_the_build(monkeypatch) -> None:
    f = copy.deepcopy(BM.tanh_fit())
    f.C0[0] += 1 << f.shift
    monkeypatch.setattr(BM, "tanh_fit", lambda: f)
    with pytest.raises(ValueError, match=f"binade {f.binades[0][0]} are not MufuTanh_v1's"):
        BM._tanh_measured.__wrapped__()


def _agree(target, *words) -> None:
    got = BS.evaluate_words(target, *words).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(TR.word_of(target), *words).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ, first {[[hex(int(w[i])) for w in words] for i in bad[:5]]}"


def _span(*ranges) -> np.ndarray:
    return np.concatenate([np.arange(lo, hi, dtype=np.uint64) for lo, hi in ranges]).astype(np.uint32)


#: (id, Definition, word ranges): a binade (two for rsq / sqrt, by exponent parity) reads every index of its table; ex2's
#: negative binade reads them through ~K, and its exponents 103 and 133 are the ends of its 5-bit shift; DivFullRcp's and
#: RsqrtApprox's subnormals, and DivFullRcp's (2^126, inf], are their range scalings
_RANGES = [("MufuEx2Ftz", BM.MufuEx2Ftz, [(0x3F800000, 0x40000000)]),
           ("MufuEx2Ftz-negative-and-shift-ends", BM.MufuEx2Ftz, [(0xBF800000, 0xC0000000), (0x33800000, 0x34000000),
                                                                  (0xC2800000, 0xC3000000)]),
           ("MufuRcpFtz", BM.MufuRcpFtz, [(0x3F800000, 0x40000000)]),
           ("MufuSqrtFtz", BM.MufuSqrtFtz, [(0x3F800000, 0x40800000)]),
           ("RsqrtApprox", BM.RsqrtApprox, [(0x3F800000, 0x40800000)]),
           ("RsqrtApprox-subnormal", BM.RsqrtApprox, [(0x00000000, 0x00800000)]),
           ("DivFullRcp-scaled", BM.DivFullRcp, [(0x00000000, 0x00800000), (0xFE800000, 0xFF800001)])]


@pytest.mark.slow
@pytest.mark.parametrize("d, ranges", [r[1:] for r in _RANGES], ids=[r[0] for r in _RANGES])
def test_every_word_of_a_range(d, ranges) -> None:
    """Every word of the ranges (2^23 to 2^24 + 2^23 words, 10-35 s each): the random words above reach about 1 % of a table's
    indices, and only a few hundred words of each scaled range."""
    _agree(bind(d), _span(*ranges))


@pytest.mark.slow
@pytest.mark.parametrize("b, end", [(0x7E800001, 0x01800000), (0x80000001, 0x00800000)], ids=["a*0.25", "a*2^24"])
def test_div_full_scale_a_where_it_rounds_or_normalizes(b, end) -> None:
    """Every a where a * 0.25 is subnormal and rounds (exponent 0 to 2, 3 * 2^23 words, about 30 s), or where a * 2^24
    normalizes a subnormal (2^23 words, about 10 s); random words reach a few thousand of them."""
    a = _span((0x00000000, end))
    _agree(bind(BM.DivFullScaleA), a, np.full(len(a), b, np.uint32))


@pytest.mark.slow
@pytest.mark.parametrize("e", [e for e, _ in BM.tanh_fit().binades], ids=lambda e: f"2^{e - 127}")
def test_every_word_of_a_tanh_binade(e) -> None:
    """Every positive word of one of MUFU.TANH's binades (2^23 words, about 8 s each; eleven binades): every segment and offset
    of the table read, through the offset's shifter and the straddling segments' exponents. The random words reach about
    0.05 % of them. Against `_tanh_words`, since the word primitive evaluates one word at a time."""
    w = _span((e << 23, (e + 1) << 23))
    got = BS.evaluate_words(bind(BM.MufuTanh), w).astype(np.uint32).reshape(-1)
    bad = np.flatnonzero(got != _tanh_words(w))
    assert not bad.size, f"MufuTanh_v2: {bad.size} of {len(w)} differ, first {[hex(int(w[i])) for i in bad[:5]]}"
