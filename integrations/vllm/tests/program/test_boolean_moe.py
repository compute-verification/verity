"""`verity.ml.definitions.registry.boolean_moe`: the Boolean versions of `registry.moe`'s element-wise Definitions agree with their word
Definitions (`trace.word_of`): the routed sum's coordinate on every word at one slot, on every pair of special words, every word
against 8 specials and random pairs at two, and at OLMoE's eight slots on random rows, rows of special words, rows that cancel to a
few ulps and rows that overflow; the sum in 64-column chunks (`MoeSum_v3`, the lift's version) on random and special rows; the router's exponential on special words, random words and words across its range, its
probabilities and its top-k on random, tied and special logits, and its selection rounds on random, tied and special
probabilities against the word rounds; GLM-5.3's sigmoid router's per-expert score on special, edge and random words, its slots
and `write_outputs` against the word slots on random, tied and mostly -inf selection scores, and the whole router on both
launches on the kernel tests' rows. They are pinned by gate count and program digest and pass `Q_word_v1{X=16, W=32}`'s cut
check."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import itertools
import json

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO
from verity.ml.circuits.cut import separable
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_moe as BM
from verity.ml.definitions.registry import moe as M
from verity.ml.definitions.numerics import moe_twins
from verity.ml.definitions.registry.prims import F32, I32

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeSumCoordinate_v2{TOPK=8}": (5222, 8395, 764, "afe7da5c3245a9ec9d6631f1b9b6b686f2c83af71edd52fd8e0c35cd2dee4af2702d65488815ed365213797fb33d77e830fe82d929c5ef3417e60cd05fc59a7d"),
    "MoeSum_v2{TOPK=8,H=8}": (41776, 67160, 6112, "4c068e14ab23ebb919f4090677ab7cd17c3935c82fe382e6f8cd1ad24bff1210ebffabe282250d4adbda4e23521fa6759ef72a6b539bd49bf4967e6e38f4a699"),
    "MoeSum_v3{TOPK=8,H=136}": (710192, 1141720, 103904,
                                "9fe815a960c0f771d368fcd1318874b93f7b806cf1bdc9032f228844b429146759c4eeaead54ae6409970969c8067fd63ccfabf0996d5daebc5cf726a1d8ceb3"),
    "NvExpf_v2": (11372, 21804, 880, "a91ca00bbe5e644fd206e1be3fe064234dc439382fb2201d823f0c8568646e186262ad31cd26a6253072854294694606fa977c8005650aaf938b23a50ebe16f7"),
    "MoeRouterTopKOrdered_v2{E=8,TOPK=2,VPT=4}": (126243, 239434, 10631,
                                                  "285955047f4821f8c5f751d032ec0bafcb8b1621eaf34a635486e7ac65588ad45d9811acfc985d00bbd5bffcc819e11ac40a0e1ee714c99e41fb5b0d0b9adcc7"),
    "MoeRouterSigmoidScore_v2": (30839, 58793, 2452, "511c827402ceb909aaf546f15ab00ac31b6712e5955806ead1ce12d1cc1a5d35755bbddd37d95aa01ad8d90ab5b23bf0ed07c3076c7cd92c5c4fad5bd89d7456"),
    "MoeRouterSigmoidBias_v2{E=8,TOPK=2,SCALE=2.5,KERNEL='block'}": (258083, 492361, 20884,
                                                                     "b96888197c14bc72fe4a4f6c5a19ae0b11b3ed1247024b175b759a35227b8d149665193205f1db6e309033793c20e42d19a24e74ff6629c046a9d86058407024"),
    "MoeRouterSigmoidBias_v2{E=8,TOPK=2,SCALE=2.5,KERNEL='warp'}": (259626, 494985, 21130,
                                                                    "27ac80523ef558eb7d6bb3d98e86fcdf0345e959eca86586de7a3dc7fd478b57bb9fef175ad1b826c88e125c587efb27e62e3c971cd997b20ba6f9fe8b436750"),
}
TARGETS = [bind(BM.MoeSumCoordinate, TOPK=8), bind(BM.MoeSum, TOPK=8, H=8), bind(BM.MoeSumV3, TOPK=8, H=136), bind(BM.NvExpf),
           pytest.param(bind(BM.MoeRouterSigmoidScore), marks=pytest.mark.slow),
           pytest.param(bind(BM.MoeRouterTopKOrdered, E=8, TOPK=2, VPT=4), marks=pytest.mark.slow),
           *[pytest.param(bind(BM.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL=k), marks=pytest.mark.slow)
             for k in ("block", "warp")]]

_EXP = [0, 1, 2, 126, 127, 128, 133, 134, 135, 253, 254, 255]
#: per sign: zero, subnormals, ones and their neighbours, the finite maximum and its neighbours, infinities and NaNs
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)
EVERY = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
#: f32: signed zeros and smallest subnormals, +-1, infinities, NaNs of both kinds and signs, the finite maxima, and the ends of
#: expf's range (above 0x42B17217 it overflows, below 0xC2CFF1B4 it is +0, near 0xC2AEAC50 its scale 2^j is the smallest normal)
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
                         0xFFC00001, 0x7F7FFFFF, 0xFF7FFFFF, 0x42B17217, 0x42B17218, 0x42B17219, 0xC2CFF1B3, 0xC2CFF1B4,
                         0xC2CFF1B5, 0xC2AEAC4F, 0xC2AEAC50, 0xC2AEAC51], np.uint32)


def _rows(out, n: int) -> np.ndarray:
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in parts], axis=1)


def _agree(target, x, word=None) -> None:
    word = word or TR.word_of(target)
    got, want = _rows(BS.evaluate_words(target, x), len(x)), _rows(E.evaluate_batch(word, x), len(x))
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(x)} rows differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


@pytest.mark.slow
def test_the_served_router_is_pinned() -> None:
    """OLMoE's router at its served statics (64 experts, 8 slots, 8 values per thread), the size its rows are costed at; about
    95 s to trace. The pinned statics above run the same builders, but only these show a change to the served circuit's size."""
    t = bind(BM.MoeRouterTopKOrdered, E=64, TOPK=8, VPT=8)
    assert BS.compile(t).counts == {"and": 1066321, "xor": 2011461, "not": 100083}


def test_the_word_views() -> None:
    assert TR.word_of(bind(BM.MoeSumCoordinate, TOPK=8)).id == bind(M.MoeSumCoordinate, TOPK=8).id == "MoeSumCoordinate_v1{TOPK=8}"
    assert TR.word_of(bind(BM.MoeSum, TOPK=8, H=8)).id == bind(M.MoeSum, TOPK=8, H=8).id == "MoeSum_v1{TOPK=8,H=8}"
    assert TR.word_of(bind(BM.MoeSumV3, TOPK=8, H=6144)).id == "MoeSum_v1{TOPK=8,H=6144}"
    assert SW.boolean_version(bind(M.MoeSum, TOPK=8, H=6144)).id == "MoeSum_v3{TOPK=8,H=6144}"
    assert TR.word_of(bind(BM.NvExpf)).id == bind(M.NvExpf).id == "NvExpf_v1"
    assert SW.boolean_version(bind(M.NvExpf)).id == "NvExpf_v2"
    assert TR.word_of(bind(BM.MoeRouterProbs, E=8, VPT=4)).id == "MoeRouterProbs_v1{E=8,VPT=4}"
    assert (TR.word_of(bind(BM.MoeRouterTopKOrdered, E=64, TOPK=8, VPT=8)).id
            == bind(M.MoeRouterTopKOrdered, E=64, TOPK=8, VPT=8).id == "MoeRouterTopKOrdered_v1{E=64,TOPK=8,VPT=8}")


def test_one_slot_on_every_word() -> None:
    """At one slot the sum is ``bf16_rn(0 + f32(x))``: x, but -0 is +0 and every NaN 0x7FFF."""
    _agree(bind(BM.MoeSumCoordinate, TOPK=1), EVERY.reshape(-1, 1))


#: signed zeros and ones, the smallest subnormal, the finite maximum, -inf and a NaN
AGAINST = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x7F7F, 0xFF80, 0x7FC0], np.uint16)


@pytest.mark.slow
def test_two_slots_on_special_pairs_every_word_and_random_pairs() -> None:
    """Every pair of special words, every word against 8 special words in either slot, and 100,000 random pairs (about 60 s): the
    only exhaustive check of the two-slot sum's rounding, which the random and special rows at eight slots sample."""
    rng = np.random.default_rng(0x5E2)
    grid = np.array(list(itertools.product(SPECIALS, repeat=2)), np.uint16)
    every = np.stack([np.repeat(EVERY, len(AGAINST)), np.tile(AGAINST, len(EVERY))], axis=1)
    rand = rng.integers(0, 1 << 16, (100_000, 2), dtype=np.int64).astype(np.uint16)
    _agree(bind(BM.MoeSumCoordinate, TOPK=2), np.concatenate([grid, every, every[:, ::-1], rand]))


@pytest.mark.slow
def test_eight_slots_on_random_special_cancelling_and_overflowing_rows() -> None:
    """OLMoE's TOPK = 8: 100,000 random rows, 20,000 rows of special words, 20,000 of nearby words of alternating sign (the sum
    cancels to a few ulps, or to zero) and 20,000 near the finite maximum of one sign (the sum overflows, or stops just short);
    about 15 s."""
    rng = np.random.default_rng(0x5E8)
    rand = rng.integers(0, 1 << 16, (100_000, 8), dtype=np.int64)
    spec = SPECIALS[rng.integers(0, len(SPECIALS), (20_000, 8))].astype(np.int64)
    base = rng.integers(0, 0x7F00, (20_000, 1), dtype=np.int64)
    canc = (base + rng.integers(-2, 3, (20_000, 8))).clip(0, 0x7F7F) | np.array([0, 1] * 4) << 15
    big = rng.integers(0x7E00, 0x7F80, (20_000, 8), dtype=np.int64) | rng.integers(0, 2, (20_000, 1)) << 15
    rows = np.concatenate([rand, spec, canc, big]).astype(np.uint16)
    _agree(bind(BM.MoeSumCoordinate, TOPK=8), rows)


def test_routed_sum_rows() -> None:
    """`MoeSum_v2` over 5,000 random tokens and 1,000 of special words: each column is its coordinate."""
    rng = np.random.default_rng(0x5E9)
    y = np.concatenate([rng.integers(0, 1 << 16, (5000, 8, 8), dtype=np.int64).astype(np.uint16),
                        SPECIALS[rng.integers(0, len(SPECIALS), (1000, 8, 8))]])
    _agree(bind(BM.MoeSum, TOPK=8, H=8), y)


def test_the_routed_sum_passes_the_word_querys_cut_check() -> None:
    prog, _ = standalone_call(bind(BM.MoeSum, TOPK=8, H=3))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok


@pytest.mark.slow
def test_the_chunked_routed_sum_rows() -> None:
    """`MoeSum_v3` at two full chunks and a remainder of 8 columns, over 150 random tokens and 100 of special words: the word
    `MoeSum_v1`."""
    rng = np.random.default_rng(0x5EA)
    y = np.concatenate([rng.integers(0, 1 << 16, (150, 8, 136), dtype=np.int64).astype(np.uint16),
                        SPECIALS[rng.integers(0, len(SPECIALS), (100, 8, 136))]])
    _agree(bind(BM.MoeSumV3, TOPK=8, H=136), y)


@pytest.mark.parametrize("H, members", [(8, [("MoeSum_v2{TOPK=8,H=8}", 1)]), (136, [("MoeSum_v2{TOPK=8,H=64}", 2), ("MoeSum_v2{TOPK=8,H=8}", 1)]),
                                        (6144, [("MoeSum_v2{TOPK=8,H=64}", 96)])])
def test_the_chunked_routed_sum_is_separable_into_chunks(H, members) -> None:
    """GLM-5.3's sum (TOPK = 8, H = 6144) is 96 members of 64 columns, each reading one 8 x 1024-bit row."""
    assert [(c.id, n) for c, n in separable(bind(BM.MoeSumV3, TOPK=8, H=H))] == members


def test_the_chunked_routed_sum_passes_the_word_querys_cut_check() -> None:
    prog, _ = standalone_call(bind(BM.MoeSumV3, TOPK=2, H=66))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok


# --- the router ----------------------------------------------------------------------------------------------------------------

#: the words within 8 ulps of each x, of either sign, where j steps (``x * log2e / 252 + 0.5`` crosses a multiple of 1/252)
J_EDGES = ((np.arange(-126, 128) * np.log(2)).astype(np.float32).view(np.uint32).astype(np.int64)[:, None]
           + np.arange(-8, 9)).reshape(-1)
J_EDGES = np.concatenate([J_EDGES, J_EDGES | 1 << 31]).astype(np.uint32)


def _expf_words(rng, n: int) -> np.ndarray:
    """F32_SPECIALS, J_EDGES, n random words and n values across expf's range (every scale 2^j, j in [-126, 126], with its
    ends)."""
    return np.concatenate([F32_SPECIALS, J_EDGES, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32),
                           rng.uniform(-110, 95, n).astype(np.float32).view(np.uint32)])


def test_jbs_low_byte_is_the_fma_rms() -> None:
    """`jb_low_byte` against `F32FmaRm_v1(s, 252, 12582913)`'s low byte for s in [+0, 1]: within 2 ulps of each k / 252 (where
    the floor steps), the ends, and 3,000 random s (every one of the 1,065,353,217 such words agrees: 13 minutes, run once)."""
    k = (np.arange(253, dtype=np.float32) / np.float32(252)).view(np.uint32).astype(np.int64)
    edges = (k[:, None] + np.arange(-2, 3)).clip(0, 0x3F800000).reshape(-1)
    rand = np.random.default_rng(0x252).integers(0, 0x3F800001, 3000, dtype=np.int64)
    s = np.concatenate([[0, 1, 0x007FFFFF, 0x00800000, 0x3F7FFFFF, 0x3F800000], edges, rand]).astype(np.uint32)
    d = TR.definition("JbLowByteCheck", 1, lambda C, s: BM.jb_low_byte(C, s) + [ZERO] * 24, (("s", BM.F32),), BM.F32,
                      register=False)
    want = [M.F32FmaRm.evaluate(int(x), 0x437C0000, 0x4B400001) & 0xFF for x in s]
    assert _rows(BS.evaluate_words(d, s), len(s)).reshape(-1).tolist() == want


def test_the_exponential_on_special_random_and_in_range_words() -> None:
    _agree(bind(BM.NvExpf), _expf_words(np.random.default_rng(0xE1), 3000))


@pytest.mark.boolean_sweep
@pytest.mark.slow
def test_the_exponential_on_many_words() -> None:
    """100,000 random words and 100,000 across expf's range (about 30 s)."""
    _agree(bind(BM.NvExpf), _expf_words(np.random.default_rng(0xE2), 100_000))


def _logits(rng, n: int, E_: int) -> np.ndarray:
    """Router logits: n random bf16 rows, n of normal values, n of small integers (equal probabilities, the lower expert first)
    and n of special words (signed zeros, subnormals, the finite maxima, infinities, NaNs)."""
    return np.concatenate([rng.integers(0, 1 << 16, (n, E_), dtype=np.int64).astype(np.uint16),
                           (rng.normal(0, 3, (n, E_)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16),
                           (rng.integers(-3, 4, (n, E_)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (n, E_))]])


def test_the_router_probabilities() -> None:
    _agree(bind(BM.MoeRouterProbs, E=8, VPT=4), _logits(np.random.default_rng(0xB1), 100, 8))


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("E_, VPT", [(8, 4), (16, 2), (8, 1)])
def test_the_router_probabilities_on_many_rows(E_, VPT) -> None:
    """1,500 rows of each kind at 8 experts over 2 threads, 16 over 8, and 8 over 8 (15 to 30 s each)."""
    _agree(bind(BM.MoeRouterProbs, E=E_, VPT=VPT), _logits(np.random.default_rng(0xB2 + E_ + VPT), 1500, E_))


def _select_rounds(B, S, p):
    """`MoeRouterTopKOrdered_v1`'s rounds on given probabilities."""
    ids = [B.call(bind(M.MoeRouterFirst, E=S.E), p)]
    sel = [B.call(bind(M.MoeRouterPick, E=S.E), p, ids[0])]
    for _ in range(1, S.TOPK):
        ids.append(B.call(bind(M.MoeRouterNext, E=S.E), p, sel[-1], ids[-1]))
        sel.append(B.call(bind(M.MoeRouterPick, E=S.E), p, ids[-1]))
    return tuple_of(array_of(sel), array_of(ids))


SELECT_ROUNDS = CompositeDefinition("SelectRounds", 1, ("E", "TOPK"),
                                    lambda S: ((("p", Array(S.E, F32)),), Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))),
                                    _select_rounds, register=False)
#: f32 probabilities: signed zeros and smallest subnormals, +-1, 0.125, infinities, NaNs, the mask -10000 and its neighbours, and
#: the finite maximum
P_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x3E000000, 0x7F800000, 0xFF800000, 0x7FC00000,
                       0xFFFFFFFF, 0xC61C4000, 0xC61C4001, 0xC61C3FFF, 0x7F7FFFFF], np.uint32)


def _probabilities(rng, n: int, E_: int) -> np.ndarray:
    """n rows each: random words, special words, probabilities in [0, 1) and multiples of 0.125 (ties)."""
    return np.concatenate([rng.integers(0, 1 << 32, (n, E_), dtype=np.uint64).astype(np.uint32),
                           P_SPECIALS[rng.integers(0, len(P_SPECIALS), (4 * n, E_))],
                           rng.uniform(0, 1, (n, E_)).astype(np.float32).view(np.uint32),
                           (rng.integers(0, 4, (n, E_)) * 0x3E000000).astype(np.uint32)])


def test_the_selection_rounds() -> None:
    _agree(bind(BM.MoeRouterSelect, E=6, TOPK=3), _probabilities(np.random.default_rng(0x5C1), 200, 6),
           bind(SELECT_ROUNDS, E=6, TOPK=3))


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("E_, TOPK", [(8, 4), (6, 3), (5, 5)])
def test_the_selection_rounds_on_many_rows(E_, TOPK) -> None:
    """2,000 rows of each kind (8,000 special) at 8 experts and 4 slots, 6 and 3, 5 and 5 (every expert, a pick tree with
    padding); 15 to 30 s each."""
    _agree(bind(BM.MoeRouterSelect, E=E_, TOPK=TOPK), _probabilities(np.random.default_rng(0x5C2 + E_), 2000, E_),
           bind(SELECT_ROUNDS, E=E_, TOPK=TOPK))


@pytest.mark.slow
def test_the_router_top_k() -> None:
    """`MoeRouterTopKOrdered_v2` at 16 experts, 4 slots, 2 values per thread, on 1,000 rows of each kind (about 30 s)."""
    _agree(bind(BM.MoeRouterTopKOrdered, E=16, TOPK=4, VPT=2), _logits(np.random.default_rng(0x70), 1000, 16))


def test_the_exponential_passes_the_word_querys_cut_check() -> None:
    prog, _ = standalone_call(bind(BM.NvExpf))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok


@pytest.mark.slow
def test_the_router_passes_the_word_querys_cut_check() -> None:
    """The router's Calls (the shift, each exponential's halves and MUFU, the normalization, the selection) compute no value
    twice (about 25 s)."""
    prog, _ = standalone_call(bind(BM.MoeRouterTopKOrdered, E=8, TOPK=2, VPT=4))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok


# --- the sigmoid router ----------------------------------------------------------------------------------------------------------

#: x: tanhf's range ends (|x/2| at 0x3F19999A and 0x41102CB4: x at 0x3F99999A and 0x41902CB4) within 3 ulps, of either sign, and
#: around 1, 1.5, 2 and the smallest normal; with F32_SPECIALS
SIGMOID_EDGES = np.array([(c + d) | s for c in (0x3F99999A, 0x41902CB4, 0x3F800000, 0x3FC00000, 0x40000000, 0x00800000)
                          for d in range(-3, 4) for s in (0, 0x80000000)], np.uint32)


def _score_words(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, b): F32_SPECIALS and SIGMOID_EDGES, n random words and n values in the sigmoid's ranges and below, against random
    words, biases in [-1, 1] and every 37th bias ``-sigmoid(x)`` (the score's +-0)."""
    x = np.concatenate([F32_SPECIALS, SIGMOID_EDGES, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32),
                        rng.uniform(-40, 40, n).astype(np.float32).view(np.uint32), rng.uniform(-1.3, 1.3, n).astype(np.float32).view(np.uint32)])
    b = np.where(rng.integers(0, 4, len(x)) == 0, rng.integers(0, 1 << 32, len(x), dtype=np.uint64).astype(np.uint32),
                 rng.uniform(-1, 1, len(x)).astype(np.float32).view(np.uint32))
    b[::37] = [moe_twins.nv_sigmoid_bits(int(w)) ^ 0x80000000 for w in x[::37]]
    return x, b


def _agree2(target, args, word=None) -> None:
    word = word or TR.word_of(target)
    n = len(args[0])
    got, want = _rows(BS.evaluate_words(target, *args), n), _rows(E.evaluate_batch(word, *args), n)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} rows differ from {word.id}, first {bad[:5]}"


@pytest.mark.slow
def test_the_sigmoid_score_on_special_random_and_in_range_words() -> None:
    _agree2(bind(BM.MoeRouterSigmoidScore), _score_words(np.random.default_rng(0x51), 2000))


@pytest.mark.boolean_sweep
@pytest.mark.slow
def test_the_sigmoid_score_on_many_words() -> None:
    """100,000 words of each kind (about 60 s)."""
    _agree2(bind(BM.MoeRouterSigmoidScore), _score_words(np.random.default_rng(0x52), 100_000))


def _bias_slots(B, S, s, v):
    """`MoeRouterSigmoidBias_v1`'s slots and `MoeRouterSigmoidNorm_v1` on given selection scores and second values."""
    ids = [B.call(bind(M.MoeRouterFirst, E=S.E), s)]
    pairs = [B.call(bind(M.MoeRouterPickPair, E=S.E), s, v, ids[0])]
    for _ in range(1, S.TOPK):
        ids.append(B.call(bind(M.MoeRouterBiasNext, E=S.E), s, pairs[-1][0], ids[-1]))
        pairs.append(B.call(bind(M.MoeRouterPickPair, E=S.E), s, v, ids[-1]))
    out = B.call(bind(M.MoeRouterSigmoidNorm, TOPK=S.TOPK, SCALE=S.SCALE, KERNEL=S.KERNEL), array_of([p[0] for p in pairs]),
                 array_of([p[1] for p in pairs]))
    return tuple_of(out, array_of(ids))


BIAS_SLOTS = CompositeDefinition("BiasSlots", 1, ("E", "TOPK", "SCALE", "KERNEL"),
                                 lambda S: ((("s", Array(S.E, F32)), ("v", Array(S.E, F32))),
                                            Tuple(Array(S.TOPK, F32), Array(S.TOPK, I32))), _bias_slots, register=False)


def _selection_scores(rng, n: int, E_: int) -> tuple[np.ndarray, np.ndarray]:
    """(s, v), n rows of each kind, s as `MoeRouterBiasScore_v1` makes them (no NaN, no -0): random words, values in [-1, 1],
    multiples of 0.125 (ties) and rows mostly -inf (fewer finite experts than slots, or none); v random words."""
    s = np.concatenate([rng.integers(0, 1 << 32, (n, E_), dtype=np.uint64).astype(np.uint32),
                        rng.uniform(-1, 1, (n, E_)).astype(np.float32).view(np.uint32),
                        (rng.integers(-2, 3, (n, E_)) * 0.125).astype(np.float32).view(np.uint32),
                        np.where(rng.integers(0, 4, (n, E_)) == 0, rng.uniform(-1, 1, (n, E_)).astype(np.float32).view(np.uint32),
                                 np.uint32(0xFF800000))])
    s = np.where(((s >> 23) & 0xFF == 0xFF) & (s & 0x7FFFFF != 0), np.uint32(0xFF800000), s)
    s = np.where(s == 0x80000000, np.uint32(0), s)
    return s, rng.integers(0, 1 << 32, s.shape, dtype=np.uint64).astype(np.uint32)


@pytest.mark.slow
@pytest.mark.parametrize("kernel", M.ROUTER_KERNELS)
def test_the_bias_slots(kernel) -> None:
    st = {"E": 6, "TOPK": 3, "SCALE": 2.5, "KERNEL": kernel}
    _agree2(bind(BM.MoeRouterBiasTopK, **st), _selection_scores(np.random.default_rng(0x5B1), 200, 6), bind(BIAS_SLOTS, **st))


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("kernel", M.ROUTER_KERNELS)
@pytest.mark.parametrize("E_, TOPK", [(16, 8), (5, 5), (8, 1)])
def test_the_bias_slots_on_many_rows(kernel, E_, TOPK) -> None:
    """2,000 rows of each kind at 16 experts and 8 slots, 5 and 5 (every expert, a pick tree with padding), 8 and 1; 10 to 40 s
    each."""
    st = {"E": E_, "TOPK": TOPK, "SCALE": 2.5, "KERNEL": kernel}
    _agree2(bind(BM.MoeRouterBiasTopK, **st), _selection_scores(np.random.default_rng(0x5B2 + E_), 2000, E_), bind(BIAS_SLOTS, **st))


def _router_rows(E_: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from tests.program.test_moe_router_sigmoid import rows

    r = rows(E_, n, seed)
    return np.array([x for x, _ in r], np.uint32), np.array([b for _, b in r], np.uint32)


@pytest.mark.slow
@pytest.mark.parametrize("kernel", M.ROUTER_KERNELS)
def test_the_sigmoid_router(kernel) -> None:
    """`MoeRouterSigmoidBias_v2` at 8 experts and 2 slots on the kernel tests' rows (ties, a deciding bias, the sigmoid's ranges,
    non-finite logits and biases, fewer finite experts than slots and none)."""
    _agree2(bind(BM.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL=kernel), _router_rows(8, 300, 0x5C))


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("kernel", M.ROUTER_KERNELS)
def test_the_sigmoid_router_at_eight_slots(kernel) -> None:
    """At 16 experts and 8 slots, 1,000 rows (about 20 s each)."""
    _agree2(bind(BM.MoeRouterSigmoidBias, E=16, TOPK=8, SCALE=2.5, KERNEL=kernel), _router_rows(16, 1000, 0x5D))


def test_the_sigmoid_routers_word_views() -> None:
    assert TR.word_of(bind(BM.MoeRouterSigmoidScore)).id == "MoeRouterSigmoidScore_v1"
    st = {"E": 256, "TOPK": 8, "SCALE": 2.5, "KERNEL": "block"}
    assert SW.boolean_version(bind(M.MoeRouterSigmoidBias, **st)).id == "MoeRouterSigmoidBias_v2{E=256,TOPK=8,SCALE=2.5,KERNEL='block'}"
    assert TR.word_of(bind(BM.MoeRouterSigmoidBias, **st)).id == bind(M.MoeRouterSigmoidBias, **st).id


@pytest.mark.slow
@pytest.mark.parametrize("kernel", M.ROUTER_KERNELS)
def test_the_sigmoid_router_passes_the_word_querys_cut_check(kernel) -> None:
    """Each value the router reads is decoded in one of its Calls (each expert's score halves and MUFUs, the slots): the cut
    check finds no value computed twice (about 60 s each)."""
    prog, _ = standalone_call(bind(BM.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL=kernel))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
