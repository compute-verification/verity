"""`verity_vllm.program.registry.boolean_moe`: the Boolean versions of `registry.moe`'s element-wise Definitions agree with their word
Definitions (`trace.word_of`): the routed sum's coordinate on every word at one slot, on every pair of special words, every word
against 8 specials and random pairs at two, and at OLMoE's eight slots on random rows, rows of special words, rows that cancel to a
few ulps and rows that overflow. They are pinned by gate count and program digest and pass `Q_word_v1{X=16, W=32}`'s cut check."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir import partition_object as PO
from verity.ir.defs import REGISTRY, bind
from verity.ml.boolean import trace as TR
from verity_vllm.program.registry import boolean_moe as BM
from verity_vllm.program.registry import moe as M

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeSumCoordinate_v2{TOPK=8}": (5222, 8395, 764, "6b71d820f6508d24d85de35f383ab531d9efe382a16ce28650cf649041cdb26e"),
    "MoeSum_v2{TOPK=8,H=8}": (41776, 67160, 6112, "9ba2b3bb9d5faad5be0299738869c260df30577c2e6245364250ea81af55f02f"),
}
TARGETS = [bind(BM.MoeSumCoordinate, TOPK=8), bind(BM.MoeSum, TOPK=8, H=8)]

_EXP = [0, 1, 2, 126, 127, 128, 133, 134, 135, 253, 254, 255]
#: per sign: zero, subnormals, ones and their neighbours, the finite maximum and its neighbours, infinities and NaNs
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)
EVERY = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got = _bits(target, *inputs)
    want = np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


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


def test_the_word_views() -> None:
    assert TR.word_of(bind(BM.MoeSumCoordinate, TOPK=8)).id == bind(M.MoeSumCoordinate, TOPK=8).id == "MoeSumCoordinate_v1{TOPK=8}"
    assert TR.word_of(bind(BM.MoeSum, TOPK=8, H=8)).id == bind(M.MoeSum, TOPK=8, H=8).id == "MoeSum_v1{TOPK=8,H=8}"


def test_one_slot_on_every_word() -> None:
    """At one slot the sum is ``bf16_rn(0 + f32(x))``: x, but -0 is +0 and every NaN 0x7FFF."""
    _agree(bind(BM.MoeSumCoordinate, TOPK=1), EVERY.reshape(-1, 1))


#: signed zeros and ones, the smallest subnormal, the finite maximum, -inf and a NaN
AGAINST = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x7F7F, 0xFF80, 0x7FC0], np.uint16)


@pytest.mark.slow
def test_two_slots_on_special_pairs_every_word_and_random_pairs() -> None:
    """Every pair of special words, every word against 8 special words in either slot, and 100,000 random pairs (about 30 s)."""
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
