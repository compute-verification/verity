"""`verity.ml.definitions.registry.boolean_marlin_moe`: the Boolean versions of `marlin_moe`'s Definitions agree with the numpy twin
(`kernels.marlin_moe_rows`) and their word Definitions: the group on every code at every scale byte, and its quiet-NaN form; the
divider on edge and random words; the schedule on random and edge words; the steps on random operands at every special scale byte
and on special activations, and their halves cut at each word of x, which have exactly the step's gates; the coordinates over every cut
pattern; a pair launch of both kinds over two experts, one outside the expert axis; and the switch finds each word GEMM's Boolean
version."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.types import Array
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import marlin_moe_rows as MR
from verity.ml.definitions.registry import boolean_marlin_moe as BM
from verity.ml.definitions.registry import marlin_moe as MM

DOT = BG.HopperBF16WgmmaDot16
I32_EDGES = [0, 1, 2, 3, 7, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFF9]
SCALE_EDGES = [0, 1, 2, 126, 127, 128, 252, 253, 254, 255]
BF16_SPECIALS = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF]


def _bf16_rows(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _every_code_at_every_scale() -> tuple[np.ndarray, np.ndarray]:
    return np.tile(np.r_[np.arange(16), np.arange(16)], (256, 1)).astype(np.uint8), np.arange(256, dtype=np.uint8)[:, None]


def test_group_is_the_twin_on_every_code_at_every_scale() -> None:
    q, s = _every_code_at_every_scale()
    got = np.asarray(BS.evaluate_words(bind(BM.MarlinMxfp4GroupV2), q, s)).reshape(256, 32)
    assert (got == MR.dequant(q, s)).all()


def test_the_steps_dequant_differs_only_in_its_nans() -> None:
    """The steps' words: each NaN of the group's (F2FP's 0x7FFF) is the quiet 0x7FC0 of its code's sign, every other word the
    group's."""
    probe = definition("MarlinQuietDequantProbe", 1, lambda C, *e: BM.dequant(C, e[:-1], e[-1], f2fp_nan=False),
                       *MM.group_sig(BM.E2M1, BM.E8M0, BM.BF16)(None), register=False)
    q, s = _every_code_at_every_scale()
    got = np.asarray(BS.evaluate_words(bind(probe), q, s)).reshape(256, 32)
    want = MR.dequant(q, s)
    nan = want == 0x7FFF
    assert nan.any() and (~nan).any()
    assert (got == np.where(nan, 0x7FC0 | (q.astype(np.uint16) >> 3) << 15, want)).all()


def test_divider_is_c_int_on_edge_and_random_words() -> None:
    probe = definition("MarlinDivRemProbe", 1, lambda C, a, b: list(BM.divrem(C, a, b)), (("a", BM.I32), ("b", BM.I32)),
                       Array(2, BM.I32), register=False)
    rng = np.random.default_rng(7)
    pairs = list(itertools.product(I32_EDGES, I32_EDGES))
    pairs += [(int(a), int(b)) for a, b in zip(rng.integers(0, 1 << 32, 200, dtype=np.uint64), rng.integers(0, 1 << 32, 200, dtype=np.uint64))]
    pairs += [(int(a), int(b)) for a, b in zip(rng.integers(0, 5000, 200), rng.integers(1, 300, 200))]
    a, b = (np.array(c, dtype=np.uint32) for c in zip(*pairs))
    got = np.asarray(BS.evaluate_words(bind(probe), a, b)).reshape(len(pairs), 2)
    assert [tuple(int(v) for v in r) for r in got] == [MM.i32_divrem(x, y) for x, y in pairs]


@pytest.mark.parametrize("KT,NT,GRID", [(4, 3, 7), (5, 2, 132)])
def test_schedule_is_the_twin_and_the_word_schedule(KT: int, NT: int, GRID: int) -> None:
    rng = np.random.default_rng(KT)
    edges = np.array(I32_EDGES, dtype=np.uint32)
    blk = np.r_[rng.integers(0, 40, 200), rng.integers(0, 1 << 32, 40, dtype=np.uint64), np.repeat(edges, len(edges))].astype(np.uint32)
    par = np.r_[rng.integers(0, 60, 200), rng.integers(0, 1 << 32, 40, dtype=np.uint64), np.tile(edges, len(edges))].astype(np.uint32)
    got = np.asarray(BS.evaluate_words(bind(BM.MarlinMoeScheduleV2, KT=KT, NT=NT, GRID=GRID), blk, par)).reshape(len(blk), -1)
    want = np.stack([MR.schedule(KT, NT, GRID, int(b), int(p)).ravel() for b, p in zip(blk, par)])
    assert (got == want).all()
    word = bind(MM.MarlinMoeSchedule, KT=KT, NT=NT, GRID=GRID)
    for i in range(0, len(blk), 17):
        assert list(evaluate(word, [int(blk[i])], [int(par[i])])) == got[i].tolist()


@pytest.mark.parametrize("from_zero", [False, True])
def test_step_is_its_word_step_at_every_special_scale(from_zero: bool) -> None:
    rng = np.random.default_rng(11 + from_zero)
    n = 96
    x = _bf16_rows(rng, (n, MM.GROUP))
    x[np.arange(len(BF16_SPECIALS)), rng.integers(0, MM.GROUP, len(BF16_SPECIALS))] = BF16_SPECIALS
    q = rng.integers(0, 16, (n, MM.GROUP)).astype(np.uint8)
    s = np.r_[np.repeat(SCALE_EDGES, 4), rng.integers(110, 135, n - 4 * len(SCALE_EDGES))].astype(np.uint8)
    acc = rng.standard_normal(n).astype(np.float32).view(np.uint32)
    acc[:4] = [0, 0x80000000, 0x7F800000, 0x7FC00000]
    lead = () if from_zero else (acc,)
    fam, word = (BM.MarlinMxfp4StepFromZeroV2, MM.MarlinMxfp4StepFromZero) if from_zero else (BM.MarlinMxfp4StepV2, MM.MarlinMxfp4Step)
    got = np.asarray(BS.evaluate_words(bind(fam, DOT=DOT), *lead, x, q, s)).reshape(n)
    ref = bind(word, DOT=MM.DOT)
    for i in range(n):
        args = ([[int(acc[i])]] if not from_zero else []) + [x[i].tolist(), q[i].tolist(), [int(s[i])]]
        assert evaluate(ref, *args) == (int(got[i]),), i
    own, column = ((BM.MarlinMxfp4StepFromZeroActivation, BM.MarlinMxfp4StepFromZeroColumn) if from_zero
                   else (BM.MarlinMxfp4StepActivation, BM.MarlinMxfp4StepColumn))
    S = SimpleNamespace(DOT=DOT)
    assert {k: MM.GROUP * own.traced(S).counts[k] + column.traced(S).counts[k] for k in ("and", "xor", "not")} == fam.traced(S).counts
    xd = np.asarray(BS.evaluate_words(bind(own, DOT=DOT), x.reshape(-1))).reshape(n, -1)
    assert (np.asarray(BS.evaluate_words(bind(column, DOT=DOT), *lead, x, q, s, xd)).reshape(n) == got).all()


@pytest.mark.parametrize("routed", [False, True])
def test_coordinate_is_the_twin_over_every_cut_pattern(routed: bool) -> None:
    rng = np.random.default_rng(5 + routed)
    K, TK = 4 * 64, 64
    cut = np.array(list(itertools.product([False, True], repeat=K // TK - 1)), dtype=bool)
    reps = 8
    cut = np.repeat(cut, reps, axis=0)
    n = len(cut)
    x = _bf16_rows(rng, (n, K))
    wq = rng.integers(0, 16, (n, K)).astype(np.uint8)
    ws = rng.integers(118, 130, (n, K // 32)).astype(np.uint8)
    g = _bf16_rows(rng, (n,))
    fam = BM.MarlinMxfp4CoordinateWV2 if routed else BM.MarlinMxfp4CoordinateV2
    lead = (x, cut.astype(np.uint8)) + ((g,) if routed else ())
    got = np.asarray(BS.evaluate_words(bind(fam, K=K, TK=TK, DOT=DOT), *lead, wq, ws)).reshape(n)
    assert (got == MR.coordinates(x, cut, wq, ws, TK=TK, g=g if routed else None)).all()


def test_an_expert_row_is_its_word_row_on_packed_bytes() -> None:
    """The bits of a packed byte, least significant first, are its two codes in k order: the Boolean row relabels the gathered
    bytes and equals the word row, which unpacks them, an expert outside [0, E) included."""
    rng = np.random.default_rng(17)
    E, K, n = 3, 64, 12
    slab = rng.integers(0, 256, (n, E * K // 2)).astype(np.uint8)
    e = np.r_[rng.integers(0, E, n - 2), E, 0xFFFFFFFF].astype(np.uint32)
    got = np.asarray(BS.evaluate_words(bind(BM.MoeExpertE2m1RowV2, E=E, K=K), e, slab)).reshape(n, K)
    word = bind(MM.MoeExpertE2m1Row, E=E, K=K)
    for i in range(n):
        assert list(evaluate(word, [int(e[i])], slab[i].tolist())) == got[i].tolist(), i


@pytest.mark.parametrize("routed", [False, True])
def test_pair_launch_over_two_experts_is_the_twin(routed: bool) -> None:
    rng = np.random.default_rng(13 + routed)
    E, K, N, TK, TN, GRID = 2, 128, 2, 64, 1, 3
    wq = rng.integers(0, 16, (E, N, K)).astype(np.uint8)
    ws = rng.integers(118, 130, (E, N, K // 32)).astype(np.uint8)
    cases = [(0, 0, 1), (1, 1, 2), (0, 2, 3), (1, 0, 5), (2, 1, 2)]
    P = len(cases)
    e, blk, par = (np.array(c, dtype=np.uint32) for c in zip(*cases))
    x = _bf16_rows(rng, (P, K))
    g = np.array(rng.random(P), dtype=np.float32).view(np.uint32)
    fam = BM.MarlinMoeExpertGemmWV2 if routed else BM.MarlinMoeExpertGemmV2
    lead = (x, e) + ((g,) if routed else ()) + (blk, par)
    packed = MR.pack_e2m1(wq).ravel()
    slabs = (np.broadcast_to(packed, (P, packed.size)), np.broadcast_to(ws.ravel(), (P, ws.size)))
    got = np.asarray(BS.evaluate_words(bind(fam, E=E, K=K, N=N, TK=TK, TN=TN, GRID=GRID, DOT=DOT), *lead, *slabs)).reshape(P, N)
    for i, (ei, b, p) in enumerate(cases):
        if ei >= E:
            assert (got[i] == 0x7FFF).all(), "an expert outside [0, E) reads NaN weights"
            continue
        want = MR.launch(x[i:i + 1], np.array([b]), p, np.array([ei]), wq, ws, np.arange(N), TK=TK, TN=TN, GRID=GRID,
                         g=g[i:i + 1] if routed else None)
        assert got[i].tolist() == want[0].tolist(), (routed, ei, b, p)


def test_the_switch_finds_each_gemms_boolean_version() -> None:
    st = dict(E=384, K=384, N=7168, TK=128, TN=64, GRID=264)
    for word, bits in ((MM.MarlinMoeExpertGemm, BM.MarlinMoeExpertGemmV2), (MM.MarlinMoeExpertGemmW, BM.MarlinMoeExpertGemmWV2)):
        assert SW.boolean_version(bind(word, **st, DOT=MM.DOT)).id == bind(bits, **st, DOT=DOT).id
