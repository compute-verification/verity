"""Marlin's W4A16 MXFP4 expert GEMMs (`registry.marlin_moe`), the DeepSeek-V4-Pro MoE's on Hopper:

* the dequant primitives on every code and scale byte, and a group against its 32 products;
* the schedule against the template's partition written out here (init_slice / init_part2_slice's segment starts);
* a split-K coordinate against its fold written out from the primitives: each segment's warp tree, folded from the highest k down;
* the numpy twin (`kernels.marlin_moe_rows`) the H200 capture compares the kernel with, against the Definitions: the group, the
  schedule, the coordinates and a whole pair launch over three experts;
* `auto_config` on the shapes DeepSeek-V4-Pro's TP8 rank launches.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.prims import F2fpBf16
from verity_vllm.program.kernels import marlin_moe_rows as MR
from verity_vllm.program.registry import marlin_moe as MM
from verity_vllm.program.registry.prims import Bf16MulBf16, F32Add

DOT = MM.DOT
E2M1_VALUES = [v for v in MM.E2M1_VALUES] + [-v for v in MM.E2M1_VALUES]


def _bf16(v: float) -> int:
    return int(np.array([v], dtype=np.float32).view(np.uint32)[0] >> 16)


def _bf16_rows(rng, shape) -> np.ndarray:
    s = np.exp2(rng.integers(-4, 4, shape[:-1] + (1,))).astype(np.float32)
    return ((rng.standard_normal(shape).astype(np.float32) * s).view(np.uint32) >> 16).astype(np.uint16)


def test_dequant_primitives_on_every_code_and_scale_byte() -> None:
    for q in range(16):
        v = E2M1_VALUES[q]
        want = 0x8000 if q == 8 else _bf16(v)
        assert MM.E2m1ToBf16.evaluate(q) == want == MR.E2M1_BF16[q], q
    for s in range(256):
        want = 0 if s == 0 else 0x7F80 if s == 255 else _bf16(2.0 ** (s - 127))
        assert MM.E8m0DequantBf16.evaluate(s) == want, s


def test_group_is_its_32_products_and_the_twin() -> None:
    rng = np.random.default_rng(1)
    for s in range(256):
        q = rng.permutation(np.r_[np.arange(16), np.arange(16)])
        got = evaluate(bind(MM.MarlinMxfp4Group), q.tolist(), [s])
        sb = MM.E8m0DequantBf16.evaluate(s)
        assert list(got) == [Bf16MulBf16.evaluate(MM.E2m1ToBf16.evaluate(int(c)), sb) for c in q], s
        assert list(got) == MR.dequant(q[None], np.array([[s]]))[0].tolist(), s
    # the specials: 0 times inf is the F2FP NaN, a scale of 0 keeps each code's sign, (0.5, 2^-126) is the subnormal 2^-127
    assert evaluate(bind(MM.MarlinMxfp4Group), [0] * 31 + [8], [255])[-2:] == (0x7FFF, 0x7FFF)
    assert evaluate(bind(MM.MarlinMxfp4Group), [9] * 32, [0])[0] == 0x8000
    assert evaluate(bind(MM.MarlinMxfp4Group), [1] * 32, [1])[0] == 0x0040
    assert evaluate(bind(MM.MarlinMxfp4Group), [15] * 32, [254])[0] == 0xFF80


def _template_cuts(KT: int, NT: int, GRID: int, blk: int, par: int) -> np.ndarray:
    """The segment starts of tile (blk, J), from the template's own loop: CTA b's part-2 run of `iters` (tile, k-tile) positions from
    iters * b, its slices split at tile ends (init_slice, init_part2_slice), part-1 tiles whole."""
    G = par * NT
    p2 = G
    if G > GRID:
        p2 = G % GRID
        if p2 * 3 <= GRID:
            p2 += GRID
    iters = -(-KT * p2 // GRID)
    starts: set[tuple[int, int]] = set()
    for b in range(GRID):
        pos, end = iters * b, min(iters * (b + 1), KT * p2)
        while pos < end:
            j2, row = divmod(pos, KT)
            starts.add((j2, row))
            pos += min(end - pos, KT - row)
    out = np.zeros((NT, KT - 1), dtype=bool)
    for J in range(NT):
        j2 = blk * NT + J - (G - p2)
        if j2 >= 0:
            for t in range(1, KT):
                out[J, t - 1] = (j2, t) in starts
    return out


@pytest.mark.parametrize("KT,NT,GRID", [(2, 1, 1), (4, 3, 7), (7, 2, 5), (56, 6, 132), (3, 56, 264), (6, 28, 528)])
def test_schedule_is_the_template_partition(KT: int, NT: int, GRID: int) -> None:
    sched = bind(MM.MarlinMoeSchedule, KT=KT, NT=NT, GRID=GRID)
    for par in sorted({1, 2, 3, GRID // NT, GRID // NT + 1, 2 * GRID // NT + 1, 40, 96} - {0}):
        for blk in sorted({0, 1, par // 2, par - 1} & set(range(par))):
            want = _template_cuts(KT, NT, GRID, blk, par)
            assert (MR.schedule(KT, NT, GRID, blk, par) == want).all(), (blk, par)
            if KT * NT <= 64:
                assert np.array(evaluate(sched, [blk], [par]), dtype=bool).reshape(NT, KT - 1).tolist() == want.tolist(), (blk, par)


def test_schedule_is_total_and_the_twin_on_any_word() -> None:
    sched = bind(MM.MarlinMoeSchedule, KT=4, NT=3, GRID=7)
    for blk, par in itertools.product([0, 5, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE]):
        got = np.array(evaluate(sched, [blk], [par]), dtype=bool).reshape(3, 3)
        assert (got == MR.schedule(4, 3, 7, blk, par)).all(), (hex(blk), hex(par))


def test_divrem_is_c_int() -> None:
    for a, b in itertools.product([0, 7, -7, 2**31 - 1, -2**31], [1, 3, -3, 2**31 - 1, -2**31, -1, 0]):
        q, r = MM.i32_divrem(a & MM.M32, b & MM.M32)
        if b == 0:
            assert (q, r) == (MM.M32, a & MM.M32)
            continue
        want_q = int(a / b) if abs(a) < 2**53 else (abs(a) // abs(b)) * (1 if (a < 0) == (b < 0) else -1)
        assert q == want_q & MM.M32 and r == (a - want_q * b) & MM.M32, (a, b)


def _fold_model(x, cut, wq, ws, TK: int) -> int:
    """The coordinate's f32 sum from the primitives: per segment (a run of k-tiles between cuts), each warp row's k16 chain from
    +0 and the CTA's tree acc_0 + (acc_1 + ...), and the segments' partials folded P_0 + (P_1 + (... + P_last))."""
    K = len(x)
    W, KT = TK // 32, K // TK
    w = [MM.Bf16MulBf16.evaluate(MM.E2m1ToBf16.evaluate(wq[k]), MM.E8m0DequantBf16.evaluate(ws[k // 32])) for k in range(K)]
    bounds = [0] + [t for t in range(1, KT) if cut[t - 1]] + [KT]
    parts = []
    for lo, hi in zip(bounds, bounds[1:]):
        acc = [0] * W
        for t in range(lo, hi):
            for r in range(W):
                for j in range(2):
                    k = t * TK + 32 * r + 16 * j
                    acc[r] = DOT.evaluate(acc[r], *x[k:k + 16], *w[k:k + 16])
        tree = {2: lambda a: F32Add.evaluate(a[0], a[1]),
                4: lambda a: F32Add.evaluate(a[0], F32Add.evaluate(a[1], F32Add.evaluate(a[3], a[2])))}[W](acc)
        parts.append(tree)
    R = parts[-1]
    for p in reversed(parts[:-1]):
        R = F32Add.evaluate(p, R)
    return R


@pytest.mark.parametrize("TK", [64, 128])
def test_split_k_coordinate_is_its_fold_and_the_twin(TK: int) -> None:
    rng = np.random.default_rng(TK)
    K = 4 * TK
    KT = K // TK
    coord = bind(MM.MarlinMxfp4Coordinate, K=K, TK=TK, DOT=DOT)
    coord_w = bind(MM.MarlinMxfp4CoordinateW, K=K, TK=TK, DOT=DOT)
    n = 2 ** (KT - 1)
    x = _bf16_rows(rng, (n, K))
    wq = rng.integers(0, 16, (n, K))
    ws = rng.integers(118, 128, (n, K // 32))
    cut = np.array(list(itertools.product([False, True], repeat=KT - 1)), dtype=bool)
    g = (np.array(rng.random(n), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    twin, twin_w = MR.coordinates(x, cut, wq, ws, TK=TK), MR.coordinates(x, cut, wq, ws, TK=TK, g=g)
    sums = set()
    for i in range(n):
        args = (x[i].tolist(), cut[i].astype(int).tolist())
        R = _fold_model(x[i].tolist(), cut[i], wq[i].tolist(), ws[i].tolist(), TK)
        sums.add(R)
        assert evaluate(coord, *args, wq[i].tolist(), ws[i].tolist()) == (F2fpBf16.evaluate(R),) == (twin[i],), i
        want_w = Bf16MulBf16.evaluate(F2fpBf16.evaluate(R), int(g[i]))
        assert evaluate(coord_w, *args, int(g[i]), wq[i].tolist(), ws[i].tolist()) == (want_w,) == (twin_w[i],), i
    assert len(sums) > 1, "every split of these operands gives one sum: the test reads no fold order"


def test_pair_launch_over_three_experts_is_the_twin() -> None:
    rng = np.random.default_rng(3)
    E, K, N, TK, TN, GRID = 3, 256, 16, 128, 8, 5
    wq = rng.integers(0, 16, (E, N, K))
    ws = rng.integers(118, 130, (E, N, K // 32))
    for routed, (e, blk, par) in itertools.product([False, True], [(0, 0, 1), (2, 1, 3), (1, 4, 5), (5, 0, 1)]):
        fam = MM.MarlinMoeExpertGemmW if routed else MM.MarlinMoeExpertGemm
        gemm = bind(fam, E=E, K=K, N=N, TK=TK, TN=TN, GRID=GRID, DOT=DOT)
        x = _bf16_rows(rng, (1, K))
        g = np.array([0.3], dtype=np.float32).view(np.uint32)
        lead = [x[0].tolist(), [e]] + ([[int(g[0])]] if routed else []) + [[blk], [par]]
        got = evaluate(gemm, *lead, wq.ravel().tolist(), ws.ravel().tolist())
        if e >= E:
            assert all(v == 0x7FFF for v in got), "an expert outside [0, E) reads NaN weights"
            continue
        want = MR.launch(x, np.array([blk]), par, np.array([e]), wq, ws, np.arange(N), TK=TK, TN=TN, GRID=GRID,
                         g=g if routed else None)
        assert list(got) == want[0].tolist(), (routed, e, blk, par)


def test_auto_config_on_the_rank_shapes() -> None:
    allow = {(128, 128): 2, (64, 128): 4, (128, 64): 4}
    # w13 (N 768, top_k 6) at one token: 6 tiles x 6 pairs x 4 = 144 under 132 x allow gives TN 128 one block per SM, and TN 64's
    # 288 two, which wins
    assert MR.auto_config(1, 768, 6, 132, allow) == (128, 64, 2)
    # sixteen tokens: 2304 is no longer small; the first 128-thread config allows 4
    assert MR.auto_config(16, 768, 6, 132, allow) == (64, 128, 4)
    # w2 (N 7168, top_k 1, prob_m 6M) at one token: 1344 is not small for any config
    assert MR.auto_config(6, 7168, 1, 132, allow) == (64, 128, 4)
    assert MR.auto_config(6, 7168, 1, 132, {(128, 128): 2}) == (128, 128, 2)
    with pytest.raises(ValueError):
        MR.auto_config(1, 768, 6, 132, {})
