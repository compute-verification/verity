"""The block-FP8 fused-MoE expert GEMMs at cc 9.0's small-batch tiles, `fp8_moe` `MoeExpertFp8Block*_v3{DOT=HopperE4m3MmaV2Dot32_v1}`:

* `fp8_moe.sm90_dot` is Triton 3.7.1's MMA rule on vLLM's tuned H200 tiles for GLM-5.3's TP8 expert shard, on the batch-invariant
  tile and at the rule's edges (the warps; a dot N that is a multiple of 8 but not of 16);
* the `_v3` forms with the MMAv2 step are its chain at the served statics (G = KB = 128, SWAP_AB on and off), and the wgmma step
  gives other words on the same operands;
* `fp8_moe_rows`'s twin with `dot="mmav2"` is that GEMM, special and subnormal words included, and is its step's two k16
  instructions, not the step's own word code;
* their Boolean versions (`_v4` and the pair forms' `_v2` at `DOT=HopperE4m3MmaV2Dot32_v2`) agree with their word views, and the
  switch maps each word view to them.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32 as MV
from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32 as HP
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import fp8_moe_rows as MR
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import fp8_moe as M
from tests.program.test_boolean_fp8 import _flat, _inputs
from tests.program.test_fp8_moe import _bf16, _ffma, _fmul, _operands, _twin_agrees, _wide

#: the entry of vLLM d9105ea80's fused_moe/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
#: each batch size M looks up (the nearest key, fused_moe.py:1442-1447): (BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps)
H200_TP8 = {1: (8, 32, 4), 2: (8, 32, 4), 3: (8, 32, 4), 4: (8, 64, 4), 6: (8, 64, 4), 8: (16, 64, 4), 16: (16, 128, 4),
            512: (16, 128, 4), 1024: (64, 128, 4), 4096: (64, 128, 4)}


def _swap(bm: int, bn: int) -> bool:
    return bm < 64 and bn >= 64


def test_sm90_dot_is_tritons_mma_rule_on_the_h200_tiles() -> None:
    """M <= 6 take MMAv2 (8 x 32, and 8 x 64 under SWAP_AB with the dot's N 8), every larger batch and the batch-invariant 64 x 64
    tile wgmma; 2 warps, or a dot N of 24 (a multiple of 8, not of 16), is MMAv2 again."""
    got = {m: M.sm90_dot(bm, bn, w, _swap(bm, bn)) for m, (bm, bn, w) in H200_TP8.items()}
    assert {m for m, d in got.items() if d is MV} == {1, 2, 3, 4, 6} and {m for m, d in got.items() if d is HP} == {8, 16, 512, 1024, 4096}
    assert M.sm90_dot(64, 64, 4, False) is HP and M.sm90_dot(64, 64, 8, False) is HP
    assert M.sm90_dot(64, 64, 2, False) is MV and M.sm90_dot(24, 64, 4, True) is MV and M.sm90_dot(32, 64, 4, True) is HP
    assert M.sm90_dot(16, 64, 4, False) is MV and M.sm90_dot(64, 16, 4, False) is HP


def _model(dot, x, sx, w, sw, K: int, G: int, KB: int, swap: bool, g: int | None = None) -> int:
    """`fused_moe_kernel`'s block-wise branch over the step `dot`: per tile, `dot` from +0, then `acc = FFMA(FMUL(d, first),
    second, acc)` with the scales swapped under SWAP_AB; the routed weight's FMUL; the bf16 cast."""
    acc = 0
    for k in range(K // KB):
        d = 0
        for lo in range(k * KB, (k + 1) * KB, 32):
            d = dot.evaluate(d, *x[lo:lo + 32], *w[lo:lo + 32])
        a, b = sx[k * KB // G], sw[k * KB // G]
        first, second = (b, a) if swap else (a, b)
        acc = _ffma(_fmul(d, first), second, acc)
    return _bf16(acc if g is None else _fmul(acc, g))


@pytest.mark.parametrize("SWAP", [False, True])
def test_the_dot_forms_with_the_mmav2_step_are_its_chain(SWAP: bool) -> None:
    """The served statics (G = KB = 128) over two K tiles and a whole and a partial weight block: w13's and w2's words are `_model`
    with the MMAv2 step, and the wgmma step gives other words on the same operands."""
    K, N, G, KB = 256, 136, 128, 128
    x, sx, w, ws = _operands(np.random.default_rng(23 + SWAP), K, G, N)
    g = int(np.float32(0.3125).view(np.uint32))
    st = {"K": K, "N": N, "G": G, "KB": KB, "SWAP": SWAP}
    up = evaluate(bind(M.MoeExpertFp8BlockGemmV3, **st, DOT=MV), x, sx, sum(w, []), sum(ws, []))
    down = evaluate(bind(M.MoeExpertFp8BlockGemmWV3, **st, DOT=MV), x, sx, g, sum(w, []), sum(ws, []))
    assert up == tuple(_model(MV, x, sx, w[n], ws[n // G], K, G, KB, SWAP) for n in range(N))
    assert down == tuple(_model(MV, x, sx, w[n], ws[n // G], K, G, KB, SWAP, g) for n in range(N))
    wgmma = evaluate(bind(M.MoeExpertFp8BlockGemmV3, **st, DOT=HP), x, sx, sum(w, []), sum(ws, []))
    assert wgmma != up, "the operands do not tell the two e4m3 steps apart"


@pytest.mark.parametrize("KB,SWAP,W,wide", [(128, True, False, False), (128, False, True, False), (32, True, True, True),
                                             (32, False, False, True)])
def test_the_twin_with_the_mmav2_step_is_the_gemm(KB: int, SWAP: bool, W: bool, wide: bool) -> None:
    """`moe_expert_fp8_block_row(dot="mmav2")` against `MoeExpertFp8BlockGemm(W)_v3{DOT=HopperE4m3MmaV2Dot32_v1}` over a ragged N,
    with the wide scales' subnormal and overflowing columns."""
    K, N, G = 256, 200, 32 if wide else 128
    rng = np.random.default_rng(31 + 2 * W + 4 * wide + 8 * SWAP)
    x, sx, w, ws = (_wide if wide else _operands)(rng, K, G, N)
    g = int(np.float32(rng.uniform(0.01, 1.0)).view(np.uint32))
    fam = M.MoeExpertFp8BlockGemmWV3 if W else M.MoeExpertFp8BlockGemmV3
    want = evaluate(bind(fam, K=K, N=N, G=G, KB=KB, SWAP=SWAP, DOT=MV), *((x, sx, g) if W else (x, sx)), sum(w, []), sum(ws, []))
    got, nonfinite = MR.moe_expert_fp8_block_row(np.array(x), np.array(sx), np.array(w), np.array(ws), G=G, KB=KB, swap=SWAP,
                                                 g=g if W else None, dot="mmav2")
    _twin_agrees(want, got, nonfinite)
    assert nonfinite.any() and not nonfinite.all() if wide else not nonfinite.any()
    assert MR.TWIN_DOT_OF[MV.id] == "mmav2"


def test_the_twin_step_is_two_k16_instructions_on_every_encoding() -> None:
    """`mmav2_dot32_e4m3` (core's bf16 k16 kernel twice on the widened bytes) against the step's word (`tc_dot_total_e4m3` on its
    two-group pipeline over the permuted operands) on random words, NaN bytes and non-finite accumulators included; the halves are
    the k mod 4 split."""
    rng = np.random.default_rng(41)
    n = 3000
    acc = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    acc[: n // 3] = (rng.standard_normal(n // 3) * 64).astype(np.float32).view(np.uint32)
    a, b = rng.integers(0, 256, (n, 32)).astype(np.uint8), rng.integers(0, 256, (n, 32)).astype(np.uint8)
    got, nonfinite = MR.mmav2_dot32_e4m3(acc, a, b)
    want = np.array([MV.evaluate(int(acc[i]), *a[i].tolist(), *b[i].tolist()) for i in range(n)], np.uint32)
    assert np.array_equal(got, want) and nonfinite.any() and not nonfinite.all()
    assert [h.tolist() for h in MR.MMAV2_HALVES] == [[k for k in range(32) if k % 4 in (0, 1)], [k for k in range(32) if k % 4 in (2, 3)]]
    assert MR.E4M3_AS_BF16[[0x38, 0xB8, 0x01, 0x7E, 0x00, 0x80]].tolist() == [0x3F80, 0xBF80, 0x3B00, 0x43E0, 0x0000, 0x8000]
    assert all((MR.E4M3_AS_BF16[e] & 0x7F80) == 0x7F80 and MR.E4M3_AS_BF16[e] & 0x7F for e in (0x7F, 0xFF))


#: on bits at the MMAv2 step: two tiles over one and two scale blocks with and without SWAP_AB, the GEMMs over a whole and a partial
#: weight block, and a pair form over 3 experts
BOOLEAN = [bind(BM.MoeExpertFp8BlockCoordinateV4, K=64, G=64, KB=32, SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32),
           bind(BM.MoeExpertFp8BlockCoordinateWV4, K=64, G=32, KB=32, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32),
           bind(BM.MoeExpertFp8BlockGemmV4, K=32, N=40, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32),
           bind(BM.MoeExpertFp8BlockGemmWV4, K=32, N=40, G=32, KB=32, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32),
           bind(BM.MoeExpertFp8CoordinateWV2, E=3, K=64, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32)]


@pytest.mark.parametrize("target", BOOLEAN, ids=lambda t: t.id)
def test_on_bits_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target) and word.defn.name == target.defn.name and word.bindings["DOT"] is MV
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 64, sum(map(ord, target.id)))
    if "E" in target.bindings:
        names = [n for n, _ in word.signature()[0]]
        i = names.index("e")
        inputs[i] = np.arange(np.asarray(inputs[i]).size, dtype=np.asarray(inputs[i]).dtype).reshape(np.shape(inputs[i])) % 4
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
