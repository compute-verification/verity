"""The block-FP8 fused-MoE expert GEMMs with the target's e4m3 k32 step (`fp8_moe` `MoeExpertFp8Block*_v3{..., DOT}`), their
expert-indexed pair forms (`fp8_moe_experts` `MoeExpertFp8Gemm(W)_v1{E,K,N,G,KB,SWAP,DOT}`), and their binding on the declared target
(`targets.moe_expert_fp8_gemm_spec`, keyed on `GemmTarget.fp8_dot`):

* with Hopper's step the `_v3` forms are the `_v1` Definitions, and with sm_120's they are its chain (`moe_model` with the step);
* one 32-wide tile the two steps sum differently, which the w13 word shows at BLOCK_SIZE_K 32, the sm_120 launch's;
* `fp8_moe_rows`'s twin with `dot="blackwell"` is the `_v3{DOT=BlackwellE4m3QmmaDot32_v1}` GEMM, special and subnormal words included;
* the pair forms at expert e are the `_v3` GEMMs over that expert's rows and scales, and an id outside [0, E) selects NaN words;
* cc 12.0 binds them with `BlackwellE4m3QmmaDot32_v1`, and the targets that record no e4m3 step refuse by name.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity.ml.prims import BlackwellE4m3QmmaDot32 as BW
from verity.ml.prims import HopperE4m3QgmmaDot32 as HP
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import fp8_moe_rows as MR
from verity_vllm.program.registry import fp8_moe as M
from verity_vllm.program.registry import fp8_moe_experts as X
from verity_vllm.program.registry import targets as T
from tests.program.test_fp8_moe import ONE, _bf16, _ffma, _fmul, _operands, _twin_agrees, _wide

PAIRS = [(M.MoeExpertFp8BlockCoordinate, M.MoeExpertFp8BlockCoordinateV3), (M.MoeExpertFp8BlockCoordinateW, M.MoeExpertFp8BlockCoordinateWV3),
         (M.MoeExpertFp8BlockGemm, M.MoeExpertFp8BlockGemmV3), (M.MoeExpertFp8BlockGemmW, M.MoeExpertFp8BlockGemmWV3)]


def _model(dot, x, sx, w, sw, K: int, G: int, KB: int, g: int | None = None) -> int:
    """`fused_moe_kernel`'s block-wise branch without SWAP_AB over the step `dot`: per tile, `dot` from +0, then
    `acc = FFMA(FMUL(d, sx), sw, acc)`; the routed weight's FMUL; the bf16 cast."""
    acc = 0
    for k in range(K // KB):
        d = 0
        for lo in range(k * KB, (k + 1) * KB, 32):
            d = dot.evaluate(d, *x[lo:lo + 32], *w[lo:lo + 32])
        acc = _ffma(_fmul(d, sx[k * KB // G]), sw[k * KB // G], acc)
    return _bf16(acc if g is None else _fmul(acc, g))


def _args(fam, x, sx, g, w, ws, rows: bool):
    weighted = "W" in fam.name.removeprefix("MoeExpertFp8Block")
    lead = (x, sx, g) if weighted else (x, sx)
    return lead + ((sum(w, []), sum(ws, [])) if rows else (w[0], ws[0]))


@pytest.mark.parametrize("v1,v3", PAIRS, ids=[p[1].name for p in PAIRS])
def test_the_dot_forms_with_the_hopper_step_are_the_v1_definitions(v1, v3) -> None:
    K, N, G = 256, 72, 64
    rows = "Gemm" in v1.name
    rng = np.random.default_rng(len(v1.name))
    x, sx, w, ws = _operands(rng, K, G, N)
    g = int(np.float32(0.4375).view(np.uint32))
    for KB, SWAP in ((32, False), (64, True)):
        st = {"K": K, "G": G, "KB": KB, "SWAP": SWAP, **({"N": N} if rows else {})}
        args = _args(v1, x, sx, g, w, ws, rows)
        assert evaluate(bind(v3, **st, DOT=HP), *args) == evaluate(bind(v1, **st), *args)


def test_the_dot_forms_with_the_sm120_step_are_its_chain() -> None:
    """The served statics (G = 64, KB = 32, no SWAP_AB) over random rows: w13's and w2's words are `_model` with sm_120's step, and
    Hopper's step gives other words on the same operands."""
    K, N, G, KB = 256, 72, 64, 32
    x, sx, w, ws = _operands(np.random.default_rng(11), K, G, N)
    g = int(np.float32(0.8125).view(np.uint32))
    up = evaluate(bind(M.MoeExpertFp8BlockGemmV3, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=BW), x, sx, sum(w, []), sum(ws, []))
    down = evaluate(bind(M.MoeExpertFp8BlockGemmWV3, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=BW), x, sx, g, sum(w, []), sum(ws, []))
    assert up == tuple(_model(BW, x, sx, w[n], ws[n // G], K, G, KB) for n in range(N))
    assert down == tuple(_model(BW, x, sx, w[n], ws[n // G], K, G, KB, g) for n in range(N))
    hopper = evaluate(bind(M.MoeExpertFp8BlockGemmV3, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=HP), x, sx, sum(w, []), sum(ws, []))
    assert hopper != up, "the operands do not tell the two e4m3 steps apart"


def test_one_tile_tells_the_steps_apart() -> None:
    """Products 1, 2^-8 and 2^-18 in one tile: Hopper's 14-bit adder drops 2^-18, so 1 + 2^-8, a bf16 tie, rounds to even (0x3F80);
    sm_120's 26-bit one keeps it, past the tie (0x3F81).  The word and the twin show it at the served BLOCK_SIZE_K 32."""
    x, w = [0] * 32, [0] * 32
    x[0] = w[0] = 0x38
    x[1] = w[1] = 0x18
    x[2] = w[2] = 0x01
    assert (HP.evaluate(0, *x, *w), BW.evaluate(0, *x, *w)) == (0x3F808000, 0x3F808020)
    for dot, name, want in ((HP, "hopper", 0x3F80), (BW, "blackwell", 0x3F81)):
        fn = bind(M.MoeExpertFp8BlockCoordinateV3, K=32, G=32, KB=32, SWAP=False, DOT=dot)
        assert evaluate(fn, x, [ONE], w, [ONE]) == (want,)
        one = np.array([ONE], np.uint32)
        got, _ = MR.moe_expert_fp8_block_row(np.array(x, np.uint8), one, np.array([w], np.uint8), one[None], G=32, KB=32, dot=name)
        assert got.tolist() == [want]


@pytest.mark.parametrize("W,wide", [(False, False), (True, False), (False, True), (True, True)])
def test_the_twin_with_the_sm120_step_is_the_gemm(W: bool, wide: bool) -> None:
    """`moe_expert_fp8_block_row(dot="blackwell")` against `MoeExpertFp8BlockGemm(W)_v3{DOT=BlackwellE4m3QmmaDot32_v1}` over a ragged
    N, with the wide scales' subnormal and overflowing columns."""
    K, N, G = 256, 200, 32 if wide else 64
    rng = np.random.default_rng(3 + 2 * W + 4 * wide)
    x, sx, w, ws = (_wide if wide else _operands)(rng, K, G, N)
    g = int(np.float32(rng.uniform(0.01, 1.0)).view(np.uint32))
    fam = M.MoeExpertFp8BlockGemmWV3 if W else M.MoeExpertFp8BlockGemmV3
    want = evaluate(bind(fam, K=K, N=N, G=G, KB=32, SWAP=False, DOT=BW), *((x, sx, g) if W else (x, sx)), sum(w, []), sum(ws, []))
    got, nonfinite = MR.moe_expert_fp8_block_row(np.array(x), np.array(sx), np.array(w), np.array(ws), G=G, KB=32, g=g if W else None,
                                                 dot="blackwell")
    _twin_agrees(want, got, nonfinite)
    assert nonfinite.any() and not nonfinite.all() if wide else not nonfinite.any()
    assert MR.TWIN_DOT_OF[BW.id] == "blackwell" and MR.TWIN_DOT_OF[HP.id] == "hopper"


@pytest.mark.parametrize("W", [False, True])
def test_the_pair_forms_gather_the_routed_experts_rows(W: bool) -> None:
    """`MoeExpertFp8Gemm(W)_v1` at expert e is the `_v3` GEMM over w[e] and ws[e] (N = 72: a whole scale block and a partial one), and
    an id outside [0, E) selects NaN weights and scales."""
    E, K, N, G = 3, 128, 72, 64
    rng = np.random.default_rng(17 + W)
    x, sx, _, _ = _operands(rng, K, G)
    stack = [_operands(rng, K, G, N)[2:] for _ in range(E)]
    w, ws = sum((s[0] for s in stack), []), sum((s[1] for s in stack), [])
    g = int(np.float32(0.6875).view(np.uint32))
    pair = bind(X.MoeExpertFp8GemmW if W else X.MoeExpertFp8Gemm, E=E, K=K, N=N, G=G, KB=32, SWAP=False, DOT=BW)
    block = bind(M.MoeExpertFp8BlockGemmWV3 if W else M.MoeExpertFp8BlockGemmV3, K=K, N=N, G=G, KB=32, SWAP=False, DOT=BW)
    lead = (x, sx, g) if W else (x, sx)
    for e in range(E):
        args = (x, sx, e, g) if W else (x, sx, e)
        assert evaluate(pair, *args, sum(w, []), sum(ws, [])) == evaluate(block, *lead, sum(stack[e][0], []), sum(stack[e][1], []))
    for e in (E, (1 << 32) - 1):
        out = evaluate(pair, *((x, sx, e, g) if W else (x, sx, e)), sum(w, []), sum(ws, []))
        assert all(v & 0x7FFF > 0x7F80 for v in out), "an id outside the experts does not select NaN"


def test_the_gathers_are_column_reads() -> None:
    col8, col32 = [0x01, 0x40, 0xC5], [ONE, 0x40000000, 0xBF800000]
    for e in range(3):
        assert X.gather_e4m3(3).evaluate(e, *col8) == col8[e]
        assert X.gather_f32(3).evaluate(e, *col32) == col32[e]
    assert (X.gather_e4m3(3).evaluate(3, *col8), X.gather_f32(3).evaluate((1 << 32) - 1, *col32)) == (0x7F, 0x7FFFFFFF)
    assert X.gather_e4m3(3).id == "GatherE4m3x3_v1" and X.gather_f32(128).id == "GatherF32x128_v1"


def _tp(cc):
    return TargetProfile(compute_capability=cc)


def test_cc12_binds_the_sm120_step_and_the_others_refuse() -> None:
    assert T.moe_expert_fp8_dot(_tp((12, 0))) is BW
    assert T.moe_expert_fp8_gemm_spec(_tp((12, 0)), 128, 4096, 384, 64, 32, False, routed=False).id == \
        "MoeExpertFp8Gemm_v1{E=128,K=4096,N=384,G=64,KB=32,SWAP=False,DOT=BlackwellE4m3QmmaDot32_v1}"
    assert T.moe_expert_fp8_gemm_spec(_tp((12, 0)), 128, 192, 4096, 64, 32, False, routed=True).id == \
        "MoeExpertFp8GemmW_v1{E=128,K=192,N=4096,G=64,KB=32,SWAP=False,DOT=BlackwellE4m3QmmaDot32_v1}"
    for cc in ((8, 0), (8, 9), (9, 0)):
        with pytest.raises(T.UnregisteredTarget, match="no registered e4m3 step"):
            T.moe_expert_fp8_gemm_spec(_tp(cc), 128, 4096, 384, 64, 32, False, routed=False)
    with pytest.raises(T.UnregisteredTarget):
        T.moe_expert_fp8_dot(_tp((10, 0)))
