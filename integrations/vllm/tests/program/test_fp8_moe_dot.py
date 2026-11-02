"""The block-FP8 fused-MoE expert GEMMs with the target's e4m3 k32 step (`fp8_moe` `MoeExpertFp8Block*_v3{..., DOT}`), their
expert-indexed pair forms (`fp8_moe_experts` `MoeExpertFp8Gemm(W)_v1{E,K,N,G,KB,SWAP,DOT}`), and their binding on the declared target
(`targets.moe_expert_fp8_gemm_spec`, keyed on `GemmTarget.fp8_moe_dot`):

* with Hopper's step the `_v3` forms are the `_v1` Definitions, and with sm_120's they are its chain (`moe_model` with the step);
* one 32-wide tile the two steps sum differently, which the w13 word shows at BLOCK_SIZE_K 32, the sm_120 launch's;
* `fp8_moe_rows`'s twin with `dot="blackwell"` is the `_v3{DOT=BlackwellE4m3QmmaDot32_v1}` GEMM, special and subnormal words included;
* the pair forms at expert e are the `_v3` GEMMs over that expert's rows and scales, and an id outside [0, E) selects NaN words;
* their Boolean versions (`boolean_fp8_moe`: `_v4` of the `_v3` forms, `_v2` of the pair forms and gathers) agree with their word
  views, routed ids in range and out of it, and the switch maps each word view to them;
* cc 12.0 binds them with `BlackwellE4m3QmmaDot32_v1`, the step blackwell_consumer records from the cc 12.0 fused-MoE capture
  (property blackwell_fp8_capture) in its own field, cc 9.0 with `HopperE4m3QgmmaDot32_v1` (the `_v1` forms' step), and the targets
  that record no e4m3 step there refuse by name.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32 as BW
from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32 as HP
from verity.ml.definitions import boolean as SW
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.numerics import fp8_moe_rows as MR
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import fp8_moe as M
from verity.ml.definitions.registry import fp8_moe_experts as X
from verity_vllm.program.frontend import gemm_targets as GT
from verity_vllm.program.frontend import targets as T
from tests.program.test_boolean_fp8 import _flat, _inputs
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
    assert (X.gather_e4m3(3).evaluate(3, *col8), X.gather_f32(3).evaluate((1 << 32) - 1, *col32)) == (0x7F, 0x7FC00000)
    assert X.gather_e4m3(3).id == "GatherE4m3x3_v1" and X.gather_f32(128).id == "GatherF32x128_v1"


def test_the_gathers_resolve_by_id_in_a_cold_decode() -> None:
    """A descriptor naming `GatherE4m3x{V}_v1` / `GatherF32x{V}_v1` decodes in a process that never built the program (the gathers
    are made per vocabulary while deriving): the lazy families resolve them like `GatherBf16x{V}_v1` (the GLM-5.3 Match of record)."""
    from verity.ml.circuits.codec import lookup_primitive
    from verity.ml.circuits.defs import REGISTRY
    assert lookup_primitive(REGISTRY, "GatherE4m3x256_v1") is X.gather_e4m3(256)
    assert lookup_primitive(REGISTRY, "GatherF32x256_v1") is X.gather_f32(256)


#: on bits: two tiles over two scale blocks (the step sm_120's or Hopper's), the GEMMs over a whole and a partial weight block, the
#: gathers, and the pair forms over 3 experts
BOOLEAN = [bind(BM.MoeExpertFp8BlockCoordinateV4, K=64, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8BlockCoordinateV4, K=64, G=64, KB=32, SWAP=True, DOT=BG.HopperE4m3QgmmaDot32),
           bind(BM.MoeExpertFp8BlockCoordinateWV4, K=64, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8BlockGemmV4, K=32, N=40, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8BlockGemmWV4, K=32, N=40, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8RowV2, E=3, K=4), bind(BM.MoeExpertFp8ScaleRowV2, E=3, K=2),
           bind(BM.MoeExpertFp8CoordinateWV2, E=3, K=64, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8GemmV2, E=3, K=32, N=40, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32),
           bind(BM.MoeExpertFp8GemmWV2, E=3, K=32, N=40, G=32, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32)]


def _routed(word, inputs: list, E: int) -> list:
    """`inputs` with the expert id operand sampled from [0, E) on all but every eighth instance (random ids are out of range)."""
    names = [n for n, _ in word.signature()[0]]
    if "e" not in names:
        return inputs
    i = names.index("e")
    ids = np.asarray(inputs[i]).copy()
    flat = ids.reshape(-1)
    flat[np.arange(flat.size) % 8 != 0] = np.arange(flat.size)[np.arange(flat.size) % 8 != 0] % E
    return inputs[:i] + [ids] + inputs[i + 1:]


@pytest.mark.parametrize("target", BOOLEAN, ids=lambda t: t.id)
def test_on_bits_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target) and word.defn.name == target.defn.name
    assert SW.boolean_version(word).id == target.id
    inputs = _routed(word, _inputs(word, 64, sum(map(ord, target.id))), target.bindings.get("E", 1))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


def test_the_v2_forms_keep_their_word_views() -> None:
    """The switch maps `_v1` to `_v2` and `_v3{DOT}` to `_v4{DOT on bits}`: a DOT form never stands for the Hopper-pinned `_v1`."""
    v1 = bind(M.MoeExpertFp8BlockCoordinate, K=64, G=32, KB=32, SWAP=False)
    v3 = bind(M.MoeExpertFp8BlockCoordinateV3, K=64, G=32, KB=32, SWAP=False, DOT=HP)
    assert SW.boolean_version(v1).id == "MoeExpertFp8BlockCoordinate_v2{K=64,G=32,KB=32,SWAP=False}"
    assert SW.boolean_version(v3).id == "MoeExpertFp8BlockCoordinate_v4{K=64,G=32,KB=32,SWAP=False,DOT=HopperE4m3QgmmaDot32_v2}"
    with pytest.raises(ValueError, match="from 1 to 1024 experts"):
        BM.gather_e4m3(1025)


def _tp(cc):
    return TargetProfile(compute_capability=cc)


def test_blackwell_consumer_records_the_moe_step_with_its_own_measurement() -> None:
    """The step is the fused-MoE kernel's own measurement, not cc 12.0's per-tensor FP8 step (`fp8_dot`): its evidence names the
    capture run and record and the moe cases it rests on."""
    gt = GT.REGISTERED["blackwell_consumer"]
    assert gt.fp8_moe_dot == "BlackwellE4m3QmmaDot32"
    [ev] = [e for e in gt.evidence if e.startswith("the block-FP8 fused-MoE experts")]
    for needle in ("blackwell_fp8_capture", "r20261003-220018-3f6a", "0d346c79c642a0a1", "moe/decode", "moe/small", "moe/block",
                   "moe/prefill", "11,791,360 words", "MoeExpertFp8BlockGemm(W)_v3{G=64,KB=32,SWAP=False,DOT=BlackwellE4m3QmmaDot32_v1}"):
        assert needle in ev, needle
    assert GT.REGISTERED["hopper"].fp8_moe_dot == GT.REGISTERED["hopper"].fp8_block_dot == "HopperE4m3QgmmaDot32"
    assert all(GT.REGISTERED[k].fp8_moe_dot is None for k in GT.REGISTERED if k not in ("blackwell_consumer", "hopper"))


def test_the_moe_step_is_keyed_on_its_own_field(monkeypatch) -> None:
    """Without `fp8_moe_dot`, cc 12.0 refuses by name although its per-tensor and blockwise steps (`fp8_dot`, `fp8_block_dot`) are
    the same step."""
    gt = dataclasses.replace(GT.REGISTERED["blackwell_consumer"], fp8_moe_dot=None)
    monkeypatch.setitem(GT.REGISTERED, "blackwell_consumer", gt)
    assert gt.fp8_dot == gt.fp8_block_dot == "BlackwellE4m3QmmaDot32"
    with pytest.raises(T.UnregisteredTarget, match=r"blackwell_consumer \(12\.0\): no registered e4m3 step \(GemmTarget.fp8_moe_dot\)"):
        T.moe_expert_fp8_gemm_spec(_tp((12, 0)), 128, 4096, 384, 64, 32, False, BM=16, BN=128, WARPS=4, routed=False)


def test_cc12_binds_the_sm120_step_cc9_the_v1_forms_step_and_the_others_refuse() -> None:
    """`_tp` is batch-invariant, so the launch dot is the target-wide `GemmTarget.fp8_moe_dot`, whatever the tile."""
    assert T.moe_expert_fp8_dot(_tp((12, 0))) is BW
    assert T.moe_expert_fp8_gemm_spec(_tp((12, 0)), 128, 4096, 384, 64, 32, False, BM=16, BN=128, WARPS=4, routed=False).id == \
        "MoeExpertFp8Gemm_v1{E=128,K=4096,N=384,G=64,KB=32,SWAP=False,DOT=BlackwellE4m3QmmaDot32_v1}"
    assert T.moe_expert_fp8_gemm_spec(_tp((12, 0)), 128, 192, 4096, 64, 32, False, BM=16, BN=128, WARPS=4, routed=True).id == \
        "MoeExpertFp8GemmW_v1{E=128,K=192,N=4096,G=64,KB=32,SWAP=False,DOT=BlackwellE4m3QmmaDot32_v1}"
    assert T.moe_expert_fp8_dot(_tp((9, 0))) is HP
    assert T.moe_expert_fp8_gemm_spec(_tp((9, 0)), 256, 6144, 512, 128, 128, True, BM=8, BN=32, WARPS=4, routed=False).id == \
        "MoeExpertFp8Gemm_v1{E=256,K=6144,N=512,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}"
    for cc in ((8, 0), (8, 9)):
        with pytest.raises(T.UnregisteredTarget, match="no registered e4m3 step"):
            T.moe_expert_fp8_gemm_spec(_tp(cc), 128, 4096, 384, 64, 32, False, BM=16, BN=128, WARPS=4, routed=False)
    with pytest.raises(T.UnregisteredTarget):
        T.moe_expert_fp8_dot(_tp((10, 0)))
