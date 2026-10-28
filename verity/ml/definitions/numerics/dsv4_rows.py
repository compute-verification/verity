"""The replay row kernels of DeepSeek-V4-Pro's served rows that no other module evaluates, one per Definition family, each the numpy
twin its Definition is held to: `mhc_twin` (the six mHC families), `dsv4_compressor_twin` (the head-512 compressed rows),
`flashmla_twin` / `flashmla_prefill_twin` (the sparse FP8 decode, the prefill and its workspace gather), `dsv4_qnorm_insert_twin`
(the query norm and KV insert), `dsv4_moe_twin` (the two routers and the clamped SwiGLU), `dsa_norm_rows.block_rmsnorm_rows` (the
q / kv norm over a masked block), `dsv4_indexer_twin` (the compressor state's score and the o-projection input's inverse-RoPE FP8
quantisation), and the hash router's `GatherI32x{V}` lookup, which is wiring only.  A row is one token: the
twins' leading token axis is 1, and each Tuple's fields are ports "0", "1", ... in the Definition's return order.  The twins are
checked word for word against their Definitions (tests/program/test_mhc.py, the kernel self-check), and on the H200 by the
captures `EVALUATOR_PROVENANCE` names, which also names the gaps.  A FlashMLA row whose DOT is not the Hopper k16 step the twins
model is declined by name.
The row kernels register from `rows` (kernel "rows").
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.numerics.ir_rows import f64_statics
from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static

#: the vocabularies of the served `GatherI32x{V}_v1` rows (DeepSeek-V4-Pro's hash router reads tid2eid at its vocabulary)
GATHER_VOCABS = (129280,)

_MHC = ("verity.ml.definitions.numerics.mhc_twin.{fn} on one token ({sig}), word for word the Definition's operations in its order "
        "(the TF32 k8 step HopperTf32WgmmaDot8_v1's candidate group sum, B rounded RNE, the squared sum's lo contraction)")

EVALUATORS = {
    "MhcPost_v1": _MHC.format(fn="post", sig="MhcPost_v1{HC,N}: new_residual's FMA chain per stream, RNE to bf16"),
    "MhcFusedPostPre_v1": _MHC.format(fn="fused_post_pre", sig="MhcFusedPostPre_v1{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}: NS "
                                      "MhcFusedSplit_v1s, then MhcPreBigFuse_v1") + " -> residual', layer input, post', comb'",
    "MhcPostPreTf32_v1": _MHC.format(fn="post_pre_tf32", sig="MhcPostPreTf32_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}: MhcPost_v1, "
                                     "MhcPrenormTf32_v1 over tf32_ranges(HC*H, NS), MhcPreBigFuse_v1") + " -> residual', layer input, post', comb'",
    "MhcPreBroadcast_v1": _MHC.format(fn="pre_broadcast", sig="MhcPreBroadcast_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}: MhcPrenormTf32_v1 "
                                      "of the embedding, MhcPreBigFuseBroadcast_v1") + " -> layer input, post, comb",
    "MhcResidualBroadcast_v1": "the embedding's H bf16 words copied to each of the HC streams (MhcResidualBroadcast_v1{HC,H}: no arithmetic)",
    "HcHead_v1": _MHC.format(fn="hc_head", sig="HcHead_v1{HC,H,TH,EPS,HC_EPS}: the threads' sqr / mix chains, halving trees, rsqrt, "
                             "sigmoid + HC_EPS, the streams' weighted sum") + " -> H bf16",
    "Dsv4MlaCompressC4_v1": "verity.ml.definitions.numerics.dsv4_compressor_twin.compress_c4 (Dsv4MlaCompressC4_v1{ROWS,D,QB,EPS}: "
                            "Dsv4MlaC4Column_v1 per column, Dsv4MlaNormRopeQuant_v1{4,D,QB,EPS}) -> NoPE e4m3 bytes, RoPE bf16, e8m0 scales",
    "Dsv4MlaCompressC128_v1": "verity.ml.definitions.numerics.dsv4_compressor_twin.compress_c128 (Dsv4MlaCompressC128_v1{W,R,D,QB,EPS}: "
                              "Dsv4MlaC128Column_v1 per column, Dsv4MlaNormRopeQuant_v1{128,D,QB,EPS}) -> NoPE e4m3 bytes, RoPE bf16, e8m0 scales",
    **{f"GatherI32x{v}_v1": f"column[token] of the {v} committed i32 words, 0xFFFFFFFF for a token outside [0, {v}) "
                            f"(GatherI32x{v}_v1, registry.dsv4_decode_indices.gather_i32: a lookup, no arithmetic)" for v in GATHER_VOCABS},
    "FlashMlaDecode_v1": "verity.ml.definitions.numerics.flashmla_twin.decode (FlashMlaDecode_v1{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}: "
                         "each row's FlashMlaRowValid_v1 and FlashMlaKvRow_v1, then per head FlashMlaSplit_v1 over each split's 64-row "
                         "blocks and FlashMlaCombine_v1 with the sink; DOT the Hopper k16 step) -> NH x (DN + DR) bf16",
    "FlashMlaPrefill_v1": "verity.ml.definitions.numerics.flashmla_prefill_twin.prefill (FlashMlaPrefill_v1{NH,NP,D,BN,DOT,SCALE}: each "
                          "row's FlashMlaPrefillValid_v1 and masking, then per head FlashMlaPrefillHead_v1's block pairs and the sink; "
                          "DOT the Hopper k16 step) -> NH x D bf16",
    "FlashMlaGatherRow_v1": "verity.ml.definitions.numerics.flashmla_prefill_twin.gather_rows (FlashMlaGatherRow_v1{DN,DR,QT}: "
                            "Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])), E8m0ShlBf16(scale[d / QT])), then the RoPE words) -> DN + DR bf16",
    "Dsv4QNormRopeKvInsert_v1": "verity.ml.definitions.numerics.dsv4_qnorm_insert_twin.insert (Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,"
                                "EPS}: Dsv4QNormRope_v1 per live head, the padded heads +0, Dsv4KvRopeQuant_v1 of the kv row) -> q_out "
                                "NHP x (DN + DR) bf16, the NoPE e4m3 bytes, the scale bytes and pad, the RoPE bf16",
    "TritonBlockRMSNorm_v1": "verity.ml.definitions.numerics.dsa_norm_rows.block_rmsnorm_rows (TritonBlockRMSNorm_v1{N,BLOCK,WARPS,EPS}: per "
                             "thread of 32 WARPS an FMA chain over its dsa_norm.layout registers of the BLOCK, the elements from N +0, "
                             "each warp's xor-butterfly, the warp sums' halving tree, r = RsqrtApprox(fma(DivFullScaleA(sum, N), "
                             "DivFullRcp(N), eps)), y = F2fpBf16((r * x) * w)) -> N bf16",
    "MoeRouterSqrtSoftplus_v1": "verity.ml.definitions.numerics.dsv4_moe_twin.router_twin (MoeRouterSqrtSoftplus_v1{E,SCALE}: "
                                "Dsv4SqrtSoftplus_v1 per expert, current = score + bias (a NaN -1e30), six slots of the first maximum "
                                "of current, the lane-0 sum, w * div.full(SCALE, sum > 0 ? sum : 1)) -> six f32 weights, six i32 ids",
    "MoeRouterHash_v1": "verity.ml.definitions.numerics.dsv4_moe_twin.hash_twin (MoeRouterHash_v1{E,SCALE}: Dsv4HashWeight_v1 of "
                        "logits[ids[k]] (an id outside [0, E) reads logits[0], MoeRouterPick_v1's select chain), the lane-0 sum, "
                        "w * SCALE / (sum > 0 ? sum : 1), FMUL then div.rn) -> six f32 weights",
    "SiluMulClamp_v1": "verity.ml.definitions.numerics.dsv4_moe_twin.clamp_twin (SiluMulClamp_v1{I,LIMIT}: SiluMulClampBf16_v1 per "
                       "column, bf16(fminf(gate, LIMIT)) and bf16(fmaxf(fminf(up, LIMIT), -LIMIT)), then SiluMulBf16_v2) -> I bf16",
    "Dsv4CompressorScore_v1": "verity.ml.definitions.numerics.dsv4_indexer_twin.score_twin (Dsv4CompressorScore_v1{W}: F32AddV2_v1 of "
                              "score[i] and ape[i], a NaN sum 0x7FFFFFFF) -> W f32",
    "Dsv4InvRopeQuantHead_v1": "verity.ml.definitions.numerics.dsv4_indexer_twin.heads_twin (Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}: "
                               "the Definition's body on numpy words, the first CHUNKS*QG - ROPE words Bf16ToF32_v1, the last ROPE "
                               "Dsv4InvRope_v1's FMA rotation, then per group of QG Dsv4InvRopeQuant_v1's halving-tree amax, ue8m0 "
                               "scale ex2, div.full and min.xorsign.abs to e4m3) -> CHUNKS*QG e4m3 bytes, CHUNKS f32 scales",
}

_MHC_IDS = EVIDENCE["dsv4_mhc_capture"]
_MHC_PROVENANCE = {
    "property": "dsv4_mhc_capture (verity-vllm/dsv4-mhc-capture/v1), tests/properties/dsv4_mhc_capture_gpu.py",
    "run": _MHC_IDS["run"],
    "art": _MHC_IDS["art"],
    "step_fit": f"{EVIDENCE['hopper_tf32_k8_step_probe']['run']} (the probe that fitted HopperTf32WgmmaDot8_v1's k8 step alone)",
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80, DeepGEMM on (VLLM_USE_DEEP_GEMM=1, vLLM's default)",
    "captured_utc": _MHC_IDS["captured_utc"],
    "cases": "the fused post + pre at 1, 2, 7, 8, 9, 15 and 16 tokens; the TF32 post + pre at 17, 64, 65, 128 and 129 (MhcPost_v1 "
             "among its pieces); the layer-0 broadcast at 1, 16, 64 and 65 with every residual_out copy; hc_head at 1, 7, 16, 64 and 129",
    "tests": ["tests/program/test_mhc.py (every mHC word Definition against mhc_twin, word for word)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/mhc.py (the reference; the twin is checked against it, never the other way round)",
}

_COMPRESSOR_IDS = EVIDENCE["dsv4_compressor_capture"]
_COMPRESSOR_PROVENANCE = {
    "property": "dsv4_compressor_capture (verity-vllm/dsv4-compressor-capture/v1), tests/properties/dsv4_compressor_capture_gpu.py",
    "run": _COMPRESSOR_IDS["run"],
    "art": _COMPRESSOR_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80, nvidia-cutlass-dsl 4.6.2",
    "captured_utc": _COMPRESSOR_IDS["captured_utc"],
    "cases": "a ratio-4 layer (state width 1024, the 8-row window) and a ratio-128 layer (W 8, R 16) into fp8_ds_mla caches: every "
             "byte of every compressed cache row",
    "tests": ["tests/program/test_kernel_self_check.py (dsv4-compressor-twin and each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/dsv4_compressor.py (the reference; the twin is checked against it, never the other way round)",
}

_FLASHMLA_IDS = EVIDENCE["dsv4_flashmla_capture"]
_FLASHMLA_PROVENANCE = {
    "property": "dsv4_flashmla_capture (verity-vllm/dsv4-flashmla-capture/v1), tests/properties/dsv4_flashmla_capture_gpu.py",
    "run": _FLASHMLA_IDS["run"],
    "art": _FLASHMLA_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80 vendoring FlashMLA 6bc49418",
    "captured_utc": _FLASHMLA_IDS["captured_utc"],
    "cases": "decode swa, c4, c128, lone, batch and edge at the token's SPLITS; every gathered workspace row; the prefill cases at "
             "the token's NP (16 heads padded to 64, the 128-token window, the sink)",
    "tests": ["tests/program/test_flashmla.py, tests/program/test_flashmla_prefill.py (the twins against registry.flashmla and "
              "registry.flashmla_prefill)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/flashmla.py, flashmla_prefill.py (the reference; the twins are checked against them, "
                   "never the other way round)",
}

_INSERT_IDS = EVIDENCE["dsv4_qnorm_insert_capture"]
_INSERT_PROVENANCE = {
    "property": "dsv4_qnorm_insert_capture (verity-vllm/dsv4-qnorm-insert-capture/v1), tests/properties/dsv4_qnorm_insert_capture_gpu.py",
    "run": _INSERT_IDS["run"],
    "art": _INSERT_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80",
    "captured_utc": _INSERT_IDS["captured_utc"],
    "cases": "every word of the q the fused q-norm / RoPE / KV-quant insert returns and every byte it writes to the fp8_ds_mla cache "
             "at 16 query heads of 512 padded to 64",
    "tests": ["tests/program/test_dsv4_qnorm_insert.py (the twin against registry.dsv4_qnorm_insert)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/dsv4_qnorm_insert.py (the reference; the twin is checked against it, never the other "
                   "way round)",
}

_MOE_IDS = EVIDENCE["dsv4_moe_capture"]
_MOE_PROVENANCE = {
    "property": "dsv4_moe_capture (verity-vllm/dsv4-moe-capture/v1), tests/properties/dsv4_moe_capture_gpu.py",
    "run": _MOE_IDS["run"],
    "run_record": _MOE_IDS["run_record"],
    "art": None,
    "gap": "the capture ran with VLLM_BATCH_INVARIANT=1, where vLLM's default is 0, so the served row's publish job names "
           "dsv4_moe_capture a gap and cites no art of it; there is none at the defaults yet",
    "device": f"cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80, the harness at {_MOE_IDS['harness']}",
    "captured_utc": _MOE_IDS["captured_utc"],
    "cases": "8,192 _dsv4_topk_kernel rows and 8,192 dsv4HashTopkSoftplusSqrt rows at E=384, SCALE=2.5 drawn to the routers' edges; "
             "act_and_mul_kernel with HAS_CLAMP at LIMIT=10, vectorised at I=384 (7,510 rows) and scalar at I=9 (1,024); each "
             "launch run twice, bit-exact against the twins, and the Programs evaluated on every row (--check)",
    "tests": ["tests/properties/test_dsv4_moe_capture.py (the twins and their element functions against registry.dsv4_moe's Programs)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/dsv4_moe.py (the reference; the twins are checked against it, never the other way "
                   "round)",
}

_Q_NORM_IDS = EVIDENCE["dsv4_q_norm_commit"]
_BLOCK_NORM_PROVENANCE = {
    "property": None,
    "run": _Q_NORM_IDS["run"],
    "art": None,
    "device": "cc 9.0, the eight devices of the row's attempt-3 Commit (the run above), vLLM d9105ea80, Triton 3.7.1",
    "captured_utc": _Q_NORM_IDS["captured_utc"],
    "cases": "the q norm (N 1536 of a 2048 block on 8 warps, eps 1e-6): 1,230 committed rows (308 distinct) on 8 ranks, read by "
             f"{_Q_NORM_IDS['rows']} (out/rows.npz) from the Commit's bundles and the checkpoint's q_norm weights, every committed "
             "output the Definition's; eps + DivFullScaleA(sum, N) * DivFullRcp(N) in two roundings misses 5 of the 308 by 1-4 "
             "bf16 words, which tests/properties/fixtures/dsv4_q_norm_commit_20261021.json holds with 4 controls",
    "sass": f"{_Q_NORM_IDS['sass']}: the served kernel's cc 9.0 SASS (Triton 3.7.1, 8 warps at BLOCK 2048) is MUFU.RCP of the "
            "scaled SIZE, FFMA rcp, scaled sum, eps, then MUFU.RSQ",
    "gap": "no probe launch of _fused_q_kv_rmsnorm_kernel: the reduction tree is the PTX's, pinned on the device only by committed "
           "rows; the kv norm (N 512, a power of two, so either epilogue) only by the re-judge's sampled kv rows (306 a rank, all "
           f"equal, {_Q_NORM_IDS['kv_rows'][0]} and {_Q_NORM_IDS['kv_rows'][1]})",
    "model": "tests/program/test_dsa_norm.py::test_block_definition_matches_the_ptx_model (the Definition against a model of that "
             "PTX and SASS at the served (N, BLOCK, WARPS) = (1536, 2048, 8) and (512, 2048, 8), and four other blocks)",
    "related": f"glm53_fused_rmsnorm_capture ({EVIDENCE['glm53_fused_rmsnorm_capture']['art']}): the same "
               "tl.sum lowering at the unmasked four-warp block (DsaRMSNorm_v1, TritonBlockRMSNorm_v1{N, N, 4, EPS}) in GLM-5.3's "
               "_fused_norm_rope_kernel",
    "tests": ["tests/properties/test_dsv4_q_norm_commit.py (the Definition and the row kernel against the committed rows)",
              "tests/program/test_dsa_norm.py (the Definition against the PTX model)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity/ml/definitions/registry/dsa_norm.py (the reference; the twin is checked against it, never the other way round)",
}

_INDEXER_TESTS = ["tests/properties/test_dsv4_indexer_capture.py (NumpyOps' steps against dsv4_indexer.WORD's primitives on special "
                  "and random words; score_twin and heads_twin against their Definitions at the served statics)",
                  "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
                  "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"]
_INDEXER_DEFINITIONS = ("verity/ml/definitions/registry/dsv4_indexer.py (the reference; the twins run its bodies and are checked against "
                        "it, never the other way round)")

_INDEXER_IDS = EVIDENCE["dsv4_indexer_capture"]
_INV_ROPE_PROVENANCE = {
    "property": "dsv4_indexer_capture (verity-vllm/dsv4-indexer-capture/v1), tests/properties/dsv4_indexer_capture_gpu.py",
    "run": _INDEXER_IDS["run"],
    "art": _INDEXER_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80",
    "captured_utc": _INDEXER_IDS["captured_utc"],
    "cases": "inv_rope/served: fused_inv_rope_fp8_quant as deep_gemm_fp8_o_proj calls it, twice, on 16 heads of 512 of a 64-head "
             "FlashMLA output over 510 tokens with special heads and rotary rows: each of the 8,160 (token, head)s' 512 e4m3 bytes "
             "and 4 f32 scales against heads_twin and Dsv4InvRopeQuantHead_v1{4,128,64}, 0 mismatches",
    "tests": _INDEXER_TESTS,
    "definitions": _INDEXER_DEFINITIONS,
}

_SCORE_PROVENANCE = {
    "property": "dsv4_compressor_capture (verity-vllm/dsv4-compressor-capture/v1), tests/properties/dsv4_compressor_capture_gpu.py",
    "run": _COMPRESSOR_IDS["run"],
    "art": _COMPRESSOR_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80",
    "captured_utc": _COMPRESSOR_IDS["captured_utc"],
    "cases": "score/c4 (W 1024, the ratio-4 attention state, the served row's width; 2,308 rows) and score/c128 (W 512; 6,146 "
             "rows): each state row save_partial_states writes, its score half against Dsv4CompressorScore_v1{W} of its score row and "
             "ape[pos % ratio], with zero, infinite and NaN scores, 0 mismatches",
    "also": {"property": "dsv4_indexer_capture", "run": _INDEXER_IDS["run"],
             "art": _INDEXER_IDS["art"],
             "cases": "score/served (W 256, the indexer's state; 4,868 rows): each score half against score_twin and the Definition, "
                      "0 mismatches"},
    "tests": _INDEXER_TESTS,
    "definitions": _INDEXER_DEFINITIONS,
}

#: family -> what pinned its twin to the H200, or the gap where nothing has yet (a family absent here has no hardware-specific
#: semantics)
PROVENANCE = {
    **{f: _MHC_PROVENANCE for f in ("MhcPost_v1", "MhcFusedPostPre_v1", "MhcPostPreTf32_v1", "MhcPreBroadcast_v1",
                                     "MhcResidualBroadcast_v1", "HcHead_v1")},
    "Dsv4MlaCompressC4_v1": _COMPRESSOR_PROVENANCE, "Dsv4MlaCompressC128_v1": _COMPRESSOR_PROVENANCE,
    **{f: _FLASHMLA_PROVENANCE for f in ("FlashMlaDecode_v1", "FlashMlaPrefill_v1", "FlashMlaGatherRow_v1")},
    "Dsv4QNormRopeKvInsert_v1": _INSERT_PROVENANCE,
    **{f: _MOE_PROVENANCE for f in ("MoeRouterSqrtSoftplus_v1", "MoeRouterHash_v1", "SiluMulClamp_v1")},
    "TritonBlockRMSNorm_v1": _BLOCK_NORM_PROVENANCE,
    "Dsv4CompressorScore_v1": _SCORE_PROVENANCE,
    "Dsv4InvRopeQuantHead_v1": _INV_ROPE_PROVENANCE,
}


def _words(a: np.ndarray, n: int, dtype) -> np.ndarray:
    a = np.asarray(a)
    if a.size != n:
        raise Declined(f"an operand of {a.size} words, the signature has {n}")
    return a.astype(dtype, copy=False).reshape(n)


def _bf16(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint16).reshape(shape)


def _f32(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint32).reshape(shape)


def _ports(*arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {str(j): np.asarray(a).reshape(-1) for j, a in enumerate(arrays)}


def _mix(HC: int) -> int:
    return HC * (2 + HC)


def _fuse(S: dict[str, Any]) -> dict[str, Any]:
    return {k: (float(S[k]) if k in ("EPS", "HC_EPS") else int(S[k])) for k in ("NS", "TN", "REPEAT", "EPS", "HC_EPS")}


def mhc_post(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPost_v1{HC,N}(x, residual, post, comb)` -> HC x N bf16 words."""
    from verity.ml.definitions.numerics import mhc_twin as W
    HC, N = int(statics["HC"]), int(statics["N"])
    out = W.post(_bf16(ops[0], 1, N), _bf16(ops[1], 1, HC, N), _f32(ops[2], 1, HC), _f32(ops[3], 1, HC, HC))
    return _ports(out.astype(np.uint16))


def _layer_operands(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> tuple:
    HC, H = int(statics["HC"]), int(statics["H"])
    OUT = _mix(HC)
    return (_bf16(ops[0], 1, H), _bf16(ops[1], 1, HC, H), _f32(ops[2], 1, HC), _f32(ops[3], 1, HC, HC), _f32(ops[4], OUT, HC * H),
            _f32(ops[5], 3), _f32(ops[6], OUT), _bf16(ops[7], H))


def _layer_ports(res, out, post, comb) -> dict[str, np.ndarray]:
    return _ports(res.astype(np.uint16), out.astype(np.uint16), post.astype(np.uint32), comb.astype(np.uint32))


def mhc_fused_post_pre(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcFusedPostPre_v1{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}(x, residual, post, comb, fn, scale, base, w)` -> ports 0 (residual', HC x H
    bf16), 1 (the layer input, H bf16), 2 (post', HC f32) and 3 (comb', HC x HC f32)."""
    from verity.ml.definitions.numerics import mhc_twin as W
    S = f64_statics(statics)
    f = _fuse(S)
    return _layer_ports(*W.fused_post_pre(*_layer_operands(S, ops), NS=f["NS"], TH=int(S["TH"]), TN=f["TN"], REPEAT=f["REPEAT"],
                                          EPS=f["EPS"], HC_EPS=f["HC_EPS"]))


def mhc_post_pre_tf32(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPostPreTf32_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}(x, residual, post, comb, fn, scale, base, w)` -> MhcFusedPostPre_v1's ports."""
    from verity.ml.definitions.numerics import mhc_twin as W
    S = f64_statics(statics)
    return _layer_ports(*W.post_pre_tf32(*_layer_operands(S, ops), **_fuse(S)))


def mhc_pre_broadcast(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPreBroadcast_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}(x, fn, scale, base, w)` -> ports 0 (the layer input, H bf16), 1 (post, HC
    f32) and 2 (comb, HC x HC f32)."""
    from verity.ml.definitions.numerics import mhc_twin as W
    S = f64_statics(statics)
    HC, H = int(S["HC"]), int(S["H"])
    out, post, comb = W.pre_broadcast(_bf16(ops[0], 1, H), _f32(ops[1], _mix(HC), H), _f32(ops[2], 3), _f32(ops[3], _mix(HC)),
                                      _bf16(ops[4], H), **_fuse(S))
    return _ports(out.astype(np.uint16), post.astype(np.uint32), comb.astype(np.uint32))


def mhc_residual_broadcast(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcResidualBroadcast_v1{HC,H}(x)` -> HC x H bf16 words, each stream the embedding's."""
    HC, H = int(statics["HC"]), int(statics["H"])
    return _ports(np.tile(_bf16(ops[0], H), HC))


def hc_head(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`HcHead_v1{HC,H,TH,EPS,HC_EPS}(residual, fn, scale, base)` -> H bf16 words."""
    from verity.ml.definitions.numerics import mhc_twin as W
    S = f64_statics(statics)
    HC, H = int(S["HC"]), int(S["H"])
    out = W.hc_head(_bf16(ops[0], 1, HC, H), _f32(ops[1], HC, HC * H), _f32(ops[2], 1), _f32(ops[3], HC), int(S["TH"]), float(S["EPS"]),
                    float(S["HC_EPS"]))
    return _ports(out.astype(np.uint16))


def _compress(rows: Callable[[dict[str, Any]], int], run: Callable[..., tuple], statics: dict[str, Any],
              ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    S = f64_statics(statics)
    D, QB, R = int(S["D"]), int(S["QB"]), rows(S)
    q, rope, e = run(_f32(ops[0], R, D), _f32(ops[1], R, D), _bf16(ops[2], D), _f32(ops[3], QB), S)
    return _ports(np.asarray(q).astype(np.uint8), np.asarray(rope).astype(np.uint16), np.asarray(e).astype(np.uint8))


def dsv4_mla_compress_c4(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4MlaCompressC4_v1{ROWS,D,QB,EPS}(kv, score, w, cs)` -> ports 0 (D - QB e4m3 bytes), 1 (QB bf16) and 2 (D / QB e8m0 bytes)."""
    from verity.ml.definitions.numerics import dsv4_compressor_twin as C
    return _compress(lambda S: int(S["ROWS"]), lambda kv, sc, w, cs, S: C.compress_c4(kv, sc, w, cs, int(S["QB"]), float(S["EPS"])),
                     statics, ops)


def dsv4_mla_compress_c128(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4MlaCompressC128_v1{W,R,D,QB,EPS}(kv, score, w, cs)` -> Dsv4MlaCompressC4_v1's ports."""
    from verity.ml.definitions.numerics import dsv4_compressor_twin as C
    return _compress(lambda S: int(S["W"]) * int(S["R"]),
                     lambda kv, sc, w, cs, S: C.compress_c128(kv, sc, w, cs, int(S["W"]), int(S["R"]), int(S["QB"]), float(S["EPS"])),
                     statics, ops)


def gather_i32(V: int, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GatherI32x{V}_v1(token, column)` -> column[token] (one i32 word), 0xFFFFFFFF for a token outside [0, V)."""
    t = int(_words(ops[0], 1, np.int64)[0]) & 0xFFFFFFFF
    t = t - (1 << 32) if t & 0x80000000 else t
    col = _words(ops[1], V, np.uint32)
    return {"0": np.array([col[t] if 0 <= t < V else 0xFFFFFFFF], dtype=np.uint32)}


def _i32(a: np.ndarray, n: int) -> np.ndarray:
    """n committed i32 words, whether they arrive as int32 or as their u32 bits, as u32 bits."""
    return (_words(a, n, np.int64) & 0xFFFFFFFF).astype(np.uint32)


def _hopper_dot(family: str, statics: dict[str, Any]) -> None:
    from verity.ml.definitions.registry.hopper import HopperBF16WgmmaDot16
    dot = prim_static(statics, "DOT", family)
    if dot != HopperBF16WgmmaDot16.id:
        raise Declined(f"{family}: no twin for DOT={dot} (the twins model {HopperBF16WgmmaDot16.id})")


def _splits(v: Any) -> tuple[int, ...]:
    """SPLITS as the spec text spells it (`[0,1,2,3,4]`) or as a binding holds it."""
    return tuple(int(s) for s in (json.loads(v) if isinstance(v, str) else v))


def flashmla_decode(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaDecode_v1{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}(q, nope, scale, rope, idx, swa_len[, extra_len], sink)` -> NH x (DN +
    DR) bf16 words, head-major."""
    from verity.ml.definitions.numerics import flashmla_twin as FT
    f = "FlashMlaDecode_v1"
    _hopper_dot(f, statics)
    S = f64_statics(statics)
    NH, NW, DN, DR, QT, BN = (int(S[k]) for k in ("NH", "NW", "DN", "DR", "QT", "BN"))
    sp = _splits(S["SPLITS"])
    N = sp[-1] * BN
    lens = 2 if N > NW else 1
    if len(ops) != 6 + lens:
        raise Declined(f"{f}: the row wires {len(ops)} args, the Definition takes {6 + lens}")
    out = FT.decode(_bf16(ops[0], NH, DN + DR), _words(ops[1], N * DN, np.uint8).reshape(N, DN),
                    _words(ops[2], N * (DN // QT), np.uint8).reshape(N, DN // QT), _bf16(ops[3], N, DR), _i32(ops[4], N),
                    [_i32(ops[5 + j], 1) for j in range(lens)], _f32(ops[5 + lens], NH), SPLITS=sp, NW=NW, QT=QT, BN=BN,
                    SCALE=float(S["SCALE"]))
    return _ports(np.asarray(out).astype(np.uint16))


def flashmla_prefill(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaPrefill_v1{NH,NP,D,BN,DOT,SCALE}(q, kv, idx, len, skv, sink)` -> NH x D bf16 words, head-major."""
    from verity.ml.definitions.numerics import flashmla_prefill_twin as FP
    _hopper_dot("FlashMlaPrefill_v1", statics)
    S = f64_statics(statics)
    NH, NP, D, BN = (int(S[k]) for k in ("NH", "NP", "D", "BN"))
    N = 2 * NP * BN
    out = FP.prefill(_bf16(ops[0], NH, D), _bf16(ops[1], N, D), _i32(ops[2], N), _i32(ops[3], 1), _i32(ops[4], 1), _f32(ops[5], NH),
                     BN=BN, SCALE=float(S["SCALE"]))
    return _ports(np.asarray(out).astype(np.uint16))


def flashmla_gather_row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaGatherRow_v1{DN,DR,QT}(nope, scale, rope)` -> DN + DR bf16 words."""
    from verity.ml.definitions.numerics import flashmla_prefill_twin as FP
    DN, DR, QT = (int(statics[k]) for k in ("DN", "DR", "QT"))
    out = FP.gather_rows(_words(ops[0], DN, np.uint8).reshape(1, DN), _words(ops[1], DN // QT, np.uint8).reshape(1, DN // QT),
                         _bf16(ops[2], 1, DR), QT)
    return _ports(np.asarray(out).astype(np.uint16))


def dsv4_qnorm_rope_kv_insert(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}(q, kv, cs)` -> ports 0 (q_out, NHP x (DN + DR) bf16), 1 (DN e4m3 bytes), 2 (DN /
    QB + 1 scale bytes, the pad last) and 3 (DR bf16)."""
    from verity.ml.definitions.numerics import dsv4_qnorm_insert_twin as QI
    S = f64_statics(statics)
    NH, NHP, DN, DR, QB = (int(S[k]) for k in ("NH", "NHP", "DN", "DR", "QB"))
    q, nope, scale, rope = QI.insert(_bf16(ops[0], NH, DN + DR), _bf16(ops[1], DN + DR), _f32(ops[2], DR), NHP=NHP, DR=DR, QB=QB,
                                     EPS=float(S["EPS"]))
    return _ports(np.asarray(q).astype(np.uint16), np.asarray(nope).astype(np.uint8), np.asarray(scale).astype(np.uint8),
                  np.asarray(rope).astype(np.uint16))


def triton_block_rmsnorm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`TritonBlockRMSNorm_v1{N,BLOCK,WARPS,EPS}(x, w)` -> N bf16 words."""
    from verity.ml.definitions.numerics.dsa_norm_rows import block_rmsnorm_rows
    S = f64_statics(statics)
    N = int(S["N"])
    r = block_rmsnorm_rows(_bf16(ops[0], 1, N), _bf16(ops[1], N), float(S["EPS"]), int(S["BLOCK"]), int(S["WARPS"]))
    return _ports(r["out"].astype(np.uint16))


def moe_router_sqrt_softplus(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeRouterSqrtSoftplus_v1{E,SCALE}(logits, bias)` -> ports 0 (the six f32 weights) and 1 (the six i32 ids)."""
    from verity.ml.definitions.numerics import dsv4_moe_twin as T
    S = f64_statics(statics)
    E = int(S["E"])
    w, ids = T.router_twin(_f32(ops[0], 1, E), _f32(ops[1], E), float(S["SCALE"]))
    return _ports(w.astype(np.uint32), (ids.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32))


def moe_router_hash(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeRouterHash_v1{E,SCALE}(logits, ids)` -> the six f32 weights.  An id outside [0, E) reads logits[0], as
    `MoeRouterPick_v1`'s select chain does."""
    from verity.ml.definitions.numerics import dsv4_moe_twin as T
    S = f64_statics(statics)
    E = int(S["E"])
    ids = _i32(ops[1], T.TOPK).view(np.int32).astype(np.int64)
    ids = np.where((ids >= 0) & (ids < E), ids, 0).reshape(1, T.TOPK)
    return _ports(T.hash_twin(_f32(ops[0], 1, E), ids, float(S["SCALE"])).astype(np.uint32))


def silu_mul_clamp(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`SiluMulClamp_v1{I,LIMIT}(gu)` -> I bf16 words."""
    from verity.ml.definitions.numerics import dsv4_moe_twin as T
    S = f64_statics(statics)
    return _ports(T.clamp_twin(_bf16(ops[0], 1, 2 * int(S["I"])), float(S["LIMIT"])).astype(np.uint16))


def dsv4_compressor_score(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4CompressorScore_v1{W}(score, ape)` -> W f32 words."""
    from verity.ml.definitions.numerics import dsv4_indexer_twin as IT
    W = int(statics["W"])
    return _ports(IT.score_twin(_f32(ops[0], 1, W), _f32(ops[1], 1, W)))


def dsv4_inv_rope_quant_head(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4InvRopeQuantHead_v1{CHUNKS,QG,ROPE}(x, cs)` -> ports 0 (CHUNKS*QG e4m3 bytes) and 1 (CHUNKS f32 scales)."""
    from verity.ml.definitions.numerics import dsv4_indexer_twin as IT
    CHUNKS, QG, ROPE = (int(statics[k]) for k in ("CHUNKS", "QG", "ROPE"))
    q, s = IT.heads_twin(_bf16(ops[0], 1, CHUNKS * QG), _f32(ops[1], 1, ROPE), QG, ROPE)
    return _ports(q, s)


#: family -> (row kernel, the ports it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "MhcPost_v1": (mhc_post, ("0",)),
    "MhcFusedPostPre_v1": (mhc_fused_post_pre, ("0", "1", "2", "3")),
    "MhcPostPreTf32_v1": (mhc_post_pre_tf32, ("0", "1", "2", "3")),
    "MhcPreBroadcast_v1": (mhc_pre_broadcast, ("0", "1", "2")),
    "MhcResidualBroadcast_v1": (mhc_residual_broadcast, ("0",)),
    "HcHead_v1": (hc_head, ("0",)),
    "Dsv4MlaCompressC4_v1": (dsv4_mla_compress_c4, ("0", "1", "2")),
    "Dsv4MlaCompressC128_v1": (dsv4_mla_compress_c128, ("0", "1", "2")),
    **{f"GatherI32x{v}_v1": (lambda statics, ops, _v=v: gather_i32(_v, statics, ops), ("0",)) for v in GATHER_VOCABS},
    "FlashMlaDecode_v1": (flashmla_decode, ("0",)),
    "FlashMlaPrefill_v1": (flashmla_prefill, ("0",)),
    "FlashMlaGatherRow_v1": (flashmla_gather_row, ("0",)),
    "Dsv4QNormRopeKvInsert_v1": (dsv4_qnorm_rope_kv_insert, ("0", "1", "2", "3")),
    "TritonBlockRMSNorm_v1": (triton_block_rmsnorm, ("0",)),
    "MoeRouterSqrtSoftplus_v1": (moe_router_sqrt_softplus, ("0", "1")),
    "MoeRouterHash_v1": (moe_router_hash, ("0",)),
    "SiluMulClamp_v1": (silu_mul_clamp, ("0",)),
    "Dsv4CompressorScore_v1": (dsv4_compressor_score, ("0",)),
    "Dsv4InvRopeQuantHead_v1": (dsv4_inv_rope_quant_head, ("0", "1")),
}


def _sample_compress(family: str) -> Callable[..., list[np.ndarray]]:
    from verity.ml.definitions.numerics import dsv4_compressor_twin as C
    return C.SAMPLES[family]


def _sample_gather(V: int) -> Callable[..., list[np.ndarray]]:
    """Tokens in [0, V) but for one in eight (negative, V and past it), and random column words."""
    def sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
        tok = rng.integers(0, V, n)
        out = rng.random(n) < 1 / 8
        tok = np.where(out, rng.choice(np.array([-1, -(1 << 31), V, V + 1, (1 << 31) - 1]), n), tok)
        return [tok.astype(np.int64).astype(np.uint32), rng.integers(0, 1 << 32, (n, V), dtype=np.uint64).astype(np.uint32)]
    return sample


def _sample_flashmla(family: str) -> Callable[..., list[np.ndarray]]:
    from verity.ml.definitions.numerics import flashmla_prefill_twin as FP
    from verity.ml.definitions.numerics import flashmla_twin as FT
    return {"FlashMlaDecode_v1": FT._sample_decode, "FlashMlaPrefill_v1": FP._sample_prefill,
            "FlashMlaGatherRow_v1": FP._sample_gather}[family]


def _sample_insert() -> Callable[..., list[np.ndarray]]:
    from verity.ml.definitions.numerics import dsv4_qnorm_insert_twin as QI
    return QI._sample_insert


#: f32 words where the routers branch: +-0, a subnormal, the least normal, 20 and its neighbours (the score's select), the ex2
#: halving at |x| log2 e = 126, the largest finite, +-inf and a NaN
_ROUTER_EDGES = np.array([0, 0x80000000, 1, 0x00800000, 0x419FFFFF, 0x41A00000, 0x41A00001, 0x42AEAC50, 0xC2AEAC50, 0x7F7FFFFF,
                          0x7F800000, 0xFF800000, 0x7FC00000], dtype=np.uint32)


def _f32_words(rng: np.random.Generator, shape: tuple[int, ...], scale: float, edges: float) -> np.ndarray:
    w = (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)
    return np.where(rng.random(shape) < edges, rng.choice(_ROUTER_EDGES, shape), w).astype(np.uint32)


def _sample_router(rng: np.random.Generator, n: int, params: Sequence[Any], *, E: int, SCALE: float) -> list[np.ndarray]:
    """Logits at a few edges; biases normal, or (one row in four) from three values, so currents tie."""
    ties = np.asarray(rng.choice([0.0, 0.25, -0.5], (n, E)), dtype=np.float32).view(np.uint32)
    b = np.where(rng.random((n, 1)) < 0.25, ties, _f32_words(rng, (n, E), 1.0, 0.03))
    return [_f32_words(rng, (n, E), 3.0, 0.1), b.astype(np.uint32)]


def _sample_hash(rng: np.random.Generator, n: int, params: Sequence[Any], *, E: int, SCALE: float) -> list[np.ndarray]:
    """Logits at a few edges and six ids in [0, E) (repeats too) but for one in ten outside it."""
    ids = np.where(rng.random((n, 6)) < 0.1, rng.choice(np.array([-1, E, E + 1, -(1 << 31)]), (n, 6)), rng.integers(0, E, (n, 6)))
    return [_f32_words(rng, (n, E), 4.0, 0.1), ids.astype(np.int64).astype(np.uint32)]


def _sample_clamp(rng: np.random.Generator, n: int, params: Sequence[Any], *, I: int, LIMIT: float) -> list[np.ndarray]:
    """bf16 gate / up words around +-LIMIT (and an ulp either side), +-0, a subnormal and beyond -88, where silu's expf overflows."""
    from verity.ml.definitions.registry.dsv4_moe import limit_bits
    lim = limit_bits(LIMIT) >> 16
    edges = np.array([lim, lim | 0x8000, lim + 1, lim - 1, (lim + 1) | 0x8000, (lim - 1) | 0x8000, 0, 0x8000, 1, 0xC2C8], np.uint16)
    gu = ((rng.standard_normal((n, 2 * I)) * 2 * LIMIT).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return [np.where(rng.random(gu.shape) < 0.2, rng.choice(edges, gu.shape), gu).astype(np.uint16)]


def _sample_score(rng: np.random.Generator, n: int, params: Sequence[Any], *, W: int) -> list[np.ndarray]:
    """Score and ape words, one in ten at `_ROUTER_EDGES` (so sums overflow, cancel to +-0, and meet infinities and NaNs)."""
    return [_f32_words(rng, (n, W), 4.0, 0.1), _f32_words(rng, (n, W), 0.5, 0.1)]


def _sample_head(rng: np.random.Generator, n: int, params: Sequence[Any], *, CHUNKS: int, QG: int, ROPE: int) -> list[np.ndarray]:
    """Heads at a scale per head (bf16 subnormals, 2^-20, 1, 3, 2^100), one in eight zero and one in eight signed powers of two
    times 448 (the clamp's edge); rotary rows of random angles, one in eight cos 0 and sin -1, one in eight an infinite cos and a
    NaN sin."""
    hd, half = CHUNKS * QG, ROPE // 2
    x = rng.standard_normal((n, hd)) * rng.choice([2.0 ** -130, 2.0 ** -20, 1.0, 3.0, 2.0 ** 100], (n, 1))
    kind = rng.integers(0, 8, n)
    x[kind == 0] = 0
    x[kind == 1] = 448.0 * np.exp2(rng.integers(-8, 8, ((kind == 1).sum(), hd))) * rng.choice([-1.0, 1.0], ((kind == 1).sum(), hd))
    ang = rng.uniform(0, 2 * np.pi, (n, half))
    cs = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32).view(np.uint32)
    rot = rng.integers(0, 8, n)
    cs[rot == 0, :half], cs[rot == 0, half:] = 0, 0xBF800000
    cs[rot == 1, 0], cs[rot == 1, ROPE - 1] = 0x7F800000, 0x7FC00000
    return [(x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16), cs]


SAMPLES: dict[str, Callable[[], Callable[..., list[np.ndarray]]]] = {
    "Dsv4MlaCompressC4_v1": lambda: _sample_compress("Dsv4MlaCompressC4_v1"),
    "Dsv4MlaCompressC128_v1": lambda: _sample_compress("Dsv4MlaCompressC128_v1"),
    **{f"GatherI32x{v}_v1": (lambda _v=v: _sample_gather(_v)) for v in GATHER_VOCABS},
    **{f: (lambda _f=f: _sample_flashmla(_f)) for f in ("FlashMlaDecode_v1", "FlashMlaPrefill_v1", "FlashMlaGatherRow_v1")},
    "Dsv4QNormRopeKvInsert_v1": _sample_insert,
    "MoeRouterSqrtSoftplus_v1": lambda: _sample_router,
    "MoeRouterHash_v1": lambda: _sample_hash,
    "SiluMulClamp_v1": lambda: _sample_clamp,
    "Dsv4CompressorScore_v1": lambda: _sample_score,
    "Dsv4InvRopeQuantHead_v1": lambda: _sample_head,
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds (the served statics are too wide for the reference)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_norm as DN
    from verity.ml.definitions.registry import dsv4_compressor as C
    from verity.ml.definitions.registry import dsv4_decode_indices as DI
    from verity.ml.definitions.registry import dsv4_indexer as IX
    from verity.ml.definitions.registry import dsv4_moe as MO
    from verity.ml.definitions.registry import dsv4_qnorm_insert as QI
    from verity.ml.definitions.registry import flashmla as FM
    from verity.ml.definitions.registry import flashmla_prefill as FP
    from verity.ml.definitions.registry import mhc as M
    from verity.ml.definitions.registry.hopper import HopperBF16WgmmaDot16
    eps = 1e-6
    fuse = dict(REPEAT=2, EPS=eps, HC_EPS=eps)
    if family.startswith("GatherI32x"):
        return [DI.gather_i32(int(family[len("GatherI32x"):-len("_v1")]))]
    attn = dict(DOT=HopperBF16WgmmaDot16, SCALE=32 ** -0.5)
    decode = dict(NH=2, DN=16, DR=16, QT=16, BN=16, **attn)
    return {
        "FlashMlaDecode_v1": [bind(FM.FlashMlaDecode, SPLITS=(0, 1, 2), NW=16, **decode),
                              bind(FM.FlashMlaDecode, SPLITS=(0, 2), NW=32, **decode)],
        "FlashMlaPrefill_v1": [bind(FP.FlashMlaPrefill, NH=2, NP=1, D=32, BN=16, **attn),
                               bind(FP.FlashMlaPrefill, NH=1, NP=2, D=32, BN=16, **attn)],
        "FlashMlaGatherRow_v1": [bind(FP.FlashMlaGatherRow, DN=32, DR=16, QT=16)],
        "Dsv4QNormRopeKvInsert_v1": [bind(QI.Dsv4QNormRopeKvInsert, NH=2, NHP=4, DN=48, DR=16, QB=16, EPS=eps)],
        "MhcPost_v1": [bind(M.MhcPost, HC=4, N=8), bind(M.MhcPost, HC=2, N=4)],
        "MhcFusedPostPre_v1": [bind(M.MhcFusedPostPre, HC=2, H=32, NS=2, TH=4, TN=1, **fuse)],
        "MhcPostPreTf32_v1": [bind(M.MhcPostPreTf32, HC=2, H=64, NS=2, TN=2, **fuse)],
        "MhcPreBroadcast_v1": [bind(M.MhcPreBroadcast, HC=2, H=64, NS=1, TN=2, **fuse)],
        "MhcResidualBroadcast_v1": [bind(M.MhcResidualBroadcast, HC=4, H=8)],
        "HcHead_v1": [bind(M.HcHead, HC=2, H=16, TH=th, EPS=eps, HC_EPS=eps) for th in (1, 2)],
        "Dsv4MlaCompressC4_v1": [bind(C.Dsv4MlaCompressC4, ROWS=rows, D=32, QB=16, EPS=eps) for rows in (8, 4)],
        "Dsv4MlaCompressC128_v1": [bind(C.Dsv4MlaCompressC128, W=2, R=4, D=32, QB=16, EPS=eps)],
        "TritonBlockRMSNorm_v1": [bind(DN.TritonBlockRMSNorm, N=n, BLOCK=b, WARPS=w, EPS=eps)
                                  for n, b, w in ((48, 64, 1), (136, 256, 2), (264, 512, 1), (512, 2048, 8))],
        "MoeRouterSqrtSoftplus_v1": [bind(MO.MoeRouterSqrtSoftplus, E=e, SCALE=2.5) for e in (6, 16)],
        "MoeRouterHash_v1": [bind(MO.MoeRouterHash, E=e, SCALE=2.5) for e in (8, 16)],
        "SiluMulClamp_v1": [bind(MO.SiluMulClamp, I=i, LIMIT=lim) for i, lim in ((8, 10.0), (9, 10.0), (4, 0.5))],
        "Dsv4CompressorScore_v1": [bind(IX.Dsv4CompressorScore, W=w) for w in (8, 64)],
        "Dsv4InvRopeQuantHead_v1": [bind(IX.Dsv4InvRopeQuantHead, CHUNKS=c, QG=g, ROPE=r)
                                    for c, g, r in ((2, 8, 4), (1, 16, 16), (3, 4, 2), (2, 32, 8))],
    }[family]


def register_rows(evaluators: dict, row: Callable, provenance: dict) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    provenance |= PROVENANCE
    for family, (fn, returns) in ROWS.items():
        sample = SAMPLES.get(family)
        row(family, fn, returns, sample=sample() if sample else None, check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["EVALUATORS", "GATHER_VOCABS", "PROVENANCE", "ROWS", "check_targets", "register_rows"]
