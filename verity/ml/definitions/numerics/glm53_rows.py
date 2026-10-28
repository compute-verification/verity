"""The replay row kernels of GLM-5.3's served rows that no other module evaluates, one per Definition family, each the numpy twin its
Definition is held to where one exists (`cuda_rmsnorm_rows`, `bf16_gemm_check`, `deepgemm_twins`, `fp8_moe_rows`,
`sparse_mla_split_twins`, `derived_rows`), else the Definition itself by core's word reference (`DsaIndexerKRow_v1`, `DsaIndexerQ_v1`,
`MoeRouterSigmoidBias_v1`: their Boolean forms do not fit a replay worker at GLM-5.3's statics).  The MoE rows read their plane
inputs (`in_q` / `in_scale`, `act_q` / `act_scale`, `router_ids`, `router_w`) as the committed words the replay resolves, and gather
the expert by the committed id.  A coordinate outside a twin's domain (a non-finite lane, an unregistered DOT / INV step, an expert
id outside [0, E)) declines the row by name.  The row kernels register from `rows` (kernel "rows"), with `PROVENANCE`, the H200
capture that holds a twin to the served kernel where the twin's Definition is not that capture's own.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static
from verity.ml.definitions.registry.fp8 import weight_blocks

EVALUATORS = {
    "RMSNormCudaVector8_v1": "verity.ml.definitions.numerics.cuda_rmsnorm_rows.rms_norm_rows (RMSNormCudaVector8_v1{N,EPS,BLOCK}: per thread one "
                             "f32 FMA square chain over its width-8 vectors, CUB BlockReduce, rsqrt.approx, two f32 products, RNE to bf16)",
    "RMSNormFusedCudaVector8_v1": "verity.ml.definitions.numerics.cuda_rmsnorm_rows.fused_add_rms_norm_rows (RMSNormFusedCudaVector8_v1{N,EPS,"
                                  "BLOCK}: the bf16 residual add, four square-pair sums per vector, CUB BlockReduce, rsqrt.approx) -> out, residual",
    "LlBf16DotprodGemm_v1": "verity.ml.definitions.registry.bf16_gemm_check.dotprod_rows (LlBf16DotprodGemm_v1{N,K,BS,VEC,FMA}: BS thread "
                            "chains from +0, the VEC/2 tail, the FMA or FMUL+FADD rounds, each warp's butterfly, the warp sums in order) -> f32",
    "LlBf16SplitKGemm_v1": "verity.ml.definitions.registry.bf16_gemm_check.splitk_rows (LlBf16SplitKGemm_v1{N,K,SPLIT_K,TILE_K,WARPS,DOT} at "
                           "DOT HopperBF16WgmmaDot16_v1: per cluster rank and MMA warp one k16 DOT chain from +0 over its blocks, the "
                           "warps' chains then the ranks' sums added in order from +0) -> f32",
    "Fp8GroupQuantUe8m0_v1": "verity.ml.definitions.numerics.deepgemm_twins.ue8m0_group_quant_row (Fp8GroupQuantUe8m0_v1{K,G}: the fmaxf fold "
                             "from 1e-10, div.rn by 448, the ue8m0 ceiling, div.rn, cvt.rn.satfinite.e4m3) -> q, s",
    "MoeExpertFp8Gemm_v1": "verity.ml.definitions.numerics.fp8_moe_rows.moe_expert_fp8_block_row over w[e], ws[e], e the committed router_ids "
                           "word (MoeExpertFp8Gemm_v1{E,K,N,G,KB,SWAP,DOT} = MoeExpertFp8BlockGemm_v3 at expert e: per K tile KB/32 DOT "
                           "steps from +0, FFMA(FMUL(d, first), second, acc), cvt.rn.bf16.f32) -> up",
    "MoeExpertFp8GemmW_v1": "MoeExpertFp8Gemm_v1's row with the committed router_w word g multiplied into the f32 accumulator (one FMUL) before "
                            "cvt.rn.bf16.f32 (MoeExpertFp8GemmW_v1 = MoeExpertFp8BlockGemmW_v3 at expert e) -> down",
    "SparseMlaSplit_v1": "verity.ml.definitions.numerics.sparse_mla_split_twins.head_split_rows per head (SparseMlaSplit_v1{T,NS,NH,DR,DV,BN,DOT,"
                         "INV,SCALE}: each split's SparseMlaBlock_v1s last to first, SparseMlaSplitEnd_v1, SparseMlaCombine_v1; NS = 1 "
                         "SparseMlaHead_v1), the DOT / INV steps resolved by name",
    "ResidualAdd_v1": "Bf16Add per coordinate (ResidualAdd_v1{N}: one f32 add of the widened words, RNE to bf16; BiasAdd_v1's row)",
    "RoPEHeadGptJ_v2": "verity.ml.definitions.numerics.derived_rows fma_rn / mul_rn / bf16_rn_torch (RoPEHeadGptJ_v2{D}: out[2i] = RopeOut_v1("
                       "x[2i], x[2i+1], cos[i], sin[i]), out[2i+1] = RopeOutAdd_v1(x[2i+1], x[2i], sin[i], cos[i]), node for node)",
    "DsaIndexerKRow_v1": "registry.dsa.DsaIndexerKRow{D,ROPE,EPS} by verity.ml.circuits.evaluation.evaluate_batch (the word reference) -> "
                         "the slot's e4m3 bytes, its f32 scale",
    "DsaIndexerQ_v1": "registry.dsa.DsaIndexerQ{H,D,ROPE} by verity.ml.circuits.evaluation.evaluate_batch (the word reference) -> "
                      "index_q_fp8 [H, D] e4m3, index_weights_out [H] f32",
    "MoeRouterSigmoidBias_v1": "registry.moe.MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL} by verity.ml.circuits.evaluation.evaluate_batch (the "
                               "word reference: _moe_C.grouped_topk's sigmoid, biased top-k and renormalisation) -> router_w f32, router_ids i32",
}

#: The H200 capture that holds cuteDSL `LLBf16SplitK`'s words to `bf16_gemm_check.splitk_rows`, word for word, at each gate's every
#: split-K M (the dispatch table's SPLIT_K, the launch's grid.z): the split-K-off record and the served default's (vLLM's cuBLAS
#: switch does not reach the cuteDSL tier).
_CAPTURE = EVIDENCE["glm53_bf16_gemm_capture"]
PROVENANCE = {"LlBf16SplitKGemm_v1": {
    "property": "glm53_bf16_gemm_capture (verity-vllm/glm53-bf16-gemm-capture/v3), tests/properties/glm53_bf16_gemm_capture_gpu.py",
    "run": _CAPTURE["run"],
    "record": _CAPTURE["record"],
    "record_served_default": _CAPTURE["record_served_default"],
    "art": _CAPTURE["art"],
    "device": "vy-glm53-eu, cc 9.0 with 132 SMs, driver 580.173.02, torch 2.13.0+cu129, vLLM 0.28.1rc1.dev472+gd9105ea80",
    "captured_utc": _CAPTURE["captured_utc"],
    "variants": {"split_k_off": "GO", "served_default": "GO"},
    "cases": {
        "glm53/router/mlp.gate": {"K": 6144, "N": 256, "TILE_K": 256, "WARPS": 4,
                                  "SPLIT_K": {"5": 6, "8": 6, "9": 8, "11": 8, "12": 6, "16": 6}, "words": 15616, "mismatch": 0},
        "dsv4-pro/router/mlp.gate": {"K": 7168, "N": 384, "TILE_K": 256, "WARPS": 4,
                                     "SPLIT_K": {"5": 6, "6": 8, "7": 7, "8": 8, "12": 8, "13": 7, "16": 7}, "words": 25728, "mismatch": 0},
    },
    "kernels": "vLLM d9105ea80 kernels/linear/cute_dsl/ll_bf16.py LLBf16SplitK, one launch of grid (1, N/16, SPLIT_K), no reduction kernel",
    "tests": ["tests/check/test_sampled_replay_glm53_rows.py::test_the_split_k_gate_at_glm53s_served_statics_is_the_definition",
              "tests/program/test_kernel_self_check.py (LlBf16SplitKGemm_v1, kernel rows)"],
    "definitions": "verity/ml/definitions/registry/ll_bf16.py: LlBf16SplitKGemm_v1{N,K,SPLIT_K,TILE_K,WARPS,DOT} (the reference; the kernel "
                   "is checked against it, never the other way round)",
}}


def _float(v: Any) -> float:
    """A float static as the spec text spells it (`2.5`, `1e-06`, `{"f64":"0x1.4p+1"}`) or as a binding holds it."""
    if isinstance(v, str) and v.lstrip().startswith("{"):
        v = json.loads(v)
    if isinstance(v, dict) and set(v) == {"f64"}:
        return float.fromhex(str(v["f64"]))
    if isinstance(v, str):
        return float.fromhex(v) if v.lstrip("-").startswith("0x") else float(v)
    return float(v)


def _flag(v: Any) -> bool:
    if isinstance(v, str):
        if v.strip().lower() not in ("true", "false"):
            raise ValueError(f"boolean static {v!r}")
        return v.strip().lower() == "true"
    return bool(v)


def _name(v: Any) -> str:
    """A string static: `'block'` and `"block"` as the spec text quotes it, or the bare word."""
    return str(v).strip().strip("'\"")


def _declines(family: str, nonfin: np.ndarray) -> None:
    if np.asarray(nonfin).any():
        raise Declined(f"{family}: {int(np.count_nonzero(nonfin))} coordinates have a non-finite operand or accumulator "
                       "(outside the vectorised twin's domain)")


def _bf16(a: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(a, dtype=np.uint16).reshape(n)


def rmsnorm_cuda_vector8(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`RMSNormCudaVector8_v1{N,EPS,BLOCK}(x, w)` -> out (N bf16 words)."""
    from verity.ml.definitions.numerics.cuda_rmsnorm_rows import rms_norm_rows
    N = int(statics["N"])
    got = rms_norm_rows(_bf16(ops[0], N).reshape(1, N), _bf16(ops[1], N), _float(statics["EPS"]), int(statics["BLOCK"]))
    return {"0": np.asarray(got["out"], dtype=np.uint16).reshape(N)}


def rmsnorm_fused_cuda_vector8(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`RMSNormFusedCudaVector8_v1{N,EPS,BLOCK}(x, res, w)` -> ports 0 (out) and 1 (the residual), N bf16 words each."""
    from verity.ml.definitions.numerics.cuda_rmsnorm_rows import fused_add_rms_norm_rows
    N = int(statics["N"])
    got = fused_add_rms_norm_rows(_bf16(ops[0], N).reshape(1, N), _bf16(ops[1], N).reshape(1, N), _bf16(ops[2], N),
                                  _float(statics["EPS"]), int(statics["BLOCK"]))
    return {"0": np.asarray(got["out"], dtype=np.uint16).reshape(N), "1": np.asarray(got["residual"], dtype=np.uint16).reshape(N)}


def ll_dotprod_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`LlBf16DotprodGemm_v1{N,K,BS,VEC,FMA}(x, w)` -> N f32 words."""
    from verity.ml.definitions.registry.bf16_gemm_check import dotprod_rows
    N, K = int(statics["N"]), int(statics["K"])
    out = dotprod_rows(_bf16(ops[0], K)[None], _bf16(ops[1], N * K).reshape(N, K), BS=int(statics["BS"]), VEC=int(statics["VEC"]),
                       FMA=int(statics["FMA"]))
    return {"0": np.asarray(out, dtype=np.uint32).reshape(N)}


def ll_splitk_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`LlBf16SplitKGemm_v1{N,K,SPLIT_K,TILE_K,WARPS,DOT}(x, w)` -> N f32 words."""
    from verity.ml.definitions.registry.bf16_gemm_check import splitk_rows
    from verity.ml.definitions.registry.mla_check import HOPPER
    f = "LlBf16SplitKGemm_v1"
    dot = prim_static(statics, "DOT", f)
    if dot != HOPPER:
        raise Declined(f"{f}: no twin for DOT={dot} (splitk_rows chains {HOPPER})")
    N, K = int(statics["N"]), int(statics["K"])
    try:
        out = splitk_rows(_bf16(ops[0], K)[None], _bf16(ops[1], N * K).reshape(N, K), SPLIT_K=int(statics["SPLIT_K"]),
                          TILE_K=int(statics["TILE_K"]), WARPS=int(statics["WARPS"]))
    except (ValueError, ZeroDivisionError) as e:
        raise Declined(f"{f}: {e}") from None
    return {"0": np.asarray(out, dtype=np.uint32).reshape(N)}


def fp8_group_quant_ue8m0(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Fp8GroupQuantUe8m0_v1{K,G}(x)` -> ports 0 (K e4m3 bytes) and 1 (K/G f32 scales)."""
    from verity.ml.definitions.numerics.deepgemm_twins import ue8m0_group_quant_row
    K, G = int(statics["K"]), int(statics["G"])
    q, s = ue8m0_group_quant_row(_bf16(ops[0], K), G=G)
    return {"0": np.asarray(q, dtype=np.uint8).reshape(K), "1": np.asarray(s, dtype=np.uint32).reshape(K // G)}


def _moe_expert_fp8(family: str, routed: bool, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity.ml.definitions.numerics.fp8_moe_rows import TWIN_DOT_OF, moe_expert_fp8_block_row
    dot = prim_static(statics, "DOT", family)
    if dot not in TWIN_DOT_OF:
        raise Declined(f"{family}: no twin for DOT={dot} (k32 step twins: {sorted(TWIN_DOT_OF)})")
    E, K, N, G = (int(statics[k]) for k in ("E", "K", "N", "G"))
    e = int(np.asarray(ops[2]).reshape(-1)[0])
    if not 0 <= e < E:
        raise Declined(f"{family}: routed expert id {e} outside [0, {E}) (the Definition's gathers select NaN words there)")
    wi = 4 if routed else 3
    w, ws = np.asarray(ops[wi], dtype=np.uint8), np.asarray(ops[wi + 1], dtype=np.uint32)
    if w.size != E * N * K or ws.size != E * weight_blocks(N, G) * (K // G):
        raise Declined(f"{family}: expert stacks of {w.size} / {ws.size} words, the signature has E*N*K = {E * N * K} / "
                       f"E*cdiv(N,G)*(K/G) = {E * weight_blocks(N, G) * (K // G)}")
    g = int(np.asarray(ops[3]).reshape(-1)[0]) if routed else None
    out, nonfin = moe_expert_fp8_block_row(
        np.asarray(ops[0], dtype=np.uint8).reshape(K), np.asarray(ops[1], dtype=np.uint32).reshape(K // G),
        w.reshape(E, N, K)[e], ws.reshape(E, weight_blocks(N, G), K // G)[e], G=G, KB=int(statics["KB"]), swap=_flag(statics["SWAP"]),
        g=g, dot=TWIN_DOT_OF[dot])
    _declines(family, nonfin)
    return {"0": np.asarray(out, dtype=np.uint16).reshape(N)}


def moe_expert_fp8_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeExpertFp8Gemm_v1{E,K,N,G,KB,SWAP,DOT}(xq, sx, e, w, ws)` -> N bf16 words (the pair's `up`)."""
    return _moe_expert_fp8("MoeExpertFp8Gemm_v1", False, statics, ops)


def moe_expert_fp8_gemm_w(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeExpertFp8GemmW_v1{E,K,N,G,KB,SWAP,DOT}(xq, sx, e, g, w, ws)` -> N bf16 words (the pair's `down`)."""
    return _moe_expert_fp8("MoeExpertFp8GemmW_v1", True, statics, ops)


def sparse_mla_split(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`SparseMlaSplit_v1{T,NS,NH,DR,DV,BN,DOT,INV,SCALE}(q_nope, q_rope, kv)` on one token -> NH x DV bf16 words, head-major."""
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.numerics.sparse_mla_split_twins import head_split_rows
    f = "SparseMlaSplit_v1"
    dot, inv = prim_static(statics, "DOT", f), prim_static(statics, "INV", f)
    if dot not in D.TC_DOT16_BY_PRIM or inv not in D.INV_SUM_BY_PRIM:
        raise Declined(f"{f}: no twin for DOT={dot} / INV={inv} (k16 steps {sorted(D.TC_DOT16_BY_PRIM)}, INV {sorted(D.INV_SUM_BY_PRIM)})")
    T, NS, NH, DR, DV, BN = (int(statics[k]) for k in ("T", "NS", "NH", "DR", "DV", "BN"))
    out, nonfin = head_split_rows(_bf16(ops[0], NH * DV).reshape(NH, DV), _bf16(ops[1], NH * DR).reshape(NH, DR),
                                  _bf16(ops[2], T * (DV + DR)).reshape(T, DV + DR), T=T, NS=NS, DR_=DR, DV=DV, BN=BN,
                                  scale=_float(statics["SCALE"]), dot16=D.TC_DOT16_BY_PRIM[dot], inv=D.INV_SUM_BY_PRIM[inv])
    _declines(f, nonfin)
    return {"0": np.asarray(out, dtype=np.uint16).reshape(NH * DV)}


def residual_add(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`ResidualAdd_v1{N}(x, r)` -> N bf16 words."""
    from verity.ml.definitions.numerics import derived_rows as D
    N = int(statics["N"])
    x, r = (D.as_f32(D.bf16_to_f32_words(_bf16(a, N))) for a in (ops[0], ops[1]))
    return {"0": np.asarray(D.f2fp_bf16(D.as_u32((x + r).astype(np.float32))), dtype=np.uint16)}


def rope_head_gptj(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`RoPEHeadGptJ_v2{D}(x, cs)`, cs = [cos[0:R] | sin[0:R]] -> D bf16 words, the pairs (2i, 2i+1) rotated."""
    from verity.ml.definitions.numerics import derived_rows as D
    Dh = int(statics["D"])
    R = Dh // 2
    xf, cs = D.bf16_to_f32_words(_bf16(ops[0], Dh)), D.bf16_to_f32_words(_bf16(ops[1], Dh))
    xe, xo, c, s = xf[0::2], xf[1::2], cs[:R], cs[R:]
    out = np.empty(Dh, dtype=np.uint16)
    out[0::2] = D.bf16_rn_torch(D.fma_rn(xe, c, D.mul_rn(xo, s) ^ np.uint32(0x80000000)))
    out[1::2] = D.bf16_rn_torch(D.fma_rn(xe, s, D.mul_rn(xo, c)))
    return {"0": out}


def _reference(definition: Any, names: tuple[str, ...], statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`definition` bound at `statics` on one row by core's word reference, its return value split into its Tuple fields (`names`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.evaluation.batch import type_dtype
    from verity.ml.circuits.evaluation.reference import signature_of
    d = bind(definition, **statics)
    params, ret = signature_of(d)
    if len(ops) != len(params):
        raise Declined(f"{d.id}: the row wires {len(ops)} args, the Definition takes {len(params)}")
    arrays = [np.asarray(ops[g]).reshape(1, -1).astype(type_dtype(pt), copy=False) for g, pt in enumerate(params)]
    if any(a.shape[1] != pt.leaves for a, pt in zip(arrays, params)):
        raise Declined(f"{d.id}: arg words {[a.shape[1] for a in arrays]} != the Definition's {[pt.leaves for pt in params]}")
    flat = np.asarray(evaluate_batch(d, *arrays, kernel="reference")).reshape(-1)
    out, off = {}, 0
    for name, ft in zip(names, ret.fields):
        out[name] = flat[off:off + ft.leaves].astype(type_dtype(ft))
        off += ft.leaves
    return out


def dsa_indexer_k_row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`DsaIndexerKRow_v1{D,ROPE,EPS}(k, w, b, cs)` -> ports 0 (D e4m3 bytes) and 1 (the f32 scale)."""
    from verity.ml.definitions.registry import dsa
    st = {"D": int(statics["D"]), "ROPE": int(statics["ROPE"]), "EPS": _float(statics["EPS"])}
    return _reference(dsa.DsaIndexerKRow, ("0", "1"), st, ops)


def dsa_indexer_q(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`DsaIndexerQ_v1{H,D,ROPE}(q, cs, w)` -> ports 0 (H x D e4m3 bytes) and 1 (H f32 weights)."""
    from verity.ml.definitions.registry import dsa
    return _reference(dsa.DsaIndexerQ, ("0", "1"), {k: int(statics[k]) for k in ("H", "D", "ROPE")}, ops)


def moe_router_sigmoid_bias(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MoeRouterSigmoidBias_v1{E,TOPK,SCALE,KERNEL}(logits, bias)` -> router_w (TOPK f32) and router_ids (TOPK i32)."""
    from verity.ml.definitions.registry import moe
    st = {"E": int(statics["E"]), "TOPK": int(statics["TOPK"]), "SCALE": _float(statics["SCALE"]), "KERNEL": _name(statics["KERNEL"])}
    return _reference(moe.MoeRouterSigmoidBias, ("router_w", "router_ids"), st, ops)


#: family -> (row kernel, the members it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "RMSNormCudaVector8_v1": (rmsnorm_cuda_vector8, ("0",)),
    "RMSNormFusedCudaVector8_v1": (rmsnorm_fused_cuda_vector8, ("0", "1")),
    "LlBf16DotprodGemm_v1": (ll_dotprod_gemm, ("0",)),
    "LlBf16SplitKGemm_v1": (ll_splitk_gemm, ("0",)),
    "Fp8GroupQuantUe8m0_v1": (fp8_group_quant_ue8m0, ("0", "1")),
    "MoeExpertFp8Gemm_v1": (moe_expert_fp8_gemm, ("0",)),
    "MoeExpertFp8GemmW_v1": (moe_expert_fp8_gemm_w, ("0",)),
    "SparseMlaSplit_v1": (sparse_mla_split, ("0",)),
    "ResidualAdd_v1": (residual_add, ("0",)),
    "RoPEHeadGptJ_v2": (rope_head_gptj, ("0",)),
    "DsaIndexerKRow_v1": (dsa_indexer_k_row, ("0", "1")),
    "DsaIndexerQ_v1": (dsa_indexer_q, ("0", "1")),
    "MoeRouterSigmoidBias_v1": (moe_router_sigmoid_bias, ("router_w", "router_ids")),
}

#: the parameter position of each MoE row's expert id, which the self-check samples in range but for one sample in sixteen
_EXPERT_ARG = {"MoeExpertFp8Gemm_v1": 2, "MoeExpertFp8GemmW_v1": 2}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds (the served statics are too wide for the reference)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.ml.definitions.registry import b1, boolean_rope, cuda_rmsnorm, deepgemm, dsa, fp8_moe_experts, hopper
    from verity.ml.definitions.registry import ll_bf16, moe, sparse_mla_split
    hop = hopper.HopperBF16WgmmaDot16
    moe_cases = [dict(E=2, K=128, N=96, G=64, KB=64, SWAP=False, DOT=HopperE4m3QgmmaDot32),
                 dict(E=2, K=128, N=96, G=64, KB=32, SWAP=True, DOT=HopperE4m3MmaV2Dot32),
                 dict(E=3, K=256, N=3, G=128, KB=128, SWAP=True, DOT=HopperE4m3QgmmaDot32)]
    mla = dict(DR=16, DV=16, BN=16, SCALE=0.0625, DOT=hop, INV=hopper.Fa3InvSum)
    return {
        "RMSNormCudaVector8_v1": [bind(cuda_rmsnorm.RMSNormCudaVector8, N=n, EPS=e, BLOCK=b) for n, e, b in ((16, 1e-6, 2), (64, 1e-5, 3), (64, 1e-5, 8))],
        "RMSNormFusedCudaVector8_v1": [bind(cuda_rmsnorm.RMSNormFusedCudaVector8, N=n, EPS=e, BLOCK=b)
                                       for n, e, b in ((16, 1e-6, 2), (64, 1e-5, 3), (64, 1e-5, 16))],
        "LlBf16DotprodGemm_v1": [bind(ll_bf16.LlBf16DotprodGemmWord, N=3, K=k, BS=bs, VEC=v, FMA=f)
                                 for k, bs, v, f in ((128, 64, 2, 1), (128, 64, 2, 0), (160, 32, 4, 1))],
        "LlBf16SplitKGemm_v1": [bind(ll_bf16.LlBf16SplitKGemmWord, N=n, K=k, SPLIT_K=s, TILE_K=t, WARPS=w, DOT=hop)
                                for n, k, s, t, w in ((3, 128, 2, 32, 2), (3, 192, 4, 32, 2), (2, 256, 3, 64, 2))],
        "Fp8GroupQuantUe8m0_v1": [bind(deepgemm.Fp8GroupQuantUe8m0, K=256, G=128)],
        "MoeExpertFp8Gemm_v1": [bind(fp8_moe_experts.MoeExpertFp8Gemm, **c) for c in moe_cases],
        "MoeExpertFp8GemmW_v1": [bind(fp8_moe_experts.MoeExpertFp8GemmW, **c) for c in moe_cases],
        "SparseMlaSplit_v1": [bind(sparse_mla_split.SparseMlaSplit, T=40, NS=2, NH=2, **mla), bind(sparse_mla_split.SparseMlaSplit, T=20, NS=1, NH=3, **mla)],
        "ResidualAdd_v1": [bind(b1.ResidualAdd, N=n) for n in (8, 16)],
        "RoPEHeadGptJ_v2": [bind(boolean_rope.RoPEHeadGptJV2, D=d) for d in (8, 16)],
        "DsaIndexerKRow_v1": [bind(dsa.DsaIndexerKRow, D=32, ROPE=16, EPS=1e-6)],
        "DsaIndexerQ_v1": [bind(dsa.DsaIndexerQ, H=2, D=32, ROPE=16)],
        "MoeRouterSigmoidBias_v1": [bind(moe.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL=k) for k in ("block", "warp")],
    }[family]


def register_rows(evaluators: dict, row: Callable, sample: Callable, provenance: dict) -> None:
    """Register this module's row evaluators, with their provenance, without growing the shared rows catalog (`sample(index)` =
    `rows._sample`)."""
    evaluators |= EVALUATORS
    provenance |= PROVENANCE
    for family, (fn, returns) in ROWS.items():
        idx = _EXPERT_ARG.get(family)
        row(family, fn, returns, sample=sample({idx: "E"}) if idx is not None else None,
            check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["EVALUATORS", "PROVENANCE", "ROWS", "check_targets", "register_rows"]
