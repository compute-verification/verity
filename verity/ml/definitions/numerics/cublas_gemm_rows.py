"""The replay row kernels of cuBLAS's bf16 GEMMs at the cc 9.0 route pin's routes (`registry.cublas_bf16`), one token row each:

* `GemmF32Out_v1{K,N,DOT}`: ``torch.mm(x, W.T, out_dtype=float32)`` on a non-split route: the ascending k16 chain of the DOT
  static's step from +0, the f32 accumulator the output;
* `CublasSplitKGemmF32Out_v1{K,N,SPLIT_K,DOT}`: the same on a split-K route (the MoE router gate past the cuteDSL tier, the
  compressors): one chain from +0 per `cublas_bf16.split_bounds` chunk, then the f32 partials added serially in split order;
* `CublasSplitKGemm_v1{K,N,SPLIT_K,DOT}`: a bf16-output linear on a split-K route: that sum rounded once by f2fp_bf16.

These are the only evaluators of the route pin's split counts: a row whose committed words came off another route (the split-K-off
switch serves the router's M17 unsplit) differs from them, word by word.  An unregistered DOT, a SPLIT_K `split_bounds` refuses,
and a coordinate with a non-finite operand or chain accumulator decline the row by name, as `rows.gemm_dot` does.  The row kernels
register from `rows` (kernel "rows") through `register_rows`.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.numerics import dense_rows
from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static

GEMM_F32_OUT, SPLITK_F32_OUT, SPLITK_BF16 = "GemmF32Out_v1", "CublasSplitKGemmF32Out_v1", "CublasSplitKGemm_v1"
_CHAIN = "dense_rows.chain_rows(DOT) -- the ascending k16 chain from +0, word for word derived_rows.dot_bf16_for(DOT) --"
EVALUATORS = {
    GEMM_F32_OUT: (f"verity.ml.definitions.numerics.cublas_gemm_rows.gemm_f32_out (GemmF32Out_v1{{K,N,DOT}}: {_CHAIN} of x against "
                   "each W row, the f32 accumulator the output; cuBLAS's non-split torch.mm out_dtype=float32, "
                   "bf16_gemm_check.f32out_rows)"),
    SPLITK_F32_OUT: (f"verity.ml.definitions.numerics.cublas_gemm_rows.splitk_f32_out (CublasSplitKGemmF32Out_v1{{K,N,SPLIT_K,DOT}}: {_CHAIN} "
                     "over each cublas_bf16.split_bounds(K, SPLIT_K) chunk, then F32Add of the partials in increasing split order; "
                     "cuBLAS's split-K route with its splitKreduce kernel, bf16_gemm_check.cublas_splitk_rows out=f32)"),
    SPLITK_BF16: (f"verity.ml.definitions.numerics.cublas_gemm_rows.splitk_bf16 (CublasSplitKGemm_v1{{K,N,SPLIT_K,DOT}}: "
                  "CublasSplitKGemmF32Out_v1's sum, then f2fp_bf16 once; bf16_gemm_check.cublas_splitk_rows out=bf16)"),
}
#: The H200 capture that holds cuBLAS's served-default routes to these kernels' models, word for word: every split-K launch's words
#: fit `contiguous-f32-forward` (`bf16_gemm_check.CUBLAS_SPLITK_HYPOTHESIS`) and none of its alternatives, at each split count the
#: route pin reads (`bf16_gemm_check.default_cublas_route`); the decode-batch scan that maps those counts past M 256 is `scan`.
_CAPTURE = EVIDENCE["glm53_bf16_gemm_capture"]
PROVENANCE = {
    "property": "glm53_bf16_gemm_capture (verity-vllm/glm53-bf16-gemm-capture/v3), tests/properties/glm53_bf16_gemm_capture_gpu.py",
    "run": _CAPTURE["run"],
    "record": _CAPTURE["record"],
    "record_served_default": _CAPTURE["record_served_default"],
    "art": _CAPTURE["art"],
    "scan": f"{EVIDENCE['cublas_route_scan']['run']} (cublas_routes.json, its conditions bf16_gemm_check.ROUTE_CONDITIONS)",
    "device": "vy-glm53-eu, cc 9.0 with 132 SMs, driver 580.173.02, torch 2.13.0+cu129 on the venv's libcublas.so.12, vLLM "
              "0.28.1rc1.dev472+gd9105ea80",
    "captured_utc": _CAPTURE["captured_utc"],
    "variants": {"split_k_off": "GO", "served_default": "GO"},
    "served_default_split_k": {
        "glm53/router/mlp.gate": {"K": 6144, "N": 256, "out": "f32", "SPLIT_K": {"17": 4, "32": 4, "128": 4, "256": 3}},
        "glm53/linear/indexer.wk_weights_proj": {"K": 6144, "N": 160, "out": "bf16", "SPLIT_K": {"1": 4, "4": 4, "16": 4, "128": 4, "256": 4}},
        "dsv4-pro/router/mlp.gate": {"K": 7168, "N": 384, "out": "f32", "SPLIT_K": {"17": 5, "32": 5, "128": 4, "256": 4}},
        "dsv4-pro/linear/indexer.weights_proj": {"K": 7168, "N": 64, "out": "bf16", "SPLIT_K": {"1": 5, "4": 5, "16": 5, "128": 5, "256": 5}},
        "dsv4-pro/f32out/compressor.kv_score.c4": {"K": 7168, "N": 2048, "out": "f32", "SPLIT_K": {"1": 3, "4": 3, "16": 3, "128": 2, "256": None}},
        "dsv4-pro/f32out/compressor.kv_score.c128": {"K": 7168, "N": 1024, "out": "f32", "SPLIT_K": {"1": 5, "4": 5, "16": 5, "128": 3, "256": 2}},
        "dsv4-pro/f32out/indexer.compressor.kv_score": {"K": 7168, "N": 512, "out": "f32", "SPLIT_K": {"1": 5, "4": 5, "16": 5, "128": 3, "256": 3}},
    },
    "kernels": "served default: a cuBLASLt nvjet_tss (f32 out) / nvjet_tst (bf16 out) splitK launch, then cublasLt::splitKreduce_kernel; "
               "split-K off (the switch's CUBLAS_WORKSPACE_CONFIG :16:8, CUBLASLT_WORKSPACE_SIZE 1): one launch, no reduction kernel",
    "tests": ["tests/check/test_sampled_replay_cublas_gemm.py::test_row_kernels_are_the_capture_models_at_glm53s_shapes",
              "tests/program/test_kernel_self_check.py (GemmF32Out_v1, CublasSplitKGemmF32Out_v1, CublasSplitKGemm_v1, kernel rows)"],
    "definitions": "verity/ml/definitions/registry/cublas_bf16.py: GemmF32Out_v1{K,N,DOT}, CublasSplitKGemmF32Out_v1 / CublasSplitKGemm_v1"
                   "{K,N,SPLIT_K,DOT} (the reference; the kernels are checked against them, never the other way round)",
}


def _operands(family: str, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    dot_id = prim_static(statics, "DOT", family)
    if dot_id not in dense_rows.STEPS:
        raise Declined(f"no registered evaluator for {family} with DOT={dot_id} (k16 step twins: {sorted(dense_rows.STEPS)})")
    K, N = int(statics["K"]), int(statics["N"])
    if len(ops) != 2:
        raise Declined(f"{family}: the row wires {len(ops)} args, the registered Definition takes 2 (x, w)")
    x, w = np.asarray(ops[0], dtype=np.uint16).reshape(-1), np.asarray(ops[1], dtype=np.uint16).reshape(-1)
    if x.size != K or w.size != N * K:
        raise Declined(f"{family}: x has {x.size} words and w {w.size}, the signature has K = {K} and N*K = {N * K}")
    return x, w.reshape(N, K), dot_id


def _chain(family: str, x: np.ndarray, w: np.ndarray, dot_id: str) -> np.ndarray:
    acc, nonfin = dense_rows.chain_rows(x[None, :], np.ascontiguousarray(w), dot_id)
    if nonfin.any():
        raise Declined(f"{family}: {int(nonfin.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                       "(outside the vectorised k16 twin's domain)")
    return acc[0]


def _splitk_sum(family: str, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> np.ndarray:
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.registry.cublas_bf16 import split_bounds

    x, w, dot_id = _operands(family, statics, ops)
    try:
        bounds = split_bounds(x.size, int(statics["SPLIT_K"]))
    except ValueError as e:
        raise Declined(f"{family}: {e}") from None
    acc = None
    for lo, hi in bounds:
        part = _chain(family, x[lo:hi], w[:, lo:hi], dot_id)
        acc = part if acc is None else D.add_rn(acc, part)
    return np.asarray(acc, dtype=np.uint32)


def gemm_f32_out(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GemmF32Out_v1{K,N,DOT}(x, w)`: x the K bf16 words of one token row, w the N x K words of W -> out, N f32 words."""
    x, w, dot_id = _operands(GEMM_F32_OUT, statics, ops)
    out = _chain(GEMM_F32_OUT, x, w, dot_id)
    return {"0": out, "out": out}


def splitk_f32_out(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`CublasSplitKGemmF32Out_v1{K,N,SPLIT_K,DOT}(x, w)` -> out, N f32 words."""
    out = _splitk_sum(SPLITK_F32_OUT, statics, ops)
    return {"0": out, "out": out}


def splitk_bf16(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`CublasSplitKGemm_v1{K,N,SPLIT_K,DOT}(x, w)` -> out, N bf16 words."""
    from verity.ml.definitions.numerics import derived_rows as D
    out = D.f2fp_bf16(_splitk_sum(SPLITK_BF16, statics, ops)).astype(np.uint16)
    return {"0": out, "out": out}


ROWS = {GEMM_F32_OUT: gemm_f32_out, SPLITK_F32_OUT: splitk_f32_out, SPLITK_BF16: splitk_bf16}


def check_targets(family: str) -> list:
    """K = 64 (four k16 blocks, the circuit-check binding's statics) and K = 160 (two 64-element chunks and a 32-element remainder at
    SPLIT_K 3), N = 3, at the Ampere and the Hopper step."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1, cublas_bf16 as CB, hopper

    dots = (b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16)
    if family == GEMM_F32_OUT:
        return [bind(CB.GemmF32OutWord, K=64, N=3, DOT=d) for d in dots]
    d = {SPLITK_F32_OUT: CB.CublasSplitKGemmF32OutWord, SPLITK_BF16: CB.CublasSplitKGemmWord}[family]
    return [bind(d, K=k, N=3, SPLIT_K=s, DOT=dot) for k, s in ((64, 4), (160, 3)) for dot in dots]


def register_rows(evaluators: dict, row: Callable, provenance: dict) -> None:
    """Register this module's row evaluators, with their provenance, without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    for family, fn in ROWS.items():
        row(family, fn, ("out",), check_targets=lambda rng, f=family: check_targets(f))
        provenance[family] = PROVENANCE


__all__ = ["EVALUATORS", "GEMM_F32_OUT", "PROVENANCE", "ROWS", "SPLITK_BF16", "SPLITK_F32_OUT", "check_targets", "gemm_f32_out",
           "register_rows", "splitk_bf16", "splitk_f32_out"]
