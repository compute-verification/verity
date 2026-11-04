"""The replay row kernel of MLA's latent absorptions, `BatchedGemm_v1{B,K,N,DOT}` (`registry.cublas_bf16`): ``torch.bmm`` on cuBLAS
against `W_UK_T` / `W_UV`, one token row over all B heads.  Member b is `Gemm_v2{K,N,DOT}`'s row of x[b] against W[b] (``w[b][n][k]``):
the ascending k16 chain of the DOT static's step from +0, then f2fp_bf16.  An unregistered DOT, and a coordinate with a non-finite
operand or accumulator, declines the row by name, as `rows.gemm_dot` does.  The row kernel registers from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static

FAMILY = "BatchedGemm_v1"
EVALUATOR = ("verity.ml.definitions.numerics.batched_gemm_rows.batched_gemm (BatchedGemm_v1{B,K,N,DOT}: per member b, derived_rows."
             "dot_bf16_for(DOT) of x[b] against W[b] -- the ascending k16 chain from +0 -- then f2fp_bf16, Gemm_v2{K,N,DOT}'s row; "
             "cuBLAS's batched bf16 GEMM of the MLA absorbs, bf16_gemm_check.bmm_rows)")
#: The H200 capture that holds cuBLAS's ``torch.bmm`` on the served W_UK_T / W_UV views to this kernel's model (`bmm_rows`), word for
#: word, at GLM-5.3's TP8 absorb statics (B 8) and the M its TP8 Commit row serves (prefill 256, decode 1), in both configurations.
_CAPTURE = EVIDENCE["glm53_bf16_gemm_capture"]
PROVENANCE = {
    "property": "glm53_bf16_gemm_capture (verity-vllm/glm53-bf16-gemm-capture/v3), tests/properties/glm53_bf16_gemm_capture_gpu.py",
    "record": _CAPTURE["record"],
    "record_served_default": _CAPTURE["record_served_default"],
    "art": _CAPTURE["art"],
    "device": "the capture node, cc 9.0 with 132 SMs (the record's gate names the part), driver 580.173.02, torch 2.13.0+cu129 on the "
              "venv's libcublas.so.12, vLLM 0.28.1rc1.dev472+gd9105ea80 (the record's cublas_version 130101 is the node's system "
              "libcublas.so.13, which that run's find_library picked, not the library torch loaded)",
    "captured_utc": _CAPTURE["captured_utc"],
    "variants": {"split_k_off": "GO", "served_default": "GO"},
    "cases": {"q_nope_absorb": {"B": 8, "K": 192, "N": 512, "M": [1, 4, 16, 128, 256], "words": 1658880, "mismatch": 0},
              "v_absorb": {"B": 8, "K": 512, "N": 256, "M": [1, 4, 16, 128, 256], "words": 829440, "mismatch": 0}},
    "kernels": "one cuBLASLt nvjet_tst launch per bmm (NNT for q_nope, TNT for v), no split-K and no reduction kernel, in both variants",
    "tests": ["tests/check/test_sampled_replay_batched_gemm.py::test_row_kernel_is_the_capture_model_at_glm53s_tp8_shapes",
              "tests/program/test_kernel_self_check.py (BatchedGemm_v1, kernel rows, against cublas_bf16.BatchedGemmWord)"],
    "definitions": "verity/ml/definitions/registry/cublas_bf16.py: BatchedGemm_v1{B,K,N,DOT} (the reference; the kernel is checked "
                   "against it, never the other way round)",
}


def batched_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`BatchedGemm_v1{B,K,N,DOT}(x, w)`: x the B x K bf16 words of one token row (member-major), w the B x N x K words of the
    absorbed weight (``w[b][n][k]``) -> out, B x N bf16 words, member-major."""
    from verity.ml.definitions.numerics import derived_rows as D

    dot_id = prim_static(statics, "DOT", FAMILY)
    try:
        dot = D.dot_bf16_for(dot_id)
    except KeyError:
        raise Declined(f"no registered evaluator for {FAMILY} with DOT={dot_id} (k16 step twins: {sorted(D.TC_DOT16_BY_PRIM)})") from None
    B, K, N = (int(statics[k]) for k in ("B", "K", "N"))
    if len(ops) != 2:
        raise Declined(f"{FAMILY}: the row wires {len(ops)} args, the registered Definition takes 2 (x, w)")
    x, w = np.asarray(ops[0], dtype=np.uint16).reshape(-1), np.asarray(ops[1], dtype=np.uint16).reshape(-1)
    if x.size != B * K or w.size != B * N * K:
        raise Declined(f"{FAMILY}: x has {x.size} words and w {w.size}, the signature has B*K = {B * K} and B*N*K = {B * N * K}")
    acc, nonfin = dot(x.reshape(B, 1, K), w.reshape(B, N, K))
    if nonfin.any():
        raise Declined(f"{FAMILY}: {int(nonfin.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                       "(outside the vectorised k16 twin's domain)")
    out = D.f2fp_bf16(acc).astype(np.uint16).reshape(B * N)
    return {"0": out, "out": out}


def check_targets(rng: np.random.Generator) -> list:
    """Two members, K = 32 (two k16 steps), N = 3, at the Ampere and the Hopper step (the circuit-check binding's statics)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1, cublas_bf16, hopper
    return [bind(cublas_bf16.BatchedGemmWord, B=2, K=32, N=3, DOT=d) for d in (b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16)]


__all__ = ["EVALUATOR", "FAMILY", "PROVENANCE", "batched_gemm", "check_targets"]
