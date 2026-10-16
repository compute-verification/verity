"""The replay row kernel of MLA's latent absorptions, `BatchedGemm_v1{B,K,N,DOT}` (`registry.cublas_bf16`): ``torch.bmm`` on cuBLAS
against `W_UK_T` / `W_UV`, one token row over all B heads.  Member b is `Gemm_v2{K,N,DOT}`'s row of x[b] against W[b] (``w[b][n][k]``):
the ascending k16 chain of the DOT static's step from +0, then f2fp_bf16.  An unregistered DOT, and a coordinate with a non-finite
operand or accumulator, declines the row by name, as `rows.gemm_dot` does.  The row kernel registers from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity_vllm.program.kernels.kernel_registry import Declined, prim_static

FAMILY = "BatchedGemm_v1"
EVALUATOR = ("verity_vllm.program.kernels.batched_gemm_rows.batched_gemm (BatchedGemm_v1{B,K,N,DOT}: per member b, derived_rows."
             "dot_bf16_for(DOT) of x[b] against W[b] -- the ascending k16 chain from +0 -- then f2fp_bf16, Gemm_v2{K,N,DOT}'s row; "
             "cuBLAS's batched bf16 GEMM of the MLA absorbs, bf16_gemm_check.bmm_rows)")


def batched_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`BatchedGemm_v1{B,K,N,DOT}(x, w)`: x the B x K bf16 words of one token row (member-major), w the B x N x K words of the
    absorbed weight (``w[b][n][k]``) -> out, B x N bf16 words, member-major."""
    from verity_vllm.program.kernels import derived_rows as D

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
    from verity_vllm.program.registry import b1, cublas_bf16, hopper
    return [bind(cublas_bf16.BatchedGemmWord, B=2, K=32, N=3, DOT=d) for d in (b1.P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16)]


__all__ = ["EVALUATOR", "FAMILY", "batched_gemm", "check_targets"]
