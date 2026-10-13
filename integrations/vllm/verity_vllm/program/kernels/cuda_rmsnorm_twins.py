"""Twin classes for the served non-batch-invariant CUDA RMSNorm Definitions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity_vllm.program.kernels import cuda_rmsnorm_rows as CNR
from verity_vllm.program.registry import cuda_rmsnorm as CR


def families(
    twin_base: type,
    words16: Callable[[Any], np.ndarray],
    rand_bf16: Callable[[np.random.Generator, int, str], np.ndarray],
    random_kinds: tuple[str, ...],
) -> tuple[type, type]:
    """Build the two classes against `twins.Twin` without creating an import cycle."""

    class RMSNormCudaVector8Twin(twin_base):
        twin_id = "rmsnorm.cuda.vector8.v1"
        base_id = "RMSNormCudaVector8_v1"
        covers = ("RMSNormCudaVector8_v1{N,EPS,BLOCK}: vLLM's aligned BF16 width-8 rms_norm_kernel path; each thread visits "
                  "vectors t, t+BLOCK, ... and accumulates one f32 FMA square chain, followed by CUB BlockReduce, rsqrtf, "
                  "two f32 products and RNE to BF16.")

        @classmethod
        def accepts(cls, st: dict) -> str | None:
            try:
                CR._check(type("S", (), st), fused=False)
            except ValueError as e:
                return str(e)
            return None

        def __call__(self, args: list[np.ndarray]) -> np.ndarray:
            N = int(self.s.N)
            x, w = words16(args[0]), words16(args[1])
            if x.size != N or w.size != N:
                raise ValueError(f"{self.spec.id}: operands {x.size}, {w.size} words")
            return CNR.rms_norm_rows(x.reshape(1, N), w, float(self.s.EPS), int(self.s.BLOCK))["out"].reshape(-1).astype(np.uint64)

        @classmethod
        def small_statics(cls, rng: np.random.Generator) -> dict:
            N = int(rng.choice([8, 16, 24, 64, 72, 256]))
            return {"N": N, "EPS": float(rng.choice([1e-5, 1e-6])),
                    "BLOCK": int(rng.integers(1, N // CR.VECTOR + 1))}

        @classmethod
        def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
            N = int(st["N"])
            wk = "gauss" if kind in random_kinds else kind
            return [rand_bf16(rng, N, kind).astype(np.uint64), rand_bf16(rng, N, wk).astype(np.uint64)]

    class RMSNormFusedCudaVector8Twin(RMSNormCudaVector8Twin):
        twin_id = "rmsnorm.fused_cuda.vector8.v1"
        base_id = "RMSNormFusedCudaVector8_v1"
        covers = ("RMSNormFusedCudaVector8_v1{N,EPS,BLOCK}: vLLM's aligned BF16 width-8 fused_add_rms_norm_kernel path; "
                  "native BF16 residual add, four adjacent square-pair sums per vector, CUB BlockReduce, rsqrtf, output and residual.")

        @classmethod
        def accepts(cls, st: dict) -> str | None:
            try:
                CR._check(type("S", (), st), fused=True)
            except ValueError as e:
                return str(e)
            return None

        def __call__(self, args: list[np.ndarray]) -> np.ndarray:
            N = int(self.s.N)
            x, res, w = (words16(a) for a in args)
            if x.size != N or res.size != N or w.size != N:
                raise ValueError(f"{self.spec.id}: operands {x.size}, {res.size}, {w.size} words")
            got = CNR.fused_add_rms_norm_rows(
                x.reshape(1, N), res.reshape(1, N), w, float(self.s.EPS), int(self.s.BLOCK))
            return np.concatenate([got["out"].reshape(-1), got["residual"].reshape(-1)]).astype(np.uint64)

        @classmethod
        def small_statics(cls, rng: np.random.Generator) -> dict:
            N = int(rng.choice([8, 16, 24, 64, 72, 256]))
            return {"N": N, "EPS": float(rng.choice([1e-5, 1e-6])),
                    "BLOCK": int(rng.integers(1, min(N, CR.LARGE_BLOCK) + 1))}

        @classmethod
        def random_args(cls, st: dict, rng: np.random.Generator, kind: str) -> list[np.ndarray]:
            N = int(st["N"])
            wk = "gauss" if kind in random_kinds else kind
            return [rand_bf16(rng, N, kind).astype(np.uint64), rand_bf16(rng, N, kind).astype(np.uint64),
                    rand_bf16(rng, N, wk).astype(np.uint64)]

    return RMSNormCudaVector8Twin, RMSNormFusedCudaVector8Twin


__all__ = ["families"]
