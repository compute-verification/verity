"""The row kernels' pieces for the Definitions whose program values are inputs (`TargetProfile.registered_values`):
`RMSNormTriton_v3`, `RMSNormFusedCuda_v4`, `Attention_v10`, `Attention_v11` (`registry.value_inputs`, `registry.fa2_check_inf`).

Each is its constant twin's row (`rows.rmsnorm_triton`, `rows.rmsnorm_fused`, `rows.attention`) with the value read from the f32 word
of the last operand; the self-check draws that word in the value's served range.  The rows register from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from verity_vllm.program.kernels.kernel_registry import Declined

SCALE_INPUT = ("Attention_v10", "Attention_v11")
EVALUATORS = {
    "Attention_v10": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v10{T,NH,KVH,D,BN}: Attention_v3's evaluator with "
                     "scale_log2 the f32 word of its last operand)",
    "Attention_v11": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v11{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}: Attention_v5's "
                     "evaluator with scale_log2 the f32 word of its last operand)",
    "RMSNormTriton_v3": "verity_vllm.program.kernels.derived_rows.rmsnorm_triton_rows (eps the f32 word of the last operand)",
    "RMSNormFusedCuda_v4": "verity_vllm.program.kernels.derived_rows.rmsnorm_fused_v2_rows (eps the f32 word of the last operand)",
}


def word32(op: np.ndarray, family: str, name: str) -> int:
    w = np.asarray(op).reshape(-1)
    if w.size != 1:
        raise Declined(f"{family}: {name} is {w.size} words, the Definition reads one f32 word")
    return int(w[0]) & 0xFFFFFFFF


def eps(ops: Sequence[np.ndarray], n: int, family: str) -> float:
    """The eps of a norm that reads it as the operand after the `n` its constant twin takes."""
    if len(ops) != n + 1:
        raise Declined(f"{family}: the row wires {len(ops)} args, the registered Definition takes {n + 1} (eps last)")
    return float(np.uint32(word32(ops[n], family, "eps")).view(np.float32))


def eps_words(rng: np.random.Generator, n: int, **statics: Any) -> np.ndarray:
    """A norm's eps: the served 1e-5 / 1e-6 on half the rows, else drawn log-uniform over [1e-8, 1e-2]."""
    e = np.where(rng.random(n) < 0.5, rng.choice([1e-5, 1e-6], n), 10.0 ** rng.uniform(-8, -2, n))
    return np.asarray(e, dtype=np.float32).view(np.uint32)


def scale_words(rng: np.random.Generator, n: int, **statics: Any) -> np.ndarray:
    """The attention scale: D^-0.5 * log2(e) of the row's D on half the rows, else drawn over [0.02, 0.5]."""
    from verity_vllm.program.registry import b1
    s = np.asarray(rng.uniform(0.02, 0.5, n), dtype=np.float32).view(np.uint32)
    return np.where(rng.random(n) < 0.5, np.uint32(b1._scale_log2_bits(int(statics["D"]))), s).astype(np.uint32)


def sample_value(rest: Callable[..., list[np.ndarray]], value: Callable[..., np.ndarray]) -> Callable[..., list[np.ndarray]]:
    """`rest` for every parameter but the last, a program value in its served range for that one."""
    def sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
        return [*rest(rng, n, params[:-1], **statics), value(rng, n, **statics)]
    return sample


def check_targets() -> dict[str, list]:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import fa2_check_inf, hopper, value_inputs as VI
    from verity_vllm.program.registry import prims as P

    attn = {"T": 20, "NH": 2, "KVH": 1, "D": 16, "BN": 16}
    return {
        "Attention_v10": [bind(VI.AttentionV10, **attn)],
        "Attention_v11": [bind(fa2_check_inf.AttentionV11, **dict(attn, T=40), DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum, MASKED_FROM=m)
                          for m in (2, 0)],
        "RMSNormTriton_v3": [bind(VI.RMSNormTritonV3, N=16)],
        "RMSNormFusedCuda_v4": [bind(VI.RMSNormFusedCudaV4, N=16)],
    }


__all__ = ["EVALUATORS", "SCALE_INPUT", "check_targets", "eps", "eps_words", "sample_value", "scale_words", "word32"]
