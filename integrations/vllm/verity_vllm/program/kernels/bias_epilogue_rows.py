"""The vectorised row of `GemmBiasF32Epilogue_v1{K,N,DOT}` (`verity_vllm.program.registry.gemm_bias`), registered in `rows`."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity_vllm.program.kernels.kernel_registry import Declined


def gemm_bias_f32_epilogue(statics: dict[str, Any], ops: Sequence[np.ndarray], dot_id: str, chunk: int) -> dict[str, np.ndarray]:
    """`GemmBiasF32Epilogue_v1{K,N,DOT}(x, W, b)` with the DOT static resolved to `dot_id`: per coordinate the f32 chain of that k16
    step's twin, plus the bias widened to f32 (one rounding), then f2fp_bf16, over `chunk` coordinates at a time.  An unregistered DOT
    or a non-finite lane is declined by name, as for `Gemm_v2`."""
    from verity_vllm.program.kernels import derived_rows as D
    family = "GemmBiasF32Epilogue_v1"
    if len(ops) != 3:
        raise ValueError(f"{family} takes 3 operands (x, W, b), the row wires {len(ops)}")
    try:
        dot = D.dot_bf16_for(dot_id)
    except KeyError:
        raise Declined(f"no registered evaluator for {family} with DOT={dot_id} (k16 step twins: {sorted(D.TC_DOT16_BY_PRIM)})") from None
    K, N = int(statics["K"]), int(statics["N"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    w = np.asarray(ops[1], dtype=np.uint16).reshape(N, K)
    b = D.as_f32(D.bf16_to_f32_words(np.asarray(ops[2], dtype=np.uint16).reshape(N)))
    out = np.empty(N, dtype=np.uint16)
    for n0 in range(0, N, chunk):
        acc, nf = dot(x[None, :], w[n0:n0 + chunk])
        if nf.any():
            raise Declined(f"{family}: {int(nf.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                           "(outside the vectorised k16 twin's domain)")
        s = (D.as_f32(acc).reshape(-1) + b[n0:n0 + chunk]).astype(np.float32)
        out[n0:n0 + chunk] = D.f2fp_bf16(D.as_u32(s)).astype(np.uint16)
    return {"0": out, "out": out}
