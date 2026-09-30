"""The vectorised row of `Fp8GroupQuantPacked_v1{K,G}` (`verity_vllm.program.registry.fp8_packed`), a replay row of `rows`: the
`derived_rows` twin of `Fp8GroupQuant_v1` with the UE8M0 ceiling, the multiply by the reciprocal of the power of two, and the packing."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity_vllm.program.kernels.derived_rows import as_f32, as_u32, bf16_to_f32_words, div_rn, mul_rn


def fp8_group_quant_packed_row(x: np.ndarray, *, G: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """One bf16 row (K,) -> (x_q (K,) e4m3 bytes, the row's (ceil(K/G/4),) packed UE8M0 scale words)."""
    from verity_vllm.program.registry.fp8 import np_f32_to_e4m3_sat
    x = np.asarray(x, dtype=np.uint16).reshape(-1)
    K = x.size
    if K % G:
        raise ValueError(f"Fp8GroupQuantPacked: K={K} is not a multiple of G={G}")
    ng = K // G
    xf = as_f32(bf16_to_f32_words(x)).reshape(ng, G)
    ax = np.abs(xf)
    m = np.max(np.where(np.isnan(ax), np.float32(-np.inf), ax), axis=1, initial=np.float32(1e-10)).astype(np.float32)
    y = np.maximum(as_f32(div_rn(as_u32(m), as_u32(np.full(ng, np.float32(448.0))))), np.float32(1e-10))
    u = as_u32(y)
    e = ((((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0)) & 0xFF).astype(np.uint32)
    inv = div_rn(as_u32(np.ones(ng, np.float32)), (e << 23).astype(np.uint32))
    q = as_f32(mul_rn(as_u32(xf), np.broadcast_to(inv[:, None], (ng, G))))
    q = np.minimum(np.where(np.isnan(q), np.float32(-448.0), np.maximum(q, np.float32(-448.0))), np.float32(448.0)).astype(np.float32)
    b = np.zeros(-(-ng // 4) * 4, dtype=np.uint32)
    b[:ng] = e
    words = (b[0::4] | (b[1::4] << 8) | (b[2::4] << 16) | (b[3::4] << 24)).astype(np.uint32)
    return np_f32_to_e4m3_sat(q).reshape(K), words


def fp8_group_quant_packed(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """The replay row: port 0 the K e4m3 words, port 1 the packed scale words, in the Definition's return order."""
    K, G = int(statics["K"]), int(statics["G"])
    xq, words = fp8_group_quant_packed_row(np.asarray(ops[0], dtype=np.uint16).reshape(K), G=G)
    return {"0": np.asarray(xq, dtype=np.uint8), "1": words}


__all__ = ["fp8_group_quant_packed", "fp8_group_quant_packed_row"]
