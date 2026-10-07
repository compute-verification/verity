"""The replay row kernel of vLLM `_C.silu_and_mul` as the served vocabulary binds it, `SiluMul_v2{I}` (`registry.act`).

`silu_mul_expf_rows` is `derived_rows.silu_mul_rows`'s shape over `SiluMulBf16_v2`: the bf16 silu through its own 2^16 table
(`silu_expf_mid_table`, built from `registry.act.silu_f32_bits`: gates past expf's overflow and gate -0 are -0, NaN and -inf 0x7FFF), the
up word through `+ 0.0f` (-0 to +0), and the product's `cvt.rn.bf16.f32` with NaN 0x7FFF.  The row kernel registers from `rows`
(kernel "rows").
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

FAMILY = "SiluMul_v2"
EVALUATOR = ("verity_vllm.program.kernels.act_rows.silu_mul_expf_rows (SiluMul_v2{I}: SiluMulBf16_v2 per element -- expf's overflow, "
             "up + 0.0f, NaN 0x7FFF; registry.act)")


@lru_cache(maxsize=None)
def silu_expf_mid_table() -> np.ndarray:
    """`cvt.rn.bf16.f32(f / (1.0f + expf(-f)))` on each of the 65,536 bf16 gate words, `SiluMulBf16_v2`'s interior."""
    from verity_vllm.program.registry import act
    from verity_vllm.program.registry import prims as P
    return np.array([P.f32_to_bf16_rn_bits(act.silu_f32_bits(g << 16), P.BF16_NAN_F2FP) for g in range(1 << 16)], dtype=np.uint16)


def silu_mul_expf_rows(gu: np.ndarray) -> dict[str, np.ndarray]:
    """`SiluMul_v2{I}` over rows: gu (R, 2I) bf16 words -> the bf16 ``mid`` (R, I) = silu(gate) and the leaf ``out`` (R, I)."""
    from verity_vllm.program.kernels import derived_rows as D
    gu = np.ascontiguousarray(gu, dtype=np.uint16)
    half = gu.shape[1] // 2
    g, u = gu[:, :half], gu[:, half:]
    mid = silu_expf_mid_table()[g]
    uf = np.where(u == 0x8000, np.uint32(0), D.bf16_to_f32_words(u))
    return {"mid": mid, "out": D.f2fp_bf16(D.mul_rn(D.bf16_to_f32_words(mid), uf))}


def silu_mul_expf(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    I = int(statics["I"])
    r = silu_mul_expf_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, 2 * I))
    return {"out": np.asarray(r["out"], dtype=np.uint16).reshape(I)}


def check_targets(rng) -> list:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import act
    return [bind(act.SiluMul, I=8)]


__all__ = ["EVALUATOR", "FAMILY", "check_targets", "silu_expf_mid_table", "silu_mul_expf", "silu_mul_expf_rows"]
