"""The replay row kernel of the exact-int8 linear's quantizer, `Int8RowQuant_v1{K}` (`registry.int8_exact`, served as
``verity_int8::row_quant``, `engine.int8_exact.quantize_rows`).

On one bf16 row x of K words: s = RN32(fmaxf(max_k |f32(x_k)|, RN32(1e-30)) / 127) (`Int8RowScale`, an exact max), and each code
s8(fminf(fmaxf(rint(RN32(f32(x_k) / s)), -127), 127)) (`Int8QuantElement`: ties to even, the clamp, `cvt.rzi`), the codes as the
Definition's port 0 and s as port 1.  A finite row's quotients are within 127 (1 + 2^-23) of zero, so neither NaN nor a signed
zero reaches a code; a non-finite word is declined before the kernel (`rows._instance`).  Registers from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

FAMILY = "Int8RowQuant_v1"
EVALUATOR = ("verity_vllm.program.kernels.int8_rows.int8_row_quant_rows (Int8RowQuant_v1{K}: the row's f32 |x| max from +0 floored at "
             "RN32(1e-30), one div.rn by 127, then rint(x / s) clamped to [-127, 127] per code; registry.int8_exact)")
TOP = np.float32(127.0)
FLOOR = np.float32(1e-30)


def int8_row_quant_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (R, K) bf16 words -> (codes (R, K) s8 bytes, s (R,) f32 words)."""
    from verity_vllm.program.kernels import derived_rows as D
    xf = D.as_f32(D.bf16_to_f32_words(np.ascontiguousarray(x, dtype=np.uint16)))
    m = np.maximum(np.max(np.abs(xf), axis=-1, initial=np.float32(0.0)), FLOOR).astype(np.float32)
    s = (m / TOP).astype(np.float32)
    q = np.clip(np.rint((xf / s[..., None]).astype(np.float32)), -TOP, TOP)
    return q.astype(np.int8).view(np.uint8), D.as_u32(s)


def int8_row_quant(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    K = int(statics["K"])
    codes, s = int8_row_quant_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, K))
    return {"0": codes.reshape(K), "1": s.reshape(1)}


def check_targets(rng) -> list:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import int8_exact
    return [bind(int8_exact.Int8RowQuant, K=k) for k in (1, 8, 33)]


__all__ = ["EVALUATOR", "FAMILY", "check_targets", "int8_row_quant", "int8_row_quant_rows"]
