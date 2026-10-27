"""The replay row kernel of the dense elementwise / reduction Definitions (`registry.dense`): the unfused ATen norm chain Gemma-2's
`GemmaRMSNorm` runs (`(1 + weight)`, square, mean, `+ eps`, rsqrt, scale, multiply, narrow; the residual add), `gelu_pytorch_tanh`'s
`GeluTanhMul` and the soft-capped logits (divide, tanh, multiply).

One row is one instance of the row's Definition, evaluated by `verity.ml.circuits.evaluation.evaluate_batch` with the family's `dense_rows` kernel
where one is registered (the IR's reference for what it declines), else the IR's reference -- never "auto", which would pick this
kernel for itself.  So the words are the Definition's, NaN payloads included, with no arithmetic of its own.  An instance past
`MAX_WORDS` argument words is declined (the reference holds it as Python ints).  Registered from `rows` (kernel "rows").
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from verity.ml.definitions.numerics.kernel_registry import Declined

MAX_WORDS = 1 << 18
EVALUATOR = ("verity.ml.definitions.numerics.ir_rows.ir_row (the row's registry.dense Definition by verity.ml.circuits.evaluation.evaluate_batch: "
             "its dense_rows kernel, else the IR reference)")
_WORD = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


def _definitions() -> dict[str, Any]:
    from verity.ml.definitions.registry import dense
    return {d.id: d for d in (
        dense.SquareBf16, dense.SquareF32, dense.MeanTriton, dense.AddScalarF32, dense.AddScalarBf16, dense.AddWidenedBf16, dense.RsqrtF32,
        dense.ScaleRowBf16, dense.ScaleRowF32, dense.MulVecF32, dense.NarrowF32ToBf16, dense.GeluTanhMul, dense.Bf16DivScalar,
        dense.Bf16Tanh, dense.Bf16MulScalar)}


FAMILIES = ("SquareBf16_v1", "SquareF32_v1", "MeanTriton_v1", "AddScalarF32_v1", "AddScalarBf16_v1", "AddWidenedBf16_v1", "RsqrtF32_v1",
            "ScaleRowBf16_v1", "ScaleRowF32_v1", "MulVecF32_v1", "NarrowF32ToBf16_v1", "GeluTanhMul_v1", "Bf16DivScalar_v1", "Bf16Tanh_v1",
            "Bf16MulScalar_v1")


def f64_statics(statics: dict[str, Any]) -> dict[str, Any]:
    """`statics` with every f64 literal the spec text spells `{"f64":"<hex>"}` (as `manifest.format.spec_statics` keeps it) as its float."""
    def one(v: Any) -> Any:
        if isinstance(v, str) and v.startswith("{"):
            v = json.loads(v)
        return float.fromhex(str(v["f64"])) if isinstance(v, dict) and set(v) == {"f64"} else v
    return {k: one(v) for k, v in statics.items()}


def ir_row(family: str, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """One ROW of `family{statics}`: arg g = the instance's parameter g, as words.  Returns {"0": words, "out": words} (a module's
    output is spelt `0` when the row is the module's return, `out` inside its body)."""
    from verity.ml.circuits.evaluation import evaluate_batch, kernels
    from verity.ml.circuits.evaluation.batch import type_dtype
    from verity.ml.circuits.evaluation.reference import signature_of
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics.dense_rows import KERNEL as DENSE

    d = bind(_definitions()[family], **f64_statics(statics))
    params, ret = signature_of(d)
    if len(ops) != len(params):
        raise Declined(f"{family}: the row wires {len(ops)} args, the Definition takes {len(params)}")
    arrays = [np.asarray(ops[g]).reshape(1, -1) for g in range(len(params))]
    if any(a.shape[1] != pt.leaves for a, pt in zip(arrays, params)):
        raise Declined(f"{family}: arg words {[a.shape[1] for a in arrays]} != the Definition's {[pt.leaves for pt in params]}")
    if sum(pt.leaves for pt in params) > MAX_WORDS:
        raise Declined(f"{family}: {sum(pt.leaves for pt in params)} argument words per instance, past {MAX_WORDS}")
    kernel = DENSE if DENSE in kernels(d) else "reference"
    out = evaluate_batch(d, *(a.astype(type_dtype(pt), copy=False) for a, pt in zip(arrays, params)), kernel=kernel).reshape(-1)
    out = out.astype(_WORD.get(int(ret.homogeneous_width() or 64), np.uint64))
    return {"0": out, "out": out}


def check_targets(family: str) -> list:
    """Self-check specialisations: 16 words (I = 16), a constant C of 1.0 where the Definition takes one."""
    from verity.ml.circuits.defs import bind
    d = _definitions()[family]
    return [bind(d, **{k: v for k, v in (("N", 16), ("I", 16), ("C", 1.0)) if k in d.statics})]


__all__ = ["EVALUATOR", "FAMILIES", "MAX_WORDS", "check_targets", "f64_statics", "ir_row"]
