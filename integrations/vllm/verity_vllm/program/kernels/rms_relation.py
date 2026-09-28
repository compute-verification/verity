"""``rmsnorm.bf16.v1`` (Triton path) and ``rmsnorm.cuda.bf16.v1`` (``_C.rms_norm``) -- the *independent*
relations for vLLM's residual-free RMSNorm nodes (W11).

``independent_rmsnorm(node, opened_inputs) -> bytes`` predicts the committed BF16 bytes of one normalised
row from the opened input row ``x`` (BF16), the opened weight ``w`` (BF16) and the description constant
``eps``, with nothing from the runtime: the arithmetic is :mod:`rms_triton_model` (vLLM's batch-invariant
``_rms_norm_kernel`` as compiled for sm_89, instruction by instruction, on the CPU).  It applies to the
nodes W1's adapter records as ``runtime_op == "verity.bi_rms_norm"`` (the Triton kernel: in the pinned
Qwen2.5-1.5B profile only ``model.layers.0.input_layernorm``).

``independent_rmsnorm_cuda`` does the same for ``runtime_op == "_C.rms_norm"`` (``vllm::rms_norm_kernel<bf16, 8>``;
never launched by the pinned engine, but W4b's third rms_norm variant and the kernel vLLM uses without
batch invariance).  The residual-carrying nodes (``_C.fused_add_rms_norm``, W4b's model, not wired) are
``unresolved`` in both.

Coverage (:func:`rms_triton_model.gate`): BF16 x / w / out, weighted, contiguous rows, ``n_cols`` divisible by 16
(Triton) / 8 (CUDA) and inside the measured family (1536 natural; 16 .. 4096 stress), finite eps, sm_89 tables.
Inside it each model was bit-exact on 4,832,256 natural and 3,094,656 stress output words (including inf / NaN /
subnormal operands, eps = 0 and eps < 0), so unlike the attention relation they do *not* decline on non-finite
operands.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from verity.ml.circuits.ops import mufu
from verity_vllm.program.kernels import rms_triton_model as RM
from verity_vllm.program.kernels.fa2_relation import GemmRelationError, _arg_cols, _const, _extra, _node_of, _words

MODEL_VERSION = "rmsnorm-triton-sm89-bf16.v1"
CUDA_MODEL_VERSION = "rmsnorm-cuda-sm89-bf16.v1"
_SQRT_SHA256 = "c5859583cdad12c5f5a6b75061417715a8489870d47216156c2f94b7479a6538"
_RCP_SHA256 = "291d1c9cc1beed5152e3805c279aa0b34d9d15cbe7ece084c6237ca5d44e7ebc"
TABLE_SHA256 = {  # what the MUFU Definitions evaluate through: another table is another Definition version, not a new pin
    "triton": {"sqrt": _SQRT_SHA256, "rcp": _RCP_SHA256},
    "cuda": {"sqrt": _SQRT_SHA256, "rcp": _RCP_SHA256, "rsq": "e9892020427a18c04c9c038775d3c793ff3920e8aee1cb7d1ca202d120fabc45"},
}
TRITON_OP = "verity.bi_rms_norm"
CUDA_OP = "_C.rms_norm"


class RmsRelationError(GemmRelationError):
    """Inputs the rms_norm model cannot be applied to."""


class RmsUnresolved(RmsRelationError):
    """Outside the measured coverage: the checker declines instead of predicting."""


_TABLES: dict[str, RM.RmsTables] = {}


def tables_dir(kernel: str = "triton") -> str:
    """Where both kernels' MUFU.SQRT / MUFU.RCP / MUFU.RSQ delta tables live: core's (``verity.ml.circuits.ops.mufu.TABLE_DIR``)."""
    return mufu.TABLE_DIR


def tables(kernel: str = "triton") -> RM.RmsTables:
    d = tables_dir(kernel)
    if d not in _TABLES:
        if not os.path.isdir(d):
            raise RmsUnresolved(f"unresolved: MUFU.SQRT / MUFU.RCP / MUFU.RSQ tables not found at {d}")
        _TABLES[d] = RM.RmsTables.load(d)
    t = _TABLES[d]
    if kernel == "cuda" and t.rsq is None:
        raise RmsUnresolved(f"unresolved: no MUFU.RSQ table (mufu_rsq_delta_int8.xz) in {d}; the CUDA kernel model needs it")
    got = {name: t.source.get(name, {}).get("sha256") for name in TABLE_SHA256[kernel]}
    if got != TABLE_SHA256[kernel]:
        raise ValueError(f"MUFU tables at {d}: sha256 {got} != pinned {TABLE_SHA256[kernel]}")
    return t


def runtime_op(node_or_inst: Any) -> str | None:
    node = _node_of(node_or_inst)
    op = _extra(node).get("runtime_op")
    if op is None and isinstance(node, Mapping):
        op = node.get("runtime_op")
    return op


def operands(node_or_inst: Any, inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    node = _node_of(node_or_inst)
    x = inputs.get("x")
    w = inputs.get("w", inputs.get("weight"))
    if x is None or w is None:
        raise RmsRelationError("rmsnorm needs opened x (the row) and w (the weight)")
    xw = _words(x)
    cols = _arg_cols(node, "x")
    if cols is not None:
        xw = xw[cols[0] : cols[1]]
    ww = _words(w)
    if xw.size != ww.size:
        raise RmsRelationError(f"x has {xw.size} words but w has {ww.size}")
    eps = _const(node, inputs, "eps")
    if eps is None:
        raise RmsRelationError("rmsnorm node without const 'eps'")
    consts = dict(_extra(node).get("consts") or {})
    return xw, ww, float(eps), consts


_OPS = {"triton": TRITON_OP, "cuda": CUDA_OP}
_KERNEL_NAMES = {"triton": "the Triton _rms_norm_kernel path", "cuda": "the CUDA vllm::rms_norm_kernel path"}


def coverage(node_or_inst: Any, inputs: Mapping[str, Any], kernel: str = "triton") -> dict[str, Any]:
    op = runtime_op(node_or_inst)
    if op is not None and op != _OPS[kernel]:
        return {"covered": False, "reason": f"unresolved: runtime_op {op!r} is not {_KERNEL_NAMES[kernel]} ({_OPS[kernel]})"}
    xw, ww, eps, consts = operands(node_or_inst, inputs)
    x = inputs.get("x")
    # numpy uint16 / int16 arrays and raw bytes are BF16 words by convention (_words); a W2 Value carries its dtype
    x_dtype = "bfloat16" if isinstance(x, (np.ndarray, bytes, bytearray, memoryview)) else str(getattr(x, "dtype", "bfloat16"))
    try:
        RM.gate(int(xw.size), x_dtype, True, True, eps, arch=consts.get("arch"), kernel=kernel)
    except RM.RmsUnresolved as e:
        return {"covered": False, "reason": f"unresolved: {e}"}
    return {"covered": True, "reason": f"measured family ({kernel} kernel, sm_89, n_cols {xw.size}, BF16, weighted, eps {eps})",
            "n_cols": int(xw.size), "kernel": kernel}


def _predict(node_or_inst: Any, inputs: Mapping[str, Any], kernel: str) -> bytes:
    cov = coverage(node_or_inst, inputs, kernel)
    if not cov["covered"]:
        raise RmsUnresolved(cov["reason"])
    xw, ww, eps, _ = operands(node_or_inst, inputs)
    out = RM.rms_norm_rows(xw[None], ww, eps, tables(kernel), kernel=kernel)
    return out.reshape(-1).astype("<u2").tobytes()


def independent_rmsnorm(node_or_inst: Any, inputs: Mapping[str, Any]) -> bytes:
    """``rmsnorm.bf16.v1``: the Triton ``_rms_norm_kernel`` (runtime_op ``verity.bi_rms_norm``)."""
    return _predict(node_or_inst, inputs, "triton")


def independent_rmsnorm_cuda(node_or_inst: Any, inputs: Mapping[str, Any]) -> bytes:
    """``rmsnorm.cuda.bf16.v1``: ``vllm::rms_norm_kernel<bf16, 8>`` (runtime_op ``_C.rms_norm``)."""
    return _predict(node_or_inst, inputs, "cuda")


RELATION_IDS = {
    "rmsnorm.bf16.v1": ("rmsnorm", independent_rmsnorm, MODEL_VERSION,
                        "Independent CPU model of vLLM's batch-invariant Triton _rms_norm_kernel (runtime_op verity.bi_rms_norm) as compiled "
                        "for sm_89: per-thread FFMA chains over 8 columns, SHFL.BFLY warp / CTA trees, FFMA(MUFU.RCP(n), sum, eps), MUFU.SQRT, "
                        "MUFU.RCP, two FMULs, RNE to BF16.  Bit-exact on 4.83M natural + 3.09M stress output words on the RTX 4090.  "
                        "Coverage-gated (BF16, weighted, contiguous, n_cols % 16 == 0 in the measured family, sm_89), else "
                        "RelationError('unresolved: ...'); the fused_add_rms_norm / _C.rms_norm nodes are unresolved here."),
    "rmsnorm.cuda.bf16.v1": ("rmsnorm", independent_rmsnorm_cuda, CUDA_MODEL_VERSION,
                             "Independent CPU model of vLLM's CUDA vllm::rms_norm_kernel<bf16, VEC_SIZE=8, HasWeight=true> (runtime_op _C.rms_norm) "
                             "as compiled for sm_89: one 8-vector per thread accumulated by an FFMA chain in element order, cub WarpReduce shfl_down "
                             "tree (offsets 1,2,4,8,16 -- lane 0's value), cub BlockReduce<1024>'s warp aggregates added sequentially by warp 0, "
                             "IEEE FP32 division by n_cols, FADD eps, MUFU.RSQ (rsqrt.approx.f32, 2^24 / 2^12 range scaling), out = bf16((x * s) * w).  "
                             "Bit-exact on 4.83M natural-input + 3.09M stress output words on the RTX 4090 (the pinned engine never launches this "
                             "kernel; the natural rows are the Triton fixture's rows replayed through torch.ops._C.rms_norm).  Coverage-gated (BF16, "
                             "weighted, contiguous, n_cols % 8 == 0 in the measured family, sm_89), else RelationError('unresolved: ...')."),
}
RELATIONS: dict[str, Any] = {rid: fn for rid, (_, fn, _, _) in RELATION_IDS.items()}


def register_with_w2(replace: bool = False) -> list[str]:
    try:
        from verity_vllm.program.kernels import relations as R
    except ImportError:
        return []
    done = []
    for rid, (op_type, fn, version, doc) in RELATION_IDS.items():
        existing = R.REGISTRY.get(rid)
        if existing is not None and existing.implemented and not replace:
            continue

        def _wrap(inst, inputs, _fn=fn):
            try:
                return _fn(inst, inputs)
            except GemmRelationError as error:
                raise R.RelationError(str(error)) from error

        R.register(rid, klass=R.INDEPENDENT, op_type=op_type, includes_full_op=True, version=version, doc=doc, owner="W11", replace=replace)(_wrap)
        done.append(rid)
    return done


REGISTERED_WITH_W2 = register_with_w2()
