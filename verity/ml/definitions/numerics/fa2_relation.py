"""``attention.bf16.v1`` -- the *independent* relation for vLLM's FlashAttention-2 node.

``independent_attention(node, opened_inputs) -> bytes`` predicts the committed BF16 bytes of one
attention output row (all ``num_heads`` heads x ``head_size``) at position ``t`` from the opened
query row ``q`` and the opened, authenticated K and V rows ``0..t`` (post-RoPE K, the V slice of the
qkv projection), the description constants (``scale``, ``num_heads``, ``num_kv_heads``, ``head_size``,
``block_size``, ``fa_version``, ``num_splits``, ``sliding_window``, ``logits_soft_cap``,
``kv_cache_dtype``) and nothing from the runtime: the arithmetic is :mod:`fa2_model` (the FA2
``flash_fwd_splitkv_kernel`` for BF16 hdim-128 on SM8x, instruction by instruction, on the CPU).

Coverage (:func:`fa2_model.coverage`): the relation predicts only inside the family that was measured
bit-exact -- sm_89, head_dim 128, (12, 2) heads, block_size 16, FA2, num_splits 1, no window / softcap
/ alibi, BF16 KV, 1 <= t+1 <= 4096, finite operands -- and raises ``RelationError('unresolved: ...')``
everywhere else.  A row whose arithmetic meets a non-finite intermediate (FP32 overflow, inf, NaN) is
also ``unresolved``: the shared MMA model saturates instead of following the hardware's inf/NaN
rules, which were not measured (production writes 0x7FFF for the whole head in the two cases run).

A prefill row is checked exactly like a decode row (one query, keys ``0..t``): the kernel's extra
fully-masked blocks for rows inside a 64-row M-block are arithmetic no-ops (P = 0, l = 0, O = 0, the
first real block rescales by ex2(-inf) = 0) -- see ``tests/test_attention_model.py``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from verity.ml.circuits.ops import mufu
from verity.ml.definitions.numerics import fa2_model as FA
from verity.ml.definitions.numerics.gemm_relation import GemmRelationError, _arg_cols, _const, _extra, _node_of, _words

MODEL_VERSION = "fa2-sm8x-hdim128-bf16.v1"
TABLE_SHA256 = {  # what the MUFU Definitions evaluate through: another table is another Definition version, not a new pin
    "ex2": "7b3249116f9ca84455389f3a9b116da2be327646fb4b544369314ee168b79d88",
    "rcp": "291d1c9cc1beed5152e3805c279aa0b34d9d15cbe7ece084c6237ca5d44e7ebc",
}


class AttentionRelationError(GemmRelationError):
    """Inputs the attention model cannot be applied to."""


class AttentionUnresolved(AttentionRelationError):
    """Outside the measured coverage: the checker declines instead of predicting."""


_TABLES: dict[str, FA.MufuTables] = {}


def tables_dir() -> str:
    """Where the MUFU.EX2 / MUFU.RCP delta tables live: core's (``verity.ml.circuits.ops.mufu.TABLE_DIR``), pinned by :data:`TABLE_SHA256`."""
    return mufu.TABLE_DIR


def tables() -> FA.MufuTables:
    d = tables_dir()
    if d not in _TABLES:
        if not os.path.isdir(d):
            raise AttentionUnresolved(f"unresolved: MUFU tables not found at {d}")
        _TABLES[d] = FA.MufuTables.load(d)
    t = _TABLES[d]
    got = {name: t.source.get(name, {}).get("sha256") for name in TABLE_SHA256}
    if got != TABLE_SHA256:
        raise ValueError(f"MUFU tables at {d}: sha256 {got} != pinned {TABLE_SHA256}")
    return t


# --- input adaptation --------------------------------------------------------------------------------

Q_NAMES = ("q", "query")
K_NAMES = ("k", "key")
V_NAMES = ("v", "value")


def _first(inputs: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for n in names:
        if n in inputs:
            return inputs[n]
    return None


def _sliced(node: Any, name: str, value: Any) -> np.ndarray:
    w = _words(value)
    cols = _arg_cols(node, name)
    if cols is not None:
        w = w[cols[0] : cols[1]]
    return w


def operands(node_or_inst: Any, inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(q (H, D) words, k (n, KVH, D) words, v (n, KVH, D) words, consts)."""
    node = _node_of(node_or_inst)
    consts = dict(_extra(node).get("consts") or {})
    for name in ("num_heads", "num_kv_heads", "head_size", "block_size", "fa_version", "num_splits", "sliding_window",
                 "logits_soft_cap", "kv_cache_dtype", "scale", "arch", "gpu", "gpu_name"):
        v = _const(node, inputs, name)
        if v is not None and name not in consts:
            consts[name] = v
    for need in ("num_heads", "num_kv_heads", "head_size", "scale"):
        if need not in consts:
            raise AttentionRelationError(f"attention node without const {need!r}")
    H, KVH, D = int(consts["num_heads"]), int(consts["num_kv_heads"]), int(consts["head_size"])
    qname = next((n for n in Q_NAMES if n in inputs), "q")
    kname = next((n for n in K_NAMES if n in inputs), "k")
    vname = next((n for n in V_NAMES if n in inputs), "v")
    q = _first(inputs, Q_NAMES)
    ks = _first(inputs, K_NAMES)
    vs = _first(inputs, V_NAMES)
    if q is None or ks is None or vs is None:
        raise AttentionRelationError("attention needs opened q, k (rows 0..t) and v (rows 0..t)")
    qw = _sliced(node, qname, q)
    if qw.size != H * D:
        raise AttentionRelationError(f"q has {qw.size} words, expected num_heads*head_size = {H * D}")
    if isinstance(ks, np.ndarray) and ks.ndim >= 2:
        kw = np.ascontiguousarray(ks).view(np.uint16).reshape(ks.shape[0], -1)
    else:
        if not isinstance(ks, (list, tuple)) or not ks:
            raise AttentionRelationError("k: expected a list of opened rows 0..t")
        kw = np.stack([_sliced(node, kname, x) for x in ks], 0)
    if isinstance(vs, np.ndarray) and vs.ndim >= 2:
        vw = np.ascontiguousarray(vs).view(np.uint16).reshape(vs.shape[0], -1)
    else:
        if not isinstance(vs, (list, tuple)) or not vs:
            raise AttentionRelationError("v: expected a list of opened rows 0..t")
        vw = np.stack([_sliced(node, vname, x) for x in vs], 0)
    n = kw.shape[0]
    if vw.shape[0] != n:
        raise AttentionRelationError(f"{n} K rows but {vw.shape[0]} V rows")
    if kw.shape[1] != KVH * D or vw.shape[1] != KVH * D:
        raise AttentionRelationError(f"K/V rows have {kw.shape[1]}/{vw.shape[1]} words, expected num_kv_heads*head_size = {KVH * D}")
    t = getattr(node_or_inst, "t", None)
    if t is not None and int(t) + 1 != n:
        raise AttentionRelationError(f"{n} K/V rows for t={t} (need t+1)")
    return qw.reshape(H, D), kw.reshape(n, KVH, D), vw.reshape(n, KVH, D), consts


def coverage(node_or_inst: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """``{"covered": bool, "reason": str, "n": int}`` without running the model."""
    q, k, v, consts = operands(node_or_inst, inputs)
    H, KVH, D = int(consts["num_heads"]), int(consts["num_kv_heads"]), int(consts["head_size"])
    n = k.shape[0]
    c = dict(consts)
    c.setdefault("window_size", c.get("sliding_window"))
    c.setdefault("softcap", c.get("logits_soft_cap", 0))
    reason = FA.coverage(1, n, H, KVH, D, c)
    if reason is None:
        nf = FA.nonfinite_words(q, k, v)
        if nf:
            reason = f"{nf} non-finite operand word(s): inf/NaN semantics not modelled"
    return {"covered": reason is None, "reason": ("unresolved: " + reason) if reason else f"measured family ({FA.MEASURED['arch'][0]}, hdim {D}, ({H},{KVH}) heads, n={n})", "n": n}


def model_row(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> FA.RowsResult:
    """The arithmetic without the gate: q (H, D), k/v (n, KVH, D) words -> RowsResult (out (1, H, D))."""
    n = k.shape[0]
    return FA.attention_call(q[None], k, v, np.array([0]), 1, n, float(scale), tables())


def independent_attention(node_or_inst: Any, inputs: Mapping[str, Any]) -> bytes:
    """Predicted BF16 bytes of the attention output row (num_heads x head_size) at t from q and K/V 0..t."""
    cov = coverage(node_or_inst, inputs)
    if not cov["covered"]:
        raise AttentionUnresolved(cov["reason"])
    q, k, v, consts = operands(node_or_inst, inputs)
    res = model_row(q, k, v, float(consts["scale"]))
    if res.nonfinite.any():
        raise AttentionUnresolved("unresolved: non-finite intermediate (FP32 overflow / inf / NaN) in "
                                  f"{int(res.nonfinite.sum())} head(s): inf/NaN semantics not modelled")
    return res.out.reshape(-1).astype("<u2").tobytes()


RELATION_IDS = {
    "attention.bf16.v1": ("attention", independent_attention,
                          "Independent CPU model of vLLM's FlashAttention-2 kernel (flash_fwd_splitkv_kernel<128,64,64,4,bf16>, "
                          "vllm-flash-attn 506341a, --use_fast_math) as compiled: Ampere mma.sync BF16 chains for QK^T and PV, "
                          "the online softmax with MUFU.EX2/MUFU.RCP tables, FTZ FP32 with the compiled FFMA contraction of the "
                          "row-sum rescale, RNE to BF16.  Bit-exact on 4.3M natural + 4.6M stress output words on the RTX 4090.  "
                          "Coverage-gated (sm_89, hdim 128, (12,2) heads, block 16, FA2, 1 split, no window/softcap, BF16 KV, "
                          "t+1 <= 4096, finite operands), else RelationError('unresolved: ...')."),
}
RELATIONS: dict[str, Any] = {rid: fn for rid, (_, fn, _) in RELATION_IDS.items()}
