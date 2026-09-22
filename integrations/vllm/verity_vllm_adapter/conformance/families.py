"""Vocabularies the port files are validated against (fail closed on anything outside them).

`KNOWN_FAMILIES` mirrors the adapter's op-family names (`verity_vllm_adapter.capture.OP_FAMILY` values, the
`family` field of every `description.json` node, plus the families the capture assigns outside that table:
`expert` (W9's per-slot expert output), `sampler`, `cast`, `kv_write`). `capture.py` imports torch, so the
CPU-only harness keeps this copy; `tests/test_conformance.py` cross-checks it against `OP_FAMILY` whenever
torch is importable.

`EXPECTATION_STATUSES` are the per-family statuses a port may declare (contract Part 2, `expectations.json`);
`BINDING_STATUSES` are the conformance statuses a registered primitive/composite may carry
(`verity.ir.defs.*Definition.conformance`, `typed-b1.md` §3).
"""

from __future__ import annotations

# family -> op_type (the pair the adapter records per node)
KNOWN_FAMILIES: dict[str, str] = {
    "gemm": "gemm",  # verity.bi_mm / aten.mm / aten.addmm / aten.matmul / aten.linear / aten.bmm; also lm_head
    "embedding": "embedding",
    "rmsnorm": "rmsnorm",  # verity.bi_rms_norm / _C.rms_norm / _C.fused_add_rms_norm / inductor fused
    "rope": "rope",
    "act_mul": "silu_mul",
    "attn": "attention",
    "kv_write": "other",
    "add": "residual_add",
    "gather": "other",
    "router": "router",  # _moe_C.topk_softmax (ids port 0; weights port 1 under its own node)
    "combine": "combine",  # _moe_C.moe_sum
    "sigmoid": "other",
    "mul": "other",
    "expert": "expert",  # fused_moe_kernel per-(token, slot) output, `e=<slot>`
    "sampler": "sampler",
    "cast": "other",
    # tensor parallelism (lane TP): the collectives are ordinary functions of the ranks' captured operands
    "all_reduce": "allreduce",  # vllm.all_reduce (row-parallel o_proj / down_proj, vocab-parallel embedding): allreduce.bf16.v1
    "all_gather": "other",  # vllm.all_gather (lm_head logits): allgather.copy.v1
    # [QZ, additive] FP8 linear (port b1-fp8): _C.dynamic_per_token_scaled_fp8_quant (port 0 = e4m3 row, port 1 = f32
    # row scale); the scaled GEMM itself (_C.cutlass_scaled_mm) is recorded under `gemm`
    "fp8_quant": "other",
}

# op_types that appear in descriptions under a family (lm_head is family gemm, op_type lm_head)
KNOWN_OP_TYPES = frozenset(set(KNOWN_FAMILIES.values()) | {"lm_head"})

EXPECTATION_STATUSES = ("supported-exact", "same-runtime", "unsupported")
UNSUPPORTED_KINDS = ("opaque", "unresolved", "not-modelled")

BINDING_STATUSES = ("exact-model-tested", "same-runtime-only", "model-untested", "not-yet-established")

# relation classes the bundle's description records (`verity_vllm.relations.CLASSES`)
RELATION_CLASS_OF_STATUS = {"supported-exact": "independent", "same-runtime": "same_runtime"}

EXECUTION_MODES = ("eager", "compiled", "compiled-graphs")
DECODING_MODES = ("plain", "speculative")
HARDWARE_GENERATIONS = ("ampere", "ada", "hopper", "blackwell")
REPLAY_EXECS = ("eager", "compiled-nograph", "compiled-fx")
