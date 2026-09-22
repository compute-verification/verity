"""``shapes.json`` for W4 (numerics): every GEMM / attention / norm / activation op of a case with
its operand shapes, dtypes, resolved kernel path (Table 2 ``runtime_impl``; ``unresolved`` when
unknown), and prefill vs decode occurrence, derived from a run's ``description.json`` plus the
boundaries (token counts per phase).

``python -m verity_vllm_adapter.shapes <run_dir> --out shapes.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verity_vllm.capture import CaptureReader
from verity_vllm.description import Description


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reader = CaptureReader(args.run_dir)
    desc = Description(reader.description)
    manifest = reader.manifest
    symbols = desc.symbols
    # token counts per phase across the capture (rows the op saw)
    rows_by_phase = {"prefill": 0, "decode": 0}
    max_prefill_chunk = 0
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        for s in b.steps:
            rows_by_phase[s.phase] = rows_by_phase.get(s.phase, 0) + len(s.positions)
            if s.phase == "prefill":
                max_prefill_chunk = max(max_prefill_chunk, len(s.positions))
    weights = reader.weights.index if reader.has_weights() else None
    wshape: dict[str, Any] = {}
    if weights is not None:
        for e in weights.entries:
            wshape[e.param] = {"shape": list(e.shape), "dtype": e.dtype}

    ops = []
    for n in desc.nodes:
        if n.op_type not in ("gemm", "lm_head", "attention", "rmsnorm", "silu_mul", "rope", "embedding", "residual_add"):
            continue
        out_shape = [symbols.get(d, d) if isinstance(d, str) else d for d in n.port_shape(0)]
        inputs = []
        for a in n.args:
            d: dict[str, Any] = {"name": a.name, "source": a.source}
            if a.source == "node":
                src = desc.node(a.op or "")
                d["op"] = a.op
                d["row_shape"] = [symbols.get(x, x) if isinstance(x, str) else x for x in src.port_shape(a.port)]
                d["dtype"] = src.port_dtype(a.port)
                cols = n.extra.get("arg_cols", {}).get(a.name)
                if cols:
                    d["cols"] = cols
                if a.t and a.t != "same":
                    d["t"] = a.t
            elif a.source == "weight":
                d["param"] = a.param
                d.update(wshape.get(a.param or "", {}))
            elif a.source == "const":
                d["value"] = a.value
            inputs.append(d)
        entry = {
            "op_id": n.op,
            "layer": n.layer,
            "op_type": n.op_type,
            "runtime_op": n.extra.get("runtime_op"),
            "runtime_impl": n.runtime_impl or "unresolved",
            "relation": n.relation,
            "output": {"row_shape": out_shape, "dtype": n.dtype, "ports": n.ports},
            "inputs": inputs,
            "consts": n.extra.get("consts", {}),
            "phases": list(n.phases),
            "positions": n.positions,
            "rows_seen": {ph: (rows_by_phase.get(ph, 0) if n.positions == "all" else None) for ph in n.phases},
            "gemm": None,
        }
        if n.op_type in ("gemm", "lm_head"):
            w = next((i for i in inputs if i["source"] == "weight"), {})
            N, K = (w.get("shape") or [None, None])[:2]
            entry["gemm"] = {
                "M_prefill_max": max_prefill_chunk, "M_decode": "num_running_requests (<= 16 here)", "N": N, "K": K,
                "layout": "x[M,K] @ weight[N,K]^T; weight passed as .t() view", "accumulate": "fp32 (Triton persistent kernel), bf16 out",
            }
        if n.op_type == "attention":
            c = n.extra.get("consts", {})
            entry["attention"] = {
                "backend": c.get("backend"), "fa_version": c.get("fa_version"), "num_splits": c.get("num_splits"),
                "num_heads": c.get("num_heads"), "num_kv_heads": c.get("num_kv_heads"), "head_size": c.get("head_size"),
                "block_size": c.get("block_size"), "scale": c.get("scale"), "kv_cache_dtype": c.get("kv_cache_dtype"),
                "kv_layout": "LBNHC (paged, [num_blocks, block_size, kv_heads, head_size] per K and V)",
                "prefill": "varlen causal over keys 0..t, max_seqlen_q = chunk length", "decode": "seqlen_q = 1, keys 0..t",
            }
        ops.append(entry)
    doc = {
        "schema": "verity-vllm-adapter/shapes/v1",
        "case": manifest.get("case"),
        "profile_id": reader.profile_id,
        "model": manifest.get("model"),
        "vllm": manifest.get("vllm"),
        "gpu": manifest.get("gpu"),
        "dtype": manifest.get("dtype"),
        "engine_flags": manifest.get("engine_flags"),
        "replay_execution": "eager (enforce_eager, no torch.compile, no CUDA graphs); custom ops enabled",
        "symbols": symbols,
        "token_rows": rows_by_phase,
        "max_prefill_chunk": max_prefill_chunk,
        "kernel_paths": {
            "gemm": "vllm.model_executor.determinism.batch_invariant.matmul_persistent (Triton, VLLM_BATCH_INVARIANT=1, SM89)",
            "rmsnorm_no_residual": "vllm.model_executor.determinism.batch_invariant.rms_norm_batch_invariant (Triton _rms_norm_kernel)",
            "rmsnorm_fused_residual": "torch.ops._C.fused_add_rms_norm (vLLM CUDA)",
            "rope": "torch.ops._C.rotary_embedding (vLLM CUDA, neox style)",
            "silu_mul": "torch.ops._C.silu_and_mul (vLLM CUDA)",
            "attention": "vllm.vllm_flash_attn.flash_attn_varlen_func (FA2 on SM89, paged KV, num_splits=1)",
            "kv_write": "vllm._custom_ops.reshape_and_cache_flash",
            "embedding": "aten::embedding",
            "sampler": "vllm.v1.worker.gpu.sample.sampler (greedy argmax on fp32 logits)",
        },
        "ops": ops,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}: {len(ops)} ops")


if __name__ == "__main__":
    main()
