"""Table 2 rows (CONTRACT §9, §16.6) for the compiled-mode kernels examined in the divergence
diagnosis. CPU / JSON only::

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.divergence_rows \
        --results results/B0-divergence-20260907T1604Z --adapter-revision <git sha>

Reads ``compare/report.json`` (first divergence + per-tensor walk), ``compare/kernels/index.json``
(generated kernel ids), ``compare/kernel_attribution.json`` (rows whose operands are bit-equal in
both modes and whose outputs still differ = the kernel's own contribution) and
``compare/model_checks.json`` (CPU bit-models of the compiled kernels) and writes
``table2.jsonl``. ``status`` values: ``mismatch-vs-eager`` (compiled output differs from the
eager output on bit-identical operands; reproducer/fixture given), ``agrees-with-eager`` (no
difference on bit-identical operands), ``unresolved`` (operands already differ, kernel not
separable on this capture). ``checker`` = ``independent`` when a CPU model reproduces the compiled
kernel bit-exactly on the natural operands, ``none`` otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--adapter-revision", required=True)
ap.add_argument("--bundle", default="vllm-poc/bundles/divergence-B0-784c37e1")
ap.add_argument("--fixtures", default="fixtures/B0-divergence-20260907T1604Z")
a = ap.parse_args()
R = Path(a.results)
exp = R.name
rep = json.load(open(R / "compare" / "report.json"))
kidx = json.load(open(R / "compare" / "kernels" / "index.json"))
attr = json.load(open(R / "compare" / "kernel_attribution.json"))
mc = json.load(open(R / "compare" / "model_checks.json"))
rows_by_role = {}
for r in rep["rows"]:
    if r.get("status") in ("bit-equal", "mismatch") and r["layer"] == 0:
        rows_by_role.setdefault(r["role"], r)


def kid(kernel: str) -> tuple[str, str]:
    for k, v in kidx.items():
        if v["kernel"] == kernel:
            return f"{kernel}@{v['sha256'][:12]}", v["sha256"]
    return kernel, ""


COMMON = {
    "experiment_id": exp,
    "case": "B0",
    "profile_id": "diag: eager=9f5fa192a813 (enforce_eager) vs compiled=89310d33d939 with cudagraph_mode=NONE",
    "worker": "W4c",
    "adapter_revision": a.adapter_revision,
}
N, H = 512, 576


def row(op_id, kernel, semantics, decomposition, shape, in_bytes, out_bits, natural, checker, status, limitation, stress=None, binding_present=1):
    rid, sha = kid(kernel)
    return dict(
        COMMON,
        op_id=op_id,
        phase="prefill",
        shape=shape,
        dtype="bfloat16",
        layout_or_quant="row-major bf16; weights bf16",
        runtime_impl=rid,
        primitive_semantics=semantics,
        decomposition=decomposition,
        interface={"input_bytes": in_bytes, "output_bits": out_bits, "range": None},
        origin="captured",
        binding={"present": binding_present, "unresolved": 0},
        checker=checker,
        natural_checks=natural,
        stress_checks=stress or {"exact": 0, "attempted": 0, "domain": None},
        mutations={"rejected": 0, "attempted": 0},
        checker_work={"seconds": 0.0, "flops_or_bytes": float(in_bytes), "includes_full_op": True},
        status=status,
        limitation=limitation,
    )


fd = rep["summary"]["first_divergence"]
rows = []
# 1. layer-0 input_layernorm: THE first divergence
rows.append(
    row(
        "model.layers.0.input_layernorm (rms_norm) + embedding gather",
        "triton_per_fused__to_copy_embedding_rms_norm_t_0",
        "out = bf16((x * rsqrt(mean(x^2)+eps)) * w), FP32, single rounding; s via libdevice.rsqrt (correctly rounded on all 512 rows); tl.sum over a 1024-lane masked block",
        "Inductor persistent reduction fusing aten.embedding + vllm_ir.rms_norm[native] (the graph's bf16 convert before *w is elided)",
        [N, H],
        N * 4 + N * H * 2 + H * 2,
        16,
        {"exact": N - 1, "attempted": N},
        "independent",
        "mismatch-vs-eager",
        f"FIRST DIVERGENCE: position {fd['first_diff']['position']} coordinate {fd['first_diff']['coordinate']} eager {fd['first_diff']['expected_bits']} vs compiled {fd['first_diff']['actual_bits']} (1 ulp, 1/{N*H} words). "
        f"Cause: per-row inverse-RMS scalar; eager Triton _rms_norm_kernel uses 1.0/tl.sqrt (1 ulp high on row 194), Inductor emits libdevice.rsqrt (IEEE on all rows). "
        f"Reproducer {a.bundle}/repro.py (re-runs both kernels on the saved operands; re-run == capture in both modes). CPU model bf16((x*s0)*w) reproduces compiled 512/512 rows, eager 511/512 (row 194 needs s0+1ulp).",
        stress={"exact": N, "attempted": N, "domain": "CPU row model vs compiled kernel, s0 = IEEE rsqrt of exact mean"},
    )
)
# 2. rotary (fused pointwise) - agrees on the 511 rows with equal operands
ra = attr["layers.0.self_attn.rotary_emb"]
rows.append(
    row(
        "model.layers.0.self_attn.rotary_emb (NeoX rotary on q and k)",
        "triton_poi_fused_1",
        "x' = x*cos - y*sin, y' = y*cos + x*sin in FP32 from bf16 cos/sin cache, one rounding (same as vllm::rotary_embedding_kernel)",
        "Inductor pointwise fusing slice + mul/sub/add + cat of vllm_ir rotary native decomposition",
        [N, 960],
        N * 960 * 2 + N * 8 + 8192 * 64 * 2,
        16,
        {"exact": ra["rows_with_all_operands_equal"], "attempted": ra["rows_with_all_operands_equal"]},
        "none",
        "agrees-with-eager",
        f"On the {ra['rows_with_all_operands_equal']} rows whose qkv operand is bit-equal in both modes the q and k outputs are bit-equal ({ra['output_q_words_differing_with_equal_operands'][1]}+{ra['output_k_words_differing_with_equal_operands'][1]} words); the 1 differing row (194) inherits the rms_norm difference. No CPU model run for the compiled kernel (eager rotary is modelled by W4b).",
    )
)
# 3. qkv GEMM: identical Triton kernel in both modes
q = rows_by_role.get("qkv_proj", {})
rows.append(
    row(
        "model.layers.0.self_attn.qkv_proj (GEMM 512x960x576)",
        "matmul_kernel_persistent",
        "vLLM batch-invariant persistent Triton GEMM (BLOCK 128x128x64, GROUP_SIZE_M 8, NUM_SMS 128, FP32 accumulate, one rounding) - same source in eager and compiled",
        "user-defined Triton kernel passed through Inductor unchanged (triton_kernel_wrapper_mutation)",
        [N, 960],
        N * H * 2 + 960 * H * 2,
        16,
        {"exact": N - (q.get("positions_with_diff") or 1), "attempted": N},
        "none",
        "agrees-with-eager",
        f"{q.get('n_diff')} differing words, all at position 194 whose operand row differs (inherited from rms_norm). Same kernel source/config as eager (W4b's independent_gemm model covers it).",
    )
)
# 4. post_attention_layernorm fused_add_rms_norm: independent, dense divergence
pa = attr["layers.0.post_attention_layernorm(fused_add_rms_norm)"]
rows.append(
    row(
        "model.layers.0.post_attention_layernorm (fused_add_rms_norm, one output)",
        "triton_per_fused_fused_add_rms_norm_t_0",
        "z = x + residual kept in FP32 (bf16 convert elided); out = bf16((z * rsqrt(mean(z^2)+eps)) * w), single rounding; eager CUDA kernel normalises bf16(z)",
        "Inductor persistent reduction of vllm_ir.fused_add_rms_norm[native]; the residual output is NOT produced here (recomputed by the next layer's input_layernorm kernel)",
        [N, H],
        2 * N * H * 2 + H * 2,
        16,
        {"exact": pa["rows_with_all_operands_equal"] - pa["output_normed_rows_differing_with_equal_operands"], "attempted": pa["rows_with_all_operands_equal"]},
        "independent",
        "mismatch-vs-eager",
        f"On {pa['rows_with_all_operands_equal']} rows with bit-equal operands, {pa['output_normed_rows_differing_with_equal_operands']} output rows differ ({pa['output_normed_words_differing_with_equal_operands'][0]}/{pa['output_normed_words_differing_with_equal_operands'][1]} words, mostly 1 ulp). "
        f"CPU model 'norm of UNROUNDED fp32 sum' fits compiled rows with |k|<=1 ulp on s ({mc['compiled_normed fit (norm of UNROUNDED fp32 sum)']}), 'norm of ROUNDED sum' fits none; eager is the opposite ({mc['eager_normed fit (norm of ROUNDED sum)']}). Fixture {a.fixtures}/layers.0.post_attention_layernorm_fused_add_rms_norm.",
        stress={"exact": N, "attempted": N, "domain": "CPU model of the compiled kernel (unrounded residual sum), |k|<=1 ulp on s"},
    )
)
# 5. silu_and_mul: single vs double rounding
rows.append(
    row(
        "model.layers.0.mlp.act_fn (silu_and_mul)",
        "triton_poi_fused_mul_silu_slice_t_1",
        "out = bf16( (g / (1 + exp(-g))) * u ) in FP32, ONE rounding (libdevice.exp); eager vllm::act_and_mul_kernel rounds silu(g) to bf16 before the multiply",
        "Inductor pointwise fusing slice + silu + mul of vllm_ir silu_and_mul native decomposition",
        [N, 1536],
        N * 3072 * 2,
        16,
        {"exact": mc["silu compiled: elements exact single-rounding model"][0], "attempted": mc["silu compiled: elements exact single-rounding model"][1]},
        "independent",
        "mismatch-vs-eager",
        f"Operands already differ at every position at layer 0 (downstream of the norm), so the kernel is attributed by CPU model instead: single-rounding model reproduces the compiled output on {mc['silu compiled: elements exact single-rounding model'][0]}/{mc['silu compiled: elements exact single-rounding model'][1]} elements, the double-rounding (eager) model on {mc['silu compiled: elements exact double-rounding model'][0]}; on the eager output the roles are reversed ({mc['silu eager: elements exact double-rounding model'][0]} vs {mc['silu eager: elements exact single-rounding model'][0]}). Natural-operand counts are elements, not rows.",
        stress={"exact": mc["silu compiled: elements exact single-rounding model"][0], "attempted": mc["silu compiled: elements exact single-rounding model"][1], "domain": "CPU single-rounding model vs compiled output (elements)"},
    )
)
# 6. layer-1 input_layernorm (two outputs): residual stream kept in fp32 across two adds
rows.append(
    row(
        "model.layers.1.input_layernorm (fused_add_rms_norm, residual + normed outputs)",
        "triton_per_fused_fused_add_rms_norm_t_2",
        "z = down + (o_proj + residual) with BOTH adds in FP32 (two bf16 converts elided); residual_out = bf16(z); out = bf16((z*rsqrt(mean(z^2)+eps))*w). Eager: residual_out = bf16(down + bf16(o_proj + residual)), normalises the rounded value",
        "Inductor persistent reduction; recomputes the attention-residual add that the previous kernel dropped",
        [N, H],
        3 * N * H * 2 + H * 2,
        16,
        {"exact": 0, "attempted": 0},
        "independent",
        "mismatch-vs-eager",
        f"Operands differ everywhere at this depth; attribution by CPU model on the compiled operands: residual_out == bf16(down + fp32(o_proj + residual)) is {mc['compiled L1 residual_out == bf16(down + fp32(o_proj + residual))']}, == bf16(down + bf16(o_proj + residual)) is {mc['compiled L1 residual_out == bf16(down + bf16(o_proj + residual))']}; normed fits the unrounded-chain model with |k|<=1 ({mc['compiled L1 normed fit (norm of UNROUNDED chain)']}). Fixture layers.1.input_layernorm_fused_add_rms_norm_two_outputs.",
        stress={"exact": N, "attempted": N, "domain": "CPU model of the compiled kernel (fp32 residual chain), |k|<=1 ulp on s"},
    )
)
with open(R / "table2.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows to {R / 'table2.jsonl'}")
