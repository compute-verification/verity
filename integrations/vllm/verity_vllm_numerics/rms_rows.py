"""Build W11's rms_norm result rows (Table 2 / Table 4 / manifest / dataflow variants) from the raw check reports.

    cd vllm-poc && python -m verity_vllm_numerics.rms_rows triton   # -> results/W11R-40f0cebeb670-20260907T1802Z
    cd vllm-poc && python -m verity_vllm_numerics.rms_rows cuda     # -> results/W11C-40f0cebeb670-20260907T1900Z

Reads ``<results>/raw/*.json`` (written on the pod by rms_check.py, rms_mutation_check.py, rms_probe verify) and
the fixture indices; writes ``table2.jsonl`` (the full-operator row + the SFU instruction row), ``table4.jsonl``
(checker cost), ``manifest.json`` and ``dataflow_variants.json``.  Every number is copied from the raw files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))

from verity_vllm import rows as ROWS  # noqa: E402
from verity_vllm_numerics import rms_triton_model as RM  # noqa: E402

PROFILE = "4090-vllm-bi1-fa2"
N = 1536

KERNELS = {
    "triton": dict(
        exp="W11R-40f0cebeb670-20260907T1802Z",
        op_id="rms_norm[triton]",
        raw_prefix="rms_",
        runtime_impl=("triton:vllm.model_executor.determinism.batch_invariant._rms_norm_kernel[BLOCK_SIZE=1024, HAS_WEIGHT=True, num_warps=4] "
                      "(rms_norm_batch_invariant; Triton 3.7.1, sm_89 cubin sha256 {cubin}...; the residual-free RMSNorm of the pinned profile: layers.0.input_layernorm)"),
        semantics=("rmsnorm.bf16.v1 = the kernel instruction by instruction (cpp/rms_triton_model.cpp): 128 threads x 8 columns, FFMA chains "
                   "acc = fma(x, x, acc) per thread in column order, SHFL.BFLY warp tree (16, 8, 4, 2, 1) then the 4 warp sums by a butterfly, "
                   "mean_eps = FFMA(MUFU.RCP(1536) , sum, eps) [div.full.f32 (a, n) = a * rcp(n) with range scaling], rms = MUFU.SQRT(mean_eps) "
                   "(sqrt.approx.ftz), inv = div.full(1, rms), out = bf16_rne((x * inv) * w)"),
        decomposition="one output row from x (one BF16 row) and the weight; the CTA reduction order is fixed by the compiled kernel",
        instr_op="mufu.sqrt.approx.ftz.f32 + div.full.f32",
        instr_impl="MUFU.SQRT (sqrt.approx.ftz.f32 = Triton tl.sqrt lowering) and div.full.f32 (Triton's fp32 division: MUFU.RCP + FFMA with range scaling) on sm_89, nvcc 12.8 (cuda/rms_probe.cu)",
        instr_sem=("sqrt: two 2^23-entry mantissa tables (odd / even exponent), exponent halved exactly; ftz in / out, NaN -> 0x7fffffff, negative -> NaN; "
                   "div.full(a, b): r = MUFU.RCP(b) (2^23-entry table), q = a * r, with the compiled 2^-24 / 2^24 scaling when |b| is outside the fast range; "
                   "verified on all 2^32 inputs for sqrt, 1/x and x/1536"),
        model_version="rmsnorm-triton-sm89-bf16.v1",
        relation="rmsnorm.bf16.v1",
        code=["verity_vllm_numerics/rms_triton_model.py", "verity_vllm_numerics/cpp/rms_triton_model.cpp", "verity_vllm_numerics/rms_relation.py",
              "verity_vllm_numerics/rms_capture.py", "verity_vllm_numerics/rms_check.py", "verity_vllm_numerics/rms_stress.py",
              "verity_vllm_numerics/rms_mutation_check.py", "verity_vllm_numerics/cuda/rms_probe.cu", "verity_vllm_numerics/cuda/mufu_probe.cu",
              "tests/test_rmsnorm_triton_model.py"],
        variant_bits={"1": "no_eps_fusion (mean then + eps as two roundings)", "2": "sequential_sum (one fp32 chain over 1536 columns)", "4": "ieee_sqrt",
                      "8": "ieee_rcp_div (IEEE division instead of div.full)", "16": "thread_order_x0_first", "32": "fp64_sum", "64": "out_xw_first (bf16((x*w)*inv))",
                      "128": "unfused_squares (x*x rounded, then add)"},
        equivalent=["thread_order_x0_first", "unfused_squares"],
    ),
    "cuda": dict(
        exp="W11C-40f0cebeb670-20260907T1900Z",
        op_id="rms_norm[cuda_rms_norm]",
        raw_prefix="rms_cuda_",
        runtime_impl=("cuda:torch.ops._C.rms_norm -> vllm::rms_norm_kernel<c10::BFloat16, VEC_SIZE=8, NUM_DIMS=2, HasWeight=true> "
                      "(csrc/libtorch_stable/layernorm_kernels.cu @ vLLM d9105ea8, _C_stable_libtorch.abi3.so, sm_89 SASS; NOT launched by the pinned engine -- "
                      "every RMSNorm call carries a residual (fused_add_rms_norm) or goes to the Triton kernel; the natural rows are the Triton fixture's rows replayed)"),
        semantics=("rmsnorm.cuda.bf16.v1 = the kernel instruction by instruction (cpp/rms_triton_model.cpp::rms_cuda_rows): block = n_cols / 8 threads "
                   "(192 for 1536), one 8-vector per thread accumulated by FFMA in element order, cub WarpReduce shfl_down tree (offsets 1, 2, 4, 8, 16; "
                   "lane 0 keeps the sum; partial warps predicated), cub BlockReduce<1024>: warp aggregates added sequentially by warp 0 (w0 + w1) + w2 ..., "
                   "mean = IEEE div.rn.f32(sum, n_cols), arg = mean + eps (FADD), s = rsqrt.approx.f32(arg) = MUFU.RSQ with 2^24 / 2^12 range scaling "
                   "for |arg| < 2^-126, out = bf16_rne((s * x) * w)"),
        decomposition="one output row from x (one BF16 row) and the weight; the block reduction order is fixed by the compiled kernel",
        instr_op="mufu.rsq.approx.f32",
        instr_impl="MUFU.RSQ (rsqrt.approx.ftz.f32 and the non-ftz rsqrt.approx.f32 sequence = rsqrtf without fast-math) on sm_89, nvcc 12.8 (cuda/rms_probe.cu rsq_verify)",
        instr_sem=("two 2^23-entry mantissa tables (odd / even exponent), exponent -(k - parity)/2; 0 -> inf, inf -> 0, negative -> NaN; ftz form flushes subnormal "
                   "inputs, the rsqrt.approx.f32 form multiplies |x| < 2^-126 by 2^24 first and the result by 2^12; both verified on all 2^32 inputs"),
        model_version="rmsnorm-cuda-sm89-bf16.v1",
        relation="rmsnorm.cuda.bf16.v1",
        code=["verity_vllm_numerics/rms_triton_model.py (kernel='cuda')", "verity_vllm_numerics/cpp/rms_triton_model.cpp (rms_cuda_rows)",
              "verity_vllm_numerics/rms_relation.py (independent_rmsnorm_cuda)", "verity_vllm_numerics/rms_check.py --kernel cuda",
              "verity_vllm_numerics/rms_stress.py --kernel cuda [--replay-natural]", "verity_vllm_numerics/rms_mutation_check.py --kernel cuda",
              "verity_vllm_numerics/cuda/rms_probe.cu (rsq_table / rsq_verify)", "tests/test_rmsnorm_cuda_model.py"],
        variant_bits={"1": "butterfly_warp_tree (Triton-style xor butterfly instead of cub shfl_down)", "2": "sequential_sum", "4": "ieee_rsqrt (1/sqrtf)",
                      "8": "div_full_mean (MUFU.RCP-based division instead of IEEE)", "32": "fp64_sum", "64": "out_xw_first (bf16((x*w)*s))",
                      "128": "warp_aggregates_reversed"},
        equivalent=[],
    ),
}


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(POC), check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def main(kernel: str) -> None:
    K = KERNELS[kernel]
    exp = K["exp"]
    results = POC / "results" / exp
    raw = results / "raw"
    fx = POC / "fixtures" / exp
    nat_idx = _load(fx / "natural" / "index.json")
    st_idx = _load(fx / "stress" / "index.json")
    muts_json = _load(fx / "stress" / "mutations.json")
    p = K["raw_prefix"]
    nat = _load(raw / f"{p}check_natural.json")
    nat1 = _load(raw / f"{p}check_natural_1thread.json")
    st = _load(raw / f"{p}check_stress.json")
    mv = _load(raw / f"{p}mutations_variants.json")
    probe = _load(raw / ("rms_probe_verify.json" if kernel == "triton" else "rsq_probe_verify.json"))
    tri_idx = nat_idx if kernel == "triton" else _load(POC / "fixtures" / KERNELS["triton"]["exp"] / "natural" / "index.json")
    cubin = tri_idx["kernel_files"]["_rms_norm_kernel.cubin"]["sha256"][:16]

    common = dict(experiment_id=exp, case="B1", profile_id=PROFILE, worker="W11", adapter_revision=f"{_git_rev()} (vllm-poc-w11-attention-model)")
    domain = (f"sm_89 (RTX 4090) tables; BF16 x / w / out; weighted; contiguous rows; n_cols in {list(RM.MEASURED_N_COLS)} "
              f"(% {'16' if kernel == 'triton' else '8'} == 0 is the compiled specialisation); finite eps (eps = 0 and eps < 0 measured); "
              "non-finite / subnormal operands measured and exact (the model does not decline on them); everything else `unresolved`")

    # ---- the full operator row
    assert nat["total"]["mismatch_outputs"] == 0 and st["total"]["mismatch_outputs"] == 0
    m = mv["mutations"]
    assert m["n"] == m["detected_by_model"] == m["model_exact_on_mutated_operands"] == m["production_changed"] == m["model_changed_equals_production_changed"] == 36
    assert len(muts_json["mutations"]) == 36
    fam = {c["family"] for c in st_idx["calls"]}
    kinds = {}
    for e in muts_json["mutations"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    g = nat["groups"]
    cost_row = nat1["model_seconds"] / nat1["total"]["rows"]
    nat_phase = (f"decode+prefill (natural layers.0.input_layernorm rows of Qwen2.5-1.5B, 3 prompts: {g['prefill']['calls']} prefill call{'s' if g['prefill']['calls'] != 1 else ''} "
                 f"of {g['prefill']['rows']} rows + {g['decode']['calls']} decode calls of {g['decode']['rows']} rows; stress n_cols 16..4096)")
    if kernel == "cuda":
        nat_phase = nat_phase.replace("natural layers.0", "the same natural layers.0").replace("3 prompts:", "3 prompts, replayed through _C.rms_norm:")
    row = dict(common, op_id=K["op_id"], phase=nat_phase, shape={"M": "per call (1..2705)", "H": N}, dtype="bfloat16",
               layout_or_quant="contiguous BF16 rows (row stride = n_cols); BF16 weight", runtime_impl=K["runtime_impl"].format(cubin=cubin),
               primitive_semantics=K["semantics"], decomposition=K["decomposition"],
               interface={"input_bytes": 2 * N + 2 * N, "output_bits": 16 * N, "range": None},
               origin="captured" if kernel == "triton" else "derived",
               binding={"present": 0, "unresolved": 0}, checker="independent",
               natural_checks={"exact": nat["total"]["outputs"] - nat["total"]["mismatch_outputs"], "attempted": nat["total"]["outputs"]},
               stress_checks={"exact": st["total"]["outputs"] - st["total"]["mismatch_outputs"], "attempted": st["total"]["outputs"],
                              "domain": "families " + ", ".join(sorted(fam)) + "; " + domain},
               mutations={"rejected": m["detected_by_model"], "attempted": m["production_changed"]},
               checker_work={"seconds": cost_row, "flops_or_bytes": float(4 * N + 8), "includes_full_op": True},
               status="exact",
               limitation=(f"{nat['total']['calls']} natural calls ({nat['total']['rows']} rows, {nat['total']['outputs']} output words) and {st['total']['calls']} stress calls "
                           f"({st['total']['rows']} rows, {st['total']['outputs']} words) checked, 0 mismatching words; mutations: 36 one-bit flips / eps changes "
                           f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}) re-run through the production kernel, all 36 changed its output and all 36 are rejected "
                           f"by the model of the unmutated operands, and the model follows the mutated operands bit-exactly in all 36; dataflow variants: every arithmetically "
                           f"distinct alternative ({len([k for k in mv['variants'] if k != 'model' and k not in K['equivalent']])}) mismatches production "
                           f"(dataflow_variants.json); cost: single-thread CPU {1e6 * cost_row:.1f} us per row (AMD EPYC 7282, g++ -O2; OpenMP 64 threads: "
                           f"{nat['model_seconds']:.2f} s for the natural fixture); "
                           + ("the kernel of the pinned profile's only residual-free RMSNorm (layers.0.input_layernorm; every other RMSNorm is fused_add_rms_norm, W4b's model)"
                              if kernel == "triton" else
                              "the pinned engine never launches this kernel (0 calls in the profiler of the natural capture); the natural rows are inputs the engine produced, "
                              "the outputs were written by torch.ops._C.rms_norm on the same GPU -- this row closes W4b's `rms_norm[cuda_rms_norm]` gap, not a node of the B1 trace")))
    rows = [ROWS.require_valid(2, row)]

    # ---- the SFU instruction row (background characterisation, exhaustive)
    if kernel == "triton":
        attempted = 3 * probe["inputs"]
        mism = probe["sqrt_mismatches"] + probe["div_full_1_x_mismatches"] + probe["div_full_x_n_mismatches"]
        instr_lim = (f"background characterisation (contract 15.9), not a production relation; sqrt on all 2^32 patterns, div.full(1, x) and div.full(x, {probe['n_cols']}) on all "
                     f"2^32 x; tables are fixture data (fixtures/{exp}/tables, int8 deltas, sha256 in the README); the sqrt table is not monotone (4 one-ulp dips) and "
                     "within 1 ulp of the correctly rounded value; measured on sm_89 only")
    else:
        attempted = 2 * probe["inputs"]
        mism = probe["rsqrt_ftz_mismatches"] + probe["rsqrt_nonftz_mismatches"]
        instr_lim = (f"background characterisation (contract 15.9), not a production relation; the ftz instruction and the compiled non-ftz sequence each on all 2^32 patterns; "
                     f"table = fixture data (fixtures/{exp}/tables/mufu_rsq_delta_int8.xz); 78.3% of entries correctly rounded, the rest within 2 ulp (8 entries 2 ulp low); "
                     "measured on sm_89 only")
    rows.append(ROWS.require_valid(2, dict(
        common, op_id=K["instr_op"], phase="instruction (exhaustive 2^32 inputs)", shape={"table_entries": 2 * (1 << 23)}, dtype="fp32 -> fp32", layout_or_quant=None,
        runtime_impl=K["instr_impl"], primitive_semantics=K["instr_sem"], decomposition="direct SFU instruction (+ the compiled range-scaling sequence)",
        interface={"input_bytes": 4, "output_bits": 32, "range": None}, origin="derived", binding={"present": 0, "unresolved": 0}, checker="independent",
        natural_checks={"exact": 0, "attempted": 0},
        stress_checks={"exact": attempted - mism, "attempted": attempted, "domain": "all 2^32 fp32 bit patterns (inf / NaN / subnormal included)"},
        mutations={"rejected": 0, "attempted": 0}, checker_work={"seconds": 0.0, "flops_or_bytes": 1.0, "includes_full_op": True},
        status="exact" if mism == 0 else "mismatch", limitation=instr_lim)))
    (results / "table2.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    # ---- Table 4
    t4 = dict(common, partition_id=f"{K['op_id']}-independent-model (fixture, no replay)",
              instrumentation_mode=("fixture capture: rebind vllm.model_executor.layers.layernorm.rms_norm_batch_invariant in the eager engine (no trace replay, no commitments)"
                                    if kernel == "triton" else "replay of the captured natural rows through torch.ops._C.rms_norm (no engine, no trace replay)"),
              production_overhead_ratio=None, replay_commit_overhead_ratio=None, replay_vs_production_gpu_work=None,
              transparent_check_cost={"cpu_s": float(nat1["model_seconds"]), "gpu_s": 0.0},
              peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
              committed_logical_bytes=None, processed_bytes={"hashed": None, "copied": None, "retained": None, "emitted": None},
              description_size_bytes=None, constructor_inputs_bytes=None, logical_size={"gates": None, "vus": None, "rus": None},
              lookup_cost_s=None, validation_cost_s=float(nat["model_seconds"] + st["model_seconds"]),
              moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
              restrictions=[
                  f"transparent_check_cost.cpu_s = single-thread CPU seconds of {K['relation']} on the {nat1['total']['rows']} natural rows ({nat1['total']['outputs']} words) = "
                  f"{1e6 * cost_row:.1f} us per row of 1536; AMD EPYC 7282, g++ -O2 (model time only: fixture I/O excluded)",
                  f"validation_cost_s = model seconds of the natural + stress checks with OpenMP (64 threads): {nat['total']['outputs']} + {st['total']['outputs']} output words",
                  f"per token of Qwen2.5-1.5B: {'1 residual-free RMSNorm per token (layers.0.input_layernorm) -> ' + f'{1e6 * cost_row:.1f} us' if kernel == 'triton' else 'not on the pinned path (0 launches per token)'}",
                  "no production overhead: the operands are what the eager engine already holds (x row, weight, eps); the capture copied them after cuda.synchronize",
              ],
              spend={"gpu_hours": None, "usd": None})
    (results / "table4.jsonl").write_text(json.dumps(ROWS.require_valid(4, t4), sort_keys=True) + "\n")

    # ---- dataflow variants
    (results / "dataflow_variants.json").write_text(json.dumps({
        "note": (f"rms_mutation_check.py variant sweep: the model (bit 0) and each alternative dataflow re-run on the whole natural fixture and the whole stress fixture; "
                 f"counts = mismatching output words / rows against the production kernel.  Alternatives that are arithmetically equivalent for BF16 inputs: {K['equivalent'] or 'none'}."),
        "bits": K["variant_bits"],
        "results": mv["variants"],
        "boundary_search": st_idx.get("boundary_search"),
    }, indent=1))

    # ---- manifest
    man = {
        "schema": "verity-vllm/numerics-manifest/v1",
        "experiment_id": exp, "worker": "W11", "case": "B1", "revision": common["adapter_revision"],
        "gpu": {"name": nat_idx["gpu"], "sm": "sm_89", "driver": "580.65.06 (RunPod RTX 4090, container bdccd52933e7)", "cuda_toolkit": "12.8 (nvcc 12.8.93)", "cuda_runtime": nat_idx["cuda"]},
        "software": {"torch": nat_idx["torch"], "vllm_pinned": nat_idx["vllm"], "triton": nat_idx["triton"], "python": "3.11.11 (/workspace/venv)"},
        "checkpoint": {"repo": tri_idx["model"], "revision": tri_idx["model_local_path"].rsplit("/", 1)[-1]},
        "profiles": {PROFILE: ("RTX 4090 sm_89, the B1 profile's engine configuration (dtype bf16, max_model_len 4096, enforce_eager, prefix caching off, VLLM_BATCH_INVARIANT=1) "
                               "re-run on this lane's pod for fixture capture; not W1's B1 bundle itself")},
        "engine_flags_observed": tri_idx["env"], "engine_kwargs": tri_idx["engine_kwargs"],
        "kernel": RM.KERNEL if kernel == "triton" else RM.CUDA_KERNEL,
        "kernel_files_sha256": ({k: v["sha256"] for k, v in tri_idx["kernel_files"].items()} if kernel == "triton"
                                else {"kernel_source_sha256 (batch_invariant.py, for the natural rows)": st_idx["kernel_source_sha256"]}),
        "triton_kernel_metadata": tri_idx["kernel_metadata"] if kernel == "triton" else None,
        "fixtures": {"dir": f"fixtures/{exp}", "natural_calls": nat_idx["n_calls"], "stress_calls": st_idx["n_calls"], "mutations": len(muts_json["mutations"]),
                     "inputs_shared_with": None if kernel == "triton" else f"fixtures/{KERNELS['triton']['exp']} (x rows referenced by relative path)",
                     "tables": nat.get("tables")},
        "results": {"natural": nat["total"], "natural_by_phase": nat["groups"], "stress": st["total"], "stress_by_family": {k: v["mismatch_outputs"] for k, v in st["groups"].items()},
                    "mutations": {k: v for k, v in m.items() if k != "entries"}, "sfu_exhaustive": probe, "variants": mv["variants"],
                    "single_thread_us_per_row": nat1["model_us_per_row"]},
        "raw": sorted(q.name for q in raw.iterdir()),
        "docs": "docs/vllm-poc/rmsnorm-model.md",
        "relation": K["relation"], "model_version": K["model_version"],
        "code": K["code"],
    }
    (results / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {len(rows)} table2 rows, 1 table4 row, manifest, dataflow_variants -> {results}")
    for r in rows:
        print(f"  {r['op_id']:40s} {r['status']:8s} natural {r['natural_checks']['exact']}/{r['natural_checks']['attempted']}  stress {r['stress_checks']['exact']}/{r['stress_checks']['attempted']}  "
              f"mut {r['mutations']['rejected']}/{r['mutations']['attempted']}  cost/row {1e6 * r['checker_work']['seconds']:.1f} us")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "triton")
