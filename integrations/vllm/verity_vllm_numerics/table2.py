"""Table 2 rows (contract §9 field names, W2's ``verity_vllm.rows.TABLE2_SPEC``) and the manifest
for the numerics experiment, built from the raw JSON the pod runs wrote.

    python -m verity_vllm_numerics.table2 --raw fixtures/results/numerics-a100-<UTC>/raw \
        --out fixtures/results/numerics-a100-<UTC> --experiment numerics-a100-<UTC> --revision <sha>

Raw inputs (all produced by modules of this package on the A100 pod):
  validate_bf16.json / validate_f16.json     random-tile validation of the per-instruction models
  hawkeye_sequence.json                      Hawkeye structured sequence (layout, neutral subgroups, records)
  gemm_vllm_triton.json (+ nightly/)         production GEMM, vLLM batch-invariant Triton kernel
  gemm_cublas.json, cublas_probe.json        production GEMM, torch default cuBLAS/cuBLASLt (no batch invariance)
  nightly/engine_kernels_{eager,compiled}.json  kernels the pinned-nightly engine launched under W1's flags
  nightly/beyond_gemm.json                   silu_and_mul, rms_norm, rotary_embedding probes
  nightly/rsqrt_probe.json, nightly/rmsnorm_table_probe.json   approximate rsqrt/div tables; RMSNorm explicit-sequence search
  engine_capture/check_cap*.json             GEMM operands captured inside the eager engine, checked via W2's registry

Row classes (contract §7): ``direct MMA instruction``, ``full production GEMM incl. accumulation and
epilogue``, ``the same GEMM inside the replayed model`` (this lane's own engine-hook fixtures of the
eager engine, ``engine_gemm_capture.py``; plus a NOT_RUN row per shape for W1's B1 capture, which was not
available), plus the beyond-GEMM ops.  Nothing in this file inspects a result to choose a
denominator: every case the pod ran is counted, unresolved ones included.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import kernel_zoo as Z

WORKER = "W4"
CASE = "B1"  # Qwen/Qwen2.5-1.5B
SHAPES = {
    "qkv_proj": (1536, 2048, True),
    "o_proj": (1536, 1536, False),
    "gate_up_proj": (1536, 17920, False),
    "down_proj": (8960, 1536, False),
    "lm_head": (1536, 151936, False),
}
PHASES = (("decode M=1", (1, 1)), ("decode M=7..16", (2, 16)), ("prefill M=64..2048", (17, 1 << 30)))
ROW_CLASS_MMA = "direct MMA instruction"
ROW_CLASS_GEMM = "full production GEMM incl. accumulation and epilogue"
ROW_CLASS_REPLAY = "the same GEMM inside the replayed model"


def load(raw: Path, name: str) -> Any:
    p = raw / name
    return json.loads(p.read_text()) if p.exists() else None


def phase_of(M: int) -> str:
    for name, (lo, hi) in PHASES:
        if lo <= M <= hi:
            return name
    raise ValueError(M)


def interface(K: int, N: int, has_bias: bool) -> dict[str, Any]:
    """One VU = one output row: the opened input row, the whole weight and the bias, 16-bit outputs."""
    return {"input_bytes": 2 * K + 2 * N * K + (2 * N if has_bias else 0), "output_bits": 16 * N, "range": None}


# --- direct MMA -------------------------------------------------------------------------------------


def mma_rows(raw: Path, common: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    hawk = load(raw, "hawkeye_sequence.json")
    for dt, tag in (("bf16", "bf16"), ("f16", "f16")):
        v = load(raw, f"validate_{tag}.json")
        if v is None:
            continue
        fam = v["families"]
        natural = {k: fam[k] for k in ("randn_zero_acc", "randn_random_acc") if k in fam}
        stress = {k: fam[k] for k in fam if k not in natural}
        n_att = sum(f["elements"] for f in natural.values())
        n_ex = n_att - sum(f["mismatches_vs_cpp"] for f in natural.values())
        s_att = sum(f["elements"] for f in stress.values())
        s_ex = s_att - sum(f["mismatches_vs_cpp"] for f in stress.values())
        h = (hawk or {}).get("dtypes", {}).get(dt)
        if h:
            s_att += h["layout"]["elements"] + h["structured"]["_records"]
            s_ex += h["layout"]["elements"] - h["layout"]["mismatches"] + h["structured"]["_records"] - h["structured"]["_mismatching_records"]
            neutral = h["neutral_subgroups"]
            s_att += neutral["subsets_tested"] * 0 + 1
            s_ex += 1 if neutral["neutral_sets_identical_to_model"] and neutral["gpu_vs_model_mismatching_tiles"] == 0 else 0
        cpp_s = sum(f["cpp_seconds"] for f in fam.values())
        m = v["model"]
        derived = (h or {}).get("structured", {}).get("_derived", {})
        rows.append({
            **common,
            "op_id": f"mma.sync.m16n8k16.{dt}.f32",
            "phase": "instruction (synthetic tiles)",
            "shape": {"m": 16, "n": 8, "k": 16},
            "dtype": f"{dt} x {dt} -> fp32",
            "layout_or_quant": "row.col",
            "runtime_impl": f"{m['instruction']} (sm_80, nvcc 12.8, verity_vllm_numerics/cuda/mma_tiles_sm80.cu)",
            "primitive_semantics": f"{m['name']} v1: groups {m['groups']}, significand width {m['width']}, exponent floor {m['zero_exponent']} ({'veritor.core.silicon.AMPERE_BF16_M16N8K16' if dt == 'bf16' else 'verity_vllm_numerics.mma.AMPERE_F16'}); Hawkeye-derived: " + ", ".join(f"{k}={v}" for k, v in derived.items() if not str(k).startswith("_")),
            "decomposition": ROW_CLASS_MMA,
            "interface": {"input_bytes": 16 * 16 * 2 + 8 * 16 * 2 + 16 * 8 * 4, "output_bits": 16 * 8 * 32, "range": None},
            "origin": "derived",
            "binding": {"present": 0, "unresolved": 0},
            "checker": "independent",
            "natural_checks": {"exact": n_ex, "attempted": n_att},
            "stress_checks": {"exact": s_ex, "attempted": s_att, "domain": "finite operands and accumulators; families: uniform bit patterns, subnormal operands/accumulators, exact cancellation, mixed magnitudes, accumulator magnitude, near-overflow products, signed zero, tiny accumulator + zero products; Hawkeye sequence: layout, neutral subgroups (exhaustive 2^17 subsets), structured records (rounding direction, internal precision, subnormals, extended range, overflow, accumulator position)"},
            "mutations": {"rejected": 0, "attempted": 0},
            "checker_work": {"seconds": cpp_s / max(1, v["total_elements"]), "flops_or_bytes": 2.0 * 16, "includes_full_op": True},
            "status": "exact" if (n_ex == n_att and s_ex == s_att) else "mismatch",
            "limitation": "background characterisation (contract §15.9), not a production relation; per output element of one instruction; C++ twin of the Python model (Python cross-checked on 1.7M elements per dtype)",
        })
    return rows


# --- production GEMM ----------------------------------------------------------------------------------


def _shape_of(label: str) -> str | None:
    for s in SHAPES:
        if label.endswith(s) or label == f"stress.{s}":
            return s
    return None


def gemm_rows(raw: Path, common: dict[str, str], fname: str, profile_id: str, runtime_note: str, replay_profile: bool, rerun_fname: str | None = None) -> list[dict[str, Any]]:
    d = load(raw, fname)
    if d is None:
        return []
    probe = load(raw, "cublas_probe.json") or {}
    serial = {(r["label"], r["M"]): r for r in probe.get("serial_splitk", [])}
    # cases re-run after a diagnosis (diagnosis.md) replace the original case with the same
    # (label, family, M, path); the original file is kept untouched under raw/
    rerun = load(raw, rerun_fname) if rerun_fname else None
    rerun_cases = {(c["label"], c.get("family"), c["M"], c.get("path", "linear")): c for c in (rerun or {}).get("cases", [])}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for c in d["cases"]:
        key_c = (c["label"], c.get("family"), c["M"], c.get("path", "linear"))
        rerun_used = key_c in rerun_cases
        orig_det = c.get("gpu_deterministic_rerun")
        if rerun_used:
            c = {**rerun_cases[key_c], "rerun": True}
        shape = _shape_of(c["label"])
        if shape is None:  # synthetic K=1040 shape
            shape = "synthetic_k1040_n256_bias"
            K, N, has_bias = c["K"], c["N"], c["bias"]
        else:
            K, N, has_bias = SHAPES[shape]
        key = (shape, phase_of(c["M"]))
        g = groups.setdefault(key, {"K": K, "N": N, "bias": has_bias, "kernels": set(), "models": set(), "nat": [0, 0], "stress": [0, 0], "domain": set(), "seconds": [], "unresolved": 0, "present": 0, "first_mismatch": None, "weak": 0, "Ms": set(), "paths": set(), "unclaimed": [0, 0], "unclaimed_kernels": set(), "rerun": [], "nondeterministic": 0})
        g["Ms"].add(c["M"])
        if rerun_used:
            g["rerun"].append(f"{c.get('family')} M={c['M']}")
        if orig_det is False:
            g["nondeterministic"] += 1
        g["paths"].add(c.get("path", "linear"))
        for k in c["kernels"]:
            if "Memcpy" not in k:
                g["kernels"].add(k.split("(")[0][:120])
        exact, attempted = c["exact"], c["attempted"]
        mf = c.get("model_found")
        if mf is None and (c["label"], c["M"]) in serial and serial[(c["label"], c["M"])]["model_found"]:
            pr = serial[(c["label"], c["M"])]
            mf = pr["model_found"]
            exact, attempted = pr["exact"], pr["attempted"]
            c = {**c, "cpu_seconds_evaluated_rows": pr["cpu_seconds_full_gemm"], "rows_evaluated": c["M"], "first_mismatch": None}
        fam = c.get("family")
        if fam in (None, "natural"):
            fam = None
        # the resolved-kernel zoo decides whether this launch signature has a claimed model; a
        # weak by-search fit on an unresolved kernel (gemvx at M=1) is not a claim
        claimed = Z.resolve(c["kernels"], c["K"], c["M"], c["bias"]) is not None
        if not claimed:
            g["unresolved"] += 1
            g["unclaimed"][0] += exact
            g["unclaimed"][1] += attempted
            for k in c["kernels"]:
                if "Memcpy" not in k and "Memset" not in k:
                    g["unclaimed_kernels"].add("cublas internal::gemvx::kernel" if "gemvx" in k else k.split("<")[0].split("(")[0][:60])
            if g["first_mismatch"] is None and c.get("first_mismatch"):
                g["first_mismatch"] = {"label": c["label"], "family": fam, "M": c["M"], **c["first_mismatch"]}
            continue
        if fam is None:
            g["nat"][0] += exact
            g["nat"][1] += attempted
        else:
            g["stress"][0] += exact
            g["stress"][1] += attempted
            g["domain"].add(fam)
        g["present"] += 1
        if mf is not None:
            g["models"].add(mf["description"].split("; ", 1)[1].rsplit("; ", 1)[0] if "; " in mf["description"] else mf["description"])
            if c.get("weak_evidence"):
                g["weak"] += 1
        elif g["first_mismatch"] is None and c.get("first_mismatch"):
            g["first_mismatch"] = {"label": c["label"], "family": fam, "M": c["M"], **c["first_mismatch"]}
        if c.get("cpu_seconds_evaluated_rows") and c.get("rows_evaluated"):
            g["seconds"].append(c["cpu_seconds_evaluated_rows"] / c["rows_evaluated"])
    rows = []
    for (shape, phase), g in sorted(groups.items(), key=lambda kv: (list(SHAPES).index(kv[0][0]) if kv[0][0] in SHAPES else 99, [p[0] for p in PHASES].index(kv[0][1]))):
        K, N, has_bias = g["K"], g["N"], g["bias"]
        nat_ok = g["nat"][0] == g["nat"][1]
        st_ok = g["stress"][0] == g["stress"][1]
        kernels = sorted(g["kernels"])
        unresolved_reason = None
        if g["unresolved"]:
            why = [Z.why_unresolved([k]) for k in kernels]
            gemv = [w for w in why if w and "gemvx" in w]
            unresolved_reason = (gemv[0] if gemv else "launch signature " + ", ".join(sorted(g["unclaimed_kernels"])) + " not in the resolved zoo") + f"; {g['unresolved']} case(s) not claimed (the single-chain model happens to agree on {g['unclaimed'][0]}/{g['unclaimed'][1]} of their coordinates; not counted)"
        if g["present"] == 0:
            checker, status = "none", "unresolved"
        elif nat_ok and st_ok:
            checker, status = "independent", "exact"
        else:
            checker, status = "independent", "mismatch"
        domain = sorted(g["domain"])
        near = "near_overflow_rows" in domain
        lim = []
        if unresolved_reason:
            lim.append(unresolved_reason)
        if g["rerun"]:
            lim.append("cases re-run after diagnosis (diagnosis.md, raw/w4b/gemm_cublas_near_overflow_rerun.json; regenerated operands, same generator and seed): " + ", ".join(g["rerun"]) + " -- the original mismatches were all the BF16 NaN word: the GPU writes the cvt.rn.bf16.f32 canonical NaN 0x7fff where the model wrote torch's CPU 0x7fc0, on coordinates whose +inf and -inf split/slice partials meet in the FP32 reduction; model NaN word set to 0x7fff; every other coordinate (finite, +inf, -inf) already agreed; GPU output bitwise deterministic (6 runs) and independent of the output buffer's prior content")
        if near and not st_ok:
            lim.append("near_overflow_rows family: the split/slice model and the GPU still differ on some coordinates after the NaN-word fix (see first mismatch); single chains saturate to +-inf and stay there on the GPU as in the model")
        if g["nondeterministic"]:
            lim.append(f"{g['nondeterministic']} case(s) flagged gpu_deterministic_rerun=false by the pre-d0e018b script (torch.equal on BF16 tensors, NaN != NaN); the bitwise rerun check (raw/w4b/det_check.txt, overflow_diagnosis.json) finds the GPU output identical across runs")
        if g["weak"]:
            lim.append(f"{g['weak']} small-M cases (<65536 coordinates) confirm the split/slice description only weakly on their own; the description is fixed by the same kernel's M=64..100 cases and the stress families")
        if not replay_profile:
            lim.append("torch default cuBLAS/cuBLASLt path (VLLM_BATCH_INVARIANT unset); the POC profile does not dispatch these kernels")
        if shape.startswith("synthetic"):
            lim.append("synthetic non-Qwen shape used to force a biased split-K launch; no candidate matched the cuBLASLt bias-in-reduce epilogue")
        first_mm = g["first_mismatch"]
        if first_mm and status != "exact":
            lim.append(f"first mismatch: {first_mm['label']} {first_mm.get('family') or 'natural'} M={first_mm['M']} row {first_mm['row']} col {first_mm['col']} gpu 0x{first_mm['gpu_bf16']} model 0x{first_mm['model_bf16']} ({first_mm['mismatching_elements']} elements differ in that case)")
        secs = sum(g["seconds"]) / len(g["seconds"]) if g["seconds"] else 0.0
        rows.append({
            **common,
            "profile_id": profile_id,
            "op_id": shape,
            "phase": phase,
            "shape": {"M": sorted(g["Ms"]), "K": K, "N": N, "bias": has_bias},
            "dtype": "bf16 x bf16 -> fp32 acc -> bf16",
            "layout_or_quant": "row-major activations x [N,K] weight (tn); unquantized",
            "runtime_impl": ("; ".join(kernels) if kernels else "unresolved") + f" [{runtime_note}; paths {sorted(g['paths'])}]",
            "primitive_semantics": ("gemm.bf16.v1: " + " | ".join(sorted(g["models"]))) if g["models"] else "none",
            "decomposition": ROW_CLASS_GEMM,
            "interface": interface(K, N, has_bias),
            "origin": "captured",
            "binding": {"present": g["present"], "unresolved": g["unresolved"]},
            "checker": checker,
            "natural_checks": {"exact": g["nat"][0], "attempted": g["nat"][1]},
            "stress_checks": {"exact": g["stress"][0], "attempted": g["stress"][1], "domain": (", ".join(domain) + "; finite outputs except near_overflow_rows") if domain else None},
            "mutations": {"rejected": 0, "attempted": 0},
            "checker_work": {"seconds": secs, "flops_or_bytes": 2.0 * K * N, "includes_full_op": False},
            "status": status,
            "limitation": "; ".join(lim) if lim else None,
        })
    return rows


# --- replay rows (NOT_RUN) ----------------------------------------------------------------------------


def engine_capture_rows(raw: Path, common: dict[str, str], profile_id: str) -> list[dict[str, Any]]:
    """Row class ``the same GEMM inside the replayed model`` from this lane's own engine-hook
    fixtures (``engine_gemm_capture.py`` + ``engine_gemm_check.py``): the operands the eager
    (replay-configuration) engine fed ``matmul_kernel_persistent`` for Qwen2.5-1.5B on the A100,
    checked with ``gemm.bf16.v1`` / ``lm_head.bf16.v1`` through W2's registry interface."""
    rows = []
    for fname, variant in (("engine_capture/check_cap.json", "full prefill"), ("engine_capture/check_cap_prefixcached.json", "prefix-cached prefill")):
        d = load(raw, fname)
        if d is None:
            continue
        meta = d["capture_meta"]
        comp = meta["compilation"]
        for s in d["summary"]:
            proj = s["proj"]
            K, N = s["K"], s["N"]
            first = s.get("first_mismatch")
            lim = [f"this lane's own evidence for the row class, on the A100 pod: engine-hook fixture ({variant}; {len(meta['prompt_lens'])} prompts, prompt lengths {meta['prompt_lens']}, max_tokens {meta['max_tokens']}) of the eager replay configuration of the pinned nightly (not W1's replay+commit run: no trace replay, no commitments; the operands are what the engine fed matmul_kernel_persistent); W1's B1 bundle (B1-c37c3f0b8363-20260907T1036Z, RTX 4090 pod) is the NOT_RUN row"]
            if first:
                lim.append(f"first mismatch: seq {first['seq']} {first['layer']} row {first['i']} col {first['j']} gpu {first['gpu']} model {first['model']} ({first['n_mismatch']} in that call)")
            rows.append({
                **common,
                "profile_id": profile_id,
                "op_id": proj,
                "phase": (f"sampled positions M={','.join(str(m) for m in s['M_values'])}, all {s['calls']} steps" if proj == "lm_head" else f"{s['phase']} M={','.join(str(m) for m in s['M_values'])}") + f" (engine, eager; {variant})",
                "shape": {"M": s["M_values"], "K": K, "N": N, "bias": False},
                "dtype": "bf16 x bf16 -> fp32 acc -> bf16",
                "layout_or_quant": "row-major activations x weight.t() view of the [N,K] parameter; unquantized",
                "runtime_impl": f"verity.bi_mm -> vllm.model_executor.determinism.batch_invariant.matmul_persistent (matmul_kernel_persistent) inside the engine: vllm {meta['vllm']}, enforce_eager={comp['enforce_eager']}, cudagraph_mode={comp['cudagraph_mode']}, custom_ops={comp['custom_ops']}, model_runner={comp['model_runner']}; interception = {meta['interception'].split(' (')[0]}",
                "primitive_semantics": ("lm_head.bf16.v1" if proj == "lm_head" else "gemm.bf16.v1") + ": " + d["model"]["spec"] + f"; per-instruction {d['model']['per_instruction']}",
                "decomposition": ROW_CLASS_REPLAY,
                "interface": interface(K, N, False),
                "origin": "captured",
                "binding": {"present": s["calls"], "unresolved": 0},
                "checker": "independent",
                "natural_checks": {"exact": s["exact"], "attempted": s["attempted"]},
                "stress_checks": {"exact": 0, "attempted": 0, "domain": None},
                "mutations": {"rejected": 0, "attempted": 0},
                "checker_work": {"seconds": s["cpu_seconds_per_row"], "flops_or_bytes": 2.0 * K * N, "includes_full_op": False},
                "status": "exact" if s["exact"] == s["attempted"] else "mismatch",
                "limitation": "; ".join(lim) + f"; registry row VUs (W2 NodeInstance/Value objects through verity_vllm.relations.REGISTRY): {s['vu_exact']}/{s['vu_attempted']} exact; rows {s['rows_exact']}/{s['rows']} exact; {s['calls']} GEMM calls, {s['cpu_seconds']:.1f} CPU s total (OMP 96 threads)",
            })
    return rows


def replay_rows(common: dict[str, str], profile_id: str, reason: str) -> list[dict[str, Any]]:
    rows = []
    for shape, (K, N, has_bias) in SHAPES.items():
        rows.append({
            **common,
            "profile_id": profile_id,
            "op_id": shape,
            "phase": "decode+prefill (W1 B1 capture)",
            "shape": {"M": "per step", "K": K, "N": N, "bias": has_bias},
            "dtype": "bf16 x bf16 -> fp32 acc -> bf16",
            "layout_or_quant": "row-major activations x [N,K] weight (tn); unquantized",
            "runtime_impl": "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent (engine-observed on the A100 under VLLM_BATCH_INVARIANT=1, eager and compiled)",
            "primitive_semantics": "gemm.bf16.v1 / lm_head.bf16.v1 / gemm.bf16.coordinate.v1 (verity_vllm_numerics.relations, registered on verity_vllm.relations)",
            "decomposition": ROW_CLASS_REPLAY,
            "interface": interface(K, N, has_bias),
            "origin": "captured",
            "binding": {"present": 0, "unresolved": 0},
            "checker": "independent",
            "natural_checks": {"exact": 0, "attempted": 0},
            "stress_checks": {"exact": 0, "attempted": 0, "domain": None},
            "mutations": {"rejected": 0, "attempted": 0},
            "checker_work": {"seconds": 0.0, "flops_or_bytes": 2.0 * K * N, "includes_full_op": False},
            "status": "NOT_RUN",
            "limitation": reason,
        })
    return rows


# --- beyond GEMM --------------------------------------------------------------------------------------


def beyond_rows(raw: Path, common: dict[str, str], profile_id: str) -> list[dict[str, Any]]:
    d = load(raw, "nightly/beyond_gemm.json") or load(raw, "beyond_gemm.json")
    if d is None:
        return []
    rows = []
    H = 1536
    s = d.get("silu_and_mul", {})
    if "natural" in s:
        att = sum(c["attempted"] for c in s["natural"])
        ex = sum(c["exact_table_model"] for c in s["natural"])
        ex_ieee = sum(c["exact_pure_ieee_model"] for c in s["natural"])
        secs = sum(c["cpu_seconds_table_model"] for c in s["natural"]) / max(1, sum(c["M"] for c in s["natural"]))
        rows.append({
            **common, "profile_id": profile_id,
            "op_id": "silu_and_mul", "phase": "decode+prefill (natural activations M=7,64,512; layers 0,13,27)",
            "shape": {"M": [c["M"] for c in s["natural"] if c["label"].startswith("layers.0")], "D": 8960, "input": "[M, 2*8960]"},
            "dtype": "bf16 -> fp32 -> bf16", "layout_or_quant": "gate|up concatenated",
            "runtime_impl": s["kernel"] + " (eager/replay profile, custom_ops=all; the compiled production profile fuses it into an Inductor Triton kernel triton_poi_fused_mul_silu_slice)",
            "primitive_semantics": "silu_and_mul.bf16.v1: " + s["model"] + f"; the table equals numpy float32 x/(1+exp(-x)) on all {s['table_entries_finite']} finite bf16 inputs ({s['table_vs_ieee_correctly_rounded']['differing_entries']} entries differ from the fp64-correctly-rounded value: x <= -89 where 1+expf(-x) overflows to inf and the GPU returns -0)",
            "decomposition": "one output row (elementwise)",
            "interface": {"input_bytes": 2 * 2 * 8960, "output_bits": 16 * 8960, "range": None},
            "origin": "captured", "binding": {"present": len(s["natural"]), "unresolved": 0},
            "checker": "independent",
            "natural_checks": {"exact": ex, "attempted": att},
            "stress_checks": {"exact": s["table_entries_finite"] - 0, "attempted": s["table_entries_finite"], "domain": "exhaustive: every finite bf16 gate value with up=1.0 (the table itself), rerun deterministic; NaN gate maps to 0x7fff"},
            "mutations": {"rejected": 0, "attempted": 0},
            "checker_work": {"seconds": secs, "flops_or_bytes": 4.0 * 8960, "includes_full_op": False},
            "status": "exact" if ex == att else "mismatch",
            "limitation": f"pure IEEE model (bf16(f32(bf16(x/(1+expf(-x)))) * up)) exact on {ex_ieee}/{att} as well; the 65536-entry table is the pinned relation",
        })
    r = d.get("rms_norm", {})
    rs = load(raw, "nightly/rsqrt_probe.json") or {}
    rt = load(raw, "nightly/rmsnorm_table_probe.json") or {}
    if "cases" in r:
        for key, kname in (("triton", r["kernels"]["triton"]), ("cuda_rms_norm", r["kernels"]["cuda_rms_norm"]), ("cuda_fused_add_rms_norm", r["kernels"]["cuda_fused_add_rms_norm"])):
            rows_n = sum(c[key]["rows"] for c in r["cases"])
            rows_ieee = sum(c[key]["rows_exact_with_ieee_rsqrt"] for c in r["cases"])
            rows_k = sum(c[key]["rows_exact_with_some_k"] for c in r["cases"])
            el = sum(c[key]["elements"] for c in r["cases"])
            el_ieee = sum(c[key]["elements_exact_with_ieee_rsqrt"] for c in r["cases"])
            hist: dict[str, int] = defaultdict(int)
            for c in r["cases"]:
                for k, v in c[key]["k_histogram"].items():
                    hist[k] += v
            best = (rt.get("best") or {}).get(key)
            tab = rs.get("rsqrtf" if key != "triton" else "triton_inv_sqrt") or {}
            explicit = None
            if best:
                b = best["best"]
                explicit = {"rows_exact": b["rows_exact"], "rows": b["rows"], "elements_exact": b["elements_exact"], "elements": b["elements"], "candidates": best["n_candidates"], "all_rows_exact": best["candidates_all_rows_exact"], "best": f"variance={b['variance']} mean={b['mean']} rsqrt={b['rsqrt']} output={b['output']}", "first_mismatch": b["first_mismatch"]}
            resolved = bool(explicit and explicit["all_rows_exact"])
            sem_pinned = ("rmsnorm.bf16.v1 (fused_add_rms_norm): residual' = bf16(x + residual) (one fp32 add, RNE); 192 threads x one 8-vector: p_t = sum over 4 bf16x2 pairs of (a^2 + b^2) accumulated sequentially in fp32; cub BlockReduce<1024>: per-warp shfl_down tree (adjacent pairs 1,2,4,8,16), the 6 warp aggregates added sequentially; mean = var / 1536 (IEEE fp32 division); arg = mean + 1e-6f; s = rsqrtf(arg) = MUFU.RSQ, evaluated from the characterised 2^24-entry int8 ulp-offset table "
                          f"(sha256 {tab.get('sha256_int8_table', '?')}, max |err| {tab.get('max_abs_ulp_error', '?')} ulp, {tab.get('fraction_correctly_rounded', 0):.3f} correctly rounded, scaling-invariant on 12 other binades); out = bf16((x' * s) * w) with two fp32 multiplies and one RNE")
            rows.append({
                **common, "profile_id": profile_id,
                "op_id": f"rms_norm[{key}]", "phase": "decode+prefill (natural + synthetic residual rows M=7,64,512; explicit-sequence search on 2048 rows)",
                "shape": {"M": [7, 64, 512], "H": H}, "dtype": "bf16 (fp32 internal) -> bf16", "layout_or_quant": None,
                "runtime_impl": kname + (" (eager/replay profile: 55 of 56 norms per forward; the compiled production profile uses Inductor triton_red_fused_fused_add_rms_norm)" if key == "cuda_fused_add_rms_norm" else (" (eager/replay profile: layers.0.input_layernorm, 1 of 56 norms per forward)" if key == "triton" else " (not launched by the engine in either configuration; unvectorised sibling of the fused kernel)")),
                "primitive_semantics": sem_pinned if resolved else "rms_norm.ieee.v0 (not pinned): fp32 sum of squares, mean, +eps, 1/sqrt, x*w*inv per element in fp32, RNE to bf16; best explicit candidate: " + (explicit["best"] if explicit else "n/a"),
                "decomposition": "one output row", "interface": {"input_bytes": 2 * H + 2 * H + (2 * H if key == "cuda_fused_add_rms_norm" else 0), "output_bits": 16 * H, "range": None},
                "origin": "captured", "binding": {"present": len(r["cases"]) if resolved else 0, "unresolved": 0 if resolved else len(r["cases"])},
                "checker": "independent" if resolved else "none",
                "natural_checks": {"exact": explicit["elements_exact"] if resolved else 0, "attempted": explicit["elements"] if resolved else 0},
                "stress_checks": {"exact": 0, "attempted": 0, "domain": None},
                "mutations": {"rejected": 0, "attempted": 0},
                "checker_work": {"seconds": 0.0, "flops_or_bytes": 4.0 * H, "includes_full_op": False},
                "status": "exact" if resolved else "unresolved",
                "limitation": (
                    (f"explicit sequence found by rmsnorm_table_probe.py: {explicit['rows_exact']}/{explicit['rows']} rows, {explicit['elements_exact']}/{explicit['elements']} elements exact on 3 layers' natural rows + synthetic outlier rows (1536 + 512 rows); {len(explicit['all_rows_exact'])} of {explicit['candidates']} candidates fit all rows ({'; '.join(explicit['all_rows_exact'])}) -- they differ only in the association of the 4 pair sums / the zero start, indistinguishable here; the rsqrt table (16 MiB int8, regenerated by rsqrt_probe.py on the GPU host) is a registered constant identified by its sha256, not committed; checker not yet wired into verity_vllm.relations (rmsnorm.bf16.v1 slot stays unimplemented until W1's description names residual/x/w for this node)")
                    if resolved else
                    (f"exploratory only (no checker claimed): IEEE-sequence model {el_ieee}/{el} elements ({rows_ieee}/{rows_n} rows); every row ({rows_k}/{rows_n}) is reproduced once the inverse-rms scalar moves by k ulp, k histogram {dict(sorted(hist.items()))}; with the characterised approximate rsqrt/div tables the best of {explicit['candidates'] if explicit else 0} explicit reduction-order candidates reaches {explicit['rows_exact'] if explicit else '?'}/{explicit['rows'] if explicit else '?'} rows ({explicit['best'] if explicit else ''}; first mismatch {explicit['first_mismatch'] if explicit else ''}) -- the fp32 summation order of this kernel is not yet pinned")
                ),
            })
    ro = d.get("rotary_embedding", {})
    if "natural" in ro:
        q_att = sum(c["q_attempted"] + c["k_attempted"] for c in ro["natural"])
        q_ex = sum(c["q_exact_with_opened_cache"] + c["k_exact_with_opened_cache"] for c in ro["natural"])
        q_f64 = sum(c["q_exact_with_float64_recomputed_cache"] for c in ro["natural"])
        q_only = sum(c["q_attempted"] for c in ro["natural"])
        secs = sum(c["cpu_seconds"] for c in ro["natural"]) / max(1, sum(c["M"] for c in ro["natural"]))
        rows.append({
            **common, "profile_id": profile_id,
            "op_id": "rotary_embedding", "phase": "decode+prefill (natural q/k rows, positions 0..27511)",
            "shape": {"M": [7, 64, 512], "heads_q": 12, "heads_k": 2, "head_dim": 128, "rotary_dim": 128}, "dtype": "bf16 (fp32 internal) -> bf16", "layout_or_quant": "NeoX (pairs i, i+64), in place",
            "runtime_impl": ro["kernel"] + " (eager/replay profile; the compiled production profile uses an Inductor Triton kernel triton_poi_fused_add_cat_index_select_mul_...)",
            "primitive_semantics": "rope.neox.bf16.v1: " + ro["model"] + "; cos_sin_cache is an opened derived leaf ([131072,128] bf16 computed on the GPU at model load)",
            "decomposition": "one (position, head) row", "interface": {"input_bytes": 2 * 128 + 2 * 128, "output_bits": 16 * 128, "range": None},
            "origin": "captured", "binding": {"present": len(ro["natural"]), "unresolved": 0},
            "checker": "independent",
            "natural_checks": {"exact": q_ex, "attempted": q_att},
            "stress_checks": {"exact": 0, "attempted": 0, "domain": None},
            "mutations": {"rejected": 0, "attempted": 0},
            "checker_work": {"seconds": secs, "flops_or_bytes": 6.0 * 128 * 14, "includes_full_op": False},
            "status": "exact" if q_ex == q_att else "mismatch",
            "limitation": f"exact only with the opened cos_sin_cache: the cache differs from an fp64 recompute in {ro['cache']['differs_from_exact_float64_recompute']}/{ro['cache']['entries']} entries and from a torch-fp32 step emulation in {ro['cache']['differs_from_torch_float32_step_emulation']}; with the fp64 cache q is exact on {q_f64}/{q_only} -- W1 must commit the cache as a derived constant leaf",
        })
    rows.append({
        **common, "profile_id": profile_id,
        "op_id": "attention", "phase": "decode+prefill", "shape": {"heads_q": 12, "heads_k": 2, "head_dim": 128}, "dtype": "bf16", "layout_or_quant": "paged KV, FlashAttention-2 splitkv",
        "runtime_impl": "flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, ...bfloat16_t>> (engine-observed, eager and compiled)",
        "primitive_semantics": "none", "decomposition": "whole_operation", "interface": {"input_bytes": 0, "output_bits": 16 * 128, "range": None},
        "origin": "captured", "binding": {"present": 0, "unresolved": 1}, "checker": "none",
        "natural_checks": {"exact": 0, "attempted": 0}, "stress_checks": {"exact": 0, "attempted": 0, "domain": None}, "mutations": {"rejected": 0, "attempted": 0},
        "checker_work": {"seconds": 0.0, "flops_or_bytes": 0.0, "includes_full_op": False},
        "status": "unresolved",
        "limitation": "not attempted: FA2 online softmax uses exp2f (approximate MUFU.EX2), a split-KV partial reduction and mma.sync accumulation over a data-dependent tile schedule; an independent model needs the same characterisation as rms_norm plus the kernel's tile order -- same_runtime is the only executable relation for now",
    })
    return rows


# --- manifest ---------------------------------------------------------------------------------------------


def manifest(raw: Path, experiment: str, revision: str, profiles: dict[str, str]) -> dict[str, Any]:
    tri = load(raw, "nightly/gemm_vllm_triton.json") or {}
    cub = load(raw, "gemm_cublas.json") or {}
    eng = load(raw, "nightly/engine_kernels_eager.json") or {}
    engc = load(raw, "nightly/engine_kernels_compiled.json") or {}
    val = load(raw, "validate_bf16.json") or {}
    return {
        "schema": "verity-vllm/numerics-manifest/v1",
        "experiment_id": experiment,
        "worker": WORKER,
        "case": CASE,
        "revision": revision,
        "gpu": {"name": (val.get("device") or {}).get("name") or cub.get("runtime", {}).get("gpu"), "sm": (val.get("device") or {}).get("sm", "sm_80"), "driver": "580 (RunPod A100-SXM4-80GB)", "cuda_toolkit": "12.8 (nvcc)"},
        "software": {
            "torch": cub.get("runtime", {}).get("torch"),
            "vllm_pinned": tri.get("runtime", {}).get("vllm"),
            "vllm_also_measured": "0.28.0 (stable; identical matmul_kernel_persistent source lines 1-101; 199/199 cases exact there too)",
            "triton": eng.get("triton"),
            "python": "3.12.14 (uv-managed venvs /workspace/venv-nightly, /workspace/venv-vllm)",
            "allow_bf16_reduced_precision_reduction": cub.get("runtime", {}).get("allow_bf16_reduced_precision_reduction"),
        },
        "checkpoint": cub.get("checkpoint") or tri.get("checkpoint"),
        "profiles": profiles,
        "engine_flags_observed": {"eager": eng.get("env"), "compiled": engc.get("env")},
        "engine_compilation_observed": {"eager": eng.get("compilation"), "compiled": engc.get("compilation")},
        "engine_gemm_kernels_observed": {"eager": eng.get("gemm_kernels"), "compiled": engc.get("gemm_kernels")},
        "per_instruction_model": cub.get("per_instruction_model") or tri.get("per_instruction_model"),
        "commands": [
            "python -m verity_vllm_numerics.validate_random --dtype {bf16,f16} --out validate_<dtype>.json",
            "python -m verity_vllm_numerics.hawkeye_sequence --out hawkeye_sequence.json",
            "python -m verity_vllm_numerics.capture_activations --out qwen_acts",
            "python -m verity_vllm_numerics.gemm_production --runtime vllm-triton --acts qwen_acts --out gemm_vllm_triton.json",
            "python -m verity_vllm_numerics.gemm_production --runtime cublas --acts qwen_acts --out gemm_cublas.json",
            "python -m verity_vllm_numerics.cublas_probe --acts qwen_acts --out cublas_probe.json",
            "python -m verity_vllm_numerics.resolve_engine_kernels engine_kernels_eager.json --eager",
            "python -m verity_vllm_numerics.resolve_engine_kernels engine_kernels_compiled.json",
            "python -m verity_vllm_numerics.beyond_gemm --acts qwen_acts --out beyond_gemm.json",
            "VLLM_BATCH_INVARIANT=1 python -m verity_vllm_numerics.engine_gemm_capture engine_capture/cap --max-tokens 6   # and --warmup-same -> cap_prefixcached",
            "OMP_NUM_THREADS=96 python -m verity_vllm_numerics.engine_gemm_check engine_capture/cap engine_capture/check_cap.json --rows-per-call 4",
            "python -m verity_vllm_numerics.rsqrt_probe --out rsqrt_probe.json --table-out rsqrt_tables.npz",
            "python -m verity_vllm_numerics.rmsnorm_table_probe --acts qwen_acts --tables rsqrt_tables.npz --out rmsnorm_table_probe.json",
            "python -m verity_vllm_numerics.overflow_diagnosis --out w4b/overflow_diagnosis.json --save-npz w4b/overflow_repro   # diagnosis.md",
            "python -m verity_vllm_numerics.gemm_production --runtime cublas --no-natural --families near_overflow_rows --no-synthetic --stress-ms 7,64,100 --stress-shapes qkv_proj,down_proj --out w4b/gemm_cublas_near_overflow_rerun.json",
            "PYTHONPATH=. python -m pytest tests/poc/test_numerics_relations.py -q   # CPU only; coverage gate + C++ chain vs pure-Python chain on a tiny shape (8 passed on the pod with W2's verity_vllm importable, 7 passed + 1 skipped without)",
        ],
        "diagnosis": "diagnosis.md (the four cuBLAS near_overflow_rows mismatches: BF16 NaN word 0x7fff vs 0x7fc0; model fixed, cases re-run exact)",
        "engine_capture": {k: ({kk: vv for kk, vv in (load(raw, f"engine_capture/{k}.json") or {}).get("capture_meta", {}).items() if kk != "env"} | {"n_calls": (load(raw, f"engine_capture/{k}.json") or {}).get("capture_meta", {}).get("n_calls"), "total_cpu_seconds": (load(raw, f"engine_capture/{k}.json") or {}).get("total_seconds")}) for k in ("check_cap", "check_cap_prefixcached") if load(raw, f"engine_capture/{k}.json")},
        "raw_files": sorted(str(p.relative_to(raw)) for p in raw.rglob("*.json")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--replay-reason", default="NOT_RUN: tensors not transferable in time. W1's B1 bundle B1-c37c3f0b8363-20260907T1036Z (cursor/vllm-poc-adapter-7e60, RTX 4090 pod 55e08e136513, sm_89, eager, dispatch-mode) commits every GEMM row (commitments.json: leaf hashes; description.json: 102 gemm nodes with runtime_op verity.bi_mm; its Table 2 has 102 gemm + 2 lm_head rows, all exact under gemm.same_runtime.v1 / lm_head.same_runtime.v1, and verify.json's relations_installed does not include gemm.bf16.v1) but the opened tensors stay on that pod; gemm.bf16.v1 / lm_head.bf16.v1 were not executed on those openings. The lane's evidence for this row class is the A100 engine-hook fixture rows above (same kernel, same model, this pod). Ada (sm_89) BF16 mma.sync is parameter-identical to Ampere's (docs/hardware-semantics.md) but that GPU's production GEMM was not measured by this lane.")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = {
        "a100-mma-sm80": "direct mma.sync probe, nvcc 12.8, no framework",
        "a100-vllm-bi1": "A100 sm_80, torch 2.13.0+cu130, vllm 0.28.1rc1.dev472+gd9105ea80 (pinned), VLLM_BATCH_INVARIANT=1 (matmul_kernel_persistent) -- the POC profile's GEMM path; beyond-GEMM ops as launched by the eager (replay) configuration",
        "a100-torch-cublas": "A100 sm_80, torch 2.13.0+cu130 default F.linear (cuBLAS/cuBLASLt heuristics, allow_bf16_reduced_precision_reduction=True), no batch invariance -- not the POC profile",
    }
    common = {"experiment_id": args.experiment, "case": CASE, "profile_id": "a100-mma-sm80", "worker": WORKER, "adapter_revision": args.revision}
    rows: list[dict[str, Any]] = []
    rows += mma_rows(raw, common)
    rows += gemm_rows(raw, common, "nightly/gemm_vllm_triton.json", "a100-vllm-bi1", "vLLM batch-invariant Triton matmul_kernel_persistent, pinned nightly", True)
    rows += gemm_rows(raw, common, "gemm_cublas.json", "a100-torch-cublas", "torch default cuBLAS/cuBLASLt", False, rerun_fname="w4b/gemm_cublas_near_overflow_rerun.json")
    rows += engine_capture_rows(raw, common, "a100-vllm-bi1")
    rows += replay_rows(common, "a100-vllm-bi1", args.replay_reason)
    rows += beyond_rows(raw, common, "a100-vllm-bi1")
    with (out / "table2.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest(raw, args.experiment, args.revision, profiles), indent=1) + "\n")
    print(f"{len(rows)} rows -> {out / 'table2.jsonl'}")
    for r in rows:
        print(f"  {r['profile_id']:18s} {r['op_id']:28s} {r['phase'][:26]:26s} {r['checker']:11s} nat {r['natural_checks']['exact']}/{r['natural_checks']['attempted']} stress {r['stress_checks']['exact']}/{r['stress_checks']['attempted']} {r['status']}")


if __name__ == "__main__":
    main()
