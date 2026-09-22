"""Diagnosis of the four ``mismatch`` rows of Table 2 (cuBLAS profile, ``near_overflow_rows``
stress family; contract §15.5 order: identity/shape -> kernel path -> arithmetic).  GPU host only.

What it does, per shape (qkv_proj K=1536 N=2048 bias; down_proj K=8960 N=1536) and M in {7, 64}:

1. regenerates the ``near_overflow_rows`` operands (``gemm_production.stress_families`` generator,
   own seed) -- rows whose FP32 accumulator overflows: every product of the ``big`` rows is
   ~2^226..2^254, far outside FP32;
2. **kernel path**: records the kernel names cuBLAS dispatches (torch.profiler) and resolves them
   in the kernel zoo (same as the original run); runs the same GEMM under vLLM's batch-invariant
   ``matmul_kernel_persistent`` (single chain) for a hardware reference of the same operands;
3. **run-to-run**: 5 cuBLAS reruns bitwise; and the output buffer dependence test -- ``out=`` a
   buffer pre-filled with zeros / +inf / NaN / a finite pattern (does the kernel read C with
   beta = 0?);
4. **arithmetic**: the resolved CPU model's output plus its per-split FP32 partials, and a
   classification of every coordinate by (gpu class, model class) with class in
   {finite, +inf, -inf, NaN}; for NaN outputs the exact 16-bit pattern; the mismatches are
   split into ``nan_encoding`` (both NaN, different word), ``gpu_nan_model_inf`` (and the
   partial signs the model had there), ``gpu_inf_model_nan``, ``other``;
5. **direct MMA** (background, one instruction): ``mma.sync.m16n8k16`` with a non-finite
   accumulator input (+inf/-inf/NaN) and product groups summing to small/huge positive/negative,
   plus products that individually overflow FP32 -- the propagation rules the chain model assumes.

Output: one JSON with all counts and the first mismatching coordinates (bit patterns) per class.

    python -m verity_vllm_numerics.overflow_diagnosis --out /workspace/results/numerics/overflow_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from . import cpu_model as CM
from . import kernel_zoo as Z
from .gemm_production import SHAPES, from_words, kernel_names, stress_families, words
from .mma import AMPERE_BF16

INF_P, INF_N = 0x7F80, 0xFF80


def cls_bf16(w: np.ndarray) -> np.ndarray:
    """0 finite, 1 +inf, 2 -inf, 3 NaN."""
    exp_all = (w & 0x7F80) == 0x7F80
    mant = (w & 0x7F) != 0
    out = np.zeros(w.shape, np.int8)
    out[exp_all & ~mant & ((w >> 15) == 0)] = 1
    out[exp_all & ~mant & ((w >> 15) == 1)] = 2
    out[exp_all & mant] = 3
    return out


def cls_f32(u: np.ndarray) -> np.ndarray:
    exp_all = (u & 0x7F800000) == 0x7F800000
    mant = (u & 0x007FFFFF) != 0
    out = np.zeros(u.shape, np.int8)
    out[exp_all & ~mant & ((u >> 31) == 0)] = 1
    out[exp_all & ~mant & ((u >> 31) == 1)] = 2
    out[exp_all & mant] = 3
    return out


CLS = ["finite", "+inf", "-inf", "nan"]


def hist_pairs(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    h: dict[str, int] = {}
    for i in range(4):
        for j in range(4):
            n = int(np.count_nonzero((a == i) & (b == j)))
            if n:
                h[f"{CLS[i]}|{CLS[j]}"] = n
    return h


def word_hist(w: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(w[mask], return_counts=True)
    return {f"0x{int(v):04x}": int(c) for v, c in zip(vals, counts)}


def split_partials(x: np.ndarray, w: np.ndarray, spec: CM.GemmModelSpec) -> np.ndarray:
    """FP32 bits of every split's chain (before reduction / epilogue): (nsplit, M, N)."""
    ntiles = x.shape[1] // AMPERE_BF16.k
    bounds = list(spec.bounds) if spec.bounds is not None else [ntiles * s // spec.nsplit for s in range(spec.nsplit + 1)]
    order = list(spec.order) if spec.order is not None else list(range(ntiles))
    out = []
    for s in range(spec.nsplit):
        tiles = order[bounds[s] : bounds[s + 1]]
        sub = CM.GemmModelSpec(spec.model_name, nsplit=1, bounds=(0, len(tiles)), order=tuple(tiles), bias_mode=0)
        _, f32 = CM.gemm_model(AMPERE_BF16, x, w, None, sub, want_f32=True)
        out.append(f32)
    return np.stack(out)


def run_cublas_variants(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> dict:
    xt, wt = from_words(x), from_words(w)
    bt = None if b is None else from_words(b)
    names, y0 = kernel_names(lambda: F.linear(xt, wt, bt))
    y0 = words(y0)
    reruns = []
    for _ in range(5):
        y = F.linear(xt, wt, bt)
        torch.cuda.synchronize()
        reruns.append(words(y))
    distinct = {r.tobytes() for r in reruns} | {y0.tobytes()}
    # output-buffer dependence: same op with out= a pre-filled buffer
    M, N = x.shape[0], w.shape[0]
    fills = {
        "zeros": torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"),
        "pos_inf": torch.full((M, N), float("inf"), dtype=torch.bfloat16, device="cuda"),
        "nan": torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda"),
        "finite_pattern": torch.full((M, N), 3.0, dtype=torch.bfloat16, device="cuda"),
    }
    out_dep = {}
    out_names = {}
    for k, buf in fills.items():
        if bt is None:
            nm, _ = kernel_names(lambda: torch.mm(xt, wt.t(), out=buf))
        else:
            nm, _ = kernel_names(lambda: torch.addmm(bt, xt, wt.t(), out=buf))
        torch.cuda.synchronize()
        out_dep[k] = words(buf)
        out_names[k] = sorted(set(nm))
    return {
        "kernels": sorted(set(names)),
        "y_profiled": y0,
        "reruns": reruns,
        "distinct_outputs_in_6_runs": len(distinct),
        "out_prefilled": out_dep,
        "out_prefilled_kernels": out_names,
    }


def triton_reference(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, BI) -> tuple[np.ndarray, str]:
    xt, wt = from_words(x), from_words(w)
    fn = None
    for cand in ("matmul_persistent", "mm_batch_invariant"):
        if hasattr(BI, cand):
            fn = getattr(BI, cand)
            break
    if fn is None:
        raise RuntimeError("no persistent matmul entry point in the batch_invariant module")
    y = fn(xt, wt.t())
    if b is not None:  # linear_batch_invariant: separate bf16 add
        y = y + from_words(b)
    torch.cuda.synchronize()
    return words(y), f"{BI.__name__}.{fn.__name__}"


def first_coords(mask: np.ndarray, gpu: np.ndarray, model: np.ndarray, parts: np.ndarray | None, k: int = 4) -> list[dict]:
    idx = np.argwhere(mask)[:k]
    out = []
    for i, j in idx:
        d = {"row": int(i), "col": int(j), "gpu": f"0x{int(gpu[i, j]):04x}", "model": f"0x{int(model[i, j]):04x}"}
        if parts is not None:
            d["model_partials_f32"] = [f"0x{int(p):08x}" for p in parts[:, i, j]]
        out.append(d)
    return out


def classify(gpu: np.ndarray, model: np.ndarray, parts: np.ndarray | None) -> dict:
    cg, cm = cls_bf16(gpu), cls_bf16(model)
    mism = gpu != model
    res = {
        "attempted": int(gpu.size),
        "exact": int(np.count_nonzero(~mism)),
        "gpu_class_hist": {CLS[i]: int(np.count_nonzero(cg == i)) for i in range(4)},
        "model_class_hist": {CLS[i]: int(np.count_nonzero(cm == i)) for i in range(4)},
        "pair_hist gpu|model": hist_pairs(cg, cm),
        "gpu_nan_words": word_hist(gpu, cg == 3),
        "model_nan_words": word_hist(model, cm == 3),
    }
    nan_enc = mism & (cg == 3) & (cm == 3)
    g_nan_m_inf = mism & (cg == 3) & ((cm == 1) | (cm == 2))
    g_inf_m_nan = mism & ((cg == 1) | (cg == 2)) & (cm == 3)
    finite_mism = mism & (cg == 0) & (cm == 0)
    other = mism & ~(nan_enc | g_nan_m_inf | g_inf_m_nan | finite_mism)
    res["mismatch_classes"] = {
        "nan_encoding (both NaN, different word)": int(np.count_nonzero(nan_enc)),
        "gpu_nan_model_inf": int(np.count_nonzero(g_nan_m_inf)),
        "gpu_inf_model_nan": int(np.count_nonzero(g_inf_m_nan)),
        "both_finite_differ": int(np.count_nonzero(finite_mism)),
        "other": int(np.count_nonzero(other)),
    }
    res["first"] = {
        "nan_encoding": first_coords(nan_enc, gpu, model, parts),
        "gpu_nan_model_inf": first_coords(g_nan_m_inf, gpu, model, parts),
        "gpu_inf_model_nan": first_coords(g_inf_m_nan, gpu, model, parts),
        "both_finite_differ": first_coords(finite_mism, gpu, model, parts),
        "other": first_coords(other, gpu, model, parts),
    }
    if parts is not None:
        cp = cls_f32(parts)  # (S, M, N)
        n_inf = np.count_nonzero((cp == 1) | (cp == 2), axis=0)
        mixed = (np.any(cp == 1, axis=0)) & (np.any(cp == 2, axis=0))
        res["model_partials"] = {
            "coords_with_any_inf_partial": int(np.count_nonzero(n_inf > 0)),
            "coords_with_mixed_sign_inf_partials": int(np.count_nonzero(mixed)),
            "coords_with_nan_partial": int(np.count_nonzero(np.any(cp == 3, axis=0))),
            "gpu_nan_where_model_partials_mixed": int(np.count_nonzero(mixed & (cg == 3))),
            "gpu_nan_where_model_partials_not_mixed": int(np.count_nonzero(~mixed & (cg == 3))),
        }
    return res


def mma_nonfinite_probe() -> dict:
    """One ``mma.sync`` per record: accumulator class x product-group class -> result word."""
    from . import probe as P

    def bf(v: float) -> int:
        return int(P.to_bf16_bits(np.array([v], np.float32))[0])

    big = 0x7F7F  # bf16 max ~3.39e38
    cases = []
    accs = {"+inf": 0x7F800000, "-inf": 0xFF800000, "nan_7fc00000": 0x7FC00000, "nan_7fffffff": 0x7FFFFFFF, "nan_ffc00000": 0xFFC00000, "zero": 0, "one": 0x3F800000, "max_f32": 0x7F7FFFFF}
    prods = {
        "zero_products": ([0] * 16, [0] * 16),
        "small_pos": ([bf(1.0)] + [0] * 15, [bf(1.0)] + [0] * 15),
        "small_neg": ([bf(1.0)] + [0] * 15, [bf(-1.0)] + [0] * 15),
        "huge_pos_single_product_2^252": ([big] + [0] * 15, [big] + [0] * 15),
        "huge_neg_single_product_2^252": ([big] + [0] * 15, [big | 0x8000] + [0] * 15),
        "huge_pos_all16": ([big] * 16, [big] * 16),
        "huge_neg_all16": ([big] * 16, [big | 0x8000] * 16),
        "huge_cancel_exact (2^252 - 2^252)": ([big, big] + [0] * 14, [big, big | 0x8000] + [0] * 14),
        "huge_cancel_group0_vs_group1": ([big] + [0] * 7 + [big] + [0] * 7, [big] + [0] * 7 + [big | 0x8000] + [0] * 7),
        "pos_2^127_x_2^1 (=2^128 overflow by one)": ([0x7F00] + [0] * 15, [bf(2.0)] + [0] * 15),
        "operand_inf_x_1": ([INF_P] + [0] * 15, [bf(1.0)] + [0] * 15),
        "operand_inf_x_0": ([INF_P] + [0] * 15, [0] + [0] * 15),
        "operand_nan_x_1": ([0x7FC0] + [0] * 15, [bf(1.0)] + [0] * 15),
    }
    A = np.zeros((len(accs) * len(prods), 16, 16), np.uint16)
    Bt = np.zeros((len(accs) * len(prods), 8, 16), np.uint16)
    C = np.zeros((len(accs) * len(prods), 16, 8), np.uint32)
    keys = []
    n = 0
    for an, ab in accs.items():
        for pn, (a, b) in prods.items():
            A[n, 0, :] = a
            Bt[n, 0, :] = b
            C[n, 0, 0] = ab
            keys.append((an, pn))
            n += 1
    D = P.mma_tiles("bf16", A, Bt, C)
    res = {}
    for i, (an, pn) in enumerate(keys):
        d = int(D[i, 0, 0])
        res.setdefault(an, {})[pn] = {"d_f32": f"0x{d:08x}", "class": CLS[int(cls_f32(np.array([d], np.uint32))[0])]}
    # the chain model's rule: break at the first saturation, keep the sign -> what does hardware do
    # with acc=+inf and a group summing to -huge?  (covered by ('+inf', 'huge_neg_*') above)
    return {"device": P.device_info(), "results": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-npz", default=None, help="directory for the reproducer operands/outputs (small M only)")
    ap.add_argument("--ms", default="7,64")
    ap.add_argument("--shapes", default="qkv_proj,down_proj")
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    info = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    try:
        import vllm
        from vllm.model_executor.determinism import batch_invariant as BI

        info["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        BI = None
        info["vllm_error"] = repr(e)
    out = {"info": info, "cases": []}
    t_all = time.time()
    for shape in args.shapes.split(","):
        K, N, has_bias = SHAPES[shape]
        for M in [int(v) for v in args.ms.split(",")]:
            fams = {f: (x, w, b) for f, x, w, b in stress_families(rng, M, K, N, has_bias)}
            x, w, b = fams["near_overflow_rows"]
            n_big = max(1, M // 8)
            case = {"shape": shape, "M": M, "K": K, "N": N, "bias": has_bias, "big_rows": list(range(n_big)),
                    "x_exponent_range_big_rows": [int(((x[:n_big] >> 7) & 0xFF).min()), int(((x[:n_big] >> 7) & 0xFF).max())],
                    "w_exponent_range": [int(((w >> 7) & 0xFF).min()), int(((w >> 7) & 0xFF).max())]}
            cu = run_cublas_variants(x, w, b)
            case["cublas_kernels"] = cu["kernels"]
            case["cublas_distinct_outputs_in_6_runs"] = cu["distinct_outputs_in_6_runs"]
            gpu = cu["y_profiled"]
            # run-to-run differences: where and what
            diffs = [int(np.count_nonzero(r != gpu)) for r in cu["reruns"]]
            case["cublas_rerun_diff_counts_vs_profiled"] = diffs
            if any(diffs):
                r = cu["reruns"][int(np.argmax(diffs))]
                case["cublas_rerun_diff_pairs profiled|rerun"] = hist_pairs(cls_bf16(gpu), cls_bf16(r))
                case["cublas_rerun_diff_first"] = first_coords(r != gpu, gpu, r, None)
            # output-buffer dependence
            dep = {}
            base = cu["out_prefilled"]["zeros"]
            for k, v in cu["out_prefilled"].items():
                dep[k] = {"differs_from_zeros_fill": int(np.count_nonzero(v != base)), "kernels": cu["out_prefilled_kernels"][k], "pairs zeros|this": hist_pairs(cls_bf16(base), cls_bf16(v)) if k != "zeros" else None}
            dep["profiled_F_linear_vs_zeros_fill"] = int(np.count_nonzero(gpu != base))
            case["output_buffer_dependence"] = dep
            # zoo-resolved model
            spec = Z.resolve(cu["kernels"], K, M, has_bias)
            case["zoo_spec"] = None if spec is None else spec.to_json()
            if spec is not None:
                # the model as the original run had it (torch's CPU NaN word 0x7fc0) ...
                spec_old = replace(spec, nan_word=CM.TORCH_CPU_NAN_WORD)
                t0 = time.time()
                model, model_f32 = CM.gemm_model(AMPERE_BF16, x, w, b, spec_old, want_f32=True)
                case["model_seconds"] = time.time() - t0
                parts = split_partials(x, w, spec) if spec.nsplit > 1 else None
                case["vs_profiled_F_linear"] = classify(gpu, model, parts)
                case["vs_out_zeros_fill"] = classify(base, model, parts)
                # ... and with the GPU's canonical NaN word (cvt.rn.bf16.f32 -> 0x7fff), now the default
                model_canon, _ = CM.gemm_model(AMPERE_BF16, x, w, b, spec, want_f32=False)
                case["vs_out_zeros_fill_model_nan_0x7fff"] = classify(base, model_canon, parts)
                case["vs_profiled_model_nan_0x7fff"] = classify(gpu, model_canon, parts)
                case["model_f32_nan_count"] = int(np.count_nonzero(cls_f32(model_f32) == 3))
                # the original run's reporting fallback: no candidate matched, so the first mismatch
                # was printed against the single-chain model, which never yields NaN
                single0 = CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=1 if has_bias else 0)
                s0, _ = CM.gemm_model(AMPERE_BF16, x, w, b, single0, want_f32=False)
                case["vs_profiled_single_chain_fallback (original first-mismatch reporting)"] = classify(gpu, s0, None)["mismatch_classes"]
            if BI is not None:
                try:
                    tri, fn = triton_reference(x, w, b, BI)
                    case["triton_entry"] = fn
                    case["triton_vs_model"] = None if spec is None else classify(tri, model, None)["mismatch_classes"]
                    single = CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=3 if has_bias else 0)
                    tm, _ = CM.gemm_model(AMPERE_BF16, x, w, b, single, want_f32=False)
                    case["triton_vs_single_chain_model"] = classify(tri, tm, None)
                    case["triton_vs_cublas_profiled"] = {"pairs triton|cublas": hist_pairs(cls_bf16(tri), cls_bf16(gpu)), "differs": int(np.count_nonzero(tri != gpu))}
                except Exception as e:  # noqa: BLE001
                    case["triton_error"] = repr(e)
            if args.save_npz and M <= 16:
                import os

                os.makedirs(args.save_npz, exist_ok=True)
                np.savez_compressed(os.path.join(args.save_npz, f"near_overflow_{shape}_M{M}.npz"), x=x, w=w, b=b if b is not None else np.zeros(0, np.uint16), cublas_profiled=gpu, cublas_out_zeros=base, cublas_out_nan=cu["out_prefilled"]["nan"], cublas_out_inf=cu["out_prefilled"]["pos_inf"], **({"model": model} if spec is not None else {}))
            out["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k in ("shape", "M", "cublas_kernels", "cublas_distinct_outputs_in_6_runs", "cublas_rerun_diff_counts_vs_profiled")}), flush=True)
            if spec is not None:
                print("   vs profiled:", json.dumps(case["vs_profiled_F_linear"]["mismatch_classes"]), flush=True)
                print("   vs zeros-fill:", json.dumps(case["vs_out_zeros_fill"]["mismatch_classes"]), flush=True)
                print("   vs zeros-fill, model NaN=0x7fff:", json.dumps(case["vs_out_zeros_fill_model_nan_0x7fff"]["mismatch_classes"]), flush=True)
                print("   buffer dependence:", json.dumps({k: v["differs_from_zeros_fill"] for k, v in dep.items() if isinstance(v, dict)}), flush=True)
            if "triton_vs_single_chain_model" in case:
                print("   triton vs single chain:", json.dumps(case["triton_vs_single_chain_model"]["mismatch_classes"]), flush=True)
    try:
        out["mma_nonfinite_probe"] = mma_nonfinite_probe()
        print(json.dumps(out["mma_nonfinite_probe"]["results"], indent=0)[:3000], flush=True)
    except Exception as e:  # noqa: BLE001
        out["mma_nonfinite_probe"] = {"error": repr(e)}
    out["seconds"] = time.time() - t_all
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"DONE in {out['seconds']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
