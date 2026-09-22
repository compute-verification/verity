"""Full production GEMM vs CPU model on Qwen2.5-1.5B shapes (GPU host only).

For each GEMM shape family (merged projections as vLLM runs them), each M and each input
family (natural Qwen activations / structured stress inputs), runs the production runtime:

* ``cublas``      -- stock ``torch.nn.functional.linear`` (cuBLAS / cuBLASLt heuristics);
* ``vllm-triton`` -- vLLM's batch-invariant override (``matmul_kernel_persistent``) as installed by
                     ``enable_batch_invariant_mode()``: ``aten::linear`` (bias added afterwards in
                     BF16) and ``aten::addmm`` (bias added in FP32 inside the kernel);

records the kernel names the runtime dispatched (torch.profiler), checks run-to-run
determinism, and searches the parametrised CPU model (``cpu_model.GemmModelSpec``) for a
bit-exact description: single ascending chain of per-instruction steps first, then split-K
variants with FP32 partial reductions.  Reports exact/attempted counts, the first mismatch
with bit patterns, and the checker's CPU cost.

Usage: python -m verity_vllm_numerics.gemm_production --runtime cublas --acts /workspace/results/numerics/qwen_acts --out results/gemm_cublas.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import cpu_model as CM
from .kernel_zoo import sliced_k_spec
from .mma import AMPERE_BF16

SHAPES = {
    # name: (K, N, has_bias) -- merged as vLLM's Qwen2 model runs them
    "qkv_proj": (1536, 2048, True),
    "o_proj": (1536, 1536, False),
    "gate_up_proj": (1536, 17920, False),
    "down_proj": (8960, 1536, False),
    "lm_head": (1536, 151936, False),
}
DEFAULT_MS = "1,7,16,64,100,512,2048"


def words(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def from_words(w: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).cuda()


def kernel_names(fn) -> tuple[list[str], object]:
    """Run ``fn`` under the profiler; return the CUDA kernel names it launched and its result."""

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = []
    for evt in prof.events():
        if getattr(evt, "device_type", None) is not None and str(evt.device_type).endswith("CUDA"):
            names.append(evt.name)
    return names, out


class Runtime:
    def __init__(self, name: str):
        self.name = name
        self.info = {"runtime": name, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        if name == "vllm-triton":
            import vllm

            try:  # pinned nightly (0.28.1rc1.dev472+gd9105ea80) moved the module
                from vllm.model_executor.determinism import batch_invariant as BI
            except ImportError:  # stable 0.28.0
                from vllm.model_executor.layers import batch_invariant as BI

            BI.enable_batch_invariant_mode()
            self.info["vllm"] = vllm.__version__
            self.info["batch_invariant_module"] = BI.__name__
            self.info["override"] = "enable_batch_invariant_mode(): aten::mm/addmm/matmul/linear -> matmul_kernel_persistent (sm8x)"
            self.info["block_default"] = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8}
            self.info["tuned_configs"] = hasattr(BI, "_get_matmul_config") or "batch_invariant_configs" in str(getattr(BI, "__file__", ""))
        else:
            self.info["allow_bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            self.info["preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library())

    def paths(self, has_bias: bool) -> list[str]:
        if self.name == "vllm-triton":
            return ["linear", "addmm"] if has_bias else ["linear"]
        return ["linear"]

    def run(self, path: str, x, w, b):
        if path == "linear":
            return F.linear(x, w, b)
        if path == "addmm":
            return torch.addmm(b, x, w.t())
        raise ValueError(path)


def candidate_specs(K: int, has_bias: bool, runtime: str, path: str, kernels: list[str] | None = None) -> list[CM.GemmModelSpec]:
    if runtime == "vllm-triton":
        bias_mode = (1 if path == "addmm" else 3) if has_bias else 0
        return [CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=bias_mode, note=f"vLLM matmul_kernel_persistent via aten::{path}")]
    specs = [CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=1 if has_bias else 0, note="no split-K")]
    ntiles16 = K // 16
    bias_modes = (1, 4) if has_bias else (0,)
    hint = " ".join(kernels or [])
    # CTA-internal slice-K (kernel names carry ``slicedAxB``): interleaved k ownership.  The
    # measured sliced1x2 kernels are 32-wide granularity; that goes first.
    sliced = []
    for gran in (2, 1, 4, 8):
        for S in (2, 4, 8):
            for bm in bias_modes:
                for first in ((0, 1) if bm == 4 and S == 2 else (0,)):
                    for red in ((0,) if S == 2 else (0, 1, 2)):
                        spec = sliced_k_spec(K, S, gran, bm, red, first)
                        if spec is not None:
                            sliced.append(spec)
    # grid-level split-K (cuBLASLt splitKreduce_kernel): contiguous K ranges of whole k blocks;
    # FP32 workspace partials (reduce 0/1/2) or BF16 workspace partials (3/4/5).
    bf16_partials = "splitKreduce_kernel<32, 16, int, __nv_bfloat16" in hint
    reds = (3, 4, 5, 0, 1, 2) if bf16_partials else (0, 1, 2, 3, 4, 5)
    if "splitKreduce" not in hint and "Memset" in hint:
        # Memset + one GEMM kernel and no reduce kernel: CUTLASS-style serial split-K through
        # the BF16 output (reduce modes 7/8) is the first hypothesis
        reds = (7, 8) + reds
    contiguous = []
    seen = set()
    for blk in (32, 64, 128):
        nb = K // blk
        for S in range(2, 33):
            if S > nb:
                break
            per = math.ceil(nb / S)
            bounds = tuple(min(nb, s * per) * blk // 16 for s in range(S + 1))
            if len(set(bounds)) != S + 1 or bounds[-1] != ntiles16 or bounds in seen:
                continue
            seen.add(bounds)
            for red in reds:
                for bm in (bias_modes + (3,) if has_bias else bias_modes):
                    kind = "BF16-rounded partials" if red >= 3 else "FP32 partials"
                    contiguous.append(CM.GemmModelSpec(AMPERE_BF16.name, nsplit=S, bounds=bounds, reduce_mode=red, bias_mode=bm, note=f"split-K {S} in {blk}-wide k blocks, {kind}"))
    if "sliced" in hint:
        specs += sliced + contiguous
    else:
        specs += contiguous + sliced
    return specs


def first_mismatch(gpu: np.ndarray, model: np.ndarray, rows: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> dict | None:
    mism = np.argwhere(gpu != model)
    if len(mism) == 0:
        return None
    i, j = (int(v) for v in mism[0])
    return {
        "row": int(rows[i]),
        "col": j,
        "gpu_bf16": f"{int(gpu[i, j]):04x}",
        "model_bf16": f"{int(model[i, j]):04x}",
        "x_row_bf16_first32": "".join(f"{int(v):04x}" for v in x[i, :32]),
        "w_col_bf16_first32": "".join(f"{int(v):04x}" for v in w[j, :32]),
        "bias_bf16": None if b is None else f"{int(b[j]):04x}",
        "mismatching_elements": int(len(mism)),
    }


def stress_families(rng: np.random.Generator, M: int, K: int, N: int, has_bias: bool):
    """Hawkeye-style structured operand families for one shape (all BF16 words)."""

    def bits(shape, elo, ehi, mant_bits=7):
        n = int(np.prod(shape))
        e = rng.integers(elo, ehi + 1, size=n).astype(np.uint16)
        m = rng.integers(0, 1 << mant_bits, size=n).astype(np.uint16)
        s = rng.integers(0, 2, size=n).astype(np.uint16)
        return ((s << 15) | (e << 7) | m).reshape(shape)

    bf16 = lambda a: torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    randn = lambda shape, scale=1.0: bf16(rng.standard_normal(shape) * scale)
    bias = bf16(rng.standard_normal(N) * 0.1) if has_bias else None

    yield "zeros_x", np.zeros((M, K), np.uint16), randn((N, K)), bias
    yield "zeros_w", randn((M, K)), np.zeros((N, K), np.uint16), bias
    # exact pairwise cancellation of products with sparse small residuals
    x = rng.standard_normal((M, K)) * 2.0**8
    w = rng.standard_normal((N, K)) * 2.0**8
    x[:, 1::2] = x[:, 0::2]
    w[:, 1::2] = -w[:, 0::2]
    mask = rng.random((M, K)) < 0.1
    x = np.where(mask, rng.standard_normal((M, K)) * 2.0**-6, x)
    yield "cancellation", bf16(x), bf16(w), bias
    # rounding boundaries: full random significands in a narrow exponent band
    yield "rounding_boundaries", bits((M, K), 120, 134), bits((N, K), 120, 134), bias
    # subnormal operands and subnormal products
    yield "subnormal_operands", bits((M, K), 0, 2), bits((N, K), 0, 130), bias
    yield "subnormal_products", bits((M, K), 60, 70), bits((N, K), 60, 70), bias
    # extreme magnitudes without overflow (|x w| up to ~2^100, sums stay finite)
    yield "extreme_magnitudes", bits((M, K), 30, 230), bits((N, K), 60, 130), bias
    # mixed: near-overflow rows (may produce inf) among normal ones
    x = randn((M, K))
    x_big = bits((M, K), 240, 254)
    x[: max(1, M // 8)] = x_big[: max(1, M // 8)]
    yield "near_overflow_rows", x, bits((N, K), 240, 254), bias
    yield "randn", randn((M, K)), randn((N, K), 0.05), bias


def evaluate_case(
    rt: Runtime,
    label: str,
    family: str,
    M: int,
    K: int,
    N: int,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    max_elements: int,
    rng: np.random.Generator,
    search: bool,
) -> list[dict]:
    xt, wt = from_words(x), from_words(w)
    bt = None if b is None else from_words(b)
    results = []
    for path in rt.paths(b is not None):
        names, y = kernel_names(lambda: rt.run(path, xt, wt, bt))
        y2 = rt.run(path, xt, wt, bt)
        torch.cuda.synchronize()
        # bitwise, not torch.equal: NaN outputs (near-overflow families) must compare equal to themselves
        deterministic = bool(torch.equal(y.view(torch.int16), y2.view(torch.int16)))
        gpu = words(y)
        # rows to evaluate on the CPU
        max_rows = max(1, max_elements // N)
        if M <= max_rows:
            rows = np.arange(M)
        else:
            rows = np.sort(rng.choice(M, size=max_rows, replace=False))
        gsub = gpu[rows]
        xsub = x[rows]
        specs = candidate_specs(K, b is not None, rt.name, path, names)
        found = None
        tried = []
        t_model = None
        # search on a sub-block first (>= ~32k coordinates where the shape allows), then confirm
        # on all evaluated rows
        srows = rows[: min(len(rows), 32)]
        scols = np.arange(min(N, max(1024, 32768 // len(srows))))
        for spec in specs:
            if not search and spec is not specs[0]:
                break
            t0 = time.time()
            mb, _ = CM.gemm_model(AMPERE_BF16, x[srows], w[scols], None if b is None else b[scols], spec, want_f32=False)
            dt = time.time() - t0
            ok = bool(np.array_equal(mb, gpu[srows][:, scols]))
            tried.append({"spec": spec.to_json(), "sub_block_exact": ok, "seconds": dt})
            if ok:
                t0 = time.time()
                mb, _ = CM.gemm_model(AMPERE_BF16, xsub, w, b, spec, want_f32=False)
                t_model = time.time() - t0
                if np.array_equal(mb, gsub):
                    found = spec
                    model_out = mb
                    break
                tried[-1]["full_exact"] = False
        if found is None:
            # report against the primary (no split-K) model on the evaluated rows
            t0 = time.time()
            model_out, _ = CM.gemm_model(AMPERE_BF16, xsub, w, b, specs[0], want_f32=False)
            t_model = time.time() - t0
        exact = int(np.sum(model_out == gsub))
        attempted = int(gsub.size)
        res = {
            "label": label,
            "family": family,
            "path": path,
            "M": M,
            "K": K,
            "N": N,
            "bias": b is not None,
            "kernels": names,
            "gpu_deterministic_rerun": deterministic,
            "rows_evaluated": int(len(rows)),
            "attempted": attempted,
            "exact": exact,
            "model_found": None if found is None else found.to_json(),
            # a split/slice description confirmed on few coordinates is weak evidence: with ~100
            # candidates and a ~1e-4 per-coordinate disagreement rate a wrong one can fit by chance
            "weak_evidence": bool(found is not None and found.nsplit > 1 and attempted < 65536),
            "candidates_tried": len(tried),
            "first_mismatch": first_mismatch(gsub, model_out, rows, xsub, w, b),
            "cpu_seconds_evaluated_rows": t_model,
            "cpu_seconds_per_coordinate": t_model / attempted if attempted else None,
            "cpu_seconds_full_gemm_est": t_model * M / len(rows),
            "k16_steps_per_coordinate": K // 16 * (found.nsplit if found else 1) if found else K // 16,
        }
        if found is None:
            res["candidates"] = tried[:6] + ([{"...": len(tried) - 6}] if len(tried) > 6 else [])
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("candidates", "kernels")}), flush=True)
        print("   kernels:", sorted(set(names)), flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["cublas", "vllm-triton"], required=True)
    ap.add_argument("--acts", default=None, help="directory written by capture_activations.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default=DEFAULT_MS)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--layers", default="0,13,27")
    ap.add_argument("--stress-ms", default="7,64,100")
    ap.add_argument("--stress-shapes", default="qkv_proj,down_proj")
    ap.add_argument("--max-elements", type=int, default=48_000_000, help="CPU-evaluated output elements per case (rows subsampled beyond)")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-natural", action="store_true")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--families", default=None, help="comma list of stress families to run (default all)")
    ap.add_argument("--no-synthetic", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20260907)
    fam_filter = None if args.families is None else set(args.families.split(","))
    rt = Runtime(args.runtime)
    out = {"runtime": rt.info, "per_instruction_model": {"name": AMPERE_BF16.name, "groups": list(AMPERE_BF16.groups), "width": AMPERE_BF16.width, "zero_exponent": AMPERE_BF16.zero_exponent}, "cases": []}
    print(json.dumps(rt.info), flush=True)
    Ms = [int(v) for v in args.ms.split(",")]
    shapes = args.shapes.split(",")
    t_all = time.time()

    if not args.no_natural and args.acts:
        meta = json.load(open(os.path.join(args.acts, "meta.json")))
        out["checkpoint"] = {k: meta[k] for k in ("repo", "revision", "weights_sha256", "tokens")}
        for li in [int(v) for v in args.layers.split(",")]:
            data = np.load(os.path.join(args.acts, f"layer{li}.npz"))
            for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
                if name not in shapes:
                    continue
                x_all, w = data[f"{name}.x"], data[f"{name}.w"]
                b = data[f"{name}.b"] if f"{name}.b" in data else None
                K, N = w.shape[1], w.shape[0]
                for M in Ms:
                    if M > x_all.shape[0]:
                        continue
                    out["cases"] += evaluate_case(rt, f"layers.{li}.{name}", "natural", M, K, N, x_all[:M], w, b, args.max_elements, rng, not args.no_search)
        if "lm_head" in shapes:
            data = np.load(os.path.join(args.acts, "lm_head.npz"))
            x_all, w = data["lm_head.x"], data["lm_head.w"]
            for M in Ms:
                if M > x_all.shape[0]:
                    continue
                out["cases"] += evaluate_case(rt, "lm_head", "natural", M, w.shape[1], w.shape[0], x_all[:M], w, None, args.max_elements, rng, not args.no_search)

    if not args.no_stress:
        for name in args.stress_shapes.split(","):
            K, N, has_bias = SHAPES[name]
            for M in [int(v) for v in args.stress_ms.split(",")]:
                for fam, x, w, b in stress_families(rng, M, K, N, has_bias):
                    if fam_filter is not None and fam not in fam_filter:
                        continue
                    out["cases"] += evaluate_case(rt, f"stress.{name}", fam, M, K, N, x, w, b, args.max_elements, rng, not args.no_search)
        # masked K tail (K not a multiple of the 64-wide K block) on a synthetic shape
        K, N = 1040, 256
        for M in (() if args.no_synthetic else (64,)):
            for fam, x, w, b in stress_families(rng, M, K, N, True):
                if fam in ("randn", "rounding_boundaries", "cancellation"):
                    if fam_filter is not None and fam not in fam_filter:
                        continue
                    out["cases"] += evaluate_case(rt, "stress.synthetic_k1040", fam, M, K, N, x, w, b, args.max_elements, rng, not args.no_search)

    out["seconds"] = time.time() - t_all
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    n_exact = sum(1 for c in out["cases"] if c["exact"] == c["attempted"])
    print(f"DONE {n_exact}/{len(out['cases'])} cases bit-exact in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
