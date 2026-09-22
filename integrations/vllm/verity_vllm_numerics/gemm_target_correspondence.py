"""Per-target correspondence of vLLM's batch-invariant GEMM with the registered ``Gemm_v1{K,N}`` Definition.

[R13 lane ampere, B-AMPERE-GEMM-TILE]  The registered numerical Definition of a bf16 ``[M,K] x W[N,K]^T`` under batch
invariance is ``Gemm_v1{K,N}`` = per output coordinate one FP32 accumulator advanced over the K/16 k16 tensor-core steps in
ascending k order from zero, then RNE to bf16 (``verity_vllm.registry.b1.GemmV2`` with ``DOT = AmpereBF16TcDot16``; the
per-instruction model is ``mma.AMPERE_BF16`` = Hawkeye's ``mma.sync.m16n8k16`` parameters, measured on an A100).  The
launch *geometry* vLLM picks for the target (``BLOCK_SIZE_M/N/K``, ``num_warps``, ``num_stages`` from
``batch_invariant_configs._get_matmul_config`` — the arch family's tuned table row, else the 128/128/64 default) decides
which output coordinates a CTA owns and how the pipeline is staged; whether it also changes the per-coordinate arithmetic
(k order, intermediate rounding, split-K) is exactly what this driver measures, per target, bit for bit.

Two phases (the pinned vLLM venv is Python 3.11, the CPU model's ``veritor.core.silicon`` needs 3.12):

``capture`` (GPU host, vLLM venv): for every GEMM shape of the model (as vLLM merges them) x every M in the list (tails
included: M not a multiple of the launched ``BLOCK_SIZE_M``) x input family, runs the production override
(``enable_batch_invariant_mode()`` -> ``aten::linear`` -> ``matmul_kernel_persistent``), records the configuration the
kernel was actually launched with (a wrapper around ``_get_matmul_config`` in the launching module), the compiled
kernel's tensor-core instruction class (Triton cache PTX: ``mma.sync`` vs ``wgmma``), the device facts, and stores the
bf16 words of ``x``, ``W`` and the output per case (``case_<n>.npz`` + ``cases.json``).

``check`` (any host with ``veritor`` importable): evaluates ``cpu_model.gemm_model`` under the single-chain spec on the
stored operands and compares bitwise.  The JSON report is the correspondence evidence a
``verity_vllm.registry.gemm_targets`` record cites; a single inexact coordinate is a refusal of the target's registration,
never a tolerance.

Usage (pod):
    PYTHONPATH=. VLLM_BATCH_INVARIANT=1 /workspace/venv-cu129/bin/python -m verity_vllm_numerics.gemm_target_correspondence \
        capture --model llama32-1b --ms 1,2,7,13,32,128,129,515,1024 --dir /workspace/cp/sweep/diag/gemm_target_a100
    OMP_NUM_THREADS=64 PYTHONPATH=. /workspace/venv-py312/bin/python -m verity_vllm_numerics.gemm_target_correspondence \
        check --dir /workspace/cp/sweep/diag/gemm_target_a100 --out /workspace/cp/sweep/diag/gemm_target_a100.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import sys
import time

import numpy as np

# GEMM shapes (K, N) as vLLM's merged linear layers launch them (RowParallel/ColumnParallel merged projections).
MODEL_SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    # Llama-3.2-1B: H=2048, NH=32, KVH=8, D=64, I=8192, V=128256
    "llama32-1b": {"qkv_proj": (2048, 3072), "o_proj": (2048, 2048), "gate_up_proj": (2048, 16384),
                   "down_proj": (8192, 2048), "lm_head": (2048, 128256)},
    # SmolLM2-135M: H=576, NH=9, KVH=3, D=64, I=1536, V=49152
    "smollm2-135m": {"qkv_proj": (576, 960), "o_proj": (576, 576), "gate_up_proj": (576, 3072),
                     "down_proj": (1536, 576), "lm_head": (576, 49152)},
    # Qwen2.5-1.5B: H=1536, NH=12, KVH=2, D=128, I=8960, V=151936 (qkv has a bias: added AFTER the kernel in bf16 by the linear override)
    "qwen25-15b": {"qkv_proj": (1536, 2048), "o_proj": (1536, 1536), "gate_up_proj": (1536, 17920),
                   "down_proj": (8960, 1536), "lm_head": (1536, 151936)},
    # [R15 moe GEMM-ADA-TABLE] Qwen3-30B-A3B: H=2048, NH=32, KVH=4, D=128, E=128 experts (fused_moe, not this kernel), router N=128,
    # V=151936; TP1 shapes and the row's TP2 shards (qkv/o_proj/lm_head column- or row-parallel halves).  lm_head at TP1 is the tabled
    # (151936, 2048) row of vLLM's ada table; at TP2 the shard (75968, 2048) is untabled -> default tiling.
    "qwen3-30b-a3b": {"qkv_proj": (2048, 5120), "o_proj": (4096, 2048), "router_gate": (2048, 128), "lm_head": (2048, 151936),
                      "qkv_proj_tp2": (2048, 2560), "o_proj_tp2": (2048, 2048), "lm_head_tp2": (2048, 75968)},
    # [R15 moe GEMM-ADA-TABLE] every (N, K) key of vLLM d9105ea80's tuned batch-invariant table for the 'ada' family (the same five keys
    # as 'hopper'/'blackwell'): the registration evidence of `gemm_targets.ADA_TUNED_GEMM` runs all of them x every M bucket + tails.
    "ada-tuned-table": {"n2048_k2048": (2048, 2048), "n2048_k6144": (6144, 2048), "n4096_k2048": (2048, 4096),
                        "n12288_k2048": (2048, 12288), "n151936_k2048": (2048, 151936)},
}
DEFAULT_MS = "1,2,7,13,32,128,129,515,1024"
STRESS_FAMILIES = ("cancellation", "rounding_boundaries", "subnormal_products", "extreme_magnitudes")


# -- capture (GPU, vLLM venv) -------------------------------------------------------------------------------------------

def _bf16_words(a: np.ndarray) -> np.ndarray:
    import torch
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def families(rng: np.random.Generator, M: int, K: int, N: int, which: list[str]):
    """Input families (bf16 words): `randn` (activation-scale x, weight-scale w) and the Hawkeye-style structured stress
    families of `gemm_production.stress_families` (rounding boundaries, exact cancellation, subnormal products, large
    magnitudes) that exercise the accumulator's rounding paths."""
    def bits(shape, elo, ehi, mant_bits=7):
        n = int(np.prod(shape))
        e = rng.integers(elo, ehi + 1, size=n).astype(np.uint16)
        m = rng.integers(0, 1 << mant_bits, size=n).astype(np.uint16)
        s = rng.integers(0, 2, size=n).astype(np.uint16)
        return ((s << 15) | (e << 7) | m).reshape(shape)

    randn = lambda shape, scale=1.0: _bf16_words(rng.standard_normal(shape) * scale)  # noqa: E731
    if "randn" in which:
        yield "randn", randn((M, K)), randn((N, K), 0.02)
    if "cancellation" in which:
        x = rng.standard_normal((M, K)) * 2.0**8
        w = rng.standard_normal((N, K)) * 2.0**8
        x[:, 1::2] = x[:, 0::2]
        w[:, 1::2] = -w[:, 0::2]
        mask = rng.random((M, K)) < 0.1
        x = np.where(mask, rng.standard_normal((M, K)) * 2.0**-6, x)
        yield "cancellation", _bf16_words(x), _bf16_words(w)
    if "rounding_boundaries" in which:
        yield "rounding_boundaries", bits((M, K), 120, 134), bits((N, K), 120, 134)
    if "subnormal_products" in which:
        yield "subnormal_products", bits((M, K), 60, 70), bits((N, K), 60, 70)
    if "extreme_magnitudes" in which:
        yield "extreme_magnitudes", bits((M, K), 30, 230), bits((N, K), 60, 130)


class LaunchRecorder:
    """Wraps `_get_matmul_config` in the module that launches `matmul_kernel_persistent` (vLLM imports it by name into
    `batch_invariant`), so the configuration of every launch is the one the kernel really received."""

    def __init__(self, BI):
        self.BI = BI
        self.calls: list[dict] = []
        self._orig = BI._get_matmul_config

    def __enter__(self):
        rec = self

        def wrapped(M, N, K, dtype, default):
            cfg = rec._orig(M, N, K, dtype, default)
            rec.calls.append({"M": int(M), "N": int(N), "K": int(K), "dtype": str(dtype), **{k: int(v) for k, v in cfg.items()}})
            return cfg

        self.BI._get_matmul_config = wrapped
        return self

    def __exit__(self, *a):
        self.BI._get_matmul_config = self._orig


def triton_ptx_instruction_classes(kernel_name: str = "matmul_kernel_persistent") -> dict:
    """Tensor-core instruction classes in the Triton-compiled PTX of `kernel_name` found in the Triton cache (evidence of
    which k16 step the launched kernel executes: `mma.sync.aligned.m16n8k16` (Ampere/Ada) vs `wgmma` (Hopper))."""
    cache = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    out = {"cache_dir": cache, "ptx_files": 0, "mma_sync_m16n8k16_bf16": 0, "wgmma": 0, "mma_other": 0, "sm_targets": []}
    for p in glob.glob(os.path.join(cache, "*", f"{kernel_name}*.ptx")):
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        out["ptx_files"] += 1
        out["mma_sync_m16n8k16_bf16"] += len(re.findall(r"mma\.sync\.aligned\.m16n8k16[.\w]*bf16", txt))
        out["wgmma"] += len(re.findall(r"wgmma\.mma_async", txt))
        out["mma_other"] += len(re.findall(r"mma\.sync\.aligned\.(?!m16n8k16)", txt))
        m = re.search(r"\.target\s+(sm_\d+)", txt)
        if m and m.group(1) not in out["sm_targets"]:
            out["sm_targets"].append(m.group(1))
    return out


def capture(args) -> int:
    import torch
    import torch.nn.functional as F
    import vllm
    from vllm.model_executor.determinism import batch_invariant as BI
    from vllm.model_executor.determinism import batch_invariant_configs as BIC
    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import num_compute_units

    words = lambda t: t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)  # noqa: E731
    from_words = lambda w: torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).cuda()  # noqa: E731

    BI.enable_batch_invariant_mode()
    props = torch.cuda.get_device_properties(0)
    cap = current_platform.get_device_capability()
    target = {"device": props.name, "compute_capability": [int(cap.major), int(cap.minor)], "num_sms": int(num_compute_units(0)),
              "multi_processor_count": int(props.multi_processor_count), "tuned_gemm_arch_family": BIC._get_tuned_matmul_arch_family(cap),
              "vllm": vllm.__version__, "torch": torch.__version__, "hostname": platform.node()}
    print(json.dumps({"target": target}), flush=True)
    os.makedirs(args.dir, exist_ok=True)

    shapes = MODEL_SHAPES[args.model]
    names = args.shapes.split(",") if args.shapes else list(shapes)
    ms = [int(m) for m in args.ms.split(",") if m]
    stress_ms = {int(m) for m in args.stress_ms.split(",") if m}
    stress_shapes = set(args.stress_shapes.split(","))
    base_fams = args.families.split(",")
    stress_fams = [f for f in args.stress_families.split(",") if f]
    rng = np.random.default_rng(args.seed)
    cases: list[dict] = []
    for name in names:
        K, N = shapes[name]
        for M in ms:
            fams = list(base_fams) + (stress_fams if (M in stress_ms and name in stress_shapes) else [])
            for fam, x, w in families(rng, M, K, N, fams):
                xt, wt = from_words(x), from_words(w)
                with LaunchRecorder(BI) as rec:
                    y = F.linear(xt, wt)
                    torch.cuda.synchronize()
                    y2 = F.linear(xt, wt)
                    torch.cuda.synchronize()
                cfg = rec.calls[0] if rec.calls else None
                deterministic = bool(torch.equal(y.view(torch.int16), y2.view(torch.int16)))
                gpu = words(y)
                max_rows = max(1, args.max_elements // N)
                rows = np.arange(M) if M <= max_rows else np.sort(rng.choice(M, size=max_rows, replace=False))
                i = len(cases)
                np.savez(os.path.join(args.dir, f"case_{i}.npz"), x=x[rows], w=w, gpu=gpu[rows], rows=rows)
                case = {"i": i, "shape": name, "M": M, "K": K, "N": N, "family": fam, "launch_config": cfg,
                        "m_tail": (M % cfg["BLOCK_SIZE_M"]) if cfg else None, "n_tail": (N % cfg["BLOCK_SIZE_N"]) if cfg else None,
                        "k_tail": (K % cfg["BLOCK_SIZE_K"]) if cfg else None, "launches_recorded": len(rec.calls),
                        "gpu_deterministic_rerun": deterministic, "rows_evaluated": int(len(rows)), "coordinates": int(len(rows) * N)}
                cases.append(case)
                print(json.dumps(case), flush=True)
                del xt, wt, y, y2
                torch.cuda.empty_cache()
    meta = {"target": target, "model": args.model, "ms": ms, "seed": args.seed, "triton_ptx": triton_ptx_instruction_classes(), "cases": cases}
    with open(os.path.join(args.dir, "cases.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({"captured": len(cases), "triton_ptx": meta["triton_ptx"]}), flush=True)
    return 0


# -- check (CPU model) --------------------------------------------------------------------------------------------------

def check(args) -> int:
    from . import cpu_model as CM
    from .mma import AMPERE_BF16

    meta = json.load(open(os.path.join(args.dir, "cases.json")))
    spec = CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=0, note="Gemm_v1{K,N}: one ascending k16 chain, zero init, no epilogue")
    CM.build()
    results: list[dict] = []
    all_exact = True
    t_start = time.time()
    for case in meta["cases"]:
        d = np.load(os.path.join(args.dir, f"case_{case['i']}.npz"))
        x, w, gpu, rows = d["x"], d["w"], d["gpu"], d["rows"]
        t0 = time.time()
        model_out, _ = CM.gemm_model(AMPERE_BF16, x, w, None, spec, want_f32=False)
        dt = time.time() - t0
        exact = int(np.sum(model_out == gpu))
        attempted = int(gpu.size)
        fm = None
        if exact != attempted:
            all_exact = False
            i, j = np.argwhere(model_out != gpu)[0]
            fm = {"row": int(rows[i]), "col": int(j), "gpu_bf16": f"{int(gpu[i, j]):04x}", "model_bf16": f"{int(model_out[i, j]):04x}",
                  "mismatching_coordinates": int(attempted - exact)}
        res = {**case, "attempted": attempted, "exact": exact, "first_mismatch": fm, "cpu_seconds": round(dt, 3)}
        results.append(res)
        print(json.dumps({k: res[k] for k in ("i", "shape", "M", "family", "m_tail", "attempted", "exact", "first_mismatch", "cpu_seconds")}), flush=True)
    configs = {}
    for r in results:
        c = r["launch_config"]
        if c:
            configs.setdefault(json.dumps({k: c[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")}, sort_keys=True), []).append(f"{r['shape']} M={r['M']}")
    report = {"target": meta["target"], "definition": {"kind": "Gemm_v1{K,N}", "dot": "AmpereBF16TcDot16 (mma.sync.m16n8k16 bf16->f32; mma.AMPERE_BF16)",
                                                       "spec": spec.to_json()},
              "model": meta["model"], "ms": meta["ms"], "seed": meta["seed"],
              "distinct_launch_configs": [{"config": json.loads(k), "cases": v} for k, v in configs.items()],
              "triton_ptx": meta["triton_ptx"], "cases": len(results), "cases_exact": sum(r["exact"] == r["attempted"] for r in results),
              "coordinates_attempted": sum(r["attempted"] for r in results), "coordinates_exact": sum(r["exact"] for r in results),
              "all_exact": all_exact, "all_deterministic": all(r["gpu_deterministic_rerun"] for r in results),
              "cpu_seconds": round(time.time() - t_start, 1), "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1), flush=True)
    return 0 if all_exact else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--model", default="llama32-1b", choices=sorted(MODEL_SHAPES))
    c.add_argument("--ms", default=DEFAULT_MS)
    c.add_argument("--shapes", default=None, help="comma list of shape names (default all of the model)")
    c.add_argument("--families", default="randn", help="input families for every (shape, M)")
    c.add_argument("--stress-families", default=",".join(STRESS_FAMILIES))
    c.add_argument("--stress-ms", default="1,7,129", help="M values that also run the stress families")
    c.add_argument("--stress-shapes", default="o_proj,down_proj")
    c.add_argument("--max-elements", type=int, default=24_000_000, help="stored/evaluated output coordinates per case (rows subsampled beyond)")
    c.add_argument("--seed", type=int, default=13)
    c.add_argument("--dir", required=True)
    k = sub.add_parser("check")
    k.add_argument("--dir", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    return capture(args) if args.phase == "capture" else check(args)


if __name__ == "__main__":
    sys.exit(main())
