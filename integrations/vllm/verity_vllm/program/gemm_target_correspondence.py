"""A target's bf16 linear, measured bit for bit against its registered GEMM Definition.

The registered Definition of a bf16 ``[M,K] x W[N,K]^T`` linear under batch invariance comes from the target's ``GemmTarget``
(``verity_vllm.program.frontend.gemm_targets``): per output coordinate one FP32 accumulator advanced over the K/16 steps of the
record's k16 step (``GemmTarget.dot``) in ascending k order from zero, then RNE to bf16.  Which kernel computes the linear
depends on the target: the pinned vLLM's ``init_batch_invariance()`` installs the Triton ``matmul_kernel_persistent`` override on
cc 8.x and leaves cuBLASLt with split-K disabled everywhere else.  Whatever runs, this driver measures whether its output is the
Definition's, on every linear (K, N) of the given model configs x every M.

``capture`` (the GPU host, the pod's vLLM environment, with ``VLLM_BATCH_INVARIANT=1`` set): ``init_batch_invariance()`` before
any CUDA work (cuBLAS reads its workspace variables when its handle is created), then per linear shape (each rank's shard for
every ``--tp``) x M x input family, ``F.linear(x, W)`` three times: once under the profiler (the kernels it launched), then twice
more (the two must be bit-identical, and equal to the first).  The whole GEMM runs on the device; a deterministic sample of its
rows and columns is stored: ``w_<n>.npz`` (the sampled weight rows) once per shape and family, ``case_<n>.npz`` per case, and
``cases.json``.  ``--bias-shapes`` also runs ``F.linear(x, W, b)`` on the named shapes (Qwen2's qkv has a bias).

``check`` (any host): ``cpu_model.gemm_model`` with the target's step parameters on the stored operands, compared word for word.
The step is cross-checked on a sample of every case's coordinates against the registered primitive itself
(``verity.ml.circuits.evaluation.evaluate_batch``).  One inexact coordinate refuses the correspondence; there is no tolerance.  A bias case
is compared under both epilogues the GEMM relations know (FP32 bias add before the rounding, as a cuBLASLt epilogue does, and
a bf16 add after it, as vLLM's Triton linear does) and reports which one holds.

    verity-vllm gemm-correspondence capture --configs <hf_configs>/LLAMA32_1B.config.json,... --tp 1,2 --seed 13 --dir $RUN/cases
    verity-vllm gemm-correspondence check --dir $RUN/cases --out $RUN/gemm_correspondence.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.config import UsageError, option

DEFAULT_MS = "1,2,3,4,5,6,7,8,12,13,16,17,31,32,33,64,65,127,128,129,255,256,257,512,515,1024,1033,2048,2049,4096,8192"
STRESS_FAMILIES = ("cancellation", "rounding_boundaries", "subnormal_products", "extreme_magnitudes")
#: vLLM's `DEFAULT_VOCAB_PADDING_SIZE`: the lm_head's rows are padded to a multiple of it before the tensor-parallel split.
VOCAB_PADDING = 64
#: the bias epilogues a case with a bias is compared under: `cpu_model.GemmModelSpec.bias_mode`
BIAS_EPILOGUES = {"fp32_before_rounding": 1, "bf16_after_rounding": 3}
SCHEMA = "verity-vllm/gemm-correspondence/v1"


def linear_shapes(config: dict, tp: int = 1) -> dict[str, tuple[int, int]]:
    """``{name: (K, N)}`` of every bf16 linear an HF config's decoder runs through the batch-invariant path, on one of ``tp``
    ranks: the attention projections, the dense MLP (or, for a mixture of experts, the router: its experts are fused-MoE kernels,
    not this linear) and the lm_head.  Column-parallel layers split N, row-parallel ones K; fewer KV heads than ranks are
    replicated (one per rank)."""
    H, NH = int(config["hidden_size"]), int(config["num_attention_heads"])
    KVH = int(config.get("num_key_value_heads") or NH)
    D = int(config.get("head_dim") or H // NH)
    if tp < 1 or NH % tp or (KVH >= tp and KVH % tp):
        raise UsageError(f"{NH} attention heads / {KVH} KV heads do not split over tp={tp}")
    V = -(-int(config["vocab_size"]) // VOCAB_PADDING) * VOCAB_PADDING
    shapes = {"qkv_proj": (H, (NH // tp + 2 * max(1, KVH // tp)) * D), "o_proj": (NH // tp * D, H)}
    experts = config.get("num_experts") or config.get("num_local_experts")
    if experts:
        shapes["router"] = (H, int(experts))
    else:
        inter = int(config["intermediate_size"])
        if inter % tp:
            raise UsageError(f"intermediate size {inter} does not split over tp={tp}")
        # GPT-NeoX-style configs (the ones carrying `rotary_pct`) have an ungated MLP: one up projection, not a merged gate and up
        if "rotary_pct" in config:
            shapes["up_proj"] = (H, inter // tp)
        else:
            shapes["gate_up_proj"] = (H, 2 * inter // tp)
        shapes["down_proj"] = (inter // tp, H)
    if V % tp:
        raise UsageError(f"padded vocabulary {V} does not split over tp={tp}")
    shapes["lm_head"] = (H, V // tp)
    return shapes


def sample_indices(n: int, cap: int, edge: int) -> np.ndarray:
    """Sorted distinct indices into ``range(n)``: all of them when ``n <= cap``, else the first and the last ``edge`` (where the
    tile tails fall) and evenly spaced ones between, ``cap`` at most."""
    if n <= cap:
        return np.arange(n)
    edge = min(edge, cap // 4)
    mid = np.linspace(edge, n - edge - 1, cap - 2 * edge).round().astype(np.int64)
    return np.unique(np.concatenate([np.arange(edge), mid, np.arange(n - edge, n)]))


def derived_seed(seed: int, *labels: Any) -> int:
    """A 63-bit seed for one operand, from the run's seed and the operand's labels."""
    h = hashlib.sha256("/".join(map(str, (seed, *labels))).encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


# -- capture (GPU host) -------------------------------------------------------------------------------------------------------

def _bf16_bits(torch, shape, elo: int, ehi: int):
    """bf16 tensors of uniformly sampled sign, exponent field in [elo, ehi] and mantissa (the torch generator, already seeded)."""
    s = torch.randint(0, 2, shape, device="cuda", dtype=torch.int32)
    e = torch.randint(elo, ehi + 1, shape, device="cuda", dtype=torch.int32)
    m = torch.randint(0, 128, shape, device="cuda", dtype=torch.int32)
    w = (s << 15) | (e << 7) | m
    return torch.where(w >= 32768, w - 65536, w).to(torch.int16).view(torch.bfloat16)


def operands(torch, family: str, M: int, K: int, N: int, seed_x: int, seed_w: int):
    """``(x [M,K], W [N,K])`` bf16 on the device for one input family: ``randn`` (activation-scale x, weight-scale W) and the
    structured stress families that exercise the accumulator's rounding paths."""
    torch.manual_seed(seed_w)
    if family in ("randn", "randn_bias"):
        w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    elif family == "cancellation":
        w = torch.randn(N, K, device="cuda") * 2.0**8
        w[:, 1::2] = -w[:, 0::2]
        w = w.to(torch.bfloat16)
    elif family == "rounding_boundaries":
        w = _bf16_bits(torch, (N, K), 120, 134)
    elif family == "subnormal_products":
        w = _bf16_bits(torch, (N, K), 60, 70)
    elif family == "extreme_magnitudes":
        w = _bf16_bits(torch, (N, K), 60, 130)
    else:
        raise UsageError(f"unknown input family {family!r}")
    torch.manual_seed(seed_x)
    if family in ("randn", "randn_bias"):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    elif family == "cancellation":
        x = torch.randn(M, K, device="cuda") * 2.0**8
        x[:, 1::2] = x[:, 0::2]
        x = torch.where(torch.rand(M, K, device="cuda") < 0.1, torch.randn(M, K, device="cuda") * 2.0**-6, x).to(torch.bfloat16)
    elif family == "rounding_boundaries":
        x = _bf16_bits(torch, (M, K), 120, 134)
    elif family == "subnormal_products":
        x = _bf16_bits(torch, (M, K), 60, 70)
    else:
        x = _bf16_bits(torch, (M, K), 30, 230)
    return x, w


def _words(torch, t) -> np.ndarray:
    return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def launched_kernels(torch, fn) -> tuple[Any, list[str]]:
    """``fn()``'s result and the names of the device kernels it launched."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if str(e.device_type).endswith("CUDA")})
    return out, names


def device_facts(torch, bi) -> dict[str, Any]:
    import vllm
    from vllm.model_executor.determinism.batch_invariant_configs import _get_tuned_matmul_arch_family
    from vllm.platforms import current_platform
    props = torch.cuda.get_device_properties(0)
    return {"device": props.name, "compute_capability": [int(props.major), int(props.minor)],
            "tuned_gemm_arch_family": _get_tuned_matmul_arch_family(current_platform.get_device_capability()),
            "multi_processor_count": int(props.multi_processor_count), "vllm": vllm.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "hostname": platform.node(), "batch_invariant_mode": bool(bi._batch_invariant_MODE),
            "torch_preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()).split(".")[-1],
            "torch_allow_bf16_reduced_precision_reduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}


def _case_plan(a: CaptureOptions) -> list[dict[str, Any]]:
    ms = [int(m) for m in a.ms.split(",") if m]
    stress_ms = {int(m) for m in a.stress_ms.split(",") if m}
    stress_shapes = {s for s in a.stress_shapes.split(",") if s}
    bias_shapes = {s for s in a.bias_shapes.split(",") if s}
    base = [f for f in a.families.split(",") if f]
    stress = [f for f in a.stress_families.split(",") if f]
    plan: dict[tuple, dict[str, Any]] = {}
    for path in [p for p in a.configs.split(",") if p]:
        config, label = json.loads(Path(path).read_text()), Path(path).name.split(".")[0]
        for tp in [int(t) for t in a.tp.split(",") if t]:
            for shape, (K, N) in linear_shapes(config, tp).items():
                fams = list(base) + (["randn_bias"] if shape in bias_shapes else [])
                fams += [f for f in stress if shape in stress_shapes]
                for family in fams:
                    for M in ms if family in base or family == "randn_bias" else sorted(stress_ms):
                        key = (K, N, family, M)
                        if key in plan:
                            plan[key]["also"].append(f"{label} tp{tp} {shape}")
                            continue
                        plan[key] = {"config": label, "tp": tp, "shape": shape, "K": K, "N": N, "family": family, "M": M, "also": []}
    return list(plan.values())


def capture(a: CaptureOptions) -> int:
    import torch
    import torch.nn.functional as F
    import vllm.envs as envs
    from vllm.model_executor.determinism import batch_invariant as bi

    if not envs.VLLM_BATCH_INVARIANT:
        raise UsageError("capture runs under VLLM_BATCH_INVARIANT=1 (the configuration every row of record declares)")
    bi.init_batch_invariance()
    target = device_facts(torch, bi)
    print(json.dumps({"target": target}), flush=True)
    out = Path(a.case_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    weights: dict[tuple, dict[str, Any]] = {}
    for p in _case_plan(a):
        wkey = (p["config"], p["tp"], p["shape"], p["family"])
        x, w = operands(torch, p["family"], p["M"], p["K"], p["N"], derived_seed(a.seed, *wkey, p["M"], "x"), derived_seed(a.seed, *wkey, "w"))
        bias = None
        if p["family"] == "randn_bias":
            torch.manual_seed(derived_seed(a.seed, *wkey, "b"))
            bias = (torch.randn(p["N"], device="cuda") * 0.02).to(torch.bfloat16)
        if wkey not in weights:
            cols = sample_indices(p["N"], a.max_cols, 256)
            wfile = f"w_{len(weights)}.npz"
            np.savez(out / wfile, w=_words(torch, w[torch.from_numpy(cols).cuda()]), cols=cols,
                     **({"b": _words(torch, bias)[cols]} if bias is not None else {}))
            weights[wkey] = {"file": wfile, "cols": cols}
        cols = weights[wkey]["cols"]
        y, kernels = launched_kernels(torch, lambda: F.linear(x, w, bias))
        y2, y3 = F.linear(x, w, bias), F.linear(x, w, bias)
        torch.cuda.synchronize()
        deterministic = bool(torch.equal(y.view(torch.int16), y2.view(torch.int16)) and torch.equal(y2.view(torch.int16), y3.view(torch.int16)))
        rows = sample_indices(p["M"], a.max_rows, 32)
        rt, ct = torch.from_numpy(rows).cuda(), torch.from_numpy(cols).cuda()
        i = len(cases)
        np.savez(out / f"case_{i}.npz", x=_words(torch, x[rt]), gpu=_words(torch, y[rt][:, ct]), rows=rows)
        case = {"i": i, **p, "w_file": weights[wkey]["file"], "bias": bias is not None, "kernels": kernels,
                "gpu_deterministic_rerun": deterministic, "rows_evaluated": int(len(rows)), "cols_evaluated": int(len(cols)),
                "coordinates": int(len(rows) * len(cols))}
        cases.append(case)
        print(json.dumps({k: case[k] for k in ("i", "config", "tp", "shape", "M", "family", "kernels", "gpu_deterministic_rerun")}), flush=True)
        del x, w, y, y2, y3
        torch.cuda.empty_cache()
    meta = {"schema": SCHEMA, "target": target, "configs": a.configs.split(","), "tp": a.tp, "ms": a.ms, "seed": a.seed,
            "max_rows": a.max_rows, "max_cols": a.max_cols, "cases": cases}
    (out / "cases.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"captured": len(cases), "kernels": sorted({k for c in cases for k in c["kernels"]})}), flush=True)
    return 0


# -- check (any host) ---------------------------------------------------------------------------------------------------------

def step_model(dot: str):
    """``(the registered step primitive, a cpu_model TCModel with its parameters)`` for a ``GemmTarget.dot`` name."""
    from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
    from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
    from verity.ml.definitions.numerics.mma import TCModel
    from verity_vllm.program.frontend.targets import DOTS

    groupsum = {"AmpereBF16TcDot16": AMPERE_BF16_M16N8K16, "HopperBF16WgmmaDot16": HOPPER_BF16_WGMMA_K16}.get(dot)
    if dot not in DOTS or groupsum is None:
        raise UsageError(f"GEMM step {dot!r} has no registered primitive with known parameters")
    return DOTS[dot], TCModel(name=groupsum.name, arch=groupsum.arch, dtype="bf16", instruction=dot, groups=tuple(groupsum.groups),
                              width=groupsum.width, zero_exponent=groupsum.zero_exponent, version=f"{dot} (cpu_model parameters)")


def crosscheck(prim, model, x: np.ndarray, w: np.ndarray, rows: int, cols: int) -> tuple[int, int]:
    """``(exact, attempted)``: the FP32 chain of ``cpu_model`` against the registered primitive's own evaluation on the first
    ``rows`` x ``cols`` coordinates."""
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.definitions.numerics import cpu_model as CM

    xs, ws = x[:rows], w[:cols]
    _, f32 = CM.gemm_model(model, xs, ws, None, CM.GemmModelSpec(model.name, bias_mode=0), want_f32=True)
    a = np.repeat(xs, len(ws), axis=0)
    b = np.tile(ws, (len(xs), 1))
    acc = np.zeros(len(a), dtype=np.uint32)
    for s in range(0, a.shape[1], 16):
        acc = evaluate_batch(prim, acc, a[:, s:s + 16], b[:, s:s + 16]).astype(np.uint32)
    return int(np.sum(acc == f32.reshape(-1))), int(acc.size)


def check_case(case: dict, d, wd, model, spec_of) -> dict[str, Any]:
    from verity.ml.definitions.numerics import cpu_model as CM

    x, gpu, rows = d["x"], d["gpu"], d["rows"]
    w, cols = wd["w"], wd["cols"]
    bias = wd["b"] if case["bias"] else None
    epilogues = BIAS_EPILOGUES if bias is not None else {"none": 0}
    by_epilogue, first = {}, None
    for name, mode in epilogues.items():
        out, _ = CM.gemm_model(model, x, w, bias, spec_of(mode), want_f32=False)
        by_epilogue[name] = int(np.sum(out == gpu))
        if first is None and by_epilogue[name] != gpu.size:
            i, j = np.argwhere(out != gpu)[0]
            first = {"epilogue": name, "row": int(rows[i]), "col": int(cols[j]), "gpu_bf16": f"{int(gpu[i, j]):04x}",
                     "model_bf16": f"{int(out[i, j]):04x}", "mismatching_coordinates": int(gpu.size - by_epilogue[name])}
    best = max(by_epilogue, key=by_epilogue.get)
    return {"attempted": int(gpu.size), "exact": by_epilogue[best], "epilogue": best, "exact_by_epilogue": by_epilogue,
            "first_mismatch": None if by_epilogue[best] == gpu.size else first}


def check(a: CheckOptions) -> int:
    from verity.ml.definitions.numerics import cpu_model as CM
    from verity_vllm.program.frontend.gemm_targets import gemm_target_for

    root = Path(a.case_dir)
    meta = json.loads((root / "cases.json").read_text())
    t = meta["target"]
    gt = gemm_target_for(t["compute_capability"], t.get("tuned_gemm_arch_family", ...))
    prim, model = step_model(gt.dot)

    def spec_of(mode: int):
        return CM.GemmModelSpec(model.name, bias_mode=mode, note=f"{gt.kind}: one ascending k16 chain from zero")

    CM.build()
    results, t0, wcache = [], time.time(), {}
    for case in meta["cases"]:
        if case["w_file"] not in wcache:
            wcache.clear()
            wcache[case["w_file"]] = dict(np.load(root / case["w_file"]))
        wd, d = wcache[case["w_file"]], np.load(root / f"case_{case['i']}.npz")
        res = {**case, **check_case(case, d, wd, model, spec_of)}
        cr_rows = min(4, len(d["x"]))
        cr_cols = min(len(wd["w"]), max(1, a.crosscheck // cr_rows))
        res["crosscheck_exact"], res["crosscheck_attempted"] = crosscheck(prim, model, d["x"], wd["w"], cr_rows, cr_cols)
        results.append(res)
        print(json.dumps({k: res[k] for k in ("i", "config", "tp", "shape", "M", "family", "attempted", "exact", "epilogue",
                                              "crosscheck_exact", "crosscheck_attempted", "first_mismatch")}), flush=True)
    report = summary(meta, gt, model, results, time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1), flush=True)
    return 0 if report["all_exact"] and report["crosscheck_all_exact"] else 1


def summary(meta: dict, gt, model, results: list[dict], seconds: float) -> dict[str, Any]:
    kernels: dict[str, list[str]] = {}
    for r in results:
        for k in r["kernels"]:
            kernels.setdefault(k, []).append(f"{r['config']} tp{r['tp']} {r['shape']} M={r['M']} {r['family']}")
    bias = [r for r in results if r["bias"]]
    return {"schema": SCHEMA, "target": meta["target"], "gemm_target": gt.describe(),
            "definition": {"kind": gt.kind, "dot": gt.dot, "step": {"name": model.name, "groups": list(model.groups), "width": model.width,
                                                                    "zero_exponent": model.zero_exponent},
                           "chain": "per coordinate one FP32 accumulator over the K/16 steps in ascending k order from zero, RNE to bf16"},
            "configs": meta["configs"], "tp": meta["tp"], "ms": meta["ms"], "seed": meta["seed"],
            "kernels": [{"name": k, "cases": len(v), "first_cases": v[:8]} for k, v in sorted(kernels.items())],
            "cases": len(results), "cases_exact": sum(r["exact"] == r["attempted"] for r in results),
            "coordinates_attempted": sum(r["attempted"] for r in results), "coordinates_exact": sum(r["exact"] for r in results),
            "bias_epilogues": {e: sum(r["exact_by_epilogue"][e] == r["attempted"] for r in bias) for e in BIAS_EPILOGUES} if bias else None,
            "crosscheck_attempted": sum(r["crosscheck_attempted"] for r in results),
            "crosscheck_exact": sum(r["crosscheck_exact"] for r in results),
            "crosscheck_all_exact": all(r["crosscheck_exact"] == r["crosscheck_attempted"] for r in results),
            "all_exact": all(r["exact"] == r["attempted"] for r in results),
            "all_deterministic": all(r["gpu_deterministic_rerun"] for r in results), "cpu_seconds": round(seconds, 1), "results": results}


# -- the command ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class CaptureOptions:
    """Run the linears on this host's GPU under vLLM's batch invariance and store a sample of their words."""

    configs: str = option("--configs", help="comma list of HF config.json paths (engine/profiles/hf_configs/)")
    tp: str = option("--tp", default="1", help="comma list of tensor-parallel degrees: every rank's shard shapes")
    ms: str = option("--ms", default=DEFAULT_MS, help="rows of the linear (tokens in the step)")
    families: str = option("--families", default="randn", help="input families of every (shape, M)")
    stress_families: str = option("--stress-families", default=",".join(STRESS_FAMILIES))
    stress_ms: str = option("--stress-ms", default="1,7,129,1033", help="the M values the stress families run at")
    stress_shapes: str = option("--stress-shapes", default="o_proj,down_proj", help="the shapes the stress families run on")
    bias_shapes: str = option("--bias-shapes", default="", help="shapes that also run F.linear(x, W, b) (randn_bias)")
    max_rows: int = option("--max-rows", type=int, default=128, help="stored rows per case (all up to this, else sampled)")
    max_cols: int = option("--max-cols", type=int, default=4096, help="stored columns per shape (all up to this, else sampled)")
    seed: int = option("--seed", type=int, help="the operands' seed (recorded)")
    case_dir: str = option("--dir", help="where the cases are written")
    cmd: str = field(default="capture", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    """Compare the stored words with the registered GEMM Definition of the capture's target."""

    case_dir: str = option("--dir", help="a capture's directory")
    out: str = option("--out", help="the report (JSON)")
    crosscheck: int = option("--crosscheck", type=int, default=256, help="coordinates per case checked against the registered step")
    cmd: str = field(default="check", init=False)


COMMANDS = {"capture": CaptureOptions, "check": CheckOptions}


def main(a: Any) -> int:
    return capture(a) if a.cmd == "capture" else check(a)
