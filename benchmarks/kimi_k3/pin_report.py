"""What a vLLM build at d9105ea80 holds for Kimi-K3, read without a GPU.

    python pin_report.py elf VLLM_DIR --out elf.json      # each extension's cubin and PTX archs, and K3's kernels by arch
    python pin_report.py ops VLLM_DIR --out ops.json      # every op the extensions register, and K3's ops by name
    python pin_report.py cutedsl --out cutedsl.json       # whether nvidia-cutlass-dsl imports and compiles for sm_90a/100a/103a

`elf` reads every `*.so` under VLLM_DIR (an installed `vllm` package or an unpacked wheel) with `cuobjdump` (CUDA_HOME's, else
PATH's). `ops` imports each extension module under the running torch and lists the registered schemas and their CUDA kernels.
`cutedsl` imports `cutlass` and `cutlass.cute` and compiles one small kernel for each arch with `CUTE_DSL_ARCH`, without a
device. Each subcommand writes one JSON document and prints a summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# Substrings of mangled kernel names: the K3 decode kernels the pin can build as CUDA (the rest of K3's decode is Triton).
K3_KERNELS = {
    "dsv3_fused_a_gemm": "fused_a_gemm_kernel",
    "fused_kda_decode": "kda_decode",
    "kimi_k3_attn_res": "attn_res_fwd",
    "marlin_moe": "Marlin",
    "flashkda": "",
    "situ_and_mul": "situ",
    "grouped_topk": "grouped_topk",
    "fp32_router_gemm": "router_gemm",
    "rms_norm": "rms_norm_kernel",
    "fused_add_rms_norm": "fused_add_rms_norm_kernel",
    "custom_all_reduce": "cross_device_reduce",
    "mla_key_concat": "fusedKimiK3MLA",
}
# Ops by qualified name; presence decides what K3's NVIDIA model can dispatch to.
K3_OPS = [
    "_C::dsv3_fused_a_gemm", "_C::fused_kda_decode", "_C::kimi_k3_attn_res", "_C::situ_and_mul", "_C::rms_norm",
    "_C::fused_add_rms_norm", "_moe_C::moe_wna16_marlin_gemm", "_moe_C::grouped_topk", "_moe_C::moe_sum",
    "_moe_C::moe_align_block_size", "_C::fp32_router_gemm", "_C::dsv3_router_gemm",
    "_C::fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert", "_C::fused_kimi_k3_mla_decode_q_concat_ds_mla_insert",
]


def _cuobjdump() -> str:
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "cuobjdump").exists():
        return str(Path(home) / "bin" / "cuobjdump")
    found = shutil.which("cuobjdump")
    if not found:
        sys.exit("cuobjdump not found: set CUDA_HOME or PATH")
    return found


def _run(argv: list[str]) -> str:
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.stdout + (p.stderr if p.returncode else "")


ELF_RE = re.compile(r"ELF file\s+\d+:\s+(\S+)")
PTX_RE = re.compile(r"PTX file\s+\d+:\s+(\S+)")
ARCH_RE = re.compile(r"\.(sm_\d+[af]?|compute_\d+[af]?)\.(?:cubin|ptx)$")
RES_ARCH_RE = re.compile(r"^\s*(?:Resource usage:|arch = )?\s*.*?(sm_\d+[af]?)")


def _archs(names: list[str]) -> list[str]:
    return sorted({m.group(1) for n in names if (m := ARCH_RE.search(n))})


def kernels_by_arch(so: Path, tool: str) -> dict[str, set[str]]:
    """Mangled kernel name -> the cubin archs that hold it, from `cuobjdump -res-usage` (one `Fatbin elf code:` block per cubin,
    each with its `arch = sm_XX` line, then a `Function NAME:` line per kernel)."""
    out: dict[str, set[str]] = {}
    arch = None
    for line in _run([tool, "-res-usage", str(so)]).splitlines():
        s = line.strip()
        if s.startswith("arch = "):
            arch = s.split("=", 1)[1].strip()
        elif s.startswith("Function ") and s.endswith(":") and arch:
            out.setdefault(s[len("Function "):-1], set()).add(arch)
    return out


def cmd_elf(args: argparse.Namespace) -> int:
    tool = _cuobjdump()
    root = Path(args.vllm_dir)
    doc: dict[str, object] = {"vllm_dir": str(root), "cuobjdump": _run([tool, "--version"]).strip().splitlines()[-1:],
                              "extensions": {}, "k3_kernels": {}}
    for so in sorted(root.rglob("*.so")):
        rel = str(so.relative_to(root))
        elfs = ELF_RE.findall(_run([tool, "--list-elf", str(so)]))
        ptxs = PTX_RE.findall(_run([tool, "--list-ptx", str(so)]))
        if not elfs and not ptxs:
            continue
        kernels = kernels_by_arch(so, tool)
        doc["extensions"][rel] = {"cubin_archs": _archs(elfs), "ptx_archs": _archs(ptxs), "cubins": len(elfs),
                                  "ptx": len(ptxs), "kernels": len(kernels)}
        for key, needle in K3_KERNELS.items():
            if key == "flashkda":
                hit = {k: a for k, a in kernels.items() if "flashkda" in rel}
            else:
                hit = {k: a for k, a in kernels.items() if needle in k}
            if hit:
                entry = doc["k3_kernels"].setdefault(key, {})
                entry[rel] = {"archs": sorted(set().union(*hit.values())), "kernels": len(hit)}
    for key in K3_KERNELS:
        doc["k3_kernels"].setdefault(key, {})
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for rel, e in doc["extensions"].items():
        print(f"{rel}: cubins {e['cubin_archs']} ptx {e['ptx_archs']} ({e['kernels']} kernels)")
    for key, where in doc["k3_kernels"].items():
        print(f"K3 {key}: " + ("; ".join(f"{r} {w['archs']} ({w['kernels']})" for r, w in where.items()) or "ABSENT"))
    return 0


def cmd_ops(args: argparse.Namespace) -> int:
    import torch

    root = Path(args.vllm_dir)
    pkg_parent = root.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    loaded, failed = [], {}
    for so in sorted(root.rglob("*.so")):
        name = ".".join(so.relative_to(pkg_parent).with_suffix("").parts).split(".abi3")[0].split(".cpython")[0]
        try:
            spec = importlib.util.spec_from_file_location(name, so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded.append(name)
        except Exception as e:  # noqa: BLE001
            try:
                torch.ops.load_library(str(so))
                loaded.append(name + " (load_library)")
            except Exception as e2:  # noqa: BLE001
                failed[name] = f"{type(e).__name__}: {e}; load_library: {type(e2).__name__}: {e2}"[:600]
    names = sorted(torch._C._dispatch_get_all_op_names())
    ours = [n for n in names if n.split("::")[0].startswith("_") and not n.startswith(("_quantized::", "_test::", "_nested"))]
    k3 = {}
    for op in K3_OPS:
        present = op in names
        cuda = bool(present and torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA"))
        k3[op] = {"registered": present, "cuda_kernel": cuda}
    doc = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "loaded": loaded, "failed": failed,
           "ops": ours, "k3_ops": k3}
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); loaded {len(loaded)}, failed {len(failed)}; {len(ours)} ops")
    for n, why in failed.items():
        print(f"  FAILED {n}: {why[:200]}")
    for op, e in k3.items():
        print(f"  {op}: {'registered' if e['registered'] else 'ABSENT'}{', CUDA kernel' if e['cuda_kernel'] else ''}")
    return 0


CUTE_PROBE = """
import cutlass
import cutlass.cute as cute

@cute.kernel
def _k(a: cute.Tensor, b: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    b[tidx] = a[tidx] * 2.0

@cute.jit
def _launch(a: cute.Tensor, b: cute.Tensor):
    _k(a, b).launch(grid=(1, 1, 1), block=(32, 1, 1))
"""


def cmd_cutedsl(args: argparse.Namespace) -> int:
    doc: dict[str, object] = {"python": sys.version.split()[0]}
    try:
        from importlib import metadata

        doc["dists"] = {d: metadata.version(d) for d in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
                                                          "nvidia-cutlass-dsl-libs-cu13", "cuda-python", "cuda-bindings",
                                                          "torch") if _has_dist(d)}
    except Exception as e:  # noqa: BLE001
        doc["dists_error"] = repr(e)
    try:
        import cutlass
        import cutlass.cute  # noqa: F401

        doc["import"] = {"ok": True, "cutlass": getattr(cutlass, "__version__", "?"), "file": cutlass.__file__}
    except Exception:  # noqa: BLE001
        doc["import"] = {"ok": False, "error": traceback.format_exc()[-2000:]}
        Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("cutlass DSL import FAILED:\n" + doc["import"]["error"])
        return 0
    doc["compile"] = {}
    for arch in ("sm_90a", "sm_100a", "sm_103a"):
        env = {**os.environ, "CUTE_DSL_ARCH": arch}
        code = CUTE_PROBE + (
            "\nimport torch\nfrom cutlass.cute.runtime import from_dlpack\n"
            "a = from_dlpack(torch.zeros(32)); b = from_dlpack(torch.zeros(32))\n"
            "fn = cute.compile(_launch, a, b)\nprint('COMPILED', type(fn).__name__)\n")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        doc["compile"][arch] = {"rc": p.returncode, "stdout": p.stdout[-1500:], "stderr": p.stderr[-3000:]}
        print(f"cute.compile {arch}: rc {p.returncode} {p.stdout.strip()[-120:]} {p.stderr.strip().splitlines()[-1:] }")
    for mod in ("vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm",
                "vllm.models.kimi_k3.nvidia.ops.latent_moe_tail",
                "vllm.models.kimi_k3.nvidia.low_latency_gemm"):
        p = subprocess.run([sys.executable, "-c", f"import {mod}; print('IMPORTED')"], capture_output=True, text=True)
        doc.setdefault("vllm_imports", {})[mod] = {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}
        print(f"import {mod}: rc {p.returncode}")
    p = subprocess.run([sys.executable, "-c", "from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import "
                        "shape_dynamic_skinny_gemm as s; print('AVAILABLE', s.is_available())"],
                       capture_output=True, text=True)
    doc["skinny_gemm_is_available"] = (p.stdout + p.stderr)[-800:]
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


def _has_dist(name: str) -> bool:
    from importlib import metadata

    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("elf", "ops"):
        p = sub.add_parser(name)
        p.add_argument("vllm_dir")
        p.add_argument("--out", required=True)
    p = sub.add_parser("cutedsl")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"elf": cmd_elf, "ops": cmd_ops, "cutedsl": cmd_cutedsl}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
