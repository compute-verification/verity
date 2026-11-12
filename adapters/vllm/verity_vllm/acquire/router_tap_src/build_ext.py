"""Build or check the router-softmax tap's CUDA op, the torch extension `verity_router_tap`: vLLM's `topkGating` kernel (the fused
softmax + top-k of `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` at d9105ea80, which the installed `_moe_C.topk_softmax` launches
for a power-of-two or multiple-of-64 expert count; `MoeRouterTopKOrdered_v1` / `MoeRouterTopKOrderedNorm_v1`) with its softmax stored.

    verity-vllm router-tap-build --vllm-src DIR --build-dir DIR     # DIR/csrc/...: the pinned sources (ops/pod_router_tap.sh)
    verity-vllm router-tap-build --check DIR/verity_router_tap.so

`tapped_kernel(text)` cuts the file's `vllm::moe` namespace (`cut_block`, sha-pinned), renames it `vllm::verity_router_tap` and makes
the eight edits of `EDITS` (`edit_block`), each at exactly one line of the block: a `Tap` template parameter and a `tap_out` /
`tap_stride` argument of `topkGating` (defaulted, so the block's own launchers still compile), then six guarded stores of the values
the ordered router commits, in `query.router_softmax.slots` order per token row: the row max after the butterfly (slot 0), each
thread's exponentials (1 + column), the reciprocal of the row sum (1 + E), each thread's probabilities after the NaN / inf clamp
(2 + E + column), and under `renormalize` each round's selected probability (2 + 2E + round) and the epilogue's scale (2 + 2E + k).
With `Tap = false` the kernel is the pinned text.  `router_tap.cu` launches it as the installed op does (`topkGatingLauncherHelper`:
the same grid, block and template arguments, the current stream) and registers `topk_softmax` (tap off), `topk_softmax_tap` (tap on)
and `tap_version`.  The flags are those vLLM builds its CUDA ops with (-O3, no --use_fast_math: `expf` is the library sequence
`NvExpf_v1` states and both divisions are IEEE), for this device's architecture.  `build_info.json` beside the .so records the sources'
sha256, the flags, the architecture, nvcc and torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from verity_vllm.config import UsageError, option

NAME = "verity_router_tap"
TAP_VERSION = 1
PIN = "d9105ea8001e0a6d77a96327d17515bb5791fb36"
KERNEL_FILE = "csrc/libtorch_stable/moe/topk_softmax_kernels.cu"
#: the pinned vLLM sources the build reads, by path under the vLLM tree -> sha256
PINNED_SOURCES: dict[str, str] = {
    KERNEL_FILE: "5e0eae8dbd76f04f9ddabb587dab1cfc2fdae48cf1966775bb9cf0b8ad6b432a",
    "csrc/cuda_compat.h": "959bc9e0e73e020d777530b18bf12160d35cc4eb976335955395729700b2f975",
    "csrc/libtorch_stable/cub_helpers.h": "91bd3c5f862387c475806a13245f0353369439a4e83962d2798e820314c48c31",
}
BLOCK_START = "namespace vllm {\nnamespace moe {\n"
BLOCK_END = "} // namespace moe\n} // namespace vllm\n"
#: sha256 of the `vllm::moe` namespace as `cut_block` cuts it from the pinned file
BLOCK_SHA256 = "99898b0622ebad17f4b974f6b89b99e255f6027a6394d25de7b2992ad7b97040"
#: the file's two macros the block uses (`TopkConstants`, `topkGatingLauncherHelper`)
MACROS = "#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#define MIN(a, b) ((a) < (b) ? (a) : (b))\n"
ROW = "static_cast<int64_t>(thread_row) * tap_stride"

#: (anchor line, "replace" | "before" | "after", text): the anchor is one whole line of the block, occurring exactly once
EDITS: tuple[tuple[str, str, str], ...] = (
    ("          typename InputType = float, ScoringFunc SF>", "replace",
     "          typename InputType = float, ScoringFunc SF, bool Tap = false>"),
    ("        const float* bias, const double routed_scaling_factor, const bool* is_padding)", "replace",
     "        const float* bias, const double routed_scaling_factor, const bool* is_padding,\n"
     "        float* __restrict__ tap_out = nullptr, const int tap_stride = 0)"),
    ("      // From this point, thread max in all the threads have the max within the row.", "before",
     "      if constexpr (Tap) {\n"
     "        if (thread_group_idx == 0) {\n"
     f"          tap_out[{ROW}] = thread_max;\n"
     "        }\n"
     "      }"),
    ("// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.", "before",
     "      if constexpr (Tap) {\n"
     "#pragma unroll\n"
     "        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {\n"
     "#pragma unroll\n"
     "          for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {\n"
     f"            tap_out[{ROW} + 1 + first_elt_read_by_thread + ldg * ELTS_PER_LDG * THREADS_PER_ROW + ii] =\n"
     "                row_chunk[ldg * ELTS_PER_LDG + ii];\n"
     "          }\n"
     "        }\n"
     "      }"),
    ("      const float reciprocal_row_sum = 1.f / row_sum;", "after",
     "      if constexpr (Tap) {\n"
     "        if (thread_group_idx == 0) {\n"
     f"          tap_out[{ROW} + 1 + NUM_EXPERTS] = reciprocal_row_sum;\n"
     "        }\n"
     "      }"),
    ("    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;", "before",
     "    if constexpr (Tap && SF == SCORING_SOFTMAX) {\n"
     "#pragma unroll\n"
     "      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {\n"
     "#pragma unroll\n"
     "        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {\n"
     f"          tap_out[{ROW} + 2 + NUM_EXPERTS + first_elt_read_by_thread + ldg * ELTS_PER_LDG * THREADS_PER_ROW + ii] =\n"
     "              row_chunk[ldg * ELTS_PER_LDG + ii];\n"
     "        }\n"
     "      }\n"
     "    }"),
    ("                selected_sum += max_val;", "after",
     "                if constexpr (Tap) {\n"
     f"                    tap_out[{ROW} + 2 + 2 * NUM_EXPERTS + k_idx] = max_val;\n"
     "                }"),
    ("        scale /= denom;", "after",
     "        if constexpr (Tap) {\n"
     f"          tap_out[{ROW} + 2 + 2 * NUM_EXPERTS + k] = scale;\n"
     "        }"),
)
HERE = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cut_block(text: str) -> str:
    """The pinned file's `vllm::moe` namespace, its opening through its closing line; refuses any other text."""
    if text.count(BLOCK_START) != 1 or text.count(BLOCK_END) != 1:
        raise ValueError(f"{KERNEL_FILE}: the vllm::moe namespace opens {text.count(BLOCK_START)} and closes {text.count(BLOCK_END)} times, not once")
    i = text.index(BLOCK_START)
    block = text[i: text.index(BLOCK_END, i) + len(BLOCK_END)]
    if sha256(block.encode()) != BLOCK_SHA256:
        raise ValueError(f"{KERNEL_FILE}: the vllm::moe namespace is not the pinned one (sha256 {sha256(block.encode())}, want {BLOCK_SHA256})")
    return block


def edit_block(block: str) -> str:
    """`EDITS` applied to a block (each anchor one whole line, exactly once) and the namespace renamed `verity_router_tap`."""
    lines = block.split("\n")
    for anchor, how, new in EDITS:
        at = [i for i, line in enumerate(lines) if line == anchor]
        if len(at) != 1:
            raise ValueError(f"{KERNEL_FILE}: the line {anchor.strip()!r} occurs {len(at)} times in the vllm::moe namespace, not once")
        i = at[0]
        add = new.split("\n")
        lines[i:i + 1] = add if how == "replace" else (add + [anchor] if how == "before" else [anchor] + add)
    out = "\n".join(lines)
    if not (out.startswith(BLOCK_START) and out.endswith(BLOCK_END)):
        raise ValueError("the block does not open and close the vllm::moe namespace")
    return ("namespace vllm {\nnamespace verity_router_tap {\n" + out[len(BLOCK_START):-len(BLOCK_END)]
            + "} // namespace verity_router_tap\n} // namespace vllm\n")


def tapped_kernel(text: str) -> str:
    """The tapped kernel header (`router_tap_kernel.cuh`) from the pinned file's text: refuses any other text."""
    return (f"// Generated by verity_vllm.acquire.router_tap_src.build_ext from vLLM {PIN} {KERNEL_FILE}: its vllm::moe\n"
            f"// namespace, renamed, with the router-softmax tap (topkGating<..., Tap = true> stores the softmax per row).  Do not edit.\n"
            "#pragma once\n#include <cassert>\n#include <cfloat>\n#include <cstdint>\n#include <type_traits>\n\n#include <cuda_runtime.h>\n"
            "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n\n#include \"cuda_compat.h\"\n#include \"cub_helpers.h\"\n\n"
            f"{MACROS}\n{edit_block(cut_block(text))}")


def pinned_sources(vllm_src: str) -> dict[str, bytes]:
    """The pinned files under `vllm_src`, each checked against its sha256."""
    out = {}
    for rel, want in PINNED_SOURCES.items():
        data = (Path(vllm_src) / rel).read_bytes()
        if sha256(data) != want:
            raise ValueError(f"{vllm_src}/{rel}: sha256 {sha256(data)}, not the pinned {want} (vLLM {PIN})")
        out[rel] = data
    return out


def nvcc_flags() -> list[str]:
    return ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", f"-DVERITY_ROUTER_TAP_VERSION={TAP_VERSION}"]


def build(vllm_src: str, build_dir: str) -> dict:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    srcs = pinned_sources(vllm_src)
    out = Path(build_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = tapped_kernel(srcs[KERNEL_FILE].decode())
    (out / "router_tap_kernel.cuh").write_text(header)
    cu = HERE / "router_tap.cu"
    cc = torch.cuda.get_device_capability(0)
    nvcc = os.path.join(CUDA_HOME or "", "bin", "nvcc")
    info = {"name": NAME, "tap_version": TAP_VERSION, "vllm": PIN, "sources": {k: sha256(v) for k, v in srcs.items()},
            "launcher": {"file": str(cu), "sha256": sha256(cu.read_bytes())}, "kernel_header_sha256": sha256(header.encode()),
            "nvcc_flags": nvcc_flags(), "arch": f"{cc[0]}.{cc[1]}", "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_home": CUDA_HOME,
            "nvcc_version": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]}
    t0 = time.time()
    load(name=NAME, sources=[str(cu)], extra_cuda_cflags=nvcc_flags(), extra_cflags=["-O3", "-DNDEBUG"],
         extra_include_paths=[str(Path(vllm_src) / "csrc"), str(Path(vllm_src) / "csrc" / "libtorch_stable"), str(out)],
         build_directory=str(out), verbose=True, is_python_module=False, with_cuda=True)
    so = out / f"{NAME}.so"
    info.update(compile_seconds=round(time.time() - t0, 1), so=str(so), so_sha256=sha256(so.read_bytes()))
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def check(so: str) -> dict:
    """Load a build and check its ops and tap version."""
    import torch

    torch.ops.load_library(so)
    ns = getattr(torch.ops, NAME)
    missing = [op for op in ("topk_softmax", "topk_softmax_tap", "tap_version") if not hasattr(ns, op)]
    version = int(ns.tap_version()) if not missing else None
    return {"so": so, "sha256": sha256(Path(so).read_bytes()), "missing_ops": missing, "tap_version": version,
            "ok": not missing and version == TAP_VERSION}


@dataclass(frozen=True, kw_only=True)
class Options:
    vllm_src: str | None = option("--vllm-src", default=None, help=f"a vLLM tree holding the pinned sources ({', '.join(PINNED_SOURCES)})")
    build_dir: str | None = option("--build-dir", default=None, help=f"where the build and {NAME}.so go")
    check: str | None = option("--check", default=None, help="check a built .so (its ops and tap version) instead of building")


def main(a: Options) -> int:
    if a.check:
        r = check(a.check)
        print(("ROUTER-TAP-OK " if r["ok"] else "ROUTER-TAP-FAIL ") + json.dumps(r), flush=True)
        return 0 if r["ok"] else 1
    if not (a.vllm_src and a.build_dir):
        raise UsageError("give --vllm-src and --build-dir (build) or --check SO")
    info = build(a.vllm_src, a.build_dir)
    print(f"BUILD-OK {info['so']} {info['so_sha256']} arch={info['arch']} ({info['compile_seconds']} s)", flush=True)
    return 0
