"""Build the pinned vllm-flash-attn's FA3 forward (hopper/, 506341a1) for bf16 head dim 64 as a torch extension, its hdim-64
causal tile optionally changed, so attn_knobs.py can time it beside the served wheel's `_vllm_fa3_C`.

    python fa3_tile_build.py --tree TREE --name NS --build-dir DIR [--tile-m 128]

TREE is a copy of the pinned tree with `integrations/vllm/verity_vllm/acquire/fa3_tap_src/apply_fa3_tap.py` applied (its
`fa3_torch_lib_verity.cpp`), built here with VERITY_TAP=0, which is the pinned kernel. The flags are that builder's
(`build_fa3_ext.py`, after the pinned CMake build of `_vllm_fa3_C`), narrowed to head dim 64: the paged-KV and the
contiguous-KV PackGQA forwards (`flash_fwd_hdim64_bf16_paged_sm90.cu`, `flash_fwd_hdim64_bf16_packgqa_sm90.cu`).

--tile-m M rewrites `tile_size_fwd_sm90`'s hdim-64 causal tile from 192 x 128 to M x 128. kBlockN, MmaPV_is_RS and
IntraWGOverlap stay, so every query row still meets the same 128-key blocks, last to first, through the same
online-softmax arithmetic; only which rows share an M tile changes. --ctas-per-sm N (with --tile-m 64, one MMA
warpgroup) runs N CTAs an SM (`occupy`), which changes only where tiles run. The op is `torch.ops.NS.fwd`, with
`_vllm_fa3_C.fwd`'s signature.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PINNED_HD64 = "return {192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local};"


def retile(tile_size_h: Path, m: int) -> None:
    s = tile_size_h.read_text()
    if s.count(PINNED_HD64) != 1:
        raise SystemExit(f"{tile_size_h}: the pinned hdim-64 tile line is not there exactly once")
    tiled = f"return {{is_causal ? {m} : 192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local}};"
    tile_size_h.write_text(s.replace(PINNED_HD64, tiled))


PINNED_MIN_BLOCKS = "static constexpr uint32_t MinBlocksPerMultiprocessor = 1;"
PINNED_MMA_REGS = "MmaRegisterRequirement = NumMmaWarpGroups == 1 ? 256 :"
PINNED_NUM_SM = "params.num_sm = get_device_prop()->multiProcessorCount - sm_margin;"


def occupy(hopper: Path, ctas: int) -> int:
    """`ctas` CTAs an SM for the one-MMA-warpgroup kernels (kBlockM 64): their launch bound, the MMA warpgroup's
    setmaxnreg target set to what the CTA's share of the register file holds once the producer drops to 56, and a
    persistent grid of `ctas` CTAs an SM (the varlen scheduler strides by gridDim.x). Returns that register target."""
    base = 65536 // ctas // 256 // 8 * 8
    regs = min(256, 2 * base - 56)
    if base < 56 or regs < 128:
        raise SystemExit(f"{ctas} CTAs an SM leave {base} registers a thread: too few")
    k, a = hopper / "flash_fwd_kernel_sm90.h", hopper / "flash_api.cpp"
    ks, s = k.read_text(), a.read_text()
    if ks.count(PINNED_MIN_BLOCKS) != 1 or ks.count(PINNED_MMA_REGS) != 1 or s.count(PINNED_NUM_SM) != 2:
        raise SystemExit(f"{hopper}: the pinned launch bound, register split or num_sm lines are not there")
    ks = ks.replace(PINNED_MIN_BLOCKS, f"static constexpr uint32_t MinBlocksPerMultiprocessor = NumMmaWarpGroups == 1 ? {ctas} : 1;")
    k.write_text(ks.replace(PINNED_MMA_REGS, f"MmaRegisterRequirement = NumMmaWarpGroups == 1 ? {regs} :"))
    a.write_text(s.replace(PINNED_NUM_SM, f"params.num_sm = (get_device_prop()->multiProcessorCount - sm_margin) * {ctas};"))
    return regs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--tile-m", type=int)
    ap.add_argument("--ctas-per-sm", type=int, default=1)
    ap.add_argument("--jobs", default="8")
    ap.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"), help="the CUDA toolkit whose libcuda stub links")
    args = ap.parse_args()
    hop = args.tree / "hopper"
    if args.tile_m:
        retile(hop / "tile_size.h", args.tile_m)
    mma_regs = occupy(hop, args.ctas_per_sm) if args.ctas_per_sm != 1 else None
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = args.jobs
    from torch.utils.cpp_extension import load

    kernels = [hop / "instantiations" / f for f in ("flash_fwd_hdim64_bf16_paged_sm90.cu", "flash_fwd_hdim64_bf16_packgqa_sm90.cu")]
    sources = [hop / "flash_api.cpp", hop / "fa3_torch_lib_verity.cpp", hop / "flash_prepare_scheduler.cu"] + kernels
    defines = ["FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT", "FLASHATTENTION_DISABLE_UNEVEN_K",
               "FLASHATTENTION_DISABLE_PYBIND", "FLASHATTENTION_VARLEN_ONLY", "FLASHATTENTION_PACKGQA_ONLY",
               "FLASHATTENTION_DISABLE_CLUSTER", "FLASHATTENTION_DISABLE_SM8x", "FLASHATTENTION_DISABLE_APPENDKV",
               "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "CUTLASS_ENABLE_GDC_FOR_SM90", "TORCH_TARGET_VERSION=0x020A000000000000ULL",
               "USE_CUDA", "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1", "FLASHATTENTION_DISABLE_FP16", "FLASHATTENTION_DISABLE_FP8",
               "FLASHATTENTION_DISABLE_SOFTCAP", "FLASHATTENTION_DISABLE_LOCAL", "FLASHATTENTION_DISABLE_SPLIT",
               "FLASHATTENTION_DISABLE_HDIM96", "FLASHATTENTION_DISABLE_HDIM128", "FLASHATTENTION_DISABLE_HDIM192",
               "FLASHATTENTION_DISABLE_HDIM256", "FLASHATTENTION_DISABLE_HDIMDIFF64", "FLASHATTENTION_DISABLE_HDIMDIFF192",
               "VERITY_TAP=0", "VERITY_HAVE_FA3_HDIM64_BF16"]
    dflags = ["-D" + d for d in defines]
    nvcc = ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-Xptxas", "-v"] + dflags
    args.build_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load(name=args.name, sources=[str(s) for s in sources], extra_cflags=["-O3", "-DNDEBUG", "-std=c++20"] + dflags,
         extra_cuda_cflags=nvcc, extra_ldflags=[f"-L{args.cuda}/lib64/stubs", "-lcuda"],
         extra_include_paths=[str(hop), str(args.tree / "csrc" / "common"), str(args.tree / "csrc" / "cutlass" / "include")],
         build_directory=str(args.build_dir), verbose=True, is_python_module=False, with_cuda=True)
    tile = [ln.strip() for ln in (hop / "tile_size.h").read_text().splitlines()
            if "is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192" in ln and not ln.strip().startswith("//")]
    info = {"name": args.name, "tile_m": args.tile_m, "ctas_per_sm": args.ctas_per_sm, "mma_regs": mma_regs,
            "seconds": round(time.time() - t0, 1),
            "so": str(args.build_dir / f"{args.name}.so"), "hd64_tile_line": tile}
    (args.build_dir / "build_info.json").write_text(json.dumps(info, indent=1))
    print("BUILD-OK", json.dumps(info))


if __name__ == "__main__":
    main()
