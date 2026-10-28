"""Build the pinned vllm-flash-attn's FA3 forward (hopper/, 506341a1) for bf16 head dim 64 as a torch extension, its hdim-64
causal tile optionally changed, so attn_knobs.py can time it beside the served wheel's `_vllm_fa3_C`.

    python fa3_tile_build.py --tree TREE --name NS --build-dir DIR [--tile-m 128] [--exp2-emu NUM/DEN --exp2-deg 3]

TREE is a copy of the pinned tree with `integrations/vllm/verity_vllm/acquire/fa3_tap_src/apply_fa3_tap.py` applied (its
`fa3_torch_lib_verity.cpp`), built here with VERITY_TAP=0, which is the pinned kernel. The flags are that builder's
(`build_fa3_ext.py`, after the pinned CMake build of `_vllm_fa3_C`), narrowed to head dim 64: the paged-KV and the
contiguous-KV PackGQA forwards (`flash_fwd_hdim64_bf16_paged_sm90.cu`, `flash_fwd_hdim64_bf16_packgqa_sm90.cu`).

--tile-m M rewrites `tile_size_fwd_sm90`'s hdim-64 causal tile from 192 x 128 to M x 128. kBlockN, MmaPV_is_RS and
IntraWGOverlap stay, so every query row still meets the same 128-key blocks, last to first, through the same
online-softmax arithmetic; only which rows share an M tile changes. The op is `torch.ops.NS.fwd`, with
`_vllm_fa3_C.fwd`'s signature.

--exp2-emu NUM/DEN computes NUM of every DEN of each thread's score columns' exp2 on the FMA pipe (`exp2_fma`, a
degree --exp2-deg polynomial) instead of MUFU, which at head dim 64 does one exp2 a score in the time the tensor cores
take for its 128 MACs. It is not byte-equal: the other columns keep FA3's exact exp2f, and an emulated one differs by at
most the polynomial's relative error (`COEFFS`), under bf16's half-ulp, before P is rounded to bf16. Without it softmax.h
is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PINNED_HD64 = "return {192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local};"
EXP_ANCHOR = "// Apply the exp to all the elements."
EXP_LINE = "            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);"
# 2^f on [-0.5, 0.5] with p(0) = 1 pinned, fitted for the least most-relative error: c1, c2, ... (degree 3: 4.6e-4,
# degree 4: 1.2e-5, both evaluated in fp32 by Horner; bf16's half-ulp is 1.95e-3).
COEFFS = {3: (0.6935547590255737, 0.2412298321723938, 0.05179294943809509),
          4: (0.6931335926055908, 0.24021488428115845, 0.05581282824277878, 0.009678369387984276)}


def retile(tile_size_h: Path, m: int) -> None:
    s = tile_size_h.read_text()
    if s.count(PINNED_HD64) != 1:
        raise SystemExit(f"{tile_size_h}: the pinned hdim-64 tile line is not there exactly once")
    tiled = f"return {{is_causal ? {m} : 192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192, is_causal || is_local, !is_local}};"
    tile_size_h.write_text(s.replace(PINNED_HD64, tiled))


def exp2_fma(deg: int) -> str:
    c = COEFFS[deg]
    p = f"{c[-1]!r}f"
    for k in reversed(c[:-1]):
        p = f"fmaf({p}, f, {k!r}f)"
    return f"""// 2^x for x <= 0 on the FMA pipe: x rounded to n by the 1.5 * 2^23 shifter, the polynomial for 2^(x - n) on
// [-0.5, 0.5], and n added to its exponent. x is clamped at -125 so that exponent stays normal: a masked key then weighs
// 2^-125 rather than 0, below every causal row's sum and output, since a valid row always keeps its own key.
__forceinline__ __device__ float exp2_fma(float x) {{
    x = fmaxf(x, -125.f);
    float const r = __fadd_rn(x, 12582912.f);
    float const f = __fsub_rn(x, __fsub_rn(r, 12582912.f));
    float const p = fmaf({p}, f, 1.f);
    return __int_as_float(__float_as_int(p) + (__float_as_int(r) << 23));
}}

"""


def emulate_exp2(softmax_h: Path, num: int, den: int, deg: int) -> None:
    """Every copy of the pinned exp2f line is split, so the tap's `scale_apply_exp2_ms` keeps scale_apply_exp2's arithmetic."""
    s = softmax_h.read_text()
    if s.count(EXP_ANCHOR) != 1 or s.find(EXP_LINE) < s.find(EXP_ANCHOR):
        raise SystemExit(f"{softmax_h}: scale_apply_exp2's anchor is not there once, or no pinned exp2f line follows it")
    if not 0 < num <= den or deg not in COEFFS:
        raise SystemExit(f"--exp2-emu {num}/{den} --exp2-deg {deg}: need 0 < NUM <= DEN and a degree in {sorted(COEFFS)}")
    split = ("            float const x = tensor(mi, ni) * scale - max_scaled;\n"
             f"            tensor(mi, ni) = ni % {den} < {num} ? exp2_fma(x) : exp2f(x);")
    softmax_h.write_text(s.replace(EXP_ANCHOR, exp2_fma(deg) + EXP_ANCHOR).replace(EXP_LINE, split))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--tile-m", type=int)
    ap.add_argument("--exp2-emu", help="NUM/DEN: that share of each thread's score columns' exp2 on the FMA pipe")
    ap.add_argument("--exp2-deg", type=int, default=3, help="the FMA exp2's polynomial degree (3 or 4)")
    ap.add_argument("--jobs", default="8")
    ap.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"), help="the CUDA toolkit whose libcuda stub links")
    args = ap.parse_args()
    hop = args.tree / "hopper"
    if args.tile_m:
        retile(hop / "tile_size.h", args.tile_m)
    emu = None
    if args.exp2_emu:
        num, den = (int(x) for x in args.exp2_emu.split("/"))
        emulate_exp2(hop / "softmax.h", num, den, args.exp2_deg)
        emu = {"num": num, "den": den, "deg": args.exp2_deg, "max_rel": {3: 4.6e-4, 4: 1.2e-5}[args.exp2_deg]}
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
    info = {"name": args.name, "tile_m": args.tile_m, "exp2_emu": emu, "seconds": round(time.time() - t0, 1),
            "so": str(args.build_dir / f"{args.name}.so"), "hd64_tile_line": tile}
    (args.build_dir / "build_info.json").write_text(json.dumps(info, indent=1))
    print("BUILD-OK", json.dumps(info))


if __name__ == "__main__":
    main()
