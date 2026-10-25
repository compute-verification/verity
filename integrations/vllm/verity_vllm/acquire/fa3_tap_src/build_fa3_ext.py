#!/usr/bin/env python
"""Build the FA3 (hopper/) forward of the pinned vllm-flash-attn (506341a1) as a standalone torch extension --
the M1 hidden-value tap `verity_fa3_matReq.so` (VERITY_TAP=3, VERITY_MAT_SRC=59) or an untapped control (VERITY_TAP=0).

Mirrors the flags of the pinned CMake build of `_vllm_fa3_C` (CMakeLists.txt: -O3 --use_fast_math --expt-relaxed-constexpr
--expt-extended-lambda, -gencode arch=compute_90a,code=sm_90a, CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED, CUTLASS_ENABLE_GDC_FOR_SM90,
FLASHATTENTION_{DISABLE_BACKWARD,DISABLE_DROPOUT,DISABLE_UNEVEN_K,DISABLE_PYBIND,VARLEN_ONLY,PACKGQA_ONLY,DISABLE_CLUSTER,DISABLE_SM8x,
DISABLE_APPENDKV}, TORCH_TARGET_VERSION, USE_CUDA) and narrows the build to what batch-invariant bf16 serving launches: bf16 only,
head dims from --hdims (64, 128), paged-KV PackGQA forward, num_splits == 1 (no split/combine), no softcap / local / fp8 / fp16.
The narrowing macros only remove template branches (each remaining instantiation is the same `run_mha_fwd_<...>` the installed
library dispatches to); `fa3_tap_xcheck.py` verifies out / lse bit-identical to the installed `_vllm_fa3_C.fwd`.

Usage:
  verity-vllm fa3-tap-build --tree /path/to/vllm-flash-attn --name verity_fa3_matReq --build-dir /path/build --hdims 64,128 \
      [--define VERITY_TAP=3 --define VERITY_MAT_SRC=59]
Writes <build-dir>/build_info.json and <build-dir>/resource_usage.txt.
"""
import glob, hashlib, json, os, subprocess, time
from dataclasses import dataclass

from verity_vllm import settings
from verity_vllm.config import option


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, kw_only=True)
class Options:
    tree: str = option('--tree', help='vllm-flash-attn tree root (contains hopper/, csrc/common, csrc/cutlass)')
    name: str = option('--name')
    build_dir: str = option('--build-dir')
    src_dir: str = option('--src-dir', default=os.path.dirname(os.path.abspath(__file__)))
    define: list[str] = option('--define', action='append', default=[])
    hdims: str = option('--hdims', default='64,128')
    arch: str = option('--arch', default='9.0a')
    no_ptxas_v: bool = option('--no-ptxas-v', default=False, action='store_true')
    tree_commit: str = option('--tree-commit', default='')


def main(args: Options):
    settings.setdefault('CUDA_HOME', '/usr/local/cuda')
    settings.put('TORCH_CUDA_ARCH_LIST', args.arch)
    settings.setdefault('MAX_JOBS', '32')
    import torch
    from torch.utils.cpp_extension import load

    tree = os.path.abspath(args.tree)
    hop = os.path.join(tree, 'hopper')
    hdims = sorted({int(h) for h in args.hdims.split(',') if h})
    assert hdims and all(h in (64, 128) for h in hdims), hdims
    kernel_files = [os.path.join(hop, 'instantiations', f'flash_fwd_hdim{hd}_bf16_paged_sm90.cu') for hd in hdims]
    for kf in kernel_files:
        assert os.path.exists(kf), kf
    sources = [os.path.join(hop, 'flash_api.cpp'), os.path.join(hop, 'fa3_torch_lib_verity.cpp'),
               os.path.join(hop, 'flash_prepare_scheduler.cu'), os.path.join(args.src_dir, 'fa3_stubs.cu')] + kernel_files
    defines = ['FLASHATTENTION_DISABLE_BACKWARD', 'FLASHATTENTION_DISABLE_DROPOUT', 'FLASHATTENTION_DISABLE_UNEVEN_K',
               'FLASHATTENTION_DISABLE_PYBIND', 'FLASHATTENTION_VARLEN_ONLY', 'FLASHATTENTION_PACKGQA_ONLY',
               'FLASHATTENTION_DISABLE_CLUSTER', 'FLASHATTENTION_DISABLE_SM8x', 'FLASHATTENTION_DISABLE_APPENDKV',
               'CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED', 'CUTLASS_ENABLE_GDC_FOR_SM90',
               'TORCH_TARGET_VERSION=0x020A000000000000ULL', 'USE_CUDA', 'CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1',
               # narrowing (branches the accepted bf16 paged serving launches never take)
               'FLASHATTENTION_DISABLE_FP16', 'FLASHATTENTION_DISABLE_FP8', 'FLASHATTENTION_DISABLE_SOFTCAP',
               'FLASHATTENTION_DISABLE_LOCAL', 'FLASHATTENTION_DISABLE_SPLIT', 'FLASHATTENTION_DISABLE_HDIM96',
               'FLASHATTENTION_DISABLE_HDIM192', 'FLASHATTENTION_DISABLE_HDIM256', 'FLASHATTENTION_DISABLE_HDIMDIFF64',
               'FLASHATTENTION_DISABLE_HDIMDIFF192'] + args.define
    for hd in hdims:
        defines.append(f'VERITY_HAVE_FA3_HDIM{hd}_BF16')
    dflags = ['-D' + d for d in defines]
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++20'] + dflags
    # sm_90a SASS via TORCH_CUDA_ARCH_LIST=9.0a (torch adds -gencode arch=compute_90a,code=sm_90a; the pinned CMake build's arch)
    nvcc_flags = ['-O3', '-DNDEBUG', '-std=c++17', '--expt-relaxed-constexpr', '--expt-extended-lambda', '--use_fast_math'] + dflags
    if not args.no_ptxas_v:
        nvcc_flags += ['-Xptxas', '-v']
    include_dirs = [hop, os.path.join(tree, 'csrc', 'common'), os.path.join(tree, 'csrc', 'cutlass', 'include')]

    os.makedirs(args.build_dir, exist_ok=True)
    nvcc = os.path.join(settings.raw('CUDA_HOME'), 'bin', 'nvcc')
    nvcc_ver = subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
    info = {
        'name': args.name, 'tree': tree, 'tree_commit': args.tree_commit,
        'sources': sources, 'source_sha256': {s: sha256(s) for s in sources},
        'header_sha256': {p: sha256(p) for p in sorted(glob.glob(os.path.join(hop, '*.h')) + glob.glob(os.path.join(hop, '*.hpp')))},
        'defines': defines, 'cxx_flags': cxx_flags, 'nvcc_flags': nvcc_flags, 'arch': args.arch, 'hdims': hdims,
        'nvcc_version': nvcc_ver, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cuda_home': settings.raw('CUDA_HOME'), 'max_jobs': settings.raw('MAX_JOBS'),
    }
    t0 = time.time()
    # CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL (as the pinned build): the TMA descriptor encode is a direct driver call -> link libcuda
    # (the toolkit's stub at link time; the driver's libcuda.so.1 at load time, as _vllm_fa3_C).
    ldflags = [f"-L{os.path.join(settings.raw('CUDA_HOME'), 'lib64', 'stubs')}", "-lcuda"]
    info['ldflags'] = ldflags
    load(name=args.name, sources=sources, extra_cflags=cxx_flags, extra_cuda_cflags=nvcc_flags, extra_ldflags=ldflags,
         extra_include_paths=include_dirs, build_directory=args.build_dir, verbose=True,
         is_python_module=False, with_cuda=True)
    info['compile_seconds'] = round(time.time() - t0, 1)
    so = os.path.join(args.build_dir, args.name + '.so')
    info['so'] = so
    info['so_sha256'] = sha256(so) if os.path.exists(so) else None
    info['so_bytes'] = os.path.getsize(so) if os.path.exists(so) else None
    cuobjdump = os.path.join(settings.raw('CUDA_HOME'), 'bin', 'cuobjdump')
    ru = subprocess.run([cuobjdump, '--dump-resource-usage', so], capture_output=True, text=True)
    with open(os.path.join(args.build_dir, 'resource_usage.txt'), 'w') as f:
        f.write(ru.stdout + ru.stderr)
    ops = [o for o in dir(getattr(torch.ops, args.name)) if not o.startswith('_')]
    info['ops'] = ops
    with open(os.path.join(args.build_dir, 'build_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print('BUILD-OK', args.name, info['compile_seconds'], 's', so, ops)
