#!/usr/bin/env python
"""Verity: build one FA2 forward variant as a standalone torch extension.

Compiles the pinned vllm-flash-attn (506341a1) hdim64 bf16 split-KV forward kernels
(+ pinned flash_api.cpp, our varlen_fwd-only registration and link stubs) with
torch.utils.cpp_extension.load, mirroring the flags of the pinned CMake build
(_vllm_fa2_C: -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda,
FLASHATTENTION_DISABLE_{BACKWARD,DROPOUT,PYBIND}, USE_CUDA, TORCH_TARGET_VERSION,
gencode sm_80 real -- the installed _vllm_fa2_C.abi3.so ships sm_80 SASS only).

Usage:
  verity-vllm fa2-tap-build --tree /path/to/csrc --name verity_fa2_base --build-dir /path/build \
      [--define VERITY_TAP=1 ...] [--files split,split_align] [--extra-src file.cu ...]
Writes <build-dir>/build_info.json and <build-dir>/resource_usage.txt.
"""
import hashlib, json, os, subprocess, sys, time, glob
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
    tree: str = option('--tree', help='csrc root (contains flash_attn/, common/, cutlass/)')
    name: str = option('--name')
    build_dir: str = option('--build-dir')
    src_dir: str = option('--src-dir', default=os.path.dirname(os.path.abspath(__file__)))
    define: list[str] = option('--define', action='append', default=[])
    files: str = option('--files', default='split_causal,split_align_noncausal,split_noncausal,split_align_causal',
                        help='comma list of kernel .cu groups to compile (per head dim)')
    hdims: str = option('--hdims', default='64',
                        help='v8: comma list of bf16 head dims whose split-KV kernels are compiled (64,96,128,256)')
    arch: str = option('--arch', default='8.0')
    no_ptxas_v: bool = option('--no-ptxas-v', default=False, action='store_true')
    tree_commit: str = option('--tree-commit', default='')


def main(args: Options):
    settings.setdefault('CUDA_HOME', '/usr/local/cuda-12.9')
    settings.put('TORCH_CUDA_ARCH_LIST', args.arch)
    settings.setdefault('MAX_JOBS', '32')
    import torch
    from torch.utils.cpp_extension import load

    tree = os.path.abspath(args.tree)
    fa = os.path.join(tree, 'flash_attn')
    src = os.path.join(fa, 'src')
    groups = {
        'split_causal': 'flash_fwd_split_hdim{hd}_bf16_causal_sm80.cu',
        'split_noncausal': 'flash_fwd_split_hdim{hd}_bf16_sm80.cu',
        'split_align_causal': 'flash_fwd_split_align_hdim{hd}_bf16_causal_sm80.cu',
        'split_align_noncausal': 'flash_fwd_split_align_hdim{hd}_bf16_sm80.cu',
        'fwd_causal': 'flash_fwd_hdim{hd}_bf16_causal_sm80.cu',
        'fwd_noncausal': 'flash_fwd_hdim{hd}_bf16_sm80.cu',
    }
    hdims = sorted({int(h) for h in args.hdims.split(',') if h})
    assert hdims and all(h in (64, 96, 128, 256) for h in hdims), hdims
    kernel_files = [os.path.join(src, groups[g].format(hd=hd)) for hd in hdims for g in args.files.split(',') if g]
    for kf in kernel_files:
        assert os.path.exists(kf), kf
    sources = [os.path.join(fa, 'flash_api.cpp'),
               os.path.join(args.src_dir, 'fa2_torch_lib_verity.cpp'),
               os.path.join(args.src_dir, 'fa2_stubs.cu')] + kernel_files
    defines = ['FLASHATTENTION_DISABLE_BACKWARD', 'FLASHATTENTION_DISABLE_DROPOUT',
               'FLASHATTENTION_DISABLE_PYBIND', 'USE_CUDA',
               'TORCH_TARGET_VERSION=0x020A000000000000ULL',
               'CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1'] + args.define
    for hd in hdims:
        if 'split_causal' in args.files or 'split_noncausal' in args.files:
            defines.append(f'VERITY_HAVE_SPLIT_HDIM{hd}_BF16')     # fa2_stubs.cu: no stub for these -> the pinned kernels link
        if 'fwd_causal' in args.files or 'fwd_noncausal' in args.files:
            assert hd == 64, 'non-split fwd kernels are only wired for hdim 64'
            defines.append('VERITY_HAVE_FWD_HDIM64_BF16')
    dflags = ['-D' + d for d in defines]
    cxx_flags = ['-O3', '-DNDEBUG', '-std=c++20'] + dflags
    nvcc_flags = ['-O3', '-DNDEBUG', '-std=c++17', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                  '--use_fast_math', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                  '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__'] + dflags
    # NOTE: the -U flags above are NOT used by the pinned build; keep them out unless requested.
    nvcc_flags = [f for f in nvcc_flags if not f.startswith('-U')]
    if not args.no_ptxas_v:
        nvcc_flags += ['-Xptxas', '-v']
    include_dirs = [fa, src, os.path.join(tree, 'common'), os.path.join(tree, 'cutlass', 'include')]

    os.makedirs(args.build_dir, exist_ok=True)
    nvcc = os.path.join(settings.raw('CUDA_HOME'), 'bin', 'nvcc')
    nvcc_ver = subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
    info = {
        'name': args.name, 'tree': tree, 'tree_commit': args.tree_commit,
        'sources': sources, 'source_sha256': {s: sha256(s) for s in sources},
        'header_sha256': {p: sha256(p) for p in sorted(glob.glob(os.path.join(src, '*.h')))},
        'defines': defines, 'cxx_flags': cxx_flags, 'nvcc_flags': nvcc_flags, 'arch': args.arch, 'hdims': hdims,
        'nvcc_version': nvcc_ver, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cuda_home': settings.raw('CUDA_HOME'), 'max_jobs': settings.raw('MAX_JOBS'),
    }
    t0 = time.time()
    load(name=args.name, sources=sources, extra_cflags=cxx_flags, extra_cuda_cflags=nvcc_flags,
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
