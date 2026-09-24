#!/usr/bin/env python3
"""R13 lane fa2: turn a pinned vllm-flash-attn tree (506341a1) into the v8 tapped tree = R9 `verity_tap_v7.patch` + the
head-dim-parametric M1 layout (C-FA2-HDIM).

  apply_v8.py <tree>/csrc/flash_attn/src <this src dir>

Steps (idempotent; each guarded by a marker):
 1. `git apply verity_tap_v7.patch` at the tree root when `verity_tap.h` is absent (the R9 tap: flash.h `Verity_tap`, kernel
    macros in `compute_attn_1rowblock_splitkv`, softmax.h `tap_scale`/`tap_inv_sum`, launch-template capacity check);
 2. overwrite `verity_tap.h` with this directory's v8 version (`Verity_mat_layout(HB, M, NB, BN, D)`,
    `Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>`);
 3. flash.h: `Verity_tap` gains `int BN, D;` (the host-declared kernel tile the stream layout is arithmetic in);
 4. flash_fwd_kernel.h: the mode-3 `VERITY_TAP_INIT` instantiates `Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>`;
 5. flash_fwd_launch_template.h: the mode-3 arm condition `Kernel_traits::kBlockN == 128` becomes
    `kBlockN == params.vt.BN && kHeadDim == params.vt.D` (declared tile must equal the selected kernel's; else skipped++).
Everything else in the tree (kernels, softmax, API) is the R9 v7 state.
"""
import os
import shutil
import subprocess
import sys

src, mine = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
tree = os.path.abspath(os.path.join(src, '..', '..', '..'))   # <tree>/csrc/flash_attn/src -> <tree>

# 1. R9 v7 patch
if not os.path.exists(os.path.join(src, 'verity_tap.h')):
    subprocess.run(['git', 'apply', os.path.join(mine, 'verity_tap_v7.patch')], cwd=tree, check=True)
    print('[apply_v8] verity_tap_v7.patch applied')

# 2. v8 tap header
shutil.copy(os.path.join(mine, 'verity_tap.h'), os.path.join(src, 'verity_tap.h'))

# 3. flash.h: BN, D in Verity_tap
p = os.path.join(src, 'flash.h')
s = open(p).read()
if 'int BN, D;' not in s:
    old = '    int tag;              // R9 cmt-hidden: per-launch tag (layer / step id), bound into every leaf header\n'
    assert old in s, 'flash.h: Verity_tap.tag line not found'
    s = s.replace(old, old + '    int BN, D;            // R13 fa2 v8: host-declared kernel tile (kBlockN, kHeadDim) the M1 layout is arithmetic in; must equal the selected kernel\'s\n')
    open(p, 'w').write(s)
    print('[apply_v8] flash.h: Verity_tap.BN/D added')

# 4. kernel macro: templated ctx
p = os.path.join(src, 'flash_fwd_kernel.h')
s = open(p).read()
old = '    Verity_mat_ctx vmat(params.vt, bidb, params.h, bidh, m_block, tidx, kNWarps, binfo.actual_seqlen_q); \\\n'
new = '    Verity_mat_ctx<kBlockM, kBlockN, kHeadDim> vmat(params.vt, bidb, params.h, bidh, m_block, tidx, kNWarps, binfo.actual_seqlen_q); \\\n'
if new not in s:
    assert old in s, 'flash_fwd_kernel.h: mode-3 VERITY_TAP_INIT not found'
    s = s.replace(old, new)
    open(p, 'w').write(s)
    print('[apply_v8] flash_fwd_kernel.h: Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>')

# 5. launch template: declared tile must match
p = os.path.join(src, 'flash_fwd_launch_template.h')
s = open(p).read()
old = ('        const bool ok = params.num_splits <= 1 && Kernel_traits::kBlockN == 128 && params.vt.diag_ptr != nullptr && params.b * params.h <= params.vt.H\n'
       '                        && params.seqlen_q <= params.vt.M && need_nb <= params.vt.NB;\n')
new = ('        // R13 fa2 v8: the host declared the tile (BN = kBlockN, D = kHeadDim) its layout is arithmetic in; arm only if this kernel IS that tile\n'
       '        const bool ok = params.num_splits <= 1 && Kernel_traits::kBlockN == params.vt.BN && Kernel_traits::kHeadDim == params.vt.D\n'
       '                        && Kernel_traits::kBlockM == 64 && params.vt.diag_ptr != nullptr && params.b * params.h <= params.vt.H\n'
       '                        && params.seqlen_q <= params.vt.M && need_nb <= params.vt.NB;\n')
if new not in s:
    assert old in s, 'flash_fwd_launch_template.h: mode-3 arm condition not found'
    s = s.replace(old, new)
    open(p, 'w').write(s)
    print('[apply_v8] flash_fwd_launch_template.h: arm iff (kBlockN, kHeadDim) == (vt.BN, vt.D)')
print('APPLY-V8-OK', src)
