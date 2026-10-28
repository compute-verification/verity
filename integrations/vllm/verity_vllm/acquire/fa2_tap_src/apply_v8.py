#!/usr/bin/env python3
"""Turn a pinned vllm-flash-attn tree (506341a1) into the v8 tapped tree = `verity_tap_v7.patch` + the
head-dim-parametric M1 layout.

  apply_v8.py <tree>/csrc/flash_attn/src <this src dir>

Steps (idempotent; each guarded by a marker):
 1. `git apply verity_tap_v7.patch` at the tree root when `verity_tap.h` is absent (the v7 tap: flash.h `Verity_tap`, kernel
    macros in `compute_attn_1rowblock_splitkv`, softmax.h `tap_scale`/`tap_inv_sum`, launch-template capacity check);
 2. overwrite `verity_tap.h` with this directory's v8 version (`Verity_mat_layout(HB, M, NB, BN, D)`,
    `Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>`);
 3. flash.h: `Verity_tap` gains `int BN, D;` (the host-declared kernel tile the stream layout is arithmetic in);
 4. flash_fwd_kernel.h: the mode-3 `VERITY_TAP_INIT` instantiates `Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>`;
 5. flash_fwd_launch_template.h: the mode-3 arm condition `Kernel_traits::kBlockN == 128` becomes
    `kBlockN == params.vt.BN && kHeadDim == params.vt.D` (declared tile must equal the selected kernel's; else skipped++);
 6. the MS class (compiled only with `VERITY_MAT_SRC & 64`): softmax.h gains `Softmax::tap_max_scaled` and `scale_apply_exp2_ms`,
    `scale_apply_exp2` with the same arithmetic that also keeps each row's `max_scaled` -- the register its exp2 FMAs read -- and
    `softmax_rescale_o` calls it through `VERITY_SCALE_APPLY_EXP2` (the pinned call without the bit); the mode-3 `VERITY_TAP_P` stores it
    (`VERITY_MAT_MS`, verity_tap.h).
Everything else in the tree (kernels, softmax, API) is the v7 state.
"""
import os
import shutil
import subprocess
import sys

src, mine = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
tree = os.path.abspath(os.path.join(src, '..', '..', '..'))   # <tree>/csrc/flash_attn/src -> <tree>

# 1. v7 patch
if not os.path.exists(os.path.join(src, 'verity_tap.h')):
    subprocess.run(['git', 'apply', os.path.join(mine, 'verity_tap_v7.patch')], cwd=tree, check=True)
    print('[apply_v8] verity_tap_v7.patch applied')

# 2. v8 tap header
shutil.copy(os.path.join(mine, 'verity_tap.h'), os.path.join(src, 'verity_tap.h'))

# 3. flash.h: BN, D in Verity_tap
p = os.path.join(src, 'flash.h')
s = open(p).read()
if 'int BN, D;' not in s:
    old = '    int tag;              // v7: per-launch tag (layer / step id), bound into every leaf header\n'
    assert old in s, 'flash.h: Verity_tap.tag line not found'
    s = s.replace(old, old + '    int BN, D;            // v8: host-declared kernel tile (kBlockN, kHeadDim) the M1 layout is arithmetic in; must equal the selected kernel\'s\n')
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
new = ('        // v8: the host declared the tile (BN = kBlockN, D = kHeadDim) its layout is arithmetic in; arm only if this kernel IS that tile\n'
       '        const bool ok = params.num_splits <= 1 && Kernel_traits::kBlockN == params.vt.BN && Kernel_traits::kHeadDim == params.vt.D\n'
       '                        && Kernel_traits::kBlockM == 64 && params.vt.diag_ptr != nullptr && params.b * params.h <= params.vt.H\n'
       '                        && params.seqlen_q <= params.vt.M && need_nb <= params.vt.NB;\n')
if 'Kernel_traits::kBlockN == params.vt.BN' not in s:
    assert old in s, 'flash_fwd_launch_template.h: mode-3 arm condition not found'
    s = s.replace(old, new)
    open(p, 'w').write(s)
    print('[apply_v8] flash_fwd_launch_template.h: arm iff (kBlockN, kHeadDim) == (vt.BN, vt.D)')

# 6. the MS class: softmax.h keeps max_scaled, the mode-3 VERITY_TAP_P stores it (both only with VERITY_MAT_SRC & 64)
MS_ON = '#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)\n'
p = os.path.join(src, 'softmax.h')
s = open(p).read()
if 'scale_apply_exp2_ms' not in s:
    member = '    TensorT row_max, row_sum;\n'
    assert s.count(member) == 1, 'softmax.h: Softmax row_max/row_sum member not found'
    s = s.replace(member, member + MS_ON + '    TensorT tap_max_scaled;   // [verity MS] each row\'s max * scale this block\'s exp2 FMAs read\n#endif\n')
    anchor = '// Apply the exp to all the elements.\ntemplate <bool zero_init=true, typename Engine0'
    assert s.count(anchor) == 1, 'softmax.h: max_scale_exp2_sum anchor not found'
    ms_fn = MS_ON + r'''// [verity MS] scale_apply_exp2 with the same arithmetic that also keeps each row's max_scaled, the register its exp2 FMAs read, in `ms`
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void scale_apply_exp2_ms(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale,
                                                    Tensor<Engine2, Layout2> &ms) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        ms(mi) = max_scaled;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}
#define VERITY_SCALE_APPLY_EXP2(t_, m_, s_) FLASH_NAMESPACE::scale_apply_exp2_ms(t_, m_, s_, tap_max_scaled)
#else
#define VERITY_SCALE_APPLY_EXP2(t_, m_, s_) FLASH_NAMESPACE::scale_apply_exp2(t_, m_, s_)
#endif

'''
    s = s.replace(anchor, ms_fn + anchor)
    call = '            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);\n'
    assert s.count(call) == 2, f'softmax.h: {s.count(call)} scale_apply_exp2 calls in softmax_rescale_o (expected 2)'
    s = s.replace(call, '            VERITY_SCALE_APPLY_EXP2(scores, row_max, softmax_scale_log2);\n')
    open(p, 'w').write(s)
    print('[apply_v8] softmax.h: tap_max_scaled via scale_apply_exp2_ms (VERITY_MAT_SRC & 64)')
p = os.path.join(src, 'flash_fwd_kernel.h')
s = open(p).read()
if 'VERITY_MAT_MS' not in s:
    old = '    if constexpr (VERITY_MAT_SRC & 8) { VERITY_MAT(vmat.rowstat(softmax.row_max, softmax.tap_scale, softmax.row_sum, n_block, vt_step)) }\n'
    assert s.count(old) == 1, 'flash_fwd_kernel.h: mode-3 VERITY_TAP_P rowstat line not found'
    s = s.replace(old, old[:-1] + ' \\\n    VERITY_MAT_MS(vmat, softmax, n_block)\n')
    open(p, 'w').write(s)
    print('[apply_v8] flash_fwd_kernel.h: VERITY_TAP_P stores the MS class')
print('APPLY-V8-OK', src)
