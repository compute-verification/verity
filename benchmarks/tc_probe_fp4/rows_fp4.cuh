// The FP4-path mma.sync rows on sm_120a (lane pouw rtx-pro GPU 4): one mnemonic per RowKind, shared by the probe's generic kernel
// (mma_fp4.cu, row_tiles_kernel) and the W1 price loops (w1_price.cu), so the rate is measured on the instruction the capture gates.
//
// Operand registers: a[4] and b[2] as the PTX fragment layouts in mma_fp4.cu's header; K = 64 kinds read E2M1 nibbles, K = 32 kinds
// 8-bit containers (an E2M1 code at bits [2,6), an E4M3 byte as is).  sfa / sfb: one byte per K-block, thread-id selectors {0, 0}.
#pragma once

#include <stdint.h>

enum RowKind {
    R_NVF4 = 0,               // kind::mxf4nvf4 scale_vec::4X ue4m3 (the nvfp4_tiles_kernel instruction)
    R_MXF4 = 1,               // kind::mxf4nvf4 scale_vec::2X ue8m0
    R_F8F6F4_E2M1 = 2,        // kind::f8f6f4 e2m1 x e2m1, unscaled
    R_F8F6F4_E2M1_E4M3 = 3,   // kind::f8f6f4 e2m1 x e4m3
    R_F8F6F4_E4M3_E2M1 = 4,   // kind::f8f6f4 e4m3 x e2m1
    R_MXF8F6F4_E2M1 = 5,      // kind::mxf8f6f4 scale_vec::1X ue8m0, e2m1 x e2m1
    R_MXF8F6F4_E2M1_E4M3 = 6, // kind::mxf8f6f4 scale_vec::1X ue8m0, e2m1 x e4m3
    R_MXF8F6F4_E4M3_E2M1 = 7, // kind::mxf8f6f4 scale_vec::1X ue8m0, e4m3 x e2m1
    R_F8F6F4_E4M3 = 8,        // kind::f8f6f4 e4m3 x e4m3: the FP8 reference rate in w1_price.cu only (its capture is benchmarks/tc_probe's)
    R_SP_NVF4 = 9,            // mma.sp m16n8k128 kind::mxf4nvf4 4X ue4m3: w1_price.cu only (mma_fp4.cu's sparse_tiles_kernel<0>, same text)
    R_SP_MXF4 = 10,           // mma.sp m16n8k128 kind::mxf4nvf4 2X ue8m0: likewise (sparse_tiles_kernel<1>)
};  // kind::mxf4nvf4 scale_vec::4X with ue8m0 is refused by ptxas 12.9 ("Illegal modifier '.scale_vec::4X' ... with type '.ue8m0'")

template <int KIND> struct RowTraits {
    static constexpr bool SPARSE = KIND == R_SP_NVF4 || KIND == R_SP_MXF4;
    static constexpr int K = SPARSE ? 128 : (KIND == R_NVF4 || KIND == R_MXF4) ? 64 : 32;  // logical K
    static constexpr int NB = (KIND == R_NVF4 || KIND == R_SP_NVF4) ? 4 : (KIND == R_MXF4 || KIND == R_SP_MXF4) ? 2
                              : (KIND >= R_MXF8F6F4_E2M1 && KIND <= R_MXF8F6F4_E4M3_E2M1) ? 1 : 0;
};

#define ROW_MMA_UNSCALED(MNEMONIC)                                                                          \
    asm volatile(MNEMONIC " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                  \
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                                          \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), \
                   "f"(c[2]), "f"(c[3]))
#define ROW_MMA_SCALED(MNEMONIC)                                                                               \
    asm volatile(MNEMONIC " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n" \
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                                             \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),    \
                   "f"(c[2]), "f"(c[3]), "r"(sfa), "r"(sfb))
// b[0..3] (K = 128 logical), e: the lane's ordered metadata word (sparsity selector 0x0)
#define ROW_MMA_SPARSE(MNEMONIC)                                                                                           \
    asm volatile(MNEMONIC " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17}, {0, 0}, {%18}, {0, 0};\n" \
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                                                         \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "f"(c[0]),     \
                   "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(e), "r"(sfa), "r"(sfb))

// b: two registers (b[2]), four for the sparse kinds (b[4]); e is read by the sparse kinds only.
template <int KIND>
__device__ __forceinline__ void row_mma(float d[4], const uint32_t a[4], const uint32_t* b, const float c[4], uint32_t sfa, uint32_t sfb,
                                        uint32_t e = 0) {
    if constexpr (KIND == R_SP_NVF4) {
        ROW_MMA_SPARSE("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3");
    } else if constexpr (KIND == R_SP_MXF4) {
        ROW_MMA_SPARSE("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0");
    } else if constexpr (KIND == R_NVF4) {
        ROW_MMA_SCALED("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3");
    } else if constexpr (KIND == R_MXF4) {
        ROW_MMA_SCALED("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0");
    } else if constexpr (KIND == R_F8F6F4_E2M1) {
        ROW_MMA_UNSCALED("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32");
    } else if constexpr (KIND == R_F8F6F4_E2M1_E4M3) {
        ROW_MMA_UNSCALED("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32");
    } else if constexpr (KIND == R_F8F6F4_E4M3_E2M1) {
        ROW_MMA_UNSCALED("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32");
    } else if constexpr (KIND == R_MXF8F6F4_E2M1) {
        ROW_MMA_SCALED("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0");
    } else if constexpr (KIND == R_MXF8F6F4_E2M1_E4M3) {
        ROW_MMA_SCALED("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0");
    } else if constexpr (KIND == R_MXF8F6F4_E4M3_E2M1) {
        ROW_MMA_SCALED("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0");
    } else {
        static_assert(KIND == R_F8F6F4_E4M3, "unknown row kind");
        ROW_MMA_UNSCALED("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32");
    }
}
