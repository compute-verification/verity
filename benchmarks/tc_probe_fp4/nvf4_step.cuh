// The NVFP4 atom on sm_120a, shared by f3_lut.cu and f2_strassen.cu: its align-add on integers (verity.ml.circuits.silicon
// BLACKWELL_SM120_NVF4, BlockScaledAlignAdd._transducer) for the CUDA-core routes, and the OMMA tile that is their unit.
#pragma once

#include <stdint.h>

namespace {

// Branch-free: shift counts are clamped to [0, 63], and a select replaces every early return.

// m * 2^e truncated toward zero onto the grid 2^lsb, as a multiple of 2^lsb (|m| < 2^36, and e - lsb <= 27 wherever it is kept).
__device__ __forceinline__ long long trunc_rz(long long m, int e, int lsb) {
    const int d = e - lsb;
    const unsigned long long am = (unsigned long long)(m < 0 ? -m : m);
    const unsigned long long r = (am << min(max(d, 0), 63)) >> min(max(-d, 0), 63);
    return m < 0 ? -(long long)r : (long long)r;
}

__device__ __forceinline__ uint32_t to_f32_rz(long long n, int lsb) {
    const bool neg = n < 0;
    const unsigned long long mag = (unsigned long long)(neg ? -n : n);
    const int e_val = 63 - __clzll((long long)mag) + lsb;
    const int grid = e_val >= -126 ? e_val - 23 : -149;
    const int shift = grid - lsb;
    const unsigned long long q = (mag << min(max(-shift, 0), 63)) >> min(max(shift, 0), 63);
    const int e = 63 - __clzll((long long)q) + grid;
    const uint32_t bits = e >= -126 ? ((uint32_t)(e + 127) << 23) | (uint32_t)(q & 0x7FFFFFull) : (uint32_t)q;
    return q == 0 ? 0u : bits | (neg ? 0x80000000u : 0u);         // q == 0: below the subnormal grid, +0 (unreachable with UE4M3)
}

// One firing on one word: c the accumulator bits; per block g its group dot gd (quarters), scale-mantissa product mm, unit
// exponent xg (-2 + ea + eb) and participation (a nonzero product and both scale mantissas nonzero).
__device__ __forceinline__ uint32_t nvf4_step(uint32_t c, const int gd[4], const int mm[4], const int xg[4], const bool part[4]) {
    const int NONE = -100000;
    int anchor = NONE;
#pragma unroll
    for (int g = 0; g < 4; g++) anchor = part[g] ? max(anchor, xg[g] - 27) : anchor;   // frac_bits 27
    const int ef = (c >> 23) & 0xFF;
    const int frac = (int)(c & 0x7FFFFFu);
    const int am0 = ef ? (frac | 0x800000) : frac;
    const int am = (c >> 31) ? -am0 : am0;
    const int ae = ef ? ef - 150 : -149;
    anchor = am != 0 ? max(anchor, (ef ? ef : 1) - 127 - 35) : anchor;                 // acc_window 36
    const int lsb = max(anchor, -174);                                                   // grid_floor
    long long total = trunc_rz(am, ae, lsb);
#pragma unroll
    for (int g = 0; g < 4; g++) total += trunc_rz((long long)gd[g] * mm[g], xg[g], lsb);
    return (anchor == NONE || total == 0) ? 0u : to_f32_rz(total, lsb);
}

// mma.sync m16n8k64 kind::mxf4nvf4 4X ue4m3, D += A B (the dense NVFP4 tile)
__device__ __forceinline__ void omma(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

}  // namespace
