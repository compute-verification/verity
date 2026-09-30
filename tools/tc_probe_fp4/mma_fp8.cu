// sm_120 FP8 rows of the Blackwell tensor-core probe (the RTX PRO 6000 FP8 capture): raw bits in and out, driven by fp8.py.
//
// Tile kernels execute EXACTLY ONE mma.sync per warp per tile (checked by fp8.py's SASS gate):
//   f8_e4m3     mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32
//   f8_e5m2     mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32
//   f8_e4m3e5m2 mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e5m2.f32
//   legacy_e4m3 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   (tools/tc_probe's sm120.mma.m16n8k32.e4m3 form)
//   mxf8_e4m3   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
// Chain kernels keep the accumulator in registers across dependent steps (one MMA of each kind inside a `#pragma unroll 1`
// loop) and store every step's D, so the host replays the chain step by step:
//   e4m3_chain  f8_e4m3 repeated nsteps times (K = 32 * nsteps)
//   mixed_chain per step either f8_e4m3 or mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 on the same accumulator
// Scalar kernels: cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32 and add.rn{,.ftz}.f32, one instruction per element (pair).
//
// Fragment layouts (PTX ISA; verified by fp8.py's exact small-integer layout check before any semantic probe runs):
//   8-bit A (16x32, 4 regs): reg r byte j -> row lane/4 + 8*(r&1), k = 4*(lane%4) + j + 16*(r>>1)
//   8-bit B (32x8, 2 regs):  reg r byte j -> col lane/4,           k = 4*(lane%4) + j + 16*r
//   bf16 A/B (K = 16): as tools/tc_probe_fp4/mma_fp4.cu's bf16_tiles_kernel
//   SFA (1X, byte 0): lane supplies row 8*(lane&1) + (lane>>2); SFB: lane supplies col lane>>2
//   D/C (4 f32): d0/d1 -> row lane/4, cols 2*(lane%4)+{0,1}; d2/d3 -> row+8
//
// Host arrays (per tile, contiguous): A u8[16][32], Bt u8[8][32] (B n-major, k contiguous), SFA u8[16], SFB u8[8], C/D u32[16][8].
// Build: nvcc -O2 -gencode=arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared (no -ftz: the scalar kernels name it themselves).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

#define MMA_F8(NAME, MNEMONIC)                                                                                          \
    __device__ __forceinline__ void NAME(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {      \
        asm volatile(MNEMONIC " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                             \
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                                                    \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), \
                       "f"(c[3]));                                                                                       \
    }

MMA_F8(mma_f8_e4m3, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32")
MMA_F8(mma_f8_e5m2, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32")
MMA_F8(mma_f8_e4m3e5m2, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e5m2.f32")
MMA_F8(mma_legacy_e4m3, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
MMA_F8(mma_bf16, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")

__device__ __forceinline__ void mma_mxf8_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4], uint32_t sfa,
                                              uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa),
          "r"(sfb));
}

__device__ __forceinline__ void load_a8(uint32_t a[4], const uint8_t* A, int lane) {
    const int row0 = lane >> 2, kcol = 4 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 4; r++) a[r] = *reinterpret_cast<const uint32_t*>(A + (row0 + 8 * (r & 1)) * 32 + kcol + 16 * (r >> 1));
}

__device__ __forceinline__ void load_b8(uint32_t b[2], const uint8_t* Bt, int lane) {
    const int col = lane >> 2, kcol = 4 * (lane & 3);
#pragma unroll
    for (int r = 0; r < 2; r++) b[r] = *reinterpret_cast<const uint32_t*>(Bt + col * 32 + kcol + 16 * r);
}

__device__ __forceinline__ void load_a16(uint32_t af[4], const uint16_t* a, int lane) {
    constexpr int K = 16;
    const int g = lane >> 2, t = lane & 3;
    af[0] = *reinterpret_cast<const uint32_t*>(a + g * K + t * 2);
    af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + t * 2);
    af[2] = *reinterpret_cast<const uint32_t*>(a + g * K + t * 2 + K / 2);
    af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + t * 2 + K / 2);
}

__device__ __forceinline__ void load_b16(uint32_t bf[2], const uint16_t* b, int lane) {
    constexpr int K = 16;
    const int g = lane >> 2, t = lane & 3;
    bf[0] = *reinterpret_cast<const uint32_t*>(b + g * K + t * 2);
    bf[1] = *reinterpret_cast<const uint32_t*>(b + g * K + t * 2 + K / 2);
}

__device__ __forceinline__ void load_c(float cf[4], const uint32_t* c, int lane) {
    const int g = lane >> 2, t = lane & 3;
    cf[0] = __uint_as_float(c[g * 8 + 2 * t]);
    cf[1] = __uint_as_float(c[g * 8 + 2 * t + 1]);
    cf[2] = __uint_as_float(c[(g + 8) * 8 + 2 * t]);
    cf[3] = __uint_as_float(c[(g + 8) * 8 + 2 * t + 1]);
}

__device__ __forceinline__ void store_d(uint32_t* d, const float df[4], int lane) {
    const int g = lane >> 2, t = lane & 3;
    d[g * 8 + 2 * t] = __float_as_uint(df[0]);
    d[g * 8 + 2 * t + 1] = __float_as_uint(df[1]);
    d[(g + 8) * 8 + 2 * t] = __float_as_uint(df[2]);
    d[(g + 8) * 8 + 2 * t + 1] = __float_as_uint(df[3]);
}

#define F8_TILES_KERNEL(KNAME, MMA)                                                                                        \
    __global__ void KNAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint32_t* __restrict__ C,  \
                          uint32_t* __restrict__ D, int ntiles) {                                                          \
        const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                                     \
        if (warp >= ntiles) return;                                                                                        \
        const int lane = threadIdx.x & 31;                                                                                 \
        uint32_t a[4], b[2];                                                                                               \
        float cf[4], df[4];                                                                                                \
        load_a8(a, A + (size_t)warp * 16 * 32, lane);                                                                      \
        load_b8(b, Bt + (size_t)warp * 8 * 32, lane);                                                                      \
        load_c(cf, C + (size_t)warp * 128, lane);                                                                          \
        MMA(df, a, b, cf);                                                                                                 \
        store_d(D + (size_t)warp * 128, df, lane);                                                                         \
    }

F8_TILES_KERNEL(f8_e4m3_tiles_kernel, mma_f8_e4m3)
F8_TILES_KERNEL(f8_e5m2_tiles_kernel, mma_f8_e5m2)
F8_TILES_KERNEL(f8_e4m3e5m2_tiles_kernel, mma_f8_e4m3e5m2)
F8_TILES_KERNEL(legacy_e4m3_tiles_kernel, mma_legacy_e4m3)

__global__ void mxf8_e4m3_tiles_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint8_t* __restrict__ SFA,
                                       const uint8_t* __restrict__ SFB, const uint32_t* __restrict__ C, uint32_t* __restrict__ D,
                                       int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    uint32_t a[4], b[2];
    float cf[4], df[4];
    load_a8(a, A + (size_t)warp * 16 * 32, lane);
    load_b8(b, Bt + (size_t)warp * 8 * 32, lane);
    const uint32_t sfa = SFA[(size_t)warp * 16 + 8 * (lane & 1) + (lane >> 2)];
    const uint32_t sfb = SFB[(size_t)warp * 8 + (lane >> 2)];
    load_c(cf, C + (size_t)warp * 128, lane);
    mma_mxf8_e4m3(df, a, b, cf, sfa, sfb);
    store_d(D + (size_t)warp * 128, df, lane);
}

// A u8[nchains][nsteps][16][32], Bt u8[nchains][nsteps][8][32], C0 u32[nchains][16][8] -> D u32[nchains][nsteps][16][8].
__global__ void e4m3_chain_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt, const uint32_t* __restrict__ C0,
                                  uint32_t* __restrict__ D, int nchains, int nsteps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= nchains) return;
    const int lane = threadIdx.x & 31;
    float cf[4], df[4];
    load_c(cf, C0 + (size_t)warp * 128, lane);
#pragma unroll 1
    for (int s = 0; s < nsteps; s++) {
        const size_t step = (size_t)warp * nsteps + s;
        uint32_t a[4], b[2];
        load_a8(a, A + step * 16 * 32, lane);
        load_b8(b, Bt + step * 8 * 32, lane);
        mma_f8_e4m3(df, a, b, cf);
        store_d(D + step * 128, df, lane);
#pragma unroll
        for (int i = 0; i < 4; i++) cf[i] = df[i];
    }
}

// kinds u8[nchains][nsteps]: 0 = the e4m3 k32 step (A8/B8), 1 = the bf16 k16 step (A16/B16); both operand arrays are indexed by
// step, so A8 u8[nchains][nsteps][16][32], B8 u8[..][8][32], A16 u16[..][16][16], B16 u16[..][8][16].
__global__ void mixed_chain_kernel(const uint8_t* __restrict__ A8, const uint8_t* __restrict__ B8, const uint16_t* __restrict__ A16,
                                   const uint16_t* __restrict__ B16, const uint8_t* __restrict__ kinds, const uint32_t* __restrict__ C0,
                                   uint32_t* __restrict__ D, int nchains, int nsteps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= nchains) return;
    const int lane = threadIdx.x & 31;
    float cf[4], df[4];
    load_c(cf, C0 + (size_t)warp * 128, lane);
#pragma unroll 1
    for (int s = 0; s < nsteps; s++) {
        const size_t step = (size_t)warp * nsteps + s;
        uint32_t a[4], b[2];
        if (kinds[step] == 0) {
            load_a8(a, A8 + step * 16 * 32, lane);
            load_b8(b, B8 + step * 8 * 32, lane);
            mma_f8_e4m3(df, a, b, cf);
        } else {
            load_a16(a, A16 + step * 16 * 16, lane);
            load_b16(b, B16 + step * 8 * 16, lane);
            mma_bf16(df, a, b, cf);
        }
        store_d(D + step * 128, df, lane);
#pragma unroll
        for (int i = 0; i < 4; i++) cf[i] = df[i];
    }
}

// Y[i] = cvt(X[2i+1] -> high byte, X[2i] -> low byte), so Y viewed as bytes is element-ordered.
__global__ void cvt_e4m3x2_kernel(const uint32_t* __restrict__ X, uint16_t* __restrict__ Y, int npairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs) return;
    uint16_t h;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(h) : "f"(__uint_as_float(X[2 * i + 1])), "f"(__uint_as_float(X[2 * i])));
    Y[i] = h;
}

__global__ void cvt_e5m2x2_kernel(const uint32_t* __restrict__ X, uint16_t* __restrict__ Y, int npairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs) return;
    uint16_t h;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;\n" : "=h"(h) : "f"(__uint_as_float(X[2 * i + 1])), "f"(__uint_as_float(X[2 * i])));
    Y[i] = h;
}

__global__ void fadd_rn_kernel(const uint32_t* __restrict__ X, const uint32_t* __restrict__ Y, uint32_t* __restrict__ Z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z;
    asm volatile("add.rn.f32 %0, %1, %2;\n" : "=f"(z) : "f"(__uint_as_float(X[i])), "f"(__uint_as_float(Y[i])));
    Z[i] = __float_as_uint(z);
}

__global__ void fadd_rn_ftz_kernel(const uint32_t* __restrict__ X, const uint32_t* __restrict__ Y, uint32_t* __restrict__ Z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z;
    asm volatile("add.rn.ftz.f32 %0, %1, %2;\n" : "=f"(z) : "f"(__uint_as_float(X[i])), "f"(__uint_as_float(Y[i])));
    Z[i] = __float_as_uint(z);
}

struct Dev {
    void* p = nullptr;
    ~Dev() { cudaFree(p); }
};

static int upload(Dev& d, const void* host, size_t bytes) {
    cudaError_t err = cudaMalloc(&d.p, bytes ? bytes : 1);
    if (err == cudaSuccess && host && bytes) err = cudaMemcpy(d.p, host, bytes, cudaMemcpyHostToDevice);
    return (int)err;
}

static int finish(void* host, const Dev& d, size_t bytes) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host, d.p, bytes, cudaMemcpyDeviceToHost);
    return (int)err;
}

}  // namespace

extern "C" {

// which: 0 f8_e4m3, 1 f8_e5m2, 2 f8_e4m3e5m2, 3 legacy_e4m3, 4 mxf8_e4m3 (SFA u8[n][16], SFB u8[n][8]; ignored otherwise).
int fp8_tiles(int which, const uint8_t* A, const uint8_t* Bt, const uint8_t* SFA, const uint8_t* SFB, const uint32_t* C, uint32_t* D, int n) {
    Dev dA, dB, dSA, dSB, dC, dD;
    int rc;
    if ((rc = upload(dA, A, (size_t)n * 512)) || (rc = upload(dB, Bt, (size_t)n * 256)) || (rc = upload(dC, C, (size_t)n * 512)) ||
        (rc = upload(dD, nullptr, (size_t)n * 512)))
        return rc;
    if (which == 4 && ((rc = upload(dSA, SFA, (size_t)n * 16)) || (rc = upload(dSB, SFB, (size_t)n * 8)))) return rc;
    const int wpb = 4, blocks = (n + wpb - 1) / wpb;
    auto a = (const uint8_t*)dA.p;
    auto b = (const uint8_t*)dB.p;
    auto c = (const uint32_t*)dC.p;
    auto d = (uint32_t*)dD.p;
    switch (which) {
        case 0: f8_e4m3_tiles_kernel<<<blocks, 32 * wpb>>>(a, b, c, d, n); break;
        case 1: f8_e5m2_tiles_kernel<<<blocks, 32 * wpb>>>(a, b, c, d, n); break;
        case 2: f8_e4m3e5m2_tiles_kernel<<<blocks, 32 * wpb>>>(a, b, c, d, n); break;
        case 3: legacy_e4m3_tiles_kernel<<<blocks, 32 * wpb>>>(a, b, c, d, n); break;
        case 4: mxf8_e4m3_tiles_kernel<<<blocks, 32 * wpb>>>(a, b, (const uint8_t*)dSA.p, (const uint8_t*)dSB.p, c, d, n); break;
        default: return (int)cudaErrorInvalidValue;
    }
    return finish(D, dD, (size_t)n * 512);
}

int e4m3_chain(const uint8_t* A, const uint8_t* Bt, const uint32_t* C0, uint32_t* D, int nchains, int nsteps) {
    const size_t steps = (size_t)nchains * nsteps;
    Dev dA, dB, dC, dD;
    int rc;
    if ((rc = upload(dA, A, steps * 512)) || (rc = upload(dB, Bt, steps * 256)) || (rc = upload(dC, C0, (size_t)nchains * 512)) ||
        (rc = upload(dD, nullptr, steps * 512)))
        return rc;
    const int wpb = 4, blocks = (nchains + wpb - 1) / wpb;
    e4m3_chain_kernel<<<blocks, 32 * wpb>>>((const uint8_t*)dA.p, (const uint8_t*)dB.p, (const uint32_t*)dC.p, (uint32_t*)dD.p, nchains, nsteps);
    return finish(D, dD, steps * 512);
}

int mixed_chain(const uint8_t* A8, const uint8_t* B8, const uint16_t* A16, const uint16_t* B16, const uint8_t* kinds, const uint32_t* C0,
                uint32_t* D, int nchains, int nsteps) {
    const size_t steps = (size_t)nchains * nsteps;
    Dev dA8, dB8, dA16, dB16, dK, dC, dD;
    int rc;
    if ((rc = upload(dA8, A8, steps * 512)) || (rc = upload(dB8, B8, steps * 256)) || (rc = upload(dA16, A16, steps * 512)) ||
        (rc = upload(dB16, B16, steps * 256)) || (rc = upload(dK, kinds, steps)) || (rc = upload(dC, C0, (size_t)nchains * 512)) ||
        (rc = upload(dD, nullptr, steps * 512)))
        return rc;
    const int wpb = 4, blocks = (nchains + wpb - 1) / wpb;
    mixed_chain_kernel<<<blocks, 32 * wpb>>>((const uint8_t*)dA8.p, (const uint8_t*)dB8.p, (const uint16_t*)dA16.p, (const uint16_t*)dB16.p,
                                             (const uint8_t*)dK.p, (const uint32_t*)dC.p, (uint32_t*)dD.p, nchains, nsteps);
    return finish(D, dD, steps * 512);
}

// fmt: 0 e4m3x2, 1 e5m2x2.  X u32[2 * npairs] -> Y u16[npairs].
int cvt_x2(int fmt, const uint32_t* X, uint16_t* Y, int npairs) {
    Dev dX, dY;
    int rc;
    if ((rc = upload(dX, X, (size_t)npairs * 8)) || (rc = upload(dY, nullptr, (size_t)npairs * 2))) return rc;
    const int tpb = 256, blocks = (npairs + tpb - 1) / tpb;
    if (fmt == 0) cvt_e4m3x2_kernel<<<blocks, tpb>>>((const uint32_t*)dX.p, (uint16_t*)dY.p, npairs);
    else cvt_e5m2x2_kernel<<<blocks, tpb>>>((const uint32_t*)dX.p, (uint16_t*)dY.p, npairs);
    return finish(Y, dY, (size_t)npairs * 2);
}

int fadd_f32(int ftz, const uint32_t* X, const uint32_t* Y, uint32_t* Z, int n) {
    Dev dX, dY, dZ;
    int rc;
    if ((rc = upload(dX, X, (size_t)n * 4)) || (rc = upload(dY, Y, (size_t)n * 4)) || (rc = upload(dZ, nullptr, (size_t)n * 4))) return rc;
    const int tpb = 256, blocks = (n + tpb - 1) / tpb;
    if (ftz) fadd_rn_ftz_kernel<<<blocks, tpb>>>((const uint32_t*)dX.p, (const uint32_t*)dY.p, (uint32_t*)dZ.p, n);
    else fadd_rn_kernel<<<blocks, tpb>>>((const uint32_t*)dX.p, (const uint32_t*)dY.p, (uint32_t*)dZ.p, n);
    return finish(Z, dZ, (size_t)n * 4);
}

// The visible device's identity: compute capability, SM count, name and the 16-byte UUID nvidia-smi prints as GPU-xxxxxxxx-...
int device_info(int* major, int* minor, int* sms, char* name, int name_len, unsigned char* uuid) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    *sms = prop.multiProcessorCount;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    for (int j = 0; j < 16; j++) uuid[j] = (unsigned char)prop.uuid.bytes[j];
    return 0;
}

}  // extern "C"
