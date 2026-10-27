// 2:4 sparse E4M3 on sm_120 (tc-model/sm120-e4m3-sp-k64): one
//   mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32
// per tile, against the dense
//   mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32
// in an in-register chain of L dependent atoms.  Raw registers in and out: the host (sp_e4m3.py) packs every fragment, the
// metadata word and the accumulator, so a layout question is settled on the host against exact integer tiles, never by a rebuild.
//   sp_e4m3_kernel:     per tile and lane: a (4 x b32: 16 stored e4m3 bytes), b (4 x b32: 16 of B's 64 x 8), e (b32 metadata),
//                       c (4 FP32 words) -> d; sparsity selector 0 (every thread of the quad holds metadata).
//   dense_chain_kernel: per tile, step and lane: a (4 x b32), b (2 x b32); c -> d after L dependent atoms, D never leaving the
//                       registers between them.
// Every D buffer is filled with 0xA5 before the launch, so a word no kernel wrote reads 0xA5A5A5A5; set_kernel_launch(0) skips the
// launches (the no-write control).
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void sp_e4m3_kernel(const uint4* __restrict__ A, const uint4* __restrict__ B, const uint32_t* __restrict__ E,
                               const uint4* __restrict__ C, uint4* __restrict__ D, int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const size_t i = (size_t)warp * 32 + (threadIdx.x & 31);
    const uint4 a = A[i], b = B[i], c = C[i];
    const uint32_t e = E[i];
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w), "f"(__uint_as_float(c.x)),
          "f"(__uint_as_float(c.y)), "f"(__uint_as_float(c.z)), "f"(__uint_as_float(c.w)), "r"(e));
    D[i] = make_uint4(__float_as_uint(d0), __float_as_uint(d1), __float_as_uint(d2), __float_as_uint(d3));
}

__global__ void dense_chain_kernel(const uint4* __restrict__ A, const uint2* __restrict__ B, const uint4* __restrict__ C,
                                   uint4* __restrict__ D, int ntiles, int L) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    const uint4 c = C[(size_t)warp * 32 + lane];
    float c0 = __uint_as_float(c.x), c1 = __uint_as_float(c.y), c2 = __uint_as_float(c.z), c3 = __uint_as_float(c.w);
#pragma unroll 1
    for (int l = 0; l < L; l++) {
        const size_t j = ((size_t)warp * L + l) * 32 + lane;
        const uint4 a = A[j];
        const uint2 b = B[j];
        float d0, d1, d2, d3;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
        c0 = d0;
        c1 = d1;
        c2 = d2;
        c3 = d3;
    }
    D[(size_t)warp * 32 + lane] = make_uint4(__float_as_uint(c0), __float_as_uint(c1), __float_as_uint(c2), __float_as_uint(c3));
}

constexpr int D_POISON = 0xA5;
int g_launch = 1;

struct Dev {
    void* p[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    ~Dev() {
        for (void* q : p) cudaFree(q);
    }
};

}  // namespace

extern "C" {

// A, B u32[n][32][4], E u32[n][32], C/D u32[n][32][4] (lane-major registers).  Returns the cudaError_t as an int.
int sp_e4m3_tiles(const uint32_t* A, const uint32_t* B, const uint32_t* E, const uint32_t* C, uint32_t* D, int n) {
    const size_t r4 = (size_t)n * 32 * 16, r1 = (size_t)n * 32 * 4;
    Dev d;
    cudaError_t err = cudaSuccess;
    const size_t sizes[5] = {r4, r4, r1, r4, r4};
    for (int k = 0; k < 5; k++)
        if ((err = cudaMalloc(&d.p[k], sizes[k])) != cudaSuccess) return (int)err;
    cudaMemcpy(d.p[0], A, r4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.p[1], B, r4, cudaMemcpyHostToDevice);
    cudaMemcpy(d.p[2], E, r1, cudaMemcpyHostToDevice);
    cudaMemcpy(d.p[3], C, r4, cudaMemcpyHostToDevice);
    if ((err = cudaMemset(d.p[4], D_POISON, r4)) != cudaSuccess) return (int)err;
    const int wpb = 4, blocks = (n + wpb - 1) / wpb;
    if (g_launch)
        sp_e4m3_kernel<<<blocks, 32 * wpb>>>((const uint4*)d.p[0], (const uint4*)d.p[1], (const uint32_t*)d.p[2],
                                             (const uint4*)d.p[3], (uint4*)d.p[4], n);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(D, d.p[4], r4, cudaMemcpyDeviceToHost);
    return (int)err;
}

// A u32[n][L][32][4], B u32[n][L][32][2], C/D u32[n][32][4].
int dense_e4m3_chain(const uint32_t* A, const uint32_t* B, const uint32_t* C, uint32_t* D, int n, int L) {
    if (L < 1) return (int)cudaErrorInvalidValue;
    const size_t ra = (size_t)n * L * 32 * 16, rb = (size_t)n * L * 32 * 8, rc = (size_t)n * 32 * 16;
    Dev d;
    cudaError_t err = cudaSuccess;
    const size_t sizes[4] = {ra, rb, rc, rc};
    for (int k = 0; k < 4; k++)
        if ((err = cudaMalloc(&d.p[k], sizes[k])) != cudaSuccess) return (int)err;
    cudaMemcpy(d.p[0], A, ra, cudaMemcpyHostToDevice);
    cudaMemcpy(d.p[1], B, rb, cudaMemcpyHostToDevice);
    cudaMemcpy(d.p[2], C, rc, cudaMemcpyHostToDevice);
    if ((err = cudaMemset(d.p[3], D_POISON, rc)) != cudaSuccess) return (int)err;
    const int wpb = 4, blocks = (n + wpb - 1) / wpb;
    if (g_launch)
        dense_chain_kernel<<<blocks, 32 * wpb>>>((const uint4*)d.p[0], (const uint2*)d.p[1], (const uint4*)d.p[2], (uint4*)d.p[3], n,
                                                 L);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(D, d.p[3], rc, cudaMemcpyDeviceToHost);
    return (int)err;
}

void set_kernel_launch(int on) { g_launch = on != 0; }

int d_poison_byte(void) { return D_POISON; }

int has_mxf4(void) { return 0; }

int device_uuid(char* out, int out_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    if (out_len < 41) return (int)cudaErrorInvalidValue;
    static const char* hex = "0123456789abcdef";
    const unsigned char* u = (const unsigned char*)prop.uuid.bytes;
    int p = 0;
    out[p++] = 'G';
    out[p++] = 'P';
    out[p++] = 'U';
    out[p++] = '-';
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[p++] = '-';
        out[p++] = hex[u[i] >> 4];
        out[p++] = hex[u[i] & 15];
    }
    out[p] = 0;
    return 0;
}

int device_identity(int* major, int* minor, char* name, int name_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    *major = prop.major;
    *minor = prop.minor;
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) name[i] = prop.name[i];
    name[i] = 0;
    return 0;
}

}  // extern "C"
