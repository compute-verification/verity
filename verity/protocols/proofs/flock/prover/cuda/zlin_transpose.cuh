// verity flock-gpu-link: the lincheck's z transpose (chunk_zlin_transpose) and its 8x8 bit transpose.
#pragma once
#include <cstdint>

// an 8x8 bit matrix, row r in byte r (bit c its column c), transposed
__device__ __forceinline__ u64 zlin_t8(u64 x) {
    u64 t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    return x ^ t ^ (t << 28);
}

// out[(s << k_log) + c] holds bit c of z's rows 8s..8s+7 (bit r: row 8s + r). A thread's 128 bytes are sixteen 8x8 bit
// transposes of byte j of its eight rows (gathered by 4x4 byte transposes); a block stages its threads' 32 KB in shared
// memory (slot q of thread t's row at q ^ (t & 7), conflict-free both ways) so a warp's stores are 512 contiguous bytes.
// 256 threads a block; out is 16-byte aligned, 128 bytes a thread from out + 128·t.
__global__ void __launch_bounds__(256) chunk_zlin_transpose(const F128* __restrict__ z, int k_log, long long n_stripes,
                                                             uint8_t* __restrict__ out) {
    __shared__ uint4 st[256 * 8];
    const long long wpr = (1LL << k_log) / 128;   // F128 words per block row
    const long long n = n_stripes * wpr;
    const int tid = threadIdx.x;
    for (long long base = (long long)blockIdx.x * 256; base < n; base += (long long)gridDim.x * 256) {
        if (base + tid < n) {
            const long long s = (base + tid) / wpr, w = (base + tid) % wpr;
            uint4 v[8];
#pragma unroll
            for (int r = 0; r < 8; r++) v[r] = __ldg(reinterpret_cast<const uint4*>(z + (8 * s + r) * wpr + w));
#pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                uint32_t x[8];
#pragma unroll
                for (int r = 0; r < 8; r++) x[r] = wi == 0 ? v[r].x : wi == 1 ? v[r].y : wi == 2 ? v[r].z : v[r].w;
                uint32_t lo[4], hi[4];   // byte r of lo[e] (hi[e]): byte 4·wi + e of row r (r + 4)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t* y = x + 4 * h;
                    const uint32_t t0 = __byte_perm(y[0], y[1], 0x5140), t1 = __byte_perm(y[0], y[1], 0x7362);
                    const uint32_t t2 = __byte_perm(y[2], y[3], 0x5140), t3 = __byte_perm(y[2], y[3], 0x7362);
                    uint32_t* o = h ? hi : lo;
                    o[0] = __byte_perm(t0, t2, 0x5410);
                    o[1] = __byte_perm(t0, t2, 0x7632);
                    o[2] = __byte_perm(t1, t3, 0x5410);
                    o[3] = __byte_perm(t1, t3, 0x7632);
                }
                u64 p[4];
#pragma unroll
                for (int e = 0; e < 4; e++) p[e] = zlin_t8((u64)hi[e] << 32 | lo[e]);
                st[tid * 8 + ((2 * wi) ^ (tid & 7))] = make_uint4((uint32_t)p[0], (uint32_t)(p[0] >> 32), (uint32_t)p[1], (uint32_t)(p[1] >> 32));
                st[tid * 8 + ((2 * wi + 1) ^ (tid & 7))] =
                    make_uint4((uint32_t)p[2], (uint32_t)(p[2] >> 32), (uint32_t)p[3], (uint32_t)(p[3] >> 32));
            }
        }
        __syncthreads();
        uint4* ob = reinterpret_cast<uint4*>(out) + base * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int q = tid + 256 * i, row = q >> 3;
            if (base + row < n) ob[q] = st[row * 8 + ((q & 7) ^ (row & 7))];
        }
        __syncthreads();
    }
}
