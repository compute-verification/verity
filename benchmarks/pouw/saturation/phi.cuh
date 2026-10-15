// The receipt step phi_probe.cu and receipt_gemm.cu fold over accumulator words: the ideal-phi model's phi_{sigma,beta,t}(y, p)
// as the page's "keyed 32-bit ARX step" (s the readout, k_t an immediate per word)
//   alu   phi1(y, p) = rotl(t, 13) + t + k_t, t = y ^ p ^ s   all on the integer ALU: ptxas emits LOP3, LEA, IADD3
//   imad  phi1(y, p) = t * (k_t | 1) + rotl(t, 13)             LOP3, SHF.L.W on the ALU, IMAD on the FMA pipe beside it
//   ROUNDS > 1 applies ROUNDS - 1 more phi1 against a constant (round r keyed k_t ^ r 0x5851f42d)
#pragma once
#include <cstdint>

__host__ __device__ __forceinline__ uint32_t rotl13(uint32_t t) {
#ifdef __CUDA_ARCH__
    return __funnelshift_l(t, t, 13);
#else
    return (t << 13) | (t >> 19);
#endif
}
__host__ __device__ __forceinline__ uint32_t key(int j, int w) { return 0x9e3779b9u * (uint32_t)(j * 4 + w + 1); }

template <int VAR>
__host__ __device__ __forceinline__ uint32_t phi1(uint32_t y, uint32_t p, uint32_t s, uint32_t k) {
    const uint32_t t = y ^ p ^ s;
    return VAR == 0 ? rotl13(t) + t + k : t * (k | 1u) + rotl13(t);
}

template <int ROUNDS, int VAR>
__host__ __device__ __forceinline__ uint32_t phi(uint32_t y, uint32_t p, uint32_t s, uint32_t k) {
    y = phi1<VAR>(y, p, s, k);
#pragma unroll
    for (int r = 1; r < ROUNDS; r++) y = phi1<VAR>(y, 0x7f4a7c15u, s, k ^ (0x5851f42du * (uint32_t)r));
    return y;
}
