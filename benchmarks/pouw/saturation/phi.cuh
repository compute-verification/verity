// The receipt step phi_probe.cu and receipt_gemm.cu fold over accumulator words: the ideal-phi model's phi_{sigma,beta,t}(y, p)
// as the page's "keyed 32-bit ARX step" (s the readout, k_t an immediate per word)
//   alu   phi1(y, p) = rotl(t, 13) + t + k_t, t = y ^ p ^ s   all on the integer ALU: ptxas emits LOP3, LEA, IADD3
//   imad  phi1(y, p) = t * (k_t | 1) + rotl(t, 13)             LOP3, SHF.L.W on the ALU, IMAD on the FMA pipe beside it
//   ROUNDS > 1 applies ROUNDS - 1 more phi1 against a constant (round r keyed k_t ^ r 0x5851f42d)
//   duo2  A5a's two-lane receipt (below), cutlass_receipt's VAR 2
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

// duo2, the receipt A5a's round 3 left standing (benchmarks/receipt_phi/phi.cu's absorb<DUO2> and finish<DUO2> on
// cursor/receipt-phi-ea24): two duo lanes (y, z) and (y2, z2), each word keyed at its position t by KK, KA (lane 1) and
// KC, KB (lane 2), KA and KB odd, with rotations of periods 31 and 29. The keys are the window's live coins in a protocol.
struct Duo2 {
    uint32_t y, z, y2, z2;
};
constexpr int D2_KEYS = 64;
__host__ __device__ constexpr int d2_rot1(int t) {
    constexpr int R[31] = {15, 18, 20, 5, 29, 25, 4, 9, 1, 6, 12, 28, 8, 7, 31, 19,
                           10, 17, 3, 11, 27, 14, 2, 26, 16, 30, 13, 23, 21, 24, 22};
    return R[t % 31];
}
__host__ __device__ constexpr int d2_rot2(int t) {
    constexpr int R[29] = {6, 23, 7, 19, 4, 14, 10, 12, 9, 28, 11, 20, 26, 2, 22,
                           17, 5, 16, 15, 3, 25, 8, 13, 29, 30, 27, 21, 24, 18};
    return R[t % 29];
}
__host__ __device__ __forceinline__ uint32_t rotlv(uint32_t x, int r) {
#ifdef __CUDA_ARCH__
    return __funnelshift_l(x, x, r);
#else
    return (x << r) | (x >> ((32 - r) & 31));
#endif
}
// key j (0 KK, 1 KA, 2 KC, 3 KB) of position t
__host__ __device__ constexpr uint32_t d2_key(int j, int t) {
    uint32_t x = 0x9e3779b9u * (uint32_t)(j * D2_KEYS + t + 1) ^ 0x632be5abu;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return j == 1 || j == 3 ? x | 1u : x;
}
template <int T>
__host__ __device__ __forceinline__ void duo2(Duo2& s, uint32_t p, uint32_t kk, uint32_t ka, uint32_t kc, uint32_t kb) {
    constexpr int r1 = d2_rot1(T), r2 = d2_rot2(T);
    const uint32_t u = s.y ^ p ^ kk;
    s.y = u * ka + s.z;
    s.z = rotlv(u, r1) + s.y;
    const uint32_t v = s.y2 ^ p ^ kc;
    s.y2 = v * kb + s.z2;
    s.z2 = rotlv(v, r2) + s.y2;
}
__host__ __device__ __forceinline__ uint32_t duo2_fold(const Duo2& s) {
    return s.y ^ rotlv(s.z, 16) ^ rotlv(s.y2 ^ rotlv(s.z2, 16), 8);
}
