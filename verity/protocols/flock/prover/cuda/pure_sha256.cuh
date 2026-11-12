// SHA-256 compression (FIPS 180-4) on the device, for the sha256/row/v1 chains of pure_sha_chain.
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define PSH_HD __host__ __device__ __forceinline__
#define PSH_CONST __device__ __constant__
#else
#define PSH_HD inline
#define PSH_CONST static const
#endif

PSH_CONST uint32_t PSH_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

PSH_HD uint32_t psh_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
PSH_HD uint32_t psh_S0(uint32_t x) { return psh_rotr(x, 2) ^ psh_rotr(x, 13) ^ psh_rotr(x, 22); }
PSH_HD uint32_t psh_S1(uint32_t x) { return psh_rotr(x, 6) ^ psh_rotr(x, 11) ^ psh_rotr(x, 25); }
PSH_HD uint32_t psh_s0(uint32_t x) { return psh_rotr(x, 7) ^ psh_rotr(x, 18) ^ (x >> 3); }
PSH_HD uint32_t psh_s1(uint32_t x) { return psh_rotr(x, 17) ^ psh_rotr(x, 19) ^ (x >> 10); }

PSH_HD void psh_compress(uint32_t* h, const uint32_t* m) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = m[i];
    for (int t = 16; t < 64; t++) w[t] = psh_s1(w[t - 2]) + w[t - 7] + psh_s0(w[t - 15]) + w[t - 16];
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int r = 0; r < 64; r++) {
        uint32_t t1 = hh + psh_S1(e) + ((e & f) ^ (~e & g)) + PSH_K[r] + w[r];
        uint32_t t2 = psh_S0(a) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
