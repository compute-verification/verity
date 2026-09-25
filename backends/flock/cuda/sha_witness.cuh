// The honest (z, a, b) of one SHA-256 compression in Flock's circuit (flock-prover r1cs_hashes::sha2,
// build_block_ab_packed_into at b684b12), written into a zeroed 2^15-bit region per buffer. Host- and device-callable
// so a host harness can check it against the Rust witness.
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
PSH_HD int psh_ctz(uint32_t x) { int n = 0; while (!(x & 1)) { x >>= 1; n++; } return n; }

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

// OR the low 32 bits of v into buf at bit offset off (buf is the compression's 1,024-word region).
PSH_HD void psh_or(uint32_t* buf, int off, uint32_t v) {
    int i = off >> 5, s = off & 31;
    buf[i] |= v << s;
    if (s) buf[i + 1] |= v >> (32 - s);
}
PSH_HD void psh_or3(uint32_t* z, uint32_t* a, uint32_t* b, int off, uint32_t vz, uint32_t va, uint32_t vb) {
    psh_or(z, off, vz); psh_or(a, off, va); psh_or(b, off, vb);
}

#define PSH_M31 0x7FFFFFFFu
#define PSH_M30 0x3FFFFFFFu
enum { PSH_H_BASE = 0, PSH_H_OUT = 256, PSH_M_BASE = 512, PSH_CH = 1024, PSH_MAJ = 3072, PSH_ROUND = 5120, PSH_SCHED = 18757,
       PSH_ENEW = 23173, PSH_ANEW = 24197, PSH_OUTC = 25221, PSH_ZPOS = 25469 };

// (z, a, b) of add x + y: 31 carry rows at `carry`; returns the sum.
PSH_HD uint32_t psh_add(uint32_t x, uint32_t y, uint32_t* z, uint32_t* a, uint32_t* b, int carry) {
    uint32_t s = x + y, cin = s ^ x ^ y, l = (x ^ cin) & PSH_M31, r = (y ^ cin) & PSH_M31;
    psh_or3(z, a, b, carry, l & r, l, r);
    return s;
}

// fused x + y + zz + w: maj1 at off, maj2 at off + 31, ripple at off + 62; returns the sum.
PSH_HD uint32_t psh_add4(uint32_t x, uint32_t y, uint32_t zz, uint32_t w, uint32_t* z, uint32_t* a, uint32_t* b, int off) {
    uint32_t xz = x ^ zz, yz = y ^ zz, w1 = xz & yz, p1 = x ^ y ^ zz, b1 = (w1 ^ zz) << 1;
    uint32_t pw = p1 ^ w, bw = b1 ^ w, w2 = pw & bw, p2 = p1 ^ b1 ^ w, b2 = (w2 ^ w) << 1;
    uint32_t s = p2 + b2, cin = s ^ p2 ^ b2, rl = p2 ^ cin, rr = b2 ^ cin;
    psh_or3(z, a, b, off, w1 & PSH_M31, xz & PSH_M31, yz & PSH_M31);
    psh_or3(z, a, b, off + 31, w2 & PSH_M31, pw & PSH_M31, bw & PSH_M31);
    psh_or3(z, a, b, off + 62, ((rl & rr) >> 1) & PSH_M30, (rl >> 1) & PSH_M30, (rr >> 1) & PSH_M30);
    return s;
}

// fused x + y + mm: maj at off, ripple at off + 31; returns the sum.
PSH_HD uint32_t psh_add3(uint32_t x, uint32_t y, uint32_t mm, uint32_t* z, uint32_t* a, uint32_t* b, int off) {
    uint32_t xm = x ^ mm, ym = y ^ mm, w = xm & ym, p = x ^ y ^ mm, bw = (w ^ mm) << 1;
    uint32_t s = p + bw, cin = s ^ p ^ bw, rl = p ^ cin, rr = bw ^ cin;
    psh_or3(z, a, b, off, w & PSH_M31, xm & PSH_M31, ym & PSH_M31);
    psh_or3(z, a, b, off + 31, ((rl & rr) >> 1) & PSH_M30, (rl >> 1) & PSH_M30, (rr >> 1) & PSH_M30);
    return s;
}

PSH_HD void psh_witness(const uint32_t* h_in, const uint32_t* m, uint32_t* z, uint32_t* a, uint32_t* b) {
    psh_or3(z, a, b, PSH_ZPOS, 1, 1, 1);
    for (int w = 0; w < 8; w++) psh_or3(z, a, b, PSH_H_BASE + 32 * w, h_in[w], h_in[w], 0xFFFFFFFFu);
    for (int i = 0; i < 16; i++) psh_or3(z, a, b, PSH_M_BASE + 32 * i, m[i], m[i], 0xFFFFFFFFu);
    uint32_t ws[64];
    for (int i = 0; i < 16; i++) ws[i] = m[i];
    for (int t = 16; t < 64; t++)
        ws[t] = psh_add4(psh_s1(ws[t - 2]), psh_s0(ws[t - 15]), ws[t - 7], ws[t - 16], z, a, b, PSH_SCHED + (t - 16) * 92);
    uint32_t A = h_in[0], B = h_in[1], C = h_in[2], D = h_in[3], E = h_in[4], F = h_in[5], G = h_in[6], H = h_in[7];
    int base = PSH_ROUND;
    for (int r = 0; r < 64; r++) {
        uint32_t fg = F ^ G, ch = E & fg;
        psh_or3(z, a, b, PSH_CH + 32 * r, ch, E, fg);
        uint32_t ba = B ^ A, ca = C ^ A, mj = ba & ca;
        psh_or3(z, a, b, PSH_MAJ + 32 * r, mj, ba, ca);
        const uint32_t k = PSH_K[r];
        const int t = psh_ctz(k) + 1;
        {
            uint32_t s = k + H, cin = s ^ k ^ H, l = (k ^ cin) & PSH_M31, rr = (H ^ cin) & PSH_M31;
            uint32_t mask = (1u << (31 - t)) - 1;
            psh_or3(z, a, b, base + 184, ((l & rr) >> t) & mask, (l >> t) & mask, (rr >> t) & mask);
            uint32_t t1 = psh_add4(s, psh_S1(E), ch ^ G, ws[r], z, a, b, base);
            uint32_t an = psh_add3(t1, psh_S0(A), mj ^ A, z, a, b, base + 92);
            uint32_t en = psh_add(D, t1, z, a, b, base + 153);
            if (r & 1) {
                psh_or3(z, a, b, PSH_ANEW + 32 * (r / 2), an, an, 0xFFFFFFFFu);
                psh_or3(z, a, b, PSH_ENEW + 32 * (r / 2), en, en, 0xFFFFFFFFu);
            }
            H = G; G = F; F = E; E = en; D = C; C = B; B = A; A = an;
        }
        base += 184 + 31 - t;
    }
    const uint32_t fin[8] = {A, B, C, D, E, F, G, H};
    for (int w = 0; w < 8; w++) {
        uint32_t s = psh_add(fin[w], h_in[w], z, a, b, PSH_OUTC + 31 * w);
        psh_or3(z, a, b, PSH_H_OUT + 32 * w, s, s, 0xFFFFFFFFu);
    }
}
