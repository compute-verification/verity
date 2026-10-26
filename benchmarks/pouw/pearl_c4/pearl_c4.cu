// Pearl-C4 on sm_120a (RTX PRO 6000, RTX 5090): the kernels of `pearl-c-nvfp4-v0` and `pearl-c-mxfp4-v0`
// (verity/ml/protocols/pouw/schemes/pearl_c4.py), for dev.py.  Every kernel takes one Args by value: 16 pointers p[] and
// 16 ints i[].  FMT 0 is NVFP4 (E2M1, UE4M3 per 16 along k), FMT 1 MXFP4 (E2M1, UE8M0 per 32).  Hashing is GPU 2's
// pouw_hash.cuh (TurboSHAKE128, SHA-256), included from the pinned commit build.sh names, and the seeds' and noise lines'
// SHA-512 is pearl_c/sha512.cuh's (Pearl-C's -h3 compression).
//
// Buffers, all row-major (scales k-tile major) and little-endian:
//   x        rows x k BF16 (A or B)                          codes    rows x k/2 bytes, E2M1, even position in the low nibble
//   scales   k/128 x rows x 128/BLK bytes: k-tile kt's scales for row r at (kt·rows + r)·128/BLK, so a CTA's rows of one
//            k-tile are contiguous                           lines    count x 16 bytes, one noise line (32 E2M1 codes) each
//   ft       32 x k/2 bytes: F's lines transposed (ft[p] nibble t = line t's code p)
//   P        rows x 64 BF16 (P_A = [Ẽ_A | bf16(A'F_B)], P_B = [bf16(Ẽ_B G - B~F_A) | -Ẽ_B])
//   alpha, inv  rows FP32 (α and RN(1/α));  ae  rows bytes (a_E, UE4M3)
//   y        m x n BF16, RN(RN(U · inv_A) · inv_B)       digests  m x n/32 x 32 bytes: TurboSHAKE128(C~ chunk | U chunk, 0x21)
//   leaves   m/64 x n/64 x 32 bytes (0x22)                 dom, root, salt, seed   32 bytes each (commit_rows)
// Pearl's lottery tickets are not computed: no PoUW check reads them (the coordinator's 06:35Z ruling for pearl-c-sm120).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "pouw_hash.cuh"

struct Args { uint64_t p[16]; int32_t i[16]; };

#define DEV __device__ __forceinline__

#include "../pearl_c/sha512.cuh"

constexpr unsigned FULL = 0xFFFFFFFFu;
enum { NVFP4 = 0, MXFP4 = 1 };
template <int FMT> struct Fmt;
template <> struct Fmt<NVFP4> { static constexpr int BLK = 16; static constexpr uint32_t ONE = 0x38383838u; };
template <> struct Fmt<MXFP4> { static constexpr int BLK = 32; static constexpr uint32_t ONE = 0x7F7Fu; };

constexpr float TOP = 2688.0f, DELTA = 0.25f, FOUR = 4.0f;
constexpr uint32_t FLOOR_BITS = 0x2F800000u, K_BETA_BITS = 0x3EA920A7u, SIXTH_BITS = 0x3E2AAAABu;
constexpr uint32_t NOISE_F_SCALE = 0x18181818u;

// ---------------------------------------------------------------- formats (IEEE, nearest even, no FTZ; built -fmad=false)
DEV float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
DEV uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
DEV uint32_t ue4m3(float v) { return (uint32_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3) & 0x7Fu; }
DEV float ue4m3f(uint32_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw((uint8_t)c, __NV_E4M3))); }
// 2^ceil(log2 v) as a UE8M0 byte for a non-negative finite FP32 word (0 for +0), clamped to [0, 254]
DEV uint32_t ue8m0_up(uint32_t w) {
  if (w == 0) return 0;
  const int field = (w >> 23) & 0xFF, frac = w & 0x7FFFFF;
  const int e = field ? field - 127 + (frac != 0) : (31 - __clz(frac)) - 149 + ((frac & (frac - 1)) != 0);
  return (uint32_t)min(max(e + 127, 0), 254);
}
DEV float pow2f(int e) { return __uint_as_float(e >= -126 ? (uint32_t)(e + 127) << 23 : 1u << (e + 149)); }
// FP32 division and square root, correctly rounded, through FP64: 53 >= 2·24 + 2 bits make the double rounding innocuous,
// and FP32's own slow paths (FADD.FTZ, FMUL.FTZ in their SASS) never enter a shipped kernel
DEV float div_rn(float a, float b) { return __double2float_rn(__ddiv_rn((double)a, (double)b)); }
DEV float sqrt_rn(float a) { return __double2float_rn(__dsqrt_rn((double)a)); }
// cvt.rn.satfinite.e2m1x2.f32 on 4 pairs: byte i = v[2i] in the low nibble, v[2i + 1] in the high one
DEV uint32_t e2m1x8(const float v[8]) {
  uint32_t r;
  asm("{.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};}"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}
DEV float e2m1f(uint32_t c) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return mag[c & 7];
}

// ---------------------------------------------------------------- mma.sync (fragment layouts: benchmarks/tc_probe_fp4/mma_fp4.cu)
template <int FMT> DEV void mma_fp4(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1, const float c[4], uint32_t sfa,
                                    uint32_t sfb) {
  if constexpr (FMT == NVFP4)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa),
          "r"(sfb));
  else
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {0, 0}, {%15}, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sfa),
          "r"(sfb));
}
DEV void mma_bf16(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------- shared memory: cp.async, ldmatrix
DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa(s)), "l"(g) : "memory"); }
DEV void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEV void ldsm4(uint32_t r[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// FP4 stage rows are 64 bytes (128 codes): 16-byte chunk c of row r at r * 64 + ((c ^ (r >> 1 & 3)) << 4)
DEV uint32_t swz64(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
// peel rows are 128 bytes (64 BF16): chunk c of row r at r * 128 + ((c ^ (r & 7)) << 4)
DEV uint32_t swz128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// ---------------------------------------------------------------- noise lines and F's transpose
// line(seed, role, index) = bytes 16·(index & 3) .. +15 of the line block ⌊index/4⌋ (pearl_c4.line_block):
// SHA-512(LINE_BLOCK ‖ seed (32 B) ‖ role ‖ c u32 LE), all 64 bytes, LINE_BLOCK = "pearl-c4/v2/line" ‖ \0 zero-padded to one
// 128-byte block, whose compression LINE_MID is (SHA-512's state after it, big-endian words)
__constant__ uint64_t LINE_MID[8] = {0xfdb83079a0e69062ull, 0x016df33feb42c07eull, 0x2947f8bafd5eee91ull,
                                     0x5c158735900dc8c1ull, 0xf3ad12a7257a2c36ull, 0x0fe5ee6d0ba4bc58ull,
                                     0x695f5fc8ee46c6ebull, 0x5acf3747d183fa80ull};
DEV uint64_t bswap64(uint64_t x) { return rs_be64((uint32_t)x, (uint32_t)(x >> 32)); }
// the line block's state: 8 big-endian words, lines 4c .. 4c + 3 in words (0, 1), (2, 3), (4, 5), (6, 7)
DEV void line_block(const uint64_t* seed, int role, uint32_t c, uint64_t st[8]) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 8; i++) st[i] = LINE_MID[i];
#pragma unroll
  for (int i = 0; i < 4; i++) w[i] = bswap64(seed[i]);
  w[4] = (uint64_t)(uint8_t)role << 56 | (uint64_t)(c & 0xFF) << 48 | (uint64_t)(c >> 8 & 0xFF) << 40 |
         (uint64_t)(c >> 16 & 0xFF) << 32 | (uint64_t)(c >> 24) << 24 | 0x80ull << 16;
#pragma unroll
  for (int i = 5; i < 15; i++) w[i] = 0;
  w[15] = 8 * (128 + 37);
  rs_compress(st, w);
}
DEV ulonglong2 line16(const uint64_t* seed, int role, int idx) {
  uint64_t st[8];
  line_block(seed, role, (uint32_t)idx >> 2, st);
  const int q = idx & 3;
  const uint64_t lo = q == 0 ? st[0] : q == 1 ? st[2] : q == 2 ? st[4] : st[6];
  const uint64_t hi = q == 0 ? st[1] : q == 1 ? st[3] : q == 2 ? st[5] : st[7];
  return make_ulonglong2(bswap64(lo), bswap64(hi));
}
// one thread per line block, lines 4·idx .. 4·idx + 3 below count.  p0 seed, p1 out; i0 role, i1 count
extern "C" __global__ void __launch_bounds__(128) lines(const Args a) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (4 * c >= a.i[1]) return;
  uint64_t st[8];
  line_block(reinterpret_cast<const uint64_t*>(a.p[0]), a.i[0], (uint32_t)c, st);
  ulonglong2* out = reinterpret_cast<ulonglong2*>(a.p[1]) + 4 * c;
#pragma unroll
  for (int q = 0; q < 4; q++)
    if (4 * c + q < a.i[1]) out[q] = make_ulonglong2(bswap64(st[2 * q]), bswap64(st[2 * q + 1]));
}

// ft[p][b] = code p of line 2b | code p of line 2b + 1 << 4.  p0 lines (k x 16 B), p1 ft (32 x k/2); i0 k
extern "C" __global__ void __launch_bounds__(256) transpose(const Args a) {
  const int K = a.i[0], idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 32 * (K / 2)) return;
  const int p = idx / (K / 2), b = idx % (K / 2);
  const uint8_t* L = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint32_t lo = (L[(2 * b) * 16 + p / 2] >> (4 * (p & 1))) & 0xF, hi = (L[(2 * b + 1) * 16 + p / 2] >> (4 * (p & 1))) & 0xF;
  reinterpret_cast<uint8_t*>(a.p[1])[idx] = (uint8_t)(lo | hi << 4);
}

// ---------------------------------------------------------------- the rows' commitments and the seeds
// audit.commit_rows as the verifier checks it: verity.ml.commitments.merkle's SHA-256 tree over a RangeIndexedDomain, with
// H(tag, parts) = SHA-256(FRAME ‖ |tag| u32 BE ‖ tag ‖ (|part| u64 BE ‖ part) ...), FRAME = "veritor/protocol/merkle/frame/v3\0",
// uint(v) = v big-endian in its fewest bytes (at least one), dom = the domain id (32 B, from the host: public before A exists),
//   leaf(i) = H("leaf", dom, uint(i), uint(i), "verity/pouw/row/v1", row i)      pad(r) = H("pad", dom, uint(r))
//   node(d, j, left, right) = H("node", dom, uint(d), uint(j), left, right)      over ranks padded to a power of two.
// Then the seed, pearl_c4.noise: H_"pearl-c4/v1/seed-A"(salt ‖ root ‖ root_b ‖ index u64 LE) (A), or
// H_"pearl-c4/v1/seed-B"(salt ‖ root) (B), H_ℓ `pearl_c4.labelled` (`labelled` below).  Messages are assembled at byte
// positions fixed at compile time (the rank's width L is a template parameter), so they stay in registers.
__constant__ uint8_t MK_FRAME[33] = "veritor/protocol/merkle/frame/v3";
template <int NW> struct Msg {
  uint32_t w[NW];
  DEV Msg() {
#pragma unroll
    for (int i = 0; i < NW; i++) w[i] = 0;
  }
  DEV void b(int pos, uint32_t v) { w[pos >> 2] |= (v & 0xFFu) << (24 - 8 * (pos & 3)); }
  DEV int u64(int pos, uint64_t v) {
#pragma unroll
    for (int i = 0; i < 8; i++) b(pos + i, (uint32_t)(v >> (56 - 8 * i)));
    return pos + 8;
  }
  template <int L> DEV int uv(int pos, uint32_t v) {
    pos = u64(pos, L);
#pragma unroll
    for (int i = 0; i < L; i++) b(pos + i, v >> (8 * (L - 1 - i)));
    return pos + L;
  }
  DEV int digest(int pos, const uint32_t h[8]) {
    pos = u64(pos, 32);
#pragma unroll
    for (int i = 0; i < 32; i++) b(pos + i, h[i >> 2] >> (24 - 8 * (i & 3)));
    return pos + 32;
  }
  template <int TL> DEV int head(const char (&tag)[TL], const uint32_t dom[8]) {
#pragma unroll
    for (int i = 0; i < 33; i++) b(i, MK_FRAME[i]);
    int pos = 33;
#pragma unroll
    for (int i = 0; i < 4; i++) b(pos + i, (uint32_t)(TL - 1) >> (24 - 8 * i));
    pos += 4;
#pragma unroll
    for (int i = 0; i < TL - 1; i++) b(pos + i, (uint8_t)tag[i]);
    return digest(pos + TL - 1, dom);
  }
  // SHA-256 padding for a LEN-byte message ending in this Msg's last block
  DEV void pad(int len) {
    b(len, 0x80);
    w[NW - 2] = 0;
    w[NW - 1] = (uint32_t)len * 8;
  }
};
template <int NW> DEV void sha_msg(const Msg<NW>& m, uint32_t h[8]) {
  ph::sha256_iv(h);
#pragma unroll
  for (int q = 0; q < NW / 16; q++) ph::sha256_compress(h, m.w + 16 * q);
}
template <int LJ> DEV void mk_node(const uint32_t dom[8], int d, int j, const uint32_t l[8], const uint32_t r[8], uint32_t h[8]) {
  Msg<48> m;
  int pos = m.head("node", dom);
  pos = m.template uv<1>(pos, d);
  pos = m.template uv<LJ>(pos, j);
  pos = m.digest(pos, l);
  m.pad(m.digest(pos, r));
  sha_msg(m, h);
}
DEV void node(const uint32_t dom[8], int d, int j, const uint32_t l[8], const uint32_t r[8], uint32_t h[8]) {
  if (j < 256) mk_node<1>(dom, d, j, l, r, h);
  else mk_node<2>(dom, d, j, l, r, h);
}
template <int L> DEV void mk_pad(const uint32_t dom[8], int rank, uint32_t h[8]) {
  Msg<32> m;
  m.pad(m.template uv<L>(m.head("pad", dom), rank));
  sha_msg(m, h);
}
// The leaf of row `rank` (V bytes, a multiple of 64): blocks 0-2 hold the P = 131 + 2L byte prefix and the row's first
// 192 − P bytes; each later block is 16 words of the row at offset 64β − P, which is S = −P mod 4 past a word boundary, so
// word t = __byte_perm(W[t], W[t + 1]) over 17 aligned words; the last block holds the row's final P mod 64 bytes and the
// padding.  The row streams in 16-byte loads, one block ahead in registers and eight ahead into L2.
template <int L> DEV void mk_leaf(const uint32_t dom[8], int rank, const uint4* R, int V, uint32_t h[8]) {
  constexpr int P = 131 + 2 * L, S = (4 - P % 4) % 4;
  constexpr uint32_t SEL = (S + 3) | (S + 2) << 4 | (S + 1) << 8 | S << 12;
  static_assert(192 - P >= 48 && (192 - P) / 4 == 14, "the rolling blocks start at word 14 of the row");
  uint4 r[4];
#pragma unroll
  for (int i = 0; i < 4; i++) r[i] = __ldg(R + i);
  {
    Msg<48> m;
    int pos = m.head("leaf", dom);
    pos = m.template uv<L>(pos, rank);
    pos = m.template uv<L>(pos, rank);
    const char schema[] = "verity/pouw/row/v1";
    pos = m.u64(pos, sizeof(schema) - 1);
#pragma unroll
    for (int i = 0; i < (int)sizeof(schema) - 1; i++) m.b(pos + i, (uint8_t)schema[i]);
    pos = m.u64(pos + sizeof(schema) - 1, (uint64_t)V);
#pragma unroll
    for (int j = 0; j < 192 - P; j++) {
      const uint4& q = r[j >> 4];
      const uint32_t w = (j & 12) == 0 ? q.x : (j & 12) == 4 ? q.y : (j & 12) == 8 ? q.z : q.w;
      m.b(P + j, w >> (8 * (j & 3)));
    }
    sha_msg(m, h);
  }
  const int NB = V / 64 - 1;
  uint4 carry = r[3], c1 = __ldg(R + 4), c2 = __ldg(R + 5), c3 = __ldg(R + 6), c4 = __ldg(R + 7);
#pragma unroll 1
  for (int b = 0; b < NB; b++) {
    const int nx = 8 + 4 * min(b, NB - 2), pf = 4 * min(b + 9, NB);
    asm volatile("prefetch.global.L2 [%0];" ::"l"(R + pf));
    const uint4 n1 = __ldg(R + nx), n2 = __ldg(R + nx + 1), n3 = __ldg(R + nx + 2), n4 = __ldg(R + nx + 3);
    const uint32_t W[17] = {carry.z, carry.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w,
                            c3.x, c3.y, c3.z, c3.w, c4.x, c4.y, c4.z};
    uint32_t blk[16];
#pragma unroll
    for (int t = 0; t < 16; t++) blk[t] = __byte_perm(W[t], W[t + 1], SEL);
    ph::sha256_compress(h, blk);
    carry = c4, c1 = n1, c2 = n2, c3 = n3, c4 = n4;
  }
  uint32_t blk[16];
  blk[0] = __byte_perm(carry.z, carry.w, SEL);
  blk[1] = __byte_perm(carry.w, 0x80u, SEL);
  blk[2] = __byte_perm(0x80u, 0u, SEL);
#pragma unroll
  for (int t = 3; t < 15; t++) blk[t] = 0;
  blk[15] = (uint32_t)(P + V) * 8;
  ph::sha256_compress(h, blk);
}
// pearl_c4.labelled: the first 32 bytes of SHA-512 over the label's block (its ll bytes, zero-padded to 128) and the len < 112
// bytes at msg, which it pads in place (msg has 128 bytes)
DEV void labelled(const char* label, int ll, uint8_t* msg, int len, uint8_t* out) {
  uint64_t st[8], w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint64_t x = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) x = x << 8 | (8 * i + t < ll ? (uint64_t)(uint8_t)label[8 * i + t] : 0);
    w[i] = x;
  }
  rs_iv(st);
  rs_compress(st, w);
  for (int i = len; i < 128; i++) msg[i] = 0;
  msg[len] = 0x80;
  msg[126] = (uint8_t)((8 * (128 + len)) >> 8);
  msg[127] = (uint8_t)(8 * (128 + len));
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint64_t x = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) x = x << 8 | msg[8 * i + t];
    w[i] = x;
  }
  rs_compress(st, w);
  for (int t = 0; t < 32; t++) out[t] = (uint8_t)(st[t >> 3] >> (56 - 8 * (t & 7)));
}
DEV void load_digest(const uint8_t* p, uint32_t h[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++) h[i] = ph::bswap32(__ldcg(reinterpret_cast<const uint32_t*>(p) + i));
}
DEV void store_digest(uint8_t* p, const uint32_t h[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++) reinterpret_cast<uint32_t*>(p)[i] = ph::bswap32(h[i]);
}
// One warp per 32 ranks: its leaves, then its subtree by shuffles; the last warp to finish builds the top levels from the
// subroots and derives the seed.  p0 rows, p1 dom, p2 scratch (32·max(1, width/32) + 192 B), p3 ticket (a u32, 0 between
// launches: the last warp resets it), p4 root out, p5 salt, p6 root_b (null: B's seed), p7 seed out; i0 rows (1 … 65,536),
// i1 row bytes (a multiple of 1,024), i2 the matmul's index
extern "C" __global__ void __launch_bounds__(32) commit_rows(const Args a) {
  const int count = a.i[0], V = a.i[1], lane = threadIdx.x, rank = 32 * blockIdx.x + lane;
  const int width = count <= 1 ? 1 : 1 << (32 - __clz(count - 1)), wb = min(width, 32), G = gridDim.x;
  uint8_t* scr = reinterpret_cast<uint8_t*>(a.p[2]);
  uint32_t dom[8], h[8];
  load_digest(reinterpret_cast<const uint8_t*>(a.p[1]), dom);
  const uint4* R = reinterpret_cast<const uint4*>(a.p[0] + (size_t)min(rank, count - 1) * V);
  if (rank < count) {
    if (rank < 256) mk_leaf<1>(dom, rank, R, V, h);
    else mk_leaf<2>(dom, rank, R, V, h);
  } else if (rank < 256) {
    mk_pad<1>(dom, rank, h);
  } else {
    mk_pad<2>(dom, rank, h);
  }
  int d = 0;
  for (int span = 1; span < wb; span <<= 1, d++) {
    uint32_t r[8];
#pragma unroll
    for (int i = 0; i < 8; i++) r[i] = __shfl_down_sync(FULL, h[i], span);
    if ((lane & (2 * span - 1)) == 0) node(dom, d, rank >> (d + 1), h, r, h);
  }
  if (lane == 0) store_digest(scr + 32 * blockIdx.x, h);
  __threadfence();
  unsigned ticket = 0;
  if (lane == 0) ticket = atomicAdd(reinterpret_cast<unsigned*>(a.p[3]), 1u);
  if (__shfl_sync(FULL, ticket, 0) != (unsigned)G - 1) return;
  __threadfence();
  for (int cnt = G; cnt > 1; cnt >>= 1, d++) {
    for (int base = 0; base < cnt / 2; base += 32) {
      const int j = base + lane;
      uint32_t l[8], r[8], o[8];
      if (j < cnt / 2) {
        load_digest(scr + 64 * j, l);
        load_digest(scr + 64 * j + 32, r);
        node(dom, d, j, l, r, o);
      }
      __syncwarp();
      if (j < cnt / 2) store_digest(scr + 32 * j, o);
      __syncwarp();
    }
  }
  if (lane) return;
  load_digest(scr, h);
  store_digest(reinterpret_cast<uint8_t*>(a.p[4]), h);
  uint8_t* msg = scr + 32 * G;
  const bool is_a = a.p[6] != 0;
  for (int i = 0; i < 32; i++) msg[i] = reinterpret_cast<const uint8_t*>(a.p[5])[i];
  store_digest(msg + 32, h);
  if (is_a) {
    for (int i = 0; i < 32; i++) msg[64 + i] = reinterpret_cast<const uint8_t*>(a.p[6])[i];
    for (int i = 0; i < 8; i++) msg[96 + i] = (uint8_t)((uint64_t)(uint32_t)a.i[2] >> (8 * i));
  }
  labelled(is_a ? "pearl-c4/v1/seed-A" : "pearl-c4/v1/seed-B", 18, msg, is_a ? 104 : 64, reinterpret_cast<uint8_t*>(a.p[7]));
  *reinterpret_cast<unsigned*>(a.p[3]) = 0;
}

// ---------------------------------------------------------------- forming
// One CTA per 16 rows.  (1) Per row, one warp: s = max|x|, ρ over every 8th position in pearl_c.s5_stats' order (lane l
// accumulates positions 8(l + 32j) by FMA, then the xor butterfly 16 ... 1), σ = ρ/4, t = s + 4σ, α = 2688/max(t, 2^-32)
// (MXFP4: floored to a power of two), β = (σα)·f32(16/√2346.125), a_E = ue4m3(β).  (2) Ẽ = bf16(E·a_E·2^-4) into P at
// column eoff (negated for B).  (3) Warps take k64 chunks: per element one noise atom D = NVF4 mma(C = α·x, E | 0 at a_E,
// F_t | 0 at 2^-4), then per block of BLK the noisy scale and one E2M1 cast of D·r_b.  Grid (rows/16, S): CTA y takes
// the y-th of S equal runs of k64 chunks, and every y computes its rows' statistics (their order is pinned; y = 0 writes
// α, 1/α, a_E and Ẽ).  E's 16 lines come from p9's seed (then y = 0 writes them to p1) or, when p9 is null, from p1.
// p0 x, p1 E lines (rows x 16 B), p2 noise basis lines (k x 16 B), p3 codes, p4 scales, p5 alpha, p6 inv, p7 ae, p8 P,
// p9 seed or null; i0 rows, i1 k, i2 eoff, i3 negate Ẽ, i4 E's role
// NVFP4's block step, V (the same bytes out; handoffs/ue4m3-scale-lut.md): 0 the amax as an integer max over |D|'s bits,
// ue4m3 then the floor as an integer max, r_b from a 128-entry table; 1 the amax by FMNMX on |D|, the floor in FP32 and one
// packed e4m3x2 encode of the thread's two blocks; 2 (form_nv, the priced path) as 1, with r_b from lut256: one PRMT and
// one conflict-free LDS per scale.  Its table (FORM_LUT_BYTES of dynamic shared memory) holds, per byte b, 32 lane copies of
// RN(1/s_b) at a 256-byte stride from tab − 256: b is 0x01 ... 0x7E (the floor 2^-9 is 0x01, SATFINITE stops at 0x7E), and
// the last row is half used, so three CTAs fit an SM's 100 KB.
constexpr int FORM_LUT_BYTES = 125 * 256 + 128;
DEV float lut256(const float* tab, uint32_t pair, int k, uint32_t lane4) {
  const uint32_t addr = __byte_perm(pair, lane4, 0x5504u | ((uint32_t)k << 4));
  return *reinterpret_cast<const float*>(reinterpret_cast<const char*>(tab) + addr - 256);
}
template <int FMT, int V = 0> DEV void form_body(const Args& a) {
  constexpr int BLK = Fmt<FMT>::BLK;
  static_assert(V == 0 || FMT == NVFP4, "the block-step variants are NVFP4's");
  const uint16_t* X = reinterpret_cast<const uint16_t*>(a.p[0]);
  const uint8_t* F = reinterpret_cast<const uint8_t*>(a.p[2]);
  uint8_t* codes = reinterpret_cast<uint8_t*>(a.p[3]);
  uint8_t* scales = reinterpret_cast<uint8_t*>(a.p[4]);
  const int K = a.i[1];
  __shared__ float s_alpha[16];
  __shared__ uint32_t s_ae[16];
  __shared__ __align__(16) uint8_t E[16 * 16];
  __shared__ float s_rcp[V == 2 ? 1 : 128];
  extern __shared__ __align__(128) uint8_t smem[];
  float* tab = reinterpret_cast<float*>(smem);
  const int r0 = blockIdx.x * 16, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool lead = blockIdx.y == 0;

  if constexpr (V == 2) {
    // warp w: bytes 16w + 1 ... 16w + 16, one divide per lane, then four rows' lane copies per STS.128
    const float r = div_rn(1.0f, ue4m3f(min(16 * warp + 1 + (lane & 15), 126)));
#pragma unroll
    for (int it = 0; it < 4; it++) {
      const int j = 4 * it + (lane >> 3), b = 16 * warp + 1 + j;
      const float v = __shfl_sync(FULL, r, j);
      if (b <= 126) reinterpret_cast<float4*>(tab)[(b - 1) * 16 + (lane & 7)] = make_float4(v, v, v, v);
    }
  } else if constexpr (FMT == NVFP4)
    for (int c = threadIdx.x; c < 128; c += blockDim.x) s_rcp[c] = div_rn(1.0f, ue4m3f(c));
  if (threadIdx.x < 16) {
    ulonglong2* eg = reinterpret_cast<ulonglong2*>(a.p[1]) + r0 + threadIdx.x;
    ulonglong2 l;
    if (a.p[9]) {
      l = line16(reinterpret_cast<const uint64_t*>(a.p[9]), a.i[4], r0 + threadIdx.x);
      if (lead) *eg = l;
    } else {
      l = *eg;
    }
    reinterpret_cast<ulonglong2*>(E)[threadIdx.x] = l;
  }
  for (int rr = warp; rr < 16; rr += 8) {
    const uint16_t* x = X + (size_t)(r0 + rr) * K;
    uint32_t mx = 0;
    float part = 0.0f;
    for (int i = lane; i < K / 8; i += 32) {
      const uint4 v = *reinterpret_cast<const uint4*>(x + 8 * i);
      part = __fmaf_rn(bf2f(v.x), bf2f(v.x), part);
      mx = max(mx, max(max(v.x & 0x7FFFu, (v.x >> 16) & 0x7FFFu), max(v.y & 0x7FFFu, (v.y >> 16) & 0x7FFFu)));
      mx = max(mx, max(max(v.z & 0x7FFFu, (v.z >> 16) & 0x7FFFu), max(v.w & 0x7FFFu, (v.w >> 16) & 0x7FFFu)));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) part = __fadd_rn(part, __shfl_xor_sync(FULL, part, o));
    mx = __reduce_max_sync(FULL, mx);
    if (lane == 0) {
      const float s = __uint_as_float(mx << 16);
      const float rho = sqrt_rn(div_rn(part, (float)(K / 8)));
      const float sigma = __fmul_rn(rho, DELTA);
      const float t = __fadd_rn(s, __fmul_rn(FOUR, sigma));
      float alpha = div_rn(TOP, fmaxf(t, __uint_as_float(FLOOR_BITS)));
      if constexpr (FMT == MXFP4) alpha = __uint_as_float(__float_as_uint(alpha) & 0xFF800000u);
      const uint32_t ae = ue4m3(__fmul_rn(__fmul_rn(sigma, alpha), __uint_as_float(K_BETA_BITS)));
      s_alpha[rr] = alpha;
      s_ae[rr] = ae;
      if (lead) {
        reinterpret_cast<float*>(a.p[5])[r0 + rr] = alpha;
        reinterpret_cast<float*>(a.p[6])[r0 + rr] = div_rn(1.0f, alpha);
        reinterpret_cast<uint8_t*>(a.p[7])[r0 + rr] = (uint8_t)ae;
      }
    }
  }
  __syncthreads();

  if (lead) {
    uint16_t* P = reinterpret_cast<uint16_t*>(a.p[8]);
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
      const int rr = i >> 5, p = i & 31;
      const uint32_t e = (E[rr * 16 + p / 2] >> (4 * (p & 1))) & 0xF;
      const float v = __fmul_rn(__fmul_rn(e2m1f(e), ue4m3f(s_ae[rr])), 0.0625f);
      const uint16_t w = (uint16_t)(__float_as_uint(v) >> 16) | (e & 8 ? 0x8000 : 0);
      P[(size_t)(r0 + rr) * 64 + a.i[2] + p] = w ^ (a.i[3] ? 0x8000 : 0);
    }
  }

  const int g = lane >> 2, t = lane & 3;
  const uint32_t ea[4] = {*reinterpret_cast<const uint32_t*>(E + g * 16 + 4 * t),
                          *reinterpret_cast<const uint32_t*>(E + (g + 8) * 16 + 4 * t), 0u, 0u};
  const uint32_t sfa = s_ae[8 * (lane & 1) + (lane >> 2)] * 0x01010101u;
  const float al0 = s_alpha[g], al1 = s_alpha[g + 8];
  const uint16_t* x0 = X + (size_t)(r0 + g) * K;
  const uint16_t* x1 = X + (size_t)(r0 + g + 8) * K;
  const int run = K / 64 / gridDim.y;
  // Column n of atom jb is position 16(n >> 1) + 2 jb + (n & 1) of the chunk, so lane (g, t) forms positions 16t ... 16t + 15
  // of rows g and g + 8: its codes are two whole words per row, and an NVFP4 block (MXFP4: half a block) is its own
  for (int c = blockIdx.y * run + warp; c < (blockIdx.y + 1) * run; c += blockDim.x / 32) {
    const uint4* pg = reinterpret_cast<const uint4*>(x0 + 64 * c + 16 * t);
    const uint4* ph = reinterpret_cast<const uint4*>(x1 + 64 * c + 16 * t);
    const uint4 g0 = pg[0], g1 = pg[1], h0 = ph[0], h1 = ph[1];
    const uint32_t xg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    const uint32_t xh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
    const uint8_t* fl = F + (size_t)(64 * c + 16 * (g >> 1) + (g & 1)) * 16 + 4 * t;
    float d[8][4];
#pragma unroll
    for (int jb = 0; jb < 8; jb++) {
      const float cf[4] = {__fmul_rn(al0, bf2f(xg[jb])), __fmul_rn(al0, bf2f(xg[jb] >> 16)), __fmul_rn(al1, bf2f(xh[jb])),
                           __fmul_rn(al1, bf2f(xh[jb] >> 16))};
      const uint32_t fb = *reinterpret_cast<const uint32_t*>(fl + 32 * jb);
      mma_fp4<NVFP4>(d[jb], ea, fb, 0u, cf, sfa, NOISE_F_SCALE);
    }
    uint32_t m0 = 0, m1 = 0;
    if constexpr (V == 0) {
#pragma unroll
      for (int jb = 0; jb < 8; jb++) {
        m0 = max(m0, max(__float_as_uint(d[jb][0]) & 0x7FFFFFFFu, __float_as_uint(d[jb][1]) & 0x7FFFFFFFu));
        m1 = max(m1, max(__float_as_uint(d[jb][2]) & 0x7FFFFFFFu, __float_as_uint(d[jb][3]) & 0x7FFFFFFFu));
      }
    } else {
      float f0 = 0.0f, f1 = 0.0f;
#pragma unroll
      for (int jb = 0; jb < 8; jb++) {
        f0 = fmaxf(f0, fmaxf(fabsf(d[jb][0]), fabsf(d[jb][1])));
        f1 = fmaxf(f1, fmaxf(fabsf(d[jb][2]), fabsf(d[jb][3])));
      }
      m0 = __float_as_uint(f0);
      m1 = __float_as_uint(f1);
    }
    if constexpr (BLK == 32) {
      m0 = max(m0, __shfl_xor_sync(FULL, m0, 1));
      m1 = max(m1, __shfl_xor_sync(FULL, m1, 1));
    }
    const float v0 = __fmul_rn(__uint_as_float(m0), __uint_as_float(SIXTH_BITS)),
                v1 = __fmul_rn(__uint_as_float(m1), __uint_as_float(SIXTH_BITS));
    uint32_t c0, c1;
    float rg, rh;
    if constexpr (V != 0) {
      const uint32_t pair = __nv_cvt_float2_to_fp8x2(make_float2(fmaxf(v0, 0x1p-9f), fmaxf(v1, 0x1p-9f)), __NV_SATFINITE, __NV_E4M3);
      c0 = pair & 0xFFu;
      c1 = pair >> 8;
      if constexpr (V == 2) {
        rg = lut256(tab, pair, 0, (uint32_t)lane << 2);
        rh = lut256(tab, pair, 1, (uint32_t)lane << 2);
      } else {
        rg = s_rcp[c0];
        rh = s_rcp[c1];
      }
    } else if constexpr (FMT == NVFP4) {
      c0 = max(ue4m3(v0), 1u);
      c1 = max(ue4m3(v1), 1u);
      rg = s_rcp[c0];
      rh = s_rcp[c1];
    } else {
      c0 = ue8m0_up(__float_as_uint(v0));
      c1 = ue8m0_up(__float_as_uint(v1));
      rg = pow2f(127 - (int)c0);
      rh = pow2f(127 - (int)c1);
    }
    uint32_t w[4];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float vg[8], vh[8];
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int jb = 4 * h + jj;
        vg[2 * jj] = __fmul_rn(d[jb][0], rg);
        vg[2 * jj + 1] = __fmul_rn(d[jb][1], rg);
        vh[2 * jj] = __fmul_rn(d[jb][2], rh);
        vh[2 * jj + 1] = __fmul_rn(d[jb][3], rh);
      }
      w[h] = e2m1x8(vg);
      w[2 + h] = e2m1x8(vh);
    }
    *reinterpret_cast<uint2*>(codes + (size_t)(r0 + g) * (K / 2) + 32 * c + 8 * t) = make_uint2(w[0], w[1]);
    *reinterpret_cast<uint2*>(codes + (size_t)(r0 + g + 8) * (K / 2) + 32 * c + 8 * t) = make_uint2(w[2], w[3]);
    if (BLK == 16 || !(t & 1)) {
      uint8_t* sp = scales + ((size_t)(c >> 1) * a.i[0] + r0 + g) * (128 / BLK) + (c & 1) * (64 / BLK) + t * 16 / BLK;
      sp[0] = (uint8_t)c0;
      sp[8 * (128 / BLK)] = (uint8_t)c1;
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) form_nv(const Args a) { form_body<NVFP4, 2>(a); }
extern "C" __global__ void __launch_bounds__(256) form_mx(const Args a) { form_body<MXFP4>(a); }
extern "C" __global__ void __launch_bounds__(256) form_nv_v0(const Args a) { form_body<NVFP4, 0>(a); }
extern "C" __global__ void __launch_bounds__(256) form_nv_v1(const Args a) { form_body<NVFP4, 1>(a); }
extern "C" __global__ void form_nv_smem(int* out) { *out = FORM_LUT_BYTES; }
extern "C" __global__ void form_mx_smem(int* out) { *out = 0; }

// ---------------------------------------------------------------- the fused products: rows x 32 chains against F
// out[row][q] = the atom chained from +0 over k/64 steps of codes[row] . ft[q] (ft at scale 1.0).  One CTA per 16·RB rows,
// one warp per 16 rows and n8 block of q.  Codes, ft and scales stream through a CH_STAGES-deep cp.async ring of CH_U k64
// chunks (256 bytes per row: 16-byte chunk j of stage row r at r·256 + (j ^ (r & 7))·16), so a chain waits on its MMAs
// alone.  p0 codes, p1 scales (null: every scale 1.0), p2 ft, p3 out; i0 rows, i1 k (512 | k), i2 mode (0: FP32 rows x 32,
// 1: bf16 into P[row][32 + q])
constexpr int CH_STAGES = 4, CH_U = 8;
template <int FMT, int RB> struct ChGeo {
  static constexpr int ROWS = 16 * RB, THREADS = 128 * RB, SFB = 128 / Fmt<FMT>::BLK, SROW = CH_U / 2 * SFB;
  static constexpr int A_BYTES = ROWS * 256, F_BYTES = 32 * 256, STAGE = A_BYTES + F_BYTES + ROWS * SROW;
  static constexpr int SMEM = CH_STAGES * STAGE;
};
template <int FMT, int RB> DEV void chain32_body(const Args& a) {
  using G = ChGeo<FMT, RB>;
  constexpr int SFB = G::SFB, PER = G::ROWS * SFB / 16;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint8_t* S = reinterpret_cast<const uint8_t*>(a.p[1]);
  const uint8_t* FT = reinterpret_cast<const uint8_t*>(a.p[2]);
  const int ROWS_ALL = a.i[0], K = a.i[1], NB = K / 64 / CH_U, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * G::ROWS, ra = (warp / 4) * 16 + g, rq = 8 * (warp % 4) + g;
  const int rs = (warp / 4) * 16 + 8 * (lane & 1) + (lane >> 2);

  auto load = [&](int b, int st) {
    uint8_t* s = smem + st * G::STAGE;
    for (int i = tid; i < G::ROWS * 16; i += G::THREADS)
      cp16(s + (i >> 4) * 256 + (((i & 15) ^ ((i >> 4) & 7)) << 4),
           A + (size_t)(r0 + (i >> 4)) * (K / 2) + b * 256 + (i & 15) * 16);
    for (int i = tid; i < 32 * 16; i += G::THREADS)
      cp16(s + G::A_BYTES + (i >> 4) * 256 + (((i & 15) ^ ((i >> 4) & 7)) << 4),
           FT + (size_t)(i >> 4) * (K / 2) + b * 256 + (i & 15) * 16);
    if (S)
      for (int i = tid; i < CH_U / 2 * PER; i += G::THREADS)
        cp16(s + G::A_BYTES + G::F_BYTES + i * 16,
             S + ((size_t)(CH_U / 2 * b + i / PER) * ROWS_ALL + r0) * SFB + (i % PER) * 16);
  };

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int st = 0; st < CH_STAGES - 1; st++) {
    if (st < NB) load(st, st);
    cp_commit();
  }
  for (int b = 0; b < NB; b++) {
    cp_wait<CH_STAGES - 2>();
    __syncthreads();
    if (b + CH_STAGES - 1 < NB) load(b + CH_STAGES - 1, (b + CH_STAGES - 1) % CH_STAGES);
    cp_commit();
    const uint8_t* sA = smem + (b % CH_STAGES) * G::STAGE;
    const uint8_t* sF = sA + G::A_BYTES;
    const uint8_t* sS = sF + G::F_BYTES;
#pragma unroll
    for (int u = 0; u < CH_U; u++) {
      // ra, ra + 8 and rq are all g mod 8, so every read's chunk is swizzled by g
      const int lo = ((2 * u) ^ g) << 4, hi = ((2 * u + 1) ^ g) << 4;
      const uint32_t af[4] = {*reinterpret_cast<const uint32_t*>(sA + ra * 256 + lo + 4 * t),
                              *reinterpret_cast<const uint32_t*>(sA + (ra + 8) * 256 + lo + 4 * t),
                              *reinterpret_cast<const uint32_t*>(sA + ra * 256 + hi + 4 * t),
                              *reinterpret_cast<const uint32_t*>(sA + (ra + 8) * 256 + hi + 4 * t)};
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sF + rq * 256 + lo + 4 * t);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(sF + rq * 256 + hi + 4 * t);
      uint32_t sf = Fmt<FMT>::ONE;
      if (S) {
        const uint8_t* sp = sS + ((u >> 1) * G::ROWS + rs) * SFB + (u & 1) * (SFB / 2);
        if constexpr (FMT == NVFP4) sf = *reinterpret_cast<const uint32_t*>(sp);
        else sf = *reinterpret_cast<const uint16_t*>(sp);
      }
      mma_fp4<FMT>(acc, af, b0, b1, acc, sf, Fmt<FMT>::ONE);
    }
  }
  cp_wait<0>();
  const int row = r0 + ra, col = 8 * (warp % 4) + 2 * t;
  if (a.i[2] == 0) {
    float* out = reinterpret_cast<float*>(a.p[3]);
    *reinterpret_cast<float2*>(out + (size_t)row * 32 + col) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2*>(out + (size_t)(row + 8) * 32 + col) = make_float2(acc[2], acc[3]);
  } else {
    uint16_t* P = reinterpret_cast<uint16_t*>(a.p[3]);
    *reinterpret_cast<uint32_t*>(P + (size_t)row * 64 + 32 + col) = f2bf(acc[0]) | (uint32_t)f2bf(acc[1]) << 16;
    *reinterpret_cast<uint32_t*>(P + (size_t)(row + 8) * 64 + 32 + col) = f2bf(acc[2]) | (uint32_t)f2bf(acc[3]) << 16;
  }
}
#define CHAIN32(NAME, FMT, RB)                                                                                  \
  extern "C" __global__ void __launch_bounds__(128 * RB) NAME(const Args a) { chain32_body<FMT, RB>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = ChGeo<FMT, RB>::SMEM; }
CHAIN32(chain32_nv, NVFP4, 1)
CHAIN32(chain32_nv_r2, NVFP4, 2)
CHAIN32(chain32_mx, MXFP4, 1)
CHAIN32(chain32_mx_r2, MXFP4, 2)

// P_B[j][q] = bf16(FP32 add(Σ_p fma(Ẽ_B[j][p], G[p][q]) from +0 in p order, −BF[j][q])), Ẽ_B = −P_B[j][32 + p] (form_*
// wrote it negated).  p0 P_B, p1 G (32 x 32 FP32), p2 BF (n x 32 FP32); i0 n
extern "C" __global__ void __launch_bounds__(256) pb(const Args a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= a.i[0] * 32) return;
  const int j = idx / 32, q = idx % 32;
  uint16_t* P = reinterpret_cast<uint16_t*>(a.p[0]);
  const float* G = reinterpret_cast<const float*>(a.p[1]);
  float acc = 0.0f;
#pragma unroll 8
  for (int p = 0; p < 32; p++) acc = __fmaf_rn(bf2f(P[(size_t)j * 64 + 32 + p] ^ 0x8000u), G[p * 32 + q], acc);
  P[(size_t)j * 64 + q] = f2bf(__fadd_rn(acc, -reinterpret_cast<const float*>(a.p[2])[idx]));
}

// ---------------------------------------------------------------- the chain, the peel, the epilogue hashing
// One CTA per BM x BN output block (128 x 128: 8 warps, 64 x 128: 4, 64 x 64: 2 for decode's narrow m, so that n/64 CTAs
// fill the SMs), warps 32 x 64 (two m16 x n64 regions each), BK = 128 codes,
// STAGES deep cp.async.  C~ = the atom chained from +0 over k/64 steps, no promotion.  Modes: PLAIN y = bf16(C~ inv inv);
// NOHASH C~, U = C~ + four BF16 m16n8k16 steps of P_A P_B^T, y from U; HASH as NOHASH, plus per region
// ph::quad_msg_digest<2> over C~ | U (0x21).
// p0 A codes, p1 A scales, p2 B codes, p3 B scales, p4 P_A, p5 P_B, p6 inv_A, p7 inv_B, p8 y, p9 digests, p10 C~ out and
// p11 U out (FP32, m x n; null: not stored); i0 m, i1 n, i2 k
enum { M_PLAIN, M_NOHASH, M_HASH };
constexpr int BK = 128, STAGES = 4;
template <int FMT, int BM, int BN> struct Geo {
  static constexpr int WN = BN / 64, WARPS = BM / 32 * WN, THREADS = 32 * WARPS, SFB = BK / Fmt<FMT>::BLK;
  static constexpr int A_BYTES = BM * 64, B_BYTES = BN * 64, SA_BYTES = BM * SFB, SB_BYTES = BN * SFB;
  static constexpr int STAGE = A_BYTES + B_BYTES + SA_BYTES + SB_BYTES;
  static constexpr int WARP_EPI = 1024;
  static constexpr int EPI = BM * 128 + BN * 128 + WARPS * WARP_EPI;
  static constexpr int SMEM = STAGES * STAGE > EPI ? STAGES * STAGE : EPI;
};

template <int FMT, int BM, int BN, int MODE> DEV void gemm_body(const Args& a) {
  using G = Geo<FMT, BM, BN>;
  constexpr int SFB = G::SFB;
  extern __shared__ __align__(128) uint8_t smem[];
  const int M = a.i[0], N = a.i[1], K = a.i[2], KB = K / BK;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wm = warp / G::WN, wn = warp % G::WN, g = lane >> 2;
  const int t = lane & 3;
  const int tiles_n = N / BN, m0 = (blockIdx.x / tiles_n) * BM, n0 = (blockIdx.x % tiles_n) * BN;
  if (m0 >= M) return;
  const uint8_t* Ag = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)m0 * (K / 2);
  const uint8_t* SAg = reinterpret_cast<const uint8_t*>(a.p[1]) + (size_t)m0 * SFB;
  const uint8_t* Bg = reinterpret_cast<const uint8_t*>(a.p[2]) + (size_t)n0 * (K / 2);
  const uint8_t* SBg = reinterpret_cast<const uint8_t*>(a.p[3]) + (size_t)n0 * SFB;

  auto load = [&](int kb, int st) {
    uint8_t* s = smem + st * G::STAGE;
    for (int i = tid; i < BM * 4; i += G::THREADS)
      cp16(s + swz64(i >> 2, i & 3), Ag + (size_t)(i >> 2) * (K / 2) + kb * 64 + (i & 3) * 16);
    for (int i = tid; i < BN * 4; i += G::THREADS)
      cp16(s + G::A_BYTES + swz64(i >> 2, i & 3), Bg + (size_t)(i >> 2) * (K / 2) + kb * 64 + (i & 3) * 16);
    for (int i = tid; i < (BM + BN) * SFB / 16; i += G::THREADS) {
      const bool isa = i < BM * SFB / 16;
      const int j = isa ? i : i - BM * SFB / 16;
      cp16(s + G::A_BYTES + G::B_BYTES + (isa ? 0 : G::SA_BYTES) + j * 16,
           (isa ? SAg + (size_t)kb * M * SFB : SBg + (size_t)kb * N * SFB) + j * 16);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; mi++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[mi][j][i] = 0.0f;

  // One k64 group's fragments: A's two m16 blocks, B's eight n8 blocks and their scales, from stage st
  struct Frag { uint32_t af[2][4], bf[8][2], sfa[2], sfb[8]; };
  auto frags = [&](int st, int at, Frag& f) {
    const uint8_t* s = smem + st * G::STAGE;
    const uint32_t sA = sa(s), sB = sa(s + G::A_BYTES);
    const uint8_t* sSA = s + G::A_BYTES + G::B_BYTES;
    const uint8_t* sSB = sSA + G::SA_BYTES;
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
      const int r = wm * 32 + mi * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
      ldsm4(f.af[mi], sA + swz64(r, 2 * at + (lane >> 4)));
      const uint8_t* sp = sSA + (wm * 32 + mi * 16 + 8 * (lane & 1) + (lane >> 2)) * SFB + at * (SFB / 2);
      f.sfa[mi] = SFB == 8 ? *reinterpret_cast<const uint32_t*>(sp) : (uint32_t)*reinterpret_cast<const uint16_t*>(sp);
    }
#pragma unroll
    for (int nj = 0; nj < 4; nj++) {
      uint32_t r4[4];
      const int r = wn * 64 + nj * 16 + (lane & 7) + 8 * (lane >> 4);
      ldsm4(r4, sB + swz64(r, 2 * at + ((lane >> 3) & 1)));
      f.bf[2 * nj][0] = r4[0];
      f.bf[2 * nj][1] = r4[1];
      f.bf[2 * nj + 1][0] = r4[2];
      f.bf[2 * nj + 1][1] = r4[3];
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint8_t* sp = sSB + (wn * 64 + 8 * j + (lane >> 2)) * SFB + at * (SFB / 2);
      f.sfb[j] = SFB == 8 ? *reinterpret_cast<const uint32_t*>(sp) : (uint32_t)*reinterpret_cast<const uint16_t*>(sp);
    }
  };
  auto mmas = [&](const Frag& f) {
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < 8; j++) mma_fp4<FMT>(acc[mi][j], f.af[mi], f.bf[j][0], f.bf[j][1], acc[mi][j], f.sfa[mi], f.sfb[j]);
  };

#pragma unroll
  for (int st = 0; st < STAGES - 1; st++) {
    if (st < KB) load(st, st);
    cp_commit();
  }
  // Fragments run one k64 group ahead of the MMAs.  The barrier sits between a tile's two groups: past it, tile kb + 1
  // has landed and every warp is done with stage kb - 1, which the next load refills.  Each accumulator still chains
  // its k64 groups in order.
  Frag f0, f1;
  cp_wait<STAGES - 2>();
  __syncthreads();
  frags(0, 0, f0);
  for (int kb = 0; kb < KB; kb++) {
    frags(kb % STAGES, 1, f1);
    mmas(f0);
    cp_wait<STAGES - 3>();
    __syncthreads();
    if (kb + STAGES - 1 < KB) load(kb + STAGES - 1, (kb + STAGES - 1) % STAGES);
    cp_commit();
    if (kb + 1 < KB) frags((kb + 1) % STAGES, 0, f0);
    mmas(f1);
  }
  cp_wait<0>();
  __syncthreads();

  const float* inv_a = reinterpret_cast<const float*>(a.p[6]);
  const float* inv_b = reinterpret_cast<const float*>(a.p[7]);
  uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[8]);
  float* Cout = reinterpret_cast<float*>(a.p[10]);
  float* Uout = reinterpret_cast<float*>(a.p[11]);
  const int wrow = m0 + wm * 32, wcol = n0 + wn * 64;

  if constexpr (MODE != M_PLAIN) {
    for (int i = tid; i < (BM + BN) * 8; i += G::THREADS) {
      const bool isa = i < BM * 8;
      const int r = (isa ? i : i - BM * 8) >> 3, c = i & 7;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(a.p[isa ? 4 : 5]) + (size_t)((isa ? m0 : n0) + r) * 128 + c * 16;
      cp16(smem + (isa ? 0 : BM * 128) + swz128(r, c), src);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
  }
  uint64_t* scratch = reinterpret_cast<uint64_t*>(smem + BM * 128 + BN * 128 + warp * G::WARP_EPI);

  // c: region mi's C~ by selects, so that nothing indexes acc by mi when ptxas keeps this loop rolled (the hash mode's)
  for (int mi = 0; mi < 2; mi++) {
    float c[8][4], u[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int i = 0; i < 4; i++) u[j][i] = c[j][i] = mi ? acc[1][j][i] : acc[0][j][i];
    if constexpr (MODE != M_PLAIN) {
      const uint32_t sPA = sa(smem), sPB = sa(smem + BM * 128);
#pragma unroll
      for (int s = 0; s < 4; s++) {
        uint32_t pa[4];
        ldsm4(pa, sPA + swz128(wm * 32 + mi * 16 + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * s + (lane >> 4)));
#pragma unroll
        for (int nj = 0; nj < 4; nj++) {
          uint32_t r4[4];
          ldsm4(r4, sPB + swz128(wn * 64 + nj * 16 + (lane & 7) + 8 * (lane >> 4), 2 * s + ((lane >> 3) & 1)));
          mma_bf16(u[2 * nj], pa, r4[0], r4[1]);
          mma_bf16(u[2 * nj + 1], pa, r4[2], r4[3]);
        }
      }
    }
    const int row = wrow + mi * 16 + g;
    if constexpr (MODE == M_HASH) {
      uint64_t dg[4];
      ph::quad_msg_digest<2>([&](int arr, int j, int i) { return arr ? u[j][i] : c[j][i]; }, ph::pearlc::D_MESSAGE,
                             scratch, dg);
      ulonglong2* dst = reinterpret_cast<ulonglong2*>(a.p[9] + ((size_t)(wrow + mi * 16 + ph::quad_row(lane)) * (N / 32) +
                                                               wcol / 32 + ph::quad_chunk(lane)) * 32);
      dst[0] = make_ulonglong2(dg[0], dg[1]);
      dst[1] = make_ulonglong2(dg[2], dg[3]);
    }
    const float ia0 = inv_a[row], ia1 = inv_a[row + 8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int col = wcol + 8 * j + 2 * t;
      const float ib0 = inv_b[col], ib1 = inv_b[col + 1];
      *reinterpret_cast<uint32_t*>(Y + (size_t)row * N + col) =
          f2bf(__fmul_rn(__fmul_rn(u[j][0], ia0), ib0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(u[j][1], ia0), ib1)) << 16;
      *reinterpret_cast<uint32_t*>(Y + (size_t)(row + 8) * N + col) =
          f2bf(__fmul_rn(__fmul_rn(u[j][2], ia1), ib0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(u[j][3], ia1), ib1)) << 16;
      if (Cout) {
        *reinterpret_cast<float2*>(Cout + (size_t)row * N + col) = make_float2(c[j][0], c[j][1]);
        *reinterpret_cast<float2*>(Cout + (size_t)(row + 8) * N + col) = make_float2(c[j][2], c[j][3]);
      }
      if (Uout) {
        *reinterpret_cast<float2*>(Uout + (size_t)row * N + col) = make_float2(u[j][0], u[j][1]);
        *reinterpret_cast<float2*>(Uout + (size_t)(row + 8) * N + col) = make_float2(u[j][2], u[j][3]);
      }
    }
  }
}

#define GEMM(NAME, FMT, BM, BN, MODE)                                                                    \
  extern "C" __global__ void __launch_bounds__(Geo<FMT, BM, BN>::THREADS, 1) NAME(const Args a) {       \
    gemm_body<FMT, BM, BN, MODE>(a);                                                                  \
  }                                                                                                   \
  extern "C" __global__ void NAME##_smem(int* out) { *out = Geo<FMT, BM, BN>::SMEM; }
#define GEMMS(F, FMT)                                     \
  GEMM(gemm_##F##_hash, FMT, 128, 128, M_HASH)            \
  GEMM(gemm_##F##_nohash, FMT, 128, 128, M_NOHASH)        \
  GEMM(gemm_##F##_plain, FMT, 128, 128, M_PLAIN)          \
  GEMM(gemm_##F##_hash_m64, FMT, 64, 128, M_HASH)         \
  GEMM(gemm_##F##_nohash_m64, FMT, 64, 128, M_NOHASH)     \
  GEMM(gemm_##F##_plain_m64, FMT, 64, 128, M_PLAIN)       \
  GEMM(gemm_##F##_hash_m64n64, FMT, 64, 64, M_HASH)       \
  GEMM(gemm_##F##_nohash_m64n64, FMT, 64, 64, M_NOHASH)   \
  GEMM(gemm_##F##_plain_m64n64, FMT, 64, 64, M_PLAIN)
GEMMS(nv, NVFP4)
GEMMS(mx, MXFP4)

// tile leaves: TurboSHAKE128 of a 64 x 64 tile's 128 message digests (rows in order, 2 chunks each), 0x22.
// p0 digests, p1 out (tiles row-major x 32 B); i0 m, i1 n
extern "C" __global__ void __launch_bounds__(64) leaves(const Args a) {
  const int tn = a.i[1] / 64, per = a.i[1] / 32, t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] / 64 * tn) return;
  const int ti = t / tn, tj = t % tn;
  const uint64_t* dg = reinterpret_cast<const uint64_t*>(a.p[0]);
  uint64_t h[4];
  ph::pearlc::tile_leaf([&](int i) { return dg[((size_t)(64 * ti + i / 8) * per + 2 * tj + (i / 4) % 2) * 4 + i % 4]; }, h);
  ulonglong2* out = reinterpret_cast<ulonglong2*>(a.p[1] + (size_t)t * 32);
  out[0] = make_ulonglong2(h[0], h[1]);
  out[1] = make_ulonglong2(h[2], h[3]);
}
