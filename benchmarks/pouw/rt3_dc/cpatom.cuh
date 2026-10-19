// RT3-DC (note:20261019T1700Z-draft-rt3-dc): li's checkpoint by atomics, on the H200 (sm_90a) and the B200 (sm_100a).
// Included by sm90_cpatom.cu and sm100_cpatom.cu after their tensor probe (../dc/probes/sm90_wgmma.cu,
// sm100_tcgen05.cu), so Lane, the OP_ macros and dc.cuh are in scope.
//
// A checkpoint is the XOR of a warp's 32 lanes x 8 chain states after a 128-word segment (segment.warp_checkpoint).
// LI-CP-W takes it by 4 LOP3 (a lane's 8 chains), REDUX (the warp), the uniform-to-vector move and lane 0's STG; CP_SEG4
// keeps four segments' words and stores them in one STG.128. The forms here move the warp's XOR to an atomic unit:
//   CP_ATS   4 LOP3, then each lane's red.shared.xor into the segment's one shared word (ATOMS, 32 lanes into one
//            word); every four segments one LDS.128 of the four words, lane 0's STG.128 and an STS.128 that zeroes
//            them (the words double-buffered, one __syncwarp a trip orders the atomics, the read and the zero).
//   CP_ATS2  3 LOP3 (y0..y6) and 2 ATOMS (that word and y7): one ALU op traded for an atomic.
//   CP_ATS8  8 ATOMS, no LOP3.
//   CP_RED   4 LOP3, then each lane's red.global.xor into the segment's one global word (REDG; RT2-HOPPER's CP_RED).
//   CP_RED2, CP_RED8 as ATS2 and ATS8 on REDG.
// ptxas turns an atomic to an address it can prove warp-uniform into REDUX and one lane's atomic, so every address is
// base + (lane & zm), zm 0 at run time (the kernel's argument, or OpArgs::mode).
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- li, as segment._li_absorb (RT2-HOPPER's transcription; ref_check.py recomputes it through segment.py) --------
__host__ __device__ constexpr uint32_t li_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t li_a(int c, int h) { return li_mix32(0xA0000000u + 2u * (32 + 2 * c + h)) | 1u; }
__host__ __device__ constexpr uint32_t li_k(int c, int h) { return li_mix32(0xB0000000u + 2u * (32 + 2 * c + h) + 1u); }
__host__ __device__ constexpr int li_entry(int c, int i) { return 4 * (c / 4 + 2 * i) + c % 4; }

// One 32-word set into the 8 chains (p: the set's words).
__host__ __device__ inline void li_set(uint32_t (&y)[8], const uint32_t* p) {
#pragma unroll
  for (int c = 0; c < 8; c++) {
#pragma unroll
    for (int h = 0; h < 2; h++)
      y[c] = (y[c] ^ p[li_entry(c, 2 * h)] ^ li_k(c, h)) * li_a(c, h) + p[li_entry(c, 2 * h + 1)];
  }
}

// ---- the atomics ---------------------------------------------------------------------------------------------------
__device__ __forceinline__ void red_xor_g(uint32_t* p, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_xor_s(uint32_t saddr, uint32_t v) {
  asm volatile("red.relaxed.cta.shared::cta.xor.b32 [%0], %1;" ::"r"(saddr), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t saddr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(saddr) : "memory");
  return v;
}
__device__ __forceinline__ void sts128_zero(uint32_t saddr) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" ::"r"(saddr), "r"(0u) : "memory");
}
__device__ __forceinline__ void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint32_t lop3x(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// y0 ^ .. ^ y6 in 3 LOP3
__device__ __forceinline__ uint32_t xor7(const uint32_t (&y)[8]) {
  return lop3x(lop3x(lop3x(y[0], y[1], y[2]), y[3], y[4]), y[5], y[6]);
}
// y0 ^ .. ^ y7 in 4 LOP3 (LI-CP-W's)
__device__ __forceinline__ uint32_t xor8x(const uint32_t (&y)[8]) {
  uint32_t x = xor7(y);
  asm volatile("xor.b32 %0,%0,%1;" : "+r"(x) : "r"(y[7]));
  return x;
}

enum { CP_SEG4, CP_ATS, CP_ATS2, CP_ATS8, CP_RED, CP_RED2, CP_RED8, CP_N };
static const char* const CP_NAMES[CP_N] = {"seg4", "ats", "ats2", "ats8", "red", "red2", "red8"};
constexpr bool cp_shared(int cp) { return cp == CP_ATS || cp == CP_ATS2 || cp == CP_ATS8; }
constexpr bool cp_global(int cp) { return cp == CP_RED || cp == CP_RED2 || cp == CP_RED8; }
constexpr int CK_NONE = CP_N;  // the timed loops' no-checkpoint form: the chains alone

// One segment's checkpoint into slot j (0..3) of the trip: the shared form into the word at sbase + 4 j, the global
// form into gw[j], CP_SEG4 returned for the trip's STG.128. sbase and gw already carry (lane & zm).
template <int CP>
__device__ __forceinline__ uint32_t cp_seg(const uint32_t (&y)[8], uint32_t sbase, uint32_t* gw, int j) {
  if constexpr (CP == CP_SEG4) {
    uint32_t x = xor8x(y);
    asm volatile("redux.sync.xor.b32 %0,%0,0xffffffff;" : "+r"(x));
    return x;
  }
  if constexpr (CP == CP_ATS) red_xor_s(sbase + 4u * j, xor8x(y));
  if constexpr (CP == CP_ATS2) {
    red_xor_s(sbase + 4u * j, xor7(y));
    red_xor_s(sbase + 4u * j, y[7]);
  }
  if constexpr (CP == CP_ATS8) {
#pragma unroll
    for (int c = 0; c < 8; c++) red_xor_s(sbase + 4u * j, y[c]);
  }
  if constexpr (CP == CP_RED) red_xor_g(gw + j, xor8x(y));
  if constexpr (CP == CP_RED2) {
    red_xor_g(gw + j, xor7(y));
    red_xor_g(gw + j, y[7]);
  }
  if constexpr (CP == CP_RED8) {
#pragma unroll
    for (int c = 0; c < 8; c++) red_xor_g(gw + j, y[c]);
  }
  return 0;
}
// The trip's tail after its four segments: CP_SEG4 lane 0's STG.128 of the four words; the shared forms read the
// buffer's four words (one LDS.128, after a __syncwarp that orders every lane's atomics before it), lane 0 stores
// them, and the buffer is zeroed for the trip after next. dst: the trip's 16 bytes of checkpoints. Returns the words'
// XOR (the exact kernel's; the timed loops drop it, their stores are their sink).
template <int CP>
__device__ __forceinline__ uint32_t cp_tail(const uint32_t (&keep)[4], uint32_t sbuf, void* dst, int lane) {
  if constexpr (CP == CP_SEG4) {
    if (lane == 0) stg128(dst, make_uint4(keep[0], keep[1], keep[2], keep[3]));
    return 0;
  }
  if constexpr (cp_shared(CP)) {
    __syncwarp();
    const uint4 v = lds128(sbuf);
    if (lane == 0) stg128(dst, v);
    sts128_zero(sbuf);
    return v.x ^ v.y ^ v.z ^ v.w;
  }
  return 0;
}

// ---- exact: li bit for bit, each form against the host's reference ---------------------------------------------------
// One warp a block. words: U units x 32 lanes x 128 words (lane-major in a unit), st: 32 lanes x 8 starting states. out
// (zeroed): U checkpoints (one a 128-word segment), then each lane's final 8 states.
template <int CP>
__global__ void __launch_bounds__(32, 1) li_exact(const uint4* __restrict__ words, const uint32_t* st, uint32_t* out,
                                                 int U, uint32_t zm) {
  __shared__ __align__(16) uint32_t sh[8];
  const int lane = threadIdx.x;
  if (lane < 8) sh[lane] = 0;
  __syncwarp();
  const uint32_t s0 = smem_u32(sh) + 4u * (lane & zm);
  uint32_t* const gw0 = out + (lane & zm);
  uint32_t y[8];
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = st[8 * lane + c];
#pragma unroll 1
  for (int u4 = 0; u4 < U / 4; u4++) {
    const uint32_t sbuf = s0 + 16u * (u4 & 1);
    uint32_t keep[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int u = 4 * u4 + j;
      const uint4* src = words + ((size_t)u * 32 + lane) * 32;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        uint32_t p[32];
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const uint4 v = src[8 * s + q];
          p[4 * q] = v.x, p[4 * q + 1] = v.y, p[4 * q + 2] = v.z, p[4 * q + 3] = v.w;
        }
        li_set(y, p);
      }
      keep[j] = cp_seg<CP>(y, sbuf, gw0 + 4 * u4, j);
    }
    cp_tail<CP>(keep, sbuf - 4u * (lane & zm), reinterpret_cast<uint4*>(out) + u4, lane);
  }
#pragma unroll
  for (int c = 0; c < 8; c++) out[U + 8 * lane + c] = y[c];
}

static uint32_t cpa_lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 32);
}

template <int CP>
static void launch_exact(const uint32_t* dw, const uint32_t* dst, uint32_t* dout, int U) {
  li_exact<CP><<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
}

// The run's reference (the same LCG, seed and layout as RT2-HOPPER's, so its ref_check.py logic applies), then every
// form's words against it. A JSON line for the reference and one a form; exit 1 when any word differs.
static int run_exact(int U) {
  const size_t nw = (size_t)U * 32 * 128;
  std::vector<uint32_t> w(nw), st(256);
  uint64_t seed = 0x5EED2026ull;
  for (auto& x : w) x = cpa_lcg(seed);
  for (auto& x : st) x = cpa_lcg(seed);
  std::vector<uint32_t> cp(U), y(st);
  for (int u = 0; u < U; u++) {
    uint32_t x = 0;
    for (int lane = 0; lane < 32; lane++) {
      uint32_t yy[8];
      for (int c = 0; c < 8; c++) yy[c] = y[8 * lane + c];
      for (int s = 0; s < 4; s++) li_set(yy, &w[((size_t)u * 32 + lane) * 128 + 32 * s]);
      for (int c = 0; c < 8; c++) y[8 * lane + c] = yy[c], x ^= yy[c];
    }
    cp[u] = x;
  }
  auto list = [](const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
  };
  printf("{\"probe\": \"ref\", \"units\": %d, \"seed\": %llu, \"cp_seg\": %s, \"final\": %s}\n", U, 0x5EED2026ull,
         list(cp).c_str(), list(y).c_str());
  uint32_t *dw, *dst, *dout;
  cudaMalloc(&dw, nw * 4);
  cudaMalloc(&dst, 256 * 4);
  cudaMalloc(&dout, (U + 256) * 4);
  cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dst, st.data(), 256 * 4, cudaMemcpyHostToDevice);
  int bad = 0;
  for (int f = 0; f < CP_N; f++) {
    cudaMemset(dout, 0, (U + 256) * 4);
    switch (f) {
      case CP_SEG4: launch_exact<CP_SEG4>(dw, dst, dout, U); break;
      case CP_ATS: launch_exact<CP_ATS>(dw, dst, dout, U); break;
      case CP_ATS2: launch_exact<CP_ATS2>(dw, dst, dout, U); break;
      case CP_ATS8: launch_exact<CP_ATS8>(dw, dst, dout, U); break;
      case CP_RED: launch_exact<CP_RED>(dw, dst, dout, U); break;
      case CP_RED2: launch_exact<CP_RED2>(dw, dst, dout, U); break;
      case CP_RED8: launch_exact<CP_RED8>(dw, dst, dout, U); break;
    }
    const cudaError_t e = cudaDeviceSynchronize();
    std::vector<uint32_t> got(U + 256);
    cudaMemcpy(got.data(), dout, (U + 256) * 4, cudaMemcpyDeviceToHost);
    int mism = 0;
    for (int u = 0; u < U; u++) mism += got[u] != cp[u];
    for (int i = 0; i < 256; i++) mism += got[U + i] != y[i];
    printf("{\"probe\": \"exact\", \"kernel\": \"li_exact<%s>\", \"mismatches\": %d, \"words_checked\": %d, "
           "\"cuda\": \"%s\"}\n",
           CP_NAMES[f], mism, U + 256, e == cudaSuccess ? "no error" : cudaGetErrorString(e));
    bad += mism != 0 || e != cudaSuccess;
  }
  cudaFree(dw);
  cudaFree(dst);
  cudaFree(dout);
  fflush(stdout);
  return bad;
}

// ---- the timed loops' trip ---------------------------------------------------------------------------------------------
// Every timed loop below runs two trips an iteration, the first into the warp's shared buffer A and global words
// wi[0..3], the second into buffer B and wi[4..7], then wi += 8: the buffers' parity is static and the global words are a
// run of fresh ones, so the loop holds no address arithmetic but the pointer's add (no LOP3 or IMAD of its own; the gate's
// LOP3 and IMAD counts are the fold's and the checkpoint's). A trip is 4 segments, each `fold()` (one segment's 64 LOP3 and
// 64 IMAD a lane: 4096 words a warp, a unit) and its checkpoint, then the trip's tail. a: the buffer's atomic address (with
// lane & zm), t: its lane-free base (the tail's LDS and STS).
template <int CP, class Fold>
__device__ __forceinline__ void cp_trip(Lane& l, Fold&& fold, uint32_t a, uint32_t t, uint32_t* wi, int lane) {
  uint32_t keep[4] = {0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    fold();
    if constexpr (CP != CK_NONE) keep[j] = cp_seg<CP>(l.r, a, wi, j);
  }
  if constexpr (CP != CK_NONE) cp_tail<CP>(keep, t, wi, lane);
}
// A warp's run of global words: 16 a loop iteration of `iters` (twice what a fixed loop writes, so an until loop that
// runs up to 2 iters trips stays in its own), every warp of every block its own.
__device__ __forceinline__ uint32_t* cp_words(const OpArgs& a, int warps_a_block, int warp, int lane) {
  return reinterpret_cast<uint32_t*>(a.g) + ((size_t)blockIdx.x * warps_a_block + warp) * (16ull * a.iters) +
         (lane & (uint32_t)a.mode);
}

// ---- lane alone: the fold with each checkpoint form, every SM, as opcode probes ---------------------------------------
// W warps a block, one block an SM (W/4 a SMSP), li_fold's lane ops on the 8 chains. The block's cycles over iters x 8
// units x W/4 warps a SMSP is a unit's SMSP-cycles a warp-slot: max(issue, ALU) a unit on a loop bound by either; an
// atomic that costs more than its issue slot shows above that.
template <int CP, int W>
DC_DEV void lr_body(const OpArgs& a) {
  __shared__ __align__(16) uint32_t sh[W][8];
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const uint32_t zm = (uint32_t)a.mode;
  if (lane < 8) sh[wp][lane] = 0;
  __syncwarp();
  const uint32_t t0 = smem_u32(&sh[wp][0]), a0 = t0 + 4u * (lane & zm);
  uint32_t* wi = cp_words(a, W, wp, lane);
  auto fold = [&] {
#pragma unroll
    for (int c = 0; c < 64; c++) {
      l.op<L_LOP3>(c);
      l.op<L_IMAD>(c);
    }
  };
  OP_BEGIN
  OP_LOOP(it) {
    cp_trip<CP>(l, fold, a0, t0, wi, lane);
    cp_trip<CP>(l, fold, a0 + 16u, t0 + 16u, wi + 4, lane);
    wi += 8;
  }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

// ---- the atomic units alone: R atomics an iteration, a warp's 32 lanes into one word (or 32 words) --------------------
// at_s1 / at_g1: every lane into the warp's one word (the checkpoint's pattern), shared (ATOMS) or global (REDG);
// at_s32 / at_g32: lane l into word l (no address conflict), so the unit's lane rate without serialization. The value
// is the lane's chain state, which the LOP3 before each atomic advances (so ptxas keeps every atomic and its operand).
// The global words are 8 a warp (32 x 8 for g32), the same every iteration: the rate of one checkpoint word's atomics.
template <bool SHARED, bool ONE, int R, int W>
DC_DEV void at_body(const OpArgs& a) {
  __shared__ __align__(16) uint32_t sh[W][32];
  Lane l(threadIdx.x);
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const uint32_t zm = (uint32_t)a.mode, off = ONE ? (lane & zm) : (uint32_t)lane;
  sh[wp][lane] = 0;
  __syncwarp();
  const uint32_t s0 = smem_u32(&sh[wp][0]) + 4u * off;
  uint32_t* const g0 = reinterpret_cast<uint32_t*>(a.g) + ((size_t)blockIdx.x * W + wp) * 1024 + off;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < R; q++) {
      l.op<L_LOP3>(q);
      if constexpr (SHARED) red_xor_s(s0, l.r[q & 7]);
      else red_xor_g(g0 + 32 * (q & 7), l.r[q & 7]);
    }
  }
  OP_END
  __syncthreads();
  OP_SINK(l.fold() ^ (SHARED ? sh[wp][lane] : g0[0]) ^ threadIdx.x)
}

#define CPA_LR(NAME, CP, W) OP_KERNEL(NAME, W * 32) { lr_body<CP, W>(a); }
#define CPA_AT(NAME, SH, ONE, R, W) OP_KERNEL(NAME, W * 32) { at_body<SH, ONE, R, W>(a); }
CPA_LR(lr_none, CK_NONE, 16)
CPA_LR(lr_seg4, CP_SEG4, 16)
CPA_LR(lr_ats, CP_ATS, 16)
CPA_LR(lr_ats2, CP_ATS2, 16)
CPA_LR(lr_ats8, CP_ATS8, 16)
CPA_LR(lr_red, CP_RED, 16)
CPA_LR(lr_red2, CP_RED2, 16)
CPA_LR(lr_red8, CP_RED8, 16)
CPA_LR(lr8_seg4, CP_SEG4, 8)
CPA_LR(lr8_ats, CP_ATS, 8)
CPA_LR(lr8_red, CP_RED, 8)
CPA_AT(at_s1, true, true, 16, 16)
CPA_AT(at_s32, true, false, 16, 16)
CPA_AT(at_g1, false, true, 16, 16)
CPA_AT(at_g32, false, false, 16, 16)
CPA_AT(at4_s1, true, true, 16, 4)
CPA_AT(at4_g1, false, true, 16, 4)

// A timed loop's opcode counts an iteration (8 units). The chains are 512 LOP3 + 512 IMAD; a checkpoint adds 4 LOP3 (3
// for the 2-atomic forms, 0 for the 8-atomic ones) and its atomics, and each trip's tail its stores. CPA_CK(form) is the
// checkpoint's part, after the chains' (the arch files' beside-tensor loops add the tensor's).
#define CPA_CK_none "LOP3.LUT=512,IMAD=512"
#define CPA_CK_seg4 "LOP3.LUT=544,IMAD=512,REDUX*=8,STG.E.128*=2"
#define CPA_CK_ats "LOP3.LUT=544,IMAD=512,ATOMS*=8,LDS.128=2,STS.128=2,STG.E.128*=2"
#define CPA_CK_ats2 "LOP3.LUT=536,IMAD=512,ATOMS*=16,LDS.128=2,STS.128=2,STG.E.128*=2"
#define CPA_CK_ats8 "LOP3.LUT=512,IMAD=512,ATOMS*=64,LDS.128=2,STS.128=2,STG.E.128*=2"
#define CPA_CK_red "LOP3.LUT=544,IMAD=512,REDG*=8"
#define CPA_CK_red2 "LOP3.LUT=536,IMAD=512,REDG*=16"
#define CPA_CK_red8 "LOP3.LUT=512,IMAD=512,REDG*=64"
#define CPA_P(NAME, W, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = W * 32, .expect = PAT, .iters = 2048}
#define CPA_LR_PROBES                                                                                              \
  CPA_P(lr_none, 16, CPA_CK_none), CPA_P(lr_seg4, 16, CPA_CK_seg4), CPA_P(lr_ats, 16, CPA_CK_ats),                  \
  CPA_P(lr_ats2, 16, CPA_CK_ats2), CPA_P(lr_ats8, 16, CPA_CK_ats8), CPA_P(lr_red, 16, CPA_CK_red),                  \
  CPA_P(lr_red2, 16, CPA_CK_red2), CPA_P(lr_red8, 16, CPA_CK_red8), CPA_P(lr8_seg4, 8, CPA_CK_seg4),                \
  CPA_P(lr8_ats, 8, CPA_CK_ats), CPA_P(lr8_red, 8, CPA_CK_red), CPA_P(at_s1, 16, "LOP3.LUT=16,ATOMS*=16"),          \
  CPA_P(at_s32, 16, "LOP3.LUT=16,ATOMS*=16"), CPA_P(at_g1, 16, "LOP3.LUT=16,REDG*=16"),                             \
  CPA_P(at_g32, 16, "LOP3.LUT=16,REDG*=16"), CPA_P(at4_s1, 4, "LOP3.LUT=16,ATOMS*=16"),                             \
  CPA_P(at4_g1, 4, "LOP3.LUT=16,REDG*=16")
