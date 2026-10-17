// pgemm: the k4s Dither GEMM pipelined on datacenter Blackwell (sm_100a), against native int8 Dither and plain FP8 and
// int8 in the same pipeline, and cuBLASLt.
//
// The product and the receipt are dgemm's (../dc/dgemm.cu, header) with the commit spacing L (32 or 64) and the receipt
// tile width BNR as parameters: C = X W^T over s8 codes; every L steps of k the s32 running sum T_s is committed; chain
// (i, tr, c) of receipt tile tr (columns BNR tr ..) owns the columns j of the tile with j % 16 == c, starts at
// duo_init(seed, (i tiles + tr) 16 + c) and absorbs, at each s = 1 .. k / L in ascending j, word T_s[i, j] at key
// (j / 16) % 16; its receipt word is duo_finish.
//
// Pipeline (one CTA a 128 x N output tile, N = FW BNR = 128; 64 + 128 FW threads, one CTA a SM):
//   warp 0       the loader (and TMEM's allocator): cp.async.bulk of each stage's operand tiles (KS = 64 bytes of k,
//                every plane of both sides) into an S-stage shared-memory ring, on full/empty mbarriers. The operands are
//                stored pre-tiled (tile_planes: each stage tile contiguous in dc.cuh's K-major no-swizzle layout).
//   warp 1       the MMA issuer: each word's tcgen05.mma into TMEM buffer w % NB, committed to tfull[b]; a stage's
//                slot is released (tcgen05.commit to empty) once its MMAs are issued
//   warps 2 ..   the fold warps (warpgroup g owns columns BNR g .., receipt tile FW tn + g; warp w reads TMEM lanes
//                32 (w % 4) ..): wait tfull[b], read the word 32 columns at a time (tcgen05.ld.32x32b.x32, the next
//                chunk's load in flight while this one is folded), extract, fold, release the buffer (tempty[b])
// Routes R:
//   K4S  egemm's k4s: planes [16h, l] on both sides, 4 L / 32 kind::f8f6f4 E4M3 MMAs a word onto one FP32 accumulator;
//        the fold warps restore it to 1.5 2^23 (tcgen05.st) as they read it; word T += bits - 0x4B400000 (one IADD3)
//   I8C  native int8, the running sum in TMEM: the fold warps read word w from buffer b, copy it into buffer b + 1
//        (tcgen05.st), onto which word w + 1's MMAs accumulate (released by tempty[b + 1]), then fold it from registers
//        (duo alone a word)
//   I8R  native int8 from a fresh accumulator each word, T += r in registers (k4s's shape with int8's MMA)
//   PF8, PI8  plain: one kind::f8f6f4 E4M3 (PF8, operand l planes) or kind::i8 (PI8) accumulator over all of k,
//        written at the end; no fold
//   K4P  k4s with the offset receipt: each word absorbed as S_w = T_w + (w + 1) 0x4B400000 (mod 2^32), a public
//        per-word offset, so the extraction is S += bits, a two-input add that ptxas may put on the IMAD pipe
//        (IMAD.IADD) beside duo's LOP3 and LEA.HI on the ALU pipe; C is T, as for every route
//   K4B  k4s's receipt with K4P's running sum: t += bits, then the word T_w = t - (w + 1) 0x4B400000 (a second
//        two-input add, of a per-word uniform) into duo; two adds a word that ptxas may split between ALU and IMAD
//   K4T  k4s with the running sum in TMEM, as int8 Dither carries it: the CTA is 128 x 256, two 128-column tiles, one
//        fold warpgroup each (receipt tile 128, k4s's FW 1 definition). Tile t's accumulator (columns 128 t) holds
//        1.5 2^23 + T - T_base and carries from word to word (the MMAs of word w + 1 accumulate onto word w's sum once
//        the tile's warpgroup has read it, tempty[t]); base' = T_base - 0x4B400000 sits in the other half (columns
//        256 + 128 t). The word is p = bits + base' (a two-input add, the IMAD pipe's), receipt unchanged; every RB
//        words the warpgroup rebases (base' = p - 0x4B400000, the accumulator restored to 1.5 2^23), RB the most words
//        whose k32 steps keep |T - T_base| within the binade's 2^22 at every partial sum
// Fold-warp modes M (the time split): FULL (the kernel of record), EXT (read and extract, no duo; C written), LD (read,
// and K4S's restore or I8C's copy, nothing computed), MMAONLY (wait for each word and release it, no read: the tensor
// and the operand pipeline alone), FOLDONLY (no tensor, no TMEM: the fold warps extract and fold register words, as
// many as FULL absorbs), and MMARES and FULLRES, MMAONLY and FULL with the operands resident in shared memory (the
// tensor without the operand feed; the kernel as a perfect feed would leave it). pf8_res is plain FP8 so.
//
// Modes:
//   pgemm selftest                   host only: k4s's plane identity on every pair, E4M3 round trip, tile_planes'
//                                    inverse
//   pgemm gate M N K SEED DIST       DIST uniform, edge (limb extremes) or max (every code -128, k4t's accumulator at its
//                                    bound): every route, L and FW of record (FULL; k4t when N is a multiple of 256), and
//                                    EXT on C, against the host reference, poisoned with 0xA5, with the no-write control
//                                    (must be rejected); plain PF8 / PI8 on C; one JSON line each; exit 4 unless all hold
//   pgemm bench REPS M N K           every arm interleaved (cuBLASLt int8 and FP8 with -DPG_CUBLASLT), median ms, and
//                                    each pgemm arm's SM clock under its own load (clock64 over globaltimer in every
//                                    CTA) and its median CTA clocks (one CTA a SM: DC's pipelined int8 kernel's rate
//                                    metric), one JSON line
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a pgemm.cu [-DPG_CUBLASLT -lcublasLt]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../dc/dc.cuh"
#ifdef PG_CUBLASLT
#include <cublasLt.h>
#endif

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

constexpr int NCH = 16;
constexpr int BM = 128;
constexpr int N = 128;   // CTA tile width (tcgen05 M = 128, N = 128: full tensor rate)
constexpr int KS = 64;   // bytes of k a stage: two k32 slices
constexpr int SMEM_RING = 192 * 1024;
constexpr uint32_t MAGIC_BITS = 0x4B400000u;

enum Route { K4S = 0, I8C = 1, I8R = 2, PF8 = 3, PI8 = 4, K4P = 5, K4B = 6, K4T = 7 };
enum Mode { FULL = 0, EXT = 1, LD = 2, MMAONLY = 3, FOLDONLY = 4, MMARES = 5, FULLRES = 6 };
// MMARES and FULLRES are MMAONLY and FULL with resident operands: the loader fills the ring once and then completes each
// stage's barrier without a copy, so the MMAs reread stale tiles (timing only: C and the receipts are wrong).
DC_HD constexpr int fold_mode(int m) { return m == MMARES ? MMAONLY : m == FULLRES ? FULL : m; }
DC_HD constexpr bool resident(int m) { return m == MMARES || m == FULLRES; }
DC_HD constexpr bool is_k4(int r) { return r == K4S || r == K4P || r == K4B; }  // the word buffers restored each word
DC_HD constexpr bool is_f8x4(int r) { return is_k4(r) || r == K4T; }             // k4s's planes and MMAs
DC_HD constexpr int planes_of(int r) { return is_f8x4(r) ? 2 : 1; }
DC_HD constexpr bool is_plain(int r) { return r == PF8 || r == PI8; }
DC_HD constexpr int nbuf_of(int r) { return is_plain(r) ? 1 : r == I8C || r == K4T ? 2 : 4; }
DC_HD constexpr int cta_n(int r) { return r == K4T ? 2 * N : N; }
DC_HD constexpr int stage_bytes(int r) { return planes_of(r) * (BM + cta_n(r)) * KS; }
DC_HD constexpr int stages_of(int r) { return SMEM_RING / stage_bytes(r) < 8 ? SMEM_RING / stage_bytes(r) : 8; }

DC_HD int limb_h(int x) { return (x + 8) >> 4; }
DC_HD int limb_l(int x) { return x - 16 * limb_h(x); }
DC_HD uint8_t e4m3_int(int v) {
  if (v == 0) return 0;
  const uint8_t sg = v < 0 ? 0x80 : 0;
  const int a = v < 0 ? -v : v;
  int e = 0;
  while ((a >> (e + 1)) != 0) e++;
  const int mi = ((a << 3) >> e) - 8;
  return sg | (uint8_t)((e + 7) << 3) | (uint8_t)mi;
}

// ---- device helpers ---------------------------------------------------------------------------------------------------
DC_DEV void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(smem_u32(bar)) : "memory");
}
DC_DEV void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
DC_DEV void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
DC_DEV void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
DC_DEV void tmem_st32c(uint32_t taddr, uint32_t v) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,"
      "%1,%1,%1,%1,%1,%1,%1,%1,%1,%1,%1};" ::"r"(taddr),
      "r"(v)
      : "memory");
}
// After tcgen05.wait::ld: the registers a load wrote, pinned so no use moves above the wait.
template <int K>
DC_DEV void pin(uint32_t (&r)[K]) {
#pragma unroll
  for (int i = 0; i < K; i++) asm volatile("" : "+r"(r[i]));
}
#define PG_ARR32(p) (*reinterpret_cast<uint32_t(*)[32]>(p))

// Chunk U (32 columns of the receipt tile, keys 2U and 2U + 1) of a word into the chains.
template <int R, int M, int U, int BNR>
DC_DEV void fold_chunk(DuoSt (&ch)[NCH], uint32_t (&t)[BNR], const uint32_t* r, uint32_t cw) {
  if constexpr (M == FULL || M == EXT || M == FOLDONLY) {
#pragma unroll
    for (int c = 0; c < 32; c++) {
      uint32_t p;
      if constexpr (R == K4S) {
        t[32 * U + c] += r[c] - MAGIC_BITS;
        p = t[32 * U + c];
      } else if constexpr (R == I8R || R == K4P) {
        t[32 * U + c] += r[c];
        p = t[32 * U + c];
      } else if constexpr (R == K4B) {
        t[32 * U + c] += r[c];
        p = t[32 * U + c] + cw;
      } else {
        p = r[c];
      }
      if constexpr (M != EXT) {
        if (c < 16) duo<(2 * U) % 16>(ch[c], p);
        else duo<(2 * U + 1) % 16>(ch[c - 16], p);
      }
    }
  }
}
// K4S / I8R: chunk U's load is in ra; chunk U + 1's goes to rb while U is restored and folded. c0 holds 1.5 2^23 in 32
// registers of its own for the whole kernel, loaded from memory: ptxas rematerializes a constant operand (32 moves a
// store). LD mode keeps one word of each chunk (x), or ptxas drops the load.
template <int R, int M, int U, int BNR>
DC_DEV void word_chunks(DuoSt (&ch)[NCH], uint32_t (&t)[BNR], uint32_t (&ra)[32], uint32_t (&rb)[32], uint32_t tsrc,
                        const uint32_t (&c0)[32], uint32_t& x, uint32_t cw) {
  if constexpr (U < BNR / 32) {
    tmem_wait_ld();
    pin(ra);
    if constexpr (is_k4(R)) tmem_st32(tsrc + 32 * U, c0);
    if constexpr (U + 1 < BNR / 32) tmem_ld32(tsrc + 32 * (U + 1), rb);
    if constexpr (M == LD) x ^= ra[0];
    fold_chunk<R, M, U, BNR>(ch, t, ra, cw);
    word_chunks<R, M, U + 1, BNR>(ch, t, rb, ra, tsrc, c0, x, cw);
  }
}
template <int U, int BNR>
DC_DEV void ld_all(uint32_t* v, uint32_t tsrc) {
  if constexpr (U < BNR / 32) {
    tmem_ld32(tsrc + 32 * U, PG_ARR32(v + 32 * U));
    ld_all<U + 1, BNR>(v, tsrc);
  }
}
template <int U, int BNR>
DC_DEV void st_all(uint32_t* v, uint32_t tdst) {
  if constexpr (U < BNR / 32) {
    tmem_st32(tdst + 32 * U, PG_ARR32(v + 32 * U));
    st_all<U + 1, BNR>(v, tdst);
  }
}
template <int R, int M, int U, int BNR>
DC_DEV void fold_all(DuoSt (&ch)[NCH], uint32_t (&t)[BNR], const uint32_t* v, uint32_t cw) {
  if constexpr (U < BNR / 32) {
    fold_chunk<R, M, U, BNR>(ch, t, v + 32 * U, cw);
    fold_all<R, M, U + 1, BNR>(ch, t, v, cw);
  }
}

// K4T: chunk U (16 columns, key U) of a tile's word. xa/ba hold its accumulator bits and base' (loaded), xb/bb take
// chunk U + 1's while U is extracted and folded. On a rebase word the chunk's p - 0x4B400000 becomes base' and the
// accumulator is restored to 1.5 2^23. LD mode keeps a word of each chunk (x) and stores what a rebase would.
template <int M, int U, bool REB>
DC_DEV void k4t_chunks(DuoSt (&ch)[NCH], uint32_t (&xa)[16], uint32_t (&ba)[16], uint32_t (&xb)[16],
                       uint32_t (&bb)[16], uint32_t tacc, uint32_t tbas, const uint32_t (&c0)[16], uint32_t one,
                       uint32_t& x) {
  if constexpr (U < N / 16) {
    tmem_wait_ld();
    pin(xa);
    pin(ba);
    if constexpr (U + 1 < N / 16) {
      tmem_ld16(tacc + 16 * (U + 1), xb);
      tmem_ld16(tbas + 16 * (U + 1), bb);
    }
    if constexpr (M == LD) {
      x ^= xa[0] ^ ba[0];
      if constexpr (REB) {
        tmem_st16(tbas + 16 * U, ba);
        tmem_st16(tacc + 16 * U, c0);
      }
    } else {
      uint32_t p[16];
#pragma unroll
      for (int c = 0; c < 16; c++) {
        p[c] = xa[c] * one + ba[c];
        if constexpr (M == EXT) asm volatile("" ::"r"(p[c]));
        else duo<U % 16>(ch[c], p[c]);
      }
      if constexpr (REB) {
#pragma unroll
        for (int c = 0; c < 16; c++) p[c] -= MAGIC_BITS;
        tmem_st16(tbas + 16 * U, p);
        tmem_st16(tacc + 16 * U, c0);
      }
    }
    k4t_chunks<M, U + 1, REB>(ch, xb, bb, xa, ba, tacc, tbas, c0, one, x);
  }
}
// K4T: one word of a tile (MMAONLY: wait and release only); REB stores the rebase and waits for it before the release.
template <int M, bool REB>
DC_DEV void k4t_word(DuoSt (&ch)[NCH], uint32_t (&xa)[16], uint32_t (&ba)[16], uint32_t (&xb)[16], uint32_t (&bb)[16],
                     uint32_t tacc, uint32_t tbas, const uint32_t (&c0)[16], uint32_t one, uint32_t& x,
                     uint64_t* tfull, uint64_t* tempty, int wd, uint32_t lane) {
  mbar_wait(tfull, wd & 1);
  tc_after_sync();
  if constexpr (M != MMAONLY) {
    tmem_ld16(tacc, xa);
    tmem_ld16(tbas, ba);
    k4t_chunks<M, 0, REB>(ch, xa, ba, xb, bb, tacc, tbas, c0, one, x);
    if constexpr (REB) tmem_wait_st();
  }
  tc_before_sync();
  __syncwarp();
  if (lane == 0) mbar_arrive(tempty);
}
// K4T FOLDONLY: the same lane work on register words. Each chunk's words feed the next chunk's (v = p), or ptxas, which
// doesn't see pin's empty asm, would compute one chunk's sums for all eight.
template <int U, bool REB>
DC_DEV void k4t_regs(DuoSt (&ch)[NCH], uint32_t (&v)[16], uint32_t (&b)[16], uint32_t one) {
  if constexpr (U < N / 16) {
    pin(v);
    pin(b);
    uint32_t p[16];
#pragma unroll
    for (int c = 0; c < 16; c++) {
      p[c] = v[c] * one + b[c];
      duo<U % 16>(ch[c], p[c]);
      v[c] = p[c];
    }
    if constexpr (REB) {
#pragma unroll
      for (int c = 0; c < 16; c++) b[c] = p[c] - MAGIC_BITS;
    }
    k4t_regs<U + 1, REB>(ch, v, b, one);
  }
}

template <int R, int MK, int L, int FW>
__global__ void __launch_bounds__(64 + 128 * FW, 1)
    pg100(const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed, int32_t* C, uint32_t* Rc,
          int write, const uint32_t* magic, uint64_t* clk) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int M = fold_mode(MK);
  __shared__ uint64_t clk0[2];  // in shared memory: registers held across the kernel cost the k4 routes spills
  if (clk && threadIdx.x == 0) {
    uint64_t g0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    clk0[0] = clock64();
    clk0[1] = g0;
  }
  constexpr int P = planes_of(R), NB = nbuf_of(R), S = stages_of(R), J = KS / 32, BNR = R == K4T ? N : N / FW;
  constexpr uint32_t TCOLS = R == K4T ? 4 * N : NB * N < 32 ? 32 : NB * N;
  static_assert(BNR % 32 == 0 && (L == 32 || L == 64) && KS % L == 0, "shape");
  static_assert(R != K4T || FW == 2, "K4T: one fold warpgroup a tile");
  extern __shared__ __align__(1024) unsigned char ring[];
  __shared__ uint64_t full[S], empty[S], tfull[NB], tempty[NB];
  __shared__ uint32_t tslot;
  const int tn = blockIdx.x, tm = blockIdx.y;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int words = k / L, kst = k / KS;
  if (threadIdx.x == 0) {
    for (int s = 0; s < S; s++) mbar_init(&full[s], 1), mbar_init(&empty[s], 1);
    for (int b = 0; b < NB; b++) mbar_init(&tfull[b], 1), mbar_init(&tempty[b], R == K4T ? 4 : 4 * FW);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(&tslot, TCOLS);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot;
  const uint32_t ring0 = smem_u32(ring);
  const uint32_t q = w % 4, grp = (w - 2) / 4;
  const uint32_t tbase = tmem + ((32u * q) << 16) + grp * BNR;
  if constexpr (is_k4(R) && M != FOLDONLY) {
    if (w >= 2) {
#pragma unroll
      for (int b = 0; b < NB; b++)
#pragma unroll
        for (int u = 0; u < BNR / 32; u++) tmem_st32c(tbase + b * N + 32 * u, MAGIC_BITS);
      tmem_wait_st();
    }
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
  }
  if constexpr (R == K4T && M != FOLDONLY) {  // accumulators 1.5 2^23, base' = 0 - 0x4B400000 (T_base = 0)
    if (w >= 2) {
#pragma unroll
      for (int u = 0; u < N / 32; u++) {
        tmem_st32c(tbase + 32 * u, MAGIC_BITS);
        tmem_st32c(tbase + 2 * N + 32 * u, 0u - MAGIC_BITS);
      }
      tmem_wait_st();
    }
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
  }
  constexpr uint32_t idesc = is_f8x4(R) || R == PF8 ? umma_idesc(1, 0, 0, BM, N) : umma_idesc(2, 1, 1, BM, N);

  if (w == 0) {
    if constexpr (M != FOLDONLY) {
      const size_t a_tile = (size_t)BM * KS, b_tile = (size_t)N * KS;
      for (int st = 0; st < kst; st++) {
        const int slot = st % S;
        if (st >= S) mbar_wait(&empty[slot], ((st / S) & 1) ^ 1);
        if (elect_one()) {
          if (resident(MK) && st >= S) {
            mbar_arrive(&full[slot]);
          } else {
            mbar_expect_tx(&full[slot], stage_bytes(R));
            const uint32_t dst = ring0 + slot * stage_bytes(R);
            constexpr int NT = cta_n(R) / N;  // B tiles a stage: [plane][tile]
#pragma unroll
            for (int p = 0; p < P; p++) {
              bulk_g2s(dst + p * a_tile, A + (((size_t)p * (m / BM) + tm) * kst + st) * a_tile, a_tile, &full[slot]);
#pragma unroll
              for (int t = 0; t < NT; t++)
                bulk_g2s(dst + P * a_tile + (p * NT + t) * b_tile,
                         B + (((size_t)p * (n / N) + NT * tn + t) * kst + st) * b_tile, b_tile, &full[slot]);
            }
          }
        }
        __syncwarp();
      }
    }
  } else if (w == 1) {
    if constexpr (M != FOLDONLY) {
      constexpr uint32_t sbo = kmaj_sbo(KS);
      auto desc = [&](int slot, int side, int p, int js) {
        const uint32_t off = slot * stage_bytes(R) + (side ? P * BM * KS + p * N * KS : p * BM * KS) + js * 256;
        return umma_desc(ring0 + off, kmaj_lbo(), sbo);
      };
      int g = 0;
      const int wl = is_plain(R) ? 1 : R == K4T ? 0 : words, per = is_plain(R) ? k / 32 : L / 32;
      if constexpr (R == K4T) {  // word wd: tile 0 then tile 1, each after its fold warpgroup released word wd - 1
        auto descb = [&](int slot, int t, int p, int js) {
          const uint32_t off = slot * stage_bytes(R) + P * BM * KS + (p * 2 + t) * N * KS + js * 256;
          return umma_desc(ring0 + off, kmaj_lbo(), sbo);
        };
        for (int wd = 0; wd < words; wd++)
          for (int t = 0; t < 2; t++) {
            if (wd > 0) mbar_wait(&tempty[t], (wd - 1) & 1);
            tc_after_sync();
            for (int j = 0; j < per; j++) {
              const int gg = wd * per + j, st = gg / J, js = gg % J, slot = st % S;
              if (t == 0 && js == 0) {
                mbar_wait(&full[slot], (st / S) & 1);
                tc_after_sync();
              }
              if (elect_one()) {
#pragma unroll
                for (int pa = 0; pa < 2; pa++)
#pragma unroll
                  for (int pb = 0; pb < 2; pb++)
                    umma_f8f6f4(tmem + t * N, desc(slot, 0, pa, js), descb(slot, t, pb, js), idesc, 1);
                if (t == 1 && js == J - 1) umma_commit(&empty[slot]);
              }
              __syncwarp();
            }
            if (elect_one()) umma_commit(&tfull[t]);
            __syncwarp();
          }
      }
      for (int wd = 0; wd < wl; wd++) {
        const int b = wd % NB;
        if constexpr (R == I8C) {
          if (wd > 0) mbar_wait(&tempty[b], ((wd - 1) / NB) & 1);
        } else if constexpr (!is_plain(R)) {
          if (wd >= NB) mbar_wait(&tempty[b], ((wd / NB) & 1) ^ 1);
        }
        tc_after_sync();
        const uint32_t tb = tmem + b * N;
        for (int j = 0; j < per; j++, g++) {
          const int st = g / J, js = g % J, slot = st % S;
          if (js == 0) {
            mbar_wait(&full[slot], (st / S) & 1);
            tc_after_sync();
          }
          if (elect_one()) {
            if constexpr (is_k4(R)) {
#pragma unroll
              for (int pa = 0; pa < 2; pa++)
#pragma unroll
                for (int pb = 0; pb < 2; pb++) umma_f8f6f4(tb, desc(slot, 0, pa, js), desc(slot, 1, pb, js), idesc, 1);
            } else if constexpr (R == PF8) {
              umma_f8f6f4(tb, desc(slot, 0, 0, js), desc(slot, 1, 0, js), idesc, g > 0);
            } else if constexpr (R == PI8) {
              umma_i8(tb, desc(slot, 0, 0, js), desc(slot, 1, 0, js), idesc, g > 0);
            } else if constexpr (R == I8C) {
              umma_i8(tb, desc(slot, 0, 0, js), desc(slot, 1, 0, js), idesc, g > 0);
            } else {
              umma_i8(tb, desc(slot, 0, 0, js), desc(slot, 1, 0, js), idesc, j > 0);
            }
            if (js == J - 1) umma_commit(&empty[slot]);
          }
          __syncwarp();
        }
        if (elect_one()) umma_commit(&tfull[b]);
        __syncwarp();
      }
    }
  } else {
    const int row = tm * BM + 32 * q + lane;
    const int tiles_r = n / BNR, tr = tn * FW + grp;
    int32_t* crow = C + (size_t)row * n + tn * cta_n(R) + grp * BNR;
    if constexpr (is_plain(R)) {
      mbar_wait(&tfull[0], 0);
      tc_after_sync();
      if (write) {
#pragma unroll
        for (int u = 0; u < BNR / 32; u++) {
          uint32_t r[32];
          tmem_ld32(tbase + 32 * u, r);
          tmem_wait_ld();
          pin(r);
#pragma unroll
          for (int c = 0; c < 32; c += 4)
            *reinterpret_cast<int4*>(crow + 32 * u + c) = make_int4((int)r[c], (int)r[c + 1], (int)r[c + 2], (int)r[c + 3]);
        }
      }
    } else {
      DuoSt ch[NCH];
#pragma unroll
      for (int c = 0; c < NCH; c++) ch[c] = duo_init(seed, (uint32_t)((row * tiles_r + tr) * NCH + c));
      uint32_t t[BNR];
#pragma unroll
      for (int i = 0; i < BNR; i++) t[i] = 0;
      uint32_t x = 0, cw = 0;  // cw: K4B's offset removal, -(w + 1) 1.5 2^23 at word w
      if constexpr (R == K4T) {
        // A k32 moves the accumulator by at most 32 (128 128 + 2 128 8 + 8 8) = 2^19 + 2^16 + 2^11, so RB words keep
        // |T - T_base| < 2^22: the accumulator stays in [2^23, 2^24), where FP32's ulp is 1.
        constexpr int RB = L == 32 ? 7 : 3;
        static_assert(RB * (L / 32) * ((1 << 19) + (1 << 16) + (1 << 11)) < (1 << 22), "K4T rebase spacing");
        // Words RB - 1, 2 RB - 1, ... rebase; the inner loops are the plain words. The word is bits one + base' with
        // one (= 1) read from memory, so ptxas can't make it an IADD3: an IMAD, on the IMAD pipe, every time.
        const uint32_t tacc = tbase, tbas = tbase + 2 * N, one = __ldg(magic + 32);
        if constexpr (M == FOLDONLY) {
          uint32_t v[16], b[16];
#pragma unroll
          for (int c = 0; c < 16; c++) {
            asm volatile("mov.b32 %0, %1;" : "=r"(v[c]) : "r"(seed + 977u * c + row));
            asm volatile("mov.b32 %0, %1;" : "=r"(b[c]) : "r"(seed ^ (31u * c + row)));
          }
#pragma unroll 1
          for (int wd = 0; wd < words;) {
#pragma unroll 1
            for (int i = 0; i < RB - 1 && wd < words; i++, wd++) k4t_regs<0, false>(ch, v, b, one);
            if (wd < words) k4t_regs<0, true>(ch, v, b, one), wd++;
          }
        } else {
          uint32_t xa[16], ba[16], xb[16], bb[16], c0[16];
#pragma unroll
          for (int c = 0; c < 16; c++) c0[c] = __ldg(magic + c);
#pragma unroll 1
          for (int wd = 0; wd < words;) {
#pragma unroll 1
            for (int i = 0; i < RB - 1 && wd < words; i++, wd++)
              k4t_word<M, false>(ch, xa, ba, xb, bb, tacc, tbas, c0, one, x, &tfull[grp], &tempty[grp], wd, lane);
            if (wd < words)
              k4t_word<M, true>(ch, xa, ba, xb, bb, tacc, tbas, c0, one, x, &tfull[grp], &tempty[grp], wd, lane),
                  wd++;
          }
          if (write && (M == FULL || M == EXT)) {
#pragma unroll
            for (int u = 0; u < N / 16; u++) {
              tmem_ld16(tacc + 16 * u, xa);
              tmem_ld16(tbas + 16 * u, ba);
              tmem_wait_ld();
              pin(xa);
              pin(ba);
#pragma unroll
              for (int c = 0; c < 16; c += 4)
                *reinterpret_cast<int4*>(crow + 16 * u + c) = make_int4(
                    (int)(xa[c] + ba[c]), (int)(xa[c + 1] + ba[c + 1]), (int)(xa[c + 2] + ba[c + 2]),
                    (int)(xa[c + 3] + ba[c + 3]));
            }
          }
        }
      } else if constexpr (M == FOLDONLY) {
        // Register words the compiler must treat as new each word (pin), so nothing is hoisted or shared.
        uint32_t v[BNR];
#pragma unroll
        for (int c = 0; c < BNR; c++) asm volatile("mov.b32 %0, %1;" : "=r"(v[c]) : "r"(seed + 977u * c + row));
#pragma unroll 1
        for (int wd = 0; wd < words; wd++) {
          pin(v);
          cw -= MAGIC_BITS;
          fold_all<R, M, 0, BNR>(ch, t, v, cw);
        }
      } else if constexpr (R == I8C) {
        uint32_t v[BNR];
#pragma unroll 1
        for (int wd = 0; wd < words; wd++) {
          const int b = wd % NB;
          mbar_wait(&tfull[b], (wd / NB) & 1);
          tc_after_sync();
          if constexpr (M != MMAONLY) {
            ld_all<0, BNR>(v, tbase + b * N);
            tmem_wait_ld();
            pin(v);
            if (wd + 1 < words) {
              st_all<0, BNR>(v, tbase + ((b + 1) % NB) * N);
              tmem_wait_st();
            }
          }
          tc_before_sync();
          __syncwarp();
          if (wd + 1 < words && lane == 0) mbar_arrive(&tempty[(b + 1) % NB]);
          fold_all<R, M, 0, BNR>(ch, t, v, 0u);
        }
        if constexpr (M == FULL || M == EXT) {
#pragma unroll
          for (int i = 0; i < BNR; i++) t[i] = v[i];
        }
      } else {
        uint32_t ra[32], rb[32], c0[32];
#pragma unroll
        for (int c = 0; c < 32; c++) c0[c] = __ldg(magic + c);
#pragma unroll 1
        for (int wd = 0; wd < words; wd++) {
          const int b = wd % NB;
          const uint32_t tsrc = tbase + b * N;
          mbar_wait(&tfull[b], (wd / NB) & 1);
          tc_after_sync();
          if constexpr (M != MMAONLY) {
            tmem_ld32(tsrc, ra);
            cw -= MAGIC_BITS;
            word_chunks<R, M, 0, BNR>(ch, t, ra, rb, tsrc, c0, x, cw);
            if constexpr (is_k4(R)) tmem_wait_st();
          }
          tc_before_sync();
          __syncwarp();
          if (lane == 0) mbar_arrive(&tempty[b]);
        }
      }
      if constexpr (M == LD) {
        if (write) crow[0] = (int32_t)x;
      }
      if constexpr (R == K4P || R == K4B) {
#pragma unroll
        for (int c = 0; c < BNR; c++) t[c] -= (uint32_t)words * MAGIC_BITS;
      }
      if (write && (M == FULL || M == EXT || M == FOLDONLY)) {
        if constexpr (R != K4T) {
#pragma unroll
          for (int c = 0; c < BNR; c += 4)
            *reinterpret_cast<int4*>(crow + c) = make_int4((int)t[c], (int)t[c + 1], (int)t[c + 2], (int)t[c + 3]);
        }
        if constexpr (M != EXT) {
#pragma unroll
          for (int c = 0; c < NCH; c++) Rc[(size_t)(row * tiles_r + tr) * NCH + c] = duo_finish(ch[c]);
        }
      }
    }
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) tmem_dealloc(tmem, TCOLS);
  if (clk && threadIdx.x == 0) {
    uint64_t g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    const size_t cta = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
    clk[2 * cta] = clock64() - clk0[0];
    clk[2 * cta + 1] = g1 - clk0[1];
  }
#endif
}

// ---- host -------------------------------------------------------------------------------------------------------------
// Distributions: 0 uniform, 1 the e4m3 edge codes, 2 max (every code -128: each product +16384, k4t's accumulator
// moves at its bound).
static const char* const DIST[] = {"uniform", "edge", "max"};
static int8_t code(uint32_t seed, uint32_t which, uint32_t i, int dist) {
  const uint32_t h = dc_mix32(dc_mix32(seed ^ (which << 28)) + i);
  if (dist == 2) return -128;
  if (dist == 0) return (int8_t)((int)(h % 255u) - 127);
  static const int8_t E[8] = {-128, 127, 119, 120, -120, -121, 0, 1};
  return E[h % 8u];
}
// The receipt by its definition (header), one chain at a time.
static void host_ref(int m, int n, int k, int bnr, int L, uint32_t seed, const std::vector<int8_t>& X,
                     const std::vector<int8_t>& W, std::vector<int32_t>& C, std::vector<uint32_t>& R, uint32_t off = 0) {
  const int tiles = n / bnr;
  C.assign((size_t)m * n, 0);
  R.assign((size_t)m * tiles * NCH, 0);
  std::vector<int32_t> T(n);
  for (int i = 0; i < m; i++) {
    std::vector<DuoSt> ch(tiles * NCH);
    for (int tr = 0; tr < tiles; tr++)
      for (int c = 0; c < NCH; c++) ch[tr * NCH + c] = duo_init(seed, (uint32_t)((i * tiles + tr) * NCH + c));
    std::fill(T.begin(), T.end(), 0);
    for (int s = 0; s < k / L; s++) {
      for (int j = 0; j < n; j++) {
        int32_t acc = 0;
        for (int kk = s * L; kk < (s + 1) * L; kk++) acc += (int32_t)X[(size_t)i * k + kk] * W[(size_t)j * k + kk];
        T[j] = (int32_t)((uint32_t)T[j] + (uint32_t)acc);
      }
      for (int tr = 0; tr < tiles; tr++)
        for (int jl = 0; jl < bnr; jl++) {
          DuoSt& st = ch[tr * NCH + jl % NCH];
          const uint32_t p = (uint32_t)T[tr * bnr + jl] + (uint32_t)(s + 1) * off;
          const int t = (jl / 16) % 16;
          const uint32_t u = st.y ^ p ^ duo_xor(t);
          st.y = u * duo_mul(t) + st.z;
          st.z = dc_rotl(u, duo_rot(t)) + st.y;
        }
    }
    for (int j = 0; j < n; j++) C[(size_t)i * n + j] = T[j];
    for (int q = 0; q < tiles * NCH; q++) R[(size_t)i * tiles * NCH + q] = duo_finish(ch[q]);
  }
}
// Plane p of a code for route r: K4S [16h, l] (E4M3), PF8 [l] (E4M3), int8 routes the code itself.
static uint8_t plane_byte(int r, int p, int8_t x) {
  if (is_k4(r)) return e4m3_int(p == 0 ? 16 * limb_h(x) : limb_l(x));
  if (r == PF8) return e4m3_int(limb_l(x));
  return (uint8_t)x;
}
// rows x k codes -> [plane][rows / rt][k / KS][rt KS bytes in dc.cuh's K-major layout]: each stage tile one bulk copy.
static std::vector<uint8_t> tile_planes(int r, const std::vector<int8_t>& x, int rows, int k, int rt) {
  const int P = planes_of(r), kst = k / KS;
  std::vector<uint8_t> o((size_t)P * rows * k);
  for (int p = 0; p < P; p++)
    for (int row = 0; row < rows; row++)
      for (int kk = 0; kk < k; kk++) {
        const size_t tile = ((size_t)p * (rows / rt) + row / rt) * kst + kk / KS;
        o[tile * rt * KS + kmaj_off(row % rt, kk % KS, KS)] = plane_byte(r, p, x[(size_t)row * k + kk]);
      }
  return o;
}
static double e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 15, mi = b & 7;
  const double v = e == 0 ? mi / 512.0 : (1 + mi / 8.0) * (double)(1 << e) / 128.0;
  return b & 0x80 ? -v : v;
}

static int run_selftest() {
  long id_bad = 0, enc_bad = 0;
  for (int x = -128; x < 128; x++) {
    for (int p = 0; p < 2; p++) {
      const int v = p == 0 ? 16 * limb_h(x) : limb_l(x);
      enc_bad += e4m3_value(e4m3_int(v)) != (double)v;
    }
    for (int y = -128; y < 128; y++) {
      const double a = e4m3_value(plane_byte(K4S, 0, (int8_t)x)) + e4m3_value(plane_byte(K4S, 1, (int8_t)x));
      const double b = e4m3_value(plane_byte(K4S, 0, (int8_t)y)) + e4m3_value(plane_byte(K4S, 1, (int8_t)y));
      id_bad += a * b != (double)x * y;
    }
  }
  // tile_planes is a permutation: every (row, k) lands once, and its inverse reads the code back.
  const int rows = 256, k = 192;
  std::vector<int8_t> x((size_t)rows * k);
  for (size_t i = 0; i < x.size(); i++) x[i] = code(5, 1, (uint32_t)i, 0);
  const auto t = tile_planes(I8C, x, rows, k, BM);
  long tile_bad = 0;
  for (int row = 0; row < rows; row++)
    for (int kk = 0; kk < k; kk++) {
      const size_t tile = (size_t)(row / BM) * (k / KS) + kk / KS;
      tile_bad += (int8_t)t[tile * BM * KS + kmaj_off(row % BM, kk % KS, KS)] != x[(size_t)row * k + kk];
    }
  printf("{\"mode\": \"selftest\", \"identity_pairs\": 65536, \"identity_bad\": %ld, \"encode_bad\": %ld, "
         "\"tile_bad\": %ld}\n",
         id_bad, enc_bad, tile_bad);
  return id_bad == 0 && enc_bad == 0 && tile_bad == 0 ? 0 : 4;
}

// An arm: route, fold mode, spacing, fold warpgroups.
struct Arm {
  int r, m, L, fw;
};
static const char* route_name(int r) {
  static const char* n[] = {"k4s", "i8c", "i8r", "pf8", "pi8", "k4p", "k4b", "k4t"};
  return n[r];
}
static const char* mode_name(int m) {
  static const char* n[] = {"full", "ext", "ld", "mma", "foldonly", "mmares", "fullres"};
  return n[m];
}
static std::string arm_name(const Arm& a) {
  if (is_plain(a.r)) return std::string(route_name(a.r)) + (a.m == FULLRES ? "_res" : "");
  return std::string(route_name(a.r)) + "_L" + std::to_string(a.L) + "_fw" + std::to_string(a.fw) + "_" +
         mode_name(a.m);
}

static uint32_t* g_magic = nullptr;  // 32 words of 1.5 2^23 (c0's source), then 1 (k4t's one)
static uint64_t* g_clk = nullptr;    // when set, each CTA writes its SM clocks and globaltimer ns (the SM clock under load)
template <int R, int M, int L, int FW>
static void launch_t(dim3 grid, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed, int32_t* C,
                     uint32_t* Rc, int write, cudaStream_t st) {
  constexpr int smem = stages_of(R) * stage_bytes(R);
  static bool init = false;
  if (!init) {
    CK(cudaFuncSetAttribute(pg100<R, M, L, FW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    init = true;
  }
  if (!g_magic) {
    std::vector<uint32_t> mg(33, MAGIC_BITS);
    mg[32] = 1;
    CK(cudaMalloc(&g_magic, 4 * mg.size()));
    CK(cudaMemcpy(g_magic, mg.data(), 4 * mg.size(), cudaMemcpyHostToDevice));
  }
  pg100<R, M, L, FW><<<grid, 64 + 128 * FW, smem, st>>>(A, B, m, n, k, seed, C, Rc, write, g_magic, g_clk);
}
template <int R, int M, int L>
static bool launch_fw(const Arm& a, dim3 g, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t s,
                      int32_t* C, uint32_t* Rc, int wr, cudaStream_t st) {
  if constexpr (R == K4T) {  // receipt tile 128 (fw1's), one warpgroup for each of the CTA's two tiles
    if (a.fw != 1) return false;
    launch_t<R, M, L, 2>(g, A, B, m, n, k, s, C, Rc, wr, st);
  } else {
    if (a.fw == 2) launch_t<R, M, L, 2>(g, A, B, m, n, k, s, C, Rc, wr, st);
    else if constexpr (M == FULL) launch_t<R, M, L, 1>(g, A, B, m, n, k, s, C, Rc, wr, st);
    else return false;
  }
  return true;
}
template <int R, int M>
static bool launch_l(const Arm& a, dim3 g, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t s,
                     int32_t* C, uint32_t* Rc, int wr, cudaStream_t st) {
  return a.L == 32 ? launch_fw<R, M, 32>(a, g, A, B, m, n, k, s, C, Rc, wr, st)
                   : launch_fw<R, M, 64>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
}
template <int R>
static bool launch_m(const Arm& a, dim3 g, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t s,
                     int32_t* C, uint32_t* Rc, int wr, cudaStream_t st) {
  switch (a.m) {
    case FULL: return launch_l<R, FULL>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    case EXT: return launch_l<R, EXT>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    case LD: return launch_l<R, LD>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    case MMAONLY: return launch_l<R, MMAONLY>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    case MMARES: return launch_l<R, MMARES>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    case FULLRES: return launch_l<R, FULLRES>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
    default: return launch_l<R, FOLDONLY>(a, g, A, B, m, n, k, s, C, Rc, wr, st);
  }
}
static void launch(const Arm& a, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed, int32_t* C,
                   uint32_t* Rc, bool write, cudaStream_t st) {
  const dim3 g(n / cta_n(a.r), m / BM);
  bool ok = true;
  if (a.r == K4S) ok = launch_m<K4S>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == K4T) ok = launch_m<K4T>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == K4P) ok = launch_m<K4P>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == K4B) ok = launch_m<K4B>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == I8C) ok = launch_m<I8C>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == I8R) ok = launch_m<I8R>(a, g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == PF8 && a.m == FULLRES) launch_t<PF8, FULLRES, 32, 2>(g, A, B, m, n, k, seed, C, Rc, write, st);
  else if (a.r == PF8) launch_t<PF8, FULL, 32, 2>(g, A, B, m, n, k, seed, C, Rc, write, st);
  else launch_t<PI8, FULL, 32, 2>(g, A, B, m, n, k, seed, C, Rc, write, st);
  if (!ok) {
    fprintf(stderr, "ERR arm %s not instantiated\n", arm_name(a).c_str());
    exit(2);
  }
  CK(cudaGetLastError());
}

// The arms of record (gated) and the time-split arms (benched).
static std::vector<Arm> record_arms() {
  std::vector<Arm> v;
  for (int L : {32, 64})
    for (int r : {K4S, K4P, K4B, I8C, I8R})
      for (int fw : {2, 1})
        if ((r != K4P && r != K4B) || fw == 2) v.push_back({r, FULL, L, fw});
  for (int L : {32, 64}) v.push_back({K4T, FULL, L, 1});
  return v;
}
static std::vector<Arm> split_arms() {
  std::vector<Arm> v;
  for (int L : {32, 64})
    for (int r : {K4S, K4P, K4B, I8C, I8R, K4T})
      for (int md : {EXT, LD, MMAONLY, FOLDONLY, MMARES, FULLRES}) v.push_back({r, md, L, r == K4T ? 1 : 2});
  return v;
}

struct Operands {
  uint8_t* A[8] = {};
  uint8_t* B[8] = {};
  int32_t* C = nullptr;
  uint32_t* R = nullptr;
  size_t c_words, r_words;
  Operands(const std::vector<int8_t>& X, const std::vector<int8_t>& W, int m, int n, int k) {
    for (int r = 0; r < 5; r++) {
      if (r == I8R || r == PI8) continue;  // the int8 routes share I8C's tiles
      const auto a = tile_planes(r, X, m, k, BM), b = tile_planes(r, W, n, k, N);
      CK(cudaMalloc(&A[r], a.size()));
      CK(cudaMalloc(&B[r], b.size()));
      CK(cudaMemcpy(A[r], a.data(), a.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(B[r], b.data(), b.size(), cudaMemcpyHostToDevice));
    }
    A[I8R] = A[PI8] = A[I8C];
    B[I8R] = B[PI8] = B[I8C];
    A[K4P] = A[K4B] = A[K4T] = A[K4S];
    B[K4P] = B[K4B] = B[K4T] = B[K4S];
    c_words = (size_t)m * n;
    r_words = (size_t)m * (n / 64) * NCH;
    CK(cudaMalloc(&C, c_words * 4));
    CK(cudaMalloc(&R, r_words * 4));
  }
};

static int run_gate(int m, int n, int k, uint32_t seed, int dist) {
  if (m % BM || n % N || k % KS) {
    fprintf(stderr, "ERR shape %dx%dx%d: m a multiple of %d, n of %d, k of %d\n", m, n, k, BM, N, KS);
    return 2;
  }
  std::vector<int8_t> X((size_t)m * k), W((size_t)n * k);
  for (size_t i = 0; i < X.size(); i++) X[i] = code(seed, 1, (uint32_t)i, dist);
  for (size_t i = 0; i < W.size(); i++) W[i] = code(seed, 2, (uint32_t)i, dist);
  Operands d(X, W, m, n, k);
  std::vector<int32_t> C(d.c_words);
  std::vector<uint32_t> R;
  bool all = true;
  auto check = [&](const Arm& a, const std::vector<int32_t>& Cr, const std::vector<uint32_t>* Rr) {
    R.assign(Rr ? Rr->size() : 0, 0);
    auto call = [&](bool write) {
      CK(cudaMemset(d.C, 0xA5, d.c_words * 4));
      CK(cudaMemset(d.R, 0xA5, d.r_words * 4));
      launch(a, d.A[a.r], d.B[a.r], m, n, k, seed, d.C, d.R, write, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(C.data(), d.C, C.size() * 4, cudaMemcpyDeviceToHost));
      if (Rr) CK(cudaMemcpy(R.data(), d.R, R.size() * 4, cudaMemcpyDeviceToHost));
      size_t bc = 0, br = 0;
      for (size_t i = 0; i < C.size(); i++) bc += C[i] != Cr[i];
      for (size_t i = 0; i < R.size(); i++) br += R[i] != (*Rr)[i];
      return std::make_pair(bc, br);
    };
    const auto ctl = call(false);
    const auto got = call(true);
    const bool ok = got.first == 0 && got.second == 0, rejected = ctl.first > 0 && (!Rr || ctl.second > 0);
    all = all && ok && rejected;
    printf("{\"mode\": \"gate\", \"arch\": \"sm_100a\", \"arm\": \"%s\", \"dist\": \"%s\", \"m\": %d, \"n\": %d, "
           "\"k\": %d, \"exact\": %s, \"c_bad\": %zu, \"r_bad\": %zu, \"control_rejected\": %s, \"words\": %zu, "
           "\"receipts\": %zu}\n",
           arm_name(a).c_str(), DIST[dist], m, n, k, ok ? "true" : "false", got.first, got.second,
           rejected ? "true" : "false", C.size(), R.size());
    fflush(stdout);
  };
  for (int L : {32, 64})
    for (int fw : {2, 1}) {
      std::vector<int32_t> Cr;
      std::vector<uint32_t> Rr, Rp;
      host_ref(m, n, k, N / fw, L, seed, X, W, Cr, Rr);
      if (fw == 2) host_ref(m, n, k, N / fw, L, seed, X, W, Cr, Rp, MAGIC_BITS);
      for (const Arm& a : record_arms())
        if (a.L == L && a.fw == fw && n % cta_n(a.r) == 0) check(a, Cr, a.r == K4P ? &Rp : &Rr);
      if (fw == 2)
        for (int r : {K4S, K4P, K4B, I8R}) check({r, EXT, L, 2}, Cr, nullptr);
      if (fw == 1 && n % cta_n(K4T) == 0) check({K4T, EXT, L, 1}, Cr, nullptr);
    }
  // Plain: C = X W^T over the codes (PI8) and over the l limbs in FP32 (PF8: every partial sum an integer below 2^24).
  std::vector<int32_t> Ci((size_t)m * n), Cf((size_t)m * n);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      int64_t si = 0, sf = 0;
      for (int kk = 0; kk < k; kk++) {
        si += (int)X[(size_t)i * k + kk] * W[(size_t)j * k + kk];
        sf += limb_l(X[(size_t)i * k + kk]) * limb_l(W[(size_t)j * k + kk]);
      }
      Ci[(size_t)i * n + j] = (int32_t)si;
      const float f = (float)sf;
      memcpy(&Cf[(size_t)i * n + j], &f, 4);
    }
  check({PI8, FULL, 32, 2}, Ci, nullptr);
  check({PF8, FULL, 32, 2}, Cf, nullptr);
  return all ? 0 : 4;
}

#ifdef PG_CUBLASLT
// cuBLASLt C^T (n x m, column-major) = W (k x n)^T X^T (k x m): int8 (s32 out) or E4M3 (FP32 out, else BF16).
struct Lt {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
  void* ws = nullptr;
  size_t wsb = 64 << 20;
  bool ok = false;
  std::string out;
  Lt(int m, int n, int k, bool fp8) {
    cublasLtCreate(&h);
    cudaMalloc(&ws, wsb);
    const cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    const cudaDataType_t outs[2] = {fp8 ? CUDA_R_32F : CUDA_R_32I, CUDA_R_16BF};
    for (int o = 0; o < (fp8 ? 2 : 1) && !ok; o++) {
      if (fp8) cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
      else cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn);
      const cudaDataType_t in = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_8I;
      cublasLtMatrixLayoutCreate(&a, in, k, n, k);
      cublasLtMatrixLayoutCreate(&b, in, k, m, k);
      cublasLtMatrixLayoutCreate(&c, outs[o], n, m, n);
      cublasLtMatmulPreference_t pref;
      cublasLtMatmulPreferenceCreate(&pref);
      cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb);
      cublasLtMatmulHeuristicResult_t r;
      int got = 0;
      if (cublasLtMatmulAlgoGetHeuristic(h, op, a, b, c, c, pref, 1, &r, &got) == CUBLAS_STATUS_SUCCESS && got) {
        algo = r.algo;
        ok = true;
        out = outs[o] == CUDA_R_16BF ? "bf16" : outs[o] == CUDA_R_32F ? "f32" : "s32";
      }
      cublasLtMatmulPreferenceDestroy(pref);
    }
  }
  void run(const void* X, const void* W, void* C, bool fp8, cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    const float onef = 1.f, zerof = 0.f;
    cublasLtMatmul(h, op, fp8 ? (const void*)&onef : (const void*)&one, W, a, X, b,
                   fp8 ? (const void*)&zerof : (const void*)&zero, C, c, C, c, &algo, ws, wsb, st);
  }
};
#endif

static int run_bench(int reps, int m, int n, int k) {
  if (m % BM || n % cta_n(K4T) || k % KS) {
    fprintf(stderr, "ERR shape %dx%dx%d: m a multiple of %d, n of %d, k of %d\n", m, n, k, BM, cta_n(K4T), KS);
    return 2;
  }
  std::vector<int8_t> X((size_t)m * k), W((size_t)n * k);
  for (size_t i = 0; i < X.size(); i++) X[i] = code(7, 1, (uint32_t)i, 0);
  for (size_t i = 0; i < W.size(); i++) W[i] = code(7, 2, (uint32_t)i, 0);
  Operands d(X, W, m, n, k);
  struct Run {
    std::string name;
    int kind;  // 0 an arm, 1 cuBLASLt int8, 2 cuBLASLt FP8
    Arm a;
  };
  std::vector<Run> runs;
  for (const Arm& a : record_arms()) runs.push_back({arm_name(a), 0, a});
  for (const Arm& a : split_arms()) runs.push_back({arm_name(a), 0, a});
  runs.push_back({"pf8", 0, {PF8, FULL, 32, 2}});
  runs.push_back({"pf8_res", 0, {PF8, FULLRES, 32, 2}});
  runs.push_back({"pi8", 0, {PI8, FULL, 32, 2}});
  std::string lt_out;
#ifdef PG_CUBLASLT
  Lt lti(m, n, k, false), ltf(m, n, k, true);
  void* lt_c = nullptr;
  CK(cudaMalloc(&lt_c, (size_t)m * n * 4));
  // cuBLASLt's operands: row-major codes (int8) and E4M3 l limbs.
  int8_t* xi;
  int8_t* wi;
  uint8_t *xf, *wf;
  CK(cudaMalloc(&xi, X.size()));
  CK(cudaMalloc(&wi, W.size()));
  CK(cudaMalloc(&xf, X.size()));
  CK(cudaMalloc(&wf, W.size()));
  CK(cudaMemcpy(xi, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(wi, W.data(), W.size(), cudaMemcpyHostToDevice));
  {
    std::vector<uint8_t> a(X.size()), b(W.size());
    for (size_t i = 0; i < a.size(); i++) a[i] = plane_byte(PF8, 0, X[i]);
    for (size_t i = 0; i < b.size(); i++) b[i] = plane_byte(PF8, 0, W[i]);
    CK(cudaMemcpy(xf, a.data(), a.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wf, b.data(), b.size(), cudaMemcpyHostToDevice));
  }
  if (lti.ok) runs.push_back({"cublaslt_i8", 1, {}});
  if (ltf.ok) runs.push_back({"cublaslt_f8", 2, {}}), lt_out = ltf.out;
#endif
  auto go = [&](const Run& r) {
    if (r.kind == 0) launch(r.a, d.A[r.a.r], d.B[r.a.r], m, n, k, 1, d.C, d.R, true, 0);
#ifdef PG_CUBLASLT
    else if (r.kind == 1) lti.run(xi, wi, lt_c, false, 0);
    else ltf.run(xf, wf, lt_c, true, 0);
#endif
  };
  for (const Run& r : runs) go(r);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<std::vector<float>> ms(runs.size());
  for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i < runs.size(); i++) {
      CK(cudaEventRecord(e0));
      go(runs[i]);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float t = 0;
      CK(cudaEventElapsedTime(&t, e0, e1));
      ms[i].push_back(t);
    }
  // Each arm's SM clock under its own load: one more launch (after a warm one) with every CTA's clock64 and
  // globaltimer deltas, summed over CTAs.
  std::vector<uint64_t> ck(2 * (size_t)(n / N) * (m / BM));
  std::vector<double> mhz(runs.size(), 0.0), cta_clk(runs.size(), 0.0);
  CK(cudaMalloc(&g_clk, ck.size() * 8));
  uint64_t* clk_buf = g_clk;
  for (size_t i = 0; i < runs.size(); i++) {
    if (runs[i].kind != 0) continue;
    const size_t ctas = (size_t)(n / cta_n(runs[i].a.r)) * (m / BM);
    g_clk = nullptr;
    go(runs[i]);
    g_clk = clk_buf;
    go(runs[i]);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ck.data(), clk_buf, ck.size() * 8, cudaMemcpyDeviceToHost));
    double c = 0, ns = 0;
    std::vector<double> per(ctas);
    for (size_t q = 0; q < ctas; q++) c += (double)ck[2 * q], ns += (double)ck[2 * q + 1], per[q] = (double)ck[2 * q];
    mhz[i] = ns > 0 ? 1e3 * c / ns : 0.0;
    std::sort(per.begin(), per.end());
    cta_clk[i] = per[ctas / 2];
  }
  g_clk = nullptr;
  CK(cudaFree(clk_buf));
  int sms = 0, khz = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  printf("{\"mode\": \"bench\", \"arch\": \"sm_100a\", \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"sms\": %d, "
         "\"clock_khz\": %d, \"cublaslt_f8_out\": \"%s\"",
         m, n, k, reps, sms, khz, lt_out.c_str());
  for (size_t i = 0; i < runs.size(); i++) {
    std::sort(ms[i].begin(), ms[i].end());
    printf(", \"%s_ms\": %.6f", runs[i].name.c_str(), ms[i][ms[i].size() / 2]);
    if (mhz[i] > 0) printf(", \"%s_mhz\": %.1f, \"%s_cta_clk\": %.0f", runs[i].name.c_str(), mhz[i], runs[i].name.c_str(),
                           cta_clk[i]);
  }
  printf("}\n");
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "selftest") return run_selftest();
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 10 || p.minor != 0) {
    fprintf(stderr, "ERR device sm_%d%d: pgemm runs on sm_100a\n", p.major, p.minor);
    return 2;
  }
  if (mode == "gate" && argc == 7) {
    const std::string dist = argv[6];
    for (int d = 0; d < 3; d++)
      if (dist == DIST[d]) return run_gate(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), (uint32_t)strtoul(argv[5], 0, 10), d);
    fprintf(stderr, "ERR dist %s: uniform, edge or max\n", dist.c_str());
    return 2;
  }
  if (mode == "bench" && argc == 6) return run_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: pgemm selftest | gate M N K SEED uniform|edge|max | bench REPS M N K\n");
  return 2;
}
