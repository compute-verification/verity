// egemm: the Dither GEMM with int8 emulated in FP8 (emu8's Karatsuba route) on Hopper (sm_90a, wgmma E4M3) and
// datacenter Blackwell (sm_100a, tcgen05 kind::f8f6f4), against native int8 in the same kernel skeleton.
//
// The product and the receipt are dgemm's (../dc/dgemm.cu, header): C = X W^T over s8 codes, the s32 running sum T_s
// committed every 32 steps of k, each word absorbed by duo into chain (i, tn, j % 16) at key (j / 16) % 16, for an
// output tile of BM x BN. The emulated route must give the same C and the same receipts bit for bit.
//
// Limbs (emu8): x = 16 h + l with h = (x + 8) >> 4 in [-8, 8], l in [-8, 7], m = h + l in [-15, 14], all exact in E4M3,
// and xy = 240 hh' + 16 mm' - 15 ll'. split writes three E4M3 planes of a side, plane-major ([3][rows][k] bytes).
//   sm_100a, k3z: A planes [4h, m, -l], B planes [4h', m', l']. Each k32 step: acc0 = (4h)(4h') then += (-l)(l'), acc1 =
//     mm', both from +0 in TMEM (2 BN columns); word T += bits(fma(16, acc1, fma(15, acc0, 1.5 2^23))) - 0x4B400000.
//     tcgen05's FP32 adder is exact for these (|acc0| < 2^16; fpx: exact at every gap <= 23).
//   sm_90a, k3h: planes [h, m, l] on both sides, three accumulators from +0 in registers; word T += bits(fma(-15, g2,
//     fma(16, g1, fma(240, g0, 1.5 2^23)))) - 0x4B400000. Hopper's FP8 adder keeps 14 bits (fpx): every term and partial
//     sum of one k32 instruction is below 2^13 here (|g| <= 32 * 225 = 7200), so each g is exact; k3z's acc0 is not.
//   sm_100a, k4s: planes [16h, l] on both sides, four MMAs (every pairing) onto one accumulator that holds 1.5 2^23 at
//     each step's start (tcgen05.st after each word leaves); word T += bits(acc) - 0x4B400000, one IADD3: the
//     extraction a cheater on int8-free hardware also pays (NOBITS), where k3z pays three lane ops.
//   Every FFMA lands in [2^23, 2^24) (ulp 1), so the bits difference is the integer increment.
// The emulated route keeps T in registers (one row a thread on sm_100a, so BN = 64 or 128; 3 x BN / 2 accumulators plus
// T on sm_90a, so BN = 64), where native int8 keeps it in its accumulator.
//
// Kernels (dgemm's correctness-first single-stage mainloop: load, fence, multiply, wait, fold a step at a time):
//   eg100<R, BN>  sm_100a: four warps, a 128 x BN tile, R = 0 kind::i8 into TMEM, 1 k3z (3 kind::f8f6f4), 2 k4s (4),
//                 3 i8r (kind::i8 from a fresh accumulator each step, the word in registers as k4s keeps it)
//   eg90<R, BN>   sm_90a: one warpgroup, a 64 x BN tile, R = 0 wgmma s8, R = 1 k3h (3 wgmma E4M3)
//   split<FAM, SIDE>  the limb planes of one side (X's is the served operand's per-request cost; W's is offline)
//   RES: the operand tiles are loaded once and every step multiplies them again, so the time is the on-chip pipeline
//   alone (MMA, extraction, fold) without global loads; its words are the product of operands whose every k-slice is
//   slice 0. PIPE (sm_100a, R != 0, resident): two TMEM buffers, step s + 1's MMAs issued before step s is absorbed,
//   so the tensor runs beside extraction and fold; gated against that product's host reference.
// Modes:
//   egemm gate M N K SEED DIST       DIST uniform (dgemm's codes) or edge (limb extremes: -128, 127, 119, 120, -120, -121,
//                                    0, 1): every route and tile against the host reference, poisoned with 0xA5, and the
//                                    no-write control (must be rejected); one JSON line a route, exit 4 unless all hold
//   egemm bench REPS M N K           the arms interleaved, streamed and resident, median ms, one JSON line
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a egemm.cu   (or 90a)
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../dc/dc.cuh"
#include "../dc/wgmma.cuh"

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

constexpr int L = 32;
constexpr int NCH = 16;
constexpr float MAGIC = 12582912.f;
constexpr uint32_t MAGIC_BITS = 0x4B400000u;

DC_HD int limb_h(int x) { return (x + 8) >> 4; }
DC_HD int limb_l(int x) { return x - 16 * limb_h(x); }
// E4M3 code of an integer with |v| <= 128 whose bits below its top four are zero (every limb value here).
DC_HD uint8_t e4m3_int(int v) {
  if (v == 0) return 0;
  const uint8_t sg = v < 0 ? 0x80 : 0;
  const int a = v < 0 ? -v : v;
  int e = 0;
  while ((a >> (e + 1)) != 0) e++;
  const int mi = ((a << 3) >> e) - 8;
  return sg | (uint8_t)((e + 7) << 3) | (uint8_t)mi;
}
// Plane p of x for route FAM (100: k3z, 104: k4s, 90: k3h) on side SIDE (0 = A = X, 1 = B = W).
DC_HD int plane_val(int fam, int side, int p, int x) {
  const int h = limb_h(x), l = limb_l(x), m = h + l;
  if (fam == 100) return p == 0 ? 4 * h : p == 1 ? m : (side == 0 ? -l : l);
  if (fam == 104) return p == 0 ? 16 * h : p == 1 ? l : 0;
  return p == 0 ? h : p == 1 ? m : l;
}

template <int FAM, int SIDE>
__global__ void split(const int8_t* src, uint8_t* dst, size_t n) {
  const size_t n4 = n / 4;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    const char4 x = reinterpret_cast<const char4*>(src)[i];
#pragma unroll
    for (int p = 0; p < 3; p++) {
      uchar4 o;
      o.x = e4m3_int(plane_val(FAM, SIDE, p, x.x));
      o.y = e4m3_int(plane_val(FAM, SIDE, p, x.y));
      o.z = e4m3_int(plane_val(FAM, SIDE, p, x.z));
      o.w = e4m3_int(plane_val(FAM, SIDE, p, x.w));
      reinterpret_cast<uchar4*>(dst + p * n)[i] = o;
    }
  }
}

DC_DEV void load_slice(unsigned char* dst, const uint8_t* src, int rows, int k, int k0) {
  for (int c = threadIdx.x; c < rows * 2; c += blockDim.x) {
    const int r = c >> 1, kc = c & 1;
    const uint4 v = *reinterpret_cast<const uint4*>(src + (size_t)r * k + k0 + 16 * kc);
    *reinterpret_cast<uint4*>(dst + kmaj_off(r, 16 * kc, L)) = v;
  }
}
DC_DEV uint32_t word_step100(uint32_t t, uint32_t a0, uint32_t a1) {
  return t + (__float_as_uint(fmaf(16.f, __uint_as_float(a1), fmaf(15.f, __uint_as_float(a0), MAGIC))) - MAGIC_BITS);
}
DC_DEV int word_step90(int t, float g0, float g1, float g2) {
  return t + (int)(__float_as_uint(fmaf(-15.f, g2, fmaf(16.f, g1, fmaf(240.f, g0, MAGIC)))) - MAGIC_BITS);
}

// ---- Hopper ---------------------------------------------------------------------------------------------------------------
DC_DEV void wgmma_e4m3_n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, "
      "1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d)
      : "memory");
}
// dgemm's absorb90: thread (warp w, lane 4 g + q) holds D[16 w + g + 8 (e >> 1)][8 ni + 2 q + (e & 1)] in d[4 ni + e].
template <int NI, int END>
DC_DEV void absorb90(DuoSt (&ch)[8], const int* d) {
  if constexpr (NI < END) {
#pragma unroll
    for (int e = 0; e < 4; e++) duo<(NI / 2) % 16>(ch[4 * (e >> 1) + 2 * (NI % 2) + (e & 1)], (uint32_t)d[4 * NI + e]);
    absorb90<NI + 1, END>(ch, d);
  }
}

template <int R, int BN, bool FOLD, bool WRITE, bool RES>
__global__ void __launch_bounds__(128) eg90(const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed,
                                            int32_t* C, uint32_t* Rc) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int NP = R == 0 ? 1 : 3;
  __shared__ __align__(1024) unsigned char sA[NP][64 * L];
  __shared__ __align__(1024) unsigned char sB[NP][BN * L];
  const int tn = blockIdx.x, tm = blockIdx.y, tiles_n = gridDim.x;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  uint64_t da[NP], db[NP];
#pragma unroll
  for (int p = 0; p < NP; p++) {
    da[p] = wgmma_desc(smem_u32(sA[p]), kmaj_lbo(), kmaj_sbo(L));
    db[p] = wgmma_desc(smem_u32(sB[p]), kmaj_lbo(), kmaj_sbo(L));
  }
  int t[BN / 2];
  float g0[R == 0 ? 1 : BN / 2], g1[R == 0 ? 1 : BN / 2], g2[R == 0 ? 1 : BN / 2];
#pragma unroll
  for (int i = 0; i < BN / 2; i++) t[i] = 0;
  DuoSt ch[8];
  auto row_of = [&](int h) { return tm * 64 + 16 * w + g + 8 * (h >> 2); };
  auto col_of = [&](int h) { return 8 * ((h >> 1) & 1) + 2 * q + (h & 1); };
  if constexpr (FOLD) {
#pragma unroll
    for (int h = 0; h < 8; h++) ch[h] = duo_init(seed, (uint32_t)((row_of(h) * tiles_n + tn) * NCH + col_of(h)));
  }
  for (int s = 0; s < k / L; s++) {
    if (!RES || s == 0) {
#pragma unroll
      for (int p = 0; p < NP; p++) {
        load_slice(sA[p], A + (size_t)p * m * k + (size_t)tm * 64 * k, 64, k, s * L);
        load_slice(sB[p], B + (size_t)p * n * k + (size_t)tn * BN * k, BN, k, s * L);
      }
    }
    fence_async_smem();
    __syncthreads();
    wgmma_fence();
    if constexpr (R == 0) {
      if constexpr (BN == 128) wgmma_s8_n128(t, da[0], db[0], s > 0);
      else wgmma_s8_n64(t, da[0], db[0], s > 0);
    } else {
      wgmma_e4m3_n64(g0, da[0], db[0], 0);
      wgmma_e4m3_n64(g1, da[NP - 2], db[NP - 2], 0);
      wgmma_e4m3_n64(g2, da[NP - 1], db[NP - 1], 0);
    }
    wgmma_commit();
    wgmma_wait<0>();
    if constexpr (R == 0) {
      wgmma_hold(t);
    } else {
      wgmma_hold(g0);
      wgmma_hold(g1);
      wgmma_hold(g2);
#pragma unroll
      for (int i = 0; i < BN / 2; i++) t[i] = word_step90(t[i], g0[i], g1[i], g2[i]);
    }
    if constexpr (FOLD) absorb90<0, BN / 8>(ch, t);
    __syncthreads();
  }
  if constexpr (WRITE) {
#pragma unroll
    for (int ni = 0; ni < BN / 8; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++)
        C[(size_t)(tm * 64 + 16 * w + g + 8 * (e >> 1)) * n + tn * BN + 8 * ni + 2 * q + (e & 1)] = t[4 * ni + e];
    if constexpr (FOLD) {
#pragma unroll
      for (int h = 0; h < 8; h++) Rc[(size_t)(row_of(h) * tiles_n + tn) * NCH + col_of(h)] = duo_finish(ch[h]);
    }
  }
#endif
}

// ---- Blackwell ------------------------------------------------------------------------------------------------------------
// Thread (warp w, lane) owns row 32 w + lane; chunk u of 16 columns of an accumulator lands in r[0..15].
template <int U, int END, bool FOLD, bool STORE>
DC_DEV void absorb100(DuoSt (&ch)[NCH], uint32_t trow, int32_t* crow) {
  if constexpr (U < END) {
    uint32_t r[16];
    tmem_ld16(trow + 16 * U, r);
    tmem_wait_ld();
    if constexpr (FOLD) {
#pragma unroll
      for (int c = 0; c < 16; c++) duo<U % 16>(ch[c], r[c]);
    }
    if constexpr (STORE) {
#pragma unroll
      for (int c = 0; c < 16; c += 4)
        *reinterpret_cast<int4*>(crow + 16 * U + c) = make_int4((int)r[c], (int)r[c + 1], (int)r[c + 2], (int)r[c + 3]);
    }
    absorb100<U + 1, END, FOLD, STORE>(ch, trow, crow);
  }
}
// k3z: acc0 at columns [0, BN), acc1 at [BN, 2 BN).
template <int U, int BN, bool FOLD>
DC_DEV void absorb100k3(DuoSt (&ch)[NCH], uint32_t (&t)[BN], uint32_t trow) {
  if constexpr (U < BN / 16) {
    uint32_t r0[16], r1[16];
    tmem_ld16(trow + 16 * U, r0);
    tmem_ld16(trow + BN + 16 * U, r1);
    tmem_wait_ld();
#pragma unroll
    for (int c = 0; c < 16; c++) {
      t[16 * U + c] = word_step100(t[16 * U + c], r0[c], r1[c]);
      if constexpr (FOLD) duo<U % 16>(ch[c], t[16 * U + c]);
    }
    absorb100k3<U + 1, BN, FOLD>(ch, t, trow);
  }
}
// k4s: the accumulator at columns [0, BN) restarts at 1.5 2^23 as each chunk leaves.
template <int U, int BN, bool FOLD>
DC_DEV void absorb100k4(DuoSt (&ch)[NCH], uint32_t (&t)[BN], uint32_t trow, const uint32_t (&c0)[16]) {
  if constexpr (U < BN / 16) {
    uint32_t r[16];
    tmem_ld16(trow + 16 * U, r);
    tmem_wait_ld();
    tmem_st16(trow + 16 * U, c0);
#pragma unroll
    for (int c = 0; c < 16; c++) {
      t[16 * U + c] += r[c] - MAGIC_BITS;
      if constexpr (FOLD) duo<U % 16>(ch[c], t[16 * U + c]);
    }
    absorb100k4<U + 1, BN, FOLD>(ch, t, trow, c0);
  }
}
// i8r: each step's s32 product from a fresh accumulator, the running word in registers (k4s's shape with int8).
template <int U, int BN, bool FOLD>
DC_DEV void absorb100r(DuoSt (&ch)[NCH], uint32_t (&t)[BN], uint32_t trow) {
  if constexpr (U < BN / 16) {
    uint32_t r[16];
    tmem_ld16(trow + 16 * U, r);
    tmem_wait_ld();
#pragma unroll
    for (int c = 0; c < 16; c++) {
      t[16 * U + c] += r[c];
      if constexpr (FOLD) duo<U % 16>(ch[c], t[16 * U + c]);
    }
    absorb100r<U + 1, BN, FOLD>(ch, t, trow);
  }
}

// R: 0 i8 (the word in TMEM), 1 k3z, 2 k4s, 3 i8r. PIPE (resident only, R != 0): two TMEM buffers, and step s + 1's
// MMAs are issued before the threads absorb step s, so the tensor overlaps extraction and fold.
template <int R, int BN, bool FOLD, bool WRITE, bool RES, bool PIPE>
__global__ void __launch_bounds__(128) eg100(const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed,
                                             int32_t* C, uint32_t* Rc) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  static_assert(!PIPE || (RES && R != 0), "PIPE: resident operands, and a route whose word is in registers");
  constexpr int BM = 128, NP = R == 1 ? 3 : R == 2 ? 2 : 1, TCOLS = R == 1 ? 2 * BN : BN;
  constexpr int NBUF = PIPE ? 2 : 1;
  __shared__ __align__(1024) unsigned char sA[NP][BM * L];
  __shared__ __align__(1024) unsigned char sB[NP][BN * L];
  __shared__ uint64_t bar[NBUF];
  __shared__ uint32_t tslot;
  const int tn = blockIdx.x, tm = blockIdx.y, tiles_n = gridDim.x;
  const uint32_t w = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
  const int row = tm * BM + 32 * w + lane;
  if (threadIdx.x == 0) {
    for (int b = 0; b < NBUF; b++) mbar_init(&bar[b], 1);
    fence_mbar_init();
  }
  if (w == 0) tmem_alloc(&tslot, NBUF * TCOLS);
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = tslot, trow = tmem + ((32u * w) << 16);
  uint64_t da[NP], db[NP];
#pragma unroll
  for (int p = 0; p < NP; p++) {
    da[p] = umma_desc(smem_u32(sA[p]), kmaj_lbo(), kmaj_sbo(L));
    db[p] = umma_desc(smem_u32(sB[p]), kmaj_lbo(), kmaj_sbo(L));
  }
  constexpr uint32_t idesc = R == 0 || R == 3 ? umma_idesc(2, 1, 1, BM, BN) : umma_idesc(1, 0, 0, BM, BN);
  DuoSt ch[NCH];
  if constexpr (FOLD) {
#pragma unroll
    for (int c = 0; c < NCH; c++) ch[c] = duo_init(seed, (uint32_t)((row * tiles_n + tn) * NCH + c));
  }
  uint32_t t[R == 0 ? 1 : BN];
#pragma unroll
  for (int i = 0; i < (R == 0 ? 1 : BN); i++) t[i] = 0;
  int32_t* crow = C + (size_t)row * n + tn * BN;
  uint32_t c0[16];
#pragma unroll
  for (int c = 0; c < 16; c++) c0[c] = MAGIC_BITS;
  if constexpr (R == 2) {
#pragma unroll
    for (int u = 0; u < NBUF * BN / 16; u++) tmem_st16(trow + 16 * u, c0);
    tmem_wait_st();
  }
  // Step s's MMAs into buffer b (one elected thread of warp 0), committed to bar[b].
  auto issue = [&](int s, int b) {
    if (w == 0) {
      if (elect_one()) {
        const uint32_t tb = tmem + b * TCOLS;
        if constexpr (R == 0) {
          umma_i8(tb, da[0], db[0], idesc, s > 0);
        } else if constexpr (R == 3) {
          umma_i8(tb, da[0], db[0], idesc, 0);
        } else if constexpr (R == 1) {
          umma_f8f6f4(tb, da[0], db[0], idesc, 0);
          umma_f8f6f4(tb, da[NP - 1], db[NP - 1], idesc, 1);
          umma_f8f6f4(tb + BN, da[NP - 2], db[NP - 2], idesc, 0);
        } else {
#pragma unroll
          for (int pa = 0; pa < 2; pa++)
#pragma unroll
            for (int pb = 0; pb < 2; pb++) umma_f8f6f4(tb, da[pa & (NP - 1)], db[pb & (NP - 1)], idesc, 1);
        }
        umma_commit(&bar[b]);
      }
      __syncwarp();
    }
  };
  auto absorb = [&](int b) {
    const uint32_t tr = trow + b * TCOLS;
    if constexpr (R == 0) {
      if constexpr (FOLD) absorb100<0, BN / 16, true, false>(ch, tr, crow);
    } else if constexpr (R == 1) {
      absorb100k3<0, BN, FOLD>(ch, t, tr);
    } else if constexpr (R == 3) {
      absorb100r<0, BN, FOLD>(ch, t, tr);
    } else {
      absorb100k4<0, BN, FOLD>(ch, t, tr, c0);
      tmem_wait_st();
    }
  };
  const int steps = k / L;
  if constexpr (PIPE) {
#pragma unroll
    for (int p = 0; p < NP; p++) {
      load_slice(sA[p], A + (size_t)p * m * k + (size_t)tm * BM * k, BM, k, 0);
      load_slice(sB[p], B + (size_t)p * n * k + (size_t)tn * BN * k, BN, k, 0);
    }
    fence_async_smem();
    tc_before_sync();
    __syncthreads();
    tc_after_sync();
    issue(0, 0);
    for (int s = 0; s < steps; s++) {
      const int b = s & 1;
      mbar_wait(&bar[b], (uint32_t)((s >> 1) & 1));
      tc_after_sync();
      if (s + 1 < steps) issue(s + 1, b ^ 1);  // buffer b ^ 1 was absorbed (and restarted) before the last barrier
      absorb(b);
      tc_before_sync();
      __syncthreads();
      tc_after_sync();
    }
  } else {
    for (int s = 0; s < steps; s++) {
      if (!RES || s == 0) {
#pragma unroll
        for (int p = 0; p < NP; p++) {
          load_slice(sA[p], A + (size_t)p * m * k + (size_t)tm * BM * k, BM, k, s * L);
          load_slice(sB[p], B + (size_t)p * n * k + (size_t)tn * BN * k, BN, k, s * L);
        }
      }
      fence_async_smem();
      tc_before_sync();
      __syncthreads();
      tc_after_sync();
      issue(s, 0);
      mbar_wait(&bar[0], (uint32_t)(s & 1));
      tc_after_sync();
      absorb(0);
      tc_before_sync();
      __syncthreads();
    }
    tc_after_sync();
  }
  if constexpr (WRITE) {
    if constexpr (R == 0) {
      absorb100<0, BN / 16, false, true>(ch, trow, crow);
    } else {
#pragma unroll
      for (int c = 0; c < BN; c += 4)
        *reinterpret_cast<int4*>(crow + c) = make_int4((int)t[c], (int)t[c + 1], (int)t[c + 2], (int)t[c + 3]);
    }
    if constexpr (FOLD) {
#pragma unroll
      for (int c = 0; c < NCH; c++) Rc[(size_t)(row * tiles_n + tn) * NCH + c] = duo_finish(ch[c]);
    }
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (w == 0) tmem_dealloc(tmem, NBUF * TCOLS);
#endif
}

// ---- host -----------------------------------------------------------------------------------------------------------------
static int g_sm = 0;
struct Arm {
  int route, bm, bn;
  const char* name;
};
// Family arms: native int8 at each emulated tile (the like-for-like ratio), int8 at dgemm's tile, emulated. On sm_100a
// the emulated routes at BN = 64 hold half the running words (k4s: 246 registers at BN = 128, two CTAs a SM), and i8r
// is native int8 in k4s's shape (a fresh accumulator each step, the word in registers), the baseline PIPE can run.
static std::vector<Arm> arms() {
  if (g_sm == 100)
    return {{0, 128, 64, "i8_bn64"},   {3, 128, 64, "i8r_bn64"},   {1, 128, 64, "k3z_bn64"},
            {2, 128, 64, "k4s_bn64"},  {0, 128, 128, "i8_bn128"}, {3, 128, 128, "i8r_bn128"},
            {1, 128, 128, "k3z_bn128"}, {2, 128, 128, "k4s_bn128"}, {0, 128, 256, "i8_bn256"}};
  return {{0, 64, 64, "i8_bn64"}, {0, 64, 128, "i8_bn128"}, {1, 64, 64, "k3h_bn64"}};
}
// The split of an arm's route: 100 k3z, 104 k4s, 90 k3h.
static int fam_of(int route) { return g_sm == 90 ? 90 : route == 2 ? 104 : 100; }
static bool has_planes(int route) { return route == 1 || route == 2; }
static bool can_pipe(const Arm& a) { return g_sm == 100 && a.route != 0; }

template <int R, bool FOLD, bool WRITE, bool RES, bool PIPE>
static void launch100(int bn, dim3 grid, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed,
                      int32_t* C, uint32_t* Rc, cudaStream_t st) {
  if (bn == 64) eg100<R, 64, FOLD, WRITE, RES, PIPE><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, Rc);
  else if (bn == 128) eg100<R, 128, FOLD, WRITE, RES, PIPE><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, Rc);
  else if constexpr (R == 0) eg100<R, 256, FOLD, WRITE, RES, PIPE><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, Rc);
}

template <bool FOLD, bool WRITE, bool RES, bool PIPE = false>
static void launch(const Arm& a, const uint8_t* A, const uint8_t* B, int m, int n, int k, uint32_t seed, int32_t* C,
                   uint32_t* R, cudaStream_t st) {
  const dim3 grid(n / a.bn, m / a.bm);
  if (g_sm == 100) {
    if constexpr (!PIPE) {
      if (a.route == 0) launch100<0, FOLD, WRITE, RES, false>(a.bn, grid, A, B, m, n, k, seed, C, R, st);
    }
    if (a.route == 1) launch100<1, FOLD, WRITE, RES, PIPE>(a.bn, grid, A, B, m, n, k, seed, C, R, st);
    else if (a.route == 2) launch100<2, FOLD, WRITE, RES, PIPE>(a.bn, grid, A, B, m, n, k, seed, C, R, st);
    else if (a.route == 3) launch100<3, FOLD, WRITE, RES, PIPE>(a.bn, grid, A, B, m, n, k, seed, C, R, st);
  } else {
    if (a.route == 0 && a.bn == 64) eg90<0, 64, FOLD, WRITE, RES><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, R);
    else if (a.route == 0) eg90<0, 128, FOLD, WRITE, RES><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, R);
    else eg90<1, 64, FOLD, WRITE, RES><<<grid, 128, 0, st>>>(A, B, m, n, k, seed, C, R);
  }
  CK(cudaGetLastError());
}
static void launch_split(int fam, int side, const int8_t* src, uint8_t* dst, size_t n, cudaStream_t st) {
  const int blocks = (int)std::min<size_t>((n / 4 + 255) / 256, 148 * 32);
  if (fam == 100 && side == 0) split<100, 0><<<blocks, 256, 0, st>>>(src, dst, n);
  else if (fam == 100) split<100, 1><<<blocks, 256, 0, st>>>(src, dst, n);
  else if (fam == 104 && side == 0) split<104, 0><<<blocks, 256, 0, st>>>(src, dst, n);
  else if (fam == 104) split<104, 1><<<blocks, 256, 0, st>>>(src, dst, n);
  else if (side == 0) split<90, 0><<<blocks, 256, 0, st>>>(src, dst, n);
  else split<90, 1><<<blocks, 256, 0, st>>>(src, dst, n);
  CK(cudaGetLastError());
}

static int8_t code(uint32_t seed, uint32_t which, uint32_t i, bool edge) {
  const uint32_t h = dc_mix32(dc_mix32(seed ^ (which << 28)) + i);
  if (!edge) return (int8_t)((int)(h % 255u) - 127);
  static const int8_t E[8] = {-128, 127, 119, 120, -120, -121, 0, 1};
  return E[h % 8u];
}
// dgemm's host_ref with the tile width a parameter: the receipt by its definition, one chain at a time.
static void host_ref(int m, int n, int k, int bn, uint32_t seed, const std::vector<int8_t>& X,
                     const std::vector<int8_t>& W, std::vector<int32_t>& C, std::vector<uint32_t>& R) {
  const int tiles_n = n / bn;
  C.assign((size_t)m * n, 0);
  R.assign((size_t)m * tiles_n * NCH, 0);
  std::vector<int32_t> T(n);
  for (int i = 0; i < m; i++) {
    std::vector<DuoSt> ch(tiles_n * NCH);
    for (int tn = 0; tn < tiles_n; tn++)
      for (int c = 0; c < NCH; c++) ch[tn * NCH + c] = duo_init(seed, (uint32_t)((i * tiles_n + tn) * NCH + c));
    std::fill(T.begin(), T.end(), 0);
    for (int s = 0; s < k / L; s++) {
      for (int j = 0; j < n; j++) {
        int32_t acc = 0;
        for (int kk = s * L; kk < (s + 1) * L; kk++) acc += (int32_t)X[(size_t)i * k + kk] * W[(size_t)j * k + kk];
        T[j] = (int32_t)((uint32_t)T[j] + (uint32_t)acc);
      }
      for (int tn = 0; tn < tiles_n; tn++)
        for (int jl = 0; jl < bn; jl++) {
          DuoSt& st = ch[tn * NCH + jl % NCH];
          const uint32_t p = (uint32_t)T[tn * bn + jl];
          const int t = (jl / 16) % 16;
          const uint32_t u = st.y ^ p ^ duo_xor(t);
          st.y = u * duo_mul(t) + st.z;
          st.z = dc_rotl(u, duo_rot(t)) + st.y;
        }
    }
    for (int j = 0; j < n; j++) C[(size_t)i * n + j] = T[j];
    for (int q = 0; q < tiles_n * NCH; q++) R[(size_t)i * tiles_n * NCH + q] = duo_finish(ch[q]);
  }
}
static std::vector<uint8_t> host_split(int fam, int side, const std::vector<int8_t>& x) {
  std::vector<uint8_t> o(3 * x.size());
  for (int p = 0; p < 3; p++)
    for (size_t i = 0; i < x.size(); i++) o[p * x.size() + i] = e4m3_int(plane_val(fam, side, p, x[i]));
  return o;
}
static double e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 15, mi = b & 7;
  const double v = e == 0 ? mi / 512.0 : (1 + mi / 8.0) * (double)(1 << e) / 128.0;
  return b & 0x80 ? -v : v;
}

// Host only: every limb value round-trips through e4m3_int, and for every x, y in [-128, 127] the planes of each family
// recombine to xy (k3z: 15 (p0 p0' + p2 p2') + 16 p1 p1'; k4s: (p0 + p1)(p0' + p1'); k3h: 240 p0 p0' + 16 p1 p1' -
// 15 p2 p2').
static int run_selftest() {
  long bad = 0, n = 0;
  for (int fam : {90, 100, 104})
    for (int x = -128; x < 128; x++)
      for (int side = 0; side < 2; side++)
        for (int p = 0; p < 3; p++) {
          const int v = plane_val(fam, side, p, x);
          bad += e4m3_value(e4m3_int(v)) != (double)v;
          n++;
        }
  long id_bad = 0;
  for (int fam : {90, 100, 104})
    for (int x = -128; x < 128; x++)
      for (int y = -128; y < 128; y++) {
        int a[3], b[3];
        for (int p = 0; p < 3; p++) a[p] = plane_val(fam, 0, p, x), b[p] = plane_val(fam, 1, p, y);
        const long v = fam == 100   ? 15L * (a[0] * b[0] + a[2] * b[2]) + 16L * a[1] * b[1]
                       : fam == 104 ? (long)(a[0] + a[1]) * (b[0] + b[1])
                                    : 240L * a[0] * b[0] + 16L * a[1] * b[1] - 15L * a[2] * b[2];
        id_bad += v != (long)x * y;
      }
  printf("{\"mode\": \"selftest\", \"limb_values\": %ld, \"encode_bad\": %ld, \"identity_pairs\": %d, \"identity_bad\": %ld}\n",
         n, bad, 3 * 256 * 256, id_bad);
  return bad == 0 && id_bad == 0 ? 0 : 4;
}

struct Dev {
  int8_t *X = nullptr, *W = nullptr;
  uint8_t *Xp[3] = {}, *Wp[3] = {};  // by route; [0] unused
  int32_t* C = nullptr;
  uint32_t* R = nullptr;
  Dev(int m, int n, int k) {
    CK(cudaMalloc(&X, (size_t)m * k));
    CK(cudaMalloc(&W, (size_t)n * k));
    for (int r = 1; r < 3; r++) {
      CK(cudaMalloc(&Xp[r], 3 * (size_t)m * k));
      CK(cudaMalloc(&Wp[r], 3 * (size_t)n * k));
    }
    CK(cudaMalloc(&C, (size_t)m * n * 4));
    CK(cudaMalloc(&R, (size_t)m * (n / 64) * NCH * 4));
  }
  void split_all(size_t nx, size_t nw) {
    for (int r = 1; r < (g_sm == 100 ? 3 : 2); r++) {
      launch_split(fam_of(r), 0, X, Xp[r], nx, 0);
      launch_split(fam_of(r), 1, W, Wp[r], nw, 0);
    }
  }
  const uint8_t* a(const Arm& arm) const {
    return has_planes(arm.route) ? Xp[arm.route] : reinterpret_cast<const uint8_t*>(X);
  }
  const uint8_t* b(const Arm& arm) const {
    return has_planes(arm.route) ? Wp[arm.route] : reinterpret_cast<const uint8_t*>(W);
  }
};

static int run_gate(int m, int n, int k, uint32_t seed, bool edge) {
  for (const Arm& a : arms())
    if (m % a.bm || n % a.bn || k % L) {
      fprintf(stderr, "ERR shape %dx%dx%d: m a multiple of %d, n of %d, k of %d\n", m, n, k, a.bm, a.bn, L);
      return 2;
    }
  std::vector<int8_t> X((size_t)m * k), W((size_t)n * k);
  for (size_t i = 0; i < X.size(); i++) X[i] = code(seed, 1, (uint32_t)i, edge);
  for (size_t i = 0; i < W.size(); i++) W[i] = code(seed, 2, (uint32_t)i, edge);
  Dev d(m, n, k);
  CK(cudaMemcpy(d.X, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.W, W.data(), W.size(), cudaMemcpyHostToDevice));
  d.split_all(X.size(), W.size());
  CK(cudaDeviceSynchronize());
  size_t split_bad = 0;
  for (int r = 1; r < (g_sm == 100 ? 3 : 2); r++) {
    std::vector<uint8_t> xp(3 * X.size()), wp(3 * W.size());
    CK(cudaMemcpy(xp.data(), d.Xp[r], xp.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(wp.data(), d.Wp[r], wp.size(), cudaMemcpyDeviceToHost));
    split_bad += (xp != host_split(fam_of(r), 0, X)) + (wp != host_split(fam_of(r), 1, W));
  }
  bool all = split_bad == 0;
  // PIPE runs resident: every step multiplies k-slice 0 again, which is the product of operands whose every k-slice is
  // slice 0, so those operands' host reference is PIPE's.
  std::vector<int8_t> Xr(X.size()), Wr(W.size());
  for (size_t i = 0; i < X.size(); i++) Xr[i] = X[i / k * k + i % k % L];
  for (size_t i = 0; i < W.size(); i++) Wr[i] = W[i / k * k + i % k % L];
  std::vector<int32_t> Cr, C((size_t)m * n);  // arms are ordered by tile, so each width's reference is computed once
  std::vector<uint32_t> Rr, R;
  for (int pipe = 0; pipe < 2; pipe++) {
    int last_bn = 0;
    for (const Arm& a : arms()) {
      if (pipe && !can_pipe(a)) continue;
      if (a.bn != last_bn) host_ref(m, n, k, a.bn, seed, pipe ? Xr : X, pipe ? Wr : W, Cr, Rr), last_bn = a.bn;
      R.resize(Rr.size());
      auto call = [&](bool write) {
        CK(cudaMemset(d.C, 0xA5, C.size() * 4));
        CK(cudaMemset(d.R, 0xA5, R.size() * 4));
        if (pipe && write) launch<true, true, true, true>(a, d.a(a), d.b(a), m, n, k, seed, d.C, d.R, 0);
        else if (pipe) launch<true, false, true, true>(a, d.a(a), d.b(a), m, n, k, seed, d.C, d.R, 0);
        else if (write) launch<true, true, false>(a, d.a(a), d.b(a), m, n, k, seed, d.C, d.R, 0);
        else launch<true, false, false>(a, d.a(a), d.b(a), m, n, k, seed, d.C, d.R, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(C.data(), d.C, C.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(R.data(), d.R, R.size() * 4, cudaMemcpyDeviceToHost));
        size_t bc = 0, br = 0;
        for (size_t i = 0; i < C.size(); i++) bc += C[i] != Cr[i];
        for (size_t i = 0; i < R.size(); i++) br += R[i] != Rr[i];
        return std::make_pair(bc, br);
      };
      const auto ctl = call(false);
      const auto got = call(true);
      const bool ok = got.first == 0 && got.second == 0, rejected = ctl.first > 0 && ctl.second > 0;
      all = all && ok && rejected;
      printf("{\"mode\": \"gate\", \"arch\": \"sm_%da\", \"arm\": \"%s%s\", \"dist\": \"%s\", \"m\": %d, \"n\": %d, "
             "\"k\": %d, \"exact\": %s, \"c_bad\": %zu, \"r_bad\": %zu, \"control_rejected\": %s, \"split_bad\": %zu, "
             "\"words\": %zu, \"receipts\": %zu}\n",
             g_sm, a.name, pipe ? "_pipe" : "", edge ? "edge" : "uniform", m, n, k, ok ? "true" : "false", got.first,
             got.second, rejected ? "true" : "false", split_bad, C.size(), R.size());
    }
  }
  return all ? 0 : 4;
}

static int run_bench(int reps, int m, int n, int k) {
  for (const Arm& a : arms())
    if (m % a.bm || n % a.bn || k % L) {
      fprintf(stderr, "ERR shape %dx%dx%d: m a multiple of %d, n of %d, k of %d\n", m, n, k, a.bm, a.bn, L);
      return 2;
    }
  std::vector<int8_t> X((size_t)m * k), W((size_t)n * k);
  for (size_t i = 0; i < X.size(); i++) X[i] = code(7, 1, (uint32_t)i, false);
  for (size_t i = 0; i < W.size(); i++) W[i] = code(7, 2, (uint32_t)i, false);
  Dev d(m, n, k);
  CK(cudaMemcpy(d.X, X.data(), X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.W, W.data(), W.size(), cudaMemcpyHostToDevice));
  d.split_all(X.size(), W.size());
  struct Run {
    std::string name;
    int arm;  // >= 0 an arm; -1 - r the split of route r's X, -11 - r its W
    bool fold, res, pipe;
  };
  std::vector<Run> runs;
  const auto as = arms();
  for (int mode = 0; mode < 3; mode++)
    for (int i = 0; i < (int)as.size(); i++) {
      if (mode == 2 && !can_pipe(as[i])) continue;
      const std::string sfx = mode == 0 ? "" : mode == 1 ? "_res" : "_pipe";
      runs.push_back({std::string(as[i].name) + "_fold" + sfx, i, true, mode > 0, mode == 2});
      runs.push_back({std::string(as[i].name) + "_nofold" + sfx, i, false, mode > 0, mode == 2});
    }
  for (int r = 1; r < (g_sm == 100 ? 3 : 2); r++) {
    const std::string sfx = r == 1 ? "" : "_k4s";
    runs.push_back({"split_x" + sfx, -1 - r, false, false, false});
    runs.push_back({"split_w" + sfx, -11 - r, false, false, false});
  }
  auto go = [&](const Run& r) {
    if (r.arm <= -11) launch_split(fam_of(-11 - r.arm), 1, d.W, d.Wp[-11 - r.arm], W.size(), 0);
    else if (r.arm < 0) launch_split(fam_of(-1 - r.arm), 0, d.X, d.Xp[-1 - r.arm], X.size(), 0);
    else {
      const Arm& a = as[r.arm];
      if (r.pipe && r.fold) launch<true, true, true, true>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
      else if (r.pipe) launch<false, true, true, true>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
      else if (r.fold && r.res) launch<true, true, true>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
      else if (r.fold) launch<true, true, false>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
      else if (r.res) launch<false, true, true>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
      else launch<false, true, false>(a, d.a(a), d.b(a), m, n, k, 1, d.C, d.R, 0);
    }
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
  int sms = 0, khz = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  printf("{\"mode\": \"bench\", \"arch\": \"sm_%da\", \"m\": %d, \"n\": %d, \"k\": %d, \"reps\": %d, \"sms\": %d, "
         "\"clock_khz\": %d", g_sm, m, n, k, reps, sms, khz);
  for (size_t i = 0; i < runs.size(); i++) {
    std::sort(ms[i].begin(), ms[i].end());
    printf(", \"%s_ms\": %.6f", runs[i].name.c_str(), ms[i][ms[i].size() / 2]);
  }
  printf("}\n");
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "selftest") return run_selftest();
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sm = p.major == 10 ? 100 : p.major == 9 ? 90 : 0;
  if (!g_sm) {
    fprintf(stderr, "ERR device sm_%d%d: this binary runs on sm_90a or sm_100a\n", p.major, p.minor);
    return 2;
  }
  if (mode == "gate" && argc == 7) {
    const std::string dist = argv[6];
    if (dist != "uniform" && dist != "edge") {
      fprintf(stderr, "ERR dist %s: uniform or edge\n", dist.c_str());
      return 2;
    }
    return run_gate(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), (uint32_t)strtoul(argv[5], 0, 10), dist == "edge");
  }
  if (mode == "bench" && argc == 6) return run_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  fprintf(stderr, "usage: egemm selftest | gate M N K SEED uniform|edge | bench REPS M N K\n");
  return 2;
}
