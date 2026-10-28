// A SHA-512 hash warpgroup beside the exact s8 linear's MMA warpgroups in one CTA (sm_90a), its registers rebalanced by
// setmaxnreg: does hashing the finished O tiles from shared memory slow the tensor pipe, and does the hash keep up?
//
// The GEMM is s8_wgmma.cu's (benchmarks/torch_gpu/s8_linear: TMA boxes of 128 bytes of depth with SWIZZLE_128B into STAGES
// stages counted on mbarriers, wgmma s8 into one s32 accumulator over all of K, the Definition's epilogue to bf16) at its
// 128 x 256 tile, with one change of order: CTA c takes the tile rows c, c + grid, ... and each row's column tiles left to
// right, so a hash thread that owns a row of the panel chains its sha512/row/v2 digest (rows.cuh's row_digest: P(8 nbytes)'s
// midstate, the row's chunks, the padding chunk's rounds alone) through the panel's tiles in order. The epilogue stores the
// bf16 words to y, as s8_wgmma.cu's does, and to the O tile in shared memory, which is either (OT 1) the whole tile, rows
// of 512 bytes 528 apart, or (OT 2) half of it at a time, rows of 256 bytes with their 16-byte units XOR-swizzled by the
// row (unit u at u ^ (row % 8)): 32 KiB less, the room for a fourth stage. Both keep the epilogue's 4-byte stores and the
// hash's 16-byte loads free of bank conflicts.
//
// Kernels (KERNS): the producer warpgroup, NM MMA warpgroups (a CTA tile of 64 NM rows by 256 columns), NH hash warpgroups,
// the MMA form, the stages and the registers each role holds after setmaxnreg (P, Y, H). An SM's four register files of
// 16,384 each hold one warp of every warpgroup, so P + NM Y + NH H <= the launch's registers x (1 + NM + NH): 504 at 3
// warpgroups (168), 512 at 4 (128), 480 at 5 (96) and 6 (80).
//   wgl_*       NM 2, NH 0: the producer at 40, two MMA warpgroups at 232; wgmma.linear's shape at 4 stages (no O tile),
//               and at 3 with the whole O tile, one m64n256k32 an atom or two m64n128k32
//   h<H>_<v>    NM 2, NH 1: the producer at P, two MMA warpgroups at Y, one hash warpgroup at H. The MMA is two
//               m64n128k32 an atom: ptxas refuses m64n256k32 below a 154-register launch (C7602, beside_probe.cu). 3
//               stages and the whole O tile, or (_s4) 4 stages and the half
//   m1h<NH>_<v> NM 1, NH 2 to 4: one MMA warpgroup (a 64-row tile, so twice the CTA tiles), the hash warpgroups at H,
//               two m64n128k32 an atom at 4 warpgroups and four m64n64k32 at 5 or 6; hash thread g hashes the O tile's
//               row g % 64
//   m2h2_<v>    NM 2, NH 2: 5 warpgroups, four m64n64k32 an atom, the MMA warpgroups at 152
// v: the hash's compression, lowreg.cu's r (sx::compress<0, 1>) or rs (lowreg.cuh's rounds, the chaining value in shared
// memory, 8 KiB a hash warpgroup). _sus: every mbarrier wait with a suspend-time hint (SUS_BIT), so the producer's and
// the MMA warpgroups' waits park rather than reissue try_wait in the hash warps' issue slots. _early, _pre: in the
// handoff the hash frees the O buffer once it has loaded a fill's last chunk, or all of the fill's chunks (the half
// tile's two, 32 registers more: H 136), rather than once it has compressed them, so the epilogue waits less.
// Arms (a launch's mode): mma (the hash warpgroups exit at once), hash (the GEMM's warpgroups exit; each hash thread hashes
// its seed row's bytes in the O buffer, the whole row or its first half, as many times as make the CTA's O tiles' bytes),
// free (both: the hash passes over the O buffer until the GEMM ends, so the tensor pipe's time beside a busy ALU is the
// GEMM's), handoff (NM 2, NH 1 only; the pipeline: the hash takes each O tile, or half tile, from the epilogue, one
// buffer, the epilogue waiting until the hash has read the last).
//
//   beside list              the kernels' names
//   beside gate M N K DIR [KERNEL...]
//                            each kernel's (default every kernel's) handoff and hash arms once: y at sampled cells against
//                            the host's Definition, and y, the digests and the seed tile to DIR for beside_check.py (hashlib)
//   beside time M N K LAUNCHES [KERNEL...]
//                            each kernel's arms in turn, LAUNCHES each: JSON lines (ms, the GEMM's and the hash's spans from
//                            %globaltimer, compressions, TOPS, the SM clock from NVML)
//   beside one M N K KERNEL ARM LAUNCHES    one kernel's arm (for Nsight Compute)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../s8_linear/s8_acc.h"
#include "lowreg.cuh"

using u64 = uint64_t;
using u32 = uint32_t;

#define DEV __device__ __forceinline__
#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

// ---------------------------------------------------------------- mbarriers, TMA, wgmma (s8_wgmma.cu's)
DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
// SUS: try_wait with a suspend-time hint (CUTLASS's ClusterBarrier::wait, 10 ms), so a waiting warp parks until the phase
// completes rather than reissuing try_wait, and leaves its scheduler's issue slots to the hash warp beside it
constexpr uint32_t SUSPEND_NS = 0x989680;
// a hash kernel's VAR: 0 r or 3 rs, with SUS_BIT, and in the handoff EARLY_BIT (the hash frees the O buffer after loading
// a fill's last chunk rather than after compressing it) or PRE_BIT (after loading every chunk of the fill into registers)
constexpr int SUS_BIT = 16, EARLY_BIT = 32, PRE_BIT = 64;
template <bool SUS = false> DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  if constexpr (SUS)
    asm volatile(
        "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra WAIT;\n}\n" ::"r"(bar),
        "r"(parity), "r"(SUSPEND_NS)
        : "memory");
  else
    asm volatile(
        "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}
template <bool SUS = false> DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait<SUS>(bar, parity);
  __syncwarp();
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// accumulator i of thread x is row 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else if constexpr (N == 128)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " ACC32_S
                 ", %32, %33, p;\n}" : ACC32_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}
DEV uint32_t words(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}
DEV unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------- the kernel
struct alignas(128) Params {
  CUtensorMap a, b;           // X: M rows of K bytes, boxes of 64 NM rows; W: N rows, boxes of 256 rows
  const float* sa;            // s_a [M]
  const float* sw;            // s_w [N]
  uint16_t* y;                // y [M, N], bf16 words
  u64* digests;               // [M, 8] (handoff: y's rows; otherwise a CTA's 128 NH slots, one a hash thread)
  const u64* mid;             // P(8 nbytes)'s midstate, a row of nbytes = 2 N (handoff) or 512 hash_tiles
  const u64* kw;              // that row's padding chunk's K + W
  const uint4* seed;          // the O tile's first bytes: 64 NM rows of 512
  unsigned long long* times;  // [grid][TSLOTS]: the GEMM's first and last %globaltimer, the hash's, its passes a warp
  int M, N, K, mode, hash_tiles, pad;
};
enum { MMA = 1, HASH = 2, HANDOFF = 4 };
constexpr int BK = 128, BN = 256, NA = 128, ROWB = 2 * BN, TSTRIDE = ROWB + 16, HROWB = ROWB / 2, MAX_NH = 4;
constexpr int TSLOTS = 4 + 4 * MAX_NH;

template <int NM, int NH> struct Cta {
  static constexpr int NWG = 1 + NM + NH, TMC = 64 * NM, LAUNCH = 65536 / (128 * NWG) / 8 * 8;
};

template <int NM, int NH, int STAGES, int OT, int VAR> struct Layout {
  static constexpr int TMC = Cta<NM, NH>::TMC, A_BYTES = TMC * BK, STAGE = (TMC + BN) * BK;
  static constexpr int O_OFF = STAGES * STAGE, S_OFF = O_OFF + (OT == 1 ? TMC * TSTRIDE : OT == 2 ? TMC * HROWB : 0);
  static constexpr int BAR_OFF = S_OFF + ((VAR & 3) == 3 ? NH * 8 * 8 * 128 : 0);
  static constexpr int SMEM = 1024 + BAR_OFF + 8 * (2 * STAGES + 2) + 8;
};

// a hash thread's row: VAR 0 sx::compress<0, 1>, all in registers; 3 lowreg.cuh's rounds, the chaining value at Ss[128 i]
// (its warpgroup's 8 KiB, the thread's column)
template <int VAR> struct Row {
  static_assert(VAR == 0 || VAR == 3, "r or rs");
  u64 st[8];
  u64* Ss;
  DEV void init(const u64* mid) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (VAR == 3)
        Ss[i * 128] = mid[i];
      else
        st[i] = mid[i];
    }
  }
  DEV void chunk(const uint8_t* p, int sw) {
    u64 w[16];
    sx::load_chunk(p, w, sw);
    compress(w);
  }
  DEV void compress(u64 (&w)[16]) {
    if constexpr (VAR == 0) {
      sx::compress<0, 1>(st, w);
    } else {
      u64 v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = Ss[i * 128];
      asm volatile("" ::: "memory");
      sx::rounds<0, 1>(v, w);
#pragma unroll
      for (int i = 0; i < 8; ++i) Ss[i * 128] += v[i];
    }
  }
  DEV void finish(const u64* kw, u64* out) {
    u64 s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = VAR == 3 ? Ss[i * 128] : st[i];
    sx::compress_kw<0, 1>(s, kw);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s[i];
  }
};

template <int R, int LAUNCH> DEV void regs_to() {
  if constexpr (R < LAUNCH) regs_dec<R>();
  if constexpr (R > LAUNCH) regs_inc<R>();
}

template <int NM, int NH, int NFORM, int STAGES, int OT, int P, int Y, int H, int VAR>
DEV void body(const Params& p) {
  using C = Cta<NM, NH>;
  using L = Layout<NM, NH, STAGES, OT, VAR>;
  constexpr int NWG = C::NWG, TMC = C::TMC;
  static_assert(NM == 1 || NM == 2, "a CTA tile of 64 or 128 rows");
  static_assert(NH <= MAX_NH && (NH == 0 || OT), "the hash warpgroups read the O tile");
  static_assert(NFORM == 64 || NFORM == 128 || (NFORM == 256 && C::LAUNCH >= 154), "ptxas refuses n256 below 154 registers");
  static_assert(P + NM * Y + NH * H <= NWG * C::LAUNCH, "the warpgroups' registers exceed the launch's");
  static_assert(L::SMEM <= 232448, "shared memory");
  constexpr bool CAN_HANDOFF = NM == 2 && NH == 1, SUS = VAR & SUS_BIT, EARLY = VAR & EARLY_BIT, PRE = VAR & PRE_BIT;
  constexpr int HV = VAR & 3;
  static_assert(!PRE || OT == 2, "pre holds a fill's chunks in registers: the half tile's two");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t s0 = smem_at(smem), base = (s0 + 1023) & ~1023u;
  uint8_t* gb = smem + (base - s0);
  const uint32_t full0 = base + L::BAR_OFF, empty0 = full0 + 8 * STAGES, tfull = empty0 + 8 * STAGES, tempty = tfull + 8;
  u32* done = reinterpret_cast<u32*>(gb + L::BAR_OFF + 8 * (2 * STAGES + 2));
  uint8_t* otile = gb + L::O_OFF;
  const int tiles_m = p.M / TMC, tiles_n = p.N / BN, KB = p.K / BK;
  const int tid = threadIdx.x, wg = tid / 128, x = tid % 128;
  const bool mma_on = p.mode & MMA, hash_on = NH > 0 && (p.mode & HASH);
  const bool handoff = CAN_HANDOFF && mma_on && hash_on && (p.mode & HANDOFF);
  unsigned long long* tout = p.times + (size_t)blockIdx.x * TSLOTS;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4 * NM);
    }
    mbar_init(tfull, 128 * NM);
    mbar_init(tempty, 128);
    *done = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the O buffer's row stride, and the hash's chunks a fill (a whole row's 4 or a half row's 2)
  constexpr int RB = OT == 2 ? HROWB : TSTRIDE, CH = OT == 2 ? 2 : 4;
  if constexpr (OT == 1)
    for (int i = tid; i < TMC * ROWB / 16; i += 128 * NWG)
      *reinterpret_cast<uint4*>(otile + (i / (ROWB / 16)) * TSTRIDE + 16 * (i % (ROWB / 16))) = p.seed[i];
  if constexpr (OT == 2)
    for (int i = tid; i < TMC * HROWB / 16; i += 128 * NWG) {
      const int r = i / (HROWB / 16), u = i % (HROWB / 16);
      *reinterpret_cast<uint4*>(otile + r * HROWB + 16 * (u ^ (r % 8))) = p.seed[r * (ROWB / 16) + u];
    }
  __syncthreads();

  if (wg == 0) {
    // the producer exits rather than returns: a path back would join its register budget with the others'
    regs_to<P, C::LAUNCH>();
    if (tid == 0 && mma_on) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      int s = 0;
      uint32_t ph = 0;
      for (int tm = blockIdx.x; tm < tiles_m; tm += gridDim.x)
        for (int tn = 0; tn < tiles_n; tn++)
          for (int kb = 0; kb < KB; kb++) {
            const uint32_t st = base + s * L::STAGE, full = full0 + 8 * s;
            mbar_wait<SUS>(empty0 + 8 * s, ph ^ 1);
            mbar_expect_tx(full, L::STAGE);
            tma_load(st, &p.a, full, kb * BK, tm * TMC);
            tma_load(st + L::A_BYTES, &p.b, full, kb * BK, tn * BN);
            if (++s == STAGES) { s = 0; ph ^= 1; }
          }
    }
    asm volatile("exit;");
  }

  if constexpr (NH > 0) {
    if (wg > NM) {
      regs_to<H, C::LAUNCH>();
      if (!hash_on) asm volatile("exit;");
      const int h = wg - 1 - NM, g = 128 * h + x;
      Row<HV> r;
      r.Ss = reinterpret_cast<u64*>(gb + L::S_OFF) + 1024 * h + x;
      const int hr = g % TMC, sw = OT == 2 ? hr % 8 : 0;
      const uint8_t* row = otile + hr * RB;
      u32 passes = 0;
      if (g == 0) tout[2] = gtime();
      if (handoff) {
        int j = 0;
        for (int tm = blockIdx.x; tm < tiles_m; tm += gridDim.x) {
          r.init(p.mid);
          for (int f = 0; f < tiles_n * (4 / CH); f++, j++) {
            mbar_wait<SUS>(tfull, j & 1);
            if constexpr (PRE) {
              u64 w[CH][16];
#pragma unroll
              for (int c = 0; c < CH; c++) sx::load_chunk(row + 128 * c, w[c], sw);
              mbar_arrive(tempty);
#pragma unroll
              for (int c = 0; c < CH; c++) r.compress(w[c]);
            } else if constexpr (EARLY) {
#pragma unroll 1
              for (int c = 0; c < CH - 1; c++) r.chunk(row + 128 * c, sw);
              u64 w[16];
              sx::load_chunk(row + 128 * (CH - 1), w, sw);
              mbar_arrive(tempty);
              r.compress(w);
            } else {
#pragma unroll 1
              for (int c = 0; c < CH; c++) r.chunk(row + 128 * c, sw);
              mbar_arrive(tempty);
            }
          }
          r.finish(p.kw, p.digests + (size_t)(tm * TMC + x) * 8);
          passes += tiles_n * (4 / CH);
        }
      } else {
        r.init(p.mid);
        if (mma_on) {
          while (*reinterpret_cast<volatile u32*>(done) < 4 * NM) {
#pragma unroll 1
            for (int c = 0; c < CH; c++) r.chunk(row + 128 * c, sw);
            passes++;
          }
        } else {
#pragma unroll 1
          for (int k = 0; k < p.hash_tiles * (4 / CH); k++) {
#pragma unroll 1
            for (int c = 0; c < CH; c++) r.chunk(row + 128 * c, sw);
          }
          passes = p.hash_tiles * (4 / CH);
        }
        r.finish(p.kw, p.digests + ((size_t)blockIdx.x * 128 * NH + g) * 8);
      }
      if (x == 0) atomicMax(tout + 3, gtime());
      if (x % 32 == 0) tout[4 + g / 32] = passes;
      asm volatile("exit;");
    }
  }

  regs_to<Y, C::LAUNCH>();
  const int wl = wg - 1, lane = tid % 32;
  if (x == 0 && wl == 0) tout[0] = gtime();
  int s = 0, j = 0;
  uint32_t ph = 0;
  int32_t d[NA];
  if (mma_on)
    for (int tm = blockIdx.x; tm < tiles_m; tm += gridDim.x)
      for (int tn = 0; tn < tiles_n; tn++) {
        for (int kb = 0; kb < KB; kb++) {
          mbar_wait_warp<SUS>(full0 + 8 * s, ph);
          const uint32_t a0 = base + s * L::STAGE + wl * 64 * BK, b0 = base + s * L::STAGE + L::A_BYTES;
          WG_FENCE;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const uint32_t sd = kb > 0 || i > 0;
#pragma unroll
            for (int q = 0; q < BN / NFORM; q++)
              imma<NFORM>(d + NFORM / 2 * q, desc_sw128(a0 + 32 * i), desc_sw128(b0 + NFORM * q * BK + 32 * i), sd);
          }
          WG_COMMIT;
          WG_WAIT(1);
          if (kb > 0 && lane == 0) mbar_arrive(empty0 + 8 * (s == 0 ? STAGES - 1 : s - 1));
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
        WG_WAIT(0);
        acc_fence<NA>(d);
        if (lane == 0) mbar_arrive(empty0 + 8 * (s == 0 ? STAGES - 1 : s - 1));
        // n128 x 2 puts columns 128.. in d[64 ..] and n64 x 4 columns 64 q.. in d[32 q ..]: accumulator 4 q + i is column
        // 8 q + 2 (x % 4) + i % 2 in every form
        // the bases hold the tile's and the lane's columns, so each q's offset is an immediate: offsets the compiler could
        // hoist out of the tile loop would take 64 registers beside the accumulators
        const int rl = wl * 64 + 16 * (x / 32) + (x % 32) / 4, row = tm * TMC + rl, c0 = tn * BN + 2 * (x % 4);
        const float r0 = p.sa[row], r1 = p.sa[row + 8];
        const float* sw = p.sw + c0;
        uint32_t* y0 = reinterpret_cast<uint32_t*>(p.y + (size_t)row * p.N + c0);
        uint32_t* y1 = reinterpret_cast<uint32_t*>(p.y + (size_t)(row + 8) * p.N + c0);
        uint32_t* o0 = reinterpret_cast<uint32_t*>(otile + rl * RB + 4 * (x % 4));
        // the half buffer's rows rl and rl + 8 share their swizzle, rl % 8
        const uint32_t ob = smem_at(otile) + rl * RB + 4 * (x % 4), s8 = (x % 32) / 4;
#pragma unroll
        for (int f = 0; f < 4 / CH; f++, j++) {
          if (handoff) mbar_wait<SUS>(tempty, (j & 1) ^ 1);
#pragma unroll
          for (int u = 0; u < NA / 4 * CH / 4; u++) {
            const int q = f * (NA / 4 * CH / 4) + u;
            const float2 cw = *reinterpret_cast<const float2*>(sw + 8 * q);
            const uint32_t w0 = words(d[4 * q], d[4 * q + 1], r0, cw.x, cw.y);
            const uint32_t w1 = words(d[4 * q + 2], d[4 * q + 3], r1, cw.x, cw.y);
            y0[4 * q] = w0;
            y1[4 * q] = w1;
            if constexpr (OT == 1) {
              o0[4 * q] = w0;
              o0[4 * q + 8 * TSTRIDE / 4] = w1;
            }
            if constexpr (OT == 2) {
              const uint32_t a = ob + 16 * (u ^ s8);
              asm volatile("st.shared.u32 [%0], %1;\n st.shared.u32 [%0+%2], %3;" ::"r"(a), "r"(w0), "n"(8 * HROWB),
                           "r"(w1)
                           : "memory");
            }
          }
          if (handoff) mbar_arrive(tfull);
        }
      }
  if (x == 0 && wl == 0) tout[1] = gtime();
  if (lane == 0) atomicAdd(done, 1u);
}

template <int NM, int NH, int NFORM, int STAGES, int OT, int P, int Y, int H, int VAR>
__global__ void __launch_bounds__(128 * Cta<NM, NH>::NWG, 1) beside(const __grid_constant__ Params p) {
  body<NM, NH, NFORM, STAGES, OT, P, Y, H, VAR>(p);
}

typedef void (*KFn)(const Params);
struct Kern {
  const char* name;
  int nm, nh, nform, stages, ot, P, Y, H, var, smem;
  KFn f;
  int nwg() const { return 1 + nm + nh; }
  int tmc() const { return 64 * nm; }
  bool has(int arm) const { return nh == 0 ? arm == 0 : nm == 2 && nh == 1 ? true : arm != 3; }
};
#define KERN(NAME, NM, NH, NFORM, STAGES, OT, P, Y, H, VAR)                                                 \
  {NAME, NM, NH, NFORM, STAGES, OT, P, Y, H, VAR, Layout<NM, NH, STAGES, OT, VAR>::SMEM,                    \
   beside<NM, NH, NFORM, STAGES, OT, P, Y, H, VAR>}
static const Kern KERNS[] = {
    KERN("wgl_n256_s4", 2, 0, 256, 4, 0, 40, 232, 0, 0), KERN("wgl_n256_s3", 2, 0, 256, 3, 1, 40, 232, 0, 0),
    KERN("wgl_n128_s3", 2, 0, 128, 3, 1, 40, 232, 0, 0), KERN("wgl_n128_s4", 2, 0, 128, 4, 0, 40, 232, 0, 0),
    KERN("h64_rs", 2, 1, 128, 3, 1, 24, 208, 64, 3),     KERN("h72_rs", 2, 1, 128, 3, 1, 24, 208, 72, 3),
    KERN("h72_r", 2, 1, 128, 3, 1, 24, 208, 72, 0),      KERN("h80_r", 2, 1, 128, 3, 1, 32, 200, 80, 0),
    KERN("h96_r", 2, 1, 128, 3, 1, 32, 192, 96, 0),      KERN("h128_r", 2, 1, 128, 3, 1, 32, 176, 128, 0),
    KERN("h96_r_s4", 2, 1, 128, 4, 2, 32, 192, 96, 0),   KERN("h128_r_s4", 2, 1, 128, 4, 2, 32, 176, 128, 0),
    KERN("m1h2_r", 1, 2, 128, 4, 1, 24, 232, 128, 0),    KERN("m1h3_r", 1, 3, 64, 4, 1, 24, 240, 72, 0),
    KERN("m1h3_rs", 1, 3, 64, 4, 1, 24, 256, 64, 3),     KERN("m1h4_rs", 1, 4, 64, 4, 1, 24, 200, 64, 3),
    KERN("m2h2_r", 2, 2, 64, 3, 1, 24, 152, 72, 0),      KERN("m2h2_r_s4", 2, 2, 64, 4, 2, 24, 152, 72, 0),
    KERN("h72_rs_sus", 2, 1, 128, 3, 1, 24, 208, 72, 3 | SUS_BIT),
    KERN("h128_r_sus", 2, 1, 128, 3, 1, 32, 176, 128, SUS_BIT),
    KERN("h128_r_s4_sus", 2, 1, 128, 4, 2, 32, 176, 128, SUS_BIT),
    KERN("m1h2_r_sus", 1, 2, 128, 4, 1, 24, 232, 128, SUS_BIT),
    KERN("h128_r_early", 2, 1, 128, 3, 1, 32, 176, 128, EARLY_BIT),
    KERN("h128_r_s4_early", 2, 1, 128, 4, 2, 32, 176, 128, EARLY_BIT),
    KERN("h136_r_s4_pre", 2, 1, 128, 4, 2, 24, 176, 136, PRE_BIT),
};
static const char* const ARMS[] = {"mma", "hash", "free", "handoff"};
static const int ARM_MODE[] = {MMA, HASH, MMA | HASH, MMA | HASH | HANDOFF};

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// a row-major (rows x depth) byte matrix as s8_wgmma.cu loads it (wgmma.py's tmap): UINT8, boxes of box_rows x 128 bytes,
// SWIZZLE_128B, L2 promotion 128B
static void tmap(EncodeTiled enc, CUtensorMap* m, void* ptr, u64 rows, u64 depth, u32 box_rows) {
  const cuuint64_t dims[2] = {depth, rows}, strides[1] = {depth};
  const cuuint32_t box[2] = {128, box_rows}, estr[2] = {1, 1};
  const CUresult r = enc(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", (int)r); exit(2); }
}

static u64 splitmix(u64 seed, u64 i) {
  u64 z = seed + (i + 1) * 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// P(8 nbytes)'s midstate and the padding chunk's K + W (rows.cuh's row_setup), nbytes % 128 == 0
static void row_setup(u64 nbytes, u64* mid, u64* kw) {
  uint8_t pb[128] = {};
  const char* tag = "verity/sha512-row/v2";
  memcpy(pb, tag, strlen(tag) + 1);
  for (int i = 0; i < 8; ++i) pb[strlen(tag) + 1 + i] = (uint8_t)((8 * nbytes) >> (56 - 8 * i));
  u64 w[16];
  for (int j = 0; j < 16; ++j) {
    w[j] = 0;
    for (int b = 0; b < 8; ++b) w[j] = w[j] << 8 | pb[8 * j + b];
  }
  sx::iv(mid);
  sx::compress<0, 1>(mid, w);
  u64 pad[16] = {};
  pad[0] = 0x8000000000000000ull;
  pad[15] = 8 * (128 + nbytes);
  sx::kw_of(pad, kw);
}

static uint16_t bf16_rn(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fff;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  double sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
  double watts() {
    unsigned mw = 0;
    if (ok) nvmlDeviceGetPowerUsage(d, &mw);
    return mw * 1e-3;
  }
};

// ma, dmid_seed and dkw_seed by NM (1 or 2): a CTA tile of 64 NM rows, so its grid, its O tiles and its hash arm's row
struct Run {
  int M, N, K, sms;
  std::vector<int8_t> X, Wt;
  std::vector<float> sa, sw;
  std::vector<uint8_t> seed;
  int8_t *dX, *dW;
  float *dsa, *dsw;
  uint16_t* dy;
  u64 *ddig, *dmid_row, *dkw_row, *dmid_seed[3], *dkw_seed[3];
  uint4* dseed;
  unsigned long long* dtimes;
  CUtensorMap ma[3], mb;
  int grid(int nm) const { return std::min(sms, M / (64 * nm)); }
  int panels(int nm) const { return (M / (64 * nm) + grid(nm) - 1) / grid(nm); }
  size_t slots() const { return std::max((size_t)M, (size_t)sms * 128 * MAX_NH); }
};

static void setup(Run& r, int M, int N, int K) {
  if (M % 128 || N % BN || K % BK) { fprintf(stderr, "M %% 128, N %% 256 and K %% 128 must be 0\n"); exit(2); }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  r.M = M, r.N = N, r.K = K, r.sms = prop.multiProcessorCount;
  r.X.resize((size_t)M * K);
  r.Wt.resize((size_t)N * K);
  for (size_t i = 0; i < r.X.size() / 8; ++i) {
    const u64 v = splitmix(1, i);
    memcpy(&r.X[8 * i], &v, 8);
  }
  for (size_t i = 0; i < r.Wt.size() / 8; ++i) {
    const u64 v = splitmix(2, i);
    memcpy(&r.Wt[8 * i], &v, 8);
  }
  r.sa.resize(M);
  r.sw.resize(N);
  for (int i = 0; i < M; ++i) r.sa[i] = (float)(1 + splitmix(3, i) % 1000) * 1e-5f;
  for (int i = 0; i < N; ++i) r.sw[i] = (float)(1 + splitmix(4, i) % 1000) * 1e-4f;
  r.seed.resize(128 * ROWB);
  for (size_t i = 0; i < r.seed.size() / 8; ++i) {
    const u64 v = splitmix(5, i);
    memcpy(&r.seed[8 * i], &v, 8);
  }
  CK(cudaMalloc(&r.dX, r.X.size()));
  CK(cudaMalloc(&r.dW, r.Wt.size()));
  CK(cudaMalloc(&r.dsa, M * 4));
  CK(cudaMalloc(&r.dsw, N * 4));
  CK(cudaMalloc(&r.dy, (size_t)M * N * 2));
  CK(cudaMalloc(&r.ddig, r.slots() * 64));
  CK(cudaMalloc(&r.dseed, r.seed.size()));
  CK(cudaMalloc(&r.dtimes, (size_t)r.sms * TSLOTS * 8));
  CK(cudaMemcpy(r.dX, r.X.data(), r.X.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dW, r.Wt.data(), r.Wt.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dsa, r.sa.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dsw, r.sw.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dseed, r.seed.data(), r.seed.size(), cudaMemcpyHostToDevice));
  u64 mid[8], kw[80];
  CK(cudaMalloc(&r.dmid_row, 64));
  CK(cudaMalloc(&r.dkw_row, 640));
  row_setup((u64)2 * N, mid, kw);
  CK(cudaMemcpy(r.dmid_row, mid, 64, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dkw_row, kw, 640, cudaMemcpyHostToDevice));
  EncodeTiled enc = nullptr;
  cudaDriverEntryPointQueryResult q;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), 12000, cudaEnableDefault, &q));
  if (!enc || q != cudaDriverEntryPointSuccess) { fprintf(stderr, "no cuTensorMapEncodeTiled\n"); exit(2); }
  for (int nm = 1; nm <= 2; ++nm) {
    CK(cudaMalloc(&r.dmid_seed[nm], 64));
    CK(cudaMalloc(&r.dkw_seed[nm], 640));
    row_setup((u64)ROWB * (N / BN) * r.panels(nm), mid, kw);
    CK(cudaMemcpy(r.dmid_seed[nm], mid, 64, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(r.dkw_seed[nm], kw, 640, cudaMemcpyHostToDevice));
    tmap(enc, &r.ma[nm], r.dX, M, K, 64 * nm);
  }
  tmap(enc, &r.mb, r.dW, N, K, BN);
  printf("{\"sms\": %d, \"device\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, \"grid\": %d}\n", r.sms, prop.name, M, N, K,
         r.grid(2));
}

static Params params(const Kern& k, const Run& r, int arm) {
  Params p;
  memset(&p, 0, sizeof p);
  p.a = r.ma[k.nm], p.b = r.mb;
  p.sa = r.dsa, p.sw = r.dsw, p.y = r.dy, p.digests = r.ddig, p.seed = r.dseed, p.times = r.dtimes;
  const bool rows = ARM_MODE[arm] & HANDOFF;
  p.mid = rows ? r.dmid_row : r.dmid_seed[k.nm];
  p.kw = rows ? r.dkw_row : r.dkw_seed[k.nm];
  p.M = r.M, p.N = r.N, p.K = r.K, p.mode = ARM_MODE[arm];
  p.hash_tiles = (r.N / BN) * r.panels(k.nm);
  return p;
}

static void launch(const Kern& k, const Run& r, const Params& p) {
  k.f<<<r.grid(k.nm), 128 * k.nwg(), k.smem>>>(p);
  CK(cudaGetLastError());
}

static const Kern* find(const char* name) {
  for (const Kern& k : KERNS)
    if (!strcmp(k.name, name)) return &k;
  fprintf(stderr, "no kernel %s\n", name);
  exit(2);
}

static void prepare(const Kern& k) {
  CK(cudaFuncSetAttribute(k.f, cudaFuncAttributeMaxDynamicSharedMemorySize, k.smem));
}

static void write_file(const std::string& path, const void* data, size_t n) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(data, 1, n, f) != n) { fprintf(stderr, "write %s failed\n", path.c_str()); exit(2); }
  fclose(f);
}

// the Definition's word for cell (row, col) on the host
static uint16_t ref_cell(const Run& r, int row, int col) {
  int32_t acc = 0;
  const int8_t* x = &r.X[(size_t)row * r.K];
  const int8_t* w = &r.Wt[(size_t)col * r.K];
  for (int k = 0; k < r.K; ++k) acc += (int32_t)x[k] * (int32_t)w[k];
  volatile float f = (float)acc;
  f = f * r.sa[row];
  f = f * r.sw[col];
  return bf16_rn(f);
}

static int gate(Run& r, const char* dir, const std::vector<const Kern*>& ks) {
  int bad = 0;
  std::vector<uint16_t> y((size_t)r.M * r.N);
  std::vector<u64> dig(r.slots() * 8);
  write_file(std::string(dir) + "/seed.bin", r.seed.data(), r.seed.size());
  for (const Kern* kp : ks) {
    const Kern& k = *kp;
    prepare(k);
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, k.f));
    // the GEMM's y with the handoff's row digests where the kernel has a handoff, else the GEMM alone; then the hash alone
    std::vector<int> arms = {k.has(3) ? 3 : 0};
    if (k.has(1)) arms.push_back(1);
    const int slots = 128 * k.nh, grid = r.grid(k.nm);
    for (int a : arms) {
      CK(cudaMemset(r.dy, 0xff, (size_t)r.M * r.N * 2));
      CK(cudaMemset(r.ddig, 0, dig.size() * 8));
      launch(k, r, params(k, r, a));
      CK(cudaDeviceSynchronize());
      int cells = 0, wrong = 0;
      if (ARM_MODE[a] & MMA) {
        CK(cudaMemcpy(y.data(), r.dy, y.size() * 2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 4096; ++i) {
          const int row = (int)(splitmix(6, i) % r.M), col = (int)(splitmix(7, i) % r.N);
          ++cells;
          wrong += y[(size_t)row * r.N + col] != ref_cell(r, row, col);
        }
        for (int row : {0, r.M - 1})
          for (int col = 0; col < r.N; ++col) ++cells, wrong += y[(size_t)row * r.N + col] != ref_cell(r, row, col);
        bad += wrong != 0;
      }
      const std::string tag = std::string(dir) + "/" + k.name + "-" + ARMS[a];
      if (ARM_MODE[a] & HASH) {
        CK(cudaMemcpy(dig.data(), r.ddig, dig.size() * 8, cudaMemcpyDeviceToHost));
        write_file(tag + ".digests", dig.data(), (size_t)((ARM_MODE[a] & HANDOFF) ? r.M : grid * slots) * 64);
        if (ARM_MODE[a] & HANDOFF) write_file(tag + ".y", y.data(), y.size() * 2);
      }
      printf("{\"kernel\": \"%s\", \"arm\": \"%s\", \"nm\": %d, \"nh\": %d, \"tm\": %d, \"slots\": %d, \"grid\": %d, "
             "\"half\": %d, \"regs\": %d, \"local_bytes\": %zu, \"smem\": %d, \"cells\": %d, \"cells_wrong\": %d, \"rows\": %d, "
             "\"hash_tiles\": %d, \"files\": \"%s\"}\n",
             k.name, ARMS[a], k.nm, k.nh, k.tmc(), slots, grid, k.ot == 2, fa.numRegs, fa.localSizeBytes, k.smem, cells, wrong,
             (ARM_MODE[a] & HANDOFF) ? r.M : grid * slots, params(k, r, a).hash_tiles, (ARM_MODE[a] & HASH) ? tag.c_str() : "");
      fflush(stdout);
    }
  }
  return bad;
}

static void time_arm(const Kern& k, Run& r, int arm, int launches, Nv& nv) {
  const Params p = params(k, r, arm);
  const int grid = r.grid(k.nm);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> ms, mhz, watts, mma_us, hash_us, comps;
  std::vector<unsigned long long> t((size_t)grid * TSLOTS);
  for (int i = 0; i <= launches; ++i) {
    CK(cudaMemset(r.dtimes, 0, t.size() * 8));
    CK(cudaEventRecord(e0));
    launch(k, r, p);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float m;
    CK(cudaEventElapsedTime(&m, e0, e1));
    if (!i) continue;
    ms.push_back(m);
    mhz.push_back(nv.sm());
    watts.push_back(nv.watts());
    CK(cudaMemcpy(t.data(), r.dtimes, t.size() * 8, cudaMemcpyDeviceToHost));
    unsigned long long m0 = ~0ull, m1 = 0, h0 = ~0ull, h1 = 0;
    double c = 0;
    for (int b = 0; b < grid; ++b) {
      const unsigned long long* q = &t[(size_t)b * TSLOTS];
      if (q[1]) m0 = std::min(m0, q[0]), m1 = std::max(m1, q[1]);
      if (q[3]) h0 = std::min(h0, q[2]), h1 = std::max(h1, q[3]);
      for (int w = 0; w < 4 * k.nh; ++w) c += 32.0 * (k.ot == 2 ? 2 : 4) * q[4 + w];
      if (q[3]) c += 128.0 * k.nh;  // the padding chunk's rounds
    }
    mma_us.push_back(m1 ? (m1 - m0) * 1e-3 : 0);
    hash_us.push_back(h1 ? (h1 - h0) * 1e-3 : 0);
    comps.push_back(c);
  }
  const double tm = median(ms), mu = median(mma_us), hu = median(hash_us), c = median(comps);
  const double ops = 2.0 * r.M * r.N * r.K;
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, k.f));
  printf("{\"kernel\": \"%s\", \"arm\": \"%s\", \"nwg\": %d, \"nm\": %d, \"nh\": %d, \"tm\": %d, \"grid\": %d, \"nform\": %d, "
         "\"stages\": %d, \"ot\": %d, \"P\": %d, \"Y\": %d, \"H\": %d, \"var\": \"%s\", \"regs\": %d, \"local_bytes\": %zu, "
         "\"ms\": %.5f, \"mma_us\": %.2f, \"hash_us\": %.2f, \"compressions\": %.0f, \"tops\": %.2f, \"hash_gcomp_per_s\": %.4f, "
         "\"mhz\": %.0f, \"watts\": %.0f, \"launches\": %d}\n",
         k.name, ARMS[arm], k.nwg(), k.nm, k.nh, k.tmc(), grid, k.nform, k.stages, k.ot, k.P, k.Y, k.H,
         (k.var & 3) == 3 ? (k.var & SUS_BIT ? "rs, suspend" : "rs") : (k.var & SUS_BIT ? "r, suspend" : "r"),
         fa.numRegs, fa.localSizeBytes, tm, mu, hu, c, (ARM_MODE[arm] & MMA) ? ops / (tm * 1e-3) * 1e-12 : 0.0,
         hu > 0 ? c / (hu * 1e-6) * 1e-9 : 0.0, median(mhz), median(watts), launches);
  fflush(stdout);
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

static const char* USAGE =
    "usage: beside list | gate M N K DIR [KERNEL...] | time M N K LAUNCHES [KERNEL...] | one M N K KERNEL ARM LAUNCHES\n";

static std::vector<const Kern*> kernels(int argc, char** argv, int from) {
  std::vector<const Kern*> ks;
  for (int a = from; a < argc; ++a) ks.push_back(find(argv[a]));
  if (ks.empty())
    for (const Kern& k : KERNS) ks.push_back(&k);
  return ks;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "list")) {
    for (const Kern& k : KERNS) printf("%s\n", k.name);
    return 0;
  }
  if (argc < 6) { fputs(USAGE, stderr); return 2; }
  Run r;
  setup(r, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  if (!strcmp(argv[1], "gate")) return gate(r, argv[5], kernels(argc, argv, 6)) ? 1 : 0;
  Nv nv;
  if (!strcmp(argv[1], "one") && argc >= 8) {
    const Kern& k = *find(argv[5]);
    int arm = -1;
    for (int a = 0; a < 4; ++a)
      if (!strcmp(ARMS[a], argv[6])) arm = a;
    if (arm < 0 || !k.has(arm)) { fputs("no such arm for that kernel\n", stderr); return 2; }
    prepare(k);
    time_arm(k, r, arm, atoi(argv[7]), nv);
    return 0;
  }
  if (strcmp(argv[1], "time")) { fputs(USAGE, stderr); return 2; }
  const int launches = atoi(argv[5]);
  for (const Kern* k : kernels(argc, argv, 6)) {
    prepare(*k);
    for (int arm = 0; arm < 4; ++arm)
      if (k->has(arm)) time_arm(*k, r, arm, launches, nv);
  }
  return 0;
}
