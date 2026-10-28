// The exact GEMM O = A W^T (s32 out) on the H100 and H200 (sm_90a) with 2-bit operands streamed from global memory packed
// four values a byte, unpacked on chip to one value a byte in shared memory, and multiplied there by warpgroup MMAs
// (wgmma m64n256k32) into one s32 accumulator over all of K.  O is torch._int_mm(A, W.t())'s int32 output, bit for bit, on
// the unpacked operands.
//
// The packed format: a row of A (M x K) or W (N x K) is K / 4 bytes, and its value k, in {0, 1, 2, 3}, is bits 2i and 2i + 1
// of byte 4 (k / 16) + k % 4, with i = (k / 4) % 4.  So each little-endian 32-bit word x of a row holds 16 consecutive
// values, and (x >> 2i) & 0x03030303 is the bytes of values 16 w + 4i to 16 w + 4i + 3, in order: an unpacked 16-byte
// chunk is four such words, two integer instructions each (one for i = 0).  The u8 form (x << (6 - 2i)) & 0xC0C0C0C0 is
// the same values times 64, whose left shifts ptxas may issue as multiplies on the FMA pipe; the u8 MMA's sums are then
// 4,096 times O's (exact while K <= 58,254), and the epilogue shifts them back.
//
// The masked form needs no shifts: W packed with its pairs reversed (value k at bits 2 (3 - i) and 2 (3 - i) + 1 of the
// same byte), x & (0x03030303 << 2i) is A's values times 4^i and y & (0x03030303 << (6 - 2i)) W's times 4^(3 - i), so every
// product is 64 times the values', one AND a 4 values, and the u8 MMA's sums are 64 times O's (exact while K < 2^31 / 576).
//
// One CTA an SM, persistent, 256 threads in two consumer warpgroups, each 64 x 256 of a 128 x 256 tile, at most 168
// registers each, so a 256-thread hash_k CTA fits beside them on the SM.  No producer warpgroup: thread 0 issues every TMA,
// all 256 threads unpack, and one named barrier a k-block (128 deep) orders the two:
//   k-block g: MMAs on unpacked slot g % SU; wait for this warpgroup's k-block g - 1; barrier; refill; unpack k-block
//              g + SU - 1 into slot (g - 1) % SU, which both warpgroups are done with
// and after a tile's last k-block, wait for all of it and store it.
// Arms (`pk_<arm>`):
//   i8    W and A stored as int8 and streamed by TMA straight into the slots (SWIZZLE_128B), nothing to unpack: the same
//         schedule on four times the bytes
//   p2    packed boxes (32 bytes of each row a k-block) by TMA into a ring of SP slots, unpacked by shifts and masks
//   p2u   the same, unpacked as u8 times 64 (above)
//   p2m   the same with W's pairs reversed, unpacked by masks alone (above)
//   res   p2's unpack of one 128 x 256 x 512 block into the four slots once, then the MMAs over them for every tile: the
//         schedule's ceiling with nothing streamed; O is (K / 512) times that block's product in every tile
//   p2x, p2ux, p2mx  p2's, p2u's and p2m's loads and unpacking with no MMAs: the unpacking's own time
// M must be a multiple of 128, N of 256 and K of 128 (512 for res) (bench.py checks): the filler picks its shape.
//
// hash_k: SHA-512 compressions (sha512_rows' sx::compress, V 0, rolled), each thread chaining its own state, for the rate
// beside the GEMM.  stamp_k: each SM's %clock64 and %globaltimer, one CTA an SM (its shared memory keeps a second off), so
// the SM clock over a span of launches is the difference of two stamps.  wait_k: holds a stream until a launch's CTAs have
// all started.
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "../s8_linear/s8_acc.h"
#include "../sha512_rows/sha512x.cuh"

#define DEV __device__ __forceinline__

// a CTA's span: %globaltimer (ns) and %clock64 at its start and end, its SM and the tiles it did
struct Stamp {
  unsigned long long t0, t1, c0, c1, sm, tiles;
};

// one launch's arguments, a __grid_constant__ parameter (bench.py's `Params`)
struct alignas(128) Params {
  CUtensorMap a, b;    // packed: boxes of 32 bytes of TM or BN rows; i8: boxes of 128 bytes, SWIZZLE_128B
  int32_t* o;          // O [M, N], s32, row-major
  Stamp* st;           // a stamp a CTA, or null
  int M, N, K, group;  // group: tile rows a raster group
  unsigned* started;   // counted up by each CTA as it starts, or null
};
static_assert(offsetof(Params, o) == 256 && offsetof(Params, M) == 272 && offsetof(Params, started) == 288 &&
                  sizeof(Params) == 384,
              "bench.py's Params");

DEV unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DEV uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// ---------------------------------------------------------------- mbarriers, TMA, shared memory, wgmma
DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
DEV uint4 lds128(uint32_t at) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(at) : "memory");
  return v;
}
DEV void sts128(uint32_t at, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(at), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
// the generic proxy's shared-memory writes (the unpacking) made visible to the async proxy (wgmma's operand reads)
DEV void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// both consumer warpgroups (named barrier 1; 0 is __syncthreads')
DEV void sync256() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0): one atom (k = 32) of a 64 x 256 warpgroup tile, s8 or u8 operands.  Accumulator i of thread x is
// row 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <bool U8> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  if constexpr (U8)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.u8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// ---------------------------------------------------------------- the GEMM
enum Load { I8 = 0, P2 = 1, P2U = 2, RES = 3, P2M = 4 };
constexpr int TM = 128, BN = 256, BK = 128, NA = BN / 2;
// an unpacked slot: A's 128 rows, then W's 256, 128 bytes each in the SWIZZLE_128B layout; a packed slot: the same rows,
// 32 bytes each
constexpr uint32_t A_BYTES = TM * BK, U_STAGE = (TM + BN) * BK, P_STAGE = U_STAGE / 4;

template <int LOAD, int SU, int SP> struct Layout {
  static_assert(SU >= 3, "a slot unpacked a k-block ahead of the one its barrier publishes");
  static_assert(LOAD == I8 ? SP == 0 : LOAD == RES ? SP == 1 && SU == 4 : SP >= SU - 1, "packed slots for the prologue");
  static constexpr int NBAR = LOAD == I8 ? SU : SP;
  static constexpr int SMEM = 1024 + SU * U_STAGE + SP * P_STAGE + 8 * NBAR;
  static_assert(U_STAGE % 1024 == 0 && A_BYTES % 1024 == 0 && SMEM <= 227 * 1024, "slots 1,024-byte aligned; an SM's shared memory");
};

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// one row's half: 16 packed bytes at src into four 16-byte chunks at chunk ((4 h + q) ^ sw) of the row at dst, in form F
// (P2, P2U, or P2M with W's reversed pairs when REV)
template <int F, bool REV> DEV void unpack_half(uint32_t src, uint32_t dst, int h, int sw) {
  constexpr uint32_t M2 = 0x03030303u, H2 = 0xC0C0C0C0u;
  const uint4 v = lds128(src);
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const uint32_t x = w[q], at = dst + (((4 * h + q) ^ sw) << 4);
    if constexpr (F == P2U)
      sts128(at, (x << 6) & H2, (x << 4) & H2, (x << 2) & H2, x & H2);
    else if constexpr (F == P2M && REV)
      sts128(at, x & (M2 << 6), x & (M2 << 4), x & (M2 << 2), x & M2);
    else if constexpr (F == P2M)
      sts128(at, x & M2, x & (M2 << 2), x & (M2 << 4), x & (M2 << 6));
    else
      sts128(at, x & M2, (x >> 2) & M2, (x >> 4) & M2, (x >> 6) & M2);
  }
}

// packed slot src into unpacked slot dst: thread tid's three (row, half) pairs of the slot's 384 rows (A's, then W's), each
// half 16 packed bytes into four 16-byte chunks at chunk (c ^ (row % 8)) of the row
template <int F> DEV void unpack(uint32_t src, uint32_t dst, int tid) {
  const int h = tid & 1, sw = (tid >> 1) & 7;
  src += (tid >> 1) * (BK / 4) + 16 * h;
  dst += (tid >> 1) * BK;
  unpack_half<F, false>(src, dst, h, sw);
  // one pass at a time: the 128 accumulators leave the unpacking about 40 registers
#pragma unroll 1
  for (int j = 1; j < 3; j++) unpack_half<F, true>(src + j * 128 * (BK / 4), dst + j * 128 * BK, h, sw);
}

// a 64 x 256 accumulator of warpgroup wg into O, each sum shifted right by S: rows r and r + 8, each pair of adjacent
// columns one 8-byte store
template <int S> DEV void store_tile(const Params& p, const int32_t* d, int tm, int tn, int wg, int x) {
  const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
  int32_t* o0 = p.o + (size_t)row * p.N + col0;
  int32_t* o1 = o0 + (size_t)8 * p.N;
#pragma unroll
  for (int j = 0; j < NA / 4; j++) {
    *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[4 * j] >> S, d[4 * j + 1] >> S);
    *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[4 * j + 2] >> S, d[4 * j + 3] >> S);
  }
}

// the start half of the CTA's stamp goes out at once, so no register holds it through the GEMM
DEV void stamp_start(const Params& p) {
  if (threadIdx.x == 0) {
    if (p.st) {
      p.st[blockIdx.x].t0 = gtimer();
      p.st[blockIdx.x].c0 = clock64();
    }
    if (p.started) atomicAdd(p.started, 1u);
  }
}
DEV void stamp_end(const Params& p, int tiles) {
  if (threadIdx.x == 0 && p.st) {
    Stamp& s = p.st[blockIdx.x];
    s.t1 = gtimer();
    s.c1 = clock64();
    s.sm = smid();
    s.tiles = (unsigned long long)tiles;
  }
}

template <int LOAD, bool MMA, int SU, int SP> DEV void body(const Params& p) {
  using L = Layout<LOAD, SU, SP>;
  constexpr bool U8 = LOAD == P2U || LOAD == P2M, STREAM = LOAD == P2 || LOAD == P2U || LOAD == P2M;
  constexpr int SHIFT = LOAD == P2U ? 12 : LOAD == P2M ? 6 : 0;
  constexpr int NP = SP > 0 ? SP : 1;  // the packed ring's modulus (i8 has none)
  extern __shared__ __align__(1024) uint8_t smem[];
  stamp_start(p);
  const uint32_t ubase = (smem_at(smem) + 1023) & ~1023u, pbase = ubase + SU * U_STAGE, bars = pbase + SP * P_STAGE;
  const int KB = p.K / BK, tiles_m = p.M / TM, tiles_n = p.N / BN, units = tiles_m * tiles_n;
  const int tid = threadIdx.x;
  // this CTA's k-blocks in order, g = 0 .. G - 1: its units blockIdx.x, + gridDim.x, ..., KB k-blocks each
  const int G = ((int)blockIdx.x < units ? (units - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0) * KB;
  if (tid == 0) {
    for (int s = 0; s < L::NBAR; s++) mbar_init(bars + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tma_prefetch(&p.a);
    tma_prefetch(&p.b);
  }
  __syncthreads();

  // thread 0: k-block n's TMA, into unpacked slot n % SU (i8) or packed slot n % SP
  auto load = [&](int n) {
    if (n >= G) return;
    const int j = n / KB, kb = n - j * KB;
    int tm, tn;
    raster(blockIdx.x + j * gridDim.x, tiles_m, tiles_n, p.group, tm, tn);
    if constexpr (LOAD == I8) {
      const uint32_t dst = ubase + (n % SU) * U_STAGE, bar = bars + 8 * (n % SU);
      mbar_expect_tx(bar, U_STAGE);
      tma_load(dst, &p.a, bar, kb * BK, tm * TM);
      tma_load(dst + A_BYTES, &p.b, bar, kb * BK, tn * BN);
    } else {
      const uint32_t dst = pbase + (n % NP) * P_STAGE, bar = bars + 8 * (n % NP);
      mbar_expect_tx(bar, P_STAGE);
      tma_load(dst, &p.a, bar, kb * (BK / 4), tm * TM);
      tma_load(dst + A_BYTES / 4, &p.b, bar, kb * (BK / 4), tn * BN);
    }
  };
  // every thread: its part of k-block n's unpacking, packed slot n % SP into unpacked slot n % SU
  auto unpack_n = [&](int n) {
    if (n >= G) return;
    mbar_wait(bars + 8 * (n % NP), (n / NP) & 1);
    unpack<LOAD>(pbase + (n % NP) * P_STAGE, ubase + (n % SU) * U_STAGE, tid);
  };

  if constexpr (LOAD == RES) {
    // tile (0, 0)'s first SU k-blocks, through packed slot 0, into the SU slots once
    for (int f = 0; f < SU; f++) {
      if (tid == 0) {
        mbar_expect_tx(bars, P_STAGE);
        tma_load(pbase, &p.a, bars, f * (BK / 4), 0);
        tma_load(pbase + A_BYTES / 4, &p.b, bars, f * (BK / 4), 0);
      }
      mbar_wait(bars, f & 1);
      unpack<P2>(pbase, ubase + f * U_STAGE, tid);
      __syncthreads();
    }
    fence_async();
    __syncthreads();
  } else if constexpr (LOAD == I8) {
    if (tid == 0)
      for (int n = 0; n < SU - 1; n++) load(n);
  } else {
    if (tid == 0)
      for (int n = 0; n < SP; n++) load(n);
    for (int n = 0; n < SU - 1; n++) {
      unpack_n(n);
      fence_async();
      sync256();
      if (tid == 0) load(n + SP);
    }
  }

  int g = 0, tiles = 0;
  for (int t = blockIdx.x; t < units; t += gridDim.x) {
    [[maybe_unused]] int32_t d[NA];
    for (int kb = 0; kb < KB; kb++, g++) {
      const int s = g % SU;
      if constexpr (LOAD == I8) mbar_wait(bars + 8 * s, (g / SU) & 1);
      if constexpr (MMA) {
        const uint32_t a0 = ubase + s * U_STAGE + (tid / 128) * 64 * BK, b0 = ubase + s * U_STAGE + A_BYTES;
        WG_FENCE;
        imma<U8>(d, desc_sw128(a0), desc_sw128(b0), kb > 0);
        imma<U8>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
        imma<U8>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
        imma<U8>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
        WG_COMMIT;
        WG_WAIT(1);
      }
      sync256();
      if constexpr (LOAD == I8) {
        if (tid == 0) load(g + SU - 1);
      } else if constexpr (STREAM) {
        // the packed slot of the k-block unpacked last iteration is free now; slot (g - 1) % SU is done in both warpgroups
        if (tid == 0 && g > 0) load(g + SU - 2 + SP);
        unpack_n(g + SU - 1);
        fence_async();
      }
    }
    if constexpr (MMA) {
      WG_WAIT(0);
      acc_fence<NA>(d);
      int tm, tn;
      raster(t, tiles_m, tiles_n, p.group, tm, tn);
      store_tile<SHIFT>(p, d, tm, tn, tid / 128, tid % 128);
    }
    tiles++;
  }
  __syncthreads();
  stamp_end(p, tiles);
}

#define PK(NAME, LOAD, MMA, SU, SP) \
  extern "C" __global__ void __maxnreg__(168) NAME(const __grid_constant__ Params p) { body<LOAD, MMA, SU, SP>(p); }
PK(pk_i8, I8, true, 4, 0)
PK(pk_p2, P2, true, 3, 6)
PK(pk_p2u, P2U, true, 3, 6)
PK(pk_res, RES, true, 4, 1)
PK(pk_p2m, P2M, true, 3, 6)
PK(pk_p2x, P2, false, 3, 6)
PK(pk_p2ux, P2U, false, 3, 6)
PK(pk_p2mx, P2M, false, 3, 6)

// what bench.py takes from this build, checked when it loads the cubin: Params' size and offsets, then the shared memory
// of the i8, p2 (p2u, p2m and their unpack-only arms) and res layouts
extern "C" __global__ void pk_layout(int* out) {
  out[0] = sizeof(Params);
  out[1] = offsetof(Params, o);
  out[2] = offsetof(Params, M);
  out[3] = offsetof(Params, started);
  out[4] = Layout<I8, 4, 0>::SMEM;
  out[5] = Layout<P2, 3, 6>::SMEM;
  out[6] = Layout<RES, 4, 1>::SMEM;
}

// ---------------------------------------------------------------- beside the GEMM
// `reps` compressions a thread, each over a block whose first eight words are xored with the chaining value; io is written
// only for a state no input reaches, so the compressions stay live
extern "C" __global__ void __maxnreg__(80) hash_k(unsigned long long* io, int reps, Stamp* out) {
  unsigned long long t0 = 0, c0 = 0;
  if (threadIdx.x == 0) {
    t0 = gtimer();
    c0 = clock64();
  }
  const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
  __syncthreads();
  if (threadIdx.x == 0)
    out[blockIdx.x] = Stamp{t0, gtimer(), c0, (unsigned long long)clock64(), smid(), (unsigned long long)reps};
}

// waits until `want` CTAs of a launch have started (Params.started), so a kernel launched after it on another stream finds
// that launch's CTAs resident
extern "C" __global__ void wait_k(volatile unsigned* started, unsigned want) {
  while (*started < want) {
  }
}

// every CTA stamps once all have arrived, so the launch's CTAs are resident together, one an SM; after a millisecond without
// them all it stamps sm = ~0 (bench.py refuses the segment)
extern "C" __global__ void stamp_k(Stamp* out, unsigned* arrive) {
  if (threadIdx.x != 0) return;
  atomicAdd(arrive, 1u);
  const unsigned long long t0 = gtimer();
  bool all = true;
  while (*(volatile unsigned*)arrive < gridDim.x)
    if (gtimer() - t0 > 1000000) {
      all = false;
      break;
    }
  out[blockIdx.x] = Stamp{gtimer(), 0, (unsigned long long)clock64(), 0, all ? smid() : ~0ull, 0};
}
