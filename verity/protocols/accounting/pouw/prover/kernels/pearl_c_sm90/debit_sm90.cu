// The words' debit counts of a Pearl-C call on the H100 and H200 (sm_90a), for debit.py: per word (a row of A' by a row of
// B~, E4M3 codes over k), `pearl_c_debit.word_debit`'s flagged atoms and later +0 promotions, as `fast_tile.word_debits`
// takes them over a tile's words.  Its own cubin: nothing here is in pearl_c_sm90.cubin, whose bytes it leaves as they are.
//
// Every operand is pearl_c_sm90.cu's row-major byte matrix with its depth contiguous (A' rows x k, B~ n x k), rows a multiple
// of 64, n of 32 and k of 128 (the promotion window, G = 4 atoms of k = 32).  A CTA (one warpgroup) holds a 64 x 32 block of
// words; a window's operands are staged in the descriptor's core matrices (`core`) and every atom is one wgmma m64n32k32
// E4M3 from shared memory, as form_s5_epw's A'F_B windows are: an atom from +0 is scale-d 0, every later one scale-d 1.
// Accumulator i of thread x is word (16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), 8 (i / 4) + 2 (x % 4) + i % 2) of the
// block, and its slot, where debit_fwd and debit_bounds keep it, is block * 2048 + 128 i + x.
//
//   debit_fwd     each window's word s0 (its four atoms from +0) into S0 (W x slots FP32 words)
//   debit_bounds  a thread a slot: the totals (one FP32 nearest-even add a window into +0, -0 made +0), the later +0
//                 promotions, and backwards from the final total the interval each total may take, as `_least` bisects it
//   debit_flags   each window's 15 masked words (the atoms outside the mask in order, from +0), promoted into the window's
//                 total, and the atoms of every mask whose total stays in its interval flagged; then the counts out
//
// Wherever a word, an accumulator or a total leaves the finite range the word is marked bad (BAD), and its tile is the
// scalar reference's (debit.py's caller replays it), as fast_tile hands it to `pearl_c_debit`.  FP32 adds are __fadd_rn,
// IEEE nearest-even with subnormals kept (build.sh's flags have no FTZ).
#include <cstdint>

#include "acc.h"

#define DEV __device__ __forceinline__
struct Args {
  uint64_t p[16];
  int32_t i[16];
};

constexpr long long TOP = 0x7F7FFFFF;

DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")
DEV uint64_t desc_plain(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
DEV uint32_t core(int r, int j) { return (r >> 3) * 256 + (j >> 4) * 128 + (r & 7) * 16 + (j & 15); }
DEV void qmma32(float* d, uint64_t da, uint64_t db, uint32_t sd) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 " ACC16_S
               ", %16, %17, p, 1, 1;\n}" : ACC16_F : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(float* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+f"(d[i])::"memory");
}

DEV long long key(uint32_t w) { return (w >> 31) ? -(long long)(w & 0x7FFFFFFFu) : (long long)w; }
DEV uint32_t word_of(long long x) { return x < 0 ? ((uint32_t)(-x) | 0x80000000u) : (uint32_t)x; }
DEV uint32_t plus_zero(uint32_t w) { return w == 0x80000000u ? 0u : w; }
DEV bool finite(uint32_t w) { return ((w >> 23) & 0xFF) != 0xFF; }
DEV uint32_t fadd(uint32_t x, uint32_t y) { return __float_as_uint(__fadd_rn(__uint_as_float(x), __uint_as_float(y))); }
// the key of RN(word(x) + w), -0 made +0; an overflow keys past every finite value
DEV long long carried(long long x, uint32_t w) { return key(plus_zero(fadd(word_of(x), w))); }
// the least key x in [-TOP, TOP] with carried(x, w) > b (GT) or >= b, else TOP + 1: carried is monotone in x
template <bool GT> DEV long long least(uint32_t w, long long b) {
  const auto pred = [&](long long x) {
    const long long c = carried(x, w);
    return GT ? c > b : c >= b;
  };
  if (!pred(TOP)) return TOP + 1;
  long long lo = -TOP, hi = TOP;
  while (lo < hi) {
    const long long mid = (lo + hi) >> 1;
    if (pred(mid)) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

// window g of the block's rows of A' (p0) and columns of B~ (p1) into atom j's core matrices: A at sA + 2048 j, B at sB + 1024 j
DEV void stage(const Args& a, int g, uint8_t* sA, uint8_t* sB) {
  const int tid = threadIdx.x, k = a.i[2], r0 = blockIdx.y * 64, c0 = blockIdx.x * 32;
  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.p[0]);
  const uint8_t* B = reinterpret_cast<const uint8_t*>(a.p[1]);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = tid + 128 * j, r = 8 * (idx / 64) + idx % 8, c = (idx / 8) % 8;
    *reinterpret_cast<uint4*>(sA + (c / 2) * 2048 + core(r, 16 * (c % 2))) =
        *reinterpret_cast<const uint4*>(A + (size_t)(r0 + r) * k + 128 * g + 16 * c);
  }
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int idx = tid + 128 * j, n = 8 * (idx / 64) + idx % 8, c = (idx / 8) % 8;
    *reinterpret_cast<uint4*>(sB + (c / 2) * 1024 + core(n, 16 * (c % 2))) =
        *reinterpret_cast<const uint4*>(B + (size_t)(c0 + n) * k + 128 * g + 16 * c);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the staged operands, written by the generic proxy, to wgmma
  __syncthreads();
}

// d = the window's atoms outside MASK (bit j: atom j left out), in order, from +0
template <int MASK> DEV void chain(float* d, uint32_t a0, uint32_t b0) {
  if constexpr (MASK == 15) {
#pragma unroll
    for (int i = 0; i < 16; i++) d[i] = 0.0f;
  } else {
    constexpr int first = (MASK & 1) == 0 ? 0 : (MASK & 2) == 0 ? 1 : (MASK & 4) == 0 ? 2 : 3;
    acc_fence<16>(d);
    WG_FENCE;
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (((MASK >> j) & 1) == 0) qmma32(d, desc_plain(a0 + 2048 * j), desc_plain(b0 + 1024 * j), j == first ? 0u : 1u);
    WG_COMMIT;
    WG_WAIT(0);
    acc_fence<16>(d);
  }
}

DEV size_t block_base() { return (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * 2048 + threadIdx.x; }

// p0 A' codes (i0 rows x k), p1 B~ codes (i1 = n rows x k), p2 S0 out; i2 k.  grid (n / 32, rows / 64), 128 threads
extern "C" __global__ void __launch_bounds__(128) debit_fwd(const Args a) {
  __shared__ __align__(128) uint8_t sA[4 * 2048], sB[4 * 1024];
  const int windows = a.i[2] / 128;
  const size_t slots = (size_t)a.i[0] * a.i[1], base = block_base();
  uint32_t* S0 = reinterpret_cast<uint32_t*>(a.p[2]);
  float d[16];
  for (int g = 0; g < windows; g++) {
    stage(a, g, sA, sB);
    chain<0>(d, sa(sA), sa(sB));
#pragma unroll
    for (int i = 0; i < 16; i++) S0[g * slots + base + 128 * i] = __float_as_uint(d[i]);
    __syncthreads();
  }
}

// p2 S0, p3 LO and p4 HI out (W x slots int32: the interval window g's total may take, bounds[g]), p5 Z out (a slot's later
// +0 promotions, u16), p6 BAD out (u8); i0 rows, i1 n, i2 k.  A thread a slot
extern "C" __global__ void __launch_bounds__(256) debit_bounds(const Args a) {
  const size_t slots = (size_t)a.i[0] * a.i[1], s = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= slots) return;
  const int windows = a.i[2] / 128;
  const uint32_t* S0 = reinterpret_cast<const uint32_t*>(a.p[2]);
  int* LO = reinterpret_cast<int*>(a.p[3]);
  int* HI = reinterpret_cast<int*>(a.p[4]);
  uint32_t t = 0;
  int zeros = 0;
  bool bad = false;
  for (int g = 0; g < windows; g++) {
    const uint32_t w = S0[g * slots + s], r = fadd(t, w);
    bad |= !finite(w) || !finite(r);
    t = plus_zero(r);
    zeros += g + 1 < windows && t == 0;
  }
  long long lo = key(t), hi = lo;
  LO[(windows - 1) * slots + s] = (int)lo;
  HI[(windows - 1) * slots + s] = (int)hi;
  for (int g = windows - 2; g >= 0; g--) {
    const uint32_t w = S0[(g + 1) * slots + s];
    const long long nlo = least<false>(w, lo), nhi = least<true>(w, hi) - 1;
    lo = nlo;
    hi = nhi;
    LO[g * slots + s] = (int)lo;
    HI[g * slots + s] = (int)hi;
  }
  reinterpret_cast<uint16_t*>(a.p[5])[s] = (uint16_t)zeros;
  reinterpret_cast<uint8_t*>(a.p[6])[s] = bad;
}

// the words of `d` (the window's masked words for MASK) promoted into the totals t, MASK's atoms flagged where that total stays
// in [lo, hi]
template <int MASK> DEV void keep(const float* d, const uint32_t* t, const int* lo, const int* hi, uint32_t* flag,
                                  uint32_t& bad) {
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const uint32_t s = __float_as_uint(d[i]), v = fadd(t[i], s);
    bad |= (uint32_t)(!finite(s) || !finite(v)) << i;
    const long long kv = key(plus_zero(v));
    if (lo[i] <= kv && kv <= hi[i]) flag[i] |= MASK;
  }
}
#define MASKED(M)                  \
  chain<M>(d, sa(sA), sa(sB));     \
  keep<M>(d, t, lo, hi, flag, bad);

// p0 A', p1 B~, p3 LO, p4 HI, p5 Z, p6 BAD (debit_bounds'), p7 atoms out and p8 zeros out (u16, rows x n, row-major), p9 bad
// out (u8, rows x n); i0 rows, i1 n, i2 k.  grid (n / 32, rows / 64), 128 threads
extern "C" __global__ void __launch_bounds__(128) debit_flags(const Args a) {
  __shared__ __align__(128) uint8_t sA[4 * 2048], sB[4 * 1024];
  const int windows = a.i[2] / 128, n = a.i[1], tid = threadIdx.x;
  const size_t slots = (size_t)a.i[0] * n, base = block_base();
  const int* LO = reinterpret_cast<const int*>(a.p[3]);
  const int* HI = reinterpret_cast<const int*>(a.p[4]);
  float d[16];
  uint32_t t[16], count[16], bad = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) t[i] = count[i] = 0;
  for (int g = 0; g < windows; g++) {
    int lo[16], hi[16];
    uint32_t flag[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      lo[i] = LO[g * slots + base + 128 * i];
      hi[i] = HI[g * slots + base + 128 * i];
      flag[i] = 0;
    }
    stage(a, g, sA, sB);
    MASKED(1) MASKED(2) MASKED(3) MASKED(4) MASKED(5) MASKED(6) MASKED(7) MASKED(8)
    MASKED(9) MASKED(10) MASKED(11) MASKED(12) MASKED(13) MASKED(14) MASKED(15)
    chain<0>(d, sa(sA), sa(sB));
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const uint32_t s = __float_as_uint(d[i]), v = fadd(t[i], s);
      bad |= (uint32_t)(!finite(s) || !finite(v)) << i;
      t[i] = plus_zero(v);
      count[i] += __popc(flag[i]);
    }
    __syncthreads();
  }
  const uint16_t* Z = reinterpret_cast<const uint16_t*>(a.p[5]);
  const uint8_t* BAD = reinterpret_cast<const uint8_t*>(a.p[6]);
  uint16_t* atoms = reinterpret_cast<uint16_t*>(a.p[7]);
  uint16_t* zeros = reinterpret_cast<uint16_t*>(a.p[8]);
  uint8_t* bad_out = reinterpret_cast<uint8_t*>(a.p[9]);
  const int r0 = blockIdx.y * 64 + 16 * (tid / 32) + (tid % 32) / 4, c0 = blockIdx.x * 32 + 2 * (tid % 4);
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const size_t at = (size_t)(r0 + 8 * ((i / 2) % 2)) * n + c0 + 8 * (i / 4) + i % 2, slot = base + 128 * i;
    atoms[at] = (uint16_t)count[i];
    zeros[at] = Z[slot];
    bad_out[at] = BAD[slot] | ((bad >> i) & 1);
  }
}
