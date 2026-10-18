// Structural levers for hm96-sha512 tile leaves in frame-v3-sha512 trees, with SHA-512 kept (benchmarks/hash/structure).
//
// Fusion: a persistent s8 GEMM (mma.sync m16n8k32, 128 x 128 x 64 CTA tiles, 8 MMA warps, cp.async 4 stages) whose CTA
// also runs HW hash warps: each tile the MMA warps finish (C as bf16, written to global memory, then a shared counter), the
// hash warps hash its leaves while the MMA warps compute the next tile. The committed operand is C's byte image (M rows of
// 2N bytes) under commit_gpu's tile layout (Decision 46): leaf pos is block (pos % ceil(M/R), pos / ceil(M/R)) of R rows x 32
// bytes, a sha512/row/v2 row of 256 R bits. Leaves use commit_gpu.cuh's own hash_after_block and leaf_from_digest, after
// the salt phase (Committer::salts), so the levels are commit_gpu's, which reference.py holds to verity.ml.commitments.
//
//   fuse check PARAMS OUT M N K R            fused and separate levels compared, C checked against cuBLASLt; OUT gets
//                                            params.bin, operand.bin, levels.bin, dump.json for reference.py verify
//   fuse bench PARAMS M N K R REPS           the arms below, interleaved rep by rep, each after a stand-in producer
//   fuse tree N ARITY REPS                   a-ary node levels over N leaves (timing only: a changed node format)
//   fuse chacha OUT N REPS                   ChaCha20-derived salts: keystream dump for check_chacha, and the salt phase's cost
//
// Arms (bench): lt (cuBLASLt s8 GEMM, s32 out, autotuned), g (this GEMM alone), gs (g, then the separate data phase: one
// thread a leaf, then Committer::nodes), rec (g, then the committer of record's tiles() + nodes()), f4 / f8 (the fused GEMM
// with 4 or 8 hash warps, then nodes()), h (the separate data phase alone), salt (the salt phase, SHA-512 PRF salts).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <nvml.h>

#include <array>
#include <chrono>

#define COMMIT_GPU_DOMS 1
#include "commit_gpu.cuh"

#define CL(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuBLASLt %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

constexpr int BM = 128, BN = 128, BK = 64, STAGES = 4, MMA_THREADS = 256;
constexpr int STAGE_BYTES = (BM + BN) * BK;
constexpr int SMEM_GEMM = STAGES * STAGE_BYTES;   // 64 KiB

DEV u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
DEV void cp_async16(u32 dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
DEV void bar_mma() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
DEV void ldsm_x4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEV void mma_s8(int* c, const u32* a, const u32* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// a stage's rows are 64 bytes (4 chunks of 16); chunk c of row r sits at c ^ ((r >> 1) & 3), so ldmatrix's 8 rows hit 32 banks
DEV u32 swz(int r, int c) { return (u32)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

// A tile leaf's bytes from C written by this kernel: through L2 (ld.global.cg), rows at or past `rows` zero.
struct TileSrcCG {
  const uint8_t* base;
  u64 K;
  long long rows;
  DEV uint4 ld16(u32 off) const {
    return (long long)(off >> 5) < rows ? __ldcg(reinterpret_cast<const uint4*>(base + (off >> 5) * K + (off & 31))) : make_uint4(0, 0, 0, 0);
  }
  DEV void pf(u32 off) const {
#pragma unroll
    for (u32 r = off >> 5; r < (off >> 5) + 4; ++r)
      if ((long long)r < rows) asm volatile("prefetch.global.L2 [%0];" ::"l"(base + r * K));
  }
};

struct Geo {
  const int8_t* A;   // M x K, row-major
  const int8_t* B;   // N x K, row-major (C = A B^T)
  uint8_t* C;        // M x N bf16, row-major: the committed operand, M rows of Kb = 2N bytes
  int M, N, K, tiles_m, tiles;
  u64 mr;            // ceil(M / R)
  u32 R;
  const u64* S;      // the salt phase's records
  u64* lv0;          // level 0
  bool pf;           // an L2 prefetch of each leaf's next block
  int diag;          // timing only (wrong levels): bit 0 the MMA warps skip the k-loop, bit 1 the hash warps exit, bit 2
                     // no tile handoff (with bit 1 only)
  bool gpu_fence;    // the tile handoff's fences at GPU scope (the first design) instead of CTA scope
  u32* flags;        // when set, the GEMM marks each tile t done by flags[t] = epoch (for hash_kernel)
  u32 epoch;
};

DEV void leaf_at(const Geo& g, u64 br, u64 bc) {
  const u64 Kb = 2ull * g.N, pos = bc * g.mr + br;
  const TileSrcCG src{g.C + br * g.R * Kb + 32 * bc, Kb, (long long)g.M - (long long)(br * g.R)};
  u64 x[8];
  hash_after_block(C.mid_P, src, 32 * g.R, x, g.pf);
  leaf_from_digest<0>(0, pos, x, g.S, g.lv0 + 8 * pos, nullptr);
}

// The hash warps' loop over the leaves of the tiles t = cta + k ctas: a warp takes 32 consecutive leaves and waits,
// uniformly, for the last tile they reach (lanes waiting on different tiles would diverge, and a sleeping path stalls the
// hashing one). Tile k is ready when *done > k (FLAGS false: the GEMM's own CTA) or flags[t] = epoch (FLAGS: a separate
// persistent kernel). Not inlined: inlined, it changes how ptxas schedules the GEMM's main loop (IMMAs serialized).
template <int HW, bool FLAGS>
__device__ __noinline__ void hash_loop(const Geo& g, int hw, int lane, int cta, int ctas, volatile int* done) {
  const int my_tiles = cta < g.tiles ? (g.tiles - cta + ctas - 1) / ctas : 0;
  const int rbs = BM / g.R, L = rbs * (2 * BN / 32);
  const u64 total = (u64)my_tiles * L;
  for (u64 s0 = 32ull * hw; s0 < total; s0 += 32 * HW) {
    const int kmax = (int)((s0 + 31 < total ? s0 + 31 : total - 1) / L);
    if (lane == 0) {
      if (FLAGS) {
        const volatile u32* f = g.flags + cta + kmax * ctas;
        while (*f != g.epoch) __nanosleep(128);
      } else {
        while (*done <= kmax) __nanosleep(64);
      }
    }
    __syncwarp();
    if (FLAGS || g.gpu_fence) __threadfence();
    else __threadfence_block();
    const u64 s = s0 + lane;
    if (s >= total) continue;
    const int k = (int)(s / L), l = (int)(s % L);
    const int t = cta + k * ctas, tm = t % g.tiles_m, tn = t / g.tiles_m;
    const u64 br = (u64)tm * rbs + l % rbs, bc = (u64)tn * (2 * BN / 32) + l / rbs;
    if (br < g.mr) leaf_at(g, br, bc);
  }
}

// HW hash warps beside 8 MMA warps (HW = 0: the plain GEMM). One CTA an SM, persistent over tiles t = blockIdx.x + k gridDim.x
// (tile (t % tiles_m, t / tiles_m)).
template <int HW>
__global__ void __launch_bounds__(MMA_THREADS + 32 * HW, 1) gemm_kernel(Geo g) {
  extern __shared__ uint4 smem[];   // commit_gpu.cuh's salt_kernel declares the same array
  uint8_t* const sm8 = reinterpret_cast<uint8_t*>(smem);
  __shared__ volatile int done;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int my_tiles = blockIdx.x < g.tiles ? (g.tiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  if (HW && tid == 0) done = 0;
  if (HW) __syncthreads();
  if constexpr (HW > 0) {
    if (tid >= MMA_THREADS) {
      if (!(g.diag & 2)) hash_loop<HW, false>(g, (tid - MMA_THREADS) >> 5, lane, blockIdx.x, gridDim.x, &done);
      return;
    }
  }
  const int wm = warp >> 2, wn = warp & 3;
  const int kt_n = (g.diag & 1) ? 0 : g.K / BK;
  for (int k = 0; k < my_tiles; ++k) {
    const int t = blockIdx.x + k * gridDim.x, m0 = (t % g.tiles_m) * BM, n0 = (t / g.tiles_m) * BN;
    auto load = [&](int stage, int kt) {
      uint8_t* base = sm8 + stage * STAGE_BYTES;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + MMA_THREADS * i, r = idx >> 2, c = idx & 3;
        const int gm = m0 + r;
        cp_async16(smem_u32(base + swz(r, c)), g.A + (size_t)(gm < g.M ? gm : g.M - 1) * g.K + kt * BK + 16 * c, gm < g.M);
        cp_async16(smem_u32(base + BM * BK + swz(r, c)), g.B + (size_t)(n0 + r) * g.K + kt * BK + 16 * c, true);
      }
    };
    int acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[a][b][c] = 0;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < kt_n) load(s, s);
      cp_commit();
    }
    for (int kt = 0; kt < kt_n; ++kt) {
      cp_wait<STAGES - 2>();
      bar_mma();
      if (kt + STAGES - 1 < kt_n) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
      cp_commit();
      const uint8_t* sa = sm8 + (kt % STAGES) * STAGE_BYTES;
      const uint8_t* sb = sa + BM * BK;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        u32 af[4][4], bf[4][2];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const int r = wm * 64 + mt * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
          ldsm_x4(smem_u32(sa + swz(r, 2 * ks + (lane >> 4))), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
        }
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int n = wn * 32 + np * 16 + (lane & 7) + 8 * (lane >> 4);
          ldsm_x4(smem_u32(sb + swz(n, 2 * ks + ((lane >> 3) & 1))), bf[2 * np][0], bf[2 * np][1], bf[2 * np + 1][0], bf[2 * np + 1][1]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) mma_s8(acc[mt][nt], af[mt], bf[nt]);
      }
    }
    cp_wait<0>();
    const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int row = m0 + wm * 64 + mt * 16 + gq, col = n0 + wn * 32 + nt * 8 + 2 * tq;
        const __nv_bfloat162 lo = __floats2bfloat162_rn((float)acc[mt][nt][0], (float)acc[mt][nt][1]);
        const __nv_bfloat162 hi = __floats2bfloat162_rn((float)acc[mt][nt][2], (float)acc[mt][nt][3]);
        if (row < g.M) *reinterpret_cast<__nv_bfloat162*>(g.C + ((size_t)row * g.N + col) * 2) = lo;
        if (row + 8 < g.M) *reinterpret_cast<__nv_bfloat162*>(g.C + ((size_t)(row + 8) * g.N + col) * 2) = hi;
      }
    bar_mma();   // every store of tile k is issued, and no warp still reads a stage the next prologue writes
    if (HW && tid == 0 && !(g.diag & 4)) {   // the consumers are this CTA's hash warps: a CTA-scope release suffices
      if (g.gpu_fence) __threadfence();
      else __threadfence_block();
      done = k + 1;
    }
    if (g.flags && tid == 0) {
      __threadfence();
      *reinterpret_cast<volatile u32*>(g.flags + t) = g.epoch;
    }
  }
}

// The persistent consumer: CTA b hashes the leaves of the GEMM CTA b's tiles as their flags turn (ctas = the GEMM's grid).
template <int HW>
__global__ void __launch_bounds__(32 * HW) hash_kernel(Geo g, int ctas) {
  hash_loop<HW, true>(g, threadIdx.x >> 5, threadIdx.x & 31, blockIdx.x, ctas, nullptr);
}

// The separate data phase: a thread a leaf, the same leaf code (pads are written once, ahead: they hold no data).
__global__ void __launch_bounds__(256) sep_leaves(Geo g, u64 n) {
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < n) leaf_at(g, pos % g.mr, pos / g.mr);
}
__global__ void pad_fill(u64* lv0, u64 n, u64 width) {
  const u64 pos = n + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < width) {
    u64 o[8];
    pad(0, pos, o);
    st8(lv0 + 8 * pos, o);
  }
}
// cuBLASLt's s32 C as the same bf16 image, for the GEMM's check
__global__ void s32_to_bf16(const int32_t* Y, uint8_t* out, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn((float)Y[i]);
}
__global__ void fill(uint8_t* p, size_t n, u32 seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    u32 x = (u32)i * 0x9E3779B1u ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}
__global__ void spin(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}
constexpr long long SPIN = 400000;

// --- the a-ary tree (timing only) ------------------------------------------------------------------------------------
// A node of `a` children: the level's midstate, then index (8 bytes) || a digests || padding, ceil((8 + 64 a + 17) / 128)
// compressions (2, 3, 5, 9 for a = 2, 4, 8, 16): the cost of an a-ary frame node, whose format no reference computes.
template <int A>
__global__ void __launch_bounds__(256) anode_level(const u64* __restrict__ in, u64* __restrict__ out, u64 n_out, int level) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  constexpr int BYTES = 8 + 64 * A, NBLK = (BYTES + 17 + 127) / 128;
  u64 st[8];
  cpy8(st, CD[0].node_mid[level]);
  for (int b = 0; b < NBLK; ++b) {
    u64 W[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const int word = 16 * b + w;   // message word: 0 the index, 1 .. 8 A the children, then the padding and length
      u64 v = 0;
      if (word == 0) v = i;
      else if (word <= 8 * A) v = __ldg(in + 8 * A * i + (word - 1));
      else if (word == 8 * A + 1) v = 0x8000000000000000ull;
      if (b == NBLK - 1 && w == 15) v = (u64)(128 + BYTES) * 8;
      W[w] = v;
    }
    sha::compress(st, W);
  }
  st8(out + 8 * i, st);
}

// --- ChaCha20 salts ---------------------------------------------------------------------------------------------------
// y for leaf pos = keystream bytes [192 pos, 192 pos + 192) of ChaCha20 (RFC 8439) under a 256-bit seed and a fixed nonce:
// blocks 3 pos .. 3 pos + 2. A thread a block.
DEV u32 rotl32(u32 x, int n) { return __funnelshift_l(x, x, n); }
#define CHQR(a, b, c, d)                    \
  a += b; d ^= a; d = rotl32(d, 16);        \
  c += d; b ^= c; b = rotl32(b, 12);        \
  a += b; d ^= a; d = rotl32(d, 8);         \
  c += d; b ^= c; b = rotl32(b, 7);
struct ChaKey { u32 k[8], nonce[3]; };
__global__ void __launch_bounds__(256) chacha_salts(ChaKey key, u64 nblocks, uint4* __restrict__ out) {
  const u64 blk = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= nblocks) return;
  u32 s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key.k[0], key.k[1], key.k[2], key.k[3],
               key.k[4],    key.k[5],    key.k[6],    key.k[7],    (u32)blk, key.nonce[0], key.nonce[1], key.nonce[2]};
  u32 x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    CHQR(x[0], x[4], x[8], x[12]) CHQR(x[1], x[5], x[9], x[13]) CHQR(x[2], x[6], x[10], x[14]) CHQR(x[3], x[7], x[11], x[15])
    CHQR(x[0], x[5], x[10], x[15]) CHQR(x[1], x[6], x[11], x[12]) CHQR(x[2], x[7], x[8], x[13]) CHQR(x[3], x[4], x[9], x[14])
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[4 * blk + i] = make_uint4(x[4 * i] + s[4 * i], x[4 * i + 1] + s[4 * i + 1], x[4 * i + 2] + s[4 * i + 2], x[4 * i + 3] + s[4 * i + 3]);
}

// --- the pipe probe: does SHA-512's integer work hide under tensor-core work on one SM? ---------------------------------
// MW tensor warps and HW SHA-512 warps in one CTA an SM. mode bit 0 runs the tensor warps (mma.sync m16n8k32 s8 on
// register operands, 8 independent accumulators; with WG on sm_90a, wgmma m64n64k32 s8 from shared memory instead, a
// warpgroup per 4 warps), bit 1 the SHA warps (chained compressions on registers). Timing only.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAVE_WGMMA 1
#endif
DEV u64 smem_desc(u32 addr) { return (u64)((addr & 0x3FFFF) >> 4) | ((u64)(128 >> 4) << 16) | ((u64)(256 >> 4) << 32); }
template <int MW, int HW, bool WG>
__global__ void __launch_bounds__(32 * (MW + HW), 1) probe_kernel(int mma_iters, int sha_iters, int mode, u64* sink) {
  __shared__ __align__(1024) uint8_t pm[16384];
  const int tid = threadIdx.x, warp = tid >> 5;
  for (int i = tid; i < 16384 / 4; i += blockDim.x) reinterpret_cast<u32*>(pm)[i] = i * 2654435761u;
  __syncthreads();
  if (warp < MW) {
    if (!(mode & 1)) return;
    if constexpr (WG) {
#ifdef HAVE_WGMMA
      int d[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) d[i] = 0;
      const u64 da = smem_desc(smem_u32(pm)), db = smem_desc(smem_u32(pm + 8192));
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      for (int it = 0; it < mma_iters; ++it) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p;\n}\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
              "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
              "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
              "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
            : "l"(da), "l"(db), "r"(1));
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 4;" ::: "memory");
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      int s = 0;
#pragma unroll
      for (int i = 0; i < 32; ++i) s += d[i];
      if (s == 0x12345678) sink[tid] = s;
#endif
    } else {
      u32 a[4] = {(u32)tid, (u32)tid * 3u, (u32)tid * 5u, (u32)tid * 7u}, b[8][2];
      int acc[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        b[j][0] = tid * 11u + j; b[j][1] = tid * 13u + j;
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[j][c] = 0;
      }
      for (int it = 0; it < mma_iters; ++it)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_s8(acc[j], a, b[j]);
      int s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) s += acc[j][0] ^ acc[j][1] ^ acc[j][2] ^ acc[j][3];
      if (s == 0x12345678) sink[tid] = s;
    }
    return;
  }
  if (!(mode & 2)) return;
  u64 st[8], w[16];
  sha::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (u64)tid * 16 + i;
  for (int r = 0; r < sha_iters; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sha::compress(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) sink[tid] = st[1];
}

// --- host ---------------------------------------------------------------------------------------------------------------

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Lt {   // Y (m x n, s32) = A (m x k) B (n x k)^T, cuBLASLt autotuned over its heuristic's top 8
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t la{}, lb{}, lc{};
  cublasLtMatmulAlgo_t algo{};
  void* ws = nullptr;
  size_t wsz = 64 << 20;
  const int8_t *A = nullptr, *B = nullptr;
  int32_t* Y = nullptr;
  int algo_index = -1;
  float best_ms = 0;
  void init(const int8_t* A_, const int8_t* B_, int m, int n, int k, cudaStream_t st) {
    A = A_; B = B_;
    CL(cublasLtCreate(&lt));
    CK(cudaMalloc(&ws, wsz));
    CK(cudaMalloc(&Y, (size_t)m * n * 4));
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, k, n, k));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, k, m, k));
    CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, n, m, n));
    cublasLtMatmulPreference_t pref;
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    cublasLtMatmulHeuristicResult_t res[8];
    int got = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &got));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < got; ++i) {
      algo = res[i].algo;
      std::vector<double> t;
      bool ok = true;
      for (int r = 0; r < 6 && ok; ++r) {
        CK(cudaEventRecord(e0, st));
        ok = run(st) == CUBLAS_STATUS_SUCCESS;
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) t.push_back(ms);
      }
      if (ok && (algo_index < 0 || median(t) < best_ms)) { best_ms = (float)median(t); algo_index = i; }
    }
    if (algo_index < 0) { fprintf(stderr, "no cuBLASLt int8 algorithm\n"); exit(3); }
    algo = res[algo_index].algo;
  }
  cublasStatus_t run(cudaStream_t st) {
    const int32_t one = 1, zero = 0;
    return cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, Y, lc, Y, lc, &algo, ws, wsz, st);
  }
};

struct Setup {
  int M, N, K;
  u32 R;
  int8_t *A = nullptr, *B = nullptr;
  uint8_t* Cb = nullptr;
  Committer cm;
  Geo g{};
  int sms = 0, grid = 0;
  u64 n = 0;
  void init(const Params& p, int M_, int N_, int K_, u32 R_) {
    M = M_; N = N_; K = K_; R = R_;
    if (K % BK || N % BN || M <= 0 || BM % R) { fprintf(stderr, "K %% 64, N %% 128, R | 128\n"); exit(2); }
    n = tile_leaves(M, 2ull * N, R);
    CK(cudaMalloc(&A, (size_t)M * K)); CK(cudaMalloc(&B, (size_t)N * K)); CK(cudaMalloc(&Cb, (size_t)M * N * 2 + 64));
    fill<<<1024, 256>>>((uint8_t*)A, (size_t)M * K, 11u);
    fill<<<1024, 256>>>((uint8_t*)B, (size_t)N * K, 13u);
    cm.init(p, n, 32 * R, false, false, 256ull * R);
    int dev; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    sms = prop.multiProcessorCount;
    const int tiles_m = (M + BM - 1) / BM, tiles = tiles_m * (N / BN);
    grid = std::min(tiles, sms);
    g = Geo{A, B, Cb, M, N, K, tiles_m, tiles, (u64)(M + R - 1) / R, R, cm.S, cm.lv[0]};
    CK(cudaFuncSetAttribute(gemm_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_GEMM));
    CK(cudaFuncSetAttribute(gemm_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_GEMM));
    CK(cudaFuncSetAttribute(gemm_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_GEMM));
    if (cm.width > n) pad_fill<<<(unsigned)((cm.width - n + 255) / 256), 256>>>(cm.lv[0], n, cm.width);
    cm.salts<true>(nullptr, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
  }
  void gemm(int hw, cudaStream_t st, bool pf = false, int diag = 0, bool gpu_fence = false) {
    Geo q = g;
    q.pf = pf;
    q.diag = diag;
    q.gpu_fence = gpu_fence;
    if (hw == 0) gemm_kernel<0><<<grid, MMA_THREADS, SMEM_GEMM, st>>>(q);
    else if (hw == 4) gemm_kernel<4><<<grid, MMA_THREADS + 128, SMEM_GEMM, st>>>(q);
    else gemm_kernel<8><<<grid, MMA_THREADS + 256, SMEM_GEMM, st>>>(q);
  }
  // the plain GEMM on st and hash_kernel<hw> beside it on a second stream, joined back into st
  u32* flags = nullptr;
  u32 epoch = 0;
  cudaStream_t st2{};
  cudaEvent_t fork{}, join{};
  void persist(int hw, cudaStream_t st) {
    if (!flags) {
      CK(cudaMalloc(&flags, (size_t)g.tiles * 4));
      CK(cudaMemset(flags, 0, (size_t)g.tiles * 4));
      CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
      CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    }
    Geo q = g;
    q.flags = flags;
    q.epoch = ++epoch;
    CK(cudaEventRecord(fork, st));
    gemm_kernel<0><<<grid, MMA_THREADS, SMEM_GEMM, st>>>(q);
    CK(cudaStreamWaitEvent(st2, fork));
    if (hw == 4) hash_kernel<4><<<grid, 128, 0, st2>>>(q, grid);
    else hash_kernel<8><<<grid, 256, 0, st2>>>(q, grid);
    CK(cudaEventRecord(join, st2));
    CK(cudaStreamWaitEvent(st, join));
  }
  void sep(cudaStream_t st) { sep_leaves<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(g, n); }
  void rec(cudaStream_t st) { cm.tiles(Cb, M, 2ull * N, R, false, st); }
  std::vector<uint8_t> levels() {
    u64 used = 0;
    for (int d = 0; d <= cm.depth; ++d) used += cm.width >> d;
    std::vector<u64> w(used * 8);
    CK(cudaMemcpy(w.data(), cm.levels, w.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> v(used * 64);   // digest byte order: the device keeps each 8 bytes as a big-endian word
    for (size_t i = 0; i < w.size(); ++i) sha::put_be64(v.data() + 8 * i, w[i]);
    return v;
  }
  void clear_levels() { CK(cudaMemset(cm.levels, 0xA5, n * 64)); for (int d = 1; d <= cm.depth; ++d) CK(cudaMemset(cm.lv[d], 0xA5, (cm.width >> d) * 64)); }
};

static int check(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "check PARAMS OUT M N K R\n"); return 2; }
  const Params p = read_params(argv[2]);
  const std::string out = argv[3];
  Setup s;
  s.init(p, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), (u32)atoi(argv[7]));
  const size_t cbytes = (size_t)s.M * s.N * 2;
  // the GEMM against cuBLASLt
  Lt lt;
  lt.init(s.A, s.B, s.M, s.N, s.K, 0);
  CL(lt.run(0));
  uint8_t* ref;
  CK(cudaMalloc(&ref, cbytes));
  s32_to_bf16<<<1024, 256>>>(lt.Y, ref, (size_t)s.M * s.N);
  std::vector<uint8_t> want(cbytes), got(cbytes);
  CK(cudaMemcpy(want.data(), ref, cbytes, cudaMemcpyDeviceToHost));
  // each arm from poisoned levels and C: its levels must equal the first arm's
  const char* arms[] = {"g+sep", "g+rec", "f4", "f8", "f4pf", "f8pf", "p4", "p8"};
  std::vector<uint8_t> first;
  bool ok = true;
  for (int a = 0; a < 8; ++a) {
    s.clear_levels();
    CK(cudaMemset(s.Cb, 0xA5, cbytes));
    if (s.cm.width > s.n) pad_fill<<<(unsigned)((s.cm.width - s.n + 255) / 256), 256>>>(s.cm.lv[0], s.n, s.cm.width);
    if (a == 0) { s.gemm(0, 0); s.sep(0); }
    if (a == 1) { s.gemm(0, 0); s.rec(0); }
    if (a == 2) s.gemm(4, 0);
    if (a == 3) s.gemm(8, 0);
    if (a == 4) s.gemm(4, 0, true);
    if (a == 5) s.gemm(8, 0, true);
    if (a == 6) s.persist(4, 0);
    if (a == 7) s.persist(8, 0);
    s.cm.nodes(0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    CK(cudaMemcpy(got.data(), s.Cb, cbytes, cudaMemcpyDeviceToHost));
    const bool c_ok = got == want;
    const auto lv = s.levels();
    if (a == 0) first = lv;
    const bool l_ok = lv == first;
    ok &= c_ok && l_ok;
    printf("{\"check_arm\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"r\": %u, \"leaves\": %llu, \"c_equals_cublaslt\": %s, \"levels_equal\": %s}\n",
           arms[a], s.M, s.N, s.K, s.R, (unsigned long long)s.n, c_ok ? "true" : "false", l_ok ? "true" : "false");
  }
  if (!out.empty() && out != "-") {
    writef(out + "/operand.bin", want.data(), cbytes);
    writef(out + "/levels.bin", first.data(), first.size());
    FILE* f = fopen((out + "/dump.json").c_str(), "w");
    fprintf(f, "{\"m\": %d, \"k\": %d, \"r\": %u, \"gemm\": [%d, %d, %d]}\n", s.M, 2 * s.N, s.R, s.M, s.N, s.K);
    fclose(f);
    const auto pv = readf(argv[2]);
    writef(out + "/params.bin", pv.data(), pv.size());
  }
  printf("{\"check\": %s}\n", ok ? "true" : "false");
  return ok ? 0 : 1;
}

static int bench(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "bench PARAMS M N K R REPS\n"); return 2; }
  const Params p = read_params(argv[2]);
  Setup s;
  s.init(p, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (u32)atoi(argv[6]));
  const int reps = atoi(argv[7]);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Lt lt;
  lt.init(s.A, s.B, s.M, s.N, s.K, st);
  cudaEvent_t e[4];
  for (auto& x : e) CK(cudaEventCreate(&x));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) { float t; CK(cudaEventElapsedTime(&t, a, b)); return (double)t; };
  // arm: name, body (events e1 / e2 split it after its first and second parts)
  enum { LT, G, GS, REC, F4, F8, H, SALT, NARMS };
  const char* names[NARMS] = {"lt", "g", "gs", "rec", "f4", "f8", "h", "salt"};
  std::vector<double> t[NARMS], t1[NARMS], t2[NARMS], mhz[NARMS];
  Nv nv;
  auto run = [&](int a) {
    spin<<<1, 1, 0, st>>>(SPIN);
    CK(cudaEventRecord(e[0], st));
    switch (a) {
      case LT: CL(lt.run(st)); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); break;
      case G: s.gemm(0, st); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); break;
      case GS: s.gemm(0, st); CK(cudaEventRecord(e[1], st)); s.sep(st); CK(cudaEventRecord(e[2], st)); s.cm.nodes(st); break;
      case REC: s.gemm(0, st); CK(cudaEventRecord(e[1], st)); s.rec(st); CK(cudaEventRecord(e[2], st)); s.cm.nodes(st); break;
      case F4: s.gemm(4, st); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); s.cm.nodes(st); break;
      case F8: s.gemm(8, st); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); s.cm.nodes(st); break;
      case H: s.sep(st); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); s.cm.nodes(st); break;
      case SALT: s.cm.salts<true>(nullptr, st); CK(cudaEventRecord(e[1], st)); CK(cudaEventRecord(e[2], st)); break;
    }
    CK(cudaEventRecord(e[3], st));
    CK(cudaEventSynchronize(e[3]));
    return std::array<double, 3>{ms(e[0], e[3]), ms(e[0], e[1]), ms(e[1], e[2])};
  };
  for (int w = 0; w < 2; ++w)
    for (int a = 0; a < NARMS; ++a) run(a);
  CK(cudaGetLastError());
  for (int r = 0; r < reps; ++r)
    for (int a = 0; a < NARMS; ++a) {
      const auto v = run(a);
      t[a].push_back(v[0]); t1[a].push_back(v[1]); t2[a].push_back(v[2]);
      mhz[a].push_back(nv.sm());
    }
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("{\"bench\": \"fuse\", \"gpu\": \"%s\", \"sms\": %d, \"m\": %d, \"n\": %d, \"k\": %d, \"r\": %u, \"leaf_bytes\": %u, \"leaves\": %llu, "
         "\"depth\": %d, \"operand_bytes\": %llu, \"lt_algo_index\": %d, \"reps\": %d, \"arms\": {",
         prop.name, s.sms, s.M, s.N, s.K, s.R, 32 * s.R, (unsigned long long)s.n, s.cm.depth, (unsigned long long)s.M * s.N * 2,
         lt.algo_index, reps);
  for (int a = 0; a < NARMS; ++a) {
    printf("%s\"%s\": {\"ms\": %.5f, \"part1_ms\": %.5f, \"part2_ms\": %.5f, \"mhz\": %.0f, \"all\": [", a ? ", " : "", names[a],
           median(t[a]), median(t1[a]), median(t2[a]), median(mhz[a]));
    for (size_t i = 0; i < t[a].size(); ++i) printf("%s%.5f", i ? ", " : "", t[a][i]);
    printf("], \"mhz_all\": [");
    for (size_t i = 0; i < mhz[a].size(); ++i) printf("%s%.0f", i ? ", " : "", mhz[a][i]);
    printf("]}");
  }
  printf("}}\n");
  fflush(stdout);
  return 0;
}

// The fused kernel taken apart (timing only; the _nohash and _nomma arms' levels are wrong by design): the GEMM alone, the
// separate leaves alone (and with the L2 prefetch), and the fused kernel at 4 and 8 hash warps: whole, with the prefetch,
// with its hash warps exiting at once, and with its MMA warps skipping the k-loop (the hash warps' own rate in the CTA);
// the handoff at GPU scope, or none; and p4 / p8, the plain GEMM beside the persistent hash_kernel.
static int diag(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "diag PARAMS M N K R REPS\n"); return 2; }
  const Params p = read_params(argv[2]);
  Setup s;
  s.init(p, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), (u32)atoi(argv[6]));
  const int reps = atoi(argv[7]);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const char* names[] = {"g", "sep", "seppf", "f4", "f4pf", "f4_nohash", "f4_nomma", "f8", "f8pf", "f8_nohash", "f8_nomma",
                         "f4gf", "f4gf_nohash", "f4_nohash_nosig", "f8gf", "p4", "p8"};
  constexpr int NA = 17;
  auto body = [&](int a) {
    if (a == 0) s.gemm(0, st);
    if (a == 1) s.sep(st);
    if (a == 2) { Geo q = s.g; q.pf = true; sep_leaves<<<(unsigned)((s.n + 255) / 256), 256, 0, st>>>(q, s.n); }
    if (a >= 3 && a < 11) {
      const int hw = a < 7 ? 4 : 8, v = (a - 3) % 4;
      s.gemm(hw, st, v == 1, v == 2 ? 2 : v == 3 ? 1 : 0);
    }
    if (a == 11) s.gemm(4, st, false, 0, true);
    if (a == 12) s.gemm(4, st, false, 2, true);
    if (a == 13) s.gemm(4, st, false, 6);
    if (a == 14) s.gemm(8, st, false, 0, true);
    if (a == 15) s.persist(4, st);
    if (a == 16) s.persist(8, st);
  };
  std::vector<double> t[NA], mz;
  Nv nv;
  for (int r = -2; r < reps; ++r)
    for (int a = 0; a < NA; ++a) {
      spin<<<1, 1, 0, st>>>(SPIN);
      CK(cudaEventRecord(e0, st));
      body(a);
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float m; CK(cudaEventElapsedTime(&m, e0, e1));
      if (r >= 0) { t[a].push_back(m); if (a == 3) mz.push_back(nv.sm()); }
    }
  CK(cudaGetLastError());
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("{\"diag\": {\"gpu\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"r\": %u, \"mhz\": %.0f", prop.name, s.M, s.N, s.K, s.R, median(mz));
  for (int a = 0; a < NA; ++a) printf(", \"%s\": %.5f", names[a], median(t[a]));
  printf("}}\n");
  return 0;
}

// Binary, 4-ary, 8-ary and 16-ary trees over n leaves (n a power of the arity, or the top level partial: ceil), a level a
// launch, captured as one CUDA graph; the binary arm is also run as Committer::nodes() for comparison.
static int tree(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "tree PARAMS N REPS\n"); return 2; }
  const Params p = read_params(argv[2]);
  const u64 n = strtoull(argv[3], nullptr, 10);
  const int reps = atoi(argv[4]);
  Committer cm;
  cm.init(p, n, 256, false);
  u64 *a, *b;
  CK(cudaMalloc(&a, n * 64 + 64 * 16)); CK(cudaMalloc(&b, n * 64 + 64 * 16));
  fill<<<1024, 256>>>((uint8_t*)a, n * 64, 5u);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  auto levels = [&](int arity, cudaStream_t s) {
    u64 cnt = n;
    int lvl = 0;
    u64 *in = a, *out = b;
    while (cnt > 1) {
      const u64 n_out = (cnt + arity - 1) / arity;
      const unsigned th = n_out >= 65536 ? 256 : 64, bl = (unsigned)((n_out + th - 1) / th);
      switch (arity) {
        case 2: anode_level<2><<<bl, th, 0, s>>>(in, out, n_out, lvl); break;
        case 4: anode_level<4><<<bl, th, 0, s>>>(in, out, n_out, lvl); break;
        case 8: anode_level<8><<<bl, th, 0, s>>>(in, out, n_out, lvl); break;
        default: anode_level<16><<<bl, th, 0, s>>>(in, out, n_out, lvl); break;
      }
      std::swap(in, out);
      cnt = n_out;
      ++lvl;
    }
    return lvl;
  };
  const int ar[4] = {2, 4, 8, 16};
  cudaGraphExec_t gx[5];
  int depth[5];
  for (int i = 0; i < 5; ++i) {
    cudaGraph_t gr;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    if (i < 4) depth[i] = levels(ar[i], st);
    else { cm.nodes(st); depth[i] = cm.depth; }
    CK(cudaStreamEndCapture(st, &gr));
    CK(cudaGraphInstantiate(&gx[i], gr, 0));
  }
  std::vector<double> t[5], mz;
  for (int r = -2; r < reps; ++r)
    for (int i = 0; i < 5; ++i) {
      spin<<<1, 1, 0, st>>>(SPIN);
      CK(cudaEventRecord(e0, st));
      CK(cudaGraphLaunch(gx[i], st));
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float m; CK(cudaEventElapsedTime(&m, e0, e1));
      if (r >= 0) { t[i].push_back(m); if (i == 0) mz.push_back(nv.sm()); }
    }
  CK(cudaGetLastError());
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("{\"tree\": {\"gpu\": \"%s\", \"leaves\": %llu, \"mhz\": %.0f", prop.name, (unsigned long long)n, median(mz));
  const char* nm[5] = {"a2", "a4", "a8", "a16", "committer_binary"};
  for (int i = 0; i < 5; ++i) printf(", \"%s\": {\"levels\": %d, \"ms\": %.5f}", nm[i], depth[i], median(t[i]));
  printf("}}\n");
  return 0;
}

// ChaCha20 salts for n leaves: the keystream's first 4 leaves dumped (OUT/chacha.bin, key and nonce in OUT/chacha.json)
// for check_chacha.py; then, timed: the keystream alone, the salt phase on given (ChaCha20) salts, and on PRF salts.
static int chacha(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "chacha PARAMS OUT N REPS\n"); return 2; }
  const Params p = read_params(argv[2]);
  const std::string out = argv[3];
  const u64 n = strtoull(argv[4], nullptr, 10);
  const int reps = atoi(argv[5]);
  ChaKey key;
  for (int i = 0; i < 8; ++i) key.k[i] = 0x03020100u + 0x04040404u * i;   // bytes 00 .. 1f
  key.nonce[0] = 0x09000000u; key.nonce[1] = 0x4a000000u; key.nonce[2] = 0;   // RFC 8439 2.3.2's nonce
  uint4* y;
  CK(cudaMalloc(&y, n * 192 + 64));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const u64 nb = 3 * n;
  chacha_salts<<<(unsigned)((nb + 255) / 256), 256, 0, st>>>(key, nb, y);
  CK(cudaStreamSynchronize(st));
  std::vector<uint8_t> h(192 * std::min<u64>(n, 4));
  CK(cudaMemcpy(h.data(), y, h.size(), cudaMemcpyDeviceToHost));
  writef(out + "/chacha.bin", h.data(), h.size());
  FILE* f = fopen((out + "/chacha.json").c_str(), "w");
  fprintf(f, "{\"key\": \"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\", \"nonce\": \"000000090000004a00000000\", \"counter0\": 0, \"leaves\": %llu}\n",
          (unsigned long long)std::min<u64>(n, 4));
  fclose(f);
  Committer cm;
  cm.init(p, n, 256, false);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  Nv nv;
  std::vector<double> tk, tg, tp, mz;
  for (int r = -2; r < reps; ++r) {
    double v[3];
    for (int a = 0; a < 3; ++a) {
      spin<<<1, 1, 0, st>>>(SPIN);
      CK(cudaEventRecord(e0, st));
      if (a == 0) chacha_salts<<<(unsigned)((nb + 255) / 256), 256, 0, st>>>(key, nb, y);
      if (a == 1) cm.salts<false>(reinterpret_cast<const uint8_t*>(y), st);
      if (a == 2) cm.salts<true>(nullptr, st);
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float m; CK(cudaEventElapsedTime(&m, e0, e1));
      v[a] = m;
    }
    if (r >= 0) { tk.push_back(v[0]); tg.push_back(v[1]); tp.push_back(v[2]); mz.push_back(nv.sm()); }
  }
  CK(cudaGetLastError());
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("{\"chacha\": {\"gpu\": \"%s\", \"leaves\": %llu, \"keystream_ms\": %.5f, \"salt_phase_given_ms\": %.5f, \"salt_phase_prf_ms\": %.5f, "
         "\"keystream_gb_per_s\": %.2f, \"mhz\": %.0f}}\n",
         prop.name, (unsigned long long)n, median(tk), median(tg), median(tp), n * 192 / median(tk) * 1e-6, median(mz));
  return 0;
}

// The pipe probe at one configuration: iterations calibrated so the tensor warps alone take ~T_MS and the SHA warps alone
// `ratio` of that, then the tensor warps alone, the SHA warps alone and both, interleaved. overlap = (T1 + T2 - T3) /
// min(T1, T2): 1 when the shorter hides entirely, 0 when they serialize, negative when running both costs more than that.
template <int MW, int HW, bool WG>
static void probe_cfg(double ratio, int reps, int sms, cudaStream_t st, u64* sink, Nv& nv, const char* gpu) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time = [&](int mi, int si, int mode) {
    spin<<<1, 1, 0, st>>>(SPIN);
    CK(cudaEventRecord(e0, st));
    probe_kernel<MW, HW, WG><<<sms, 32 * (MW + HW), 0, st>>>(mi, si, mode, sink);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float m; CK(cudaEventElapsedTime(&m, e0, e1));
    return (double)m;
  };
  constexpr double T_MS = 2.0;
  int mi = 1024, si = 64;
  for (int k = 0; k < 2; ++k) {
    const double t1 = time(mi, si, 1), t2 = time(mi, si, 2);
    mi = std::max(1, (int)(mi * T_MS / t1));
    si = std::max(1, (int)(si * T_MS * ratio / t2));
  }
  std::vector<double> t[3], mz;
  for (int r = -2; r < reps; ++r)
    for (int m = 1; m <= 3; ++m) {
      const double v = time(mi, si, m);
      if (r >= 0) { t[m - 1].push_back(v); if (m == 3) mz.push_back(nv.sm()); }
    }
  CK(cudaGetLastError());
  const double T1 = median(t[0]), T2 = median(t[1]), T3 = median(t[2]);
  const double ops = WG ? (double)sms * (MW / 4) * mi * 64 * 64 * 32 * 2 : (double)sms * MW * mi * 8 * 16 * 8 * 32 * 2;
  const double comps = (double)sms * HW * 32 * si;
  printf("{\"probe\": {\"gpu\": \"%s\", \"mma_warps\": %d, \"sha_warps\": %d, \"wgmma\": %s, \"ratio\": %.2f, \"mma_iters\": %d, "
         "\"sha_iters\": %d, \"t_mma_ms\": %.4f, \"t_sha_ms\": %.4f, \"t_both_ms\": %.4f, \"overlap\": %.3f, "
         "\"tops_alone\": %.1f, \"gcomp_per_s_alone\": %.3f, \"mhz\": %.0f}}\n",
         gpu, MW, HW, WG ? "true" : "false", ratio, mi, si, T1, T2, T3, (T1 + T2 - T3) / std::min(T1, T2),
         ops / T1 * 1e-9, comps / T2 * 1e-6, median(mz));
  fflush(stdout);
}

static int probe(int argc, char** argv) {
  const int reps = argc >= 3 ? atoi(argv[2]) : 9;
  int dev; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  u64* sink;
  CK(cudaMalloc(&sink, 1024 * 8));
  Nv nv;
  const int sms = prop.multiProcessorCount;
  for (double ratio : {1.0, 0.25}) {
    probe_cfg<8, 4, false>(ratio, reps, sms, st, sink, nv, prop.name);
    probe_cfg<8, 8, false>(ratio, reps, sms, st, sink, nv, prop.name);
    probe_cfg<4, 4, false>(ratio, reps, sms, st, sink, nv, prop.name);
    if (prop.major == 9) {
      probe_cfg<8, 4, true>(ratio, reps, sms, st, sink, nv, prop.name);
      probe_cfg<4, 4, true>(ratio, reps, sms, st, sink, nv, prop.name);
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "probe")) return probe(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "diag")) return diag(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "check")) return check(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return bench(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "tree")) return tree(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "chacha")) return chacha(argc, argv);
  fprintf(stderr, "usage: fuse check|bench|tree|chacha|probe|diag ...\n");
  return 2;
}
