// ro-bound-sketch-blake3/v0 in the fused p512 decode + FP8 GEMM on sm_120 (`bound.py` is the reference): ra-codec-wide's
// `ww_gemm` (benchmarks/word_codec/wide.cu, included, so its launchers and the unbound baseline `ww_gemm_sketch` are in
// the same library) with the sketch's ±1 matrix replaced by B = R ⊕ M, M the window's masks and R the matmul's bytes:
// a share of the previous linear's output (tile), or for words 1 … 3 of a lane's record the GEMM's own accumulators one
// word earlier (slice). Build as wide_run.sh builds wide.cu, at WW_BITS=512, with -I benchmarks/bandwidth.
//
// One tile a CTA, warp i holding task (4 G + i, kc), lane l's record L = 4 words of 64 bytes, one per four k32 steps.
// The words are taken in turn (a rolled loop): the sketch of word a (its encoded limbs against B), its decode in place,
// then its four k32 MMAs, which give the snapshots that bind word a + 1. So a slice step reads accumulators that exist
// only once the previous word's product is done.

#include "../word_codec/wide.cu"

#ifdef WW_HOST
#error "bound.cu is device code"
#endif

static_assert(WW_BITS == 512 && L == 4 && N == 16 && PB == 512, "p512: a lane's record is four words of four k32 steps");
static_assert(KW == 1 && TB == 1 && GEMM_WARPS == 4 && SK_M == 16, "one tile a CTA: four warps at one chunk");

static const char BOUND_NAME[] = "ro-bound-sketch-blake3/v0";

#ifndef RB_K
#define RB_K KMAX  // the one k the bound kernel is built for
#endif
static_assert(RB_K >= 1 && RB_K <= KMAX, "RB_K");

namespace rb {

constexpr int UNBOUND = 0, TILE = 1, PLAIN = 2;

__device__ __forceinline__ void xor4(uint4& b, const uint4& r) { b.x ^= r.x, b.y ^= r.y, b.z ^= r.z, b.w ^= r.w; }

// The lane's tile R for its next step, advancing: R word idx = 2048 warp + 128 s + 4 lane (+ register) is z_u's word
// idx mod s4, zprev's word (u s4 + idx mod s4) mod z4 (`bound.z_share`).
struct ZCursor {
  const uint32_t* z;
  uint32_t base, r, step, s4, z4;
  __device__ __forceinline__ uint4 next() {
    uint32_t o = base + r;
    if (o >= z4) o -= z4;
    r += step;
    if (r >= s4) r -= s4;
    return __ldg(reinterpret_cast<const uint4*>(z + o));
  }
};

// BIND: UNBOUND (masks are wide_sketch's ±1 sign fragments, R = 0: ro-sketch-blake3/v0 itself), TILE, a slice width
// 32, 64, 128, or PLAIN (decode and GEMM, no sketch: the receipts' baseline at every batch). masks:
// `bound.mask_fragments` (T · SK_M · 512 bytes); zprev: Mb × Nz FP32 words, z4 = Mb · Nz, s4 the share
// (`bound.share_words`); sk null or 256 int32 a tile; tags null, or the receipts (vector v = 16 blockIdx.x + r of the
// call at index jbase + v) at tags[8 v …]. NB8 = 8 (M = 33 … 64) holds 32 accumulators: two CTAs an SM.
template <int KR, int NB8, bool PARTIALS, int BIND>
__global__ void __launch_bounds__(GEMM_WARPS * 32, NB8 == 8 ? 2 : WW_OCC)
    bound_kernel(const uint32_t* __restrict__ E, const uint8_t* __restrict__ x, float* __restrict__ y, int Mb, int Nn,
                 int Kd, uint64_t wbase, uint64_t jbase, bw::Key3 key, uint32_t* __restrict__ tags,
                 const uint4* __restrict__ masks, int32_t* __restrict__ sk, const uint32_t* __restrict__ zprev,
                 uint32_t z4, uint32_t s4) {
  constexpr int UNITS = TASK_COLS / 16, SW = BIND >= 32 ? BIND / 32 : 0;
  static_assert(BIND == UNBOUND || BIND == TILE || BIND == PLAIN || BIND == 32 || BIND == 64 || BIND == 128, "BIND");
  __shared__ __align__(16) uint8_t xs[NB8 * 8][TASK_COLS];  // row r's 16-byte unit c at c ^ (r mod 8), as gemm_kernel
  __shared__ int32_t skp[GEMM_WARPS][16][SK_ROW];
  const int nchunks = Kd / TASK_COLS;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int kc = blockIdx.x % nchunks, nb = (blockIdx.x / nchunks) * GEMM_WARPS + warp;
  const uint64_t q = (uint64_t)nb * nchunks + kc;
  for (int e = threadIdx.x; e < NB8 * 8 * UNITS; e += GEMM_WARPS * 32) {
    const int b = e / UNITS, c = e % UNITS;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < Mb) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * (c ^ (b & 7))]) = v;
  }
  __syncthreads();
  uint32_t w[L][N];
  const uint32_t* rec = E + q * (32ull * L * N) + lane;
#pragma unroll
  for (int i = 0; i < L; ++i)
#pragma unroll
    for (int e = 0; e < N; ++e) w[i][e] = __ldcs(rec + 32 * (i * N + e));

  ZCursor zc{};
  uint4 rz[4];
  if constexpr (BIND != UNBOUND && BIND != PLAIN) {
    zc = {zprev, (blockIdx.x * s4) % z4, (2048u * warp + 4u * lane) % s4, 128u % s4, s4, z4};
#pragma unroll
    for (int e = 0; e < 4; ++e) rz[e] = zc.next();
  }
  const uint4* mg = masks + (size_t)warp * SK_M * 32 + lane;
  uint32_t t[N], t32[N];
  if constexpr (KR > 0) tw_init(t, t32, wbase + q * (32ull * L) + lane, LANES);
  int d[2][4] = {};
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

#pragma unroll 1
  for (int a = 0; a < L; ++a) {
    // the sketch of word a (encoded, in w[0]): steps 4a … 4a + 3
    if constexpr (BIND != PLAIN) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t av[4] = {w[0][4 * e], w[0][4 * e + 1], w[0][4 * e + 2], w[0][4 * e + 3]};
        uint4 b = __ldg(mg + 32 * (4 * a + e));
        if constexpr (BIND != UNBOUND) xor4(b, rz[e]);
        mma_s8(d[0], av, b.x, b.y);
        mma_s8(d[1], av, b.z, b.w);
      }
    }
    if constexpr (BIND == TILE) {
      if (a + 1 < L) {
#pragma unroll
        for (int e = 0; e < 4; ++e) rz[e] = zc.next();
      }
    }
    if constexpr (KR > 0) {
      decode_tw<KR>(w[0], t);
      tw_step(t, t32);
    }
    // its four k32 steps, tt = 4a + e; a slice's end snapshots R for word a + 1
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tt = 4 * a + e;
      const uint32_t av[4] = {take4(w, 128 * e), take4(w, 128 * e + 32), take4(w, 128 * e + 64), take4(w, 128 * e + 96)};
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][16 * ((2 * tt) ^ g) + 4 * tq]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][16 * ((2 * tt + 1) ^ g) + 4 * tq]);
        mma_e4m3(acc[j], av, b0, b1);
      }
      if constexpr (SW > 0) {
        if ((e + 1) % SW == 0) {
#pragma unroll
          for (int e1 = 0; e1 < SW; ++e1) {
            uint4 v = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int j = 0; j < NB8; ++j)
              if (NB8 >= SW ? j % SW == e1 : j == e1 % NB8)
                xor4(v, make_uint4(__float_as_uint(acc[j][0]), __float_as_uint(acc[j][1]), __float_as_uint(acc[j][2]),
                                   __float_as_uint(acc[j][3])));
            rz[e + 1 - SW + e1] = v;
          }
        }
      }
    }
    uint32_t w0[N];
    copy_k<N>(w0, w[0]);
#pragma unroll
    for (int i = 0; i + 1 < L; ++i) copy_k<N>(w[i], w[i + 1]);
    copy_k<N>(w[L - 1], w0);
  }

#pragma unroll
  for (int j = 0; j < NB8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
      if (b < Mb) {
        if constexpr (PARTIALS) y[((size_t)kc * Mb + b) * Nn + n] = acc[j][e];
        else atomicAdd(y + (size_t)b * Nn + n, acc[j][e]);
      }
    }
  }
  if constexpr (BIND == PLAIN) return;
#pragma unroll
  for (int c8 = 0; c8 < 2; ++c8)
#pragma unroll
    for (int e = 0; e < 4; ++e) skp[warp][g + 8 * (e >> 1)][8 * c8 + 2 * tq + (e & 1)] = d[c8][e];
  __syncthreads();
  for (int i = threadIdx.x; i < 16 * SK_K; i += GEMM_WARPS * 32) {
    const int r = i / SK_K, c = i % SK_K;
    int v = 0;
#pragma unroll
    for (int u = 0; u < GEMM_WARPS; ++u) v += skp[u][r][c];
    skp[0][r][c] = v;
    if (sk) sk[(size_t)blockIdx.x * (16 * SK_K) + i] = v;
  }
  if (tags) {
    __syncthreads();
    if (warp == 0) {
      const int r = lane & 15;
      uint32_t cv[8];
      sketch_tag(key, jbase + 16ull * blockIdx.x + r, skp[0][r], cv);
      if (lane < 16) {
        uint4* o = reinterpret_cast<uint4*>(tags + 8 * (16ull * blockIdx.x + r));
        o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
        o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
      }
    }
  }
}

// BLAKE3's key under a tag name, `bw::b3_key`'s derivation (wide.cu's `sketch_key` at ro-sketch-blake3/v0).
static bw::Key3 named_key(const uint8_t* seed, const char* name) {
  std::vector<uint8_t> p = bw::prefix_block(bw::TAG_DOMAIN, name, seed);
  uint64_t dg[8];
  bw::sha512_host(p.data(), p.size(), dg);
  bw::Key3 k;
  for (int i = 0; i < 8; ++i) {
    const uint32_t be = (uint32_t)(i & 1 ? dg[i / 2] : dg[i / 2] >> 32);
    k.k[i] = (be >> 24) | ((be >> 8) & 0xff00u) | ((be << 8) & 0xff0000u) | (be << 24);
  }
  return k;
}

static int nb8_of(int Mb) { return Mb <= 8 ? 1 : Mb <= 16 ? 2 : Mb <= 32 ? 4 : 8; }

#define RB_NB8(NB8, F) \
  do {                 \
    switch (NB8) {     \
      case 1: F(1); break; \
      case 2: F(2); break; \
      case 4: F(4); break; \
      default: F(8); break; \
    }                  \
  } while (0)

}  // namespace rb

extern "C" {

int rb_constants(int* out) {
  out[0] = RB_K;
  out[1] = WW_OCC;
  return 0;
}

// The bound fused GEMM. mode: 0 unbound (masks = wide_sketch's sign fragments, receipts under ro-sketch-blake3/v0), 1
// tile, 32 / 64 / 128 slice (masks = `bound.mask_fragments`, receipts under ro-bound-sketch-blake3/v0), 2 plain (no
// sketch: masks, zprev, sk, seed and tags unread, sk, seed and tags null); k must be RB_K;
// zprev Mb × Nz FP32 for a bound mode (Nz a multiple of 4); seed and tags both or neither (the receipts hashed in the
// kernel); sk null or 256 int32 a tile; Mb ≤ 64.
int rb_gemm(int mode, int k, const uint32_t* E, const uint8_t* x, float* y, int Mb, int Nn, int Kd, uint64_t wbase,
            int partials, const void* masks, int32_t* sk, uint64_t jbase, const uint8_t* seed, uint32_t* tags,
            const float* zprev, int Nz, cudaStream_t s) {
  if (mode != rb::UNBOUND && mode != rb::TILE && mode != rb::PLAIN && mode != 32 && mode != 64 && mode != 128) return 1;
  if (mode == rb::PLAIN ? sk || seed || tags : !masks || (seed != nullptr) != (tags != nullptr)) return 1;
  if (k != RB_K || Mb < 1 || Mb > 64 || Nn % (16 * GEMM_WARPS) || Kd % TASK_COLS || wbase % (32ull * L)) return 1;
  const int grid = (Nn / 16 / GEMM_WARPS) * (Kd / TASK_COLS);
  uint32_t z4 = 0, s4 = 0;
  if (mode != rb::UNBOUND && mode != rb::PLAIN) {
    if (!zprev || Nz < 1 || Nz % 4 || (uint64_t)Mb * Nz >= (1ull << 32)) return 1;
    z4 = (uint32_t)Mb * (uint32_t)Nz;
    s4 = (uint32_t)(4 * (((uint64_t)z4 + 4ull * grid - 1) / (4ull * grid)));
    if (s4 > 8192 || (uint64_t)grid * s4 >= (1ull << 32)) return 1;
  }
  bw::Key3 key = {};
  if (seed) key = rb::named_key(seed, mode == rb::UNBOUND ? SKETCH_NAME : BOUND_NAME);
  const uint4* mk = reinterpret_cast<const uint4*>(masks);
  const uint32_t* zp = reinterpret_cast<const uint32_t*>(zprev);
#define RB_LAUNCH(NB8, P, B)                                                                                      \
  rb::bound_kernel<RB_K, NB8, P, B><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key, tags, \
                                                                    mk, sk, zp, z4, s4)
#define RB_MODE(NB8, P)                       \
  do {                                        \
    switch (mode) {                           \
      case 0: RB_LAUNCH(NB8, P, 0); break;    \
      case 1: RB_LAUNCH(NB8, P, 1); break;    \
      case 2: RB_LAUNCH(NB8, P, 2); break;    \
      case 32: RB_LAUNCH(NB8, P, 32); break;  \
      case 64: RB_LAUNCH(NB8, P, 64); break;  \
      default: RB_LAUNCH(NB8, P, 128); break; \
    }                                         \
  } while (0)
#define RB_P(NB8) do { if (partials) RB_MODE(NB8, true); else RB_MODE(NB8, false); } while (0)
  RB_NB8(rb::nb8_of(Mb), RB_P);
#undef RB_P
#undef RB_MODE
#undef RB_LAUNCH
  return cudaGetLastError() ? 4 : 0;
}

// The receipts of n vectors of sk (16 int32 each) under ro-bound-sketch-blake3/v0, vector v's (index jbase + v) at
// tags[8 v …].
int rb_receipts(const int32_t* sk, uint64_t n, uint64_t jbase, const uint8_t* seed, uint32_t* tags, cudaStream_t s) {
  if (!n) return 0;
  sketch_tags_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(sk, n, jbase, rb::named_key(seed, BOUND_NAME), tags);
  return cudaGetLastError() ? 4 : 0;
}

// The leaves (16 receipts each) and RFC 6962 tree over n receipts of the workspace (`ww_workspace_tags`), root (64
// bytes) or null.
int rb_commit(void* workspace, uint64_t n, const uint8_t* seed, uint8_t* root, cudaStream_t s) {
  return commit_named(workspace, n, BOUND_NAME, seed, root, s);
}

// The bound kernel's resident CTAs an SM for mode, Mb and partials, or −1.
int rb_occupancy(int mode, int Mb, int partials) {
  if (Mb < 1 || Mb > 64) return -1;
  int r = -1;
#define RB_OCC(NB8, P, B) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, rb::bound_kernel<RB_K, NB8, P, B>, \
                                                                        GEMM_WARPS * 32, 0)
#define RB_MODE(NB8, P)                    \
  do {                                     \
    switch (mode) {                        \
      case 0: RB_OCC(NB8, P, 0); break;    \
      case 1: RB_OCC(NB8, P, 1); break;    \
      case 2: RB_OCC(NB8, P, 2); break;    \
      case 32: RB_OCC(NB8, P, 32); break;  \
      case 64: RB_OCC(NB8, P, 64); break;  \
      case 128: RB_OCC(NB8, P, 128); break; \
      default: return -1;                  \
    }                                      \
  } while (0)
#define RB_P(NB8) do { if (partials) RB_MODE(NB8, true); else RB_MODE(NB8, false); } while (0)
  RB_NB8(rb::nb8_of(Mb), RB_P);
#undef RB_P
#undef RB_MODE
#undef RB_OCC
  return r;
}

}  // extern "C"
