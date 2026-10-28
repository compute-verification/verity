// The remote session's prover kernels beyond the device reader's (`../codec_tensor/reader.cu`, included whole, so one
// library holds both and shares its pass constants kC and kSketchMid):
//
//   payload leaf j   x = SHA512(P(32768) ‖ payload_j) (sha512/row/v2 of the 4 KiB block), then the reader's hm96 leaf
//                    under salt_j = SHAKE256(D(store salt) ‖ key ‖ label ‖ u64be j) and its frame-v3-sha512 leaf
//                    digest at position j: the driver's `receipts.hidden(payload_j, salt_j)` in `audit.payload_tree`
//   gather           64-byte rows from a list of device addresses (a window's openings, across its passes' kept
//                    levels, their recomputed subtrees and the payload tree)
//   pass             one shard's part of a pass (its blocks at store positions base …): its receipts by one persistent
//                    grid on tensor cores, each subtree's leaves and levels hashed by a persistent grid on a second
//                    stream while the later blocks are read, keeping the shard's levels h0 … hc (hc: a chunk's subtree),
//                    and the padding leaves past the store this GPU hashes up to level hc (`rs_pass`)
//   top              the receipt tree's levels hc … H on one GPU, from every shard's and every padding share's level
//                    hc read over P2P, the root then written to each GPU's root ring (`rs_top`)
//   subtrees         the levels below h0 and the receipts of the subtrees a window's openings fall in, recomputed from
//                    a shard under the pass's constants (`rs_subtrees`)
//   in place         a store where its tensors hold it (vLLM's weights and KV pages, `device.InPlace`): a plan's
//                    positions mapped to their blocks' device addresses and store indices (`rs_map`), then their
//                    receipts read through the map and the receipt tree hashed behind them as `rs_pass` hashes it, the
//                    last leaf short and the padding to the tree's width included (`rs_mapped_pass`); an opening's
//                    subtree is its receipts (`rs_mapped_receipts`) and reader.cu's leaves and levels
//
// A store spans the visible GPUs, a contiguous shard of whole chunks on each, and the receipt tree is the one over the
// whole store in block order (`receipts.tree`): every position, leaf and node index is the store's. One library serves
// every GPU, each with its own copy of the constants (__constant__ is per device; the runtime follows the caller's
// current device). The pass constants are the library's own, so a second copy of the library, loaded from another
// path, recomputes openings under one pass's constants while the first runs the next passes under theirs.
//
// rb_set takes the payload tree's constants as it takes a pass's: row_prefix = sha512_row_v2_prefix(32768), the
// payload domain's id, salt_in = D(store salt) ‖ key ‖ label, positions = N.
//
// The SHA-512 compressions and the row/v2 digests of the receipt tree's leaves are libvc's device functions
// (`../hash/lib/vc_device.cuh`); the hm96 leaf after its salt and the frame-v3 leaf, pad and node are reader.cu's
// (`hm96_leaf`, `node_of`, rc's `pad_rt`), under its pass constants kC, kL, kNodeTpl and gNib, which its rb_set,
// rb_chain_set and rb_chain set. reader.cu's own kernels (rb_leaves, rb_pads, rb_tree and level_kernel) hash the same
// tree unfused, one kernel a level: the fused one's cross-check.
#include "../hash/lib/vc_device.cuh"
#include "../codec_tensor/reader.cu"

namespace rb {

// Payload leaves [c0, c0 + n) from their blocks (data, leaf c0's first), into dig.
__global__ void __launch_bounds__(128) payload_leaves_kernel(const uint4 *__restrict__ data, u64 c0, u64 n,
                                                             u64 *__restrict__ dig) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  const uint4 *E = data + t * (BLOCK / 16);
  u64 x[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = kC.row_mid[i];
#pragma unroll 1
  for (int k = 0; k < BLOCK / 128; ++k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 v = __ldg(E + 8 * k + i);
      w[2 * i] = be_word(v.x, v.y), w[2 * i + 1] = be_word(v.z, v.w);
    }
    sha512_compress(x, w);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x80ULL << 56, w[15] = (u64)(128 + BLOCK) * 8;
  sha512_compress(x, w);
  u64 d[8];
  hm96_leaf(c0 + t, x, d);
  store_digest(dig, tid, n, d);
}

// The 16-row sketch on tensor cores, fused with its receipt: σ (blocks × 16) = E · Sᵀ by mma.m16n8k32 as reader.cu's
// sketch64 with its first two n-tiles, so the CUDA cores stay free for the leaves and the tree of the chunk before.
// Lane (g, t)'s B registers of n-tile n at p are S's row 8 n + g at bytes 64 p + 16 t … + 15 (x, y for k-step 2 p, z, w
// for 2 p + 1), held as bits in one word (`pack16_kernel`): bit 8 i + 4 n + 2 s + b is element i of register (n, s, b),
// a bit of 1 is −1. A warp takes 32 blocks, stages their σ in shared memory and each lane hashes one into its receipt,
// sketch_hash_kernel's compression, so σ never leaves the SM. The sums are exact, so the receipts are sketch_kernel's.
constexpr int SK16_P = BLOCK / 64, SK16_WARPS = 8, SK16_PACK = SK16_P * 32;
constexpr u64 CONSUMER_READY_BLOCKS = 1ULL << 13;

__global__ void __launch_bounds__(128) pack16_kernel(const int8_t *__restrict__ signs, u32 *__restrict__ pk) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= SK16_PACK) return;
  const int p = q >> 5, g = (q & 31) >> 2, t = q & 3;
  u32 w = 0;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int nt = r >> 2, s = (r >> 1) & 1, b = r & 1;
    const int8_t *row = signs + (8 * nt + g) * BLOCK + 64 * p + 16 * t + 8 * s + 4 * b;
#pragma unroll
    for (int i = 0; i < 4; ++i) w |= (u32)(row[i] < 0) << (8 * i + r);
  }
  pk[q] = w;
}

// The receipts of the warp's 32 positions b0 … b0 + 31 (of n) into ys, S the warp's staging (32 · SK_ROWS int32) and sg
// the packed signs in shared memory: load(d, p) puts lane (g, t)'s 16 bytes at p of the tile's blocks 16 mt + g (+ 8
// for hi) in d[mt][hi], zero past the blocks (sk64_load's), and index(q) is position q's store index.
template <class Load, class Index>
__device__ __forceinline__ void sk16_tile_of(Load load, Index index, u64 n, u64 b0, const u32 *sg, int *S,
                                             u64 *__restrict__ ys, int lane) {
  const int g = lane >> 2, t = lane & 3;
  int acc[2][2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0;
  uint4 cur[2][2], nxt[2][2];
  load(cur, 0);
#pragma unroll 1
  for (int p = 0; p < SK16_P; ++p) {
    if (p + 1 < SK16_P) load(nxt, p + 1);
    const u32 w = sg[p * 32 + lane];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int r = 4 * nt;
      const u32 b00 = sign_reg(w, r), b01 = sign_reg(w, r + 1), b10 = sign_reg(w, r + 2), b11 = sign_reg(w, r + 3);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint4 &lo = cur[mt][0], &hi = cur[mt][1];
        imma16832(acc[mt][nt], lo.x, hi.x, lo.y, hi.y, b00, b01);
        imma16832(acc[mt][nt], lo.z, hi.z, lo.w, hi.w, b10, b11);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) cur[mt][0] = nxt[mt][0], cur[mt][1] = nxt[mt][1];
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int blk = 16 * mt + g;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      *(int2 *)(S + blk * SK_ROWS + 8 * nt + 2 * t) = make_int2(acc[mt][nt][0], acc[mt][nt][1]);
      *(int2 *)(S + (blk + 8) * SK_ROWS + 8 * nt + 2 * t) = make_int2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  }
  __syncwarp();
  const u64 q = b0 + lane;
  if (q < n) {
    const uint4 *V = (const uint4 *)(S + lane * SK_ROWS);
    u64 w[16], s[8];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = index(q);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = V[i];
      w[1 + 2 * i] = be_word(v.x, v.y), w[2 + 2 * i] = be_word(v.z, v.w);
    }
    w[9] = 0x80ULL << 56, w[15] = (u64)(128 + 8 + 4 * SK_ROWS) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = kSketchMid[i];
    sha::compress(s, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) ys[8 * q + i] = s[i];
  }
  __syncwarp();
}

// The receipts of the warp's 32 blocks b0 … b0 + 31 (of n, block 0 at data, position j0 + b) into ys, S the warp's
// staging (32 · SK_ROWS int32) and sg the packed signs in shared memory.
__device__ __forceinline__ void sk16_tile(const uint4 *__restrict__ data, u64 j0, u64 n, u64 b0, const u32 *sg, int *S,
                                          u64 *__restrict__ ys, int lane) {
  const int g = lane >> 2, t = lane & 3;
  sk16_tile_of([&](uint4 (&d)[2][2], int p) { sk64_load<false, 2>(d, data, nullptr, b0, n, p, g, t); },
               [&](u64 q) { return j0 + q; }, n, b0, sg, S, ys, lane);
}

// sk16_tile over positions b0 … b0 + 31 of n, position q's block read at addrs[q] and hashed at store index js[q]
// (`map_kernel`): lane q loads position b0 + q's address, and lane (g, t) takes its blocks' from lanes 16 mt + 8 hi + g.
__device__ __forceinline__ void sk16_mapped_tile(const u64 *__restrict__ addrs, const u64 *__restrict__ js, u64 n,
                                                 u64 b0, const u32 *sg, int *S, u64 *__restrict__ ys, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const u64 q = b0 + lane, mine = q < n ? __ldg(addrs + q) : 0;
  const uint4 *at[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hi = 0; hi < 2; ++hi)
      at[mt][hi] = reinterpret_cast<const uint4 *>(__shfl_sync(~0u, mine, 16 * mt + 8 * hi + g));
  sk16_tile_of([&](uint4 (&d)[2][2], int p) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hi = 0; hi < 2; ++hi)
        d[mt][hi] = at[mt][hi] ? ld_l2_256(at[mt][hi] + 4 * p + t) : make_uint4(0, 0, 0, 0);
  }, [&](u64 x) { return __ldg(js + x); }, n, b0, sg, S, ys, lane);
}

__device__ __forceinline__ void sk16_prologue(u32 *sg, const u32 *__restrict__ pk) {
  for (int i = threadIdx.x; i < SK16_PACK; i += blockDim.x) sg[i] = __ldg(pk + i);
  __syncthreads();
}

// A pass's signs from its key on the device, packed as pack16_kernel packs them: row c of S is the first 512 bytes of
// the key's counter-mode stream at index ("row", c) (`verity.ml.randomness`'s Key.stream: block k is
// SHA512(head ‖ u64be k), head = stream tag ‖ key ‖ frames(("row", c)), whose last byte is c for c < 256), bit u of
// the row (byte u / 8, LSB first) 1 where S[c][u] = −1. head is row 0's (at most 103 bytes, so a block is one
// compression). A thread a (row, block) of the stream, then the 2048 words from shared memory.
struct StreamHead {
  uint8_t b[104];
  int n;
};

__global__ void __launch_bounds__(128) signs16_kernel(const StreamHead h, u32 *__restrict__ pk) {
  __shared__ uint8_t bits[SK_ROWS][BLOCK / 8];
  {
    const int c = threadIdx.x >> 3, k = threadIdx.x & 7;
    uint8_t m[256];
    for (int i = 0; i < 256; ++i) m[i] = i < h.n ? h.b[i] : 0;
    m[h.n - 1] = (uint8_t)c;
    m[h.n + 7] = (uint8_t)k;
    u64 st[8];
    sha512_bytes(m, h.n + 8, st, &bits[c][64 * k]);
  }
  __syncthreads();
  for (int q = threadIdx.x; q < SK16_PACK; q += blockDim.x) {
    const int p = q >> 5, g = (q & 31) >> 2, t = q & 3;
    u32 w = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int nt = r >> 2, s = (r >> 1) & 1, b = r & 1, u = 64 * p + 16 * t + 8 * s + 4 * b;
#pragma unroll
      for (int i = 0; i < 4; ++i) w |= (u32)((bits[8 * nt + g][(u + i) >> 3] >> ((u + i) & 7)) & 1) << (8 * i + r);
    }
    pk[q] = w;
  }
}

// A chained pass derives everything whose input is the previous pass's root on the device.  The host sets the
// window's challenge once (`rs_chain_set`); pass i then computes
//
//   seed_i = SHA512(SEED_DOMAIN || challenge || u32be(i) || root_(i-1)),
//
// the sketch-sign key and row streams, and the sketch receipt's first-block midstate.  `rb_chain` (reader.cu)
// completes the receipt tree's constants from that seed.  Thus no seed, sign key, or tree constant crosses the host
// between passes.  The 64-byte seed is retained with the pass because the opener's second library uses it to restore
// that pass's constants.
constexpr int CHAIN_THREADS = 128;

struct ChainPass {
  uint8_t seed[160], derive[256], receipt[128], row[128];
  int seed_len, seed_i, seed_root, derive_len, derive_seed, receipt_seed;
  int row_len, row_key, row_c, row_ctr;
};

struct ChainPassState {
  u64 receipt_mid[8];
};

__constant__ ChainPass kChainPass;
__device__ ChainPassState gChainPass;

__global__ void __launch_bounds__(CHAIN_THREADS) chain_prepare_kernel(unsigned pass, const u64 *__restrict__ previous,
                                                                      uint8_t *__restrict__ seed_out,
                                                                      u32 *__restrict__ pk) {
  __shared__ uint8_t seed[64], key[32], bits[SK_ROWS][BLOCK / 8];
  const int tid = threadIdx.x;
  uint8_t m[256];
  u64 h[8];
  if (tid == 0) {
    for (int b = 0; b < 256; ++b) m[b] = 0;
    for (int b = 0; b < kChainPass.seed_len; ++b) m[b] = kChainPass.seed[b];
    for (int b = 0; b < 4; ++b) m[kChainPass.seed_i + b] = (uint8_t)(pass >> (24 - 8 * b));
    for (int b = 0; b < 64; ++b)
      m[kChainPass.seed_root + b] = (uint8_t)(previous[b >> 3] >> (56 - 8 * (b & 7)));
    sha512_bytes(m, kChainPass.seed_len, h, seed);
    for (int b = 0; b < 64; ++b) seed_out[b] = seed[b];
  }
  __syncthreads();
  if (tid == 0) {
    for (int b = 0; b < 256; ++b) m[b] = 0;
    for (int b = 0; b < kChainPass.derive_len; ++b) m[b] = kChainPass.derive[b];
    for (int b = 0; b < 64; ++b) m[kChainPass.derive_seed + b] = seed[b];
    sha512_bytes(m, kChainPass.derive_len, h, nullptr);
    for (int b = 0; b < 32; ++b) key[b] = (uint8_t)(h[b >> 3] >> (56 - 8 * (b & 7)));
    for (int b = 0; b < 256; ++b) m[b] = 0;
    for (int b = 0; b < 128; ++b) m[b] = kChainPass.receipt[b];
    for (int b = 0; b < 64; ++b) m[kChainPass.receipt_seed + b] = seed[b];
    sha512_bytes(m, -1, h, nullptr);
    for (int k = 0; k < 8; ++k) gChainPass.receipt_mid[k] = h[k];
  }
  __syncthreads();
  {
    const int c = tid >> 3, block = tid & 7;
    for (int b = 0; b < 256; ++b) m[b] = 0;
    for (int b = 0; b < kChainPass.row_len; ++b) m[b] = kChainPass.row[b];
    for (int b = 0; b < 32; ++b) m[kChainPass.row_key + b] = key[b];
    m[kChainPass.row_c] = (uint8_t)c;
    for (int b = 0; b < 8; ++b) m[kChainPass.row_ctr + b] = (uint8_t)((u64)block >> (56 - 8 * b));
    uint8_t *out = &bits[c][64 * block];
    sha512_bytes(m, kChainPass.row_len, h, out);
  }
  __syncthreads();
  for (int q = tid; q < SK16_PACK; q += CHAIN_THREADS) {
    const int p = q >> 5, g = (q & 31) >> 2, t = q & 3;
    u32 w = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int nt = r >> 2, s = (r >> 1) & 1, b = r & 1, u = 64 * p + 16 * t + 8 * s + 4 * b;
#pragma unroll
      for (int x = 0; x < 4; ++x)
        w |= (u32)((bits[8 * nt + g][(u + x) >> 3] >> ((u + x) & 7)) & 1) << (8 * x + r);
    }
    pk[q] = w;
  }
}

__global__ void __launch_bounds__(32 * SK16_WARPS, 2) sketch16_receipts_kernel(const uint4 *__restrict__ data, u64 j0,
                                                                               u64 n, const u32 *__restrict__ pk,
                                                                               u64 *__restrict__ ys) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tiles = (n + 31) / 32;
  for (u64 tile = blockIdx.x * (u64)SK16_WARPS + warp; tile < tiles; tile += (u64)gridDim.x * SK16_WARPS)
    sk16_tile(data, j0, n, tile * 32, sg, st[warp], ys, lane);
}

// A store where its tensors hold it (`device.InPlace`): region r is store blocks first … first + blocks − 1, block b
// at addr + 4096 b, or, paged (plane_blocks > 0), in plane q = b / plane_blocks of its pages (page q / planes, plane
// q mod planes) at addr + page · page_stride + plane · plane_stride, 4096 (b mod plane_blocks) bytes in: vLLM's
// PagedBacking.where. Regions are in store order.
struct MapRegion {
  u64 first, blocks, addr, plane_blocks, planes, page_stride, plane_stride;
};
static_assert(sizeof(MapRegion) == 56, "a region is 7 words");

// Position p's device address and its store index j, plan[p]: the plan's runs as (first, position) pairs in position
// order, the first at position 0.
__global__ void __launch_bounds__(128) map_kernel(const u64 *__restrict__ runs, u64 nruns,
                                                  const MapRegion *__restrict__ regions, int nregions, u64 n,
                                                  u64 *__restrict__ addrs, u64 *__restrict__ js) {
  const u64 p = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (p >= n) return;
  u64 lo = 0, hi = nruns;
  while (hi - lo > 1) {
    const u64 m = (lo + hi) / 2;
    if (__ldg(runs + 2 * m + 1) <= p) lo = m;
    else hi = m;
  }
  const u64 j = __ldg(runs + 2 * lo) + p - __ldg(runs + 2 * lo + 1);
  int a = 0, b = nregions;
  while (b - a > 1) {
    const int m = (a + b) / 2;
    if (regions[m].first <= j) a = m;
    else b = m;
  }
  const MapRegion r = regions[a];
  const u64 x = j - r.first, q = r.plane_blocks ? x / r.plane_blocks : 0;
  addrs[p] = r.plane_blocks ? r.addr + q / r.planes * r.page_stride + q % r.planes * r.plane_stride +
                                  x % r.plane_blocks * BLOCK
                            : r.addr + x * BLOCK;
  js[p] = j;
}

__global__ void __launch_bounds__(32 * SK16_WARPS, 2) mapped16_receipts_kernel(const u64 *__restrict__ addrs,
                                                                               const u64 *__restrict__ js, u64 n,
                                                                               const u32 *__restrict__ pk,
                                                                               u64 *__restrict__ ys) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tiles = (n + 31) / 32;
  for (u64 tile = blockIdx.x * (u64)SK16_WARPS + warp; tile < tiles; tile += (u64)gridDim.x * SK16_WARPS)
    sk16_mapped_tile(addrs, js, n, tile * 32, sg, st[warp], ys, lane);
}

// reader.cu's read_kernel through the map: every byte of positions 0 … n − 1 once, where addrs puts it.
__global__ void __launch_bounds__(256) mapped_read_kernel(const u64 *__restrict__ addrs, u64 n,
                                                          unsigned long long *xr) {
  constexpr u64 PER = BLOCK / 16;
  u32 acc = 0;
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n * PER; i += (u64)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(reinterpret_cast<const uint4 *>(__ldg(addrs + i / PER)) + i % PER);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) atomicXor(xr, 1ull);
}

// Tile `tile` of a shard's pass (its 32 blocks' receipts into ys, S the warp's staging, base the shard's first store
// position), then counted in ready[its consumer group] (`tiles_per` tiles a group) once its receipts are visible to
// every CTA.
__device__ __forceinline__ void pass_tile(const uint4 *__restrict__ data, u64 n, u64 base, const u32 *sg, int *S,
                                          u64 *ys, unsigned *ready, unsigned tile, unsigned tiles_per, int lane) {
  sk16_tile(data, base, n, (u64)tile * 32, sg, S, ys, lane);
  __threadfence();
  __syncwarp();
  if (!lane) atomicAdd(ready + tile / tiles_per, 1u);
}

// The whole pass's receipts by a persistent grid, a warp taking the next tile of 32 blocks from ctr[0] and, once its
// receipts are stored, counting it in ctr[1 + its consumer group], so a retained subtree can start while later
// receipts are read, with no kernel boundary on the reads between groups. Each CTA first sets *started: from then on
// its warps take tiles until none is left, so the consumer (`pass_consumer_kernel`) can wait on them.
__global__ void __launch_bounds__(32 * SK16_WARPS, 2) sketch16_pass_kernel(const uint4 *__restrict__ data, u64 n,
                                                                           u64 base, const u32 *__restrict__ pk,
                                                                           u64 *__restrict__ ys, unsigned *ctr,
                                                                           unsigned tiles_per, unsigned *started) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  if (!threadIdx.x) atomicExch(started, 1u);
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tiles = (n + 31) / 32;
  for (;;) {
    unsigned tile = 0;
    if (!lane) tile = atomicAdd(ctr, 1u);
    tile = __shfl_sync(~0u, tile, 0);
    if (tile >= tiles) break;
    pass_tile(data, n, base, sg, st[warp], ys, ctr + 1, tile, tiles_per, lane);
  }
}

// One thread waiting until *flag reaches want, so what follows on its stream reads what the counted tiles wrote.
__global__ void wait_kernel(const unsigned *flag, unsigned want) {
  while (*(volatile const unsigned *)flag < want) __nanosleep(256);
  __threadfence();
}

// One full receipt-tree leaf from its 16 receipts. All pass leaves are full because the harness's store length is a
// multiple of PER_LEAF. This is reader.cu's leaves_kernel specialized to that case, followed by the same hm96 leaf.
// Y is read through the non-coherent cache: the consumer reads a group's receipts only once the group is counted, and
// a tile's receipts are whole cache lines, so no line of them is cached before it is written.
__device__ __forceinline__ void pass_leaf(const u64 *__restrict__ Y, u64 c, u64 (&d)[8]) {
  u64 x[8];
  vc::hash_words(kC.row_mid, 128, Y, 8 * PER_LEAF, x, kC.kw_xpad);
  hm96_leaf(c, x, d);
}

__device__ __forceinline__ u64 pass_kept_at(u64 width, int h0, int h) {
  u64 at = 0;
  for (int g = h0; g < h; ++g) at += width >> g;
  return at;
}

// A consumer warp's tile (`pass_consumer_kernel`): the next one from ctr[0], if any is left, by pass_tile. It is out
// of line because inlined, its registers crowd the consumer's leaf loop, which then recomputes an address for every
// receipt it loads.
__device__ __noinline__ void help_tile(const uint4 *__restrict__ data, u64 n, u64 base, const u32 *pk, int *S, u64 *ys,
                                       unsigned *ctr, unsigned tiles_per, int lane) {
  unsigned tile = 0;
  if (!lane) tile = atomicAdd(ctr, 1u);
  tile = __shfl_sync(~0u, tile, 0);
  if (tile < n / 32) pass_tile(data, n, base, pk, S, ys, ctr + 1, tile, tiles_per, lane);
}

// A shard's receipt tree behind its receipts, by persistent grids of CONSUMER_THREADS-thread CTAs sharing ctr
// (rs_pass's layout): the next tile, each group's counted tiles, each chunk's completed subtrees, the next subtree to
// claim and the receipt grid's started flag. A CTA claims the next 2^h0-leaf subtree in order, waits until the group
// holding it is counted, hashes its leaves and its levels below h0 in shared memory and keeps its root. The CTA
// completing a chunk's last subtree builds that chunk's retained levels up to hc, the chunk's root. Leaf and node
// indices are the store's (the shard's first leaf is base_leaf); kept is the shard's own, level h at the sum of
// nl >> g over h0 <= g < h, nl the shard's leaves.
//
// Progress needs no residency and no scheduling order. While its group is short, a warp waits only if every tile is
// taken (by warps already running, which finish them without waiting) or a receipt CTA has started (whose warps take
// tiles until none is left); otherwise it takes the next tile and computes it itself (`help_tile`, the signs read from
// pk). So no wait depends on a CTA that isn't resident, whatever the stream priorities, the order the grids are placed
// in or what a co-tenant holds, and a pass whose receipt grid never gets an SM still completes.
constexpr int CONSUMER_THREADS = 128;
constexpr size_t CONSUMER_STAGING = (size_t)CONSUMER_THREADS / 32 * 32 * SK_ROWS * sizeof(int);

__global__ void __launch_bounds__(CONSUMER_THREADS) pass_consumer_kernel(const uint4 *__restrict__ data, u64 n,
                                                                         u64 base_leaf, const u32 *__restrict__ pk,
                                                                         u64 *ys, unsigned *ctr, u64 *__restrict__ kept,
                                                                         u64 chunk, u64 ready_blocks, int h0, int hc,
                                                                         u64 chunks) {
  extern __shared__ __align__(16) u64 scratch[];
  __shared__ unsigned claimed;
  __shared__ int chunk_last;
  const u64 lc = chunk / PER_LEAF, leaves = 1ULL << h0, groups = lc >> h0, subtrees = n / (PER_LEAF << h0);
  const u64 tiles = n / 32, nl = n / PER_LEAF, base = base_leaf * PER_LEAF;
  const unsigned tiles_per = (unsigned)(ready_blocks / 32);
  unsigned *ready = ctr + 1, *subtree_done = ready + n / ready_blocks;
  unsigned *claim = subtree_done + chunks;
  const unsigned *started = claim + 1;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  bool live = false;   // a receipt CTA has started (lane 0's; it stays set for the pass)
  for (;;) {
    if (!threadIdx.x) claimed = atomicAdd(claim, 1u);
    __syncthreads();
    const u64 q = claimed;
    if (q >= subtrees) return;
    const u64 k = q / groups, first = (q - k * groups) * leaves, c0 = k * lc;
    const unsigned *group = ready + (c0 + first) * PER_LEAF / ready_blocks;
    for (;;) {
      int act = 0;   // 0: the group is counted, 1: take a tile, 2: wait
      if (!lane && *(volatile const unsigned *)group < tiles_per) {
        live = live || *(volatile const unsigned *)started;
        act = !live && *(volatile const unsigned *)ctr < tiles ? 1 : 2;
      }
      act = __shfl_sync(~0u, act, 0);
      if (!act) break;
      if (act == 2) {
        __nanosleep(256);
        continue;
      }
      help_tile(data, n, base, pk, reinterpret_cast<int *>(scratch) + warp * 32 * SK_ROWS, ys, ctr, tiles_per, lane);
    }
    __threadfence();
    __syncthreads();   // the staging the warps computed tiles in is the leaves' buffer
    u64 *in = scratch, *out = scratch + 8 * leaves;
    for (u64 j = threadIdx.x; j < leaves; j += blockDim.x) {
      u64 d[8];
      pass_leaf(ys + 8 * PER_LEAF * (c0 + first + j), base_leaf + c0 + first + j, d);
#pragma unroll
      for (int x = 0; x < 8; ++x) in[8 * j + x] = d[x];
    }
    __syncthreads();
    for (int h = 0; h < h0; ++h) {
      const u64 m = leaves >> (h + 1), i0 = (base_leaf + c0 + first) >> (h + 1);
      for (u64 j = threadIdx.x; j < m; j += blockDim.x) {
        u64 d[8];
        node_of(h, i0 + j, in + 16 * j, in + 16 * j + 8, d);
#pragma unroll
        for (int x = 0; x < 8; ++x) out[8 * j + x] = d[x];
      }
      __syncthreads();
      u64 *tmp = in;
      in = out;
      out = tmp;
    }
    if (!threadIdx.x) {
      const u64 at = pass_kept_at(nl, h0, h0) + ((c0 + first) >> h0);
#pragma unroll
      for (int x = 0; x < 8; ++x) kept[8 * at + x] = in[x];
    }
    __threadfence();
    if (!threadIdx.x) chunk_last = atomicAdd(subtree_done + k, 1u) + 1 == groups;
    __syncthreads();
    if (chunk_last) {
      __threadfence();
      for (int h = h0; h < hc; ++h) {
        const u64 m = lc >> (h + 1), i0 = (base_leaf + c0) >> (h + 1);
        const u64 *level_in = kept + 8 * (pass_kept_at(nl, h0, h) + (c0 >> h));
        u64 *level_out = kept + 8 * (pass_kept_at(nl, h0, h + 1) + (c0 >> (h + 1)));
        for (u64 j = threadIdx.x; j < m; j += blockDim.x) {
          u64 d[8];
          node_of(h, i0 + j, level_in + 16 * j, level_in + 16 * j + 8, d);
#pragma unroll
          for (int x = 0; x < 8; ++x) level_out[8 * j + x] = d[x];
        }
        __syncthreads();
      }
    }
    __syncthreads();   // before the next claim rewrites `claimed` and the staging
  }
}

// The in-place pass (`rs_mapped_pass`) counts its tiles in groups of CONSUMER_READY_BLOCKS positions, the last group
// what is left; mapped_pass_tile is pass_tile through the map.
constexpr unsigned MAPPED_READY_TILES = CONSUMER_READY_BLOCKS / 32;

__device__ __forceinline__ void mapped_pass_tile(const u64 *__restrict__ addrs, const u64 *__restrict__ js, u64 n,
                                                 const u32 *sg, int *S, u64 *ys, unsigned *ready, unsigned tile,
                                                 int lane) {
  sk16_mapped_tile(addrs, js, n, (u64)tile * 32, sg, S, ys, lane);
  __threadfence();
  __syncwarp();
  if (!lane) atomicAdd(ready + tile / MAPPED_READY_TILES, 1u);
}

// sketch16_pass_kernel through the map.
__global__ void __launch_bounds__(32 * SK16_WARPS, 2) mapped16_pass_kernel(const u64 *__restrict__ addrs,
                                                                           const u64 *__restrict__ js, u64 n,
                                                                           const u32 *__restrict__ pk,
                                                                           u64 *__restrict__ ys, unsigned *ctr,
                                                                           unsigned *started) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  if (!threadIdx.x) atomicExch(started, 1u);
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tiles = (n + 31) / 32;
  for (;;) {
    unsigned tile = 0;
    if (!lane) tile = atomicAdd(ctr, 1u);
    tile = __shfl_sync(~0u, tile, 0);
    if (tile >= tiles) break;
    mapped_pass_tile(addrs, js, n, sg, st[warp], ys, ctr + 1, tile, lane);
  }
}

// help_tile through the map, out of line for the same reason.
__device__ __noinline__ void mapped_help_tile(const u64 *__restrict__ addrs, const u64 *__restrict__ js, u64 n,
                                              const u32 *pk, int *S, u64 *ys, unsigned *ctr, int lane) {
  unsigned tile = 0;
  if (!lane) tile = atomicAdd(ctr, 1u);
  tile = __shfl_sync(~0u, tile, 0);
  if (tile < (n + 31) / 32) mapped_pass_tile(addrs, js, n, pk, S, ys, ctr + 1, tile, lane);
}

// Leaf c of a pass of n positions from its receipts at Y (leaves_kernel's digest, the last leaf short), or past the
// last leaf the padding leaf c (pads_kernel's). Y is read through the non-coherent cache, as pass_leaf reads it.
__device__ __forceinline__ void mapped_leaf(const u64 *__restrict__ Y, u64 c, u64 n, u64 (&d)[8]) {
  const u64 leaves = (n + PER_LEAF - 1) / PER_LEAF;
  if (c >= leaves) {
    rc::pad_rt(kL, c, d);
    return;
  }
  const int m = c + 1 < leaves ? PER_LEAF : (int)(n - PER_LEAF * c);
  u64 x[8];
  if (m == PER_LEAF) vc::hash_words(kC.row_mid, 128, Y, 8 * PER_LEAF, x, kC.kw_xpad);
  else vc::hash_words(kC.row_mid_last, 128, Y, 8 * m, x, nullptr);
  hm96_leaf(c, x, d);
}

// The in-place pass's receipt tree behind its receipts (`rs_mapped_pass`), pass_consumer_kernel's protocol over a
// pass of n positions, its ⌈n / 16⌉ leaves padded to 2^H. ctr holds the next tile, each group's counted tiles, the next
// subtree to claim, the receipt grid's started flag and, for each node above level h0, its children done. A CTA claims
// the next 2^h0-leaf subtree, the padding's first (they wait on nothing), waits as pass_consumer_kernel does until the
// group holding it is counted, hashes its leaves (`mapped_leaf`) and its levels below h0 in shared memory and keeps
// its root; then one thread climbs, hashing each node whose other child is done, up to the root. kept is level h at
// the sum of 2^(H − g) over h0 <= g < h, the root last.
__global__ void __launch_bounds__(CONSUMER_THREADS) mapped_consumer_kernel(const u64 *__restrict__ addrs,
                                                                           const u64 *__restrict__ js, u64 n,
                                                                           const u32 *__restrict__ pk, u64 *ys,
                                                                           unsigned *ctr, u64 *kept, int h0, int H) {
  extern __shared__ __align__(16) u64 scratch[];
  __shared__ unsigned claimed;
  const u64 leaves = 1ULL << h0, nl = (n + PER_LEAF - 1) / PER_LEAF, width = 1ULL << H;
  const u64 subtrees = width >> h0, real = (nl + leaves - 1) >> h0, tiles = (n + 31) / 32;
  unsigned *ready = ctr + 1, *claim = ready + (n + CONSUMER_READY_BLOCKS - 1) / CONSUMER_READY_BLOCKS;
  const unsigned *started = claim + 1;
  unsigned *arrived = claim + 2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  bool live = false;   // a receipt CTA has started (lane 0's; it stays set for the pass)
  for (;;) {
    if (!threadIdx.x) claimed = atomicAdd(claim, 1u);
    __syncthreads();
    const u64 k = claimed;
    if (k >= subtrees) return;
    const u64 q = k < subtrees - real ? real + k : k - (subtrees - real), first = q << h0;
    if (q < real) {
      const u64 g = first * PER_LEAF / CONSUMER_READY_BLOCKS, left = tiles - g * MAPPED_READY_TILES;
      const unsigned want = left < MAPPED_READY_TILES ? (unsigned)left : MAPPED_READY_TILES;
      for (;;) {
        int act = 0;   // 0: the group is counted, 1: take a tile, 2: wait
        if (!lane && *(volatile const unsigned *)(ready + g) < want) {
          live = live || *(volatile const unsigned *)started;
          act = !live && *(volatile const unsigned *)ctr < tiles ? 1 : 2;
        }
        act = __shfl_sync(~0u, act, 0);
        if (!act) break;
        if (act == 2) {
          __nanosleep(256);
          continue;
        }
        mapped_help_tile(addrs, js, n, pk, reinterpret_cast<int *>(scratch) + warp * 32 * SK_ROWS, ys, ctr, lane);
      }
    }
    __threadfence();
    __syncthreads();   // the staging the warps computed tiles in is the leaves' buffer
    u64 *in = scratch, *out = scratch + 8 * leaves;
    for (u64 j = threadIdx.x; j < leaves; j += blockDim.x) {
      u64 d[8];
      mapped_leaf(ys + 8 * PER_LEAF * (first + j), first + j, n, d);
#pragma unroll
      for (int x = 0; x < 8; ++x) in[8 * j + x] = d[x];
    }
    __syncthreads();
    for (int h = 0; h < h0; ++h) {
      const u64 m = leaves >> (h + 1), i0 = first >> (h + 1);
      for (u64 j = threadIdx.x; j < m; j += blockDim.x) {
        u64 d[8];
        node_of(h, i0 + j, in + 16 * j, in + 16 * j + 8, d);
#pragma unroll
        for (int x = 0; x < 8; ++x) out[8 * j + x] = d[x];
      }
      __syncthreads();
      u64 *tmp = in;
      in = out;
      out = tmp;
    }
    if (!threadIdx.x) {
      u64 at = q, d[8];
#pragma unroll
      for (int x = 0; x < 8; ++x) d[x] = in[x];
      for (int h = h0;; ++h) {
        u64 *node = kept + 8 * (pass_kept_at(width, h0, h) + at);
#pragma unroll
        for (int x = 0; x < 8; ++x) node[x] = d[x];
        if (h == H) break;
        __threadfence();
        if (!atomicAdd(arrived + pass_kept_at(width, h0 + 1, h + 1) + (at >> 1), 1u)) break;
        __threadfence();
        const u64 *sib = kept + 8 * (pass_kept_at(width, h0, h) + (at ^ 1));
        u64 L[8], R[8];
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          const u64 o = __ldcg(sib + x);
          L[x] = at & 1 ? o : d[x];
          R[x] = at & 1 ? d[x] : o;
        }
        at >>= 1;
        node_of(h, at, L, R, d);
      }
    }
    __syncthreads();   // before the next claim rewrites `claimed` and the staging
  }
}

// The receipt tree's levels hc … H on one CTA: level hc is the segments' nodes in order (each shard's chunk roots,
// then each GPU's share of the padding), read over P2P and copied into top, then each level above it from the one
// below; top holds level h at the sum of 2^(H − g) over hc <= g < h, the root last, which goes to each of roots.
// Peer reads bypass the caches, since a segment's buffer is new each pass but its address may not be.
constexpr int TOP_SEGS = 32, TOP_ROOTS = 16, TOP_THREADS = 1024;

struct TopSegs {
  const u64 *ptr[TOP_SEGS];
  u64 count[TOP_SEGS];
  u64 *roots[TOP_ROOTS];
  int n, nroots;
};

__global__ void __launch_bounds__(TOP_THREADS) top_kernel(const TopSegs s, u64 *__restrict__ top, int hc, int H) {
  const u64 width = 1ULL << (H - hc);
  for (u64 j = threadIdx.x; j < width; j += blockDim.x) {
    u64 at = j;
    int k = 0;
    while (k + 1 < s.n && at >= s.count[k]) at -= s.count[k], ++k;
    const u64 *src = s.ptr[k] + 8 * at;
#pragma unroll
    for (int x = 0; x < 8; ++x) top[8 * j + x] = __ldcv(src + x);
  }
  __syncthreads();
  u64 in_at = 0, m = width;
  for (int h = hc; h < H; ++h) {
    for (u64 j = threadIdx.x; j < m / 2; j += blockDim.x) {
      u64 d[8];
      node_of(h, j, top + 8 * (in_at + 2 * j), top + 8 * (in_at + 2 * j + 1), d);
#pragma unroll
      for (int x = 0; x < 8; ++x) top[8 * (in_at + m + j) + x] = d[x];
    }
    __syncthreads();
    in_at += m;
    m /= 2;
  }
  if (threadIdx.x < 8 * s.nroots) s.roots[threadIdx.x >> 3][threadIdx.x & 7] = top[8 * in_at + (threadIdx.x & 7)];
  __threadfence_system();
}

// The subtrees' kernels behind rs_subtrees: subtree q's first leaf c[q], its 16 · 2^h0 receipts at ys + q · 16 · 2^h0
// · 8, its nodes at nodes + q · (2^(h0+1) − 2) · 8 (level g at the sum of 2^(h0 − f) over f < g).
constexpr int PER_LEAF_LOG2 = 4;
static_assert(PER_LEAF == 1 << PER_LEAF_LOG2, "a leaf is 16 receipts");

__global__ void __launch_bounds__(32 * SK16_WARPS, 2) subtree_receipts_kernel(const uint4 *__restrict__ data, u64 base,
                                                                              const u64 *__restrict__ c, int h0,
                                                                              const u32 *__restrict__ pk,
                                                                              u64 *__restrict__ ys, u64 tiles) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tile = blockIdx.x * (u64)SK16_WARPS + warp;
  if (tile >= tiles) return;
  const int per_log2 = h0 + PER_LEAF_LOG2 - 5;   // tiles a subtree
  const u64 q = tile >> per_log2, b0 = (tile & ((1ULL << per_log2) - 1)) * 32, j0 = c[q] * PER_LEAF;
  sk16_tile(data + (j0 - base) * (BLOCK / 16), j0, (u64)PER_LEAF << h0, b0, sg, st[warp],
            ys + (q << (h0 + PER_LEAF_LOG2)) * 8, lane);
}

__global__ void __launch_bounds__(128) subtree_leaves_kernel(const u64 *__restrict__ ys, const u64 *__restrict__ c,
                                                             int h0, u64 n, u64 *__restrict__ nodes) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  const u64 q = t >> h0, l = t & ((1ULL << h0) - 1), per = (2ULL << h0) - 2;
  u64 d[8];
  pass_leaf(ys + t * PER_LEAF * 8, c[q] + l, d);
  if (tid < n)
#pragma unroll
    for (int i = 0; i < 8; ++i) nodes[(q * per + l) * 8 + i] = d[i];
}

__global__ void __launch_bounds__(128) subtree_level_kernel(u64 *__restrict__ nodes, const u64 *__restrict__ c, int h0,
                                                            int h, u64 n) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (tid >= n) return;
  const int wide = h0 - h - 1;   // log2 of the subtree's nodes at level h + 1
  const u64 q = tid >> wide, i = tid & ((1ULL << wide) - 1), per = (2ULL << h0) - 2;
  u64 at = 0;
  for (int f = 0; f < h; ++f) at += 1ULL << (h0 - f);
  u64 *nd = nodes + q * per * 8;
  const u64 *in = nd + 8 * (at + 2 * i);
  u64 d[8];
  node_of(h, (c[q] >> (h + 1)) + i, in, in + 8, d);
#pragma unroll
  for (int k = 0; k < 8; ++k) nd[8 * (at + (1ULL << (h0 - h)) + i) + k] = d[k];
}

// Row q of out is the 64 bytes at device address addrs[q]: a window's openings (receipts, tree nodes of every pass and
// of the payload tree) gathered in one launch, so the opener waits on the passes' kernels once a window.
__global__ void __launch_bounds__(128) gather64_kernel(const u64 *__restrict__ addrs, u64 n, uint4 *__restrict__ out) {
  const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (t >= 4 * n) return;
  const uint4 *src = reinterpret_cast<const uint4 *>(addrs[t >> 2]);
  out[t] = src[t & 3];
}

}  // namespace rb

extern "C" {

int rs_payload_leaves(const void *data, uint64_t c0, uint64_t n, uint64_t *dig, cudaStream_t s) {
  if (!data || !dig || n < 1) return 1;
  payload_leaves_kernel<<<grid(n, 128), 128, 0, s>>>((const uint4 *)data, c0, n, dig);
  return cudaGetLastError() ? 4 : 0;
}

int rs_gather64(const uint64_t *addrs, uint64_t n, uint64_t *out, cudaStream_t s) {
  if (!addrs || !out || n < 1) return 1;
  gather64_kernel<<<grid(4 * n, 128), 128, 0, s>>>(addrs, n, (uint4 *)out);
  return cudaGetLastError() ? 4 : 0;
}

// A receipts grid over n blocks: ctas CTAs of kernel k, or with ctas 0 as many as are resident at once, and no more
// than the blocks fill.
static int receipts_grid(const void *k, uint64_t n, int ctas, unsigned *g) {
  if (!ctas) {
    int dev, sms, per;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 32 * SK16_WARPS, 0))
      return 4;
    ctas = sms * per;
  }
  const uint64_t most = (n + 32 * SK16_WARPS - 1) / (32 * SK16_WARPS);
  *g = (unsigned)(most < (uint64_t)ctas ? most : (uint64_t)ctas);
  return 0;
}

// Receipts of blocks j0 … j0 + n − 1 (data, block j0's first) into ys under the signs packed in pk (`rs_pack16`), on
// ctas CTAs (0: as many as are resident at once).
static int sketch16_receipts(const void *data, uint64_t j0, uint64_t n, const uint32_t *pk, uint64_t *ys, int ctas,
                             cudaStream_t s) {
  if (!data || !pk || !ys || n < 1 || ctas < 0) return 1;
  unsigned g;
  if (receipts_grid((const void *)sketch16_receipts_kernel, n, ctas, &g)) return 4;
  sketch16_receipts_kernel<<<g, 32 * SK16_WARPS, 0, s>>>((const uint4 *)data, j0, n, pk, ys);
  return cudaGetLastError() ? 4 : 0;
}

// The 16 rows of signs (int8 ±1, row-major) as sketch16_receipts_kernel's B registers: SK16_PACK words into pk.
int rs_pack16(const void *signs, uint32_t *pk, cudaStream_t s) {
  if (!signs || !pk) return 1;
  pack16_kernel<<<grid(SK16_PACK, 128), 128, 0, s>>>((const int8_t *)signs, pk);
  return cudaGetLastError() ? 4 : 0;
}

// The packed signs of a pass from its stream head (`signs16_kernel`): SK16_PACK words into pk.
int rs_signs16(const uint8_t *head, int head_len, uint32_t *pk, cudaStream_t s) {
  StreamHead h{};
  if (!head || !pk || head_len < 1 || head_len > (int)sizeof h.b || head_len + 8 + 17 > 128) return 1;
  memcpy(h.b, head, head_len);
  h.n = head_len;
  signs16_kernel<<<1, 128, 0, s>>>(h, pk);
  return cudaGetLastError() ? 4 : 0;
}

// The fixed messages for `chain_prepare_kernel`, at one window's challenge.  The framing is
// `verity.ml.randomness` v2, exactly as `benchmarks/codec_tensor/wt.cu::wt_set_pass` constructs it for
// WT_RECEIPT=1.
int rs_chain_set(const uint8_t *challenge, cudaStream_t s) {
  if (!challenge) return 1;
  ChainPass P{};
  auto put = [](uint8_t *m, int &o, const void *src, int len) {
    memcpy(m + o, src, len);
    o += len;
  };
  auto frame = [&](uint8_t *m, int &o, char tag, const void *body, int len) {
    m[o++] = (uint8_t)tag;
    for (int b = 7; b >= 0; --b) m[o++] = (uint8_t)((u64)len >> (8 * b));
    put(m, o, body, len);
  };
  auto frame_int = [&](uint8_t *m, int &o, u64 v) {
    uint8_t be[8];
    int len = 0;
    for (int b = 7; b >= 0; --b)
      if (len || (v >> (8 * b)) & 0xFF || b == 0) be[len++] = (uint8_t)(v >> (8 * b));
    frame(m, o, 'i', be, len);
  };
  const uint8_t zero[64] = {};
  int o = 0;
  put(P.seed, o, "verity/remote-bandwidth/seed/v0", 32);
  put(P.seed, o, challenge, 32);
  P.seed_i = o;
  put(P.seed, o, zero, 4);
  P.seed_root = o;
  put(P.seed, o, zero, 64);
  P.seed_len = o;

  o = 0;
  put(P.derive, o, "verity/randomness/derive/v2", 28);
  P.derive_seed = o + 9;
  frame(P.derive, o, 'b', zero, 64);
  const char *sign_domain = "verity/remote-bandwidth/sketch-signs/v0";
  frame(P.derive, o, 's', sign_domain, (int)strlen(sign_domain));
  uint8_t ctx[128];
  int c = 0;
  frame(ctx, c, 's', "k", 1);
  frame_int(ctx, c, SK_ROWS);
  frame(ctx, c, 's', "n", 1);
  frame_int(ctx, c, BLOCK);
  frame(ctx, c, 's', "tag", 3);
  const char *name = "sketch-sha512";
  frame(ctx, c, 's', name, (int)strlen(name));
  frame(P.derive, o, 'b', ctx, c);
  P.derive_len = o;

  o = 0;
  put(P.receipt, o, "verity/remote-bandwidth/sketch-receipt/v0", 42);
  P.receipt_seed = o + 9;
  frame(P.receipt, o, 'b', zero, 64);

  o = 0;
  put(P.row, o, "verity/randomness/stream/v2", 28);
  P.row_key = o;
  put(P.row, o, zero, 32);
  uint8_t idx[32];
  c = 0;
  frame(idx, c, 's', "row", 3);
  frame_int(idx, c, 0);
  frame(P.row, o, 'b', idx, c);
  P.row_c = o - 1;
  P.row_ctr = o;
  put(P.row, o, zero, 8);
  P.row_len = o;
  return cudaMemcpyToSymbolAsync(kChainPass, &P, sizeof P, 0, cudaMemcpyHostToDevice, s) ? 4 : 0;
}

// Prepare pass `i` from the previous root, put its seed and packed signs in device memory, and install both the sketch
// receipt and receipt-tree constants on this stream.
int rs_chain_prepare(unsigned i, const uint64_t *previous, uint8_t *seed, uint32_t *pk, cudaStream_t s) {
  if (!previous || !seed || !pk) return 1;
  chain_prepare_kernel<<<1, CHAIN_THREADS, 0, s>>>(i, previous, seed, pk);
  void *state = nullptr;
  if (cudaGetLastError() || cudaGetSymbolAddress(&state, gChainPass)) return 4;
  uint8_t *p = static_cast<uint8_t *>(state);
  if (cudaMemcpyToSymbolAsync(kSketchMid, p + offsetof(ChainPassState, receipt_mid), 64, 0,
                              cudaMemcpyDeviceToDevice, s))
    return 4;
  return rb_chain(seed, s);
}

int rs_sketch16_receipts(const void *data, uint64_t j0, uint64_t n, const uint32_t *pk, uint64_t *ys, int ctas,
                         cudaStream_t s) {
  return sketch16_receipts(data, j0, n, pk, ys, ctas, s);
}

// A plan of n positions mapped to the store where its tensors hold it (`map_kernel`): each position's device address
// into addrs and its store index into js. runs is the plan's nruns (first, position) pairs in position order, the
// first at position 0, and regions nregions MapRegions (7 words each) in store order, covering every block the runs
// name; all on the device.
int rs_map(const uint64_t *runs, uint64_t nruns, const uint64_t *regions, int nregions, uint64_t n, uint64_t *addrs,
           uint64_t *js, cudaStream_t s) {
  if (!runs || !regions || !addrs || !js || nruns < 1 || nregions < 1 || n < 1) return 1;
  map_kernel<<<grid(n, 128), 128, 0, s>>>(runs, nruns, reinterpret_cast<const MapRegion *>(regions), nregions, n,
                                          addrs, js);
  return cudaGetLastError() ? 4 : 0;
}

// rs_sketch16_receipts over n positions, each read where rs_map's addrs puts it and hashed at its js.
int rs_mapped_receipts(const uint64_t *addrs, const uint64_t *js, uint64_t n, const uint32_t *pk, uint64_t *ys,
                       int ctas, cudaStream_t s) {
  if (!addrs || !js || !pk || !ys || n < 1 || ctas < 0) return 1;
  unsigned g;
  if (receipts_grid((const void *)mapped16_receipts_kernel, n, ctas, &g)) return 4;
  mapped16_receipts_kernel<<<g, 32 * SK16_WARPS, 0, s>>>(addrs, js, n, pk, ys);
  return cudaGetLastError() ? 4 : 0;
}

// rb_read through rs_map's addrs: n positions' bytes, on blocks CTAs.
int rs_mapped_read(const uint64_t *addrs, uint64_t n, unsigned long long *xr, int blocks, cudaStream_t s) {
  if (!addrs || !xr || n < 1 || blocks < 1) return 1;
  mapped_read_kernel<<<blocks, 256, 0, s>>>(addrs, n, xr);
  return cudaGetLastError() ? 4 : 0;
}

static int log2_exact(uint64_t n) {
  if (!n || n & (n - 1)) return -1;
  int h = 0;
  while (n >> h > 1) ++h;
  return h;
}

// Node offsets of the kept levels h0 … H of a tree of nl leaves, level h0 first: level h starts at the sum of
// nl >> g over h0 <= g < h.
static uint64_t kept_at(uint64_t nl, int h0, int h) {
  uint64_t at = 0;
  for (int g = h0; g < h; ++g) at += nl >> g;
  return at;
}

// The consumer CTAs of a pass beside a receipt grid of ctas CTAs, by both kernels' occupancy (an SM holds occ_p receipt
// CTAs of `producer` or occ_c CTAs of `consumer`): `side` as many as fit in what the receipt grid leaves of the device,
// at least one an SM, and `tail` the rest of a full device, which take the slots the receipt grid frees when it ends.
// Neither count bounds progress (`pass_consumer_kernel`); together they are what the device holds once the reads are
// done.
static int pass_consumers(const void *producer, const void *consumer, int ctas, size_t shared, uint64_t subtrees,
                          uint64_t *side, uint64_t *tail) {
  int dev, sms, occ_p, occ_c;
  if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_p, producer, 32 * SK16_WARPS, 0) ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, consumer, CONSUMER_THREADS, shared) ||
      occ_p < 1 || occ_c < 1)
    return 4;
  const int64_t all = (int64_t)sms * occ_c, fit = all - ((int64_t)ctas * occ_c + occ_p - 1) / occ_p;
  const int64_t s = fit > sms ? fit : sms;
  *side = (uint64_t)s < subtrees ? (uint64_t)s : subtrees;
  const uint64_t rest = all > s ? (uint64_t)(all - s) : 0, left = subtrees - *side;
  *tail = rest < left ? rest : left;
  return 0;
}

// One shard's part of a pass: the n blocks at data, the store's blocks base … base + n − 1 (both whole chunks). Its
// receipts by one persistent grid of ctas CTAs on sa (`sketch16_pass_kernel`, tensor cores; 0: one an SM), and for
// h0 <= 8 persistent consumer grids (`pass_consumer_kernel`, `pass_consumers`), one on sb beside the receipt grid and
// one behind it on sa, that build each 2^h0-leaf subtree as soon as the receipt grid counts its group's tiles done and
// each chunk's levels up to its root at hc once its subtrees are, so the shard's tree follows the reads without a leaf
// or level kernel boundary, and the tree hashing left when the reads end has the whole device. pk is the pass's packed
// signs (`rs_chain_prepare`); ctr is scratch of 3 + n / min(chunk, CONSUMER_READY_BLOCKS) + n / chunk words: the next
// tile, each group's counted tiles, each chunk's completed subtrees, the next subtree to claim and whether a receipt CTA
// has started. ys holds the shard's n receipts (8 words each). dig and up are scratch for h0 > 8, whose tree is built
// chunk by chunk behind one waiting thread. Levels h0 … hc of the shard go to kept (level h0 first, `kept_at` over the
// shard's n / 16 leaves); levels below h0 are not kept. The padding leaves pad_r0 … pad_r0 + pad_n − 1 (whole chunks'
// worth, at the store's leaf positions) are hashed on sb from the start, under the tensor-core receipts, up to their
// pad_n / lc nodes at hc in pad_out, `pad` scratch of 2 · pad_n digests. The levels above hc are `rs_top`'s. sa waits
// for the whole shard pass before anything queued on it after. consumers > 0 is the one consumer grid's CTAs on sb, for
// a shard read beside another on the same device (pinned host memory over PCIe, whose few receipt CTAs leave it the
// device but whose consumers would otherwise fill it).
int rs_pass(const void *data, uint64_t n, uint64_t base, const uint32_t *pk, unsigned *ctr, uint64_t *ys, uint64_t *dig,
            uint64_t *up, uint64_t *kept, int h0, uint64_t chunk, int ctas, uint64_t pad_r0, uint64_t pad_n,
            uint64_t *pad, uint64_t *pad_out, cudaStream_t sa, cudaStream_t sb, int consumers) {
  constexpr int DEVICES = 64;
  static cudaEvent_t go[DEVICES], end[DEVICES];
  static bool made[DEVICES];
  int dev;
  if (cudaGetDevice(&dev) || dev < 0 || dev >= DEVICES) return 4;
  if (!made[dev]) {
    if (cudaEventCreateWithFlags(&go[dev], cudaEventDisableTiming) ||
        cudaEventCreateWithFlags(&end[dev], cudaEventDisableTiming))
      return 4;
    made[dev] = true;
  }
  if (!data || !pk || !ctr || !ys || !dig || !up || !kept || !n || n % PER_LEAF || chunk % 32 || chunk % PER_LEAF ||
      !chunk || n % chunk || base % chunk || ctas < 0 || consumers < 0)
    return 1;
  const uint64_t lc = chunk / PER_LEAF, chunks = n / chunk, base_leaf = base / PER_LEAF, nl = n / PER_LEAF;
  const int hc = log2_exact(lc);
  if (hc < 0 || h0 < 1 || h0 > hc || n / 32 >= (1ULL << 32)) return 1;
  if (pad_n && (!pad || !pad_out || pad_n % lc || pad_r0 % lc)) return 1;
  const bool fused = h0 <= 8;
  const uint64_t group_blocks = (uint64_t)PER_LEAF << h0;
  const uint64_t groups = n / group_blocks;
  const uint64_t ready_blocks = chunk < CONSUMER_READY_BLOCKS ? chunk : CONSUMER_READY_BLOCKS;
  const uint64_t ready_groups = n / ready_blocks;
  if (!ctas && cudaDeviceGetAttribute(&ctas, cudaDevAttrMultiProcessorCount, dev)) return 4;
  const size_t tree_shared = 2 * (1ULL << h0) * 8 * sizeof(uint64_t);
  const size_t shared = tree_shared > CONSUMER_STAGING ? tree_shared : CONSUMER_STAGING;
  uint64_t side = 0, tail = 0;
  int rc;
  if (fused) {
    if (consumers)
      side = (uint64_t)consumers < groups ? (uint64_t)consumers : groups;
    else if ((rc = pass_consumers((const void *)sketch16_pass_kernel, (const void *)pass_consumer_kernel, ctas, shared,
                                  groups, &side, &tail)))
      return rc;
    if (groups + side + tail >= (1ULL << 32)) return 1;
  }
  unsigned *started = ctr + 2 + ready_groups + chunks;
  if (cudaMemsetAsync(ctr, 0, 4 * (3 + ready_groups + chunks), sa) || cudaEventRecord(go[dev], sa) ||
      cudaStreamWaitEvent(sb, go[dev], 0))
    return 4;
  const unsigned ready_tiles = (unsigned)((fused ? ready_blocks : chunk) / 32);
  sketch16_pass_kernel<<<ctas, 32 * SK16_WARPS, 0, sa>>>((const uint4 *)data, n, base, pk, ys, ctr, ready_tiles,
                                                         started);
  if (cudaGetLastError()) return 4;
  // These nodes depend on this pass's seed, but not on its receipts: hashed at the start, under the reads.
  if (pad_n) {
    uint64_t *pin = pad, *scratch = pad + 8 * pad_n;
    if ((rc = rb_pads(pad_r0, pad_n, pin, sb))) return rc;
    for (int h = 0; h < hc; ++h) {
      const uint64_t m = pad_n >> (h + 1), i0 = pad_r0 >> (h + 1);
      uint64_t *out = h + 1 == hc ? pad_out : scratch;
      level_kernel<<<grid(m, 128), 128, 0, sb>>>(pin, out, h, i0, m);
      if (out == scratch) scratch += 8 * m;
      pin = out;
    }
  }
  // h0 <= 8 fits the consumer's two 2^h0-digest reduction buffers in CUDA's default dynamic-shared-memory limit. Keep the old exact path
  // for diagnostic configurations with larger retained subtrees.
  if (fused) {
    pass_consumer_kernel<<<(unsigned)side, CONSUMER_THREADS, shared, sb>>>((const uint4 *)data, n, base_leaf, pk, ys,
                                                                           ctr, kept, chunk, ready_blocks, h0, hc,
                                                                           chunks);
    if (tail)
      pass_consumer_kernel<<<(unsigned)tail, CONSUMER_THREADS, shared, sa>>>((const uint4 *)data, n, base_leaf, pk, ys,
                                                                             ctr, kept, chunk, ready_blocks, h0, hc,
                                                                             chunks);
  } else {
    for (uint64_t k = 0; k < chunks; ++k) {
      const int slot = k & 1;
      uint64_t *dig_k = dig + slot * lc * 8, *up_k = up + slot * lc * 8;
      wait_kernel<<<1, 1, 0, sb>>>(ctr + 1 + k, (unsigned)(chunk / 32));
      if ((rc = rb_leaves(ys + k * chunk * 8, base_leaf + k * lc, lc, dig_k, nullptr, sb))) return rc;
      const uint64_t *in = dig_k;
      uint64_t *scratch = up_k;
      for (int h = 0; h < hc; ++h) {
        const uint64_t m = lc >> (h + 1), i0 = (base_leaf + k * lc) >> (h + 1);
        uint64_t *out = h + 1 >= h0 ? kept + 8 * (kept_at(nl, h0, h + 1) + ((k * lc) >> (h + 1))) : scratch;
        level_kernel<<<grid(m, 128), 128, 0, sb>>>(in, out, h, i0, m);
        if (out == scratch) scratch += 8 * m;
        in = out;
      }
    }
  }
  if (cudaEventRecord(end[dev], sb) || cudaStreamWaitEvent(sa, end[dev], 0)) return 4;
  return cudaGetLastError() ? 4 : 0;
}

// The in-place pass (`InPlace.run_pass`) as rs_pass runs a shard's: positions 0 … n − 1, each read where rs_map's
// addrs puts it and hashed at its js, their receipts into ys (n × 8 words) by one persistent grid of ctas CTAs on sa
// (`mapped16_pass_kernel`; 0: one an SM), and the receipt tree behind them by persistent consumer grids
// (`mapped_consumer_kernel`, `pass_consumers`), one on sb beside the receipt grid and one behind it on sa, so the tree
// follows the reads with no kernel boundary. The tree is `receipts.tree`'s under the pass's constants set: ⌈n / 16⌉
// leaves, the last short, padded to 2^H leaves (H the least that holds them). Its levels h0 … H (h0 <= min(H, 8)) go
// to kept, 2^(H − h0 + 1) − 1 digests, level h at the sum of 2^(H − g) over h0 <= g < h, the root last; levels below
// h0 are not kept. ctr is scratch of ctr_words >= 2 + ⌈n / CONSUMER_READY_BLOCKS⌉ + 2^(H − h0) words. sa waits for the
// whole pass before anything queued on it after.
int rs_mapped_pass(const uint64_t *addrs, const uint64_t *js, uint64_t n, const uint32_t *pk, unsigned *ctr,
                   uint64_t ctr_words, uint64_t *ys, uint64_t *kept, int h0, int ctas, cudaStream_t sb,
                   cudaStream_t sa) {
  constexpr int DEVICES = 64;
  static cudaEvent_t go[DEVICES], end[DEVICES];
  static bool made[DEVICES];
  int dev;
  if (cudaGetDevice(&dev) || dev < 0 || dev >= DEVICES) return 4;
  if (!made[dev]) {
    if (cudaEventCreateWithFlags(&go[dev], cudaEventDisableTiming) ||
        cudaEventCreateWithFlags(&end[dev], cudaEventDisableTiming))
      return 4;
    made[dev] = true;
  }
  if (!addrs || !js || !pk || !ctr || !ys || !kept || n < 1 || n >= (1ULL << 36) || h0 < 0 || h0 > 8 || ctas < 0)
    return 1;
  const uint64_t nl = (n + PER_LEAF - 1) / PER_LEAF;
  int H = 0;
  while ((1ULL << H) < nl) ++H;
  if (h0 > H) return 1;
  const uint64_t groups = (n + CONSUMER_READY_BLOCKS - 1) / CONSUMER_READY_BLOCKS, subtrees = 1ULL << (H - h0);
  const uint64_t words = 2 + groups + subtrees;
  if (ctr_words < words) return 1;
  if (!ctas && cudaDeviceGetAttribute(&ctas, cudaDevAttrMultiProcessorCount, dev)) return 4;
  const size_t tree_shared = 2 * (1ULL << h0) * 8 * sizeof(uint64_t);
  const size_t shared = tree_shared > CONSUMER_STAGING ? tree_shared : CONSUMER_STAGING;
  uint64_t side, tail;
  int rc;
  if ((rc = pass_consumers((const void *)mapped16_pass_kernel, (const void *)mapped_consumer_kernel, ctas, shared,
                           subtrees, &side, &tail)))
    return rc;
  unsigned *started = ctr + 2 + groups;
  if (cudaMemsetAsync(ctr, 0, 4 * words, sa) || cudaEventRecord(go[dev], sa) || cudaStreamWaitEvent(sb, go[dev], 0))
    return 4;
  mapped16_pass_kernel<<<ctas, 32 * SK16_WARPS, 0, sa>>>(addrs, js, n, pk, ys, ctr, started);
  mapped_consumer_kernel<<<(unsigned)side, CONSUMER_THREADS, shared, sb>>>(addrs, js, n, pk, ys, ctr, kept, h0, H);
  if (tail)
    mapped_consumer_kernel<<<(unsigned)tail, CONSUMER_THREADS, shared, sa>>>(addrs, js, n, pk, ys, ctr, kept, h0, H);
  if (cudaGetLastError() || cudaEventRecord(end[dev], sb) || cudaStreamWaitEvent(sa, end[dev], 0)) return 4;
  return 0;
}

// The receipt tree's levels hc … H on this GPU (`top_kernel`, one CTA on s): level hc is the nsegs segments in order,
// counts[k] nodes at ptrs[k] (each shard's chunk roots at its kept level hc, then each GPU's padding nodes at hc, on any
// GPU this one can read), 2^(H − hc) in all; top gets level h at the sum of 2^(H − g) over hc <= g < h, the root last,
// which is also written to each of the nroots roots (a slot of each GPU's root ring, or any memory this GPU can write).
int rs_top(const uint64_t *const *ptrs, const uint64_t *counts, int nsegs, uint64_t *top, int hc, int H,
           uint64_t *const *roots, int nroots, cudaStream_t s) {
  if (!ptrs || !counts || !top || nsegs < 1 || nsegs > TOP_SEGS || nroots < 0 || nroots > TOP_ROOTS || (nroots && !roots) ||
      hc < 0 || H < hc || H - hc > 20)
    return 1;
  TopSegs S{};
  uint64_t sum = 0;
  for (int k = 0; k < nsegs; ++k) {
    if (counts[k] && !ptrs[k]) return 1;
    S.ptr[k] = ptrs[k], S.count[k] = counts[k], sum += counts[k];
  }
  for (int r = 0; r < nroots; ++r) {
    if (!roots[r]) return 1;
    S.roots[r] = roots[r];
  }
  if (sum != 1ULL << (H - hc)) return 1;
  S.n = nsegs, S.nroots = nroots;
  top_kernel<<<1, TOP_THREADS, 0, s>>>(S, top, hc, H);
  return cudaGetLastError() ? 4 : 0;
}

// Peer access between every pair of visible GPUs that supports it, for rs_top's reads and writes; 5 if some pair can't.
int rs_enable_peers(void) {
  int count, here, rc = 0;
  if (cudaGetDeviceCount(&count) || cudaGetDevice(&here)) return 4;
  for (int i = 0; i < count; ++i) {
    if (cudaSetDevice(i)) return 4;
    for (int j = 0; j < count; ++j) {
      int ok = 0;
      if (i == j) continue;
      if (cudaDeviceCanAccessPeer(&ok, i, j)) return 4;
      if (!ok) {
        rc = 5;
        continue;
      }
      const cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      else if (e) return 4;
    }
  }
  return cudaSetDevice(here) ? 4 : rc;
}

// What the openings of a pass need below its kept levels, recomputed from the store under the pass constants set: for
// each first leaf c[q] (a multiple of 2^h0, c on the device), the subtree's 16 · 2^h0 receipts into
// ys + q · 16 · 2^h0 · 8 and its levels 0 … h0 − 1 into nodes + q · (2^(h0+1) − 2) · 8, level 0 first, in one launch for
// the receipts, one for the leaves and one a level; pk is the pass's packed signs. Every leaf of the pass must be whole
// (positions a multiple of 16). data is a shard whose first block is the store's block base, which holds every
// subtree in c.
int rs_subtrees(const void *data, uint64_t base, const uint32_t *pk, const uint64_t *c, uint64_t count, int h0,
                uint64_t *ys, uint64_t *nodes, cudaStream_t s) {
  if (!data || !pk || !c || !ys || !nodes || h0 < 1 || h0 > 20 || h0 + 1 > NODE_LEVELS) return 1;
  if (!count) return 0;
  const uint64_t tiles = count << (h0 + PER_LEAF_LOG2 - 5);
  subtree_receipts_kernel<<<grid(tiles, SK16_WARPS) , 32 * SK16_WARPS, 0, s>>>((const uint4 *)data, base, c, h0,
                                                                               pk, ys, tiles);
  subtree_leaves_kernel<<<grid(count << h0, 128), 128, 0, s>>>(ys, c, h0, count << h0, nodes);
  for (int h = 0; h + 1 < h0; ++h)
    subtree_level_kernel<<<grid(count << (h0 - h - 1), 128), 128, 0, s>>>(nodes, c, h0, h, count << (h0 - h - 1));
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
