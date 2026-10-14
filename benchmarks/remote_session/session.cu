// The remote session's prover kernels beyond the device reader's (`../codec_tensor/reader.cu`, included whole, so one
// library holds both and shares its pass constants kC and kSketchMid):
//
//   payload leaf j   x = SHA512(P(32768) ‖ payload_j) (sha512/row/v2 of the 4 KiB block), then the reader's hm96 leaf
//                    under salt_j = SHAKE256(D(store salt) ‖ key ‖ label ‖ u64be j) and its frame-v3-sha512 leaf
//                    digest at position j: the driver's `receipts.hidden(payload_j, salt_j)` in `audit.payload_tree`
//   gather           64-byte rows from a list of device addresses (a window's openings, across its passes' kept
//                    levels, their recomputed subtrees and the payload tree)
//   pass             a pass's receipts by one persistent grid on tensor cores, each chunk's leaves and subtree hashed on
//                    a second stream while the later chunks are read (`rs_pass`), keeping the tree's levels from h0 up
//   subtrees         the levels below h0 and the receipts of the subtrees a window's openings fall in, recomputed from
//                    the store under the pass's constants (`rs_subtrees`)
//
// The pass constants are the library's own (__constant__), so a second copy of the library, loaded from another path,
// recomputes openings under one pass's constants while the first runs the next passes under theirs.
//
// rb_set takes the payload tree's constants as it takes a pass's: row_prefix = sha512_row_v2_prefix(32768), the
// payload domain's id, salt_in = D(store salt) ‖ key ‖ label, positions = N.
#include "../codec_tensor/reader.cu"

namespace rb {

// The hm96 leaf over x (a row's sha512/row/v2 digest) at position c, salted by salt_of(c): reader.cu's leaves_kernel
// after its row hash, for a row of any length.
__device__ __forceinline__ void hm96_leaf(u64 c, const u64 (&x)[8], u64 (&d)[8]) {
  u64 S[24], w[16];
  salt_of(c, S);
  u32 Ys[48];
#pragma unroll
  for (int i = 0; i < 24; ++i) Ys[2 * i] = (u32)S[i], Ys[2 * i + 1] = (u32)(S[i] >> 32);
  u32 M[16];
#pragma unroll
  for (int o = 0; o < 16; ++o) M[o] = 0;
#pragma unroll 1
  for (int b = 0; b < 32; ++b) {
    u32 kb[63];
#pragma unroll
    for (int u = 0; u < 63; ++u) kb[u] = __funnelshift_r(kC.key[u], kC.key[u + 1], b);
#pragma unroll
    for (int o = 0; o < 16; ++o) {
      u32 acc = 0;
#pragma unroll
      for (int v = 0; v < 48; ++v) acc ^= kb[o + v] & Ys[v];
      M[o] |= (u32)(__popc(acc) & 1) << b;
    }
  }
  u64 cd[8], val[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cd[i] = kC.salt_mid[i];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = bswap64(S[i]);
  sha512_compress(cd, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i < 8 ? bswap64(S[16 + i]) : 0;
  w[8] = 0x80ULL << 56, w[15] = (u64)(128 + SALT) * 8;
  sha512_compress(cd, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) val[i] = kC.leaf_mid[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = x[i] ^ bswap64(((u64)M[2 * i + 1] << 32) | M[2 * i]), w[8 + i] = cd[i];
  sha512_compress(val, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = 0x80ULL << 56, w[15] = 256 * 8;
  sha512_compress(val, w);
  Msg<32> msg;
  int done = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = kSha512Iv[i];
  put_head(msg, HEAD_LEAF);
  put_uint(msg, c);
  put_uint(msg, c);
  flush_block(msg, d, done);
  put_schema(msg);
  put_digest(msg, val);
  sha512_finish(msg, d, done);
}

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

// The receipts of the warp's 32 blocks b0 … b0 + 31 (of n, block 0 at data, position j0 + b) into ys, S the warp's
// staging (32 · SK_ROWS int32) and sg the packed signs in shared memory.
__device__ __forceinline__ void sk16_tile(const uint4 *__restrict__ data, u64 j0, u64 n, u64 b0, const u32 *sg, int *S,
                                          u64 *__restrict__ ys, int lane) {
  const int g = lane >> 2, t = lane & 3;
  int acc[2][2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0;
  uint4 cur[2][2], nxt[2][2];
  sk64_load<false, 2>(cur, data, nullptr, b0, n, 0, g, t);
#pragma unroll 1
  for (int p = 0; p < SK16_P; ++p) {
    if (p + 1 < SK16_P) sk64_load<false, 2>(nxt, data, nullptr, b0, n, p + 1, g, t);
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
    w[0] = j0 + q;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = V[i];
      w[1 + 2 * i] = be_word(v.x, v.y), w[2 + 2 * i] = be_word(v.z, v.w);
    }
    w[9] = 0x80ULL << 56, w[15] = (u64)(128 + 8 + 4 * SK_ROWS) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = kSketchMid[i];
    sha512_compress(s, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) ys[8 * q + i] = s[i];
  }
  __syncwarp();
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

// The whole pass's receipts by a persistent grid, a warp taking the next tile of 32 blocks from ctr[0] and, once its
// receipts are stored, counting it in ctr[1 + its consumer group] (`tiles_per` tiles per group), so a retained subtree
// can start while later receipts are read, with no kernel boundary on the reads between groups.
__global__ void __launch_bounds__(32 * SK16_WARPS, 2) sketch16_pass_kernel(const uint4 *__restrict__ data, u64 n,
                                                                           const u32 *__restrict__ pk,
                                                                           u64 *__restrict__ ys, unsigned *ctr,
                                                                           unsigned tiles_per) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  sk16_prologue(sg, pk);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tiles = (n + 31) / 32;
  for (;;) {
    unsigned tile = 0;
    if (!lane) tile = atomicAdd(ctr, 1u);
    tile = __shfl_sync(~0u, tile, 0);
    if (tile >= tiles) break;
    sk16_tile(data, 0, n, (u64)tile * 32, sg, st[warp], ys, lane);
    __threadfence();
    __syncwarp();
    if (!lane) atomicAdd(ctr + 1 + tile / tiles_per, 1u);
  }
}

// One thread waiting until *flag reaches want, so what follows on its stream reads what the counted tiles wrote.
__global__ void wait_kernel(const unsigned *flag, unsigned want) {
  while (*(volatile const unsigned *)flag < want) __nanosleep(256);
  __threadfence();
}

// One full receipt-tree leaf from its 16 receipts. All pass leaves are full because the harness's store length is a
// multiple of PER_LEAF. This is reader.cu's leaves_kernel specialized to that case, followed by the same hm96 leaf.
__device__ __forceinline__ void pass_leaf(const u64 *__restrict__ Y, u64 c, u64 (&d)[8]) {
  u64 x[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = kC.row_mid[i];
#pragma unroll 1
  for (int blk = 0; blk < 9; ++blk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      w[i] = q < 8 * PER_LEAF ? Y[q] : q == 8 * PER_LEAF ? 0x80ULL << 56 : 0;
    }
    if (blk == 8) w[15] = (u64)(128 + 64 * PER_LEAF) * 8;
    sha512_compress(x, w);
  }
  hm96_leaf(c, x, d);
}

__device__ __forceinline__ u64 pass_kept_at(u64 width, int h0, int h) {
  u64 at = 0;
  for (int g = h0; g < h; ++g) at += width >> g;
  return at;
}

// A block owns one 2^h0-leaf subtree of a completed receipt chunk. The last subtree block of each chunk builds that
// chunk's retained levels, and the last completed chunk builds the upper levels through the root. Thus the leaf and
// every level follow the producer in one launch without serializing a chunk's leaves in one CTA.
__global__ void __launch_bounds__(128) pass_consumer_kernel(const u64 *__restrict__ ys, const unsigned *__restrict__ ready,
                                                            unsigned *__restrict__ subtree_done,
                                                            unsigned *__restrict__ chunks_done, u64 *__restrict__ kept,
                                                            u64 width, u64 chunk, int h0, int hc, int H, u64 chunks) {
  const u64 lc = chunk / PER_LEAF, leaves = 1ULL << h0, groups = lc >> h0;
  const u64 q = blockIdx.x, k = q / groups, first = (q - k * groups) * leaves, c0 = k * lc;
  const unsigned tiles_per = (unsigned)(leaves * PER_LEAF / 32);
  while (*(volatile const unsigned *)(ready + q) < tiles_per) __nanosleep(256);
  __threadfence();
  extern __shared__ u64 scratch[];
  u64 *in = scratch, *out = scratch + 8 * leaves;
  for (u64 j = threadIdx.x; j < leaves; j += blockDim.x) {
    u64 d[8];
    pass_leaf(ys + 8 * PER_LEAF * (c0 + first + j), c0 + first + j, d);
#pragma unroll
    for (int x = 0; x < 8; ++x) in[8 * j + x] = d[x];
  }
  __syncthreads();
  for (int h = 0; h < h0; ++h) {
    const u64 m = leaves >> (h + 1), i0 = (c0 + first) >> (h + 1);
    for (u64 j = threadIdx.x; j < m; j += blockDim.x) {
      u64 d[8];
      node_digest(kC.node_mid[h], i0 + j, in + 16 * j, in + 16 * j + 8, d);
#pragma unroll
      for (int x = 0; x < 8; ++x) out[8 * j + x] = d[x];
    }
    __syncthreads();
    u64 *tmp = in;
    in = out;
    out = tmp;
  }
  if (!threadIdx.x) {
    const u64 at = pass_kept_at(width, h0, h0) + ((c0 + first) >> h0);
#pragma unroll
    for (int x = 0; x < 8; ++x) kept[8 * at + x] = in[x];
  }
  __threadfence();
  __shared__ int last;
  if (!threadIdx.x) last = atomicAdd(subtree_done + k, 1u) + 1 == groups;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int h = h0; h < hc; ++h) {
    const u64 m = lc >> (h + 1), i0 = c0 >> (h + 1);
    const u64 *level_in = kept + 8 * (pass_kept_at(width, h0, h) + (c0 >> h));
    u64 *level_out = kept + 8 * (pass_kept_at(width, h0, h + 1) + i0);
    for (u64 j = threadIdx.x; j < m; j += blockDim.x) {
      u64 d[8];
      node_digest(kC.node_mid[h], i0 + j, level_in + 16 * j, level_in + 16 * j + 8, d);
#pragma unroll
      for (int x = 0; x < 8; ++x) level_out[8 * j + x] = d[x];
    }
    __syncthreads();
  }
  __threadfence();
  if (!threadIdx.x) last = atomicAdd(chunks_done, 1u) + 1 == chunks;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int h = hc; h < H; ++h) {
    const u64 m = width >> (h + 1);
    const u64 *level_in = kept + 8 * pass_kept_at(width, h0, h);
    u64 *level_out = kept + 8 * pass_kept_at(width, h0, h + 1);
    for (u64 i = threadIdx.x; i < m; i += blockDim.x) {
      u64 d[8];
      node_digest(kC.node_mid[h], i, level_in + 16 * i, level_in + 16 * i + 8, d);
#pragma unroll
      for (int x = 0; x < 8; ++x) level_out[8 * i + x] = d[x];
    }
    __syncthreads();
  }
}

// The subtrees' kernels behind rs_subtrees: subtree q's first leaf c[q], its 16 · 2^h0 receipts at ys + q · 16 · 2^h0
// · 8, its nodes at nodes + q · (2^(h0+1) − 2) · 8 (level g at the sum of 2^(h0 − f) over f < g).
constexpr int PER_LEAF_LOG2 = 4;
static_assert(PER_LEAF == 1 << PER_LEAF_LOG2, "a leaf is 16 receipts");

__global__ void __launch_bounds__(32 * SK16_WARPS, 2) subtree_receipts_kernel(const uint4 *__restrict__ data,
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
  sk16_tile(data + j0 * (BLOCK / 16), j0, (u64)PER_LEAF << h0, b0, sg, st[warp], ys + (q << (h0 + PER_LEAF_LOG2)) * 8,
            lane);
}

__global__ void __launch_bounds__(128) subtree_leaves_kernel(const u64 *__restrict__ ys, const u64 *__restrict__ c,
                                                             int h0, u64 n, u64 *__restrict__ nodes) {
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, t = tid < n ? tid : n - 1;
  const u64 q = t >> h0, l = t & ((1ULL << h0) - 1), per = (2ULL << h0) - 2;
  const u64 *Y = ys + t * PER_LEAF * 8;
  u64 x[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = kC.row_mid[i];
#pragma unroll 1
  for (int blk = 0; blk < 9; ++blk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int k = 16 * blk + i;
      w[i] = k < 8 * PER_LEAF ? Y[k] : k == 8 * PER_LEAF ? 0x80ULL << 56 : 0;
    }
    if (blk == 8) w[15] = (u64)(128 + 64 * PER_LEAF) * 8;
    sha512_compress(x, w);
  }
  u64 d[8];
  hm96_leaf(c[q] + l, x, d);
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
  node_digest(kC.node_mid[h], (c[q] >> (h + 1)) + i, in, in + 8, d);
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

// Receipts of blocks j0 … j0 + n − 1 (data, block j0's first) into ys under the signs packed in pk (`rs_pack16`), on
// ctas CTAs (0: as many as are resident at once).
static int sketch16_receipts(const void *data, uint64_t j0, uint64_t n, const uint32_t *pk, uint64_t *ys, int ctas,
                             cudaStream_t s) {
  if (!data || !pk || !ys || n < 1 || ctas < 0) return 1;
  if (!ctas) {
    int dev, sms, per;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, sketch16_receipts_kernel, 32 * SK16_WARPS, 0))
      return 4;
    ctas = sms * per;
  }
  const uint64_t most = (n + 32 * SK16_WARPS - 1) / (32 * SK16_WARPS);
  const unsigned g = (unsigned)(most < (uint64_t)ctas ? most : (uint64_t)ctas);
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

// One pass over n blocks at data: its receipts by one persistent grid of ctas CTAs on sa (`sketch16_pass_kernel`, tensor
// cores; 0: one an SM, so a consumer CTA fits beside each), and on sb, one persistent consumer CTA per chunk hashes its
// leaves and tree as soon as the producer counts the chunk's tiles done. A final consumer CTA joins the chunk roots, so
// the whole receipt tree follows the reads without a leaf or level kernel boundary. pk is the pass's packed signs
// (`rs_signs16`); ctr is scratch for the next tile, each 2^h0-leaf group's completed receipt tiles, each chunk's
// completed subtrees and completed chunks; ys
// holds all n receipts (8 words each). dig and up remain fallback scratch for an unusually large h0. `pad` is two
// digests per canonical padding leaf: the
// leaves followed by scratch for their levels below h0. Padding is prepared on sb as soon as the pass starts, under
// the tensor-core receipts, rather than as a tail after the real chunks. Levels h0 … H of the tree go to kept (level
// h0 first, `kept_at`), the root last; levels below h0 are not kept. sa waits for the whole pass before anything
// queued on it after.
int rs_pass(const void *data, uint64_t n, const uint32_t *pk, unsigned *ctr, uint64_t *ys, uint64_t *dig, uint64_t *up,
            uint64_t *pad, uint64_t *kept, int h0, uint64_t chunk, int ctas, cudaStream_t sa, cudaStream_t sb) {
  static cudaEvent_t go, end;
  static bool made = false;
  if (!made) {
    if (cudaEventCreateWithFlags(&go, cudaEventDisableTiming) || cudaEventCreateWithFlags(&end, cudaEventDisableTiming))
      return 4;
    made = true;
  }
  if (!data || !pk || !ctr || !ys || !dig || !up || !pad || !kept || n % PER_LEAF || chunk % 32 ||
      chunk % PER_LEAF || !chunk || n % chunk || ctas < 0)
    return 1;
  const uint64_t nl = n / PER_LEAF, lc = chunk / PER_LEAF;
  if (nl < 2) return 1;
  const uint64_t width = 1ULL << (64 - __builtin_clzll(nl - 1));
  const uint64_t chunks = n / chunk;
  const int H = log2_exact(width), hc = log2_exact(lc);
  if (H < 0 || hc < 0 || h0 < 1 || h0 > hc || n / 32 >= (1ULL << 32)) return 1;
  const bool fused = h0 <= 8;
  const uint64_t group_blocks = (uint64_t)PER_LEAF << h0;
  const uint64_t groups = n / group_blocks;
  if (!ctas) {
    int dev;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&ctas, cudaDevAttrMultiProcessorCount, dev)) return 4;
  }
  int rc;
  if (cudaMemsetAsync(ctr, 0, 4 * (2 + groups + chunks), sa) || cudaEventRecord(go, sa) ||
      cudaStreamWaitEvent(sb, go, 0))
    return 4;
  const unsigned ready_tiles = (unsigned)((fused ? group_blocks : chunk) / 32);
  sketch16_pass_kernel<<<ctas, 32 * SK16_WARPS, 0, sa>>>((const uint4 *)data, n, pk, ys, ctr, ready_tiles);
  if (cudaGetLastError()) return 4;
  // These nodes depend on this pass's seed, but not on its receipts. Compute each once at the start and reuse the
  // retained subtree roots when the real and padded halves are joined above hc.
  const uint64_t pads = width - nl;
  if (pads) {
    uint64_t *pin = pad, *scratch = pad + 8 * pads;
    if ((rc = rb_pads(nl, pads, pin, sb))) return rc;
    for (int h = 0; h < hc; ++h) {
      const uint64_t m = pads >> (h + 1), i0 = nl >> (h + 1);
      uint64_t *out = h + 1 >= h0 ? kept + 8 * (kept_at(width, h0, h + 1) + i0) : scratch;
      level_kernel<<<grid(m, 128), 128, 0, sb>>>(pin, out, h, i0, m);
      if (out == scratch) scratch += 8 * m;
      pin = out;
    }
  }
  // h0 <= 8 fits the consumer's two 2^h0-digest reduction buffers in CUDA's default dynamic-shared-memory limit. Keep the old exact path
  // for diagnostic configurations with larger retained subtrees.
  if (fused) {
    const size_t shared = 2 * (1ULL << h0) * 8 * sizeof(uint64_t);
    if (groups > 0xFFFFFFFFULL) return 1;
    pass_consumer_kernel<<<groups, 128, shared, sb>>>(ys, ctr + 1, ctr + 1 + groups, ctr + 1 + groups + chunks, kept,
                                                     width, chunk, h0, hc, H, chunks);
  } else {
    for (uint64_t k = 0; k < chunks; ++k) {
      const int slot = k & 1;
      uint64_t *dig_k = dig + slot * lc * 8, *up_k = up + slot * lc * 8;
      wait_kernel<<<1, 1, 0, sb>>>(ctr + 1 + k, (unsigned)(chunk / 32));
      if ((rc = rb_leaves(ys + k * chunk * 8, k * lc, lc, dig_k, nullptr, sb))) return rc;
      const uint64_t *in = dig_k;
      uint64_t *scratch = up_k;
      for (int h = 0; h < hc; ++h) {
        const uint64_t m = lc >> (h + 1), i0 = (k * lc) >> (h + 1);
        uint64_t *out = h + 1 >= h0 ? kept + 8 * (kept_at(width, h0, h + 1) + i0) : scratch;
        level_kernel<<<grid(m, 128), 128, 0, sb>>>(in, out, h, i0, m);
        if (out == scratch) scratch += 8 * m;
        in = out;
      }
    }
    const uint64_t *in = kept + 8 * kept_at(width, h0, hc);
    for (int h = hc; h < H; ++h) {
      const uint64_t m = width >> (h + 1);
      uint64_t *out = kept + 8 * kept_at(width, h0, h + 1);
      level_kernel<<<grid(m, 128), 128, 0, sb>>>(in, out, h, 0, m);
      in = out;
    }
  }
  if (cudaEventRecord(end, sb) || cudaStreamWaitEvent(sa, end, 0)) return 4;
  return cudaGetLastError() ? 4 : 0;
}

// What the openings of a pass need below its kept levels, recomputed from the store under the pass constants set: for
// each first leaf c[q] (a multiple of 2^h0, c on the device), the subtree's 16 · 2^h0 receipts into
// ys + q · 16 · 2^h0 · 8 and its levels 0 … h0 − 1 into nodes + q · (2^(h0+1) − 2) · 8, level 0 first, in one launch for
// the receipts, one for the leaves and one a level; pk is the pass's packed signs. Every leaf of the pass must be whole
// (positions a multiple of 16).
int rs_subtrees(const void *data, const uint32_t *pk, const uint64_t *c, uint64_t count, int h0, uint64_t *ys,
                uint64_t *nodes, cudaStream_t s) {
  if (!data || !pk || !c || !ys || !nodes || h0 < 1 || h0 > 20 || h0 + 1 > NODE_LEVELS) return 1;
  if (!count) return 0;
  const uint64_t tiles = count << (h0 + PER_LEAF_LOG2 - 5);
  subtree_receipts_kernel<<<grid(tiles, SK16_WARPS) , 32 * SK16_WARPS, 0, s>>>((const uint4 *)data, c, h0,
                                                                               pk, ys, tiles);
  subtree_leaves_kernel<<<grid(count << h0, 128), 128, 0, s>>>(ys, c, h0, count << h0, nodes);
  for (int h = 0; h + 1 < h0; ++h)
    subtree_level_kernel<<<grid(count << (h0 - h - 1), 128), 128, 0, s>>>(nodes, c, h0, h, count << (h0 - h - 1));
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
