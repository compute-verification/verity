// The remote session's prover kernels beyond the device reader's (`../codec_tensor/reader.cu`, included whole, so one
// library holds both and shares its pass constants kC and kSketchMid):
//
//   payload leaf j   x = SHA512(P(32768) ‖ payload_j) (sha512/row/v2 of the 4 KiB block), then the reader's hm96 leaf
//                    under salt_j = SHAKE256(D(store salt) ‖ key ‖ label ‖ u64be j) and its frame-v3-sha512 leaf
//                    digest at position j: the driver's `receipts.hidden(payload_j, salt_j)` in `audit.payload_tree`
//   gather           64-byte rows from a list of device addresses (a window's openings, across its passes' kept
//                    levels, their recomputed subtrees and the payload tree)
//   pass             a pass in chunks on two streams, each chunk's receipts, leaves and subtree hashed while the next
//                    chunk is read (`rs_pass`), keeping the tree's levels from h0 up
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

__global__ void __launch_bounds__(32 * SK16_WARPS, 2) sketch16_receipts_kernel(const uint4 *__restrict__ data, u64 j0,
                                                                               u64 n, const u32 *__restrict__ pk,
                                                                               u64 *__restrict__ ys) {
  __shared__ u32 sg[SK16_PACK];
  __shared__ __align__(16) int st[SK16_WARPS][32 * SK_ROWS];
  for (int i = threadIdx.x; i < SK16_PACK; i += blockDim.x) sg[i] = __ldg(pk + i);
  __syncthreads();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  int *S = st[warp];
  const u64 tiles = (n + 31) / 32;
  for (u64 tile = blockIdx.x * (u64)SK16_WARPS + warp; tile < tiles; tile += (u64)gridDim.x * SK16_WARPS) {
    const u64 b0 = tile * 32;
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

// One pass over n blocks at data, in chunks of `chunk` blocks: chunk k's receipts on sa (`sketch16_receipts`, tensor
// cores) while chunk k − 1's leaves and subtree run on sb (CUDA cores), so the hashing runs under the next chunk's
// reads. sk is scratch for the packed signs (SK16_PACK words). Two slots of ys (8 words a receipt) and dig (8 words a
// leaf), and up (8 words for each of chunk/16 nodes), alternate between chunks. Levels h0 … H of the tree go to kept (level h0 first, `kept_at`), the root last; levels below h0 are not kept.
// sa waits for the whole pass before anything queued on it after.
int rs_pass(const void *data, uint64_t n, const void *signs, int *sk, uint64_t *ys, uint64_t *dig, uint64_t *up,
            uint64_t *kept, int h0, uint64_t chunk, int ctas, cudaStream_t sa, cudaStream_t sb) {
  static cudaEvent_t a[2], b[2], end;
  static bool made = false;
  if (!made) {
    for (int k = 0; k < 2; ++k)
      if (cudaEventCreateWithFlags(&a[k], cudaEventDisableTiming) || cudaEventCreateWithFlags(&b[k], cudaEventDisableTiming))
        return 4;
    if (cudaEventCreateWithFlags(&end, cudaEventDisableTiming)) return 4;
    made = true;
  }
  if (!data || !signs || !sk || !ys || !dig || !up || !kept || n % PER_LEAF || chunk % PER_LEAF || !chunk || n % chunk)
    return 1;
  const uint64_t nl = n / PER_LEAF, lc = chunk / PER_LEAF;
  const int H = log2_exact(nl), hc = log2_exact(lc);
  if (H < 0 || hc < 0 || h0 < 1 || h0 > hc) return 1;
  const uint64_t chunks = n / chunk;
  const uint32_t *pk = (const uint32_t *)sk;
  int rc = rs_pack16(signs, (uint32_t *)sk, sa);
  if (rc) return rc;
  for (uint64_t k = 0; k < chunks; ++k) {
    const int slot = k & 1;
    uint64_t *ys_k = ys + slot * chunk * 8, *dig_k = dig + slot * lc * 8, *up_k = up + slot * lc * 8;
    if (k >= 2 && cudaStreamWaitEvent(sa, b[slot], 0)) return 4;
    if ((rc = sketch16_receipts((const uint8_t *)data + k * chunk * BLOCK, k * chunk, chunk, pk, ys_k, ctas, sa)))
      return rc;
    if (cudaEventRecord(a[slot], sa) || cudaStreamWaitEvent(sb, a[slot], 0)) return 4;
    if ((rc = rb_leaves(ys_k, k * lc, lc, dig_k, nullptr, sb))) return rc;
    const uint64_t *in = dig_k;
    uint64_t *scratch = up_k;
    for (int h = 0; h < hc; ++h) {
      const uint64_t m = lc >> (h + 1), i0 = (k * lc) >> (h + 1);
      uint64_t *out = h + 1 >= h0 ? kept + 8 * (kept_at(nl, h0, h + 1) + i0) : scratch;
      level_kernel<<<grid(m, 128), 128, 0, sb>>>(in, out, h, i0, m);
      if (out == scratch) scratch += 8 * m;
      in = out;
    }
    if (cudaEventRecord(b[slot], sb)) return 4;
  }
  const uint64_t *in = kept + 8 * kept_at(nl, h0, hc);
  for (int h = hc; h < H; ++h) {
    const uint64_t m = nl >> (h + 1);
    uint64_t *out = kept + 8 * kept_at(nl, h0, h + 1);
    level_kernel<<<grid(m, 128), 128, 0, sb>>>(in, out, h, 0, m);
    in = out;
  }
  if (cudaEventRecord(end, sb) || cudaStreamWaitEvent(sa, end, 0)) return 4;
  return cudaGetLastError() ? 4 : 0;
}

// What the openings of a pass need below its kept levels, recomputed from the store under the pass constants set: for
// each first leaf c[q] (a multiple of 2^h0), the subtree's 16 · 2^h0 receipts into ys + q · 16 · 2^h0 · 8 and its
// levels 0 … h0 − 1 into nodes + q · (2^(h0+1) − 2) · 8, level 0 first. sk is scratch for the packed signs (SK16_PACK words).
int rs_subtrees(const void *data, const void *signs, const uint64_t *c, uint64_t count, int h0, int *sk, uint64_t *ys,
                uint64_t *nodes, cudaStream_t s) {
  if (!data || !signs || !c || !sk || !ys || !nodes || h0 < 1 || h0 > 20) return 1;
  const uint64_t leaves = 1ULL << h0, blocks = PER_LEAF * leaves, per = 2 * leaves - 2;
  int rc = rs_pack16(signs, (uint32_t *)sk, s);
  if (rc) return rc;
  for (uint64_t q = 0; q < count; ++q) {
    if (c[q] % leaves) return 1;
    const uint64_t j0 = c[q] * PER_LEAF;
    uint64_t *ys_q = ys + q * blocks * 8, *nd = nodes + q * per * 8;
    if ((rc = sketch16_receipts((const uint8_t *)data + j0 * BLOCK, j0, blocks, (const uint32_t *)sk, ys_q, 0, s)) ||
        (rc = rb_leaves(ys_q, c[q], leaves, nd, nullptr, s)))
      return rc;
    const uint64_t *in = nd;
    uint64_t *out = nd + 8 * leaves;
    for (int h = 0; h + 1 < h0; ++h) {
      const uint64_t m = leaves >> (h + 1);
      level_kernel<<<grid(m, 128), 128, 0, s>>>(in, out, h, c[q] >> (h + 1), m);
      in = out;
      out += 8 * m;
    }
  }
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
