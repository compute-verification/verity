// The remote session's prover kernels beyond the device reader's (`../codec_tensor/reader.cu`, included whole, so one
// library holds both and shares its pass constants kC and kSketchMid):
//
//   payload leaf j   x = SHA512(P(32768) ‖ payload_j) (sha512/row/v2 of the 4 KiB block), then the reader's hm96 leaf
//                    under salt_j = SHAKE256(D(store salt) ‖ key ‖ label ‖ u64be j) and its frame-v3-sha512 leaf
//                    digest at position j: the driver's `receipts.hidden(payload_j, salt_j)` in `audit.payload_tree`
//   gather           64-byte rows from a list of device addresses (a window's openings, across its passes' kept
//                    receipts and trees)
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

}  // extern "C"
