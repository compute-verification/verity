// The rows-blake3 research arm's kernels (verity_vllm.commit.step_rows_b3): native_rows.cu's leaf, pad and node with keyed BLAKE3
// (archive/pouw-blake3/pearl_c/hash.cuh's compression) in every hash position, bit for bit step_rows_b3 (the hash lane's
// hypothetical hm96-blake3).  Digests are 32 bytes in byte order (little-endian words), each stored in a 64-byte slot (the digest, then
// zeros), so native_collect.cpp's windows, trees and roots are rows-sha512's.  The launchers take native_rows.cu's arguments.
//
// Per position pos of a domain of n leaves (H = BLAKE3 keyed by the arm's public key, one chunk for every message but the row's):
//   x    = H(Q(l) || row)                                         row = l bytes at rows + (pos - first) * stride, Q one 64-byte block
//   y    = salt (128 bytes): given, or y_r = H(Ks || u64be(pos) || u64be(r)), r < 4
//   c    = H(salt_prefix || y);  b = x ^ M_key y;  h = H(leaf_prefix || b || c)
//   leaf = H(frame || "leaf" || domain id || uint(pos) || uint(pos) || "hm96-sha512/row/v2" || h)   (framed parts)
//   pad  = H(frame || "pad" || domain id || uint(pos)) for n <= pos < width;  node(level, index, left, right) up the tree
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using u64 = uint64_t;
using u32 = uint32_t;
#define DEV __device__ __forceinline__
#define HD __host__ __device__ __forceinline__

constexpr int MAX_NB = 4;
constexpr int LEVELS = 33;
constexpr int THREADS = 256;
constexpr int SALT = 128;                    // 2 * 32 + 64
constexpr int KB = 160;                      // the hm96 key's bytes, a T row's
constexpr int SCHEMA_LEN = 18;               // "hm96-sha512/row/v2": the templates' offsets assume its length
constexpr int MAX_STACK = 11;                // chaining values a row of up to 1 MiB keeps (log2 of its 1 KiB chunks, plus one)
constexpr u32 B3_START = 1, B3_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16;

HD u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
HD void b3g(u32* v, int a, int b, int c, int d, u32 x, u32 y) {
  v[a] = v[a] + v[b] + x; v[d] = rotr(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + y; v[d] = rotr(v[d] ^ v[a], 8); v[c] = v[c] + v[d]; v[b] = rotr(v[b] ^ v[c], 7);
}
// one BLAKE3 compression: the first 8 output words; out may alias cv
HD void b3c(const u32* cv, const u32* msg, u64 counter, u32 len, u32 flags, u32* out) {
  constexpr u32 IV[4] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A};
  constexpr int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  u32 v[16], m[16], t[16];
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = cv[i]; v[i + 8] = i < 4 ? IV[i] : 0; }
  v[12] = (u32)counter; v[13] = (u32)(counter >> 32); v[14] = len; v[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    b3g(v, 0, 4, 8, 12, m[0], m[1]); b3g(v, 1, 5, 9, 13, m[2], m[3]); b3g(v, 2, 6, 10, 14, m[4], m[5]); b3g(v, 3, 7, 11, 15, m[6], m[7]);
    b3g(v, 0, 5, 10, 15, m[8], m[9]); b3g(v, 1, 6, 11, 12, m[10], m[11]); b3g(v, 2, 7, 8, 13, m[12], m[13]); b3g(v, 3, 4, 9, 14, m[14], m[15]);
#pragma unroll
    for (int i = 0; i < 16; i++) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

struct B3K {
  u32 key[8];                        // the hash key: every message's chaining value before its first block
  u32 mid_q[8];                      // after Q(l)
  u32 mid_salt[8];                   // after the salt prefix
  u32 mid_leaf[8];                   // after leaf_prefix
  u32 ks[8];                         // the salt key
  u32 mid_fl[8], mid_pad[8], mid_node[8];   // after a frame leaf's, pad's, node's first block (the domain id ends at byte 81; positions after)
  u32 fl_tpl[MAX_NB][32];            // a frame leaf's bytes 64 .. 191 per nb = len(uint(pos)), pos and h zero
  u32 pad_tpl[MAX_NB][16];           // a pad's bytes 64 .. 127 per nb, rank zero
  u32 node_tpl[MAX_NB][32];          // a node's bytes 64 .. 191 per ni = len(uint(index)), level, index and children zero
  u32 lb;                            // row bytes
};

DEV int nbytes(u64 x) { int n = 1; while (x >> (8 * n) && n < 8) ++n; return n; }
DEV uint4 ldg16(const uint8_t* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
DEV void cpy8(u32* d, const u32* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = s[i];
}
DEV void ld8(const uint8_t* p, u32* d) {
  const uint4 a = ldg16(p), b = ldg16(p + 16);
  d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w; d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}
DEV void ld8_written(const uint8_t* p, u32* d) {
  const uint4 a = reinterpret_cast<const uint4*>(p)[0], b = reinterpret_cast<const uint4*>(p)[1];
  d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w; d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}
// n / 8 digests' words at p, each followed by 32 zero bytes when slot
DEV void st_words(uint8_t* p, const u32* d, int n) {
  uint4* o = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int i = 0; i < n / 4; ++i) o[i] = make_uint4(d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]);
}
DEV void st_slot(uint8_t* p, const u32* d) {
  uint4* o = reinterpret_cast<uint4*>(p);
  o[0] = make_uint4(d[0], d[1], d[2], d[3]);
  o[1] = make_uint4(d[4], d[5], d[6], d[7]);
  o[2] = o[3] = make_uint4(0, 0, 0, 0);
}

// byte OFF of the little-endian words m |= v; the low N bytes of v big-endian at OFF; a digest (8 words) at OFF
template <int OFF>
DEV void or_byte(u32* m, u32 v) { m[OFF / 4] |= v << (8 * (OFF % 4)); }
template <int OFF, int N>
DEV void or_be(u32* m, u64 v) {
  if constexpr (N > 0) {
    or_byte<OFF + N - 1>(m, (u32)v & 0xffu);
    or_be<OFF, N - 1>(m, v >> 8);
  }
}
template <int OFF>
DEV void or_dig(u32* m, const u32* d) {
  constexpr int K0 = OFF / 4, S = 8 * (OFF % 4);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (S == 0) {
      m[K0 + i] |= d[i];
    } else {
      m[K0 + i] |= d[i] << S;
      m[K0 + i + 1] |= d[i] >> (32 - S);
    }
  }
}
DEV void parent(const u32* key, const u32* L, const u32* R, u32 flags, u32* out) {
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { m[i] = L[i]; m[8 + i] = R[i]; }
  b3c(key, m, 0, 64, B3_KEYED | B3_PARENT | flags, out);
}

// --- the frame: leaf, pad, node (offsets in the message less its constant first block) --------------------------------

template <int NB>
DEV void frame_leaf_nb(const B3K& K, u64 pos, const u32* h, u32* out) {
  u32 m[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) m[i] = K.fl_tpl[NB - 1][i];
  or_be<89 - 64, NB>(m, pos);
  or_be<97 + NB - 64, NB>(m, pos);
  or_dig<131 + 2 * NB - 64>(m, h);
  cpy8(out, K.mid_fl);
  b3c(out, m, 0, 64, B3_KEYED, out);
  b3c(out, m + 16, 0, 163 + 2 * NB - 128, B3_KEYED | B3_END | B3_ROOT, out);
}
DEV void frame_leaf(const B3K& K, u64 pos, const u32* h, u32* out) {
  switch (nbytes(pos)) {
    case 1: frame_leaf_nb<1>(K, pos, h, out); break;
    case 2: frame_leaf_nb<2>(K, pos, h, out); break;
    case 3: frame_leaf_nb<3>(K, pos, h, out); break;
    default: frame_leaf_nb<4>(K, pos, h, out); break;
  }
}

template <int NB>
DEV void pad_nb(const B3K& K, u64 rank, u32* out) {
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = K.pad_tpl[NB - 1][i];
  or_be<88 - 64, NB>(m, rank);
  cpy8(out, K.mid_pad);
  b3c(out, m, 0, 88 + NB - 64, B3_KEYED | B3_END | B3_ROOT, out);
}
DEV void pad(const B3K& K, u64 rank, u32* out) {
  switch (nbytes(rank)) {
    case 1: pad_nb<1>(K, rank, out); break;
    case 2: pad_nb<2>(K, rank, out); break;
    case 3: pad_nb<3>(K, rank, out); break;
    default: pad_nb<4>(K, rank, out); break;
  }
}

template <int NI>
DEV void node_ni(const B3K& K, int level, u64 index, const u32* L, const u32* R, u32* out) {
  u32 m[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) m[i] = K.node_tpl[NI - 1][i];
  or_byte<89 - 64>(m, (u32)level);
  or_be<98 - 64, NI>(m, index);
  or_dig<106 + NI - 64>(m, L);
  or_dig<146 + NI - 64>(m, R);
  cpy8(out, K.mid_node);
  b3c(out, m, 0, 64, B3_KEYED, out);
  b3c(out, m + 16, 0, 178 + NI - 128, B3_KEYED | B3_END | B3_ROOT, out);
}
DEV void node(const B3K& K, int level, u64 index, const u32* L, const u32* R, u32* out) {
  switch (nbytes(index)) {
    case 1: node_ni<1>(K, level, index, L, R, out); break;
    case 2: node_ni<2>(K, level, index, L, R, out); break;
    case 3: node_ni<3>(K, level, index, L, R, out); break;
    default: node_ni<4>(K, level, index, L, R, out); break;
  }
}

// --- the leaf's parts ----------------------------------------------------------------------------------------------

// x = H(Q || row): BLAKE3's chunks of 16 blocks (chunk 0 is Q and the row's first 960 bytes), their chaining values merged as BLAKE3's
// tree merges them.  p 16-byte aligned; the loads may run up to 15 bytes past the row, inside its stride.
DEV void row_digest(const B3K& K, const uint8_t* p, u32 lb, u32* x) {
  const u32 total = 64 + lb, nblk = (total + 63) / 64;
  u32 cv[8], stack[MAX_STACK][8];
  int sp = 0;
  cpy8(cv, K.mid_q);
  for (u32 k = 1; k < nblk; ++k) {
    const u32 base = 64 * (k - 1);
    u32 m[16];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const u32 off = base + 16 * q;
      const uint4 v = off < lb ? ldg16(p + off) : make_uint4(0, 0, 0, 0);
      m[4 * q] = v.x; m[4 * q + 1] = v.y; m[4 * q + 2] = v.z; m[4 * q + 3] = v.w;
    }
    if (base + 64 > lb) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const u32 at = base + 4 * i;
        if (at >= lb) m[i] = 0;
        else if (at + 4 > lb) m[i] &= (1u << (8 * (lb - at))) - 1u;
      }
    }
    const u32 ch = k >> 4, bi = k & 15;
    const bool last = k == nblk - 1, end = bi == 15 || last;
    if (bi == 0) cpy8(cv, K.key);
    const u32 f = B3_KEYED | (bi == 0 ? B3_START : 0) | (end ? B3_END : 0) | (last && ch == 0 ? B3_ROOT : 0);
    b3c(cv, m, ch, last ? total - 64 * k : 64, f, cv);
    if (end && !last) {
      for (u32 t = ch + 1; !(t & 1); t >>= 1) parent(K.key, stack[--sp], cv, 0, cv);
      cpy8(stack[sp++], cv);
    }
  }
  while (sp > 0) {
    --sp;
    parent(K.key, stack[sp], cv, sp == 0 ? B3_ROOT : 0, cv);
  }
  cpy8(x, cv);
}

DEV void salt_prf(const B3K& K, u64 pos, u32* Y) {
#pragma unroll
  for (int r = 0; r < SALT / 32; ++r) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = K.ks[i];
    m[8] = __byte_perm((u32)(pos >> 32), 0, 0x0123); m[9] = __byte_perm((u32)pos, 0, 0x0123);
    m[10] = 0; m[11] = (u32)r << 24;
#pragma unroll
    for (int i = 12; i < 16; ++i) m[i] = 0;
    b3c(K.key, m, 0, 48, B3_KEYED | B3_START | B3_END | B3_ROOT, Y + 8 * r);
  }
}

// c = H(salt_prefix || y)
DEV void salt_digest(const B3K& K, const u32* Y, u32* c) {
  cpy8(c, K.mid_salt);
  b3c(c, Y, 0, 64, B3_KEYED, c);
  b3c(c, Y + 16, 0, 64, B3_KEYED | B3_END | B3_ROOT, c);
}

// M_key y as 8 little-endian u32: for salt bytes p = 16 q + r, bytes [16 q, 16 q + 48) of T_{y_p} xored into A_r, and A_r shifted down
// r bytes into m (native_rows.cu's hm96_mask at a 32-byte digest).  T: 256 rows of 160 bytes (step_rows_b3.t_table).
DEV void hm96_mask(const uint8_t* T, const u32* Y, u32* mk) {
#pragma unroll
  for (int w = 0; w < 8; ++w) mk[w] = 0;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    u32 A[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) A[i] = 0;
#pragma unroll
    for (int q = 0; q < SALT / 16; ++q) {
      const int p = 16 * q + r;
      const u32 v = (Y[p >> 2] >> (8 * (p & 3))) & 0xFFu;
      const uint8_t* t = T + KB * v + 16 * q;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const uint4 u = ldg16(t + 16 * i);
        A[4 * i] ^= u.x; A[4 * i + 1] ^= u.y; A[4 * i + 2] ^= u.z; A[4 * i + 3] ^= u.w;
      }
    }
    const int sw = r >> 2, s = 8 * (r & 3);
#pragma unroll
    for (int w = 0; w < 8; ++w) mk[w] ^= s ? __funnelshift_r(A[w + sw], A[w + sw + 1], s) : A[w + sw];
  }
}

// native_rows.cu's rows_leaf_kernel: b || c (64 bytes) into a 128-byte slot of bc, h and the leaf into 64-byte slots; salts 128 B a row.
__global__ void __launch_bounds__(THREADS)
    rows_b3_leaf_kernel(const __grid_constant__ B3K K, const uint8_t* __restrict__ rows, u64 stride, u64 n_rows,
                        const uint8_t* __restrict__ salts, const uint8_t* __restrict__ T, u64 first, u64 m, u64 n_total,
                        uint8_t* __restrict__ out, uint8_t* __restrict__ bc, uint8_t* __restrict__ hs) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  u32 o[8];
  if (i < n_rows && pos < n_total) {
    u32 x[8], Y[32], c[8], B[16], h[8], mk[8];
    row_digest(K, rows + i * stride, K.lb, x);
    if (salts) {
#pragma unroll
      for (int j = 0; j < 4; ++j) ld8(salts + SALT * i + 32 * j, Y + 8 * j);
    } else {
      salt_prf(K, pos, Y);
    }
    hm96_mask(T, Y, mk);
    salt_digest(K, Y, c);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      B[w] = x[w] ^ mk[w];
      B[8 + w] = c[w];
    }
    if (bc) {
      st_words(bc + 128 * i, B, 16);
      const u32 z[16] = {};
      st_words(bc + 128 * i + 64, z, 16);
    }
    cpy8(h, K.mid_leaf);
    b3c(h, B, 0, 64, B3_KEYED | B3_END | B3_ROOT, h);
    if (hs) st_slot(hs + 64 * i, h);
    frame_leaf(K, pos, h, o);
  } else {
    pad(K, pos, o);
  }
  st_slot(out + 64 * i, o);
}

__global__ void __launch_bounds__(THREADS) rows_b3_level_kernel(const __grid_constant__ B3K K, const uint8_t* __restrict__ in,
                                                                uint8_t* __restrict__ out, u64 n_out, int level, u64 first_index) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  u32 L[8], R[8], o[8];
  ld8(in + 128 * i, L);
  ld8(in + 128 * i + 64, R);
  node(K, level, first_index + i, L, R, o);
  st_slot(out + 64 * i, o);
}

// native_rows.cu's rows_levels_kernel
__global__ void __launch_bounds__(THREADS) rows_b3_levels_kernel(const __grid_constant__ B3K K, const uint8_t* in, uint8_t* out, u64 n_in,
                                                                 int level, u64 base, int n_levels) {
  const u64 first = (u64)blockIdx.x << n_levels;
  in += 64 * first;
  for (int k = 0; k < n_levels; ++k) {
    const u64 n_out = n_in / 2, len_out = 1ull << (n_levels - k - 1), at = first >> (k + 1);
    uint8_t* dst = out + 64 * at;
    for (u64 i = threadIdx.x; i < len_out; i += blockDim.x) {
      u32 L[8], R[8], o[8];
      ld8_written(in + 128 * i, L);
      ld8_written(in + 128 * i + 64, R);
      node(K, level + k, (base >> (k + 1)) + at + i, L, R, o);
      st_slot(dst + 64 * i, o);
    }
    __syncthreads();
    in = dst;
    out += 64 * n_out;
    n_in = n_out;
  }
}

// --- host: the step's parameters ---------------------------------------------------------------------------------

const char FRAME[] = "veritor/protocol/merkle/frame/v3";  // + its NUL: 33 bytes

std::vector<uint8_t> message(const char* tag, const std::vector<std::vector<uint8_t>>& parts) {
  std::vector<uint8_t> m(FRAME, FRAME + sizeof FRAME);
  const size_t tl = strlen(tag);
  for (int i = 3; i >= 0; --i) m.push_back((uint8_t)(tl >> (8 * i)));
  m.insert(m.end(), tag, tag + tl);
  for (const auto& p : parts) {
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t)((u64)p.size() >> (8 * i)));
    m.insert(m.end(), p.begin(), p.end());
  }
  return m;
}
std::vector<uint8_t> zeros(int n) { return std::vector<uint8_t>(n, 0); }
void words_of(const uint8_t* b, u32* w, int n) {
  for (int i = 0; i < n; ++i) w[i] = (u32)b[4 * i] | (u32)b[4 * i + 1] << 8 | (u32)b[4 * i + 2] << 16 | (u32)b[4 * i + 3] << 24;
}
// the chaining value after a message's first 64 bytes (the message is longer)
void first_block(const u32* key, const uint8_t* block, u32* cv) {
  u32 w[16];
  words_of(block, w, 16);
  b3c(key, w, 0, 64, B3_KEYED | B3_START, cv);
}
// bytes 64 .. 64 + 4 n of msg (zero past its end) as n words; false if the message is not longer than one block or than that
bool tail_words(const std::vector<uint8_t>& msg, u32* w, int n) {
  if (msg.size() <= 64 || msg.size() > 64 + 4 * (size_t)n) return false;
  std::vector<uint8_t> b(4 * n, 0);
  memcpy(b.data(), msg.data() + 64, msg.size() - 64);
  words_of(b.data(), w, n);
  return true;
}

}  // namespace

extern "C" int verity_rows_b3_params_bytes() { return static_cast<int>(sizeof(B3K)); }

// The step's parameters into `out` (verity_rows_b3_params_bytes()): the hash key (32 bytes), the domain id (32), the salt key Ks (32),
// Q(l), the salt prefix and the leaf prefix (64 each), the leaf schema (SCHEMA_LEN bytes) and the row's bytes l / 8.  0, or -1 for a
// schema of another length or a template that does not have the messages' block count.
extern "C" int verity_rows_b3_build(const uint8_t* hash_key, const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix,
                                    const uint8_t* salt_prefix, const uint8_t* leaf_prefix, const char* schema, int schema_len, uint32_t lb,
                                    void* out) {
  if (schema_len != SCHEMA_LEN || lb == 0) return -1;
  B3K k{};
  k.lb = lb;
  words_of(hash_key, k.key, 8);
  words_of(ks, k.ks, 8);
  first_block(k.key, row_prefix, k.mid_q);
  first_block(k.key, salt_prefix, k.mid_salt);
  first_block(k.key, leaf_prefix, k.mid_leaf);
  const std::vector<uint8_t> did(domain_id, domain_id + 32);
  const std::vector<uint8_t> sch(schema, schema + schema_len);
  for (int nb = 1; nb <= MAX_NB; ++nb) {
    const auto fl = message("leaf", {did, zeros(nb), zeros(nb), sch, zeros(32)});
    const auto pm = message("pad", {did, zeros(nb)});
    const auto nm = message("node", {did, std::vector<uint8_t>{0}, zeros(nb), zeros(32), zeros(32)});
    if (fl.size() != (size_t)(163 + 2 * nb) || pm.size() != (size_t)(88 + nb) || nm.size() != (size_t)(178 + nb)) return -1;
    if (!tail_words(fl, k.fl_tpl[nb - 1], 32) || !tail_words(pm, k.pad_tpl[nb - 1], 16) || !tail_words(nm, k.node_tpl[nb - 1], 32)) return -1;
    if (nb == 1) {
      first_block(k.key, fl.data(), k.mid_fl);
      first_block(k.key, pm.data(), k.mid_pad);
      first_block(k.key, nm.data(), k.mid_node);
    }
  }
  memcpy(out, &k, sizeof k);
  return 0;
}

extern "C" int verity_rows_b3_leaves_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                            const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out, uint8_t* bc,
                                            uint8_t* hs, uint64_t max_blocks, cudaStream_t st) {
  const B3K* p = static_cast<const B3K*>(params);
  const u64 chunk = max_blocks ? max_blocks * THREADS : m;
  for (u64 c = 0; c < m; c += chunk) {
    const u64 mc = m - c < chunk ? m - c : chunk;
    rows_b3_leaf_kernel<<<static_cast<unsigned>((mc + THREADS - 1) / THREADS), THREADS, 0, st>>>(
        *p, c < n_rows ? rows + c * stride : rows, stride, c < n_rows ? n_rows - c : 0, salts && c < n_rows ? salts + SALT * c : salts, tn,
        first + c, mc, n_total, out + 64 * c, bc ? bc + 128 * c : bc, hs ? hs + 64 * c : hs);
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

extern "C" int verity_rows_b3_level_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_out, int level,
                                           uint64_t first_index, uint64_t max_blocks, cudaStream_t st) {
  if (!n_out) return 0;
  if (level < 0 || level + 1 >= LEVELS) return static_cast<int>(cudaErrorInvalidValue);
  const B3K* p = static_cast<const B3K*>(params);
  const u64 chunk = max_blocks ? max_blocks * THREADS : n_out;
  for (u64 c = 0; c < n_out; c += chunk) {
    const u64 nc = n_out - c < chunk ? n_out - c : chunk;
    rows_b3_level_kernel<<<static_cast<unsigned>((nc + THREADS - 1) / THREADS), THREADS, 0, st>>>(*p, in + 128 * c, out + 64 * c, nc, level,
                                                                                                  first_index + c);
    if (const cudaError_t rc = cudaGetLastError()) return static_cast<int>(rc);
  }
  return 0;
}

extern "C" int verity_rows_b3_levels_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_in, int level, uint64_t base,
                                            int n_levels, cudaStream_t st) {
  if (n_levels <= 0) return 0;
  if (n_levels > 10 || level < 0 || level + n_levels >= LEVELS || !n_in || n_in % (1ull << n_levels))
    return static_cast<int>(cudaErrorInvalidValue);
  const B3K* p = static_cast<const B3K*>(params);
  rows_b3_levels_kernel<<<static_cast<unsigned>(n_in >> n_levels), THREADS, 0, st>>>(*p, in, out, n_in, level, base, n_levels);
  return static_cast<int>(cudaGetLastError());
}
