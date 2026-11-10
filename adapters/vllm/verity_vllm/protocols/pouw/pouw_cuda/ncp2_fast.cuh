// `ncp2.cuh`'s values computed the serving kernel's way: every hash from word-sized pieces held in registers (no byte
// streaming, no runtime-indexed arrays), and a tile's checkpoints hashed as they are made. `ncp2_host.cpp`'s
// `ncp2_commit_fast` runs them on the CPU and `ncp2_fused.cu` on the device; `pouw_device.first_difference` holds both to
// `pouw_native`.
#pragma once
#include "ncp2.cuh"

namespace ncp2 {

HD uint32_t bswap32(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, 0, 0x0123);
#else
  return __builtin_bswap32(x);
#endif
}

// Bytes 1, 2, 3 of ``x`` then byte 0 of ``y``, as one big-endian word: a message word when little-endian stream words sit
// 3 bytes into the message (after the 27-byte unit tag).
HD uint32_t shift3(uint32_t x, uint32_t y) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, y, 0x1234);
#else
  return (((x >> 8) & 0xFFu) << 24) | (((x >> 16) & 0xFFu) << 16) | ((x >> 24) << 8) | (y & 0xFFu);
#endif
}

HD uint32_t be32(const uint8_t* p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

HD uint64_t be64(const uint8_t* p) { return ((uint64_t)be32(p) << 32) | be32(p + 4); }

HD uint64_t le64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

// Sixteen bytes as four little-endian words (16-byte aligned on the device).
HD void load16(const uint8_t* p, uint32_t w[4]) {
#ifdef __CUDA_ARCH__
  uint4 v = *(const uint4*)p;
  w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
#else
  for (int i = 0; i < 4; ++i) w[i] = p[4 * i] | (p[4 * i + 1] << 8) | (p[4 * i + 2] << 16) | ((uint32_t)p[4 * i + 3] << 24);
#endif
}

// ---- a unit leaf over a word stream of a multiple of 16 words -----------------------------------------------------------
// SHA-256(UNIT_TAG ‖ words as little-endian u32). With the 27-byte tag in front, every 16 stream words fill exactly one
// block: its big-endian words are shift3 over [the previous 16 words' last 7 | these 16's first 10], and the tag is the
// first "previous 7".

struct WordLeaf {
  uint32_t h[8];
  uint32_t p[7];
  uint32_t n;
};

HD void wl_init(WordLeaf& s) {
  const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint8_t* T = (const uint8_t*)NCP2_UNIT_TAG;
#pragma unroll
  for (int i = 0; i < 8; ++i) s.h[i] = iv[i];
#pragma unroll
  for (int k = 0; k < 7; ++k)
    s.p[k] = (k ? (uint32_t)T[4 * k - 1] : 0u) | ((uint32_t)T[4 * k] << 8) | ((uint32_t)T[4 * k + 1] << 16) | ((uint32_t)T[4 * k + 2] << 24);
  s.n = 0;
}

HD void wl_block(WordLeaf& s, const uint32_t w[16]) {
  uint32_t b[16];
#pragma unroll
  for (int i = 0; i < 6; ++i) b[i] = shift3(s.p[i], s.p[i + 1]);
  b[6] = shift3(s.p[6], w[0]);
#pragma unroll
  for (int i = 0; i < 9; ++i) b[7 + i] = shift3(w[i], w[i + 1]);
  sha256_compress(s.h, b);
#pragma unroll
  for (int i = 0; i < 7; ++i) s.p[i] = w[9 + i];
  s.n += 16;
}

HD void wl_final(WordLeaf& s, uint8_t out[32]) {
  const uint64_t bits = (27 + 4ull * s.n) * 8;
  uint32_t b[16];
#pragma unroll
  for (int i = 0; i < 6; ++i) b[i] = shift3(s.p[i], s.p[i + 1]);
  b[6] = shift3(s.p[6], 0x80);
#pragma unroll
  for (int i = 7; i < 14; ++i) b[i] = 0;
  b[14] = (uint32_t)(bits >> 32);
  b[15] = (uint32_t)bits;
  sha256_compress(s.h, b);
#pragma unroll
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(s.h[i >> 2] >> (24 - 8 * (i & 3)));
}

// One word's leaf (a dequantization unit): 31 bytes, one block.
HD void leaf_word1(uint32_t w, uint8_t out[32]) {
  WordLeaf s;
  wl_init(s);
  uint32_t b[16];
#pragma unroll
  for (int i = 0; i < 6; ++i) b[i] = shift3(s.p[i], s.p[i + 1]);
  b[6] = shift3(s.p[6], w);
  b[7] = shift3(w, 0x80);
#pragma unroll
  for (int i = 8; i < 15; ++i) b[i] = 0;
  b[15] = 31 * 8;
  sha256_compress(s.h, b);
#pragma unroll
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(s.h[i >> 2] >> (24 - 8 * (i & 3)));
}

// A digest's leaf (digest and node units): its 16 big-endian words as the stream.
HD void leaf_digest_fast(const uint8_t d[64], uint8_t out[32]) {
  WordLeaf s;
  wl_init(s);
  uint32_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = be32(d + 4 * i);
  wl_block(s, w);
  wl_final(s, out);
}

// A strip of bytes, each byte one stream word (form_x, and form_y's Y), 16 bytes per block; ``nbytes`` a multiple of 16.
HD void wl_bytes(WordLeaf& s, const uint8_t* p, size_t nbytes) {
  for (size_t o = 0; o < nbytes; o += 16) {
    uint32_t q[4], w[16];
    load16(p + o, q);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = (q[i >> 2] >> (8 * (i & 3))) & 0xFFu;
    wl_block(s, w);
  }
}

// ---- SHA-512 over an 8-byte tag and 8-byte-aligned data (the key's row, strip and node digests) ------------------------------

HD void sha512_tag8(uint64_t tag, const uint8_t* d, size_t nbytes, uint8_t out[64]) {
  uint64_t h[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                   0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  const size_t nw = 1 + nbytes / 8, full = nw / 16, r = nw % 16;
  const uint64_t bits = (uint64_t)(8 + nbytes) * 8;
  uint64_t w[16];
  for (size_t b = 0; b < full; ++b) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      size_t j = 16 * b + i;
      w[i] = j == 0 ? tag : be64(d + 8 * (j - 1));
    }
    sha512_compress(h, w);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    size_t j = 16 * full + i;
    w[i] = (size_t)i < r ? (j == 0 ? tag : be64(d + 8 * (j - 1))) : ((size_t)i == r ? 0x8000000000000000ull : 0ull);
  }
  if (r <= 13) {
    w[14] = 0;
    w[15] = bits;
    sha512_compress(h, w);
  } else {
    sha512_compress(h, w);
#pragma unroll
    for (int i = 0; i < 15; ++i) w[i] = 0;
    w[15] = bits;
    sha512_compress(h, w);
  }
#pragma unroll
  for (int i = 0; i < 64; ++i) out[i] = (uint8_t)(h[i >> 3] >> (56 - 8 * (i & 7)));
}

HD void row_digest_fast(const int8_t* row, int k, uint8_t out[64]) {
  sha512_tag8(be64((const uint8_t*)NCP2_TAG_ROW), (const uint8_t*)row, (size_t)k, out);
}

HD void strip_digest_fast(const uint8_t rows[D * 64], uint8_t out[64]) {
  sha512_tag8(be64((const uint8_t*)NCP2_TAG_STRIP), rows, D * 64, out);
}

HD void tree_node_fast(const uint8_t* left, const uint8_t* right, uint8_t out[64]) {
  uint8_t both[128];
  for (int i = 0; i < 64; ++i) {
    both[i] = left[i];
    both[64 + i] = right[i];
  }
  sha512_tag8(be64((const uint8_t*)NCP2_TAG_NODE), both, 128, out);
}

// `rfc6962` with the fast node.
HD const uint8_t* rfc6962_fast(const uint8_t* strips, int lo, int hi, uint8_t* nodes, int* count) {
  struct Frame { int lo, hi, split; const uint8_t* left; };
  Frame st[21];
  int top = 0;
  const uint8_t* done = nullptr;
  st[0] = {lo, hi, 0, nullptr};
  while (true) {
    Frame& f = st[top];
    if (done == nullptr) {
      if (f.hi - f.lo == 1) {
        done = strips + 64 * (size_t)f.lo;
      } else {
        int split = 1;
        while (split * 2 < f.hi - f.lo) split *= 2;
        f.split = split;
        st[++top] = {f.lo, f.lo + split, 0, nullptr};
        continue;
      }
    } else if (f.left == nullptr) {
      f.left = done;
      done = nullptr;
      st[++top] = {f.lo + f.split, f.hi, 0, nullptr};
      continue;
    } else {
      uint8_t* out = nodes + 64 * (size_t)(*count)++;
      tree_node_fast(f.left, done, out);
      done = out;
    }
    if (top == 0) return done;
    --top;
  }
}

// K in one Keccak block: D_A's words as lanes, then the tag, salt, forward, position and weight id, fill 16 lanes (128 bytes).
HD void call_key_fast(const uint8_t salt[32], uint64_t forward, uint64_t position, uint64_t weight, const uint8_t da[64],
                      uint8_t out[32]) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) s[w] = be64(da + 8 * w);
  s[8] = le64((const uint8_t*)NCP2_TAG_KEY);
#pragma unroll
  for (int i = 0; i < 4; ++i) s[9 + i] = le64(salt + 8 * i);
  s[13] = forward;
  s[14] = position;
  s[15] = weight;
  s[16] ^= 0x1Full ^ (0x80ull << 56);
  keccak_p<24>(s);
#pragma unroll
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(s[i >> 3] >> (8 * (i & 7)));
}

// ---- the noise, squeezed a block at a time --------------------------------------------------------------------------------

HD void squeeze_noise(uint64_t s[25], int k, int8_t* out) {
  int l = 0;
  while (true) {
#pragma unroll
    for (int j = 0; j < 17; ++j) {
      const uint64_t v = s[j];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        if (l < k) out[l] = (int8_t)noise((uint8_t)(v >> (8 * b)));
        ++l;
      }
    }
    if (l >= k) return;
    keccak_p<24>(s);
  }
}

HD void e1_row_fast(const uint8_t key[32], uint32_t rowid, int k, int8_t* e) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  const uint64_t v = ((uint64_t)rowid << 32) | rowid;
#pragma unroll
  for (int i = 0; i < 5; ++i) s[i] = v;
  s[5] = le64((const uint8_t*)NCP2_TAG_E1);
#pragma unroll
  for (int i = 0; i < 4; ++i) s[6 + i] = le64(key + 8 * i);
  s[10] ^= 0x1Full;
  s[16] ^= 0x80ull << 56;
  keccak_p<24>(s);
  squeeze_noise(s, k, e);
}

HD void f1_col_fast(const uint8_t salt[32], uint64_t weight, uint32_t colid, int k, int8_t* f) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  const uint64_t v = ((uint64_t)colid << 32) | colid;
#pragma unroll
  for (int i = 0; i < 5; ++i) s[i] = v;
  s[5] = le64((const uint8_t*)NCP2_TAG_F1);
#pragma unroll
  for (int i = 0; i < 4; ++i) s[6 + i] = le64(salt + 8 * i);
  s[10] = weight;
  s[11] ^= 0x1Full;
  s[16] ^= 0x80ull << 56;
  keccak_p<24>(s);
  squeeze_noise(s, k, f);
}

// The no-hash control's noise: a counter-based stand-in in the same range, so X and Y are formed at the same cost.
HD void standin_noise(uint32_t id, uint32_t salt, int k, int8_t* out) {
  uint32_t x = id * 0x9E3779B9u ^ salt;
  for (int l = 0; l < k; ++l) {
    x = x * 1664525u + 1013904223u;
    out[l] = (int8_t)noise((uint8_t)(x >> 24));
  }
}

// ---- a tile, hashed as it is computed -------------------------------------------------------------------------------------

// Σ over 4 bytes of u8(a)·s8(b), plus c.
HD int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
#ifdef __CUDA_ARCH__
  int32_t d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  for (int i = 0; i < 4; ++i) c += (int32_t)((a >> (8 * i)) & 0xFFu) * (int32_t)(int8_t)(b >> (8 * i));
  return c;
#endif
}

// One tile: X strip rows ``xs`` and Y strip columns ``ys`` (16 × k3 bytes each), ``run(j)`` the running sum of output
// j = 16·r + c, starting at c₀. Every 16-deep step adds X·Y's step to each running sum and hands row r's 16 words to the
// leaf as one block, step-major then row-major; at the end ``run`` holds Z. With HASH false, the same arithmetic unhashed.
template <bool HASH, class Run>
HD void tile_fused(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  WordLeaf wl;
  if (HASH) wl_init(wl);
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    for (int r = 0; r < 16; ++r) {
      uint32_t xv[4], w[16];
      load16(xs + (size_t)r * k3 + 16 * t, xv);
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        int32_t acc = dot4(xv[0], yv[c][0], 0);
        acc = dot4(xv[1], yv[c][1], acc);
        acc = dot4(xv[2], yv[c][2], acc);
        acc = dot4(xv[3], yv[c][3], acc);
        uint32_t v = run(16 * r + c) + (uint32_t)acc;
        run(16 * r + c) = v;
        w[c] = v;
      }
      if (HASH) wl_block(wl, w);
    }
  }
  if (HASH) wl_final(wl, leaf);
}

// The low-byte leaf variant (cheap-binding's NCP-INT row 4; its γ is unproved: timed only, not committed, never a result): TurboSHAKE128 with domain byte 0x0B over
// the low byte of every checkpoint in the word leaf's order, 8 rows (128 bytes, 16 lanes) per permutation, each block's other
// 5 lanes of the rate zero.
constexpr uint8_t LOWBYTE_DS = 0x0B;

template <class Run>
HD void tile_fused_lowbyte(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int r = 8 * half + rr;
        uint32_t xv[4];
        load16(xs + (size_t)r * k3 + 16 * t, xv);
        uint64_t lo = 0, hi = 0;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          int32_t acc = dot4(xv[0], yv[c][0], 0);
          acc = dot4(xv[1], yv[c][1], acc);
          acc = dot4(xv[2], yv[c][2], acc);
          acc = dot4(xv[3], yv[c][3], acc);
          uint32_t v = run(16 * r + c) + (uint32_t)acc;
          run(16 * r + c) = v;
          if (c < 8) lo |= (uint64_t)(v & 0xFFu) << (8 * c);
          else hi |= (uint64_t)(v & 0xFFu) << (8 * (c - 8));
        }
        s[2 * rr] ^= lo;
        s[2 * rr + 1] ^= hi;
      }
      if (t == steps - 1 && half == 1) {
        s[16] ^= LOWBYTE_DS;
        s[20] ^= 0x80ull << 56;
      }
      keccak_p<12>(s);
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) leaf[i] = (uint8_t)(s[i >> 3] >> (8 * (i & 7)));
}

// ---- the units tree's node, from registers ------------------------------------------------------------------------------------
// SHA-256(NODE_TAG (33 bytes) ‖ left ‖ right) over big-endian words: two blocks, the children shifted one byte in.

HD void units_node_fast(const uint32_t L[8], const uint32_t R[8], uint32_t out[8]) {
  const uint8_t* T = (const uint8_t*)NCP2_NODE_TAG;
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint32_t U[16], b[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    U[i] = L[i];
    U[8 + i] = R[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = be32(T + 4 * i);
  b[8] = ((uint32_t)T[32] << 24) | (U[0] >> 8);
#pragma unroll
  for (int i = 1; i < 8; ++i) b[8 + i] = (U[i - 1] << 24) | (U[i] >> 8);
  sha256_compress(h, b);
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = (U[7 + i] << 24) | (U[8 + i] >> 8);
  b[8] = (U[15] << 24) | 0x00800000u;
#pragma unroll
  for (int i = 9; i < 15; ++i) b[i] = 0;
  b[15] = (NODE_TAG_LEN + 64) * 8;
  sha256_compress(h, b);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = h[i];
}

}  // namespace ncp2
