// pouw_hash_bench's check vectors: the inputs, every case, the output record and its JSON lines. Included by pouw_hash_check.cu
// (the check kernels, and the host build's main) and pouw_hash_bench.cu (the timed kernels reuse the streams' seeds).
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pouw_hash.cuh"

constexpr uint64_t PHI = 0x9E3779B97F4A7C15ull;
constexpr uint32_t PHI32 = 0x9E3779B9u;

PH_HD uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// the check tile's words (finite FP32: bit 30 clear), and the quad check's fragment words
PH_HD uint32_t tile_word(int which, int r, int c) { return mix32(2u * (64u * r + c) + which) & 0xBFFFFFFFu; }
PH_HD uint32_t quad_word(int a, int row, int col) { return mix32(0xA5000000u ^ (uint32_t)(a << 16 | row << 8 | col)); }
PH_HD uint8_t ptn(size_t i) { return (uint8_t)(i % 251); }

// ---------------------------------------------------------------- the check vectors
constexpr int NTS = 11, NTSL = 12, NSH = 6, NSHA = 10, NSHAW = 7, NB3 = 8, NTREE = 3;
// the m = 32 tile leaf's two-level form (P6) takes two new domain bytes: these are stand-ins for timing, not a statement's
constexpr uint8_t D_SUBLEAF = 0x24, D_TOPLEAF = 0x25;
struct TsCase { int n, d, out; };
PH_HD TsCase ts_case(int i) {
  const TsCase C[NTS] = {{0, 0x1F, 32}, {0, 0x1F, 200}, {1, 0x1F, 32}, {17, 0x01, 32}, {167, 0x06, 32}, {168, 0x1F, 32},
                         {169, 0x21, 64}, {255, 0x21, 32}, {256, 0x21, 32}, {289, 0x0B, 32}, {4096, 0x22, 32}};
  return C[i];
}
PH_HD int tsl_case(int i) {
  const int C[NTSL] = {0, 1, 20, 21, 22, 32, 41, 42, 63, 128, 160, 512};
  return C[i];
}
PH_HD int sh_case(int i) {
  const int C[NSH] = {0, 135, 136, 137, 168, 1000};
  return C[i];
}
PH_HD int sha_case(int i) {
  const int C[NSHA] = {0, 1, 55, 56, 63, 64, 65, 119, 120, 1000};
  return C[i];
}
PH_HD int shaw_case(int i) {
  const int C[NSHAW] = {0, 1, 13, 14, 16, 30, 64};
  return C[i];
}
PH_HD int b3_case(int i) {
  const int C[NB3] = {0, 1, 63, 64, 65, 104, 1023, 1024};
  return C[i];
}
PH_HD int tree_case(int i) {
  const int C[NTREE] = {2048, 16384, 32768};
  return C[i];
}

struct CheckOut {
  uint8_t ts[NTS][200], tsl[NTSL][32], tss[3][32], shake128[NSH][32], shake256[NSH][32], sha[NSHA][32], shaw[NSHAW][32];
  uint8_t b3[NB3][32], b3k[NB3][32], b3block[32];
  uint64_t pc_d0[4], pc_d127[4], pc_leaf[4], pc_leaf_stream[4], pc_leaf_quad[4], pc_leaf_narrow[4], pc_rows[4];
  uint32_t pc_ticket[2][8];
  uint64_t quad[32][4], msg_stream[4], ts_shf[4];
  uint32_t sha_stream[8], b3_stream[8];
  // BLAKE3's trees (1 KB chunks) and the fine tree (64-byte leaves) of ptn(len), one thread and on a warp or block; the m = 32
  // tile leaf (64 digests, 2 KB) flat and two-level; the quad digest under keyed BLAKE3
  uint32_t b3tree[NTREE][8], b3fine[NTREE][8], b3tree_warp[2][8], b3fine_warp[8], b3fine_block[8], quad_b3[32][8];
  uint64_t leaf32_flat[4], leaf32_two[4];
};

template <class Lane>
PH_HD void tsl_dispatch(int i, uint8_t D, const Lane& lane, uint64_t out[4]) {
  switch (tsl_case(i)) {
    case 0: ph::ts128<0>(D, lane, out); break;
    case 1: ph::ts128<1>(D, lane, out); break;
    case 20: ph::ts128<20>(D, lane, out); break;
    case 21: ph::ts128<21>(D, lane, out); break;
    case 22: ph::ts128<22>(D, lane, out); break;
    case 32: ph::ts128<32>(D, lane, out); break;
    case 41: ph::ts128<41>(D, lane, out); break;
    case 42: ph::ts128<42>(D, lane, out); break;
    case 63: ph::ts128<63>(D, lane, out); break;
    case 128: ph::ts128<128>(D, lane, out); break;
    case 160: ph::ts128<160>(D, lane, out); break;
    default: ph::ts128<512>(D, lane, out); break;
  }
}
template <class Word>
PH_HD void shaw_dispatch(int i, const Word& word, uint32_t h[8]) {
  switch (shaw_case(i)) {
    case 0: ph::sha256<0>(word, h); break;
    case 1: ph::sha256<1>(word, h); break;
    case 13: ph::sha256<13>(word, h); break;
    case 14: ph::sha256<14>(word, h); break;
    case 16: ph::sha256<16>(word, h); break;
    case 30: ph::sha256<30>(word, h); break;
    default: ph::sha256<64>(word, h); break;
  }
}
PH_HD uint8_t tsl_domain(int i) { return (uint8_t)(0x21 + i % 3); }
PH_HD void put_lanes(uint8_t* out, const uint64_t* x, int n) {
  for (int i = 0; i < 8 * n; ++i) out[i] = (uint8_t)(x[i / 8] >> (8 * (i % 8)));
}
PH_HD void put_be(uint8_t* out, const uint32_t* h) {
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(h[i >> 2] >> (24 - 8 * (i & 3)));
}
PH_HD void put_le(uint8_t* out, const uint32_t* h) {
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(h[i >> 2] >> (8 * (i & 3)));
}
PH_HD uint64_t ptn_lane(int j) {
  uint64_t x = 0;
  for (int b = 7; b >= 0; --b) x = x << 8 | ptn(8 * j + b);
  return x;
}
PH_HD uint32_t ptn_le_word(int j) {
  return (uint32_t)ptn(4 * j) | (uint32_t)ptn(4 * j + 1) << 8 | (uint32_t)ptn(4 * j + 2) << 16 | (uint32_t)ptn(4 * j + 3) << 24;
}
PH_HD void b3_test_key(uint32_t key[8]) {
  for (int i = 0; i < 8; ++i) key[i] = (uint32_t)(4 * i) | (uint32_t)(4 * i + 1) << 8 | (uint32_t)(4 * i + 2) << 16 | (uint32_t)(4 * i + 3) << 24;
}
// The m = 32 tile leaf over 256 lanes (64 digests): flat, TurboSHAKE128 under D_LEAF (13 permutations); two-level, 4 sub-leaves
// of 64 lanes under D_SUBLEAF (4 each, in parallel), then their 4 digests under D_TOPLEAF (1).
template <class Lane>
PH_HD void leaf32_two_level(const Lane& lane, uint64_t out[4]) {
  uint64_t sub[16];
  for (int s = 0; s < 4; ++s) ph::ts128<64>(D_SUBLEAF, [&](int j) { return lane(64 * s + j); }, sub + 4 * s);
  ph::ts128<16>(D_TOPLEAF, [&](int j) { return sub[j]; }, out);
}
PH_HD uint32_t ptn_be_word(int j) {
  return (uint32_t)ptn(4 * j) << 24 | (uint32_t)ptn(4 * j + 1) << 16 | (uint32_t)ptn(4 * j + 2) << 8 | ptn(4 * j + 3);
}
// the timed streams' first iteration of thread 0 (seed z = PHI): msg's lanes z + j PHI; sha's and blake3's chains of 3 blocks
PH_HD void msg_stream(uint64_t z, uint64_t d[4]) {
  ph::pearlc::msg_digest([z](int j) { return z + (uint64_t)j * PHI; }, d);
}
PH_HD void sha_chain(uint32_t h[8], uint32_t z32, long long n) {
#pragma unroll 1
  for (long long b = 0; b < n; ++b) {
    uint32_t m[16];
    const uint32_t base = z32 + (uint32_t)(b * 16) * PHI32;
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = base + (uint32_t)i * PHI32;
    ph::sha256_compress(h, m);
  }
}
PH_HD void b3_chain(uint32_t cv[8], uint32_t z32, long long n) {
#pragma unroll 1
  for (long long b = 0; b < n; ++b) {
    uint32_t m[16];
    const uint32_t base = z32 + (uint32_t)(b * 16) * PHI32;
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = base + (uint32_t)i * PHI32;
    ph::blake3_compress(cv, m, 0, 64, 0, cv);
  }
}

// Everything but the quad transpose (which the device runs on a whole warp), in parts that the device runs as separate
// kernels (one function each keeps cicc's time down); buf holds 4,096 bytes of scratch. The loops over cases stay rolled: an
// unrolled loop would copy every inlined permutation once per case.
template <int PART>
PH_HD void run_checks(uint8_t* buf, CheckOut* o) {
  for (int i = 0; i < 4096; ++i) buf[i] = ptn(i);
  if constexpr (PART == 0) {
#pragma unroll 1
    for (int i = 0; i < NTS; ++i) {
      const TsCase c = ts_case(i);
      ph::sponge_bytes<12, 168>(buf, c.n, (uint8_t)c.d, o->ts[i], c.out);
    }
#pragma unroll 1
    for (int i = 0; i < 3; ++i) {
      const int nl = i == 0 ? 21 : i == 1 ? 42 : 512;
      ph::Ts128Stream st;
      ph::tss_init(st);
      for (int j = 0; j < nl; ++j) ph::tss_lane(st, ptn_lane(j));
      uint64_t d[4];
      ph::tss_final(st, 0x22, d);
      put_lanes(o->tss[i], d, 4);
    }
  }
  if constexpr (PART == 1) {
#pragma unroll 1
    for (int i = 0; i < NTSL; ++i) {
      uint64_t d[4];
      tsl_dispatch(i, tsl_domain(i), [](int j) { return ptn_lane(j); }, d);
      put_lanes(o->tsl[i], d, 4);
    }
  }
  if constexpr (PART == 2) {
#pragma unroll 1
    for (int i = 0; i < NSH; ++i) {
      ph::sponge_bytes<24, 168>(buf, sh_case(i), 0x1F, o->shake128[i], 32);
      ph::sponge_bytes<24, 136>(buf, sh_case(i), 0x1F, o->shake256[i], 32);
    }
  }
  if constexpr (PART == 3) {
#pragma unroll 1
    for (int i = 0; i < NSHA; ++i) ph::sha256_bytes(buf, sha_case(i), o->sha[i]);
#pragma unroll 1
    for (int i = 0; i < NSHAW; ++i) {
      uint32_t h[8];
      shaw_dispatch(i, [](int j) { return ptn_be_word(j); }, h);
      put_be(o->shaw[i], h);
    }
    uint32_t key[8];
    for (int i = 0; i < 8; ++i) key[i] = (uint32_t)(4 * i) | (uint32_t)(4 * i + 1) << 8 | (uint32_t)(4 * i + 2) << 16 | (uint32_t)(4 * i + 3) << 24;
#pragma unroll 1
    for (int i = 0; i < NB3; ++i) {
      ph::blake3_chunk(nullptr, buf, b3_case(i), o->b3[i]);
      ph::blake3_chunk(key, buf, b3_case(i), o->b3k[i]);
    }
    uint32_t m[16], h[8];
    for (int i = 0; i < 16; ++i) m[i] = (uint32_t)ptn(4 * i) | (uint32_t)ptn(4 * i + 1) << 8 | (uint32_t)ptn(4 * i + 2) << 16 | (uint32_t)ptn(4 * i + 3) << 24;
    ph::blake3_keyed_block(key, m, h);
    put_le(o->b3block, h);
    ph::sha256_iv(o->sha_stream);
    sha_chain(o->sha_stream, (uint32_t)(PHI >> 32), 3);
    for (int i = 0; i < 8; ++i) o->b3_stream[i] = (uint32_t)PHI + (uint32_t)i * PHI32;
    b3_chain(o->b3_stream, (uint32_t)(PHI >> 32), 3);
  }
  if constexpr (PART == 4) {
    // Pearl-C on the check tile: digests into buf (128 x 32 B), the leaf two ways (the quad transpose's is k_check_quad's)
    uint64_t* dg = reinterpret_cast<uint64_t*>(buf);
#pragma unroll 1
    for (int r = 0; r < 64; ++r)
#pragma unroll 1
      for (int h = 0; h < 2; ++h)
        ph::pearlc::msg_digest(
            [&](int j) {
              const int arr = j >> 4, col = 32 * h + 2 * (j & 15);
              return ph::lane_u32x2(tile_word(arr, r, col), tile_word(arr, r, col + 1));
            },
            dg + 4 * (2 * r + h));
    for (int i = 0; i < 4; ++i) o->pc_d0[i] = dg[i], o->pc_d127[i] = dg[4 * 127 + i];
    ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o->pc_leaf);
    ph::Ts128Stream st;
    ph::tss_init(st);
    for (int i = 0; i < ph::pearlc::LEAF_LANES; ++i) ph::tss_lane(st, dg[i]);
    ph::tss_final(st, ph::pearlc::D_LEAF, o->pc_leaf_stream);
  }
  if constexpr (PART == 6) {
    uint32_t iv[8];
    ph::b3_iv(iv);
#pragma unroll 1
    for (int i = 0; i < NTREE; ++i) {
      ph::blake3_tree<32>(iv, 0, 16, tree_case(i) / 1024, [](int j) { return ptn_le_word(j); }, o->b3tree[i]);
      ph::blake3_tree<512>(iv, 0, 1, tree_case(i) / 64, [](int j) { return ptn_le_word(j); }, o->b3fine[i]);
    }
  }
  if constexpr (PART == 5) {
    uint8_t seed[32], label[20], sub[32];
    const char* L = "pearl/v4/FP8/jackpot";
    for (int i = 0; i < 32; ++i) seed[i] = (uint8_t)(0xC0 + i);
    for (int i = 0; i < 20; ++i) label[i] = (uint8_t)L[i];
    uint32_t sk[8], tk[8];
    for (int i = 0; i < 8; ++i) sk[i] = (uint32_t)seed[4 * i] | (uint32_t)seed[4 * i + 1] << 8 | (uint32_t)seed[4 * i + 2] << 16 | (uint32_t)seed[4 * i + 3] << 24;
    ph::blake3_chunk(sk, label, 20, sub);
    for (int i = 0; i < 8; ++i) tk[i] = (uint32_t)sub[4 * i] | (uint32_t)sub[4 * i + 1] << 8 | (uint32_t)sub[4 * i + 2] << 16 | (uint32_t)sub[4 * i + 3] << 24;
#pragma unroll 1
    for (int k = 0; k < 2; ++k)
      ph::pearlc::ticket(tk, [k](int r, int c) { return tile_word(0, 48 * k + r, 32 * k + c); }, o->pc_ticket[k]);
    ph::pearlc::row_segment<128>([&](int j) { return ph::le64(buf + 8 * j); }, o->pc_rows);
    msg_stream(PHI, o->msg_stream);
    ph::ts128<256>(ph::pearlc::D_LEAF, [](int j) { return ptn_lane(j); }, o->leaf32_flat);
    leaf32_two_level([](int j) { return ptn_lane(j); }, o->leaf32_two);
    // TurboSHAKE128(ptn(168), 0x21) with the other rotations (not PH_KECCAK_SHF's)
    uint64_t st[25] = {};
    for (int w = 0; w < ph::TS_RATE; ++w) st[w] ^= ptn_lane(w);
    ph::keccak_p<12, !PH_KECCAK_SHF>(st);
    st[0] ^= 0x21, st[ph::TS_RATE - 1] ^= 0x8000000000000000ull;
    ph::keccak_p<12, !PH_KECCAK_SHF>(st);
    for (int i = 0; i < 4; ++i) o->ts_shf[i] = st[i];
  }
}
// lane's fragment word for the quad check: array a, n8 block j, c[i] of the m16n8 layout on quad_word's grid
PH_HD float quad_frag(int lane, int a, int j, int i) {
  const int row = (lane >> 2) + 8 * (i >> 1), col = 8 * j + 2 * (lane & 3) + (i & 1);
  const uint32_t w = quad_word(a, row, col);
  float f;
  memcpy(&f, &w, 4);
  return f;
}
// the check tile's leaf through the quad transpose: region ρ's fragments are tile rows 16ρ …, columns 0 … 63
PH_HD float tile_frag(int region, int lane, int a, int j, int i) {
  const int row = 16 * region + (lane >> 2) + 8 * (i >> 1), col = 8 * j + 2 * (lane & 3) + (i & 1);
  const uint32_t w = tile_word(a, row, col);
  float f;
  memcpy(&f, &w, 4);
  return f;
}
// the same leaf through narrow warp tiles (64 x 32, GPU 1's): warp `chunk` holds columns 32·chunk …, and region p pairs its
// m16 blocks 2p and 2p + 1
PH_HD float narrow_frag(int chunk, int p, int lane, int a, int j, int i) {
  const int row = 16 * (2 * p + (j >> 2)) + (lane >> 2) + 8 * (i >> 1), col = 32 * chunk + 8 * (j & 3) + 2 * (lane & 3) + (i & 1);
  const uint32_t w = tile_word(a, row, col);
  float f;
  memcpy(&f, &w, 4);
  return f;
}

static void hex(const char* where, const char* what, int len, int d, const uint8_t* p, int n) {
  printf("{\"group\":\"hash_check\",\"where\":\"%s\",\"what\":\"%s\",\"len\":%d,\"d\":%d,\"out\":\"", where, what, len, d);
  for (int i = 0; i < n; ++i) printf("%02x", p[i]);
  printf("\"}\n");
}
static void print_checks(const CheckOut& o, const char* w) {
  uint8_t b[64];
  for (int i = 0; i < NTS; ++i) hex(w, "turboshake128 bytes", ts_case(i).n, ts_case(i).d, o.ts[i], ts_case(i).out);
  for (int i = 0; i < NTSL; ++i) hex(w, "turboshake128 lanes", tsl_case(i), tsl_domain(i), o.tsl[i], 32);
  for (int i = 0; i < 3; ++i) hex(w, "turboshake128 stream", i == 0 ? 21 : i == 1 ? 42 : 512, 0x22, o.tss[i], 32);
  for (int i = 0; i < NSH; ++i) hex(w, "shake128", sh_case(i), 0x1F, o.shake128[i], 32), hex(w, "shake256", sh_case(i), 0x1F, o.shake256[i], 32);
  for (int i = 0; i < NSHA; ++i) hex(w, "sha256 bytes", sha_case(i), 0, o.sha[i], 32);
  for (int i = 0; i < NSHAW; ++i) hex(w, "sha256 words", shaw_case(i), 0, o.shaw[i], 32);
  for (int i = 0; i < NB3; ++i) hex(w, "blake3", b3_case(i), 0, o.b3[i], 32), hex(w, "blake3 keyed", b3_case(i), 0, o.b3k[i], 32);
  hex(w, "blake3 keyed block", 64, 0, o.b3block, 32);
  put_lanes(b, o.pc_d0, 4), hex(w, "pearl-c digest 0", 256, 0x21, b, 32);
  put_lanes(b, o.pc_d127, 4), hex(w, "pearl-c digest 127", 256, 0x21, b, 32);
  put_lanes(b, o.pc_leaf, 4), hex(w, "pearl-c leaf", 4096, 0x22, b, 32);
  put_lanes(b, o.pc_leaf_stream, 4), hex(w, "pearl-c leaf stream", 4096, 0x22, b, 32);
  put_lanes(b, o.pc_leaf_quad, 4), hex(w, "pearl-c leaf quad", 4096, 0x22, b, 32);
  put_lanes(b, o.pc_leaf_narrow, 4), hex(w, "pearl-c leaf narrow", 4096, 0x22, b, 32);
  put_lanes(b, o.pc_rows, 4), hex(w, "pearl-c row segment", 1024, 0x23, b, 32);
  put_le(b, o.pc_ticket[0]), hex(w, "pearl-c ticket 0 0", 64, 0, b, 32);
  put_le(b, o.pc_ticket[1]), hex(w, "pearl-c ticket 48 32", 64, 0, b, 32);
  for (int l = 0; l < 32; ++l) {
    char what[32];
    snprintf(what, sizeof what, "quad lane %d", l);
    put_lanes(b, o.quad[l], 4), hex(w, what, 256, 0x21, b, 32);
  }
  put_lanes(b, o.msg_stream, 4), hex(w, "msg stream", 256, 0x21, b, 32);
  put_lanes(b, o.ts_shf, 4), hex(w, "turboshake128 other rotations", 21, 0x21, b, 32);
  put_be(b, o.sha_stream), hex(w, "sha stream 3 blocks", 192, 0, b, 32);
  put_le(b, o.b3_stream), hex(w, "blake3 stream 3 blocks", 192, 0, b, 32);
  for (int i = 0; i < NTREE; ++i)
    put_le(b, o.b3tree[i]), hex(w, "blake3 tree", tree_case(i), 0, b, 32), put_le(b, o.b3fine[i]), hex(w, "blake3 fine tree", tree_case(i), 0, b, 32);
  put_le(b, o.b3tree_warp[0]), hex(w, "blake3 tree warp", 2048, 0, b, 32);
  put_le(b, o.b3tree_warp[1]), hex(w, "blake3 tree warp", 32768, 0, b, 32);
  put_le(b, o.b3fine_warp), hex(w, "blake3 fine tree warp", 2048, 0, b, 32);
  put_le(b, o.b3fine_block), hex(w, "blake3 fine tree block", 32768, 0, b, 32);
  put_lanes(b, o.leaf32_flat, 4), hex(w, "leaf32 flat", 2048, ph::pearlc::D_LEAF, b, 32);
  put_lanes(b, o.leaf32_two, 4), hex(w, "leaf32 two-level", 2048, D_TOPLEAF, b, 32);
  for (int l = 0; l < 32; ++l) {
    char what[32];
    snprintf(what, sizeof what, "quad b3 lane %d", l);
    put_le(b, o.quad_b3[l]), hex(w, what, 256, 0, b, 32);
  }
  fflush(stdout);
}
