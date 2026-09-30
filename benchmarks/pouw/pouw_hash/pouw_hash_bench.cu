// pouw_hash.cuh's vectors, rates and epilogue cost, as JSON lines.
//   pouw_hash_host check                  (g++ -x c++, no CUDA) the vectors from the host build
//   pouw_hash_bench check                 the same vectors from the device (one thread; the quad transpose on one warp)
//   pouw_hash_bench rates [ms] [filter]   every lane busy, then one thread's latency, then the concurrency sweep
//   pouw_hash_bench epi [ms] [filter]     an FP8 mma.sync GEMM tile loop with Pearl-C's epilogue, hashed and not
// The device build refuses to run (exit 3) unless the visible device's UUID is $POUW_EXPECT_UUID. check.py holds the check
// lines to their references; run.sh runs check first and times nothing unless it passes.
//
// Rates are message bytes per SM per SM clock (per_sm_clk, from each block's clock64 span on its SM) and GB/s (CUDA events
// over the launch). The units per byte that the accounting uses are the device's MACs per SM per clock over per_sm_clk:
// check.py's summary divides by a measured GEMM rate; nothing here assumes one.
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
constexpr int NTS = 11, NTSL = 12, NSH = 6, NSHA = 10, NSHAW = 7, NB3 = 8;
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

struct CheckOut {
  uint8_t ts[NTS][200], tsl[NTSL][32], tss[3][32], shake128[NSH][32], shake256[NSH][32], sha[NSHA][32], shaw[NSHAW][32];
  uint8_t b3[NB3][32], b3k[NB3][32], b3block[32];
  uint64_t pc_d0[4], pc_d127[4], pc_leaf[4], pc_leaf_stream[4], pc_leaf_quad[4], pc_rows[4];
  uint32_t pc_ticket[2][8];
  uint64_t quad[32][4], msg_stream[4];
  uint32_t sha_stream[8], b3_stream[8];
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

// Everything but the quad transpose (which the device runs on a whole warp): buf holds 4,096 bytes of scratch.
PH_HD void run_checks(uint8_t* buf, CheckOut* o) {
  for (int i = 0; i < 4096; ++i) buf[i] = ptn(i);
  for (int i = 0; i < NTS; ++i) {
    const TsCase c = ts_case(i);
    ph::sponge_bytes<12, 168>(buf, c.n, (uint8_t)c.d, o->ts[i], c.out);
  }
  for (int i = 0; i < NTSL; ++i) {
    uint64_t d[4];
    tsl_dispatch(i, tsl_domain(i), [](int j) { return ptn_lane(j); }, d);
    put_lanes(o->tsl[i], d, 4);
  }
  for (int i = 0; i < 3; ++i) {
    const int nl = i == 0 ? 21 : i == 1 ? 42 : 512;
    ph::Ts128Stream st;
    ph::tss_init(st);
    for (int j = 0; j < nl; ++j) ph::tss_lane(st, ptn_lane(j));
    uint64_t d[4];
    ph::tss_final(st, 0x22, d);
    put_lanes(o->tss[i], d, 4);
  }
  for (int i = 0; i < NSH; ++i) {
    ph::sponge_bytes<24, 168>(buf, sh_case(i), 0x1F, o->shake128[i], 32);
    ph::sponge_bytes<24, 136>(buf, sh_case(i), 0x1F, o->shake256[i], 32);
  }
  for (int i = 0; i < NSHA; ++i) ph::sha256_bytes(buf, sha_case(i), o->sha[i]);
  for (int i = 0; i < NSHAW; ++i) {
    uint32_t h[8];
    shaw_dispatch(i, [](int j) { return ptn_be_word(j); }, h);
    put_be(o->shaw[i], h);
  }
  uint32_t key[8];
  for (int i = 0; i < 8; ++i) key[i] = (uint32_t)(4 * i) | (uint32_t)(4 * i + 1) << 8 | (uint32_t)(4 * i + 2) << 16 | (uint32_t)(4 * i + 3) << 24;
  for (int i = 0; i < NB3; ++i) {
    ph::blake3_chunk(nullptr, buf, b3_case(i), o->b3[i]);
    ph::blake3_chunk(key, buf, b3_case(i), o->b3k[i]);
  }
  {
    uint32_t m[16], h[8];
    for (int i = 0; i < 16; ++i) m[i] = (uint32_t)ptn(4 * i) | (uint32_t)ptn(4 * i + 1) << 8 | (uint32_t)ptn(4 * i + 2) << 16 | (uint32_t)ptn(4 * i + 3) << 24;
    ph::blake3_keyed_block(key, m, h);
    put_le(o->b3block, h);
  }
  // Pearl-C on the check tile: digests into buf (128 x 32 B), the leaf three ways, a row segment, two tickets
  uint64_t* dg = reinterpret_cast<uint64_t*>(buf);
  for (int r = 0; r < 64; ++r)
    for (int h = 0; h < 2; ++h)
      ph::pearlc::msg_digest(
          [&](int j) {
            const int arr = j >> 4, col = 32 * h + 2 * (j & 15);
            return ph::lane_u32x2(tile_word(arr, r, col), tile_word(arr, r, col + 1));
          },
          dg + 4 * (2 * r + h));
  for (int i = 0; i < 4; ++i) o->pc_d0[i] = dg[i], o->pc_d127[i] = dg[4 * 127 + i];
  ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o->pc_leaf);
  {
    ph::Ts128Stream st;
    ph::tss_init(st);
    for (int i = 0; i < ph::pearlc::LEAF_LANES; ++i) ph::tss_lane(st, dg[i]);
    ph::tss_final(st, ph::pearlc::D_LEAF, o->pc_leaf_stream);
  }
  {
    uint8_t seed[32], label[20], sub[32];
    const char* L = "pearl/v4/FP8/jackpot";
    for (int i = 0; i < 32; ++i) seed[i] = (uint8_t)(0xC0 + i);
    for (int i = 0; i < 20; ++i) label[i] = (uint8_t)L[i];
    uint32_t sk[8], tk[8];
    for (int i = 0; i < 8; ++i) sk[i] = (uint32_t)seed[4 * i] | (uint32_t)seed[4 * i + 1] << 8 | (uint32_t)seed[4 * i + 2] << 16 | (uint32_t)seed[4 * i + 3] << 24;
    ph::blake3_chunk(sk, label, 20, sub);
    for (int i = 0; i < 8; ++i) tk[i] = (uint32_t)sub[4 * i] | (uint32_t)sub[4 * i + 1] << 8 | (uint32_t)sub[4 * i + 2] << 16 | (uint32_t)sub[4 * i + 3] << 24;
    ph::pearlc::ticket(tk, [](int r, int c) { return tile_word(0, r, c); }, o->pc_ticket[0]);
    ph::pearlc::ticket(tk, [](int r, int c) { return tile_word(0, 48 + r, 32 + c); }, o->pc_ticket[1]);
  }
  for (int i = 0; i < 4096; ++i) buf[i] = ptn(i);
  ph::pearlc::row_segment<128>([&](int j) { return ph::le64(buf + 8 * j); }, o->pc_rows);
  msg_stream(PHI, o->msg_stream);
  ph::sha256_iv(o->sha_stream);
  sha_chain(o->sha_stream, (uint32_t)(PHI >> 32), 3);
  for (int i = 0; i < 8; ++i) o->b3_stream[i] = (uint32_t)PHI + (uint32_t)i * PHI32;
  b3_chain(o->b3_stream, (uint32_t)(PHI >> 32), 3);
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
  put_lanes(b, o.pc_rows, 4), hex(w, "pearl-c row segment", 1024, 0x23, b, 32);
  put_le(b, o.pc_ticket[0]), hex(w, "pearl-c ticket 0 0", 64, 0, b, 32);
  put_le(b, o.pc_ticket[1]), hex(w, "pearl-c ticket 48 32", 64, 0, b, 32);
  for (int l = 0; l < 32; ++l) {
    char what[32];
    snprintf(what, sizeof what, "quad lane %d", l);
    put_lanes(b, o.quad[l], 4), hex(w, what, 256, 0x21, b, 32);
  }
  put_lanes(b, o.msg_stream, 4), hex(w, "msg stream", 256, 0x21, b, 32);
  put_be(b, o.sha_stream), hex(w, "sha stream 3 blocks", 192, 0, b, 32);
  put_le(b, o.b3_stream), hex(w, "blake3 stream 3 blocks", 192, 0, b, 32);
  fflush(stdout);
}

#ifndef __CUDACC__
// ---------------------------------------------------------------- the host build
int main(int argc, char** argv) {
  if (argc < 2 || strcmp(argv[1], "check")) {
    fprintf(stderr, "usage: pouw_hash_host check\n");
    return 2;
  }
  static uint8_t buf[4096];
  static CheckOut o;
  run_checks(buf, &o);
  ph::quad_msg_digest_warp<2>([](int lane, int a, int j, int i) { return quad_frag(lane, a, j, i); }, 0x21, o.quad);
  uint64_t dg[4 * ph::pearlc::DIGESTS], d[32][4];
  for (int region = 0; region < 4; ++region) {
    ph::quad_msg_digest_warp<2>([&](int lane, int a, int j, int i) { return tile_frag(region, lane, a, j, i); }, 0x21, d);
    for (int lane = 0; lane < 32; ++lane)
      for (int i = 0; i < 4; ++i) dg[4 * ph::pearlc::digest_index(region, lane) + i] = d[lane][i];
  }
  ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o.pc_leaf_quad);
  print_checks(o, "host");
  return 0;
}
#else
// ---------------------------------------------------------------- the device build
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <strings.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));              \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

struct Rec {
  unsigned long long c0, c1, t0, t1;
  unsigned smid, pad;
};
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned get_smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
#define BEGIN_REC                  \
  __syncthreads();                 \
  unsigned long long _c0 = clock64(), _t0 = gtimer();
#define END_REC(rec)                                                                                          \
  __syncthreads();                                                                                            \
  if (threadIdx.x == 0) (rec)[blockIdx.x] = Rec{_c0, (unsigned long long)clock64(), _t0, gtimer(), get_smid(), 0};

// ---- the check kernels
__global__ void k_check(uint8_t* buf, CheckOut* o) { run_checks(buf, o); }
__global__ void k_check_quad(CheckOut* o, uint64_t* dg) {
  __shared__ uint64_t scratch[128];
  const int lane = threadIdx.x;
  uint64_t d[4];
  ph::quad_msg_digest<2>([&](int a, int j, int i) { return quad_frag(lane, a, j, i); }, 0x21, scratch, d);
  for (int i = 0; i < 4; ++i) o->quad[lane][i] = d[i];
  for (int region = 0; region < 4; ++region) {
    ph::quad_msg_digest<2>([&](int a, int j, int i) { return tile_frag(region, lane, a, j, i); }, 0x21, scratch, d);
    for (int i = 0; i < 4; ++i) dg[4 * ph::pearlc::digest_index(region, lane) + i] = d[i];
  }
  __syncwarp();
  if (lane == 0) ph::pearlc::tile_leaf([&](int i) { return dg[i]; }, o->pc_leaf_quad);
}

// ---- every lane busy, and one thread's latency (the same kernels on one thread)
PH_HD uint64_t seed(uint32_t t, long long it) { return (uint64_t)(t + 1) * PHI ^ (uint64_t)it << 32; }
template <int ROUNDS>
__global__ void k_perm(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = seed(t, i);
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) ph::keccak_p<ROUNDS>(s);
  END_REC(rec)
  uint64_t x = 0;
#pragma unroll
  for (int i = 0; i < 25; ++i) x ^= s[i];
  out[t] = x;
}
__global__ void k_sha(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t h[8];
  ph::sha256_iv(h);
  BEGIN_REC
  sha_chain(h, (uint32_t)(seed(t, 0) >> 32), iters);
  END_REC(rec)
  out[t] = (uint64_t)h[0] << 32 | h[7];
}
__global__ void k_b3(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cv[8];
  for (int i = 0; i < 8; ++i) cv[i] = (uint32_t)seed(t, 0) + (uint32_t)i * PHI32;
  BEGIN_REC
  b3_chain(cv, (uint32_t)(seed(t, 0) >> 32), iters);
  END_REC(rec)
  out[t] = (uint64_t)cv[0] << 32 | cv[7];
}
// Pearl-C's message digest from registers: 256 message bytes, 2 permutations
__global__ void k_msg(long long iters, Rec* rec, uint64_t* out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    uint64_t d[4];
    msg_stream(seed(t, it), d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}
// the same messages from mma.sync fragments in registers, through the quad transpose (1 KB of shared memory per warp)
__global__ void k_quad(long long iters, Rec* rec, uint64_t* out) {
  extern __shared__ uint64_t smem[];
  uint64_t* scratch = smem + (threadIdx.x >> 5) * 128;
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    const uint32_t z = (uint32_t)seed(t, it);
    uint64_t d[4];
    ph::quad_msg_digest<2>([z](int a, int j, int i) { return __uint_as_float(z + (uint32_t)(32 * a + 4 * j + i) * PHI32); }, 0x21,
                           scratch, d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}
// Pearl-C's tile leaf from an L2-resident buffer of digests: 4,096 message bytes, 25 permutations
__global__ void k_leaf(long long iters, const uint64_t* dg, int nleaf, Rec* rec, uint64_t* out) {
  const uint32_t T = gridDim.x * blockDim.x, t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t acc = 0;
  BEGIN_REC
#pragma unroll 1
  for (long long it = 0; it < iters; ++it) {
    const uint64_t* p = dg + (size_t)((t + it * T) % nleaf) * ph::pearlc::LEAF_LANES;
    uint64_t d[4];
    ph::pearlc::tile_leaf([p](int i) { return p[i]; }, d);
    acc ^= d[0] ^ d[3];
  }
  END_REC(rec)
  out[t] = acc;
}

// ---- the epilogue: an FP8 mma.sync tile loop with Pearl-C's checked words
// Each warp owns a 32 x 64 output tile (2 regions of m16 x n64: 16 FP8 m16n8k32 per k-step, 64 FP32 accumulators) and runs
// `tiles` tiles of `ksteps` k-steps (k = 32 ksteps), operands from a shared-memory ring (lane-major, conflict-free, the same
// bytes for every warp: a stand-in for the real kernel's ldmatrix stages, without global loads). After the chain: C~ is copied,
// the peel (four BF16 m16n8k16 steps per n8 block, on stand-in factors) turns the accumulators into U, y = bf16(U) is stored,
// and with HASH each region's 32 messages (C~ ‖ U per row and chunk) go through quad_msg_digest into a digest buffer (the
// leaves are a separate pass: k_leaf's rate). Without HASH the same kernel skips the digests: the difference is the hashing's
// cost inside the kernel, at the kernel's own register pressure and warps in flight.
constexpr int EPI_STAGES = 4, EPI_A = 2 * 32 * 4, EPI_B = 32 * 16;   // words per stage: A (2 m16 blocks), B (8 n8 blocks)
__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int HASH, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) k_epi(int tiles, int ksteps, Rec* rec, uint64_t* digests, uint32_t* y) {
  extern __shared__ uint32_t sm[];
  uint32_t* ring = sm;                                                                       // EPI_STAGES x (EPI_A + EPI_B)
  uint64_t* scratch = reinterpret_cast<uint64_t*>(sm + EPI_STAGES * (EPI_A + EPI_B)) + (threadIdx.x >> 5) * 128;
  const int lane = threadIdx.x & 31, warp = blockIdx.x * WARPS + (threadIdx.x >> 5);
  for (int i = threadIdx.x; i < EPI_STAGES * (EPI_A + EPI_B); i += blockDim.x) ring[i] = 0x38303830u ^ (mix32(i) & 0x07070707u);
  uint32_t pa[4] = {0x3F803F80u, 0x3F003F00u, 0x3E803E80u, 0x3F403F40u}, pb0 = 0x3F803F80u, pb1 = 0x3C003C00u;
  BEGIN_REC
#pragma unroll 1
  for (int tile = 0; tile < tiles; ++tile) {
    float acc[2][8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[r][j][i] = 0.f;
#pragma unroll 1
    for (int ks = 0; ks < ksteps; ++ks) {
      const uint32_t* st = ring + (ks % EPI_STAGES) * (EPI_A + EPI_B);
      uint32_t a[2][4];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint4 v = reinterpret_cast<const uint4*>(st)[r * 32 + lane];
        a[r][0] = v.x, a[r][1] = v.y, a[r][2] = v.z, a[r][3] = v.w;
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = reinterpret_cast<const uint4*>(st + EPI_A)[q * 32 + lane];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          mma_fp8(acc[r][2 * q], a[r], v.x, v.y);
          mma_fp8(acc[r][2 * q + 1], a[r], v.z, v.w);
        }
      }
    }
    float ct[2][8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) ct[r][j][i] = acc[r][j][i];
#pragma unroll
    for (int step = 0; step < 4; ++step)
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_bf16(acc[r][j], pa, pb0 ^ step, pb1);
    uint32_t* yt = y + ((size_t)warp * 2 + (tile & 1)) * 1024;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 v2 = __floats2bfloat162_rn(acc[r][j][0], acc[r][j][1]);
        __nv_bfloat162 w2 = __floats2bfloat162_rn(acc[r][j][2], acc[r][j][3]);
        yt[(r * 8 + j) * 64 + lane] = *reinterpret_cast<uint32_t*>(&v2);
        yt[(r * 8 + j) * 64 + 32 + lane] = *reinterpret_cast<uint32_t*>(&w2);
      }
    if (HASH) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        uint64_t d[4];
        ph::quad_msg_digest<2>([&](int arr, int j, int i) { return arr ? acc[r][j][i] : ct[r][j][i]; }, ph::pearlc::D_MESSAGE,
                               scratch, d);
        uint64_t* out = digests + (((size_t)warp * 2 + (tile & 1)) * 64 + r * 32 + lane) * 4;
        out[0] = d[0], out[1] = d[1], out[2] = d[2], out[3] = d[3];
      }
    }
  }
  END_REC(rec)
}
// ---- the harness
static nvmlDevice_t g_nvml;
static bool g_nvml_ok = false;
static int g_sms;
static double g_target_ms = 100;
static char g_uuid[64];
struct Clk {
  unsigned sm = 0, mem = 0, power_mw = 0, temp = 0;
  unsigned long long thr = 0;
};
static Clk read_clk() {
  Clk c;
  if (!g_nvml_ok) return c;
  nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &c.sm);
  nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_MEM, &c.mem);
  nvmlDeviceGetCurrentClocksEventReasons(g_nvml, &c.thr);
  nvmlDeviceGetPowerUsage(g_nvml, &c.power_mw);
  nvmlDeviceGetTemperature(g_nvml, NVML_TEMPERATURE_GPU, &c.temp);
  return c;
}
static std::string clk_json(const char* k, const Clk& c) {
  char b[256];
  snprintf(b, sizeof b, "\"%s\":{\"sm_mhz\":%u,\"mem_mhz\":%u,\"reasons\":\"0x%llx\",\"power_w\":%.1f,\"temp_c\":%u}", k, c.sm, c.mem,
           c.thr, c.power_mw / 1000.0, c.temp);
  return b;
}
struct RecStats {
  double per_sm_clk, per_sm_clk_min, per_sm_clk_max, f_eff_ghz, cyc_per_block;
  int sms;
};
static RecStats rec_stats(const std::vector<Rec>& r, double per_block) {
  std::map<unsigned, std::pair<unsigned long long, unsigned long long>> span;
  std::map<unsigned, int> cnt;
  double cyc = 0, ns = 0;
  for (auto& x : r) {
    auto it = span.find(x.smid);
    if (it == span.end()) span[x.smid] = {x.c0, x.c1};
    else it->second.first = std::min(it->second.first, x.c0), it->second.second = std::max(it->second.second, x.c1);
    cnt[x.smid]++, cyc += double(x.c1 - x.c0), ns += double(x.t1 - x.t0);
  }
  RecStats s{0, 1e30, 0, cyc / ns, cyc / r.size(), (int)span.size()};
  for (auto& kv : span) {
    const double v = per_block * cnt[kv.first] / double(kv.second.second - kv.second.first);
    s.per_sm_clk += v, s.per_sm_clk_min = std::min(s.per_sm_clk_min, v), s.per_sm_clk_max = std::max(s.per_sm_clk_max, v);
  }
  s.per_sm_clk /= span.size();
  return s;
}
static float time_launch(const std::function<void(long long)>& launch, long long iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)), CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  launch(iters);
  CK(cudaEventRecord(b)), CK(cudaEventSynchronize(b)), CK(cudaGetLastError());
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a), cudaEventDestroy(b);
  return ms;
}
static long long calibrate(const std::function<void(long long)>& launch, long long iters, double target_ms) {
  for (;;) {
    const float ms = time_launch(launch, iters);
    if (ms >= target_ms / 4) return (long long)(iters * target_ms / ms) + 1;
    iters *= 4;
  }
}

// One timed row: calibrate to target_ms, one warm launch, one timed launch between NVML reads. `bytes` per thread per
// iteration; `single` times one thread (latency: cycles per iteration).
template <class Kern, class Go>
static void row(const char* filter, const char* group, const char* name, Kern kern, double bytes, int threads, int blocks_per_sm,
                size_t smem, Go go, const char* extra = "", double macs = 0) {
  if (filter && !strstr(name, filter) && !strstr(group, filter)) return;
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, kern));
  if (smem > 48 * 1024) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem));
  const int bps = blocks_per_sm ? std::min(blocks_per_sm, occ) : occ;
  const int blocks = threads == 1 ? 1 : g_sms * bps;
  Rec* rec;
  CK(cudaMalloc(&rec, blocks * sizeof(Rec)));
  auto launch = [&](long long it) { go(blocks, threads, smem, it, rec); };
  launch(1);
  CK(cudaDeviceSynchronize());
  const long long iters = calibrate(launch, 1, g_target_ms);
  time_launch(launch, std::max(1LL, iters / 4));
  const Clk before = read_clk();
  const float ms = time_launch(launch, iters);
  const Clk after = read_clk();
  std::vector<Rec> h(blocks);
  CK(cudaMemcpy(h.data(), rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost));
  const double per_block = bytes * threads * iters;
  const RecStats st = rec_stats(h, per_block);
  printf("{\"group\":\"%s\",\"name\":\"%s\",\"bytes_per_iter\":%.1f,\"regs\":%d,\"local_bytes\":%zu,\"smem\":%zu,\"threads\":%d,"
         "\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"iters\":%lld,\"ms\":%.4f,\"gb_s\":%.3f,\"per_sm_clk\":%.5f,\"per_sm_clk_min\":%.5f,"
         "\"per_sm_clk_max\":%.5f,\"macs_per_sm_clk\":%.2f,\"cyc_per_iter\":%.2f,\"f_eff_ghz\":%.4f,\"sms\":%d,%s,%s%s}\n",
         group, name, bytes, fa.numRegs, fa.localSizeBytes, smem, threads, bps, bps * ((threads + 31) / 32), iters, ms,
         per_block * blocks / (ms * 1e-3) / 1e9, st.per_sm_clk, st.per_sm_clk_min, st.per_sm_clk_max,
         bytes > 0 ? st.per_sm_clk * macs / bytes : rec_stats(h, macs * threads * iters).per_sm_clk, st.cyc_per_block / iters,
         st.f_eff_ghz, st.sms, clk_json("clk_before", before).c_str(), clk_json("clk_after", after).c_str(), extra);
  fflush(stdout);
  cudaFree(rec);
}

static int check_uuid(const cudaDeviceProp& prop) {
  const unsigned char* u = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  snprintf(g_uuid, sizeof g_uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3],
           u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  const char* want = getenv("POUW_EXPECT_UUID");
  printf("{\"group\":\"device\",\"name\":\"%s\",\"uuid\":\"%s\",\"sms\":%d,\"cc\":\"%d.%d\",\"clock_khz\":%d,\"smem_optin\":%zu,"
         "\"regs_per_sm\":%d,\"expect_uuid\":\"%s\"}\n",
         prop.name, g_uuid, prop.multiProcessorCount, prop.major, prop.minor, prop.clockRate, prop.sharedMemPerBlockOptin,
         prop.regsPerMultiprocessor, want ? want : "");
  if (!want || strcasecmp(want, g_uuid)) {
    fprintf(stderr, "REFUSED: the visible device is %s, not $POUW_EXPECT_UUID (%s)\n", g_uuid, want ? want : "unset");
    return 3;
  }
  if (nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByUUID(g_uuid, &g_nvml) == NVML_SUCCESS) g_nvml_ok = true;
  return 0;
}

int main(int argc, char** argv) {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  if (int rc = check_uuid(prop)) return rc;
  const char* mode = argc > 1 ? argv[1] : "";
  if (!strcmp(mode, "check")) {
    uint8_t* buf;
    CheckOut* o;
    uint64_t* dg;
    CK(cudaMalloc(&buf, 4096)), CK(cudaMalloc(&o, sizeof(CheckOut))), CK(cudaMalloc(&dg, 4096));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 32768));
    k_check<<<1, 1>>>(buf, o);
    CK(cudaDeviceSynchronize());
    k_check_quad<<<1, 32>>>(o, dg);
    CK(cudaDeviceSynchronize());
    CheckOut h;
    CK(cudaMemcpy(&h, o, sizeof h, cudaMemcpyDeviceToHost));
    print_checks(h, "device");
    return 0;
  }
  if (argc > 2) g_target_ms = atof(argv[2]);
  const char* filter = argc > 3 ? argv[3] : nullptr;
  const size_t maxT = (size_t)g_sms * 2048;
  uint64_t* out;
  CK(cudaMalloc(&out, maxT * 8));
  if (!strcmp(mode, "rates")) {
    const int nleaf = 4096;   // 16 MB of digests: L2-resident on a 128 MB L2
    uint64_t* dg;
    CK(cudaMalloc(&dg, (size_t)nleaf * ph::pearlc::LEAF_LANES * 8));
    CK(cudaMemset(dg, 0x5a, (size_t)nleaf * ph::pearlc::LEAF_LANES * 8));
    for (int single = 0; single < 2; ++single) {
      const int th = single ? 1 : 128;
      const char* g = single ? "latency" : "rate";
      row(filter, g, "p12: Keccak-p[1600, 12] (TurboSHAKE128), 168 B per permutation", k_perm<12>, 168, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<12><<<b, t>>>(it, r, out); });
      row(filter, g, "f24: Keccak-f[1600] (SHAKE128's rate), 168 B per permutation", k_perm<24>, 168, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<24><<<b, t>>>(it, r, out); });
      row(filter, g, "sha: SHA-256 compressions, chained, 64 B each", k_sha, 64, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_sha<<<b, t>>>(it, r, out); });
      row(filter, g, "b3: BLAKE3 compressions, chained, 64 B each", k_b3, 64, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_b3<<<b, t>>>(it, r, out); });
      row(filter, g, "msg: Pearl-C message digest from registers, 256 B, 2 permutations", k_msg, 256, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_msg<<<b, t>>>(it, r, out); });
      if (!single)
        row(filter, g, "quad: Pearl-C message digest from mma.sync fragments via the quad transpose, 256 B", k_quad, 256, th, 0,
            (size_t)th / 32 * 1024, [&](int b, int t, size_t s, long long it, Rec* r) { k_quad<<<b, t, s>>>(it, r, out); });
      row(filter, g, "leaf: Pearl-C tile leaf from L2, 4,096 B, 25 permutations", k_leaf, 4096, th, 0, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_leaf<<<b, t>>>(it, dg, nleaf, r, out); });
    }
    // chains in flight: one block per SM of 1 … 32 warps (Keccak-p12, SHA-256, the quad digest)
    for (int w : {1, 2, 4, 8, 12, 16, 24, 32}) {
      char ex[64];
      snprintf(ex, sizeof ex, ",\"sweep_warps\":%d", w);
      row(filter, "sweep", "p12", k_perm<12>, 168, 32 * w, 1, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_perm<12><<<b, t>>>(it, r, out); }, ex);
      row(filter, "sweep", "sha", k_sha, 64, 32 * w, 1, 0,
          [&](int b, int t, size_t, long long it, Rec* r) { k_sha<<<b, t>>>(it, r, out); }, ex);
      row(filter, "sweep", "quad", k_quad, 256, 32 * w, 1, (size_t)w * 1024,
          [&](int b, int t, size_t s, long long it, Rec* r) { k_quad<<<b, t, s>>>(it, r, out); }, ex);
    }
    return 0;
  }
  if (!strcmp(mode, "epi")) {
    // per warp 32 x 64 outputs per tile; bytes hashed per tile per thread: 2 regions x 256 B / 32 lanes x 32 = 512 B per lane
    uint64_t* dg;
    uint32_t* y;
    CK(cudaMalloc(&dg, (size_t)g_sms * 64 * 2 * 64 * 32)), CK(cudaMalloc(&y, (size_t)g_sms * 64 * 2 * 32 * 64 * 2));
    const size_t base = EPI_STAGES * (EPI_A + EPI_B) * 4;
    for (int ks : {32, 64, 128, 256}) {
      const double macs = 2048.0 * ks;   // per thread per tile: 32 x 64 x k per warp
      char ex[96], nm[64];
      for (int hash = 0; hash < 2; ++hash) {
        snprintf(ex, sizeof ex, ",\"k\":%d,\"hash\":%d", 32 * ks, hash);
        const double bytes = hash ? 512 : 0;   // 2 regions x 32 messages x 256 B per warp, per lane
        auto launch = [&](auto kern) {
          return [&, kern](int b, int t, size_t s, long long it, Rec* r) { kern<<<b, t, s>>>((int)it, ks, r, dg, y); };
        };
        snprintf(nm, sizeof nm, "epi w4 k%d %s", 32 * ks, hash ? "hash" : "nohash");
        if (hash) row(filter, "epi", nm, k_epi<1, 4>, bytes, 128, 0, base + 4 * 1024, launch(k_epi<1, 4>), ex, macs);
        else row(filter, "epi", nm, k_epi<0, 4>, bytes, 128, 0, base, launch(k_epi<0, 4>), ex, macs);
        snprintf(nm, sizeof nm, "epi w8 k%d %s", 32 * ks, hash ? "hash" : "nohash");
        if (hash) row(filter, "epi", nm, k_epi<1, 8>, bytes, 256, 0, base + 8 * 1024, launch(k_epi<1, 8>), ex, macs);
        else row(filter, "epi", nm, k_epi<0, 8>, bytes, 256, 0, base, launch(k_epi<0, 8>), ex, macs);
      }
    }
    return 0;
  }
  fprintf(stderr, "usage: pouw_hash_bench check | rates [ms] [filter] | epi [ms] [filter]\n");
  return 2;
}
#endif
