// libvc's Pearl-C -h3 pieces interleaved rep by rep with the kernels they replace, at Llama-3.1-8B's decode shapes: A's row
// digests (vc_rows_launch, sha512/row-seg/v1 of FP32 rows) against hash_fast.cuh's s512_row_segs + s512_folds and hash_h3's
// s512_rows (a CTA a row); the tile leaves (vc_tiles_launch) against s512_tile_segs + s512_folds and s512_tile_leaves (a CTA a
// tile). Each arm's output is compared byte for byte with hash_fast's before anything is timed. libvc's row seeds and A's
// tree (salts, leaves from x, tree) are timed beside them for the record; nothing in hash_fast computes them.
//
//   pearl REPS      JSON lines on stdout (shapes: m in 32, 128; rows k in 4096, 14336; tiles n in 4096, 6144, 28672)
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../../../pouw/pearl_c/hash_h3.cuh"
#include "../../../pouw/pearl_c_sm120/hash_fast.cuh"
#include "../vc.h"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                             \
    }                                                                                      \
  } while (0)

namespace {

__global__ void fill(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed ^ (uint32_t)(i >> 32) * 0x85EBCA77u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    p[i] = (uint8_t)x;
  }
}
__global__ void spin(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}

unsigned cdiv(uint64_t a, uint64_t b) { return (unsigned)((a + b - 1) / b); }

// Q(n_bits) (rowleaf.sha512_row_seg_prefix): "verity/sha512-row-seg/v1\0" || u64be n_bits || u64be s || u32be 1024
std::vector<uint8_t> q_block(uint64_t n_bits) {
  std::vector<uint8_t> b(128, 0);
  const char tag[] = "verity/sha512-row-seg/v1";
  memcpy(b.data(), tag, sizeof tag);   // with its NUL: 25 bytes
  const uint64_t s = (n_bits + 8191) / 8192;
  for (int i = 0; i < 8; ++i) b[25 + i] = (uint8_t)(n_bits >> (56 - 8 * i));
  for (int i = 0; i < 8; ++i) b[33 + i] = (uint8_t)(s >> (56 - 8 * i));
  b[41] = 0, b[42] = 0, b[43] = 4, b[44] = 0;
  return b;
}

struct Plan {
  std::vector<uint8_t> p;
  void* tables = nullptr;
  explicit Plan(const vc_spec& s) {
    p.resize(vc_plan_bytes());
    if (vc_plan_build(&s, p.data())) { fprintf(stderr, "vc_plan_build\n"); exit(2); }
    CK(cudaMalloc(&tables, vc_tables_bytes(p.data()) + 16));
    if (vc_tables_launch(p.data(), tables, 0)) { fprintf(stderr, "vc_tables_launch\n"); exit(2); }
  }
};

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  unsigned sm() { unsigned c = 0; if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c); return c; }
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Arm {
  std::string name;
  std::function<void(cudaStream_t)> run;
  const uint8_t* out;   // what it writes, compared with the first arm's; nullptr: timed only
  uint64_t out_bytes;
};

bool all_ok = true;

// each arm's output against arms[0]'s, then every arm once a rep, in turn, after a spin that covers the enqueue
void interleave(const char* piece, const char* shape, std::vector<Arm>& arms, int reps, cudaStream_t st, Nv& nv) {
  std::vector<uint8_t> want, got;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (!arms[i].out) continue;
    CK(cudaMemsetAsync((void*)arms[i].out, 0xA5, arms[i].out_bytes, st));
    arms[i].run(st);
    arms[i].run(st);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    std::vector<uint8_t>& v = i ? got : want;
    v.resize(arms[i].out_bytes);
    CK(cudaMemcpy(v.data(), arms[i].out, v.size(), cudaMemcpyDeviceToHost));
    if (i) {
      const bool same = got == want;
      printf("{\"piece\": \"%s\", \"shape\": \"%s\", \"same_bytes\": \"%s\", \"against\": \"%s\", \"bytes\": %llu, \"ok\": %s}\n", piece,
             shape, arms[i].name.c_str(), arms[0].name.c_str(), (unsigned long long)v.size(), same ? "true" : "false");
      if (!same) { all_ok = false; arms[i].name += "!"; }
    }
  }
  arms.erase(std::remove_if(arms.begin(), arms.end(), [](const Arm& a) { return a.name.back() == '!'; }), arms.end());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto timed = [&](Arm& a, unsigned& mhz) {
    spin<<<1, 1, 0, st>>>(200000);
    CK(cudaEventRecord(e0, st));
    a.run(st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    mhz = nv.sm();
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    return 1e3 * t;
  };
  for (auto& a : arms) { unsigned m; timed(a, m); timed(a, m); }
  std::vector<std::vector<double>> t(arms.size()), mz(arms.size());
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < arms.size(); ++i) {
      unsigned m;
      t[i].push_back(timed(arms[i], m));
      mz[i].push_back(m);
    }
  for (size_t i = 0; i < arms.size(); ++i) {
    printf("{\"piece\": \"%s\", \"shape\": \"%s\", \"arm\": \"%s\", \"us\": %.2f, \"mhz\": %.0f, \"us_all\": [", piece, shape,
           arms[i].name.c_str(), median(t[i]), median(mz[i]));
    for (size_t r = 0; r < t[i].size(); ++r) printf("%s%.2f", r ? ", " : "", t[i][r]);
    printf("], \"mhz_all\": [");
    for (size_t r = 0; r < mz[i].size(); ++r) printf("%s%.0f", r ? ", " : "", mz[i][r]);
    printf("]}\n");
  }
  fflush(stdout);
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || strcmp(argv[1], "pearl")) { fprintf(stderr, "pearl REPS\n"); return 2; }
  const int reps = atoi(argv[2]);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("{\"device\": \"%s\", \"sms\": %d}\n", prop.name, prop.multiProcessorCount);
  if (vc_init()) { fprintf(stderr, "vc_init\n"); return 2; }
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Nv nv;
  const Args none{};
  s512_heads<<<cdiv(RS_HEAD_SEGS + RS_MAX_SEGS + 1, 128), 128>>>(none);
  CK(cudaDeviceSynchronize());
  uint8_t did[64];
  for (int i = 0; i < 64; ++i) did[i] = (uint8_t)(i * 37 + 1);
  const size_t big = 128ull * 28672 * 4;
  // the largest shapes' segment digests (64 bytes a 1 KiB segment: m 128 x k 14336, and m 128 x n 28672) and units
  const size_t seg_bytes = 64ull * 128 * 28672 * 2 * 4 / 1024, out_bytes = 64ull * 2 * 448;
  auto fits = [&](const char* what, uint64_t need, size_t have) {
    if (need > have) { fprintf(stderr, "%s needs %llu bytes, has %zu\n", what, (unsigned long long)need, have); exit(2); }
  };
  uint8_t *A, *Cm, *U, *seg, *out_f, *out_r, *out_v, *scratch;
  CK(cudaMalloc(&A, 128ull * 14336 * 4 + 64));
  CK(cudaMalloc(&Cm, big + 64));
  CK(cudaMalloc(&U, big + 64));
  CK(cudaMalloc(&seg, seg_bytes));
  CK(cudaMalloc(&out_f, out_bytes));
  CK(cudaMalloc(&out_r, out_bytes));
  CK(cudaMalloc(&out_v, out_bytes));
  CK(cudaMalloc(&scratch, seg_bytes));
  fill<<<1024, 256>>>(A, 128ull * 14336 * 4 + 64, 3u);
  fill<<<1024, 256>>>(Cm, big + 64, 5u);
  fill<<<1024, 256>>>(U, big + 64, 9u);
  CK(cudaDeviceSynchronize());
  for (int m : {32, 128})
    for (int k : {4096, 14336}) {
      const int s = k / 256;
      const auto q = q_block(32ull * k);
      vc_spec sp{};
      sp.domain_id = did;
      sp.row_scheme = VC_ROW_SEG;
      sp.row_bits = 32ull * k;
      sp.row_prefix = q.data();
      Plan pl(sp);
      fits("s512_row_segs", 64ull * m * s, seg_bytes);
      fits("vc_rows_launch", vc_rows_scratch_bytes(pl.p.data(), m), seg_bytes);
      Args af{}, ao{}, ar{};
      af.p[0] = (uint64_t)A, af.p[1] = (uint64_t)seg, af.i[0] = m, af.i[1] = k;
      ao.p[0] = (uint64_t)seg, ao.p[1] = (uint64_t)out_f, ao.i[0] = m, ao.i[1] = s, ao.i[2] = m, ao.i[3] = 0;
      ar.p[0] = (uint64_t)A, ar.p[1] = (uint64_t)out_r, ar.i[0] = m, ar.i[1] = k;
      std::vector<Arm> arms = {
          {"hash_fast", [=](cudaStream_t x) {
             s512_row_segs<<<cdiv((uint64_t)m * s, RS_FAST_NT), RS_FAST_NT, 0, x>>>(af);
             s512_folds<<<cdiv(m, RS_FOLD_NT), RS_FOLD_NT, 0, x>>>(ao);
           }, out_f, 64ull * m},
          {"hash_h3_ref", [=](cudaStream_t x) { s512_rows<<<m, 128, 0, x>>>(ar); }, out_r, 64ull * m},
          {"libvc", [&, m, k](cudaStream_t x) {
             if (vc_rows_launch(pl.p.data(), pl.tables, A, 4ull * k, m, out_v, scratch, x)) { fprintf(stderr, "vc_rows_launch\n"); exit(3); }
           }, out_v, 64ull * m},
      };
      const std::string shape = "m" + std::to_string(m) + "_k" + std::to_string(k);
      interleave("row_digests", shape.c_str(), arms, reps, st, nv);
    }
  for (int m : {32, 128})
    for (int n : {4096, 6144, 28672}) {
      const int tn = n / 64, full = m / 64 * tn, sl = m % 64 / 2, tiles = full + (sl ? tn : 0);
      const auto qf = q_block(8ull * 64 * 512), qb = q_block(8ull * (m % 64) * 512);
      vc_spec sp{};
      sp.domain_id = did;
      sp.row_scheme = VC_ROW_TILE;
      sp.tile_rows = (uint32_t)m;
      sp.tile_cols = (uint32_t)n;
      sp.row_prefix = qf.data();
      sp.row_prefix_b = sl ? qb.data() : nullptr;
      Plan pl(sp);
      Args af{}, ao{}, ar{};
      af.p[0] = (uint64_t)Cm, af.p[1] = (uint64_t)U, af.p[2] = (uint64_t)seg, af.i[0] = m, af.i[1] = n;
      ao.p[0] = (uint64_t)seg, ao.p[1] = (uint64_t)out_f, ao.i[0] = tiles, ao.i[1] = 32, ao.i[2] = full, ao.i[3] = sl;
      ar.p[0] = (uint64_t)Cm, ar.p[1] = (uint64_t)U, ar.p[2] = (uint64_t)out_r, ar.i[0] = m, ar.i[1] = n;
      const uint64_t nseg = (uint64_t)full * 32 + (uint64_t)tn * sl;
      if (vc_tiles_count(pl.p.data()) != (uint64_t)tiles) { fprintf(stderr, "vc_tiles_count\n"); return 2; }
      fits("s512_tile_segs", 64 * nseg, seg_bytes);
      fits("vc_tiles_launch", vc_tiles_scratch_bytes(pl.p.data()), seg_bytes);
      fits("tile leaves", 64ull * tiles, out_bytes);
      std::vector<Arm> arms = {
          {"hash_fast", [=](cudaStream_t x) {
             s512_tile_segs<<<cdiv(nseg, RS_FAST_NT), RS_FAST_NT, 0, x>>>(af);
             s512_folds<<<cdiv(tiles, RS_FOLD_NT), RS_FOLD_NT, 0, x>>>(ao);
           }, out_f, 64ull * tiles},
          {"hash_h3_ref", [=](cudaStream_t x) { s512_tile_leaves<<<tiles, 32, 0, x>>>(ar); }, out_r, 64ull * tiles},
          {"libvc", [&](cudaStream_t x) {
             if (vc_tiles_launch(pl.p.data(), pl.tables, Cm, U, out_v, scratch, x)) { fprintf(stderr, "vc_tiles_launch\n"); exit(3); }
           }, out_v, 64ull * tiles},
      };
      const std::string shape = "m" + std::to_string(m) + "_n" + std::to_string(n);
      interleave("tile_leaves", shape.c_str(), arms, reps, st, nv);
    }
  // timed only: row seeds and A's tree (salts, leaves from the row digests, the tree) over m rows, ChaCha20 salts
  for (int m : {32, 128}) {
    const int k = 4096;
    const auto q = q_block(32ull * k);
    std::vector<uint8_t> label(128, 0), key32(32, 7), sk(32, 11);
    vc_spec sp{};
    sp.domain_id = did;
    sp.row_scheme = VC_ROW_SEG;
    sp.row_bits = 32ull * k;
    sp.row_prefix = q.data();
    sp.seed_label = label.data();
    Plan seeds(sp);
    uint8_t *salt, *rootb, *x, *seeds_out, *lv, *rec;
    CK(cudaMalloc(&salt, 64));
    CK(cudaMalloc(&rootb, 64));
    CK(cudaMalloc(&x, 64ull * m));
    CK(cudaMalloc(&seeds_out, 32ull * m));
    CK(cudaMalloc(&lv, vc_tree_bytes(m)));
    CK(cudaMalloc(&rec, vc_salt_scratch_bytes(vc_tree_width(m))));
    CK(cudaMemset(salt, 1, 64));
    CK(cudaMemset(rootb, 2, 64));
    if (vc_rows_launch(seeds.p.data(), seeds.tables, A, 4ull * k, m, x, scratch, st)) return 3;
    CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> salt_prefix(128, 0), leaf_prefix(128, 0), hkey(256, 0);
    for (int i = 0; i < 256; ++i) hkey[i] = (uint8_t)(i * 151 + 7);
    salt_prefix[0] = 'S', leaf_prefix[0] = 'L';
    static const char schema[] = "hm96-sha512/row-seg/v1";
    vc_spec hs{};
    hs.domain_id = did;
    hs.schema = schema;
    hs.schema_len = 22;
    hs.salt_prefix = salt_prefix.data();
    hs.leaf_prefix = leaf_prefix.data();
    hs.key = hkey.data();
    hs.salt_source = VC_SALT_CHACHA64;
    hs.salt_key = sk.data();
    hs.salt_nonce = 5;
    Plan tree(hs);
    const uint64_t w = vc_tree_width(m);
    std::vector<Arm> arms = {
        {"libvc_row_seeds", [&](cudaStream_t s) {
           if (vc_row_seeds_launch(seeds.p.data(), x, m, salt, rootb, 3, seeds_out, s)) exit(3);
         }, nullptr, 0},
        {"libvc_a_tree", [&](cudaStream_t s) {
           if (vc_salts_launch(tree.p.data(), tree.tables, nullptr, 0, w, m, rec, lv, s) ||
               vc_leaves_x_launch(tree.p.data(), x, 0, m, m, rec, lv, nullptr, nullptr, s) || vc_tree_launch(tree.p.data(), lv, m, s))
             exit(3);
         }, nullptr, 0},
        {"libvc_a_tree_fused", [&](cudaStream_t s) {
           if (vc_leaves_fused_launch(tree.p.data(), tree.tables, nullptr, 0, x, nullptr, 0, w, m, lv, nullptr, nullptr, s) ||
               vc_tree_launch(tree.p.data(), lv, m, s))
             exit(3);
         }, nullptr, 0},
    };
    const std::string shape = "m" + std::to_string(m);
    interleave("a_side", shape.c_str(), arms, reps, st, nv);
  }
  return all_ok ? 0 : 1;
}
