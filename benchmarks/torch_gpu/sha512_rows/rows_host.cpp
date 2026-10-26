// The kernels' host twin (rows.cu's extern "C" calls, one loop a kernel, on host memory), built by engine.host_lib with the
// system C++ compiler: what the CPU tests hold to the reference. The stream and variant arguments are ignored.
#include "rows.cuh"

using namespace sxr;

namespace {
struct D {
  static void run(u64* st, u64* w) { sx::compress<0, 1>(st, w); }
};
}  // namespace

extern "C" {

int sxr_variants() { return 1; }
const char* sxr_variant_name(int i) { return i == 0 ? "host" : nullptr; }
const char* sxr_default() { return "host"; }

void sxr_midstate(const uint8_t* block, uint64_t* out) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int k = 0; k < 8; ++k) v = v << 8 | block[8 * i + k];
    w[i] = v;
  }
  sx::iv(out);
  D::run(out, w);
}

int sxr_row_setup(void*, RowSpec* rows, uint32_t G) {
  for (u32 g = 0; g < G; ++g) row_setup<D>(rows[g]);
  return 0;
}

int sxr_rows(void*, int, const RowSpec* rows, const uint32_t* order, uint32_t G, uint32_t S, uint64_t* x) {
  for (u64 t = 0; t < (u64)G * S; ++t) {
    const u64 j = t / S, s = t % S, g = order ? order[j] : j;
    row_digest<D>(rows[g], s, x + 8 * (g * S + s));
  }
  return 0;
}

int sxr_leaf_setup(void*, LeafRow* rows, uint32_t G, const uint64_t* dids, const uint8_t* schema, uint32_t schema_len) {
  for (u32 g = 0; g < G; ++g) leaf_setup<D>(rows[g], dids, schema, schema_len);
  return 0;
}

int sxr_leaves(void*, const LeafArgs* a) {
  for (u64 t = 0; t < (u64)a->G * a->S; ++t) leaf<D>(*a, a->T, 256, (u32)(t / a->S), (u32)(t % a->S));
  return 0;
}

int sxr_frame_leaves(void*, const uint64_t* dids, uint32_t D_, const uint64_t* values, uint32_t T, uint32_t n, uint64_t rank0,
                     const uint8_t* schema, uint32_t schema_len, uint64_t* out) {
  for (u64 t = 0; t < (u64)T * n; ++t)
    frame_leaf<D>(dids + 8 * (D_ == 1 ? 0 : t / n), rank0 + t % n, schema, schema_len, values + 8 * t, out + 8 * t);
  return 0;
}

int sxr_node_mids(void*, const uint64_t* dids, uint32_t D_, uint32_t level, uint64_t* mids) {
  for (u32 d = 0; d < D_; ++d) node_mid<D>(dids + 8 * (u64)d, level, mids + 8 * (u64)d);
  return 0;
}

int sxr_level(void*, const uint64_t* mids, uint32_t D_, const uint64_t* in, uint32_t T, uint32_t w, uint64_t* out) {
  for (u64 t = 0; t < (u64)T * w; ++t) {
    const u64 tree = t / w, i = t % w;
    const u64* l = in + 8 * (2 * tree * w + 2 * i);
    node<D>(mids + 8 * (D_ == 1 ? 0 : tree), i, l, l + 8, out + 8 * t);
  }
  return 0;
}

int sxr_pads(void*, const uint64_t* dids, uint32_t D_, uint32_t T, uint64_t first, uint32_t n, uint64_t* out) {
  for (u64 t = 0; t < (u64)T * n; ++t) pad<D>(dids + 8 * (D_ == 1 ? 0 : t / n), first + t % n, out + 8 * t);
  return 0;
}

}
