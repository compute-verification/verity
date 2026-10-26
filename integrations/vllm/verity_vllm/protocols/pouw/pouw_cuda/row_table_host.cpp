// `row_table.cuh` on the CPU, one row and one node at a time: the twin `row_tables_device.host_step` drives, so the CPU tests hold
// the device's functions to `commit.row_tables.step_roots` on a machine without a GPU.
#include <vector>

#include "row_table.cuh"

using namespace rowtab;

// The frame tree over ``n`` leaves (``lv``, 8 words each) under ``did``, padded to a power of two: its root.
static void tree(const uint64_t did[8], uint32_t n, const uint64_t* lv, uint64_t root[8]) {
  const uint32_t W = width(n);
  std::vector<uint64_t> cur(8 * (size_t)W);
  for (uint32_t r = 0; r < W; ++r) {
    if (r < n) {
      for (int i = 0; i < 8; ++i) cur[8 * (size_t)r + i] = lv[8 * (size_t)r + i];
    } else {
      frame_pad(did, r, &cur[8 * (size_t)r]);
    }
  }
  for (uint32_t l = 0; (W >> l) > 1; ++l) {
    for (uint32_t j = 0; j < (W >> (l + 1)); ++j) {
      uint64_t o[8];
      frame_node(did, l, j, &cur[16 * (size_t)j], &cur[16 * (size_t)j + 8], o);
      for (int i = 0; i < 8; ++i) cur[8 * (size_t)j + i] = o[i];
    }
  }
  for (int i = 0; i < 8; ++i) root[i] = cur[i];
}

// The step's root then its tables' roots into ``out`` ((T + 1) x 8 words), from host copies of the codes and scales; 0, or 1 for
// a malformed table.
extern "C" int rt_step_host(const Tab* tabs, const uint32_t* perm, uint64_t S, uint32_t N, const uint64_t* hm,
                            const uint64_t* prefixes, const Args* a, uint64_t* out) {
  if (a->T == 0) return 1;
  for (uint32_t i = 0; i < a->T; ++i)
    if (tabs[i].n == 0 || tabs[i].K % 2 || tabs[i].s != segments(tabs[i].K) || tabs[i].W != width(tabs[i].n)) return 1;
  uint64_t ms[16];
  midstates(prefixes, ms);
  std::vector<uint64_t> dig(8 * (size_t)S), leaves(8 * (size_t)N), roots(8 * (size_t)a->T);
  for (uint64_t i = 0; i < S; ++i) step_segment(tabs, a->T, perm, i, &dig[8 * i]);
  for (uint32_t g = 0; g < N; ++g) step_row(tabs, *a, g, dig.data(), hm, ms, &leaves[8 * (size_t)g]);
  for (uint32_t i = 0; i < a->T; ++i) {
    tree(tabs[i].did, tabs[i].n, &leaves[8 * (size_t)tabs[i].row0], out + 8 * (1 + (size_t)i));
    frame_leaf(a->did, i, ROWTAB_ROOTS_SCHEMA, out + 8 * (1 + (size_t)i), &roots[8 * (size_t)i]);
  }
  tree(a->did, a->T, roots.data(), out);
  return 0;
}
