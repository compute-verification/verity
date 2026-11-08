// `interior_rows.cuh` on the CPU: the device library's twin, over host pointers, which the CPU tests hold to
// `verity.ml.windows.rows.commit_ports`.
#include "interior_rows.cuh"

using namespace introws;

extern "C" int ir_midstates_host(const uint64_t* prefixes, uint64_t* ms) {
  midstates(prefixes, ms);
  return 0;
}

extern "C" int ir_leaves_host(const Seg* s, uint32_t S, uint64_t total, const uint64_t* hm, const uint64_t* ms) {
  if (S == 0 || total == 0) return -1;
  for (uint64_t g = 0; g < total; ++g) seg_leaf(s, S, g, hm, ms);
  return 0;
}

extern "C" int ir_nodes_host(const Seg* s, uint32_t S, uint64_t total) {
  if (S == 0 || total == 0) return -1;
  for (uint64_t g = 0; g < total; ++g) seg_node(s, S, g);
  return 0;
}
