// `registered_rows.cuh` on the CPU, one row and one node at a time: the twin `pouw_h1_registered_device.host_commit` drives,
// so the CPU tests hold the device's functions to `pouw_h1_registered.commit` on a machine without a GPU.
#include "registered_rows.cuh"

using namespace regrows;

// The value ``t`` (its ``rows`` every row's bytes, its ``salts`` the given salts or 0): its b || c into ``bcs`` (n x 16 words)
// and its tree into ``levels`` ((2 W - 1) x 8 words), from host memory; 0, or 1 for a malformed table.
extern "C" int rr_commit_host(const Table* t, const uint64_t* hm, const uint64_t* prefixes, uint64_t* bcs, uint64_t* levels) {
  if (t->n == 0 || t->W != width(t->n) || t->depth != depth(t->W) || t->row_bytes == 0) return 1;
  uint64_t ms[16];
  midstates(prefixes, ms);
  for (uint32_t j = 0; j < t->n; ++j)
    row_leaf(*t, j, (const uint8_t*)t->rows + (size_t)t->row_bytes * j, hm, ms, bcs + 16 * (size_t)j, levels + 8 * (size_t)j);
  for (uint32_t r = t->n; r < t->W; ++r) pad(*t, r, levels + 8 * (size_t)r);
  for (uint32_t l = 1; l <= t->depth; ++l)
    for (uint32_t j = 0; j < (t->W >> l); ++j) node(*t, l, j, levels);
  return 0;
}

// The ``p->n`` public salts from position ``p->first`` into ``out`` (n x 24 words); 0, or 1 for a malformed head.
extern "C" int rr_public_salts_host(const Pub* p, uint64_t* out) {
  if (p->n == 0 || p->len > 8 * PUB_WORDS) return 1;
  for (uint32_t i = 0; i < p->n; ++i) public_salt(*p, p->first + i, out + 24 * (size_t)i);
  return 0;
}
