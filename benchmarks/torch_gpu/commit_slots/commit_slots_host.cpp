// The host twin of commit_slots.cu: the same thread functions (commit_slots.cuh), one loop a kernel, for the CPU tests that
// hold them to slots.py. Built by slots_gpu.host_lib with the system C++ compiler.
#include "commit_slots.cuh"

using namespace cs;

extern "C" {

void cs_midstate(const uint8_t* block, uint64_t* out) { midstate(block, out); }

int cs_tail(const uint64_t* did, uint64_t n, int depth, uint32_t* lv, uint64_t* nodes) { return tail(did, n, depth, lv, nodes); }

int cs_rows(void*, const Ctx* c, const Step* st, const Grp* g) {
  for (u32 j = 0; j < g->nrows; ++j)
    for (u32 s = 0; s < st->S; ++s) row_thread(*c, *st, *g, j, s, c->T, 256);
  return 0;
}

int cs_units(void*, const Ctx* c, const Step* st, const Grp* g) {
  for (u32 i = 0; i < g->nunits; ++i)
    for (u32 s = 0; s < st->S; ++s) unit_thread(*c, *st, *g, i, s);
  return 0;
}

int cs_fold(void*, const Ctx* c, const Step* st, const Grp* g) {
  for (u32 s = 0; s < st->S; ++s) fold_thread(*c, *st, *g, s);
  return 0;
}

int cs_finish(void*, const Ctx* c, const Step* st) {
  for (u32 s = 0; s < st->S; ++s) finish_thread(*c, *st, s);
  for (u32 i = st->S; i < st->W; ++i) step_pad_thread(*st, i);
  u64 *in = st->lv_a, *out = st->lv_b;
  for (u32 l = 0; l < st->depth; ++l) {
    for (u32 i = 0; i < (st->W >> (l + 1)); ++i) level_thread(*st, in, out, (int)l, i);
    u64* t = in; in = out; out = t;
  }
  memcpy(st->root, in, 64);
  return 0;
}

}
