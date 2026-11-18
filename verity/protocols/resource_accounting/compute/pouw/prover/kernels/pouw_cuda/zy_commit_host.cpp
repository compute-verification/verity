// `zy_commit.cuh` on the CPU, one record and one node at a time: the twin `pouw_h1_registered_device.host_zy` drives, so the
// CPU tests hold the device's functions to the call table's and the registered value's own twins (`row_table_host.cpp`,
// `registered_rows_host.cpp`) on a machine without a GPU.
#include <vector>

#include "zy_commit.cuh"

using namespace zyc;

extern "C" uint64_t zy_args_bytes() { return sizeof(Args); }

// The ``a->n`` records at ``rows`` (n x R bytes) under the call table's salts at ``call_salts`` and the value's at
// ``reg_salts`` (each n x 192 bytes, record u's at byte 192 u): the call table's tree into ``call`` ((2 W - 1) x 8 words),
// the value's b || c into ``bcs`` (n x 16) and its tree into ``levels`` ((2 W - 1) x 8), its pads and nodes
// `registered_rows.cuh`'s under ``t`` (`rr_tree`'s, on the device); 0, or 1 for a malformed layout or no salts.
extern "C" int zy_commit_host(const Args* a, const regrows::Table* t, const uint8_t* rows, const uint8_t* call_salts,
                              const uint8_t* reg_salts, u64* call, u64* bcs, u64* levels) {
  std::vector<Plan> p(1);
  if (plan(*a, p[0]) || t->n != a->n || t->W != rowtab::width(a->n) || t->depth != rowtab::depth(t->W) || !call_salts ||
      !reg_salts)
    return 1;
  std::vector<uint8_t> tab(rc::TN_BYTES);
  for (u32 i = 0; i < 512; ++i) rc::nib_entry(a->hm_key, i, tab.data());
  const rc::NibTab<false> T{tab.data()};
  for (u32 u = 0; u < a->n; ++u)
    record(p[0], T, u, rows + (size_t)a->R * u, call_salts + (size_t)SALT * u, reg_salts + (size_t)SALT * u,
           call + 8 * (size_t)u, bcs + 16 * (size_t)u, levels + 8 * (size_t)u);
  for (u32 r = a->n; r < t->W; ++r) {
    call_pad(p[0], r, call + 8 * (size_t)r);
    regrows::pad(*t, r, levels + 8 * (size_t)r);
  }
  for (u32 l = 1; l <= t->depth; ++l)
    for (u32 j = 0; j < (t->W >> l); ++j) {
      call_node(p[0], t->W, l, j, call);
      regrows::node(*t, l, j, levels);
    }
  return 0;
}
