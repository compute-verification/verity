// The fill kernel's host twin (stream.cu's sxs_fill, one loop, on host memory), built by engine.host_lib with the system C++
// compiler: what the CPU tests hold to `Key.block`. The stream argument is ignored. It counts its compressions
// (sxs_compressions).
#include <cstdint>

namespace {
uint64_t compressions;
}  // namespace

#include "stream.cuh"

using namespace sxs;

namespace {
struct D {
  static void compress(u64* st, u64* w) {
    ++compressions;
    sx::compress<0, 1>(st, w);
  }
};
}  // namespace

extern "C" {

uint64_t sxs_compressions(int reset) {
  const uint64_t n = compressions;
  if (reset) compressions = 0;
  return n;
}

int sxs_fill(void*, const uint8_t* head, uint32_t len, uint64_t first, uint64_t leaves, uint8_t* out) {
  Head h;
  if (head_of(head, len, h)) return REFUSED_HEAD;
  if (!range_ok(first, leaves)) return REFUSED_RANGE;
  for (u64 t = 0; t < LEAF_BLOCKS * leaves; ++t) {
    u64 st[8];
    block<D>(h, LEAF_BLOCKS * first + t, st);
    for (int k = 0; k < 64; ++k) out[64 * t + k] = (uint8_t)(st[k / 8] >> (56 - 8 * (k % 8)));
  }
  return 0;
}

}  // extern "C"
