// receipts.cu's calls on host memory, one loop, built by sketch.py with the system C++ compiler: what the CPU tests hold
// to the reference. The stream argument is ignored.
#include <cstring>

#include "receipts.cuh"

extern "C" {

void p2r_midstate(const uint8_t* block, uint64_t* out) { p2r::midstate(block, out); }

int p2r_receipts(void*, const uint64_t* mid, const uint64_t* sigma, uint64_t g0, uint64_t n, uint64_t* y) {
  for (uint64_t t = 0; t < n; ++t) {
    uint64_t d[8];
    p2r::receipt<0, 1>(mid, g0 + t, sigma + 8 * t, d);
    for (int i = 0; i < 8; ++i) y[8 * t + i] = sx::bswap64(d[i]);
  }
  return 0;
}

}
