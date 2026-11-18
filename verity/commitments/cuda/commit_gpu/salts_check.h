// What Committer::salts (commit_gpu.cuh) refuses on the host before a salt kernel is handed its buffer's address. The
// kernels read 192 bytes a leaf by 16-byte loads from the address they are given and cannot tell where the buffer ends,
// so its owner says its length: n leaves take 192 n bytes, and on the small path each of the ns slots salted in the
// launch takes its own n after the last slot's, slots s0 .. s0 + ns - 1 of the committer's. Plain C++, so a CPU test
// includes it.
#pragma once

#include <cstdint>
#include <cstdio>

namespace commit_gpu {

// 0 where the kernels may read the buffer; else 1 with the refusal in why (cap bytes).
inline int salts_refused(uint64_t addr, uint64_t bytes, uint64_t n, bool small, int slots, int s0, int ns, char* why,
                         size_t cap) {
  if (addr == 0 || addr % 16) {
    snprintf(why, cap, "salts at %#llx: a device buffer at a multiple of 16 bytes (a salt is read by 16-byte loads)",
             (unsigned long long)addr);
    return 1;
  }
  if (small ? (s0 < 0 || ns < 1 || s0 > slots - ns) : (s0 != 0 || ns != 1)) {
    snprintf(why, cap, "salt slots %d .. %d of %d", s0, s0 + ns - 1, small ? slots : 1);
    return 1;
  }
  if (bytes != 192 * n * (uint64_t)ns) {
    snprintf(why, cap, "salts: %llu bytes for %llu leaves in %d slot%s, which take %llu (192 a leaf)",
             (unsigned long long)bytes, (unsigned long long)n, ns, ns == 1 ? "" : "s",
             (unsigned long long)(192 * n * (uint64_t)ns));
    return 1;
  }
  return 0;
}

}  // namespace commit_gpu
