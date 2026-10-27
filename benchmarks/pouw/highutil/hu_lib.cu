// libv31.so for HIGHUTIL: V31-GPU's v31.cu (benchmarks/pouw/accel_v31, built with its V31_LIB binding) and two functions more,
// which the period's GPU driver calls instead of v31_set_b, v31_root and v31_levels. Those wait on the host (for the
// shared compute stream, or for a call's commit) and copy through pageable memory; the period changes weights every call
// (step g reads transformer block g mod NB) and keeps every call's tree, so those waits would drain the device's queue
// each step. Build: run.sh (v31lib.build_lib's flags on this file).
#include "../accel_v31/v31.cu"

#include <unordered_map>

static std::unordered_map<void*, int8_t*> g_own_b;   // each context's own weight buffer while it points elsewhere

extern "C" {
// The weights at device pointer B (n x k int8, which the caller keeps alive until the calls that read them are done)
// for the next call's form, with no copy and no wait: a kernel takes the pointer when it is launched, so the calls
// already issued keep theirs. Null: the context's own buffer again (before v31_destroy, which frees it).
int hu_point_b(void* h, const void* B) {
  Fill& f = ctx_of(h)->f;
  auto it = g_own_b.find(h);
  if (it == g_own_b.end()) it = g_own_b.emplace(h, f.B).first;
  f.B = B ? static_cast<int8_t*>(const_cast<void*>(B)) : it->second;
  if (!B) g_own_b.erase(it);
  f.wformed = false;
  return 0;
}

// Call c's tree levels ((2 width - 1) nodes, leaves first, root last, each node 8 native u64 words: v31_levels' bytes
// before its big-endian conversion) into dst, page-locked host memory the caller keeps until the copy is done
// (v31_sync), copied on the commit stream after c's commit and returning at once. Call c + 2's commit, which reuses
// the copy's buffer, runs after it in that stream's order.
int hu_levels_async(void* h, uint64_t c, void* dst) {
  Fill& f = ctx_of(h)->f;
  if (c >= f.calls || c + 2 < f.calls) return fail("call " + std::to_string(c) + " isn't one of the last two calls");
  const int b = (int)(c & 1);
  if (f.pend == b) f.flush();
  CK(cudaMemcpyAsync(dst, f.cm[b].levels, f.levels_count() * 64, cudaMemcpyDeviceToHost, f.s1));
  return 0;
}
}
