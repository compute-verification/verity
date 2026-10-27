// libv31.so for HIGHUTIL: V31-GPU's v31.cu (vendored unmodified, built with its V31_LIB binding) and one function more,
// hu_point_b, which the period's GPU driver calls instead of v31_set_b. v31_set_b waits for the shared compute stream and
// copies the weights in; the period changes weights every call (step g reads transformer block g mod NB), so that wait
// would drain the device's queue each call. Build: run.sh (v31lib.build_lib's flags on this file).
#include "v31.cu"

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
}
