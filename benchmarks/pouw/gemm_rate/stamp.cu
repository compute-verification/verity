// stamp_k: each SM's %globaltimer, %clock64 and %smid at one moment, for torch_rate.py's windows. One CTA an SM: the launch
// asks for more than half an SM's shared memory, which keeps a second CTA off it, and every CTA stamps only once all have
// arrived; sm ~0 where they weren't all resident within a millisecond (summarize.span refuses the window). out: 3 words a CTA.
extern "C" __global__ void stamp_k(unsigned long long* out, unsigned* arrive) {
  if (threadIdx.x) return;
  auto gt = [] {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  };
  atomicAdd(arrive, 1u);
  const unsigned long long t0 = gt();
  bool all = true;
  while (*(volatile unsigned*)arrive < gridDim.x)
    if (gt() - t0 > 1000000) {
      all = false;
      break;
    }
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  out[3 * blockIdx.x] = gt();
  out[3 * blockIdx.x + 1] = (unsigned long long)clock64();
  out[3 * blockIdx.x + 2] = all ? sm : ~0ull;
}
