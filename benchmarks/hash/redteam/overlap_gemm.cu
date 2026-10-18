// Red-team probe of prefill-overlap's GEMM arms, built from its overlap.cu unedited (OVERLAP_CU, the file's path). For each
// prefill shape, every registered GEMM's output (and cuBLASLt's own, rerun) against cuBLASLt's, compared as bf16 bit
// patterns, so a NaN or a skipped write fails, which overlap.cu's maxrel (fmaxf drops NaN; no poison between arms) would
// pass. The output buffer is filled with 0xA5 before each call, and each call runs twice: alone, and with the claim's hash
// variant (d128_4) committing that shape's tensor on the low-priority stream beside it.
//   overlap_gemm gemm PDIR   (anything else goes to overlap.cu's own main: dump, bench, nsys)
#define main overlap_main
#include OVERLAP_CU
#undef main

namespace ov {
__global__ void bits_ne(const uint16_t* a, const uint16_t* b, size_t n, unsigned long long* bad, unsigned long long* nan) {
  unsigned long long c = 0, q = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    c += a[i] != b[i];
    q += (a[i] & 0x7F80u) == 0x7F80u;
  }
  if (c) atomicAdd(bad, c);
  if (q) atomicAdd(nan, q);
}

static int gemm_probe(const std::string& pdir) {
  World w;
  w.init(pdir);
  const Variant& hv = variants()[0];
  unsigned long long* cnt;
  CK(cudaMallocManaged(&cnt, 16));
  int fails = 0;
  for (const auto& s : SHAPES) {
    tune(w, s);
    auto gs = gemm_arms(w, s);
    const bf16* R = s.residual ? w.R[s.name] : nullptr;
    const size_t n = (size_t)M * s.n;
    auto* ref = reinterpret_cast<const uint16_t*>(w.out[0]);
    auto* got = reinterpret_cast<const uint16_t*>(w.out[1]);
    CK(cudaMemsetAsync(w.out[0], 0x5A, n * 2, w.A));
    gs[0].run(w.X[s.name], w.W[s.name], R, w.out[0], w.A);
    cnt[0] = cnt[1] = 0;
    bits_ne<<<1024, 256, 0, w.A>>>(ref, ref, n, cnt, cnt + 1);
    CK(cudaDeviceSynchronize());
    printf("{\"gemm_ref\": \"%s\", \"gemm\": \"lt\", \"words\": %zu, \"nan_or_inf\": %llu}\n", s.name.c_str(), n, cnt[1]);
    fails += cnt[1] != 0;
    for (const auto& g : gs)
      for (int beside = 0; beside < 2; ++beside) {
        CK(cudaMemsetAsync(w.out[1], 0xA5, n * 2, w.A));
        w.nev = 0;
        w.join();
        g.run(w.X[s.name], w.W[s.name], R, w.out[1], w.A);
        if (beside) w.hs[0][s.hashed]->run(hv, reinterpret_cast<const uint8_t*>(w.T[0][s.hashed]), w.B);
        w.join();
        cnt[0] = cnt[1] = 0;
        bits_ne<<<1024, 256, 0, w.A>>>(got, ref, n, cnt, cnt + 1);
        CK(cudaDeviceSynchronize());
        printf("{\"gemm_bits\": \"%s\", \"gemm\": \"%s\", \"beside\": \"%s\", \"words\": %zu, \"differing\": %llu, \"nan_or_inf\": %llu}\n",
               s.name.c_str(), g.name.c_str(), beside ? hv.name.c_str() : "none", n, cnt[0], cnt[1]);
        fails += cnt[0] != 0 || cnt[1] != 0;
      }
    fflush(stdout);
  }
  printf("{\"gemm_probe_fails\": %d}\n", fails);
  return fails ? 1 : 0;
}
}  // namespace ov

int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "gemm")) return ov::gemm_probe(argv[2]);
  return overlap_main(argc, argv);
}
