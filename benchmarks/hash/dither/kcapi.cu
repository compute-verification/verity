// cgemm.cu's arms (c_fx1, cquant, cwedit, cedited:c_fx1, egemm's plain cuBLASLt, ...) and its device gate behind a C API, so
// kbench.py can interleave them rep by rep with vLLM's cutlass_scaled_mm and torch's cuBLASLt in one process, on one stream.
// cgemm.cu is compute accounting's (benchmarks/pouw/edited_gemm), included unchanged.
//   nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared -o libkcapi.so \
//     benchmarks/hash/dither/kcapi.cu -lcuda -lcublasLt -lnvidia-ml
#define CGEMM_LIB
#include "../../pouw/edited_gemm/cgemm.cu"
#include <map>

static void kc_init() {
  static bool done = false;
  if (done) return;
  CL(cublasLtCreate(&g_lt));
  CK(cudaMalloc(&g_ws, g_wsz));
  done = true;
}

extern "C" {
int kc_gate(const char* dir, int m, int n, int k, unsigned seed) { kc_init(); return cgate(dir, m, n, k, seed); }
void* kc_setup(int m, int n, int k, unsigned seed) {
  kc_init();
  CP* C = new CP;
  csetup(*C, m, n, k, seed, false);
  return C;
}
void* kc_stream(void* c) { return (void*)static_cast<CP*>(c)->E.st; }
// every setup's arms on one stream (kbench.py's, so CUDA graphs capture them); kc_free destroys the setup's own stream only
static std::map<void*, cudaStream_t> g_own;
void kc_set_stream(void* c, void* s) {
  CP* C = static_cast<CP*>(c);
  if (!g_own.count(c)) g_own[c] = C->E.st;
  C->E.st = (cudaStream_t)s;
}
int kc_grid(void* c) { return static_cast<CP*>(c)->E.grid; }
int kc_lt_algo(void* c) { return static_cast<CP*>(c)->E.plain.algo_id; }
void* kc_arm(const char* name) { return new std::function<void(CP&)>(carm(name)); }
void kc_run(void* c, void* arm, int calls) {
  auto& f = *static_cast<std::function<void(CP&)>*>(arm);
  for (int i = 0; i < calls; i++) f(*static_cast<CP*>(c));
}
void kc_free(void* c) {
  if (g_own.count(c)) {
    static_cast<CP*>(c)->E.st = g_own[c];
    g_own.erase(c);
  }
  cfree(*static_cast<CP*>(c));
  delete static_cast<CP*>(c);
}
unsigned kc_sm_mhz() {
  static Nv nv;
  return nv.sm();
}
int kc_last_error() { return (int)cudaGetLastError(); }
}
