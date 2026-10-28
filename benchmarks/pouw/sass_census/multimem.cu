// What multimem does on a card with no NVLink (the RTX PRO 6000), for the coverage table's sm_120 LDGMC row
// (machine_model/data/cover/rows.toml, CMP-166). It prints the device's multicast attribute, what cuMulticastCreate
// answers, and what multimem.ld_reduce (LDGMC) and multimem.red do on ordinary cudaMalloc memory, each in a child
// process of its own (CUDA is initialized after the fork), so a fault ends only that child. The hypothesis: the card
// supports no multicast object, so no address is a multimem one; on a unicast address LDGMC faults or returns the one
// copy's word, and multimem.red faults or adds once: a load or an L2 reduction, which `move` and `atom` dominate.
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

constexpr int N = 32;
constexpr float STORED = 3.0f;

__global__ void ld_reduce(const float* p, float* out) {
  float v;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(v) : "l"(p + threadIdx.x) : "memory");
  out[threadIdx.x] = v;
}

__global__ void red(float* p) {
  asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;" ::"l"(p + threadIdx.x), "f"(1.0f) : "memory");
}

static int child(bool reduce_load) {
  float *p, *out, h[N];
  for (float& x : h) x = STORED;
  if (cudaMalloc(&p, N * sizeof(float)) != cudaSuccess || cudaMalloc(&out, N * sizeof(float)) != cudaSuccess) return 3;
  cudaMemcpy(p, h, sizeof h, cudaMemcpyHostToDevice);
  cudaMemset(out, 0, sizeof h);
  if (reduce_load) ld_reduce<<<1, N>>>(p, out); else red<<<1, N>>>(p);
  const cudaError_t e = cudaDeviceSynchronize();
  const char* what = reduce_load ? "multimem.ld_reduce.add.f32 on unicast memory" : "multimem.red.add.f32 on unicast memory";
  if (e != cudaSuccess) {
    printf("%s: fault (%s: %s)\n", what, cudaGetErrorName(e), cudaGetErrorString(e));
    return 1;
  }
  cudaMemcpy(h, reduce_load ? out : p, sizeof h, cudaMemcpyDeviceToHost);
  int same = 0;
  for (float x : h) same += x == h[0];
  printf("%s: returned, %s[0] = %g (stored %g; %d of %d lanes equal)\n", what, reduce_load ? "result" : "memory", h[0],
         STORED, same, N);
  return reduce_load ? (h[0] == STORED && same == N ? 0 : 2) : (h[0] == STORED + 1 && same == N ? 0 : 2);
}

static int run_child(bool reduce_load) {
  fflush(stdout);
  const pid_t pid = fork();
  if (pid == 0) {
    const int rc = child(reduce_load);
    fflush(stdout);
    _exit(rc);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

int main() {
  const int ld = run_child(true), rd = run_child(false);
  if (cuInit(0) != CUDA_SUCCESS) { printf("cuInit failed\n"); return 3; }
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  char name[256];
  cuDeviceGetName(name, sizeof name, dev);
  int mc = -1;
  cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);
  CUmulticastObjectProp prop;
  memset(&prop, 0, sizeof prop);
  prop.numDevices = 1;
  prop.size = 2u << 20;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t gran = 0;
  const CUresult g = cuMulticastGetGranularity(&gran, &prop, CU_MULTICAST_GRANULARITY_MINIMUM);
  CUmemGenericAllocationHandle h = 0;
  const CUresult c = cuMulticastCreate(&h, &prop);
  const char *gs = "?", *cs = "?";
  cuGetErrorName(g, &gs);
  cuGetErrorName(c, &cs);
  printf("device %s: CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = %d; cuMulticastGetGranularity: %s; cuMulticastCreate: %s\n",
         name, mc, gs, cs);
  if (c == CUDA_SUCCESS) cuMemRelease(h);
  const bool none = mc == 0 && c != CUDA_SUCCESS;
  const bool held = none && (ld == 0 || ld == 1) && (rd == 0 || rd == 1);
  printf("verdict: %s: multicast %s; LDGMC on unicast memory %s; multimem.red on unicast memory %s\n",
         held ? "hypothesis held" : "hypothesis failed", none ? "unsupported" : "SUPPORTED",
         ld == 0 ? "returns the stored word (a load)" : ld == 1 ? "faults" : "returns something else",
         rd == 0 ? "adds once (an L2 reduction)" : rd == 1 ? "faults" : "does something else");
  return held ? 0 : 1;
}
