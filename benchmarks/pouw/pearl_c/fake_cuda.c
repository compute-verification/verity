// CPU stand-in for libcuda.so.1, for run.py's dry run off the pod (tests/test_pearl_c_sm120.py): "device" memory is host
// memory, copies and memsets are bounds-checked against their allocation, a launch fails as the driver's does when its
// dynamic shared memory exceeds 48 KB without cuFuncSetAttribute, and every allocation, stream, stream wait and launch (name,
// stream, grid, block, dynamic shared memory, the 16 pointers and 16 ints of Args) is appended to $FAKE_LOG as one JSON line.
// Streams are handles only: work runs in issue order, and a launch or wait on a destroyed stream fails.  Kernels do nothing, except
// the *_smem probes, which write $FAKE_SMEM (default 197728).  Functions are the names in $FAKE_KERNELS (comma-separated).
// Launches and asynchronous copies log their stream; cuLaunchKernelEx takes programmatic stream serialization alone, and logs
// "pdl":1 for a launch that allows it.
// The device is an H100 unless $FAKE_NAME, $FAKE_SMS, $FAKE_MAXDYN (the opt-in dynamic shared memory limit), $FAKE_CC
// ("major.minor") and $FAKE_UUID (32 hex digits) say otherwise.  A function's static shared memory is 0, so it may opt in to
// all of $FAKE_MAXDYN.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// the driver API's types and the constants used here (cuda.h, CUDA 12.9), so that the fake builds with a C compiler alone
typedef int CUresult, CUdevice, CUdevice_attribute, CUfunction_attribute;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUevent, *CUstream;
typedef unsigned long long CUdeviceptr;
#define CUDAAPI
enum { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_OUT_OF_MEMORY = 2, CUDA_ERROR_INVALID_DEVICE = 101,
       CUDA_ERROR_INVALID_IMAGE = 200, CUDA_ERROR_INVALID_CONTEXT = 201, CUDA_ERROR_INVALID_HANDLE = 400, CUDA_ERROR_NOT_FOUND = 500,
       CUDA_ERROR_ILLEGAL_ADDRESS = 700 };
enum { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
       CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8 };

// cuLaunchKernelEx's config and attribute (CUlaunchAttributeValue is a 64-byte union; the one attribute taken is an int)
typedef struct { int id; char pad[4]; int value; char rest[60]; } CUlaunchAttribute;
typedef struct { unsigned gx, gy, gz, bx, by, bz, smem; CUstream s; CUlaunchAttribute* attrs; unsigned n; } CUlaunchConfig;
enum { CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6 };

enum { MAX_ALLOC = 4096, MAX_FN = 128 };
struct Alloc { uintptr_t base; size_t size; };
static struct Alloc g_alloc[MAX_ALLOC];
static int g_nalloc, g_mod, g_ctx;
static char g_names[MAX_FN][64];
static int g_maxdyn[MAX_FN], g_nfn = -1;
static double g_ms;
static FILE* g_log;

static FILE* log_file(void) {
  if (!g_log) { const char* p = getenv("FAKE_LOG"); g_log = fopen(p ? p : "/dev/null", "a"); }
  return g_log;
}
static struct Alloc* find(uintptr_t p, size_t n) {
  for (int i = 0; i < g_nalloc; i++)
    if (g_alloc[i].size && p >= g_alloc[i].base && p + n <= g_alloc[i].base + g_alloc[i].size) return &g_alloc[i];
  return NULL;
}
static void load_names(void) {
  if (g_nfn >= 0) return;
  g_nfn = 0;
  const char* s = getenv("FAKE_KERNELS");
  if (!s) return;
  char buf[4096];
  snprintf(buf, sizeof buf, "%s", s);
  for (char* t = strtok(buf, ","); t && g_nfn < MAX_FN; t = strtok(NULL, ",")) snprintf(g_names[g_nfn++], 64, "%s", t);
}

CUresult CUDAAPI cuInit(unsigned int flags) { return flags ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS; }
CUresult CUDAAPI cuDriverGetVersion(int* v) { *v = 12090; return CUDA_SUCCESS; }
static int env_int(const char* name, int dflt) { const char* v = getenv(name); return v ? atoi(v) : dflt; }
static int max_dyn(void) { return env_int("FAKE_MAXDYN", 232448); }
CUresult CUDAAPI cuDeviceGetCount(int* n) { *n = 1; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGet(CUdevice* d, int ordinal) { if (ordinal) return CUDA_ERROR_INVALID_DEVICE; *d = 0; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice d) {
  (void)d; const char* v = getenv("FAKE_NAME"); snprintf(name, len, "%s", v ? v : "NVIDIA H100 80GB HBM3"); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDeviceGetUuid_v2(char* uuid, CUdevice d) {
  (void)d; const char* v = getenv("FAKE_UUID");
  for (int i = 0; i < 16; i++) { unsigned b = 0; if (v && strlen(v) >= 32) sscanf(v + 2 * i, "%2x", &b); uuid[i] = (char)b; }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDeviceGetAttribute(int* v, CUdevice_attribute a, CUdevice d) {
  (void)d; const char* cc = getenv("FAKE_CC"); int major = 9, minor = 0;
  if (cc) sscanf(cc, "%d.%d", &major, &minor);
  *v = a == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ? env_int("FAKE_SMS", 132)
     : a == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? major : a == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR ? minor
     : a == CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ? max_dyn() : 0;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* c, CUdevice d) { (void)d; *c = (CUcontext)&g_ctx; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxSetCurrent(CUcontext c) { return c == (CUcontext)&g_ctx ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT; }
CUresult CUDAAPI cuCtxSynchronize(void) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuModuleLoadData(CUmodule* m, const void* image) {
  const unsigned char* p = image;
  if (memcmp(p, "\x7f" "ELF", 4) || p[18] != 190) return CUDA_ERROR_INVALID_IMAGE;
  *m = (CUmodule)&g_mod; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleGetFunction(CUfunction* f, CUmodule m, const char* name) {
  if (m != (CUmodule)&g_mod) return CUDA_ERROR_INVALID_HANDLE;
  load_names();
  for (int i = 0; i < g_nfn; i++)
    if (!strcmp(g_names[i], name)) { *f = (CUfunction)(uintptr_t)(i + 1); g_maxdyn[i] = g_maxdyn[i] ? g_maxdyn[i] : 49152; return CUDA_SUCCESS; }
  return CUDA_ERROR_NOT_FOUND;
}
CUresult CUDAAPI cuFuncSetAttribute(CUfunction f, CUfunction_attribute a, int v) {
  const int i = (int)(uintptr_t)f - 1;
  if (i < 0 || i >= g_nfn || a != CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES || v < 0 || v > max_dyn()) return CUDA_ERROR_INVALID_VALUE;
  g_maxdyn[i] = v;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncGetAttribute(int* v, CUfunction_attribute a, CUfunction f) {
  const int i = (int)(uintptr_t)f - 1;
  if (i < 0 || i >= g_nfn) return CUDA_ERROR_INVALID_HANDLE;
  if (a == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) *v = 0;
  else if (a == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) *v = g_maxdyn[i];
  else return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* p, size_t n) {
  if (g_nalloc == MAX_ALLOC) return CUDA_ERROR_OUT_OF_MEMORY;
  void* q = aligned_alloc(256, (n + 255) / 256 * 256);
  if (!q) return CUDA_ERROR_OUT_OF_MEMORY;
  g_alloc[g_nalloc++] = (struct Alloc){(uintptr_t)q, n};
  *p = (CUdeviceptr)(uintptr_t)q;
  fprintf(log_file(), "{\"op\":\"alloc\",\"ptr\":%llu,\"bytes\":%zu}\n", (unsigned long long)*p, n);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemFree_v2(CUdeviceptr p) {
  for (int i = 0; i < g_nalloc; i++)
    if (g_alloc[i].size && g_alloc[i].base == (uintptr_t)p) {
      free((void*)g_alloc[i].base); g_alloc[i].size = 0;
      fprintf(log_file(), "{\"op\":\"free\",\"ptr\":%llu}\n", (unsigned long long)p);
      return CUDA_SUCCESS;
    }
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr p, unsigned char v, size_t n) {
  if (!find((uintptr_t)p, n)) return CUDA_ERROR_INVALID_VALUE;
  memset((void*)(uintptr_t)p, v, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr d, const void* h, size_t n) {
  if (!find((uintptr_t)d, n)) return CUDA_ERROR_INVALID_VALUE;
  memcpy((void*)(uintptr_t)d, h, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH_v2(void* h, CUdeviceptr d, size_t n) {
  if (!find((uintptr_t)d, n)) return CUDA_ERROR_INVALID_VALUE;
  memcpy(h, (void*)(uintptr_t)d, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) {
  if (!find((uintptr_t)d, n) || !find((uintptr_t)s, n)) return CUDA_ERROR_INVALID_VALUE;
  memmove((void*)(uintptr_t)d, (void*)(uintptr_t)s, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) {
  if (!find((uintptr_t)d, n) || !find((uintptr_t)s, n)) return CUDA_ERROR_INVALID_VALUE;
  memmove((void*)(uintptr_t)d, (void*)(uintptr_t)s, n);
  fprintf(log_file(), "{\"op\":\"copy\",\"dst\":%llu,\"src\":%llu,\"bytes\":%zu,\"stream\":%llu}\n", d, s, n,
          (unsigned long long)(uintptr_t)st);
  fflush(log_file());
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemGetInfo_v2(size_t* f, size_t* t) { *f = 80ull << 30; *t = 80ull << 30; return CUDA_SUCCESS; }

struct Stream { int live; };
static int stream_ok(CUstream s) { return !s || ((struct Stream*)s)->live == 1; }
CUresult CUDAAPI cuStreamCreate(CUstream* s, unsigned flags) {
  if (flags > 1) return CUDA_ERROR_INVALID_VALUE;
  struct Stream* q = calloc(1, sizeof *q);
  q->live = 1; *s = q;
  fprintf(log_file(), "{\"op\":\"stream\",\"ptr\":%llu}\n", (unsigned long long)(uintptr_t)q);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy_v2(CUstream s) {
  if (!s || !stream_ok(s)) return CUDA_ERROR_INVALID_HANDLE;
  ((struct Stream*)s)->live = 0; return CUDA_SUCCESS;
}

struct Args { uint64_t p[16]; int32_t i[16]; };
static CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem,
                       CUstream s, void** params, int pdl) {
  const int i = (int)(uintptr_t)f - 1;
  if (i < 0 || i >= g_nfn || !params || !stream_ok(s)) return CUDA_ERROR_INVALID_HANDLE;
  if (!gx || !gy || !gz || bx * by * bz > 1024 || !bx) return CUDA_ERROR_INVALID_VALUE;
  if ((int)smem > g_maxdyn[i]) return CUDA_ERROR_INVALID_VALUE;
  const struct Args* a = params[0];
  FILE* L = log_file();
  fprintf(L, "{\"op\":\"launch\",\"name\":\"%s\",%s\"stream\":%llu,\"grid\":[%u,%u,%u],\"block\":%u,\"smem\":%u,\"p\":[",
          g_names[i], pdl ? "\"pdl\":1," : "", (unsigned long long)(uintptr_t)s, gx, gy, gz, bx, smem);
  for (int j = 0; j < 16; j++) fprintf(L, "%s%llu", j ? "," : "", (unsigned long long)a->p[j]);
  fprintf(L, "],\"i\":[");
  for (int j = 0; j < 16; j++) fprintf(L, "%s%d", j ? "," : "", a->i[j]);
  fprintf(L, "]}\n");
  fflush(L);
  const size_t len = strlen(g_names[i]);
  if (len > 5 && !strcmp(g_names[i] + len - 5, "_smem")) {
    if (!find((uintptr_t)a->p[0], 4)) return CUDA_ERROR_ILLEGAL_ADDRESS;
    const char* v = getenv("FAKE_SMEM");
    *(int*)(uintptr_t)a->p[0] = v ? atoi(v) : 197728;
  }
  g_ms += 0.1;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                unsigned smem, CUstream s, void** params, void** extra) {
  (void)extra;
  return launch(f, gx, gy, gz, bx, by, bz, smem, s, params, 0);
}
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* c, CUfunction f, void** params, void** extra) {
  (void)extra;
  int pdl = 0;
  if (!c || (c->n && !c->attrs)) return CUDA_ERROR_INVALID_VALUE;
  for (unsigned j = 0; j < c->n; j++) {
    if (c->attrs[j].id != CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) return CUDA_ERROR_INVALID_VALUE;
    pdl = c->attrs[j].value != 0;
  }
  return launch(f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->smem, c->s, params, pdl);
}
struct Ev { double t; };
CUresult CUDAAPI cuEventCreate(CUevent* e, unsigned flags) { (void)flags; *e = (CUevent)calloc(1, sizeof(struct Ev)); return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventRecord(CUevent e, CUstream s) {
  if (!stream_ok(s)) return CUDA_ERROR_INVALID_HANDLE;
  ((struct Ev*)e)->t = g_ms; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream s, CUevent e, unsigned flags) {
  if (!stream_ok(s) || !e || flags) return CUDA_ERROR_INVALID_VALUE;
  fprintf(log_file(), "{\"op\":\"wait\",\"stream\":%llu}\n", (unsigned long long)(uintptr_t)s);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventSynchronize(CUevent e) { (void)e; return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventElapsedTime(float* ms, CUevent a, CUevent b) { *ms = (float)(((struct Ev*)b)->t - ((struct Ev*)a)->t); return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventDestroy_v2(CUevent e) { free(e); return CUDA_SUCCESS; }
CUresult CUDAAPI cuGetErrorString(CUresult r, const char** s) { (void)r; *s = "fake error"; return CUDA_SUCCESS; }
