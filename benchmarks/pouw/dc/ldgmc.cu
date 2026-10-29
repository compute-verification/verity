// The census's 21 LDGMC gaps on the H200 (sm_90a), priced per 32-bit register written: multimem.ld_reduce from a
// multicast object, which the NVSwitch reduces over the group's copies and returns. The opcode-probe harness allocates
// plain device memory, so this program makes the multicast object itself (one device in its group, the leased GPU)
// and times each form as the harness does: one block of 1024 threads an SM, U independent chains a thread, each a
// fixed cell of the form's width, alternating between two sets a trip so ptxas can't hoist the loads, every loaded
// word folded into its chain (LOP3, then IADD3), the block's SM cycles from the first barrier to the last. Every
// returned word is checked: with one copy in the group, each form returns the word stored, so the sink is known.
//
//   ldgmc --list                  the forms, one a line: kernel, the SASS form it must lower to, registers a lane
//   ldgmc [--reps R] [--iters N]  times every form; one line a form, then a verdict line
// Exit: 0 every form timed and checked, 1 a check failed, 3 the node has no multicast object (LDGMC can't run).
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int TPB = 1024, U = 4, SMS = 132;
constexpr size_t NTOT = (size_t)SMS * TPB;

struct Rec {
  long long cyc;
  int smid, pad;
};

__device__ __forceinline__ unsigned hi_lo(unsigned long long v) { return (unsigned)v ^ (unsigned)(v >> 32); }

// One form: NREG 32-bit registers a lane, the PTX that loads them into w[0..NREG). Cell (set, u) of thread g is at
// ((set·U + u)·NTOT + g)·NREG words, so a warp's load is one contiguous run of 32 cells.
#define FORM(NAME, NREG, LOAD)                                                                                       \
  extern "C" __global__ void __launch_bounds__(TPB, 1)                                                               \
      NAME(const unsigned char* mc, Rec* rec, unsigned* sink, int iters) {                                            \
    constexpr size_t S = NTOT * NREG * 4;                                                                             \
    const unsigned char* base = mc + ((size_t)blockIdx.x * TPB + threadIdx.x) * (NREG * 4);                           \
    unsigned acc[U];                                                                                                  \
    _Pragma("unroll") for (int u = 0; u < U; u++) acc[u] = threadIdx.x * 0x9E3779B9u + u;                             \
    __syncthreads();                                                                                                  \
    const long long t0 = clock64();                                                                                   \
    _Pragma("unroll 1") for (int it = 0; it < iters; ++it) {                                                          \
      const unsigned char* s = base + (it & 1) * (U * S);                                                             \
      _Pragma("unroll") for (int u = 0; u < U; u++) {                                                                 \
        const unsigned char* p = s + u * S;                                                                           \
        unsigned w[NREG];                                                                                             \
        LOAD;                                                                                                         \
        unsigned f = w[0];                                                                                            \
        _Pragma("unroll") for (int r = 1; r < NREG; r++) f ^= w[r];                                                   \
        acc[u] += f;                                                                                                  \
      }                                                                                                               \
    }                                                                                                                 \
    __syncthreads();                                                                                                  \
    if (threadIdx.x == 0) {                                                                                           \
      rec[blockIdx.x].cyc = clock64() - t0;                                                                           \
      int sm;                                                                                                         \
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));                                                                 \
      rec[blockIdx.x].smid = sm;                                                                                      \
    }                                                                                                                 \
    unsigned t = 0;                                                                                                   \
    _Pragma("unroll") for (int u = 0; u < U; u++) t ^= acc[u];                                                        \
    sink[blockIdx.x * TPB + threadIdx.x] = t;                                                                         \
  }

#define MM "multimem.ld_reduce.relaxed.sys.global."
#define L64(OP)                                                                  \
  {                                                                              \
    unsigned long long v;                                                        \
    asm volatile(MM OP " %0, [%1];" : "=l"(v) : "l"(p) : "memory");              \
    w[0] = (unsigned)v;                                                          \
    w[1] = (unsigned)(v >> 32);                                                  \
  }
#define LF64(OP)                                                                 \
  {                                                                              \
    double v;                                                                    \
    asm volatile(MM OP " %0, [%1];" : "=d"(v) : "l"(p) : "memory");              \
    const unsigned long long b = __double_as_longlong(v);                        \
    w[0] = (unsigned)b;                                                          \
    w[1] = (unsigned)(b >> 32);                                                  \
  }
#define LV2R(OP) asm volatile(MM OP " {%0,%1}, [%2];" : "=r"(w[0]), "=r"(w[1]) : "l"(p) : "memory")
#define LV4R(OP) \
  asm volatile(MM OP " {%0,%1,%2,%3}, [%4];" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p) : "memory")
#define LV2F(OP)                                                                 \
  {                                                                              \
    float x, y;                                                                  \
    asm volatile(MM OP " {%0,%1}, [%2];" : "=f"(x), "=f"(y) : "l"(p) : "memory"); \
    w[0] = __float_as_uint(x);                                                   \
    w[1] = __float_as_uint(y);                                                   \
  }
#define LV4F(OP)                                                                                           \
  {                                                                                                        \
    float x, y, z, q;                                                                                      \
    asm volatile(MM OP " {%0,%1,%2,%3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(q) : "l"(p) : "memory"); \
    w[0] = __float_as_uint(x);                                                                             \
    w[1] = __float_as_uint(y);                                                                             \
    w[2] = __float_as_uint(z);                                                                             \
    w[3] = __float_as_uint(q);                                                                             \
  }
#define LF32(OP)                                                                 \
  {                                                                              \
    float x;                                                                     \
    asm volatile(MM OP " %0, [%1];" : "=f"(x) : "l"(p) : "memory");              \
    w[0] = __float_as_uint(x);                                                   \
  }

FORM(add_u64, 2, L64("add.u64"))
FORM(add_v4_bf16x2, 4, LV4R("add.v4.bf16x2"))
FORM(add_v4_f16x2, 4, LV4R("add.v4.f16x2"))
FORM(add_v2_f32, 2, LV2F("add.v2.f32"))
FORM(add_v4_f32, 4, LV4F("add.v4.f32"))
FORM(add_f64, 2, LF64("add.f64"))
FORM(and_b64, 2, L64("and.b64"))
FORM(add_acc_v2_bf16x2, 2, LV2R("add.acc::f32.v2.bf16x2"))
FORM(add_acc_v4_bf16x2, 4, LV4R("add.acc::f32.v4.bf16x2"))
FORM(add_acc_v2_f16x2, 2, LV2R("add.acc::f32.v2.f16x2"))
FORM(add_acc_v4_f16x2, 4, LV4R("add.acc::f32.v4.f16x2"))
FORM(max_u64, 2, L64("max.u64"))
FORM(max_v4_bf16x2, 4, LV4R("max.v4.bf16x2"))
FORM(max_v4_f16x2, 4, LV4R("max.v4.f16x2"))
FORM(max_s64, 2, L64("max.s64"))
FORM(min_u64, 2, L64("min.u64"))
FORM(min_v4_bf16x2, 4, LV4R("min.v4.bf16x2"))
FORM(min_v4_f16x2, 4, LV4R("min.v4.f16x2"))
FORM(min_s64, 2, L64("min.s64"))
FORM(or_b64, 2, L64("or.b64"))
FORM(xor_b64, 2, L64("xor.b64"))
FORM(add_f32, 1, LF32("add.f32"))

typedef void (*Kernel)(const unsigned char*, Rec*, unsigned*, int);
struct Form {
  const char* name;
  Kernel k;
  const char* sass;  // the census's form
  int nreg;
  bool gap;
};
static const Form FORMS[] = {
    {"add_u64", add_u64, "LDGMC.E.ADD.64.STRONG.SYS", 2, true},
    {"add_v4_bf16x2", add_v4_bf16x2, "LDGMC.E.ADD.BF16x8.RN.STRONG.SYS", 4, true},
    {"add_v4_f16x2", add_v4_f16x2, "LDGMC.E.ADD.F16x8.RN.STRONG.SYS", 4, true},
    {"add_v2_f32", add_v2_f32, "LDGMC.E.ADD.F32x2.RN.STRONG.SYS", 2, true},
    {"add_v4_f32", add_v4_f32, "LDGMC.E.ADD.F32x4.RN.STRONG.SYS", 4, true},
    {"add_f64", add_f64, "LDGMC.E.ADD.F64.RN.STRONG.SYS", 2, true},
    {"and_b64", and_b64, "LDGMC.E.AND.64.STRONG.SYS", 2, true},
    {"add_acc_v2_bf16x2", add_acc_v2_bf16x2, "LDGMC.E.F32ADD.BF16x4.RN.STRONG.SYS", 2, true},
    {"add_acc_v4_bf16x2", add_acc_v4_bf16x2, "LDGMC.E.F32ADD.BF16x8.RN.STRONG.SYS", 4, true},
    {"add_acc_v2_f16x2", add_acc_v2_f16x2, "LDGMC.E.F32ADD.F16x4.RN.STRONG.SYS", 2, true},
    {"add_acc_v4_f16x2", add_acc_v4_f16x2, "LDGMC.E.F32ADD.F16x8.RN.STRONG.SYS", 4, true},
    {"max_u64", max_u64, "LDGMC.E.MAX.64.STRONG.SYS", 2, true},
    {"max_v4_bf16x2", max_v4_bf16x2, "LDGMC.E.MAX.BF16x8.RN.STRONG.SYS", 4, true},
    {"max_v4_f16x2", max_v4_f16x2, "LDGMC.E.MAX.F16x8.RN.STRONG.SYS", 4, true},
    {"max_s64", max_s64, "LDGMC.E.MAX.S64.STRONG.SYS", 2, true},
    {"min_u64", min_u64, "LDGMC.E.MIN.64.STRONG.SYS", 2, true},
    {"min_v4_bf16x2", min_v4_bf16x2, "LDGMC.E.MIN.BF16x8.RN.STRONG.SYS", 4, true},
    {"min_v4_f16x2", min_v4_f16x2, "LDGMC.E.MIN.F16x8.RN.STRONG.SYS", 4, true},
    {"min_s64", min_s64, "LDGMC.E.MIN.S64.STRONG.SYS", 2, true},
    {"or_b64", or_b64, "LDGMC.E.OR.64.STRONG.SYS", 2, true},
    {"xor_b64", xor_b64, "LDGMC.E.XOR.64.STRONG.SYS", 2, true},
    {"add_f32", add_f32, "LDGMC.E.ADD.F32.RN.STRONG.SYS", 1, false},
};

// Each 16-bit half is 0x3C00 plus 10 bits of the word's index: a finite normal value read as f16, bf16, f32 or f64.
__host__ __device__ inline unsigned fill_word(size_t i) {
  const unsigned lo = 0x3C00u + (unsigned)(i & 0x3FF), hi = 0x3C00u + (unsigned)((i >> 10) & 0x3FF);
  return lo | (hi << 16);
}

__global__ void fill(unsigned* uc, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    uc[i] = fill_word(i);
}

#define CU(x)                                                                    \
  do {                                                                           \
    const CUresult r_ = (x);                                                     \
    if (r_ != CUDA_SUCCESS) {                                                    \
      const char* s_ = "?";                                                      \
      cuGetErrorName(r_, &s_);                                                   \
      printf("ldgmc: %s: %s\n", #x, s_);                                         \
      return 3;                                                                  \
    }                                                                            \
  } while (0)

static unsigned expected(int nreg, int iters, unsigned tid_global, unsigned tid) {
  unsigned t = 0;
  for (int u = 0; u < U; u++) {
    unsigned acc = tid * 0x9E3779B9u + u;
    for (int it = 0; it < iters; it++) {
      const size_t w0 = (((size_t)(it & 1) * U + u) * NTOT + tid_global) * nreg;
      unsigned f = fill_word(w0);
      for (int r = 1; r < nreg; r++) f ^= fill_word(w0 + r);
      acc += f;
    }
    t ^= acc;
  }
  return t;
}

int main(int argc, char** argv) {
  int reps = 5, iters = 64;
  bool list = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--list")) list = true;
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
  }
  if (list) {
    for (const Form& f : FORMS) printf("%s %s %d %s\n", f.name, f.sass, f.nreg, f.gap ? "gap" : "reference");
    return 0;
  }
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  char devname[256];
  CU(cuDeviceGetName(devname, sizeof devname, dev));
  int mc_ok = 0, sms = 0;
  CU(cuDeviceGetAttribute(&mc_ok, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
  CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  printf("device %s: %d SMs, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = %d\n", devname, sms, mc_ok);
  if (sms > SMS) {
    printf("ldgmc: %d SMs, more than the %d the layout holds\n", sms, SMS);
    return 3;
  }
  if (!mc_ok) {
    printf("verdict: no multicast object on this node, so no LDGMC form can run here\n");
    return 3;
  }
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));

  const size_t want = NTOT * 2 * U * 4 * 4;
  CUmulticastObjectProp mp;
  memset(&mp, 0, sizeof mp);
  mp.numDevices = 1;
  mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mp.size = want;
  size_t mg = 0;
  CU(cuMulticastGetGranularity(&mg, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  CUmemAllocationProp ap;
  memset(&ap, 0, sizeof ap);
  ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ap.location.id = dev;
  ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t ag = 0;
  CU(cuMemGetAllocationGranularity(&ag, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  const size_t gran = std::max(mg, ag), size = (want + gran - 1) / gran * gran;
  mp.size = size;
  CUmemGenericAllocationHandle mch, ph;
  CU(cuMulticastCreate(&mch, &mp));
  CU(cuMulticastAddDevice(mch, dev));
  CU(cuMemCreate(&ph, size, &ap, 0));
  CU(cuMulticastBindMem(mch, 0, ph, 0, size, 0));
  CUdeviceptr uc, mcp;
  CU(cuMemAddressReserve(&uc, size, gran, 0, 0));
  CU(cuMemMap(uc, size, 0, ph, 0));
  CU(cuMemAddressReserve(&mcp, size, gran, 0, 0));
  CU(cuMemMap(mcp, size, 0, mch, 0));
  CUmemAccessDesc ad;
  memset(&ad, 0, sizeof ad);
  ad.location = ap.location;
  ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(uc, size, &ad, 1));
  CU(cuMemSetAccess(mcp, size, &ad, 1));
  printf("multicast object: 1 device, %zu bytes (granularity %zu)\n", size, gran);
  fill<<<sms * 4, 256>>>((unsigned*)uc, size / 4);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("ldgmc: fill failed\n");
    return 3;
  }

  Rec* rec;
  unsigned* sink;
  cudaMalloc(&rec, sms * sizeof(Rec));
  cudaMalloc(&sink, (size_t)sms * TPB * sizeof(unsigned));
  std::vector<Rec> hr(sms);
  std::vector<unsigned> hs((size_t)sms * TPB);
  bool all_ok = true;
  printf("form sass regs_a_lane cycles_median words_a_sm_clock price_a_ldgmc_register price_a_loop_register checked\n");
  for (const Form& f : FORMS) {
    std::vector<double> cyc;
    bool ok = true;
    for (int r = 0; r < reps + 1; r++) {
      cudaMemset(sink, 0xA5, (size_t)sms * TPB * sizeof(unsigned));
      f.k<<<sms, TPB>>>((const unsigned char*)mcp, rec, sink, iters);
      const cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        printf("%s %s fault: %s\n", f.name, f.sass, cudaGetErrorString(e));
        return 1;
      }
      cudaMemcpy(hr.data(), rec, sms * sizeof(Rec), cudaMemcpyDeviceToHost);
      cudaMemcpy(hs.data(), sink, hs.size() * sizeof(unsigned), cudaMemcpyDeviceToHost);
      for (int b = 0; b < sms && ok && (r == 0 || r == reps); b++)
        for (int t = 0; t < TPB; t++)
          if (hs[(size_t)b * TPB + t] != expected(f.nreg, iters, (unsigned)(b * TPB + t), (unsigned)t)) {
            ok = false;
            break;
          }
      if (r == 0) continue;  // the first pass warms the switch's path and the code
      std::vector<long long> c(sms);
      for (int b = 0; b < sms; b++) c[b] = hr[b].cyc;
      std::nth_element(c.begin(), c.begin() + sms / 2, c.end());
      cyc.push_back((double)c[sms / 2]);
    }
    std::nth_element(cyc.begin(), cyc.begin() + cyc.size() / 2, cyc.end());
    const double med = cyc[cyc.size() / 2];
    const double words = (double)TPB * U * f.nreg * iters / med;
    const double loop_regs = (double)TPB * U * (f.nreg + f.nreg / 2 + 1) * iters / med;
    printf("%s %s %d %.0f %.5f %.2f %.2f %s\n", f.name, f.sass, f.nreg, med, words, 4096.0 / words, 4096.0 / loop_regs,
           ok ? "ok" : "MISMATCH");
    all_ok = all_ok && ok;
  }
  printf("verdict: %s\n", all_ok ? "every form timed, and every returned word is the stored one" : "a check failed");
  return all_ok ? 0 : 1;
}
