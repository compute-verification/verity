// PCIe rates between pinned host DRAM and every visible GPU at once: the copy engines (cudaMemcpyAsync) host to device,
// device to host and both together, and kernels that read (zero copy) or write mapped pinned host memory, alone and beside
// D2H copies. One host thread per GPU; every GPU runs the same mode at the same time, for --seconds.
//
//   pcie --modes h2d,d2h,bidir,zcr,zcw,zcr_d2h,zcr_zcw --seconds 3 --mib 1024 --host-node local|N [--grid G] [--start-at T]
//        [--thp 1]
//
// --host-node local puts each GPU's host buffer on the GPU's own NUMA node (sysfs numa_node of its PCI device), remote on
// the other of two nodes; N binds every buffer to node N. --grid: the zero-copy kernels' blocks (default 4 a SM) of 512 threads.
// Gates, before any timing (exit 3 on a failure): each GPU's zero-copy fold of its host buffer equals the host's fold;
// the fold of the buffer copied H2D equals it too; a device pattern copied D2H and one written by zcw read back exactly.
// One JSON line per mode: each GPU's GB/s (both directions for bidir and zcr_d2h) and the sums.
#include <cuda_runtime.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(6);                                                                                   \
    }                                                                                            \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void zc_read(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
  size_t stride = (size_t)gridDim.x * blockDim.x, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint4 a = {0, 0, 0, 0};
  for (; i + 3 * stride < n; i += 4 * stride) {
    uint4 x = p[i], y = p[i + stride], z = p[i + 2 * stride], w = p[i + 3 * stride];
    a.x ^= x.x ^ y.x ^ z.x ^ w.x;
    a.y ^= x.y ^ y.y ^ z.y ^ w.y;
    a.z ^= x.z ^ y.z ^ z.z ^ w.z;
    a.w ^= x.w ^ y.w ^ z.w ^ w.w;
  }
  for (; i < n; i += stride) {
    uint4 x = p[i];
    a.x ^= x.x;
    a.y ^= x.y;
    a.z ^= x.z;
    a.w ^= x.w;
  }
  unsigned long long v = ((unsigned long long)(a.x ^ a.z) << 32) | (a.y ^ a.w);
  for (int o = 16; o; o >>= 1) v ^= __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) atomicXor(out, v);
}

__global__ void zc_write(uint4* p, size_t n, uint32_t tag) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = make_uint4((uint32_t)i ^ tag, (uint32_t)(i >> 32), tag, ~(uint32_t)i);
}

static uint64_t host_fold(const uint4* p, size_t n) {
  uint32_t a[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    a[0] ^= p[i].x;
    a[1] ^= p[i].y;
    a[2] ^= p[i].z;
    a[3] ^= p[i].w;
  }
  return ((uint64_t)(a[0] ^ a[2]) << 32) | (a[1] ^ a[3]);
}

struct Opts {
  std::vector<std::string> modes;
  double seconds = 3, start_at = 0;
  size_t bytes = 1ull << 30;
  int host_node = -1, grid = 0, thp = 1;
};

struct Rate {
  double in = 0, out = 0;  // GB/s host to device, device to host
};

static int gpu_numa(int dev) {
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
  for (char* c = bus; *c; ++c) *c = tolower(*c);
  std::string path = std::string("/sys/bus/pci/devices/") + bus + "/numa_node";
  FILE* f = fopen(path.c_str(), "r");
  int n = -1;
  if (f) {
    if (fscanf(f, "%d", &n) != 1) n = -1;
    fclose(f);
  }
  return n;
}

static void* host_alloc(size_t n, int node, int thp) {
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  madvise(p, n, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  if (node >= 0) {
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, n, 2 /* MPOL_BIND */, &mask, 64, 1 /* MPOL_MF_STRICT */) != 0) {
      perror("mbind");
      exit(5);
    }
  }
  return p;
}

struct Gpu {
  int dev, node, sms;
  std::string bus;
  uint4 *host, *host_dev, *dev_buf;  // host_dev: the host buffer's device address
  size_t n16;
  cudaStream_t s[4];
  unsigned long long* fold;
  std::vector<Rate> rates;
  bool gate_ok = true;
  std::string gate_msg;
};

static pthread_barrier_t barrier;
static Opts opt;

// One stream's work in a mode: an op over a region of the buffers, repeated in rounds for --seconds.
struct Lane {
  enum Op { H2D, D2H, ZCR, ZCW } op;
  size_t off, len;  // bytes of the host and device buffers this lane uses
};

static std::vector<Lane> lanes_of(const std::string& m, size_t bytes) {
  size_t half = bytes / 2;
  if (m == "h2d") return {{Lane::H2D, 0, bytes}};
  if (m == "d2h") return {{Lane::D2H, 0, bytes}};
  if (m == "bidir") return {{Lane::H2D, 0, half}, {Lane::D2H, half, half}};
  if (m == "zcr") return {{Lane::ZCR, 0, bytes}};
  if (m == "zcw") return {{Lane::ZCW, 0, bytes}};
  if (m == "zcr_d2h") return {{Lane::ZCR, 0, half}, {Lane::D2H, half, half}};
  if (m == "zcr_zcw") return {{Lane::ZCR, 0, half}, {Lane::ZCW, half, half}};
  fprintf(stderr, "unknown mode %s\n", m.c_str());
  exit(2);
}

static void run_lane(Gpu& g, const Lane& L, cudaStream_t s, bool timed, double* in, double* out) {
  CK(cudaSetDevice(g.dev));
  const size_t chunk = 64ull << 20;
  const int grid = opt.grid ? opt.grid : 4 * g.sms;
  uint8_t* H = (uint8_t*)g.host + L.off;
  uint8_t* HD = (uint8_t*)g.host_dev + L.off;
  uint8_t* D = (uint8_t*)g.dev_buf + L.off;
  cudaEvent_t ev[2];
  CK(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));
  auto enqueue = [&](int r) -> double {
    if (L.op == Lane::H2D || L.op == Lane::D2H) {
      for (int k = 0; k < 4; ++k) {
        size_t o = ((size_t)(r * 4 + k) * chunk) % L.len;
        if (L.op == Lane::H2D) CK(cudaMemcpyAsync(D + o, H + o, chunk, cudaMemcpyHostToDevice, s));
        else CK(cudaMemcpyAsync(H + o, D + o, chunk, cudaMemcpyDeviceToHost, s));
      }
      return 4.0 * chunk;
    }
    if (L.op == Lane::ZCR) zc_read<<<grid, 512, 0, s>>>((const uint4*)HD, L.len / 16, g.fold);
    else zc_write<<<grid, 512, 0, s>>>((uint4*)HD, L.len / 16, (uint32_t)r);
    CK(cudaGetLastError());
    return (double)L.len;
  };
  if (!timed) {  // one untimed round, so module loads and first mappings stay out of the time
    enqueue(0);
    CK(cudaStreamSynchronize(s));
    return;
  }
  double bytes = 0, pend = 0, t0 = now();
  for (int r = 0;; ++r) {
    double b = enqueue(r);
    CK(cudaEventRecord(ev[r & 1], s));
    if (r > 0) {
      CK(cudaEventSynchronize(ev[(r - 1) & 1]));
      bytes += pend;
      if (now() - t0 >= opt.seconds) {
        CK(cudaStreamSynchronize(s));
        bytes += b;
        break;
      }
    }
    pend = b;
  }
  double rate = bytes / (now() - t0) / 1e9;
  (L.op == Lane::H2D || L.op == Lane::ZCR ? *in : *out) += rate;
  cudaEventDestroy(ev[0]);
  cudaEventDestroy(ev[1]);
}

static void run_mode(Gpu& g, const std::string& m) {
  std::vector<Lane> lanes = lanes_of(m, g.n16 * 16);
  double in = 0, out = 0;
  for (size_t k = 0; k < lanes.size(); ++k) run_lane(g, lanes[k], g.s[k], false, &in, &out);
  pthread_barrier_wait(&barrier);
  std::vector<double> ins(lanes.size()), outs(lanes.size());
  std::vector<std::thread> th;
  for (size_t k = 0; k < lanes.size(); ++k)
    th.emplace_back(run_lane, std::ref(g), std::cref(lanes[k]), g.s[k], true, &ins[k], &outs[k]);
  for (auto& t : th) t.join();
  for (size_t k = 0; k < lanes.size(); ++k) in += ins[k], out += outs[k];
  g.rates.push_back({in, out});
}

static void gate(Gpu& g) {
  const int grid = opt.grid ? opt.grid : 4 * g.sms;
  uint64_t want = host_fold(g.host, g.n16);
  CK(cudaMemset(g.fold, 0, 8));
  zc_read<<<grid, 512>>>(g.host_dev, g.n16, g.fold);
  unsigned long long got = 0;
  CK(cudaMemcpy(&got, g.fold, 8, cudaMemcpyDeviceToHost));
  if (got != want) g.gate_ok = false, g.gate_msg += "zcr fold differs; ";
  CK(cudaMemcpy(g.dev_buf, g.host, g.n16 * 16, cudaMemcpyHostToDevice));
  CK(cudaMemset(g.fold, 0, 8));
  zc_read<<<grid, 512>>>(g.dev_buf, g.n16, g.fold);
  CK(cudaMemcpy(&got, g.fold, 8, cudaMemcpyDeviceToHost));
  if (got != want) g.gate_ok = false, g.gate_msg += "h2d fold differs; ";
  zc_write<<<grid, 512>>>(g.dev_buf, g.n16, 0x5a5a5a5au);
  CK(cudaMemcpy(g.host, g.dev_buf, g.n16 * 16, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < g.n16; i += 4099)
    if (g.host[i].x != ((uint32_t)i ^ 0x5a5a5a5au) || g.host[i].w != ~(uint32_t)i) {
      g.gate_ok = false, g.gate_msg += "d2h pattern differs; ";
      break;
    }
  zc_write<<<grid, 512>>>(g.host_dev, g.n16, 0xa5a5a5a5u);
  CK(cudaDeviceSynchronize());
  for (size_t i = 0; i < g.n16; i += 4099)
    if (g.host[i].x != ((uint32_t)i ^ 0xa5a5a5a5u) || g.host[i].y != (uint32_t)(i >> 32)) {
      g.gate_ok = false, g.gate_msg += "zcw pattern differs; ";
      break;
    }
}

static std::atomic<int> gates_failed{0};

static void worker(Gpu* g) {
  CK(cudaSetDevice(g->dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, g->dev));
  g->sms = p.multiProcessorCount;
  char bus[32];
  CK(cudaDeviceGetPCIBusId(bus, sizeof bus, g->dev));
  g->bus = bus;
  int numa = gpu_numa(g->dev);
  g->node = opt.host_node >= 0 ? opt.host_node : opt.host_node == -2 ? 1 - numa : numa;
  g->n16 = opt.bytes / 16;
  g->host = (uint4*)host_alloc(opt.bytes, g->node, opt.thp);
  if (!g->host) exit(5);
  uint64_t* h = (uint64_t*)g->host;
  for (size_t i = 0; i < opt.bytes / 8; ++i) h[i] = (i + 1) * 0x9e3779b97f4a7c15ull ^ ((uint64_t)g->dev << 56);
  CK(cudaHostRegister(g->host, opt.bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  CK(cudaHostGetDevicePointer((void**)&g->host_dev, g->host, 0));
  CK(cudaMalloc(&g->dev_buf, opt.bytes));
  CK(cudaMalloc(&g->fold, 8));
  for (auto& s : g->s) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  gate(*g);
  if (!g->gate_ok) gates_failed++;
  pthread_barrier_wait(&barrier);
  if (gates_failed) return;
  for (auto& m : opt.modes) run_mode(*g, m);
}

int main(int argc, char** argv) {
  std::string modes = "h2d,d2h,bidir,zcr,zcw,zcr_d2h,zcr_zcw";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "--modes") modes = v;
    else if (a == "--seconds") opt.seconds = atof(v.c_str());
    else if (a == "--mib") opt.bytes = (size_t)atol(v.c_str()) << 20;
    else if (a == "--host-node") opt.host_node = v == "local" ? -1 : v == "remote" ? -2 : atoi(v.c_str());
    else if (a == "--grid") opt.grid = atoi(v.c_str());
    else if (a == "--thp") opt.thp = atoi(v.c_str());
    else if (a == "--start-at") opt.start_at = atof(v.c_str());
    else {
      fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  for (size_t p = 0, q; p < modes.size(); p = q + 1) {
    q = modes.find(',', p);
    if (q == std::string::npos) q = modes.size();
    opt.modes.push_back(modes.substr(p, q - p));
  }
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  std::vector<Gpu> gpus(n);
  pthread_barrier_init(&barrier, nullptr, n + 1);
  std::vector<std::thread> th;
  for (int d = 0; d < n; ++d) {
    gpus[d].dev = d;
    th.emplace_back(worker, &gpus[d]);
  }
  pthread_barrier_wait(&barrier);  // gates done
  for (auto& g : gpus)
    printf("{\"gate\": \"%s\", \"gpu\": %d, \"bus\": \"%s\", \"host_node\": %d, \"msg\": \"%s\"}\n",
           g.gate_ok ? "ok" : "FAILED", g.dev, g.bus.c_str(), g.node, g.gate_msg.c_str());
  fflush(stdout);
  if (gates_failed) {
    for (auto& t : th) t.join();
    return 3;
  }
  if (opt.start_at > 0) {
    double wait = opt.start_at - std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
  }
  for (size_t k = 0; k < opt.modes.size(); ++k) pthread_barrier_wait(&barrier);
  for (auto& t : th) t.join();
  for (size_t k = 0; k < opt.modes.size(); ++k) {
    double sin = 0, sout = 0;
    std::string per = "[";
    for (auto& g : gpus) {
      char b[160];
      snprintf(b, sizeof b, "%s{\"gpu\": %d, \"bus\": \"%s\", \"host_node\": %d, \"in\": %.2f, \"out\": %.2f}",
               per.size() > 1 ? ", " : "", g.dev, g.bus.c_str(), g.node, g.rates[k].in, g.rates[k].out);
      per += b;
      sin += g.rates[k].in;
      sout += g.rates[k].out;
    }
    per += "]";
    printf("{\"mode\": \"%s\", \"gpus\": %d, \"mib\": %zu, \"grid\": %d, \"thp\": %d, \"seconds\": %.1f, "
           "\"in_gb_per_s\": %.2f, \"out_gb_per_s\": %.2f, \"per_gpu\": %s}\n",
           opt.modes[k].c_str(), n, opt.bytes >> 20, opt.grid ? opt.grid : 4 * gpus[0].sms, opt.thp, opt.seconds, sin,
           sout, per.c_str());
  }
  return 0;
}
