// lat: how late an honest answer to a timed PoUS challenge arrives. Three subcommands:
//   serve  the responder: one block per challenge from GPU memory, sent over UDP from one busy-polling thread pinned to
//          one CPU, read either by a D2H copy on the highest-priority stream (op 1, the copy engines) or by a kernel on
//          that stream storing it into mapped host memory (op 5, the SMs); the same block size from host memory (op 2);
//          a bare ping (op 3). Each answer carries the responder's device and server times.
//   probe  the challenger: sequential challenges, the next sent when the last answer is whole, audits of k back to back;
//          one record per challenge (elapsed, server, device, flags), lost answers kept as lost.
//   load   a serving-like load on one GPU (bf16 GEMMs, device-to-device copies, pinned H2D/D2H copies) and on host CPUs.
// nvcc -O2 -std=c++17 -arch=sm_120 -o lat lat.cu -lcublas    (all three)
// g++ -O2 -std=c++17 -x c++ -o lat lat.cu -lpthread          (probe, and serve with op 1 answered from host memory)
// Arguments are key=value; `peer` repeats. The wire format and record layout are in README.md.

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
      exit(3);                                                                                  \
    }                                                                                           \
  } while (0)
#endif

static inline uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}
static double wall_s() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return double(t.tv_sec) + 1e-9 * double(t.tv_nsec);
}

constexpr uint32_t MAGIC = 0x54414c50;
enum : uint32_t { OP_GPU = 1, OP_HOST = 2, OP_PING = 3, OP_HELLO = 4, OP_GPU_SM = 5 };
static bool gpu_op(uint32_t op) { return op == OP_GPU || op == OP_GPU_SM; }
// Answer flags: the responder lost its CPU since the previous request arrived (1), the challenger lost its CPU during
// this challenge (2), no whole answer before the timeout (4), the payload differed from the expected block (8).
enum : uint16_t { F_RESP_PREEMPTED = 1, F_CHAL_PREEMPTED = 2, F_LOST = 4, F_BAD_DATA = 8 };
struct Req {
  uint32_t magic, op, seq, idx;
};
struct Hdr {
  uint32_t magic, seq;
  uint16_t op;
  uint8_t chunk, nchunks;
  uint32_t device_ns, server_ns, flags;
};
struct Rec {
  uint32_t elapsed_ns, server_ns, device_ns;
  uint16_t op, flags;
};
static_assert(sizeof(Req) == 16, "request is 16 bytes");
static_assert(sizeof(Hdr) == 24, "answer header is 24 bytes");
static_assert(sizeof(Rec) == 16, "record is 16 bytes");
// A challenge's op field carries the answer's largest datagram payload in its high 16 bits (0: 1,472 bytes, a 1,500-byte
// MTU less the IPv4 and UDP headers), so the challenger fits answers to its path's MTU.
constexpr size_t DGRAM = 1472;
constexpr size_t MIN_DGRAM = 512;
constexpr int MAX_CHUNKS = 64;

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

struct Args {
  std::map<std::string, std::vector<std::string>> kv;
  Args(int argc, char** argv) {
    for (int i = 2; i < argc; i++) {
      std::string s = argv[i];
      auto eq = s.find('=');
      if (eq == std::string::npos) {
        fprintf(stderr, "argument %s is not key=value\n", argv[i]);
        exit(2);
      }
      kv[s.substr(0, eq)].push_back(s.substr(eq + 1));
    }
  }
  std::string s(const char* k, const char* d = nullptr) const {
    auto it = kv.find(k);
    if (it != kv.end()) return it->second.back();
    if (!d) {
      fprintf(stderr, "missing %s=\n", k);
      exit(2);
    }
    return d;
  }
  long long i(const char* k, long long d) const { return kv.count(k) ? atoll(s(k).c_str()) : d; }
  std::vector<std::string> all(const char* k) const { return kv.count(k) ? kv.at(k) : std::vector<std::string>{}; }
};

static uint64_t mix(uint64_t i, uint64_t seed) {
  uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}
constexpr uint64_t GPU_SEED = 0x5ee0, HOST_SEED = 0x7a11;

static long nivcsw() {
  rusage r;
  getrusage(RUSAGE_THREAD, &r);
  return r.ru_nivcsw;
}

static std::vector<int> allowed_cpus() {
  cpu_set_t s;
  CPU_ZERO(&s);
  sched_getaffinity(0, sizeof s, &s);
  std::vector<int> out;
  for (int c = 0; c < CPU_SETSIZE; c++)
    if (CPU_ISSET(c, &s)) out.push_back(c);
  return out;
}

static std::vector<int> cpu_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) continue;
    auto dash = part.find('-');
    int lo = atoi(part.substr(0, dash).c_str()), hi = dash == std::string::npos ? lo : atoi(part.substr(dash + 1).c_str());
    for (int c = lo; c <= hi; c++) out.push_back(c);
  }
  return out;
}

static std::map<int, unsigned long long> idle_ticks() {
  std::map<int, unsigned long long> out;
  std::ifstream f("/proc/stat");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("cpu", 0) != 0 || line.size() < 4 || !isdigit(line[3])) continue;
    std::istringstream is(line.substr(3));
    int c;
    unsigned long long v[8] = {};
    is >> c;
    for (auto& x : v) is >> x;
    out[c] = v[3] + v[4];
  }
  return out;
}

// cpu=N pins to N; cpu=auto pins to the allowed CPU that was idlest over half a second (CPU 0 last).
static int pin(const std::string& spec) {
  int cpu = -1;
  if (spec == "auto") {
    auto a = idle_ticks();
    usleep(500000);
    auto b = idle_ticks();
    unsigned long long best = 0;
    for (int c : allowed_cpus()) {
      unsigned long long d = b[c] - a[c] + (c == 0 ? 0 : 1);
      if (cpu < 0 || d > best) best = d, cpu = c;
    }
  } else if (spec != "none") {
    cpu = atoi(spec.c_str());
  }
  if (cpu >= 0) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof s, &s)) {
      perror("sched_setaffinity");
      exit(2);
    }
  }
  return cpu;
}

// Keep every other thread of this process (the CUDA runtime's) off `cpu`, so they never preempt the polling thread.
static int isolate(int cpu, const std::vector<int>& allowed) {
  if (cpu < 0) return 0;
  cpu_set_t s;
  CPU_ZERO(&s);
  for (int c : allowed)
    if (c != cpu) CPU_SET(c, &s);
  int moved = 0;
  const pid_t self = pid_t(gettid());
  DIR* d = opendir("/proc/self/task");
  while (dirent* e = d ? readdir(d) : nullptr) {
    const pid_t tid = pid_t(atoi(e->d_name));
    if (tid > 0 && tid != self && CPU_COUNT(&s) && sched_setaffinity(tid, sizeof s, &s) == 0) moved++;
  }
  if (d) closedir(d);
  return moved;
}

static sockaddr_in addr(const std::string& hostport) {
  auto colon = hostport.rfind(':');
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(uint16_t(atoi(hostport.substr(colon + 1).c_str())));
  if (inet_pton(AF_INET, hostport.substr(0, colon).c_str(), &a.sin_addr) != 1) {
    fprintf(stderr, "bad address %s\n", hostport.c_str());
    exit(2);
  }
  return a;
}

static int udp_socket(int port) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  int buf = 8 << 20;
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
  if (port > 0) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(uint16_t(port));
    if (bind(fd, (sockaddr*)&a, sizeof a)) {
      perror("bind");
      exit(2);
    }
  }
  return fd;
}

#ifdef __CUDACC__
__global__ void fill_words(uint64_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    p[i] = x;
  }
}
__global__ void read_block(const uint64_t* __restrict__ src, uint64_t* __restrict__ mapped, size_t words) {
  for (size_t i = threadIdx.x; i < words; i += blockDim.x) mapped[i] = src[i];
}
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t x = uint32_t(i) * 2654435761u ^ seed;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    p[i] = __float2bfloat16(float(x & 0xffff) / 32768.0f - 1.0f);
  }
}
#endif

// ------------------------------------------------------------------------------------------------------------- GPU load

struct GpuLoad {
  std::atomic<uint64_t> flops{0}, d2d{0}, h2d{0}, d2h{0};
  std::vector<std::thread> threads;
};

#ifdef __CUDACC__
// Keeps two of `enqueue`'s operations in flight on `s` until g_stop, adding `per` to `counter` for each completed.
template <class F>
static void looped(std::atomic<uint64_t>& counter, uint64_t per, F&& enqueue, cudaStream_t s) {
  cudaEvent_t ev[2];
  for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
  for (uint64_t i = 0; !g_stop; i++) {
    enqueue();
    CK(cudaEventRecord(ev[i & 1], s));
    if (i) {
      CK(cudaEventSynchronize(ev[(i + 1) & 1]));
      counter += per;
    }
  }
  CK(cudaStreamSynchronize(s));
}
#endif

// bf16 GEMMs (gemm=N, N^3), device-to-device copies (d2d_mib=M) and pinned H2D/D2H copies (pcie_mib=P, pcie=both|h2d|d2h)
// on device `dev`, each on its own default-priority stream and thread, until g_stop.
static void start_gpu_load(const Args& a, int dev, GpuLoad& L) {
#ifdef __CUDACC__
  const int gemm = int(a.i("gemm", 0));
  const size_t d2d_bytes = size_t(a.i("d2d_mib", 0)) << 20, pcie_bytes = size_t(a.i("pcie_mib", 0)) << 20;
  const std::string pcie = a.s("pcie", "both");
  if (gemm > 0)
    L.threads.emplace_back([=, &L] {
      CK(cudaSetDevice(dev));
      cudaStream_t s;
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      __nv_bfloat16 *A, *B, *C;
      const size_t n2 = size_t(gemm) * gemm;
      CK(cudaMalloc(&A, n2 * 2));
      CK(cudaMalloc(&B, n2 * 2));
      CK(cudaMalloc(&C, n2 * 2));
      fill_bf16<<<1024, 256, 0, s>>>(A, n2, 1);
      fill_bf16<<<1024, 256, 0, s>>>(B, n2, 2);
      cublasHandle_t h;
      cublasCreate(&h);
      cublasSetStream(h, s);
      const float alpha = 1.0f, beta = 0.0f;
      looped(L.flops, 2ull * gemm * gemm * gemm, [&] {
        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, gemm, gemm, gemm, &alpha, A, CUDA_R_16BF, gemm, B,
                                         CUDA_R_16BF, gemm, &beta, C, CUDA_R_16BF, gemm, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
          fprintf(stderr, "cublasGemmEx: %d\n", int(st));
          exit(3);
        }
      }, s);
      cublasDestroy(h);
    });
  if (d2d_bytes)
    L.threads.emplace_back([=, &L] {
      CK(cudaSetDevice(dev));
      cudaStream_t s;
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      uint8_t *x, *y;
      CK(cudaMalloc(&x, d2d_bytes));
      CK(cudaMalloc(&y, d2d_bytes));
      looped(L.d2d, 2 * d2d_bytes, [&] { CK(cudaMemcpyAsync(y, x, d2d_bytes, cudaMemcpyDeviceToDevice, s)); }, s);
    });
  for (int dir = 0; dir < 2 && pcie_bytes; dir++) {
    if ((dir == 0 && pcie == "d2h") || (dir == 1 && pcie == "h2d")) continue;
    L.threads.emplace_back([=, &L] {
      CK(cudaSetDevice(dev));
      cudaStream_t s;
      CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      uint8_t *hbuf, *dbuf;
      CK(cudaHostAlloc(&hbuf, pcie_bytes, cudaHostAllocDefault));
      memset(hbuf, 1, pcie_bytes);
      CK(cudaMalloc(&dbuf, pcie_bytes));
      if (dir == 0)
        looped(L.h2d, pcie_bytes, [&] { CK(cudaMemcpyAsync(dbuf, hbuf, pcie_bytes, cudaMemcpyHostToDevice, s)); }, s);
      else
        looped(L.d2h, pcie_bytes, [&] { CK(cudaMemcpyAsync(hbuf, dbuf, pcie_bytes, cudaMemcpyDeviceToHost, s)); }, s);
    });
  }
#else
  (void)a, (void)dev, (void)L;
#endif
}

// ---------------------------------------------------------------------------------------------------------------- serve

static int serve(const Args& a) {
  const int port = int(a.i("port", 47310));
  const size_t block = size_t(a.i("block", 4096));
  const size_t gpu_bytes = size_t(a.i("gpu_mib", 8192)) << 20;
  const size_t host_bytes = size_t(a.i("host_mib", 1024)) << 20;
  const uint64_t gblocks = gpu_bytes / block, hblocks = host_bytes / block;
  (void)gblocks;
  if (block % 8 || block > (MIN_DGRAM - sizeof(Hdr)) * MAX_CHUNKS) {
    fprintf(stderr, "block must be a multiple of 8 and at most %zu\n", (MIN_DGRAM - sizeof(Hdr)) * MAX_CHUNKS);
    return 2;
  }
  const std::vector<int> allowed = allowed_cpus();
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);

  std::vector<uint64_t> host(host_bytes / 8);
  for (size_t i = 0; i < host.size(); i++) host[i] = mix(i, HOST_SEED);
  const uint8_t* hbase = reinterpret_cast<const uint8_t*>(host.data());

#ifdef __CUDACC__
  CK(cudaSetDevice(0));
  uint8_t* dbase;
  CK(cudaMalloc(&dbase, gpu_bytes));
  fill_words<<<1024, 256>>>(reinterpret_cast<uint64_t*>(dbase), gpu_bytes / 8, GPU_SEED);
  CK(cudaDeviceSynchronize());
  int least, greatest;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t stream;
  CK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
  cudaEvent_t ev;
  CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  uint8_t *pinned, *mapped, *mapped_dev;
  CK(cudaHostAlloc(&pinned, block, cudaHostAllocDefault));
  CK(cudaHostAlloc(&mapped, block, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&mapped_dev, mapped, 0));
  for (int i = 0; i < 1000; i++) {
    CK(cudaMemcpyAsync(pinned, dbase + (mix(i, 1) % gblocks) * block, block, cudaMemcpyDeviceToHost, stream));
    read_block<<<1, 256, 0, stream>>>(reinterpret_cast<const uint64_t*>(dbase + (mix(i, 2) % gblocks) * block),
                                      reinterpret_cast<uint64_t*>(mapped_dev), block / 8);
    CK(cudaEventRecord(ev, stream));
    CK(cudaEventSynchronize(ev));
  }
  const bool on_gpu = true;
#else
  const bool on_gpu = false;
#endif
  // A load in this process shares the responder's CUDA context, so stream priority, not time slicing, orders them.
  GpuLoad inload;
  start_gpu_load(a, 0, inload);
  const double load_t0 = wall_s();
  // Pinned only now, so the threads the CUDA runtime started keep the process's CPUs; then moved off ours.
  const int cpu = pin(a.s("cpu", "none"));
  const int moved = isolate(cpu, allowed);

  int fd = udp_socket(port);
  std::vector<sockaddr_in> peers;
  for (auto& p : a.all("peer")) peers.push_back(addr(p));
  std::vector<uint64_t> peer_seen(peers.size(), 0);
  {
    std::ofstream r(a.s("ready", "/dev/null"));
    r << "{\"port\": " << port << ", \"cpu\": " << cpu << ", \"gpu\": " << (on_gpu ? "true" : "false") << "}\n";
  }

  uint64_t count[6] = {}, bad = 0, send_err = 0, hellos = 0, spins = 0, last_req = 0, last_hello = 0;
  uint64_t max_server[6] = {}, max_device = 0;
  long last_cs = nivcsw(), preempted = 0;
  Req req;
  Hdr hdr[MAX_CHUNKS];
  iovec iov[2 * MAX_CHUNKS];
  mmsghdr msgs[MAX_CHUNKS];
  while (true) {
    sockaddr_in from{};
    socklen_t fl = sizeof from;
    ssize_t n = recvfrom(fd, &req, sizeof req, MSG_DONTWAIT, (sockaddr*)&from, &fl);
    if (n < 0) {
      if ((++spins & 4095) == 0) {
        if (g_stop) break;
        uint64_t t = now_ns();
        // A peer's stateful ZK server passes our answers only after we have sent to it; a hello opens that path, and is
        // sent only while no challenge has arrived for 5 ms.
        if (t - last_req > 5000000 && t - last_hello > 100000000) {
          for (size_t p = 0; p < peers.size(); p++)
            if (t - peer_seen[p] > 1000000000ull) {
              Req h{MAGIC, OP_HELLO, 0, 0};
              sendto(fd, &h, sizeof h, 0, (sockaddr*)&peers[p], sizeof peers[p]);
              hellos++;
            }
          last_hello = t;
        }
      }
      continue;
    }
    const uint64_t t0 = now_ns();
    last_req = t0;
    const size_t dgram = (req.op >> 16) ? (req.op >> 16) : DGRAM;
    req.op &= 0xffff;
    if (n != sizeof(Req) || req.magic != MAGIC || req.op < OP_GPU || req.op > OP_GPU_SM || dgram < MIN_DGRAM ||
        dgram > DGRAM) {
      bad++;
      continue;
    }
    const size_t chunk = dgram - sizeof(Hdr);
    for (size_t p = 0; p < peers.size(); p++)
      if (peers[p].sin_addr.s_addr == from.sin_addr.s_addr && peers[p].sin_port == from.sin_port) peer_seen[p] = t0;
    const long cs = nivcsw();
    const uint32_t flags = cs != last_cs ? F_RESP_PREEMPTED : 0;
    preempted += cs != last_cs;
    last_cs = cs;
    count[req.op]++;
    if (req.op == OP_HELLO) continue;
    const uint8_t* payload = nullptr;
    size_t plen = 0;
    uint32_t device = 0;
    if (gpu_op(req.op) && on_gpu) {
#ifdef __CUDACC__
      const uint8_t* src = dbase + (req.idx % gblocks) * block;
      if (req.op == OP_GPU) {
        CK(cudaMemcpyAsync(pinned, src, block, cudaMemcpyDeviceToHost, stream));
      } else {
        read_block<<<1, 256, 0, stream>>>(reinterpret_cast<const uint64_t*>(src), reinterpret_cast<uint64_t*>(mapped_dev),
                                          block / 8);
      }
      CK(cudaEventRecord(ev, stream));
      cudaError_t e;
      while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) {
      }
      CK(e);
      device = uint32_t(now_ns() - t0);
      payload = req.op == OP_GPU ? pinned : mapped;
      plen = block;
#endif
    } else if (gpu_op(req.op) || req.op == OP_HOST) {
      payload = hbase + (req.idx % hblocks) * block;
      plen = block;
    }
    const int nchunks = plen ? int((plen + chunk - 1) / chunk) : 1;
    const uint32_t server = uint32_t(now_ns() - t0);
    for (int c = 0; c < nchunks; c++) {
      hdr[c] = Hdr{MAGIC, req.seq, uint16_t(req.op), uint8_t(c), uint8_t(nchunks), device, server, flags};
      iov[2 * c] = iovec{&hdr[c], sizeof(Hdr)};
      size_t off = size_t(c) * chunk, len = plen ? std::min(chunk, plen - off) : 0;
      iov[2 * c + 1] = iovec{const_cast<uint8_t*>(payload) + off, len};
      msgs[c].msg_hdr = msghdr{&from, fl, &iov[2 * c], size_t(len ? 2 : 1), nullptr, 0, 0};
    }
    if (sendmmsg(fd, msgs, nchunks, 0) != nchunks) send_err++;
    max_server[req.op] = std::max<uint64_t>(max_server[req.op], server);
    max_device = std::max<uint64_t>(max_device, device);
  }
  for (auto& t : inload.threads) t.join();
  const double load_s = wall_s() - load_t0;
  rusage ru;
  getrusage(RUSAGE_THREAD, &ru);
  std::ofstream st(a.s("stats", "/dev/stdout"));
  st << "{\"cpu\": " << cpu << ", \"gpu\": " << (on_gpu ? "true" : "false") << ", \"block\": " << block
     << ", \"gpu_bytes\": " << gpu_bytes << ", \"answered\": {\"gpu\": " << count[OP_GPU] << ", \"gpu_sm\": "
     << count[OP_GPU_SM] << ", \"host\": " << count[OP_HOST] << ", \"ping\": " << count[OP_PING] << ", \"hello\": "
     << count[OP_HELLO] << "}, \"bad\": " << bad << ", \"send_errors\": " << send_err << ", \"hellos_sent\": " << hellos
     << ", \"threads_moved_off_cpu\": " << moved << ", \"max_server_ns\": {\"gpu\": " << max_server[OP_GPU]
     << ", \"gpu_sm\": " << max_server[OP_GPU_SM] << ", \"host\": " << max_server[OP_HOST] << ", \"ping\": "
     << max_server[OP_PING] << "}, \"max_device_ns\": " << max_device
     << ", \"requests_after_preemption\": " << preempted << ", \"nivcsw\": " << ru.ru_nivcsw
     << ", \"nvcsw\": " << ru.ru_nvcsw << ", \"in_process_load\": {\"seconds\": " << load_s << ", \"gemm_tflops\": "
     << inload.flops / load_s / 1e12 << ", \"d2d_hbm_gbps\": " << inload.d2d / load_s / 1e9 << ", \"h2d_gbps\": "
     << inload.h2d / load_s / 1e9 << ", \"d2h_gbps\": " << inload.d2h / load_s / 1e9 << "}}\n";
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------- probe

static bool expected_block(const uint8_t* got, uint32_t op, uint64_t idx, size_t block, uint64_t gblocks,
                           uint64_t hblocks, bool gpu_answer) {
  const bool from_gpu = gpu_op(op) && gpu_answer;
  const uint64_t b = idx % (from_gpu ? gblocks : hblocks), seed = from_gpu ? GPU_SEED : HOST_SEED;
  for (size_t w = 0; w < block / 8; w++) {
    uint64_t x;
    memcpy(&x, got + 8 * w, 8);
    if (x != mix(b * (block / 8) + w, seed)) return false;
  }
  return true;
}

static int probe(const Args& a) {
  const sockaddr_in to = addr(a.s("to"));
  const int lport = int(a.i("lport", 0));
  const size_t block = size_t(a.i("block", 4096));
  const uint64_t gblocks = (uint64_t(a.i("gpu_mib", 8192)) << 20) / block;
  const uint64_t hblocks = (uint64_t(a.i("host_mib", 1024)) << 20) / block;
  const long long pings = a.i("pings", 20000), rounds = a.i("rounds", 200000), warm = a.i("warm", 2000);
  const long long k = a.i("k", 111);
  std::vector<uint32_t> ops;
  for (int c : cpu_list(a.s("ops", "1,5,1,5,2"))) ops.push_back(uint32_t(c));
  const uint64_t gap_ns = uint64_t(a.i("gap_us", 2000)) * 1000, timeout_ns = uint64_t(a.i("timeout_ms", 50)) * 1000000;
  const std::string out = a.s("out");
  const int cpu = pin(a.s("cpu", "auto"));

  int fd = udp_socket(lport);
  if (connect(fd, (const sockaddr*)&to, sizeof to)) {
    perror("connect");
    return 2;
  }
  const size_t dgram = size_t(a.i("dgram", DGRAM)), chunk = dgram - sizeof(Hdr);
  if (dgram < MIN_DGRAM || dgram > DGRAM || block > chunk * MAX_CHUNKS) {
    fprintf(stderr, "dgram must be in [%zu, %zu] and hold the block in %d datagrams\n", MIN_DGRAM, DGRAM, MAX_CHUNKS);
    return 2;
  }
  std::vector<uint8_t> buf(2048), data(block + chunk);
  uint32_t seq = 0;
  uint64_t stale = 0, short_chunks = 0;
  bool gpu_answer = true;

  // One challenge: the record, and the payload reassembled in `data`.
  auto one = [&](uint32_t op, uint32_t idx) -> Rec {
    seq++;
    Req req{MAGIC, op | (dgram == DGRAM ? 0u : uint32_t(dgram) << 16), seq, idx};
    const long cs0 = nivcsw();
    const uint64_t t0 = now_ns();
    send(fd, &req, sizeof req, 0);
    uint64_t got = 0, need = ~0ull, t1 = 0;
    Hdr h{};
    while (true) {
      ssize_t n = recv(fd, buf.data(), buf.size(), MSG_DONTWAIT);
      if (n < 0) {
        if (now_ns() - t0 > timeout_ns) break;
        continue;
      }
      if (size_t(n) < sizeof(Hdr)) {
        stale++;
        continue;
      }
      memcpy(&h, buf.data(), sizeof h);
      if (h.magic != MAGIC || h.seq != seq || h.op != op || h.nchunks == 0 || h.nchunks > MAX_CHUNKS) {
        stale++;
        continue;
      }
      const size_t len = size_t(n) - sizeof(Hdr), off = size_t(h.chunk) * chunk;
      const size_t want = op == OP_PING ? 0 : std::min(chunk, block - std::min(block, off));
      if (len != want) short_chunks++;
      if (len && off + len <= data.size()) memcpy(data.data() + off, buf.data() + sizeof(Hdr), len);
      got |= 1ull << h.chunk;
      need = h.nchunks == 64 ? ~0ull : (1ull << h.nchunks) - 1;
      if (got == need) {
        t1 = now_ns();
        break;
      }
    }
    const long cs1 = nivcsw();
    Rec r{};
    r.op = uint16_t(op);
    r.server_ns = h.server_ns;
    r.device_ns = h.device_ns;
    r.flags = uint16_t(h.flags & F_RESP_PREEMPTED) | (cs1 != cs0 ? F_CHAL_PREEMPTED : 0);
    if (!t1) {
      r.elapsed_ns = UINT32_MAX;
      r.flags |= F_LOST;
    } else {
      r.elapsed_ns = uint32_t(std::min<uint64_t>(t1 - t0, UINT32_MAX - 1));
      if (gpu_op(op) && h.device_ns == 0) gpu_answer = false;
      if (op != OP_PING && !expected_block(data.data(), op, idx, block, gblocks, hblocks, gpu_answer)) r.flags |= F_BAD_DATA;
    }
    return r;
  };

  // Open both ZK servers' state for this flow: hellos and pings until a ping comes back.
  const uint64_t punch0 = now_ns();
  bool open = false;
  while (now_ns() - punch0 < 60000000000ull && !open) {
    Req h{MAGIC, OP_HELLO, 0, 0};
    send(fd, &h, sizeof h, 0);
    Rec r = one(OP_PING, 0);
    open = !(r.flags & F_LOST);
    if (!open) usleep(20000);
  }
  if (!open) {
    fprintf(stderr, "no answer from %s within 60 s\n", a.s("to").c_str());
    return 4;
  }
  const double punch_s = 1e-9 * double(now_ns() - punch0);

  uint64_t rng = uint64_t(a.i("seed", 1));
  auto next_idx = [&]() { return uint32_t((rng = mix(rng, 0x1d)) % std::max(gblocks, hblocks)); };
  for (long long i = 0; i < warm; i++) one(OP_PING, 0);
  for (long long i = 0; i < warm; i++) one(ops[size_t(i) % ops.size()], next_idx());

  std::vector<Rec> recs;
  recs.reserve(size_t(pings + rounds));
  const double t_start = wall_s();
  const long cs_start = nivcsw();
  auto paced = [&](long long i) {
    if (k > 0 && i % k == k - 1) {
      const uint64_t t = now_ns();
      while (now_ns() - t < gap_ns) {
      }
    }
  };
  for (long long i = 0; i < pings; i++) {
    recs.push_back(one(OP_PING, 0));
    paced(i);
  }
  const uint64_t max_ns = uint64_t(a.i("max_s", 0)) * 1000000000ull, rounds0 = now_ns();
  long long done = 0;
  for (; done < rounds; done++) {
    if ((done & 1023) == 0 && max_ns && now_ns() - rounds0 > max_ns) break;
    recs.push_back(one(ops[size_t(done) % ops.size()], next_idx()));
    paced(done);
  }
  const double t_end = wall_s();

  uint64_t lost = 0, bad_data = 0;
  for (auto& r : recs) lost += (r.flags & F_LOST) != 0, bad_data += (r.flags & F_BAD_DATA) != 0;
  FILE* f = fopen((out + ".bin").c_str(), "wb");
  fwrite(recs.data(), sizeof(Rec), recs.size(), f);
  fclose(f);
  char host[256] = {};
  gethostname(host, sizeof host - 1);
  std::ofstream m(out + ".json");
  m << "{\"host\": \"" << host << "\", \"to\": \"" << a.s("to") << "\", \"cpu\": " << cpu << ", \"block\": " << block
    << ", \"pings\": " << pings << ", \"rounds\": " << done << ", \"rounds_asked\": " << rounds << ", \"max_s\": "
    << max_ns / 1000000000ull << ", \"k\": " << k << ", \"gap_us\": " << gap_ns / 1000
    << ", \"ops\": \"" << a.s("ops", "1,5,1,5,2") << "\", \"dgram\": " << dgram << ", \"timeout_ms\": " << timeout_ns / 1000000 << ", \"gpu_answer\": "
    << (gpu_answer ? "true" : "false") << ", \"lost\": " << lost << ", \"bad_data\": " << bad_data
    << ", \"stale_chunks\": " << stale << ", \"short_chunks\": " << short_chunks
    << ", \"challenger_nivcsw\": " << nivcsw() - cs_start << ", \"punch_s\": " << punch_s << ", \"start\": " << std::fixed
    << t_start << ", \"end\": " << t_end << ", \"record\": \"<IIIHH elapsed_ns server_ns device_ns op flags\"}\n";
  fprintf(stderr, "%s: %zu records, %llu lost, %llu bad\n", out.c_str(), recs.size(), (unsigned long long)lost,
          (unsigned long long)bad_data);
  return bad_data ? 5 : 0;
}

// ----------------------------------------------------------------------------------------------------------------- load

static int load(const Args& a) {
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  const std::vector<int> spin_cpus = cpu_list(a.s("spin_cpus", "")), mem_cpus = cpu_list(a.s("mem_cpus", ""));
  const size_t mem_bytes = size_t(a.i("mem_mib", 256)) << 20;
  std::atomic<uint64_t> spun{0}, copied{0};
  std::vector<std::thread> threads;
  for (int c : spin_cpus)
    threads.emplace_back([&, c] {
      pin(std::to_string(c));
      uint64_t x = uint64_t(c), local = 0;
      while (!g_stop) {
        for (int i = 0; i < 1 << 20; i++) x = x * 6364136223846793005ull + 1442695040888963407ull;
        local++;
        if ((local & 63) == 0) spun += 64;
      }
      if (x == 42) fprintf(stderr, "!");
    });
  for (int c : mem_cpus)
    threads.emplace_back([&, c] {
      pin(std::to_string(c));
      std::vector<uint8_t> src(mem_bytes, uint8_t(c)), dst(mem_bytes);
      while (!g_stop) {
        memcpy(dst.data(), src.data(), mem_bytes);
        std::swap(src, dst);
        copied += mem_bytes;
      }
    });

  GpuLoad gl;
  start_gpu_load(a, int(a.i("device", 0)), gl);

  const long long seconds = a.i("seconds", 0);
  std::ofstream rep(a.s("report", "/dev/stdout"));
  uint64_t p[6] = {};
  const double t0 = wall_s();
  double tp = t0;
  while (!g_stop && (seconds <= 0 || wall_s() - t0 < seconds)) {
    usleep(1000000);
    const double t = wall_s(), dt = t - tp;
    const uint64_t c[6] = {gl.flops.load(), gl.d2d.load(), gl.h2d.load(), gl.d2h.load(), spun.load(), copied.load()};
    rep << "{\"t\": " << std::fixed << t << ", \"gemm_tflops\": " << (c[0] - p[0]) / dt / 1e12
        << ", \"d2d_hbm_gbps\": " << (c[1] - p[1]) / dt / 1e9 << ", \"h2d_gbps\": " << (c[2] - p[2]) / dt / 1e9
        << ", \"d2h_gbps\": " << (c[3] - p[3]) / dt / 1e9 << ", \"spin_gloops\": " << (c[4] - p[4]) * double(1 << 20) / dt / 1e9
        << ", \"host_memcpy_gbps\": " << (c[5] - p[5]) / dt / 1e9 << "}" << std::endl;
    std::copy(c, c + 6, p);
    tp = t;
  }
  g_stop = 1;
  for (auto& t : threads) t.join();
  for (auto& t : gl.threads) t.join();
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  Args a(argc, argv);
  if (cmd == "serve") return serve(a);
  if (cmd == "probe") return probe(a);
  if (cmd == "load") return load(a);
  fprintf(stderr, "usage: lat serve|probe|load key=value...\n");
  return 2;
}
