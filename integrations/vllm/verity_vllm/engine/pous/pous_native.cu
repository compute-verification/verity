// POUS native code for the serving process: batched SHAKE256 and Feistel-10 kernels (one warp per message) and the audit responder,
// a native thread that answers challenges from C in device memory over TCP, with no Python and no GIL on the path.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
__constant__ int KECCAK_RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39,
                                   41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

constexpr int RATE = 136;  // SHAKE256: 1088-bit rate
constexpr int RATE_WORDS = RATE / 8;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int MAX_TWEAK = 62;

// One warp holds one Keccak state: lane x + 5y holds word (x, y); lanes 25..31 shadow lane % 25 and are never read.
struct Lane {
  int t1, t2, t3, t4, dm, dp, rho, pi, c1, c2;
  bool first;
};

__device__ __forceinline__ Lane lane_of(int lane) {
  const int v = lane % 25, x = v % 5, y = v / 5;
  Lane l;
  l.t1 = x + 5 * ((y + 1) % 5);
  l.t2 = x + 5 * ((y + 2) % 5);
  l.t3 = x + 5 * ((y + 3) % 5);
  l.t4 = x + 5 * ((y + 4) % 5);
  l.dm = (x + 4) % 5 + 5 * y;
  l.dp = (x + 1) % 5 + 5 * y;
  l.rho = KECCAK_RHO[v];
  l.pi = (x + 3 * y) % 5 + 5 * x;  // pi: word (x, y) moves to (y, 2x + 3y), so lane (X, Y) takes ((X + 3Y) % 5, X)
  l.c1 = (x + 1) % 5 + 5 * y;
  l.c2 = (x + 2) % 5 + 5 * y;
  l.first = lane == 0;
  return l;
}

__device__ __forceinline__ uint64_t shfl64(uint64_t v, int src) { return __shfl_sync(0xffffffffu, v, src); }
__device__ __forceinline__ uint64_t rotlv(uint64_t v, int r) { return r ? (v << r) | (v >> (64 - r)) : v; }

__device__ __forceinline__ uint64_t keccak_f(uint64_t a, const Lane& l) {
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    const uint64_t c = a ^ shfl64(a, l.t1) ^ shfl64(a, l.t2) ^ shfl64(a, l.t3) ^ shfl64(a, l.t4);
    const uint64_t d = shfl64(c, l.dm) ^ rotlv(shfl64(c, l.dp), 1);
    a = rotlv(a ^ d, l.rho);
    const uint64_t b = shfl64(a, l.pi);
    a = b ^ (~shfl64(b, l.c1) & shfl64(b, l.c2));
    if (l.first) a ^= KECCAK_RC[r];
  }
  return a;
}

// dst[0:olen] (^)= SHAKE256(pre[0:plen] || body[0:blen])[0:olen], computed by the calling warp.
__device__ void shake_warp(const uint8_t* pre, int plen, const uint8_t* body, int64_t blen, uint8_t* dst, int64_t olen,
                           bool xor_out, const Lane& l, int lane) {
  const int64_t len = plen + blen;
  const int64_t nblk = len / RATE + 1;
  uint64_t a = 0;
  for (int64_t b = 0; b < nblk; ++b) {
    if (lane < RATE_WORDS) {
      const int64_t base = b * RATE + lane * 8;
      uint64_t w = 0;
      if (base >= plen && base + 8 <= len) {
        const uint8_t* p = body + (base - plen);
#pragma unroll
        for (int k = 0; k < 8; ++k) w |= (uint64_t)p[k] << (8 * k);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int64_t pos = base + k;
          uint8_t byte = 0;
          if (pos < plen) byte = pre[pos];
          else if (pos < len) byte = body[pos - plen];
          if (pos == len) byte ^= 0x1F;
          w |= (uint64_t)byte << (8 * k);
        }
        if (b == nblk - 1 && lane == RATE_WORDS - 1) w ^= 0x80ULL << 56;
      }
      a ^= w;
    }
    a = keccak_f(a, l);
  }
  for (int64_t o = 0; o < olen; o += RATE) {
    if (o > 0) a = keccak_f(a, l);
    if (lane < RATE_WORDS) {
      const int64_t base = o + lane * 8;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t pos = base + k;
        if (pos < olen) {
          const uint8_t byte = (uint8_t)(a >> (8 * k));
          dst[pos] = xor_out ? (uint8_t)(dst[pos] ^ byte) : byte;
        }
      }
    }
  }
  __syncwarp();
}

// Row n: out[n, :olen] (^)= SHAKE256(prefix[n, :plen] || body[n, :blen])[0:olen]; one warp per row.
__global__ void shake256_kernel(const uint8_t* prefix, int plen, int64_t prefix_stride, const uint8_t* body, int64_t blen,
                                int64_t body_stride, uint8_t* out, int64_t olen, int64_t out_stride, int xor_out, int64_t rows) {
  const int lane = threadIdx.x % 32;
  const int64_t n = (int64_t)blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
  if (n >= rows) return;
  const Lane l = lane_of(lane);
  shake_warp(prefix + n * prefix_stride, plen, body + n * body_stride, blen, out + n * out_stride, olen, xor_out != 0, l, lane);
}

// Ten Feistel rounds in place on row n of x (halves A || B of `half` bytes): round i computes
// F_i = SHAKE256(domain || u8(i) || tweak[n] || src)[0:half] and xors it into the other half, odd rounds A ^= F_i(B),
// even rounds B ^= F_i(A); the inverse runs i = 10 .. 1.  That is (L, R) -> (R, L ^ F_i(R)) with the output left as A || B.
__global__ void feistel_kernel(uint8_t* x, int64_t x_stride, int64_t half, const uint8_t* tweak, int tlen, int64_t tweak_stride,
                               int domain, int inverse, int64_t rows) {
  __shared__ uint8_t prefix[WARPS_PER_BLOCK][MAX_TWEAK + 2];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int64_t n = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp;
  if (n >= rows) return;
  const Lane l = lane_of(lane);
  uint8_t* pre = prefix[warp];
  for (int j = lane; j < tlen; j += 32) pre[2 + j] = tweak[n * tweak_stride + j];
  if (lane == 0) pre[0] = (uint8_t)domain;
  uint8_t* a = x + n * x_stride;
  uint8_t* b = a + half;
  for (int step = 0; step < 10; ++step) {
    const int i = inverse ? 10 - step : step + 1;
    if (lane == 0) pre[1] = (uint8_t)i;
    __syncwarp();
    if (i % 2) shake_warp(pre, 2 + tlen, b, half, a, half, true, l, lane);
    else shake_warp(pre, 2 + tlen, a, half, b, half, true, l, lane);
  }
}

static void check_rows(const torch::Tensor& t, const char* name, int64_t rows) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.dim() == 2, name, " must be a 2-D uint8 CUDA tensor");
  TORCH_CHECK(t.size(0) == rows || (t.size(0) == 1 && t.stride(0) == 0) || t.size(1) == 0, name, " has the wrong row count");
  TORCH_CHECK(t.size(1) == 0 || t.stride(1) == 1, name, " rows must be contiguous");
}

static unsigned grid_of(int64_t rows) { return (unsigned)((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK); }

// out (^)= SHAKE256(prefix || body)[0:out.size(1)] row by row, on the current stream.  `prefix` may have stride(0) == 0.
void shake256_rows(torch::Tensor prefix, torch::Tensor body, torch::Tensor out, bool xor_out) {
  const int64_t rows = out.size(0);
  check_rows(prefix, "prefix", rows);
  check_rows(body, "body", rows);
  check_rows(out, "out", rows);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  shake256_kernel<<<grid_of(rows), 32 * WARPS_PER_BLOCK, 0, stream>>>(
      prefix.size(1) ? prefix.data_ptr<uint8_t>() : nullptr, (int)prefix.size(1), prefix.size(0) == 1 ? 0 : prefix.stride(0),
      body.data_ptr<uint8_t>(), body.size(1), body.stride(0), out.data_ptr<uint8_t>(), out.size(1), out.stride(0),
      xor_out ? 1 : 0, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Feistel-10 in place on the rows of x ([R, 2h]) with round functions SHAKE256(domain || u8(i) || tweak[r] || .).
void feistel_rows(torch::Tensor x, torch::Tensor tweak, int64_t domain, bool inverse) {
  const int64_t rows = x.size(0);
  check_rows(x, "x", rows);
  check_rows(tweak, "tweak", rows);
  TORCH_CHECK(x.size(1) % 2 == 0, "a balanced Feistel network needs an even width");
  TORCH_CHECK(tweak.size(1) <= MAX_TWEAK, "tweak longer than ", MAX_TWEAK, " bytes");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  feistel_kernel<<<grid_of(rows), 32 * WARPS_PER_BLOCK, 0, stream>>>(
      x.data_ptr<uint8_t>(), x.stride(0), x.size(1) / 2, tweak.size(1) ? tweak.data_ptr<uint8_t>() : nullptr, (int)tweak.size(1),
      tweak.size(0) == 1 ? 0 : tweak.stride(0), (int)domain, inverse ? 1 : 0, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the audit responder ----------------------------------------------------------------------------------------------
// Wire (little-endian): request = u64 op, u64 arg.  op 1 = challenge block `arg`, op 2 = ping (no device work), op 3 = close.
// Response = u64 status (0 ok, 1 index out of range, 2 CUDA error), u64 device_ns (copy start to end event), u64 server_ns
// (request received to response ready), then `block_bytes` payload bytes.

static bool read_all(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r <= 0) return false;
    p += r;
    n -= (size_t)r;
  }
  return true;
}

static bool write_all(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  while (n) {
    ssize_t r = ::send(fd, p, n, MSG_NOSIGNAL);
    if (r <= 0) return false;
    p += r;
    n -= (size_t)r;
  }
  return true;
}

class Responder {
 public:
  Responder(int64_t base, int64_t blocks, int64_t block_bytes, int device, std::string host, int port)
      : base_(reinterpret_cast<const uint8_t*>(base)), blocks_(blocks), block_bytes_(block_bytes), device_(device),
        host_(std::move(host)), port_(port) {}
  ~Responder() { stop(); }

  int start() {
    TORCH_CHECK(!running_.load(), "responder already running");
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(listen_fd_ >= 0, "socket() failed");
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    TORCH_CHECK(::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) == 1, "bad host ", host_);
    TORCH_CHECK(::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) == 0, "bind() failed on ", host_, ":", port_);
    TORCH_CHECK(::listen(listen_fd_, 4) == 0, "listen() failed");
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, (sockaddr*)&addr, &len);
    port_ = ntohs(addr.sin_port);
    running_.store(true);
    thread_ = std::thread([this] { serve(); });
    return port_;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    int c = conn_fd_.load();
    if (c >= 0) ::shutdown(c, SHUT_RDWR);
    if (thread_.joinable()) thread_.join();
  }

  int64_t answered() const { return answered_.load(); }
  int64_t pings() const { return pings_.load(); }
  int port() const { return port_; }

 private:
  void serve() {
    cudaSetDevice(device_);
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    uint8_t* host = nullptr;
    cudaHostAlloc((void**)&host, (size_t)block_bytes_ + 24, cudaHostAllocDefault);
    std::memset(host, 0, (size_t)block_bytes_ + 24);
    while (running_.load()) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) continue;
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      conn_fd_.store(fd);
      uint64_t req[2];
      while (running_.load() && read_all(fd, req, sizeof(req))) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t* head = reinterpret_cast<uint64_t*>(host);
        uint8_t* payload = host + 24;
        head[0] = 0;
        head[1] = 0;
        if (req[0] == 3) break;
        if (req[0] == 1) {
          if (req[1] >= (uint64_t)blocks_) {
            head[0] = 1;
          } else {
            cudaEventRecord(ev0, stream);
            cudaMemcpyAsync(payload, base_ + (size_t)req[1] * (size_t)block_bytes_, (size_t)block_bytes_,
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev1, stream);
            if (cudaEventSynchronize(ev1) != cudaSuccess) {
              head[0] = 2;
            } else {
              float ms = 0.f;
              cudaEventElapsedTime(&ms, ev0, ev1);
              head[1] = (uint64_t)((double)ms * 1e6);
            }
            answered_.fetch_add(1);
          }
        } else {
          pings_.fetch_add(1);
        }
        head[2] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        if (!write_all(fd, host, (size_t)block_bytes_ + 24)) break;
      }
      conn_fd_.store(-1);
      ::close(fd);
    }
    cudaFreeHost(host);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
  }

  const uint8_t* base_;
  int64_t blocks_, block_bytes_;
  int device_;
  std::string host_;
  int port_;
  int listen_fd_ = -1;
  std::atomic<int> conn_fd_{-1};
  std::atomic<bool> running_{false};
  std::atomic<int64_t> answered_{0}, pings_{0};
  std::thread thread_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shake256_rows", &shake256_rows, "out (^)= SHAKE256(prefix || body)[0:out.size(1)], row by row");
  m.def("feistel_rows", &feistel_rows, "Feistel-10 in place, SHAKE256 round functions under a domain byte and per-row tweak");
  pybind11::class_<Responder>(m, "Responder")
      .def(pybind11::init<int64_t, int64_t, int64_t, int, std::string, int>())
      .def("start", &Responder::start)
      .def("stop", &Responder::stop)
      .def("answered", &Responder::answered)
      .def("pings", &Responder::pings)
      .def("port", &Responder::port);
}
