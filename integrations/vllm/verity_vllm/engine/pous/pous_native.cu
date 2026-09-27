// POUS native code for the serving process: a batched SHAKE256 kernel (one thread per message) and the audit responder,
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

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

__device__ __forceinline__ void keccak_f(uint64_t st[25]) {
  const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    uint64_t t = st[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      int j = piln[i];
      uint64_t b0 = st[j];
      st[j] = rotl64(t, rotc[i]);
      t = b0;
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      uint64_t b[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) b[i] = st[j + i];
#pragma unroll
      for (int i = 0; i < 5; ++i) st[j + i] ^= (~b[(i + 1) % 5]) & b[(i + 2) % 5];
    }
    st[0] ^= KECCAK_RC[r];
  }
}

constexpr int RATE = 136;  // SHAKE256: 1088-bit rate

// Row n: out[n, :olen] (^)= SHAKE256(prefix[n, :plen] || body[n, :blen])[0:olen].
__global__ void shake256_kernel(const uint8_t* __restrict__ prefix, int plen, int64_t prefix_stride,
                                const uint8_t* __restrict__ body, int64_t blen, int64_t body_stride,
                                uint8_t* __restrict__ out, int64_t olen, int64_t out_stride, int xor_out, int64_t rows) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= rows) return;
  const uint8_t* pre = prefix + n * prefix_stride;
  const uint8_t* bod = body + n * body_stride;
  uint8_t* dst = out + n * out_stride;
  const int64_t len = plen + blen;
  const int64_t nblk = len / RATE + 1;
  uint64_t st[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) st[i] = 0;
  for (int64_t b = 0; b < nblk; ++b) {
    const int64_t base = b * RATE;
    if (base >= plen && base + RATE <= len) {
      const uint8_t* p = bod + (base - plen);
#pragma unroll
      for (int w = 0; w < RATE / 8; ++w) {
        uint64_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) word |= (uint64_t)__ldg(p + w * 8 + k) << (8 * k);
        st[w] ^= word;
      }
    } else {
#pragma unroll
      for (int w = 0; w < RATE / 8; ++w) {
        uint64_t word = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int64_t pos = base + w * 8 + k;
          uint8_t byte = 0;
          if (pos < plen) byte = pre[pos];
          else if (pos < len) byte = __ldg(bod + (pos - plen));
          if (pos == len) byte ^= 0x1F;
          if (b == nblk - 1 && w == RATE / 8 - 1 && k == 7) byte ^= 0x80;
          word |= (uint64_t)byte << (8 * k);
        }
        st[w] ^= word;
      }
    }
    keccak_f(st);
  }
  for (int64_t o = 0; o < olen; o += RATE) {
    if (o > 0) keccak_f(st);
    const int64_t take = olen - o < RATE ? olen - o : RATE;
#pragma unroll
    for (int w = 0; w < RATE / 8; ++w) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int64_t i = w * 8 + k;
        if (i < take) {
          uint8_t byte = (uint8_t)(st[w] >> (8 * k));
          dst[o + i] = xor_out ? (uint8_t)(dst[o + i] ^ byte) : byte;
        }
      }
    }
  }
}

static void check_rows(const torch::Tensor& t, const char* name, int64_t rows) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.dim() == 2, name, " must be a 2-D uint8 CUDA tensor");
  TORCH_CHECK(t.size(0) == rows || (t.size(0) == 1 && t.stride(0) == 0) || t.size(1) == 0, name, " has the wrong row count");
  TORCH_CHECK(t.size(1) == 0 || t.stride(1) == 1, name, " rows must be contiguous");
}

// out (^)= SHAKE256(prefix || body)[0:out.size(1)] row by row, on the current stream.  `prefix` may have stride(0) == 0.
void shake256_rows(torch::Tensor prefix, torch::Tensor body, torch::Tensor out, bool xor_out) {
  const int64_t rows = out.size(0);
  check_rows(prefix, "prefix", rows);
  check_rows(body, "body", rows);
  check_rows(out, "out", rows);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  const int threads = 32;
  const int64_t blocks = (rows + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream();
  shake256_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
      prefix.size(1) ? prefix.data_ptr<uint8_t>() : nullptr, (int)prefix.size(1), prefix.size(0) == 1 ? 0 : prefix.stride(0),
      body.data_ptr<uint8_t>(), body.size(1), body.stride(0), out.data_ptr<uint8_t>(), out.size(1), out.stride(0),
      xor_out ? 1 : 0, rows);
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
  pybind11::class_<Responder>(m, "Responder")
      .def(pybind11::init<int64_t, int64_t, int64_t, int, std::string, int>())
      .def("start", &Responder::start)
      .def("stop", &Responder::stop)
      .def("answered", &Responder::answered)
      .def("pings", &Responder::pings)
      .def("port", &Responder::port);
}
