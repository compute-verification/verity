// The honest-latency soak's native halves (`pipeline/pous_soak.py`), built with g++ against the CUDA runtime and loaded with
// ctypes: the soak responder (the series' native responder, `engine/pous/pous_native.cu` Responder, with contiguous reads of any
// length, a per-connection configuration and the kernel's software RX/TX timestamps) and the tuned verifier's hot loop.
//
// Wire (little-endian).  Request = u64 op | nbytes << 8, u64 arg.  op 2 = ping, op 3 = close, op 4 = `nbytes` contiguous bytes of
// C from block `arg`, op 5 = configure the connection (arg bit 0: tuned, bit 1: timestamping off, bit 2: zero-copy sends).
// Response = HEADER_WORDS u64,
// then `nbytes` bytes (configure: none; a ping's are the staging buffer's, no device work):
//   0 status (0 ok, 1 out of range, 2 CUDA error, 3 bad op)   1 device_ns (copy start to end event; configure: mlock call ns)
//   2 server_ns (request read to response ready, steady clock; configure: effective SO_SNDBUF)
//   3 kernel RX timestamp of the request (CLOCK_REALTIME ns, SO_TIMESTAMPING software; 0 when none)
//   4 realtime when the request was read   5 realtime just before the response's first send
//   6 the newest kernel TX timestamp read from the error queue   7 its key (the TCP byte offset mod 2^32, SOF_TIMESTAMPING_OPT_ID)
//   8 send calls of the previous response  9 realtime when the previous response's last send returned
//   10 flags: 1 tuned, 2 mlockall in effect, 4 RX timestamp seen, 8 TX timestamps on, 16 the last QUICKACK set held
//   11 the core that answered (sched_getcpu)   12 answers on this connection   13 nbytes
//   14 the key of the previous response's last byte (configure: effective SO_RCVBUF)   15 MAGIC
//   16 realtime before the copy's first event   17 realtime when the copy's end event first queried done
//   18 EAGAINs of the previous response's sends   19 op   20 TX timestamps read from the error queue since the last header
//   21 SCHED_FIFO: 1 in effect, else the errno of the last attempt << 8
//   22 the kernel TX timestamp of the previous response's last byte, matched by key (0 when not yet read)
//   23 the same for the response before it (a timestamp read only after the next request arrived)
//   24 zero-copy: ns this response waited, after the request was read, for the previous response's completion (the staging
//      buffer is reused only once the kernel released it)   25 zero-copy completions received on this connection
//   flags also: 32 timestamping off (the twin cells), 64 SCHED_FIFO in effect, 128 zero-copy sends (SO_ZEROCOPY, MSG_ZEROCOPY),
//   256 a completion so far reported the kernel copied the data (SO_EE_CODE_ZEROCOPY_COPIED: loopback always does, on delivery)
// The kernel TX timestamp of a response is only known after its send, so a header reports the newest one read so far; the error
// queue is drained while the thread waits for the next request, never between a request and its response.
#ifdef SOAK_FAKE_CUDA
// Host memory in place of the device (tests): the copy is a memcpy, the events read the steady clock.
#include <chrono>
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef long long* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorNotReady = 600, cudaStreamNonBlocking = 1, cudaHostAllocDefault = 0, cudaMemcpyDeviceToHost = 2 };
static inline long long fake_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline cudaError_t cudaDeviceGetStreamPriorityRange(int* a, int* b) { *a = 0; *b = -1; return cudaSuccess; }
static inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, int, int) { *s = nullptr; return cudaSuccess; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new long long(0); return cudaSuccess; }
static inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { *e = fake_now(); return cudaSuccess; }
static inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(*b - *a) / 1e6f; return cudaSuccess; }
static inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) { std::memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaHostAlloc(void** p, size_t n, int) { *p = std::malloc(n); return *p ? cudaSuccess : 2; }
static inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
#else
#include <cuda_runtime.h>
#endif
#include <errno.h>
#include <fcntl.h>
#include <linux/errqueue.h>
#include <linux/net_tstamp.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <arpa/inet.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <algorithm>
#include <vector>

namespace {

constexpr int HEADER_WORDS = 26;
constexpr size_t HEADER_BYTES = HEADER_WORDS * 8;
constexpr uint64_t MAGIC = 0x534f414b00000002ull;  // "SOAK", version 2
constexpr uint64_t BLOCK = 65536;
constexpr int TS_FLAGS = SOF_TIMESTAMPING_RX_SOFTWARE | SOF_TIMESTAMPING_TX_SOFTWARE | SOF_TIMESTAMPING_SOFTWARE |
                         SOF_TIMESTAMPING_OPT_ID | SOF_TIMESTAMPING_OPT_TSONLY;
constexpr int TUNED_BUF = 1 << 20;

// TUNED_BUF-byte send and receive buffers where the kernel grants them whole (net.core.[wr]mem_max >= TUNED_BUF / 2).  A
// smaller grant would lock the buffer below what autotuning reaches (tcp_[wr]mem's max): a loopback peer's 64 KiB segments then
// close the receive window around a lost one, and the connection waits out the probe timer's backoff.  Else autotuned.
static bool grants_tuned_buf(const char* path) {
  long v = 0;
  if (FILE* f = std::fopen(path, "r")) {
    if (std::fscanf(f, "%ld", &v) != 1) v = 0;
    std::fclose(f);
  }
  return 2 * v >= TUNED_BUF;
}

static void tune_buffers(int fd) {
  int b = TUNED_BUF;
  if (grants_tuned_buf("/proc/sys/net/core/wmem_max")) ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &b, sizeof(b));
  if (grants_tuned_buf("/proc/sys/net/core/rmem_max")) ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &b, sizeof(b));
}

inline int64_t realtime_ns() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

// The software timestamp in a message's control data (scm_timestamping.ts[0]), or 0.
inline int64_t cmsg_ts(msghdr* m) {
  for (cmsghdr* c = CMSG_FIRSTHDR(m); c; c = CMSG_NXTHDR(m, c)) {
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SO_TIMESTAMPING) {
      timespec ts[3];
      std::memcpy(ts, CMSG_DATA(c), sizeof(ts));
      return (int64_t)ts[0].tv_sec * 1000000000 + ts[0].tv_nsec;
    }
  }
  return 0;
}

// One message of the error queue: 0 when it is empty, 1 with a TX timestamp in (*ts, *key), 3 with a zero-copy completion
// (its last id in *zhi, its code in *zcode) when those are asked for, 2 for any other message.
inline int read_tx(int fd, int64_t* ts, uint64_t* key, uint32_t* zhi = nullptr, uint32_t* zcode = nullptr) {
  char ctrl[512];
  msghdr m{};
  m.msg_control = ctrl;
  m.msg_controllen = sizeof(ctrl);
  if (::recvmsg(fd, &m, MSG_ERRQUEUE | MSG_DONTWAIT) < 0) return 0;
  int64_t t = 0;
  uint64_t k = 0;
  bool snd = false;
  for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SO_TIMESTAMPING) {
      timespec tt[3];
      std::memcpy(tt, CMSG_DATA(c), sizeof(tt));
      t = (int64_t)tt[0].tv_sec * 1000000000 + tt[0].tv_nsec;
    } else if ((c->cmsg_level == SOL_IP && c->cmsg_type == IP_RECVERR) ||
               (c->cmsg_level == SOL_IPV6 && c->cmsg_type == IPV6_RECVERR)) {
      sock_extended_err e;
      std::memcpy(&e, CMSG_DATA(c), sizeof(e));
      if (e.ee_origin == SO_EE_ORIGIN_TIMESTAMPING && e.ee_info == SCM_TSTAMP_SND) {
        k = e.ee_data;
        snd = true;
      } else if (e.ee_origin == SO_EE_ORIGIN_ZEROCOPY && zhi) {
        *zhi = e.ee_data;
        *zcode = e.ee_code;
        return 3;
      }
    }
  }
  if (!snd || !t) return 2;
  *ts = t;
  *key = k;
  return 1;
}

class SoakResponder {
 public:
  SoakResponder(uint64_t base, uint64_t total, int device, const char* host, int port, int cpu, uint64_t max_bytes)
      : base_(reinterpret_cast<const uint8_t*>(base)), total_(total), device_(device), host_(host), port_(port), cpu_(cpu),
        max_bytes_(max_bytes) {}

  int start() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return -errno;
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    // The kernel turns RX software timestamping on for the whole stack lazily (a deferred work item) when the first socket
    // asks; the listening socket asks for the responder's lifetime, so no connection's first requests go unstamped.
    // (RX flags only: a listening socket refuses SOF_TIMESTAMPING_OPT_ID with EINVAL.)
    int rxf = SOF_TIMESTAMPING_RX_SOFTWARE | SOF_TIMESTAMPING_SOFTWARE;
    if (::setsockopt(listen_fd_, SOL_SOCKET, SO_TIMESTAMPING, &rxf, sizeof(rxf)) != 0) return -errno;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) return -EINVAL;
    if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) != 0) return -errno;
    if (::listen(listen_fd_, 4) != 0) return -errno;
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, (sockaddr*)&addr, &len);
    port_ = ntohs(addr.sin_port);
    running_.store(true);
    thread_ = std::thread([this] { serve(); });
    while (tid_.load() == 0 && ready_err_.load() == 0) std::this_thread::yield();
    return ready_err_.load() ? -ready_err_.load() : port_;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    int c = conn_fd_.load();
    if (c >= 0) ::shutdown(c, SHUT_RDWR);
    if (thread_.joinable()) thread_.join();
  }

  int64_t stat(int which) const {
    switch (which) {
      case 0: return answered_.load();
      case 1: return pings_.load();
      case 2: return tid_.load();
      case 3: return connections_.load();
      case 4: return errors_.load();
      case 5: return mlocked_.load();
      default: return -1;
    }
  }

 private:
  // false when the connection ended (peer closed, error, or stopping)
  bool read_request(int fd, uint64_t req[2], int64_t* krx, bool* tx_pending, int64_t* ktx, uint64_t* kkey) {
    auto* p = reinterpret_cast<uint8_t*>(req);
    size_t n = 16;
    *krx = 0;
    if (ts_off_) {
      while (n) {
        ssize_t r = ::recv(fd, p, n, MSG_DONTWAIT);
        if (r > 0) {
          p += r;
          n -= (size_t)r;
        } else if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK) || !running_.load(std::memory_order_relaxed)) {
          return false;
        }
      }
      return true;
    }
    while (n) {
      char ctrl[256];
      iovec iov{p, n};
      msghdr m{};
      m.msg_iov = &iov;
      m.msg_iovlen = 1;
      m.msg_control = ctrl;
      m.msg_controllen = sizeof(ctrl);
      ssize_t r = ::recvmsg(fd, &m, MSG_DONTWAIT);
      if (r > 0) {
        if (!*krx) *krx = cmsg_ts(&m);
        p += r;
        n -= (size_t)r;
      } else if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK) || !running_.load(std::memory_order_relaxed)) {
        return false;
      } else if ((*tx_pending || zc_pending()) && n == 16) {
        drain_errq(tx_pending, ktx, kkey);
      }
    }
    return true;
  }

  bool zc_pending() const { return zc_ && zc_done_ != zc_sent_; }

  // The error queue's TX timestamps and zero-copy completions, until it is empty.
  void drain_errq(bool* tx_pending, int64_t* ktx, uint64_t* kkey) {
    int64_t t;
    uint64_t k;
    uint32_t zhi = 0, zcode = 0;
    int r2;
    while ((r2 = read_tx(fd_, &t, &k, &zhi, &zcode)) != 0) {
      if (r2 == 3) {
        zc_done_ = zhi + 1;
        ++zc_completions_;
        if (zcode & SO_EE_CODE_ZEROCOPY_COPIED) zc_copied_ = true;
        continue;
      }
      if (r2 != 1) continue;
      ++errq_reads_;
      *ktx = t;
      *kkey = k;
      for (int j = 0; j < 2; ++j)
        if (k == want_key_[j] && !want_ts_[j]) want_ts_[j] = t;
      if ((uint32_t)(k - tx_key_) < 0x80000000u) *tx_pending = false;
    }
  }

  void configure(int fd, bool tuned, bool ts_off, bool zerocopy, uint64_t* head) {
    tuned_ = tuned;
    if (zerocopy && !zc_) {
      int one = 1;
      zc_ = ::setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one)) == 0;
      if (!zc_) errors_.fetch_add(1);
    }
    if (ts_off && !ts_off_) {
      int zero = 0;
      ::setsockopt(fd, SOL_SOCKET, SO_TIMESTAMPING, &zero, sizeof(zero));
      ts_off_ = true;
    }
    if (tuned) {
      sched_param sp{};
      sp.sched_priority = 1;
      int e = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
      fifo_ = e == 0 ? 1 : (uint64_t)e << 8;
    } else if (fifo_ == 1) {
      sched_param sp{};
      pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
      fifo_ = 0;
    }
    int64_t t0 = realtime_ns();
    if (tuned && !mlocked_.load()) {
      mlocked_.store(::mlockall(MCL_CURRENT | MCL_ONFAULT) == 0 ? 1 : 0);
      if (!mlocked_.load()) errors_.fetch_add(1);
    } else if (!tuned && mlocked_.load()) {
      ::munlockall();
      mlocked_.store(0);
    }
    head[1] = (uint64_t)(realtime_ns() - t0);
    if (tuned) tune_buffers(fd);
    int v = 0;
    socklen_t l = sizeof(v);
    ::getsockopt(fd, SOL_SOCKET, SO_SNDBUF, &v, &l);
    head[2] = (uint64_t)v;
    l = sizeof(v);
    ::getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &v, &l);
    head[14] = (uint64_t)v;
  }

  void serve() {
    if (cpu_ >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu_, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    if (cudaSetDevice(device_) != cudaSuccess) {
      ready_err_.store(ENODEV);
      return;
    }
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    uint8_t* host = nullptr;
    size_t cap = HEADER_BYTES + (size_t)max_bytes_;
    if (cudaHostAlloc((void**)&host, cap, cudaHostAllocDefault) != cudaSuccess) {
      ready_err_.store(ENOMEM);
      return;
    }
    std::memset(host, 0, cap);
    tid_.store((int64_t)::syscall(SYS_gettid));
    auto* head = reinterpret_cast<uint64_t*>(host);
    uint8_t* payload = host + HEADER_BYTES;
    while (running_.load()) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) continue;
      connections_.fetch_add(1);
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      int tsf = TS_FLAGS;
      bool ts_on = ::setsockopt(fd, SOL_SOCKET, SO_TIMESTAMPING, &tsf, sizeof(tsf)) == 0;
      ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
      conn_fd_.store(fd);
      fd_ = fd;
      tuned_ = false;
      ts_off_ = false;
      zc_ = zc_copied_ = false;
      zc_sent_ = zc_done_ = 0;
      zc_completions_ = 0;
      tx_key_ = 0;
      errq_reads_ = 0;
      want_key_[0] = want_key_[1] = ~0ull;
      want_ts_[0] = want_ts_[1] = 0;
      uint64_t sent = 0, seq = 0, prev_calls = 0, prev_eagain = 0;
      int64_t prev_end = 0, ktx = 0;
      uint64_t kkey = 0;
      bool tx_pending = false, quick_ok = false;
      uint64_t req[2];
      int64_t krx = 0;
      while (running_.load() && read_request(fd, req, &krx, &tx_pending, &ktx, &kkey)) {
        auto t0 = std::chrono::steady_clock::now();
        int64_t urx = realtime_ns();
        uint64_t op = req[0] & 0xff, nbytes = req[0] >> 8;
        if (op == 3) break;
        // Zero-copy: the previous response's pages are the staging buffer's until the kernel completes it.
        int64_t zc_wait = 0;
        if (zc_pending()) {
          int64_t w0 = realtime_ns();
          while (zc_pending() && running_.load(std::memory_order_relaxed) && realtime_ns() - w0 < 1000000000)
            drain_errq(&tx_pending, &ktx, &kkey);
          zc_wait = realtime_ns() - w0;
          if (zc_pending()) errors_.fetch_add(1);  // a second without it: counted, and the checker sees any damage
        }
        std::memset(head, 0, HEADER_BYTES);
        head[24] = (uint64_t)zc_wait;
        if (op == 4) {
          head[16] = (uint64_t)realtime_ns();
          if (nbytes == 0 || nbytes > max_bytes_ || req[1] > total_ / BLOCK || req[1] * BLOCK + nbytes > total_) {
            head[0] = 1;
          } else {
            cudaEventRecord(ev0, stream);
            cudaMemcpyAsync(payload, base_ + req[1] * BLOCK, nbytes, cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev1, stream);
            cudaError_t st;
            while ((st = cudaEventQuery(ev1)) == cudaErrorNotReady) {
            }
            head[17] = (uint64_t)realtime_ns();
            if (st != cudaSuccess) {
              head[0] = 2;
              errors_.fetch_add(1);
            } else {
              float ms = 0.f;
              cudaEventElapsedTime(&ms, ev0, ev1);
              head[1] = (uint64_t)((double)ms * 1e6);
            }
          }
          answered_.fetch_add(1);
        } else if (op == 5) {
          configure(fd, req[1] & 1, req[1] & 2, req[1] & 4, head);
          nbytes = 0;
        } else {
          if (op != 2) head[0] = 3;
          pings_.fetch_add(1);
          if (op != 2 || nbytes > max_bytes_) nbytes = 0;
        }
        head[3] = (uint64_t)krx;
        head[4] = (uint64_t)urx;
        head[6] = (uint64_t)ktx;
        head[7] = kkey;
        head[8] = prev_calls;
        head[9] = (uint64_t)prev_end;
        head[10] = (tuned_ ? 1 : 0) | (mlocked_.load() ? 2 : 0) | (krx ? 4 : 0) | (ts_on && !ts_off_ ? 8 : 0) | (quick_ok ? 16 : 0) |
                   (ts_off_ ? 32 : 0) | (fifo_ == 1 ? 64 : 0) | (zc_ ? 128 : 0) | (zc_copied_ ? 256 : 0);
        head[11] = (uint64_t)sched_getcpu();
        head[12] = seq++;
        head[13] = nbytes;
        if (op != 5) head[14] = tx_key_;
        head[15] = MAGIC;
        head[18] = prev_eagain;
        head[19] = op;
        head[20] = errq_reads_;
        errq_reads_ = 0;
        head[21] = fifo_;
        head[22] = (uint64_t)want_ts_[0];
        head[23] = (uint64_t)want_ts_[1];
        head[25] = zc_completions_;
        size_t want = HEADER_BYTES + nbytes;
        head[2] = op == 5 ? head[2]
                          : (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        head[5] = (uint64_t)realtime_ns();
        const uint8_t* p = host;
        size_t left = want;
        uint64_t calls = 0, eagain = 0;
        bool ok = true;
        while (left) {
          iovec iov{const_cast<uint8_t*>(p), left};
          msghdr m{};
          m.msg_iov = &iov;
          m.msg_iovlen = 1;
          ssize_t r = ::sendmsg(fd, &m, MSG_NOSIGNAL | MSG_DONTWAIT | (zc_ ? MSG_ZEROCOPY : 0));
          ++calls;
          if (r > 0) {
            p += r;
            left -= (size_t)r;
            if (zc_) ++zc_sent_;
          } else if (zc_ && r < 0 && errno == ENOBUFS) {
            ++eagain;  // the socket's option memory is full of uncompleted notifications: reap them
            drain_errq(&tx_pending, &ktx, &kkey);
          } else if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            ok = false;
            break;
          } else {
            ++eagain;
          }
        }
        prev_end = realtime_ns();
        prev_eagain = eagain;
        if (!ok) break;
        sent += want;
        tx_key_ = (sent - 1) & 0xffffffffu;
        want_key_[1] = want_key_[0];
        want_ts_[1] = want_ts_[0];
        want_key_[0] = tx_key_;
        want_ts_[0] = 0;
        tx_pending = ts_on && !ts_off_;
        prev_calls = calls;
        if (tuned_) quick_ok = ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)) == 0;
      }
      conn_fd_.store(-1);
      ::close(fd);
    }
    if (mlocked_.load()) ::munlockall();
    cudaFreeHost(host);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
  }

  const uint8_t* base_;
  uint64_t total_;
  int device_;
  std::string host_;
  int port_;
  int cpu_;
  uint64_t max_bytes_;
  int listen_fd_ = -1, fd_ = -1;
  bool tuned_ = false, ts_off_ = false, zc_ = false, zc_copied_ = false;
  uint32_t zc_sent_ = 0, zc_done_ = 0;  // zero-copy sends made and completed (notification ids are per send call, from 0)
  uint64_t zc_completions_ = 0;
  uint64_t tx_key_ = 0, errq_reads_ = 0, fifo_ = 0;
  uint64_t want_key_[2] = {~0ull, ~0ull};  // the last-byte keys of the last two responses, newest first
  int64_t want_ts_[2] = {0, 0};             // their TX timestamps, once read
  std::atomic<int> conn_fd_{-1};
  std::atomic<bool> running_{false};
  std::atomic<int64_t> answered_{0}, pings_{0}, tid_{0}, connections_{0}, errors_{0}, mlocked_{0}, ready_err_{0};
  std::thread thread_;
};

volatile sig_atomic_t g_stop = 0;
struct sigaction g_prev_term;

void on_term(int sig) {
  g_stop = 1;
  if (g_prev_term.sa_handler != SIG_DFL && g_prev_term.sa_handler != SIG_IGN && g_prev_term.sa_handler != nullptr) {
    g_prev_term.sa_handler(sig);
  }
}

}  // namespace

// ---- the tuned verifier's hot loop -----------------------------------------------------------------------------------------
// Record columns (int64), shared with pous_soak.REC (the Python baseline fills the same columns, 5-7 left 0):
enum {
  R_SCHED, R_SEND0, R_SEND1, R_FIRST, R_RECV, R_KRX_FIRST, R_KRX_LAST, R_KTX, R_RECV_CALLS, R_RING_WAIT, R_SEQ, R_BLK, R_NBYTES,
  R_OP, R_HEAD, R_N = R_HEAD + HEADER_WORDS
};
// Ring slot: 64 bytes of metadata (seq, blk, nbytes, cell, op), then the response as received.
constexpr size_t SLOT_META = 64;

extern "C" {

void* soak_responder_new(uint64_t base, uint64_t total, int device, const char* host, int port, int cpu, uint64_t max_bytes) {
  return new SoakResponder(base, total, device, host, port, cpu, max_bytes);
}
int soak_responder_start(void* r) { return static_cast<SoakResponder*>(r)->start(); }
void soak_responder_stop(void* r) { static_cast<SoakResponder*>(r)->stop(); }
int64_t soak_responder_stat(void* r, int which) { return static_cast<SoakResponder*>(r)->stat(which); }
void soak_responder_free(void* r) { delete static_cast<SoakResponder*>(r); }

int soak_header_words() { return HEADER_WORDS; }
int soak_record_words() { return R_N; }
uint64_t soak_magic() { return MAGIC; }
int soak_ts_flags() { return TS_FLAGS; }

int soak_install_sigterm() {
  struct sigaction sa{};
  sa.sa_handler = on_term;
  sigemptyset(&sa.sa_mask);
  return sigaction(SIGTERM, &sa, &g_prev_term);
}
int soak_stopped() { return g_stop; }

int soak_mlockall(int on) { return on ? ::mlockall(MCL_CURRENT | MCL_ONFAULT) : ::munlockall(); }

// Socket options of a tuned client connection: SO_TIMESTAMPING (its TX timestamps are drained by the C loop only), the 1 MiB
// buffers; returns the effective SO_RCVBUF.
int soak_client_tune(int fd, int tuned, int ts) {
  if (tuned && ts) {
    int tsf = TS_FLAGS;
    ::setsockopt(fd, SOL_SOCKET, SO_TIMESTAMPING, &tsf, sizeof(tsf));
  }
  if (tuned) tune_buffers(fd);
  int v = 0;
  socklen_t l = sizeof(v);
  ::getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &v, &l);
  return v;
}

// One cell: n requests at realtime t_sched[i] (spinning; late ones go at once) for `nbytes` from block blk[i], each response
// received into ring slot seq % nslots and published through ctl[0] (head); ctl[1] is the checker's tail, ctl[2] its halt flag.
// `*reqs` counts requests sent on this connection (for the TX timestamps' keys) and is advanced.  Returns the records written;
// *status: 0 done, 1 the connection failed, 2 the checker halted, 3 SIGTERM, 4 the deadline passed, 5 an answer stalled (not
// complete `stall_ns` after its request: a receiver whose window stays shut never recovers, and the connection is spent).
int64_t soak_client_run(int fd, int64_t n, const int64_t* t_sched, const int64_t* blk, const int64_t* ops, int64_t nbytes,
                        uint8_t* ring, int64_t slot_bytes, int64_t nslots, volatile int64_t* ctl, int64_t* rec, int64_t deadline_ns,
                        int64_t cell, int tuned, int ts, int64_t* reqs, int64_t* status, int64_t stall_ns) {
  const int one = 1;
  const size_t want = HEADER_BYTES + (size_t)nbytes;
  int64_t r0 = *reqs;
  *status = 0;
  int64_t i = 0;
  auto drain = [&](void) {
    int64_t ts = 0;
    uint64_t key = 0;
    int r;
    while ((r = read_tx(fd, &ts, &key)) != 0) {
      int64_t req = (int64_t)((key + 1) / 16) - 1 - r0;
      if (r == 1 && req >= 0 && req < n) rec[req * R_N + R_KTX] = ts;
    }
  };
  for (; i < n; ++i) {
    int64_t* r = rec + i * R_N;
    std::memset(r, 0, sizeof(int64_t) * R_N);
    int64_t seq = __atomic_load_n(&ctl[0], __ATOMIC_RELAXED);
    int64_t w0 = realtime_ns();
    while (seq - __atomic_load_n(&ctl[1], __ATOMIC_ACQUIRE) >= nslots) {
      if (g_stop || ctl[2] || realtime_ns() > deadline_ns) break;
    }
    r[R_RING_WAIT] = realtime_ns() - w0;
    if (seq - __atomic_load_n(&ctl[1], __ATOMIC_ACQUIRE) >= nslots) { *status = g_stop ? 3 : ctl[2] ? 2 : 4; return i; }
    for (;;) {
      if (g_stop) { *status = 3; return i; }
      if (ctl[2]) { *status = 2; return i; }
      int64_t now = realtime_ns();
      if (now >= t_sched[i]) break;
      if (now > deadline_ns) { *status = 4; return i; }
      if (ts) drain();
    }
    uint8_t* slot = ring + (seq % nslots) * slot_bytes;
    auto* meta = reinterpret_cast<int64_t*>(slot);
    meta[0] = seq;
    meta[1] = blk[i];
    meta[2] = nbytes;
    meta[3] = cell;
    meta[4] = ops[i];
    uint8_t* resp = slot + SLOT_META;
    uint64_t q[2] = {(uint64_t)ops[i] | ((uint64_t)nbytes << 8), (uint64_t)blk[i]};
    r[R_SCHED] = t_sched[i];
    r[R_SEQ] = seq;
    r[R_BLK] = blk[i];
    r[R_NBYTES] = nbytes;
    r[R_OP] = ops[i];
    if (tuned) ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    iovec qv{q, sizeof(q)};
    msghdr qm{};
    qm.msg_iov = &qv;
    qm.msg_iovlen = 1;
    r[R_SEND0] = realtime_ns();
    ssize_t s = ::sendmsg(fd, &qm, MSG_NOSIGNAL | MSG_DONTWAIT);
    r[R_SEND1] = realtime_ns();
    if (s != (ssize_t)sizeof(q)) { *status = 1; return i; }
    ++*reqs;
    size_t got = 0;
    while (got < want) {
      char ctrl[256];
      iovec iov{resp + got, want - got};
      msghdr m{};
      m.msg_iov = &iov;
      m.msg_iovlen = 1;
      m.msg_control = ctrl;
      m.msg_controllen = sizeof(ctrl);
      ssize_t k = ts ? ::recvmsg(fd, &m, MSG_DONTWAIT) : ::recv(fd, resp + got, want - got, MSG_DONTWAIT);
      if (k > 0) {
        int64_t t = realtime_ns();
        int64_t kt = ts ? cmsg_ts(&m) : 0;
        if (!got) {
          r[R_FIRST] = t;
          r[R_KRX_FIRST] = kt;
        }
        r[R_KRX_LAST] = kt;
        r[R_RECV_CALLS] += 1;
        got += (size_t)k;
        if (got == want) r[R_RECV] = t;
      } else if (k == 0 || (errno != EAGAIN && errno != EWOULDBLOCK) || g_stop) {
        *status = k == 0 || !g_stop ? 1 : 3;
        return i;
      } else if (realtime_ns() - r[R_SEND1] > stall_ns) {
        *status = 5;
        return i;
      }
    }
    std::memcpy(r + R_HEAD, resp, HEADER_BYTES);
    __atomic_store_n(&ctl[0], seq + 1, __ATOMIC_RELEASE);
    if (tuned) ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    // TX timestamps queue on the error queue, which is charged to the receive buffer: a cell behind its schedule never waits
    // above, and undrained they shut the receive window (about 7,000 requests on a 6 MB buffer).
    if (ts) drain();
  }
  if (ts) drain();
  return i;
}

// The cost of reading `clk` back to back: min, p50 and p99.9 of n successive differences (ns).
int soak_clock_cost(int clk, int64_t n, int64_t* out) {
  std::vector<int64_t> d((size_t)n);
  timespec a, b;
  clock_gettime(clk, &a);
  for (int64_t i = 0; i < n; ++i) {
    clock_gettime(clk, &b);
    d[(size_t)i] = (int64_t)(b.tv_sec - a.tv_sec) * 1000000000 + (b.tv_nsec - a.tv_nsec);
    a = b;
  }
  std::sort(d.begin(), d.end());
  out[0] = d[0];
  out[1] = d[(size_t)(n / 2)];
  out[2] = d[(size_t)(n * 999 / 1000)];
  return 0;
}

// The TX timestamps still queued after a cell (its last requests'), into the cell's records.
void soak_client_drain(int fd, int64_t* rec, int64_t n, int64_t r0) {
  int64_t ts = 0;
  uint64_t key = 0;
  for (int spins = 0; spins < 1000; ++spins) {
    int r;
    while ((r = read_tx(fd, &ts, &key)) != 0) {
      int64_t req = (int64_t)((key + 1) / 16) - 1 - r0;
      if (r == 1 && req >= 0 && req < n) rec[req * R_N + R_KTX] = ts;
    }
  }
}

}  // extern "C"
