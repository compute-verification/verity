// bwstream: the certified PCIe stream (stream.py, `stream/v1`) on one RTX PRO 6000, beside the ro-blake3/v1 tag pass.
//
//   absorb        a CTA per 64 KiB group of S, a thread per chunk of M: at load j of compression g, thread c reads AES
//                 block 64·(4g + j) + c of the group (a warp load is 512 contiguous bytes, a compression 4 KiB), 16
//                 keyed compressions under counter 64·q + c, then six
//                 parent levels in shared memory to the group's node.  Every chunk's chaining value stays in device memory
//                 for openings.  `absorb_final`, one CTA, takes the groups' nodes to D, ROOT on the last parent.
//   window        a CUDA graph per stream size: the key's copy, then the stream's pieces copied host to device on BRANCHES
//                 chains (copy engines), each piece's absorb launched when its copy lands, the final launch, D back to the
//                 host; launched beside bw_tag_pass on its own stream, after the honest prover's preparation (a warm pass
//                 and a 512 MiB read scrub, as gpu.Prover).
//   host          AES-128-CTR from stream_host.cpp (VAES), on threads pinned one a core first, then siblings.
//
// Modes (each writes one JSON object to --out/<mode>.json; every gate runs before any timing and a failed gate exits 3):
//   gates   the host generators against stream_vectors.json's keystream; the device absorb and the host reference
//           against its digests and group nodes; the tag pass against expect.json's ro-blake3/v1 root
//   link    PCIe ceilings: H2D and D2H copies over piece sizes and streams; a kernel reading pinned host memory (zero
//           copy); H2D with zero copy; H2D with D2H; all three
//   hbm     a device read of 4 GiB alone and beside H2D at the link's rate (into 1 GiB, and into an 8 MiB ring), and beside
//           zero copy
//   gen     the host keystream's rate: per thread (vaes, aesni, openssl), and over thread counts
//   pipe    sustained generation and H2D: G threads fill a pinned ring and the copies drain it, for each G
//   absorb  the absorb's own rate, from device memory and from pinned host memory (zero copy)
//   window  streamed windows: tag only, stream only and both, over stream sizes; every window's D checked against the host
//           reference and its root against an untimed pass under the same seed; one window per size that skips every
//           64th piece, which must fail both; the verifier's openings timed.  With --live G the challenger generates S
//           only after σ, on G pinned threads in LIVE_SLOT slots taken in order, and each piece (--first-piece doubling
//           to --piece) is copied once its slots are written; every byte sent is checked against S regenerated from K_s
//   fold    what a zero-copy absorb with a cheap leaf (each 1 KiB chunk's XOR, fold_chunks) costs the tag pass, over
//           grids from a few whole SMs to one warp on every SM; every fold checked against the host's
//   foldlive  the fold of a stream the challenger generates after σ on --live G threads (fold_live reads behind a
//           frontier in mapped host memory), on 2, 4 and 8 SMs, over --sizes-mib; every fold checked against S regenerated

#include "tagger.cu"

#include <pthread.h>
#include <sched.h>
#include <sys/random.h>

#include <atomic>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>

extern "C" {
void sg_expand(const uint8_t key[16], uint8_t rk[176]);
void sg_ctr_vaes(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks);
void sg_ctr_vaes_nt(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks);
void sg_ctr_aesni(const uint8_t rk[176], uint64_t first, uint8_t* out, uint64_t blocks);
int sg_ctr_openssl(const uint8_t key[16], uint64_t first, uint8_t* out, uint64_t blocks);
void sg_flush(const uint8_t* p, uint64_t n);
}

#define OK(x)                                                                                           \
  do {                                                                                                  \
    cudaError_t e_ = (x);                                                                               \
    if (e_ != cudaSuccess) {                                                                            \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(4);                                                                                          \
    }                                                                                                   \
  } while (0)

namespace st {

using bw::B3_CHUNK_END;
using bw::B3_CHUNK_START;
using bw::B3_KEYED;
using bw::B3_PARENT;
using bw::B3_ROOT;
using bw::b3_compress;

constexpr int GROUP = 64, AES = 16;
constexpr uint64_t CHUNK = 1024, GROUP_BYTES = GROUP * CHUNK;
constexpr int FINAL_THREADS = 512, FINAL_MAX = 2048;
constexpr int BRANCHES = 4;
constexpr size_t SCRUB = 512ull << 20;
static const char ABSORB_DOMAIN[] = "verity/bandwidth/stream/v1";
static const char ABSORB_NAME[] = "stream-blake3/v1";

// ---------------------------------------------------------------------------------------------------------------------
// Device.

__global__ void __launch_bounds__(GROUP) absorb_kernel(const uint4* __restrict__ S, uint64_t first_group,
                                                       const uint32_t* __restrict__ key, uint32_t* __restrict__ cvs,
                                                       uint32_t* __restrict__ nodes) {
  __shared__ uint32_t sm[GROUP][8];
  const uint64_t q = first_group + blockIdx.x;
  const int c = threadIdx.x;
  uint32_t k[8], cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = k[i] = key[i];
  const uint4* base = S + q * (GROUP_BYTES / 16) + c;
  uint4 cur[4], nxt[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) cur[j] = base[j * GROUP];
  const uint32_t counter = (uint32_t)(q * GROUP + c);
#pragma unroll 1
  for (int g = 0; g < 16; ++g) {
    if (g < 15) {
#pragma unroll
      for (int j = 0; j < 4; ++j) nxt[j] = base[(4 * (g + 1) + j) * GROUP];
    }
    uint32_t m[16] = {cur[0].x, cur[0].y, cur[0].z, cur[0].w, cur[1].x, cur[1].y, cur[1].z, cur[1].w,
                      cur[2].x, cur[2].y, cur[2].z, cur[2].w, cur[3].x, cur[3].y, cur[3].z, cur[3].w};
    b3_compress(cv, m, counter, 64, B3_KEYED | (g == 0 ? B3_CHUNK_START : 0) | (g == 15 ? B3_CHUNK_END : 0), cv);
#pragma unroll
    for (int j = 0; j < 4; ++j) cur[j] = nxt[j];
  }
  uint4* o = (uint4*)(cvs + (q * GROUP + c) * 8);
  o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
  o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
#pragma unroll
  for (int i = 0; i < 8; ++i) sm[c][i] = cv[i];
  __syncthreads();
  for (int w = GROUP / 2; w >= 1; w >>= 1) {
    uint32_t r[8];
    if (c < w) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) m[i] = sm[2 * c][i], m[8 + i] = sm[2 * c + 1][i];
      b3_compress(k, m, 0, 64, B3_KEYED | B3_PARENT, r);
    }
    __syncthreads();
    if (c < w)
#pragma unroll
      for (int i = 0; i < 8; ++i) sm[c][i] = r[i];
    __syncthreads();
  }
  if (c < 8) nodes[q * 8 + c] = sm[0][c];
}

// The groups' n nodes (2 <= n <= FINAL_MAX) to D: bottom-up, an odd last node promoted, ROOT on the last parent.
__global__ void __launch_bounds__(FINAL_THREADS) absorb_final(const uint32_t* __restrict__ nodes, uint32_t n,
                                                              const uint32_t* __restrict__ key, uint32_t* __restrict__ out) {
  __shared__ uint32_t sm[FINAL_MAX / 2][8];
  uint32_t k[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) k[i] = key[i];
  for (uint32_t i = threadIdx.x; i < (n + 1) / 2; i += blockDim.x) {
    if (2 * i + 1 < n) {
      uint32_t m[16], r[8];
#pragma unroll
      for (int t = 0; t < 16; ++t) m[t] = nodes[16 * i + t];
      b3_compress(k, m, 0, 64, B3_KEYED | B3_PARENT | (n == 2 ? B3_ROOT : 0), r);
#pragma unroll
      for (int t = 0; t < 8; ++t) sm[i][t] = r[t];
    } else {
#pragma unroll
      for (int t = 0; t < 8; ++t) sm[i][t] = nodes[16 * i + t];
    }
  }
  __syncthreads();
  for (uint32_t width = (n + 1) / 2; width > 1;) {
    const uint32_t next = (width + 1) / 2, i = threadIdx.x;
    uint32_t r[8];
    if (i < next) {
      if (2 * i + 1 < width) {
        uint32_t m[16];
#pragma unroll
        for (int t = 0; t < 8; ++t) m[t] = sm[2 * i][t], m[8 + t] = sm[2 * i + 1][t];
        b3_compress(k, m, 0, 64, B3_KEYED | B3_PARENT | (width == 2 ? B3_ROOT : 0), r);
      } else {
#pragma unroll
        for (int t = 0; t < 8; ++t) r[t] = sm[2 * i][t];
      }
    }
    __syncthreads();
    if (i < next)
#pragma unroll
      for (int t = 0; t < 8; ++t) sm[i][t] = r[t];
    __syncthreads();
    width = next;
  }
  if (threadIdx.x < 8) out[threadIdx.x] = sm[0][threadIdx.x];
}

// Reads n16 uint4 of p (device or mapped host memory) once, four loads in flight a thread.
__global__ void read_any(const uint4* __restrict__ p, uint64_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n16; i += 4 * stride) {
    uint4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
    acc ^= a.x ^ b.y ^ c.z ^ d.w;
  }
  for (; i < n16; i += stride) acc ^= __ldcs(p + i).x;
#pragma unroll
  for (int d = 16; d; d >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, d);
  if ((threadIdx.x & 31) == 0) atomicXor(sink, acc);
}

// A leaf far cheaper than BLAKE3, for `fold`: out[c] = the XOR of 1 KiB chunk c's 64 uint4, read from p (mapped host
// memory) by one warp, four chunks (eight loads a lane) in flight.
__global__ void fold_chunks(const uint4* __restrict__ p, uint64_t chunks, uint4* __restrict__ out) {
  const uint32_t lane = threadIdx.x & 31;
  const uint64_t warp = (blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) >> 5;
  const uint64_t warps = ((uint64_t)gridDim.x * blockDim.x) >> 5;
  for (uint64_t c0 = 4 * warp; c0 < chunks; c0 += 4 * warps) {
    uint4 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = make_uint4(0, 0, 0, 0);
      if (c0 + k < chunks) {
        const uint4 x = __ldcs(p + (c0 + k) * 64 + lane), y = __ldcs(p + (c0 + k) * 64 + 32 + lane);
        v[k] = make_uint4(x.x ^ y.x, x.y ^ y.y, x.z ^ y.z, x.w ^ y.w);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        v[k].x ^= __shfl_xor_sync(0xffffffffu, v[k].x, d);
        v[k].y ^= __shfl_xor_sync(0xffffffffu, v[k].y, d);
        v[k].z ^= __shfl_xor_sync(0xffffffffu, v[k].z, d);
        v[k].w ^= __shfl_xor_sync(0xffffffffu, v[k].w, d);
      }
      if (lane == 0 && c0 + k < chunks) out[c0 + k] = v[k];
    }
  }
}

// fold_chunks over a stream the host is still writing, a CTA at a time: CTA b's i-th step folds the 4·(warps a CTA)
// chunks from (i·gridDim + b) times that, once *frontier (mapped host memory: epoch << 32 | the slots written, in
// order) covers them.  Thread 0 reads the frontier only when the last value it read doesn't cover the step, so with
// the host ahead a CTA reads it rarely: system-scope reads of host memory are slow and serialized (a flag read per 4 KiB
// held the fold to 7-10 GB/s and slowed the tag pass 25-50%).  The loads after the step's barrier depend on what thread
// 0 read; the read scrub between windows leaves no line of the buffer in L1 or L2.  With deadline_ns, a CTA starts no
// step later than that after its own start, and steps[b] is the steps CTA b finished.  chunks is a multiple of 4.
__global__ void fold_live(const uint4* __restrict__ p, uint64_t chunks, uint4* __restrict__ out,
                          const unsigned long long* frontier, uint64_t slot_chunks, uint32_t epoch, uint64_t deadline_ns,
                          uint32_t* steps) {
  __shared__ int stop[2];  // alternating, so that thread 0 never rewrites a step's flag before every thread has read it
  const uint32_t lane = threadIdx.x & 31;
  const uint64_t per = 4 * (blockDim.x >> 5);
  uint64_t known = 0, t0 = 0;  // thread 0's: the chunks the frontier has covered, and the CTA's start
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  uint32_t step = 0;
  for (uint64_t base = blockIdx.x * per; base < chunks; base += gridDim.x * per, ++step) {
    const uint64_t need = min(base + per, chunks);
    if (threadIdx.x == 0) {
      bool late = false;
      // bounded (about 2 s of polls), so that a lost frontier fails the fold's check instead of hanging the run
      for (uint32_t tries = 0;; ++tries) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        late = deadline_ns && t - t0 >= deadline_ns;
        if (late || known >= need || tries >= (1u << 22)) break;
        unsigned long long v;
        asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(frontier) : "memory");
        known = (v >> 32) == epoch ? (v & 0xffffffffu) * slot_chunks : 0;
        if (known < need) __nanosleep(256);
      }
      stop[step & 1] = late;
    }
    __syncthreads();
    if (stop[step & 1]) break;
    const uint64_t c0 = base + 4 * (threadIdx.x >> 5);
    if (c0 >= chunks) continue;
    uint4 x[4], y[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      x[k] = __ldcs(p + (c0 + k) * 64 + lane);
      y[k] = __ldcs(p + (c0 + k) * 64 + 32 + lane);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      uint4 v = make_uint4(x[k].x ^ y[k].x, x[k].y ^ y[k].y, x[k].z ^ y[k].z, x[k].w ^ y[k].w);
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        v.x ^= __shfl_xor_sync(0xffffffffu, v.x, d);
        v.y ^= __shfl_xor_sync(0xffffffffu, v.y, d);
        v.z ^= __shfl_xor_sync(0xffffffffu, v.z, d);
        v.w ^= __shfl_xor_sync(0xffffffffu, v.w, d);
      }
      if (lane == 0) out[c0 + k] = v;
    }
  }
  if (threadIdx.x == 0 && steps) steps[blockIdx.x] = step;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host: timing, threads, JSON.

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int> core_order() {
  // The CPUs this process may use, one per core first (lowest sibling), then the remaining siblings.
  cpu_set_t set;
  sched_getaffinity(0, sizeof set, &set);
  std::vector<int> first, rest;
  std::map<std::pair<int, int>, int> seen;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &set)) continue;
    auto rd = [&](const char* what) {
      std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/" + what);
      int v = -1;
      f >> v;
      return v;
    };
    std::pair<int, int> core{rd("physical_package_id"), rd("core_id")};
    if (seen.count(core))
      rest.push_back(c);
    else
      seen[core] = c, first.push_back(c);
  }
  first.insert(first.end(), rest.begin(), rest.end());
  return first;
}

static void pin(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

// Runs f(t) on threads 0..n-1, thread t pinned to cpus[t].
static void parallel(int n, const std::vector<int>& cpus, const std::function<void(int)>& f) {
  std::vector<std::thread> th;
  for (int t = 0; t < n; ++t)
    th.emplace_back([&, t] {
      pin(cpus[t % cpus.size()]);
      f(t);
    });
  for (auto& x : th) x.join();
}

// A thread pinned to `cpu` that runs each job posted to it, polling between jobs so that a job posted inside a timed
// window starts without a wake-up.
struct Spinner {
  std::atomic<uint64_t> posted{0}, finished{0};
  std::atomic<bool> quit{false};
  std::function<void()> job;
  std::thread th;
  explicit Spinner(int cpu)
      : th([this, cpu] {
          pin(cpu);
          for (uint64_t seen = 0;;) {
            while (posted.load(std::memory_order_acquire) == seen) {
              if (quit.load(std::memory_order_relaxed)) return;
              sched_yield();
            }
            job();
            finished.store(++seen, std::memory_order_release);
          }
        }) {}
  void post(std::function<void()> f) {
    job = std::move(f);
    posted.fetch_add(1, std::memory_order_release);
  }
  void wait() {
    while (finished.load(std::memory_order_acquire) != posted.load(std::memory_order_relaxed)) {
    }
  }
  ~Spinner() {
    quit = true;
    th.join();
  }
};

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static std::string num(double x) {
  char b[64];
  snprintf(b, sizeof b, "%.6g", x);
  return b;
}

static std::string stats(const std::vector<double>& v) {
  if (v.empty()) return "null";
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  return "{\"n\":" + std::to_string(s.size()) + ",\"median\":" + num(median(s)) + ",\"min\":" + num(s.front()) +
         ",\"max\":" + num(s.back()) + ",\"p90\":" + num(s[(size_t)(0.9 * (s.size() - 1))]) + "}";
}

static std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) s += d[p[i] >> 4], s += d[p[i] & 15];
  return s;
}

static std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> out(s.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = (uint8_t)std::stoul(s.substr(2 * i, 2), nullptr, 16);
  return out;
}

// The few fields a gate reads from a JSON file: "name": "hex" or "name": number, found by name (names are unique).
static std::string field(const std::string& text, const std::string& name) {
  size_t p = text.find("\"" + name + "\"");
  if (p == std::string::npos) {
    fprintf(stderr, "field %s missing\n", name.c_str());
    exit(3);
  }
  p = text.find(':', p) + 1;
  while (text[p] == ' ' || text[p] == '"') ++p;
  size_t e = p;
  while (e < text.size() && text[e] != '"' && text[e] != ',' && text[e] != '}' && text[e] != '\n') ++e;
  return text.substr(p, e - p);
}

static std::string slurp(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    exit(3);
  }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

static void fresh(uint8_t* p, size_t n) {
  for (size_t off = 0; off < n;) {
    ssize_t got = getrandom(p + off, n - off, 0);
    if (got > 0) off += (size_t)got;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Host: the stream, its reference digest and the verifier's openings.

struct Gen {
  uint8_t rk[176];
  uint8_t key[16];
  explicit Gen(const uint8_t k[16]) {
    memcpy(key, k, 16);
    sg_expand(key, rk);
  }
};

// S[0, n) on `threads` pinned threads (n a multiple of 512 bytes per thread's share), with non-temporal stores: the
// stream is bound for the copy engines, not the host's cache.
static void generate(const Gen& g, uint8_t* out, size_t n, int threads, const std::vector<int>& cpus, uint64_t first = 0) {
  const uint64_t blocks = n / 16, per = ((blocks + threads - 1) / threads + 31) / 32 * 32;
  parallel(threads, cpus, [&](int t) {
    uint64_t b0 = t * per, b1 = std::min<uint64_t>(blocks, b0 + per);
    if (b0 < b1) sg_ctr_vaes_nt(g.rk, first + b0, out + 16 * b0, b1 - b0);
  });
}

static void key_words(const uint8_t key[32], uint32_t w[8]) {
  for (int i = 0; i < 8; ++i) w[i] = key[4 * i] | key[4 * i + 1] << 8 | key[4 * i + 2] << 16 | (uint32_t)key[4 * i + 3] << 24;
}

// K_a = SHA512(prefix(ABSORB_DOMAIN, ABSORB_NAME, σ))[0:32], as BLAKE3 key words.
static void absorb_key(const uint8_t seed[32], uint32_t w[8]) {
  std::vector<uint8_t> p = bw::prefix_block(ABSORB_DOMAIN, ABSORB_NAME, seed);
  uint64_t s[8];
  bw::sha512_host(p.data(), p.size(), s);
  uint8_t d[64];
  bw::digest_bytes(s, d);
  key_words(d, w);
}

// The AES block of S (its counter) that is the k-th 16 bytes of chunk i of M (stream.counter).
static uint64_t counter_of(uint64_t chunk, int k) {
  return (chunk / GROUP) * (GROUP_BYTES / AES) + (uint64_t)k * GROUP + chunk % GROUP;
}

// Chunk `chunk`'s chaining value from its 64 AES blocks, blocks[k] the k-th.
static void chunk_cv_host(const uint32_t k[8], uint64_t chunk, const uint8_t* const blocks[64], uint32_t cv[8]) {
  for (int i = 0; i < 8; ++i) cv[i] = k[i];
  for (int g = 0; g < 16; ++g) {
    uint32_t m[16];
    for (int j = 0; j < 4; ++j) memcpy(m + 4 * j, blocks[4 * g + j], 16);
    b3_compress(cv, m, (uint32_t)chunk, 64, B3_KEYED | (g == 0 ? B3_CHUNK_START : 0) | (g == 15 ? B3_CHUNK_END : 0), cv);
  }
}

static void parent_host(const uint32_t k[8], const uint32_t* l, const uint32_t* r, bool root, uint32_t out[8]) {
  uint32_t m[16];
  memcpy(m, l, 32);
  memcpy(m + 8, r, 32);
  b3_compress(k, m, 0, 64, B3_KEYED | B3_PARENT | (root ? B3_ROOT : 0), out);
}

// Every level of the tree over the chunk chaining values (8 words each), bottom-up, the top under ROOT.
static std::vector<std::vector<uint32_t>> tree_levels(const uint32_t k[8], std::vector<uint32_t> cvs) {
  std::vector<std::vector<uint32_t>> lv{std::move(cvs)};
  while (lv.back().size() > 8) {
    const std::vector<uint32_t>& cur = lv.back();
    const size_t n = cur.size() / 8;
    std::vector<uint32_t> nxt(((n + 1) / 2) * 8);
    for (size_t i = 0; i < n; i += 2) {
      if (i + 1 < n)
        parent_host(k, &cur[8 * i], &cur[8 * i + 8], n == 2, &nxt[4 * i]);
      else
        memcpy(&nxt[4 * i], &cur[8 * i], 32);
    }
    lv.push_back(std::move(nxt));
  }
  return lv;
}

// The host's reference D over S (n bytes), on `threads` threads.
static void reference_digest(const uint32_t k[8], const uint8_t* S, size_t n, int threads, const std::vector<int>& cpus,
                             uint32_t D[8], std::vector<uint32_t>* group_nodes = nullptr) {
  const uint64_t chunks = n / CHUNK;
  std::vector<uint32_t> cvs(chunks * 8);
  parallel(threads, cpus, [&](int t) {
    for (uint64_t i = t; i < chunks; i += threads) {
      const uint8_t* blocks[64];
      for (int b = 0; b < 64; ++b) blocks[b] = S + AES * counter_of(i, b);
      chunk_cv_host(k, i, blocks, &cvs[8 * i]);
    }
  });
  auto lv = tree_levels(k, std::move(cvs));
  memcpy(D, lv.back().data(), 32);
  if (group_nodes) *group_nodes = lv[6];
}

// The verifier's check of s openings: chunk i regenerated from K_s alone (64 AES blocks), its chaining value, then the
// prover's path to D.  Returns the samples that fail.
static int verify_openings(const Gen& gen, const uint32_t k[8], const uint32_t D[8], uint64_t chunks,
                           const std::vector<std::vector<uint32_t>>& lv, const std::vector<uint64_t>& samples) {
  int bad = 0;
  alignas(64) uint8_t buf[64][16];
  for (uint64_t i : samples) {
    const uint8_t* blocks[64];
    for (int b = 0; b < 64; ++b) {
      sg_ctr_aesni(gen.rk, counter_of(i, b), buf[b], 1);
      blocks[b] = buf[b];
    }
    uint32_t node[8];
    chunk_cv_host(k, i, blocks, node);
    uint64_t j = i;
    for (size_t level = 0; level + 1 < lv.size(); ++level) {
      const uint64_t width = lv[level].size() / 8;
      if ((j ^ 1) < width) {  // the prover's opening: the sibling it holds at this level
        const uint32_t* sib = &lv[level][8 * (j ^ 1)];
        uint32_t out[8];
        if (j % 2 == 0)
          parent_host(k, node, sib, width == 2, out);
        else
          parent_host(k, sib, node, width == 2, out);
        memcpy(node, out, 32);
      }
      j >>= 1;
    }
    bad += memcmp(node, D, 32) != 0;
  }
  (void)chunks;
  return bad;
}

// ---------------------------------------------------------------------------------------------------------------------
// The device side of a stream size: buffers, and the window's graph.

struct Device {
  int sms = 0;
  uint32_t* sink = nullptr;
  uint8_t* scrub = nullptr;
};

struct Stream {
  size_t n = 0, piece = 0;
  uint64_t chunks = 0, groups = 0, pieces = 0;
  uint8_t* host = nullptr;  // pinned, the challenger's
  uint8_t* dev = nullptr;   // where the copies land
  uint32_t *cvs = nullptr, *nodes = nullptr, *key_dev = nullptr, *D_dev = nullptr;
  uint32_t *key_host = nullptr, *D_host = nullptr;  // pinned
  cudaGraphExec_t exec = nullptr, skip_exec = nullptr;
};

static void alloc_stream(Stream& s, size_t n, size_t piece) {
  s.n = n;
  s.piece = piece;
  s.chunks = n / CHUNK;
  s.groups = n / GROUP_BYTES;
  s.pieces = (n + piece - 1) / piece;
  OK(cudaHostAlloc(&s.host, n, cudaHostAllocDefault));
  OK(cudaMalloc(&s.dev, n));
  OK(cudaMalloc(&s.cvs, s.chunks * 32));
  OK(cudaMalloc(&s.nodes, s.groups * 32));
  OK(cudaMalloc(&s.key_dev, 32));
  OK(cudaMalloc(&s.D_dev, 32));
  OK(cudaHostAlloc(&s.key_host, 32, cudaHostAllocDefault));
  OK(cudaHostAlloc(&s.D_host, 32, cudaHostAllocDefault));
}

static void free_stream(Stream& s) {
  if (s.exec) cudaGraphExecDestroy(s.exec);
  if (s.skip_exec) cudaGraphExecDestroy(s.skip_exec);
  cudaFreeHost(s.host);
  cudaFree(s.dev);
  cudaFree(s.cvs);
  cudaFree(s.nodes);
  cudaFree(s.key_dev);
  cudaFree(s.D_dev);
  cudaFreeHost(s.key_host);
  cudaFreeHost(s.D_host);
  s = Stream{};
}

// The window's graph: key copy; pieces copied on BRANCHES chains; piece p's absorb after its copy (and the key); the
// final launch after every absorb; D to the host.  `skip` > 0 leaves out the copy of every piece p with p % skip == 0
// (a device that did not take those pieces off the link).
static cudaGraphExec_t build_graph(Stream& s, int skip, int priority) {
  cudaGraph_t g;
  OK(cudaGraphCreate(&g, 0));
  cudaGraphNode_t key_node;
  OK(cudaGraphAddMemcpyNode1D(&key_node, g, nullptr, 0, s.key_dev, s.key_host, 32, cudaMemcpyHostToDevice));
  std::vector<cudaGraphNode_t> last(BRANCHES, nullptr), absorbs;
  const uint64_t groups_per_piece = s.piece / GROUP_BYTES;
  for (uint64_t p = 0; p < s.pieces; ++p) {
    std::vector<cudaGraphNode_t> deps{key_node};
    const size_t off = p * s.piece, len = std::min(s.piece, s.n - off);
    if (!(skip > 0 && p % skip == 0)) {
      cudaGraphNode_t cp;
      cudaGraphNode_t* prev = last[p % BRANCHES] ? &last[p % BRANCHES] : nullptr;
      OK(cudaGraphAddMemcpyNode1D(&cp, g, prev, prev ? 1 : 0, s.dev + off, s.host + off, len, cudaMemcpyHostToDevice));
      last[p % BRANCHES] = cp;
      deps.push_back(cp);
    }
    const uint4* S = (const uint4*)s.dev;
    uint64_t first = p * groups_per_piece;
    void* args[] = {(void*)&S, &first, &s.key_dev, &s.cvs, &s.nodes};
    cudaKernelNodeParams kp = {};
    kp.func = (void*)absorb_kernel;
    kp.gridDim = dim3((unsigned)(len / GROUP_BYTES));
    kp.blockDim = dim3(GROUP);
    kp.kernelParams = args;
    cudaGraphNode_t kn;
    OK(cudaGraphAddKernelNode(&kn, g, deps.data(), deps.size(), &kp));
    cudaLaunchAttributeValue v = {};
    v.priority = priority;
    OK(cudaGraphKernelNodeSetAttribute(kn, cudaLaunchAttributePriority, &v));
    absorbs.push_back(kn);
  }
  uint32_t n = (uint32_t)s.groups;
  void* fargs[] = {&s.nodes, &n, &s.key_dev, &s.D_dev};
  cudaKernelNodeParams fp = {};
  fp.func = (void*)absorb_final;
  fp.gridDim = dim3(1);
  fp.blockDim = dim3(FINAL_THREADS);
  fp.kernelParams = fargs;
  cudaGraphNode_t fn, dn;
  OK(cudaGraphAddKernelNode(&fn, g, absorbs.data(), absorbs.size(), &fp));
  cudaLaunchAttributeValue v = {};
  v.priority = priority;
  OK(cudaGraphKernelNodeSetAttribute(fn, cudaLaunchAttributePriority, &v));
  OK(cudaGraphAddMemcpyNode1D(&dn, g, &fn, 1, s.D_host, s.D_dev, 32, cudaMemcpyDeviceToHost));
  cudaGraphExec_t exec;
  OK(cudaGraphInstantiate(&exec, g, cudaGraphInstantiateFlagUseNodePriority));
  OK(cudaGraphDestroy(g));
  return exec;
}

// The device absorb over s.dev (already holding S) without copies: D into s.D_host.
static void absorb_resident(Stream& s, const uint8_t* src, cudaStream_t st) {
  OK(cudaMemcpyAsync(s.key_dev, s.key_host, 32, cudaMemcpyHostToDevice, st));
  absorb_kernel<<<(unsigned)s.groups, GROUP, 0, st>>>((const uint4*)src, 0, s.key_dev, s.cvs, s.nodes);
  absorb_final<<<1, FINAL_THREADS, 0, st>>>(s.nodes, (uint32_t)s.groups, s.key_dev, s.D_dev);
  OK(cudaMemcpyAsync(s.D_host, s.D_dev, 32, cudaMemcpyDeviceToHost, st));
}

// The window's stream without a graph: pieces copied on COPY_STREAMS streams, each piece's absorb on `abs` after its
// copy's event, then the final launch and D to the host, all on `abs`, whose events `t0` and `t1` bracket it.  With
// `copies` false the absorbs read `src` (resident, or mapped host memory: zero copy); with `absorbs` false `abs` only
// waits for the copies; `skip` > 0 leaves out the copy of every piece p with p % skip == 0 (the control).
constexpr int COPY_STREAMS = 2;
struct Path {
  cudaStream_t abs = nullptr;
  std::vector<cudaStream_t> copy;
  std::vector<cudaEvent_t> ev;
  cudaEvent_t t0, t1;
};

static Path make_path(int priority, size_t max_pieces) {
  Path P;
  OK(cudaStreamCreateWithPriority(&P.abs, cudaStreamNonBlocking, priority));
  P.copy.resize(COPY_STREAMS);
  for (auto& c : P.copy) OK(cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking));
  P.ev.resize(max_pieces);
  for (auto& e : P.ev) OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  OK(cudaEventCreate(&P.t0));
  OK(cudaEventCreate(&P.t1));
  return P;
}

static void enqueue_path(Stream& s, Path& P, bool copies, bool absorbs, const uint8_t* src, int skip = 0) {
  OK(cudaEventRecord(P.t0, P.abs));
  if (absorbs) OK(cudaMemcpyAsync(s.key_dev, s.key_host, 32, cudaMemcpyHostToDevice, P.abs));
  if (absorbs && !copies) {
    absorb_kernel<<<(unsigned)s.groups, GROUP, 0, P.abs>>>((const uint4*)src, 0, s.key_dev, s.cvs, s.nodes);
  } else {
    for (uint64_t p = 0; p < s.pieces; ++p) {
      const size_t off = p * s.piece, len = std::min(s.piece, s.n - off);
      cudaStream_t c = P.copy[p % COPY_STREAMS];
      if (!(skip > 0 && p % skip == 0)) OK(cudaMemcpyAsync(s.dev + off, s.host + off, len, cudaMemcpyHostToDevice, c));
      OK(cudaEventRecord(P.ev[p], c));
      if (absorbs) {
        OK(cudaStreamWaitEvent(P.abs, P.ev[p]));
        absorb_kernel<<<(unsigned)(len / GROUP_BYTES), GROUP, 0, P.abs>>>((const uint4*)s.dev, off / GROUP_BYTES,
                                                                          s.key_dev, s.cvs, s.nodes);
      }
    }
    if (!absorbs)
      for (uint64_t p = s.pieces > COPY_STREAMS ? s.pieces - COPY_STREAMS : 0; p < s.pieces; ++p)
        OK(cudaStreamWaitEvent(P.abs, P.ev[p]));
  }
  if (absorbs) {
    absorb_final<<<1, FINAL_THREADS, 0, P.abs>>>(s.nodes, (uint32_t)s.groups, s.key_dev, s.D_dev);
    OK(cudaMemcpyAsync(s.D_host, s.D_dev, 32, cudaMemcpyDeviceToHost, P.abs));
  }
  OK(cudaEventRecord(P.t1, P.abs));
}

// ---------------------------------------------------------------------------------------------------------------------
// Copies: `total` bytes host to device (or back) in pieces round-robin over `streams`; returns GB/s by device events.

struct Copier {
  std::vector<cudaStream_t> s;
  std::vector<cudaEvent_t> e0, e1;
  explicit Copier(int n) : s(n), e0(n), e1(n) {
    for (int i = 0; i < n; ++i) {
      OK(cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking));
      OK(cudaEventCreate(&e0[i]));
      OK(cudaEventCreate(&e1[i]));
    }
  }
  // Enqueue; `ring` > 0 wraps the device offsets in a ring of that many bytes.
  void enqueue(uint8_t* dst, const uint8_t* src, size_t total, size_t piece, cudaMemcpyKind kind, size_t src_len,
               size_t ring = 0) {
    for (size_t i = 0; i < s.size(); ++i) OK(cudaEventRecord(e0[i], s[i]));
    size_t k = 0;
    for (size_t off = 0; off < total; off += piece, ++k) {
      const size_t len = std::min(piece, total - off);
      const size_t so = off % src_len, d = ring ? off % ring : off;
      OK(cudaMemcpyAsync(dst + d, src + so, std::min(len, src_len - so), kind, s[k % s.size()]));
    }
    for (size_t i = 0; i < s.size(); ++i) OK(cudaEventRecord(e1[i], s[i]));
  }
  // The span from the earliest start to the latest end, ms.
  double span_ms() {
    for (auto x : s) OK(cudaStreamSynchronize(x));
    double best = 0;
    for (size_t i = 0; i < s.size(); ++i)
      for (size_t j = 0; j < s.size(); ++j) {
        float ms;
        OK(cudaEventElapsedTime(&ms, e0[i], e1[j]));
        best = std::max(best, (double)ms);
      }
    return best;
  }
};

struct Timer {
  cudaEvent_t a, b;
  Timer() {
    OK(cudaEventCreate(&a));
    OK(cudaEventCreate(&b));
  }
  float ms() {
    OK(cudaEventSynchronize(b));
    float t;
    OK(cudaEventElapsedTime(&t, a, b));
    return t;
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// The modes.

struct Args {
  std::string mode, out = ".", vectors, expect, suffix;
  std::string path = "streams", priority = "high";  // window: the stream's enqueue (streams or graph), its priority
  uint64_t blocks = 524288;
  int reps = 7, windows = 20, samples = 1849, gen_threads = 16, live = 0;
  bool flush = false;  // window --live: evict the challenger's buffer from the host's caches before σ
  size_t piece = 1 << 20, interfere_mib = 64, first_piece = 2 << 20;
  std::vector<size_t> sizes_mib = {16, 32, 48, 56, 60, 64};
  double deadline_us = 0;              // foldlive: fold_live stops this long after it starts (0: never)
  std::vector<int> fold_ctas = {2, 4, 8};  // foldlive: the grids, in whole SMs
};

static Device dev;
static std::vector<int> cpus;

static void write(const Args& a, const std::string& name, const std::string& json) {
  std::ofstream f(a.out + "/" + name + a.suffix + ".json");
  f << json << "\n";
  printf("%s\n", json.c_str());
  fflush(stdout);
}

static int fail(const char* what) {
  fprintf(stderr, "GATE FAILED: %s\n", what);
  printf("GATE FAILED: %s\n", what);
  return 3;
}

// gates: host keystream, device absorb and host reference, tag root.
static int mode_gates(const Args& a) {
  std::string v = slurp(a.vectors);
  std::string j = "{";
  // 1. The host generators against the reference keystream at two counters (one across a 32-bit carry).
  std::vector<uint8_t> key = unhex(field(v, "key_s"));
  Gen gen(key.data());
  for (const char* which : {"low", "carry"}) {
    std::string pre = std::string("keystream_") + which;
    uint64_t first = std::stoull(field(v, pre + "_counter"));
    std::vector<uint8_t> want = unhex(field(v, pre));
    const uint64_t blocks = want.size() / 16;
    alignas(64) static uint8_t got[3][4096];
    sg_ctr_vaes(gen.rk, first, got[0], blocks);
    sg_ctr_aesni(gen.rk, first, got[1], blocks);
    int ossl = sg_ctr_openssl(gen.key, first, got[2], blocks);
    for (int i = 0; i < 3; ++i)
      if ((i < 2 || ossl == 0) && memcmp(got[i], want.data(), want.size())) return fail(pre.c_str());
    j += "\"" + pre + "\":{\"vaes\":true,\"aesni\":true,\"openssl\":" + (ossl == 0 ? "true" : "\"absent\"") + "},";
  }
  // 2. The device absorb and the host reference against the reference digests, for each pinned stream.
  std::vector<uint8_t> seed = unhex(field(v, "seed"));
  uint32_t kw[8];
  absorb_key(seed.data(), kw);
  if (hex((const uint8_t*)kw, 32) != field(v, "key_a")) return fail("absorb key");
  for (const char* name : {"two", "three"}) {
    std::string pre = std::string("stream_") + name;
    size_t n = std::stoull(field(v, pre + "_bytes"));
    Stream s;
    alloc_stream(s, n, GROUP_BYTES);
    generate(gen, s.host, n, 1, cpus);
    memcpy(s.key_host, kw, 32);
    s.exec = build_graph(s, 0, 0);
    OK(cudaMemset(s.dev, 0xA5, n));
    OK(cudaMemset(s.D_dev, 0xA5, 32));
    OK(cudaGraphLaunch(s.exec, 0));
    OK(cudaDeviceSynchronize());
    uint32_t ref[8];
    std::vector<uint32_t> nodes;
    reference_digest(kw, s.host, n, 4, cpus, ref, &nodes);
    std::string want = field(v, pre + "_digest");
    if (hex((const uint8_t*)s.D_host, 32) != want) return fail((pre + " device digest").c_str());
    if (hex((const uint8_t*)ref, 32) != want) return fail((pre + " host digest").c_str());
    std::vector<uint32_t> dn(s.groups * 8);
    OK(cudaMemcpy(dn.data(), s.nodes, dn.size() * 4, cudaMemcpyDeviceToHost));
    if (hex((const uint8_t*)dn.data(), dn.size() * 4) != field(v, pre + "_group_nodes")) return fail("group nodes");
    // the control: the same graph without the stream's copies over a poisoned buffer must not match
    cudaGraphExec_t ctl = build_graph(s, 1, 0);
    OK(cudaMemset(s.dev, 0xA5, n));
    OK(cudaGraphLaunch(ctl, 0));
    OK(cudaDeviceSynchronize());
    cudaGraphExecDestroy(ctl);
    if (hex((const uint8_t*)s.D_host, 32) == want) return fail("control matched");
    // the fold leaf stream-fold/v1, zero copy over the challenger's buffer: fold_chunks, and fold_live on two CTAs behind
    // a frontier that covers the stream; behind a frontier of another epoch, with a 1 ns deadline, fold_live starts no
    // step and writes nothing; one byte of S changed moves exactly its fold byte
    const std::string fwant = field(v, pre + "_fold");
    uint8_t* hdev;
    OK(cudaHostGetDevicePointer((void**)&hdev, s.host, 0));
    unsigned long long *fr, *fr_dev;
    OK(cudaHostAlloc(&fr, 64, cudaHostAllocMapped));
    OK(cudaHostGetDevicePointer((void**)&fr_dev, fr, 0));
    *fr = 1ull << 32 | 1;
    uint4* fout;
    uint32_t* fsteps;
    OK(cudaMalloc(&fout, s.chunks * 16));
    OK(cudaMalloc(&fsteps, 2 * 4));
    std::vector<uint8_t> fg(s.chunks * 16);
    uint32_t st[2];
    auto run_fold = [&](int how) {
      OK(cudaMemset(fout, 0, fg.size()));
      OK(cudaMemset(fsteps, 0xFF, sizeof st));
      if (how == 0)
        fold_chunks<<<1, 1024>>>((const uint4*)hdev, s.chunks, fout);
      else
        fold_live<<<2, 1024>>>((const uint4*)hdev, s.chunks, fout, fr_dev, s.chunks, how, how == 2 ? 1 : 0, fsteps);
      OK(cudaGetLastError());
      OK(cudaDeviceSynchronize());
      OK(cudaMemcpy(fg.data(), fout, fg.size(), cudaMemcpyDeviceToHost));
      OK(cudaMemcpy(st, fsteps, sizeof st, cudaMemcpyDeviceToHost));
    };
    run_fold(0);
    if (hex(fg.data(), fg.size()) != fwant) return fail((pre + " fold_chunks").c_str());
    run_fold(1);
    if (hex(fg.data(), fg.size()) != fwant) return fail((pre + " fold_live").c_str());
    if (st[0] + st[1] != (s.chunks + 127) / 128) return fail((pre + " fold_live steps").c_str());
    run_fold(2);
    if (st[0] + st[1] != 0 || std::any_of(fg.begin(), fg.end(), [](uint8_t b) { return b != 0; }))
      return fail((pre + " fold_live deadline").c_str());
    const size_t at = s.n / 2 + 37;
    s.host[at] ^= 0x5A;
    run_fold(0);
    s.host[at] ^= 0x5A;
    const std::vector<uint8_t> fw = unhex(fwant);
    for (size_t k = 0; k < fg.size(); ++k)
      if ((fg[k] ^ fw[k]) != (k == 16 * (at / CHUNK) + at % 16 ? 0x5A : 0)) return fail((pre + " fold control").c_str());
    cudaFree(fout);
    cudaFree(fsteps);
    cudaFreeHost(fr);
    j += "\"" + pre + "\":{\"bytes\":" + std::to_string(n) +
         ",\"device\":true,\"host\":true,\"control_rejected\":true,\"fold_chunks\":true,\"fold_live\":true,"
         "\"fold_deadline\":true,\"fold_control\":true},";
    free_stream(s);
  }
  // 3. The tag pass over the full store against expect.json's ro-blake3/v1 root.
  std::string e = slurp(a.expect);
  const size_t full = e.find("\"full\"");
  if (full == std::string::npos) return fail("expect.json has no full store");
  e = e.substr(full);
  if (std::stoull(field(e, "blocks")) != a.blocks) return fail("expect.json is for another store size");
  std::vector<uint8_t> tseed = unhex(field(e, "seed"));
  uint8_t* store;
  OK(cudaMalloc(&store, a.blocks * bw::BLOCK_BYTES));
  if (bw_fill_test_store(store, a.blocks)) return fail("fill store");
  void* ws;
  OK(cudaMalloc(&ws, bw_workspace_bytes(bw::TAG_B3, a.blocks)));
  uint8_t root[64];
  if (bw_tag_pass(bw::TAG_B3, store, a.blocks, tseed.data(), ws, root, nullptr)) return fail("tag pass");
  const size_t at = e.find("\"ro-blake3/v1\"");
  if (at == std::string::npos || hex(root, 64) != field(e.substr(at), "ro-blake3/v1")) return fail("tag root");
  cudaFree(store);
  cudaFree(ws);
  j += "\"tag_root\":true,\"passed\":true}";
  write(a, "gates", j);
  return 0;
}

// The link's ceilings.
static int mode_link(const Args& a) {
  const size_t N = 1ull << 30;
  uint8_t *h, *h2, *d, *d2, *hm;
  OK(cudaHostAlloc(&h, N, cudaHostAllocDefault));
  OK(cudaHostAlloc(&h2, N, cudaHostAllocDefault));
  OK(cudaHostAlloc(&hm, N, cudaHostAllocMapped));
  OK(cudaMalloc(&d, N));
  OK(cudaMalloc(&d2, N));
  memset(h, 1, N);
  memset(h2, 2, N);
  memset(hm, 3, N);
  uint8_t* hm_dev;
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  std::string j = "{\"bytes\":" + std::to_string(N) + ",";
  // H2D and D2H over piece sizes and stream counts.
  double best_h2d = 0, best_d2h = 0;
  size_t best_piece = 0;
  int best_streams = 0;
  j += "\"h2d\":[";
  bool first = true;
  for (size_t piece : {256u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20})
    for (int ns : {1, 2, 4, 8}) {
      Copier c(ns);
      std::vector<double> r;
      for (int rep = 0; rep < a.reps; ++rep) {
        c.enqueue(d, h, N, piece, cudaMemcpyHostToDevice, N);
        r.push_back(N / (c.span_ms() * 1e-3) / 1e9);
      }
      double m = median(r);
      if (m > best_h2d) best_h2d = m, best_piece = piece, best_streams = ns;
      j += std::string(first ? "" : ",") + "{\"piece\":" + std::to_string(piece) + ",\"streams\":" + std::to_string(ns) +
           ",\"gbs\":" + stats(r) + "}";
      first = false;
    }
  j += "],\"d2h\":[";
  first = true;
  for (size_t piece : {1u << 20, 4u << 20, 16u << 20})
    for (int ns : {1, 2, 4}) {
      Copier c(ns);
      std::vector<double> r;
      for (int rep = 0; rep < a.reps; ++rep) {
        c.enqueue(h2, d2, N, piece, cudaMemcpyDeviceToHost, N);
        r.push_back(N / (c.span_ms() * 1e-3) / 1e9);
      }
      best_d2h = std::max(best_d2h, median(r));
      j += std::string(first ? "" : ",") + "{\"piece\":" + std::to_string(piece) + ",\"streams\":" + std::to_string(ns) +
           ",\"gbs\":" + stats(r) + "}";
      first = false;
    }
  j += "],";
  // Zero copy: a kernel reading pinned host memory, over grid sizes.
  cudaStream_t zs;
  OK(cudaStreamCreateWithFlags(&zs, cudaStreamNonBlocking));
  Timer zt;
  double best_zc = 0;
  unsigned best_grid = 0, best_threads = 0;
  j += "\"zero_copy\":[";
  first = true;
  for (unsigned per : {1u, 2u, 4u, 8u})
    for (unsigned threads : {256u, 512u, 1024u}) {
      std::vector<double> r;
      for (int rep = 0; rep < a.reps; ++rep) {
        OK(cudaEventRecord(zt.a, zs));
        read_any<<<dev.sms * per, threads, 0, zs>>>((const uint4*)hm_dev, N / 16, dev.sink);
        OK(cudaEventRecord(zt.b, zs));
        r.push_back(N / (zt.ms() * 1e-3) / 1e9);
      }
      double m = median(r);
      if (m > best_zc) best_zc = m, best_grid = dev.sms * per, best_threads = threads;
      j += std::string(first ? "" : ",") + "{\"grid\":" + std::to_string(dev.sms * per) +
           ",\"threads\":" + std::to_string(threads) + ",\"gbs\":" + stats(r) + "}";
      first = false;
    }
  j += "],";
  // Concurrent: each arm's bytes sized from its alone rate so that, sharing the link, they end together; rates by
  // each arm's own span, and the union's bytes over the union's span.
  auto concurrent = [&](bool h2d, bool zc, bool d2h, double fh, double fz, double fd) {
    const double target_ms = 40;
    size_t bh = h2d ? (size_t)(best_h2d * fh * target_ms * 1e6) / best_piece * best_piece : 0;
    size_t bz = zc ? (size_t)(best_zc * fz * target_ms * 1e6) / 4096 * 4096 : 0;
    size_t bd = d2h ? (size_t)(best_d2h * fd * target_ms * 1e6) / best_piece * best_piece : 0;
    bh = std::min(bh, N), bz = std::min(bz, N), bd = std::min(bd, N);
    Copier ch(best_streams), cd(best_streams);
    cudaEvent_t start, end;
    OK(cudaEventCreate(&start));
    OK(cudaEventCreate(&end));
    OK(cudaEventRecord(start, zs));
    for (auto x : ch.s) OK(cudaStreamWaitEvent(x, start));
    for (auto x : cd.s) OK(cudaStreamWaitEvent(x, start));
    if (bh) ch.enqueue(d, h, bh, best_piece, cudaMemcpyHostToDevice, N);
    if (bd) cd.enqueue(h2, d2, bd, best_piece, cudaMemcpyDeviceToHost, N);
    OK(cudaEventRecord(zt.a, zs));
    if (bz) read_any<<<best_grid, best_threads, 0, zs>>>((const uint4*)hm_dev, bz / 16, dev.sink);
    OK(cudaEventRecord(zt.b, zs));
    if (bh) for (size_t i = 0; i < ch.s.size(); ++i) OK(cudaStreamWaitEvent(zs, ch.e1[i]));
    if (bd) for (size_t i = 0; i < cd.s.size(); ++i) OK(cudaStreamWaitEvent(zs, cd.e1[i]));
    OK(cudaEventRecord(end, zs));
    OK(cudaEventSynchronize(end));
    float span;
    OK(cudaEventElapsedTime(&span, start, end));
    double rh = bh ? bh / (ch.span_ms() * 1e-3) / 1e9 : 0, rz = bz ? bz / (zt.ms() * 1e-3) / 1e9 : 0,
           rd = bd ? bd / (cd.span_ms() * 1e-3) / 1e9 : 0;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return std::vector<double>{rh, rz, rd, (bh + bz) / (span * 1e-3) / 1e9, span};
  };
  struct Case {
    const char* name;
    bool h2d, zc, d2h;
  } cases[] = {{"h2d+zero_copy", true, true, false}, {"h2d+d2h", true, false, true}, {"h2d+zero_copy+d2h", true, true, true}};
  j += "\"concurrent\":[";
  first = true;
  for (const Case& cs : cases) {
    // First pass at half the link each, then resize to the shares the first pass measured.
    std::vector<double> r0 = concurrent(cs.h2d, cs.zc, cs.d2h, 0.5, 0.5, 1.0);
    double tot = r0[0] + r0[1];
    double fh = tot > 0 ? r0[0] / best_h2d : 0.5, fz = tot > 0 && best_zc > 0 ? r0[1] / best_zc : 0.5;
    std::vector<double> rh, rz, rd, ri;
    for (int rep = 0; rep < a.reps; ++rep) {
      std::vector<double> r = concurrent(cs.h2d, cs.zc, cs.d2h, fh, fz, 1.0);
      rh.push_back(r[0]), rz.push_back(r[1]), rd.push_back(r[2]), ri.push_back(r[3]);
    }
    j += std::string(first ? "" : ",") + "{\"case\":\"" + cs.name + "\",\"h2d_gbs\":" + stats(rh) +
         ",\"zero_copy_gbs\":" + stats(rz) + ",\"d2h_gbs\":" + stats(rd) + ",\"ingress_gbs\":" + stats(ri) + "}";
    first = false;
  }
  j += "],\"best\":{\"h2d_gbs\":" + num(best_h2d) + ",\"piece\":" + std::to_string(best_piece) +
       ",\"streams\":" + std::to_string(best_streams) + ",\"d2h_gbs\":" + num(best_d2h) + ",\"zero_copy_gbs\":" +
       num(best_zc) + ",\"zero_copy_grid\":" + std::to_string(best_grid) + ",\"zero_copy_threads\":" +
       std::to_string(best_threads) + "}}";
  write(a, "link", j);
  cudaFreeHost(h);
  cudaFreeHost(h2);
  cudaFreeHost(hm);
  cudaFree(d);
  cudaFree(d2);
  return 0;
}

// HBM: a device read of 4 GiB alone and beside H2D, and beside zero copy.
static int mode_hbm(const Args& a) {
  const size_t R = 4ull << 30, N = 1ull << 30, ring = 8u << 20, piece = 4u << 20;
  uint8_t *buf, *d, *h, *hm, *hm_dev;
  OK(cudaMalloc(&buf, R));
  OK(cudaMemset(buf, 7, R));
  OK(cudaMalloc(&d, N));
  OK(cudaHostAlloc(&h, N, cudaHostAllocDefault));
  OK(cudaHostAlloc(&hm, N, cudaHostAllocMapped));
  memset(h, 1, N);
  memset(hm, 3, N);
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  cudaStream_t rs, zs;
  OK(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));
  OK(cudaStreamCreateWithFlags(&zs, cudaStreamNonBlocking));
  Timer rt, zt;
  // The read: a full grid of 512-thread CTAs, eight a SM, as bw_read_scrub.  Alone first.
  auto read = [&]() {
    OK(cudaEventRecord(rt.a, rs));
    read_any<<<dev.sms * 8, 512, 0, rs>>>((const uint4*)buf, R / 16, dev.sink);
    OK(cudaEventRecord(rt.b, rs));
  };
  std::string j = "{\"read_bytes\":" + std::to_string(R) + ",";
  std::vector<double> alone;
  for (int rep = 0; rep < a.reps; ++rep) {
    read();
    alone.push_back(R / (rt.ms() * 1e-3) / 1e9);
  }
  j += "\"read_alone_gbs\":" + stats(alone) + ",\"beside\":[";
  struct Case {
    const char* name;
    size_t ring;
    bool zc;
  } cases[] = {{"h2d_1gib", 0, false}, {"h2d_8mib_ring", ring, false}, {"zero_copy_8sms", 0, true}};
  bool first = true;
  for (const Case& cs : cases) {
    std::vector<double> rr, side;
    for (int rep = 0; rep < a.reps; ++rep) {
      Copier c(4);
      // The side traffic starts first and outlasts the read: 512 MiB at about 56 GB/s is 9 ms against about 3 ms.
      const size_t side_bytes = 512u << 20;
      if (cs.zc) {
        OK(cudaEventRecord(zt.a, zs));
        read_any<<<8, 1024, 0, zs>>>((const uint4*)hm_dev, side_bytes / 16, dev.sink);
        OK(cudaEventRecord(zt.b, zs));
      } else {
        c.enqueue(d, h, side_bytes, piece, cudaMemcpyHostToDevice, N, cs.ring);
      }
      // Let the side traffic reach its rate before the read starts.
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      read();
      double rms = rt.ms();
      double sms = cs.zc ? zt.ms() : c.span_ms();
      rr.push_back(R / (rms * 1e-3) / 1e9);
      side.push_back(side_bytes / (sms * 1e-3) / 1e9);
      OK(cudaDeviceSynchronize());
    }
    j += std::string(first ? "" : ",") + "{\"case\":\"" + cs.name + "\",\"read_gbs\":" + stats(rr) +
         ",\"side_gbs\":" + stats(side) + "}";
    first = false;
  }
  j += "]}";
  write(a, "hbm", j);
  cudaFree(buf);
  cudaFree(d);
  cudaFreeHost(h);
  cudaFreeHost(hm);
  return 0;
}

// The host keystream's rate.
static int mode_gen(const Args& a) {
  uint8_t key[16];
  fresh(key, 16);
  Gen gen(key);
  std::string j = "{\"cpus\":" + std::to_string(cpus.size()) + ",\"order\":[";
  for (size_t i = 0; i < cpus.size(); ++i) j += (i ? "," : "") + std::to_string(cpus[i]);
  j += "],";
  const double dur = 0.25;
  // Per thread: into a 1 MiB buffer (in cache) and into its share of a 1 GiB pinned buffer (memory).
  auto run = [&](int threads, int impl, size_t buf_bytes, uint8_t* big) {
    std::vector<double> per(threads);
    parallel(threads, cpus, [&](int t) {
      uint8_t* out = big ? big + t * buf_bytes : (uint8_t*)aligned_alloc(64, buf_bytes);
      const uint64_t blocks = buf_bytes / 16;
      uint64_t done = 0, ctr = (uint64_t)t << 40;
      double t0 = now_s(), t1 = t0;
      while (t1 - t0 < dur) {
        if (impl == 0) sg_ctr_vaes(gen.rk, ctr, out, blocks);
        else if (impl == 1) sg_ctr_aesni(gen.rk, ctr, out, blocks);
        else if (impl == 2) sg_ctr_openssl(gen.key, ctr, out, blocks);
        else sg_ctr_vaes_nt(gen.rk, ctr, out, blocks);
        ctr += blocks;
        done += buf_bytes;
        t1 = now_s();
      }
      per[t] = done / (t1 - t0) / 1e9;
      if (!big) free(out);
    });
    double sum = 0;
    for (double x : per) sum += x;
    return sum;
  };
  const char* names[] = {"vaes", "aesni", "openssl"};
  j += "\"one_thread_cache_gbs\":{";
  for (int impl = 0; impl < 3; ++impl) j += std::string(impl ? "," : "") + "\"" + names[impl] + "\":" + num(run(1, impl, 1 << 20, nullptr));
  j += "},";
  const size_t big_bytes = 1ull << 30;
  uint8_t* big;
  OK(cudaHostAlloc(&big, big_bytes, cudaHostAllocDefault));
  memset(big, 0, big_bytes);
  j += "\"threads\":[";
  bool first = true;
  for (int t : {1, 2, 4, 6, 8, 12, 16, 24, 32, 48}) {
    if (t > (int)cpus.size()) break;
    double cache = run(t, 0, 1 << 20, nullptr);
    double mem = run(t, 0, (big_bytes / t) / 512 * 512, big);
    double nt = run(t, 3, (big_bytes / t) / 512 * 512, big);
    j += std::string(first ? "" : ",") + "{\"threads\":" + std::to_string(t) + ",\"vaes_cache_gbs\":" + num(cache) +
         ",\"vaes_pinned_gbs\":" + num(mem) + ",\"vaes_nt_pinned_gbs\":" + num(nt) + "}";
    first = false;
  }
  j += "]}";
  cudaFreeHost(big);
  write(a, "gen", j);
  return 0;
}

// Sustained generation and H2D: G threads fill a pinned ring of slots (regular or non-temporal stores), each taking the
// next piece when it is free, and one thread copies them out in order on COPY_STREAMS streams to a device ring, for
// each slot size and G.
static int mode_pipe(const Args& a) {
  const size_t ring_bytes = 256u << 20, dring = 64u << 20;
  uint8_t *ring, *d;
  OK(cudaHostAlloc(&ring, ring_bytes, cudaHostAllocDefault));
  memset(ring, 0, ring_bytes);
  OK(cudaMalloc(&d, dring));
  uint8_t key[16];
  fresh(key, 16);
  Gen gen(key);
  std::string j = "{\"ring_bytes\":" + std::to_string(ring_bytes) + ",\"runs\":[";
  bool first = true;
  for (size_t slot : {2u << 20, 8u << 20})
    for (int nt = 0; nt < 2; ++nt)
      for (int G : {2, 3, 4, 6, 8}) {
        if (G + 1 > (int)cpus.size()) break;
        const int R = (int)(ring_bytes / slot);
        std::vector<std::atomic<int64_t>> ready(R), freed(R);
        for (int i = 0; i < R; ++i) ready[i] = -1, freed[i] = i;  // slot i may take piece freed[i]
        std::atomic<bool> stop{false};
        std::atomic<int64_t> next{0};  // pieces go to whichever generator is free, so one slow core delays only its own
        std::vector<std::thread> th;
        for (int t = 0; t < G; ++t)
          th.emplace_back([&, t] {
            pin(cpus[1 + t]);
            for (int64_t k = next.fetch_add(1); !stop.load(std::memory_order_relaxed); k = next.fetch_add(1)) {
              const int sl = (int)(k % R);
              while (freed[sl].load(std::memory_order_acquire) != k)
                if (stop.load(std::memory_order_relaxed)) return;
              (nt ? sg_ctr_vaes_nt : sg_ctr_vaes)(gen.rk, (uint64_t)k * (slot / 16), ring + sl * slot, slot / 16);
              ready[sl].store(k, std::memory_order_release);
            }
          });
        pin(cpus[0]);
        std::vector<cudaStream_t> ss(COPY_STREAMS);
        for (auto& x : ss) OK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
        std::vector<cudaEvent_t> ev(R);
        for (auto& e : ev) OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        const double dur = 0.5;
        int64_t sent = 0, done = 0, starved = 0;
        const double t0 = now_s();
        while (now_s() - t0 < dur) {
          while (done < sent && cudaEventQuery(ev[done % R]) == cudaSuccess) {
            freed[done % R].store(done + R, std::memory_order_release);
            ++done;
          }
          if (sent - done < R && ready[sent % R].load(std::memory_order_acquire) == sent) {
            const int sl = (int)(sent % R);
            cudaStream_t c = ss[sent % COPY_STREAMS];
            OK(cudaMemcpyAsync(d + (sent * slot) % dring, ring + sl * slot, slot, cudaMemcpyHostToDevice, c));
            OK(cudaEventRecord(ev[sl], c));
            ++sent;
          } else if (sent - done < R) {
            ++starved;
          }
        }
        for (auto& x : ss) OK(cudaStreamSynchronize(x));
        const double t2 = now_s();
        stop = true;
        for (auto& x : th) x.join();
        j += std::string(first ? "" : ",") + "{\"slot\":" + std::to_string(slot) + ",\"nt\":" + (nt ? "true" : "false") +
             ",\"generators\":" + std::to_string(G) + ",\"gbs\":" + num(sent * (double)slot / (t2 - t0) / 1e9) +
             ",\"pieces\":" + std::to_string(sent) + ",\"copier_starved_polls\":" + std::to_string(starved) + "}";
        first = false;
        for (auto& x : ss) cudaStreamDestroy(x);
        for (auto& e : ev) cudaEventDestroy(e);
      }
  j += "]}";
  cudaFreeHost(ring);
  cudaFree(d);
  write(a, "pipe", j);
  return 0;
}

// The absorb's own rate: from device memory with the whole GPU, and from pinned host memory.
static int mode_absorb(const Args& a) {
  const size_t n = 64u << 20;
  Stream s;
  alloc_stream(s, n, a.piece);
  uint8_t key[16];
  fresh(key, 16);
  Gen gen(key);
  generate(gen, s.host, n, a.gen_threads, cpus);
  OK(cudaMemcpy(s.dev, s.host, n, cudaMemcpyHostToDevice));
  uint8_t* hm;
  OK(cudaHostAlloc(&hm, n, cudaHostAllocMapped));
  memcpy(hm, s.host, n);
  uint8_t* hm_dev;
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  uint8_t seed[32];
  fresh(seed, 32);
  absorb_key(seed, s.key_host);
  uint32_t ref[8];
  reference_digest(s.key_host, s.host, n, a.gen_threads, cpus, ref);
  cudaStream_t st;
  OK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Timer t;
  std::string j = "{\"bytes\":" + std::to_string(n);
  for (int src = 0; src < 2; ++src) {
    std::vector<double> r;
    for (int rep = 0; rep < a.reps; ++rep) {
      OK(cudaMemsetAsync(s.D_dev, 0, 32, st));
      bw_read_scrub(dev.scrub, SCRUB);
      OK(cudaEventRecord(t.a, st));
      absorb_resident(s, src ? hm_dev : s.dev, st);
      OK(cudaEventRecord(t.b, st));
      r.push_back(n / (t.ms() * 1e-3) / 1e9);
      OK(cudaStreamSynchronize(st));
      if (memcmp(s.D_host, ref, 32)) return fail("absorb digest");
    }
    j += std::string(src ? ",\"zero_copy_gbs\":" : ",\"device_gbs\":") + stats(r);
  }
  j += ",\"digest_checked\":true}";
  cudaFreeHost(hm);
  free_stream(s);
  write(a, "absorb", j);
  return 0;
}

// The tag pass's store and workspace, filled once.
struct Store {
  uint8_t* E = nullptr;
  void* ws = nullptr;
};

static Store make_store(const Args& a) {
  Store s;
  OK(cudaMalloc(&s.E, a.blocks * bw::BLOCK_BYTES));
  if (bw_fill_test_store(s.E, a.blocks)) exit(fail("fill store"));
  OK(cudaMalloc(&s.ws, bw_workspace_bytes(bw::TAG_B3, a.blocks)));
  return s;
}

static int priority_of(const std::string& name) {
  int lo, hi;
  OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  if (name == "high") return hi;
  if (name == "low") return lo;
  fprintf(stderr, "--priority is high or low\n");
  exit(2);
}

static std::vector<uint64_t> draw(uint64_t chunks, int s) {
  std::vector<uint64_t> out(s);
  for (auto& x : out) {
    uint64_t r;
    fresh((uint8_t*)&r, 8);
    x = r % chunks;
  }
  return out;
}

// Streamed windows: for each stream size, windows of the tag pass beside the stream, then the stream alone, each after
// the honest prover's preparation (an untimed pass, an untimed stream of the last window's bytes, the read scrub); the
// stream is the challenger's fresh bytes, generated before σ and sent after it (with --live, generated after σ as it is
// sent).  Every window's D is checked against the host's reference over the bytes sent and its root against an untimed
// pass under the same seed; the first window of each size times the verifier's openings; and a window whose device
// skips every 64th piece must fail both.
constexpr size_t LIVE_SLOT = 256u << 10;
static int mode_window(const Args& a) {
  Store E = make_store(a);
  int lo, hi;
  OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  const int prio = priority_of(a.priority);
  const bool graph = a.path == "graph", live = a.live > 0;
  if (live && (graph || a.live + 2 > (int)cpus.size() || a.first_piece % LIVE_SLOT || a.piece % LIVE_SLOT)) {
    fprintf(stderr, "--live G takes --path streams, G + 2 CPUs, and pieces in %zu-byte slots\n", LIVE_SLOT);
    return 2;
  }
  // --live: the main thread sends, G generators and the tag pass each have a CPU of their own.
  std::vector<std::unique_ptr<Spinner>> gens;
  std::unique_ptr<Spinner> tagger;
  if (live) {
    pin(cpus[0]);
    for (int t = 0; t < a.live; ++t) gens.emplace_back(new Spinner(cpus[1 + t]));
    tagger.reset(new Spinner(cpus[1 + a.live]));
  }
  cudaStream_t s_tag;
  OK(cudaStreamCreateWithPriority(&s_tag, cudaStreamNonBlocking, lo));
  std::string j = "{\"blocks\":" + std::to_string(a.blocks) + ",\"piece\":" + std::to_string(a.piece) +
                  ",\"path\":\"" + a.path + "\",\"priority\":\"" + a.priority + "\",\"copy_streams\":" +
                  std::to_string(graph ? BRANCHES : COPY_STREAMS) + ",\"samples\":" + std::to_string(a.samples) +
                  ",\"live_generators\":" + std::to_string(a.live) +
                  (live ? ",\"live_slot\":" + std::to_string(LIVE_SLOT) + ",\"first_piece\":" +
                              std::to_string(a.first_piece) + ",\"flush\":" + (a.flush ? "true" : "false")
                        : std::string()) +
                  ",\"sizes\":[";
  uint8_t throwaway[32] = {0}, root[64], check[64];
  std::vector<size_t> sizes{0};
  for (size_t m : a.sizes_mib) sizes.push_back(m << 20);
  bool first = true;
  for (size_t n : sizes) {
    Stream s;
    std::vector<std::pair<size_t, size_t>> spans;  // --live: the pieces (offset, bytes)
    for (size_t off = 0, len = a.first_piece; live && off < n; off += len, len = std::min(2 * len, a.piece))
      spans.push_back({off, std::min(len, n - off)});
    Path P = make_path(prio, std::max<size_t>(spans.size(), n ? (n + a.piece - 1) / a.piece : 1));
    const size_t slots = live ? n / LIVE_SLOT : 0;
    std::unique_ptr<std::atomic<uint8_t>[]> ready(new std::atomic<uint8_t>[slots + 1]);
    std::atomic<uint64_t> next{0};
    // --live, per piece: when its bytes were all written (host, from σ) and when its copy landed (device, from P.t0)
    std::vector<cudaEvent_t> landed(spans.size());
    for (auto& e : landed) OK(cudaEventCreate(&e));
    std::vector<std::vector<double>> ready_ms(spans.size()), landed_ms(spans.size());
    // generate's non-temporal stores need 64-byte alignment, which a vector's storage doesn't have
    std::unique_ptr<uint8_t, decltype(&free)> regen((uint8_t*)(live && n ? aligned_alloc(4096, n) : nullptr), &free);
    Timer gt;
    if (n) {
      alloc_stream(s, n, a.piece);
      if (graph) {
        s.exec = build_graph(s, 0, prio);
        s.skip_exec = build_graph(s, 64, prio);
      }
    }
    auto launch = [&](bool skip) {  // the stream after σ; its device span from ms()
      if (graph) {
        OK(cudaEventRecord(gt.a, P.abs));
        OK(cudaGraphLaunch(skip ? s.skip_exec : s.exec, P.abs));
        OK(cudaEventRecord(gt.b, P.abs));
      } else {
        enqueue_path(s, P, true, true, nullptr, skip ? 64 : 0);
      }
    };
    auto stream_ms = [&]() {
      float ms;
      if (graph) return (double)gt.ms();
      OK(cudaEventSynchronize(P.t1));
      OK(cudaEventElapsedTime(&ms, P.t0, P.t1));
      return (double)ms;
    };
    auto prepare = [&]() {
      bw_tag_pass(bw::TAG_B3, E.E, a.blocks, throwaway, E.ws, root, s_tag);
      if (n) {
        launch(false);
        OK(cudaStreamSynchronize(P.abs));
      }
      bw_read_scrub(dev.scrub, SCRUB);
    };
    // --live, at σ: the tag pass and the generators start, and each piece goes out once its slots are written.
    auto send_live = [&](const Gen& gen, bool tags, const uint8_t* seed, double sigma) {
      if (tags)
        tagger->post([&, seed] {
          OK(cudaSetDevice(0));
          bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, root, s_tag);
        });
      for (auto& g : gens)
        g->post([&] {
          for (uint64_t k = next.fetch_add(1); k < slots; k = next.fetch_add(1)) {
            sg_ctr_vaes_nt(gen.rk, k * (LIVE_SLOT / 16), s.host + k * LIVE_SLOT, LIVE_SLOT / 16);
            ready[k].store(1, std::memory_order_release);
          }
        });
      OK(cudaEventRecord(P.t0, P.abs));
      OK(cudaMemcpyAsync(s.key_dev, s.key_host, 32, cudaMemcpyHostToDevice, P.abs));
      for (size_t p = 0; p < spans.size(); ++p) {
        const size_t off = spans[p].first, len = spans[p].second;
        for (size_t k = off / LIVE_SLOT; k < (off + len) / LIVE_SLOT; ++k)
          while (!ready[k].load(std::memory_order_acquire)) {
          }
        if (tags) ready_ms[p].push_back((now_s() - sigma) * 1e3);
        cudaStream_t c = P.copy[p % COPY_STREAMS];
        OK(cudaMemcpyAsync(s.dev + off, s.host + off, len, cudaMemcpyHostToDevice, c));
        OK(cudaEventRecord(P.ev[p], c));
        OK(cudaEventRecord(landed[p], c));
        OK(cudaStreamWaitEvent(P.abs, P.ev[p]));
        absorb_kernel<<<(unsigned)(len / GROUP_BYTES), GROUP, 0, P.abs>>>((const uint4*)s.dev, off / GROUP_BYTES,
                                                                          s.key_dev, s.cvs, s.nodes);
      }
      absorb_final<<<1, FINAL_THREADS, 0, P.abs>>>(s.nodes, (uint32_t)s.groups, s.key_dev, s.D_dev);
      OK(cudaMemcpyAsync(s.D_host, s.D_dev, 32, cudaMemcpyDeviceToHost, P.abs));
      OK(cudaEventRecord(P.t1, P.abs));
      OK(cudaStreamSynchronize(P.abs));
      if (tags) tagger->wait();
      for (auto& g : gens) g->wait();
      for (size_t p = 0; tags && p < spans.size(); ++p) {
        float ms;
        OK(cudaEventElapsedTime(&ms, P.t0, landed[p]));
        landed_ms[p].push_back(ms);
      }
    };
    std::vector<double> both, stream_only, dev_pass, dev_stream, verify_ms;
    int bad_d = 0, bad_root = 0, bad_open = 0, bad_bytes = 0;
    for (int arm = 0; arm < (n ? 2 : 1); ++arm)
      for (int w = 0; w < a.windows; ++w) {
        uint8_t seed[32], ks[16];
        fresh(seed, 32);
        fresh(ks, 16);
        Gen gen(ks);
        prepare();
        if (n && !live) generate(gen, s.host, n, a.gen_threads, cpus);
        if (n && live) {
          if (a.flush) {
            const size_t per = (n / a.gen_threads + 63) / 64 * 64;
            parallel(a.gen_threads, cpus, [&](int t) {
              if (t * per < n) sg_flush(s.host + t * per, std::min(per, n - t * per));
            });
          }
          for (size_t k = 0; k < slots; ++k) ready[k].store(0, std::memory_order_relaxed);
          next.store(0, std::memory_order_release);
        }
        // σ: from here the window's clock runs.
        const double t0 = now_s();
        if (n && live) {
          absorb_key(seed, s.key_host);
          send_live(gen, arm == 0, seed, t0);
        } else {
          if (n) {
            absorb_key(seed, s.key_host);
            launch(false);
          }
          if (arm == 0) bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, root, s_tag);
          if (n) OK(cudaStreamSynchronize(P.abs));
        }
        const double t1 = now_s();
        if (n && live) {
          generate(gen, regen.get(), n, a.gen_threads, cpus);
          bad_bytes += memcmp(regen.get(), s.host, n) != 0;
        }
        float tags_ms = 0, pass_ms = 0;
        if (arm == 0) bw_last_pass_ms(&tags_ms, &pass_ms);
        const double sd = n ? stream_ms() : 0;
        if (n) {
          uint32_t ref[8], D[8];
          memcpy(D, s.D_host, 32);
          reference_digest(s.key_host, s.host, n, a.gen_threads, cpus, ref);
          bad_d += memcmp(D, ref, 32) != 0;
          if (w == 0 && arm == 0) {
            std::vector<uint32_t> cv(s.chunks * 8);
            OK(cudaMemcpy(cv.data(), s.cvs, cv.size() * 4, cudaMemcpyDeviceToHost));
            auto lv = tree_levels(s.key_host, std::move(cv));
            auto smp = draw(s.chunks, a.samples);
            const double v0 = now_s();
            bad_open += verify_openings(gen, s.key_host, D, s.chunks, lv, smp);
            verify_ms.push_back((now_s() - v0) * 1e3);
          }
        }
        if (arm == 0) {
          bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, check, s_tag);
          bad_root += memcmp(root, check, 64) != 0;
          both.push_back((t1 - t0) * 1e3);
          dev_pass.push_back(pass_ms);
          dev_stream.push_back(sd);
        } else {
          stream_only.push_back((t1 - t0) * 1e3);
        }
      }
    std::string control = "null";
    if (n) {
      uint8_t seed[32], ks[16];
      fresh(seed, 32);
      fresh(ks, 16);
      Gen gen(ks);
      generate(gen, s.host, n, a.gen_threads, cpus);
      OK(cudaMemset(s.dev, 0xA5, n));
      absorb_key(seed, s.key_host);
      launch(true);
      OK(cudaStreamSynchronize(P.abs));
      uint32_t ref[8], D[8];
      memcpy(D, s.D_host, 32);
      reference_digest(s.key_host, s.host, n, a.gen_threads, cpus, ref);
      std::vector<uint32_t> cv(s.chunks * 8);
      OK(cudaMemcpy(cv.data(), s.cvs, cv.size() * 4, cudaMemcpyDeviceToHost));
      auto lv = tree_levels(s.key_host, std::move(cv));
      uint32_t own[8];  // the prover's best: its own D, with paths in its own tree
      memcpy(own, lv.back().data(), 32);
      int failed = verify_openings(gen, s.key_host, own, s.chunks, lv, draw(s.chunks, a.samples));
      control = "{\"skipped_pieces\":" + std::to_string((s.pieces + 63) / 64) + ",\"digest_matches\":" +
                (memcmp(D, ref, 32) ? "false" : "true") + ",\"openings_failed\":" + std::to_string(failed) + "}";
      if (!memcmp(D, ref, 32) || failed == 0) return fail("skip control accepted");
    }
    j += std::string(first ? "" : ",") + "{\"bytes\":" + std::to_string(n) + ",\"window_ms\":" + stats(both) +
         ",\"device_pass_ms\":" + stats(dev_pass) + ",\"device_stream_ms\":" + stats(dev_stream) +
         ",\"stream_only_ms\":" + stats(stream_only) + ",\"verify_ms\":" + stats(verify_ms) +
         ",\"bad_digest\":" + std::to_string(bad_d) + ",\"bad_root\":" + std::to_string(bad_root) +
         ",\"bad_openings\":" + std::to_string(bad_open) +
         (live ? ",\"pieces\":" + std::to_string(spans.size()) + ",\"bad_bytes\":" + std::to_string(bad_bytes) +
                     [&] {  // per piece, medians over the windows beside the tag pass
                       std::string b = ",\"piece_bytes\":[", r = "],\"piece_ready_ms\":[", l = "],\"piece_landed_ms\":[";
                       for (size_t p = 0; p < spans.size(); ++p) {
                         const char* sep = p ? "," : "";
                         b += sep + std::to_string(spans[p].second);
                         r += sep + num(median(ready_ms[p]));
                         l += sep + num(median(landed_ms[p]));
                       }
                       return b + r + l + "]";
                     }()
               : std::string()) +
         ",\"skip_control\":" + control + "}";
    first = false;
    if (bad_d || bad_root || bad_open || bad_bytes) {
      j += "]}";
      write(a, "window", j);
      return fail("a window's digest, root, openings or bytes");
    }
    for (auto& e : landed) cudaEventDestroy(e);
    if (n) free_stream(s);
  }
  j += "]}";
  write(a, "window", j);
  cudaFree(E.E);
  cudaFree(E.ws);
  return 0;
}

// What the stream costs the tag pass, part by part: the pass alone and beside the copies alone, the copies and
// absorbs, the absorbs over resident bytes, the absorbs straight from host memory (zero copy) and the graph, for each
// absorb priority and piece size; then the stream alone.  The stream is one fixed key and bytes, its D checked against
// the host's reference in every arm that absorbs.
static int mode_interfere(const Args& a) {
  Store E = make_store(a);
  int lo, hi;
  OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s_tag;
  OK(cudaStreamCreateWithPriority(&s_tag, cudaStreamNonBlocking, lo));
  const size_t n = a.interfere_mib << 20;
  uint8_t ks[16], seed[32], throwaway[32] = {0}, root[64], want[64];
  fresh(ks, 16);
  fresh(seed, 32);
  Gen gen(ks);
  uint8_t *hm, *hm_dev;
  OK(cudaHostAlloc(&hm, n, cudaHostAllocMapped));
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  generate(gen, hm, n, a.gen_threads, cpus);
  if (bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, want, s_tag)) return fail("tag pass");
  std::string j = "{\"blocks\":" + std::to_string(a.blocks) + ",\"stream_bytes\":" + std::to_string(n) +
                  ",\"copy_streams\":" + std::to_string(COPY_STREAMS) + ",\"configs\":[";
  bool first = true;
  int bad = 0;
  for (const char* pname : {"high", "low"})
    for (size_t piece : {1u << 20, 4u << 20, 16u << 20}) {
      const int prio = pname[0] == 'h' ? hi : lo;
      Stream s;
      alloc_stream(s, n, piece);
      memcpy(s.host, hm, n);
      OK(cudaMemcpy(s.dev, s.host, n, cudaMemcpyHostToDevice));
      absorb_key(seed, s.key_host);
      uint32_t ref[8];
      reference_digest(s.key_host, s.host, n, a.gen_threads, cpus, ref);
      Path P = make_path(prio, s.pieces);
      s.exec = build_graph(s, 0, prio);
      enum Arm { TAG, COPIES, STREAM, RESIDENT, ZERO_COPY, GRAPH, STREAM_ALONE, ZERO_COPY_ALONE };
      const char* names[] = {"tag",   "tag+copies", "tag+stream",   "tag+absorb_resident", "tag+absorb_zero_copy",
                             "tag+graph", "stream_alone", "absorb_zero_copy_alone"};
      j += std::string(first ? "" : ",") + "{\"priority\":\"" + pname + "\",\"piece\":" + std::to_string(piece) +
           ",\"arms\":{";
      first = false;
      for (int arm = TAG; arm <= ZERO_COPY_ALONE; ++arm) {
        std::vector<double> wall, pass, side;
        Timer gt;
        for (int r = 0; r < a.windows; ++r) {
          bw_tag_pass(bw::TAG_B3, E.E, a.blocks, throwaway, E.ws, root, s_tag);
          bw_read_scrub(dev.scrub, SCRUB);
          memset(s.D_host, 0, 32);
          const double t0 = now_s();
          const bool absorbs = arm != COPIES;
          if (arm == COPIES || arm == STREAM || arm == STREAM_ALONE) enqueue_path(s, P, true, arm != COPIES, nullptr);
          if (arm == RESIDENT) enqueue_path(s, P, false, true, s.dev);
          if (arm == ZERO_COPY || arm == ZERO_COPY_ALONE) enqueue_path(s, P, false, true, hm_dev);
          if (arm == GRAPH) {
            OK(cudaEventRecord(gt.a, P.abs));
            OK(cudaGraphLaunch(s.exec, P.abs));
            OK(cudaEventRecord(gt.b, P.abs));
          }
          const bool tag = arm < STREAM_ALONE;
          if (tag) bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, root, s_tag);
          OK(cudaStreamSynchronize(P.abs));
          const double t1 = now_s();
          float tags_ms = 0, pass_ms = 0;
          if (tag) {
            bw_last_pass_ms(&tags_ms, &pass_ms);
            pass.push_back(pass_ms);
            bad += memcmp(root, want, 64) != 0;
          }
          wall.push_back((t1 - t0) * 1e3);
          if (arm != TAG) {
            float ms;
            if (arm == GRAPH) {
              ms = gt.ms();
            } else {
              OK(cudaEventSynchronize(P.t1));
              OK(cudaEventElapsedTime(&ms, P.t0, P.t1));
            }
            side.push_back(ms);
            if (absorbs) bad += memcmp(s.D_host, ref, 32) != 0;
          }
        }
        j += std::string(arm ? "," : "") + "\"" + names[arm] + "\":{\"wall_ms\":" + stats(wall) +
             ",\"device_pass_ms\":" + stats(pass) + ",\"device_stream_ms\":" + stats(side) + "}";
      }
      j += "}}";
      free_stream(s);
      for (auto& c : P.copy) cudaStreamDestroy(c);
      cudaStreamDestroy(P.abs);
    }
  j += "],\"bad\":" + std::to_string(bad) + "}";
  write(a, "interfere", j);
  cudaFreeHost(hm);
  return bad ? fail("an interference arm's root or digest") : 0;
}

// What a zero-copy absorb with a cheap leaf would cost the tag pass: the pass alone, then beside fold_chunks over
// --interfere-mib of mapped host memory (launched first, at high priority), for grids from a few whole SMs to one
// warp on every SM; then each grid alone, for its rate.  Every fold is checked against the host's, every root against
// an untimed pass.
static int mode_fold(const Args& a) {
  Store E = make_store(a);
  int lo, hi;
  OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s_tag, s_fold;
  OK(cudaStreamCreateWithPriority(&s_tag, cudaStreamNonBlocking, lo));
  OK(cudaStreamCreateWithPriority(&s_fold, cudaStreamNonBlocking, hi));
  const size_t n = a.interfere_mib << 20, chunks = n / 1024;
  uint8_t ks[16], seed[32], throwaway[32] = {0}, root[64], want[64];
  fresh(ks, 16);
  fresh(seed, 32);
  Gen gen(ks);
  uint8_t *hm, *hm_dev;
  OK(cudaHostAlloc(&hm, n, cudaHostAllocMapped));
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  generate(gen, hm, n, a.gen_threads, cpus);
  std::vector<uint32_t> ref(chunks * 4, 0), got(chunks * 4);
  for (size_t c = 0; c < chunks; ++c)
    for (size_t w = 0; w < 256; ++w) ref[4 * c + w % 4] ^= ((const uint32_t*)hm)[256 * c + w];
  uint4* dout;
  OK(cudaMalloc(&dout, chunks * 16));
  if (bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, want, s_tag)) return fail("tag pass");
  struct Grid {
    int ctas, threads;
  };
  const std::vector<Grid> grids = {{0, 0},           {4, 1024},         {8, 1024},         {16, 1024},
                                   {dev.sms, 32},    {dev.sms, 64},     {dev.sms, 128},    {2 * dev.sms, 128}};
  Timer ft;
  std::string j = "{\"blocks\":" + std::to_string(a.blocks) + ",\"bytes\":" + std::to_string(n) + ",\"grids\":[";
  int bad = 0;
  for (size_t gi = 0; gi < grids.size(); ++gi) {
    const Grid g = grids[gi];
    std::vector<double> wall, pass, fold_ms, alone_ms;
    for (int alone = 0; alone < (g.ctas ? 2 : 1); ++alone)
      for (int r = 0; r < a.windows; ++r) {
        bw_tag_pass(bw::TAG_B3, E.E, a.blocks, throwaway, E.ws, root, s_tag);
        bw_read_scrub(dev.scrub, SCRUB);
        OK(cudaMemsetAsync(dout, 0, chunks * 16, s_fold));
        OK(cudaStreamSynchronize(s_fold));
        const double t0 = now_s();
        if (g.ctas) {
          OK(cudaEventRecord(ft.a, s_fold));
          fold_chunks<<<g.ctas, g.threads, 0, s_fold>>>((const uint4*)hm_dev, chunks, dout);
          OK(cudaEventRecord(ft.b, s_fold));
        }
        if (!alone) bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, root, s_tag);
        OK(cudaStreamSynchronize(s_fold));
        const double t1 = now_s();
        if (!alone) {
          float tags_ms = 0, pass_ms = 0;
          bw_last_pass_ms(&tags_ms, &pass_ms);
          pass.push_back(pass_ms);
          wall.push_back((t1 - t0) * 1e3);
          bad += memcmp(root, want, 64) != 0;
        }
        if (g.ctas) {
          (alone ? alone_ms : fold_ms).push_back(ft.ms());
          OK(cudaMemcpy(got.data(), dout, chunks * 16, cudaMemcpyDeviceToHost));
          bad += got != ref;
        }
      }
    auto gbs = [&](const std::vector<double>& ms) { return ms.empty() ? 0.0 : n / (median(ms) * 1e-3) / 1e9; };
    j += std::string(gi ? "," : "") + "{\"ctas\":" + std::to_string(g.ctas) + ",\"threads\":" +
         std::to_string(g.threads) + ",\"window_ms\":" + stats(wall) + ",\"device_pass_ms\":" + stats(pass) +
         ",\"fold_ms\":" + stats(fold_ms) + ",\"fold_gbs\":" + num(gbs(fold_ms)) + ",\"fold_alone_ms\":" +
         stats(alone_ms) + ",\"fold_alone_gbs\":" + num(gbs(alone_ms)) + "}";
  }
  j += "],\"bad\":" + std::to_string(bad) + "}";
  write(a, "fold", j);
  cudaFree(dout);
  cudaFreeHost(hm);
  return bad ? fail("a fold or a root") : 0;
}

// The fold of a live stream: at σ the challenger starts generating S on --live G pinned threads, LIVE_SLOT slots taken
// in order, and a publisher thread advances the frontier (mapped host memory) over the slots written, in order;
// fold_live reads the stream zero-copy behind the frontier.  Beside the tag pass, alone, alone with S written and the
// frontier set before σ, and fold_chunks over that, for each size in --sizes-mib and for 2, 4 and 8 whole SMs.  With
// --deadline-us D, fold_live stops D after it starts, and the window certifies the chunks it folded: every fold is
// either S's (regenerated from K_s) or untouched, and the folds that match are the CTAs' finished steps.  The buffer is
// evicted from the host's caches before σ; every root is checked against an untimed pass.
static int mode_fold_live(const Args& a) {
  if (a.live < 1 || a.live + 2 > (int)cpus.size() ||
      std::any_of(a.fold_ctas.begin(), a.fold_ctas.end(), [](int c) { return c < 1 || c > 64; })) {
    fprintf(stderr, "foldlive takes --live G, G + 2 CPUs and --fold-ctas from 1 to 64\n");
    return 2;
  }
  Store E = make_store(a);
  int lo, hi;
  OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s_tag, s_fold;
  OK(cudaStreamCreateWithPriority(&s_tag, cudaStreamNonBlocking, lo));
  OK(cudaStreamCreateWithPriority(&s_fold, cudaStreamNonBlocking, hi));
  pin(cpus[0]);
  std::vector<std::unique_ptr<Spinner>> gens;
  for (int t = 0; t < a.live; ++t) gens.emplace_back(new Spinner(cpus[1 + t]));
  Spinner publisher(cpus[1 + a.live]);
  size_t nmax = 0;
  for (size_t m : a.sizes_mib) nmax = std::max(nmax, m << 20);
  const size_t smax = nmax / LIVE_SLOT;
  uint8_t *hm, *hm_dev;
  unsigned long long *frontier, *frontier_dev;
  OK(cudaHostAlloc(&hm, nmax, cudaHostAllocMapped));
  OK(cudaHostGetDevicePointer((void**)&hm_dev, hm, 0));
  OK(cudaHostAlloc(&frontier, 64, cudaHostAllocMapped));
  OK(cudaHostGetDevicePointer((void**)&frontier_dev, frontier, 0));
  *frontier = 0;
  std::unique_ptr<std::atomic<uint32_t>[]> done(new std::atomic<uint32_t>[smax]);
  for (size_t k = 0; k < smax; ++k) done[k].store(0);
  std::unique_ptr<uint8_t, decltype(&free)> regen((uint8_t*)aligned_alloc(4096, nmax), &free);
  std::vector<uint32_t> ref(nmax / 1024 * 4), got(nmax / 1024 * 4);
  uint4* dout;
  OK(cudaMalloc(&dout, nmax / 1024 * 16));
  uint32_t* dsteps;
  OK(cudaMalloc(&dsteps, 64 * 4));
  uint8_t seed[32], throwaway[32] = {0}, root[64], want[64];
  fresh(seed, 32);
  if (bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, want, s_tag)) return fail("tag pass");
  std::atomic<uint64_t> next{0};
  uint32_t epoch = 0;
  int bad = 0;
  Timer ft;
  const uint64_t deadline_ns = (uint64_t)(a.deadline_us * 1e3);
  std::string j = "{\"blocks\":" + std::to_string(a.blocks) + ",\"live_generators\":" + std::to_string(a.live) +
                  ",\"live_slot\":" + std::to_string(LIVE_SLOT) + ",\"deadline_us\":" + num(a.deadline_us) +
                  ",\"runs\":[";
  bool first = true;
  for (int ctas : a.fold_ctas)
    for (size_t m : a.sizes_mib) {
      const size_t n = m << 20, chunks = n / 1024, slots = n / LIVE_SLOT;
      // arm 0 beside the tag pass, 1 alone, 2 alone with S written and the frontier set before σ, 3 fold_chunks over that
      std::vector<double> pass, wall[4], fold_ms[4], landed_mb[4];
      for (int arm = 0; arm < 4; ++arm)
        for (int r = 0; r < a.windows; ++r) {
          uint8_t ks[16];
          fresh(ks, 16);
          Gen gen(ks);
          ++epoch;
          bw_tag_pass(bw::TAG_B3, E.E, a.blocks, throwaway, E.ws, root, s_tag);
          bw_read_scrub(dev.scrub, SCRUB);
          const size_t per = (n / a.gen_threads + 63) / 64 * 64;
          parallel(a.gen_threads, cpus, [&](int t) {
            if (t * per < n) sg_flush(hm + t * per, std::min(per, n - t * per));
          });
          if (arm >= 2) {
            generate(gen, hm, n, a.gen_threads, cpus);
            __atomic_store_n(frontier, (unsigned long long)epoch << 32 | slots, __ATOMIC_RELEASE);
          }
          OK(cudaMemsetAsync(dout, 0, chunks * 16, s_fold));
          OK(cudaStreamSynchronize(s_fold));
          next.store(0, std::memory_order_release);
          // σ: from here the window's clock runs.
          const double t0 = now_s();
          OK(cudaEventRecord(ft.a, s_fold));
          if (arm == 3)
            fold_chunks<<<ctas, 1024, 0, s_fold>>>((const uint4*)hm_dev, chunks, dout);
          else
            fold_live<<<ctas, 1024, 0, s_fold>>>((const uint4*)hm_dev, chunks, dout, frontier_dev, LIVE_SLOT / 1024,
                                                 epoch, deadline_ns, dsteps);
          OK(cudaEventRecord(ft.b, s_fold));
          if (arm < 2) {
            for (auto& g : gens)
              g->post([&] {
                for (uint64_t k = next.fetch_add(1); k < slots; k = next.fetch_add(1)) {
                  sg_ctr_vaes_nt(gen.rk, k * (LIVE_SLOT / 16), hm + k * LIVE_SLOT, LIVE_SLOT / 16);
                  done[k].store(epoch, std::memory_order_release);
                }
              });
            publisher.post([&] {
              for (size_t f = 0; f < slots;) {
                if (done[f].load(std::memory_order_acquire) != epoch) continue;
                while (f < slots && done[f].load(std::memory_order_acquire) == epoch) ++f;
                __atomic_store_n(frontier, (unsigned long long)epoch << 32 | f, __ATOMIC_RELEASE);
              }
            });
          }
          if (arm == 0) bw_tag_pass(bw::TAG_B3, E.E, a.blocks, seed, E.ws, root, s_tag);
          OK(cudaStreamSynchronize(s_fold));
          const double t1 = now_s();
          for (auto& g : gens) g->wait();
          publisher.wait();
          if (arm == 0) {
            float tags_ms = 0, pass_ms = 0;
            bw_last_pass_ms(&tags_ms, &pass_ms);
            pass.push_back(pass_ms);
            bad += memcmp(root, want, 64) != 0;
          }
          wall[arm].push_back((t1 - t0) * 1e3);
          fold_ms[arm].push_back(ft.ms());
          generate(gen, regen.get(), n, a.gen_threads, cpus);
          std::fill(ref.begin(), ref.begin() + chunks * 4, 0);
          parallel(a.gen_threads, cpus, [&](int t) {
            for (size_t c = t; c < chunks; c += a.gen_threads)
              for (size_t w = 0; w < 256; ++w) ref[4 * c + w % 4] ^= ((const uint32_t*)regen.get())[256 * c + w];
          });
          OK(cudaMemcpy(got.data(), dout, chunks * 16, cudaMemcpyDeviceToHost));
          uint64_t match = 0, wrong = 0, finished = chunks;
          for (size_t c = 0; c < chunks; ++c) {
            const bool same = !memcmp(&got[4 * c], &ref[4 * c], 16);
            const bool zero = !(got[4 * c] | got[4 * c + 1] | got[4 * c + 2] | got[4 * c + 3]);
            match += same;
            wrong += !same && !zero;
          }
          if (arm < 3) {
            uint32_t st[64];
            OK(cudaMemcpy(st, dsteps, ctas * 4, cudaMemcpyDeviceToHost));
            finished = 0;
            for (int b = 0; b < ctas; ++b) finished += (uint64_t)st[b] * 128;
          }
          bad += wrong || match != finished || (!deadline_ns && match != chunks);
          landed_mb[arm].push_back(match * 1024 / 1e6);
        }
      auto gbs = [&](const std::vector<double>& ms) { return n / (median(ms) * 1e-3) / 1e9; };
      j += std::string(first ? "" : ",") + "{\"ctas\":" + std::to_string(ctas) + ",\"threads\":1024,\"bytes\":" +
           std::to_string(n) + ",\"window_ms\":" + stats(wall[0]) + ",\"device_pass_ms\":" + stats(pass) +
           ",\"fold_ms\":" + stats(fold_ms[0]) + ",\"fold_gbs\":" + num(gbs(fold_ms[0])) + ",\"alone_window_ms\":" +
           stats(wall[1]) + ",\"fold_alone_ms\":" + stats(fold_ms[1]) + ",\"fold_alone_gbs\":" +
           num(gbs(fold_ms[1])) + ",\"ready_window_ms\":" + stats(wall[2]) + ",\"fold_ready_ms\":" +
           stats(fold_ms[2]) + ",\"fold_ready_gbs\":" + num(gbs(fold_ms[2])) + ",\"fold_unflagged_ms\":" +
           stats(fold_ms[3]) + ",\"fold_unflagged_gbs\":" + num(gbs(fold_ms[3])) + ",\"landed_mb\":" +
           stats(landed_mb[0]) + ",\"alone_landed_mb\":" + stats(landed_mb[1]) + ",\"ready_landed_mb\":" +
           stats(landed_mb[2]) + "}";
      first = false;
    }
  j += "],\"bad\":" + std::to_string(bad) + "}";
  write(a, "fold_live", j);
  cudaFree(dout);
  cudaFree(dsteps);
  cudaFreeHost(hm);
  cudaFreeHost(frontier);
  return bad ? fail("a live fold or a root") : 0;
}

}  // namespace st

int main(int argc, char** argv) {
  st::Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto v = [&]() { return std::string(argv[++i]); };
    if (k == "--mode") a.mode = v();
    else if (k == "--out") a.out = v();
    else if (k == "--vectors") a.vectors = v();
    else if (k == "--expect") a.expect = v();
    else if (k == "--blocks") a.blocks = std::stoull(v());
    else if (k == "--reps") a.reps = std::stoi(v());
    else if (k == "--windows") a.windows = std::stoi(v());
    else if (k == "--samples") a.samples = std::stoi(v());
    else if (k == "--gen-threads") a.gen_threads = std::stoi(v());
    else if (k == "--piece") a.piece = std::stoull(v());
    else if (k == "--live") a.live = std::stoi(v());
    else if (k == "--first-piece") a.first_piece = std::stoull(v());
    else if (k == "--flush") a.flush = true;
    else if (k == "--path") a.path = v();
    else if (k == "--priority") a.priority = v();
    else if (k == "--suffix") a.suffix = v();
    else if (k == "--interfere-mib") a.interfere_mib = std::stoull(v());
    else if (k == "--deadline-us") a.deadline_us = std::stod(v());
    else if (k == "--fold-ctas") {
      a.fold_ctas.clear();
      std::stringstream ss(v());
      std::string x;
      while (std::getline(ss, x, ',')) a.fold_ctas.push_back(std::stoi(x));
    }
    else if (k == "--sizes-mib") {
      a.sizes_mib.clear();
      std::stringstream ss(v());
      std::string x;
      while (std::getline(ss, x, ',')) a.sizes_mib.push_back(std::stoull(x));
    } else {
      fprintf(stderr, "unknown argument %s\n", k.c_str());
      return 2;
    }
  }
  st::cpus = st::core_order();
  OK(cudaSetDevice(0));
  OK(cudaDeviceGetAttribute(&st::dev.sms, cudaDevAttrMultiProcessorCount, 0));
  OK(cudaMalloc(&st::dev.sink, 4));
  OK(cudaMalloc(&st::dev.scrub, st::SCRUB));
  OK(cudaMemset(st::dev.scrub, 0, st::SCRUB));
  if (a.mode == "gates") return st::mode_gates(a);
  if (a.mode == "link") return st::mode_link(a);
  if (a.mode == "hbm") return st::mode_hbm(a);
  if (a.mode == "gen") return st::mode_gen(a);
  if (a.mode == "pipe") return st::mode_pipe(a);
  if (a.mode == "absorb") return st::mode_absorb(a);
  if (a.mode == "window") return st::mode_window(a);
  if (a.mode == "interfere") return st::mode_interfere(a);
  if (a.mode == "fold") return st::mode_fold(a);
  if (a.mode == "foldlive") return st::mode_fold_live(a);
  fprintf(stderr, "unknown mode %s\n", a.mode.c_str());
  return 2;
}
