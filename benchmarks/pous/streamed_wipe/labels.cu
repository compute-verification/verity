// The streamed-label wipe's labels on one GPU (verity/core/protocols/pous/streamed_wipe.py):
//
//   L_i = SHA-512(LABEL_TAG ‖ K ‖ u64be i)      75 bytes, one SHA-512 compression a 64-byte label
//
// One thread computes one label at a time from a constant message whose index words it ORs in, and writes the digest
// big-endian, as hashlib's.  Modes (each prints one JSON line):
//
//   check  --key HEX --start S --count C        the labels as hex lines after a JSON line, each checked on the host
//   rate   --key HEX --labels N [--chunk C]     N labels into a device buffer of C labels, reused; and the same N with
//                                               only their XOR kept (compute alone)
//   stream --key HEX --labels N [--chunk C]     generate and copy to pinned host memory, double-buffered on two streams
//   dump   --key HEX --labels N --out PATH      N labels, from 0, written to PATH (the round-trip test's table)
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120 labels.cu -o wipelabels
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

static const char LABEL_TAG[] = "verity-pous/streamed-wipe/label/v1";  // + its NUL: 35 bytes
constexpr int TAG_BYTES = 35, KEY_BYTES = 32, MSG_BYTES = TAG_BYTES + KEY_BYTES + 8;

#define SHA512_K                                                                                                 \
  {0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, \
   0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, \
   0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, \
   0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, \
   0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL, \
   0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, \
   0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, \
   0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL, \
   0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, \
   0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, \
   0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, \
   0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, \
   0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL, \
   0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, \
   0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, \
   0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL}
#define SHA512_IV                                                                                               \
  {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, \
   0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL}

__constant__ uint64_t K_DEV[80] = SHA512_K;
__constant__ uint64_t IV_DEV[8] = SHA512_IV;
__constant__ uint64_t MSG_DEV[16];
static const uint64_t K_HOST[80] = SHA512_K;
static const uint64_t IV_HOST[8] = SHA512_IV;

__host__ __device__ __forceinline__ uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

// One SHA-512 compression of the big-endian words `w` (clobbered) into `s` (benchmarks/bandwidth/tagger.cu's).
__host__ __device__ __forceinline__ void compress(uint64_t s[8], uint64_t w[16]) {
#ifdef __CUDA_ARCH__
  const uint64_t* K = K_DEV;
#else
  const uint64_t* K = K_HOST;
#endif
  uint64_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const uint64_t w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (rotr(w15, 1) ^ rotr(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] + (rotr(w2, 19) ^ rotr(w2, 61) ^ (w2 >> 6));
      }
      const uint64_t t1 = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + K[r + t] + w[t];
      const uint64_t t2 = (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

__device__ __forceinline__ uint64_t bswap64(uint64_t v) {
  uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
  return ((uint64_t)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// The index sits at bytes 67..74: the low 40 bits of word 8 and the top 24 bits of word 9.
__device__ __forceinline__ void label_state(uint64_t i, uint64_t s[8]) {
  uint64_t w[16];
#pragma unroll
  for (int t = 0; t < 16; ++t) w[t] = MSG_DEV[t];
  w[8] |= i >> 24;
  w[9] |= (i & 0xffffffULL) << 40;
#pragma unroll
  for (int t = 0; t < 8; ++t) s[t] = IV_DEV[t];
  compress(s, w);
}

__global__ void fill_kernel(ulonglong2* __restrict__ out, uint64_t start, uint64_t n) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
    uint64_t s[8];
    label_state(start + t, s);
    ulonglong2* o = out + 4 * t;
#pragma unroll
    for (int q = 0; q < 4; ++q) o[q] = make_ulonglong2(bswap64(s[2 * q]), bswap64(s[2 * q + 1]));
  }
}

__global__ void xor_kernel(uint64_t* __restrict__ acc, uint64_t start, uint64_t n) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t x = 0;
  for (uint64_t t = tid; t < n; t += stride) {
    uint64_t s[8];
    label_state(start + t, s);
#pragma unroll
    for (int q = 0; q < 8; ++q) x ^= s[q];
  }
  acc[tid] ^= x;
}

static std::vector<uint8_t> message_bytes(const uint8_t key[32], uint64_t i) {
  std::vector<uint8_t> m(LABEL_TAG, LABEL_TAG + TAG_BYTES);
  m.insert(m.end(), key, key + KEY_BYTES);
  for (int b = 7; b >= 0; --b) m.push_back((uint8_t)(i >> (8 * b)));
  return m;
}

// The padded block of the message at index 0, as big-endian words (the device ORs the index in).
static void message_words(const uint8_t key[32], uint64_t w[16]) {
  std::vector<uint8_t> m = message_bytes(key, 0);
  m.push_back(0x80);
  m.resize(128, 0);
  const uint64_t bits = (uint64_t)MSG_BYTES * 8;
  for (int b = 0; b < 8; ++b) m[120 + b] = (uint8_t)(bits >> (8 * (7 - b)));
  for (int t = 0; t < 16; ++t) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v = (v << 8) | m[8 * t + b];
    w[t] = v;
  }
}

static void sha512_host(const std::vector<uint8_t>& msg, uint8_t out[64]) {
  uint64_t s[8];
  memcpy(s, IV_HOST, sizeof s);
  std::vector<uint8_t> m(msg);
  m.push_back(0x80);
  while (m.size() % 128 != 112) m.push_back(0);
  for (int b = 0; b < 8; ++b) m.push_back(0);
  const uint64_t bits = (uint64_t)msg.size() * 8;
  for (int b = 7; b >= 0; --b) m.push_back((uint8_t)(bits >> (8 * b)));
  for (size_t off = 0; off < m.size(); off += 128) {
    uint64_t w[16];
    for (int t = 0; t < 16; ++t) {
      uint64_t v = 0;
      for (int b = 0; b < 8; ++b) v = (v << 8) | m[off + 8 * t + b];
      w[t] = v;
    }
    compress(s, w);
  }
  for (int t = 0; t < 8; ++t)
    for (int b = 0; b < 8; ++b) out[8 * t + b] = (uint8_t)(s[t] >> (8 * (7 - b)));
}

static bool parse_hex(const char* hex, uint8_t* out, size_t n) {
  if (strlen(hex) != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) {
    unsigned v;
    if (sscanf(hex + 2 * i, "%2x", &v) != 1) return false;
    out[i] = (uint8_t)v;
  }
  return true;
}

struct Args {
  std::string mode, out;
  uint8_t key[32] = {};
  uint64_t start = 0, count = 16, labels = 1ULL << 30, chunk = 1ULL << 26;
  int threads = 256, blocks_per_sm = 8;
};

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int grid(const Args& a) {
  int dev, sms;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms * a.blocks_per_sm;
}

static void device_json() {
  int dev;
  cudaDeviceProp p;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&p, dev));
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  printf("\"device\": {\"name\": \"%s\", \"sms\": %d, \"clock_khz\": %d, \"pci_bus\": %d}", p.name, p.multiProcessorCount,
         clock_khz, p.pciBusID);
}

static int mode_check(const Args& a) {
  ulonglong2* d;
  CK(cudaMalloc(&d, a.count * 64));
  fill_kernel<<<grid(a), a.threads>>>(d, a.start, a.count);
  CK(cudaGetLastError());
  std::vector<uint8_t> h(a.count * 64);
  CK(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost));
  uint64_t bad = 0;
  for (uint64_t t = 0; t < a.count; ++t) {
    uint8_t ref[64];
    sha512_host(message_bytes(a.key, a.start + t), ref);
    bad += memcmp(ref, h.data() + 64 * t, 64) != 0;
  }
  printf("{\"mode\": \"check\", \"start\": %llu, \"count\": %llu, \"host_mismatches\": %llu, ", (unsigned long long)a.start,
         (unsigned long long)a.count, (unsigned long long)bad);
  device_json();
  printf("}\n");
  for (uint64_t t = 0; t < a.count; ++t) {
    for (int b = 0; b < 64; ++b) printf("%02x", h[64 * t + b]);
    printf("\n");
  }
  CK(cudaFree(d));
  return bad ? 3 : 0;
}

static int mode_rate(const Args& a) {
  const uint64_t chunk = a.chunk < a.labels ? a.chunk : a.labels;
  ulonglong2* d;
  CK(cudaMalloc(&d, chunk * 64));
  const int g = grid(a);
  uint64_t* acc;
  CK(cudaMalloc(&acc, (size_t)g * a.threads * 8));
  CK(cudaMemset(acc, 0, (size_t)g * a.threads * 8));
  fill_kernel<<<g, a.threads>>>(d, 0, chunk);  // warm
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (uint64_t s = 0; s < a.labels; s += chunk) fill_kernel<<<g, a.threads>>>(d, s, (a.labels - s) < chunk ? a.labels - s : chunk);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaGetLastError());
  float ms_store;
  CK(cudaEventElapsedTime(&ms_store, e0, e1));
  CK(cudaEventRecord(e0));
  for (uint64_t s = 0; s < a.labels; s += chunk) xor_kernel<<<g, a.threads>>>(acc, s, (a.labels - s) < chunk ? a.labels - s : chunk);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaGetLastError());
  float ms_xor;
  CK(cudaEventElapsedTime(&ms_xor, e0, e1));
  const double bytes = (double)a.labels * 64;
  printf("{\"mode\": \"rate\", \"labels\": %llu, \"chunk\": %llu, \"grid\": %d, \"threads\": %d, \"store_s\": %.6f, "
         "\"store_GBps\": %.3f, \"xor_s\": %.6f, \"xor_GBps\": %.3f, \"labels_per_s\": %.6e, ",
         (unsigned long long)a.labels, (unsigned long long)chunk, g, a.threads, ms_store / 1e3, bytes / (ms_store / 1e3) / 1e9,
         ms_xor / 1e3, bytes / (ms_xor / 1e3) / 1e9, a.labels / (ms_store / 1e3));
  device_json();
  printf("}\n");
  CK(cudaFree(d));
  CK(cudaFree(acc));
  return 0;
}

static int mode_stream(const Args& a) {
  const uint64_t chunk = a.chunk < a.labels ? a.chunk : a.labels;
  const int g = grid(a);
  ulonglong2* d[2];
  uint8_t* h[2];
  cudaStream_t st[2];
  for (int b = 0; b < 2; ++b) {
    CK(cudaMalloc(&d[b], chunk * 64));
    CK(cudaHostAlloc(&h[b], chunk * 64, cudaHostAllocDefault));
    CK(cudaStreamCreate(&st[b]));
  }
  fill_kernel<<<g, a.threads, 0, st[0]>>>(d[0], 0, chunk);
  CK(cudaMemcpyAsync(h[0], d[0], chunk * 64, cudaMemcpyDeviceToHost, st[0]));
  CK(cudaDeviceSynchronize());
  const double t0 = now_s();
  uint64_t n = 0;
  for (uint64_t s = 0, c = 0; s < a.labels; s += chunk, ++c) {
    const int b = c & 1;
    const uint64_t m = (a.labels - s) < chunk ? a.labels - s : chunk;
    fill_kernel<<<g, a.threads, 0, st[b]>>>(d[b], s, m);
    CK(cudaMemcpyAsync(h[b], d[b], m * 64, cudaMemcpyDeviceToHost, st[b]));
    n += m;
  }
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  const double dt = now_s() - t0;
  // The copy alone, device to pinned host, for the same bytes.
  const double t1 = now_s();
  for (uint64_t s = 0, c = 0; s < a.labels; s += chunk, ++c) {
    const int b = c & 1;
    const uint64_t m = (a.labels - s) < chunk ? a.labels - s : chunk;
    CK(cudaMemcpyAsync(h[b], d[b], m * 64, cudaMemcpyDeviceToHost, st[b]));
  }
  CK(cudaDeviceSynchronize());
  const double dc = now_s() - t1;
  printf("{\"mode\": \"stream\", \"labels\": %llu, \"chunk\": %llu, \"s\": %.6f, \"GBps\": %.3f, \"copy_only_s\": %.6f, "
         "\"copy_only_GBps\": %.3f, ",
         (unsigned long long)n, (unsigned long long)chunk, dt, n * 64.0 / dt / 1e9, dc, n * 64.0 / dc / 1e9);
  device_json();
  printf("}\n");
  for (int b = 0; b < 2; ++b) {
    CK(cudaFree(d[b]));
    CK(cudaFreeHost(h[b]));
    CK(cudaStreamDestroy(st[b]));
  }
  return 0;
}

static int mode_dump(const Args& a) {
  const uint64_t chunk = a.chunk < a.labels ? a.chunk : a.labels;
  FILE* f = fopen(a.out.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", a.out.c_str());
    return 2;
  }
  ulonglong2* d;
  uint8_t* h;
  CK(cudaMalloc(&d, chunk * 64));
  CK(cudaHostAlloc(&h, chunk * 64, cudaHostAllocDefault));
  const double t0 = now_s();
  for (uint64_t s = 0; s < a.labels; s += chunk) {
    const uint64_t m = (a.labels - s) < chunk ? a.labels - s : chunk;
    fill_kernel<<<grid(a), a.threads>>>(d, s, m);
    CK(cudaMemcpy(h, d, m * 64, cudaMemcpyDeviceToHost));
    if (fwrite(h, 64, m, f) != m) {
      fprintf(stderr, "short write to %s\n", a.out.c_str());
      return 2;
    }
  }
  fclose(f);
  printf("{\"mode\": \"dump\", \"labels\": %llu, \"s\": %.3f}\n", (unsigned long long)a.labels, now_s() - t0);
  CK(cudaFree(d));
  CK(cudaFreeHost(h));
  return 0;
}

int main(int argc, char** argv) {
  Args a;
  if (argc < 2) {
    fprintf(stderr, "usage: wipelabels check|rate|stream|dump --key HEX [--start S] [--count C] [--labels N] [--chunk C] [--out P]\n");
    return 2;
  }
  a.mode = argv[1];
  bool keyed = false;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const char* v = argv[i + 1];
    if (k == "--key") keyed = parse_hex(v, a.key, 32);
    else if (k == "--start") a.start = strtoull(v, nullptr, 10);
    else if (k == "--count") a.count = strtoull(v, nullptr, 10);
    else if (k == "--labels") a.labels = strtoull(v, nullptr, 10);
    else if (k == "--chunk") a.chunk = strtoull(v, nullptr, 10);
    else if (k == "--threads") a.threads = atoi(v);
    else if (k == "--blocks-per-sm") a.blocks_per_sm = atoi(v);
    else if (k == "--out") a.out = v;
    else {
      fprintf(stderr, "unknown flag %s\n", k.c_str());
      return 2;
    }
  }
  if (!keyed) {
    fprintf(stderr, "--key takes 64 hex digits\n");
    return 2;
  }
  uint64_t w[16];
  message_words(a.key, w);
  CK(cudaMemcpyToSymbol(MSG_DEV, w, sizeof w));
  if (a.mode == "check") return mode_check(a);
  if (a.mode == "rate") return mode_rate(a);
  if (a.mode == "stream") return mode_stream(a);
  if (a.mode == "dump") return mode_dump(a);
  fprintf(stderr, "unknown mode %s\n", a.mode.c_str());
  return 2;
}
