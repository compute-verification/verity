// The kernel-engineering timing protocol for a standalone .cu bench (header only, C++17): the 0xA5 poisoning and its
// no-write control, interleaved items with NVML's SM clock on every item, one summary line an arm, and one dump format
// (`dump.py` reads it, `compare.py` compares two). `harness/bench.py` is the same protocol for an arm of the harness;
// the constants are its.
//
//   gk::poison_gate(dev, bytes, stream, arm, accept)      the poisoned pass, and the control that writes nothing
//   gk::timed_items(arms, reps, stream, nvml)             warm-up to a steady clock, then rep by rep, arms rotating
//   gk::summary(timing, arms)                             "GK arm=... ms=... sm_mhz=... clock_gap=...% [tags]"
//   gk::Dump(dir).bytes(name, dev, n, s) / .digests(...)  DIR/NAME.bin and DIR/dump.jsonl
#pragma once
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#define GK_CK(x)                                                                                 \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));     \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

namespace gk {

constexpr unsigned char POISON = 0xA5;
constexpr uint32_t POISON_WORD = 0xA5A5A5A5u;
// harness/bench.py: a steady round's SM clock change at most, and the clock gap the summary tags
constexpr double STEADY = 0.005, CLOCK_GAP = 0.005;
constexpr double ITEM_MS = 40.0, WARMUP_MS = 10000.0, WARMUP_MAX_MS = 60000.0;
// harness/native.py's BAD_THROTTLE: sw_power_cap, hw_slowdown, sw_thermal, hw_thermal, hw_power_brake
constexpr unsigned long long BAD_THROTTLE = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;

// ---- poisoning --------------------------------------------------------------------------------------------------------

// On the stream the kernel runs on: a memset on the default stream raced a GEMM on another stream (44878194f).
inline void poison(void* dev, size_t bytes, cudaStream_t s) { GK_CK(cudaMemsetAsync(dev, POISON, bytes, s)); }

inline size_t poisoned_words(const void* host, size_t bytes) {
  size_t n = 0;
  const unsigned char* p = static_cast<const unsigned char*>(host);
  for (size_t i = 0; i + 4 <= bytes; i += 4) {
    uint32_t w;
    std::memcpy(&w, p + i, 4);
    n += w == POISON_WORD;
  }
  return n;
}

struct PoisonGate {
  bool accepted = false;          // the arm's output, written over the poison, passed `accept`
  bool control_rejected = false;  // the poison alone (nothing written) failed `accept`
  size_t left = 0;                // 4-byte words of the arm's output still 0xA5A5A5A5
  bool ok() const { return accepted && control_rejected; }
};

// Poison `dev`, run `arm(s)`, and judge the bytes with `accept(host, bytes)`; then the no-write control: poison again,
// run `control(s)` (by default nothing), and `accept` must reject it. A gate whose control passes shows nothing.
template <class Arm, class Accept>
PoisonGate poison_gate(void* dev, size_t bytes, cudaStream_t s, Arm&& arm, Accept&& accept,
                       std::function<void(cudaStream_t)> control = {}) {
  std::vector<unsigned char> h(bytes);
  auto pass = [&](auto&& run) {
    poison(dev, bytes, s);
    run(s);
    GK_CK(cudaMemcpyAsync(h.data(), dev, bytes, cudaMemcpyDeviceToHost, s));
    GK_CK(cudaStreamSynchronize(s));
  };
  PoisonGate g;
  pass(arm);
  g.left = poisoned_words(h.data(), bytes);
  g.accepted = accept(h.data(), bytes);
  pass([&](cudaStream_t st) { if (control) control(st); });
  g.control_rejected = !accept(h.data(), bytes);
  return g;
}

// ---- NVML, by dlopen, read only --------------------------------------------------------------------------------------

struct Sample {
  unsigned sm_mhz = 0;  // 0: unknown
  unsigned long long throttle = 0;
};

class Nvml {
 public:
  explicit Nvml(int cuda_device) {
    lib_ = dlopen(getenv("POUW_LIBNVML") ? getenv("POUW_LIBNVML") : "libnvidia-ml.so.1", RTLD_NOW);
    if (!lib_) return;
    auto init = reinterpret_cast<int (*)()>(dlsym(lib_, "nvmlInit_v2"));
    auto by_uuid = reinterpret_cast<int (*)(const char*, void**)>(dlsym(lib_, "nvmlDeviceGetHandleByUUID"));
    clock_ = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib_, "nvmlDeviceGetClockInfo"));
    void* r = dlsym(lib_, "nvmlDeviceGetCurrentClocksEventReasons");
    reasons_ = reinterpret_cast<int (*)(void*, unsigned long long*)>(
        r ? r : dlsym(lib_, "nvmlDeviceGetCurrentClocksThrottleReasons"));
    cudaDeviceProp p;
    GK_CK(cudaGetDeviceProperties(&p, cuda_device));
    char u[64];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
    snprintf(u, sizeof u, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2],
             b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    uuid_ = u;
    if (!init || !by_uuid || !clock_ || init() != 0 || by_uuid(u, &h_) != 0) h_ = nullptr;
  }
  bool ok() const { return h_ != nullptr; }
  const std::string& uuid() const { return uuid_; }
  Sample sample() const {
    Sample s;
    if (!h_) return s;
    unsigned v = 0;
    if (clock_(h_, 1 /* NVML_CLOCK_SM */, &v) == 0) s.sm_mhz = v;
    if (reasons_) reasons_(h_, &s.throttle);
    return s;
  }

 private:
  void* lib_ = nullptr;
  void* h_ = nullptr;
  int (*clock_)(void*, int, unsigned*) = nullptr;
  int (*reasons_)(void*, unsigned long long*) = nullptr;
  std::string uuid_;
};

// ---- interleaved items ----------------------------------------------------------------------------------------------

struct Item;

struct Arm {
  std::string name;
  std::function<void(cudaStream_t, int)> enqueue;  // enqueue n calls on the stream
  std::function<void(const Item&)> after = {};     // called after each timed item (not the warm-up): read its results
  int calls = 1;                                   // calls an item; timed_items sizes it to ITEM_MS
};

struct Item {
  int arm = 0, rep = 0, calls = 0;
  double ms = 0;
  Sample before, during, after;  // NVML before the item, while the GPU runs it, and after it
  unsigned sm_mhz() const { return during.sm_mhz ? during.sm_mhz : after.sm_mhz; }
};

struct Timing {
  std::vector<Item> items;
  std::vector<double> warmup_mhz;  // each warm-up round's median SM clock
  double warmup_ms = 0;
  bool steady = false;
};

inline double median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

class Events {
 public:
  Events() { GK_CK(cudaEventCreate(&e0_)); GK_CK(cudaEventCreate(&e1_)); }
  ~Events() { cudaEventDestroy(e0_); cudaEventDestroy(e1_); }
  Item time(const Arm& a, cudaStream_t s, const Nvml& nv) {
    Item it;
    it.calls = a.calls;
    it.before = nv.sample();
    GK_CK(cudaEventRecord(e0_, s));
    a.enqueue(s, a.calls);
    GK_CK(cudaEventRecord(e1_, s));
    it.during = nv.sample();
    GK_CK(cudaEventSynchronize(e1_));
    it.after = nv.sample();
    float ms = 0;
    GK_CK(cudaEventElapsedTime(&ms, e0_, e1_));
    it.ms = ms;
    return it;
  }

 private:
  cudaEvent_t e0_, e1_;
};

// Size every arm's item to `item_ms`; warm up (rounds of every arm) until at least `warmup_ms` of device time and a
// round's median SM clock within STEADY of the last round's, or `warmup_max_ms`; then `reps` reps, every arm once a rep,
// the order rotating so no arm always follows the same one.
inline Timing timed_items(std::vector<Arm>& arms, int reps, cudaStream_t s, const Nvml& nv, double item_ms = ITEM_MS,
                          double warmup_ms = WARMUP_MS, double warmup_max_ms = WARMUP_MAX_MS) {
  Events ev;
  Timing t;
  for (Arm& a : arms) {
    a.calls = 1;
    for (int round = 0; round < 3; round++) {
      const Item it = ev.time(a, s, nv);
      const double per = std::max(it.ms / it.calls, 1e-4);
      a.calls = std::max(1, (int)std::ceil(item_ms / per));
    }
  }
  while (t.warmup_ms < warmup_max_ms) {
    std::vector<double> mhz;
    for (const Arm& a : arms) {
      const Item it = ev.time(a, s, nv);
      t.warmup_ms += it.ms;
      if (it.sm_mhz()) mhz.push_back(it.sm_mhz());
    }
    t.warmup_mhz.push_back(mhz.empty() ? NAN : median(mhz));
    const size_t n = t.warmup_mhz.size();
    t.steady = n >= 2 && !std::isnan(t.warmup_mhz[n - 1]) && !std::isnan(t.warmup_mhz[n - 2]) &&
               std::fabs(t.warmup_mhz[n - 1] - t.warmup_mhz[n - 2]) <= STEADY * t.warmup_mhz[n - 2];
    if (t.warmup_ms >= warmup_ms && t.steady) break;
  }
  for (int r = 0; r < reps; r++)
    for (size_t k = 0; k < arms.size(); k++) {
      const int a = (int)((k + r) % arms.size());
      Item it = ev.time(arms[a], s, nv);
      it.arm = a;
      it.rep = r;
      t.items.push_back(it);
      if (arms[a].after) arms[a].after(it);
    }
  return t;
}

// One line an arm: its median per-call ms and SM clock, the clock gap to arm 0 (the baseline), and the tags a reader
// must see: clock-gap (over CLOCK_GAP), throttled (a BAD_THROTTLE reason on any item), unsteady (the warm-up never
// settled), no-clock (NVML gave no clock).
inline void summary(const Timing& t, const std::vector<Arm>& arms, FILE* f = stdout) {
  std::vector<double> base;
  for (const Item& it : t.items)
    if (it.arm == 0 && it.sm_mhz()) base.push_back(it.sm_mhz());
  const double base_mhz = median(base);
  for (size_t a = 0; a < arms.size(); a++) {
    std::vector<double> ms, mhz;
    bool throttled = false;
    for (const Item& it : t.items)
      if (it.arm == (int)a) {
        ms.push_back(it.ms / it.calls);
        if (it.sm_mhz()) mhz.push_back(it.sm_mhz());
        throttled |= ((it.before.throttle | it.during.throttle | it.after.throttle) & BAD_THROTTLE) != 0;
      }
    const double m = median(mhz), gap = std::fabs(m - base_mhz) / base_mhz;
    fprintf(f, "GK arm=%s reps=%zu ms=%.6f sm_mhz=%.0f clock_gap=%.2f%%%s%s%s%s\n", arms[a].name.c_str(), ms.size(),
            median(ms), m, 100 * gap, gap > CLOCK_GAP ? " [clock-gap]" : "", throttled ? " [throttled]" : "",
            t.steady ? "" : " [unsteady]", std::isnan(m) ? " [no-clock]" : "");
  }
}

// Every item as a JSON line, for dump.py's reader: {"gk": "item", "arm", "rep", "calls", "ms", "sm_mhz": {...}, ...}.
inline void items_jsonl(const Timing& t, const std::vector<Arm>& arms, FILE* f) {
  for (const Item& it : t.items)
    fprintf(f,
            "{\"gk\": \"item\", \"arm\": \"%s\", \"rep\": %d, \"calls\": %d, \"ms\": %.6f, \"sm_mhz\": {\"before\": %u, "
            "\"during\": %u, \"after\": %u}, \"throttle\": \"0x%llx\"}\n",
            arms[it.arm].name.c_str(), it.rep, it.calls, it.ms, it.before.sm_mhz, it.during.sm_mhz, it.after.sm_mhz,
            it.before.throttle | it.during.throttle | it.after.throttle);
}

// ---- the dump format ------------------------------------------------------------------------------------------------

// DIR/NAME.bin holds a buffer's bytes; DIR/dump.jsonl a line a buffer: {"name", "file", "bytes", "layout"}. Layout
// "bytes" is the buffer as the device holds it. "sha512-be" is digests the device holds as 8 native uint64 words, written
// as SHA-512's big-endian output bytes: EGEMM dumped the native words, and no reference matched (63f78c35a).
class Dump {
 public:
  explicit Dump(std::string dir) : dir_(std::move(dir)) { std::filesystem::create_directories(dir_); }

  void bytes(const std::string& name, const void* dev, size_t n, cudaStream_t s) {
    write(name, fetch(dev, n, s), "bytes");
  }
  void digests(const std::string& name, const uint64_t* dev, size_t count, cudaStream_t s) {
    std::vector<unsigned char> h = fetch(dev, count * 64, s), out(h.size());
    for (size_t w = 0; w < count * 8; w++) {
      uint64_t v;
      std::memcpy(&v, h.data() + 8 * w, 8);
      for (int b = 0; b < 8; b++) out[8 * w + b] = (unsigned char)(v >> (56 - 8 * b));
    }
    write(name, out, "sha512-be");
  }

 private:
  static std::vector<unsigned char> fetch(const void* dev, size_t n, cudaStream_t s) {
    std::vector<unsigned char> h(n);
    GK_CK(cudaMemcpyAsync(h.data(), dev, n, cudaMemcpyDeviceToHost, s));
    GK_CK(cudaStreamSynchronize(s));
    return h;
  }
  void write(const std::string& name, const std::vector<unsigned char>& h, const char* layout) {
    const std::string file = name + ".bin";
    FILE* f = fopen((dir_ + "/" + file).c_str(), "wb");
    if (!f || fwrite(h.data(), 1, h.size(), f) != h.size()) {
      fprintf(stderr, "ERR dump %s/%s\n", dir_.c_str(), file.c_str());
      exit(2);
    }
    fclose(f);
    FILE* j = fopen((dir_ + "/dump.jsonl").c_str(), "a");
    fprintf(j, "{\"name\": \"%s\", \"file\": \"%s\", \"bytes\": %zu, \"layout\": \"%s\"}\n", name.c_str(), file.c_str(),
            h.size(), layout);
    fclose(j);
  }
  std::string dir_;
};

}  // namespace gk
